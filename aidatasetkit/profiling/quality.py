"""Runs the quality checks and collects their findings.

:class:`DataQualityInspector` is deliberately thin. It builds the shared context,
calls each check in turn, and gathers the results. All the judgement lives in the
individual functions in :mod:`aidatasetkit.profiling.checks`, which is what keeps
this class from growing into a conditional the length of the file.

Nothing here modifies the data. The inspector reports; the analyst decides.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Sequence

import pandas as pd

from aidatasetkit.core.config import KitConfig
from aidatasetkit.core.exceptions import SchemaError, ValidationError
from aidatasetkit.core.types import DatasetProfile, QualityReport, TargetProfile
from aidatasetkit.profiling.checks import DEFAULT_CHECKS, Check
from aidatasetkit.profiling.context import QualityContext
from aidatasetkit.profiling.profiler import DataProfiler
from aidatasetkit.profiling.task_detector import TaskDetector

__all__ = ["DataQualityInspector"]

_logger = logging.getLogger(__name__)


class DataQualityInspector:
    """Applies a collection of independent checks to a dataframe.

    Args:
        config: Every threshold the checks compare against.
        checks: The checks to run. Defaults to
            :data:`~aidatasetkit.profiling.checks.DEFAULT_CHECKS`. Supplying a
            different collection is how the set is narrowed or extended; no check
            reads a flag to decide whether it should run.
        profiler: Used when the caller does not already hold a profile.
        task_detector: Used when a target is named but no target profile is given.

    Example:
        >>> import pandas as pd
        >>> report = DataQualityInspector().inspect(pd.DataFrame({"a": [1, 1, 1]}))
        >>> report.by_code("constant_column")[0].column
        'a'
    """

    def __init__(
        self,
        config: KitConfig | None = None,
        *,
        checks: Sequence[Check] | None = None,
        profiler: DataProfiler | None = None,
        task_detector: TaskDetector | None = None,
    ) -> None:
        self._config = config if config is not None else KitConfig()
        self._checks: tuple[Check, ...] = tuple(
            DEFAULT_CHECKS if checks is None else checks
        )
        self._profiler = profiler if profiler is not None else DataProfiler(self._config)
        self._task_detector = (
            task_detector if task_detector is not None else TaskDetector(self._config)
        )

    @property
    def config(self) -> KitConfig:
        """The configuration the checks compare against."""
        return self._config

    @property
    def checks(self) -> tuple[Check, ...]:
        """The checks this inspector runs, in reporting order."""
        return self._checks

    def inspect(
        self,
        frame: pd.DataFrame,
        *,
        target: Hashable | None = None,
        id_column: Hashable | None = None,
        profile: DatasetProfile | None = None,
        target_profile: TargetProfile | None = None,
    ) -> QualityReport:
        """Run every check and collect the findings.

        Args:
            frame: The dataframe to inspect. It is read, never modified.
            target: Name of the target column, enabling the imbalance and leakage
                checks.
            id_column: Name of a column the caller already treats as an
                identifier, so it is not reported as a surprise.
            profile: A profile of the same frame, reused when available instead of
                measuring twice.
            target_profile: What is already known about the target. Detected from
                the frame when a target is named and this is omitted.

        Returns:
            A :class:`~aidatasetkit.core.types.QualityReport`.

        Raises:
            ValidationError: If ``frame`` is not a :class:`pandas.DataFrame`.
            SchemaError: If a named column is not present in the frame.
        """
        if not isinstance(frame, pd.DataFrame):
            raise ValidationError(
                f"A pandas DataFrame is required, got {type(frame).__name__}."
            )
        for label, column in (("target", target), ("id_column", id_column)):
            if column is not None and column not in frame.columns:
                raise SchemaError(
                    f"The {label} {column!r} is not a column of this frame."
                )

        context = QualityContext(
            profile=profile if profile is not None else self._profiler.profile(frame),
            config=self._config,
            target=target,
            target_profile=self._resolve_target_profile(frame, target, target_profile),
            id_column=id_column,
        )

        issues = []
        for check in self._checks:
            issues.extend(check(frame, context))

        return QualityReport(issues=tuple(issues))

    def _resolve_target_profile(
        self,
        frame: pd.DataFrame,
        target: Hashable | None,
        target_profile: TargetProfile | None,
    ) -> TargetProfile | None:
        """Reuse a supplied target profile, or detect one when a target is named.

        A target whose task cannot be inferred is not an inspection failure. The
        checks that need it are skipped, the reason is logged, and every other
        check still runs.
        """
        if target_profile is not None or target is None:
            return target_profile
        try:
            return self._task_detector.detect(frame[target], target_name=target)
        except Exception as error:
            _logger.info(
                "Target %r could not be profiled (%s); target-dependent checks "
                "will be skipped.",
                target,
                error,
            )
            return None
