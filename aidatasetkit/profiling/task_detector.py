"""What kind of prediction problem a target poses.

This is schema analysis, not machine learning. No model is fitted here and none
is needed: the answer follows from the target's dtype and the shape of its value
distribution.

The detector is deliberately reluctant. An integer target with a handful of
distinct values in a small sample is genuinely ambiguous -- it could be classes,
ordinal grades, or a count to regress on -- and guessing wrong silently would
send every later stage down the wrong path. Those cases raise
:class:`~aidatasetkit.core.exceptions.AmbiguousTaskError` and ask for a hint
instead.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from aidatasetkit.core.config import KitConfig
from aidatasetkit.core.counting import value_counts
from aidatasetkit.core.exceptions import (
    AmbiguousTaskError,
    EmptyDataError,
    UnsupportedTaskError,
    ValidationError,
)
from aidatasetkit.core.schema import numeric_target_values, target_holds_quantities
from aidatasetkit.core.types import (
    SUPERVISED_TASKS,
    NumericSummary,
    TargetProfile,
    TaskType,
)
from aidatasetkit.statistics.engine import StatisticsEngine

__all__ = ["TaskDetector", "UNRESOLVED"]


class _Unresolved:
    """Sentinel distinguishing "no positive label given" from "the label is None"."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "UNRESOLVED"


#: Marker meaning the caller did not specify a positive label.
UNRESOLVED = _Unresolved()


class TaskDetector:
    """Infers the task family and summarises the target.

    Args:
        config: Supplies ``task_detection_max_classes`` and
            ``task_detection_unique_ratio``, the two thresholds separating a
            small set of classes from a continuous target.

    Example:
        >>> import pandas as pd
        >>> profile = TaskDetector().detect(pd.Series(["yes", "no", "yes"]))
        >>> profile.task_type.value, profile.is_binary
        ('classification', True)
    """

    def __init__(self, config: KitConfig | None = None) -> None:
        self._config = config if config is not None else KitConfig()

    @property
    def config(self) -> KitConfig:
        """The configuration these decisions are taken against."""
        return self._config

    def detect(
        self,
        y: Any,
        *,
        hint: TaskType | str | None = None,
        target_name: Hashable | None = None,
        positive_label: Any = UNRESOLVED,
    ) -> TargetProfile:
        """Describe the target and decide what kind of problem it poses.

        Args:
            y: The target values: a :class:`pandas.Series`, list, or array.
            hint: ``"classification"`` or ``"regression"`` to settle an ambiguous
                target. Supplying a hint that contradicts the data is an error,
                not an override.
            target_name: Name recorded on the profile. Taken from the series name
                when not given.
            positive_label: Which class counts as positive in binary
                classification. When omitted, it is resolved only for the two
                conventional encodings and otherwise left unresolved.

        Returns:
            A :class:`~aidatasetkit.core.types.TargetProfile`.

        Raises:
            ValidationError: If ``y`` is unusable or the hint is not a task name.
            EmptyDataError: If no non-missing values remain.
            AmbiguousTaskError: If the target could reasonably be either classes
                or a continuous quantity and no hint was given.
            UnsupportedTaskError: If the hint names a task family this version
                does not implement, or contradicts the data.
        """
        series = self._as_series(y)
        name = target_name if target_name is not None else series.name
        missing_count = int(series.isna().sum())
        present = series.dropna()

        if present.empty:
            raise EmptyDataError(
                "The target contains no non-missing values, so no task can be inferred."
            )

        requested = self._coerce_hint(hint)
        task, note = self._resolve_task(present, requested)

        if task is TaskType.CLASSIFICATION:
            return self._classification_profile(
                present, name, missing_count, note, positive_label
            )
        return self._regression_profile(present, name, missing_count, note)

    # ------------------------------------------------------------------ #
    # Input
    # ------------------------------------------------------------------ #

    @staticmethod
    def _as_series(y: Any) -> pd.Series:
        """Materialise the target as a one-dimensional series."""
        if isinstance(y, pd.Series):
            return y
        if isinstance(y, pd.DataFrame):
            raise ValidationError(
                "The target must be one-dimensional; a DataFrame was given. "
                "Select a single column first."
            )
        if isinstance(y, (str, bytes, dict, set, frozenset)):
            raise ValidationError(
                f"The target must be a sequence of values, got {type(y).__name__}."
            )
        if isinstance(y, pd.Index):
            return pd.Series(y)

        array = np.asarray(y)
        if array.ndim != 1:
            raise ValidationError(
                f"The target must be one-dimensional, got {array.ndim} dimensions."
            )
        return pd.Series(array)

    @staticmethod
    def _coerce_hint(hint: TaskType | str | None) -> TaskType | None:
        """Validate a caller-supplied task hint."""
        if hint is None:
            return None
        task = TaskType.coerce(hint)
        if task not in SUPERVISED_TASKS:
            raise UnsupportedTaskError(
                f"Task {task.value!r} has no target to detect. This version supports "
                f"{', '.join(sorted(member.value for member in SUPERVISED_TASKS))}."
            )
        return task

    # ------------------------------------------------------------------ #
    # Task resolution
    # ------------------------------------------------------------------ #

    def _resolve_task(
        self, present: pd.Series, requested: TaskType | None
    ) -> tuple[TaskType, str]:
        """Decide the task family and explain how the decision was reached.

        A hint is the caller saying what they intend, and it is checked against
        the data in *both* directions. Regression has always been refused on a
        target holding labels. Classification was not refused on a target holding
        quantities, and the consequence was not a bad model but a silent one: the
        profile said classification, the target encoder trusted the profile, and a
        column of measurements became class indices with nothing raised anywhere.
        """
        unique_count = int(present.nunique())
        # Values, not dtype. A column of floats stored under object dtype holds
        # nothing but quantities, and asking pandas for the dtype answers that it
        # is not numeric at all.
        is_numeric = numeric_target_values(present) is not None

        if requested is TaskType.REGRESSION:
            if not is_numeric:
                raise UnsupportedTaskError(
                    f"Regression was requested but the target has dtype "
                    f"{present.dtype!r}, which holds labels rather than quantities."
                )
            if unique_count < 2:
                # The classification branch below and the inference path both
                # refuse this; the regression branch did not, so a constant target
                # became a valid run in which every model scored a perfect
                # RMSE of 0.0 and the leaderboard looked excellent.
                raise UnsupportedTaskError(
                    "Regression was requested but the target holds a single "
                    "distinct value, so there is nothing to predict. Every model "
                    "would score perfectly by returning that value."
                )
            return TaskType.REGRESSION, "regression requested by the caller"

        if requested is TaskType.CLASSIFICATION:
            if unique_count < 2:
                raise UnsupportedTaskError(
                    "Classification was requested but the target has only "
                    f"{unique_count} distinct value(s); at least two are needed."
                )
            if target_holds_quantities(present, self._config):
                raise UnsupportedTaskError(self._quantity_refusal(present, unique_count))
            return TaskType.CLASSIFICATION, "classification requested by the caller"

        return self._infer_task(present, unique_count, is_numeric)

    def _quantity_refusal(self, present: pd.Series, unique_count: int) -> str:
        """Explain why a requested classification is not being carried out.

        Four things a reader needs: what they asked for, what the data looks
        like, what encoding it would cost them, and how to proceed. The task is
        never quietly reinterpreted -- answering a different question than the
        one asked is how a caller ends up trusting a result they did not request.
        """
        if self._all_integral(present):
            evidence = (
                f"{unique_count} distinct whole numbers, above the "
                f"{self._config.task_detection_max_classes}-class limit"
            )
        else:
            evidence = f"{unique_count} distinct values, not all of them whole numbers"
        return (
            f"Classification was requested, but this target holds quantities: "
            f"{evidence}. Encoding them as classes would replace every "
            "measurement with an arbitrary index and destroy the thing being "
            'predicted. Pass hint="regression" if that is what this column is, '
            "or name the label column if you meant a different one. The task you "
            "asked for is not reinterpreted for you."
        )

    def _all_integral(self, present: pd.Series) -> bool:
        """Whether every present value is a whole number."""
        values = numeric_target_values(present)
        if values is None:
            return False
        return bool(np.all(np.isfinite(values)) and np.all(values == np.floor(values)))

    def _infer_task(
        self, present: pd.Series, unique_count: int, is_numeric: bool
    ) -> tuple[TaskType, str]:
        """Infer the task family from dtype and value distribution."""
        if pdt.is_bool_dtype(present):
            return TaskType.CLASSIFICATION, "boolean target"

        if not is_numeric:
            if unique_count < 2:
                raise UnsupportedTaskError(
                    "The target holds a single distinct value, so there is nothing "
                    "to predict."
                )
            # All-distinct labels only suggest an identifier once there are enough
            # of them. Two rows holding "churn" and "stay" are all-distinct too,
            # and that is an ordinary binary target.
            if (
                unique_count == len(present)
                and unique_count > self._config.task_detection_max_classes
            ):
                raise AmbiguousTaskError(
                    f"Every one of the {len(present)} target values is distinct, so "
                    "the column looks like an identifier or free text rather than a "
                    "set of classes. Supply an explicit task if this is intended."
                )
            return TaskType.CLASSIFICATION, f"non-numeric target with {unique_count} classes"

        if unique_count < 2:
            raise UnsupportedTaskError(
                "The target holds a single distinct value, so there is nothing to "
                "predict."
            )
        if unique_count == 2:
            return TaskType.CLASSIFICATION, "numeric target with exactly two values"

        # Through the shared reader, so that an object-dtype column of numbers
        # reaches this branch with the same values a float64 column would.
        # ``is_numeric`` was decided by the same function, so this cannot be None.
        values = numeric_target_values(present)
        integral = bool(np.all(np.isfinite(values)) and np.all(values == np.floor(values)))
        if not integral:
            return TaskType.REGRESSION, "numeric target with non-integral values"

        if unique_count > self._config.task_detection_max_classes:
            return (
                TaskType.REGRESSION,
                f"integer target with {unique_count} distinct values, above the "
                f"{self._config.task_detection_max_classes}-class limit",
            )

        unique_ratio = unique_count / len(present)
        if unique_ratio <= self._config.task_detection_unique_ratio:
            return (
                TaskType.CLASSIFICATION,
                f"integer target with {unique_count} distinct values repeating across "
                f"{len(present)} rows",
            )

        raise AmbiguousTaskError(
            f"The target is an integer column with {unique_count} distinct values "
            f"across {len(present)} rows ({unique_ratio:.1%} distinct). That could be "
            f"{unique_count} classes, an ordinal grade, or a quantity to regress on, "
            'and guessing would be wrong too often. Pass hint="classification" or '
            'hint="regression".'
        )

    # ------------------------------------------------------------------ #
    # Profiles
    # ------------------------------------------------------------------ #

    def _classification_profile(
        self,
        present: pd.Series,
        name: Hashable | None,
        missing_count: int,
        note: str,
        positive_label: Any,
    ) -> TargetProfile:
        """Summarise a classification target."""
        counted = value_counts(present)
        class_counts = {_plain(label): int(count) for label, count in counted.items()}
        total = sum(class_counts.values())
        class_ratios = {label: count / total for label, count in class_counts.items()}
        classes = tuple(sorted(class_counts, key=_sort_key))
        n_classes = len(classes)
        is_binary = n_classes == 2

        majority = max(class_counts.values())
        minority = min(class_counts.values())

        resolved, label, label_note = self._resolve_positive_label(
            classes, is_binary, positive_label
        )

        return TargetProfile(
            task_type=TaskType.CLASSIFICATION,
            target_name=None if name is None else str(name),
            sample_count=int(len(present)),
            missing_count=missing_count,
            unique_count=n_classes,
            n_classes=n_classes,
            is_binary=is_binary,
            classes=classes,
            class_counts=class_counts,
            class_ratios=class_ratios,
            positive_label=label,
            positive_label_resolved=resolved,
            imbalance_ratio=float(majority) / float(minority) if minority else None,
            numeric=None,
            detection_note=f"{note}; {label_note}",
        )

    def _resolve_positive_label(
        self, classes: tuple[Any, ...], is_binary: bool, positive_label: Any
    ) -> tuple[bool, Any, str]:
        """Choose which class counts as positive, or decline to choose.

        Only two encodings carry a conventional meaning: ``0``/``1`` and
        ``False``/``True``. For anything else -- ``"churn"``/``"stay"``, say --
        there is no way to know which side is the event of interest, and inventing
        one would silently invert precision, recall, and ROC-AUC later.
        """
        if not isinstance(positive_label, _Unresolved):
            if positive_label not in classes:
                raise ValidationError(
                    f"positive_label={positive_label!r} is not among the observed "
                    f"classes {list(classes)}."
                )
            return True, positive_label, "positive label supplied by the caller"

        if not is_binary:
            return False, None, "no positive label: the target is not binary"

        # Booleans are tested first and by type, because ``False == 0`` and
        # ``True == 1`` in Python: a plain set comparison against {0, 1} matches a
        # boolean target too and would report the positive label as 1.
        if all(isinstance(label, bool) for label in classes):
            if set(classes) == {False, True}:
                return True, True, "positive label True by the False/True convention"
        elif set(classes) == {0, 1}:
            return True, 1, "positive label 1 by the 0/1 convention"

        return (
            False,
            None,
            "positive label unresolved: the labels "
            f"{list(classes)} carry no conventional ordering, so it must be supplied",
        )

    @staticmethod
    def _regression_profile(
        present: pd.Series,
        name: Hashable | None,
        missing_count: int,
        note: str,
    ) -> TargetProfile:
        """Summarise a regression target, leaving class fields empty."""
        engine = StatisticsEngine(present, nan_policy="omit", allow_inf=True)
        quartiles = engine.quartiles()
        summary = NumericSummary(
            minimum=engine.min(),
            maximum=engine.max(),
            mean=engine.mean(),
            median=quartiles.q2,
            std=engine.std() if engine.count > 1 else None,
            q25=quartiles.q1,
            q75=quartiles.q3,
            iqr=quartiles.q3 - quartiles.q1,
        )

        return TargetProfile(
            task_type=TaskType.REGRESSION,
            target_name=None if name is None else str(name),
            sample_count=int(len(present)),
            missing_count=missing_count,
            unique_count=int(present.nunique()),
            n_classes=None,
            is_binary=False,
            classes=None,
            class_counts=None,
            class_ratios=None,
            positive_label=None,
            positive_label_resolved=False,
            imbalance_ratio=None,
            numeric=summary,
            detection_note=note,
        )


def _plain(label: Any) -> Any:
    """Convert a NumPy scalar label to its Python equivalent."""
    return label.item() if isinstance(label, np.generic) else label


def _sort_key(label: Any) -> tuple[int, str]:
    """Order labels deterministically even when their types are not comparable."""
    if isinstance(label, bool):
        return (0, str(int(label)))
    if isinstance(label, (int, float)):
        try:
            return (0, f"{float(label):020.6f}")
        except OverflowError:
            # A Python int beyond float64's range. Its exact digits, in the same
            # layout the float branch produces for an integer (``5`` and ``5.0``
            # both give ``0000000000005.000000``), keep the order deterministic
            # without the conversion that cannot be done.
            return (0, f"{label:013d}.000000")
    return (1, str(label))
