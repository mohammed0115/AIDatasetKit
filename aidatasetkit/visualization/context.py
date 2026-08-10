"""What the advisor is allowed to know.

The context is a read-only view over work that has *already been done*. The
profiler measured the columns, the quality inspector judged them, the task
detector read the target; none of that is repeated here. This package contains no
dtype detection, no missing-value scan, no cardinality count, no identifier
heuristic, and no outlier rule, because all five already exist one layer down and
a second copy would eventually disagree with the first.

The frame is carried because rendering needs values, and because a measured
correlation cannot be read off a profile. Everything else is looked up.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from functools import cached_property

import pandas as pd

from aidatasetkit.core.exceptions import SchemaError
from aidatasetkit.core.types import (
    ColumnProfile,
    DatasetProfile,
    QualityIssue,
    QualityReport,
    TargetProfile,
)
from aidatasetkit.visualization.config import VisualizationConfig

__all__ = ["VisualizationContext"]


@dataclass(frozen=True)
class VisualizationContext:
    """Everything a recommendation rule may consult.

    Attributes:
        frame: The caller's dataframe. Read, never modified.
        profile: Structural measurements from the profiler.
        quality: Findings from the quality inspector. An empty report is valid
            and simply means nothing was flagged.
        target_profile: What the task detector concluded, when a target exists.
        config: Budgets and scoring policy.
    """

    frame: pd.DataFrame
    profile: DatasetProfile
    quality: QualityReport = field(default_factory=QualityReport)
    target_profile: TargetProfile | None = None
    target_column: Hashable | None = None
    config: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.frame, pd.DataFrame):
            raise SchemaError(
                f"A pandas DataFrame is required, got {type(self.frame).__name__}."
            )

    # ------------------------------------------------------------------ #
    # Target
    # ------------------------------------------------------------------ #

    @property
    def target(self) -> Hashable | None:
        """The target column's label, when one was resolved.

        The label is carried verbatim rather than recovered from
        ``TargetProfile.target_name``, which the task detector stores as a
        *string*. A frame built from a numpy array has integer labels, and
        matching ``"2"`` against them fails -- which would silently disable every
        target-aware rule while the caller believed a target was set.
        """
        if self.target_column is not None:
            return self.target_column if self.target_column in self.frame.columns else None
        if self.target_profile is None:
            return None
        name = self.target_profile.target_name
        if name is None:
            return None
        return name if name in self.frame.columns else None

    @property
    def has_target(self) -> bool:
        """Whether target-aware rules can run.

        Both halves are required: the column must be in the frame *and* the task
        detector must have reached a conclusion about it. A target whose task
        could not be inferred still identifies a column -- so it is not charted as
        an ordinary feature -- but there is no task to tailor anything to.
        """
        return self.target is not None and self.target_profile is not None

    def is_target(self, column: Hashable) -> bool:
        """Whether ``column`` is the declared target, profiled or not."""
        return self.target is not None and column == self.target

    # ------------------------------------------------------------------ #
    # Columns
    # ------------------------------------------------------------------ #

    @property
    def column_profiles(self) -> tuple[ColumnProfile, ...]:
        """Every column profile, in frame order."""
        return self.profile.column_profiles

    @property
    def feature_profiles(self) -> tuple[ColumnProfile, ...]:
        """Column profiles excluding the target."""
        return tuple(
            profile for profile in self.column_profiles if not self.is_target(profile.name)
        )

    @cached_property
    def _by_name(self) -> dict[Hashable, ColumnProfile]:
        """Column profiles by label.

        ``DatasetProfile.column`` scans its tuple, which is fine for one lookup
        and quadratic when the scoring engine asks once per column of every
        chart. The index is built once per context instead.
        """
        return {profile.name: profile for profile in self.profile.column_profiles}

    def column(self, name: Hashable) -> ColumnProfile:
        """Return one column's profile.

        Raises:
            SchemaError: If the column was not profiled.
        """
        try:
            return self._by_name[name]
        except (KeyError, TypeError):
            raise SchemaError(
                f"Column {name!r} is not present in the profiled frame."
            ) from None

    def profile_of(self, name: Hashable) -> ColumnProfile | None:
        """Return one column's profile, or ``None`` when it was not profiled."""
        try:
            return self._by_name.get(name)
        except TypeError:
            return None

    # ------------------------------------------------------------------ #
    # Quality lookups
    # ------------------------------------------------------------------ #

    def issues_for(self, column: Hashable) -> tuple[QualityIssue, ...]:
        """Return the quality findings attached to one column."""
        return self.quality.by_column(column)

    def has_issue(self, column: Hashable, code: str) -> bool:
        """Whether one column carries a finding of a given code."""
        return any(issue.code == code for issue in self.quality.by_column(column))

    def issue(self, column: Hashable, code: str) -> QualityIssue | None:
        """Return one column's finding of a given code, if present."""
        for issue in self.quality.by_column(column):
            if issue.code == code:
                return issue
        return None

    def dataset_issues(self, code: str) -> tuple[QualityIssue, ...]:
        """Return dataset-level findings of a given code."""
        return tuple(
            issue for issue in self.quality.by_code(code) if issue.column is None
        )
