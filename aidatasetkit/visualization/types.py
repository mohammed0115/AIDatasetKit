"""Renderer-neutral vocabulary for visualization.

Nothing in this module knows that matplotlib exists. A :class:`ChartSpec` says
*what* should be drawn and *why*; it holds no figure, no axes, and no array. That
separation is what lets the planning layer run, be tested, and be serialised in an
environment where no plotting library is installed at all.

Two records are deliberately kept apart:

:class:`ChartSpec` and :class:`VisualizationPlan`
    Decisions. Serialisable, comparable, free of data.

:class:`PreparedChart`
    Render input. Holds the actual numbers, so it is *not* part of plan
    serialisation.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from aidatasetkit.core.types import TaskType, jsonable

__all__ = [
    "ChartType",
    "VisualizationReason",
    "VisualizationWarning",
    "ScoreContribution",
    "ChartScore",
    "ChartSpec",
    "SuppressedCandidate",
    "PlanMetadata",
    "VisualizationPlan",
    "SamplingInfo",
    "PreparedChart",
]


class ChartType(StrEnum):
    """The chart vocabulary this version supports.

    Kept small on purpose. Each entry answers a question an analyst actually asks
    of a tabular dataset before modelling; anything that would need a second
    library, an interaction model, or a screen of its own is out of scope.
    """

    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    BAR = "bar"
    SCATTER = "scatter"
    CORRELATION_HEATMAP = "correlation_heatmap"
    MISSING_VALUES = "missing_values"
    TARGET_DISTRIBUTION = "target_distribution"
    GROUPED_BOX = "grouped_box"
    GROUPED_BAR = "grouped_bar"


#: Chart types that plot one feature against the target.
TARGET_RELATIONSHIP_CHARTS: frozenset[ChartType] = frozenset(
    {ChartType.GROUPED_BOX, ChartType.GROUPED_BAR, ChartType.SCATTER}
)


@dataclass(frozen=True, slots=True)
class VisualizationReason:
    """Why a chart was recommended, or why a candidate was dropped.

    ``evidence`` carries the measurements the decision rested on, so a
    recommendation can be argued with rather than merely accepted.
    ``requires_review`` marks a reason that rests on a heuristic finding from the
    quality inspector and therefore states a question, not a conclusion.
    """

    code: str
    message: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    requires_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the reason."""
        return {
            "code": self.code,
            "message": self.message,
            "evidence": {str(k): jsonable(v) for k, v in self.evidence.items()},
            "requires_review": self.requires_review,
        }


@dataclass(frozen=True, slots=True)
class VisualizationWarning:
    """Something the viewer should know before reading the chart."""

    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable mapping of the warning."""
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class ScoreContribution:
    """One named term of a chart's score.

    Every term is separately visible so that a ranking can be explained rather
    than asserted. There is no opaque importance number anywhere in this package.
    """

    factor: str
    value: float
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the contribution."""
        return {"factor": self.factor, "value": self.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ChartScore:
    """A chart's priority, together with the terms that produced it."""

    total: float
    contributions: tuple[ScoreContribution, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the score."""
        return {
            "total": self.total,
            "contributions": [c.to_dict() for c in self.contributions],
        }


@dataclass(frozen=True, slots=True)
class ChartSpec:
    """What should be drawn, and why. Never how.

    Holds no figure, no axes, and no data. Column labels are preserved exactly as
    pandas reports them so they still index the frame, and are stringified only
    when serialising -- the same rule the profiler follows.
    """

    chart_type: ChartType
    columns: tuple[Hashable, ...]
    title: str
    reason: VisualizationReason
    score: ChartScore | None = None
    target: Hashable | None = None
    warnings: tuple[VisualizationWarning, ...] = ()

    @property
    def priority(self) -> float:
        """The chart's score, or zero when it has not been ranked."""
        return 0.0 if self.score is None else self.score.total

    def identity(self) -> tuple[str, tuple[tuple[str, str], ...]]:
        """A comparable key for duplicate detection.

        Column order is normalised so that a scatter of ``(a, b)`` and one of
        ``(b, a)`` are recognised as the same request.

        Each label carries its type. Keying on the string alone would make the
        integer label ``1`` and the string label ``"1"`` -- which can coexist in
        one frame -- indistinguishable, so a chart of the second would be dropped
        as a duplicate of the first, with a suppression reason claiming an
        identical chart was already included. A wrong explanation is worse than a
        missing one.
        """
        labels = tuple((type(column).__name__, str(column)) for column in self.columns)
        if self.chart_type is ChartType.SCATTER:
            labels = tuple(sorted(labels))
        return (self.chart_type.value, labels)

    def with_score(self, score: ChartScore) -> ChartSpec:
        """Return a copy carrying ``score``."""
        return ChartSpec(
            chart_type=self.chart_type,
            columns=self.columns,
            title=self.title,
            reason=self.reason,
            score=score,
            target=self.target,
            warnings=self.warnings,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the specification."""
        return {
            "chart_type": self.chart_type.value,
            "columns": [str(column) for column in self.columns],
            "title": self.title,
            "reason": self.reason.to_dict(),
            "score": None if self.score is None else self.score.to_dict(),
            "target": None if self.target is None else str(self.target),
            "warnings": [warning.to_dict() for warning in self.warnings],
        }


@dataclass(frozen=True, slots=True)
class SuppressedCandidate:
    """A candidate that was generated and then dropped, with the reason.

    Recorded so that "why is there no chart for this column?" has an answer.
    """

    chart_type: ChartType
    columns: tuple[Hashable, ...]
    reason: VisualizationReason

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the suppression."""
        return {
            "chart_type": self.chart_type.value,
            "columns": [str(column) for column in self.columns],
            "reason": self.reason.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PlanMetadata:
    """Enough context to explain how a plan came about.

    Not experiment tracking: it answers how big the data was, what task was
    assumed, how many candidates were considered, and how many survived.
    """

    row_count: int
    column_count: int
    candidates_generated: int
    candidates_suppressed: int
    charts_returned: int
    max_charts: int
    target_name: str | None = None
    task_type: TaskType | None = None
    sampling_expected: bool = False
    config: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the plan metadata."""
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "candidates_generated": self.candidates_generated,
            "candidates_suppressed": self.candidates_suppressed,
            "charts_returned": self.charts_returned,
            "max_charts": self.max_charts,
            "target_name": self.target_name,
            "task_type": None if self.task_type is None else self.task_type.value,
            "sampling_expected": self.sampling_expected,
            "config": {str(k): jsonable(v) for k, v in self.config.items()},
        }


@dataclass(frozen=True, slots=True)
class VisualizationPlan:
    """The ranked set of charts recommended for a dataset."""

    charts: tuple[ChartSpec, ...] = ()
    suppressed: tuple[SuppressedCandidate, ...] = ()
    metadata: PlanMetadata | None = None

    def __len__(self) -> int:
        return len(self.charts)

    def __iter__(self):
        return iter(self.charts)

    def of_type(self, chart_type: ChartType) -> tuple[ChartSpec, ...]:
        """Return the recommended charts of one type."""
        return tuple(chart for chart in self.charts if chart.chart_type is chart_type)

    def for_column(self, column: Hashable) -> tuple[ChartSpec, ...]:
        """Return the recommended charts involving one column."""
        return tuple(chart for chart in self.charts if column in chart.columns)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the whole plan.

        Contains no dataframe, array, figure, or renderer.
        """
        return {
            "charts": [chart.to_dict() for chart in self.charts],
            "suppressed": [entry.to_dict() for entry in self.suppressed],
            "metadata": None if self.metadata is None else self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SamplingInfo:
    """What happened to the rows on the way to the renderer.

    Sampling is never silent: even when none occurred the record says so, so a
    reader can tell "all the data" from "ten thousand of eight million".
    """

    source_rows: int
    rendered_rows: int
    sampled: bool = False
    strategy: str = "none"
    random_state: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the sampling record."""
        return {
            "source_rows": self.source_rows,
            "rendered_rows": self.rendered_rows,
            "sampled": self.sampled,
            "strategy": self.strategy,
            "random_state": self.random_state,
        }


@dataclass(frozen=True, slots=True)
class PreparedChart:
    """Render-ready data for one chart.

    The only object in this package that carries actual values. It is handed
    straight to a renderer and is deliberately absent from plan serialisation.
    """

    spec: ChartSpec
    data: Mapping[str, Any]
    sampling: SamplingInfo
    warnings: tuple[VisualizationWarning, ...] = ()

    @property
    def chart_type(self) -> ChartType:
        """The type of chart to draw."""
        return self.spec.chart_type

    def describe(self) -> dict[str, Any]:
        """Return the serialisable context, without the render data itself."""
        return {
            "spec": self.spec.to_dict(),
            "sampling": self.sampling.to_dict(),
            "warnings": [warning.to_dict() for warning in self.warnings],
            "data_keys": sorted(str(key) for key in self.data),
        }


def normalise_columns(columns: Sequence[Hashable] | Hashable) -> tuple[Hashable, ...]:
    """Coerce a column argument into a tuple without splitting a string label."""
    if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
        return (columns,)
    return tuple(columns)
