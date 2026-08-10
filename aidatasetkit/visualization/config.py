"""Configuration for visualization planning and rendering.

Kept separate from :class:`~aidatasetkit.core.config.KitConfig` on purpose.
Chart counts, point budgets, and renderer names are not modelling thresholds, and
mixing them would make every component depend on settings it has no interest in.

Scoring weights live here too, as a typed policy rather than constants scattered
through the ranking code. Every number that changes a recommendation is visible
in one place and serialisable into the plan that used it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from aidatasetkit.core.exceptions import ConfigurationError

__all__ = ["ScoringWeights", "VisualizationConfig"]


@dataclass(frozen=True, slots=True)
class ScoringWeights:
    """How much each factor moves a chart's priority.

    Positive weights promote, negative ones demote. The defaults encode one
    editorial judgement: a chart that answers a question about the *target*
    outranks a generic distribution, and a chart the quality report has a
    specific reason to want outranks one nobody asked for.

    Attributes:
        target_relevance: Added when the chart involves the target column.
        statistical_evidence: Multiplied by a measured association strength in
            ``[0, 1]``. Only ever applied where a correlation was actually
            computed.
        quality_relevance: Added when a quality finding specifically motivates
            this chart, such as flagged outliers motivating a box plot.
        leakage_review: Added when the quality inspector flagged a possible
            leakage relationship, so the chart surfaces for review.
        near_constant_penalty: Subtracted for a column with almost no variation.
        high_cardinality_penalty: Subtracted for a categorical column that needs
            truncating to be readable.
        missing_penalty: Multiplied by the column's missing ratio.
        readability_penalty: Multiplied by how close a bar chart comes to the
            category budget.
    """

    target_relevance: float = 0.30
    statistical_evidence: float = 0.25
    quality_relevance: float = 0.20
    leakage_review: float = 0.15
    near_constant_penalty: float = 0.35
    high_cardinality_penalty: float = 0.15
    missing_penalty: float = 0.20
    readability_penalty: float = 0.10

    def __post_init__(self) -> None:
        for entry in dataclasses.fields(self):
            value = getattr(self, entry.name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ConfigurationError(
                    f"{entry.name} must be a number, got {value!r}."
                )
            if value < 0:
                raise ConfigurationError(
                    f"{entry.name} must not be negative, got {value}. Penalties are "
                    "named as such and subtracted; a negative weight would invert "
                    "the factor's meaning."
                )

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable mapping of the weights."""
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    """Budgets and policy for the visualization layer.

    Attributes:
        max_charts: Upper bound on the charts a plan may contain. Conservative by
            default: a plan nobody reads is worth nothing.
        max_points: Row budget per rendered chart. Beyond it, preparation samples
            deterministically and says so.
        max_categories: Category budget for a bar chart. Beyond it the remainder
            is folded into a single "Other" bar rather than drawn.
        max_pairwise_columns: How many numeric columns may take part in pairwise
            candidate generation. This is the bound that stops a wide frame from
            producing thousands of scatter candidates.
        max_pairwise_charts: How many feature-to-feature scatter candidates may be
            generated in total.
        max_correlation_columns: How many numeric columns may have their
            correlation with the target measured. Bounds the cost on very wide
            frames; the columns are chosen from profile metadata, without a scan.
        min_heatmap_columns: Fewer numeric columns than this and a heatmap says
            less than the individual charts already do.
        max_heatmap_columns: More than this and the cells are unreadable, so a
            bounded subset is chosen and recorded.
        min_rows_for_distribution: Below this many non-missing values a
            distribution chart is noise.
        stratify_target_samples: Whether sampling for a classification-target
            chart should preserve class proportions.
        random_state: Seed for every sampling decision.
        renderer: Name of the renderer to resolve when one is needed.
        weights: The scoring policy.
    """

    max_charts: int = 10
    max_points: int = 10_000
    max_categories: int = 20
    max_pairwise_columns: int = 8
    max_pairwise_charts: int = 3
    max_correlation_columns: int = 50
    min_heatmap_columns: int = 3
    max_heatmap_columns: int = 15
    min_rows_for_distribution: int = 10
    stratify_target_samples: bool = True
    random_state: int = 42
    renderer: str = "matplotlib"
    weights: ScoringWeights = field(default_factory=ScoringWeights)

    def __post_init__(self) -> None:
        positive = (
            "max_charts",
            "max_points",
            "max_categories",
            "max_pairwise_columns",
            "max_correlation_columns",
            "min_heatmap_columns",
            "max_heatmap_columns",
        )
        for name in positive:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ConfigurationError(
                    f"{name} must be a positive integer, got {value!r}."
                )
        for name in ("max_pairwise_charts", "min_rows_for_distribution"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ConfigurationError(
                    f"{name} must be a non-negative integer, got {value!r}."
                )
        if self.min_heatmap_columns > self.max_heatmap_columns:
            raise ConfigurationError(
                f"min_heatmap_columns ({self.min_heatmap_columns}) cannot exceed "
                f"max_heatmap_columns ({self.max_heatmap_columns})."
            )
        if self.min_heatmap_columns < 2:
            raise ConfigurationError(
                "min_heatmap_columns must be at least 2; a correlation matrix of "
                "one column says nothing."
            )
        if not isinstance(self.renderer, str) or not self.renderer.strip():
            raise ConfigurationError(
                f"renderer must be a non-empty string, got {self.renderer!r}."
            )
        if not isinstance(self.weights, ScoringWeights):
            raise ConfigurationError(
                f"weights must be a ScoringWeights, got {type(self.weights).__name__}."
            )

    def replace(self, **changes: Any) -> VisualizationConfig:
        """Return a new config with ``changes`` applied and re-validated."""
        unknown = set(changes) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise ConfigurationError(f"Unknown visualization options: {sorted(unknown)}.")
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of every configured value."""
        return dataclasses.asdict(self)
