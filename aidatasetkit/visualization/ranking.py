"""Scoring charts by how much they are worth looking at.

Deterministic arithmetic over named factors. There is no learned importance
model, no hidden weighting, and nothing that could give two different answers for
the same input. Every term that moves a chart up or down is recorded on the score
itself, so a ranking can be read as a sentence rather than trusted as a number.

Weights live in :class:`~aidatasetkit.visualization.config.ScoringWeights`, not as
constants in this file.
"""

from __future__ import annotations

from collections.abc import Hashable

from aidatasetkit.core.types import ColumnProfile
from aidatasetkit.visualization.context import VisualizationContext
from aidatasetkit.visualization.types import (
    ChartScore,
    ChartSpec,
    ChartType,
    ScoreContribution,
)

__all__ = ["ChartRanker", "BASE_PRIORITY"]

#: Starting priority per chart type, before any evidence is applied.
#:
#: The ordering encodes one judgement: a chart about the whole dataset or the
#: target answers a question the analyst has before they have any others.
BASE_PRIORITY: dict[ChartType, float] = {
    ChartType.TARGET_DISTRIBUTION: 0.90,
    ChartType.MISSING_VALUES: 0.75,
    ChartType.CORRELATION_HEATMAP: 0.65,
    ChartType.GROUPED_BOX: 0.60,
    ChartType.GROUPED_BAR: 0.60,
    ChartType.SCATTER: 0.55,
    ChartType.HISTOGRAM: 0.50,
    ChartType.BAR: 0.45,
    ChartType.BOX_PLOT: 0.40,
}


class ChartRanker:
    """Assigns each candidate a priority and orders the survivors."""

    def score(
        self,
        spec: ChartSpec,
        context: VisualizationContext,
        associations: dict[Hashable, float] | None = None,
    ) -> ChartSpec:
        """Return ``spec`` carrying a :class:`ChartScore`."""
        weights = context.config.weights
        contributions: list[ScoreContribution] = [
            ScoreContribution(
                factor="base",
                value=BASE_PRIORITY.get(spec.chart_type, 0.40),
                detail=f"base priority for {spec.chart_type.value}",
            )
        ]

        contributions.extend(self._target_terms(spec, context, weights))
        contributions.extend(self._evidence_terms(spec, associations or {}, weights))
        contributions.extend(self._quality_terms(spec, context, weights))
        contributions.extend(self._readability_terms(spec, context, weights))

        total = sum(contribution.value for contribution in contributions)
        return spec.with_score(
            ChartScore(total=round(total, 6), contributions=tuple(contributions))
        )

    def rank(self, specs: list[ChartSpec]) -> list[ChartSpec]:
        """Order by descending priority, breaking ties deterministically.

        The tie-break runs on chart type and stringified column labels, never on
        set iteration or object identity, so the same input always yields the same
        order in the same process and in the next one.
        """
        return sorted(
            specs,
            key=lambda spec: (
                -spec.priority,
                spec.chart_type.value,
                tuple(str(column) for column in spec.columns),
            ),
        )

    # ------------------------------------------------------------------ #
    # Factors
    # ------------------------------------------------------------------ #

    def _target_terms(self, spec, context, weights) -> list[ScoreContribution]:
        """Charts that involve the target answer the question being asked."""
        if not context.has_target or context.target not in spec.columns:
            return []
        return [
            ScoreContribution(
                factor="target_relevance",
                value=weights.target_relevance,
                detail="chart involves the target column",
            )
        ]

    def _evidence_terms(self, spec, associations, weights) -> list[ScoreContribution]:
        """A measured association is the only thing that earns this term.

        Only the chart's *own* recorded measurement counts. Falling back to
        whichever of its columns happened to appear in the association table would
        award a dataset-level chart -- a missing-value view names every column
        with a gap -- the correlation of an unrelated feature, making its score
        depend on the frame's column order.
        """
        strength = spec.reason.evidence.get("abs_correlation")
        if strength is None:
            return []
        bounded = max(0.0, min(1.0, float(strength)))
        return [
            ScoreContribution(
                factor="statistical_evidence",
                value=weights.statistical_evidence * bounded,
                detail=f"measured absolute correlation {bounded:.2f}",
            )
        ]

    def _quality_terms(self, spec, context, weights) -> list[ScoreContribution]:
        """Findings from the quality inspector that motivate or demote a chart."""
        terms: list[ScoreContribution] = []

        if spec.chart_type is ChartType.MISSING_VALUES:
            terms.append(
                ScoreContribution(
                    factor="quality_relevance",
                    value=weights.quality_relevance,
                    detail="the dataset has missing values to show",
                )
            )
        if spec.chart_type is ChartType.BOX_PLOT and spec.reason.code == (
            "flagged_outlier_candidates"
        ):
            terms.append(
                ScoreContribution(
                    factor="quality_relevance",
                    value=weights.quality_relevance,
                    detail="outlier candidates were flagged for this column",
                )
            )
        if spec.chart_type is ChartType.TARGET_DISTRIBUTION and context.has_target:
            if context.has_issue(context.target, "class_imbalance"):
                terms.append(
                    ScoreContribution(
                        factor="quality_relevance",
                        value=weights.quality_relevance,
                        detail="the target was reported as imbalanced",
                    )
                )
        if spec.reason.code == "possible_target_leakage_review":
            terms.append(
                ScoreContribution(
                    factor="leakage_review",
                    value=weights.leakage_review,
                    detail="flagged for possible target leakage and needs review",
                )
            )

        terms.extend(self._column_penalties(spec, context, weights))
        return terms

    #: Charts that describe the frame rather than particular columns.
    _DATASET_LEVEL = (ChartType.MISSING_VALUES, ChartType.CORRELATION_HEATMAP)

    def _column_penalties(self, spec, context, weights) -> list[ScoreContribution]:
        """Demote charts whose column is barely worth drawing.

        Dataset-level charts are exempt. They name every column they cover, so
        charging them one penalty per column would push the missing-value chart
        further down the further worse the missing data got -- exactly backwards.
        """
        if spec.chart_type in self._DATASET_LEVEL:
            return []

        terms: list[ScoreContribution] = []
        for column in spec.columns:
            if context.is_target(column):
                continue
            profile = self._profile_for(context, column)
            if profile is None:
                continue
            if profile.is_near_constant:
                terms.append(
                    ScoreContribution(
                        factor="near_constant_penalty",
                        value=-weights.near_constant_penalty,
                        detail=(
                            f"{column!s} holds one value in "
                            f"{profile.dominant_ratio:.1%} of rows"
                        ),
                    )
                )
            if profile.is_high_cardinality:
                terms.append(
                    ScoreContribution(
                        factor="high_cardinality_penalty",
                        value=-weights.high_cardinality_penalty,
                        detail=f"{column!s} has {profile.unique_count} categories",
                    )
                )
            if profile.missing_ratio > 0.0:
                terms.append(
                    ScoreContribution(
                        factor="missing_penalty",
                        value=-weights.missing_penalty * profile.missing_ratio,
                        detail=f"{column!s} is {profile.missing_ratio:.1%} missing",
                    )
                )
        return terms

    def _readability_terms(self, spec, context, weights) -> list[ScoreContribution]:
        """A bar chart crowded with categories is harder to read."""
        if spec.chart_type not in (ChartType.BAR, ChartType.GROUPED_BAR):
            return []
        profile = self._profile_for(context, spec.columns[0])
        if profile is None or profile.unique_count <= 1:
            return []
        crowding = min(1.0, profile.unique_count / context.config.max_categories)
        if crowding <= 0.5:
            return []
        return [
            ScoreContribution(
                factor="readability_penalty",
                value=-weights.readability_penalty * crowding,
                detail=f"{profile.unique_count} categories to draw",
            )
        ]

    @staticmethod
    def _profile_for(
        context: VisualizationContext, column: Hashable
    ) -> ColumnProfile | None:
        """Look up a column profile, tolerating a chart on a column not profiled."""
        return context.profile_of(column)
