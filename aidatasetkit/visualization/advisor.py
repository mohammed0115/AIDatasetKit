"""Turning what is known about a dataset into a ranked set of charts.

The advisor sequences four components and contributes no rules of its own:

.. code-block:: text

    candidate generation   what could be shown, bounded at the source
            |
    scoring                how much each is worth looking at
            |
    redundancy filtering   drop what another chart already said
            |
    ranking + top N        order, then cut to the budget

Scoring runs before redundancy filtering deliberately: when two charts overlap,
the filter should keep the better one, and it cannot know which that is until both
have been scored.

Nothing in this module imports a plotting library, and nothing it returns
contains one.
"""

from __future__ import annotations

import logging

from aidatasetkit.visualization.candidates import CandidateGenerator
from aidatasetkit.visualization.context import VisualizationContext
from aidatasetkit.visualization.ranking import ChartRanker
from aidatasetkit.visualization.redundancy import RedundancyFilter
from aidatasetkit.visualization.types import (
    ChartSpec,
    PlanMetadata,
    SuppressedCandidate,
    VisualizationPlan,
    VisualizationReason,
)

__all__ = ["VisualizationAdvisor"]

_logger = logging.getLogger(__name__)


class VisualizationAdvisor:
    """Recommends what to visualize, and explains every choice.

    Args:
        generator: Produces candidate charts from profile and quality metadata.
        ranker: Scores and orders them.
        redundancy: Removes charts that duplicate another chart's message.
    """

    def __init__(
        self,
        generator: CandidateGenerator | None = None,
        ranker: ChartRanker | None = None,
        redundancy: RedundancyFilter | None = None,
    ) -> None:
        self._generator = generator if generator is not None else CandidateGenerator()
        self._ranker = ranker if ranker is not None else ChartRanker()
        self._redundancy = redundancy if redundancy is not None else RedundancyFilter()

    def recommend(self, context: VisualizationContext) -> VisualizationPlan:
        """Return a ranked plan for ``context``.

        The same frame, profile, quality report, target profile, and config always
        produce the same plan: the same candidates, the same scores, and the same
        order.
        """
        batch = self._generator.generate(context)

        scored = [
            self._ranker.score(spec, context, batch.associations) for spec in batch.specs
        ]
        ordered = self._ranker.rank(scored)
        kept, redundant = self._redundancy.filter(ordered, context)

        selected = kept[: context.config.max_charts]
        over_budget = self._over_budget(kept[context.config.max_charts :], context)

        suppressed = tuple(batch.suppressed) + tuple(redundant) + tuple(over_budget)
        _logger.debug(
            "recommended %d chart(s) from %d candidate(s); %d suppressed",
            len(selected),
            len(batch.specs),
            len(suppressed),
        )

        return VisualizationPlan(
            charts=tuple(selected),
            suppressed=suppressed,
            metadata=self._metadata(context, batch, selected, suppressed),
        )

    @staticmethod
    def _over_budget(
        remainder: list[ChartSpec], context: VisualizationContext
    ) -> list[SuppressedCandidate]:
        """Record charts that were good enough but did not fit the budget."""
        return [
            SuppressedCandidate(
                chart_type=spec.chart_type,
                columns=spec.columns,
                reason=VisualizationReason(
                    code="over_max_charts",
                    message=(
                        f"Suppressed because the plan is limited to "
                        f"{context.config.max_charts} charts and this one ranked "
                        "below the cut."
                    ),
                    evidence={
                        "priority": spec.priority,
                        "max_charts": context.config.max_charts,
                    },
                ),
            )
            for spec in remainder
        ]

    @staticmethod
    def _metadata(
        context: VisualizationContext,
        batch,
        selected: list[ChartSpec],
        suppressed: tuple[SuppressedCandidate, ...],
    ) -> PlanMetadata:
        """Record enough to explain how this plan came about."""
        target_profile = context.target_profile
        return PlanMetadata(
            row_count=context.profile.row_count,
            column_count=context.profile.column_count,
            candidates_generated=len(batch.specs),
            candidates_suppressed=len(suppressed),
            charts_returned=len(selected),
            max_charts=context.config.max_charts,
            target_name=None if context.target is None else str(context.target),
            task_type=None if target_profile is None else target_profile.task_type,
            sampling_expected=context.profile.row_count > context.config.max_points,
            config=context.config.to_dict(),
        )
