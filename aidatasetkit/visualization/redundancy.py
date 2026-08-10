"""Dropping charts that say what another chart already said.

Two charts can both be individually justified and still be a waste of the
analyst's attention together. This filter removes the second one and records why.

It is a *recommendation-level* filter and nothing more. No feature is called
useless, no column is dropped, and no data is touched -- a suppressed chart can
still be drawn on request through the manual API.
"""

from __future__ import annotations

import logging

from aidatasetkit.visualization.context import VisualizationContext
from aidatasetkit.visualization.types import (
    ChartSpec,
    ChartType,
    SuppressedCandidate,
    VisualizationReason,
)

__all__ = ["RedundancyFilter"]

_logger = logging.getLogger(__name__)


class RedundancyFilter:
    """Removes charts whose message duplicates one already kept."""

    def filter(
        self, specs: list[ChartSpec], context: VisualizationContext
    ) -> tuple[list[ChartSpec], list[SuppressedCandidate]]:
        """Return the charts worth keeping, and the ones dropped as redundant.

        ``specs`` is expected in priority order, so that when two charts overlap
        the better-scoring one is the survivor.
        """
        kept: list[ChartSpec] = []
        dropped: list[SuppressedCandidate] = []
        seen_identities: set[tuple[str, tuple[str, ...]]] = set()
        distribution_columns: set[str] = set()

        for spec in specs:
            identity = spec.identity()
            if identity in seen_identities:
                dropped.append(
                    self._drop(
                        spec,
                        "duplicate_chart",
                        (
                            f"Suppressed because an identical {spec.chart_type.value} "
                            "for the same column(s) is already included."
                        ),
                    )
                )
                continue

            overlap = self._distribution_overlap(spec, distribution_columns)
            if overlap is not None:
                dropped.append(overlap)
                continue

            seen_identities.add(identity)
            if spec.chart_type in (ChartType.HISTOGRAM, ChartType.BOX_PLOT):
                distribution_columns.add(str(spec.columns[0]))
            kept.append(spec)

        if dropped:
            _logger.debug("redundancy filter dropped %d chart(s)", len(dropped))
        return kept, dropped

    @staticmethod
    def _distribution_overlap(
        spec: ChartSpec, distribution_columns: set[str]
    ) -> SuppressedCandidate | None:
        """Suppress a second univariate view of a column with nothing new to add.

        A histogram and a box plot of the same column both describe one
        distribution. Keeping both is justified when the box plot exists *because
        outliers were flagged* -- then it answers a question the histogram does
        not. A box plot recommended only for having a non-zero spread is
        redundant once the histogram is in.
        """
        if spec.chart_type is not ChartType.BOX_PLOT:
            return None
        if spec.reason.code == "flagged_outlier_candidates":
            return None
        if str(spec.columns[0]) not in distribution_columns:
            return None
        return SuppressedCandidate(
            chart_type=spec.chart_type,
            columns=spec.columns,
            reason=VisualizationReason(
                code="redundant_distribution_view",
                message=(
                    f"Suppressed because a histogram of {spec.columns[0]!s} already "
                    "describes this distribution, and no outliers were flagged that "
                    "a box plot would add."
                ),
                evidence={"kept_chart_type": ChartType.HISTOGRAM.value},
            ),
        )

    @staticmethod
    def _drop(spec: ChartSpec, code: str, message: str) -> SuppressedCandidate:
        """Build the suppression record for a dropped chart."""
        return SuppressedCandidate(
            chart_type=spec.chart_type,
            columns=spec.columns,
            reason=VisualizationReason(code=code, message=message),
        )
