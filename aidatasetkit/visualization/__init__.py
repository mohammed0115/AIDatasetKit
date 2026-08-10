"""Visualization: deciding what is worth looking at, then drawing it.

Two questions, kept strictly apart:

============================  =================================================
Question                      Owner
============================  =================================================
*What* should be shown?       advisor, candidates, ranking, redundancy
*How* is it drawn?            renderers
============================  =================================================

Everything above the renderer line is plain data. A
:class:`~aidatasetkit.visualization.types.ChartSpec` holds no figure and no
array, so a plan can be produced, ranked, explained, and serialised on a machine
with no plotting library installed at all. Importing this package imports no
plotting library either: the renderer is resolved by name at the moment a render
is actually requested.

The recommendations are rules, not learning. Every one of them follows from
measurements the profiler, the quality inspector, the task detector, and the
statistics package already took, and each carries the evidence it rested on. This
package contains no dtype detection, no missing-value scan, no cardinality count,
no identifier heuristic, no outlier rule, and no correlation formula, because all
six already exist elsewhere in the library.

Nothing here modifies data. A chart may exclude a value it cannot draw; the
caller's frame is unchanged.

**Extending later.** A future evaluation chart -- a confusion matrix, a ROC curve
-- becomes an artifact carrying its own numbers, a ``ChartType`` member, a
preparation rule, and a drawing rule. The advisor, the scoring policy, and the
renderer protocol are untouched, because a fitted model's output enters at the
same point a prepared column does.
"""

from aidatasetkit.visualization.advisor import VisualizationAdvisor
from aidatasetkit.visualization.config import ScoringWeights, VisualizationConfig
from aidatasetkit.visualization.context import VisualizationContext
from aidatasetkit.visualization.service import VisualizationService
from aidatasetkit.visualization.types import (
    ChartScore,
    ChartSpec,
    ChartType,
    PlanMetadata,
    PreparedChart,
    SamplingInfo,
    SuppressedCandidate,
    VisualizationPlan,
    VisualizationReason,
    VisualizationWarning,
)

__all__ = [
    "ChartScore",
    "ChartSpec",
    "ChartType",
    "PlanMetadata",
    "PreparedChart",
    "SamplingInfo",
    "ScoringWeights",
    "SuppressedCandidate",
    "VisualizationAdvisor",
    "VisualizationConfig",
    "VisualizationContext",
    "VisualizationPlan",
    "VisualizationReason",
    "VisualizationService",
    "VisualizationWarning",
]
