"""Statistics: descriptive measures, moments, frequencies, and association.

The package is split by the *shape of the input* rather than by textbook chapter,
which is what keeps each part small and testable:

===========================  ===================================================
Input                        Entry point
===========================  ===================================================
one numeric sample           :class:`~aidatasetkit.statistics.engine.StatisticsEngine`
one sample of any type       :class:`~aidatasetkit.statistics.frequency.FrequencyTable`
two paired numeric samples   :func:`~aidatasetkit.statistics.bivariate.covariance`,
                             :func:`~aidatasetkit.statistics.bivariate.correlation`
a moment of any order        :mod:`aidatasetkit.statistics.moments`
===========================  ===================================================
"""

from aidatasetkit.statistics.bivariate import (
    CorrelationMethod,
    correlation,
    covariance,
)
from aidatasetkit.statistics.engine import (
    Center,
    DescriptiveSummary,
    Quartiles,
    StatisticsEngine,
)
from aidatasetkit.statistics.frequency import (
    FrequencyNanPolicy,
    FrequencySort,
    FrequencyTable,
)
from aidatasetkit.statistics.moments import (
    central_moment,
    moment_about,
    raw_moment,
)

__all__ = [
    "Center",
    "CorrelationMethod",
    "DescriptiveSummary",
    "FrequencyNanPolicy",
    "FrequencySort",
    "FrequencyTable",
    "Quartiles",
    "StatisticsEngine",
    "central_moment",
    "correlation",
    "covariance",
    "moment_about",
    "raw_moment",
]
