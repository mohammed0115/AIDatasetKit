"""Profiling: describing a dataset, flagging what needs attention, reading a target.

Three questions, three answers, kept separate on purpose:

==========================  =====================================================
Question                    Answer
==========================  =====================================================
What does this look like?   :class:`~aidatasetkit.core.types.DatasetProfile`
What deserves attention?    :class:`~aidatasetkit.core.types.QualityReport`
What problem is this?       :class:`~aidatasetkit.core.types.TargetProfile`
==========================  =====================================================

A profile describes and judges nothing. A quality report judges and changes
nothing. A target profile decides what kind of prediction problem the data poses,
without fitting a model to find out.

Nothing in this package fills, drops, casts, clips, encodes, or excludes
anything. It detects, explains, and recommends; every decision stays with the
analyst.
"""

from aidatasetkit.profiling.checks import DEFAULT_CHECKS, Check
from aidatasetkit.profiling.context import QualityContext
from aidatasetkit.profiling.profiler import DataProfiler
from aidatasetkit.profiling.quality import DataQualityInspector
from aidatasetkit.profiling.task_detector import UNRESOLVED, TaskDetector

__all__ = [
    "DEFAULT_CHECKS",
    "UNRESOLVED",
    "Check",
    "DataProfiler",
    "DataQualityInspector",
    "QualityContext",
    "TaskDetector",
]
