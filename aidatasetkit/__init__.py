"""AIDatasetKit: from a raw tabular frame to a model-ready dataset.

The library is a toolkit, not an AutoML system. It profiles, measures, warns,
prepares, and ranks; it never edits data or picks a model on the analyst's
behalf.

Phase 1 is under construction. The public surface grows one layer at a time and
this module re-exports only what is implemented and tested.
"""

from importlib.metadata import PackageNotFoundError, version
from logging import NullHandler, getLogger

from aidatasetkit.core import (
    AIDatasetKitError,
    Backend,
    ColumnKind,
    ColumnKinds,
    Estimator,
    Interpretability,
    KitConfig,
    PreprocessingProfile,
    ProbabilisticEstimator,
    RunMetadata,
    Severity,
    TargetProfile,
    TaskType,
    detect_column_kinds,
    to_float_array,
)

try:
    __version__ = version("aidatasetkit")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+unknown"

# A library configures no handlers; the application decides where logs go.
getLogger(__name__).addHandler(NullHandler())

__all__ = [
    "AIDatasetKitError",
    "Backend",
    "ColumnKind",
    "ColumnKinds",
    "Estimator",
    "Interpretability",
    "KitConfig",
    "PreprocessingProfile",
    "ProbabilisticEstimator",
    "RunMetadata",
    "Severity",
    "TargetProfile",
    "TaskType",
    "__version__",
    "detect_column_kinds",
    "to_float_array",
]
