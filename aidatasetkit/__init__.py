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


def __getattr__(name: str):
    """Resolve the high-level entry point on first use, not on import.

    ``from aidatasetkit import AIDataFacade`` has to work -- an API nobody can
    find is not an API. But the facade reaches the training layer and therefore
    scikit-learn, which costs several seconds, and importing it eagerly would
    make ``import aidatasetkit`` pay that price for every caller including the
    ones that only wanted a type. The CLI in particular keeps its file checks
    ahead of any heavy import so that a mistyped path costs milliseconds; an
    eager facade here would quietly undo that.

    PEP 562 lets the name be discoverable and the cost be deferred to whoever
    actually asks for it.
    """
    if name == "AIDataFacade":
        from aidatasetkit.facade import AIDataFacade

        return AIDataFacade
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Add the lazily-resolved names to the module's real directory.

    A union, not a replacement. Returning ``__all__`` alone hid every submodule
    and every module dunder, so ``dir(aidatasetkit)`` stopped showing ``core``,
    ``training`` and the rest -- names the documentation tells people to import.
    """
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "AIDataFacade",
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
