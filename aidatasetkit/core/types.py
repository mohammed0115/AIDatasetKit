"""Shared vocabulary: enums, structural protocols, and cross-cutting records.

Everything here is dependency-free with respect to the rest of the library, which
is what lets ``models`` and ``preprocessing`` exchange information without
importing each other.

The estimator contract in this module is structural on purpose. It is defined as
a :class:`typing.Protocol` rather than ``sklearn.base.BaseEstimator`` so that a
future PyTorch or TensorFlow strategy can satisfy it with a thin adapter, without
any change to the facade, the comparator, or the public API.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, Self, runtime_checkable

from aidatasetkit.core.exceptions import SchemaError, ValidationError
from aidatasetkit.core.provenance import EnvironmentVersions, capture_environment

__all__ = [
    "TaskType",
    "SUPERVISED_TASKS",
    "Backend",
    "Interpretability",
    "INTERPRETABILITY_RANK",
    "Severity",
    "ColumnKind",
    "ColumnKinds",
    "Estimator",
    "ProbabilisticEstimator",
    "PreprocessingProfile",
    "TargetProfile",
    "RunMetadata",
]


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class _CoercibleStrEnum(StrEnum):
    """A string enum that can be built from user input with a clear error."""

    @classmethod
    def coerce(cls, value: Self | str) -> Self:
        """Return the member matching ``value``, case-insensitively.

        Raises:
            ValidationError: If ``value`` does not name a member.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalised = value.strip().lower().replace("-", "_").replace(" ", "_")
            for member in cls:
                if member.value == normalised:
                    return member
        options = ", ".join(member.value for member in cls)
        raise ValidationError(
            f"{value!r} is not a valid {cls.__name__}. Valid options: {options}."
        )


class TaskType(_CoercibleStrEnum):
    """Machine-learning task family that a model or a dataset belongs to.

    A model declares which family it serves. Whether a *classification* dataset is
    binary or multiclass is a property of the data, not of the model, and lives in
    :class:`TargetProfile` instead.
    """

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"


#: Task families that require a target column.
SUPERVISED_TASKS: frozenset[TaskType] = frozenset(
    {TaskType.CLASSIFICATION, TaskType.REGRESSION}
)


class Backend(_CoercibleStrEnum):
    """Library that actually implements a model strategy.

    Declared for every strategy so that a mixed catalog stays legible. Values
    beyond ``SKLEARN`` are reserved for adapters; declaring them costs nothing and
    keeps the enum stable when those adapters land.
    """

    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


class Interpretability(_CoercibleStrEnum):
    """How readable a fitted model is to a human analyst.

    Reporting metadata only. It never influences preprocessing or model
    selection; it is a column in the comparison table so the analyst can weigh
    accuracy against explainability themselves.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


#: Sort order for :class:`Interpretability` (higher value is more interpretable).
INTERPRETABILITY_RANK: Mapping[Interpretability, int] = {
    Interpretability.LOW: 0,
    Interpretability.MEDIUM: 1,
    Interpretability.HIGH: 2,
}


class Severity(_CoercibleStrEnum):
    """Severity attached to a data-quality finding."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ColumnKind(_CoercibleStrEnum):
    """Structural kind of a dataframe column.

    This is a description of the *dtype*, not a modelling decision. Mapping kinds
    onto model features (for instance, treating booleans as categorical) is the
    job of the feature detector.
    """

    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    OTHER = "other"


# --------------------------------------------------------------------------- #
# Estimator contract (backend-agnostic)
# --------------------------------------------------------------------------- #


@runtime_checkable
class Estimator(Protocol):
    """The minimum interface a model strategy must produce.

    Any object exposing these four methods qualifies, whether it comes from
    scikit-learn, a gradient-boosting library, or a neural-network wrapper.
    ``get_params``/``set_params`` are part of the contract because pipeline
    composition and cross-validation clone estimators.

    Note:
        ``isinstance`` checks against a runtime-checkable protocol verify that the
        attributes exist, not that their signatures match.
    """

    def fit(self, X: Any, y: Any = None) -> Any: ...

    def predict(self, X: Any) -> Any: ...

    def get_params(self, deep: bool = True) -> dict[str, Any]: ...

    def set_params(self, **params: Any) -> Any: ...


@runtime_checkable
class ProbabilisticEstimator(Estimator, Protocol):
    """An :class:`Estimator` that can also produce class probabilities."""

    def predict_proba(self, X: Any) -> Any: ...


# --------------------------------------------------------------------------- #
# Structural records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ColumnKinds:
    """Structural classification of every column in a dataframe.

    Column labels are preserved exactly as pandas reports them; they are only
    stringified when serialising, so integer labels still index their frame.
    """

    mapping: Mapping[Hashable, ColumnKind]

    def of(self, column: Hashable) -> ColumnKind:
        """Return the kind of ``column``.

        Raises:
            SchemaError: If the column is not present.
        """
        try:
            return self.mapping[column]
        except KeyError:
            raise SchemaError(f"Column {column!r} is not present in the profiled frame.") from None

    def columns_of(self, *kinds: ColumnKind) -> tuple[Hashable, ...]:
        """Return, in original frame order, the columns matching any of ``kinds``."""
        wanted = set(kinds)
        return tuple(name for name, kind in self.mapping.items() if kind in wanted)

    @property
    def numeric(self) -> tuple[Hashable, ...]:
        """Numeric columns, excluding booleans."""
        return self.columns_of(ColumnKind.NUMERIC)

    @property
    def boolean(self) -> tuple[Hashable, ...]:
        """Boolean columns."""
        return self.columns_of(ColumnKind.BOOLEAN)

    @property
    def categorical(self) -> tuple[Hashable, ...]:
        """String, categorical, and other label-like columns."""
        return self.columns_of(ColumnKind.CATEGORICAL)

    @property
    def datetime(self) -> tuple[Hashable, ...]:
        """Datetime columns, timezone-aware or naive."""
        return self.columns_of(ColumnKind.DATETIME)

    @property
    def other(self) -> tuple[Hashable, ...]:
        """Columns whose dtype the library does not classify."""
        return self.columns_of(ColumnKind.OTHER)

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable mapping of column name to kind."""
        return {str(name): kind.value for name, kind in self.mapping.items()}


@dataclass(frozen=True, slots=True)
class PreprocessingProfile:
    """The subset of model capabilities that changes the preprocessing pipeline.

    Two models with an equal profile can share one fitted preprocessor, which is
    what keeps preprocessing reusable while still adapting to each algorithm. The
    profile -- not the model name -- is the cache key used by the builder.
    """

    requires_scaling: bool
    supports_sparse_input: bool
    handles_missing_values: bool

    @property
    def key(self) -> str:
        """A stable identifier suitable for caching and for provenance records."""
        return (
            f"scaling={int(self.requires_scaling)}"
            f",sparse={int(self.supports_sparse_input)}"
            f",native_nan={int(self.handles_missing_values)}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the profile."""
        return {
            "requires_scaling": self.requires_scaling,
            "supports_sparse_input": self.supports_sparse_input,
            "handles_missing_values": self.handles_missing_values,
            "key": self.key,
        }


@dataclass(frozen=True, slots=True)
class TargetProfile:
    """What the target column turned out to be.

    Produced by the task detector. ``detection_note`` records how the task was
    resolved, so a surprising downstream result can be traced back to the moment
    the task was decided.
    """

    task_type: TaskType
    n_classes: int | None = None
    is_binary: bool = False
    positive_label: Any = None
    class_counts: Mapping[Any, int] | None = None
    imbalance_ratio: float | None = None
    detection_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the target profile."""
        return {
            "task_type": self.task_type.value,
            "n_classes": self.n_classes,
            "is_binary": self.is_binary,
            "positive_label": _jsonable(self.positive_label),
            "class_counts": (
                None
                if self.class_counts is None
                else {str(label): int(count) for label, count in self.class_counts.items()}
            ),
            "imbalance_ratio": self.imbalance_ratio,
            "detection_note": self.detection_note,
        }


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """How a single result was produced.

    Attached to training results and evaluation reports so that any number in a
    report can be traced to the model, the parameters, the data size, the
    preprocessing profile, and the library versions behind it.

    ``final_refit_on_full_data`` records whether the estimator was refitted on the
    full training set before predicting. That step is never silent: it is a flag
    here and an informational log record when it happens.
    """

    model_name: str
    task_type: TaskType
    model_parameters: Mapping[str, Any]
    random_state: int | None
    training_rows: int
    feature_count: int
    preprocessing_profile: str
    training_time_seconds: float | None = None
    transformed_feature_count: int | None = None
    final_refit_on_full_data: bool = False
    environment: EnvironmentVersions = field(default_factory=capture_environment)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the run metadata."""
        return {
            "model_name": self.model_name,
            "task_type": self.task_type.value,
            "model_parameters": {
                str(key): _jsonable(value) for key, value in self.model_parameters.items()
            },
            "random_state": self.random_state,
            "training_rows": self.training_rows,
            "feature_count": self.feature_count,
            "transformed_feature_count": self.transformed_feature_count,
            "preprocessing_profile": self.preprocessing_profile,
            "training_time_seconds": self.training_time_seconds,
            "final_refit_on_full_data": self.final_refit_on_full_data,
            "environment": self.environment.to_dict(),
        }


def _jsonable(value: Any) -> Any:
    """Coerce a value into something ``json.dumps`` accepts.

    Model parameters routinely contain estimator instances, numpy scalars, and
    ``None``; provenance must survive all of them rather than raise.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    item_method = getattr(value, "item", None)
    if callable(item_method) and getattr(value, "ndim", None) == 0:
        return _jsonable(item_method())
    return repr(value)
