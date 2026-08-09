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

import pandas as pd

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
    "NumericSummary",
    "ColumnProfile",
    "DatasetProfile",
    "QualityIssue",
    "QualityReport",
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
    """Severity attached to a data-quality finding.

    ``ERROR`` is reserved for findings that are certain and that will break a
    downstream step, such as an infinity in a numeric column. Anything resting on
    a heuristic is at most a ``WARNING``.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


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
class NumericSummary:
    """Basic descriptive statistics for one numeric sample.

    Every field is optional because a sample can be too small or too uniform for
    the statistic to exist. ``None`` means "not defined for this data", never
    "zero" and never "not computed".
    """

    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    q25: float | None = None
    q75: float | None = None
    iqr: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """Return a JSON-serialisable mapping of the summary."""
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "q25": self.q25,
            "q75": self.q75,
            "iqr": self.iqr,
        }


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """What one column looks like.

    A description, not a judgement. The boolean flags record measurements against
    configured thresholds; deciding what to do about them belongs to the quality
    inspector and ultimately to the analyst.

    ``is_id_like`` is the one flag resting on a heuristic rather than a
    measurement, and the quality issue derived from it always asks for review.
    """

    name: Hashable
    detected_kind: ColumnKind
    pandas_dtype: str
    count: int
    missing_count: int
    missing_ratio: float
    unique_count: int
    unique_ratio: float
    is_constant: bool
    is_near_constant: bool
    is_high_cardinality: bool
    is_id_like: bool
    dominant_value: Any = None
    dominant_ratio: float | None = None
    infinite_count: int = 0
    memory_usage_bytes: int = 0
    numeric: NumericSummary | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the column profile."""
        return {
            "name": str(self.name),
            "detected_kind": self.detected_kind.value,
            "pandas_dtype": self.pandas_dtype,
            "count": self.count,
            "missing_count": self.missing_count,
            "missing_ratio": self.missing_ratio,
            "unique_count": self.unique_count,
            "unique_ratio": self.unique_ratio,
            "is_constant": self.is_constant,
            "is_near_constant": self.is_near_constant,
            "is_high_cardinality": self.is_high_cardinality,
            "is_id_like": self.is_id_like,
            "dominant_value": _jsonable(self.dominant_value),
            "dominant_ratio": self.dominant_ratio,
            "infinite_count": self.infinite_count,
            "memory_usage_bytes": self.memory_usage_bytes,
            "numeric": None if self.numeric is None else self.numeric.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """What the dataset looks like.

    Answers "what am I holding?" and nothing else. It reports no opinions; see
    :class:`QualityReport` for "what deserves attention?".
    """

    row_count: int
    column_count: int
    duplicate_row_count: int
    duplicate_row_ratio: float
    total_missing_count: int
    total_missing_ratio: float
    memory_usage_bytes: int
    column_profiles: tuple[ColumnProfile, ...]

    def column(self, name: Hashable) -> ColumnProfile:
        """Return the profile of one column.

        Raises:
            SchemaError: If the column was not profiled.
        """
        for profile in self.column_profiles:
            if profile.name == name:
                return profile
        raise SchemaError(f"Column {name!r} is not present in this profile.")

    def columns_of_kind(self, *kinds: ColumnKind) -> tuple[Hashable, ...]:
        """Return the names of the columns matching any of ``kinds``."""
        wanted = set(kinds)
        return tuple(
            profile.name
            for profile in self.column_profiles
            if profile.detected_kind in wanted
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the whole profile."""
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "duplicate_row_count": self.duplicate_row_count,
            "duplicate_row_ratio": self.duplicate_row_ratio,
            "total_missing_count": self.total_missing_count,
            "total_missing_ratio": self.total_missing_ratio,
            "memory_usage_bytes": self.memory_usage_bytes,
            "column_profiles": [profile.to_dict() for profile in self.column_profiles],
        }

    def to_frame(self) -> pd.DataFrame:
        """Return the column profiles as a dataframe, one row per column."""
        return pd.DataFrame([profile.to_dict() for profile in self.column_profiles])


@dataclass(frozen=True, slots=True)
class QualityIssue:
    """One finding about the data that deserves a human decision.

    ``requires_review`` marks a finding produced by a heuristic. Such a finding
    states what was observed and why it is suspicious; it never states a
    conclusion, and nothing in this library acts on it automatically.
    """

    code: str
    severity: Severity
    message: str
    column: Hashable | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    recommendation: str | None = None
    requires_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the issue."""
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "column": None if self.column is None else str(self.column),
            "details": {str(key): _jsonable(value) for key, value in self.details.items()},
            "recommendation": self.recommendation,
            "requires_review": self.requires_review,
        }


@dataclass(frozen=True, slots=True)
class QualityReport:
    """What deserves attention in the data.

    Complements :class:`DatasetProfile`: the profile describes, this one flags.
    Nothing here has changed the data.
    """

    issues: tuple[QualityIssue, ...] = ()

    @property
    def issue_count(self) -> int:
        """The total number of findings."""
        return len(self.issues)

    @property
    def errors(self) -> tuple[QualityIssue, ...]:
        """Findings that are certain and will break a downstream step."""
        return self.by_severity(Severity.ERROR)

    @property
    def warnings(self) -> tuple[QualityIssue, ...]:
        """Findings that need a decision before modelling."""
        return self.by_severity(Severity.WARNING)

    @property
    def infos(self) -> tuple[QualityIssue, ...]:
        """Findings recorded for context."""
        return self.by_severity(Severity.INFO)

    @property
    def has_errors(self) -> bool:
        """Whether any error-level finding was raised."""
        return bool(self.errors)

    @property
    def has_warnings(self) -> bool:
        """Whether any warning-level finding was raised."""
        return bool(self.warnings)

    @property
    def needs_review(self) -> tuple[QualityIssue, ...]:
        """Heuristic findings awaiting a human judgement."""
        return tuple(issue for issue in self.issues if issue.requires_review)

    def by_severity(self, severity: Severity) -> tuple[QualityIssue, ...]:
        """Return the findings at one severity."""
        return tuple(issue for issue in self.issues if issue.severity is severity)

    def by_code(self, code: str) -> tuple[QualityIssue, ...]:
        """Return the findings carrying one code."""
        return tuple(issue for issue in self.issues if issue.code == code)

    def by_column(self, column: Hashable) -> tuple[QualityIssue, ...]:
        """Return the findings attached to one column."""
        return tuple(issue for issue in self.issues if issue.column == column)

    @property
    def codes(self) -> tuple[str, ...]:
        """Every distinct code present, in order of first appearance."""
        return tuple(dict.fromkeys(issue.code for issue in self.issues))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the whole report."""
        return {
            "issue_count": self.issue_count,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "info_count": len(self.infos),
            "has_errors": self.has_errors,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_frame(self) -> pd.DataFrame:
        """Return the findings as a dataframe, one row per issue."""
        return pd.DataFrame([issue.to_dict() for issue in self.issues])


@dataclass(frozen=True, slots=True)
class TargetProfile:
    """What kind of prediction problem the target might pose.

    Produced by the task detector. ``detection_note`` records how the task was
    resolved, so a surprising downstream result can be traced back to the moment
    the task was decided.

    Classification fields are ``None`` on a regression target and vice versa;
    nothing is fabricated to fill a slot that has no meaning.
    """

    task_type: TaskType
    target_name: str | None = None
    sample_count: int = 0
    missing_count: int = 0
    unique_count: int = 0
    n_classes: int | None = None
    is_binary: bool = False
    classes: tuple[Any, ...] | None = None
    class_counts: Mapping[Any, int] | None = None
    class_ratios: Mapping[Any, float] | None = None
    positive_label: Any = None
    positive_label_resolved: bool = False
    imbalance_ratio: float | None = None
    numeric: NumericSummary | None = None
    detection_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the target profile."""
        return {
            "task_type": self.task_type.value,
            "target_name": self.target_name,
            "sample_count": self.sample_count,
            "missing_count": self.missing_count,
            "unique_count": self.unique_count,
            "n_classes": self.n_classes,
            "is_binary": self.is_binary,
            "classes": (
                None if self.classes is None else [_jsonable(label) for label in self.classes]
            ),
            "class_counts": (
                None
                if self.class_counts is None
                else {str(label): int(count) for label, count in self.class_counts.items()}
            ),
            "class_ratios": (
                None
                if self.class_ratios is None
                else {str(label): float(share) for label, share in self.class_ratios.items()}
            ),
            "positive_label": _jsonable(self.positive_label),
            "positive_label_resolved": self.positive_label_resolved,
            "imbalance_ratio": self.imbalance_ratio,
            "numeric": None if self.numeric is None else self.numeric.to_dict(),
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
