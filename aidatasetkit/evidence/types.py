"""The audit contract: what an artifact contains and what each part means.

Everything here is a record of something another layer already decided. Nothing
in this module measures, infers, or judges a dataset -- the profiler, the quality
inspector, the planner, and the model registry did that, and evidence exists to
state their conclusions in one place, in one format, with the source of each one
attached.

The artifact is versioned separately from the library. A reader five versions
from now needs to know which contract a stored file was written against, and the
package version does not answer that: most releases will not change the schema,
and the ones that do must be identifiable without consulting a changelog.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd

from aidatasetkit.core.exceptions import EvidenceError
from aidatasetkit.core.provenance import EnvironmentVersions
from aidatasetkit.core.types import Severity, TaskType
from aidatasetkit.evidence.serialization import canonical, type_name

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "AuditArtifact",
    "AuditStage",
    "ColumnEvidence",
    "ConfigIdentity",
    "DatasetIdentity",
    "DecisionEvidence",
    "EvidenceSource",
    "FeatureLineage",
    "FindingEvidence",
    "FitScope",
    "ChunkedApproximationEvidence",
    "ChunkedColumnEvidence",
    "ChunkedProfilingEvidence",
    "ChunkedSamplingEvidence",
    "IngestionEvidence",
    "LabelRef",
    "ModelEvidence",
    "TargetEvidence",
    "Verdict",
]

#: Version of the artifact *format*, not of the library. Bumped whenever the
#: shape or meaning of the record changes -- a field added, removed or redefined
#: -- so a stored artifact can always be read against the contract it was written
#: against. An additive change is a minor bump, anything else a major one.
#: ``lineage.json`` carries the same version. The publication layout
#: (``runs/<run_id>`` and ``CURRENT``) is versioned separately.
#:
#: 1.0 -- the first published contract.
#: 1.1 -- G1-W1: the top-level ``ingestion`` record.
#: 1.2 -- G1-W5: the top-level ``chunked_profiling`` record. ``null`` unless the
#:       caller opted into chunked CSV/TSV profiling. Approximate numbers live
#:       only inside that record and are labeled; they are not verdict inputs.
#: 1.3 -- G1-W6: the top-level ``source_selector`` record. ``null`` unless a
#:       relation inside the file was chosen. A SQLite table is
#:       ``{"kind": "table", "name": ...}``. An Excel worksheet is
#:       ``{"kind": "sheet", "name": ...}``. The kind is an open string; ``sheet``
#:       does not require a new schema version.
#: 1.4 -- G1-W8: the top-level ``date_inference`` record. A full audit records
#:       the fixed strict-ISO policy and any textual columns that met it.
#:       ``null`` on the chunked profile-only route, which does not scan dates.
#:       The record is evidence. It does not convert values and does not change
#:       detected kind, verdict, or the dataset fingerprint.
#: 1.5 -- G2-W1: each column evidence record gains a nullable ``datetime``
#:       summary. Native datetime columns carry bounded range, observed spacing,
#:       and optional explicit fixed-cadence gap measurements.
ARTIFACT_SCHEMA_VERSION = "1.5"


class EvidenceSource(StrEnum):
    """Which layer established a recorded fact.

    A reader's first question about any line in an audit is "who says so". An
    enum rather than a module path: paths move when files are reorganised, and a
    stored artifact would then name a module that no longer exists.
    """

    PROFILING = "profiling"
    QUALITY = "quality"
    TASK_DETECTION = "task_detection"
    PREPROCESSING = "preprocessing"
    MODEL_CAPABILITIES = "model_capabilities"
    ENVIRONMENT = "environment"


class FitScope(StrEnum):
    """What data a learned transformation was allowed to see.

    The field exists for leakage auditing. ``TRAINING_ONLY`` is a claim that a
    statistic came from training rows and nothing else, and it is the difference
    between a defensible pipeline and one whose validation score means nothing.
    """

    #: Learned from the fitting rows alone.
    TRAINING_ONLY = "training_only"
    #: Nothing is learned; the step is a pure function of each row.
    NO_FIT = "no_fit"
    #: The column never reaches a transformer.
    NOT_APPLICABLE = "not_applicable"


class AuditStage(StrEnum):
    """How far a run got, and therefore how much the artifact can say.

    An audit is useful before a preprocessor exists and useful again after one
    has been fitted. Recording which of those produced the file stops a reader
    from taking the absence of lineage as evidence that nothing was transformed.
    ``profiled`` is earlier than ``inspected``: the population was measured, and
    quality inspection, task detection and preprocessing did not run.
    """

    #: Dataset profiling completed. Quality, task detection and preprocessing did not run.
    PROFILED = "profiled"
    #: Profiled and quality-checked. No preprocessing was planned.
    INSPECTED = "inspected"
    #: A preprocessing plan exists. Nothing has been fitted.
    PLANNED = "planned"
    #: A preprocessor was fitted, so lineage is observed rather than intended.
    PREPARED = "prepared"


class Verdict(StrEnum):
    """The single conservative conclusion an artifact carries.

    Deliberately not a grade and deliberately not a compliance claim. Each value
    says what was found, not what is true of the data in general -- ``READY``
    means no blocker was found by the checks that ran, which is a much narrower
    statement than "safe".
    """

    #: No findings above INFO, and nothing awaiting a human decision.
    READY = "ready"
    #: Warnings were raised. Nothing blocks proceeding.
    READY_WITH_WARNINGS = "ready_with_warnings"
    #: Something needs a person: a held-back feature, an unresolved ambiguity.
    REVIEW_REQUIRED = "review_required"
    #: An error-severity finding, or the analysis could not complete.
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class LabelRef:
    """A column label, kept safe from the ``0`` versus ``"0"`` collision.

    Column labels are not always strings, and two labels with the same text form
    are still two columns. Carrying the original type alongside the text means an
    artifact can be read back without guessing.
    """

    name: str
    label_type: str = "str"

    @classmethod
    def of(cls, label: Any) -> LabelRef:
        """Return a reference to ``label``, preserving its type."""
        return cls(name=str(label), label_type=type_name(label))

    @property
    def token(self) -> str:
        """A collision-free text form, usable as a key or a fingerprint input."""
        return f"{self.label_type}:{self.name}"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "label_type": self.label_type}


@dataclass(frozen=True, slots=True)
class FindingEvidence:
    """One quality finding, preserved structurally rather than as prose.

    Every field comes from a :class:`~aidatasetkit.core.types.QualityIssue`. The
    ``details`` mapping is carried through unchanged because it is where the
    numbers live -- a count, a ratio, the threshold that was crossed -- and a
    finding reduced to its message is a finding nobody can act on
    programmatically.
    """

    code: str
    severity: Severity
    message: str
    column: LabelRef | None = None
    recommendation: str | None = None
    requires_review: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)
    source: EvidenceSource = EvidenceSource.QUALITY

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "column": self.column.to_dict() if self.column else None,
            "recommendation": self.recommendation,
            "requires_review": self.requires_review,
            "details": canonical(dict(self.details), path=f"finding[{self.code}].details"),
            "source": self.source.value,
        }


@dataclass(frozen=True, slots=True)
class ColumnEvidence:
    """What profiling measured about one column.

    Deliberately free of dataset values. Counts, ratios, and flags describe a
    column without reproducing it, which is the line this artifact does not
    cross by default -- see :mod:`aidatasetkit.evidence.builder` for how the most
    frequent value is handled.
    """

    name: LabelRef
    detected_kind: str
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
    infinite_count: int
    dominant_ratio: float | None = None
    dominant_value_digest: str | None = None
    numeric: Mapping[str, Any] | None = None
    source: EvidenceSource = EvidenceSource.PROFILING
    datetime: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.datetime is not None:
            _validate_datetime_evidence(self.datetime)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.to_dict(),
            "detected_kind": self.detected_kind,
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
            "infinite_count": self.infinite_count,
            "dominant_ratio": self.dominant_ratio,
            "dominant_value_digest": self.dominant_value_digest,
            "numeric": canonical(dict(self.numeric), path="column.numeric")
            if self.numeric
            else None,
            "datetime": canonical(dict(self.datetime), path="column.datetime")
            if self.datetime
            else None,
            "source": self.source.value,
        }


_DATETIME_FIELDS = frozenset(
    {
        "total_count",
        "valid_count",
        "missing_count",
        "distinct_valid_count",
        "duplicate_timestamp_count",
        "minimum",
        "maximum",
        "timezone",
        "resolution",
        "interval_count",
        "distinct_spacing_count",
        "minimum_spacing",
        "maximum_spacing",
        "is_regular",
        "observed_cadence",
        "expected_cadence",
        "missing_expected_observation_count",
        "gap_count",
        "off_grid_observation_count",
    }
)
_DATETIME_COUNT_FIELDS = (
    "total_count",
    "valid_count",
    "missing_count",
    "distinct_valid_count",
    "duplicate_timestamp_count",
    "interval_count",
    "distinct_spacing_count",
)


def _validate_datetime_evidence(value: Mapping[str, Any]) -> None:
    """Refuse internally inconsistent temporal facts before serialization."""
    if not isinstance(value, Mapping) or set(value) != _DATETIME_FIELDS:
        raise EvidenceError("Datetime evidence must contain exactly the schema 1.5 fields.")

    for key in _DATETIME_COUNT_FIELDS:
        count = value[key]
        if type(count) is not int or count < 0:
            raise EvidenceError(f"Datetime evidence {key} must be a nonnegative integer.")
    if value["total_count"] != value["valid_count"] + value["missing_count"]:
        raise EvidenceError("Datetime evidence total_count must equal valid_count + missing_count.")
    if value["distinct_valid_count"] > value["valid_count"]:
        raise EvidenceError("Datetime evidence distinct_valid_count exceeds valid_count.")
    if value["duplicate_timestamp_count"] != (
        value["valid_count"] - value["distinct_valid_count"]
    ):
        raise EvidenceError("Datetime evidence duplicate count is inconsistent.")
    expected_intervals = max(value["distinct_valid_count"] - 1, 0)
    if value["interval_count"] != expected_intervals:
        raise EvidenceError("Datetime evidence interval_count is inconsistent.")
    if value["distinct_spacing_count"] > value["interval_count"]:
        raise EvidenceError("Datetime evidence distinct spacing count exceeds interval_count.")

    minimum = _parse_datetime_endpoint(value["minimum"], "minimum")
    maximum = _parse_datetime_endpoint(value["maximum"], "maximum")
    if (minimum is None) != (maximum is None):
        raise EvidenceError("Datetime evidence minimum and maximum must both be null or present.")
    if (minimum is None) != (value["valid_count"] == 0):
        raise EvidenceError("Datetime evidence endpoints are inconsistent with valid_count.")

    timezone = value["timezone"]
    if timezone is not None and (not isinstance(timezone, str) or not timezone):
        raise EvidenceError("Datetime evidence timezone must be a non-empty string or null.")
    if minimum is not None:
        endpoints_aware = minimum[1]
        if maximum[1] != endpoints_aware:
            raise EvidenceError("Datetime evidence endpoints disagree on timezone awareness.")
        if (timezone is not None) != endpoints_aware:
            raise EvidenceError("Datetime evidence timezone does not match its endpoints.")
        if minimum > maximum:
            raise EvidenceError("Datetime evidence minimum is later than maximum.")

    if not isinstance(value["resolution"], str) or value["resolution"] not in {
        "s",
        "ms",
        "us",
        "ns",
    }:
        raise EvidenceError("Datetime evidence resolution is unsupported.")

    minimum_spacing = _parse_positive_duration(value["minimum_spacing"], "minimum_spacing")
    maximum_spacing = _parse_positive_duration(value["maximum_spacing"], "maximum_spacing")
    if (value["interval_count"] == 0) != (minimum_spacing is None):
        raise EvidenceError("Datetime evidence spacing is inconsistent with interval_count.")
    if (minimum_spacing is None) != (maximum_spacing is None):
        raise EvidenceError("Datetime evidence spacing bounds must both be null or present.")
    if minimum_spacing is not None and minimum_spacing > maximum_spacing:
        raise EvidenceError("Datetime evidence minimum_spacing exceeds maximum_spacing.")

    regular = value["is_regular"]
    if value["interval_count"] == 0:
        if regular is not None or value["distinct_spacing_count"] != 0:
            raise EvidenceError("Datetime evidence with no intervals must have unknown regularity.")
        if value["observed_cadence"] is not None:
            raise EvidenceError("Datetime evidence without intervals cannot have observed cadence.")
    elif regular is True:
        if value["distinct_spacing_count"] != 1:
            raise EvidenceError("Regular datetime evidence must have one distinct spacing.")
        if value["observed_cadence"] != value["minimum_spacing"] or (
            value["observed_cadence"] != value["maximum_spacing"]
        ):
            raise EvidenceError("Regular datetime evidence cadence and spacing bounds disagree.")
    elif regular is False:
        if value["distinct_spacing_count"] < 2 or value["observed_cadence"] is not None:
            raise EvidenceError("Irregular datetime evidence has inconsistent cadence.")
    else:
        raise EvidenceError("Datetime evidence with intervals must state regularity.")

    expected_cadence = value["expected_cadence"]
    gap_fields = (
        "missing_expected_observation_count",
        "gap_count",
        "off_grid_observation_count",
    )
    if expected_cadence is None:
        if any(value[key] is not None for key in gap_fields):
            raise EvidenceError("Datetime gap counts require an expected cadence.")
    else:
        _parse_positive_duration(expected_cadence, "expected_cadence")
        for key in gap_fields:
            count = value[key]
            if type(count) is not int or count < 0:
                raise EvidenceError(
                    f"Datetime evidence {key} must be a nonnegative integer with a cadence."
                )
        if value["gap_count"] > value["missing_expected_observation_count"]:
            raise EvidenceError("Datetime evidence gap_count exceeds missing positions.")
        if value["off_grid_observation_count"] > value["distinct_valid_count"]:
            raise EvidenceError("Datetime evidence off-grid count exceeds distinct timestamps.")


_DATETIME_ENDPOINT = re.compile(
    r"(?P<year>-?[0-9]{3,12})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
    r"T(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})"
    r"(?:\.(?P<fraction>[0-9]{6}|[0-9]{9}))?"
    r"(?P<offset>(?P<sign>[+-])(?P<offset_hour>[0-9]{2}):"
    r"(?P<offset_minute>[0-9]{2})(?::(?P<offset_second>[0-9]{2})"
    r"(?:\.(?P<offset_fraction>[0-9]{6}))?)?)?"
)


def _parse_datetime_endpoint(value: Any, field_name: str) -> tuple[int, bool] | None:
    """Validate pandas' canonical ISO form; compare instants with unbounded integers."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise EvidenceError(f"Datetime evidence {field_name} must be ISO text or null.")
    match = _DATETIME_ENDPOINT.fullmatch(value)
    if match is None:
        raise EvidenceError(f"Datetime evidence {field_name} is not canonical endpoint text.")
    parts = match.groupdict()
    try:
        year, month, day, hour, minute, second = (
            int(parts[key]) for key in ("year", "month", "day", "hour", "minute", "second")
        )
        if parts["year"] != f"{year:04d}":
            raise ValueError("noncanonical year")
        month_lengths = (31, 29 if calendar.isleap(year) else 28, 31, 30, 31, 30,
                         31, 31, 30, 31, 30, 31)
        if not (1 <= month <= 12 and 1 <= day <= month_lengths[month - 1]):
            raise ValueError("impossible calendar date")
        if not (hour < 24 and minute < 60 and second < 60):
            raise ValueError("impossible time")
        fraction = parts["fraction"]
        nanoseconds = int(fraction.ljust(9, "0")) if fraction else 0
        if fraction and (nanoseconds == 0 or (len(fraction) == 9 and nanoseconds % 1000 == 0)):
            raise ValueError("noncanonical fraction")

        offset = 0
        if parts["offset"]:
            offset_hour = int(parts["offset_hour"])
            offset_minute = int(parts["offset_minute"])
            offset_second = int(parts["offset_second"] or 0)
            offset_fraction = int(parts["offset_fraction"] or 0)
            if not (offset_hour < 24 and offset_minute < 60 and offset_second < 60):
                raise ValueError("impossible offset")
            if parts["offset_fraction"] and offset_fraction == 0:
                raise ValueError("noncanonical offset fraction")
            if parts["offset_second"] and offset_second == offset_fraction == 0:
                raise ValueError("noncanonical offset seconds")
            offset = ((offset_hour * 60 + offset_minute) * 60 + offset_second) * 10**9
            offset += offset_fraction * 1000
            if parts["sign"] == "-":
                if offset == 0:
                    raise ValueError("noncanonical negative zero offset")
                offset = -offset

        # Gregorian ordinal arithmetic works for year zero and negative/expanded
        # years too. 719163 is the ordinal of 1970-01-01.
        previous_year = year - 1
        days = (
            365 * previous_year + previous_year // 4 - previous_year // 100
            + previous_year // 400
            + sum(month_lengths[:month - 1])
            + day - 719163
        )
        instant = ((days * 24 + hour) * 3600 + minute * 60 + second) * 10**9
        instant += nanoseconds - offset
        if not ((-(2**63) + 1) * 10**9 <= instant <= (2**63 - 1) * 10**9):
            raise ValueError("outside native datetime range")
        return instant, parts["offset"] is not None
    except ValueError as error:
        raise EvidenceError(f"Datetime evidence {field_name} is not a valid canonical endpoint.") from error


def _parse_positive_duration(value: Any, field_name: str) -> pd.Timedelta | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"Datetime evidence {field_name} must be duration text.")
    try:
        duration = pd.Timedelta(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise EvidenceError(f"Datetime evidence {field_name} is not a fixed duration.") from error
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        raise EvidenceError(f"Datetime evidence {field_name} must be positive and finite.")
    if duration.isoformat() != value:
        raise EvidenceError(f"Datetime evidence {field_name} must use canonical ISO duration text.")
    return duration


@dataclass(frozen=True, slots=True)
class DecisionEvidence:
    """What preprocessing decided about one feature, and why.

    The interesting field is ``model_requirement``. A step is often present not
    because of the column but because of the estimator that will consume it, and
    an audit that showed the scaler without saying which capability asked for it
    would leave the reader to guess at a causal story. It names the declared
    capability and nothing beyond it.
    """

    feature: LabelRef
    role: str
    action: str
    reason_code: str
    reason: str
    steps: tuple[str, ...] = ()
    requires_review: bool = False
    input_kind: str | None = None
    outputs: tuple[str, ...] = ()
    fit_scope: FitScope = FitScope.NOT_APPLICABLE
    model_requirement: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    source: EvidenceSource = EvidenceSource.PREPROCESSING

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature.to_dict(),
            "role": self.role,
            "action": self.action,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "steps": list(self.steps),
            "requires_review": self.requires_review,
            "input_kind": self.input_kind,
            "outputs": list(self.outputs),
            "fit_scope": self.fit_scope.value,
            "model_requirement": self.model_requirement,
            "details": canonical(dict(self.details), path="decision.details"),
            "source": self.source.value,
        }


@dataclass(frozen=True, slots=True)
class FeatureLineage:
    """Where one input column ended up.

    Taken from the preprocessing layer's own lineage rather than derived by
    reading output names. ``city_Riyadh`` looks like it decomposes into a column
    and a level, and a category containing an underscore is all it takes for that
    guess to be wrong.
    """

    source: LabelRef
    source_kind: str
    action: str
    steps: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    fit_scope: FitScope = FitScope.NOT_APPLICABLE
    reason: str | None = None
    observed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "source_kind": self.source_kind,
            "action": self.action,
            "steps": list(self.steps),
            "outputs": list(self.outputs),
            "fit_scope": self.fit_scope.value,
            "reason": self.reason,
            "observed": self.observed,
        }


@dataclass(frozen=True, slots=True)
class ModelEvidence:
    """The model context a preparation was planned for.

    Capabilities only. No estimator is constructed, nothing is fitted, and no
    estimator object reaches the artifact -- what matters for an audit is which
    declared requirements shaped the pipeline, and those are metadata.
    """

    canonical_name: str
    backend: str
    task_type: TaskType
    capabilities: Mapping[str, Any]
    preprocessing_profile: str
    default_params: Mapping[str, Any] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    source: EvidenceSource = EvidenceSource.MODEL_CAPABILITIES

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "backend": self.backend,
            "task_type": self.task_type.value,
            "capabilities": canonical(dict(self.capabilities), path="model.capabilities"),
            "preprocessing_profile": self.preprocessing_profile,
            "default_params": canonical(dict(self.default_params), path="model.params"),
            "aliases": list(self.aliases),
            "source": self.source.value,
        }


@dataclass(frozen=True, slots=True)
class TargetEvidence:
    """What was concluded about the target, if one was named."""

    name: LabelRef
    task_type: TaskType
    n_classes: int | None = None
    is_binary: bool | None = None
    class_labels: tuple[str, ...] = ()
    class_counts: Mapping[str, int] = field(default_factory=dict)
    imbalance_ratio: float | None = None
    positive_label: str | None = None
    positive_label_resolved: bool = False
    missing_count: int = 0
    detection_note: str | None = None
    source: EvidenceSource = EvidenceSource.TASK_DETECTION

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.to_dict(),
            "task_type": self.task_type.value,
            "n_classes": self.n_classes,
            "is_binary": self.is_binary,
            "class_labels": list(self.class_labels),
            "class_counts": canonical(dict(self.class_counts), path="target.class_counts"),
            "imbalance_ratio": self.imbalance_ratio,
            "positive_label": self.positive_label,
            "positive_label_resolved": self.positive_label_resolved,
            "missing_count": self.missing_count,
            "detection_note": self.detection_note,
            "source": self.source.value,
        }


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    """Which dataset this artifact is about, without holding any of it."""

    fingerprint: str
    schema_fingerprint: str
    algorithm: str
    row_count: int
    column_count: int
    duplicate_row_count: int
    total_missing_count: int
    columns: tuple[LabelRef, ...] = ()
    name: str | None = None

    def to_dict(self, *, include_name: bool = True) -> dict[str, Any]:
        """Return the identity.

        Args:
            include_name: Whether the caller-supplied label is included. It is
                excluded from the semantic view: renaming ``train.csv`` to
                ``train_copy.csv`` changes nothing about what was found, and an
                evidence fingerprint that moved when a file was renamed would
                report a difference where there is none.
        """
        payload = {
            "fingerprint": self.fingerprint,
            "schema_fingerprint": self.schema_fingerprint,
            "algorithm": self.algorithm,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "duplicate_row_count": self.duplicate_row_count,
            "total_missing_count": self.total_missing_count,
            "columns": [column.to_dict() for column in self.columns],
        }
        if include_name:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True, slots=True)
class SourceSelectorEvidence:
    """The relation chosen inside a source file.

    ``kind`` is ``table`` for an ordinary SQLite table and ``sheet`` for an
    Excel worksheet. ``name`` is that relation's exact name. This is ``null``
    on every source that is already one table.
    """

    kind: str
    name: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "name": self.name}


@dataclass(frozen=True, slots=True)
class IngestionEvidence:
    """How the audited table was read, as :mod:`aidatasetkit.ingestion` recorded it.

    Present when this library read the source: ``load_table``, or the opt-in
    chunked CSV/TSV scan. An artifact built from a frame the caller loaded some
    other way records ``null`` rather than a reconstruction. Holds no path and
    no cell value.

    ``memory_bytes`` is pandas' deep size of a materialized frame. It is
    ``null`` when the source was profiled in chunks and no frame was built, so
    the figure is not applicable. A materialized table always records an integer.
    """

    source_kind: str
    format: str | None
    encoding: str | None
    delimiter: str | None
    delimiter_source: str | None
    header: bool | None
    row_count: int
    column_count: int
    memory_bytes: int | None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "format": self.format,
            "encoding": self.encoding,
            "delimiter": self.delimiter,
            "delimiter_source": self.delimiter_source,
            "header": self.header,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "memory_bytes": self.memory_bytes,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class ChunkedSamplingEvidence:
    """How quartiles were drawn, and the bound on that draw.

    The sample is capped. ``requested_size`` is that cap. Each column's
    ``actual_size`` lives on its approximation, next to the finite population
    it was drawn from.
    """

    method: str
    seed: int
    requested_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "seed": self.seed,
            "requested_size": self.requested_size,
        }


@dataclass(frozen=True, slots=True)
class ChunkedApproximationEvidence:
    """One quartile of a bounded deterministic sample.

    ``label`` is ``deterministic_approximation``. The number is not an exact
    population quartile and is not a verdict input.
    """

    column: str
    field: str
    method: str
    label: str
    value: float
    seed: int
    requested_size: int
    actual_size: int
    population_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "field": self.field,
            "method": self.method,
            "label": self.label,
            "value": self.value,
            "seed": self.seed,
            "requested_size": self.requested_size,
            "actual_size": self.actual_size,
            "population_size": self.population_size,
        }


@dataclass(frozen=True, slots=True)
class ChunkedColumnEvidence:
    """Exact online measurements for one column. No quartile, no raw value."""

    name: str
    label_type: str
    pandas_dtype: str
    count: int
    missing_count: int
    finite_count: int
    infinite_count: int
    unique_count: int
    minimum: float | None = None
    maximum: float | None = None
    sum: float | None = None
    mean: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label_type": self.label_type,
            "pandas_dtype": self.pandas_dtype,
            "count": self.count,
            "missing_count": self.missing_count,
            "finite_count": self.finite_count,
            "infinite_count": self.infinite_count,
            "unique_count": self.unique_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "sum": self.sum,
            "mean": self.mean,
        }


@dataclass(frozen=True, slots=True)
class ChunkedProfilingEvidence:
    """The opt-in CSV/TSV scan, recorded only when a caller asked for it.

    ``full_population_scanned`` is true only when every validated row was
    visited. Quartiles live in ``approximate_metrics``. Quality, task detection
    and preprocessing are named in ``unavailable_metrics`` and are not invented.
    """

    mode: str
    format: str
    chunk_rows: int
    rows_scanned: int
    population_rows: int
    population_columns: int
    full_population_scanned: bool
    bounded_memory: bool
    exact_metrics: tuple[str, ...]
    approximate_metrics: tuple[ChunkedApproximationEvidence, ...]
    unavailable_metrics: tuple[str, ...]
    sampling: ChunkedSamplingEvidence
    population_fingerprint: str
    fingerprint_algorithm: str
    fingerprint_scope: str
    duplicate_row_count: int
    columns: tuple[ChunkedColumnEvidence, ...]
    temporary_storage: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "format": self.format,
            "chunk_rows": self.chunk_rows,
            "rows_scanned": self.rows_scanned,
            "population_rows": self.population_rows,
            "population_columns": self.population_columns,
            "full_population_scanned": self.full_population_scanned,
            "bounded_memory": self.bounded_memory,
            "exact_metrics": list(self.exact_metrics),
            "approximate_metrics": [item.to_dict() for item in self.approximate_metrics],
            "unavailable_metrics": list(self.unavailable_metrics),
            "sampling": self.sampling.to_dict(),
            "population_fingerprint": self.population_fingerprint,
            "fingerprint_algorithm": self.fingerprint_algorithm,
            "fingerprint_scope": self.fingerprint_scope,
            "duplicate_row_count": self.duplicate_row_count,
            "columns": [column.to_dict() for column in self.columns],
            "temporary_storage": self.temporary_storage,
        }


@dataclass(frozen=True, slots=True)
class ConfigIdentity:
    """Which settings steered the run.

    ``settings`` is the canonicalised material the fingerprint was taken over, so
    a reader can see what was included rather than trusting that the right things
    were.
    """

    fingerprint: str
    settings: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "settings": canonical(dict(self.settings), path="config.settings"),
        }


@dataclass(frozen=True, slots=True)
class DateColumnInference:
    """One textual column recognized as strict ISO date or datetime text.

    Counts only. The column's values are not stored.
    """

    column: str
    inferred_semantic_type: str
    format_family: str
    non_blank_count: int
    blank_count: int
    matched_count: int
    invalid_count: int
    timezone_policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "inferred_semantic_type": self.inferred_semantic_type,
            "format_family": self.format_family,
            "non_blank_count": self.non_blank_count,
            "blank_count": self.blank_count,
            "matched_count": self.matched_count,
            "invalid_count": self.invalid_count,
            "timezone_policy": self.timezone_policy,
        }


@dataclass(frozen=True, slots=True)
class DateInferenceEvidence:
    """Strict ISO recognition for one full audit, or absent when it did not run."""

    method: str
    population: str
    full_population_scanned: bool
    minimum_non_blank: int
    required_match_ratio: float
    conversion_performed: bool
    affects_detected_kind: bool
    affects_verdict: bool
    columns: tuple[DateColumnInference, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "population": self.population,
            "full_population_scanned": self.full_population_scanned,
            "minimum_non_blank": self.minimum_non_blank,
            "required_match_ratio": self.required_match_ratio,
            "conversion_performed": self.conversion_performed,
            "affects_detected_kind": self.affects_detected_kind,
            "affects_verdict": self.affects_verdict,
            "columns": [column.to_dict() for column in self.columns],
        }


@dataclass(frozen=True, slots=True)
class AuditArtifact:
    """One run's evidence, in one object.

    Everything here was decided elsewhere. The artifact's own contribution is
    identity, ordering, and the guarantee that what it holds can be written out
    and read back without losing meaning.

    ``created_at`` is metadata and is excluded from
    :attr:`semantic_fingerprint` on purpose: two audits of the same data under
    the same settings are the same audit, and a clock should not be able to say
    otherwise.
    """

    schema_version: str
    stage: AuditStage
    verdict: Verdict
    dataset: DatasetIdentity
    config: ConfigIdentity
    environment: EnvironmentVersions
    created_at: str
    columns: tuple[ColumnEvidence, ...] = ()
    findings: tuple[FindingEvidence, ...] = ()
    decisions: tuple[DecisionEvidence, ...] = ()
    lineage: tuple[FeatureLineage, ...] = ()
    target: TargetEvidence | None = None
    model: ModelEvidence | None = None
    plan_fingerprint: str | None = None
    warnings: tuple[str, ...] = ()
    known_limitations: tuple[str, ...] = ()
    verdict_reasons: tuple[str, ...] = ()
    ingestion: IngestionEvidence | None = None
    chunked_profiling: ChunkedProfilingEvidence | None = None
    source_selector: SourceSelectorEvidence | None = None
    date_inference: DateInferenceEvidence | None = None

    # ---------------------------------------------------------------- #
    # Serialisation
    # ---------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical mapping. This is the artifact's real format."""
        payload = self.semantic_dict()
        # The label is metadata, so it travels with the timestamp and the
        # environment rather than with the evidence.
        payload["dataset"] = self.dataset.to_dict(include_name=True)
        payload["provenance"] = {
            "created_at": self.created_at,
            "environment": self.environment.to_dict(),
            "semantic_fingerprint": self.semantic_fingerprint,
        }
        return payload

    def semantic_dict(self) -> dict[str, Any]:
        """Return the artifact without anything that varies between identical runs.

        The clock and the recorded environment are not here. What remains is the
        evidence itself, which is what two artifacts should be compared on -- and
        it is also what :attr:`semantic_fingerprint` is taken over, so the
        fingerprint cannot depend on when the run happened.
        """
        return {
            "schema_version": self.schema_version,
            "stage": self.stage.value,
            "verdict": self.verdict.value,
            "verdict_reasons": list(self.verdict_reasons),
            "dataset": self.dataset.to_dict(include_name=False),
            "ingestion": self.ingestion.to_dict() if self.ingestion else None,
            "chunked_profiling": (
                canonical(self.chunked_profiling.to_dict(), path="chunked_profiling")
                if self.chunked_profiling is not None
                else None
            ),
            "source_selector": (
                self.source_selector.to_dict() if self.source_selector is not None else None
            ),
            "date_inference": (
                self.date_inference.to_dict() if self.date_inference is not None else None
            ),
            "config": self.config.to_dict(),
            "target": self.target.to_dict() if self.target else None,
            "model": self.model.to_dict() if self.model else None,
            "plan_fingerprint": self.plan_fingerprint,
            "columns": [column.to_dict() for column in self.columns],
            "findings": [finding.to_dict() for finding in self.findings],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "lineage": [entry.to_dict() for entry in self.lineage],
            "warnings": list(self.warnings),
            "known_limitations": list(self.known_limitations),
        }

    @property
    def semantic_fingerprint(self) -> str:
        """A digest of the evidence, ignoring when and where it was produced."""
        from aidatasetkit.evidence.fingerprint import config_fingerprint

        return config_fingerprint(self.semantic_dict())

    def lineage_dict(self) -> dict[str, Any]:
        """Return the standalone lineage artifact."""
        return {
            "schema_version": self.schema_version,
            "dataset_fingerprint": self.dataset.fingerprint,
            "stage": self.stage.value,
            "features": [entry.to_dict() for entry in self.lineage],
        }

    # ---------------------------------------------------------------- #
    # Reading
    # ---------------------------------------------------------------- #

    @property
    def findings_by_severity(self) -> dict[str, int]:
        """How many findings of each severity, for a summary line."""
        counts = {severity.value: 0 for severity in Severity}
        for finding in self.findings:
            counts[finding.severity.value] += 1
        return counts

    @property
    def review_items(self) -> tuple[str, ...]:
        """Everything a person is being asked to decide."""
        items = [
            f"{finding.code}"
            + (f" ({finding.column.name})" if finding.column else "")
            for finding in self.findings
            if finding.requires_review
        ]
        items += [
            f"{decision.reason_code} ({decision.feature.name})"
            for decision in self.decisions
            if decision.requires_review
        ]
        return tuple(dict.fromkeys(items))

    def differs_from(self, other: AuditArtifact) -> dict[str, bool]:
        """Compare the identities that decide whether two runs are comparable."""
        return {
            "dataset": self.dataset.fingerprint != other.dataset.fingerprint,
            "schema": self.dataset.schema_fingerprint != other.dataset.schema_fingerprint,
            "config": self.config.fingerprint != other.config.fingerprint,
            "plan": self.plan_fingerprint != other.plan_fingerprint,
            "evidence": self.semantic_fingerprint != other.semantic_fingerprint,
        }


def _sequence(values: Sequence[Any] | None) -> tuple[Any, ...]:
    """Normalise an optional sequence to a tuple."""
    return tuple(values) if values else ()
