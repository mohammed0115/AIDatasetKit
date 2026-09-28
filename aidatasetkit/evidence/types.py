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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

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
    "IngestionEvidence",
    "LabelRef",
    "ModelEvidence",
    "TargetEvidence",
    "Verdict",
]

#: Version of the artifact *format*, not of the library. Bumped only when the
#: shape or meaning of a recorded field changes, so a stored artifact can always
#: be read against the contract it was written against.
ARTIFACT_SCHEMA_VERSION = "1.0"


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
    """

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
            "source": self.source.value,
        }


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
class IngestionEvidence:
    """How the audited table was read, as :mod:`aidatasetkit.ingestion` recorded it.

    Present only when the table came through ``load_table``; an artifact built
    from a frame the caller loaded some other way records ``null`` rather than a
    reconstruction. Holds no path and no cell value.
    """

    source_kind: str
    format: str | None
    encoding: str | None
    delimiter: str | None
    delimiter_source: str | None
    header: bool | None
    row_count: int
    column_count: int
    memory_bytes: int
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
