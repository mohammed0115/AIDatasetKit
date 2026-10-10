"""Assembling an audit artifact from facts other layers already established.

This module measures nothing. It receives a
:class:`~aidatasetkit.core.types.DatasetProfile`, a
:class:`~aidatasetkit.core.types.QualityReport`, a
:class:`~aidatasetkit.preprocessing.PreprocessingPlan`, and whatever else the
caller produced, and it records what they say. A missing ratio in an artifact is
the ratio the profiler computed; recomputing it here would create a second
number that could disagree with the first, and the disagreement would be
invisible.

**Privacy is a default, not an option.** One field on the profiling side holds a
raw dataset value: ``ColumnProfile.dominant_value``, the most frequent entry in a
column. For an email column that is somebody's address; for a free-text column it
is a sentence out of the data. It is replaced by a plain SHA-256 digest unless
the caller explicitly asks otherwise, so an artifact can still answer "is the
most common value the same as last week" without carrying the value itself.

The digest is **not salted and not a privacy guarantee**. A value drawn from a
small or guessable domain -- a boolean, a country code, a category from a known
list -- can be recovered by hashing candidates. It is an identifier for a value,
not a way of hiding one, and ``docs/privacy.md`` says so in those words.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import logging
import re
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import pandas as pd

from aidatasetkit.core.config import KitConfig
from aidatasetkit.core.exceptions import EvidenceError
from aidatasetkit.core.provenance import EnvironmentVersions, capture_environment
from aidatasetkit.core.types import (
    DatasetProfile,
    QualityReport,
    TargetProfile,
    TaskType,
)
from aidatasetkit.evidence.serialization import canonical
from aidatasetkit.evidence.fingerprint import (
    ALGORITHM,
    config_fingerprint,
    dataset_fingerprint,
    schema_fingerprint,
)
from aidatasetkit.evidence.policy import decide_verdict
from aidatasetkit.ingestion.types import LoadMetadata
from aidatasetkit.evidence.types import (
    ARTIFACT_SCHEMA_VERSION,
    AuditArtifact,
    AuditStage,
    ColumnEvidence,
    ConfigIdentity,
    DatasetIdentity,
    DecisionEvidence,
    EvidenceSource,
    FeatureLineage,
    FindingEvidence,
    FitScope,
    ChunkedApproximationEvidence,
    ChunkedColumnEvidence,
    ChunkedProfilingEvidence,
    ChunkedSamplingEvidence,
    IngestionEvidence,
    LabelRef,
    ModelEvidence,
    TargetEvidence,
    Verdict,
)

__all__ = ["AuditBuilder", "CHUNKED_AUDIT_UNAVAILABLE", "KNOWN_LIMITATIONS"]

#: Returned by the chunked route. The full quality, task and preprocessing audit
#: did not run, so the artifact must not claim a readiness verdict.
CHUNKED_AUDIT_UNAVAILABLE = (
    "Full-table audit verdict is unavailable in chunked profiling mode."
)

_logger = logging.getLogger(__name__)

#: Stated on every artifact. These are the boundaries of what the checks in this
#: library can establish, written plainly so a reader is not left to infer them
#: from silence. Deliberately phrased as limits rather than as reassurance.
KNOWN_LIMITATIONS: tuple[str, ...] = (
    "Leakage detection is statistical. A feature that encodes the outcome for "
    "reasons the numbers do not show will not be found here.",
    # Narrowed twice, for the same reason each time: an artifact travels further
    # than any other document this library produces, so it must not carry a scope
    # statement the library has outgrown. S6 removed regression from the
    # out-of-scope list when nine regressors were held to the same executed
    # capability contracts as the classifiers; S9 removed clustering when six
    # clusterers were. What has *not* moved either time is the verdict, which is
    # still the one thing verified end to end for classification alone.
    "Only the classification readiness verdict has been verified end to end. "
    "Regression and clustering models and their preprocessing are verified; "
    "anomaly detection, dimensionality reduction, time series, text, and images "
    "remain out of scope.",
    "Datetime columns are profiled but never turned into features automatically; "
    "any calendar engineering is the analyst's decision.",
    "Preprocessing decisions are proposals recorded as evidence. This artifact "
    "does not prove a model was trained on the data it describes.",
    "A dataset fingerprint identifies content, not provenance. It cannot show "
    "where the data came from or whether it was collected lawfully.",
    "Numeric values beyond the float32 range are unsupported by several "
    "estimators and are not rejected at profiling time.",
    "Values the analyst supplied as configuration -- ordinal orders, explicit "
    "mappings -- are recorded verbatim, because an audit that hid them could not "
    "show which ordering was applied. Redaction covers values read from the "
    "data, not values handed to the tool.",
)

#: Detail keys whose *text* is a name rather than data, and may be kept.
#:
#: This is an allowlist, and the direction matters. An earlier version listed the
#: keys known to hold values and redacted those, which is safe only for the
#: checks that existed when the list was written: ``possible_numeric_stored_as_text``
#: reports ``non_numeric_examples``, a list of raw cells, and walked straight
#: past it into ``audit.json``. Denying by default means a check added tomorrow
#: is private until somebody deliberately says otherwise.
#:
#: Numbers and booleans in ``details`` are measurements and are always kept; only
#: text is suspect, because only text carries a value somebody could read.
#: Each entry is vocabulary this library defines, not text taken from a dataset:
#: a column name, a detected kind, or the name of a signal that fired. Redacting
#: them makes a finding unreadable without protecting anybody -- an early version
#: hashed ``signals`` and turned "deterministic_mapping" into a digest.
#: ``tests/unit/test_evidence_artifact.py`` pins the full set of text keys the
#: checks actually emit, so this list cannot quietly go stale.
_SAFE_TEXT_DETAIL_KEYS: frozenset[str] = frozenset(
    {
        "target",
        "column",
        "other_column",
        "compared_with",
        "other_numeric_features",
        "kind",
        "detected_kind",
        "dtype",
        "code",
        "signals",
    }
)


#: Steps that take something from the data they are fitted on.
#:
#: The distinction is the whole point of :class:`FitScope`, so it is drawn per
#: step rather than assumed for all of them. A median, a mean, a most-frequent
#: category, a scaler's centre and spread, and a one-hot vocabulary are all read
#: off the fitting rows. So is the categorical sentinel, less obviously: the
#: planner picks a filler that collides with no category *present in the data*,
#: which makes it a function of that data.
_LEARNED_STEPS: frozenset[str] = frozenset(
    {
        "median_imputation",
        "mean_imputation",
        "most_frequent_imputation",
        "sentinel_imputation",
        "standard_scaling",
        "minmax_scaling",
        "robust_scaling",
        "onehot_encoding",
    }
)

#: Steps that learn nothing, listed so that "no_fit" is a checked claim rather
#: than the absence of a match.
#:
#: An ordinal encoding uses the order the analyst supplied, not one inferred from
#: the column; an explicit mapping is the analyst's table; a constant fill is a
#: configured number; and parsing text to a float is a per-row function. None of
#: them could carry information from one row to another, which is what makes them
#: uninteresting for leakage and worth saying so explicitly.
_STATELESS_STEPS: frozenset[str] = frozenset(
    {
        "constant_imputation",
        "ordinal_encoding",
        "explicit_mapping",
        "numeric_text_conversion",
    }
)


class AuditBuilder:
    """Collects established facts and emits one :class:`AuditArtifact`.

    Args:
        redact_values: Whether raw dataset values are kept out of the artifact.
            True by default. Setting it False makes the most frequent value of
            each column appear in the output, which is occasionally what an
            analyst auditing their own non-sensitive data wants and is never a
            safe default.
        dataset_name: A label for the artifact, such as a file name. Recorded as
            given; it is metadata, not identity.

    Example:
        >>> builder = AuditBuilder(dataset_name="train.csv")           # doctest: +SKIP
        >>> artifact = builder.build(frame, profile=profile, quality=report)
    """

    def __init__(
        self, *, redact_values: bool = True, dataset_name: str | None = None
    ) -> None:
        self._redact = redact_values
        self._dataset_name = dataset_name

    # ------------------------------------------------------------------ #
    # Building
    # ------------------------------------------------------------------ #

    def build(
        self,
        frame: pd.DataFrame,
        *,
        profile: DatasetProfile,
        quality: QualityReport | None = None,
        target: TargetProfile | None = None,
        plan: Any | None = None,
        lineage: Mapping[Hashable, Sequence[str]] | None = None,
        model: Any | None = None,
        kit_config: KitConfig | None = None,
        settings: Mapping[str, Any] | None = None,
        warnings: Sequence[str] = (),
        blocked_reason: str | None = None,
        environment: EnvironmentVersions | None = None,
        created_at: str | None = None,
        ingestion: LoadMetadata | None = None,
    ) -> AuditArtifact:
        """Assemble the artifact.

        Args:
            frame: The audited frame. Read for identity only -- never modified,
                and none of its values reach the artifact.
            profile: What :class:`~aidatasetkit.profiling.DataProfiler` measured.
            quality: What the quality inspector found.
            target: What the task detector concluded, if a target was named.
            plan: A ``PreprocessingPlan``, when one was produced.
            lineage: The preprocessing layer's own ``lineage()`` output, when a
                preprocessor was fitted. Never re-derived from output names.
            model: A ``ModelRegistration`` giving the model context, if any.
            kit_config: The thresholds the profiler and the quality inspector ran
                under. Every finding in this artifact depends on them -- change
                ``high_cardinality_threshold`` and a column starts or stops being
                reported -- so without it the config fingerprint would answer
                "were the same settings in force?" with a confident yes when it
                could not know. Left out, the artifact records that it was not
                supplied rather than assuming the defaults.
            settings: Everything else that steered the run.
            warnings: Non-fatal problems encountered while producing the run.
            blocked_reason: Set when the analysis could not complete, which makes
                the verdict ``BLOCKED`` while still producing an artifact.
            environment: Captured versions. Defaults to the current environment.
            created_at: ISO timestamp. Defaults to now, in UTC.

        Returns:
            A complete :class:`AuditArtifact`.
        """
        stage = self._stage(plan, lineage)
        columns = tuple(self._column_evidence(column) for column in profile.column_profiles)
        issues = quality.issues if quality is not None else ()
        findings = tuple(self._finding_evidence(issue) for issue in issues)
        decisions = tuple(self._decision_evidence(plan, model)) if plan is not None else ()
        lineage_entries = tuple(self._lineage_evidence(plan, lineage))
        verdict, reasons = decide_verdict(
            findings, decisions, blocked_reason=blocked_reason
        )

        identity = DatasetIdentity(
            fingerprint=dataset_fingerprint(frame),
            schema_fingerprint=schema_fingerprint(frame),
            algorithm=ALGORITHM,
            row_count=profile.row_count,
            column_count=profile.column_count,
            duplicate_row_count=profile.duplicate_row_count,
            total_missing_count=profile.total_missing_count,
            columns=tuple(LabelRef.of(label) for label in frame.columns),
            name=self._dataset_name,
        )
        resolved_settings = self._settings(settings, plan, model, target, kit_config)

        return AuditArtifact(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            stage=stage,
            verdict=verdict,
            verdict_reasons=reasons,
            dataset=identity,
            config=ConfigIdentity(
                fingerprint=config_fingerprint(resolved_settings),
                settings=resolved_settings,
            ),
            environment=environment or capture_environment(),
            created_at=created_at or _now(),
            columns=columns,
            findings=findings,
            decisions=decisions,
            lineage=lineage_entries,
            target=self._target_evidence(target),
            model=self._model_evidence(model),
            plan_fingerprint=getattr(plan, "fingerprint", None),
            warnings=tuple(warnings),
            known_limitations=KNOWN_LIMITATIONS,
            ingestion=self._ingestion_evidence(ingestion, frame),
            chunked_profiling=None,
        )

    def build_chunked(
        self,
        profile: Any,
        *,
        dataset_name: str = "dataset",
        settings: Mapping[str, Any] | None = None,
        environment: EnvironmentVersions | None = None,
        created_at: str | None = None,
    ) -> AuditArtifact:
        """Publish a profile-only artifact for an opt-in chunked scan.

        The frame is not an argument: this route has no table to audit. The
        verdict is blocked because quality, task detection and preprocessing
        were not run. Approximate quartiles stay inside ``chunked_profiling``
        and are not passed to :func:`decide_verdict`.
        """
        evidence = _chunked_evidence(profile)
        # Approximate values are not verdict inputs.
        verdict, reasons = decide_verdict((), (), blocked_reason=CHUNKED_AUDIT_UNAVAILABLE)
        identity = DatasetIdentity(
            name=dataset_name,
            fingerprint=profile.population_fingerprint,
            schema_fingerprint=profile.schema_fingerprint,
            algorithm=ALGORITHM,
            row_count=profile.population_rows,
            column_count=profile.population_columns,
            duplicate_row_count=profile.duplicate_row_count,
            total_missing_count=sum(column.missing_count for column in profile.columns),
            columns=tuple(LabelRef.of(column.label) for column in profile.columns),
        )
        # The scan contract is the identity. Paths, clocks and caller extras stay out.
        del settings
        resolved_settings = canonical(_chunked_settings(profile))
        stage = AuditStage.PROFILED
        return AuditArtifact(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            stage=stage,
            verdict=verdict,
            verdict_reasons=reasons,
            dataset=identity,
            config=ConfigIdentity(
                fingerprint=config_fingerprint(resolved_settings),
                settings=resolved_settings,
            ),
            environment=environment or capture_environment(),
            created_at=created_at or _now(),
            columns=(),
            findings=(),
            decisions=(),
            lineage=(),
            target=None,
            model=None,
            plan_fingerprint=None,
            warnings=(),
            known_limitations=KNOWN_LIMITATIONS,
            ingestion=_chunked_ingestion(profile),
            chunked_profiling=evidence,
        )

    @staticmethod
    def _ingestion_evidence(
        metadata: LoadMetadata | None, frame: pd.DataFrame
    ) -> IngestionEvidence | None:
        """Record how the table was read, refusing metadata that describes another table.

        ``None`` stays ``None``: an artifact built from a frame loaded some other
        way says it has no ingestion record rather than inventing one.
        """
        if metadata is None:
            return None
        if metadata.row_count != len(frame) or metadata.column_count != frame.shape[1]:
            raise EvidenceError(
                f"The ingestion metadata describes {metadata.row_count} rows and "
                f"{metadata.column_count} columns, but the audited frame has "
                f"{len(frame)} and {frame.shape[1]}. An artifact cannot record how one "
                "table was read beside the evidence about another."
            )
        return IngestionEvidence(**{
            key: tuple(value) if key == "warnings" else value
            for key, value in metadata.to_dict().items()
        })

    # ------------------------------------------------------------------ #
    # Per-section recording
    # ------------------------------------------------------------------ #

    @staticmethod
    def _stage(plan: Any | None, lineage: Mapping[Any, Any] | None) -> AuditStage:
        if lineage is not None:
            return AuditStage.PREPARED
        if plan is not None:
            return AuditStage.PLANNED
        return AuditStage.INSPECTED

    def _column_evidence(self, column: Any) -> ColumnEvidence:
        """Record one column's measurements, with its most frequent value hidden."""
        numeric = column.numeric.to_dict() if column.numeric is not None else None
        return ColumnEvidence(
            name=LabelRef.of(column.name),
            detected_kind=column.detected_kind.value,
            pandas_dtype=str(column.pandas_dtype),
            count=column.count,
            missing_count=column.missing_count,
            missing_ratio=column.missing_ratio,
            unique_count=column.unique_count,
            unique_ratio=column.unique_ratio,
            is_constant=column.is_constant,
            is_near_constant=column.is_near_constant,
            is_high_cardinality=column.is_high_cardinality,
            is_id_like=column.is_id_like,
            infinite_count=column.infinite_count,
            dominant_ratio=column.dominant_ratio,
            dominant_value_digest=self._dominant(column.dominant_value),
            numeric=numeric,
        )

    def _dominant(self, value: Any) -> str | None:
        """Represent the most frequent value without reproducing it.

        A digest still answers the questions an audit asks of this field -- has
        the most common value changed since the last run, do two columns share
        one -- while an email address, a name, or a free-text note never leaves
        the machine that held the data.
        """
        if value is None:
            return None
        if not self._redact:
            return str(value)
        digest = hashlib.sha256(f"{type(value).__name__}:{value}".encode()).hexdigest()
        return f"sha256:{digest[:32]}"

    def _finding_evidence(self, issue: Any) -> FindingEvidence:
        """Record a finding, taking any raw value out of it first.

        A finding's message is prose written for the analyst who owns the data,
        and two of them quote the value they are about. The artifact is the
        shareable object, so the value is replaced there -- in the details *and*
        in the message, since a digest in one and the address in the other would
        be no protection at all.
        """
        details = dict(issue.details or {})
        message = issue.message
        if self._redact:
            for key in sorted(details):
                if key in _SAFE_TEXT_DETAIL_KEYS:
                    continue
                # A key that names a value hides one whatever its type; every
                # other key hides one only when it is text.
                redacted, removed = self._redact_detail(
                    details[key], numbers_too=_names_a_data_value(key)
                )
                if not removed:
                    continue
                details[key] = redacted
                for raw, digest in removed:
                    message = _scrub(message, raw, digest)
        return FindingEvidence(
            code=issue.code,
            severity=issue.severity,
            message=message,
            column=LabelRef.of(issue.column) if issue.column is not None else None,
            recommendation=issue.recommendation,
            requires_review=issue.requires_review,
            details=details,
        )

    def _decision_evidence(self, plan: Any, model: Any | None):
        """Record every planned decision, naming the capability behind each step."""
        for decision in plan.decisions:
            spec = plan.spec_for(decision.feature)
            yield DecisionEvidence(
                feature=LabelRef.of(decision.feature),
                role=decision.role.value,
                action=decision.action.value,
                reason_code=decision.reason_code,
                reason=decision.reason,
                steps=tuple(decision.steps),
                requires_review=decision.requires_review,
                input_kind=spec.column_kind.value if spec is not None else None,
                fit_scope=self._fit_scope(decision),
                model_requirement=self._model_requirement(decision, plan, model),
                details=dict(decision.details or {}),
            )

    def _redact_detail(
        self, value: Any, *, numbers_too: bool = False
    ) -> tuple[Any, list[tuple[Any, str]]]:
        """Replace a value in a detail with a digest, recursively.

        Returns the redacted value and every (original, digest) pair replaced, so
        the finding's message can have the same values taken out of it.

        Args:
            value: The detail value.
            numbers_too: Whether numbers are values rather than measurements. A
                count of 240 describes a column; a most-frequent salary of
                987654.0 *is* somebody's salary, and the two are both floats.
        """
        if isinstance(value, str) or (
            numbers_too and isinstance(value, (int, float)) and not isinstance(value, bool)
        ):
            digest = self._dominant(value)
            return digest, [(value, digest)]
        if isinstance(value, (list, tuple)):
            out, removed = [], []
            for item in value:
                redacted, pairs = self._redact_detail(item, numbers_too=numbers_too)
                out.append(redacted)
                removed.extend(pairs)
            return out, removed
        if isinstance(value, Mapping):
            out, removed = {}, []
            for key, item in value.items():
                redacted, pairs = self._redact_detail(
                    item, numbers_too=numbers_too or _names_a_data_value(str(key))
                )
                out[key] = redacted
                removed.extend(pairs)
            return out, removed
        return value, []

    @staticmethod
    def _fit_scope(decision: Any) -> FitScope:
        """What the recorded steps learned, and from where.

        Every transformer the preprocessing layer builds is fitted on the frame
        it is given and nothing else, so a step that learns anything learns it
        from the training rows. A step list with no learned step is a pure
        per-row function, and a feature that is not included never reaches one.
        """
        if decision.action.value != "include":
            return FitScope.NOT_APPLICABLE
        if any(step in _LEARNED_STEPS for step in decision.steps):
            return FitScope.TRAINING_ONLY
        unknown = set(decision.steps) - _STATELESS_STEPS
        if unknown:
            # A step this module has not classified. Reporting "nothing was
            # learned" would be a claim about leakage made from ignorance, so the
            # artifact says the safer thing and the gap is logged.
            _logger.warning(
                "unclassified preprocessing step(s) %s; fit scope reported as "
                "training_only because a stronger claim cannot be justified",
                sorted(unknown),
            )
            return FitScope.TRAINING_ONLY
        return FitScope.NO_FIT

    @staticmethod
    def _model_requirement(decision: Any, plan: Any, model: Any | None) -> str | None:
        """Name the declared capability that put a step in this decision.

        Bounded on purpose. It states which capability the profile declares and
        which step follows from it, and claims nothing about why the algorithm
        needs it -- that belongs to the model's own documentation, not to an
        audit of one dataset.
        """
        profile = getattr(plan, "preprocessing_profile", None)
        if profile is None or decision.action.value != "include":
            return None
        named = f"model {model.canonical_name!r} " if model is not None else "the selected model "
        steps = set(decision.steps)

        if profile.requires_scaling and steps & {
            "standard_scaling",
            "minmax_scaling",
            "robust_scaling",
        }:
            return f"{named}declares requires_scaling=true"
        # Only the *numeric* fills follow from the capability. A categorical
        # column is imputed whichever model is chosen -- the steps are identical
        # for a model that takes NaN natively and one that does not -- so naming
        # the capability there would invent a cause the plan never had.
        if not profile.handles_missing_values and steps & {
            "median_imputation",
            "mean_imputation",
            "constant_imputation",
        }:
            return f"{named}declares handles_missing_values=false"
        if profile.handles_missing_values and not steps & {
            "median_imputation",
            "mean_imputation",
            "constant_imputation",
            "most_frequent_imputation",
            "sentinel_imputation",
        }:
            return f"{named}declares handles_missing_values=true, so gaps are kept"
        if not profile.supports_sparse_input and "onehot_encoding" in steps:
            return f"{named}declares supports_sparse_input=false, so the output is dense"
        return None

    def _lineage_evidence(self, plan: Any | None, lineage: Mapping[Any, Any] | None):
        """Record where each input column went.

        When a fitted preprocessor is available its own ``lineage()`` is the
        authority and the outputs are observed. Otherwise the plan's steps are
        recorded as intended rather than observed, and ``observed`` says which.
        """
        if plan is None:
            return
        observed = lineage is not None
        produced = {str(key): tuple(value) for key, value in (lineage or {}).items()}
        for decision in plan.decisions:
            spec = plan.spec_for(decision.feature)
            yield FeatureLineage(
                source=LabelRef.of(decision.feature),
                source_kind=decision.role.value,
                action=decision.action.value,
                steps=tuple(decision.steps),
                outputs=produced.get(str(decision.feature), ()),
                fit_scope=self._fit_scope(decision),
                reason=decision.reason,
                observed=observed and str(decision.feature) in produced,
            )

    @staticmethod
    def _target_evidence(target: TargetProfile | None) -> TargetEvidence | None:
        if target is None or target.target_name is None:
            return None
        return TargetEvidence(
            name=LabelRef.of(target.target_name),
            task_type=target.task_type,
            n_classes=target.n_classes,
            is_binary=target.is_binary,
            class_labels=tuple(str(label) for label in (target.classes or ())),
            class_counts={
                str(label): int(count) for label, count in (target.class_counts or {}).items()
            },
            imbalance_ratio=target.imbalance_ratio,
            positive_label=None if target.positive_label is None else str(target.positive_label),
            positive_label_resolved=bool(target.positive_label_resolved),
            missing_count=target.missing_count,
            detection_note=target.detection_note,
        )

    @staticmethod
    def _model_evidence(model: Any | None) -> ModelEvidence | None:
        """Record the model context as metadata. No estimator is constructed."""
        if model is None:
            return None
        capabilities = model.capabilities
        entry = model.to_dict()
        return ModelEvidence(
            canonical_name=model.canonical_name,
            backend=capabilities.backend.value,
            task_type=capabilities.task_type,
            capabilities=capabilities.to_dict(),
            preprocessing_profile=capabilities.preprocessing_profile().key,
            default_params=dict(entry.get("default_params") or {}),
            aliases=tuple(model.aliases),
        )

    def _settings(
        self,
        settings: Mapping[str, Any] | None,
        plan: Any | None,
        model: Any | None,
        target: TargetProfile | None,
        kit_config: KitConfig | None,
    ) -> dict[str, Any]:
        """Gather everything that could have changed the run into one mapping.

        Whatever enters here enters the config fingerprint, so the rule is: if
        changing it would change what the run did, it belongs.
        """
        resolved: dict[str, Any] = dict(settings or {})
        resolved.setdefault("redact_values", self._redact)
        # Recorded as null rather than as the defaults when it was not supplied.
        # Claiming defaults we were never shown would put a wrong answer in the
        # one field whose whole job is to say what was in force.
        resolved.setdefault(
            "thresholds", kit_config.to_dict() if kit_config is not None else None
        )
        if target is not None and target.target_name is not None:
            resolved.setdefault("target", LabelRef.of(target.target_name).token)
            resolved.setdefault("task", target.task_type.value)
        if plan is not None:
            resolved.setdefault("preprocessing", dict(plan.config))
            resolved.setdefault("preprocessing_profile", plan.preprocessing_profile.key)
            resolved.setdefault("categorical_sentinel", plan.categorical_sentinel)
        if model is not None:
            resolved.setdefault("model", model.canonical_name)
        return resolved


def _names_a_data_value(key: str) -> bool:
    """Whether a details key holds a value read out of the dataset.

    Text-versus-number is not the distinction that matters, and assuming it was
    let a salary through: ``near_constant_column`` reports
    ``dominant_value: 987654.0``, a number as identifying as any string, beside
    ``dominant_ratio: 0.9958``, a measurement that must survive. Both are floats.
    Only the key separates them.

    A pattern rather than a list, so a check added later that reports
    ``median_value`` or ``top_examples`` is covered without anyone remembering.
    """
    return key in {"value", "values", "example", "examples", "sample", "samples"} or key.endswith(
        ("_value", "_values", "_example", "_examples", "_sample", "_samples")
    )


def _scrub(message: str, value: Any, replacement: str) -> str:
    """Remove one raw value from a message, without rewriting the rest of it.

    The quoted form is tried first and, if it matched, nothing else is tried. An
    earlier version applied both spellings unconditionally: for a column whose
    dominant value is ``"a"``, the second pass replaced every letter "a" in the
    sentence -- including the ones inside the digest the first pass had just
    inserted -- and turned the message into rubble.

    The bare form is only replaced on word boundaries, so a one-character value
    can no longer eat the prose around it.
    """
    if value is None:
        return message
    quoted = repr(value)
    if quoted and quoted in message:
        return message.replace(quoted, replacement)

    bare = str(value)
    if not bare:
        return message
    return re.sub(rf"(?<!\w){re.escape(bare)}(?!\w)", replacement, message)


def _now() -> str:
    """The current UTC time, to the second.

    Second resolution rather than microsecond: the field is metadata a person
    reads, and spurious precision invites the mistake of treating it as identity.
    """
    return (
        _datetime.datetime.now(_datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _chunked_settings(profile: Any) -> dict[str, Any]:
    """Settings that can change a chunked scan. No path, clock, or environment."""
    return {
        "mode": profile.mode,
        "chunk_rows": profile.chunk_rows,
        "quantile_sample_size": profile.sampling_requested_size,
        "sampling_method": profile.sampling_method,
        "sampling_seed": profile.sampling_seed,
        "encoding": profile.encoding,
        "delimiter": profile.delimiter,
        "delimiter_source": profile.delimiter_source,
        "header": profile.header,
        "limits": dict(profile.effective_limits),
        "fingerprint_algorithm": profile.fingerprint_algorithm,
    }


def _chunked_ingestion(profile: Any) -> IngestionEvidence:
    """How the chunked scan read the file. ``memory_bytes`` is not applicable."""
    return IngestionEvidence(
        source_kind="file",
        format=profile.format,
        encoding=profile.encoding,
        delimiter=profile.delimiter,
        delimiter_source=profile.delimiter_source,
        header=profile.header,
        row_count=profile.population_rows,
        column_count=profile.population_columns,
        memory_bytes=None,
        warnings=tuple(profile.ingestion_warnings),
    )


def _chunked_evidence(profile: Any) -> ChunkedProfilingEvidence:
    """Copy a chunked scan into the immutable artifact record.

    The scan object stays in the profiling layer. The artifact stores this
    copy, and the verdict is chosen before any approximate value is read.
    """
    return ChunkedProfilingEvidence(
        mode=profile.mode,
        format=profile.format,
        chunk_rows=profile.chunk_rows,
        rows_scanned=profile.rows_scanned,
        population_rows=profile.population_rows,
        population_columns=profile.population_columns,
        full_population_scanned=profile.full_population_scanned,
        bounded_memory=profile.bounded_memory,
        exact_metrics=tuple(profile.exact_metrics),
        approximate_metrics=tuple(
            ChunkedApproximationEvidence(
                column=item.column,
                field=item.field,
                method=item.method,
                label=item.label,
                value=item.value,
                seed=item.seed,
                requested_size=item.requested_size,
                actual_size=item.actual_size,
                population_size=item.population_size,
            )
            for item in profile.approximations
        ),
        unavailable_metrics=tuple(profile.unavailable_metrics),
        sampling=ChunkedSamplingEvidence(
            method=profile.sampling_method,
            seed=profile.sampling_seed,
            requested_size=profile.sampling_requested_size,
        ),
        population_fingerprint=profile.population_fingerprint,
        fingerprint_algorithm=profile.fingerprint_algorithm,
        fingerprint_scope=profile.fingerprint_scope,
        duplicate_row_count=profile.duplicate_row_count,
        columns=tuple(
            ChunkedColumnEvidence(
                name=str(column.label),
                label_type=type(column.label).__name__,
                pandas_dtype=column.pandas_dtype,
                count=column.count,
                missing_count=column.missing_count,
                finite_count=column.finite_count,
                infinite_count=column.infinite_count,
                unique_count=column.unique_count,
                minimum=column.minimum,
                maximum=column.maximum,
                sum=column.sum,
                mean=column.mean,
            )
            for column in profile.columns
        ),
        temporary_storage=profile.temporary_storage,
    )
