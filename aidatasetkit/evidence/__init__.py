"""Evidence: what happened between the raw dataset and the model, and why.

Every other package in this library establishes facts. This one connects them
into a single record that can be stored, compared, diffed, and read by a person
who was not there.

=============================  ==================================================
Component                      Answers
=============================  ==================================================
:class:`AuditBuilder`          What did this run establish?
:class:`AuditArtifact`         The record itself, in one object
:func:`dataset_fingerprint`    Is this the same data as last time?
:func:`config_fingerprint`     Were the same settings in force?
:func:`decide_verdict`         Does anything stop a person proceeding?
:func:`render_report`          The same evidence, for a human
=============================  ==================================================

Three properties define the layer.

**It never recomputes.** Evidence receives a profile, a quality report, a plan,
and a lineage map, and records what they say. A second measurement here could
disagree with the first, and nobody would see the disagreement.

**It never carries raw data.** Counts, ratios, dtypes, thresholds, and digests
describe a dataset without reproducing it. The one profiling field that holds a
real value -- the most frequent entry in a column -- is hashed by default.

**JSON is the artifact.** The HTML report is a rendering of the same canonical
mapping, so the two cannot drift apart; a future SARIF or CI renderer is another
reader of the same document rather than another producer of a different one.

Evidence depends on the layers below it and nothing depends on evidence, which
is what lets it be added, changed, or removed without touching a measurement.
"""

from aidatasetkit.evidence.builder import KNOWN_LIMITATIONS, AuditBuilder
from aidatasetkit.evidence.fingerprint import (
    ALGORITHM,
    config_fingerprint,
    dataset_fingerprint,
    schema_fingerprint,
)
from aidatasetkit.evidence.policy import VERDICT_ORDER, decide_verdict, verdict_at_least
from aidatasetkit.evidence.report import render_report
from aidatasetkit.evidence.serialization import canonical, canonical_json, label_token
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
    IngestionEvidence,
    LabelRef,
    ModelEvidence,
    TargetEvidence,
    Verdict,
)

from aidatasetkit.evidence.publication import (
    CURRENT_NAME,
    MANIFEST_NAME,
    PUBLICATION_SCHEMA_VERSION,
    PublishedRun,
    publish_run,
    read_current,
)

__all__ = [
    "ALGORITHM",
    "ARTIFACT_SCHEMA_VERSION",
    "AuditArtifact",
    "AuditBuilder",
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
    "CURRENT_NAME",
    "KNOWN_LIMITATIONS",
    "MANIFEST_NAME",
    "PUBLICATION_SCHEMA_VERSION",
    "LabelRef",
    "ModelEvidence",
    "PublishedRun",
    "TargetEvidence",
    "VERDICT_ORDER",
    "Verdict",
    "canonical",
    "canonical_json",
    "config_fingerprint",
    "dataset_fingerprint",
    "decide_verdict",
    "label_token",
    "publish_run",
    "read_current",
    "render_report",
    "schema_fingerprint",
    "verdict_at_least",
]
