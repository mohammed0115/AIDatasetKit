# Limitations

**This page is the canonical list.** Every limitation the library states anywhere
— in an artifact's `known_limitations`, in the README, in the release notes —
appears here, and a test asserts that the artifact's list is a subset of this
page. If two documents ever disagree, this one is right.

## Scope

**Only tabular supervised classification has been verified end to end.**
Regression, clustering, anomaly detection, dimensionality reduction, time series,
text, and images are out of scope for this alpha. Some of the machinery would
appear to work on them; none of it has been verified for them.

**CSV only, from the command line.** The Python API accepts any pandas
DataFrame. The CLI reads CSV and nothing else — no Parquet, Excel, databases, or
cloud storage.

## Detection

**Leakage detection is statistical.** A feature that encodes the outcome for
reasons the numbers do not show will not be found. See
[safety-model.md](safety-model.md#leakage-what-is-and-is-not-detectable).

**Identifier detection is a heuristic.** A column of distinct values with an
identifier-shaped name is *reported*, never dropped. A genuine measurement that
happens to be unique per row looks the same.

**Datetime columns are profiled but never engineered.** No day-of-week, no
month, no elapsed time. Calendar features are a modelling decision.

## Numbers

**Infinity is not supported as a trainable value.** A column containing one is
held back and the run is blocked, with the reason recorded.

**Values beyond the float32 range are not rejected.** Several scikit-learn
estimators cast input to float32 internally, where a finite value above
approximately 3.4e38 becomes infinite and the estimator raises. Profiling does
not currently flag this.

**Very small feature ranges can be discarded by tree models.** scikit-learn
treats a feature as constant when its spread falls below an absolute threshold of
1e-7 after a float32 cast. A quantity recorded in units so small that the whole
column spans less than a ten-millionth is silently dropped by the tree family.
Standardising rescues it.

## What the artifact records

**A preprocessing decision is a proposal, not a receipt.** The artifact records
what *would* be done to each column and why. It does not prove that any of it was
executed, nor that a model was trained on the data described.

**Configuration you supplied is recorded verbatim.** Ordinal orders and explicit
mappings are your values, and redaction does not touch them — hiding them would
stop the artifact showing which ordering was applied, and would make two
different orderings indistinguishable to the config fingerprint. If those
category names are sensitive, the artifact is sensitive. See
[privacy.md](privacy.md).

## The artifact

**An audit does not prove a model was trained on the data it describes.** It
records what preprocessing *would* do. Connecting an artifact to a trained model
is future work.

**A dataset fingerprint is comparable within one environment.** It is built on
pandas' row hasher, so a different pandas major version can change the digest for
identical data. Every artifact records the versions that produced it.

**A dataset fingerprint identifies content, not provenance.** It cannot show
where the data came from, who collected it, or whether that was lawful.

**Row and column order are part of dataset identity.** Sorting a file changes its
fingerprint. This is deliberate — see
[audit-artifact.md](audit-artifact.md#identity-and-diffing) — but it will
surprise anyone expecting set semantics.

**Schema compatibility is not promised yet.** Artifacts carry a schema version so
that future migration is possible. No migration path exists today.

## Not implemented

Model training, evaluation, metrics, model comparison, hyperparameter tuning,
cross-validation orchestration, feature importance, SHAP, resampling, automatic
class weighting, model selection of any kind. Several of these are deliberate
product boundaries rather than missing work.
