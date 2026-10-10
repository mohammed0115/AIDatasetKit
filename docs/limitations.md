# Limitations

**This page is the canonical list.** Every limitation the library states anywhere
— in an artifact's `known_limitations`, in the README, in the release notes —
appears here, and a test asserts that the artifact's list is a subset of this
page. If two documents ever disagree, this one is right.

## Scope

**Only the classification readiness verdict has been verified end to end.**
Anomaly detection, dimensionality reduction, time series, text, and images are
out of scope. Some of the machinery would appear to work on them; none of it has
been verified for them.

**Regression is verified at the model layer, not at the verdict.** The nine
regressors are held to the same executed capability contracts as the classifiers,
and they reach preprocessing through the same capability triple — the same
registry, the same factory, the same plan. What has *not* been verified for a
regression target is the readiness verdict itself: its thresholds, and the
leakage checks behind it, were built and measured against classification targets.
Class-imbalance reporting correctly does not apply to a continuous target; the
association-based leakage signal has not been re-measured for one. An audit of a
regression target records this in its own warnings.

**Clustering is verified at the model layer, and its metrics are in-sample.** The
six clusterers are held to the same executed capability contracts and reach
preprocessing through the same capability triple. Three limitations are specific
to them and none is a defect to be fixed later:

- **The metrics score the rows the model was fitted on.** There is no target for
  a score to be inflated by and the three metrics are internal, so this is not
  the leakage the supervised layers refuse — but it does mean a silhouette
  describes *this* partition and is not evidence that it will reproduce on new
  data.
- **No default ranking metric exists, deliberately.** All three internal metrics
  reward compact convex clusters, and all three were measured preferring a wrong
  partition to a correct one on non-convex data. A comparison reports the three
  and orders by none.
- **Half the catalogue cannot label an unseen row.** `DBSCAN`, `OPTICS` and
  `AgglomerativeClustering` define no rule for a row they were not fitted on, and
  asking them to assign one is refused rather than approximated.

The readiness verdict has not been verified for a clustering run either. Its
thresholds and its leakage checks were built against a target, and a clustering
frame has none.

**Delimited text, JSON, columnar and Excel files only.** `aidatasetkit.ingestion.load_table`
— which the CLI uses — reads `.csv`, `.tsv`, `.json` (an array of flat
objects), `.jsonl`/`.ndjson`, `.parquet`, `.feather`/`.arrow` and `.xlsx`
files, a pandas DataFrame, or a list of records, and nothing else: no old-format
`.xls`, databases, URLs or cloud storage. A `.txt` file is refused even when it
holds CSV. Parquet and Feather need the optional `parquet` extra (pyarrow);
`.xlsx` needs the `excel` extra (openpyxl) and reads one worksheet (the first
row is the header; a multi-sheet workbook needs `sheet=` or the CLI `--sheet`
flag); macro-enabled workbooks are refused. A header with no data rows is
empty input, and a blank data row is kept as missing cells. JSON is read whole (bounded by the source-byte limit),
while JSONL is checked as it streams.

- **Encodings are never guessed.** utf-8 (default), utf-8-sig, latin-1 and
  cp1256 are read when named; anything else — UTF-16 included — is refused.
- **Delimiters** `,` `;` tab `|` are detected from the first 64 KiB, and the
  whole file is then validated. A file that is consistent under two delimiters
  is refused as ambiguous; a row with too few or too many fields is refused, not
  padded.
- **Resource limits are still guards.** By default files are refused
  above 64 MiB, 1,000,000 data rows, 1,000 columns, 10,000,000 data cells or
  1,000,000 characters in one CSV/TSV field. DataFrames and records are checked
  before analysis/materialization as applicable; callers may provide an
  immutable `IngestionLimits` policy. The CLI uses the same finite defaults and
  exposes overrides. The default audit still loads an accepted file as one
  pandas table.
- **Chunked profiling is opt-in and CSV/TSV only.**
  `profile_delimited_chunks` (and `aidatasetkit audit --chunked-profile`) scans
  the whole population in chunks and does not load that population as one
  table. Counts, missing values, finite and infinite counts, minimum, maximum,
  sum, mean, distinct values and duplicate rows are exact. Distinct values and
  row identity are kept on disk. Quartiles are a bounded deterministic sample
  (`quantile_sample_size`, default 4096), labeled `deterministic_approximation`,
  and are not a verdict input. The chunked command publishes a profile-only
  artifact at stage `profiled` whose verdict is blocked, because quality, task
  detection and preprocessing did not run. Its ingestion record describes the
  file and sets `memory_bytes` to null. The default audit does not call this
  path and keeps the stages `inspected`, `planned` and `prepared`. It is not
  certified as G1-25.
- **Not implemented yet:** date inference, generators and arbitrary iterables.
  A refused input is never published as a new audit.

## Detection

**Leakage detection is statistical.** A feature that encodes the outcome for
reasons the numbers do not show will not be found. See
[safety-model.md](safety-model.md#leakage-what-is-and-is-not-detectable).

**Identifier detection is a heuristic.** A column of distinct values with an
identifier-shaped name is *reported*, never dropped. A genuine measurement that
happens to be unique per row looks the same.

**Scaling applies to the numeric branch only.** A model declaring
`requires_scaling` gets its numeric columns standardised; ordinal-encoded and
one-hot columns are passed through as encoded. For a distance model — `knn` in
either family — an ordinal column arrives with raw integer levels beside
standardised numerics and weighs more in the distance than its spread deserves.
Measured on a three-level ordinal beside one standardised numeric: standard
deviations 1.0 and 0.8165.

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
Standardising rescues it. Measured on the regressors as well as the classifiers:
importance 0.9997 at a range of 5.5, 0.2321 at 5.5e-07, and 0.0000 at 5.5e-08.

**A regression target below about 1.5e-8 in standard deviation silently stops
the tree family.** scikit-learn's tree splitter compares a candidate split's
impurity improvement against a tolerance derived from double eps. When the
target's variance falls beneath it, every node looks pure, no split is taken,
and `decision_tree_regressor`, `random_forest_regressor`, `extra_trees_regressor`
and `gradient_boosting_regressor` return the training mean for every row — with
no error, no warning, and a perfectly well-shaped array of finite numbers.
Measured: R² 1.000 at a target standard deviation of 2.2, 0.706 at 2.2e-08, and a
constant prediction at 2.2e-09. `linear_regression`, `ridge_regression`,
`knn_regressor` and `hist_gradient_boosting_regressor` are unaffected. This is
the target-side mirror of the feature-side threshold above. Rescaling the target
rescues it; the library will not do that for you, because the units of the thing
being predicted are yours.

**StandardScaler overflows above a magnitude near 1e200, and underflows below
about 1e-161.** The variance pass squares each value. Above the range, a column
of finite float64 values produces an all-`NaN` scaled column and a numpy
`RuntimeWarning`; no built-in model can reach that silently, because no model
both requires scaling and consumes `NaN` natively, so the `NaN` reaches an
estimator that refuses it — though the resulting error names `NaN` rather than
the overflow behind it. Below the range the variance underflows to exactly zero,
scikit-learn sets `scale_ = 1.0`, and the column passes through **completely
unscaled** with no warning at all — so a model that asked for scaling silently
does not get it. That direction is the quieter of the two.

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
identical data. Every artifact records the versions that produced it. The same
holds for recorded dtype names: pandas 3 reports a text column as `str` where
pandas 2 reports `object`, so the same file audited under the two majors yields
two different, equally correct artifacts. The test suite keeps one exact fixture
per supported major and asserts that they differ in those fields and no others.

**A dataset fingerprint identifies content, not provenance.** It cannot show
where the data came from, who collected it, or whether that was lawful.

**Row and column order are part of dataset identity.** Sorting a file changes its
fingerprint. This is deliberate — see
[audit-artifact.md](audit-artifact.md#identity-and-diffing) — but it will
surprise anyone expecting set semantics.

**Schema compatibility is not promised yet.** Artifacts carry a schema version so
that future migration is possible. No migration path exists today.

## Training and comparison

**A comparison is one split, not cross-validation.** Models are trained on one
training set and judged on one evaluation set, drawn once and shared by every
model in the run. A different seed gives a different split and can give a
different order. Cross-validation is future work; it is not a hidden default.

**Multiclass ROC-AUC is not computed.** It needs an explicit one-vs-rest or
one-vs-one policy and every class present in the evaluation rows. This version
publishes no such policy and reports the metric as not applicable, with that
reason, rather than choosing one silently.

**R² is reported as undefined on a constant evaluation target**, and with fewer
than two evaluation rows. scikit-learn answers a constant target with `1.0` for a
perfect constant prediction and `0.0` for a wrong one — the same `0.0` whether
the prediction is off by 1.5 or by 892.5.

**A ranking is not a recommendation.** It says which model ranked first under one
dataset, one split, one preprocessing contract, one configuration and one metric.
It says nothing about which model suits the problem.

**Timing is observational.** It never decides a ranking and is deliberately
excluded from serialised results, because it would make two identical runs
compare unequal.

See [training-and-comparison.md](training-and-comparison.md).

## Not implemented

Hyperparameter tuning, cross-validation orchestration, parallel model training,
feature importance, SHAP, resampling, automatic class weighting, model
persistence, and model selection of any kind. Several of these are deliberate
product boundaries rather than missing work.
