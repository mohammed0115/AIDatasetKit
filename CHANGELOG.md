# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions before `1.0` may change public interfaces and the artifact schema; see
[API stability](#api-stability).

## 0.1.0a1 — unreleased

First alpha, not yet released: nothing has been published to PyPI, TestPyPI or
as a GitHub release, and no version is tagged. Everything below is new.

### G1-W5 chunked profiling for CSV and TSV

Evidence: `docs/evidence/G1_W5_CHUNKED_PROFILING_REPORT.md`. Not certified.

- **Opt-in chunked profiling** of `.csv` and `.tsv` via
  `profile_delimited_chunks` and `aidatasetkit audit --chunked-profile`. The
  chunked command does not call `load_table` or `DataProfiler.profile`. The
  default audit still loads the whole table and does not call this path.
  Resource limits are unchanged and still refuse an over-limit file before a
  scratch directory is created.
- The scan covers the population in bounded memory. Counts, missing values,
  finite and infinite counts, duplicate rows, and the numeric minimum, maximum,
  sum and mean are exact online accumulators. Distinct values and row identity
  use length-prefixed keys in scratch storage. Quartiles are always a bounded
  deterministic sample (`deterministic_reservoir`), labeled
  `deterministic_approximation`, and are not a verdict input. The fingerprint
  is the population fingerprint, folded from scratch files in fixed-size
  blocks. Scratch files are private and removed on success, failure and
  interruption.
- The chunked artifact is profile-only, at stage `profiled`. Its verdict is
  blocked with the reason "Full-table audit verdict is unavailable in chunked
  profiling mode." Quality findings, task detection and preprocessing are
  recorded as unavailable. The ingestion record describes the file;
  `memory_bytes` is null because no frame was built. The config fingerprint
  covers the scan contract (chunk size, reservoir size, sampling, encoding,
  delimiter, header, resource limits, fingerprint algorithm) and not the path
  or the clock. The reservoir capacity parameter is `quantile_sample_size`.
- **Artifact schema `1.2`.** The artifact gains a typed `chunked_profiling`
  record, `null` on the default path. Publication schema remains `1.0`.
  Package version remains `0.1.0a1`. G1-25 stays
  `MISSING_PENDING_CERTIFICATION`. Recorded progress stays G1 16/26 and
  overall 108/232.

### G1-W4 format expansion: Excel .xlsx

Evidence: `docs/evidence/G1_W4_XLSX_READER_REPORT.md`.

- **Excel `.xlsx`** reader through the new optional `excel` extra (openpyxl).
  A workbook is one worksheet's table: the first row is the header, and the
  row/column/cell budgets are enforced from the worksheet's declared dimensions
  before any cell is materialized. A multi-sheet workbook is refused as
  ambiguous unless the `sheet=` argument names one; a macro-enabled archive
  (`xl/vbaProject.bin`) is refused; a corrupt zip is `MalformedInputError`,
  never a bare openpyxl/zip error. Formulas read as their cached values
  (`data_only=True`); a missing openpyxl is `MissingDependencyError` naming the
  extra.
- `load_table` gains a keyword-only `sheet` argument, refused for every
  non-xlsx source. The CLI accepts `.xlsx` and `--sheet`. A header with no
  data rows is empty input, the same as every other reader; a blank data row
  is kept. Encoding, delimiter and `header=False` are refused on an Excel
  file rather than ignored.
- Artifact schema remains `1.1`, publication schema `1.0`, fingerprints
  unchanged. 6 dedicated mutations, 6 killed (`scripts/g1_w4_mutations.py`).

### G1-W3 format expansion: JSON, JSONL, Parquet, Feather

Evidence: `docs/evidence/G1_W3_FORMAT_READERS_REPORT.md`.

- **JSON and JSONL** readers in `aidatasetkit.ingestion`: a `.json` file is an
  array of flat objects, a `.jsonl`/`.ndjson` file is one object per line.
  Duplicate keys, nested values, non-standard `NaN`/`Infinity` constants and
  non-object records are refused with structured errors, never read silently.
  JSONL is checked as it streams: a limit refusal stops reading at the
  offending line. Both share the one records authority in the loader.
- **Parquet and Feather** readers through the new optional `parquet` extra
  (pyarrow, floor 14.0.1 — the CVE-2023-47248 fix). Parquet row, column and
  cell budgets are enforced from the footer before any data is materialized;
  Feather enforces columns from the IPC schema before data and rows/cells
  before the pandas conversion. Corrupt files are `MalformedInputError`;
  a missing pyarrow is `MissingDependencyError` naming the extra.
- CLI audits accept the new formats; `audit.json` records their ingestion
  metadata. Artifact schema remains `1.1`, publication schema `1.0`,
  fingerprints unchanged.
- 11 dedicated mutations on the new guards, 11 killed
  (`scripts/g1_w3_mutations.py`); the G1-W1 and G1-W2 harnesses re-run green.

### G1-W2 resource governance and large-input safety

Evidence: `docs/evidence/G1_W2_RESOURCE_GOVERNANCE_REPORT.md`.

- Added immutable `IngestionLimits` policy with finite defaults for source bytes,
  rows, columns, cells, CSV/TSV field length, records, record keys and record
  character volume.
- Enforced limits before pandas parsing or analysis for files, DataFrames and
  supported records. CSV/TSV row, shape and field checks fail during the
  existing streaming validation pass.
- Added structured resource-limit errors and matching CLI flags. Omitted CLI
  flags inherit the finite library defaults; resource refusal does not create a
  publication or replace `CURRENT`.
- Artifact schema remains `1.1`; publication schema and fingerprints remain
  unchanged. This work does not add true chunked profiling or generator support.

### G1-W1 ingestion foundation

Evidence: `docs/evidence/G1_W1_INGESTION_FOUNDATION_REPORT.md`.

- **`aidatasetkit.ingestion.load_table`** — the one table-reading authority:
  `.csv` and `.tsv` paths (`pathlib.Path`, never a bare string), pandas
  DataFrames and lists of records, returning `LoadedTable(frame, metadata)`.
  Every input is read correctly or refused with an `IngestionError`; nothing is
  read into a wrong table silently.
- **Fixed — a semicolon CSV was read as one column** (audit P0-2). Delimiters
  `,` `;` tab `|` are detected with quoting honoured; an ambiguous file is
  refused, a named delimiter the file does not use is refused, and every row's
  field count is validated — pandas pads a short row, this refuses it.
- **Explicit encodings**: utf-8, utf-8-sig (or a byte-order mark), latin-1,
  cp1256. No detection; an undecodable byte is `EncodingError`, never a raw
  `UnicodeDecodeError`.
- **CLI**: reads `.tsv` as well as `.csv`; new `--encoding` and `--delimiter`
  (`tab` for a tab). Duplicate-header detection moved into ingestion.
- **Artifact**: a new `ingestion` record in `audit.json` and an *Input* section in
  `report.html`. The semantic fingerprint moved (`0f0fd1d5…` → `00783893…`); the
  migration is recorded.
- **Artifact schema `1.1`** (G1-W1 closure). As first merged (`90ecfae`), G1-W1
  added the `ingestion` key but left `schema_version` at `1.0`, against the
  contract that a change of shape is identifiable from the version alone. It is
  now `1.1`, a minor bump because the change is additive; `lineage.json` carries
  the same version. `publication_schema_version` stays `1.0` — the run layout
  did not change — and so does the package version. The fingerprint moved again
  (`00783893…` → `3e93dd5e…`), recorded as its own migration step.
- Not yet: automatic encoding detection, input size limits, date inference,
  chunked reading, JSON, Excel, Parquet.

### G0 baseline stabilisation

Evidence for every item: `docs/evidence/G0_FINAL_CERTIFICATION_REPORT.md`.

- **Breaking — output layout.** `aidatasetkit audit` publishes each run as a
  complete set under `<output>/runs/<run_id>/` with a `manifest.json` of sizes
  and SHA-256 digests, and makes it current by atomically replacing
  `<output>/CURRENT`. Read runs with `aidatasetkit.evidence.read_current`, which
  verifies the whole set or refuses it. Nothing is written at the root of
  `--output` any more; files an older version left there are not touched.
- **Validation and final test are separate.** `AIDataFacade.evaluate()` scores
  the validation rows, which also rank a comparison;
  `status["validation_used_for_selection"]` says when they did.
  `evaluate_final()` scores the external test frame once and freezes the
  experiment: comparing, selecting and training refuse until new data is loaded.
- **Breaking — `KitConfig.cv_folds` removed.** Nothing read it; there is no
  cross-validation. The config fingerprint moved, and the migration is recorded.
- **Multicollinearity check** (twelfth default check), bounded to 50 numeric
  columns and 20,000 rows, overflow-safe, with a validated threshold.
- **Overflow.** Integers beyond float64 are counted exactly on pandas 2; means,
  medians, quantiles and standard deviations of values near 1e308 are exact
  instead of `nan`/`-inf`; a range beyond float64 is recorded as undefined.
- **Dependency floors** raised to the tested minimum: numpy 1.26.4, pandas 2.1.4,
  scipy 1.11.4, scikit-learn 1.6.1 (from an untested 1.4). SciPy 1.11 works again.
- The package description no longer claims a "signed-off" artifact.

### The guided workflow

- **`AIDataFacade`**, exported from the package root — one object walking load,
  profile, statistics, quality, prepare, compare, select, train, evaluate,
  predict. Every method delegates; the facade owns sequence and state and decides
  nothing about data. It is optional, and for identical inputs it produces
  results identical to calling the layers directly, which the suite asserts.
- The root export is **lazy** (PEP 562), so `import aidatasetkit` still pulls in
  no scikit-learn and the CLI keeps its fast path for a mistyped file name.
- `prepare()` **fits nothing**. It derives the plan for each distinct capability
  profile from the training side of the split; fitting there would move a fit
  ahead of the split and destroy evaluation.
- Data the quality checks call `BLOCKED` **cannot be trained through the facade**,
  and there is deliberately no override parameter. `aidatasetkit.training`
  remains available for anyone who has read the findings and means to proceed.
- **`aidatasetkit.prediction`** — `PredictionResult` and `predict_frame`.
  Classification predictions come back as the caller's original labels;
  probability columns are named by those labels rather than positioned.
  Regression predictions touch no encoder.

### Clustering

- **Six clusterers** — `kmeans`, `minibatch_kmeans`, `dbscan`, `optics`,
  `agglomerative` and `birch` — on the same `ModelStrategy`, the same
  `ModelCapabilities`, the same registry and the same factory as the eighteen
  supervised models. Twenty-four models still need only **five** preprocessors
  between them, and all five are now shared across task families: Gaussian naive
  Bayes was the last model sitting alone on a pipeline, and OPTICS and
  Agglomerative joined it.
- **Every one of the six requires scaling**, which is six measurements that
  happened to agree rather than a family-level assumption. On three separated
  blobs with one feature recorded in a unit `1e5` larger, DBSCAN scored an
  adjusted Rand index of **0.000** unscaled against **1.000** standardised — it
  recovered nothing at all, because `eps` is an absolute distance and nothing in
  the data says what a column was measured in.
- **`supports_out_of_sample_assignment`** — a new capability, false by default.
  `KMeans`, `MiniBatchKMeans` and `Birch` keep enough fitted state to place a row
  they never saw; `DBSCAN`, `OPTICS` and `AgglomerativeClustering` produce a
  labelling of the fitted rows and define no rule for any other row. Asking one
  of those three to assign is refused with the reason, rather than approximated
  by refitting — which would produce a different partition, not an assignment
  into this one.
- **`Estimator` was split.** It required `predict`, and three of these six do not
  have it, so a new `Fittable` (fit, get_params, set_params) sits beneath both
  `Estimator` and a new `ClusterEstimator` (fit_predict). The members required by
  `Estimator` are unchanged, so nothing that satisfied it before stops doing so.
  The overlap between `Estimator` and `ClusterEstimator` is exactly the set of
  models declaring out-of-sample assignment, and the contract tests assert the
  protocol and the declaration agree rather than trusting either alone.
- `ClusteringRunner` and `ClusteringResult` in `aidatasetkit.training`, kept apart
  from `ModelTrainer` rather than folded into it. **Its metrics are in-sample**,
  computed on the rows the model was fitted on, and the module says so at length:
  there is no target for a score to be inflated by, the metrics are internal, and
  three of the six could not score a held-out set even in principle. A high
  silhouette describes this partition; it is not evidence that it will reproduce.
- Noise is never counted as a cluster. `-1` is the algorithm declining to assign,
  so it is excluded before any metric is computed and reported separately as
  `noise_count`.
- `AIDataFacade(task="clustering")` takes **no target**, and naming one is
  refused rather than ignored. The supervised half of the object refuses with a
  message naming the reason, and `BLOCKED` data cannot be clustered either.
- **`silhouette`, `davies_bouldin` and `calinski_harabasz`, and no default
  ranking metric.** The absence is the decision. On two interleaved half-moons
  where the correct partition is known, all three preferred the **wrong** one,
  unanimously — KMeans's convex mis-split scored `+0.488` / `0.781` / `440.7`
  against DBSCAN's correct `+0.327` / `1.167` / `193.5` — because all three
  reward compact convex blobs. Nor is a good score evidence that structure
  exists: 300 rows from a single Gaussian scored a silhouette of `+0.347` at
  k=3. A clustering comparison therefore reports all three and ranks by none.
- The silhouette is **reported rather than computed** above 10,000 scored rows.
  It needs every pairwise distance: measured here, 20,000 rows took 164 seconds
  and a 3.2 GB matrix. It is not approximated by sampling, because a sampled
  silhouette is not the silhouette and would differ between two runs of identical
  data. Raise `silhouette_row_limit` to compute it in full.
- `davies_bouldin` and `calinski_harabasz` are reported unavailable on a sparse
  matrix, which scikit-learn refuses for those two and accepts for the
  silhouette. The matrix is not densified: it is sparse because a one-hot
  encoding made it wide, and materialising it could cost more than the metric.

### Changed — **breaking**

- **Artifacts built before clustering landed have a different
  `semantic_fingerprint`.** `supports_out_of_sample_assignment` is inside
  `capabilities.to_dict()`, which is inside the model evidence, which is inside
  `semantic_dict()` — so every artifact that records a model fingerprints
  differently. The field is emitted for supervised models too, where the answer
  is a trivial `false`, because a key present on some rows and absent on others
  is one every consumer has to branch on.

  The field was deliberately **not** special-cased out of the artifact builder to
  keep the old number stable. Hiding a capability from the evidence to preserve
  an identity is arranging the record to match the fingerprint, and the
  fingerprint exists to report that the record changed.

  A second field moved with it: the `known_limitations` entry describing scope no
  longer says clustering is out of scope, because six clusterers are now held to
  the same executed capability contracts as the eighteen supervised models. That
  entry is inside `semantic_dict()` too. It is the second time that sentence has
  been narrowed — S6 did it for regression — and both times for the same reason:
  an artifact travels further than any other document here, so it must not carry
  a scope statement the library has outgrown. The verdict itself is still
  verified for classification only, and that half of the sentence did not move.

  For the committed example audit the identity moved from
  `4d3139d4a04e34bd34243c80d8232475e30edbfe9daa9f660ef2038f37d95bbc` to
  `b8581966475da37125414a12afc3144d6b1468258ff3dcb3149368406b884852`. A
  regression test reconstructs the old artifact by deleting that one key and
  restoring that one sentence, and asserts the result reproduces the old
  fingerprint exactly — so the migration is proven to have carried nothing else
  with it. The artifact schema is unchanged at `1.0`. Regenerate stored artifacts
  with `examples/audit_churn/generate_artifacts.py`, or compare against the new
  value.

- **`KitConfig.random_state` must be a plain `int`.** It was annotated `int` and
  enforced nothing, so a `numpy.random.RandomState` or `Generator` could be
  stored there and handed by reference to every consumer — mutable state rather
  than a seed, advancing on every draw, so two runs configured identically would
  not agree. Splitting made that reachable through a supported public call.
  `None` and `bool` are refused too: the first means "draw from global entropy",
  and the second would silently become the seed `1`.

### Training, evaluation, and comparison

- `ModelComparator` and `ModelTrainer` — one split drawn once and shared by every
  model in a run, capability-driven preparation per model, an independent fit per
  model, and one evaluation policy. Comparing ten models is exactly ten fits;
  there is no search, no grid, and no cross-validation.
- Metrics that say why they are missing. `accuracy`, `precision`, `recall`, `f1`
  and `roc_auc` for classification; `mae`, `mse`, `rmse` and `r2` for regression.
  A metric that cannot be computed carries a status and a reason rather than a
  `NaN` or a silent omission.
- Binary metrics follow the **resolved positive label**, not encoded class 1.
  `"churn"` sorts before `"stay"`, so the event of interest is column zero, and
  `predict_proba[:, 1]` would have measured the wrong class.
- Values are reported unnegated: `mae = 4.2`, never
  `neg_mean_absolute_error = -4.2`. Every metric carries its direction.
- Ranking defaults to `f1` for classification and `rmse` for regression, with
  alphabetical tie-breaking. A model with no ranking metric is reported as
  unranked rather than placed somewhere in the order.
- R² is reported as undefined on a constant evaluation target, where
  scikit-learn returns a plausible `0.0` whether the prediction is off by 1.5 or
  by 892.5.

### Models

- **Nine classifiers and nine regressors**, on one `ModelStrategy`, one
  `ModelCapabilities`, one registry and one factory. Every capability was
  verified by running the estimator, not by reading its documentation.
  Eighteen models need only five preprocessors between them, because the cache
  key is a capability triple rather than a model name.
- Regression models and their preprocessing are verified. The **readiness
  verdict** is not: its thresholds, and the leakage checks behind it, were built
  and measured against classification targets. A regression audit says so.
- **Eight short aliases name a family rather than a model.** `dummy`,
  `baseline`, `decision_tree`, `extra_trees`, `gradient_boosting`,
  `hist_gradient_boosting`, `knn` and `random_forest` each answer for a
  classifier *and* a regressor, so resolving one without a task raises
  `AmbiguousModelAliasError`.

  Name the family — `ModelFactory.create("random_forest",
  task="classification")`, or `--model random_forest --task classification` — or
  use the canonical name, `random_forest_classifier`. Canonical names are never
  ambiguous, and neither are `logistic`, `logreg`, `gnb` or `ridge`.

  The registry refuses rather than preferring one family, because preferring one
  would be choosing arbitrarily on the user's behalf. `name in registry` still
  answers `True` for these eight: they *are* registered, and reporting a known
  model as unknown would send a caller looking for a typo.

  **If you built against a pre-release checkout** in which these aliases resolved
  to the classifier, this is the one call that changes.

- **Artifacts built against a pre-release checkout have a different
  `semantic_fingerprint`.** The `known_limitations` entry describing scope
  changed once regression was verified, and that entry is inside
  `semantic_dict()`, so identical data under identical settings now fingerprints
  differently than it did before regression landed. The artifact schema is
  unchanged at `1.0`; this note is the record of why the identity moved.

### Audit and evidence

- `AuditArtifact`: one canonical, versioned, deterministic record of what an
  analysis established — dataset identity, quality findings, preprocessing
  decisions, feature lineage, model context, environment, and known limitations.
- Artifact schema `1.0`, versioned **independently of the package** so stored
  artifacts stay readable as the library changes.
- Deterministic dataset and configuration fingerprints. Content is identified
  without being stored; the semantic fingerprint excludes the clock, the
  environment, and the file name, so two runs of the same data compare equal.
- Redaction on by default: text taken from a dataset is hashed unless its key is
  vocabulary this library defines.
- `lineage.json` — each input column and what it became, taken from the
  preprocessing layer rather than parsed back out of output names.
- `report.html` — a standalone page rendered from the same canonical mapping as
  `audit.json`, with no scripts, no network requests, and no dependencies.

### Command line

- `aidatasetkit audit` — read a `.csv` or `.tsv`, profile, check, plan, and write
  three artifacts.
- Documented exit codes for CI: `0` below threshold, `1` could not run,
  `2` threshold met, `3` blocked.
- `--fail-on never|warning|review|error`, defaulting to `review`.
- A dataset that cannot be prepared still produces a full artifact with verdict
  `blocked` and the reason recorded.
- Nothing is ever trained. `--model` supplies a capability context only.

### Analysis layers

- **Statistics** — every degenerate case that numpy or scipy answers with `NaN`
  raises instead, so a number in a report is never a silent placeholder.
- **Profiling and data quality** — per-column measurement and twelve independent
  checks covering leakage, identifiers, constants, cardinality, missingness,
  outliers, duplicates, multicollinearity, and numbers stored as text.
- **Smart visualization** — chart *recommendations* separated entirely from
  rendering; matplotlib stays optional.
- **Preprocessing** — capability-driven planning. The plan is a proposal you can
  read, with a reason for every decision, and it is never applied behind your
  back.
- **Classification catalog** — nine scikit-learn classifiers whose declared
  capabilities were each verified by running the estimator.

### API stability

Interfaces and the artifact schema may change before `1.0`. The artifact carries
`schema_version` so a future release can recognise and migrate an older file.

### License

- Apache License 2.0 (`Apache-2.0`), chosen by the repository owner and applied
  in the metadata and in `LICENSE`. See `docs/LICENSE_DECISION.md`.

### Known limitations

See `docs/limitations.md`, which is the single canonical list and travels inside
every artifact.
