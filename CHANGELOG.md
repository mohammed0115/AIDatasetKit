# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions before `1.0` may change public interfaces and the artifact schema; see
[API stability](#api-stability).

## 0.1.0a1 — unreleased

First public alpha. Everything below is new, because nothing was public before.

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

- `aidatasetkit audit` — profile, check, plan, and write three artifacts.
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
