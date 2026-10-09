# AIDatasetKit

**Prove what happened between your data and your model.**

AIDatasetKit audits a tabular dataset *before* you train on it. It profiles every
column, checks for the problems that quietly ruin a model — leakage, identifiers,
constants, missing values — records exactly what preprocessing would do and why,
and writes the whole thing to a machine-readable audit artifact you can commit,
diff, and fail a CI build on.

```
pip install -e .
aidatasetkit audit train.csv --target Churn --model logistic_regression
```

```
AIDatasetKit audit

Dataset:  train.csv
          600 rows x 10 columns
          fingerprint 8135be14ca667643

Verdict:  BLOCKED
          error finding target_leakage_exact_duplicate on Churn_Copy

Findings: 1 error, 4 warning, 1 info

Key issues:
  - Churn_Copy: Column 'Churn_Copy' is exactly equal to the target 'Churn' in every row. Training on it would measure nothing.
  - CustomerID: Column 'CustomerID' has 600 distinct values, above the configured threshold of 50.
  - HighCardinalityFeature: Column 'HighCardinalityFeature' has 220 distinct values, above the configured threshold of 50.
  - CustomerID: Column 'CustomerID' may be a record identifier: 100.0% of its values are distinct and its name or value structure resembles an identifier. This is a heuristic and needs review.
  - HighCardinalityFeature: Column 'HighCardinalityFeature' may contain information about the target 'Churn' that would not be available at prediction time (signals: deterministic_mapping). This is a heuristic and needs review.

Needs review: possible_id_like (CustomerID), possible_target_leakage (HighCardinalityFeature), high_cardinality (HighCardinalityFeature)

Artifacts (now current in aidk-audit/CURRENT):
  aidk-audit/runs/20260927T101530123456Z-…/audit.json
  aidk-audit/runs/20260927T101530123456Z-…/lineage.json
  aidk-audit/runs/20260927T101530123456Z-…/report.html
```

Exit code `3`. Your CI job just caught a target leak before anyone trained on it.

---

## What this is not

Being clear about this saves you time:

- **Not an AutoML tool.** It never picks a model, tunes a parameter, or decides
  a threshold for you.
- **Not a cleaning tool.** It does not fill, drop, encode, or fix anything. It
  tells you what it would do and why, and you decide.
- **Not a compliance product.** It produces evidence. It does not certify
  anything, and no output of it should be read as a guarantee.
- **Not a training framework.** Nothing here fits a model. The `--model` flag
  selects a *capability context* so the audit can explain why a scaler is in the
  plan; no estimator is ever constructed.

## Install

Requires Python 3.11 or newer.

```bash
git clone <this repository>
cd aidatasetkit
pip install -e .
```

Optional extras:

```bash
pip install -e ".[viz]"        # matplotlib, for chart rendering
pip install -e ".[boosting]"   # reserved for future boosting backends; no model uses it yet
pip install -e ".[dev]"        # pytest
```

## 60 seconds

Generate the example dataset and audit it:

```bash
python examples/audit_churn/make_dataset.py
aidatasetkit audit examples/audit_churn/train.csv \
    --target Churn \
    --model logistic_regression \
    --output ./aidk-audit/
```

Open the `report.html` the summary names in a browser. Every audit is published
as a complete run under `aidk-audit/runs/<run_id>/`, and `aidk-audit/CURRENT`
names the latest; read it through the verifying reader rather than by path. To
diff audits over time, commit the current run's `audit.json`:

```python
from aidatasetkit.evidence import read_current

run = read_current("aidk-audit")          # CURRENT -> manifest -> every file checked
open("audit.json", "w", encoding="utf-8").write(run.text("audit.json"))
```

In CI — the exit code gates the build:

```yaml
- name: Audit the training data
  run: aidatasetkit audit data/train.csv --target Churn --fail-on review

- name: Keep the evidence
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: dataset-audit
    path: aidk-audit/
```

Or in any shell-based pipeline:

```bash
aidatasetkit audit train.csv --target Churn --fail-on review || exit $?
```

`--fail-on review` stops the build when something needs a human decision;
`--fail-on error` only when something is certainly wrong.

> **Before sharing an artifact.** It contains no cell values by default, but it
> does contain column names, class labels, numeric minima and maxima, and
> one-hot category names. See `docs/privacy.md`.

## From Python

```python
import pandas as pd

from aidatasetkit.evidence import AuditBuilder, canonical_json
from aidatasetkit.profiling import DataProfiler, DataQualityInspector

frame = pd.read_csv("train.csv")
profile = DataProfiler().profile(frame)
quality = DataQualityInspector().inspect(frame, profile=profile, target="Churn")

artifact = AuditBuilder(dataset_name="train.csv").build(
    frame, profile=profile, quality=quality
)

print(artifact.verdict.value)           # blocked -- Churn_Copy duplicates the target
print(artifact.review_items)            # what a person still has to decide
open("audit.json", "w").write(canonical_json(artifact.to_dict()))
```

Every layer is usable on its own: profiling without quality checks, quality
checks without preprocessing, preprocessing without a model.

## The guided workflow

For the common path from a dataframe to predictions, one object walks the whole
sequence:

```python
from aidatasetkit import AIDataFacade

ai = AIDataFacade(
    target="Churn",
    id_column="CustomerID",
    task="classification",
    positive_label="churn",     # "churn" sorts first, so it encodes to 0
)
ai.load(train_df, test_df)

ai.profile()
ai.statistics()
ai.check_quality()              # BLOCKED data cannot be trained through here
ai.prepare()                    # plans the pipelines; fits nothing

results = ai.compare_models()   # one split, every model, one metric policy
print(results.to_frame())

ai.select_model("logistic_regression")   # your decision, not the ranking's

ai.train()
print(ai.evaluate()["f1"].value)          # validation: these rows also ranked the models

final = ai.evaluate_final()               # the independent estimate, once, on the test frame;
print(final.report["f1"].value)           # compare/select/train now refuse until a new load()

predictions = ai.predict_test()          # original labels, not 0/1
print(predictions.to_frame().head())
```

Regression works the same way — `AIDataFacade(target="price")` infers the task,
and predictions come back as quantities that never touch a label encoder.

**`compare_models()` is not hyperparameter optimization.** Comparing ten models
is exactly ten fits at their published defaults. And it does not select anything:
a ranking holds under one dataset, one split and one metric, so choosing stays
explicit.

The facade is optional. Everything above is the same code the section before it
calls directly, and for identical inputs the two routes produce identical
results. See `docs/facade.md`.

## Clustering

Data with no labels takes the same path, minus the target:

```python
ai = AIDataFacade(task="clustering")     # no target, and naming one is refused
ai.load(customers)
ai.check_quality()

result = ai.cluster("kmeans", n_clusters=3)
print(result.n_clusters, result.noise_count, result.cluster_sizes)
print(result.evaluation["silhouette"].value)

result.assign(new_customers)             # only for models that can
```

Six algorithms — `kmeans`, `minibatch_kmeans`, `dbscan`, `optics`,
`agglomerative`, `birch` — on the same registry, factory and capability-driven
preprocessing as the eighteen supervised models. Twenty-four models still need
only five preprocessors between them.

Three things about it are worth knowing before you read a number it produces:

- **Half of them cannot label a row they were not fitted on.** `DBSCAN`, `OPTICS`
  and `AgglomerativeClustering` produce a labelling of the fitted rows and define
  no rule for any other row, so `assign()` refuses rather than refitting. The
  capability is recorded per model, measured by calling it.
- **The metrics are in-sample and internal.** They describe how tidily this model
  divided *this* data. A high silhouette is not evidence the clusters will
  reproduce.
- **There is no default ranking metric, deliberately.** On two interleaved
  half-moons, all three internal metrics preferred KMeans's *wrong* convex split
  to DBSCAN's correct one. Ranking on any of them would have put the model that
  found the real structure last. `docs/clustering.md` has the numbers.

## What you get

| File | What it is |
|---|---|
| `audit.json` | The artifact. Canonical, versioned, deterministic, diffable. |
| `lineage.json` | Every input column and what it became. |
| `report.html` | The same evidence for a human. No server, no network. |
| `manifest.json` | Run id, time, and the size and SHA-256 of each file above. |
| `CURRENT` (one level up) | Which run is current, and the digest of its manifest. |

`audit.json` is the record; the HTML is a rendering of it. They cannot disagree.

**The three files are published together or not at all.** They are written into a
private staging directory, checked against the manifest, moved into
`runs/<run_id>/`, and only then made current by replacing `CURRENT` in one atomic
step. A crash, a full disk, or a second audit writing into the same directory at
the same moment leaves `CURRENT` naming a complete run — the old one or the new
one, never a mixture. `read_current()` checks every file against the manifest
before returning it and refuses a set that does not match.

## The safety philosophy

**Detect, explain, recommend — never silently modify.** Every version of this
library has been built on one rule: it does not change your data behind your
back. A column that looks like an identifier is *held back and reported*, not
dropped. An ordinal order is never guessed from the alphabet. Numbers stored as
text are never quietly parsed.

**Metadata that lies is worse than no metadata.** Every capability a model
declares was verified by running the estimator, and the contract tests re-verify
them on each run rather than trusting a table.

**Privacy by default.** Text that comes out of your data — the most frequent
value of a column, a value quoted inside a finding — is hashed unless you pass
`--include-values`. Some real observations do remain: numeric minima and maxima,
target class labels, one-hot category names, and any ordering you supplied
yourself. `docs/privacy.md` lists all of them.

**Bounded language.** No output says "safe", "compliant", or "leakage-free". It
says *no known blocker found*, *review required*, *possible leakage*,
*verified train-only fit*.

## Scope of this alpha

Supported: `.csv` and `.tsv` files (comma, semicolon, tab or pipe, detected or
named; utf-8, utf-8-sig, latin-1 or cp1256, never guessed), `.json` files
holding an array of flat objects, `.jsonl`/`.ndjson` files holding one object
per line, `.parquet` and `.feather`/`.arrow` files (via the optional `parquet`
extra), `.xlsx` workbooks (one sheet, via the optional `excel` extra; a
multi-sheet workbook needs `sheet=` or `--sheet`), pandas DataFrames and lists
of records · tabular data · binary and multiclass classification readiness ·
regression model contexts · scikit-learn model contexts · profiling · quality
and leakage diagnostics · capability-driven preprocessing plans · feature
lineage · audit artifacts.

Also merged, unreleased: model training, evaluation, model comparison, the
`AIDataFacade` workflow above, and clustering.

Not supported yet: anomaly detection, dimensionality reduction, time series,
text, images, hyperparameter tuning, cross-validation, model persistence,
databases, cloud storage, automatic encoding detection,
date inference, true chunked profiling or generators/other arbitrary iterables.

Every supported input now has finite resource limits before pandas parsing or
analysis: 64 MiB file bytes, 1,000,000 data rows, 1,000 columns, 10,000,000
data cells and 1,000,000 characters per CSV/TSV field by default. The Python
API accepts an immutable `IngestionLimits` policy; the CLI exposes the same
defaults through `--max-input-bytes`, `--max-rows`, `--max-columns`,
`--max-cells` and `--max-field-length`. Refusal is structured and leaves
publication untouched. These are guards, not chunked processing: accepted
inputs are still loaded as one pandas table.

Regression models and their preprocessing are held to the same executed
capability contracts as the classifiers. The *readiness verdict* is not: its
thresholds, and the leakage checks behind it, were built and measured against
classification targets. An audit of a regression target says so in its own
output.

## Known limitations

- Leakage detection is statistical. A feature that encodes the outcome for
  reasons the numbers do not show will not be found.
- The readiness verdict has been verified end to end for classification only.
  Regression models and preprocessing are verified; the thresholds that turn
  findings into a verdict were measured against classification targets.
- Datetime columns are profiled but never turned into features automatically.
- An audit records what preprocessing *would* do. It does not prove a model was
  trained on the data it describes.
- A dataset fingerprint identifies content, not provenance.

`docs/limitations.md` is the canonical list. A summary of it travels inside
every artifact, so a reader always has the essentials to hand.

## Roadmap

Nothing below is cancelled. The order changed deliberately: an audit layer real
people use is worth more right now than a larger catalogue nobody has tried.

**Current alpha (`0.1.0a1`)** — foundation · statistics · profiling and data
quality · visualization planning · capability-driven preprocessing · nine
classifiers · evidence and provenance · the CLI.

**Merged since the alpha, unreleased** — nine regressors, on the same registry,
the same factory and the same capability-driven preprocessing. Eighteen models
still need only five preprocessors between them, because the cache key is a
capability triple and not a model name. One behaviour changed with them: eight
short aliases (`random_forest`, `knn`, `dummy`, and five others) now name both a
classifier and a regressor, so they need a task to settle them. `CHANGELOG.md`
lists all eight.

Also merged: **training, evaluation and model comparison**. One split drawn once
and shared by every model, capability-driven preparation per model, metrics that
say why they are missing rather than going quiet, and a ranking that claims only
what it measured. `docs/training-and-comparison.md` describes it. It is not
hyperparameter optimization: comparing ten models is exactly ten fits.

**Next, after alpha feedback** — richer leakage evidence · shaped by what alpha
users actually report.

**Planned expansion** — anomaly detection and dimensionality reduction · external
model backends (XGBoost, LightGBM, CatBoost) · deep learning · richer leakage
evidence · artifact diffing across runs · SARIF output for code-scanning
integrations.

What ships next is shaped by what alpha users report, not by this list's order.

## License

AIDatasetKit is licensed under the Apache License 2.0. See `LICENSE` for the
full text, and `docs/LICENSE_DECISION.md` for the decision record.

## Documentation

The full documentation ships with the source distribution, under `docs/`:

| File | What it covers |
|---|---|
| `docs/getting-started.md` | Install, first audit, every CLI argument, CI usage |
| `docs/audit-artifact.md` | Every field of `audit.json`, and what each one means |
| `docs/lineage.md` | What happened to each column, and how to read it |
| `docs/safety-model.md` | What the verdict claims, and what it does not |
| `docs/privacy.md` | Exactly what an artifact does and does not reveal |
| `docs/facade.md` | The guided workflow, its state, and its safety behaviour |
| `docs/training-and-comparison.md` | Training, metrics, ranking, and what a comparison claims |
| `docs/clustering.md` | The six clusterers, what their metrics claim, and what they do not |
| `docs/limitations.md` | What this release cannot do |
| `CONTRIBUTING.md` | Setup, tests, architecture boundaries, extension points |

Read `docs/privacy.md` before sharing an artifact outside your team.

## Research & Real-World Comparative Study

The repository includes a reproducible [comparative research lab](https://github.com/mohammed0115/AIDatasetKit/blob/main/examples/research/AIDatasetKit_Real_World_Comparative_Study.ipynb) that contrasts a conventional `pandas`/`scikit-learn` workflow with AIDatasetKit across public tabular datasets. It evaluates trade-offs in **safety, transparency, reproducibility, workflow complexity, flexibility, and runtime**; it does not claim universal superiority or replace domain validation. The [research protocol](https://github.com/mohammed0115/AIDatasetKit/blob/main/docs/research/AIDatasetKit_Research_Protocol.md) defines the datasets, fairness criteria, controlled stress tests, limitations, and threats to validity.

## Status

An unreleased alpha (`0.1.0a1`): not published to PyPI, TestPyPI or as a GitHub release, and no version is tagged. Apache-2.0 licensed. The artifact schema is versioned independently of the
package (currently `1.2`) so that stored artifacts stay readable as the library changes.
Interfaces may still move. Feedback on the audit artifact — what is missing, what
is unclear, what you would want to fail a build on — is the most useful thing you
can send.
