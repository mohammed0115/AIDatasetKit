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

Artifacts:
  ./aidk-audit/audit.json
  ./aidk-audit/lineage.json
  ./aidk-audit/report.html
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

Open `aidk-audit/report.html` in a browser. Commit `aidk-audit/audit.json` to
your repository and the next run will diff against it.

In CI:

```yaml
- run: aidatasetkit audit data/train.csv --target Churn
```

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

## What you get

| File | What it is |
|---|---|
| `audit.json` | The artifact. Canonical, versioned, deterministic, diffable. |
| `lineage.json` | Every input column and what it became. |
| `report.html` | The same evidence for a human. No server, no network. |

`audit.json` is the record; the HTML is a rendering of it. They cannot disagree.

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
yourself. [docs/privacy.md](docs/privacy.md) lists all of them.

**Bounded language.** No output says "safe", "compliant", or "leakage-free". It
says *no known blocker found*, *review required*, *possible leakage*,
*verified train-only fit*.

## Scope of this alpha

Supported: pandas DataFrames and CSV files · tabular data · binary and multiclass
classification readiness · scikit-learn model contexts · profiling · quality and
leakage diagnostics · capability-driven preprocessing plans · feature lineage ·
audit artifacts.

Not supported yet: regression, clustering, anomaly detection, dimensionality
reduction, time series, text, images, model training, model comparison,
hyperparameter tuning, databases, cloud storage, Parquet, Excel.

## Known limitations

- Leakage detection is statistical. A feature that encodes the outcome for
  reasons the numbers do not show will not be found.
- Only tabular supervised classification has been verified end to end.
- Datetime columns are profiled but never turned into features automatically.
- An audit records what preprocessing *would* do. It does not prove a model was
  trained on the data it describes.
- A dataset fingerprint identifies content, not provenance.

The full list travels inside every artifact, so a reader always has it.

## Roadmap

Next, and shaped by what alpha users report: regression readiness, richer
leakage evidence, artifact diffing across runs, SARIF output for code-scanning
integrations. Model training and evaluation are deliberately later — the value
here is what happens *before* a model exists.

## Release blockers

**This alpha is not ready to publish.** One item is outstanding:

- **LICENSE DECISION REQUIRED.** `pyproject.toml` has declared MIT since the
  first commit, but there is no `LICENSE` file in the repository. That
  declaration was left exactly as found: choosing a licence, and removing one,
  are both decisions for the owner rather than for a tool. Add a `LICENSE` file
  that matches the declaration, or change the declaration, before publishing.

## Documentation

- [Getting started](docs/getting-started.md)
- [The audit artifact](docs/audit-artifact.md)
- [Feature lineage](docs/lineage.md)
- [Safety model](docs/safety-model.md)
- [Privacy](docs/privacy.md)
- [Limitations](docs/limitations.md)
- [Contributing](CONTRIBUTING.md)

## Status

Public alpha (`0.1.0a1`). The artifact schema is versioned independently of the
package (`1.0`) so that stored artifacts stay readable as the library changes.
Interfaces may still move. Feedback on the audit artifact — what is missing, what
is unclear, what you would want to fail a build on — is the most useful thing you
can send.
