# Getting started

## Install

```bash
git clone <this repository>
cd aidatasetkit
pip install -e ".[dev]"
```

Python 3.11 or newer. The core dependencies are numpy, pandas, scipy, and
scikit-learn — all of which you almost certainly already have.

## Your first audit

```bash
python examples/audit_churn/make_dataset.py
aidatasetkit audit examples/audit_churn/train.csv --target Churn
```

That prints a summary and writes three files to `./aidk-audit/`. The dataset is
synthetic and deliberately broken, so the audit has something to say.

Add a model context to find out how the data would be *prepared* for a specific
estimator:

```bash
aidatasetkit audit examples/audit_churn/train.csv \
    --target Churn \
    --model logistic_regression
```

Now the artifact also contains a preprocessing plan and feature lineage, and each
step names the capability that asked for it.

## What the arguments do

| Argument | Effect |
|---|---|
| `path` | A CSV file. The only format the alpha reads. |
| `--target` | The label column. Without it you get dataset evidence only. |
| `--task` | `classification` or `regression`. Detected from the target if omitted. |
| `--model` | A model name or alias. Adds capability-driven preprocessing evidence. **Requires `--target`.** Nothing is trained. |
| `--output` | Where the artifacts go. Defaults to `./aidk-audit`. |
| `--fail-on` | `never`, `warning`, `review`, `error`. Default `review`. |
| `--include-values` | Turn redaction **off everywhere**: the most frequent value of each column and every value quoted in a finding are written in plain text. Off by default. |
| `--debug` | Show the full traceback instead of one clear line. |

### Aliases name a family, not a model

Eight short aliases each answer for both a classifier and a regressor:

`random_forest` · `extra_trees` · `decision_tree` · `gradient_boosting` ·
`hist_gradient_boosting` · `knn` · `dummy` · `baseline`

Pass `--task` to say which, or use the canonical name:

```bash
aidatasetkit audit train.csv --target Price --task regression --model random_forest
aidatasetkit audit train.csv --target Price --model random_forest_regressor
```

Without one of those the run stops and names both candidates. Nothing is chosen
for you — preferring one family by convention is exactly the kind of quiet
decision this library refuses to make.

## Before you share an artifact

An audit artifact is designed to travel — into a repository, a CI run, an email.
By default it carries no cell values, but it is not free of information: column
names, class labels, numeric minima and maxima, one-hot category names, and any
ordering you supplied are all in it. Read [privacy.md](privacy.md) before
publishing one, and never pass `--include-values` on data you would not publish.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | The verdict is below your `--fail-on` threshold (default: below `review_required`). |
| `1` | The command could not run: missing file, unknown target, duplicate headers, bad option. |
| `2` | The verdict met the threshold and is not `blocked`. |
| `3` | The verdict is `blocked` and that met the threshold. |

`2` does not mean `review_required` specifically — under `--fail-on warning` a
`ready_with_warnings` verdict returns `2` as well. The code answers *"did this
meet the bar you set"*; which verdict it was is in `audit.json` and in the
summary printed above it.

`2` and `3` are policy outcomes, not errors — the tool worked and is telling you
what it found. Only `1` means the audit itself failed.

## Using it in CI

```yaml
- name: Audit the training data
  run: aidatasetkit audit data/train.csv --target Churn

- name: Keep the evidence
  uses: actions/upload-artifact@v4
  with:
    name: dataset-audit
    path: aidk-audit/
```

Commit `aidk-audit/audit.json` and the diff on the next run shows exactly what
changed about your data — the file is canonical JSON with sorted keys, so a git
diff is readable rather than noise.

## From Python

The CLI is a thin wrapper. Every component is public:

```python
import pandas as pd

from aidatasetkit.evidence import AuditBuilder
from aidatasetkit.models import ModelFactory
from aidatasetkit.preprocessing import PreprocessingPlanner, PreprocessorBuilder
from aidatasetkit.profiling import DataProfiler, DataQualityInspector, TaskDetector

frame = pd.read_csv("train.csv")
target = "Churn"

profile = DataProfiler().profile(frame)
quality = DataQualityInspector().inspect(frame, profile=profile, target=target)
detected = TaskDetector().detect(frame[target], target_name=target)

model = ModelFactory.registration("logistic_regression")
plan = PreprocessingPlanner().plan(
    frame, profile, model.capabilities.preprocessing_profile(),
    quality=quality, target=target,
)

features = frame.drop(columns=[target])
preprocessor = PreprocessorBuilder().build(plan, features)
preprocessor.fit(features)

artifact = AuditBuilder(dataset_name="train.csv").build(
    frame,
    profile=profile,
    quality=quality,
    target=detected,
    plan=plan,
    lineage=preprocessor.lineage(),
    model=model,
)
```

Notice what the audit builder is given: results, not data to analyse. It records
what the other layers concluded and computes nothing of its own.

## Next

- [The audit artifact](audit-artifact.md) — every field, and what it means
- [Feature lineage](lineage.md) — what happened to each column
- [Safety model](safety-model.md) — what the verdict does and does not claim
