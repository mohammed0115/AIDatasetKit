# AIDataFacade

One guided path through the library, for one dataset and one question.

```python
from aidatasetkit import AIDataFacade

ai = AIDataFacade(
    target="Churn",
    id_column="CustomerID",
    task="classification",
    positive_label="churn",
)

ai.load(train_df, test_df)

ai.profile()
ai.statistics()
ai.check_quality()

ai.prepare()

results = ai.compare_models()
results.to_frame()

ai.select_model("logistic_regression")

ai.train()
report = ai.evaluate()

predictions = ai.predict_test()
```

## Philosophy

The facade shortens the syntax of the common path. It does not shorten the
thinking.

Every method delegates to the layer that owns the work — the task detector, the
profiler, the statistics engine, the quality inspector, the planner, the training
layer, the target encoder. Not one line of the facade decides anything about
data. What it owns is *sequence and state*: which results exist, which have gone
stale, and what a step needs before it can run.

That distinction is why there is no `run()`, no `auto()`, and no `best_model()`.
Comparison ranks; choosing is a person's job. A facade that quietly trained rank
one would be teaching that a ranking is an answer rather than a measurement.

## Lifecycle

| method | data-science meaning | fits anything? |
|---|---|---|
| `load()` | data ownership | no |
| `profile()` | understand structure | no |
| `statistics()` | quantitative description | no |
| `check_quality()` | validate the data | no |
| `prepare()` | preparation readiness and planning | **no** |
| `compare_models()` | one fair experiment | yes, per model |
| `select_model()` | a human decision | no |
| `train()` | model fitting | yes |
| `cluster()` | partition unlabelled rows | yes |
| `evaluate()` | measurement on held-out rows | no |
| `predict_test()` | inference | no |

`profile()`, `statistics()` and `check_quality()` are three independent readings
of one loaded frame. Call them in any order, or not at all — they are not a
required sequence, and the facade does not pretend they are.

## What `prepare()` means

**It fits nothing**, and that is the contract rather than a limitation.

Preprocessing depends on model *capabilities*. The eighteen built-in models need
five distinct pipelines between them, so there is no single prepared matrix to
hand back before a model is chosen. More importantly, a preprocessor fitted here
would have seen every row — and the training layer has not drawn its
train/evaluation split yet. Fitting before that split is the most effective way
there is to destroy a model's evaluation, and a convenience method must not be
the thing that does it.

So `prepare()` derives the plan for each distinct capability profile, which
surfaces any planning failure early and records what would happen to each column.
Fitting stays where the split is.

## Clustering

A session created with `task="clustering"` takes **no target**, and naming one is
refused rather than ignored — ignoring the column and clustering on it are two
different results, and neither is what was asked for.

```python
ai = AIDataFacade(task="clustering")
ai.load(customers)
ai.check_quality()
ai.prepare()

result = ai.cluster("kmeans", n_clusters=3)
result.assign(new_customers)      # only for models that can
```

The two halves of the object do not mix. `compare_models`, `select_model`,
`train`, `evaluate`, `predict_test` and `predict` all refuse on a clustering
session, and `cluster()` refuses on a supervised one — because running it there
would quietly cluster on the target column as though it were a feature.

Everything before the modelling step is unchanged: profiling, statistics, quality
and planning all run, minus the checks that need a target. `prepare()` still fits
nothing, and derives its plans from every loaded row rather than from a training
half, because a clustering run draws no split and that is what it will do.

`BLOCKED` data cannot be clustered either. An infinity wrecks a distance exactly
as it wrecks a coefficient, and there is no override here for the same reason
there is none for training.

`cluster()` returns a `ClusteringResult`, and `docs/clustering.md` covers what its
metrics claim — briefly: they are in-sample, they measure geometry rather than
correctness, and there is deliberately no default ranking metric.

## Safety

`check_quality()` runs with the target named, so the target-dependent checks —
leakage above all — stay active. A feature exactly equal to the target is
detected.

**Data the quality checks call `BLOCKED` cannot be trained through the facade.**
There is deliberately no override parameter. The library never had one, because
nothing in it trained until recently, and putting a bypass on the convenient path
makes it the bypass everybody takes. If you have read the findings and intend to
proceed anyway, use `aidatasetkit.training` directly — that route is public,
documented, and requires saying so explicitly.

The verdict and its reasons are available as `ai.verdict` and
`ai.verdict_reasons`, with the full `QualityReport` from `check_quality()`.
`READY`, `READY_WITH_WARNINGS`, `REVIEW_REQUIRED` and `BLOCKED` are preserved as
they are — nothing is reduced to a boolean.

## Comparison

`compare_models()` is a thin call into the training layer. **It is not
hyperparameter optimization**: comparing ten models is exactly ten fits at their
published defaults. It returns the full `ComparisonResult` — outcomes, metrics,
ranking, failures, preprocessing profiles, split context — not a "best model"
string.

**It does not select anything.** A ranking holds under one dataset, one split,
one preprocessing contract, one configuration and one metric. Deciding what to do
about it is yours.

## External test data

Training data and test data have different jobs:

```
training data  →  internal train/evaluation split  →  development and measurement
test data      →  final prediction
```

Test data never takes part in preprocessing fit, model comparison, ranking, or
evaluation. `evaluate()` scores the held-out side of the training split — never
the training rows, never the test frame.

The test frame may or may not carry the target. Inference does not need the
answer.

## Predictions

Classification predictions come back as **the original labels**. A model trained
on `"churn"` and `"stay"` learned integers; handing those back would make you
reverse a mapping the library chose.

Regression predictions are quantities and pass through no encoder at all.

Probabilities, when requested and when the model declares it can produce them,
arrive as a frame whose **columns are the original class labels** — because
`[:, 1]` is the most reliable way there is to report a confident number about the
wrong class.

`id_column` is carried through beside the predictions so an answer can be joined
back to its row. It is not a feature and not an index.

## State and invalidation

Loading a new frame **discards everything** derived from the old one — profile,
statistics, quality, plans, comparison, selection, fit, evaluation, predictions.
A session holding a model trained on the previous frame would answer questions
about data it had never seen.

Selecting a different model discards the previous fit, its evaluation and its
predictions. Running a new comparison discards a selection made from the old one.

Failed operations leave the previous state intact: a load that raises does not
half-replace your data, and a fit that raises does not mark the session trained.

`ai.status` reports what the session holds. It carries no estimator, no
preprocessor and no data.

## The escape hatch

**AIDataFacade is optional.** Everything beneath it is public and composable:

```python
from aidatasetkit.profiling import DataProfiler, DataQualityInspector, TaskDetector
from aidatasetkit.preprocessing import PreprocessingPlanner, PreprocessorBuilder
from aidatasetkit.models import ModelFactory
from aidatasetkit.training import ModelTrainer, ModelComparator, split_rows
from aidatasetkit.evaluation import evaluate_classification, evaluate_regression
from aidatasetkit.prediction import predict_frame
```

Use them directly when you want three preprocessors side by side, the plan
without the fit, or a workflow the facade does not model. The facade must not
replace or deprecate the composable architecture — and for identical inputs the
two routes produce identical results, which the test suite asserts.

## Not in this version

- **Cross-validation.** One train/evaluation split. `compare_models(cv=5)` does
  not exist and does not silently appear.
- **Hyperparameter tuning.** No `tune()`, no search space.
- **Persistence.** No `ai.save()`. A fitted model lives as long as its result.
- **Parallelism.** Correctness first.
- **Automatic model selection.** By design, permanently.

See [limitations.md](limitations.md) and
[training-and-comparison.md](training-and-comparison.md).
