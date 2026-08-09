# AIDatasetKit

A toolkit that takes tabular data from a raw frame to a model-ready dataset:
profiling, statistics, quality checks, capability-driven preprocessing, baselines,
and model comparison.

It is **not** an AutoML system. It measures, warns, prepares, and ranks. It never
edits your data and never picks a model for you — those decisions stay with the
analyst.

## Status

Phase 1 is under construction and this README grows with it. The full guide
(quick start, statistics, profiling, comparison, prediction, and how to add a
model strategy) lands at the end of Phase 1.

| Step | Scope | State |
| --- | --- | --- |
| S0 | Foundation: vocabulary, configuration, validation, provenance | Done, tested |
| S1 | Statistics engine | Not started |
| S2 | Profiling, quality inspection, task detection | Not started |
| S3–S8 | Models, preprocessing, training, evaluation, prediction, facade | Not started |
| S9–S11 | Regression models, optional backends, docs and examples | Not started |

## Install

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Optional model backends are never required:

```bash
.venv/bin/python -m pip install -e ".[boosting]"   # adds xgboost
```

## Run the tests

```bash
.venv/bin/python -m pytest
```

## Architecture in one paragraph

The library is built as strictly ordered layers. `core` defines the shared
vocabulary and depends on nothing internal; `statistics` and `profiling` measure;
`preprocessing` and `models` prepare; `training`, `evaluation`, and `prediction`
run the workflow; `facade` composes them and contains no logic of its own. A
module may only import from a lower layer, and that rule is enforced by a test
rather than by convention — see
[tests/integration/test_architecture_boundaries.py](tests/integration/test_architecture_boundaries.py).

Two decisions shape everything else:

- **Preprocessing follows model capabilities, not model names.** Each model
  declares whether it needs scaling, accepts sparse input, and handles missing
  values natively. Those three flags form a `PreprocessingProfile`, and the
  profile — not the model — is the cache key. Sixteen Phase 1 models collapse into
  four distinct preprocessors, so preparation stays reusable while still adapting
  to each algorithm.
- **The estimator contract is backend-agnostic.** Model strategies return anything
  satisfying the `Estimator` protocol (`fit`, `predict`, `get_params`,
  `set_params`), not `sklearn.base.BaseEstimator`. scikit-learn is today's
  implementation, not the abstraction.

## Licence

MIT.
