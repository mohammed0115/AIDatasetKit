# Training, evaluation, and model comparison

This page describes what the library does when it trains a model, what it
measures, and what a comparison result does and does not claim.

**It is not hyperparameter optimization.** There is no search, no grid, no
Bayesian anything. Every model is fitted at its published defaults unless you
pass parameters yourself, and comparing ten models is exactly ten fits.

## What training means

```python
from aidatasetkit.training import ModelComparator

result = ModelComparator().compare(frame, target="Churn", positive_label="churn")
result.to_frame()
```

Behind that, in order:

1. The **task detector** resolves the target. A hint is checked against the data,
   not accepted from you.
2. The rows are **split once**, before any model is touched.
3. For each model, its **capabilities** choose a preprocessing plan.
4. The preprocessor is **fitted on the training rows alone**.
5. An **independent estimator** is fitted on the transformed training matrix.
6. The evaluation rows are **transformed, never fitted**, and scored.

The evaluation rows here are the **validation** partition. A comparison ranks on
them, so once it has, they have taken part in choosing the model, and a score on
them afterwards is a development measurement rather than an independent one. The
independent estimate comes from an external test frame the comparison never saw —
see [facade.md](facade.md#external-test-data) for `evaluate_final()`, which takes
it once and then freezes the experiment.

Step 4 is the promise this library exists for. Imputation medians, scaler
centres, and one-hot vocabularies are learned from training rows and from nothing
else. An evaluation row holding an extreme value does not move any of them, and
that is asserted by comparing learned state with and without such a row present —
not by watching which methods were called.

## Fair comparison is not identical preprocessing

Every model in a run sees the **same raw training rows** and is judged on the
**same evaluation rows**. What differs is the preparation each one's capabilities
call for:

| model | scaled | gaps | matrix |
|---|---|---|---|
| `knn_classifier` | yes | imputed | sparse-capable |
| `decision_tree_classifier` | no | left in place | dense |
| `gaussian_nb` | yes | imputed | dense |

Flattening those differences would make the table tidier and less true. A
k-nearest-neighbour model handed unscaled features is not a worse algorithm — it
is a badly prepared one, and recording it as the former would misdescribe what
happened.

Capability-identical models share a preprocessing *blueprint* and never share a
fitted preprocessor. Two models can resolve to one pipeline shape; neither can
see the other's learned median.

## Classification metrics

`accuracy` · `precision` · `recall` · `f1` · `roc_auc`

**The positive class is not class 1.** Once labels become integers,
`predict_proba[:, 1]` is the probability of whichever label sorted second.
`"churn"` sorts before `"stay"`, so for an ordinary churn dataset the event of
interest is column **zero**. Pass `positive_label=` to say which class matters;
without it, precision, recall and F1 fall back to macro averaging (which makes no
claim about which class is interesting) and ROC-AUC reports itself undefined
rather than answering about the wrong class.

**Multiclass uses macro averaging**, stated on every metric it produces. Macro
weights every class equally, which keeps a rare class visible; micro would
collapse to accuracy and hide exactly the failure the metric exists to show.

**Multiclass ROC-AUC is not computed.** It needs an explicit one-vs-rest or
one-vs-one policy and every class present in the evaluation rows. This version
publishes no such policy, and reports `not_applicable` with that reason rather
than picking one silently.

## Regression metrics

`mae` · `mse` · `rmse` · `r2`

RMSE is the root of the mean squared error, computed from it directly so the two
cannot drift apart.

**R² is reported as undefined** when the evaluation target does not vary, or when
there are fewer than two evaluation rows. scikit-learn answers a constant target
with `1.0` for a perfect constant prediction and `0.0` for a wrong one — the same
`0.0` whether the prediction is off by 1.5 or by 892.5. Those are not scores that
describe anything, and all of them would otherwise be ranked.

## Metric direction, and why values are not negated

Every metric carries its direction:

- **higher is better** — accuracy, precision, recall, F1, ROC-AUC, R²
- **lower is better** — MAE, MSE, RMSE

You see `mae = 4.2`, never `neg_mean_absolute_error = -4.2`. The negation is a
convention for optimisers, not something a reader should have to undo.

## When a metric has no value

A missing metric is never silently dropped and never reported as `NaN`. It is
present, with a status and a reason:

| status | means |
|---|---|
| `available` | a number that can be compared |
| `not_applicable` | meaningless for this task or target shape |
| `unsupported_by_model` | the estimator cannot produce the required output |
| `undefined` | mathematically undefined on these rows |
| `failed` | the computation raised |

ROC-AUC absent because a model publishes no probabilities is a fact about the
*model*. Absent because the evaluation rows hold one class is a fact about the
*split*. A single `NaN` would say neither.

## Ranking

The default ranking metric is **`f1`** for classification and **`rmse`** for
regression.

`f1` rather than accuracy: on a 95/5 split, accuracy rewards a model that never
predicts the minority class at all, which is the most common way a comparison
table misleads someone. `rmse` rather than R²: it is in the units of the target,
and it is defined whenever there is at least one evaluation row.

**Ties break alphabetically by canonical name.** Registration order, dictionary
order and hash order are all invisible to a reader and all capable of changing
between runs.

**A model with no value for the ranking metric is not ranked**, and appears in
the result as `unranked` rather than being placed somewhere in the order. A
failed model is likewise absent from the ranking and present in the table.

## The baseline

`dummy_classifier` and `dummy_regressor` answer the question a comparison is
really for: *did this model beat a trivial strategy?* Nothing assumes the
baseline ranks last. A real model may legitimately do worse, and when it does
that fact stays visible.

## What a comparison result claims

Not "random forest is the best model". It claims:

> under **this dataset**, **this split**, **this preprocessing contract**, **this
> configuration** and **this metric**, that model ranked first.

The result carries all six so the sentence can be written out in full.

## Failure

One model failing does not discard the others: it is recorded with its identity
and reason, the run is marked partial, and the rest are ranked normally. If
*every* model fails for the same reason, that is a property of the data rather
than of any model, and it is raised once instead of appearing as nine identical
rows.

Unexpected internal exceptions are not caught. Turning a bug in the orchestration
into "model failed" would send you looking in your dataset for something that was
never there.

## Reproducibility

Fix `random_state` and a run repeats: the same split, the same fits, the same
order. `random_state` must be a plain integer — a generator object is mutable
state, and every draw advances it, so two runs configured identically would not
agree.

Timing is recorded on each result and is deliberately absent from the serialised
record. It is observational, it never decides a ranking, and including it would
make two identical runs compare unequal.

## Not in this version

- **Cross-validation.** A single train/evaluation split is what this stage
  publishes. Cross-validation is future work, not a hidden default.
- **Hyperparameter search** of any kind.
- **Parallel training.** Correctness first; parallelism complicates determinism,
  failure isolation, and testing.
- **Model persistence.** A fitted model lives as long as its result object.
- **Automatic rebalancing or class weighting.** Those are modelling decisions.
  Imbalanced data is evaluated honestly instead: a high accuracy beside a poor
  minority-class F1 stays visible as both numbers.

See [limitations.md](limitations.md) for the canonical list.
