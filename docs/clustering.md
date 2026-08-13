# Clustering

Six algorithms, on the same registry, factory and capability-driven preprocessing
as the eighteen supervised models. Nothing here is a second modelling subsystem.

```python
from aidatasetkit import AIDataFacade

ai = AIDataFacade(task="clustering")     # no target, and naming one is refused
ai.load(customers)
ai.check_quality()

result = ai.cluster("kmeans", n_clusters=3)
print(result.n_clusters, result.noise_count, result.cluster_sizes)
print(result.evaluation["silhouette"].value)
```

Or without the facade:

```python
from aidatasetkit.training import ClusteringRunner

result = ClusteringRunner().run(customers, model="dbscan", model_params={"eps": 0.7})
```

## Read this before you read a number

Three limitations are specific to clustering. None of them is a defect waiting to
be fixed; each is a property of the problem.

**The metrics are in-sample.** The partition is scored on the same rows the model
was fitted on. In supervised training that would be the leakage this library
exists to prevent, and the reason it is not the same failure here is narrow:
there is no target, so no score can be inflated by having seen the answer; the
three metrics are *internal*, measuring geometry rather than agreement with
anything held back; and for `dbscan`, `optics` and `agglomerative` there is no
out-of-sample assignment at all, so a held-out set could not be scored even in
principle.

What it does mean is that a high silhouette describes how tidily this model
divided **this** data. It is not evidence that the clusters will reproduce.

**There is no default ranking metric, and the absence is the decision.** All three
metrics reward compact convex clusters. Measured on two interleaved half-moons,
where the correct partition is known:

| Partition | silhouette | davies_bouldin | calinski_harabasz |
|---|---|---|---|
| KMeans, **wrong** | `+0.488` | `0.781` | `440.7` |
| DBSCAN, **correct** | `+0.327` | `1.167` | `193.5` |

All three preferred the wrong answer, unanimously. A default would have ranked
the model that found the real structure last on every one of the three. Nor is a
good score evidence that structure exists at all: 300 rows drawn from a *single*
Gaussian — `default_rng(11)`, no clusters in it whatsoever — scored a silhouette
of `+0.347` at k=3.

So a clustering comparison reports all three and orders by none. Name a metric
explicitly if you have read the above and want an order.

**`n_clusters` is never chosen for you.** `k` is the question being asked, not a
tuning knob, and nothing here runs an elbow or a silhouette sweep to pick it. The
scikit-learn defaults are restated and left alone — they are inherited defaults,
not recommendations, and are almost certainly wrong for any particular dataset.

## The catalogue

| Model | Aliases | Scaling | Sparse | Native NaN | Assigns unseen rows | Interpretability |
|---|---|:-:|:-:|:-:|:-:|---|
| `kmeans_clustering` | `kmeans` | ✓ | ✓ | ✗ | **✓** | high |
| `minibatch_kmeans_clustering` | `minibatch_kmeans` | ✓ | ✓ | ✗ | **✓** | high |
| `birch_clustering` | `birch` | ✓ | ✓ | ✗ | **✓** | medium |
| `dbscan_clustering` | `dbscan` | ✓ | ✓ | ✗ | ✗ | medium |
| `optics_clustering` | `optics` | ✓ | ✗ | ✗ | ✗ | low |
| `agglomerative_clustering` | `agglomerative`, `hierarchical` | ✓ | ✗ | ✗ | ✗ | medium |

Every entry was measured by running the estimator. The six occupy two capability
profiles, both of which already existed, so twenty-four built-in models still
need only five preprocessors between them.

### Every one of them requires scaling

That is six separate measurements that happened to agree, not a family-level
assumption. All six decide membership by a distance, and none of them can know
what a column was measured in. On three separated blobs with one feature recorded
in a unit `1e5` larger — an income beside a ratio — as adjusted Rand index
against the truth:

| Model | ARI unscaled | ARI scaled |
|---|---|---|
| `dbscan` | **`0.000`** | `1.000` |
| `minibatch_kmeans` | `0.631` | `1.000` |
| `agglomerative` | `0.637` | `1.000` |
| `birch` | `0.637` | `1.000` |
| `kmeans` | `0.656` | `1.000` |
| `optics` | `0.051` | `0.061` |

DBSCAN recovered **nothing** unscaled: `eps` is an absolute distance, and every
point was further apart than `eps` along the large-unit axis. This is the starkest
`requires_scaling` measurement in the whole catalogue.

### OPTICS at its defaults is not a drop-in DBSCAN

The last row above is the honest one. OPTICS did not recover the structure scaled
*or* unscaled. On three separated blobs drawn with `default_rng(0)`, and again
with thirty uniform outliers from `default_rng(99)` added:

| Frame | OPTICS | DBSCAN |
|---|---|---|
| 300 clean rows | 13 clusters, 186 noise | 3 clusters, 17 noise |
| 330 rows, 30 outliers | 13 clusters, 215 noise | 3 clusters, 47 noise |

It called well over half the rows noise on data DBSCAN partitioned cleanly.

Nothing corrects this, because correcting it would mean choosing `xi` or
`min_samples` by looking at the data. It is recorded instead. OPTICS earns its
place by not needing a single global `eps` — it can find clusters of differing
density, which DBSCAN cannot — and that strength is worth defaults this weak.

### Half of them cannot label a row they never saw

`KMeans`, `MiniBatchKMeans` and `Birch` keep enough fitted state to place a new
row. `DBSCAN`, `OPTICS` and `AgglomerativeClustering` produce a labelling of the
fitted rows and define no rule for any other row — they have no `predict` at all,
and that is what the algorithms are, not an omission in scikit-learn.

```python
result.assign(new_customers)     # kmeans, minibatch_kmeans, birch
                                 # anything else: TrainingError, with the reason
```

Refitting on the new rows would produce a *different partition*, not an
assignment into this one, so it is refused rather than approximated.

Birch is the row worth reading twice: a hierarchical method that can be deployed
while Agglomerative cannot, because it kept a tree of summaries rather than the
rows.

## Noise is not a cluster

`dbscan` and `optics` label a row `-1` when they decline to assign it. That is not
cluster number −1; it is the algorithm declining. So:

- `-1` rows are dropped before any metric is computed, and the count is reported
  on every metric as `noise_row_count`;
- `result.noise_count` reports it separately from `result.n_clusters`;
- `result.cluster_sizes` never contains a `-1` key.

## When a metric has no value

Every metric comes back with a number or a reason, never a `NaN` and never a
silent omission.

| Situation | Status | Why |
|---|---|---|
| One cluster found | `undefined` | All three compare clusters with each other |
| Every row its own cluster | `undefined` | No within-cluster distance exists |
| Every row labelled noise | `undefined` | There is no cluster to measure |
| Sparse matrix | `not_applicable` | scikit-learn defines two of the three for dense input only |
| Above 10,000 scored rows | `not_applicable` | The silhouette is O(n²); see below |

**The silhouette is reported rather than computed above 10,000 rows.** It needs
every pairwise distance: measured here, 1,000 rows took 0.4s, 5,000 took 9.0s,
and 20,000 took **164 seconds** and a 3.2 GB matrix. It is not approximated by
sampling — a sampled silhouette is not the silhouette, and would differ between
two runs of identical data, which makes it uncomparable. Pass
`silhouette_row_limit` to compute it in full.

**A sparse matrix is not densified** for `davies_bouldin` and `calinski_harabasz`.
It is sparse because a one-hot encoding made it wide, and materialising it could
cost far more memory than the metric is worth. The silhouette accepts sparse
input and is still attempted.

## What is not verified

The readiness verdict. Its thresholds and the leakage checks behind it were built
and measured against classification targets, and a clustering frame has no target
at all. `check_quality()` still runs every check that describes the data alone,
and `BLOCKED` data cannot be clustered through the facade — an infinity wrecks a
distance exactly as it wrecks a coefficient.

See `docs/limitations.md`, which is the canonical list.
