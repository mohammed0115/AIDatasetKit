"""The clustering metric policy: three numbers, and no ranking built on them.

The tests that matter most in this file are the ones asserting what the library
*refuses* to do. All three internal metrics can be computed, and none of them is
allowed to become a default ordering, because measurement showed all three
preferring a wrong partition to a correct one.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons

from aidatasetkit.core.exceptions import ValidationError
from aidatasetkit.core.types import TaskType
from aidatasetkit.evaluation import (
    CLUSTERING_METRICS,
    DEFAULT_RANKING_METRIC,
    NOISE_LABEL,
    SILHOUETTE_ROW_LIMIT,
    MetricDirection,
    MetricStatus,
    evaluate_clustering,
    metric_direction,
)


def _blobs(n: int = 300, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [8.0, 8.0], [-8.0, 7.0]])
    return centres[rng.integers(0, 3, n)] + rng.normal(0, 0.7, (n, 2))


def _partition(X: np.ndarray, k: int = 3) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)


class TestTheReportShape:
    def test_all_three_metrics_are_always_attempted(self):
        X = _blobs()
        report = evaluate_clustering(X, _partition(X))
        assert tuple(m.name for m in report.metrics) == CLUSTERING_METRICS

    def test_the_report_is_labelled_clustering_and_carries_no_classes(self):
        X = _blobs()
        report = evaluate_clustering(X, _partition(X))
        assert report.task_type is TaskType.CLUSTERING
        assert report.class_labels is None

    def test_each_metric_carries_its_direction(self):
        assert metric_direction("silhouette") is MetricDirection.HIGHER_IS_BETTER
        assert metric_direction("davies_bouldin") is MetricDirection.LOWER_IS_BETTER
        assert (
            metric_direction("calinski_harabasz") is MetricDirection.HIGHER_IS_BETTER
        )

    def test_every_available_metric_records_the_partition_it_scored(self):
        X = _blobs()
        report = evaluate_clustering(X, _partition(X))
        for metric in report.available:
            assert metric.detail["n_clusters"] == 3
            assert metric.detail["scored_row_count"] == 300
            assert metric.detail["noise_row_count"] == 0

    def test_row_count_is_the_whole_evaluation_set(self):
        X = _blobs()
        report = evaluate_clustering(X, _partition(X))
        assert report.row_count == len(X)

    def test_a_length_mismatch_is_refused_rather_than_zipped(self):
        X = _blobs()
        with pytest.raises(ValidationError, match="labels"):
            evaluate_clustering(X[:10], _partition(X))

    def test_a_two_dimensional_label_array_is_refused_clearly(self):
        """Found by adversarial review: numpy answered with an axis-1 IndexError."""
        X = _blobs(n=20)
        with pytest.raises(ValidationError, match="one label per row"):
            evaluate_clustering(X, np.zeros((20, 2), dtype=int))

    def test_string_labels_are_refused_clearly(self):
        """Found by adversarial review: numpy answered with ``ufunc 'less'``.

        Cluster names are not cluster ids -- two runs would order them
        differently -- so this is refused rather than encoded on the caller's
        behalf.
        """
        X = _blobs(n=20)
        with pytest.raises(ValidationError, match="numeric ids"):
            evaluate_clustering(X, np.array(["a"] * 10 + ["b"] * 10))


class TestThereIsNoDefaultRankingMetricForClustering:
    """The §36 decision, asserted as behaviour rather than left in a docstring."""

    def test_clustering_is_absent_from_the_ranking_defaults(self):
        assert TaskType.CLUSTERING not in DEFAULT_RANKING_METRIC

    def test_the_supervised_defaults_are_untouched(self):
        assert DEFAULT_RANKING_METRIC[TaskType.CLASSIFICATION] == "f1"
        assert DEFAULT_RANKING_METRIC[TaskType.REGRESSION] == "rmse"

    def test_all_three_metrics_prefer_a_wrong_partition_on_non_convex_data(self):
        """The measurement the absence rests on, re-run rather than quoted.

        Two interleaved half-moons. The correct partition is the one DBSCAN
        finds; KMeans cuts them in half the wrong way. If any of the three
        metrics ever stops preferring the wrong answer here, the reasoning
        behind having no default has changed and this should be revisited --
        which is why the assertion is that they *do* prefer it.
        """
        X, truth = make_moons(n_samples=300, noise=0.06, random_state=0)

        wrong = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
        right = DBSCAN(eps=0.25, min_samples=5).fit_predict(X)

        from sklearn.metrics import adjusted_rand_score

        assert adjusted_rand_score(truth, right) > 0.99, "DBSCAN did not find the truth"
        assert adjusted_rand_score(truth, wrong) < 0.5, "KMeans did not get it wrong"

        bad = evaluate_clustering(X, wrong)
        good = evaluate_clustering(X, right)

        assert bad["silhouette"].value > good["silhouette"].value
        # Lower is better, so the wrong partition scoring lower is it winning.
        assert bad["davies_bouldin"].value < good["davies_bouldin"].value
        assert bad["calinski_harabasz"].value > good["calinski_harabasz"].value

    def test_a_respectable_score_is_returned_for_data_with_no_structure(self):
        """A positive silhouette is not evidence that clusters exist.

        The exact value is pinned because it is quoted in
        :data:`DEFAULT_RANKING_METRIC`, in ``docs/clustering.md`` and in the
        changelog. A documented number nothing re-derives is a number that
        quietly stops being true -- which is how an earlier draft of this file
        came to quote a figure measured on an already-advanced generator.
        """
        formless = np.random.default_rng(11).normal(0, 1, (300, 2))
        report = evaluate_clustering(formless, _partition(formless))
        assert round(report["silhouette"].value, 3) == 0.347


class TestNoiseRowsAreExcludedBeforeScoring:
    def test_noise_rows_do_not_take_part(self):
        """Scoring ``-1`` as a cluster would measure the rejects' compactness.

        Proven by construction: the same partition is scored twice, once with
        thirty extra rows marked noise and once with those rows absent
        altogether. The two must agree exactly.
        """
        X = _blobs()
        labels = _partition(X)

        rng = np.random.default_rng(4)
        outliers = rng.uniform(-40, 40, (30, 2))
        widened = np.vstack([X, outliers])
        widened_labels = np.concatenate([labels, np.full(30, NOISE_LABEL)])

        without = evaluate_clustering(X, labels)
        with_noise = evaluate_clustering(widened, widened_labels)

        for name in CLUSTERING_METRICS:
            assert with_noise[name].value == without[name].value

    def test_a_negative_label_that_is_not_the_noise_sentinel_is_refused(self):
        """Found by adversarial review: ``-2`` was counted as an ordinary cluster.

        Only ``-1`` means noise here. Another library's convention, or a
        mistake, would otherwise have pooled a second set of declined rows into
        the score and produced a number that means nothing -- silently, because
        a negative integer is a perfectly valid array element.
        """
        X = _blobs(n=60)
        with pytest.raises(ValidationError, match=r"\[-2\]"):
            evaluate_clustering(X, np.array([-2] * 30 + [3] * 30))

    def test_the_noise_sentinel_itself_is_still_accepted(self):
        """Otherwise the guard above would have banned the one legal negative."""
        X = _blobs(n=60)
        report = evaluate_clustering(
            X, np.array([NOISE_LABEL] * 10 + [0] * 25 + [1] * 25)
        )
        assert report["silhouette"].detail["noise_row_count"] == 10

    def test_the_counts_are_reported_rather_than_hidden(self):
        X = _blobs()
        labels = _partition(X).copy()
        labels[:40] = NOISE_LABEL
        report = evaluate_clustering(X, labels)
        detail = report["silhouette"].detail
        assert detail["noise_row_count"] == 40
        assert detail["scored_row_count"] == 260
        assert report.row_count == 300


class TestDegeneratePartitionsAreUndefinedRatherThanWrong:
    @pytest.mark.parametrize("name", CLUSTERING_METRICS)
    def test_a_single_cluster_has_no_metric(self, name):
        X = _blobs()
        metric = evaluate_clustering(X, np.zeros(len(X), dtype=int))[name]
        assert metric.status is MetricStatus.UNDEFINED
        assert metric.value is None
        assert "1 cluster" in metric.reason

    @pytest.mark.parametrize("name", CLUSTERING_METRICS)
    def test_every_row_in_its_own_cluster_has_no_metric(self, name):
        """scikit-learn raises the same ValueError it raises for one cluster.

        Separated here so the reason distinguishes the two, because they are
        opposite problems and the fix for each is the opposite of the other.
        """
        X = _blobs(n=10)
        metric = evaluate_clustering(X, np.arange(10))[name]
        assert metric.status is MetricStatus.UNDEFINED
        assert "cluster of its own" in metric.reason

    @pytest.mark.parametrize("name", CLUSTERING_METRICS)
    def test_an_all_noise_partition_has_no_metric(self, name):
        X = _blobs()
        metric = evaluate_clustering(X, np.full(len(X), NOISE_LABEL))[name]
        assert metric.status is MetricStatus.UNDEFINED
        assert "labelled noise" in metric.reason

    def test_no_degenerate_case_reports_a_nan(self):
        """The whole point of a status: never a number that means nothing."""
        X = _blobs()
        for labels in (
            np.zeros(len(X), dtype=int),
            np.full(len(X), NOISE_LABEL),
        ):
            for metric in evaluate_clustering(X, labels):
                assert metric.value is None


class TestSparseInputIsHandledPerMetric:
    def test_the_silhouette_is_computed_and_the_other_two_are_not(self):
        """scikit-learn accepts sparse for one of the three and refuses two.

        The two are reported unavailable with the reason rather than densified,
        because a sparse matrix here is usually a wide one-hot encoding.
        """
        X = _blobs()
        report = evaluate_clustering(csr_matrix(np.abs(X)), _partition(X))
        assert report["silhouette"].is_available
        for name in ("davies_bouldin", "calinski_harabasz"):
            metric = report[name]
            assert metric.status is MetricStatus.NOT_APPLICABLE
            assert "dense" in metric.reason

    def test_a_dense_matrix_gets_all_three(self):
        X = _blobs()
        report = evaluate_clustering(X, _partition(X))
        assert len(report.available) == 3


class TestTheSilhouetteRowLimit:
    def test_it_is_reported_rather_than_computed_above_the_limit(self):
        X = _blobs()
        report = evaluate_clustering(X, _partition(X), silhouette_row_limit=100)
        metric = report["silhouette"]
        assert metric.status is MetricStatus.NOT_APPLICABLE
        assert "silhouette_row_limit" in metric.reason

    def test_the_other_two_are_unaffected(self):
        """They are O(n) and O(n*k); only the silhouette needs every pair."""
        X = _blobs()
        report = evaluate_clustering(X, _partition(X), silhouette_row_limit=100)
        assert report["davies_bouldin"].is_available
        assert report["calinski_harabasz"].is_available

    def test_the_reason_states_a_legible_size(self):
        """A 720 KB matrix reported as "0.0 GB" reads as a bug in the message."""
        X = _blobs()
        report = evaluate_clustering(X, _partition(X), silhouette_row_limit=100)
        assert "0.0 GB" not in report["silhouette"].reason

    def test_the_default_limit_leaves_ordinary_data_alone(self):
        assert SILHOUETTE_ROW_LIMIT >= 10_000
        X = _blobs()
        assert evaluate_clustering(X, _partition(X))["silhouette"].is_available

    def test_nothing_is_ever_approximated_by_sampling(self):
        """A sampled silhouette differs between two runs of identical data.

        Asserted through repetition rather than by reading the source: a metric
        that sampled would not return the same number twice.
        """
        X = _blobs()
        labels = _partition(X)
        first = evaluate_clustering(X, labels)["silhouette"].value
        second = evaluate_clustering(X, labels)["silhouette"].value
        assert first == second
