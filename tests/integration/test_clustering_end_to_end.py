"""Clustering through the runner and through the facade, on data with real structure.

Two things are being proven here beyond "it runs". First, that the unsupervised
path reuses the same preprocessing, the same registry and the same safety gate as
the supervised one rather than growing a parallel implementation. Second, that
the places where clustering genuinely differs -- no target, noise labels, and
three models that cannot assign a new row -- are handled explicitly and not by
accident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aidatasetkit import AIDataFacade
from aidatasetkit.core.config import KitConfig
from aidatasetkit.core.exceptions import (
    IncompatibleModelError,
    SchemaError,
    TrainingError,
    WorkflowStateError,
)
from aidatasetkit.core.types import TaskType
from aidatasetkit.evaluation import NOISE_LABEL
from aidatasetkit.training import ClusteringRunner


@pytest.fixture(scope="module")
def customers() -> pd.DataFrame:
    """Three genuinely separated groups, in columns of wildly different units.

    An income in tens of thousands beside a ratio near zero is the ordinary
    frame that makes ``requires_scaling`` load-bearing, so this fixture exercises
    the scaler rather than tiptoeing around it.
    """
    rng = np.random.default_rng(3)
    n = 300
    centres = np.array([[0.0, 0.0], [8.0, 8.0], [-8.0, 7.0]])
    points = centres[rng.integers(0, 3, n)] + rng.normal(0, 0.7, (n, 2))
    return pd.DataFrame(
        {
            "income": points[:, 0] * 10_000 + 60_000,
            "engagement_ratio": points[:, 1] / 20,
            "plan": rng.choice(["basic", "plus", "pro"], n),
        }
    )


class TestTheRunner:
    def test_it_clusters_without_a_target_at_all(self, customers):
        result = ClusteringRunner().run(
            customers, model="kmeans", model_params={"n_clusters": 3}
        )
        assert result.task_type is TaskType.CLUSTERING
        assert result.model_name == "kmeans_clustering"
        assert result.n_clusters == 3
        assert result.labels.shape == (len(customers),)

    def test_every_column_is_a_feature_because_none_is_a_target(self, customers):
        result = ClusteringRunner().run(customers, model="kmeans")
        assert result.feature_count == customers.shape[1]
        assert set(result.lineage) == set(customers.columns)

    def test_the_categorical_column_was_encoded_by_the_shared_planner(self, customers):
        """Preprocessing is the supervised layer's, not a clustering copy of it."""
        result = ClusteringRunner().run(customers, model="kmeans")
        assert result.lineage["plan"] == ("plan_basic", "plan_plus", "plan_pro")
        assert result.transformed_feature_count == 5

    def test_the_labels_are_read_only(self, customers):
        """Mutating them would invalidate every metric already computed."""
        result = ClusteringRunner().run(customers, model="kmeans")
        with pytest.raises(ValueError):
            result.labels[0] = 99

    def test_cluster_sizes_sum_to_the_assigned_rows(self, customers):
        result = ClusteringRunner().run(
            customers, model="kmeans", model_params={"n_clusters": 3}
        )
        assert sum(result.cluster_sizes.values()) == len(customers)
        assert result.noise_count == 0

    def test_it_refuses_an_empty_frame(self, customers):
        with pytest.raises(TrainingError, match="nothing to cluster"):
            ClusteringRunner().run(customers.head(0), model="kmeans")

    def test_it_refuses_a_supervised_model(self, customers):
        with pytest.raises(IncompatibleModelError):
            ClusteringRunner().run(customers, model="logistic_regression")

    def test_the_run_is_reproducible(self, customers):
        first = ClusteringRunner().run(customers, model="kmeans")
        second = ClusteringRunner().run(customers, model="kmeans")
        assert (first.labels == second.labels).all()
        assert first.plan_fingerprint == second.plan_fingerprint

    def test_a_shared_establish_gives_the_same_answer_as_not_sharing(self, customers):
        """The optimisation must not change the result, only the cost."""
        runner = ClusteringRunner()
        shared = runner.establish(customers)
        with_shared = runner.run(customers, model="kmeans", established=shared)
        without = ClusteringRunner().run(customers, model="kmeans")
        assert (with_shared.labels == without.labels).all()

    @pytest.mark.parametrize(
        "model", ["kmeans", "minibatch_kmeans", "dbscan", "optics", "agglomerative", "birch"]
    )
    def test_every_built_in_clusterer_runs_through_it(self, customers, model):
        result = ClusteringRunner().run(customers, model=model)
        assert result.row_count == len(customers)
        assert result.labels.shape == (len(customers),)


class TestOutOfSampleAssignment:
    def test_a_centroid_model_assigns_new_rows(self, customers):
        result = ClusteringRunner().run(
            customers, model="kmeans", model_params={"n_clusters": 3}
        )
        assigned = result.assign(customers.head(5))
        assert assigned.shape == (5,)
        assert set(assigned) <= set(result.cluster_sizes)

    def test_assignment_agrees_with_the_fitted_labelling(self, customers):
        """Otherwise `assign` answers a different question from `labels`."""
        result = ClusteringRunner().run(
            customers, model="kmeans", model_params={"n_clusters": 3}
        )
        assert (result.assign(customers) == result.labels).all()

    def test_the_preprocessor_is_applied_and_never_refitted(self, customers):
        """A refit on the new rows would move the scaler's centre.

        Proven by assigning a deliberately skewed slice: if the scaler were
        refitted on it, the same rows would land in different clusters than they
        did during the fit.
        """
        result = ClusteringRunner().run(
            customers, model="kmeans", model_params={"n_clusters": 3}
        )
        skewed = customers.nlargest(30, "income")
        assert (result.assign(skewed) == result.labels[skewed.index]).all()

    @pytest.mark.parametrize("model", ["dbscan", "optics", "agglomerative"])
    def test_a_model_that_cannot_assign_refuses_rather_than_refitting(
        self, customers, model
    ):
        result = ClusteringRunner().run(customers, model=model)
        assert not result.supports_out_of_sample_assignment
        with pytest.raises(TrainingError, match="cannot assign rows it was not fitted on"):
            result.assign(customers.head(5))


class TestNoiseSurvivesTheWholePath:
    @pytest.fixture(scope="class")
    @staticmethod
    def with_outliers(customers) -> pd.DataFrame:
        rng = np.random.default_rng(17)
        strays = pd.DataFrame(
            {
                "income": rng.uniform(0, 400_000, 25),
                "engagement_ratio": rng.uniform(-5, 5, 25),
                "plan": rng.choice(["basic", "plus", "pro"], 25),
            }
        )
        return pd.concat([customers, strays], ignore_index=True)

    def test_density_clustering_reports_its_noise(self, with_outliers):
        result = ClusteringRunner().run(
            with_outliers, model="dbscan", model_params={"eps": 0.7}
        )
        assert result.noise_count > 0
        assert (result.labels == NOISE_LABEL).sum() == result.noise_count

    def test_noise_is_not_counted_as_a_cluster(self, with_outliers):
        result = ClusteringRunner().run(
            with_outliers, model="dbscan", model_params={"eps": 0.7}
        )
        assert NOISE_LABEL not in result.cluster_sizes
        assert sum(result.cluster_sizes.values()) + result.noise_count == len(
            with_outliers
        )
        assert result.n_clusters == len(result.cluster_sizes)


class TestTheSerialisedRecord:
    def test_it_carries_no_per_row_labels(self, customers):
        """A label per row is data about a record and does not belong in evidence."""
        result = ClusteringRunner().run(customers, model="kmeans")
        payload = result.to_dict()
        assert "labels" not in payload
        import json

        assert str(result.labels.tolist()[:20])[1:-1] not in json.dumps(payload)

    def test_it_carries_no_estimator_or_preprocessor(self, customers):
        result = ClusteringRunner().run(customers, model="kmeans")
        for value in result.to_dict().values():
            assert not hasattr(value, "fit")

    def test_it_is_json_serialisable(self, customers):
        import json

        result = ClusteringRunner().run(customers, model="kmeans")
        assert json.loads(json.dumps(result.to_dict()))["task_type"] == "clustering"

    def test_it_records_the_partition_shape_and_the_capability(self, customers):
        result = ClusteringRunner().run(
            customers, model="dbscan", model_params={"eps": 0.7}
        )
        payload = result.to_dict()
        assert payload["n_clusters"] == result.n_clusters
        assert payload["noise_count"] == result.noise_count
        assert payload["supports_out_of_sample_assignment"] is False


class TestTheFacade:
    def test_a_clustering_session_needs_no_target(self, customers):
        ai = AIDataFacade(task="clustering").load(customers)
        assert ai.task is TaskType.CLUSTERING
        assert ai.status["target"] is None

    def test_a_supervised_session_still_requires_one(self):
        with pytest.raises(SchemaError, match="target column name is required"):
            AIDataFacade()

    def test_naming_a_target_for_clustering_is_refused_not_ignored(self):
        """Ignoring it and clustering on it are different results.

        Neither is what was asked for, so neither is chosen silently.
        """
        with pytest.raises(SchemaError, match="clustering has no target"):
            AIDataFacade(task="clustering", target="income")

    def test_the_whole_sequence_runs(self, customers):
        ai = AIDataFacade(task="clustering").load(customers)
        ai.profile()
        ai.statistics()
        ai.check_quality()
        plans = ai.prepare()
        result = ai.cluster("kmeans", n_clusters=3)
        assert set(plans) == {
            "scaling=1,sparse=1,native_nan=0",
            "scaling=1,sparse=0,native_nan=0",
        }
        assert result.n_clusters == 3
        assert ai.clustering is result

    def test_prepare_still_fits_nothing(self, customers):
        ai = AIDataFacade(task="clustering").load(customers)
        for plan in ai.prepare().values():
            assert not hasattr(plan, "transform")

    def test_it_matches_calling_the_runner_directly(self, customers):
        """The facade is a shortcut, not a second implementation."""
        through_facade = AIDataFacade(task="clustering").load(customers).cluster(
            "kmeans", n_clusters=3
        )
        direct = ClusteringRunner().run(
            customers, model="kmeans", model_params={"n_clusters": 3}
        )
        assert (through_facade.labels == direct.labels).all()
        assert through_facade.plan_fingerprint == direct.plan_fingerprint

    @pytest.mark.parametrize(
        "operation",
        ["compare_models", "select_model", "train", "evaluate", "predict_test"],
    )
    def test_the_supervised_half_is_refused(self, customers, operation):
        ai = AIDataFacade(task="clustering").load(customers)
        method = getattr(ai, operation)
        arguments = ("kmeans",) if operation == "select_model" else ()
        with pytest.raises(WorkflowStateError, match="task='clustering'"):
            method(*arguments)

    def test_clustering_is_refused_on_a_supervised_session(self, customers):
        ai = AIDataFacade(target="income").load(customers)
        with pytest.raises(WorkflowStateError, match="supervised run"):
            ai.cluster("kmeans")

    def test_repeat_clustering_replaces_the_previous_result(self, customers):
        ai = AIDataFacade(task="clustering").load(customers)
        first = ai.cluster("kmeans", n_clusters=3)
        second = ai.cluster("kmeans", n_clusters=4)
        assert ai.clustering is second
        assert first.n_clusters == 3 and second.n_clusters == 4

    def test_a_failed_cluster_call_keeps_the_previous_result(self, customers):
        """Found by adversarial review: the reset ran before the fit.

        ``train`` has always reset only after its fit succeeded -- "a fit that
        raised must not leave a session marked trained" -- and ``cluster`` was
        discarding a perfectly good partition on its way out of a failure.
        """
        ai = AIDataFacade(task="clustering").load(customers)
        good = ai.cluster("kmeans", n_clusters=3)
        with pytest.raises(IncompatibleModelError):
            ai.cluster("logistic_regression")
        assert ai.clustering is good

    def test_reporting_a_clustering_before_running_one_is_refused(self, customers):
        ai = AIDataFacade(task="clustering").load(customers)
        with pytest.raises(WorkflowStateError, match="cluster\\(\\) has not been run"):
            ai.clustering

    def test_blocked_data_cannot_be_clustered_either(self):
        """The safety gate is not optional for the unsupervised half.

        An infinity wrecks a distance exactly as it wrecks a coefficient, and a
        convenience method that skipped the check for clustering would be a
        bypass in all but name.
        """
        frame = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, np.inf] * 25,
                "b": np.arange(100, dtype=float),
            }
        )
        ai = AIDataFacade(task="clustering").load(frame)
        with pytest.raises(WorkflowStateError, match="BLOCKED"):
            ai.cluster("kmeans")

    def test_loading_a_new_frame_discards_the_clustering(self, customers):
        ai = AIDataFacade(task="clustering").load(customers)
        ai.cluster("kmeans")
        ai.load(customers.head(100))
        with pytest.raises(WorkflowStateError):
            ai.clustering


class TestConfigurationIsShared:
    def test_the_seed_reaches_the_estimator(self, customers):
        runner = ClusteringRunner(config=KitConfig(random_state=7))
        assert runner.run(customers, model="kmeans").random_state == 7

    def test_a_model_without_a_seed_records_none_rather_than_inventing_one(
        self, customers
    ):
        result = ClusteringRunner().run(customers, model="dbscan")
        assert result.random_state is None

    def test_an_explicit_override_is_what_gets_recorded(self, customers):
        result = ClusteringRunner().run(
            customers, model="kmeans", model_params={"random_state": 11}
        )
        assert result.random_state == 11
        assert result.model_parameters["random_state"] == 11
