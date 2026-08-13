"""Tests for :class:`ModelCapabilities`."""

from __future__ import annotations

import json

import pytest

from aidatasetkit.core.exceptions import IncompatibleModelError, ValidationError
from aidatasetkit.core.types import (
    Backend,
    Interpretability,
    PreprocessingProfile,
    TargetProfile,
    TaskType,
)
from aidatasetkit.models import ModelCapabilities, default_registry


def capabilities(**overrides) -> ModelCapabilities:
    defaults = {
        "task_type": TaskType.CLASSIFICATION,
        "backend": Backend.SKLEARN,
        "supports_predict_proba": True,
        "requires_scaling": False,
        "supports_sparse_input": True,
        "supports_multiclass": True,
        "handles_missing_values": False,
        "interpretability_level": Interpretability.HIGH,
    }
    return ModelCapabilities(**{**defaults, **overrides})


class TestPreprocessingProfile:
    def test_only_the_three_pipeline_relevant_facts_are_extracted(self):
        profile = capabilities(requires_scaling=True).preprocessing_profile()
        assert profile == PreprocessingProfile(
            requires_scaling=True, supports_sparse_input=True, handles_missing_values=False
        )

    def test_facts_that_do_not_change_the_pipeline_do_not_change_the_profile(self):
        """Interpretability and baseline status are reporting metadata only."""
        first = capabilities(interpretability_level=Interpretability.HIGH, is_baseline=True)
        second = capabilities(interpretability_level=Interpretability.LOW, is_baseline=False)
        assert first.preprocessing_profile() == second.preprocessing_profile()

    @pytest.mark.parametrize("field", ["requires_scaling", "supports_sparse_input"])
    def test_pipeline_relevant_facts_do_change_the_profile(self, field):
        first = capabilities(**{field: True}).preprocessing_profile()
        second = capabilities(**{field: False}).preprocessing_profile()
        assert first != second
        assert first.key != second.key

    def test_the_profile_is_usable_as_a_cache_key(self):
        cache = {capabilities().preprocessing_profile(): "shared"}
        assert cache[capabilities(is_baseline=True).preprocessing_profile()] == "shared"

    def test_every_built_in_model_maps_to_the_documented_profile(self):
        """The full table, so a capability cannot change without saying so here."""
        profiles = {
            entry.canonical_name: entry.capabilities.preprocessing_profile().key
            for entry in default_registry().catalog()
        }
        assert profiles == {
            # The tree family keeps native NaN and gives up sparse: scikit-learn
            # supports each alone and refuses the two together. Both families
            # answer this the same way, and both were measured.
            "decision_tree_classifier": "scaling=0,sparse=0,native_nan=1",
            "extra_trees_classifier": "scaling=0,sparse=0,native_nan=1",
            "random_forest_classifier": "scaling=0,sparse=0,native_nan=1",
            "hist_gradient_boosting_classifier": "scaling=0,sparse=0,native_nan=1",
            "decision_tree_regressor": "scaling=0,sparse=0,native_nan=1",
            "extra_trees_regressor": "scaling=0,sparse=0,native_nan=1",
            "random_forest_regressor": "scaling=0,sparse=0,native_nan=1",
            "hist_gradient_boosting_regressor": "scaling=0,sparse=0,native_nan=1",
            # The baselines genuinely take both, because they read neither.
            "dummy_classifier": "scaling=0,sparse=1,native_nan=1",
            "dummy_regressor": "scaling=0,sparse=1,native_nan=1",
            "gradient_boosting_classifier": "scaling=0,sparse=1,native_nan=0",
            "gradient_boosting_regressor": "scaling=0,sparse=1,native_nan=0",
            "knn_classifier": "scaling=1,sparse=1,native_nan=0",
            "logistic_regression": "scaling=1,sparse=1,native_nan=0",
            "knn_regressor": "scaling=1,sparse=1,native_nan=0",
            # Both linear models ask for scaling, for different reasons: Ridge
            # because the penalty reads the unit a column was recorded in, and
            # ordinary least squares because lstsq truncates a small singular
            # value and drops a real column near a 1e6 magnitude ratio.
            "linear_regression": "scaling=1,sparse=1,native_nan=0",
            "ridge_regression": "scaling=1,sparse=1,native_nan=0",
            "gaussian_nb": "scaling=1,sparse=0,native_nan=0",
            # Every clusterer needs scaling, and none of them accepts a NaN, so
            # they land on two keys that already existed. The split is sparse:
            # OPTICS routes through scipy distance metrics, and Agglomerative
            # needs a dense array for its linkage.
            "kmeans_clustering": "scaling=1,sparse=1,native_nan=0",
            "minibatch_kmeans_clustering": "scaling=1,sparse=1,native_nan=0",
            "dbscan_clustering": "scaling=1,sparse=1,native_nan=0",
            "birch_clustering": "scaling=1,sparse=1,native_nan=0",
            "optics_clustering": "scaling=1,sparse=0,native_nan=0",
            "agglomerative_clustering": "scaling=1,sparse=0,native_nan=0",
        }

    def test_twenty_four_models_need_only_five_preprocessors(self):
        from tests.conftest import BUILT_IN_PROFILE_COUNT

        keys = {
            entry.capabilities.preprocessing_profile()
            for entry in default_registry().catalog()
        }
        assert len(keys) == BUILT_IN_PROFILE_COUNT

    def test_neither_regression_nor_clustering_added_a_profile(self):
        """Fifteen more models, and the number of preprocessors did not move.

        Not an arrangement: it is what happens when the cache key is a capability
        triple. Every profile a regressor or a clusterer needs was already
        required by some classifier, so both families are strict subsets of the
        five. Clustering is the sharper case -- six algorithms with nothing in
        common mechanically, sharing two pipelines with a logistic regression and
        a naive Bayes.
        """
        from aidatasetkit.core.types import TaskType

        by_task: dict[TaskType, set] = {}
        for entry in default_registry().catalog():
            by_task.setdefault(entry.task_type, set()).add(
                entry.capabilities.preprocessing_profile()
            )
        classification = by_task[TaskType.CLASSIFICATION]
        assert len(by_task[TaskType.REGRESSION]) == 4
        assert by_task[TaskType.REGRESSION] < classification
        assert len(by_task[TaskType.CLUSTERING]) == 2
        assert by_task[TaskType.CLUSTERING] < classification

    def test_models_from_different_families_do_share_a_profile(self):
        """The reuse is across task families, not only within one."""
        profiles: dict[str, set[str]] = {}
        for entry in default_registry().catalog():
            profiles.setdefault(
                entry.capabilities.preprocessing_profile().key, set()
            ).add(entry.task_type.value)
        mixed = [key for key, tasks in profiles.items() if len(tasks) > 1]
        # All five, once clustering arrived. Gaussian naive Bayes was the last
        # model sitting alone on a profile, and OPTICS and Agglomerative joined
        # it -- so there is no pipeline in this library built for one family.
        assert len(mixed) == len(profiles) == 5
        # And one of the five serves all three families at once, which is the
        # strongest form of the claim: a single preprocessor for a logistic
        # regression, a ridge, a k-nearest-neighbours and a KMeans.
        assert sum(1 for tasks in profiles.values() if len(tasks) == 3) == 1

    def test_logistic_regression_requires_scaling(self):
        capability = default_registry().resolve("logistic_regression").capabilities
        assert capability.requires_scaling
        assert capability.preprocessing_profile().requires_scaling

    def test_the_baseline_does_not_require_scaling(self):
        capability = default_registry().resolve("dummy_classifier").capabilities
        assert not capability.requires_scaling


class TestInvalidCombinations:
    def test_a_regressor_cannot_claim_class_probabilities(self):
        with pytest.raises(ValidationError, match="supports_predict_proba is meaningless"):
            capabilities(task_type=TaskType.REGRESSION, supports_multiclass=False)

    def test_a_regressor_cannot_claim_multiclass_support(self):
        with pytest.raises(ValidationError, match="supports_multiclass is meaningless"):
            capabilities(
                task_type=TaskType.REGRESSION,
                supports_predict_proba=False,
                supports_multiclass=True,
            )

    def test_a_coherent_regressor_is_accepted(self):
        capability = capabilities(
            task_type=TaskType.REGRESSION,
            supports_predict_proba=False,
            supports_multiclass=False,
        )
        assert capability.task_type is TaskType.REGRESSION

    @pytest.mark.parametrize(
        "task", [TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION, TaskType.DIMENSIONALITY_REDUCTION]
    )
    def test_the_same_rule_applies_to_unsupervised_families(self, task):
        with pytest.raises(ValidationError):
            capabilities(task_type=task)


class TestTargetCompatibility:
    def _target(self, task=TaskType.CLASSIFICATION, n_classes=2) -> TargetProfile:
        return TargetProfile(task_type=task, n_classes=n_classes, is_binary=n_classes == 2)

    def test_a_matching_task_is_accepted(self):
        capabilities().validate_for(self._target())

    def test_a_regression_target_is_refused_before_any_fitting(self):
        with pytest.raises(IncompatibleModelError, match="classification tasks"):
            capabilities().validate_for(self._target(TaskType.REGRESSION, None))

    def test_a_multiclass_target_is_refused_by_a_binary_only_model(self):
        with pytest.raises(IncompatibleModelError, match="two classes only"):
            capabilities(supports_multiclass=False).validate_for(self._target(n_classes=5))

    def test_a_multiclass_target_is_accepted_by_a_multiclass_model(self):
        capabilities(supports_multiclass=True).validate_for(self._target(n_classes=5))

    def test_a_binary_target_is_accepted_by_a_binary_only_model(self):
        capabilities(supports_multiclass=False).validate_for(self._target(n_classes=2))

    def test_compatibility_can_be_asked_without_an_exception(self):
        assert capabilities().is_compatible_with(self._target())
        assert not capabilities().is_compatible_with(self._target(TaskType.REGRESSION, None))

    def test_an_unknown_class_count_does_not_block_a_binary_only_model(self):
        capabilities(supports_multiclass=False).validate_for(self._target(n_classes=None))

    def test_every_classifier_accepts_a_binary_classification_target(self):
        for entry in default_registry().catalog(task=TaskType.CLASSIFICATION):
            entry.capabilities.validate_for(self._target())

    def test_every_classifier_refuses_a_regression_target(self):
        for entry in default_registry().catalog(task=TaskType.CLASSIFICATION):
            with pytest.raises(IncompatibleModelError):
                entry.capabilities.validate_for(self._target(TaskType.REGRESSION, None))

    def test_every_regressor_accepts_a_regression_target(self):
        for entry in default_registry().catalog(task=TaskType.REGRESSION):
            entry.capabilities.validate_for(self._target(TaskType.REGRESSION, None))

    def test_every_regressor_refuses_a_classification_target(self):
        """Task separation is symmetric: neither family can stand in for the other."""
        for entry in default_registry().catalog(task=TaskType.REGRESSION):
            with pytest.raises(IncompatibleModelError, match="regression tasks"):
                entry.capabilities.validate_for(self._target())

    def test_a_regressor_is_not_asked_about_class_counts(self):
        """A many-class target must not trip the binary-only check on a regressor.

        Every regressor declares ``supports_multiclass=False`` because classes
        are meaningless for it. If ``validate_for`` consulted that field outside
        classification, a regression target would be refused for having too many
        distinct values -- which is what a regression target is.
        """
        many = TargetProfile(task_type=TaskType.REGRESSION, n_classes=None)
        for entry in default_registry().catalog(task=TaskType.REGRESSION):
            entry.capabilities.validate_for(many)


class TestSerialisation:
    def test_it_round_trips_through_json(self):
        payload = json.loads(json.dumps(capabilities().to_dict()))
        assert payload["task_type"] == "classification"
        assert payload["backend"] == "sklearn"
        assert payload["interpretability_level"] == "high"
        assert payload["is_baseline"] is False

    def test_the_derived_profile_key_is_included(self):
        payload = capabilities(requires_scaling=True).to_dict()
        assert payload["preprocessing_profile"] == "scaling=1,sparse=1,native_nan=0"

    def test_it_is_immutable(self):
        with pytest.raises(AttributeError):
            capabilities().requires_scaling = True

    def test_equal_capabilities_compare_equal(self):
        assert capabilities() == capabilities()


class TestInterpretability:
    def test_every_model_declares_a_real_level(self):
        for entry in default_registry().catalog():
            assert isinstance(
                entry.capabilities.interpretability_level, Interpretability
            )

    def test_the_catalog_uses_more_than_one_level(self):
        """A field where every model scored the same would be describing nothing."""
        levels = {
            entry.capabilities.interpretability_level
            for entry in default_registry().catalog()
        }
        assert levels == {
            Interpretability.HIGH,
            Interpretability.MEDIUM,
            Interpretability.LOW,
        }

    def test_the_levels_follow_what_a_fitted_model_lets_you_read(self):
        """A tree is rules, a forest is importances, a neighbourhood is neither."""
        levels = {
            entry.canonical_name: entry.capabilities.interpretability_level
            for entry in default_registry().catalog()
        }
        assert levels["decision_tree_classifier"] is Interpretability.HIGH
        assert levels["random_forest_classifier"] is Interpretability.MEDIUM
        assert levels["knn_classifier"] is Interpretability.LOW
        # The same reading, applied to the regressors independently.
        assert levels["linear_regression"] is Interpretability.HIGH
        assert levels["decision_tree_regressor"] is Interpretability.HIGH
        assert levels["random_forest_regressor"] is Interpretability.MEDIUM
        assert levels["hist_gradient_boosting_regressor"] is Interpretability.LOW
        assert levels["knn_regressor"] is Interpretability.LOW

    def test_the_regression_levels_match_what_the_fitted_object_exposes(self):
        """Not a vibe: HIGH means readable coefficients or rules, MEDIUM means
        importances and no decision function, LOW means neither."""
        import numpy as np

        from aidatasetkit.models import ModelFactory

        rng = np.random.default_rng(5)
        features = rng.normal(size=(60, 3))
        target = features[:, 0] * 2 - features[:, 1]

        expectations = {
            "linear_regression": ("coef_", Interpretability.HIGH),
            "ridge_regression": ("coef_", Interpretability.HIGH),
            "decision_tree_regressor": ("feature_importances_", Interpretability.HIGH),
            "random_forest_regressor": ("feature_importances_", Interpretability.MEDIUM),
            "extra_trees_regressor": ("feature_importances_", Interpretability.MEDIUM),
            "gradient_boosting_regressor": (
                "feature_importances_",
                Interpretability.MEDIUM,
            ),
            "hist_gradient_boosting_regressor": (None, Interpretability.LOW),
            "knn_regressor": (None, Interpretability.LOW),
        }
        for name, (attribute, level) in expectations.items():
            overrides = {}
            params = ModelFactory.create(name).get_params()
            if "n_estimators" in params:
                overrides["n_estimators"] = 5
            if "max_iter" in params and "early_stopping" in params:
                overrides["max_iter"] = 10
            fitted = ModelFactory.create(name, **overrides).fit(features, target)
            assert default_registry().resolve(name).capabilities.interpretability_level is level
            if attribute is None:
                assert not hasattr(fitted, "feature_importances_"), name
                assert not hasattr(fitted, "coef_"), name
            else:
                assert hasattr(fitted, attribute), name

    def test_it_is_typed_rather_than_a_bare_string(self):
        assert isinstance(capabilities().interpretability_level, Interpretability)

    def test_an_unknown_level_is_refused(self):
        with pytest.raises(ValidationError):
            Interpretability.coerce("extremely_high")
