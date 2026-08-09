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

    def test_the_two_built_in_models_map_to_the_documented_profiles(self):
        profiles = {
            entry.canonical_name: entry.capabilities.preprocessing_profile().key
            for entry in default_registry().catalog()
        }
        assert profiles == {
            "dummy_classifier": "scaling=0,sparse=1,native_nan=1",
            "logistic_regression": "scaling=1,sparse=1,native_nan=0",
        }

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

    def test_both_built_in_models_accept_a_binary_classification_target(self):
        for entry in default_registry().catalog():
            entry.capabilities.validate_for(self._target())

    def test_both_built_in_models_refuse_a_regression_target(self):
        for entry in default_registry().catalog():
            with pytest.raises(IncompatibleModelError):
                entry.capabilities.validate_for(self._target(TaskType.REGRESSION, None))


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
    def test_both_proof_models_are_readable(self):
        for entry in default_registry().catalog():
            assert entry.capabilities.interpretability_level is Interpretability.HIGH

    def test_it_is_typed_rather_than_a_bare_string(self):
        assert isinstance(capabilities().interpretability_level, Interpretability)

    def test_an_unknown_level_is_refused(self):
        with pytest.raises(ValidationError):
            Interpretability.coerce("extremely_high")
