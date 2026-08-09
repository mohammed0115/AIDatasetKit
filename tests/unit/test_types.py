"""Tests for the shared vocabulary.

The estimator-protocol tests are the executable proof of the backend-agnostic
condition: a class that has never heard of scikit-learn satisfies the contract,
which is what will let a PyTorch or TensorFlow strategy plug in later without
touching the facade, the comparator, or any public signature.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from aidatasetkit.core import (
    INTERPRETABILITY_RANK,
    SUPERVISED_TASKS,
    Backend,
    Estimator,
    Interpretability,
    PreprocessingProfile,
    ProbabilisticEstimator,
    RunMetadata,
    Severity,
    TargetProfile,
    TaskType,
)
from aidatasetkit.core.exceptions import ValidationError


class FakeBackendRegressor:
    """A minimal estimator with no scikit-learn ancestry whatsoever."""

    def __init__(self, learning_rate: float = 0.1) -> None:
        self.learning_rate = learning_rate

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def get_params(self, deep: bool = True) -> dict:
        return {"learning_rate": self.learning_rate}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class FakeBackendClassifier(FakeBackendRegressor):
    """The same, plus probabilities."""

    def predict_proba(self, X):
        return np.tile([0.5, 0.5], (len(X), 1))


class ForeignBackendAdapter(BaseEstimator):
    """A stand-in for a future PyTorch or TensorFlow strategy.

    It implements no scikit-learn algorithm. It only adopts the estimator
    conventions -- parameters in ``__init__``, fitted attributes ending in an
    underscore -- which is the same approach taken by established adapters such
    as skorch.
    """

    def __init__(self, learning_rate: float = 0.1) -> None:
        self.learning_rate = learning_rate

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.constant_ = float(np.mean(y)) if y is not None else 0.0
        return self

    def predict(self, X):
        return np.full(len(X), self.constant_)


class NotAnEstimator:
    def fit(self, X, y=None):
        return self


class TestEstimatorProtocol:
    def test_a_non_sklearn_class_satisfies_the_estimator_contract(self):
        assert isinstance(FakeBackendRegressor(), Estimator)

    def test_sklearn_estimators_satisfy_the_same_contract(self):
        for estimator in (LogisticRegression(), LinearRegression(), DecisionTreeRegressor()):
            assert isinstance(estimator, Estimator)

    def test_probabilistic_contract_separates_proba_capable_models(self):
        assert isinstance(FakeBackendClassifier(), ProbabilisticEstimator)
        assert not isinstance(FakeBackendRegressor(), ProbabilisticEstimator)
        assert isinstance(LogisticRegression(), ProbabilisticEstimator)
        assert not isinstance(LinearRegression(), ProbabilisticEstimator)

    def test_an_incomplete_class_does_not_satisfy_the_contract(self):
        assert not isinstance(NotAnEstimator(), Estimator)

    def test_the_contract_does_not_reference_sklearn_base_estimator(self):
        from sklearn.base import BaseEstimator

        assert not issubclass(FakeBackendRegressor, BaseEstimator)
        assert isinstance(FakeBackendRegressor(), Estimator)

    def test_raw_duck_typing_is_not_enough_for_pipeline_composition(self):
        """Documents the real boundary between our contract and scikit-learn's.

        Our protocol is deliberately narrow, but pipeline composition is
        scikit-learn machinery, and since scikit-learn 1.6 that machinery requires
        estimator tags. A raw duck-typed object survives ``fit`` and then fails on
        ``predict``; inside cross-validation it degrades to silent NaN scores
        instead of raising. That is why every strategy must return an adapter, as
        the next test demonstrates.
        """
        features = np.array([[1.0], [2.0], [3.0]])
        target = np.array([1.0, 2.0, 3.0])
        pipeline = Pipeline([("model", FakeBackendRegressor())]).fit(features, target)

        with pytest.raises(AttributeError, match="__sklearn_tags__"):
            pipeline.predict(features)

    def test_a_foreign_backend_adapter_composes_clones_and_cross_validates(self):
        """The proof that a future PyTorch strategy needs no architectural change.

        ``ForeignBackendAdapter`` owns no scikit-learn algorithm; it only borrows
        the estimator conventions. That is all a neural-network strategy will need
        to do, and none of the facade, comparator, evaluator, or public signatures
        are involved.
        """
        pipeline = Pipeline([("model", ForeignBackendAdapter())])
        features = np.arange(20, dtype=float).reshape(-1, 1)
        target = np.arange(20, dtype=float)

        pipeline.fit(features, target)
        assert pipeline.predict(features).shape == (20,)
        assert pipeline.get_params()["model__learning_rate"] == 0.1
        assert isinstance(clone(ForeignBackendAdapter(0.5)), ForeignBackendAdapter)

        scores = cross_val_score(
            Pipeline([("model", ForeignBackendAdapter())]),
            features,
            target,
            cv=3,
            scoring="neg_mean_absolute_error",
        )
        assert scores.shape == (3,)

    def test_the_adapter_also_satisfies_our_own_contract(self):
        assert isinstance(ForeignBackendAdapter(), Estimator)


class TestEnums:
    def test_task_types_cover_the_planned_families(self):
        assert {task.value for task in TaskType} == {
            "classification",
            "regression",
            "clustering",
            "anomaly_detection",
            "dimensionality_reduction",
        }

    def test_only_classification_and_regression_need_a_target(self):
        assert SUPERVISED_TASKS == {TaskType.CLASSIFICATION, TaskType.REGRESSION}

    def test_backends_are_declared_for_future_adapters(self):
        assert {"sklearn", "xgboost", "pytorch", "tensorflow"} <= {b.value for b in Backend}

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("classification", TaskType.CLASSIFICATION),
            ("REGRESSION", TaskType.REGRESSION),
            ("  Anomaly-Detection ", TaskType.ANOMALY_DETECTION),
            ("dimensionality reduction", TaskType.DIMENSIONALITY_REDUCTION),
            (TaskType.CLUSTERING, TaskType.CLUSTERING),
        ],
    )
    def test_coercion_accepts_user_spelling(self, raw, expected):
        assert TaskType.coerce(raw) is expected

    def test_coercion_lists_the_valid_options_on_failure(self):
        with pytest.raises(ValidationError, match="classification, regression"):
            TaskType.coerce("clasification")

    @pytest.mark.parametrize("enum_type", [TaskType, Backend, Interpretability, Severity])
    def test_enums_are_strings_so_reports_serialise_directly(self, enum_type):
        member = next(iter(enum_type))
        assert isinstance(member, str)
        assert json.dumps({"value": member}) == f'{{"value": "{member.value}"}}'

    def test_interpretability_has_a_defined_sort_order(self):
        ranked = sorted(Interpretability, key=INTERPRETABILITY_RANK.__getitem__)
        assert ranked == [
            Interpretability.LOW,
            Interpretability.MEDIUM,
            Interpretability.HIGH,
        ]


class TestPreprocessingProfile:
    def test_equal_capabilities_share_one_cache_key(self):
        first = PreprocessingProfile(True, True, False)
        second = PreprocessingProfile(
            requires_scaling=True, supports_sparse_input=True, handles_missing_values=False
        )
        assert first == second
        assert first.key == second.key
        assert hash(first) == hash(second)

    def test_different_capabilities_produce_different_keys(self):
        keys = {
            PreprocessingProfile(scaling, sparse, nan).key
            for scaling in (True, False)
            for sparse in (True, False)
            for nan in (True, False)
        }
        assert len(keys) == 8

    def test_key_is_stable_and_readable(self):
        profile = PreprocessingProfile(
            requires_scaling=True, supports_sparse_input=False, handles_missing_values=True
        )
        assert profile.key == "scaling=1,sparse=0,native_nan=1"

    def test_profile_is_usable_as_a_dictionary_key(self):
        cache = {PreprocessingProfile(False, True, False): "shared"}
        assert cache[PreprocessingProfile(False, True, False)] == "shared"


class TestTargetProfile:
    def test_serialisation_survives_numpy_labels(self):
        profile = TargetProfile(
            task_type=TaskType.CLASSIFICATION,
            n_classes=2,
            is_binary=True,
            positive_label=np.int64(1),
            class_counts={np.int64(0): 80, np.int64(1): 20},
            imbalance_ratio=0.2,
            detection_note="two distinct integer values",
        )
        payload = json.loads(json.dumps(profile.to_dict()))
        assert payload["task_type"] == "classification"
        assert payload["positive_label"] == 1
        assert payload["class_counts"] == {"0": 80, "1": 20}

    def test_regression_target_has_no_class_information(self):
        profile = TargetProfile(task_type=TaskType.REGRESSION)
        assert profile.n_classes is None
        assert profile.is_binary is False
        assert json.loads(json.dumps(profile.to_dict()))["class_counts"] is None


class TestRunMetadata:
    def _metadata(self, **overrides) -> RunMetadata:
        defaults = {
            "model_name": "logistic_regression",
            "task_type": TaskType.CLASSIFICATION,
            "model_parameters": {"C": 1.0, "max_iter": 1000},
            "random_state": 42,
            "training_rows": 800,
            "feature_count": 12,
            "preprocessing_profile": "scaling=1,sparse=1,native_nan=0",
        }
        return RunMetadata(**{**defaults, **overrides})

    def test_it_records_everything_needed_to_explain_a_result(self):
        payload = self._metadata(training_time_seconds=1.25).to_dict()
        assert payload["model_name"] == "logistic_regression"
        assert payload["task_type"] == "classification"
        assert payload["model_parameters"] == {"C": 1.0, "max_iter": 1000}
        assert payload["random_state"] == 42
        assert payload["training_rows"] == 800
        assert payload["feature_count"] == 12
        assert payload["preprocessing_profile"] == "scaling=1,sparse=1,native_nan=0"
        assert payload["training_time_seconds"] == 1.25

    def test_environment_versions_are_captured_automatically(self):
        environment = self._metadata().to_dict()["environment"]
        assert environment["aidatasetkit"] != ""
        assert set(environment) == {
            "aidatasetkit",
            "python",
            "numpy",
            "pandas",
            "scipy",
            "scikit_learn",
        }

    def test_final_refit_is_recorded_and_defaults_to_false(self):
        assert self._metadata().final_refit_on_full_data is False
        assert self._metadata(final_refit_on_full_data=True).to_dict()[
            "final_refit_on_full_data"
        ]

    def test_awkward_parameter_values_do_not_break_serialisation(self):
        payload = self._metadata(
            model_parameters={
                "estimator": DecisionTreeRegressor(),
                "n_jobs": None,
                "weights": np.float64(0.5),
                "classes": [np.int64(0), np.int64(1)],
                "nested": {"alpha": np.float32(0.25)},
            }
        ).to_dict()
        json.dumps(payload)
        assert payload["model_parameters"]["n_jobs"] is None
        assert payload["model_parameters"]["weights"] == 0.5
        assert payload["model_parameters"]["classes"] == [0, 1]
        assert "DecisionTreeRegressor" in payload["model_parameters"]["estimator"]

    def test_metadata_is_immutable(self):
        with pytest.raises(AttributeError):
            self._metadata().model_name = "other"
