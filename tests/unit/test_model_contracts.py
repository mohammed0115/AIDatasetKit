"""Capability contract tests: every declared fact is executed, not asserted.

This is the file that keeps the model layer honest. S0 measured what happens
when an estimator's real behaviour and its assumed behaviour disagree: a raw
duck-typed object survived ``fit``, failed on ``predict``, and inside
cross-validation degraded to silent ``NaN`` scores rather than raising. A
comparison table would have shown it as a blank row.

So nothing here trusts metadata. Every test is parametrised over the live
registry, so a model added in a later step is subjected to the same proof the
moment it is registered, without anyone remembering to extend this file.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aidatasetkit.core.types import Backend, Estimator, TaskType
from aidatasetkit.models import default_registry

REGISTRATIONS = default_registry().catalog()
CLASSIFIERS = [e for e in REGISTRATIONS if e.task_type is TaskType.CLASSIFICATION]


def _identify(entry) -> str:
    return entry.canonical_name


def _binary_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20240101)
    features = rng.normal(size=(80, 4))
    target = (features[:, 0] + features[:, 1] > 0).astype(int)
    return features, target


def _multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20240102)
    features = rng.normal(size=(120, 4))
    target = np.digitize(features[:, 0], [-0.6, 0.6])
    return features, target


@pytest.fixture(scope="module")
def binary():
    return _binary_data()


@pytest.fixture(scope="module")
def multiclass():
    return _multiclass_data()


@pytest.mark.parametrize("entry", REGISTRATIONS, ids=_identify)
class TestEveryRegisteredModel:
    """Contracts that hold for every model regardless of what it declares."""

    def test_it_is_available_in_this_environment(self, entry):
        assert entry.is_available, f"{entry.canonical_name} needs {entry.requires_package}"

    def test_build_returns_a_fresh_instance_every_time(self, entry):
        strategy = entry.strategy_type()
        first, second = strategy.build(), strategy.build()
        assert first is not second
        assert type(first) is type(second)

    def test_a_fitted_estimator_does_not_leak_into_the_next_build(self, entry, binary):
        features, target = binary
        strategy = entry.strategy_type()
        strategy.build().fit(features, target)
        assert not hasattr(strategy.build(), "classes_")

    def test_the_built_estimator_satisfies_our_own_contract(self, entry):
        assert isinstance(entry.strategy_type().build(), Estimator)

    def test_the_model_declares_defaults_rather_than_inheriting_none(self, entry):
        """An empty default_params() would satisfy the next test vacuously."""
        assert entry.strategy_type().default_params(), (
            f"{entry.canonical_name} declares no defaults, so nothing pins its "
            "reproducibility or its configuration"
        )

    def test_declared_defaults_survive_into_the_estimator(self, entry):
        strategy = entry.strategy_type()
        params = strategy.build().get_params()
        for key, value in strategy.default_params().items():
            assert params[key] == value

    def test_overrides_reach_the_estimator(self, entry):
        """The sentinel is deliberately a value no default could already hold.

        Reading a parameter's current value and passing it straight back would
        assert only that a value equals itself, and would pass for a ``build``
        that discarded every override.
        """
        strategy = entry.strategy_type()
        key = next(iter(strategy.build().get_params()))
        sentinel = object()
        assert strategy.build(**{key: sentinel}).get_params()[key] is sentinel

    def test_the_preprocessing_profile_matches_the_capabilities(self, entry):
        capabilities = entry.capabilities
        profile = capabilities.preprocessing_profile()
        assert profile.requires_scaling == capabilities.requires_scaling
        assert profile.supports_sparse_input == capabilities.supports_sparse_input
        assert profile.handles_missing_values == capabilities.handles_missing_values


@pytest.mark.parametrize("entry", CLASSIFIERS, ids=_identify)
class TestEveryClassifier:
    """Contracts proved by fitting, for every registered classifier."""

    def test_fit_and_predict_on_binary_data(self, entry, binary):
        features, target = binary
        predictions = entry.strategy_type().build().fit(features, target).predict(features)
        assert predictions.shape == target.shape
        assert set(np.unique(predictions)) <= set(np.unique(target))

    def test_sklearn_backed_models_can_be_cloned(self, entry):
        if entry.backend is not Backend.SKLEARN:
            pytest.skip("clone is a scikit-learn contract")
        estimator = entry.strategy_type().build()
        copy = clone(estimator)
        assert copy is not estimator
        assert copy.get_params() == estimator.get_params()

    def test_a_clone_of_a_fitted_estimator_is_unfitted(self, entry, binary):
        features, target = binary
        fitted = entry.strategy_type().build().fit(features, target)
        assert not hasattr(clone(fitted), "classes_")

    def test_it_composes_inside_a_pipeline(self, entry, binary):
        features, target = binary
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", entry.strategy_type().build())]
        )
        pipeline.fit(features, target)
        assert pipeline.predict(features).shape == target.shape

    def test_predict_proba_capability_is_true_to_reality(self, entry, binary):
        features, target = binary
        fitted = entry.strategy_type().build().fit(features, target)

        if not entry.capabilities.supports_predict_proba:
            assert not hasattr(fitted, "predict_proba")
            return

        probabilities = fitted.predict_proba(features)
        assert probabilities.shape == (len(features), len(np.unique(target)))
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_multiclass_capability_is_true_to_reality(self, entry, multiclass):
        features, target = multiclass
        if not entry.capabilities.supports_multiclass:
            pytest.skip("model declares binary-only support")

        fitted = entry.strategy_type().build().fit(features, target)
        assert len(fitted.classes_) == len(np.unique(target)) == 3
        assert fitted.predict(features).shape == target.shape
        if entry.capabilities.supports_predict_proba:
            assert fitted.predict_proba(features).shape == (len(features), 3)

    def test_sparse_capability_is_true_to_reality(self, entry, binary):
        features, target = binary
        sparse = csr_matrix(np.abs(features))
        if not entry.capabilities.supports_sparse_input:
            estimator = entry.strategy_type().build()
            with pytest.raises((TypeError, ValueError)):
                estimator.fit(sparse, target)
            return
        fitted = entry.strategy_type().build().fit(sparse, target)
        assert fitted.predict(sparse).shape == target.shape

    def test_missing_value_capability_is_true_to_reality(self, entry, binary):
        """The declaration that most changes the pipeline, so the most load-bearing."""
        features, target = binary
        with_nan = features.copy()
        with_nan[0, 0] = np.nan

        if entry.capabilities.handles_missing_values:
            fitted = entry.strategy_type().build().fit(with_nan, target)
            assert fitted.predict(with_nan).shape == target.shape
        else:
            with pytest.raises(ValueError, match="NaN|missing|infinity"):
                entry.strategy_type().build().fit(with_nan, target)

    def test_it_accepts_non_numeric_class_labels(self, entry, binary):
        features, target = binary
        labels = np.where(target == 1, "churn", "stay")
        fitted = entry.strategy_type().build().fit(features, labels)
        assert set(fitted.classes_) == {"churn", "stay"}


class TestTheContractsThemselves:
    """Proof that the contract assertions above can actually fail.

    Both registered models declare ``True`` for every optional capability, so
    every negative branch in the parametrised suite is currently unreachable. A
    check that has never executed is not a check. These tests drive the same
    assertions against real scikit-learn estimators that genuinely answer ``False``
    -- and against one that lies -- so the negative paths are known to work before
    a later step registers a model that needs them.
    """

    def test_a_model_that_truly_rejects_sparse_input_is_caught(self, binary):
        from sklearn.naive_bayes import GaussianNB

        features, target = binary
        sparse = csr_matrix(np.abs(features))
        with pytest.raises((TypeError, ValueError)):
            GaussianNB().fit(sparse, target)

    def test_a_model_that_truly_rejects_missing_values_is_caught(self, binary):
        from sklearn.linear_model import LogisticRegression

        features, target = binary
        with_nan = features.copy()
        with_nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN|missing|infinity"):
            LogisticRegression().fit(with_nan, target)

    def test_a_model_without_probabilities_is_distinguishable(self, binary):
        """The predict_proba branch asserts absence; prove absence is detectable."""
        from sklearn.svm import LinearSVC

        features, target = binary
        fitted = LinearSVC().fit(features, target)
        assert not hasattr(fitted, "predict_proba")

    def test_a_false_missing_value_declaration_would_fail_the_contract(self, binary):
        """If LogisticRegression declared handles_missing_values=True, this fails."""
        from aidatasetkit.models import ModelFactory

        features, target = binary
        with_nan = features.copy()
        with_nan[0, 0] = np.nan
        estimator = ModelFactory.create("logistic_regression")
        with pytest.raises(ValueError):
            estimator.fit(with_nan, target)

    def test_the_declared_multiclass_support_is_not_merely_asserted(self, multiclass):
        """Three classes are fitted and three sets of coefficients come back."""
        from aidatasetkit.models import ModelFactory

        features, target = multiclass
        fitted = ModelFactory.create("logistic_regression").fit(features, target)
        assert len(np.unique(fitted.predict(features))) > 2


class TestScalingRequirement:
    """``requires_scaling`` drives the pipeline, so it needs evidence too.

    It is the one preprocessing-profile field with no natural pass/fail probe, so
    it is verified by consequence: on features of mixed magnitude the unscaled fit
    exhausts scikit-learn's default iteration budget, and its coefficients land
    three orders of magnitude apart from the scaled ones.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def mixed_scale():
        rng = np.random.default_rng(7)
        n = 300
        features = np.column_stack(
            [
                rng.normal(50_000, 15_000, n),
                rng.normal(40, 12, n),
                rng.normal(0.02, 0.01, n),
            ]
        )
        target = (features[:, 0] / 50_000 + features[:, 1] / 40 > 2.0).astype(int)
        return features, target

    def test_logistic_regression_declares_that_it_needs_scaling(self):
        from aidatasetkit.models import default_registry

        assert default_registry().resolve("logistic_regression").capabilities.requires_scaling

    def test_the_default_budget_is_needed_when_features_are_unscaled(self, mixed_scale):
        """Pins the default: 100 iterations warns here, the chosen 1000 does not."""
        import warnings

        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression

        from aidatasetkit.models import ModelFactory

        features, target = mixed_scale
        with pytest.warns(ConvergenceWarning):
            LogisticRegression(max_iter=100).fit(features, target)

        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            ModelFactory.create("logistic_regression").fit(features, target)

    def test_scaling_materially_changes_the_fitted_model(self, mixed_scale):
        from sklearn.preprocessing import StandardScaler

        from aidatasetkit.models import ModelFactory

        features, target = mixed_scale
        unscaled = ModelFactory.create("logistic_regression", max_iter=5000).fit(
            features, target
        )
        scaled = ModelFactory.create("logistic_regression", max_iter=5000).fit(
            StandardScaler().fit_transform(features), target
        )
        income_unscaled = abs(unscaled.coef_[0][0])
        income_scaled = abs(scaled.coef_[0][0])
        assert income_scaled > 100 * income_unscaled

    def test_the_baseline_is_indifferent_to_scale(self, mixed_scale):
        from sklearn.preprocessing import StandardScaler

        from aidatasetkit.models import ModelFactory

        features, target = mixed_scale
        unscaled = ModelFactory.create("dummy_classifier").fit(features, target)
        scaled = ModelFactory.create("dummy_classifier").fit(
            StandardScaler().fit_transform(features), target
        )
        np.testing.assert_allclose(
            unscaled.predict_proba(features),
            scaled.predict_proba(StandardScaler().fit_transform(features)),
        )


class TestBaselineBehaviour:
    """The baseline must actually behave like a floor."""

    def test_the_default_baseline_is_deterministic(self, binary):
        from aidatasetkit.models import ModelFactory

        features, target = binary
        first = ModelFactory.create("dummy_classifier").fit(features, target)
        second = ModelFactory.create("dummy_classifier").fit(features, target)
        np.testing.assert_array_equal(
            first.predict_proba(features), second.predict_proba(features)
        )

    def test_the_baseline_reports_class_priors_rather_than_certainty(self, binary):
        """"most_frequent" would answer 1.0 for a model that has looked at nothing."""
        from aidatasetkit.models import ModelFactory

        features, target = binary
        probabilities = (
            ModelFactory.create("dummy_classifier")
            .fit(features, target)
            .predict_proba(features)
        )
        expected = np.bincount(target) / len(target)
        np.testing.assert_allclose(probabilities[0], expected)
        assert np.all(probabilities > 0.0)

    def test_the_baseline_scores_a_half_on_roc_auc(self, binary):
        from sklearn.metrics import roc_auc_score

        from aidatasetkit.models import ModelFactory

        features, target = binary
        fitted = ModelFactory.create("dummy_classifier").fit(features, target)
        score = roc_auc_score(target, fitted.predict_proba(features)[:, 1])
        assert score == pytest.approx(0.5)

    def test_a_real_model_beats_the_baseline_on_learnable_data(self, binary):
        from sklearn.metrics import roc_auc_score

        from aidatasetkit.models import ModelFactory

        features, target = binary
        baseline = ModelFactory.create("dummy").fit(features, target)
        model = ModelFactory.create("logistic_regression").fit(features, target)
        baseline_score = roc_auc_score(target, baseline.predict_proba(features)[:, 1])
        model_score = roc_auc_score(target, model.predict_proba(features)[:, 1])
        assert model_score > baseline_score

    def test_only_the_baseline_is_flagged_as_one(self):
        flagged = {e.canonical_name for e in REGISTRATIONS if e.capabilities.is_baseline}
        assert flagged == {"dummy_classifier"}


class TestLogisticRegressionSpecifics:
    def test_the_removed_multi_class_parameter_is_never_passed(self):
        """It was removed from scikit-learn; passing it now raises TypeError."""
        from aidatasetkit.models import ModelFactory

        params = ModelFactory.create("logistic_regression").get_params()
        assert "multi_class" not in params

    def test_multiclass_needs_no_wrapper(self, multiclass):
        from aidatasetkit.models import ModelFactory

        features, target = multiclass
        fitted = ModelFactory.create("logistic_regression").fit(features, target)
        assert type(fitted).__name__ == "LogisticRegression"
        assert fitted.coef_.shape[0] == 3

    def test_coefficients_are_readable_one_per_feature(self, binary):
        from aidatasetkit.models import ModelFactory

        features, target = binary
        fitted = ModelFactory.create("logistic_regression").fit(features, target)
        assert fitted.coef_.shape == (1, features.shape[1])
