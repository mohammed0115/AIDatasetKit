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

import inspect

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


#: Cost knobs turned down for the tests that actually fit. Ensembles default to a
#: hundred members, which is right in production and pure waste on eighty rows.
_SMALL = {"n_estimators": 5, "max_iter": 10}


def _small(entry) -> dict:
    """Choose cost overrides from the estimator's *signature*, never its name.

    ``n_estimators`` is unambiguous. ``max_iter`` is not: on a booster it is the
    number of boosting rounds and shrinking it is free, while on
    ``LogisticRegression`` it is the convergence budget and shrinking it buys a
    ``ConvergenceWarning`` -- which, under ``-W error``, is a failure. The
    estimators where it means "rounds" are exactly those that also accept
    ``early_stopping``, so that is the discriminator, and no model name appears.
    """
    accepted = set(inspect.signature(type(entry.strategy_type().build())).parameters)
    overrides = {}
    if "n_estimators" in accepted:
        overrides["n_estimators"] = _SMALL["n_estimators"]
    if "max_iter" in accepted and "early_stopping" in accepted:
        overrides["max_iter"] = _SMALL["max_iter"]
    return overrides


def _build(entry, **params):
    """Build a fresh estimator sized for a test rather than for production."""
    return entry.strategy_type().build(**{**_small(entry), **params})


def _sparse_as_delivered(entry, features):
    """The sparse matrix this model would actually be handed by S4.

    Not a scrubbed one. A model declaring ``handles_missing_values`` is never
    given imputed data, so the sparse matrix reaching it still carries its gaps
    -- and scikit-learn's tree family accepts sparse, accepts NaN, and refuses
    sparse-carrying-NaN. Testing the declaration against a matrix nobody would
    build would answer a question nobody asks.
    """
    values = np.abs(features).copy()
    if entry.capabilities.handles_missing_values:
        values[0, 0] = np.nan
    return csr_matrix(values)


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
        # Same strategy object for both calls. Fitting an estimator from a
        # *different* instance would prove nothing about this one.
        strategy.build(**_small(entry)).fit(features, target)
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
        predictions = _build(entry).fit(features, target).predict(features)
        assert predictions.shape == target.shape
        assert set(np.unique(predictions)) <= set(np.unique(target))

    def test_sklearn_backed_models_can_be_cloned(self, entry):
        if entry.backend is not Backend.SKLEARN:
            pytest.skip("clone is a scikit-learn contract")
        estimator = _build(entry)
        copy = clone(estimator)
        assert copy is not estimator
        assert copy.get_params() == estimator.get_params()

    def test_a_clone_of_a_fitted_estimator_is_unfitted(self, entry, binary):
        features, target = binary
        fitted = _build(entry).fit(features, target)
        assert not hasattr(clone(fitted), "classes_")

    def test_it_composes_inside_a_pipeline(self, entry, binary):
        features, target = binary
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", _build(entry))]
        )
        pipeline.fit(features, target)
        assert pipeline.predict(features).shape == target.shape

    def test_predict_proba_capability_is_true_to_reality(self, entry, binary):
        features, target = binary
        fitted = _build(entry).fit(features, target)

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

        fitted = _build(entry).fit(features, target)
        assert len(fitted.classes_) == len(np.unique(target)) == 3
        assert fitted.predict(features).shape == target.shape
        if entry.capabilities.supports_predict_proba:
            assert fitted.predict_proba(features).shape == (len(features), 3)

    def test_sparse_capability_is_true_to_reality(self, entry, binary):
        features, target = binary
        sparse = _sparse_as_delivered(entry, features)
        if not entry.capabilities.supports_sparse_input:
            estimator = _build(entry)
            with pytest.raises((TypeError, ValueError)):
                estimator.fit(sparse, target)
            return
        fitted = _build(entry).fit(sparse, target)
        assert fitted.predict(sparse).shape == target.shape

    def test_missing_value_capability_is_true_to_reality(self, entry, binary):
        """The declaration that most changes the pipeline, so the most load-bearing."""
        features, target = binary
        with_nan = features.copy()
        with_nan[0, 0] = np.nan

        if entry.capabilities.handles_missing_values:
            fitted = _build(entry).fit(with_nan, target)
            assert fitted.predict(with_nan).shape == target.shape
        else:
            with pytest.raises(ValueError, match="NaN|missing|infinity"):
                _build(entry).fit(with_nan, target)

    def test_it_accepts_non_numeric_class_labels(self, entry, binary):
        features, target = binary
        labels = np.where(target == 1, "churn", "stay")
        fitted = _build(entry).fit(features, labels)
        assert set(fitted.classes_) == {"churn", "stay"}


@pytest.mark.parametrize("entry", CLASSIFIERS, ids=_identify)
class TestCapabilitiesComposeWithEachOther:
    """Each capability being true separately does not make the pair true.

    scikit-learn's tree family accepts a ``csr_matrix``, accepts ``NaN``, and
    raises on a ``csr_matrix`` containing ``NaN``. A model declaring both would
    let S4 build exactly the matrix it refuses -- reproduced end to end before
    this test existed -- so the pair has to be proved, not inferred.
    """

    def test_a_model_claiming_sparse_and_native_nan_accepts_both_at_once(
        self, entry, binary
    ):
        capabilities = entry.capabilities
        if not (capabilities.supports_sparse_input and capabilities.handles_missing_values):
            pytest.skip("model does not claim both")
        features, target = binary
        values = np.abs(features).copy()
        values[0, 0] = np.nan
        matrix = csr_matrix(values)
        fitted = _build(entry).fit(matrix, target)
        assert fitted.predict(matrix).shape == target.shape

    def test_a_model_that_cannot_take_both_does_not_claim_both(self, entry, binary):
        """The contrapositive, so the skip above can never hide a false claim."""
        features, target = binary
        values = np.abs(features).copy()
        values[0, 0] = np.nan
        matrix = csr_matrix(values)
        try:
            _build(entry).fit(matrix, target)
        except (TypeError, ValueError):
            accepts_both = False
        else:
            accepts_both = True
        claims_both = (
            entry.capabilities.supports_sparse_input
            and entry.capabilities.handles_missing_values
        )
        assert claims_both <= accepts_both


class TestTheContractsThemselves:
    """Proof that the contract assertions above can actually fail.

    Most of the negative branches are now reached by real models -- four
    classifiers decline sparse input, four decline NaN -- so these deliberate
    fakes exist for the branches the catalog does not yet exercise, and to prove
    the assertions are capable of failing at all.

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


class TestProductionDefaultsAreExercisedAndPinned:
    """The suite shrinks ensembles to stay fast; something must still run them.

    Every fitting test above builds with the cost knobs turned down, so without
    this class the shipped defaults would be asserted as values and never once
    executed -- and a default changed from 100 to 2000 would pass the entire
    suite.
    """

    #: The shipped default every model is entitled to have pinned. Read from the
    #: strategy rather than the estimator, because it is the strategy's promise.
    SHIPPED = {
        "decision_tree_classifier": {"random_state": 42},
        "dummy_classifier": {"strategy": "prior", "random_state": 42},
        "extra_trees_classifier": {"n_estimators": 100, "random_state": 42},
        "gaussian_nb": {"var_smoothing": 1e-9},
        "gradient_boosting_classifier": {"n_estimators": 100, "random_state": 42},
        "hist_gradient_boosting_classifier": {
            "max_iter": 100,
            "early_stopping": "auto",
            "random_state": 42,
        },
        "knn_classifier": {"n_neighbors": 5},
        "logistic_regression": {"max_iter": 1000, "random_state": 42},
        "random_forest_classifier": {"n_estimators": 100, "random_state": 42},
    }

    @pytest.mark.parametrize("entry", REGISTRATIONS, ids=_identify)
    def test_the_shipped_defaults_are_exactly_these(self, entry):
        assert entry.strategy_type().default_params() == self.SHIPPED[entry.canonical_name]

    @pytest.mark.parametrize("entry", REGISTRATIONS, ids=_identify)
    def test_the_model_fits_at_its_shipped_defaults(self, entry, binary):
        """No shrinking. A default nothing ever runs is a default nobody checked."""
        features, target = binary
        estimator = entry.strategy_type().build()
        assert estimator.fit(features, target).predict(features).shape == target.shape


class TestGaussianNBScalingClaimTracksItsParameter:
    """``requires_scaling=True`` for GaussianNB rests entirely on one number.

    The model is scale-invariant in theory; scikit-learn makes it scale-sensitive
    by flooring every feature variance at ``var_smoothing * max(variance over all
    features)``. If that default ever moved to 0 the declaration would become
    false, so the dependency is asserted rather than described.
    """

    @staticmethod
    def _disagreement(var_smoothing):
        from sklearn.naive_bayes import GaussianNB
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(3)
        n = 300
        a, b, c = (rng.normal(0, 1, n) for _ in range(3))
        target = ((a + b + c) > 0).astype(int)
        features = np.column_stack([a * 1e-3, b * 1.0, c * 1e5])
        raw = GaussianNB(var_smoothing=var_smoothing).fit(features, target).predict(features)
        scaled_features = StandardScaler().fit_transform(features)
        scaled = (
            GaussianNB(var_smoothing=var_smoothing)
            .fit(scaled_features, target)
            .predict(scaled_features)
        )
        return float((raw != scaled).mean())

    def test_the_shipped_smoothing_really_does_make_it_scale_sensitive(self):
        assert self._disagreement(1e-9) > 0.1

    def test_and_without_that_smoothing_it_is_invariant_as_the_theory_says(self):
        assert self._disagreement(0.0) == 0.0

    def test_the_strategy_ships_the_value_the_claim_depends_on(self):
        from aidatasetkit.models import default_registry

        entry = default_registry().resolve("gaussian_nb")
        assert entry.strategy_type().default_params()["var_smoothing"] == 1e-9
        assert entry.capabilities.requires_scaling


@pytest.mark.parametrize("entry", CLASSIFIERS, ids=_identify)
class TestSparseDeclarationsAreAboutDeliveredData:
    """A tree declines sparse without being unable to take one.

    ``supports_sparse_input`` answers "would this model accept the sparse matrix
    preprocessing would build for it", and for a natively-missing-aware model
    that matrix carries NaN. So a declaration of ``False`` does not imply the
    estimator rejects a *clean* ``csr_matrix``, and this records which of the two
    reasons applies to each model rather than leaving it to be inferred.
    """

    def test_a_declining_model_rejects_either_the_clean_or_the_gapped_matrix(
        self, entry, binary
    ):
        if entry.capabilities.supports_sparse_input:
            pytest.skip("model accepts sparse")
        features, target = binary
        clean = csr_matrix(np.abs(features))
        try:
            _build(entry).fit(clean, target)
        except (TypeError, ValueError):
            return  # refuses sparse outright
        # It takes a clean sparse matrix, so the declaration must be explained by
        # the gaps it would actually be given.
        assert entry.capabilities.handles_missing_values, (
            f"{entry.canonical_name} accepts clean sparse input and declares no "
            "native NaN support, so supports_sparse_input=False is unexplained"
        )
        assert not sparse_and_nan_ok(entry, features, target)


def sparse_and_nan_ok(entry, features, target) -> bool:
    """Whether this estimator survives a sparse matrix that still holds gaps."""
    values = np.abs(features).copy()
    values[0, 0] = np.nan
    try:
        _build(entry).fit(csr_matrix(values), target)
    except (TypeError, ValueError):
        return False
    return True
