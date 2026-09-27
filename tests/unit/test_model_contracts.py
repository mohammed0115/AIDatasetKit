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

from aidatasetkit.core.types import (
    Backend,
    ClusterEstimator,
    Estimator,
    Fittable,
    TaskType,
)
from aidatasetkit.models import default_registry

REGISTRATIONS = default_registry().catalog()
CLASSIFIERS = [e for e in REGISTRATIONS if e.task_type is TaskType.CLASSIFICATION]
REGRESSORS = [e for e in REGISTRATIONS if e.task_type is TaskType.REGRESSION]


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


def _continuous_data() -> tuple[np.ndarray, np.ndarray]:
    """A genuine quantity, not a label wearing a float dtype.

    The values are non-integral, negative in places, and span a real range, so a
    regressor fitted on them is doing regression rather than counting two
    classes that happen to be spelled 0.0 and 1.0.
    """
    rng = np.random.default_rng(20240103)
    features = rng.normal(size=(80, 4))
    target = 3.0 * features[:, 0] - 2.0 * features[:, 1] + rng.normal(0, 0.1, 80)
    return features, target


@pytest.fixture(scope="module")
def binary():
    return _binary_data()


@pytest.fixture(scope="module")
def multiclass():
    return _multiclass_data()


@pytest.fixture(scope="module")
def continuous():
    return _continuous_data()


def _data_for(entry) -> tuple[np.ndarray, np.ndarray]:
    """Features and a target this model's task can actually be fitted on.

    Chosen from the declared ``task_type``, never from the model's name, so a
    model registered in the wrong family fails these contracts rather than
    quietly being handed the target it wanted.
    """
    if entry.task_type is TaskType.REGRESSION:
        return _continuous_data()
    return _binary_data()


def _fitted_attributes(estimator) -> set[str]:
    """The trailing-underscore attributes a fitted scikit-learn estimator gains."""
    return {
        name
        for name in vars(estimator)
        if name.endswith("_") and not name.startswith("_")
    }


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

    def test_a_fitted_estimator_does_not_leak_into_the_next_build(self, entry):
        """Whatever fitting produced, the next build must not already have it.

        Named attributes were the earlier form of this check, and ``classes_``
        does not exist on a regressor -- so on half the catalog it asserted the
        absence of something that was never going to be there. Comparing the
        attribute sets before and after a fit asks the same question of every
        model, and the first assertion keeps it from passing vacuously if a
        backend ever stops recording its fitted state this way.
        """
        features, target = _data_for(entry)
        strategy = entry.strategy_type()
        # Same strategy object for both calls. Fitting an estimator from a
        # *different* instance would prove nothing about this one.
        fitted = strategy.build(**_small(entry)).fit(features, target)
        learned = _fitted_attributes(fitted)
        assert learned, f"{entry.canonical_name} records no fitted state to leak"
        assert not (_fitted_attributes(strategy.build()) & learned)

    def test_the_built_estimator_satisfies_our_own_contract(self, entry):
        """Every model is Fittable; only some are Estimator, and that is the point.

        S9 split the protocol because three clusterers have no ``predict`` at
        all. Asserting ``Estimator`` for all of them would have been asserting a
        method that is not there -- so the narrow protocol is asserted exactly
        where the capability says it should hold, which makes this test fail if
        either the declaration or the protocol drifts from the other.
        """
        built = entry.strategy_type().build()
        assert isinstance(built, Fittable)

        if entry.task_type is TaskType.CLUSTERING:
            assert isinstance(built, ClusterEstimator)
            expected = entry.capabilities.supports_out_of_sample_assignment
            assert isinstance(built, Estimator) is expected, (
                f"{entry.canonical_name} declares "
                f"supports_out_of_sample_assignment={expected} but "
                f"{'has' if isinstance(built, Estimator) else 'lacks'} predict()"
            )
        else:
            assert isinstance(built, Estimator)
            assert not entry.capabilities.supports_out_of_sample_assignment

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


@pytest.mark.parametrize("entry", REGRESSORS, ids=_identify)
class TestEveryRegressor:
    """Contracts proved by fitting, for every registered regressor.

    The same shape as :class:`TestEveryClassifier`, and deliberately not the same
    assertions: a regression prediction is a quantity, so what has to be checked
    is that it is numeric, one number per row, finite, and free of the
    classification-only surface a regressor must never advertise.
    """

    def test_fit_and_predict_on_a_continuous_target(self, entry, continuous):
        features, target = continuous
        predictions = _build(entry).fit(features, target).predict(features)
        assert predictions.shape == target.shape

    def test_the_predictions_are_numbers_rather_than_labels(self, entry, continuous):
        features, target = continuous
        predictions = _build(entry).fit(features, target).predict(features)
        assert np.issubdtype(predictions.dtype, np.number)
        assert not np.issubdtype(predictions.dtype, np.bool_)

    def test_the_predictions_are_finite(self, entry, continuous):
        features, target = continuous
        predictions = _build(entry).fit(features, target).predict(features)
        assert np.isfinite(predictions).all()

    def test_the_predictions_are_one_dimensional(self, entry, continuous):
        """A column vector would silently broadcast against y in a later metric."""
        features, target = continuous
        predictions = _build(entry).fit(features, target).predict(features)
        assert predictions.ndim == 1

    def test_it_predicts_values_off_the_training_grid(self, entry, continuous):
        """A regressor is not choosing from a fixed set the way a classifier is.

        The baseline is the exception and says so: it predicts one constant. Every
        other model here has to be able to answer with a number, and this is what
        separates genuine regression from a classifier wearing a float dtype.
        """
        features, target = continuous
        predictions = _build(entry).fit(features, target).predict(features)
        distinct = len(np.unique(predictions))
        if entry.capabilities.is_baseline:
            assert distinct == 1
        else:
            assert distinct > 1

    def test_it_does_not_advertise_class_probabilities(self, entry, continuous):
        """The declaration, and the object, must agree that there are no classes."""
        assert entry.capabilities.supports_predict_proba is False
        features, target = continuous
        fitted = _build(entry).fit(features, target)
        assert not hasattr(fitted, "predict_proba")
        assert not hasattr(fitted, "classes_")

    def test_it_does_not_claim_multiclass_support(self, entry):
        assert entry.capabilities.supports_multiclass is False

    def test_sklearn_backed_models_can_be_cloned(self, entry):
        if entry.backend is not Backend.SKLEARN:
            pytest.skip("clone is a scikit-learn contract")
        estimator = _build(entry)
        copy = clone(estimator)
        assert copy is not estimator
        assert copy.get_params() == estimator.get_params()

    def test_a_clone_of_a_fitted_estimator_is_unfitted(self, entry, continuous):
        features, target = continuous
        fitted = _build(entry).fit(features, target)
        assert _fitted_attributes(fitted)
        assert not _fitted_attributes(clone(fitted))

    def test_it_composes_inside_a_pipeline(self, entry, continuous):
        features, target = continuous
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", _build(entry))])
        pipeline.fit(features, target)
        assert pipeline.predict(features).shape == target.shape

    def test_sparse_capability_is_true_to_reality(self, entry, continuous):
        features, target = continuous
        sparse = _sparse_as_delivered(entry, features)
        if not entry.capabilities.supports_sparse_input:
            estimator = _build(entry)
            with pytest.raises((TypeError, ValueError)):
                estimator.fit(sparse, target)
            return
        fitted = _build(entry).fit(sparse, target)
        assert fitted.predict(sparse).shape == target.shape

    def test_missing_value_capability_is_true_to_reality(self, entry, continuous):
        """The declaration that most changes the pipeline, so the most load-bearing."""
        features, target = continuous
        with_nan = features.copy()
        with_nan[0, 0] = np.nan

        if entry.capabilities.handles_missing_values:
            fitted = _build(entry).fit(with_nan, target)
            assert fitted.predict(with_nan).shape == target.shape
        else:
            with pytest.raises(ValueError, match="NaN|missing|infinity"):
                _build(entry).fit(with_nan, target)

    def test_an_entirely_missing_row_still_gets_a_number(self, entry, continuous):
        """Native NaN support has to survive the row with nothing in it."""
        if not entry.capabilities.handles_missing_values:
            pytest.skip("model declares no native NaN support")
        features, target = continuous
        with_nan = features.copy()
        with_nan[:20, 0] = np.nan
        with_nan[3, :] = np.nan
        predictions = _build(entry).fit(with_nan, target).predict(with_nan)
        assert np.isfinite(predictions).all()

    def test_the_caller_data_is_not_modified(self, entry, continuous):
        features, target = continuous
        before_x, before_y = features.copy(), target.copy()
        _build(entry).fit(features, target).predict(features)
        np.testing.assert_array_equal(features, before_x)
        np.testing.assert_array_equal(target, before_y)


@pytest.mark.parametrize("entry", REGISTRATIONS, ids=_identify)
class TestCapabilitiesComposeWithEachOther:
    """Each capability being true separately does not make the pair true.

    scikit-learn's tree family accepts a ``csr_matrix``, accepts ``NaN``, and
    raises on a ``csr_matrix`` containing ``NaN``. A model declaring both would
    let S4 build exactly the matrix it refuses -- reproduced end to end before
    this test existed -- so the pair has to be proved, not inferred.

    Both families are covered. The regression trees answer exactly as the
    classification trees do, and that was measured rather than carried over.
    """

    def test_a_model_claiming_sparse_and_native_nan_accepts_both_at_once(self, entry):
        capabilities = entry.capabilities
        if not (capabilities.supports_sparse_input and capabilities.handles_missing_values):
            pytest.skip("model does not claim both")
        features, target = _data_for(entry)
        values = np.abs(features).copy()
        values[0, 0] = np.nan
        matrix = csr_matrix(values)
        fitted = _build(entry).fit(matrix, target)
        assert fitted.predict(matrix).shape == target.shape

    def test_a_model_that_cannot_take_both_does_not_claim_both(self, entry):
        """The contrapositive, so the skip above can never hide a false claim."""
        features, target = _data_for(entry)
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
        # The canonical name, not the "dummy" alias: since S6 registered a
        # regression baseline under the same alias, the bare form is ambiguous
        # and the registry refuses it rather than picking one.
        baseline = ModelFactory.create("dummy_classifier").fit(features, target)
        model = ModelFactory.create("logistic_regression").fit(features, target)
        baseline_score = roc_auc_score(target, baseline.predict_proba(features)[:, 1])
        model_score = roc_auc_score(target, model.predict_proba(features)[:, 1])
        assert model_score > baseline_score

    def test_only_the_baselines_are_flagged_as_ones(self):
        """One baseline per task family, and nothing else claiming to be a floor."""
        flagged = {e.canonical_name for e in REGISTRATIONS if e.capabilities.is_baseline}
        assert flagged == {"dummy_classifier", "dummy_regressor"}

    def test_each_task_family_has_exactly_one(self):
        for task in (TaskType.CLASSIFICATION, TaskType.REGRESSION):
            flagged = [
                e.canonical_name
                for e in REGISTRATIONS
                if e.task_type is task and e.capabilities.is_baseline
            ]
            assert len(flagged) == 1, f"{task.value}: {flagged}"


class TestLogisticRegressionSpecifics:
    def test_the_removed_multi_class_parameter_is_never_passed(self):
        """This strategy never sets it, on any supported scikit-learn.

        The parameter was deprecated and then removed: scikit-learn 1.8.0 no
        longer has it and passing it raises ``TypeError``, while 1.6.1 through
        1.7.2 still list it in ``get_params()`` with the placeholder value
        ``"deprecated"`` (all measured). The earlier form of this test asserted
        the key was absent from ``get_params()`` -- a property of the installed
        scikit-learn, which failed on 1.7 although this library passes nothing.

        What this library controls is what it resolves and hands over, so that is
        what is asserted: nothing named ``multi_class`` is resolved, and the
        estimator's value is either absent or scikit-learn's untouched default.
        """
        from aidatasetkit.models import ModelFactory, default_registry

        strategy = default_registry().resolve("logistic_regression").strategy_type()
        assert "multi_class" not in strategy.resolve_params()
        params = ModelFactory.create("logistic_regression").get_params()
        assert params.get("multi_class", "deprecated") == "deprecated"

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
        "decision_tree_regressor": {"random_state": 42},
        # No seed: unlike DummyClassifier, DummyRegressor's constructor takes
        # none, and every strategy it offers is deterministic.
        "dummy_regressor": {"strategy": "mean"},
        "extra_trees_regressor": {"n_estimators": 100, "random_state": 42},
        "gradient_boosting_regressor": {"n_estimators": 100, "random_state": 42},
        "hist_gradient_boosting_regressor": {
            "max_iter": 100,
            "early_stopping": "auto",
            "random_state": 42,
        },
        "knn_regressor": {"n_neighbors": 5},
        "linear_regression": {"fit_intercept": True},
        "random_forest_regressor": {"n_estimators": 100, "random_state": 42},
        "ridge_regression": {"alpha": 1.0, "random_state": 42},
        # Clustering. `n_clusters` restates scikit-learn's default in every case
        # and is a placeholder, not a recommendation; `n_init=10` is the one
        # deliberate deviation, measured in the centroid module.
        "kmeans_clustering": {"n_clusters": 8, "n_init": 10, "random_state": 42},
        "minibatch_kmeans_clustering": {
            "n_clusters": 8,
            "n_init": 10,
            "random_state": 42,
        },
        # No seed: neither density constructor accepts one, and both are
        # deterministic given their input.
        "dbscan_clustering": {"eps": 0.5, "min_samples": 5},
        "optics_clustering": {"min_samples": 5},
        "agglomerative_clustering": {
            "n_clusters": 2,
            "linkage": "ward",
            "metric": "euclidean",
        },
        "birch_clustering": {
            "n_clusters": 3,
            "threshold": 0.5,
            "branching_factor": 50,
        },
    }

    @pytest.mark.parametrize("entry", REGISTRATIONS, ids=_identify)
    def test_the_shipped_defaults_are_exactly_these(self, entry):
        assert entry.strategy_type().default_params() == self.SHIPPED[entry.canonical_name]

    def test_the_table_names_every_registered_model_and_nothing_else(self):
        """A model added without a row here would otherwise KeyError, not fail."""
        assert set(self.SHIPPED) == {e.canonical_name for e in REGISTRATIONS}

    @pytest.mark.parametrize("entry", REGISTRATIONS, ids=_identify)
    def test_the_model_fits_at_its_shipped_defaults(self, entry):
        """No shrinking. A default nothing ever runs is a default nobody checked.

        A clusterer is asked for one label per row rather than one prediction
        per target, and through ``fit_predict``, because three of the six have
        no ``predict`` to call afterwards.
        """
        features, target = _data_for(entry)
        estimator = entry.strategy_type().build()
        if entry.task_type is TaskType.CLUSTERING:
            assert estimator.fit_predict(features).shape == (features.shape[0],)
            return
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


@pytest.mark.parametrize("entry", REGISTRATIONS, ids=_identify)
class TestSparseDeclarationsAreAboutDeliveredData:
    """A tree declines sparse without being unable to take one.

    ``supports_sparse_input`` answers "would this model accept the sparse matrix
    preprocessing would build for it", and for a natively-missing-aware model
    that matrix carries NaN. So a declaration of ``False`` does not imply the
    estimator rejects a *clean* ``csr_matrix``, and this records which of the two
    reasons applies to each model rather than leaving it to be inferred.
    """

    def test_a_declining_model_rejects_either_the_clean_or_the_gapped_matrix(self, entry):
        if entry.capabilities.supports_sparse_input:
            pytest.skip("model accepts sparse")
        features, target = _data_for(entry)
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


def _r2(y, predicted) -> float:
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - residual / total


def _shrink(name: str) -> dict:
    """Cost overrides for a model named by string, chosen from its signature.

    The same discriminator :func:`_small` uses, so no test anywhere reads a model
    name to decide how to build it.
    """
    from aidatasetkit.models import ModelFactory

    accepted = set(ModelFactory.create(name).get_params())
    overrides = {}
    if "n_estimators" in accepted:
        overrides["n_estimators"] = _SMALL["n_estimators"]
    if "max_iter" in accepted and "early_stopping" in accepted:
        overrides["max_iter"] = _SMALL["max_iter"]
    return overrides


class TestTheRegressionScalingDeclarations:
    """Both linear models ask for scaling, and the two reasons are different.

    Ridge because its penalty reads the unit a column was recorded in.
    Ordinary least squares because ``scipy.linalg.lstsq``, given ``tol=1e-6`` as
    ``cond``, truncates a small singular value and silently discards a real
    feature. The maths is scale-equivariant; the solver is not.

    An earlier version of this file asserted the opposite for OLS, on the
    strength of the ``customer_shaped`` frame below -- which sits just under the
    truncation threshold and so showed no difference at all. Adversarial review
    supplied an equally ordinary frame that sits above it. Both frames are kept:
    one shows the declaration is not vacuous, the other shows why it is needed.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def customer_shaped():
        """An income near 60,000, a ratio near 0.5, a tenure in months.

        Exactly the shape of the frame the classification integration tests
        already use. Nothing exotic: this is what a customer table looks like.
        """
        rng = np.random.default_rng(20250101)
        n = 240
        income = rng.normal(60_000, 15_000, n).round(2)
        ratio = rng.normal(0.5, 0.05, n).round(4)
        tenure = rng.integers(1, 60, n).astype(float)
        price = (
            0.004 * income + 900.0 * ratio + 3.0 * tenure + rng.normal(0, 5.0, n)
        )
        return np.column_stack([income, ratio, tenure]), price

    @staticmethod
    def _both_ways(name, features, target):
        from aidatasetkit.models import ModelFactory

        raw = ModelFactory.create(name).fit(features, target)
        scaled_features = StandardScaler().fit_transform(features)
        scaled = ModelFactory.create(name).fit(scaled_features, target)
        return _r2(target, raw.predict(features)), _r2(
            target, scaled.predict(scaled_features)
        )

    def test_ridge_declares_that_it_needs_scaling(self):
        from aidatasetkit.models import default_registry

        assert default_registry().resolve("ridge_regression").capabilities.requires_scaling

    def test_and_the_penalty_really_does_cost_it_accuracy_unscaled(self, customer_shaped):
        """Measured 0.90 against 0.997 on this frame, and it is not marginal."""
        features, target = customer_shaped
        raw, scaled = self._both_ways("ridge_regression", features, target)
        assert scaled - raw > 0.05

    def test_the_penalty_is_what_does_it_rather_than_the_conditioning(self):
        """A well-conditioned matrix, so only the penalty can be responsible.

        Ordinary least squares is scale-equivariant: multiply a column by k and
        its coefficient divides by k. Ridge is not, because ``alpha`` shrinks a
        coefficient whose size depends on the unit its feature was recorded in.
        """
        from sklearn.linear_model import LinearRegression, Ridge

        rng = np.random.default_rng(21)
        n = 200
        features = rng.normal(0, 1, size=(n, 3))
        target = features @ np.array([1.0, 2.0, -1.5]) + rng.normal(0, 0.3, n)

        scales = np.array([1.0, 1.0, 1000.0])
        stretched = features * scales
        assert np.linalg.cond(stretched) < 1e4  # nothing ill-conditioned here

        ols_small = LinearRegression().fit(features, target).coef_
        ols_large = LinearRegression().fit(stretched, target).coef_ * scales
        np.testing.assert_allclose(ols_small, ols_large, rtol=1e-6)

        ridge_small = Ridge(alpha=1.0).fit(features, target).coef_
        ridge_large = Ridge(alpha=1.0).fit(stretched, target).coef_ * scales
        assert not np.allclose(ridge_small, ridge_large, rtol=1e-6)

    def test_linear_regression_declares_that_it_needs_scaling_too(self):
        from aidatasetkit.models import default_registry

        capability = default_registry().resolve("linear_regression").capabilities
        assert capability.requires_scaling

    def test_the_maths_alone_would_not_require_it(self, customer_shaped):
        """Kept deliberately: the declaration is about the solver, not the model.

        On a frame below the truncation threshold, raw and standardised OLS fits
        are identical to every digit. If the reason were the algebra, this would
        fail -- and someone reading ``requires_scaling=True`` would draw the
        wrong conclusion about why.
        """
        features, target = customer_shaped
        raw, scaled = self._both_ways("linear_regression", features, target)
        assert raw == pytest.approx(scaled, abs=1e-9)

    @pytest.fixture(scope="class")
    @staticmethod
    def dollars_and_a_rate():
        """Revenue in dollars beside a conversion rate. An ordinary pair.

        Their standard deviations differ by about 6.6e7, which is all it takes.
        Nothing here is contrived: any table pairing a currency amount with a
        proportion looks like this.
        """
        rng = np.random.default_rng(4)
        n = 500
        revenue = rng.normal(2_000_000, 400_000, n)
        rate = rng.normal(0.03, 0.006, n)
        target = 1e-6 * revenue + 50.0 * rate + rng.normal(0, 1e-3, n)
        return np.column_stack([revenue, rate]), target

    def test_the_solver_silently_discards_a_real_column_unscaled(
        self, dollars_and_a_rate
    ):
        """What the unscaled solve does, for both behaviours scikit-learn has had.

        Measured on scikit-learn 1.9.0: ``rank_`` drops to 1 and the rate's
        coefficient comes back as numerical zero -- not an accuracy wobble, a
        smaller model, with nothing raised. Measured on 1.5.2 through 1.8.0: the
        same solve keeps full rank, and the fit is scale-equivariant.

        Neither behaviour is assumed. Whichever one the installed version has,
        its consequences are asserted, so a solver change is noticed rather than
        silently absorbed -- and the declaration's protection is asserted
        unconditionally by the next test.
        """
        from aidatasetkit.models import ModelFactory

        features, target = dollars_and_a_rate
        fitted = ModelFactory.create("linear_regression").fit(features, target)
        if fitted.rank_ < features.shape[1]:
            assert fitted.rank_ == 1
            assert abs(fitted.coef_[1]) < 1e-12, "the rate coefficient should be zero"
            assert _r2(target, fitted.predict(features)) < 0.7
        else:
            scaled_features = StandardScaler().fit_transform(features)
            scaled = ModelFactory.create("linear_regression").fit(scaled_features, target)
            np.testing.assert_allclose(
                fitted.predict(features), scaled.predict(scaled_features), rtol=1e-6
            )

    def test_and_the_scaler_the_declaration_asks_for_recovers_it(
        self, dollars_and_a_rate
    ):
        from aidatasetkit.models import ModelFactory

        features, target = dollars_and_a_rate
        scaled = StandardScaler().fit_transform(features)
        fitted = ModelFactory.create("linear_regression").fit(scaled, target)
        assert fitted.rank_ == 2
        assert _r2(target, fitted.predict(scaled)) > 0.99

    def test_the_truncation_threshold_is_where_it_is_claimed_to_be(self):
        """Sweeping the spread: identical below ~1e6; above it, version-dependent.

        Below the threshold every measured scikit-learn keeps full rank. Above it
        scikit-learn 1.9.0 truncates (``rank_`` 1, R2 below 0.9) and 1.5.2
        through 1.8.0 do not. The ordinary case is asserted unconditionally; the
        extreme case asserts the consequences of whichever behaviour is installed.
        """
        from aidatasetkit.models import ModelFactory

        rng = np.random.default_rng(31)
        n = 400
        first, second = rng.normal(0, 1, n), rng.normal(0, 1, n)
        target = first * 2.0 + second * 3.0 + rng.normal(0, 0.1, n)

        ordinary = np.column_stack([first, second * 1e3])
        fitted = ModelFactory.create("linear_regression").fit(ordinary, target)
        assert fitted.rank_ == 2
        assert _r2(target, fitted.predict(ordinary)) > 0.99

        extreme = np.column_stack([first, second * 1e8])
        solved = ModelFactory.create("linear_regression").fit(extreme, target)
        if solved.rank_ < 2:
            assert solved.rank_ == 1
            assert _r2(target, solved.predict(extreme)) < 0.9
        else:
            assert _r2(target, solved.predict(extreme)) > 0.99

    def test_knn_needs_scaling_because_a_distance_has_units(self):
        from aidatasetkit.models import default_registry

        assert default_registry().resolve("knn_regressor").capabilities.requires_scaling

    def test_and_the_neighbourhood_really_does_change(self):
        rng = np.random.default_rng(7)
        n = 300
        a, b, c = (rng.normal(0, 1, n) for _ in range(3))
        target = a + b + c + rng.normal(0, 0.05, n)
        features = np.column_stack([a * 1e-3, b * 1.0, c * 1e5])
        raw, scaled = self._both_ways("knn_regressor", features, target)
        assert scaled - raw > 0.3

    def test_the_tree_family_is_genuinely_indifferent(self):
        """The negative branch: scaling must not be claimed where it changes nothing."""
        from aidatasetkit.models import ModelFactory, default_registry

        rng = np.random.default_rng(7)
        n = 300
        a, b, c = (rng.normal(0, 1, n) for _ in range(3))
        target = a + b + c
        features = np.column_stack([a * 1e-3, b * 1.0, c * 1e5])
        scaled_features = StandardScaler().fit_transform(features)
        for name in ("decision_tree_regressor", "random_forest_regressor"):
            assert not default_registry().resolve(name).capabilities.requires_scaling
            raw_pred = (
                ModelFactory.create(name, **_shrink(name))
                .fit(features, target)
                .predict(features)
            )
            scaled_pred = (
                ModelFactory.create(name, **_shrink(name))
                .fit(scaled_features, target)
                .predict(scaled_features)
            )
            np.testing.assert_allclose(raw_pred, scaled_pred)


class TestRandomnessIsMeasuredRatherThanAssumed:
    """Which seeds are load-bearing, proved on data where randomness can bite.

    A noise-free frame that every model fits perfectly would show four seeds
    agreeing and prove nothing. These use noisy, correlated features and score on
    held-out rows, so a model whose fit genuinely depends on its seed cannot hide.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def adversarial():
        rng = np.random.default_rng(2024)
        n = 400
        base = rng.normal(size=(n, 6))
        features = np.column_stack([base, base[:, 0] + rng.normal(0, 0.01, n)])
        target = base[:, 0] * 2 + base[:, 1] - base[:, 2] + rng.normal(0, 1.5, n)
        return features[:300], target[:300], features[300:]

    #: Measured **at the shipped defaults**, one entry per regressor that accepts
    #: a seed. The value is the smallest held-out spread across four seeds that
    #: the model must produce, in units of the target -- not a boolean, because a
    #: boolean is what let an earlier version of this table pass on a difference
    #: of ``3.3e-16``.
    #:
    #: That earlier version compared ``ndarray.tobytes()``, so two fits agreeing
    #: to the last mantissa bit counted as "different" and the class whose stated
    #: purpose is to avoid a vacuous determinism check was vacuous for gradient
    #: boosting -- whose five shrunken stages agree to floating-point noise.
    #: Requiring a spread the target's own scale can notice is what fixes it.
    #:
    #: ``0.0`` means the seed is genuinely inert at the shipped configuration.
    LOAD_BEARING = {
        "decision_tree_regressor": 1.0,
        "extra_trees_regressor": 0.3,
        "gradient_boosting_regressor": 0.1,
        "random_forest_regressor": 0.3,
        # Inert at the published solver="auto", which resolves to a deterministic
        # route. Published anyway so that switching to sag or saga -- which do
        # draw on it -- stays reproducible.
        "ridge_regression": 0.0,
        # Inert *at this row count*. Above scikit-learn's 10,000-row threshold
        # early stopping engages, the seed picks the validation split, and it
        # becomes load-bearing -- see the dedicated test below. Recording it as
        # an unconditional constant would have been a claim about a property that
        # depends on the data.
        "hist_gradient_boosting_regressor": 0.0,
    }

    @staticmethod
    def _predict(name, seed, adversarial, shrink=False):
        """Fit at the shipped defaults unless a test explicitly asks otherwise.

        Shrinking is what made the old version of this check vacuous, so it is
        off by default here even though it costs time: a reproducibility claim
        measured at a configuration nobody ships is not a claim about the
        library.
        """
        from aidatasetkit.models import ModelFactory

        train_x, train_y, test_x = adversarial
        overrides = _shrink(name) if shrink else {}
        estimator = ModelFactory.create(name, random_state=seed, **overrides)
        return estimator.fit(train_x, train_y).predict(test_x)

    @pytest.mark.parametrize("name", sorted(LOAD_BEARING))
    def test_the_same_seed_reproduces_the_same_model(self, name, adversarial):
        first = self._predict(name, 0, adversarial)
        np.testing.assert_array_equal(first, self._predict(name, 0, adversarial))

    @pytest.mark.parametrize("name", sorted(LOAD_BEARING))
    def test_whether_the_seed_matters_is_recorded_correctly(self, name, adversarial):
        """Measured as a spread, not as "the bytes differ"."""
        predictions = [self._predict(name, seed, adversarial) for seed in (0, 1, 7, 999)]
        spread = max(float(np.max(np.abs(p - predictions[0]))) for p in predictions)
        required = self.LOAD_BEARING[name]
        if required:
            assert spread >= required, (
                f"{name}: four seeds moved the held-out predictions by only "
                f"{spread:.3g}, below the {required} this table records. A seed "
                "that changes nothing measurable is not load-bearing."
            )
        else:
            assert spread == 0.0, (
                f"{name}: this table records the seed as inert, but four seeds "
                f"moved the held-out predictions by {spread:.3g}."
            )

    def test_the_old_byte_comparison_would_have_passed_on_nothing(self, adversarial):
        """The regression test for the vacuity, kept as evidence.

        At five boosting stages the four seeds agree to floating-point noise. The
        superseded assertion -- distinct ``tobytes()`` -- passes here; the spread
        assertion above does not, which is the whole point of the change.
        """
        shrunken = [
            self._predict("gradient_boosting_regressor", seed, adversarial, shrink=True)
            for seed in (0, 1, 7, 999)
        ]
        spread = max(float(np.max(np.abs(p - shrunken[0]))) for p in shrunken)
        assert len({p.tobytes() for p in shrunken}) == 4  # the old check passes
        assert spread < 1e-12  # ...on a difference of nothing

    def test_the_hist_booster_seed_becomes_load_bearing_above_ten_thousand_rows(self):
        """The data-dependent half of that model's entry, measured rather than assumed.

        ``early_stopping="auto"`` turns itself on above 10,000 training rows and
        withholds a tenth as a validation set, chosen with ``random_state``.
        Different seeds then stop at different iterations and fit different
        models. Below the threshold the seed is genuinely inert.
        """
        from aidatasetkit.models import ModelFactory

        rng = np.random.default_rng(15)
        train_rows = 10_500
        total = train_rows + 300
        features = rng.normal(size=(total, 4))
        target = features[:, 0] * 2 + rng.normal(0, 1.0, total)

        fitted = {
            seed: ModelFactory.create("hist_gradient_boosting_regressor", random_state=seed)
            .fit(features[:train_rows], target[:train_rows])
            for seed in (0, 1, 7)
        }
        assert all(model.do_early_stopping_ for model in fitted.values())
        assert len({model.n_iter_ for model in fitted.values()}) > 1

        predictions = [model.predict(features[train_rows:]) for model in fitted.values()]
        spread = max(float(np.max(np.abs(p - predictions[0]))) for p in predictions)
        assert spread > 0.1

    def test_every_seeded_regressor_appears_in_that_table(self):
        seeded = {
            entry.canonical_name
            for entry in REGRESSORS
            if "random_state" in entry.strategy_type().default_params()
        }
        assert seeded == set(self.LOAD_BEARING)

    @pytest.mark.parametrize(
        "name", ["dummy_regressor", "linear_regression", "knn_regressor"]
    )
    def test_a_seedless_model_is_not_given_an_invented_seed(self, name):
        from aidatasetkit.models import ModelFactory

        assert "random_state" not in ModelFactory.create(name).get_params()
        assert "random_state" not in ModelFactory.strategy(name).default_params()

    @pytest.mark.parametrize(
        "name", ["dummy_regressor", "linear_regression", "knn_regressor"]
    )
    def test_and_is_deterministic_without_one(self, name, adversarial):
        from aidatasetkit.models import ModelFactory

        train_x, train_y, test_x = adversarial
        first = ModelFactory.create(name).fit(train_x, train_y).predict(test_x)
        second = ModelFactory.create(name).fit(train_x, train_y).predict(test_x)
        np.testing.assert_array_equal(first, second)


class TestMultiOutputIsMeasuredRatherThanAssumed:
    """Two continuous outputs: seven regressors take them, two do not.

    ``ModelCapabilities`` has no multi-output field and S6 does not add one.
    Nothing in this library produces a two-column target, so a capability nothing
    consults would be a contract with no reader. What is recorded instead is the
    measurement, and the split -- both boosters refuse, everything else accepts --
    is exact and will fail here if a backend changes it.
    """

    ACCEPTS = {
        "decision_tree_regressor",
        "dummy_regressor",
        "extra_trees_regressor",
        "knn_regressor",
        "linear_regression",
        "random_forest_regressor",
        "ridge_regression",
    }
    REFUSES = {"gradient_boosting_regressor", "hist_gradient_boosting_regressor"}

    @staticmethod
    def _two_column_target(continuous):
        features, target = continuous
        return features, np.column_stack([target, -0.5 * target + 1.0])

    def test_the_two_sets_together_are_the_whole_catalog(self):
        assert self.ACCEPTS | self.REFUSES == {e.canonical_name for e in REGRESSORS}
        assert not (self.ACCEPTS & self.REFUSES)

    @pytest.mark.parametrize("name", sorted(ACCEPTS))
    def test_an_accepting_model_returns_one_column_per_output(self, name, continuous):
        from aidatasetkit.models import ModelFactory

        features, target = self._two_column_target(continuous)
        estimator = ModelFactory.create(name, **_shrink(name))
        predictions = estimator.fit(features, target).predict(features)
        assert predictions.shape == target.shape

    @pytest.mark.parametrize("name", sorted(REFUSES))
    def test_a_refusing_model_says_so_rather_than_silently_flattening(
        self, name, continuous
    ):
        from aidatasetkit.models import ModelFactory

        features, target = self._two_column_target(continuous)
        with pytest.raises(ValueError, match="1d array|column-vector|shape"):
            ModelFactory.create(name, **_shrink(name)).fit(features, target)


class TestTheRegressionBaseline:
    """The baseline must actually behave like a floor."""

    def test_it_predicts_the_training_mean(self, continuous):
        from aidatasetkit.models import ModelFactory

        features, target = continuous
        predictions = (
            ModelFactory.create("dummy_regressor").fit(features, target).predict(features)
        )
        np.testing.assert_allclose(predictions, target.mean())

    def test_it_is_deterministic(self, continuous):
        from aidatasetkit.models import ModelFactory

        features, target = continuous
        first = ModelFactory.create("dummy_regressor").fit(features, target)
        second = ModelFactory.create("dummy_regressor").fit(features, target)
        np.testing.assert_array_equal(
            first.predict(features), second.predict(features)
        )

    def test_it_scores_zero_on_r_squared_which_is_the_correct_floor(self, continuous):
        features, target = continuous
        predictions = np.full_like(target, target.mean())
        assert _r2(target, predictions) == pytest.approx(0.0)

    def test_a_real_model_beats_it_on_learnable_data(self, continuous):
        from aidatasetkit.models import ModelFactory

        features, target = continuous
        baseline = ModelFactory.create("dummy_regressor").fit(features, target)
        model = ModelFactory.create("linear_regression").fit(features, target)
        assert _r2(target, model.predict(features)) > _r2(
            target, baseline.predict(features)
        )

    def test_it_survives_a_sparse_matrix_holding_gaps_because_it_reads_nothing(
        self, continuous
    ):
        from aidatasetkit.models import ModelFactory

        features, target = continuous
        values = np.abs(features).copy()
        values[0, 0] = np.nan
        matrix = csr_matrix(values)
        fitted = ModelFactory.create("dummy_regressor").fit(matrix, target)
        assert fitted.predict(matrix).shape == target.shape


class TestTheRegressionCapabilityTestsCanActuallyFail:
    """Mutation checks: a wrong declaration must break something.

    A capability suite that passed whatever the metadata said would be
    decoration. Each test here takes a real estimator and asserts the *opposite*
    of what the catalog declares, proving the corresponding contract assertion
    has something to catch.
    """

    def test_claiming_ridge_needs_no_scaling_would_be_a_measurable_lie(self):
        """The assertion behind ``requires_scaling=True`` for Ridge."""
        from sklearn.linear_model import Ridge

        rng = np.random.default_rng(20250101)
        n = 240
        income = rng.normal(60_000, 15_000, n)
        ratio = rng.normal(0.5, 0.05, n)
        target = 0.004 * income + 900.0 * ratio + rng.normal(0, 5.0, n)
        features = np.column_stack([income, ratio])
        raw = _r2(target, Ridge().fit(features, target).predict(features))
        scaled_features = StandardScaler().fit_transform(features)
        scaled = _r2(target, Ridge().fit(scaled_features, target).predict(scaled_features))
        assert scaled - raw > 0.05, "if this ever holds, the declaration is wrong"

    def test_claiming_linear_regression_needs_no_scaling_would_be_a_measurable_lie(self):
        """The declaration adversarial review corrected, pinned against reverting.

        Where the solver truncates -- scikit-learn 1.9.0 -- ``requires_scaling``
        going back to ``False`` would let S4 hand this matrix over unscaled and a
        real feature would vanish; the assertion is on ``rank_``, because the
        defect is structural. Where it does not -- 1.5.2 through 1.8.0 -- the
        declaration must still be free: scaling cannot change an OLS prediction,
        so ``True`` is never the wrong answer. Both are asserted; neither assumed.
        The scaled fit keeps full rank on every version, unconditionally.
        """
        from sklearn.linear_model import LinearRegression

        rng = np.random.default_rng(4)
        n = 500
        revenue = rng.normal(2_000_000, 400_000, n)
        rate = rng.normal(0.03, 0.006, n)
        target = 1e-6 * revenue + 50.0 * rate + rng.normal(0, 1e-3, n)
        features = np.column_stack([revenue, rate])
        scaled_features = StandardScaler().fit_transform(features)

        scaled = LinearRegression().fit(scaled_features, target)
        assert scaled.rank_ == 2, "the scaler the declaration asks for must keep every feature"

        unscaled = LinearRegression().fit(features, target)
        if unscaled.rank_ < 2:
            assert unscaled.rank_ == 1
        else:
            np.testing.assert_allclose(
                unscaled.predict(features), scaled.predict(scaled_features), rtol=1e-6
            )

    def test_claiming_native_nan_where_there_is_none_would_be_caught(self, continuous):
        """If linear_regression declared handles_missing_values=True, this fails."""
        from aidatasetkit.models import ModelFactory

        features, target = continuous
        with_nan = features.copy()
        with_nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            ModelFactory.create("linear_regression").fit(with_nan, target)

    def test_claiming_sparse_support_where_there_is_none_would_be_caught(
        self, continuous
    ):
        """If hist_gradient_boosting_regressor declared sparse, this fails."""
        from aidatasetkit.models import ModelFactory

        features, target = continuous
        with pytest.raises(TypeError, match="[Ss]parse"):
            ModelFactory.create("hist_gradient_boosting_regressor", max_iter=10).fit(
                csr_matrix(np.abs(features)), target
            )

    def test_a_regressor_registered_as_a_classifier_would_not_construct(self):
        """``ModelCapabilities`` refuses the combination before registration."""
        from aidatasetkit.core.exceptions import ValidationError
        from aidatasetkit.core.types import Backend, Interpretability
        from aidatasetkit.models import ModelCapabilities

        with pytest.raises(ValidationError, match="supports_predict_proba is meaningless"):
            ModelCapabilities(
                task_type=TaskType.REGRESSION,
                backend=Backend.SKLEARN,
                supports_predict_proba=True,
                requires_scaling=False,
                supports_sparse_input=True,
                supports_multiclass=False,
                handles_missing_values=False,
                interpretability_level=Interpretability.HIGH,
            )

    def test_advertising_probabilities_on_a_regressor_would_be_caught(self, continuous):
        """The object side of the same claim: no regressor grows a predict_proba."""
        for entry in REGRESSORS:
            fitted = _build(entry).fit(*continuous)
            assert not hasattr(fitted, "predict_proba"), entry.canonical_name

    def test_moving_a_load_bearing_seed_really_does_change_the_model(self):
        """If random_state stopped being passed through, this fails."""
        from aidatasetkit.models import ModelFactory

        rng = np.random.default_rng(2024)
        n = 300
        features = rng.normal(size=(n, 6))
        target = features[:, 0] * 2 + rng.normal(0, 1.5, n)
        first = ModelFactory.create(
            "random_forest_regressor", n_estimators=20, random_state=1
        ).fit(features, target)
        second = ModelFactory.create(
            "random_forest_regressor", n_estimators=20, random_state=2
        ).fit(features, target)
        assert not np.array_equal(first.predict(features), second.predict(features))
