"""The regression target contract: a quantity stays a quantity.

The single worst thing that could happen to a regression target in this library
is that something helpfully turns it into class indices. A price of 249.99 would
become "class 37", every model would fit successfully, every metric would report
a number, and nothing anywhere would say that the thing being predicted had been
destroyed.

So the contract has one rule and this file proves it from several directions:
**nothing encodes a regression target**. The library's only label encoder refuses
one outright, no regression strategy touches ``y`` at all, and the values reach
the estimator exactly as the caller wrote them -- negative, zero, integral,
enormous, tiny, or repeated.

There is no ``RegressionTargetEncoder`` and S6 does not add one. A component that
transformed ``y`` would be a component that could get it wrong; the correct
number of transformations to apply to a quantity is none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core.exceptions import PreprocessingError, UnsupportedTaskError
from aidatasetkit.core.types import TargetProfile, TaskType
from aidatasetkit.models import ModelFactory
from aidatasetkit.preprocessing import TargetLabelEncoder
from aidatasetkit.profiling import TaskDetector

from tests.conftest import BUILT_IN_REGRESSORS

#: A few models spanning all four regression preprocessing profiles, so a target
#: question is asked of every distinct path without fitting nine models per case.
ACROSS_PROFILES = [
    "dummy_regressor",
    "linear_regression",
    "ridge_regression",
    "decision_tree_regressor",
]


def features(n: int = 60, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3))


def shrink(name: str) -> dict:
    params = ModelFactory.create(name).get_params()
    overrides = {}
    if "n_estimators" in params:
        overrides["n_estimators"] = 5
    if "max_iter" in params and "early_stopping" in params:
        overrides["max_iter"] = 10
    return overrides


class TestNothingLabelEncodesARegressionTarget:
    """The one rule, stated four ways."""

    def test_the_encoder_refuses_a_resolved_regression_target(self):
        profile = TargetProfile(task_type=TaskType.REGRESSION, target_name="price")
        with pytest.raises(UnsupportedTaskError, match="must not be label encoded"):
            TargetLabelEncoder().fit(pd.Series([1.5, 2.5, 3.5]), profile)

    def test_and_refuses_an_obviously_continuous_one_with_no_profile_at_all(self):
        """Without a TargetProfile the encoder still catches the certain case."""
        with pytest.raises(UnsupportedTaskError, match="regression target"):
            TargetLabelEncoder().fit(pd.Series([10.5, 20.25, 30.75, 40.125]))

    def test_the_refusal_explains_what_would_have_been_destroyed(self):
        profile = TargetProfile(task_type=TaskType.REGRESSION)
        with pytest.raises(UnsupportedTaskError) as raised:
            TargetLabelEncoder().fit(pd.Series([1.5, 2.5]), profile)
        message = str(raised.value)
        assert "quantities" in message
        assert "class indices" in message

    def test_no_regression_strategy_imports_an_encoder_of_any_kind(self):
        """A strategy that transformed y would be preprocessing by another name."""
        from pathlib import Path

        package = (
            Path(__file__).resolve().parents[2] / "aidatasetkit" / "models" / "regression"
        )
        modules = sorted(package.glob("*.py"))
        assert len(modules) >= 7, "the regression package was not found"
        for path in modules:
            source = path.read_text(encoding="utf-8")
            for forbidden in (
                "LabelEncoder",
                "StandardScaler",
                "SimpleImputer",
                "OneHotEncoder",
                "aidatasetkit.preprocessing",
            ):
                assert forbidden not in source, f"{path.name} names {forbidden}"


class TestTheTargetReachesTheEstimatorUnchanged:
    """Every shape of quantity an analyst actually has.

    Each case asserts two things: the model fits, and the caller's target object
    is not modified. A library that quietly cast, sorted, or rescaled ``y`` would
    pass the first and fail the second.
    """

    CASES = {
        "floating point": np.array([1.5, -2.25, 0.0, 99.75, 3.125] * 12),
        "integer valued": np.arange(-30, 30).astype("int64"),
        "negative only": np.linspace(-500.0, -1.0, 60),
        "all zero": np.zeros(60),
        "zero among positives": np.concatenate([np.zeros(10), np.arange(1, 51.0)]),
        "very small": np.linspace(1e-12, 6e-12, 60),
        "very large finite": np.linspace(1e12, 6e12, 60),
        "heavily repeated": np.repeat([2.5, 7.5, 2.5, 7.5], 15),
        "mixed sign spanning zero": np.linspace(-1e6, 1e6, 60),
    }

    @pytest.mark.parametrize("name", ACROSS_PROFILES)
    @pytest.mark.parametrize("label", sorted(CASES))
    def test_it_fits_and_predicts_finite_numbers(self, name, label):
        target = self.CASES[label]
        estimator = ModelFactory.create(name, **shrink(name))
        predictions = estimator.fit(features(len(target)), target).predict(
            features(len(target))
        )
        assert predictions.shape == target.shape
        assert np.isfinite(predictions).all()
        assert np.issubdtype(predictions.dtype, np.number)

    @pytest.mark.parametrize("label", sorted(CASES))
    def test_and_the_model_actually_learned_something_from_it(self, label):
        """Shape and finiteness are not enough, and assuming they were hid a defect.

        An earlier version of this file checked only that the predictions were
        finite and one per row. For the ``very small`` case they were -- and the
        model had learned nothing, returning the training mean for every row.
        A prediction that is finite, correctly shaped, and constant is exactly
        what a silently useless fit looks like.

        ``very small`` is excluded here and pinned separately below, because for
        four of the nine regressors the constant answer is real behaviour rather
        than a defect in this library.
        """
        if label in {"all zero", "very small"}:
            pytest.skip("no signal to recover, or the documented backend cliff")
        target = self.CASES[label]
        rows = len(target)
        rng = np.random.default_rng(21)
        informative = np.column_stack(
            [target + rng.normal(0, float(np.std(target)) * 1e-3 or 1e-12, rows)]
        )
        predictions = (
            ModelFactory.create("decision_tree_regressor")
            .fit(informative, target)
            .predict(informative)
        )
        assert len(np.unique(predictions)) > 1, (
            "every row got the same prediction from a perfectly informative "
            "feature, so the fit learned nothing"
        )


    @pytest.mark.parametrize("label", sorted(CASES))
    def test_the_caller_target_is_not_modified(self, label):
        target = self.CASES[label].copy()
        before = target.copy()
        ModelFactory.create("linear_regression").fit(features(len(target)), target)
        np.testing.assert_array_equal(target, before)

    def test_an_integer_target_is_not_quietly_made_categorical(self):
        """Sixty distinct integers are a quantity, and stay one.

        The failure this guards against is subtle: an integer dtype is exactly
        what a label column looks like, so a library that decided by dtype would
        turn a count into sixty classes and nothing would complain.
        """
        target = np.arange(-30, 30).astype("int64")
        fitted = ModelFactory.create("linear_regression").fit(features(60), target)
        predictions = fitted.predict(features(60))
        assert not hasattr(fitted, "classes_")
        # Genuine interpolation: a regressor answers between the training values.
        assert not np.all(np.isin(predictions, target))

    def test_a_target_of_two_distinct_floats_is_still_regressed_on(self):
        """The most classification-shaped regression target there is."""
        target = np.array([2.5, 7.5] * 30)
        fitted = ModelFactory.create("linear_regression").fit(features(60), target)
        assert not hasattr(fitted, "classes_")
        assert fitted.predict(features(60)).shape == (60,)


class TestTheSmallTargetCliffIsRecordedRatherThanDiscovered:
    """Below a target standard deviation of about 1.5e-8, four regressors stop.

    scikit-learn's tree splitter compares the impurity improvement of a candidate
    split against a tolerance derived from double eps. When the target's variance
    falls under it every node looks pure, no split is ever taken, and the fitted
    model answers with the training mean for every row -- no error, no warning,
    and a perfectly well-shaped array of finite numbers.

    It is the target-side mirror of the feature-side ``FEATURE_THRESHOLD`` the
    tree family already documents, and it is a backend limitation rather than a
    defect here. It is pinned because an undocumented cliff that returns a
    plausible answer is the most dangerous kind, and because this project's rule
    is that a limitation nobody wrote down is the failure the limitations page
    exists to prevent.

    Found by adversarial review, not by the test above it -- which asked whether
    the predictions were finite and correctly shaped, and they were.

    The fix available to a user is the same one that rescues the feature side:
    rescale. Multiplying the target by a constant changes the *units* of the
    thing being predicted, so the library does not do it for them.
    """

    AFFECTED = [
        "decision_tree_regressor",
        "random_forest_regressor",
        "extra_trees_regressor",
        "gradient_boosting_regressor",
    ]
    UNAFFECTED = [
        "linear_regression",
        "ridge_regression",
        "knn_regressor",
        "hist_gradient_boosting_regressor",
    ]

    @staticmethod
    def _fit(name, scale):
        rng = np.random.default_rng(5)
        rows = 300
        X = rng.normal(size=(rows, 3))
        y = (2 * X[:, 0] - X[:, 1]) * scale
        return ModelFactory.create(name, **shrink(name)).fit(X, y).predict(X), y

    def test_the_two_lists_together_are_the_whole_catalog(self):
        assert set(self.AFFECTED) | set(self.UNAFFECTED) == set(BUILT_IN_REGRESSORS) - {
            "dummy_regressor"
        }

    @pytest.mark.parametrize("name", AFFECTED)
    def test_at_an_ordinary_target_scale_it_learns(self, name):
        predictions, _ = self._fit(name, 1.0)
        assert len(np.unique(predictions)) > 1

    @pytest.mark.parametrize("name", AFFECTED)
    def test_and_below_the_cliff_it_answers_with_one_constant(self, name):
        """The defect is constancy. Which constant is an ensemble detail.

        A single tree returns the training mean exactly. A forest averages the
        means of its bootstrap samples, so it lands near the training mean
        without equalling it -- asserting equality would fail for the right
        reason and read like the finding was wrong.
        """
        predictions, target = self._fit(name, 1e-9)
        assert len(np.unique(predictions)) == 1, (
            "this model is recorded as affected by the small-target cliff, but "
            "it produced varying predictions -- re-measure and update the list"
        )
        # A location estimate, not a fit: the constant sits inside the target's
        # own spread, and carries none of the signal the features hold.
        assert abs(predictions[0] - target.mean()) < float(np.std(target))

    @pytest.mark.parametrize("name", UNAFFECTED)
    def test_the_other_regressors_are_unaffected(self, name):
        predictions, _ = self._fit(name, 1e-9)
        assert len(np.unique(predictions)) > 1, f"{name} is affected after all"

    def test_rescaling_the_target_recovers_the_fit(self):
        """The user's remedy, which the library will not apply on their behalf."""
        below, _ = self._fit("decision_tree_regressor", 1e-9)
        above, _ = self._fit("decision_tree_regressor", 1.0)
        assert len(np.unique(below)) == 1
        assert len(np.unique(above)) > 1

    def test_the_failure_is_silent_which_is_why_it_is_documented(self):
        """No exception, no warning: the array is finite and correctly shaped."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            predictions, _ = self._fit("random_forest_regressor", 1e-12)
        assert np.isfinite(predictions).all()
        assert predictions.shape == (300,)

    def test_the_canonical_limitations_page_records_it(self):
        """A limitation known and not written down is the failure that page prevents."""
        from pathlib import Path

        page = (
            Path(__file__).resolve().parents[2] / "docs" / "limitations.md"
        ).read_text(encoding="utf-8")
        assert "regression target below" in page.lower()
        assert "training mean" in page


class TestPandasTargetTypes:
    """A pandas Series, and the nullable dtypes people actually have."""

    @staticmethod
    def _target(values, dtype):
        return pd.Series(values, dtype=dtype, name="price")

    @pytest.mark.parametrize("dtype", ["float64", "float32"])
    def test_float_dtypes_fit_directly(self, dtype):
        target = self._target(np.arange(60) * 1.5, dtype)
        fitted = ModelFactory.create("linear_regression").fit(features(60), target)
        assert fitted.predict(features(60)).shape == (60,)

    @pytest.mark.parametrize("dtype", ["int64", "int32"])
    def test_integer_dtypes_fit_directly_and_stay_regression(self, dtype):
        """An integer dtype is exactly what a label column looks like."""
        target = self._target(np.arange(-30, 30), dtype)
        fitted = ModelFactory.create("linear_regression").fit(features(60), target)
        assert not hasattr(fitted, "classes_")
        predictions = fitted.predict(features(60))
        assert predictions.shape == (60,)
        assert np.issubdtype(predictions.dtype, np.floating), (
            "a regressor answers with a quantity, not with one of the training values"
        )

    @pytest.mark.parametrize("dtype", ["Int64", "Float64"])
    def test_a_nullable_dtype_without_gaps_fits(self, dtype):
        """pd.NA is what breaks scikit-learn, not the extension dtype itself."""
        target = self._target(np.arange(60) * 2, dtype)
        fitted = ModelFactory.create("linear_regression").fit(features(60), target)
        assert fitted.predict(features(60)).shape == (60,)

    @pytest.mark.parametrize("dtype", ["Int64", "Float64"])
    def test_a_nullable_dtype_carrying_a_gap_is_refused_rather_than_guessed(self, dtype):
        """A row with no label cannot be trained on, and nothing invents one."""
        values = pd.array([1, 2, None] * 20, dtype=dtype)
        target = pd.Series(values, name="price")
        with pytest.raises((ValueError, TypeError)):
            ModelFactory.create("linear_regression").fit(features(60), target)

    def test_the_target_name_is_preserved_because_nothing_touches_it(self):
        target = self._target(np.arange(60) * 1.0, "float64")
        ModelFactory.create("linear_regression").fit(features(60), target)
        assert target.name == "price"

    def test_the_series_itself_is_unchanged_after_fitting(self):
        target = self._target(np.arange(60) * 1.0, "float64")
        before = target.copy()
        ModelFactory.create("ridge_regression").fit(features(60), target)
        pd.testing.assert_series_equal(target, before)

    def test_the_index_is_not_reset_or_reordered(self):
        target = pd.Series(
            np.arange(60) * 1.0, index=pd.RangeIndex(1000, 1060), name="price"
        )
        before = target.index.copy()
        ModelFactory.create("linear_regression").fit(features(60), target)
        pd.testing.assert_index_equal(target.index, before)


class TestTheMissingTargetPolicyIsTheExistingOne:
    """S6 invents no policy of its own. There already is one, in two places."""

    def test_the_task_detector_counts_missing_labels_without_dropping_the_column(self):
        detected = TaskDetector().detect(
            pd.Series([1.5, 2.5, np.nan, 4.5, 5.5, 6.5, 7.5]), target_name="price"
        )
        assert detected.task_type is TaskType.REGRESSION
        assert detected.missing_count == 1

    def test_a_target_with_no_observed_value_at_all_is_refused(self):
        from aidatasetkit.core.exceptions import EmptyDataError

        with pytest.raises(EmptyDataError):
            TaskDetector().detect(pd.Series([np.nan, np.nan]), target_name="price")

    def test_the_classification_encoder_states_the_policy_for_its_own_family(self):
        """Named here so the two families' policies can be read side by side."""
        with pytest.raises(PreprocessingError, match="missing value"):
            TargetLabelEncoder().fit(pd.Series(["a", "b", None, "a"]))

    def test_a_missing_regression_label_reaches_the_estimator_and_is_refused_there(self):
        """No AIDatasetKit component consumes y in S6, so nothing can pre-empt it.

        Recorded rather than papered over: the error is scikit-learn's, it names
        the problem correctly, and inventing a wrapper to restate it would put a
        component between the caller and their target for no gain. Whichever
        stage first *consumes* a regression target owns this check, and that
        stage is S7.
        """
        target = np.array([1.5, 2.5, np.nan] * 20)
        with pytest.raises(ValueError, match="NaN|missing|infinity"):
            ModelFactory.create("linear_regression").fit(features(60), target)


class TestTheDetectedTaskAndTheChosenModelAgree:
    """The factory checks a model against a resolved target before constructing."""

    @staticmethod
    def _detect(values):
        return TaskDetector().detect(pd.Series(values), target_name="price")

    @pytest.mark.parametrize("name", BUILT_IN_REGRESSORS)
    def test_every_regressor_accepts_a_detected_regression_target(self, name):
        detected = self._detect(np.linspace(0.5, 60.5, 60))
        assert detected.task_type is TaskType.REGRESSION
        assert ModelFactory.create(name, target=detected, **shrink(name)) is not None

    @pytest.mark.parametrize("name", BUILT_IN_REGRESSORS)
    def test_and_refuses_a_detected_classification_target(self, name):
        from aidatasetkit.core.exceptions import IncompatibleModelError

        detected = self._detect(["churn", "stay"] * 30)
        assert detected.task_type is TaskType.CLASSIFICATION
        with pytest.raises(IncompatibleModelError, match="regression tasks"):
            ModelFactory.create(name, target=detected)

    def test_the_refusal_happens_before_anything_is_constructed(self):
        """The point is to fail here rather than inside sklearn several steps later."""
        from aidatasetkit.core.exceptions import IncompatibleModelError

        detected = self._detect(["churn", "stay"] * 30)
        with pytest.raises(IncompatibleModelError):
            ModelFactory.strategy("ridge_regression", target=detected)

    def test_an_integer_target_the_detector_calls_regression_stays_regression(self):
        """Sixty distinct integers: the detector says regression and the model agrees."""
        detected = self._detect(np.arange(60) * 7)
        assert detected.task_type is TaskType.REGRESSION
        assert ModelFactory.create("linear_regression", target=detected) is not None

    def test_a_target_the_detector_calls_ambiguous_is_not_settled_by_the_model(self):
        """Five distinct integers in a small sample could be classes or a count.

        The detector refuses and asks for a hint. Nothing in the model layer
        overrides that -- a model does not get to decide what the data is.
        """
        from aidatasetkit.core.exceptions import AmbiguousTaskError

        with pytest.raises(AmbiguousTaskError):
            TaskDetector().detect(pd.Series([1, 2, 3, 4, 5] * 4), target_name="grade")

    def test_a_hint_settles_it_and_the_regressor_then_accepts_it(self):
        detected = TaskDetector().detect(
            pd.Series([1, 2, 3, 4, 5] * 4), hint="regression", target_name="grade"
        )
        assert detected.task_type is TaskType.REGRESSION
        assert ModelFactory.create("linear_regression", target=detected) is not None
