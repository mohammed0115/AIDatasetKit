"""Every regressor, through the real S4 preprocessing path.

The claim S6 has to make good on is the same one S5 made, and it is narrow and
testable: a model's :class:`~aidatasetkit.core.types.PreprocessingProfile` is the
*only* thing that decides how its data is prepared. Nothing reads a model's name,
and nothing reads its task either. So these tests run the whole path -- profile,
plan, build, fit, predict -- for all nine regressors over one realistic mixed
frame, and then check the individual claims that profile makes: a scaler where
scaling is required and none where it is not, imputation where NaN cannot be
consumed and untouched NaN where it can, a sparse matrix where sparse is accepted
and a dense array where it is refused.

One difference from the classification file, and it is the point of §7: there is
no target encoder anywhere below. A regression target is passed to ``fit``
exactly as the caller wrote it.

Nothing here evaluates a model. Which of these regressors is any good is a
question for S7, and asking it now would be the beginning of AutoML.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.pipeline import Pipeline

from aidatasetkit.core.exceptions import PreprocessingError
from aidatasetkit.core.types import PreprocessingProfile
from aidatasetkit.models import ModelFactory, default_registry
from aidatasetkit.preprocessing import (
    BlueprintCache,
    HighCardinalityPolicy,
    PreprocessingConfig,
    PreprocessingPlanner,
    PreprocessorBuilder,
)
from aidatasetkit.profiling import DataProfiler, DataQualityInspector

REGRESSORS = default_registry().catalog(task="regression")
NAMES = [entry.canonical_name for entry in REGRESSORS]

#: Ensemble sizes are turned down by *signature*, exactly as the contract tests
#: do it, so that no test anywhere names a model to decide how to build it.
_SMALL = {"n_estimators": 5, "max_iter": 10}


def small_params(name: str) -> dict:
    accepted = set(inspect.signature(type(ModelFactory.create(name))).parameters)
    params = {}
    if "n_estimators" in accepted:
        params["n_estimators"] = _SMALL["n_estimators"]
    if "max_iter" in accepted and "early_stopping" in accepted:
        params["max_iter"] = _SMALL["max_iter"]
    return params


def make_estimator(name: str, **overrides):
    return ModelFactory.create(name, **{**small_params(name), **overrides})


def mixed_frame(n: int = 240, seed: int = 20250101) -> pd.DataFrame:
    """Numeric on two very different scales, nominal, boolean, and gaps.

    The same shape the classification integration tests use. The scales differ
    deliberately: a frame whose columns are already comparable could not tell a
    model that needs scaling apart from one that does not.
    """
    rng = np.random.default_rng(seed)
    income = rng.normal(60_000, 15_000, n).round(2)
    ratio = rng.normal(0.5, 0.05, n).round(4)
    frame = pd.DataFrame(
        {
            "income": income,
            "ratio": ratio,
            "city": rng.choice(["Riyadh", "Jeddah"], n),
            "is_member": rng.choice([True, False], n),
        }
    )
    frame.loc[frame.index[:20], "income"] = np.nan
    return frame


def make_target(frame: pd.DataFrame) -> pd.Series:
    """A genuine continuous quantity, not a label wearing a float dtype.

    Built from the column that has no gaps, so a model's ability to fit does not
    depend on how its profile handled the missing incomes. The values are
    non-integral and span a real range.
    """
    ratio = frame["ratio"].to_numpy()
    return pd.Series(900.0 * ratio + 37.5, name="price")


def prepare(frame: pd.DataFrame, profile: PreprocessingProfile, config=None):
    """Profile, inspect, plan, and build -- the real path, no shortcuts."""
    config = config or PreprocessingConfig()
    dataset_profile = DataProfiler().profile(frame)
    quality = DataQualityInspector().inspect(frame, profile=dataset_profile)
    plan = PreprocessingPlanner(config).plan(frame, dataset_profile, profile, quality=quality)
    return plan, PreprocessorBuilder(config).build(plan, frame)


def densify(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def r2(y, predicted) -> float:
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - residual / total


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return mixed_frame()


@pytest.fixture(scope="module")
def target(frame) -> pd.Series:
    return make_target(frame)


@pytest.mark.parametrize("name", NAMES)
class TestEveryRegressorRunsThroughPreprocessing:
    """One realistic frame, nine models, no model-specific handling anywhere."""

    def test_it_fits_and_predicts_on_a_continuous_target(self, name, frame, target):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())

        matrix = preprocessor.fit_transform(frame)
        estimator = make_estimator(name).fit(matrix, target)
        predictions = estimator.predict(preprocessor.transform(frame))

        assert predictions.shape == (len(frame),)
        assert np.isfinite(predictions).all()

    def test_the_target_went_in_untouched(self, name, frame, target):
        """No encoder, no cast, no reordering: the caller's Series, as written."""
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        before = target.copy()
        make_estimator(name).fit(preprocessor.fit_transform(frame), target)
        pd.testing.assert_series_equal(target, before)

    def test_it_fits_inside_a_single_sklearn_pipeline(self, name, frame, target):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())

        pipeline = Pipeline([("prep", preprocessor), ("model", make_estimator(name))])
        pipeline.fit(frame, target)
        assert pipeline.predict(frame).shape == (len(frame),)

    def test_it_never_grows_a_classification_surface(self, name, frame, target):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        fitted = make_estimator(name).fit(preprocessor.fit_transform(frame), target)
        assert not hasattr(fitted, "predict_proba")
        assert not hasattr(fitted, "classes_")

    def test_an_unseen_category_predicts_without_refitting_anything(
        self, name, frame, target
    ):
        """Trained on Riyadh and Jeddah; scored on Dammam."""
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        estimator = make_estimator(name).fit(preprocessor.fit_transform(frame), target)

        unseen = frame.iloc[:30].assign(city="Dammam")
        transformed = preprocessor.transform(unseen)

        assert transformed.shape[1] == preprocessor.n_features_out
        predictions = estimator.predict(transformed)
        assert predictions.shape == (30,)
        assert np.isfinite(predictions).all()

    def test_the_unseen_category_became_all_zeros_rather_than_an_error(
        self, name, frame
    ):
        """The existing S4 unknown-category contract, not a second one for S6."""
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        preprocessor.fit(frame)

        names = list(preprocessor.get_feature_names_out())
        city_columns = [i for i, column in enumerate(names) if str(column).startswith("city")]
        assert city_columns, "the one-hot columns for city were not found"

        unseen = densify(preprocessor.transform(frame.iloc[:5].assign(city="Dammam")))
        assert not unseen[:, city_columns].any()

    def test_scoring_does_not_move_what_was_learned(self, name, frame):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        preprocessor.fit(frame)
        before = list(preprocessor.get_feature_names_out())

        preprocessor.transform(frame.iloc[:30].assign(city="Dammam"))
        assert list(preprocessor.get_feature_names_out()) == before

    def test_the_caller_frame_is_never_modified(self, name, frame, target):
        capabilities = ModelFactory.registration(name).capabilities
        before = frame.copy(deep=True)
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        make_estimator(name).fit(preprocessor.fit_transform(frame), target)
        preprocessor.transform(frame)
        pd.testing.assert_frame_equal(frame, before)
        assert list(frame.columns) == list(before.columns)
        assert frame.dtypes.equals(before.dtypes)


@pytest.mark.parametrize("name", NAMES)
class TestCapabilitiesDriveThePipeline:
    """The positive and negative branch of each declaration, on real models."""

    def test_the_scaler_is_present_exactly_when_scaling_is_required(self, name, frame):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        preprocessor.fit(frame)
        steps = preprocessor.transformer.named_transformers_["numeric"].named_steps
        assert ("scaler" in steps) is capabilities.requires_scaling

    def test_the_imputer_is_present_exactly_when_nan_cannot_be_consumed(self, name, frame):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        preprocessor.fit(frame)
        steps = preprocessor.transformer.named_transformers_["numeric"].named_steps
        assert ("imputer" in steps) is not capabilities.handles_missing_values

    def test_missing_values_survive_only_where_they_are_understood(self, name, frame):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        out = densify(preprocessor.fit_transform(frame))
        assert bool(np.isnan(out).any()) is capabilities.handles_missing_values

    def test_what_preprocessing_produces_is_what_the_estimator_accepts(
        self, name, frame, target
    ):
        """The end of the chain: whatever S4 emitted, the model must take it."""
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        matrix = preprocessor.fit_transform(frame)
        fitted = make_estimator(name).fit(matrix, target)
        assert fitted.predict(matrix).shape == (len(frame),)

    def test_the_plan_records_the_capability_rather_than_the_model(self, name, frame):
        """The recorded reason must be about the requirement, not about who has it."""
        capabilities = ModelFactory.registration(name).capabilities
        plan, _ = prepare(frame, capabilities.preprocessing_profile())
        decision = plan.decision_for("income")
        assert name not in decision.reason
        assert "regress" not in decision.reason
        assert decision.details["requires_scaling"] is capabilities.requires_scaling
        assert (
            decision.details["handles_missing_values"]
            is capabilities.handles_missing_values
        )


class TestNoModelNameOrTaskReachesPreprocessing:
    """The integration direction, asserted rather than assumed."""

    def test_a_bare_profile_produces_the_same_plan_as_the_model_does(self, frame):
        """Hand-built capability triple against the real one: identical plans.

        If anything anywhere consulted the model, these two would differ.
        """
        for name in NAMES:
            capabilities = ModelFactory.registration(name).capabilities
            from_model, _ = prepare(frame, capabilities.preprocessing_profile())
            from_triple, _ = prepare(
                frame,
                PreprocessingProfile(
                    requires_scaling=capabilities.requires_scaling,
                    supports_sparse_input=capabilities.supports_sparse_input,
                    handles_missing_values=capabilities.handles_missing_values,
                ),
            )
            assert from_model.fingerprint == from_triple.fingerprint, name

    def test_a_regressor_and_a_classifier_sharing_a_profile_get_one_plan(self, frame):
        """The strongest form: the task family does not reach preprocessing either."""
        pairs = [
            ("decision_tree_regressor", "decision_tree_classifier"),
            ("dummy_regressor", "dummy_classifier"),
            ("gradient_boosting_regressor", "gradient_boosting_classifier"),
            ("knn_regressor", "knn_classifier"),
        ]
        for regressor, classifier in pairs:
            first, _ = prepare(
                frame,
                ModelFactory.registration(regressor).capabilities.preprocessing_profile(),
            )
            second, _ = prepare(
                frame,
                ModelFactory.registration(classifier).capabilities.preprocessing_profile(),
            )
            assert first.fingerprint == second.fingerprint, f"{regressor}/{classifier}"

    def test_the_factory_can_choose_a_profile_without_preprocessing_imported(self):
        import subprocess
        import sys

        script = (
            "import sys;"
            "from aidatasetkit.models import ModelFactory;"
            "p = ModelFactory.registration('ridge_regression').capabilities."
            "preprocessing_profile();"
            "print(p.key);"
            "print(any(m.startswith('aidatasetkit.preprocessing') for m in sys.modules))"
        )
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.split()
        assert out[0] == "scaling=1,sparse=1,native_nan=0"
        assert out[1] == "False"

    def test_no_regression_module_mentions_preprocessing(self):
        from pathlib import Path

        package = (
            Path(__file__).resolve().parents[2] / "aidatasetkit" / "models" / "regression"
        )
        for path in package.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "aidatasetkit.preprocessing" not in source, path

    def test_no_preprocessing_module_mentions_a_regression_model(self):
        """The reverse direction: S4 gained no knowledge of what S6 registered."""
        from pathlib import Path

        package = Path(__file__).resolve().parents[2] / "aidatasetkit" / "preprocessing"
        for path in package.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for name in NAMES:
                assert name not in source, f"{path.name} names {name}"
            for fragment in ("Ridge", "RandomForestRegressor", "regressor"):
                assert fragment not in source, f"{path.name} names {fragment}"


class TestTheDenseAndSparsePathsAreBothReal:
    """Proved with actual regressors rather than invented profiles."""

    @pytest.fixture(scope="class")
    @staticmethod
    def wide() -> pd.DataFrame:
        rng = np.random.default_rng(99)
        n = 300
        return pd.DataFrame(
            {"num": rng.normal(0, 1, n).round(3), "cat": [f"c{i % 40}" for i in range(n)]}
        )

    @pytest.fixture(scope="class")
    @staticmethod
    def onehot() -> PreprocessingConfig:
        return PreprocessingConfig(high_cardinality_policy=HighCardinalityPolicy.ONEHOT)

    @pytest.mark.parametrize(
        "name", ["knn_regressor", "ridge_regression", "linear_regression",
                 "gradient_boosting_regressor"]
    )
    def test_a_sparse_capable_model_receives_a_sparse_matrix(self, name, wide, onehot):
        capabilities = ModelFactory.registration(name).capabilities
        assert capabilities.supports_sparse_input
        _, preprocessor = prepare(wide, capabilities.preprocessing_profile(), onehot)
        assert sparse.issparse(preprocessor.fit_transform(wide))

    @pytest.mark.parametrize(
        "name", ["decision_tree_regressor", "hist_gradient_boosting_regressor"]
    )
    def test_a_dense_only_model_receives_a_dense_array(self, name, wide, onehot):
        capabilities = ModelFactory.registration(name).capabilities
        assert not capabilities.supports_sparse_input
        _, preprocessor = prepare(wide, capabilities.preprocessing_profile(), onehot)
        assert not sparse.issparse(preprocessor.fit_transform(wide))

    def test_the_dense_only_booster_would_have_refused_the_sparse_form(
        self, wide, onehot
    ):
        """The negative branch: densifying is not decoration, it is required."""
        _, preprocessor = prepare(
            wide,
            PreprocessingProfile(
                requires_scaling=False,
                supports_sparse_input=True,
                handles_missing_values=False,
            ),
            onehot,
        )
        as_sparse = preprocessor.fit_transform(wide)
        assert sparse.issparse(as_sparse)
        y = wide["num"].to_numpy() * 3.0
        with pytest.raises(TypeError, match="[Ss]parse"):
            make_estimator("hist_gradient_boosting_regressor").fit(as_sparse, y)

    @pytest.mark.parametrize(
        "name", ["decision_tree_regressor", "hist_gradient_boosting_regressor"]
    )
    def test_and_accepts_the_dense_form_it_is_actually_given(self, name, wide, onehot):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(wide, capabilities.preprocessing_profile(), onehot)
        matrix = preprocessor.fit_transform(wide)
        y = wide["num"].to_numpy() * 3.0
        assert make_estimator(name).fit(matrix, y).predict(matrix).shape == y.shape


class TestTheNativeNaNPathIsReal:
    """A model that understands gaps is not handed a median instead."""

    NATIVE = [
        "decision_tree_regressor",
        "random_forest_regressor",
        "extra_trees_regressor",
        "hist_gradient_boosting_regressor",
        "dummy_regressor",
    ]

    @pytest.mark.parametrize("name", NATIVE)
    def test_the_gaps_reach_the_estimator(self, name, frame):
        capabilities = ModelFactory.registration(name).capabilities
        assert capabilities.handles_missing_values
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        matrix = densify(preprocessor.fit_transform(frame))
        assert np.isnan(matrix).sum() == 20

    @pytest.mark.parametrize("name", NATIVE)
    def test_and_it_fits_on_them(self, name, frame, target):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        matrix = preprocessor.fit_transform(frame)
        assert make_estimator(name).fit(matrix, target).predict(matrix).shape == (
            len(frame),
        )

    @pytest.mark.parametrize(
        "name",
        [
            "linear_regression",
            "ridge_regression",
            "gradient_boosting_regressor",
            "knn_regressor",
        ],
    )
    def test_a_model_that_cannot_take_gaps_is_never_shown_one(self, name, frame):
        capabilities = ModelFactory.registration(name).capabilities
        assert not capabilities.handles_missing_values
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        assert not np.isnan(densify(preprocessor.fit_transform(frame))).any()

    @pytest.mark.parametrize(
        "name",
        [
            "linear_regression",
            "ridge_regression",
            "gradient_boosting_regressor",
            "knn_regressor",
        ],
    )
    def test_and_would_genuinely_have_refused_one(self, name, frame, target):
        """The negative branch: imputation is required, not merely tidy."""
        native = PreprocessingProfile(
            requires_scaling=ModelFactory.registration(name).capabilities.requires_scaling,
            supports_sparse_input=False,
            handles_missing_values=True,
        )
        _, preprocessor = prepare(frame, native)
        with_gaps = densify(preprocessor.fit_transform(frame))
        assert np.isnan(with_gaps).any()
        with pytest.raises(ValueError, match="NaN|missing|infinity"):
            make_estimator(name).fit(with_gaps, target)

    def test_the_imputed_value_is_the_training_median_and_nothing_cleverer(self, frame):
        """What a model without native support actually receives, stated exactly.

        Read through the gradient booster because it is the one regressor that
        neither consumes gaps nor asks for scaling, so the value reaching it is
        the median unmodified rather than the median standardised.
        """
        capabilities = ModelFactory.registration(
            "gradient_boosting_regressor"
        ).capabilities
        assert not capabilities.requires_scaling
        assert not capabilities.handles_missing_values
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        matrix = densify(preprocessor.fit_transform(frame))
        expected = float(frame["income"].median())
        np.testing.assert_allclose(matrix[:20, 0], expected)


class TestScalingActuallyChangesTheNumbers:
    """requires_scaling is a claim about the data the model receives."""

    def test_a_scaled_model_receives_standardised_columns(self, frame):
        capabilities = ModelFactory.registration("ridge_regression").capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        out = densify(preprocessor.fit_transform(frame))
        numeric = out[:, :2]
        assert np.allclose(numeric.mean(axis=0), 0.0, atol=1e-8)
        assert np.allclose(numeric.std(axis=0), 1.0, atol=1e-8)

    def test_an_unscaled_model_receives_the_original_magnitudes(self, frame):
        capabilities = ModelFactory.registration(
            "gradient_boosting_regressor"
        ).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        out = densify(preprocessor.fit_transform(frame))
        assert np.nanmax(np.abs(out[:, 0])) > 1_000.0

    def test_a_scaled_and_an_unscaled_model_receive_different_matrices(self, frame):
        scaled = ModelFactory.registration("ridge_regression").capabilities
        unscaled = ModelFactory.registration("gradient_boosting_regressor").capabilities
        _, a = prepare(frame, scaled.preprocessing_profile())
        _, b = prepare(frame, unscaled.preprocessing_profile())
        assert not np.allclose(
            densify(a.fit_transform(frame)), densify(b.fit_transform(frame))
        )

    def test_both_linear_models_are_handed_the_same_standardised_matrix(self, frame):
        """They agree on the declaration, so S4 cannot tell them apart -- correctly.

        The reasons differ (a scale-dependent penalty; a rank-truncating solve)
        and the requirement is identical, which is exactly the situation the
        capability triple exists to collapse.
        """
        first = ModelFactory.registration("linear_regression").capabilities
        second = ModelFactory.registration("ridge_regression").capabilities
        assert first.preprocessing_profile() == second.preprocessing_profile()
        _, a = prepare(frame, first.preprocessing_profile())
        _, b = prepare(frame, second.preprocessing_profile())
        np.testing.assert_allclose(
            densify(a.fit_transform(frame)), densify(b.fit_transform(frame))
        )

    def test_and_the_scaler_rescues_the_rank_truncation_end_to_end(self, target):
        """The defect that changed this declaration, through the real S4 path.

        The protection is asserted unconditionally: the path the declaration
        selects keeps both features on every supported scikit-learn. What the
        path it prevents would have done depends on the version -- 1.9.0 truncates
        to one feature; 1.5.2 through 1.8.0 keep both and predict identically --
        and the consequences of whichever one is installed are asserted.
        """
        rng = np.random.default_rng(4)
        n = 400
        wide = pd.DataFrame(
            {
                "revenue": rng.normal(2_000_000, 400_000, n),
                "rate": rng.normal(0.03, 0.006, n),
            }
        )
        y = 1e-6 * wide["revenue"].to_numpy() + 50.0 * wide["rate"].to_numpy()

        capabilities = ModelFactory.registration("linear_regression").capabilities
        assert capabilities.requires_scaling
        _, scaled = prepare(wide, capabilities.preprocessing_profile())
        fitted = make_estimator("linear_regression").fit(scaled.fit_transform(wide), y)
        assert fitted.rank_ == 2

        _, unscaled = prepare(
            wide,
            PreprocessingProfile(
                requires_scaling=False,
                supports_sparse_input=capabilities.supports_sparse_input,
                handles_missing_values=capabilities.handles_missing_values,
            ),
        )
        unscaled_matrix = unscaled.fit_transform(wide)
        would_have_been = make_estimator("linear_regression").fit(unscaled_matrix, y)
        if would_have_been.rank_ < 2:
            assert would_have_been.rank_ == 1
        else:
            np.testing.assert_allclose(
                would_have_been.predict(unscaled_matrix),
                fitted.predict(scaled.transform(wide)),
                rtol=1e-6,
            )

    def test_and_the_scaler_is_what_ridge_needed(self, frame, target):
        """End to end: the profile's scaler recovers the accuracy the penalty lost."""
        capabilities = ModelFactory.registration("ridge_regression").capabilities
        _, scaled = prepare(frame, capabilities.preprocessing_profile())
        _, unscaled = prepare(
            frame,
            PreprocessingProfile(
                requires_scaling=False,
                supports_sparse_input=capabilities.supports_sparse_input,
                handles_missing_values=capabilities.handles_missing_values,
            ),
        )
        with_scaler = scaled.fit_transform(frame)
        without = unscaled.fit_transform(frame)
        good = make_estimator("ridge_regression").fit(with_scaler, target)
        poor = make_estimator("ridge_regression").fit(without, target)
        assert r2(target, good.predict(with_scaler)) > r2(target, poor.predict(without))


class TestInfinityCannotBypassS4:
    """Infinity is not a supported trainable value, and regression is no exception."""

    @pytest.fixture(scope="class")
    @staticmethod
    def with_infinity() -> pd.DataFrame:
        rng = np.random.default_rng(17)
        n = 200
        values = rng.normal(100, 10, n)
        values[3] = np.inf
        values[7] = -np.inf
        return pd.DataFrame({"amount": values, "steady": rng.normal(0, 1, n).round(3)})

    @pytest.mark.parametrize("name", NAMES)
    def test_the_planner_holds_the_column_back_and_says_why(self, name, with_infinity):
        capabilities = ModelFactory.registration(name).capabilities
        plan, _ = prepare(with_infinity, capabilities.preprocessing_profile())
        decision = plan.decision_for("amount")
        assert decision.reason_code == "infinite_values"
        assert "amount" not in plan.included_features
        assert "infinit" in decision.reason

    @pytest.mark.parametrize("name", NAMES)
    def test_an_infinity_arriving_only_at_inference_is_refused_too(
        self, name, with_infinity
    ):
        """The training frame was clean; the scoring frame is not."""
        capabilities = ModelFactory.registration(name).capabilities
        clean = with_infinity.drop(columns=["amount"])
        _, preprocessor = prepare(clean, capabilities.preprocessing_profile())
        preprocessor.fit(clean)

        later = clean.iloc[:10].copy()
        later.loc[later.index[0], "steady"] = np.inf
        with pytest.raises(PreprocessingError, match="infinite"):
            preprocessor.transform(later)

    def test_the_project_error_arrives_rather_than_a_raw_sklearn_one(
        self, with_infinity
    ):
        """S4 owns this condition, so S4's message is what the user reads."""
        capabilities = ModelFactory.registration("linear_regression").capabilities
        config = PreprocessingConfig(force_include=("amount",))
        _, preprocessor = prepare(
            with_infinity, capabilities.preprocessing_profile(), config
        )
        with pytest.raises(PreprocessingError) as raised:
            preprocessor.fit(with_infinity)
        message = str(raised.value)
        assert "amount" in message
        assert "infinite" in message
        assert "will not replace an infinity" in message

    def test_infinity_parsed_out_of_text_is_caught_as_well(self):
        """"inf" in a numeric-text column becomes an infinity, and is refused."""
        from aidatasetkit.preprocessing import NumericTextPolicy

        rows = 120
        frame = pd.DataFrame(
            {
                "amount_text": [["1", "2", "3", "inf"][i % 4] for i in range(rows)],
                "steady": np.linspace(0, 1, rows),
            }
        )
        capabilities = ModelFactory.registration("ridge_regression").capabilities
        config = PreprocessingConfig(numeric_text_policy=NumericTextPolicy.CONVERT)
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile(), config)
        with pytest.raises(PreprocessingError, match="infinite"):
            preprocessor.fit(frame)

    def test_an_infinite_target_is_not_S4s_business_but_still_fails_loudly(self, frame):
        """Recorded, not silently accepted: S4 guards features, sklearn guards y."""
        capabilities = ModelFactory.registration("linear_regression").capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        matrix = preprocessor.fit_transform(frame)
        broken = make_target(frame).to_numpy().copy()
        broken[0] = np.inf
        with pytest.raises(ValueError, match="infinity|inf"):
            make_estimator("linear_regression").fit(matrix, broken)


class TestFeatureNamesAndLineageStayValid:
    """S4 remains the authority; nothing is derived from an estimator."""

    @pytest.mark.parametrize("name", NAMES)
    def test_the_output_names_match_the_matrix_width(self, name, frame):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        matrix = preprocessor.fit_transform(frame)
        names = preprocessor.get_feature_names_out()
        assert len(names) == matrix.shape[1]
        assert len(set(names)) == len(names)

    @pytest.mark.parametrize("name", NAMES)
    def test_lineage_covers_every_included_column_and_only_those(self, name, frame):
        capabilities = ModelFactory.registration(name).capabilities
        plan, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        preprocessor.fit(frame)
        lineage = preprocessor.lineage()
        assert set(lineage) == set(plan.included_features)

    @pytest.mark.parametrize("name", NAMES)
    def test_every_lineage_entry_names_a_real_output_column(self, name, frame):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        preprocessor.fit(frame)
        produced = set(preprocessor.get_feature_names_out())
        for source, outputs in preprocessor.lineage().items():
            assert outputs, source
            assert set(outputs) <= produced, source

    def test_a_one_hot_column_traces_back_to_the_column_it_came_from(self, frame):
        capabilities = ModelFactory.registration("ridge_regression").capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        preprocessor.fit(frame)
        lineage = preprocessor.lineage()
        assert len(lineage["city"]) == 2
        assert all("Riyadh" in c or "Jeddah" in c for c in lineage["city"])

    def test_lineage_is_identical_for_two_models_sharing_a_profile(self, frame):
        """It follows the plan, so two models on one plan cannot disagree."""
        first = ModelFactory.registration(
            "ridge_regression"
        ).capabilities.preprocessing_profile()
        second = ModelFactory.registration(
            "knn_regressor"
        ).capabilities.preprocessing_profile()
        assert first == second
        _, a = prepare(frame, first)
        _, b = prepare(frame, second)
        a.fit(frame)
        b.fit(frame)
        assert a.lineage() == b.lineage()


class TestBlueprintReuseIsCapabilityBased:
    """Nine regressors, four preprocessors. Nothing consults a model name."""

    NATIVE_NAN_GROUP = (
        "decision_tree_regressor",
        "random_forest_regressor",
        "extra_trees_regressor",
        "hist_gradient_boosting_regressor",
    )

    @staticmethod
    def _cache_over(frame, names):
        cache = BlueprintCache()
        config = PreprocessingConfig()
        dataset_profile = DataProfiler().profile(frame)
        quality = DataQualityInspector().inspect(frame, profile=dataset_profile)
        builder = PreprocessorBuilder(config, cache=cache)
        for name in names:
            profile = ModelFactory.registration(name).capabilities.preprocessing_profile()
            plan = PreprocessingPlanner(config).plan(
                frame, dataset_profile, profile, quality=quality
            )
            builder.build(plan, frame)
        return cache

    def test_the_natively_missing_aware_models_resolve_to_one_profile(self):
        profiles = {
            ModelFactory.registration(name).capabilities.preprocessing_profile()
            for name in self.NATIVE_NAN_GROUP
        }
        assert len(profiles) == 1

    def test_those_four_produce_one_identical_plan(self, frame):
        fingerprints = {
            prepare(
                frame,
                ModelFactory.registration(name).capabilities.preprocessing_profile(),
            )[0].fingerprint
            for name in self.NATIVE_NAN_GROUP
        }
        assert len(fingerprints) == 1

    def test_the_cache_serves_them_from_one_blueprint(self, frame):
        assert len(self._cache_over(frame, self.NATIVE_NAN_GROUP)) == 1

    def test_a_differently_capable_model_gets_its_own_blueprint(self, frame):
        cache = self._cache_over(
            frame,
            (
                "decision_tree_regressor",      # no scaler, dense, gaps kept
                "ridge_regression",             # scaler, sparse, imputed
                "gradient_boosting_regressor",  # no scaler, sparse, imputed
            ),
        )
        assert len(cache) == 3

    def test_the_two_linear_models_now_share_one(self, frame):
        """They ask for the same matrix, so they get the same blueprint.

        Their reasons differ -- a scale-dependent penalty against a
        rank-truncating solve -- and the cache is not told the reasons. An
        earlier version of this file asserted the opposite, because an earlier
        version of the catalog declared OLS scale-free.
        """
        assert len(self._cache_over(frame, ("ridge_regression", "linear_regression"))) == 1

    def test_the_nine_regressors_need_four_preprocessors(self, frame):
        assert len(self._cache_over(frame, NAMES)) == 4

    def test_and_all_eighteen_models_still_need_only_five(self, frame):
        from tests.conftest import BUILT_IN_CLASSIFIERS, BUILT_IN_PROFILE_COUNT

        cache = self._cache_over(frame, list(BUILT_IN_CLASSIFIERS) + NAMES)
        assert len(cache) == BUILT_IN_PROFILE_COUNT

    def test_a_regressor_reuses_a_classifier_blueprint(self, frame):
        """The cache key knows nothing about the task, so it cannot separate them."""
        cache = self._cache_over(
            frame, ("decision_tree_classifier", "decision_tree_regressor")
        )
        assert len(cache) == 1

    def test_different_execution_semantics_still_produce_different_keys(self, frame):
        """Reuse must not become collision: two real requirements, two blueprints."""
        cache = self._cache_over(
            frame, ("ridge_regression", "gradient_boosting_regressor")
        )
        assert len(cache) == 2


class TestNoFittedStateIsShared:
    """A reused blueprint must never mean a reused fit."""

    def test_two_models_on_one_blueprint_get_separate_preprocessors(self, frame):
        cache = BlueprintCache()
        config = PreprocessingConfig()
        dataset_profile = DataProfiler().profile(frame)
        builder = PreprocessorBuilder(config, cache=cache)
        profile = ModelFactory.registration(
            "decision_tree_regressor"
        ).capabilities.preprocessing_profile()
        plan = PreprocessingPlanner(config).plan(frame, dataset_profile, profile)

        first, second = builder.build(plan, frame), builder.build(plan, frame)
        assert first is not second
        assert first.transformer is not second.transformer

    def test_fitting_one_leaves_the_other_unfitted(self, frame):
        cache = BlueprintCache()
        config = PreprocessingConfig()
        dataset_profile = DataProfiler().profile(frame)
        builder = PreprocessorBuilder(config, cache=cache)
        profile = ModelFactory.registration(
            "random_forest_regressor"
        ).capabilities.preprocessing_profile()
        plan = PreprocessingPlanner(config).plan(frame, dataset_profile, profile)

        first, second = builder.build(plan, frame), builder.build(plan, frame)
        first.fit(frame)
        with pytest.raises(PreprocessingError, match="not been fitted"):
            second.transform(frame)

    def test_fitting_a_regressor_pipeline_does_not_touch_a_classifier_one(self, frame):
        """The two families share a blueprint; they must not share a fit."""
        config = PreprocessingConfig()
        dataset_profile = DataProfiler().profile(frame)
        builder = PreprocessorBuilder(config, cache=BlueprintCache())
        profile = ModelFactory.registration(
            "decision_tree_regressor"
        ).capabilities.preprocessing_profile()
        plan = PreprocessingPlanner(config).plan(frame, dataset_profile, profile)

        regression_side = builder.build(plan, frame)
        classification_side = builder.build(plan, frame)
        regression_side.fit(frame.iloc[:100])
        with pytest.raises(PreprocessingError, match="not been fitted"):
            classification_side.transform(frame)

    @pytest.mark.parametrize("name", NAMES)
    def test_each_build_produces_an_unfitted_estimator(self, name, frame, target):
        strategy = ModelFactory.strategy(name)
        _, preprocessor = prepare(frame, strategy.capabilities.preprocessing_profile())
        matrix = preprocessor.fit_transform(frame)

        fitted = strategy.build(**small_params(name)).fit(matrix, target)
        learned = {
            attribute
            for attribute in vars(fitted)
            if attribute.endswith("_") and not attribute.startswith("_")
        }
        assert learned
        fresh = strategy.build(**small_params(name))
        assert not (set(vars(fresh)) & learned)


class TestEdgeDatasets:
    """Shapes of data an analyst actually has, proved once across all nine."""

    @pytest.mark.parametrize("name", NAMES)
    def test_an_all_numeric_frame(self, name):
        rng = np.random.default_rng(11)
        frame = pd.DataFrame(
            {"a": rng.normal(0, 1, 150).round(3), "b": rng.normal(500, 90, 150).round(2)}
        )
        y = frame["a"].to_numpy() * 4.0 - 1.5
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        matrix = preprocessor.fit_transform(frame)
        assert make_estimator(name).fit(matrix, y).predict(matrix).shape == y.shape

    @pytest.mark.parametrize("name", NAMES)
    def test_a_single_boolean_feature(self, name):
        rng = np.random.default_rng(12)
        flag = rng.choice([True, False], 150)
        frame = pd.DataFrame({"flag": flag, "a": rng.normal(0, 1, 150).round(3)})
        y = flag.astype(float) * 10.0 + 0.5
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        matrix = preprocessor.fit_transform(frame)
        assert make_estimator(name).fit(matrix, y).predict(matrix).shape == y.shape

    @pytest.mark.parametrize("name", NAMES)
    def test_a_constant_target_is_fitted_without_complaint(self, name, frame):
        """Degenerate, and not the library's business to refuse."""
        y = np.full(len(frame), 7.25)
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        matrix = preprocessor.fit_transform(frame)
        predictions = make_estimator(name).fit(matrix, y).predict(matrix)
        assert np.isfinite(predictions).all()


class TestSparseAndNativeNaNDoNotCollide:
    """The S5 defect, re-proved against the regression catalog.

    The tree family accepts a ``csr_matrix``, and accepts ``NaN``, and refuses a
    ``csr_matrix`` containing ``NaN``. Declaring both capabilities would license
    S4 to build precisely that matrix. The regressors were measured
    independently and answer the same way; this proves the declarations hold on
    the frame that produces the collision.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def wide_with_gaps() -> pd.DataFrame:
        rng = np.random.default_rng(8)
        n = 400
        frame = pd.DataFrame(
            {
                "amount": rng.normal(500, 80, n).round(2),
                "city": [f"city_{i % 40}" for i in range(n)],
            }
        )
        frame.loc[frame.index[:25], "amount"] = np.nan
        return frame

    @pytest.fixture(scope="class")
    @staticmethod
    def onehot() -> PreprocessingConfig:
        return PreprocessingConfig(high_cardinality_policy=HighCardinalityPolicy.ONEHOT)

    @pytest.mark.parametrize("name", NAMES)
    def test_no_model_is_ever_handed_a_sparse_matrix_holding_gaps(
        self, name, wide_with_gaps, onehot
    ):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(
            wide_with_gaps, capabilities.preprocessing_profile(), onehot
        )
        matrix = preprocessor.fit_transform(wide_with_gaps)
        if not sparse.issparse(matrix):
            return
        assert not np.isnan(matrix.data).any() or capabilities.handles_missing_values

    @pytest.mark.parametrize("name", NAMES)
    def test_every_model_fits_what_this_frame_produces_for_it(
        self, name, wide_with_gaps, onehot
    ):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(
            wide_with_gaps, capabilities.preprocessing_profile(), onehot
        )
        matrix = preprocessor.fit_transform(wide_with_gaps)
        rng = np.random.default_rng(8)
        y = rng.normal(0, 1, len(wide_with_gaps)) * 5.0
        assert make_estimator(name).fit(matrix, y).predict(matrix).shape == y.shape

    def test_the_tree_family_is_given_a_dense_matrix(self, wide_with_gaps, onehot):
        for name in (
            "decision_tree_regressor",
            "random_forest_regressor",
            "extra_trees_regressor",
        ):
            capabilities = ModelFactory.registration(name).capabilities
            _, preprocessor = prepare(
                wide_with_gaps, capabilities.preprocessing_profile(), onehot
            )
            matrix = preprocessor.fit_transform(wide_with_gaps)
            assert not sparse.issparse(matrix), name
            assert np.isnan(matrix).any(), f"{name} lost its native NaN path"

    def test_the_combination_really_would_have_failed(self, wide_with_gaps, onehot):
        """Without the capability correction, this is the error users would see."""
        _, preprocessor = prepare(
            wide_with_gaps,
            PreprocessingProfile(
                requires_scaling=False,
                supports_sparse_input=True,
                handles_missing_values=True,
            ),
            onehot,
        )
        matrix = preprocessor.fit_transform(wide_with_gaps)
        assert sparse.issparse(matrix) and np.isnan(matrix.data).any()
        y = np.arange(len(wide_with_gaps)) * 1.5
        with pytest.raises(ValueError, match="NaN"):
            make_estimator("random_forest_regressor").fit(matrix, y)

    def test_the_baseline_survives_that_matrix_because_it_reads_nothing(
        self, wide_with_gaps, onehot
    ):
        capabilities = ModelFactory.registration("dummy_regressor").capabilities
        assert capabilities.supports_sparse_input and capabilities.handles_missing_values
        _, preprocessor = prepare(
            wide_with_gaps, capabilities.preprocessing_profile(), onehot
        )
        matrix = preprocessor.fit_transform(wide_with_gaps)
        assert sparse.issparse(matrix) and np.isnan(matrix.data).any()
        y = np.arange(len(wide_with_gaps)) * 1.5
        fitted = make_estimator("dummy_regressor").fit(matrix, y)
        assert fitted.predict(matrix).shape == y.shape


class TestS4ProtectsTheBinningEdge:
    """HistGradientBoostingRegressor dies on a column with no observed value.

    Not a scikit-learn message but a raw numpy one out of the binning code:
    ``ValueError("window shape cannot be larger than input array shape")``.
    Re-measured on the regressor. The model's NaN capability is still true -- one
    observed value is enough -- and the reason a user never meets this is that S4
    excludes an all-missing column at planning time.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def with_a_dead_column() -> pd.DataFrame:
        rng = np.random.default_rng(4)
        n = 300
        return pd.DataFrame({"a": rng.normal(0, 1, n).round(3), "dead": [np.nan] * n})

    def test_the_raw_estimator_really_does_die_on_it(self, with_a_dead_column):
        """Measured per version; the protection below holds on every one.

        scikit-learn 1.9.0 dies with numpy's ``window shape`` ValueError; 1.5.2
        through 1.8.0 fit the all-missing column without complaint. Whichever the
        installed version does is asserted precisely -- the only error accepted is
        that one -- and the two tests that follow assert, unconditionally, that
        S4 never hands the column over.
        """
        values = with_a_dead_column.to_numpy(dtype="float64")
        y = values[:, 0] * 3.0
        try:
            fitted = make_estimator("hist_gradient_boosting_regressor").fit(values, y)
        except ValueError as error:
            assert "window shape" in str(error)
        else:
            assert fitted.predict(values).shape == y.shape

    def test_but_the_plan_excludes_the_column_with_a_reason(self, with_a_dead_column):
        capabilities = ModelFactory.registration(
            "hist_gradient_boosting_regressor"
        ).capabilities
        plan, _ = prepare(with_a_dead_column, capabilities.preprocessing_profile())
        decision = plan.decision_for("dead")
        assert decision.reason_code == "no_observed_values"
        assert "dead" not in plan.included_features

    def test_so_the_real_path_fits(self, with_a_dead_column):
        capabilities = ModelFactory.registration(
            "hist_gradient_boosting_regressor"
        ).capabilities
        _, preprocessor = prepare(with_a_dead_column, capabilities.preprocessing_profile())
        matrix = preprocessor.fit_transform(with_a_dead_column)
        y = with_a_dead_column["a"].to_numpy() * 3.0
        fitted = make_estimator("hist_gradient_boosting_regressor").fit(matrix, y)
        assert fitted.predict(matrix).shape == y.shape

    @pytest.mark.parametrize("name", NAMES)
    def test_no_model_is_handed_a_column_with_nothing_in_it(self, name, with_a_dead_column):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(with_a_dead_column, capabilities.preprocessing_profile())
        matrix = densify(preprocessor.fit_transform(with_a_dead_column))
        assert not any(np.isnan(matrix[:, i]).all() for i in range(matrix.shape[1]))
