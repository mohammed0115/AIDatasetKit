"""Every classifier, through the real S4 preprocessing path.

The claim S5 has to make good on is narrow and testable: a model's
:class:`~aidatasetkit.core.types.PreprocessingProfile` is the *only* thing that
decides how its data is prepared. Nothing reads a model's name. So these tests
run the whole path -- profile, plan, build, fit, predict -- for all nine
classifiers over one realistic mixed frame, and then check the individual claims
that profile makes: a scaler where scaling is required and none where it is not,
imputation where NaN cannot be consumed and untouched NaN where it can, a sparse
matrix where sparse is accepted and a dense array where it is refused.

Nothing here evaluates a model. Which of these classifiers is any good is a
question for S7, and asking it now would be the beginning of AutoML.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.pipeline import Pipeline

from aidatasetkit.core.types import PreprocessingProfile, TaskType
from aidatasetkit.models import ModelFactory, default_registry
from aidatasetkit.preprocessing import (
    BlueprintCache,
    HighCardinalityPolicy,
    PreprocessingConfig,
    PreprocessingPlanner,
    PreprocessorBuilder,
    TargetLabelEncoder,
)
from aidatasetkit.profiling import DataProfiler, DataQualityInspector

CLASSIFIERS = default_registry().catalog(task="classification")
NAMES = [entry.canonical_name for entry in CLASSIFIERS]

#: Ensemble sizes are turned down by *signature*, exactly as the contract tests
#: do it, so that no test anywhere names a model to decide how to build it.
_SMALL = {"n_estimators": 5, "max_iter": 10}


def small_params(name: str) -> dict:
    import inspect

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

    The scales differ deliberately: a frame whose columns are already comparable
    could not tell a model that needs scaling apart from one that does not.
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


def make_target(frame: pd.DataFrame, classes: int = 2) -> pd.Series:
    """A deterministic label that does not depend on the column holding gaps."""
    score = frame["ratio"].to_numpy()
    if classes == 2:
        return pd.Series(np.where(score > np.median(score), "churn", "stay"))
    cuts = np.quantile(score, [1 / 3, 2 / 3])
    return pd.Series(np.array(["red", "green", "blue"])[np.digitize(score, cuts)])


def prepare(frame: pd.DataFrame, profile: PreprocessingProfile, config=None):
    """Profile, inspect, plan, and build -- the real path, no shortcuts."""
    config = config or PreprocessingConfig()
    dataset_profile = DataProfiler().profile(frame)
    quality = DataQualityInspector().inspect(frame, profile=dataset_profile)
    plan = PreprocessingPlanner(config).plan(frame, dataset_profile, profile, quality=quality)
    return plan, PreprocessorBuilder(config).build(plan, frame)


def densify(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return mixed_frame()


@pytest.fixture(scope="module")
def binary_target(frame) -> pd.Series:
    return make_target(frame, 2)


@pytest.fixture(scope="module")
def multiclass_target(frame) -> pd.Series:
    return make_target(frame, 3)


@pytest.mark.parametrize("name", NAMES)
class TestEveryClassifierRunsThroughPreprocessing:
    """One realistic frame, nine models, no model-specific handling anywhere."""

    def test_it_fits_and_predicts_on_a_binary_target(self, name, frame, binary_target):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        encoder = TargetLabelEncoder()
        y = encoder.fit_transform(binary_target)

        features = preprocessor.fit_transform(frame)
        estimator = make_estimator(name).fit(features, y)
        predictions = estimator.predict(preprocessor.transform(frame))

        assert predictions.shape == y.shape
        assert set(np.unique(predictions)) <= set(np.unique(y))

    def test_it_fits_inside_a_single_sklearn_pipeline(self, name, frame, binary_target):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        y = TargetLabelEncoder().fit_transform(binary_target)

        pipeline = Pipeline([("prep", preprocessor), ("model", make_estimator(name))])
        pipeline.fit(frame, y)
        assert pipeline.predict(frame).shape == y.shape

    def test_it_handles_a_multiclass_target_when_it_claims_to(
        self, name, frame, multiclass_target
    ):
        capabilities = ModelFactory.registration(name).capabilities
        if not capabilities.supports_multiclass:
            pytest.skip("model declares binary-only support")
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        y = TargetLabelEncoder().fit_transform(multiclass_target)

        estimator = make_estimator(name).fit(preprocessor.fit_transform(frame), y)
        assert len(estimator.classes_) == 3
        assert estimator.predict(preprocessor.transform(frame)).shape == y.shape

    def test_probabilities_are_real_when_it_claims_them(self, name, frame, binary_target):
        capabilities = ModelFactory.registration(name).capabilities
        if not capabilities.supports_predict_proba:
            pytest.skip("model declares no probability support")
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        y = TargetLabelEncoder().fit_transform(binary_target)

        estimator = make_estimator(name).fit(preprocessor.fit_transform(frame), y)
        probabilities = estimator.predict_proba(preprocessor.transform(frame))

        assert probabilities.shape == (len(frame), 2)
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_an_unseen_category_predicts_without_refitting_anything(
        self, name, frame, binary_target
    ):
        """Trained on Riyadh and Jeddah; scored on Dammam."""
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        y = TargetLabelEncoder().fit_transform(binary_target)
        estimator = make_estimator(name).fit(preprocessor.fit_transform(frame), y)

        unseen = frame.iloc[:30].assign(city="Dammam")
        transformed = preprocessor.transform(unseen)

        assert transformed.shape[1] == preprocessor.n_features_out
        assert estimator.predict(transformed).shape == (30,)

    def test_scoring_does_not_move_what_was_learned(self, name, frame, binary_target):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        preprocessor.fit(frame)
        before = list(preprocessor.get_feature_names_out())

        preprocessor.transform(frame.iloc[:30].assign(city="Dammam"))
        assert list(preprocessor.get_feature_names_out()) == before


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
        self, name, frame, binary_target
    ):
        """The end of the chain: whatever S4 emitted, the model must take it."""
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        features = preprocessor.fit_transform(frame)
        y = TargetLabelEncoder().fit_transform(binary_target)
        assert make_estimator(name).fit(features, y).predict(features).shape == y.shape


class TestTheDenseAndSparsePathsAreBothReal:
    """Proved with actual classifiers rather than invented profiles."""

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

    @pytest.mark.parametrize("name", ["knn_classifier", "gradient_boosting_classifier"])
    def test_a_sparse_capable_model_receives_a_sparse_matrix(self, name, wide, onehot):
        capabilities = ModelFactory.registration(name).capabilities
        assert capabilities.supports_sparse_input
        _, preprocessor = prepare(wide, capabilities.preprocessing_profile(), onehot)
        assert sparse.issparse(preprocessor.fit_transform(wide))

    @pytest.mark.parametrize(
        "name", ["gaussian_nb", "hist_gradient_boosting_classifier"]
    )
    def test_a_dense_only_model_receives_a_dense_array(self, name, wide, onehot):
        capabilities = ModelFactory.registration(name).capabilities
        assert not capabilities.supports_sparse_input
        _, preprocessor = prepare(wide, capabilities.preprocessing_profile(), onehot)
        assert not sparse.issparse(preprocessor.fit_transform(wide))

    @pytest.mark.parametrize(
        "name", ["gaussian_nb", "hist_gradient_boosting_classifier"]
    )
    def test_the_dense_only_model_would_have_refused_the_sparse_form(
        self, name, wide, onehot
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
        y = (wide["num"] > 0).astype(int).to_numpy()
        with pytest.raises((TypeError, ValueError)):
            make_estimator(name).fit(as_sparse, y)

    @pytest.mark.parametrize("name", ["gaussian_nb", "hist_gradient_boosting_classifier"])
    def test_and_accepts_the_dense_form_it_is_actually_given(self, name, wide, onehot):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(wide, capabilities.preprocessing_profile(), onehot)
        features = preprocessor.fit_transform(wide)
        y = (wide["num"] > 0).astype(int).to_numpy()
        assert make_estimator(name).fit(features, y).predict(features).shape == y.shape


class TestTheNativeNaNPathIsReal:
    """A model that understands gaps is not handed a median instead."""

    NATIVE = [
        "decision_tree_classifier",
        "random_forest_classifier",
        "extra_trees_classifier",
        "hist_gradient_boosting_classifier",
    ]

    @pytest.mark.parametrize("name", NATIVE)
    def test_the_gaps_reach_the_estimator(self, name, frame, binary_target):
        capabilities = ModelFactory.registration(name).capabilities
        assert capabilities.handles_missing_values
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        features = densify(preprocessor.fit_transform(frame))
        assert np.isnan(features).sum() == 20

    @pytest.mark.parametrize("name", NATIVE)
    def test_and_it_fits_on_them(self, name, frame, binary_target):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        features = preprocessor.fit_transform(frame)
        y = TargetLabelEncoder().fit_transform(binary_target)
        assert make_estimator(name).fit(features, y).predict(features).shape == y.shape

    @pytest.mark.parametrize(
        "name",
        [
            "gradient_boosting_classifier",
            "knn_classifier",
            "gaussian_nb",
            "logistic_regression",
        ],
    )
    def test_a_model_that_cannot_take_gaps_is_never_shown_one(self, name, frame):
        capabilities = ModelFactory.registration(name).capabilities
        assert not capabilities.handles_missing_values
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        assert not np.isnan(densify(preprocessor.fit_transform(frame))).any()

    @pytest.mark.parametrize(
        "name", ["gradient_boosting_classifier", "knn_classifier", "gaussian_nb"]
    )
    def test_and_would_genuinely_have_refused_one(self, name, frame):
        """The negative branch: imputation is required, not merely tidy."""
        native = PreprocessingProfile(
            requires_scaling=ModelFactory.registration(name).capabilities.requires_scaling,
            supports_sparse_input=False,
            handles_missing_values=True,
        )
        _, preprocessor = prepare(frame, native)
        with_gaps = densify(preprocessor.fit_transform(frame))
        assert np.isnan(with_gaps).any()
        y = (frame["ratio"] > frame["ratio"].median()).astype(int).to_numpy()
        with pytest.raises(ValueError, match="NaN|missing|infinity"):
            make_estimator(name).fit(with_gaps, y)


class TestScalingActuallyChangesTheNumbers:
    """requires_scaling is a claim about the data the model receives."""

    def test_a_scaled_model_receives_standardised_columns(self, frame):
        capabilities = ModelFactory.registration("knn_classifier").capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        out = densify(preprocessor.fit_transform(frame))
        numeric = out[:, :2]
        assert np.allclose(numeric.mean(axis=0), 0.0, atol=1e-8)
        assert np.allclose(numeric.std(axis=0), 1.0, atol=1e-8)

    def test_an_unscaled_model_receives_the_original_magnitudes(self, frame):
        capabilities = ModelFactory.registration("decision_tree_classifier").capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        out = densify(preprocessor.fit_transform(frame))
        assert np.nanmax(np.abs(out[:, 0])) > 1_000.0

    def test_the_two_receive_different_matrices(self, frame):
        scaled = ModelFactory.registration("knn_classifier").capabilities
        unscaled = ModelFactory.registration("gradient_boosting_classifier").capabilities
        _, a = prepare(frame, scaled.preprocessing_profile())
        _, b = prepare(frame, unscaled.preprocessing_profile())
        assert not np.allclose(densify(a.fit_transform(frame)), densify(b.fit_transform(frame)))


class TestBlueprintReuseIsCapabilityBased:
    """Nine classifiers, five preprocessors. Nothing consults a model name."""

    NATIVE_NAN_GROUP = (
        "decision_tree_classifier",
        "random_forest_classifier",
        "extra_trees_classifier",
        "hist_gradient_boosting_classifier",
    )

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
        cache = BlueprintCache()
        config = PreprocessingConfig()
        dataset_profile = DataProfiler().profile(frame)
        quality = DataQualityInspector().inspect(frame, profile=dataset_profile)
        builder = PreprocessorBuilder(config, cache=cache)

        for name in self.NATIVE_NAN_GROUP:
            profile = ModelFactory.registration(name).capabilities.preprocessing_profile()
            plan = PreprocessingPlanner(config).plan(
                frame, dataset_profile, profile, quality=quality
            )
            builder.build(plan, frame)

        assert len(cache) == 1

    def test_a_differently_capable_model_gets_its_own_blueprint(self, frame):
        cache = BlueprintCache()
        config = PreprocessingConfig()
        dataset_profile = DataProfiler().profile(frame)
        quality = DataQualityInspector().inspect(frame, profile=dataset_profile)
        builder = PreprocessorBuilder(config, cache=cache)

        for name in ("decision_tree_classifier", "knn_classifier", "gaussian_nb"):
            profile = ModelFactory.registration(name).capabilities.preprocessing_profile()
            plan = PreprocessingPlanner(config).plan(
                frame, dataset_profile, profile, quality=quality
            )
            builder.build(plan, frame)

        assert len(cache) == 3

    def test_the_nine_classifiers_need_five_preprocessors(self, frame):
        cache = BlueprintCache()
        config = PreprocessingConfig()
        dataset_profile = DataProfiler().profile(frame)
        quality = DataQualityInspector().inspect(frame, profile=dataset_profile)
        builder = PreprocessorBuilder(config, cache=cache)

        for name in NAMES:
            profile = ModelFactory.registration(name).capabilities.preprocessing_profile()
            plan = PreprocessingPlanner(config).plan(
                frame, dataset_profile, profile, quality=quality
            )
            builder.build(plan, frame)

        assert len(cache) == 5


class TestNoFittedStateIsShared:
    """A reused blueprint must never mean a reused fit."""

    def test_two_models_on_one_blueprint_get_separate_preprocessors(self, frame):
        cache = BlueprintCache()
        config = PreprocessingConfig()
        dataset_profile = DataProfiler().profile(frame)
        builder = PreprocessorBuilder(config, cache=cache)
        profile = ModelFactory.registration(
            "decision_tree_classifier"
        ).capabilities.preprocessing_profile()
        plan = PreprocessingPlanner(config).plan(frame, dataset_profile, profile)

        first, second = builder.build(plan, frame), builder.build(plan, frame)
        assert first is not second
        assert first.transformer is not second.transformer

    def test_fitting_one_leaves_the_other_unfitted(self, frame):
        from aidatasetkit.core.exceptions import PreprocessingError

        cache = BlueprintCache()
        config = PreprocessingConfig()
        dataset_profile = DataProfiler().profile(frame)
        builder = PreprocessorBuilder(config, cache=cache)
        profile = ModelFactory.registration(
            "random_forest_classifier"
        ).capabilities.preprocessing_profile()
        plan = PreprocessingPlanner(config).plan(frame, dataset_profile, profile)

        first, second = builder.build(plan, frame), builder.build(plan, frame)
        first.fit(frame)
        with pytest.raises(PreprocessingError, match="not been fitted"):
            second.transform(frame)

    @pytest.mark.parametrize("name", NAMES)
    def test_each_build_produces_an_unfitted_estimator(self, name, frame, binary_target):
        strategy = ModelFactory.strategy(name)
        _, preprocessor = prepare(
            frame, strategy.capabilities.preprocessing_profile()
        )
        features = preprocessor.fit_transform(frame)
        y = TargetLabelEncoder().fit_transform(binary_target)

        strategy.build(**small_params(name)).fit(features, y)
        assert not hasattr(strategy.build(**small_params(name)), "classes_")


class TestTargetLabelsSurviveTheRoundTrip:
    @pytest.mark.parametrize("name", NAMES)
    def test_string_labels_come_back_as_they_went_in(self, name, frame, binary_target):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        encoder = TargetLabelEncoder()
        y = encoder.fit_transform(binary_target)

        estimator = make_estimator(name).fit(preprocessor.fit_transform(frame), y)
        decoded = encoder.inverse_transform(
            estimator.predict(preprocessor.transform(frame))
        )
        assert set(decoded) <= {"churn", "stay"}
        assert decoded.dtype == np.asarray(binary_target).dtype

    @pytest.mark.parametrize("name", NAMES)
    def test_three_string_labels_survive_too(self, name, frame, multiclass_target):
        capabilities = ModelFactory.registration(name).capabilities
        if not capabilities.supports_multiclass:
            pytest.skip("model declares binary-only support")
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        encoder = TargetLabelEncoder()
        y = encoder.fit_transform(multiclass_target)

        estimator = make_estimator(name).fit(preprocessor.fit_transform(frame), y)
        decoded = encoder.inverse_transform(
            estimator.predict(preprocessor.transform(frame))
        )
        assert set(decoded) <= {"red", "green", "blue"}


class TestThePositiveLabelIsNeverAssumedToBeIndexOne:
    """"churn" sorts before "stay", so the positive class is column 0 here.

    S7 will read a probability column by this mapping. Nothing in S5 may quietly
    assume the interesting class is the second one.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def imbalanced(frame) -> pd.Series:
        """Deliberately not 50/50.

        On a balanced target the baseline returns [0.5, 0.5] and its two
        probability columns are indistinguishable, so a test that column 0 is
        not column 1 would pass vacuously for every other model and fail for the
        one model that is behaving correctly. A 30/70 split gives every model,
        the baseline included, two columns that can actually be told apart.
        """
        threshold = frame["ratio"].quantile(0.3)
        return pd.Series(np.where(frame["ratio"] <= threshold, "churn", "stay"))

    @pytest.fixture(scope="class")
    @staticmethod
    def encoder(imbalanced) -> TargetLabelEncoder:
        return TargetLabelEncoder(positive_label="churn").fit(imbalanced)

    def test_the_positive_class_really_is_encoded_as_zero(self, encoder):
        assert list(encoder.classes_) == ["churn", "stay"]
        assert encoder.positive_column_index() == 0

    @pytest.mark.parametrize("name", NAMES)
    def test_the_estimator_agrees_with_the_encoder_about_class_order(
        self, name, frame, imbalanced, encoder
    ):
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        y = encoder.transform(imbalanced)
        estimator = make_estimator(name).fit(preprocessor.fit_transform(frame), y)
        assert list(estimator.classes_) == [0, 1]

    @pytest.mark.parametrize("name", NAMES)
    def test_reading_the_positive_column_by_index_one_would_be_wrong(
        self, name, frame, imbalanced, encoder
    ):
        """The two columns must be distinguishable, or the test proves nothing."""
        capabilities = ModelFactory.registration(name).capabilities
        if not capabilities.supports_predict_proba:
            pytest.skip("model declares no probability support")
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        y = encoder.transform(imbalanced)
        estimator = make_estimator(name).fit(preprocessor.fit_transform(frame), y)

        probabilities = estimator.predict_proba(preprocessor.transform(frame))
        churn = probabilities[:, encoder.positive_column_index()]
        assert np.allclose(churn, probabilities[:, 0])
        assert not np.allclose(churn, probabilities[:, 1])


class TestEdgeDatasets:
    """Shapes of data an analyst actually has, proved once across all nine."""

    @pytest.mark.parametrize("name", NAMES)
    def test_an_all_numeric_frame(self, name):
        rng = np.random.default_rng(11)
        frame = pd.DataFrame(
            {"a": rng.normal(0, 1, 150).round(3), "b": rng.normal(500, 90, 150).round(2)}
        )
        y = (frame["a"] > 0).astype(int).to_numpy()
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        features = preprocessor.fit_transform(frame)
        assert make_estimator(name).fit(features, y).predict(features).shape == y.shape

    @pytest.mark.parametrize("name", NAMES)
    def test_a_single_boolean_feature(self, name):
        rng = np.random.default_rng(12)
        flag = rng.choice([True, False], 150)
        frame = pd.DataFrame({"flag": flag, "a": rng.normal(0, 1, 150).round(3)})
        y = flag.astype(int)
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        features = preprocessor.fit_transform(frame)
        assert make_estimator(name).fit(features, y).predict(features).shape == y.shape

    @pytest.mark.parametrize("name", NAMES)
    def test_an_imbalanced_target_is_fitted_without_rebalancing(self, name, frame):
        """AIDatasetKit must not resample or choose class weights on its own."""
        rng = np.random.default_rng(13)
        y = (rng.random(len(frame)) < 0.05).astype(int)
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(frame, capabilities.preprocessing_profile())
        features = preprocessor.fit_transform(frame)
        estimator = make_estimator(name)
        assert estimator.get_params().get("class_weight", None) is None
        assert estimator.fit(features, y).predict(features).shape == y.shape


class TestModelsNeverSeeThePreprocessingLayer:
    """The integration direction, asserted rather than assumed."""

    def test_the_factory_can_choose_a_profile_without_preprocessing_imported(self):
        import subprocess
        import sys

        script = (
            "import sys;"
            "from aidatasetkit.models import ModelFactory;"
            "p = ModelFactory.registration('random_forest').capabilities."
            "preprocessing_profile();"
            "print(p.key);"
            "print(any(m.startswith('aidatasetkit.preprocessing') for m in sys.modules))"
        )
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.split()
        assert out[0] == "scaling=0,sparse=0,native_nan=1"
        assert out[1] == "False"

    def test_no_classification_module_mentions_preprocessing(self):
        from pathlib import Path

        package = Path(__file__).resolve().parents[2] / "aidatasetkit" / "models"
        for path in package.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "aidatasetkit.preprocessing" not in source, path

    def test_no_model_strategy_names_a_preprocessing_step(self):
        """A strategy that mentioned a scaler would be preprocessing by another name."""
        from pathlib import Path

        package = Path(__file__).resolve().parents[2] / "aidatasetkit" / "models"
        for path in (package / "classification").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for forbidden in ("StandardScaler", "SimpleImputer", "OneHotEncoder"):
                assert forbidden not in source, f"{path} names {forbidden}"


class TestSparseAndNativeNaNDoNotCollide:
    """A defect found by capability verification, before it could ship.

    The tree family accepts a ``csr_matrix``, and accepts ``NaN``, and refuses a
    ``csr_matrix`` containing ``NaN``. Declaring both capabilities licensed S4 to
    build precisely that matrix, and the failure needed nothing exotic: a
    high-cardinality categorical column, some missing numbers, and a random
    forest. It surfaced as a raw scikit-learn ``ValueError`` at fit.
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
        features = preprocessor.fit_transform(wide_with_gaps)
        if not sparse.issparse(features):
            return
        assert not np.isnan(features.data).any() or capabilities.handles_missing_values

    @pytest.mark.parametrize("name", NAMES)
    def test_every_model_fits_what_this_frame_produces_for_it(
        self, name, wide_with_gaps, onehot
    ):
        """The end-to-end reproduction, now passing for all nine."""
        capabilities = ModelFactory.registration(name).capabilities
        _, preprocessor = prepare(
            wide_with_gaps, capabilities.preprocessing_profile(), onehot
        )
        features = preprocessor.fit_transform(wide_with_gaps)
        rng = np.random.default_rng(8)
        y = (rng.random(len(wide_with_gaps)) < 0.4).astype(int)
        assert make_estimator(name).fit(features, y).predict(features).shape == y.shape

    def test_the_tree_family_is_now_given_a_dense_matrix(self, wide_with_gaps, onehot):
        for name in (
            "decision_tree_classifier",
            "random_forest_classifier",
            "extra_trees_classifier",
        ):
            capabilities = ModelFactory.registration(name).capabilities
            _, preprocessor = prepare(
                wide_with_gaps, capabilities.preprocessing_profile(), onehot
            )
            features = preprocessor.fit_transform(wide_with_gaps)
            assert not sparse.issparse(features), name
            assert np.isnan(features).any(), f"{name} lost its native NaN path"

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
        features = preprocessor.fit_transform(wide_with_gaps)
        assert sparse.issparse(features) and np.isnan(features.data).any()
        y = (np.arange(len(wide_with_gaps)) % 2).astype(int)
        with pytest.raises(ValueError, match="NaN"):
            make_estimator("random_forest_classifier").fit(features, y)

    def test_the_baseline_survives_that_matrix_because_it_reads_nothing(
        self, wide_with_gaps, onehot
    ):
        capabilities = ModelFactory.registration("dummy_classifier").capabilities
        assert capabilities.supports_sparse_input and capabilities.handles_missing_values
        _, preprocessor = prepare(
            wide_with_gaps, capabilities.preprocessing_profile(), onehot
        )
        features = preprocessor.fit_transform(wide_with_gaps)
        assert sparse.issparse(features) and np.isnan(features.data).any()
        y = (np.arange(len(wide_with_gaps)) % 2).astype(int)
        fitted = make_estimator("dummy_classifier").fit(features, y)
        assert fitted.predict(features).shape == y.shape


class TestS4ProtectsTheBinningEdge:
    """HistGradientBoosting dies on a column with no observed value at all.

    Not a sklearn message but a raw numpy one out of the binning code:
    ``ValueError("window shape cannot be larger than input array shape")``. The
    model's NaN capability is still true -- one observed value is enough -- and
    the reason a user never meets this is that S4 excludes an all-missing column
    at planning time. That protection is asserted here rather than trusted.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def with_a_dead_column() -> pd.DataFrame:
        rng = np.random.default_rng(4)
        n = 300
        return pd.DataFrame(
            {"a": rng.normal(0, 1, n).round(3), "dead": [np.nan] * n}
        )

    def test_the_raw_estimator_really_does_die_on_it(self, with_a_dead_column):
        values = with_a_dead_column.to_numpy(dtype="float64")
        y = (values[:, 0] > 0).astype(int)
        with pytest.raises(ValueError, match="window shape"):
            make_estimator("hist_gradient_boosting_classifier").fit(values, y)

    def test_but_the_plan_excludes_the_column_with_a_reason(self, with_a_dead_column):
        capabilities = ModelFactory.registration(
            "hist_gradient_boosting_classifier"
        ).capabilities
        plan, _ = prepare(with_a_dead_column, capabilities.preprocessing_profile())
        decision = plan.decision_for("dead")
        assert decision.reason_code == "no_observed_values"
        assert "dead" not in plan.included_features

    def test_so_the_real_path_fits(self, with_a_dead_column):
        capabilities = ModelFactory.registration(
            "hist_gradient_boosting_classifier"
        ).capabilities
        _, preprocessor = prepare(with_a_dead_column, capabilities.preprocessing_profile())
        features = preprocessor.fit_transform(with_a_dead_column)
        y = (with_a_dead_column["a"] > 0).astype(int).to_numpy()
        fitted = make_estimator("hist_gradient_boosting_classifier").fit(features, y)
        assert fitted.predict(features).shape == y.shape

    @pytest.mark.parametrize("name", NAMES)
    def test_no_model_is_handed_a_column_with_nothing_in_it(self, name, with_a_dead_column):
        capabilities = ModelFactory.registration(name).capabilities
        plan, preprocessor = prepare(
            with_a_dead_column, capabilities.preprocessing_profile()
        )
        features = densify(preprocessor.fit_transform(with_a_dead_column))
        assert not any(np.isnan(features[:, i]).all() for i in range(features.shape[1]))
