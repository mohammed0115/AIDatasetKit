"""Tests for building and fitting preprocessors.

The properties that matter here are safety properties: what the fitted
transformer learned, what it refuses to learn from, and what it leaves alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from aidatasetkit.core.exceptions import (
    NoUsableFeaturesError,
    PreprocessingError,
    SchemaError,
)
from aidatasetkit.core.types import PreprocessingProfile
from aidatasetkit.preprocessing import (
    BlueprintCache,
    CategoricalImputation,
    HighCardinalityPolicy,
    NumericImputation,
    PreprocessingConfig,
    PreprocessingPlanner,
    PreprocessorBuilder,
    UnknownOrdinalPolicy,
)
from aidatasetkit.profiling import DataProfiler, DataQualityInspector

SCALED = PreprocessingProfile(True, True, False)
UNSCALED = PreprocessingProfile(False, True, False)
DENSE = PreprocessingProfile(False, False, False)
NATIVE_NAN = PreprocessingProfile(False, True, True)


def make(frame, profile=UNSCALED, config=None, target=None):
    """Plan and build in one step, the way a caller would."""
    config = config or PreprocessingConfig()
    dataset_profile = DataProfiler().profile(frame)
    quality = DataQualityInspector().inspect(frame, target=target, profile=dataset_profile)
    plan = PreprocessingPlanner(config).plan(
        frame, dataset_profile, profile, quality=quality, target=target
    )
    return plan, PreprocessorBuilder(config).build(plan, frame)


@pytest.fixture
def train() -> pd.DataFrame:
    index = np.arange(200)
    return pd.DataFrame(
        {
            "age": np.where(index % 11 == 0, np.nan, (20 + index % 45).astype("float64")),
            "city": np.where(
                index % 13 == 0,
                None,
                [["Riyadh", "Jeddah", "Mecca"][v % 3] for v in index],
            ),
            "active": (index % 3 == 0),
        }
    )


class TestNumeric:
    def test_missing_numbers_are_imputed_from_training(self, train):
        _, pre = make(train)
        out = pre.fit_transform(train)
        assert not np.isnan(np.asarray(out, dtype="float64")).any()

    def test_the_median_comes_from_training_rows_only(self, train):
        _, pre = make(train)
        pre.fit(train)
        imputer = pre.transformer.named_transformers_["numeric"].named_steps["imputer"]
        expected = float(train["age"].median())
        assert imputer.statistics_[0] == pytest.approx(expected)

    def test_a_huge_test_value_does_not_change_the_learned_median(self, train):
        _, pre = make(train)
        pre.fit(train)
        learned = float(
            pre.transformer.named_transformers_["numeric"].named_steps["imputer"].statistics_[0]
        )
        test = pd.DataFrame({"age": [1_000_000.0], "city": ["Riyadh"], "active": [True]})
        pre.transform(test)
        after = float(
            pre.transformer.named_transformers_["numeric"].named_steps["imputer"].statistics_[0]
        )
        assert after == learned

    @pytest.mark.parametrize(
        ("strategy", "expected"),
        [(NumericImputation.MEDIAN, "median"), (NumericImputation.MEAN, "mean")],
    )
    def test_the_imputation_strategy_is_configurable(self, train, strategy, expected):
        config = PreprocessingConfig(numeric_imputation=strategy)
        _, pre = make(train, config=config)
        pre.fit(train)
        assert pre.transformer.named_transformers_["numeric"].named_steps[
            "imputer"
        ].strategy == expected

    def test_constant_imputation_uses_the_configured_value(self, train):
        config = PreprocessingConfig(
            numeric_imputation=NumericImputation.CONSTANT, numeric_fill_value=-1.0
        )
        _, pre = make(train, config=config)
        pre.fit(train)
        imputer = pre.transformer.named_transformers_["numeric"].named_steps["imputer"]
        assert imputer.statistics_[0] == -1.0

    def test_scaling_is_applied_only_when_the_profile_asks(self, train):
        _, scaled = make(train, SCALED)
        _, plain = make(train, UNSCALED)
        scaled.fit(train)
        plain.fit(train)
        assert "scaler" in scaled.transformer.named_transformers_["numeric"].named_steps
        assert "scaler" not in plain.transformer.named_transformers_["numeric"].named_steps

    def test_a_native_nan_model_gets_no_imputer(self, train):
        _, pre = make(train, NATIVE_NAN)
        pre.fit(train)
        steps = pre.transformer.named_transformers_["numeric"].named_steps
        assert "imputer" not in steps

    @pytest.mark.parametrize("dtype", ["Int64", "Float64"])
    def test_nullable_numeric_dtypes_transform(self, dtype):
        frame = pd.DataFrame({"n": pd.Series([1, 2, None, 4] * 30, dtype=dtype)})
        _, pre = make(frame)
        assert pre.fit_transform(frame).shape == (120, 1)


class TestCategorical:
    def test_categories_are_learned_from_training(self, train):
        _, pre = make(train)
        pre.fit(train)
        encoder = pre.transformer.named_transformers_["nominal"].named_steps["encoder"]
        assert sorted(encoder.categories_[0]) == ["Jeddah", "Mecca", "Riyadh"]

    def test_an_unknown_category_does_not_crash_inference(self, train):
        _, pre = make(train)
        pre.fit(train)
        test = pd.DataFrame({"age": [30.0], "city": ["Dammam"], "active": [True]})
        assert pre.transform(test).shape[0] == 1

    def test_an_unknown_category_becomes_all_zeros(self, train):
        plan, pre = make(train)
        pre.fit(train)
        test = pd.DataFrame({"age": [30.0], "city": ["Dammam"], "active": [True]})
        names = list(pre.get_feature_names_out())
        out = np.asarray(_dense(pre.transform(test)))
        city_columns = [i for i, n in enumerate(names) if n.startswith("city")]
        assert out[0, city_columns].sum() == 0.0

    def test_an_unknown_category_never_joins_the_learned_vocabulary(self, train):
        _, pre = make(train)
        pre.fit(train)
        test = pd.DataFrame({"age": [30.0], "city": ["Dammam"], "active": [True]})
        pre.transform(test)
        encoder = pre.transformer.named_transformers_["nominal"].named_steps["encoder"]
        assert "Dammam" not in list(encoder.categories_[0])

    def test_missing_labels_are_imputed(self, train):
        _, pre = make(train)
        out = _dense(pre.fit_transform(train))
        assert not np.isnan(out).any()

    def test_a_boolean_column_is_one_hot_encoded(self, train):
        _, pre = make(train)
        pre.fit(train)
        assert any("active" in name for name in pre.get_feature_names_out())

    def test_a_nullable_boolean_column_works(self):
        frame = pd.DataFrame({"flag": pd.Series([True, None, False] * 40, dtype="boolean")})
        _, pre = make(frame)
        assert _dense(pre.fit_transform(frame)).shape[0] == 120

    def test_missing_and_unknown_together(self):
        """["A", None, "B"] trained, ["C", None] at inference."""
        frame = pd.DataFrame({"c": ["A", None, "B"] * 40})
        _, pre = make(frame)
        pre.fit(frame)
        out = _dense(pre.transform(pd.DataFrame({"c": ["C", None]})))
        assert out.shape[0] == 2


class TestSentinelCollision:
    def test_a_real_sentinel_value_is_not_merged_with_generated_ones(self):
        frame = pd.DataFrame({"c": ["__missing__", "a", "b", None] * 30})
        config = PreprocessingConfig(
            categorical_imputation=CategoricalImputation.CONSTANT
        )
        _, pre = make(frame, config=config)
        pre.fit(frame)
        encoder = pre.transformer.named_transformers_["nominal"].named_steps["encoder"]
        categories = set(encoder.categories_[0])
        assert "__missing__" in categories
        assert "__missing___" in categories

    def test_rows_that_had_data_stay_distinguishable(self):
        frame = pd.DataFrame(
            {"c": ["__missing__"] * 40 + ["a"] * 40 + ["b"] * 40 + [None] * 40}
        )
        config = PreprocessingConfig(categorical_imputation=CategoricalImputation.CONSTANT)
        _, pre = make(frame, config=config)
        out = _dense(pre.fit_transform(frame))
        assert out.sum(axis=0).tolist().count(40.0) == 4

    def test_no_collision_means_the_plain_sentinel_is_used(self):
        frame = pd.DataFrame({"c": ["a", "b", "c", None] * 30})
        config = PreprocessingConfig(categorical_imputation=CategoricalImputation.CONSTANT)
        _, pre = make(frame, config=config)
        pre.fit(frame)
        encoder = pre.transformer.named_transformers_["nominal"].named_steps["encoder"]
        assert "__missing__" in set(encoder.categories_[0])


class TestOrdinal:
    @pytest.fixture
    def education(self) -> pd.DataFrame:
        index = np.arange(120)
        return pd.DataFrame(
            {"education": [["low", "mid", "high"][v % 3] for v in index]}
        )

    @pytest.fixture
    def config(self) -> PreprocessingConfig:
        return PreprocessingConfig(ordinal_orders={"education": ["low", "mid", "high"]})

    def test_the_supplied_order_is_used(self, education, config):
        _, pre = make(education, config=config)
        out = np.asarray(_dense(pre.fit_transform(education)))
        by_level = dict(zip(education["education"], out.ravel()))
        assert by_level["low"] < by_level["mid"] < by_level["high"]

    def test_an_unknown_level_fails_clearly_by_default(self, education, config):
        _, pre = make(education, config=config)
        pre.fit(education)
        with pytest.raises(ValueError, match="unknown categories"):
            pre.transform(pd.DataFrame({"education": ["doctorate"]}))

    def test_an_unknown_level_can_be_encoded_by_explicit_policy(self, education):
        config = PreprocessingConfig(
            ordinal_orders={"education": ["low", "mid", "high"]},
            unknown_ordinal_policy=UnknownOrdinalPolicy.ENCODE,
            unknown_ordinal_value=-1,
        )
        _, pre = make(education, config=config)
        pre.fit(education)
        out = _dense(pre.transform(pd.DataFrame({"education": ["doctorate"]})))
        assert out.ravel()[0] == -1.0


class TestExplicitMapping:
    def test_a_supplied_mapping_replaces_the_encoder(self):
        frame = pd.DataFrame({"gender": ["M", "F"] * 60})
        config = PreprocessingConfig(explicit_mappings={"gender": {"M": 1, "F": 0}})
        plan, pre = make(frame, config=config)
        out = np.asarray(_dense(pre.fit_transform(frame)))
        assert set(out.ravel().tolist()) == {0.0, 1.0}
        assert plan.decision_for("gender").steps == ("explicit_mapping",)

    def test_an_unmapped_value_is_refused_rather_than_guessed(self):
        frame = pd.DataFrame({"gender": ["M", "F"] * 60})
        config = PreprocessingConfig(explicit_mappings={"gender": {"M": 1, "F": 0}})
        _, pre = make(frame, config=config)
        pre.fit(frame)
        with pytest.raises(PreprocessingError, match="no entry in the mapping"):
            pre.transform(pd.DataFrame({"gender": ["X"]}))

    def test_the_source_frame_is_untouched(self):
        frame = pd.DataFrame({"gender": ["M", "F"] * 60})
        before = frame.copy(deep=True)
        config = PreprocessingConfig(explicit_mappings={"gender": {"M": 1, "F": 0}})
        _, pre = make(frame, config=config)
        pre.fit_transform(frame)
        pd.testing.assert_frame_equal(frame, before)


class TestSparseAndDense:
    def test_sparse_output_is_preserved_for_a_capable_model(self):
        index = np.arange(600)
        frame = pd.DataFrame({"c": [f"v{v % 40}" for v in index]})
        config = PreprocessingConfig(high_cardinality_policy=HighCardinalityPolicy.ONEHOT)
        _, pre = make(frame, UNSCALED, config=config)
        assert sparse.issparse(pre.fit_transform(frame))

    def test_a_dense_only_model_gets_a_dense_matrix(self):
        index = np.arange(600)
        frame = pd.DataFrame({"c": [f"v{v % 40}" for v in index]})
        config = PreprocessingConfig(high_cardinality_policy=HighCardinalityPolicy.ONEHOT)
        _, pre = make(frame, DENSE, config=config)
        assert not sparse.issparse(pre.fit_transform(frame))

    def test_high_cardinality_does_not_explode_by_default(self):
        """The default holds the column back rather than making 5,000 columns."""
        index = np.arange(20_000)
        frame = pd.DataFrame(
            {"code": [f"c{v}" for v in index], "n": (index % 37).astype("float64")}
        )
        plan, pre = make(frame)
        assert "code" in {str(c) for c in plan.review_features}
        assert pre.fit_transform(frame).shape[1] == 1


class TestBlueprintCacheAndState:
    def test_two_profiles_that_match_share_one_cache_entry(self, train):
        cache = BlueprintCache()
        builder = PreprocessorBuilder(cache=cache)
        dataset_profile = DataProfiler().profile(train)
        planner = PreprocessingPlanner()

        first = planner.plan(train, dataset_profile, PreprocessingProfile(True, True, False))
        second = planner.plan(train, dataset_profile, PreprocessingProfile(True, True, False))
        builder.build(first, train)
        builder.build(second, train)
        assert len(cache) == 1

    def test_a_different_profile_gets_its_own_entry(self, train):
        cache = BlueprintCache()
        builder = PreprocessorBuilder(cache=cache)
        dataset_profile = DataProfiler().profile(train)
        planner = PreprocessingPlanner()
        builder.build(planner.plan(train, dataset_profile, SCALED), train)
        builder.build(planner.plan(train, dataset_profile, UNSCALED), train)
        assert len(cache) == 2

    def test_the_cache_is_not_keyed_by_model_name(self, train):
        """Nothing in the key mentions a model."""
        _, pre = make(train)
        cache = BlueprintCache()
        key = cache.key_for(pre.plan)
        assert key == (pre.plan.preprocessing_profile.key, pre.plan.fingerprint)
        assert "dummy" not in str(key) and "logistic" not in str(key)

    def test_two_preprocessors_from_one_plan_are_independent(self, train):
        plan, first = make(train)
        second = PreprocessorBuilder().build(plan, train)
        assert first is not second
        assert first.transformer is not second.transformer

    def test_fitting_one_does_not_fit_the_other(self, train):
        plan, first = make(train)
        second = PreprocessorBuilder().build(plan, train)
        first.fit(train)
        assert first.is_fitted
        assert not second.is_fitted

    def test_fitting_one_does_not_change_the_other_s_statistics(self, train):
        plan, first = make(train)
        second = PreprocessorBuilder().build(plan, train)
        first.fit(train)
        other = train.copy()
        other["age"] = other["age"] * 100.0
        second.fit(other)
        a = first.transformer.named_transformers_["numeric"].named_steps["imputer"].statistics_[0]
        b = second.transformer.named_transformers_["numeric"].named_steps["imputer"].statistics_[0]
        assert a != pytest.approx(b)

    def test_transforming_before_fitting_is_refused(self, train):
        _, pre = make(train)
        with pytest.raises(PreprocessingError, match="has not been fitted"):
            pre.transform(train)


class TestFeatureNamesAndLineage:
    def test_feature_names_are_available(self, train):
        _, pre = make(train)
        pre.fit(train)
        names = list(pre.get_feature_names_out())
        assert "age" in names
        assert any(name.startswith("city_") for name in names)

    def test_the_transformed_count_matches_the_names(self, train):
        _, pre = make(train)
        pre.fit(train)
        assert pre.n_features_out == len(pre.get_feature_names_out())
        assert pre.n_features_out == _dense(pre.transform(train)).shape[1]

    def test_lineage_maps_each_input_to_its_outputs(self, train):
        _, pre = make(train)
        pre.fit(train)
        lineage = pre.lineage()
        assert lineage["age"] == ("age",)
        assert set(lineage["city"]) == {"city_Jeddah", "city_Mecca", "city_Riyadh"}

    def test_lineage_covers_every_included_feature(self, train):
        plan, pre = make(train)
        pre.fit(train)
        assert set(pre.lineage()) == set(plan.included_features)

    def test_a_category_containing_a_separator_is_still_counted_correctly(self):
        frame = pd.DataFrame({"c": ["a_b", "c_d", "e_f"] * 40, "n": np.arange(120.0) % 31})
        _, pre = make(frame)
        pre.fit(frame)
        assert len(pre.lineage()["c"]) == 3


class TestImmutabilityAndOrder:
    def test_fitting_does_not_modify_the_frame(self, train):
        before = train.copy(deep=True)
        _, pre = make(train)
        pre.fit_transform(train)
        pd.testing.assert_frame_equal(train, before)

    def test_transforming_does_not_modify_the_frame(self, train):
        _, pre = make(train)
        pre.fit(train)
        before = train.copy(deep=True)
        pre.transform(train)
        pd.testing.assert_frame_equal(train, before)

    def test_dtypes_and_index_survive(self, train):
        dtypes = train.dtypes.copy()
        index = train.index.copy()
        _, pre = make(train)
        pre.fit_transform(train)
        pd.testing.assert_series_equal(train.dtypes, dtypes)
        pd.testing.assert_index_equal(train.index, index)

    def test_rows_are_neither_dropped_nor_reordered(self, train):
        _, pre = make(train)
        out = _dense(pre.fit_transform(train))
        assert out.shape[0] == len(train)

    def test_row_order_is_preserved(self):
        frame = pd.DataFrame({"n": [3.0, 1.0, 2.0] * 40})
        _, pre = make(frame)
        out = _dense(pre.fit_transform(frame)).ravel()
        np.testing.assert_allclose(out, frame["n"].to_numpy())

    def test_a_non_default_index_is_not_reset(self, train):
        shifted = train.set_index(pd.RangeIndex(1000, 1000 + len(train)))
        _, pre = make(shifted)
        out = _dense(pre.fit_transform(shifted))
        assert out.shape[0] == len(shifted)
        assert shifted.index[0] == 1000


class TestEdgeCases:
    def test_all_numeric_works(self):
        frame = pd.DataFrame({"a": np.arange(60.0) % 13, "b": np.arange(60.0) % 17})
        _, pre = make(frame)
        assert _dense(pre.fit_transform(frame)).shape == (60, 2)

    def test_all_categorical_works(self):
        frame = pd.DataFrame({"a": ["x", "y"] * 30, "b": ["p", "q", "r"] * 20})
        _, pre = make(frame)
        assert _dense(pre.fit_transform(frame)).shape[0] == 60

    def test_a_single_feature_works(self):
        frame = pd.DataFrame({"only": np.arange(60.0) % 13})
        _, pre = make(frame)
        assert _dense(pre.fit_transform(frame)).shape == (60, 1)

    def test_no_usable_features_fails_before_any_model_sees_it(self):
        frame = pd.DataFrame({"constant": ["x"] * 60})
        with pytest.raises(NoUsableFeaturesError, match="nothing to fit"):
            make(frame)

    def test_that_failure_points_at_the_plan(self):
        frame = pd.DataFrame({"constant": ["x"] * 60})
        with pytest.raises(NoUsableFeaturesError, match="plan.describe"):
            make(frame)

    def test_a_missing_column_at_transform_is_reported_clearly(self, train):
        _, pre = make(train)
        pre.fit(train)
        with pytest.raises(SchemaError, match="not in the frame"):
            pre.transform(train.drop(columns=["city"]))

    def test_a_non_dataframe_is_refused(self, train):
        _, pre = make(train)
        with pytest.raises(SchemaError, match="DataFrame is required"):
            pre.fit(np.zeros((3, 2)))

    def test_integer_column_labels_round_trip(self):
        frame = pd.DataFrame({0: np.arange(60.0) % 13, 1: ["a", "b"] * 30})
        plan, pre = make(frame)
        pre.fit(frame)
        assert plan.label_normalisation.applied
        assert set(pre.lineage()) == {0, 1}


def _dense(matrix):
    """Return a dense array whether or not the transformer produced a sparse one."""
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


class TestNumericTextConversion:
    """The opt-in path. Left unwired, the policy skipped the review hold and then
    one-hot encoded the number-strings -- the opposite of what was asked for."""

    @pytest.fixture
    def texty(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"amount": [str(v % 37) if v % 20 else "unknown" for v in range(200)]}
        )

    def test_the_default_holds_the_column_back(self, texty):
        from aidatasetkit.preprocessing import FeatureAction, NumericTextPolicy

        config = PreprocessingConfig(numeric_text_policy=NumericTextPolicy.REVIEW)
        plan, _ = _plan_only(texty, config)
        assert plan.decision_for("amount").action is FeatureAction.REVIEW

    def test_opting_in_actually_parses_the_column(self, texty):
        from aidatasetkit.preprocessing import FeatureRole, NumericTextPolicy

        config = PreprocessingConfig(numeric_text_policy=NumericTextPolicy.CONVERT)
        plan, pre = make(texty, config=config)
        decision = plan.decision_for("amount")
        assert decision.role is FeatureRole.NUMERIC
        assert "numeric_text_conversion" in decision.steps
        out = _dense(pre.fit_transform(texty))
        assert out.shape == (200, 1)
        assert np.asarray(out).dtype.kind == "f"

    def test_opting_in_does_not_one_hot_encode_the_numbers(self, texty):
        from aidatasetkit.preprocessing import NumericTextPolicy

        config = PreprocessingConfig(numeric_text_policy=NumericTextPolicy.CONVERT)
        _, pre = make(texty, config=config)
        pre.fit(texty)
        assert pre.n_features_out == 1

    def test_unparseable_values_become_missing_and_are_imputed(self, texty):
        from aidatasetkit.preprocessing import NumericTextPolicy

        config = PreprocessingConfig(numeric_text_policy=NumericTextPolicy.CONVERT)
        _, pre = make(texty, config=config)
        out = _dense(pre.fit_transform(texty))
        assert not np.isnan(out).any()

    def test_the_source_frame_is_unchanged(self, texty):
        from aidatasetkit.preprocessing import NumericTextPolicy

        before = texty.copy(deep=True)
        config = PreprocessingConfig(numeric_text_policy=NumericTextPolicy.CONVERT)
        _, pre = make(texty, config=config)
        pre.fit_transform(texty)
        pd.testing.assert_frame_equal(texty, before)


def _plan_only(frame, config):
    """Plan without building, for columns the plan will hold back."""
    dataset_profile = DataProfiler().profile(frame)
    quality = DataQualityInspector().inspect(frame, profile=dataset_profile)
    plan = PreprocessingPlanner(config).plan(
        frame, dataset_profile, UNSCALED, quality=quality
    )
    return plan, None
