"""Regression tests for defects the adversarial review found and reproduced.

Each failed before the fix beside it. What most of them have in common is a plan
that read correctly and a fitted pipeline that did something else -- a column
silently dropped, a sentinel silently merged, a NaN reaching a model that had
declared it could not take one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from aidatasetkit.core.exceptions import (
    AmbiguousFeatureRoleError,
    SchemaError,
    UnsupportedTaskError,
)
from aidatasetkit.core.types import PreprocessingProfile
from aidatasetkit.preprocessing import (
    CategoricalImputation,
    FeatureAction,
    PreprocessingConfig,
    PreprocessingPlanner,
    PreprocessorBuilder,
    TargetLabelEncoder,
)
from aidatasetkit.profiling import DataProfiler, DataQualityInspector

UNSCALED = PreprocessingProfile(False, True, False)
SCALED = PreprocessingProfile(True, True, False)
NATIVE_NAN = PreprocessingProfile(False, True, True)


def build(frame, config=None, profile=UNSCALED, pass_frame=True):
    config = config or PreprocessingConfig()
    dataset_profile = DataProfiler().profile(frame)
    quality = DataQualityInspector().inspect(frame, profile=dataset_profile)
    plan = PreprocessingPlanner(config).plan(frame, dataset_profile, profile, quality=quality)
    return plan, PreprocessorBuilder(config).build(plan, frame if pass_frame else None)


def dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


class TestSentinelTravelsOnThePlan:
    """The planner checked the sentinel against the data; the builder threw it away."""

    @pytest.fixture
    def colliding(self) -> pd.DataFrame:
        return pd.DataFrame({"c": ["__missing__", "a", "b", None] * 30})

    @pytest.fixture
    def config(self) -> PreprocessingConfig:
        return PreprocessingConfig(categorical_imputation=CategoricalImputation.CONSTANT)

    def test_the_plan_records_the_collision_free_sentinel(self, colliding, config):
        plan, _ = build(colliding, config)
        assert plan.categorical_sentinel == "__missing___"

    def test_building_without_a_frame_still_uses_it(self, colliding, config):
        _, pre = build(colliding, config, pass_frame=False)
        pre.fit(colliding)
        categories = set(
            pre.transformer.named_transformers_["nominal"].named_steps["encoder"].categories_[0]
        )
        assert "__missing__" in categories and "__missing___" in categories

    def test_real_and_generated_missing_stay_distinguishable(self, colliding, config):
        _, pre = build(colliding, config, pass_frame=False)
        out = dense(pre.fit_transform(colliding))
        assert out.shape[1] == 4

    def test_the_sentinel_is_part_of_the_plan_identity(self, colliding, config):
        plan, _ = build(colliding, config)
        assert plan.categorical_sentinel in plan.to_dict()["categorical_sentinel"]


class TestOrdinalNeverTakesASentinel:
    """A sentinel has no rank, so the encoder refused it outright."""

    @pytest.fixture
    def education(self) -> pd.DataFrame:
        return pd.DataFrame({"e": ["lo", "mid", "hi", None] * 30})

    @pytest.fixture
    def config(self) -> PreprocessingConfig:
        return PreprocessingConfig(
            ordinal_orders={"e": ["lo", "mid", "hi"]},
            categorical_imputation=CategoricalImputation.CONSTANT,
        )

    def test_it_fits(self, education, config):
        _, pre = build(education, config)
        assert dense(pre.fit_transform(education)).shape == (120, 1)

    def test_the_plan_says_most_frequent_is_used_instead(self, education, config):
        plan, _ = build(education, config)
        decision = plan.decision_for("e")
        assert "most_frequent_imputation" in decision.steps
        assert "no position in the order" in decision.reason

    def test_no_level_below_the_lowest_is_invented(self, education, config):
        _, pre = build(education, config)
        out = dense(pre.fit_transform(education)).ravel()
        assert out.min() >= 0.0
        assert set(np.unique(out)) <= {0.0, 1.0, 2.0}


class TestMappedColumnsAreTreatedAsNumbers:
    """A mapped column skipped imputation and scaling entirely."""

    @pytest.fixture
    def gendered(self) -> pd.DataFrame:
        return pd.DataFrame({"g": ["M", "F", None] * 40})

    @pytest.fixture
    def config(self) -> PreprocessingConfig:
        return PreprocessingConfig(explicit_mappings={"g": {"M": 1, "F": 0}})

    def test_no_nan_reaches_a_model_that_cannot_take_one(self, gendered, config):
        _, pre = build(gendered, config, UNSCALED)
        assert not np.isnan(dense(pre.fit_transform(gendered))).any()

    def test_scaling_is_applied_when_the_profile_requires_it(self, gendered, config):
        _, pre = build(gendered, config, SCALED)
        pre.fit(gendered)
        steps = pre.transformer.named_transformers_["nominal_mapped"].named_steps
        assert "scaler" in steps

    def test_no_imputer_when_the_model_handles_gaps_itself(self, gendered, config):
        _, pre = build(gendered, config, NATIVE_NAN)
        pre.fit(gendered)
        steps = pre.transformer.named_transformers_["nominal_mapped"].named_steps
        assert "imputer" not in steps


class TestFullyMissingColumns:
    """Planned INCLUDE, then dropped inside the imputer without a word."""

    @pytest.fixture
    def empty_column(self) -> pd.DataFrame:
        return pd.DataFrame({"a": [np.nan] * 60, "b": np.arange(60.0) % 13})

    def test_it_is_excluded_with_a_reason(self, empty_column):
        plan, _ = build(empty_column)
        decision = plan.decision_for("a")
        assert decision.action is FeatureAction.EXCLUDE
        assert decision.reason_code == "no_observed_values"

    def test_the_output_width_matches_the_plan(self, empty_column):
        plan, pre = build(empty_column)
        pre.fit(empty_column)
        assert pre.n_features_out == len(plan.included_features) == 1

    def test_nothing_disappears_between_plan_and_fit(self, empty_column):
        plan, pre = build(empty_column)
        pre.fit(empty_column)
        assert set(pre.lineage()) == set(plan.included_features)


class TestExtensionDtypes:
    """pd.NA reached scikit-learn whenever no imputer stood in front of it."""

    @pytest.mark.parametrize("dtype", ["Int64", "Float64"])
    def test_a_nullable_column_fits_under_a_native_nan_profile(self, dtype):
        frame = pd.DataFrame({"n": pd.Series([1, 2, None, 4] * 15, dtype=dtype)})
        _, pre = build(frame, profile=NATIVE_NAN)
        out = dense(pre.fit_transform(frame))
        assert out.shape == (60, 1)
        assert np.isnan(out).sum() == 15

    @pytest.mark.parametrize("dtype", ["Int64", "Float64"])
    def test_a_nullable_column_fits_under_an_imputing_profile(self, dtype):
        frame = pd.DataFrame({"n": pd.Series([1, 2, None, 4] * 15, dtype=dtype)})
        _, pre = build(frame, profile=UNSCALED)
        assert not np.isnan(dense(pre.fit_transform(frame))).any()

    def test_a_mixed_type_object_column_still_encodes(self):
        frame = pd.DataFrame({"c": pd.Series([1, "a", 2.5, "b"] * 20, dtype=object)})
        _, pre = build(frame)
        assert dense(pre.fit_transform(frame)).shape[0] == 80


class TestNothingVanishesSilently:
    def test_a_column_missing_from_the_profile_is_refused(self):
        frame = pd.DataFrame({"a": np.arange(60.0) % 13, "b": np.arange(60.0) % 7})
        partial = DataProfiler().profile(frame[["a"]])
        with pytest.raises(SchemaError, match="not in the supplied DatasetProfile"):
            PreprocessingPlanner().plan(frame, partial, UNSCALED)

    def test_multiindex_columns_are_refused_clearly(self):
        frame = pd.DataFrame({("a", "x"): [1.0, 2.0] * 30, ("a", "y"): [3.0, 4.0] * 30})
        with pytest.raises(SchemaError, match="MultiIndex"):
            build(frame)

    def test_that_refusal_says_how_to_fix_it(self):
        frame = pd.DataFrame({("a", "x"): [1.0, 2.0] * 30, ("a", "y"): [3.0, 4.0] * 30})
        with pytest.raises(SchemaError, match="Flatten"):
            build(frame)


class TestRegressionTargetGuard:
    def test_a_continuous_target_is_refused_without_a_profile(self):
        """Silently turning prices into class indices destroys the quantity."""
        with pytest.raises(UnsupportedTaskError, match="non-integral"):
            TargetLabelEncoder().fit(pd.Series(np.linspace(0.0, 1000.0, 200)))

    def test_integer_coded_classes_still_encode(self):
        encoded = TargetLabelEncoder().fit_transform(pd.Series([0, 1, 0, 1] * 30))
        assert set(np.unique(encoded)) == {0, 1}

    def test_whole_number_floats_are_still_allowed(self):
        encoded = TargetLabelEncoder().fit_transform(pd.Series([0.0, 1.0] * 30))
        assert set(np.unique(encoded)) == {0, 1}


class TestConfigIntegrity:
    def test_a_column_cannot_have_both_an_order_and_a_mapping(self):
        frame = pd.DataFrame({"e": ["lo", "hi"] * 30})
        config = PreprocessingConfig(
            ordinal_orders={"e": ["lo", "hi"]}, explicit_mappings={"e": {"lo": 0, "hi": 1}}
        )
        with pytest.raises(AmbiguousFeatureRoleError, match="two different encodings"):
            build(frame, config)

    def test_the_plan_s_config_snapshot_cannot_be_edited_afterwards(self):
        frame = pd.DataFrame({"a": np.arange(60.0) % 13})
        plan, _ = build(frame)
        with pytest.raises(TypeError):
            plan.config["numeric_scaler"] = "tampered"


class TestFeatureNameCollisions:
    def test_colliding_output_names_do_not_break_the_name_lookup(self):
        """A one-hot level can match another column's label exactly."""
        frame = pd.DataFrame({"a_x": np.arange(60.0) % 13, "a": ["x", "y"] * 30})
        _, pre = build(frame)
        pre.fit(frame)
        names = list(pre.get_feature_names_out())
        assert len(names) == len(set(names))
        assert len(names) == dense(pre.transform(frame)).shape[1]


class TestBuilderHonoursThePlansConfiguration:
    """The builder used its own config, so what ran could differ from the plan."""

    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"a": np.arange(60.0) % 13})

    def test_a_mismatched_configuration_is_refused(self, frame):
        from aidatasetkit.core.exceptions import PreprocessingError
        from aidatasetkit.preprocessing import NumericImputation

        planned = PreprocessingConfig(numeric_imputation=NumericImputation.MEDIAN)
        dataset_profile = DataProfiler().profile(frame)
        plan = PreprocessingPlanner(planned).plan(frame, dataset_profile, UNSCALED)

        other = PreprocessingConfig(numeric_imputation=NumericImputation.MEAN)
        with pytest.raises(PreprocessingError, match="different preprocessing configuration"):
            PreprocessorBuilder(other).build(plan, frame)

    def test_the_refusal_names_both_values(self, frame):
        from aidatasetkit.core.exceptions import PreprocessingError
        from aidatasetkit.preprocessing import NumericScaler

        planned = PreprocessingConfig(numeric_scaler=NumericScaler.STANDARD)
        dataset_profile = DataProfiler().profile(frame)
        plan = PreprocessingPlanner(planned).plan(frame, dataset_profile, SCALED)

        other = PreprocessingConfig(numeric_scaler=NumericScaler.ROBUST)
        with pytest.raises(PreprocessingError, match="numeric_scaler"):
            PreprocessorBuilder(other).build(plan, frame)

    def test_the_matching_configuration_builds(self, frame):
        config = PreprocessingConfig()
        dataset_profile = DataProfiler().profile(frame)
        plan = PreprocessingPlanner(config).plan(frame, dataset_profile, UNSCALED)
        assert PreprocessorBuilder(config).build(plan, frame) is not None
