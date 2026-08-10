"""Tests for feature detection and planning.

The plan is the product: what will happen to each column, and why. These tests
assert on decisions and reasons rather than on transformer objects, because the
planner deliberately builds none.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core.exceptions import (
    AmbiguousFeatureRoleError,
    ConfigurationError,
    MissingOrdinalOrderError,
    SchemaError,
)
from aidatasetkit.core.types import PreprocessingProfile
from aidatasetkit.preprocessing import (
    FeatureAction,
    FeatureDetector,
    FeatureRole,
    HighCardinalityPolicy,
    NumericScaler,
    NumericTextPolicy,
    PreprocessingConfig,
    PreprocessingPlanner,
)
from aidatasetkit.profiling import DataProfiler, DataQualityInspector

SCALED = PreprocessingProfile(
    requires_scaling=True, supports_sparse_input=True, handles_missing_values=False
)
UNSCALED = PreprocessingProfile(
    requires_scaling=False, supports_sparse_input=True, handles_missing_values=False
)
NATIVE_NAN = PreprocessingProfile(
    requires_scaling=False, supports_sparse_input=True, handles_missing_values=True
)


def plan_for(frame, profile=UNSCALED, config=None, target=None, quality=None):
    config = config or PreprocessingConfig()
    dataset_profile = DataProfiler().profile(frame)
    if quality is None:
        quality = DataQualityInspector().inspect(
            frame, target=target, profile=dataset_profile
        )
    return PreprocessingPlanner(config).plan(
        frame, dataset_profile, profile, quality=quality, target=target
    )


@pytest.fixture
def simple_frame() -> pd.DataFrame:
    index = np.arange(200)
    return pd.DataFrame(
        {
            "age": (20 + index % 45).astype("float64"),
            "city": [["Riyadh", "Jeddah", "Dammam"][v % 3] for v in index],
            "active": (index % 2 == 0),
            "churn": (index % 5 == 0).astype("int64"),
        }
    )


class TestRoleDetection:
    def test_a_numeric_column_is_numeric(self, simple_frame):
        plan = plan_for(simple_frame, target="churn")
        assert plan.spec_for("age").role is FeatureRole.NUMERIC

    def test_a_multi_valued_label_is_nominal(self, simple_frame):
        assert plan_for(simple_frame, target="churn").spec_for("city").role is (
            FeatureRole.NOMINAL
        )

    def test_a_boolean_column_is_binary_categorical_not_numeric(self, simple_frame):
        """bool subclasses int in Python; the schema classifier does not agree."""
        spec = plan_for(simple_frame, target="churn").spec_for("active")
        assert spec.role is FeatureRole.BINARY_CATEGORICAL
        assert spec.role_source == "schema:boolean"

    def test_a_two_valued_string_column_is_binary_categorical(self):
        frame = pd.DataFrame({"gender": ["M", "F"] * 60})
        assert plan_for(frame).spec_for("gender").role is FeatureRole.BINARY_CATEGORICAL

    def test_a_datetime_column_is_datetime(self):
        frame = pd.DataFrame(
            {"when": pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(60), "D")}
        )
        assert plan_for(frame).spec_for("when").role is FeatureRole.DATETIME

    def test_every_role_records_the_rule_that_chose_it(self, simple_frame):
        for spec in plan_for(simple_frame, target="churn").specs:
            assert spec.role_source

    @pytest.mark.parametrize(
        ("dtype", "values"),
        [
            ("Int64", [1, 2, None, 4]),
            ("Float64", [1.5, 2.5, None, 4.5]),
            ("boolean", [True, None, False, True]),
            ("string", ["a", "b", None, "a"]),
            ("category", ["x", "y", "x", "y"]),
        ],
    )
    def test_pandas_extension_dtypes_are_classified(self, dtype, values):
        frame = pd.DataFrame({"c": pd.Series(values * 15, dtype=dtype)})
        spec = plan_for(frame).spec_for("c")
        assert spec.role is not FeatureRole.UNSUPPORTED


class TestCapabilityDrivenScaling:
    def test_a_model_that_needs_scaling_gets_a_scaler(self, simple_frame):
        decision = plan_for(simple_frame, SCALED, target="churn").decision_for("age")
        assert "standard_scaling" in decision.steps

    def test_a_model_that_does_not_need_scaling_gets_none(self, simple_frame):
        decision = plan_for(simple_frame, UNSCALED, target="churn").decision_for("age")
        assert not any("scaling" in step for step in decision.steps)

    def test_the_difference_follows_the_profile_not_the_model_name(self, simple_frame):
        scaled = plan_for(simple_frame, SCALED, target="churn")
        unscaled = plan_for(simple_frame, UNSCALED, target="churn")
        assert scaled.fingerprint != unscaled.fingerprint

    def test_the_two_built_in_models_differ_by_capability(self, simple_frame):
        from aidatasetkit.models import default_registry

        plans = {
            entry.canonical_name: plan_for(
                simple_frame, entry.capabilities.preprocessing_profile(), target="churn"
            )
            for entry in default_registry().catalog()
        }
        steps = {
            name: plan.decision_for("age").steps for name, plan in plans.items()
        }
        assert any("standard_scaling" in s for s in steps.values())
        assert any(not any("scaling" in step for step in s) for s in steps.values())

    def test_native_missing_support_removes_the_imputer(self, simple_frame):
        decision = plan_for(simple_frame, NATIVE_NAN, target="churn").decision_for("age")
        assert not any("imputation" in step for step in decision.steps)

    def test_imputation_is_planned_even_when_training_data_is_complete(self, simple_frame):
        """A test row may have a gap where the training rows had none."""
        decision = plan_for(simple_frame, UNSCALED, target="churn").decision_for("age")
        assert "median_imputation" in decision.steps

    @pytest.mark.parametrize(
        ("scaler", "step"),
        [
            (NumericScaler.STANDARD, "standard_scaling"),
            (NumericScaler.MINMAX, "minmax_scaling"),
            (NumericScaler.ROBUST, "robust_scaling"),
        ],
    )
    def test_the_scaler_is_configurable(self, simple_frame, scaler, step):
        config = PreprocessingConfig(numeric_scaler=scaler)
        plan = plan_for(simple_frame, SCALED, config=config, target="churn")
        assert step in plan.decision_for("age").steps


class TestExclusionAndReview:
    def test_a_constant_column_is_excluded_with_a_reason(self):
        frame = pd.DataFrame({"country": ["SA"] * 100, "age": np.arange(100.0) % 37})
        decision = plan_for(frame).decision_for("country")
        assert decision.action is FeatureAction.EXCLUDE
        assert decision.reason_code == "constant_feature"

    def test_a_near_constant_column_is_kept_and_flagged(self):
        frame = pd.DataFrame({"plan": ["basic"] * 199 + ["pro"], "age": np.arange(200.0) % 37})
        decision = plan_for(frame).decision_for("plan")
        assert decision.action is FeatureAction.INCLUDE
        assert decision.requires_review

    def test_a_probable_identifier_is_held_for_review_not_dropped(self):
        frame = pd.DataFrame({"customer_id": [f"C{i:05d}" for i in range(200)]})
        decision = plan_for(frame).decision_for("customer_id")
        assert decision.action is FeatureAction.REVIEW
        assert decision.reason_code == "possible_id_like"
        assert "force_include" in decision.reason

    def test_a_confirmed_identifier_is_excluded_outright(self):
        frame = pd.DataFrame({"customer_id": [f"C{i:05d}" for i in range(200)]})
        config = PreprocessingConfig(confirmed_id_columns=("customer_id",))
        decision = plan_for(frame, config=config).decision_for("customer_id")
        assert decision.action is FeatureAction.EXCLUDE
        assert decision.reason_code == "confirmed_identifier"

    def test_a_high_cardinality_category_is_held_back(self):
        frame = pd.DataFrame({"city": [f"c{i}" for i in range(400)]})
        decision = plan_for(frame).decision_for("city")
        assert decision.action is FeatureAction.REVIEW
        assert decision.reason_code in ("high_cardinality", "possible_id_like")

    def test_high_cardinality_can_be_accepted_explicitly(self):
        index = np.arange(600)
        frame = pd.DataFrame({"city": [f"c{v % 120}" for v in index]})
        config = PreprocessingConfig(high_cardinality_policy=HighCardinalityPolicy.ONEHOT)
        decision = plan_for(frame, config=config).decision_for("city")
        assert decision.action is FeatureAction.INCLUDE

    def test_a_datetime_column_is_held_for_review(self):
        frame = pd.DataFrame(
            {"when": pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(60), "D")}
        )
        decision = plan_for(frame).decision_for("when")
        assert decision.action is FeatureAction.REVIEW
        assert "datetime" in decision.reason_code

    def test_an_infinite_column_is_held_for_review(self):
        values = np.arange(100.0) % 37
        values[3] = np.inf
        frame = pd.DataFrame({"ratio": values})
        decision = plan_for(frame).decision_for("ratio")
        assert decision.action is FeatureAction.REVIEW
        assert decision.reason_code == "infinite_values"

    @pytest.mark.parametrize("bad", [np.inf, -np.inf])
    def test_both_signs_of_infinity_are_held_back(self, bad):
        values = np.arange(100.0) % 37
        values[3] = bad
        frame = pd.DataFrame({"ratio": values})
        decision = plan_for(frame).decision_for("ratio")
        assert decision.action is FeatureAction.REVIEW
        assert "does not support infinity" in decision.reason

    def test_a_column_whose_categories_collide_as_text_is_held_back(self):
        """1 and "1" are different categories; encoding would merge them."""
        frame = pd.DataFrame({"c": pd.Series([1, "1", 2] * 40, dtype=object)})
        decision = plan_for(frame).decision_for("c")
        assert decision.action is FeatureAction.REVIEW
        assert decision.reason_code == "categorical_type_collision"

    def test_numeric_text_is_held_for_review_not_converted(self):
        frame = pd.DataFrame({"amount": [str(v % 37) for v in range(200)]})
        decision = plan_for(frame).decision_for("amount")
        assert decision.action is FeatureAction.REVIEW
        assert decision.reason_code == "possible_numeric_stored_as_text"

    def test_nothing_is_ever_dropped_without_a_recorded_decision(self, simple_frame):
        plan = plan_for(simple_frame, target="churn")
        planned = {str(d.feature) for d in plan.decisions}
        expected = {str(c) for c in simple_frame.columns} - {"churn"}
        assert planned == expected


class TestLeakagePolicy:
    @pytest.fixture
    def leaky(self) -> pd.DataFrame:
        index = np.arange(200)
        churn = (index % 5 == 0).astype("int64")
        return pd.DataFrame(
            {"copy": churn, "age": (20 + index % 40).astype("float64"), "churn": churn}
        )

    def test_a_feature_proved_equal_to_the_target_is_excluded(self, leaky):
        decision = plan_for(leaky, target="churn").decision_for("copy")
        assert decision.action is FeatureAction.EXCLUDE
        assert decision.reason_code == "target_leakage_exact_duplicate"

    def test_that_exclusion_can_be_switched_off(self, leaky):
        config = PreprocessingConfig(drop_exact_target_duplicates=False)
        assert plan_for(leaky, config=config, target="churn").decision_for("copy").action is (
            FeatureAction.INCLUDE
        )

    def test_a_merely_suspected_leak_is_kept_and_flagged(self):
        """Heuristic findings must not remove a feature by themselves."""
        index = np.arange(200)
        churn = (index % 5 == 0).astype("int64")
        frame = pd.DataFrame(
            {
                "churn_reason": np.where(churn == 1, "price", "none"),
                "churn": churn,
            }
        )
        decision = plan_for(frame, target="churn").decision_for("churn_reason")
        assert decision.action is FeatureAction.INCLUDE
        assert decision.requires_review


class TestOverrides:
    def test_an_override_beats_schema_detection(self, simple_frame):
        config = PreprocessingConfig(nominal_features=("age",))
        spec = plan_for(simple_frame, config=config, target="churn").spec_for("age")
        assert spec.role is FeatureRole.NOMINAL
        assert spec.role_source == "override:nominal_features"

    def test_force_include_overrides_a_review_hold(self):
        frame = pd.DataFrame({"customer_id": [f"C{i:05d}" for i in range(200)]})
        config = PreprocessingConfig(force_include=("customer_id",))
        decision = plan_for(frame, config=config).decision_for("customer_id")
        assert decision.action is FeatureAction.INCLUDE
        assert decision.requires_review

    def test_force_exclude_beats_everything(self, simple_frame):
        config = PreprocessingConfig(force_exclude=("age",))
        assert plan_for(simple_frame, config=config, target="churn").decision_for("age").action is (
            FeatureAction.EXCLUDE
        )

    def test_an_unknown_override_column_is_refused(self, simple_frame):
        config = PreprocessingConfig(numeric_features=("nope",))
        with pytest.raises(SchemaError, match="not in the frame"):
            plan_for(simple_frame, config=config, target="churn")

    def test_contradictory_roles_are_refused(self, simple_frame):
        config = PreprocessingConfig(
            numeric_features=("age",),
            ordinal_features=("age",),
            ordinal_orders={"age": [1, 2]},
        )
        with pytest.raises(AmbiguousFeatureRoleError, match="one role"):
            plan_for(simple_frame, config=config, target="churn")

    def test_the_target_cannot_be_overridden_as_a_feature(self, simple_frame):
        config = PreprocessingConfig(numeric_features=("churn",))
        with pytest.raises(AmbiguousFeatureRoleError, match="target"):
            plan_for(simple_frame, config=config, target="churn")

    def test_include_and_exclude_together_are_refused(self):
        with pytest.raises(ConfigurationError, match="force-included and force-excluded"):
            PreprocessingConfig(force_include=("a",), force_exclude=("a",))

    def test_a_bare_string_override_is_refused(self):
        with pytest.raises(ConfigurationError, match="sequence of column labels"):
            PreprocessingConfig(numeric_features="age")


class TestOrdinal:
    @pytest.fixture
    def education(self) -> pd.DataFrame:
        index = np.arange(180)
        return pd.DataFrame(
            {"education": [["high_school", "bachelor", "master"][v % 3] for v in index]}
        )

    def test_an_explicit_order_makes_a_column_ordinal(self, education):
        config = PreprocessingConfig(
            ordinal_orders={"education": ["high_school", "bachelor", "master"]}
        )
        spec = plan_for(education, config=config).spec_for("education")
        assert spec.role is FeatureRole.ORDINAL
        assert spec.ordinal_order == ("high_school", "bachelor", "master")

    def test_order_is_never_inferred_without_configuration(self, education):
        assert plan_for(education).spec_for("education").role is FeatureRole.NOMINAL

    def test_declaring_ordinal_without_an_order_is_refused(self, education):
        config = PreprocessingConfig(ordinal_features=("education",))
        with pytest.raises(MissingOrdinalOrderError, match="no order was given"):
            plan_for(education, config=config)

    def test_the_refusal_explains_why_inference_is_refused(self, education):
        config = PreprocessingConfig(ordinal_features=("education",))
        with pytest.raises(MissingOrdinalOrderError) as error:
            plan_for(education, config=config)
        message = str(error.value)
        assert "alphabet" in message and "frequency" in message

    def test_a_one_level_order_is_refused(self):
        with pytest.raises(ConfigurationError, match="at least two levels"):
            PreprocessingConfig(ordinal_orders={"e": ["only"]})

    def test_a_repeated_level_is_refused(self):
        with pytest.raises(ConfigurationError, match="repeats a level"):
            PreprocessingConfig(ordinal_orders={"e": ["a", "b", "a"]})


class TestDeterminismAndSerialisation:
    def test_the_same_input_yields_the_same_plan(self, simple_frame):
        first = plan_for(simple_frame, SCALED, target="churn")
        second = plan_for(simple_frame, SCALED, target="churn")
        assert first.to_dict() == second.to_dict()
        assert first.fingerprint == second.fingerprint

    def test_the_plan_serialises_to_plain_structures(self, simple_frame):
        payload = json.loads(json.dumps(plan_for(simple_frame, SCALED, target="churn").to_dict()))
        assert payload["numeric_features"] == ["age"]
        assert payload["decisions"]

    def test_no_transformer_leaks_into_the_plan(self, simple_frame):
        text = json.dumps(plan_for(simple_frame, SCALED, target="churn").to_dict())
        for forbidden in ("SimpleImputer", "OneHotEncoder", "StandardScaler", "ColumnTransformer"):
            assert forbidden not in text

    def test_the_fingerprint_ignores_wording_but_not_structure(self, simple_frame):
        scaled = plan_for(simple_frame, SCALED, target="churn")
        again = plan_for(simple_frame, SCALED, target="churn")
        unscaled = plan_for(simple_frame, UNSCALED, target="churn")
        assert scaled.fingerprint == again.fingerprint != unscaled.fingerprint

    def test_the_plan_describes_itself_for_a_human(self, simple_frame):
        description = plan_for(simple_frame, SCALED, target="churn").describe()
        assert "age" in description and "Reason:" in description

    def test_planning_does_not_modify_the_frame(self, simple_frame):
        before = simple_frame.copy(deep=True)
        plan_for(simple_frame, SCALED, target="churn")
        pd.testing.assert_frame_equal(simple_frame, before)


class TestLabelHandling:
    def test_duplicate_column_labels_are_refused(self):
        frame = pd.DataFrame(np.arange(20).reshape(10, 2), columns=["a", "a"])
        with pytest.raises(SchemaError, match="Duplicate column labels"):
            plan_for(frame)

    def test_integer_labels_are_normalised_reversibly(self):
        frame = pd.DataFrame({0: np.arange(100.0) % 31, 1: np.arange(100.0) % 17})
        plan = plan_for(frame)
        assert plan.label_normalisation.applied
        assert plan.label_normalisation.original("0") == 0

    def test_string_labels_need_no_normalisation(self, simple_frame):
        assert not plan_for(simple_frame, target="churn").label_normalisation.applied

    def test_labels_that_collide_when_stringified_are_refused(self):
        frame = pd.DataFrame({1: np.arange(60.0) % 13, "1": np.arange(60.0) % 7})
        with pytest.raises(SchemaError, match="share a text form"):
            plan_for(frame)
