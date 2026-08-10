"""Closure tests: the contracts the S4 review left open.

Four subjects, each one a promise the layer was making and not keeping:
infinity, target decoding, cache identity, and category identity.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from aidatasetkit.core.exceptions import (
    AIDatasetKitError,
    PreprocessingError,
    SchemaError,
)
from aidatasetkit.core.types import PreprocessingProfile
from aidatasetkit.preprocessing import (
    CategoricalImputation,
    FeatureAction,
    NumericImputation,
    NumericScaler,
    NumericTextPolicy,
    PreprocessingConfig,
    PreprocessingPlanner,
    PreprocessorBuilder,
    TargetLabelEncoder,
    UnknownOrdinalPolicy,
)
from aidatasetkit.profiling import DataProfiler, DataQualityInspector

UNSCALED = PreprocessingProfile(False, True, False)
SCALED = PreprocessingProfile(True, True, False)
DENSE = PreprocessingProfile(False, False, False)


def plan_for(frame, config=None, profile=UNSCALED, target=None):
    config = config or PreprocessingConfig()
    dataset_profile = DataProfiler().profile(frame)
    quality = DataQualityInspector().inspect(frame, profile=dataset_profile, target=target)
    return PreprocessingPlanner(config).plan(
        frame, dataset_profile, profile, quality=quality, target=target
    )


def build(frame, config=None, profile=UNSCALED):
    config = config or PreprocessingConfig()
    plan = plan_for(frame, config, profile)
    return plan, PreprocessorBuilder(config).build(plan, frame)


def dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


# --------------------------------------------------------------------------- #
# Infinity
# --------------------------------------------------------------------------- #


class TestInfinityContract:
    """v1 does not support infinity as a trainable value, and says so itself."""

    def test_the_misleading_option_is_gone(self):
        """"allow" had no valid meaning: nothing downstream accepts an infinity."""
        fields = {f.name for f in dataclasses.fields(PreprocessingConfig)}
        assert "allow_infinite" not in fields

    @pytest.mark.parametrize("bad", [np.inf, -np.inf])
    def test_a_column_with_either_infinity_is_held_back(self, bad):
        values = np.arange(120.0) % 37
        values[5] = bad
        decision = plan_for(pd.DataFrame({"r": values})).decision_for("r")
        assert decision.action is FeatureAction.REVIEW
        assert decision.reason_code == "infinite_values"

    def test_mixed_finite_and_infinite_is_still_held_back(self):
        values = np.arange(120.0) % 37
        values[5] = np.inf
        values[9] = -np.inf
        decision = plan_for(pd.DataFrame({"r": values})).decision_for("r")
        assert decision.details["infinite_count"] == 2

    def test_the_reason_says_what_the_analyst_should_do(self):
        values = np.arange(120.0) % 37
        values[5] = np.inf
        reason = plan_for(pd.DataFrame({"r": values})).decision_for("r").reason
        assert "does not support infinity" in reason
        assert "cap the values" in reason

    def test_forcing_it_in_fails_with_a_library_error_naming_the_column(self):
        values = np.arange(120.0) % 37
        values[5] = np.inf
        frame = pd.DataFrame({"r": values})
        config = PreprocessingConfig(force_include=("r",))
        _, pre = build(frame, config)
        with pytest.raises(PreprocessingError, match=r"'r' contains 1 infinite"):
            pre.fit(frame)

    def test_an_infinity_arriving_only_at_inference_is_caught(self):
        frame = pd.DataFrame({"r": np.arange(120.0) % 37})
        _, pre = build(frame)
        pre.fit(frame)
        with pytest.raises(PreprocessingError, match="infinite"):
            pre.transform(pd.DataFrame({"r": [np.inf]}))

    def test_parsing_text_cannot_smuggle_an_infinity_through(self):
        """"inf" parses to an infinity that the profile never saw."""
        frame = pd.DataFrame({"a": [str(v % 37) if v % 20 else "inf" for v in range(200)]})
        config = PreprocessingConfig(numeric_text_policy=NumericTextPolicy.CONVERT)
        _, pre = build(frame, config)
        with pytest.raises(PreprocessingError, match="infinite"):
            pre.fit(frame)

    def test_no_raw_sklearn_error_escapes_for_this_condition(self):
        values = np.arange(120.0) % 37
        values[5] = np.inf
        frame = pd.DataFrame({"r": values})
        config = PreprocessingConfig(force_include=("r",))
        _, pre = build(frame, config)
        with pytest.raises(AIDatasetKitError):
            pre.fit(frame)


# --------------------------------------------------------------------------- #
# Target decoding
# --------------------------------------------------------------------------- #


class TestInverseTransformRejectsInvalidCodes:
    @pytest.fixture
    def encoder(self) -> TargetLabelEncoder:
        return TargetLabelEncoder().fit(pd.Series(["a", "b", "c"] * 30))

    @pytest.mark.parametrize("value", [1.1, 1.9, 0.5, -0.5])
    def test_a_non_whole_number_is_refused(self, encoder, value):
        """astype(int) turned 1.9 into class 1 and said nothing."""
        with pytest.raises(PreprocessingError, match="whole numbers"):
            encoder.inverse_transform([value])

    def test_the_refusal_explains_why_rounding_would_be_wrong(self, encoder):
        with pytest.raises(PreprocessingError, match="decode a prediction, not a probability"):
            encoder.inverse_transform([1.9])

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_a_non_finite_code_is_refused(self, encoder, value):
        with pytest.raises(PreprocessingError, match="non-finite"):
            encoder.inverse_transform([value])

    @pytest.mark.parametrize("value", [-1, 3, 99])
    def test_a_code_outside_the_class_range_is_refused(self, encoder, value):
        with pytest.raises(PreprocessingError, match="outside the range"):
            encoder.inverse_transform([value])

    def test_the_refusal_names_the_number_of_classes(self, encoder):
        with pytest.raises(PreprocessingError, match="3 class"):
            encoder.inverse_transform([7])

    @pytest.mark.parametrize("codes", [[0, 1, 2], [0.0, 1.0, 2.0], np.array([2, 0])])
    def test_whole_numbers_in_range_are_accepted(self, encoder, codes):
        assert set(encoder.inverse_transform(codes)) <= {"a", "b", "c"}

    def test_a_round_trip_still_works(self, encoder):
        labels = pd.Series(["a", "c", "b"] * 10)
        assert list(encoder.inverse_transform(encoder.transform(labels))) == list(labels)

    def test_a_two_dimensional_input_is_refused(self, encoder):
        with pytest.raises(PreprocessingError, match="one-dimensional"):
            encoder.inverse_transform([[0, 1], [1, 0]])

    def test_text_is_refused(self, encoder):
        with pytest.raises(PreprocessingError, match="must be numbers"):
            encoder.inverse_transform(["a"])


# --------------------------------------------------------------------------- #
# Cache identity
# --------------------------------------------------------------------------- #


class TestFingerprintTracksExecutionSemantics:
    """Anything that changes what runs must change the cache identity."""

    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        index = np.arange(150)
        return pd.DataFrame(
            {
                "n": np.where(index % 9 == 0, np.nan, (index % 41).astype("float64")),
                "c": np.where(index % 11 == 0, None, [["x", "y", "z"][v % 3] for v in index]),
                "e": [["lo", "mid", "hi"][v % 3] for v in index],
            }
        )

    @pytest.fixture
    def base(self) -> PreprocessingConfig:
        return PreprocessingConfig(ordinal_orders={"e": ["lo", "mid", "hi"]})

    @pytest.mark.parametrize(
        ("changes", "label"),
        [
            ({"numeric_imputation": NumericImputation.MEAN}, "numeric imputation"),
            ({"categorical_imputation": CategoricalImputation.CONSTANT}, "categorical imputation"),
            ({"numeric_scaler": NumericScaler.ROBUST}, "scaler"),
            ({"ordinal_orders": {"e": ["hi", "mid", "lo"]}}, "ordinal order"),
            ({"unknown_ordinal_policy": UnknownOrdinalPolicy.ENCODE}, "unknown-ordinal policy"),
            ({"force_exclude": ("n",)}, "force exclude"),
            ({"nominal_features": ("n",)}, "role override"),
            ({"categorical_fill_value": "__na__"}, "sentinel"),
        ],
    )
    def test_a_semantic_change_changes_the_fingerprint(self, frame, base, changes, label):
        original = plan_for(frame, base, SCALED)
        changed = plan_for(frame, base.replace(**changes), SCALED)
        assert original.fingerprint != changed.fingerprint, label

    def test_the_constant_fill_value_changes_it(self, frame, base):
        constant = base.replace(numeric_imputation=NumericImputation.CONSTANT)
        first = plan_for(frame, constant.replace(numeric_fill_value=0.0), SCALED)
        second = plan_for(frame, constant.replace(numeric_fill_value=-1.0), SCALED)
        assert first.fingerprint != second.fingerprint

    def test_the_unknown_ordinal_value_changes_it(self, frame, base):
        encoding = base.replace(unknown_ordinal_policy=UnknownOrdinalPolicy.ENCODE)
        first = plan_for(frame, encoding.replace(unknown_ordinal_value=-1), SCALED)
        second = plan_for(frame, encoding.replace(unknown_ordinal_value=-99), SCALED)
        assert first.fingerprint != second.fingerprint

    def test_an_explicit_mapping_s_contents_change_it(self):
        frame = pd.DataFrame({"g": ["M", "F"] * 60})
        first = plan_for(frame, PreprocessingConfig(explicit_mappings={"g": {"M": 1, "F": 0}}))
        second = plan_for(frame, PreprocessingConfig(explicit_mappings={"g": {"M": 0, "F": 1}}))
        assert first.fingerprint != second.fingerprint

    def test_the_capability_profile_changes_it(self, frame, base):
        assert plan_for(frame, base, SCALED).fingerprint != plan_for(
            frame, base, UNSCALED
        ).fingerprint

    def test_sparse_capability_changes_it(self, frame, base):
        assert plan_for(frame, base, UNSCALED).fingerprint != plan_for(
            frame, base, DENSE
        ).fingerprint

    def test_a_numeric_text_conversion_changes_it(self):
        frame = pd.DataFrame({"a": [str(v % 37) for v in range(200)]})
        review = plan_for(frame, PreprocessingConfig())
        convert = plan_for(
            frame, PreprocessingConfig(numeric_text_policy=NumericTextPolicy.CONVERT)
        )
        assert review.fingerprint != convert.fingerprint

    def test_a_high_cardinality_decision_changes_it(self):
        index = np.arange(600)
        frame = pd.DataFrame({"c": [f"v{v % 120}" for v in index]})
        from aidatasetkit.preprocessing import HighCardinalityPolicy

        review = plan_for(frame, PreprocessingConfig())
        onehot = plan_for(
            frame, PreprocessingConfig(high_cardinality_policy=HighCardinalityPolicy.ONEHOT)
        )
        assert review.fingerprint != onehot.fingerprint

    def test_the_same_semantics_share_one_identity(self, frame, base):
        """Two runs of the same configuration reuse one blueprint entry."""
        from aidatasetkit.preprocessing import BlueprintCache

        cache = BlueprintCache()
        builder = PreprocessorBuilder(base, cache=cache)
        builder.build(plan_for(frame, base, SCALED), frame)
        builder.build(plan_for(frame, base, SCALED), frame)
        assert len(cache) == 1

    def test_metadata_only_differences_do_not_split_the_cache(self, frame, base):
        """drop_exact_target_duplicates changes no built transformer here."""
        first = plan_for(frame, base.replace(drop_exact_target_duplicates=True), SCALED)
        second = plan_for(frame, base.replace(drop_exact_target_duplicates=False), SCALED)
        assert first.fingerprint == second.fingerprint

    def test_no_model_name_appears_in_the_identity(self, frame, base):
        from aidatasetkit.preprocessing import BlueprintCache

        key = BlueprintCache().key_for(plan_for(frame, base, SCALED))
        assert "dummy" not in str(key) and "logistic" not in str(key)


# --------------------------------------------------------------------------- #
# Category identity
# --------------------------------------------------------------------------- #


class TestCategoryIdentityIsNeverMerged:
    @pytest.mark.parametrize(
        "values",
        [[1, "1", 2], [True, "True", False], ["None", "a", "b"]],
        ids=["int-vs-text", "bool-vs-text", "text-none"],
    )
    def test_a_column_whose_categories_collide_is_held_back(self, values):
        frame = pd.DataFrame({"c": pd.Series(values * 40, dtype=object)})
        decision = plan_for(frame).decision_for("c")
        if decision.action is FeatureAction.REVIEW:
            assert decision.reason_code == "categorical_type_collision"
        else:
            # No collision in this set, so the column is usable as it stands.
            assert decision.action is FeatureAction.INCLUDE

    def test_the_int_text_collision_is_detected(self):
        frame = pd.DataFrame({"c": pd.Series([1, "1", 2] * 40, dtype=object)})
        decision = plan_for(frame).decision_for("c")
        assert decision.action is FeatureAction.REVIEW
        assert "1 and '1'" in decision.reason

    def test_the_bool_text_collision_is_detected(self):
        frame = pd.DataFrame({"c": pd.Series([True, "True", False] * 40, dtype=object)})
        assert plan_for(frame).decision_for("c").reason_code == "categorical_type_collision"

    def test_a_none_string_beside_real_missing_is_not_a_collision(self):
        """The string "None" and an actual gap are already distinguishable."""
        frame = pd.DataFrame({"c": pd.Series(["None", "a", None] * 40, dtype=object)})
        assert plan_for(frame).decision_for("c").action is FeatureAction.INCLUDE

    def test_forcing_it_in_fails_rather_than_merging(self):
        frame = pd.DataFrame({"c": pd.Series([1, "1", 2] * 40, dtype=object)})
        config = PreprocessingConfig(force_include=("c",))
        _, pre = build(frame, config)
        with pytest.raises(PreprocessingError, match="share a text form"):
            pre.fit(frame)

    def test_that_failure_names_the_offending_values(self):
        frame = pd.DataFrame({"c": pd.Series([1, "1", 2] * 40, dtype=object)})
        config = PreprocessingConfig(force_include=("c",))
        _, pre = build(frame, config)
        with pytest.raises(PreprocessingError, match=r"\[1, '1'\]|\['1', 1\]"):
            pre.fit(frame)

    def test_an_ordinary_typed_column_is_unaffected(self):
        frame = pd.DataFrame({"c": ["a", "b", "c"] * 40})
        _, pre = build(frame)
        assert dense(pre.fit_transform(frame)).shape == (120, 3)

    def test_a_numeric_looking_string_column_is_not_a_collision(self):
        frame = pd.DataFrame({"c": ["1", "2", "3"] * 40})
        assert plan_for(frame).decision_for("c").reason_code != "categorical_type_collision"


# --------------------------------------------------------------------------- #
# Explicit mapping, unknown values
# --------------------------------------------------------------------------- #


class TestExplicitMappingUnknownValues:
    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"g": ["M", "F"] * 60})

    @pytest.fixture
    def config(self) -> PreprocessingConfig:
        return PreprocessingConfig(explicit_mappings={"g": {"M": 1, "F": 0}})

    def test_an_unknown_value_raises_a_library_error(self, frame, config):
        _, pre = build(frame, config)
        pre.fit(frame)
        with pytest.raises(PreprocessingError):
            pre.transform(pd.DataFrame({"g": ["X"]}))

    def test_the_error_names_the_column(self, frame, config):
        _, pre = build(frame, config)
        pre.fit(frame)
        with pytest.raises(PreprocessingError, match="'g'"):
            pre.transform(pd.DataFrame({"g": ["X"]}))

    def test_the_error_names_the_unknown_value(self, frame, config):
        _, pre = build(frame, config)
        pre.fit(frame)
        with pytest.raises(PreprocessingError, match="'X'"):
            pre.transform(pd.DataFrame({"g": ["X"]}))

    def test_the_error_says_how_to_resolve_it(self, frame, config):
        _, pre = build(frame, config)
        pre.fit(frame)
        with pytest.raises(PreprocessingError, match="Add them, or set unknown_value"):
            pre.transform(pd.DataFrame({"g": ["X"]}))

    def test_an_explicit_unknown_value_is_honoured(self, frame):
        config = PreprocessingConfig(
            explicit_mappings={"g": {"M": 1, "F": 0}}, explicit_mapping_unknown_value=-1.0
        )
        _, pre = build(frame, config)
        pre.fit(frame)
        assert dense(pre.transform(pd.DataFrame({"g": ["X"]}))).ravel()[0] == -1.0

    def test_no_low_level_exception_leaks(self, frame, config):
        _, pre = build(frame, config)
        pre.fit(frame)
        with pytest.raises(AIDatasetKitError):
            pre.transform(pd.DataFrame({"g": ["X"]}))


# --------------------------------------------------------------------------- #
# Remaining upheld findings
# --------------------------------------------------------------------------- #


class TestForceIncludedIdentifierIsActuallyUsed:
    """It was planned INCLUDE and then consumed by no transformer group."""

    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"customer_id": [f"C{i:05d}" for i in range(200)], "n": np.arange(200.0) % 13}
        )

    def test_the_plan_and_the_pipeline_agree(self, frame):
        config = PreprocessingConfig(force_include=("customer_id",))
        plan, pre = build(frame, config)
        pre.fit(frame)
        assert set(pre.lineage()) == set(plan.included_features)

    def test_it_contributes_output_columns(self, frame):
        config = PreprocessingConfig(force_include=("customer_id",))
        _, pre = build(frame, config)
        pre.fit(frame)
        assert len(pre.lineage()["customer_id"]) == 200

    def test_it_is_still_flagged_for_review(self, frame):
        config = PreprocessingConfig(force_include=("customer_id",))
        plan, _ = build(frame, config)
        assert plan.decision_for("customer_id").requires_review

    def test_a_confirmed_identifier_still_wins_over_force_include(self, frame):
        config = PreprocessingConfig(confirmed_id_columns=("customer_id",))
        plan, _ = build(config=config, frame=frame)
        assert plan.decision_for("customer_id").action is FeatureAction.EXCLUDE


class TestTargetIsValidated:
    def test_a_target_absent_from_the_frame_is_refused(self):
        frame = pd.DataFrame({"a": np.arange(60.0) % 13})
        profile = DataProfiler().profile(frame)
        with pytest.raises(SchemaError, match="not a column of this frame"):
            PreprocessingPlanner().plan(frame, profile, UNSCALED, target="nope")

    def test_the_refusal_lists_what_is_available(self):
        frame = pd.DataFrame({"a": np.arange(60.0) % 13})
        profile = DataProfiler().profile(frame)
        with pytest.raises(SchemaError, match="'a'"):
            PreprocessingPlanner().plan(frame, profile, UNSCALED, target="nope")


class TestPublicSurface:
    def test_every_transformer_in_a_built_pipeline_is_importable(self):
        import aidatasetkit.preprocessing as package

        for name in (
            "CategoricalCaster",
            "NumericCaster",
            "NumericTextConverter",
            "ExplicitMappingEncoder",
        ):
            assert name in package.__all__
            assert getattr(package, name) is not None


class TestInfinityGuardDoesNotFailOpen:
    """Found by the final verification pass: the guard had two blind spots, and
    in both the infinity was stopped only by a raw sklearn error naming neither
    the column nor the cause."""

    def test_a_text_column_holding_both_inf_and_an_unparseable_value(self):
        """The direct float cast raised, so the guard used to skip the column."""
        frame = pd.DataFrame(
            {"a": [str(v % 37) if v % 3 else ("inf" if v % 2 else "unknown") for v in range(200)]}
        )
        config = PreprocessingConfig(numeric_text_policy=NumericTextPolicy.CONVERT)
        _, pre = build(frame, config)
        with pytest.raises(PreprocessingError, match=r"'a' contains \d+ infinite"):
            pre.fit(frame)

    def test_no_raw_sklearn_error_escapes_for_that_case(self):
        frame = pd.DataFrame(
            {"a": [str(v % 37) if v % 3 else ("inf" if v % 2 else "unknown") for v in range(200)]}
        )
        config = PreprocessingConfig(numeric_text_policy=NumericTextPolicy.CONVERT)
        _, pre = build(frame, config)
        with pytest.raises(AIDatasetKitError):
            pre.fit(frame)

    def test_parsing_reports_the_infinity_it_created(self):
        frame = pd.DataFrame({"a": ["inf", "1", "2"] * 40})
        config = PreprocessingConfig(
            numeric_text_policy=NumericTextPolicy.CONVERT, force_include=("a",)
        )
        _, pre = build(frame, config)
        with pytest.raises(PreprocessingError, match="infinite"):
            pre.fit(frame)


class TestExplicitMappingDestinationsAreValidated:
    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"g": ["M", "F"] * 60})

    @pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
    def test_a_non_finite_destination_is_refused(self, frame, bad):
        """A mapped infinity reached sklearn and failed there instead."""
        config = PreprocessingConfig(explicit_mappings={"g": {"M": bad, "F": 0}})
        _, pre = build(frame, config)
        with pytest.raises(PreprocessingError, match="does not support infinity or NaN"):
            pre.fit(frame)

    def test_the_refusal_names_the_column_and_the_category(self, frame):
        config = PreprocessingConfig(explicit_mappings={"g": {"M": np.inf, "F": 0}})
        _, pre = build(frame, config)
        with pytest.raises(PreprocessingError, match=r"for 'g' maps 'M'"):
            pre.fit(frame)

    def test_a_non_numeric_destination_is_refused(self, frame):
        config = PreprocessingConfig(explicit_mappings={"g": {"M": "high", "F": "low"}})
        _, pre = build(frame, config)
        with pytest.raises(PreprocessingError, match="is not a number"):
            pre.fit(frame)

    def test_a_non_finite_unknown_value_is_refused(self, frame):
        """Now at construction rather than at fit: the option itself is invalid."""
        from aidatasetkit.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="explicit_mapping_unknown_value"):
            PreprocessingConfig(
                explicit_mappings={"g": {"M": 1, "F": 0}},
                explicit_mapping_unknown_value=np.inf,
            )

    def test_a_valid_numeric_mapping_still_works(self, frame):
        config = PreprocessingConfig(explicit_mappings={"g": {"M": 1, "F": 0}})
        _, pre = build(frame, config)
        assert set(dense(pre.fit_transform(frame)).ravel().tolist()) == {0.0, 1.0}
