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


class TestPlansAndFittedPreprocessorsPersist:
    """The config snapshot was a mappingproxy: immutable, and unpicklable with it.

    A fitted preprocessor that cannot be saved is a fitted preprocessor that has
    to be refitted in every process that needs it.
    """

    @pytest.fixture
    def fitted(self):
        frame = pd.DataFrame({"a": np.arange(60.0) % 13, "c": ["x", "y", "z"] * 20})
        plan, pre = build(frame)
        return frame, plan, pre.fit(frame)

    def test_a_plan_survives_a_round_trip(self, fitted):
        import pickle

        _, plan, _ = fitted
        assert pickle.loads(pickle.dumps(plan)).fingerprint == plan.fingerprint

    def test_a_fitted_preprocessor_survives_a_round_trip(self, fitted):
        import pickle

        frame, _, pre = fitted
        restored = pickle.loads(pickle.dumps(pre))
        assert np.allclose(dense(restored.transform(frame)), dense(pre.transform(frame)))

    def test_a_plan_can_be_deep_copied(self, fitted):
        import copy

        _, plan, _ = fitted
        assert copy.deepcopy(plan).to_dict() == plan.to_dict()

    def test_the_snapshot_is_still_read_only(self, fitted):
        _, plan, _ = fitted
        with pytest.raises(TypeError):
            plan.config["numeric_scaler"] = "tampered"


class TestOutputNamesAndLineageAgree:
    """Names were derived twice, by different routes, and could disagree."""

    @pytest.fixture
    def colliding(self) -> pd.DataFrame:
        # One-hot on "a" produces a_b_c; one-hot on "a_b" produces a_b_c as well.
        return pd.DataFrame({"a": ["b_c", "x"] * 30, "a_b": ["c", "y"] * 30})

    def test_names_are_unique_even_inside_one_group(self, colliding):
        _, pre = build(colliding)
        pre.fit(colliding)
        names = list(pre.get_feature_names_out())
        assert len(names) == len(set(names))

    def test_there_is_one_name_per_output_column(self, colliding):
        _, pre = build(colliding)
        out = dense(pre.fit_transform(colliding))
        assert pre.n_features_out == out.shape[1]

    def test_lineage_names_are_the_names_that_exist(self, colliding):
        _, pre = build(colliding)
        pre.fit(colliding)
        produced = {name for names in pre.lineage().values() for name in names}
        assert produced == set(pre.get_feature_names_out())

    def test_lineage_still_attributes_ordinary_columns(self):
        frame = pd.DataFrame({"a": np.arange(60.0) % 13, "c": ["x", "y", "z"] * 20})
        _, pre = build(frame)
        pre.fit(frame)
        assert pre.lineage() == {"a": ("a",), "c": ("c_x", "c_y", "c_z")}


class TestFittingFrameMustSupportThePlan:
    def test_a_planned_column_that_is_empty_here_is_refused(self):
        """The imputer would drop it and warn, leaving the output silently narrow."""
        from aidatasetkit.core.exceptions import PreprocessingError

        frame = pd.DataFrame({"a": np.arange(60.0) % 13, "b": np.arange(60.0) % 7})
        _, pre = build(frame)
        empty_here = frame.assign(a=np.nan)
        with pytest.raises(PreprocessingError, match="no observed value"):
            pre.fit(empty_here)

    def test_the_refusal_names_the_column_and_the_remedy(self):
        from aidatasetkit.core.exceptions import PreprocessingError

        frame = pd.DataFrame({"a": np.arange(60.0) % 13, "b": np.arange(60.0) % 7})
        _, pre = build(frame)
        with pytest.raises(PreprocessingError, match=r"\['a'\].*re-plan"):
            pre.fit(frame.assign(a=np.nan))

    @pytest.mark.parametrize("method", ["fit", "transform"])
    def test_a_frame_with_no_rows_is_refused_clearly(self, method):
        from aidatasetkit.core.exceptions import PreprocessingError

        frame = pd.DataFrame({"a": np.arange(60.0) % 13})
        _, pre = build(frame)
        pre.fit(frame)
        with pytest.raises(PreprocessingError, match="no rows"):
            getattr(pre, method)(frame.iloc[0:0])


class TestNaNColumnLabels:
    def test_a_nan_label_is_refused_where_it_can_be_explained(self):
        """Every later lookup compares against it and silently never matches."""
        frame = pd.DataFrame({np.nan: np.arange(60.0) % 13, "b": np.arange(60.0) % 7})
        with pytest.raises(SchemaError, match="NaN"):
            build(frame)

    def test_the_refusal_says_what_to_do(self):
        frame = pd.DataFrame({np.nan: np.arange(60.0) % 13, "b": np.arange(60.0) % 7})
        with pytest.raises(SchemaError, match="Name the column"):
            build(frame)


class TestKnownRefusalsSpeakForThemselves:
    """Each of these was a documented condition delivered as a raw library error.

    "could not convert string to float: 'oops'" and "unknown categories in
    column 0" name neither the column nor the remedy, and read like a defect in
    the preprocessor rather than a fact about the data.
    """

    def test_unparseable_text_in_a_numeric_column_names_the_column(self):
        from aidatasetkit.core.exceptions import PreprocessingError

        frame = pd.DataFrame({"a": np.arange(60.0) % 13, "b": np.arange(60.0) % 7})
        _, pre = build(frame)
        pre.fit(frame)
        with pytest.raises(PreprocessingError, match=r"'a'.*not numbers.*'oops'"):
            pre.transform(frame.assign(a=["oops"] * 60))

    def test_a_numeric_column_arriving_as_text_still_transforms(self):
        """Digits stored as text are lossless; only unparseable values are refused."""
        frame = pd.DataFrame({"a": np.arange(60.0) % 13, "b": np.arange(60.0) % 7})
        _, pre = build(frame)
        pre.fit(frame)
        as_text = frame.assign(a=frame["a"].astype(str))
        assert np.allclose(dense(pre.transform(as_text)), dense(pre.transform(frame)))

    @pytest.fixture
    def ordinal(self):
        frame = pd.DataFrame({"e": ["lo", "mid", "hi"] * 20})
        config = PreprocessingConfig(ordinal_orders={"e": ["lo", "mid", "hi"]})
        _, pre = build(frame, config)
        return frame, pre.fit(frame)

    def test_an_unranked_ordinal_value_names_the_column_and_the_order(self, ordinal):
        from aidatasetkit.core.exceptions import PreprocessingError

        frame, pre = ordinal
        with pytest.raises(PreprocessingError, match=r"'e'.*\['zz'\].*'lo', 'mid', 'hi'"):
            pre.transform(frame.assign(e=["lo", "mid", "zz"] * 20))

    def test_that_refusal_offers_both_ways_out(self, ordinal):
        from aidatasetkit.core.exceptions import PreprocessingError

        frame, pre = ordinal
        with pytest.raises(PreprocessingError, match="ordinal_orders.*ENCODE"):
            pre.transform(frame.assign(e=["zz"] * 60))

    def test_ranked_values_are_untouched_by_the_guard(self, ordinal):
        frame, pre = ordinal
        assert list(dense(pre.transform(frame)).ravel()[:3]) == [0.0, 1.0, 2.0]

    def test_the_encode_policy_still_codes_unknowns(self):
        from aidatasetkit.preprocessing import UnknownOrdinalPolicy

        frame = pd.DataFrame({"e": ["lo", "mid", "hi"] * 20})
        config = PreprocessingConfig(
            ordinal_orders={"e": ["lo", "mid", "hi"]},
            unknown_ordinal_policy=UnknownOrdinalPolicy.ENCODE,
        )
        _, pre = build(frame, config)
        pre.fit(frame)
        assert set(np.unique(dense(pre.transform(frame.assign(e=["zz"] * 60))))) == {-1.0}

    def test_missing_ordinal_values_are_still_imputed_not_refused(self):
        frame = pd.DataFrame({"e": ["lo", None, "hi"] * 20})
        config = PreprocessingConfig(ordinal_orders={"e": ["lo", "mid", "hi"]})
        _, pre = build(frame, config)
        assert not np.isnan(dense(pre.fit_transform(frame))).any()


class TestMixedTypeTargets:
    """scikit-learn's "uniformly strings or numbers" names nothing actionable."""

    @pytest.fixture
    def mixed(self) -> pd.Series:
        return pd.Series([1, "a", 2.5] * 20, dtype=object)

    def test_a_mixed_target_is_refused_with_its_own_message(self, mixed):
        from aidatasetkit.core.exceptions import PreprocessingError

        with pytest.raises(PreprocessingError, match="mixes numbers.*with text"):
            TargetLabelEncoder().fit(mixed)

    def test_the_refusal_shows_both_kinds_of_value(self, mixed):
        from aidatasetkit.core.exceptions import PreprocessingError

        with pytest.raises(PreprocessingError, match=r"1.*'a'"):
            TargetLabelEncoder().fit(mixed)

    def test_it_says_why_casting_is_not_the_answer(self, mixed):
        from aidatasetkit.core.exceptions import PreprocessingError

        with pytest.raises(PreprocessingError, match="not obviously the same class"):
            TargetLabelEncoder().fit(mixed)

    @pytest.mark.parametrize(
        "labels", [["a", "b"], [1, 2], [1.0, 2.0], [True, False]]
    )
    def test_a_uniform_target_still_encodes(self, labels):
        encoded = TargetLabelEncoder().fit_transform(pd.Series(labels * 30))
        assert set(np.unique(encoded)) == {0, 1}


class TestTargetEncoderEdges:
    """Three inputs that slipped past the encoder's own guards into raw errors."""

    @pytest.fixture
    def encoder(self):
        return TargetLabelEncoder().fit(pd.Series(["a", "b"] * 20))

    def test_a_missing_label_at_transform_time_is_named_as_such(self, encoder):
        """It was dropped before the unseen-label check, then reported as "unseen: nan"."""
        from aidatasetkit.core.exceptions import PreprocessingError

        with pytest.raises(PreprocessingError, match="missing value"):
            encoder.transform(pd.Series(["a", None, "b"]))

    def test_that_message_is_about_the_row_not_the_vocabulary(self, encoder):
        from aidatasetkit.core.exceptions import PreprocessingError

        with pytest.raises(PreprocessingError, match="without a label cannot be encoded"):
            encoder.transform(pd.Series(["a", None]))

    @pytest.mark.parametrize("code", [1, np.int64(1), np.array(1)])
    def test_a_single_code_inverts_to_a_single_label(self, encoder, code):
        assert encoder.inverse_transform(code).tolist() == ["b"]

    def test_an_unrepresentable_integer_is_refused_clearly(self, encoder):
        from aidatasetkit.core.exceptions import PreprocessingError

        with pytest.raises(PreprocessingError, match="within the range of a class index"):
            encoder.inverse_transform(2**1024)

    def test_a_matrix_of_codes_is_still_ambiguous_and_refused(self, encoder):
        from aidatasetkit.core.exceptions import PreprocessingError

        with pytest.raises(PreprocessingError, match="one-dimensional"):
            encoder.inverse_transform(np.array([[0], [1]]))

    def test_ordinary_round_trips_are_unaffected(self, encoder):
        labels = pd.Series(["b", "a", "b"])
        assert encoder.inverse_transform(encoder.transform(labels)).tolist() == ["b", "a", "b"]


class TestInputFeaturesStillChecked:
    """Names are now derived from what was fitted; the argument must still agree."""

    @pytest.fixture
    def fitted(self):
        frame = pd.DataFrame({"a": np.arange(60.0) % 13, "c": ["x", "y", "z"] * 20})
        _, pre = build(frame)
        return pre.fit(frame)

    def test_matching_names_are_accepted(self, fitted):
        assert list(fitted.get_feature_names_out(["a", "c"])) == list(
            fitted.get_feature_names_out()
        )

    def test_names_from_a_different_frame_are_refused(self, fitted):
        with pytest.raises(SchemaError, match="does not match the columns"):
            fitted.get_feature_names_out(["a", "elsewhere"])


class TestNumpyStringLabels:
    """numpy.str_ subclasses str, so the label check passed it and sklearn did not.

    Reached by anything that renames from an array: ``frame.rename(columns=dict(
    zip(frame.columns, cleaned)))`` where ``cleaned`` is a numpy array.
    """

    @pytest.fixture
    def numpy_labelled(self) -> pd.DataFrame:
        frame = pd.DataFrame({"a": np.arange(60.0) % 13, "c": np.arange(60.0) % 7})
        return frame.rename(columns=dict(zip(frame.columns, np.array(["p", "q"]))))

    def test_such_a_frame_fits(self, numpy_labelled):
        _, pre = build(numpy_labelled)
        assert dense(pre.fit_transform(numpy_labelled)).shape == (60, 2)

    def test_the_labels_are_normalised_and_the_reason_recorded(self, numpy_labelled):
        plan, _ = build(numpy_labelled)
        assert plan.label_normalisation.applied
        assert "str_" in plan.label_normalisation.reason

    def test_lineage_still_speaks_the_caller_s_labels(self, numpy_labelled):
        _, pre = build(numpy_labelled)
        pre.fit(numpy_labelled)
        assert set(pre.lineage()) == set(numpy_labelled.columns)

    def test_ordinary_string_labels_are_left_alone(self):
        frame = pd.DataFrame({"a": np.arange(60.0) % 13, "c": np.arange(60.0) % 7})
        plan, _ = build(frame)
        assert not plan.label_normalisation.applied


class TestOrdinalLevelsNeedNotBeStrings:
    """The caster hands the encoder text; the encoder was given the raw levels.

    Every value then matched nothing: silently the unknown sentinel under the
    ENCODE policy -- a whole column of -1 where the analyst wrote 1, 2, 3.
    """

    @pytest.mark.parametrize(
        "levels", [[1, 2, 3], [1.5, 2.5, 3.5], ["lo", "mid", "hi"]]
    )
    def test_levels_encode_to_their_ranks(self, levels):
        frame = pd.DataFrame({"e": levels * 20})
        config = PreprocessingConfig(ordinal_orders={"e": levels})
        _, pre = build(frame, config)
        assert set(np.unique(dense(pre.fit_transform(frame)))) == {0.0, 1.0, 2.0}

    def test_the_encode_policy_does_not_swallow_the_whole_column(self):
        from aidatasetkit.preprocessing import UnknownOrdinalPolicy

        frame = pd.DataFrame({"e": [1, 2, 3] * 20})
        config = PreprocessingConfig(
            ordinal_orders={"e": [1, 2, 3]},
            unknown_ordinal_policy=UnknownOrdinalPolicy.ENCODE,
        )
        _, pre = build(frame, config)
        encoded = dense(pre.fit_transform(frame))
        assert -1.0 not in set(np.unique(encoded))

    def test_a_genuinely_unranked_number_is_still_refused(self):
        from aidatasetkit.core.exceptions import PreprocessingError

        frame = pd.DataFrame({"e": [1, 2, 3] * 20})
        _, pre = build(frame, PreprocessingConfig(ordinal_orders={"e": [1, 2, 3]}))
        pre.fit(frame)
        with pytest.raises(PreprocessingError, match="does not rank"):
            pre.transform(frame.assign(e=[9] * 60))


class TestFingerprintCoversWhatChangesTheOutput:
    @pytest.fixture
    def mapped(self) -> pd.DataFrame:
        return pd.DataFrame({"g": ["M", "F"] * 30, "a": np.arange(60.0) % 13})

    def test_the_unmapped_value_is_part_of_the_identity(self, mapped):
        """Same steps, same mapping, different numbers in the matrix."""
        base = PreprocessingConfig(explicit_mappings={"g": {"M": 1, "F": 0}})
        plan_a, _ = build(mapped, base)
        plan_b, _ = build(mapped, base.replace(explicit_mapping_unknown_value=-1.0))
        assert plan_a.fingerprint != plan_b.fingerprint

    def test_a_policy_that_changes_a_column_changes_the_fingerprint(self):
        from aidatasetkit.preprocessing import HighCardinalityPolicy

        frame = pd.DataFrame(
            {"h": [f"v{i % 150}" for i in range(300)], "a": np.arange(300.0) % 13}
        )
        held, _ = build(frame)
        encoded, _ = build(
            frame, PreprocessingConfig(high_cardinality_policy=HighCardinalityPolicy.ONEHOT)
        )
        assert held.fingerprint != encoded.fingerprint


class TestTextCollisionSurvivesLookalikeValues:
    """The guard read distinctness from Series.unique(), which merges by ``==``.

    With ``True`` in the column the integer ``1`` is dropped as its duplicate, and
    the collision between that ``1`` and the string ``"1"`` becomes invisible --
    so 54 of 81 rows landed in a level that owns 27 of them.
    """

    @pytest.fixture
    def lookalikes(self) -> pd.DataFrame:
        column = pd.Series(([True, 1, "1"] * 27)[:81], dtype=object)
        return pd.DataFrame({"c": column, "a": np.arange(81.0) % 13})

    def test_the_collision_is_detected(self, lookalikes):
        plan, _ = build(lookalikes)
        assert plan.decision_for("c").reason_code == "categorical_type_collision"

    def test_the_column_is_held_for_review_not_encoded(self, lookalikes):
        plan, _ = build(lookalikes)
        assert plan.decision_for("c").action is FeatureAction.REVIEW

    @pytest.mark.parametrize(
        "values", [[1, "1"], [1, "1", 1.0], [True, 1, "1"], [1.0, 1, "1"]]
    )
    def test_every_ordering_is_caught(self, values):
        frame = pd.DataFrame(
            {"c": pd.Series((values * 40)[:80], dtype=object), "a": np.arange(80.0) % 13}
        )
        plan, _ = build(frame)
        assert plan.decision_for("c").action is FeatureAction.REVIEW

    @pytest.mark.parametrize(
        "values", [["a", "b"], [1, 2], [True, False], [1.5, 2.5], [1, 1.0], [True, "x"]]
    )
    def test_columns_with_no_collision_are_not_flagged_as_one(self, values):
        """Held back for another reason is fine; held back as a collision is not."""
        frame = pd.DataFrame(
            {"c": pd.Series((values * 40)[:80], dtype=object), "a": np.arange(80.0) % 13}
        )
        plan, _ = build(frame)
        assert plan.decision_for("c").reason_code != "categorical_type_collision"


class TestFillValuesMustBeRealNumbers:
    """An infinite fill value put an infinity in the matrix the guards were built to keep out."""

    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
    @pytest.mark.parametrize(
        "option", ["numeric_fill_value", "explicit_mapping_unknown_value"]
    )
    def test_a_non_finite_fill_value_is_refused(self, option, value):
        from aidatasetkit.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="must be a finite number"):
            PreprocessingConfig(**{option: value})

    def test_finite_values_are_still_accepted(self):
        config = PreprocessingConfig(
            numeric_fill_value=0.0, explicit_mapping_unknown_value=-1.0
        )
        assert config.numeric_fill_value == 0.0

    def test_no_unknown_value_is_still_the_default(self):
        assert PreprocessingConfig().explicit_mapping_unknown_value is None

    def test_a_nan_option_can_no_longer_make_a_config_mismatch_itself(self):
        """NaN != NaN, so the builder used to refuse the very config that planned."""
        from aidatasetkit.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            PreprocessingConfig(numeric_fill_value=float("nan"))


class TestSentinelIsReservedAtInferenceToo:
    """It was chosen against the training rows; a later frame can still hold it."""

    @pytest.fixture
    def training(self) -> pd.DataFrame:
        values = [["a", "b", "c"][i % 3] for i in range(60)]
        values[0] = None
        return pd.DataFrame({"c": pd.Series(values, dtype=object)})

    @pytest.fixture
    def config(self) -> PreprocessingConfig:
        return PreprocessingConfig(categorical_imputation=CategoricalImputation.CONSTANT)

    def test_a_real_sentinel_value_at_inference_is_refused(self, training, config):
        from aidatasetkit.core.exceptions import PreprocessingError

        _, pre = build(training, config)
        pre.fit(training)
        with pytest.raises(PreprocessingError, match="reserved to mark a missing category"):
            pre.transform(training.assign(c=["__missing__"] * 60))

    def test_the_refusal_names_the_column_and_a_way_out(self, training, config):
        from aidatasetkit.core.exceptions import PreprocessingError

        _, pre = build(training, config)
        pre.fit(training)
        with pytest.raises(PreprocessingError, match=r"\['c'\].*categorical_fill_value"):
            pre.transform(training.assign(c=["__missing__"] * 60))

    def test_ordinary_frames_still_transform(self, training, config):
        _, pre = build(training, config)
        pre.fit(training)
        assert dense(pre.transform(training)).shape[0] == 60

    def test_most_frequent_imputation_reserves_nothing(self, training):
        _, pre = build(training)
        pre.fit(training)
        assert dense(pre.transform(training.assign(c=["__missing__"] * 60))).shape[0] == 60
