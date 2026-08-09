"""Tests for :class:`DataProfiler`."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core import ColumnKind, KitConfig
from aidatasetkit.core.exceptions import SchemaError, ValidationError
from aidatasetkit.core.types import DatasetProfile
from aidatasetkit.profiling import DataProfiler


@pytest.fixture
def profiler() -> DataProfiler:
    return DataProfiler()


class TestDatasetLevelMeasurements:
    def test_shape_is_reported(self, profiler, problematic_frame):
        profile = profiler.profile(problematic_frame)
        assert profile.row_count == len(problematic_frame)
        assert profile.column_count == len(problematic_frame.columns)
        assert len(profile.column_profiles) == profile.column_count

    def test_duplicate_rows_are_counted(self, profiler, problematic_frame):
        profile = profiler.profile(problematic_frame)
        assert profile.duplicate_row_count == 4
        assert profile.duplicate_row_ratio == pytest.approx(
            4 / len(problematic_frame)
        )

    def test_a_frame_without_duplicates_reports_none(self, profiler, clean_frame):
        profile = profiler.profile(clean_frame)
        assert profile.duplicate_row_count == 0
        assert profile.duplicate_row_ratio == 0.0

    def test_missing_values_are_totalled_across_columns(self, profiler, problematic_frame):
        profile = profiler.profile(problematic_frame)
        assert profile.total_missing_count == int(problematic_frame.isna().sum().sum())
        cells = problematic_frame.size
        assert profile.total_missing_ratio == pytest.approx(
            profile.total_missing_count / cells
        )

    def test_memory_usage_is_reported(self, profiler, clean_frame):
        profile = profiler.profile(clean_frame)
        assert profile.memory_usage_bytes > 0
        assert profile.memory_usage_bytes == int(
            clean_frame.memory_usage(deep=True).sum()
        )

    def test_column_lookup_by_name(self, profiler, clean_frame):
        profile = profiler.profile(clean_frame)
        assert profile.column("age").name == "age"

    def test_unknown_column_lookup_raises(self, profiler, clean_frame):
        with pytest.raises(SchemaError, match="not present"):
            profiler.profile(clean_frame).column("nope")

    def test_columns_can_be_selected_by_kind(self, profiler, clean_frame):
        profile = profiler.profile(clean_frame)
        assert set(profile.columns_of_kind(ColumnKind.CATEGORICAL)) == {"segment"}


class TestColumnKinds:
    def test_kinds_come_from_the_shared_schema_detector(self, profiler, mixed_frame):
        from aidatasetkit.core import detect_column_kinds

        profile = profiler.profile(mixed_frame)
        expected = detect_column_kinds(mixed_frame)
        for column_profile in profile.column_profiles:
            assert column_profile.detected_kind is expected.of(column_profile.name)

    def test_every_dtype_family_is_classified(self, profiler, mixed_frame):
        profile = profiler.profile(mixed_frame)
        assert profile.column("int_col").detected_kind is ColumnKind.NUMERIC
        assert profile.column("nullable_int_col").detected_kind is ColumnKind.NUMERIC
        assert profile.column("bool_col").detected_kind is ColumnKind.BOOLEAN
        assert profile.column("string_col").detected_kind is ColumnKind.CATEGORICAL
        assert profile.column("category_col").detected_kind is ColumnKind.CATEGORICAL
        assert profile.column("datetime_col").detected_kind is ColumnKind.DATETIME

    def test_the_pandas_dtype_is_recorded_verbatim(self, profiler, mixed_frame):
        profile = profiler.profile(mixed_frame)
        assert profile.column("int_col").pandas_dtype == "int64"
        assert profile.column("nullable_int_col").pandas_dtype == "Int64"


class TestColumnLevelMeasurements:
    def test_counts_and_ratios_agree_with_pandas(self, profiler, problematic_frame):
        profile = profiler.profile(problematic_frame)
        income = profile.column("income")
        assert income.missing_count == int(problematic_frame["income"].isna().sum())
        assert income.count == int(problematic_frame["income"].notna().sum())
        assert income.missing_ratio == pytest.approx(
            income.missing_count / len(problematic_frame)
        )

    def test_unique_counts_exclude_missing(self, profiler):
        frame = pd.DataFrame({"a": [1, 1, 2, None]})
        column = profiler.profile(frame).column("a")
        assert column.unique_count == 2
        assert column.count == 3
        assert column.unique_ratio == pytest.approx(2 / 3)

    def test_constant_columns_are_flagged(self, profiler, problematic_frame):
        profile = profiler.profile(problematic_frame)
        assert profile.column("country").is_constant
        assert not profile.column("country").is_near_constant
        assert profile.column("country").dominant_value == "SA"

    def test_near_constant_columns_are_flagged_against_the_configured_threshold(
        self, profiler, problematic_frame
    ):
        column = profiler.profile(problematic_frame).column("plan")
        assert column.is_near_constant
        assert not column.is_constant
        assert column.dominant_ratio >= KitConfig().near_constant_threshold

    def test_the_near_constant_threshold_is_configurable(self, problematic_frame):
        relaxed = DataProfiler(KitConfig(near_constant_threshold=0.999))
        assert not relaxed.profile(problematic_frame).column("plan").is_near_constant

    def test_high_cardinality_is_flagged_for_label_columns_only(
        self, profiler, problematic_frame
    ):
        profile = profiler.profile(problematic_frame)
        assert profile.column("city").is_high_cardinality
        assert not profile.column("temperature").is_high_cardinality

    def test_infinities_are_counted_separately_from_missing(self, profiler, problematic_frame):
        column = profiler.profile(problematic_frame).column("ratio")
        assert column.infinite_count == 2
        assert column.missing_count == 0

    def test_memory_usage_is_reported_per_column(self, profiler, clean_frame):
        profile = profiler.profile(clean_frame)
        assert all(p.memory_usage_bytes > 0 for p in profile.column_profiles)


class TestNumericSummary:
    def test_statistics_delegate_to_the_statistics_engine(self, profiler, clean_frame):
        from aidatasetkit.statistics import StatisticsEngine

        column = profiler.profile(clean_frame).column("age")
        engine = StatisticsEngine(clean_frame["age"])
        assert column.numeric.mean == pytest.approx(engine.mean())
        assert column.numeric.std == pytest.approx(engine.std())
        assert column.numeric.median == pytest.approx(engine.median())
        assert column.numeric.q25 == pytest.approx(engine.quartiles().q1)
        assert column.numeric.q75 == pytest.approx(engine.quartiles().q3)
        assert column.numeric.iqr == pytest.approx(engine.iqr())
        assert column.numeric.minimum == pytest.approx(engine.min())
        assert column.numeric.maximum == pytest.approx(engine.max())

    def test_non_numeric_columns_have_no_summary(self, profiler, clean_frame):
        assert profiler.profile(clean_frame).column("segment").numeric is None

    def test_infinities_are_excluded_from_the_summary(self, profiler, problematic_frame):
        summary = profiler.profile(problematic_frame).column("ratio").numeric
        assert np.isfinite(summary.minimum)
        assert np.isfinite(summary.maximum)

    def test_undefined_statistics_are_none_rather_than_invented(self, profiler):
        column = profiler.profile(pd.DataFrame({"a": [5.0]})).column("a")
        assert column.numeric.mean == pytest.approx(5.0)
        assert column.numeric.std is None

    def test_a_fully_missing_numeric_column_is_handled(self, profiler):
        column = profiler.profile(pd.DataFrame({"a": [np.nan, np.nan]})).column("a")
        assert column.count == 0
        assert column.numeric is None

    def test_a_column_of_only_infinities_reports_an_empty_summary(self, profiler):
        column = profiler.profile(pd.DataFrame({"a": [np.inf, -np.inf]})).column("a")
        assert column.infinite_count == 2
        assert column.numeric.mean is None


class TestIdentifierHeuristic:
    def test_an_identifier_named_column_is_flagged(self, profiler, problematic_frame):
        assert profiler.profile(problematic_frame).column("customer_id").is_id_like

    def test_a_unique_continuous_measurement_is_not_an_identifier(
        self, profiler, problematic_frame
    ):
        """Uniqueness alone must never be enough."""
        column = profiler.profile(problematic_frame).column("temperature")
        assert column.unique_ratio > 0.95
        assert not column.is_id_like

    def test_a_legitimate_high_cardinality_label_is_not_an_identifier(
        self, profiler, problematic_frame
    ):
        column = profiler.profile(problematic_frame).column("city")
        assert column.is_high_cardinality
        assert not column.is_id_like

    def test_uuid_values_are_recognised_without_a_name_signal(self, profiler):
        frame = pd.DataFrame(
            {
                "reference_token": [
                    f"{value:08x}-0000-4000-8000-{value:012x}" for value in range(30)
                ]
            }
        )
        assert profiler.profile(frame).column("reference_token").is_id_like

    def test_a_sequential_integer_counter_is_recognised(self, profiler):
        frame = pd.DataFrame({"row": list(range(50))})
        assert profiler.profile(frame).column("row").is_id_like

    def test_a_unique_but_scattered_integer_column_is_not_a_counter(self, profiler):
        frame = pd.DataFrame({"reading": [value * 977 for value in range(50)]})
        assert not profiler.profile(frame).column("reading").is_id_like

    def test_a_column_with_missing_values_is_not_an_identifier(self, profiler):
        frame = pd.DataFrame({"customer_id": [f"CUST-{v:05d}" for v in range(29)] + [None]})
        assert not profiler.profile(frame).column("customer_id").is_id_like

    def test_a_low_uniqueness_column_is_never_an_identifier(self, profiler):
        frame = pd.DataFrame({"account_id": ["A", "B"] * 25})
        assert not profiler.profile(frame).column("account_id").is_id_like

    def test_datetime_columns_are_never_identifiers(self, profiler, problematic_frame):
        assert not profiler.profile(problematic_frame).column("signup_date").is_id_like

    def test_the_uniqueness_threshold_is_configurable(self):
        frame = pd.DataFrame({"order_id": [f"ORD-{v:05d}" for v in range(10)] + ["ORD-00000"] * 3})
        assert not DataProfiler(KitConfig(id_uniqueness_threshold=0.99)).profile(
            frame
        ).column("order_id").is_id_like


class TestImmutability:
    def test_profiling_does_not_modify_the_frame(self, profiler, problematic_frame):
        before = problematic_frame.copy(deep=True)
        profiler.profile(problematic_frame)
        pd.testing.assert_frame_equal(problematic_frame, before)

    def test_dtypes_are_not_converted(self, profiler, mixed_frame):
        before = mixed_frame.dtypes.copy()
        profiler.profile(mixed_frame)
        pd.testing.assert_series_equal(mixed_frame.dtypes, before)

    def test_column_order_is_preserved(self, profiler, problematic_frame):
        profile = profiler.profile(problematic_frame)
        assert [p.name for p in profile.column_profiles] == list(problematic_frame.columns)


class TestSerialisation:
    def test_the_profile_round_trips_through_json(self, profiler, problematic_frame):
        payload = json.loads(json.dumps(profiler.profile(problematic_frame).to_dict()))
        assert payload["row_count"] == len(problematic_frame)
        assert len(payload["column_profiles"]) == len(problematic_frame.columns)

    def test_non_string_column_labels_serialise_safely(self, profiler):
        frame = pd.DataFrame({0: [1, 2], 1: ["a", "b"]})
        profile = profiler.profile(frame)
        assert profile.column_profiles[0].name == 0
        payload = json.loads(json.dumps(profile.to_dict()))
        assert payload["column_profiles"][0]["name"] == "0"

    def test_numpy_dominant_values_serialise(self, profiler):
        frame = pd.DataFrame({"a": np.array([1, 1, 2], dtype="int64")})
        payload = json.loads(json.dumps(profiler.profile(frame).to_dict()))
        assert payload["column_profiles"][0]["dominant_value"] == 1

    def test_the_profile_renders_as_a_frame(self, profiler, clean_frame):
        rendered = profiler.profile(clean_frame).to_frame()
        assert len(rendered) == len(clean_frame.columns)
        assert "detected_kind" in rendered.columns

    def test_the_profile_is_immutable(self, profiler, clean_frame):
        with pytest.raises(AttributeError):
            profiler.profile(clean_frame).row_count = 0


class TestEdgeCases:
    def test_an_empty_frame_is_profiled_rather_than_rejected(self, profiler):
        profile = profiler.profile(pd.DataFrame())
        assert isinstance(profile, DatasetProfile)
        assert profile.row_count == 0
        assert profile.column_count == 0

    def test_a_frame_with_columns_but_no_rows_is_profiled(self, profiler):
        profile = profiler.profile(pd.DataFrame({"a": pd.Series(dtype="float64")}))
        assert profile.row_count == 0
        assert profile.column("a").count == 0
        assert not profile.column("a").is_constant

    def test_a_single_row_frame_is_profiled(self, profiler):
        profile = profiler.profile(pd.DataFrame({"a": [1.0]}))
        assert profile.column("a").is_constant

    @pytest.mark.parametrize("value", [None, [1, 2], pd.Series([1, 2])])
    def test_non_dataframe_input_is_rejected(self, profiler, value):
        with pytest.raises(ValidationError, match="DataFrame is required"):
            profiler.profile(value)

    def test_duplicate_labels_are_rejected(self, profiler):
        frame = pd.DataFrame(np.arange(4).reshape(2, 2), columns=["a", "a"])
        with pytest.raises(SchemaError, match="Duplicate column labels"):
            profiler.profile(frame)

    def test_unhashable_cells_get_an_explanation_not_a_pandas_traceback(self, profiler):
        frame = pd.DataFrame({"a": [[1], [2]], "b": [1, 2]})
        with pytest.raises(SchemaError, match="unhashable values"):
            profiler.profile(frame)
