"""Tests for column-kind detection.

These lock in the ordering rules that dtype predicates make easy to get wrong:
pandas reports booleans as numeric and categoricals as string-like.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core import ColumnKind, detect_column_kinds, detect_kind
from aidatasetkit.core.exceptions import SchemaError, ValidationError


class TestKindDetection:
    def test_every_kind_is_detected(self, mixed_frame):
        kinds = detect_column_kinds(mixed_frame)
        assert kinds.numeric == ("int_col", "float_col", "nullable_int_col")
        assert kinds.boolean == ("bool_col",)
        assert kinds.categorical == ("string_col", "object_col", "category_col")
        assert kinds.datetime == ("datetime_col",)
        assert kinds.other == ("timedelta_col",)

    def test_booleans_are_not_reported_as_numeric(self, mixed_frame):
        assert detect_column_kinds(mixed_frame).of("bool_col") is ColumnKind.BOOLEAN

    def test_categoricals_are_not_reported_as_numeric_even_when_backed_by_numbers(self):
        frame = pd.DataFrame({"c": pd.Categorical([1, 2, 1])})
        assert detect_column_kinds(frame).of("c") is ColumnKind.CATEGORICAL

    def test_pandas_three_string_dtype_is_categorical(self):
        frame = pd.DataFrame({"s": pd.Series(["a", "b"], dtype="string")})
        assert detect_column_kinds(frame).of("s") is ColumnKind.CATEGORICAL

    def test_legacy_object_strings_are_categorical(self):
        frame = pd.DataFrame({"s": pd.Series(["a", "b"], dtype="object")})
        assert detect_column_kinds(frame).of("s") is ColumnKind.CATEGORICAL

    def test_timezone_aware_datetimes_are_datetime(self):
        frame = pd.DataFrame(
            {"d": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True)}
        )
        assert detect_column_kinds(frame).of("d") is ColumnKind.DATETIME

    def test_nullable_numeric_dtypes_are_numeric(self):
        frame = pd.DataFrame({"n": pd.Series([1, None], dtype="Int64")})
        assert detect_column_kinds(frame).of("n") is ColumnKind.NUMERIC

    def test_nullable_boolean_is_boolean(self):
        frame = pd.DataFrame({"b": pd.Series([True, None], dtype="boolean")})
        assert detect_column_kinds(frame).of("b") is ColumnKind.BOOLEAN

    def test_detect_kind_matches_frame_level_detection(self, mixed_frame):
        kinds = detect_column_kinds(mixed_frame)
        for column in mixed_frame.columns:
            assert detect_kind(mixed_frame[column]) is kinds.of(column)


class TestStructuralGuarantees:
    def test_column_order_is_preserved(self, mixed_frame):
        kinds = detect_column_kinds(mixed_frame)
        assert list(kinds.mapping) == list(mixed_frame.columns)

    def test_non_string_labels_are_preserved_so_they_still_index_the_frame(self):
        frame = pd.DataFrame({0: [1, 2], 1: ["a", "b"]})
        kinds = detect_column_kinds(frame)
        assert kinds.numeric == (0,)
        assert frame[list(kinds.numeric)].shape == (2, 1)

    def test_serialisation_stringifies_labels(self):
        frame = pd.DataFrame({0: [1, 2]})
        assert detect_column_kinds(frame).to_dict() == {"0": "numeric"}

    def test_columns_of_accepts_several_kinds(self, mixed_frame):
        kinds = detect_column_kinds(mixed_frame)
        selected = kinds.columns_of(ColumnKind.BOOLEAN, ColumnKind.DATETIME)
        assert selected == ("bool_col", "datetime_col")

    def test_a_frame_with_rows_removed_still_classifies(self, mixed_frame):
        kinds = detect_column_kinds(mixed_frame.iloc[0:0])
        assert kinds.of("int_col") is ColumnKind.NUMERIC

    def test_frame_is_not_modified(self, mixed_frame):
        before = mixed_frame.copy(deep=True)
        detect_column_kinds(mixed_frame)
        pd.testing.assert_frame_equal(mixed_frame, before)


class TestRejectedInput:
    def test_duplicate_labels_are_rejected(self):
        frame = pd.DataFrame(np.arange(4).reshape(2, 2), columns=["a", "a"])
        with pytest.raises(SchemaError, match="Duplicate column labels"):
            detect_column_kinds(frame)

    @pytest.mark.parametrize("value", [None, [1, 2, 3], pd.Series([1, 2])])
    def test_non_dataframe_input_is_rejected(self, value):
        with pytest.raises(ValidationError, match="DataFrame is required"):
            detect_column_kinds(value)

    def test_unknown_column_lookup_is_an_error(self, mixed_frame):
        kinds = detect_column_kinds(mixed_frame)
        with pytest.raises(SchemaError, match="not present"):
            kinds.of("missing_column")
