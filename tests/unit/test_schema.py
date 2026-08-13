"""Tests for column-kind and target-semantics detection.

These lock in the ordering rules that dtype predicates make easy to get wrong:
pandas reports booleans as numeric and categoricals as string-like.

The target half of this module exists because a dtype predicate stood in for a
semantic one and two components drew different conclusions from it. Its tests
are written against the *values*, not the dtypes, for the same reason.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core import ColumnKind, detect_column_kinds, detect_kind
from aidatasetkit.core.config import KitConfig
from aidatasetkit.core.exceptions import SchemaError, ValidationError
from aidatasetkit.core.schema import numeric_target_values, target_holds_quantities


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


class TestNumericTargetValues:
    """Dtype is evidence about a target, not proof."""

    @pytest.mark.parametrize(
        "series",
        [
            pd.Series([1.5, 2.5, 3.5]),
            pd.Series([1, 2, 3]),
            pd.Series([1, 2, 3], dtype="Int64"),
            pd.Series([1.5, 2.5], dtype="Float32"),
            pd.Series([1.5, np.nan, 2.5]),
        ],
        ids=["float64", "int64", "nullable-int", "nullable-float", "with-gaps"],
    )
    def test_numeric_dtypes_are_read(self, series):
        values = numeric_target_values(series)
        assert values is not None
        assert values.dtype == np.float64
        assert np.isfinite(values).all()

    @pytest.mark.parametrize(
        "values",
        [
            [1.25, 2.50, 3.75],
            [1, 2, 3],
            [Decimal("1.25"), Decimal("2.50")],
            [Fraction(1, 4), Fraction(1, 2)],
            [np.float64(1.5), np.int64(2)],
        ],
        ids=["python-floats", "python-ints", "decimals", "fractions", "numpy-scalars"],
    )
    def test_object_dtype_holding_numbers_is_read_too(self, values):
        """The gap that let a column of quantities be called "non-numeric"."""
        series = pd.Series(values, dtype=object)
        assert series.dtype == object
        result = numeric_target_values(series)
        assert result is not None
        np.testing.assert_allclose(result, [float(v) for v in values])

    @pytest.mark.parametrize(
        "series",
        [
            pd.Series(["a", "b", "c"]),
            pd.Series(["a", "b"], dtype=object),
            pd.Series([True, False]),
            pd.Series([1, "two", 3], dtype=object),
            pd.Series(pd.Categorical(["low", "high"])),
            pd.Series(pd.to_datetime(["2024-01-01", "2024-02-01"])),
            pd.Series([1 + 2j, 3 + 4j]),
            pd.Series([np.nan, np.nan]),
        ],
        ids=[
            "strings", "object-strings", "booleans", "mixed-object",
            "categorical", "datetime", "complex", "all-missing",
        ],
    )
    def test_everything_else_is_labels_or_nothing(self, series):
        assert numeric_target_values(series) is None

    def test_a_boolean_column_is_a_label_even_though_pandas_calls_it_numeric(self):
        assert pd.api.types.is_numeric_dtype(pd.Series([True, False]))
        assert numeric_target_values(pd.Series([True, False])) is None

    def test_an_integer_beyond_float64_answers_none_rather_than_raising(self):
        """numpy answers this with OverflowError, which is neither TypeError nor
        ValueError -- the sibling module learned that the hard way."""
        series = pd.Series([10**400, 1], dtype=object)
        assert numeric_target_values(series) is None

    def test_the_caller_series_is_not_modified(self):
        series = pd.Series([1.5, np.nan, 2.5])
        before = series.copy()
        numeric_target_values(series)
        pd.testing.assert_series_equal(series, before)


class TestTargetHoldsQuantities:
    """One answer to "is this a quantity", applied by two components."""

    @pytest.mark.parametrize(
        "series",
        [
            pd.Series([1.5, 2.25, 3.75, 4.5] * 15),
            pd.Series([1.25, 2.50, 3.75, 4.10] * 15, dtype=object),
            pd.Series(np.arange(60) * 7),
            pd.Series(np.arange(60) * 7.0),
            pd.Series(np.arange(60), dtype="Int64"),
            pd.Series([float(i) for i in range(60)], dtype=object),
        ],
        ids=[
            "non-integral floats", "object-dtype floats", "60 distinct ints",
            "60 distinct whole floats", "nullable ints", "object whole floats",
        ],
    )
    def test_quantities_are_recognised(self, series):
        assert target_holds_quantities(series)

    @pytest.mark.parametrize(
        "series",
        [
            pd.Series([0, 1] * 30),
            pd.Series([0, 1, 2, 3, 4] * 12),
            pd.Series(["churn", "stay"] * 30),
            pd.Series(["a", "b", "c"] * 20, dtype=object),
            pd.Series([0.0, 1.0] * 30, dtype=object),
            pd.Series([True, False] * 30),
            pd.Series([1.5, 2.5] * 30),
            pd.Series(["only"] * 30),
            pd.Series(np.arange(20) % 20),
        ],
        ids=[
            "binary ints", "five classes", "strings", "object strings",
            "object whole-float labels", "booleans", "two distinct floats",
            "one value", "exactly the class limit",
        ],
    )
    def test_labels_are_not_mistaken_for_quantities(self, series):
        assert not target_holds_quantities(series)

    def test_two_distinct_values_are_a_binary_label_whatever_they_are_spelled_as(self):
        """The detector's own rule, mirrored so the two cannot diverge.

        Two points do not describe a curve, so a two-valued float column is a
        binary label. This is the one case where the shared answer is *more*
        permissive than the guard it replaced, and it is deliberate: the previous
        guard refused what the detector called classification.
        """
        assert not target_holds_quantities(pd.Series([9.99, 19.99] * 30))

    def test_the_ambiguous_middle_is_not_claimed_as_a_quantity(self):
        """Five repeated whole numbers could be classes, grades, or a count.

        The detector refuses to decide without a hint. This function answers
        False, because a caller who has reached the encoder has already said
        which they meant by calling it.
        """
        assert not target_holds_quantities(pd.Series([1, 2, 3, 4, 5] * 4))

    def test_the_class_limit_comes_from_the_shared_configuration(self):
        series = pd.Series(np.arange(30))
        assert target_holds_quantities(series)
        generous = KitConfig(task_detection_max_classes=50)
        assert not target_holds_quantities(series, generous)

    def test_a_non_finite_value_makes_it_a_quantity(self):
        """Mirrors the detector: an infinity fails the whole-number test."""
        series = pd.Series([1.0, 2.0, np.inf] * 20)
        assert target_holds_quantities(series)
