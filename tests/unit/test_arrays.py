"""Tests for the numeric input gateway."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core import to_float_array
from aidatasetkit.core.arrays import to_float_arrays
from aidatasetkit.core.exceptions import (
    EmptyDataError,
    MissingValueError,
    NonFiniteValueError,
    NonNumericDataError,
    ShapeError,
    ValidationError,
)


class TestAcceptedContainers:
    @pytest.mark.parametrize(
        "data",
        [
            [1, 2, 3],
            (1, 2, 3),
            np.array([1, 2, 3]),
            np.array([1.0, 2.0, 3.0]),
            pd.Series([1, 2, 3]),
            pd.Index([1, 2, 3]),
            pd.Series([1, 2, 3], dtype="Int64"),
            pd.Series([1.0, 2.0, 3.0], dtype="Float64"),
        ],
    )
    def test_supported_containers_produce_the_same_float_array(self, data):
        result = to_float_array(data)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_booleans_are_numeric(self):
        np.testing.assert_array_equal(
            to_float_array([True, False, True]), np.array([1.0, 0.0, 1.0])
        )

    def test_result_is_a_copy_and_does_not_alias_the_input(self):
        source = np.array([1.0, 2.0, 3.0])
        result = to_float_array(source)
        result[0] = 99.0
        assert source[0] == 1.0

    def test_input_series_is_not_modified(self):
        series = pd.Series([1.0, 2.0, 3.0])
        to_float_array(series)
        assert series.tolist() == [1.0, 2.0, 3.0]


class TestMissingValues:
    def test_missing_values_raise_by_default(self):
        with pytest.raises(MissingValueError, match="1 missing value"):
            to_float_array([1.0, np.nan, 3.0])

    def test_omit_is_explicit_and_drops_missing_values(self):
        np.testing.assert_array_equal(
            to_float_array([1.0, np.nan, 3.0], nan_policy="omit"), np.array([1.0, 3.0])
        )

    def test_pandas_na_is_treated_as_missing(self):
        series = pd.Series([1, pd.NA, 3], dtype="Int64")
        with pytest.raises(MissingValueError):
            to_float_array(series)
        np.testing.assert_array_equal(
            to_float_array(series, nan_policy="omit"), np.array([1.0, 3.0])
        )

    def test_omitting_everything_is_an_error_not_an_empty_result(self):
        with pytest.raises(EmptyDataError, match="only missing values"):
            to_float_array([np.nan, np.nan], nan_policy="omit")

    def test_unknown_policy_is_rejected(self):
        with pytest.raises(ValidationError, match="nan_policy"):
            to_float_array([1.0, 2.0], nan_policy="propagate")


class TestInfinities:
    def test_infinities_raise_by_default(self):
        with pytest.raises(NonFiniteValueError, match="1 infinite value"):
            to_float_array([1.0, np.inf, 3.0])

    def test_negative_infinity_is_also_rejected(self):
        with pytest.raises(NonFiniteValueError):
            to_float_array([1.0, -np.inf])

    def test_infinities_survive_when_explicitly_allowed(self):
        result = to_float_array([1.0, np.inf], allow_inf=True)
        assert np.isinf(result[1])

    def test_missing_policy_is_applied_before_the_infinity_check(self):
        result = to_float_array([1.0, np.nan, np.inf], nan_policy="omit", allow_inf=True)
        np.testing.assert_array_equal(result, np.array([1.0, np.inf]))


class TestRejectedInput:
    def test_empty_input_is_an_error(self):
        with pytest.raises(EmptyDataError):
            to_float_array([])

    @pytest.mark.parametrize("data", [[[1, 2], [3, 4]], np.zeros((2, 2))])
    def test_two_dimensional_input_is_rejected(self, data):
        with pytest.raises(ShapeError, match="one-dimensional"):
            to_float_array(data)

    def test_dataframe_gets_a_targeted_message(self):
        with pytest.raises(ShapeError, match="Select a single column"):
            to_float_array(pd.DataFrame({"a": [1, 2]}))

    def test_scalar_is_rejected(self):
        with pytest.raises(ShapeError):
            to_float_array(5)

    @pytest.mark.parametrize("data", ["abc", b"abc", {"a": 1}, {1, 2, 3}])
    def test_containers_that_cannot_hold_a_sample_are_rejected(self, data):
        with pytest.raises(ValidationError):
            to_float_array(data)

    def test_text_values_are_rejected(self):
        with pytest.raises(NonNumericDataError):
            to_float_array(["a", "b"])

    @pytest.mark.parametrize(
        "series",
        [
            pd.Series(["1", "2"], dtype="string"),
            pd.Series(["1", "2"], dtype="object"),
            pd.Series(["1.5", None], dtype="object"),
        ],
    )
    def test_numeric_looking_text_is_rejected_not_parsed(self, series):
        """A numeric column stored as text is a quality finding, not a conversion."""
        with pytest.raises(NonNumericDataError, match="text|numeric"):
            to_float_array(series)

    def test_categorical_columns_are_rejected(self):
        with pytest.raises(NonNumericDataError, match="categorical"):
            to_float_array(pd.Series(pd.Categorical([1, 2, 1])))

    def test_object_columns_holding_real_numbers_are_accepted(self):
        result = to_float_array(pd.Series([1, 2.5, None], dtype="object"), nan_policy="omit")
        np.testing.assert_array_equal(result, np.array([1.0, 2.5]))

    def test_timedelta_series_is_rejected(self):
        with pytest.raises(NonNumericDataError, match="timedelta|Datetime"):
            to_float_array(pd.to_timedelta([1, 2], unit="D").to_series())

    def test_datetimes_are_rejected_rather_than_silently_cast(self):
        with pytest.raises(NonNumericDataError, match="Datetime"):
            to_float_array(pd.to_datetime(["2024-01-01", "2024-01-02"]).to_numpy())

    def test_error_messages_use_the_supplied_name(self):
        with pytest.raises(EmptyDataError, match="weights"):
            to_float_array([], name="weights")


class TestPairedConversion:
    def test_both_inputs_are_converted(self):
        x, y = to_float_arrays([1, 2, 3], pd.Series([4.0, 5.0, 6.0]))
        np.testing.assert_array_equal(x, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(y, np.array([4.0, 5.0, 6.0]))

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ShapeError, match="same length"):
            to_float_arrays([1, 2, 3], [1, 2])

    def test_missing_values_raise_by_default(self):
        with pytest.raises(MissingValueError, match="incomplete pair"):
            to_float_arrays([1.0, np.nan], [1.0, 2.0])

    def test_omission_removes_the_whole_pair_from_both_sides(self):
        """Independent cleaning would misalign the two series."""
        x, y = to_float_arrays(
            [1.0, np.nan, 3.0, 4.0], [10.0, 20.0, np.nan, 40.0], nan_policy="omit"
        )
        np.testing.assert_array_equal(x, np.array([1.0, 4.0]))
        np.testing.assert_array_equal(y, np.array([10.0, 40.0]))

    def test_the_two_results_always_have_equal_length(self):
        x, y = to_float_arrays(
            [1.0, np.nan, 3.0], [np.nan, 2.0, 3.0], nan_policy="omit"
        )
        assert x.size == y.size == 1

    def test_no_complete_pairs_is_an_error(self):
        with pytest.raises(EmptyDataError, match="No complete pairs"):
            to_float_arrays([1.0, np.nan], [np.nan, 2.0], nan_policy="omit")

    def test_empty_inputs_are_rejected(self):
        with pytest.raises(EmptyDataError):
            to_float_arrays([], [])

    def test_infinities_raise_by_default(self):
        with pytest.raises(NonFiniteValueError):
            to_float_arrays([1.0, np.inf], [1.0, 2.0])

    def test_infinities_survive_when_allowed(self):
        x, _ = to_float_arrays([1.0, np.inf], [1.0, 2.0], allow_inf=True)
        assert np.isinf(x[1])

    def test_two_dimensional_input_is_rejected(self):
        with pytest.raises(ShapeError, match="one-dimensional"):
            to_float_arrays(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_unknown_policy_is_rejected(self):
        with pytest.raises(ValidationError, match="nan_policy"):
            to_float_arrays([1.0], [2.0], nan_policy="propagate")

    def test_error_messages_use_the_supplied_names(self):
        with pytest.raises(ShapeError, match="feature.*target|target.*feature"):
            to_float_arrays([1.0, 2.0], [1.0], names=("feature", "target"))
