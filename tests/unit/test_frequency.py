"""Tests for :class:`FrequencyTable`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core.exceptions import (
    EmptyDataError,
    MissingValueError,
    NonNumericDataError,
    ShapeError,
    ValidationError,
)
from aidatasetkit.statistics import FrequencyTable

LABELS = ["a", "b", "a", "c", "a", "b"]
NUMBERS = [1, 2, 2, 3, 3, 3]


class TestCounting:
    def test_counts_match_pandas(self):
        table = FrequencyTable(LABELS)
        expected = pd.Series(LABELS).value_counts().to_dict()
        assert table.counts() == expected

    def test_totals_are_reported(self):
        table = FrequencyTable(LABELS)
        assert table.total == 6
        assert table.n_distinct == 3
        assert len(table) == 3

    def test_categorical_values_are_counted_without_numeric_conversion(self):
        assert FrequencyTable(["red", "blue", "red"]).counts() == {"blue": 1, "red": 2}

    @pytest.mark.parametrize(
        "data",
        [
            LABELS,
            tuple(LABELS),
            np.array(LABELS),
            pd.Series(LABELS),
            pd.Series(LABELS, dtype="category"),
            pd.Index(LABELS),
        ],
    )
    def test_supported_containers_agree(self, data):
        assert FrequencyTable(data).counts() == {"a": 3, "b": 2, "c": 1}

    def test_booleans_are_countable(self):
        assert FrequencyTable([True, False, True]).counts() == {False: 1, True: 2}

    def test_numeric_labels_are_plain_python_values(self):
        table = FrequencyTable(np.array([1, 2, 2]))
        assert list(table.counts()) == [1, 2]
        assert all(isinstance(value, int) for value in table.counts())


class TestOrdering:
    def test_rows_are_ordered_by_value_so_cumulation_is_meaningful(self):
        assert list(FrequencyTable(NUMBERS).counts()) == [1, 2, 3]

    def test_string_values_are_ordered_alphabetically(self):
        assert list(FrequencyTable(["c", "a", "b"]).counts()) == ["a", "b", "c"]

    def test_count_ordering_is_available(self):
        table = FrequencyTable(LABELS, sort="count")
        assert list(table.counts()) == ["a", "b", "c"]

    def test_count_ordering_breaks_ties_by_value(self):
        table = FrequencyTable(["z", "y", "z", "y", "x"], sort="count")
        assert list(table.counts()) == ["y", "z", "x"]

    def test_unorderable_mixed_values_still_produce_a_stable_table(self):
        table = FrequencyTable(pd.Series([1, "a", 1], dtype=object))
        assert table.total == 3
        assert table.n_distinct == 2

    def test_an_unknown_sort_is_rejected(self):
        with pytest.raises(ValidationError, match="sort"):
            FrequencyTable(LABELS, sort="frequency")


class TestProportions:
    def test_relative_frequencies_sum_to_one(self):
        table = FrequencyTable(LABELS)
        assert sum(table.relative().values()) == pytest.approx(1.0)

    def test_percentages_sum_to_one_hundred(self):
        table = FrequencyTable(LABELS)
        assert sum(table.percentage().values()) == pytest.approx(100.0)

    def test_relative_frequency_matches_the_count_ratio(self):
        table = FrequencyTable(LABELS)
        assert table.relative()["a"] == pytest.approx(3 / 6)
        assert table.percentage()["a"] == pytest.approx(50.0)

    def test_percentages_are_a_hundred_times_the_proportions(self):
        table = FrequencyTable(LABELS)
        for value, share in table.relative().items():
            assert table.percentage()[value] == pytest.approx(100.0 * share)

    def test_an_uneven_split_still_sums_exactly(self):
        table = FrequencyTable(["a", "b", "c"])
        assert sum(table.relative().values()) == pytest.approx(1.0)
        assert sum(table.percentage().values()) == pytest.approx(100.0)


class TestCumulation:
    def test_cumulative_counts_accumulate_in_table_order(self):
        assert FrequencyTable(NUMBERS).cumulative() == {1: 1, 2: 3, 3: 6}

    def test_the_final_cumulative_count_is_the_total(self):
        table = FrequencyTable(LABELS)
        assert list(table.cumulative().values())[-1] == table.total

    def test_cumulative_relative_frequencies_end_at_one(self):
        table = FrequencyTable(LABELS)
        assert list(table.cumulative_relative().values())[-1] == pytest.approx(1.0)

    def test_cumulative_relative_frequencies_are_non_decreasing(self):
        values = list(FrequencyTable(NUMBERS).cumulative_relative().values())
        assert all(a <= b for a, b in zip(values, values[1:]))


class TestWeightedMean:
    def test_it_equals_the_mean_of_the_ungrouped_sample(self):
        assert FrequencyTable(NUMBERS).weighted_mean() == pytest.approx(np.mean(NUMBERS))

    def test_it_matches_its_definition(self):
        table = FrequencyTable(NUMBERS)
        counts = table.counts()
        expected = sum(value * count for value, count in counts.items()) / sum(
            counts.values()
        )
        assert table.weighted_mean() == pytest.approx(expected)

    def test_it_handles_a_heavily_weighted_distribution(self):
        assert FrequencyTable([1] * 99 + [100]).weighted_mean() == pytest.approx(1.99)

    def test_categorical_values_cannot_have_a_mean(self):
        with pytest.raises(NonNumericDataError, match="numeric"):
            FrequencyTable(LABELS).weighted_mean()

    def test_a_counted_missing_category_cannot_have_a_mean(self):
        table = FrequencyTable([1.0, 2.0, np.nan], nan_policy="include")
        with pytest.raises(NonNumericDataError):
            table.weighted_mean()


class TestMissingValues:
    def test_missing_values_raise_by_default(self):
        with pytest.raises(MissingValueError, match="missing"):
            FrequencyTable(["a", None, "b"])

    def test_omission_removes_them_from_the_total(self):
        table = FrequencyTable(["a", None, "b"], nan_policy="omit")
        assert table.total == 2
        assert table.n_distinct == 2

    def test_inclusion_counts_them_as_a_category(self):
        table = FrequencyTable([1.0, 2.0, np.nan], nan_policy="include")
        assert table.total == 3
        assert table.n_distinct == 3

    def test_an_unknown_policy_is_rejected(self):
        with pytest.raises(ValidationError, match="nan_policy"):
            FrequencyTable(LABELS, nan_policy="propagate")


class TestOutputViews:
    def test_to_frame_has_one_row_per_distinct_value(self):
        frame = FrequencyTable(LABELS).to_frame()
        assert list(frame.columns) == [
            "value",
            "count",
            "relative",
            "percentage",
            "cumulative",
            "cumulative_relative",
        ]
        assert len(frame) == 3
        assert frame["count"].sum() == 6

    def test_to_dict_is_json_serialisable(self):
        import json

        payload = json.loads(json.dumps(FrequencyTable(LABELS).to_dict()))
        assert payload["counts"]["a"] == 3

    def test_repr_is_informative(self):
        assert "total=6" in repr(FrequencyTable(LABELS))

    def test_returned_mappings_do_not_alias_internal_state(self):
        table = FrequencyTable(LABELS)
        table.counts()["a"] = 999
        assert table.counts()["a"] == 3


class TestRejectedInput:
    def test_empty_input_is_an_error(self):
        with pytest.raises(EmptyDataError):
            FrequencyTable([])

    def test_all_missing_under_omission_is_an_error(self):
        with pytest.raises(EmptyDataError):
            FrequencyTable([None, None], nan_policy="omit")

    @pytest.mark.parametrize("data", ["abc", {"a": 1}, {1, 2}])
    def test_containers_that_are_not_samples_are_rejected(self, data):
        with pytest.raises(ValidationError):
            FrequencyTable(data)

    def test_a_dataframe_is_rejected(self):
        with pytest.raises(ShapeError, match="one-dimensional"):
            FrequencyTable(pd.DataFrame({"a": [1, 2]}))

    def test_two_dimensional_input_is_rejected(self):
        with pytest.raises(ShapeError, match="one-dimensional"):
            FrequencyTable(np.zeros((2, 2)))
