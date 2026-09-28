"""Counting values that float64 cannot hold, identically on pandas 2 and 3.

pandas 2.x raises a bare ``OverflowError`` from ``value_counts`` when a column
holds a Python integer beyond float64's range; pandas 3 does not. These tests run
unchanged under both majors (CI runs both), so the same assertion proves the
fallback on pandas 2 and the native path on pandas 3 agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core.counting import exact_tally, value_counts
from aidatasetkit.profiling import DataProfiler, TaskDetector

HUGE = 10**400


def _as_pairs(counted: pd.Series) -> list[tuple[object, int]]:
    """Index and counts as plain pairs, with every missing label made one token."""
    return [
        ("<missing>" if pd.isna(label) else label, int(count))
        for label, count in counted.items()
    ]


ORDINARY = {
    "integers": pd.Series([3, 1, 2, 1, 3, 3]),
    "strings": pd.Series(["b", "a", "b", "c", "a", "b"]),
    "ties_in_appearance_order": pd.Series([2, 1, 2, 1, 5]),
    "floats_with_nan": pd.Series([1.5, np.nan, 1.5, 2.5, np.nan]),
    "big_floats": pd.Series([1e308, -1e308, 1e308, 0.0]),
    "infinities": pd.Series([np.inf, -np.inf, np.inf, 1.0]),
    "nullable_int": pd.Series([1, None, 1, 2], dtype="Int64"),
    "nullable_float": pd.Series([0.5, None, 0.5], dtype="Float64"),
    "nullable_bool": pd.Series([True, None, False, True], dtype="boolean"),
    "mixed_object": pd.Series([1, "1", 1, "x", 2.0, "x"], dtype=object),
}


class TestTheFallbackAgreesWithPandas:
    """The exact tally is only trustworthy if it is pandas' tally, exactly."""

    @pytest.mark.parametrize("name", sorted(ORDINARY))
    def test_on_ordinary_data(self, name):
        series = ORDINARY[name]
        assert _as_pairs(exact_tally(series)) == _as_pairs(series.value_counts())

    @pytest.mark.parametrize("name", sorted(ORDINARY))
    def test_the_wrapper_counts_exactly_what_pandas_counts(self, name):
        """Every count is pandas' count; only the order of ties is fixed here."""
        series = ORDINARY[name]
        ours, theirs = value_counts(series), series.value_counts()
        assert sorted(_as_pairs(ours), key=repr) == sorted(_as_pairs(theirs), key=repr)
        assert ours.index.dtype == theirs.index.dtype

    @pytest.mark.parametrize("name", sorted(ORDINARY))
    def test_the_wrapper_orders_ties_by_first_appearance(self, name):
        series = ORDINARY[name]
        assert _as_pairs(value_counts(series)) == _as_pairs(exact_tally(series))


class TestTieOrderDoesNotDependOnTheMachine:
    """Found on Linux CI: pandas 2.1 breaks ties with an unstable, CPU-dispatched sort.

    Two ubuntu runners with identical packages recorded different dominant values
    for a column of unique identifiers, because every value tied and pandas 2.1's
    quicksort (AVX-512 where the CPU has it) ordered the ties differently. Here
    pandas is made to return its ties in reverse, as such a sort may, and nothing
    downstream is allowed to notice.
    """

    @pytest.fixture
    @staticmethod
    def unstable(monkeypatch):
        original = pd.Series.value_counts

        def reversed_ties(self, *args, **kwargs):
            counted = original(self, *args, **kwargs)
            order = sorted(range(len(counted)), key=lambda i: (-int(counted.iloc[i]), -i))
            return counted.iloc[order]

        monkeypatch.setattr(pd.Series, "value_counts", reversed_ties)

    def test_the_simulation_really_reverses_ties(self, unstable):
        """Otherwise the tests below could pass against a no-op."""
        assert list(pd.Series(["a", "b", "c"]).value_counts().index) == ["c", "b", "a"]

    def test_the_tally_keeps_first_appearance(self, unstable):
        series = pd.Series(["b", "a", "c", "a", "b", "d"])
        assert list(value_counts(series).index) == ["b", "a", "c", "d"]

    def test_missing_values_take_their_first_position(self, unstable):
        series = pd.Series([np.nan, 2.0, 1.0, np.nan, 1.0, 2.0])
        labels = ["<missing>" if pd.isna(v) else v for v in value_counts(series, dropna=False).index]
        assert labels == ["<missing>", 2.0, 1.0]

    def test_unobserved_categories_come_last(self, unstable):
        series = pd.Series(pd.Categorical(["y", "x"], categories=["z", "x", "y"]))
        assert list(value_counts(series).index) == ["y", "x", "z"]

    def test_a_unique_identifier_column_has_the_same_dominant_value(self, unstable):
        frame = pd.DataFrame({"CustomerID": [f"C{i:04d}" for i in range(600, 0, -1)]})
        column = DataProfiler().profile(frame).column_profiles[0]
        assert column.dominant_value == "C0600"

    def test_the_golden_audit_is_unchanged(self, unstable):
        import json

        from aidatasetkit.evidence.fingerprint import config_fingerprint
        from tests.golden import semantic_fixture_path

        root = str(Path(__file__).resolve().parents[2])
        if root not in sys.path:
            sys.path.insert(0, root)
        from examples.audit_churn.generate_artifacts import build_artifact

        example = Path(root) / "examples" / "audit_churn" / "train.csv"
        rebuilt = build_artifact(pd.read_csv(example)).semantic_dict()
        stored = json.loads(semantic_fixture_path().read_text(encoding="utf-8"))
        assert config_fingerprint(rebuilt) == config_fingerprint(stored)

    def test_missing_values_are_counted_when_asked(self):
        series = pd.Series([np.nan, 1.0, np.nan, 2.0, np.nan])
        tally = _as_pairs(exact_tally(series, dropna=False))
        assert tally[0] == ("<missing>", 3)
        assert sorted(tally[1:], key=str) == [(1.0, 1), (2.0, 1)]


class TestIntegersBeyondFloat64:
    @staticmethod
    def _column():
        return pd.Series([HUGE] * 3 + [1, 2] * 24 + [-HUGE], dtype=object)

    def test_the_tally_does_not_raise(self):
        value_counts(self._column())

    def test_every_value_keeps_its_exact_identity(self):
        labels = set(value_counts(self._column()).index)
        assert HUGE in labels and -HUGE in labels
        assert all(type(label) is int for label in labels)

    def test_counts_and_order_match_pandas_3(self):
        """Most frequent first; ties in order of first appearance."""
        assert _as_pairs(value_counts(self._column())) == [(1, 24), (2, 24), (HUGE, 3), (-HUGE, 1)]

    def test_missing_values_are_dropped_by_default(self):
        column = pd.Series([HUGE, None, HUGE, 1], dtype=object)
        assert _as_pairs(value_counts(column)) == [(HUGE, 2), (1, 1)]

    def test_and_kept_when_asked(self):
        column = pd.Series([HUGE, None, None, None, HUGE, 1], dtype=object)
        assert _as_pairs(value_counts(column, dropna=False))[0] == ("<missing>", 3)

    def test_an_unhashable_cell_still_raises_type_error(self):
        """Callers translate TypeError into a message about the column."""
        column = pd.Series([HUGE, [1, 2]], dtype=object)
        with pytest.raises(TypeError):
            exact_tally(column)


class TestTheLayersThatTally:
    def test_profiling_a_column_of_huge_integers(self):
        frame = pd.DataFrame({"account": pd.Series([HUGE] * 3 + [1, 2] * 24, dtype=object)})
        column = DataProfiler().profile(frame).column_profiles[0]
        assert column.count == 51
        assert column.unique_count == 3
        assert column.dominant_value == 1
        assert column.dominant_ratio == pytest.approx(24 / 51)

    def test_the_profile_is_identical_twice(self):
        frame = pd.DataFrame({"account": pd.Series([HUGE] * 3 + [1, 2] * 24, dtype=object)})
        assert DataProfiler().profile(frame).to_dict() == DataProfiler().profile(frame).to_dict()

    @pytest.mark.parametrize(
        "values",
        [
            [1e308, -1e308, 1e308, 0.0] * 10,
            [np.inf, -np.inf, 1.0, 2.0] * 10,
            [np.nan, 1.0, 2.0, 1.0] * 10,
        ],
        ids=["big_floats", "infinities", "nan"],
    )
    def test_profiling_extreme_floats(self, values):
        frame = pd.DataFrame({"x": values})
        DataProfiler().profile(frame)

    @pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
    def test_profiling_nullable_dtypes(self, dtype):
        values = [True, False, None, True] * 10 if dtype == "boolean" else [1, 2, None, 1] * 10
        frame = pd.DataFrame({"x": pd.Series(values, dtype=dtype)})
        column = DataProfiler().profile(frame).column_profiles[0]
        assert column.missing_count == 10

    def test_profiling_a_mixed_object_column(self):
        frame = pd.DataFrame({"x": pd.Series([1, "1", HUGE, "x", 2.5] * 8, dtype=object)})
        column = DataProfiler().profile(frame).column_profiles[0]
        assert column.unique_count == 5

    def test_detecting_a_target_of_huge_integer_classes(self):
        target = pd.Series([HUGE] * 30 + [1] * 30, dtype=object)
        profile = TaskDetector().detect(target, target_name="t")
        assert set(profile.class_counts.values()) == {30}
        assert HUGE in profile.class_counts

    def test_the_class_order_is_deterministic(self):
        target = pd.Series([HUGE] * 30 + [1] * 30 + [-HUGE] * 30, dtype=object)
        first = TaskDetector().detect(target, target_name="t")
        second = TaskDetector().detect(target.iloc[::-1].reset_index(drop=True), target_name="t")
        assert first.classes == second.classes


class TestTheVectorisedOrderEqualsTheExactOne:
    """G0.1: the vectorised tie ordering is the per-element loop's, value for value.

    ``_ties_by_python_loop`` is the ordering 79e7cc8 introduced and every fixture
    was checked against. The vectorised path replaced it for speed only, so for
    every input it must produce the identical tally -- same labels, same counts,
    same order -- including when pandas hands its ties back reversed.
    """

    BATTERY = {
        "strings_tied": pd.Series(["b", "a", "c", "a", "b", "d", "c"]),
        "numeric_tied": pd.Series([3, 1, 2, 1, 3, 2, 5]),
        "mixed_compatible": pd.Series([1, 1.0, True, 2, 2.0, "1", "1"], dtype=object),
        "nan": pd.Series([np.nan, 2.0, 1.0, np.nan, 1.0, 2.0]),
        "none_and_nan": pd.Series([None, "a", np.nan, "b", "a", None, "b"], dtype=object),
        "nullable_int": pd.Series([2, None, 1, 2, None, 1], dtype="Int64"),
        "nullable_bool": pd.Series([True, None, False, None, True, False], dtype="boolean"),
        "near_float_max": pd.Series([1e308, -1e308, 1e308, 1.7e308, -1e308, 1.7e308]),
        "infinities": pd.Series([np.inf, -np.inf, 1.0, np.inf, -np.inf, 1.0]),
        "single_value": pd.Series([7]),
        "empty": pd.Series([], dtype="float64"),
        "all_distinct": pd.Series(np.random.default_rng(3).permutation(500).astype("int64")),
        "low_cardinality": pd.Series(list("abcabcabcab")),
        "categorical_unobserved": pd.Series(pd.Categorical(["y", "x", "y", "x"], categories=["z", "x", "y"])),
        "datetimes": pd.Series(pd.to_datetime(["2024-01-02", None, "2024-01-01", "2024-01-02", None, "2024-01-01"])),
    }

    @staticmethod
    def _reference(series, dropna):
        from aidatasetkit.core.counting import _ties_by_python_loop

        counted = series.value_counts(dropna=dropna)
        if len(counted) > 1 and counted.duplicated().any():
            counted = _ties_by_python_loop(series, counted)
        return counted

    @pytest.mark.parametrize("name", sorted(BATTERY))
    @pytest.mark.parametrize("dropna", [True, False])
    def test_same_tally_as_the_loop(self, name, dropna):
        series = self.BATTERY[name]
        pd.testing.assert_series_equal(value_counts(series, dropna=dropna), self._reference(series, dropna))

    @pytest.mark.parametrize("name", sorted(BATTERY))
    @pytest.mark.parametrize("dropna", [True, False])
    def test_same_tally_when_pandas_reverses_its_ties(self, name, dropna, monkeypatch):
        original = pd.Series.value_counts

        def reversed_ties(self, *args, **kwargs):
            counted = original(self, *args, **kwargs)
            order = sorted(range(len(counted)), key=lambda i: (-int(counted.iloc[i]), -i))
            return counted.iloc[order]

        expected = self._reference(self.BATTERY[name], dropna)
        monkeypatch.setattr(pd.Series, "value_counts", reversed_ties)
        pd.testing.assert_series_equal(value_counts(self.BATTERY[name], dropna=dropna), expected)

    @pytest.mark.parametrize("name", sorted(BATTERY))
    def test_repeated_calls_agree(self, name):
        series = self.BATTERY[name]
        first = value_counts(series, dropna=False)
        for _ in range(3):
            pd.testing.assert_series_equal(value_counts(series, dropna=False), first)

    @pytest.mark.parametrize("name", sorted(BATTERY))
    def test_the_input_is_not_modified(self, name):
        series = self.BATTERY[name]
        before = series.copy()
        value_counts(series, dropna=False)
        pd.testing.assert_series_equal(series, before)

    def test_huge_integers_still_take_the_exact_path(self):
        series = pd.Series([HUGE, 1, HUGE, 2, 1], dtype=object)
        assert _as_pairs(value_counts(series)) == [(HUGE, 2), (1, 2), (2, 1)]

    def test_one_is_not_the_string_one(self):
        labels = list(value_counts(pd.Series([1, "1", 1, "1", 2], dtype=object)).index)
        assert labels == [1, "1", 2]
        assert type(labels[0]) is int and type(labels[1]) is str

    def test_an_unmatchable_label_falls_back_to_the_exact_loop(self, monkeypatch):
        """If the vectorised path cannot place a label, the loop decides; nothing is guessed."""
        from aidatasetkit.core import counting

        monkeypatch.setattr(counting, "_first_positions", lambda series, counted: None)
        series = self.BATTERY["strings_tied"]
        pd.testing.assert_series_equal(value_counts(series), self._reference(series, True))
