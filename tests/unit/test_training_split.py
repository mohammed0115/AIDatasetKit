"""The experimental boundary: drawn once, drawn the same way twice, never crossed.

Every number a comparison produces depends on this split being right, and the two
ways it goes wrong are both silent. A row on both sides means a model is judged on
what it learned from. A row on neither means the reported training size is a
fiction. Both are checked on every split rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core import KitConfig
from aidatasetkit.core.exceptions import ConfigurationError, TrainingError
from aidatasetkit.core.types import TaskType
from aidatasetkit.training import split_rows


@pytest.fixture
def labels() -> pd.Series:
    """A 30/70 binary target, imbalanced enough for stratification to matter."""
    return pd.Series(["churn"] * 60 + ["stay"] * 140)


@pytest.fixture
def quantities() -> pd.Series:
    return pd.Series(np.linspace(-50.0, 250.0, 200))


class TestEveryRowHasExactlyOneRole:
    @pytest.mark.parametrize("task", [TaskType.CLASSIFICATION, TaskType.REGRESSION])
    def test_the_two_sides_are_disjoint(self, labels, quantities, task):
        y = labels if task is TaskType.CLASSIFICATION else quantities
        split = split_rows(y, task)
        assert not (set(split.train_positions) & set(split.evaluation_positions))

    @pytest.mark.parametrize("task", [TaskType.CLASSIFICATION, TaskType.REGRESSION])
    def test_together_they_account_for_every_row(self, labels, quantities, task):
        y = labels if task is TaskType.CLASSIFICATION else quantities
        split = split_rows(y, task)
        covered = set(split.train_positions) | set(split.evaluation_positions)
        assert covered == set(range(len(y)))
        assert split.train_count + split.evaluation_count == len(y)

    def test_the_evaluation_share_follows_the_configuration(self, labels):
        split = split_rows(labels, TaskType.CLASSIFICATION, KitConfig(validation_size=0.25))
        assert split.evaluation_count == pytest.approx(len(labels) * 0.25, abs=1)


class TestReproducibility:
    def test_the_same_seed_gives_the_same_split(self, labels):
        first = split_rows(labels, TaskType.CLASSIFICATION, KitConfig(random_state=7))
        second = split_rows(labels, TaskType.CLASSIFICATION, KitConfig(random_state=7))
        np.testing.assert_array_equal(first.train_positions, second.train_positions)
        assert first.fingerprint == second.fingerprint

    def test_a_different_seed_can_give_a_different_split(self, labels):
        """Otherwise the seed is decoration and the previous test proves nothing."""
        first = split_rows(labels, TaskType.CLASSIFICATION, KitConfig(random_state=7))
        second = split_rows(labels, TaskType.CLASSIFICATION, KitConfig(random_state=99))
        assert first.fingerprint != second.fingerprint

    def test_the_fingerprint_distinguishes_row_identity_not_row_count(self, labels):
        """Equal counts are not equal rows, which is the whole fairness question."""
        first = split_rows(labels, TaskType.CLASSIFICATION, KitConfig(random_state=7))
        second = split_rows(labels, TaskType.CLASSIFICATION, KitConfig(random_state=99))
        assert first.evaluation_count == second.evaluation_count
        assert first.fingerprint != second.fingerprint

    def test_regression_splits_are_reproducible_too(self, quantities):
        first = split_rows(quantities, TaskType.REGRESSION, KitConfig(random_state=3))
        second = split_rows(quantities, TaskType.REGRESSION, KitConfig(random_state=3))
        assert first.fingerprint == second.fingerprint


class TestStratification:
    def test_classification_preserves_the_class_distribution(self, labels):
        split = split_rows(labels, TaskType.CLASSIFICATION)
        assert split.stratified
        whole = labels.value_counts(normalize=True)
        for side in (split.take(labels), split.take(labels, evaluation=True)):
            observed = side.value_counts(normalize=True)
            for label in whole.index:
                assert observed[label] == pytest.approx(whole[label], abs=0.02)

    def test_regression_does_not_pretend_to_stratify(self, quantities):
        split = split_rows(quantities, TaskType.REGRESSION)
        assert not split.stratified
        assert "no classes to preserve" in split.note

    def test_a_class_too_rare_to_split_is_explained_rather_than_crashing(self):
        y = pd.Series(["common"] * 99 + ["unique"])
        split = split_rows(y, TaskType.CLASSIFICATION)
        assert not split.stratified
        assert "appears 1 time" in split.note

    def test_turning_shuffle_off_disables_stratification_and_says_so(self, labels):
        split = split_rows(labels, TaskType.CLASSIFICATION, KitConfig(shuffle=False))
        assert not split.stratified
        assert "shuffle is off" in split.note

    def test_a_rare_class_still_reaches_both_sides_when_stratified(self):
        y = pd.Series(["a"] * 180 + ["rare"] * 20)
        split = split_rows(y, TaskType.CLASSIFICATION)
        assert split.stratified
        assert "rare" in set(split.take(y))
        assert "rare" in set(split.take(y, evaluation=True))

    def test_the_flag_records_the_outcome_rather_than_the_request(self):
        """Asking for stratification is not the same as getting it.

        scikit-learn allocates per-class evaluation rows by rounding, and a class
        rare enough rounds to zero on one side -- leaving a split that requested
        stratification and did not achieve it. The flag is what a reader trusts
        when a per-class metric looks strange, so it reports what happened.
        """
        y = pd.Series(["a"] * 300 + ["b"] * 300 + ["rare"] * 3)
        split = split_rows(y, TaskType.CLASSIFICATION)
        both_sides = set(split.take(y).unique()) == set(
            split.take(y, evaluation=True).unique()
        )
        assert split.stratified is both_sides


class TestRowsWithoutALabel:
    def test_a_missing_target_value_is_refused_before_splitting(self):
        """A row with no label can neither train nor judge."""
        y = pd.Series(["a", "b", None, "a"] * 25)
        with pytest.raises(TrainingError, match="missing value"):
            split_rows(y, TaskType.CLASSIFICATION)

    def test_the_refusal_says_the_decision_is_the_caller_s(self):
        y = pd.Series([1.0, 2.0, np.nan] * 30)
        with pytest.raises(TrainingError, match="deliberately"):
            split_rows(y, TaskType.REGRESSION)

    def test_a_complete_target_still_splits(self):
        y = pd.Series(["a", "b"] * 50)
        assert split_rows(y, TaskType.CLASSIFICATION).train_count == 80


class TestTheSplitCannotBeEditedAfterTheFact:
    def test_the_position_arrays_are_read_only(self):
        """The fingerprint's purpose is cross-run verification of row identity."""
        split = split_rows(pd.Series(["a", "b"] * 50), TaskType.CLASSIFICATION)
        with pytest.raises(ValueError, match="read-only|assignment destination"):
            split.train_positions[0] = 999

    def test_and_so_the_fingerprint_cannot_be_changed(self):
        split = split_rows(pd.Series(["a", "b"] * 50), TaskType.CLASSIFICATION)
        before = split.fingerprint
        with pytest.raises(ValueError):
            split.evaluation_positions[0] = 12345
        assert split.fingerprint == before


class TestPositionalSelection:
    """Row identity is a position, never an index label."""

    @pytest.fixture
    def awkward(self):
        return pd.DataFrame(
            {"a": np.arange(100.0), "y": ["x", "z"] * 50},
            index=[f"row-{i:03d}" for i in range(100)],
        )

    def test_a_string_index_selects_the_right_rows(self, awkward):
        split = split_rows(awkward["y"], TaskType.CLASSIFICATION)
        train = split.take(awkward)
        assert list(train["a"]) == [float(p) for p in split.train_positions]

    def test_a_non_contiguous_integer_index_is_not_used_as_a_position(self):
        frame = pd.DataFrame(
            {"a": np.arange(50.0), "y": ["x", "z"] * 25},
            index=np.arange(1000, 1100, 2),
        )
        split = split_rows(frame["y"], TaskType.CLASSIFICATION)
        evaluation = split.take(frame, evaluation=True)
        assert list(evaluation["a"]) == [float(p) for p in split.evaluation_positions]

    def test_a_duplicated_index_label_does_not_multiply_rows(self):
        frame = pd.DataFrame({"a": np.arange(40.0), "y": ["x", "z"] * 20}, index=[7] * 40)
        split = split_rows(frame["y"], TaskType.CLASSIFICATION)
        assert len(split.take(frame)) == split.train_count
        assert len(split.take(frame, evaluation=True)) == split.evaluation_count

    def test_taking_from_data_of_the_wrong_length_is_refused(self, awkward):
        split = split_rows(awkward["y"], TaskType.CLASSIFICATION)
        with pytest.raises(TrainingError, match="drawn for 100 rows"):
            split.take(awkward.iloc[:50])

    def test_the_caller_frame_is_not_modified(self, awkward):
        before = awkward.copy(deep=True)
        split = split_rows(awkward["y"], TaskType.CLASSIFICATION)
        split.take(awkward)
        split.take(awkward, evaluation=True)
        pd.testing.assert_frame_equal(awkward, before)


class TestTinyAndInvalidData:
    def test_a_single_row_cannot_be_split(self):
        with pytest.raises(TrainingError, match="at least 2 rows"):
            split_rows(pd.Series([1.0]), TaskType.REGRESSION)

    def test_zero_rows_cannot_be_split(self):
        with pytest.raises(TrainingError, match="at least 2 rows"):
            split_rows(pd.Series([], dtype="float64"), TaskType.REGRESSION)

    def test_a_split_that_would_empty_the_training_side_is_refused(self):
        """Three rows at 99% leaves nothing to learn from.

        The evaluation side cannot be emptied this way -- scikit-learn rounds it
        up to at least one row -- so this is the direction that is actually
        reachable, and it arrives as the project's own error rather than
        scikit-learn's.
        """
        with pytest.raises(TrainingError, match="could not be divided"):
            split_rows(
                pd.Series([1.0, 2.0, 3.0]),
                TaskType.REGRESSION,
                KitConfig(validation_size=0.99),
            )

    def test_the_refusal_names_the_setting_that_caused_it(self):
        with pytest.raises(TrainingError, match="validation_size=0.99"):
            split_rows(
                pd.Series([1.0, 2.0, 3.0]),
                TaskType.REGRESSION,
                KitConfig(validation_size=0.99),
            )

    def test_two_rows_split_into_one_each(self):
        """The smallest split that means anything, and it is allowed."""
        split = split_rows(pd.Series([1.0, 2.0]), TaskType.REGRESSION)
        assert split.train_count == 1 and split.evaluation_count == 1


class TestTheSplitRecord:
    def test_it_serialises_without_carrying_any_data(self, labels):
        import json

        payload = json.loads(json.dumps(split_rows(labels, TaskType.CLASSIFICATION).to_dict()))
        assert payload["train_rows"] + payload["evaluation_rows"] == len(labels)
        assert "positions" not in payload
        for value in payload.values():
            assert not isinstance(value, list), "row identities are data, not metadata"

    def test_it_records_how_the_decision_was_reached(self, labels):
        split = split_rows(labels, TaskType.CLASSIFICATION)
        assert split.note
        assert split.to_dict()["note"] == split.note


class TestTheSeedMustBeASeed:
    """S7 consumes ``random_state`` for splitting, which made this reachable.

    A generator object is mutable state: every draw advances it, so two runs
    configured identically would not agree -- the one thing a reproducibility
    contract cannot allow. The field was annotated ``int`` and enforced nothing.
    """

    def test_a_plain_int_is_accepted(self):
        assert KitConfig(random_state=7).random_state == 7

    @pytest.mark.parametrize(
        "seed",
        [
            pytest.param(np.int64(42), id="np.int64"),
            pytest.param(np.int32(42), id="np.int32"),
            pytest.param(np.uint16(42), id="np.uint16"),
            pytest.param(np.arange(3)[1], id="element-of-an-array"),
        ],
    )
    def test_a_numpy_integer_is_accepted_too(self, seed):
        """Immutable, deterministic, and what indexing an array hands you.

        Refusing these would turn an ordinary way of getting hold of a number
        into an error, and the message about mutable generator state would be
        untrue of them.
        """
        assert int(KitConfig(random_state=seed).random_state) == int(seed)

    @pytest.mark.parametrize(
        "seed",
        [
            pytest.param(np.random.RandomState(0), id="RandomState"),
            pytest.param(np.random.default_rng(0), id="Generator"),
            pytest.param(None, id="None"),
            pytest.param(1.5, id="float"),
            pytest.param("42", id="str"),
            pytest.param(True, id="bool"),
        ],
    )
    def test_anything_that_is_not_an_integer_is_refused(self, seed):
        with pytest.raises(ConfigurationError, match="must be an integer"):
            KitConfig(random_state=seed)

    def test_the_refusal_explains_why_a_generator_is_not_a_seed(self):
        with pytest.raises(ConfigurationError, match="mutable state"):
            KitConfig(random_state=np.random.RandomState(0))

    def test_a_bool_is_refused_rather_than_read_as_one(self):
        """``True`` is an int in Python and would silently become the seed 1."""
        with pytest.raises(ConfigurationError, match="bool"):
            KitConfig(random_state=True)

    @pytest.mark.parametrize("seed", [-1, 2**32, 2**40])
    def test_a_seed_outside_numpys_range_is_refused_here_not_later(self, seed):
        """Otherwise it fails inside a splitter, blaming validation_size."""
        with pytest.raises(ConfigurationError, match="between 0 and"):
            KitConfig(random_state=seed)

    def test_the_boundary_values_are_accepted(self):
        assert KitConfig(random_state=0).random_state == 0
        assert KitConfig(random_state=2**32 - 1).random_state == 2**32 - 1

    def test_and_this_is_what_kept_splitting_reproducible(self, labels):
        """The defect, had it survived, expressed through the S7 public path."""
        with pytest.raises(ConfigurationError):
            split_rows(
                labels,
                TaskType.CLASSIFICATION,
                KitConfig(random_state=np.random.RandomState(0)),
            )
