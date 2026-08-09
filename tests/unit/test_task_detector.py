"""Tests for :class:`TaskDetector`."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core import KitConfig, TaskType
from aidatasetkit.core.exceptions import (
    AmbiguousTaskError,
    EmptyDataError,
    UnsupportedTaskError,
    ValidationError,
)
from aidatasetkit.profiling import TaskDetector


@pytest.fixture
def detector() -> TaskDetector:
    return TaskDetector()


class TestBinaryClassification:
    @pytest.mark.parametrize(
        "values",
        [
            ["yes", "no", "yes", "no"],
            [0, 1, 0, 1],
            [True, False, True, False],
            pd.Categorical(["a", "b", "a", "b"]),
            [1.5, 2.5, 1.5, 2.5],
        ],
    )
    def test_two_distinct_values_are_binary_classification(self, detector, values):
        profile = detector.detect(pd.Series(values))
        assert profile.task_type is TaskType.CLASSIFICATION
        assert profile.is_binary
        assert profile.n_classes == 2

    def test_class_counts_and_ratios_are_reported(self, detector):
        profile = detector.detect(pd.Series(["a"] * 8 + ["b"] * 2))
        assert profile.class_counts == {"a": 8, "b": 2}
        assert profile.class_ratios == {"a": pytest.approx(0.8), "b": pytest.approx(0.2)}
        assert profile.imbalance_ratio == pytest.approx(4.0)
        assert sum(profile.class_ratios.values()) == pytest.approx(1.0)

    def test_a_balanced_target_has_an_imbalance_ratio_of_one(self, detector):
        assert detector.detect(pd.Series([0, 1] * 10)).imbalance_ratio == pytest.approx(1.0)

    def test_binary_targets_carry_no_regression_summary(self, detector):
        assert detector.detect(pd.Series([0, 1, 0])).numeric is None


class TestMulticlassClassification:
    def test_several_text_classes_are_multiclass(self, detector):
        profile = detector.detect(pd.Series(["red", "blue", "green", "red"]))
        assert profile.task_type is TaskType.CLASSIFICATION
        assert not profile.is_binary
        assert profile.n_classes == 3

    def test_a_categorical_target_with_many_classes_is_not_regression(self, detector):
        """Text labels are classes however many of them there are."""
        labels = [f"class_{value % 30}" for value in range(600)]
        profile = detector.detect(pd.Series(labels))
        assert profile.task_type is TaskType.CLASSIFICATION
        assert profile.n_classes == 30

    def test_repeating_integer_codes_are_multiclass(self, detector):
        profile = detector.detect(pd.Series([value % 4 for value in range(400)]))
        assert profile.task_type is TaskType.CLASSIFICATION
        assert profile.n_classes == 4

    def test_multiclass_targets_have_no_positive_label(self, detector):
        profile = detector.detect(pd.Series(["a", "b", "c", "a"]))
        assert profile.positive_label is None
        assert not profile.positive_label_resolved
        assert "not binary" in profile.detection_note


class TestRegression:
    def test_continuous_values_are_regression(self, detector):
        profile = detector.detect(pd.Series([10.3, 22.7, 18.2, 9.9, 31.4]))
        assert profile.task_type is TaskType.REGRESSION
        assert profile.n_classes is None
        assert profile.classes is None
        assert profile.class_counts is None

    def test_an_integer_target_with_many_values_is_regression(self, detector):
        """Ages are integers and are not classes."""
        ages = pd.Series([18 + value % 63 for value in range(500)])
        profile = detector.detect(ages)
        assert profile.task_type is TaskType.REGRESSION

    def test_a_regression_summary_is_reported(self, detector):
        from aidatasetkit.statistics import StatisticsEngine

        values = pd.Series([10.3, 22.7, 18.2, 9.9, 31.4, 27.6])
        profile = detector.detect(values)
        engine = StatisticsEngine(values)
        assert profile.numeric.mean == pytest.approx(engine.mean())
        assert profile.numeric.std == pytest.approx(engine.std())
        assert profile.numeric.minimum == pytest.approx(9.9)
        assert profile.numeric.maximum == pytest.approx(31.4)

    def test_regression_targets_carry_no_class_fields(self, detector):
        profile = detector.detect(pd.Series([1.5, 2.5, 3.5, 4.5]))
        assert profile.class_ratios is None
        assert profile.positive_label is None
        assert profile.imbalance_ratio is None


class TestAmbiguity:
    def test_a_small_distinct_integer_target_is_ambiguous(self, detector):
        with pytest.raises(AmbiguousTaskError, match="could be"):
            detector.detect(pd.Series([1, 2, 3, 4, 5]))

    def test_the_ambiguity_message_names_both_readings(self, detector):
        with pytest.raises(AmbiguousTaskError) as error:
            detector.detect(pd.Series([1, 2, 3, 4, 5]))
        message = str(error.value)
        assert "classification" in message and "regression" in message

    def test_a_hint_settles_it_as_classification(self, detector):
        profile = detector.detect(pd.Series([1, 2, 3, 4, 5]), hint="classification")
        assert profile.task_type is TaskType.CLASSIFICATION
        assert profile.n_classes == 5

    def test_a_hint_settles_it_as_regression(self, detector):
        profile = detector.detect(pd.Series([1, 2, 3, 4, 5]), hint="regression")
        assert profile.task_type is TaskType.REGRESSION

    def test_a_large_all_distinct_text_target_is_ambiguous(self, detector):
        """Free text or an identifier, but not a set of classes."""
        with pytest.raises(AmbiguousTaskError, match="identifier or free text"):
            detector.detect(pd.Series([f"note_{value}" for value in range(50)]))

    def test_a_small_all_distinct_text_target_is_still_classification(self, detector):
        """Two rows reading "churn" and "stay" are all-distinct and ordinary."""
        assert detector.detect(pd.Series(["churn", "stay"])).n_classes == 2

    def test_thresholds_are_configurable(self):
        detector = TaskDetector(KitConfig(task_detection_unique_ratio=1.0))
        assert detector.detect(pd.Series([1, 2, 3, 4, 5])).task_type is (
            TaskType.CLASSIFICATION
        )

    def test_a_contradictory_regression_hint_is_refused(self, detector):
        with pytest.raises(UnsupportedTaskError, match="labels rather than quantities"):
            detector.detect(pd.Series(["red", "blue"]), hint="regression")

    def test_a_single_valued_target_cannot_be_predicted(self, detector):
        with pytest.raises(UnsupportedTaskError, match="nothing to predict"):
            detector.detect(pd.Series([1, 1, 1]))

    def test_an_unsupervised_hint_is_refused(self, detector):
        with pytest.raises(UnsupportedTaskError, match="no target"):
            detector.detect(pd.Series([1, 2, 1]), hint="clustering")

    def test_an_unknown_hint_is_rejected(self, detector):
        with pytest.raises(ValidationError, match="not a valid TaskType"):
            detector.detect(pd.Series([0, 1]), hint="supervised")


class TestPositiveLabel:
    def test_the_zero_one_convention_resolves_to_one(self, detector):
        profile = detector.detect(pd.Series([0, 1, 0, 1]))
        assert profile.positive_label == 1
        assert profile.positive_label_resolved
        assert "0/1 convention" in profile.detection_note

    def test_the_false_true_convention_resolves_to_true(self, detector):
        profile = detector.detect(pd.Series([True, False, True]))
        assert profile.positive_label is True
        assert profile.positive_label_resolved

    def test_arbitrary_labels_are_left_unresolved(self, detector):
        """Nothing says which of "churn" and "stay" is the event of interest."""
        profile = detector.detect(pd.Series(["churn", "stay", "churn"]))
        assert profile.positive_label is None
        assert not profile.positive_label_resolved
        assert "must be supplied" in profile.detection_note

    def test_an_explicit_positive_label_is_honoured(self, detector):
        profile = detector.detect(
            pd.Series(["churn", "stay", "churn"]), positive_label="churn"
        )
        assert profile.positive_label == "churn"
        assert profile.positive_label_resolved

    def test_an_explicit_label_overrides_the_convention(self, detector):
        profile = detector.detect(pd.Series([0, 1, 0]), positive_label=0)
        assert profile.positive_label == 0

    def test_a_label_that_is_not_a_class_is_rejected(self, detector):
        with pytest.raises(ValidationError, match="not among the observed classes"):
            detector.detect(pd.Series(["churn", "stay"]), positive_label="cancelled")

    def test_numeric_labels_beyond_zero_and_one_stay_unresolved(self, detector):
        profile = detector.detect(pd.Series([1, 2, 1, 2]))
        assert not profile.positive_label_resolved


class TestTargetMetadata:
    def test_the_name_is_taken_from_the_series(self, detector):
        profile = detector.detect(pd.Series([0, 1, 0], name="Churn"))
        assert profile.target_name == "Churn"

    def test_an_explicit_name_wins(self, detector):
        profile = detector.detect(pd.Series([0, 1, 0], name="a"), target_name="Churn")
        assert profile.target_name == "Churn"

    def test_missing_values_are_counted_and_excluded(self, detector):
        profile = detector.detect(pd.Series([0, 1, None, 1]))
        assert profile.missing_count == 1
        assert profile.sample_count == 3
        assert profile.class_counts == {0: 1, 1: 2}

    def test_the_detection_note_explains_the_decision(self, detector):
        assert detector.detect(pd.Series([0.5, 1.5, 2.5])).detection_note

    def test_classes_are_ordered_deterministically(self, detector):
        first = detector.detect(pd.Series(["c", "a", "b", "a"])).classes
        second = detector.detect(pd.Series(["b", "c", "a", "a"])).classes
        assert first == second == ("a", "b", "c")


class TestInputHandling:
    @pytest.mark.parametrize(
        "values", [[0, 1, 0, 1], np.array([0, 1, 0, 1]), pd.Index([0, 1, 0, 1])]
    )
    def test_supported_containers_agree(self, detector, values):
        assert detector.detect(values).task_type is TaskType.CLASSIFICATION

    def test_an_all_missing_target_is_rejected(self, detector):
        with pytest.raises(EmptyDataError, match="no non-missing values"):
            detector.detect(pd.Series([None, None], dtype="object"))

    def test_a_dataframe_target_is_rejected(self, detector):
        with pytest.raises(ValidationError, match="one-dimensional"):
            detector.detect(pd.DataFrame({"a": [0, 1]}))

    @pytest.mark.parametrize("value", ["abc", {"a": 1}, {1, 2}])
    def test_containers_that_are_not_samples_are_rejected(self, detector, value):
        with pytest.raises(ValidationError):
            detector.detect(value)

    def test_detection_does_not_modify_the_input(self, detector):
        series = pd.Series([0, 1, None, 1])
        before = series.copy(deep=True)
        detector.detect(series)
        pd.testing.assert_series_equal(series, before)


class TestSerialisation:
    def test_a_classification_profile_round_trips_through_json(self, detector):
        payload = json.loads(
            json.dumps(detector.detect(pd.Series([0, 1, 0, 0])).to_dict())
        )
        assert payload["task_type"] == "classification"
        assert payload["class_counts"] == {"0": 3, "1": 1}
        assert payload["positive_label"] == 1

    def test_a_regression_profile_round_trips_through_json(self, detector):
        payload = json.loads(
            json.dumps(detector.detect(pd.Series([1.5, 2.5, 3.5, 9.0])).to_dict())
        )
        assert payload["task_type"] == "regression"
        assert payload["class_counts"] is None
        assert payload["numeric"]["mean"] == pytest.approx(4.125)

    def test_numpy_class_labels_serialise(self, detector):
        profile = detector.detect(pd.Series(np.array([0, 1, 1], dtype="int64")))
        assert json.loads(json.dumps(profile.to_dict()))["classes"] == [0, 1]
