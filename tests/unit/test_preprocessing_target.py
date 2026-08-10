"""Tests for target label encoding.

The property that matters most is not reversibility -- it is that the positive
class survives. Once labels become integers, the probability column an analyst
reads is decided by alphabetical order, which has nothing to do with which
outcome they care about.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core.exceptions import PreprocessingError, UnsupportedTaskError
from aidatasetkit.core.types import TargetProfile, TaskType
from aidatasetkit.preprocessing import TargetLabelEncoder
from aidatasetkit.profiling import TaskDetector


class TestBinaryEncoding:
    @pytest.fixture
    def labels(self) -> pd.Series:
        return pd.Series(["stay", "churn"] * 50, name="Churn")

    def test_labels_become_integers(self, labels):
        encoded = TargetLabelEncoder().fit_transform(labels)
        assert set(np.unique(encoded)) == {0, 1}

    def test_encoding_is_reversible(self, labels):
        encoder = TargetLabelEncoder()
        encoded = encoder.fit_transform(labels)
        assert list(encoder.inverse_transform(encoded)) == list(labels)

    def test_the_mapping_is_recorded_both_ways(self, labels):
        encoder = TargetLabelEncoder().fit(labels)
        encoding = encoder.encoding
        for label, code in encoding.mapping.items():
            assert encoding.inverse[code] == label

    def test_encoding_is_deterministic(self, labels):
        first = TargetLabelEncoder().fit(labels).encoding
        second = TargetLabelEncoder().fit(labels.sample(frac=1.0, random_state=1)).encoding
        assert first.mapping == second.mapping

    def test_the_caller_s_series_is_untouched(self, labels):
        before = labels.copy(deep=True)
        TargetLabelEncoder().fit_transform(labels)
        pd.testing.assert_series_equal(labels, before)

    def test_a_numpy_target_is_untouched(self):
        values = np.array(["a", "b", "a", "b"])
        before = values.copy()
        TargetLabelEncoder().fit_transform(values)
        np.testing.assert_array_equal(values, before)


class TestPositiveLabel:
    @pytest.fixture
    def labels(self) -> pd.Series:
        return pd.Series(["stay", "churn"] * 50, name="Churn")

    def test_the_positive_class_is_not_assumed_to_be_column_one(self, labels):
        """"churn" sorts first, so the positive class is column 0."""
        encoder = TargetLabelEncoder(positive_label="churn").fit(labels)
        assert encoder.encoding.positive_label_encoded == 0
        assert encoder.positive_column_index() == 0

    def test_the_original_positive_label_is_preserved(self, labels):
        encoder = TargetLabelEncoder(positive_label="churn").fit(labels)
        assert encoder.encoding.positive_label == "churn"
        assert encoder.encoding.positive_label_resolved

    def test_it_is_taken_from_the_target_profile_when_not_given(self, labels):
        profile = TaskDetector().detect(labels, target_name="Churn", positive_label="churn")
        encoder = TargetLabelEncoder().fit(labels, profile)
        assert encoder.encoding.positive_label == "churn"
        assert encoder.positive_column_index() == 0

    def test_an_explicit_label_beats_the_profile(self, labels):
        profile = TaskDetector().detect(labels, target_name="Churn", positive_label="churn")
        encoder = TargetLabelEncoder(positive_label="stay").fit(labels, profile)
        assert encoder.encoding.positive_label == "stay"
        assert encoder.positive_column_index() == 1

    def test_an_unresolved_positive_class_refuses_to_guess(self, labels):
        """Returning 1 here is exactly the silent inversion this prevents."""
        encoder = TargetLabelEncoder().fit(labels)
        assert not encoder.encoding.positive_label_resolved
        with pytest.raises(PreprocessingError, match="No positive label was resolved"):
            encoder.positive_column_index()

    def test_the_zero_one_convention_still_resolves(self):
        labels = pd.Series([0, 1] * 50)
        profile = TaskDetector().detect(labels)
        encoder = TargetLabelEncoder().fit(labels, profile)
        assert encoder.positive_column_index() == 1

    def test_a_label_that_is_not_a_class_is_refused(self, labels):
        with pytest.raises(PreprocessingError, match="not one of the target's classes"):
            TargetLabelEncoder(positive_label="cancelled").fit(labels)


class TestMulticlass:
    @pytest.fixture
    def labels(self) -> pd.Series:
        return pd.Series(["red", "blue", "green"] * 40)

    def test_all_classes_are_encoded(self, labels):
        encoded = TargetLabelEncoder().fit_transform(labels)
        assert set(np.unique(encoded)) == {0, 1, 2}

    def test_encoding_is_reversible(self, labels):
        encoder = TargetLabelEncoder()
        encoded = encoder.fit_transform(labels)
        assert list(encoder.inverse_transform(encoded)) == list(labels)

    def test_no_positive_class_is_invented(self, labels):
        encoder = TargetLabelEncoder().fit(labels)
        assert encoder.encoding.positive_label is None
        assert not encoder.encoding.positive_label_resolved

    def test_the_mapping_is_deterministic(self, labels):
        first = TargetLabelEncoder().fit(labels).encoding.mapping
        second = TargetLabelEncoder().fit(labels).encoding.mapping
        assert first == second == {"blue": 0, "green": 1, "red": 2}


class TestRegressionIsRefused:
    def test_a_regression_target_is_not_label_encoded(self):
        values = pd.Series([10.5, 22.7, 18.2, 9.9, 31.4] * 20)
        profile = TaskDetector().detect(values)
        assert profile.task_type is TaskType.REGRESSION
        with pytest.raises(UnsupportedTaskError, match="must not be label encoded"):
            TargetLabelEncoder().fit(values, profile)

    def test_the_refusal_explains_what_would_be_destroyed(self):
        profile = TargetProfile(task_type=TaskType.REGRESSION)
        with pytest.raises(UnsupportedTaskError, match="quantities"):
            TargetLabelEncoder().fit(pd.Series([1.5, 2.5]), profile)


class TestRejectedInput:
    def test_an_unseen_label_at_transform_is_refused(self):
        encoder = TargetLabelEncoder().fit(pd.Series(["a", "b"] * 30))
        with pytest.raises(PreprocessingError, match="not present in training"):
            encoder.transform(pd.Series(["c"]))

    def test_a_missing_label_is_refused(self):
        with pytest.raises(PreprocessingError, match="missing value"):
            TargetLabelEncoder().fit(pd.Series(["a", None, "b"]))

    def test_a_single_class_target_is_refused(self):
        with pytest.raises(PreprocessingError, match="at least two"):
            TargetLabelEncoder().fit(pd.Series(["only"] * 30))

    def test_a_dataframe_target_is_refused(self):
        with pytest.raises(PreprocessingError, match="one-dimensional"):
            TargetLabelEncoder().fit(pd.DataFrame({"y": [0, 1]}))

    def test_using_the_encoder_before_fitting_is_refused(self):
        encoder = TargetLabelEncoder()
        with pytest.raises(PreprocessingError, match="has not been fitted"):
            encoder.transform(pd.Series(["a"]))
        with pytest.raises(PreprocessingError, match="has not been fitted"):
            encoder.encoding


class TestSerialisation:
    def test_the_encoding_round_trips_through_json(self):
        encoder = TargetLabelEncoder(positive_label="churn").fit(
            pd.Series(["stay", "churn"] * 40)
        )
        payload = json.loads(json.dumps(encoder.encoding.to_dict()))
        assert payload["classes"] == ["churn", "stay"]
        assert payload["positive_label"] == "churn"
        assert payload["positive_label_encoded"] == 0

    def test_numpy_labels_serialise(self):
        encoder = TargetLabelEncoder().fit(pd.Series(np.array([0, 1, 1, 0], dtype="int64")))
        json.dumps(encoder.encoding.to_dict())
