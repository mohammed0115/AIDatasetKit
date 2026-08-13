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


class TestTheDetectorAndTheEncoderCannotDisagree:
    """The closure that ends three silent-corruption routes.

    Both components answer "is this a quantity or a set of labels", and they used
    to answer it separately. The detector asked pandas for a dtype; the encoder
    looked for non-integral floats. Three inputs made them disagree, and each
    disagreement resolved the same way -- a measurement became a class index,
    with nothing raised at any point:

    ==========================================  ==============  ===============
    Target                                      Detector said   Encoder did
    ==========================================  ==============  ===============
    continuous floats, hint="classification"    classification  encoded 0..5
    object-dtype floats, no hint                classification  encoded 0..3
    60 distinct integers, no profile            **regression**  encoded 0..59
    ==========================================  ==============  ===============

    The third row is the one that cannot be argued with: the library's own
    detector called the column regression, and its own encoder turned it into
    class indices anyway. Both now read
    :func:`~aidatasetkit.core.schema.target_holds_quantities`.
    """

    #: Targets that hold quantities, whatever their dtype says.
    QUANTITIES = {
        "non-integral floats": pd.Series([1.5, 2.25, 3.75, 4.5, 5.125, 6.75] * 10),
        "object-dtype floats": pd.Series([1.25, 2.50, 3.75, 4.10] * 15, dtype=object),
        "60 distinct integers": pd.Series(np.arange(60) * 7),
        "60 distinct whole floats": pd.Series(np.arange(60) * 7.0),
        "prices with gaps": pd.Series([10.5, 22.7, np.nan, 18.2, 31.4] * 12),
    }

    #: Targets that are genuinely labels and must keep working untouched.
    LABELS = {
        "binary integers": pd.Series([0, 1] * 30),
        "multiclass integers": pd.Series([0, 1, 2, 3, 4] * 12),
        "string labels": pd.Series(["churn", "stay"] * 30),
        "object-dtype strings": pd.Series(["a", "b", "c"] * 20, dtype=object),
        "object-dtype whole floats": pd.Series([0.0, 1.0] * 30, dtype=object),
        "booleans": pd.Series([True, False] * 30),
    }

    # -- the three routes ------------------------------------------------- #

    def test_a_classification_hint_cannot_override_continuous_evidence(self):
        """Defect A: the hint was accepted on any column with two distinct values."""
        values = self.QUANTITIES["non-integral floats"]
        assert TaskDetector().detect(values).task_type is TaskType.REGRESSION
        with pytest.raises(UnsupportedTaskError, match="holds quantities"):
            TaskDetector().detect(values, hint="classification")

    def test_that_refusal_says_everything_a_reader_needs(self):
        with pytest.raises(UnsupportedTaskError) as raised:
            TaskDetector().detect(
                self.QUANTITIES["non-integral floats"], hint="classification"
            )
        message = str(raised.value)
        assert "Classification was requested" in message      # what was asked for
        assert "holds quantities" in message                  # what the data is
        assert "destroy" in message                           # what it would cost
        assert 'hint="regression"' in message                 # how to proceed
        assert "not reinterpreted for you" in message         # and what was not done

    def test_object_dtype_no_longer_hides_numeric_semantics(self):
        """Defect B: is_numeric_dtype answers False for a column of floats."""
        values = self.QUANTITIES["object-dtype floats"]
        assert values.dtype == object
        profile = TaskDetector().detect(values)
        assert profile.task_type is TaskType.REGRESSION
        assert "non-numeric" not in (profile.detection_note or "")

    def test_an_integer_target_the_detector_calls_regression_is_refused(self):
        """Defect C: the encoder's guard only ever looked for non-integral floats."""
        values = self.QUANTITIES["60 distinct integers"]
        assert TaskDetector().detect(values).task_type is TaskType.REGRESSION
        with pytest.raises(UnsupportedTaskError, match="must not be"):
            TargetLabelEncoder().fit(values)

    # -- the general contract --------------------------------------------- #

    @pytest.mark.parametrize("label", sorted(QUANTITIES))
    def test_no_quantity_is_encoded_without_a_profile(self, label):
        with pytest.raises(UnsupportedTaskError, match="must not be label encoded"):
            TargetLabelEncoder().fit(self.QUANTITIES[label])

    @pytest.mark.parametrize("label", sorted(QUANTITIES))
    def test_and_the_detector_agrees_it_is_regression(self, label):
        """The two answers side by side. This is the invariant that was missing."""
        assert TaskDetector().detect(self.QUANTITIES[label]).task_type is (
            TaskType.REGRESSION
        )

    @pytest.mark.parametrize("label", sorted(QUANTITIES))
    def test_the_refusal_is_a_project_error_not_a_backend_one(self, label):
        with pytest.raises(UnsupportedTaskError) as raised:
            TargetLabelEncoder().fit(self.QUANTITIES[label])
        assert type(raised.value).__module__.startswith("aidatasetkit")
        assert "sklearn" not in str(raised.value)
        assert len(str(raised.value)) > 80, "an error nobody can act on is not one"

    @pytest.mark.parametrize("label", sorted(QUANTITIES))
    def test_the_caller_target_is_unchanged_after_a_refusal(self, label):
        values = self.QUANTITIES[label]
        before = values.copy()
        with pytest.raises(UnsupportedTaskError):
            TargetLabelEncoder().fit(values)
        pd.testing.assert_series_equal(values, before)

    @pytest.mark.parametrize("label", sorted(LABELS))
    def test_every_genuine_label_target_still_encodes(self, label):
        """The half that must not have moved. Refusing these would be worse."""
        values = self.LABELS[label]
        encoded = TargetLabelEncoder().fit_transform(values)
        assert len(encoded) == len(values)
        assert set(np.unique(encoded)) == set(range(values.nunique()))

    @pytest.mark.parametrize("label", sorted(LABELS))
    def test_and_round_trips_back_to_what_went_in(self, label):
        values = self.LABELS[label]
        encoder = TargetLabelEncoder()
        decoded = encoder.inverse_transform(encoder.fit_transform(values))
        assert list(decoded) == list(values)

    def test_a_five_class_integer_target_is_still_accepted(self):
        """The ambiguous middle. The detector will not decide it without a hint;
        calling the encoder *is* the caller deciding it."""
        from aidatasetkit.core.exceptions import AmbiguousTaskError

        values = self.LABELS["multiclass integers"]
        with pytest.raises(AmbiguousTaskError):
            TaskDetector().detect(values)
        assert len(np.unique(TargetLabelEncoder().fit_transform(values))) == 5

    def test_a_regression_target_survives_the_path_that_does_not_encode(self):
        """The other half of the contract: on the regression path y is untouched."""
        from aidatasetkit.models import ModelFactory

        values = self.QUANTITIES["60 distinct integers"]
        profile = TaskDetector().detect(values)
        assert profile.task_type is TaskType.REGRESSION

        features = np.random.default_rng(3).normal(size=(len(values), 2))
        before = values.copy()
        ModelFactory.create("linear_regression", target=profile).fit(features, values)
        pd.testing.assert_series_equal(values, before)
        assert values.dtype == before.dtype


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
