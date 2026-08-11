"""Canonical serialisation: what may enter an artifact, and what may not.

The refusals matter more than the conversions. An audit artifact that quietly
absorbed an estimator as ``repr(estimator)`` would look like a record and would
not be one, so every test below that asserts a raise is asserting the property
this module exists for.
"""

from __future__ import annotations

import datetime
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from aidatasetkit.core.exceptions import EvidenceError, SerializationError
from aidatasetkit.core.types import Severity
from aidatasetkit.evidence import canonical, canonical_json, label_token


class TestWhatMayEnterAnArtifact:
    @pytest.mark.parametrize(
        "value", [None, True, False, 0, -3, 1.5, "text", "", "unicode: مرحبا"]
    )
    def test_plain_scalars_pass_through(self, value):
        assert canonical(value) == value

    def test_enums_become_their_value(self):
        assert canonical(Severity.WARNING) == "warning"

    def test_nested_containers_are_walked(self):
        assert canonical({"a": [1, {"b": (2, 3)}]}) == {"a": [1, {"b": [2, 3]}]}

    def test_numpy_scalars_become_python_numbers(self):
        assert canonical(np.int64(7)) == 7
        assert canonical(np.float64(1.5)) == 1.5
        assert canonical(np.bool_(True)) is True

    def test_a_datetime_becomes_an_iso_string(self):
        moment = datetime.datetime(2025, 2, 1, 12, 30, tzinfo=datetime.timezone.utc)
        assert canonical(moment) == "2025-02-01T12:30:00+00:00"

    def test_sets_become_sorted_lists(self):
        assert canonical({"c", "a", "b"}) == ["a", "b", "c"]

    def test_mapping_keys_become_strings(self):
        assert canonical({1: "a"}) == {"1": "a"}


class TestWhatMayNot:
    """Each of these would be a leak, and each names what it found."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (DecisionTreeClassifier(), "estimator"),
            (StandardScaler(), "estimator"),
            (pd.Series([1, 2, 3]), "pandas"),
            (pd.DataFrame({"a": [1]}), "pandas"),
            (np.array([1, 2, 3]), "numpy array"),
        ],
    )
    def test_an_object_that_should_never_be_here_is_refused(self, value, expected):
        with pytest.raises(SerializationError, match=expected):
            canonical(value)

    def test_a_callable_is_refused(self):
        with pytest.raises(SerializationError, match="callable"):
            canonical(len)

    def test_a_lambda_is_refused(self):
        with pytest.raises(SerializationError, match="callable"):
            canonical(lambda x: x)

    def test_an_arbitrary_object_is_refused_rather_than_repr_ed(self):
        class Custom:
            def __repr__(self):
                return "<looks like data>"

        with pytest.raises(SerializationError, match="no canonical form"):
            canonical(Custom())

    def test_the_refusal_names_where_the_value_sat(self):
        with pytest.raises(SerializationError, match=r"\$\.model\.estimator"):
            canonical({"model": {"estimator": DecisionTreeClassifier()}})

    def test_an_estimator_nested_in_a_list_is_still_refused(self):
        with pytest.raises(SerializationError, match=r"\$\.plan\[1\]"):
            canonical({"plan": ["scale", StandardScaler()]})

    def test_infinity_is_refused_because_json_cannot_hold_it(self):
        with pytest.raises(SerializationError, match="infinity"):
            canonical({"ratio": float("inf")})

    def test_not_a_number_becomes_null_rather_than_invalid_json(self):
        """NaN is a missing measurement; JSON's spelling for that is null."""
        assert canonical({"ratio": float("nan")}) == {"ratio": None}

    def test_these_are_all_evidence_errors(self):
        with pytest.raises(EvidenceError):
            canonical(DecisionTreeClassifier())


class TestCanonicalJsonIsStable:
    @pytest.fixture
    def payload(self) -> dict:
        return {"z": 1, "a": {"n": [3, 2, 1], "m": "x"}, "k": True}

    def test_keys_are_sorted_regardless_of_insertion_order(self, payload):
        shuffled = {"k": True, "a": {"m": "x", "n": [3, 2, 1]}, "z": 1}
        assert canonical_json(payload) == canonical_json(shuffled)

    def test_list_order_is_preserved_because_it_is_meaning(self, payload):
        assert canonical_json(payload) != canonical_json(
            {**payload, "a": {"n": [1, 2, 3], "m": "x"}}
        )

    def test_the_output_parses_back(self, payload):
        assert json.loads(canonical_json(payload)) == json.loads(canonical_json(payload))

    def test_no_memory_address_can_appear(self):
        with pytest.raises(SerializationError):
            canonical_json({"scaler": StandardScaler()})


class TestColumnLabelsStaySeparable:
    """``0`` and ``"0"`` are two columns and must never become one."""

    def test_an_integer_label_and_its_text_form_differ(self):
        assert label_token(0) != label_token("0")

    def test_the_type_is_visible_in_the_token(self):
        assert label_token(0) == "int:0"
        assert label_token("0") == "str:0"

    @pytest.mark.parametrize("label", [0, 1.5, "age", True, None])
    def test_every_label_kind_produces_a_distinct_token(self, label):
        assert label_token(label).endswith(str(label))

    def test_tokens_across_mixed_labels_stay_unique(self):
        labels = [0, "0", 1, "1", 1.0, True]
        tokens = [label_token(label) for label in labels]
        # 1 and True share a text form and a type name differs, so the token
        # distinguishes them; 1 and 1.0 differ by type name.
        assert len(set(tokens)) == len({(type(x).__name__, str(x)) for x in labels})
