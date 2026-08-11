"""Dataset and configuration identity.

A fingerprint is only worth having if its rules are exact, so these tests are the
specification: each one names a change and asserts whether identity survives it.
A rule that is documented but untested is a rule that will quietly stop holding.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core.exceptions import SerializationError
from aidatasetkit.evidence import (
    config_fingerprint,
    dataset_fingerprint,
    schema_fingerprint,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    index = np.arange(200)
    return pd.DataFrame(
        {
            "age": (20 + index % 41).astype("int64"),
            "score": (index % 37) * 1.5,
            "city": [["Riyadh", "Jeddah"][value % 2] for value in index],
            "active": (index % 2 == 0),
        }
    )


class TestWhatKeepsIdentity:
    def test_the_same_frame_twice(self, frame):
        assert dataset_fingerprint(frame) == dataset_fingerprint(frame)

    def test_a_copy_is_the_same_dataset(self, frame):
        assert dataset_fingerprint(frame) == dataset_fingerprint(frame.copy(deep=True))

    def test_the_index_is_addressing_not_data(self, frame):
        """Reading a CSV with or without index_col must not change what it is."""
        reindexed = frame.set_axis(np.arange(1000, 1000 + len(frame)))
        assert dataset_fingerprint(reindexed) == dataset_fingerprint(frame)

    def test_a_named_index_changes_nothing_either(self, frame):
        named = frame.copy()
        named.index = pd.Index(np.arange(len(frame)), name="row")
        assert dataset_fingerprint(named) == dataset_fingerprint(frame)


class TestWhatBreaksIdentity:
    def test_one_changed_cell(self, frame):
        changed = frame.copy()
        changed.loc[changed.index[0], "age"] = 999
        assert dataset_fingerprint(changed) != dataset_fingerprint(frame)

    def test_a_dtype_change_with_the_same_visible_values(self, frame):
        """1 and 1.0 print the same and are not the same measurement."""
        retyped = frame.astype({"age": "float64"})
        assert dataset_fingerprint(retyped) != dataset_fingerprint(frame)

    def test_column_order(self, frame):
        reordered = frame[["score", "age", "city", "active"]]
        assert dataset_fingerprint(reordered) != dataset_fingerprint(frame)

    def test_row_order(self, frame):
        shuffled = frame.iloc[::-1]
        assert dataset_fingerprint(shuffled) != dataset_fingerprint(frame)

    def test_a_renamed_column(self, frame):
        assert dataset_fingerprint(frame.rename(columns={"age": "Age"})) != (
            dataset_fingerprint(frame)
        )

    def test_a_dropped_row(self, frame):
        assert dataset_fingerprint(frame.iloc[:-1]) != dataset_fingerprint(frame)

    def test_a_dropped_column(self, frame):
        assert dataset_fingerprint(frame.drop(columns=["city"])) != (
            dataset_fingerprint(frame)
        )

    def test_a_missing_value_where_there_was_one(self, frame):
        gapped = frame.astype({"age": "float64"})
        marked = gapped.copy()
        marked.loc[marked.index[3], "age"] = np.nan
        assert dataset_fingerprint(marked) != dataset_fingerprint(gapped)


class TestSchemaFingerprintReadsNoValues:
    def test_two_frames_with_the_same_shape_share_a_schema(self, frame):
        changed = frame.copy()
        changed.loc[changed.index[0], "age"] = 999
        assert schema_fingerprint(changed) == schema_fingerprint(frame)

    def test_but_not_the_same_dataset(self, frame):
        changed = frame.copy()
        changed.loc[changed.index[0], "age"] = 999
        assert dataset_fingerprint(changed) != dataset_fingerprint(frame)

    def test_a_dtype_change_moves_the_schema(self, frame):
        assert schema_fingerprint(frame.astype({"age": "float64"})) != (
            schema_fingerprint(frame)
        )


class TestNonStringLabelsStaySeparate:
    def test_an_integer_label_is_not_its_text_form(self):
        values = [1.0, 2.0, 3.0]
        numeric = pd.DataFrame({0: values})
        textual = pd.DataFrame({"0": values})
        assert dataset_fingerprint(numeric) != dataset_fingerprint(textual)

    def test_mixed_labels_do_not_collide(self):
        frame = pd.DataFrame({0: [1.0, 2.0], "0": [1.0, 2.0]})
        assert len(frame.columns) == 2
        assert dataset_fingerprint(frame) != dataset_fingerprint(
            frame.rename(columns={0: "zero"})
        )


class TestChunkingDoesNotChangeTheAnswer:
    def test_a_frame_larger_than_one_chunk_hashes_consistently(self):
        """The chunk size is an implementation detail and must not be visible."""
        from aidatasetkit.evidence import fingerprint as module

        rows = 250_000
        frame = pd.DataFrame({"a": np.arange(rows) % 1000})
        original = module._CHUNK_ROWS
        try:
            module._CHUNK_ROWS = 100_000
            large = dataset_fingerprint(frame)
            module._CHUNK_ROWS = 7_919
            small = dataset_fingerprint(frame)
        finally:
            module._CHUNK_ROWS = original
        assert large == small

    def test_an_empty_frame_still_has_an_identity(self):
        empty = pd.DataFrame({"a": pd.Series(dtype="float64")})
        assert len(dataset_fingerprint(empty)) == 64


class TestFingerprintsSurviveAFreshInterpreter:
    """Hash randomisation must not reach identity."""

    @staticmethod
    def _run(seed: str) -> str:
        script = (
            "import numpy as np, pandas as pd;"
            "from aidatasetkit.evidence import dataset_fingerprint;"
            "index = np.arange(500);"
            "frame = pd.DataFrame({'a': index % 97 * 1.5,"
            " 'b': [f'c{i % 40}' for i in index],"
            " 'c': (index % 2 == 0)});"
            "print(dataset_fingerprint(frame))"
        )
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": ""},
        ).stdout.strip()

    @pytest.fixture(scope="class")
    @staticmethod
    def digests() -> set[str]:
        runner = TestFingerprintsSurviveAFreshInterpreter._run
        return {runner(seed) for seed in ("0", "1", "999")}

    def test_three_hash_seeds_agree(self, digests):
        assert len(digests) == 1

    def test_and_agree_with_this_process(self, digests):
        index = np.arange(500)
        frame = pd.DataFrame(
            {
                "a": index % 97 * 1.5,
                "b": [f"c{i % 40}" for i in index],
                "c": (index % 2 == 0),
            }
        )
        assert digests == {dataset_fingerprint(frame)}


class TestConfigFingerprint:
    def test_the_same_settings_agree(self):
        settings = {"target": "Churn", "task": "classification", "threshold": 0.5}
        assert config_fingerprint(settings) == config_fingerprint(dict(settings))

    def test_key_order_is_not_a_setting(self):
        assert config_fingerprint({"a": 1, "b": 2}) == config_fingerprint({"b": 2, "a": 1})

    def test_a_changed_value_changes_identity(self):
        assert config_fingerprint({"a": 1}) != config_fingerprint({"a": 2})

    def test_an_added_setting_changes_identity(self):
        assert config_fingerprint({"a": 1}) != config_fingerprint({"a": 1, "b": None})

    def test_nested_settings_are_included(self):
        left = {"preprocessing": {"numeric_imputation": "median"}}
        right = {"preprocessing": {"numeric_imputation": "mean"}}
        assert config_fingerprint(left) != config_fingerprint(right)

    def test_an_unserialisable_setting_is_refused_rather_than_hashed_as_text(self):
        """Hashing repr() would make identity depend on a memory address."""
        from sklearn.preprocessing import StandardScaler

        with pytest.raises(SerializationError):
            config_fingerprint({"scaler": StandardScaler()})


class TestDigestConstruction:
    def test_parts_cannot_be_confused_by_concatenation(self):
        from aidatasetkit.evidence.fingerprint import digest_of

        assert digest_of("ab", "c") != digest_of("a", "bc")

    def test_the_same_parts_agree(self):
        from aidatasetkit.evidence.fingerprint import digest_of

        assert digest_of("a", "b") == digest_of("a", "b")
