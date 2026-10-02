"""G1-W3: a JSON or JSONL file is a table of flat records, or a structured refusal.

Fixtures are written by each test into its own temporary directory, so every
byte a test depends on is visible in the test. Several tests are adversarial:
they pass only if a weaker implementation -- ``json.loads`` defaults, the
extension alone, a silent guess -- would have failed them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from aidatasetkit.core.exceptions import (
    CellLimitError,
    ColumnLimitError,
    EmptyInputError,
    EncodingError,
    FileSizeLimitError,
    InvalidIngestionOptionsError,
    KeyLimitError,
    MalformedInputError,
    RecordLimitError,
)
from aidatasetkit.ingestion import (
    IngestionLimits,
    LoadOptions,
    SourceKind,
    TableFormat,
    load_table,
    resolve_format,
)

RECORDS = [{"a": 1, "b": "x", "ok": True}, {"a": 2, "b": None, "ok": False}]


def write(tmp_path: Path, name: str, text: str, encoding: str = "utf-8") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


def write_json(tmp_path: Path, name: str = "data.json") -> Path:
    return write(tmp_path, name, json.dumps(RECORDS))


def write_jsonl(tmp_path: Path, name: str = "data.jsonl") -> Path:
    return write(tmp_path, name, "".join(json.dumps(record) + "\n" for record in RECORDS))


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


class TestJsonReading:
    def test_a_json_array_of_objects_round_trips(self, tmp_path):
        loaded = load_table(write_json(tmp_path))
        assert loaded.frame.shape == (2, 3)
        assert list(loaded.frame.columns) == ["a", "b", "ok"]
        assert loaded.frame["a"].tolist() == [1, 2]
        assert loaded.frame["ok"].tolist() == [True, False]
        assert pd.isna(loaded.frame["b"].iloc[1])

    def test_the_same_records_from_python_give_the_same_frame(self, tmp_path):
        """One authority: a file of records and the records themselves agree."""
        from_file = load_table(write_json(tmp_path)).frame
        from_records = load_table(RECORDS).frame
        assert from_file.equals(from_records)

    def test_columns_keep_the_order_their_keys_first_appeared_in(self, tmp_path):
        path = write(tmp_path, "order.json", '[{"z": 1, "a": 2}, {"m": 3}]')
        assert list(load_table(path).frame.columns) == ["z", "a", "m"]

    def test_a_key_missing_from_a_record_is_a_missing_cell_with_a_warning(self, tmp_path):
        loaded = load_table(write(tmp_path, "sparse.json", '[{"a": 1, "b": 2}, {"a": 3}]'))
        assert pd.isna(loaded.frame["b"].iloc[1])
        assert loaded.metadata.warnings == (
            "1 of 2 records lack at least one of the 2 keys; those cells are missing values.",
        )

    def test_metadata_describes_the_file(self, tmp_path):
        loaded = load_table(write_json(tmp_path))
        metadata = loaded.metadata
        assert metadata.source_kind is SourceKind.FILE
        assert metadata.format is TableFormat.JSON
        assert metadata.encoding == "utf-8"
        assert metadata.delimiter is None and metadata.delimiter_source is None
        assert metadata.header is None
        assert (metadata.row_count, metadata.column_count) == (2, 3)
        assert metadata.memory_bytes > 0

    def test_a_utf8_bom_is_honoured_and_recorded(self, tmp_path):
        path = tmp_path / "bom.json"
        path.write_bytes(b'\xef\xbb\xbf' + json.dumps(RECORDS).encode("utf-8"))
        loaded = load_table(path)
        assert loaded.metadata.encoding == "utf-8-sig"
        assert "utf-8-sig" in loaded.metadata.warnings[0]
        assert loaded.frame.shape == (2, 3)

    def test_a_named_encoding_is_used(self, tmp_path):
        path = write(tmp_path, "latin.json", '[{"city": "São Paulo"}]', "latin-1")
        loaded = load_table(path, options=LoadOptions(encoding="latin-1"))
        assert loaded.frame["city"].iloc[0] == "São Paulo"
        assert loaded.metadata.encoding == "latin-1"

    @pytest.mark.parametrize("name", ["data.json", "DATA.JSON"])
    def test_the_json_suffix_is_matched_case_insensitively(self, tmp_path, name):
        assert resolve_format(tmp_path / name) is TableFormat.JSON

    def test_an_undecodable_file_is_an_encoding_error(self, tmp_path):
        path = write(tmp_path, "latin.json", '[{"city": "São Paulo"}]', "latin-1")
        with pytest.raises(EncodingError, match="cannot be decoded"):
            load_table(path)


class TestJsonlReading:
    def test_one_object_per_line_round_trips(self, tmp_path):
        loaded = load_table(write_jsonl(tmp_path))
        assert loaded.frame.shape == (2, 3)
        assert loaded.metadata.format is TableFormat.JSONL
        assert pd.isna(loaded.frame["b"].iloc[1])

    def test_ndjson_is_jsonl(self, tmp_path):
        path = write_jsonl(tmp_path, "data.ndjson")
        assert resolve_format(path) is TableFormat.JSONL
        assert load_table(path).frame.shape == (2, 3)

    def test_blank_lines_are_ignored(self, tmp_path):
        path = write(tmp_path, "blank.jsonl", '{"a": 1}\n\n{"a": 2}\n\n')
        loaded = load_table(path)
        assert loaded.frame["a"].tolist() == [1, 2]

    def test_a_line_of_whitespace_is_not_blank(self, tmp_path):
        path = write(tmp_path, "spaces.jsonl", '{"a": 1}\n   \n{"a": 2}\n')
        with pytest.raises(MalformedInputError, match="line 2"):
            load_table(path)


# --------------------------------------------------------------------------- #
# Structured refusals: never a guessed table
# --------------------------------------------------------------------------- #


class TestJsonShapeRefusals:
    @pytest.mark.parametrize(
        ("text", "kind"),
        [('{"a": 1}', "an object"), ("42", "a number"), ('"a,b"', "a string"), ("true", "a boolean")],
        ids=["object", "number", "string", "boolean"],
    )
    def test_the_top_level_must_be_an_array(self, tmp_path, text, kind):
        with pytest.raises(MalformedInputError, match=f"this document is {kind}"):
            load_table(write(tmp_path, "top.json", text))

    @pytest.mark.parametrize("text", ["[]", "  [ ]  "])
    def test_an_empty_array_is_empty_input(self, tmp_path, text):
        with pytest.raises(EmptyInputError, match="empty array"):
            load_table(write(tmp_path, "empty.json", text))

    @pytest.mark.parametrize("text", ["", "   \n  "])
    def test_an_empty_file_is_empty_input(self, tmp_path, text):
        with pytest.raises(EmptyInputError):
            load_table(write(tmp_path, "empty.json", text))

    def test_an_array_with_a_non_object_element_is_refused_with_its_position(self, tmp_path):
        path = write(tmp_path, "rows.json", '[{"a": 1}, [1, 2]]')
        with pytest.raises(MalformedInputError, match="record 1 is an array, not an object"):
            load_table(path)

    def test_a_nested_object_is_refused_naming_its_key(self, tmp_path):
        path = write(tmp_path, "nested.json", '[{"a": {"x": 1}}]')
        with pytest.raises(MalformedInputError, match="holds an object under 'a'"):
            load_table(path)

    def test_a_nested_array_is_refused_naming_its_key(self, tmp_path):
        path = write(tmp_path, "nested.json", '[{"a": [1, 2]}]')
        with pytest.raises(MalformedInputError, match="holds an array under 'a'"):
            load_table(path)

    def test_a_duplicate_key_is_refused_not_kept_last(self, tmp_path):
        """json.loads alone keeps the last value silently -- the quiet misread."""
        text = '[{"a": 1, "a": 2}]'
        assert json.loads(text) == [{"a": 2}]  # the weaker implementation
        with pytest.raises(MalformedInputError, match="repeats the key 'a'"):
            load_table(write(tmp_path, "dup.json", text))

    @pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
    def test_non_standard_constants_are_refused(self, tmp_path, constant):
        """Python's json accepts these; valid JSON does not have them."""
        assert json.loads(f"[{{\"a\": {constant}}}]")  # the weaker implementation
        with pytest.raises(MalformedInputError, match=constant.replace("-", r"\-")):
            load_table(write(tmp_path, "const.json", f'[{{"a": {constant}}}]'))

    def test_invalid_json_reports_line_and_column_not_content(self, tmp_path):
        sentinel = "SENTINEL_7f3a9c_do_not_leak"
        path = write(tmp_path, "broken.json", f'[{{"a": 1}}, {{"{sentinel}": ]]')
        with pytest.raises(MalformedInputError, match="line 1, column") as caught:
            load_table(path)
        assert sentinel not in str(caught.value)

    def test_binary_content_named_json(self, tmp_path):
        path = tmp_path / "bin.json"
        path.write_bytes(bytes(range(256)) * 4)
        with pytest.raises(MalformedInputError, match="NUL bytes"):
            load_table(path)


class TestJsonlRefusals:
    def test_a_malformed_line_names_its_number(self, tmp_path):
        path = write(tmp_path, "bad.jsonl", '{"a": 1}\nnot json\n{"a": 2}\n')
        with pytest.raises(MalformedInputError, match="line 2 of bad.jsonl"):
            load_table(path)

    def test_a_non_object_line_names_its_number(self, tmp_path):
        path = write(tmp_path, "bad.jsonl", '{"a": 1}\n[1, 2]\n')
        with pytest.raises(MalformedInputError, match="line 2 is an array, not an object"):
            load_table(path)

    def test_a_duplicate_key_names_its_line(self, tmp_path):
        path = write(tmp_path, "dup.jsonl", '{"a": 1, "a": 2}\n')
        with pytest.raises(MalformedInputError, match="line 1 .* repeats the key 'a'"):
            load_table(path)

    def test_an_empty_jsonl_file_is_empty_input(self, tmp_path):
        with pytest.raises(EmptyInputError):
            load_table(write(tmp_path, "empty.jsonl", "\n\n"))


# --------------------------------------------------------------------------- #
# Resource limits
# --------------------------------------------------------------------------- #


class TestJsonLimits:
    @pytest.mark.parametrize(
        ("limit", "exact"),
        [
            ("max_records", 2),
            ("max_keys_per_record", 3),
            ("max_columns", 3),
            ("max_cells", 6),
            # a=1, b="x", ok=True -> 1+1+4 chars; a=2, b=None, ok=False -> 1+4+5.
            ("max_record_chars", 16),
        ],
    )
    @pytest.mark.parametrize("writer", [write_json, write_jsonl], ids=["json", "jsonl"])
    def test_exactly_at_each_limit_is_accepted_and_one_under_is_refused(
        self, tmp_path, writer, limit, exact
    ):
        path = writer(tmp_path)
        assert load_table(path, limits=IngestionLimits(**{limit: exact})).frame.shape == (2, 3)
        with pytest.raises(
            (RecordLimitError, KeyLimitError, ColumnLimitError, CellLimitError)
        ) as caught:
            load_table(path, limits=IngestionLimits(**{limit: exact - 1}))
        assert caught.value.limit_name == limit

    def test_the_byte_preflight_applies_before_any_json_parsing(self, tmp_path, monkeypatch):
        path = write_json(tmp_path)

        def parser_reached(*args, **kwargs):
            raise AssertionError("oversized input reached parsing")

        monkeypatch.setattr("aidatasetkit.ingestion.loader.read_json_text", parser_reached)
        with pytest.raises(FileSizeLimitError):
            load_table(path, limits=IngestionLimits(max_source_bytes=path.stat().st_size - 1))

    def test_a_refusal_never_carries_a_rejected_value(self, tmp_path):
        sentinel = "SENTINEL_7f3a9c_do_not_leak"
        path = write(tmp_path, "rows.jsonl", f'{{"a": 1}}\n{{"a": "{sentinel}"}}\n')
        with pytest.raises(RecordLimitError) as caught:
            load_table(path, limits=IngestionLimits(max_records=1))
        exposed = " ".join([str(caught.value), repr(caught.value), *map(repr, vars(caught.value).values())])
        assert sentinel not in exposed, "RecordLimitError leaked a rejected value"
        assert str(tmp_path) not in exposed

    def test_a_jsonl_refusal_reads_no_line_past_the_decisive_one(self, tmp_path, monkeypatch):
        import aidatasetkit.ingestion.json_text as json_text

        path = write(
            tmp_path, "many.jsonl", "".join(json.dumps({"a": index}) + "\n" for index in range(1_000))
        )
        consumed = []
        real = json_text._iter_jsonl

        def counting(p, encoding):
            for label, record in real(p, encoding):
                consumed.append(label)
                yield label, record

        monkeypatch.setattr("aidatasetkit.ingestion.json_text._iter_jsonl", counting)
        with pytest.raises(RecordLimitError):
            load_table(path, limits=IngestionLimits(max_records=2))
        # The two permitted records and the one line that crossed the limit.
        assert len(consumed) == 3, f"parsing read {len(consumed)} lines after the limit was crossed"


class TestJsonOptions:
    def test_a_delimiter_option_is_refused_for_json(self, tmp_path):
        with pytest.raises(InvalidIngestionOptionsError, match="delimiter"):
            load_table(write_json(tmp_path), options=LoadOptions(delimiter=","))

    def test_header_false_is_refused_for_json(self, tmp_path):
        with pytest.raises(InvalidIngestionOptionsError, match="header"):
            load_table(write_json(tmp_path), options=LoadOptions(header=False))

    def test_a_delimiter_option_is_refused_for_jsonl(self, tmp_path):
        with pytest.raises(InvalidIngestionOptionsError, match="delimiter"):
            load_table(write_jsonl(tmp_path), options=LoadOptions(delimiter=";"))
