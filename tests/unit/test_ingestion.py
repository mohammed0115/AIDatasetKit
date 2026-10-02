"""G1-W1: every input is read correctly or refused -- never read into a wrong table.

Fixtures are written by each test into its own temporary directory, so every
byte a test depends on is visible in the test. Several tests are adversarial:
they pass only if a weaker implementation -- pandas' defaults, the extension
alone, a silent guess -- would have failed them, and some show that failure
directly.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core.exceptions import (
    AIDatasetKitError,
    AmbiguousDelimiterError,
    ConfigurationError,
    DuplicateHeadersError,
    EmptyDataError,
    EmptyInputError,
    EncodingError,
    IngestionError,
    InputNotFoundError,
    InvalidIngestionOptionsError,
    MalformedInputError,
    SchemaError,
    UnsupportedFormatError,
)
from aidatasetkit.ingestion import (
    SUPPORTED_DELIMITERS,
    DelimiterSource,
    LoadOptions,
    SourceKind,
    TableFormat,
    load_table,
    resolve_format,
)
from aidatasetkit.ingestion.delimited import SAMPLE_CHARS
from aidatasetkit.profiling import DataProfiler

PACKAGE = Path(__file__).resolve().parents[2] / "aidatasetkit" / "ingestion"


def write(tmp_path: Path, name: str, text: str, encoding: str = "utf-8") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


def columns(loaded) -> list:
    return list(loaded.frame.columns)


# --------------------------------------------------------------------------- #
# Delimiters
# --------------------------------------------------------------------------- #


class TestDelimiterDetection:
    @pytest.mark.parametrize(
        ("delimiter", "text"),
        [
            (",", "a,b,label\n1,x,0\n2,y,1\n"),
            (";", "a;b;label\n1;x;0\n2;y;1\n"),
            ("\t", "a\tb\tlabel\n1\tx\t0\n2\ty\t1\n"),
            ("|", "a|b|label\n1|x|0\n2|y|1\n"),
        ],
        ids=["comma", "semicolon", "tab", "pipe"],
    )
    def test_each_supported_delimiter_is_detected_in_a_csv(self, tmp_path, delimiter, text):
        loaded = load_table(write(tmp_path, "data.csv", text))
        assert columns(loaded) == ["a", "b", "label"]
        assert loaded.frame.shape == (2, 3)
        assert loaded.metadata.delimiter == delimiter
        assert loaded.metadata.delimiter_source is DelimiterSource.DETECTED

    def test_p0_2_reproduced_and_closed(self, tmp_path):
        """The audit's P0-2: pandas' defaults read this as one column, silently."""
        path = write(tmp_path, "semi.csv", "a;b;label\n1;x;0\n2;y;1\n")
        assert list(pd.read_csv(path).columns) == ["a;b;label"]  # the old reader
        assert columns(load_table(path)) == ["a", "b", "label"]  # the authority

    def test_a_tab_separated_file_named_csv_is_split_on_tabs(self, tmp_path):
        """Choosing by extension alone would read this as one column."""
        loaded = load_table(write(tmp_path, "tabbed.csv", "a\tb\n1\t2\n3\t4\n"))
        assert columns(loaded) == ["a", "b"]

    def test_a_delimiter_inside_quotes_is_a_value(self, tmp_path):
        loaded = load_table(write(tmp_path, "q.csv", 'name,notes\n"a;b",1\n"c,d",2\n'))
        assert columns(loaded) == ["name", "notes"]
        assert loaded.frame["name"].tolist() == ["a;b", "c,d"]

    def test_a_newline_inside_quotes_stays_in_the_value(self, tmp_path):
        loaded = load_table(write(tmp_path, "nl.csv", 'k,v\n1,"line1\nline2"\n2,x\n'))
        assert loaded.frame.shape == (2, 2)
        assert loaded.frame["v"].tolist() == ["line1\nline2", "x"]

    def test_blank_lines_are_ignored(self, tmp_path):
        loaded = load_table(write(tmp_path, "blank.csv", "a,b\n\n1,2\n\n3,4\n\n"))
        assert loaded.frame.shape == (2, 2)

    def test_a_line_of_spaces_in_a_two_column_table_is_malformed(self, tmp_path):
        with pytest.raises(MalformedInputError, match="fields"):
            load_table(write(tmp_path, "spaces.csv", "a,b\n1,2\n   \n3,4\n"))

    def test_a_single_column_is_accepted_and_said_so(self, tmp_path):
        loaded = load_table(write(tmp_path, "one.csv", "only\n1\n2\n3\n"))
        assert columns(loaded) == ["only"]
        assert loaded.metadata.delimiter is None
        assert loaded.metadata.delimiter_source is None
        assert any("single column" in note for note in loaded.metadata.warnings)

    def test_an_ambiguous_file_is_refused_not_guessed(self, tmp_path):
        with pytest.raises(AmbiguousDelimiterError) as caught:
            load_table(write(tmp_path, "amb.csv", "x,y;z\n1,2;3\n4,5;6\n"))
        message = str(caught.value)
        assert "','" in message and "';'" in message and "explicit" in message

    def test_an_explicit_delimiter_resolves_the_ambiguity(self, tmp_path):
        path = write(tmp_path, "amb.csv", "x,y;z\n1,2;3\n4,5;6\n")
        loaded = load_table(path, options=LoadOptions(delimiter=";"))
        assert columns(loaded) == ["x,y", "z"]
        assert loaded.metadata.delimiter_source is DelimiterSource.EXPLICIT

    def test_a_wrong_explicit_delimiter_is_refused(self, tmp_path):
        """Named ',' for a ';' file would read one column: refused, not obeyed."""
        path = write(tmp_path, "semi.csv", "a;b;label\n1;x;0\n2;y;1\n")
        with pytest.raises(MalformedInputError, match="consistently separated by ';'"):
            load_table(path, options=LoadOptions(delimiter=","))

    def test_an_explicit_delimiter_on_a_true_single_column_is_fine(self, tmp_path):
        loaded = load_table(write(tmp_path, "one.csv", "only\n1\n2\n"), options=LoadOptions(delimiter=";"))
        assert columns(loaded) == ["only"]

    def test_detection_reads_well_past_the_first_line(self, tmp_path):
        """A header with no delimiter-looking noise does not decide alone."""
        rows = "\n".join(f"{i};{i * 2}" for i in range(300))
        loaded = load_table(write(tmp_path, "long.csv", "left;right\n" + rows + "\n"))
        assert loaded.frame.shape == (300, 2)

    def test_a_file_larger_than_the_sample_is_read_whole(self, tmp_path):
        rows = "\n".join(f"{i},value-{i:08d}" for i in range(SAMPLE_CHARS // 10))
        path = write(tmp_path, "big.csv", "k,v\n" + rows + "\n")
        assert path.stat().st_size > SAMPLE_CHARS
        assert load_table(path).frame.shape == (SAMPLE_CHARS // 10, 2)

    def test_arabic_text_is_read_intact(self, tmp_path):
        loaded = load_table(write(tmp_path, "ar.csv", "اسم,مدينة\nأحمد,الخرطوم\nسارة,أم درمان\n"))
        assert columns(loaded) == ["اسم", "مدينة"]
        assert loaded.frame["مدينة"].tolist() == ["الخرطوم", "أم درمان"]

    def test_values_and_names_are_not_altered(self, tmp_path):
        text = 'id, spaced name ,"quoted"\n007," x ",1\n'
        loaded = load_table(write(tmp_path, "keep.csv", text))
        assert columns(loaded) == [value for value in pd.read_csv(tmp_path / "keep.csv").columns]

    def test_detection_is_deterministic(self, tmp_path):
        path = write(tmp_path, "d.csv", "a;b\n1;2\n3;4\n")
        assert load_table(path).metadata == load_table(path).metadata

    def test_no_header_numbers_the_columns(self, tmp_path):
        loaded = load_table(write(tmp_path, "nh.csv", "1,2\n3,4\n"), options=LoadOptions(header=False))
        assert columns(loaded) == [0, 1]
        assert loaded.frame.shape == (2, 2)
        assert loaded.metadata.header is False


class TestTsv:
    def test_a_tsv_is_tab_separated_by_format(self, tmp_path):
        loaded = load_table(write(tmp_path, "d.tsv", "a\tb\n1\t2\n"))
        assert columns(loaded) == ["a", "b"]
        assert loaded.metadata.format is TableFormat.TSV
        assert loaded.metadata.delimiter == "\t"
        assert loaded.metadata.delimiter_source is DelimiterSource.FORMAT

    def test_a_comma_file_named_tsv_is_refused(self, tmp_path):
        with pytest.raises(MalformedInputError, match="named .tsv"):
            load_table(write(tmp_path, "wrong.tsv", "a,b\n1,2\n3,4\n"))

    def test_a_single_column_tsv_is_accepted_and_said_so(self, tmp_path):
        loaded = load_table(write(tmp_path, "one.tsv", "only\n1\n2\n"))
        assert columns(loaded) == ["only"]
        assert loaded.metadata.warnings

    def test_an_inconsistent_tsv_is_malformed(self, tmp_path):
        with pytest.raises(MalformedInputError):
            load_table(write(tmp_path, "rag.tsv", "a\tb\n1\t2\t3\n"))


# --------------------------------------------------------------------------- #
# Encodings
# --------------------------------------------------------------------------- #


class TestEncodings:
    def test_utf8_is_the_default(self, tmp_path):
        loaded = load_table(write(tmp_path, "u.csv", "name,v\nJosé,1\n"))
        assert loaded.metadata.encoding == "utf-8"
        assert loaded.frame["name"].tolist() == ["José"]

    def test_a_utf8_byte_order_mark_is_honoured(self, tmp_path):
        loaded = load_table(write(tmp_path, "bom.csv", "a,b\n1,2\n", "utf-8-sig"))
        assert columns(loaded) == ["a", "b"]
        assert loaded.metadata.encoding == "utf-8-sig"
        assert any("byte-order mark" in note for note in loaded.metadata.warnings)

    def test_utf8_sig_may_be_named(self, tmp_path):
        path = write(tmp_path, "bom.csv", "a,b\n1,2\n", "utf-8-sig")
        loaded = load_table(path, options=LoadOptions(encoding="utf-8-sig"))
        assert columns(loaded) == ["a", "b"]

    def test_latin1_when_named(self, tmp_path):
        path = write(tmp_path, "l.csv", "name,v\nJosé,1\n", "latin-1")
        loaded = load_table(path, options=LoadOptions(encoding="latin-1"))
        assert loaded.frame["name"].tolist() == ["José"]
        assert loaded.metadata.encoding == "latin-1"

    def test_cp1256_arabic_when_named(self, tmp_path):
        path = write(tmp_path, "w.csv", "اسم;قيمة\nأحمد;1\n", "cp1256")
        loaded = load_table(path, options=LoadOptions(encoding="cp1256"))
        assert columns(loaded) == ["اسم", "قيمة"]
        assert loaded.frame.iloc[0, 0] == "أحمد"

    def test_the_wrong_encoding_is_a_structured_error(self, tmp_path):
        path = write(tmp_path, "w.csv", "اسم;قيمة\nأحمد;1\n", "cp1256")
        with pytest.raises(EncodingError, match="cannot be decoded as utf-8") as caught:
            load_table(path)
        # The public contract is the library's error, never the raw codec one.
        assert not isinstance(caught.value, UnicodeDecodeError)
        assert isinstance(caught.value, AIDatasetKitError)

    def test_an_invalid_byte_after_the_sample_is_still_caught(self, tmp_path):
        head = "k,v\n" + "\n".join(f"{i},x" for i in range(SAMPLE_CHARS // 4)) + "\n"
        path = tmp_path / "late.csv"
        path.write_bytes(head.encode() + b"9,\xff\xfe\n")
        with pytest.raises(EncodingError):
            load_table(path)

    @pytest.mark.parametrize("alias", ["utf8", "UTF-8", "latin1", "iso-8859-1"])
    def test_aliases_map_to_the_canonical_label(self, alias):
        assert LoadOptions(encoding=alias).encoding in ("utf-8", "latin-1")

    @pytest.mark.parametrize("unsupported", ["cp1252", "utf-16", "shift_jis", "no-such-codec"])
    def test_other_encodings_are_refused(self, unsupported):
        with pytest.raises(InvalidIngestionOptionsError, match="not supported"):
            LoadOptions(encoding=unsupported)


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


class TestRefusals:
    def test_zero_bytes(self, tmp_path):
        with pytest.raises(EmptyInputError, match="0 bytes"):
            load_table(write(tmp_path, "z.csv", ""))

    def test_whitespace_only(self, tmp_path):
        with pytest.raises(EmptyInputError, match="whitespace"):
            load_table(write(tmp_path, "w.csv", "  \n\n\t \n"))

    def test_header_only(self, tmp_path):
        with pytest.raises(EmptyInputError, match="no data rows"):
            load_table(write(tmp_path, "h.csv", "a,b,c\n"))

    def test_an_unclosed_quote(self, tmp_path):
        with pytest.raises(MalformedInputError, match="quot"):
            load_table(write(tmp_path, "q.csv", 'a,b\n"1,2\n3,4\n'))

    def test_too_many_fields(self, tmp_path):
        with pytest.raises(MalformedInputError, match="fields"):
            load_table(write(tmp_path, "m.csv", "a,b\n1,2\n3,4,5\n"))

    def test_too_few_fields(self, tmp_path):
        """pandas pads a short row with NaN; the contract says every row is complete."""
        path = write(tmp_path, "s.csv", "a,b,c\n1,2,3\n4,5\n")
        assert pd.read_csv(path).shape == (2, 3)  # pandas accepts it silently
        with pytest.raises(MalformedInputError, match="fields"):
            load_table(path)

    def test_a_short_row_after_the_sample_is_caught(self, tmp_path):
        rows = "\n".join(f"{i},x,y" for i in range(SAMPLE_CHARS // 6))
        path = write(tmp_path, "late.csv", "a,b,c\n" + rows + "\n1,2\n")
        with pytest.raises(MalformedInputError, match="record"):
            load_table(path)

    def test_binary_content_named_csv(self, tmp_path):
        path = tmp_path / "bin.csv"
        path.write_bytes(bytes(range(256)) * 4)
        with pytest.raises(MalformedInputError, match="NUL bytes"):
            load_table(path)

    def test_utf16_is_refused_as_binary(self, tmp_path):
        with pytest.raises(MalformedInputError, match="NUL bytes"):
            load_table(write(tmp_path, "u16.csv", "a,b\n1,2\n", "utf-16"))

    @pytest.mark.parametrize("name", ["data.xlsx", "data.xml", "data", "data.txt"])
    def test_unsupported_suffixes(self, tmp_path, name):
        path = write(tmp_path, name, "a,b\n1,2\n")
        with pytest.raises(UnsupportedFormatError, match="not a supported format"):
            load_table(path)

    def test_a_txt_file_of_valid_csv_is_still_refused(self, tmp_path):
        """.txt says nothing about being a table; reading it as CSV would be a guess."""
        with pytest.raises(UnsupportedFormatError):
            load_table(write(tmp_path, "d.txt", "a,b\n1,2\n"))

    def test_the_suffix_is_matched_case_insensitively(self, tmp_path):
        assert resolve_format(Path("DATA.CSV")) is TableFormat.CSV
        assert resolve_format(Path("x.TsV")) is TableFormat.TSV

    @pytest.mark.parametrize("delimiter", [":", " ", "ab", "", "\n"])
    def test_an_invalid_delimiter_option(self, delimiter):
        with pytest.raises(InvalidIngestionOptionsError):
            LoadOptions(delimiter=delimiter)

    def test_a_non_boolean_header_option(self):
        with pytest.raises(InvalidIngestionOptionsError):
            LoadOptions(header=1)

    @pytest.mark.parametrize("suffix", [".csv", ".tsv"])
    def test_duplicate_headers_before_pandas_renames_them(self, tmp_path, suffix):
        separator = "," if suffix == ".csv" else "\t"
        path = write(tmp_path, f"dup{suffix}", separator.join(["a", "a", "b"]) + "\n" + separator.join("123") + "\n")
        assert "a.1" in pd.read_csv(path, sep=separator).columns  # what pandas would do
        with pytest.raises(DuplicateHeadersError, match="duplicate column headers"):
            load_table(path)

    def test_a_missing_file(self, tmp_path):
        with pytest.raises(InputNotFoundError):
            load_table(tmp_path / "absent.csv")

    def test_a_directory(self, tmp_path):
        folder = tmp_path / "folder.csv"
        folder.mkdir()
        with pytest.raises(InputNotFoundError, match="not a regular file"):
            load_table(folder)

    def test_a_plain_string_is_never_guessed_at(self, tmp_path):
        write(tmp_path, "d.csv", "a,b\n1,2\n")
        with pytest.raises(UnsupportedFormatError, match="pathlib.Path"):
            load_table(str(tmp_path / "d.csv"))

    @pytest.mark.parametrize("source", [42, None, {"a": [1]}, b"a,b\n1,2"])
    def test_other_source_types(self, source):
        with pytest.raises(UnsupportedFormatError):
            load_table(source)


class TestTheErrorContract:
    @pytest.mark.parametrize(
        ("error", "also"),
        [
            (UnsupportedFormatError, IngestionError),
            (InvalidIngestionOptionsError, ConfigurationError),
            (EmptyInputError, EmptyDataError),
            (DuplicateHeadersError, SchemaError),
            (AmbiguousDelimiterError, MalformedInputError),
            (EncodingError, IngestionError),
            (InputNotFoundError, IngestionError),
        ],
    )
    def test_every_refusal_is_a_library_error_in_the_existing_categories(self, error, also):
        assert issubclass(error, IngestionError)
        assert issubclass(error, AIDatasetKitError)
        assert issubclass(error, also)

    def test_no_message_quotes_a_cell_value(self, tmp_path):
        secret = "aisha.al-otaibi@example.com"
        path = write(tmp_path, "s.csv", f"email,v\n{secret},1\n{secret},2,3\n")
        with pytest.raises(MalformedInputError) as caught:
            load_table(path)
        assert secret not in str(caught.value)

    def test_no_message_carries_the_absolute_path(self, tmp_path):
        with pytest.raises(IngestionError) as caught:
            load_table(write(tmp_path, "h.csv", "a,b\n"))
        assert str(tmp_path) not in str(caught.value)


# --------------------------------------------------------------------------- #
# In-memory sources
# --------------------------------------------------------------------------- #


class TestDataFrames:
    def test_the_frame_is_returned_as_is_and_described(self):
        frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        loaded = load_table(frame)
        assert loaded.frame is frame
        metadata = loaded.metadata
        assert metadata.source_kind is SourceKind.DATAFRAME
        assert (metadata.format, metadata.encoding, metadata.delimiter, metadata.header) == (None, None, None, None)
        assert (metadata.row_count, metadata.column_count) == (2, 2)
        assert metadata.memory_bytes == int(frame.memory_usage(deep=True, index=True).sum())

    def test_the_frame_is_not_modified(self):
        frame = pd.DataFrame({"a": [1, None], "b": ["x", "y"]}, index=[10, 20])
        before = frame.copy(deep=True)
        load_table(frame)
        pd.testing.assert_frame_equal(frame, before)

    @pytest.mark.parametrize("frame", [pd.DataFrame(), pd.DataFrame({"a": []}), pd.DataFrame(index=[0, 1])])
    def test_an_empty_frame(self, frame):
        with pytest.raises(EmptyInputError):
            load_table(frame)

    def test_duplicate_labels(self):
        frame = pd.DataFrame([[1, 2]], columns=["a", "a"])
        with pytest.raises(DuplicateHeadersError):
            load_table(frame)

    def test_file_options_do_not_apply(self):
        with pytest.raises(InvalidIngestionOptionsError):
            load_table(pd.DataFrame({"a": [1]}), options=LoadOptions(delimiter=";"))


class TestRecords:
    def test_records_become_a_frame(self):
        loaded = load_table([{"name": "A", "score": 10}, {"name": "B", "score": 20}])
        assert columns(loaded) == ["name", "score"]
        assert loaded.frame["score"].tolist() == [10, 20]
        assert loaded.metadata.source_kind is SourceKind.RECORDS
        assert loaded.metadata.delimiter is None and loaded.metadata.encoding is None

    def test_columns_follow_first_appearance(self):
        loaded = load_table([{"b": 1, "a": 2}, {"c": 3, "a": 4}])
        assert columns(loaded) == ["b", "a", "c"]

    def test_a_missing_key_is_a_missing_cell_and_counted(self):
        loaded = load_table([{"a": 1, "b": 2}, {"a": 3}, {"b": 4}])
        assert loaded.frame["b"].isna().tolist() == [False, True, False]
        assert loaded.frame["a"].isna().tolist() == [False, False, True]
        assert loaded.metadata.warnings == ("2 of 3 records lack at least one of the 2 keys; those cells are missing values.",)

    def test_uniform_records_raise_no_warning(self):
        assert load_table([{"a": 1}, {"a": 2}]).metadata.warnings == ()

    def test_a_tuple_of_records_works_too(self):
        assert load_table(({"a": 1}, {"a": 2})).frame.shape == (2, 1)

    def test_the_records_are_not_modified(self):
        records = [{"a": 1, "nested": [1, 2]}, {"a": 2}]
        before = copy.deepcopy(records)
        load_table(records)
        assert records == before

    def test_empty(self):
        with pytest.raises(EmptyInputError):
            load_table([])

    def test_every_record_empty(self):
        with pytest.raises(EmptyInputError):
            load_table([{}, {}])

    @pytest.mark.parametrize("records", [[1, 2, 3], ["a", "b"], [{"a": 1}, ["x"]], [(1, 2)]])
    def test_scalars_and_sequences_are_not_records(self, records):
        with pytest.raises(MalformedInputError, match="not a mapping"):
            load_table(records)

    def test_keys_must_be_strings(self):
        with pytest.raises(MalformedInputError, match="must be strings"):
            load_table([{1: "x"}])


# --------------------------------------------------------------------------- #
# Integration with the layers above, and the boundary
# --------------------------------------------------------------------------- #


class TestWhatIsLoadedIsWhatIsProfiled:
    def test_profiling_a_loaded_csv_equals_profiling_the_reference_frame(self, tmp_path):
        rng = np.random.default_rng(7)
        frame = pd.DataFrame({"x": rng.normal(0, 1, 50).round(4), "c": rng.choice(list("pqr"), 50)})
        path = tmp_path / "ref.csv"
        frame.to_csv(path, index=False)
        loaded = load_table(path)
        reference = pd.read_csv(path)
        pd.testing.assert_frame_equal(loaded.frame, reference)
        assert DataProfiler().profile(loaded.frame).to_dict() == DataProfiler().profile(reference).to_dict()

    def test_metadata_serialises_to_plain_json(self, tmp_path):
        import json

        payload = load_table(write(tmp_path, "d.csv", "a;b\n1;2\n")).metadata.to_dict()
        assert json.loads(json.dumps(payload, allow_nan=False)) == payload
        assert payload["delimiter"] == ";" and payload["format"] == "csv"
        assert str(tmp_path) not in json.dumps(payload)


class TestTheBoundary:
    """Ingestion reads what it is given and nothing else."""

    FORBIDDEN_IMPORTS = {"socket", "urllib", "http", "requests", "httpx", "pickle", "shelve", "marshal", "subprocess"}

    @pytest.mark.parametrize("module", sorted(p.name for p in PACKAGE.glob("*.py")))
    def test_no_network_no_deserialisation_no_eval(self, module):
        tree = ast.parse((PACKAGE / module).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not {alias.name.split(".")[0] for alias in node.names} & self.FORBIDDEN_IMPORTS
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in self.FORBIDDEN_IMPORTS
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec", "compile", "__import__"}

    def test_it_does_not_profile(self, monkeypatch, tmp_path):
        from aidatasetkit.profiling import profiler

        def refuse(*args, **kwargs):
            raise AssertionError("ingestion must not profile")

        monkeypatch.setattr(profiler.DataProfiler, "profile", refuse)
        load_table(write(tmp_path, "d.csv", "a,b\n1,2\n"))

    def test_it_writes_nothing(self, tmp_path):
        path = write(tmp_path, "d.csv", "a,b\n1,2\n")
        before = sorted(p.name for p in tmp_path.iterdir())
        load_table(path)
        assert sorted(p.name for p in tmp_path.iterdir()) == before

    def test_every_supported_delimiter_is_listed_once(self):
        assert len(set(SUPPORTED_DELIMITERS)) == len(SUPPORTED_DELIMITERS) == 4
