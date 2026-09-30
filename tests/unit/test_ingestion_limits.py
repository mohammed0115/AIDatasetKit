from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aidatasetkit.cli.main import build_parser, main
from aidatasetkit.core.exceptions import (
    CellLimitError,
    ColumnLimitError,
    FileSizeLimitError,
    FieldLengthLimitError,
    InvalidIngestionOptionsError,
    KeyLimitError,
    RecordLimitError,
    RowLimitError,
)
from aidatasetkit.ingestion import IngestionLimits, LoadOptions, load_table
from aidatasetkit.facade import AIDataFacade


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class TestIngestionLimits:
    def test_defaults_are_finite_and_immutable(self):
        limits = IngestionLimits()
        assert limits.max_source_bytes == 64 * 1024 * 1024
        with pytest.raises(InvalidIngestionOptionsError):
            IngestionLimits(max_rows=0)
        with pytest.raises(InvalidIngestionOptionsError):
            IngestionLimits(max_columns=-1)
        with pytest.raises(InvalidIngestionOptionsError):
            IngestionLimits(max_cells=True)
        with pytest.raises(AttributeError):
            limits.max_rows = 1

    def test_file_bytes_are_rejected_before_parsing(self, tmp_path):
        path = write(tmp_path / "data.csv", "a,b\n1,2\n")
        assert load_table(path, limits=IngestionLimits(max_source_bytes=path.stat().st_size)).frame.shape == (1, 2)
        with pytest.raises(FileSizeLimitError) as caught:
            load_table(path, limits=IngestionLimits(max_source_bytes=path.stat().st_size - 1))
        assert caught.value.limit_name == "max_source_bytes"
        assert caught.value.input_kind == "file"
        assert str(path) not in str(caught.value)

    def test_rows_columns_cells_and_field_length_are_guarded(self, tmp_path):
        path = write(tmp_path / "data.csv", "a,b\n1,2\n3,4\n")
        assert load_table(path, limits=IngestionLimits(max_rows=2)).frame.shape == (2, 2)
        assert load_table(path, limits=IngestionLimits(max_columns=2)).frame.shape == (2, 2)
        assert load_table(path, limits=IngestionLimits(max_cells=4)).frame.shape == (2, 2)
        with pytest.raises(RowLimitError):
            load_table(path, limits=IngestionLimits(max_rows=1))
        with pytest.raises(ColumnLimitError):
            load_table(path, limits=IngestionLimits(max_columns=1))
        with pytest.raises(CellLimitError):
            load_table(path, limits=IngestionLimits(max_cells=3))
        with pytest.raises(FieldLengthLimitError):
            load_table(write(tmp_path / "long.csv", "a\n\u0623\u062d\u0645\u062f\n"), limits=IngestionLimits(max_field_length=2))

    def test_row_refusal_stops_before_a_later_malformed_record(self, tmp_path):
        path = write(tmp_path / "early.csv", 'a,b\n1,2\n3,4\n"unterminated\n')
        with pytest.raises(RowLimitError):
            load_table(path, options=LoadOptions(delimiter=","), limits=IngestionLimits(max_rows=1))

    def test_long_field_within_policy_is_not_rejected_by_csv_default(self, tmp_path):
        value = "x" * 200_000
        path = write(tmp_path / "long.csv", f"a\n{value}\n")
        assert load_table(path, limits=IngestionLimits(max_field_length=200_000)).frame.iloc[0, 0] == value

    def test_records_character_limit_and_large_integer_cell_guard(self):
        records = [{"a": "abcd"}]
        assert load_table(records, limits=IngestionLimits(max_record_chars=4)).frame.shape == (1, 1)
        with pytest.raises(CellLimitError):
            load_table(records, limits=IngestionLimits(max_record_chars=3))
        with pytest.raises(CellLimitError):
            load_table(
                [{"a": 1, "b": 2}] * 3,
                limits=IngestionLimits(max_cells=5, max_records=1_000_000_000_000_000_000),
            )

    def test_dataframe_is_not_copied_or_mutated(self):
        frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        before = frame.copy(deep=True)
        with pytest.raises(CellLimitError):
            load_table(frame, limits=IngestionLimits(max_cells=3))
        pd.testing.assert_frame_equal(frame, before)
        assert load_table(frame, limits=IngestionLimits(max_cells=4)).frame is frame

    def test_records_are_rejected_before_dataframe_materialization(self):
        records = [{"a": 1}, {"a": 2}]
        with pytest.raises(RecordLimitError):
            load_table(records, limits=IngestionLimits(max_records=1))
        with pytest.raises(KeyLimitError):
            load_table([{"a": 1, "b": 2}], limits=IngestionLimits(max_keys_per_record=1))
        assert records == [{"a": 1}, {"a": 2}]

    def test_facade_checks_limits_before_storing_the_frame(self):
        facade = AIDataFacade(target="label", task="classification")
        frame = pd.DataFrame({"a": [1, 2], "label": [0, 1]})
        with pytest.raises(CellLimitError):
            facade.load(frame, limits=IngestionLimits(max_cells=3))
        assert facade.stage.value == "empty"

    def test_facade_does_not_start_analysis_after_refusal(self, monkeypatch):
        facade = AIDataFacade(target="label", task="classification")

        def should_not_run(*args, **kwargs):
            raise AssertionError("analysis started after ingestion refusal")

        monkeypatch.setattr("aidatasetkit.facade.facade.DataProfiler.profile", should_not_run)
        monkeypatch.setattr("aidatasetkit.facade.facade.DataQualityInspector.inspect", should_not_run)
        frame = pd.DataFrame({"a": [1, 2], "label": [0, 1]})
        with pytest.raises(CellLimitError):
            facade.load(frame, limits=IngestionLimits(max_cells=3))

    def test_cli_help_and_invalid_limit(self, capsys):
        with pytest.raises(SystemExit) as help_exit:
            main(["audit", "--help"])
        assert help_exit.value.code == 0
        assert "--max-input-bytes" in capsys.readouterr().out
        with pytest.raises(SystemExit) as invalid_exit:
            main(["audit", "missing.csv", "--max-rows", "0"])
        assert invalid_exit.value.code == 1
        assert "positive integer" in capsys.readouterr().err

    def test_cli_defaults_match_library_defaults(self):
        args = build_parser().parse_args(["audit", "input.csv"])
        defaults = IngestionLimits()
        assert args.max_source_bytes == defaults.max_source_bytes
        assert args.max_rows == defaults.max_rows
        assert args.max_columns == defaults.max_columns
        assert args.max_cells == defaults.max_cells
        assert args.max_field_length == defaults.max_field_length


class TestResourceGovernanceGuarantees:
    """Each test here defends one G1-W2 guarantee against a named weakening.

    ``scripts/g1_w2_mutations.py`` applies each weakening to a copy of the tree
    and requires the matching test below to fail for that reason.
    """

    def test_every_default_is_the_documented_finite_integer(self):
        limits = IngestionLimits()
        assert {name: getattr(limits, name) for name in IngestionLimits.__slots__} == {
            "max_source_bytes": 67_108_864,
            "max_rows": 1_000_000,
            "max_columns": 1_000,
            "max_cells": 10_000_000,
            "max_field_length": 1_000_000,
            "max_records": 1_000_000,
            "max_keys_per_record": 1_000,
            "max_record_chars": 10_000_000,
        }
        for name in IngestionLimits.__slots__:
            value = getattr(limits, name)
            assert type(value) is int and value > 0, f"{name} default is not a finite positive int: {value!r}"

    def test_every_cli_default_equals_the_library_default(self):
        args = vars(build_parser().parse_args(["audit", "input.csv"]))
        defaults = IngestionLimits()
        exposed = ("max_source_bytes", "max_rows", "max_columns", "max_cells", "max_field_length")
        mismatches = {
            name: (args[name], getattr(defaults, name))
            for name in exposed
            if args[name] != getattr(defaults, name)
        }
        assert mismatches == {}, f"CLI default differs from IngestionLimits: {mismatches}"

    def test_oversized_file_never_reaches_a_parser(self, tmp_path, monkeypatch):
        path = write(tmp_path / "data.csv", "a,b\n1,2\n")

        def parser_reached(*args, **kwargs):
            raise AssertionError("oversized input reached parsing")

        monkeypatch.setattr("aidatasetkit.ingestion.loader.plan_delimited", parser_reached)
        monkeypatch.setattr("aidatasetkit.ingestion.loader.pd.read_csv", parser_reached)
        with pytest.raises(FileSizeLimitError) as caught:
            load_table(path, limits=IngestionLimits(max_source_bytes=path.stat().st_size - 1))
        assert caught.value.observed_value == path.stat().st_size

    @pytest.mark.parametrize(
        ("limit", "value"),
        [("max_rows", 3), ("max_columns", 2), ("max_cells", 6), ("max_field_length", 3)],
    )
    def test_a_csv_exactly_at_each_limit_is_accepted(self, tmp_path, limit, value):
        path = write(tmp_path / "data.csv", "a,b\n1,abc\n2,de\n3,f\n")
        assert load_table(path, limits=IngestionLimits(**{limit: value})).frame.shape == (3, 2)
        with pytest.raises((RowLimitError, ColumnLimitError, CellLimitError, FieldLengthLimitError)) as caught:
            load_table(path, limits=IngestionLimits(**{limit: value - 1}))
        assert caught.value.limit_name == limit

    def test_in_memory_sources_exactly_at_each_limit_are_accepted(self):
        frame = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert load_table(frame, limits=IngestionLimits(max_rows=3, max_columns=2, max_cells=6)).frame is frame
        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        exact = IngestionLimits(max_records=2, max_keys_per_record=2, max_cells=4, max_record_chars=4)
        assert load_table(records, limits=exact).frame.shape == (2, 2)

    def test_one_record_over_the_limit_is_refused(self):
        records = [{"a": index} for index in range(4)]
        with pytest.raises(RecordLimitError) as caught:
            load_table(records, limits=IngestionLimits(max_records=3))
        assert (caught.value.configured_limit, caught.value.observed_value) == (3, 4)

    def test_dataframe_over_the_cell_limit_is_refused_with_its_shape(self):
        frame = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(CellLimitError) as caught:
            load_table(frame, limits=IngestionLimits(max_cells=5))
        assert (caught.value.limit_name, caught.value.observed_value) == ("max_cells", 6)

    @pytest.mark.parametrize(
        ("rows", "columns", "max_cells", "exceeds"),
        [
            # 65,537 x 65,536 = 2**32 + 65,536: a 32-bit product wraps to 65,536.
            (65_537, 65_536, 10_000_000, True),
            # 2**53 + 1 cells: a float product rounds down to exactly 2**53.
            (2**53 + 1, 1, 2**53, True),
            (2**40, 2**40, 2**80, False),
            (2**40, 2**40, 2**80 - 1, True),
            (3, 2, 6, False),
            (3, 2, 5, True),
        ],
    )
    def test_cell_budget_is_exact_for_huge_logical_shapes(self, rows, columns, max_cells, exceeds):
        from aidatasetkit.ingestion.types import _cells_exceed

        assert _cells_exceed(rows, columns, max_cells) is exceeds

    def test_every_cell_check_goes_through_the_one_arithmetic_authority(self, tmp_path, monkeypatch):
        calls = []

        def always_exceeds(rows, columns, max_cells):
            calls.append((rows, columns))
            return True

        monkeypatch.setattr("aidatasetkit.ingestion.loader._cells_exceed", always_exceeds)
        monkeypatch.setattr("aidatasetkit.ingestion.delimited._cells_exceed", always_exceeds)
        with pytest.raises(CellLimitError):
            load_table(pd.DataFrame({"a": [1]}))
        with pytest.raises(CellLimitError):
            load_table([{"a": 1}])
        with pytest.raises(CellLimitError):
            load_table(write(tmp_path / "data.csv", "a\n1\n"))
        # The file path checks the budget on every record, the header (0 data rows) first.
        assert calls == [(1, 1), (1, 1), (0, 1)]

    def test_rejected_cell_values_never_appear_in_the_error(self, tmp_path):
        sentinel = "SENTINEL_7f3a9c_do_not_leak"
        cases = [
            (write(tmp_path / "rows.csv", f"a,b\n1,2\n{sentinel},4\n"), IngestionLimits(max_rows=1)),
            (write(tmp_path / "cells.csv", f"a,b\n1,2\n{sentinel},4\n"), IngestionLimits(max_cells=2)),
            (write(tmp_path / "field.csv", f"a,b\n1,{sentinel}\n"), IngestionLimits(max_field_length=8)),
            ([{"a": 1}, {"a": sentinel}], IngestionLimits(max_records=1)),
            ([{"a": sentinel}], IngestionLimits(max_record_chars=4)),
        ]
        for source, limits in cases:
            with pytest.raises((RowLimitError, CellLimitError, FieldLengthLimitError, RecordLimitError)) as caught:
                load_table(source, limits=limits)
            error = caught.value
            exposed = " ".join([str(error), repr(error), *map(repr, vars(error).values())])
            assert sentinel not in exposed, f"{type(error).__name__} leaked a rejected value"
            assert str(tmp_path) not in exposed

    def test_row_refusal_reads_no_record_past_the_decisive_one(self, tmp_path, monkeypatch):
        import csv as real_csv
        import io

        path = write(tmp_path / "many.csv", "a,b\n" + "".join(f"{i},{i}\n" for i in range(1_000)))
        consumed = []
        reader = real_csv.reader

        def counting_reader(source, *args, **kwargs):
            inner = reader(source, *args, **kwargs)
            if isinstance(source, io.StringIO):  # the detection sample, not the file
                return inner

            def rows():
                for record in inner:
                    consumed.append(record)
                    yield record

            return rows()

        monkeypatch.setattr("aidatasetkit.ingestion.delimited.csv.reader", counting_reader)
        with pytest.raises(RowLimitError):
            load_table(path, options=LoadOptions(delimiter=","), limits=IngestionLimits(max_rows=2))
        # The header, two permitted rows, and the one row that crossed the limit.
        assert len(consumed) == 4, f"validation read {len(consumed)} records after the limit was crossed"

    def test_disabling_the_field_limit_reads_the_file(self, tmp_path):
        # None is the documented way to disable one limit; it must not overflow
        # csv.field_size_limit, which takes a 32-bit C long on Windows.
        path = write(tmp_path / "data.csv", "a,b\n1," + "x" * 200_000 + "\n")
        assert load_table(path, limits=IngestionLimits(max_field_length=None)).frame.shape == (1, 2)

    def test_oversized_field_is_refused_before_pandas(self, tmp_path, monkeypatch):
        path = write(tmp_path / "long.csv", "a,b\n1," + "x" * 50 + "\n")

        def pandas_reached(*args, **kwargs):
            raise AssertionError("oversized field reached pandas")

        monkeypatch.setattr("aidatasetkit.ingestion.loader.pd.read_csv", pandas_reached)
        with pytest.raises(FieldLengthLimitError) as caught:
            load_table(path, limits=IngestionLimits(max_field_length=10))
        assert caught.value.limit_name == "max_field_length"

    def test_dataframe_is_returned_itself_with_order_and_dtypes(self, monkeypatch):
        frame = pd.DataFrame({"z": [1, 2], "a": ["x", "y"], "m": [0.5, 1.5]})
        dtypes = frame.dtypes.copy()

        def copied(*args, **kwargs):
            raise AssertionError("DataFrame copied during validation")

        monkeypatch.setattr(pd.DataFrame, "copy", copied)
        loaded = load_table(frame).frame
        assert loaded is frame
        assert list(loaded.columns) == ["z", "a", "m"]
        assert loaded.dtypes.equals(dtypes)

    def test_facade_refuses_before_target_detection(self, monkeypatch):
        facade = AIDataFacade(target="label", task="classification")

        def should_not_run(*args, **kwargs):
            raise AssertionError("target detection ran before the resource refusal")

        monkeypatch.setattr("aidatasetkit.facade.facade.TaskDetector.detect", should_not_run)
        frame = pd.DataFrame({"a": [1, 2], "label": [0, 1]})
        with pytest.raises(CellLimitError):
            facade.load(frame, limits=IngestionLimits(max_cells=3))
        assert facade.stage.value == "empty"
