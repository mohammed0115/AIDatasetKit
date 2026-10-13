"""G1-W10: a local .xls workbook is one worksheet, or a structured refusal.

xlrd and xlwt are optional to the core install. This module is skipped where
either is missing. xlwt only writes fixtures.
"""

from __future__ import annotations

import datetime
import importlib.util
import struct
from pathlib import Path

import pandas as pd
import pytest

xlrd = pytest.importorskip("xlrd", reason="the excel extra is not installed")
xlwt = pytest.importorskip("xlwt", reason="xls fixtures need xlwt")

from aidatasetkit.cli.main import main
from aidatasetkit.core.exceptions import (
    CellLimitError,
    ColumnLimitError,
    EmptyInputError,
    FieldLengthLimitError,
    FileSizeLimitError,
    InvalidIngestionOptionsError,
    MalformedInputError,
    MissingDependencyError,
    RowLimitError,
    UnsupportedFormatError,
)
from aidatasetkit.evidence.builder import AuditBuilder
from aidatasetkit.ingestion import load_table
from aidatasetkit.ingestion.types import IngestionLimits, LoadOptions, TableFormat
from aidatasetkit.profiling import DataProfiler

_DATE = datetime.datetime(2020, 1, 2, 3, 4, 5)
_SECRET = "MACRO_SECRET_DO_NOT_PUBLISH"
_DATE_STYLE = xlwt.easyxf(num_format_str="YYYY-MM-DD HH:MM:SS")


def _sheet(book, title: str, rows: list[list], *, visibility: int = 0, dates_1904: bool = False):
    worksheet = book.add_sheet(title)
    worksheet.visibility = visibility
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            if isinstance(value, datetime.datetime):
                worksheet.write(row_index, column_index, value, _DATE_STYLE)
            elif value is None:
                continue
            else:
                worksheet.write(row_index, column_index, value)
    return worksheet


def write_xls(path: Path, rows: list[list] | None = None, *, dates_1904: bool = False) -> Path:
    """One sheet named Data. The default body is a small mixed row."""
    book = xlwt.Workbook()
    book.dates_1904 = dates_1904
    body = rows if rows is not None else [["a"], [1]]
    _sheet(book, "Data", body)
    path = Path(path)
    if path.suffix.lower() != ".xls":
        path = path / "data.xls" if path.is_dir() else path.with_suffix(".xls")
    book.save(path)
    return path


def write_book(path: Path, sheets: list[tuple]) -> Path:
    """sheets: (title, rows, visibility)."""
    book = xlwt.Workbook()
    for title, rows, visibility in sheets:
        _sheet(book, title, rows, visibility=visibility)
    book.save(path)
    return path


def _patch_formula(path: Path, row: int, column: int, result: bytes) -> None:
    data = bytearray(path.read_bytes())
    index = 0
    while True:
        found = data.find(b"\x06\x00", index)
        if found < 0 or found + 18 > len(data):
            raise AssertionError("formula record not found")
        record_row = int.from_bytes(data[found + 4 : found + 6], "little")
        record_column = int.from_bytes(data[found + 6 : found + 8], "little")
        length = int.from_bytes(data[found + 2 : found + 4], "little")
        if record_row == row and record_column == column and length >= 14:
            data[found + 10 : found + 18] = result
            path.write_bytes(data)
            return
        index = found + 1


def _artifact(path: Path, sheet: str | None = None, created_at: str = "2020-01-01T00:00:00Z"):
    loaded = load_table(path, sheet=sheet)
    profile = DataProfiler().profile(loaded.frame)
    return AuditBuilder(dataset_name="dataset").build(
        loaded.frame,
        profile=profile,
        ingestion=loaded.metadata,
        source_selector=loaded.selector,
        settings={"task_hint": None},
        created_at=created_at,
    )


class TestSheetSelection:
    def test_one_worksheet_is_selected_automatically(self, tmp_path):
        path = write_xls(tmp_path)
        loaded = load_table(path)
        assert loaded.metadata.format is TableFormat.XLS
        assert loaded.metadata.source_kind.value == "file"
        assert loaded.metadata.encoding is None
        assert loaded.metadata.delimiter is None
        assert loaded.selector.to_dict() == {"kind": "sheet", "name": "Data"}
        assert list(loaded.frame.columns) == ["a"]
        assert loaded.frame.iloc[0, 0] == 1

    def test_several_worksheets_require_a_name(self, tmp_path):
        path = write_book(
            tmp_path / "many.xls",
            [("First", [["a"], [1]], 0), ("Data", [["b"], [2]], 0)],
        )
        try:
            load_table(path)
        except MalformedInputError as exc:
            assert "sheet=" in str(exc)
            return
        raise AssertionError("first worksheet silently selected from multiple sheets")

    def test_an_exact_name_is_selected(self, tmp_path):
        path = write_book(
            tmp_path / "named.xls",
            [("First", [["a"], [1]], 0), ("Data", [["b"], [2]], 0)],
        )
        loaded = load_table(path, sheet="Data")
        assert loaded.frame.iloc[0, 0] == 2
        assert loaded.selector.name == "Data"

    def test_a_missing_name_is_refused(self, tmp_path):
        path = write_xls(tmp_path)
        with pytest.raises(MalformedInputError, match="no sheet named"):
            load_table(path, sheet="Missing")

    def test_matching_is_case_sensitive(self, tmp_path):
        path = write_xls(tmp_path)
        try:
            load_table(path, sheet="data")
        except MalformedInputError:
            return
        raise AssertionError("case-folded sheet matching")

    def test_matching_keeps_whitespace(self, tmp_path):
        path = write_xls(tmp_path)
        with pytest.raises(MalformedInputError, match="no sheet named"):
            load_table(path, sheet="Data ")

    def test_a_hidden_sheet_counts_as_eligible(self, tmp_path):
        path = write_book(
            tmp_path / "hidden.xls",
            [("Data", [["a"], [1]], 0), ("Hide", [["b"], [2]], 1)],
        )
        with pytest.raises(MalformedInputError, match="2 sheets"):
            load_table(path)
        loaded = load_table(path, sheet="Hide")
        assert loaded.frame.iloc[0, 0] == 2

    def test_a_very_hidden_sheet_can_be_the_only_table(self, tmp_path):
        path = write_book(tmp_path / "very.xls", [("Only", [["a"], [1]], 2)])
        loaded = load_table(path)
        assert loaded.selector.name == "Only"
        assert loaded.frame.iloc[0, 0] == 1

    def test_an_empty_sheet_is_empty_input(self, tmp_path):
        path = write_xls(tmp_path, [])
        with pytest.raises(EmptyInputError, match="no data"):
            load_table(path)

    def test_a_header_without_data_is_empty_input(self, tmp_path):
        path = write_xls(tmp_path, [["a"]])
        with pytest.raises(EmptyInputError, match="header row"):
            load_table(path)


class TestCellTypes:
    def test_text_number_boolean_blank_and_date(self, tmp_path):
        path = write_xls(
            tmp_path,
            [["t", "n", "b", "z", "d"], ["hello", 1.5, True, None, _DATE]],
        )
        frame = load_table(path).frame
        assert frame.iloc[0]["t"] == "hello"
        assert frame.iloc[0]["n"] == 1.5
        assert frame.iloc[0]["b"] == True
        assert frame.iloc[0]["z"] is None or pd.isna(frame.iloc[0]["z"])
        assert frame.iloc[0]["d"] == _DATE

    @pytest.mark.parametrize("dates_1904", [False, True])
    def test_both_excel_date_systems(self, tmp_path, dates_1904):
        path = write_xls(tmp_path / f"mode-{int(dates_1904)}.xls", [["d"], [_DATE]], dates_1904=dates_1904)
        assert load_table(path).frame.iloc[0, 0] == _DATE

    def test_a_cached_formula_result_is_the_value(self, tmp_path):
        path = write_xls(tmp_path, [["v"], [xlwt.Formula("1+1")]])
        _patch_formula(path, 1, 0, struct.pack("<d", 2.0))
        value = load_table(path).frame.iloc[0, 0]
        assert value == 2.0, "formula expression exposed or treated as value"
        assert "1+1" not in str(value)

    def test_an_excel_error_cell_is_refused_without_its_code(self, tmp_path):
        path = write_xls(tmp_path, [["v"], [xlwt.Formula("1/0")]])
        _patch_formula(path, 1, 0, bytes([2, 0, 7, 0, 0, 0, 0xFF, 0xFF]))
        try:
            load_table(path)
        except MalformedInputError as exc:
            text = str(exc)
            if "error cell" not in text or "7" in text or "DIV" in text:
                raise AssertionError("Excel error cell silently accepted") from None
            return
        raise AssertionError("Excel error cell silently accepted")

    def test_unselected_sheet_content_is_not_published(self, tmp_path):
        path = write_book(
            tmp_path / "macro.xls",
            [("Data", [["a"], [1]], 0), ("VBA", [["note"], [_SECRET]], 1)],
        )
        loaded = load_table(path, sheet="Data")
        published = str(loaded.frame.values) + str(loaded.metadata.to_dict())
        assert _SECRET not in published
        assert loaded.frame.iloc[0, 0] == 1


class TestLimits:
    def test_the_byte_limit_is_enforced_before_open(self, tmp_path, monkeypatch):
        path = write_xls(tmp_path)

        def opened(*args, **kwargs):
            raise AssertionError("byte limit checked after open")

        monkeypatch.setattr(xlrd, "open_workbook", opened)
        try:
            load_table(path, limits=IngestionLimits(max_source_bytes=path.stat().st_size - 1))
        except FileSizeLimitError:
            return
        except AssertionError:
            raise
        raise AssertionError("byte limit checked after open")

    def test_max_rows_is_enforced(self, tmp_path):
        path = write_xls(tmp_path, [["a"], [1], [2]])
        assert load_table(path, limits=IngestionLimits(max_rows=2)).frame.shape == (2, 1)
        try:
            load_table(path, limits=IngestionLimits(max_rows=1))
        except RowLimitError:
            return
        raise AssertionError("row limit ignored")

    def test_max_columns_is_enforced(self, tmp_path):
        path = write_xls(tmp_path, [["a", "b"], [1, 2]])
        try:
            load_table(path, limits=IngestionLimits(max_columns=1))
        except ColumnLimitError:
            return
        raise AssertionError("column limit ignored")

    def test_max_cells_is_enforced(self, tmp_path):
        path = write_xls(tmp_path, [["a", "b"], [1, 2]])
        try:
            load_table(path, limits=IngestionLimits(max_cells=3))
        except CellLimitError:
            return
        raise AssertionError("cell limit ignored")

    def test_max_field_length_is_enforced(self, tmp_path):
        path = write_xls(tmp_path, [["a"], ["abcd"]])
        try:
            load_table(path, limits=IngestionLimits(max_field_length=3))
        except FieldLengthLimitError as exc:
            assert "abcd" not in str(exc)
            return
        raise AssertionError("field length limit ignored")

    def test_an_unselected_sheet_does_not_trip_selected_limits(self, tmp_path):
        path = write_book(
            tmp_path / "wide.xls",
            [
                ("Keep", [["a"], [1]], 0),
                ("Big", [["a", "b", "c"], [1, 2, 3], [4, 5, 6], [7, 8, 9]], 0),
            ],
        )
        loaded = load_table(
            path,
            sheet="Keep",
            limits=IngestionLimits(max_rows=1, max_columns=1, max_cells=2),
        )
        assert loaded.selector.name == "Keep"
        assert loaded.frame.shape == (1, 1)

    def test_a_refusal_publishes_nothing(self, tmp_path):
        path = write_xls(tmp_path, [["a"], [1], [2]])
        output = tmp_path / "out"
        code = main(
            ["audit", str(path), "--target", "a", "--max-rows", "1", "--output", str(output), "--fail-on", "never"]
        )
        assert code != 0 and not (output / "CURRENT").exists(), "refusal publishes partial output"


class TestEvidence:
    def test_the_selected_sheet_is_recorded(self, tmp_path):
        path = write_book(
            tmp_path / "evidence.xls",
            [("First", [["a"], [1]], 0), ("Data", [["b"], [2]], 0)],
        )
        loaded = load_table(path, sheet="Data")
        assert loaded.selector is not None and loaded.selector.to_dict() == {
            "kind": "sheet",
            "name": "Data",
        }, "selected sheet omitted from evidence"
        artifact = _artifact(path, sheet="Data")
        assert artifact.source_selector.to_dict() == {"kind": "sheet", "name": "Data"}

    def test_the_selected_sheet_enters_the_config_fingerprint(self, tmp_path):
        path = write_book(
            tmp_path / "finger.xls",
            [("Keep", [["a"], [1]], 0), ("Other", [["a"], [2]], 0)],
        )
        artifact = _artifact(path, sheet="Keep")
        assert artifact.config.settings.get("source_selector") == {
            "kind": "sheet",
            "name": "Keep",
        }, "selected sheet omitted from config fingerprint"
        other = _artifact(path, sheet="Other").config.fingerprint
        assert artifact.config.fingerprint != other


class TestRefusals:
    def test_a_missing_xlrd_names_the_extra(self, tmp_path, monkeypatch):
        path = write_xls(tmp_path)
        real = importlib.util.find_spec

        def hidden(name, *args, **kwargs):
            if name == "xlrd":
                return None
            return real(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", hidden)
        try:
            load_table(path)
        except MissingDependencyError as exc:
            assert "aidatasetkit[excel]" in str(exc)
            return
        raise AssertionError("missing xlrd accepted")

    def test_xlsx_still_uses_openpyxl(self, tmp_path, monkeypatch):
        openpyxl = pytest.importorskip("openpyxl")
        path = tmp_path / "book.xlsx"
        workbook = openpyxl.Workbook()
        workbook.active.append(["a"])
        workbook.active.append([1])
        workbook.save(path)

        def opened(*args, **kwargs):
            raise AssertionError("xlsx dispatched to xlrd")

        monkeypatch.setattr(xlrd, "open_workbook", opened)
        try:
            loaded = load_table(path)
        except Exception:
            raise AssertionError("xlsx dispatched to xlrd") from None
        assert loaded.metadata.format is TableFormat.XLSX
        assert loaded.frame.iloc[0, 0] == 1

    def test_html_xml_and_zip_payloads_are_refused(self, tmp_path, monkeypatch):
        samples = {
            "page.xls": b"<html><table><tr><td>1</td></tr></table></html>",
            "markup.xls": b"<?xml version='1.0'?><Workbook></Workbook>",
            "zip.xls": b"PK\x03\x04" + b"not-a-workbook",
        }

        def opened(*args, **kwargs):
            raise AssertionError("disguised non-BIFF file accepted")

        monkeypatch.setattr(xlrd, "open_workbook", opened)
        for name, payload in samples.items():
            path = tmp_path / name
            path.write_bytes(payload)
            try:
                load_table(path)
            except MalformedInputError:
                continue
            except AssertionError:
                raise
            raise AssertionError("disguised non-BIFF file accepted")

    def test_xlsm_and_xlsb_suffixes_stay_unsupported(self, tmp_path):
        for name in ("book.xlsm", "book.xlsb"):
            path = tmp_path / name
            path.write_bytes(b"PK\x03\x04")
            with pytest.raises(UnsupportedFormatError):
                load_table(path)

    def test_encoding_and_delimiter_are_refused(self, tmp_path):
        path = write_xls(tmp_path)
        with pytest.raises(InvalidIngestionOptionsError, match="encoding"):
            load_table(path, options=LoadOptions(encoding="latin-1"))
        with pytest.raises(InvalidIngestionOptionsError, match="delimiter"):
            load_table(path, options=LoadOptions(delimiter=","))

    def test_xlrd_is_not_a_core_dependency(self):
        text = Path(__file__).resolve().parents[2].joinpath("pyproject.toml").read_text(encoding="utf-8")
        core = text.split("[project.optional-dependencies]", 1)[0]
        assert "xlrd" not in core, "xlrd becomes core dependency"


class TestCommandLine:
    def test_sheet_selects_the_named_worksheet(self, tmp_path):
        path = write_book(
            tmp_path / "cli.xls",
            [("Other", [["x"], [1]], 0), ("Data", [["a"], [1], [0]], 0)],
        )
        output = tmp_path / "out"
        code = main(
            [
                "audit",
                str(path),
                "--sheet",
                "Data",
                "--target",
                "a",
                "--fail-on",
                "never",
                "--output",
                str(output),
            ]
        )
        assert code == 0, "cli sheet selection failed"
        assert (output / "CURRENT").is_file()
