"""G1-W4: an .xlsx file is one worksheet's table, or a structured refusal.

openpyxl is an optional dependency, so this whole module is skipped where the
``excel`` extra is not installed; the absence itself is tested in
``tests/integration/test_ingestion_without_extras.py``. Fixtures are written by
each test into its own temporary directory.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

openpyxl = pytest.importorskip("openpyxl", reason="the excel extra is not installed")

from aidatasetkit.core.exceptions import (
    CellLimitError,
    ColumnLimitError,
    DuplicateHeadersError,
    EmptyInputError,
    FileSizeLimitError,
    InvalidIngestionOptionsError,
    MalformedInputError,
    RowLimitError,
)
from aidatasetkit.ingestion import (
    IngestionLimits,
    LoadOptions,
    SourceKind,
    TableFormat,
    load_table,
    resolve_format,
)

ROWS = [["a", "b", "ok"], [1, "x", True], [2, "y", False]]


def write_xlsx(tmp_path: Path, rows=ROWS, name: str = "data.xlsx") -> Path:
    path = tmp_path / name
    wb = openpyxl.Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    wb.save(path)
    return path


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


class TestExcelReading:
    def test_a_single_sheet_round_trips(self, tmp_path):
        loaded = load_table(write_xlsx(tmp_path))
        assert loaded.frame.shape == (2, 3)
        assert list(loaded.frame.columns) == ["a", "b", "ok"]
        assert loaded.frame["a"].tolist() == [1, 2]
        assert loaded.frame["ok"].tolist() == [True, False]

    def test_metadata_describes_the_file(self, tmp_path):
        metadata = load_table(write_xlsx(tmp_path)).metadata
        assert metadata.source_kind is SourceKind.FILE
        assert metadata.format is TableFormat.XLSX
        assert metadata.encoding is None
        assert metadata.delimiter is None
        assert (metadata.row_count, metadata.column_count) == (2, 3)

    def test_the_suffix_is_matched(self, tmp_path):
        assert resolve_format(tmp_path / "data.xlsx") is TableFormat.XLSX

    def test_a_named_sheet_is_read(self, tmp_path):
        path = tmp_path / "named.xlsx"
        wb = openpyxl.Workbook()
        wb.active.title = "First"
        wb.active.append(["x"])
        wb.active.append([0])
        wb.create_sheet("Data").append(["a"])
        wb["Data"].append([42])
        wb.save(path)
        loaded = load_table(path, sheet="Data")
        assert list(loaded.frame.columns) == ["a"]
        assert loaded.frame["a"].iloc[0] == 42

    def test_blank_cells_read_as_missing(self, tmp_path):
        path = write_xlsx(tmp_path, [["a", "b"], [1, None]])
        assert pd.isna(load_table(path).frame["b"].iloc[0])


# --------------------------------------------------------------------------- #
# Structured refusals
# --------------------------------------------------------------------------- #


class TestExcelRefusals:
    def test_a_multi_sheet_workbook_is_ambiguous(self, tmp_path):
        path = tmp_path / "m.xlsx"
        wb = openpyxl.Workbook()
        wb.active.title = "A"
        wb.active.append(["x"])
        wb.active.append([1])
        wb.create_sheet("B").append(["y"])
        wb["B"].append([2])
        wb.save(path)
        with pytest.raises(MalformedInputError, match="2 sheets"):
            load_table(path)

    def test_a_macro_enabled_workbook_is_refused(self, tmp_path):
        path = write_xlsx(tmp_path)
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("xl/vbaProject.bin", b"\x00")
        with pytest.raises(MalformedInputError, match="VBA macros"):
            load_table(path)

    def test_a_corrupt_file_is_malformed_not_a_bare_error(self, tmp_path):
        path = tmp_path / "broken.xlsx"
        path.write_bytes(b"this is not a zip archive")
        with pytest.raises(MalformedInputError, match="not a readable"):
            load_table(path)

    def test_a_zip_without_a_workbook_part_is_malformed(self, tmp_path):
        path = tmp_path / "notxlsx.xlsx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("random.txt", "no workbook here")
        with pytest.raises(MalformedInputError, match="no workbook part"):
            load_table(path)

    def test_duplicate_headers_are_refused(self, tmp_path):
        path = write_xlsx(tmp_path, [["a", "a"], [1, 2]])
        with pytest.raises(DuplicateHeadersError, match="repeats the column label"):
            load_table(path)

    def test_an_empty_sheet_is_empty_input(self, tmp_path):
        path = tmp_path / "empty.xlsx"
        openpyxl.Workbook().save(path)
        with pytest.raises(EmptyInputError):
            load_table(path)

    def test_an_unknown_named_sheet_is_refused(self, tmp_path):
        with pytest.raises(MalformedInputError, match="no sheet named"):
            load_table(write_xlsx(tmp_path), sheet="Nope")

    def test_a_delimiter_option_is_refused(self, tmp_path):
        with pytest.raises(InvalidIngestionOptionsError, match="delimiter"):
            load_table(write_xlsx(tmp_path), options=LoadOptions(delimiter=","))


# --------------------------------------------------------------------------- #
# Resource limits, enforced from the declared dimensions before materializing
# --------------------------------------------------------------------------- #


class TestExcelLimits:
    @pytest.mark.parametrize(
        ("limit", "exact"), [("max_rows", 2), ("max_columns", 3), ("max_cells", 9)]
    )
    def test_exactly_at_each_limit_is_accepted_and_one_under_is_refused(
        self, tmp_path, limit, exact
    ):
        path = write_xlsx(tmp_path)
        assert load_table(path, limits=IngestionLimits(**{limit: exact})).frame.shape == (2, 3)
        with pytest.raises((RowLimitError, ColumnLimitError, CellLimitError)) as caught:
            load_table(path, limits=IngestionLimits(**{limit: exact - 1}))
        assert caught.value.limit_name == limit

    def test_an_over_limit_sheet_is_refused_before_its_cells_are_read(self, tmp_path, monkeypatch):
        path = write_xlsx(tmp_path)

        def cells_read(*args, **kwargs):
            raise AssertionError("an over-limit sheet was materialized")

        monkeypatch.setattr("openpyxl.worksheet._read_only.ReadOnlyWorksheet.iter_rows", cells_read)
        with pytest.raises(RowLimitError):
            load_table(path, limits=IngestionLimits(max_rows=1))

    def test_the_byte_preflight_applies_before_openpyxl_is_called(self, tmp_path, monkeypatch):
        path = write_xlsx(tmp_path)

        def reader_reached(*args, **kwargs):
            raise AssertionError("oversized input reached the excel reader")

        monkeypatch.setattr("aidatasetkit.ingestion.loader.read_excel", reader_reached)
        with pytest.raises(FileSizeLimitError):
            load_table(path, limits=IngestionLimits(max_source_bytes=path.stat().st_size - 1))
