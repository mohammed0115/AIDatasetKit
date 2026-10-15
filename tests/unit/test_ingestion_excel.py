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

from aidatasetkit import __version__
from aidatasetkit.core.exceptions import (
    CellLimitError,
    ColumnLimitError,
    DuplicateHeadersError,
    EmptyInputError,
    FieldLengthLimitError,
    FileSizeLimitError,
    InvalidIngestionOptionsError,
    MalformedInputError,
    RowLimitError,
)
from aidatasetkit.evidence import ARTIFACT_SCHEMA_VERSION, PUBLICATION_SCHEMA_VERSION, AuditBuilder
from aidatasetkit.evidence.report import render_report
from aidatasetkit.ingestion import (
    IngestionLimits,
    LoadOptions,
    SourceKind,
    TableFormat,
    load_table,
    resolve_format,
)
from aidatasetkit.ingestion.excel import _over_field_length
from aidatasetkit.profiling import DataProfiler

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

    def test_a_blank_data_row_is_kept(self, tmp_path):
        frame = load_table(write_xlsx(tmp_path, [["a", "b"], [1, 2], [None, None], [3, 4]])).frame
        assert frame.shape == (3, 2)
        assert frame["a"].tolist()[:1] == [1]
        assert pd.isna(frame["a"].iloc[1]) and pd.isna(frame["b"].iloc[1])
        assert frame["a"].iloc[2] == 3


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

    @pytest.mark.parametrize(
        "options",
        [LoadOptions(encoding="latin-1"), LoadOptions(header=False), LoadOptions(delimiter="|")],
    )
    def test_text_options_are_refused(self, tmp_path, options):
        with pytest.raises(InvalidIngestionOptionsError, match="LoadOptions"):
            load_table(write_xlsx(tmp_path), options=options)

    def test_a_header_only_sheet_is_empty_input(self, tmp_path):
        with pytest.raises(EmptyInputError, match="no data rows"):
            load_table(write_xlsx(tmp_path, [["a", "b"]]))


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


# --------------------------------------------------------------------------- #
# G1-W7: exact worksheet selection, evidence, and selected-sheet limits.
# Not a certification. G1-24 stays pending later review.
# --------------------------------------------------------------------------- #


_EMPTY_RELS = (
    b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    b"</Relationships>"
)


def _book(path: Path, sheets: list[tuple[str, list, str]]) -> Path:
    wb = openpyxl.Workbook()
    first = True
    for title, rows, state in sheets:
        if first:
            ws = wb.active
            ws.title = title
            first = False
        else:
            ws = wb.create_sheet(title)
        for row in rows:
            ws.append(list(row))
        ws.sheet_state = state
    wb.save(path)
    return path


def _with_chartsheet(path: Path, rows, chart_name: str = "Plot", worksheet: str = "Data") -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = worksheet
    for row in rows:
        ws.append(list(row))
    wb.create_chartsheet(chart_name)
    wb.save(path)
    _patch_chart_rels(path)
    return path


def _chartsheets_only(path: Path, chart_name: str = "Plot") -> Path:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    wb.create_chartsheet(chart_name)
    wb.save(path)
    _patch_chart_rels(path)
    return path


def _patch_chart_rels(path: Path) -> None:
    import io

    out = io.BytesIO()
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(out, "w") as dest:
        names = set(source.namelist())
        for item in source.infolist():
            dest.writestr(item, source.read(item.filename))
        for name in list(names):
            if not name.startswith("xl/chartsheets/") or not name.endswith(".xml"):
                continue
            if "/_rels/" in name:
                continue
            rels = f"xl/chartsheets/_rels/{Path(name).name}.rels"
            if rels not in names:
                dest.writestr(rels, _EMPTY_RELS)
    path.write_bytes(out.getvalue())


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
        path = _book(tmp_path / "one.xlsx", [("Only", [["a"], [1]], "visible")])
        try:
            loaded = load_table(path)
        except Exception:
            raise AssertionError("default single-sheet behavior changes")
        assert list(loaded.frame.columns) == ["a"]
        assert loaded.selector.to_dict() == {"kind": "sheet", "name": "Only"}

    def test_several_worksheets_require_a_name(self, tmp_path):
        path = _book(
            tmp_path / "many.xlsx",
            [("First", [["a"], [1]], "visible"), ("Second", [["b"], [2]], "visible")],
        )
        try:
            load_table(path)
        except MalformedInputError as error:
            assert "pass sheet=" in str(error)
            return
        raise AssertionError("first sheet always selected")

    def test_an_exact_non_first_worksheet_is_read(self, tmp_path):
        path = _book(
            tmp_path / "named.xlsx",
            [("First", [["a"], [1]], "visible"), ("Data", [["b"], [7]], "visible")],
        )
        try:
            loaded = load_table(path, sheet="Data")
        except MalformedInputError:
            raise AssertionError("sheet argument ignored")
        assert loaded.selector.name == "Data", "sheet argument ignored"
        assert loaded.frame["b"].iloc[0] == 7

    def test_an_empty_sheet_name_is_refused(self, tmp_path):
        with pytest.raises(MalformedInputError, match="no sheet named"):
            load_table(write_xlsx(tmp_path), sheet="")

    def test_an_unknown_sheet_name_does_not_fall_back(self, tmp_path):
        try:
            load_table(write_xlsx(tmp_path), sheet="Missing")
        except MalformedInputError as error:
            assert "no sheet named" in str(error)
            return
        raise AssertionError("missing sheet silently falls back")

    def test_a_non_string_sheet_is_refused(self, tmp_path):
        path = write_xlsx(tmp_path)
        for value in (0, 1, 1.5, True):
            with pytest.raises(InvalidIngestionOptionsError, match="must name a worksheet"):
                load_table(path, sheet=value)

    def test_a_numeric_looking_sheet_name_is_a_name(self, tmp_path):
        path = _book(
            tmp_path / "index.xlsx",
            [("First", [["a"], [1]], "visible"), ("0", [["n"], [8]], "visible"), ("1", [["n"], [9]], "visible")],
        )
        loaded = load_table(path, sheet="0")
        assert loaded.selector.name == "0"
        assert loaded.frame["n"].iloc[0] == 8
        assert load_table(path, sheet="1").frame["n"].iloc[0] == 9

    def test_a_unicode_worksheet_name_is_exact(self, tmp_path):
        composed = "caf\u00e9"
        decomposed = "cafe\u0301"
        path = _book(tmp_path / "unicode.xlsx", [(composed, [["a"], [1]], "visible")])
        loaded = load_table(path, sheet=composed)
        assert loaded.selector.name == composed
        with pytest.raises(MalformedInputError, match="no sheet named"):
            load_table(path, sheet=decomposed)

    def test_an_arabic_worksheet_name_is_exact(self, tmp_path):
        title = "ورقة"
        path = _book(tmp_path / "arabic.xlsx", [(title, [["a"], [1]], "visible")])
        loaded = load_table(path)
        assert loaded.selector.name == title
        assert load_table(path, sheet=title).selector.name == title
        with pytest.raises(MalformedInputError):
            load_table(path, sheet="ورقه")

    def test_sheet_matching_is_case_sensitive(self, tmp_path):
        path = _book(tmp_path / "case.xlsx", [("Data", [["a"], [1]], "visible")])
        assert load_table(path, sheet="Data").frame["a"].iloc[0] == 1
        with pytest.raises(MalformedInputError, match="no sheet named"):
            load_table(path, sheet="data")
        with pytest.raises(MalformedInputError, match="no sheet named"):
            load_table(path, sheet="DATA")

    def test_sheet_matching_keeps_whitespace(self, tmp_path):
        path = _book(
            tmp_path / "space.xlsx",
            [("A B", [["a"], [1]], "visible"), ("A  B", [["a"], [2]], "visible")],
        )
        assert load_table(path, sheet="A  B").frame["a"].iloc[0] == 2
        with pytest.raises(MalformedInputError, match="no sheet named"):
            load_table(path, sheet="A B ")

    def test_a_hidden_worksheet_stays_eligible(self, tmp_path):
        path = _book(
            tmp_path / "hidden.xlsx",
            [("Data", [["a"], [1]], "visible"), ("Secret", [["a"], [5]], "hidden")],
        )
        try:
            load_table(path)
        except MalformedInputError as error:
            assert "pass sheet=" in str(error)
        else:
            raise AssertionError("hidden worksheet removed from eligibility")
        loaded = load_table(path, sheet="Secret")
        assert loaded.selector.name == "Secret"
        assert loaded.frame["a"].iloc[0] == 5

    def test_a_very_hidden_worksheet_stays_eligible(self, tmp_path):
        # openpyxl will not save a workbook whose only worksheet is hidden.
        # A very-hidden worksheet beside a visible one is a valid fixture, and
        # a patched workbook.xml can leave that very-hidden sheet as the only
        # eligible worksheet.
        mixed = _book(
            tmp_path / "mixed.xlsx",
            [("Data", [["a"], [1]], "visible"), ("Buried", [["a"], [4]], "veryHidden")],
        )
        with pytest.raises(MalformedInputError, match="pass sheet="):
            load_table(mixed)
        assert load_table(mixed, sheet="Buried").selector.name == "Buried"
        alone = _book(tmp_path / "buried.xlsx", [("Buried", [["a"], [4]], "visible")])
        text = alone.read_bytes()
        alone.write_bytes(text.replace(b'state="visible"', b'state="veryHidden"', 1))
        loaded = load_table(alone)
        assert loaded.selector.name == "Buried"
        assert loaded.frame["a"].iloc[0] == 4

    def test_a_chartsheet_name_is_refused(self, tmp_path, monkeypatch):
        path = _with_chartsheet(tmp_path / "chart.xlsx", [["a"], [1]])

        def scanned(*args, **kwargs):
            raise AssertionError("iter_rows on a chartsheet")

        monkeypatch.setattr("aidatasetkit.ingestion.excel._rows", scanned)
        with pytest.raises(MalformedInputError, match="chartsheet"):
            load_table(path, sheet="Plot")

    def test_a_worksheet_beside_chartsheets_is_selected(self, tmp_path):
        path = _with_chartsheet(tmp_path / "both.xlsx", [["a"], [3]])
        try:
            loaded = load_table(path)
        except Exception:
            raise AssertionError("chartsheet treated as worksheet")
        assert loaded.selector.name == "Data", "chartsheet treated as worksheet"
        assert loaded.frame["a"].iloc[0] == 3

    def test_a_workbook_with_only_chartsheets_is_empty(self, tmp_path):
        path = _chartsheets_only(tmp_path / "charts.xlsx")
        with pytest.raises(EmptyInputError, match="no worksheet"):
            load_table(path)


class TestSheetOptionIsolation:
    def test_sheet_is_refused_outside_xlsx(self, tmp_path):
        samples = {
            ".csv": "a\n1\n",
            ".tsv": "a\t1\n",
            ".json": "[]\n",
            ".jsonl": "{\"a\": 1}\n",
            ".parquet": "not-parquet",
            ".feather": "not-feather",
            ".sqlite": "not-sqlite",
        }
        for suffix, body in samples.items():
            path = tmp_path / f"sample{suffix}"
            path.write_text(body, encoding="utf-8")
            try:
                load_table(path, sheet="Data")
            except InvalidIngestionOptionsError as error:
                assert "xlsx" in str(error)
                continue
            raise AssertionError("sheet= accepted for non-XLSX input")
        for source in (pd.DataFrame({"a": [1]}), [{"a": 1}]):
            try:
                load_table(source, sheet="Data")
            except InvalidIngestionOptionsError:
                continue
            raise AssertionError("sheet= accepted for non-XLSX input")

    def test_sheet_is_refused_with_chunked_profile(self, tmp_path):
        from aidatasetkit.cli.main import main

        path = tmp_path / "rows.csv"
        path.write_text("a,churn\n1,0\n", encoding="utf-8")
        output = tmp_path / "out"
        code = main(["audit", str(path), "--chunked-profile", "--sheet", "Data", "--output", str(output)])
        assert code != 0
        assert not output.exists()


class TestSheetEvidence:
    def test_an_explicit_sheet_is_recorded(self, tmp_path):
        path = _book(
            tmp_path / "explicit.xlsx",
            [("First", [["a"], [1]], "visible"), ("Data", [["b"], [2]], "visible")],
        )
        loaded = load_table(path, sheet="Data")
        assert loaded.selector is not None and loaded.selector.to_dict() == {
            "kind": "sheet",
            "name": "Data",
        }, "explicit selector omitted from evidence"
        artifact = _artifact(path, sheet="Data")
        assert artifact.source_selector.to_dict() == {"kind": "sheet", "name": "Data"}
        assert artifact.semantic_dict()["source_selector"] == {"kind": "sheet", "name": "Data"}

    def test_an_automatic_sheet_is_recorded(self, tmp_path):
        path = _book(tmp_path / "auto.xlsx", [("Only", [["a"], [1]], "visible")])
        loaded = load_table(path)
        assert loaded.selector is not None and loaded.selector.to_dict() == {
            "kind": "sheet",
            "name": "Only",
        }, "automatic single sheet not recorded"

    def test_the_report_renders_a_sheet_selector(self, tmp_path):
        path = _book(tmp_path / "report.xlsx", [("Data", [["a"], [1]], "visible")])
        html = render_report(_artifact(path).to_dict())
        assert "<th>sheet</th><td>Data</td>" in html
        assert "<th>table</th>" not in html

    def test_the_selected_sheet_enters_the_config_fingerprint(self, tmp_path):
        path = _book(tmp_path / "finger.xlsx", [("Data", [["a"], [1]], "visible")])
        artifact = _artifact(path, sheet="Data")
        assert artifact.config.settings.get("source_selector") == {
            "kind": "sheet",
            "name": "Data",
        }, "selector omitted from config fingerprint"
        again = _artifact(path, sheet="Data", created_at="2030-05-05T00:00:00Z")
        assert artifact.config.fingerprint == again.config.fingerprint

    def test_two_sheet_names_change_the_config_fingerprint(self, tmp_path):
        path = _book(
            tmp_path / "two.xlsx",
            [("Keep", [["a"], [1]], "visible"), ("Other", [["a"], [2]], "visible")],
        )
        left = _artifact(path, sheet="Keep").config.fingerprint
        right = _artifact(path, sheet="Other").config.fingerprint
        assert left != right, "two sheets share one config fingerprint"

    def test_automatic_and_explicit_selection_share_identity(self, tmp_path):
        path = _book(tmp_path / "same.xlsx", [("Only", [["a"], [1]], "visible")])
        automatic = _artifact(path)
        explicit = _artifact(path, sheet="Only")
        assert automatic.source_selector.to_dict() == explicit.source_selector.to_dict()
        assert automatic.config.fingerprint == explicit.config.fingerprint

    def test_a_non_xlsx_selector_stays_null(self, tmp_path):
        path = tmp_path / "rows.csv"
        path.write_text("a\n1\n", encoding="utf-8")
        loaded = load_table(path)
        assert loaded.selector is None
        profile = DataProfiler().profile(loaded.frame)
        artifact = AuditBuilder(dataset_name="dataset").build(
            loaded.frame,
            profile=profile,
            ingestion=loaded.metadata,
            source_selector=loaded.selector,
            settings={"task_hint": None},
            created_at="2020-01-01T00:00:00Z",
        )
        assert artifact.source_selector is None
        assert artifact.semantic_dict()["source_selector"] is None
        assert "source_selector" not in artifact.config.settings
        html = render_report(artifact.to_dict())
        assert "<th>sheet</th>" not in html
        assert "<th>table</th>" not in html


class TestSheetLimits:
    def test_the_selected_sheet_row_limit_is_enforced(self, tmp_path):
        path = _book(tmp_path / "rows.xlsx", [("Data", [["a"], [1], [2]], "visible")])
        try:
            load_table(path, sheet="Data", limits=IngestionLimits(max_rows=1))
        except RowLimitError:
            return
        raise AssertionError("row limit ignored")

    def test_the_selected_sheet_column_limit_is_enforced(self, tmp_path):
        path = _book(tmp_path / "cols.xlsx", [("Data", [["a", "b"], [1, 2]], "visible")])
        with pytest.raises(ColumnLimitError):
            load_table(path, sheet="Data", limits=IngestionLimits(max_columns=1))

    def test_the_selected_sheet_cell_limit_is_enforced(self, tmp_path):
        path = _book(tmp_path / "cells.xlsx", [("Data", [["a", "b"], [1, 2]], "visible")])
        try:
            load_table(path, sheet="Data", limits=IngestionLimits(max_cells=3))
        except CellLimitError:
            return
        raise AssertionError("column or cell limit ignored")

    def test_an_oversized_textual_cell_is_refused(self, tmp_path):
        path = write_xlsx(tmp_path, [["a"], ["abcd"]])
        try:
            load_table(path, limits=IngestionLimits(max_field_length=3))
        except FieldLengthLimitError as error:
            assert error.limit_name == "max_field_length"
            return
        raise AssertionError("max_field_length ignored")

    def test_numbers_and_booleans_are_not_measured_as_text(self, tmp_path):
        path = write_xlsx(tmp_path, [["n", "f"], [1_000_000, True]])
        loaded = load_table(path, limits=IngestionLimits(max_field_length=3))
        assert loaded.frame["n"].iloc[0] == 1_000_000
        assert _over_field_length(b"abcd", 3) is True
        assert _over_field_length(bytearray(b"abcd"), 3) is True
        assert _over_field_length(1_000_000, 3) is False
        assert _over_field_length(True, 1) is False

    def test_field_length_is_refused_before_the_frame(self, tmp_path, monkeypatch):
        path = write_xlsx(tmp_path, [["a"], ["abcd"]])

        def published(*args, **kwargs):
            raise AssertionError("cells/DataFrame published before limit refusal")

        monkeypatch.setattr("aidatasetkit.ingestion.excel._frame", published)
        with pytest.raises(FieldLengthLimitError):
            load_table(path, limits=IngestionLimits(max_field_length=3))

    def test_an_unselected_worksheet_does_not_trip_logical_limits(self, tmp_path):
        path = _book(
            tmp_path / "wide.xlsx",
            [
                ("Keep", [["a"], [1]], "visible"),
                ("Big", [["a", "b", "c"], [1, 2, 3], [4, 5, 6], [7, 8, 9]], "visible"),
            ],
        )
        loaded = load_table(path, sheet="Keep", limits=IngestionLimits(max_rows=1, max_columns=1, max_cells=2))
        assert loaded.selector.name == "Keep"
        assert loaded.frame.shape == (1, 1)

    def test_a_refusal_closes_the_workbook_and_publishes_nothing(self, tmp_path, monkeypatch):
        from aidatasetkit.cli.main import main

        path = write_xlsx(tmp_path, [["a"], ["abcd"]])
        closed: list[object] = []
        real = openpyxl.workbook.workbook.Workbook.close

        def spy(self, *args, **kwargs):
            closed.append(self)
            return real(self, *args, **kwargs)

        monkeypatch.setattr(openpyxl.workbook.workbook.Workbook, "close", spy)
        with pytest.raises(FieldLengthLimitError):
            load_table(path, limits=IngestionLimits(max_field_length=3))
        assert closed, "workbook left open"

        output = tmp_path / "out"
        code = main(
            ["audit", str(write_xlsx(tmp_path, [["a"], [1], [2]], name="cli.xlsx")),
             "--max-rows", "1", "--output", str(output), "--fail-on", "never"]
        )
        assert code != 0
        assert not (output / "CURRENT").exists()


class TestSheetSecurity:
    def test_the_workbook_is_opened_read_only_and_data_only(self, tmp_path, monkeypatch):
        seen: dict = {}
        real = openpyxl.load_workbook

        def spy(path, **kwargs):
            seen.update(kwargs)
            return real(path, **kwargs)

        monkeypatch.setattr(openpyxl, "load_workbook", spy)
        load_table(write_xlsx(tmp_path))
        assert seen.get("read_only") is True
        assert seen.get("data_only") is True

    def test_formula_text_is_not_returned(self, tmp_path):
        path = tmp_path / "formula.xlsx"
        wb = openpyxl.Workbook()
        wb.active.append(["a"])
        wb.active["A2"] = "=1+1"
        wb.save(path)
        value = load_table(path).frame["a"].iloc[0]
        assert not (isinstance(value, str) and value.startswith("=")), "formula text is not evaluated"

    def test_a_zero_byte_workbook_is_malformed(self, tmp_path):
        path = tmp_path / "empty.xlsx"
        path.write_bytes(b"")
        with pytest.raises(MalformedInputError, match="not a readable"):
            load_table(path)


class TestSheetSchema:
    def test_schema_13_accepts_table_sheet_and_null_selectors(self, tmp_path):
        import sqlite3

        assert ARTIFACT_SCHEMA_VERSION == "1.5"
        assert PUBLICATION_SCHEMA_VERSION == "1.0"
        assert __version__ == "0.1.0a1"
        sheet = _artifact(_book(tmp_path / "one.xlsx", [("Data", [["a"], [1]], "visible")]))
        assert sheet.schema_version == "1.5"
        assert sheet.source_selector.to_dict() == {"kind": "sheet", "name": "Data"}
        csv_path = tmp_path / "rows.csv"
        csv_path.write_text("a\n1\n", encoding="utf-8")
        csv_loaded = load_table(csv_path)
        csv_artifact = AuditBuilder(dataset_name="dataset").build(
            csv_loaded.frame,
            profile=DataProfiler().profile(csv_loaded.frame),
            ingestion=csv_loaded.metadata,
            source_selector=csv_loaded.selector,
            settings={"task_hint": None},
            created_at="2020-01-01T00:00:00Z",
        )
        assert csv_artifact.source_selector is None
        database = tmp_path / "one.sqlite"
        connection = sqlite3.connect(database)
        connection.execute("CREATE TABLE people (n INTEGER)")
        connection.execute("INSERT INTO people VALUES (1)")
        connection.commit()
        connection.close()
        table = load_table(database)
        assert table.selector.to_dict() == {"kind": "table", "name": "people"}

    def test_openpyxl_is_not_a_core_dependency(self):
        text = Path(__file__).resolve().parents[2].joinpath("pyproject.toml").read_text(encoding="utf-8")
        core = text.split("[project.optional-dependencies]", 1)[0]
        assert "openpyxl" not in core, "openpyxl becomes a core dependency"
