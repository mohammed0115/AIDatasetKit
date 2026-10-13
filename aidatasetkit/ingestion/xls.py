"""Reading legacy Excel workbooks (.xls): one worksheet, through xlrd.

xlrd is an optional dependency on the ``excel`` extra, pinned to the 2.x line.
Importing this module never imports it. A missing install is a
:class:`MissingDependencyError` naming the extra, not a traceback.

``.xlsx`` stays on openpyxl. This module is not used for it.

The file must start with the BIFF compound-document signature. A ZIP archive
(``.xlsx``, ``.xlsm``, ``.xlsb``) or an HTML or XML table renamed to ``.xls``
is refused before xlrd opens it. xlrd 2.x does not evaluate formulas and does
not return formula text; only the cached result is kept. A missing cached
result is an empty cell. An Excel error result is refused without the code or
the formula. Macros are not executed, extracted, or described. xlrd ignores
VBA streams; that is not a scan and not a safety verdict.

Eligible relations are the worksheets xlrd lists, including hidden and very
hidden ones. Chartsheets, drawings, and defined names are not worksheets and
are not tables. More than one worksheet requires ``sheet=``. The match is
exact. One worksheet is selected automatically. The chosen title is
``source_selector`` kind ``sheet``.

The first row is the header, same as ``.xlsx``. The file-size budget is
applied by the loader before this function opens the workbook. Row, column,
cell, and field-length budgets apply to the selected worksheet only, before a
DataFrame is built. xlrd may parse that worksheet inside the accepted byte
budget; this is not a streaming reader.
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from typing import Any

import pandas as pd

from aidatasetkit.core.exceptions import (
    CellLimitError,
    ColumnLimitError,
    EmptyInputError,
    FieldLengthLimitError,
    InvalidIngestionOptionsError,
    MalformedInputError,
    MissingDependencyError,
    RowLimitError,
)
from aidatasetkit.ingestion.excel import _frame, _reject_long_fields
from aidatasetkit.ingestion.types import (
    IngestionLimits,
    LoadOptions,
    SourceKind,
    SourceSelector,
    TableFormat,
    _cells_exceed,
)

__all__ = ["read_xls"]

_OLE_SIGNATURE = bytes.fromhex("D0CF11E0A1B11AE1")
_ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_MARKUP_PREFIXES = (b"<html", b"<!doctype", b"<?xml", b"<workbook", b"<table")


def read_xls(
    path: Path,
    fmt: TableFormat,
    options: LoadOptions,
    limits: IngestionLimits,
    sheet: "str | None" = None,
) -> tuple[pd.DataFrame, SourceSelector]:
    """The frame held by one worksheet of an .xls file, and which worksheet."""
    if not isinstance(sheet, str) and sheet is not None:
        raise InvalidIngestionOptionsError(
            f"sheet must name a worksheet, got {type(sheet).__name__}."
        )
    if not options.is_default:
        raise InvalidIngestionOptionsError(
            "LoadOptions (encoding, delimiter, header) apply to text files; "
            f"{path.name} is an Excel file, which describes itself."
        )
    _reject_disguised(path)
    _require_xlrd()
    book = None
    try:
        book = _open_book(path)
        chosen = _choose_sheet(path, book, sheet)
        try:
            _check_dimensions(path, chosen, limits)
            rows = _materialize(path, book, chosen)
        except (RowLimitError, ColumnLimitError, CellLimitError, FieldLengthLimitError):
            raise  # xls refusal publishes nothing
        title = chosen.name
    finally:
        if book is not None:
            book.release_resources()
    _reject_long_fields(path, rows, limits)  # selected xls cells
    return _frame(path, rows), SourceSelector(kind="sheet", name=title)  # xls sheet evidence


def _require_xlrd() -> None:
    """Import is deferred. Absence names the extra instead of raising ImportError."""
    try:
        spec = importlib.util.find_spec("xlrd")
    except ImportError:
        spec = None
    if spec is None:
        raise MissingDependencyError(
            "Reading .xls files needs xlrd, which is not installed. It is "
            "an optional dependency: pip install 'aidatasetkit[excel]'."
        )


def _reject_disguised(path: Path) -> None:
    """Refuse a renamed ZIP, HTML, or XML file before xlrd treats it as BIFF."""
    with path.open("rb") as handle:
        head = handle.read(512)
    if head.startswith(_ZIP_SIGNATURES):
        raise MalformedInputError(
            f"{path.name} is not a BIFF .xls workbook (ZIP archive)."
        )
    sample = head.lstrip(b"\xef\xbb\xbf \t\r\n\x0c")
    if sample[:64].lower().startswith(_MARKUP_PREFIXES):
        raise MalformedInputError(
            f"{path.name} is not a BIFF .xls workbook (markup)."
        )
    if not head.startswith(_OLE_SIGNATURE):
        raise MalformedInputError(f"{path.name} is not a BIFF .xls workbook.")


def _open_book(path: Path):
    """The workbook. formatting_info distinguishes date cells from numbers."""
    import xlrd

    try:
        return xlrd.open_workbook(
            filename=str(path),
            formatting_info=True,
            on_demand=True,
            logfile=io.StringIO(),
        )
    except Exception as error:
        raise MalformedInputError(
            f"{path.name} is not a readable .xls workbook ({type(error).__name__})."
        ) from None


def _choose_sheet(path: Path, book, sheet: "str | None"):
    """The one worksheet to read. Hidden worksheets stay eligible."""
    names = list(book.sheet_names())
    if sheet is not None:
        for title in names:
            if title == sheet:
                return book.sheet_by_name(title)
        raise MalformedInputError(
            f"{path.name} has no sheet named {sheet!r}. Sheets: {', '.join(names)}."
        )
    if len(names) == 0:
        raise EmptyInputError(
            f"{path.name} has no worksheet; there is no table to load."
        )
    if len(names) == 1:
        return book.sheet_by_name(names[0])
    raise MalformedInputError(
        f"{path.name} has {len(names)} sheets ({', '.join(names)}). Choosing one "
        "would be a guess; pass sheet= to name it."
    )


def _check_dimensions(path: Path, sheet, limits: IngestionLimits) -> None:
    """Selected-sheet shape only. An unselected sheet is not measured."""
    rows = sheet.nrows
    columns = sheet.ncols
    data_rows = rows - 1 if rows else 0
    if rows == 0 or columns == 0:
        raise EmptyInputError(
            f"{path.name} has no data; there is no table to load."
        )
    if rows == 1:
        raise EmptyInputError(
            f"{path.name} has a header row and no data rows; there is nothing to audit."
        )
    if limits.max_rows is not None and data_rows > limits.max_rows:  # selected xls rows
        raise RowLimitError(
            f"{path.name} exceeds the row limit ({data_rows} data rows > {limits.max_rows}).",
            limit_name="max_rows", configured_limit=limits.max_rows,
            observed_value=data_rows, input_kind=SourceKind.FILE.value,
        )
    if limits.max_columns is not None and columns > limits.max_columns:  # selected xls columns
        raise ColumnLimitError(
            f"{path.name} exceeds the column limit ({columns} > {limits.max_columns}).",
            limit_name="max_columns", configured_limit=limits.max_columns,
            observed_value=columns, input_kind=SourceKind.FILE.value,
        )
    if limits.max_cells is not None and _cells_exceed(rows, columns, limits.max_cells):  # selected xls rectangle
        cells = rows * columns
        raise CellLimitError(
            f"{path.name} exceeds the cell limit ({cells} > {limits.max_cells}).",
            limit_name="max_cells", configured_limit=limits.max_cells,
            observed_value=cells, input_kind=SourceKind.FILE.value,
        )


def _materialize(path: Path, book, sheet) -> list[tuple[Any, ...]]:
    """Cached cell values. The formula expression is not read."""
    rows: list[tuple[Any, ...]] = []
    width = sheet.ncols
    for row_index in range(sheet.nrows):
        values = [
            _cached_value(path, book, sheet.cell(row_index, column_index), row_index, column_index)
            for column_index in range(width)
        ]
        rows.append(tuple(values))
    return rows


def _cached_value(path: Path, book, cell, row_index: int, column_index: int) -> Any:
    """One cell as a Python value. Error codes and formula text stay out."""
    import xlrd

    row = row_index + 1
    column = column_index + 1
    if cell.ctype == xlrd.XL_CELL_ERROR:
        raise MalformedInputError(
            f"{path.name} has an Excel error cell at row {row} column {column}."
        )
    if cell.ctype in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK):
        return None
    if cell.ctype == xlrd.XL_CELL_TEXT:
        return cell.value
    if cell.ctype == xlrd.XL_CELL_BOOLEAN:
        return bool(cell.value)
    if cell.ctype == xlrd.XL_CELL_DATE:
        try:
            return xlrd.xldate.xldate_as_datetime(cell.value, book.datemode)
        except xlrd.xldate.XLDateError:
            raise MalformedInputError(
                f"{path.name} has an invalid Excel date at row {row} column {column}."
            ) from None
    if cell.ctype == xlrd.XL_CELL_NUMBER:
        return cell.value  # cached number, never the formula expression
    raise MalformedInputError(
        f"{path.name} has an unsupported cell at row {row} column {column}."
    )
