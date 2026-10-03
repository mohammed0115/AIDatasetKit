"""The one table-loading authority: :func:`load_table`."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from aidatasetkit.core.exceptions import (
    CellLimitError,
    ColumnLimitError,
    DuplicateHeadersError,
    EmptyInputError,
    EncodingError,
    InputNotFoundError,
    InvalidIngestionOptionsError,
    FileSizeLimitError,
    KeyLimitError,
    MalformedInputError,
    RecordLimitError,
    RowLimitError,
    UnsupportedFormatError,
)
from aidatasetkit.ingestion.columnar import read_columnar
from aidatasetkit.ingestion.delimited import plan_delimited
from aidatasetkit.ingestion.excel import read_excel
from aidatasetkit.ingestion.formats import resolve_format
from aidatasetkit.ingestion.json_text import read_json_text
from aidatasetkit.ingestion.types import (
    LoadedTable,
    IngestionLimits,
    LoadMetadata,
    LoadOptions,
    SourceKind,
    TableFormat,
    _cells_exceed,
)

__all__ = ["load_table"]


def load_table(
    source: Any,
    *,
    options: LoadOptions | None = None,
    limits: IngestionLimits | None = None,
    sheet: str | None = None,
) -> LoadedTable:
    """Load a table, or refuse it with a structured error. Never a wrong table.

    Args:
        source: A path (:class:`pathlib.Path` or any :class:`os.PathLike`) to a
            ``.csv``, ``.tsv``, ``.json``, ``.jsonl``, ``.ndjson``, ``.parquet``,
            ``.feather``, ``.arrow`` or ``.xlsx`` file; a :class:`pandas.DataFrame`;
            or a list or tuple of mappings, one per row. A plain ``str`` is
            refused rather than guessed at -- pass ``Path("data.csv")``.
        options: How a file is read. In-memory sources accept only the defaults.
        sheet: The worksheet to read from an ``.xlsx`` file. Required when the
            workbook has more than one sheet; meaningless for any other source.

    Returns:
        A :class:`LoadedTable`. Loading profiles nothing and trains nothing.

    Raises:
        IngestionError: One of its subclasses, for every refusal.
    """
    options = options if options is not None else LoadOptions()
    limits = limits if limits is not None else IngestionLimits()
    if isinstance(source, pd.DataFrame):
        _require_default(options, "a DataFrame")
        if sheet is not None:
            raise InvalidIngestionOptionsError("sheet= applies to .xlsx files only.")
        return _from_dataframe(source, limits)
    if isinstance(source, (str, bytes)):
        raise UnsupportedFormatError(
            f"A {type(source).__name__} is not accepted as a source, so it can never "
            "be mistaken for data or for a path. Pass pathlib.Path(...) for a file."
        )
    if isinstance(source, os.PathLike):
        return _from_file(Path(os.fspath(source)), options, limits, sheet)
    if isinstance(source, (list, tuple)):
        _require_default(options, "records")
        if sheet is not None:
            raise InvalidIngestionOptionsError("sheet= applies to .xlsx files only.")
        return _from_records(source, limits)
    raise UnsupportedFormatError(
        f"Cannot load a {type(source).__name__}. Supported: a path to a .csv, "
        ".tsv, .json, .jsonl, .parquet, .feather or .xlsx file, a pandas "
        "DataFrame, or a list of mappings."
    )


def _require_default(options: LoadOptions, what: str) -> None:
    if not options.is_default:
        raise InvalidIngestionOptionsError(
            f"LoadOptions (encoding, delimiter, header) apply to files; {what} has "
            "none of these to set."
        )


def _memory(frame: pd.DataFrame) -> int:
    return int(frame.memory_usage(deep=True, index=True).sum())


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #


def _from_file(
    path: Path, options: LoadOptions, limits: IngestionLimits, sheet: str | None = None
) -> LoadedTable:
    fmt = resolve_format(path)
    if not path.exists():
        raise InputNotFoundError(f"No such file: {path.name}.")
    if not path.is_file():
        raise InputNotFoundError(f"{path.name} is not a regular file.")
    if limits.max_source_bytes is not None:
        size = path.stat().st_size
        if size > limits.max_source_bytes:
            raise FileSizeLimitError(
                f"{path.name} exceeds the source byte limit ({size} > {limits.max_source_bytes}).",
                limit_name="max_source_bytes", configured_limit=limits.max_source_bytes,
                observed_value=size, input_kind=SourceKind.FILE.value,
            )

    if fmt is not TableFormat.XLSX and sheet is not None:
        raise InvalidIngestionOptionsError(
            f"sheet= applies to .xlsx files only; {path.name} is {fmt.value}."
        )
    if fmt in (TableFormat.JSON, TableFormat.JSONL):
        records, names, encoding, notes = read_json_text(path, fmt, options, limits)
        frame, frame_notes = _records_frame(records, names, limits, SourceKind.FILE.value)
        return LoadedTable(
            frame=frame,
            metadata=LoadMetadata(
                source_kind=SourceKind.FILE,
                format=fmt,
                encoding=encoding,
                delimiter=None,
                delimiter_source=None,
                header=None,
                row_count=int(len(frame)),
                column_count=int(frame.shape[1]),
                memory_bytes=_memory(frame),
                warnings=notes + frame_notes,
            ),
        )

    if fmt in (TableFormat.PARQUET, TableFormat.FEATHER):
        frame = read_columnar(path, fmt, options, limits)
        return LoadedTable(
            frame=frame,
            metadata=LoadMetadata(
                source_kind=SourceKind.FILE,
                format=fmt,
                encoding=None,
                delimiter=None,
                delimiter_source=None,
                header=None,
                row_count=int(len(frame)),
                column_count=int(frame.shape[1]),
                memory_bytes=_memory(frame),
                warnings=(),
            ),
        )
    if fmt is TableFormat.XLSX:
        frame = read_excel(path, fmt, options, limits, sheet=sheet)
        return LoadedTable(
            frame=frame,
            metadata=LoadMetadata(
                source_kind=SourceKind.FILE,
                format=fmt,
                encoding=None,
                delimiter=None,
                delimiter_source=None,
                header=None,
                row_count=int(len(frame)),
                column_count=int(frame.shape[1]),
                memory_bytes=_memory(frame),
                warnings=(),
            ),
        )

    plan = plan_delimited(path, fmt, options, limits)
    try:
        frame = pd.read_csv(
            path,
            sep=plan.delimiter,
            encoding=plan.encoding,
            header=0 if options.header else None,
            skip_blank_lines=True,
        )
    except UnicodeDecodeError as error:
        raise EncodingError(
            f"{path.name} cannot be decoded as {plan.encoding} (byte offset {error.start})."
        ) from None
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as error:
        raise MalformedInputError(
            f"{path.name} passed validation but pandas could not parse it "
            f"({type(error).__name__}). Refused rather than read differently."
        ) from None

    if len(frame) != plan.data_rows or frame.shape[1] != plan.columns:
        raise MalformedInputError(
            f"{path.name}: the validating parser found {plan.data_rows} data row(s) and "
            f"{plan.columns} column(s), pandas found {len(frame)} and {frame.shape[1]}. "
            "Two parsers disagreeing about a file means it cannot be read reliably."
        )
    metadata = LoadMetadata(
        source_kind=SourceKind.FILE,
        format=fmt,
        encoding=plan.encoding,
        delimiter=None if plan.single_column else plan.delimiter,
        delimiter_source=None if plan.single_column else plan.delimiter_source,
        header=options.header,
        row_count=int(len(frame)),
        column_count=int(frame.shape[1]),
        memory_bytes=_memory(frame),
        warnings=plan.warnings,
    )
    return LoadedTable(frame=frame, metadata=metadata)


# --------------------------------------------------------------------------- #
# In-memory sources
# --------------------------------------------------------------------------- #


def _from_dataframe(frame: pd.DataFrame, limits: IngestionLimits) -> LoadedTable:
    """Checked and described, never copied or modified."""
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise EmptyInputError(
            f"The DataFrame has {frame.shape[0]} row(s) and {frame.shape[1]} column(s); "
            "there is no table to load."
        )
    rows, columns = map(int, frame.shape)
    if limits.max_rows is not None and rows > limits.max_rows:
        raise RowLimitError(
            f"DataFrame exceeds the row limit ({rows} > {limits.max_rows}).",
            limit_name="max_rows", configured_limit=limits.max_rows,
            observed_value=rows, input_kind=SourceKind.DATAFRAME.value,
        )
    if limits.max_columns is not None and columns > limits.max_columns:
        raise ColumnLimitError(
            f"DataFrame exceeds the column limit ({columns} > {limits.max_columns}).",
            limit_name="max_columns", configured_limit=limits.max_columns,
            observed_value=columns, input_kind=SourceKind.DATAFRAME.value,
        )
    if limits.max_cells is not None and _cells_exceed(rows, columns, limits.max_cells):
        cells = rows * columns
        raise CellLimitError(
            f"DataFrame exceeds the cell limit ({cells} > {limits.max_cells}).",
            limit_name="max_cells", configured_limit=limits.max_cells,
            observed_value=cells, input_kind=SourceKind.DATAFRAME.value,
        )
    duplicated = frame.columns[frame.columns.duplicated()]
    if len(duplicated):
        names = sorted({str(name) for name in duplicated})
        raise DuplicateHeadersError(
            f"The DataFrame has duplicate column labels ({', '.join(names)}). Give "
            "each column its own name first."
        )
    return LoadedTable(frame=frame, metadata=_in_memory(SourceKind.DATAFRAME, frame, ()))


def _from_records(records: Sequence[Any], limits: IngestionLimits) -> LoadedTable:
    """Rows as mappings. Columns in order of first appearance; missing keys are missing cells.

    A record that lacks a key another record has gets a missing value there, and
    the warnings say how many records that affected. Keys must be strings. The
    caller's list and mappings are read, never modified.
    """
    if len(records) == 0:
        raise EmptyInputError("The list of records is empty; there is no table to load.")
    if limits.max_records is not None and len(records) > limits.max_records:
        raise RecordLimitError(
            f"Records exceed the record limit ({len(records)} > {limits.max_records}).",
            limit_name="max_records", configured_limit=limits.max_records,
            observed_value=len(records), input_kind=SourceKind.RECORDS.value,
        )
    columns: dict[str, None] = {}
    total_chars = 0
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise MalformedInputError(
                f"Record {position} is a {type(record).__name__}, not a mapping. Every "
                "record must map column names to values."
            )
        if limits.max_keys_per_record is not None and len(record) > limits.max_keys_per_record:
            raise KeyLimitError(
                f"Record exceeds the key limit ({len(record)} > {limits.max_keys_per_record}).",
                limit_name="max_keys_per_record", configured_limit=limits.max_keys_per_record,
                observed_value=len(record), input_kind=SourceKind.RECORDS.value,
            )
        for key in record:
            if not isinstance(key, str):
                raise MalformedInputError(
                    f"Record {position} has a {type(key).__name__} key; column names "
                    "given as records must be strings."
                )
            columns.setdefault(key, None)
        if limits.max_record_chars is not None:
            total_chars += sum(len(str(value)) for value in record.values())
            if total_chars > limits.max_record_chars:
                raise CellLimitError(
                    "Records exceed the accumulated value-character limit.",
                    limit_name="max_record_chars", configured_limit=limits.max_record_chars,
                    observed_value=total_chars, input_kind=SourceKind.RECORDS.value,
                )
    frame, notes = _records_frame(records, list(columns), limits, SourceKind.RECORDS.value)
    return LoadedTable(frame=frame, metadata=_in_memory(SourceKind.RECORDS, frame, notes))


def _records_frame(
    records: Sequence[Mapping[str, Any]],
    names: list[str],
    limits: IngestionLimits,
    input_kind: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """The frame for checked records: the column and cell budgets, then the build.

    Columns are the keys in order of first appearance. A record that lacks a
    key gets a missing value there, and the warnings say how many records that
    affected. Shared by the records source and by the JSON readers, so a table
    of records means the same thing wherever the records came from.
    """
    if not names:
        raise EmptyInputError("Every record is empty; there are no columns to load.")
    if limits.max_columns is not None and len(names) > limits.max_columns:
        raise ColumnLimitError(
            f"Records exceed the column limit ({len(names)} > {limits.max_columns}).",
            limit_name="max_columns", configured_limit=limits.max_columns,
            observed_value=len(names), input_kind=input_kind,
        )
    if limits.max_cells is not None and _cells_exceed(len(records), len(names), limits.max_cells):
        cells = len(records) * len(names)
        raise CellLimitError(
            f"Records exceed the cell limit ({cells} > {limits.max_cells}).",
            limit_name="max_cells", configured_limit=limits.max_cells,
            observed_value=cells, input_kind=input_kind,
        )
    frame = pd.DataFrame([dict(record) for record in records], columns=names)
    incomplete = sum(1 for record in records if len(record) < len(names))
    notes: tuple[str, ...] = ()
    if incomplete:
        notes = (
            f"{incomplete} of {len(records)} records lack at least one of the "
            f"{len(names)} keys; those cells are missing values.",
        )
    return frame, notes


def _in_memory(kind: SourceKind, frame: pd.DataFrame, notes: tuple[str, ...]) -> LoadMetadata:
    return LoadMetadata(
        source_kind=kind,
        format=None,
        encoding=None,
        delimiter=None,
        delimiter_source=None,
        header=None,
        row_count=int(frame.shape[0]),
        column_count=int(frame.shape[1]),
        memory_bytes=_memory(frame),
        warnings=notes,
    )
