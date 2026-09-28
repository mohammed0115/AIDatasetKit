"""The one table-loading authority: :func:`load_table`."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from aidatasetkit.core.exceptions import (
    DuplicateHeadersError,
    EmptyInputError,
    EncodingError,
    InputNotFoundError,
    InvalidIngestionOptionsError,
    MalformedInputError,
    UnsupportedFormatError,
)
from aidatasetkit.ingestion.delimited import plan_delimited
from aidatasetkit.ingestion.formats import resolve_format
from aidatasetkit.ingestion.types import (
    LoadedTable,
    LoadMetadata,
    LoadOptions,
    SourceKind,
)

__all__ = ["load_table"]


def load_table(source: Any, *, options: LoadOptions | None = None) -> LoadedTable:
    """Load a table, or refuse it with a structured error. Never a wrong table.

    Args:
        source: A path (:class:`pathlib.Path` or any :class:`os.PathLike`) to a
            ``.csv`` or ``.tsv`` file; a :class:`pandas.DataFrame`; or a list or
            tuple of mappings, one per row. A plain ``str`` is refused rather
            than guessed at -- pass ``Path("data.csv")``.
        options: How a file is read. In-memory sources accept only the defaults.

    Returns:
        A :class:`LoadedTable`. Loading profiles nothing and trains nothing.

    Raises:
        IngestionError: One of its subclasses, for every refusal.
    """
    options = options if options is not None else LoadOptions()
    if isinstance(source, pd.DataFrame):
        _require_default(options, "a DataFrame")
        return _from_dataframe(source)
    if isinstance(source, (str, bytes)):
        raise UnsupportedFormatError(
            f"A {type(source).__name__} is not accepted as a source, so it can never "
            "be mistaken for data or for a path. Pass pathlib.Path(...) for a file."
        )
    if isinstance(source, os.PathLike):
        return _from_file(Path(os.fspath(source)), options)
    if isinstance(source, (list, tuple)):
        _require_default(options, "records")
        return _from_records(source)
    raise UnsupportedFormatError(
        f"Cannot load a {type(source).__name__}. Supported: a path to a .csv or .tsv "
        "file, a pandas DataFrame, or a list of mappings."
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


def _from_file(path: Path, options: LoadOptions) -> LoadedTable:
    fmt = resolve_format(path)
    if not path.exists():
        raise InputNotFoundError(f"No such file: {path.name}.")
    if not path.is_file():
        raise InputNotFoundError(f"{path.name} is not a regular file.")

    plan = plan_delimited(path, fmt, options)
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


def _from_dataframe(frame: pd.DataFrame) -> LoadedTable:
    """Checked and described, never copied or modified."""
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise EmptyInputError(
            f"The DataFrame has {frame.shape[0]} row(s) and {frame.shape[1]} column(s); "
            "there is no table to load."
        )
    duplicated = frame.columns[frame.columns.duplicated()]
    if len(duplicated):
        names = sorted({str(name) for name in duplicated})
        raise DuplicateHeadersError(
            f"The DataFrame has duplicate column labels ({', '.join(names)}). Give "
            "each column its own name first."
        )
    return LoadedTable(frame=frame, metadata=_in_memory(SourceKind.DATAFRAME, frame, ()))


def _from_records(records: Sequence[Any]) -> LoadedTable:
    """Rows as mappings. Columns in order of first appearance; missing keys are missing cells.

    A record that lacks a key another record has gets a missing value there, and
    the warnings say how many records that affected. Keys must be strings. The
    caller's list and mappings are read, never modified.
    """
    if len(records) == 0:
        raise EmptyInputError("The list of records is empty; there is no table to load.")
    columns: dict[str, None] = {}
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise MalformedInputError(
                f"Record {position} is a {type(record).__name__}, not a mapping. Every "
                "record must map column names to values."
            )
        for key in record:
            if not isinstance(key, str):
                raise MalformedInputError(
                    f"Record {position} has a {type(key).__name__} key; column names "
                    "given as records must be strings."
                )
            columns.setdefault(key, None)
    if not columns:
        raise EmptyInputError("Every record is empty; there are no columns to load.")

    names = list(columns)
    frame = pd.DataFrame([dict(record) for record in records], columns=names)
    incomplete = sum(1 for record in records if len(record) < len(names))
    notes = ()
    if incomplete:
        notes = (
            f"{incomplete} of {len(records)} records lack at least one of the "
            f"{len(names)} keys; those cells are missing values.",
        )
    return LoadedTable(frame=frame, metadata=_in_memory(SourceKind.RECORDS, frame, notes))


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
