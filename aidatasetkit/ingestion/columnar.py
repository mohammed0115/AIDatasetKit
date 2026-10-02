"""Reading Parquet and Feather: columnar binary tables through pyarrow.

pyarrow is an optional dependency -- the ``parquet`` extra -- so importing this
module never imports it; the import happens when a columnar file is actually
read, and its absence is a :class:`MissingDependencyError` that names the
extra, not a traceback.

The two formats differ in what can be known before data is read. A Parquet
footer carries the row and column counts, so the row, column and cell budgets
are enforced from the footer before a single value is materialized, and the
counts found while reading must agree with it. A Feather (Arrow IPC) footer
carries only the schema, so the column budget is enforced before the data is
read and the row and cell budgets immediately after the Arrow read, before the
pandas conversion; the loader's ``max_source_bytes`` preflight bounds the file
itself either way. Anything that is not a readable file of its format is
refused with :class:`MalformedInputError`, never a bare pyarrow error.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from aidatasetkit.core.exceptions import (
    CellLimitError,
    ColumnLimitError,
    DuplicateHeadersError,
    EmptyInputError,
    InvalidIngestionOptionsError,
    MalformedInputError,
    MissingDependencyError,
    RowLimitError,
)
from aidatasetkit.ingestion.types import (
    IngestionLimits,
    LoadOptions,
    SourceKind,
    TableFormat,
    _cells_exceed,
)

__all__ = ["read_columnar"]


def read_columnar(
    path: Path,
    fmt: TableFormat,
    options: LoadOptions,
    limits: IngestionLimits,
) -> pd.DataFrame:
    """The frame held by a Parquet or Feather file, or a structured refusal.

    Raises:
        InvalidIngestionOptionsError, MissingDependencyError, EmptyInputError,
        MalformedInputError, DuplicateHeadersError, RowLimitError,
        ColumnLimitError, CellLimitError.
    """
    if not options.is_default:
        raise InvalidIngestionOptionsError(
            f"LoadOptions (encoding, delimiter, header) apply to text files; "
            f"{path.name} is a binary {fmt.value} file, which describes itself."
        )
    pa = _pyarrow(path, fmt)
    if fmt is TableFormat.PARQUET:
        return _parquet_frame(path, pa, limits)
    return _feather_frame(path, pa, limits)


def _pyarrow(path: Path, fmt: TableFormat):
    """The pyarrow module, or a refusal naming the extra that provides it."""
    try:
        spec = importlib.util.find_spec("pyarrow")
    except ImportError:
        spec = None
    if spec is None:
        raise MissingDependencyError(
            f"Reading {fmt.value} files needs pyarrow, which is not installed. "
            "It is an optional dependency: pip install 'aidatasetkit[parquet]'."
        )
    import pyarrow as pa

    return pa


# --------------------------------------------------------------------------- #
# Parquet: the footer knows the shape before any data is read
# --------------------------------------------------------------------------- #


def _parquet_frame(path: Path, pa, limits: IngestionLimits) -> pd.DataFrame:
    import pyarrow.parquet as pq

    try:
        footer = pq.read_metadata(path)
    except (pa.lib.ArrowException, OSError) as error:
        raise MalformedInputError(
            f"{path.name} is not a readable Parquet file ({type(error).__name__})."
        ) from None
    _check_names(path, footer.schema.names)
    _check_shape(path, footer.num_rows, footer.num_columns, limits)
    try:
        table = pq.read_table(path)
    except (pa.lib.ArrowException, OSError) as error:
        raise MalformedInputError(
            f"{path.name} is not a readable Parquet file ({type(error).__name__})."
        ) from None
    if table.num_rows != footer.num_rows or table.num_columns != footer.num_columns:
        raise MalformedInputError(
            f"{path.name}: the footer promised {footer.num_rows} row(s) and "
            f"{footer.num_columns} column(s), the data holds {table.num_rows} and "
            f"{table.num_columns}. A file that disagrees with its own footer "
            "cannot be read reliably."
        )
    return table.to_pandas()


# --------------------------------------------------------------------------- #
# Feather: the footer knows the schema; the rows are known once read
# --------------------------------------------------------------------------- #


def _feather_frame(path: Path, pa, limits: IngestionLimits) -> pd.DataFrame:
    try:
        reader = pa.ipc.open_file(path)
    except (pa.lib.ArrowException, OSError) as error:
        raise MalformedInputError(
            f"{path.name} is not a readable Feather file ({type(error).__name__})."
        ) from None
    names = reader.schema.names
    _check_names(path, names)
    if limits.max_columns is not None and len(names) > limits.max_columns:
        raise ColumnLimitError(
            f"Feather file exceeds the column limit ({len(names)} > {limits.max_columns}).",
            limit_name="max_columns", configured_limit=limits.max_columns,
            observed_value=len(names), input_kind=SourceKind.FILE.value,
        )
    try:
        table = reader.read_all()
    except (pa.lib.ArrowException, OSError) as error:
        raise MalformedInputError(
            f"{path.name} is not a readable Feather file ({type(error).__name__})."
        ) from None
    _check_shape(path, table.num_rows, len(names), limits)
    return table.to_pandas()


# --------------------------------------------------------------------------- #
# The shape budgets
# --------------------------------------------------------------------------- #


def _check_names(path: Path, names: list[str]) -> None:
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise DuplicateHeadersError(
            f"{path.name} has duplicate column labels ({', '.join(duplicated)}). "
            "Give each column its own name first."
        )


def _check_shape(path: Path, rows: int, columns: int, limits: IngestionLimits) -> None:
    """Empty, row, column and cell checks against a known logical shape."""
    if rows == 0 or columns == 0:
        raise EmptyInputError(
            f"{path.name} has {rows} row(s) and {columns} column(s); there is no table to load."
        )
    if limits.max_rows is not None and rows > limits.max_rows:
        raise RowLimitError(
            f"{path.name} exceeds the row limit ({rows} > {limits.max_rows}).",
            limit_name="max_rows", configured_limit=limits.max_rows,
            observed_value=rows, input_kind=SourceKind.FILE.value,
        )
    if limits.max_columns is not None and columns > limits.max_columns:
        raise ColumnLimitError(
            f"{path.name} exceeds the column limit ({columns} > {limits.max_columns}).",
            limit_name="max_columns", configured_limit=limits.max_columns,
            observed_value=columns, input_kind=SourceKind.FILE.value,
        )
    if limits.max_cells is not None and _cells_exceed(rows, columns, limits.max_cells):
        cells = rows * columns
        raise CellLimitError(
            f"{path.name} exceeds the cell limit ({cells} > {limits.max_cells}).",
            limit_name="max_cells", configured_limit=limits.max_cells,
            observed_value=cells, input_kind=SourceKind.FILE.value,
        )
