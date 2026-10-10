"""Read one ordinary table from a local SQLite file.

The file is opened read-only. The caller names a table, or the file has exactly
one ordinary table and that one is used. Views, virtual tables, the shadow
tables that belong to virtual tables, and SQLite's own ``sqlite_*`` tables are
not tables this reader will load. No SQL text is accepted from the caller.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from aidatasetkit.core.exceptions import (
    CellLimitError,
    ColumnLimitError,
    DuplicateHeadersError,
    EmptyInputError,
    FieldLengthLimitError,
    InputNotFoundError,
    InvalidIngestionOptionsError,
    MalformedInputError,
    RowLimitError,
)
from aidatasetkit.ingestion.types import (
    IngestionLimits,
    LoadOptions,
    SourceKind,
    SourceSelector,
    TableFormat,
    _cells_exceed,
)

__all__ = ["quote_identifier", "read_sqlite"]


def quote_identifier(name: str) -> str:
    """Quote a catalog identifier. This is not a place for a caller statement."""
    return '"' + name.replace('"', '""') + '"'


def read_sqlite(
    path: Path,
    fmt: TableFormat,
    options: LoadOptions,
    limits: IngestionLimits,
    *,
    table: str | None = None,
) -> tuple[pd.DataFrame, SourceSelector]:
    """One ordinary table as a DataFrame, after the resource checks.

    Raises:
        InvalidIngestionOptionsError, InputNotFoundError, EmptyInputError,
        MalformedInputError, DuplicateHeadersError, RowLimitError,
        ColumnLimitError, CellLimitError, FieldLengthLimitError.
    """
    del fmt
    if not options.is_default:
        raise InvalidIngestionOptionsError(
            "LoadOptions (encoding, delimiter, header) apply to text files; "
            f"{path.name} is a SQLite database, which describes itself."
        )
    if table is not None and not isinstance(table, str):
        raise InvalidIngestionOptionsError(
            f"table must be a string or None, got {type(table).__name__}."
        )
    if path.is_file() and path.stat().st_size == 0:
        raise EmptyInputError(
            f"{path.name} is empty (0 bytes); there is no table to read."
        )
    connection = _connect(path)
    try:
        try:
            return _read(connection, path, limits, table)
        except sqlite3.Error:
            raise MalformedInputError(
                f"{path.name} is not a readable SQLite database."
            ) from None
    finally:
        connection.close()


def _connect(path: Path) -> sqlite3.Connection:
    """Open ``path`` so SQLite cannot write it or create it."""
    if not path.is_file():
        raise InputNotFoundError(f"No such file: {path.name}.")
    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        raise MalformedInputError(
            f"{path.name} is not a readable SQLite database."
        ) from None
    _harden(connection)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _harden(connection: sqlite3.Connection) -> None:
    """Defense in depth. Missing constants leave mode=ro and query_only in force."""
    if not callable(getattr(connection, "setconfig", None)):
        return
    _set_if_available(connection, "SQLITE_DBCONFIG_DEFENSIVE", 1)
    _set_if_available(connection, "SQLITE_DBCONFIG_TRUSTED_SCHEMA", 0)
    _set_if_available(connection, "SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION", 0)


def _set_if_available(connection: sqlite3.Connection, name: str, value: int) -> None:
    constant = getattr(sqlite3, name, None)
    if constant is None:
        return
    try:
        connection.setconfig(constant, value)
    except sqlite3.Error:
        return


def _read(
    connection: sqlite3.Connection,
    path: Path,
    limits: IngestionLimits,
    table: str | None,
) -> tuple[pd.DataFrame, SourceSelector]:
    chosen = _choose(connection, path, table)
    names = _column_names(connection, path, chosen)
    quoted = quote_identifier(chosen)
    count = int(connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
    _check_shape(path, count, len(names), limits)
    _check_field_lengths(connection, path, quoted, names, limits)
    rows = _fetch_rows(connection, quoted)
    return _frame(names, rows), SourceSelector(kind="table", name=chosen)


def _choose(connection: sqlite3.Connection, path: Path, table: str | None) -> str:
    names = _eligible_names(connection, path)
    if table is None:
        if len(names) == 0:
            raise EmptyInputError(
                f"{path.name} has no ordinary table; there is no table to load."
            )
        if len(names) != 1:
            listed = ", ".join(names)
            raise MalformedInputError(
                f"{path.name} has {len(names)} tables ({listed}). Choosing one "
                "would be a guess; pass table= to name it."
            )
        return names[0]
    if table not in names:
        raise MalformedInputError(
            f"{path.name} has no ordinary table named {table!r}."
        )
    return table


def _eligible_names(connection: sqlite3.Connection, path: Path) -> tuple[str, ...]:
    """Ordinary user tables, in catalog order.

    ``PRAGMA table_list`` classifies view, virtual, shadow and table. When that
    pragma is absent, a database with no virtual-table definition still uses
    ``sqlite_master``. A virtual table without ``table_list`` is refused: its
    shadow tables cannot be told apart from ordinary tables without guessing
    their names.
    """
    classified = _relation_types(connection)
    if classified is None:
        if _defines_virtual_table(connection):
            raise MalformedInputError(
                f"{path.name} has a virtual table whose shadow tables this "
                "SQLite cannot classify."
            )
        rows = connection.execute(
            "SELECT name, type, sql FROM sqlite_master"
        ).fetchall()
        return tuple(
            name
            for name, kind, sql in rows
            if _is_ordinary_table(name, kind, sql)
        )
    return tuple(
        name
        for name, kind in classified
        if _is_ordinary_table(name, kind, "")
    )


def _relation_types(connection: sqlite3.Connection) -> list[tuple[str, str]] | None:
    """``(name, type)`` from ``PRAGMA table_list``, or None when it is absent.

    An unknown pragma returns no rows. A supported catalog always names
    ``sqlite_schema``, so an empty result means the classification is missing.
    """
    rows = connection.execute("PRAGMA table_list").fetchall()
    if not rows or len(rows[0]) < 3:
        return None
    return [
        (str(name), str(kind))
        for schema, name, kind, *_rest in rows
        if schema == "main"
    ]


def _defines_virtual_table(connection: sqlite3.Connection) -> bool:
    rows = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return any(
        sql is not None and sql.lstrip().upper().startswith("CREATE VIRTUAL")
        for (sql,) in rows
    )


def _is_ordinary_table(name: str, kind: str, sql: str | None) -> bool:
    """True for a catalog type of ``table`` that is not an internal name.

    View, virtual and shadow are excluded by the type SQLite reports. The kind
    check below them refuses any other non-table type. It allows those three
    through so each has its own exclusion. Names are never used as a guess
    about shadow storage. On the ``sqlite_master`` fallback, a virtual table
    is stored with type ``table`` and is excluded by its ``CREATE VIRTUAL``
    statement; shadow tables of that virtual table are not on this path,
    because the database was refused before the fallback built an allowlist.
    """
    if kind == "view":
        return False
    if kind == "virtual":
        return False
    if kind == "shadow":
        return False
    if kind != "table" and kind not in {"view", "virtual", "shadow"}:
        return False
    if name.startswith("sqlite_"):
        return False
    if sql is None or sql.lstrip().upper().startswith("CREATE VIRTUAL"):
        return False
    return True


def _column_names(
    connection: sqlite3.Connection, path: Path, table: str
) -> list[str]:
    rows = connection.execute(
        "SELECT name FROM pragma_table_info(?)", (table,)
    ).fetchall()
    names = [row[0] for row in rows]
    if not names:
        raise MalformedInputError(
            f"{path.name} has no columns in ordinary table {table!r}."
        )
    if len(set(names)) != len(names):
        raise DuplicateHeadersError(
            f"{path.name} has two columns with the same name in table {table!r}."
        )
    return names


def _check_shape(path: Path, rows: int, columns: int, limits: IngestionLimits) -> None:
    if limits.max_columns is not None and columns > limits.max_columns:
        raise ColumnLimitError(
            f"{path.name} exceeds the column limit ({columns} > {limits.max_columns}).",
            limit_name="max_columns",
            configured_limit=limits.max_columns,
            observed_value=columns,
            input_kind=SourceKind.FILE.value,
        )
    if limits.max_rows is not None and rows > limits.max_rows:
        raise RowLimitError(
            f"{path.name} exceeds the row limit ({rows} > {limits.max_rows}).",
            limit_name="max_rows",
            configured_limit=limits.max_rows,
            observed_value=rows,
            input_kind=SourceKind.FILE.value,
        )
    if limits.max_cells is not None and _cells_exceed(rows, columns, limits.max_cells):
        cells = rows * columns
        raise CellLimitError(
            f"{path.name} exceeds the cell limit ({cells} > {limits.max_cells}).",
            limit_name="max_cells",
            configured_limit=limits.max_cells,
            observed_value=cells,
            input_kind=SourceKind.FILE.value,
        )


def _check_field_lengths(
    connection: sqlite3.Connection,
    path: Path,
    quoted: str,
    names: list[str],
    limits: IngestionLimits,
) -> None:
    if limits.max_field_length is None:
        return
    longest_text = 0
    longest_blob = 0
    for name in names:
        column = quote_identifier(name)
        text_length = connection.execute(
            f"SELECT MAX(LENGTH({column})) FROM {quoted} "
            f"WHERE typeof({column}) = 'text'"
        ).fetchone()[0]
        blob_length = connection.execute(
            f"SELECT MAX(LENGTH({column})) FROM {quoted} "
            f"WHERE typeof({column}) = 'blob'"
        ).fetchone()[0]
        if text_length is not None:
            longest_text = max(longest_text, int(text_length))
        if blob_length is not None:
            longest_blob = max(longest_blob, int(blob_length))
    if longest_text > limits.max_field_length:
        raise FieldLengthLimitError(
            f"{path.name} exceeds the field length limit "
            f"({longest_text} > {limits.max_field_length}).",
            limit_name="max_field_length",
            configured_limit=limits.max_field_length,
            observed_value=longest_text,
            input_kind=SourceKind.FILE.value,
        )
    if longest_blob > limits.max_field_length:
        raise FieldLengthLimitError(
            f"{path.name} exceeds the field length limit "
            f"({longest_blob} > {limits.max_field_length}).",
            limit_name="max_field_length",
            configured_limit=limits.max_field_length,
            observed_value=longest_blob,
            input_kind=SourceKind.FILE.value,
        )


def _fetch_rows(connection: sqlite3.Connection, quoted: str) -> list[tuple]:
    return connection.execute(f"SELECT * FROM {quoted}").fetchall()


def _frame(names: list[str], rows: list[tuple]) -> pd.DataFrame:
    columns = {
        name: [row[index] for row in rows] for index, name in enumerate(names)
    }
    return pd.DataFrame(columns, columns=names, dtype=object)
