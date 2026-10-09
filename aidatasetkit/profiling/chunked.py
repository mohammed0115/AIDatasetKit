"""Opt-in chunked profiling for CSV and TSV files.

:func:`aidatasetkit.ingestion.load_table` and :meth:`DataProfiler.profile` are
unchanged. This function is a separate entry point: it is used only when a
caller asks for it. The file is still refused when it exceeds
:class:`~aidatasetkit.ingestion.IngestionLimits`. Nothing here publishes an
audit, and nothing here is consulted by the verdict.

The scan reads the population in chunks. Counts, missing values, and the
numeric minimum, maximum, sum and mean are exact. Quartiles are exact unless
the caller sets ``approximate_quantiles_above`` below the number of finite
values; that case is returned in :attr:`ChunkedProfile.approximations` with the
label ``deterministic_approximation`` and is omitted from the exact quartile
fields. The fingerprint is the population fingerprint, the same digest
:func:`~aidatasetkit.evidence.fingerprint.dataset_fingerprint` would compute
for the whole table.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aidatasetkit.core.exceptions import (
    FileSizeLimitError,
    InputNotFoundError,
    InvalidIngestionOptionsError,
    MalformedInputError,
)
from aidatasetkit.ingestion.delimited import plan_delimited
from aidatasetkit.ingestion.formats import resolve_format
from aidatasetkit.ingestion.types import (
    IngestionLimits,
    LoadOptions,
    SourceKind,
    TableFormat,
)
from aidatasetkit.statistics.engine import StatisticsEngine

__all__ = [
    "ChunkedApproximation",
    "ChunkedColumn",
    "ChunkedProfile",
    "profile_delimited_chunks",
]

_APPROXIMATION_LABEL = "deterministic_approximation"
_APPROXIMATION_METHOD = "deterministic_even_stride"

#: Must stay equal to ``aidatasetkit.evidence.fingerprint._CHUNK_ROWS``. Object
#: columns mix the per-value type names in groups of this size, so a different
#: group size would change the population fingerprint. A test holds the two
#: constants together; this module does not import evidence.
_FINGERPRINT_CHUNK_ROWS = 100_000


@dataclass(frozen=True, slots=True)
class ChunkedColumn:
    """Exact measurements for one column of the whole population."""

    name: str
    pandas_dtype: str
    count: int
    missing_count: int
    unique_count: int
    minimum: float | None = None
    maximum: float | None = None
    sum: float | None = None
    mean: float | None = None
    q25: float | None = None
    median: float | None = None
    q75: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "pandas_dtype": self.pandas_dtype,
            "count": self.count,
            "missing_count": self.missing_count,
            "unique_count": self.unique_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "sum": self.sum,
            "mean": self.mean,
            "q25": self.q25,
            "median": self.median,
            "q75": self.q75,
        }


@dataclass(frozen=True, slots=True)
class ChunkedApproximation:
    """A deterministic stand-in that is not an exact population measurement.

    ``label`` is always ``deterministic_approximation``. Callers that decide a
    verdict must ignore these values; the exact fields on :class:`ChunkedColumn`
    do not carry them.
    """

    column: str
    field: str
    method: str
    label: str
    value: float

    def to_dict(self) -> dict[str, object]:
        return {
            "column": self.column,
            "field": self.field,
            "method": self.method,
            "label": self.label,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class ChunkedProfile:
    """What a chunked CSV/TSV scan measured.

    ``population_fingerprint`` covers every row. ``temporary_storage`` is
    ``removed`` only after the scratch directory has been deleted.
    ``approximations`` is empty when every recorded number is exact.
    """

    format: str
    chunk_rows: int
    population_rows: int
    population_columns: int
    population_fingerprint: str
    duplicate_row_count: int
    columns: tuple[ChunkedColumn, ...]
    approximations: tuple[ChunkedApproximation, ...]
    temporary_storage: str

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "chunk_rows": self.chunk_rows,
            "population_rows": self.population_rows,
            "population_columns": self.population_columns,
            "population_fingerprint": self.population_fingerprint,
            "fingerprint_algorithm": "sha256/pandas-hash-v1",
            "fingerprint_scope": "population",
            "duplicate_row_count": self.duplicate_row_count,
            "columns": [column.to_dict() for column in self.columns],
            "approximations": [item.to_dict() for item in self.approximations],
            "temporary_storage": self.temporary_storage,
        }


def profile_delimited_chunks(
    source: str | Path,
    *,
    options: LoadOptions | None = None,
    limits: IngestionLimits | None = None,
    chunk_rows: int = 10_000,
    approximate_quantiles_above: int | None = None,
) -> ChunkedProfile:
    """Profile a CSV or TSV file in chunks, without replacing the default audit.

    Args:
        source: Path of a ``.csv`` or ``.tsv`` file.
        options: Encoding, delimiter and header. The same object
            :func:`~aidatasetkit.ingestion.load_table` accepts.
        limits: Resource limits. Defaults to :class:`IngestionLimits`. An
            over-limit file is refused before a scratch directory is created.
        chunk_rows: How many data rows are held at once. Must be at least 1.
        approximate_quantiles_above: When set, a numeric column with more
            finite values than this uses a deterministic even stride for its
            quartiles and records that fact. The minimum, maximum, sum and
            mean stay exact. ``None`` keeps every quartile exact.

    Raises:
        InvalidIngestionOptionsError: The path is not CSV or TSV, or a chunk
            setting is not a positive integer.
        InputNotFoundError, MalformedInputError: The file cannot be read as a
            table.
        ResourceLimitError: The file exceeds ``limits``.
    """
    path = Path(source)
    options = options if options is not None else LoadOptions()
    limits = limits if limits is not None else IngestionLimits()
    _check_chunk_settings(chunk_rows, approximate_quantiles_above)
    if not path.exists():
        raise InputNotFoundError(f"No such file: {path.name}.")
    if not path.is_file():
        raise InputNotFoundError(f"{path.name} is not a regular file.")
    fmt = resolve_format(path)
    if fmt not in (TableFormat.CSV, TableFormat.TSV):
        raise InvalidIngestionOptionsError(
            f"Chunked profiling is available for CSV and TSV only, and {path.name} "
            f"is {fmt.value}."
        )
    if limits.max_source_bytes is not None:
        size = path.stat().st_size
        if size > limits.max_source_bytes:
            raise FileSizeLimitError(
                f"{path.name} exceeds the source byte limit ({size} > {limits.max_source_bytes}).",
                limit_name="max_source_bytes",
                configured_limit=limits.max_source_bytes,
                observed_value=size,
                input_kind=SourceKind.FILE.value,
            )
    plan = plan_delimited(path, fmt, options, limits)

    directory: Path | None = None
    try:
        directory = Path(tempfile.mkdtemp(prefix="aidatasetkit-chunked-"))
        os.chmod(directory, 0o700)
        scanned = _scan(
            path,
            plan,
            options,
            directory,
            fmt=fmt,
            chunk_rows=chunk_rows,
            approximate_quantiles_above=approximate_quantiles_above,
        )
    finally:
        if directory is not None:
            shutil.rmtree(directory)
    return ChunkedProfile(temporary_storage="removed", **scanned)


def _check_chunk_settings(chunk_rows: int, approximate_quantiles_above: int | None) -> None:
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise InvalidIngestionOptionsError(
            f"chunk_rows must be a positive integer, got {chunk_rows!r}."
        )
    if approximate_quantiles_above is None:
        return
    if (
        isinstance(approximate_quantiles_above, bool)
        or not isinstance(approximate_quantiles_above, int)
        or approximate_quantiles_above < 1
    ):
        raise InvalidIngestionOptionsError(
            "approximate_quantiles_above must be a positive integer or None, "
            f"got {approximate_quantiles_above!r}."
        )


def _iter_chunks(path: Path, plan, options: LoadOptions, chunk_rows: int):
    reader = pd.read_csv(
        path,
        sep=plan.delimiter,
        encoding=plan.encoding,
        header=0 if options.header else None,
        skip_blank_lines=True,
        chunksize=chunk_rows,
    )
    if type(reader).__name__ != "TextFileReader":
        raise RuntimeError("chunked profiling read the file without a chunk size")
    return reader


def _promote(path: Path, plan, options: LoadOptions, chunk_rows: int):
    carriers: dict[object, pd.Series] = {}
    order: list[object] = []
    for chunk in _iter_chunks(path, plan, options, chunk_rows):
        for label in chunk.columns:
            empty = chunk[label].iloc[:0]
            if label not in carriers:
                order.append(label)
                carriers[label] = empty
            else:
                carriers[label] = pd.concat([carriers[label], empty], ignore_index=True)
    return order, {label: carriers[label].dtype for label in order}


def _private(path: Path) -> None:
    os.chmod(path, 0o600)


def _encode(value: object) -> tuple[str, str]:
    if value is None or value is pd.NA:
        return ("missing", "")
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    if missing:
        return ("missing", "")
    if isinstance(value, (bool, np.bool_)):
        return ("bool", "1" if bool(value) else "0")
    if isinstance(value, (np.integer, int)):
        return ("int", str(int(value)))
    if isinstance(value, (np.floating, float)):
        return ("float", struct.pack("<d", float(value)).hex())
    return ("str", str(value))


def _row_key(row: tuple[object, ...]) -> str:
    return "\x1e".join(f"{kind}:{payload}" for kind, payload in (_encode(value) for value in row))


def _even_stride(values: np.ndarray, cap: int) -> np.ndarray:
    indices = np.linspace(0, values.size - 1, num=cap, dtype=np.int64)
    return values[np.unique(indices)]


def _scan(
    path: Path,
    plan,
    options: LoadOptions,
    directory: Path,
    *,
    fmt: TableFormat,
    chunk_rows: int,
    approximate_quantiles_above: int | None,
) -> dict[str, object]:
    columns, dtypes = _promote(path, plan, options, chunk_rows)
    if len(columns) != plan.columns:
        raise MalformedInputError(
            f"{path.name}: chunked profiling saw {len(columns)} columns, not the "
            f"{plan.columns} the validating parser counted."
        )

    database = directory / "rows.sqlite"
    connection = sqlite3.connect(database)
    _private(database)
    connection.execute("PRAGMA journal_mode=MEMORY")
    connection.execute("CREATE TABLE keys (k TEXT PRIMARY KEY)")

    hash_paths = {label: directory / f"hash-{index}.bin" for index, label in enumerate(columns)}
    hash_handles = {label: open(hash_paths[label], "wb") for label in columns}
    for handle_path in hash_paths.values():
        _private(handle_path)
    numeric_paths: dict[object, Path] = {}
    numeric_handles: dict[object, object] = {}
    for index, label in enumerate(columns):
        dtype = dtypes[label]
        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            numeric_paths[label] = directory / f"num-{index}.bin"
            numeric_handles[label] = open(numeric_paths[label], "wb")
            _private(numeric_paths[label])

    missing = {label: 0 for label in columns}
    present_count = {label: 0 for label in columns}
    distinct: dict[object, set[object]] = {label: set() for label in columns}
    type_names: dict[object, list[str]] = {label: [] for label in columns}
    seen_rows = 0
    try:
        for chunk in _iter_chunks(path, plan, options, chunk_rows):
            cast = chunk.astype(dtypes)
            if list(cast.columns) != list(columns):
                raise MalformedInputError(
                    f"{path.name}: a chunk did not have the columns of the first chunk."
                )
            for label in columns:
                series = cast[label]
                missing[label] += int(series.isna().sum())
                present = series.dropna()
                present_count[label] += int(len(present))
                for value in present.tolist():
                    if isinstance(value, np.generic):
                        value = value.item()
                    distinct[label].add(value)
                hashed = pd.util.hash_pandas_object(series, index=False)
                hash_handles[label].write(hashed.to_numpy(dtype="uint64").tobytes())
                if dtypes[label] == object:
                    type_names[label].extend(type(value).__name__ for value in series)
                handle = numeric_handles.get(label)
                if handle is not None:
                    array = series.to_numpy(dtype="float64", na_value=np.nan)
                    handle.write(np.ascontiguousarray(array, dtype="<f8").tobytes())
            connection.executemany(
                "INSERT OR IGNORE INTO keys(k) VALUES (?)",
                ((_row_key(row),) for row in cast.itertuples(index=False, name=None)),
            )
            seen_rows += len(cast)
        connection.commit()
        if seen_rows != plan.data_rows:
            raise MalformedInputError(
                f"{path.name}: chunked profiling saw {seen_rows} data rows, "
                f"not the {plan.data_rows} the validating parser counted."
            )
        distinct_keys = int(connection.execute("SELECT COUNT(*) FROM keys").fetchone()[0])
    finally:
        connection.close()
        for handle in hash_handles.values():
            handle.close()
        for handle in numeric_handles.values():
            handle.close()

    fingerprint = _population_fingerprint(columns, dtypes, seen_rows, hash_paths, type_names)
    approximations: list[ChunkedApproximation] = []
    column_records = []
    for label in columns:
        minimum = maximum = total = mean = q25 = median = q75 = None
        numeric_path = numeric_paths.get(label)
        if numeric_path is not None:
            values = np.fromfile(numeric_path, dtype="<f8")
            finite = values[np.isfinite(values)]
            if finite.size:
                engine = StatisticsEngine(finite)
                minimum = float(engine.min())
                maximum = float(engine.max())
                mean = float(engine.mean())
                total = float(finite.sum())
                quartiles, approximated = _quartiles(finite, approximate_quantiles_above)
                if approximated:
                    for field, value in (
                        ("q25", quartiles.q1),
                        ("median", quartiles.q2),
                        ("q75", quartiles.q3),
                    ):
                        approximations.append(
                            ChunkedApproximation(
                                column=str(label),
                                field=field,
                                method=_APPROXIMATION_METHOD,
                                label=_APPROXIMATION_LABEL,
                                value=float(value),
                            )
                        )
                else:
                    q25, median, q75 = float(quartiles.q1), float(quartiles.q2), float(quartiles.q3)
        column_records.append(
            ChunkedColumn(
                name=str(label),
                pandas_dtype=str(dtypes[label]),
                count=present_count[label],
                missing_count=missing[label],
                unique_count=len(distinct[label]),
                minimum=minimum,
                maximum=maximum,
                sum=total,
                mean=mean,
                q25=q25,
                median=median,
                q75=q75,
            )
        )
    return {
        "format": fmt.value,
        "chunk_rows": chunk_rows,
        "population_rows": seen_rows,
        "population_columns": len(columns),
        "population_fingerprint": fingerprint,
        "duplicate_row_count": seen_rows - distinct_keys,
        "columns": tuple(column_records),
        "approximations": tuple(approximations),
    }


def _quartiles(finite: np.ndarray, cap: int | None):
    if cap is not None and finite.size > cap:
        return StatisticsEngine(_even_stride(finite, cap)).quartiles(), True
    return StatisticsEngine(finite).quartiles(), False


def _label_token(label: object) -> str:
    """The same text :func:`aidatasetkit.evidence.serialization.label_token` uses."""
    return f"{type(label).__name__}:{label}"


def _digest_of(*parts: str) -> str:
    """The same length-delimited digest as ``evidence.fingerprint.digest_of``."""
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
    return digest.hexdigest()


def _population_fingerprint(columns, dtypes, row_count: int, hash_paths, type_names) -> str:
    parts = [f"rows={row_count}", f"columns={len(columns)}"]
    for position, label in enumerate(columns):
        parts.append(f"{position}|{_label_token(label)}|{dtypes[label]!s}")
    digest = hashlib.sha256()
    digest.update(_digest_of(*parts).encode("ascii"))
    for label in columns:
        digest.update(b"\x00")
        digest.update(_label_token(label).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(hash_paths[label].read_bytes())
        if dtypes[label] == object:
            names = type_names[label]
            for start in range(0, max(len(names), 1), _FINGERPRINT_CHUNK_ROWS):
                group = names[start : start + _FINGERPRINT_CHUNK_ROWS]
                if not group:
                    continue
                digest.update("\x1f".join(group).encode("utf-8"))
    return digest.hexdigest()
