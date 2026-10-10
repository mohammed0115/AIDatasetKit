"""Opt-in chunked profiling for CSV and TSV files.

:func:`aidatasetkit.ingestion.load_table` and :meth:`DataProfiler.profile` are
unchanged. This function is a separate entry point: it is used only when a
caller asks for it. The file is still refused when it exceeds
:class:`~aidatasetkit.ingestion.IngestionLimits`. Nothing here publishes an
audit, and nothing here is consulted by a readiness verdict.

The scan reads the population in chunks. Counts, missing values, finite and
infinite counts, and the numeric minimum, maximum, sum and mean are exact
online accumulators. Distinct values and duplicate rows live in a scratch
database, not in a Python set. Quartiles are always a bounded deterministic
sample and are labeled ``deterministic_approximation``. The fingerprint is the
population fingerprint, the same digest
:func:`~aidatasetkit.evidence.fingerprint.dataset_fingerprint` would compute
for the whole table. Hash bytes are folded in fixed-size blocks.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import shutil
import sqlite3
import struct
import tempfile
from dataclasses import dataclass, replace
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
_QUANTILE_METHOD = "deterministic_reservoir"
_QUANTILE_SEED = 0
_QUANTILE_SAMPLE_SIZE = 4096
_READ_BLOCK = 65_536
_FINGERPRINT_ALGORITHM = "sha256/pandas-hash-v1"
_FINGERPRINT_SCOPE = "population"
_MODE = "chunked_profile"

#: Must stay equal to ``aidatasetkit.evidence.fingerprint._CHUNK_ROWS``. Object
#: columns mix the per-value type names in groups of this size, so a different
#: group size would change the population fingerprint. A test holds the two
#: constants together; this module does not import evidence.
_FINGERPRINT_CHUNK_ROWS = 100_000

EXACT_METRICS = (
    "count",
    "missing_count",
    "finite_count",
    "infinite_count",
    "unique_count",
    "minimum",
    "maximum",
    "sum",
    "mean",
    "duplicate_row_count",
)
UNAVAILABLE_METRICS = (
    "std",
    "dominant_value",
    "quality_findings",
    "task_detection",
    "preprocessing",
    "correlation",
)


@dataclass(frozen=True, slots=True)
class ChunkedColumn:
    """Exact online measurements for one column. Quartiles are not stored here."""

    label: object
    pandas_dtype: str
    count: int
    missing_count: int
    finite_count: int
    infinite_count: int
    unique_count: int
    minimum: float | None = None
    maximum: float | None = None
    sum: float | None = None
    mean: float | None = None

    @property
    def name(self) -> str:
        return str(self.label)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label_type": type(self.label).__name__,
            "pandas_dtype": self.pandas_dtype,
            "count": self.count,
            "missing_count": self.missing_count,
            "finite_count": self.finite_count,
            "infinite_count": self.infinite_count,
            "unique_count": self.unique_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "sum": self.sum,
            "mean": self.mean,
        }


@dataclass(frozen=True, slots=True)
class ChunkedApproximation:
    """A quartile of a bounded deterministic sample, not of the population."""

    column: str
    field: str
    method: str
    label: str
    value: float
    seed: int
    requested_size: int
    actual_size: int
    population_size: int

    def to_dict(self) -> dict[str, object]:
        return {
            "column": self.column,
            "field": self.field,
            "method": self.method,
            "label": self.label,
            "value": self.value,
            "seed": self.seed,
            "requested_size": self.requested_size,
            "actual_size": self.actual_size,
            "population_size": self.population_size,
        }


@dataclass(frozen=True, slots=True)
class ChunkedProfile:
    """What a bounded chunked CSV/TSV scan measured.

    ``population_fingerprint`` covers every row. ``temporary_storage`` is
    ``removed`` only after the scratch directory has been deleted.
    ``full_population_scanned`` is true only when every validated row was
    visited. Quartiles live in ``approximations``.
    """

    format: str
    chunk_rows: int
    rows_scanned: int
    population_rows: int
    population_columns: int
    full_population_scanned: bool
    bounded_memory: bool
    exact_metrics: tuple[str, ...]
    unavailable_metrics: tuple[str, ...]
    population_fingerprint: str
    schema_fingerprint: str
    fingerprint_algorithm: str
    fingerprint_scope: str
    sampling_method: str
    sampling_seed: int
    sampling_requested_size: int
    encoding: str
    delimiter: str | None
    delimiter_source: str | None
    header: bool
    ingestion_warnings: tuple[str, ...]
    effective_limits: tuple[tuple[str, int | None], ...]
    duplicate_row_count: int
    columns: tuple[ChunkedColumn, ...]
    approximations: tuple[ChunkedApproximation, ...]
    temporary_storage: str

    @property
    def mode(self) -> str:
        return _MODE

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "format": self.format,
            "chunk_rows": self.chunk_rows,
            "rows_scanned": self.rows_scanned,
            "population_rows": self.population_rows,
            "population_columns": self.population_columns,
            "full_population_scanned": self.full_population_scanned,
            "bounded_memory": self.bounded_memory,
            "exact_metrics": list(self.exact_metrics),
            "approximate_metrics": [item.to_dict() for item in self.approximations],
            "unavailable_metrics": list(self.unavailable_metrics),
            "sampling": {
                "method": self.sampling_method,
                "seed": self.sampling_seed,
                "requested_size": self.sampling_requested_size,
            },
            "population_fingerprint": self.population_fingerprint,
            "schema_fingerprint": self.schema_fingerprint,
            "fingerprint_algorithm": self.fingerprint_algorithm,
            "fingerprint_scope": self.fingerprint_scope,
            "duplicate_row_count": self.duplicate_row_count,
            "columns": [column.to_dict() for column in self.columns],
            "temporary_storage": self.temporary_storage,
        }


def profile_delimited_chunks(
    source: str | Path,
    *,
    options: LoadOptions | None = None,
    limits: IngestionLimits | None = None,
    chunk_rows: int = 10_000,
    quantile_sample_size: int | None = None,
) -> ChunkedProfile:
    """Profile a CSV or TSV file in bounded memory, without replacing the default audit.

    Args:
        source: Path of a ``.csv`` or ``.tsv`` file.
        options: Encoding, delimiter and header. The same object
            :func:`~aidatasetkit.ingestion.load_table` accepts.
        limits: Resource limits. Defaults to :class:`IngestionLimits`. An
            over-limit file is refused before a scratch directory is created.
        chunk_rows: How many data rows are held at once. Must be at least 1.
        quantile_sample_size: Capacity of the deterministic quartile reservoir.
            Quartiles are always approximate. ``None`` uses 4096. The minimum,
            maximum, sum and mean stay exact.

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
    _check_chunk_settings(chunk_rows, quantile_sample_size)
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
    plan = plan_delimited(path, fmt, options, _cell_limited(limits))

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
            sample_size=_sample_size(quantile_sample_size),
        )
    finally:
        if directory is not None:
            shutil.rmtree(directory)
    return ChunkedProfile(
        temporary_storage="removed",
        encoding=plan.encoding,
        delimiter=None if plan.single_column else plan.delimiter,
        delimiter_source=None if plan.single_column else plan.delimiter_source.value,
        header=options.header,
        ingestion_warnings=plan.warnings,
        effective_limits=_limit_pairs(limits),
        **scanned,
    )


def _cell_limited(limits: IngestionLimits) -> IngestionLimits:
    return replace(limits)


def _limit_pairs(limits: IngestionLimits) -> tuple[tuple[str, int | None], ...]:
    names = (
        "max_source_bytes",
        "max_rows",
        "max_columns",
        "max_cells",
        "max_field_length",
        "max_records",
        "max_keys_per_record",
        "max_record_chars",
    )
    return tuple((name, getattr(limits, name)) for name in names)


def _sample_size(requested: int | None) -> int:
    if requested is None:
        return _QUANTILE_SAMPLE_SIZE
    return requested


def _check_chunk_settings(chunk_rows: int, quantile_sample_size: int | None) -> None:
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise InvalidIngestionOptionsError(
            f"chunk_rows must be a positive integer, got {chunk_rows!r}."
        )
    if quantile_sample_size is None:
        return
    if (
        isinstance(quantile_sample_size, bool)
        or not isinstance(quantile_sample_size, int)
        or quantile_sample_size < 1
    ):
        raise InvalidIngestionOptionsError(
            "quantile_sample_size must be a positive integer or None, "
            f"got {quantile_sample_size!r}."
        )


def _iter_chunks(
    path: Path,
    plan,
    options: LoadOptions,
    chunk_rows: int,
    *,
    as_text: bool = False,
):
    kwargs: dict[str, object] = {}
    if as_text:
        # Raw field text, so a chunk of one empty cell is not inferred as float.
        kwargs["dtype"] = str
        kwargs["na_filter"] = False
        kwargs["keep_default_na"] = False
    reader = pd.read_csv(
        path,
        sep=plan.delimiter,
        encoding=plan.encoding,
        header=0 if options.header else None,
        skip_blank_lines=True,
        chunksize=chunk_rows,
        **kwargs,
    )
    if type(reader).__name__ != "TextFileReader":
        raise RuntimeError("chunked profiling read the file without a chunk size")
    return reader


_NA_TOKENS = frozenset(
    {
        "",
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "-NaN",
        "-nan",
        "1.#IND",
        "1.#QNAN",
        "<NA>",
        "N/A",
        "NA",
        "NULL",
        "NaN",
        "None",
        "n/a",
        "nan",
        "null",
    }
)
_INT_TOKEN = re.compile(r"[+-]?\d+\Z")
_FLOAT_TOKEN = re.compile(
    r"[+-]?(?:\d+\.\d*|\.\d+|\d+[eE][+-]?\d+|\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+[eE][+-]?\d+)\Z"
)
_INF_TOKENS = frozenset(
    {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
)
_native_dtypes: dict[str, object] | None = None


def _native_dtype(kind: str) -> object:
    """The dtype this pandas build stores for a column of ``kind``.

    Pandas 3 reads text as a string dtype. Pandas 2 reads it as object. A
    boolean column that also has missing values stays object on both, because
    that is what ``read_csv`` stores. No file is read here.
    """
    global _native_dtypes
    if _native_dtypes is None:
        text = (
            pd.StringDtype(na_value=np.nan)
            if int(pd.__version__.split(".", 1)[0]) >= 3
            else np.dtype(object)
        )
        _native_dtypes = {
            "int": np.dtype("int64"),
            "float": np.dtype("float64"),
            "bool": np.dtype(bool),
            "string": text,
            "object": np.dtype(object),
        }
    return _native_dtypes[kind]


@dataclass
class _Kinds:
    missing: bool = False
    boolean: bool = False
    integer: bool = False
    floating: bool = False
    text: bool = False


def _token_kind(token: str) -> str:
    if token in _NA_TOKENS:
        return "missing"
    lowered = token.lower()
    if lowered in {"true", "false"}:
        return "bool"
    if lowered in _INF_TOKENS:
        return "float"
    if _INT_TOKEN.fullmatch(token):
        return "int"
    if _FLOAT_TOKEN.fullmatch(token):
        return "float"
    return "text"


def _resolve_dtype(kinds: _Kinds) -> object:
    if kinds.text or (kinds.boolean and (kinds.integer or kinds.floating)):
        return _native_dtype("string")
    if kinds.boolean and kinds.missing:
        return _native_dtype("object")
    if kinds.boolean:
        return _native_dtype("bool")
    if kinds.floating or (kinds.integer and kinds.missing):
        return _native_dtype("float")
    if kinds.integer:
        return _native_dtype("int")
    return _native_dtype("float")


def _infer_dtypes(path: Path, plan, options: LoadOptions, chunk_rows: int):
    """Column dtypes for the whole file, independent of how the rows are grouped."""
    kinds: dict[object, _Kinds] = {}
    order: list[object] = []
    for chunk in _iter_chunks(path, plan, options, chunk_rows, as_text=True):
        for label in chunk.columns:
            state = kinds.get(label)
            if state is None:
                state = _Kinds()
                kinds[label] = state
                order.append(label)
            for token in chunk[label].tolist():
                kind = _token_kind(token if isinstance(token, str) else str(token))
                if kind == "missing":
                    state.missing = True
                elif kind == "bool":
                    state.boolean = True
                elif kind == "int":
                    state.integer = True
                elif kind == "float":
                    state.floating = True
                else:
                    state.text = True
    return order, {label: _resolve_dtype(kinds[label]) for label in order}


def _private(path: Path) -> None:
    os.chmod(path, 0o600)


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (str, bytes, bool, np.bool_)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _encode_cell(value: object) -> bytes:
    """Length-prefixed identity of one cell. Distinct values stay distinct."""
    if isinstance(value, np.generic):
        value = value.item()
    if _is_missing(value):
        return b"\x00"
    if isinstance(value, bool):
        return b"\x01" + (b"\x01" if value else b"\x00")
    if isinstance(value, int):
        payload = str(value).encode("ascii")
        return b"\x02" + struct.pack("<Q", len(payload)) + payload
    if isinstance(value, float):
        return b"\x03" + struct.pack("<d", value)
    if isinstance(value, str):
        payload = value.encode("utf-8")
        return b"\x04" + struct.pack("<Q", len(payload)) + payload
    if isinstance(value, bytes):
        return b"\x05" + struct.pack("<Q", len(value)) + value
    payload = str(value).encode("utf-8")
    return b"\x06" + struct.pack("<Q", len(payload)) + payload


def _row_key(values: tuple[object, ...]) -> bytes:
    return b"".join(_encode_cell(value) for value in values)


def _update_from_file(digest: "hashlib._Hash", path: Path) -> None:
    with path.open("rb") as handle:
        while block := handle.read(_READ_BLOCK):
            digest.update(block)


class _Reservoir:
    """Algorithm R. The sample never grows past ``cap`` values."""

    def __init__(self, cap: int, seed: int) -> None:
        self.cap = cap
        self.values: list[float] = []
        self.seen = 0
        self._rng = random.Random(seed)

    def add(self, value: float) -> None:
        self.seen += 1
        if len(self.values) < self.cap:
            self.values.append(value)
            return
        index = self._rng.randrange(self.seen)
        if index < self.cap:
            self.values[index] = value


@dataclass
class _Running:
    count: int = 0
    missing: int = 0
    finite: int = 0
    infinite: int = 0
    minimum: float | None = None
    maximum: float | None = None
    total: float = 0.0


class _ColumnWindow:
    """Per-column hash bytes on disk, one bounded file per fingerprint window.

    The parent digest is updated later, column by column, so a window that fills
    during the scan does not interleave one column's bytes with the next.
    """

    def __init__(self, directory: Path, index: int, label: object, is_object: bool) -> None:
        self.directory = directory
        self.index = index
        self.label = label
        self.is_object = is_object
        self._window = 0
        self.pending = 0
        self._type_count = 0
        self._hash_paths: list[Path] = []
        self._type_paths: list[Path] = []
        self._hash: object | None = None
        self._type: object | None = None
        self._open_next()

    def add(self, series: pd.Series) -> None:
        start = 0
        count = len(series)
        while start < count:
            room = _FINGERPRINT_CHUNK_ROWS - self.pending
            take = min(room, count - start)
            piece = series.iloc[start : start + take]
            hashed = pd.util.hash_pandas_object(piece, index=False)
            self._hash.write(hashed.to_numpy(dtype="uint64").tobytes())
            if self.is_object:
                self._write_types(piece)
            self.pending += take
            start += take
            if self.pending == _FINGERPRINT_CHUNK_ROWS:
                self._close_current()
                self._window += 1
                self._open_next()

    def finish(self) -> None:
        if self.pending == 0 and self._window > 0:
            self._close_current()
            self._hash_paths.pop().unlink()
            if self.is_object:
                self._type_paths.pop().unlink()
            return
        self._close_current()

    def fold(self, digest: "hashlib._Hash") -> None:
        digest.update(b"\x00")
        digest.update(_label_token(self.label).encode("utf-8"))
        digest.update(b"\x00")
        for position, hash_path in enumerate(self._hash_paths):
            _update_from_file(digest, hash_path)
            if self.is_object:
                _update_from_file(digest, self._type_paths[position])

    def close(self) -> None:
        self._close_current()

    def _open_next(self) -> None:
        hash_path = self.directory / f"hash-{self.index}-{self._window}.bin"
        self._hash = hash_path.open("wb")
        _private(hash_path)
        self._hash_paths.append(hash_path)
        if self.is_object:
            type_path = self.directory / f"type-{self.index}-{self._window}.bin"
            self._type = type_path.open("wb")
            _private(type_path)
            self._type_paths.append(type_path)
        self.pending = 0
        self._type_count = 0

    def _close_current(self) -> None:
        if self._hash is not None and not self._hash.closed:
            self._hash.close()
        if self._type is not None and not self._type.closed:
            self._type.close()

    def _write_types(self, series: pd.Series) -> None:
        assert self._type is not None
        for value in series:
            if self._type_count:
                self._type.write(b"\x1f")
            self._type.write(type(value).__name__.encode("utf-8"))
            self._type_count += 1


def _numeric(dtype: object) -> bool:
    return bool(pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype))


def _observe_numeric(running: _Running, series: pd.Series, reservoir: _Reservoir) -> None:
    values = series.to_numpy(dtype="float64", na_value=np.nan)
    missing = np.isnan(values)
    running.missing += int(missing.sum())
    infinite = np.isinf(values)
    running.infinite += int(infinite.sum())
    finite = values[np.isfinite(values)]
    running.finite += int(finite.size)
    running.count += int(finite.size) + int(infinite.sum())
    for item in finite:
        number = float(item)
        running.total += number
        if running.minimum is None or number < running.minimum:
            running.minimum = number
        if running.maximum is None or number > running.maximum:
            running.maximum = number
        reservoir.add(number)


def _observe_other(running: _Running, series: pd.Series) -> None:
    missing = int(series.isna().sum())
    running.missing += missing
    running.count += int(len(series)) - missing


def _scan(
    path: Path,
    plan,
    options: LoadOptions,
    directory: Path,
    *,
    fmt: TableFormat,
    chunk_rows: int,
    sample_size: int,
) -> dict[str, object]:
    columns, dtypes = _infer_dtypes(path, plan, options, chunk_rows)
    if len(columns) != plan.columns:
        raise MalformedInputError(
            f"{path.name}: chunked profiling saw {len(columns)} columns, not the "
            f"{plan.columns} the validating parser counted."
        )

    database = directory / "rows.sqlite"
    connection = sqlite3.connect(database)
    _private(database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-64")
    connection.execute(
        "CREATE TABLE cells (col INTEGER NOT NULL, key BLOB NOT NULL, PRIMARY KEY (col, key))"
    )
    connection.execute("CREATE TABLE rowkeys (key BLOB PRIMARY KEY)")

    digest = hashlib.sha256()
    digest.update(_schema_fingerprint(columns, dtypes, plan.data_rows).encode("ascii"))
    windows = [
        _ColumnWindow(directory, index, label, dtypes[label] == object)
        for index, label in enumerate(columns)
    ]
    running = {label: _Running() for label in columns}
    reservoirs = {
        label: _Reservoir(sample_size, _QUANTILE_SEED)
        for label in columns
        if _numeric(dtypes[label])
    }
    seen_rows = 0
    try:
        for chunk in _iter_chunks(path, plan, options, chunk_rows):
            cast = chunk.astype(dtypes)
            if list(cast.columns) != list(columns):
                raise MalformedInputError(
                    f"{path.name}: a chunk did not have the columns of the first chunk."
                )
            for index, label in enumerate(columns):
                series = cast[label]
                windows[index].add(series)
                if label in reservoirs:
                    _observe_numeric(running[label], series, reservoirs[label])
                else:
                    _observe_other(running[label], series)
                present = [
                    (index, _encode_cell(value))
                    for value in series.tolist()
                    if not _is_missing(value)
                ]
                connection.executemany(
                    "INSERT OR IGNORE INTO cells(col, key) VALUES (?, ?)",
                    present,
                )
            rows = [
                (_row_key(row),)
                for row in cast.itertuples(index=False, name=None)
            ]
            connection.executemany(
                "INSERT OR IGNORE INTO rowkeys(key) VALUES (?)",
                rows,
            )
            seen_rows += len(cast)
        connection.commit()
        for window in windows:
            window.finish()
        for window in windows:
            window.fold(digest)
        scanned = seen_rows == plan.data_rows
        if not scanned:
            raise MalformedInputError(
                f"{path.name}: chunked profiling saw {seen_rows} data rows, "
                f"not the {plan.data_rows} the validating parser counted."
            )
        distinct_rows = int(connection.execute("SELECT COUNT(*) FROM rowkeys").fetchone()[0])
        unique_counts = {
            int(column_index): int(count)
            for column_index, count in connection.execute(
                "SELECT col, COUNT(*) FROM cells GROUP BY col"
            )
        }
    finally:
        connection.close()
        for window in windows:
            window.close()

    approximations: list[ChunkedApproximation] = []
    column_records: list[ChunkedColumn] = []
    for index, label in enumerate(columns):
        state = running[label]
        minimum = maximum = total = mean = None
        if label in reservoirs and state.finite:
            minimum = state.minimum
            maximum = state.maximum
            total = state.total
            mean = state.total / state.finite
            quartiles = StatisticsEngine(
                np.asarray(reservoirs[label].values, dtype=np.float64)
            ).quartiles()
            for field, value in (
                ("q25", quartiles.q1),
                ("median", quartiles.q2),
                ("q75", quartiles.q3),
            ):
                approximations.append(
                    ChunkedApproximation(
                        column=str(label),
                        field=field,
                        method=_QUANTILE_METHOD,
                        label=_APPROXIMATION_LABEL,
                        value=float(value),
                        seed=_QUANTILE_SEED,
                        requested_size=sample_size,
                        actual_size=len(reservoirs[label].values),
                        population_size=reservoirs[label].seen,
                    )
                )
        column_records.append(
            ChunkedColumn(
                label=label,
                pandas_dtype=str(dtypes[label]),
                count=state.count,
                missing_count=state.missing,
                finite_count=state.finite,
                infinite_count=state.infinite,
                unique_count=unique_counts.get(index, 0),
                minimum=minimum,
                maximum=maximum,
                sum=total,
                mean=mean,
            )
        )
    return {
        "format": fmt.value,
        "chunk_rows": chunk_rows,
        "rows_scanned": seen_rows,
        "population_rows": plan.data_rows,
        "population_columns": len(columns),
        "full_population_scanned": scanned,
        "bounded_memory": True,
        "exact_metrics": EXACT_METRICS,
        "unavailable_metrics": UNAVAILABLE_METRICS,
        "population_fingerprint": digest.hexdigest(),
        "schema_fingerprint": _schema_fingerprint(columns, dtypes, plan.data_rows),
        "fingerprint_algorithm": _FINGERPRINT_ALGORITHM,
        "fingerprint_scope": _FINGERPRINT_SCOPE,
        "sampling_method": _QUANTILE_METHOD,
        "sampling_seed": _QUANTILE_SEED,
        "sampling_requested_size": sample_size,
        "duplicate_row_count": seen_rows - distinct_rows,
        "columns": tuple(column_records),
        "approximations": tuple(approximations),
    }


def _schema_fingerprint(columns, dtypes, row_count: int) -> str:
    parts = [f"rows={row_count}", f"columns={len(columns)}"]
    for position, label in enumerate(columns):
        parts.append(f"{position}|{_label_token(label)}|{dtypes[label]!s}")
    return _digest_of(*parts)


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
