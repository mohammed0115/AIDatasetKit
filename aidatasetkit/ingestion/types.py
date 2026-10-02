"""The typed contract of table loading: what may be asked, and what comes back.

Plain dataclasses and enums; the parsing lives in :mod:`.delimited` and :mod:`.loader`.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from aidatasetkit.core.exceptions import InvalidIngestionOptionsError

__all__ = [
    "DEFAULT_ENCODING",
    "DelimiterSource",
    "IngestionLimits",
    "LoadMetadata",
    "LoadOptions",
    "LoadedTable",
    "SUPPORTED_DELIMITERS",
    "SUPPORTED_ENCODINGS",
    "SourceKind",
    "TableFormat",
]


class SourceKind(StrEnum):
    """What the caller handed to :func:`~aidatasetkit.ingestion.load_table`."""

    FILE = "file"
    DATAFRAME = "dataframe"
    RECORDS = "records"


class TableFormat(StrEnum):
    """File formats this version reads. In-memory sources have none."""

    CSV = "csv"
    TSV = "tsv"
    JSON = "json"
    JSONL = "jsonl"


class DelimiterSource(StrEnum):
    """How the delimiter in force was arrived at."""

    #: Found by :mod:`aidatasetkit.ingestion.delimited`'s detection.
    DETECTED = "detected"
    #: Named by the caller in :class:`LoadOptions`.
    EXPLICIT = "explicit"
    #: Fixed by the format: a ``.tsv`` file is tab-separated.
    FORMAT = "format"


def _validate_limit(name: str, value: int | None) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
    ):
        raise InvalidIngestionOptionsError(
            f"{name} must be a positive integer or None to disable it."
        )


def _cells_exceed(rows: int, columns: int, max_cells: int) -> bool:
    """Whether ``rows * columns > max_cells``, in exact integer arithmetic.

    The one authority for the cell budget. Dividing the limit instead of
    multiplying the shape keeps the comparison exact for any logical shape,
    with no fixed-width wrap and no float rounding.
    """
    return rows > max_cells // columns


@dataclass(frozen=True, slots=True)
class IngestionLimits:
    """Finite resource policy applied before and during table materialization."""

    max_source_bytes: int | None = 64 * 1024 * 1024
    max_rows: int | None = 1_000_000
    max_columns: int | None = 1_000
    max_cells: int | None = 10_000_000
    max_field_length: int | None = 1_000_000
    max_records: int | None = 1_000_000
    max_keys_per_record: int | None = 1_000
    max_record_chars: int | None = 10_000_000

    def __post_init__(self) -> None:
        for name in (
            "max_source_bytes",
            "max_rows",
            "max_columns",
            "max_cells",
            "max_field_length",
            "max_records",
            "max_keys_per_record",
            "max_record_chars",
        ):
            _validate_limit(name, getattr(self, name))


#: Delimiters detection chooses among, and the only ones an option may name.
SUPPORTED_DELIMITERS: tuple[str, ...] = (",", ";", "\t", "|")

#: Encodings that may be named. Nothing is guessed: ``utf-8`` is the default,
#: a UTF-8 byte-order mark switches it to ``utf-8-sig``, and anything else must
#: be asked for by name. ``latin-1`` decodes every byte sequence, so choosing it
#: for a file in another encoding cannot fail -- it produces the wrong text.
SUPPORTED_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "latin-1", "cp1256")

DEFAULT_ENCODING = "utf-8"

#: Python's canonical codec name -> the label this library records.
_CANONICAL: dict[str, str] = {codecs.lookup(name).name: name for name in SUPPORTED_ENCODINGS}


@dataclass(frozen=True, slots=True)
class LoadOptions:
    """How a file is read. In-memory sources accept only the defaults.

    Attributes:
        encoding: One of :data:`SUPPORTED_ENCODINGS`; aliases Python recognises
            for them (``utf8``, ``iso-8859-1``, ...) are accepted and recorded
            under the canonical label.
        delimiter: ``None`` to detect, or one of :data:`SUPPORTED_DELIMITERS`
            to use that one. An explicit delimiter skips detection, never
            validation.
        header: Whether the first record names the columns. With ``False`` the
            columns are numbered from 0 and every record is data.
    """

    encoding: str = DEFAULT_ENCODING
    delimiter: str | None = None
    header: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.encoding, str):
            raise InvalidIngestionOptionsError(
                f"encoding must be a string, got {type(self.encoding).__name__}."
            )
        try:
            canonical = _CANONICAL.get(codecs.lookup(self.encoding).name)
        except LookupError:
            canonical = None
        if canonical is None:
            raise InvalidIngestionOptionsError(
                f"encoding {self.encoding!r} is not supported. Supported: "
                f"{', '.join(SUPPORTED_ENCODINGS)}. Encodings are never guessed; "
                "name the one the file was written in."
            )
        object.__setattr__(self, "encoding", canonical)
        if self.delimiter is not None and self.delimiter not in SUPPORTED_DELIMITERS:
            raise InvalidIngestionOptionsError(
                f"delimiter {self.delimiter!r} is not supported. Supported: "
                f"{', '.join(repr(d) for d in SUPPORTED_DELIMITERS)}, or None to detect."
            )
        if not isinstance(self.header, bool):
            raise InvalidIngestionOptionsError(
                f"header must be True or False, got {self.header!r}."
            )

    @property
    def is_default(self) -> bool:
        return self == LoadOptions()


@dataclass(frozen=True, slots=True)
class LoadMetadata:
    """What loading established about the source. Never a cell value or a path.

    Fields that do not apply to the source are ``None`` rather than a made-up
    value: a DataFrame has no encoding, and records have no delimiter.

    Attributes:
        source_kind: File, DataFrame or records.
        format: The file format, or ``None`` for an in-memory source.
        encoding: The encoding the file was decoded with, or ``None``.
        delimiter: The delimiter the file was split on; ``None`` for an
            in-memory source and for a file read as a single column.
        delimiter_source: How that delimiter was chosen, or ``None``.
        header: Whether the first record named the columns; ``None`` when not
            applicable.
        row_count: Rows of the resulting frame.
        column_count: Columns of the resulting frame.
        memory_bytes: ``frame.memory_usage(deep=True, index=True).sum()`` -- the
            pandas measurement including the index and object payloads.
            Deterministic for the same data under the same pandas and Python.
        warnings: Plain-language notes about how the source was read.
    """

    source_kind: SourceKind
    format: TableFormat | None
    encoding: str | None
    delimiter: str | None
    delimiter_source: DelimiterSource | None
    header: bool | None
    row_count: int
    column_count: int
    memory_bytes: int
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind.value,
            "format": None if self.format is None else self.format.value,
            "encoding": self.encoding,
            "delimiter": self.delimiter,
            "delimiter_source": None if self.delimiter_source is None else self.delimiter_source.value,
            "header": self.header,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "memory_bytes": self.memory_bytes,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class LoadedTable:
    """A loaded table and what loading established about it.

    Attributes:
        frame: The table. For a DataFrame source this is the caller's own object,
            returned as-is: loading reads it and never writes to it.
        metadata: How it was read.
    """

    frame: Any
    metadata: LoadMetadata = field(repr=False)
