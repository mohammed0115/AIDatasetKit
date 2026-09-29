"""Ingestion: the one authority that turns a source into a table.

:func:`load_table` accepts a ``.csv`` or ``.tsv`` path, a pandas DataFrame, or a
list of records, and returns a :class:`LoadedTable` -- the frame and typed
:class:`LoadMetadata` about how it was read -- or raises an
:class:`~aidatasetkit.core.exceptions.IngestionError`. Every input is either
read correctly or refused; nothing is read into a wrong table without a word.

Not here, on purpose: profiling, quality checks, training, network access,
scraping, and every format other than CSV and TSV. Encodings are never guessed;
date inference and chunked reading are not implemented yet. Finite resource
limits are enforced by :func:`load_table` before analysis.

``load_table`` is resolved lazily (PEP 562), so importing the contracts does not
import the parsing modules. It does not avoid pandas: ``aidatasetkit.core`` already
imports it.
"""

from __future__ import annotations

from typing import Any

from aidatasetkit.core.exceptions import (
    AmbiguousDelimiterError,
    DuplicateHeadersError,
    EmptyInputError,
    EncodingError,
    IngestionError,
    InputNotFoundError,
    InvalidIngestionOptionsError,
    MalformedInputError,
    ResourceLimitError,
    FileSizeLimitError,
    RowLimitError,
    ColumnLimitError,
    CellLimitError,
    FieldLengthLimitError,
    RecordLimitError,
    KeyLimitError,
    UnsupportedFormatError,
)
from aidatasetkit.ingestion.formats import SUPPORTED_SUFFIXES, resolve_format
from aidatasetkit.ingestion.types import (
    DEFAULT_ENCODING,
    SUPPORTED_DELIMITERS,
    SUPPORTED_ENCODINGS,
    DelimiterSource,
    IngestionLimits,
    LoadedTable,
    LoadMetadata,
    LoadOptions,
    SourceKind,
    TableFormat,
)

__all__ = [
    "AmbiguousDelimiterError",
    "DEFAULT_ENCODING",
    "DelimiterSource",
    "IngestionLimits",
    "DuplicateHeadersError",
    "EmptyInputError",
    "EncodingError",
    "IngestionError",
    "InputNotFoundError",
    "InvalidIngestionOptionsError",
    "LoadMetadata",
    "LoadOptions",
    "LoadedTable",
    "MalformedInputError",
    "ResourceLimitError",
    "FileSizeLimitError",
    "RowLimitError",
    "ColumnLimitError",
    "CellLimitError",
    "FieldLengthLimitError",
    "RecordLimitError",
    "KeyLimitError",
    "SUPPORTED_DELIMITERS",
    "SUPPORTED_ENCODINGS",
    "SUPPORTED_SUFFIXES",
    "SourceKind",
    "TableFormat",
    "UnsupportedFormatError",
    "load_table",
    "resolve_format",
]


def __getattr__(name: str) -> Any:
    if name == "load_table":
        from aidatasetkit.ingestion.loader import load_table

        return load_table
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
