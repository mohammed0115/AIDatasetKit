"""Reading JSON and JSONL text: an array of flat objects, or one object per line.

A ``.json`` file is a table when its top level is an array of flat objects --
one object per row, with scalar values. A ``.jsonl`` file (``.ndjson`` is the
same format under another name) is a table when every line is one such object.
Anything else -- a top-level object or scalar, a nested object or array inside
a record, the non-standard constants ``NaN`` and ``Infinity`` -- is refused
with a structured error rather than read into a guessed table. Duplicate keys
inside one object are refused too: JSON permits them and most parsers keep the
last one silently, which is exactly the quiet misread this library exists to
prevent.

The formats differ in one way that matters. A JSONL stream is checked as it is
read, so a file that crosses a limit is refused at the offending line without
reading the rest. A JSON document is a single value and must be parsed whole,
so its size is bounded by the loader's ``max_source_bytes`` preflight and its
records are checked after parsing. Both end at the same authority as records
passed from Python: the record, key and value-character budgets are enforced
while the records arrive, the column and cell budgets before the frame is
built, columns keep the order their keys first appeared in, and a key missing
from a record becomes a missing cell there.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from aidatasetkit.core.exceptions import (
    CellLimitError,
    EmptyInputError,
    EncodingError,
    InvalidIngestionOptionsError,
    KeyLimitError,
    MalformedInputError,
    RecordLimitError,
)
from aidatasetkit.ingestion.types import (
    IngestionLimits,
    LoadOptions,
    SourceKind,
    TableFormat,
)

__all__ = ["read_json_text"]

_UTF8_BOM = b"\xef\xbb\xbf"

#: Bytes inspected for the byte-order mark and for binary content.
_PROBE_BYTES = 64 * 1024


def read_json_text(
    path: Path,
    fmt: TableFormat,
    options: LoadOptions,
    limits: IngestionLimits,
) -> tuple[list[Mapping[str, Any]], list[str], str, tuple[str, ...]]:
    """Parse a JSON or JSONL file into checked records.

    Returns:
        ``(records, column names, encoding, warnings)``. The frame itself is
        built by the loader, the one authority every record source shares.

    Raises:
        EmptyInputError, EncodingError, InvalidIngestionOptionsError,
        MalformedInputError, RecordLimitError, KeyLimitError, CellLimitError.
    """
    _check_options(path, options)
    encoding, notes = _probe(path, options)
    if fmt is TableFormat.JSON:
        records, names = _collect(_document(path, encoding), path.name, limits)
    else:
        records, names = _collect(_iter_jsonl(path, encoding), path.name, limits)
    if not records:
        raise EmptyInputError(f"{path.name} has no records; there is no table to load.")
    return records, names, encoding, notes


def _check_options(path: Path, options: LoadOptions) -> None:
    if options.delimiter is not None:
        raise InvalidIngestionOptionsError(
            f"delimiter applies to delimited text; {path.name} is JSON, whose "
            "values are self-describing."
        )
    if not options.header:
        raise InvalidIngestionOptionsError(
            f"header=False applies to delimited text; {path.name} is JSON, whose "
            "objects name their own columns."
        )


def _probe(path: Path, options: LoadOptions) -> tuple[str, tuple[str, ...]]:
    """The encoding to decode with, and any note about how it was chosen."""
    with open(path, "rb") as handle:
        probe = handle.read(_PROBE_BYTES)
    if not probe:
        raise EmptyInputError(f"{path.name} is empty (0 bytes); there is no table to read.")
    if b"\x00" in probe:
        raise MalformedInputError(
            f"{path.name} contains NUL bytes, so it is binary content, not JSON text."
        )
    encoding = options.encoding
    notes: tuple[str, ...] = ()
    if encoding == "utf-8" and probe.startswith(_UTF8_BOM):
        encoding = "utf-8-sig"
        notes = ("A UTF-8 byte-order mark was found, so the file was read as utf-8-sig.",)
    return encoding, notes


# --------------------------------------------------------------------------- #
# The two document shapes
# --------------------------------------------------------------------------- #


def _document(path: Path, encoding: str) -> Iterator[tuple[str, Any]]:
    """The records of a whole JSON document, tagged by position."""
    try:
        text = path.read_text(encoding=encoding)
    except UnicodeDecodeError as error:
        raise EncodingError(
            f"{path.name} cannot be decoded as {encoding} (byte offset {error.start})."
        ) from None
    if not text.strip():
        raise EmptyInputError(
            f"{path.name} contains only whitespace; there is no table to read."
        )
    parsed = _loads(text, path.name)
    if not isinstance(parsed, list):
        raise MalformedInputError(
            f"{path.name}: the top level of a JSON table is an array of objects; "
            f"this document is {_kind(parsed)}."
        )
    if not parsed:
        raise EmptyInputError(f"{path.name} holds an empty array; there is no table to load.")
    return ((f"record {position}", record) for position, record in enumerate(parsed))


def _iter_jsonl(path: Path, encoding: str) -> Iterator[tuple[str, Any]]:
    """One ``(line label, value)`` pair per line, parsed as the file is read.

    Completely blank lines are ignored, as they are for CSV; a line of
    whitespace is not blank, and is not valid JSON.
    """
    try:
        with open(path, encoding=encoding, errors="strict") as handle:
            for number, line in enumerate(handle, start=1):
                if line in ("\n", "\r\n", ""):
                    continue
                yield f"line {number}", _loads(line, path.name, number)
    except UnicodeDecodeError as error:
        raise EncodingError(
            f"{path.name} cannot be decoded as {encoding} (byte offset {error.start})."
        ) from None


# --------------------------------------------------------------------------- #
# Parsing that never keeps a quiet wrong value
# --------------------------------------------------------------------------- #


class _DuplicateKey(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


class _NonStandardConstant(ValueError):
    def __init__(self, constant: str) -> None:
        super().__init__(constant)
        self.constant = constant


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    for key, _value in pairs:
        if key in seen:
            raise _DuplicateKey(key)
        seen.add(key)
    return dict(pairs)


def _no_constant(constant: str) -> Any:
    raise _NonStandardConstant(constant)


def _loads(text: str, name: str, line: int | None = None) -> Any:
    """One JSON value, with duplicate keys and non-standard constants refused."""
    try:
        return json.loads(
            text, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant
        )
    except json.JSONDecodeError as error:
        detail = (
            f"line {error.lineno}, column {error.colno}: {error.msg}"
            if line is None
            else error.msg
        )
        where = name if line is None else f"line {line} of {name}"
        raise MalformedInputError(f"{where} is not valid JSON ({detail}).") from None
    except _DuplicateKey as error:
        where = f"line {line} of {name}" if line is not None else f"a record in {name}"
        raise MalformedInputError(
            f"{where} repeats the key {error.key!r}. JSON permits duplicate keys "
            "and most parsers keep the last one silently; name each column once."
        ) from None
    except _NonStandardConstant as error:
        where = name if line is None else f"line {line} of {name}"
        raise MalformedInputError(
            f"{where} uses {error.constant}, which is not valid JSON. Write a "
            "string or null instead."
        ) from None


def _kind(value: Any) -> str:
    if isinstance(value, Mapping):
        return "an object"
    if isinstance(value, list):
        return "an array"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    return "a string"


# --------------------------------------------------------------------------- #
# Records, checked as they arrive
# --------------------------------------------------------------------------- #


def _collect(
    tagged: Iterator[tuple[str, Any]], name: str, limits: IngestionLimits
) -> tuple[list[Mapping[str, Any]], list[str]]:
    """Every record, checked as it arrives; a refusal stops the iteration.

    For a JSONL file that means the rest of the file is never read. For a JSON
    document the records were parsed together, and the checks run over the list.
    """
    records: list[Mapping[str, Any]] = []
    names: dict[str, None] = {}
    total_chars = 0
    for where, record in tagged:
        if limits.max_records is not None and len(records) >= limits.max_records:
            raise RecordLimitError(
                f"{name} exceeds the record limit (more than {limits.max_records} records).",
                limit_name="max_records", configured_limit=limits.max_records,
                observed_value=len(records) + 1, input_kind=SourceKind.FILE.value,
            )
        if not isinstance(record, Mapping):
            raise MalformedInputError(
                f"{name}: {where} is {_kind(record)}, not an object. Every record "
                "of a JSON table maps column names to values."
            )
        if limits.max_keys_per_record is not None and len(record) > limits.max_keys_per_record:
            raise KeyLimitError(
                f"{name}: {where} exceeds the key limit "
                f"({len(record)} > {limits.max_keys_per_record}).",
                limit_name="max_keys_per_record", configured_limit=limits.max_keys_per_record,
                observed_value=len(record), input_kind=SourceKind.FILE.value,
            )
        for key, value in record.items():
            if isinstance(value, (dict, list)):
                raise MalformedInputError(
                    f"{name}: {where} holds {_kind(value)} under {key!r}. "
                    "A JSON table is flat; restructure the file, or parse it "
                    "yourself and pass the records from Python."
                )
            names.setdefault(key, None)
        if limits.max_record_chars is not None:
            total_chars += sum(len(str(value)) for value in record.values())
            if total_chars > limits.max_record_chars:
                raise CellLimitError(
                    f"{name} exceeds the accumulated value-character limit.",
                    limit_name="max_record_chars", configured_limit=limits.max_record_chars,
                    observed_value=total_chars, input_kind=SourceKind.FILE.value,
                )
        records.append(record)
    return records, list(names)
