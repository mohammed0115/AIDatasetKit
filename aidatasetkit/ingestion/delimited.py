"""Reading delimited text: detect the delimiter, prove the structure, then parse.

A delimited file is read in three steps, and each one can only refuse, never
repair:

1. **Detect** (unless the caller named a delimiter or the format fixes it). The
   first :data:`SAMPLE_CHARS` characters are parsed with :mod:`csv` once per
   supported delimiter, in strict mode, so quoting is honoured -- a ``;`` inside
   ``"a;b"`` is a value, not a separator. A delimiter qualifies when every
   non-blank record of the sample splits into the same number of fields, and
   that number is at least two. Exactly one qualifier is used. More than one is
   :class:`AmbiguousDelimiterError`: a guess would have a real chance of the
   wrong columns. None, with every record a single field under every delimiter,
   is a single-column table and says so in the warnings. None otherwise is
   :class:`MalformedInputError`.
2. **Validate** the whole file in one streaming pass with :mod:`csv` under the
   chosen delimiter: constant memory, every record counted, every field count
   checked against the first record's. This is the only way to see a *short*
   row -- pandas fills its missing trailing fields with NaN without a word.
   Duplicate header names are caught here, before pandas renames them.
3. **Parse** once with :func:`pandas.read_csv` under the same delimiter and
   encoding, then check that pandas found exactly the number of data rows the
   validation pass counted. Two parsers disagreeing is refused, not resolved.

The cost is two full passes over the file (the :mod:`csv` validation and the
pandas parse) plus a bounded sample; no second copy of the file is held in
memory, and nothing here streams the table itself -- the frame is loaded whole.

Blank lines -- lines with no characters at all -- are ignored, as pandas
ignores them. A line of spaces is a record, and in a table of two or more
columns it has the wrong number of fields and is refused.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

from aidatasetkit.core.exceptions import (
    AmbiguousDelimiterError,
    CellLimitError,
    ColumnLimitError,
    DuplicateHeadersError,
    EmptyInputError,
    EncodingError,
    FieldLengthLimitError,
    MalformedInputError,
    RowLimitError,
)
from aidatasetkit.ingestion.types import (
    DelimiterSource,
    IngestionLimits,
    LoadOptions,
    SUPPORTED_DELIMITERS,
    TableFormat,
)

__all__ = ["SAMPLE_CHARS", "DelimitedPlan", "plan_delimited"]

#: Characters read for detection. Large enough for hundreds of ordinary records,
#: small enough to cost nothing next to the full parse.
SAMPLE_CHARS = 64 * 1024

#: Bytes inspected for the byte-order mark and for binary content.
_PROBE_BYTES = 64 * 1024

_UTF8_BOM = b"\xef\xbb\xbf"

_NAMES = {",": "','", ";": "';'", "\t": "tab", "|": "'|'"}


@dataclass(frozen=True, slots=True)
class DelimitedPlan:
    """Everything the parse needs, established before pandas is called."""

    encoding: str
    delimiter: str
    delimiter_source: DelimiterSource
    single_column: bool
    data_rows: int
    columns: int
    warnings: tuple[str, ...]


def plan_delimited(
    path: Path,
    fmt: TableFormat,
    options: LoadOptions,
    limits: IngestionLimits | None = None,
) -> DelimitedPlan:
    """Detect and validate; return the plan the parse will follow.

    Raises:
        EmptyInputError, EncodingError, MalformedInputError,
        AmbiguousDelimiterError, DuplicateHeadersError.
    """
    limits = limits if limits is not None else IngestionLimits()
    with open(path, "rb") as handle:
        probe = handle.read(_PROBE_BYTES)
    if not probe:
        raise EmptyInputError(f"{path.name} is empty (0 bytes); there is no table to read.")
    if b"\x00" in probe:
        raise MalformedInputError(
            f"{path.name} contains NUL bytes, so it is binary content or text in an "
            f"unsupported encoding such as UTF-16, not a {fmt.value.upper()} file. "
            f"Supported encodings are utf-8, utf-8-sig, latin-1 and cp1256."
        )

    encoding = options.encoding
    notes: list[str] = []
    if encoding == "utf-8" and probe.startswith(_UTF8_BOM):
        encoding = "utf-8-sig"
        notes.append("A UTF-8 byte-order mark was found, so the file was read as utf-8-sig.")

    sample, complete = _read_sample(path, encoding)
    if not sample.strip() and complete:
        raise EmptyInputError(f"{path.name} contains only whitespace; there is no table to read.")

    delimiter, source, single = _choose(path, sample, complete, fmt, options)
    if single:
        notes.append(
            "No supported delimiter (',', ';', tab, '|') separates the records, so "
            "the file was read as a single column."
        )
    rows, columns = _validate(path, encoding, delimiter, options.header, limits)
    return DelimitedPlan(
        encoding=encoding,
        delimiter=delimiter,
        delimiter_source=source,
        single_column=single,
        data_rows=rows,
        columns=columns,
        warnings=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def _read_sample(path: Path, encoding: str) -> tuple[str, bool]:
    """The first :data:`SAMPLE_CHARS` characters, and whether that is the whole file."""
    try:
        with open(path, encoding=encoding, newline="", errors="strict") as handle:
            sample = handle.read(SAMPLE_CHARS)
            complete = handle.read(1) == ""
    except UnicodeDecodeError as error:
        raise _encoding_error(path, encoding, error) from None
    return sample, complete


@dataclass(frozen=True, slots=True)
class _Shape:
    widths: tuple[int, ...]
    broken: bool

    @property
    def consistent(self) -> bool:
        return not self.broken and bool(self.widths) and len(set(self.widths)) == 1 and self.widths[0] >= 2

    @property
    def single(self) -> bool:
        return not self.broken and bool(self.widths) and set(self.widths) == {1}


def _shape(sample: str, complete: bool, delimiter: str) -> _Shape:
    """Field counts of the sample's records under one delimiter.

    When the sample is not the whole file its last record may be cut short, so
    it is dropped -- and a quoting error at the very end is the cut, not a fault.
    """
    widths: list[int] = []
    broken = False
    try:
        for record in csv.reader(io.StringIO(sample), delimiter=delimiter, strict=True):
            if record:
                widths.append(len(record))
    except csv.Error:
        broken = complete
    if not complete and widths:
        widths.pop()
    return _Shape(tuple(widths), broken)


def _choose(
    path: Path, sample: str, complete: bool, fmt: TableFormat, options: LoadOptions
) -> tuple[str, DelimiterSource, bool]:
    """The delimiter to use, how it was chosen, and whether the table is one column."""
    shapes = {delimiter: _shape(sample, complete, delimiter) for delimiter in SUPPORTED_DELIMITERS}

    if options.delimiter is not None:
        # An explicit delimiter skips detection, not validation. Named ',' for a
        # file separated by ';' it would read one column of 'a;b;c' -- the very
        # misread this module exists to prevent -- so that is refused.
        chosen = shapes[options.delimiter]
        others = [d for d in SUPPORTED_DELIMITERS if d != options.delimiter and shapes[d].consistent]
        if chosen.single and others:
            raise MalformedInputError(
                f"{path.name} does not contain the requested delimiter "
                f"{_NAMES[options.delimiter]}, but it is consistently separated by "
                f"{_NAMES[others[0]]}. Reading it under {_NAMES[options.delimiter]} would "
                "produce one column; pass the right delimiter, or omit it to detect."
            )
        return options.delimiter, DelimiterSource.EXPLICIT, False

    if fmt is TableFormat.TSV:
        tab = shapes["\t"]
        if tab.consistent:
            return "\t", DelimiterSource.FORMAT, False
        others = [d for d in SUPPORTED_DELIMITERS if d != "\t" and shapes[d].consistent]
        if tab.single and others:
            raise MalformedInputError(
                f"{path.name} is named .tsv but its records are separated by "
                f"{_NAMES[others[0]]}, not tabs. Rename it .csv, or pass the delimiter explicitly."
            )
        if tab.single:
            return "\t", DelimiterSource.FORMAT, True
        return "\t", DelimiterSource.FORMAT, False  # validation reports the inconsistency

    qualifying = [d for d in SUPPORTED_DELIMITERS if shapes[d].consistent]
    if len(qualifying) == 1:
        return qualifying[0], DelimiterSource.DETECTED, False
    if len(qualifying) > 1:
        described = ", ".join(f"{_NAMES[d]} ({shapes[d].widths[0]} columns)" for d in qualifying)
        raise AmbiguousDelimiterError(
            f"{path.name} splits into a consistent table under more than one "
            f"delimiter: {described}. Choosing one would be a guess; name the "
            "delimiter explicitly."
        )
    if all(shape.single for shape in shapes.values()):
        return ",", DelimiterSource.DETECTED, True
    if all(shape.broken for shape in shapes.values()):
        raise MalformedInputError(
            f"{path.name} has malformed quoting under every supported delimiter: a "
            "quote character is left open or stray. Check the quotes."
        )
    # No delimiter gives every record the same width. Say what the most
    # plausible one found, so the message points at the broken rows.
    best = max(SUPPORTED_DELIMITERS, key=lambda d: sum(1 for w in shapes[d].widths if w > 1))
    widths = shapes[best].widths
    raise MalformedInputError(
        f"{path.name} has records with different numbers of fields under every "
        f"supported delimiter (under {_NAMES[best]}: between {min(widths, default=0)} "
        f"and {max(widths, default=0)} fields). Fix the rows, or name the delimiter "
        "explicitly to see which record breaks."
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _validate(
    path: Path,
    encoding: str,
    delimiter: str,
    header: bool,
    limits: IngestionLimits,
) -> tuple[int, int]:
    """One streaming pass: every record's width, the header, the row count.

    Returns:
        ``(data_rows, columns)``.
    """
    width: int | None = None
    records = 0
    names: list[str] = []
    anything = False
    try:
        with open(path, encoding=encoding, newline="", errors="strict") as handle:
            reader = csv.reader(handle, delimiter=delimiter, strict=True)
            try:
                for record in reader:
                    if not record:
                        continue
                    records += 1
                    if width is None:
                        width = len(record)
                        names = record
                        if limits.max_columns is not None and width > limits.max_columns:
                            raise ColumnLimitError(
                                f"{path.name} exceeds the column limit ({width} > {limits.max_columns}).",
                                limit_name="max_columns", configured_limit=limits.max_columns,
                                observed_value=width, input_kind="file",
                            )
                    elif len(record) != width:
                        raise MalformedInputError(
                            f"{path.name}: record {records} (ending on line {reader.line_num}) "
                            f"has {len(record)} field(s) under {_NAMES[delimiter]}, but the "
                            f"first record has {width}. Every row must have the same "
                            "number of fields."
                        )
                    data_rows = records - (1 if header else 0)
                    if limits.max_rows is not None and data_rows > limits.max_rows:
                        raise RowLimitError(
                            f"{path.name} exceeds the row limit ({data_rows} > {limits.max_rows}).",
                            limit_name="max_rows", configured_limit=limits.max_rows,
                            observed_value=data_rows, input_kind="file",
                        )
                    if limits.max_cells is not None and data_rows > limits.max_cells // width:
                        cells = data_rows * width
                        raise CellLimitError(
                            f"{path.name} exceeds the cell limit ({cells} > {limits.max_cells}).",
                            limit_name="max_cells", configured_limit=limits.max_cells,
                            observed_value=cells, input_kind="file",
                        )
                    if limits.max_field_length is not None:
                        longest = max((len(value) for value in record), default=0)
                        if longest > limits.max_field_length:
                            raise FieldLengthLimitError(
                                f"{path.name} exceeds the field-length limit.",
                                limit_name="max_field_length", configured_limit=limits.max_field_length,
                                observed_value=longest, input_kind="file",
                            )
                    if not anything and any(value.strip() for value in record):
                        anything = True
            except csv.Error as error:
                raise MalformedInputError(
                    f"{path.name}: malformed quoting near line {reader.line_num} ({error}). "
                    "Check for an unclosed or stray quote character."
                ) from None
    except UnicodeDecodeError as error:
        raise _encoding_error(path, encoding, error) from None

    if records == 0 or not anything:
        raise EmptyInputError(f"{path.name} contains no values; there is no table to read.")
    if header:
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise DuplicateHeadersError(
                f"{path.name} has duplicate column headers ({', '.join(duplicated)}). "
                "pandas would rename them to make them unique, and an audit of renamed "
                "columns would describe a dataset that does not exist. Give each column "
                "its own name first."
            )
        if records == 1:
            raise EmptyInputError(
                f"{path.name} has a header row and no data rows; there is nothing to audit."
            )
    return records - (1 if header else 0), width or 0


def _encoding_error(path: Path, encoding: str, error: UnicodeDecodeError) -> EncodingError:
    return EncodingError(
        f"{path.name} cannot be decoded as {encoding}: an invalid byte sequence at "
        f"byte offset {error.start}. Encodings are not guessed; pass the one the file "
        "was written in (supported: utf-8, utf-8-sig, latin-1, cp1256)."
    )
