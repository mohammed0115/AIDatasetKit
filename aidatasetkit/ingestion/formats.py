"""Which file formats exist, decided by suffix and nothing else."""

from __future__ import annotations

from pathlib import Path

from aidatasetkit.core.exceptions import UnsupportedFormatError
from aidatasetkit.ingestion.types import TableFormat

__all__ = ["SUPPORTED_SUFFIXES", "resolve_format"]

#: Suffix (lower-cased) -> format. ``.txt`` is deliberately absent: a text file
#: says nothing about being a table, and reading one as CSV would be a guess.
#: ``.ndjson`` is JSON Lines under its other name.
SUPPORTED_SUFFIXES: dict[str, TableFormat] = {
    ".csv": TableFormat.CSV,
    ".tsv": TableFormat.TSV,
    ".json": TableFormat.JSON,
    ".jsonl": TableFormat.JSONL,
    ".ndjson": TableFormat.JSONL,
}


def resolve_format(path: Path) -> TableFormat:
    """The format a path names by its suffix.

    Raises:
        UnsupportedFormatError: For any other suffix, naming the supported ones.
    """
    suffix = Path(path).suffix.lower()
    try:
        return SUPPORTED_SUFFIXES[suffix]
    except KeyError:
        raise UnsupportedFormatError(
            f"{suffix or 'a file without a suffix'} is not a supported format. "
            "Supported now: .csv, .tsv, .json, .jsonl and .ndjson files, or a "
            "pandas DataFrame or a list of records from Python. Excel and "
            "Parquet are not supported yet."
        ) from None
