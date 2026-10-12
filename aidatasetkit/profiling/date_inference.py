"""Strict ISO date and datetime recognition for textual columns.

Recognition is evidence. It does not convert values, does not read the host
locale or timezone, and does not change a DataFrame, a detected kind, or a
verdict. The grammar is anchored to the whole string. Calendar checks use the
standard-library ``datetime`` constructor, which rejects impossible dates
without consulting a locale.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime as _datetime

import pandas as pd

METHOD = "strict_iso_full_population_v1"
MINIMUM_NON_BLANK = 8
FORMAT_FAMILY = "strict_iso8601"

_DATE = re.compile(r"\A(?P<y>[0-9]{4})-(?P<m>[0-9]{2})-(?P<d>[0-9]{2})\Z")
_DATETIME = re.compile(
    r"\A(?P<y>[0-9]{4})-(?P<m>[0-9]{2})-(?P<d>[0-9]{2})"
    r"T(?P<hh>[0-9]{2}):(?P<mm>[0-9]{2}):(?P<ss>[0-9]{2})"
    r"(?:\.[0-9]+)?"
    r"(?P<zone>Z|[+-][0-9]{2}:[0-9]{2})?\Z"
)


@dataclass(frozen=True, slots=True)
class StrictIsoColumn:
    """Counts for one column that met the strict ISO rule."""

    column: str
    inferred_semantic_type: str
    non_blank_count: int
    blank_count: int
    matched_count: int
    invalid_count: int
    timezone_policy: str


@dataclass(frozen=True, slots=True)
class StrictIsoInference:
    """The fixed full-population policy and any columns that met it."""

    method: str
    population: str
    full_population_scanned: bool
    minimum_non_blank: int
    required_match_ratio: float
    conversion_performed: bool
    affects_detected_kind: bool
    affects_verdict: bool
    columns: tuple[StrictIsoColumn, ...]


def _host_locale_unused() -> None:
    """The grammar does not consult the process locale."""
    return


def _host_zone_unused() -> None:
    """The grammar does not consult the host timezone."""
    return


def _components_valid(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> bool:
    try:
        _datetime(year, month, day, hour, minute, second)
    except ValueError:
        return False
    return True


def _zone_policy(zone: str | None) -> str | None:
    """Return the recorded policy, or ``None`` when the offset is impossible.

    ``Z`` stays ``utc_z``. A numeric offset stays ``explicit_offset``. Neither
    form is converted to the host zone.
    """
    if zone is None:
        return "naive"
    if zone == "Z":
        return "utc_z"
    hours = int(zone[1:3])
    minutes = int(zone[4:6])
    if hours > 23 or minutes > 59:
        return None
    return "explicit_offset"


def _classify(text: str) -> tuple[str, str] | None:
    """Return ``(semantic type, timezone policy)`` or nothing."""
    _host_locale_unused()
    _host_zone_unused()
    timed = _DATETIME.fullmatch(text)
    if timed is not None:
        policy = _zone_policy(timed.group("zone"))
        if policy is None:
            return None
        if not _components_valid(
            int(timed.group("y")),
            int(timed.group("m")),
            int(timed.group("d")),
            int(timed.group("hh")),
            int(timed.group("mm")),
            int(timed.group("ss")),
        ):
            return None
        return ("datetime", policy)
    dated = _DATE.fullmatch(text)
    if dated is not None:
        if not _components_valid(int(dated.group("y")), int(dated.group("m")), int(dated.group("d"))):
            return None
        return ("date", "naive")
    return None  # strict ISO only; slash dates and identifiers are refused


def _is_blank(value: object) -> bool:
    """Null and ``""`` are blank. Whitespace is a value, and it is not stripped."""
    if value is None or value is pd.NA:
        return True
    if isinstance(value, str):
        return value == ""  # empty is blank; whitespace is not
    if isinstance(value, float) and value != value:
        return True
    return type(value).__name__ in {"NAType", "NaTType"}


def _is_textual(series: pd.Series) -> bool:
    """Textual and object columns are candidates. Numbers and datetimes are not."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return False
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return False
    if pd.api.types.is_string_dtype(series):
        return True
    return series.dtype == object


def _policy(policies: set[str]) -> str:
    if len(policies) == 1:
        return next(iter(policies))
    return "mixed"


def _examine_column(series: pd.Series, name: str) -> StrictIsoColumn | None:
    if not _is_textual(series):
        return None
    blank = 0
    non_blank = 0
    matched = 0
    invalid = 0
    blocked = False
    kinds: set[str] = set()
    policies: set[str] = set()
    examined = series
    for value in examined:
        if _is_blank(value):
            blank += 1
            continue
        if not isinstance(value, str):
            blocked = True
            continue
        non_blank += 1
        parsed = _classify(value)
        if parsed is None:
            invalid += 1
            continue
        matched += 1
        kinds.add(parsed[0])
        policies.add(parsed[1])
    if blocked or invalid or non_blank < MINIMUM_NON_BLANK or matched != non_blank:
        return None
    semantic = "datetime" if "datetime" in kinds else "date"
    return StrictIsoColumn(
        column=name,
        inferred_semantic_type=semantic,
        non_blank_count=non_blank,
        blank_count=blank,
        matched_count=matched,
        invalid_count=invalid,
        timezone_policy=_policy(policies),
    )


def infer_strict_iso(frame: pd.DataFrame) -> StrictIsoInference:
    """Scan every textual column of an accepted full table.

    The frame is read and not copied. A column is recorded only when every
    non-blank value is a valid strict ISO date or datetime and at least
    ``MINIMUM_NON_BLANK`` such values were seen.
    """
    found: list[StrictIsoColumn] = []
    for label in frame.columns:
        column = _examine_column(frame[label], str(label))
        if column is not None:
            found.append(column)
    return StrictIsoInference(
        method=METHOD,
        population="full",
        full_population_scanned=True,
        minimum_non_blank=MINIMUM_NON_BLANK,
        required_match_ratio=1.0,
        conversion_performed=False,
        affects_detected_kind=False,
        affects_verdict=False,
        columns=tuple(found),
    )
