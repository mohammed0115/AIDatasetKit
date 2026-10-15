"""Full-table profiling for supported native pandas datetime columns.

Published results are bounded, but distinct-coordinate sorting can require
O(n) auxiliary memory and O(n log n) time for n rows. This is not streaming
datetime profiling; file-backed full-table routes retain their existing
ingestion limits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aidatasetkit.core.exceptions import ConfigurationError, DomainError
from aidatasetkit.core.types import DatetimeSummary

__all__ = ["summarize_datetime"]

_SUPPORTED_UNITS = frozenset({"s", "ms", "us", "ns"})


def summarize_datetime(
    series: pd.Series, *, expected_cadence: str | None
) -> DatetimeSummary | None:
    """Summarize a native datetime series without changing source values."""
    if not isinstance(series.array, pd.arrays.DatetimeArray):
        return None

    unit = str(series.array.unit)
    if unit not in _SUPPORTED_UNITS:
        raise ConfigurationError(
            f"Datetime resolution {unit!r} is unsupported. Supported resolutions: "
            f"{sorted(_SUPPORTED_UNITS)}."
        )

    total_count = int(len(series))
    missing_count = int(series.isna().sum())
    valid_count = total_count - missing_count
    coordinates = np.unique(series.array.asi8[series.array.asi8 != np.iinfo("int64").min])
    distinct_valid_count = int(coordinates.size)
    duplicate_count = valid_count - distinct_valid_count
    minimum = series.min() if valid_count else None
    maximum = series.max() if valid_count else None

    deltas = [
        int(coordinates[index]) - int(coordinates[index - 1])
        for index in range(1, distinct_valid_count)
    ]
    interval_count = len(deltas)
    distinct_deltas = set(deltas)
    is_regular = None if not deltas else len(distinct_deltas) == 1
    observed_cadence = (
        _duration(deltas[0], unit) if is_regular is True else None
    )

    cadence_units = _cadence_units(expected_cadence, unit)
    missing_expected, gap_count, off_grid = _gap_counts(
        coordinates, cadence_units
    )
    timezone = getattr(series.dtype, "tz", None)

    return DatetimeSummary(
        total_count=total_count,
        valid_count=valid_count,
        missing_count=missing_count,
        distinct_valid_count=distinct_valid_count,
        duplicate_timestamp_count=duplicate_count,
        minimum=minimum.isoformat() if minimum is not None else None,
        maximum=maximum.isoformat() if maximum is not None else None,
        timezone=str(timezone) if timezone is not None else None,
        resolution=unit,
        interval_count=interval_count,
        distinct_spacing_count=len(distinct_deltas),
        minimum_spacing=_duration(min(deltas), unit) if deltas else None,
        maximum_spacing=_duration(max(deltas), unit) if deltas else None,
        is_regular=is_regular,
        observed_cadence=observed_cadence,
        expected_cadence=expected_cadence,
        missing_expected_observation_count=missing_expected,
        gap_count=gap_count,
        off_grid_observation_count=off_grid,
    )


def _duration(value: int, unit: str) -> str:
    try:
        return pd.Timedelta(value, unit=unit).isoformat()
    except (OverflowError, ValueError) as error:
        raise DomainError(
            f"Datetime spacing {value} {unit} units exceeds pandas' supported "
            "fixed-duration range; it cannot be serialized without overflow."
        ) from error


def _cadence_units(cadence: str | None, unit: str) -> int | None:
    if cadence is None:
        return None
    try:
        duration = pd.Timedelta(cadence)
        if pd.isna(duration) or duration <= pd.Timedelta(0):
            raise ValueError
        exact = duration.as_unit(unit, round_ok=False)
    except (OverflowError, TypeError, ValueError) as error:
        raise ConfigurationError(
            "expected_datetime_cadence must be a positive, finite fixed duration "
            "exactly representable at the datetime column's resolution."
        ) from error
    value = int(exact.asm8.view("i8"))
    if value <= 0 or value == np.iinfo("int64").min:
        raise ConfigurationError(
            "expected_datetime_cadence must be a positive duration representable "
            f"as {unit!r} without overflow."
        )
    return value


def _gap_counts(
    coordinates: np.ndarray, cadence: int | None
) -> tuple[int | None, int | None, int | None]:
    if cadence is None:
        return None, None, None
    if coordinates.size == 0:
        return 0, 0, 0

    anchor = int(coordinates[0])
    off_grid = 0
    missing = 0
    gaps = 0
    previous_grid_index = 0
    has_previous_on_grid = False
    for raw in coordinates:
        coordinate = int(raw)
        elapsed = coordinate - anchor
        grid_index, remainder = divmod(elapsed, cadence)
        if remainder:
            off_grid += 1
            continue

        if has_previous_on_grid:
            absent = grid_index - previous_grid_index - 1
            if absent:
                missing += absent
                gaps += 1
        previous_grid_index = grid_index
        has_previous_on_grid = True

    last_grid_index = (int(coordinates[-1]) - anchor) // cadence
    trailing_absent = last_grid_index - previous_grid_index
    if trailing_absent:
        missing += trailing_absent
        gaps += 1
    return missing, gaps, off_grid
