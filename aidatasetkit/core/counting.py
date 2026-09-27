"""Counting the values of a column without letting pandas convert them first.

``Series.value_counts`` builds an index from the distinct values, and on pandas
2.x building that index runs type inference that tries to cast every Python
``int`` to ``float64``. An integer beyond float64's range -- a 400-digit account
number read as text and converted, say -- does not fit, and the whole tally dies
with a bare ``OverflowError: int too large to convert to float`` that names no
column. pandas 3 no longer does this. Every layer that tallies raw user values
calls :func:`value_counts` here so the difference between the two majors stops at
this module.

The fallback is exact rather than approximate. Nothing is converted to float,
nothing is rounded, and the counted values are the caller's own objects, so a
400-digit integer comes back as that integer. The order matches pandas: most
frequent first, and ties in order of first appearance, which is what pandas 3
returns for the same input -- so the two majors produce the same tally, not two
different plausible ones.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = ["value_counts"]


def value_counts(series: pd.Series, *, dropna: bool = True) -> pd.Series:
    """``series.value_counts(dropna=dropna)`` in a fixed order, surviving huge ints.

    pandas' own tally is used whenever it works, so every count is pandas' count.
    Only the ``OverflowError`` that pandas 2.x raises while inferring the index
    type is caught; any other error propagates, and in particular the
    ``TypeError`` of an unhashable value still reaches callers that translate it
    into a message about the column.

    **The order is fixed here, not inherited.** pandas 2.1 sorts the counts with
    an unstable quicksort, and numpy dispatches that sort at runtime to a
    CPU-specific implementation (AVX-512 where available), so the order of *tied*
    values depends on the machine. The profiler records the first entry as a
    column's dominant value, and for a column of unique identifiers every value
    ties -- so the audit of one file differed between two CI runners with the same
    versions installed. Ties are therefore put in order of first appearance, which
    is what pandas 3 returns; a tally with no ties is returned untouched.
    """
    try:
        counted = series.value_counts(dropna=dropna)
    except OverflowError:
        return exact_tally(series, dropna=dropna)
    if len(counted) > 1 and counted.duplicated().any():
        counted = _ties_in_order_of_appearance(series, counted)
    return counted


class _Missing:
    """One key for every missing value, which never compares equal to itself."""


_MISSING = _Missing()


def _ties_in_order_of_appearance(series: pd.Series, counted: pd.Series) -> pd.Series:
    """Reorder ``counted``: count descending, then first appearance in ``series``.

    A label that never appears -- an unobserved category, counted as zero -- sorts
    after every label that does, in the order pandas listed it.
    """
    first_seen: dict[Any, int] = {}
    for position, value in enumerate(series.tolist()):
        key = _MISSING if _is_missing(value) else value
        if key not in first_seen:
            first_seen[key] = position
    unseen = len(first_seen)

    def rank(index: int) -> tuple[int, int, int]:
        label = counted.index[index]
        key = _MISSING if _is_missing(label) else label
        return (-int(counted.iloc[index]), first_seen.get(key, unseen), index)

    return counted.iloc[sorted(range(len(counted)), key=rank)]


def exact_tally(series: pd.Series, *, dropna: bool = True) -> pd.Series:
    """Count values in pure Python, keeping each value as the caller's object.

    Public for the test that holds it equal to pandas on ordinary data; callers
    use :func:`value_counts`.

    Raises:
        TypeError: If a value is unhashable, exactly as pandas would.
    """
    counts: dict[Any, int] = {}
    missing = 0
    for value in series.tolist():
        if _is_missing(value):
            missing += 1
            continue
        counts[value] = counts.get(value, 0) + 1

    pairs = list(counts.items())
    if missing and not dropna:
        pairs.append((float("nan"), missing))
    # ``sorted`` is stable, so equal counts keep first-appearance order.
    pairs.sort(key=lambda pair: -pair[1])

    index = pd.Index([value for value, _ in pairs], dtype=object, name=series.name)
    return pd.Series([count for _, count in pairs], index=index, dtype="int64", name="count")


def _is_missing(value: Any) -> bool:
    """Whether a single cell is a missing value, never raising on odd cells.

    ``pd.isna`` answers with an array for a list-valued cell; such a cell is not
    missing, and hashing it next is what raises, as it should.
    """
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
