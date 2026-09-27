"""Frequency distributions over values of any type.

:class:`FrequencyTable` is the one part of this package that does *not* route its
input through :func:`~aidatasetkit.core.arrays.to_float_array`. Counting
``"male"`` and ``"female"`` is a legitimate frequency analysis, so the values stay
as they are. Only :meth:`FrequencyTable.weighted_mean` needs numbers, and it asks
for the conversion at that point -- and reports clearly when the values cannot
supply it.

Rows are ordered by value rather than by count, because a cumulative frequency is
only meaningful along an ordering of the values themselves.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from aidatasetkit.core.arrays import to_float_array
from aidatasetkit.core.counting import value_counts
from aidatasetkit.core.exceptions import (
    EmptyDataError,
    MissingValueError,
    NonNumericDataError,
    ShapeError,
    ValidationError,
)

__all__ = ["FrequencyTable", "FrequencyNanPolicy", "FrequencySort"]

#: How a frequency table treats missing values.
#:
#: This extends the two-value policy used elsewhere with ``"include"``, because
#: "how often is this field missing?" is a question a frequency table is expected
#: to answer, whereas it is never a sensible input to a mean.
FrequencyNanPolicy = Literal["raise", "omit", "include"]

#: Row ordering. ``"value"`` is required for cumulative columns to mean anything.
FrequencySort = Literal["value", "count"]

_VALID_POLICIES = ("raise", "omit", "include")
_VALID_SORTS = ("value", "count")


class FrequencyTable:
    """Counts, proportions, and cumulative totals over a sample of any dtype.

    Args:
        data: A one-dimensional sample: a list, tuple, :class:`numpy.ndarray`,
            :class:`pandas.Series`, or :class:`pandas.Index`. Values may be
            numeric, textual, boolean, or categorical.
        nan_policy: ``"raise"`` (the default) rejects missing values, ``"omit"``
            drops them, and ``"include"`` counts them as their own category.
        sort: ``"value"`` (the default) orders rows by value ascending, which is
            what makes the cumulative columns meaningful. ``"count"`` orders by
            descending frequency, with ties broken by value.

    Raises:
        ValidationError: If an argument is invalid or the input is unusable.
        EmptyDataError: If no observations remain.
        MissingValueError: If values are missing and ``nan_policy="raise"``.

    Example:
        >>> table = FrequencyTable(["a", "b", "a", "c", "a"])
        >>> table.counts()
        {'a': 3, 'b': 1, 'c': 1}
        >>> table.percentage()["a"]
        60.0
    """

    def __init__(
        self,
        data: Any,
        *,
        nan_policy: FrequencyNanPolicy = "raise",
        sort: FrequencySort = "value",
    ) -> None:
        if nan_policy not in _VALID_POLICIES:
            raise ValidationError(
                f"nan_policy must be one of {list(_VALID_POLICIES)}, got {nan_policy!r}."
            )
        if sort not in _VALID_SORTS:
            raise ValidationError(
                f"sort must be one of {list(_VALID_SORTS)}, got {sort!r}."
            )

        series = self._as_series(data)
        missing = int(series.isna().sum())

        if missing:
            if nan_policy == "raise":
                raise MissingValueError(
                    f"data contains {missing} missing value(s). Pass "
                    'nan_policy="omit" to drop them or nan_policy="include" to '
                    "count them as a category."
                )
            if nan_policy == "omit":
                series = series.dropna()

        if series.empty:
            raise EmptyDataError("data contains no observations to count.")

        self._nan_policy = nan_policy
        self._sort = sort
        self._counts = self._build_counts(series, sort)

    @staticmethod
    def _as_series(data: Any) -> pd.Series:
        """Materialise the input as a one-dimensional pandas Series."""
        if isinstance(data, (str, bytes, dict, set, frozenset)):
            raise ValidationError(
                f"data must be a sequence of observations, got {type(data).__name__}."
            )
        if isinstance(data, pd.DataFrame):
            raise ShapeError(
                "data must be one-dimensional; a DataFrame was given. "
                "Select a single column first."
            )
        if isinstance(data, pd.Series):
            return data
        if isinstance(data, pd.Index):
            return pd.Series(data)

        array = np.asarray(data, dtype=object if not hasattr(data, "dtype") else None)
        if array.ndim != 1:
            raise ShapeError(
                f"data must be one-dimensional, got an array with {array.ndim} dimensions."
            )
        return pd.Series(array)

    @staticmethod
    def _build_counts(series: pd.Series, sort: FrequencySort) -> dict[Any, int]:
        """Count values and place them in the requested order."""
        counted = value_counts(series, dropna=False)
        pairs = [(_normalise(label), int(count)) for label, count in counted.items()]

        if sort == "count":
            try:
                pairs.sort(key=lambda pair: (-pair[1], pair[0]))
            except TypeError:
                pairs.sort(key=lambda pair: -pair[1])
        else:
            try:
                pairs.sort(key=lambda pair: pair[0])
            except TypeError:
                pairs.sort(key=lambda pair: str(pair[0]))

        return dict(pairs)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(total={self.total}, "
            f"n_distinct={self.n_distinct}, sort={self._sort!r})"
        )

    def __len__(self) -> int:
        return len(self._counts)

    # ------------------------------------------------------------------ #
    # Shape of the table
    # ------------------------------------------------------------------ #

    @property
    def total(self) -> int:
        """The number of observations counted."""
        return int(sum(self._counts.values()))

    @property
    def n_distinct(self) -> int:
        """The number of distinct values."""
        return len(self._counts)

    @property
    def values(self) -> tuple[Any, ...]:
        """The distinct values, in table order."""
        return tuple(self._counts)

    # ------------------------------------------------------------------ #
    # Frequencies
    # ------------------------------------------------------------------ #

    def counts(self) -> dict[Any, int]:
        """Return the absolute frequency of each value."""
        return dict(self._counts)

    def relative(self) -> dict[Any, float]:
        """Return the relative frequency of each value.

        .. math:: f_i = \\frac{n_i}{n}

        The proportions sum to ``1`` up to floating-point tolerance.
        """
        total = self.total
        return {value: count / total for value, count in self._counts.items()}

    def percentage(self) -> dict[Any, float]:
        """Return the percentage frequency of each value.

        The percentages sum to ``100`` up to floating-point tolerance.
        """
        total = self.total
        return {value: 100.0 * count / total for value, count in self._counts.items()}

    def cumulative(self) -> dict[Any, int]:
        """Return the running total of counts, in table order."""
        running = 0
        result: dict[Any, int] = {}
        for value, count in self._counts.items():
            running += count
            result[value] = running
        return result

    def cumulative_relative(self) -> dict[Any, float]:
        """Return the running total of relative frequencies, in table order.

        The final entry is ``1`` up to floating-point tolerance.
        """
        total = self.total
        return {value: count / total for value, count in self.cumulative().items()}

    # ------------------------------------------------------------------ #
    # Derived statistics
    # ------------------------------------------------------------------ #

    def weighted_mean(self) -> float:
        """Return the mean of the distribution, weighting each value by its count.

        .. math:: \\bar{x} = \\frac{\\sum x_i n_i}{\\sum n_i}

        This is the grouped-data mean of the original Java library, and it equals
        the arithmetic mean of the ungrouped sample.

        Returns:
            The frequency-weighted mean.

        Raises:
            NonNumericDataError: If the values are not numeric, or a missing value
                was counted as a category under ``nan_policy="include"``.
        """
        try:
            values = to_float_array(list(self._counts), name="frequency table values")
        except (NonNumericDataError, ValidationError) as error:
            raise NonNumericDataError(
                "A weighted mean requires numeric values, but this frequency table "
                f"counts values that are not numeric: {error}"
            ) from error

        weights = np.fromiter(self._counts.values(), dtype="float64", count=len(self._counts))
        return float(np.sum(values * weights) / np.sum(weights))

    def to_frame(self) -> pd.DataFrame:
        """Return the whole table as a dataframe, one row per distinct value."""
        return pd.DataFrame(
            {
                "value": list(self._counts),
                "count": list(self._counts.values()),
                "relative": list(self.relative().values()),
                "percentage": list(self.percentage().values()),
                "cumulative": list(self.cumulative().values()),
                "cumulative_relative": list(self.cumulative_relative().values()),
            }
        )

    def to_dict(self) -> dict[str, Mapping[str, Any]]:
        """Return a JSON-serialisable view of the table."""
        return {
            "counts": {str(value): count for value, count in self._counts.items()},
            "relative": {str(value): share for value, share in self.relative().items()},
            "percentage": {
                str(value): share for value, share in self.percentage().items()
            },
        }


def _normalise(label: Any) -> Any:
    """Convert a NumPy scalar label to its Python equivalent for clean keys."""
    if isinstance(label, np.generic):
        return label.item()
    return label
