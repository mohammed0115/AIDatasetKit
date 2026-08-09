"""The single entry point for turning user input into a numeric array.

Every statistical function accepts lists, numpy arrays, and pandas series. Rather
than each of them re-implementing the coercion -- and each drifting in how it
treats missing values, infinities, and text -- they all call
:func:`to_float_array`.

The policy is deliberately strict. Missing values raise unless the caller opts
into dropping them, and infinities raise unless the caller opts into keeping
them, because a mean that silently returns ``nan`` or ``inf`` is a defect that
surfaces three steps later as an unexplainable model.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from aidatasetkit.core.exceptions import (
    EmptyDataError,
    MissingValueError,
    NonFiniteValueError,
    NonNumericDataError,
    ShapeError,
    ValidationError,
)

__all__ = ["NanPolicy", "to_float_array", "to_float_arrays"]

#: How :func:`to_float_array` reacts to missing values.
#:
#: ``"raise"``
#:     Any ``NaN`` or ``NA`` aborts with :class:`MissingValueError`. This is the
#:     default: dropping observations changes the answer, so it is opt-in.
#: ``"omit"``
#:     Missing values are removed before the calculation.
NanPolicy = Literal["raise", "omit"]

_VALID_POLICIES: tuple[str, ...] = get_args(NanPolicy)

#: Types that are iterable but are never valid numeric input.
_REJECTED_TYPES = (str, bytes, dict, set, frozenset)


def to_float_array(
    data: Any,
    *,
    nan_policy: NanPolicy = "raise",
    allow_inf: bool = False,
    name: str = "data",
) -> np.ndarray:
    """Coerce one-dimensional input into a finite ``float64`` array.

    Args:
        data: A list, tuple, :class:`numpy.ndarray`, :class:`pandas.Series`, or
            :class:`pandas.Index` of numeric values.
        nan_policy: ``"raise"`` to reject missing values, ``"omit"`` to drop them.
        allow_inf: Whether ``inf`` and ``-inf`` may survive into the result.
        name: Label for this input, used in error messages.

    Returns:
        A new one-dimensional ``float64`` array. The input is never modified.

    Raises:
        ValidationError: If ``nan_policy`` is unknown or ``data`` is a type that
            cannot hold a numeric sample.
        ShapeError: If ``data`` is not one-dimensional.
        NonNumericDataError: If the values cannot be read as numbers.
        EmptyDataError: If no observations remain.
        MissingValueError: If values are missing and ``nan_policy="raise"``.
        NonFiniteValueError: If values are infinite and ``allow_inf`` is false.
    """
    if nan_policy not in _VALID_POLICIES:
        raise ValidationError(
            f"nan_policy must be one of {list(_VALID_POLICIES)}, got {nan_policy!r}."
        )

    values = _as_float64(data, name)

    if values.ndim != 1:
        raise ShapeError(
            f"{name} must be one-dimensional, got an array with {values.ndim} dimensions "
            f"and shape {values.shape}."
        )
    if values.size == 0:
        raise EmptyDataError(f"{name} is empty; at least one observation is required.")

    missing = np.isnan(values)
    if missing.any():
        if nan_policy == "raise":
            raise MissingValueError(
                f"{name} contains {int(missing.sum())} missing value(s). "
                'Pass nan_policy="omit" to drop them, or impute them first.'
            )
        values = values[~missing]
        if values.size == 0:
            raise EmptyDataError(
                f"{name} contains only missing values; nothing remains after omitting them."
            )

    if not allow_inf:
        infinite = np.isinf(values)
        if infinite.any():
            raise NonFiniteValueError(
                f"{name} contains {int(infinite.sum())} infinite value(s). "
                "Pass allow_inf=True to keep them, or clean them first."
            )

    return values


def to_float_arrays(
    x: Any,
    y: Any,
    *,
    nan_policy: NanPolicy = "raise",
    allow_inf: bool = False,
    names: tuple[str, str] = ("x", "y"),
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce two paired inputs, applying the missing-value policy *jointly*.

    Paired statistics must not clean their inputs independently. Dropping missing
    values from ``x`` and from ``y`` separately would shift the two series
    relative to each other and silently correlate mismatched observations, so
    omission here removes whole pairs.

    Args:
        x: First input, one-dimensional and numeric.
        y: Second input, of the same length as ``x``.
        nan_policy: ``"raise"`` to reject missing values, ``"omit"`` to drop any
            pair in which either side is missing.
        allow_inf: Whether infinities may survive into the results.
        names: Labels for the two inputs, used in error messages.

    Returns:
        Two new one-dimensional ``float64`` arrays of equal length.

    Raises:
        ValidationError: If ``nan_policy`` is unknown or an input is unusable.
        ShapeError: If either input is not one-dimensional, or the lengths differ.
        EmptyDataError: If no complete pairs remain.
        MissingValueError: If a pair is incomplete and ``nan_policy="raise"``.
        NonFiniteValueError: If values are infinite and ``allow_inf`` is false.
    """
    if nan_policy not in _VALID_POLICIES:
        raise ValidationError(
            f"nan_policy must be one of {list(_VALID_POLICIES)}, got {nan_policy!r}."
        )

    x_values = _as_float64(x, names[0])
    y_values = _as_float64(y, names[1])

    for values, label in ((x_values, names[0]), (y_values, names[1])):
        if values.ndim != 1:
            raise ShapeError(
                f"{label} must be one-dimensional, got an array with {values.ndim} "
                f"dimensions and shape {values.shape}."
            )

    if x_values.size != y_values.size:
        raise ShapeError(
            f"{names[0]} and {names[1]} must have the same length, got "
            f"{x_values.size} and {y_values.size}."
        )
    if x_values.size == 0:
        raise EmptyDataError(
            f"{names[0]} and {names[1]} are empty; at least one pair is required."
        )

    missing = np.isnan(x_values) | np.isnan(y_values)
    if missing.any():
        if nan_policy == "raise":
            raise MissingValueError(
                f"{names[0]} and {names[1]} contain {int(missing.sum())} incomplete "
                'pair(s). Pass nan_policy="omit" to drop them, or impute them first.'
            )
        x_values = x_values[~missing]
        y_values = y_values[~missing]
        if x_values.size == 0:
            raise EmptyDataError(
                f"No complete pairs remain in {names[0]} and {names[1]} after omitting "
                "missing values."
            )

    if not allow_inf:
        for values, label in ((x_values, names[0]), (y_values, names[1])):
            infinite = np.isinf(values)
            if infinite.any():
                raise NonFiniteValueError(
                    f"{label} contains {int(infinite.sum())} infinite value(s). "
                    "Pass allow_inf=True to keep them, or clean them first."
                )

    return x_values, y_values


def _as_float64(data: Any, name: str) -> np.ndarray:
    """Convert supported containers into a ``float64`` array without copying twice."""
    if isinstance(data, _REJECTED_TYPES):
        raise ValidationError(
            f"{name} must be a sequence of numbers, got {type(data).__name__}."
        )
    if isinstance(data, pd.DataFrame):
        raise ShapeError(
            f"{name} must be one-dimensional; a DataFrame was given. "
            "Select a single column first."
        )

    if isinstance(data, (pd.Series, pd.Index)):
        return _from_pandas(data, name)

    try:
        array = np.asarray(data)
    except (TypeError, ValueError) as error:
        raise NonNumericDataError(f"{name} could not be read as an array: {error}") from error

    return _cast_numeric(array, name)


def _from_pandas(data: pd.Series | pd.Index, name: str) -> np.ndarray:
    """Convert a pandas container, mapping ``pd.NA`` onto ``nan``.

    Text-backed dtypes are refused rather than parsed. Pandas is willing to turn
    ``"1"`` into ``1.0``, but a numeric column stored as text is a data-quality
    finding the inspector should report, not something a statistics call should
    quietly repair.
    """
    dtype = data.dtype

    if isinstance(dtype, pd.CategoricalDtype):
        raise NonNumericDataError(
            f"{name} is a categorical column. Convert it to an explicit numeric "
            "representation first if its categories are meant to be measured."
        )
    if pdt.is_bool_dtype(dtype) or pdt.is_numeric_dtype(dtype):
        try:
            return np.asarray(data.to_numpy(dtype="float64", na_value=np.nan), dtype="float64")
        except (TypeError, ValueError) as error:
            raise NonNumericDataError(
                f"{name} has dtype {dtype!r}, which does not hold numeric values: {error}"
            ) from error
    if pdt.is_datetime64_any_dtype(dtype) or pdt.is_timedelta64_dtype(dtype):
        raise NonNumericDataError(
            f"{name} has dtype {dtype!r}. Datetime and timedelta values are not numeric "
            "input; convert them to an explicit numeric representation first."
        )
    if dtype == object:
        return _cast_numeric(np.asarray(data), name)

    raise NonNumericDataError(
        f"{name} has dtype {dtype!r}, which does not hold numeric values. "
        "Convert it to a numeric dtype first."
    )


def _cast_numeric(array: np.ndarray, name: str) -> np.ndarray:
    """Cast an already-materialised array to ``float64``, rejecting non-numbers."""
    if array.dtype.kind in "mM":
        raise NonNumericDataError(
            f"{name} has dtype {array.dtype!r}. Datetime and timedelta values are not "
            "numeric input; convert them to an explicit numeric representation first."
        )
    if array.dtype.kind in "SU" or _contains_text(array):
        raise NonNumericDataError(
            f"{name} contains text values; convert them to numbers first."
        )
    try:
        return array.astype("float64", copy=True)
    except (TypeError, ValueError) as error:
        raise NonNumericDataError(
            f"{name} contains values that are not numeric: {error}"
        ) from error


def _contains_text(array: np.ndarray) -> bool:
    """Report whether an object array holds any string or bytes element."""
    if array.dtype != object:
        return False
    return any(isinstance(value, (str, bytes)) for value in array.ravel().tolist())
