"""Single source of truth for column-type detection.

Both the profiler and the feature detector classify columns. If each implemented
its own rules they would eventually disagree, and a model would silently train on
a different set of columns than the profile reported. They both call this module
instead.

Detection uses ``pandas.api.types`` predicates rather than ``select_dtypes``,
because dtype *names* are not stable across pandas versions: in pandas 3 string
columns are ``StringDtype`` rather than ``object``, so selecting on ``"object"``
is deprecated and will eventually miss every text column.

The same argument applies to *targets*, and the second half of this module exists
because it was not applied there soon enough. The task detector and the target
encoder each decided independently whether a target held quantities or labels,
and they disagreed: an integer column the detector called regression was label
encoded anyway, and a column of floats stored under ``object`` dtype was called
"non-numeric" by both. A quantity turned into class indices is not an error a
user can see -- every model still fits and every metric still returns a number --
so the two answers are computed here, once, and both components read them.
"""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from aidatasetkit.core.config import KitConfig
from aidatasetkit.core.exceptions import SchemaError, ValidationError
from aidatasetkit.core.types import ColumnKind, ColumnKinds

__all__ = [
    "detect_column_kinds",
    "detect_kind",
    "numeric_target_values",
    "target_holds_quantities",
]


def detect_kind(series: pd.Series) -> ColumnKind:
    """Classify a single column by dtype.

    The order of the checks matters. Booleans and complex numbers are both
    numeric to pandas, and pandas categoricals answer ``True`` to the string-dtype
    predicate, so the narrower kinds are tested first.
    """
    if pdt.is_bool_dtype(series):
        return ColumnKind.BOOLEAN
    if pdt.is_datetime64_any_dtype(series):
        return ColumnKind.DATETIME
    if isinstance(series.dtype, pd.CategoricalDtype):
        return ColumnKind.CATEGORICAL
    if pdt.is_complex_dtype(series):
        # Numeric to pandas, but every statistic here is a real-valued formula.
        # Left as NUMERIC, a mean or a standard deviation would be computed from
        # the real part alone and reported as if it described the column.
        return ColumnKind.OTHER
    if pdt.is_numeric_dtype(series):
        return ColumnKind.NUMERIC
    if pdt.is_string_dtype(series) or series.dtype == object:
        return ColumnKind.CATEGORICAL
    return ColumnKind.OTHER


def detect_column_kinds(frame: pd.DataFrame) -> ColumnKinds:
    """Classify every column of ``frame`` by structural kind.

    Args:
        frame: The dataframe to inspect. It is read, never modified.

    Returns:
        A :class:`~aidatasetkit.core.types.ColumnKinds` preserving column order.

    Raises:
        ValidationError: If ``frame`` is not a :class:`pandas.DataFrame`.
        SchemaError: If column labels are duplicated, since ``frame[label]`` would
            then return a frame rather than a series and every downstream
            component would misbehave.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValidationError(
            f"A pandas DataFrame is required, got {type(frame).__name__}."
        )

    labels = list(frame.columns)
    duplicates = sorted({str(label) for label in labels if labels.count(label) > 1})
    if duplicates:
        raise SchemaError(
            f"Duplicate column labels are not supported: {duplicates}. "
            "Rename or drop them before profiling."
        )

    return ColumnKinds(mapping={label: detect_kind(frame[label]) for label in labels})


# --------------------------------------------------------------------------- #
# Target semantics
# --------------------------------------------------------------------------- #


def numeric_target_values(series: pd.Series) -> np.ndarray | None:
    """Return a target's present values as ``float64``, or ``None`` if they are labels.

    Dtype is evidence about a target, not proof. A column of Python floats under
    ``object`` dtype -- which is what pandas produces from a mixed-type read, or
    from ``Decimal`` values, or from a list of numbers built in Python -- answers
    ``False`` to :func:`pandas.api.types.is_numeric_dtype` while holding nothing
    but quantities. Asking the *values* is what closes that gap.

    Booleans are excluded deliberately: pandas calls them numeric, and two
    booleans are a label rather than a quantity.

    Args:
        series: The target column. Read, never modified. Missing values are
            dropped before the answer is formed.

    Returns:
        The present values as a one-dimensional ``float64`` array, or ``None``
        when they are not real numbers at all.
    """
    if pdt.is_bool_dtype(series):
        return None

    present = series.dropna()
    if present.empty:
        return None

    if pdt.is_numeric_dtype(present):
        # Complex is numeric to pandas and meaningless as a regression target;
        # leaving it here would only change which error a caller eventually sees.
        if pdt.is_complex_dtype(present):
            return None
        return _as_float64(present)

    if present.dtype != object:
        return None

    # Every present value has to be a real number. ``numbers.Number`` covers
    # ``int``, ``float``, the numpy scalar types, ``Decimal`` and ``Fraction``;
    # ``bool`` is a subclass of ``int`` and is excluded above and again here.
    if not all(
        isinstance(value, numbers.Number) and not isinstance(value, (bool, complex))
        for value in present
    ):
        return None
    return _as_float64(present)


def _as_float64(present: pd.Series) -> np.ndarray | None:
    """Convert already-validated values, answering ``None`` if they will not fit.

    ``OverflowError`` is caught alongside the usual two because numpy answers a
    Python ``int`` beyond ``float64`` range with it, and it is a subclass of
    neither.
    """
    try:
        return np.asarray(present.to_numpy(), dtype="float64")
    except (TypeError, ValueError, OverflowError):
        return None


def target_holds_quantities(
    series: pd.Series, config: KitConfig | None = None
) -> bool:
    """Whether these target values are a quantity rather than a set of labels.

    This is the shared answer behind two questions that must never diverge: what
    task the detector infers, and whether the target encoder is allowed to turn
    the values into class indices. The rules are the detector's own, stated once
    here so that both components apply them identically:

    * values that are not real numbers are labels;
    * one distinct value is neither -- there is nothing to predict;
    * exactly two distinct values are a binary label, whatever they are spelled
      as, because two points do not describe a curve;
    * anything non-integral is a quantity;
    * whole numbers are a quantity once there are more distinct ones than
      ``task_detection_max_classes``.

    What is deliberately *not* here is the ambiguous middle -- a handful of
    repeated whole numbers, which could be classes, grades, or a count. This
    function answers ``False`` for those, because they are exactly the case the
    detector refuses to decide without a hint, and a caller reaching the encoder
    has already supplied that hint by calling it.

    Args:
        series: The target column. Read, never modified.
        config: Supplies ``task_detection_max_classes``. Defaults to the shared
            configuration, so the encoder and the detector agree without the
            encoder having to be handed one.

    Returns:
        ``True`` when the values are quantities that must not be label encoded.
    """
    values = numeric_target_values(series)
    if values is None or values.size == 0:
        return False

    unique_count = int(series.dropna().nunique())
    if unique_count <= 2:
        return False

    integral = bool(np.all(np.isfinite(values)) and np.all(values == np.floor(values)))
    if not integral:
        return True

    thresholds = config if config is not None else KitConfig()
    return unique_count > thresholds.task_detection_max_classes
