"""Single source of truth for column-type detection.

Both the profiler and the feature detector classify columns. If each implemented
its own rules they would eventually disagree, and a model would silently train on
a different set of columns than the profile reported. They both call this module
instead.

Detection uses ``pandas.api.types`` predicates rather than ``select_dtypes``,
because dtype *names* are not stable across pandas versions: in pandas 3 string
columns are ``StringDtype`` rather than ``object``, so selecting on ``"object"``
is deprecated and will eventually miss every text column.
"""

from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

from aidatasetkit.core.exceptions import SchemaError, ValidationError
from aidatasetkit.core.types import ColumnKind, ColumnKinds

__all__ = ["detect_column_kinds", "detect_kind"]


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
