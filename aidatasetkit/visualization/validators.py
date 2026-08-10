"""Checking a manual chart request before anything is drawn.

A manual request that does not suit its data should fail here, naming the column
and the reason, rather than deep inside a plotting library where the message names
an array shape.

Automatic policy and manual control are deliberately different. The advisor
*suppresses* an identifier column, because nobody asked for it. A caller who names
that column explicitly gets their chart, with a warning attached -- they may well
have a reason, and it is not this library's place to overrule them.
"""

from __future__ import annotations

from collections.abc import Hashable

import pandas as pd

from aidatasetkit.core.exceptions import InvalidVisualizationRequest, SchemaError
from aidatasetkit.core.types import ColumnKind, ColumnProfile, DatasetProfile
from aidatasetkit.visualization.types import ChartType, VisualizationWarning

__all__ = [
    "require_frame",
    "require_column",
    "require_numeric",
    "require_label_like",
    "manual_warnings",
]

#: Column kinds a bar chart can describe.
_LABEL_KINDS = (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN)


def require_frame(frame: object) -> pd.DataFrame:
    """Return ``frame`` if it is a dataframe.

    Raises:
        SchemaError: If it is not.
    """
    if not isinstance(frame, pd.DataFrame):
        raise SchemaError(
            f"A pandas DataFrame is required, got {type(frame).__name__}."
        )
    return frame


def require_column(
    frame: pd.DataFrame, profile: DatasetProfile, column: Hashable
) -> ColumnProfile:
    """Return the profile of ``column``, checking it exists in both.

    Raises:
        SchemaError: If the column is absent from the frame or the profile.
    """
    if column not in frame.columns:
        available = [str(name) for name in list(frame.columns)[:20]]
        raise SchemaError(
            f"Column {column!r} is not in the frame. Available columns include: "
            f"{available}."
        )
    return profile.column(column)


def require_numeric(column_profile: ColumnProfile, chart_type: ChartType) -> None:
    """Reject a chart that needs numbers on a column that has none.

    Raises:
        InvalidVisualizationRequest: If the column is not numeric.
    """
    if column_profile.detected_kind is ColumnKind.NUMERIC:
        return
    raise InvalidVisualizationRequest(
        f"A {chart_type.value} needs a numeric column, but "
        f"{column_profile.name!r} was detected as "
        f"{column_profile.detected_kind.value} (dtype {column_profile.pandas_dtype}). "
        "Use a bar chart for category frequencies."
    )


def require_label_like(column_profile: ColumnProfile, chart_type: ChartType) -> None:
    """Reject a category chart on a column that is not label-like.

    A numeric column is refused because a bar per distinct number is a histogram
    drawn badly.

    Raises:
        InvalidVisualizationRequest: If the column is not categorical or boolean.
    """
    if column_profile.detected_kind in _LABEL_KINDS:
        return
    if column_profile.detected_kind is ColumnKind.NUMERIC:
        raise InvalidVisualizationRequest(
            f"A {chart_type.value} needs a categorical column, but "
            f"{column_profile.name!r} is numeric with "
            f"{column_profile.unique_count} distinct values. Use a histogram for a "
            "numeric distribution."
        )
    raise InvalidVisualizationRequest(
        f"A {chart_type.value} needs a categorical column, but "
        f"{column_profile.name!r} was detected as "
        f"{column_profile.detected_kind.value}."
    )


def manual_warnings(
    column_profile: ColumnProfile, max_categories: int
) -> tuple[VisualizationWarning, ...]:
    """Return what the caller should know about charting this column.

    These are warnings, never refusals: the request is honoured either way.
    """
    warnings: list[VisualizationWarning] = []

    if column_profile.is_constant:
        warnings.append(
            VisualizationWarning(
                code="constant_column",
                message=(
                    f"{column_profile.name!s} holds the single value "
                    f"{column_profile.dominant_value!r}; the chart will show one bar."
                ),
            )
        )
    elif column_profile.is_near_constant:
        warnings.append(
            VisualizationWarning(
                code="near_constant_column",
                message=(
                    f"{column_profile.name!s} holds one value in "
                    f"{column_profile.dominant_ratio:.1%} of rows, so the chart will "
                    "be dominated by it."
                ),
            )
        )

    if column_profile.is_id_like:
        warnings.append(
            VisualizationWarning(
                code="possible_id_like",
                message=(
                    f"{column_profile.name!s} looks like a record identifier "
                    f"({column_profile.unique_ratio:.1%} distinct). Automatic "
                    "recommendations skip such columns; this chart was requested "
                    "explicitly."
                ),
            )
        )

    if column_profile.unique_count > max_categories:
        warnings.append(
            VisualizationWarning(
                code="categories_truncated",
                message=(
                    f"{column_profile.name!s} has {column_profile.unique_count} "
                    f"categories; the {max_categories} most frequent are shown and the "
                    "rest grouped as 'Other'."
                ),
            )
        )

    if column_profile.missing_count:
        warnings.append(
            VisualizationWarning(
                code="missing_values_excluded",
                message=(
                    f"{column_profile.missing_count} missing value(s) in "
                    f"{column_profile.name!s} are excluded from the chart. They are "
                    "not imputed and the source data is unchanged."
                ),
            )
        )

    if column_profile.infinite_count:
        warnings.append(
            VisualizationWarning(
                code="infinite_values_excluded",
                message=(
                    f"{column_profile.infinite_count} infinite value(s) in "
                    f"{column_profile.name!s} are excluded from the chart."
                ),
            )
        )

    return tuple(warnings)
