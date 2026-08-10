"""Turning a decision into render-ready numbers.

Read-only, in the strict sense: nothing here imputes, encodes, clips, scales,
removes an outlier, or changes a dtype, and the caller's frame is byte-for-byte
unchanged afterwards. Rows are excluded from a *plot* -- a missing value cannot be
drawn -- but never from the data.

Two responsibilities sit here rather than in the advisor, on purpose. Whether a
scatter is *worth drawing* is a question about the data; how many of its eight
million points can be *drawn legibly* is a question about the canvas. Sampling
therefore never changes a recommendation, and it is never silent: every prepared
chart carries a :class:`~aidatasetkit.visualization.types.SamplingInfo` saying
what happened, even when the answer is "nothing".
"""

from __future__ import annotations

import logging
from collections.abc import Hashable

import numpy as np
import pandas as pd

from aidatasetkit.core.arrays import to_float_arrays
from aidatasetkit.core.exceptions import AIDatasetKitError, InvalidVisualizationRequest
from aidatasetkit.core.types import ColumnKind, TaskType
from aidatasetkit.statistics.bivariate import correlation
from aidatasetkit.visualization.config import VisualizationConfig
from aidatasetkit.visualization.context import VisualizationContext
from aidatasetkit.visualization.types import (
    ChartSpec,
    ChartType,
    PreparedChart,
    SamplingInfo,
    VisualizationWarning,
)

__all__ = ["ChartPreparer", "OTHER_LABEL"]

_logger = logging.getLogger(__name__)

#: Label for the bucket holding categories beyond the display budget.
OTHER_LABEL = "Other"


class ChartPreparer:
    """Extracts the values one chart needs, within the configured budgets."""

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        self._config = config if config is not None else VisualizationConfig()

    @property
    def config(self) -> VisualizationConfig:
        """The budgets this preparer works to."""
        return self._config

    def prepare(self, context: VisualizationContext, spec: ChartSpec) -> PreparedChart:
        """Return render-ready data for ``spec``.

        Raises:
            InvalidVisualizationRequest: If the chart type has no preparation rule.
        """
        handler = _HANDLERS.get(spec.chart_type)
        if handler is None:
            # str(), not .value: an error path must not itself assume the value
            # it is complaining about has the right type.
            raise InvalidVisualizationRequest(
                f"No preparation rule exists for chart type {str(spec.chart_type)!r}. "
                f"Supported: {sorted(chart.value for chart in _HANDLERS)}."
            )
        return handler(self, context, spec)

    # ------------------------------------------------------------------ #
    # Univariate
    # ------------------------------------------------------------------ #

    def _prepare_histogram(
        self, context: VisualizationContext, spec: ChartSpec
    ) -> PreparedChart:
        """Finite values of one numeric column."""
        column = spec.columns[0]
        values, excluded = _finite_values(context.frame[column])
        sampled, info = self._sample_array(values, self._config_for(context))
        return PreparedChart(
            spec=spec,
            data={"values": sampled, "column": str(column)},
            sampling=info,
            warnings=_exclusion_warnings(column, excluded),
        )

    def _prepare_box(self, context: VisualizationContext, spec: ChartSpec) -> PreparedChart:
        """Finite values of one numeric column, for quartiles and whiskers."""
        prepared = self._prepare_histogram(context, spec)
        return PreparedChart(
            spec=spec,
            data=prepared.data,
            sampling=prepared.sampling,
            warnings=prepared.warnings,
        )

    def _prepare_bar(self, context: VisualizationContext, spec: ChartSpec) -> PreparedChart:
        """Category counts, truncated to the budget with an "Other" bucket."""
        column = spec.columns[0]
        config = self._config_for(context)
        labels, counts, folded, total = self._category_counts(context.frame[column], config)
        warnings: list[VisualizationWarning] = []
        missing = int(len(context.frame)) - total
        if missing:
            warnings.append(
                VisualizationWarning(
                    code="missing_values_excluded",
                    message=(
                        f"{missing} missing value(s) in {column!s} are excluded from "
                        "the chart. They are not imputed and the source data is "
                        "unchanged."
                    ),
                )
            )
        if folded:
            warnings.append(
                VisualizationWarning(
                    code="categories_truncated",
                    message=(
                        f"{folded} categories beyond the top "
                        f"{config.max_categories} are grouped as "
                        f"{OTHER_LABEL!r}. Every observation is still counted."
                    ),
                )
            )
        return PreparedChart(
            spec=spec,
            data={
                "labels": labels,
                "counts": counts,
                "column": str(column),
                "folded_categories": folded,
                "total_count": total,
            },
            sampling=SamplingInfo(
                source_rows=int(len(context.frame)), rendered_rows=total
            ),
            warnings=tuple(warnings),
        )

    # ------------------------------------------------------------------ #
    # Dataset level
    # ------------------------------------------------------------------ #

    def _prepare_missing(
        self, context: VisualizationContext, spec: ChartSpec
    ) -> PreparedChart:
        """Missing-value percentage per column, worst first.

        Read straight off the profile; nothing is recounted here.
        """
        wanted = set(spec.columns) if spec.columns else None
        entries = [
            (str(profile.name), profile.missing_ratio * 100.0, profile.missing_count)
            for profile in context.column_profiles
            if profile.missing_count and (wanted is None or profile.name in wanted)
        ]
        entries.sort(key=lambda item: (-item[1], item[0]))
        return PreparedChart(
            spec=spec,
            data={
                "labels": [name for name, _, _ in entries],
                "percentages": [pct for _, pct, _ in entries],
                "counts": [count for _, _, count in entries],
                "row_count": context.profile.row_count,
            },
            sampling=SamplingInfo(
                source_rows=context.profile.row_count,
                rendered_rows=context.profile.row_count,
            ),
        )

    def _prepare_heatmap(
        self, context: VisualizationContext, spec: ChartSpec
    ) -> PreparedChart:
        """A correlation matrix over the columns the advisor selected.

        Every cell is produced by
        :func:`aidatasetkit.statistics.bivariate.correlation`. No correlation
        formula is written in this package; an undefined pair -- a constant
        column, no complete pairs -- becomes ``None`` rather than a fabricated
        zero, which would read as "measured no relationship".
        """
        columns = list(spec.columns)
        size = len(columns)
        matrix: list[list[float | None]] = [[None] * size for _ in range(size)]
        undefined = 0

        for i in range(size):
            matrix[i][i] = 1.0
            for j in range(i + 1, size):
                value = _safe_correlation(
                    context.frame[columns[i]], context.frame[columns[j]]
                )
                if value is None:
                    undefined += 1
                matrix[i][j] = value
                matrix[j][i] = value

        warnings: tuple[VisualizationWarning, ...] = ()
        if undefined:
            warnings = (
                VisualizationWarning(
                    code="undefined_correlations",
                    message=(
                        f"{undefined} column pair(s) have no defined correlation, "
                        "usually because one column is constant. Those cells are "
                        "blank rather than zero."
                    ),
                ),
            )

        return PreparedChart(
            spec=spec,
            data={
                "labels": [str(column) for column in columns],
                "matrix": matrix,
                "method": "pearson",
            },
            sampling=SamplingInfo(
                source_rows=context.profile.row_count,
                rendered_rows=context.profile.row_count,
            ),
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    # Target and relationships
    # ------------------------------------------------------------------ #

    def _prepare_target(
        self, context: VisualizationContext, spec: ChartSpec
    ) -> PreparedChart:
        """The target's distribution, as bars for classes or a histogram for values."""
        target_profile = context.target_profile
        if target_profile is not None and target_profile.task_type is TaskType.REGRESSION:
            prepared = self._prepare_histogram(context, spec)
            return PreparedChart(
                spec=spec,
                data={**prepared.data, "mode": "histogram"},
                sampling=prepared.sampling,
                warnings=prepared.warnings,
            )
        prepared = self._prepare_bar(context, spec)
        return PreparedChart(
            spec=spec,
            data={**prepared.data, "mode": "bar"},
            sampling=prepared.sampling,
            warnings=prepared.warnings,
        )

    def _prepare_scatter(
        self, context: VisualizationContext, spec: ChartSpec
    ) -> PreparedChart:
        """Two numeric columns, with incomplete pairs removed together.

        Pairwise deletion is delegated to
        :func:`aidatasetkit.core.arrays.to_float_arrays`: dropping missing values
        from each axis independently would shift the two series against each other
        and plot points that never existed.
        """
        x_column, y_column = spec.columns[0], spec.columns[1]
        try:
            x_values, y_values = to_float_arrays(
                context.frame[x_column],
                context.frame[y_column],
                nan_policy="omit",
                allow_inf=True,
                names=(str(x_column), str(y_column)),
            )
        except AIDatasetKitError as error:
            raise InvalidVisualizationRequest(
                f"A scatter of {x_column!r} against {y_column!r} cannot be drawn: {error}"
            ) from error

        # Infinities are excluded here rather than refused upstream: a point at
        # infinity has no position on an axis, but the other rows are perfectly
        # drawable, and a recommended chart that cannot be prepared is a defect.
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        infinite_pairs = int(finite.size - finite.sum())
        x_values, y_values = x_values[finite], y_values[finite]
        if x_values.size == 0:
            raise InvalidVisualizationRequest(
                f"A scatter of {x_column!r} against {y_column!r} has no finite pairs "
                "to draw."
            )

        dropped = int(len(context.frame)) - int(x_values.size) - infinite_pairs
        indices, info = self._sample_indices(
            int(x_values.size), self._config_for(context)
        )
        warnings: list[VisualizationWarning] = []
        if dropped:
            warnings.append(
                VisualizationWarning(
                    code="incomplete_pairs_excluded",
                    message=(
                        f"{dropped} row(s) lack a value on one of the two axes and are "
                        "excluded from the plot. The source data is unchanged."
                    ),
                )
            )
        if infinite_pairs:
            warnings.append(
                VisualizationWarning(
                    code="infinite_values_excluded",
                    message=(
                        f"{infinite_pairs} row(s) hold an infinite value on one of the "
                        "two axes and are excluded from the plot."
                    ),
                )
            )

        return PreparedChart(
            spec=spec,
            data={
                "x": x_values[indices],
                "y": y_values[indices],
                "x_label": str(x_column),
                "y_label": str(y_column),
            },
            sampling=info,
            warnings=tuple(warnings),
        )

    def _prepare_grouped_box(
        self, context: VisualizationContext, spec: ChartSpec
    ) -> PreparedChart:
        """One box per group: a numeric column split by a label column.

        The point budget applies to the *chart*, so the frame is sampled once
        before grouping rather than once per group -- which would have allowed
        ``max_categories x max_points`` values in a chart documented to hold
        ``max_points``. When the grouping column is the classification target and
        stratified sampling is enabled, the sample preserves class proportions so
        that a rare class does not vanish from the picture.
        """
        config = self._config_for(context)
        value_column, group_column = self._split_grouped(context, spec)
        frame = context.frame[[group_column, value_column]].dropna()
        frame = frame[np.isfinite(frame[value_column].to_numpy(dtype="float64"))]

        labels, _, folded, _ = self._category_counts(frame[group_column], config)
        kept = labels[:-1] if folded else labels
        as_text = frame[group_column].astype(str)
        retained = frame[as_text.isin(kept)]
        dropped_to_truncation = int(len(frame)) - int(len(retained))

        stratify = config.stratify_target_samples and context.is_target(group_column)
        sampled_frame, info = self._sample_frame(
            retained, config, stratify_by=as_text.loc[retained.index] if stratify else None
        )

        sampled_text = sampled_frame[group_column].astype(str)
        groups: list[np.ndarray] = []
        kept_labels: list[str] = []
        for label in kept:
            values = sampled_frame.loc[
                sampled_text == label, value_column
            ].to_numpy(dtype="float64")
            if values.size:
                groups.append(values)
                kept_labels.append(label)

        return PreparedChart(
            spec=spec,
            data={
                "groups": groups,
                "labels": kept_labels,
                "value_label": str(value_column),
                "group_label": str(group_column),
            },
            sampling=SamplingInfo(
                source_rows=int(len(context.frame)),
                rendered_rows=info.rendered_rows,
                sampled=info.sampled,
                strategy=info.strategy,
                random_state=info.random_state,
            ),
            warnings=_truncation_warnings(group_column, folded, dropped_to_truncation, config),
        )

    def _prepare_grouped_bar(
        self, context: VisualizationContext, spec: ChartSpec
    ) -> PreparedChart:
        """Counts of one label column split by the target's classes."""
        config = self._config_for(context)
        feature, target = spec.columns[0], spec.columns[1]
        frame = context.frame[[feature, target]].dropna()

        labels, _, folded, _ = self._category_counts(frame[feature], config)
        feature_labels = labels[:-1] if folded else labels

        # One grouped tally over the raw values, then stringify only the small
        # result. Comparing the whole frame once per (label, class) pair scanned
        # it max_categories x n_classes times; converting both columns to text
        # first cost another two passes over every row.
        tally = frame.groupby([feature, target], observed=True, dropna=True).size()
        by_label: dict[str, dict[str, int]] = {}
        for (label_value, class_value), count in tally.items():
            by_label.setdefault(str(label_value), {})[str(class_value)] = int(count)

        class_labels = sorted({str(value) for value in frame[target].unique()})
        counts = [
            [by_label.get(label, {}).get(cls, 0) for cls in class_labels]
            for label in feature_labels
        ]

        rendered = int(sum(sum(row) for row in counts))
        dropped_to_truncation = int(len(frame)) - rendered
        return PreparedChart(
            spec=spec,
            data={
                "labels": feature_labels,
                "class_labels": class_labels,
                "counts": counts,
                "feature_label": str(feature),
                "target_label": str(target),
            },
            sampling=SamplingInfo(
                source_rows=int(len(context.frame)), rendered_rows=rendered
            ),
            warnings=_truncation_warnings(feature, folded, dropped_to_truncation, config),
        )

    # ------------------------------------------------------------------ #
    # Shared machinery
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_grouped(
        context: VisualizationContext, spec: ChartSpec
    ) -> tuple[Hashable, Hashable]:
        """Decide which of a grouped chart's two columns holds the values.

        The answer comes from the profiled :class:`ColumnKind`, not from a fresh
        dtype test. pandas calls a boolean column numeric; the library's single
        classifier deliberately does not, and asking pandas here would put a
        boolean feature on the value axis and a continuous target on the grouping
        axis -- an inverted chart from a correct plan.
        """
        first, second = spec.columns[0], spec.columns[1]
        if context.column(first).detected_kind is ColumnKind.NUMERIC:
            return first, second
        return second, first

    def _config_for(self, context: VisualizationContext) -> VisualizationConfig:
        """Prefer the context's budgets, so one config governs the whole run."""
        return context.config if context.config is not None else self._config

    def _category_counts(
        self, series: pd.Series, config: VisualizationConfig | None = None
    ) -> tuple[list[str], list[int], int, int]:
        """Count categories, folding the tail beyond the budget into "Other".

        Every non-missing observation stays counted: the folded bucket holds the
        sum of the tail, so the bars still add up to the column's non-null total.
        """
        counted = series.dropna().astype(str).value_counts()
        total = int(counted.sum())
        budget = (config if config is not None else self._config).max_categories

        if len(counted) <= budget:
            ordered = counted.sort_values(
                ascending=False, kind="stable"
            )
            return (
                [str(label) for label in ordered.index],
                [int(value) for value in ordered.to_numpy()],
                0,
                total,
            )

        head = counted.iloc[:budget]
        tail = counted.iloc[budget:]
        # A dataset may genuinely contain a category called "Other". Folding the
        # tail under the same label would merge two different things into one bar
        # and make the sentinel indistinguishable from real data.
        sentinel = OTHER_LABEL
        existing = {str(label) for label in counted.index}
        while sentinel in existing:
            sentinel = f"{sentinel} (remaining)"
        labels = [str(label) for label in head.index] + [sentinel]
        counts = [int(value) for value in head.to_numpy()] + [int(tail.sum())]
        return labels, counts, int(len(tail)), total

    def _sample_array(
        self, values: np.ndarray, config: VisualizationConfig | None = None
    ) -> tuple[np.ndarray, SamplingInfo]:
        """Cut an array down to the point budget, deterministically."""
        config = config if config is not None else self._config
        source = int(values.size)
        if source <= config.max_points:
            return values, SamplingInfo(source_rows=source, rendered_rows=source)

        generator = np.random.default_rng(config.random_state)
        chosen = np.sort(
            generator.choice(source, size=config.max_points, replace=False)
        )
        _logger.debug("sampled %d of %d points", chosen.size, source)
        return values[chosen], SamplingInfo(
            source_rows=source,
            rendered_rows=int(chosen.size),
            sampled=True,
            strategy="random",
            random_state=config.random_state,
        )

    def _sample_frame(
        self,
        frame: pd.DataFrame,
        config: VisualizationConfig,
        stratify_by: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, SamplingInfo]:
        """Cut a frame down to the point budget, deterministically.

        With ``stratify_by`` the budget is split across groups in proportion to
        their size, and every group keeps at least one row. A plain random sample
        of a 0.1 percent class can miss it entirely, and a chart drawn from that
        sample would show a class that does not exist to be absent.
        """
        source = int(len(frame))
        if source <= config.max_points:
            return frame, SamplingInfo(source_rows=source, rendered_rows=source)

        generator = np.random.default_rng(config.random_state)
        if stratify_by is None:
            positions = np.sort(
                generator.choice(source, size=config.max_points, replace=False)
            )
            strategy = "random"
        else:
            positions = self._stratified_positions(stratify_by, config, generator)
            strategy = "stratified"

        _logger.debug("sampled %d of %d rows (%s)", positions.size, source, strategy)
        return frame.iloc[positions], SamplingInfo(
            source_rows=source,
            rendered_rows=int(positions.size),
            sampled=True,
            strategy=strategy,
            random_state=config.random_state,
        )

    @staticmethod
    def _stratified_positions(
        groups: pd.Series, config: VisualizationConfig, generator: np.random.Generator
    ) -> np.ndarray:
        """Choose row positions that preserve each group's share of the budget."""
        total = int(len(groups))
        codes = groups.to_numpy()
        chosen: list[np.ndarray] = []

        for label in sorted({str(value) for value in codes}):
            positions = np.flatnonzero(codes.astype(str) == label)
            share = max(1, int(round(config.max_points * positions.size / total)))
            take = min(share, positions.size)
            chosen.append(generator.choice(positions, size=take, replace=False))

        return np.sort(np.concatenate(chosen))

    def _sample_indices(
        self, size: int, config: VisualizationConfig | None = None
    ) -> tuple[np.ndarray, SamplingInfo]:
        """Choose which of ``size`` rows to draw, deterministically."""
        config = config if config is not None else self._config
        if size <= config.max_points:
            return np.arange(size), SamplingInfo(source_rows=size, rendered_rows=size)

        generator = np.random.default_rng(config.random_state)
        chosen = np.sort(generator.choice(size, size=config.max_points, replace=False))
        return chosen, SamplingInfo(
            source_rows=size,
            rendered_rows=int(chosen.size),
            sampled=True,
            strategy="random",
            random_state=config.random_state,
        )


def _finite_values(series: pd.Series) -> tuple[np.ndarray, tuple[int, int]]:
    """Return the finite values of a numeric column, and what was left out."""
    try:
        raw = series.to_numpy(dtype="float64", na_value=np.nan)
    except (TypeError, ValueError) as error:
        raise InvalidVisualizationRequest(
            f"Column {series.name!r} cannot be read as numbers: {error}"
        ) from error

    missing = int(np.isnan(raw).sum())
    infinite = int(np.isinf(raw).sum())
    return raw[np.isfinite(raw)], (missing, infinite)


def _exclusion_warnings(
    column: Hashable, excluded: tuple[int, int]
) -> tuple[VisualizationWarning, ...]:
    """Disclose values a chart could not draw."""
    missing, infinite = excluded
    warnings: list[VisualizationWarning] = []
    if missing:
        warnings.append(
            VisualizationWarning(
                code="missing_values_excluded",
                message=(
                    f"{missing} missing value(s) in {column!s} are excluded from the "
                    "chart. They are not imputed and the source data is unchanged."
                ),
            )
        )
    if infinite:
        warnings.append(
            VisualizationWarning(
                code="infinite_values_excluded",
                message=(
                    f"{infinite} infinite value(s) in {column!s} are excluded from the "
                    "chart, which would otherwise have no readable axis."
                ),
            )
        )
    return tuple(warnings)


def _truncation_warnings(
    column: Hashable, folded: int, dropped: int, config: VisualizationConfig
) -> tuple[VisualizationWarning, ...]:
    """Disclose rows a grouped chart could not show.

    A grouped chart has no "Other" box to fold the tail into, so those rows leave
    the picture entirely. Saying nothing would let a reader take the visible
    groups for the whole dataset.
    """
    if not folded:
        return ()
    return (
        VisualizationWarning(
            code="categories_truncated",
            message=(
                f"{column!s} has more than {config.max_categories} categories; the "
                f"{config.max_categories} most frequent are shown and {dropped} "
                f"row(s) in the remaining {folded} categories are not drawn. The "
                "source data is unchanged."
            ),
        ),
    )


def _safe_correlation(left: pd.Series, right: pd.Series) -> float | None:
    """Pearson correlation of two columns, or ``None`` when undefined."""
    try:
        return correlation(
            left.astype("float64"),
            right.astype("float64"),
            nan_policy="omit",
            allow_inf=False,
        )
    except (AIDatasetKitError, TypeError, ValueError):
        return None


#: Which preparation rule serves each chart type.
_HANDLERS = {
    ChartType.HISTOGRAM: ChartPreparer._prepare_histogram,
    ChartType.BOX_PLOT: ChartPreparer._prepare_box,
    ChartType.BAR: ChartPreparer._prepare_bar,
    ChartType.MISSING_VALUES: ChartPreparer._prepare_missing,
    ChartType.CORRELATION_HEATMAP: ChartPreparer._prepare_heatmap,
    ChartType.TARGET_DISTRIBUTION: ChartPreparer._prepare_target,
    ChartType.SCATTER: ChartPreparer._prepare_scatter,
    ChartType.GROUPED_BOX: ChartPreparer._prepare_grouped_box,
    ChartType.GROUPED_BAR: ChartPreparer._prepare_grouped_bar,
}
