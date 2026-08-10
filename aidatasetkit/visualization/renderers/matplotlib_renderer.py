"""The matplotlib rendering backend.

Named ``matplotlib_renderer`` rather than ``matplotlib`` so that this module can
never shadow the library it imports.

Figures are built through :class:`matplotlib.figure.Figure` directly, not through
``pyplot``. A library has no business installing a global figure manager in its
caller's process, choosing a GUI backend, or leaving figures alive in pyplot's
registry until someone closes them. The returned figure is the caller's to display,
save, embed, or discard; :meth:`VisualizationService.show` is where an explicit
display happens.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure

from aidatasetkit.core.exceptions import UnsupportedChartError
from aidatasetkit.visualization.types import ChartType, PreparedChart

__all__ = ["MatplotlibRenderer"]

#: Default figure size in inches.
_FIGURE_SIZE = (8.0, 5.0)

#: Colour used for every mark, so charts read as one family.
_PRIMARY = "#4C72B0"
_SECONDARY = "#DD8452"


class MatplotlibRenderer:
    """Draws every chart type this version supports."""

    name = "matplotlib"

    def supports(self, chart_type: ChartType) -> bool:
        """Whether this backend can draw ``chart_type``."""
        return chart_type in _DISPATCH

    def render(self, prepared: PreparedChart, figure: Figure | None = None) -> Figure:
        """Draw ``prepared`` and return the figure.

        Nothing is displayed and no global state is touched. When ``figure`` is
        given the chart is drawn into it, which lets a caller that owns a
        pyplot-managed figure display the result without this renderer ever
        touching pyplot.

        Raises:
            UnsupportedChartError: If the chart type has no drawing rule.
        """
        drawer = _DISPATCH.get(prepared.chart_type)
        if drawer is None:
            # str(), not .value: the error path must not assume the value it is
            # complaining about has the right type.
            raise UnsupportedChartError(
                f"The matplotlib renderer cannot draw "
                f"{str(prepared.chart_type)!r}. Supported: "
                f"{sorted(chart.value for chart in _DISPATCH)}."
            )

        if figure is None:
            figure = Figure(figsize=_FIGURE_SIZE, layout="constrained")
        axes = figure.add_subplot(111)
        drawer(self, axes, prepared)
        axes.set_title(_title_with_notes(prepared))
        return figure

    # ------------------------------------------------------------------ #
    # Drawing rules
    # ------------------------------------------------------------------ #

    def _draw_histogram(self, axes, prepared: PreparedChart) -> None:
        values = np.asarray(prepared.data["values"])
        bins = _bin_count(values.size)
        axes.hist(values, bins=bins, color=_PRIMARY, edgecolor="white", linewidth=0.5)
        axes.set_xlabel(prepared.data["column"])
        axes.set_ylabel("Count")

    def _draw_box(self, axes, prepared: PreparedChart) -> None:
        values = np.asarray(prepared.data["values"])
        axes.boxplot(
            values,
            patch_artist=True,
            boxprops={"facecolor": _PRIMARY, "alpha": 0.7},
            medianprops={"color": "black"},
        )
        axes.set_ylabel(prepared.data["column"])
        axes.set_xticks([])

    def _draw_bar(self, axes, prepared: PreparedChart) -> None:
        labels = list(prepared.data["labels"])
        counts = list(prepared.data["counts"])
        positions = np.arange(len(labels))
        axes.bar(positions, counts, color=_PRIMARY)
        axes.set_xticks(positions)
        axes.set_xticklabels(labels, rotation=45, ha="right")
        axes.set_xlabel(prepared.data.get("column", ""))
        axes.set_ylabel("Count")

    def _draw_scatter(self, axes, prepared: PreparedChart) -> None:
        axes.scatter(
            np.asarray(prepared.data["x"]),
            np.asarray(prepared.data["y"]),
            s=12,
            alpha=0.6,
            color=_PRIMARY,
            edgecolors="none",
        )
        axes.set_xlabel(prepared.data["x_label"])
        axes.set_ylabel(prepared.data["y_label"])

    def _draw_heatmap(self, axes, prepared: PreparedChart) -> None:
        labels = list(prepared.data["labels"])
        matrix = np.array(
            [[np.nan if value is None else value for value in row] for row in prepared.data["matrix"]],
            dtype="float64",
        )
        image = axes.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        axes.set_xticks(np.arange(len(labels)))
        axes.set_yticks(np.arange(len(labels)))
        axes.set_xticklabels(labels, rotation=45, ha="right")
        axes.set_yticklabels(labels)
        axes.figure.colorbar(image, ax=axes, label="Pearson correlation")

    def _draw_missing(self, axes, prepared: PreparedChart) -> None:
        labels = list(prepared.data["labels"])
        percentages = list(prepared.data["percentages"])
        positions = np.arange(len(labels))
        axes.barh(positions, percentages, color=_SECONDARY)
        axes.set_yticks(positions)
        axes.set_yticklabels(labels)
        axes.invert_yaxis()
        axes.set_xlabel("Missing (%)")

    def _draw_target(self, axes, prepared: PreparedChart) -> None:
        if prepared.data.get("mode") == "histogram":
            self._draw_histogram(axes, prepared)
            return
        self._draw_bar(axes, prepared)

    def _draw_grouped_box(self, axes, prepared: PreparedChart) -> None:
        groups = [np.asarray(group) for group in prepared.data["groups"]]
        labels = list(prepared.data["labels"])
        if not groups:
            axes.text(0.5, 0.5, "No groups to display", ha="center", va="center")
            return
        axes.boxplot(
            groups,
            tick_labels=labels,
            patch_artist=True,
            boxprops={"facecolor": _PRIMARY, "alpha": 0.7},
            medianprops={"color": "black"},
        )
        axes.set_xlabel(prepared.data["group_label"])
        axes.set_ylabel(prepared.data["value_label"])
        axes.tick_params(axis="x", rotation=45)

    def _draw_grouped_bar(self, axes, prepared: PreparedChart) -> None:
        labels = list(prepared.data["labels"])
        class_labels = list(prepared.data["class_labels"])
        counts = prepared.data["counts"]
        positions = np.arange(len(labels))
        width = 0.8 / max(1, len(class_labels))

        for index, class_label in enumerate(class_labels):
            offsets = positions - 0.4 + width * (index + 0.5)
            heights = [row[index] for row in counts]
            axes.bar(offsets, heights, width=width, label=str(class_label))

        axes.set_xticks(positions)
        axes.set_xticklabels(labels, rotation=45, ha="right")
        axes.set_xlabel(prepared.data["feature_label"])
        axes.set_ylabel("Count")
        axes.legend(title=prepared.data["target_label"], fontsize="small")


def _bin_count(size: int) -> int:
    """Choose a bin count that stays readable across sample sizes."""
    if size < 2:
        return 1
    return int(min(50, max(10, round(np.sqrt(size)))))


def _title_with_notes(prepared: PreparedChart) -> str:
    """Put the chart's title on the figure, disclosing sampling in it.

    A sampled chart that does not say so invites the viewer to read a subset as
    the whole, so the note travels on the image itself rather than only in the
    metadata beside it.
    """
    title = prepared.spec.title
    if prepared.sampling.sampled:
        title += (
            f"\n(showing {prepared.sampling.rendered_rows:,} of "
            f"{prepared.sampling.source_rows:,} rows, "
            f"random_state={prepared.sampling.random_state})"
        )
    return title


#: Which drawing rule serves each chart type.
_DISPATCH: dict[ChartType, Any] = {
    ChartType.HISTOGRAM: MatplotlibRenderer._draw_histogram,
    ChartType.BOX_PLOT: MatplotlibRenderer._draw_box,
    ChartType.BAR: MatplotlibRenderer._draw_bar,
    ChartType.SCATTER: MatplotlibRenderer._draw_scatter,
    ChartType.CORRELATION_HEATMAP: MatplotlibRenderer._draw_heatmap,
    ChartType.MISSING_VALUES: MatplotlibRenderer._draw_missing,
    ChartType.TARGET_DISTRIBUTION: MatplotlibRenderer._draw_target,
    ChartType.GROUPED_BOX: MatplotlibRenderer._draw_grouped_box,
    ChartType.GROUPED_BAR: MatplotlibRenderer._draw_grouped_bar,
}
