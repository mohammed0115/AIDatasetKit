"""The public entry point for visualization.

One object covers both halves of the problem:

*Ask what is worth looking at* -- :meth:`VisualizationService.recommend` returns a
ranked, explained plan and touches no plotting library.

*Draw something specific* -- the manual methods honour exactly what was asked for,
validating it first so that a mismatch is reported against the column rather than
against an array shape.

The two are deliberately different in temperament. The advisor suppresses an
identifier column because nobody asked about it; a caller who names that column
gets their chart with a warning attached. Automatic policy is a default, not a
veto.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Sequence
from typing import Any

import pandas as pd

from aidatasetkit.core.config import KitConfig
from aidatasetkit.core.exceptions import InvalidVisualizationRequest, SchemaError
from aidatasetkit.core.types import DatasetProfile, QualityReport, TargetProfile
from aidatasetkit.profiling.profiler import DataProfiler
from aidatasetkit.profiling.quality import DataQualityInspector
from aidatasetkit.profiling.task_detector import TaskDetector
from aidatasetkit.visualization.advisor import VisualizationAdvisor
from aidatasetkit.visualization.config import VisualizationConfig
from aidatasetkit.visualization.context import VisualizationContext
from aidatasetkit.visualization.preparation import ChartPreparer
from aidatasetkit.visualization.renderers.base import VisualizationRenderer, get_renderer
from aidatasetkit.visualization.types import (
    ChartSpec,
    ChartType,
    PreparedChart,
    VisualizationPlan,
    VisualizationReason,
)
from aidatasetkit.visualization.validators import (
    manual_warnings,
    require_column,
    require_frame,
    require_label_like,
    require_numeric,
)

__all__ = ["VisualizationService"]

_logger = logging.getLogger(__name__)


class VisualizationService:
    """Recommends, prepares, and renders charts for a tabular dataset.

    Args:
        config: Budgets and scoring policy for the visualization layer.
        kit_config: Thresholds handed to the profiler, inspector, and task
            detector when this service has to compute them itself.
        advisor: Recommendation engine. Injectable for testing.
        preparer: Render-preparation engine.
        profiler: Used when :meth:`recommend` is called without a profile.
        inspector: Used when :meth:`recommend` is called without a quality report.
        task_detector: Used when a target column is named but not yet profiled.
        renderer: A ready renderer. When omitted, one is resolved lazily by name
            at the first render, so nothing imports a plotting library until then.

    Example:
        >>> import pandas as pd
        >>> from aidatasetkit.visualization import VisualizationService
        >>> frame = pd.DataFrame({"age": [21, 34, 46, 29, 55, 38, 41, 27, 33, 60]})
        >>> plan = VisualizationService().recommend(frame)
        >>> plan.charts[0].chart_type.value
        'histogram'
    """

    def __init__(
        self,
        config: VisualizationConfig | None = None,
        *,
        kit_config: KitConfig | None = None,
        advisor: VisualizationAdvisor | None = None,
        preparer: ChartPreparer | None = None,
        profiler: DataProfiler | None = None,
        inspector: DataQualityInspector | None = None,
        task_detector: TaskDetector | None = None,
        renderer: VisualizationRenderer | None = None,
    ) -> None:
        self._config = config if config is not None else VisualizationConfig()
        self._kit_config = kit_config if kit_config is not None else KitConfig()
        self._advisor = advisor if advisor is not None else VisualizationAdvisor()
        self._preparer = preparer if preparer is not None else ChartPreparer(self._config)
        self._profiler = profiler if profiler is not None else DataProfiler(self._kit_config)
        self._inspector = (
            inspector if inspector is not None else DataQualityInspector(self._kit_config)
        )
        self._task_detector = (
            task_detector if task_detector is not None else TaskDetector(self._kit_config)
        )
        self._renderer = renderer

    @property
    def config(self) -> VisualizationConfig:
        """The budgets and scoring policy in force."""
        return self._config

    # ------------------------------------------------------------------ #
    # Recommendation
    # ------------------------------------------------------------------ #

    def recommend(
        self,
        frame: pd.DataFrame,
        *,
        profile: DatasetProfile | None = None,
        quality_report: QualityReport | None = None,
        target_profile: TargetProfile | None = None,
        target: Hashable | None = None,
    ) -> VisualizationPlan:
        """Return a ranked, explained plan of what to visualize.

        Any analysis not supplied is computed once here and reused throughout, so
        nothing is measured twice.

        Args:
            frame: The dataset. Read, never modified.
            profile: A profile of the same frame, if already computed.
            quality_report: Findings for the same frame, if already computed.
            target_profile: What the task detector concluded, if already computed.
            target: Name of the target column, when no ``target_profile`` is given.

        Returns:
            A :class:`~aidatasetkit.visualization.types.VisualizationPlan`.

        Raises:
            SchemaError: If ``frame`` is not a dataframe, or ``target`` is absent.
        """
        context = self.context(
            frame,
            profile=profile,
            quality_report=quality_report,
            target_profile=target_profile,
            target=target,
        )
        return self._advisor.recommend(context)

    def context(
        self,
        frame: pd.DataFrame,
        *,
        profile: DatasetProfile | None = None,
        quality_report: QualityReport | None = None,
        target_profile: TargetProfile | None = None,
        target: Hashable | None = None,
    ) -> VisualizationContext:
        """Assemble the analysis the advisor and preparer both work from."""
        require_frame(frame)
        if target is not None and target not in frame.columns:
            raise SchemaError(f"The target {target!r} is not a column of this frame.")

        resolved_profile = profile if profile is not None else self._profiler.profile(frame)
        resolved_target = self._resolve_target_profile(frame, target, target_profile)
        target_name = self._target_name(resolved_target, target)

        if quality_report is None:
            quality_report = self._inspector.inspect(
                frame,
                target=target_name,
                profile=resolved_profile,
                target_profile=resolved_target,
            )

        return VisualizationContext(
            frame=frame,
            profile=resolved_profile,
            quality=quality_report,
            target_profile=resolved_target,
            target_column=target_name,
            config=self._config,
        )

    def _resolve_target_profile(
        self,
        frame: pd.DataFrame,
        target: Hashable | None,
        target_profile: TargetProfile | None,
    ) -> TargetProfile | None:
        """Reuse a supplied target profile, or detect one when a target is named.

        A target whose task cannot be inferred is not a failure: the plan simply
        contains no target-aware charts, and the reason is logged.
        """
        if target_profile is not None or target is None:
            return target_profile
        try:
            return self._task_detector.detect(frame[target], target_name=target)
        except Exception as error:
            _logger.info(
                "Target %r could not be profiled (%s); the plan will contain no "
                "target-aware charts.",
                target,
                error,
            )
            return None

    @staticmethod
    def _target_name(
        target_profile: TargetProfile | None, target: Hashable | None
    ) -> Hashable | None:
        """The target label to use, preferring the explicit argument."""
        if target is not None:
            return target
        return None if target_profile is None else target_profile.target_name

    # ------------------------------------------------------------------ #
    # Preparation and rendering
    # ------------------------------------------------------------------ #

    def prepare(
        self,
        frame: pd.DataFrame,
        spec: ChartSpec,
        *,
        context: VisualizationContext | None = None,
    ) -> PreparedChart:
        """Extract the values ``spec`` needs, within the configured budgets.

        When no context is supplied, one is built around ``spec.target``. Building
        it blind would lose the target, and a regression target's distribution
        would then be drawn as a bar chart of stringified floats -- a wrong chart
        from a correct plan.
        """
        resolved = (
            context
            if context is not None
            else self.context(frame, target=spec.target)
        )
        return self._preparer.prepare(resolved, spec)

    def render(
        self,
        frame: pd.DataFrame,
        spec: ChartSpec,
        *,
        context: VisualizationContext | None = None,
    ) -> Any:
        """Draw one chart and return the renderer's figure.

        Nothing is displayed; the figure is the caller's to save, embed, or show.

        Raises:
            MissingDependencyError: If the renderer's plotting library is absent.
        """
        prepared = self.prepare(frame, spec, context=context)
        return self.renderer.render(prepared)

    def render_plan(
        self, frame: pd.DataFrame, plan: VisualizationPlan
    ) -> tuple[Any, ...]:
        """Draw every chart in a plan, reusing one analysis for all of them.

        The target is taken from the plan's own metadata, so the charts drawn are
        the charts that were planned.
        """
        target = None if plan.metadata is None else plan.metadata.target_name
        context = self.context(frame, target=self._resolve_label(frame, target))
        return tuple(self.render(frame, spec, context=context) for spec in plan.charts)

    def show(self, frame: pd.DataFrame, plan: VisualizationPlan | ChartSpec) -> None:
        """Draw and display. The one place this library touches ``pyplot``.

        Figures are created *by* pyplot here and drawn into, rather than created
        independently and grafted onto a borrowed canvas. The caller can therefore
        close them the ordinary way, and repeated calls do not accumulate
        unreachable figure managers in matplotlib's global registry.

        Raises:
            MissingDependencyError: If the renderer's plotting library is absent.
        """
        from matplotlib import pyplot

        specs = (plan,) if isinstance(plan, ChartSpec) else tuple(plan.charts)
        target = (
            plan.target
            if isinstance(plan, ChartSpec)
            else (None if plan.metadata is None else plan.metadata.target_name)
        )
        context = self.context(frame, target=self._resolve_label(frame, target))

        for spec in specs:
            prepared = self.prepare(frame, spec, context=context)
            self.renderer.render(prepared, figure=pyplot.figure())
        pyplot.show()

    @staticmethod
    def _resolve_label(frame: pd.DataFrame, name: Any) -> Any:
        """Match a recorded target name back onto the frame's actual label.

        Plan metadata stores the target as a string for serialisation; the frame
        may label that column with an integer.
        """
        if name is None or name in frame.columns:
            return name
        for column in frame.columns:
            if str(column) == str(name):
                return column
        return None

    @property
    def renderer(self) -> VisualizationRenderer:
        """The renderer, resolved on first use and cached afterwards."""
        if self._renderer is None:
            self._renderer = get_renderer(self._config.renderer)
        return self._renderer

    # ------------------------------------------------------------------ #
    # Manual charts
    # ------------------------------------------------------------------ #

    def histogram(self, frame: pd.DataFrame, column: Hashable) -> PreparedChart:
        """Prepare a histogram of one numeric column, on request."""
        return self._manual(frame, ChartType.HISTOGRAM, (column,), numeric=(column,))

    def boxplot(self, frame: pd.DataFrame, column: Hashable) -> PreparedChart:
        """Prepare a box plot of one numeric column, on request."""
        return self._manual(frame, ChartType.BOX_PLOT, (column,), numeric=(column,))

    def bar(self, frame: pd.DataFrame, column: Hashable) -> PreparedChart:
        """Prepare a bar chart of one categorical column, on request."""
        return self._manual(frame, ChartType.BAR, (column,), label_like=(column,))

    def scatter(self, frame: pd.DataFrame, x: Hashable, y: Hashable) -> PreparedChart:
        """Prepare a scatter of two numeric columns, on request."""
        if x == y:
            raise InvalidVisualizationRequest(
                f"A scatter needs two different columns; {x!r} was given for both axes."
            )
        return self._manual(frame, ChartType.SCATTER, (x, y), numeric=(x, y))

    def correlation(
        self, frame: pd.DataFrame, columns: Sequence[Hashable] | None = None
    ) -> PreparedChart:
        """Prepare a correlation heatmap over the given numeric columns.

        With no columns named, every numeric column is used, up to the configured
        heatmap budget.
        """
        context = self.context(frame)
        chosen = self._heatmap_columns(context, columns)
        spec = ChartSpec(
            chart_type=ChartType.CORRELATION_HEATMAP,
            columns=chosen,
            title=f"Correlation between {len(chosen)} numeric columns",
            reason=VisualizationReason(
                code="manual_request",
                message="Requested directly rather than recommended.",
                evidence={"columns": [str(column) for column in chosen]},
            ),
        )
        return self._preparer.prepare(context, spec)

    def missing(self, frame: pd.DataFrame) -> PreparedChart:
        """Prepare the missing-value view for a frame, on request."""
        context = self.context(frame)
        columns = tuple(
            profile.name for profile in context.column_profiles if profile.missing_count
        )
        if not columns:
            raise InvalidVisualizationRequest(
                "This frame has no missing values, so a missing-value chart would be "
                "empty."
            )
        spec = ChartSpec(
            chart_type=ChartType.MISSING_VALUES,
            columns=columns,
            title="Missing values by column",
            reason=VisualizationReason(
                code="manual_request",
                message="Requested directly rather than recommended.",
                evidence={"columns_with_missing": len(columns)},
            ),
        )
        return self._preparer.prepare(context, spec)

    #: The quality view this version provides: missingness by column.
    quality = missing

    def _manual(
        self,
        frame: pd.DataFrame,
        chart_type: ChartType,
        columns: tuple[Hashable, ...],
        *,
        numeric: tuple[Hashable, ...] = (),
        label_like: tuple[Hashable, ...] = (),
    ) -> PreparedChart:
        """Validate an explicit request, then prepare it."""
        context = self.context(frame)
        profiles = {
            column: require_column(frame, context.profile, column) for column in columns
        }
        for column in numeric:
            require_numeric(profiles[column], chart_type)
        for column in label_like:
            require_label_like(profiles[column], chart_type)

        warnings = tuple(
            warning
            for column in columns
            for warning in manual_warnings(profiles[column], self._config.max_categories)
        )
        spec = ChartSpec(
            chart_type=chart_type,
            columns=columns,
            title=self._manual_title(chart_type, columns),
            reason=VisualizationReason(
                code="manual_request",
                message="Requested directly rather than recommended.",
                evidence={"columns": [str(column) for column in columns]},
            ),
            warnings=warnings,
        )
        prepared = self._preparer.prepare(context, spec)
        return PreparedChart(
            spec=spec,
            data=prepared.data,
            sampling=prepared.sampling,
            warnings=_merge_warnings(warnings, prepared.warnings),
        )

    def _heatmap_columns(
        self, context: VisualizationContext, columns: Sequence[Hashable] | None
    ) -> tuple[Hashable, ...]:
        """Resolve which columns a manual heatmap should cover."""
        from aidatasetkit.core.types import ColumnKind

        if columns is not None:
            chosen = tuple(columns)
            for column in chosen:
                require_numeric(
                    require_column(context.frame, context.profile, column),
                    ChartType.CORRELATION_HEATMAP,
                )
        else:
            chosen = tuple(
                profile.name
                for profile in context.column_profiles
                if profile.detected_kind is ColumnKind.NUMERIC
            )[: self._config.max_heatmap_columns]

        if len(chosen) < 2:
            raise InvalidVisualizationRequest(
                f"A correlation heatmap needs at least two numeric columns, got "
                f"{len(chosen)}."
            )
        return chosen

    @staticmethod
    def _manual_title(chart_type: ChartType, columns: tuple[Hashable, ...]) -> str:
        """A plain title for an explicitly requested chart."""
        if chart_type is ChartType.SCATTER:
            return f"{columns[0]!s} against {columns[1]!s}"
        if chart_type is ChartType.BAR:
            return f"Frequency of {columns[0]!s}"
        if chart_type is ChartType.BOX_PLOT:
            return f"Spread of {columns[0]!s}"
        return f"Distribution of {columns[0]!s}"


def _merge_warnings(*groups):
    """Combine warning groups, keeping the first of each code.

    The validator and the preparer can both notice the same thing -- a truncated
    category list, say -- and phrase it differently. One concern deserves one
    warning, so the first wording wins.
    """
    seen: dict[str, object] = {}
    for group in groups:
        for warning in group:
            seen.setdefault(warning.code, warning)
    return tuple(seen.values())
