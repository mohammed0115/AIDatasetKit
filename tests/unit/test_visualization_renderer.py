"""Tests for the renderer seam and the matplotlib backend."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core.exceptions import MissingDependencyError, UnsupportedChartError
from aidatasetkit.visualization import ChartType, VisualizationService
from aidatasetkit.visualization.renderers import (
    RENDERERS,
    VisualizationRenderer,
    available_renderers,
    get_renderer,
)

matplotlib = pytest.importorskip("matplotlib", reason="the viz extra is not installed")


@pytest.fixture
def service() -> VisualizationService:
    return VisualizationService()


class TestRendererRegistry:
    def test_the_known_renderers_are_listed(self):
        assert "matplotlib" in available_renderers()

    def test_an_unknown_renderer_is_refused(self):
        with pytest.raises(UnsupportedChartError, match="No renderer named"):
            get_renderer("crayons")

    def test_the_registry_records_what_each_renderer_needs(self):
        registration = RENDERERS["matplotlib"]
        assert registration.requires_package == "matplotlib"
        assert registration.install_hint() == "pip install aidatasetkit[viz]"

    def test_a_missing_backend_explains_how_to_install_it(self, monkeypatch):
        from aidatasetkit.visualization.renderers import base

        broken = base.RendererRegistration(
            name="ghost",
            module="aidatasetkit.visualization.renderers.not_a_module",
            attribute="Ghost",
            requires_package="ghostlib",
            extra="ghost",
        )
        monkeypatch.setitem(base.RENDERERS, "ghost", broken)
        with pytest.raises(MissingDependencyError) as error:
            get_renderer("ghost")
        assert "pip install aidatasetkit[ghost]" in str(error.value)
        assert "planning do not need it" in str(error.value)

    def test_the_matplotlib_renderer_satisfies_the_protocol(self):
        assert isinstance(get_renderer("matplotlib"), VisualizationRenderer)


class TestMatplotlibRenderer:
    @pytest.fixture
    def renderer(self):
        return get_renderer("matplotlib")

    def test_it_supports_every_chart_type_this_version_defines(self, renderer):
        unsupported = [
            chart.value for chart in ChartType if not renderer.supports(chart)
        ]
        assert unsupported == []

    @pytest.mark.parametrize("chart_type", list(ChartType), ids=lambda c: c.value)
    def test_every_chart_type_renders_to_a_figure(
        self, service, renderer, chart_type, binary_frame, missing_heavy_frame, wide_frame
    ):
        from matplotlib.figure import Figure

        prepared = _prepared_for(service, chart_type, binary_frame, missing_heavy_frame, wide_frame)
        figure = renderer.render(prepared)
        assert isinstance(figure, Figure)
        assert figure.axes

    def test_rendering_does_not_open_a_pyplot_figure(self, service, renderer, binary_frame):
        """A library has no business filling its caller's pyplot registry."""
        from matplotlib import pyplot

        pyplot.close("all")
        before = len(pyplot.get_fignums())
        renderer.render(service.histogram(binary_frame, "age"))
        assert len(pyplot.get_fignums()) == before

    def test_an_unknown_chart_type_is_refused_clearly(self, service, renderer, binary_frame):
        from aidatasetkit.visualization.types import PreparedChart

        prepared = service.histogram(binary_frame, "age")
        broken = PreparedChart(
            spec=prepared.spec, data=prepared.data, sampling=prepared.sampling
        )
        object.__setattr__(broken.spec, "chart_type", "not_a_chart")
        with pytest.raises(UnsupportedChartError, match="cannot draw"):
            renderer.render(broken)

    def test_a_sampled_chart_says_so_on_the_figure(self, renderer):
        from aidatasetkit.visualization import VisualizationConfig

        frame = pd.DataFrame({"x": np.arange(30_000, dtype="float64") % 7_919})
        service = VisualizationService(VisualizationConfig(max_points=500))
        figure = renderer.render(service.histogram(frame, "x"))
        assert "showing 500 of 30,000 rows" in figure.axes[0].get_title()

    def test_an_unsampled_chart_says_nothing_extra(self, service, renderer, binary_frame):
        figure = renderer.render(service.histogram(binary_frame, "age"))
        assert "showing" not in figure.axes[0].get_title()

    def test_the_service_renders_a_whole_plan(self, service, binary_frame):
        from matplotlib.figure import Figure

        plan = service.recommend(binary_frame, target="churn")
        figures = service.render_plan(binary_frame, plan)
        assert len(figures) == len(plan.charts)
        assert all(isinstance(figure, Figure) for figure in figures)

    def test_rendering_leaves_the_frame_unchanged(self, service, binary_frame):
        before = binary_frame.copy(deep=True)
        service.render_plan(binary_frame, service.recommend(binary_frame, target="churn"))
        pd.testing.assert_frame_equal(binary_frame, before)

    def test_a_blank_correlation_cell_does_not_break_rendering(self, service, renderer):
        frame = pd.DataFrame(
            {
                "flat": [5.0] * 60,
                "a": (np.arange(60) % 17).astype("float64"),
                "b": (np.arange(60) % 23).astype("float64"),
            }
        )
        figure = renderer.render(service.correlation(frame, ["flat", "a", "b"]))
        assert figure.axes


def _prepared_for(service, chart_type, binary_frame, missing_heavy_frame, wide_frame):
    """Build a prepared chart of each type from the golden fixtures."""
    from aidatasetkit.visualization.types import ChartSpec, VisualizationReason

    def spec(kind, *columns, target=None):
        return ChartSpec(
            chart_type=kind,
            columns=tuple(columns),
            title=f"{kind.value} test",
            reason=VisualizationReason(code="test", message="test"),
            target=target,
        )

    if chart_type is ChartType.HISTOGRAM:
        return service.histogram(binary_frame, "age")
    if chart_type is ChartType.BOX_PLOT:
        return service.boxplot(binary_frame, "income")
    if chart_type is ChartType.BAR:
        return service.bar(binary_frame, "segment")
    if chart_type is ChartType.SCATTER:
        return service.scatter(binary_frame, "age", "income")
    if chart_type is ChartType.CORRELATION_HEATMAP:
        return service.correlation(wide_frame, ["f000", "f001", "f002"])
    if chart_type is ChartType.MISSING_VALUES:
        return service.missing(missing_heavy_frame)

    context = service.context(binary_frame, target="churn")
    if chart_type is ChartType.TARGET_DISTRIBUTION:
        return service.prepare(
            binary_frame,
            spec(ChartType.TARGET_DISTRIBUTION, "churn", target="churn"),
            context=context,
        )
    if chart_type is ChartType.GROUPED_BOX:
        return service.prepare(
            binary_frame, spec(ChartType.GROUPED_BOX, "age", "churn", target="churn"), context=context
        )
    return service.prepare(
        binary_frame,
        spec(ChartType.GROUPED_BAR, "segment", "churn", target="churn"),
        context=context,
    )
