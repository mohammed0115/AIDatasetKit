"""Tests for the service: the manual API, its validation, and its errors.

Automatic policy and manual control are different on purpose. The advisor
suppresses an identifier because nobody asked; a caller who names one gets their
chart with a warning. These tests pin that distinction.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core.exceptions import (
    AIDatasetKitError,
    InvalidVisualizationRequest,
    SchemaError,
)
from aidatasetkit.visualization import (
    ChartType,
    VisualizationConfig,
    VisualizationService,
)


@pytest.fixture
def service() -> VisualizationService:
    return VisualizationService()


class TestManualCharts:
    def test_a_histogram_of_a_numeric_column(self, service, binary_frame):
        prepared = service.histogram(binary_frame, "age")
        assert prepared.chart_type is ChartType.HISTOGRAM
        assert len(prepared.data["values"]) == len(binary_frame)

    def test_a_box_plot_of_a_numeric_column(self, service, binary_frame):
        assert service.boxplot(binary_frame, "income").chart_type is ChartType.BOX_PLOT

    def test_a_bar_chart_of_a_categorical_column(self, service, binary_frame):
        prepared = service.bar(binary_frame, "segment")
        assert set(prepared.data["labels"]) == {"A", "B", "C"}

    def test_a_scatter_of_two_numeric_columns(self, service, binary_frame):
        prepared = service.scatter(binary_frame, "age", "income")
        assert prepared.data["x_label"] == "age"
        assert prepared.data["y_label"] == "income"

    def test_a_correlation_heatmap(self, service, binary_frame):
        prepared = service.correlation(binary_frame, ["age", "income"])
        assert prepared.data["labels"] == ["age", "income"]

    def test_a_heatmap_without_named_columns_uses_every_numeric_one(
        self, service, binary_frame
    ):
        assert len(service.correlation(binary_frame).data["labels"]) == 3

    def test_a_missing_value_view(self, service, missing_heavy_frame):
        prepared = service.missing(missing_heavy_frame)
        assert prepared.data["labels"] == ["gappier", "gappy"]
        assert prepared.data["percentages"][0] > prepared.data["percentages"][1]

    def test_the_quality_view_is_the_missing_view(self, service, missing_heavy_frame):
        assert service.quality(missing_heavy_frame).chart_type is ChartType.MISSING_VALUES

    def test_a_manual_chart_says_it_was_requested_not_recommended(
        self, service, binary_frame
    ):
        assert service.histogram(binary_frame, "age").spec.reason.code == "manual_request"


class TestManualValidation:
    def test_a_histogram_of_a_category_is_refused_clearly(self, service, binary_frame):
        with pytest.raises(InvalidVisualizationRequest, match="needs a numeric column"):
            service.histogram(binary_frame, "segment")

    def test_the_refusal_suggests_the_right_chart(self, service, binary_frame):
        with pytest.raises(InvalidVisualizationRequest, match="bar chart"):
            service.histogram(binary_frame, "segment")

    def test_a_bar_of_a_numeric_column_is_refused_clearly(self, service, binary_frame):
        with pytest.raises(InvalidVisualizationRequest, match="needs a categorical"):
            service.bar(binary_frame, "income")

    def test_that_refusal_suggests_a_histogram(self, service, binary_frame):
        with pytest.raises(InvalidVisualizationRequest, match="histogram"):
            service.bar(binary_frame, "income")

    def test_a_scatter_with_a_category_axis_is_refused(self, service, binary_frame):
        with pytest.raises(InvalidVisualizationRequest, match="needs a numeric column"):
            service.scatter(binary_frame, "segment", "age")

    def test_a_scatter_of_one_column_against_itself_is_refused(self, service, binary_frame):
        with pytest.raises(InvalidVisualizationRequest, match="two different columns"):
            service.scatter(binary_frame, "age", "age")

    def test_an_unknown_column_names_what_is_available(self, service, binary_frame):
        with pytest.raises(SchemaError, match="not in the frame"):
            service.histogram(binary_frame, "nope")

    def test_a_boxplot_of_a_datetime_column_is_refused(self, service):
        frame = pd.DataFrame(
            {"when": pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(30), "D")}
        )
        with pytest.raises(InvalidVisualizationRequest, match="datetime"):
            service.boxplot(frame, "when")

    def test_a_heatmap_needs_two_numeric_columns(self, service):
        frame = pd.DataFrame({"only": np.arange(50, dtype="float64") % 13})
        with pytest.raises(InvalidVisualizationRequest, match="at least two numeric"):
            service.correlation(frame)

    def test_a_missing_chart_on_complete_data_is_refused(self, service, binary_frame):
        with pytest.raises(InvalidVisualizationRequest, match="no missing values"):
            service.missing(binary_frame)

    def test_a_non_dataframe_is_refused(self, service):
        with pytest.raises(SchemaError, match="DataFrame is required"):
            service.histogram([1, 2, 3], "x")

    def test_duplicate_column_labels_are_refused_by_the_profiler(self, service):
        frame = pd.DataFrame(np.arange(20).reshape(10, 2), columns=["a", "a"])
        with pytest.raises(SchemaError, match="Duplicate column labels"):
            service.histogram(frame, "a")

    def test_every_manual_failure_is_a_library_error(self, service, binary_frame):
        """Never a bare matplotlib or numpy traceback."""
        for call in (
            lambda: service.histogram(binary_frame, "segment"),
            lambda: service.bar(binary_frame, "income"),
            lambda: service.histogram(binary_frame, "missing_column"),
            lambda: service.scatter(binary_frame, "segment", "age"),
        ):
            with pytest.raises(AIDatasetKitError):
                call()


class TestManualWarnings:
    def test_an_identifier_may_be_charted_on_request_with_a_warning(
        self, service, identifier_frame
    ):
        """The advisor suppresses it; an explicit request is honoured."""
        prepared = service.histogram(identifier_frame, "row_id")
        assert "possible_id_like" in {w.code for w in prepared.warnings}
        assert len(prepared.data["values"]) == len(identifier_frame)

    def test_a_constant_column_may_be_charted_on_request_with_a_warning(
        self, service, constant_frame
    ):
        prepared = service.bar(constant_frame, "country")
        assert "constant_column" in {w.code for w in prepared.warnings}

    def test_a_near_constant_column_warns(self, service, constant_frame):
        prepared = service.bar(constant_frame, "plan")
        assert "near_constant_column" in {w.code for w in prepared.warnings}

    def test_missing_values_are_disclosed(self, service, missing_heavy_frame):
        prepared = service.histogram(missing_heavy_frame, "gappy")
        assert "missing_values_excluded" in {w.code for w in prepared.warnings}

    def test_high_cardinality_truncation_is_disclosed(self, service, high_cardinality_frame):
        prepared = service.bar(high_cardinality_frame, "city")
        assert "categories_truncated" in {w.code for w in prepared.warnings}

    def test_warnings_are_not_duplicated(self, service, high_cardinality_frame):
        prepared = service.bar(high_cardinality_frame, "city")
        codes = [warning.code for warning in prepared.warnings]
        assert len(codes) == len(set(codes))


class TestAnalysisReuse:
    def test_a_supplied_profile_is_used_rather_than_recomputed(self, binary_frame):
        from aidatasetkit.profiling import DataProfiler

        calls = {"count": 0}

        class CountingProfiler(DataProfiler):
            def profile(self, frame):
                calls["count"] += 1
                return super().profile(frame)

        service = VisualizationService(profiler=CountingProfiler())
        profile = DataProfiler().profile(binary_frame)
        service.recommend(binary_frame, profile=profile)
        assert calls["count"] == 0

    def test_a_supplied_quality_report_is_used(self, binary_frame):
        from aidatasetkit.profiling import DataProfiler, DataQualityInspector

        profile = DataProfiler().profile(binary_frame)
        report = DataQualityInspector().inspect(binary_frame, profile=profile)
        plan = VisualizationService().recommend(
            binary_frame, profile=profile, quality_report=report
        )
        assert plan.charts

    def test_one_context_serves_a_whole_plan(self, service, binary_frame):
        context = service.context(binary_frame, target="churn")
        plan = service.recommend(
            binary_frame,
            profile=context.profile,
            quality_report=context.quality,
            target_profile=context.target_profile,
        )
        for chart in plan.charts:
            assert service.prepare(binary_frame, chart, context=context) is not None

    def test_an_unknown_target_is_refused(self, service, binary_frame):
        with pytest.raises(SchemaError, match="not a column"):
            service.recommend(binary_frame, target="nope")


class TestNoDuplicatedAnalysis:
    def test_the_package_contains_no_second_profiler(self):
        """Column kinds, cardinality, and identity all come from one place."""
        from pathlib import Path

        import aidatasetkit.visualization as package

        forbidden = (
            "is_datetime64",
            "select_dtypes",
            "value_counts().index",
            "nunique() /",
            "def detect_kind",
            "def _is_id_like",
        )
        for path in Path(package.__file__).parent.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for marker in forbidden:
                assert marker not in source, f"{path.name} re-implements {marker}"

    def test_correlation_is_delegated_to_the_statistics_package(self):
        from pathlib import Path

        import aidatasetkit.visualization as package

        sources = [
            path.read_text(encoding="utf-8")
            for path in Path(package.__file__).parent.rglob("*.py")
        ]
        joined = "\n".join(sources)
        assert "from aidatasetkit.statistics.bivariate import correlation" in joined
        assert "np.corrcoef" not in joined
        assert ".corr(" not in joined

    def test_the_missing_view_reads_the_profile_rather_than_recounting(
        self, service, missing_heavy_frame
    ):
        prepared = service.missing(missing_heavy_frame)
        profile = service.context(missing_heavy_frame).profile
        expected = {
            str(p.name): p.missing_count for p in profile.column_profiles if p.missing_count
        }
        assert dict(zip(prepared.data["labels"], prepared.data["counts"])) == expected


class TestConfig:
    def test_defaults_are_conservative(self):
        config = VisualizationConfig()
        assert config.max_charts == 10
        assert config.max_points == 10_000

    def test_the_config_is_immutable(self):
        with pytest.raises(AttributeError):
            VisualizationConfig().max_charts = 99

    def test_replace_revalidates(self):
        from aidatasetkit.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            VisualizationConfig().replace(max_charts=0)

    def test_replace_rejects_unknown_options(self):
        from aidatasetkit.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="Unknown visualization options"):
            VisualizationConfig().replace(colour="blue")

    @pytest.mark.parametrize(
        "field", ["max_charts", "max_points", "max_categories", "max_pairwise_columns"]
    )
    def test_budgets_must_be_positive(self, field):
        from aidatasetkit.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match=field):
            VisualizationConfig(**{field: 0})

    def test_the_heatmap_range_must_be_coherent(self):
        from aidatasetkit.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="cannot exceed"):
            VisualizationConfig(min_heatmap_columns=10, max_heatmap_columns=5)

    def test_the_config_serialises(self):
        payload = json.loads(json.dumps(VisualizationConfig().to_dict()))
        assert payload["weights"]["target_relevance"] > 0

    def test_a_negative_weight_is_refused(self):
        from aidatasetkit.core.exceptions import ConfigurationError

        from aidatasetkit.visualization import ScoringWeights

        with pytest.raises(ConfigurationError, match="must not be negative"):
            ScoringWeights(target_relevance=-1.0)
