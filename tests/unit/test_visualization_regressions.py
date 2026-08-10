"""Regression tests for defects the adversarial review found and reproduced.

Each one failed before the fix beside it. They are kept together because what
they have in common is the failure mode: a plan that was correct, rendered or
prepared into something that was not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.visualization import (
    ChartSpec,
    ChartType,
    VisualizationConfig,
    VisualizationReason,
    VisualizationService,
)


@pytest.fixture
def service() -> VisualizationService:
    return VisualizationService()


def spec_for(chart_type: ChartType, *columns, target=None) -> ChartSpec:
    return ChartSpec(
        chart_type=chart_type,
        columns=tuple(columns),
        title="test",
        reason=VisualizationReason(code="test", message="test"),
        target=target,
    )


@pytest.fixture
def regression_target_frame() -> pd.DataFrame:
    index = np.arange(300)
    return pd.DataFrame(
        {
            "area": 40.0 + (index % 97) * 2.5,
            "price": 100_000.0 + (index % 89) * 3_137.0,
        }
    )


class TestTargetSurvivesTheRenderPath:
    """The plan knew the task; preparation used to forget it."""

    def test_preparing_without_a_context_keeps_the_regression_target_continuous(
        self, service, regression_target_frame
    ):
        plan = service.recommend(regression_target_frame, target="price")
        chart = plan.of_type(ChartType.TARGET_DISTRIBUTION)[0]
        prepared = service.prepare(regression_target_frame, chart)
        assert prepared.data["mode"] == "histogram"

    def test_render_plan_keeps_the_regression_target_continuous(
        self, service, regression_target_frame
    ):
        plan = service.recommend(regression_target_frame, target="price")
        context = service.context(
            regression_target_frame,
            target=service._resolve_label(regression_target_frame, plan.metadata.target_name),
        )
        chart = plan.of_type(ChartType.TARGET_DISTRIBUTION)[0]
        assert service.prepare(
            regression_target_frame, chart, context=context
        ).data["mode"] == "histogram"

    def test_a_classification_target_still_renders_as_bars(self, service, binary_frame):
        plan = service.recommend(binary_frame, target="churn")
        chart = plan.of_type(ChartType.TARGET_DISTRIBUTION)[0]
        assert service.prepare(binary_frame, chart).data["mode"] == "bar"

    def test_the_whole_plan_renders_with_the_target_intact(
        self, service, regression_target_frame
    ):
        pytest.importorskip("matplotlib")
        plan = service.recommend(regression_target_frame, target="price")
        assert len(service.render_plan(regression_target_frame, plan)) == len(plan.charts)


class TestNonStringColumnLabels:
    """A frame built from a numpy array labels its columns with integers."""

    @pytest.fixture
    def integer_labels(self) -> pd.DataFrame:
        index = np.arange(300)
        return pd.DataFrame(
            np.column_stack(
                [(index % 45).astype(float), (index % 7 == 0).astype(float)]
            )
        )

    def test_an_integer_target_label_is_honoured(self, service, integer_labels):
        plan = service.recommend(integer_labels, target=1)
        assert plan.metadata.target_name == "1"
        assert plan.of_type(ChartType.TARGET_DISTRIBUTION)

    def test_the_integer_target_is_not_charted_as_a_feature(self, service, integer_labels):
        plan = service.recommend(integer_labels, target=1)
        histograms = {c.columns[0] for c in plan.of_type(ChartType.HISTOGRAM)}
        assert 1 not in histograms

    def test_target_aware_rules_fire_for_an_integer_label(self, service, integer_labels):
        plan = service.recommend(integer_labels, target=1)
        assert plan.of_type(ChartType.GROUPED_BOX)


class TestGroupedAxesUseTheProfiledKind:
    """pandas calls a boolean numeric; the library's one classifier does not."""

    @pytest.fixture
    def boolean_feature(self) -> pd.DataFrame:
        index = np.arange(300)
        return pd.DataFrame(
            {
                "flag": (index % 2 == 0),
                "price": (100_000.0 + (index % 61) * 1_000.0),
            }
        )

    def test_a_boolean_feature_becomes_the_grouping_axis(self, service, boolean_feature):
        context = service.context(boolean_feature, target="price")
        prepared = service.prepare(
            boolean_feature,
            spec_for(ChartType.GROUPED_BOX, "flag", "price", target="price"),
            context=context,
        )
        assert prepared.data["group_label"] == "flag"
        assert prepared.data["value_label"] == "price"

    def test_a_numeric_feature_still_becomes_the_value_axis(self, service, binary_frame):
        context = service.context(binary_frame, target="churn")
        prepared = service.prepare(
            binary_frame,
            spec_for(ChartType.GROUPED_BOX, "age", "churn", target="churn"),
            context=context,
        )
        assert prepared.data["value_label"] == "age"


class TestGroupedTruncationIsDisclosed:
    """A grouped chart has no "Other" box, so the tail leaves the picture."""

    @pytest.fixture
    def many_groups(self) -> pd.DataFrame:
        index = np.arange(3_000)
        return pd.DataFrame(
            {
                "bucket": [f"b{value % 60}" for value in index],
                "value": (index % 97).astype("float64"),
            }
        )

    def test_dropped_rows_are_reported(self, service, many_groups):
        context = service.context(many_groups)
        prepared = service.prepare(
            many_groups, spec_for(ChartType.GROUPED_BOX, "value", "bucket"), context=context
        )
        codes = {warning.code for warning in prepared.warnings}
        assert "categories_truncated" in codes

    def test_the_warning_says_how_many_rows_are_not_drawn(self, service, many_groups):
        context = service.context(many_groups)
        prepared = service.prepare(
            many_groups, spec_for(ChartType.GROUPED_BOX, "value", "bucket"), context=context
        )
        warning = next(w for w in prepared.warnings if w.code == "categories_truncated")
        assert "not drawn" in warning.message

    def test_only_the_budgeted_number_of_groups_is_drawn(self, service, many_groups):
        context = service.context(many_groups)
        prepared = service.prepare(
            many_groups, spec_for(ChartType.GROUPED_BOX, "value", "bucket"), context=context
        )
        assert len(prepared.data["groups"]) <= service.config.max_categories

    def test_an_untruncated_grouped_chart_warns_about_nothing(self, service, binary_frame):
        context = service.context(binary_frame, target="churn")
        prepared = service.prepare(
            binary_frame,
            spec_for(ChartType.GROUPED_BOX, "age", "churn", target="churn"),
            context=context,
        )
        assert not prepared.warnings


class TestGroupedSamplingIsHonest:
    def test_an_unsampled_grouped_chart_does_not_claim_a_sample(self, service, binary_frame):
        context = service.context(binary_frame, target="churn")
        prepared = service.prepare(
            binary_frame,
            spec_for(ChartType.GROUPED_BOX, "age", "churn", target="churn"),
            context=context,
        )
        assert prepared.sampling.sampled is False
        assert prepared.sampling.strategy == "none"

    def test_the_point_budget_covers_the_whole_chart_not_each_group(self):
        """Per-group budgets allowed max_categories x max_points in one chart."""
        index = np.arange(60_000)
        frame = pd.DataFrame(
            {
                "grp": [f"g{value % 6}" for value in index],
                "value": (index % 977).astype("float64"),
            }
        )
        service = VisualizationService(VisualizationConfig(max_points=1_000))
        context = service.context(frame)
        prepared = service.prepare(
            frame, spec_for(ChartType.GROUPED_BOX, "value", "grp"), context=context
        )
        drawn = sum(group.size for group in prepared.data["groups"])
        assert drawn <= 1_000
        assert prepared.sampling.rendered_rows == drawn

    def test_a_stratified_sample_keeps_a_rare_class(self):
        """A plain random sample of a 0.2 percent class can miss it entirely."""
        index = np.arange(100_000)
        frame = pd.DataFrame(
            {
                "value": (index % 991).astype("float64"),
                "y": (index % 500 == 0).astype("int64"),
            }
        )
        service = VisualizationService(VisualizationConfig(max_points=1_000))
        context = service.context(frame, target="y")
        prepared = service.prepare(
            frame, spec_for(ChartType.GROUPED_BOX, "value", "y", target="y"), context=context
        )
        assert prepared.sampling.strategy == "stratified"
        assert len(prepared.data["groups"]) == 2
        assert all(group.size > 0 for group in prepared.data["groups"])


class TestScatterToleratesInfinities:
    """A recommended chart that cannot be prepared is a defect."""

    @pytest.fixture
    def infinite_frame(self) -> pd.DataFrame:
        index = np.arange(300)
        values = (index % 31).astype("float64")
        values[5] = np.inf
        return pd.DataFrame(
            {"ratio": values, "price": 100_000.0 + (index % 89) * 3_137.0}
        )

    def test_every_recommended_chart_can_be_prepared(self, service, infinite_frame):
        plan = service.recommend(infinite_frame, target="price")
        context = service.context(infinite_frame, target="price")
        for chart in plan.charts:
            service.prepare(infinite_frame, chart, context=context)

    def test_infinite_points_are_excluded_and_disclosed(self, service, infinite_frame):
        prepared = service.scatter(infinite_frame, "ratio", "price")
        assert np.isfinite(prepared.data["x"]).all()
        assert "infinite_values_excluded" in {w.code for w in prepared.warnings}

    def test_a_scatter_of_nothing_but_infinities_is_still_refused(self, service):
        from aidatasetkit.core.exceptions import InvalidVisualizationRequest

        frame = pd.DataFrame({"a": [np.inf] * 50, "b": np.arange(50, dtype="float64")})
        with pytest.raises(InvalidVisualizationRequest, match="no finite pairs"):
            service.scatter(frame, "a", "b")


class TestScoresDoNotBorrowEvidence:
    """A chart's score must rest on its own measurement, not a neighbour's."""

    def test_a_dataset_level_chart_earns_no_statistical_evidence(self, service):
        index = np.arange(300)
        frame = pd.DataFrame(
            {
                "a": np.where(index % 5 == 0, np.nan, (index % 37).astype("float64")),
                "b": (index % 37).astype("float64") * 2.0,
                "y": (index % 7 == 0).astype("int64"),
            }
        )
        plan = service.recommend(frame, target="y")
        missing = plan.of_type(ChartType.MISSING_VALUES)[0]
        factors = {c.factor for c in missing.score.contributions}
        assert "statistical_evidence" not in factors

    def test_a_relationship_chart_with_a_measurement_still_earns_it(
        self, service, regression_target_frame
    ):
        plan = service.recommend(regression_target_frame, target="price")
        measured = [c for c in plan.charts if "abs_correlation" in c.reason.evidence]
        assert measured
        for chart in measured:
            assert "statistical_evidence" in {c.factor for c in chart.score.contributions}

    def test_column_order_does_not_change_any_score(self, service):
        index = np.arange(300)
        frame = pd.DataFrame(
            {
                "a": np.where(index % 5 == 0, np.nan, (index % 37).astype("float64")),
                "b": (index % 41).astype("float64"),
                "y": (index % 7 == 0).astype("int64"),
            }
        )
        flipped = frame[["b", "a", "y"]]
        original = {c.identity(): c.priority for c in service.recommend(frame, target="y")}
        reordered = {c.identity(): c.priority for c in service.recommend(flipped, target="y")}
        assert original == reordered


class TestNoCorrelationAgainstNominalClasses:
    """Class 2 is not twice class 1, so Pearson against codes means nothing."""

    @pytest.fixture
    def multiclass_codes(self) -> pd.DataFrame:
        index = np.arange(300)
        return pd.DataFrame(
            {
                "feature": (index % 53).astype("float64"),
                "y": (index % 3).astype("int64"),
            }
        )

    def test_no_correlation_is_claimed_for_a_multiclass_target(
        self, service, multiclass_codes
    ):
        plan = service.recommend(multiclass_codes, target="y")
        for chart in plan.charts:
            if chart.target is not None:
                assert "abs_correlation" not in chart.reason.evidence

    def test_the_feature_is_still_compared_with_the_target(self, service, multiclass_codes):
        plan = service.recommend(multiclass_codes, target="y")
        assert plan.of_type(ChartType.GROUPED_BOX)

    def test_a_binary_target_still_earns_a_measurement(self, service, binary_frame):
        plan = service.recommend(binary_frame, target="churn")
        measured = [c for c in plan.charts if "abs_correlation" in c.reason.evidence]
        assert measured


class TestPreparerHonoursTheContextBudgets:
    def test_a_context_config_overrides_the_preparer_default(self, binary_frame):
        from aidatasetkit.visualization.preparation import ChartPreparer

        service = VisualizationService(VisualizationConfig(max_categories=2))
        context = service.context(binary_frame)
        prepared = ChartPreparer(VisualizationConfig()).prepare(
            context, spec_for(ChartType.BAR, "segment")
        )
        assert len(prepared.data["labels"]) == 3


class TestMissingViewHonoursItsSpec:
    def test_a_spec_naming_one_column_draws_only_that_column(
        self, service, missing_heavy_frame
    ):
        context = service.context(missing_heavy_frame)
        prepared = service.prepare(
            missing_heavy_frame,
            spec_for(ChartType.MISSING_VALUES, "gappy"),
            context=context,
        )
        assert prepared.data["labels"] == ["gappy"]

    def test_the_recommended_chart_still_covers_every_affected_column(
        self, service, missing_heavy_frame
    ):
        plan = service.recommend(missing_heavy_frame)
        chart = plan.of_type(ChartType.MISSING_VALUES)[0]
        prepared = service.prepare(missing_heavy_frame, chart)
        assert set(prepared.data["labels"]) == {"gappy", "gappier"}


class TestAutomaticBarDisclosesMissingValues:
    def test_a_recommended_bar_chart_discloses_dropped_rows(self, service):
        index = np.arange(200)
        frame = pd.DataFrame(
            {"segment": np.where(index % 4 == 0, None, [["A", "B"][v % 2] for v in index])}
        )
        context = service.context(frame)
        prepared = service.prepare(
            frame, spec_for(ChartType.BAR, "segment"), context=context
        )
        assert "missing_values_excluded" in {w.code for w in prepared.warnings}
        assert sum(prepared.data["counts"]) == int(frame["segment"].notna().sum())


class TestShowDoesNotLeakFigures:
    def test_rendering_a_plan_opens_no_pyplot_figure(self, service, binary_frame):
        pyplot = pytest.importorskip("matplotlib.pyplot")
        pyplot.close("all")
        before = len(pyplot.get_fignums())
        service.render_plan(binary_frame, service.recommend(binary_frame, target="churn"))
        assert len(pyplot.get_fignums()) == before

    def test_the_renderer_can_draw_into_a_caller_owned_figure(self, service, binary_frame):
        pyplot = pytest.importorskip("matplotlib.pyplot")
        pyplot.close("all")
        figure = pyplot.figure()
        returned = service.renderer.render(
            service.histogram(binary_frame, "age"), figure=figure
        )
        assert returned is figure
        assert figure.axes
        pyplot.close("all")
