"""Tests for what the advisor recommends, and what it refuses to.

The recommendation rules are the product here, so most of these assertions are
semantic: does a numeric column get a distribution, does an identifier get
nothing, is the target chart at the top. Whole-plan comparisons are avoided --
they break for reasons that have nothing to do with the rule under test.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.visualization import (
    ChartType,
    VisualizationConfig,
    VisualizationService,
)


@pytest.fixture
def service() -> VisualizationService:
    return VisualizationService()


def plan_for(service: VisualizationService, frame: pd.DataFrame, **kwargs):
    return service.recommend(frame, **kwargs)


def columns_of(plan, chart_type: ChartType) -> set[str]:
    return {str(chart.columns[0]) for chart in plan.of_type(chart_type)}


def suppression_codes(plan, column: str) -> set[str]:
    return {
        entry.reason.code
        for entry in plan.suppressed
        if column in {str(name) for name in entry.columns}
    }


class TestNumericColumns:
    def test_a_meaningful_numeric_column_gets_a_histogram(self, service, binary_frame):
        plan = plan_for(service, binary_frame)
        assert "age" in columns_of(plan, ChartType.HISTOGRAM)

    def test_the_histogram_explains_itself_with_evidence(self, service, binary_frame):
        chart = next(
            c for c in plan_for(service, binary_frame).of_type(ChartType.HISTOGRAM)
        )
        assert chart.reason.code == "numeric_distribution"
        assert chart.reason.evidence["detected_kind"] == "numeric"
        assert chart.reason.evidence["unique_count"] > 1

    def test_flagged_outliers_raise_the_box_plot_above_the_histogram(
        self, service, outlier_frame
    ):
        plan = plan_for(service, outlier_frame)
        box = next(c for c in plan.charts if c.chart_type is ChartType.BOX_PLOT)
        histogram = next(
            c
            for c in plan.charts
            if c.chart_type is ChartType.HISTOGRAM and c.columns == box.columns
        )
        assert box.reason.code == "flagged_outlier_candidates"
        assert box.priority > histogram.priority

    def test_the_outlier_reason_carries_the_inspector_s_numbers(
        self, service, outlier_frame
    ):
        plan = plan_for(service, outlier_frame)
        box = next(
            c
            for c in plan.charts
            if c.chart_type is ChartType.BOX_PLOT and str(c.columns[0]) == "measurement"
        )
        assert box.reason.evidence["outlier_count"] >= 2
        assert box.reason.evidence["lower_bound"] < box.reason.evidence["upper_bound"]


class TestCategoricalColumns:
    def test_a_low_cardinality_category_gets_a_bar(self, service, binary_frame):
        assert "segment" in columns_of(plan_for(service, binary_frame), ChartType.BAR)

    def test_a_high_cardinality_category_is_charted_but_flagged_for_truncation(
        self, service, high_cardinality_frame
    ):
        plan = plan_for(service, high_cardinality_frame)
        bar = next(c for c in plan.of_type(ChartType.BAR) if str(c.columns[0]) == "city")
        assert {warning.code for warning in bar.warnings} == {"categories_truncated"}

    def test_a_high_cardinality_category_is_not_called_an_identifier(
        self, service, high_cardinality_frame
    ):
        plan = plan_for(service, high_cardinality_frame)
        assert "possible_id_like" not in suppression_codes(plan, "city")


class TestSuppression:
    def test_a_numeric_identifier_gets_no_automatic_histogram(
        self, service, identifier_frame
    ):
        """dtype alone must never be enough to chart a key."""
        plan = plan_for(service, identifier_frame)
        assert "row_id" not in columns_of(plan, ChartType.HISTOGRAM)
        assert "possible_id_like" in suppression_codes(plan, "row_id")

    def test_a_textual_identifier_gets_no_automatic_bar(self, service, identifier_frame):
        plan = plan_for(service, identifier_frame)
        assert "customer_id" not in columns_of(plan, ChartType.BAR)
        assert "possible_id_like" in suppression_codes(plan, "customer_id")

    def test_a_unique_measurement_is_still_charted(self, service, identifier_frame):
        """Uniqueness is not identity; a temperature reading is a measurement."""
        plan = plan_for(service, identifier_frame)
        assert "temperature" in columns_of(plan, ChartType.HISTOGRAM)

    def test_a_constant_column_is_suppressed_with_a_reason(self, service, constant_frame):
        plan = plan_for(service, constant_frame)
        assert "country" not in columns_of(plan, ChartType.BAR)
        assert "constant_column" in suppression_codes(plan, "country")

    def test_a_near_constant_column_is_penalised_not_removed(
        self, service, constant_frame
    ):
        """The documented policy: demote, do not silently delete."""
        plan = plan_for(service, constant_frame)
        bar = next(c for c in plan.of_type(ChartType.BAR) if str(c.columns[0]) == "plan")
        factors = {c.factor for c in bar.score.contributions}
        assert "near_constant_penalty" in factors
        age = next(
            c for c in plan.of_type(ChartType.HISTOGRAM) if str(c.columns[0]) == "age"
        )
        assert age.priority > bar.priority

    def test_a_column_with_too_few_rows_is_suppressed(self, service):
        """Values chosen non-sequential so the identifier rule does not fire first."""
        frame = pd.DataFrame({"tiny": [3.5, 91.2, 17.8]})
        plan = plan_for(service, frame)
        assert "too_few_observations" in suppression_codes(plan, "tiny")

    def test_datetime_columns_are_not_charted_in_this_version(self, service):
        frame = pd.DataFrame(
            {
                "when": pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(50), "D"),
                "value": np.arange(50, dtype="float64"),
            }
        )
        plan = plan_for(service, frame)
        assert "unsupported_column_kind" in suppression_codes(plan, "when")


class TestMissingValues:
    def test_a_frame_with_gaps_gets_a_missing_chart(self, service, missing_heavy_frame):
        plan = plan_for(service, missing_heavy_frame)
        assert plan.of_type(ChartType.MISSING_VALUES)

    def test_a_frame_without_gaps_gets_no_empty_missing_chart(self, service, binary_frame):
        plan = plan_for(service, binary_frame)
        assert not plan.of_type(ChartType.MISSING_VALUES)
        assert any(entry.reason.code == "no_missing_values" for entry in plan.suppressed)

    def test_the_missing_chart_names_only_the_affected_columns(
        self, service, missing_heavy_frame
    ):
        chart = plan_for(service, missing_heavy_frame).of_type(ChartType.MISSING_VALUES)[0]
        assert {str(c) for c in chart.columns} == {"gappy", "gappier"}


class TestTargetAwareness:
    def test_a_classification_target_gets_a_distribution_chart(self, service, binary_frame):
        plan = plan_for(service, binary_frame, target="churn")
        assert plan.of_type(ChartType.TARGET_DISTRIBUTION)

    def test_the_target_chart_outranks_generic_exploration(self, service, binary_frame):
        plan = plan_for(service, binary_frame, target="churn")
        assert plan.charts[0].chart_type is ChartType.TARGET_DISTRIBUTION

    def test_a_numeric_feature_is_compared_with_a_classification_target(
        self, service, binary_frame
    ):
        plan = plan_for(service, binary_frame, target="churn")
        grouped = {str(c.columns[0]) for c in plan.of_type(ChartType.GROUPED_BOX)}
        assert {"age", "income"} & grouped

    def test_a_categorical_feature_is_compared_with_a_classification_target(
        self, service, binary_frame
    ):
        plan = plan_for(service, binary_frame, target="churn")
        assert {str(c.columns[0]) for c in plan.of_type(ChartType.GROUPED_BAR)} == {
            "segment"
        }

    def test_a_multiclass_target_is_handled(self, service, multiclass_frame):
        plan = plan_for(service, multiclass_frame, target="grade")
        assert plan.of_type(ChartType.TARGET_DISTRIBUTION)
        assert plan.metadata.task_type.value == "classification"

    def test_a_regression_target_gets_a_distribution_and_scatters(
        self, service, regression_frame
    ):
        plan = plan_for(service, regression_frame, target="price")
        assert plan.of_type(ChartType.TARGET_DISTRIBUTION)
        scatter_targets = {
            str(c.columns[1]) for c in plan.of_type(ChartType.SCATTER) if c.target
        }
        assert scatter_targets == {"price"}

    def test_an_imbalanced_target_is_promoted_and_says_why(self, service):
        index = np.arange(400)
        frame = pd.DataFrame(
            {"feature": (index % 31).astype("float64"), "y": (index % 50 == 0).astype(int)}
        )
        plan = plan_for(service, frame, target="y")
        chart = plan.of_type(ChartType.TARGET_DISTRIBUTION)[0]
        assert "imbalance" in chart.reason.message.lower()
        assert any(
            c.factor == "quality_relevance" for c in chart.score.contributions
        )

    def test_no_target_means_no_target_charts(self, service, binary_frame):
        plan = plan_for(service, binary_frame)
        assert not plan.of_type(ChartType.TARGET_DISTRIBUTION)
        assert plan.metadata.target_name is None

    def test_an_undetectable_target_does_not_abort_planning(self, service):
        """An ambiguous target loses the target charts, nothing else."""
        frame = pd.DataFrame(
            {
                "x": ((np.arange(50) * 977) % 313).astype("float64"),
                "y": [1, 2, 3, 4, 5] * 10,
            }
        )
        plan = service.recommend(frame, target="y")
        assert plan.charts
        assert not plan.of_type(ChartType.TARGET_DISTRIBUTION)

    def test_an_undetectable_target_is_not_charted_as_an_ordinary_feature(self, service):
        frame = pd.DataFrame(
            {
                "x": ((np.arange(50) * 977) % 313).astype("float64"),
                "y": [1, 2, 3, 4, 5] * 10,
            }
        )
        plan = service.recommend(frame, target="y")
        assert "y" not in {str(c.columns[0]) for c in plan.charts}


class TestLeakageLanguage:
    @pytest.fixture
    def leaky_frame(self) -> pd.DataFrame:
        index = np.arange(300)
        churn = (index % 7 == 0).astype("int64")
        return pd.DataFrame(
            {
                "tenure": (index % 41).astype("float64"),
                "churn_reason": np.where(churn == 1, "price", "none"),
                "churn": churn,
            }
        )

    def test_a_flagged_relationship_is_surfaced_for_review(self, service, leaky_frame):
        plan = plan_for(service, leaky_frame, target="churn")
        flagged = [
            c for c in plan.charts if c.reason.code == "possible_target_leakage_review"
        ]
        assert flagged
        assert all(chart.reason.requires_review for chart in flagged)

    def test_the_wording_never_claims_leakage_was_proved(self, service, leaky_frame):
        plan = plan_for(service, leaky_frame, target="churn")
        for chart in plan.charts:
            message = chart.reason.message.lower()
            assert "proves" not in message
            assert "confirms" not in message
            if chart.reason.code == "possible_target_leakage_review":
                assert "possible" in message and "review" in message

    def test_nothing_is_removed_on_account_of_leakage(self, service, leaky_frame):
        before = leaky_frame.copy(deep=True)
        plan_for(service, leaky_frame, target="churn")
        pd.testing.assert_frame_equal(leaky_frame, before)


class TestEvidenceLanguage:
    def test_a_correlation_claim_records_the_measurement(self, service, regression_frame):
        plan = plan_for(service, regression_frame, target="price")
        measured = [
            c for c in plan.charts if "abs_correlation" in c.reason.evidence
        ]
        for chart in measured:
            assert 0.0 <= chart.reason.evidence["abs_correlation"] <= 1.0
            assert chart.reason.evidence["correlation_method"] == "pearson"

    def test_no_recommendation_claims_causation(self, service, binary_frame):
        plan = plan_for(service, binary_frame, target="churn")
        for chart in plan.charts:
            lowered = chart.reason.message.lower()
            assert "causes" not in lowered
            assert "because of" not in lowered
            assert "ai discovered" not in lowered

    def test_every_chart_carries_a_reason(self, service, binary_frame):
        for chart in plan_for(service, binary_frame, target="churn").charts:
            assert chart.reason.code and chart.reason.message

    def test_every_suppression_carries_a_reason(self, service, identifier_frame):
        for entry in plan_for(service, identifier_frame).suppressed:
            assert entry.reason.code and entry.reason.message


class TestHeatmap:
    def test_too_few_numeric_columns_means_no_heatmap(self, service, binary_frame):
        """With churn as the target, only two numeric features remain."""
        plan = plan_for(service, binary_frame, target="churn")
        assert not plan.of_type(ChartType.CORRELATION_HEATMAP)
        assert any(
            entry.reason.code == "too_few_numeric_columns" for entry in plan.suppressed
        )

    def test_enough_numeric_columns_earns_a_heatmap(self, service, wide_frame):
        plan = plan_for(service, wide_frame)
        assert plan.of_type(ChartType.CORRELATION_HEATMAP)

    def test_a_wide_frame_gets_a_bounded_readable_heatmap(self, service, wide_frame):
        config = VisualizationConfig()
        chart = plan_for(service, wide_frame).of_type(ChartType.CORRELATION_HEATMAP)[0]
        assert len(chart.columns) <= config.max_heatmap_columns
        assert chart.warnings[0].code == "heatmap_columns_truncated"

    def test_the_heatmap_records_which_columns_it_used(self, service, wide_frame):
        chart = plan_for(service, wide_frame).of_type(ChartType.CORRELATION_HEATMAP)[0]
        assert len(chart.reason.evidence["included_columns"]) == len(chart.columns)


class TestBoundedGeneration:
    def test_a_wide_frame_does_not_explode_into_pairs(self, service, wide_frame):
        """200 numeric columns make 19,900 pairs; none of them is enumerated."""
        plan = plan_for(service, wide_frame)
        scatters = plan.of_type(ChartType.SCATTER)
        assert len(scatters) <= VisualizationConfig().max_pairwise_charts

    def test_candidate_generation_stays_proportional_to_columns(self, service, wide_frame):
        plan = plan_for(service, wide_frame)
        assert plan.metadata.candidates_generated < 3 * len(wide_frame.columns)

    def test_pairwise_charts_respect_the_configured_limit(self, service, wide_frame):
        service = VisualizationService(VisualizationConfig(max_pairwise_charts=1))
        plan = service.recommend(wide_frame)
        assert len(plan.of_type(ChartType.SCATTER)) <= 1

    def test_pairwise_generation_can_be_switched_off(self, service, wide_frame):
        service = VisualizationService(VisualizationConfig(max_pairwise_charts=0))
        plan = service.recommend(wide_frame)
        assert not [c for c in plan.of_type(ChartType.SCATTER) if c.target is None]


class TestBudget:
    def test_max_charts_is_never_exceeded(self, wide_frame):
        for limit in (1, 3, 5, 10):
            service = VisualizationService(VisualizationConfig(max_charts=limit))
            assert len(service.recommend(wide_frame).charts) <= limit

    def test_charts_cut_by_the_budget_are_recorded(self, wide_frame):
        service = VisualizationService(VisualizationConfig(max_charts=2))
        plan = service.recommend(wide_frame)
        assert any(entry.reason.code == "over_max_charts" for entry in plan.suppressed)

    def test_the_survivors_are_the_highest_ranked(self, wide_frame):
        full = VisualizationService(VisualizationConfig(max_charts=50)).recommend(wide_frame)
        small = VisualizationService(VisualizationConfig(max_charts=3)).recommend(wide_frame)
        assert [c.identity() for c in small.charts] == [
            c.identity() for c in full.charts[:3]
        ]


class TestDeterminism:
    def test_the_same_input_yields_the_same_plan(self, binary_frame):
        first = VisualizationService().recommend(binary_frame, target="churn")
        second = VisualizationService().recommend(binary_frame, target="churn")
        assert first.to_dict() == second.to_dict()

    def test_repeated_calls_on_one_service_agree(self, service, wide_frame):
        assert service.recommend(wide_frame).to_dict() == service.recommend(
            wide_frame
        ).to_dict()

    def test_scores_are_stable(self, service, binary_frame):
        first = [c.priority for c in service.recommend(binary_frame, target="churn").charts]
        second = [c.priority for c in service.recommend(binary_frame, target="churn").charts]
        assert first == second

    def test_ordering_is_by_score_then_a_stable_tie_break(self, service, wide_frame):
        charts = service.recommend(wide_frame).charts
        keys = [
            (-c.priority, c.chart_type.value, tuple(str(x) for x in c.columns))
            for c in charts
        ]
        assert keys == sorted(keys)

    def test_column_order_does_not_change_the_recommendation_set(self, service, binary_frame):
        reversed_frame = binary_frame[list(binary_frame.columns)[::-1]]
        original = {c.identity() for c in service.recommend(binary_frame, target="churn")}
        flipped = {c.identity() for c in service.recommend(reversed_frame, target="churn")}
        assert original == flipped


class TestRedundancy:
    def test_a_second_view_of_one_distribution_is_dropped(self, service):
        """No outliers were flagged, so the box plot adds nothing to the histogram."""
        frame = pd.DataFrame({"steady": np.linspace(0.0, 100.0, 200)})
        plan = service.recommend(frame)
        assert len(plan.of_type(ChartType.HISTOGRAM)) == 1
        assert not plan.of_type(ChartType.BOX_PLOT)
        assert any(
            entry.reason.code == "redundant_distribution_view"
            for entry in plan.suppressed
        )

    def test_both_views_survive_when_outliers_justify_the_second(
        self, service, outlier_frame
    ):
        plan = service.recommend(outlier_frame)
        assert "measurement" in columns_of(plan, ChartType.HISTOGRAM)
        assert "measurement" in columns_of(plan, ChartType.BOX_PLOT)

    def test_no_two_charts_share_an_identity(self, service, binary_frame):
        plan = service.recommend(binary_frame, target="churn")
        identities = [chart.identity() for chart in plan.charts]
        assert len(identities) == len(set(identities))

    def test_scatter_identity_ignores_axis_order(self):
        from aidatasetkit.visualization.types import ChartSpec, VisualizationReason

        reason = VisualizationReason(code="x", message="y")
        first = ChartSpec(ChartType.SCATTER, ("a", "b"), "t", reason)
        second = ChartSpec(ChartType.SCATTER, ("b", "a"), "t", reason)
        assert first.identity() == second.identity()


class TestPlanMetadata:
    def test_the_plan_records_how_it_came_about(self, service, binary_frame):
        plan = service.recommend(binary_frame, target="churn")
        meta = plan.metadata
        assert meta.row_count == len(binary_frame)
        assert meta.column_count == len(binary_frame.columns)
        assert meta.candidates_generated >= meta.charts_returned
        assert meta.charts_returned == len(plan.charts)
        assert meta.target_name == "churn"
        assert meta.task_type.value == "classification"

    def test_the_config_that_produced_the_plan_is_recorded(self, service, binary_frame):
        plan = service.recommend(binary_frame)
        assert plan.metadata.config["max_charts"] == service.config.max_charts

    def test_expected_sampling_is_flagged_before_any_rendering(self):
        frame = pd.DataFrame({"x": np.arange(50_000, dtype="float64")})
        service = VisualizationService(VisualizationConfig(max_points=1_000))
        assert service.recommend(frame).metadata.sampling_expected is True

    def test_small_frames_expect_no_sampling(self, service, binary_frame):
        assert service.recommend(binary_frame).metadata.sampling_expected is False


class TestSerialisation:
    def test_a_plan_round_trips_through_json(self, service, binary_frame):
        payload = json.loads(json.dumps(service.recommend(binary_frame, target="churn").to_dict()))
        assert payload["metadata"]["target_name"] == "churn"
        assert payload["charts"]
        assert all("reason" in chart for chart in payload["charts"])

    def test_no_frame_or_array_leaks_into_the_plan(self, service, binary_frame):
        payload = service.recommend(binary_frame, target="churn").to_dict()
        text = json.dumps(payload)
        for forbidden in ("ndarray", "DataFrame", "<Figure", "AxesSubplot", "object at 0x"):
            assert forbidden not in text
        # The renderer appears only as a configured name, never as an object.
        assert payload["metadata"]["config"]["renderer"] == "matplotlib"

    def test_scores_and_contributions_serialise(self, service, binary_frame):
        payload = service.recommend(binary_frame, target="churn").to_dict()
        score = payload["charts"][0]["score"]
        assert isinstance(score["total"], float)
        assert score["contributions"]

    def test_non_string_column_labels_serialise_safely(self, service):
        """Values are scattered so the column is a measurement, not a counter."""
        frame = pd.DataFrame({0: ((np.arange(50) * 977) % 313).astype("float64")})
        plan = service.recommend(frame)
        assert plan.charts[0].columns == (0,)
        assert json.loads(json.dumps(plan.to_dict()))["charts"][0]["columns"] == ["0"]


class TestImmutability:
    def test_recommending_does_not_modify_the_frame(self, service, binary_frame):
        before = binary_frame.copy(deep=True)
        service.recommend(binary_frame, target="churn")
        pd.testing.assert_frame_equal(binary_frame, before)

    def test_recommending_does_not_change_dtypes(self, service, missing_heavy_frame):
        before = missing_heavy_frame.dtypes.copy()
        service.recommend(missing_heavy_frame)
        pd.testing.assert_series_equal(missing_heavy_frame.dtypes, before)

    def test_missing_values_survive_recommendation(self, service, missing_heavy_frame):
        before = int(missing_heavy_frame.isna().sum().sum())
        service.recommend(missing_heavy_frame)
        assert int(missing_heavy_frame.isna().sum().sum()) == before

    def test_the_index_is_untouched(self, service, binary_frame):
        before = binary_frame.index.copy()
        service.recommend(binary_frame, target="churn")
        pd.testing.assert_index_equal(binary_frame.index, before)
