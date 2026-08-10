"""Tests for render preparation: sampling, truncation, and immutability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core.exceptions import InvalidVisualizationRequest
from aidatasetkit.visualization import (
    ChartSpec,
    ChartType,
    VisualizationConfig,
    VisualizationService,
    VisualizationReason,
)
from aidatasetkit.visualization.preparation import OTHER_LABEL


def spec_for(chart_type: ChartType, *columns, target=None) -> ChartSpec:
    return ChartSpec(
        chart_type=chart_type,
        columns=tuple(columns),
        title="test",
        reason=VisualizationReason(code="test", message="test"),
        target=target,
    )


@pytest.fixture
def service() -> VisualizationService:
    return VisualizationService()


class TestSampling:
    @pytest.fixture
    def big_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"x": np.arange(50_000, dtype="float64") % 9_973})

    def test_a_small_frame_is_not_sampled(self, service, binary_frame):
        prepared = service.histogram(binary_frame, "age")
        assert prepared.sampling.sampled is False
        assert prepared.sampling.strategy == "none"
        assert prepared.sampling.rendered_rows == prepared.sampling.source_rows

    def test_a_large_frame_is_sampled_to_the_budget(self, big_frame):
        service = VisualizationService(VisualizationConfig(max_points=1_000))
        prepared = service.histogram(big_frame, "x")
        assert prepared.sampling.sampled is True
        assert prepared.sampling.rendered_rows == 1_000
        assert prepared.sampling.source_rows == 50_000
        assert len(prepared.data["values"]) == 1_000

    def test_the_sampling_record_names_the_seed_and_strategy(self, big_frame):
        service = VisualizationService(
            VisualizationConfig(max_points=1_000, random_state=7)
        )
        info = service.histogram(big_frame, "x").sampling
        assert info.strategy == "random"
        assert info.random_state == 7

    def test_the_same_seed_draws_the_same_sample(self, big_frame):
        service = VisualizationService(VisualizationConfig(max_points=500))
        first = service.histogram(big_frame, "x").data["values"]
        second = service.histogram(big_frame, "x").data["values"]
        np.testing.assert_array_equal(first, second)

    def test_a_different_seed_draws_a_different_sample(self, big_frame):
        first = VisualizationService(
            VisualizationConfig(max_points=500, random_state=1)
        ).histogram(big_frame, "x").data["values"]
        second = VisualizationService(
            VisualizationConfig(max_points=500, random_state=2)
        ).histogram(big_frame, "x").data["values"]
        assert not np.array_equal(first, second)

    def test_sampling_never_invents_a_value(self, big_frame):
        service = VisualizationService(VisualizationConfig(max_points=500))
        drawn = set(service.histogram(big_frame, "x").data["values"].tolist())
        assert drawn <= set(big_frame["x"].tolist())

    def test_a_sampled_scatter_records_the_same_facts(self):
        frame = pd.DataFrame(
            {
                "a": np.arange(30_000, dtype="float64"),
                "b": (np.arange(30_000) * 7 % 991).astype("float64"),
            }
        )
        service = VisualizationService(VisualizationConfig(max_points=2_000))
        prepared = service.scatter(frame, "a", "b")
        assert prepared.sampling.sampled is True
        assert len(prepared.data["x"]) == len(prepared.data["y"]) == 2_000

    def test_the_frame_is_unchanged_by_sampling(self, big_frame):
        before = big_frame.copy(deep=True)
        VisualizationService(VisualizationConfig(max_points=100)).histogram(big_frame, "x")
        pd.testing.assert_frame_equal(big_frame, before)


class TestHighCardinality:
    @pytest.fixture
    def many_categories(self) -> pd.DataFrame:
        index = np.arange(50_000)
        return pd.DataFrame({"code": [f"c{value % 50_000}" for value in index]})

    def test_fifty_thousand_categories_do_not_become_fifty_thousand_bars(
        self, service, many_categories
    ):
        prepared = service.bar(many_categories, "code")
        assert len(prepared.data["labels"]) <= service.config.max_categories + 1

    def test_the_tail_is_folded_into_one_other_bucket(self, service, many_categories):
        prepared = service.bar(many_categories, "code")
        assert prepared.data["labels"][-1] == OTHER_LABEL
        assert prepared.data["folded_categories"] > 0

    def test_every_observation_is_still_counted(self, service, many_categories):
        prepared = service.bar(many_categories, "code")
        assert sum(prepared.data["counts"]) == int(many_categories["code"].notna().sum())

    def test_counts_are_preserved_when_values_are_missing(self, service):
        frame = pd.DataFrame({"c": ["a"] * 30 + ["b"] * 20 + [None] * 10})
        prepared = service.bar(frame, "c")
        assert sum(prepared.data["counts"]) == 50

    def test_no_truncation_when_the_column_fits(self, service, binary_frame):
        prepared = service.bar(binary_frame, "segment")
        assert prepared.data["folded_categories"] == 0
        assert OTHER_LABEL not in prepared.data["labels"]

    def test_the_budget_is_configurable(self, many_categories):
        service = VisualizationService(VisualizationConfig(max_categories=5))
        prepared = service.bar(many_categories, "code")
        assert len(prepared.data["labels"]) == 6

    def test_the_most_frequent_categories_are_the_ones_kept(self, service):
        frame = pd.DataFrame({"c": ["common"] * 100 + [f"rare{i}" for i in range(60)]})
        prepared = service.bar(frame, "c")
        assert prepared.data["labels"][0] == "common"
        assert prepared.data["counts"][0] == 100


class TestExclusions:
    def test_missing_values_are_excluded_and_disclosed(self, service, missing_heavy_frame):
        prepared = service.histogram(missing_heavy_frame, "gappy")
        codes = {warning.code for warning in prepared.warnings}
        assert "missing_values_excluded" in codes
        assert not np.isnan(prepared.data["values"]).any()

    def test_infinities_are_excluded_and_disclosed(self, service):
        frame = pd.DataFrame({"r": [1.0, 2.0, np.inf, -np.inf] * 20})
        prepared = service.histogram(frame, "r")
        assert "infinite_values_excluded" in {w.code for w in prepared.warnings}
        assert np.isfinite(prepared.data["values"]).all()

    def test_a_scatter_drops_whole_pairs_not_single_axes(self, service):
        frame = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0, 4.0] * 10,
                "b": [10.0, 20.0, np.nan, 40.0] * 10,
            }
        )
        prepared = service.scatter(frame, "a", "b")
        assert len(prepared.data["x"]) == len(prepared.data["y"]) == 20
        np.testing.assert_array_equal(
            np.unique(prepared.data["x"]), np.array([1.0, 4.0])
        )

    def test_incomplete_pairs_are_disclosed(self, service):
        frame = pd.DataFrame({"a": [1.0, np.nan] * 20, "b": [1.0, 2.0] * 20})
        prepared = service.scatter(frame, "a", "b")
        assert "incomplete_pairs_excluded" in {w.code for w in prepared.warnings}

    def test_a_scatter_with_no_complete_pairs_is_an_error_not_an_empty_plot(self, service):
        frame = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]})
        with pytest.raises(InvalidVisualizationRequest):
            service.scatter(frame, "a", "b")


class TestHeatmapPreparation:
    def test_the_matrix_is_square_with_a_unit_diagonal(self, service, wide_frame):
        prepared = service.correlation(wide_frame, ["f000", "f001", "f002"])
        matrix = prepared.data["matrix"]
        assert len(matrix) == len(matrix[0]) == 3
        assert all(matrix[i][i] == 1.0 for i in range(3))

    def test_the_matrix_is_symmetric(self, service, wide_frame):
        matrix = service.correlation(wide_frame, ["f000", "f001", "f002"]).data["matrix"]
        for i in range(3):
            for j in range(3):
                assert matrix[i][j] == matrix[j][i]

    def test_values_agree_with_the_statistics_package(self, service, wide_frame):
        from aidatasetkit.statistics import correlation

        prepared = service.correlation(wide_frame, ["f000", "f001"])
        expected = correlation(wide_frame["f000"], wide_frame["f001"])
        assert prepared.data["matrix"][0][1] == pytest.approx(expected)

    def test_an_undefined_pair_is_blank_rather_than_zero(self, service):
        """Zero would read as "measured no relationship"."""
        frame = pd.DataFrame(
            {
                "flat": [5.0] * 60,
                "a": np.arange(60, dtype="float64") % 17,
                "b": np.arange(60, dtype="float64") % 23,
            }
        )
        prepared = service.correlation(frame, ["flat", "a", "b"])
        assert prepared.data["matrix"][0][1] is None
        assert "undefined_correlations" in {w.code for w in prepared.warnings}

    def test_missing_values_do_not_break_the_matrix(self, service, missing_heavy_frame):
        prepared = service.correlation(
            missing_heavy_frame, ["complete", "gappy", "gappier"]
        )
        assert prepared.data["matrix"][0][1] is not None


class TestGroupedPreparation:
    def test_a_grouped_box_splits_values_by_category(self, service, binary_frame):
        context = service.context(binary_frame, target="churn")
        prepared = service.prepare(
            binary_frame, spec_for(ChartType.GROUPED_BOX, "age", "churn"), context=context
        )
        assert len(prepared.data["groups"]) == 2
        assert prepared.data["value_label"] == "age"

    def test_a_grouped_bar_counts_classes_within_categories(self, service, binary_frame):
        context = service.context(binary_frame, target="churn")
        prepared = service.prepare(
            binary_frame,
            spec_for(ChartType.GROUPED_BAR, "segment", "churn"),
            context=context,
        )
        assert prepared.data["labels"]
        assert len(prepared.data["counts"][0]) == len(prepared.data["class_labels"])
        assert sum(sum(row) for row in prepared.data["counts"]) == len(binary_frame)

    def test_a_regression_grouped_box_puts_the_target_on_the_value_axis(self, service):
        index = np.arange(120)
        frame = pd.DataFrame(
            {
                "grade": [["a", "b", "c"][v % 3] for v in index],
                "price": (index % 61).astype("float64") * 1_000,
            }
        )
        context = service.context(frame, target="price")
        prepared = service.prepare(
            frame, spec_for(ChartType.GROUPED_BOX, "grade", "price"), context=context
        )
        assert prepared.data["value_label"] == "price"
        assert prepared.data["group_label"] == "grade"


class TestImmutability:
    @pytest.mark.parametrize(
        "call",
        [
            lambda s, f: s.histogram(f, "age"),
            lambda s, f: s.boxplot(f, "income"),
            lambda s, f: s.bar(f, "segment"),
            lambda s, f: s.scatter(f, "age", "income"),
            lambda s, f: s.correlation(f, ["age", "income"]),
            lambda s, f: s.recommend(f, target="churn"),
        ],
    )
    def test_no_preparation_path_modifies_the_frame(self, service, binary_frame, call):
        before = binary_frame.copy(deep=True)
        call(service, binary_frame)
        pd.testing.assert_frame_equal(binary_frame, before)

    def test_preparing_a_plan_leaves_the_frame_alone(self, service, missing_heavy_frame):
        before = missing_heavy_frame.copy(deep=True)
        plan = service.recommend(missing_heavy_frame)
        context = service.context(missing_heavy_frame)
        for chart in plan.charts:
            service.prepare(missing_heavy_frame, chart, context=context)
        pd.testing.assert_frame_equal(missing_heavy_frame, before)

    def test_dtypes_survive_preparation(self, service, binary_frame):
        before = binary_frame.dtypes.copy()
        service.histogram(binary_frame, "age")
        pd.testing.assert_series_equal(binary_frame.dtypes, before)


class TestPreparationCoverage:
    def test_every_chart_type_has_a_preparation_rule(self, service, binary_frame):
        from aidatasetkit.visualization.preparation import _HANDLERS

        assert set(_HANDLERS) == set(ChartType)

    def test_an_unknown_chart_type_is_reported_clearly(self, service, binary_frame):
        from aidatasetkit.visualization.preparation import ChartPreparer

        context = service.context(binary_frame)
        preparer = ChartPreparer()
        broken = spec_for(ChartType.HISTOGRAM, "age")
        object.__setattr__(broken, "chart_type", "not_a_chart")
        with pytest.raises(InvalidVisualizationRequest, match="No preparation rule"):
            preparer.prepare(context, broken)
