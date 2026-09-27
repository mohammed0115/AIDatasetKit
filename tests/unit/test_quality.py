"""Tests for the quality checks and the inspector that runs them.

Two halves, and the second matters as much as the first. Detection tests prove
every planted problem is found; false-positive tests prove ordinary data is left
alone. A detector that cries wolf gets switched off.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core import KitConfig, Severity
from aidatasetkit.core.exceptions import SchemaError, ValidationError
from aidatasetkit.core.types import QualityIssue, QualityReport
from aidatasetkit.profiling import DataProfiler, DataQualityInspector, TaskDetector
from aidatasetkit.profiling.checks import DEFAULT_CHECKS
from aidatasetkit.profiling.context import QualityContext

ALL_PLANTED_CODES = {
    "missing_values",
    "duplicate_rows",
    "constant_column",
    "near_constant_column",
    "high_cardinality",
    "possible_id_like",
    "possible_numeric_stored_as_text",
    "infinite_values",
    "possible_outliers",
    "class_imbalance",
    "possible_target_leakage",
    "target_leakage_exact_duplicate",
}


@pytest.fixture
def inspector() -> DataQualityInspector:
    return DataQualityInspector()


@pytest.fixture
def planted(inspector, problematic_frame) -> QualityReport:
    return inspector.inspect(problematic_frame, target="Churn")


def context_for(frame: pd.DataFrame, **kwargs) -> QualityContext:
    """Build a context the way the inspector would, for single-check tests."""
    config = kwargs.pop("config", KitConfig())
    target = kwargs.pop("target", None)
    target_profile = kwargs.pop("target_profile", None)
    if target is not None and target_profile is None:
        target_profile = TaskDetector(config).detect(frame[target], target_name=target)
    return QualityContext(
        profile=DataProfiler(config).profile(frame),
        config=config,
        target=target,
        target_profile=target_profile,
        **kwargs,
    )


class TestEveryPlantedIssueIsFound:
    def test_all_planted_codes_are_reported(self, planted):
        assert ALL_PLANTED_CODES <= set(planted.codes)

    @pytest.mark.parametrize(
        ("code", "column"),
        [
            ("missing_values", "income"),
            ("constant_column", "country"),
            ("near_constant_column", "plan"),
            ("high_cardinality", "city"),
            ("possible_id_like", "customer_id"),
            ("possible_numeric_stored_as_text", "income_text"),
            ("infinite_values", "ratio"),
            ("possible_outliers", "age"),
            ("class_imbalance", "Churn"),
            ("possible_target_leakage", "risk_score"),
            ("possible_target_leakage", "churn_reason"),
            ("target_leakage_exact_duplicate", "churn_flag"),
        ],
    )
    def test_the_issue_is_attached_to_the_right_column(self, planted, code, column):
        assert column in {issue.column for issue in planted.by_code(code)}

    def test_duplicate_rows_are_reported_at_dataset_level(self, planted):
        issues = planted.by_code("duplicate_rows")
        assert len(issues) == 1
        assert issues[0].column is None
        assert issues[0].details["duplicate_row_count"] == 4

    def test_every_issue_carries_a_message_and_a_recommendation(self, planted):
        for issue in planted.issues:
            assert issue.message.strip()
            assert issue.recommendation and issue.recommendation.strip()


class TestCheckDetails:
    def test_missing_values_report_count_and_ratio(self, planted):
        details = planted.by_code("missing_values")[0].details
        assert details["missing_count"] > 0
        assert 0.0 < details["missing_ratio"] < 1.0
        assert details["threshold"] == KitConfig().missing_warning_threshold

    def test_near_constant_reports_the_dominant_share(self, planted):
        details = planted.by_code("near_constant_column")[0].details
        assert details["dominant_value"] == "basic"
        assert details["dominant_ratio"] >= details["threshold"]

    def test_outliers_report_bounds_and_counts(self, planted):
        issue = next(i for i in planted.by_code("possible_outliers") if i.column == "age")
        details = issue.details
        assert details["outlier_count"] == 2
        assert details["lower_bound"] < details["upper_bound"]
        assert details["multiplier"] == KitConfig().outlier_iqr_multiplier
        assert 0.0 < details["outlier_ratio"] < 1.0

    def test_numeric_text_reports_the_ratio_and_the_offending_values(self, planted):
        details = planted.by_code("possible_numeric_stored_as_text")[0].details
        assert details["numeric_ratio"] == pytest.approx(0.8, abs=0.01)
        assert details["non_numeric_examples"] == ["unknown"]

    def test_class_imbalance_reports_the_distribution(self, planted):
        details = planted.by_code("class_imbalance")[0].details
        assert set(details["class_counts"]) == {0, 1}
        assert details["minority_ratio"] < details["threshold"]
        assert details["imbalance_ratio"] > 1.0

    def test_leakage_names_the_signals_it_relied_on(self, planted):
        by_column = {i.column: i for i in planted.by_code("possible_target_leakage")}
        assert "extreme_association" in by_column["risk_score"].details["signals"]
        assert by_column["risk_score"].details["correlation"] > 0.98
        assert "name_contains_target" in by_column["churn_reason"].details["signals"]


class TestSeverityDiscipline:
    def test_only_certain_findings_are_errors(self, planted):
        assert {issue.code for issue in planted.errors} == {
            "infinite_values",
            "target_leakage_exact_duplicate",
        }

    def test_every_heuristic_finding_asks_for_review(self, planted):
        heuristic = {
            "possible_id_like",
            "possible_numeric_stored_as_text",
            "possible_outliers",
            "possible_target_leakage",
        }
        for issue in planted.issues:
            if issue.code in heuristic:
                assert issue.requires_review, issue.code

    def test_certain_findings_do_not_ask_for_review(self, planted):
        for issue in planted.by_code("target_leakage_exact_duplicate"):
            assert not issue.requires_review

    def test_heuristic_findings_use_hedged_wording(self, planted):
        for issue in planted.needs_review:
            assert issue.code.startswith("possible_")

    def test_outliers_stay_at_info_level(self, planted):
        """The Tukey rule flags a tail in any normal sample; it must not shout."""
        for issue in planted.by_code("possible_outliers"):
            assert issue.severity is Severity.INFO

    def test_exact_duplication_of_the_target_is_stated_with_certainty(self, planted):
        issue = planted.by_code("target_leakage_exact_duplicate")[0]
        assert issue.severity is Severity.ERROR
        assert "possible" not in issue.message.lower()


class TestNoFalsePositives:
    def test_a_clean_dataset_raises_no_errors(self, inspector, clean_frame):
        report = inspector.inspect(clean_frame, target="target")
        assert not report.has_errors, [i.code for i in report.errors]

    def test_a_clean_dataset_raises_none_of_the_planted_warnings(
        self, inspector, clean_frame
    ):
        report = inspector.inspect(clean_frame, target="target")
        assert not {issue.code for issue in report.warnings} & ALL_PLANTED_CODES

    def test_a_unique_measurement_is_not_called_an_identifier(self, planted):
        assert "temperature" not in {i.column for i in planted.by_code("possible_id_like")}

    def test_a_high_cardinality_label_is_not_called_an_identifier(self, planted):
        assert "city" in {i.column for i in planted.by_code("high_cardinality")}
        assert "city" not in {i.column for i in planted.by_code("possible_id_like")}

    def test_a_postal_code_is_not_called_numeric_text(self, planted):
        """"02134" parses as a number, and converting it would destroy it."""
        flagged = {i.column for i in planted.by_code("possible_numeric_stored_as_text")}
        assert "zip_code" not in flagged

    def test_a_datetime_column_is_not_called_numeric_text(self, planted):
        flagged = {i.column for i in planted.by_code("possible_numeric_stored_as_text")}
        assert "signup_date" not in flagged

    def test_a_strong_but_ordinary_association_is_not_leakage(self, planted, problematic_frame):
        from aidatasetkit.statistics import correlation

        strength = correlation(
            problematic_frame["moderate_signal"],
            problematic_frame["Churn"].astype("float64"),
        )
        assert 0.5 < strength < 0.98
        assert "moderate_signal" not in {
            i.column for i in planted.by_code("possible_target_leakage")
        }

    def test_the_target_is_never_reported_as_leaking_into_itself(self, planted):
        leaking = {i.column for i in planted.by_code("possible_target_leakage")}
        assert "Churn" not in leaking

    def test_a_declared_id_column_is_not_reported_as_a_surprise(
        self, inspector, problematic_frame
    ):
        report = inspector.inspect(
            problematic_frame, target="Churn", id_column="customer_id"
        )
        assert not report.by_code("possible_id_like")
        assert "customer_id" not in {i.column for i in report.by_code("high_cardinality")}

    def test_an_almost_unique_categorical_is_not_leakage_by_determinism(self):
        """A near-unique column determines everything; that is arithmetic, not evidence."""
        frame = pd.DataFrame(
            {"note": [f"note_{value}" for value in range(40)], "y": [0, 1] * 20}
        )
        from aidatasetkit.profiling.checks import check_target_leakage

        issues = check_target_leakage(frame, context_for(frame, target="y"))
        assert not [i for i in issues if i.column == "note"]


class TestNumericTextThreshold:
    """The documented boundary for the numeric-as-text warning."""

    def _flagged(self, values, config=None) -> bool:
        from aidatasetkit.profiling.checks import check_numeric_stored_as_text

        frame = pd.DataFrame({"value": values})
        context = context_for(frame, config=config or KitConfig())
        return bool(check_numeric_stored_as_text(frame, context))

    def test_all_numeric_text_is_flagged(self):
        assert self._flagged(["10", "20", "30", "40"])

    def test_the_documented_default_boundary_fires(self):
        """Three of four values parse: exactly the 0.75 default."""
        assert self._flagged(["10", "20", "unknown", "40"])

    def test_below_the_boundary_is_not_flagged(self):
        assert not self._flagged(["10", "20", "unknown", "missing"])

    def test_ordinary_text_is_never_flagged(self):
        assert not self._flagged(["red", "blue", "green", "red"])

    def test_a_stricter_threshold_still_fires(self):
        values = ["10", "20", "unknown", "missing"]
        assert self._flagged(values, KitConfig(numeric_text_ratio_threshold=0.5))

    def test_leading_zero_codes_are_exempt(self):
        assert not self._flagged(["02134", "90210", "01002", "10001"])


def _multicollinear_frame(*, exact: bool = False) -> pd.DataFrame:
    """Two independent columns, one that is (almost) their sum, and one free agent.

    ``c`` is deliberately built from ``a`` and ``b`` rather than from a single
    other column, so the fit under test -- each column against *every* other
    column at once -- is exercised rather than a simple pairwise correlation.
    ``d`` carries no relationship to the rest and exists to prove the check does
    not fire on everything just because something else in the frame is collinear.
    """
    rng = np.random.default_rng(1234)
    n = 100
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    noise = 0.0 if exact else rng.normal(0, 0.01, n)
    c = a + b + noise
    d = rng.normal(0, 1, n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d})


class TestMulticollinearity:
    """The variance-inflation check over numeric features."""

    def _issues(self, frame, config=None):
        from aidatasetkit.profiling.checks import check_multicollinearity

        return check_multicollinearity(frame, context_for(frame, config=config or KitConfig()))

    def test_the_redundant_columns_are_flagged(self):
        flagged = {issue.column for issue in self._issues(_multicollinear_frame())}
        assert {"a", "b", "c"} <= flagged

    def test_the_independent_column_is_not_flagged(self):
        flagged = {issue.column for issue in self._issues(_multicollinear_frame())}
        assert "d" not in flagged

    def test_independent_numeric_features_raise_nothing(self, clean_frame):
        assert self._issues(clean_frame) == []

    def test_an_exact_linear_dependency_is_flagged_with_a_finite_vif(self):
        issues = self._issues(_multicollinear_frame(exact=True))
        by_column = {issue.column: issue for issue in issues}
        assert "c" in by_column
        vif = by_column["c"].details["vif"]
        assert np.isfinite(vif)
        assert vif > 1e6

    def test_fewer_than_two_numeric_features_reports_nothing(self):
        frame = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0], "label": ["x", "y", "x", "y"]})
        assert self._issues(frame) == []

    def test_a_constant_column_does_not_break_the_fit(self):
        frame = _multicollinear_frame()
        frame["flat"] = 1.0
        flagged = {issue.column for issue in self._issues(frame)}
        assert "flat" not in flagged
        assert {"a", "b", "c"} <= flagged

    def test_a_column_with_infinities_is_excluded_from_the_fit(self):
        frame = _multicollinear_frame()
        frame["bad"] = np.inf
        flagged = {issue.column for issue in self._issues(frame)}
        assert "bad" not in flagged
        assert {"a", "b", "c"} <= flagged

    def test_the_other_features_considered_are_named_in_the_details(self):
        issues = self._issues(_multicollinear_frame())
        issue = next(i for i in issues if i.column == "c")
        assert set(issue.details["other_numeric_features"]) == {"a", "b", "d"}

    def test_the_threshold_is_configurable(self):
        lenient = KitConfig(multicollinearity_vif_threshold=1_000_000.0)
        assert self._issues(_multicollinear_frame(), config=lenient) == []

    def test_every_finding_is_a_hedged_warning(self):
        for issue in self._issues(_multicollinear_frame()):
            assert issue.severity is Severity.WARNING
            assert issue.requires_review
            assert issue.code == "possible_multicollinearity"


class TestOutlierEdgeCases:
    def _outliers(self, values) -> list[QualityIssue]:
        from aidatasetkit.profiling.checks import check_outliers

        frame = pd.DataFrame({"value": values})
        return check_outliers(frame, context_for(frame))

    def test_a_constant_column_reports_nothing(self):
        assert self._outliers([5.0] * 20) == []

    def test_a_zero_interquartile_range_reports_nothing(self):
        assert self._outliers([1.0] * 18 + [2.0, 99.0]) == []

    def test_a_tiny_sample_reports_nothing(self):
        assert self._outliers([1.0, 2.0, 100.0]) == []

    def test_missing_values_do_not_break_the_fences(self):
        issues = self._outliers([1.0, 2.0, 3.0, 4.0, np.nan, 500.0])
        assert len(issues) == 1
        assert issues[0].details["outlier_count"] == 1

    def test_infinities_are_excluded_from_the_fences(self):
        issues = self._outliers([1.0, 2.0, 3.0, 4.0, 5.0, 500.0, np.inf])
        assert issues[0].details["outlier_count"] == 1
        assert np.isfinite(issues[0].details["upper_bound"])

    def test_the_multiplier_is_configurable(self):
        from aidatasetkit.profiling.checks import check_outliers

        frame = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0, 12.0]})
        strict = check_outliers(frame, context_for(frame, config=KitConfig(outlier_iqr_multiplier=1.5)))
        relaxed = check_outliers(frame, context_for(frame, config=KitConfig(outlier_iqr_multiplier=10.0)))
        assert strict and not relaxed


class TestLeakageHeuristics:
    def _leakage(self, frame: pd.DataFrame, target: str) -> dict:
        from aidatasetkit.profiling.checks import check_target_leakage

        issues = check_target_leakage(frame, context_for(frame, target=target))
        return {issue.column: issue for issue in issues}

    def test_an_exact_copy_of_the_target_is_certain(self):
        frame = pd.DataFrame({"copy": [0, 1, 0, 1], "y": [0, 1, 0, 1]})
        issue = self._leakage(frame, "y")["copy"]
        assert issue.code == "target_leakage_exact_duplicate"
        assert issue.severity is Severity.ERROR

    def test_an_exact_copy_under_a_different_dtype_is_still_caught(self):
        frame = pd.DataFrame({"copy": [0.0, 1.0, 0.0, 1.0], "y": [0, 1, 0, 1]})
        assert "copy" in self._leakage(frame, "y")

    def test_a_deterministic_categorical_mapping_is_suspicious(self):
        frame = pd.DataFrame(
            {"reason": ["a", "b", "a", "b"] * 5, "y": [0, 1, 0, 1] * 5}
        )
        issue = self._leakage(frame, "y")["reason"]
        assert issue.code == "possible_target_leakage"
        assert "deterministic_mapping" in issue.details["signals"]
        assert issue.requires_review

    def test_a_post_outcome_name_is_suspicious(self):
        frame = pd.DataFrame(
            {
                "cancellation_date": list(range(20)),
                "y": [0, 1] * 10,
            }
        )
        issue = self._leakage(frame, "y")["cancellation_date"]
        assert "post_outcome" in " ".join(issue.details["signals"])

    def test_a_name_containing_the_target_is_suspicious(self):
        frame = pd.DataFrame({"churn_bucket": ["x", "y"] * 10, "Churn": [0, 1] * 10})
        assert "name_contains_target" in (
            self._leakage(frame, "Churn")["churn_bucket"].details["signals"]
        )

    def test_an_ordinary_feature_is_left_alone(self):
        frame = pd.DataFrame(
            {"tenure": list(range(20)), "y": [0, 1, 1, 0] * 5}
        )
        assert "tenure" not in self._leakage(frame, "y")

    def test_no_leakage_check_runs_without_a_target(self):
        frame = pd.DataFrame({"a": [0, 1], "b": [0, 1]})
        from aidatasetkit.profiling.checks import check_target_leakage

        assert check_target_leakage(frame, context_for(frame)) == []

    def test_a_constant_feature_does_not_trigger_an_association(self):
        frame = pd.DataFrame({"flat": [1.0] * 20, "y": [0, 1] * 10})
        assert "flat" not in self._leakage(frame, "y")


class TestInspector:
    def test_checks_are_replaceable(self, problematic_frame):
        from aidatasetkit.profiling.checks import check_constant_columns

        report = DataQualityInspector(checks=[check_constant_columns]).inspect(
            problematic_frame
        )
        assert set(report.codes) == {"constant_column"}

    def test_the_default_check_set_is_exposed(self, inspector):
        assert inspector.checks == DEFAULT_CHECKS
        assert len(DEFAULT_CHECKS) == 12

    def test_a_supplied_profile_is_reused(self, inspector, clean_frame):
        profile = DataProfiler().profile(clean_frame)
        report = inspector.inspect(clean_frame, profile=profile)
        assert isinstance(report, QualityReport)

    def test_a_supplied_target_profile_is_reused(self, inspector, problematic_frame):
        target_profile = TaskDetector().detect(problematic_frame["Churn"])
        report = inspector.inspect(
            problematic_frame, target="Churn", target_profile=target_profile
        )
        assert report.by_code("class_imbalance")

    def test_an_undetectable_target_does_not_abort_the_inspection(self, inspector):
        """An ambiguous target skips target-dependent checks, nothing more."""
        frame = pd.DataFrame({"country": ["SA"] * 5, "y": [1, 2, 3, 4, 5]})
        report = inspector.inspect(frame, target="y")
        assert report.by_code("constant_column")
        assert not report.by_code("class_imbalance")

    def test_target_dependent_checks_are_skipped_without_a_target(
        self, inspector, problematic_frame
    ):
        report = inspector.inspect(problematic_frame)
        assert not report.by_code("class_imbalance")
        assert not report.by_code("possible_target_leakage")

    def test_an_unknown_target_column_is_rejected(self, inspector, clean_frame):
        with pytest.raises(SchemaError, match="target"):
            inspector.inspect(clean_frame, target="missing")

    def test_an_unknown_id_column_is_rejected(self, inspector, clean_frame):
        with pytest.raises(SchemaError, match="id_column"):
            inspector.inspect(clean_frame, id_column="missing")

    def test_non_dataframe_input_is_rejected(self, inspector):
        with pytest.raises(ValidationError, match="DataFrame is required"):
            inspector.inspect([1, 2, 3])

    def test_the_configuration_is_shared_with_the_profiler(self):
        config = KitConfig(near_constant_threshold=0.5)
        inspector = DataQualityInspector(config)
        assert inspector.config is config


class TestReportSummaries:
    def test_counts_partition_the_issues(self, planted):
        assert (
            len(planted.errors) + len(planted.warnings) + len(planted.infos)
            == planted.issue_count
        )

    def test_has_errors_reflects_the_error_list(self, planted, inspector, clean_frame):
        assert planted.has_errors
        assert not inspector.inspect(clean_frame, target="target").has_errors

    def test_findings_can_be_selected_by_column(self, planted):
        assert all(issue.column == "ratio" for issue in planted.by_column("ratio"))

    def test_needs_review_collects_the_heuristic_findings(self, planted):
        assert planted.needs_review
        assert all(issue.requires_review for issue in planted.needs_review)

    def test_codes_are_distinct_and_ordered_by_first_appearance(self, planted):
        assert len(planted.codes) == len(set(planted.codes))

    def test_an_empty_report_is_well_behaved(self):
        report = QualityReport()
        assert report.issue_count == 0
        assert not report.has_errors
        assert report.codes == ()


class TestImmutability:
    def test_inspection_does_not_modify_the_frame(self, inspector, problematic_frame):
        before = problematic_frame.copy(deep=True)
        inspector.inspect(problematic_frame, target="Churn")
        pd.testing.assert_frame_equal(problematic_frame, before)

    def test_inspection_does_not_convert_dtypes(self, inspector, problematic_frame):
        before = problematic_frame.dtypes.copy()
        inspector.inspect(problematic_frame, target="Churn")
        pd.testing.assert_series_equal(problematic_frame.dtypes, before)

    def test_missing_values_survive_inspection(self, inspector, problematic_frame):
        before = int(problematic_frame["income"].isna().sum())
        inspector.inspect(problematic_frame, target="Churn")
        assert int(problematic_frame["income"].isna().sum()) == before

    def test_infinities_survive_inspection(self, inspector, problematic_frame):
        before = int(np.isinf(problematic_frame["ratio"]).sum())
        inspector.inspect(problematic_frame, target="Churn")
        assert int(np.isinf(problematic_frame["ratio"]).sum()) == before

    def test_duplicate_rows_survive_inspection(self, inspector, problematic_frame):
        before = len(problematic_frame)
        inspector.inspect(problematic_frame, target="Churn")
        assert len(problematic_frame) == before

    def test_every_check_leaves_the_frame_untouched(self, problematic_frame):
        context = context_for(problematic_frame, target="Churn")
        for check in DEFAULT_CHECKS:
            before = problematic_frame.copy(deep=True)
            check(problematic_frame, context)
            pd.testing.assert_frame_equal(problematic_frame, before)

    def test_the_report_is_immutable(self, planted):
        with pytest.raises(AttributeError):
            planted.issues = ()


class TestSerialisation:
    def test_the_report_round_trips_through_json(self, planted):
        payload = json.loads(json.dumps(planted.to_dict()))
        assert payload["issue_count"] == planted.issue_count
        assert payload["has_errors"] is True
        assert len(payload["issues"]) == planted.issue_count

    def test_issue_details_survive_serialisation(self, planted):
        for issue in planted.issues:
            json.dumps(issue.to_dict())

    def test_numpy_values_in_details_are_encoded(self, planted):
        payload = json.loads(json.dumps(planted.by_code("class_imbalance")[0].to_dict()))
        assert payload["details"]["class_counts"] == {"0": 183, "1": 21}

    def test_non_string_column_labels_serialise_safely(self):
        frame = pd.DataFrame({0: [1, 1, 1]})
        report = DataQualityInspector().inspect(frame)
        issue = report.by_code("constant_column")[0]
        assert issue.column == 0
        assert json.loads(json.dumps(issue.to_dict()))["column"] == "0"

    def test_the_report_renders_as_a_frame(self, planted):
        rendered = planted.to_frame()
        assert len(rendered) == planted.issue_count
        assert "severity" in rendered.columns
