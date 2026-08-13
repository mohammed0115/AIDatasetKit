"""The metric policy: what is measured, and what absence means.

The oracle tests are the point of this file. Every metric is checked against
scikit-learn computed independently on the same arrays, because the ways a metric
goes quietly wrong -- the wrong positive class, the wrong averaging, a sign
inversion, RMSE reported as MSE, the wrong probability column -- all produce a
number that looks entirely reasonable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn import metrics as skmetrics

from aidatasetkit.core.exceptions import ValidationError
from aidatasetkit.core.types import TaskType
from aidatasetkit.evaluation import (
    CLASSIFICATION_METRICS,
    DEFAULT_RANKING_METRIC,
    REGRESSION_METRICS,
    EvaluationReport,
    MetricDirection,
    MetricStatus,
    MetricValue,
    evaluate_classification,
    evaluate_regression,
    metric_direction,
)
from aidatasetkit.preprocessing import TargetLabelEncoder


class TestMetricValue:
    """A number, or a reason. Never both, and never neither."""

    def test_an_available_metric_carries_a_number(self):
        metric = MetricValue("f1", 0.8, MetricDirection.HIGHER_IS_BETTER)
        assert metric.is_available and metric.value == 0.8

    def test_an_available_metric_without_a_value_is_refused(self):
        with pytest.raises(ValidationError, match="carries no value"):
            MetricValue("f1", None, MetricDirection.HIGHER_IS_BETTER)

    def test_an_available_metric_cannot_be_non_finite(self):
        """A nan that reached a comparison table would be ranked."""
        with pytest.raises(ValidationError, match="non-finite"):
            MetricValue("r2", float("nan"), MetricDirection.HIGHER_IS_BETTER)

    def test_an_unavailable_metric_must_say_why(self):
        with pytest.raises(ValidationError, match="without saying why"):
            MetricValue(
                "roc_auc", None, MetricDirection.HIGHER_IS_BETTER,
                status=MetricStatus.UNDEFINED,
            )

    def test_an_unavailable_metric_cannot_also_carry_a_value(self):
        with pytest.raises(ValidationError, match="still"):
            MetricValue(
                "roc_auc", 0.5, MetricDirection.HIGHER_IS_BETTER,
                status=MetricStatus.UNDEFINED, reason="x",
            )

    def test_a_value_and_a_reason_together_are_refused(self):
        with pytest.raises(ValidationError, match="both"):
            MetricValue(
                "f1", 0.8, MetricDirection.HIGHER_IS_BETTER, reason="unnecessary"
            )

    @pytest.mark.parametrize(
        "direction,left,right,expected",
        [
            (MetricDirection.HIGHER_IS_BETTER, 0.9, 0.7, True),
            (MetricDirection.HIGHER_IS_BETTER, 0.7, 0.9, False),
            (MetricDirection.LOWER_IS_BETTER, 4.0, 9.0, True),
            (MetricDirection.LOWER_IS_BETTER, 9.0, 4.0, False),
        ],
    )
    def test_better_follows_the_direction(self, direction, left, right, expected):
        a = MetricValue("m", left, direction)
        b = MetricValue("m", right, direction)
        assert a.is_better_than(b) is expected

    def test_comparing_two_different_metrics_is_refused(self):
        a = MetricValue("f1", 0.9, MetricDirection.HIGHER_IS_BETTER)
        b = MetricValue("accuracy", 0.9, MetricDirection.HIGHER_IS_BETTER)
        with pytest.raises(ValidationError, match="different metrics"):
            a.is_better_than(b)

    def test_an_unavailable_metric_is_neither_better_nor_worse(self):
        """The rule that stops a failed model from winning a ranking."""
        present = MetricValue("f1", 0.9, MetricDirection.HIGHER_IS_BETTER)
        absent = MetricValue(
            "f1", None, MetricDirection.HIGHER_IS_BETTER,
            status=MetricStatus.UNSUPPORTED_BY_MODEL, reason="no probabilities",
        )
        with pytest.raises(ValidationError, match="not available on both sides"):
            present.is_better_than(absent)

    def test_it_survives_json(self):
        import json

        metric = MetricValue("f1", 0.8, MetricDirection.HIGHER_IS_BETTER)
        assert json.loads(json.dumps(metric.to_dict()))["value"] == 0.8


class TestEvaluationReport:
    def test_a_duplicated_metric_is_refused(self):
        one = MetricValue("f1", 0.8, MetricDirection.HIGHER_IS_BETTER)
        with pytest.raises(ValidationError, match="more than once"):
            EvaluationReport(TaskType.CLASSIFICATION, (one, one), row_count=10)

    def test_a_metric_never_attempted_raises_rather_than_returning_none(self):
        """Different from a metric attempted and unavailable, which is present."""
        report = EvaluationReport(
            TaskType.REGRESSION,
            (MetricValue("mae", 1.0, MetricDirection.LOWER_IS_BETTER),),
            row_count=5,
        )
        assert "mae" in report
        with pytest.raises(KeyError, match="not attempted"):
            report["roc_auc"]


class TestDirections:
    """The direction table is what a ranking reads, so it has to be right."""

    @pytest.mark.parametrize("name", ["accuracy", "precision", "recall", "f1", "roc_auc", "r2"])
    def test_these_are_better_when_larger(self, name):
        assert metric_direction(name) is MetricDirection.HIGHER_IS_BETTER

    @pytest.mark.parametrize("name", ["mae", "mse", "rmse"])
    def test_these_are_better_when_smaller(self, name):
        assert metric_direction(name) is MetricDirection.LOWER_IS_BETTER

    def test_every_published_metric_has_a_direction(self):
        for name in CLASSIFICATION_METRICS + REGRESSION_METRICS:
            assert isinstance(metric_direction(name), MetricDirection)

    def test_the_default_ranking_metrics_are_stated_for_both_families(self):
        assert DEFAULT_RANKING_METRIC[TaskType.CLASSIFICATION] == "f1"
        assert DEFAULT_RANKING_METRIC[TaskType.REGRESSION] == "rmse"


@pytest.fixture
def churn_encoder():
    """"churn" sorts first, so the interesting class is column 0, not 1."""
    import pandas as pd

    encoder = TargetLabelEncoder(positive_label="churn")
    encoder.fit(pd.Series(["churn", "stay"] * 40))
    return encoder


class TestBinaryClassificationOracle:
    """Every number checked against sklearn computed independently."""

    @pytest.fixture
    def scored(self, churn_encoder):
        rng = np.random.default_rng(11)
        n = 200
        y_true = rng.integers(0, 2, n)
        y_pred = np.where(rng.random(n) < 0.8, y_true, 1 - y_true)
        proba = np.zeros((n, 2))
        confidence = rng.uniform(0.55, 0.99, n)
        proba[np.arange(n), y_pred] = confidence
        proba[np.arange(n), 1 - y_pred] = 1 - confidence
        return y_true, y_pred, proba

    def test_accuracy_matches_sklearn(self, scored, churn_encoder):
        y_true, y_pred, proba = scored
        report = evaluate_classification(
            y_true, y_pred, probabilities=proba, encoding=churn_encoder, class_count=2
        )
        assert report["accuracy"].value == pytest.approx(
            skmetrics.accuracy_score(y_true, y_pred)
        )

    @pytest.mark.parametrize(
        "name,function",
        [
            ("precision", skmetrics.precision_score),
            ("recall", skmetrics.recall_score),
            ("f1", skmetrics.f1_score),
        ],
    )
    def test_class_metrics_use_the_resolved_positive_class(
        self, scored, churn_encoder, name, function
    ):
        """The oracle that catches the pos_label=1 assumption.

        "churn" encodes to 0, so the correct answer is sklearn's with
        ``pos_label=0``. The naive default would give the score for "stay".
        """
        y_true, y_pred, proba = scored
        report = evaluate_classification(
            y_true, y_pred, probabilities=proba, encoding=churn_encoder, class_count=2
        )
        assert churn_encoder.positive_column_index() == 0
        expected = function(y_true, y_pred, pos_label=0, zero_division=0.0)
        assert report[name].value == pytest.approx(expected)

    @pytest.mark.parametrize("name", ["precision", "recall", "f1"])
    def test_and_that_is_not_the_same_as_the_naive_answer(
        self, scored, churn_encoder, name
    ):
        """If it were, the previous test would prove nothing."""
        y_true, y_pred, _ = scored
        function = {
            "precision": skmetrics.precision_score,
            "recall": skmetrics.recall_score,
            "f1": skmetrics.f1_score,
        }[name]
        for_class_zero = function(y_true, y_pred, pos_label=0, zero_division=0.0)
        for_class_one = function(y_true, y_pred, pos_label=1, zero_division=0.0)
        assert for_class_zero != pytest.approx(for_class_one)

    def test_roc_auc_uses_the_positive_column(self, scored, churn_encoder):
        y_true, y_pred, proba = scored
        report = evaluate_classification(
            y_true, y_pred, probabilities=proba, encoding=churn_encoder, class_count=2
        )
        expected = skmetrics.roc_auc_score((y_true == 0).astype(int), proba[:, 0])
        assert report["roc_auc"].value == pytest.approx(expected)

    def test_and_the_other_column_would_have_given_one_minus_that(self, scored):
        """Why the column choice is not cosmetic."""
        y_true, _, proba = scored
        first = skmetrics.roc_auc_score((y_true == 0).astype(int), proba[:, 0])
        second = skmetrics.roc_auc_score((y_true == 1).astype(int), proba[:, 1])
        assert first == pytest.approx(second)  # symmetric here...
        wrong = skmetrics.roc_auc_score((y_true == 0).astype(int), proba[:, 1])
        assert wrong == pytest.approx(1 - first)  # ...but not if the column slips

    def test_the_detail_records_which_class_was_measured(self, scored, churn_encoder):
        y_true, y_pred, proba = scored
        report = evaluate_classification(
            y_true, y_pred, probabilities=proba, encoding=churn_encoder, class_count=2
        )
        assert report["f1"].detail["positive_label"] == "churn"
        assert report["roc_auc"].detail["positive_column"] == 0


class TestWhenAMetricIsUnavailable:
    """Four different absences, four different statuses."""

    @pytest.fixture
    def basic(self):
        y_true = np.array([0, 1] * 30)
        y_pred = np.array([0, 1] * 30)
        return y_true, y_pred

    def test_no_probabilities_means_unsupported_by_model(self, basic, churn_encoder):
        y_true, y_pred = basic
        report = evaluate_classification(
            y_true, y_pred, probabilities=None, encoding=churn_encoder, class_count=2
        )
        entry = report["roc_auc"]
        assert entry.status is MetricStatus.UNSUPPORTED_BY_MODEL
        assert "supports_predict_proba=False" in entry.reason

    def test_multiclass_means_not_applicable(self):
        y_true = np.array([0, 1, 2] * 20)
        report = evaluate_classification(
            y_true, y_true, probabilities=np.eye(3)[y_true], class_count=3
        )
        entry = report["roc_auc"]
        assert entry.status is MetricStatus.NOT_APPLICABLE
        assert "one-vs-rest" in entry.reason

    def test_an_unresolved_positive_label_means_undefined(self, basic):
        import pandas as pd

        y_true, y_pred = basic
        unresolved = TargetLabelEncoder().fit(pd.Series(["churn", "stay"] * 30))
        report = evaluate_classification(
            y_true, y_pred, probabilities=np.eye(2)[y_pred],
            encoding=unresolved, class_count=2,
        )
        entry = report["roc_auc"]
        assert entry.status is MetricStatus.UNDEFINED
        assert "positive_label" in entry.reason

    def test_one_class_in_the_evaluation_rows_means_undefined(self, churn_encoder):
        y_true = np.zeros(20, dtype=int)
        report = evaluate_classification(
            y_true, y_true, probabilities=np.tile([0.9, 0.1], (20, 1)),
            encoding=churn_encoder, class_count=2,
        )
        entry = report["roc_auc"]
        assert entry.status is MetricStatus.UNDEFINED
        assert "single class" in entry.reason

    def test_every_metric_is_present_even_when_unavailable(self, basic, churn_encoder):
        """No silent dropping: the report always lists the whole policy."""
        y_true, y_pred = basic
        report = evaluate_classification(
            y_true, y_pred, probabilities=None, encoding=churn_encoder, class_count=2
        )
        assert tuple(m.name for m in report) == CLASSIFICATION_METRICS


class TestMulticlassPolicy:
    def test_averaging_is_macro_and_says_so(self):
        y_true = np.array([0, 1, 2] * 30)
        y_pred = np.array([0, 1, 2] * 30)
        report = evaluate_classification(y_true, y_pred, class_count=3)
        for name in ("precision", "recall", "f1"):
            assert report[name].detail["averaging"] == "macro"

    def test_macro_matches_sklearn_macro(self):
        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 3, 150)
        y_pred = np.where(rng.random(150) < 0.7, y_true, (y_true + 1) % 3)
        report = evaluate_classification(y_true, y_pred, class_count=3)
        for name, function in (
            ("precision", skmetrics.precision_score),
            ("recall", skmetrics.recall_score),
            ("f1", skmetrics.f1_score),
        ):
            expected = function(y_true, y_pred, average="macro", zero_division=0.0)
            assert report[name].value == pytest.approx(expected)

    def test_binary_defaults_are_not_applied_to_multiclass(self):
        """Macro over three classes is not the binary score for class 1."""
        rng = np.random.default_rng(4)
        y_true = rng.integers(0, 3, 150)
        y_pred = np.where(rng.random(150) < 0.7, y_true, (y_true + 1) % 3)
        report = evaluate_classification(y_true, y_pred, class_count=3)
        naive = skmetrics.f1_score(
            y_true, y_pred, labels=[1], average="macro", zero_division=0.0
        )
        assert report["f1"].value != pytest.approx(naive)

    def test_a_class_the_model_never_predicts_scores_zero_without_warning(self):
        """zero_division is a decision here, not a warning to be silenced."""
        import warnings

        y_true = np.array([0, 1, 2] * 20)
        y_pred = np.array([0, 1, 1] * 20)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            report = evaluate_classification(y_true, y_pred, class_count=3)
        assert report["precision"].is_available


class TestRegressionOracle:
    @pytest.fixture
    def scored(self):
        rng = np.random.default_rng(21)
        y_true = rng.normal(100, 25, 200)
        y_pred = y_true + rng.normal(0, 4, 200)
        return y_true, y_pred

    def test_mae_matches_sklearn(self, scored):
        y_true, y_pred = scored
        report = evaluate_regression(y_true, y_pred)
        assert report["mae"].value == pytest.approx(
            skmetrics.mean_absolute_error(y_true, y_pred)
        )

    def test_mse_matches_sklearn(self, scored):
        y_true, y_pred = scored
        report = evaluate_regression(y_true, y_pred)
        assert report["mse"].value == pytest.approx(
            skmetrics.mean_squared_error(y_true, y_pred)
        )

    def test_rmse_is_the_root_of_mse_and_not_mse(self, scored):
        """The confusion that would look entirely plausible in a table."""
        y_true, y_pred = scored
        report = evaluate_regression(y_true, y_pred)
        mse = skmetrics.mean_squared_error(y_true, y_pred)
        assert report["rmse"].value == pytest.approx(math.sqrt(mse))
        assert report["rmse"].value != pytest.approx(mse)
        assert report["rmse"].value == pytest.approx(
            skmetrics.root_mean_squared_error(y_true, y_pred)
        )

    def test_r2_matches_sklearn(self, scored):
        y_true, y_pred = scored
        report = evaluate_regression(y_true, y_pred)
        assert report["r2"].value == pytest.approx(skmetrics.r2_score(y_true, y_pred))

    def test_the_values_are_reported_unnegated(self, scored):
        """MAE = 4.2, never neg_mean_absolute_error = -4.2."""
        y_true, y_pred = scored
        report = evaluate_regression(y_true, y_pred)
        for name in ("mae", "mse", "rmse"):
            assert report[name].value > 0

    def test_every_regression_metric_is_attempted(self, scored):
        y_true, y_pred = scored
        report = evaluate_regression(y_true, y_pred)
        assert tuple(m.name for m in report) == REGRESSION_METRICS


class TestR2EdgeCases:
    """R² needs variance to explain, and sometimes there is none."""

    def test_a_constant_target_makes_it_undefined(self):
        y_true = np.full(30, 7.5)
        report = evaluate_regression(y_true, np.full(30, 7.5))
        entry = report["r2"]
        assert entry.status is MetricStatus.UNDEFINED
        assert "no variance" in entry.reason

    def test_sklearn_would_have_answered_with_a_plausible_number_there(self):
        """Exactly why the policy is explicit rather than inherited.

        On a constant target scikit-learn returns ``1.0`` for a perfect constant
        prediction and ``0.0`` for a wrong one -- the *same* ``0.0`` whether the
        prediction is off by 1.5 or by 892.5. Those are not scores that describe
        anything, and all three would be ranked without complaint.
        """
        y_true = np.full(30, 7.5)
        assert skmetrics.r2_score(y_true, np.full(30, 7.5)) == 1.0
        assert skmetrics.r2_score(y_true, np.full(30, 9.0)) == 0.0
        assert skmetrics.r2_score(y_true, np.full(30, 900.0)) == 0.0

    @pytest.mark.parametrize("prediction", [7.5, 9.0, 900.0])
    def test_and_this_policy_calls_all_of_them_undefined(self, prediction):
        report = evaluate_regression(np.full(30, 7.5), np.full(30, prediction))
        assert report["r2"].status is MetricStatus.UNDEFINED

    def test_a_single_evaluation_row_makes_it_undefined(self):
        report = evaluate_regression(np.array([5.0]), np.array([5.2]))
        assert report["r2"].status is MetricStatus.UNDEFINED
        assert "at least two" in report["r2"].reason

    def test_but_the_other_three_are_still_computed_on_one_row(self):
        report = evaluate_regression(np.array([5.0]), np.array([5.2]))
        assert report["mae"].value == pytest.approx(0.2)
        assert report["rmse"].value == pytest.approx(0.2)

    def test_a_badly_negative_r2_is_still_a_real_score(self):
        """Worse than the mean is a legitimate result, not an error."""
        rng = np.random.default_rng(5)
        y_true = rng.normal(0, 1, 50)
        report = evaluate_regression(y_true, -y_true * 5)
        assert report["r2"].is_available and report["r2"].value < 0
