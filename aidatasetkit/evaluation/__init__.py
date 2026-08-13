"""Evaluation: what to measure, and what it means when there is nothing to report.

This layer computes metrics and nothing else. It does not split, does not fit,
does not rank, and does not know which models exist. Given true values and
predicted ones it returns an :class:`EvaluationReport` -- every metric the policy
attempted, each either carrying a number or carrying the reason it does not.

The reason matters as much as the number. ROC-AUC missing because a decision
tree publishes no probabilities is a fact about the model; missing because the
evaluation rows hold one class is a fact about the split; missing because no
positive label was resolved is a fact about the target contract. A single
``NaN`` says none of those, and an omitted row invites the reader to assume.
"""

from aidatasetkit.evaluation.metrics import (
    CLASSIFICATION_METRICS,
    CLUSTERING_METRICS,
    DEFAULT_RANKING_METRIC,
    MULTICLASS_AVERAGING,
    NOISE_LABEL,
    REGRESSION_METRICS,
    SILHOUETTE_ROW_LIMIT,
    evaluate_classification,
    evaluate_clustering,
    evaluate_regression,
    metric_direction,
)
from aidatasetkit.evaluation.types import (
    EvaluationReport,
    MetricDirection,
    MetricStatus,
    MetricValue,
)

__all__ = [
    "CLASSIFICATION_METRICS",
    "CLUSTERING_METRICS",
    "DEFAULT_RANKING_METRIC",
    "EvaluationReport",
    "MULTICLASS_AVERAGING",
    "MetricDirection",
    "MetricStatus",
    "MetricValue",
    "NOISE_LABEL",
    "REGRESSION_METRICS",
    "SILHOUETTE_ROW_LIMIT",
    "evaluate_classification",
    "evaluate_clustering",
    "evaluate_regression",
    "metric_direction",
]
