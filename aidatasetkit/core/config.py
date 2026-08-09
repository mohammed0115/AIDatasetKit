"""Central configuration.

Every threshold the library uses lives here. No component defines its own magic
number, so an analyst can see and change all tunable behaviour in one place, and
the effective settings can be serialised into a provenance record.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from aidatasetkit.core.exceptions import ConfigurationError

__all__ = ["KitConfig"]


@dataclass(frozen=True, slots=True)
class KitConfig:
    """Thresholds and defaults shared by every component.

    Attributes:
        random_state: Seed used for every splitter and every stochastic model.
        validation_size: Fraction of the training data held out for evaluation.
        cv_folds: Number of folds used by cross-validation and model comparison.
        shuffle: Whether splitters shuffle before splitting.
        missing_warning_threshold: Missing-value fraction above which a column is
            reported as a quality issue.
        near_constant_threshold: Dominant-value fraction above which a column is
            reported as near-constant.
        high_cardinality_threshold: Distinct-value count above which a
            categorical column is reported as high-cardinality.
        id_uniqueness_threshold: Uniqueness ratio above which a column is
            reported as identifier-like.
        outlier_iqr_multiplier: Multiplier applied to the IQR when flagging
            outliers with the Tukey rule.
        outlier_zscore_threshold: Absolute z-score above which a value is flagged
            when the z-score method is selected.
        leakage_correlation_threshold: Absolute correlation with the target above
            which a feature is flagged for leakage review. Heuristic only.
        imbalance_threshold: Minority-class fraction below which accuracy is
            reported as potentially misleading.
        dense_encoding_warning_categories: Total one-hot category count above
            which a dense-only model is reported as a memory risk.
        task_detection_max_classes: Upper bound on distinct target values for an
            integer target to be read as classification.
        task_detection_unique_ratio: Upper bound on the distinct-to-row ratio for
            an integer target to be read as classification.
        probability_column: Output column name for binary-classification
            probabilities.
        predicted_class_column: Output column name for multiclass predictions.
        prediction_column: Output column name for regression predictions.
    """

    random_state: int = 42
    validation_size: float = 0.2
    cv_folds: int = 5
    shuffle: bool = True

    missing_warning_threshold: float = 0.2
    near_constant_threshold: float = 0.98
    high_cardinality_threshold: int = 50
    id_uniqueness_threshold: float = 0.95
    outlier_iqr_multiplier: float = 1.5
    outlier_zscore_threshold: float = 3.0
    leakage_correlation_threshold: float = 0.98
    imbalance_threshold: float = 0.2
    dense_encoding_warning_categories: int = 1000

    task_detection_max_classes: int = 20
    task_detection_unique_ratio: float = 0.05

    probability_column: str = "predicted_probability"
    predicted_class_column: str = "predicted_class"
    prediction_column: str = "prediction"

    def __post_init__(self) -> None:
        self._require_open_unit_interval("validation_size", self.validation_size)
        self._require_closed_unit_interval("missing_warning_threshold", self.missing_warning_threshold)
        self._require_closed_unit_interval("near_constant_threshold", self.near_constant_threshold)
        self._require_closed_unit_interval("id_uniqueness_threshold", self.id_uniqueness_threshold)
        self._require_closed_unit_interval("leakage_correlation_threshold", self.leakage_correlation_threshold)
        self._require_open_unit_interval("imbalance_threshold", self.imbalance_threshold)
        self._require_closed_unit_interval("task_detection_unique_ratio", self.task_detection_unique_ratio)

        if self.cv_folds < 2:
            raise ConfigurationError(f"cv_folds must be at least 2, got {self.cv_folds}.")
        if self.high_cardinality_threshold < 1:
            raise ConfigurationError(
                f"high_cardinality_threshold must be at least 1, got {self.high_cardinality_threshold}."
            )
        if self.dense_encoding_warning_categories < 1:
            raise ConfigurationError(
                "dense_encoding_warning_categories must be at least 1, "
                f"got {self.dense_encoding_warning_categories}."
            )
        if self.task_detection_max_classes < 2:
            raise ConfigurationError(
                f"task_detection_max_classes must be at least 2, got {self.task_detection_max_classes}."
            )
        if self.outlier_iqr_multiplier <= 0:
            raise ConfigurationError(
                f"outlier_iqr_multiplier must be positive, got {self.outlier_iqr_multiplier}."
            )
        if self.outlier_zscore_threshold <= 0:
            raise ConfigurationError(
                f"outlier_zscore_threshold must be positive, got {self.outlier_zscore_threshold}."
            )

        for name in ("probability_column", "predicted_class_column", "prediction_column"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"{name} must be a non-empty string, got {value!r}.")

    @staticmethod
    def _require_open_unit_interval(name: str, value: float) -> None:
        if not 0.0 < float(value) < 1.0:
            raise ConfigurationError(f"{name} must lie strictly between 0 and 1, got {value!r}.")

    @staticmethod
    def _require_closed_unit_interval(name: str, value: float) -> None:
        if not 0.0 <= float(value) <= 1.0:
            raise ConfigurationError(f"{name} must lie between 0 and 1 inclusive, got {value!r}.")

    def replace(self, **changes: Any) -> KitConfig:
        """Return a new config with ``changes`` applied and re-validated."""
        unknown = set(changes) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise ConfigurationError(f"Unknown configuration options: {sorted(unknown)}.")
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of every configured value."""
        return dataclasses.asdict(self)
