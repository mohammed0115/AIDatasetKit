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
        multicollinearity_vif_threshold: Variance inflation factor above which a
            numeric feature is flagged as largely redundant with the other
            numeric features. 10 is the conventional cutoff in regression
            diagnostics; a VIF below it is never flagged regardless of how the
            rest of the threshold is read.
        imbalance_threshold: Minority-class fraction below which accuracy is
            reported as potentially misleading.
        dense_encoding_warning_categories: Total one-hot category count above
            which a dense-only model is reported as a memory risk.
        numeric_text_ratio_threshold: Fraction of a text column's values that must
            parse as numbers before the column is reported as numeric data stored
            as text.
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
    multicollinearity_vif_threshold: float = 10.0
    imbalance_threshold: float = 0.2
    dense_encoding_warning_categories: int = 1000
    numeric_text_ratio_threshold: float = 0.75

    task_detection_max_classes: int = 20
    task_detection_unique_ratio: float = 0.05

    probability_column: str = "predicted_probability"
    predicted_class_column: str = "predicted_class"
    prediction_column: str = "prediction"

    def __post_init__(self) -> None:
        self._require_seed("random_state", self.random_state)
        self._require_open_unit_interval("validation_size", self.validation_size)
        self._require_closed_unit_interval("missing_warning_threshold", self.missing_warning_threshold)
        self._require_closed_unit_interval("near_constant_threshold", self.near_constant_threshold)
        self._require_closed_unit_interval("id_uniqueness_threshold", self.id_uniqueness_threshold)
        self._require_closed_unit_interval("leakage_correlation_threshold", self.leakage_correlation_threshold)
        self._require_open_unit_interval("imbalance_threshold", self.imbalance_threshold)
        self._require_closed_unit_interval("task_detection_unique_ratio", self.task_detection_unique_ratio)
        self._require_closed_unit_interval(
            "numeric_text_ratio_threshold", self.numeric_text_ratio_threshold
        )

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
        import math
        import numbers

        vif_threshold = self.multicollinearity_vif_threshold
        if isinstance(vif_threshold, bool) or not isinstance(vif_threshold, numbers.Real):
            raise ConfigurationError(
                "multicollinearity_vif_threshold must be a number, got "
                f"{type(vif_threshold).__name__} ({vif_threshold!r})."
            )
        if not math.isfinite(vif_threshold):
            # NaN compares false with everything, so every VIF would pass a
            # "below the threshold" test as not-below and be flagged; infinity
            # would silently switch the check off. Neither is a threshold.
            raise ConfigurationError(
                "multicollinearity_vif_threshold must be finite, got "
                f"{vif_threshold!r}. To stop the check from running, leave it out "
                "of the inspector's checks instead."
            )
        if vif_threshold <= 1.0:
            raise ConfigurationError(
                "multicollinearity_vif_threshold must be greater than 1, got "
                f"{self.multicollinearity_vif_threshold}. A variance inflation "
                "factor is never below 1, so a threshold at or below it would "
                "flag every numeric feature regardless of the data."
            )

        for name in ("probability_column", "predicted_class_column", "prediction_column"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"{name} must be a non-empty string, got {value!r}.")

    @staticmethod
    def _require_seed(name: str, value: Any) -> None:
        """Refuse a seed that is not a plain integer.

        The field is annotated ``int`` and nothing enforced it, so a
        ``numpy.random.RandomState`` or ``Generator`` could be stored here and
        handed by reference to every consumer. That is not a seed: it is mutable
        state, and each draw advances it, so two runs configured identically
        produce different results. Splitting made the defect reachable through a
        supported public call, which is what turned it from a latent flaw into
        one worth refusing.

        ``numpy.int64`` and friends are accepted. They are immutable, they are
        exactly as deterministic as a Python ``int``, scikit-learn takes them
        everywhere, and ``np.arange(3)[0]`` produces one -- refusing them would
        turn an ordinary way of getting hold of a number into an error, and the
        message about mutable generator state would be untrue of them.

        ``bool`` is excluded explicitly. It is a subclass of ``int``, so
        ``random_state=True`` would otherwise be silently accepted as the seed 1.
        ``None`` is refused too: scikit-learn reads it as "draw from global
        entropy", which is the one thing a reproducibility contract cannot allow.

        The range is checked as well as the type. numpy accepts a seed in
        ``[0, 2**32 - 1]``, and a value outside it fails much later inside a
        splitter, where the error this library raises would blame
        ``validation_size`` for something ``random_state`` did.
        """
        import numbers

        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ConfigurationError(
                f"{name} must be an integer, got {type(value).__name__} "
                f"({value!r}). A generator object is mutable state rather than a "
                "seed -- every draw advances it, so two runs configured "
                "identically would not agree. Pass the integer you would have "
                "seeded that generator with."
            )
        if not 0 <= int(value) <= 2**32 - 1:
            raise ConfigurationError(
                f"{name} must lie between 0 and 2**32 - 1, got {value!r}. Outside "
                "that range numpy refuses it, and the refusal would arrive from a "
                "splitter several steps later."
            )

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
