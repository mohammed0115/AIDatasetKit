"""What a metric is, and what it means when there isn't one.

A metric that could not be computed is not a zero, and it is not a ``NaN``. It is
one of several quite different situations, and collapsing them loses exactly the
information a reader needs:

===========================  ===============================================
Status                       Means
===========================  ===============================================
``AVAILABLE``                A number was computed and can be compared.
``NOT_APPLICABLE``           Meaningless for this task or target shape.
``UNSUPPORTED_BY_MODEL``     The estimator cannot produce the required output.
``UNDEFINED``                Mathematically undefined on this evaluation set.
``FAILED``                   The computation raised.
===========================  ===============================================

The distinction is not academic. ROC-AUC absent because a decision tree exposes
no probabilities is a fact about the *model*; ROC-AUC absent because the
evaluation rows all share one class is a fact about the *split*; and R² absent
because the target never varies is a fact about the *data*. A single ``NaN``
would say none of those, and a comparison table that silently omitted the row
would let two models be ranked as though they had been measured the same way.

Direction is carried on every metric for the same reason. ``MAE = 4.2`` and
``R2 = 0.91`` are both good news and both bad news depending on which way the
scale runs, and a ranking that hard-coded "bigger is better" would put the worst
regressor first. scikit-learn solves this by negating -- ``neg_mean_absolute_error
= -4.2`` -- which is correct arithmetic and a terrible thing to show a reader.
The real value is reported here and the direction travels beside it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from aidatasetkit.core.exceptions import ValidationError
from aidatasetkit.core.types import TaskType

__all__ = [
    "MetricDirection",
    "MetricStatus",
    "MetricValue",
    "EvaluationReport",
]


class MetricDirection(StrEnum):
    """Which way a metric's scale runs."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class MetricStatus(StrEnum):
    """Why a metric does or does not carry a number."""

    AVAILABLE = "available"
    NOT_APPLICABLE = "not_applicable"
    UNSUPPORTED_BY_MODEL = "unsupported_by_model"
    UNDEFINED = "undefined"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MetricValue:
    """One measurement, or one recorded reason there isn't one.

    Attributes:
        name: The metric's canonical name, such as ``"f1"`` or ``"rmse"``.
        value: The number, or ``None`` whenever ``status`` is not ``AVAILABLE``.
        direction: Which way the scale runs. Present even when the value is not,
            because a reader comparing two absent metrics still needs to know
            what would have been better.
        status: Why the value is or is not present.
        reason: A sentence a reader can act on, required whenever the metric is
            unavailable and forbidden when it is not -- an explanation attached
            to a number that exists would be describing nothing.
        detail: Optional structured extras, such as per-class scores. JSON-safe.
    """

    name: str
    value: float | None
    direction: MetricDirection
    status: MetricStatus = MetricStatus.AVAILABLE
    reason: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status is MetricStatus.AVAILABLE:
            if self.value is None:
                raise ValidationError(
                    f"Metric {self.name!r} is marked available but carries no value."
                )
            if not math.isfinite(float(self.value)):
                raise ValidationError(
                    f"Metric {self.name!r} is marked available but its value is "
                    f"{self.value!r}. A non-finite score is a result that needs a "
                    "status of its own, not a number to be compared."
                )
            if self.reason is not None:
                raise ValidationError(
                    f"Metric {self.name!r} carries both a value and a reason it is "
                    "missing. Only one of those can be true."
                )
        else:
            if self.value is not None:
                raise ValidationError(
                    f"Metric {self.name!r} is marked {self.status.value} but still "
                    "carries a value."
                )
            if not self.reason:
                raise ValidationError(
                    f"Metric {self.name!r} is unavailable without saying why. A "
                    "reader cannot act on a blank."
                )

    @property
    def is_available(self) -> bool:
        """Whether this metric carries a comparable number."""
        return self.status is MetricStatus.AVAILABLE

    def is_better_than(self, other: MetricValue) -> bool:
        """Whether this value is the better of two, in this metric's direction.

        Raises:
            ValidationError: If the two are different metrics, or either is
                unavailable. Comparing an absent score to a present one has no
                answer, and inventing one is how a failed model wins a ranking.
        """
        if self.name != other.name:
            raise ValidationError(
                f"{self.name!r} and {other.name!r} are different metrics and cannot "
                "be compared."
            )
        if not (self.is_available and other.is_available):
            raise ValidationError(
                f"Metric {self.name!r} is not available on both sides, so neither "
                "is better. Filter to available metrics before ranking."
            )
        if self.direction is MetricDirection.HIGHER_IS_BETTER:
            return float(self.value) > float(other.value)  # type: ignore[arg-type]
        return float(self.value) < float(other.value)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the metric."""
        from aidatasetkit.core.types import jsonable

        return {
            "name": self.name,
            "value": None if self.value is None else float(self.value),
            "direction": self.direction.value,
            "status": self.status.value,
            "reason": self.reason,
            "detail": {str(key): jsonable(item) for key, item in self.detail.items()},
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Every metric computed for one model on one evaluation set.

    Deliberately not a bare mapping of name to float. The absent metrics are as
    much a part of the report as the present ones, and a plain dictionary can
    only represent them by leaving them out -- which is precisely the silence
    this type exists to prevent.

    Attributes:
        task_type: The task family the metric policy was chosen for.
        metrics: Every metric the policy attempted, available or not, in policy
            order.
        row_count: How many evaluation rows produced these numbers.
        class_labels: The original target labels, for a classification report.
            ``None`` for regression, where there are none.
    """

    task_type: TaskType
    metrics: tuple[MetricValue, ...]
    row_count: int
    class_labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        names = [metric.name for metric in self.metrics]
        if len(names) != len(set(names)):
            duplicated = sorted({name for name in names if names.count(name) > 1})
            raise ValidationError(
                f"An evaluation report lists {duplicated} more than once. Two "
                "numbers under one name cannot both be the metric."
            )

    def __getitem__(self, name: str) -> MetricValue:
        """Return one metric by name.

        Raises:
            KeyError: If the policy never attempted it. This is deliberately
                different from a metric that was attempted and came back
                unavailable -- that one is present, with its reason.
        """
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(
            f"{name!r} was not attempted for a {self.task_type.value} task. "
            f"Attempted: {[metric.name for metric in self.metrics]}."
        )

    def __contains__(self, name: object) -> bool:
        return any(metric.name == name for metric in self.metrics)

    def __iter__(self):
        return iter(self.metrics)

    def __len__(self) -> int:
        return len(self.metrics)

    @property
    def available(self) -> tuple[MetricValue, ...]:
        """The metrics that carry a comparable number."""
        return tuple(metric for metric in self.metrics if metric.is_available)

    @property
    def unavailable(self) -> tuple[MetricValue, ...]:
        """The metrics that do not, each with its reason."""
        return tuple(metric for metric in self.metrics if not metric.is_available)

    def value_of(self, name: str) -> float | None:
        """Return one metric's number, or ``None`` if it has none."""
        return self[name].value

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the whole report."""
        return {
            "task_type": self.task_type.value,
            "row_count": self.row_count,
            "class_labels": None if self.class_labels is None else list(self.class_labels),
            "metrics": [metric.to_dict() for metric in self.metrics],
        }
