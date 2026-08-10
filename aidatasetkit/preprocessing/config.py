"""Configuration for the preprocessing layer.

Separate from :class:`~aidatasetkit.core.config.KitConfig` because imputation
strategies and encoder policies are not modelling thresholds, and because this is
where an analyst exercises control: which columns are really identifiers, what
order an ordinal feature has, what to do with a category nobody saw during
training.

Every option here either states a fact the library cannot discover -- an ordering,
a semantic mapping -- or overrides a default the library chose conservatively.
Nothing here makes the library guess harder.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from aidatasetkit.core.exceptions import ConfigurationError
from aidatasetkit.core.types import jsonable

__all__ = [
    "NumericImputation",
    "CategoricalImputation",
    "NumericScaler",
    "HighCardinalityPolicy",
    "UnknownOrdinalPolicy",
    "NumericTextPolicy",
    "PreprocessingConfig",
]


class NumericImputation(StrEnum):
    """How a missing number is filled."""

    MEDIAN = "median"
    MEAN = "mean"
    CONSTANT = "constant"


class CategoricalImputation(StrEnum):
    """How a missing label is filled."""

    MOST_FREQUENT = "most_frequent"
    CONSTANT = "constant"


class NumericScaler(StrEnum):
    """Which scaler is used when the model asks for scaling."""

    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"


class HighCardinalityPolicy(StrEnum):
    """What to do with a categorical column that has too many levels.

    ``REVIEW`` is the default and holds the column back. One-hot encoding a
    column with half a million levels produces half a million columns, and a
    memory failure is a poor way to learn that.
    """

    REVIEW = "review"
    ONEHOT = "onehot"


class UnknownOrdinalPolicy(StrEnum):
    """What to do with an ordinal value that was never seen in training.

    ``ERROR`` is the default: an unknown value has no position in the ordering,
    and inventing one puts a number on the axis that means nothing.
    """

    ERROR = "error"
    ENCODE = "encode"


class NumericTextPolicy(StrEnum):
    """What to do with numbers stored as text.

    ``REVIEW`` is the default. Converting looks harmless and is not: the values
    that fail to parse become missing, and a column of sentinels turns into a
    column of imputed medians without anyone deciding that.
    """

    REVIEW = "review"
    CONVERT = "convert"


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Strategies, policies, and analyst overrides.

    Attributes:
        numeric_imputation: Fill strategy for missing numbers.
        numeric_fill_value: Value used when ``numeric_imputation`` is constant.
        categorical_imputation: Fill strategy for missing labels.
        categorical_fill_value: Sentinel used when ``categorical_imputation`` is
            constant. If a real category already carries this value, the planner
            picks a non-colliding one and records it, so generated missings never
            merge with genuine data.
        numeric_scaler: Which scaler to use *if* the model requires scaling.
            Scaling is never applied merely because a column is numeric.
        ordinal_orders: The ordering of each ordinal feature, lowest first. The
            library never infers one.
        explicit_mappings: Analyst-supplied value-to-number mappings, replacing
            the encoder for those columns.
        numeric_features, nominal_features, ordinal_features: Role overrides.
        confirmed_id_columns: Columns the analyst confirms are identifiers. These
            are excluded outright rather than held for review.
        force_include: Columns to keep even when a default would hold them back.
        force_exclude: Columns to drop regardless of anything else.
        high_cardinality_policy: What to do with a many-levelled category.
        unknown_ordinal_policy: What to do with an unseen ordinal value.
        unknown_ordinal_value: Encoded value used when that policy is ``ENCODE``.
        numeric_text_policy: What to do with numbers stored as text.
        explicit_mapping_unknown_value: What an unmapped value becomes in a column
            with an explicit mapping. ``None`` -- the default -- makes an unmapped
            value an error naming the column and the offending values, rather
            than a number nobody chose.
        drop_exact_target_duplicates: Whether a feature the quality inspector
            proved equal to the target is excluded. True by default -- that
            finding is certain, not heuristic.
    """

    numeric_imputation: NumericImputation = NumericImputation.MEDIAN
    numeric_fill_value: float = 0.0
    categorical_imputation: CategoricalImputation = CategoricalImputation.MOST_FREQUENT
    categorical_fill_value: str = "__missing__"
    numeric_scaler: NumericScaler = NumericScaler.STANDARD

    ordinal_orders: Mapping[Hashable, Sequence[Any]] = field(default_factory=dict)
    explicit_mappings: Mapping[Hashable, Mapping[Any, Any]] = field(default_factory=dict)

    numeric_features: tuple[Hashable, ...] = ()
    nominal_features: tuple[Hashable, ...] = ()
    ordinal_features: tuple[Hashable, ...] = ()
    confirmed_id_columns: tuple[Hashable, ...] = ()
    force_include: tuple[Hashable, ...] = ()
    force_exclude: tuple[Hashable, ...] = ()

    high_cardinality_policy: HighCardinalityPolicy = HighCardinalityPolicy.REVIEW
    unknown_ordinal_policy: UnknownOrdinalPolicy = UnknownOrdinalPolicy.ERROR
    unknown_ordinal_value: int = -1
    numeric_text_policy: NumericTextPolicy = NumericTextPolicy.REVIEW
    explicit_mapping_unknown_value: float | None = None
    drop_exact_target_duplicates: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "numeric_imputation", NumericImputation(self.numeric_imputation))
        object.__setattr__(
            self, "categorical_imputation", CategoricalImputation(self.categorical_imputation)
        )
        object.__setattr__(self, "numeric_scaler", NumericScaler(self.numeric_scaler))
        object.__setattr__(
            self, "high_cardinality_policy", HighCardinalityPolicy(self.high_cardinality_policy)
        )
        object.__setattr__(
            self, "unknown_ordinal_policy", UnknownOrdinalPolicy(self.unknown_ordinal_policy)
        )
        object.__setattr__(self, "numeric_text_policy", NumericTextPolicy(self.numeric_text_policy))

        for name in (
            "numeric_features",
            "nominal_features",
            "ordinal_features",
            "confirmed_id_columns",
            "force_include",
            "force_exclude",
        ):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise ConfigurationError(
                    f"{name} must be a sequence of column labels, not a single "
                    f"{type(value).__name__}. Pass ({value!r},) to name one column."
                )
            object.__setattr__(self, name, tuple(value))

        if not isinstance(self.categorical_fill_value, str) or not self.categorical_fill_value:
            raise ConfigurationError(
                "categorical_fill_value must be a non-empty string, got "
                f"{self.categorical_fill_value!r}."
            )
        for label, order in self.ordinal_orders.items():
            if isinstance(order, (str, bytes)) or not isinstance(order, Sequence):
                raise ConfigurationError(
                    f"The ordinal order for {label!r} must be a sequence of values "
                    "from lowest to highest."
                )
            if len(order) < 2:
                raise ConfigurationError(
                    f"The ordinal order for {label!r} needs at least two levels, got "
                    f"{list(order)}."
                )
            if len(set(map(str, order))) != len(order):
                raise ConfigurationError(
                    f"The ordinal order for {label!r} repeats a level: {list(order)}."
                )

        overlap = set(self.force_include) & set(self.force_exclude)
        if overlap:
            raise ConfigurationError(
                f"Columns cannot be both force-included and force-excluded: "
                f"{sorted(map(str, overlap))}."
            )

    def replace(self, **changes: Any) -> PreprocessingConfig:
        """Return a new configuration with ``changes`` applied and re-validated."""
        unknown = set(changes) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise ConfigurationError(f"Unknown preprocessing options: {sorted(unknown)}.")
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of every configured value."""
        return {
            field_.name: jsonable(getattr(self, field_.name))
            for field_ in dataclasses.fields(self)
        }
