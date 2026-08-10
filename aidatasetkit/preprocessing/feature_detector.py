"""Deciding what kind of feature each column is.

Detection only. Nothing here chooses a strategy or builds a transformer; it
answers "what is this column?" and records which rule supplied the answer.

The library's one dtype classifier is :func:`core.schema.detect_column_kinds`,
and the profiler's measurements are read rather than repeated. What this module
adds is the step from *kind* to *role*: text can be a nominal label, an ordinal
level, a two-valued flag, or a record key, and only the analyst can tell some of
those apart.

**Precedence.** Every role is decided by the first rule that applies, in this
order, and the winner is recorded on the spec:

1. explicit user override (``numeric_features``, ``nominal_features``, …)
2. confirmed semantic configuration (``ordinal_orders``, ``explicit_mappings``,
   ``confirmed_id_columns``)
3. profiling and quality evidence (identifier-like, high cardinality, numbers as
   text)
4. schema detection (:class:`ColumnKind`)
5. safe default
6. review

Nothing depends on the order the rules happen to be written in.
"""

from __future__ import annotations

from collections.abc import Hashable

import pandas as pd

from aidatasetkit.core.exceptions import (
    AmbiguousFeatureRoleError,
    MissingOrdinalOrderError,
    SchemaError,
)
from aidatasetkit.core.types import ColumnKind, ColumnProfile, DatasetProfile, QualityReport
from aidatasetkit.preprocessing.config import NumericTextPolicy, PreprocessingConfig
from aidatasetkit.preprocessing.types import FeatureRole, FeatureSpec

__all__ = ["FeatureDetector"]

#: Quality findings that hold a column back for a human decision.
_REVIEW_CODES = (
    "possible_numeric_stored_as_text",
    "possible_target_leakage",
    "infinite_values",
)


class FeatureDetector:
    """Assigns a :class:`FeatureRole` to every column.

    Args:
        config: Overrides and semantic declarations.
    """

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self._config = config if config is not None else PreprocessingConfig()

    @property
    def config(self) -> PreprocessingConfig:
        """The configuration whose overrides take precedence."""
        return self._config

    def detect(
        self,
        frame: pd.DataFrame,
        profile: DatasetProfile,
        quality: QualityReport | None = None,
        *,
        target: Hashable | None = None,
    ) -> tuple[FeatureSpec, ...]:
        """Describe every column of ``frame`` except the target.

        Args:
            frame: The training frame. Read, never modified.
            profile: Measurements already taken by the profiler.
            quality: Findings already made by the quality inspector.
            target: The target column, excluded from the feature set.

        Returns:
            One :class:`FeatureSpec` per feature column, in frame order.

        Raises:
            SchemaError: If an override names a column that does not exist.
            AmbiguousFeatureRoleError: If overrides assign two roles to one column.
            MissingOrdinalOrderError: If a column is declared ordinal with no order.
        """
        self._validate_overrides(frame, target)
        report = quality if quality is not None else QualityReport()
        by_column = {p.name: p for p in profile.column_profiles}

        return tuple(
            self._describe(column, by_column[column], report, frame[column])
            for column in frame.columns
            if column != target and column in by_column
        )

    # ------------------------------------------------------------------ #
    # Validation of what the analyst asked for
    # ------------------------------------------------------------------ #

    def _validate_overrides(self, frame: pd.DataFrame, target: Hashable | None) -> None:
        """Check every named column exists and no column is named twice."""
        config = self._config
        groups = {
            "numeric_features": config.numeric_features,
            "nominal_features": config.nominal_features,
            "ordinal_features": config.ordinal_features,
            "confirmed_id_columns": config.confirmed_id_columns,
            "force_include": config.force_include,
            "force_exclude": config.force_exclude,
        }
        known = set(frame.columns)

        for name, columns in groups.items():
            missing = [c for c in columns if c not in known]
            if missing:
                raise SchemaError(
                    f"{name} names column(s) that are not in the frame: "
                    f"{[str(c) for c in missing]}. Available: "
                    f"{[str(c) for c in list(frame.columns)[:20]]}."
                )
        for label in config.ordinal_orders:
            if label not in known:
                raise SchemaError(
                    f"ordinal_orders names {label!r}, which is not a column of this frame."
                )
        for label in config.explicit_mappings:
            if label not in known:
                raise SchemaError(
                    f"explicit_mappings names {label!r}, which is not a column of this frame."
                )
        if target is not None and target in known:
            for name, columns in groups.items():
                if target in columns:
                    raise AmbiguousFeatureRoleError(
                        f"{target!r} is the target and cannot also appear in {name}."
                    )

        role_groups = {
            FeatureRole.NUMERIC: config.numeric_features,
            FeatureRole.NOMINAL: config.nominal_features,
            FeatureRole.ORDINAL: config.ordinal_features,
        }
        seen: dict[Hashable, FeatureRole] = {}
        for role, columns in role_groups.items():
            for column in columns:
                if column in seen:
                    raise AmbiguousFeatureRoleError(
                        f"{column!r} is declared both {seen[column].value} and "
                        f"{role.value}. A column has one role; choose which."
                    )
                seen[column] = role

        both = set(config.ordinal_orders) & set(config.explicit_mappings)
        if both:
            raise AmbiguousFeatureRoleError(
                f"Column(s) {sorted(map(str, both))} have both an ordinal order and "
                "an explicit mapping. Those are two different encodings; declare one."
            )

        for column in config.ordinal_features:
            if column not in config.ordinal_orders:
                raise MissingOrdinalOrderError(
                    f"{column!r} is declared ordinal but no order was given. Supply "
                    f'ordinal_orders={{{column!r}: [...]}} listing its levels from '
                    "lowest to highest. The order is never inferred: alphabet, first "
                    "appearance, and frequency all produce a number that looks like a "
                    "measurement and is not one."
                )

    # ------------------------------------------------------------------ #
    # Role assignment
    # ------------------------------------------------------------------ #

    def _describe(
        self,
        column: Hashable,
        profile: ColumnProfile,
        quality: QualityReport,
        values: pd.Series,
    ) -> FeatureSpec:
        """Build the specification for one column."""
        review_codes = self._review_codes(column, quality)
        if profile.detected_kind is ColumnKind.CATEGORICAL and _has_text_collision(values):
            review_codes = tuple(sorted(review_codes + ("categorical_type_collision",)))
        role, source = self._resolve_role(column, profile, review_codes)
        order = self._config.ordinal_orders.get(column)
        mapping = self._config.explicit_mappings.get(column)

        return FeatureSpec(
            name=column,
            column_kind=profile.detected_kind,
            role=role,
            role_source=source,
            cardinality=profile.unique_count,
            missing_count=profile.missing_count,
            missing_ratio=profile.missing_ratio,
            is_constant=profile.is_constant,
            is_near_constant=profile.is_near_constant,
            is_high_cardinality=profile.is_high_cardinality,
            is_id_like=profile.is_id_like,
            infinite_count=profile.infinite_count,
            ordinal_order=None if order is None else tuple(order),
            explicit_mapping=None if mapping is None else dict(mapping),
            review_codes=review_codes,
        )

    def _resolve_role(
        self, column: Hashable, profile: ColumnProfile, review_codes: tuple[str, ...]
    ) -> tuple[FeatureRole, str]:
        """Return the role and the precedence rule that decided it."""
        config = self._config

        # 1. Explicit user override.
        if column in config.numeric_features:
            return FeatureRole.NUMERIC, "override:numeric_features"
        if column in config.nominal_features:
            return FeatureRole.NOMINAL, "override:nominal_features"
        if column in config.ordinal_features:
            return FeatureRole.ORDINAL, "override:ordinal_features"

        # 2. Confirmed semantic configuration.
        if column in config.confirmed_id_columns:
            return FeatureRole.ID_LIKE, "config:confirmed_id_columns"
        if column in config.ordinal_orders:
            return FeatureRole.ORDINAL, "config:ordinal_orders"
        if column in config.explicit_mappings:
            return FeatureRole.NOMINAL, "config:explicit_mappings"
        if (
            "possible_numeric_stored_as_text" in review_codes
            and config.numeric_text_policy is NumericTextPolicy.CONVERT
        ):
            # The analyst accepted the consequences of parsing this column, so it
            # becomes numeric here rather than being one-hot encoded as text.
            return FeatureRole.NUMERIC, "config:numeric_text_policy"

        # 3. Profiling and quality evidence. force_include is an explicit override
        #    of this heuristic, so the column falls through to schema detection and
        #    gets a role a transformer group can actually consume.
        if profile.is_id_like and column not in config.force_include:
            return FeatureRole.ID_LIKE, "profile:is_id_like"

        # 4. Schema detection, then 5. safe defaults.
        kind = profile.detected_kind
        if kind is ColumnKind.NUMERIC:
            return FeatureRole.NUMERIC, "schema:numeric"
        if kind is ColumnKind.DATETIME:
            return FeatureRole.DATETIME, "schema:datetime"
        if kind is ColumnKind.BOOLEAN:
            return FeatureRole.BINARY_CATEGORICAL, "schema:boolean"
        if kind is ColumnKind.CATEGORICAL:
            if profile.unique_count == 2:
                return FeatureRole.BINARY_CATEGORICAL, "schema:two_valued_categorical"
            return FeatureRole.NOMINAL, "schema:categorical"

        # 6. Anything else is not something this version knows how to encode.
        return FeatureRole.UNSUPPORTED, "schema:unsupported_kind"

    @staticmethod
    def _review_codes(column: Hashable, quality: QualityReport) -> tuple[str, ...]:
        """Return the quality findings that bear on preprocessing this column."""
        codes = {
            issue.code
            for issue in quality.by_column(column)
            if issue.code in _REVIEW_CODES or issue.code == "target_leakage_exact_duplicate"
        }
        return tuple(sorted(codes))


def _has_text_collision(values: pd.Series) -> bool:
    """Whether two distinct categories in this column share a text form.

    ``1`` and ``"1"`` are different categories. Any encoding that goes through
    text would merge them, and nothing downstream could separate them again.
    """
    distinct = values.dropna().unique()
    if len(distinct) < 2:
        return False
    return len({str(value) for value in distinct}) < len(distinct)
