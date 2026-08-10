"""Turning a plan into transformers, and fitting them on training rows only.

Two things are kept carefully apart here.

**A blueprint is shared; fitted state never is.** Two models with the same
:class:`~aidatasetkit.core.types.PreprocessingProfile` need the same pipeline
*shape*, and computing it twice is waste -- so blueprints are cached by
capability profile and plan structure, never by model name. But a fitted
transformer holds the training median, the learned categories, the scaler's
centre; handing the same fitted object to a second run would leak one run's data
into another's. Every :meth:`PreprocessorBuilder.build` therefore constructs
fresh, unfitted transformers.

**Fitting sees training rows only.** Categories, medians, and scales come from
whatever frame is passed to ``fit``, and ``transform`` applies them unchanged. A
category that appears only at inference becomes all zeros rather than an error,
and never joins the learned vocabulary.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from aidatasetkit.core.exceptions import (
    NoUsableFeaturesError,
    PreprocessingError,
    SchemaError,
)
from aidatasetkit.preprocessing.config import (
    CategoricalImputation,
    NumericImputation,
    NumericScaler,
    PreprocessingConfig,
    UnknownOrdinalPolicy,
)
from aidatasetkit.preprocessing.transformers import (
    CategoricalCaster,
    ExplicitMappingEncoder,
    NumericCaster,
    NumericTextConverter,
)
from aidatasetkit.preprocessing.types import (
    FeatureAction,
    FeatureRole,
    PreprocessingPlan,
)

__all__ = ["PreprocessorBuilder", "FittedPreprocessor", "BlueprintCache"]

_logger = logging.getLogger(__name__)

_SCALERS = {
    NumericScaler.STANDARD: StandardScaler,
    NumericScaler.MINMAX: MinMaxScaler,
    NumericScaler.ROBUST: RobustScaler,
}


class BlueprintCache:
    """Remembers which pipeline shape a plan calls for.

    Keyed by the capability profile and the plan's structural fingerprint, never
    by a model name -- so two models that need the same preprocessing share the
    entry, and a third that needs something different does not.

    The cache stores *descriptions*. Nothing fitted is kept, so nothing learned
    from one training run can reach another.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], tuple[str, ...]] = {}

    def key_for(self, plan: PreprocessingPlan) -> tuple[str, str]:
        """Return the identity two equivalent plans share."""
        return (plan.preprocessing_profile.key, plan.fingerprint)

    def register(self, plan: PreprocessingPlan) -> tuple[str, str]:
        """Record this plan's shape and return its key."""
        key = self.key_for(plan)
        self._entries.setdefault(key, tuple(sorted(str(c) for c in plan.included_features)))
        return key

    def __contains__(self, plan: object) -> bool:
        return isinstance(plan, PreprocessingPlan) and self.key_for(plan) in self._entries

    def __len__(self) -> int:
        return len(self._entries)


class FittedPreprocessor:
    """A preprocessor that knows its own plan.

    Wraps a :class:`~sklearn.compose.ColumnTransformer` and adds what the
    transformer cannot answer on its own: which plan produced it, which input
    column each output column came from, and whether the frame it is being asked
    to transform is the shape it was fitted for.

    Args:
        transformer: The unfitted ColumnTransformer built from ``plan``.
        plan: The plan it executes.
    """

    def __init__(self, transformer: ColumnTransformer, plan: PreprocessingPlan) -> None:
        self._transformer = transformer
        self._plan = plan
        self._fitted = False

    @property
    def plan(self) -> PreprocessingPlan:
        """The plan this preprocessor executes."""
        return self._plan

    @property
    def transformer(self) -> ColumnTransformer:
        """The underlying scikit-learn transformer."""
        return self._transformer

    @property
    def is_fitted(self) -> bool:
        """Whether training statistics have been learned."""
        return self._fitted

    def fit(self, X: pd.DataFrame, y: Any = None) -> FittedPreprocessor:
        """Learn every statistic from ``X`` alone."""
        self._transformer.fit(self._prepare(X), y)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> Any:
        """Apply the learned transformation to ``X``.

        Raises:
            PreprocessingError: If called before fitting.
        """
        self._require_fitted()
        return self._transformer.transform(self._prepare(X))

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> Any:
        """Fit on ``X`` and return it transformed."""
        result = self._transformer.fit_transform(self._prepare(X), y)
        self._fitted = True
        return result

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the name of every output column.

        Plain names are preferred, but two groups can produce the same one -- a
        one-hot level that happens to match another column's label, say. Rather
        than raising where ``transform`` succeeds, the group prefix is turned back
        on so every name is unique.
        """
        self._require_fitted()
        try:
            return self._transformer.get_feature_names_out(input_features)
        except ValueError:
            previous = self._transformer.verbose_feature_names_out
            self._transformer.verbose_feature_names_out = True
            try:
                names = self._transformer.get_feature_names_out(input_features)
            finally:
                self._transformer.verbose_feature_names_out = previous
            _logger.debug("output names collided; group prefixes re-enabled")
            return names

    @property
    def n_features_out(self) -> int:
        """How many columns the transformation produces."""
        return int(len(self.get_feature_names_out()))

    def lineage(self) -> dict[Hashable, tuple[str, ...]]:
        """Map each input column to the output columns it produced.

        One-hot encoding turns ``city`` into ``city_Riyadh`` and ``city_Jeddah``;
        this says so. Built from each group's own encoder rather than by parsing
        names, so a category containing an underscore cannot confuse it.
        """
        self._require_fitted()
        normalisation = self._plan.label_normalisation
        result: dict[Hashable, list[str]] = {}

        for name, fitted, columns in self._transformer.transformers_:
            if name == "remainder" or fitted in ("drop", "passthrough"):
                continue
            outputs = list(fitted.get_feature_names_out([str(c) for c in columns]))
            widths = _output_widths(fitted, columns, len(outputs))
            position = 0
            for column, width in zip(columns, widths):
                original = normalisation.original(str(column))
                produced = outputs[position : position + width]
                result.setdefault(original, []).extend(str(o) for o in produced)
                position += width

        return {column: tuple(names) for column, names in result.items()}

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate the frame and apply any recorded label renaming.

        The caller's frame is never modified: renaming produces a new object that
        shares the same column data.
        """
        if not isinstance(X, pd.DataFrame):
            raise SchemaError(f"A pandas DataFrame is required, got {type(X).__name__}.")

        missing = [c for c in self._plan.included_features if c not in X.columns]
        if missing:
            raise SchemaError(
                f"This preprocessor needs column(s) {[str(c) for c in missing]}, which "
                "are not in the frame it was given."
            )

        self._require_finite(X)
        normalisation = self._plan.label_normalisation
        if not normalisation.applied:
            return X
        return X.rename(columns=dict(normalisation.mapping))

    def _require_finite(self, X: pd.DataFrame) -> None:
        """Refuse infinities before scikit-learn reports them without a column name.

        This version does not support infinity as a trainable value. The planner
        holds such a column back, but one can still arrive here -- through
        force_include, through a test row, or created by parsing text such as
        "inf" -- so the check runs on every fit and every transform.
        """
        for column in self._plan.numeric_features:
            if column not in X.columns:
                continue
            try:
                values = X[column].to_numpy(dtype="float64", na_value=np.nan)
            except (TypeError, ValueError):
                continue
            infinite = int(np.isinf(values).sum())
            if infinite:
                raise PreprocessingError(
                    f"Column {column!r} contains {infinite} infinite value(s), which "
                    "this version does not support as a trainable value. Remove the "
                    "rows, cap the values, or exclude the column -- the library will "
                    "not replace an infinity with a number nobody measured."
                )

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise PreprocessingError(
                "This preprocessor has not been fitted; call fit(X_train) first. "
                "Fitting on anything but the training rows would leak information "
                "the model is not entitled to."
            )


class PreprocessorBuilder:
    """Builds unfitted transformers from a plan.

    Args:
        config: Strategies the plan was made under. The plan records its own
            configuration; this is used for the transformer parameters that the
            plan describes but does not carry.
        cache: Blueprint cache, shared across builds to prove that two models
            with the same capability profile resolve to one entry.
    """

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        cache: BlueprintCache | None = None,
    ) -> None:
        self._config = config if config is not None else PreprocessingConfig()
        self._cache = cache if cache is not None else BlueprintCache()

    @property
    def config(self) -> PreprocessingConfig:
        """The strategies this builder instantiates."""
        return self._config

    @property
    def cache(self) -> BlueprintCache:
        """The blueprint cache, keyed by capability profile and plan shape."""
        return self._cache

    def build(
        self, plan: PreprocessingPlan, frame: pd.DataFrame | None = None
    ) -> FittedPreprocessor:
        """Construct a new, unfitted preprocessor for ``plan``.

        Every call returns fresh transformer objects. Two preprocessors built
        from one plan share no learned state, so fitting one cannot influence the
        other.

        Args:
            plan: What to build.
            frame: The training frame, used only to choose a sentinel that
                collides with no real category.

        Returns:
            An unfitted :class:`FittedPreprocessor`.

        Raises:
            NoUsableFeaturesError: If the plan includes no features.
        """
        if not plan.included_features:
            raise NoUsableFeaturesError(
                "The plan includes no features, so there is nothing to fit. "
                f"{len(plan.excluded_features)} column(s) were excluded and "
                f"{len(plan.review_features)} held for review; see plan.describe() "
                "for the reason for each."
            )

        self._require_matching_config(plan)
        self._cache.register(plan)
        profile = plan.preprocessing_profile
        # The plan already checked this against the training data; recomputing it
        # here from an optional frame would silently merge a genuine
        # "__missing__" category with the generated one whenever no frame is given.
        sentinel = plan.categorical_sentinel
        groups: list[tuple[str, Pipeline | str, list[str]]] = []

        plain, from_text = self._split_numeric(plan)
        if plain:
            groups.append(("numeric", self._numeric_pipeline(profile), plain))
        if from_text:
            groups.append(
                ("numeric_text", self._numeric_pipeline(profile, parse_text=True), from_text)
            )

        for role, label in (
            (FeatureRole.BINARY_CATEGORICAL, "binary"),
            (FeatureRole.NOMINAL, "nominal"),
        ):
            mapped, encoded = self._split_mapped(plan, role)
            if encoded:
                groups.append(
                    (label, self._categorical_pipeline(profile, sentinel), encoded)
                )
            if mapped:
                groups.append(
                    (
                        f"{label}_mapped",
                        self._mapping_pipeline(plan, mapped, profile),
                        mapped,
                    )
                )

        ordinal = _names(plan.ordinal_features, plan)
        if ordinal:
            groups.append(("ordinal", self._ordinal_pipeline(plan, sentinel), ordinal))

        transformer = ColumnTransformer(
            transformers=groups,
            remainder="drop",
            sparse_threshold=0.3 if profile.supports_sparse_input else 0.0,
            verbose_feature_names_out=False,
        )
        _logger.debug(
            "built preprocessor with %d group(s) for profile %s",
            len(groups),
            profile.key,
        )
        return FittedPreprocessor(transformer, plan)

    # ------------------------------------------------------------------ #
    # Pipelines
    # ------------------------------------------------------------------ #

    #: Configuration that decides which transformers get constructed. A plan
    #: describes what will happen in terms of these; building under different
    #: ones would execute something the plan does not describe.
    _CONSTRUCTION_KEYS = (
        "numeric_imputation",
        "numeric_fill_value",
        "categorical_imputation",
        "numeric_scaler",
        "unknown_ordinal_policy",
        "unknown_ordinal_value",
        "explicit_mapping_unknown_value",
    )

    def _require_matching_config(self, plan: PreprocessingPlan) -> None:
        """Refuse to build a plan under a configuration it was not made under.

        The plan is a promise about what will happen. Building it with a
        different scaler or imputation strategy would quietly break that promise,
        and the plan -- which is what the analyst read, and what provenance
        keeps -- would describe something that never ran.
        """
        if not plan.config:
            return
        mine = self._config.to_dict()
        differences = {
            key: (plan.config.get(key), mine.get(key))
            for key in self._CONSTRUCTION_KEYS
            if key in plan.config and plan.config[key] != mine.get(key)
        }
        if differences:
            detail = ", ".join(
                f"{key}: plan={planned!r} builder={built!r}"
                for key, (planned, built) in sorted(differences.items())
            )
            raise PreprocessingError(
                "This plan was made under a different preprocessing configuration "
                f"than the builder is using ({detail}). Build with the same "
                "configuration you planned with, so that what runs is what the plan "
                "describes."
            )

    def _split_numeric(self, plan: PreprocessingPlan) -> tuple[list[str], list[str]]:
        """Separate numeric columns that must be parsed from text first."""
        plain: list[str] = []
        from_text: list[str] = []
        for column in plan.numeric_features:
            steps = plan.decision_for(column).steps
            target = from_text if "numeric_text_conversion" in steps else plain
            target.append(_name(column, plan))
        return plain, from_text

    def _numeric_pipeline(self, profile, parse_text: bool = False) -> Pipeline:
        """Impute unless the model handles gaps itself; scale only if asked."""
        steps: list[tuple[str, Any]] = []
        if parse_text:
            steps.append(("parser", NumericTextConverter(errors="missing")))
        else:
            # Extension dtypes carry pd.NA, which scikit-learn refuses. Casting
            # first means a nullable column behaves like any other number even
            # when the model needs no imputer in front of it.
            steps.append(("cast", NumericCaster()))
        if not profile.handles_missing_values:
            steps.append(("imputer", self._numeric_imputer()))
        if profile.requires_scaling:
            steps.append(("scaler", _SCALERS[self._config.numeric_scaler]()))
        return Pipeline(steps)

    def _numeric_imputer(self) -> SimpleImputer:
        strategy = self._config.numeric_imputation
        if strategy is NumericImputation.CONSTANT:
            return SimpleImputer(strategy="constant", fill_value=self._config.numeric_fill_value)
        return SimpleImputer(strategy=strategy.value)

    def _categorical_pipeline(self, profile, sentinel: str) -> Pipeline:
        """Impute, then one-hot with unknown categories tolerated.

        ``handle_unknown="ignore"`` is what keeps inference alive: a city nobody
        saw during training becomes all zeros instead of an exception.
        """
        return Pipeline(
            [
                ("cast", CategoricalCaster()),
                ("imputer", self._categorical_imputer(sentinel)),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=bool(profile.supports_sparse_input),
                    ),
                ),
            ]
        )

    def _categorical_imputer(self, sentinel: str) -> SimpleImputer:
        if self._config.categorical_imputation is CategoricalImputation.CONSTANT:
            return SimpleImputer(strategy="constant", fill_value=sentinel)
        return SimpleImputer(strategy="most_frequent")

    def _ordinal_pipeline(self, plan: PreprocessingPlan, sentinel: str) -> Pipeline:
        """Encode against the analyst's order, with an explicit unknown policy."""
        categories = []
        for column in plan.ordinal_features:
            order = plan.spec_for(column).ordinal_order
            if not order:
                from aidatasetkit.core.exceptions import MissingOrdinalOrderError

                raise MissingOrdinalOrderError(
                    f"{column!r} is planned as ordinal but carries no order."
                )
            categories.append(list(order))

        config = self._config
        if config.unknown_ordinal_policy is UnknownOrdinalPolicy.ENCODE:
            encoder = OrdinalEncoder(
                categories=categories,
                handle_unknown="use_encoded_value",
                unknown_value=config.unknown_ordinal_value,
            )
        else:
            encoder = OrdinalEncoder(categories=categories)

        # most_frequent, never the sentinel: a sentinel has no rank in the
        # analyst's ordering, so the encoder would refuse it or invent a level.
        return Pipeline(
            [
                ("cast", CategoricalCaster()),
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", encoder),
            ]
        )

    def _mapping_pipeline(
        self, plan: PreprocessingPlan, columns: list[str], profile: Any
    ) -> Pipeline:
        """Apply the analyst's own value-to-number mapping."""
        mappings = {}
        normalisation = plan.label_normalisation
        for name in columns:
            original = normalisation.original(name)
            mappings[name] = dict(plan.spec_for(original).explicit_mapping or {})
        steps: list[tuple[str, Any]] = [
            (
                "mapping",
                ExplicitMappingEncoder(
                    mappings=mappings,
                    unknown_value=self._config.explicit_mapping_unknown_value,
                ),
            )
        ]
        # A mapped column is a number like any other: it needs the same gap
        # filling and the same scaling the capability profile asked for, or a NaN
        # reaches a model that declared it cannot take one.
        if not profile.handles_missing_values:
            steps.append(("imputer", self._numeric_imputer()))
        if profile.requires_scaling:
            steps.append(("scaler", _SCALERS[self._config.numeric_scaler]()))
        return Pipeline(steps)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _split_mapped(
        self, plan: PreprocessingPlan, role: FeatureRole
    ) -> tuple[list[str], list[str]]:
        """Separate columns with an explicit mapping from those to be encoded."""
        mapped: list[str] = []
        encoded: list[str] = []
        for decision in plan.decisions:
            if decision.action is not FeatureAction.INCLUDE or decision.role is not role:
                continue
            name = _name(decision.feature, plan)
            spec = plan.spec_for(decision.feature)
            (mapped if spec.explicit_mapping is not None else encoded).append(name)
        return mapped, encoded

def _name(column: Hashable, plan: PreprocessingPlan) -> str:
    """Return the name a column carries inside the transformer."""
    normalisation = plan.label_normalisation
    if normalisation.applied:
        return normalisation.mapping[column]
    return column if isinstance(column, str) else str(column)


def _names(columns: Sequence[Hashable], plan: PreprocessingPlan) -> list[str]:
    """Return transformer-facing names for a group of columns."""
    return [_name(column, plan) for column in columns]


def _output_widths(fitted: Any, columns: Sequence[Any], total: int) -> list[int]:
    """How many output columns each input column produced.

    Read off the encoder's learned categories where one exists, so a category
    whose name contains a separator cannot be miscounted.
    """
    encoder = fitted
    if isinstance(fitted, Pipeline):
        encoder = fitted.steps[-1][1]

    categories = getattr(encoder, "categories_", None)
    if categories is not None and isinstance(encoder, OneHotEncoder):
        return [len(values) for values in categories]
    if len(columns):
        return [total // len(columns)] * len(columns)
    return []
