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
    OrdinalDomainGuard,
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
        self._resolved_cache: tuple[list[tuple[str, Any]], list[str]] | None = None

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
        self._transformer.fit(self._prepare(X, fitting=True), y)
        self._fitted = True
        self._resolved_cache = None
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
        result = self._transformer.fit_transform(self._prepare(X, fitting=True), y)
        self._fitted = True
        self._resolved_cache = None
        return result

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the name of every output column, in output order.

        Plain names are preferred, but two outputs can carry the same one -- a
        one-hot level that happens to match another column's label, say. Rather
        than raising where ``transform`` succeeds, the group prefix is turned back
        on, and a numeric suffix settles anything still colliding after that. The
        result is always as wide as the transformed matrix and always unique.

        Args:
            input_features: Optional input names, checked against the ones seen
                at fit time. The names are derived from what was fitted, so this
                only ever confirms the caller and this object agree.

        Raises:
            SchemaError: If ``input_features`` names something else.
        """
        self._require_fitted()
        if input_features is not None:
            seen = [str(c) for c in getattr(self._transformer, "feature_names_in_", [])]
            given = [str(f) for f in input_features]
            if seen and given != seen:
                raise SchemaError(
                    "input_features does not match the columns this preprocessor was "
                    f"fitted on. Expected {seen}, got {given}."
                )
        return np.asarray(self._resolved()[1], dtype=object)

    @property
    def n_features_out(self) -> int:
        """How many columns the transformation produces."""
        return int(len(self.get_feature_names_out()))

    def lineage(self) -> dict[Hashable, tuple[str, ...]]:
        """Map each input column to the output columns it produced.

        One-hot encoding turns ``city`` into ``city_Riyadh`` and ``city_Jeddah``;
        this says so. Built by walking each group's fitted steps rather than by
        parsing names, so a category containing an underscore cannot confuse it,
        and the names are the very ones :meth:`get_feature_names_out` returns.
        """
        self._require_fitted()
        normalisation = self._plan.label_normalisation
        pairs, names = self._resolved()
        result: dict[Hashable, list[str]] = {}
        for (_, column), name in zip(pairs, names):
            result.setdefault(normalisation.original(str(column)), []).append(name)
        return {column: tuple(produced) for column, produced in result.items()}

    def _resolved(self) -> tuple[list[tuple[str, Any]], list[str]]:
        """Attribute every output column to its source, and name it uniquely.

        One computation behind both :meth:`get_feature_names_out` and
        :meth:`lineage`: derived separately they could disagree, and a lineage
        entry naming a column absent from the output is worse than no lineage.
        """
        if self._resolved_cache is None:
            pairs: list[tuple[str, Any]] = []
            bases: list[str] = []
            for group, fitted, columns in self._transformer.transformers_:
                if group == "remainder" or fitted in ("drop", "passthrough"):
                    continue
                for column, produced in _group_pairs(fitted, columns):
                    pairs.append((group, column))
                    bases.append(produced)

            names = bases
            if len(set(names)) != len(names):
                names = [f"{group}__{base}" for (group, _), base in zip(pairs, bases)]
                _logger.debug("output names collided; group prefixes re-enabled")
            if len(set(names)) != len(names):
                # Two outputs of the *same* group can still collide, and no prefix
                # separates those. Silence here would mean a name lookup quietly
                # returning the wrong column.
                seen: dict[str, int] = {}
                unique: list[str] = []
                for name in names:
                    count = seen.get(name, 0)
                    seen[name] = count + 1
                    unique.append(name if count == 0 else f"{name}__{count + 1}")
                names = unique
                _logger.debug("output names collided within a group; suffixes added")

            self._resolved_cache = (pairs, names)
        return self._resolved_cache

    def _prepare(self, X: pd.DataFrame, *, fitting: bool = False) -> pd.DataFrame:
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

        if not len(X):
            # scikit-learn answers this with "Found array with 0 sample(s)", which
            # names no column and reads like a data-shape bug in the library.
            raise PreprocessingError(
                "The frame has no rows. Preprocessing needs at least one row, both to "
                "learn from and to transform."
            )

        if fitting:
            self._require_observed(X)
        else:
            self._require_no_sentinel(X)
        self._require_finite(X)
        normalisation = self._plan.label_normalisation
        if not normalisation.applied:
            return X
        return X.rename(columns=dict(normalisation.mapping))

    def _require_no_sentinel(self, X: pd.DataFrame) -> None:
        """Refuse a real category that is indistinguishable from a filled gap.

        The sentinel is chosen against the training rows, so it collides with no
        real category *there*. A frame arriving later can still contain it, and
        one-hot encoding would then put a genuine value and a generated missing in
        the same column, with nothing to say which rows were which.
        """
        # From the plan's own snapshot: the plan is what this object promised to
        # execute, and the builder already refused to build under any other.
        if self._plan.config.get("categorical_imputation") != CategoricalImputation.CONSTANT:
            return
        sentinel = self._plan.categorical_sentinel
        categorical = (
            self._plan.nominal_features
            + self._plan.binary_features
            + self._plan.ordinal_features
        )
        offending = [
            column
            for column in categorical
            if column in X.columns
            and bool((X[column].astype("object").map(_text_of) == sentinel).any())
        ]
        if offending:
            raise PreprocessingError(
                f"Column(s) {[str(c) for c in offending]} contain the value {sentinel!r}, "
                "which this preprocessor reserved to mark a missing category when it was "
                "fitted. Encoding it would merge real rows with filled ones. Rename the "
                "value, or set categorical_fill_value to something the data never uses "
                "and re-plan."
            )

    def _require_observed(self, X: pd.DataFrame) -> None:
        """Refuse to fit on a column the plan includes but this frame never fills.

        The imputer has nothing to learn from an all-missing column, so it drops
        it and warns -- leaving the fitted transformer narrower than the plan
        promised, with no error anyone would notice. The planner already excludes
        such a column; reaching here means the fitting frame is not the frame the
        plan was made from.
        """
        empty = [
            column
            for column in self._plan.included_features
            if column in X.columns and not int(X[column].notna().sum())
        ]
        if empty:
            raise PreprocessingError(
                f"Column(s) {[str(c) for c in empty]} are included in the plan but hold no "
                "observed value in this frame, so nothing can be learned from them. The "
                "plan was made from different data -- re-plan on the frame you are fitting, "
                "or exclude those columns."
            )

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
                # A text column bound for the numeric-text parser cannot be cast
                # directly. Coercing is enough to see whether it holds an
                # infinity; skipping here would let one straight through.
                values = pd.to_numeric(X[column], errors="coerce").to_numpy(
                    dtype="float64", na_value=np.nan
                )
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
            # As text, because CategoricalCaster hands the encoder text. Given the
            # levels as written -- [1, 2, 3] -- the raw values would match nothing,
            # and every row would encode as unknown: silently the sentinel under
            # ENCODE, a raw sklearn error under ERROR. Config validation already
            # refuses two levels sharing a text form, so nothing merges here.
            categories.append([str(value) for value in order])

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
        steps = []
        if config.unknown_ordinal_policy is UnknownOrdinalPolicy.ERROR:
            # First, while the columns still carry their names. Refusing an
            # unranked value is this policy's whole purpose, and left to the
            # encoder it arrives as "unknown categories in column 0", which
            # identifies nothing the analyst can act on. Imputation only ever
            # introduces values already seen, so nothing later needs re-checking.
            steps.append(
                ("domain", OrdinalDomainGuard(dict(zip(plan.ordinal_features, categories))))
            )
        steps += [
            ("cast", CategoricalCaster()),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
        return Pipeline(steps)

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


def _group_pairs(fitted: Any, columns: Sequence[Any]) -> list[tuple[Any, str]]:
    """Pair every output column of one group with the input column behind it.

    Walked step by step rather than inferred from totals: a one-hot encoder
    widens a column, an imputer with nothing to learn from drops one, and either
    alone makes a positional guess attribute outputs to the wrong input. Reading
    each step's own names keeps the attribution exact.
    """
    steps = fitted.steps if isinstance(fitted, Pipeline) else [("step", fitted)]
    current = [str(column) for column in columns]
    sources = list(columns)

    for _, step in steps:
        if step is None or isinstance(step, str):
            continue
        try:
            produced = [str(name) for name in step.get_feature_names_out(current)]
        except (AttributeError, ValueError, TypeError):  # pragma: no cover - defensive
            break

        categories = getattr(step, "categories_", None)
        if isinstance(step, OneHotEncoder) and categories is not None:
            widths = [len(values) for values in categories]
            if len(widths) == len(sources) and sum(widths) == len(produced):
                sources = [
                    source for source, width in zip(sources, widths) for _ in range(width)
                ]
        elif len(produced) != len(sources):
            # A step dropped inputs. Survivors keep their names, so the names say
            # which ones they were.
            position = {name: index for index, name in enumerate(current)}
            if all(name in position for name in produced):
                sources = [sources[position[name]] for name in produced]
            else:  # pragma: no cover - no transformer built here behaves this way
                sources = sources[: len(produced)]
        current = produced

    if len(sources) != len(current):  # pragma: no cover - defensive
        sources = (sources + [sources[-1] if sources else None] * len(current))[: len(current)]
    return list(zip(sources, current))


def _text_of(value: Any) -> Any:
    """The text form of a present value; missing stays missing."""
    return value if value is None or value != value else str(value)
