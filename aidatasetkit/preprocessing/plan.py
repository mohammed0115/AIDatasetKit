"""Turning what a feature *is* into what should happen to it.

The planner decides; it builds nothing. Every decision names the steps it
intends, the reason, and whether a human should look -- so the whole intent can
be read and argued with before a single value is touched.

Two principles run through the rules. **Scaling follows the model, not the
dtype**: a numeric column is scaled when the model's capabilities ask for it and
never merely because it holds numbers. And **the library stops rather than
guesses**: a many-levelled category, numbers stored as text, a datetime, an
infinity, a probable identifier -- each is held back and named, because the
alternative is a plausible-looking pipeline built on an invented assumption.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Sequence
from types import MappingProxyType
from typing import Any

import pandas as pd

from aidatasetkit.core.exceptions import PreprocessingError, SchemaError
from aidatasetkit.core.types import (
    DatasetProfile,
    PreprocessingProfile,
    QualityReport,
    TargetProfile,
)
from aidatasetkit.preprocessing.config import (
    CategoricalImputation,
    HighCardinalityPolicy,
    NumericImputation,
    NumericScaler,
    NumericTextPolicy,
    PreprocessingConfig,
)
from aidatasetkit.preprocessing.feature_detector import FeatureDetector
from aidatasetkit.preprocessing.types import (
    FeatureAction,
    FeatureDecision,
    FeatureRole,
    FeatureSpec,
    LabelNormalisation,
    PreprocessingPlan,
)

__all__ = ["PreprocessingPlanner", "resolve_sentinel"]

_logger = logging.getLogger(__name__)

#: Step names used in decisions, so a plan reads the same way every time.
STEP_MEDIAN_IMPUTE = "median_imputation"
STEP_MEAN_IMPUTE = "mean_imputation"
STEP_CONSTANT_IMPUTE = "constant_imputation"
STEP_MOST_FREQUENT_IMPUTE = "most_frequent_imputation"
STEP_SENTINEL_IMPUTE = "sentinel_imputation"
STEP_ONEHOT = "onehot_encoding"
STEP_ORDINAL = "ordinal_encoding"
STEP_EXPLICIT_MAPPING = "explicit_mapping"
STEP_NUMERIC_TEXT = "numeric_text_conversion"

_SCALER_STEPS = {
    NumericScaler.STANDARD: "standard_scaling",
    NumericScaler.MINMAX: "minmax_scaling",
    NumericScaler.ROBUST: "robust_scaling",
}

_NUMERIC_IMPUTE_STEPS = {
    NumericImputation.MEDIAN: STEP_MEDIAN_IMPUTE,
    NumericImputation.MEAN: STEP_MEAN_IMPUTE,
    NumericImputation.CONSTANT: STEP_CONSTANT_IMPUTE,
}


def resolve_sentinel(frame: pd.DataFrame, columns: Sequence[Hashable], wanted: str) -> str:
    """Return a fill value no real category already uses.

    A dataset may genuinely contain the string ``"__missing__"``. Filling gaps
    with the same value would merge rows that had data with rows that did not,
    and nothing downstream could tell them apart again.
    """
    existing: set[str] = set()
    for column in columns:
        if column in frame.columns:
            existing.update(str(value) for value in frame[column].dropna().unique())

    sentinel = wanted
    while sentinel in existing:
        sentinel = f"{sentinel}_"
    return sentinel


class PreprocessingPlanner:
    """Produces the plan for one frame and one set of model capabilities."""

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        detector: FeatureDetector | None = None,
    ) -> None:
        self._config = config if config is not None else PreprocessingConfig()
        self._detector = detector if detector is not None else FeatureDetector(self._config)

    @property
    def config(self) -> PreprocessingConfig:
        """The strategies and overrides this planner applies."""
        return self._config

    def plan(
        self,
        frame: pd.DataFrame,
        profile: DatasetProfile,
        capabilities: Any,
        *,
        quality: QualityReport | None = None,
        target: Hashable | None = None,
        target_profile: TargetProfile | None = None,
    ) -> PreprocessingPlan:
        """Decide what preprocessing to apply, without applying any of it.

        Args:
            frame: The training frame. Read, never modified.
            profile: Measurements from the profiler.
            capabilities: A :class:`PreprocessingProfile`, or anything exposing
                ``preprocessing_profile()`` -- which is what a model's
                capabilities do. Scaling and sparsity follow from this, never
                from the model's name.
            quality: Findings from the quality inspector.
            target: The target column, excluded from the features.
            target_profile: What the task detector concluded, recorded on the plan.

        Returns:
            A :class:`PreprocessingPlan`.
        """
        preprocessing_profile = _as_profile(capabilities)
        report = quality if quality is not None else QualityReport()
        _require_complete_profile(frame, profile, target)
        specs = self._detector.detect(frame, profile, report, target=target)

        sentinel = resolve_sentinel(
            frame,
            [spec.name for spec in specs],
            self._config.categorical_fill_value,
        )
        decisions = tuple(
            self._decide(spec, preprocessing_profile, sentinel) for spec in specs
        )

        plan = PreprocessingPlan(
            decisions=decisions,
            specs=specs,
            preprocessing_profile=preprocessing_profile,
            config=MappingProxyType(self._config.to_dict()),
            label_normalisation=_plan_label_normalisation(frame),
            target_name=None if target is None else str(target),
            categorical_sentinel=sentinel,
        )
        _logger.debug(
            "planned %d feature(s): %d included, %d excluded, %d for review",
            len(decisions),
            len(plan.included_features),
            len(plan.excluded_features),
            len(plan.review_features),
        )
        return plan

    # ------------------------------------------------------------------ #
    # One decision per feature
    # ------------------------------------------------------------------ #

    def _decide(
        self, spec: FeatureSpec, profile: PreprocessingProfile, sentinel: str
    ) -> FeatureDecision:
        """Choose what happens to one feature, in precedence order."""
        config = self._config

        if spec.name in config.force_exclude:
            return self._exclude(
                spec, "force_exclude", f"Excluded because {spec.name!r} is in force_exclude."
            )

        forced = spec.name in config.force_include
        blocked = None if forced else self._blocking_decision(spec, sentinel)
        if blocked is not None:
            return blocked

        if spec.role is FeatureRole.NUMERIC:
            return self._numeric(spec, profile, forced)
        if spec.role is FeatureRole.ORDINAL:
            return self._ordinal(spec, sentinel)
        if spec.role in (FeatureRole.NOMINAL, FeatureRole.BINARY_CATEGORICAL):
            return self._categorical(spec, profile, sentinel)
        if spec.role is FeatureRole.ID_LIKE:
            return self._identifier(spec, forced)
        if spec.role is FeatureRole.DATETIME:
            return self._review(
                spec,
                "datetime_requires_feature_engineering",
                f"Held back because {spec.name!r} is a datetime. Turning a timestamp "
                "into a number needs a decision about what the number means -- an "
                "age, an elapsed interval, a day of week -- and this version does "
                "not make it for you.",
            )
        return self._review(
            spec,
            "unsupported_column_kind",
            f"Held back because {spec.name!r} has kind {spec.column_kind.value}, "
            "which this version does not know how to encode.",
        )

    def _blocking_decision(self, spec: FeatureSpec, sentinel: str) -> FeatureDecision | None:
        """Findings that stop a feature before its role is considered."""
        config = self._config

        if spec.name in config.confirmed_id_columns:
            return self._exclude(
                spec,
                "confirmed_identifier",
                f"Excluded because {spec.name!r} was confirmed as an identifier. An "
                "identifier lets a model memorise rows rather than learn from them.",
            )
        if spec.cardinality == 0 or spec.missing_ratio >= 1.0:
            return self._exclude(
                spec,
                "no_observed_values",
                f"Excluded because {spec.name!r} is missing in every row. Nothing can "
                "be learned from it, and imputation would drop it silently at fit "
                "time without saying so.",
            )
        if spec.is_constant:
            return self._exclude(
                spec,
                "constant_feature",
                f"Excluded because {spec.name!r} holds one value in every row, so it "
                "cannot separate anything.",
            )
        if "target_leakage_exact_duplicate" in spec.review_codes:
            if config.drop_exact_target_duplicates:
                return self._exclude(
                    spec,
                    "target_leakage_exact_duplicate",
                    f"Excluded because {spec.name!r} was proved equal to the target in "
                    "every row. Training on it would measure nothing.",
                )
        if spec.infinite_count and not config.allow_infinite:
            return self._review(
                spec,
                "infinite_values",
                f"Held back because {spec.name!r} contains {spec.infinite_count} "
                "infinite value(s). No estimator here accepts one, and replacing it "
                "with a finite number would be a measurement nobody took.",
            )
        if (
            "possible_numeric_stored_as_text" in spec.review_codes
            and config.numeric_text_policy is NumericTextPolicy.REVIEW
        ):
            return self._review(
                spec,
                "possible_numeric_stored_as_text",
                f"Held back because {spec.name!r} is text that mostly parses as "
                "numbers. Converting it would turn every value that fails to parse "
                'into a missing one; set numeric_text_policy="convert" to accept that.',
            )
        if (
            spec.is_high_cardinality
            and spec.role in (FeatureRole.NOMINAL, FeatureRole.BINARY_CATEGORICAL)
            and config.high_cardinality_policy is HighCardinalityPolicy.REVIEW
        ):
            return self._review(
                spec,
                "high_cardinality",
                f"Held back because {spec.name!r} has {spec.cardinality} distinct "
                "values. One-hot encoding would add one column per value; set "
                'high_cardinality_policy="onehot" to do it anyway.',
                details={"cardinality": spec.cardinality},
            )
        return None

    # ------------------------------------------------------------------ #
    # Per-role strategies
    # ------------------------------------------------------------------ #

    def _numeric(
        self, spec: FeatureSpec, profile: PreprocessingProfile, forced: bool
    ) -> FeatureDecision:
        """Impute, and scale only when the model asks for it."""
        config = self._config
        steps: list[str] = []
        reasons: list[str] = [f"{spec.name!r} is numeric"]

        if spec.role_source == "config:numeric_text_policy":
            steps.append(STEP_NUMERIC_TEXT)
            reasons.append(
                "you set numeric_text_policy=convert, so the text is parsed into "
                "numbers first"
            )

        if profile.handles_missing_values:
            reasons.append(
                "the model consumes missing values natively, so no imputation is added"
            )
        else:
            # Imputation is added whether or not this column currently has gaps:
            # the fitted pipeline must survive a test row that does, and a step
            # present only when the training data happened to be complete would
            # fail exactly when it mattered.
            steps.append(_NUMERIC_IMPUTE_STEPS[config.numeric_imputation])
            reasons.append(f"gaps are filled by {config.numeric_imputation.value}")

        if profile.requires_scaling:
            steps.append(_SCALER_STEPS[config.numeric_scaler])
            reasons.append(
                f"the model requires scaling, so {config.numeric_scaler.value} scaling "
                "is applied"
            )
        else:
            reasons.append("the model does not require scaling, so none is applied")

        return FeatureDecision(
            feature=spec.name,
            role=FeatureRole.NUMERIC,
            action=FeatureAction.INCLUDE,
            reason_code="numeric_feature",
            reason="Included because " + "; ".join(reasons) + ".",
            steps=tuple(steps),
            requires_review=self._flagged(spec, forced),
            details={
                "requires_scaling": profile.requires_scaling,
                "handles_missing_values": profile.handles_missing_values,
                "missing_count": spec.missing_count,
                "is_near_constant": spec.is_near_constant,
            },
        )

    def _categorical(
        self, spec: FeatureSpec, profile: PreprocessingProfile, sentinel: str
    ) -> FeatureDecision:
        """One-hot, unless the analyst supplied an explicit mapping.

        A two-valued column is encoded the same way as any other label column.
        Collapsing it to a single 0/1 requires deciding which value is the one,
        and nothing in ``["M", "F"]`` says which. A column costs less than a
        silently inverted meaning.
        """
        if spec.explicit_mapping is not None:
            return FeatureDecision(
                feature=spec.name,
                role=spec.role,
                action=FeatureAction.INCLUDE,
                reason_code="explicit_mapping",
                reason=(
                    f"Included with the mapping you supplied for {spec.name!r}; no "
                    "encoding is inferred."
                ),
                steps=(STEP_EXPLICIT_MAPPING,),
                requires_review=self._flagged(spec, False),
                details={"levels": len(spec.explicit_mapping)},
            )

        impute_step, impute_reason = self._categorical_imputation(spec, sentinel)
        binary = spec.role is FeatureRole.BINARY_CATEGORICAL
        return FeatureDecision(
            feature=spec.name,
            role=spec.role,
            action=FeatureAction.INCLUDE,
            reason_code="binary_categorical" if binary else "nominal_feature",
            reason=(
                f"Included because {spec.name!r} is a "
                + ("two-valued label" if binary else "label with no inherent order")
                + f" with {spec.cardinality} distinct values; {impute_reason}, then "
                "one-hot encoding, which leaves an unseen category as all zeros "
                "rather than failing."
            ),
            steps=(impute_step, STEP_ONEHOT),
            requires_review=self._flagged(spec, False),
            details={
                "cardinality": spec.cardinality,
                "sparse_output": profile.supports_sparse_input,
                "missing_count": spec.missing_count,
            },
        )

    def _ordinal(self, spec: FeatureSpec, sentinel: str) -> FeatureDecision:
        """Encode against the order the analyst supplied, never an inferred one."""
        impute_step, impute_reason = self._categorical_imputation(
            spec, sentinel, ordinal=True
        )
        order = spec.ordinal_order or ()
        return FeatureDecision(
            feature=spec.name,
            role=FeatureRole.ORDINAL,
            action=FeatureAction.INCLUDE,
            reason_code="ordinal_feature",
            reason=(
                f"Included because you declared an order for {spec.name!r} "
                f"({' < '.join(str(v) for v in order)}); {impute_reason}, then ordinal "
                "encoding against that order."
            ),
            steps=(impute_step, STEP_ORDINAL),
            requires_review=self._flagged(spec, False),
            details={"levels": len(order), "order": [str(v) for v in order]},
        )

    def _identifier(self, spec: FeatureSpec, forced: bool) -> FeatureDecision:
        """Hold back a probable key, unless the analyst insists otherwise."""
        if forced:
            return FeatureDecision(
                feature=spec.name,
                role=FeatureRole.ID_LIKE,
                action=FeatureAction.INCLUDE,
                reason_code="identifier_force_included",
                reason=(
                    f"Included because {spec.name!r} is in force_include, even though "
                    "it was identified as a record key."
                ),
                steps=(),
                requires_review=True,
                details={"unique_values": spec.cardinality},
            )
        return self._review(
            spec,
            "possible_id_like",
            f"Held back because {spec.name!r} looks like a record identifier "
            f"({spec.cardinality} distinct values). This is a heuristic from the "
            "quality inspector; add it to force_include to use it anyway, or to "
            "confirmed_id_columns to settle the matter.",
            details={"unique_values": spec.cardinality},
        )

    def _categorical_imputation(
        self, spec: FeatureSpec, sentinel: str, ordinal: bool = False
    ) -> tuple[str, str]:
        """Which imputation a label column gets, and how to describe it.

        An ordinal column never takes the sentinel. The sentinel has no position
        in the analyst's ordering, so the encoder would either refuse it or place
        it below the lowest real level -- inventing a rank nobody declared.
        """
        if ordinal:
            return (
                STEP_MOST_FREQUENT_IMPUTE,
                "gaps are filled with the most frequent level, because a sentinel "
                "would have no position in the order you supplied",
            )
        if self._config.categorical_imputation is CategoricalImputation.CONSTANT:
            return (
                STEP_SENTINEL_IMPUTE,
                f"gaps become the sentinel {sentinel!r}, chosen so it collides with no "
                "real category",
            )
        return STEP_MOST_FREQUENT_IMPUTE, "gaps are filled with the most frequent value"

    # ------------------------------------------------------------------ #
    # Shared shapes
    # ------------------------------------------------------------------ #

    @staticmethod
    def _flagged(spec: FeatureSpec, forced: bool) -> bool:
        """Whether an included feature still deserves a human look."""
        return bool(spec.review_codes) or spec.is_near_constant or forced

    @staticmethod
    def _exclude(
        spec: FeatureSpec, code: str, reason: str, details: dict | None = None
    ) -> FeatureDecision:
        """Build an exclusion that says so out loud."""
        return FeatureDecision(
            feature=spec.name,
            role=spec.role,
            action=FeatureAction.EXCLUDE,
            reason_code=code,
            reason=reason,
            steps=(),
            requires_review=False,
            details=details or {},
        )

    @staticmethod
    def _review(
        spec: FeatureSpec, code: str, reason: str, details: dict | None = None
    ) -> FeatureDecision:
        """Build a hold-for-review that says what would settle it."""
        return FeatureDecision(
            feature=spec.name,
            role=spec.role,
            action=FeatureAction.REVIEW,
            reason_code=code,
            reason=reason,
            steps=(),
            requires_review=True,
            details=details or {},
        )


def _as_profile(capabilities: Any) -> PreprocessingProfile:
    """Accept a profile, or anything that can produce one.

    Read structurally rather than by importing the models package: preprocessing
    and models are kept apart deliberately, and they exchange exactly one thing --
    the capability profile that ``core`` defines for the purpose.
    """
    if isinstance(capabilities, PreprocessingProfile):
        return capabilities
    producer = getattr(capabilities, "preprocessing_profile", None)
    if callable(producer):
        profile = producer()
        if isinstance(profile, PreprocessingProfile):
            return profile
    raise PreprocessingError(
        "capabilities must be a PreprocessingProfile, or expose "
        f"preprocessing_profile(); got {type(capabilities).__name__}."
    )


def _require_complete_profile(
    frame: pd.DataFrame, profile: DatasetProfile, target: Hashable | None
) -> None:
    """Every feature column must have been profiled.

    A column the profile does not mention would otherwise get no specification,
    no decision, and no mention in the plan -- a silent disappearance, which is
    the one thing this layer must never do.
    """
    profiled = {column.name for column in profile.column_profiles}
    missing = [
        column for column in frame.columns if column != target and column not in profiled
    ]
    if missing:
        raise SchemaError(
            f"Column(s) {[str(c) for c in missing]} are in the frame but not in the "
            "supplied DatasetProfile. Profile the same frame you are planning for, "
            "so that every column gets a recorded decision."
        )


def _plan_label_normalisation(frame: pd.DataFrame) -> LabelNormalisation:
    """Decide whether column labels must be renamed for scikit-learn.

    scikit-learn refuses a frame whose labels are of mixed types, and reduces
    all-integer labels to positional names like ``x0``, which loses the label. In
    either case the builder renames on an internal copy and records the mapping,
    so the caller's frame is untouched and every transformed name traces back.
    """
    if isinstance(frame.columns, pd.MultiIndex):
        raise SchemaError(
            "MultiIndex column labels are not supported. Flatten them -- for "
            'example with frame.columns = ["_".join(map(str, c)) for c in '
            "frame.columns] -- before preprocessing."
        )

    labels = list(frame.columns)
    if not labels or all(isinstance(label, str) for label in labels):
        return LabelNormalisation()

    types = sorted({type(label).__name__ for label in labels})
    mapping = {label: str(label) for label in labels}
    if len(set(mapping.values())) != len(mapping):
        collisions = sorted(
            {value for value in mapping.values() if list(mapping.values()).count(value) > 1}
        )
        raise SchemaError(
            f"Column labels {collisions} are distinct but share a text form, so they "
            "cannot both be named in a scikit-learn pipeline. Rename one before "
            "preprocessing."
        )
    return LabelNormalisation(
        applied=True,
        mapping=mapping,
        reason=(
            f"Column labels have type(s) {types}; scikit-learn requires string names, "
            "so they are renamed on an internal copy. The caller's frame is unchanged."
        ),
    )
