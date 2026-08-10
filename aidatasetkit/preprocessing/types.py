"""Vocabulary for describing what preprocessing will do, and why.

Everything here is a *description*. A :class:`PreprocessingPlan` says which
column gets imputed, encoded, scaled, excluded, or held back, and on what
grounds; it holds no transformer, no fitted state, and no data. That is what lets
a plan be read, argued with, and serialised before anything touches the data.

The split runs all the way through the package:

======================  ====================================================
Object                  Answers
======================  ====================================================
:class:`FeatureSpec`    What kind of feature is this?
:class:`FeatureDecision` What should happen to it, and why?
:class:`PreprocessingPlan` The whole intent, serialisable
built preprocessor      How that intent is executed
fitted preprocessor     What was learned from the training rows
======================  ====================================================
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from aidatasetkit.core.types import ColumnKind, PreprocessingProfile, jsonable

__all__ = [
    "FeatureRole",
    "FeatureAction",
    "FeatureSpec",
    "FeatureDecision",
    "LabelNormalisation",
    "PreprocessingPlan",
    "TargetEncoding",
]


class FeatureRole(StrEnum):
    """What a column is, for the purpose of preprocessing.

    A *role* is not a dtype. ``ColumnKind`` says a column holds text; the role
    says whether that text is a label with no order (``NOMINAL``), a label with
    an analyst-supplied order (``ORDINAL``), a two-valued flag
    (``BINARY_CATEGORICAL``), or a record key (``ID_LIKE``). Only the first of
    those is decidable from the data alone.
    """

    NUMERIC = "numeric"
    BINARY_CATEGORICAL = "binary_categorical"
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    DATETIME = "datetime"
    ID_LIKE = "id_like"
    UNSUPPORTED = "unsupported"


class FeatureAction(StrEnum):
    """What the plan will do with a feature.

    ``REVIEW`` is deliberately distinct from ``EXCLUDE``. Excluding says the
    library is confident the column contributes nothing; review says the library
    is *not* confident and has stopped rather than guessed. Both keep the column
    out of the fitted pipeline, and both say so in the plan -- neither is a
    silent drop.

    A feature can also be included *and* flagged: ``INCLUDE`` with
    ``requires_review`` set means the column is used and still deserves a look,
    which is how a possible-leakage finding is handled.
    """

    INCLUDE = "include"
    EXCLUDE = "exclude"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """What kind of feature this is, and what is known about it.

    Assembled from the profiler and the quality inspector rather than measured
    again here. ``role_source`` records which precedence rule decided the role,
    so a surprising plan can be traced to the rule that produced it.
    """

    name: Hashable
    column_kind: ColumnKind
    role: FeatureRole
    role_source: str
    cardinality: int = 0
    missing_count: int = 0
    missing_ratio: float = 0.0
    is_constant: bool = False
    is_near_constant: bool = False
    is_high_cardinality: bool = False
    is_id_like: bool = False
    infinite_count: int = 0
    ordinal_order: tuple[Any, ...] | None = None
    explicit_mapping: Mapping[Any, Any] | None = None
    review_codes: tuple[str, ...] = ()

    @property
    def requires_review(self) -> bool:
        """Whether anything about this feature was flagged for a human."""
        return bool(self.review_codes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the specification."""
        return {
            "name": str(self.name),
            "column_kind": self.column_kind.value,
            "role": self.role.value,
            "role_source": self.role_source,
            "cardinality": self.cardinality,
            "missing_count": self.missing_count,
            "missing_ratio": self.missing_ratio,
            "is_constant": self.is_constant,
            "is_near_constant": self.is_near_constant,
            "is_high_cardinality": self.is_high_cardinality,
            "is_id_like": self.is_id_like,
            "infinite_count": self.infinite_count,
            "ordinal_order": (
                None
                if self.ordinal_order is None
                else [jsonable(value) for value in self.ordinal_order]
            ),
            "explicit_mapping": (
                None
                if self.explicit_mapping is None
                else {str(k): jsonable(v) for k, v in self.explicit_mapping.items()}
            ),
            "review_codes": list(self.review_codes),
        }


@dataclass(frozen=True, slots=True)
class FeatureDecision:
    """What will happen to one feature, and the reason.

    ``steps`` names the transformations in order, as they will be applied. It is
    a description, not a construction: no transformer object appears here.
    """

    feature: Hashable
    role: FeatureRole
    action: FeatureAction
    reason_code: str
    reason: str
    steps: tuple[str, ...] = ()
    requires_review: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the decision."""
        return {
            "feature": str(self.feature),
            "role": self.role.value,
            "action": self.action.value,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "steps": list(self.steps),
            "requires_review": self.requires_review,
            "details": {str(k): jsonable(v) for k, v in self.details.items()},
        }


@dataclass(frozen=True, slots=True)
class LabelNormalisation:
    """A recorded, reversible renaming of non-string column labels.

    scikit-learn refuses a frame whose labels are of mixed types, and reduces
    all-integer labels to positional names such as ``x0``, losing the label
    entirely. When either applies, the builder renames columns on an internal
    copy and records the mapping here, so a transformed feature name can always
    be traced back to the caller's own label.
    """

    applied: bool = False
    mapping: Mapping[Hashable, str] = field(default_factory=dict)
    reason: str | None = None

    def original(self, normalised: str) -> Hashable:
        """Return the caller's label for a normalised name."""
        for source, target in self.mapping.items():
            if target == normalised:
                return source
        return normalised

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the renaming."""
        return {
            "applied": self.applied,
            "mapping": {str(k): v for k, v in self.mapping.items()},
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class PreprocessingPlan:
    """Everything preprocessing intends to do, before any of it happens.

    Holds no transformer and no fitted state. Two plans built from the same
    frame, analysis, capabilities, and configuration are equal, and serialise
    identically.
    """

    decisions: tuple[FeatureDecision, ...]
    specs: tuple[FeatureSpec, ...]
    preprocessing_profile: PreprocessingProfile
    config: Mapping[str, Any] = field(default_factory=dict)
    label_normalisation: LabelNormalisation = field(default_factory=LabelNormalisation)
    target_name: str | None = None
    #: Fill value for constant categorical imputation, already checked against the
    #: training data so it collides with no real category. Recorded on the plan
    #: because the builder must use the same one the plan described.
    categorical_sentinel: str = "__missing__"

    # ------------------------------------------------------------------ #
    # Feature groups
    # ------------------------------------------------------------------ #

    def _included(self, role: FeatureRole) -> tuple[Hashable, ...]:
        return tuple(
            decision.feature
            for decision in self.decisions
            if decision.action is FeatureAction.INCLUDE and decision.role is role
        )

    @property
    def numeric_features(self) -> tuple[Hashable, ...]:
        """Columns entering the numeric pipeline, in plan order."""
        return self._included(FeatureRole.NUMERIC)

    @property
    def nominal_features(self) -> tuple[Hashable, ...]:
        """Columns entering the nominal (one-hot) pipeline."""
        return self._included(FeatureRole.NOMINAL)

    @property
    def binary_features(self) -> tuple[Hashable, ...]:
        """Two-valued label columns entering the binary pipeline."""
        return self._included(FeatureRole.BINARY_CATEGORICAL)

    @property
    def ordinal_features(self) -> tuple[Hashable, ...]:
        """Columns entering the ordinal pipeline, with analyst-supplied order."""
        return self._included(FeatureRole.ORDINAL)

    @property
    def included_features(self) -> tuple[Hashable, ...]:
        """Every column the fitted preprocessor will consume."""
        return tuple(
            decision.feature
            for decision in self.decisions
            if decision.action is FeatureAction.INCLUDE
        )

    @property
    def excluded_features(self) -> tuple[Hashable, ...]:
        """Columns deliberately left out."""
        return tuple(
            decision.feature
            for decision in self.decisions
            if decision.action is FeatureAction.EXCLUDE
        )

    @property
    def review_features(self) -> tuple[Hashable, ...]:
        """Columns held back pending an analyst decision."""
        return tuple(
            decision.feature
            for decision in self.decisions
            if decision.action is FeatureAction.REVIEW
        )

    @property
    def flagged_features(self) -> tuple[Hashable, ...]:
        """Every column carrying a review flag, included or not."""
        return tuple(
            decision.feature for decision in self.decisions if decision.requires_review
        )

    def decision_for(self, feature: Hashable) -> FeatureDecision:
        """Return the decision recorded for one column.

        Raises:
            KeyError: If the column was not planned.
        """
        for decision in self.decisions:
            if decision.feature == feature:
                return decision
        raise KeyError(f"No preprocessing decision was recorded for {feature!r}.")

    def spec_for(self, feature: Hashable) -> FeatureSpec:
        """Return the specification recorded for one column."""
        for spec in self.specs:
            if spec.name == feature:
                return spec
        raise KeyError(f"No feature specification was recorded for {feature!r}.")

    # ------------------------------------------------------------------ #
    # Identity and output
    # ------------------------------------------------------------------ #

    @property
    def fingerprint(self) -> str:
        """A stable identity for the *structure* of this plan.

        Two plans with the same fingerprint build the same transformer, so a
        blueprint can be reused between them. Reasons and review flags are
        excluded: they explain a decision without changing what gets built.

        Everything that *does* change what gets built is here -- the capability
        profile, the sentinel, each column's role and ordered steps, its ordinal
        levels, its explicit mapping, and the parameter values that the step names
        alone do not pin down (a constant fill value, an unknown-ordinal
        encoding). Two plans that would execute differently cannot collide.
        """
        import hashlib

        parts = [
            self.preprocessing_profile.key,
            self.categorical_sentinel,
            f"numeric_fill={self.config.get('numeric_fill_value')!r}",
            f"unknown_ordinal={self.config.get('unknown_ordinal_policy')!r}"
            f":{self.config.get('unknown_ordinal_value')!r}",
        ]
        for decision in self.decisions:
            if decision.action is not FeatureAction.INCLUDE:
                continue
            spec = self.spec_for(decision.feature)
            mapping = spec.explicit_mapping
            parts.append(
                "|".join(
                    (
                        str(decision.feature),
                        decision.role.value,
                        ",".join(decision.steps),
                        ""
                        if spec.ordinal_order is None
                        else ",".join(str(v) for v in spec.ordinal_order),
                        ""
                        if mapping is None
                        else ",".join(
                            f"{k!r}={v!r}" for k, v in sorted(mapping.items(), key=lambda i: str(i[0]))
                        ),
                    )
                )
            )
        digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
        return digest[:32]

    def describe(self) -> str:
        """Return a readable account of every decision, for a human to check."""
        lines: list[str] = []
        for decision in self.decisions:
            arrow = " -> ".join(decision.steps) if decision.steps else decision.action.value
            flag = "  [requires review]" if decision.requires_review else ""
            lines.append(f"{decision.feature!s}\n  {arrow}\n  Reason: {decision.reason}{flag}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the whole plan.

        Contains no transformer, no dataframe, and no fitted state.
        """
        return {
            "preprocessing_profile": self.preprocessing_profile.to_dict(),
            "fingerprint": self.fingerprint,
            "target_name": self.target_name,
            "numeric_features": [str(c) for c in self.numeric_features],
            "binary_features": [str(c) for c in self.binary_features],
            "nominal_features": [str(c) for c in self.nominal_features],
            "ordinal_features": [str(c) for c in self.ordinal_features],
            "excluded_features": [str(c) for c in self.excluded_features],
            "review_features": [str(c) for c in self.review_features],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "specs": [spec.to_dict() for spec in self.specs],
            "categorical_sentinel": self.categorical_sentinel,
            "label_normalisation": self.label_normalisation.to_dict(),
            "config": {str(k): jsonable(v) for k, v in self.config.items()},
        }


@dataclass(frozen=True, slots=True)
class TargetEncoding:
    """How a classification target's labels map onto integers, both ways.

    The positive label travels with the mapping on purpose. Once labels become
    integers, ``predict_proba[:, 1]`` is the probability of whichever class
    sorted second -- which is not necessarily the event the analyst cares about.
    Recording both the original and the encoded positive label is what stops a
    later metric from being computed for the wrong class.
    """

    classes: tuple[Any, ...]
    mapping: Mapping[Any, int]
    inverse: Mapping[int, Any]
    positive_label: Any = None
    positive_label_encoded: int | None = None
    positive_label_resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the encoding."""
        return {
            "classes": [jsonable(value) for value in self.classes],
            "mapping": {str(k): int(v) for k, v in self.mapping.items()},
            "inverse": {str(k): jsonable(v) for k, v in self.inverse.items()},
            "positive_label": jsonable(self.positive_label),
            "positive_label_encoded": self.positive_label_encoded,
            "positive_label_resolved": self.positive_label_resolved,
        }


def as_tuple(value: Sequence[Hashable] | Hashable | None) -> tuple[Hashable, ...]:
    """Coerce a column argument into a tuple without splitting a string label."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return (value,)
    return tuple(value)
