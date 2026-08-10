"""Deciding which charts are worth considering.

Rules, not learning. Every recommendation here follows from measurements the
profiler and the quality inspector already took, and each one records the
evidence it rested on.

**The bound that matters.** A hundred numeric columns make 4,950 pairs. This
module never builds that list: pairwise candidates are generated *from a
bounded shortlist*, chosen from profile metadata before any data is touched.

    per-column candidates        O(m)              m = columns
    target association           O(k · n)          k = max_correlation_columns
    feature-to-feature pairs     O(p² · n)         p = max_pairwise_columns
    dataset-level candidates     O(1) decisions

With the defaults (``k = 50``, ``p = 8``) the pairwise term is at most 28
correlations regardless of how wide the frame is. Nothing is generated in order to
be thrown away.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Iterator

import pandas as pd

from aidatasetkit.core.exceptions import AIDatasetKitError
from aidatasetkit.core.types import ColumnKind, ColumnProfile, TaskType
from aidatasetkit.statistics.bivariate import correlation
from aidatasetkit.visualization.context import VisualizationContext
from aidatasetkit.visualization.types import (
    ChartSpec,
    ChartType,
    SuppressedCandidate,
    VisualizationReason,
    VisualizationWarning,
)

__all__ = ["CandidateGenerator", "CandidateBatch"]

_logger = logging.getLogger(__name__)

#: Column kinds that a bar chart can describe.
_LABEL_KINDS = (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN)


class CandidateBatch:
    """Candidates generated, plus the ones ruled out and why."""

    def __init__(self) -> None:
        self.specs: list[ChartSpec] = []
        self.suppressed: list[SuppressedCandidate] = []
        #: Measured target associations, reused later by the ranking engine.
        self.associations: dict[Hashable, float] = {}

    def add(self, spec: ChartSpec) -> None:
        """Record a candidate."""
        self.specs.append(spec)

    def drop(
        self, chart_type: ChartType, columns: tuple[Hashable, ...], reason: VisualizationReason
    ) -> None:
        """Record that a candidate was considered and ruled out."""
        self.suppressed.append(SuppressedCandidate(chart_type, columns, reason))

    def __len__(self) -> int:
        return len(self.specs)


class CandidateGenerator:
    """Produces the charts worth considering for one dataset."""

    def generate(self, context: VisualizationContext) -> CandidateBatch:
        """Return every candidate chart for ``context``.

        Order is deterministic: dataset-level charts, then the target, then
        per-column charts in frame order, then bounded relationship charts.
        """
        batch = CandidateBatch()

        self._dataset_level(context, batch)
        self._target(context, batch)
        for profile in context.feature_profiles:
            self._per_column(context, profile, batch)
        self._relationships(context, batch)

        _logger.debug(
            "generated %d visualization candidates, suppressed %d",
            len(batch.specs),
            len(batch.suppressed),
        )
        return batch

    # ------------------------------------------------------------------ #
    # Dataset-level
    # ------------------------------------------------------------------ #

    def _dataset_level(self, context: VisualizationContext, batch: CandidateBatch) -> None:
        """Charts about the frame as a whole."""
        missing = context.profile.total_missing_count
        if missing:
            columns = tuple(
                profile.name for profile in context.column_profiles if profile.missing_count
            )
            batch.add(
                ChartSpec(
                    chart_type=ChartType.MISSING_VALUES,
                    columns=columns,
                    title="Missing values by column",
                    reason=VisualizationReason(
                        code="dataset_has_missing_values",
                        message=(
                            f"Selected because {len(columns)} column(s) contain missing "
                            f"values, {context.profile.total_missing_ratio:.1%} of all cells."
                        ),
                        evidence={
                            "total_missing_count": missing,
                            "total_missing_ratio": context.profile.total_missing_ratio,
                            "columns_with_missing": len(columns),
                        },
                    ),
                )
            )
        else:
            batch.drop(
                ChartType.MISSING_VALUES,
                (),
                VisualizationReason(
                    code="no_missing_values",
                    message="Suppressed because the dataset has no missing values.",
                    evidence={"total_missing_count": 0},
                ),
            )

        self._heatmap(context, batch)

    def _heatmap(self, context: VisualizationContext, batch: CandidateBatch) -> None:
        """A correlation heatmap, when the numeric width makes one readable."""
        eligible = list(self._heatmap_eligible(context))
        config = context.config

        if len(eligible) < config.min_heatmap_columns:
            batch.drop(
                ChartType.CORRELATION_HEATMAP,
                tuple(profile.name for profile in eligible),
                VisualizationReason(
                    code="too_few_numeric_columns",
                    message=(
                        f"Suppressed because only {len(eligible)} numeric column(s) "
                        f"qualify; a matrix needs at least {config.min_heatmap_columns} "
                        "to say more than the individual charts already do."
                    ),
                    evidence={
                        "eligible_columns": len(eligible),
                        "minimum": config.min_heatmap_columns,
                    },
                ),
            )
            return

        warnings: tuple[VisualizationWarning, ...] = ()
        selected = eligible
        if len(eligible) > config.max_heatmap_columns:
            selected = self._shortlist(eligible, config.max_heatmap_columns)
            warnings = (
                VisualizationWarning(
                    code="heatmap_columns_truncated",
                    message=(
                        f"{len(eligible)} numeric columns qualify; the "
                        f"{config.max_heatmap_columns} with the most distinct values "
                        "and fewest missing values are shown."
                    ),
                ),
            )

        names = tuple(profile.name for profile in selected)
        batch.add(
            ChartSpec(
                chart_type=ChartType.CORRELATION_HEATMAP,
                columns=names,
                title=f"Correlation between {len(names)} numeric columns",
                reason=VisualizationReason(
                    code="numeric_correlation_overview",
                    message=(
                        f"Selected because {len(names)} numeric columns can be compared "
                        "pairwise in one readable matrix."
                    ),
                    evidence={
                        "eligible_columns": len(eligible),
                        "included_columns": [str(name) for name in names],
                    },
                ),
                warnings=warnings,
            )
        )

    # ------------------------------------------------------------------ #
    # Target
    # ------------------------------------------------------------------ #

    def _target(self, context: VisualizationContext, batch: CandidateBatch) -> None:
        """The target's own distribution."""
        target_profile = context.target_profile
        if target_profile is None or not context.has_target:
            return

        target = context.target
        imbalance = context.issue(target, "class_imbalance")
        evidence: dict[str, object] = {"task_type": target_profile.task_type.value}

        if target_profile.task_type is TaskType.CLASSIFICATION:
            evidence["n_classes"] = target_profile.n_classes
            message = (
                f"Selected because the target has {target_profile.n_classes} classes "
                "and their balance determines how every later score must be read."
            )
            if imbalance is not None:
                evidence["imbalance_ratio"] = target_profile.imbalance_ratio
                message = (
                    "Selected because the quality inspector reported an imbalanced "
                    f"target: the rarest of {target_profile.n_classes} classes covers "
                    f"{imbalance.details.get('minority_ratio', 0.0):.1%} of rows."
                )
        else:
            message = (
                "Selected because the shape of a regression target decides which "
                "error measures are meaningful."
            )

        batch.add(
            ChartSpec(
                chart_type=ChartType.TARGET_DISTRIBUTION,
                columns=(target,),
                target=target,
                title=f"Distribution of target {target!s}",
                reason=VisualizationReason(
                    code="target_distribution",
                    message=message,
                    evidence=evidence,
                ),
            )
        )

    # ------------------------------------------------------------------ #
    # Per column
    # ------------------------------------------------------------------ #

    def _per_column(
        self,
        context: VisualizationContext,
        profile: ColumnProfile,
        batch: CandidateBatch,
    ) -> None:
        """Univariate charts for one feature."""
        blocked = self._blocking_reason(context, profile)
        if blocked is not None:
            batch.drop(self._natural_chart(profile), (profile.name,), blocked)
            return

        if profile.detected_kind is ColumnKind.NUMERIC:
            self._numeric_column(context, profile, batch)
        elif profile.detected_kind in _LABEL_KINDS:
            self._label_column(context, profile, batch)

    def _blocking_reason(
        self, context: VisualizationContext, profile: ColumnProfile
    ) -> VisualizationReason | None:
        """Why this column should get no automatic chart at all, if so.

        Identifier suppression is the reason a unique integer key never becomes a
        histogram just because its dtype is numeric.
        """
        if profile.is_constant:
            return VisualizationReason(
                code="constant_column",
                message=(
                    f"Suppressed because {profile.name!r} holds the single value "
                    f"{profile.dominant_value!r}; a chart of it would be one bar."
                ),
                evidence={"is_constant": True, "unique_count": profile.unique_count},
            )
        if profile.is_id_like:
            return VisualizationReason(
                code="possible_id_like",
                message=(
                    f"Suppressed because {profile.name!r} looks like a record "
                    f"identifier ({profile.unique_ratio:.1%} distinct). Plotting an "
                    "identifier shows the row numbering, not the data."
                ),
                evidence={
                    "is_id_like": True,
                    "unique_ratio": profile.unique_ratio,
                    "detected_kind": profile.detected_kind.value,
                },
                requires_review=True,
            )
        if profile.count < context.config.min_rows_for_distribution:
            return VisualizationReason(
                code="too_few_observations",
                message=(
                    f"Suppressed because {profile.name!r} has only {profile.count} "
                    "non-missing value(s), too few for a distribution to mean anything."
                ),
                evidence={
                    "count": profile.count,
                    "minimum": context.config.min_rows_for_distribution,
                },
            )
        if profile.detected_kind in (ColumnKind.DATETIME, ColumnKind.OTHER):
            return VisualizationReason(
                code="unsupported_column_kind",
                message=(
                    f"Suppressed because {profile.name!r} is of kind "
                    f"{profile.detected_kind.value}, which this version does not chart."
                ),
                evidence={"detected_kind": profile.detected_kind.value},
            )
        return None

    @staticmethod
    def _natural_chart(profile: ColumnProfile) -> ChartType:
        """The chart this column would have received had it not been suppressed."""
        if profile.detected_kind is ColumnKind.NUMERIC:
            return ChartType.HISTOGRAM
        return ChartType.BAR

    def _numeric_column(
        self,
        context: VisualizationContext,
        profile: ColumnProfile,
        batch: CandidateBatch,
    ) -> None:
        """Histogram always; box plot when spread or flagged outliers justify it."""
        evidence = {
            "detected_kind": profile.detected_kind.value,
            "unique_count": profile.unique_count,
            "is_constant": profile.is_constant,
            "is_id_like": profile.is_id_like,
        }
        near_constant = profile.is_near_constant
        message = (
            f"Selected because {profile.name!r} is numeric with "
            f"{profile.unique_count} distinct values, enough variation for a "
            "distribution view."
        )
        if near_constant:
            evidence["dominant_ratio"] = profile.dominant_ratio
            message = (
                f"Selected with low priority: {profile.name!r} is numeric but one "
                f"value covers {profile.dominant_ratio:.1%} of rows, so the "
                "distribution is nearly degenerate."
            )

        batch.add(
            ChartSpec(
                chart_type=ChartType.HISTOGRAM,
                columns=(profile.name,),
                title=f"Distribution of {profile.name!s}",
                reason=VisualizationReason(
                    code="numeric_distribution", message=message, evidence=evidence
                ),
            )
        )

        outlier_issue = context.issue(profile.name, "possible_outliers")
        summary = profile.numeric
        has_spread = (
            summary is not None
            and summary.iqr is not None
            and summary.iqr > 0.0
            and not near_constant
        )

        if outlier_issue is not None:
            batch.add(
                ChartSpec(
                    chart_type=ChartType.BOX_PLOT,
                    columns=(profile.name,),
                    title=f"Spread and outliers of {profile.name!s}",
                    reason=VisualizationReason(
                        code="flagged_outlier_candidates",
                        message=(
                            "Selected because the quality inspector flagged "
                            f"{outlier_issue.details.get('outlier_count', 0)} value(s) "
                            "outside the Tukey fences, which a box plot shows directly."
                        ),
                        evidence={
                            "outlier_count": outlier_issue.details.get("outlier_count"),
                            "outlier_ratio": outlier_issue.details.get("outlier_ratio"),
                            "lower_bound": outlier_issue.details.get("lower_bound"),
                            "upper_bound": outlier_issue.details.get("upper_bound"),
                        },
                        requires_review=True,
                    ),
                )
            )
        elif has_spread:
            batch.add(
                ChartSpec(
                    chart_type=ChartType.BOX_PLOT,
                    columns=(profile.name,),
                    title=f"Spread of {profile.name!s}",
                    reason=VisualizationReason(
                        code="numeric_spread",
                        message=(
                            f"Selected because {profile.name!r} has a non-zero "
                            "interquartile range, so quartiles and whiskers are "
                            "informative."
                        ),
                        evidence={"iqr": summary.iqr, "q25": summary.q25, "q75": summary.q75},
                    ),
                )
            )
        else:
            batch.drop(
                ChartType.BOX_PLOT,
                (profile.name,),
                VisualizationReason(
                    code="no_meaningful_spread",
                    message=(
                        f"Suppressed because {profile.name!r} has no interquartile "
                        "spread to summarise."
                    ),
                    evidence={"iqr": None if summary is None else summary.iqr},
                ),
            )

    def _label_column(
        self,
        context: VisualizationContext,
        profile: ColumnProfile,
        batch: CandidateBatch,
    ) -> None:
        """A bar chart of category frequencies, truncated when necessary."""
        config = context.config
        warnings: tuple[VisualizationWarning, ...] = ()
        evidence = {
            "detected_kind": profile.detected_kind.value,
            "unique_count": profile.unique_count,
            "is_high_cardinality": profile.is_high_cardinality,
        }

        if profile.unique_count > config.max_categories:
            warnings = (
                VisualizationWarning(
                    code="categories_truncated",
                    message=(
                        f"{profile.name!s} has {profile.unique_count} categories; the "
                        f"{config.max_categories} most frequent are shown and the rest "
                        "are grouped as 'Other'."
                    ),
                ),
            )
            message = (
                f"Selected because {profile.name!r} is categorical with "
                f"{profile.unique_count} distinct values; the most frequent "
                f"{config.max_categories} are shown."
            )
        else:
            message = (
                f"Selected because {profile.name!r} is categorical with "
                f"{profile.unique_count} distinct values, few enough to read as bars."
            )

        batch.add(
            ChartSpec(
                chart_type=ChartType.BAR,
                columns=(profile.name,),
                title=f"Frequency of {profile.name!s}",
                reason=VisualizationReason(
                    code="categorical_frequency", message=message, evidence=evidence
                ),
                warnings=warnings,
            )
        )

    # ------------------------------------------------------------------ #
    # Relationships
    # ------------------------------------------------------------------ #

    def _relationships(self, context: VisualizationContext, batch: CandidateBatch) -> None:
        """Feature-to-target and a bounded number of feature-to-feature charts."""
        if context.has_target:
            self._target_relationships(context, batch)
        self._feature_pairs(context, batch)

    def _target_relationships(
        self, context: VisualizationContext, batch: CandidateBatch
    ) -> None:
        """One chart per shortlisted feature, against the target."""
        target_profile = context.target_profile
        if target_profile is None:
            return
        target = context.target
        classification = target_profile.task_type is TaskType.CLASSIFICATION

        numeric = self._shortlist(
            [p for p in context.feature_profiles if self._chartable_numeric(context, p)],
            context.config.max_correlation_columns,
        )
        associations = self._measure_associations(context, numeric, classification)
        batch.associations.update(associations)

        ranked_numeric = sorted(
            numeric,
            key=lambda p: (-associations.get(p.name, 0.0), str(p.name)),
        )[: context.config.max_pairwise_columns]

        for profile in ranked_numeric:
            batch.add(
                self._numeric_vs_target(context, profile, associations, classification)
            )

        # Ranked before slicing: taking the first N in frame order would drop a
        # leakage-flagged feature purely because it sits to the right of others.
        eligible_labels = [
            p
            for p in context.feature_profiles
            if p.detected_kind in _LABEL_KINDS
            and self._blocking_reason(context, p) is None
            and p.unique_count <= context.config.max_categories
        ]
        labels = sorted(
            eligible_labels,
            key=lambda p: (
                0 if context.issue(p.name, "possible_target_leakage") else 1,
                p.missing_ratio,
                str(p.name),
            ),
        )[: context.config.max_pairwise_columns]

        for profile in labels:
            chart_type = ChartType.GROUPED_BAR if classification else ChartType.GROUPED_BOX
            batch.add(
                ChartSpec(
                    chart_type=chart_type,
                    columns=(profile.name, target),
                    target=target,
                    title=f"{profile.name!s} against {target!s}",
                    reason=self._relationship_reason(
                        context, profile, "categorical_against_target", classification
                    ),
                )
            )

    def _numeric_vs_target(
        self,
        context: VisualizationContext,
        profile: ColumnProfile,
        associations: dict[Hashable, float],
        classification: bool,
    ) -> ChartSpec:
        """Build the chart comparing one numeric feature with the target."""
        target = context.target
        chart_type = ChartType.GROUPED_BOX if classification else ChartType.SCATTER
        strength = associations.get(profile.name)

        code = "numeric_against_target"
        evidence: dict[str, object] = {"detected_kind": profile.detected_kind.value}
        if strength is not None:
            evidence["abs_correlation"] = strength
            evidence["correlation_method"] = "pearson"
            message = (
                f"Selected because the measured Pearson correlation between "
                f"{profile.name!r} and {target!r} is {strength:.2f} in absolute value."
            )
        else:
            message = (
                f"Selected for comparison of {profile.name!r} against the "
                f"{'classification' if classification else 'regression'} target."
            )

        leakage = context.issue(profile.name, "possible_target_leakage")
        requires_review = False
        if leakage is not None:
            code = "possible_target_leakage_review"
            requires_review = True
            evidence["leakage_signals"] = list(leakage.details.get("signals", ()))
            message = (
                f"Selected for review because the quality inspector flagged a possible "
                f"target-leakage relationship between {profile.name!r} and {target!r}. "
                "The chart shows the relationship; it does not establish leakage."
            )

        return ChartSpec(
            chart_type=chart_type,
            columns=(profile.name, target),
            target=target,
            title=f"{profile.name!s} against {target!s}",
            reason=VisualizationReason(
                code=code,
                message=message,
                evidence=evidence,
                requires_review=requires_review,
            ),
        )

    def _relationship_reason(
        self,
        context: VisualizationContext,
        profile: ColumnProfile,
        code: str,
        classification: bool,
    ) -> VisualizationReason:
        """Reason text for a categorical feature compared with the target."""
        leakage = context.issue(profile.name, "possible_target_leakage")
        if leakage is not None:
            return VisualizationReason(
                code="possible_target_leakage_review",
                message=(
                    f"Selected for review because the quality inspector flagged a "
                    f"possible target-leakage relationship for {profile.name!r}. The "
                    "chart shows the relationship; it does not establish leakage."
                ),
                evidence={"leakage_signals": list(leakage.details.get("signals", ()))},
                requires_review=True,
            )
        return VisualizationReason(
            code=code,
            message=(
                f"Selected for comparison of {profile.name!r} against the "
                f"{'classification' if classification else 'regression'} target."
            ),
            evidence={
                "detected_kind": profile.detected_kind.value,
                "unique_count": profile.unique_count,
            },
        )

    def _feature_pairs(self, context: VisualizationContext, batch: CandidateBatch) -> None:
        """A few scatter charts between numeric features, never the full product.

        The shortlist is capped at ``max_pairwise_columns`` *before* any pair is
        formed, so the number of correlations computed is bounded by that cap
        squared no matter how wide the frame is.
        """
        config = context.config
        if config.max_pairwise_charts == 0:
            return

        eligible = [
            p
            for p in context.feature_profiles
            if self._chartable_numeric(context, p) and not p.is_near_constant
        ]
        shortlist = self._shortlist(eligible, config.max_pairwise_columns)
        if len(shortlist) < 2:
            return

        scored: list[tuple[float, ColumnProfile, ColumnProfile]] = []
        for index, first in enumerate(shortlist):
            for second in shortlist[index + 1 :]:
                strength = self._safe_correlation(
                    context.frame[first.name], context.frame[second.name]
                )
                if strength is not None:
                    scored.append((strength, first, second))

        scored.sort(key=lambda item: (-item[0], str(item[1].name), str(item[2].name)))
        for rank, (strength, first, second) in enumerate(
            scored[: config.max_pairwise_charts], start=1
        ):
            batch.add(
                ChartSpec(
                    chart_type=ChartType.SCATTER,
                    columns=(first.name, second.name),
                    title=f"{first.name!s} against {second.name!s}",
                    reason=VisualizationReason(
                        code="numeric_pair_association",
                        message=(
                            f"Selected because the measured Pearson correlation between "
                            f"{first.name!r} and {second.name!r} is {strength:.2f} in "
                            f"absolute value, ranked {rank} of {len(scored)} pairs among "
                            "the shortlisted numeric columns."
                        ),
                        evidence={
                            "abs_correlation": strength,
                            "correlation_method": "pearson",
                            "pair_rank": rank,
                            "pairs_considered": len(scored),
                            "shortlisted_columns": len(shortlist),
                            "eligible_columns": len(eligible),
                        },
                    ),
                )
            )

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    def _heatmap_eligible(self, context: VisualizationContext) -> Iterator[ColumnProfile]:
        """Numeric columns worth putting in a correlation matrix."""
        for profile in context.column_profiles:
            if context.is_target(profile.name):
                continue
            if self._chartable_numeric(context, profile) and not profile.is_near_constant:
                yield profile

    def _chartable_numeric(
        self, context: VisualizationContext, profile: ColumnProfile
    ) -> bool:
        """Whether a numeric column may take part in an automatic chart."""
        return (
            profile.detected_kind is ColumnKind.NUMERIC
            and self._blocking_reason(context, profile) is None
        )

    @staticmethod
    def _shortlist(profiles: list[ColumnProfile], limit: int) -> list[ColumnProfile]:
        """Pick ``limit`` columns using metadata alone, deterministically.

        Preference goes to columns with more distinct values and fewer missing
        ones, since those carry the most to look at. Distinct-value count is a
        proxy for how much there is to see, not a measure of variance, and the
        wording of anything derived from this ranking says so. The tie-break on
        the string label keeps the choice stable across runs.
        """
        if len(profiles) <= limit:
            return list(profiles)
        return sorted(
            profiles,
            key=lambda p: (-p.unique_count, p.missing_ratio, str(p.name)),
        )[:limit]

    def _measure_associations(
        self,
        context: VisualizationContext,
        profiles: list[ColumnProfile],
        classification: bool,
    ) -> dict[Hashable, float]:
        """Measure each shortlisted feature's correlation with the target.

        Delegated to :func:`aidatasetkit.statistics.bivariate.correlation`; no
        correlation formula is written in this package. A pair that is
        mathematically undefined -- a constant column, no complete pairs -- simply
        contributes no measurement rather than a fabricated zero.

        Nothing is measured against a *multiclass* target. Its integer codes are
        nominal: class 2 is not twice class 1, so a Pearson coefficient against
        them is a number about the encoding rather than about the data. Reporting
        one as evidence would be worse than reporting none.
        """
        target_series = context.frame[context.target]
        if classification and not self._target_is_orderable(context, target_series):
            return {}

        measured: dict[Hashable, float] = {}
        for profile in profiles:
            strength = self._safe_correlation(context.frame[profile.name], target_series)
            if strength is not None:
                measured[profile.name] = strength
        return measured

    @staticmethod
    def _target_is_orderable(
        context: VisualizationContext, target_series: pd.Series
    ) -> bool:
        """Whether a correlation against this target would mean anything.

        Only a binary target qualifies among classification targets: with two
        classes the coefficient measures separation, and its sign is the only
        thing the arbitrary coding affects. Three or more nominal codes make it
        meaningless.
        """
        if not _is_numeric_series(target_series):
            return False
        profile = context.target_profile
        return profile is not None and bool(profile.is_binary)

    @staticmethod
    def _safe_correlation(left: pd.Series, right: pd.Series) -> float | None:
        """Absolute Pearson correlation, or ``None`` when it is undefined."""
        try:
            return abs(
                correlation(
                    left.astype("float64"),
                    right.astype("float64"),
                    nan_policy="omit",
                    allow_inf=False,
                )
            )
        except (AIDatasetKitError, TypeError, ValueError):
            return None


def _is_numeric_series(series: pd.Series) -> bool:
    """Whether a series holds numbers a correlation could be measured against."""
    return bool(pd.api.types.is_numeric_dtype(series))
