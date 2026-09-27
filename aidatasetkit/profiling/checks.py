"""Independent data-quality checks.

Every check is a plain function with the same signature::

    check(frame, context) -> list[QualityIssue]

That keeps them separate, individually testable, and easy to extend: a new check
is a new function added to :data:`DEFAULT_CHECKS`, not a new branch in a growing
conditional.

None of them modifies the frame. A check reports what it saw, why it is worth a
look, and what the analyst might do about it. Acting on the finding is the
analyst's decision, in every case without exception.

Findings that rest on a heuristic carry ``requires_review=True`` and a code
prefixed ``possible_``. Only two findings are stated as certain: an infinity in a
numeric column, and a feature that is exactly equal to the target.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable

import numpy as np
import pandas as pd

from aidatasetkit.core.types import ColumnKind, QualityIssue, Severity, TaskType
from aidatasetkit.profiling.context import QualityContext
from aidatasetkit.statistics.bivariate import correlation

__all__ = [
    "Check",
    "DEFAULT_CHECKS",
    "check_missing_values",
    "check_duplicate_rows",
    "check_constant_columns",
    "check_near_constant_columns",
    "check_high_cardinality",
    "check_id_like_columns",
    "check_numeric_stored_as_text",
    "check_infinite_values",
    "check_outliers",
    "check_multicollinearity",
    "check_class_imbalance",
    "check_target_leakage",
]

#: The contract every quality check follows.
Check = Callable[[pd.DataFrame, QualityContext], list[QualityIssue]]

#: Names suggesting a value recorded *after* the outcome it would predict.
#:
#: Bare ``score`` is deliberately absent: ``credit_score`` and ``exam_score`` are
#: ordinary pre-outcome features, and flagging every one of them would train the
#: analyst to ignore this check.
_POST_OUTCOME_PATTERN = re.compile(
    r"(^|[^a-z])(outcome|result|resolution|final|closed|cancel(?:led|lation)?|"
    r"churn(?:ed)?|converted|conversion|repaid|refund(?:ed)?|settled|"
    r"label|target|ground_?truth|probability|prediction|predicted)([^a-z]|$)",
    re.IGNORECASE,
)

#: Suffixes suggesting a timestamp attached to an event that may follow the outcome.
_EVENT_TIME_PATTERN = re.compile(r"(_at|_date|_time|_timestamp|_ts)$", re.IGNORECASE)

#: A text value shaped like a code rather than a quantity, such as a ZIP code.
_LEADING_ZERO_PATTERN = re.compile(r"^[+-]?0\d")

#: Values inspected when testing whether text is really numeric.
_TEXT_SAMPLE_SIZE = 1000

#: Floor on a regression's unexplained variance when turning it into a variance
#: inflation factor. An exact linear dependency drives the unexplained share to
#: zero and the VIF to a mathematical infinity, which ``json.dumps`` would emit
#: as the non-standard ``Infinity`` token. The floor keeps every VIF finite --
#: and, at ``1 / 1e-10``, unmistakably large -- so the artifact stays strict JSON.
_MIN_UNEXPLAINED_VARIANCE = 1e-10

#: Most numeric columns the variance-inflation check will fit. The work is one
#: least-squares solve per column, each over every other column, so it grows with
#: the cube of the column count. Past this limit the check declines and says so
#: rather than run for an unbounded time on a wide frame.
_VIF_MAX_COLUMNS = 50

#: Most rows the variance-inflation check fits on. Beyond it an evenly spaced,
#: deterministic subset is used and the finding records how many rows that was.
#: Evenly spaced rather than the first rows: a frame sorted by time or by any
#: column would otherwise be judged on one end of itself.
_VIF_MAX_ROWS = 20_000


# --------------------------------------------------------------------------- #
# A. Missing values
# --------------------------------------------------------------------------- #


def check_missing_values(frame: pd.DataFrame, context: QualityContext) -> list[QualityIssue]:
    """Report columns that contain missing values. Nothing is filled."""
    issues: list[QualityIssue] = []
    threshold = context.config.missing_warning_threshold

    for profile in context.profile.column_profiles:
        if not profile.missing_count:
            continue
        severe = profile.missing_ratio > threshold
        issues.append(
            QualityIssue(
                code="missing_values",
                severity=Severity.WARNING if severe else Severity.INFO,
                column=profile.name,
                message=(
                    f"Column {profile.name!r} has {profile.missing_count} missing "
                    f"value(s) ({profile.missing_ratio:.1%} of rows)."
                ),
                details={
                    "missing_count": profile.missing_count,
                    "missing_ratio": profile.missing_ratio,
                    "threshold": threshold,
                },
                recommendation=(
                    "Decide on an imputation strategy, or confirm the values are "
                    "missing by design before modelling."
                ),
            )
        )
    return issues


# --------------------------------------------------------------------------- #
# B. Duplicate rows
# --------------------------------------------------------------------------- #


def check_duplicate_rows(frame: pd.DataFrame, context: QualityContext) -> list[QualityIssue]:
    """Report repeated rows. Nothing is dropped."""
    count = context.profile.duplicate_row_count
    if not count:
        return []

    return [
        QualityIssue(
            code="duplicate_rows",
            severity=Severity.WARNING,
            column=None,
            message=(
                f"{count} row(s) repeat an earlier row "
                f"({context.profile.duplicate_row_ratio:.1%} of the dataset)."
            ),
            details={
                "duplicate_row_count": count,
                "duplicate_row_ratio": context.profile.duplicate_row_ratio,
            },
            recommendation=(
                "Confirm whether the repetition is genuine or an artefact of a "
                "join before deciding to keep or remove the rows."
            ),
        )
    ]


# --------------------------------------------------------------------------- #
# C and D. Constant and near-constant columns
# --------------------------------------------------------------------------- #


def check_constant_columns(frame: pd.DataFrame, context: QualityContext) -> list[QualityIssue]:
    """Report columns holding a single value. Nothing is removed."""
    return [
        QualityIssue(
            code="constant_column",
            severity=Severity.WARNING,
            column=profile.name,
            message=(
                f"Column {profile.name!r} holds the single value "
                f"{profile.dominant_value!r} in every non-missing row, so it "
                "carries no information."
            ),
            details={
                "value": profile.dominant_value,
                "count": profile.count,
            },
            recommendation=(
                "A constant feature cannot contribute to a model. Confirm it is "
                "not the result of a filtering mistake upstream."
            ),
        )
        for profile in context.profile.column_profiles
        if profile.is_constant
    ]


def check_near_constant_columns(
    frame: pd.DataFrame, context: QualityContext
) -> list[QualityIssue]:
    """Report columns dominated by one value, using the configured threshold."""
    threshold = context.config.near_constant_threshold
    return [
        QualityIssue(
            code="near_constant_column",
            severity=Severity.WARNING,
            column=profile.name,
            message=(
                f"Column {profile.name!r} holds the value {profile.dominant_value!r} "
                f"in {profile.dominant_ratio:.1%} of non-missing rows, at or above "
                f"the configured {threshold:.0%} threshold."
            ),
            details={
                "dominant_value": profile.dominant_value,
                "dominant_ratio": profile.dominant_ratio,
                "threshold": threshold,
                "unique_count": profile.unique_count,
            },
            recommendation=(
                "Very little variation is left to learn from. Check whether the "
                "rare values are meaningful or are data-entry noise."
            ),
        )
        for profile in context.profile.column_profiles
        if profile.is_near_constant
    ]


# --------------------------------------------------------------------------- #
# E. High-cardinality categorical columns
# --------------------------------------------------------------------------- #


def check_high_cardinality(
    frame: pd.DataFrame, context: QualityContext
) -> list[QualityIssue]:
    """Report label columns with many distinct values.

    This is a statement about encoding cost, not about identity. A column can be
    high-cardinality and still be a genuine feature; whether it also looks like an
    identifier is a separate finding from :func:`check_id_like_columns`.
    """
    threshold = context.config.high_cardinality_threshold
    return [
        QualityIssue(
            code="high_cardinality",
            severity=Severity.WARNING,
            column=profile.name,
            message=(
                f"Column {profile.name!r} has {profile.unique_count} distinct "
                f"values, above the configured threshold of {threshold}."
            ),
            details={
                "unique_count": profile.unique_count,
                "unique_ratio": profile.unique_ratio,
                "threshold": threshold,
                "also_id_like": profile.is_id_like,
            },
            recommendation=(
                "One-hot encoding would add one column per distinct value. "
                "Consider grouping rare categories or a different encoding."
            ),
        )
        for profile in context.profile.column_profiles
        if profile.is_high_cardinality and not context.is_reserved(profile.name)
    ]


# --------------------------------------------------------------------------- #
# F. Identifier-like columns
# --------------------------------------------------------------------------- #


def check_id_like_columns(
    frame: pd.DataFrame, context: QualityContext
) -> list[QualityIssue]:
    """Report columns that behave like record identifiers.

    Heuristic. The finding names the evidence it rests on and always asks for
    review; it never asserts that the column is an identifier.
    """
    issues: list[QualityIssue] = []

    for profile in context.profile.column_profiles:
        if not profile.is_id_like or context.is_reserved(profile.name):
            continue
        issues.append(
            QualityIssue(
                code="possible_id_like",
                severity=Severity.WARNING,
                column=profile.name,
                requires_review=True,
                message=(
                    f"Column {profile.name!r} may be a record identifier: "
                    f"{profile.unique_ratio:.1%} of its values are distinct and its "
                    "name or value structure resembles an identifier. This is a "
                    "heuristic and needs review."
                ),
                details={
                    "unique_count": profile.unique_count,
                    "unique_ratio": profile.unique_ratio,
                    "threshold": context.config.id_uniqueness_threshold,
                    "detected_kind": profile.detected_kind.value,
                },
                recommendation=(
                    "An identifier used as a feature lets a model memorise rows. "
                    "If this is an identifier, declare it as the id column rather "
                    "than a feature."
                ),
            )
        )
    return issues


# --------------------------------------------------------------------------- #
# G. Numeric values stored as text
# --------------------------------------------------------------------------- #


def check_numeric_stored_as_text(
    frame: pd.DataFrame, context: QualityContext
) -> list[QualityIssue]:
    """Report text columns whose values are mostly numbers.

    Fires when at least ``config.numeric_text_ratio_threshold`` of the non-missing
    values parse as numbers. At the default of 0.75 that includes columns holding
    a few textual sentinels such as ``"unknown"`` among otherwise numeric values,
    which is the common real-world shape.

    Columns whose values carry leading zeros are exempt. A postal code such as
    ``"02134"`` parses as a number but is not one, and converting it would destroy
    the value. Nothing is converted here regardless.

    Inspection is bounded to the first 1000 non-missing values per column, so the
    cost is independent of dataset height.
    """
    issues: list[QualityIssue] = []
    threshold = context.config.numeric_text_ratio_threshold

    for profile in context.profile.column_profiles:
        if profile.detected_kind is not ColumnKind.CATEGORICAL or not profile.count:
            continue

        sample = frame[profile.name].dropna().head(_TEXT_SAMPLE_SIZE).astype(str)
        if sample.empty:
            continue

        parsed = pd.to_numeric(sample, errors="coerce")
        numeric_ratio = float(parsed.notna().sum()) / float(len(sample))
        if numeric_ratio < threshold:
            continue

        leading_zeros = int(sample.str.match(_LEADING_ZERO_PATTERN).sum())
        if leading_zeros:
            continue

        non_numeric = sorted({str(value) for value in sample[parsed.isna()]})[:5]
        issues.append(
            QualityIssue(
                code="possible_numeric_stored_as_text",
                severity=Severity.WARNING,
                column=profile.name,
                requires_review=True,
                message=(
                    f"Column {profile.name!r} is stored as text but "
                    f"{numeric_ratio:.1%} of the inspected values parse as numbers."
                ),
                details={
                    "numeric_ratio": numeric_ratio,
                    "threshold": threshold,
                    "inspected_values": int(len(sample)),
                    "non_numeric_examples": non_numeric,
                    "pandas_dtype": profile.pandas_dtype,
                },
                recommendation=(
                    "If these are measurements, convert the column to a numeric "
                    "dtype yourself. The library does not convert it for you, "
                    "because the non-numeric values would silently become missing."
                ),
            )
        )
    return issues


# --------------------------------------------------------------------------- #
# H. Infinite values
# --------------------------------------------------------------------------- #


def check_infinite_values(
    frame: pd.DataFrame, context: QualityContext
) -> list[QualityIssue]:
    """Report infinities in numeric columns. Nothing is replaced.

    This is one of only two error-level findings: an infinity is a measurement,
    not a heuristic, and it breaks scaling, imputation, and most estimators.
    """
    return [
        QualityIssue(
            code="infinite_values",
            severity=Severity.ERROR,
            column=profile.name,
            message=(
                f"Column {profile.name!r} contains {profile.infinite_count} "
                "infinite value(s), which pandas does not count as missing."
            ),
            details={
                "infinite_count": profile.infinite_count,
                "count": profile.count,
            },
            recommendation=(
                "Infinities survive imputation and break scaling and most "
                "estimators. Decide whether they represent a division by zero, a "
                "sentinel, or a genuine overflow before continuing."
            ),
        )
        for profile in context.profile.column_profiles
        if profile.infinite_count
    ]


# --------------------------------------------------------------------------- #
# I. Outlier candidates
# --------------------------------------------------------------------------- #


def check_outliers(frame: pd.DataFrame, context: QualityContext) -> list[QualityIssue]:
    """Report values outside the Tukey fences. Nothing is removed or clipped.

    The rule is ``Q1 - k * IQR`` to ``Q3 + k * IQR`` with ``k`` from the
    configuration. Reported at info level on purpose: this rule flags roughly
    0.7 percent of any normally distributed sample by construction, so raising it
    as a warning would bury genuine warnings under routine tail behaviour.

    Columns are skipped when the interquartile range is zero, when fewer than four
    finite values are present, or when the column is constant -- in each case the
    fences carry no information.
    """
    issues: list[QualityIssue] = []
    multiplier = context.config.outlier_iqr_multiplier

    for profile in context.profile.column_profiles:
        if profile.detected_kind is not ColumnKind.NUMERIC or profile.numeric is None:
            continue
        if profile.is_constant or profile.count < 4:
            continue

        summary = profile.numeric
        if summary.iqr is None or summary.q25 is None or summary.q75 is None:
            continue
        if summary.iqr == 0.0:
            continue

        lower = summary.q25 - multiplier * summary.iqr
        upper = summary.q75 + multiplier * summary.iqr

        values = frame[profile.name].to_numpy(dtype="float64", na_value=np.nan)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue

        outlier_count = int(np.count_nonzero((finite < lower) | (finite > upper)))
        if not outlier_count:
            continue

        issues.append(
            QualityIssue(
                code="possible_outliers",
                severity=Severity.INFO,
                column=profile.name,
                requires_review=True,
                message=(
                    f"Column {profile.name!r} has {outlier_count} value(s) outside "
                    f"the Tukey fences [{lower:.4g}, {upper:.4g}] "
                    f"({outlier_count / finite.size:.1%} of finite values)."
                ),
                details={
                    "outlier_count": outlier_count,
                    "outlier_ratio": outlier_count / finite.size,
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                    "q25": summary.q25,
                    "q75": summary.q75,
                    "iqr": summary.iqr,
                    "multiplier": multiplier,
                },
                recommendation=(
                    "These are candidates, not errors. Extreme values are often "
                    "the most informative rows; inspect them before deciding."
                ),
            )
        )
    return issues


# --------------------------------------------------------------------------- #
# J. Multicollinearity among numeric features
# --------------------------------------------------------------------------- #


def check_multicollinearity(
    frame: pd.DataFrame, context: QualityContext
) -> list[QualityIssue]:
    """Report numeric features whose variation is largely redundant with the rest.

    For each eligible numeric feature, an ordinary least-squares fit against
    every *other* eligible feature gives an R-squared; the variance inflation
    factor is ``1 / (1 - R-squared)``. A VIF of 1 means the feature carries
    information none of the others do. A VIF at or above the configured
    threshold -- 10 by convention -- means most of its variance is already
    present elsewhere in the feature set.

    The fit is solved with :func:`numpy.linalg.lstsq`, not by inverting a
    correlation matrix. Matrix inversion is the textbook shortcut for VIF, but
    on nearly dependent columns it is numerically unstable enough to return a
    large *negative* number where the true value is a large positive one --
    floating-point cancellation in the inverse, not a property of the data.
    A least-squares solve degrades smoothly instead, including on a feature
    that is an exact linear combination of the others.

    Constant, infinite-valued, and non-numeric columns are excluded before the
    fit. Each already has its own check, and a constant column would make every
    fit singular for a reason that has nothing to do with multicollinearity.
    Rows with a missing value in any eligible column are excluded from this fit
    only; nothing is imputed and the frame is not modified.

    Fewer than two eligible columns, or no more rows than columns after the
    exclusions, leaves nothing to compare, and the check reports nothing rather
    than fit a regression with no degrees of freedom left to test it.

    The cost is bounded. More than ``_VIF_MAX_COLUMNS`` eligible columns and the
    check declines with an informational finding naming the count, instead of
    running a cubic amount of work; more than ``_VIF_MAX_ROWS`` complete rows and
    an evenly spaced deterministic subset is fitted, recorded in the finding.

    Each column is centred and divided by its largest absolute deviation before
    the fit. A VIF is invariant to the scale of every column, so this changes no
    answer, and it keeps values near ``1e300`` from overflowing to infinity when
    squared. A column whose values cannot be represented as finite float64 at all
    -- a Python integer beyond its range, say -- is left out of the fit, since no
    number computed from it would mean anything.
    """
    candidates = [
        profile.name
        for profile in context.feature_profiles
        if profile.detected_kind in (ColumnKind.NUMERIC, ColumnKind.BOOLEAN)
        and not profile.is_constant
        and not profile.infinite_count
    ]
    if len(candidates) > _VIF_MAX_COLUMNS:
        return [_multicollinearity_not_assessed(len(candidates))]

    converted: dict[Hashable, np.ndarray] = {}
    for column in candidates:
        values = _as_finite_float(frame[column])
        if values is not None:
            converted[column] = values
    columns = list(converted)
    if len(columns) < 2:
        return []

    raw = np.column_stack([converted[column] for column in columns])
    complete = raw[~np.isnan(raw).any(axis=1)]
    available_rows = len(complete)
    if available_rows > _VIF_MAX_ROWS:
        positions = np.linspace(0, available_rows - 1, _VIF_MAX_ROWS).round().astype("int64")
        complete = complete[positions]
    if len(complete) <= len(columns):
        return []

    matrix = _centred_and_bounded(complete)
    threshold = context.config.multicollinearity_vif_threshold
    issues: list[QualityIssue] = []

    for position, column in enumerate(columns):
        target_values = matrix[:, position]
        predictors = np.delete(matrix, position, axis=1)
        design = np.column_stack([np.ones(len(predictors)), predictors])

        coefficients, _, _, _ = np.linalg.lstsq(design, target_values, rcond=None)
        residuals = target_values - design @ coefficients

        total_variance = float(np.sum((target_values - target_values.mean()) ** 2))
        if not np.isfinite(total_variance) or total_variance == 0.0:
            continue

        unexplained = float(np.sum(residuals**2)) / total_variance
        if not np.isfinite(unexplained):
            # Never reached on centred, bounded data; kept so that a NaN can only
            # ever mean "not measured" and never be written as a VIF.
            continue
        r_squared = min(max(1.0 - unexplained, 0.0), 1.0)
        vif = 1.0 / max(1.0 - r_squared, _MIN_UNEXPLAINED_VARIANCE)
        if vif < threshold:
            continue

        issues.append(
            QualityIssue(
                code="possible_multicollinearity",
                severity=Severity.WARNING,
                column=column,
                requires_review=True,
                message=(
                    f"Column {column!r} has a variance inflation factor of "
                    f"{vif:.3g} against the other numeric features, at or above "
                    f"the configured threshold of {threshold:.3g}. Most of its "
                    "variation is redundant with features already in the set."
                ),
                details={
                    "vif": vif,
                    "threshold": threshold,
                    "r_squared": r_squared,
                    "other_numeric_features": [name for name in columns if name != column],
                    "rows_used": int(len(complete)),
                    "rows_available": int(available_rows),
                },
                recommendation=(
                    "High collinearity does not hurt a model's predictions by "
                    "itself, but it destabilises coefficients in linear models "
                    "and makes per-feature importance unreliable. Consider "
                    "dropping or combining one of the correlated features if "
                    "you need to interpret coefficients."
                ),
            )
        )
    return issues


def _as_finite_float(series: pd.Series) -> np.ndarray | None:
    """The column as float64 with gaps as NaN, or ``None`` if it cannot be.

    ``None`` is returned when a value does not fit in a float64 at all (a Python
    integer beyond its range raises ``OverflowError`` on conversion) or converts
    to an infinity. Such a column has no finite number to contribute, and letting
    one infinity into the matrix would turn every fit it touches into NaN.
    Nullable dtypes are converted with their missing values as NaN, so a gap is
    excluded row-wise like any other.
    """
    try:
        values = series.to_numpy(dtype="float64", na_value=np.nan)
    except (OverflowError, TypeError, ValueError):
        return None
    if np.isinf(values).any():
        return None
    return values


def _centred_and_bounded(matrix: np.ndarray) -> np.ndarray:
    """Bring every column into ``[-1, 1]``, then centre it, then rescale it.

    The first division comes before any sum: taking the mean of values near
    ``1e308`` would itself overflow to infinity. After it every value is at most
    1 in magnitude, so neither the mean nor any later square can overflow. The
    VIF is invariant to the location and scale of every column, so no answer
    changes.
    """
    magnitude = np.abs(matrix).max(axis=0)
    magnitude[magnitude == 0.0] = 1.0
    bounded = matrix / magnitude
    centred = bounded - bounded.mean(axis=0)
    spread = np.abs(centred).max(axis=0)
    spread[spread == 0.0] = 1.0
    return centred / spread


def _multicollinearity_not_assessed(eligible: int) -> QualityIssue:
    """Say that the check declined, rather than say nothing."""
    return QualityIssue(
        code="multicollinearity_not_assessed",
        severity=Severity.INFO,
        message=(
            f"Multicollinearity was not assessed: {eligible} numeric features "
            f"exceed the {_VIF_MAX_COLUMNS} this check fits, and the work grows "
            "with the cube of that count. No finding here is not evidence that "
            "the features are independent."
        ),
        details={"eligible_columns": eligible, "column_limit": _VIF_MAX_COLUMNS},
    )


# --------------------------------------------------------------------------- #
# K. Class imbalance
# --------------------------------------------------------------------------- #


def check_class_imbalance(
    frame: pd.DataFrame, context: QualityContext
) -> list[QualityIssue]:
    """Report an uneven class distribution. No resampling or reweighting occurs."""
    target = context.target_profile
    if target is None or target.task_type is not TaskType.CLASSIFICATION:
        return []
    if not target.class_ratios:
        return []

    minority_label = min(target.class_ratios, key=target.class_ratios.__getitem__)
    minority_ratio = target.class_ratios[minority_label]
    if minority_ratio >= context.config.imbalance_threshold:
        return []

    return [
        QualityIssue(
            code="class_imbalance",
            severity=Severity.WARNING,
            column=context.target,
            message=(
                f"The rarest class {minority_label!r} covers {minority_ratio:.1%} of "
                f"rows, below the configured {context.config.imbalance_threshold:.0%} "
                "threshold."
            ),
            details={
                "class_counts": dict(target.class_counts or {}),
                "class_ratios": dict(target.class_ratios),
                "minority_label": minority_label,
                "minority_ratio": minority_ratio,
                "imbalance_ratio": target.imbalance_ratio,
                "threshold": context.config.imbalance_threshold,
            },
            recommendation=(
                "Accuracy is misleading at this ratio; a model predicting only the "
                "majority class would score "
                f"{1 - minority_ratio:.1%}. Judge by recall, precision, or ROC-AUC."
            ),
        )
    ]


# --------------------------------------------------------------------------- #
# L. Possible target leakage
# --------------------------------------------------------------------------- #


def check_target_leakage(
    frame: pd.DataFrame, context: QualityContext
) -> list[QualityIssue]:
    """Report features that may encode the answer. Nothing is dropped or excluded.

    Four conservative signals, in decreasing certainty:

    1. The feature is exactly equal to the target. This is the only certain case
       and the only one reported as an error.
    2. The feature determines the target: every one of its values maps to a single
       target class. Columns near uniqueness are exempt, since an identifier
       trivially determines everything.
    3. The feature correlates with the target at or above the configured
       threshold, which defaults to 0.98.
    4. The feature is *named* like something recorded after the outcome.

    Every signal but the first produces ``possible_target_leakage`` with
    ``requires_review=True``. A strong-but-ordinary association is not leakage,
    and this check is deliberately reluctant to say otherwise.

    Cost is linear in cells: one pass per feature, no pairwise row comparison.
    """
    if context.target is None or context.target not in frame.columns:
        return []

    issues: list[QualityIssue] = []
    target_series = frame[context.target]

    for profile in context.feature_profiles:
        column = profile.name
        if column not in frame.columns:
            continue
        series = frame[column]

        if _is_exact_duplicate(series, target_series):
            issues.append(_exact_duplicate_issue(column, context.target))
            continue

        signals: list[str] = []
        details: dict[str, object] = {}

        determinism = _deterministic_mapping(series, target_series, profile.unique_count)
        if determinism is not None:
            signals.append("deterministic_mapping")
            details["distinct_feature_values"] = determinism

        association = _association_with_target(series, target_series, profile)
        if association is not None and abs(association) >= context.config.leakage_correlation_threshold:
            signals.append("extreme_association")
            details["correlation"] = association
            details["correlation_threshold"] = context.config.leakage_correlation_threshold

        name_signal = _suspicious_name(column, context.target)
        if name_signal:
            signals.append(name_signal)

        if signals:
            issues.append(_possible_leakage_issue(column, context.target, signals, details))

    return issues


def _is_exact_duplicate(series: pd.Series, target: pd.Series) -> bool:
    """Whether a feature carries exactly the target's values, row for row."""
    if series.isna().any() or target.isna().any():
        return False
    if series.equals(target):
        return True
    try:
        return bool(np.array_equal(series.to_numpy(), target.to_numpy()))
    except (TypeError, ValueError):
        return False


def _deterministic_mapping(
    series: pd.Series, target: pd.Series, unique_count: int
) -> int | None:
    """Return the feature's cardinality if each of its values fixes the target.

    Returns ``None`` when the mapping is not deterministic, or when the feature is
    so nearly unique that determinism is a mathematical inevitability rather than
    evidence of anything.
    """
    rows = len(series)
    if rows == 0 or unique_count < 2:
        return None
    if unique_count > max(2, rows // 2):
        return None

    frame = pd.DataFrame({"feature": series, "target": target}).dropna()
    if frame.empty:
        return None
    try:
        distinct_targets = frame.groupby("feature", observed=True, dropna=True)[
            "target"
        ].nunique()
    except TypeError:
        return None
    if distinct_targets.empty or not bool((distinct_targets <= 1).all()):
        return None
    return int(unique_count)


def _association_with_target(
    series: pd.Series, target: pd.Series, profile
) -> float | None:
    """Return the Pearson correlation with the target, when both are numeric."""
    if profile.detected_kind not in (ColumnKind.NUMERIC, ColumnKind.BOOLEAN):
        return None
    if profile.is_constant:
        return None
    if not (pd.api.types.is_numeric_dtype(target) or pd.api.types.is_bool_dtype(target)):
        return None
    if target.nunique(dropna=True) < 2:
        return None

    try:
        return correlation(
            series.astype("float64"), target.astype("float64"), nan_policy="omit"
        )
    except Exception:
        # A degenerate pair carries no association to report; the dedicated
        # constant-column and missing-value checks cover those cases.
        return None


def _suspicious_name(column: Hashable, target: Hashable) -> str | None:
    """Return the name-based signal a column triggers, if any."""
    name = str(column)
    target_name = str(target)
    lowered = name.lower()

    if lowered != target_name.lower() and target_name.lower() in lowered:
        return "name_contains_target"
    if _EVENT_TIME_PATTERN.search(name) and _POST_OUTCOME_PATTERN.search(name):
        return "post_outcome_timestamp_name"
    if _POST_OUTCOME_PATTERN.search(name):
        return "post_outcome_name"
    return None


def _exact_duplicate_issue(column: Hashable, target: Hashable) -> QualityIssue:
    """Build the one leakage finding that is certain rather than heuristic."""
    return QualityIssue(
        code="target_leakage_exact_duplicate",
        severity=Severity.ERROR,
        column=column,
        requires_review=False,
        message=(
            f"Column {column!r} is exactly equal to the target {target!r} in every "
            "row. Training on it would measure nothing."
        ),
        details={"target": str(target)},
        recommendation=(
            "Remove this column from the feature set yourself, or confirm it is a "
            "duplicate export of the label."
        ),
    )


def _possible_leakage_issue(
    column: Hashable,
    target: Hashable,
    signals: list[str],
    details: dict[str, object],
) -> QualityIssue:
    """Build a heuristic leakage finding that states its evidence and asks for review."""
    return QualityIssue(
        code="possible_target_leakage",
        severity=Severity.WARNING,
        column=column,
        requires_review=True,
        message=(
            f"Column {column!r} may contain information about the target "
            f"{target!r} that would not be available at prediction time "
            f"(signals: {', '.join(signals)}). This is a heuristic and needs review."
        ),
        details={"signals": signals, "target": str(target), **details},
        recommendation=(
            "Ask whether this value is known before the outcome occurs. If it is "
            "recorded afterwards, it inflates validation scores and the model will "
            "fail in production. Nothing has been removed."
        ),
    )


#: The checks the inspector runs by default, in reporting order.
DEFAULT_CHECKS: tuple[Check, ...] = (
    check_missing_values,
    check_duplicate_rows,
    check_constant_columns,
    check_near_constant_columns,
    check_high_cardinality,
    check_id_like_columns,
    check_numeric_stored_as_text,
    check_infinite_values,
    check_outliers,
    check_multicollinearity,
    check_class_imbalance,
    check_target_leakage,
)
