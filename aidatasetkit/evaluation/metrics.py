"""What to measure, and when a measurement is not available.

Three policies, one per task family. Each decides which metrics to attempt, and
for every one it attempts it returns either a number or a recorded reason -- never
a silent omission, because a comparison table with a hole in it invites the
reader to fill the hole with an assumption.

Four decisions in here are worth stating out loud, because each is a place where
the obvious implementation is wrong.

**The positive class is not class 1.** Once labels become integers,
``predict_proba[:, 1]`` is the probability of whichever label sorted second.
``"churn"`` sorts before ``"stay"``, so for the most ordinary churn dataset in
the world the interesting class is column *zero*. scikit-learn's ``pos_label``
defaults to 1 and would answer confidently about the wrong outcome. Every binary
metric here reads the positive class from the
:class:`~aidatasetkit.preprocessing.types.TargetEncoding` that produced the
integers, and refuses rather than guessing when none was resolved.

**RMSE is not MSE.** They differ by a square root, they are reported in different
units, and a table that labelled one as the other would be wrong in a way that
looks entirely plausible. It is computed as the root of the mean squared error
and checked against that definition in the tests.

**A metric is not a warning to be silenced.** ``zero_division`` is passed
explicitly rather than left to default, so a class the model never predicts
scores a deliberate zero instead of emitting ``UndefinedMetricWarning``. Nothing
here installs a global warning filter.

**A clustering metric is not a verdict.** The three internal measures score the
geometry of a partition, and geometry is not correctness -- measured, all three
prefer a wrong convex split to the right non-convex one. So clustering has no
default ranking metric, and :data:`DEFAULT_RANKING_METRIC` records why at
length. Its absence from that mapping is a decision, not an omission.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from aidatasetkit.core.exceptions import ValidationError
from aidatasetkit.core.types import TaskType
from aidatasetkit.evaluation.types import (
    EvaluationReport,
    MetricDirection,
    MetricStatus,
    MetricValue,
)

__all__ = [
    "CLASSIFICATION_METRICS",
    "CLUSTERING_METRICS",
    "NOISE_LABEL",
    "REGRESSION_METRICS",
    "SILHOUETTE_ROW_LIMIT",
    "DEFAULT_RANKING_METRIC",
    "evaluate_classification",
    "evaluate_clustering",
    "evaluate_regression",
    "metric_direction",
]

#: Every metric a classification evaluation attempts, in report order.
CLASSIFICATION_METRICS: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
)

#: Every metric a regression evaluation attempts, in report order.
REGRESSION_METRICS: tuple[str, ...] = ("mae", "mse", "rmse", "r2")

#: Every metric a clustering evaluation attempts, in report order.
#:
#: All three are *internal* measures: they score the geometry of a partition
#: against no ground truth, because in clustering there is none. Read
#: :data:`DEFAULT_RANKING_METRIC` for what that means for ordering models.
CLUSTERING_METRICS: tuple[str, ...] = (
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
)

#: The label a density-based clusterer gives a row it declines to assign.
#:
#: ``-1`` is not a cluster and must never be counted as one. Every metric here
#: drops these rows before scoring, because a noise bucket is not a group and
#: measuring its compactness would be measuring nothing.
NOISE_LABEL = -1

#: Above this many scored rows, the silhouette is reported rather than computed.
#:
#: The silhouette needs every pairwise distance, so it is O(n^2) in both time and
#: memory. Measured on this machine: 1,000 rows took 0.4s; 5,000 took 9.0s;
#: 20,000 took **164 seconds** and needed a 3.2 GB distance matrix. A library
#: that quietly took three minutes and 3 GB to fill in one table cell would be
#: making a decision on the caller's behalf about what their time is worth.
#:
#: scikit-learn offers ``sample_size`` to approximate it. That is deliberately
#: not used: it would put a number in the table that is neither the silhouette
#: nor reproducible across runs, and a metric that changes when nothing changed
#: cannot be compared. The metric is reported unavailable instead, with the row
#: count in the reason, and the limit can be raised per call.
SILHOUETTE_ROW_LIMIT = 10_000

#: Which way each metric's scale runs. Stated once, read everywhere, so a
#: ranking cannot disagree with a report about what "better" means.
_DIRECTIONS: dict[str, MetricDirection] = {
    "accuracy": MetricDirection.HIGHER_IS_BETTER,
    "precision": MetricDirection.HIGHER_IS_BETTER,
    "recall": MetricDirection.HIGHER_IS_BETTER,
    "f1": MetricDirection.HIGHER_IS_BETTER,
    "roc_auc": MetricDirection.HIGHER_IS_BETTER,
    "mae": MetricDirection.LOWER_IS_BETTER,
    "mse": MetricDirection.LOWER_IS_BETTER,
    "rmse": MetricDirection.LOWER_IS_BETTER,
    "r2": MetricDirection.HIGHER_IS_BETTER,
    "silhouette": MetricDirection.HIGHER_IS_BETTER,
    "davies_bouldin": MetricDirection.LOWER_IS_BETTER,
    "calinski_harabasz": MetricDirection.HIGHER_IS_BETTER,
}

#: The metric a comparison ranks on unless the caller names another.
#:
#: ``f1`` for classification rather than ``accuracy``: accuracy on a 95/5 split
#: rewards a model that never predicts the minority class at all, which is the
#: single most common way a comparison table misleads someone.
#:
#: ``rmse`` for regression rather than ``r2``: it is in the units of the target,
#: it is defined on any evaluation set with at least one row, and ``r2`` is
#: undefined whenever the evaluation target happens not to vary.
#:
#: **Clustering is deliberately absent, and its absence is the decision.** There
#: is no ground truth to be right about, so all three clustering metrics score
#: geometry -- and geometry is not the same thing as correctness. Measured on two
#: interleaved half-moons, where the correct partition is known:
#:
#: ==========================  ============  ================  ===================
#: Partition                   silhouette    davies_bouldin    calinski_harabasz
#: ==========================  ============  ================  ===================
#: KMeans, **wrong**           ``+0.488``    ``0.781``         ``440.7``
#: DBSCAN, **correct**         ``+0.327``    ``1.167``         ``193.5``
#: ==========================  ============  ================  ===================
#:
#: All three preferred the wrong answer, and they preferred it unanimously,
#: because all three reward compact convex blobs and the true clusters are
#: neither. A default here would have silently ranked the model that found the
#: real structure last on every one of the three.
#:
#: Worse, the scores are not even evidence that structure exists: on 300 rows
#: drawn from a *single* Gaussian -- ``default_rng(11)``, no clusters in it at
#: all -- KMeans at k=3 scored a perfectly respectable silhouette of ``+0.347``.
#:
#: So a clustering comparison reports all three and ranks by none. Every model
#: comes back unranked, through the same path S7 already uses for a model with
#: no ranking metric. A caller who has read the above and wants an order can name
#: a metric explicitly; what this library will not do is pick one for them and
#: let the table imply a winner.
DEFAULT_RANKING_METRIC: dict[TaskType, str] = {
    TaskType.CLASSIFICATION: "f1",
    TaskType.REGRESSION: "rmse",
}

#: The averaging applied to multiclass precision, recall and F1.
#:
#: ``macro`` treats every class as equally important, which is the reading that
#: keeps a rare class visible. ``micro`` would collapse to accuracy on a
#: single-label problem and hide exactly the failure the metric is there to show;
#: ``weighted`` would re-introduce the majority class's dominance by the back
#: door. The choice is stated on every multiclass metric it produces.
MULTICLASS_AVERAGING = "macro"


def metric_direction(name: str) -> MetricDirection:
    """Return which way a metric's scale runs.

    Raises:
        KeyError: If the metric is not one this library computes.
    """
    return _DIRECTIONS[name]


def _unavailable(name: str, status: MetricStatus, reason: str) -> MetricValue:
    return MetricValue(
        name=name, value=None, direction=_DIRECTIONS[name], status=status, reason=reason
    )


def _available(name: str, value: float, **detail: Any) -> MetricValue:
    """Wrap a computed number, demoting a non-finite result to ``UNDEFINED``.

    A metric formula can be perfectly correct and still have no answer on a
    particular evaluation set. Reporting the resulting ``nan`` as though it were
    a score is how an undefined result ends up being ranked.
    """
    number = float(value)
    if not math.isfinite(number):
        return _unavailable(
            name,
            MetricStatus.UNDEFINED,
            f"{name} evaluated to {number!r} on these rows, which is not a score "
            "that can be compared.",
        )
    return MetricValue(
        name=name, value=number, direction=_DIRECTIONS[name], detail=detail
    )


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def evaluate_classification(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    *,
    probabilities: np.ndarray | None = None,
    encoding: Any = None,
    class_count: int | None = None,
) -> EvaluationReport:
    """Measure a fitted classifier against held-out rows.

    Args:
        y_true: Encoded true labels for the evaluation rows.
        y_predicted: Encoded predictions for the same rows, in the same order.
        probabilities: ``predict_proba`` output, or ``None`` when the estimator
            does not advertise probability support. Never inferred from a model
            name, and never called speculatively.
        encoding: The :class:`~aidatasetkit.preprocessing.types.TargetEncoding`
            the integers came from. It carries the positive label, which is what
            keeps a binary metric attached to the class the analyst cares about.
        class_count: How many classes the *training* target held. Taken from
            training rather than from the evaluation rows, because an evaluation
            split that happens to miss a rare class has not turned a three-class
            problem into a two-class one.

    Returns:
        An :class:`EvaluationReport` holding every metric in
        :data:`CLASSIFICATION_METRICS`, available or not.
    """
    from sklearn import metrics as skmetrics

    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)
    observed = int(len(np.unique(y_true)))
    total_classes = class_count if class_count is not None else observed
    is_binary = total_classes == 2

    results: list[MetricValue] = [
        _available("accuracy", skmetrics.accuracy_score(y_true, y_predicted))
    ]

    positive = _positive_class(encoding) if is_binary else None
    for name, function in (
        ("precision", skmetrics.precision_score),
        ("recall", skmetrics.recall_score),
        ("f1", skmetrics.f1_score),
    ):
        results.append(
            _class_metric(
                name,
                function,
                y_true,
                y_predicted,
                is_binary,
                positive,
                encoding,
                total_classes,
            )
        )

    results.append(
        _roc_auc(y_true, probabilities, is_binary, positive, encoding, total_classes)
    )

    return EvaluationReport(
        task_type=TaskType.CLASSIFICATION,
        metrics=tuple(results),
        row_count=int(len(y_true)),
        class_labels=_class_labels(encoding),
    )


def _positive_class(encoding: Any) -> int | None:
    """The encoded positive class, or ``None`` when none was resolved.

    Accepts either the fitted :class:`TargetLabelEncoder` or the
    :class:`TargetEncoding` it carries. Only the encoder was understood at first,
    so passing the record -- which this function's own caller documents as
    acceptable -- sent a perfectly well resolved positive class into a bare
    ``except`` and reported it as unresolved. The metric then answered about the
    wrong class, or declined to answer at all, for a target that had said exactly
    which class mattered.

    This module never decides which class is interesting; that belongs to the
    target contract, which refuses to guess.
    """
    if encoding is None:
        return None
    resolved = getattr(encoding, "encoding", encoding)
    if not getattr(resolved, "positive_label_resolved", False):
        return None
    index = getattr(resolved, "positive_label_encoded", None)
    return None if index is None else int(index)


def _class_labels(encoding: Any) -> tuple[str, ...] | None:
    if encoding is None:
        return None
    classes = getattr(getattr(encoding, "encoding", encoding), "classes", None)
    if classes is None:
        return None
    return tuple(str(label) for label in classes)


def _class_metric(
    name: str,
    function: Any,
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    is_binary: bool,
    positive: int | None,
    encoding: Any,
    total_classes: int,
) -> MetricValue:
    """Precision, recall or F1, with the averaging decision made explicitly.

    ``labels=`` is pinned to the classes the *training* target held, and that is
    load-bearing rather than tidy. Left out, scikit-learn infers the averaged set
    from ``set(y_true) | set(y_predicted)`` -- which is per-model, because each
    model predicts a different set. Two models judged on the same rows would then
    have their macro scores divided by different denominators, and the model that
    simply declined to predict a hard class would be rewarded for it. That is
    enough to change which model ranks first.

    ``zero_division=0.0`` is passed rather than left to default: a class the
    model never predicts has a precision that is genuinely undefined, and
    scikit-learn's answer to that is a warning plus a zero. Choosing the zero
    deliberately is the same number with the warning turned into a decision, and
    the detail records when the division was degenerate so a reader is not left
    to read that zero as a measurement.
    """
    labels = list(range(total_classes))

    if is_binary and positive is not None:
        value = function(
            y_true, y_predicted, pos_label=positive, average="binary", zero_division=0.0
        )
        label = None
        if encoding is not None:
            resolved = getattr(encoding, "encoding", encoding)
            label = str(getattr(resolved, "positive_label", None))
        return _available(
            name,
            value,
            averaging="binary",
            positive_label=label,
            degenerate=_is_degenerate(name, y_true, y_predicted, positive),
        )

    if is_binary:
        # Binary, but nobody said which class is the event of interest. Macro
        # makes no claim about that -- it weights both classes equally -- so it is
        # the honest answer, and the detail says so rather than leaving a reader
        # to assume a positive class was chosen.
        value = function(
            y_true, y_predicted, labels=labels,
            average=MULTICLASS_AVERAGING, zero_division=0.0,
        )
        return _available(
            name, value, averaging=MULTICLASS_AVERAGING, positive_label=None
        )

    value = function(
        y_true, y_predicted, labels=labels,
        average=MULTICLASS_AVERAGING, zero_division=0.0,
    )
    return _available(
        name, value, averaging=MULTICLASS_AVERAGING, averaged_over=len(labels)
    )


def _is_degenerate(
    name: str, y_true: np.ndarray, y_predicted: np.ndarray, positive: int
) -> bool:
    """Whether this binary score is a ``0/0`` that ``zero_division`` turned into 0.

    Recorded rather than promoted to ``UNDEFINED``: the zero is the conventional
    answer and stays rankable, but a reader comparing it against a real 0.0
    deserves to know which one they are looking at.
    """
    if name == "precision":
        return not bool(np.any(y_predicted == positive))
    if name in ("recall", "f1"):
        return not bool(np.any(y_true == positive))
    return False


def _roc_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray | None,
    is_binary: bool,
    positive: int | None,
    encoding: Any,
    total_classes: int,
) -> MetricValue:
    """ROC-AUC, or the specific reason there isn't one.

    Four distinct ways this metric goes missing, and they are different facts
    about different things: the model, the target contract, the evaluation split,
    and this library's own scope.
    """
    from sklearn import metrics as skmetrics

    if probabilities is None:
        return _unavailable(
            "roc_auc",
            MetricStatus.UNSUPPORTED_BY_MODEL,
            "This model does not advertise probability estimates, and ROC-AUC "
            "cannot be computed from hard predictions. Its capabilities record "
            "supports_predict_proba=False.",
        )

    if not is_binary:
        return _unavailable(
            "roc_auc",
            MetricStatus.NOT_APPLICABLE,
            f"ROC-AUC over {total_classes} classes needs an explicit "
            "one-vs-rest or one-vs-one policy, and every class present in the "
            "evaluation rows. This version publishes no such policy rather than "
            "picking one silently.",
        )

    if positive is None:
        return _unavailable(
            "roc_auc",
            MetricStatus.UNDEFINED,
            "No positive label was resolved for this target, so no probability "
            "column can be called the positive one -- and swapping the two gives "
            "1 - AUC. Construct the target encoder with positive_label=... to "
            "say which class is the event of interest.",
        )

    if len(np.unique(y_true)) < 2:
        return _unavailable(
            "roc_auc",
            MetricStatus.UNDEFINED,
            "The evaluation rows hold a single class, so there is no pair of "
            "classes to separate and ROC-AUC has no value. This is a fact about "
            "the split rather than about the model.",
        )

    scores = np.asarray(probabilities)
    if scores.ndim != 2 or scores.shape[1] <= positive:
        return _unavailable(
            "roc_auc",
            MetricStatus.FAILED,
            f"The probability matrix has shape {scores.shape}, which has no "
            f"column {positive} for the positive class.",
        )

    label = None
    if encoding is not None:
        resolved = getattr(encoding, "encoding", encoding)
        label = str(getattr(resolved, "positive_label", None))
    try:
        value = skmetrics.roc_auc_score(
            (y_true == positive).astype(int), scores[:, positive]
        )
    except ValueError as error:
        return _unavailable("roc_auc", MetricStatus.FAILED, str(error))
    return _available(
        "roc_auc", value, positive_label=label, positive_column=int(positive)
    )


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #


def evaluate_regression(y_true: np.ndarray, y_predicted: np.ndarray) -> EvaluationReport:
    """Measure a fitted regressor against held-out rows.

    Args:
        y_true: True target values for the evaluation rows, as quantities. They
            were never label encoded and are not encoded here.
        y_predicted: Predictions for the same rows, in the same order.

    Returns:
        An :class:`EvaluationReport` holding every metric in
        :data:`REGRESSION_METRICS`.
    """
    from sklearn import metrics as skmetrics

    y_true = np.asarray(y_true, dtype="float64")
    y_predicted = np.asarray(y_predicted, dtype="float64")

    mse = float(skmetrics.mean_squared_error(y_true, y_predicted))
    results = [
        _available("mae", skmetrics.mean_absolute_error(y_true, y_predicted)),
        _available("mse", mse),
        # The root of the mean squared error, computed here rather than taken
        # from a differently-named function, so the two can never drift apart.
        _available("rmse", math.sqrt(mse)),
        _r2(y_true, y_predicted),
    ]

    return EvaluationReport(
        task_type=TaskType.REGRESSION,
        metrics=tuple(results),
        row_count=int(len(y_true)),
    )


def _r2(y_true: np.ndarray, y_predicted: np.ndarray) -> MetricValue:
    """R², or the reason it has no value on these rows.

    R² is the fraction of the target's variance the model accounts for, so it
    needs the target to have some. On a constant evaluation target the
    denominator is zero: scikit-learn answers ``1.0`` for a perfect constant
    prediction and ``0.0`` for a wrong one -- the *same* ``0.0`` whether the
    prediction is off by 1.5 or by 892.5. None of those describes anything, and
    all of them would be ranked. Fewer than two rows is the other degenerate
    case: there is no spread for a score to be a fraction of.
    """
    if len(y_true) < 2:
        return _unavailable(
            "r2",
            MetricStatus.UNDEFINED,
            f"R2 needs at least two evaluation rows to have a variance to "
            f"explain, and there {'is' if len(y_true) == 1 else 'are'} "
            f"{len(y_true)}.",
        )
    variance = float(np.var(y_true))
    if variance == 0.0:
        # Distinguished because they are different facts. A genuinely constant
        # column is a property of the data; a variance that underflowed to zero
        # is a property of the magnitudes, and telling somebody their values are
        # all the same when they are not would send them looking for the wrong
        # thing.
        distinct = int(np.unique(y_true).size)
        if distinct == 1:
            reason = (
                "Every evaluation target is the same value, so there is no "
                "variance to explain and R2 is undefined. This is a fact about "
                "the evaluation rows rather than about the model."
            )
        else:
            reason = (
                f"The {distinct} distinct evaluation targets are close enough "
                "together that their variance underflows to zero at this "
                "magnitude, leaving R2 with a denominator of zero. Rescaling the "
                "target restores it."
            )
        return _unavailable("r2", MetricStatus.UNDEFINED, reason)

    from sklearn import metrics as skmetrics

    return _available("r2", skmetrics.r2_score(y_true, y_predicted))


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #


def evaluate_clustering(
    X: Any,
    labels: np.ndarray,
    *,
    silhouette_row_limit: int = SILHOUETTE_ROW_LIMIT,
) -> EvaluationReport:
    """Measure the geometry of a partition, with no ground truth to check against.

    Read :data:`DEFAULT_RANKING_METRIC` before using these numbers to choose a
    model. All three are internal measures, all three reward compact convex
    clusters, and all three were measured preferring a *wrong* partition to a
    correct one on non-convex data.

    Two things happen here that do not happen in the supervised policies.

    **Noise rows are dropped before scoring.** A row labelled :data:`NOISE_LABEL`
    is one the algorithm declined to assign, so it belongs to no cluster and
    cannot contribute to a measure of how good the clusters are. Scoring it as
    though ``-1`` were a cluster would measure the compactness of everything the
    model rejected, pooled together, which describes nothing. How many rows this
    removed is recorded on every metric.

    **The matrix must be the one the model clustered.** These metrics recompute
    distances, so handing them the raw frame while the model saw a scaled matrix
    would score a geometry the model never worked in. Callers inside this library
    pass the transformed matrix; the parameter is positional and named ``X`` to
    make that hard to get wrong.

    Args:
        X: The feature matrix the labels were produced from -- dense array or
            sparse matrix, with one row per label.
        labels: Cluster assignment per row, with ``-1`` meaning noise.
        silhouette_row_limit: Above this many scored rows the silhouette is
            reported unavailable rather than computed. See
            :data:`SILHOUETTE_ROW_LIMIT` for the measurements behind the default.

    Returns:
        An :class:`EvaluationReport` holding every metric in
        :data:`CLUSTERING_METRICS`, each with a number or a reason.

    Raises:
        ValidationError: If ``X`` and ``labels`` describe different numbers of
            rows. Scoring them against each other would silently compare a row
            with another row's label.
    """
    from scipy import sparse

    labels = np.asarray(labels)
    # Both guards exist because the failures they replace were numpy's, and
    # numpy's name the operation rather than the mistake: a column of string
    # labels died in `ufunc 'less'`, and a 2-D array in "boolean index did not
    # match indexed array along axis 1". Neither tells a caller what to change.
    if labels.ndim != 1:
        raise ValidationError(
            f"Cluster labels must be one label per row, but an array of shape "
            f"{labels.shape} was given. Flatten it, or pass the column holding "
            "the assignment."
        )
    if not np.issubdtype(labels.dtype, np.number) and not np.issubdtype(
        labels.dtype, np.bool_
    ):
        raise ValidationError(
            f"Cluster labels must be numeric ids, but their dtype is "
            f"{labels.dtype}. Names are not cluster ids: two runs would order "
            "them differently and nothing here could tell which group is which. "
            "Encode them to integers, keeping "
            f"{NOISE_LABEL} for any rows the algorithm declined to assign."
        )

    total_rows = int(labels.shape[0])
    if X.shape[0] != total_rows:
        raise ValidationError(
            f"The matrix has {X.shape[0]} rows and there are {total_rows} "
            "labels. One label describes one row, and pairing them by position "
            "when the counts differ would score rows against other rows' "
            "clusters."
        )

    # Only -1 means noise. Another negative label is either a mistake or another
    # library's convention, and counting it as an ordinary cluster would pool a
    # second set of rejected rows into the score without a word. Refused rather
    # than guessed, because guessing which convention was meant is exactly the
    # kind of invention this library does not do.
    stray = np.unique(labels[(labels < 0) & (labels != NOISE_LABEL)])
    if stray.size:
        raise ValidationError(
            f"Cluster label(s) {stray.tolist()} are negative but are not "
            f"{NOISE_LABEL}, which is the only value this library reads as "
            "noise. Scoring them as ordinary clusters would measure the "
            "compactness of rows something declined to assign. Relabel them to "
            f"{NOISE_LABEL} if they are noise, or to non-negative ids if they "
            "are clusters."
        )

    assigned = labels != NOISE_LABEL
    noise_rows = int(total_rows - assigned.sum())
    scored_rows = int(assigned.sum())
    cluster_count = int(np.unique(labels[assigned]).size)

    context: dict[str, Any] = {
        "n_clusters": cluster_count,
        "scored_row_count": scored_rows,
        "noise_row_count": noise_rows,
    }

    blocker = _partition_blocker(cluster_count, scored_rows, noise_rows, total_rows)
    if blocker is not None:
        return EvaluationReport(
            task_type=TaskType.CLUSTERING,
            metrics=tuple(
                _unavailable(name, MetricStatus.UNDEFINED, blocker)
                for name in CLUSTERING_METRICS
            ),
            row_count=total_rows,
        )

    # Sliced only once the partition is known to be scoreable, so a degenerate
    # labelling never pays for a copy of the matrix.
    scored_matrix = X[assigned]
    scored_labels = labels[assigned]
    is_sparse = sparse.issparse(scored_matrix)

    return EvaluationReport(
        task_type=TaskType.CLUSTERING,
        metrics=(
            _silhouette(
                scored_matrix, scored_labels, silhouette_row_limit, scored_rows, context
            ),
            _dense_only_metric("davies_bouldin", scored_matrix, scored_labels, is_sparse, context),
            _dense_only_metric(
                "calinski_harabasz", scored_matrix, scored_labels, is_sparse, context
            ),
        ),
        row_count=total_rows,
    )


def _partition_blocker(
    cluster_count: int, scored_rows: int, noise_rows: int, total_rows: int
) -> str | None:
    """Why no clustering metric can be computed, or ``None`` if they can.

    Every one of these is a real result rather than an error: a clusterer that
    found one cluster, or rejected every row, has told the analyst something. It
    just cannot be scored, and reporting the three metrics as ``UNDEFINED`` with
    the reason says both halves of that.

    The upper bound matters as much as the lower. scikit-learn refuses a
    labelling with as many clusters as rows -- with one row each there is no
    within-cluster distance -- and it raises the same ``ValueError`` it raises
    for one cluster, so the two are separated here rather than after the fact.
    """
    if scored_rows == 0:
        return (
            f"All {total_rows} rows were labelled noise, so there is no cluster "
            "to measure. That is a result about the data and the parameters -- "
            "for a density method, usually that eps is too small for the scale "
            "the features are on."
        )
    if cluster_count < 2:
        detail = (
            f" ({noise_rows} further rows were labelled noise)" if noise_rows else ""
        )
        return (
            f"The partition has {cluster_count} cluster"
            f"{'' if cluster_count == 1 else 's'}{detail}. Every one of these "
            "metrics compares each cluster with the others, so at least two are "
            "needed for any of them to have a value."
        )
    if cluster_count >= scored_rows:
        return (
            f"Each of the {scored_rows} scored rows is in a cluster of its own, "
            "so there is no within-cluster distance to compare against the "
            "between-cluster ones. A partition this fine describes the rows "
            "rather than any structure in them."
        )
    return None


def _readable_size(byte_count: float) -> str:
    """Format a byte count at a unit that makes it legible.

    Written because the obvious ``f"{n / 1e9:.1f} GB"`` reported a genuine
    2 MB matrix as "0.0 GB", which reads as a bug in the message rather than as
    the small number it is.
    """
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if byte_count >= scale:
            return f"{byte_count / scale:.1f} {unit}"
    return f"{int(byte_count)} bytes"


def _silhouette(
    X: Any,
    labels: np.ndarray,
    row_limit: int,
    scored_rows: int,
    context: dict[str, Any],
) -> MetricValue:
    """Mean silhouette coefficient, or the reason it was not computed.

    The row limit is checked before the call rather than after, because "after"
    for this metric means minutes of wall clock and gigabytes of memory that a
    caller never asked to spend.
    """
    if scored_rows > row_limit:
        return _unavailable(
            "silhouette",
            MetricStatus.NOT_APPLICABLE,
            f"Not computed: the silhouette needs every pairwise distance, so "
            f"{scored_rows} rows would require a {_readable_size(scored_rows ** 2 * 8)} "
            f"distance matrix. That is above the limit of {row_limit} rows. It was "
            "not approximated by sampling, because a sampled silhouette is not the "
            "silhouette and would differ between two runs of the same data. "
            "Raise silhouette_row_limit to compute it in full.",
        )
    from sklearn import metrics as skmetrics

    try:
        value = skmetrics.silhouette_score(X, labels)
    except (ValueError, MemoryError) as error:
        return _unavailable("silhouette", MetricStatus.FAILED, str(error))
    return _available("silhouette", value, **context)


def _dense_only_metric(
    name: str, X: Any, labels: np.ndarray, is_sparse: bool, context: dict[str, Any]
) -> MetricValue:
    """Davies-Bouldin or Calinski-Harabasz, both of which need a dense matrix.

    scikit-learn refuses a sparse matrix for these two while accepting one for
    the silhouette, so a model that declares ``supports_sparse_input`` can be
    scored on one metric and not the other two. Densifying to get past it is not
    done here: the matrix is sparse because one-hot encoding made it wide, and
    quietly materialising it is how an evaluation step becomes the thing that
    exhausts memory. The caller is told which form is needed instead.
    """
    if is_sparse:
        return _unavailable(
            name,
            MetricStatus.NOT_APPLICABLE,
            f"{name} is defined by scikit-learn for a dense matrix only, and this "
            "model was given a sparse one. It was not densified, because a sparse "
            "matrix here is usually a wide one-hot encoding and materialising it "
            "could cost far more memory than the metric is worth. The silhouette "
            "accepts sparse input and was attempted.",
        )
    from sklearn import metrics as skmetrics

    computation = {
        "davies_bouldin": skmetrics.davies_bouldin_score,
        "calinski_harabasz": skmetrics.calinski_harabasz_score,
    }[name]
    try:
        value = computation(np.asarray(X), labels)
    except (ValueError, MemoryError) as error:
        return _unavailable(name, MetricStatus.FAILED, str(error))
    return _available(name, value, **context)
