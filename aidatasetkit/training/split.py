"""Deciding which rows a model learns from, and which rows judge it.

The split is the experimental boundary. Everything downstream depends on it being
drawn once, drawn deterministically, and never crossed -- so it is a first-class
object here rather than a pair of frames passed around by convention.

Three properties are enforced rather than assumed:

**Every row has exactly one role.** The two index arrays are checked to be
disjoint and to account for every row. A split that quietly dropped a row would
train on less data than it reported; one that duplicated a row would evaluate on
data it had trained on.

**Rows are identified positionally, not by index label.** A caller's frame may
carry any index -- strings, gaps, duplicates, or labels that overlap between two
frames they consider separate. Selecting by label would then align the wrong
rows, and pandas would do it silently. Everything here uses ``iloc``.

**A class distribution is preserved where that is meaningful, and not pretended
otherwise.** Classification splits stratify so a rare class is present on both
sides. Regression does not: there is nothing to stratify on, and quantile
bucketing would be a policy this library has not published. Saying so is better
than a stratification that only appears to exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aidatasetkit.core.config import KitConfig
from aidatasetkit.core.counting import value_counts
from aidatasetkit.core.exceptions import TrainingError
from aidatasetkit.core.types import TaskType

__all__ = ["DataSplit", "split_rows"]


@dataclass(frozen=True, slots=True)
class DataSplit:
    """Which rows train, which rows evaluate, and how that was decided.

    Attributes:
        train_positions: Positional indices of the training rows.
        evaluation_positions: Positional indices of the evaluation rows.
        stratified: Whether the class distribution was preserved.
        validation_size: The configured evaluation fraction.
        random_state: The seed the split was drawn with.
        note: How the split was reached, in a sentence, for the record.
    """

    train_positions: np.ndarray
    evaluation_positions: np.ndarray
    stratified: bool
    validation_size: float
    random_state: int
    note: str

    @property
    def train_count(self) -> int:
        """How many rows the model learns from."""
        return int(len(self.train_positions))

    @property
    def evaluation_count(self) -> int:
        """How many rows judge it."""
        return int(len(self.evaluation_positions))

    @property
    def fingerprint(self) -> str:
        """A digest of the row identities on each side.

        Two comparison runs that claim to have used the same evaluation rows can
        be checked against each other with this. Equal row *counts* prove nothing
        -- the whole point of a fair comparison is that the rows themselves match.
        """
        import hashlib

        digest = hashlib.sha256()
        for label, positions in (
            (b"train", self.train_positions),
            (b"evaluation", self.evaluation_positions),
        ):
            digest.update(label)
            digest.update(np.asarray(positions, dtype="int64").tobytes())
        return digest.hexdigest()[:16]

    def take(self, data: pd.DataFrame | pd.Series, *, evaluation: bool = False) -> Any:
        """Return one side of the split, selected positionally.

        Args:
            data: A frame or series with one row per original row.
            evaluation: Take the evaluation rows rather than the training rows.

        Raises:
            TrainingError: If ``data`` is not the length the split was drawn for.
                A silent mismatch here would align predictions against the wrong
                targets.
        """
        expected = self.train_count + self.evaluation_count
        if len(data) != expected:
            raise TrainingError(
                f"This split was drawn for {expected} rows but was given "
                f"{len(data)}. The split and the data it selects from have to be "
                "the same length, or the rows selected are not the rows meant."
            )
        positions = self.evaluation_positions if evaluation else self.train_positions
        return data.iloc[positions]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the split's shape.

        The row *identities* are deliberately absent: they are data, and the
        digest already answers "were these the same rows" without carrying them.
        """
        return {
            "train_rows": self.train_count,
            "evaluation_rows": self.evaluation_count,
            "stratified": self.stratified,
            "validation_size": self.validation_size,
            "random_state": self.random_state,
            "fingerprint": self.fingerprint,
            "note": self.note,
        }


def split_rows(
    y: pd.Series,
    task_type: TaskType,
    config: KitConfig | None = None,
) -> DataSplit:
    """Draw the one split every model in a run will share.

    Args:
        y: The target, one value per row. Read for its length and, for
            classification, for its class distribution. Never modified.
        task_type: Decides whether stratification is meaningful.
        config: Supplies ``validation_size``, ``random_state`` and ``shuffle``.
            These already existed for this purpose; nothing new is invented here.

    Returns:
        A :class:`DataSplit`.

    Raises:
        TrainingError: If there are too few rows to divide, or if a class is too
            rare to appear on both sides of a stratified split.
    """
    from sklearn.model_selection import train_test_split

    settings = config if config is not None else KitConfig()
    total = int(len(y))

    if total < 2:
        raise TrainingError(
            f"A train/evaluation split needs at least 2 rows and this data has "
            f"{total}. Supply more rows, or an explicit evaluation set."
        )

    missing = int(y.isna().sum())
    if missing:
        # Caught here rather than several steps later. A row with no label cannot
        # train and cannot judge, so assigning it a side is deciding something
        # about data that has nothing to say -- and the failure otherwise surfaces
        # from inside a backend, naming an array rather than the target.
        raise TrainingError(
            f"The target holds {missing} missing value(s), so those rows can "
            "neither be trained on nor evaluated against. Drop or fill them "
            "deliberately before splitting -- doing it here would be a decision "
            "about your data made without asking."
        )

    positions = np.arange(total)
    stratify, note = _stratification(y, task_type, settings, total)

    try:
        train, evaluation = train_test_split(
            positions,
            test_size=settings.validation_size,
            random_state=settings.random_state,
            shuffle=settings.shuffle,
            stratify=stratify,
        )
    except ValueError as error:
        raise TrainingError(
            f"The rows could not be divided into a training and an evaluation "
            f"set at validation_size={settings.validation_size}: {error}"
        ) from error

    if not len(train) or not len(evaluation):
        raise TrainingError(
            f"Splitting {total} row(s) at validation_size="
            f"{settings.validation_size} left one side empty "
            f"({len(train)} train, {len(evaluation)} evaluation). A model cannot "
            "be trained on nothing, nor judged on nothing."
        )

    # Frozen after sorting. The dataclass is immutable but a numpy array handed
    # out by reference is not, and this object's fingerprint exists to let two
    # runs prove they used the same rows -- a claim worth nothing if the rows can
    # be edited afterwards.
    train_positions = np.sort(train)
    evaluation_positions = np.sort(evaluation)
    train_positions.flags.writeable = False
    evaluation_positions.flags.writeable = False

    split = DataSplit(
        train_positions=train_positions,
        evaluation_positions=evaluation_positions,
        stratified=_really_stratified(y, train_positions, evaluation_positions, stratify),
        validation_size=float(settings.validation_size),
        random_state=int(settings.random_state),
        note=note,
    )
    _require_every_row_has_one_role(split, total)
    return split


def _really_stratified(
    y: pd.Series, train: np.ndarray, evaluation: np.ndarray, stratify: Any
) -> bool:
    """Whether the class distribution was *actually* preserved.

    Asking for stratification is not the same as getting it. scikit-learn
    allocates per-class evaluation rows by rounding, and a class rare enough can
    round to zero on one side -- leaving a split that reports ``stratified=True``
    while a class is missing from the evaluation set entirely. Since the flag is
    what a reader trusts when a per-class metric looks strange, it records the
    outcome rather than the request.
    """
    if stratify is None:
        return False
    return set(y.iloc[train].unique()) == set(y.iloc[evaluation].unique())


def _stratification(
    y: pd.Series, task_type: TaskType, settings: KitConfig, total: int
) -> tuple[Any, str]:
    """Decide whether to preserve the class distribution, and say why.

    Stratification is refused rather than attempted when a class is too rare for
    both sides to hold one, because scikit-learn's message for that names a
    minimum group size and not the class, and because an unstratified split of
    such data is a defensible outcome the caller should be told about.
    """
    if task_type is not TaskType.CLASSIFICATION:
        return None, (
            "rows shuffled and divided without stratification; a continuous "
            "target has no classes to preserve, and bucketing one into strata "
            "would be a policy this version does not publish"
        )

    if not settings.shuffle:
        return None, (
            "rows divided in their original order because shuffle is off; "
            "stratification needs a shuffle to draw from"
        )

    counts = value_counts(y)
    rarest = int(counts.min())
    if rarest < 2:
        label = counts.idxmin()
        return None, (
            f"the class distribution could not be preserved: {label!r} appears "
            f"{rarest} time(s), and stratifying needs at least two so that one "
            "can fall on each side"
        )

    evaluation_rows = int(round(total * settings.validation_size))
    if evaluation_rows < len(counts) or (total - evaluation_rows) < len(counts):
        return None, (
            f"the class distribution could not be preserved: {len(counts)} "
            f"classes do not fit in an evaluation set of {evaluation_rows} row(s)"
        )

    return y, (
        f"rows shuffled and divided with the class distribution preserved across "
        f"{len(counts)} classes"
    )


def _require_every_row_has_one_role(split: DataSplit, total: int) -> None:
    """No row lost, no row in both places.

    Cheap to check and impossible to notice otherwise: a duplicated row means a
    model is judged on something it learned from, and a dropped row means the
    reported training size is a fiction.
    """
    train = set(split.train_positions.tolist())
    evaluation = set(split.evaluation_positions.tolist())

    overlap = train & evaluation
    if overlap:
        raise TrainingError(
            f"{len(overlap)} row(s) were assigned to both the training and the "
            "evaluation set. A model judged on rows it learned from is not being "
            "judged at all."
        )
    if len(train) + len(evaluation) != total:
        missing = total - len(train | evaluation)
        raise TrainingError(
            f"The split accounts for {len(train) + len(evaluation)} of {total} "
            f"row(s); {missing} would have no role. Every row must train or "
            "evaluate."
        )
