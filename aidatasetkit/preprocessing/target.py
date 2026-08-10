"""Representing a classification target as integers, reversibly.

This is the *only* place a label encoder belongs. Encoding a feature that way
would tell a model that category 2 sits twice as far from zero as category 1,
which is a claim about the data that nothing in the data supports. Encoding the
*target* is different: the integers are names, and the model never does
arithmetic on them.

The positive label travels with the mapping. Once ``"churn"`` and ``"stay"``
become ``0`` and ``1``, ``predict_proba[:, 1]`` is the probability of whichever
label sorted second -- which may be exactly the wrong one. Recording both the
original and the encoded positive class is what stops a later metric from being
computed for the class nobody asked about.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from aidatasetkit.core.exceptions import PreprocessingError, UnsupportedTaskError
from aidatasetkit.core.types import TargetProfile, TaskType
from aidatasetkit.preprocessing.types import TargetEncoding

__all__ = ["TargetLabelEncoder"]


class TargetLabelEncoder:
    """Encodes and decodes a classification target.

    Args:
        positive_label: Which class counts as positive. Taken from the
            :class:`TargetProfile` when one is supplied to :meth:`fit` and the
            caller gives none.

    Example:
        >>> import pandas as pd
        >>> encoder = TargetLabelEncoder(positive_label="churn")
        >>> encoded = encoder.fit_transform(pd.Series(["stay", "churn", "stay"]))
        >>> encoder.encoding.positive_label_encoded
        0
        >>> list(encoder.inverse_transform(encoded))
        ['stay', 'churn', 'stay']
    """

    def __init__(self, positive_label: Any = None) -> None:
        self._positive_label = positive_label
        self._encoder: LabelEncoder | None = None
        self._encoding: TargetEncoding | None = None

    @property
    def encoding(self) -> TargetEncoding:
        """The learned mapping, in both directions.

        Raises:
            PreprocessingError: If called before :meth:`fit`.
        """
        if self._encoding is None:
            raise PreprocessingError(
                "This target encoder has not been fitted; call fit(y_train) first."
            )
        return self._encoding

    @property
    def classes_(self) -> np.ndarray:
        """The original class labels, in encoded order."""
        return np.asarray(self.encoding.classes, dtype=object)

    def fit(self, y: Any, target_profile: TargetProfile | None = None) -> TargetLabelEncoder:
        """Learn the mapping from the training labels only.

        Args:
            y: Training labels.
            target_profile: What the task detector concluded. A regression target
                is refused here rather than silently turned into class indices.

        Raises:
            UnsupportedTaskError: If the target is a regression target.
            PreprocessingError: If the labels are unusable, or the configured
                positive label is not among them.
        """
        if target_profile is not None and target_profile.task_type is not TaskType.CLASSIFICATION:
            raise UnsupportedTaskError(
                f"A {target_profile.task_type.value} target must not be label "
                "encoded: its values are quantities, and replacing them with class "
                "indices would destroy the thing being predicted. Pass it through "
                "unchanged."
            )

        series = _as_series(y)
        if target_profile is None:
            _refuse_continuous(series)
        if series.isna().any():
            raise PreprocessingError(
                f"The target contains {int(series.isna().sum())} missing value(s). "
                "Rows without a label cannot be trained on; drop or fill them "
                "deliberately first."
            )

        encoder = LabelEncoder()
        encoder.fit(series.to_numpy())
        classes = tuple(encoder.classes_.tolist())
        if len(classes) < 2:
            raise PreprocessingError(
                f"The target has {len(classes)} distinct value(s); classification "
                "needs at least two."
            )

        mapping = {label: index for index, label in enumerate(classes)}
        inverse = {index: label for label, index in mapping.items()}

        positive, encoded_positive, resolved = self._resolve_positive(
            classes, mapping, target_profile
        )

        self._encoder = encoder
        self._encoding = TargetEncoding(
            classes=classes,
            mapping=mapping,
            inverse=inverse,
            positive_label=positive,
            positive_label_encoded=encoded_positive,
            positive_label_resolved=resolved,
        )
        return self

    def transform(self, y: Any) -> np.ndarray:
        """Encode labels using the mapping learned from training.

        Raises:
            PreprocessingError: If a label was never seen during fitting.
        """
        if self._encoder is None:
            raise PreprocessingError(
                "This target encoder has not been fitted; call fit(y_train) first."
            )
        series = _as_series(y)
        known = set(self.encoding.classes)
        unseen = sorted({str(v) for v in series.dropna().unique() if v not in known})
        if unseen:
            raise PreprocessingError(
                f"The target contains label(s) {unseen[:5]} that were not present in "
                f"training. Known classes: {[str(c) for c in self.encoding.classes]}."
            )
        return self._encoder.transform(series.to_numpy())

    def fit_transform(self, y: Any, target_profile: TargetProfile | None = None) -> np.ndarray:
        """Fit on these labels and return them encoded."""
        return self.fit(y, target_profile).transform(y)

    def inverse_transform(self, y: Any) -> np.ndarray:
        """Turn encoded values back into the caller's original labels."""
        if self._encoder is None:
            raise PreprocessingError(
                "This target encoder has not been fitted; call fit(y_train) first."
            )
        return self._encoder.inverse_transform(np.asarray(y).astype(int))

    def positive_column_index(self) -> int:
        """Which ``predict_proba`` column holds the positive class.

        Raises:
            PreprocessingError: If no positive label was resolved, in which case
                assuming column 1 would be a guess.
        """
        encoding = self.encoding
        if not encoding.positive_label_resolved or encoding.positive_label_encoded is None:
            raise PreprocessingError(
                "No positive label was resolved for this target, so no probability "
                f"column can be called the positive one. Classes are "
                f"{[str(c) for c in encoding.classes]}; construct the encoder with "
                "positive_label=... to say which is the event of interest."
            )
        return int(encoding.positive_label_encoded)

    def _resolve_positive(
        self,
        classes: tuple[Any, ...],
        mapping: dict[Any, int],
        target_profile: TargetProfile | None,
    ) -> tuple[Any, int | None, bool]:
        """Decide which class is positive, preferring what the caller said."""
        candidate = self._positive_label
        source_resolved = candidate is not None

        if candidate is None and target_profile is not None:
            if target_profile.positive_label_resolved:
                candidate = target_profile.positive_label
                source_resolved = candidate is not None

        if candidate is None:
            return None, None, False
        if candidate not in mapping:
            raise PreprocessingError(
                f"positive_label={candidate!r} is not one of the target's classes "
                f"{[str(c) for c in classes]}."
            )
        return candidate, mapping[candidate], source_resolved


def _refuse_continuous(series: pd.Series) -> None:
    """Stop an obviously continuous target from becoming class indices.

    A :class:`TargetProfile` settles the question properly, and passing one is the
    reliable route. Without it, this catches the case that cannot be anything
    else: a float column holding values that are not whole numbers. Turning
    those into 0..n-1 would silently destroy the quantity being predicted.
    """
    if not pd.api.types.is_float_dtype(series):
        return
    values = series.dropna().to_numpy(dtype="float64", na_value=np.nan)
    if values.size and not np.all(values == np.floor(values)):
        raise UnsupportedTaskError(
            "This target holds non-integral numbers, so it looks like a regression "
            "target and must not be label encoded: its values are quantities, and "
            "replacing them with class indices would destroy what is being "
            "predicted. Pass a TargetProfile if it really is a classification "
            "target with numeric labels."
        )


def _as_series(y: Any) -> pd.Series:
    """Wrap the target without copying or modifying the caller's object."""
    if isinstance(y, pd.Series):
        return y
    if isinstance(y, pd.DataFrame):
        raise PreprocessingError(
            "The target must be one-dimensional; a DataFrame was given. Select a "
            "single column first."
        )
    array = np.asarray(y)
    if array.ndim != 1:
        raise PreprocessingError(
            f"The target must be one-dimensional, got {array.ndim} dimensions."
        )
    return pd.Series(array)
