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
from aidatasetkit.core.schema import target_holds_quantities
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

        _require_uniform_labels(series)
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
        if series.isna().any():
            # The unseen-label check below drops NaN before comparing, so without
            # this a null label reaches LabelEncoder and comes back as "previously
            # unseen labels: nan" -- which reads as a vocabulary problem rather
            # than a row with no label at all.
            raise PreprocessingError(
                f"The target contains {int(series.isna().sum())} missing value(s). "
                "A row without a label cannot be encoded; drop or fill them "
                "deliberately first."
            )
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
        """Turn encoded values back into the caller's original labels.

        Only whole numbers inside the class range are accepted. ``1`` and ``1.0``
        both name class 1; ``1.9`` names nothing, and casting it to an ``int``
        would answer "class 1" to a question that had no valid answer.

        Raises:
            PreprocessingError: If a value is missing, infinite, not a whole
                number, or outside the range of class indices.
        """
        if self._encoder is None:
            raise PreprocessingError(
                "This target encoder has not been fitted; call fit(y_train) first."
            )
        codes = self._validate_codes(y)
        return self._encoder.inverse_transform(codes)

    def _validate_codes(self, y: Any) -> np.ndarray:
        """Check every value names an actual class, and return it as an index."""
        try:
            values = np.asarray(y, dtype="float64")
        except (TypeError, ValueError, OverflowError) as error:
            # OverflowError is numpy's answer to an int too large for float64. It
            # is not a subclass of the other two, and uncaught it escapes as a
            # bare "int too large to convert to float".
            raise PreprocessingError(
                f"Encoded class labels must be numbers within the range of a class "
                f"index, got {type(y).__name__}: {error}"
            ) from error

        # A single code is not ambiguous the way a matrix of them is, so it is
        # read as a sequence of one rather than refused.
        values = np.atleast_1d(values)
        if values.ndim > 1:
            raise PreprocessingError(
                f"Encoded class labels must be one-dimensional, got {values.ndim} "
                "dimensions."
            )
        if not np.all(np.isfinite(values)):
            offenders = values[~np.isfinite(values)][:5]
            raise PreprocessingError(
                f"Encoded class labels contain non-finite value(s) {offenders.tolist()}. "
                "Every value must name a class."
            )
        if not np.all(values == np.floor(values)):
            offenders = sorted({float(v) for v in values[values != np.floor(values)]})[:5]
            raise PreprocessingError(
                f"Encoded class labels must be whole numbers, got {offenders}. "
                "Rounding them here would answer with a class the caller never named; "
                "decode a prediction, not a probability."
            )

        count = len(self.encoding.classes)
        out_of_range = sorted({int(v) for v in values if not 0 <= v < count})[:5]
        if out_of_range:
            raise PreprocessingError(
                f"Encoded class label(s) {out_of_range} are outside the range of the "
                f"{count} class(es) learned in training (0 to {count - 1})."
            )
        return values.astype("int64")

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


def _require_uniform_labels(series: pd.Series) -> None:
    """Refuse a target that mixes strings with numbers.

    scikit-learn answers this with "Encoders require their input argument must be
    uniformly strings or numbers", which says nothing about which column, which
    values, or what to do. Casting the numbers to strings would look like a fix
    and is not: ``1`` and ``"1"`` are then one class, and nobody decided that.
    """
    numeric_types = (int, float, complex, np.number)
    values = series.to_numpy()
    if len({isinstance(value, numeric_types) for value in values}) > 1:
        numbers = sorted({repr(v) for v in values if isinstance(v, numeric_types)})
        text = sorted({repr(v) for v in values if not isinstance(v, numeric_types)})
        raise PreprocessingError(
            f"The target mixes numbers ({', '.join(numbers[:3])}) with text "
            f"({', '.join(text[:3])}). Which class a value belongs to would depend on "
            "how it was written down, so the labels have to be made one kind or the "
            "other deliberately -- 1 and '1' are not obviously the same class."
        )


def _refuse_continuous(series: pd.Series) -> None:
    """Stop a continuous target from becoming class indices.

    A :class:`TargetProfile` settles the question properly and passing one is the
    reliable route. Without it this has to answer the question itself -- and the
    answer has to be the *same* answer, which is why it comes from
    :func:`~aidatasetkit.core.schema.target_holds_quantities` rather than from
    rules written here.

    It used to be written here, and the two drifted in three separate ways. The
    guard tested ``is_float_dtype``, so a column of Python floats under ``object``
    dtype walked past the very case its docstring claimed to catch. It looked only
    for non-integral values, so an integer column the detector itself resolved as
    regression -- sixty distinct readings, well above the class limit -- was
    encoded into ``0..59``. And nothing anywhere compared the two components'
    conclusions, so the library could say *regression* in one breath and hand back
    class indices in the next, with no error raised at any point.

    Deciding this from a shared function is the fix. The thresholds it applies are
    the detector's own, so the two cannot answer differently again.

    Raises:
        UnsupportedTaskError: If the values are quantities.
    """
    if not target_holds_quantities(series):
        return
    raise UnsupportedTaskError(
        "This target holds quantities rather than labels, so it must not be "
        "label encoded: replacing each value with a class index would destroy "
        "the thing being predicted. The same values passed to TaskDetector "
        "resolve as a regression target. Pass that TargetProfile through if you "
        "want the check made explicitly, or pass the labels themselves if the "
        "column you meant is a different one."
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
