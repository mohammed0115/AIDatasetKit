"""Applying a fitted model to rows it has never seen.

Everything here is *application*. The preprocessor is transformed with, never
fitted; the target encoder is inverted, never re-learned; the estimator is
called, never rebuilt. A frame arriving at this module cannot change anything a
model knows, and that is the whole reason inference is a separate step rather
than a convenience method that happens to refit.

Two decisions carry the weight.

**A classification prediction comes back in the caller's vocabulary.** The model
answers with integers because that is what it was trained on, and this module
inverts the mapping the target encoder recorded. Handing back ``0`` for a model
trained on ``"churn"`` would make the user reverse a decision this library made.

**A probability column is named, not numbered.** ``predict_proba`` returns
columns in the estimator's ``classes_`` order, which is the encoded order, which
for ``"churn"``/``"stay"`` puts the interesting class first. Labelling the
columns with the original classes is what stops ``[:, 1]`` from being read as
"the positive one".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from aidatasetkit.core.exceptions import PredictionValidationError
from aidatasetkit.core.types import TaskType
from aidatasetkit.prediction.types import PredictionResult

__all__ = ["predict_frame"]


def predict_frame(
    training: Any,
    frame: pd.DataFrame,
    *,
    id_column: str | None = None,
    with_probabilities: bool = False,
) -> PredictionResult:
    """Predict for ``frame`` using an already-fitted training result.

    Args:
        training: A :class:`~aidatasetkit.training.trainer.TrainingResult`. Its
            fitted preprocessor and estimator are *used*, never refitted.
        frame: Rows to predict for. Read, never modified. The target column may
            be present or absent -- inference does not need the answer, and if it
            is there it is ignored rather than consulted.
        id_column: A column carried through beside the predictions so a row can
            be identified afterwards. It is not a feature and not an index; the
            preprocessor selects the columns it was fitted for and this one is
            simply read alongside.
        with_probabilities: Attach class probabilities, if the model declares it
            can produce them. Asked of the declared capability rather than of the
            fitted object.

    Returns:
        A :class:`PredictionResult`.

    Raises:
        PredictionValidationError: If the frame has no rows, or the identifier
            column is missing.
    """
    if not isinstance(frame, pd.DataFrame):
        raise PredictionValidationError(
            f"A pandas DataFrame is required to predict from, got "
            f"{type(frame).__name__}."
        )
    if not len(frame):
        raise PredictionValidationError(
            "The frame has no rows, so there is nothing to predict for."
        )
    if id_column is not None and id_column not in frame.columns:
        raise PredictionValidationError(
            f"The identifier column {id_column!r} is not in this frame. Available: "
            f"{[str(c) for c in list(frame.columns)[:20]]}."
        )

    # transform, never fit_transform. An unseen category follows the vocabulary
    # learned in training and becomes all zeros; it does not join it.
    matrix = training.preprocessor.transform(frame)
    raw = np.asarray(training.estimator.predict(matrix))

    ids = None
    if id_column is not None:
        # Positional, deliberately. A duplicated or non-monotonic index label
        # would otherwise decide which prediction belongs to which row, and
        # pandas would align them silently.
        ids = frame[id_column].to_numpy(copy=True)

    if training.task_type is TaskType.REGRESSION:
        return PredictionResult(
            predictions=raw,
            task_type=TaskType.REGRESSION,
            model_name=training.model_name,
            row_count=int(len(frame)),
            ids=ids,
            id_name=id_column,
        )

    encoder = training.target_encoding
    if encoder is None:  # pragma: no cover - a classifier always carries one
        raise PredictionValidationError(
            f"{training.model_name} was trained for classification but carries no "
            "target encoding, so its integer answers cannot be turned back into "
            "labels."
        )

    return PredictionResult(
        predictions=np.asarray(encoder.inverse_transform(raw)),
        task_type=TaskType.CLASSIFICATION,
        model_name=training.model_name,
        row_count=int(len(frame)),
        ids=ids,
        id_name=id_column,
        probabilities=_probabilities(training, matrix, with_probabilities),
    )


def _probabilities(training: Any, matrix: Any, wanted: bool) -> pd.DataFrame | None:
    """Class probabilities with the original labels as column names.

    ``None`` rather than an exception when the model has none: "this model cannot
    do that" is an answer, and a caller asking for probabilities across several
    models should not have to know in advance which of them oblige.
    """
    if not wanted or not training.supports_predict_proba:
        return None
    if not hasattr(training.estimator, "predict_proba"):  # pragma: no cover
        return None

    values = np.asarray(training.estimator.predict_proba(matrix))
    encoder = training.target_encoding
    # The estimator's column order is its classes_ order, which is the encoded
    # order, which is the order the encoder recorded. Reading the labels back
    # through the encoder is what keeps the name attached to the right column.
    labels = [str(label) for label in encoder.classes_]
    if values.shape[1] != len(labels):  # pragma: no cover - defensive
        raise PredictionValidationError(
            f"The model returned {values.shape[1]} probability column(s) for "
            f"{len(labels)} class(es), so no column can be named reliably."
        )
    return pd.DataFrame(values, columns=labels)
