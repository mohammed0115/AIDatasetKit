"""What a prediction run produced.

Predictions are the one output of this library that is *not* primarily a record.
An audit artifact exists to be read later; a prediction exists to be used now,
and turning a float64 array into a list of JSON numbers to satisfy a
serialisation contract nobody asked for would cost precision for nothing. So the
arrays stay arrays, and :meth:`PredictionResult.to_frame` is how they become
something to write out.

The one thing that is *not* left to the caller is the label. A classifier trained
on ``"churn"`` and ``"stay"`` learned integers, and handing those integers back
would make the user reverse a mapping this library chose. Predictions come back
in the caller's own vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aidatasetkit.core.exceptions import ValidationError
from aidatasetkit.core.types import TaskType

__all__ = ["PredictionResult"]


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Predictions for one frame, from one fitted model.

    Attributes:
        predictions: One value per row. For classification these are the
            *original* labels the caller trained on, never the encoded integers.
            For regression they are quantities, never passed through any encoder.
        task_type: The task the model serves.
        model_name: The canonical name of the model that produced them.
        row_count: How many rows were predicted.
        ids: The identifier column, aligned positionally with ``predictions``,
            or ``None`` if none was declared. Carried so a prediction can be
            joined back to the row it belongs to.
        id_name: The name that column had.
        probabilities: A frame of class probabilities whose **columns are the
            original class labels**, or ``None``. Named rather than positional
            because ``[:, 1]`` is the single most reliable way to report a
            confident number about the wrong class.
    """

    predictions: np.ndarray
    task_type: TaskType
    model_name: str
    row_count: int
    ids: np.ndarray | None = None
    id_name: str | None = None
    probabilities: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if len(self.predictions) != self.row_count:
            raise ValidationError(
                f"{self.model_name} produced {len(self.predictions)} prediction(s) "
                f"for {self.row_count} row(s). One of the two is wrong, and a "
                "prediction misaligned with its row is worse than no prediction."
            )
        if self.ids is not None and len(self.ids) != self.row_count:
            raise ValidationError(
                f"{len(self.ids)} identifier(s) were carried for "
                f"{self.row_count} prediction(s); they cannot be paired."
            )
        if self.probabilities is not None:
            if len(self.probabilities) != self.row_count:
                raise ValidationError(
                    f"{len(self.probabilities)} probability row(s) for "
                    f"{self.row_count} prediction(s)."
                )
            if self.task_type is not TaskType.CLASSIFICATION:
                raise ValidationError(
                    f"Class probabilities were attached to a "
                    f"{self.task_type.value} result, where there are no classes."
                )

    @property
    def class_labels(self) -> tuple[str, ...] | None:
        """The classes the probability columns name, in column order."""
        if self.probabilities is None:
            return None
        return tuple(str(column) for column in self.probabilities.columns)

    def to_frame(self) -> pd.DataFrame:
        """Return the predictions as a dataframe, ids first if there are any.

        A fresh frame every time, over **copies**. Handing the stored arrays to
        the constructor aliased them for object dtype -- which is every
        string-labelled classification run -- so editing the returned frame
        rewrote what the result itself reported.

        A collision between the identifier's name and an output column is an
        error rather than a silent overwrite: an id column called ``prediction``
        would otherwise vanish under the predictions, and the frame would look
        complete.
        """
        reserved = {"prediction"}
        if self.probabilities is not None:
            reserved |= {f"probability_{label}" for label in self.probabilities.columns}

        columns: dict[str, Any] = {}
        if self.ids is not None:
            name = self.id_name or "id"
            if name in reserved:
                raise ValidationError(
                    f"The identifier column is named {name!r}, which collides with "
                    "an output column of this frame. Rename it before predicting, "
                    "or the identifier would be overwritten by the prediction."
                )
            columns[name] = np.array(self.ids, copy=True)
        columns["prediction"] = np.array(self.predictions, copy=True)

        frame = pd.DataFrame(columns)
        if self.probabilities is not None:
            for label in self.probabilities.columns:
                frame[f"probability_{label}"] = self.probabilities[label].to_numpy(
                    copy=True
                )
        return frame

    def describe(self) -> dict[str, Any]:
        """Return a JSON-safe description of the run -- shape, not values.

        The predictions themselves are deliberately absent. They are data, and
        this is the summary a log or an audit trail can carry without becoming a
        copy of the answer.
        """
        return {
            "model_name": self.model_name,
            "task_type": self.task_type.value,
            "row_count": self.row_count,
            "has_ids": self.ids is not None,
            "id_name": self.id_name,
            "class_labels": (
                None if self.class_labels is None else list(self.class_labels)
            ),
        }
