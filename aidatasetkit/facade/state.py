"""What a facade has done so far, and what that entitles it to do next.

The workflow has real dependencies -- you cannot evaluate a model nobody trained
-- and it has an obvious *tutorial* order that is not the same thing. Profiling,
describing and checking quality are three independent readings of one loaded
frame; requiring them in a fixed sequence would be ceremony, and ceremony is what
makes a library feel like it is hiding something.

So this records what exists, not what step the user is on. A method asks whether
its prerequisites are present and says which one is missing when they are not.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = ["FinalEvaluation", "Stage"]


class Stage(StrEnum):
    """A named milestone, used for reporting rather than for gatekeeping.

    Gatekeeping is done by asking whether the specific thing a method needs
    exists. This enum answers the different question a person asks when they look
    at a session and want to know roughly where they are.
    """

    #: Constructed; no data yet.
    EMPTY = "empty"
    #: Training data loaded and the task resolved.
    LOADED = "loaded"
    #: Preparation validated and plans derived. Nothing fitted.
    PREPARED = "prepared"
    #: A comparison has been run.
    COMPARED = "compared"
    #: A model has been chosen by a person.
    MODEL_SELECTED = "model_selected"
    #: That model has been fitted.
    TRAINED = "trained"
    #: It has been measured on held-out rows.
    EVALUATED = "evaluated"
    #: It has been measured, once, on the external test frame. The experiment
    #: is frozen: nothing that could choose or refit a model is allowed again
    #: until new data is loaded.
    FINAL_EVALUATED = "final_evaluated"


@dataclass(frozen=True, slots=True)
class FinalEvaluation:
    """The one measurement taken on the external test frame.

    Attributes:
        model_name: The canonical name of the model that was frozen and measured.
        report: Every metric, computed on the test rows with the fitted
            preprocessor applied and never refitted.
        test_rows: How many test rows produced it.
        rows_also_in_training: Test rows that are exact copies of a row in the
            training frame -- every column, dtype included. A copy is a row the
            model may have learned, so a non-zero count means the estimate is
            optimistic by an amount this number bounds. ``None`` when the frames
            share no columns or hold values that cannot be hashed.
        validation_used_for_selection: Whether a comparison ranked models on the
            same validation rows the trained model was later measured on -- the
            reason the validation score is not an independent estimate and this
            one is.
    """

    model_name: str
    report: Any
    test_rows: int
    rows_also_in_training: int | None
    validation_used_for_selection: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the measurement."""
        return {
            "model_name": self.model_name,
            "test_rows": self.test_rows,
            "rows_also_in_training": self.rows_also_in_training,
            "validation_used_for_selection": self.validation_used_for_selection,
            "report": self.report.to_dict(),
        }
