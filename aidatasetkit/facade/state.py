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

from enum import StrEnum

__all__ = ["Stage"]


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
