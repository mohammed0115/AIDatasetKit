"""Exception hierarchy for the whole library.

Every error raised by ``aidatasetkit`` derives from :class:`AIDatasetKitError`, so
callers can catch library failures without also swallowing unrelated bugs.

The library never silences a problem: invalid input raises rather than returning
``NaN``, an empty frame, or a silently-dropped row.
"""

from __future__ import annotations

__all__ = [
    "AIDatasetKitError",
    "ValidationError",
    "EmptyDataError",
    "ShapeError",
    "NonNumericDataError",
    "MissingValueError",
    "NonFiniteValueError",
    "DomainError",
    "ConfigurationError",
    "SchemaError",
    "TaskError",
    "AmbiguousTaskError",
    "UnsupportedTaskError",
    "IncompatibleModelError",
    "ModelError",
    "UnknownModelError",
    "DuplicateModelError",
    "InvalidModelParameterError",
    "AmbiguousModelAliasError",
    "MissingDependencyError",
    "VisualizationError",
    "InvalidVisualizationRequest",
    "UnsupportedChartError",
    "WorkflowStateError",
    "PredictionValidationError",
]


class AIDatasetKitError(Exception):
    """Base class for every error raised by this library."""


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


class ValidationError(AIDatasetKitError):
    """Input did not satisfy a documented precondition."""


class EmptyDataError(ValidationError):
    """Input contained no usable observations."""


class ShapeError(ValidationError):
    """Input had the wrong dimensionality."""


class NonNumericDataError(ValidationError):
    """Input could not be interpreted as numeric values."""


class MissingValueError(ValidationError):
    """Input contained missing values under ``nan_policy="raise"``."""


class NonFiniteValueError(ValidationError):
    """Input contained ``inf`` or ``-inf`` where finite values were required."""


class DomainError(ValidationError):
    """Values fell outside the mathematical domain of the requested operation.

    Raised, for example, by the geometric and harmonic means when the input
    contains non-positive values.
    """


# --------------------------------------------------------------------------- #
# Configuration and dataset structure
# --------------------------------------------------------------------------- #


class ConfigurationError(AIDatasetKitError):
    """A :class:`~aidatasetkit.core.config.KitConfig` value was invalid."""


class SchemaError(AIDatasetKitError):
    """The dataset structure was unusable (missing or duplicated columns)."""


# --------------------------------------------------------------------------- #
# Task resolution
# --------------------------------------------------------------------------- #


class TaskError(AIDatasetKitError):
    """Base class for machine-learning task resolution failures."""


class AmbiguousTaskError(TaskError):
    """The task could not be inferred from the target and no override was given.

    The library refuses to guess between, say, multiclass classification and
    regression for an integer target; pass ``task=`` explicitly instead.
    """


class UnsupportedTaskError(TaskError):
    """The requested task family is not implemented in this version."""


class IncompatibleModelError(TaskError):
    """A model's declared capabilities do not match the resolved task."""


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #


class ModelError(AIDatasetKitError):
    """Base class for model registry and construction failures."""


class UnknownModelError(ModelError):
    """No model is registered under the requested name or alias."""


class DuplicateModelError(ModelError):
    """A registration would overwrite or shadow an existing model.

    The registry never replaces silently: a clashing canonical name or an alias
    that could not be resolved unambiguously is refused at registration time.
    """


class InvalidModelParameterError(ModelError):
    """A parameter was passed that the underlying estimator does not accept."""


class AmbiguousModelAliasError(ModelError):
    """A short alias matched more than one model and no task narrowed it down."""


# --------------------------------------------------------------------------- #
# Optional dependencies
# --------------------------------------------------------------------------- #


class MissingDependencyError(AIDatasetKitError):
    """An optional package this feature needs is not installed.

    Raised by the model registry for an absent backend and by the renderer
    registry for an absent plotting library. It hangs directly off the base
    error rather than under :class:`ModelError`, because a missing plotting
    library is not a model failure and ``except ModelError`` should not swallow
    it.
    """


# --------------------------------------------------------------------------- #
# Workflow and output
# --------------------------------------------------------------------------- #


class VisualizationError(AIDatasetKitError):
    """Base class for visualization planning and rendering failures."""


class InvalidVisualizationRequest(VisualizationError):
    """A manually requested chart does not suit the data it was given.

    Raised instead of letting the plotting library fail deep inside its own
    stack, where the message names an array shape rather than the column.
    """


class UnsupportedChartError(VisualizationError):
    """The requested chart type is not implemented by the selected renderer."""


class WorkflowStateError(AIDatasetKitError):
    """A facade method was called before its prerequisites were satisfied."""


class PredictionValidationError(AIDatasetKitError):
    """A prediction frame failed its output contract."""
