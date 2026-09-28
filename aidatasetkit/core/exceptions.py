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
    "PreprocessingError",
    "AmbiguousFeatureRoleError",
    "MissingOrdinalOrderError",
    "NoUsableFeaturesError",
    "VisualizationError",
    "InvalidVisualizationRequest",
    "UnsupportedChartError",
    "EvidenceError",
    "SerializationError",
    "PublicationError",
    "NoPublishedRunError",
    "CorruptPublicationError",
    "IngestionError",
    "InputNotFoundError",
    "UnsupportedFormatError",
    "InvalidIngestionOptionsError",
    "EncodingError",
    "MalformedInputError",
    "AmbiguousDelimiterError",
    "EmptyInputError",
    "DuplicateHeadersError",
    "TrainingError",
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


class PreprocessingError(AIDatasetKitError):
    """Base class for preprocessing planning and building failures."""


class AmbiguousFeatureRoleError(PreprocessingError):
    """A feature was assigned two incompatible roles.

    Raised when overrides contradict each other -- a column declared both
    numeric and ordinal, say -- rather than letting one silently win.
    """


class MissingOrdinalOrderError(PreprocessingError):
    """A feature was declared ordinal without the order its encoding needs.

    The library never infers an ordering from the alphabet, from first
    appearance, or from frequency: those produce a number that looks like a
    measurement and means nothing.
    """


class NoUsableFeaturesError(PreprocessingError):
    """Every column was excluded or held back for review, leaving nothing to fit."""


class VisualizationError(AIDatasetKitError):
    """Base class for visualization planning and rendering failures."""


class InvalidVisualizationRequest(VisualizationError):
    """A manually requested chart does not suit the data it was given.

    Raised instead of letting the plotting library fail deep inside its own
    stack, where the message names an array shape rather than the column.
    """


class UnsupportedChartError(VisualizationError):
    """The requested chart type is not implemented by the selected renderer."""


class EvidenceError(AIDatasetKitError):
    """Raised when an audit artifact cannot be built or recorded truthfully."""


class SerializationError(EvidenceError):
    """Raised when a value has no honest canonical form in an artifact.

    Almost always an object that should never have reached the artifact at all --
    an estimator, a DataFrame, an array. Coercing it to text would put something
    that looks like a record into a document whose whole purpose is to be one.
    """


class PublicationError(EvidenceError):
    """Raised when a run's artifacts could not be published as one complete set.

    Whatever was published before is left exactly as it was: the failure happened
    before the single step that makes a new run current, or that step itself did
    not complete.
    """


class NoPublishedRunError(PublicationError):
    """Raised when an output directory has no completed run to read."""


class CorruptPublicationError(PublicationError):
    """Raised when the current run does not match its own manifest.

    A missing, extra, resized or altered file, a manifest that is not the one the
    pointer names, or a pointer that cannot be read. The set is refused whole: a
    reader handed part of a run, or a run from two generations, would be handed a
    record that never existed.
    """


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


class IngestionError(AIDatasetKitError):
    """A table could not be loaded as the one it claims to be.

    Every refusal of :func:`aidatasetkit.ingestion.load_table` is one of these.
    Messages name the cause and what to do about it; none of them quotes a cell
    value, because the input may be anyone's data.
    """


class InputNotFoundError(IngestionError):
    """The path to load does not exist, or is not a regular file."""


class UnsupportedFormatError(IngestionError):
    """The source is of a kind or format this version cannot load."""


class InvalidIngestionOptionsError(IngestionError, ConfigurationError):
    """A :class:`~aidatasetkit.ingestion.LoadOptions` value is invalid or does not apply."""


class EncodingError(IngestionError):
    """The bytes could not be decoded with the encoding in force."""


class MalformedInputError(IngestionError):
    """The content is not a well-formed table of the declared format.

    Inconsistent field counts, broken quoting, binary content, or two parsers
    disagreeing about how many rows there are. Refused rather than repaired: a
    table guessed out of a broken file would be evidence about a file nobody has.
    """


class AmbiguousDelimiterError(MalformedInputError):
    """More than one supported delimiter yields a consistent table.

    Picking one would be a guess with a coin-flip chance of silently producing
    the wrong columns. The caller says which one instead.
    """


class EmptyInputError(IngestionError, EmptyDataError):
    """There is no data row to load: no bytes, only whitespace, or only a header."""


class DuplicateHeadersError(IngestionError, SchemaError):
    """Two columns share a name.

    Detected before pandas can rename them to ``a.1``, ``a.2``: an audit of
    renamed columns would describe a dataset that does not exist.
    """


class TrainingError(AIDatasetKitError):
    """Raised when a run cannot be trained, split, or compared as asked.

    Conditions this library owns: too few rows to divide, a split that would
    leave one side empty, evaluation data whose columns do not match what
    training saw, a model that cannot serve the resolved target. Not for a
    backend's own refusal -- an estimator that declines its input raises its own
    error, and wrapping every one of those would hide real defects behind a
    uniform message.
    """


class WorkflowStateError(AIDatasetKitError):
    """A facade method was called before its prerequisites were satisfied."""


class PredictionValidationError(AIDatasetKitError):
    """A prediction frame failed its output contract."""
