"""Models: representing, registering, discovering, and constructing estimators.

Five responsibilities, deliberately kept apart:

==========================  ===================================================
Component                   Answers
==========================  ===================================================
:class:`ModelStrategy`      How AIDatasetKit constructs a model
:class:`ModelCapabilities`  What the model can do and what it needs
:class:`ModelRegistry`      What models AIDatasetKit knows about
:class:`ModelFactory`       How a request becomes a fresh estimator
``PreprocessingProfile``    Which requirements change pipeline assembly
==========================  ===================================================

Importing this package registers every built-in model. That happens through the
explicit strategy imports below, so the answer to "does this model exist?" never
depends on which module the caller imported first.

Nothing here trains, evaluates, ranks, or selects. The factory constructs exactly
what was asked for.
"""

from aidatasetkit.models import classification as _classification  # noqa: F401
from aidatasetkit.models import clustering as _clustering  # noqa: F401
from aidatasetkit.models import regression as _regression  # noqa: F401
from aidatasetkit.models.base import ModelStrategy
from aidatasetkit.models.capabilities import ModelCapabilities
from aidatasetkit.models.factory import ModelFactory
from aidatasetkit.models.registry import (
    ModelRegistration,
    ModelRegistry,
    default_registry,
    register_model,
)

__all__ = [
    "ModelCapabilities",
    "ModelFactory",
    "ModelRegistration",
    "ModelRegistry",
    "ModelStrategy",
    "default_registry",
    "register_model",
]
