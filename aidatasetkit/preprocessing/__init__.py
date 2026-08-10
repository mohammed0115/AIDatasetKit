"""Preprocessing: from what is known about a dataset to a fitted transformation.

Five responsibilities, deliberately separate:

=============================  ==================================================
Component                      Answers
=============================  ==================================================
:class:`FeatureDetector`       What kind of feature is this?
:class:`PreprocessingPlanner`  What should happen to it, and why?
:class:`PreprocessorBuilder`   How is that expressed as transformers?
:class:`FittedPreprocessor`    What was learned from the training rows?
:class:`TargetLabelEncoder`    How is the target represented, reversibly?
=============================  ==================================================

Two rules shape everything here.

**Preprocessing follows the model's capabilities, not its name.** Scaling is
added because a model declares it needs scaling; sparsity is preserved because a
model declares it accepts sparse input. Two models with the same
:class:`~aidatasetkit.core.types.PreprocessingProfile` share one blueprint, and a
model added later inherits the right treatment without anyone editing a
conditional.

**The library stops rather than guesses.** An ordinal order is never inferred
from the alphabet; numbers stored as text are never quietly parsed; a datetime is
never turned into an integer; an infinity is never replaced; a probable
identifier is never fed to a model. Each of those is reported in the plan with
the reason and what would settle it. Nothing is dropped silently, and the
caller's data is never modified.
"""

from aidatasetkit.preprocessing.builder import (
    BlueprintCache,
    FittedPreprocessor,
    PreprocessorBuilder,
)
from aidatasetkit.preprocessing.config import (
    CategoricalImputation,
    HighCardinalityPolicy,
    NumericImputation,
    NumericScaler,
    NumericTextPolicy,
    PreprocessingConfig,
    UnknownOrdinalPolicy,
)
from aidatasetkit.preprocessing.feature_detector import FeatureDetector
from aidatasetkit.preprocessing.plan import PreprocessingPlanner
from aidatasetkit.preprocessing.target import TargetLabelEncoder
from aidatasetkit.preprocessing.transformers import (
    CategoricalCaster,
    ExplicitMappingEncoder,
    NumericCaster,
    NumericTextConverter,
    OrdinalDomainGuard,
)
from aidatasetkit.preprocessing.types import (
    FeatureAction,
    FeatureDecision,
    FeatureRole,
    FeatureSpec,
    LabelNormalisation,
    PreprocessingPlan,
    TargetEncoding,
)

__all__ = [
    "BlueprintCache",
    "CategoricalCaster",
    "CategoricalImputation",
    "ExplicitMappingEncoder",
    "FeatureAction",
    "FeatureDecision",
    "FeatureDetector",
    "FeatureRole",
    "FeatureSpec",
    "FittedPreprocessor",
    "HighCardinalityPolicy",
    "LabelNormalisation",
    "NumericCaster",
    "NumericImputation",
    "NumericScaler",
    "NumericTextConverter",
    "NumericTextPolicy",
    "OrdinalDomainGuard",
    "PreprocessingConfig",
    "PreprocessingPlan",
    "PreprocessingPlanner",
    "PreprocessorBuilder",
    "TargetEncoding",
    "TargetLabelEncoder",
    "UnknownOrdinalPolicy",
]
