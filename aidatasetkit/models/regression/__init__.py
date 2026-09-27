"""Regression strategies.

Importing this package registers every regressor it contains. The strategy
modules are imported explicitly below rather than discovered, so which models
exist never depends on import order or on a filesystem scan.

Regression is not a second modelling subsystem. These nine models use the same
:class:`~aidatasetkit.models.base.ModelStrategy`, the same
:class:`~aidatasetkit.models.capabilities.ModelCapabilities`, the same registry
and the same factory as the classifiers, and they reach S4 through the same
three-answer :class:`~aidatasetkit.core.types.PreprocessingProfile`. Nothing
anywhere asks whether a model is a regressor in order to prepare its data.

The nine occupy four distinct profile keys:

=========================================  ========  =======  ===========
Profile                                    Scaling   Sparse   Native NaN
=========================================  ========  =======  ===========
decision tree, forests, hist boosting      no        no       yes
dummy                                      no        yes      yes
gradient boosting                          no        yes      no
linear regression, ridge, k nearest        yes       yes      no
=========================================  ========  =======  ===========

Every one of those four keys already existed in the classification catalog, so
eighteen built-in models still need only five preprocessors between them. That
is not a coincidence arranged here; it is what happens when the cache key is a
capability triple rather than a model name.

Two rows are worth reading twice.

**A penalised linear solve and a distance vote share a preprocessor**, which no
taxonomy of algorithms would suggest. Ridge and k-nearest-neighbours have nothing
in common as algorithms and need exactly the same matrix, because the only thing
consulted is the three answers.

**Both linear models ask for scaling, for different reasons.** Ridge because its
penalty reads the unit a column was recorded in; ordinary least squares because
``scipy.linalg.lstsq`` truncates a small singular value and silently discards a
real feature at a column-magnitude ratio around ``1e6`` -- which a dollar amount
beside a proportion reaches easily. That was measured on scikit-learn 1.9.0;
1.5.2 through 1.8.0 keep full rank, where the declaration is harmless. See
:mod:`aidatasetkit.models.regression.linear`; the second reason was found by
adversarial review after an earlier version of this catalog declared OLS
scale-free on the strength of a frame that sat just under the cliff.
"""

from aidatasetkit.models.regression.boosting import (
    GradientBoostingRegressorStrategy,
    HistGradientBoostingRegressorStrategy,
)
from aidatasetkit.models.regression.dummy import DummyRegressorStrategy
from aidatasetkit.models.regression.forest import (
    ExtraTreesRegressorStrategy,
    RandomForestRegressorStrategy,
)
from aidatasetkit.models.regression.linear import (
    LinearRegressionStrategy,
    RidgeRegressionStrategy,
)
from aidatasetkit.models.regression.neighbors import KNeighborsRegressorStrategy
from aidatasetkit.models.regression.tree import DecisionTreeRegressorStrategy

__all__ = [
    "DecisionTreeRegressorStrategy",
    "DummyRegressorStrategy",
    "ExtraTreesRegressorStrategy",
    "GradientBoostingRegressorStrategy",
    "HistGradientBoostingRegressorStrategy",
    "KNeighborsRegressorStrategy",
    "LinearRegressionStrategy",
    "RandomForestRegressorStrategy",
    "RidgeRegressionStrategy",
]
