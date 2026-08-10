"""Classification strategies.

Importing this package registers every classifier it contains. The strategy
modules are imported explicitly below rather than discovered, so which models
exist never depends on import order or on a filesystem scan.

The nine built-in classifiers occupy five distinct
:class:`~aidatasetkit.core.types.PreprocessingProfile` keys, which is what S4
caches on:

=========================================  ========  =======  ===========
Profile                                    Scaling   Sparse   Native NaN
=========================================  ========  =======  ===========
decision tree, forests, hist boosting      no        no       yes
dummy                                      no        yes      yes
gradient boosting                          no        yes      no
logistic regression, k nearest             yes       yes      no
gaussian naive Bayes                       yes       no       no
=========================================  ========  =======  ===========

Four models share the first row and two share the fourth, so nine classifiers
need five preprocessors. Nothing arranged that; it falls out of what the
algorithms actually require.

The first row is worth reading twice. Those four take ``NaN`` natively, and the
three tree-based ones also take a ``csr_matrix`` -- but not one containing
``NaN``, which is exactly the matrix a natively-missing-aware model would be
handed. Since the profile describes one data contract rather than three
independent facts, they declare the pair that holds and keep the half that
matters: a learned direction for a gap, rather than a median nobody measured.
"""

from aidatasetkit.models.classification.boosting import (
    GradientBoostingClassifierStrategy,
    HistGradientBoostingClassifierStrategy,
)
from aidatasetkit.models.classification.dummy import DummyClassifierStrategy
from aidatasetkit.models.classification.forest import (
    ExtraTreesClassifierStrategy,
    RandomForestClassifierStrategy,
)
from aidatasetkit.models.classification.logistic import LogisticRegressionStrategy
from aidatasetkit.models.classification.naive_bayes import GaussianNBStrategy
from aidatasetkit.models.classification.neighbors import KNeighborsClassifierStrategy
from aidatasetkit.models.classification.tree import DecisionTreeClassifierStrategy

__all__ = [
    "DecisionTreeClassifierStrategy",
    "DummyClassifierStrategy",
    "ExtraTreesClassifierStrategy",
    "GaussianNBStrategy",
    "GradientBoostingClassifierStrategy",
    "HistGradientBoostingClassifierStrategy",
    "KNeighborsClassifierStrategy",
    "LogisticRegressionStrategy",
    "RandomForestClassifierStrategy",
]
