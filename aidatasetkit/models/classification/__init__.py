"""Classification strategies.

Importing this package registers every classifier it contains. The strategy
modules are imported explicitly below rather than discovered, so which models
exist never depends on import order or on a filesystem scan.
"""

from aidatasetkit.models.classification.dummy import DummyClassifierStrategy
from aidatasetkit.models.classification.logistic import LogisticRegressionStrategy

__all__ = ["DummyClassifierStrategy", "LogisticRegressionStrategy"]
