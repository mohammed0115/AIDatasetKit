"""Training: the order in which the other layers are called, and the one promise.

Nothing here is an algorithm. The task comes from the detector, the model from
the registry, the preparation from the capability profile, the encoding from the
target contract, the metrics from the evaluation policy. This layer owns the
sequence and one guarantee that runs through all of it:

    Nothing is ever fitted on evaluation rows.

The split is drawn once, before any model is touched, and every model in a
comparison trains on the same rows and is judged on the same rows. What differs
between them is the preparation each one's capabilities call for -- and that
difference is the point, not a flaw to be normalised away.

What comes out is an ordering under one dataset, one split, one preprocessing
contract, one configuration and one metric. It is not a best model, and the
result carries enough context to write that sentence out in full.
"""

from aidatasetkit.training.clustering import ClusteringResult, ClusteringRunner
from aidatasetkit.training.comparison import (
    ComparisonResult,
    ModelComparator,
    ModelOutcome,
)
from aidatasetkit.training.split import DataSplit, split_rows
from aidatasetkit.training.trainer import ModelTrainer, TrainingResult

__all__ = [
    "ClusteringResult",
    "ClusteringRunner",
    "ComparisonResult",
    "DataSplit",
    "ModelComparator",
    "ModelOutcome",
    "ModelTrainer",
    "TrainingResult",
    "split_rows",
]
