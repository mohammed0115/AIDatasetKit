"""Forests of randomised trees.

Two ensembles that differ in one decision and share everything else. Both average
many trees, so both inherit a tree's indifference to feature scale and a tree's
native handling of missing values; neither can be read as a rule, so both report
``MEDIUM`` interpretability -- importances, but no decision function a person
follows.

They are separate models rather than one strategy with a switch because the
difference is in the algorithm, not in a parameter this library should be
choosing. A random forest searches for the best threshold on each candidate
feature; extra trees draw the threshold at random and keep the best of those.
That trades a little more bias for a lot less variance and runs faster, and which
one suits a dataset is not something AIDatasetKit decides.

Capabilities were verified against the installed scikit-learn, not assumed:

* Scale-invariance, measured across eight orders of magnitude, produced 100%
  identical predictions raw versus standardised for both. The one caveat is the
  same absolute ``FEATURE_THRESHOLD`` of ``1e-7`` described in
  :mod:`aidatasetkit.models.classification.tree`: a feature whose entire range
  is below it is dropped as constant. It does not make scaling a requirement at
  any ordinary magnitude.
* Native NaN handling is inherited from the underlying trees and was verified on
  both: fit with 30% of the informative feature missing, ``predict`` accepting
  NaN on unseen rows, and an entirely missing row still returning a class.
* Neither declares sparse support, and the reason is not that they lack it.
  Each fits a ``csr_matrix`` and each fits NaN, but a ``csr_matrix`` carrying
  NaN raises ``ValueError("Input X contains NaN")``. A model declaring native
  missing support is never handed imputed data, so the sparse matrix it would
  actually receive is the one it refuses. The declaration states the pair that
  holds together, and keeps the more valuable half.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from aidatasetkit.core.types import Backend, Estimator, Interpretability, TaskType
from aidatasetkit.models.base import ModelStrategy
from aidatasetkit.models.capabilities import ModelCapabilities
from aidatasetkit.models.registry import register_model

__all__ = ["ExtraTreesClassifierStrategy", "RandomForestClassifierStrategy"]


#: Shared by both forests: every capability below is a property of averaging
#: randomised trees, and both were verified independently rather than one being
#: copied from the other.
_FOREST_CAPABILITIES = dict(
    task_type=TaskType.CLASSIFICATION,
    backend=Backend.SKLEARN,
    # Verified: the mean of the per-tree leaf frequencies, in [0, 1], summing to 1.
    supports_predict_proba=True,
    # Verified: identical predictions on raw and standardised features.
    requires_scaling=False,
    # Verified, and NOT the obvious answer -- see the note below. Each forest
    # accepts a csr_matrix, and accepts NaN, and rejects a csr_matrix holding
    # NaN. Native missing support is the pair member worth keeping.
    supports_sparse_input=False,
    # Verified: three classes fitted directly, no wrapper.
    supports_multiclass=True,
    # Verified: NaN accepted at fit and at predict, per-split direction learned.
    handles_missing_values=True,
    # Importances, but no readable decision function.
    interpretability_level=Interpretability.MEDIUM,
    is_baseline=False,
)


class _ForestStrategy(ModelStrategy):
    """What the two forests share: a tree count and a seed.

    Not a public base class and not an extension point -- it exists only so the
    same default and the same reasoning are not written twice. Both concrete
    strategies remain independently registered models with independently verified
    capabilities.
    """

    def default_params(self) -> dict[str, Any]:
        """Return conservative, untuned defaults.

        ``n_estimators=100`` restates the scikit-learn default rather than
        overriding it. It is written out because the catalog reports
        ``default_params``, and how many trees an analyst is getting is worth
        seeing there; writing it also pins the value against a future change to
        the library's default. It is not tuned: no search was run, and no data was
        consulted. Fewer trees would make the ensemble noisier for no benefit
        beyond speed, and a caller who wants speed can pass their own.

        ``random_state`` fixes the bootstrap sample and the feature subsampling.
        Without it the same data gives a different model on every run, which is
        the one failure a comparison table cannot survive.
        """
        return {"n_estimators": 100, "random_state": self.config.random_state}


@register_model(aliases=("random_forest",))
class RandomForestClassifierStrategy(_ForestStrategy):
    """Bagged trees, each split chosen as the best over a random feature subset."""

    name: ClassVar[str] = "random_forest_classifier"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(**_FOREST_CAPABILITIES)

    def build(self, **params: Any) -> Estimator:
        """Construct a new, unfitted :class:`~sklearn.ensemble.RandomForestClassifier`."""
        return self._construct(
            RandomForestClassifier, self.resolve_params(**params), self.name
        )


@register_model(aliases=("extra_trees",))
class ExtraTreesClassifierStrategy(_ForestStrategy):
    """Trees whose split thresholds are drawn at random rather than searched.

    The extra randomness is the point: it lowers variance and costs less to fit,
    at the price of a little more bias on any single tree.
    """

    name: ClassVar[str] = "extra_trees_classifier"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(**_FOREST_CAPABILITIES)

    def build(self, **params: Any) -> Estimator:
        """Construct a new, unfitted :class:`~sklearn.ensemble.ExtraTreesClassifier`."""
        return self._construct(
            ExtraTreesClassifier, self.resolve_params(**params), self.name
        )
