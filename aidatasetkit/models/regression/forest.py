"""Forests of randomised regression trees.

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

Each was measured separately -- neither inherits the other's answers, and neither
inherits its classification twin's:

* Scale-invariance, across eight orders of magnitude, produced identical
  predictions raw versus standardised for both. The one caveat is the same
  absolute ``FEATURE_THRESHOLD`` of ``1e-7`` recorded in
  :mod:`aidatasetkit.models.regression.tree`.
* Native NaN handling was verified on both: fitted with 30% of the informative
  feature missing and one entirely missing row, predicting finite values.
* Neither declares sparse support, and the reason is not that they lack it. Each
  fits a ``csr_matrix`` and each fits NaN, and a ``csr_matrix`` carrying NaN
  raises ``ValueError("Input X contains NaN")``. A model declaring native missing
  support is never handed imputed data, so the sparse matrix it would actually
  receive is the one it refuses.
* Both take a two-column ``y`` and return a two-column prediction. That is
  recorded here as measured behaviour; ``ModelCapabilities`` has no multi-output
  field, and inventing one for a shape nothing in the library produces would be
  adding an unused contract.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from aidatasetkit.core.types import Backend, Estimator, Interpretability, TaskType
from aidatasetkit.models.base import ModelStrategy
from aidatasetkit.models.capabilities import ModelCapabilities
from aidatasetkit.models.registry import register_model

__all__ = ["ExtraTreesRegressorStrategy", "RandomForestRegressorStrategy"]


#: Shared by both forests: every capability below is a property of averaging
#: randomised trees, and both were verified independently rather than one being
#: copied from the other.
_FOREST_CAPABILITIES = dict(
    task_type=TaskType.REGRESSION,
    backend=Backend.SKLEARN,
    supports_predict_proba=False,
    supports_multiclass=False,
    # Verified: identical predictions on raw and standardised features.
    requires_scaling=False,
    # Verified, and NOT the obvious answer -- see the note above. Each forest
    # accepts a csr_matrix, and accepts NaN, and rejects a csr_matrix holding
    # NaN. Native missing support is the pair member worth keeping.
    supports_sparse_input=False,
    # Verified: NaN accepted at fit and at predict, predictions finite.
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
        the library's default. It is not tuned: no search was run, and no data
        was consulted.

        ``random_state`` fixes the bootstrap sample and the feature subsampling,
        and was measured to be load-bearing for both forests **at these shipped
        defaults**: on a noisy, correlated 300-row fit scored on held-out rows,
        four seeds moved the predictions by up to ``0.75`` for the random forest
        and ``0.77`` for the extra trees. Without it the same data gives a
        different model on every run, which is the one failure a comparison table
        cannot survive.

        The configuration those figures were taken at is stated because an
        earlier version of this docstring quoted numbers measured at
        ``n_estimators=30`` while attaching them to ``n_estimators=100``. A
        measurement whose configuration is not recorded is a number nobody can
        reproduce, which is the same defect as metadata that lies.
        """
        return {"n_estimators": 100, "random_state": self.config.random_state}


@register_model(aliases=("random_forest",))
class RandomForestRegressorStrategy(_ForestStrategy):
    """Bagged trees, each split chosen as the best over a random feature subset."""

    name: ClassVar[str] = "random_forest_regressor"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(**_FOREST_CAPABILITIES)

    def build(self, **params: Any) -> Estimator:
        """Construct a new, unfitted :class:`~sklearn.ensemble.RandomForestRegressor`."""
        return self._construct(
            RandomForestRegressor, self.resolve_params(**params), self.name
        )


@register_model(aliases=("extra_trees",))
class ExtraTreesRegressorStrategy(_ForestStrategy):
    """Trees whose split thresholds are drawn at random rather than searched.

    The extra randomness is the point: it lowers variance and costs less to fit,
    at the price of a little more bias on any single tree.
    """

    name: ClassVar[str] = "extra_trees_regressor"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(**_FOREST_CAPABILITIES)

    def build(self, **params: Any) -> Estimator:
        """Construct a new, unfitted :class:`~sklearn.ensemble.ExtraTreesRegressor`."""
        return self._construct(
            ExtraTreesRegressor, self.resolve_params(**params), self.name
        )
