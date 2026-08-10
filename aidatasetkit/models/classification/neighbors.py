"""Classification by nearest neighbours.

The model that makes the capability-driven design visible. It declares
``requires_scaling=True`` and S4 puts a scaler in the numeric branch -- not
because anything anywhere recognises the name "knn", but because a distance is
being computed and a distance is a statement about units.

The requirement was measured rather than argued. On features spanning eight
orders of magnitude, all genuinely informative, predictions fitted on raw
features agreed with predictions fitted on standardised features on only 75% of
rows, and accuracy moved from 0.763 to 0.963. Unscaled, the largest-magnitude
feature decides every neighbourhood and the other two are noise.

It is also the only model here with no ``random_state`` in its constructor, and
none is invented for it. Its fit stores the training set and its prediction is a
deterministic majority vote.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.neighbors import KNeighborsClassifier

from aidatasetkit.core.types import Backend, Estimator, Interpretability, TaskType
from aidatasetkit.models.base import ModelStrategy
from aidatasetkit.models.capabilities import ModelCapabilities
from aidatasetkit.models.registry import register_model

__all__ = ["KNeighborsClassifierStrategy"]


@register_model(aliases=("knn",))
class KNeighborsClassifierStrategy(ModelStrategy):
    """Majority vote among the k closest training rows.

    Interpretability is ``LOW`` on this project's definition: a fitted instance
    exposes neither importances nor a decision function. That the *mechanism* is
    easy to describe is a different thing from the fitted model being readable,
    and conflating the two would put a misleading word in the comparison table.
    """

    name: ClassVar[str] = "knn_classifier"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.CLASSIFICATION,
        backend=Backend.SKLEARN,
        # Verified: the fraction of the k neighbours in each class, summing to 1.
        supports_predict_proba=True,
        # Verified: 75% prediction agreement and accuracy 0.763 -> 0.963 between
        # raw and standardised features of differing magnitude.
        requires_scaling=True,
        # Verified: fits and predicts on a scipy csr_matrix.
        supports_sparse_input=True,
        # Verified: three classes fitted directly.
        supports_multiclass=True,
        # Verified: raises ValueError("Input X contains NaN"). A distance to a
        # missing coordinate is undefined, so there is nothing to fall back on.
        handles_missing_values=False,
        # Verified: a fitted instance exposes no feature_importances_, no coef_.
        interpretability_level=Interpretability.LOW,
        is_baseline=False,
    )

    def default_params(self) -> dict[str, Any]:
        """Return conservative, untuned defaults.

        ``n_neighbors=5`` restates the scikit-learn default. It is written out
        because this estimator takes no seed, so without it the catalog would
        report no defaults at all for the model whose single most consequential
        parameter is this one -- and because pinning it means a change to
        scikit-learn's default cannot silently change every prediction this
        library makes. No search was run and no data was consulted.

        ``random_state`` is deliberately absent: the constructor does not accept
        one, and passing a seed to a deterministic estimator would be theatre.
        """
        return {"n_neighbors": 5}

    def build(self, **params: Any) -> Estimator:
        """Construct a new, unfitted :class:`~sklearn.neighbors.KNeighborsClassifier`."""
        return self._construct(
            KNeighborsClassifier, self.resolve_params(**params), self.name
        )
