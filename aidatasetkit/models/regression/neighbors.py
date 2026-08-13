"""Regression by nearest neighbours.

The model that makes the capability-driven design visible on the regression side.
It declares ``requires_scaling=True`` and S4 puts a scaler in the numeric branch
-- not because anything anywhere recognises the name "knn", but because a distance
is being computed and a distance is a statement about units.

The requirement was measured rather than argued. On features spanning eight
orders of magnitude, all genuinely informative, the raw fit and the standardised
fit disagreed by half a standard deviation of the target, and ``R2`` moved from
``0.4864`` to ``0.9661``. Unscaled, the largest-magnitude feature decides every
neighbourhood and the other two are noise.

It shares a preprocessing profile with
:class:`~aidatasetkit.models.regression.linear.RidgeRegressionStrategy`, which is
worth noticing: a penalised linear solve and a distance vote have nothing in
common as algorithms, and they need exactly the same matrix. That is the point of
caching on capabilities.

This is one of two regressors whose constructor takes no ``random_state`` --
:class:`~aidatasetkit.models.regression.linear.LinearRegressionStrategy` is the
other, and :class:`~aidatasetkit.models.regression.dummy.DummyRegressorStrategy`
is a third -- and none is invented for any of them. This one stores the training
set and averages; that average is deterministic.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.neighbors import KNeighborsRegressor

from aidatasetkit.core.types import Backend, Estimator, Interpretability, TaskType
from aidatasetkit.models.base import ModelStrategy
from aidatasetkit.models.capabilities import ModelCapabilities
from aidatasetkit.models.registry import register_model

__all__ = ["KNeighborsRegressorStrategy"]


@register_model(aliases=("knn",))
class KNeighborsRegressorStrategy(ModelStrategy):
    """The mean target of the k closest training rows.

    Interpretability is ``LOW`` on this project's definition: a fitted instance
    exposes neither importances nor a decision function. That the *mechanism* is
    easy to describe is a different thing from the fitted model being readable,
    and conflating the two would put a misleading word in the comparison table.
    """

    name: ClassVar[str] = "knn_regressor"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.REGRESSION,
        backend=Backend.SKLEARN,
        supports_predict_proba=False,
        supports_multiclass=False,
        # Verified: R2 0.4864 raw against 0.9661 standardised, on features of
        # differing magnitude that are all genuinely informative.
        requires_scaling=True,
        # Verified: fits and predicts on a scipy csr_matrix.
        supports_sparse_input=True,
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
        """Construct a new, unfitted :class:`~sklearn.neighbors.KNeighborsRegressor`."""
        return self._construct(
            KNeighborsRegressor, self.resolve_params(**params), self.name
        )
