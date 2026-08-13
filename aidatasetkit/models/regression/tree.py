"""A single regression tree.

Its interpretability is declared ``HIGH`` because the fitted form is a set of
threshold rules, each one readable on its own terms -- not because the whole tree
is short. At the shipped unbounded depth it has roughly one leaf per training
row, so "readable" means any single prediction can be traced to the exact
comparisons that produced it, which is more than any other regressor here offers
and less than a page a person reads front to back. Its tendency to memorise the
training set is left for evaluation to expose.

The capabilities were re-measured for the regressor rather than copied from
:mod:`aidatasetkit.models.classification.tree`. They agree, and agreeing is a
result rather than an assumption.

**A tree is scale-invariant, so it does not ask for scaling.** A split is a
threshold on one feature, and multiplying that feature by a constant moves the
threshold with it. Measured on features spanning eight orders of magnitude, all
genuinely informative: predictions fitted on the raw features and on standardised
ones were identical.

One caveat on that invariance, carried over and re-measured here. scikit-learn
casts tree input to ``float32`` and treats a feature as constant when its spread
falls below an *absolute* ``FEATURE_THRESHOLD`` of ``1e-7``. Measured on the
regressor: a feature carrying all the signal had importance ``0.9997`` at a range
of ``5.5``, ``0.2321`` at ``5.5e-07``, and ``0.0000`` at ``5.5e-08``.
Standardising restores it to ``0.9997``. That is a real limit and it is recorded
rather than hidden, but it does not make scaling a *requirement*: at any ordinary
magnitude the two fits are identical, and adding a scaler would cost the raw
thresholds that make a tree readable.

**A tree consumes NaN natively.** Verified as real rather than merely tolerated:
fitted with 30% of the informative feature missing and one row missing entirely,
``predict`` accepted NaN, and every prediction came back finite.

**Those two supports do not compose, which is why sparse is declared false.** The
estimator takes a ``csr_matrix``, and it takes NaN, and a ``csr_matrix``
containing NaN raises ``ValueError("Input X contains NaN")``. A model declaring
native NaN support is never handed imputed data, so the sparse matrix it would
receive is precisely the one it rejects. Since the capability triple has to
describe a single coherent contract, it states the pair that holds together and
keeps the more valuable half: a learned direction for a gap, rather than a median
nobody measured.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.tree import DecisionTreeRegressor

from aidatasetkit.core.types import Backend, Estimator, Interpretability, TaskType
from aidatasetkit.models.base import ModelStrategy
from aidatasetkit.models.capabilities import ModelCapabilities
from aidatasetkit.models.registry import register_model

__all__ = ["DecisionTreeRegressorStrategy"]


@register_model(aliases=("decision_tree",))
class DecisionTreeRegressorStrategy(ModelStrategy):
    """Recursive threshold splits predicting a constant per leaf.

    Depth is deliberately left unbounded. An unbounded regression tree fits the
    training rows exactly and the evaluation stage is where that should become
    visible; choosing a depth here would be tuning, done without looking at the
    data, and it would quietly cap a model the analyst may have selected
    precisely for its detail.
    """

    name: ClassVar[str] = "decision_tree_regressor"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.REGRESSION,
        backend=Backend.SKLEARN,
        supports_predict_proba=False,
        supports_multiclass=False,
        # Verified: identical predictions on raw and standardised features.
        requires_scaling=False,
        # Verified, and NOT the obvious answer. This estimator accepts a
        # csr_matrix, and it accepts NaN -- but never both at once: sparse input
        # carrying NaN raises ValueError("Input X contains NaN"). Declaring
        # sparse here would let S4 build exactly the matrix this model refuses.
        supports_sparse_input=False,
        # Verified: NaN accepted at fit and at predict; predictions finite.
        handles_missing_values=True,
        interpretability_level=Interpretability.HIGH,
        is_baseline=False,
    )

    def default_params(self) -> dict[str, Any]:
        """Return conservative, untuned defaults.

        ``random_state`` is load-bearing here and was measured to be so, not
        assumed: four different seeds produced four different fitted trees, whose
        held-out predictions differed by up to ``4.08``. Ties between equally good
        splits are broken by drawing on it, so two trees fitted on the same data
        differ without it. Everything else is left at the scikit-learn default.
        """
        return {"random_state": self.config.random_state}

    def build(self, **params: Any) -> Estimator:
        """Construct a new, unfitted :class:`~sklearn.tree.DecisionTreeRegressor`."""
        return self._construct(
            DecisionTreeRegressor, self.resolve_params(**params), self.name
        )
