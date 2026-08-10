"""A single decision tree.

The one model in the catalog whose fitted form is a set of rules a person can
read end to end, which is why its interpretability is declared ``HIGH`` while its
tendency to memorise the training set is left for evaluation to expose.

Two capabilities were verified against the installed scikit-learn and neither
matches the answer most people carry from memory.

**A tree is scale-invariant, so it does not ask for scaling.** A split is a
threshold on one feature, and multiplying that feature by a constant moves the
threshold with it. Measured on features spanning eight orders of magnitude
(``x*1e-3``, ``x*1.0``, ``x*1e5``, all genuinely informative): predictions fitted
on the raw features and on standardised ones agreed on 100% of rows.

One caveat on that invariance, found by trying to refute it. scikit-learn casts
tree input to ``float32`` and treats a feature as constant when its spread falls
below an *absolute* ``FEATURE_THRESHOLD`` of ``1e-7``. A feature whose entire
range is smaller than that -- a quantity recorded in units so small the whole
column spans less than a ten-millionth -- is discarded silently: measured
importance ``0.0000`` at a range of ``5.2e-08``, against ``1.0000`` for the same
information at range ``5.2``. Standardising rescues it. That is a real limit and
it is recorded here rather than hidden, but it does not make scaling a
*requirement*: at any ordinary magnitude the two fits are bit-identical, and
adding a scaler would cost the raw thresholds that make a tree readable.

**A tree consumes NaN natively in this version.** Missing-value support reached
``DecisionTreeClassifier`` in scikit-learn 1.3, and 1.9 is installed. Verified as
real rather than merely tolerated: with 30% of the informative feature missing it
still fit, ``predict`` accepted NaN on unseen rows, a row whose features were
entirely missing still produced a class, and the fitted ``tree_`` exposes a
``missing_go_to_left`` array containing both directions -- the tree learns which
way a missing value should fall at each split rather than substituting a value.

Declaring that truthfully is what lets S4 skip imputation for this model. A
median substituted into a column is a value nobody observed; a learned direction
is a decision the model made from the data.

**Those two supports do not compose, which is why sparse is declared false.** The
estimator takes a ``csr_matrix``, and it takes NaN, and a ``csr_matrix``
containing NaN raises ``ValueError("Input X contains NaN")``. A model declaring
native NaN support is never handed imputed data, so the sparse matrix it would
receive is precisely the one it rejects -- reproduced end to end on an ordinary
frame with a high-cardinality city column and some missing amounts. Since the
capability triple has to describe a single coherent contract, it states the pair
that holds together.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.tree import DecisionTreeClassifier

from aidatasetkit.core.types import Backend, Estimator, Interpretability, TaskType
from aidatasetkit.models.base import ModelStrategy
from aidatasetkit.models.capabilities import ModelCapabilities
from aidatasetkit.models.registry import register_model

__all__ = ["DecisionTreeClassifierStrategy"]


@register_model(aliases=("decision_tree",))
class DecisionTreeClassifierStrategy(ModelStrategy):
    """Recursive threshold splits, readable as rules.

    Depth is deliberately left unbounded. An unbounded tree overfits and the
    evaluation stage is where that should become visible; choosing a depth here
    would be tuning, done without looking at the data, and it would quietly cap a
    model the analyst may have selected precisely for its detail.
    """

    name: ClassVar[str] = "decision_tree_classifier"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.CLASSIFICATION,
        backend=Backend.SKLEARN,
        # Verified: predict_proba returns class frequencies at the reached leaf,
        # shaped (n_samples, n_classes), in [0, 1], rows summing to 1.
        supports_predict_proba=True,
        # Verified: identical predictions on raw and standardised features.
        requires_scaling=False,
        # Verified, and NOT the obvious answer. This estimator accepts a
        # csr_matrix, and it accepts NaN -- but never both at once: sparse input
        # carrying NaN raises ValueError("Input X contains NaN"). The capability
        # triple describes one coherent data contract rather than three separate
        # facts, so it has to state the combination that actually holds. Native
        # missing support is kept because it is the more valuable of the two --
        # the model learns a direction for a gap instead of being handed a median
        # nobody observed -- and sparsity is only a memory optimisation. Declaring
        # sparse here would let S4 build exactly the matrix this model refuses.
        supports_sparse_input=False,
        # Verified: three classes fitted directly, no wrapper.
        supports_multiclass=True,
        # Verified: NaN accepted at fit and at predict; the tree learns a
        # missing-value direction per split.
        handles_missing_values=True,
        interpretability_level=Interpretability.HIGH,
        is_baseline=False,
    )

    def default_params(self) -> dict[str, Any]:
        """Return conservative, untuned defaults.

        ``random_state`` is not cosmetic here. Ties between equally good splits
        are broken by drawing on it, so two trees fitted on the same data can
        differ without it. Everything else is left at the scikit-learn default.
        """
        return {"random_state": self.config.random_state}

    def build(self, **params: Any) -> Estimator:
        """Construct a new, unfitted :class:`~sklearn.tree.DecisionTreeClassifier`."""
        return self._construct(
            DecisionTreeClassifier, self.resolve_params(**params), self.name
        )
