"""The baseline regressor.

A baseline is not a competitor. It exists so that every other score can be read
as an improvement over *something*, and a model that fails to beat it has told
you something important.

Every capability below was verified against the installed scikit-learn rather
than assumed, and the contract tests re-verify them on each run.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.dummy import DummyRegressor

from aidatasetkit.core.types import Backend, Estimator, Interpretability, TaskType
from aidatasetkit.models.base import ModelStrategy
from aidatasetkit.models.capabilities import ModelCapabilities
from aidatasetkit.models.registry import register_model

__all__ = ["DummyRegressorStrategy"]


@register_model(aliases=("dummy", "baseline"))
class DummyRegressorStrategy(ModelStrategy):
    """Predicts one constant, ignoring every feature.

    The default strategy is ``"mean"``, which is scikit-learn's own default and
    is chosen deliberately rather than inherited:

    *It is the honest floor for squared error.* The mean is the constant that
    minimises mean squared error, so a model that cannot beat this baseline on
    MSE has extracted nothing from the features. ``"median"`` would be the
    corresponding floor for absolute error, and picking it here would quietly
    decide which error measure the analyst cares about -- a decision S7 has not
    been asked to make and this layer certainly has not.

    *It is deterministic.* Two runs on the same data give the same number, so the
    figure a real model is being compared against does not move underneath it.

    Any strategy scikit-learn accepts -- ``"median"``, ``"quantile"``,
    ``"constant"`` -- can be passed explicitly to override it.
    """

    name: ClassVar[str] = "dummy_regressor"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.REGRESSION,
        backend=Backend.SKLEARN,
        # Meaningless for a regression target, and refused by ModelCapabilities
        # if it were claimed: there are no classes to assign probabilities to.
        supports_predict_proba=False,
        supports_multiclass=False,
        # Verified: predictions identical on raw and standardised features,
        # which is unsurprising for a model that reads none of them.
        requires_scaling=False,
        supports_sparse_input=True,
        # Verified, and true for an unusual reason: the estimator ignores X
        # entirely, so it fits happily on missing values, on sparse matrices, and
        # on both at once. It is the only regressor here that takes a csr_matrix
        # still carrying its gaps. Declaring this truthfully means the baseline
        # skips numeric imputation, which changes nothing about its predictions
        # because it never looks at the features it is given.
        handles_missing_values=True,
        interpretability_level=Interpretability.HIGH,
        is_baseline=True,
    )

    def default_params(self) -> dict[str, Any]:
        """Return a deterministic baseline configuration.

        ``random_state`` is deliberately absent: unlike ``DummyClassifier``, this
        estimator's constructor does not accept one, and every strategy it offers
        is deterministic. Inventing a seed for it would be theatre.
        """
        return {"strategy": "mean"}

    def build(self, **params: Any) -> Estimator:
        """Construct a new, unfitted :class:`~sklearn.dummy.DummyRegressor`."""
        return self._construct(DummyRegressor, self.resolve_params(**params), self.name)
