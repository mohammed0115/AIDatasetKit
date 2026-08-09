"""Logistic regression.

A linear model whose fitted coefficients can be read one per feature, which is
why its interpretability is declared ``HIGH`` while its accuracy on non-linear
data often is not.

Every capability below was verified against the installed scikit-learn. One
verification mattered more than the rest: ``multi_class`` was removed from
``LogisticRegression`` and passing it now raises ``TypeError``. Multiclass
support is native and automatic, and this strategy never passes that parameter.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.linear_model import LogisticRegression

from aidatasetkit.core.types import Backend, Estimator, Interpretability, TaskType
from aidatasetkit.models.base import ModelStrategy
from aidatasetkit.models.capabilities import ModelCapabilities
from aidatasetkit.models.registry import register_model

__all__ = ["LogisticRegressionStrategy"]


@register_model(aliases=("logistic", "logreg"))
class LogisticRegressionStrategy(ModelStrategy):
    """Linear classification with probability estimates.

    Scaling is declared as required for two independent reasons: the default
    ``lbfgs`` solver converges poorly when features differ by orders of
    magnitude, and the L2 penalty is applied to coefficients whose size depends
    on the scale of their feature, so an unscaled model penalises features
    unevenly for reasons that have nothing to do with the data.
    """

    name: ClassVar[str] = "logistic_regression"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.CLASSIFICATION,
        backend=Backend.SKLEARN,
        supports_predict_proba=True,
        requires_scaling=True,
        supports_sparse_input=True,
        # Verified: multiclass is fitted natively, without a wrapper and without
        # the removed multi_class parameter.
        supports_multiclass=True,
        # Verified: fitting on NaN raises ValueError("Input X contains NaN").
        handles_missing_values=False,
        interpretability_level=Interpretability.HIGH,
        is_baseline=False,
    )

    def default_params(self) -> dict[str, Any]:
        """Return conservative, untuned defaults.

        ``max_iter`` is raised from the scikit-learn default of 100 because that
        budget is exhausted on features of mixed magnitude -- measured here at
        exactly 100 iterations and a ``ConvergenceWarning`` on a frame holding an
        income, an age, and a ratio, where 1000 converges in 106. A model that
        stopped early because of a warning nobody read is a silent failure, and
        it is the unscaled case that reaches this default, since a caller who
        skips the scaler this strategy asks for is precisely the one who needs
        the headroom. Nothing else is tuned, and no default is chosen by looking
        at the data.

        ``random_state`` is inert for the default ``lbfgs`` solver, which is
        deterministic. It is set so that switching to ``saga`` or ``liblinear``
        stays reproducible without the caller having to remember.
        """
        return {"max_iter": 1000, "random_state": self.config.random_state}

    def build(self, **params: Any) -> Estimator:
        """Construct a new, unfitted :class:`~sklearn.linear_model.LogisticRegression`."""
        return self._construct(
            LogisticRegression, self.resolve_params(**params), self.name
        )
