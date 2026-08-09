"""The baseline classifier.

A baseline is not a competitor. It exists so that every other score can be read
as an improvement over *something*, and a model that fails to beat it has told
you something important.

Every capability below was verified against the installed scikit-learn rather
than assumed, and the contract tests re-verify them on each run.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.dummy import DummyClassifier

from aidatasetkit.core.types import Backend, Estimator, Interpretability, TaskType
from aidatasetkit.models.base import ModelStrategy
from aidatasetkit.models.capabilities import ModelCapabilities
from aidatasetkit.models.registry import register_model

__all__ = ["DummyClassifierStrategy"]


@register_model(aliases=("dummy", "baseline"))
class DummyClassifierStrategy(ModelStrategy):
    """Predicts from the class distribution alone, ignoring every feature.

    The default strategy is ``"prior"``, chosen over the alternatives for two
    reasons that matter to a baseline:

    *It is deterministic.* ``"stratified"`` samples from the class distribution,
    so the same data scores differently on every run and the number a real model
    is being compared against moves underneath it.

    *Its probabilities are honest.* ``"prior"`` returns the empirical class
    priors, measured here as ``[0.5, 0.5]`` on a balanced sample.
    ``"most_frequent"`` returns a hard ``[1.0, 0.0]`` -- a claim of certainty from
    a model that has looked at nothing, which makes log-loss infinite and any
    probability-based comparison meaningless.

    Both give a ROC-AUC of 0.5, which is the correct floor. Any strategy accepted
    by scikit-learn can be passed explicitly to override the default.
    """

    name: ClassVar[str] = "dummy_classifier"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.CLASSIFICATION,
        backend=Backend.SKLEARN,
        supports_predict_proba=True,
        requires_scaling=False,
        supports_sparse_input=True,
        # Verified, and true for an unusual reason: the estimator ignores X
        # entirely, so it fits happily on missing values, on sparse matrices, and
        # even on text. Declaring this truthfully means the baseline skips numeric
        # imputation, which changes nothing about its predictions because it never
        # looks at the features it is given.
        handles_missing_values=True,
        supports_multiclass=True,
        interpretability_level=Interpretability.HIGH,
        is_baseline=True,
    )

    def default_params(self) -> dict[str, Any]:
        """Return a deterministic baseline configuration.

        ``random_state`` is inert for ``"prior"``. It is set anyway so that
        overriding the strategy to a randomised one stays reproducible -- a
        baseline that changes between runs is precisely the failure this default
        is chosen to avoid.
        """
        return {"strategy": "prior", "random_state": self.config.random_state}

    def build(self, **params: Any) -> Estimator:
        """Construct a new, unfitted :class:`~sklearn.dummy.DummyClassifier`."""
        return self._construct(
            DummyClassifier, self.resolve_params(**params), self.name
        )
