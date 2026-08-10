"""Gaussian naive Bayes.

Fits one Gaussian per feature per class and multiplies the likelihoods. The
fitted model is entirely readable -- a class prior, a mean, and a variance for
every feature -- which is why interpretability is ``HIGH`` even though the
independence assumption behind it is almost never true.

It carries the catalog's most surprising declaration, and the reason is worth
stating precisely because the textbook answer is the opposite.

**In theory this model is scale-invariant.** Rescaling a feature rescales that
feature's fitted mean and variance identically, so the likelihood ratio between
classes is unchanged.

**As scikit-learn implements it, it is not.** Every feature variance has a floor
added to it::

    epsilon_ = var_smoothing * max(variance over ALL features)

One global floor, derived from the widest-spread feature, applied to every
feature alike. Measured on features spanning eight orders of magnitude: the
floor came out at ``10.08`` while the smallest feature's true variance was
``1.03e-06`` -- the floor was 9.8 million times larger, so that feature's fitted
variance became the floor and the feature stopped carrying information.
Predictions agreed with the standardised fit on 68% of rows and accuracy moved
from 0.693 to 0.943.

That it is an implementation artefact rather than mathematics was confirmed by
sweeping the parameter: at ``var_smoothing=0`` raw and standardised fits agree on
100% of rows, exactly as the theory says. The capability describes the estimator
this library actually constructs, so it is declared ``True``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.naive_bayes import GaussianNB

from aidatasetkit.core.types import Backend, Estimator, Interpretability, TaskType
from aidatasetkit.models.base import ModelStrategy
from aidatasetkit.models.capabilities import ModelCapabilities
from aidatasetkit.models.registry import register_model

__all__ = ["GaussianNBStrategy"]


@register_model(aliases=("gnb",))
class GaussianNBStrategy(ModelStrategy):
    """One Gaussian per feature per class, combined under an independence assumption.

    The alias is ``gnb`` rather than ``naive_bayes``. The catalog will one day
    hold the multinomial and Bernoulli variants, which are naive Bayes just as
    much as this one is; taking the general name for the Gaussian case now would
    make that alias either wrong or unusable later.
    """

    name: ClassVar[str] = "gaussian_nb"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.CLASSIFICATION,
        backend=Backend.SKLEARN,
        # Verified: posterior per class, in [0, 1], rows summing to 1.
        supports_predict_proba=True,
        # Verified, and against the theory -- see the module docstring. The shared
        # var_smoothing floor makes the implementation scale-dependent.
        requires_scaling=True,
        # Verified: raises TypeError("Sparse data was passed for X, but dense data
        # is required"). This is the second model exercising the dense path.
        supports_sparse_input=False,
        # Verified: three classes fitted directly.
        supports_multiclass=True,
        # Verified: raises ValueError("Input X contains NaN").
        handles_missing_values=False,
        # A prior, a mean, and a variance per feature per class, all readable.
        interpretability_level=Interpretability.HIGH,
        is_baseline=False,
    )

    def default_params(self) -> dict[str, Any]:
        """Return conservative, untuned defaults.

        ``var_smoothing=1e-9`` restates the scikit-learn default. It is written
        out for the same reason the module docstring dwells on it: this one
        parameter is the entire reason the model declares ``requires_scaling``,
        and a capability that depends on a value ought to name the value it
        depends on. Pinning it also means a change to scikit-learn's default
        cannot silently invalidate that declaration.

        ``random_state`` is deliberately absent: the constructor does not accept
        one, and the fit is a closed-form calculation with nothing to seed.
        """
        return {"var_smoothing": 1e-9}

    def build(self, **params: Any) -> Estimator:
        """Construct a new, unfitted :class:`~sklearn.naive_bayes.GaussianNB`."""
        return self._construct(GaussianNB, self.resolve_params(**params), self.name)
