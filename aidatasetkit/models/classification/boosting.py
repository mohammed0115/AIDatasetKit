"""Gradient-boosted trees, in scikit-learn's two implementations.

These two are in the catalog together because they look interchangeable and are
not. They boost trees on the gradient of a loss, they are both indifferent to
feature scale, and they disagree on the two capabilities that actually change how
a pipeline is built:

============================  ==========  ==================
Capability                    Gradient    HistGradient
============================  ==========  ==================
``supports_sparse_input``     ``True``    ``False``
``handles_missing_values``    ``False``   ``True``
============================  ==========  ==================

Both were measured, not looked up. ``HistGradientBoostingClassifier`` refuses a
``csr_matrix`` with ``TypeError("Sparse data was passed for X, but dense data is
required")``, and ``GradientBoostingClassifier`` refuses NaN with
``ValueError("Input X contains NaN")``. Declaring those honestly is the whole
mechanism: the two models land on different
:class:`~aidatasetkit.core.types.PreprocessingProfile` keys, so S4 densifies for
one and imputes for the other without any code anywhere asking which model this
is.

Neither exposes a readable decision function. ``GradientBoostingClassifier``
publishes ``feature_importances_`` and so reports ``MEDIUM``;
``HistGradientBoostingClassifier`` publishes no importances at all -- verified on
a fitted instance -- and so reports ``LOW``. That is a statement about what the
fitted object lets you read, not about accuracy.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

from aidatasetkit.core.types import Backend, Estimator, Interpretability, TaskType
from aidatasetkit.models.base import ModelStrategy
from aidatasetkit.models.capabilities import ModelCapabilities
from aidatasetkit.models.registry import register_model

__all__ = [
    "GradientBoostingClassifierStrategy",
    "HistGradientBoostingClassifierStrategy",
]


@register_model(aliases=("gradient_boosting",))
class GradientBoostingClassifierStrategy(ModelStrategy):
    """Stage-wise boosting of small regression trees on the loss gradient."""

    name: ClassVar[str] = "gradient_boosting_classifier"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.CLASSIFICATION,
        backend=Backend.SKLEARN,
        # Verified: shaped (n_samples, n_classes), in [0, 1], rows summing to 1.
        supports_predict_proba=True,
        # Verified: identical predictions on raw and standardised features.
        requires_scaling=False,
        # Verified: fits and predicts on a scipy csr_matrix.
        supports_sparse_input=True,
        # Verified: three classes fitted directly.
        supports_multiclass=True,
        # Verified: raises ValueError("Input X contains NaN"). Unlike the plain
        # trees it is built from, this one has no missing-value path.
        handles_missing_values=False,
        interpretability_level=Interpretability.MEDIUM,
        is_baseline=False,
    )

    def default_params(self) -> dict[str, Any]:
        """Return conservative, untuned defaults.

        ``n_estimators=100`` restates the scikit-learn default so the catalog
        reports how many boosting stages are being run, and pins it against a
        change to that default. ``learning_rate`` is left alone: it trades against
        the stage count, and moving either without the other is tuning.

        ``random_state`` is genuinely load-bearing here even at the default
        ``subsample=1.0``, because the underlying trees permute features when
        breaking tied splits.
        """
        return {"n_estimators": 100, "random_state": self.config.random_state}

    def build(self, **params: Any) -> Estimator:
        """Construct a new, unfitted :class:`~sklearn.ensemble.GradientBoostingClassifier`."""
        return self._construct(
            GradientBoostingClassifier, self.resolve_params(**params), self.name
        )


@register_model(aliases=("hist_gradient_boosting",))
class HistGradientBoostingClassifierStrategy(ModelStrategy):
    """Boosting over binned features, with a learned direction for missing values.

    The binning is what makes it fast and what makes it dense-only: the
    implementation histograms each feature into at most 255 bins up front, and
    that pass needs a dense array. The same pass is what gives it native missing
    support -- a missing value gets its own bin and each split learns which side
    it belongs on.

    One documented limit, found while trying to refute the missing-value claim: a
    feature column with *no* observed value at all kills ``fit`` with a raw numpy
    ``ValueError("window shape cannot be larger than input array shape")`` out of
    that binning pass. A single observed value is enough to avoid it, so the
    capability itself stands. A user never meets it because S4 excludes an
    all-missing column before planning -- asserted in the integration tests
    rather than assumed here.
    """

    name: ClassVar[str] = "hist_gradient_boosting_classifier"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.CLASSIFICATION,
        backend=Backend.SKLEARN,
        # Verified: shaped (n_samples, n_classes), in [0, 1], rows summing to 1.
        supports_predict_proba=True,
        # Verified: identical predictions on raw and standardised features.
        requires_scaling=False,
        # Verified: raises TypeError("Sparse data was passed for X, but dense data
        # is required"). This is the model that exercises the dense path.
        supports_sparse_input=False,
        # Verified: three classes fitted directly.
        supports_multiclass=True,
        # Verified: fits with 30% of the informative feature missing, predicts on
        # unseen rows containing NaN, and classifies an entirely missing row.
        handles_missing_values=True,
        # Verified: a fitted instance exposes no feature_importances_ and no coef_.
        interpretability_level=Interpretability.LOW,
        is_baseline=False,
    )

    def default_params(self) -> dict[str, Any]:
        """Return conservative, untuned defaults.

        ``max_iter=100`` restates the scikit-learn default -- this estimator
        counts boosting iterations rather than estimators -- so the catalog
        reports the budget for both boosters in comparable terms.

        ``early_stopping`` is left at ``"auto"``, which enables it above 10,000
        rows. That is scikit-learn's decision and it is honest about it; turning
        it off would change the algorithm, and turning it on unconditionally would
        hold back a validation split the analyst never asked to give up.

        ``random_state`` fixes both the binning subsample and, where early
        stopping engages, the validation split.
        """
        return {"max_iter": 100, "random_state": self.config.random_state}

    def build(self, **params: Any) -> Estimator:
        """Construct a new, unfitted :class:`~sklearn.ensemble.HistGradientBoostingClassifier`."""
        return self._construct(
            HistGradientBoostingClassifier, self.resolve_params(**params), self.name
        )
