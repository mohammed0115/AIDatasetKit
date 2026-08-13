"""Gradient-boosted regression trees, in scikit-learn's two implementations.

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

Both were measured on the regressors, not carried over from the classifiers.
``HistGradientBoostingRegressor`` refuses a ``csr_matrix`` with
``TypeError("Sparse data was passed for X, but dense data is required")``, and
``GradientBoostingRegressor`` refuses NaN with ``ValueError("Input X contains
NaN")``. Declaring those honestly is the whole mechanism: the two models land on
different :class:`~aidatasetkit.core.types.PreprocessingProfile` keys, so S4
densifies for one and imputes for the other without any code anywhere asking
which model this is.

Neither exposes a readable decision function. ``GradientBoostingRegressor``
publishes ``feature_importances_`` and so reports ``MEDIUM``;
``HistGradientBoostingRegressor`` publishes no importances at all -- verified on
a fitted instance -- and so reports ``LOW``. That is a statement about what the
fitted object lets you read, not about accuracy.

Both are also the two regressors here that refuse a two-column ``y``, answering
``ValueError("y should be a 1d array")``. Every other regressor in the catalog
accepts one. Nothing in this library produces a multi-output target, so that is
recorded as a measurement rather than turned into a capability field.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from aidatasetkit.core.types import Backend, Estimator, Interpretability, TaskType
from aidatasetkit.models.base import ModelStrategy
from aidatasetkit.models.capabilities import ModelCapabilities
from aidatasetkit.models.registry import register_model

__all__ = [
    "GradientBoostingRegressorStrategy",
    "HistGradientBoostingRegressorStrategy",
]


@register_model(aliases=("gradient_boosting",))
class GradientBoostingRegressorStrategy(ModelStrategy):
    """Stage-wise boosting of small regression trees on the loss gradient."""

    name: ClassVar[str] = "gradient_boosting_regressor"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.REGRESSION,
        backend=Backend.SKLEARN,
        supports_predict_proba=False,
        supports_multiclass=False,
        # Verified: predictions on raw and standardised features agreed to
        # 0.029 of a standard deviation of the target, with R2 0.8964 against
        # 0.8987 -- the residual wobble of a stage-wise fit, not a scale
        # sensitivity. A split is still a threshold on one feature.
        requires_scaling=False,
        # Verified: fits and predicts on a scipy csr_matrix.
        supports_sparse_input=True,
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
        breaking tied splits. Measured **at these shipped defaults**, on a noisy
        correlated 300-row fit scored on held-out rows: four seeds moved the
        predictions by up to ``0.24``. At five stages the same four seeds agree
        to ``3.3e-16``, which is why the contract test measures a spread rather
        than asking whether the bytes differ.
        """
        return {"n_estimators": 100, "random_state": self.config.random_state}

    def build(self, **params: Any) -> Estimator:
        """Construct a new, unfitted :class:`~sklearn.ensemble.GradientBoostingRegressor`."""
        return self._construct(
            GradientBoostingRegressor, self.resolve_params(**params), self.name
        )


@register_model(aliases=("hist_gradient_boosting",))
class HistGradientBoostingRegressorStrategy(ModelStrategy):
    """Boosting over binned features, with a learned direction for missing values.

    The binning is what makes it fast and what makes it dense-only: the
    implementation histograms each feature into at most 255 bins up front, and
    that pass needs a dense array. The same pass is what gives it native missing
    support -- a missing value gets its own bin and each split learns which side
    it belongs on.

    One documented limit, re-measured on the regressor: a feature column with *no*
    observed value at all kills ``fit`` with a raw numpy ``ValueError("window
    shape cannot be larger than input array shape")`` out of that binning pass. A
    single observed value is enough to avoid it, so the capability itself stands.
    A user never meets it because S4 excludes an all-missing column before
    planning -- asserted in the integration tests rather than assumed here.

    It is also the one model in this module that survives a feature above the
    ``float32`` range. The tree family, the other booster, and every estimator
    that casts to ``float32`` raise on a finite ``float64`` value near ``1e39``;
    this one bins instead of casting and fits. That is a difference between
    backends, not a quality of the algorithm, and it is recorded in
    ``docs/limitations.md`` rather than turned into a capability nothing consults.
    """

    name: ClassVar[str] = "hist_gradient_boosting_regressor"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.REGRESSION,
        backend=Backend.SKLEARN,
        supports_predict_proba=False,
        supports_multiclass=False,
        # Verified: identical predictions on raw and standardised features.
        requires_scaling=False,
        # Verified: raises TypeError("Sparse data was passed for X, but dense data
        # is required"). This is the regressor that exercises the dense path.
        supports_sparse_input=False,
        # Verified: fits with 30% of the informative feature missing, predicts on
        # rows containing NaN, and returns finite values throughout.
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

        ``early_stopping="auto"`` restates scikit-learn's default, and it is
        written out because of what it does rather than to pad the list. Measured
        at the published value: at 500 rows ``do_early_stopping_`` is ``False``,
        and at 10,001 rows it is ``True`` and a tenth of the training data is
        withheld as a validation set. An analyst reading ``default_params`` in the
        catalog to answer "what am I actually getting" deserves to see that. The
        value is not changed: turning it off would alter the algorithm, and
        forcing it on would give up a validation split nobody asked to give up.

        ``random_state`` fixes both the binning subsample and, where early
        stopping engages, the validation split. Whether it matters depends on the
        data, and both halves were measured: on a 300-row frame with early
        stopping inactive, four seeds produced one identical model; above 10,000
        training rows early stopping turns on, three seeds stopped at iterations
        ``59``, ``45`` and ``47``, and the held-out predictions moved by ``0.48``.
        So the seed is published because it becomes load-bearing exactly where
        the algorithm starts making a decision with it.
        """
        return {
            "max_iter": 100,
            "early_stopping": "auto",
            "random_state": self.config.random_state,
        }

    def build(self, **params: Any) -> Estimator:
        """Construct a new, unfitted :class:`~sklearn.ensemble.HistGradientBoostingRegressor`."""
        return self._construct(
            HistGradientBoostingRegressor, self.resolve_params(**params), self.name
        )
