"""Least squares, with and without a penalty.

Both declare ``requires_scaling=True``, and the interesting part is that they
arrive there for entirely different reasons -- one mathematical, one numerical.
Neither reason is the one the rule of thumb gives.

**Ridge is not scale-equivariant, because the penalty is not.** ``alpha``
shrinks coefficients, and how big a coefficient is depends on the unit its
feature was recorded in: a quantity measured in thousands earns a small
coefficient and is barely penalised, while the same information recorded as a
ratio earns a large one and is crushed. On an ordinary customer-shaped frame --
an income near 60,000, a ratio near 0.5, a tenure in months, condition number
``5.2e5``, nothing exotic -- Ridge scored ``R2 = 0.900515`` unscaled against
``0.997057`` scaled, and the coefficient on the ratio came back as ``372`` where
the truth was ``900``. That is the penalty deciding which feature matters by
reading the unit it was written in. Scaling is what stops it.

**Ordinary least squares is scale-equivariant in exact arithmetic, and the
solver is not.** Multiply a column by a thousand and its coefficient divides by
a thousand, leaving every prediction where it was: measured at ``8.2e-13`` on a
well-conditioned matrix. But scikit-learn solves it through
``scipy.linalg.lstsq`` with ``tol=1e-6`` passed as ``cond``, which truncates
singular values below that fraction of the largest -- and a column-magnitude
ratio of about ``1e6`` is enough to push a genuinely informative feature under
the cut. It is then *silently discarded*: no error, no warning, a coefficient of
``-1.1e-15``, and a fit that looks like it worked.

.. rubric:: How ordinary the failing case is

This is the whole reason the declaration is ``True`` rather than ``False``.
Measured on frames nobody would call unusual:

=====================================  ======  ============  ============
Frame                                  rank_   R2 unscaled   R2 scaled
=====================================  ======  ============  ============
revenue in dollars, conversion rate    1 of 2  ``0.607``     ``0.999996``
income, parts-per-thousand rate, +2    3 of 4  ``0.984``     ``1.000000``
income, ratio near 0.5, tenure         3 of 3  ``0.997074``  ``0.997074``
=====================================  ======  ============  ============

The third row is the frame this library's own integration tests use, and it sits
just under the cliff -- which is exactly why an earlier version of this module
declared ``False`` on the strength of it. A dollar amount beside a proportion
differs by ``1e7`` or more and is completely ordinary; the third row was not
representative, and treating it as though it were meant routing realistic data
past the scaler.

The cost of declaring ``True`` is real and worth naming: the fitted coefficients
become per-standard-deviation rather than per-unit, which is less directly
readable, and interpretability stays ``HIGH`` because one coefficient per feature
is still exactly what you get. That cost is smaller than a silently rank-deficient
fit. A model that stopped short because of a condition nobody was told about is
the failure this project exists to prevent, and ``lstsq`` truncating a column is
that failure with no warning attached at all.

A fitted estimator reports ``rank_`` and ``singular_``, so the condition remains
detectable by anyone who wants to check a fit they ran themselves.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.linear_model import LinearRegression, Ridge

from aidatasetkit.core.types import Backend, Estimator, Interpretability, TaskType
from aidatasetkit.models.base import ModelStrategy
from aidatasetkit.models.capabilities import ModelCapabilities
from aidatasetkit.models.registry import register_model

__all__ = ["LinearRegressionStrategy", "RidgeRegressionStrategy"]


@register_model()
class LinearRegressionStrategy(ModelStrategy):
    """Unpenalised least squares, readable one coefficient per feature.

    No alias is registered. ``linear_regression`` is already the short, ordinary
    name for this model, and an alias that only restated it would occupy the
    namespace without helping anyone.
    """

    name: ClassVar[str] = "linear_regression"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.REGRESSION,
        backend=Backend.SKLEARN,
        supports_predict_proba=False,
        supports_multiclass=False,
        # Verified, and NOT for the reason the rule of thumb gives. The maths is
        # scale-equivariant -- raw and standardised fits agree to 8.2e-13 on a
        # well-conditioned matrix. The solver is not: scikit-learn passes
        # tol=1e-6 to scipy.linalg.lstsq as cond, so a column-magnitude ratio
        # near 1e6 truncates a real singular value and silently drops an
        # informative feature (rank_ 1 of 2, coefficient -1.1e-15, R2 0.607
        # against 0.999996 scaled, on revenue-in-dollars beside a conversion
        # rate). That is an ordinary pair of columns, not an exotic one.
        requires_scaling=True,
        # Verified: fits and predicts on a scipy csr_matrix, and the coefficients
        # are identical to the dense fit.
        supports_sparse_input=True,
        # Verified: raises ValueError("Input X contains NaN"). A least-squares
        # solve has no path for a missing coordinate.
        handles_missing_values=False,
        interpretability_level=Interpretability.HIGH,
        is_baseline=False,
    )

    def default_params(self) -> dict[str, Any]:
        """Return conservative, untuned defaults.

        ``fit_intercept=True`` restates the scikit-learn default. It is written
        out because it is the one parameter that changes what model is being
        fitted -- forcing the line through the origin is a claim about the data,
        not a tuning knob -- and because the catalog reports ``default_params``,
        where an analyst asking "what am I actually getting" deserves to see it.
        Pinning it also means a change to scikit-learn's default cannot silently
        change every prediction this library makes.

        ``random_state`` is deliberately absent: the constructor does not accept
        one, and the solve is deterministic.

        ``tol`` is deliberately left at the scikit-learn default of ``1e-6``,
        even though it is the parameter behind the rank truncation described in
        the module docstring. Lowering it would trade a silent dropped column
        for a silently ill-conditioned solve, which is not an improvement, and
        choosing a value for it would be tuning. Scaling is the correct answer to
        conditioning, and that is what ``requires_scaling=True`` asks for.
        """
        return {"fit_intercept": True}

    def build(self, **params: Any) -> Estimator:
        """Construct a new, unfitted :class:`~sklearn.linear_model.LinearRegression`."""
        return self._construct(
            LinearRegression, self.resolve_params(**params), self.name
        )


@register_model(aliases=("ridge",))
class RidgeRegressionStrategy(ModelStrategy):
    """Least squares with an L2 penalty on the coefficients."""

    name: ClassVar[str] = "ridge_regression"

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        task_type=TaskType.REGRESSION,
        backend=Backend.SKLEARN,
        supports_predict_proba=False,
        supports_multiclass=False,
        # Verified on an ordinary customer-shaped frame, not a contrived one:
        # R2 0.900515 unscaled against 0.997057 scaled, with the coefficient on
        # the small-magnitude feature shrunk from 900 to 372. The penalty reads
        # the unit a column was recorded in; scaling is what stops it.
        requires_scaling=True,
        # Verified: fits and predicts on a scipy csr_matrix. The default
        # solver="auto" takes a different route for sparse input than for dense,
        # and the two agreed to 1.9e-4 on the same data -- the same model, solved
        # differently. Reproducibility therefore holds per input form; it is not
        # a promise that a dense and a sparse fit are bit-identical.
        supports_sparse_input=True,
        # Verified: raises ValueError("Input X contains NaN").
        handles_missing_values=False,
        interpretability_level=Interpretability.HIGH,
        is_baseline=False,
    )

    def default_params(self) -> dict[str, Any]:
        """Return conservative, untuned defaults.

        ``alpha=1.0`` restates the scikit-learn default. It is written out
        because it *is* the model: alpha is the entire difference between this
        strategy and :class:`LinearRegressionStrategy`, and a catalog that did
        not report how much shrinkage an analyst is getting would be hiding the
        one number that matters. No search was run and no data was consulted;
        choosing alpha by looking at the data is tuning, and tuning is not S6's
        work.

        ``random_state`` is inert for the default ``solver="auto"``, which
        resolves to a deterministic route -- measured: four different seeds
        produced one identical fit. It is set anyway so that switching to
        ``"sag"`` or ``"saga"``, which do draw on it, stays reproducible without
        the caller having to remember.
        """
        return {"alpha": 1.0, "random_state": self.config.random_state}

    def build(self, **params: Any) -> Estimator:
        """Construct a new, unfitted :class:`~sklearn.linear_model.Ridge`."""
        return self._construct(Ridge, self.resolve_params(**params), self.name)
