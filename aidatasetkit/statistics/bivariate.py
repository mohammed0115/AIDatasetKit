"""Statistics over two paired samples.

Both functions take their inputs through
:func:`~aidatasetkit.core.arrays.to_float_arrays`, which applies the
missing-value policy to whole *pairs*. Cleaning the two series independently would
shift them against each other and quietly measure the association between
mismatched observations.

Neither function invents a number when the association is undefined. A constant
input makes every correlation coefficient a division by zero; SciPy answers
``nan`` with a warning that is easy to miss, and this module raises instead.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
from scipy import stats

from aidatasetkit.core.arrays import NanPolicy, to_float_arrays
from aidatasetkit.core.exceptions import ValidationError
from aidatasetkit.statistics.guards import (
    require_defined_result,
    require_minimum_size,
    require_variation,
    strict_numerics,
    validate_order,
)

__all__ = ["covariance", "correlation", "CorrelationMethod"]

#: Supported measures of association.
CorrelationMethod = Literal["pearson", "spearman", "kendall"]

_CORRELATION_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], Any]] = {
    "pearson": stats.pearsonr,
    "spearman": stats.spearmanr,
    "kendall": stats.kendalltau,
}

#: Smallest sample each method needs before its result carries any meaning.
_MINIMUM_PAIRS: dict[str, int] = {"pearson": 2, "spearman": 2, "kendall": 2}


def covariance(
    x: Any,
    y: Any,
    *,
    ddof: int = 1,
    nan_policy: NanPolicy = "raise",
    allow_inf: bool = False,
) -> float:
    """Return the covariance of two paired samples.

    .. math::

        \\mathrm{cov}(x, y) =
        \\frac{1}{n - \\mathrm{ddof}} \\sum (x_i - \\bar{x})(y_i - \\bar{y})

    Args:
        x: First sample.
        y: Second sample, of the same length as ``x``.
        ddof: Delta degrees of freedom. ``0`` gives the population covariance,
            ``1`` -- the default -- the sample covariance.
        nan_policy: ``"raise"`` to reject missing values, ``"omit"`` to drop whole
            incomplete pairs.
        allow_inf: Whether infinities are permitted.

    Returns:
        The covariance. Unlike a correlation it is defined for constant input,
        where it is zero.

    Raises:
        ShapeError: If the two samples differ in length.
        DomainError: If the sample is too small for the requested ``ddof``.
    """
    validated_ddof = validate_order(ddof, name="ddof")
    x_values, y_values = to_float_arrays(
        x, y, nan_policy=nan_policy, allow_inf=allow_inf
    )
    require_minimum_size(
        x_values, validated_ddof + 1, f"Covariance with ddof={validated_ddof}"
    )

    with strict_numerics(f"Covariance with ddof={validated_ddof}"):
        deviations_x = x_values - x_values.mean()
        deviations_y = y_values - y_values.mean()
        return float(
            np.sum(deviations_x * deviations_y) / (x_values.size - validated_ddof)
        )


def correlation(
    x: Any,
    y: Any,
    *,
    method: CorrelationMethod = "pearson",
    nan_policy: NanPolicy = "raise",
    allow_inf: bool = False,
) -> float:
    """Return the correlation between two paired samples.

    Args:
        x: First sample.
        y: Second sample, of the same length as ``x``.
        method: ``"pearson"`` for the linear correlation coefficient,
            ``"spearman"`` for the rank correlation, or ``"kendall"`` for
            Kendall's tau-b.
        nan_policy: ``"raise"`` to reject missing values, ``"omit"`` to drop whole
            incomplete pairs.
        allow_inf: Whether infinities are permitted.

    Returns:
        The coefficient, in ``[-1, 1]``.

    Raises:
        ValidationError: If ``method`` is not one of the three supported names.
        ShapeError: If the two samples differ in length.
        DomainError: If the correlation is undefined -- fewer than two pairs, or
            either sample constant. No substitute value is returned in that case,
            because zero would read as "measured no association" when in fact
            nothing was measurable.
    """
    if method not in _CORRELATION_FUNCTIONS:
        raise ValidationError(
            f"method must be one of {sorted(_CORRELATION_FUNCTIONS)}, got {method!r}."
        )

    x_values, y_values = to_float_arrays(
        x, y, nan_policy=nan_policy, allow_inf=allow_inf
    )

    label = f"{method.capitalize()} correlation"
    require_minimum_size(x_values, _MINIMUM_PAIRS[method], label)
    require_variation(x_values, f"{label} (first sample)")
    require_variation(y_values, f"{label} (second sample)")

    with strict_numerics(label):
        result = _CORRELATION_FUNCTIONS[method](x_values, y_values)

    return require_defined_result(float(result.statistic), label)
