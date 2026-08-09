"""Moments of a distribution.

Three functions cover what the original Java library split across "moments about
the mean", "moments about zero", and "moments about an arbitrary origin". They
are one formula with a different centre:

.. math::

    m_k(c) = \\frac{1}{n} \\sum_{i=1}^{n} (x_i - c)^k

============================  ===========  ==========================
Function                      Centre       Old Java name
============================  ===========  ==========================
:func:`raw_moment`            ``0``        moments about zero
:func:`central_moment`        mean         moments about the mean
:func:`moment_about`          ``origin``   moments about arbitrary origin
============================  ===========  ==========================

The identities that follow from the definition are enforced by the test suite:
``raw_moment(x, 1)`` is the mean, ``central_moment(x, 1)`` is zero, and
``central_moment(x, 2)`` is the population variance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from aidatasetkit.core.arrays import NanPolicy, to_float_array
from aidatasetkit.core.exceptions import ValidationError
from aidatasetkit.statistics.guards import strict_numerics, validate_order

__all__ = ["raw_moment", "central_moment", "moment_about", "moment_of_values", "validate_origin"]


def moment_of_values(values: np.ndarray, order: int, center: float | None) -> float:
    """Compute a moment from an already-validated array.

    Args:
        values: A validated one-dimensional ``float64`` array.
        order: A non-negative integer order.
        center: The point to take the moment about, or ``None`` for the mean.

    Returns:
        The moment as a float. Order ``0`` is ``1.0`` by definition.

    Note:
        A constant sample is computed directly rather than through SciPy. Every
        deviation is then exactly zero, which SciPy reports as catastrophic
        cancellation even though the answer is exact and uninteresting.
    """
    if values.size and float(np.ptp(values)) == 0.0:
        return _constant_sample_moment(float(values[0]), order, center)

    with strict_numerics(f"moment of order {order}"):
        if center is None:
            result = stats.moment(values, order=order)
        else:
            result = stats.moment(values, order=order, center=center)
    return float(result)


def _constant_sample_moment(value: float, order: int, center: float | None) -> float:
    """Return the exact moment of a sample whose observations are all identical."""
    if order == 0:
        return 1.0
    reference = value if center is None else center
    return float((value - reference) ** order)


def raw_moment(
    data: Any,
    order: int,
    *,
    nan_policy: NanPolicy = "raise",
    allow_inf: bool = False,
) -> float:
    """Return the raw moment of ``order`` about zero.

    .. math:: m'_k = \\frac{1}{n} \\sum x_i^k

    Args:
        data: A one-dimensional numeric sample.
        order: A non-negative integer order. Order ``1`` is the arithmetic mean.
        nan_policy: ``"raise"`` to reject missing values, ``"omit"`` to drop them.
        allow_inf: Whether infinities are permitted.

    Returns:
        The raw moment.

    Raises:
        ValidationError: If ``order`` is not a non-negative integer.
        DomainError: If the moment is undefined for this input.
    """
    validated_order = validate_order(order)
    values = to_float_array(data, nan_policy=nan_policy, allow_inf=allow_inf)
    return moment_of_values(values, validated_order, center=0.0)


def central_moment(
    data: Any,
    order: int,
    *,
    nan_policy: NanPolicy = "raise",
    allow_inf: bool = False,
) -> float:
    """Return the central moment of ``order`` about the mean.

    .. math:: m_k = \\frac{1}{n} \\sum (x_i - \\bar{x})^k

    Order ``2`` is the population variance, that is the variance with
    ``ddof=0`` -- not the sample variance.

    Args:
        data: A one-dimensional numeric sample.
        order: A non-negative integer order.
        nan_policy: ``"raise"`` to reject missing values, ``"omit"`` to drop them.
        allow_inf: Whether infinities are permitted.

    Returns:
        The central moment.

    Raises:
        ValidationError: If ``order`` is not a non-negative integer.
        DomainError: If the moment is undefined for this input.
    """
    validated_order = validate_order(order)
    values = to_float_array(data, nan_policy=nan_policy, allow_inf=allow_inf)
    return moment_of_values(values, validated_order, center=None)


def moment_about(
    data: Any,
    origin: float,
    order: int,
    *,
    nan_policy: NanPolicy = "raise",
    allow_inf: bool = False,
) -> float:
    """Return the moment of ``order`` about an arbitrary ``origin``.

    .. math:: m_k(c) = \\frac{1}{n} \\sum (x_i - c)^k

    Args:
        data: A one-dimensional numeric sample.
        origin: The point to take the moment about.
        order: A non-negative integer order.
        nan_policy: ``"raise"`` to reject missing values, ``"omit"`` to drop them.
        allow_inf: Whether infinities are permitted.

    Returns:
        The moment about ``origin``.

    Raises:
        ValidationError: If ``order`` is not a non-negative integer, or ``origin``
            is not a finite number.
        DomainError: If the moment is undefined for this input.
    """
    validated_order = validate_order(order)
    validated_origin = validate_origin(origin)
    values = to_float_array(data, nan_policy=nan_policy, allow_inf=allow_inf)
    return moment_of_values(values, validated_order, center=validated_origin)


def validate_origin(origin: float) -> float:
    """Validate that ``origin`` is a finite real number."""
    if isinstance(origin, bool) or not isinstance(origin, (int, float, np.integer, np.floating)):
        raise ValidationError(
            f"origin must be a real number, got {type(origin).__name__}."
        )
    value = float(origin)
    if not np.isfinite(value):
        raise ValidationError(f"origin must be finite, got {origin!r}.")
    return value
