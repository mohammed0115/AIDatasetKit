"""Numerical preconditions shared by the statistics package.

NumPy and SciPy answer degenerate input with ``nan``, with a ``RuntimeWarning``,
or -- worst of all -- with a plausible number. Measured on the pinned stack:

===============================  ==========================================
Call                             Result
===============================  ==========================================
``hmean([1, 2, 0])``             ``0.0``, no warning at all
``gmean([1, 2, 0])``             ``0.0``, no warning at all
``skew([3, 3, 3, 3])``           ``nan`` with a ``RuntimeWarning``
``var([5], ddof=1)``             ``nan`` with a ``RuntimeWarning``
``pearsonr(constant, y)``        ``nan`` with a ``ConstantInputWarning``
``skew([1, 2], bias=False)``     ``0.0``, no warning
===============================  ==========================================

A library that passes those through produces a report full of ``nan`` whose cause
surfaces three steps later. Every statistic in this package therefore checks its
preconditions first and raises :class:`~aidatasetkit.core.exceptions.DomainError`
with the reason, and wraps the underlying call so that a warning cannot escape
silently either.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
from scipy import stats

from aidatasetkit.core.exceptions import DomainError, ValidationError

__all__ = [
    "require_minimum_size",
    "require_positive_values",
    "require_variation",
    "require_defined_result",
    "validate_order",
    "strict_numerics",
]

#: SciPy warning categories that signal a mathematically degenerate input.
_DEGENERACY_WARNINGS: tuple[type[Warning], ...] = (
    RuntimeWarning,
    stats.ConstantInputWarning,
    stats.NearConstantInputWarning,
)


def require_minimum_size(values: np.ndarray, minimum: int, operation: str) -> None:
    """Reject a sample too small for ``operation`` to be defined.

    Raises:
        DomainError: If fewer than ``minimum`` observations are present.
    """
    if values.size < minimum:
        raise DomainError(
            f"{operation} requires at least {minimum} observation(s), got {values.size}."
        )


def require_positive_values(values: np.ndarray, operation: str) -> None:
    """Reject non-positive values for a strictly-positive-domain statistic.

    Raises:
        DomainError: If any value is zero or negative.
    """
    offending = values[values <= 0]
    if offending.size:
        raise DomainError(
            f"{operation} is defined only for strictly positive values, but the input "
            f"contains {offending.size} value(s) that are zero or negative "
            f"(for example {offending[0]!r})."
        )


def require_variation(values: np.ndarray, operation: str) -> None:
    """Reject a constant sample where ``operation`` would divide by zero.

    Raises:
        DomainError: If every observation is identical.
    """
    # ``ptp`` subtracts the extremes, which overflows for values near the float64
    # ceiling; the overflowed answer is infinity, which is not zero, which is the
    # right verdict -- so the warning is silenced rather than acted on.
    with np.errstate(over="ignore"):
        constant = bool(values.size) and float(np.ptp(values)) == 0.0
    if constant:
        raise DomainError(
            f"{operation} is undefined for a constant input; every one of the "
            f"{values.size} observations equals {values.flat[0]!r}."
        )


def require_defined_result(result: float, operation: str) -> float:
    """Reject a non-finite result rather than reporting it as a measurement.

    Raises:
        DomainError: If the computed value is ``nan``.
    """
    if np.isnan(result):
        raise DomainError(
            f"{operation} is mathematically undefined for this input; no value is "
            "reported rather than substituting a placeholder."
        )
    return float(result)


def validate_order(order: int, name: str = "order") -> int:
    """Validate a moment order.

    Args:
        order: The requested moment order.
        name: Parameter label used in error messages.

    Returns:
        The order as a plain ``int``.

    Raises:
        ValidationError: If the order is not a non-negative integer. Booleans are
            rejected explicitly, since ``True`` would otherwise pass as ``1``.
    """
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
        raise ValidationError(
            f"{name} must be a non-negative integer, got {type(order).__name__}."
        )
    if order < 0:
        raise ValidationError(f"{name} must be non-negative, got {order}.")
    return int(order)


@contextmanager
def strict_numerics(operation: str) -> Iterator[None]:
    """Turn numerical-degeneracy warnings from the reference libraries into errors.

    A second line of defence behind the explicit precondition checks: if SciPy or
    NumPy reports a degenerate input through a warning, it becomes a
    :class:`~aidatasetkit.core.exceptions.DomainError` instead of a number nobody
    inspects. Unrelated warning categories are left untouched.
    """
    with warnings.catch_warnings():
        for category in _DEGENERACY_WARNINGS:
            warnings.simplefilter("error", category)
        try:
            yield
        except _DEGENERACY_WARNINGS as error:
            raise DomainError(
                f"{operation} is not defined for this input: {error}"
            ) from error
