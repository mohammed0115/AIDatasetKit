"""Descriptive statistics for a single numeric sample.

:class:`StatisticsEngine` validates and converts its input once, then answers
questions about it. Every method is a measurement; none of them modifies data.
Cleaning happens elsewhere -- the only observations this class ever discards are
missing ones, and only when the caller asks for that with ``nan_policy="omit"``.

Two conventions are made explicit rather than inherited, because NumPy, pandas,
and SciPy disagree on both:

**Degrees of freedom.** ``ddof=0`` is the population variance, dividing by *n*.
``ddof=1`` is the sample variance, dividing by *n - 1*. This library defaults to
``ddof=1`` because analysts almost always hold a sample, but the parameter is
present on every affected method so the choice is never implicit.

**Bias correction.** ``skewness`` and ``kurtosis`` default to ``bias=False``, the
sample-size-corrected estimators, which match :meth:`pandas.Series.skew` and
:meth:`pandas.Series.kurt`. Passing ``bias=True`` gives the uncorrected moment
ratios, which are SciPy's defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, NamedTuple

import numpy as np
from scipy import stats

from aidatasetkit.core.arrays import NanPolicy, to_float_array
from aidatasetkit.core.exceptions import DomainError, ValidationError
from aidatasetkit.statistics.guards import (
    require_defined_result,
    require_minimum_size,
    require_positive_values,
    require_variation,
    strict_numerics,
    validate_order,
)
from aidatasetkit.statistics.moments import moment_of_values, validate_origin

__all__ = ["StatisticsEngine", "Quartiles", "DescriptiveSummary"]

#: Which centre :meth:`StatisticsEngine.mean_absolute_deviation` measures from.
Center = Literal["mean", "median"]


class Quartiles(NamedTuple):
    """The three quartiles of a sample."""

    q1: float
    q2: float
    q3: float


@dataclass(frozen=True, slots=True)
class DescriptiveSummary:
    """A typed summary of one numeric sample.

    ``skewness``, ``kurtosis``, and the dispersion measures are ``None`` when the
    sample is too small or too uniform for them to be defined. That is a reported
    absence, not a hidden failure: the individual methods still raise
    :class:`~aidatasetkit.core.exceptions.DomainError` for the same input, because
    a caller asking for one number wants to know it cannot be computed, whereas a
    caller asking for an overview wants the rest of the overview.
    """

    count: int
    n_omitted: int
    mean: float
    minimum: float
    maximum: float
    range: float
    median: float
    q25: float
    q75: float
    iqr: float
    variance: float | None
    std: float | None
    skewness: float | None
    kurtosis: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the summary."""
        return asdict(self)


class StatisticsEngine:
    """Descriptive statistics over a validated numeric sample.

    Args:
        data: A one-dimensional numeric sample: a list, tuple,
            :class:`numpy.ndarray`, :class:`pandas.Series`, or
            :class:`pandas.Index`.
        nan_policy: ``"raise"`` (the default) rejects missing values;
            ``"omit"`` drops them, and the count of dropped observations is
            reported by :attr:`n_omitted`.
        allow_inf: Whether infinities may take part in the calculations. False by
            default, because a single infinity turns most results into ``inf`` or
            ``nan``.

    Raises:
        ValidationError: If the input is not a usable numeric sample.
        EmptyDataError: If no observations remain.
        MissingValueError: If values are missing and ``nan_policy="raise"``.
        NonFiniteValueError: If values are infinite and ``allow_inf`` is false.

    Example:
        >>> engine = StatisticsEngine([2, 4, 4, 4, 5, 5, 7, 9])
        >>> engine.mean()
        5.0
        >>> engine.variance(ddof=0)
        4.0
    """

    def __init__(
        self,
        data: Any,
        *,
        nan_policy: NanPolicy = "raise",
        allow_inf: bool = False,
    ) -> None:
        observed = len(data) if hasattr(data, "__len__") else None
        self._values = to_float_array(
            data, nan_policy=nan_policy, allow_inf=allow_inf, name="data"
        )
        self._nan_policy = nan_policy
        self._allow_inf = allow_inf
        self._n_omitted = 0 if observed is None else observed - int(self._values.size)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(count={self.count}, "
            f"nan_policy={self._nan_policy!r})"
        )

    # ------------------------------------------------------------------ #
    # Sample
    # ------------------------------------------------------------------ #

    @property
    def values(self) -> np.ndarray:
        """A read-only view of the validated sample."""
        view = self._values.view()
        view.flags.writeable = False
        return view

    @property
    def count(self) -> int:
        """The number of observations used in every calculation."""
        return int(self._values.size)

    @property
    def n_omitted(self) -> int:
        """How many observations were dropped as missing."""
        return self._n_omitted

    # ------------------------------------------------------------------ #
    # Location
    # ------------------------------------------------------------------ #

    def mean(self) -> float:
        """Return the arithmetic mean, :math:`\\bar{x} = \\frac{1}{n} \\sum x_i`."""
        return float(np.mean(self._values))

    def median(self) -> float:
        """Return the median, interpolating between the two central values."""
        return float(np.median(self._values))

    def mode(self) -> tuple[float, ...]:
        """Return every most-frequent value, in ascending order.

        A distribution can have more than one mode, so this returns all of them
        rather than picking the smallest. When every observation occurs exactly
        once there is no mode and the result is empty -- which is itself the
        finding, and more useful than returning the whole sample.
        """
        distinct, counts = np.unique(self._values, return_counts=True)
        highest = int(counts.max())
        if highest == 1:
            return ()
        return tuple(float(value) for value in distinct[counts == highest])

    # ------------------------------------------------------------------ #
    # Extent
    # ------------------------------------------------------------------ #

    def min(self) -> float:
        """Return the smallest observation."""
        return float(np.min(self._values))

    def max(self) -> float:
        """Return the largest observation."""
        return float(np.max(self._values))

    def range(self) -> float:
        """Return the range, :math:`\\max(x) - \\min(x)`."""
        return float(np.ptp(self._values))

    def sum(self) -> float:
        """Return the sum of the observations."""
        return float(np.sum(self._values))

    def sum_of_squares(self) -> float:
        """Return the uncorrected sum of squares, :math:`\\sum x_i^2`.

        This is the sum of the squared *values*, not of the squared deviations
        from the mean. The name is ambiguous in the wider literature, so the
        formula is stated here explicitly.
        """
        return float(np.sum(np.square(self._values)))

    # ------------------------------------------------------------------ #
    # Dispersion
    # ------------------------------------------------------------------ #

    def variance(self, ddof: int = 1) -> float:
        """Return the variance.

        .. math:: s^2 = \\frac{1}{n - \\mathrm{ddof}} \\sum (x_i - \\bar{x})^2

        Args:
            ddof: Delta degrees of freedom. ``0`` gives the population variance
                (divide by *n*); ``1``, the default, gives the sample variance
                (divide by *n - 1*).

        Returns:
            The variance.

        Raises:
            ValidationError: If ``ddof`` is not a non-negative integer.
            DomainError: If the sample is too small for the requested ``ddof``.
        """
        validated_ddof = validate_order(ddof, name="ddof")
        require_minimum_size(
            self._values, validated_ddof + 1, f"variance with ddof={validated_ddof}"
        )
        with strict_numerics(f"variance with ddof={validated_ddof}"):
            return float(np.var(self._values, ddof=validated_ddof))

    def std(self, ddof: int = 1) -> float:
        """Return the standard deviation, the square root of :meth:`variance`.

        Args:
            ddof: Delta degrees of freedom. ``0`` is the population standard
                deviation, ``1`` the sample standard deviation.

        Returns:
            The standard deviation.

        Raises:
            ValidationError: If ``ddof`` is not a non-negative integer.
            DomainError: If the sample is too small for the requested ``ddof``.
        """
        validated_ddof = validate_order(ddof, name="ddof")
        require_minimum_size(
            self._values, validated_ddof + 1, f"standard deviation with ddof={validated_ddof}"
        )
        with strict_numerics(f"standard deviation with ddof={validated_ddof}"):
            return float(np.std(self._values, ddof=validated_ddof))

    def mean_absolute_deviation(self, center: Center = "mean") -> float:
        """Return the mean absolute deviation about the mean or the median.

        .. math:: \\mathrm{MAD} = \\frac{1}{n} \\sum |x_i - c|

        This is *not* the median absolute deviation, the robust estimator that
        shares the acronym MAD and takes the median of the absolute deviations
        rather than their mean. That statistic is not provided under this name.

        Args:
            center: ``"mean"`` or ``"median"``, the point :math:`c` to measure
                deviations from.

        Returns:
            The mean absolute deviation.

        Raises:
            ValidationError: If ``center`` is not one of the two accepted values.
        """
        if center == "mean":
            reference = self.mean()
        elif center == "median":
            reference = self.median()
        else:
            raise ValidationError(
                f'center must be "mean" or "median", got {center!r}.'
            )
        return float(np.mean(np.abs(self._values - reference)))

    # ------------------------------------------------------------------ #
    # Alternative means
    # ------------------------------------------------------------------ #

    def geometric_mean(self) -> float:
        """Return the geometric mean.

        .. math:: G = \\left( \\prod_{i=1}^{n} x_i \\right)^{1/n}

        Defined only for strictly positive values. A zero would collapse the
        product and a negative value would make the root complex, so both raise
        rather than returning the ``0.0`` that SciPy reports for a sample
        containing a zero.

        Returns:
            The geometric mean.

        Raises:
            DomainError: If any observation is zero or negative.
        """
        require_positive_values(self._values, "The geometric mean")
        with strict_numerics("The geometric mean"):
            return float(stats.gmean(self._values))

    def harmonic_mean(self) -> float:
        """Return the harmonic mean.

        .. math:: H = \\frac{n}{\\sum_{i=1}^{n} 1/x_i}

        Defined only for strictly positive values. A zero makes a reciprocal
        undefined; SciPy answers ``0.0`` without warning, which this method
        refuses to pass on.

        Returns:
            The harmonic mean.

        Raises:
            DomainError: If any observation is zero or negative.
        """
        require_positive_values(self._values, "The harmonic mean")
        with strict_numerics("The harmonic mean"):
            return float(stats.hmean(self._values))

    def rms(self) -> float:
        """Return the quadratic mean, also called the root mean square.

        .. math:: \\mathrm{RMS} = \\sqrt{\\frac{1}{n} \\sum x_i^2}

        This is a different statistic from :meth:`geometric_mean`; the two are
        equal only in degenerate cases. Unlike the geometric and harmonic means it
        accepts negative values, since the squaring removes the sign.
        """
        return float(np.sqrt(np.mean(np.square(self._values))))

    # ------------------------------------------------------------------ #
    # Position
    # ------------------------------------------------------------------ #

    def percentile(self, q: float | list[float] | np.ndarray) -> float | np.ndarray:
        """Return the value below which ``q`` percent of observations fall.

        Uses linear interpolation between the closest ranks, which is the default
        for both :func:`numpy.percentile` and :meth:`pandas.Series.quantile`.

        Args:
            q: A percentage in ``[0, 100]``, or a sequence of them.

        Returns:
            A float for a scalar ``q``, otherwise an array in the order requested.

        Raises:
            ValidationError: If any percentage falls outside ``[0, 100]`` or is
                not a real number.
        """
        requested = np.asarray(q, dtype="float64")
        if requested.ndim > 1:
            raise ValidationError(
                f"q must be a percentage or a one-dimensional sequence of them, got "
                f"an array with {requested.ndim} dimensions."
            )
        if not np.all(np.isfinite(requested)):
            raise ValidationError("q must contain only finite percentages.")
        if np.any((requested < 0) | (requested > 100)):
            raise ValidationError(
                f"q must lie between 0 and 100 inclusive, got {q!r}."
            )

        result = np.percentile(self._values, requested)
        return float(result) if requested.ndim == 0 else np.asarray(result)

    def quartiles(self) -> Quartiles:
        """Return the first, second, and third quartiles."""
        q1, q2, q3 = np.percentile(self._values, [25.0, 50.0, 75.0])
        return Quartiles(float(q1), float(q2), float(q3))

    def iqr(self) -> float:
        """Return the interquartile range, :math:`Q_3 - Q_1`."""
        quartiles = self.quartiles()
        return quartiles.q3 - quartiles.q1

    def z_scores(self, ddof: int = 0) -> np.ndarray:
        """Return the standardised observations.

        .. math:: z_i = \\frac{x_i - \\bar{x}}{s}

        Args:
            ddof: Delta degrees of freedom for the standard deviation in the
                denominator. Defaults to ``0``, matching
                :func:`scipy.stats.zscore`.

        Returns:
            An array of standardised values, in the order of the input.

        Raises:
            DomainError: If the sample is constant, which would divide by zero.
        """
        require_variation(self._values, "A z-score")
        deviation = self.std(ddof=ddof)
        return (self._values - self.mean()) / deviation

    # ------------------------------------------------------------------ #
    # Shape
    # ------------------------------------------------------------------ #

    def skewness(self, bias: bool = False) -> float:
        """Return the skewness, the third standardised moment.

        Args:
            bias: When false (the default) the sample-size-corrected estimator
                :math:`G_1` is returned, matching :meth:`pandas.Series.skew`. When
                true the uncorrected :math:`g_1` is returned, matching the default
                of :func:`scipy.stats.skew`.

        Returns:
            The skewness. Zero indicates symmetry; positive values indicate a
            longer right tail.

        Raises:
            DomainError: If the sample is constant, or too small for the chosen
                estimator -- three observations for the corrected form, two for
                the uncorrected one.
        """
        minimum = 2 if bias else 3
        require_minimum_size(self._values, minimum, "Skewness")
        require_variation(self._values, "Skewness")
        with strict_numerics("Skewness"):
            return require_defined_result(
                stats.skew(self._values, bias=bias), "Skewness"
            )

    def kurtosis(self, fisher: bool = True, bias: bool = False) -> float:
        """Return the kurtosis, the fourth standardised moment.

        Args:
            fisher: When true (the default) the excess kurtosis is returned, so a
                normal distribution scores ``0``. When false, Pearson's definition
                is used and a normal distribution scores ``3``.
            bias: When false (the default) the sample-size-corrected estimator
                :math:`G_2` is returned, matching :meth:`pandas.Series.kurt`. When
                true the uncorrected :math:`g_2` is returned, matching the default
                of :func:`scipy.stats.kurtosis`.

        Returns:
            The kurtosis.

        Raises:
            DomainError: If the sample is constant, or too small for the chosen
                estimator -- four observations for the corrected form, two for the
                uncorrected one.
        """
        minimum = 2 if bias else 4
        require_minimum_size(self._values, minimum, "Kurtosis")
        require_variation(self._values, "Kurtosis")
        with strict_numerics("Kurtosis"):
            return require_defined_result(
                stats.kurtosis(self._values, fisher=fisher, bias=bias), "Kurtosis"
            )

    # ------------------------------------------------------------------ #
    # Moments
    # ------------------------------------------------------------------ #

    def raw_moment(self, order: int) -> float:
        """Return the raw moment of ``order`` about zero.

        .. math:: m'_k = \\frac{1}{n} \\sum x_i^k

        Order ``1`` is the arithmetic mean.
        """
        return moment_of_values(self._values, validate_order(order), center=0.0)

    def central_moment(self, order: int) -> float:
        """Return the central moment of ``order`` about the mean.

        .. math:: m_k = \\frac{1}{n} \\sum (x_i - \\bar{x})^k

        Order ``1`` is zero and order ``2`` is the population variance.
        """
        return moment_of_values(self._values, validate_order(order), center=None)

    def moment_about(self, origin: float, order: int) -> float:
        """Return the moment of ``order`` about an arbitrary ``origin``.

        .. math:: m_k(c) = \\frac{1}{n} \\sum (x_i - c)^k
        """
        return moment_of_values(
            self._values, validate_order(order), center=validate_origin(origin)
        )

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def describe(self) -> DescriptiveSummary:
        """Return a typed overview of the sample.

        Statistics that are undefined for this sample -- dispersion for a single
        observation, shape for a constant or very small one -- are reported as
        ``None`` rather than omitted or faked.
        """
        quartiles = self.quartiles()
        return DescriptiveSummary(
            count=self.count,
            n_omitted=self.n_omitted,
            mean=self.mean(),
            minimum=self.min(),
            maximum=self.max(),
            range=self.range(),
            median=quartiles.q2,
            q25=quartiles.q1,
            q75=quartiles.q3,
            iqr=quartiles.q3 - quartiles.q1,
            variance=_or_none(self.variance),
            std=_or_none(self.std),
            skewness=_or_none(self.skewness),
            kurtosis=_or_none(self.kurtosis),
        )


def _or_none(measure: Any) -> float | None:
    """Return the measurement, or ``None`` when it is undefined for this sample."""
    try:
        return measure()
    except DomainError:
        return None
