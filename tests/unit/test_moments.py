"""Tests for the moment functions.

The mathematical identities are checked directly, because they are the property
most likely to be broken by a formula copied across from an older implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from aidatasetkit.core.exceptions import (
    MissingValueError,
    NonFiniteValueError,
    ValidationError,
)
from aidatasetkit.statistics import central_moment, moment_about, raw_moment

SAMPLE = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
ARRAY = np.array(SAMPLE)


class TestIdentities:
    def test_zeroth_moment_is_one(self):
        assert raw_moment(SAMPLE, 0) == pytest.approx(1.0)
        assert central_moment(SAMPLE, 0) == pytest.approx(1.0)
        assert moment_about(SAMPLE, 3.0, 0) == pytest.approx(1.0)

    def test_first_raw_moment_is_the_mean(self):
        assert raw_moment(SAMPLE, 1) == pytest.approx(np.mean(ARRAY))

    def test_first_central_moment_is_zero(self):
        assert central_moment(SAMPLE, 1) == pytest.approx(0.0, abs=1e-12)

    def test_second_central_moment_is_the_population_variance(self):
        assert central_moment(SAMPLE, 2) == pytest.approx(np.var(ARRAY, ddof=0))

    def test_second_central_moment_is_not_the_sample_variance(self):
        assert central_moment(SAMPLE, 2) != pytest.approx(np.var(ARRAY, ddof=1))

    def test_moment_about_zero_equals_the_raw_moment(self):
        for order in range(5):
            assert moment_about(SAMPLE, 0.0, order) == pytest.approx(
                raw_moment(SAMPLE, order)
            )

    def test_moment_about_the_mean_equals_the_central_moment(self):
        mean = float(np.mean(ARRAY))
        for order in range(5):
            assert moment_about(SAMPLE, mean, order) == pytest.approx(
                central_moment(SAMPLE, order)
            )


class TestAgainstReferences:
    @pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 5])
    def test_central_moment_matches_scipy(self, order):
        assert central_moment(SAMPLE, order) == pytest.approx(
            stats.moment(ARRAY, order)  # positional: the keyword is order= only from SciPy 1.12
        )

    @pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 5])
    def test_raw_moment_matches_its_definition(self, order):
        assert raw_moment(SAMPLE, order) == pytest.approx(np.mean(ARRAY**order))

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    @pytest.mark.parametrize("origin", [-2.5, 0.0, 3.0, 100.0])
    def test_moment_about_matches_its_definition(self, order, origin):
        assert moment_about(SAMPLE, origin, order) == pytest.approx(
            np.mean((ARRAY - origin) ** order)
        )

    def test_third_central_moment_relates_to_skewness(self):
        """Uncorrected skewness is the third central moment over sigma cubed."""
        third = central_moment(SAMPLE, 3)
        sigma = np.std(ARRAY, ddof=0)
        assert third / sigma**3 == pytest.approx(stats.skew(ARRAY))

    def test_fourth_central_moment_relates_to_kurtosis(self):
        fourth = central_moment(SAMPLE, 4)
        sigma = np.std(ARRAY, ddof=0)
        assert fourth / sigma**4 - 3.0 == pytest.approx(stats.kurtosis(ARRAY))


class TestInputHandling:
    @pytest.mark.parametrize(
        "data", [SAMPLE, ARRAY, pd.Series(SAMPLE), tuple(SAMPLE), np.array([2, 4, 6])]
    )
    def test_supported_containers_are_accepted(self, data):
        assert isinstance(raw_moment(data, 2), float)

    def test_missing_values_raise_by_default(self):
        with pytest.raises(MissingValueError):
            central_moment([1.0, np.nan, 3.0], 2)

    def test_omission_is_explicit(self):
        assert central_moment([1.0, np.nan, 3.0], 2, nan_policy="omit") == pytest.approx(
            np.var([1.0, 3.0], ddof=0)
        )

    def test_infinities_raise_by_default(self):
        with pytest.raises(NonFiniteValueError):
            raw_moment([1.0, np.inf], 2)

    def test_a_single_observation_has_defined_moments(self):
        assert raw_moment([4.0], 2) == pytest.approx(16.0)
        assert central_moment([4.0], 2) == pytest.approx(0.0)

    def test_a_constant_sample_has_zero_central_moments(self):
        for order in (1, 2, 3, 4):
            assert central_moment([3.0] * 5, order) == pytest.approx(0.0)


class TestOrderValidation:
    @pytest.mark.parametrize("order", [-1, -5])
    def test_negative_orders_are_rejected(self, order):
        with pytest.raises(ValidationError, match="non-negative"):
            raw_moment(SAMPLE, order)

    @pytest.mark.parametrize("order", [1.5, "2", None, [2]])
    def test_non_integer_orders_are_rejected(self, order):
        with pytest.raises(ValidationError, match="integer"):
            central_moment(SAMPLE, order)

    def test_booleans_are_not_accepted_as_orders(self):
        """True would otherwise slip through as the integer 1."""
        with pytest.raises(ValidationError, match="integer"):
            raw_moment(SAMPLE, True)

    def test_numpy_integers_are_accepted(self):
        assert raw_moment(SAMPLE, np.int64(2)) == pytest.approx(
            raw_moment(SAMPLE, 2)
        )

    @pytest.mark.parametrize("origin", ["3", None, [1.0], np.nan, np.inf, True])
    def test_invalid_origins_are_rejected(self, origin):
        with pytest.raises(ValidationError, match="origin"):
            moment_about(SAMPLE, origin, 2)

    def test_integer_origins_are_accepted(self):
        assert moment_about(SAMPLE, 3, 2) == pytest.approx(
            moment_about(SAMPLE, 3.0, 2)
        )
