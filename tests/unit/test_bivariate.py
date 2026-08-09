"""Tests for covariance and correlation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from aidatasetkit.core.exceptions import (
    DomainError,
    EmptyDataError,
    MissingValueError,
    NonFiniteValueError,
    ShapeError,
    ValidationError,
)
from aidatasetkit.statistics import correlation, covariance

X = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
Y = [1.0, 3.0, 2.0, 5.0, 4.0, 8.0, 6.0, 9.0]


class TestCovariance:
    def test_sample_covariance_matches_numpy(self):
        assert covariance(X, Y, ddof=1) == pytest.approx(np.cov(X, Y, ddof=1)[0, 1])

    def test_population_covariance_matches_numpy(self):
        assert covariance(X, Y, ddof=0) == pytest.approx(np.cov(X, Y, ddof=0)[0, 1])

    def test_sample_covariance_matches_pandas(self):
        assert covariance(X, Y) == pytest.approx(pd.Series(X).cov(pd.Series(Y)))

    def test_the_default_is_the_sample_covariance(self):
        assert covariance(X, Y) == covariance(X, Y, ddof=1)

    def test_the_two_conventions_differ(self):
        assert covariance(X, Y, ddof=0) != pytest.approx(covariance(X, Y, ddof=1))

    def test_it_is_symmetric(self):
        assert covariance(X, Y) == pytest.approx(covariance(Y, X))

    def test_covariance_with_itself_is_the_variance(self):
        assert covariance(X, X, ddof=1) == pytest.approx(np.var(X, ddof=1))

    def test_a_constant_input_gives_zero_covariance(self):
        """Unlike a correlation, this is well defined rather than undefined."""
        assert covariance([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)

    def test_a_single_pair_has_no_sample_covariance(self):
        with pytest.raises(DomainError, match="at least 2"):
            covariance([1.0], [2.0], ddof=1)

    def test_a_single_pair_has_a_population_covariance(self):
        assert covariance([1.0], [2.0], ddof=0) == pytest.approx(0.0)


class TestCorrelation:
    def test_pearson_matches_scipy(self):
        assert correlation(X, Y) == pytest.approx(stats.pearsonr(X, Y).statistic)

    def test_pearson_matches_pandas(self):
        assert correlation(X, Y) == pytest.approx(pd.Series(X).corr(pd.Series(Y)))

    def test_spearman_matches_scipy(self):
        assert correlation(X, Y, method="spearman") == pytest.approx(
            stats.spearmanr(X, Y).statistic
        )

    def test_kendall_matches_scipy(self):
        assert correlation(X, Y, method="kendall") == pytest.approx(
            stats.kendalltau(X, Y).statistic
        )

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_perfect_positive_association_is_one(self, method):
        assert correlation([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], method=method) == (
            pytest.approx(1.0)
        )

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_perfect_negative_association_is_minus_one(self, method):
        assert correlation([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], method=method) == (
            pytest.approx(-1.0)
        )

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_every_coefficient_stays_within_bounds(self, method):
        assert -1.0 <= correlation(X, Y, method=method) <= 1.0

    def test_rank_methods_see_a_monotone_relationship_that_pearson_does_not(self):
        """A defining difference between the linear and the rank coefficients."""
        squares = [value**2 for value in [1.0, 2.0, 3.0, 4.0, 5.0]]
        linear = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert correlation(linear, squares, method="spearman") == pytest.approx(1.0)
        assert correlation(linear, squares, method="pearson") < 1.0

    def test_it_is_symmetric(self):
        for method in ("pearson", "spearman", "kendall"):
            assert correlation(X, Y, method=method) == pytest.approx(
                correlation(Y, X, method=method)
            )

    def test_an_unknown_method_is_rejected(self):
        with pytest.raises(ValidationError, match="method"):
            correlation(X, Y, method="cosine")


class TestUndefinedCorrelation:
    """A correlation that cannot be computed is reported, never replaced by zero."""

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_a_constant_first_sample_raises(self, method):
        with pytest.raises(DomainError, match="constant"):
            correlation([2.0, 2.0, 2.0], [1.0, 2.0, 3.0], method=method)

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_a_constant_second_sample_raises(self, method):
        with pytest.raises(DomainError, match="constant"):
            correlation([1.0, 2.0, 3.0], [5.0, 5.0, 5.0], method=method)

    def test_both_constant_raises(self):
        with pytest.raises(DomainError, match="constant"):
            correlation([1.0, 1.0], [2.0, 2.0])

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_a_single_pair_raises(self, method):
        with pytest.raises(DomainError, match="at least 2"):
            correlation([1.0], [2.0], method=method)

    def test_no_scipy_warning_escapes_to_the_caller(self):
        """SciPy signals these cases with a warning that is easy to overlook."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(DomainError):
                correlation([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])


class TestPairedInputHandling:
    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ShapeError, match="same length"):
            covariance([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_missing_values_raise_by_default(self):
        with pytest.raises(MissingValueError, match="incomplete pair"):
            correlation([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])

    def test_omission_drops_whole_pairs_not_single_values(self):
        """The core correctness property of paired cleaning.

        Dropping independently would leave ``x = [1, 3]`` beside ``y = [10, 20]``
        and correlate observations that never belonged together.
        """
        x = [1.0, np.nan, 3.0, 4.0]
        y = [10.0, 20.0, np.nan, 40.0]
        assert correlation(x, y, nan_policy="omit") == pytest.approx(
            stats.pearsonr([1.0, 4.0], [10.0, 40.0]).statistic
        )

    def test_omission_matches_the_pandas_pairwise_convention(self):
        x = pd.Series([1.0, np.nan, 3.0, 4.0, 7.0])
        y = pd.Series([10.0, 20.0, np.nan, 40.0, 50.0])
        assert correlation(x, y, nan_policy="omit") == pytest.approx(x.corr(y))
        assert covariance(x, y, nan_policy="omit") == pytest.approx(x.cov(y))

    def test_no_complete_pairs_is_an_error(self):
        with pytest.raises(EmptyDataError, match="No complete pairs"):
            covariance([1.0, np.nan], [np.nan, 2.0], nan_policy="omit")

    def test_infinities_raise_by_default(self):
        with pytest.raises(NonFiniteValueError):
            covariance([1.0, np.inf], [1.0, 2.0])

    def test_empty_inputs_are_rejected(self):
        with pytest.raises(EmptyDataError):
            covariance([], [])

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (X, ["a"] * len(X)),
            (np.zeros((2, 2)), np.zeros((2, 2))),
            (pd.DataFrame({"a": [1, 2]}), [1, 2]),
        ],
    )
    def test_unusable_inputs_are_rejected(self, x, y):
        with pytest.raises(ValidationError):
            covariance(x, y)

    @pytest.mark.parametrize(
        "data",
        [
            (X, Y),
            (np.array(X), np.array(Y)),
            (pd.Series(X), pd.Series(Y)),
            (tuple(X), tuple(Y)),
        ],
    )
    def test_supported_containers_agree(self, data):
        assert correlation(*data) == pytest.approx(stats.pearsonr(X, Y).statistic)

    def test_inputs_are_not_modified(self):
        x = np.array(X)
        y = np.array(Y)
        before_x, before_y = x.copy(), y.copy()
        correlation(x, y)
        np.testing.assert_array_equal(x, before_x)
        np.testing.assert_array_equal(y, before_y)
