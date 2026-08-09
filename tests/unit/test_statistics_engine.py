"""Tests for :class:`StatisticsEngine`.

Every numeric result is checked against an independent reference implementation
from NumPy, SciPy, or pandas rather than against a hand-copied constant, so a
formula transcribed incorrectly from the original Java library cannot pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from aidatasetkit.core.exceptions import (
    DomainError,
    EmptyDataError,
    MissingValueError,
    NonFiniteValueError,
    NonNumericDataError,
    ValidationError,
)
from aidatasetkit.statistics import DescriptiveSummary, Quartiles, StatisticsEngine

SAMPLE = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
ASYMMETRIC = [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0]


@pytest.fixture
def engine() -> StatisticsEngine:
    return StatisticsEngine(SAMPLE)


@pytest.fixture
def asymmetric() -> StatisticsEngine:
    return StatisticsEngine(ASYMMETRIC)


class TestInputHandling:
    @pytest.mark.parametrize(
        "data",
        [
            SAMPLE,
            tuple(SAMPLE),
            np.array(SAMPLE),
            np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype="int64"),
            pd.Series(SAMPLE),
            pd.Series(SAMPLE, dtype="Float64"),
            pd.Index(SAMPLE),
        ],
    )
    def test_every_supported_container_gives_the_same_answer(self, data):
        assert StatisticsEngine(data).mean() == pytest.approx(np.mean(SAMPLE))

    def test_integer_input_does_not_truncate_results(self):
        engine = StatisticsEngine([1, 2])
        assert engine.mean() == pytest.approx(1.5)

    def test_the_input_is_never_modified(self):
        source = np.array(SAMPLE)
        before = source.copy()
        StatisticsEngine(source).z_scores()
        np.testing.assert_array_equal(source, before)

    def test_the_exposed_sample_is_read_only(self, engine):
        with pytest.raises(ValueError):
            engine.values[0] = 99.0

    def test_count_reports_the_observations_used(self, engine):
        assert engine.count == len(SAMPLE)
        assert engine.n_omitted == 0

    def test_repr_is_informative(self, engine):
        assert "count=8" in repr(engine)


class TestMissingAndNonFinite:
    def test_missing_values_raise_by_default(self):
        with pytest.raises(MissingValueError):
            StatisticsEngine([1.0, np.nan, 3.0])

    def test_omission_is_explicit_and_reported(self):
        engine = StatisticsEngine([1.0, np.nan, 3.0], nan_policy="omit")
        assert engine.count == 2
        assert engine.n_omitted == 1
        assert engine.mean() == pytest.approx(2.0)

    def test_infinities_raise_by_default(self):
        with pytest.raises(NonFiniteValueError):
            StatisticsEngine([1.0, np.inf])

    def test_empty_input_raises(self):
        with pytest.raises(EmptyDataError):
            StatisticsEngine([])

    def test_text_input_raises(self):
        with pytest.raises(NonNumericDataError):
            StatisticsEngine(["a", "b"])


class TestLocation:
    def test_mean_matches_numpy(self, engine):
        assert engine.mean() == pytest.approx(np.mean(SAMPLE))

    def test_median_matches_numpy(self, engine):
        assert engine.median() == pytest.approx(np.median(SAMPLE))

    def test_median_interpolates_for_an_even_sample(self):
        assert StatisticsEngine([1.0, 2.0, 3.0, 4.0]).median() == pytest.approx(2.5)

    def test_mode_returns_the_single_most_frequent_value(self, engine):
        assert engine.mode() == (4.0,)

    def test_mode_reports_every_mode_of_a_multimodal_sample(self):
        assert StatisticsEngine([1.0, 1.0, 2.0, 2.0, 3.0]).mode() == (1.0, 2.0)

    def test_a_sample_of_distinct_values_has_no_mode(self):
        assert StatisticsEngine([1.0, 2.0, 3.0]).mode() == ()

    def test_mode_of_a_constant_sample_is_that_value(self):
        assert StatisticsEngine([7.0, 7.0, 7.0]).mode() == (7.0,)


class TestExtent:
    def test_min_max_and_range_match_numpy(self, engine):
        assert engine.min() == pytest.approx(np.min(SAMPLE))
        assert engine.max() == pytest.approx(np.max(SAMPLE))
        assert engine.range() == pytest.approx(np.ptp(SAMPLE))

    def test_sum_matches_numpy(self, engine):
        assert engine.sum() == pytest.approx(np.sum(SAMPLE))

    def test_sum_of_squares_is_the_uncorrected_form(self, engine):
        """Sigma x^2, not Sigma (x - xbar)^2."""
        assert engine.sum_of_squares() == pytest.approx(np.sum(np.square(SAMPLE)))
        corrected = np.sum(np.square(np.array(SAMPLE) - np.mean(SAMPLE)))
        assert engine.sum_of_squares() != pytest.approx(corrected)


class TestDispersion:
    def test_population_variance_matches_numpy_ddof_zero(self, engine):
        assert engine.variance(ddof=0) == pytest.approx(np.var(SAMPLE, ddof=0))

    def test_sample_variance_matches_numpy_and_pandas_ddof_one(self, engine):
        assert engine.variance(ddof=1) == pytest.approx(np.var(SAMPLE, ddof=1))
        assert engine.variance(ddof=1) == pytest.approx(pd.Series(SAMPLE).var())

    def test_the_default_is_the_sample_variance(self, engine):
        assert engine.variance() == engine.variance(ddof=1)

    def test_the_two_conventions_differ(self, engine):
        assert engine.variance(ddof=0) != pytest.approx(engine.variance(ddof=1))

    def test_population_and_sample_standard_deviations_match_references(self, engine):
        assert engine.std(ddof=0) == pytest.approx(np.std(SAMPLE, ddof=0))
        assert engine.std(ddof=1) == pytest.approx(np.std(SAMPLE, ddof=1))
        assert engine.std(ddof=1) == pytest.approx(pd.Series(SAMPLE).std())

    def test_the_default_is_the_sample_standard_deviation(self, engine):
        assert engine.std() == engine.std(ddof=1)

    def test_standard_deviation_is_the_root_of_the_variance(self, engine):
        for ddof in (0, 1):
            assert engine.std(ddof=ddof) == pytest.approx(
                np.sqrt(engine.variance(ddof=ddof))
            )

    def test_sample_variance_of_one_observation_raises_instead_of_nan(self):
        """NumPy answers nan with a RuntimeWarning here."""
        with pytest.raises(DomainError, match="at least 2"):
            StatisticsEngine([5.0]).variance(ddof=1)

    def test_population_variance_of_one_observation_is_zero(self):
        assert StatisticsEngine([5.0]).variance(ddof=0) == pytest.approx(0.0)

    def test_variance_of_a_constant_sample_is_zero(self):
        assert StatisticsEngine([3.0, 3.0, 3.0]).variance() == pytest.approx(0.0)

    @pytest.mark.parametrize("ddof", [-1, 1.5, "1", True, None])
    def test_invalid_ddof_is_rejected(self, engine, ddof):
        with pytest.raises(ValidationError, match="ddof"):
            engine.variance(ddof=ddof)


class TestMeanAbsoluteDeviation:
    def test_deviation_about_the_mean_matches_a_manual_computation(self, engine):
        expected = np.mean(np.abs(np.array(SAMPLE) - np.mean(SAMPLE)))
        assert engine.mean_absolute_deviation() == pytest.approx(expected)

    def test_deviation_about_the_median_matches_a_manual_computation(self, engine):
        expected = np.mean(np.abs(np.array(SAMPLE) - np.median(SAMPLE)))
        assert engine.mean_absolute_deviation(center="median") == pytest.approx(expected)

    def test_the_default_centre_is_the_mean(self, engine):
        assert engine.mean_absolute_deviation() == engine.mean_absolute_deviation("mean")

    def test_it_is_not_the_robust_median_absolute_deviation(self, asymmetric):
        """The two statistics share an acronym and are not interchangeable."""
        values = np.array(ASYMMETRIC)
        robust = np.median(np.abs(values - np.median(values)))
        assert asymmetric.mean_absolute_deviation(center="median") != pytest.approx(robust)

    def test_deviation_about_the_median_is_never_larger_than_about_the_mean(self):
        """A defining property of the median: it minimises absolute deviation."""
        for sample in (SAMPLE, ASYMMETRIC, [1.0, 100.0, 100.0]):
            engine = StatisticsEngine(sample)
            about_median = engine.mean_absolute_deviation("median")
            about_mean = engine.mean_absolute_deviation("mean")
            assert about_median <= about_mean + 1e-12

    @pytest.mark.parametrize("center", ["average", "mode", "", None])
    def test_an_unknown_centre_is_rejected(self, engine, center):
        with pytest.raises(ValidationError, match="center"):
            engine.mean_absolute_deviation(center=center)


class TestAlternativeMeans:
    def test_geometric_mean_matches_scipy(self, engine):
        assert engine.geometric_mean() == pytest.approx(stats.gmean(SAMPLE))

    def test_geometric_mean_matches_its_definition(self, engine):
        expected = np.exp(np.mean(np.log(SAMPLE)))
        assert engine.geometric_mean() == pytest.approx(expected)

    def test_harmonic_mean_matches_scipy(self, engine):
        assert engine.harmonic_mean() == pytest.approx(stats.hmean(SAMPLE))

    def test_harmonic_mean_matches_its_definition(self, engine):
        expected = len(SAMPLE) / np.sum(1.0 / np.array(SAMPLE))
        assert engine.harmonic_mean() == pytest.approx(expected)

    def test_rms_matches_its_definition(self, engine):
        assert engine.rms() == pytest.approx(np.sqrt(np.mean(np.square(SAMPLE))))

    def test_geometric_mean_is_not_the_quadratic_mean(self, engine):
        """These are distinct formulas that are easy to conflate."""
        assert engine.geometric_mean() == pytest.approx(4.6032156, rel=1e-6)
        assert engine.rms() == pytest.approx(5.3851648, rel=1e-6)
        assert engine.geometric_mean() != pytest.approx(engine.rms())

    def test_the_means_obey_the_classical_inequality(self, engine):
        """HM <= GM <= AM <= RMS for positive values, with equality only if constant."""
        assert (
            engine.harmonic_mean()
            < engine.geometric_mean()
            < engine.mean()
            < engine.rms()
        )

    def test_all_means_coincide_for_a_constant_positive_sample(self):
        engine = StatisticsEngine([4.0, 4.0, 4.0])
        assert engine.harmonic_mean() == pytest.approx(4.0)
        assert engine.geometric_mean() == pytest.approx(4.0)
        assert engine.rms() == pytest.approx(4.0)

    @pytest.mark.parametrize("sample", [[1.0, 2.0, 0.0], [1.0, -2.0, 3.0], [-1.0]])
    def test_geometric_mean_rejects_non_positive_values(self, sample):
        """SciPy returns 0.0 without warning for a sample containing a zero."""
        with pytest.raises(DomainError, match="strictly positive"):
            StatisticsEngine(sample).geometric_mean()

    @pytest.mark.parametrize("sample", [[1.0, 2.0, 0.0], [1.0, -2.0, 3.0], [-1.0]])
    def test_harmonic_mean_rejects_non_positive_values(self, sample):
        with pytest.raises(DomainError, match="strictly positive"):
            StatisticsEngine(sample).harmonic_mean()

    def test_rms_accepts_negative_values(self):
        assert StatisticsEngine([-3.0, 4.0]).rms() == pytest.approx(np.sqrt(12.5))


class TestPosition:
    def test_quartiles_match_numpy(self, engine):
        expected = np.percentile(SAMPLE, [25, 50, 75])
        assert engine.quartiles() == pytest.approx(tuple(expected))

    def test_quartiles_match_pandas(self, engine):
        expected = pd.Series(SAMPLE).quantile([0.25, 0.5, 0.75]).to_numpy()
        np.testing.assert_allclose(np.array(engine.quartiles()), expected)

    def test_quartiles_are_named(self, engine):
        quartiles = engine.quartiles()
        assert isinstance(quartiles, Quartiles)
        assert quartiles.q2 == pytest.approx(engine.median())

    def test_percentile_matches_numpy_for_a_scalar(self, engine):
        for q in (0, 10, 50, 90, 100):
            assert engine.percentile(q) == pytest.approx(np.percentile(SAMPLE, q))

    def test_percentile_accepts_a_sequence(self, engine):
        np.testing.assert_allclose(
            engine.percentile([10, 90]), np.percentile(SAMPLE, [10, 90])
        )

    def test_iqr_matches_scipy(self, engine):
        assert engine.iqr() == pytest.approx(stats.iqr(SAMPLE))

    def test_iqr_is_the_distance_between_the_outer_quartiles(self, engine):
        quartiles = engine.quartiles()
        assert engine.iqr() == pytest.approx(quartiles.q3 - quartiles.q1)

    @pytest.mark.parametrize("q", [-1, 101, np.nan, np.inf])
    def test_percentile_rejects_values_outside_the_range(self, engine, q):
        with pytest.raises(ValidationError, match="q must"):
            engine.percentile(q)


class TestZScores:
    def test_z_scores_match_scipy(self, engine):
        np.testing.assert_allclose(engine.z_scores(), stats.zscore(SAMPLE))

    def test_default_uses_the_population_standard_deviation(self, engine):
        np.testing.assert_allclose(engine.z_scores(), engine.z_scores(ddof=0))

    def test_sample_standard_deviation_is_available(self, engine):
        np.testing.assert_allclose(
            engine.z_scores(ddof=1), stats.zscore(SAMPLE, ddof=1)
        )

    def test_standardised_values_have_zero_mean_and_unit_variance(self, engine):
        scores = engine.z_scores()
        assert np.mean(scores) == pytest.approx(0.0, abs=1e-12)
        assert np.std(scores, ddof=0) == pytest.approx(1.0)

    def test_a_constant_sample_raises_instead_of_dividing_by_zero(self):
        with pytest.raises(DomainError, match="constant"):
            StatisticsEngine([2.0, 2.0, 2.0]).z_scores()


class TestShape:
    def test_skewness_matches_pandas_by_default(self, asymmetric):
        assert asymmetric.skewness() == pytest.approx(pd.Series(ASYMMETRIC).skew())

    def test_skewness_matches_scipy_with_the_correction(self, asymmetric):
        assert asymmetric.skewness() == pytest.approx(
            stats.skew(ASYMMETRIC, bias=False)
        )

    def test_uncorrected_skewness_matches_the_scipy_default(self, asymmetric):
        assert asymmetric.skewness(bias=True) == pytest.approx(stats.skew(ASYMMETRIC))

    def test_the_two_skewness_conventions_differ(self, asymmetric):
        assert asymmetric.skewness() != pytest.approx(asymmetric.skewness(bias=True))

    def test_a_symmetric_sample_has_zero_skewness(self):
        assert StatisticsEngine([1.0, 2.0, 3.0, 4.0, 5.0]).skewness() == pytest.approx(
            0.0, abs=1e-12
        )

    def test_a_right_tailed_sample_is_positively_skewed(self):
        engine = StatisticsEngine([1.0] * 20 + [1000.0])
        assert engine.skewness() > 3.0

    def test_kurtosis_matches_pandas_by_default(self, asymmetric):
        assert asymmetric.kurtosis() == pytest.approx(pd.Series(ASYMMETRIC).kurt())

    def test_kurtosis_matches_scipy_with_the_correction(self, asymmetric):
        assert asymmetric.kurtosis() == pytest.approx(
            stats.kurtosis(ASYMMETRIC, fisher=True, bias=False)
        )

    def test_uncorrected_kurtosis_matches_the_scipy_default(self, asymmetric):
        assert asymmetric.kurtosis(bias=True) == pytest.approx(
            stats.kurtosis(ASYMMETRIC)
        )

    def test_pearson_kurtosis_exceeds_fisher_by_three(self, asymmetric):
        assert asymmetric.kurtosis(fisher=False) == pytest.approx(
            asymmetric.kurtosis(fisher=True) + 3.0
        )

    @pytest.mark.parametrize("sample", [[1.0], [1.0, 2.0]])
    def test_corrected_skewness_needs_three_observations(self, sample):
        with pytest.raises(DomainError, match="at least 3"):
            StatisticsEngine(sample).skewness()

    @pytest.mark.parametrize("sample", [[1.0, 2.0], [1.0, 2.0, 3.0]])
    def test_corrected_kurtosis_needs_four_observations(self, sample):
        """SciPy returns -1.5 for three observations without warning."""
        with pytest.raises(DomainError, match="at least 4"):
            StatisticsEngine(sample).kurtosis()

    def test_shape_of_a_constant_sample_raises_instead_of_nan(self):
        """SciPy returns nan with a RuntimeWarning for both of these."""
        engine = StatisticsEngine([5.0, 5.0, 5.0, 5.0, 5.0])
        with pytest.raises(DomainError, match="constant"):
            engine.skewness()
        with pytest.raises(DomainError, match="constant"):
            engine.kurtosis()


class TestMomentMethods:
    def test_first_raw_moment_is_the_mean(self, engine):
        assert engine.raw_moment(1) == pytest.approx(engine.mean())

    def test_first_central_moment_is_zero(self, engine):
        assert engine.central_moment(1) == pytest.approx(0.0, abs=1e-12)

    def test_second_central_moment_is_the_population_variance(self, engine):
        assert engine.central_moment(2) == pytest.approx(engine.variance(ddof=0))

    def test_moment_about_an_origin_matches_its_definition(self, engine):
        expected = np.mean((np.array(SAMPLE) - 3.0) ** 3)
        assert engine.moment_about(3.0, 3) == pytest.approx(expected)

    def test_moment_about_zero_is_the_raw_moment(self, engine):
        assert engine.moment_about(0.0, 3) == pytest.approx(engine.raw_moment(3))

    def test_moment_about_the_mean_is_the_central_moment(self, engine):
        assert engine.moment_about(engine.mean(), 4) == pytest.approx(
            engine.central_moment(4)
        )


class TestDescribe:
    def test_it_returns_a_typed_summary(self, engine):
        summary = engine.describe()
        assert isinstance(summary, DescriptiveSummary)
        assert summary.count == 8
        assert summary.mean == pytest.approx(np.mean(SAMPLE))
        assert summary.std == pytest.approx(pd.Series(SAMPLE).std())
        assert summary.q25 == pytest.approx(np.percentile(SAMPLE, 25))
        assert summary.median == pytest.approx(np.median(SAMPLE))
        assert summary.q75 == pytest.approx(np.percentile(SAMPLE, 75))
        assert summary.iqr == pytest.approx(stats.iqr(SAMPLE))
        assert summary.skewness == pytest.approx(pd.Series(SAMPLE).skew())
        assert summary.kurtosis == pytest.approx(pd.Series(SAMPLE).kurt())

    def test_every_field_agrees_with_the_matching_method(self, engine):
        summary = engine.describe()
        assert summary.minimum == engine.min()
        assert summary.maximum == engine.max()
        assert summary.range == engine.range()
        assert summary.variance == engine.variance()

    def test_it_is_a_value_not_a_side_effect(self, engine, capsys):
        engine.describe()
        assert capsys.readouterr().out == ""

    def test_it_is_json_serialisable(self, engine):
        import json

        assert "skewness" in json.dumps(engine.describe().to_dict())

    def test_undefined_statistics_are_reported_as_absent(self):
        summary = StatisticsEngine([5.0]).describe()
        assert summary.count == 1
        assert summary.mean == pytest.approx(5.0)
        assert summary.std is None
        assert summary.variance is None
        assert summary.skewness is None
        assert summary.kurtosis is None

    def test_a_constant_sample_reports_dispersion_but_not_shape(self):
        summary = StatisticsEngine([3.0, 3.0, 3.0, 3.0]).describe()
        assert summary.variance == pytest.approx(0.0)
        assert summary.range == pytest.approx(0.0)
        assert summary.skewness is None
        assert summary.kurtosis is None

    def test_omitted_observations_are_reported(self):
        summary = StatisticsEngine([1.0, np.nan, 3.0], nan_policy="omit").describe()
        assert summary.count == 2
        assert summary.n_omitted == 1

    def test_the_summary_is_immutable(self, engine):
        with pytest.raises(AttributeError):
            engine.describe().mean = 0.0


class TestSingleObservation:
    """A one-element sample: everything that is defined, and nothing that is not."""

    @pytest.fixture
    def single(self) -> StatisticsEngine:
        return StatisticsEngine([4.0])

    def test_location_and_extent_are_defined(self, single):
        assert single.mean() == pytest.approx(4.0)
        assert single.median() == pytest.approx(4.0)
        assert single.min() == single.max() == pytest.approx(4.0)
        assert single.range() == pytest.approx(0.0)
        assert single.rms() == pytest.approx(4.0)
        assert single.geometric_mean() == pytest.approx(4.0)

    def test_sample_dispersion_is_undefined(self, single):
        with pytest.raises(DomainError):
            single.variance(ddof=1)
        with pytest.raises(DomainError):
            single.std(ddof=1)

    def test_shape_is_undefined(self, single):
        with pytest.raises(DomainError):
            single.skewness()
        with pytest.raises(DomainError):
            single.kurtosis()
