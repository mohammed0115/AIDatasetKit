"""Finite values near the float64 ceiling, measured without overflow.

Every value here is finite, and so is every true answer except two: the range
and the interquartile range of ``[1e308, -1e308]`` are ``2e308``, which float64
cannot hold. Before this was fixed, the mean of such a column came back as
``nan``, its median as ``-inf`` and its IQR as ``inf``; the profile carried them,
and the audit either failed to serialise or wrote a number that was not the
answer. Each test runs with numpy's RuntimeWarning raised as an error, so an
overflow that is merely hidden fails the test rather than passing it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core import KitConfig
from aidatasetkit.core.exceptions import DomainError
from aidatasetkit.evidence import AuditBuilder, canonical_json
from aidatasetkit.profiling import DataProfiler, DataQualityInspector
from aidatasetkit.statistics import StatisticsEngine

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

SYMMETRIC = [1e308, -1e308] * 20
SKEWED = [1e308, -1e308, 1e308, 0.0] * 10


class TestTheEngine:
    def test_the_mean_of_a_symmetric_extreme_sample_is_zero(self):
        assert StatisticsEngine(SYMMETRIC).mean() == 0.0

    def test_the_mean_of_a_skewed_extreme_sample(self):
        # 20 * 1e308 - 10 * 1e308 over 40 observations.
        assert StatisticsEngine(SKEWED).mean() == pytest.approx(2.5e307, rel=1e-12)

    def test_the_median_is_not_minus_infinity(self):
        assert StatisticsEngine(SYMMETRIC).median() == 0.0

    def test_the_quartiles_are_finite(self):
        q1, q2, q3 = StatisticsEngine(SYMMETRIC).quartiles()
        assert (q1, q2, q3) == (-1e308, 0.0, 1e308)

    def test_a_percentile_is_finite(self):
        assert StatisticsEngine(SYMMETRIC).percentile(50) == 0.0

    def test_the_standard_deviation_is_the_true_one(self):
        expected = 1e308 * np.sqrt(40 / 39)
        assert StatisticsEngine(SYMMETRIC).std() == pytest.approx(expected, rel=1e-12)

    def test_a_range_beyond_float64_is_refused_not_infinite(self):
        with pytest.raises(DomainError, match="exceeds the largest float64"):
            StatisticsEngine(SYMMETRIC).range()

    def test_an_iqr_beyond_float64_is_refused_not_infinite(self):
        with pytest.raises(DomainError, match="exceeds the largest float64"):
            StatisticsEngine(SYMMETRIC).iqr()

    def test_the_summary_records_them_as_undefined(self):
        summary = StatisticsEngine(SYMMETRIC).describe()
        assert summary.range is None
        assert summary.iqr is None
        json.dumps(summary.to_dict(), allow_nan=False)

    def test_skewness_of_an_extreme_sample_does_not_warn(self):
        StatisticsEngine(SKEWED).describe()


class TestOrdinaryDataIsUntouched:
    """The fallback only runs when the direct form overflows."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_every_statistic_is_numpy_s_to_the_last_bit(self, seed):
        values = np.random.default_rng(seed).normal(50, 13, 999)
        engine = StatisticsEngine(values)
        assert engine.mean() == float(np.mean(values))
        assert engine.median() == float(np.median(values))
        assert engine.std() == float(np.std(values, ddof=1))
        assert engine.range() == float(np.ptp(values))
        assert engine.quartiles() == tuple(
            float(v) for v in np.percentile(values, [25.0, 50.0, 75.0])
        )


class TestTheAudit:
    @pytest.mark.parametrize("values", [SYMMETRIC, SKEWED], ids=["symmetric", "skewed"])
    def test_an_audit_of_extreme_floats_is_strict_json(self, values):
        frame = pd.DataFrame({"x": values, "label": [0, 1] * (len(values) // 2)})
        config = KitConfig()
        profile = DataProfiler(config).profile(frame)
        quality = DataQualityInspector(config).inspect(frame, profile=profile, target="label")
        artifact = AuditBuilder().build(frame, profile=profile, quality=quality, kit_config=config)
        json.loads(canonical_json(artifact.to_dict()))

    def test_the_profile_records_the_true_centre(self):
        frame = pd.DataFrame({"x": SYMMETRIC})
        numeric = DataProfiler().profile(frame).column_profiles[0].numeric
        assert numeric.mean == 0.0
        assert numeric.median == 0.0
        assert numeric.iqr is None
