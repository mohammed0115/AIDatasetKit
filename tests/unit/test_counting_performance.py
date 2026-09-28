"""The tally stays vectorised: G0.1 guards against the 79e7cc8 regression.

79e7cc8 made tie ordering deterministic with a per-element Python loop. Every
value of an all-distinct column ties, so a 100k-row column cost about fifty
Python function calls per row -- 5,000,675 calls, 2.2 s, for one
``value_counts`` -- and profiling a 100k x 20 frame went from 2.1 s to 23.7 s.
No test measured it.

Two guards, neither of which depends on how fast a machine is:

1. **Algorithmic.** Count the Python-level function calls one ``value_counts``
   makes, at two sizes ten times apart. A vectorised tally makes the same number
   of calls whatever the length; the per-element loop made about four more per
   extra row. The bound is on the *growth*, so it holds on any runner.
2. **Relative.** Time ``value_counts`` against pandas' own ``Series.value_counts``
   on the same data in the same process -- warm-up discarded, median of seven --
   and bound the ratio. Both sides slow down together on a slow runner; only a
   change in the algorithm moves the ratio. An absolute ceiling is kept as a
   secondary, generous sanity bound.
"""

from __future__ import annotations

import statistics
import sys
import time

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core.counting import value_counts
from aidatasetkit.profiling import DataProfiler

SEED = 20260927

#: Extra Python calls allowed when the input grows tenfold. The vectorised path
#: adds none per row; pandas' own internals add a handful of fixed-size calls.
#: The regressed loop added about 72,000 between these two sizes.
MAX_CALL_GROWTH = 500


def _python_calls(function, *args) -> int:
    """How many Python-level frames ``function(*args)`` enters."""
    calls = 0

    def tracer(frame, event, arg):
        nonlocal calls
        if event == "call":
            calls += 1

    previous = sys.getprofile()
    sys.setprofile(tracer)
    try:
        function(*args)
    finally:
        sys.setprofile(previous)
    return calls


def _all_distinct_floats(n: int) -> pd.Series:
    return pd.Series(np.random.default_rng(SEED).normal(0, 1, n))


def _all_distinct_strings(n: int) -> pd.Series:
    return pd.Series([f"id-{i:07d}" for i in np.random.default_rng(SEED).permutation(n)])


def _tied_pairs(n: int) -> pd.Series:
    values = np.repeat(np.arange(n // 2), 2)
    np.random.default_rng(SEED).shuffle(values)
    return pd.Series(values)


def _with_missing(n: int) -> pd.Series:
    values = np.random.default_rng(SEED).normal(0, 1, n)
    values[::17] = np.nan
    return pd.Series(values)


SHAPES = {
    "all_distinct_floats": _all_distinct_floats,
    "all_distinct_strings": _all_distinct_strings,
    "tied_pairs": _tied_pairs,
    "floats_with_missing": _with_missing,
}


class TestTheTallyIsVectorised:
    @pytest.mark.parametrize("shape", sorted(SHAPES))
    @pytest.mark.parametrize("dropna", [True, False])
    def test_python_calls_do_not_grow_with_the_rows(self, shape, dropna):
        small, large = SHAPES[shape](2_000), SHAPES[shape](20_000)
        tally = lambda series: value_counts(series, dropna=dropna)  # noqa: E731
        tally(small)  # warm any lazy imports
        growth = _python_calls(tally, large) - _python_calls(tally, small)
        assert growth < MAX_CALL_GROWTH, (
            f"{shape}: {growth} more Python calls for 18,000 more rows -- the tally "
            "is doing Python work per element again"
        )

    def test_profiling_an_all_distinct_column_does_not_grow_either(self):
        profiler = DataProfiler()
        small = pd.DataFrame({"u": _all_distinct_floats(2_000)})
        large = pd.DataFrame({"u": _all_distinct_floats(20_000)})
        profiler.profile(small)
        growth = _python_calls(profiler.profile, large) - _python_calls(profiler.profile, small)
        assert growth < MAX_CALL_GROWTH, f"profiling added {growth} calls for 18,000 rows"

    def test_the_guard_can_fail(self):
        """A per-element loop of the kind 79e7cc8 had is caught by the same bound."""

        def per_element(series, dropna=True):
            counted = series.value_counts(dropna=dropna)
            return sorted(range(len(counted)), key=lambda i: (-int(counted.iloc[i]), i))

        small, large = _all_distinct_floats(2_000), _all_distinct_floats(20_000)
        growth = _python_calls(per_element, large) - _python_calls(per_element, small)
        assert growth >= MAX_CALL_GROWTH


#: Our tally may cost this many times pandas' own ``value_counts`` on the same
#: data. It does pandas' work plus one factorisation and one stable sort; the
#: regressed loop cost about 60 times pandas' own.
MAX_RATIO_TO_PANDAS = 6.0

#: Secondary, generous absolute ceiling for 100k rows, in seconds. The regressed
#: loop took 2.2 s on a 2019 laptop CPU; the vectorised path takes hundredths.
ABSOLUTE_CEILING_S = 1.0


def _median_seconds(function, *args, repeats: int = 7) -> float:
    function(*args)  # warm-up, discarded
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(*args)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


class TestTheTallyStaysCloseToPandas:
    @pytest.mark.parametrize("shape", ["all_distinct_floats", "all_distinct_strings", "tied_pairs"])
    def test_relative_cost_at_100k_rows(self, shape):
        series = SHAPES[shape](100_000)
        ours = _median_seconds(value_counts, series)
        theirs = _median_seconds(lambda s: s.value_counts(), series)
        ratio = ours / theirs
        assert ratio <= MAX_RATIO_TO_PANDAS, (
            f"{shape}: value_counts {ours:.4f}s vs pandas {theirs:.4f}s, "
            f"ratio {ratio:.1f} > {MAX_RATIO_TO_PANDAS}"
        )
        assert ours <= ABSOLUTE_CEILING_S, f"{shape}: {ours:.3f}s for 100k rows"
