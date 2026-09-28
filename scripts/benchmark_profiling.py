"""Reproducible profiling benchmark (G0.1).

    python scripts/benchmark_profiling.py [repeats]

Deterministic data (seed 20260927), one discarded warm-up per case, then
``repeats`` timed runs (default 7) reporting the median, and a separate
tracemalloc run for peak Python memory. Each case also prints a digest of its
result, so two checkouts can be compared for identical output as well as
speed: run it with PYTHONPATH pointing at each checkout, one after the other,
on an otherwise idle machine.

It is a measurement tool, not a test. The regression guards that run in CI are
in tests/unit/test_counting_performance.py and do not depend on runner speed.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd

from aidatasetkit.core.counting import value_counts
from aidatasetkit.profiling import DataProfiler

SEED = 20260927
ROWS = 100_000


def _mixed(rows: int, cols: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    half = cols // 2
    data = {f"n{i}": rng.normal(0, 1, rows) for i in range(half)}
    data |= {f"c{i}": rng.choice(list("abcdefgh"), rows) for i in range(cols - half)}
    return pd.DataFrame(data)


def cases() -> dict:
    rng = np.random.default_rng(SEED)
    pairs = np.repeat(np.arange(ROWS // 2), 2)
    rng.shuffle(pairs)
    special = rng.normal(0, 1, ROWS)
    special[rng.choice(ROWS, 5000, replace=False)] = np.nan
    special[rng.choice(ROWS, 500, replace=False)] = np.inf
    special[rng.choice(ROWS, 500, replace=False)] = -np.inf
    nullable = pd.array(rng.integers(0, 1000, ROWS), dtype="Int64")
    nullable[rng.choice(ROWS, 5000, replace=False)] = pd.NA
    strings = [f"id-{i:07d}" for i in rng.permutation(ROWS)]
    return {
        "profile_10k_x20": ("profile", _mixed(10_000)),
        "profile_100k_x20": ("profile", _mixed(ROWS)),
        "profile_100k_unique_numeric": ("profile", pd.DataFrame({"u": rng.normal(0, 1, ROWS)})),
        "profile_100k_low_cardinality": ("profile", pd.DataFrame({"c": rng.choice(list("abcdefgh"), ROWS)})),
        "profile_100k_all_tied_pairs": ("profile", pd.DataFrame({"t": pairs.astype("int64")})),
        "profile_10k_huge_integers": ("profile", pd.DataFrame({"h": pd.Series([10**400 + i % 500 for i in range(10_000)], dtype=object)})),
        "profile_100k_near_1e308": ("profile", pd.DataFrame({"f": 1e308 * rng.uniform(-1.0, 1.0, ROWS)})),
        "profile_100k_nan_inf": ("profile", pd.DataFrame({"s": special})),
        "profile_100k_unique_strings": ("profile", pd.DataFrame({"s": strings})),
        "profile_100k_nullable_int64": ("profile", pd.DataFrame({"i": nullable})),
        "value_counts_100k_unique_numeric": ("value_counts", pd.Series(rng.normal(0, 1, ROWS))),
        "value_counts_100k_unique_strings": ("value_counts", pd.Series(strings)),
    }


def _run(kind: str, data):
    return DataProfiler().profile(data) if kind == "profile" else value_counts(data)


def _digest(kind: str, result) -> str:
    if kind == "profile":
        payload = json.dumps(result.to_dict(), sort_keys=True, default=repr)
    else:
        payload = json.dumps([[repr(label), int(count)] for label, count in result.items()])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main() -> None:
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    print(f"pandas {pd.__version__}, numpy {np.__version__}, python {sys.version.split()[0]}, repeats {repeats}")
    for name, (kind, data) in cases().items():
        _run(kind, data)  # warm-up, discarded
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = _run(kind, data)
            times.append(time.perf_counter() - start)
        tracemalloc.start()
        _run(kind, data)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        print(
            f"{name:36s} median={statistics.median(times):8.4f}s  "
            f"peak={peak / 1e6:7.2f}MB  digest={_digest(kind, result)}"
        )


if __name__ == "__main__":
    main()
