"""G1-W1: what reading through ``load_table`` costs next to a bare ``pd.read_csv``.

Usage: python scripts/benchmark_ingestion.py [rows] [runs]

A seeded CSV (``rows`` x 10 columns: numbers, text, a quoted field with the
delimiter inside it) is written to a temporary directory, once with ``,`` and
once with ``;``. Each reader is warmed up once and then timed ``runs`` times
sequentially; the median is reported. Peak Python allocation is measured in a
separate run with ``tracemalloc`` so it does not distort the timings.

The baseline is ``pd.read_csv(path, sep=...)`` with the delimiter already
known -- the fastest correct reading, which the old CLI did not do (it used the
default ``,`` and misread the ``;`` file).
"""

from __future__ import annotations

import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from aidatasetkit.ingestion import load_table


def make(path: Path, rows: int, sep: str) -> None:
    rng = np.random.default_rng(20260928)
    frame = pd.DataFrame(
        {
            **{f"x{i}": rng.normal(0, 1, rows).round(6) for i in range(6)},
            "n": rng.integers(0, 1_000_000, rows),
            "city": rng.choice(["Khartoum", "Jeddah", "Oslo", "Lima"], rows),
            "note": rng.choice([f"a{sep}b", "plain", "x y z"], rows),
            "label": rng.integers(0, 2, rows),
        }
    )
    frame.to_csv(path, index=False, sep=sep)


def timed(function, runs: int) -> float:
    function()  # warm-up
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def peak(function) -> int:
    tracemalloc.start()
    function()
    _, high = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return high


def main() -> None:
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(f"pandas {pd.__version__}, numpy {np.__version__}, python {sys.version.split()[0]}")
    print(f"rows={rows} columns=10 runs={runs} (median after one warm-up)")
    with tempfile.TemporaryDirectory() as folder:
        for sep in (",", ";"):
            path = Path(folder) / f"bench_{'comma' if sep == ',' else 'semicolon'}.csv"
            make(path, rows, sep)
            baseline = lambda: pd.read_csv(path, sep=sep)  # noqa: E731
            ours = lambda: load_table(path)  # noqa: E731
            t_base, t_ours = timed(baseline, runs), timed(ours, runs)
            m_base, m_ours = peak(baseline), peak(ours)
            print(
                f"sep={sep!r} size={path.stat().st_size / 1e6:.1f} MB | "
                f"read_csv {t_base:.3f} s, load_table {t_ours:.3f} s, ratio {t_ours / t_base:.2f} | "
                f"peak read_csv {m_base / 1e6:.1f} MB, load_table {m_ours / 1e6:.1f} MB, "
                f"ratio {m_ours / m_base:.2f}"
            )


if __name__ == "__main__":
    main()
