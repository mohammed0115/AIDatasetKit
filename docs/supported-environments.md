# Supported environments

Two environments are supported, and both are pinned exactly in `constraints/`:

| | Minimum | Reference |
|---|---|---|
| Python | 3.11 | 3.12 |
| NumPy | 1.26.4 | 2.5.2 |
| pandas | 2.1.4 | 3.0.5 |
| SciPy | 1.11.4 | 1.18.0 |
| scikit-learn | 1.6.1 | 1.9.0 |
| pytest (`dev`) | 8.0.0 | 8.3.3 |
| matplotlib (`viz`) | 3.9.0 | 3.11.2 |
| xgboost / lightgbm / catboost (`boosting`) | 2.0.0 / 4.0.0 / 1.2 | 3.4.1 / 4.7.0 / 1.2.10 |
| pyarrow (`parquet`) | 14.0.1 | 25.0.1 |
| openpyxl (`excel`) | 3.1.0 | 3.1.5 |

```bash
pip install -c constraints/minimum.txt   -e ".[dev]"      # or reference.txt
pip install -c constraints/reference.txt -e ".[dev,viz,boosting]"
```

The floors in `pyproject.toml` are the minimum column exactly. They record what
the full suite has passed on, not what might work, and
`tests/unit/test_dependency_contract.py` fails if the two drift apart. There are
no upper bounds: nothing newer than the reference column is known to break, and
nothing newer has been run either.

## Why each floor is where it is

**Python 3.11.** The declared `requires-python`. Run on CPython 3.11.16.

**scikit-learn 1.6.1.** The first version on which every capability the catalog
declares is true. On 1.5.2, `ExtraTreesClassifier` and `ExtraTreesRegressor`
refuse NaN (`handles_missing_values=True` would be false), and 43 tests fail in the
six files that were run. The previously declared `>=1.4` was never tested and is
not supported.

**SciPy 1.11.4.** The declared floor, kept. It needed one fix: the library passed
the moment order to `scipy.stats.moment` as `order=`, a keyword that exists only
from SciPy 1.12, so every moment raised on 1.11. The order is now positional,
which every version accepts.

**pandas 2.1.4 and NumPy 1.26.4.** The declared floors, run at their latest patch
releases. pandas 2 is kept, not dropped: two defects that only pandas 2 exposes
were fixed rather than avoided by requiring pandas 3 (a 400-digit integer broke
`value_counts`; see `aidatasetkit/core/counting.py`).

**Extras.** Run at exactly the declared floors on the minimum set, and at the
reference versions on the reference set.

**pyarrow 14.0.1.** The first release with the fix for CVE-2023-47248,
arbitrary code execution while deserializing a malicious Parquet or Arrow IPC
file. Older pyarrow is never acceptable for a library whose premise is reading
untrusted tables, so the floor is the fix, not the oldest version that imports.

**openpyxl 3.1.0.** The floor for the `.xlsx` reader: the 3.1 line reads the
worksheet's declared dimensions in `read_only` mode without materializing the
cells, which is what the row/column/cell preflight relies on.

## What differs between the two, by design

**Recorded dtype names.** pandas 3 reports a text column as `str`; pandas 2 as
`object`. The audit records what pandas reported, so the same file produces two
different, equally correct artifacts, and the dataset and schema digests derived
from them differ too. The suite keeps one exact golden fixture per pandas major
and asserts they differ in those fields and nothing else
(`tests/unit/test_golden_fixtures.py`).

**What scikit-learn does to unprepared input.** scikit-learn 1.9.0 truncates the
rank of an unscaled least-squares fit at a column-magnitude spread near `1e6`, and
its histogram gradient boosting refuses an all-missing column; 1.6.1 through 1.8.0
do neither. The library's protections -- scaling where `requires_scaling` says so,
excluding an all-missing column -- hold on every version, and the tests assert
both behaviours of the dependency rather than assuming one.

## Platforms

| Platform | Status |
|---|---|
| Windows | Run: minimum, reference, and both with extras -- locally on Windows 11 and on GitHub's `windows-latest` (CI run 36328760194, commit `79e7cc8`), plus build/install/smoke. |
| Linux | Run: minimum, reference, and both with extras on GitHub's `ubuntu-latest` (Ubuntu 24.04; CI run 36328760194, commit `79e7cc8`), plus build/install/smoke. |
| macOS | Not run, not claimed. |

A workflow file is not evidence; these rows name the run that is. The first Linux
run found a real defect -- tied counts ordered by a CPU-dispatched unstable sort
on pandas 2.1 -- fixed in `core/counting.py` before the run above.

## Windows notes

The determinism tests start a child interpreter with an almost empty
environment. On Windows that environment must carry `SystemRoot`, or the child
cannot import scikit-learn (`WinError 10106`); `tests/conftest.py::isolated_env`
passes exactly that one variable. Committed fixtures are read as UTF-8
explicitly, because a Windows locale such as cp1252 would otherwise misread them.
