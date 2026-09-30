# G1-W2 mutation closure: full suite

```text
TESTED_SHA  = 4252c521865a27cbc672e997dd0a7e519714f43b
TREE        = git archive of TESTED_SHA; PYTHONPATH set to the export, and
              aidatasetkit imported from the export (checked before the run)
COMMAND     = python -m pytest -rfE --junitxml=junit.xml -p no:cacheprovider
PLATFORM    = win32, Python 3.12.3, pytest 8.3.3, pluggy 1.6.0
LIBRARIES   = NumPy 2.4.1, pandas 2.3.3, SciPy 1.17.0, scikit-learn 1.8.0
START/END   = 2026-09-30T09:45:14Z / 2026-09-30T09:52:51Z
EXIT        = 0
SUMMARY     = 4373 passed, 46 skipped in 451.12s (0:07:31)
COLLECTION  = collected 4418 items / 1 skipped
JUNIT       = tests=4419 failures=0 errors=0 skipped=46 time=451.075
XFAIL/XPASS = 0 / 0; DESELECTED = 0
```

The mutation harness (about 80 s) ran concurrently for part of this window, so
the duration is an upper bound, not a benchmark.

Reconciliation. The collection line counts 4418 test items; the "/ 1 skipped"
is a module skipped at collection time, reported as one result but not an
item. 4418 + 1 = 4419 results = JUnit `tests` = 4373 passed + 46 skipped.

Skips (all conditional and pre-existing, none xfail): 39 model-contract
non-applicable cases, 2 regression-target cases, 3 visualization-regression
cases without matplotlib, 1 Windows-minimum environment case, and the
collection-time module skip `tests/unit/test_visualization_renderer.py:18`
("the viz extra is not installed"); 39 + 2 + 3 + 1 + 1 = 46.

Against the previous full run (`f128ccd`, 4350 passed, 46 skipped): this
change adds 23 test cases (`TestResourceGovernanceGuarantees`), and
4350 + 23 = 4373. The previous run therefore collected 4395 items / 1 skipped
(4396 results). The earlier report's "4,394 collected" was off by one; no raw
log of that run was kept, and the number is superseded by this one.
