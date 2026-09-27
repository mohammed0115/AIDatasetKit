# GitHub Actions — run 36328760194 (passing) and run 36327967662 (the Linux defect)

Branch `g0-certification`. Workflow `.github/workflows/ci.yml`, trigger `push`. Counts are read from each job's log
(final pytest line) and cross-checked against the JUnit artifact the job uploaded.

## Run 36328760194 — commit `79e7cc82374b468ed58fa4896a1dc2bad7448a46` — conclusion **success**

https://github.com/mohammed0115/AIDatasetKit/actions/runs/36328760194

| Job | Job id | OS | Python | Dependencies | Collected | Passed | Failed | Errors | Skipped | Job duration |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ubuntu-latest / build, install and smoke | 108646519376 | ubuntu-24.04 | 3.12.14 | build 1.6.1, twine 7.0.0 (smoke) | — | smoke: wheel: install, CLI, audit, artifacts, exit code -- all good; sdist: install, CLI, audit, artifacts, exit code -- all good; == Release smoke test passed | 0 | 0 | — | 60 s |
| ubuntu-latest / minimum | 108646519617 | ubuntu-24.04 | 3.11.16 | numpy 1.26.4, pandas 2.1.4, scipy 1.11.4, scikit-learn 1.6.1, pytest 8.0.0 | 4113 | 4067 | 0 | 0 | 47 | 129 s |
| ubuntu-latest / minimum + extras | 108646519605 | ubuntu-24.04 | 3.11.16 | numpy 1.26.4, pandas 2.1.4, scipy 1.11.4, scikit-learn 1.6.1, pytest 8.0.0, matplotlib 3.9.0, xgboost 2.0.0, lightgbm 4.0.0, catboost 1.2 | 4135 | 4091 | 0 | 0 | 44 | 152 s |
| ubuntu-latest / reference | 108646519530 | ubuntu-24.04 | 3.12.14 | numpy 2.5.2, pandas 3.0.5, scipy 1.18.0, scikit-learn 1.9.0, pytest 8.3.3 | 4113 | 4067 | 0 | 0 | 47 | 186 s |
| ubuntu-latest / reference + extras | 108646519507 | ubuntu-24.04 | 3.12.14 | numpy 2.5.2, pandas 3.0.5, scipy 1.18.0, scikit-learn 1.9.0, pytest 8.3.3, matplotlib 3.11.2, xgboost 3.4.1, lightgbm 4.7.0, catboost 1.2.10 | 4135 | 4091 | 0 | 0 | 44 | 234 s |
| windows-latest / build, install and smoke | 108646519479 | windows-latest | 3.12.10 | build 1.6.1, twine 7.0.0 (smoke) | — | smoke: atasetKit/dist/aidatasetkit-0.1.0a1-py3-none-any.whl: PASSED; setKit/AIDatasetKit/dist/aidatasetkit-0.1.0a1.tar.gz: PASSED; wheel: install, CLI, audit, artifacts, exit code -- all good; sdist: install, CLI, audit, artifacts, exit code -- all good; == Release smoke test passed | 0 | 0 | — | 145 s |
| windows-latest / minimum | 108646519603 | windows-latest | 3.11.9 | numpy 1.26.4, pandas 2.1.4, scipy 1.11.4, scikit-learn 1.6.1, pytest 8.0.0 | 4113 | 4068 | 0 | 0 | 46 | 236 s |
| windows-latest / minimum + extras | 108646519588 | windows-latest | 3.11.9 | numpy 1.26.4, pandas 2.1.4, scipy 1.11.4, scikit-learn 1.6.1, pytest 8.0.0, matplotlib 3.9.0, xgboost 2.0.0, lightgbm 4.0.0, catboost 1.2 | 4135 | 4092 | 0 | 0 | 43 | 261 s |
| windows-latest / reference | 108646519557 | windows-latest | 3.12.10 | numpy 2.5.2, pandas 3.0.5, scipy 1.18.0, scikit-learn 1.9.0, pytest 8.3.3 | 4113 | 4068 | 0 | 0 | 46 | 292 s |
| windows-latest / reference + extras | 108646519629 | windows-latest | 3.12.10 | numpy 2.5.2, pandas 3.0.5, scipy 1.18.0, scikit-learn 1.9.0, pytest 8.3.3, matplotlib 3.11.2, xgboost 3.4.1, lightgbm 4.7.0, catboost 1.2.10 | 4135 | 4092 | 0 | 0 | 43 | 308 s |

Linux coverage verified by test name in the uploaded JUnit reports (each of the four ubuntu jobs):
publication 72 passed + 1 Windows-only skip; failure injection 30; directory-sync injection 4 (the real POSIX
`_fsync_directory` runs in every other publish); process kill 6; concurrent writers 1; tampering 15; CLI
all-or-nothing and concurrent audits 7; golden audit 7; golden fixture diff 5; fingerprint chain 6; tie order 7;
final evaluation / leakage 39; packaging 73. Failures 0, errors 0 in every report.

## Run 36327967662 — commit `9d96e89e9dd5906989b5f9d887dd4c01ecb7172d` — conclusion **failure**

https://github.com/mohammed0115/AIDatasetKit/actions/runs/36327967662

Nine of ten jobs passed. `ubuntu-latest / minimum + extras` (job 108644293697): `3 failed, 4072 passed, 44 skipped`.

```
FAILED tests/integration/test_audit_end_to_end.py::TestTheGoldenSemanticArtifact::test_the_evidence_still_matches_it_exactly
FAILED tests/integration/test_audit_end_to_end.py::TestTheGoldenSemanticArtifact::test_the_canonical_bytes_match_too
FAILED tests/unit/test_capability_fingerprint_migration.py::TestTheCurrentCode::test_it_reproduces_the_fixture_for_this_pandas_major
- : "sha256:33da28196a2df143222e8b26571ce4d3",   (fixture: CustomerID dominant_value_digest)
+ : "sha256:d74be9daea1f1a4d69888e12ea278473",   (this runner)
```

Same numpy 1.26.4 / pandas 2.1.4 / scipy 1.11.4 / scikit-learn 1.6.1 as the passing `ubuntu-latest / minimum` job.
Cause and fix: see the certification report, section *Linux failures and repairs*.
