# G1-W5 — chunked profiling for CSV and TSV

G1-W5 is **PASS**. G1-25 is `SUPPORTED_AND_TESTED`.

`CERTIFIED_CODE_SHA` is the code tip recorded below. `DOCS_SEAL_SHA` is the
documentation-only commit that adds this record. This file names
`CERTIFIED_CODE_SHA` and does not name `DOCS_SEAL_SHA` or any CI run started
by that documentation commit.

## Identity

```text
CERTIFIED_CODE_SHA = 5f65a20331823e49f35b25a657c37868aea509bd
API_CLEANUP_SHA    = dbd6447cd6a0279a08bf14ab564aced0c661a5d3
BASELINE_MAIN_SHA  = 749048703fb885ec688bc3c57f91c4e16edbf50e
BRANCH             = g1-w5-chunked-profiling
DOCS_SEAL_SHA      = not recorded in this file
```

`API_CLEANUP_SHA` removes the ignored `settings` argument from
`AuditBuilder.build_chunked`. `CERTIFIED_CODE_SHA` is that change plus the
changelog note of the same removal. `main` was fast-forwarded from
`BASELINE_MAIN_SHA` to `CERTIFIED_CODE_SHA` before this documentation commit.

Recorded progress on `CERTIFIED_CODE_SHA` is G1 17/26 = 65.38% and overall
109/232 = 46.98%. Artifact schema `1.2`. Publication schema `1.0`. Package
version `0.1.0a1`. G1-24 stays `MISSING`.

## Contract

- Opt-in only. `load_table`, `DataProfiler.profile`, and `aidatasetkit audit`
  without `--chunked-profile` do not take the chunked path. The default
  artifact carries `chunked_profiling: null`.
- With `--chunked-profile`, the command does not call `load_table` or
  `DataProfiler.profile`. It publishes a profile-only artifact at stage
  `profiled`. The verdict is blocked: "Full-table audit verdict is unavailable
  in chunked profiling mode." Approximate values are not verdict inputs.
- The ingestion record names the file contract: source kind, format, encoding,
  delimiter, delimiter source, header, and the population shape. `memory_bytes`
  is null. The config fingerprint is only the scan contract produced by
  `_chunked_settings(profile)`, including `quantile_sample_size`, and excludes
  paths and timestamps. `AuditBuilder.build_chunked` takes no `settings`
  argument. Passing `settings=` raises `TypeError`.
- CSV and TSV only. Any other suffix is `InvalidIngestionOptionsError`.
- The scan covers every row the validating parser accepted. The fingerprint
  is that population, and it matches `dataset_fingerprint` of the full table.
  Hash bytes are read back in fixed-size blocks.
- Exact: row count, duplicate rows, per-column count, missing count, finite
  and infinite counts, distinct count, and, for a numeric column, minimum,
  maximum, sum and mean. Distinct values and row identity are length-prefixed
  and stored on disk.
- Quartiles are always a bounded deterministic reservoir sample
  (`deterministic_reservoir`), labeled `deterministic_approximation`. They are
  not stored as exact column fields.
- `IngestionLimits` are unchanged, including `max_cells`. An over-limit file
  is refused before a scratch directory is created.
- Scratch lives in a `0700` directory; files are `0600`; the directory is
  removed on success, on failure and on interruption.
- Artifact schema `1.2`. `chunked_profiling` is a typed record, `null` on the
  default path. Publication schema `1.0`. Package version `0.1.0a1`.

## Local runs at CERTIFIED_CODE_SHA

Interpreter: CPython 3.12.14 (`.venv`). pandas 3.0.5.

| Run | Result |
|---|---|
| Focused chunked tests | 35 passed (`tests/unit/test_chunked_profiling.py`), including `test_build_chunked_has_no_settings_parameter` |
| Schema, migration, golden, and packaging | Dedicated slice with the focused tests: 149 passed in 33.62s. Full-suite counts: migration 27, golden 5, packaging 73, evidence artifact 139, all inside the suite below |
| `scripts/g1_w5_mutations.py` | 19/19 KILLED, 0 SURVIVED, baseline exit 0, tree restored after each mutation. `rev=5f65a20331823e49f35b25a657c37868aea509bd` python=3.12.14 |
| Full suite | 4526 passed, 47 skipped, 0 failed, 0 errors in 430.35s. JUnit `tests=4573 failures=0 errors=0 skipped=47` |
| `scripts/release_smoke_test.sh` | wheel `aidatasetkit-0.1.0a1-py3-none-any.whl` and sdist `aidatasetkit-0.1.0a1.tar.gz`: build, twine check, clean install, CLI audit, schema `1.2`. Version `0.1.0a1`. Nothing published |

The nineteen weakenings are the sixteen from the bounded-memory repair, plus
chunked artifact falsely marked inspected, chunked settings omitted from
config identity, and chunked ingestion evidence omitted.

## CI

These runs are for `CERTIFIED_CODE_SHA`. This section does not name a CI run
of the commit that adds it.

```text
CERTIFIED_CODE_SHA = 5f65a20331823e49f35b25a657c37868aea509bd
BRANCH_CI_RUN      = 38061380208
BRANCH_CI_URL      = https://github.com/mohammed0115/AIDatasetKit/actions/runs/38061380208
BRANCH_CI_SHA      = 5f65a20331823e49f35b25a657c37868aea509bd
BRANCH_CI_EVENT    = push
BRANCH_CI_BRANCH   = g1-w5-chunked-profiling
BRANCH_CI_RESULT   = success
BRANCH_CI_JOBS     = 10/10
BRANCH_CI_STARTED  = 2026-10-10T14:53:05Z
BRANCH_CI_UPDATED  = 2026-10-10T14:59:42Z
MAIN_CI_RUN        = 38061887792
MAIN_CI_URL        = https://github.com/mohammed0115/AIDatasetKit/actions/runs/38061887792
MAIN_CI_SHA        = 5f65a20331823e49f35b25a657c37868aea509bd
MAIN_CI_EVENT      = push
MAIN_CI_BRANCH     = main
MAIN_CI_RESULT     = success
MAIN_CI_JOBS       = 10/10
MAIN_CI_STARTED    = 2026-10-10T15:00:45Z
MAIN_CI_UPDATED    = 2026-10-10T15:06:24Z
```

## Scope confirmation

G1-W6 was not started. JSON, Parquet, Feather, and Excel stay on the
full-table path. No tag, release, or PyPI publication occurred.
