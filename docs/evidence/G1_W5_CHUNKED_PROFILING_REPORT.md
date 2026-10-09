# G1-W5 — chunked profiling for CSV and TSV

Local evidence for the G1-W5 wave. This file does not certify the wave and
does not name a CI run. Branch CI is the next gate, recorded for review
outside this file.

## Scope and baseline

```text
BASELINE_SHA = 749048703fb885ec688bc3c57f91c4e16edbf50e
BRANCH       = g1-w5-chunked-profiling
CODE_SHA     = f33eb3975faa5121dcf6fab13336a1360fc8867b
```

`BASELINE_SHA` is certified `main`. G1-25 stays
`MISSING_PENDING_CERTIFICATION`. Recorded progress stays G1 16/26 = 61.54%
and overall 108/232 = 46.55%.

## Contract

- Opt-in only. `load_table`, `DataProfiler.profile`, and `aidatasetkit audit`
  without `--chunked-profile` do not take the chunked path. The default
  artifact carries `chunked_profiling: null`.
- CSV and TSV only. Any other suffix is `InvalidIngestionOptionsError`.
- The scan covers every row the validating parser accepted. The fingerprint
  is that population, and it matches `dataset_fingerprint` of the full table.
- Exact: row count, duplicate rows, per-column count, missing count, distinct
  count, and, for a numeric column, minimum, maximum, sum and mean.
- Quartiles are exact unless the caller sets `approximate_quantiles_above`
  below the number of finite values. That case is returned only as
  `deterministic_approximation` / `deterministic_even_stride`, and the exact
  quartile fields stay null. The verdict does not read the record.
- `IngestionLimits` are unchanged. An over-limit file is refused before a
  scratch directory is created.
- Scratch lives in a `0700` directory; files are `0600`; the directory is
  removed on success and on failure.
- Artifact schema `1.2` (additive). Publication schema `1.0`. Package version
  `0.1.0a1`.

## Local runs at CODE_SHA

Interpreter: CPython 3.12.14 (`.venv`). pandas 3.0.5.

| Run | Result |
|---|---|
| Focused chunked, migration, golden, packaging, and schema tests | passed before the full suite |
| `scripts/g1_w5_mutations.py` | 9/9 KILLED, baseline exit 0, tree restored after each mutation. `rev=f33eb3975faa5121dcf6fab13336a1360fc8867b` |
| Full suite | 4508 passed, 47 skipped, 0 failed in 421.02s. JUnit `tests=4555 failures=0 errors=0 skipped=47` |
| `scripts/release_smoke_test.sh` | wheel and sdist: build, twine check, clean install, CLI audit, schema `1.2`, verdict `blocked` on the synthetic file. Version `0.1.0a1`. Nothing published |

The nine weakenings are: caller row limit ignored, JSON accepted, file read
in one piece, first chunk treated as the population, scratch directory left
behind, scratch files group-readable, approximate quartile unlabeled,
approximation changes the verdict, default audit opts in.

## Not in this wave

No merge to `main`. No change to the certified totals. No G1-W6 work.
JSON, Parquet, Feather, and Excel stay on the full-table path.
