# G1-W5 — chunked profiling for CSV and TSV

Local evidence for the G1-W5 wave. This file does not certify the wave and
does not name a CI run. Branch CI is the next gate, recorded for review
outside this file.

## Scope and baseline

```text
BASELINE_SHA = 749048703fb885ec688bc3c57f91c4e16edbf50e
BRANCH       = g1-w5-chunked-profiling
CODE_SHA     = b4a7e28515f2f0eb6844efa729776de74254866f
```

`BASELINE_SHA` is certified `main`. G1-25 stays
`MISSING_PENDING_CERTIFICATION`. Recorded progress stays G1 16/26 = 61.54%
and overall 108/232 = 46.55%.

## Contract

- Opt-in only. `load_table`, `DataProfiler.profile`, and `aidatasetkit audit`
  without `--chunked-profile` do not take the chunked path. The default
  artifact carries `chunked_profiling: null`.
- With `--chunked-profile`, the command does not call `load_table` or
  `DataProfiler.profile`. It publishes a profile-only artifact. The verdict
  is blocked: "Full-table audit verdict is unavailable in chunked profiling
  mode." Approximate values are not verdict inputs.
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

## Local runs at CODE_SHA

Interpreter: CPython 3.12.14 (`.venv`). pandas 3.0.5.

| Run | Result |
|---|---|
| Focused chunked tests, then migration, golden, packaging, schema, and architecture tests | 29 chunked tests passed; the wider slice passed 249 |
| `scripts/g1_w5_mutations.py` | 16/16 KILLED, baseline exit 0, tree restored after each mutation. `rev=b4a7e28515f2f0eb6844efa729776de74254866f` |
| Full suite | 4520 passed, 47 skipped, 0 failed in 398.68s. JUnit `tests=4567 failures=0 errors=0 skipped=47` |
| `scripts/release_smoke_test.sh` | wheel and sdist: build, twine check, clean install, CLI audit, schema `1.2`. Version `0.1.0a1`. Nothing published |

The sixteen weakenings are: caller row limit ignored, JSON accepted, file read
in one piece, first chunk treated as the population, scratch directory left
behind, scratch files group-readable, approximate quartile unlabeled,
approximation changes the verdict, default audit opts in, `max_cells` bypass,
chunked evidence falsely claims the full population, chunk-size-dependent
result, full-table materialization in CLI chunked mode, collision-unsafe row
encoding, complete scratch-file read into memory, and chunked evidence omitted.

## Not in this wave

No merge to `main`. No change to the certified totals. No G1-W6 work.
JSON, Parquet, Feather, and Excel stay on the full-table path.
