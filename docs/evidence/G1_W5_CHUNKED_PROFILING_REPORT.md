# G1-W5 — chunked profiling for CSV and TSV

Local evidence for the G1-W5 wave. This file does not certify the wave and
does not name a CI run. Branch CI is the next gate, recorded for review
outside this file.

## Scope and baseline

```text
BASELINE_SHA = 749048703fb885ec688bc3c57f91c4e16edbf50e
BRANCH       = g1-w5-chunked-profiling
CODE_SHA     = the semantic-closure implementation on this branch
```

`BASELINE_SHA` is certified `main`. G1-25 stays
`MISSING_PENDING_CERTIFICATION`. Recorded progress stays G1 16/26 = 61.54%
and overall 108/232 = 46.55%.

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
  is null. The config fingerprint is the effective scan settings, including
  `quantile_sample_size`, and excludes paths and timestamps.
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

The bounded-memory repair's 16/16 matrix remains the result of
`b4a7e28515f2f0eb6844efa729776de74254866f`. This closure adds the stage,
ingestion record, config identity and `quantile_sample_size` rename. Its
measured counts are written here once that local run is recorded. They do not
name a CI run.

The harness kills nineteen weakenings. The sixteen from the bounded-memory
repair, plus: chunked artifact falsely marked inspected, chunked settings
omitted from config identity, and chunked ingestion evidence omitted.

## Not in this wave

No merge to `main`. No change to the certified totals. No G1-W6 work.
JSON, Parquet, Feather, and Excel stay on the full-table path.
