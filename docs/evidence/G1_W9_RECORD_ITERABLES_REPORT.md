# G1-W9 — Bounded synchronous iterables of mapping records

G1-W9 is **PASS**. G1-13 is `SUPPORTED_AND_TESTED`.

`CERTIFIED_CODE_SHA` is the tip integrated to `main` before this record.
`DOCS_SEAL_SHA` is the documentation-only commit that adds this certification.
This file names `CERTIFIED_CODE_SHA` and does not name `DOCS_SEAL_SHA` or any
CI run started by that documentation commit.

```text
STARTING_HEAD            = 300ce4374fde84dbb93e7e13e5ec5ae703f27e9e
CERTIFIED_CODE_SHA       = 1f436ad0091318ad552bc722d8d56afe184e0b51
DOCS_SEAL_SHA            = not recorded in this file
BRANCH                   = g1-w9-record-iterables
G1_W9_STATUS             = PASS
G1_13_STATUS             = SUPPORTED_AND_TESTED
ARTIFACT_SCHEMA          = 1.4
PUBLICATION_SCHEMA       = 1.0
PACKAGE                  = 0.1.0a1
G1_PROGRESS              = 21/26 = 80.77%
OVERALL_PROGRESS         = 113/232 = 48.71%
```

Before this seal the recorded progress was G1 20/26 = 76.92% and overall
112/232 = 48.28%. Iterable support belongs to `load_table`. `AIDataFacade.load`
remains DataFrame-only.

## What this wave does

`load_table` accepts a synchronous iterable of mappings in addition to a list or tuple. The iterable is consumed once. `iter(source)` is called once. At most `max_records + 1` items are pulled. The extra item is not validated, copied, or kept, and nothing after it is pulled. Each accepted mapping is shallow-copied with `dict(record)` before the next pull.

The records limits apply: `max_records`, `max_keys_per_record`, `max_columns`, `max_cells`, and `max_record_chars`. `max_rows` and `max_field_length` are not applied on this path, so an iterable agrees with the equivalent list. Absent keys count as cells in the rectangle. Column order is first key appearance. Row order is yield order.

The result is one DataFrame with `source_kind` `records`, null format, null `source_selector`, and `memory_bytes` of that frame. A list and an equivalent iterable share the frame, the missing-key warning, and the dataset, semantic, and config fingerprints. No generator type, function name, repr, or address is recorded.

A `types.GeneratorType` iterator is closed on success and on every refusal. Other iterators are not closed. Lists, tuples, and DataFrames stay on their existing branches and are not closed.

## What this wave does not do

This is not a streaming profiler. Async iterables and async generators are refused before iteration, and `aclose()` is not called. A bare mapping, `str`, `bytes`, `bytearray`, `set`, `frozenset`, and `range` are refused. Dictionary keys are not rows.

`AIDataFacade.load` remains DataFrame-only. Closing G1-13 does not require the facade to accept an iterator. The composition is `facade.load(load_table(source).frame)`.

There is no new dependency, CLI flag, schema field, or package version. Schema stays `1.4`. Publication schema stays `1.0`. Package version stays `0.1.0a1`.

Blocking iterators, network-backed iterators, side effects while pulling, hostile mappings, expensive `__str__` on values, secrets inside the caller's own exceptions, and nested mutable values remain the caller's responsibility. The library does not claim a wall-clock or network cancellation bound.

## CI

These runs are for `CERTIFIED_CODE_SHA`. This section does not name a CI run of the commit that adds it.

```text
CERTIFIED_CODE_SHA = 1f436ad0091318ad552bc722d8d56afe184e0b51
BRANCH_CI_RUN      = 38264865799
BRANCH_CI_URL      = https://github.com/mohammed0115/AIDatasetKit/actions/runs/38264865799
BRANCH_CI_SHA      = 1f436ad0091318ad552bc722d8d56afe184e0b51
BRANCH_CI_EVENT    = push
BRANCH_CI_BRANCH   = g1-w9-record-iterables
BRANCH_CI_RESULT   = success
BRANCH_CI_JOBS     = 10/10
BRANCH_CI_STARTED  = 2026-10-12T14:42:25Z
BRANCH_CI_UPDATED  = 2026-10-12T14:49:42Z
MAIN_CI_RUN        = 38291797343
MAIN_CI_URL        = https://github.com/mohammed0115/AIDatasetKit/actions/runs/38291797343
MAIN_CI_SHA        = 1f436ad0091318ad552bc722d8d56afe184e0b51
MAIN_CI_EVENT      = push
MAIN_CI_BRANCH     = main
MAIN_CI_RESULT     = success
MAIN_CI_JOBS       = 10/10
MAIN_CI_STARTED    = 2026-10-12T18:34:12Z
MAIN_CI_UPDATED    = 2026-10-12T18:43:22Z
```

## Status

G1-W9 = PASS. G1-13 = `SUPPORTED_AND_TESTED` for `load_table` synchronous iterable ingestion. `AIDataFacade.load` remains DataFrame-only. G1-17 stays `PARTIAL`. G1-14, G1-22, G1-24, and G1-25 stay `SUPPORTED_AND_TESTED`. G1-15, G1-05, G1-10, and G1-11 stay `MISSING`. G2-16 stays `MISSING`. Artifact schema `1.4`. Publication schema `1.0`. Package version `0.1.0a1`. Recorded progress is G1 21/26 = 80.77% and overall 113/232 = 48.71%.
