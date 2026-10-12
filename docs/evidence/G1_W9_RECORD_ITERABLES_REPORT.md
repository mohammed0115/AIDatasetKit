# G1-W9 — Bounded synchronous iterables of mapping records

Status: implemented, pending review. This is not PASS. G1-13 is not promoted.

```text
STARTING_HEAD            = 300ce4374fde84dbb93e7e13e5ec5ae703f27e9e
BRANCH                   = g1-w9-record-iterables
G1_13_STATUS             = PARTIAL_PENDING_CERTIFICATION
ARTIFACT_SCHEMA          = 1.4
PUBLICATION_SCHEMA       = 1.0
PACKAGE                  = 0.1.0a1
G1_PROGRESS              = 20/26 = 76.92%
OVERALL_PROGRESS         = 112/232 = 48.28%
```

Branch CI is not named in this file. The review package reports it. `origin/main` stays at the starting SHA until a later review says otherwise.

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

## Progress

Until a later review:

- G1-13 remains `PARTIAL_PENDING_CERTIFICATION`
- G1 remains 20/26 = 76.92%
- overall remains 112/232 = 48.28%

A later promotion would be the first time those three figures move, to G1 21/26 = 80.77% and overall 113/232 = 48.71%.
