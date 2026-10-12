# G1-W8 — Strict ISO date and datetime text

Status: implemented, pending review. This is not PASS. G1-22 is not promoted.

```text
STARTING_HEAD            = 270570ffbe5712c118365753bb58d7bf8a1523a0
REVIEWED_HEAD            = b47b0c3229ff2e98ca0afe33d639dfe60e1b5442
REPAIR_COMMIT            = d671b1db5eaa3476c2c0a9bf8194edd6785f5741
BRANCH                   = g1-w8-date-text-inference
G1_22_STATUS             = PARTIAL_PENDING_CERTIFICATION
ARTIFACT_SCHEMA          = 1.4
PUBLICATION_SCHEMA       = 1.0
PACKAGE                  = 0.1.0a1
G1_PROGRESS              = 19/26 = 73.08%
OVERALL_PROGRESS         = 111/232 = 47.84%
```

Branch CI is not named in this file. The review package reports it. `origin/main` stays at the starting SHA until a later review says otherwise.

## What this wave does

A full default audit inspects textual columns. When every non-blank value is a strict ISO date or datetime, and at least 8 such values are present, the artifact records that column under `date_inference`.

Supported forms, matched against the whole string before calendar validation:

- `YYYY-MM-DD`
- `YYYY-MM-DDTHH:MM:SS`
- an optional fractional second
- an optional trailing `Z` (recorded as `utc_z`)
- an optional numeric offset `±HH:MM` (recorded as `explicit_offset`, not shifted to the host zone)

`T` and the seconds field are required. Surrounding whitespace is not removed. A date inside a longer string is not accepted. Space-separated date and time is not accepted. Slash dates, month names, locale forms, `YYYY-MM`, year-only values, `YYYYMMDD`, Unix timestamps, and Excel serials are not accepted.

Nulls and empty strings are ignored. A whitespace-only string blocks the column. One invalid value blocks the column. The scan reads the whole column; it does not sample. A constant valid date can be recognized. A mix of date-only and datetime values is recorded as `datetime`. Mixing naive, `Z`, and explicit offsets is recorded as `mixed` and is not normalized.

## What this wave does not do

Recognition is not conversion. The loaded DataFrame, its dtypes, `LoadedTable`, operational `detected_kind`, column profile kind, quality findings, task detection, target role, feature role, preprocessing plan, lineage, verdict, dataset fingerprint, config fingerprint, and visualization behavior stay as they were. There is no caller-selectable date policy, no CLI flag, and no `load_table` parameter. `parse_dates` is not set. No new third-party dependency is added.

Chunked profiling does not run this scan. Its artifact has `date_inference: null` and cannot claim `full_population_scanned` for dates. The blocked chunked verdict is unchanged.

`datetime64` columns, including timezone-aware ones, stay on the existing datetime path. Python `datetime.date` objects are not textual candidates.

## Schema

Artifact schema moves from `1.3` to `1.4` because `date_inference` is a new top-level key. A full audit always carries the policy object. `columns` is empty when nothing qualified. Publication schema stays `1.0`. Package version stays `0.1.0a1`.

The churn example has no strict ISO column. Its semantic fingerprint moved from `2a33aff562bf32116d2f48587eda19af85e9a2db956a7e6c00b8a89c1bcf7c75` to `60375e40773cb0849085a4867c31e609b90c34a60a2cb8842c9e34f37b3a237e` only because the schema version and the empty `date_inference` record were added. Deleting that key and setting the version back to `1.3` reproduces the previous identity. The dataset fingerprint did not move.

## Stale illustrative output

Review of `b47b0c3229ff2e98ca0afe33d639dfe60e1b5442` found that `examples/audit_churn/audit.json`, `lineage.json`, and `report.html` were still schema `1.3` and omitted `date_inference`. The semantic fixtures were already `1.4`. Current illustrations must not contradict the artifact the package writes.

Repair commit `d671b1db5eaa3476c2c0a9bf8194edd6785f5741` regenerates those three files through `examples/audit_churn/generate_artifacts.py`. The generated audit is schema `1.4`, carries `date_inference` with `method` `strict_iso_full_population_v1`, `conversion_performed` false, `affects_detected_kind` false, `affects_verdict` false, and an empty `columns` list. The HTML renders that section, states that conversion was not performed, and leaves each column's detected kind as profiled. `tests/unit/test_current_example_contract.py` fails if `ARTIFACT_SCHEMA_VERSION` moves and these files are not regenerated.

`expected_audit_semantic.json` and `expected_audit_semantic.pandas2.json` were not rewritten by this repair. Their semantic fingerprint stays `60375e40773cb0849085a4867c31e609b90c34a60a2cb8842c9e34f37b3a237e` on the pandas 3 fixture. The only environment field that changed in the illustration is `python`, from `3.12.3` to `3.12.14`, because the generator records the interpreter that produced the file. Created-at stays `2025-02-01T00:00:00Z`. No workspace path was written.

G1-22 is still not certified. This repair does not mark G1-W8 PASS.

## Evidence shape

```text
date_inference:
  method: strict_iso_full_population_v1
  population: full
  full_population_scanned: true
  minimum_non_blank: 8
  required_match_ratio: 1.0
  conversion_performed: false
  affects_detected_kind: false
  affects_verdict: false
  columns: [ column, inferred_semantic_type, format_family, counts, timezone_policy ]
```

The report renders this section apart from detected kind and states that conversion was not performed.

## Progress

Until a later review:

- G1-22 remains `PARTIAL_PENDING_CERTIFICATION`
- G1 remains 19/26 = 73.08%
- overall remains 111/232 = 47.84%

A later promotion would be the first time those three figures move.
