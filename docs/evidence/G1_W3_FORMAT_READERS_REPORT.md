# G1-W3 — Format readers: JSON, JSONL, Parquet, Feather

Evidence for the G1-W3 wave of `docs/AIDATASETKIT_EXECUTION_ROADMAP.md`
(audit rows G1-06…G1-09). Same discipline as the G1-W1/G1-W2 reports: every
claim below names the run that proves it.

## Scope and baseline

```text
PRE_CLOSURE_SHA  = 6ba8817891f59f8c0c4861a06f9338f8ae2202d6 (main = origin/main, clean)
BRANCH           = g1-w3-format-readers
FINAL_BRANCH_SHA = 638d9374c3317d10f0de56ba5e7a3d8e53b1760d
```

Wave content, per the roadmap: Parquet and Feather (pyarrow as an optional
extra), JSON and JSONL. G1-W1 and G1-W2 are certified and were not redone.

## What was built

- **`aidatasetkit/ingestion/json_text.py`** — a `.json` file is read as an
  array of flat objects; a `.jsonl`/`.ndjson` file as one object per line.
  Duplicate keys, nested values, non-standard `NaN`/`Infinity` constants and
  non-object records are refused with structured errors, never read into a
  guessed table. JSONL is checked as it streams, so a limit refusal stops
  reading at the offending line; a JSON document is one value, bounded by the
  loader's `max_source_bytes` preflight. Both end at the one records authority
  in the loader, shared as `_records_frame`; the literal anchors the certified
  G1-W1/G1-W2 mutation harnesses match were left intact.
- **`aidatasetkit/ingestion/columnar.py`** — Parquet enforces the row, column
  and cell budgets from the footer **before any value is materialized**, and
  the counts found while reading must agree with the footer. Feather's IPC
  footer carries only the schema, so the column budget is enforced before the
  data and the row/cell budgets on the Arrow table before the pandas
  conversion; the loader's byte preflight bounds the file either way. Corrupt
  files are `MalformedInputError`, never a bare pyarrow error; a missing
  pyarrow is `MissingDependencyError` naming the extra.
- **Dependency**: new optional extra `parquet = ["pyarrow>=14.0.1"]`. The floor
  is the CVE-2023-47248 fix (arbitrary code execution deserializing a
  malicious Parquet/Arrow-IPC file), not the oldest importable version. Pinned
  in both constraints files; installed in the CI extras jobs.

## Test evidence

Focused runs during development (reference venv, CPython 3.12.14, pandas
3.0.5, pyarrow 25.0.1):

```text
tests/unit/test_ingestion_json.py + test_ingestion.py + test_ingestion_limits.py = 197 passed
tests/unit/test_ingestion_columnar.py + test_ingestion_without_pyarrow.py
    + test_ingestion.py + test_dependency_contract.py                    = 166 passed
tests/integration/test_cli_ingestion.py                                  =  22 passed
```

Full suite from the exact committed state, run from the checkout:

```text
TESTED_SHA  = 638d9374c3317d10f0de56ba5e7a3d8e53b1760d
COMMAND     = python -m pytest -rfE --junitxml=junit.xml -q
PLATFORM    = linux, CPython 3.12.14, pandas 3.0.5, pyarrow 25.0.1
RESULT      = 4456 passed, 47 skipped, 0 failed, 0 errors in 516.61s
JUNIT       = tests=4503 failures=0 errors=0 skipped=47 time=516.545
```

The two failures seen mid-wave were both closed before this run:
`test_an_unsupported_file_type_says_so` (the CLI message grew new formats) and
`TestNumpyStringLabels::test_the_labels_are_normalised_and_the_reason_recorded`
— the fixture assumed `rename()` keeps `np.str_` labels; pandas 3 coerces them
to plain `str` through every public construction path, so the fixture now
builds the index element-by-element with `dtype=object`. The sklearn rejection
the guard defends is unchanged, so the defect stays real.

## Mutation evidence

`scripts/g1_w3_mutations.py` — 11 dedicated mutations on the **new** seams
only, same archive/anchor/kill/restore discipline as the certified G1-W2
harness:

| ID | Weakening | Killing test | Result |
|---|---|---|---|
| M-W3-01 | JSON top-level array check removed | test_the_top_level_must_be_an_array | KILLED |
| M-W3-02 | duplicate keys kept (json.loads default) | test_a_duplicate_key_is_refused_not_kept_last, test_a_duplicate_key_names_its_line | KILLED |
| M-W3-03 | nested objects/arrays accepted | test_a_nested_object_is_refused_naming_its_key, test_a_nested_array_is_refused_naming_its_key | KILLED |
| M-W3-04 | NaN/Infinity accepted | test_non_standard_constants_are_refused | KILLED |
| M-W3-05 | JSONL refusal deferred to end of file | test_a_jsonl_refusal_reads_no_line_past_the_decisive_one | KILLED |
| M-W3-06 | JSON record boundary `>=` → `>` | test_exactly_at_each_limit_is_accepted_and_one_under_is_refused (json, jsonl) | KILLED |
| M-W3-07 | Parquet footer shape preflight skipped | test_a_parquet_over_the_row_limit_is_refused_without_reading_data | KILLED |
| M-W3-08 | corrupt Parquet → bare pyarrow error | test_a_corrupt_file_is_malformed_input_never_a_bare_pyarrow_error[broken.parquet] | KILLED |
| M-W3-09 | Feather shape checked against faked rows | test_a_feather_over_the_row_limit_is_refused_before_the_pandas_conversion | KILLED |
| M-W3-10 | text options accepted for binary formats | test_text_options_are_refused_for_binary_formats | KILLED |
| M-W3-11 | missing pyarrow → bare ImportError | test_a_columnar_read_names_the_extra | KILLED |

```text
G1_W3_MUTATIONS = 11 total, 11 KILLED, 0 SURVIVED, 0 HARNESS_ERROR, 0 TEST_ENVIRONMENT_ERROR
G1_W2 harness re-run on this tree: 13/13 KILLED, 0 errors
G1_W1 harness re-run on this tree: 13/13 (exit 1 each, none silent)
```

## Compatibility

Artifact schema `1.1`, publication schema `1.0`, package version `0.1.0a1`,
semantic fingerprint: unchanged. No path under `aidatasetkit/evidence`,
`examples`, golden fixtures, or the model/training layers changed. One new
optional dependency (`pyarrow`) added under the `parquet` extra; the core
`dependencies` list is untouched and the dependency-contract tests pass.

## CI

```text
BRANCH_CI = pending (recorded once observed on FINAL_BRANCH_SHA)
MAIN_CI   = pending
```

## Scope confirmation

G1-W4 (XLSX), chunked reading, date inference, encoding detection, databases,
URLs, generators: none started. No tag, release, or PyPI publication occurred.
