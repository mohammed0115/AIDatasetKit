# G1-W6 — read-only SQLite table

G1-W6 is not a PASS. G1-14 stays `MISSING_PENDING_CERTIFICATION`.

This file records the local evidence for the implementation on
`IMPLEMENTATION_SHA`. It does not name a CI run. Branch CI on the tip that
adds this file is a later gate and is not a certification. `main` was not
moved.

## Identity

```text
STARTING_HEAD       = 431570c06dbf54125396909c2909c85f8e32e9de
IMPLEMENTATION_SHA  = 97d5283c632ef385c05e827a65fc1ac11da12e78
BRANCH              = g1-w6-sqlite-reader
BASELINE_MAIN_SHA   = 431570c06dbf54125396909c2909c85f8e32e9de
```

`318d61cbf7aa9564280bf9b9a623fbc0ccc87495` adds the reader, schema `1.3`, and
the mutation harness. `IMPLEMENTATION_SHA` is that commit plus the two test
and harness adjustments that make the missing-path and unwrapped-error
weakenings fail for the reason they name.

Recorded progress stays G1 17/26 = 65.38% and overall 109/232 = 46.98%.
Artifact schema `1.3`. Publication schema `1.0`. Package version `0.1.0a1`.
G1-25 stays `SUPPORTED_AND_TESTED`.

## Contract

- Local files ending in `.sqlite` or `.sqlite3` only. `.db` is refused.
- One ordinary user table. A view, a virtual table, and any name that begins
  with `sqlite_` are not eligible. Zero eligible tables is empty input. Exactly
  one is selected when `table` is omitted. More than one requires
  `load_table(..., table="name")` or `aidatasetkit audit --table NAME`.
  A missing, view, virtual, or internal name is a structured refusal. `table=`
  on any other format is a structured option error.
- The name is resolved against the catalog allowlist, then quoted by
  `quote_identifier`. Caller SQL, `ATTACH`, `executescript`, and extension
  loading are not accepted.
- The source must be an existing regular file. `max_source_bytes` is enforced
  before SQLite is opened. The connection is `sqlite3.connect(uri, uri=True)`
  with `mode=ro`, then `PRAGMA query_only=ON`. `setconfig` enables
  `SQLITE_DBCONFIG_DEFENSIVE`, disables trusted schema, and disables
  load-extension only when that method and those constants exist.
- `max_columns`, `max_rows`, `max_cells`, and `max_field_length` for both TEXT
  and BLOB are enforced from catalog metadata, `COUNT(*)`, and `LENGTH()`
  before the table is fetched or placed in a DataFrame. A refusal publishes
  nothing.
- INTEGER, REAL, TEXT, BLOB, and NULL stay `int`, `float`, `str`, `bytes`, and
  `None`. Column order is the catalog order. An empty eligible table returns
  an empty DataFrame with those columns. A zero-byte file is empty input. A
  file SQLite cannot read is a stable malformed-input error that does not
  include the raw path or SQL.
- `source_selector` is `{"kind": "table", "name": "<selected table>"}` on a
  SQLite artifact and `null` otherwise. The selected table is part of the
  config fingerprint. Warnings do not carry the table name. No SQL text is
  recorded. Excel sheet evidence was not retrofitted.
- Chunked profiling remains CSV and TSV only. PostgreSQL, SQLAlchemy, database
  URLs, and query-result ingestion stay unsupported.

## Schema migration

Golden semantic fingerprint before this field, schema `1.2`:

`39f4c43867f1b30f47a42986187c3cce4b8dbad1540e5e7a369e0e142d5d5d47`

After schema `1.3` and `source_selector: null`:

`2a33aff562bf32116d2f48587eda19af85e9a2db956a7e6c00b8a89c1bcf7c75`

Deleting `source_selector` and restoring schema `1.2` reproduces the earlier
hash. The embedded config fingerprint of the frame-built example is unchanged,
because a null selector is not written into `config.settings`.

## Local runs at IMPLEMENTATION_SHA

Interpreter: CPython 3.12.14 (`.venv`). pandas 3.0.5.

| Run | Result |
|---|---|
| Focused SQLite tests | 40 passed (`tests/unit/test_ingestion_sqlite.py`) |
| Evidence, fingerprint, schema, migration | 30 passed (`tests/unit/test_capability_fingerprint_migration.py`) |
| Golden fixtures and golden semantic artifact | 5 + 7 passed |
| G1-W5 chunked profiling | 35 passed (`tests/unit/test_chunked_profiling.py`) |
| Packaging | 73 passed |
| Architecture boundaries | 107 passed |
| Focused slice above, excluding packaging and architecture | 117 passed |
| `scripts/g1_w6_mutations.py` | 18/18 KILLED, 0 SURVIVED, baseline exit 0, tree restored after each mutation. `rev=97d5283c632ef385c05e827a65fc1ac11da12e78` python=3.12.14 |
| Full suite | 4571 passed, 47 skipped, 0 failed, 0 errors in 407.51s. JUnit `tests=4618 failures=0 errors=0 skipped=47 time=407.459` |
| `scripts/release_smoke_test.sh` | wheel `aidatasetkit-0.1.0a1-py3-none-any.whl` and sdist `aidatasetkit-0.1.0a1.tar.gz`: build, twine check, clean install, CLI audit, schema `1.3`. Version `0.1.0a1`. Nothing published |

The full suite was run again after `IMPLEMENTATION_SHA`, with these
documentation edits present and before this file was committed. The smoke test
ran on `318d61cbf7aa9564280bf9b9a623fbc0ccc87495`. That commit and
`IMPLEMENTATION_SHA` contain the same library code; the follow-up changes only
the missing-path test and the leak-mutation anchor.

## Mutations

Each weakening started from `IMPLEMENTATION_SHA`, matched its anchor once, and
was restored. Non-zero exit alone was not accepted.

| Id | Weakening | Killed because |
|---|---|---|
| M-W6-01 | `mode=ro` removed | `test_mode_ro_is_used` — the URI no longer contained `mode=ro` |
| M-W6-02 | `query_only` removed | `test_query_only_is_enabled` — `PRAGMA query_only=ON` was not executed |
| M-W6-03 | missing path creates a database | `test_a_missing_database_is_not_created` — the absent path was created |
| M-W6-04 | arbitrary/user SQL accepted | `test_an_injection_shaped_name_is_not_executed_as_sql` — a statement was not `PRAGMA` or `SELECT` |
| M-W6-05 | view accepted | `test_a_view_is_excluded` — a view was loaded |
| M-W6-06 | virtual table accepted | `test_a_virtual_table_is_excluded` — a virtual table was loaded |
| M-W6-07 | `sqlite_*` internal table accepted | `test_an_internal_table_is_excluded` — an internal table was eligible |
| M-W6-08 | multiple tables silently select one | `test_multiple_tables_require_an_explicit_name` — one table was chosen without `table=` |
| M-W6-09 | row limit checked after materialization | `test_too_many_rows_are_refused_before_fetch` — the row fetch ran before the limit refusal |
| M-W6-10 | `max_cells` ignored | `test_too_many_cells_are_refused` — an over-cell table was loaded |
| M-W6-11 | `max_field_length` ignores TEXT | `test_text_length_is_its_own_check` — oversized TEXT was loaded |
| M-W6-12 | `max_field_length` ignores BLOB | `test_an_oversized_blob_is_refused` — an oversized BLOB was loaded |
| M-W6-13 | table selector omitted from artifact evidence | `test_the_artifact_records_the_table` — `source_selector` was null |
| M-W6-14 | table selector omitted from config fingerprint | `test_a_different_table_changes_the_config_fingerprint` — two tables shared a config fingerprint |
| M-W6-15 | `.db` accepted | `test_a_db_suffix_is_refused` — a `.db` file was accepted |
| M-W6-16 | `table=` accepted for a non-SQLite format | `test_table_is_refused_for_other_formats` — `table=` was accepted for another format |
| M-W6-17 | SQLite error leaks raw path/SQL or escapes unwrapped | `test_a_corrupt_database_does_not_leak_the_driver_error` — the driver error escaped the stable malformed-input error |
| M-W6-18 | schema incorrectly remains `1.2` | `test_the_artifact_records_the_table` — the artifact schema was not `1.3` |

## Status

Not certified. Expected only after a later certification: G1 18/26 = 69.23%
and overall 110/232 = 47.41%.
