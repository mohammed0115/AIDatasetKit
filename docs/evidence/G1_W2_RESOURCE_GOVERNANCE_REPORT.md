# G1-W2 Resource Governance and Large-Input Safety

```text
G1_W2_AUTHORIZATION = GO
SCOPE = resource governance and large-input safety only
BRANCH = g1-w2-resource-governance
BASELINE_SHA = 79b66dad2863bd97cbc641d0353f6d65ad9ec839
FIRST_CODE_SHA = d2712fb7ccb197c4e88805c38e4471fb8ca9b903
LOCAL_TESTED_SHA = f128ccd6b63507e0ba4baa2dcc0526360fc31e0d
PACKAGING_TESTED_SHA = d398926e87bed127b7dfb4971d971d7f6dc29ca5
FINAL_BRANCH_SHA = PENDING
FINAL_MAIN_SHA = PENDING
CURRENT_HEAD = d398926e87bed127b7dfb4971d971d7f6dc29ca5
ORIGIN_MAIN = 79b66dad2863bd97cbc641d0353f6d65ad9ec839
WORKTREE_CLEAN = YES
G1_W2_FINAL_GATE = BLOCKED_PENDING_REMOTE_CI
G1_W3_AUTHORIZATION = NO_GO_PENDING_CTO_REVIEW
```

## Authority and policy

`aidatasetkit.ingestion.load_table` is the single ingestion authority. It accepts
CSV/TSV `Path` values, pandas DataFrames, and list/tuple records. The immutable
`IngestionLimits` policy is defined in `aidatasetkit/ingestion/types.py` and is
passed through `loader.py` and `delimited.py`.

Default finite limits:

| Limit | Default |
|---|---:|
| `max_source_bytes` | 67,108,864 |
| `max_rows` | 1,000,000 |
| `max_columns` | 1,000 |
| `max_cells` | 10,000,000 |
| `max_field_length` | 1,000,000 characters |
| `max_records` | 1,000,000 |
| `max_keys_per_record` | 1,000 |
| `max_record_chars` | 10,000,000 characters |

The defaults are conservative first-line operational ceilings, not empirically
optimal limits: 64 MiB bounds a single whole-table file allocation; one million
rows bounds parser work; 1,000 columns and 10 million cells bound table shape
and downstream allocation volume; one million characters bounds a parser field;
one million records and 1,000 keys bound in-memory record inputs; and 10 million
stringified value characters bounds their validation/materialization payload.
Python callers may override each limit with another positive integer or
explicitly pass `None` to disable that one limit. CLI callers can override bytes,
rows, columns, cells and field length; omitted CLI flags retain finite defaults.

Exact public API delta:

- New public export: `aidatasetkit.ingestion.IngestionLimits`.
- `load_table(source, *, options=None, limits=None)` adds an optional
  keyword-only policy; existing calls remain compatible.
- `AIDataFacade.load(train, test=None, *, limits=None)` adds an optional
  keyword-only policy and delegates to `load_table` before analysis.
- `aidatasetkit.ingestion` exports `ResourceLimitError` and typed subclasses:
  `FileSizeLimitError`, `RowLimitError`, `ColumnLimitError`, `CellLimitError`,
  `FieldLengthLimitError`, `RecordLimitError` and `KeyLimitError`.
- CLI options added: `--max-input-bytes`, `--max-rows`, `--max-columns`,
  `--max-cells` and `--max-field-length`.
- Package version and runtime dependency manifests are unchanged.

Positive integers are accepted; `None` is an explicit Python API disable value.
Booleans, zero, negative and non-integer values are refused. CLI omission uses
the same finite defaults and exposes the first five file/table limits.

## Enforcement route

- Files are checked for regular-file status and byte size before delimiter
  planning and before `pandas.read_csv`.
- CSV/TSV shape, data rows, data cells and field length are checked during the
  existing strict streaming validation pass. The parser's native field-size
  guard avoids a second Python scan of every field.
- DataFrames are checked for rows, columns and cells before memory measurement;
  the original object is returned without mutation or deep copy.
- Records are checked for record count, keys, accumulated value characters,
  columns and cells before DataFrame materialization.
- The facade delegates to `load_table` before storing frames or beginning
  profiling, quality inspection, task detection or model work.
- `ResourceLimitError` and its typed subclasses carry a stable code,
  `limit_name`, configured limit, observed value and input kind. Messages do
  not include source paths or cell values.

Generators and arbitrary iterables remain refused. Accepted inputs are still
materialized as one pandas table; this is not true chunked profiling.

## Safety evidence

Focused tests:

- `tests/unit/test_ingestion_limits.py`: finite defaults, immutable policy,
  invalid policies, exact and over boundaries, Unicode field length, DataFrame
  identity/no mutation, records, CLI defaults, and facade short-circuiting.
- `tests/integration/test_cli_ingestion.py`: structured resource refusal,
  one-line CLI error, no cell/path leakage, and preservation of an existing
  publication and `CURRENT`.
- `tests/integration/test_facade.py`: existing facade workflow compatibility.
- `tests/integration/test_architecture_boundaries.py`: import and layer rules.

Latest focused result before this report was written: `218 passed in 12.82s`.
After the additional boundary and publication tests, the narrower runs passed:
`120 passed` in the ingestion unit slice and `119 passed` in the CLI/facade
integration slice. The architecture/package slice passed `180 passed, 1
skipped` (the Windows-minimum isolated-environment test is skipped locally).
The final local suite at `f128ccd6b63507e0ba4baa2dcc0526360fc31e0d` passed
`4350 passed, 46 skipped`.

Exact verification commands and state:

| Check | Command | SHA/state | Result |
|---|---|---|---|
| Focused safety | `python -m pytest -q tests/unit/test_ingestion_limits.py tests/unit/test_ingestion.py tests/integration/test_cli_ingestion.py tests/integration/test_facade.py` | `f128ccd` code; docs-only descendants | 124 passed in the unit-ingestion/facade slice; later CLI/facade slice 119 passed |
| Current artifact/safety | `python -m pytest -q tests/unit/test_capability_fingerprint_migration.py tests/unit/test_golden_fixtures.py tests/unit/test_evidence_artifact.py tests/unit/test_ingestion_limits.py tests/integration/test_cli_ingestion.py` | `d398926` | 200 passed in 92.48 s |
| Full suite | `python -m pytest -rfE --junitxml=junit.xml` | `f128ccd`; `d398926` is documentation-only | 4,350 passed, 46 skipped, 0 failed, 0 errors |
| Syntax | `python -m compileall -q aidatasetkit` | code matches `f128ccd` | pass |

No test file is deleted in the baseline delta. No new skip or xfail was added.
The mutation runner changes only isolated temporary copies; no mutation is
active in package code. All 13 mutations exited nonzero against their targeted
selectors on `2e9cf42`:

| Mutation | Targeted failing test selector |
|---|---|
| Wrong/default pandas delimiter | `test_ingestion.py` + `test_cli_ingestion.py -k semicolon` |
| Delimiter chosen from extension | `test_ingestion.py -k tab_separated_file_named_csv` |
| Silent delimiter ambiguity | `test_ingestion.py -k ambiguous` |
| Duplicate-header acceptance | `test_ingestion.py -k duplicate_headers` |
| Raw decode error leakage | `test_ingestion.py -k wrong_encoding` |
| Drop ingestion metadata | `test_cli_ingestion.py -k "semicolon_file or input_section"` |
| Absolute path in artifact | `test_cli_ingestion.py -k absolute_path` |
| Publish before ingestion | `test_cli_ingestion.py -k "ambiguous_file_is_refused or structured_refusals"` |
| Bypass source-byte check | `test_ingestion_limits.py -k file_bytes` |
| Reject exact byte boundary | `test_ingestion_limits.py -k file_bytes` |
| Skip DataFrame cell guard | `test_ingestion_limits.py -k dataframe` |
| Skip record-count guard | `test_ingestion_limits.py -k records` |
| Make CLI defaults unlimited | `test_ingestion_limits.py -k cli_defaults` |

The refusal path is ordered before profiling and publication. The integration
test first publishes a valid run, then refuses an over-limit input and verifies
that the old manifest, `CURRENT`, and run directory set are unchanged.

## Performance evidence

The existing `scripts/benchmark_ingestion.py` was used with a seeded 10-column
CSV on Python 3.12.3, pandas 2.3.3 and NumPy 2.4.1. Median timings used one
warm-up and three measured runs; peak Python allocation used a separate
`tracemalloc` run.

| Rows | Delimiter | Size | `read_csv` | `load_table` | Ratio | Peak ratio |
|---:|:---:|---:|---:|---:|---:|---:|
| 10,000 | `,` | 0.8 MB | 0.028 s | 0.102 s | 3.69x | 1.00x |
| 10,000 | `;` | 0.8 MB | 0.039 s | 0.135 s | 3.47x | 1.00x |
| 100,000 | `,` | 7.9 MB | 0.233 s | 0.658 s | 2.82x | 1.00x |
| 100,000 | `;` | 7.9 MB | 0.170 s | 0.474 s | 2.79x | 1.00x |

The parser-native field-length guard reduced the earlier 100k-row complete-load
measurement from approximately 3.8--5.8x to approximately 2.8x. The benchmark
is evidence, not a CI timing threshold. Accepted input remains whole-table
pandas ingestion, so this work does not claim chunked scalability.

Current rerun, Python 3.12.3 / pandas 2.3.3 / NumPy 2.4.1 / Windows, seeded
10-column CSV, one warmup and three measured benchmark runs:

| Measurement | Result |
|---|---|
| `BENCHMARK_10K` comma | 0.059 s `load_table` / 0.022 s `read_csv` = 2.72x; peak 1.00x |
| `BENCHMARK_10K` semicolon | 0.070 s / 0.020 s = 3.57x; peak 1.00x |
| `BENCHMARK_100K` comma | 0.304 s / 0.103 s = 2.95x; peak 1.00x |
| `BENCHMARK_100K` semicolon | 0.571 s / 0.183 s = 3.12x; peak 1.00x |

Additional Windows measurements use a deterministic 100k × 10 table; medians
over five calls unless noted:

| Measurement | Result | Qualification |
|---|---:|---|
| File preflight `Path.stat()` | 9.3 μs | metadata check only, not a full disk-read measurement |
| DataFrame `load_table(frame)` | 0.405 ms | includes shape checks and memory metadata; same object, no deep copy |
| Records `load_table(records)` | 0.597 s | includes validation and DataFrame materialization |
| Early row refusal | 9.66 ms | `max_rows=1`; stops before later malformed content |
| Late row refusal | 0.426 s | `max_rows=99,999`; consumes through final over-limit row |
| Peak Python allocation change | 0% (`1.00x`) | `tracemalloc`; does not measure native allocations/process RSS |

The committed early-stop test puts an unterminated quoted record after the
first over-limit row and observes `RowLimitError`, proving the validation loop
does not consume that remainder. File byte refusal occurs before delimiter
planning and `pandas.read_csv`. The performance guard is evidence plus this
control-flow test, not a flaky wall-clock CI assertion. Complete 100k load
overhead is about 3x versus bare pandas; measured Python peak allocation is
unchanged. It is an explicit validation-time cost, not an unexplained memory
regression.

## Packaging and clean-install smoke

The committed authoritative command was run under Git for Windows Bash with
Python 3.14.2 on `PATH`:
`./scripts/release_smoke_test.sh`. Start `2026-09-29T13:17:21Z`, end
`2026-09-29T13:27:38Z`, exit `0`. It freshly built wheel and sdist, ran `twine
check` on both, installed each artifact into separate clean environments with
dependencies, checked import/version/CLI help, and audited a synthetic dataset.
Nothing was published.

| Artifact | File | SHA-256 | Clean installation | Resource-specific smoke |
|---|---|---|---|---|
| Wheel | `aidatasetkit-0.1.0a1-py3-none-any.whl` | `BF8E7437DD374F1DEC6DB5D3C2E4224DA583137736E0D91820D5AED0B70CFD29` | pass, separate fresh Python 3.12 environment | pass: valid publication (exit 3), then `RowLimitError` (exit 1); no traceback/path/value leak; `CURRENT` unchanged and no partial run |
| Sdist | `aidatasetkit-0.1.0a1.tar.gz` | `1E9F637BBFEC9A1CFED9A33D0EB9654461E5ECB199EA8A37E61EF1C1C05AD870` | pass, separate fresh Python 3.12 environment | pass: same assertions as wheel |

Both resource smoke environments asserted package version `0.1.0a1`, artifact
schema `1.1`, publication schema `1.0`, public `IngestionLimits`, and all five
CLI flags. Twine passed for wheel and sdist. The first ad-hoc smoke attempts
had harness quoting/stderr mistakes and are not counted; the authoritative
script and final functional assertions completed successfully.

## Artifact and rollback contract

No artifact shape, `ARTIFACT_SCHEMA_VERSION`, `PUBLICATION_SCHEMA_VERSION`, or
fingerprint contract changed in G1-W2. The resource policy is enforced before an
artifact can be built. A refusal creates no new publication and cannot replace
an existing `CURRENT` pointer.

Against baseline `79b66da`, there are no changed paths under
`aidatasetkit/evidence`, `examples/audit_churn`, or golden fixtures, and no
changes to configuration, fingerprint implementation, or serialized evidence
types. Current constants remain artifact schema `1.1` and publication schema
`1.0`. The focused migration/golden/artifact tests passed in the combined
`200 passed` run. Semantic fingerprint changed: **NO**. Golden semantic
difference: **NONE**.

## Capability accounting

- G1-26 Resource limits: implementation and local evidence are complete; the
  authoritative row remains `PARTIAL` until remote branch/final/main CI.
- G1-25 Chunking/streaming: remains `MISSING`; validation streaming is not
  chunked profiling.
- G1-13 records: remains `PARTIAL`; generators remain refused.
- G1-17 encodings: remains `PARTIAL`; automatic detection remains out of scope.
- G1-22 date inference: remains `PARTIAL`; no text-date inference was started.
- Certified accounting stays at G1 `10/26 = 38.46%`, overall
  `102/232 = 43.97%` until all remote CI and main-integration gates pass.

Complete G0-G12 progress table, retaining pre-G1-W2 certified counts until
remote closure:

| Gate | Proven | Applicable | Certified % | State |
|---|---:|---:|---:|---|
| G0 | 16 | 16 | 100.00% | PASS |
| G0.1 | — | — | — | PASS, outside G0-G12 denominator |
| G1 | 10 | 26 | 38.46% | G1-W2 local evidence complete; CI pending |
| G2 | 15 | 22 | 68.18% | AUDIT |
| G3 | 2 | 17 | 11.76% | AUDIT |
| G4 | 9 | 13 | 69.23% | AUDIT |
| G5 | 0 | 15 | 0.00% | AUDIT |
| G6 | 2 | 12 | 16.67% | AUDIT |
| G7 | 0 | 13 | 0.00% | AUDIT |
| G8 | 15 | 23 | 65.22% | AUDIT |
| G9 | 8 | 12 | 66.67% | AUDIT |
| G10 | 18 | 28 | 64.29% | AUDIT |
| G11 | 5 | 21 | 23.81% | AUDIT |
| G12 | 2 | 14 | 14.29% | AUDIT |
| **All G0-G12** | **102** | **232** | **43.97%** | **CI closure pending** |

Affected row: G1-26 Resource limits, previous `MISSING`, interim `PARTIAL`;
implementation and local behavior are present, but row promotion is deferred
until branch, final evidence commit, and main CI pass. G1-25 remains `MISSING`;
G1-13, G1-17, G1-22 remain `PARTIAL`. P0-1 and P0-2 remain `CLOSED`.

## Remote CI and integration

`gh auth status` reports that no GitHub host is logged in. No remote G1-W2
branch ref or CI run has been verified yet. The fields below are intentionally
not claimed as passing:

```text
BRANCH_CI_RUN = NOT_RUN
BRANCH_CI_SHA = NOT_RUN
BRANCH_CI_JOBS = 0/10
BRANCH_CI_STATUS = BLOCKED_PENDING_PUSH_AND_CI_ACCESS
FINAL_BRANCH_CI_RUN = NOT_RUN
FINAL_BRANCH_CI_SHA = NOT_RUN
FINAL_BRANCH_CI_JOBS = 0/10
FINAL_BRANCH_CI_STATUS = BLOCKED
MAIN_CI_RUN = NOT_RUN
MAIN_CI_SHA = NOT_RUN
MAIN_CI_JOBS = 0/10
MAIN_CI_STATUS = BLOCKED
FINAL_MAIN_SHA = NOT_AVAILABLE
```

`origin/main` was fetched and is still exactly
`79b66dad2863bd97cbc641d0353f6d65ad9ec839`; local branch is six commits ahead
with no divergence from that baseline. No main integration or push has occurred.

G1-W3 was not started. No new formats, networking, databases, model-training,
visualization, Masari or MWIE work was performed.
