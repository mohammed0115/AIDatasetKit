# G1-W2 Resource Governance and Large-Input Safety

```text
G1_W2_AUTHORIZATION = GO
SCOPE = resource governance and large-input safety only
BRANCH = g1-w2-resource-governance
PRE_SHA = 79b66dad2863bd97cbc641d0353f6d65ad9ec839
BASELINE_SHA = 79b66dad2863bd97cbc641d0353f6d65ad9ec839
FIRST_CODE_SHA = d2712fb7ccb197c4e88805c38e4471fb8ca9b903
TESTED_SHA = f128ccd6b63507e0ba4baa2dcc0526360fc31e0d
LOCAL_TESTED_SHA = f128ccd6b63507e0ba4baa2dcc0526360fc31e0d
LOCAL_PACKAGING_ARTIFACT_SOURCE_SHA = d398926e87bed127b7dfb4971d971d7f6dc29ca5
PACKAGING_CI_TESTED_SHA = 5c1c966d9382b7b10a8ca722a8a8c5987cc6f2c0
FINAL_BRANCH_SHA = 5c1c966d9382b7b10a8ca722a8a8c5987cc6f2c0
FINAL_MAIN_SHA = 5c1c966d9382b7b10a8ca722a8a8c5987cc6f2c0
REPORT_CLOSURE_COMMIT = pending this docs-only commit
CURRENT_HEAD_BEFORE_CLOSURE = 5c1c966d9382b7b10a8ca722a8a8c5987cc6f2c0
ORIGIN_MAIN_PRE_INTEGRATION = 79b66dad2863bd97cbc641d0353f6d65ad9ec839
WORKTREE_CLEAN_BEFORE_REPORT_CLOSURE = YES
G1_W2_FINAL_GATE = PENDING_FINAL_REPORT_COMMIT_CI
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
| Focused ingestion units | `python -m pytest -q tests/unit/test_ingestion_limits.py tests/unit/test_ingestion.py` | `f128ccd` code; docs-only descendants | 124 passed |
| Focused CLI/facade | `python -m pytest -q tests/integration/test_cli_ingestion.py tests/integration/test_facade.py` | same code | 119 passed |
| Current artifact/safety | `python -m pytest -q tests/unit/test_capability_fingerprint_migration.py tests/unit/test_golden_fixtures.py tests/unit/test_evidence_artifact.py tests/unit/test_ingestion_limits.py tests/integration/test_cli_ingestion.py` | `d398926` | 200 passed in 92.48 s |
| Full suite | `python -m pytest -rfE --junitxml=junit.xml` | `f128ccd`; `d398926` is documentation-only | 4,350 passed, 46 skipped, 0 failed, 0 errors |
| Syntax | `python -m compileall -q aidatasetkit` | code matches `f128ccd` | pass |

Test environment: Windows, Python 3.12.3, pytest 8.3.3, NumPy 2.4.1,
pandas 2.3.3, SciPy 1.17.0 and scikit-learn 1.8.0. CI covers Python 3.11
minimum and Python 3.12 reference constraints, with and without extras, on
Ubuntu and Windows.

Full-suite timing record: the successful run completed in 829.93 s, exit 0.
The first attempt
was interrupted at 208 passed by a terminal input collision and is not counted.
The successful rerun collected 4,394 tests and ended with 4,350 passed, 46
skipped, zero failed and zero errors.

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

Additional Windows measurements use a deterministic 100k × 10 table.
`Path.stat()` is median over five calls; DataFrame, records, early refusal and
late refusal are medians over three calls:

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

The focused resource smoke command was run from each environment's `Scripts`
directory as follows, substituting the wheel or sdist artifact and a separate
fresh environment per format:

```text
<env>\Scripts\pip.exe install -q <artifact>
<env>\Scripts\python.exe -c "import aidatasetkit; assert ARTIFACT_SCHEMA_VERSION == '1.1'; assert PUBLICATION_SCHEMA_VERSION == '1.0'; assert IngestionLimits().max_rows == 1000000"
<env>\Scripts\aidatasetkit.exe audit --help
<env>\Scripts\aidatasetkit.exe audit input.csv --target Churn --output out
<env>\Scripts\aidatasetkit.exe audit input.csv --target Churn --output out --max-rows 1
```

The first audit published successfully (exit 3 on this fixture). The last
returned `RowLimitError` with exit 1, no traceback or path/cell leakage, no new
run, and byte-identical `CURRENT`.
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

- G1-26 Resource limits: `SUPPORTED_AND_TESTED`; finite limits are enforced
  across file, DataFrame, records, CLI and facade paths with focused tests,
  mutation evidence, clean-install smoke, branch CI and main CI.
- G1-25 Chunking/streaming: remains `MISSING`; validation streaming is not
  chunked profiling.
- G1-13 records: remains `PARTIAL`; generators remain refused.
- G1-17 encodings: remains `PARTIAL`; automatic detection remains out of scope.
- G1-22 date inference: remains `PARTIAL`; no text-date inference was started.
- Certified accounting after G1-W2 closure is G1 `11/26 = 42.31%`, overall
  `103/232 = 44.40%`.

Complete G0-G12 progress table, retaining pre-G1-W2 certified counts until
remote closure:

| Gate | Proven | Applicable | Certified % | State |
|---|---:|---:|---:|---|
| G0 | 16 | 16 | 100.00% | PASS |
| G0.1 | — | — | — | PASS, outside G0-G12 denominator |
| G1 | 11 | 26 | 42.31% | G1-W2 PASS |
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
| **All G0-G12** | **103** | **232** | **44.40%** | **G1-W2 closed** |

Affected row: G1-26 Resource limits, previous `MISSING`, new
`SUPPORTED_AND_TESTED`; evidence is this implementation, tests, mutation
results, benchmarks, clean wheel/sdist smoke, branch/final/main CI. G1-25
remains `MISSING`; G1-13, G1-17, G1-22 remain `PARTIAL`. P0-1 and P0-2 remain
`CLOSED`.

## Remote CI and integration

`gh auth status` reports that no GitHub host is logged in. GitKraken normal
pushes published the branch and main; public Actions API evidence verified each
run and SHA. Branch run `36577314053` passed on `f8d01fd` (10/10, 0 failures,
0 errors, 5m35s). Final evidence run `36578258496` passed on `5c1c966` (10/10,
0 failures, 0 errors, 6m11s). Main run `36679042688` passed on `5c1c966` (10/10,
0 failures, 0 errors; completed at `2026-09-30T06:42:28Z`).

```text
BRANCH_CI_RUN = 36577314053
BRANCH_CI_SHA = f8d01fd2ff33508de995176242c0014d22258eef
BRANCH_CI_JOBS = 10/10
BRANCH_CI_STATUS = PASS
FINAL_BRANCH_CI_RUN = 36578258496
FINAL_BRANCH_CI_SHA = 5c1c966d9382b7b10a8ca722a8a8c5987cc6f2c0
FINAL_BRANCH_CI_JOBS = 10/10
FINAL_BRANCH_CI_STATUS = PASS
MAIN_CI_RUN = 36679042688
MAIN_CI_SHA = 5c1c966d9382b7b10a8ca722a8a8c5987cc6f2c0
MAIN_CI_JOBS = 10/10
MAIN_CI_STATUS = PASS
FINAL_MAIN_SHA = 5c1c966d9382b7b10a8ca722a8a8c5987cc6f2c0
```

| Gate | Run | SHA | Jobs | Failures/errors | Result | Start (UTC) | Completion (UTC) | Duration |
|---|---:|---|---:|---:|---|---|---|---:|
| Branch CI | 36577314053 | `f8d01fd2ff33508de995176242c0014d22258eef` | 10/10 | 0/0 | PASS | 2026-09-29 13:44:16 | 2026-09-29 13:49:51 | 5m35s |
| Final branch CI | 36578258496 | `5c1c966d9382b7b10a8ca722a8a8c5987cc6f2c0` | 10/10 | 0/0 | PASS | 2026-09-29 13:51:45 | 2026-09-29 13:57:56 | 6m11s |
| Main CI | 36679042688 | `5c1c966d9382b7b10a8ca722a8a8c5987cc6f2c0` | 10/10 | 0/0 | PASS | 2026-09-30 06:35:45 | 2026-09-30 06:41:30 | 5m45s |

`origin/main` was fetched and is still exactly
`79b66dad2863bd97cbc641d0353f6d65ad9ec839` immediately before integration.
Local `main` was fast-forwarded to `5c1c966`, pushed normally without a merge
commit, and its CI run passed. No force-push, tag, release or PyPI publication
occurred. G1-W3 was not started.

The final report-closure commit will be a documentation-only descendant of
`5c1c966`; it will receive a fresh branch workflow and fresh main workflow
before becoming the final repository head. The completed run records above
apply to the pre-closure code/evidence SHA `5c1c966`.

## Closure fields

```text
RESOURCE_POLICY_AUTHORITY = aidatasetkit.ingestion.types.IngestionLimits
DEFAULT_MAX_BYTES = 67108864
DEFAULT_MAX_ROWS = 1000000
DEFAULT_MAX_COLUMNS = 1000
DEFAULT_MAX_CELLS = 10000000
DEFAULT_MAX_FIELD_LENGTH = 1000000
DEFAULT_MAX_RECORDS = 1000000
DEFAULT_MAX_KEYS_PER_RECORD = 1000
DEFAULT_MAX_RECORD_CHARS = 10000000
FILE_PREFLIGHT = stat size check before plan_delimited and pandas.read_csv
CSV_TSV_GUARDS = row/column/cell/field checks in strict streaming validation
DATAFRAME_GUARDS = rows/columns/cells before memory metadata; identity retained
RECORD_GUARDS = count/keys/chars/columns/cells before DataFrame construction
EARLY_TERMINATION = RowLimitError before a later malformed record is consumed
NO_PARTIAL_PUBLICATION = YES
PREVIOUS_CURRENT_PRESERVED = YES
ARTIFACT_SCHEMA = 1.1
PUBLICATION_SCHEMA = 1.0
SEMANTIC_FINGERPRINT_CHANGED = NO
GOLDEN_SEMANTIC_DIFF = NONE
FOCUSED_TESTS = 124 ingestion unit tests; 119 CLI/facade integration tests;
                 200 artifact/fingerprint/ingestion/CLI tests
FULL_TESTS = 4350 passed, 46 skipped, 0 failed, 0 errors
MUTATIONS = 13 total, 13 killed, 0 survived
WHEEL_INSTALL = PASS; WHEEL_SMOKE = PASS; WHEEL_SHA256 = BF8E7437DD374F1DEC6DB5D3C2E4224DA583137736E0D91820D5AED0B70CFD29
SDIST_INSTALL = PASS; SDIST_SMOKE = PASS; SDIST_SHA256 = 1E9F637BBFEC9A1CFED9A33D0EB9654461E5ECB199EA8A37E61EF1C1C05AD870
TWINE_CHECK = PASS (wheel and sdist)
G1_CERTIFIED_PROGRESS = 11/26 = 42.31%
OVERALL_CERTIFIED_PROGRESS = 103/232 = 44.40%
P0_1_STATUS = CLOSED
P0_2_STATUS = CLOSED
G1_W3_STARTED = NO
```

Affected capability row:

| Row | Capability | Previous | New | Evidence | Why this classification |
|---|---|---|---|---|---|
| G1-26 | Resource limits (source bytes, rows, columns) | `MISSING` | `SUPPORTED_AND_TESTED` | policy and guards; focused tests; 13/13 mutations; wheel/sdist smoke; branch/final/main CI 10/10 | All declared supported source kinds have finite defaults, structured refusal, and tests at the enforcement boundaries; CI and clean-install gates pass. |

Remaining G1 gaps: G1-25 true chunked profiling remains `MISSING`; G1-13 generators remain unsupported (`PARTIAL`); G1-17 automatic encoding detection remains absent (`PARTIAL`); G1-22 text-date inference remains absent (`PARTIAL`). No G1-W3 work was started.

G1-W3 was not started. No new formats, networking, databases, model-training,
visualization, Masari or MWIE work was performed.
