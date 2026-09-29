# G1-W2 Resource Governance and Large-Input Safety

```text
G1_W2_AUTHORIZATION = GO
SCOPE = resource governance and large-input safety only
BRANCH = g1-w2-resource-governance
BASELINE_SHA = 79b66dad2863bd97cbc641d0353f6d65ad9ec839
FIRST_CODE_SHA = d2712fb7ccb197c4e88805c38e4471fb8ca9b903
CURRENT_TESTED_SHA = f128ccd6b63507e0ba4baa2dcc0526360fc31e0d
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

## Artifact and rollback contract

No artifact shape, `ARTIFACT_SCHEMA_VERSION`, `PUBLICATION_SCHEMA_VERSION`, or
fingerprint contract changed in G1-W2. The resource policy is enforced before an
artifact can be built. A refusal creates no new publication and cannot replace
an existing `CURRENT` pointer.

## Capability accounting

- G1-26 Resource limits: `SUPPORTED_AND_TESTED` after final evidence sign-off.
- G1-25 Chunking/streaming: remains `MISSING`; validation streaming is not
  chunked profiling.
- G1-13 records: remains `PARTIAL`; generators remain refused.
- G1-17 encodings: remains `PARTIAL`; automatic detection remains out of scope.
- G1-22 date inference: remains `PARTIAL`; no text-date inference was started.
- Final accounting for this wave is G1 `11/26 = 42.31%`, overall
  `103/232 = 44.40%`.

G1-W3 was not started. No new formats, networking, databases, model-training,
visualization, Masari or MWIE work was performed.
