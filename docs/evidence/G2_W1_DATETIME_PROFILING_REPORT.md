# G2-W1 Datetime Profiling Report

```text
WAVE                         = G2-W1
TARGET_CAPABILITY            = G2-16
IMPLEMENTATION_STATUS        = IMPLEMENTED_PENDING_CTO_REVIEW
CERTIFIED_STATUS             = MISSING_PENDING_CERTIFICATION
STARTING_HEAD                = 92da30368775489ecf5e0d059148d69b0bd49e13
BRANCH                       = g2-w1-datetime-profiling
ARTIFACT_SCHEMA_BEFORE       = 1.4
PUBLICATION_SCHEMA           = 1.0 (unchanged)
PACKAGE_VERSION              = 0.1.0a1 (unchanged)
REMOTE_ACTIONS               = BRANCH_PUSH_AND_BRANCH_CI_ONLY
```

## Authorized scope

This wave adds bounded datetime profiling and fixed-duration temporal-gap
analysis for native pandas datetime columns. It does not parse object or string
columns, convert source values, extend chunked profiling, infer calendar
frequencies, resample, interpolate, normalize dates, forecast, or expand facade
inputs. G1-W8 strict ISO text recognition remains a separate evidence-only
feature with unchanged recognition and conversion behavior.

## DatetimeSummary contract

### Eligible dtypes

- A column is eligible only when pandas classifies its dtype with
  `is_datetime64_any_dtype` and its storage is a native pandas `DatetimeArray`.
  Unsupported datetime extensions, including Arrow timestamps, are not
  summarized. Their pandas kind classification can differ across pandas majors.
- Naive `datetime64[s]`, `datetime64[ms]`, `datetime64[us]`, and
  `datetime64[ns]` dtypes are supported.
- Timezone-aware pandas datetime dtypes are supported. Their physical unit is
  recorded from the dtype and their timezone identity is recorded separately.
- Object, string, categorical, period, date-object, mixed-timezone object, and
  timedelta columns are not eligible and are not parsed or coerced.

### Counts and range

- `total_count` is the number of rows in the column.
- `valid_count` is the number of non-`NaT` rows.
- `missing_count` is the number of `NaT` rows.
- The invariant is `total_count == valid_count + missing_count`.
- `minimum` and `maximum` are the earliest and latest valid timestamps.
  They are `null` when no valid timestamp exists.
- Duplicate timestamp count is `valid_count - distinct_valid_count`: for each
  valid timestamp, the first occurrence is not a duplicate and every additional
  occurrence is. `NaT` values are excluded.

### Timestamp and timezone serialization

- Naive timestamps serialize as deterministic ISO 8601 text without an offset.
- A timezone-aware minimum or maximum serializes in the source dtype's timezone,
  including the UTC offset applicable at that instant. The summary also records
  `timezone` as pandas' stable text identity for the dtype (for example
  `Europe/London`, `UTC`, or a fixed-offset identity). Naive columns record
  `timezone: null`.
- `resolution` records the physical pandas datetime unit (`s`, `ms`, `us`, or
  `ns`). Serialization preserves all precision represented by the timestamp and
  does not convert the source column.
- Internal comparisons may use integer timeline coordinates. For aware dtypes
  those coordinates denote UTC instants, while published endpoints remain in
  the source timezone. This makes DST elapsed-time spacing deterministic without
  erasing timezone meaning.

## Observation-spacing contract

Frequency means observed spacing, not proof of a collection schedule.
Calculations use sorted distinct valid timestamps, so input order and duplicate
rows do not change spacing:

- `distinct_valid_count` is the number of distinct valid timestamps.
- `interval_count` is `max(distinct_valid_count - 1, 0)`.
- `minimum_spacing` and `maximum_spacing` are bounded ISO 8601 fixed-duration
  strings, or `null` when there is no interval.
- `distinct_spacing_count` is the number of different consecutive interval
  durations and is zero when there is no interval.
- `is_regular` is `true` only when at least one interval exists and every
  consecutive interval is exactly equal. It is `null` for zero or one distinct
  timestamp.
- `observed_cadence` is the exact interval only when `is_regular` is true.
  Otherwise it is `null`. No mode, median, or most-common interval becomes an
  inferred schedule.
- Zero distinct timestamps: no range, no intervals, regularity unknown, cadence
  unknown.
- One distinct timestamp: minimum equals maximum, no intervals, regularity
  unknown, cadence unknown.
- Two distinct timestamps: one interval, exact regularity is true, and that
  sole observed interval is reported as the observed cadence. This remains an
  observation, not an expected schedule.
- Irregular observations are represented by `is_regular: false`,
  `distinct_spacing_count > 1`, and the bounded minimum/maximum spacing. No raw
  timestamp or interval list is emitted.

## Expected cadence and gaps

`KitConfig.expected_datetime_cadence` is the minimal configuration extension.
It is `null` by default. A non-null value applies to every native datetime
column profiled in that run and must be a positive, finite, fixed-duration
string accepted by pandas. Calendar durations or offsets, zero, negatives,
booleans, numbers, and unparseable values are rejected with
`ConfigurationError`. The validated cadence is serialized canonically as an
ISO 8601 duration and participates in the config fingerprint.

The temporal grid is anchored at the column's minimum valid timestamp. The
summary makes no claim about expected observations before the minimum or after
the maximum.

- A distinct timestamp is on-grid only when its elapsed duration from the
  anchor is an exact integer multiple of the expected cadence.
- `off_grid_observation_count` counts distinct valid timestamps that are not
  exact multiples. Duplicate rows do not increase it.
- `missing_expected_observation_count` counts unoccupied grid positions across
  the complete closed interval from the minimum through maximum observed instant.
  It is a scalar arithmetic result; missing timestamps are never materialized.
- `gap_count` counts contiguous runs of unoccupied grid positions, including a
  trailing run when the maximum observation is off-grid.
- On-grid observations occupy their exact cadence positions. Off-grid
  observations do not fill a grid position, but the maximum off-grid observation
  still extends the closed window used to count expected positions.
- Duplicate timestamps, off-grid observations, and missing expected
  observations remain three independent counts.
- Without an expected cadence, all gap fields are `null`; irregular spacing is
  descriptive evidence only and is not classified as defective.

## Range and overflow behavior

Spacing arithmetic uses the dtype's signed 64-bit integer unit rather than
coercing all values to nanoseconds. This supports the full pandas range for each
eligible resolution and avoids overflow for valid second-, millisecond-, or
microsecond-resolution timestamps outside the nanosecond range.

Expected cadence is converted exactly into the column's unit. If it has finer
precision than that unit, cannot be represented as a positive signed 64-bit
unit count, or would make subtraction/counting overflow, profiling raises an
explicit domain/configuration error rather than rounding, wrapping, or reporting
success-shaped values.

## Evidence, schema, fingerprints, and reports

- `ColumnProfile.datetime` holds `DatetimeSummary`; numeric and datetime
  summaries are separate.
- `ColumnEvidence.datetime` publishes the bounded summary. It contains no raw
  lists, row samples, or per-gap timestamps.
- The additive serialized field changes the artifact shape, so artifact schema
  advances from `1.4` to `1.5` through the existing migration chain.
- Publication schema remains `1.0`; publication layout and manifest semantics
  do not change.
- Package version remains `0.1.0a1`; no repository rule requires a package
  version change for this additive development wave.
- Dataset and schema fingerprint algorithms are unchanged because source values,
  dtypes, row order, and column order are unchanged.
- Semantic artifact fingerprints change because schema `1.5` and datetime
  evidence are part of the semantic record.
- Config fingerprints change when an explicit cadence is supplied. Existing
  artifacts that do not record a `KitConfig` retain `thresholds: null`; artifacts
  that record a full `KitConfig` include the new default-null setting and move
  according to the existing fingerprint migration contract.
- The HTML report renders bounded datetime evidence from the canonical artifact
  mapping and does not recompute it.

## Resource, privacy, and immutability guarantees

- Profiling reads but never mutates the caller's DataFrame, Series, index,
  timezone, dtype, values, or row order.
- For a datetime column with `n` rows and `u` distinct valid timestamps, the
  current implementation has worst-case `O(n log n)` time due to distinct-value
  sorting and spacing analysis, with `O(n)` worst-case auxiliary memory for
  filtering, sorting, and intermediate delta structures. Published output is
  `O(1)` per datetime column.
- File-backed full-table routes retain their existing ingestion limits. Direct
  in-memory DataFrames do not gain a new datetime-specific size limit. This is
  not streaming datetime profiling.
- No raw timestamp lists, row samples, unbounded gap arrays, or missing-timestamp
  materialization are permitted.
- Minimum and maximum timestamps are aggregate evidence intentionally required
  by G2-16. Endpoint aggregates may still be sensitive. No other source
  timestamp is published.

## Closure boundary

Local implementation and validation can mark G2-W1 only as
`IMPLEMENTED_PENDING_CTO_REVIEW`. G2-16 retains its current certified status and
certified progress remains G1 `22/26 = 84.62%`, overall
`114/232 = 49.14%`, and G2 `15/22 = 68.18%`.

### Historical implementation validation

The following results belong to the reviewed implementation preceding the
mutation-closure repair, not to certification of the expanded matrix.

- Focused datetime contract tests: pandas 3, 56 passed and one optional
  PyArrow-extension test skipped because PyArrow is not installed; pandas 2,
  57 passed.
- Focused evidence, fingerprint, date-inference, chunked, and golden regressions:
  pandas 3, 352 passed; pandas 2, 349 passed and three optional-dependency tests
  skipped.
- Full reference suite: 4,864 passed, 47 skipped.
- Full minimum suite: 4,738 passed, 61 skipped.
- Datetime mutation gate: all 20 semantic mutations were killed by their
  selected tests; no mutant survived and every scratch tree was restored.
- Release smoke test: wheel and source distribution built, passed metadata
  validation, installed into clean environments, and passed CLI/audit checks.
- Pylance diagnostics on changed Python sources and tests: no errors.
- Example verdict remained `blocked` before and after the schema migration.
  Dataset fingerprints and verdict logic are unchanged; the semantic
  fingerprint moved as expected for schema `1.5` and its added config/evidence
  fields.

These are local results, not certification. The reviewed candidate
`769e9ea041d3d66702f18c4dac451e25f1b776fb` subsequently passed branch CI run
38691797127, 10/10. That run does not validate the mutation-closure candidate.

### CTO mutation-closure repair

The repair preserves M-G2W1-01 through M-G2W1-20 unchanged and adds 17 dedicated
mutants. All 38 required IDs are covered by 37 executable mutants:
**M-G2W1-27 maps exactly to M-G2W1-10**. M10 removes the regularity guard from
observed cadence and publishes the first interval of irregular observations;
its proving test rejects that invented cadence. This is the identical
weakening requested by M27, not a substitution of a passing baseline test.

| Required ID | Dedicated weakening / semantic kill |
|---|---|
| M-G2W1-21 | Missing `NaT` cadence accepted / `missing cadence acceptance` |
| M-G2W1-22 | `1 month` silently normalized to 30 days / `non-fixed cadence acceptance` |
| M-G2W1-23 | Native eligibility guard removed / unsupported Arrow storage must return no summary, not crash |
| M-G2W1-24 | Source datetime array reversed in place / values, index, dtype, order, timezone, and retained source identities checked |
| M-G2W1-25 | Sorted duplicates replace unique coordinates / duplicate spacing coordinates rejected |
| M-G2W1-26 | NaT coordinate exclusion removed / NaT participation or resulting overflow rejected |
| M-G2W1-27 | Exact mapping to preserved M-G2W1-10, described above |
| M-G2W1-28 | Observed two-point interval promoted to expected cadence / no expected schedule or gap classification without caller cadence |
| M-G2W1-29 | Gap calculation receives repeated coordinates / distinct off-grid count remains one |
| M-G2W1-30 | Missing positions replaced with gap count / three missing positions in one contiguous gap remain `(3, 1)` |
| M-G2W1-31 | Canonical temporal evidence emptied / endpoints, timezone, unit, observed and expected cadence required |
| M-G2W1-32 | Temporal missing count enters verdict policy / captured verdict inputs and decision evidence must be identical |
| M-G2W1-33 | Dataclass field order displaces positional source / `EvidenceSource.QUALITY` must remain the source argument |
| M-G2W1-34 | Duplicate-count invariant disabled / malformed evidence must raise `EvidenceError` |
| M-G2W1-35 | Chunked scan calls native summarizer / chunked-isolation sentinel must fail |
| M-G2W1-36 | Actual `pd.date_range` materialization inserted / call guard fails before allocation, independent of memory or timing |
| M-G2W1-37 | Artifact schema constant reverted to 1.4 / exact schema and reversible 1.4-to-1.5 fingerprint-chain proof |
| M-G2W1-38 | Exported pandas-2 fixture reverted to 1.4 / pandas-major schema, full default config, config digest, temporal field shape, and verdict contract |

The stable non-fixed input `1 month` is rejected by both pandas 2.1.4 and
pandas 3.0.5. The unsupported-extension witness calls the summarizer directly
as well as the full profiler, so pandas-major kind differences cannot hide the
eligibility mutation. PyArrow is installed for both local closure runs.

Harness rules now additionally expose `baseline_exit`, `baseline_skipped`, the
required-ID mapping, and final `tree_restored` in the JSON record. A skipped
witness cannot count as a baseline pass or a kill. Expected testcase names are
matched exactly (allowing only that test's parametrizations), every failure
must carry its dedicated semantic reason, and unrelated failures or test
errors are rejected. Mutation edits are restored in `finally`, with full
exported-file SHA-256 manifests compared after each mutant.

Initial expanded-matrix evidence on committed test/harness revision
`7038c3d88b7cea22a673af39d575b9cfd04dea13`:

- Python 3.12.14 / pandas 3.0.5: baseline exit 0, 37 killed, 0 survived,
  0 harness errors, 0 test-environment errors, all trees restored.
- Python 3.11.17 / pandas 2.1.4: baseline exit 0, 37 killed, 0 survived,
  0 harness errors, 0 test-environment errors, all trees restored.
- Focused datetime, closure, migration and current-example contracts: 116 passed
  on each pandas major, no skips.
- Full reference suite after closure: 4,882 passed, 47 skipped, 787.43 seconds.
- Full minimum suite after closure: 4,782 passed, 56 skipped, 465.10 seconds.
  Existing platform, declared model-capability, and unavailable optional-extra
  skips remain visible in JUnit; no tests were disabled or weakened. In
  particular the Arrow mutation witness ran, without a skip, in both environments.
- Packaging/build smoke is not re-triggered by this repair: no packaging,
  release script, dependency manifest, or production file changed. The final
  branch CI still includes both packaging jobs.
- No production code, example or fixture changes; no regenerated artifacts.
  Artifact schema 1.5, publication schema 1.0, package 0.1.0a1, fingerprints,
  approved semantics, and the Golden verdict are unchanged by this repair.

The complete matrix must also run against the final committed branch tip,
using `scripts/g2_w1_mutations.py <new-scratch-dir> --rev <exact-tip>`.
Only exported committed revisions may be mutated; the live worktree is never
a mutation target. Final exact-tip branch CI (10/10) and CTO review remain
separate gates, not implied by these local results.

Remaining certification requirements:

1. Branch CI on the exact candidate code SHA.
2. Windows and Linux matrix success, expected 10/10.
3. CTO review.
4. Main integration and main CI on the certified code SHA.
5. Separate documentation seal.
