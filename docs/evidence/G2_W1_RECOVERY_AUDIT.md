# G2-W1 Read-Only Recovery Audit

**Audit date:** 2026-10-15
**Authorization:** CTO authorized read-only recovery audit only.

## RECOVERY_GATE

**Audit completed. Recovery decision: `RECOVERABLE_WITH_REPAIR`.**

The interrupted work is coherently attributable to G2-W1, but it is **not ready for implementation-complete status or certification**. The principal blockers are the missing fingerprint migration, an outdated pandas-2 fixture, incomplete cadence validation, and compatibility/test gaps.

No project code, tests, fixture generators, formatters, or collection commands were executed. Read-only Git inspection, source review, and JSON structural comparisons were used. `git diff --check` reported no whitespace errors.

## BASELINE_STATUS

| Reference | Recorded SHA |
|---|---|
| Certified baseline | `92da30368775489ecf5e0d059148d69b0bd49e13` |
| HEAD | `92da30368775489ecf5e0d059148d69b0bd49e13` |
| Local main | `92da30368775489ecf5e0d059148d69b0bd49e13` |
| Local origin/main reference | `92da30368775489ecf5e0d059148d69b0bd49e13` |

**Commit baseline matches. Worktree remains dirty and preserved.** No fetch was performed during this recovery audit; the remote-tracking reference above is the locally recorded reference.

No applicable `AGENTS.md` was found in the repository or its ancestor directories. [CONTRIBUTING.md](../../CONTRIBUTING.md) was reviewed.

## CURRENT_BRANCH

`g2-w1-datetime-profiling`

- No configured upstream.
- No branch `remote`/`merge` tracking configuration.
- The VS Code merge-base setting references `origin/main`; that is **not** an upstream.
- Local main tracks origin/main without recorded divergence.
- No candidate implementation commit exists: all implementation changes remain uncommitted.

## WORKTREE_STATUS

**19 modified tracked files, 3 untracked files, no staged changes.**

The complete porcelain entries are represented in the file-by-file table below. Initial and final status matched.

## CHANGE_PROVENANCE

Relevant reflog entries:

| Time | Event |
|---|---|
| `2026-10-14T12:36:39+03:00` | Branch created from HEAD at `92da303` |
| `2026-10-14T12:36:39+03:00` | HEAD moved from main to `g2-w1-datetime-profiling` |
| `2026-10-13T18:54:35+03:00` | Main fast-forwarded to the certified baseline |

The untracked report's modification time precedes those of the new source and test files, consistent with contract-first work. However, **filesystem times and reflogs cannot establish authorship or prove the original content sequence**. No implementation commits document intermediate steps.

A pre-existing pytest cache contains an older datetime boundary-test failure marker and 34 datetime collection entries. These are **stale interruption clues, not current test results**; the present source statically defines 23 test methods, expanding to 33 parameterized cases.

## AUTHORITATIVE_G2_W1_REQUIREMENT

The [roadmap's G2 section](../AIDATASETKIT_EXECUTION_ROADMAP.md) authorizes:

- `DatetimeSummary` and temporal-gap checking.
- Datetime minimum, maximum, frequency, and gaps.
- A test for every new field.
- A recorded migration in the fingerprint chain.
- A fixture for every supported pandas major.
- A stop for review if a new finding changes the Golden verdict.

The preceding implementation directive further restricts this to native pandas datetime columns, including timezone-aware dtypes, with:

- No object/string parsing or source mutation.
- Exact observed-spacing recognition, not invented schedules.
- Explicit validated fixed-duration cadence for gap analysis.
- Separate duplicate, missing-expected, and off-grid counts.
- Bounded published summaries.
- No chunked datetime extension or facade input expansion.

## CAPABILITY_IDS_AFFECTED

- **Direct target: G2-16 — Datetime range / frequency / gaps.**
- Its current certified classification remains **`MISSING`**, as recorded in the [capability matrix](POST_G0_CAPABILITY_REALITY_AUDIT.md).
- G1-22 strict ISO recognition is a compatibility boundary, not a new capability claim.
- Evidence/redaction and structured findings are affected integration surfaces, not additional authorized capability promotions.

Certified progress remains:

- G1: **22/26 = 84.62%**, still **IN PROGRESS**.
- G2: **15/22 = 68.18%**.
- Overall: **114/232 = 49.14%**.
- G1-W10 remains PASS.

## FILE_BY_FILE_CLASSIFICATION

` M` means modified but unstaged; `??` means untracked.

| Porcelain | File | Classification |
|---|---|---|
| ` M` | [core/__init__.py](../../aidatasetkit/core/__init__.py) | G2-W1: exports the new summary type. |
| ` M` | [core/config.py](../../aidatasetkit/core/config.py) | G2-W1: global optional expected cadence and validation. Requires repair. |
| ` M` | [core/types.py](../../aidatasetkit/core/types.py) | G2-W1: 19-field summary and nullable profile field. |
| ` M` | [evidence/builder.py](../../aidatasetkit/evidence/builder.py) | G2-W1: copies datetime measurements into evidence. |
| ` M` | [evidence/report.py](../../aidatasetkit/evidence/report.py) | G2-W1: datetime HTML table. Incomplete explanatory context. |
| ` M` | [evidence/types.py](../../aidatasetkit/evidence/types.py) | G2-W1: schema 1.5 and nullable evidence field. Compatibility concern. |
| ` M` | [profiling/profiler.py](../../aidatasetkit/profiling/profiler.py) | G2-W1: invokes native datetime summarization. |
| ` M` | [audit.json](../../examples/audit_churn/audit.json) | Generated current example: additive shape/config/fingerprint changes. |
| ` M` | [expected_audit_semantic.json](../../examples/audit_churn/expected_audit_semantic.json) | Golden semantic fixture update for pandas 3 only. |
| ` M` | [lineage.json](../../examples/audit_churn/lineage.json) | Generated example: schema label only. |
| ` M` | [report.html](../../examples/audit_churn/report.html) | Generated example: identities and schema footer only. |
| ` M` | [test_chunked_profiling.py](../../tests/unit/test_chunked_profiling.py) | Updates schema assertions only; no new datetime-isolation test. |
| ` M` | [test_current_example_contract.py](../../tests/unit/test_current_example_contract.py) | Updates live-schema assertion only. |
| ` M` | [test_date_inference.py](../../tests/unit/test_date_inference.py) | Updates schema assertions; old migration test/name now misleading. |
| ` M` | [test_evidence_artifact.py](../../tests/unit/test_evidence_artifact.py) | Updates schema assertions only. |
| ` M` | [test_ingestion_excel.py](../../tests/unit/test_ingestion_excel.py) | Updates schema assertions only; no reader change. |
| ` M` | [test_ingestion_sqlite.py](../../tests/unit/test_ingestion_sqlite.py) | Updates schema assertions only; no reader change. |
| ` M` | [test_packaging.py](../../tests/unit/test_packaging.py) | Updates artifact-schema assertion; package version unchanged. |
| ` M` | [test_record_iterables.py](../../tests/unit/test_record_iterables.py) | Updates schema assertions only; no input expansion. |
| `??` | [profiling/datetime.py](../../aidatasetkit/profiling/datetime.py) | New G2-W1 implementation. Preserve and repair. |
| `??` | [G2_W1_DATETIME_PROFILING_REPORT.md](G2_W1_DATETIME_PROFILING_REPORT.md) | Contract document; no completed validation record. |
| `??` | [test_datetime_profiling.py](../../tests/unit/test_datetime_profiling.py) | New focused tests; useful but incomplete. |

## IMPLEMENTATION_SUMMARY

The interrupted implementation follows the intended layer direction:

1. Core defines configuration and the summary type.
2. Profiling calculates temporal measurements.
3. Evidence copies those measurements without recomputing them.
4. HTML renders the canonical evidence.

Numeric summaries remain separate. No resampling, interpolation, forecasting, calendar-frequency inference, or input conversion was introduced.

**There is no new quality check or temporal `QualityIssue`.** Explicit-cadence gaps are currently measurements only, not structured findings.

## DATETIME_SEMANTIC_CONTRACT

The implemented behavior is:

| Area | Current behavior |
|---|---|
| Eligibility | Profiler dispatches every `ColumnKind.DATETIME` column to the new module. |
| Resolution | Reads array unit; accepts `s`, `ms`, `us`, `ns`. |
| Counts | Total rows, non-NaT rows, NaT rows, distinct valid timestamps. |
| Duplicates | `valid_count - distinct_valid_count`; NaT excluded. |
| Endpoints | Source-series minimum/maximum, serialized with timestamp `isoformat()`. |
| Timezone | Source dtype timezone's string identity; aware endpoints retain applicable offsets. |
| Internal coordinates | Integer physical-unit values; aware values represent UTC instants internally. |
| Ordering | Sorted distinct valid timestamps; input ordering and repeated rows do not affect spacing. |
| Frequency | Exact equality of consecutive elapsed intervals; no mode-based schedule. |
| Zero/one distinct value | No spacing; regularity and observed cadence unknown. |
| Two distinct values | One observed interval, marked regular; not an expected schedule. |
| Irregular series | Regularity false, observed cadence null, spacing extrema/count retained. |
| Expected cadence | One optional fixed duration applies to every native datetime column. |
| Grid | Anchored at minimum valid timestamp; exact integer multiples only. |
| Off-grid | Counts distinct off-grid timestamps, not repeated rows. |
| Gaps | Counts absent grid positions between consecutive **on-grid** timestamps. |
| No expected cadence | Missing-expected, gap, and off-grid fields are null. |
| Empty valid population with cadence | Those three counts become zero. |

Source review found no source-value assignments, timezone conversions, or in-place sorting.

**Important unresolved edge:** with observations at `00:00` and `01:30`, and hourly cadence, the implementation reports zero missing expected observations and one off-grid observation. The `01:00` grid point is not counted because there is no later on-grid observation. The owner must approve this narrower “between on-grid endpoints” contract or require counting throughout the observed minimum–maximum window.

## EVIDENCE_AND_SCHEMA_IMPACT

The bump from **1.4 to 1.5 is justified** by the additive per-column serialized field.

However, the migration is **not complete**:

- [The fingerprint migration chain](../../tests/unit/test_capability_fingerprint_migration.py) still ends at G1-W8/schema 1.4.
- It still expects semantic identity `60375e40…`.
- There is no rollback step removing the new column fields and cadence setting while restoring the previous identity.
- [The artifact contract documentation](../audit-artifact.md) still declares 1.4.
- [README schema documentation](../../README.md) still says 1.4.

Datetime evidence is a generic mapping passed through canonical serialization. There is **no new validation of temporal field types, count invariants, endpoint/regularity consistency, or cadence/null relationships**.

Publication schema remains **1.0**, and package version remains **0.1.0a1**. No inspected rule requires changing either.

## FINGERPRINT_IMPACT

Dataset/schema fingerprint algorithms are unchanged.

The JSON comparison confirmed that the changed example retains the same dataset, findings, plan, date inference, verdict, and verdict reasons.

| Identity | Before | After |
|---|---|---|
| Config | `03a79ae16ffa080b15e8f75ff8a0082ce968e4822d956c5d5658ced267d38dc6` | `f8a131e7b5b843345461e7c478fe5161bba2090b74250c2e8c5708f363ae005d` |
| Stored semantic | `60375e40773cb0849085a4867c31e609b90c34a60a2cb8842c9e34f37b3a237e` | `d889bc29c93faa0133d4bee392d9cff9e07c781bf54c2b720bc31fb57f6c7daf` |

The **default config identity changes even without datetime columns**, because recording a complete configuration now includes `expected_datetime_cadence: null`. The evidence report acknowledges this, but the required migration chain does not.

These are inspected stored values, not newly rebuilt artifact results.

## EXAMPLE_AND_GOLDEN_IMPACT

Read-only structural comparison established:

| Changed generated file | Exact reason for change |
|---|---|
| [audit.json](../../examples/audit_churn/audit.json) | Ten nullable datetime fields; default-null cadence setting; resulting config/semantic identities; schema 1.5. Timestamp and environment unchanged. |
| [expected_audit_semantic.json](../../examples/audit_churn/expected_audit_semantic.json) | Same semantic shape/config changes, without provenance. It matches the changed illustrative audit's semantic content. |
| [lineage.json](../../examples/audit_churn/lineage.json) | Only schema 1.4 → 1.5; lineage itself unchanged. |
| [report.html](../../examples/audit_churn/report.html) | Config/evidence fingerprint display and schema footer. No datetime section because the example has no native datetime column. |

**Golden verdict: `blocked` → `blocked`.** Findings and verdict reasons are unchanged. No evidence of verdict concealment was found.

But [the pandas-2 fixture](../../examples/audit_churn/expected_audit_semantic.pandas2.json) remains at schema 1.4, without the new fields/config entry. This statically contradicts the cross-major fixture contracts and predicts failures in fixture agreement and pandas-2 reproduction tests. Those failures were **not executed**.

The generated changes are consistent with a reference-environment refresh, but Git cannot prove which command produced them.

## DEFAULT_PATH_IMPACT

- Native datetime profiling now always sorts/deduplicates and calculates temporal evidence.
- Even with no expected cadence, unsupported ranges can now fail a previously simpler profiling path.
- All normal serialized column profiles/evidence gain a nullable field.
- Default configuration fingerprints move when the full configuration is recorded.
- No new temporal quality findings alter verdict policy.
- Non-native datetime extension arrays need an explicit compatibility guard; see defects below.

## CHUNKED_MODE_IMPACT

[Chunked profiling](../../aidatasetkit/profiling/chunked.py) is unchanged and does not call the datetime summarizer.

Chunked artifacts still:

- Have no normal column-evidence entries.
- Leave date inference null.
- Remain profile-only with the existing blocked verdict.

Their artifact schema nevertheless moves to 1.5 because the version constant is global. Existing chunked tests only had version assertions updated; no new explicit test proves the datetime summarizer cannot be invoked from that route.

## DATE_INFERENCE_INTERACTION

[Strict ISO inference](../../aidatasetkit/profiling/date_inference.py) is unchanged.

- Text stays textual/categorical.
- Native datetime columns remain excluded from textual recognition.
- The new tests check separation.
- No conversion or threshold change was found.

One existing rollback test now creates a schema-1.3-labelled record while retaining newer datetime fields. It no longer demonstrates restoration of the historical shape.

## PUBLIC_API_IMPACT

Additive intended changes:

- Public [DatetimeSummary](../../aidatasetkit/core/types.py).
- New configuration field.
- New profile/evidence serialized field.

**Compatibility defect:** [ColumnEvidence](../../aidatasetkit/evidence/types.py) inserts `datetime` **before** the existing positional `source` argument. An existing caller passing `source` positionally will bind it to `datetime` instead. The new field should not displace that positional slot.

[AIDataFacade.load](../../aidatasetkit/aidatasetkit/facade/facade.py) remains DataFrame-only.

## TEST_COVERAGE_REVIEW

The new tests meaningfully cover:

- Naive/aware endpoints and counts.
- NaT, partial missingness, one/two distinct timestamps.
- Unsorted data, duplicates, regular/irregular spacing.
- Explicit gaps and separate off-grid counts.
- Basic cadence rejection and resolution mismatch.
- Spring-forward elapsed-time semantics.
- Four resolutions, some wide-range/boundary cases.
- Serialization, identity separation, immutability, and endpoint-only publication.
- Strict ISO separation.

Material gaps and false-positive risks:

- No new fixture for either pandas major containing actual datetime evidence; the existing pandas-2 fixture is stale.
- No targeted mutation harness or demonstrated kills.
- No missing-like cadence cases such as `"NaT"`.
- No empty datetime series, duplicate-only series, or cadence-enabled all-NaT case.
- No duplicate off-grid timestamp case to prove distinct counting.
- No trailing-off-grid window test.
- No fall-back repeated-wall-clock assertion, fixed-offset zone case, or host timezone/locale determinism check.
- Boundary tests cover seconds, not all units and aware boundaries.
- No malformed temporal evidence validation tests.
- No datetime HTML rendering test.
- No positional evidence-constructor compatibility test.
- The determinism test reuses one computed profile, so it cannot detect nondeterministic profiling.
- The “without materialization” test asserts counts on a short gap; it would not kill an implementation that materialized missing timestamps internally.
- Output checks do not establish a scaling/resource bound.

**Focused tests, full suite, compatibility matrix, and mutations: NOT RUN under this authorization.**

## SECURITY_AND_PRIVACY_REVIEW

A read-only security review found no introduced exploitable vulnerabilities or exposed credentials.

| # | Severity | File | Lines | Vulnerability | Confidence |
|---|---|---|---|---|---|
| — | — | — | — | No security vulnerabilities identified in reviewed changes | — |

No introduced local-path exposure or stray temporary file was identified among the 22 worktree entries. Pre-existing ignored caches and environment files were preserved.

Privacy behavior is bounded but requires accurate documentation:

- No timestamp lists, samples, gap arrays, or per-missing-timestamp materialization occur in the new code.
- Minimum/maximum timestamps are intentionally published, including when they identify a singleton observation.
- Existing “no dataset values” evidence documentation needs to describe this authorized endpoint exception.

This is not a claim that endpoint aggregates are non-sensitive.

## DEFECTS_AND_GAPS

| Priority | Finding |
|---|---|
| Blocking | Missing schema/fingerprint migration-chain entry and rollback proof. |
| Blocking | Pandas-2 semantic fixture remains incompatible with the new schema/configuration. |
| High | [Cadence validation](../../aidatasetkit/core/config.py) lacks a missing-value check. `"NaT"` parses to pandas NaT, does not satisfy `<= 0`, and serializes as `"NaT"` instead of failing configuration validation. Installed pandas source/tests confirm those constituent behaviors; no project runtime check was executed. |
| High | [Datetime dispatch](../../aidatasetkit/profiling/datetime.py) assumes `.unit` and `.asi8` exist for everything classified DATETIME. Pandas also classifies Arrow timestamp extension dtypes as datetime, while Arrow arrays do not provide those native-array attributes. Eligibility must be narrowed without changing existing kind detection. |
| High | New evidence field displaces the public positional `source` parameter. |
| Medium | Resource claims are inaccurate: Boolean filtering and `np.unique` allocate population-sized intermediates and sort valid input. Worst-case work is `O(n log n)` and auxiliary memory includes `O(n)`, not solely `O(u)` as documented. The profiler's linear-complexity docstring is also now stale. |
| Medium | No temporal evidence invariant validation. |
| Medium | HTML omits the configured expected cadence, grid anchor semantics, regularity, resolution, and gap-interval count. Its displayed counts cannot be fully interpreted from the table alone. |
| Review | Trailing-off-grid missing-count semantics require an owner decision. |
| Review | No structured temporal quality findings exist; decide whether measurements alone satisfy wave acceptance. |
| Unverified | Overflow handling catches only `OverflowError` in duration serialization; supported pandas also exposes `OutOfBoundsTimedelta` as a `ValueError`. Cross-version range/error behavior needs focused proof. |
| Incomplete | Mutation evidence, per-major datetime fixtures, compatibility validation, and final local results are absent. |

The report still says `CONTRACT_RECORDED_BEFORE_IMPLEMENTATION`. It contains no premature PASS/certification claim, but its range/resource guarantees exceed what has been demonstrated.

## RECOVERY_DECISION

**`RECOVERABLE_WITH_REPAIR`**

No unrelated capability implementation was identified. Preserve the work; do not accept it as complete as-is.

## PROPOSED_REPAIR_PLAN

**Proposal only — nothing implemented.**

1. **Obtain owner decisions** on trailing-off-grid window semantics and whether explicit-cadence anomalies require structured quality findings.
2. **Repair eligibility and cadence validation**:
   - Restrict new summarization to supported native datetime arrays.
   - Preserve existing classification and handling of other extension dtypes.
   - Reject missing-like, nonpositive, calendar/ambiguous, and unrepresentable cadences explicitly.
   - Verify canonical cadence round trips and exact unit conversion on both pandas majors.
3. **Repair compatibility and evidence contracts**:
   - Preserve the positional evidence-constructor API.
   - Validate temporal invariants.
   - Document authorized endpoint disclosure.
   - Present expected cadence and interpretation context in HTML.
4. **Correct resource/range documentation** to match the actual algorithm; verify inherited ingestion limits remain effective. Do not expand chunked profiling.
5. **Complete schema migration**:
   - Add the exact before/after fingerprint step.
   - Roll back nullable datetime fields, the cadence setting, config identity, and schema label.
   - Prove restoration of the certified 1.4 semantic identity.
   - Update schema documentation.
6. **Complete tests and mutation checks** for the listed gaps, including counting-rule and no-materialization safeguards.
7. **Under separate implementation authorization**, reconcile fixtures for pandas 2 and 3 using the approved contract. Review every semantic delta; stop immediately for any Golden verdict change.
8. **Only after execution is authorized**, run focused tests, compatibility checks, mutations, then the complete suite. Record exact commands/results before making small local commits.

## FILES_THAT_SHOULD_REMAIN

Preserve **all 22 current modified/untracked files**. None warrants deletion or restoration during recovery.

## FILES_REQUIRING_REVIEW

Primary review targets are:

- [Datetime implementation](../../aidatasetkit/profiling/datetime.py).
- [Configuration](../../aidatasetkit/core/config.py).
- [Evidence types](../../aidatasetkit/evidence/types.py).
- [HTML presentation](../../aidatasetkit/evidence/report.py).
- [Focused tests](../../tests/unit/test_datetime_profiling.py).
- [Wave report](G2_W1_DATETIME_PROFILING_REPORT.md).
- All four changed generated examples.

Missing companion updates require review in the [migration chain](../../tests/unit/test_capability_fingerprint_migration.py), [pandas-2 fixture](../../examples/audit_churn/expected_audit_semantic.pandas2.json), and schema documentation.

## FILES_SUSPECTED_UNRELATED

**NONE.** The ingestion, packaging, and record-iterable test edits are attributable to global schema-version assertions, not unrelated functionality.

## OWNER_DECISIONS_REQUIRED

1. Approve or revise missing-observation counting when the latest observation is off-grid.
2. Confirm whether scalar temporal measurements suffice or structured explicit-cadence findings are required.
3. Authorize a repair wave and its subsequent validation.

Certification still requires branch CI on the exact candidate SHA, the expected Windows/Linux **10/10** matrix, CTO review, main integration and main CI on the certified code SHA, and a separate documentation seal.

## CHANGES_MADE

**NONE.**

No edits to project code, formatting, staging, commits, branch changes, stash operations, cleanup, fixture generation, test execution, or remote actions occurred during the recovery audit. This document records that audit at CTO request.

**G2-W1 remains incomplete and uncertified. G2-16 remains `MISSING`. Certified progress is unchanged. STOP FOR CTO REVIEW.**
