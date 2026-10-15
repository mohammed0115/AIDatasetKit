# The audit artifact

`audit.json` is the record. Everything else — the HTML report, the terminal
summary, any future SARIF or dashboard output — is a *rendering* of it. That
direction is deliberate: two producers of the same document eventually disagree,
and an audit that disagrees with itself is worthless.

## Schema version

```json
{ "schema_version": "1.5" }
```

The artifact format is versioned **separately from the package**. Most releases
will not change the shape of the record; the ones that do must be identifiable
without consulting a changelog. Adding, removing or redefining a field bumps the
version: an additive change is a minor bump, anything else a major one. A later
`0.4.0` may still write `1.5`. `lineage.json` carries the same version.

| Version | Change |
|---|---|
| `1.0` | The first published contract. |
| `1.1` | G1-W1: the top-level `ingestion` record. |
| `1.2` | G1-W5: the top-level `chunked_profiling` record. `null` unless the caller opted into chunked CSV/TSV profiling. |
| `1.3` | G1-W6: the top-level `source_selector` record. `null` unless a relation inside the file was chosen. A SQLite table is `{"kind": "table", "name": "<table>"}`. An Excel worksheet is `{"kind": "sheet", "name": "<exact title>"}`. Kind `sheet` uses this same record; it is not a new schema version. |
| `1.4` | G1-W8: the top-level `date_inference` record. A full audit records the fixed strict-ISO policy and any textual columns that met it. `null` on the chunked profile-only route. The record does not convert values and does not change detected kind, the verdict, or the dataset fingerprint. Removing the key and setting the version back to `1.3` restores the previous semantic artifact. |
| `1.5` | G2-W1: each column record gains nullable `datetime` evidence. Eligible native datetime columns record counts, minimum and maximum, timezone, resolution, exact observed spacing, and optional fixed-cadence gap counts. This is measurement-only evidence: it does not add findings or affect the verdict. Minimum and maximum endpoint aggregates may still be sensitive. |

An artifact with an `ingestion` key and `schema_version` `1.0` exists: `main` at
`90ecfae` wrote it, before the version was corrected. Read it as `1.1`; nothing
else differs.

## Datetime evidence

Every column record in schema `1.5` has a nullable `datetime` field. It is
populated only for native pandas datetime arrays, including timezone-aware
arrays; text and object columns are not parsed or converted. Counts distinguish
valid rows, `NaT` rows, distinct timestamps, and duplicate rows (every valid
occurrence after the first). Minimum and maximum are the only source timestamps
published, formatted in the source timezone with its applicable UTC offset.

Observation spacing is calculated over sorted distinct instants. Equal
consecutive elapsed durations establish exact regular spacing; irregular
spacing does not infer a cadence or establish that observations are defective.
The optional `KitConfig.expected_datetime_cadence` is a positive fixed duration.
When set, the grid is anchored at the minimum and covers the closed observed
window. Missing expected positions, contiguous missing runs, off-grid distinct
timestamps, and duplicate rows are separate measurements. Off-grid timestamps
do not occupy expected grid positions. With no explicit cadence, the gap
measurements are `null`. The evidence is bounded and does not materialize
missing timestamps or publish timestamp lists. Chunked profiling does not
include datetime summaries.

## Where a run is published

The CLI never writes the artifacts straight into `--output`. Each audit is one run:

```
<output>/
  CURRENT                   {"run_id": ..., "manifest_sha256": ..., "publication_schema_version": "1.0"}
  runs/<run_id>/
    audit.json  lineage.json  report.html
    manifest.json           {"run_id", "created_at", "files": {name: {"bytes", "sha256"}}}
  .staging/                 private to writers in progress; never read
```

The files are written in full and synced in a staging directory of their own,
re-read against the manifest, moved into `runs/<run_id>/`, and only then made
current: a new `CURRENT` is written beside the old one and replaces it with
`os.replace`, the single commit point. Before that step any failure leaves the
previous run current; the step itself either happens or does not. Two audits
into one directory each publish a complete run and the later commit is current;
there is no lock to be left behind by a killed process.

`aidatasetkit.evidence.read_current(output)` is the reader. It follows `CURRENT`,
checks the manifest's digest against it, and checks that the run holds exactly
the listed files at the listed sizes and digests, returning the bytes it checked.
Anything else raises `CorruptPublicationError`; no run yet raises
`NoPublishedRunError`. Finished runs are never deleted, and a staging directory
left by a killed writer is never swept, because either may belong to a process
still using it.

The layout is versioned separately (`publication_schema_version`), from the
artifact schema below and from the package.

## Top-level shape

| Key | What it holds |
|---|---|
| `schema_version` | The artifact contract this file was written against. |
| `stage` | `profiled`, `inspected`, `planned`, or `prepared` — how far the run got. |
| `verdict` | `ready`, `ready_with_warnings`, `review_required`, `blocked`. |
| `verdict_reasons` | Why, in the order the policy applied them. |
| `dataset` | Identity: fingerprints, shape, column list. No values. |
| `ingestion` | How the table was read: source kind, format, encoding, delimiter and whether it was detected, named or fixed by the format, header, rows, columns, memory, warnings. `null` when the caller built the frame itself. No path, no values. |
| `chunked_profiling` | Opt-in CSV/TSV chunked scan, or `null` when the caller did not ask for it. A typed record: mode, chunk size, rows scanned, population size, `full_population_scanned`, exact and approximate and unavailable metrics, sampling method/seed/size, population fingerprint, and temporary-storage status. Quartiles are approximate and are not verdict inputs. The chunked route's verdict is blocked because the full audit did not run. |
| `source_selector` | The relation chosen inside the file, or `null` when the source is already one table. A SQLite table is `{"kind": "table", "name": "<catalog name>"}`. An Excel worksheet is `{"kind": "sheet", "name": "<exact title>"}`, including when that worksheet was the only eligible one. The name is not SQL and not a worksheet index. It is included in the config fingerprint, so two audits that differ only by the selected table or worksheet do not share a config identity. The HTML report labels the row with the kind. |
| `config` | The settings that steered the run, and their fingerprint. |
| `target` | What the task detector concluded, if a target was named. |
| `model` | The capability context, if one was supplied. No estimator. |
| `plan_fingerprint` | Identity of the preprocessing plan. |
| `columns` | Per-column measurements from profiling. |
| `findings` | Quality findings, preserved structurally. |
| `decisions` | What preprocessing would do to each feature, and why. |
| `lineage` | Where each input column ended up. |
| `warnings` | Non-fatal problems from the run itself. |
| `known_limitations` | What this library cannot establish. |
| `provenance` | Timestamp, environment versions, evidence fingerprint. |

## How the table was read

The CLI reads every file through `aidatasetkit.ingestion.load_table`, and the
`ingestion` record says what it did (a 60-row semicolon file; `memory_bytes` is pandas'
deep memory estimate and differs between pandas versions):

```json
"ingestion": {"source_kind": "file", "format": "csv", "encoding": "utf-8",
  "delimiter": ";", "delimiter_source": "detected", "header": true,
  "row_count": 60, "column_count": 4, "memory_bytes": 8512, "warnings": []}
```

`delimiter_source` is `detected`, `explicit` (the caller named it) or `format`
(a `.tsv` is tab-separated). A single-column file has `delimiter: null` and a
warning. Fields that do not apply — the encoding of a DataFrame — are `null`, not
invented. `memory_bytes` is pandas' deep size of a materialized frame. The
chunked CSV/TSV scan records the same ingestion fields and sets `memory_bytes`
to `null`, because no frame was built and the figure is not applicable.
`AuditBuilder.build(..., ingestion=loaded.metadata)` records it and
refuses metadata whose row or column count disagrees with the frame. The HTML
report shows it as its *Input* section. The record was added in G1-W1, and the
schema moved to `1.1` with it. Each moved the semantic fingerprint, and
`tests/unit/test_capability_fingerprint_migration.py` records both as separate
steps: setting the version back to `1.0` reproduces the intermediate identity,
and then removing the key reproduces the one before G1-W1.

## Stages

An audit is useful before a preprocessor exists and useful again afterwards, so
the artifact records which it is:

- **`profiled`** — dataset profiling completed. Quality inspection, task
  detection and preprocessing did not run. This is the chunked CSV/TSV route.
- **`inspected`** — profiled and quality-checked. No preprocessing was planned.
- **`planned`** — a plan exists; nothing has been fitted. Lineage steps are
  *intended*, and each lineage entry has `"observed": false`.
- **`prepared`** — a preprocessor was fitted, so lineage outputs are *observed*.

Without this, an empty `lineage` would be ambiguous between "nothing was
transformed" and "we never got that far".

## Findings

Quality findings are preserved as structure, never flattened into prose:

```json
{
  "code": "target_leakage_exact_duplicate",
  "severity": "error",
  "message": "Column 'Churn_Copy' is exactly equal to the target 'Churn' ...",
  "column": { "name": "Churn_Copy", "label_type": "str" },
  "recommendation": "Remove it, or confirm ...",
  "requires_review": false,
  "details": { "target": "Churn" },
  "source": "quality"
}
```

`details` is where the numbers live. A finding reduced to its message is a
finding nobody can act on programmatically.

## Decisions

```json
{
  "feature": { "name": "MonthlyCharges", "label_type": "str" },
  "role": "numeric",
  "action": "include",
  "reason_code": "numeric_feature",
  "steps": ["median_imputation", "standard_scaling"],
  "fit_scope": "training_only",
  "model_requirement": "model 'logistic_regression' declares requires_scaling=true",
  "source": "preprocessing"
}
```

`model_requirement` is the field that makes a plan explicable. A scaler is often
present because of the *estimator*, not the column, and an audit that showed the
step without naming the capability would leave the reader to invent a reason. It
states the declared capability and claims nothing beyond it.

## Fit scope

The field exists for leakage auditing:

| Value | Meaning |
|---|---|
| `training_only` | The step learned something from the fitting rows. |
| `no_fit` | Nothing is learned; the step is a pure per-row function. |
| `not_applicable` | The feature never reaches a transformer. |

The classification is per step, not a blanket claim. A median, a most-frequent
category, a scaler's centre, a one-hot vocabulary, and the categorical sentinel
are all read off the data. An ordinal encoding uses the *order you supplied*, an
explicit mapping is *your* table, and parsing text to a float is per-row — none
of those can carry information between rows.

## Column labels

Labels are not always strings, and `0` and `"0"` are two different columns. Every
label is recorded as an object rather than a bare string:

```json
{ "name": "0", "label_type": "int" }
```

## Identity and diffing

Four fingerprints answer four different questions:

| Fingerprint | Question |
|---|---|
| `dataset.fingerprint` | Is this the same data? |
| `dataset.schema_fingerprint` | Is this the same shape and dtypes? |
| `config.fingerprint` | Were the same settings in force? |
| `provenance.semantic_fingerprint` | Is this the same evidence? |

The **semantic fingerprint excludes the timestamp and the environment**. Two
audits of the same data under the same settings are the same audit, and a clock
should not be able to say otherwise. In Python:

```python
differences = one.differs_from(two)
# {'dataset': True, 'schema': False, 'config': False, 'plan': False, 'evidence': True}
```

Canonical JSON has sorted keys and fixed separators, so `git diff` on two audits
shows what changed rather than reshuffled formatting.

**Fingerprints are comparable within one environment, not across every
environment.** The dataset digest is built on pandas' own row hasher, so a
different pandas major version can produce a different digest for identical data.
That is why every artifact records the versions that produced it, and why
`differs_from` is meant for comparing runs of the same installation. Comparing
across environments, check `schema_fingerprint` and the recorded environment
first — a changed dataset fingerprint beside a changed pandas version is not
evidence that the data changed.

## What is never in it

No DataFrames, Series, or arrays. No estimator or transformer objects. No
callables, no loggers, no memory addresses. This is enforced rather than
intended: the serializer refuses anything it does not explicitly allow, instead
of falling back to `repr()`. See [privacy.md](privacy.md) for raw values.
