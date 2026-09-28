# The audit artifact

`audit.json` is the record. Everything else — the HTML report, the terminal
summary, any future SARIF or dashboard output — is a *rendering* of it. That
direction is deliberate: two producers of the same document eventually disagree,
and an audit that disagrees with itself is worthless.

## Schema version

```json
{ "schema_version": "1.0" }
```

The artifact format is versioned **separately from the package**. Most releases
will not change the shape of a recorded field; the ones that do must be
identifiable without consulting a changelog. `0.1.0a1` writes schema `1.0`, and a
later `0.4.0` may still write `1.0`.

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
| `stage` | `inspected`, `planned`, or `prepared` — how far the run got. |
| `verdict` | `ready`, `ready_with_warnings`, `review_required`, `blocked`. |
| `verdict_reasons` | Why, in the order the policy applied them. |
| `dataset` | Identity: fingerprints, shape, column list. No values. |
| `ingestion` | How the table was read: source kind, format, encoding, delimiter and whether it was detected, named or fixed by the format, header, rows, columns, memory, warnings. `null` when the caller built the frame itself. No path, no values. |
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
invented. `AuditBuilder.build(..., ingestion=loaded.metadata)` records it and
refuses metadata whose row or column count disagrees with the frame. The HTML
report shows it as its *Input* section. The record was added in G1-W1, which
moved the semantic fingerprint; `tests/unit/test_capability_fingerprint_migration.py`
records the step and reproduces the previous fingerprint by removing the key.

## Stages

An audit is useful before a preprocessor exists and useful again afterwards, so
the artifact records which it is:

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
