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

## Top-level shape

| Key | What it holds |
|---|---|
| `schema_version` | The artifact contract this file was written against. |
| `stage` | `inspected`, `planned`, or `prepared` — how far the run got. |
| `verdict` | `ready`, `ready_with_warnings`, `review_required`, `blocked`. |
| `verdict_reasons` | Why, in the order the policy applied them. |
| `dataset` | Identity: fingerprints, shape, column list. No values. |
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

## What is never in it

No DataFrames, Series, or arrays. No estimator or transformer objects. No
callables, no loggers, no memory addresses. This is enforced rather than
intended: the serializer refuses anything it does not explicitly allow, instead
of falling back to `repr()`. See [privacy.md](privacy.md) for raw values.
