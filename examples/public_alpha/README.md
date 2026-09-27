# Public alpha demo

A synthetic customer table with four planted problems, and the audit that finds
them. Everything here is generated from a fixed seed; no real data appears.

## Run it

```bash
python examples/public_alpha/make_sample.py

aidatasetkit audit examples/public_alpha/sample.csv \
    --target Churn \
    --task classification \
    --output ./aidk-audit/
```

Exit code **3** — the audit found something that stops a build.

## What it produces

```
aidk-audit/
├── CURRENT                     which run is current, and its manifest digest
└── runs/<run_id>/
    ├── audit.json              the canonical machine-readable record
    ├── lineage.json            each input column and what it became
    ├── report.html             the same evidence, rendered for a person
    └── manifest.json           size and SHA-256 of each file above
```

`lineage.json` is populated only when a preprocessing plan exists, which needs a
model context. The command above has none, so it is written with an empty
`features` list and `stage: inspected` — an honest "nothing was planned yet".
Add `--model logistic_regression` to see the transformations:

```bash
aidatasetkit audit examples/public_alpha/sample.csv \
    --target Churn --model logistic_regression --output ./aidk-audit/
```

`audit.json` is the record. `report.html` is a rendering of it — never the other
way round, so the two cannot drift apart.

## What it finds, and why each one matters

| Column | Finding | Why it matters |
|---|---|---|
| `ChurnedFlag` | **error** — exactly equal to the target | A model trained on it scores near-perfectly and has learned nothing. This is what blocks the run. |
| `CustomerID` | **review** — identifier-like, 100% distinct | A key is not a feature. It is *reported*, never dropped: the check is a heuristic and the decision is yours. |
| `SupportTicketID` | **warning** — 380 levels | One-hot encoding this produces 380 columns. It is held back for review rather than encoded. |
| `Churn` | **warning** — the rarest class is 16.8% | Accuracy will look good while the model ignores the minority class. |
| `TenureMonths` | recorded — 39 missing values | Not a finding. The plan says how they would be filled, and by what. |

Nothing in this list was fixed, filled, or dropped. The audit reports; you decide.

## What to look at first

Open `report.html` in a browser. Then read `audit.json` — it is sorted, canonical
JSON meant to be diffed in git, so committing it makes the next run show you
exactly what changed about your data.

## Use it in CI

```bash
aidatasetkit audit data/train.csv --target Churn --fail-on review
```

`--fail-on review` fails the build when something needs a human decision.
`--fail-on error` is more permissive: only a certain problem stops it.

## Before sharing an artifact

It contains no cell values by default, but it does contain column names, class
labels, numeric minima and maxima, and one-hot category names. See
`docs/privacy.md`.
