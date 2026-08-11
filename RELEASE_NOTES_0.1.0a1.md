# AIDatasetKit 0.1.0a1 — Public Alpha

**The safety and audit layer for tabular machine learning.**

This release exists to answer one question, in a form a machine and a person can
both read: *what happened between your data and your model, and why?*

```bash
pip install aidatasetkit
aidatasetkit audit train.csv --target Churn --task classification --output ./aidk-audit/
```

You get three files, a readable summary, and an exit code your CI can act on.

## What is included

- **Dataset profiling** — every column measured: kind, dtype, missingness,
  cardinality, dominance, infinities.
- **Quality and leakage diagnostics** — eleven checks, each with a severity, a
  reason, and the numbers behind it. A feature exactly equal to the target is
  reported as an error; a heuristic finding is marked as needing review rather
  than acted on.
- **Preprocessing planning** — what would be done to each column, and why. Driven
  by the *capabilities* a model declares, never by its name. Nothing is applied
  to your data.
- **Feature lineage** — `City` → imputation → one-hot → `City_Riyadh`,
  `City_Jeddah`. Taken from the preprocessing layer, not guessed from names.
- **Evidence and provenance** — one canonical `audit.json` with dataset identity,
  configuration identity, environment versions, and a verdict with its reasons.
- **A CLI built for CI** — documented exit codes and a `--fail-on` threshold.

## Why alpha

Three honest reasons.

**The interfaces have not met real users yet.** They were designed against
synthetic datasets and one author's judgement. The first serious external dataset
will teach us something we did not anticipate.

**The artifact schema will move.** It carries `schema_version` precisely because
we expect that. Version `1.0` of the schema is a starting point, not a promise.

**Only one task family is verified end to end.** Classification readiness has
been proved; regression, clustering, and the rest are designed for but not
delivered.

## Verified scope

Verified in this release: pandas DataFrames and CSV files · tabular data · binary
and multiclass classification readiness · scikit-learn model contexts · profiling
· quality and leakage diagnostics · visualization planning · capability-driven
preprocessing plans · feature lineage · audit artifacts · a CI-usable CLI.

**Verified environment** — the versions this release was actually tested against:

| Component | Version |
|---|---|
| Python | 3.12.3 |
| NumPy | 2.5.2 |
| pandas | 3.0.5 |
| SciPy | 1.18.0 |
| scikit-learn | 1.9.0 |

That is narrower than the **declared dependency ranges** in `pyproject.toml`
(`numpy>=1.26`, `pandas>=2.1`, `scipy>=1.11`, `scikit-learn>=1.4`). The ranges say
what should work; the table says what was run. Report anything that disagrees.

## Not yet a complete public workflow

Regression · model training orchestration · model comparison · clustering ·
anomaly detection · dimensionality reduction · deep learning · hosted service ·
enterprise policy management.

**These are on the roadmap, not cancelled.** The order changed deliberately: an
audit layer that real people use is worth more right now than a larger catalogue
nobody has tried.

## Known limitations

`docs/limitations.md` is the canonical list, and it travels inside every artifact
so a reader always has it. The most important one:

> Leakage detection is statistical. A feature that encodes the outcome for reasons
> the numbers do not show will not be found here.

## Expected breaking changes before 1.0

- Artifact field names and nesting may change; `schema_version` will move with
  them.
- CLI flags may be renamed or added. Exit codes are meant to be stable, but are
  not guaranteed before `1.0`.
- Python APIs — builder signatures, evidence dataclasses — may change shape.

## Before you share an artifact

An audit artifact deliberately avoids your data, but it is not empty of
information: it contains column names, class labels, numeric minima and maxima,
one-hot category names, and any ordering you supplied. `docs/privacy.md` lists
all of it precisely. Read that before publishing one.

## Reporting bugs

Open an issue with: the command you ran, the terminal output, and — if you can
share it — the `audit.json` it produced. The artifact is designed to be the bug
report: it carries the fingerprints, the environment, and the evidence.

The most useful feedback right now is not "it crashed" but **"the artifact does
not tell me X, and I needed X"**.

## Roadmap

**Current alpha** — S0 foundation through S5.5 evidence and provenance.

**Next, after alpha feedback** — S6 regression · S7 training and evaluation ·
S8 unified facade.

**Planned expansion** — S9 clustering · S10 anomaly detection and dimensionality
reduction · external model backends · deep learning.

## Not claimed

This release is not production-ready, is not certified against any framework, and
does not guarantee that data passing its checks is free of leakage. It reports
what its checks found. That is a smaller claim than it sounds, and it is the only
one the evidence supports.
