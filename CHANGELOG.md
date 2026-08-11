# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions before `1.0` may change public interfaces and the artifact schema; see
[API stability](#api-stability).

## 0.1.0a1 — unreleased

First public alpha. Everything below is new, because nothing was public before.

### Audit and evidence

- `AuditArtifact`: one canonical, versioned, deterministic record of what an
  analysis established — dataset identity, quality findings, preprocessing
  decisions, feature lineage, model context, environment, and known limitations.
- Artifact schema `1.0`, versioned **independently of the package** so stored
  artifacts stay readable as the library changes.
- Deterministic dataset and configuration fingerprints. Content is identified
  without being stored; the semantic fingerprint excludes the clock, the
  environment, and the file name, so two runs of the same data compare equal.
- Redaction on by default: text taken from a dataset is hashed unless its key is
  vocabulary this library defines.
- `lineage.json` — each input column and what it became, taken from the
  preprocessing layer rather than parsed back out of output names.
- `report.html` — a standalone page rendered from the same canonical mapping as
  `audit.json`, with no scripts, no network requests, and no dependencies.

### Command line

- `aidatasetkit audit` — profile, check, plan, and write three artifacts.
- Documented exit codes for CI: `0` below threshold, `1` could not run,
  `2` threshold met, `3` blocked.
- `--fail-on never|warning|review|error`, defaulting to `review`.
- A dataset that cannot be prepared still produces a full artifact with verdict
  `blocked` and the reason recorded.
- Nothing is ever trained. `--model` supplies a capability context only.

### Analysis layers

- **Statistics** — every degenerate case that numpy or scipy answers with `NaN`
  raises instead, so a number in a report is never a silent placeholder.
- **Profiling and data quality** — per-column measurement and eleven independent
  checks covering leakage, identifiers, constants, cardinality, missingness,
  outliers, duplicates, and numbers stored as text.
- **Smart visualization** — chart *recommendations* separated entirely from
  rendering; matplotlib stays optional.
- **Preprocessing** — capability-driven planning. The plan is a proposal you can
  read, with a reason for every decision, and it is never applied behind your
  back.
- **Classification catalog** — nine scikit-learn classifiers whose declared
  capabilities were each verified by running the estimator.

### API stability

Interfaces and the artifact schema may change before `1.0`. The artifact carries
`schema_version` so a future release can recognise and migrate an older file.

### License

- Apache License 2.0 (`Apache-2.0`), chosen by the repository owner and applied
  in the metadata and in `LICENSE`. See `docs/LICENSE_DECISION.md`.

### Known limitations

See `docs/limitations.md`, which is the single canonical list and travels inside
every artifact.
