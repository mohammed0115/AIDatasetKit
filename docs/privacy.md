# Privacy

An audit artifact is meant to be shared — committed to a repository, attached to
a CI run, sent to a colleague. So the default is that it does not contain your
data.

## What is never in an artifact by default

- Raw cell values.
- The most frequent value of a column. This is the one profiling field that holds
  real data — for an email column it is somebody's address, for a free-text
  column a sentence out of your dataset — and it is replaced by a SHA-256 digest.
- Values quoted inside a quality finding. Two checks (`constant_column` and
  `near_constant_column`) name the value they are about. Evidence removes it from
  both the `details` and the message text; a digest in one and the address in the
  other would be no protection at all.

## What *is* in an artifact

Be clear-eyed about this. An artifact still reveals:

- **Column names.** `patient_hiv_status` is disclosive even with no values.
- **Row and column counts**, missing counts, distinct counts, ratios.
- **Data types** and detected kinds.
- **Numeric summary statistics** — min, max, mean, quartiles. A minimum and a
  maximum are real values from your data.
- **Target class labels** — `churn`, `stay`. Needed to interpret the audit at
  all.
- **One-hot output names**, which contain the *categories* of encoded columns:
  `City_Riyadh`, `City_Jeddah`.

That last one deserves a note. Feature lineage exists to tell you what a column
became, and for a one-hot encoding those names *are* the categories. The exposure
is bounded: only low-cardinality columns are encoded by default, because a column
above the cardinality threshold is held back for review rather than encoded. If
your low-cardinality categories are themselves sensitive, do not share the
artifact.

## Opting in to raw values

```bash
aidatasetkit audit train.csv --target Churn --include-values
```

This puts the most frequent value of each column into the artifact in plain
text. It is occasionally what you want when auditing your own non-sensitive data
and it is never a safe default. The choice is recorded in the artifact's config
section and changes the config fingerprint, so an artifact always says which mode
produced it.

## Hashes are identifiers, not encryption

The digests in an artifact exist so two runs can be compared: *has the most
common value changed since last week?* They are **not** a privacy mechanism in
the cryptographic sense. A SHA-256 of a value from a small or guessable domain
can be reversed by trying candidates — a digest of a boolean, a country code, or
a category from a known list offers no protection at all.

Treat a digest as an identifier for a value, not as a way of hiding one.

## Your responsibility

AIDatasetKit writes files to the directory you choose. What happens to them
afterwards is outside its control. Before publishing an artifact:

- Read it. It is JSON with sorted keys and it is meant to be read.
- Check the column names against your disclosure rules.
- Check the numeric summaries — a minimum and a maximum are real observations.
- If you used `--include-values`, assume it contains data.

## No compliance claims

This project makes no claim of compliance with GDPR, HIPAA, SOC 2, or any other
framework, and holds no certification. It produces evidence that may be useful
in such a conversation. It does not conclude one.
