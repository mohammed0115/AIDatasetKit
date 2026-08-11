# Security

## Reporting

This is a public alpha with no production users. If you find a security issue,
open an issue describing it, or contact the maintainer privately if you believe
disclosure would put someone's data at risk.

## What this software does with your data

AIDatasetKit reads a DataFrame or a CSV file you point it at, computes statistics
from it in memory, and writes artifacts to a directory you choose. It makes **no
network requests**, sends no telemetry, and writes nothing outside the output
directory.

The generated `report.html` is a standalone page with inlined CSS and no scripts,
no external stylesheets, and no remote images. Opening it makes no network
requests.

## Data in artifacts

By default an audit artifact contains no raw cell values — see
[docs/privacy.md](docs/privacy.md) for exactly what it does and does not contain,
including the cases where a real value can still appear (numeric minima and
maxima, one-hot category names, target class labels).

`--include-values` deliberately puts data into the artifact. Use it only on data
you are willing to share as widely as the artifact will travel.

## Hashes

The digests in an artifact are **identifiers, not encryption**. A SHA-256 of a
value from a small or guessable domain — a boolean, a country code, a category
from a known list — can be reversed by trying candidates. Do not treat a digest
as a way of hiding a value.

## Untrusted input

The CSV parser is pandas'. A malicious CSV can exhaust memory, and column names
from an untrusted file are carried into the artifact and the HTML report. Report
output is HTML-escaped, so a column named `<script>alert(1)</script>` renders as
text rather than executing — but treat any artifact built from untrusted input as
untrusted content.

## No compliance claims

This project holds no certification and makes no claim of compliance with any
regulatory framework.
