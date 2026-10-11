# G1-W7 — Excel worksheet selection evidence

G1-W7 is **not PASS**. G1-24 is `PARTIAL_PENDING_CERTIFICATION`.

This file records the implementation that is waiting for review. It does not
promote G1-24 and it does not change recorded progress. It names the
implementation commit and does not name a CI run started by the documentation
commit that adds this file.

## Identity

```text
STARTING_HEAD        = 59368ea382ce1cde0a3ceb4bcd7dfbb19f54ab66
IMPLEMENTATION_SHA   = b5b5403e5877889c3ec13ddc98a21cbd4ca7b379
BRANCH               = g1-w7-excel-sheet-evidence
BASELINE_MAIN_SHA    = 59368ea382ce1cde0a3ceb4bcd7dfbb19f54ab66
G1_24_STATUS         = PARTIAL_PENDING_CERTIFICATION
G1_PROGRESS          = 18/26 = 69.23%
OVERALL_PROGRESS     = 110/232 = 47.41%
ARTIFACT_SCHEMA      = 1.3
PUBLICATION_SCHEMA   = 1.0
PACKAGE_VERSION      = 0.1.0a1
```

Recorded progress stays at the G1-W6 seal. A later review that promotes G1-24
would move those figures to G1 19/26 = 73.08% and overall 111/232 = 47.84%.
That promotion is not made here.

## Contract

`load_table(path, *, sheet=None)` is unchanged as a signature. `sheet` is not
renamed, integer indexes are refused, and there is no second Excel API.

An eligible relation is an openpyxl worksheet in any visibility state. A
chartsheet is not eligible. Exactly one eligible worksheet is selected when
`sheet` is omitted. More than one is `MalformedInputError` until `sheet=`
names one. The name match is exact: case-sensitive, whitespace-sensitive, no
trimming, no Unicode normalization. An empty string is a missing name, not an
omission. A non-string `sheet` is `InvalidIngestionOptionsError`. `sheet=` on
any non-`.xlsx` input is `InvalidIngestionOptionsError`.

Every successful `.xlsx` load returns `source_selector` `{"kind": "sheet",
"name": "<exact title>"}`, including the automatic single-worksheet case. The
same record enters the config fingerprint. Non-`.xlsx` selectors stay `null`.
SQLite stays `kind` `table`. Schema `1.3` already stores an open `kind`
string, so `sheet` does not create schema `1.4`.

`max_source_bytes` still applies to the whole file. `max_rows`,
`max_columns`, `max_cells` and `max_field_length` apply only to the selected
worksheet. Dimensions are checked before the selected cells are read.
`max_field_length` is checked on `str`, `bytes` and `bytearray` values while
those cells are scanned, and before `DataFrame` construction. Other Python
values are not turned into text in order to measure them. The workbook is
closed on success and on refusal. No partial artifact is published.

openpyxl stays the optional `excel` extra. `read_only=True` and
`data_only=True` stay. Macro-enabled workbooks stay refused. `keep_links` is
not changed. Formulas are not evaluated.

## Local validation on the implementation commit

Python 3.12.14 in `.venv`.

| Check | Result |
|---|---|
| Excel, openpyxl-absence and CLI ingestion | 89 passed |
| Schema, artifact and fingerprint | 198 passed |
| Golden fixtures | 5 passed |
| SQLite / G1-W6 | 46 passed |
| Chunked profiling / G1-W5 | 35 passed |
| Packaging and architecture | 180 passed |
| Those groups together | 553 passed |
| Full suite | 4615 passed, 47 skipped, 0 failed, 410.25s |
| JUnit | tests=4662 failures=0 errors=0 skipped=47 time=410.190 |
| `scripts/g1_w7_mutations.py` | 17/17 KILLED, baseline exit 0, each tree restored |

Each mutation applied one weakening, failed its named assertion, and restored
the exported tree:

| ID | Kill reason |
|---|---|
| M-W7-01 | sheet argument ignored |
| M-W7-02 | first sheet always selected |
| M-W7-03 | missing sheet silently falls back |
| M-W7-04 | automatic single sheet not recorded |
| M-W7-05 | explicit selector omitted from evidence |
| M-W7-06 | selector omitted from config fingerprint |
| M-W7-07 | two sheets share one config fingerprint |
| M-W7-08 | sheet= accepted for non-XLSX input |
| M-W7-09 | chartsheet treated as worksheet |
| M-W7-10 | hidden worksheet removed from eligibility |
| M-W7-11 | row limit ignored |
| M-W7-12 | column or cell limit ignored |
| M-W7-13 | max_field_length ignored |
| M-W7-14 | cells/DataFrame published before limit refusal |
| M-W7-15 | openpyxl becomes a core dependency |
| M-W7-16 | default single-sheet behavior changes |
| M-W7-17 | SQLite table selector behavior changes |

`scripts/release_smoke_test.sh` on this tree: wheel and sdist for `0.1.0a1`,
`twine check` passed, clean installs without the excel extra, installed CLI
audit schema `1.3`. Nothing was published. The installed-wheel `.xlsx` check
is part of the same review and uses that wheel plus the excel extra.

## Notes

- The OOXML writer will not keep two worksheets whose titles differ only by
  case, so case sensitivity is proven by refusing `data` and `DATA` when the
  only worksheet is `Data`.
- openpyxl refuses to save a workbook whose only worksheet is hidden. A
  very-hidden worksheet beside a visible one is a normal fixture. A second
  fixture rewrites `state="visible"` to `state="veryHidden"` in `workbook.xml`
  after a normal save, and that lone worksheet is still selected.
- A chartsheet written by openpyxl needs an empty relationships part before
  `read_only=True` can open it. Eligibility uses `Workbook.worksheets`, which
  excludes chartsheets. A chartsheet name is refused before any row scan.
- Historical G1-W4 and G1-W5 reports are unchanged.
