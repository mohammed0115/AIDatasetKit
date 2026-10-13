# G1-W10 — Bounded legacy .xls worksheet ingestion

G1-W10 is **PASS**. G1-05 is `SUPPORTED_AND_TESTED`.

`CERTIFIED_CODE_SHA` is the tip integrated to `main` before this record.
`DOCS_SEAL_SHA` is the documentation-only commit that adds this certification.
This file names `CERTIFIED_CODE_SHA` and does not name `DOCS_SEAL_SHA` or any
CI run started by that documentation commit.

```text
STARTING_HEAD            = bf79e2c726ade2ed6f5ec2a62ccaeed90d082397
CERTIFIED_CODE_SHA       = 0c816d5c778b3367c71f17b458e4aef230a0d629
DOCS_SEAL_SHA            = not recorded in this file
BRANCH                   = g1-w10-xls-reader
G1_W10_STATUS            = PASS
G1_05_STATUS             = SUPPORTED_AND_TESTED
ARTIFACT_SCHEMA          = 1.4
PUBLICATION_SCHEMA       = 1.0
PACKAGE                  = 0.1.0a1
G1_PROGRESS              = 22/26 = 84.62%
OVERALL_PROGRESS         = 114/232 = 49.14%
```

Before this seal the recorded progress was G1 21/26 = 80.77% and overall
113/232 = 48.71%.

## What this wave does

`load_table` reads one worksheet from a local regular `.xls` file. `TableFormat.XLS` is `xls`. The suffix `.xls` selects it. There is no new public function and no new CLI flag. `sheet=` and `--sheet` are the existing worksheet arguments.

xlrd is added to the existing `excel` extra as `xlrd>=2.0.2,<3`. openpyxl is unchanged and remains the `.xlsx` reader. xlrd is not a core dependency. A minimum or core install refuses `.xls` with `MissingDependencyError` naming `aidatasetkit[excel]`. `.xlsx` is never dispatched to xlrd.

The file is inspected before it is treated as BIFF. The OLE compound-document signature is required. A ZIP archive, including a renamed `.xlsx`, and HTML or XML markup are refused. `.xlsm` and `.xlsb` stay unsupported by suffix.

Exactly one eligible worksheet may omit `sheet=`. Several eligible worksheets require an exact, case-sensitive, whitespace-sensitive title. Hidden and very-hidden worksheets stay eligible. Charts, embedded objects, defined names, and embedded worksheets are not tables.

The first row is the header, same as `.xlsx`. Empty and blank cells are missing. Text is `str`. Boolean is `bool`. A number is the numeric value xlrd returns. A date is a Python `datetime` from xlrd's conversion and `book.datemode` (1900 and 1904). An Excel error cell is `MalformedInputError` without the code, the formula, or the cell text.

Formulas are not evaluated. The formula expression is not stored as the cell value. Only the cached calculation result xlrd exposes is kept. A missing cached value is a missing cell.

Macros and VBA are not executed, extracted, fingerprinted, or published. The reader ignores them. That ignore is not a scan and not a safety verdict. A readable workbook is not refused only because those streams are ignored.

`max_source_bytes` is applied by the loader before xlrd opens the workbook. xlrd may parse inside that accepted byte budget. This is not a streaming reader. `max_rows`, `max_columns`, `max_cells`, and `max_field_length` apply to the selected worksheet after its shape is known and before the DataFrame is built. An oversized unselected sheet does not trip those selected-sheet limits. The file-size limit still covers the whole workbook. A refusal publishes no frame, artifact, `CURRENT` pointer, or partial output. Workbook resources are released on success and on every failure path.

`encoding` and `delimiter` stay invalid for `.xls`. The header contract is unchanged.

Evidence records `format=xls`, `source_kind=file`, and `source_selector={"kind": "sheet", "name": "<exact title>"}`. Encoding and delimiter stay null. The selected sheet enters the config fingerprint the same way an `.xlsx` sheet does. Paths, workbook representations, formulas, macros, and embedded content do not enter evidence.

## What this wave does not do

It does not read `.xlsm`, `.xlsb`, XML, YAML, PostgreSQL, SQLAlchemy, or query results. It does not detect encodings. It does not change G1-17, G1-10, G1-11, or G1-15. It does not change the artifact schema, the publication schema, or the package version. Existing golden artifacts are not regenerated. `AIDataFacade.load` stays DataFrame-only.

## CI

These runs are for `CERTIFIED_CODE_SHA`. This section does not name a CI run of the commit that adds it.

```text
CERTIFIED_CODE_SHA = 0c816d5c778b3367c71f17b458e4aef230a0d629
BRANCH_CI_RUN      = 38387353796
BRANCH_CI_URL      = https://github.com/mohammed0115/AIDatasetKit/actions/runs/38387353796
BRANCH_CI_SHA      = 0c816d5c778b3367c71f17b458e4aef230a0d629
BRANCH_CI_EVENT    = push
BRANCH_CI_BRANCH   = g1-w10-xls-reader
BRANCH_CI_RESULT   = success
BRANCH_CI_JOBS     = 10/10
BRANCH_CI_STARTED  = 2026-10-13T11:01:24Z
BRANCH_CI_UPDATED  = 2026-10-13T11:07:31Z
MAIN_CI_RUN        = 38421556058
MAIN_CI_URL        = https://github.com/mohammed0115/AIDatasetKit/actions/runs/38421556058
MAIN_CI_SHA        = 0c816d5c778b3367c71f17b458e4aef230a0d629
MAIN_CI_EVENT      = push
MAIN_CI_BRANCH     = main
MAIN_CI_RESULT     = success
MAIN_CI_JOBS       = 10/10
MAIN_CI_STARTED    = 2026-10-13T15:22:51Z
MAIN_CI_UPDATED    = 2026-10-13T15:29:57Z
```

## Status

G1-W10 = PASS. G1-05 = `SUPPORTED_AND_TESTED`. xlrd stays on the optional `excel` extra. `.xlsx` still uses openpyxl. The reader is bounded by `max_source_bytes` and is not a streaming reader. Formulas are not evaluated. VBA is ignored by the reader and is not a security scan or a safety verdict. G1-17 stays `PARTIAL`. G1-10, G1-11, and G1-15 stay `MISSING`. G1-13 stays `SUPPORTED_AND_TESTED`. Artifact schema `1.4`. Publication schema `1.0`. Package version `0.1.0a1`. Recorded progress is G1 22/26 = 84.62% and overall 114/232 = 49.14%.
