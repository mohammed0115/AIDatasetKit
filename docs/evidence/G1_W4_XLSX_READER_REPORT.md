# G1-W4 — Excel .xlsx reader

Evidence for the G1-W4 wave of `docs/AIDATASETKIT_EXECUTION_ROADMAP.md`
(audit row G1-04). Every claim below names the run that proves it.

## Scope and baseline

```text
PRE_CLOSURE_SHA  = dc70d8c60a3defdc35a805fbe7b6996b2fdb800c (main = origin/main after G1-W3)
BRANCH           = g1-w4-xlsx-reader
CODE_SHA         = 01436719d36607044e638665b631e465c33c42ba
```

Wave content, per the roadmap: XLSX (openpyxl as an optional extra), with
sheets and an explicit sheet choice, refusing macro-enabled files.

## What was built

**`aidatasetkit/ingestion/excel.py`** — an `.xlsx` workbook is read as one
worksheet's table:

- The first row is the header. Duplicate header cells are refused
  (`DuplicateHeadersError`); an empty header cell is named by position.
- The row, column and cell budgets are enforced from the worksheet's declared
  dimensions **before** any cell is materialized (`read_only` mode).
- A workbook with more than one sheet is ambiguous: refused unless the new
  keyword-only `sheet=` argument to `load_table` names one. `sheet=` is refused
  for every non-xlsx source.
- A macro-enabled archive (an `xl/vbaProject.bin` member) is refused: a
  `.xlsx` is a data container, and macro content is outside the trust this
  library extends to an input file.
- A corrupt or non-workbook zip is `MalformedInputError`, never a bare
  openpyxl/zip error.
- Formulas read as their cached values (`data_only=True`); the formula text is
  never executed.
- A missing openpyxl is `MissingDependencyError` naming the `excel` extra.

**Dependency**: new optional extra `excel = ["openpyxl>=3.1.0"]`. The 3.1 line
reads the worksheet's declared dimensions in `read_only` mode without
materializing the cells, which the preflight relies on. Pinned in both
constraints files; installed in the CI extras jobs.

## Test evidence

Focused run (reference venv, CPython 3.12.14, pandas 3.0.5, openpyxl 3.1.5):

```text
tests/unit/test_ingestion_excel.py + test_ingestion_without_openpyxl.py
    + test_ingestion.py + test_dependency_contract.py
    + tests/integration/test_cli_ingestion.py                          = 186 passed
```

Full suite from the committed code at `0143671` (reference venv, CPython 3.12.14,
pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0, openpyxl 3.1.5):

```text
TESTED_SHA  = 01436719d36607044e638665b631e465c33c42ba
COMMAND     = python -m pytest -rfE --junitxml=/tmp/g1w4-junit.xml -q
PLATFORM    = linux, CPython 3.12.14
RESULT      = 4487 passed, 47 skipped, 0 failed, 0 errors in 476.69s
JUNIT       = tests=4534 failures=0 errors=0 skipped=47 time=476.640
```

## Defects closed before that run

The first implementation read a header-only sheet as a zero-row frame, dropped
blank data rows after the dimension budget had counted them, ignored encoding
and `header=False`, and gave the CLI no way to pass `sheet=`.

| ID | Symptom | Cause | Fix | Regression | Commit |
|---|---|---|---|---|---|
| DEFECT-W4-01 | Header-only `.xlsx` loaded as an empty frame | `_frame` raised `EmptyInputError` only when the header was empty too | Refuse a header with no data rows, same message contract as delimited text | `test_a_header_only_sheet_is_empty_input` | `0e0fad8` |
| DEFECT-W4-02 | `encoding` and `header=False` were ignored | Only an explicit delimiter was refused | Refuse any non-default `LoadOptions`, as the columnar reader does | `test_text_options_are_refused` | `0e0fad8` |
| DEFECT-W4-03 | An all-empty data row disappeared from the frame | Rows with no values were filtered out after the preflight | Keep the row; its cells are missing | `test_a_blank_data_row_is_kept` | `0e0fad8` |
| DEFECT-W4-04 | A multi-sheet workbook could not name its sheet from the CLI, and the audit help omitted `.xlsx` | `load_table` was called without `sheet=` | `--sheet`, forwarded as `sheet=args.sheet` | `test_a_named_sheet_is_the_one_audited` | `0143671` |

## Mutation evidence

`scripts/g1_w4_mutations.py` — 9 dedicated mutations on the Excel seams.
Re-run on `0143671`, CPython 3.12.14. A kill requires the anchor to match once,
pytest exit 1, the expected reason in the failure text, and a restored tree.
Exit code alone is not a kill.

| ID | Weakening | Killing test | Result |
|---|---|---|---|
| M-W4-01 | macro-enabled archive accepted | test_a_macro_enabled_workbook_is_refused | KILLED |
| M-W4-02 | multi-sheet ambiguity resolved to first sheet | test_a_multi_sheet_workbook_is_ambiguous | KILLED |
| M-W4-03 | worksheet dimension preflight skipped | test_an_over_limit_sheet_is_refused_before_its_cells_are_read | KILLED |
| M-W4-04 | corrupt zip surfaces a bare error | test_a_corrupt_file_is_malformed_not_a_bare_error | KILLED |
| M-W4-05 | duplicate headers accepted | test_duplicate_headers_are_refused | KILLED |
| M-W4-06 | missing openpyxl → bare ImportError | test_an_xlsx_read_names_the_extra | KILLED |
| M-W4-07 | header-only worksheet accepted | test_a_header_only_sheet_is_empty_input | KILLED |
| M-W4-08 | text options accepted for an Excel file | test_text_options_are_refused | KILLED |
| M-W4-09 | CLI does not forward the named worksheet | test_a_named_sheet_is_the_one_audited | KILLED |

```text
G1_W4_MUTATIONS = 9 total, 9 KILLED, 0 SURVIVED, 0 HARNESS_ERROR, 0 TEST_ENVIRONMENT_ERROR
G1_W1 harness re-run on 0143671: 13/13 exit 1 (that harness records exit code, not a semantic reason)
G1_W2 harness re-run on 0143671: 13/13 KILLED
G1_W3 harness re-run on 0143671: 11/11 KILLED
```

## Compatibility

Artifact schema `1.1`, publication schema `1.0`, package version `0.1.0a1`,
semantic fingerprint: unchanged. One new optional dependency (`openpyxl`) under
the `excel` extra; the core `dependencies` list is untouched and the
dependency-contract tests pass. `load_table` gained a keyword-only `sheet`
argument, and the CLI gains optional `--sheet` (default `None`, the same as
omitting it). Both are backward-compatible extensions.

Certified counts are unchanged by the defect fixes: G1-04 was already
`SUPPORTED_AND_TESTED` on this branch. G1 = 16/26 = 61.54%. Overall =
108/232 = 46.55%. `origin/main` remains 107/232 = 46.12% until this branch
is integrated.

## CI

```text
BRANCH_CI = pending (this branch has not been pushed; no remote run exists)
MAIN_CI   = pending (not integrated)
```

Local evidence is not branch CI and is not main CI.

## Scope confirmation

G1-W5 (chunked profiling), date inference, encoding detection, databases, URLs,
generators: none started. No tag, release, or PyPI publication occurred.
