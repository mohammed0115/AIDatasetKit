# G1-W4 — Excel .xlsx reader

Evidence for the G1-W4 wave of `docs/AIDATASETKIT_EXECUTION_ROADMAP.md`
(audit row G1-04). Every claim below names the run that proves it.

## Scope and baseline

```text
PRE_CLOSURE_SHA  = dc70d8c60a3defdc35a805fbe7b6996b2fdb800c (main = origin/main after G1-W3)
BRANCH           = g1-w4-xlsx-reader
FINAL_BRANCH_SHA = <filled at push>
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

Full suite from the exact committed state:

```text
TESTED_SHA  = <filled after the run>
RESULT      = <filled after the run>
JUNIT       = <filled after the run>
```

## Mutation evidence

`scripts/g1_w4_mutations.py` — 6 dedicated mutations on the new seams:

| ID | Weakening | Killing test | Result |
|---|---|---|---|
| M-W4-01 | macro-enabled archive accepted | test_a_macro_enabled_workbook_is_refused | KILLED |
| M-W4-02 | multi-sheet ambiguity resolved to first sheet | test_a_multi_sheet_workbook_is_ambiguous | KILLED |
| M-W4-03 | worksheet dimension preflight skipped | test_an_over_limit_sheet_is_refused_before_its_cells_are_read | KILLED |
| M-W4-04 | corrupt zip surfaces a bare error | test_a_corrupt_file_is_malformed_not_a_bare_error | KILLED |
| M-W4-05 | duplicate headers accepted | test_duplicate_headers_are_refused | KILLED |
| M-W4-06 | missing openpyxl → bare ImportError | test_an_xlsx_read_names_the_extra | KILLED |

```text
G1_W4_MUTATIONS = 6 total, 6 KILLED, 0 SURVIVED, 0 HARNESS_ERROR, 0 TEST_ENVIRONMENT_ERROR
G1_W1 harness re-run on this tree: 13/13
G1_W2 harness re-run on this tree: 13/13 KILLED
G1_W3 harness re-run on this tree: 11/11 KILLED
```

## Compatibility

Artifact schema `1.1`, publication schema `1.0`, package version `0.1.0a1`,
semantic fingerprint: unchanged. One new optional dependency (`openpyxl`) under
the `excel` extra; the core `dependencies` list is untouched and the
dependency-contract tests pass. `load_table` gained a keyword-only `sheet`
argument -- a backward-compatible extension (existing callers pass nothing).

## CI

```text
BRANCH_CI = pending (recorded once observed on FINAL_BRANCH_SHA)
MAIN_CI   = pending
```

## Scope confirmation

G1-W5 (chunked profiling), date inference, encoding detection, databases, URLs,
generators: none started. No tag, release, or PyPI publication occurred.
