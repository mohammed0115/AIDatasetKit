# G1-W2 dedicated mutation run

```text
COMMAND   = python scripts/g1_w2_mutations.py <scratch> --rev 4252c521865a27cbc672e997dd0a7e519714f43b
TREE      = git archive of 4252c521865a27cbc672e997dd0a7e519714f43b (working tree untouched)
PLATFORM  = Windows 11, Python 3.12.3, pytest 8.3.3, pandas 2.3.3, NumPy 2.4.1
DATE      = 2026-09-30
BASELINE  = every selector passed on the unmutated export (exit 0); package imported from the export
EXIT      = 0
RESULT    = {'total': 13, 'KILLED': 13, 'SURVIVED': 0, 'HARNESS_ERROR': 0, 'TEST_ENVIRONMENT_ERROR': 0}
```

| ID | Result | pytest exit | Restored | Failing tests (reason required by the harness) |
|---|---|---:|---|---|
| M-W2-01 | KILLED | 1 | yes | `test_oversized_file_never_reaches_a_parser` ("oversized input reached parsing") |
| M-W2-02 | KILLED | 1 | yes | `test_a_csv_exactly_at_each_limit_is_accepted[max_rows-3]` ("RowLimitError") |
| M-W2-03 | KILLED | 1 | yes | `test_dataframe_over_the_cell_limit_is_refused_with_its_shape` ("DID NOT RAISE") |
| M-W2-04 | KILLED | 1 | yes | `test_one_record_over_the_limit_is_refused` ("DID NOT RAISE") |
| M-W2-05 | KILLED | 1 | yes | `test_facade_refuses_before_target_detection` ("target detection ran before the resource refusal") |
| M-W2-06 | KILLED | 1 | yes | `test_rejected_cell_values_never_appear_in_the_error` ("RowLimitError leaked a rejected value") |
| M-W2-07 | KILLED | 1 | yes | `test_a_resource_refusal_leaves_the_previous_run_as_it_was` ("rejected" run directory) |
| M-W2-08 | KILLED | 1 | yes | `test_row_refusal_reads_no_record_past_the_decisive_one` ("records after the limit was crossed") |
| M-W2-09 | KILLED | 1 | yes | `test_every_cli_default_equals_the_library_default` ("'max_rows': (2000000, 1000000)") |
| M-W2-10 | KILLED | 1 | yes | `test_every_default_is_the_documented_finite_integer` ("'max_rows': None") |
| M-W2-11 | KILLED | 1 | yes | `test_cell_budget_is_exact_for_huge_logical_shapes[65537-65536-10000000-True]` ("assert False is True"); also the `2**53 + 1` and `2**80 - 1` cases |
| M-W2-12 | KILLED | 1 | yes | `test_dataframe_is_returned_itself_with_order_and_dtypes` ("DataFrame copied during validation"); `test_dataframe_is_not_copied_or_mutated` (identity) |
| M-W2-13 | KILLED | 1 | yes | `test_oversized_field_is_refused_before_pandas` ("oversized field reached pandas"); `test_rows_columns_cells_and_field_length_are_guarded` ("DID NOT RAISE") |

After the run: `git status --short` empty, `git diff --check` exit 0.

## Harness controls (same SHA)

Four deliberately bad mutations, run through the same harness to show it does
not count a non-zero exit as a kill:

| Control | Expected | Observed |
|---|---|---|
| comment-only edit (equivalent mutant) | SURVIVED | SURVIVED, exit 0 |
| real weakening, wrong expected reason | HARNESS_ERROR | HARNESS_ERROR, exit 1, reason unmet |
| anchor absent | HARNESS_ERROR | HARNESS_ERROR, anchor matched 0 times |
| syntax error in `loader.py` | HARNESS_ERROR | HARNESS_ERROR, pytest exit 4 |

Run with `--rev 02057aa` (selectors absent there): 13 TEST_ENVIRONMENT_ERROR,
baseline exit 4, 0 KILLED.

## Historical G1-W1 script (reported separately, not counted above)

`python scripts/ingestion_mutations.py <export of 4252c52> <scratch>`: all 13
entries exit 1. Entries 1-8 are the G1-W1 ingestion mutations; entries 9-13 are
the earlier G1-W2 exit-code-only checks. Neither set counts toward the 13
dedicated G1-W2 mutations.
