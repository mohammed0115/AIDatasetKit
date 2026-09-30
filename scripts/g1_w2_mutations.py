"""G1-W2 adversarial check: each weakening of resource governance must be killed.

Usage: python scripts/g1_w2_mutations.py <scratch-dir> [--rev REV]

The tree at REV (default HEAD) is exported with ``git archive`` into
``<scratch-dir>/tree``; the working tree is never touched. Each mutation is
applied there, its selector is run, and the original bytes are written back.

A mutation is KILLED only when every one of these holds:

- each (old -> new) replacement matched its anchor exactly once;
- pytest exited 1 with no collection or setup errors;
- every expected test failed, and its failure text contains the expected reason;
- afterwards every file in the tree hashes as it did before the mutation.

Exit 0 from pytest is SURVIVED. Anything else -- a missing anchor, an import
error, a test failing for a different reason, an unrestored file -- is
HARNESS_ERROR. If the unmutated tree does not pass every selector, or the
package would be imported from somewhere other than the exported tree, every
mutation is TEST_ENVIRONMENT_ERROR.

The G1-W1 mutations stay in ``scripts/ingestion_mutations.py``; none is repeated
here. The script exits 0 only when all mutations are KILLED.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

LIMITS = "tests/unit/test_ingestion_limits.py::TestIngestionLimits"
GUARANTEES = "tests/unit/test_ingestion_limits.py::TestResourceGovernanceGuarantees"
CLI_REFUSALS = "tests/integration/test_cli_ingestion.py::TestRefusalsPublishNothing"


@dataclass(frozen=True)
class Mutation:
    id: str
    weakening: str
    edits: tuple[tuple[str, str, str], ...]  # (file, old, new)
    selector: tuple[str, ...]
    expected: tuple[tuple[str, str], ...]  # (test id fragment, reason fragment)


FACADE_CHECK = (
    "        if limits is not None:\n"
    "            load_table(train, limits=limits)\n"
    "            if test is not None:\n"
    "                load_table(test, limits=limits)\n"
)

MUTATIONS = (
    Mutation(
        "M-W2-01", "file-byte preflight bypassed",
        (("aidatasetkit/ingestion/loader.py",
          "    if limits.max_source_bytes is not None:\n        size = path.stat().st_size",
          "    if False:\n        size = path.stat().st_size"),),
        (f"{GUARANTEES}::test_oversized_file_never_reaches_a_parser",),
        (("test_oversized_file_never_reaches_a_parser", "oversized input reached parsing"),),
    ),
    Mutation(
        "M-W2-02", "row boundary `>` changed to `>=`",
        (("aidatasetkit/ingestion/delimited.py",
          "data_rows > limits.max_rows:", "data_rows >= limits.max_rows:"),),
        (f"{GUARANTEES}::test_a_csv_exactly_at_each_limit_is_accepted",),
        (("test_a_csv_exactly_at_each_limit_is_accepted[max_rows-3]", "RowLimitError"),),
    ),
    Mutation(
        "M-W2-03", "DataFrame total-cell check neutralised",
        (("aidatasetkit/ingestion/loader.py",
          "if limits.max_cells is not None and _cells_exceed(rows, columns, limits.max_cells):",
          "if False and _cells_exceed(rows, columns, limits.max_cells):"),),
        (f"{GUARANTEES}::test_dataframe_over_the_cell_limit_is_refused_with_its_shape",),
        (("test_dataframe_over_the_cell_limit_is_refused_with_its_shape", "DID NOT RAISE"),),
    ),
    Mutation(
        "M-W2-04", "records allowed one beyond max_records",
        (("aidatasetkit/ingestion/loader.py",
          "len(records) > limits.max_records:", "len(records) > limits.max_records + 1:"),),
        (f"{GUARANTEES}::test_one_record_over_the_limit_is_refused",),
        (("test_one_record_over_the_limit_is_refused", "DID NOT RAISE"),),
    ),
    Mutation(
        "M-W2-05", "facade limit check moved after target detection",
        (("aidatasetkit/facade/facade.py", FACADE_CHECK + '        self._require_frame(train, "train")',
          '        self._require_frame(train, "train")'),
         ("aidatasetkit/facade/facade.py", "        self._reset_everything()\n        self._train = train",
          FACADE_CHECK + "        self._reset_everything()\n        self._train = train")),
        (f"{GUARANTEES}::test_facade_refuses_before_target_detection",),
        (("test_facade_refuses_before_target_detection", "target detection ran before the resource refusal"),),
    ),
    Mutation(
        "M-W2-06", "rejected record values added to the row-limit message",
        (("aidatasetkit/ingestion/delimited.py",
          'f"{path.name} exceeds the row limit ({data_rows} > {limits.max_rows}).",',
          'f"{path.name} exceeds the row limit ({data_rows} > {limits.max_rows}): {record}.",'),),
        (f"{GUARANTEES}::test_rejected_cell_values_never_appear_in_the_error",),
        (("test_rejected_cell_values_never_appear_in_the_error", "RowLimitError leaked a rejected value"),),
    ),
    Mutation(
        "M-W2-07", "a run directory is created after a ResourceLimitError",
        (("aidatasetkit/cli/main.py",
          "    except AIDatasetKitError as error:\n        return _fail(",
          "    except AIDatasetKitError as error:\n"
          '        if any(c.__name__ == "ResourceLimitError" for c in type(error).__mro__):\n'
          '            (args.output / "runs" / "rejected").mkdir(parents=True, exist_ok=True)\n'
          "        return _fail("),),
        (f"{CLI_REFUSALS}::test_a_resource_refusal_leaves_the_previous_run_as_it_was",),
        (("test_a_resource_refusal_leaves_the_previous_run_as_it_was", "rejected"),),
    ),
    Mutation(
        "M-W2-08", "row refusal deferred until the whole file is scanned",
        (("aidatasetkit/ingestion/delimited.py",
          "    anything = False\n    previous_field_limit",
          "    anything = False\n    row_error = None\n    previous_field_limit"),
         ("aidatasetkit/ingestion/delimited.py",
          "                        raise RowLimitError(\n", "                        row_error = RowLimitError(\n"),
         ("aidatasetkit/ingestion/delimited.py",
          "                        anything = True\n        except csv.Error as error:",
          "                        anything = True\n"
          "                if row_error is not None:\n"
          "                    raise row_error\n"
          "        except csv.Error as error:")),
        (f"{GUARANTEES}::test_row_refusal_reads_no_record_past_the_decisive_one",),
        (("test_row_refusal_reads_no_record_past_the_decisive_one", "records after the limit was crossed"),),
    ),
    Mutation(
        "M-W2-09", "CLI --max-rows default differs from IngestionLimits",
        (("aidatasetkit/cli/main.py",
          '"Maximum data rows.", _DEFAULT_INGESTION_LIMITS.max_rows)',
          '"Maximum data rows.", 2_000_000)'),),
        (f"{GUARANTEES}::test_every_cli_default_equals_the_library_default",),
        (("test_every_cli_default_equals_the_library_default", "'max_rows': (2000000, 1000000)"),),
    ),
    Mutation(
        "M-W2-10", "Python max_rows default made unlimited",
        (("aidatasetkit/ingestion/types.py",
          "    max_rows: int | None = 1_000_000\n", "    max_rows: int | None = None\n"),),
        (f"{GUARANTEES}::test_every_default_is_the_documented_finite_integer",),
        (("test_every_default_is_the_documented_finite_integer", "'max_rows': None"),),
    ),
    Mutation(
        "M-W2-11", "cell budget computed in wrapping 32-bit arithmetic",
        (("aidatasetkit/ingestion/types.py",
          "    return rows > max_cells // columns\n",
          "    return (rows * columns) & 0xFFFFFFFF > max_cells\n"),),
        (f"{GUARANTEES}::test_cell_budget_is_exact_for_huge_logical_shapes",),
        (("test_cell_budget_is_exact_for_huge_logical_shapes[65537-65536-10000000-True]", "assert False is True"),),
    ),
    Mutation(
        "M-W2-12", "DataFrame deep-copied during validation and the copy returned",
        (("aidatasetkit/ingestion/loader.py",
          "    return LoadedTable(frame=frame, metadata=_in_memory(SourceKind.DATAFRAME, frame, ()))",
          "    frame = frame.copy(deep=True)\n"
          "    return LoadedTable(frame=frame, metadata=_in_memory(SourceKind.DATAFRAME, frame, ()))"),),
        (f"{GUARANTEES}::test_dataframe_is_returned_itself_with_order_and_dtypes",
         f"{LIMITS}::test_dataframe_is_not_copied_or_mutated"),
        (("test_dataframe_is_returned_itself_with_order_and_dtypes", "DataFrame copied during validation"),
         ("test_dataframe_is_not_copied_or_mutated", " is ")),
    ),
    Mutation(
        "M-W2-13", "max_field_length no longer applied by validation",
        (("aidatasetkit/ingestion/delimited.py",
          "    if limits.max_field_length is not None:\n        csv.field_size_limit(limits.max_field_length)",
          "    if False:\n        csv.field_size_limit(limits.max_field_length)"),),
        (f"{GUARANTEES}::test_oversized_field_is_refused_before_pandas",
         f"{LIMITS}::test_rows_columns_cells_and_field_length_are_guarded"),
        (("test_oversized_field_is_refused_before_pandas", "oversized field reached pandas"),
         ("test_rows_columns_cells_and_field_length_are_guarded", "DID NOT RAISE")),
    ),
)


def export(rev: str, tree: Path) -> str:
    sha = subprocess.run(["git", "rev-parse", rev], capture_output=True, text=True, check=True).stdout.strip()
    archive = subprocess.run(["git", "archive", "--format=tar", sha], capture_output=True, check=True).stdout
    if tree.exists():
        shutil.rmtree(tree)
    tree.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(tree, filter="data")
    return sha


def manifest(tree: Path) -> dict[str, str]:
    return {
        str(path.relative_to(tree)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(tree.rglob("*"))
        if path.is_file()
    }


def environment(tree: Path) -> dict[str, str]:
    return dict(os.environ, PYTHONPATH=str(tree), PYTHONDONTWRITEBYTECODE="1")


def run_pytest(tree: Path, selector: tuple[str, ...], junit: Path) -> int:
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", f"--junitxml={junit}", *selector]
    return subprocess.run(command, cwd=tree, env=environment(tree), capture_output=True).returncode


def failures(junit: Path) -> tuple[dict[str, str], int]:
    """Failed test ids mapped to their failure text, and the number of errored cases."""
    root = ET.parse(junit).getroot()
    failed: dict[str, str] = {}
    errored = 0
    for case in root.iter("testcase"):
        name = f"{case.get('classname')}::{case.get('name')}"
        for failure in case.findall("failure"):
            failed[name] = (failure.get("message") or "") + "\n" + (failure.text or "")
        errored += len(case.findall("error"))
    return failed, errored


def main() -> int:
    scratch = Path(sys.argv[1]).resolve()
    rev = sys.argv[sys.argv.index("--rev") + 1] if "--rev" in sys.argv else "HEAD"
    tree = scratch / "tree"
    sha = export(rev, tree)
    pristine = manifest(tree)
    results = []

    probe = subprocess.run(
        [sys.executable, "-c", "import aidatasetkit; print(aidatasetkit.__file__)"],
        cwd=tree, env=environment(tree), capture_output=True, text=True,
    )
    imported_from_tree = Path(probe.stdout.strip()).resolve().is_relative_to(tree)
    selectors = tuple(dict.fromkeys(s for m in MUTATIONS for s in m.selector))
    baseline_exit = run_pytest(tree, selectors, scratch / "baseline.xml")
    environment_ok = imported_from_tree and baseline_exit == 0 and manifest(tree) == pristine

    for mutation in MUTATIONS:
        record = {"id": mutation.id, "weakening": mutation.weakening,
                  "files": sorted({edit[0] for edit in mutation.edits}),
                  "selector": list(mutation.selector), "exit": None, "restored": None, "detail": ""}
        if not environment_ok:
            record.update(result="TEST_ENVIRONMENT_ERROR",
                          detail=f"imported_from_tree={imported_from_tree} baseline_exit={baseline_exit}")
            results.append(record)
            continue
        originals = {rel: (tree / rel).read_bytes() for rel in record["files"]}
        applied = True
        for rel, old, new in mutation.edits:
            path = tree / rel
            text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
            if text.count(old) != 1:
                applied = False
                record["detail"] = f"anchor matched {text.count(old)} times in {rel}"
                break
            path.write_bytes(text.replace(old, new).encode("utf-8"))
        if applied:
            junit = scratch / f"{mutation.id}.xml"
            record["exit"] = run_pytest(tree, mutation.selector, junit)
        for rel, content in originals.items():
            (tree / rel).write_bytes(content)
        record["restored"] = manifest(tree) == pristine

        if not applied:
            record["result"] = "HARNESS_ERROR"
        elif not record["restored"]:
            record.update(result="HARNESS_ERROR", detail="tree differs from the export after restoration")
        elif record["exit"] == 0:
            record["result"] = "SURVIVED"
        elif record["exit"] != 1:
            record.update(result="HARNESS_ERROR", detail=f"pytest exit {record['exit']} is not a test failure")
        else:
            failed, errored = failures(junit)
            missing = [
                f"{test}: {reason!r}" for test, reason in mutation.expected
                if not any(test in name and reason in text for name, text in failed.items())
            ]
            record["killed_by"] = sorted(name.rsplit("::", 1)[-1] for name in failed)
            if errored or missing:
                record.update(result="HARNESS_ERROR", detail=f"errored={errored} unmet={missing}")
            else:
                record["result"] = "KILLED"
        results.append(record)

    summary = {
        "rev": sha, "python": sys.version.split()[0], "baseline_exit": baseline_exit,
        "imported_from_tree": imported_from_tree, "total": len(results),
        **{state: sum(r["result"] == state for r in results)
           for state in ("KILLED", "SURVIVED", "HARNESS_ERROR", "TEST_ENVIRONMENT_ERROR")},
        "mutations": results,
    }
    (scratch / "g1_w2_mutations.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"rev={sha} python={summary['python']} baseline_exit={baseline_exit} imported_from_tree={imported_from_tree}")
    for r in results:
        print(f"{r['id']}  {r['result']:22s} exit={r['exit']} restored={r['restored']}  {r['weakening']}  {r['detail']}")
    print({state: summary[state] for state in ("total", "KILLED", "SURVIVED", "HARNESS_ERROR", "TEST_ENVIRONMENT_ERROR")})
    return 0 if summary["KILLED"] == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
