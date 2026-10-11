"""G1-W7 adversarial check: each weakening of worksheet selection must be killed.

Usage: python scripts/g1_w7_mutations.py <scratch-dir> [--rev REV]

The tree at REV is exported with ``git archive``. Each weakening is applied
there, its selector run, and the original bytes written back. A mutation is
KILLED only when every replacement matched its anchor exactly once, pytest
exited 1 with no errored cases, every expected test failed with its expected
reason, and the tree hashes as before. A non-zero exit alone is not a kill.
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

EXCEL_TESTS = "tests/unit/test_ingestion_excel.py"
SQLITE_TESTS = "tests/unit/test_ingestion_sqlite.py"
EXCEL = "aidatasetkit/ingestion/excel.py"
LOADER = "aidatasetkit/ingestion/loader.py"
BUILDER = "aidatasetkit/evidence/builder.py"
SQLITE = "aidatasetkit/ingestion/sqlite.py"
PROJECT = "pyproject.toml"


@dataclass(frozen=True)
class Mutation:
    id: str
    weakening: str
    edits: tuple[tuple[str, str, str], ...]
    selector: tuple[str, ...]
    expected: tuple[tuple[str, str], ...]


MUTATIONS = (
    Mutation(
        "M-W7-01", "sheet argument ignored",
        ((EXCEL,
          "    if sheet is not None:\n        return _named_worksheet(path, wb, eligible, names, sheet)\n",
          "    if False:\n        return _named_worksheet(path, wb, eligible, names, sheet)\n"),),
        (f"{EXCEL_TESTS}::TestSheetSelection::test_an_exact_non_first_worksheet_is_read",),
        (("test_an_exact_non_first_worksheet_is_read", "sheet argument ignored"),),
    ),
    Mutation(
        "M-W7-02", "first sheet always selected",
        ((EXCEL,
          "    raise MalformedInputError(\n"
          "        f\"{path.name} has {len(names)} sheets ({', '.join(names)}). Choosing one \"\n"
          "        \"would be a guess; pass sheet= to name it.\"\n"
          "    )\n",
          "    return eligible[0]\n"),),
        (f"{EXCEL_TESTS}::TestSheetSelection::test_several_worksheets_require_a_name",),
        (("test_several_worksheets_require_a_name", "first sheet always selected"),),
    ),
    Mutation(
        "M-W7-03", "missing sheet silently falls back",
        ((EXCEL,
          "    raise MalformedInputError(\n"
          "        f\"{path.name} has no sheet named {sheet!r}. Sheets: {', '.join(names)}.\"\n"
          "    )\n",
          "    return eligible[0]\n"),),
        (f"{EXCEL_TESTS}::TestSheetSelection::test_an_unknown_sheet_name_does_not_fall_back",),
        (("test_an_unknown_sheet_name_does_not_fall_back", "missing sheet silently falls back"),),
    ),
    Mutation(
        "M-W7-04", "automatic single sheet not recorded",
        ((EXCEL,
          "    if requested is None:\n        recorded = chosen.title  # automatic selection is recorded too\n",
          "    if requested is None:\n        return None\n"),),
        (f"{EXCEL_TESTS}::TestSheetEvidence::test_an_automatic_sheet_is_recorded",),
        (("test_an_automatic_sheet_is_recorded", "automatic single sheet not recorded"),),
    ),
    Mutation(
        "M-W7-05", "explicit selector omitted from evidence",
        ((EXCEL,
          '    return SourceSelector(kind="sheet", name=recorded)\n',
          "    return None\n"),),
        (f"{EXCEL_TESTS}::TestSheetEvidence::test_an_explicit_sheet_is_recorded",),
        (("test_an_explicit_sheet_is_recorded", "explicit selector omitted from evidence"),),
    ),
    Mutation(
        "M-W7-06", "selector omitted from config fingerprint",
        ((BUILDER,
          '    recorded["source_selector"] = {"kind": selector.kind, "name": selector.name}\n',
          '    if selector.kind != "sheet":\n'
          '        recorded["source_selector"] = {"kind": selector.kind, "name": selector.name}\n'),),
        (f"{EXCEL_TESTS}::TestSheetEvidence::test_the_selected_sheet_enters_the_config_fingerprint",),
        (("test_the_selected_sheet_enters_the_config_fingerprint", "selector omitted from config fingerprint"),),
    ),
    Mutation(
        "M-W7-07", "two sheets share one config fingerprint",
        ((BUILDER,
          '    recorded["source_selector"] = {"kind": selector.kind, "name": selector.name}\n',
          '    recorded["source_selector"] = {"kind": selector.kind, "name": ""}\n'),),
        (f"{EXCEL_TESTS}::TestSheetEvidence::test_two_sheet_names_change_the_config_fingerprint",),
        (("test_two_sheet_names_change_the_config_fingerprint", "two sheets share one config fingerprint"),),
    ),
    Mutation(
        "M-W7-08", "sheet= accepted for non-XLSX input",
        ((LOADER,
          "    if fmt is not TableFormat.XLSX and sheet is not None:\n",
          "    if False and fmt is not TableFormat.XLSX and sheet is not None:\n"),),
        (f"{EXCEL_TESTS}::TestSheetOptionIsolation::test_sheet_is_refused_outside_xlsx",),
        (("test_sheet_is_refused_outside_xlsx", "sheet= accepted for non-XLSX input"),),
    ),
    Mutation(
        "M-W7-09", "chartsheet treated as worksheet",
        ((EXCEL,
          "    return [ws for ws in wb.worksheets if _worksheet_is_eligible(ws)]\n",
          "    return [ws for ws in wb._sheets if _worksheet_is_eligible(ws)]\n"),),
        (f"{EXCEL_TESTS}::TestSheetSelection::test_a_worksheet_beside_chartsheets_is_selected",),
        (("test_a_worksheet_beside_chartsheets_is_selected", "chartsheet treated as worksheet"),),
    ),
    Mutation(
        "M-W7-10", "hidden worksheet removed from eligibility",
        ((EXCEL,
          '    if state == "hidden":\n        return True\n',
          '    if state == "hidden":\n        return False\n'),),
        (f"{EXCEL_TESTS}::TestSheetSelection::test_a_hidden_worksheet_stays_eligible",),
        (("test_a_hidden_worksheet_stays_eligible", "hidden worksheet removed from eligibility"),),
    ),
    Mutation(
        "M-W7-11", "row limit ignored",
        ((EXCEL,
          "    if limits.max_rows is not None and rows - 1 > limits.max_rows:\n",
          "    if False and limits.max_rows is not None and rows - 1 > limits.max_rows:\n"),),
        (f"{EXCEL_TESTS}::TestSheetLimits::test_the_selected_sheet_row_limit_is_enforced",),
        (("test_the_selected_sheet_row_limit_is_enforced", "row limit ignored"),),
    ),
    Mutation(
        "M-W7-12", "column or cell limit ignored",
        ((EXCEL,
          "    if limits.max_cells is not None and _cells_exceed(rows, columns, limits.max_cells):\n",
          "    if False and limits.max_cells is not None and _cells_exceed(rows, columns, limits.max_cells):\n"),),
        (f"{EXCEL_TESTS}::TestSheetLimits::test_the_selected_sheet_cell_limit_is_enforced",),
        (("test_the_selected_sheet_cell_limit_is_enforced", "column or cell limit ignored"),),
    ),
    Mutation(
        "M-W7-13", "max_field_length ignored",
        ((EXCEL,
          "        return len(value) > limit\n",
          "        return False\n"),),
        (f"{EXCEL_TESTS}::TestSheetLimits::test_an_oversized_textual_cell_is_refused",),
        (("test_an_oversized_textual_cell_is_refused", "max_field_length ignored"),),
    ),
    Mutation(
        "M-W7-14", "cells/DataFrame published before limit refusal",
        ((EXCEL,
          "        rows = _rows(ws, path)\n        _reject_long_fields(path, rows, limits)\n",
          "        rows = _rows(ws, path)\n        _frame(path, rows)\n        _reject_long_fields(path, rows, limits)\n"),),
        (f"{EXCEL_TESTS}::TestSheetLimits::test_field_length_is_refused_before_the_frame",),
        (("test_field_length_is_refused_before_the_frame", "cells/DataFrame published before limit refusal"),),
    ),
    Mutation(
        "M-W7-15", "openpyxl becomes a core dependency",
        ((PROJECT,
          '    "scikit-learn>=1.6.1",\n]\n',
          '    "scikit-learn>=1.6.1",\n    "openpyxl>=3.1.0",\n]\n'),),
        (f"{EXCEL_TESTS}::TestSheetSchema::test_openpyxl_is_not_a_core_dependency",),
        (("test_openpyxl_is_not_a_core_dependency", "openpyxl becomes a core dependency"),),
    ),
    Mutation(
        "M-W7-16", "default single-sheet behavior changes",
        ((EXCEL,
          "    if len(names) == 1:\n        return eligible[0]\n",
          "    if False:\n        return eligible[0]\n"),),
        (f"{EXCEL_TESTS}::TestSheetSelection::test_one_worksheet_is_selected_automatically",),
        (("test_one_worksheet_is_selected_automatically", "default single-sheet behavior changes"),),
    ),
    Mutation(
        "M-W7-17", "SQLite table selector behavior changes",
        ((SQLITE,
          '    return _frame(names, rows), SourceSelector(kind="table", name=chosen)\n',
          '    return _frame(names, rows), SourceSelector(kind="sheet", name=chosen)\n'),),
        (f"{SQLITE_TESTS}::TestSelection::test_one_table_is_selected",),
        (("test_one_table_is_selected", "SQLite table selector behavior changes"),),
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
    command = [
        sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
        f"--junitxml={junit}", *selector,
    ]
    return subprocess.run(command, cwd=tree, env=environment(tree), capture_output=True).returncode


def failures(junit: Path) -> tuple[dict[str, str], int]:
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
    selectors = tuple(dict.fromkeys(item for mutation in MUTATIONS for item in mutation.selector))
    baseline_exit = run_pytest(tree, selectors, scratch / "baseline.xml")
    environment_ok = imported_from_tree and baseline_exit == 0 and manifest(tree) == pristine

    for mutation in MUTATIONS:
        record = {
            "id": mutation.id, "weakening": mutation.weakening,
            "files": sorted({edit[0] for edit in mutation.edits}),
            "selector": list(mutation.selector), "exit": None, "restored": None, "detail": "",
        }
        if not environment_ok:
            record.update(
                result="TEST_ENVIRONMENT_ERROR",
                detail=f"imported_from_tree={imported_from_tree} baseline_exit={baseline_exit}",
            )
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
        **{state: sum(item["result"] == state for item in results)
           for state in ("KILLED", "SURVIVED", "HARNESS_ERROR", "TEST_ENVIRONMENT_ERROR")},
        "mutations": results,
    }
    (scratch / "g1_w7_mutations.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"rev={sha} python={summary['python']} baseline_exit={baseline_exit} "
        f"imported_from_tree={imported_from_tree}"
    )
    for record in results:
        print(
            f"{record['id']}  {record['result']:22s} exit={record['exit']} "
            f"restored={record['restored']}  {record['weakening']}  {record['detail']}"
        )
    print({state: summary[state] for state in ("total", "KILLED", "SURVIVED", "HARNESS_ERROR", "TEST_ENVIRONMENT_ERROR")})
    return 0 if summary["KILLED"] == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
