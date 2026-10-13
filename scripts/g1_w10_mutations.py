"""G1-W10 adversarial check: each weakening of .xls ingestion must be killed.

Usage: python scripts/g1_w10_mutations.py <scratch-dir> [--rev REV]

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

TESTS = "tests/unit/test_ingestion_xls.py"
LOADER = "aidatasetkit/ingestion/loader.py"
XLS = "aidatasetkit/ingestion/xls.py"
BUILDER = "aidatasetkit/evidence/builder.py"
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
        "M-W10-01", "missing xlrd accepted",
        ((XLS,
          "    if spec is None:\n",
          "    if False and spec is None:\n"),),
        (f"{TESTS}::TestRefusals::test_a_missing_xlrd_names_the_extra",),
        (("test_a_missing_xlrd_names_the_extra", "missing xlrd accepted"),),
    ),
    Mutation(
        "M-W10-02", "xlrd becomes core dependency",
        ((PROJECT,
          '    "scikit-learn>=1.6.1",\n',
          '    "scikit-learn>=1.6.1",\n    "xlrd>=2.0.2,<3",\n'),),
        (f"{TESTS}::TestRefusals::test_xlrd_is_not_a_core_dependency",),
        (("test_xlrd_is_not_a_core_dependency", "xlrd becomes core dependency"),),
    ),
    Mutation(
        "M-W10-03", "xlsx dispatched to xlrd",
        ((LOADER,
          "        frame, selector = read_excel(path, fmt, options, limits, sheet=sheet)\n",
          "        frame, selector = read_xls(path, fmt, options, limits, sheet=sheet)\n"),),
        (f"{TESTS}::TestRefusals::test_xlsx_still_uses_openpyxl",),
        (("test_xlsx_still_uses_openpyxl", "xlsx dispatched to xlrd"),),
    ),
    Mutation(
        "M-W10-04", "disguised non-BIFF file accepted",
        ((XLS,
          "    _reject_disguised(path)\n",
          "    pass  # signature check removed\n"),),
        (f"{TESTS}::TestRefusals::test_html_xml_and_zip_payloads_are_refused",),
        (("test_html_xml_and_zip_payloads_are_refused", "disguised non-BIFF file accepted"),),
    ),
    Mutation(
        "M-W10-05", "byte limit checked after open",
        ((LOADER,
          "        if size > limits.max_source_bytes:\n",
          "        if False and size > limits.max_source_bytes:\n"),),
        (f"{TESTS}::TestLimits::test_the_byte_limit_is_enforced_before_open",),
        (("test_the_byte_limit_is_enforced_before_open", "byte limit checked after open"),),
    ),
    Mutation(
        "M-W10-06", "row limit ignored",
        ((XLS,
          "    if limits.max_rows is not None and data_rows > limits.max_rows:  # selected xls rows\n",
          "    if False and limits.max_rows is not None and data_rows > limits.max_rows:  # selected xls rows\n"),),
        (f"{TESTS}::TestLimits::test_max_rows_is_enforced",),
        (("test_max_rows_is_enforced", "row limit ignored"),),
    ),
    Mutation(
        "M-W10-07", "column limit ignored",
        ((XLS,
          "    if limits.max_columns is not None and columns > limits.max_columns:  # selected xls columns\n",
          "    if False and limits.max_columns is not None and columns > limits.max_columns:  # selected xls columns\n"),),
        (f"{TESTS}::TestLimits::test_max_columns_is_enforced",),
        (("test_max_columns_is_enforced", "column limit ignored"),),
    ),
    Mutation(
        "M-W10-08", "cell limit ignored",
        ((XLS,
          "    if limits.max_cells is not None and _cells_exceed(rows, columns, limits.max_cells):  # selected xls rectangle\n",
          "    if False and limits.max_cells is not None and _cells_exceed(rows, columns, limits.max_cells):  # selected xls rectangle\n"),),
        (f"{TESTS}::TestLimits::test_max_cells_is_enforced",),
        (("test_max_cells_is_enforced", "cell limit ignored"),),
    ),
    Mutation(
        "M-W10-09", "field-length limit ignored",
        ((XLS,
          "    _reject_long_fields(path, rows, limits)  # selected xls cells\n",
          "    if False:\n        _reject_long_fields(path, rows, limits)  # selected xls cells\n"),),
        (f"{TESTS}::TestLimits::test_max_field_length_is_enforced",),
        (("test_max_field_length_is_enforced", "field length limit ignored"),),
    ),
    Mutation(
        "M-W10-10", "first worksheet silently selected from multiple sheets",
        ((XLS,
          "    if len(names) == 1:\n",
          "    if len(names) >= 1:\n"),),
        (f"{TESTS}::TestSheetSelection::test_several_worksheets_require_a_name",),
        (("test_several_worksheets_require_a_name",
          "first worksheet silently selected from multiple sheets"),),
    ),
    Mutation(
        "M-W10-11", "case-folded sheet matching",
        ((XLS,
          "            if title == sheet:\n",
          "            if title.casefold() == sheet.casefold():\n"),),
        (f"{TESTS}::TestSheetSelection::test_matching_is_case_sensitive",),
        (("test_matching_is_case_sensitive", "case-folded sheet matching"),),
    ),
    Mutation(
        "M-W10-12", "selected sheet omitted from evidence",
        ((XLS,
          '    return _frame(path, rows), SourceSelector(kind="sheet", name=title)  # xls sheet evidence\n',
          '    return _frame(path, rows), SourceSelector(kind="sheet", name="")  # xls sheet evidence\n'),),
        (f"{TESTS}::TestEvidence::test_the_selected_sheet_is_recorded",),
        (("test_the_selected_sheet_is_recorded", "selected sheet omitted from evidence"),),
    ),
    Mutation(
        "M-W10-13", "selected sheet omitted from config fingerprint",
        ((BUILDER,
          "    if selector is None:\n        return settings\n",
          "    if selector is None or True:\n        return settings\n"),),
        (f"{TESTS}::TestEvidence::test_the_selected_sheet_enters_the_config_fingerprint",),
        (("test_the_selected_sheet_enters_the_config_fingerprint",
          "selected sheet omitted from config fingerprint"),),
    ),
    Mutation(
        "M-W10-14", "formula expression exposed or treated as value",
        ((XLS,
          "        return cell.value  # cached number, never the formula expression\n",
          '        return f"={cell.value:g}"  # cached number, never the formula expression\n'),),
        (f"{TESTS}::TestCellTypes::test_a_cached_formula_result_is_the_value",),
        (("test_a_cached_formula_result_is_the_value",
          "formula expression exposed or treated as value"),),
    ),
    Mutation(
        "M-W10-15", "Excel error cell silently accepted",
        ((XLS,
          "    if cell.ctype == xlrd.XL_CELL_ERROR:\n",
          "    if False and cell.ctype == xlrd.XL_CELL_ERROR:\n"),),
        (f"{TESTS}::TestCellTypes::test_an_excel_error_cell_is_refused_without_its_code",),
        (("test_an_excel_error_cell_is_refused_without_its_code",
          "Excel error cell silently accepted"),),
    ),
    Mutation(
        "M-W10-16", "refusal publishes partial output",
        ((XLS,
          "            raise  # xls refusal publishes nothing\n",
          '            return pd.DataFrame({"a": [1]}), SourceSelector(kind="sheet", name="a")\n'),),
        (f"{TESTS}::TestLimits::test_a_refusal_publishes_nothing",),
        (("test_a_refusal_publishes_nothing", "refusal publishes partial output"),),
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
    (scratch / "g1_w10_mutations.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
