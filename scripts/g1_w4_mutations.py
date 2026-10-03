"""G1-W4 adversarial check: each weakening of the Excel-reader guards must be killed.

Usage: python scripts/g1_w4_mutations.py <scratch-dir> [--rev REV]

Same discipline as the G1-W2/G1-W3 harnesses: the tree at REV is exported with
``git archive``; each weakening is applied there, its selector run, the original
bytes written back. A mutation is KILLED only when every replacement matched its
anchor exactly once, pytest exited 1 with no errored cases, every expected test
failed with its expected reason, and the tree hashes as before.
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
ABSENCE_TESTS = "tests/integration/test_ingestion_without_openpyxl.py"

EXCEL = "aidatasetkit/ingestion/excel.py"


@dataclass(frozen=True)
class Mutation:
    id: str
    weakening: str
    edits: tuple[tuple[str, str, str], ...]
    selector: tuple[str, ...]
    expected: tuple[tuple[str, str], ...]


MUTATIONS = (
    Mutation(
        "M-W4-01", "macro-enabled archive accepted",
        ((EXCEL,
          '    if "xl/vbaProject.bin" in names:\n        raise MalformedInputError(',
          '    if False:\n        raise MalformedInputError('),),
        (f"{EXCEL_TESTS}::TestExcelRefusals::test_a_macro_enabled_workbook_is_refused",),
        (("test_a_macro_enabled_workbook_is_refused", "DID NOT RAISE"),),
    ),
    Mutation(
        "M-W4-02", "multi-sheet ambiguity silently resolved to the first sheet",
        ((EXCEL,
          '    raise MalformedInputError(\n        f"{path.name} has {len(names)} sheets',
          '    return wb[names[0]]\n    raise MalformedInputError(\n        f"{path.name} has {len(names)} sheets'),),
        (f"{EXCEL_TESTS}::TestExcelRefusals::test_a_multi_sheet_workbook_is_ambiguous",),
        (("test_a_multi_sheet_workbook_is_ambiguous", "DID NOT RAISE"),),
    ),
    Mutation(
        "M-W4-03", "worksheet dimension preflight skipped",
        ((EXCEL,
          "        ws = _choose_sheet(path, wb, sheet)\n        _check_dimensions(path, ws, limits)\n",
          "        ws = _choose_sheet(path, wb, sheet)\n"),),
        (f"{EXCEL_TESTS}::TestExcelLimits::test_an_over_limit_sheet_is_refused_before_its_cells_are_read",),
        (("test_an_over_limit_sheet_is_refused_before_its_cells_are_read", "an over-limit sheet was materialized"),),
    ),
    Mutation(
        "M-W4-04", "corrupt zip surfaces a bare error",
        ((EXCEL,
          "    if not zipfile.is_zipfile(path):\n        raise MalformedInputError(",
          "    if False:\n        raise MalformedInputError("),),
        (f"{EXCEL_TESTS}::TestExcelRefusals::test_a_corrupt_file_is_malformed_not_a_bare_error",),
        (("test_a_corrupt_file_is_malformed_not_a_bare_error", "BadZipFile"),),
    ),
    Mutation(
        "M-W4-05", "duplicate headers accepted",
        ((EXCEL,
          '        if cell is not None and name in seen:\n            raise DuplicateHeadersError(',
          '        if False:\n            raise DuplicateHeadersError('),),
        (f"{EXCEL_TESTS}::TestExcelRefusals::test_duplicate_headers_are_refused",),
        (("test_duplicate_headers_are_refused", "DID NOT RAISE"),),
    ),
    Mutation(
        "M-W4-06", "missing openpyxl surfaces a bare ImportError",
        ((EXCEL,
          "    if spec is None:\n        raise MissingDependencyError(",
          "    if False:\n        raise MissingDependencyError("),),
        (f"{ABSENCE_TESTS}::TestIngestionWithoutOpenpyxl::test_an_xlsx_read_names_the_extra",),
        (("test_an_xlsx_read_names_the_extra", "REFUSED"),),
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
    (scratch / "g1_w4_mutations.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"rev={sha} python={summary['python']} baseline_exit={baseline_exit} imported_from_tree={imported_from_tree}")
    for r in results:
        print(f"{r['id']}  {r['result']:22s} exit={r['exit']} restored={r['restored']}  {r['weakening']}  {r['detail']}")
    print({state: summary[state] for state in ("total", "KILLED", "SURVIVED", "HARNESS_ERROR", "TEST_ENVIRONMENT_ERROR")})
    return 0 if summary["KILLED"] == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
