"""G1-W5 adversarial check: each weakening of chunked CSV/TSV profiling must be killed.

Usage: python scripts/g1_w5_mutations.py <scratch-dir> [--rev REV]

Same discipline as the G1-W4 harness: the tree at REV is exported with
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

TESTS = "tests/unit/test_chunked_profiling.py"
CHUNKED = "aidatasetkit/profiling/chunked.py"


@dataclass(frozen=True)
class Mutation:
    id: str
    weakening: str
    edits: tuple[tuple[str, str, str], ...]
    selector: tuple[str, ...]
    expected: tuple[tuple[str, str], ...]


MUTATIONS = (
    Mutation(
        "M-W5-01", "caller row limit ignored",
        ((CHUNKED,
          "    plan = plan_delimited(path, fmt, options, limits)\n",
          "    plan = plan_delimited(path, fmt, options, None)\n"),),
        (f"{TESTS}::test_a_row_over_the_limit_is_refused",),
        (("test_a_row_over_the_limit_is_refused", "DID NOT RAISE"),),
    ),
    Mutation(
        "M-W5-02", "JSON accepted as a chunked table",
        ((CHUNKED,
          "    if fmt not in (TableFormat.CSV, TableFormat.TSV):\n",
          "    if False:\n"),),
        (f"{TESTS}::test_only_csv_and_tsv_are_accepted",),
        (("test_only_csv_and_tsv_are_accepted", "DID NOT RAISE"),),
    ),
    Mutation(
        "M-W5-03", "the file is read in one piece",
        ((CHUNKED,
          "        chunksize=chunk_rows,\n",
          ""),),
        (f"{TESTS}::test_the_file_is_read_in_chunks",),
        (("test_the_file_is_read_in_chunks", "without a chunk size"),),
    ),
    Mutation(
        "M-W5-04", "only the first chunk is the population",
        ((CHUNKED,
          "            seen_rows += len(cast)\n        connection.commit()\n",
          "            seen_rows += len(cast)\n            break\n        connection.commit()\n"),),
        (f"{TESTS}::test_the_population_fingerprint_matches_the_full_table",),
        (("test_the_population_fingerprint_matches_the_full_table", "not the"),),
    ),
    Mutation(
        "M-W5-05", "scratch directory left behind",
        ((CHUNKED,
          "        if directory is not None:\n            shutil.rmtree(directory)\n",
          "        if directory is not None:\n            pass\n"),),
        (
            f"{TESTS}::test_the_temporary_directory_is_removed",
            f"{TESTS}::test_the_temporary_directory_is_removed_when_profiling_fails",
        ),
        (
            ("test_the_temporary_directory_is_removed", "temporary directory survived"),
            ("test_the_temporary_directory_is_removed_when_profiling_fails", "temporary directory survived"),
        ),
    ),
    Mutation(
        "M-W5-06", "scratch files are group-readable",
        ((CHUNKED,
          "    os.chmod(path, 0o600)\n",
          "    os.chmod(path, 0o644)\n"),),
        (f"{TESTS}::test_temporary_storage_is_private",),
        (("test_temporary_storage_is_private", "temporary storage is group-readable"),),
    ),
    Mutation(
        "M-W5-07", "an approximate quartile is unlabeled",
        ((CHUNKED,
          "                if approximated:\n",
          "                if False:\n"),),
        (f"{TESTS}::test_an_approximation_is_labeled",),
        (("test_an_approximation_is_labeled", "approximation was not labeled"),),
    ),
    Mutation(
        "M-W5-08", "an approximation changes the verdict",
        (("aidatasetkit/evidence/builder.py",
          "        # Approximations stay in the chunked record; the verdict never reads them.\n"
          "        verdict, reasons = decide_verdict(\n"
          "            findings, decisions, blocked_reason=blocked_reason\n"
          "        )\n",
          "        if chunked_profiling is not None and chunked_profiling.approximations:\n"
          "            blocked_reason = \"approximate\"\n"
          "        # Approximations stay in the chunked record; the verdict never reads them.\n"
          "        verdict, reasons = decide_verdict(\n"
          "            findings, decisions, blocked_reason=blocked_reason\n"
          "        )\n"),),
        (f"{TESTS}::test_a_labeled_approximation_does_not_change_the_verdict",),
        (("test_a_labeled_approximation_does_not_change_the_verdict", "approximation changed the verdict inputs"),),
    ),
    Mutation(
        "M-W5-09", "the default audit opts into chunked profiling",
        (("aidatasetkit/cli/main.py",
          "    if args.chunked_profile:\n",
          "    if True:\n"),),
        (f"{TESTS}::test_the_default_audit_does_not_chunk",),
        (("test_the_default_audit_does_not_chunk", "default audit used chunked profiling"),),
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
    (scratch / "g1_w5_mutations.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"rev={sha} python={summary['python']} baseline_exit={baseline_exit} imported_from_tree={imported_from_tree}")
    for record in results:
        print(f"{record['id']}  {record['result']:22s} exit={record['exit']} restored={record['restored']}  {record['weakening']}  {record['detail']}")
    print({state: summary[state] for state in ("total", "KILLED", "SURVIVED", "HARNESS_ERROR", "TEST_ENVIRONMENT_ERROR")})
    return 0 if summary["KILLED"] == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
