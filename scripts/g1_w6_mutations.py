"""G1-W6 adversarial check: each weakening of the SQLite reader must be killed.

Usage: python scripts/g1_w6_mutations.py <scratch-dir> [--rev REV]

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

TESTS = "tests/unit/test_ingestion_sqlite.py"
SQLITE = "aidatasetkit/ingestion/sqlite.py"
BUILDER = "aidatasetkit/evidence/builder.py"


@dataclass(frozen=True)
class Mutation:
    id: str
    weakening: str
    edits: tuple[tuple[str, str, str], ...]
    selector: tuple[str, ...]
    expected: tuple[tuple[str, str], ...]


MUTATIONS = (
    Mutation(
        "M-W6-01", "mode=ro removed",
        ((SQLITE, '?mode=ro"', '"'),),
        (f"{TESTS}::TestSecurity::test_mode_ro_is_used",),
        (("test_mode_ro_is_used", "mode=ro"),),
    ),
    Mutation(
        "M-W6-02", "query_only removed",
        ((SQLITE, '    connection.execute("PRAGMA query_only=ON")\n', "    pass\n"),),
        (f"{TESTS}::TestSecurity::test_query_only_is_enabled",),
        (("test_query_only_is_enabled", "query_only"),),
    ),
    Mutation(
        "M-W6-03", "missing path creates a database",
        ((SQLITE,
          "    if not path.is_file():\n"
          "        raise InputNotFoundError(f\"No such file: {path.name}.\")\n"
          "    uri = f\"{path.resolve().as_uri()}?mode=ro\"\n"
          "    try:\n"
          "        connection = sqlite3.connect(uri, uri=True)\n",
          "    uri = path\n"
          "    try:\n"
          "        connection = sqlite3.connect(uri)\n"),),
        (f"{TESTS}::TestSecurity::test_a_missing_database_is_not_created",),
        (("test_a_missing_database_is_not_created", "missing path creates a database"),),
    ),
    Mutation(
        "M-W6-04", "arbitrary/user SQL accepted",
        ((SQLITE,
          '    return connection.execute(f"SELECT * FROM {quoted}").fetchall()\n',
          "    return connection.execute(quoted).fetchall()\n"),),
        (f"{TESTS}::TestSelection::test_an_injection_shaped_name_is_not_executed_as_sql",),
        (("test_an_injection_shaped_name_is_not_executed_as_sql", "arbitrary/user SQL accepted"),),
    ),
    Mutation(
        "M-W6-05", "view accepted",
        ((SQLITE, '    if kind != "table":\n', '    if kind != "table" and kind != "view":\n'),),
        (f"{TESTS}::TestSelection::test_a_view_is_excluded",),
        (("test_a_view_is_excluded", "view accepted"),),
    ),
    Mutation(
        "M-W6-06", "virtual table accepted",
        ((SQLITE, '.startswith("CREATE VIRTUAL")', '.startswith("CREATE NEVER")'),),
        (f"{TESTS}::TestSelection::test_a_virtual_table_is_excluded",),
        (("test_a_virtual_table_is_excluded", "virtual table accepted"),),
    ),
    Mutation(
        "M-W6-07", "sqlite_* internal table accepted",
        ((SQLITE, '    if name.startswith("sqlite_"):\n        return False\n', ""),),
        (f"{TESTS}::TestSelection::test_an_internal_table_is_excluded",),
        (("test_an_internal_table_is_excluded", "sqlite_* internal table accepted"),),
    ),
    Mutation(
        "M-W6-08", "multiple tables silently select one",
        ((SQLITE, "        if len(names) != 1:\n", "        if False:\n"),),
        (f"{TESTS}::TestSelection::test_multiple_tables_require_an_explicit_name",),
        (("test_multiple_tables_require_an_explicit_name", "multiple tables silently select one"),),
    ),
    Mutation(
        "M-W6-09", "row limit checked after materialization",
        ((SQLITE,
          "    count = int(connection.execute(f\"SELECT COUNT(*) FROM {quoted}\").fetchone()[0])\n"
          "    _check_shape(path, count, len(names), limits)\n"
          "    _check_field_lengths(connection, path, quoted, names, limits)\n"
          "    rows = _fetch_rows(connection, quoted)\n",
          "    rows = _fetch_rows(connection, quoted)\n"
          "    count = int(connection.execute(f\"SELECT COUNT(*) FROM {quoted}\").fetchone()[0])\n"
          "    _check_shape(path, count, len(names), limits)\n"
          "    _check_field_lengths(connection, path, quoted, names, limits)\n"),),
        (f"{TESTS}::TestLimits::test_too_many_rows_are_refused_before_fetch",),
        (("test_too_many_rows_are_refused_before_fetch", "row limit checked after materialization"),),
    ),
    Mutation(
        "M-W6-10", "max_cells ignored",
        ((SQLITE,
          "    if limits.max_cells is not None and _cells_exceed(rows, columns, limits.max_cells):\n",
          "    if False and _cells_exceed(rows, columns, limits.max_cells):\n"),),
        (f"{TESTS}::TestLimits::test_too_many_cells_are_refused",),
        (("test_too_many_cells_are_refused", "max_cells ignored"),),
    ),
    Mutation(
        "M-W6-11", "max_field_length ignores TEXT",
        ((SQLITE,
          "    if longest_text > limits.max_field_length:\n",
          "    if False and longest_text > limits.max_field_length:\n"),),
        (f"{TESTS}::TestLimits::test_text_length_is_its_own_check",),
        (("test_text_length_is_its_own_check", "max_field_length ignores TEXT"),),
    ),
    Mutation(
        "M-W6-12", "max_field_length ignores BLOB",
        ((SQLITE,
          "    if longest_blob > limits.max_field_length:\n",
          "    if False and longest_blob > limits.max_field_length:\n"),),
        (f"{TESTS}::TestLimits::test_an_oversized_blob_is_refused",),
        (("test_an_oversized_blob_is_refused", "max_field_length ignores BLOB"),),
    ),
    Mutation(
        "M-W6-13", "table selector omitted from artifact evidence",
        ((BUILDER,
          "            source_selector=_selector_evidence(source_selector),\n",
          "            source_selector=None,\n"),),
        (f"{TESTS}::TestEvidence::test_the_artifact_records_the_table",),
        (("test_the_artifact_records_the_table", "table selector omitted from artifact evidence"),),
    ),
    Mutation(
        "M-W6-14", "table selector omitted from config fingerprint",
        ((BUILDER,
          "        resolved_settings = _with_source_selector(resolved_settings, source_selector)\n",
          ""),),
        (f"{TESTS}::TestEvidence::test_a_different_table_changes_the_config_fingerprint",),
        (("test_a_different_table_changes_the_config_fingerprint", "table selector omitted from config fingerprint"),),
    ),
    Mutation(
        "M-W6-15", ".db accepted",
        (("aidatasetkit/ingestion/formats.py",
          '    ".sqlite3": TableFormat.SQLITE,\n',
          '    ".sqlite3": TableFormat.SQLITE,\n    ".db": TableFormat.SQLITE,\n'),),
        (f"{TESTS}::TestSelection::test_a_db_suffix_is_refused",),
        (("test_a_db_suffix_is_refused", ".db accepted"),),
    ),
    Mutation(
        "M-W6-16", "table= accepted for a non-SQLite format",
        (("aidatasetkit/ingestion/loader.py",
          "    if fmt is not TableFormat.SQLITE and table is not None:\n",
          "    if False and table is not None:\n"),),
        (f"{TESTS}::TestSelection::test_table_is_refused_for_other_formats",),
        (("test_table_is_refused_for_other_formats", "table= accepted for a non-SQLite format"),),
    ),
    Mutation(
        "M-W6-17", "SQLite error leaks raw path or escapes unwrapped",
        ((SQLITE,
          "            return _read(connection, path, limits, table)\n"
          "        except sqlite3.Error:\n"
          "            raise MalformedInputError(\n"
          "                f\"{path.name} is not a readable SQLite database.\"\n"
          "            ) from None\n",
          "            return _read(connection, path, limits, table)\n"
          "        except sqlite3.Error as error:\n"
          "            raise error\n"),),
        (f"{TESTS}::TestSecurity::test_a_corrupt_database_does_not_leak_the_driver_error",),
        (("test_a_corrupt_database_does_not_leak_the_driver_error", "raw path"),),
    ),
    Mutation(
        "M-W6-18", "schema incorrectly remains 1.2",
        (("aidatasetkit/evidence/types.py",
          'ARTIFACT_SCHEMA_VERSION = "1.3"\n',
          'ARTIFACT_SCHEMA_VERSION = "1.2"\n'),),
        (f"{TESTS}::TestEvidence::test_the_artifact_records_the_table",),
        (("test_the_artifact_records_the_table", "schema incorrectly remains 1.2"),),
    ),
)


def export(rev: str, tree: Path) -> str:
    sha = subprocess.run(
        ["git", "rev-parse", rev], capture_output=True, text=True, check=True
    ).stdout.strip()
    archive = subprocess.run(
        ["git", "archive", "--format=tar", sha], capture_output=True, check=True
    ).stdout
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
    (scratch / "g1_w6_mutations.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
