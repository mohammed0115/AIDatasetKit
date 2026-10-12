"""G1-W9 adversarial check: each weakening of record-iterable ingestion must be killed.

Usage: python scripts/g1_w9_mutations.py <scratch-dir> [--rev REV]

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

TESTS = "tests/unit/test_record_iterables.py"
LOADER = "aidatasetkit/ingestion/loader.py"
FACADE = "aidatasetkit/facade/facade.py"


@dataclass(frozen=True)
class Mutation:
    id: str
    weakening: str
    edits: tuple[tuple[str, str, str], ...]
    selector: tuple[str, ...]
    expected: tuple[tuple[str, str], ...]


MUTATIONS = (
    Mutation(
        "M-W9-01", "iterator materialized with list() before limits",
        ((LOADER,
          "    copied: list[dict[str, Any]] = []\n",
          "    copied = list(iterator)\n"),),
        (f"{TESTS}::TestLimits::test_materializing_with_list_is_not_how_the_limit_is_enforced",),
        (("test_materializing_with_list_is_not_how_the_limit_is_enforced",
          "iterator materialized with list() before limits"),),
    ),
    Mutation(
        "M-W9-02", "max_records ignored",
        ((LOADER,
          "        if limits.max_records is not None and accepted >= limits.max_records:\n",
          "        if False and accepted >= limits.max_records:\n"),),
        (f"{TESTS}::TestLimits::test_max_records_is_enforced",),
        (("test_max_records_is_enforced", "max_records ignored"),),
    ),
    Mutation(
        "M-W9-03", "iterator continues beyond max_records + 1",
        ((LOADER,
          "        if limits.max_records is not None and accepted >= limits.max_records:\n",
          "        if limits.max_records is not None and accepted >= limits.max_records + 1:\n"),),
        (f"{TESTS}::TestLimits::test_the_pull_stops_at_max_records_plus_one",),
        (("test_the_pull_stops_at_max_records_plus_one",
          "iterator continues beyond max_records + 1"),),
    ),
    Mutation(
        "M-W9-04", "non-mapping yielded item accepted",
        ((LOADER,
          "        if not isinstance(record, Mapping):\n"
          "            raise MalformedInputError(\n"
          "                f\"Record {accepted} is a {type(record).__name__}, not a mapping. Every \"\n",
          "        if False and not isinstance(record, Mapping):\n"
          "            raise MalformedInputError(\n"
          "                f\"Record {accepted} is a {type(record).__name__}, not a mapping. Every \"\n"),),
        (f"{TESTS}::TestMappingRules::test_a_non_mapping_item_is_refused",),
        (("test_a_non_mapping_item_is_refused", "non-mapping yielded item accepted"),),
    ),
    Mutation(
        "M-W9-05", "later-row columns ignored",
        ((LOADER,
          "            names.setdefault(key, None)\n",
          "            if accepted == 0:\n"
          "                names.setdefault(key, None)\n"),),
        (f"{TESTS}::TestMappingRules::test_a_later_row_can_add_a_column",),
        (("test_a_later_row_can_add_a_column", "later-row columns ignored"),),
    ),
    Mutation(
        "M-W9-06", "max_columns ignored",
        ((LOADER,
          "        if limits.max_columns is not None and len(names) > limits.max_columns:  # running union of this iterable\n",
          "        if False and len(names) > limits.max_columns:  # running union of this iterable\n"),),
        (f"{TESTS}::TestLimits::test_later_row_columns_stop_the_pull",),
        (("test_later_row_columns_stop_the_pull", "max_columns ignored"),),
    ),
    Mutation(
        "M-W9-07", "max_cells ignored",
        ((LOADER,
          "        if names and limits.max_cells is not None and _cells_exceed(accepted + 1, len(names), limits.max_cells):\n",
          "        if False and names and limits.max_cells is not None and _cells_exceed(accepted + 1, len(names), limits.max_cells):\n"),),
        (f"{TESTS}::TestLimits::test_later_row_cells_stop_the_pull",),
        (("test_later_row_cells_stop_the_pull", "max_cells ignored"),),
    ),
    Mutation(
        "M-W9-08", "max_record_chars ignored",
        ((LOADER,
          "            total_chars += sum(len(str(value)) for value in shallow.values())\n"
          "            if total_chars > limits.max_record_chars:\n",
          "            total_chars += sum(len(str(value)) for value in shallow.values())\n"
          "            if False and total_chars > limits.max_record_chars:\n"),),
        (f"{TESTS}::TestLimits::test_record_character_budget_applies_to_an_iterable",),
        (("test_record_character_budget_applies_to_an_iterable", "max_record_chars ignored"),),
    ),
    Mutation(
        "M-W9-09", "iterator exception publishes partial output",
        ((LOADER,
          "        return LoadedTable(frame=frame, metadata=_in_memory(SourceKind.RECORDS, frame, notes))  # iterable materialization\n"
          "    finally:\n",
          "        return LoadedTable(frame=frame, metadata=_in_memory(SourceKind.RECORDS, frame, notes))  # iterable materialization\n"
          "    except Exception:\n"
          "        frame = pd.DataFrame([{\"kept\": 1}])\n"
          "        return LoadedTable(frame=frame, metadata=_in_memory(SourceKind.RECORDS, frame, ()))\n"
          "    finally:\n"),),
        (f"{TESTS}::TestCleanup::test_an_iterator_exception_is_not_published",),
        (("test_an_iterator_exception_is_not_published",
          "iterator exception publishes partial output"),),
    ),
    Mutation(
        "M-W9-10", "iterator consumed twice",
        ((LOADER,
          "    iterator: Iterator[Any] = iter(source)\n",
          "    iterator: Iterator[Any] = iter(source); iter(source)\n"),),
        (f"{TESTS}::TestOnePass::test_iter_is_called_once",),
        (("test_iter_is_called_once", "iterator consumed twice"),),
    ),
    Mutation(
        "M-W9-11", "yielded mapping retained by reference",
        ((LOADER,
          "        shallow = dict(record)\n",
          "        shallow = record\n"),),
        (f"{TESTS}::TestMappingRules::test_the_yielded_mapping_is_not_retained",),
        (("test_the_yielded_mapping_is_not_retained",
          "yielded mapping retained by reference"),),
    ),
    Mutation(
        "M-W9-12", "iterator identity enters evidence",
        ((LOADER,
          "        return LoadedTable(frame=frame, metadata=_in_memory(SourceKind.RECORDS, frame, notes))  # iterable materialization\n",
          "        return LoadedTable(frame=frame, metadata=_in_memory(SourceKind.RECORDS, frame, notes + (repr(iterator),)))  # iterable materialization\n"),),
        (f"{TESTS}::TestEquivalence::test_evidence_has_no_iterator_identity",),
        (("test_evidence_has_no_iterator_identity", "iterator identity enters evidence"),),
    ),
    Mutation(
        "M-W9-13", "source type salts a fingerprint",
        ((LOADER,
          "        frame, notes = _records_frame(copied, names, limits, SourceKind.RECORDS.value)\n",
          "        frame, notes = _records_frame(copied, names, limits, SourceKind.RECORDS.value)\n"
          "        frame = frame.copy()\n"
          "        frame[\"iterable_source\"] = type(iterator).__name__\n"),),
        (f"{TESTS}::TestEquivalence::test_list_and_iterator_fingerprints_match",),
        (("test_list_and_iterator_fingerprints_match", "source type salts a fingerprint"),),
    ),
    Mutation(
        "M-W9-14", "mapping/string dispatched as record iterable",
        ((LOADER,
          "    if isinstance(source, (Mapping, bytearray, range, set, frozenset)):\n",
          "    if isinstance(source, (bytearray, range, set, frozenset)):\n"),),
        (f"{TESTS}::TestRefusals::test_a_bare_mapping_is_not_a_record_iterable",),
        (("test_a_bare_mapping_is_not_a_record_iterable",
          "mapping or string misread as records"),),
    ),
    Mutation(
        "M-W9-15", "async iterable accepted",
        ((LOADER,
          "    if isinstance(source, (AsyncIterable, AsyncIterator)):\n",
          "    if False and isinstance(source, (AsyncIterable, AsyncIterator)):\n"),),
        (f"{TESTS}::TestRefusals::test_an_async_generator_is_refused_before_iteration",),
        (("test_an_async_generator_is_refused_before_iteration", "async iterable accepted"),),
    ),
    Mutation(
        "M-W9-16", "GeneratorType not closed",
        ((LOADER,
          "        if isinstance(iterator, GeneratorType):\n"
          "            iterator.close()\n",
          "        if isinstance(iterator, GeneratorType):\n"
          "            pass\n"),),
        (f"{TESTS}::TestCleanup::test_a_generator_is_closed_when_the_record_limit_refuses_it",),
        (("test_a_generator_is_closed_when_the_record_limit_refuses_it",
          "GeneratorType not closed"),),
    ),
    Mutation(
        "M-W9-17", "custom re-iterable refused",
        ((LOADER,
          "    if isinstance(source, Iterable):\n",
          "    if isinstance(source, Iterator):\n"),),
        (f"{TESTS}::TestSources::test_a_custom_reiterable_is_accepted",),
        (("test_a_custom_reiterable_is_accepted", "custom re-iterable refused"),),
    ),
    Mutation(
        "M-W9-18", "facade contract silently broadened",
        ((FACADE,
          "        if not isinstance(frame, pd.DataFrame):\n",
          "        if False and not isinstance(frame, pd.DataFrame):\n"),),
        (f"{TESTS}::TestFacadeBoundary::test_the_facade_still_requires_a_dataframe",),
        (("test_the_facade_still_requires_a_dataframe",
          "facade contract silently broadened"),),
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
    (scratch / "g1_w9_mutations.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
