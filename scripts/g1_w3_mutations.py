"""G1-W3 adversarial check: each weakening of the new format guards must be killed.

Usage: python scripts/g1_w3_mutations.py <scratch-dir> [--rev REV]

Same discipline as ``scripts/g1_w2_mutations.py``: the tree at REV (default
HEAD) is exported with ``git archive`` into ``<scratch-dir>/tree``; the working
tree is never touched. Each mutation is applied there, its selector is run, and
the original bytes are written back. A mutation is KILLED only when every
replacement matched its anchor exactly once, pytest exited 1 with no errored
cases, every expected test failed with its expected reason text, and the tree
hashes as it did before. Exit 0 is SURVIVED; anything else is HARNESS_ERROR.

The mutations here weaken only authorities introduced by G1-W3 (the JSON/JSONL
and columnar readers); G1-W1 and G1-W2 authorities are covered by their own
scripts, which are not repeated. The script exits 0 only when all mutations
are KILLED.
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

JSON_TESTS = "tests/unit/test_ingestion_json.py"
COLUMNAR_TESTS = "tests/unit/test_ingestion_columnar.py"
ABSENCE_TESTS = "tests/integration/test_ingestion_without_pyarrow.py"

JSON_TEXT = "aidatasetkit/ingestion/json_text.py"
COLUMNAR = "aidatasetkit/ingestion/columnar.py"


@dataclass(frozen=True)
class Mutation:
    id: str
    weakening: str
    edits: tuple[tuple[str, str, str], ...]  # (file, old, new)
    selector: tuple[str, ...]
    expected: tuple[tuple[str, str], ...]  # (test id fragment, reason fragment)


MUTATIONS = (
    Mutation(
        "M-W3-01", "JSON top-level array check removed",
        ((JSON_TEXT,
          "    if not isinstance(parsed, list):\n        raise MalformedInputError(",
          "    if False:\n        raise MalformedInputError("),),
        (f"{JSON_TESTS}::TestJsonShapeRefusals::test_the_top_level_must_be_an_array",),
        (("test_the_top_level_must_be_an_array[object]", "this document is an object"),),
    ),
    Mutation(
        "M-W3-02", "duplicate keys silently kept (json.loads default)",
        ((JSON_TEXT,
          "        return json.loads(\n            text, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant\n        )",
          "        return json.loads(text, parse_constant=_no_constant)"),),
        (f"{JSON_TESTS}::TestJsonShapeRefusals::test_a_duplicate_key_is_refused_not_kept_last",
         f"{JSON_TESTS}::TestJsonlRefusals::test_a_duplicate_key_names_its_line"),
        (("test_a_duplicate_key_is_refused_not_kept_last", "DID NOT RAISE"),
         ("test_a_duplicate_key_names_its_line", "DID NOT RAISE")),
    ),
    Mutation(
        "M-W3-03", "nested objects and arrays accepted into cells",
        ((JSON_TEXT,
          "            if isinstance(value, (dict, list)):\n                raise MalformedInputError(",
          "            if False:\n                raise MalformedInputError("),),
        (f"{JSON_TESTS}::TestJsonShapeRefusals::test_a_nested_object_is_refused_naming_its_key",
         f"{JSON_TESTS}::TestJsonShapeRefusals::test_a_nested_array_is_refused_naming_its_key"),
        (("test_a_nested_object_is_refused_naming_its_key", "DID NOT RAISE"),
         ("test_a_nested_array_is_refused_naming_its_key", "DID NOT RAISE")),
    ),
    Mutation(
        "M-W3-04", "non-standard NaN/Infinity constants accepted",
        ((JSON_TEXT,
          "def _no_constant(constant: str) -> Any:\n    raise _NonStandardConstant(constant)",
          "def _no_constant(constant: str) -> Any:\n    return float(constant)"),),
        (f"{JSON_TESTS}::TestJsonShapeRefusals::test_non_standard_constants_are_refused",),
        (("test_non_standard_constants_are_refused", "DID NOT RAISE"),),
    ),
    Mutation(
        "M-W3-05", "JSONL record refusal deferred until the whole file is read",
        ((JSON_TEXT,
          "    records: list[Mapping[str, Any]] = []\n    names: dict[str, None] = {}\n    total_chars = 0\n",
          "    records: list[Mapping[str, Any]] = []\n    names: dict[str, None] = {}\n    total_chars = 0\n    deferred = None\n"),
         (JSON_TEXT,
          "            raise RecordLimitError(\n                f\"{name} exceeds the record limit",
          "            deferred = RecordLimitError(\n                f\"{name} exceeds the record limit"),
         (JSON_TEXT,
          "        records.append(record)\n    return records, list(names)",
          "        records.append(record)\n    if deferred is not None:\n        raise deferred\n    return records, list(names)")),
        (f"{JSON_TESTS}::TestJsonLimits::test_a_jsonl_refusal_reads_no_line_past_the_decisive_one",),
        (("test_a_jsonl_refusal_reads_no_line_past_the_decisive_one", "lines after the limit was crossed"),),
    ),
    Mutation(
        "M-W3-06", "JSON record boundary `>=` weakened to `>`",
        ((JSON_TEXT,
          "        if limits.max_records is not None and len(records) >= limits.max_records:",
          "        if limits.max_records is not None and len(records) > limits.max_records:"),),
        (f"{JSON_TESTS}::TestJsonLimits::test_exactly_at_each_limit_is_accepted_and_one_under_is_refused",),
        (("test_exactly_at_each_limit_is_accepted_and_one_under_is_refused[json-max_records-2]", "DID NOT RAISE"),
         ("test_exactly_at_each_limit_is_accepted_and_one_under_is_refused[jsonl-max_records-2]", "DID NOT RAISE")),
    ),
    Mutation(
        "M-W3-07", "Parquet footer shape preflight skipped",
        ((COLUMNAR,
          "    _check_names(path, footer.schema.names)\n    _check_shape(path, footer.num_rows, footer.num_columns, limits)\n",
          "    _check_names(path, footer.schema.names)\n"),),
        (f"{COLUMNAR_TESTS}::TestColumnarLimits::test_a_parquet_over_the_row_limit_is_refused_without_reading_data",),
        (("test_a_parquet_over_the_row_limit_is_refused_without_reading_data", "an over-limit parquet was materialized"),),
    ),
    Mutation(
        "M-W3-08", "corrupt Parquet surfaces a bare pyarrow error",
        ((COLUMNAR,
          "    except (pa.lib.ArrowException, OSError) as error:\n        raise MalformedInputError(\n            f\"{path.name} is not a readable Parquet file ({type(error).__name__}).\"\n        ) from None\n    _check_names",
          "    except (pa.lib.ArrowException, OSError) as error:\n        raise\n    _check_names"),),
        (f"{COLUMNAR_TESTS}::TestColumnarRefusals::test_a_corrupt_file_is_malformed_input_never_a_bare_pyarrow_error",),
        (("test_a_corrupt_file_is_malformed_input_never_a_bare_pyarrow_error[broken.parquet]", "ArrowInvalid"),),
    ),
    Mutation(
        "M-W3-09", "Feather row and cell budgets checked against a faked shape",
        ((COLUMNAR,
          "    _check_shape(path, table.num_rows, len(names), limits)",
          "    _check_shape(path, 1, len(names), limits)"),),
        (f"{COLUMNAR_TESTS}::TestColumnarLimits::test_a_feather_over_the_row_limit_is_refused_before_the_pandas_conversion",),
        (("test_a_feather_over_the_row_limit_is_refused_before_the_pandas_conversion", "DID NOT RAISE"),),
    ),
    Mutation(
        "M-W3-10", "text options accepted for binary formats",
        ((COLUMNAR,
          "    if not options.is_default:\n        raise InvalidIngestionOptionsError(",
          "    if False:\n        raise InvalidIngestionOptionsError("),),
        (f"{COLUMNAR_TESTS}::TestColumnarRefusals::test_text_options_are_refused_for_binary_formats",),
        (("test_text_options_are_refused_for_binary_formats", "DID NOT RAISE"),),
    ),
    Mutation(
        "M-W3-11", "missing pyarrow surfaces a bare ImportError, not the named extra",
        ((COLUMNAR,
          "    if spec is None:\n        raise MissingDependencyError(",
          "    if False:\n        raise MissingDependencyError("),),
        (f"{ABSENCE_TESTS}::TestIngestionWithoutPyarrow::test_a_columnar_read_names_the_extra",),
        (("test_a_columnar_read_names_the_extra", "REFUSED"),),
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
    (scratch / "g1_w3_mutations.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"rev={sha} python={summary['python']} baseline_exit={baseline_exit} imported_from_tree={imported_from_tree}")
    for r in results:
        print(f"{r['id']}  {r['result']:22s} exit={r['exit']} restored={r['restored']}  {r['weakening']}  {r['detail']}")
    print({state: summary[state] for state in ("total", "KILLED", "SURVIVED", "HARNESS_ERROR", "TEST_ENVIRONMENT_ERROR")})
    return 0 if summary["KILLED"] == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
