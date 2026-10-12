"""G1-W8 adversarial check: each weakening of strict ISO recognition must be killed.

Usage: python scripts/g1_w8_mutations.py <scratch-dir> [--rev REV]

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

TESTS = "tests/unit/test_date_inference.py"
RECOGNITION = "aidatasetkit/profiling/date_inference.py"
BUILDER = "aidatasetkit/evidence/builder.py"
TYPES = "aidatasetkit/evidence/types.py"


@dataclass(frozen=True)
class Mutation:
    id: str
    weakening: str
    edits: tuple[tuple[str, str, str], ...]
    selector: tuple[str, ...]
    expected: tuple[tuple[str, str], ...]


MUTATIONS = (
    Mutation(
        "M-W8-01", "identifier-like text inferred as a date",
        ((RECOGNITION,
          '_DATE = re.compile(r"\\A(?P<y>[0-9]{4})-(?P<m>[0-9]{2})-(?P<d>[0-9]{2})\\Z")\n',
          '_DATE = re.compile(r"\\A(?:ACC-)?(?P<y>[0-9]{4})-?(?P<m>[0-9]{2})-?(?P<d>[0-9]{2})\\Z")\n'),),
        (f"{TESTS}::TestNegativeRecognition::test_identifier_text_is_not_a_date",),
        (("test_identifier_text_is_not_a_date", "identifier-like text inferred as a date"),),
    ),
    Mutation(
        "M-W8-02", "ambiguous slash date accepted",
        ((RECOGNITION,
          "    return None  # strict ISO only; slash dates and identifiers are refused\n",
          "    if len(text) == 10 and text[2] == \"/\" and text[5] == \"/\":\n"
          "        year, month, day = (int(part) for part in (text[6:10], text[3:5], text[0:2]))\n"
          "        if _components_valid(year, month, day):\n"
          "            return (\"date\", \"naive\")\n"
          "    return None  # strict ISO only; slash dates and identifiers are refused\n"),),
        (f"{TESTS}::TestNegativeRecognition::test_slash_dates_are_refused",),
        (("test_slash_dates_are_refused", "ambiguous slash date accepted"),),
    ),
    Mutation(
        "M-W8-03", "invalid value ignored",
        ((RECOGNITION,
          "        if parsed is None:\n            invalid += 1\n            continue\n",
          "        if parsed is None:\n            matched += 1\n            continue\n"),),
        (f"{TESTS}::TestNegativeRecognition::test_one_invalid_value_blocks_the_column",),
        (("test_one_invalid_value_blocks_the_column", "invalid value ignored"),),
    ),
    Mutation(
        "M-W8-04", "null or empty values counted as failures",
        ((RECOGNITION,
          "        if _is_blank(value):\n            blank += 1\n            continue\n",
          "        if _is_blank(value):\n            invalid += 1\n            non_blank += 1\n            continue\n"),),
        (f"{TESTS}::TestPositiveRecognition::test_nulls_and_empty_strings_are_not_failures",),
        (("test_nulls_and_empty_strings_are_not_failures", "null or empty values counted as failures"),),
    ),
    Mutation(
        "M-W8-05", "minimum observation threshold bypassed",
        ((RECOGNITION,
          "    if blocked or invalid or non_blank < MINIMUM_NON_BLANK or matched != non_blank:\n",
          "    if blocked or invalid or non_blank < 1 or matched != non_blank:\n"),),
        (f"{TESTS}::TestNegativeRecognition::test_seven_values_are_not_enough",),
        (("test_seven_values_are_not_enough", "minimum observation threshold bypassed"),),
    ),
    Mutation(
        "M-W8-06", "inference omitted from typed evidence",
        ((BUILDER,
          "        columns=tuple(_column_date_evidence(column) for column in facts.columns),\n",
          "        columns=tuple(_column_date_evidence(column) for column in ()),\n"),),
        (f"{TESTS}::TestEvidenceAndSchema::test_positive_inference_is_in_the_artifact",),
        (("test_positive_inference_is_in_the_artifact", "inference omitted from typed evidence"),),
    ),
    Mutation(
        "M-W8-07", "fixed policy/method omitted or mislabeled",
        ((RECOGNITION,
          'METHOD = "strict_iso_full_population_v1"\n',
          'METHOD = "guessed"\n'),),
        (f"{TESTS}::TestEvidenceAndSchema::test_the_method_name_is_fixed",),
        (("test_the_method_name_is_fixed", "fixed policy/method omitted or mislabeled"),),
    ),
    Mutation(
        "M-W8-08", "host locale changes the decision",
        ((RECOGNITION,
          "def _host_locale_unused() -> None:\n"
          '    """The grammar does not consult the process locale."""\n'
          "    return\n",
          "def _host_locale_unused() -> None:\n"
          '    """The grammar does not consult the process locale."""\n'
          "    import locale\n"
          "    locale.getlocale()\n"
          "    return\n"),),
        (f"{TESTS}::TestDeterminism::test_the_host_locale_is_not_consulted",),
        (("test_the_host_locale_is_not_consulted", "host locale changes the decision"),),
    ),
    Mutation(
        "M-W8-09", "host timezone changes the decision",
        ((RECOGNITION,
          "def _host_zone_unused() -> None:\n"
          '    """The grammar does not consult the host timezone."""\n'
          "    return\n",
          "def _host_zone_unused() -> None:\n"
          '    """The grammar does not consult the host timezone."""\n'
          "    _datetime.now()\n"
          "    return\n"),),
        (f"{TESTS}::TestDeterminism::test_the_host_timezone_is_not_consulted",),
        (("test_the_host_timezone_is_not_consulted", "host timezone changes the decision"),),
    ),
    Mutation(
        "M-W8-10", "sampling presented as full population",
        ((RECOGNITION,
          "    examined = series\n",
          "    examined = series.head(8)\n"),),
        (f"{TESTS}::TestNegativeRecognition::test_a_late_invalid_value_is_seen",),
        (("test_a_late_invalid_value_is_seen", "sampling presented as full population"),),
    ),
    Mutation(
        "M-W8-11", "default full audit fails to run inference",
        ((BUILDER,
          "        facts = infer_strict_iso(frame)\n",
          "        facts = _unavailable_date_inference()\n"),),
        (f"{TESTS}::TestEvidenceAndSchema::test_a_full_audit_runs_inference",),
        (("test_a_full_audit_runs_inference", "default full audit fails to run inference"),),
    ),
    Mutation(
        "M-W8-12", "inferred evidence changes detected_kind or verdict",
        ((BUILDER,
          "def _keep_detected_kinds(\n"
          "    columns: tuple[ColumnEvidence, ...], facts: StrictIsoInference\n"
          ") -> tuple[ColumnEvidence, ...]:\n"
          '    """Inference must not rewrite operational detected_kind."""\n'
          "    return columns\n",
          "def _keep_detected_kinds(\n"
          "    columns: tuple[ColumnEvidence, ...], facts: StrictIsoInference\n"
          ") -> tuple[ColumnEvidence, ...]:\n"
          '    """Inference must not rewrite operational detected_kind."""\n'
          "    from dataclasses import replace\n"
          "    inferred = {item.column for item in facts.columns}\n"
          "    return tuple(\n"
          "        replace(column, detected_kind=\"datetime\")\n"
          "        if column.name.name in inferred\n"
          "        else column\n"
          "        for column in columns\n"
          "    )\n"),),
        (f"{TESTS}::TestIsolation::test_inference_does_not_change_kind_or_verdict",),
        (("test_inference_does_not_change_kind_or_verdict", "inferred evidence changes detected_kind or verdict"),),
    ),
    Mutation(
        "M-W8-13", "dataset fingerprint changes without conversion",
        ((BUILDER,
          "def _frame_for_fingerprint(frame: pd.DataFrame) -> pd.DataFrame:\n"
          '    """The fingerprint is of the accepted table, not a converted copy."""\n'
          "    return frame\n",
          "def _frame_for_fingerprint(frame: pd.DataFrame) -> pd.DataFrame:\n"
          '    """The fingerprint is of the accepted table, not a converted copy."""\n'
          "    converted = frame.copy()\n"
          "    converted[converted.columns[0]] = \"changed\"\n"
          "    return converted\n"),),
        (f"{TESTS}::TestIsolation::test_the_dataset_fingerprint_is_the_loaded_table",),
        (("test_the_dataset_fingerprint_is_the_loaded_table", "dataset fingerprint changes without conversion"),),
    ),
    Mutation(
        "M-W8-14", "malformed date crashes the audit",
        ((RECOGNITION,
          "    try:\n"
          "        _datetime(year, month, day, hour, minute, second)\n"
          "    except ValueError:\n"
          "        return False\n"
          "    return True\n",
          "    _datetime(year, month, day, hour, minute, second)\n"
          "    return True\n"),),
        (f"{TESTS}::TestNegativeRecognition::test_a_malformed_date_does_not_crash",),
        (("test_a_malformed_date_does_not_crash", "malformed date crashes the audit"),),
    ),
    Mutation(
        "M-W8-15", "chunked mode claims full date inference",
        ((BUILDER,
          "            date_inference=None,\n",
          "            date_inference=DateInferenceEvidence(\n"
          "                method=\"strict_iso_full_population_v1\",\n"
          "                population=\"full\",\n"
          "                full_population_scanned=True,\n"
          "                minimum_non_blank=8,\n"
          "                required_match_ratio=1.0,\n"
          "                conversion_performed=False,\n"
          "                affects_detected_kind=False,\n"
          "                affects_verdict=False,\n"
          "            ),\n"),),
        (f"{TESTS}::TestEvidenceAndSchema::test_a_chunked_artifact_does_not_claim_date_inference",),
        (("test_a_chunked_artifact_does_not_claim_date_inference", "chunked mode claims full date inference"),),
    ),
    Mutation(
        "M-W8-16", "schema remains 1.3 or migration is omitted",
        ((TYPES,
          'ARTIFACT_SCHEMA_VERSION = "1.4"\n',
          'ARTIFACT_SCHEMA_VERSION = "1.3"\n'),),
        (f"{TESTS}::TestEvidenceAndSchema::test_the_artifact_schema_is_1_4",),
        (("test_the_artifact_schema_is_1_4", "schema remains 1.3 or migration is omitted"),),
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
    (scratch / "g1_w8_mutations.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
