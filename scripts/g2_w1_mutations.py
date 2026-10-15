"""Adversarial checks for the G2-W1 datetime profiling contract.

Usage: python scripts/g2_w1_mutations.py <scratch-dir> [--rev REV]

The selected revision is exported to an isolated tree. Each mutation weakens
one contract, runs only its proving test, restores the original bytes, and
checks that the exported tree is pristine before continuing. A kill requires
the intended test to fail, no test errors, and exact tree restoration.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = "tests/unit/test_datetime_profiling.py"
DATETIME = "aidatasetkit/profiling/datetime.py"
CONFIG = "aidatasetkit/core/config.py"
CLOSURE_TESTS = "tests/unit/test_datetime_mutation_closure.py"
TYPES = "aidatasetkit/evidence/types.py"


@dataclass(frozen=True)
class Mutation:
    id: str
    weakening: str
    edits: tuple[tuple[str, str, str], ...]
    selector: str
    expected_test: str
    expected_failure: str


MUTATIONS = (
    Mutation(
        "M-G2W1-01",
        "total count omits missing rows",
        ((DATETIME, "total_count=total_count,", "total_count=valid_count,"),),
        f"{TESTS}::TestEligibilityAndCounts::test_naive_datetime_counts_range_and_timezone",
        "test_naive_datetime_counts_range_and_timezone",
        "assert 3 == 4",
    ),
    Mutation(
        "M-G2W1-02",
        "valid count includes missing rows",
        ((DATETIME, "valid_count=valid_count,", "valid_count=total_count,"),),
        f"{TESTS}::TestEligibilityAndCounts::test_naive_datetime_counts_range_and_timezone",
        "test_naive_datetime_counts_range_and_timezone",
        "assert 4 == 3",
    ),
    Mutation(
        "M-G2W1-03",
        "missing count is discarded",
        ((DATETIME, "missing_count=missing_count,", "missing_count=0,"),),
        f"{TESTS}::TestEligibilityAndCounts::test_naive_datetime_counts_range_and_timezone",
        "test_naive_datetime_counts_range_and_timezone",
        "assert 0 == 1",
    ),
    Mutation(
        "M-G2W1-04",
        "duplicate rows are not counted",
        ((DATETIME, "duplicate_timestamp_count=duplicate_count,", "duplicate_timestamp_count=0,"),),
        f"{TESTS}::TestEligibilityAndCounts::test_naive_datetime_counts_range_and_timezone",
        "test_naive_datetime_counts_range_and_timezone",
        "assert 0 == 1",
    ),
    Mutation(
        "M-G2W1-05",
        "range endpoints are reversed",
        (
            (DATETIME, "minimum=minimum.isoformat() if minimum is not None else None,", "minimum=maximum.isoformat() if maximum is not None else None,"),
            (DATETIME, "maximum=maximum.isoformat() if maximum is not None else None,", "maximum=minimum.isoformat() if minimum is not None else None,"),
        ),
        f"{TESTS}::TestEligibilityAndCounts::test_naive_datetime_counts_range_and_timezone",
        "test_naive_datetime_counts_range_and_timezone",
        "2024-01-03T00:00:00",
    ),
    Mutation(
        "M-G2W1-06",
        "timezone identity is omitted",
        ((DATETIME, "timezone=str(timezone) if timezone is not None else None,", "timezone=None,"),),
        f"{TESTS}::TestEligibilityAndCounts::test_timezone_aware_datetime_preserves_zone_and_endpoint_offsets",
        "test_timezone_aware_datetime_preserves_zone_and_endpoint_offsets",
        "Europe/London",
    ),
    Mutation(
        "M-G2W1-07",
        "interval count uses the wrong population",
        ((DATETIME, "interval_count=interval_count,", "interval_count=distinct_valid_count,"),),
        f"{TESTS}::TestObservedSpacing::test_two_distinct_timestamps_recognize_the_only_exact_spacing",
        "test_two_distinct_timestamps_recognize_the_only_exact_spacing",
        "assert 2 == 1",
    ),
    Mutation(
        "M-G2W1-08",
        "distinct spacing count equals intervals",
        ((DATETIME, "distinct_spacing_count=len(distinct_deltas),", "distinct_spacing_count=interval_count,"),),
        f"{TESTS}::TestObservedSpacing::test_unsorted_duplicates_do_not_change_frequency",
        "test_unsorted_duplicates_do_not_change_frequency",
        "assert 2 == 1",
    ),
    Mutation(
        "M-G2W1-09",
        "irregular observations are marked regular",
        ((DATETIME, "is_regular = None if not deltas else len(distinct_deltas) == 1", "is_regular = None if not deltas else True"),),
        f"{TESTS}::TestObservedSpacing::test_irregular_spacing_does_not_invent_a_cadence",
        "test_irregular_spacing_does_not_invent_a_cadence",
        "assert True is False",
    ),
    Mutation(
        "M-G2W1-10",
        "irregular observations receive a cadence",
        ((DATETIME, "_duration(deltas[0], unit) if is_regular is True else None", "_duration(deltas[0], unit) if deltas else None"),),
        f"{TESTS}::TestObservedSpacing::test_irregular_spacing_does_not_invent_a_cadence",
        "test_irregular_spacing_does_not_invent_a_cadence",
        "is None",
    ),
    Mutation(
        "M-G2W1-11",
        "spacing bounds are reversed",
        (
            (DATETIME, "minimum_spacing=_duration(min(deltas), unit) if deltas else None,", "minimum_spacing=_duration(max(deltas), unit) if deltas else None,"),
            (DATETIME, "maximum_spacing=_duration(max(deltas), unit) if deltas else None,", "maximum_spacing=_duration(min(deltas), unit) if deltas else None,"),
        ),
        f"{TESTS}::TestObservedSpacing::test_irregular_spacing_does_not_invent_a_cadence",
        "test_irregular_spacing_does_not_invent_a_cadence",
        "P2DT0H0M0S",
    ),
    Mutation(
        "M-G2W1-12",
        "unknown cadence is reported as zero gaps",
        ((DATETIME, "return None, None, None", "return 0, 0, 0"),),
        f"{TESTS}::TestExpectedCadence::test_default_does_not_classify_gaps",
        "test_default_does_not_classify_gaps",
        "assert 0 is None",
    ),
    Mutation(
        "M-G2W1-13",
        "missing expected positions are not counted",
        ((DATETIME, "            missing += absent\n", "            pass\n"),),
        f"{TESTS}::TestExpectedCadence::test_missing_expected_observations_are_counted_without_materialization",
        "test_missing_expected_observations_are_counted_without_materialization",
        "assert 0 == 2",
    ),
    Mutation(
        "M-G2W1-14",
        "trailing gap run is not counted",
        (
            (
                DATETIME,
                "    if trailing_absent:\n        missing += trailing_absent\n        gaps += 1\n",
                "    if trailing_absent:\n        missing += trailing_absent\n",
            ),
        ),
        f"{TESTS}::TestExpectedCadence::test_grid_covers_trailing_off_grid_observed_window",
        "test_grid_covers_trailing_off_grid_observed_window",
        "assert 0 == 1",
    ),
    Mutation(
        "M-G2W1-15",
        "off-grid observations are not counted",
        ((DATETIME, "            off_grid += 1\n", "            pass\n"),),
        f"{TESTS}::TestExpectedCadence::test_off_grid_observations_are_separate_from_gaps_and_duplicates",
        "test_off_grid_observations_are_separate_from_gaps_and_duplicates",
        "assert 0 == 1",
    ),
    Mutation(
        "M-G2W1-16",
        "expected grid is not anchored at the minimum",
        ((DATETIME, "    anchor = int(coordinates[0])\n", "    anchor = 0\n"),),
        f"{TESTS}::TestExpectedCadence::test_expected_grid_is_anchored_at_the_minimum_timestamp",
        "test_expected_grid_is_anchored_at_the_minimum_timestamp",
        "missing_expected_observation_count=",
    ),
    Mutation(
        "M-G2W1-17",
        "off-grid maximum does not extend the closed window",
        (
            (
                DATETIME,
                "    last_grid_index = (int(coordinates[-1]) - anchor) // cadence\n",
                "    last_grid_index = previous_grid_index\n",
            ),
        ),
        f"{TESTS}::TestExpectedCadence::test_grid_covers_trailing_off_grid_observed_window",
        "test_grid_covers_trailing_off_grid_observed_window",
        "assert 0 == 1",
    ),
    Mutation(
        "M-G2W1-18",
        "zero cadence is accepted",
        ((CONFIG, "cadence <= pd.Timedelta(0)", "cadence < pd.Timedelta(0)"),),
        f"{TESTS}::TestExpectedCadence::test_invalid_cadences_are_rejected",
        "test_invalid_cadences_are_rejected",
        "DID NOT RAISE",
    ),
    Mutation(
        "M-G2W1-19",
        "cadence is rounded to column resolution",
        ((DATETIME, "as_unit(unit, round_ok=False)", "as_unit(unit, round_ok=True)"),),
        f"{TESTS}::TestExpectedCadence::test_cadence_finer_than_column_resolution_is_rejected",
        "test_cadence_finer_than_column_resolution_is_rejected",
        "Regex pattern did not match",
    ),
    Mutation(
        "M-G2W1-20",
        "timezone-aware spacing uses wall time instead of elapsed instants",
        (
            (
                DATETIME,
                "coordinates = np.unique(series.array.asi8[series.array.asi8 != np.iinfo(\"int64\").min])",
                "coordinates = np.unique(series.dt.tz_localize(None).array.asi8[series.dt.tz_localize(None).array.asi8 != np.iinfo(\"int64\").min])",
            ),
        ),
        f"{TESTS}::TestTimezoneAndResolution::test_dst_spacing_uses_elapsed_instants_and_preserves_timezone",
        "test_dst_spacing_uses_elapsed_instants_and_preserves_timezone",
        "assert False is True",
    ),
)

# M-G2W1-27 is exactly M-G2W1-10: replace the regularity guard with
# "if deltas" and publish the first interval for irregular observations.
REQUIRED_MAPPINGS = {"M-G2W1-27": "M-G2W1-10"}


def closure_mutation(
    number: int, weakening: str, edit: tuple[str, str, str], test: str, reason: str
) -> Mutation:
    return Mutation(
        f"M-G2W1-{number:02d}", weakening, (edit,),
        f"{CLOSURE_TESTS}::{test}", test, reason,
    )


MUTATIONS += (
    closure_mutation(
        21, "missing cadence is accepted",
        (CONFIG, "        if value is None:\n            return\n",
         '        if value is None or value == "NaT":\n            return\n'),
        "test_missing_cadence_is_rejected", "missing cadence acceptance",
    ),
    closure_mutation(
        22, "calendar cadence is silently normalized as fixed",
        (CONFIG, "cadence = pd.Timedelta(value)",
         'cadence = pd.Timedelta("30 days" if value == "1 month" else value)'),
        "test_non_fixed_cadence_is_rejected", "non-fixed cadence acceptance",
    ),
    closure_mutation(
        23, "unsupported extensions bypass native eligibility guard",
        (DATETIME, "if not isinstance(series.array, pd.arrays.DatetimeArray):",
         "if False:"),
        "test_unsupported_extension_is_safely_unsummarized",
        "unsupported datetime eligibility guard failed",
    ),
    closure_mutation(
        24, "profiling reorders source datetime storage",
        (DATETIME, "    total_count = int(len(series))",
         "    series.array[:] = series.array[::-1]\n    total_count = int(len(series))"),
        "test_profiling_preserves_source_identity_values_index_dtype_order_timezone",
        "source datetime mutation or reordering",
    ),
    closure_mutation(
        25, "duplicate coordinates enter spacing",
        (DATETIME,
         'np.unique(series.array.asi8[series.array.asi8 != np.iinfo("int64").min])',
         'np.sort(series.array.asi8[series.array.asi8 != np.iinfo("int64").min])'),
        "test_duplicates_are_excluded_from_spacing_coordinates",
        "duplicate spacing coordinates",
    ),
    closure_mutation(
        26, "NaT exclusion is removed from temporal coordinates",
        (DATETIME,
         'np.unique(series.array.asi8[series.array.asi8 != np.iinfo("int64").min])',
         "np.unique(series.array.asi8)"),
        "test_nat_is_excluded_from_duplicate_and_spacing_coordinates",
        "NaT entered temporal coordinates",
    ),
    closure_mutation(
        28, "two observed timestamps supply an expected schedule",
        (DATETIME, "expected_cadence=expected_cadence,",
         "expected_cadence=expected_cadence or observed_cadence,"),
        "test_two_observations_do_not_imply_expected_schedule",
        "observed spacing became expected schedule",
    ),
    closure_mutation(
        29, "off-grid counting includes repeated coordinates",
        (DATETIME, "        coordinates, cadence_units\n",
         '        np.sort(series.array.asi8[series.array.asi8 != np.iinfo("int64").min]), cadence_units\n'),
        "test_repeated_off_grid_observations_are_counted_distinctly",
        "off-grid duplicates inflated count",
    ),
    closure_mutation(
        30, "missing positions are replaced by gap runs",
        (DATETIME, "    return missing, gaps, off_grid",
         "    return gaps, gaps, off_grid"),
        "test_missing_positions_and_contiguous_gaps_are_different_counts",
        "missing positions conflated with gap runs",
    ),
    closure_mutation(
        31, "canonical datetime evidence loses its context",
        (TYPES, 'canonical(dict(self.datetime), path="column.datetime")',
         'canonical({}, path="column.datetime")'),
        "test_canonical_datetime_evidence_keeps_endpoint_and_cadence_context",
        "canonical datetime context omitted",
    ),
    closure_mutation(
        32, "datetime gap measurements enter verdict policy",
        ("aidatasetkit/evidence/builder.py",
         "            findings, decisions, blocked_reason=blocked_reason\n",
         '            findings, decisions, blocked_reason=("datetime gap" if any(\n'
         '                column.datetime and column.datetime["missing_expected_observation_count"]\n'
         '                for column in columns\n'
         '            ) else blocked_reason)\n'),
        "test_datetime_measurements_are_isolated_from_verdict_inputs",
        "datetime measurements entered verdict inputs",
    ),
    closure_mutation(
        33, "datetime field displaces positional source argument",
        (TYPES,
         "    source: EvidenceSource = EvidenceSource.PROFILING\n"
         "    datetime: Mapping[str, Any] | None = None",
         "    datetime: Mapping[str, Any] | None = None\n"
         "    source: EvidenceSource = EvidenceSource.PROFILING"),
        "test_positional_column_source_remains_compatible",
        "positional source argument displaced",
    ),
    closure_mutation(
        34, "duplicate-count evidence invariant is not validated",
        (TYPES, '    if value["duplicate_timestamp_count"] != (\n',
         '    if False and value["duplicate_timestamp_count"] != (\n'),
        "test_inconsistent_datetime_evidence_requires_evidence_error",
        "inconsistent datetime evidence accepted without EvidenceError",
    ),
    Mutation(
        "M-G2W1-35", "chunked scan invokes native datetime summarization",
        (("aidatasetkit/profiling/chunked.py",
          "    path = Path(source)\n",
          "    from aidatasetkit.profiling.profiler import summarize_datetime\n"
          '    summarize_datetime(pd.Series([], dtype="datetime64[ns]"), expected_cadence=None)\n'
          "    path = Path(source)\n"),),
        f"{TESTS}::TestEvidenceAndIsolation::test_chunked_builder_does_not_call_datetime_summarizer",
        "test_chunked_builder_does_not_call_datetime_summarizer",
        "chunked mode invoked datetime profiling",
    ),
    closure_mutation(
        36, "expected timestamps are materialized before gap arithmetic",
        (DATETIME, "    missing_expected, gap_count, off_grid = _gap_counts(",
         "    if expected_cadence is not None and valid_count:\n"
         "        pd.date_range(start=minimum, end=maximum, freq=pd.Timedelta(expected_cadence))\n"
         "    missing_expected, gap_count, off_grid = _gap_counts("),
        "test_missing_timestamps_are_never_materialized",
        "missing timestamps were explicitly materialized",
    ),
    Mutation(
        "M-G2W1-37", "artifact schema remains 1.4",
        ((TYPES, 'ARTIFACT_SCHEMA_VERSION = "1.5"', 'ARTIFACT_SCHEMA_VERSION = "1.4"'),),
        "tests/unit/test_capability_fingerprint_migration.py::TestTheChain::test_datetime_schema_step_is_exact_and_reversible",
        "test_datetime_schema_step_is_exact_and_reversible",
        "datetime migration chain incomplete",
    ),
    Mutation(
        "M-G2W1-38", "pandas-2 semantic fixture retains schema 1.4",
        (("examples/audit_churn/expected_audit_semantic.pandas2.json",
          '"schema_version": "1.5"', '"schema_version": "1.4"'),),
        "tests/unit/test_current_example_contract.py::TestCommittedCurrentExamples::test_pandas_major_datetime_fixture_contract[expected_audit_semantic.pandas2.json]",
        "test_pandas_major_datetime_fixture_contract[expected_audit_semantic.pandas2.json]",
        "pandas-major datetime fixture contract",
    ),
)


def export(revision: str, tree: Path) -> str:
    sha = subprocess.run(
        ["git", "rev-parse", revision],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    archive = subprocess.run(
        ["git", "archive", "--format=tar", sha],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    tree.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(tree, filter="data")
    return sha


def manifest(tree: Path) -> dict[str, str]:
    return {
        str(path.relative_to(tree)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(tree.rglob("*"))
        if path.is_file()
    }


def environment(tree: Path) -> dict[str, str]:
    return dict(
        os.environ,
        PYTHONPATH=str(tree),
        PYTHONDONTWRITEBYTECODE="1",
    )


def run_pytest(
    tree: Path, selectors: tuple[str, ...], junit: Path
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            f"--junitxml={junit}",
            *selectors,
        ],
        cwd=tree,
        env=environment(tree),
        capture_output=True,
        text=True,
    )


def failures(junit: Path) -> tuple[dict[str, str], int]:
    root = ET.parse(junit).getroot()
    failed: dict[str, str] = {}
    errored = 0
    for case in root.iter("testcase"):
        name = f"{case.get('classname')}::{case.get('name')}"
        for failure in case.findall("failure"):
            failed[name] = (failure.get("message") or "") + "\n" + (
                failure.text or ""
            )
        errored += len(case.findall("error"))
    return failed, errored


def skipped(junit: Path) -> int:
    return sum(
        len(case.findall("skipped"))
        for case in ET.parse(junit).getroot().iter("testcase")
    )


def is_expected_test(name: str, expected: str) -> bool:
    test_name = name.rsplit("::", 1)[-1]
    return test_name == expected or (
        "[" not in expected and test_name.startswith(expected + "[")
    )


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/g2_w1_mutations.py <scratch-dir> [--rev REV]")
        return 2
    scratch = Path(sys.argv[1]).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    revision = sys.argv[sys.argv.index("--rev") + 1] if "--rev" in sys.argv else "HEAD"
    tree = scratch / "tree"
    if tree.exists():
        raise FileExistsError(f"refusing to overwrite existing mutation tree: {tree}")
    sha = export(revision, tree)
    pristine = manifest(tree)
    probe = subprocess.run(
        [sys.executable, "-c", "import aidatasetkit; print(aidatasetkit.__file__)"],
        cwd=tree,
        env=environment(tree),
        capture_output=True,
        text=True,
    )
    imported_from_tree = (
        probe.returncode == 0
        and Path(probe.stdout.strip()).resolve().is_relative_to(tree)
    )
    selectors = tuple(dict.fromkeys(mutation.selector for mutation in MUTATIONS))
    baseline = run_pytest(tree, selectors, scratch / "baseline.xml")
    baseline_failed = (
        [] if baseline.returncode == 0 else [(baseline.returncode, baseline.stdout, baseline.stderr)]
    )
    baseline_skipped = skipped(scratch / "baseline.xml") if baseline.returncode == 0 else None
    environment_ok = (
        imported_from_tree and not baseline_failed and baseline_skipped == 0
        and manifest(tree) == pristine
    )
    records: list[dict[str, object]] = []

    for mutation in MUTATIONS:
        record: dict[str, object] = {
            "id": mutation.id,
            "weakening": mutation.weakening,
            "selector": mutation.selector,
            "exit": None,
            "restored": None,
            "detail": "",
        }
        if not environment_ok:
            record.update(
                result="TEST_ENVIRONMENT_ERROR",
                detail=f"imported_from_tree={imported_from_tree}; baseline failures={len(baseline_failed)}; skipped={baseline_skipped}",
            )
            records.append(record)
            continue

        edited_files: dict[str, bytes] = {}
        applied = True
        result = None
        try:
            for relative, old, new in mutation.edits:
                path = tree / relative
                if relative not in edited_files:
                    edited_files[relative] = path.read_bytes()
                text = path.read_text(encoding="utf-8")
                if text.count(old) != 1:
                    applied = False
                    record["detail"] = f"anchor matched {text.count(old)} times in {relative}"
                    break
                path.write_text(text.replace(old, new), encoding="utf-8", newline="")
            if applied:
                result = run_pytest(
                    tree,
                    (mutation.selector,),
                    scratch / f"{mutation.id}.xml",
                )
                record["exit"] = result.returncode
        finally:
            for relative, contents in edited_files.items():
                (tree / relative).write_bytes(contents)
        record["restored"] = manifest(tree) == pristine

        if not applied:
            record["result"] = "HARNESS_ERROR"
        elif not record["restored"]:
            record.update(
                result="HARNESS_ERROR",
                detail="tree differs from the export after restoration",
            )
        elif result is None or result.returncode != 1:
            record.update(
                result="HARNESS_ERROR" if result is not None and result.returncode != 0 else "SURVIVED",
                detail="" if result is None else f"pytest exit {result.returncode}",
            )
        else:
            failures_found, error_count = failures(scratch / f"{mutation.id}.xml")
            expected = [
                (name, details)
                for name, details in failures_found.items()
                if is_expected_test(name, mutation.expected_test)
            ]
            if (
                error_count
                or not expected
                or skipped(scratch / f"{mutation.id}.xml")
                or not all(mutation.expected_failure in details for _, details in expected)
                or len(expected) != len(failures_found)
            ):
                record.update(
                    result="HARNESS_ERROR",
                    detail=(
                        f"errors={error_count}; expected_test_failures={len(expected)}; "
                        f"all_failures={len(failures_found)}"
                    ),
                )
            else:
                record["result"] = "KILLED"
                record["killed_by"] = sorted(
                    name.rsplit("::", 1)[-1] for name in failures_found
                )
        records.append(record)

    summary = {
        "revision": sha,
        "python": sys.version.split()[0],
        "imported_from_tree": imported_from_tree,
        "baseline_exit": baseline.returncode,
        "baseline_skipped": baseline_skipped,
        "baseline_failures": len(baseline_failed),
        "required_mappings": REQUIRED_MAPPINGS,
        "tree_restored": manifest(tree) == pristine,
        "total": len(records),
        **{
            state: sum(record.get("result") == state for record in records)
            for state in (
                "KILLED",
                "SURVIVED",
                "HARNESS_ERROR",
                "TEST_ENVIRONMENT_ERROR",
            )
        },
        "mutations": records,
    }
    (scratch / "g2_w1_mutations.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"revision={sha} python={summary['python']} "
        f"imported_from_tree={imported_from_tree} "
        f"baseline_exit={baseline.returncode} "
        f"baseline_failures={len(baseline_failed)}"
    )
    for record in records:
        print(
            f"{record['id']}  {record['result']:22s} "
            f"exit={record['exit']} restored={record['restored']} "
            f"{record['weakening']} {record['detail']}"
        )
    print(
        {
            state: summary[state]
            for state in (
                "total",
                "KILLED",
                "SURVIVED",
                "HARNESS_ERROR",
                "TEST_ENVIRONMENT_ERROR",
            )
        }
    )
    return 0 if environment_ok and summary["tree_restored"] and summary["KILLED"] == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
