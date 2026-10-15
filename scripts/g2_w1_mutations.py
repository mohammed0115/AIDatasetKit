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
        "assert 2 == 1",
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
    environment_ok = imported_from_tree and not baseline_failed and manifest(tree) == pristine
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
                detail=f"imported_from_tree={imported_from_tree}; baseline failures={len(baseline_failed)}",
            )
            records.append(record)
            continue

        edited_files: dict[str, bytes] = {}
        applied = True
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

        result = None
        if applied:
            result = run_pytest(
                tree,
                (mutation.selector,),
                scratch / f"{mutation.id}.xml",
            )
            record["exit"] = result.returncode

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
                if mutation.expected_test in name
            ]
            if (
                error_count
                or not expected
                or not any(mutation.expected_failure in details for _, details in expected)
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
        "baseline_failures": len(baseline_failed),
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
    return 0 if summary["KILLED"] == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
