"""``aidatasetkit audit`` -- the first thing a new user runs.

The command orchestrates existing public components and adds no analysis of its
own. It reads a file, calls the profiler, the quality inspector, the task
detector, the planner, and the evidence builder in that order, and writes what
they produced. Every judgement it reports was made by a layer below it.

Two behaviours are worth stating up front because they are unusual.

**It does not train a model.** A ``--model`` argument selects a *capability
context*, so the plan can record why a scaler is present or why the output is
dense. No estimator is ever constructed and no model is ever fitted. The
*preprocessor* is fitted, which is a different thing and the reason lineage can
report what a column actually became rather than only what was intended.

**A failed analysis still produces an artifact.** If preprocessing cannot be
planned -- an infinity in a column, duplicate labels, no usable features -- the
run does not end with a traceback and an empty directory. It writes the audit it
does have, with the verdict ``BLOCKED`` and the reason recorded, because "this
dataset could not be prepared, and here is exactly why" is the most useful thing
an audit tool can say on a bad day.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from aidatasetkit.core.exceptions import AIDatasetKitError
from aidatasetkit.evidence import (
    AuditBuilder,
    Verdict,
    canonical_json,
    render_report,
    verdict_at_least,
)

__all__ = ["main", "EXIT_CODES"]

_logger = logging.getLogger(__name__)

#: What the process returns, and what each value means to a CI job.
#:
#: ==== ==============================================================
#: Code Meaning
#: ==== ==============================================================
#: 0    The run completed and the verdict is below the failure threshold.
#: 1    The command could not run: bad file, bad target, bad option.
#: 2    The verdict reached ``REVIEW_REQUIRED`` and that meets the threshold.
#: 3    The verdict reached ``BLOCKED`` and that meets the threshold.
#: ==== ==============================================================
#:
#: Note that 2 and 3 are *policy* outcomes, not errors: the tool worked, and it
#: is reporting what it found. Only 1 means the audit itself failed.
EXIT_CODES: dict[str, int] = {"ok": 0, "usage": 1, "review": 2, "blocked": 3}

#: ``--fail-on`` values, mapped to the least serious verdict that trips them.
#: ``never`` is handled separately and deliberately absent here, so the mapping
#: contains only thresholds that mean something.
_FAIL_ON: dict[str, Verdict] = {
    "warning": Verdict.READY_WITH_WARNINGS,
    "review": Verdict.REVIEW_REQUIRED,
    "error": Verdict.BLOCKED,
}

#: Every accepted ``--fail-on`` value, including the one that never fails.
_FAIL_ON_CHOICES: tuple[str, ...] = ("never", "warning", "review", "error")


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser. Separate so tests can inspect the contract."""
    parser = argparse.ArgumentParser(
        prog="aidatasetkit",
        description=(
            "Audit a tabular dataset before training: profile it, check it for "
            "quality and leakage problems, record what preprocessing would do to "
            "it, and write the evidence to disk."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit",
        help="Inspect a CSV file and write audit.json, lineage.json, and report.html.",
    )
    audit.add_argument("path", type=Path, help="Path to a CSV file.")
    audit.add_argument(
        "--target", default=None, help="Column holding the label, if there is one."
    )
    audit.add_argument(
        "--task",
        default=None,
        choices=["classification", "regression"],
        help="Task hint. Detected from the target when omitted.",
    )
    audit.add_argument(
        "--model",
        default=None,
        help=(
            "Optional model context, by canonical name or alias. Used only to "
            "record which capabilities shaped the plan; nothing is trained."
        ),
    )
    audit.add_argument(
        "--output",
        type=Path,
        default=Path("./aidk-audit"),
        help="Directory for the artifacts. Created if missing.",
    )
    audit.add_argument(
        "--fail-on",
        default="review",
        choices=_FAIL_ON_CHOICES,
        help=(
            "Lowest verdict that should make this command exit non-zero. "
            "Defaults to 'review': a dataset with something awaiting a human "
            "decision stops the build, because a safety gate that waves those "
            "through is not a gate. Use 'error' to block only on errors, "
            "'warning' to be stricter, or 'never' to always exit 0."
        ),
    )
    audit.add_argument(
        "--include-values",
        action="store_true",
        help=(
            "Record the most frequent value of each column in plain text. Off by "
            "default: that value is real data and may identify a person."
        ),
    )
    audit.add_argument(
        "--debug", action="store_true", help="Show the full traceback on failure."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code.

    Ordinary user mistakes -- a missing file, an unknown target, an unknown model
    -- are reported as one clear line and exit 1. A traceback is shown only with
    ``--debug``, because a stack trace tells a user nothing they can act on about
    their own data.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _audit(args)
    except AIDatasetKitError as error:
        return _fail(f"{type(error).__name__}: {error}", args)
    except FileNotFoundError as error:
        return _fail(f"File not found: {error.filename}", args)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        return _fail("Interrupted.", args)
    except Exception as error:  # noqa: BLE001 - the CLI is the last line of defence
        return _fail(f"{type(error).__name__}: {error}", args)


def _fail(message: str, args: argparse.Namespace) -> int:
    if getattr(args, "debug", False):
        raise
    print(f"error: {message}", file=sys.stderr)
    return EXIT_CODES["usage"]


def _audit(args: argparse.Namespace) -> int:
    """Run one audit. Every analytical step here belongs to a public component.

    The file checks come first, deliberately, and before any heavy import. Telling
    somebody their path is wrong should not cost the six seconds it takes to load
    pandas and scikit-learn.
    """
    path: Path = args.path
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return EXIT_CODES["usage"]
    if path.suffix.lower() != ".csv":
        print(
            f"error: {path.suffix or 'that file type'} is not supported yet. "
            "The alpha reads CSV only.",
            file=sys.stderr,
        )
        return EXIT_CODES["usage"]

    import pandas as pd

    from aidatasetkit.models import ModelFactory
    from aidatasetkit.preprocessing import PreprocessingPlanner, PreprocessorBuilder
    from aidatasetkit.profiling import DataProfiler, DataQualityInspector, TaskDetector

    frame = pd.read_csv(path)
    if args.target is not None and args.target not in frame.columns:
        print(
            f"error: target {args.target!r} is not a column in {path.name}. "
            f"Columns are: {', '.join(map(str, frame.columns[:12]))}"
            + (" ..." if frame.shape[1] > 12 else ""),
            file=sys.stderr,
        )
        return EXIT_CODES["usage"]

    registration = None
    if args.model is not None:
        registration = ModelFactory.registration(args.model, task=args.task)
        registration.require_available()

    # --- established facts, each from the layer that owns it ---------------
    profile = DataProfiler().profile(frame)
    quality = DataQualityInspector().inspect(frame, profile=profile, target=args.target)
    target_profile = None
    if args.target is not None:
        target_profile = TaskDetector().detect(
            frame[args.target], hint=args.task, target_name=args.target
        )

    plan = None
    lineage = None
    warnings: list[str] = []
    blocked_reason = None
    if registration is not None and args.target is not None:
        try:
            plan = PreprocessingPlanner().plan(
                frame,
                profile,
                registration.capabilities.preprocessing_profile(),
                quality=quality,
                target=args.target,
            )
        except AIDatasetKitError as error:
            # The artifact is more useful now, not less: it can say exactly why
            # this dataset cannot be prepared.
            blocked_reason = f"preprocessing could not be planned: {error}"
            warnings.append(blocked_reason)
            _logger.info("preprocessing planning failed", exc_info=True)

    if plan is not None:
        # Fitting the *preprocessor* is what turns intended lineage into observed
        # lineage: without it the artifact can say a column would be one-hot
        # encoded but not what it became. No estimator is built and no model is
        # trained -- that is S7's work and it is not done here.
        try:
            features = frame.drop(columns=[args.target])
            preprocessor = PreprocessorBuilder().build(plan, features)
            preprocessor.fit(features)
            lineage = preprocessor.lineage()
        except AIDatasetKitError as error:
            blocked_reason = f"preprocessing could not be fitted: {error}"
            warnings.append(blocked_reason)
            _logger.info("preprocessor fitting failed", exc_info=True)

    artifact = AuditBuilder(
        redact_values=not args.include_values, dataset_name=path.name
    ).build(
        frame,
        profile=profile,
        quality=quality,
        target=target_profile,
        plan=plan,
        lineage=lineage,
        model=registration,
        # Deliberately excludes --fail-on: it decides this process's exit code
        # and changes nothing about what was found. Two audits that differ only
        # in that flag are the same audit, and the config fingerprint has to say
        # so.
        settings={"task_hint": args.task, "source_file": path.name},
        warnings=warnings,
        blocked_reason=blocked_reason,
    )

    written = _write(artifact, args.output)
    _print_summary(artifact, written)
    return _exit_code(artifact.verdict, args.fail_on)


def _write(artifact: Any, output: Path) -> dict[str, Path]:
    """Write the three artifacts. JSON first: it is the record, HTML renders it."""
    output.mkdir(parents=True, exist_ok=True)
    payload = artifact.to_dict()
    paths = {
        "audit.json": output / "audit.json",
        "lineage.json": output / "lineage.json",
        "report.html": output / "report.html",
    }
    paths["audit.json"].write_text(canonical_json(payload), encoding="utf-8")
    paths["lineage.json"].write_text(
        canonical_json(artifact.lineage_dict()), encoding="utf-8"
    )
    paths["report.html"].write_text(render_report(payload), encoding="utf-8")
    return paths


def _print_summary(artifact: Any, written: dict[str, Path]) -> None:
    """One screen, not one thousand lines."""
    counts = artifact.findings_by_severity
    dataset = artifact.dataset
    print("AIDatasetKit audit")
    print()
    print(f"Dataset:  {dataset.name or 'dataset'}")
    print(f"          {dataset.row_count:,} rows x {dataset.column_count:,} columns")
    print(f"          fingerprint {dataset.fingerprint[:16]}")
    print()
    print(f"Verdict:  {artifact.verdict.value.replace('_', ' ').upper()}")
    print()
    print(
        f"Findings: {counts['error']} error, {counts['warning']} warning, "
        f"{counts['info']} info"
    )
    highlights = [
        finding
        for finding in artifact.findings
        if finding.severity.value in {"error", "warning"}
    ][:5]
    if highlights:
        print()
        print("Key issues:")
        for finding in highlights:
            where = f"{finding.column.name}: " if finding.column else ""
            print(f"  - {where}{finding.message}")
    if artifact.review_items:
        print()
        print(f"Needs review: {', '.join(artifact.review_items[:5])}")
    print()
    print("Artifacts:")
    for path in written.values():
        print(f"  {path}")
    print()


def _exit_code(verdict: Verdict, fail_on: str) -> int:
    """Map a verdict to a process exit code under the chosen policy."""
    if fail_on == "never":
        return EXIT_CODES["ok"]
    threshold = _FAIL_ON[fail_on]
    if not verdict_at_least(verdict, threshold):
        return EXIT_CODES["ok"]
    if verdict is Verdict.BLOCKED:
        return EXIT_CODES["blocked"]
    if verdict is Verdict.REVIEW_REQUIRED:
        return EXIT_CODES["review"]
    return EXIT_CODES["review"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
