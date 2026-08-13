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

from aidatasetkit.core.exceptions import (
    AIDatasetKitError,
    AmbiguousModelAliasError,
    IncompatibleModelError,
)
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
#: 0    The verdict is below the failure threshold.
#: 1    The command could not run: bad file, bad option, unreadable output path.
#: 2    The verdict met the threshold and is not ``BLOCKED``.
#: 3    The verdict is ``BLOCKED`` and that meets the threshold.
#: ==== ==============================================================
#:
#: 2 does not mean ``REVIEW_REQUIRED`` specifically. Under ``--fail-on warning``
#: a ``READY_WITH_WARNINGS`` verdict also returns 2, because the code answers
#: "did this meet the bar you set", not "which verdict was it" -- the verdict
#: itself is in ``audit.json`` and in the summary above it.
#:
#: 2 and 3 are *policy* outcomes, not errors: the tool worked and is reporting
#: what it found. Only 1 means the audit itself could not be produced.
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


class _Parser(argparse.ArgumentParser):
    """An argument parser whose usage errors obey this command's exit codes.

    ``argparse`` exits 2 on a bad flag, and 2 is the code this tool documents as
    "the verdict met your threshold". A CI job seeing 2 could not tell a dataset
    that needs review from a typo in the command line, which makes the whole
    exit-code contract unusable for the job it exists for. Usage errors are
    exit 1, like every other thing the command could not do.
    """

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        print(f"error: {message}", file=sys.stderr)
        raise SystemExit(EXIT_CODES["usage"])


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser. Separate so tests can inspect the contract."""
    from aidatasetkit import __version__

    parser = _Parser(
        prog="aidatasetkit",
        description=(
            "The safety and audit layer for tabular machine learning. Inspect a "
            "dataset before you train on it: profile every column, check for "
            "leakage and quality problems, record what preprocessing would do "
            "and why, and write the evidence to disk."
        ),
        epilog=(
            "Exit codes: 0 the verdict is below your --fail-on threshold; "
            "1 the command could not run; 2 the verdict met the threshold and is "
            "not blocked; 3 the verdict is blocked. Nothing is ever trained."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"aidatasetkit {__version__}",
        help="Show the installed version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)

    audit = subparsers.add_parser(
        "audit",
        help="Inspect a CSV file and write audit.json, lineage.json, and report.html.",
        description=(
            "Audit one CSV file. Writes three artifacts: audit.json (the canonical "
            "machine-readable record), lineage.json (each input column and what it "
            "became), and report.html (the same evidence rendered for a person). "
            "Nothing is trained, and your data is never modified."
        ),
        epilog=(
            "Exit codes: 0 below the --fail-on threshold; 1 the command could not "
            "run; 2 the threshold was met and the verdict is not blocked; 3 the "
            "verdict is blocked. An artifact holds counts, ratios and digests "
            "rather than your data, but it does hold column names, class labels, "
            "numeric minima and maxima, and one-hot category names -- review one "
            "before sharing it outside your team."
        ),
    )
    audit.add_argument("path", type=Path, help="Path to a CSV file.")
    audit.add_argument(
        "--target", default=None, help="Column holding the label, if there is one."
    )
    audit.add_argument(
        "--task",
        default=None,
        choices=["classification", "regression"],
        help=(
            "Task hint, and the way to say which family a shared model alias "
            "means. Detected from the target when omitted. Regression models "
            "and their preprocessing are verified; the readiness verdict is "
            "not, and a regression run says so."
        ),
    )
    audit.add_argument(
        "--model",
        default=None,
        help=(
            "Optional model context, by canonical name or alias. Used only to "
            "record which capabilities shaped the plan; nothing is trained. "
            "Most short aliases name a family and need --task to settle them."
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
            "Turn OFF redaction everywhere: the most frequent value of each "
            "column, and any value quoted inside a quality finding, are written "
            "in plain text. Off by default, because those are real data."
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

    if args.output.exists() and not args.output.is_dir():
        print(
            f"error: --output must be a directory, and {args.output} is a file.",
            file=sys.stderr,
        )
        return EXIT_CODES["usage"]

    import pandas as pd

    from aidatasetkit.core import KitConfig
    from aidatasetkit.models import ModelFactory
    from aidatasetkit.preprocessing import PreprocessingPlanner, PreprocessorBuilder
    from aidatasetkit.profiling import DataProfiler, DataQualityInspector, TaskDetector

    frame = pd.read_csv(path)
    # pandas turns a repeated header into a.1, a.2 and carries on, so the raw
    # header line is read separately to see what the file actually says.
    raw_header = pd.read_csv(path, header=None, nrows=1)
    names = [str(name) for name in raw_header.iloc[0]] if len(raw_header) else []
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        # pandas turns a repeated header into a.1, a.2 and carries on. The audit
        # would then describe columns the file does not contain, and its
        # fingerprint would identify a frame nobody has.
        print(
            "error: the file has duplicate column headers "
            f"({', '.join(duplicated)}), which pandas "
            "renamed to make them unique. An audit of renamed columns would "
            "describe a dataset that does not exist. Give each column its own "
            "name first.",
            file=sys.stderr,
        )
        return EXIT_CODES["usage"]

    if args.target is not None and args.target not in frame.columns:
        print(
            f"error: target {args.target!r} is not a column in {path.name}. "
            f"Columns are: {', '.join(map(str, frame.columns[:12]))}"
            + (" ..." if frame.shape[1] > 12 else ""),
            file=sys.stderr,
        )
        return EXIT_CODES["usage"]

    plan = None
    lineage = None
    warnings: list[str] = []
    blocked_reason: str | None = None

    registration = None
    if args.model is not None:
        if args.target is None:
            # The model context only shapes a preprocessing plan, and a plan
            # needs a target. Recording the model while silently planning
            # nothing would put a model in the artifact that influenced nothing.
            print(
                "error: --model needs --target. A model context only shapes the "
                "preprocessing plan, and a plan needs to know the label column.",
                file=sys.stderr,
            )
            return EXIT_CODES["usage"]
        try:
            registration = ModelFactory.registration(args.model, task=args.task)
        except AmbiguousModelAliasError as error:
            # The registry's message is written for the Python API and says to
            # "Pass task=", a keyword no command line accepts. The candidates it
            # names are the useful part, so they are kept and the instruction is
            # translated into the flag this program actually has.
            print(
                f"error: {error} Here that means --task classification or "
                "--task regression.",
                file=sys.stderr,
            )
            return EXIT_CODES["usage"]
        registration.require_available()

    # --- established facts, each from the layer that owns it ---------------
    # One configuration, used by every layer and recorded in the artifact. Every
    # finding below depends on these thresholds, so the audit has to carry them.
    kit_config = KitConfig()
    profile = DataProfiler(kit_config).profile(frame)
    quality = DataQualityInspector(kit_config).inspect(
        frame, profile=profile, target=args.target
    )
    target_profile = None
    if args.target is not None:
        try:
            target_profile = TaskDetector(kit_config).detect(
                frame[args.target], hint=args.task, target_name=args.target
            )
        except AIDatasetKitError as error:
            # A target nobody can type is exactly the case an audit should
            # explain rather than abort on. The profiling evidence is already in
            # hand, and it is more useful with the reason attached than not at
            # all.
            blocked_reason = f"the target could not be typed: {error}"
            warnings.append(blocked_reason)
            _logger.info("task detection failed", exc_info=True)

    if target_profile is not None and target_profile.task_type.value == "regression":
        # Accepted rather than refused: the profiling and quality evidence is
        # just as useful for a regression target.
        #
        # The warning is narrower than it was. Until S6 nothing regression-shaped
        # had been verified at all, so it said so. S6 held nine regressors to the
        # same executed capability contracts the classifiers pass, and drove them
        # through the same S4 path, so continuing to claim otherwise would be the
        # same failure in reverse -- telling a reader something is unproven when
        # it has been proven. What remains genuinely unverified is named instead.
        warnings.append(
            "This run was audited as a regression task. The profiling, quality, "
            "preprocessing and model-capability evidence applies: regression "
            "models are held to the same verified capability contracts as "
            "classifiers. What has not been verified is the readiness verdict "
            "itself -- its thresholds, and the leakage checks behind it, were "
            "built and measured against classification targets."
        )

    if registration is not None and target_profile is not None:
        # The model was resolved before the target was typed -- it has to be,
        # because --task may be absent and the registry needs *something* to
        # narrow a shared alias. So the compatibility check that
        # ModelFactory.create(name, target=...) performs has to be repeated here,
        # against what the detector actually concluded.
        #
        # Without it an audit will happily record a regressor beside a
        # classification target, with warnings == [] -- an artifact asserting a
        # contradiction, which is worse than one that refuses to be written. The
        # plan is skipped rather than the run aborted, because the profiling and
        # quality evidence is still true and still worth having.
        try:
            registration.capabilities.validate_for(target_profile)
        except IncompatibleModelError as error:
            blocked_reason = (
                f"the model does not match the detected target: {error}"
            )
            warnings.append(blocked_reason)
            _logger.info("model/target task mismatch", exc_info=True)

    if registration is not None and args.target is not None and blocked_reason is None:
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
        kit_config=kit_config,
        # Deliberately excludes --fail-on: it decides this process's exit code
        # and changes nothing about what was found. Two audits that differ only
        # in that flag are the same audit, and the config fingerprint has to say
        # so.
        # source_file is deliberately absent for the same reason --fail-on is:
        # a file name changes nothing about what was found.
        settings={"task_hint": args.task},
        warnings=warnings,
        blocked_reason=blocked_reason,
    )

    written = _write(artifact, args.output)
    _print_summary(artifact, written)
    return _exit_code(artifact.verdict, args.fail_on)


def _write(artifact: Any, output: Path) -> dict[str, Path]:
    """Write the three artifacts, or leave the directory as it was found.

    Everything is rendered into memory before anything reaches disk. Rendering is
    where a failure is plausible -- a label that breaks an assumption, a value
    that will not serialise -- and a directory holding audit.json without the
    report is worse than one holding nothing: the next reader cannot tell a
    finished run from a half-written one.
    """
    payload = artifact.to_dict()
    rendered = {
        "audit.json": canonical_json(payload),
        "lineage.json": canonical_json(artifact.lineage_dict()),
        "report.html": render_report(payload),
    }

    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / name for name in rendered}
    for name, text in rendered.items():
        paths[name].write_text(text, encoding="utf-8")
    return paths


#: How many items the summary shows before saying how many it left out. A screen
#: of five is readable; twenty is a wall nobody reads.
_SHOWN = 5


def _print_summary(artifact: Any, written: dict[str, Path]) -> None:
    """One screen, and honest about what it left off it.

    Every list here is truncated, and every truncation says so. A summary that
    showed five of twenty issues without a word would be read as "there are
    five" -- the same silent narrowing this library refuses to do to data.
    """
    counts = artifact.findings_by_severity
    dataset = artifact.dataset
    print("AIDatasetKit audit")
    print()
    print(f"Dataset:  {dataset.name or 'dataset'}")
    print(f"          {dataset.row_count:,} rows x {dataset.column_count:,} columns")
    print(f"          fingerprint {dataset.fingerprint[:16]}")
    print()
    for warning in artifact.warnings[:_SHOWN]:
        print(f"Note:     {warning}")
    if artifact.warnings:
        print()
    print(f"Verdict:  {artifact.verdict.value.replace('_', ' ').upper()}")
    for reason in artifact.verdict_reasons[:_SHOWN]:
        print(f"          {reason}")
    _print_remainder(len(artifact.verdict_reasons), _SHOWN)
    print()
    print(
        f"Findings: {counts['error']} error, {counts['warning']} warning, "
        f"{counts['info']} info"
    )

    # Most serious first, so a truncated list can never hide the finding that
    # decided the verdict.
    rank = {"error": 0, "warning": 1, "info": 2}
    highlights = sorted(
        (f for f in artifact.findings if f.severity.value in {"error", "warning"}),
        key=lambda f: (rank.get(f.severity.value, 9), f.code),
    )
    if highlights:
        print()
        print("Key issues:")
        for finding in highlights[:_SHOWN]:
            where = f"{finding.column.name}: " if finding.column else ""
            print(f"  - {where}{finding.message}")
        _print_remainder(len(highlights), _SHOWN, indent="  ")
    if artifact.review_items:
        print()
        shown = ", ".join(artifact.review_items[:_SHOWN])
        remaining = len(artifact.review_items) - _SHOWN
        more = f", and {remaining} more" if remaining > 0 else ""
        print(f"Needs review: {shown}{more}")
    print()
    print("Artifacts:")
    for path in written.values():
        print(f"  {path}")
    print()


def _print_remainder(total: int, shown: int, indent: str = "          ") -> None:
    """Say how many items were left out, or nothing when none were."""
    remaining = total - shown
    if remaining > 0:
        print(f"{indent}... and {remaining} more, in audit.json")


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
