"""Turning findings into one conservative conclusion.

The policy is deliberately small, deliberately typed, and deliberately built on
the severity semantics S2 already established rather than a second opinion about
what counts as serious. It answers a yes/no question a CI job can act on, and it
records *why* it answered that way, because a verdict without its reasons is a
number nobody can argue with.

Nothing here decides whether data is safe. It reports whether the checks that ran
found something that stops a person proceeding without looking.
"""

from __future__ import annotations

from collections.abc import Sequence

from aidatasetkit.core.types import Severity
from aidatasetkit.evidence.types import DecisionEvidence, FindingEvidence, Verdict

__all__ = ["VERDICT_ORDER", "decide_verdict", "verdict_at_least"]

#: Least to most serious. Used for ``--fail-on`` comparisons, so the order is
#: part of the CLI contract rather than an implementation detail.
VERDICT_ORDER: tuple[Verdict, ...] = (
    Verdict.READY,
    Verdict.READY_WITH_WARNINGS,
    Verdict.REVIEW_REQUIRED,
    Verdict.BLOCKED,
)


def verdict_at_least(verdict: Verdict, threshold: Verdict) -> bool:
    """Whether ``verdict`` is at least as serious as ``threshold``."""
    return VERDICT_ORDER.index(verdict) >= VERDICT_ORDER.index(threshold)


def decide_verdict(
    findings: Sequence[FindingEvidence],
    decisions: Sequence[DecisionEvidence] = (),
    *,
    blocked_reason: str | None = None,
) -> tuple[Verdict, tuple[str, ...]]:
    """Return the run's verdict and the reasons behind it.

    The rules, in the order they are applied:

    1. The analysis could not complete -- ``BLOCKED``. Nothing else can be
       concluded from a run that stopped.
    2. Any ``ERROR`` finding -- ``BLOCKED``. S2 reserves that severity for a
       condition it considers disqualifying, and re-litigating it here would put
       two disagreeing opinions in one library.
    3. Anything awaiting a person -- ``REVIEW_REQUIRED``. That is a finding
       flagged ``requires_review``, or a feature the planner held back rather
       than deciding for the analyst.
    4. Any ``WARNING`` -- ``READY_WITH_WARNINGS``.
    5. Otherwise ``READY``, which claims only that these checks found no blocker.

    Args:
        findings: Quality findings recorded in the artifact.
        decisions: Preprocessing decisions, consulted for held-back features.
        blocked_reason: Set when the run itself failed, which outranks everything.

    Returns:
        The verdict, and the human-readable reasons for it in the order applied.
    """
    reasons: list[str] = []

    if blocked_reason:
        return Verdict.BLOCKED, (blocked_reason,)

    errors = [finding for finding in findings if finding.severity is Severity.ERROR]
    if errors:
        for finding in errors:
            where = f" on {finding.column.name}" if finding.column else ""
            reasons.append(f"error finding {finding.code}{where}")
        return Verdict.BLOCKED, tuple(reasons)

    for finding in findings:
        if finding.requires_review:
            where = f" on {finding.column.name}" if finding.column else ""
            reasons.append(f"{finding.code}{where} needs a decision")
    for decision in decisions:
        if decision.requires_review:
            reasons.append(
                f"{decision.feature.name} was held back ({decision.reason_code})"
            )
    if reasons:
        return Verdict.REVIEW_REQUIRED, tuple(reasons)

    warnings = [finding for finding in findings if finding.severity is Severity.WARNING]
    if warnings:
        for finding in warnings:
            where = f" on {finding.column.name}" if finding.column else ""
            reasons.append(f"warning finding {finding.code}{where}")
        return Verdict.READY_WITH_WARNINGS, tuple(reasons)

    return Verdict.READY, ("no finding above informational severity",)
