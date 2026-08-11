"""Regenerate the example artifacts committed beside this script.

Two kinds of file are written and they serve different purposes.

``audit.json``, ``lineage.json``, and ``report.html`` are illustrations. A reader
who has not installed anything can open them and see what the tool produces. They
contain a timestamp and the environment they were produced in, so they change
whenever they are regenerated.

``expected_audit_semantic.json`` is a fixture. It is the artifact with the
timestamp and environment removed, which is exactly the part that should not
change unless the evidence itself changes. The test suite compares against it, so
a schema drift or a shifted finding shows up as a failing test with a readable
diff rather than as a surprise in somebody's pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aidatasetkit.evidence import AuditBuilder, canonical_json, render_report
from aidatasetkit.models import ModelFactory
from aidatasetkit.preprocessing import PreprocessingPlanner, PreprocessorBuilder
from aidatasetkit.profiling import DataProfiler, DataQualityInspector, TaskDetector

HERE = Path(__file__).parent
TARGET = "Churn"
MODEL = "logistic_regression"

#: Frozen so the illustrative files do not churn on every regeneration.
FIXED_TIMESTAMP = "2025-02-01T00:00:00Z"


def build_artifact(frame: pd.DataFrame):
    """Run the real pipeline. Nothing here is specific to the example."""
    profile = DataProfiler().profile(frame)
    quality = DataQualityInspector().inspect(frame, profile=profile, target=TARGET)
    detected = TaskDetector().detect(frame[TARGET], target_name=TARGET)
    registration = ModelFactory.registration(MODEL)

    plan = PreprocessingPlanner().plan(
        frame,
        profile,
        registration.capabilities.preprocessing_profile(),
        quality=quality,
        target=TARGET,
    )
    features = frame.drop(columns=[TARGET])
    preprocessor = PreprocessorBuilder().build(plan, features)
    preprocessor.fit(features)

    return AuditBuilder(dataset_name="train.csv").build(
        frame,
        profile=profile,
        quality=quality,
        target=detected,
        plan=plan,
        lineage=preprocessor.lineage(),
        model=registration,
        settings={"source_file": "train.csv", "task_hint": None},
        created_at=FIXED_TIMESTAMP,
    )


def main() -> None:
    frame = pd.read_csv(HERE / "train.csv")
    artifact = build_artifact(frame)
    payload = artifact.to_dict()

    (HERE / "audit.json").write_text(canonical_json(payload), encoding="utf-8")
    (HERE / "lineage.json").write_text(
        canonical_json(artifact.lineage_dict()), encoding="utf-8"
    )
    (HERE / "report.html").write_text(render_report(payload), encoding="utf-8")
    (HERE / "expected_audit_semantic.json").write_text(
        canonical_json(artifact.semantic_dict()), encoding="utf-8"
    )
    print(f"verdict: {artifact.verdict.value}")
    print(f"evidence fingerprint: {artifact.semantic_fingerprint[:16]}")
    for name in ("audit.json", "lineage.json", "report.html", "expected_audit_semantic.json"):
        print(f"  wrote {HERE / name}")


if __name__ == "__main__":
    main()
