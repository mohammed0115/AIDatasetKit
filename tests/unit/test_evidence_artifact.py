"""The audit artifact: what it records, what it refuses to record, and its verdict.

Three properties carry most of the weight here and each has its own class:
evidence is *aggregated* rather than recomputed, raw data does *not* appear by
default, and building an artifact changes nothing it was given.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core.types import PreprocessingProfile, Severity
from aidatasetkit.evidence import (
    ARTIFACT_SCHEMA_VERSION,
    AuditBuilder,
    AuditStage,
    FitScope,
    LabelRef,
    Verdict,
    canonical_json,
    decide_verdict,
    render_report,
    verdict_at_least,
)
from aidatasetkit.evidence.types import DecisionEvidence, FindingEvidence
from aidatasetkit.models import ModelFactory
from aidatasetkit.preprocessing import PreprocessingPlanner, PreprocessorBuilder
from aidatasetkit.profiling import DataProfiler, DataQualityInspector, TaskDetector

SENSITIVE = "aisha.al-otaibi@example.com"


def sensitive_frame(n: int = 200) -> pd.DataFrame:
    """A frame whose most frequent values are things nobody should publish."""
    index = np.arange(n)
    age = (22 + index % 40).astype("float64")
    age[index % 9 == 0] = np.nan
    return pd.DataFrame(
        {
            "email": [SENSITIVE] * n,
            "diagnosis_note": ["patient reports chest pain"] * n,
            "age": age,
            "city": [["Riyadh", "Jeddah"][value % 2] for value in index],
            "churn": (index % 4 == 0).astype("int64"),
        }
    )


def facts(frame: pd.DataFrame, target: str = "churn", model: str | None = None):
    """Produce the established facts an audit aggregates. No evidence yet."""
    profile = DataProfiler().profile(frame)
    quality = DataQualityInspector().inspect(frame, profile=profile, target=target)
    detected = TaskDetector().detect(frame[target], target_name=target)
    registration = ModelFactory.registration(model) if model else None
    plan = None
    lineage = None
    if registration is not None:
        plan = PreprocessingPlanner().plan(
            frame,
            profile,
            registration.capabilities.preprocessing_profile(),
            quality=quality,
            target=target,
        )
        features = frame.drop(columns=[target])
        preprocessor = PreprocessorBuilder().build(plan, features)
        preprocessor.fit(features)
        lineage = preprocessor.lineage()
    return profile, quality, detected, plan, lineage, registration


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return sensitive_frame()


@pytest.fixture(scope="module")
def full(frame):
    profile, quality, target, plan, lineage, model = facts(
        frame, model="logistic_regression"
    )
    return AuditBuilder(dataset_name="train.csv").build(
        frame,
        profile=profile,
        quality=quality,
        target=target,
        plan=plan,
        lineage=lineage,
        model=model,
    )


class TestPrivacyIsTheDefault:
    """The line this artifact does not cross without being asked."""

    def test_the_most_frequent_value_never_appears(self, full):
        assert SENSITIVE not in canonical_json(full.to_dict())

    def test_free_text_never_appears_either(self, full):
        assert "chest pain" not in canonical_json(full.to_dict())

    def test_it_is_recorded_as_a_digest_instead(self, full):
        email = next(c for c in full.columns if c.name.name == "email")
        assert email.dominant_value_digest.startswith("sha256:")

    def test_the_digest_still_answers_the_question_it_is_there_for(self, frame):
        """Has the most common value changed since the last run?"""
        profile, quality, target, _, _, _ = facts(frame)
        first = AuditBuilder().build(frame, profile=profile, quality=quality, target=target)
        moved = frame.copy()
        moved["email"] = "someone.else@example.com"
        profile2, quality2, target2, _, _, _ = facts(moved)
        second = AuditBuilder().build(
            moved, profile=profile2, quality=quality2, target=target2
        )
        digests = [
            (a.name.name, a.dominant_value_digest, b.dominant_value_digest)
            for a, b in zip(first.columns, second.columns)
        ]
        changed = {name for name, one, two in digests if one != two}
        assert changed == {"email"}

    def test_the_ratio_survives_because_it_is_not_the_value(self, full):
        email = next(c for c in full.columns if c.name.name == "email")
        assert email.dominant_ratio == 1.0

    def test_an_analyst_can_opt_in_explicitly(self, frame):
        profile, quality, target, _, _, _ = facts(frame)
        artifact = AuditBuilder(redact_values=False).build(
            frame, profile=profile, quality=quality, target=target
        )
        assert SENSITIVE in canonical_json(artifact.to_dict())

    def test_and_the_choice_is_recorded_in_the_config(self, full):
        assert full.config.settings["redact_values"] is True

    def test_opting_in_changes_the_config_fingerprint(self, frame):
        profile, quality, target, _, _, _ = facts(frame)
        private = AuditBuilder().build(frame, profile=profile, quality=quality, target=target)
        public = AuditBuilder(redact_values=False).build(
            frame, profile=profile, quality=quality, target=target
        )
        assert private.config.fingerprint != public.config.fingerprint


class TestNothingIsRecomputed:
    """The artifact repeats what it was told, exactly."""

    def test_missing_ratios_come_from_the_profile(self, frame, full):
        profile = DataProfiler().profile(frame)
        recorded = {c.name.name: c.missing_ratio for c in full.columns}
        measured = {str(c.name): c.missing_ratio for c in profile.column_profiles}
        assert recorded == measured

    def test_findings_come_from_the_quality_report(self, frame, full):
        profile = DataProfiler().profile(frame)
        quality = DataQualityInspector().inspect(frame, profile=profile, target="churn")
        assert [f.code for f in full.findings] == [i.code for i in quality.issues]

    def test_finding_details_are_carried_through_except_raw_values(self, frame, full):
        """Measurements survive verbatim; the two keys holding data do not."""
        from aidatasetkit.evidence.builder import _REDACTED_DETAIL_KEYS

        profile = DataProfiler().profile(frame)
        quality = DataQualityInspector().inspect(frame, profile=profile, target="churn")
        original = {i.code: dict(i.details or {}) for i in quality.issues}
        for finding in full.findings:
            for key, value in finding.details.items():
                if key in _REDACTED_DETAIL_KEYS:
                    assert str(value).startswith("sha256:")
                else:
                    assert value == original[finding.code][key]

    def test_a_finding_that_quotes_a_value_has_it_scrubbed_from_the_message(self, full):
        constant = [f for f in full.findings if f.code == "constant_column"]
        assert constant, "the fixture should contain a constant column"
        for finding in constant:
            assert SENSITIVE not in finding.message
            assert "sha256:" in finding.message

    def test_every_other_finding_message_is_untouched(self, frame, full):
        from aidatasetkit.evidence.builder import _REDACTED_DETAIL_KEYS

        profile = DataProfiler().profile(frame)
        quality = DataQualityInspector().inspect(frame, profile=profile, target="churn")
        by_code = {i.code: i.message for i in quality.issues}
        for finding in full.findings:
            if not set(finding.details) & _REDACTED_DETAIL_KEYS:
                assert finding.message == by_code[finding.code]

    def test_decisions_come_from_the_plan(self, frame, full):
        _, _, _, plan, _, _ = facts(frame, model="logistic_regression")
        assert [d.feature.name for d in full.decisions] == [
            str(d.feature) for d in plan.decisions
        ]

    def test_lineage_outputs_come_from_the_preprocessor_not_from_name_parsing(
        self, frame, full
    ):
        _, _, _, _, lineage, _ = facts(frame, model="logistic_regression")
        recorded = {
            entry.source.name: tuple(entry.outputs)
            for entry in full.lineage
            if entry.outputs
        }
        expected = {str(k): tuple(v) for k, v in lineage.items()}
        assert recorded == expected


class TestBuildingChangesNothing:
    def test_the_frame_is_untouched(self, frame):
        before = frame.copy(deep=True)
        profile, quality, target, plan, lineage, model = facts(
            frame, model="logistic_regression"
        )
        AuditBuilder().build(
            frame,
            profile=profile,
            quality=quality,
            target=target,
            plan=plan,
            lineage=lineage,
            model=model,
        )
        pd.testing.assert_frame_equal(frame, before)

    def test_the_profile_and_report_are_untouched(self, frame):
        profile, quality, target, _, _, _ = facts(frame)
        before_profile = copy.deepcopy(profile.to_dict())
        before_quality = copy.deepcopy(quality.to_dict())
        AuditBuilder().build(frame, profile=profile, quality=quality, target=target)
        assert profile.to_dict() == before_profile
        assert quality.to_dict() == before_quality

    def test_the_plan_is_untouched(self, frame):
        _, _, _, plan, _, _ = facts(frame, model="logistic_regression")
        before = copy.deepcopy(plan.to_dict())
        profile = DataProfiler().profile(frame)
        AuditBuilder().build(frame, profile=profile, plan=plan)
        assert plan.to_dict() == before

    def test_the_artifact_itself_cannot_be_edited(self, full):
        with pytest.raises(Exception):
            full.verdict = Verdict.READY


class TestSchemaVersionAndStages:
    def test_the_artifact_declares_its_own_schema_version(self, full):
        assert full.schema_version == ARTIFACT_SCHEMA_VERSION == "1.0"

    def test_the_version_is_in_the_serialised_form(self, full):
        assert json.loads(canonical_json(full.to_dict()))["schema_version"] == "1.0"

    def test_it_is_not_the_package_version(self):
        from aidatasetkit.core.provenance import package_version

        assert ARTIFACT_SCHEMA_VERSION != package_version("aidatasetkit")

    def test_inspection_only_is_a_valid_artifact(self, frame):
        profile, quality, target, _, _, _ = facts(frame)
        artifact = AuditBuilder().build(
            frame, profile=profile, quality=quality, target=target
        )
        assert artifact.stage is AuditStage.INSPECTED
        assert artifact.decisions == () and artifact.lineage == ()

    def test_a_plan_without_a_fit_is_planned(self, frame):
        profile, quality, target, plan, _, _ = facts(frame, model="logistic_regression")
        artifact = AuditBuilder().build(
            frame, profile=profile, quality=quality, target=target, plan=plan
        )
        assert artifact.stage is AuditStage.PLANNED
        assert artifact.decisions
        assert all(not entry.observed for entry in artifact.lineage)

    def test_a_fitted_preprocessor_is_prepared(self, full):
        assert full.stage is AuditStage.PREPARED
        assert any(entry.observed for entry in full.lineage)

    def test_a_run_with_no_target_still_produces_evidence(self, frame):
        profile = DataProfiler().profile(frame)
        artifact = AuditBuilder().build(frame, profile=profile)
        assert artifact.target is None
        assert artifact.columns
        assert artifact.verdict is Verdict.READY


class TestFitScopeIsRecorded:
    def test_learned_steps_are_marked_training_only(self, full):
        age = next(d for d in full.decisions if d.feature.name == "age")
        assert age.fit_scope is FitScope.TRAINING_ONLY

    def test_an_excluded_feature_reaches_no_transformer(self, full):
        excluded = [d for d in full.decisions if d.action != "include"]
        assert excluded
        assert all(d.fit_scope is FitScope.NOT_APPLICABLE for d in excluded)

    def test_every_decision_carries_a_scope(self, full):
        assert all(isinstance(d.fit_scope, FitScope) for d in full.decisions)


class TestModelEvidence:
    def test_the_capabilities_that_shaped_the_plan_are_recorded(self, full):
        assert full.model.canonical_name == "logistic_regression"
        assert full.model.capabilities["requires_scaling"] is True

    def test_the_scaler_names_the_capability_that_asked_for_it(self, full):
        age = next(d for d in full.decisions if d.feature.name == "age")
        assert "requires_scaling=true" in age.model_requirement

    def test_no_estimator_object_reaches_the_artifact(self, full):
        text = canonical_json(full.to_dict())
        assert "LogisticRegression(" not in text
        assert "object at 0x" not in text

    def test_a_run_without_a_model_says_so_rather_than_inventing_one(self, frame):
        profile = DataProfiler().profile(frame)
        artifact = AuditBuilder().build(frame, profile=profile)
        assert artifact.model is None


class TestVerdictPolicy:
    @staticmethod
    def _finding(severity, requires_review=False, code="x"):
        return FindingEvidence(
            code=code, severity=severity, message="m", requires_review=requires_review
        )

    def test_no_findings_is_ready(self):
        verdict, reasons = decide_verdict([])
        assert verdict is Verdict.READY and reasons

    def test_info_alone_is_still_ready(self):
        verdict, _ = decide_verdict([self._finding(Severity.INFO)])
        assert verdict is Verdict.READY

    def test_a_warning_is_ready_with_warnings(self):
        verdict, _ = decide_verdict([self._finding(Severity.WARNING)])
        assert verdict is Verdict.READY_WITH_WARNINGS

    def test_anything_awaiting_a_person_is_review_required(self):
        verdict, _ = decide_verdict(
            [self._finding(Severity.WARNING, requires_review=True)]
        )
        assert verdict is Verdict.REVIEW_REQUIRED

    def test_an_error_blocks(self):
        verdict, _ = decide_verdict([self._finding(Severity.ERROR)])
        assert verdict is Verdict.BLOCKED

    def test_an_error_outranks_a_review_item(self):
        verdict, _ = decide_verdict(
            [
                self._finding(Severity.ERROR, code="e"),
                self._finding(Severity.WARNING, requires_review=True, code="w"),
            ]
        )
        assert verdict is Verdict.BLOCKED

    def test_a_held_back_feature_asks_for_review(self):
        decision = DecisionEvidence(
            feature=LabelRef.of("city"),
            role="nominal",
            action="review",
            reason_code="high_cardinality",
            reason="too many levels",
            requires_review=True,
        )
        verdict, reasons = decide_verdict([], [decision])
        assert verdict is Verdict.REVIEW_REQUIRED
        assert "city" in reasons[0]

    def test_a_failed_run_blocks_whatever_else_was_found(self):
        verdict, reasons = decide_verdict(
            [self._finding(Severity.INFO)], blocked_reason="infinity in ratio"
        )
        assert verdict is Verdict.BLOCKED
        assert reasons == ("infinity in ratio",)

    def test_the_reasons_are_always_recorded(self):
        for severity in Severity:
            _, reasons = decide_verdict([self._finding(severity)])
            assert reasons

    @pytest.mark.parametrize(
        "verdict,threshold,expected",
        [
            (Verdict.READY, Verdict.READY_WITH_WARNINGS, False),
            (Verdict.READY_WITH_WARNINGS, Verdict.READY_WITH_WARNINGS, True),
            (Verdict.REVIEW_REQUIRED, Verdict.BLOCKED, False),
            (Verdict.BLOCKED, Verdict.REVIEW_REQUIRED, True),
        ],
    )
    def test_severity_ordering(self, verdict, threshold, expected):
        assert verdict_at_least(verdict, threshold) is expected


class TestArtifactIdentityAndDiffing:
    def test_the_same_run_twice_has_the_same_semantic_fingerprint(self, frame):
        profile, quality, target, _, _, _ = facts(frame)
        one = AuditBuilder().build(
            frame, profile=profile, quality=quality, target=target, created_at="A"
        )
        two = AuditBuilder().build(
            frame, profile=profile, quality=quality, target=target, created_at="B"
        )
        assert one.semantic_fingerprint == two.semantic_fingerprint

    def test_the_clock_does_not_reach_identity(self, frame):
        profile = DataProfiler().profile(frame)
        one = AuditBuilder().build(frame, profile=profile, created_at="2020-01-01T00:00:00Z")
        two = AuditBuilder().build(frame, profile=profile, created_at="2099-12-31T23:59:59Z")
        assert one.semantic_fingerprint == two.semantic_fingerprint
        assert one.created_at != two.created_at

    def test_but_it_is_still_recorded(self, full):
        assert full.to_dict()["provenance"]["created_at"]

    def test_a_changed_dataset_moves_the_evidence_fingerprint(self, frame):
        profile = DataProfiler().profile(frame)
        one = AuditBuilder().build(frame, profile=profile)
        changed = frame.copy()
        changed.loc[changed.index[0], "age"] = 999.0
        two = AuditBuilder().build(changed, profile=DataProfiler().profile(changed))
        assert one.semantic_fingerprint != two.semantic_fingerprint

    def test_differs_from_names_what_moved(self, frame):
        profile = DataProfiler().profile(frame)
        one = AuditBuilder().build(frame, profile=profile)
        changed = frame.copy()
        changed.loc[changed.index[0], "age"] = 999.0
        two = AuditBuilder().build(changed, profile=DataProfiler().profile(changed))
        differences = one.differs_from(two)
        assert differences["dataset"] is True
        assert differences["schema"] is False

    def test_two_identical_runs_diff_to_nothing(self, frame):
        profile = DataProfiler().profile(frame)
        one = AuditBuilder().build(frame, profile=profile, created_at="A")
        two = AuditBuilder().build(frame, profile=profile, created_at="B")
        assert not any(one.differs_from(two).values())

    def test_the_serialised_form_is_byte_stable(self, frame):
        profile = DataProfiler().profile(frame)
        one = AuditBuilder().build(frame, profile=profile, created_at="A")
        two = AuditBuilder().build(frame, profile=profile, created_at="A")
        assert canonical_json(one.to_dict()) == canonical_json(two.to_dict())


class TestKnownLimitations:
    def test_every_artifact_carries_them(self, full):
        assert len(full.known_limitations) >= 5

    def test_they_are_bounded_rather_than_reassuring(self, full):
        text = " ".join(full.known_limitations).lower()
        for forbidden in ("guaranteed", "certified", "compliant", "leakage-free"):
            assert forbidden not in text

    def test_no_verdict_claims_safety(self):
        assert {v.value for v in Verdict} == {
            "ready",
            "ready_with_warnings",
            "review_required",
            "blocked",
        }


class TestReportRendersFromTheCanonicalRecord:
    def test_the_verdict_agrees_with_the_json(self, full):
        payload = full.to_dict()
        assert payload["verdict"].replace("_", " ").upper() in render_report(payload)

    def test_the_dataset_fingerprint_agrees(self, full):
        payload = full.to_dict()
        assert payload["dataset"]["fingerprint"] in render_report(payload)

    def test_every_finding_code_appears(self, full):
        payload = full.to_dict()
        html = render_report(payload)
        for finding in payload["findings"]:
            assert finding["code"] in html

    def test_the_report_leaks_nothing_the_json_does_not(self, full):
        assert SENSITIVE not in render_report(full.to_dict())

    def test_it_is_a_standalone_page_with_no_external_requests(self, full):
        html = render_report(full.to_dict())
        assert html.startswith("<!doctype html>")
        for forbidden in ("http://", "https://", "<script"):
            assert forbidden not in html

    def test_html_special_characters_in_a_column_name_are_escaped(self):
        frame = pd.DataFrame({"<script>x": [1.0, 2.0] * 30, "b": [3.0, 4.0] * 30})
        artifact = AuditBuilder().build(frame, profile=DataProfiler().profile(frame))
        html = render_report(artifact.to_dict())
        assert "<script>x" not in html
        assert "&lt;script&gt;x" in html


class TestDeterminismAcrossInterpreters:
    """Hash randomisation must not reach the evidence.

    Built in separate processes under different seeds, with the timestamp and
    environment held constant, so the only thing that could differ is dict and
    set ordering inside the library.
    """

    SCRIPT = (
        "import numpy as np, pandas as pd;"
        "from aidatasetkit.evidence import AuditBuilder, canonical_json;"
        "from aidatasetkit.profiling import DataProfiler, DataQualityInspector;"
        "from aidatasetkit.models import ModelFactory;"
        "from aidatasetkit.preprocessing import PreprocessingPlanner;"
        "i = np.arange(200);"
        "f = pd.DataFrame({'a': (i % 37) * 1.5,"
        " 'b': [['x','y','z'][v % 3] for v in i],"
        " 'c': (i % 2 == 0),"
        " 'label': (i % 4 == 0).astype('int64')});"
        "p = DataProfiler().profile(f);"
        "q = DataQualityInspector().inspect(f, profile=p, target='label');"
        "e = ModelFactory.registration('logistic_regression');"
        "plan = PreprocessingPlanner().plan(f, p, e.capabilities.preprocessing_profile(),"
        " quality=q, target='label');"
        "art = AuditBuilder(dataset_name='d.csv').build(f, profile=p, quality=q,"
        " plan=plan, model=e, created_at='2025-01-01T00:00:00Z');"
        "print(canonical_json(art.semantic_dict(), indent=None));"
        "print(art.semantic_fingerprint)"
    )

    @pytest.fixture(scope="class")
    @staticmethod
    def runs() -> dict[str, tuple[str, str]]:
        import subprocess
        import sys

        results = {}
        for seed in ("0", "1", "999"):
            out = subprocess.run(
                [sys.executable, "-c", TestDeterminismAcrossInterpreters.SCRIPT],
                capture_output=True,
                text=True,
                check=True,
                env={"PYTHONHASHSEED": seed, "PATH": ""},
            ).stdout.strip().splitlines()
            results[seed] = (out[0], out[1])
        return results

    def test_the_canonical_json_is_byte_identical(self, runs):
        assert len({payload for payload, _ in runs.values()}) == 1

    def test_the_evidence_fingerprint_agrees(self, runs):
        assert len({digest for _, digest in runs.values()}) == 1

    def test_all_three_seeds_actually_ran(self, runs):
        assert set(runs) == {"0", "1", "999"}


class TestRoundTrip:
    """to_dict -> JSON -> parse must lose nothing an auditor needs."""

    @pytest.fixture(scope="class")
    @staticmethod
    def parsed(full) -> dict:
        return json.loads(canonical_json(full.to_dict()))

    def test_nothing_is_lost(self, full, parsed):
        assert parsed == json.loads(canonical_json(full.to_dict()))

    @pytest.mark.parametrize(
        "section",
        [
            "schema_version",
            "stage",
            "verdict",
            "verdict_reasons",
            "dataset",
            "config",
            "target",
            "model",
            "columns",
            "findings",
            "decisions",
            "lineage",
            "known_limitations",
            "provenance",
        ],
    )
    def test_every_section_survives(self, parsed, section):
        assert section in parsed

    def test_counts_survive_exactly(self, full, parsed):
        assert len(parsed["findings"]) == len(full.findings)
        assert len(parsed["decisions"]) == len(full.decisions)
        assert len(parsed["lineage"]) == len(full.lineage)
        assert len(parsed["columns"]) == len(full.columns)

    def test_the_fingerprints_survive(self, full, parsed):
        assert parsed["dataset"]["fingerprint"] == full.dataset.fingerprint
        assert parsed["config"]["fingerprint"] == full.config.fingerprint
        assert (
            parsed["provenance"]["semantic_fingerprint"] == full.semantic_fingerprint
        )

    def test_a_reparsed_artifact_still_renders(self, parsed):
        assert render_report(parsed).startswith("<!doctype html>")

    def test_label_types_survive_so_labels_stay_separable(self):
        frame = pd.DataFrame({0: [1.0, 2.0] * 30, "0": [3.0, 4.0] * 30})
        artifact = AuditBuilder().build(frame, profile=DataProfiler().profile(frame))
        parsed = json.loads(canonical_json(artifact.to_dict()))
        labels = {(c["name"]["name"], c["name"]["label_type"]) for c in parsed["columns"]}
        assert labels == {("0", "int"), ("0", "str")}


class TestEvidenceLibraryDoesNotPrint:
    """The library logs; the CLI prints. Mixing them corrupts piped output."""

    def test_no_print_call_exists_in_the_evidence_package(self):
        """Parsed, not grepped: ``semantic_fingerprint(self)`` contains "print(".

        A substring search reports that as a violation, which is the kind of
        false positive that gets a guard deleted rather than fixed.
        """
        import ast
        from pathlib import Path

        import aidatasetkit.evidence as package

        offenders = []
        for path in Path(package.__file__).parent.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                ):
                    offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, f"print() called in the library: {offenders}"

    def test_the_guard_can_actually_fail(self):
        """Proof the AST check finds a print call when there is one."""
        import ast

        tree = ast.parse("def f():\n    print('leak')\n")
        found = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        assert len(found) == 1

    def test_building_an_artifact_writes_nothing_to_stdout(self, frame, capsys):
        profile = DataProfiler().profile(frame)
        AuditBuilder().build(frame, profile=profile)
        captured = capsys.readouterr()
        assert captured.out == ""
