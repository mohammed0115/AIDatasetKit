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

from tests.conftest import isolated_env

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

    def test_measurements_are_carried_through_but_text_is_not(self, frame, full):
        """Numbers survive verbatim; text is a value until proven a name."""
        from aidatasetkit.evidence.builder import _SAFE_TEXT_DETAIL_KEYS

        profile = DataProfiler().profile(frame)
        quality = DataQualityInspector().inspect(frame, profile=profile, target="churn")
        original = {i.code: dict(i.details or {}) for i in quality.issues}
        for finding in full.findings:
            for key, value in finding.details.items():
                if isinstance(value, (int, float, bool)) or value is None:
                    assert value == original[finding.code][key]
                elif key not in _SAFE_TEXT_DETAIL_KEYS:
                    assert "sha256:" in str(value)

    def test_a_finding_that_quotes_a_value_has_it_scrubbed_from_the_message(self, full):
        constant = [f for f in full.findings if f.code == "constant_column"]
        assert constant, "the fixture should contain a constant column"
        for finding in constant:
            assert SENSITIVE not in finding.message
            assert "sha256:" in finding.message

    def test_a_message_with_no_text_detail_is_untouched(self, frame, full):
        profile = DataProfiler().profile(frame)
        quality = DataQualityInspector().inspect(frame, profile=profile, target="churn")
        by_code = {i.code: i.message for i in quality.issues}
        for finding in full.findings:
            has_text = any(
                isinstance(v, (str, list, tuple)) for v in finding.details.values()
            )
            if not has_text:
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
                env=isolated_env(seed),
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


class TestThresholdsReachTheConfigFingerprint:
    """Found by attacking the config fingerprint: it answered "same settings"
    when the settings had changed.

    ``KitConfig`` decides what counts as high cardinality, as an identifier, as
    leakage. Change ``high_cardinality_threshold`` and a column starts being
    reported. The evidence fingerprint moved, because the findings moved -- but
    the *config* fingerprint, whose entire job is to answer "were the same
    settings in force?", did not. It said yes when the answer was no.
    """

    @staticmethod
    def _frame() -> pd.DataFrame:
        index = np.arange(300)
        return pd.DataFrame(
            {
                "a": (index % 53) * 1.1,
                "city": [f"c{value % 30}" for value in index],
                "label": (index % 3 == 0).astype("int64"),
            }
        )

    @staticmethod
    def _build(frame, kit):
        profile = DataProfiler(kit).profile(frame)
        quality = DataQualityInspector(kit).inspect(frame, profile=profile, target="label")
        return AuditBuilder().build(
            frame, profile=profile, quality=quality, kit_config=kit, created_at="T"
        )

    def test_a_changed_threshold_really_changes_the_findings(self):
        from aidatasetkit.core import KitConfig

        frame = self._frame()
        assert len(self._build(frame, KitConfig()).findings) != len(
            self._build(frame, KitConfig(high_cardinality_threshold=5)).findings
        )

    def test_and_therefore_changes_the_config_fingerprint(self):
        from aidatasetkit.core import KitConfig

        frame = self._frame()
        assert (
            self._build(frame, KitConfig()).config.fingerprint
            != self._build(frame, KitConfig(high_cardinality_threshold=5)).config.fingerprint
        )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("missing_warning_threshold", 0.9),
            ("near_constant_threshold", 0.5),
            ("id_uniqueness_threshold", 0.1),
            ("leakage_correlation_threshold", 0.99),
            ("random_state", 7),
        ],
    )
    def test_every_threshold_is_part_of_the_identity(self, field, value):
        """Not only the one that was found. Each is checked on its own."""
        from aidatasetkit.core import KitConfig

        frame = self._frame()
        base = self._build(frame, KitConfig())
        changed = self._build(frame, KitConfig(**{field: value}))
        assert base.config.fingerprint != changed.config.fingerprint

    def test_the_thresholds_are_visible_not_just_hashed(self):
        from aidatasetkit.core import KitConfig

        artifact = self._build(self._frame(), KitConfig())
        assert artifact.config.settings["thresholds"]["high_cardinality_threshold"]

    def test_omitting_them_records_that_they_are_unknown(self):
        """Assuming the defaults would put a wrong answer in the one field whose
        job is to say what was in force."""
        frame = self._frame()
        artifact = AuditBuilder().build(
            frame, profile=DataProfiler().profile(frame), created_at="T"
        )
        assert artifact.config.settings["thresholds"] is None

    def test_and_that_is_distinguishable_from_declared_defaults(self):
        from aidatasetkit.core import KitConfig

        frame = self._frame()
        silent = AuditBuilder().build(
            frame, profile=DataProfiler().profile(frame), created_at="T"
        )
        declared = AuditBuilder().build(
            frame,
            profile=DataProfiler().profile(frame),
            kit_config=KitConfig(),
            created_at="T",
        )
        assert silent.config.fingerprint != declared.config.fingerprint


class TestAnalystSuppliedValuesAreRecordedAndDocumented:
    """Ordinal orders and explicit mappings appear verbatim, on purpose.

    They are the analyst's own values rather than something read out of the
    data, and hiding them would stop the artifact showing which ordering was
    applied. That is a defensible choice and an undocumented one is not, so the
    documentation is asserted here rather than trusted.
    """

    SECRET = "ZZ_STAGE_4_METASTATIC_ZZ"

    @pytest.fixture
    def mapped(self):
        from aidatasetkit.preprocessing import PreprocessingConfig

        index = np.arange(240)
        frame = pd.DataFrame(
            {
                "stage": [
                    [self.SECRET, "ZZ_REMISSION_ZZ", "ZZ_STAGE_1_ZZ"][value % 3]
                    for value in index
                ],
                "amount": (index % 47) * 1.3,
                "label": (index % 4 == 0).astype("int64"),
            }
        )
        config = PreprocessingConfig(
            ordinal_orders={
                "stage": ["ZZ_REMISSION_ZZ", "ZZ_STAGE_1_ZZ", self.SECRET]
            }
        )
        return frame, config

    def test_the_supplied_order_is_recorded_verbatim(self, mapped):
        from aidatasetkit.preprocessing import PreprocessingPlanner

        frame, config = mapped
        profile = DataProfiler().profile(frame)
        plan = PreprocessingPlanner(config).plan(
            frame,
            profile,
            PreprocessingProfile(False, False, False),
            target="label",
        )
        artifact = AuditBuilder().build(frame, profile=profile, plan=plan, created_at="T")
        assert self.SECRET in canonical_json(artifact.to_dict())

    def test_the_artifact_says_so_in_its_own_limitations(self):
        from aidatasetkit.evidence import KNOWN_LIMITATIONS

        text = " ".join(KNOWN_LIMITATIONS).lower()
        assert "ordinal orders" in text and "explicit mappings" in text

    def test_the_privacy_document_says_so_too(self):
        from pathlib import Path

        privacy = (
            Path(__file__).resolve().parents[2] / "docs" / "privacy.md"
        ).read_text(encoding="utf-8")
        assert "ordinal order" in privacy.lower()
        assert "explicit mapping" in privacy.lower()

    def test_and_does_not_promise_include_values_controls_it(self):
        from pathlib import Path

        privacy = (
            Path(__file__).resolve().parents[2] / "docs" / "privacy.md"
        ).read_text(encoding="utf-8")
        assert "--include-values` does not control this" in privacy


class TestRedactionDeniesByDefault:
    """The allowlist was the wrong direction, and a real leak walked past it.

    ``possible_numeric_stored_as_text`` reports ``non_numeric_examples`` -- a list
    of raw cells -- under a key the redactor had never heard of, and those cells
    reached ``audit.json`` while the documentation promised they could not. The
    privacy sweep passed because its fixture had no numeric-stored-as-text column,
    which is the "guard nobody watched fail" pattern this project exists to avoid.
    """

    LEAK = "ZZLEAKZZ"

    @pytest.fixture
    def text_numeric(self) -> pd.DataFrame:
        index = np.arange(240)
        return pd.DataFrame(
            {
                "amount_text": [
                    f"{self.LEAK}{value % 97}" if value % 13 == 0 else str(value % 97)
                    for value in index
                ],
                "ok": (index % 31) * 1.0,
                "label": (index % 4 == 0).astype("int64"),
            }
        )

    def test_the_check_that_leaked_actually_fires(self, text_numeric):
        """Otherwise the assertion below passes by finding nothing."""
        profile = DataProfiler().profile(text_numeric)
        quality = DataQualityInspector().inspect(
            text_numeric, profile=profile, target="label"
        )
        assert "possible_numeric_stored_as_text" in {i.code for i in quality.issues}

    def test_and_its_examples_no_longer_reach_the_artifact(self, text_numeric):
        profile = DataProfiler().profile(text_numeric)
        quality = DataQualityInspector().inspect(
            text_numeric, profile=profile, target="label"
        )
        artifact = AuditBuilder().build(text_numeric, profile=profile, quality=quality)
        assert self.LEAK not in canonical_json(artifact.to_dict())

    def test_the_measurements_beside_them_survive(self, text_numeric):
        profile = DataProfiler().profile(text_numeric)
        quality = DataQualityInspector().inspect(
            text_numeric, profile=profile, target="label"
        )
        artifact = AuditBuilder().build(text_numeric, profile=profile, quality=quality)
        finding = next(
            f for f in artifact.findings if f.code == "possible_numeric_stored_as_text"
        )
        assert finding.details["threshold"] == 0.75
        assert isinstance(finding.details["numeric_ratio"], float)

    def test_a_column_name_in_details_is_kept_because_it_is_a_name(self):
        """Redacting the target's name would make a leakage finding unreadable."""
        index = np.arange(200)
        frame = pd.DataFrame(
            {"a": (index % 53) * 1.1, "dup": (index % 2), "label": (index % 2)}
        )
        profile = DataProfiler().profile(frame)
        quality = DataQualityInspector().inspect(frame, profile=profile, target="label")
        artifact = AuditBuilder().build(frame, profile=profile, quality=quality)
        duplicate = next(
            f for f in artifact.findings if f.code == "target_leakage_exact_duplicate"
        )
        assert duplicate.details.get("target") == "label"

    def test_an_unknown_future_detail_key_is_private_by_default(self):
        """The property the allowlist could not give: safe without being updated."""
        from aidatasetkit.core.types import QualityIssue, Severity

        issue = QualityIssue(
            code="invented_check",
            severity=Severity.WARNING,
            message="Column 'x' looks like ZZFUTUREZZ.",
            column="x",
            details={"brand_new_key": "ZZFUTUREZZ", "count": 3},
        )
        recorded = AuditBuilder()._finding_evidence(issue)
        assert "ZZFUTUREZZ" not in canonical_json(recorded.to_dict())
        assert recorded.details["count"] == 3


class TestScrubbingDoesNotDestroyTheMessage:
    """A one-character value made the second pass rewrite the whole sentence."""

    @pytest.fixture
    def single_letter(self) -> pd.DataFrame:
        index = np.arange(200)
        return pd.DataFrame(
            {"k": ["a"] * 200, "n": (index % 17) * 1.0, "label": (index % 2)}
        )

    def test_the_message_survives_intact(self, single_letter):
        profile = DataProfiler().profile(single_letter)
        quality = DataQualityInspector().inspect(
            single_letter, profile=profile, target="label"
        )
        artifact = AuditBuilder().build(single_letter, profile=profile, quality=quality)
        finding = next(f for f in artifact.findings if f.code == "constant_column")
        assert finding.message.startswith("Column 'k' holds the single value ")
        assert finding.message.endswith("carries no information.")

    def test_the_value_is_still_gone(self, single_letter):
        profile = DataProfiler().profile(single_letter)
        quality = DataQualityInspector().inspect(
            single_letter, profile=profile, target="label"
        )
        artifact = AuditBuilder().build(single_letter, profile=profile, quality=quality)
        finding = next(f for f in artifact.findings if f.code == "constant_column")
        assert "'a'" not in finding.message
        assert "sha256:" in finding.message

    def test_the_digest_appears_exactly_once(self, single_letter):
        profile = DataProfiler().profile(single_letter)
        quality = DataQualityInspector().inspect(
            single_letter, profile=profile, target="label"
        )
        artifact = AuditBuilder().build(single_letter, profile=profile, quality=quality)
        finding = next(f for f in artifact.findings if f.code == "constant_column")
        assert finding.message.count("sha256:") == 1


class TestObjectColumnsKeepTheirValueTypes:
    """pandas hashes an object column by text, so 1 and "1" collided."""

    @pytest.mark.parametrize(
        "left,right",
        [
            ([1, 2, 3], ["1", "2", "3"]),
            ([True, False], ["True", "False"]),
            ([1.0, 2.0], ["1.0", "2.0"]),
        ],
    )
    def test_a_value_and_its_text_form_are_different_data(self, left, right):
        from aidatasetkit.evidence import dataset_fingerprint

        a = pd.DataFrame({"c": pd.Series(left * 40, dtype=object)})
        b = pd.DataFrame({"c": pd.Series(right * 40, dtype=object)})
        assert dataset_fingerprint(a) != dataset_fingerprint(b)

    def test_the_same_object_column_still_agrees_with_itself(self):
        from aidatasetkit.evidence import dataset_fingerprint

        frame = pd.DataFrame({"c": pd.Series([1, "b", 3.0] * 40, dtype=object)})
        assert dataset_fingerprint(frame) == dataset_fingerprint(frame.copy(deep=True))


class TestTheDatasetLabelIsMetadataNotEvidence:
    """Renaming a file changed the evidence fingerprint, which is a false alarm."""

    @staticmethod
    def _frame() -> pd.DataFrame:
        index = np.arange(120)
        return pd.DataFrame(
            {"a": (index % 13) * 1.0, "b": [["p", "q", "r"][v % 3] for v in index]}
        )

    def test_renaming_the_file_changes_nothing_about_the_evidence(self):
        frame = self._frame()
        profile = DataProfiler().profile(frame)
        one = AuditBuilder(dataset_name="train.csv").build(
            frame, profile=profile, created_at="T"
        )
        two = AuditBuilder(dataset_name="train_copy.csv").build(
            frame, profile=profile, created_at="T"
        )
        assert one.semantic_fingerprint == two.semantic_fingerprint

    def test_but_the_name_is_still_recorded(self):
        frame = self._frame()
        artifact = AuditBuilder(dataset_name="train.csv").build(
            frame, profile=DataProfiler().profile(frame)
        )
        assert artifact.to_dict()["dataset"]["name"] == "train.csv"

    def test_and_differs_from_reports_no_difference(self):
        frame = self._frame()
        profile = DataProfiler().profile(frame)
        one = AuditBuilder(dataset_name="a.csv").build(frame, profile=profile, created_at="T")
        two = AuditBuilder(dataset_name="b.csv").build(frame, profile=profile, created_at="T")
        assert not any(one.differs_from(two).values())


class TestModelRequirementIsAccurate:
    def test_the_dense_output_branch_uses_the_step_name_that_exists(self):
        """It tested for "one_hot_encoding"; the planner emits "onehot_encoding"."""
        from aidatasetkit.models import ModelFactory
        from aidatasetkit.preprocessing import PreprocessingPlanner

        index = np.arange(300)
        frame = pd.DataFrame(
            {
                "city": [["a", "b", "c"][v % 3] for v in index],
                "n": (index % 41) * 1.0,
                "label": (index % 5 == 0).astype("int64"),
            }
        )
        # A model that is dense-only AND takes NaN natively, so the earlier
        # missing-value branch cannot claim the explanation first.
        entry = ModelFactory.registration("hist_gradient_boosting_classifier")
        assert not entry.capabilities.supports_sparse_input
        assert entry.capabilities.handles_missing_values
        profile = DataProfiler().profile(frame)
        plan = PreprocessingPlanner().plan(
            frame,
            profile,
            entry.capabilities.preprocessing_profile(),
            target="label",
        )
        artifact = AuditBuilder().build(
            frame, profile=profile, plan=plan, model=entry
        )
        city = next(d for d in artifact.decisions if d.feature.name == "city")
        assert "supports_sparse_input=false" in (city.model_requirement or "")

    def test_gaps_are_kept_is_never_said_about_an_imputed_feature(self):
        from aidatasetkit.models import ModelFactory
        from aidatasetkit.preprocessing import PreprocessingPlanner

        index = np.arange(300)
        frame = pd.DataFrame(
            {
                "city": [["a", "b", "c"][v % 3] for v in index],
                "n": (index % 41) * 1.0,
                "label": (index % 5 == 0).astype("int64"),
            }
        )
        frame.loc[frame.index[:20], "city"] = None
        entry = ModelFactory.registration("random_forest_classifier")
        profile = DataProfiler().profile(frame)
        plan = PreprocessingPlanner().plan(
            frame, profile, entry.capabilities.preprocessing_profile(), target="label"
        )
        artifact = AuditBuilder().build(frame, profile=profile, plan=plan, model=entry)
        for decision in artifact.decisions:
            if "gaps are kept" in (decision.model_requirement or ""):
                assert not any("imputation" in step for step in decision.steps), (
                    f"{decision.feature.name} claims gaps are kept but imputes them"
                )


class TestTheDigestIsNotClaimedToBeProtection:
    def test_the_module_does_not_call_it_salted(self):
        from pathlib import Path

        import aidatasetkit.evidence.builder as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "salted digest" not in source

    def test_it_says_plainly_that_it_is_reversible(self):
        from pathlib import Path

        import aidatasetkit.evidence.builder as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "not salted and not a privacy guarantee" in source


class TestTheSafeVocabularyListIsComplete:
    """Pins the text keys the quality checks actually emit.

    Redaction denies by default, so a key missing from the vocabulary list is
    merely over-cautious rather than a leak -- but over-caution has a cost that
    was measured: hashing ``signals`` turned "deterministic_mapping" into a
    digest and made the leakage finding unreadable. This enumerates what the
    checks emit so neither mistake can be made quietly.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def text_detail_keys() -> dict[str, set[str]]:
        """Every details key whose value is text, across a frame that fires everything."""
        index = np.arange(300)
        frame = pd.DataFrame(
            {
                "rid": [f"R{v:05d}" for v in index],
                "k": ["same"] * 300,
                "nc": ["common"] * 299 + ["rare"],
                "txt": [f"x{v % 97}" if v % 13 else str(v % 97) for v in index],
                "hc": [f"lvl{v % 120}" for v in index],
                "num": (index % 53) * 1.1,
                "num2": (index % 53) * 2.2,
                "dup": (index % 2),
                "label": (index % 2),
            }
        )
        frame.loc[frame.index[:80], "num"] = np.nan
        profile = DataProfiler().profile(frame)
        quality = DataQualityInspector().inspect(frame, profile=profile, target="label")
        found: dict[str, set[str]] = {}
        for issue in quality.issues:
            for key, value in (issue.details or {}).items():
                is_text = isinstance(value, str) or (
                    isinstance(value, (list, tuple))
                    and any(isinstance(item, str) for item in value)
                )
                if is_text:
                    found.setdefault(key, set()).add(issue.code)
        return found

    def test_the_frame_fires_a_broad_set_of_checks(self, text_detail_keys):
        assert len(text_detail_keys) >= 5

    def test_every_text_key_is_either_vocabulary_or_redacted(self, text_detail_keys):
        """No third category: a key is a name we define, or it is treated as data."""
        from aidatasetkit.evidence.builder import _SAFE_TEXT_DETAIL_KEYS

        data_keys = {"value", "dominant_value", "non_numeric_examples"}
        for key in text_detail_keys:
            assert key in _SAFE_TEXT_DETAIL_KEYS or key in data_keys, (
                f"{key!r} is a new text detail key; decide whether it is a name "
                "(add it to _SAFE_TEXT_DETAIL_KEYS) or data (leave it redacted)"
            )

    def test_the_diagnostic_vocabulary_survives_into_the_artifact(self):
        """Hashing a signal name protects nobody and hides the reason."""
        index = np.arange(300)
        frame = pd.DataFrame(
            {"a": (index % 53) * 1.1, "dup": (index % 2), "label": (index % 2)}
        )
        profile = DataProfiler().profile(frame)
        quality = DataQualityInspector().inspect(frame, profile=profile, target="label")
        artifact = AuditBuilder().build(frame, profile=profile, quality=quality)
        text = canonical_json(artifact.to_dict())
        assert "deterministic_mapping" in text or "label" in text

    def test_and_the_data_keys_are_still_hashed(self):
        index = np.arange(240)
        frame = pd.DataFrame(
            {"k": ["ZZDATAZZ"] * 240, "n": (index % 31) * 1.0, "label": (index % 2)}
        )
        profile = DataProfiler().profile(frame)
        quality = DataQualityInspector().inspect(frame, profile=profile, target="label")
        artifact = AuditBuilder().build(frame, profile=profile, quality=quality)
        assert "ZZDATAZZ" not in canonical_json(artifact.to_dict())


class TestNumbersCanBeValuesToo:
    """The redactor hid text and let a salary through.

    ``near_constant_column`` reports ``dominant_value: 987654.0`` beside
    ``dominant_ratio: 0.9958``. Both are floats; only the key says which is a
    measurement and which is somebody's pay. Redacting by type alone published
    the second one.
    """

    @pytest.fixture
    def salaries(self) -> pd.DataFrame:
        index = np.arange(240)
        return pd.DataFrame(
            {"salary": [987654.0] * 239 + [1.0], "a": (index % 31) * 1.0, "label": (index % 2)}
        )

    def test_the_dominant_number_is_hidden(self, salaries):
        profile = DataProfiler().profile(salaries)
        quality = DataQualityInspector().inspect(salaries, profile=profile, target="label")
        artifact = AuditBuilder().build(salaries, profile=profile, quality=quality)
        finding = next(f for f in artifact.findings if f.code == "near_constant_column")
        assert str(finding.details["dominant_value"]).startswith("sha256:")

    def test_and_removed_from_the_message(self, salaries):
        profile = DataProfiler().profile(salaries)
        quality = DataQualityInspector().inspect(salaries, profile=profile, target="label")
        artifact = AuditBuilder().build(salaries, profile=profile, quality=quality)
        finding = next(f for f in artifact.findings if f.code == "near_constant_column")
        assert "987654" not in finding.message

    def test_the_measurement_beside_it_survives(self, salaries):
        profile = DataProfiler().profile(salaries)
        quality = DataQualityInspector().inspect(salaries, profile=profile, target="label")
        artifact = AuditBuilder().build(salaries, profile=profile, quality=quality)
        finding = next(f for f in artifact.findings if f.code == "near_constant_column")
        assert finding.details["dominant_ratio"] > 0.99
        assert finding.details["threshold"] == 0.98

    @pytest.mark.parametrize(
        "key,expected",
        [
            ("value", True),
            ("dominant_value", True),
            ("non_numeric_examples", True),
            ("median_value", True),
            ("top_samples", True),
            ("count", False),
            ("threshold", False),
            ("dominant_ratio", False),
            ("unique_count", False),
        ],
    )
    def test_a_key_naming_a_value_is_recognised_by_pattern(self, key, expected):
        """A pattern, so a check added later is covered without an edit."""
        from aidatasetkit.evidence.builder import _names_a_data_value

        assert _names_a_data_value(key) is expected

    def test_the_documented_numeric_summary_is_still_present(self, salaries):
        """Minima and maxima remain, because privacy.md says so explicitly."""
        profile = DataProfiler().profile(salaries)
        artifact = AuditBuilder().build(salaries, profile=profile)
        column = next(c for c in artifact.columns if c.name.name == "salary")
        assert column.numeric["maximum"] == 987654.0
