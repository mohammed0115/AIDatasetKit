"""The audit as a user meets it: one command, three files, one exit code.

These tests run the real console entry point in a real subprocess against the
golden dataset. Calling the Python functions would prove the library works and
would not prove that the thing a user actually types works, which is the claim a
public alpha has to make.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.cli import EXIT_CODES
from aidatasetkit.evidence import AuditBuilder, Verdict, canonical_json, read_current
from aidatasetkit.profiling import DataProfiler, DataQualityInspector

from tests.golden import semantic_fixture_path

#: The synthetic frame lives beside its generator so the fixture and the example
#: a reader runs are the same file.
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "audit_churn" / "train.csv"


#: Each subprocess pays the full import cost of pandas and scikit-learn, so an
#: invocation is run once and its result reused. Keyed by the exact argument
#: list, which means two tests asking the same question share one process and a
#: test asking a different one still gets its own.
_RUNS: dict[tuple[str, ...], subprocess.CompletedProcess] = {}


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke the CLI the way a user does: as a separate process."""
    if args not in _RUNS:
        _RUNS[args] = subprocess.run(
            [
                sys.executable,
                "-c",
                "from aidatasetkit.cli.main import main; raise SystemExit(main())",
                *args,
            ],
            capture_output=True,
            text=True,
        )
    return _RUNS[args]


@pytest.fixture(scope="module")
def audited(tmp_path_factory) -> tuple[subprocess.CompletedProcess, Path]:
    output = tmp_path_factory.mktemp("audit")
    result = run_cli(
        "audit",
        str(EXAMPLE),
        "--target",
        "Churn",
        "--model",
        "logistic_regression",
        "--output",
        str(output),
    )
    return result, output


@pytest.fixture(scope="module")
def artifact(audited) -> dict:
    _, output = audited
    return json.loads(read_current(output).text("audit.json"))


class TestTheGoldenDatasetExists:
    def test_the_example_is_committed(self):
        assert EXAMPLE.exists(), "run examples/audit_churn/make_dataset.py"

    def test_it_is_synthetic_and_small(self):
        frame = pd.read_csv(EXAMPLE)
        assert len(frame) == 600
        assert list(frame.columns) == [
            "CustomerID",
            "Age",
            "MonthlyCharges",
            "City",
            "ContractType",
            "Education",
            "IsActive",
            "HighCardinalityFeature",
            "Churn",
            "Churn_Copy",
        ]

    def test_regenerating_it_produces_the_same_bytes(self):
        """The fixture is deterministic, so a diff means a real change."""
        from examples.audit_churn.make_dataset import build

        rebuilt = build()
        on_disk = pd.read_csv(EXAMPLE)
        pd.testing.assert_frame_equal(rebuilt, on_disk, check_dtype=False)


class TestTheCommandRuns:
    def test_it_exits_with_the_blocked_code(self, audited):
        result, _ = audited
        assert result.returncode == EXIT_CODES["blocked"], result.stderr

    def test_it_writes_all_three_artifacts(self, audited):
        _, output = audited
        assert set(read_current(output).contents) == {"audit.json", "lineage.json", "report.html"}

    def test_the_summary_is_short_enough_to_read(self, audited):
        result, _ = audited
        assert len(result.stdout.splitlines()) < 40

    def test_the_summary_names_the_verdict(self, audited):
        result, _ = audited
        assert "BLOCKED" in result.stdout

    def test_the_summary_names_the_artifacts(self, audited):
        result, _ = audited
        for name in ("audit.json", "lineage.json", "report.html"):
            assert name in result.stdout

    def test_nothing_is_written_to_stderr_on_success(self, audited):
        result, _ = audited
        assert result.stderr == ""


class TestGoldenFindings:
    """What this dataset is designed to provoke, asserted by code not by eye."""

    def _finding(self, artifact, code, column=None):
        return [
            finding
            for finding in artifact["findings"]
            if finding["code"] == code
            and (column is None or (finding["column"] or {}).get("name") == column)
        ]

    def test_the_exact_target_duplicate_is_an_error(self, artifact):
        found = self._finding(artifact, "target_leakage_exact_duplicate", "Churn_Copy")
        assert found and found[0]["severity"] == "error"

    def test_that_error_is_what_blocks_the_run(self, artifact):
        assert artifact["verdict"] == "blocked"
        assert any("Churn_Copy" in reason for reason in artifact["verdict_reasons"])

    def test_the_identifier_is_flagged_for_review(self, artifact):
        found = self._finding(artifact, "possible_id_like", "CustomerID")
        assert found and found[0]["requires_review"] is True

    def test_the_high_cardinality_column_is_flagged(self, artifact):
        assert self._finding(artifact, "high_cardinality", "HighCardinalityFeature")

    def test_the_missing_values_in_age_are_recorded(self, artifact):
        age = next(c for c in artifact["columns"] if c["name"]["name"] == "Age")
        assert age["missing_count"] > 0
        assert 0.0 < age["missing_ratio"] < 1.0

    def test_the_target_is_described(self, artifact):
        target = artifact["target"]
        assert target["name"]["name"] == "Churn"
        assert target["task_type"] == "classification"
        assert target["is_binary"] is True


class TestGoldenDecisionsAndLineage:
    def _decision(self, artifact, feature):
        return next(
            d for d in artifact["decisions"] if d["feature"]["name"] == feature
        )

    def test_the_identifier_is_held_back_rather_than_used(self, artifact):
        assert self._decision(artifact, "CustomerID")["action"] == "review"

    def test_the_target_duplicate_is_excluded(self, artifact):
        assert self._decision(artifact, "Churn_Copy")["action"] == "exclude"

    def test_the_high_cardinality_column_is_not_encoded(self, artifact):
        assert self._decision(artifact, "HighCardinalityFeature")["action"] == "review"

    def test_the_nominal_column_is_one_hot_encoded(self, artifact):
        assert "onehot_encoding" in self._decision(artifact, "City")["steps"]

    def test_education_is_nominal_because_no_order_was_supplied(self, artifact):
        """An ordinal order is never guessed, so this column is not ordinal."""
        decision = self._decision(artifact, "Education")
        assert decision["role"] != "ordinal"
        assert "ordinal_encoding" not in decision["steps"]

    def test_age_is_imputed(self, artifact):
        assert "median_imputation" in self._decision(artifact, "Age")["steps"]

    def test_scaling_is_present_and_attributed_to_the_model(self, artifact):
        decision = self._decision(artifact, "MonthlyCharges")
        assert "standard_scaling" in decision["steps"]
        assert "requires_scaling=true" in decision["model_requirement"]

    def test_learned_steps_are_marked_training_only(self, artifact):
        assert self._decision(artifact, "Age")["fit_scope"] == "training_only"

    def test_lineage_records_the_one_hot_outputs(self, audited):
        _, output = audited
        lineage = json.loads(read_current(output).text("lineage.json"))
        city = next(e for e in lineage["features"] if e["source"]["name"] == "City")
        assert set(city["outputs"]) == {"City_Dammam", "City_Jeddah", "City_Riyadh"}

    def test_lineage_records_what_produced_nothing(self, audited):
        _, output = audited
        lineage = json.loads(read_current(output).text("lineage.json"))
        held = next(
            e for e in lineage["features"] if e["source"]["name"] == "CustomerID"
        )
        assert held["outputs"] == []
        assert held["action"] == "review"

    def test_lineage_carries_its_own_schema_and_dataset_identity(self, audited, artifact):
        _, output = audited
        lineage = json.loads(read_current(output).text("lineage.json"))
        assert lineage["schema_version"] == artifact["schema_version"]
        assert lineage["dataset_fingerprint"] == artifact["dataset"]["fingerprint"]


class TestReportAndJsonAgree:
    def test_the_report_shows_the_same_verdict(self, audited, artifact):
        _, output = audited
        html = read_current(output).text("report.html")
        assert artifact["verdict"].replace("_", " ").upper() in html

    def test_the_report_shows_the_same_fingerprint(self, audited, artifact):
        _, output = audited
        assert artifact["dataset"]["fingerprint"] in read_current(output).text("report.html")

    def test_every_finding_reaches_the_page(self, audited, artifact):
        _, output = audited
        html = read_current(output).text("report.html")
        for finding in artifact["findings"]:
            assert finding["code"] in html

    def test_every_decision_reaches_the_page(self, audited, artifact):
        _, output = audited
        html = read_current(output).text("report.html")
        for decision in artifact["decisions"]:
            assert decision["feature"]["name"] in html

    def test_the_page_needs_no_network(self, audited):
        _, output = audited
        html = read_current(output).text("report.html")
        assert "http://" not in html and "https://" not in html


class TestExitCodes:
    @pytest.fixture(scope="class")
    @staticmethod
    def missing(tmp_path_factory):
        return run_cli("audit", str(tmp_path_factory.mktemp("gone") / "nope.csv"))

    def test_a_missing_file_is_a_usage_error(self, missing):
        assert missing.returncode == EXIT_CODES["usage"]
        assert "no such file" in missing.stderr

    def test_an_unsupported_file_type_says_so(self, tmp_path):
        path = tmp_path / "data.parquet"
        path.write_text("not really parquet", encoding="utf-8")
        result = run_cli("audit", str(path))
        assert result.returncode == EXIT_CODES["usage"]
        assert "not a supported format" in result.stderr
        assert ".csv and .tsv" in result.stderr

    def test_an_unknown_target_lists_the_real_columns(self, tmp_path):
        result = run_cli("audit", str(EXAMPLE), "--target", "Nope", "--output", str(tmp_path))
        assert result.returncode == EXIT_CODES["usage"]
        assert "CustomerID" in result.stderr

    def test_an_unknown_model_is_refused_clearly(self, tmp_path):
        result = run_cli(
            "audit", str(EXAMPLE), "--target", "Churn", "--model", "random_forrest",
            "--output", str(tmp_path),
        )
        assert result.returncode == EXIT_CODES["usage"]
        assert "random_forest" in result.stderr

    def test_no_traceback_is_shown_for_an_ordinary_mistake(self, missing):
        assert "Traceback" not in missing.stderr

    def test_fail_on_never_exits_zero_even_when_blocked(self, tmp_path):
        result = run_cli(
            "audit", str(EXAMPLE), "--target", "Churn", "--fail-on", "never",
            "--output", str(tmp_path),
        )
        assert result.returncode == EXIT_CODES["ok"]

    def test_a_clean_dataset_exits_zero(self, tmp_path):
        """Every cycle length is coprime with the label's period of two.

        An earlier version used ``["x", "y"][i % 2]`` beside a ``i % 2`` label,
        which makes that column determine the target exactly -- and the audit
        correctly reported leakage and exited 2. The fixture was wrong, not the
        tool, and the shape of that mistake is worth keeping written down.
        """
        clean = tmp_path / "clean.csv"
        index = np.arange(200)
        pd.DataFrame(
            {
                "a": (index % 37) * 1.5,
                "b": [["x", "y", "z"][value % 3] for value in index],
                "label": (index % 2).astype("int64"),
            }
        ).to_csv(clean, index=False)
        result = run_cli(
            "audit", str(clean), "--target", "label", "--output", str(tmp_path / "out")
        )
        assert result.returncode == EXIT_CODES["ok"], result.stdout + result.stderr

    def test_fail_on_warning_is_stricter(self, tmp_path):
        clean = tmp_path / "warn.csv"
        index = np.arange(200)
        frame = pd.DataFrame(
            {
                "a": (index % 37) * 1.5,
                "constant": ["same"] * 200,
                "label": (index % 2).astype("int64"),
            }
        )
        frame.to_csv(clean, index=False)
        lenient = run_cli(
            "audit", str(clean), "--target", "label", "--output", str(tmp_path / "a")
        )
        strict = run_cli(
            "audit", str(clean), "--target", "label", "--fail-on", "warning",
            "--output", str(tmp_path / "b"),
        )
        assert lenient.returncode == EXIT_CODES["ok"]
        assert strict.returncode != EXIT_CODES["ok"]


class TestAuditWithoutAModelOrTarget:
    """An audit must be useful before anything has been decided."""

    @pytest.fixture(scope="class")
    @staticmethod
    def bare(tmp_path_factory):
        output = tmp_path_factory.mktemp("bare")
        result = run_cli("audit", str(EXAMPLE), "--output", str(output))
        return result, json.loads(read_current(output).text("audit.json"))

    @pytest.fixture(scope="class")
    @staticmethod
    def targeted(tmp_path_factory):
        output = tmp_path_factory.mktemp("targeted")
        run_cli("audit", str(EXAMPLE), "--target", "Churn", "--output", str(output))
        return json.loads(read_current(output).text("audit.json"))

    def test_a_bare_audit_still_produces_artifacts(self, bare):
        result, artifact = bare
        assert result.returncode in {
            EXIT_CODES["ok"],
            EXIT_CODES["review"],
            EXIT_CODES["blocked"],
        }
        assert artifact["schema_version"]

    def test_it_records_the_inspection_stage(self, bare):
        _, artifact = bare
        assert artifact["stage"] == "inspected"
        assert artifact["model"] is None
        assert artifact["decisions"] == []

    def test_it_still_profiles_every_column(self, bare):
        _, artifact = bare
        assert len(artifact["columns"]) == 10

    def test_a_target_without_a_model_plans_nothing_but_describes_the_task(self, targeted):
        assert targeted["target"]["name"]["name"] == "Churn"
        assert targeted["stage"] == "inspected"


class TestBlockedAnalysisStillProducesEvidence:
    """The most important failure mode: a dataset that cannot be prepared."""

    @pytest.fixture(scope="class")
    @staticmethod
    def unpreparable(tmp_path_factory) -> Path:
        """Infinity in a numeric column: S4 refuses to plan this."""
        tmp_path = tmp_path_factory.mktemp("inf")
        index = np.arange(200)
        ratio = (index % 17).astype("float64")
        ratio[5] = np.inf
        ratio[6] = -np.inf
        frame = pd.DataFrame(
            {
                "ratio": ratio,
                "city": [["a", "b"][value % 2] for value in index],
                "label": (index % 2).astype("int64"),
            }
        )
        path = tmp_path / "infinite.csv"
        frame.to_csv(path, index=False)
        return path

    @pytest.fixture(scope="class")
    @staticmethod
    def blocked(unpreparable, tmp_path_factory):
        output = tmp_path_factory.mktemp("blocked")
        result = run_cli(
            "audit", str(unpreparable), "--target", "label", "--model",
            "logistic_regression", "--output", str(output),
        )
        return result, output, json.loads(read_current(output).text("audit.json"))

    def test_the_command_does_not_crash(self, blocked):
        result, _, _ = blocked
        assert "Traceback" not in result.stderr

    def test_artifacts_are_still_written(self, blocked):
        _, output, _ = blocked
        assert set(read_current(output).contents) == {"audit.json", "lineage.json", "report.html"}

    def test_the_verdict_explains_what_stopped_it(self, blocked):
        _, _, artifact = blocked
        assert artifact["verdict"] == "blocked"
        assert any("infinit" in reason.lower() for reason in artifact["verdict_reasons"])

    def test_the_profiling_evidence_survives_the_failure(self, blocked):
        _, _, artifact = blocked
        assert len(artifact["columns"]) == 3
        ratio = next(c for c in artifact["columns"] if c["name"]["name"] == "ratio")
        assert ratio["infinite_count"] == 2


class TestPrivacySweepAcrossEveryCheck:
    """A behavioural guard, so a new quality check cannot leak unnoticed.

    Rather than listing the fields that hold raw values -- a list that goes stale
    the moment somebody adds a check -- this builds a frame in which every value
    is a unique marker and asserts that no marker survives into the artifact.
    """

    MARKER = "ZZSENSITIVEZZ"

    @pytest.fixture
    def marked(self) -> pd.DataFrame:
        index = np.arange(240)
        return pd.DataFrame(
            {
                "identifier": [f"{self.MARKER}-id-{value:04d}" for value in index],
                "constant": [f"{self.MARKER}-constant"] * 240,
                "near_constant": [f"{self.MARKER}-common"] * 239 + [f"{self.MARKER}-rare"],
                "high_card": [f"{self.MARKER}-lvl-{value % 200}" for value in index],
                "nominal": [f"{self.MARKER}-{['a', 'b'][value % 2]}" for value in index],
                "numeric": (index % 31) * 1.5,
                "label": (index % 4 == 0).astype("int64"),
            }
        )

    def test_no_marker_survives_into_the_artifact(self, marked):
        profile = DataProfiler().profile(marked)
        quality = DataQualityInspector().inspect(marked, profile=profile, target="label")
        artifact = AuditBuilder().build(marked, profile=profile, quality=quality)
        text = canonical_json(artifact.to_dict())
        assert self.MARKER not in text

    def test_the_findings_were_actually_produced(self, marked):
        """Otherwise the sweep above passes by finding nothing at all."""
        profile = DataProfiler().profile(marked)
        quality = DataQualityInspector().inspect(marked, profile=profile, target="label")
        codes = {issue.code for issue in quality.issues}
        assert {"constant_column", "near_constant_column", "high_cardinality"} <= codes

    def test_column_names_are_still_present_because_they_are_not_values(self, marked):
        profile = DataProfiler().profile(marked)
        artifact = AuditBuilder().build(marked, profile=profile)
        text = canonical_json(artifact.to_dict())
        assert "near_constant" in text


class TestTheGoldenSemanticArtifact:
    """A committed fixture that catches schema drift with a readable diff.

    Compared against the *semantic* artifact -- the evidence with the timestamp
    and the environment removed -- because those two fields are supposed to
    differ between runs and comparing them would make the fixture fail every
    time anyone upgraded pandas.
    """

    #: The fixture for the running pandas major; see tests/golden.py for why there
    #: are two and what is allowed to differ between them.
    EXPECTED = semantic_fixture_path()

    @pytest.fixture(scope="class")
    @staticmethod
    def rebuilt() -> dict:
        import sys as _sys

        root = str(EXAMPLE.resolve().parents[2])
        if root not in _sys.path:
            _sys.path.insert(0, root)
        from examples.audit_churn.generate_artifacts import build_artifact

        return build_artifact(pd.read_csv(EXAMPLE)).semantic_dict()

    def test_the_fixture_is_committed(self):
        assert self.EXPECTED.exists(), (
            "run examples/audit_churn/generate_artifacts.py"
        )

    def test_the_evidence_still_matches_it_exactly(self, rebuilt):
        expected = json.loads(self.EXPECTED.read_text(encoding="utf-8"))
        assert rebuilt == expected

    def test_the_canonical_bytes_match_too(self, rebuilt):
        assert canonical_json(rebuilt) == self.EXPECTED.read_text(encoding="utf-8")

    def test_it_carries_no_timestamp_or_environment(self):
        expected = json.loads(self.EXPECTED.read_text(encoding="utf-8"))
        assert "provenance" not in expected
        assert "created_at" not in canonical_json(expected)

    def test_it_holds_no_absolute_path(self):
        assert "/home/" not in self.EXPECTED.read_text(encoding="utf-8")

    def test_the_illustrative_artifacts_are_committed_too(self):
        for name in ("audit.json", "lineage.json", "report.html"):
            assert EXAMPLE.with_name(name).exists(), name

    def test_the_committed_report_renders_the_committed_artifact(self):
        from aidatasetkit.evidence import render_report

        payload = json.loads(EXAMPLE.with_name("audit.json").read_text(encoding="utf-8"))
        assert render_report(payload) == EXAMPLE.with_name("report.html").read_text(encoding="utf-8")


class TestTheSummaryNeverUnderstatesTheArtifact:
    """Five of twenty issues, printed with no ellipsis, reads as "there are five".

    This is the same silent narrowing the library refuses to do to data, in the
    one place a user looks first.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def crowded(tmp_path_factory):
        directory = tmp_path_factory.mktemp("crowded")
        path = directory / "lots.csv"
        index = np.arange(300)
        columns = {"label": (index % 2)}
        for n in range(8):
            columns[f"id{n}"] = [f"{n}-{v:05d}" for v in index]
        for n in range(4):
            columns[f"k{n}"] = ["same"] * 300
        pd.DataFrame(columns).to_csv(path, index=False)
        output = directory / "out"
        result = run_cli("audit", str(path), "--target", "label", "--output", str(output))
        return result, json.loads(read_current(output).text("audit.json"))

    def test_the_dataset_really_does_overflow_the_summary(self, crowded):
        _, artifact = crowded
        serious = [f for f in artifact["findings"] if f["severity"] in {"error", "warning"}]
        assert len(serious) > 5

    def test_the_summary_says_how_many_it_left_out(self, crowded):
        result, _ = crowded
        assert "and 15 more" in result.stdout

    def test_the_review_list_says_so_too(self, crowded):
        result, _ = crowded
        assert re.search(r"Needs review:.*and \d+ more", result.stdout)

    def test_the_most_serious_finding_is_never_truncated_away(self, tmp_path):
        """An error decided the verdict; the summary must not hide it behind warnings."""
        index = np.arange(300)
        columns = {"label": (index % 2), "dup": (index % 2)}
        for n in range(8):
            columns[f"k{n}"] = ["same"] * 300
        path = tmp_path / "error_last.csv"
        pd.DataFrame(columns).to_csv(path, index=False)
        result = run_cli(
            "audit", str(path), "--target", "label", "--output", str(tmp_path / "o")
        )
        assert "exactly equal to the target" in result.stdout

    def test_the_verdict_reasons_are_printed(self, crowded):
        result, artifact = crowded
        assert artifact["verdict_reasons"][0] in result.stdout


class TestEveryFailureStillProducesAnArtifact:
    """The promise in the module docstring, tested against distinct causes."""

    @pytest.mark.parametrize(
        "name,columns,target",
        [
            ("untypeable_target", {"a": [1.0, 2.0] * 100, "label": [None] * 200}, "label"),
            (
                "infinite",
                {"r": [float("inf")] + [1.0] * 199, "c": ["x", "y"] * 100, "label": [0, 1] * 100},
                "label",
            ),
        ],
    )
    def test_artifacts_exist_and_the_verdict_is_blocked(
        self, tmp_path, name, columns, target
    ):
        path = tmp_path / f"{name}.csv"
        pd.DataFrame(columns).to_csv(path, index=False)
        output = tmp_path / name
        result = run_cli(
            "audit", str(path), "--target", target, "--model", "logistic_regression",
            "--output", str(output),
        )
        assert "Traceback" not in result.stderr
        published = set(read_current(output).contents)
        assert published == {"audit.json", "lineage.json", "report.html"}, name
        artifact = json.loads(read_current(output).text("audit.json"))
        assert artifact["verdict"] == "blocked"
        assert artifact["verdict_reasons"]

    def test_the_profiling_evidence_survives(self, tmp_path):
        path = tmp_path / "untypeable.csv"
        pd.DataFrame({"a": [1.0, 2.0] * 100, "label": [None] * 200}).to_csv(path, index=False)
        output = tmp_path / "o"
        run_cli("audit", str(path), "--target", "label", "--output", str(output))
        artifact = json.loads(read_current(output).text("audit.json"))
        assert len(artifact["columns"]) == 2


class TestInputsTheAuditMustRefuse:
    def test_duplicate_headers_are_refused_rather_than_renamed(self, tmp_path):
        """pandas turns a,a into a,a.1 and the artifact would describe a file nobody has."""
        path = tmp_path / "dup.csv"
        path.write_text("a,a,label\n1,2,0\n3,4,1\n", encoding="utf-8")
        result = run_cli("audit", str(path), "--target", "label", "--output", str(tmp_path / "o"))
        assert result.returncode == EXIT_CODES["usage"]
        assert "duplicate column headers" in result.stderr
        assert not (tmp_path / "o" / "CURRENT").exists()
        assert not (tmp_path / "o" / "runs").exists()

    def test_output_pointing_at_a_file_is_refused_before_the_work(self, tmp_path):
        existing = tmp_path / "not_a_dir"
        existing.write_text("x", encoding="utf-8")
        result = run_cli("audit", str(EXAMPLE), "--output", str(existing))
        assert result.returncode == EXIT_CODES["usage"]
        assert "must be a directory" in result.stderr

    def test_a_model_without_a_target_is_refused_rather_than_ignored(self, tmp_path):
        result = run_cli(
            "audit", str(EXAMPLE), "--model", "logistic_regression",
            "--output", str(tmp_path / "o"),
        )
        assert result.returncode == EXIT_CODES["usage"]
        assert "--model needs --target" in result.stderr

    def test_none_of_these_show_a_traceback(self, tmp_path):
        path = tmp_path / "dup.csv"
        path.write_text("a,a,label\n1,2,0\n3,4,1\n", encoding="utf-8")
        for args in (
            ("audit", str(path), "--target", "label", "--output", str(tmp_path / "a")),
            ("audit", str(EXAMPLE), "--model", "knn", "--output", str(tmp_path / "b")),
        ):
            assert "Traceback" not in run_cli(*args).stderr


class TestExitCodeTwoMeansThresholdMet:
    def test_warnings_trip_the_warning_threshold(self, tmp_path):
        path = tmp_path / "warn.csv"
        index = np.arange(240)
        pd.DataFrame(
            {"a": (index % 37) * 1.5, "k": ["same"] * 240, "label": (index % 2)}
        ).to_csv(path, index=False)
        strict = run_cli(
            "audit", str(path), "--target", "label", "--fail-on", "warning",
            "--output", str(tmp_path / "a"),
        )
        assert strict.returncode == EXIT_CODES["review"]

    def test_and_the_verdict_itself_says_which_it_was(self, tmp_path):
        path = tmp_path / "warn.csv"
        index = np.arange(240)
        pd.DataFrame(
            {"a": (index % 37) * 1.5, "k": ["same"] * 240, "label": (index % 2)}
        ).to_csv(path, index=False)
        run_cli(
            "audit", str(path), "--target", "label", "--fail-on", "warning",
            "--output", str(tmp_path / "b"),
        )
        artifact = json.loads(read_current(tmp_path / "b").text("audit.json"))
        assert artifact["verdict"] == "ready_with_warnings"

    def test_the_documented_table_matches_the_code(self):
        from pathlib import Path

        docs = (Path(__file__).resolve().parents[2] / "docs" / "getting-started.md").read_text(encoding="utf-8")
        assert "does not mean `review_required` specifically" in docs


class TestUsageErrorsDoNotCollideWithPolicyCodes:
    """argparse exits 2, and 2 is the code this tool documents as "threshold met".

    A CI job seeing 2 could not tell a dataset needing review from a typo in the
    command line, which makes the exit-code contract useless for the one job it
    exists to do.
    """

    @pytest.mark.parametrize(
        "args",
        [
            ("audit", "--bogus-flag"),
            ("audit",),
            (),
            ("--nope",),
            ("nosuchcommand",),
        ],
    )
    def test_a_usage_error_exits_one(self, args):
        assert run_cli(*args).returncode == EXIT_CODES["usage"]

    def test_version_still_exits_zero(self):
        result = run_cli("--version")
        assert result.returncode == 0
        assert "aidatasetkit" in result.stdout

    def test_help_still_exits_zero(self):
        assert run_cli("--help").returncode == 0

    def test_a_blocked_audit_still_exits_three(self, tmp_path):
        result = run_cli(
            "audit", str(EXAMPLE), "--target", "Churn", "--output", str(tmp_path / "o")
        )
        assert result.returncode == EXIT_CODES["blocked"]


class TestArtifactsAreWrittenTogetherOrNotAtAll:
    """The CLI's side of the publication contract; the contract itself is
    exercised step by step in tests/unit/test_publication.py."""

    def test_all_three_appear(self, tmp_path):
        run_cli("audit", str(EXAMPLE), "--target", "Churn", "--output", str(tmp_path / "o"))
        layout = sorted(p.name for p in (tmp_path / "o").iterdir())
        assert layout == [".staging", "CURRENT", "runs"]
        run = read_current(tmp_path / "o")
        assert sorted(run.contents) == ["audit.json", "lineage.json", "report.html"]
        assert sorted(p.name for p in run.directory.iterdir()) == [
            "audit.json", "lineage.json", "manifest.json", "report.html",
        ]

    def test_nothing_is_written_at_the_root_that_could_pass_for_a_run(self, tmp_path):
        run_cli("audit", str(EXAMPLE), "--target", "Churn", "--output", str(tmp_path / "o"))
        for name in ("audit.json", "lineage.json", "report.html"):
            assert not (tmp_path / "o" / name).exists(), name

    def test_legacy_root_files_are_left_alone_and_named(self, tmp_path):
        output = tmp_path / "o"
        output.mkdir()
        (output / "audit.json").write_text("{}", encoding="utf-8")
        result = run_cli("audit", str(EXAMPLE), "--target", "Churn", "--output", str(output))
        assert (output / "audit.json").read_text(encoding="utf-8") == "{}"
        assert "came from an older version" in result.stdout
        assert json.loads(read_current(output).text("audit.json"))["verdict"]

    def test_two_audits_at_once_each_publish_a_complete_run(self, tmp_path):
        output = tmp_path / "shared"
        command = [
            sys.executable, "-c",
            "from aidatasetkit.cli.main import main; raise SystemExit(main())",
            "audit", str(EXAMPLE), "--target", "Churn", "--output", str(output),
            "--fail-on", "never",
        ]
        first = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        second = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert first.wait(timeout=300) == 0, first.stderr.read().decode()
        assert second.wait(timeout=300) == 0, second.stderr.read().decode()
        runs = sorted((output / "runs").iterdir())
        assert len(runs) == 2
        assert read_current(output).run_id in {run.name for run in runs}

    def test_a_publication_failure_leaves_the_previous_run_current(self, tmp_path, monkeypatch, capsys):
        from aidatasetkit.cli.main import main
        from aidatasetkit.evidence import publication

        output = tmp_path / "o"
        assert main(["audit", str(EXAMPLE), "--target", "Churn", "--output", str(output), "--fail-on", "never"]) == 0
        before = read_current(output).run_id

        def refuse(source, target):
            raise OSError("injected: disk full")

        monkeypatch.setattr(publication, "_replace", refuse)
        code = main(["audit", str(EXAMPLE), "--target", "Churn", "--output", str(output), "--fail-on", "never"])
        assert code == EXIT_CODES["usage"]
        assert "PublicationError" in capsys.readouterr().err
        assert read_current(output).run_id == before

    def test_rendering_happens_before_anything_is_written(self):
        """A directory holding audit.json without the report is worse than one
        holding nothing: a reader cannot tell a finished run from a half one."""
        import inspect
        import sys

        import aidatasetkit.cli.main  # noqa: F401  -- registers the module

        # By full name: the package re-exports the `main` function, which shadows
        # the `main` submodule, so the plain import expression returns a function.
        # See the note in aidatasetkit/cli/__init__.py.
        module = sys.modules["aidatasetkit.cli.main"]
        source = inspect.getsource(module._write)
        render_at = source.index("rendered = {")
        publish_at = source.index("publish_run(output")
        assert render_at < publish_at

    def test_rerunning_publishes_a_second_complete_run(self, tmp_path):
        """Nothing is overwritten: the first run stays whole, the second is current."""
        output = tmp_path / "twice"
        run_cli("audit", str(EXAMPLE), "--target", "Churn", "--output", str(output))
        first = read_current(output)
        # A different flag, so run_cli's per-arguments cache really runs it again.
        run_cli("audit", str(EXAMPLE), "--target", "Churn", "--output", str(output), "--fail-on", "never")
        second = read_current(output)
        assert first.run_id != second.run_id
        assert sorted(p.name for p in (output / "runs").iterdir()) == sorted(
            [first.run_id, second.run_id]
        )
        assert (first.directory / "audit.json").read_bytes() == first.contents["audit.json"]


class TestAnUnverifiedTaskSaysSo:
    def test_regression_is_accepted_but_flagged(self, tmp_path):
        output = tmp_path / "reg"
        result = run_cli(
            "audit", str(EXAMPLE), "--target", "Churn", "--task", "regression",
            "--output", str(output),
        )
        assert "Note:" in result.stdout
        artifact = json.loads(read_current(output).text("audit.json"))
        assert any("regression" in w for w in artifact["warnings"])

    def test_the_warning_no_longer_claims_the_models_are_unverified(self, tmp_path):
        """S6 verified them, and saying otherwise would be a false claim.

        The narrowing is the point: what is still unproven for a regression audit
        is the readiness verdict, not the model layer underneath it.
        """
        output = tmp_path / "reg"
        run_cli(
            "audit", str(EXAMPLE), "--target", "Churn", "--task", "regression",
            "--output", str(output),
        )
        artifact = json.loads(read_current(output).text("audit.json"))
        note = next(w for w in artifact["warnings"] if "regression task" in w)
        assert "has not been verified" in note
        assert "readiness verdict" in note
        assert "model-capability evidence for regression has not been verified" not in note

    def test_classification_carries_no_such_warning(self, tmp_path):
        output = tmp_path / "cls"
        run_cli(
            "audit", str(EXAMPLE), "--target", "Churn", "--task", "classification",
            "--output", str(output),
        )
        artifact = json.loads(read_current(output).text("audit.json"))
        assert not any("not been verified" in w for w in artifact["warnings"])


class TestRegressionModelsAreReachableFromTheCommandLine:
    """S6 registered nine regressors; the CLI needed no change to accept one.

    ``--model`` resolves through the same registry as everything else, and
    ``--task`` was already there to narrow a shared alias. These tests pin that
    the existing contract really does carry regression, rather than assuming it.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def ridge(tmp_path_factory):
        output = tmp_path_factory.mktemp("ridge")
        result = run_cli(
            "audit", str(EXAMPLE), "--target", "Churn", "--task", "regression",
            "--model", "ridge_regression", "--output", str(output),
        )
        return result, json.loads(read_current(output).text("audit.json"))

    def test_a_regression_model_context_produces_a_full_artifact(self, ridge):
        result, artifact = ridge
        assert "Traceback" not in result.stderr
        assert artifact["model"]["canonical_name"] == "ridge_regression"
        assert artifact["model"]["task_type"] == "regression"
        assert artifact["model"]["capabilities"]["supports_predict_proba"] is False

    def test_the_recorded_profile_is_the_capability_key_and_nothing_else(self, ridge):
        _, artifact = ridge
        assert (
            artifact["model"]["preprocessing_profile"] == "scaling=1,sparse=1,native_nan=0"
        )

    def test_no_estimator_object_reached_the_artifact(self, ridge):
        _, artifact = ridge
        assert "Ridge(" not in json.dumps(artifact)

    def test_the_plan_it_recorded_scales_because_the_capability_said_so(self, ridge):
        _, artifact = ridge
        steps = [
            step for decision in artifact["decisions"] for step in decision["steps"]
        ]
        assert any("scaling" in step for step in steps)

    def test_and_records_the_requirement_that_put_it_there(self, ridge):
        _, artifact = ridge
        scaled = [
            decision
            for decision in artifact["decisions"]
            if any("scaling" in step for step in decision["steps"])
        ]
        assert scaled
        for decision in scaled:
            assert decision["model_requirement"] is not None
            assert "ridge" not in decision["reason"].lower()

    def test_a_shared_alias_narrows_by_task_from_the_command_line(self, tmp_path):
        output = tmp_path / "alias"
        result = run_cli(
            "audit", str(EXAMPLE), "--target", "Churn", "--task", "regression",
            "--model", "random_forest", "--output", str(output),
        )
        assert result.returncode != EXIT_CODES["usage"], result.stderr
        artifact = json.loads(read_current(output).text("audit.json"))
        assert artifact["model"]["canonical_name"] == "random_forest_regressor"

    def test_the_same_alias_under_the_other_task_gives_the_classifier(self, tmp_path):
        output = tmp_path / "alias_cls"
        run_cli(
            "audit", str(EXAMPLE), "--target", "Churn", "--task", "classification",
            "--model", "random_forest", "--output", str(output),
        )
        artifact = json.loads(read_current(output).text("audit.json"))
        assert artifact["model"]["canonical_name"] == "random_forest_classifier"

    def test_an_ambiguous_alias_is_refused_clearly_rather_than_guessed(self, tmp_path):
        """No --task, and a name both families answer to. It must not pick one."""
        result = run_cli(
            "audit", str(EXAMPLE), "--target", "Churn", "--model", "random_forest",
            "--output", str(tmp_path / "ambiguous"),
        )
        assert result.returncode == EXIT_CODES["usage"]
        assert "Traceback" not in result.stderr
        assert "random_forest_classifier" in result.stderr
        assert "random_forest_regressor" in result.stderr

    def test_and_nothing_was_written_for_that_run(self, tmp_path):
        output = tmp_path / "ambiguous_out"
        run_cli(
            "audit", str(EXAMPLE), "--target", "Churn", "--model", "random_forest",
            "--output", str(output),
        )
        assert not output.exists() or not list(output.iterdir())

    def test_a_classifier_named_under_the_regression_task_is_a_contradiction(
        self, tmp_path
    ):
        result = run_cli(
            "audit", str(EXAMPLE), "--target", "Churn", "--task", "regression",
            "--model", "logistic_regression", "--output", str(tmp_path / "wrong"),
        )
        assert result.returncode == EXIT_CODES["usage"]
        assert "Traceback" not in result.stderr
        assert "classification" in result.stderr

    def test_the_ambiguity_error_names_the_flag_the_command_line_actually_has(
        self, tmp_path
    ):
        """The registry says "Pass task=", which is a Python keyword, not a flag."""
        result = run_cli(
            "audit", str(EXAMPLE), "--target", "Churn", "--model", "knn",
            "--output", str(tmp_path / "flagged"),
        )
        assert result.returncode == EXIT_CODES["usage"]
        assert "--task classification" in result.stderr


class TestAModelMustMatchTheTargetTheDetectorFound:
    """An artifact recording a regressor beside a classification target asserts
    a contradiction, and used to do so with ``warnings == []``.

    The model has to be resolved before the target is typed -- ``--task`` may be
    absent and the registry needs something to narrow a shared alias with -- so
    the compatibility check ``ModelFactory.create(name, target=...)`` performs is
    repeated once the detector has spoken. The run is not aborted: the profiling
    and quality evidence is still true and still worth having. What is withheld
    is the preprocessing plan, which would otherwise be built for a model that
    cannot serve this target.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def mismatched(tmp_path_factory):
        output = tmp_path_factory.mktemp("mismatch")
        result = run_cli(
            "audit", str(EXAMPLE), "--target", "Churn",
            "--model", "ridge_regression", "--output", str(output),
        )
        return result, json.loads(read_current(output).text("audit.json"))

    def test_the_run_completes_rather_than_aborting(self, mismatched):
        result, artifact = mismatched
        assert "Traceback" not in result.stderr
        assert artifact["columns"], "the profiling evidence should survive"

    def test_the_mismatch_is_reported_rather_than_recorded_silently(self, mismatched):
        _, artifact = mismatched
        assert artifact["warnings"]
        assert any(
            "does not match the detected target" in w for w in artifact["warnings"]
        )

    def test_the_warning_names_both_families(self, mismatched):
        _, artifact = mismatched
        note = next(
            w for w in artifact["warnings"] if "does not match the detected target" in w
        )
        assert "regression" in note and "classification" in note

    def test_no_preprocessing_plan_was_built_for_the_wrong_model(self, mismatched):
        _, artifact = mismatched
        assert artifact["decisions"] == []
        assert artifact["plan_fingerprint"] is None

    def test_the_verdict_is_blocked_rather_than_ready(self, mismatched):
        _, artifact = mismatched
        assert artifact["verdict"] == "blocked"

    def test_the_matching_direction_still_works_untouched(self, tmp_path):
        """The guard must not fire on a model that does serve the target."""
        output = tmp_path / "matched"
        run_cli(
            "audit", str(EXAMPLE), "--target", "Churn",
            "--model", "logistic_regression", "--output", str(output),
        )
        artifact = json.loads(read_current(output).text("audit.json"))
        assert not any(
            "does not match the detected target" in w for w in artifact["warnings"]
        )
        assert artifact["decisions"], "a compatible model should still get a plan"
