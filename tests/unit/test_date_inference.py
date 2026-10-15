"""G1-W8: strict ISO recognition is evidence, and the table stays as it was loaded."""

from __future__ import annotations

import copy
import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from aidatasetkit import __version__
from aidatasetkit.core.config import KitConfig
from aidatasetkit.evidence import (
    ARTIFACT_SCHEMA_VERSION,
    PUBLICATION_SCHEMA_VERSION,
    AuditBuilder,
    canonical_json,
    render_report,
)
from aidatasetkit.evidence.fingerprint import config_fingerprint, dataset_fingerprint
from aidatasetkit.evidence.policy import decide_verdict
from aidatasetkit.ingestion import load_table
from aidatasetkit.models import ModelFactory
from aidatasetkit.preprocessing import PreprocessingPlanner, PreprocessorBuilder
from aidatasetkit.profiling import DataProfiler, DataQualityInspector, TaskDetector
from aidatasetkit.profiling.chunked import profile_delimited_chunks
from aidatasetkit.profiling.date_inference import infer_strict_iso

ROOT = Path(__file__).resolve().parents[2]


def _frame(column: str, values: list) -> pd.DataFrame:
    return pd.DataFrame({column: values})


def _audit(frame: pd.DataFrame, **kwargs):
    profile = kwargs.pop("profile", None)
    if profile is None:
        profile = DataProfiler().profile(frame)
    return AuditBuilder().build(frame, profile=profile, **kwargs)


def _names(artifact) -> list[str]:
    return [column.column for column in artifact.date_inference.columns]


def _repeat(value, count: int = 8) -> list:
    return [value] * count


class TestPositiveRecognition:
    @pytest.mark.parametrize(
        "value, semantic, policy",
        [
            ("2024-01-31", "date", "naive"),
            ("2024-02-29", "date", "naive"),
            ("2024-01-31T13:45:00", "datetime", "naive"),
            ("2024-01-31T13:45:00.123456789", "datetime", "naive"),
            ("2024-01-31T13:45:00Z", "datetime", "utc_z"),
            ("2024-01-31T13:45:00+03:00", "datetime", "explicit_offset"),
            ("2024-01-31T13:45:00-05:30", "datetime", "explicit_offset"),
        ],
    )
    def test_strict_iso_forms(self, value, semantic, policy):
        facts = infer_strict_iso(_frame("created_at", _repeat(value)))
        column = facts.columns[0]
        assert column.inferred_semantic_type == semantic
        assert column.timezone_policy == policy
        assert column.matched_count == column.non_blank_count == 8
        assert column.invalid_count == 0

    def test_a_mix_of_date_and_datetime_is_datetime(self):
        values = _repeat("2024-01-31", 4) + _repeat("2024-01-31T13:45:00Z", 4)
        column = infer_strict_iso(_frame("created_at", values)).columns[0]
        assert column.inferred_semantic_type == "datetime"
        assert column.timezone_policy == "mixed"

    def test_a_constant_date_column_is_inferred(self):
        column = infer_strict_iso(_frame("created_at", _repeat("2024-03-01", 12))).columns[0]
        assert column.inferred_semantic_type == "date"
        assert column.non_blank_count == 12

    def test_nulls_and_empty_strings_are_not_failures(self):
        values = _repeat("2024-01-31") + [None, "", None]
        column = infer_strict_iso(_frame("created_at", values)).columns
        assert column, "null or empty values counted as failures"
        assert column[0].non_blank_count == 8
        assert column[0].blank_count == 3
        assert column[0].matched_count == 8

    def test_more_than_eight_values(self):
        column = infer_strict_iso(_frame("created_at", _repeat("2024-01-31", 9))).columns[0]
        assert column.non_blank_count == 9

    def test_row_order_does_not_change_the_decision(self):
        values = _repeat("2024-01-31", 5) + _repeat("2024-06-15T08:00:00+00:00", 5)
        forward = infer_strict_iso(_frame("created_at", values))
        backward = infer_strict_iso(_frame("created_at", list(reversed(values))))
        assert forward == backward


class TestLoadedText:
    def test_csv_text(self, tmp_path: Path):
        path = tmp_path / "dates.csv"
        path.write_text("created_at\n" + "\n".join(_repeat("2024-01-31")) + "\n", encoding="utf-8")
        assert _names(_audit(load_table(path).frame)) == ["created_at"]

    def test_json_text(self, tmp_path: Path):
        path = tmp_path / "dates.json"
        path.write_text(
            json.dumps([{"created_at": "2024-01-31"} for _ in range(8)]),
            encoding="utf-8",
        )
        assert _names(_audit(load_table(path).frame)) == ["created_at"]

    def test_jsonl_text(self, tmp_path: Path):
        path = tmp_path / "dates.jsonl"
        path.write_text(
            "".join(json.dumps({"created_at": "2024-01-31"}) + "\n" for _ in range(8)),
            encoding="utf-8",
        )
        assert _names(_audit(load_table(path).frame)) == ["created_at"]

    def test_xlsx_text_cells(self, tmp_path: Path):
        openpyxl = pytest.importorskip("openpyxl")
        path = tmp_path / "dates.xlsx"
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(["created_at"])
        for value in _repeat("2024-01-31"):
            sheet.append([value])
        workbook.save(path)
        loaded = load_table(path)
        assert loaded.frame["created_at"].map(type).eq(str).all()
        assert _names(_audit(loaded.frame)) == ["created_at"]

    def test_sqlite_text(self, tmp_path: Path):
        path = tmp_path / "dates.sqlite"
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE events (created_at TEXT)")
        connection.executemany(
            "INSERT INTO events VALUES (?)",
            [(value,) for value in _repeat("2024-01-31")],
        )
        connection.commit()
        connection.close()
        assert _names(_audit(load_table(path).frame)) == ["created_at"]

    def test_a_dataframe_string_column(self):
        frame = load_table(_frame("created_at", _repeat("2024-01-31"))).frame
        assert _names(_audit(frame)) == ["created_at"]

    def test_records_containing_strings(self):
        records = [{"created_at": "2024-01-31"} for _ in range(8)]
        assert _names(_audit(load_table(records).frame)) == ["created_at"]


class TestNegativeRecognition:
    @pytest.mark.parametrize(
        "value",
        [
            "01/02/2024",
            "2024/01/31",
            "01/02/03",
            "2024-01",
            "2024",
            "20240131",
            "ACC-20240131",
            "due on 2024-01-31",
            " 2024-01-31",
            "2024-01-31 ",
            "   ",
            "٢٠٢٤-٠١-٣١",
            "January 31, 2024",
            "2024-01-31T13:45",
            "2024-01-31 13:45:00",
            "2024-01-31T13:45:00+03:60",
            "2024-01-31T13:45:00+24:00",
            "0000-01-01",
            "2024-13-01",
            "2024-01-31T25:00:00",
            "2024-01-31T13:45:60",
            "2023-02-29",
        ],
    )
    def test_unsupported_or_invalid_text_is_not_inferred(self, value):
        assert infer_strict_iso(_frame("created_at", _repeat(value))).columns == ()

    def test_slash_dates_are_refused(self):
        facts = infer_strict_iso(_frame("created_at", _repeat("01/02/2024")))
        assert facts.columns == (), "ambiguous slash date accepted"

    def test_identifier_text_is_not_a_date(self):
        facts = infer_strict_iso(_frame("code", _repeat("ACC-20240131")))
        assert facts.columns == (), "identifier-like text inferred as a date"

    def test_seven_values_are_not_enough(self):
        facts = infer_strict_iso(_frame("created_at", _repeat("2024-01-31", 7)))
        assert facts.columns == (), "minimum observation threshold bypassed"

    def test_one_invalid_value_blocks_the_column(self):
        values = _repeat("2024-01-31") + ["2024-02-30"]
        facts = infer_strict_iso(_frame("created_at", values))
        assert facts.columns == (), "invalid value ignored"

    def test_a_late_invalid_value_is_seen(self):
        values = _repeat("2024-01-31") + ["not-a-date"]
        facts = infer_strict_iso(_frame("created_at", values))
        assert facts.columns == (), "sampling presented as full population"

    def test_whitespace_only_blocks_positive_inference(self):
        values = _repeat("2024-01-31") + [" \t"]
        assert infer_strict_iso(_frame("created_at", values)).columns == ()

    def test_mixed_strings_and_integers_are_not_inferred(self):
        values = _repeat("2024-01-31") + [20240131]
        assert infer_strict_iso(_frame("created_at", values)).columns == ()

    def test_a_unix_timestamp_numeric_column_is_not_a_candidate(self):
        facts = infer_strict_iso(_frame("ts", [1_700_000_000 + index for index in range(8)]))
        assert facts.columns == ()

    def test_an_excel_serial_numeric_column_is_not_a_candidate(self):
        facts = infer_strict_iso(_frame("serial", [45000 + index for index in range(8)]))
        assert facts.columns == ()

    def test_a_malformed_date_does_not_crash(self):
        frame = _frame("created_at", _repeat("2024-02-29") + ["2024-02-30"])
        try:
            artifact = _audit(frame)
        except Exception as error:
            raise AssertionError("malformed date crashes the audit") from error
        assert _names(artifact) == []


class TestEvidenceAndSchema:
    def test_positive_inference_is_in_the_artifact(self):
        artifact = _audit(_frame("created_at", _repeat("2024-01-31T00:00:00Z")))
        assert "created_at" in _names(artifact), "inference omitted from typed evidence"
        column = artifact.date_inference.columns[0]
        assert column.inferred_semantic_type == "datetime"
        assert column.format_family == "strict_iso8601"
        assert column.timezone_policy == "utc_z"
        assert column.non_blank_count == 8

    def test_a_full_audit_runs_inference(self):
        artifact = _audit(_frame("created_at", _repeat("2024-01-31")))
        assert "created_at" in _names(artifact), "default full audit fails to run inference"
        assert artifact.date_inference.full_population_scanned is True

    def test_the_method_name_is_fixed(self):
        artifact = _audit(_frame("created_at", _repeat("2024-01-31")))
        record = artifact.date_inference
        assert record.method == "strict_iso_full_population_v1", (
            "fixed policy/method omitted or mislabeled"
        )
        assert record.population == "full"
        assert record.minimum_non_blank == 8
        assert record.required_match_ratio == 1.0
        assert record.conversion_performed is False
        assert record.affects_detected_kind is False
        assert record.affects_verdict is False

    def test_no_positive_column_still_records_the_policy(self):
        artifact = _audit(_frame("name", _repeat("alpha")))
        assert artifact.date_inference is not None
        assert artifact.date_inference.columns == ()
        assert artifact.date_inference.method == "strict_iso_full_population_v1"

    def test_raw_values_are_not_stored(self):
        artifact = _audit(_frame("created_at", _repeat("2024-01-31")))
        assert "2024-01-31" not in canonical_json(artifact.to_dict())

    def test_counts_ignore_blanks_and_keep_invalid_at_zero(self):
        values = _repeat("2024-05-06", 10) + [None, None, ""]
        column = _audit(_frame("created_at", values)).date_inference.columns[0]
        assert column.non_blank_count == 10
        assert column.blank_count == 3
        assert column.matched_count == 10
        assert column.invalid_count == 0

    def test_the_artifact_schema_is_1_4(self):
        assert ARTIFACT_SCHEMA_VERSION == "1.5", "G2-W1 schema migration is omitted"
        assert PUBLICATION_SCHEMA_VERSION == "1.0"
        assert __version__ == "0.1.0a1"
        artifact = _audit(_frame("created_at", _repeat("2024-01-31")))
        assert artifact.schema_version == "1.5"
        payload = artifact.to_dict()
        assert payload["schema_version"] == "1.5"
        assert payload["date_inference"]["conversion_performed"] is False

    def test_rollback_restores_the_1_3_shape_without_moving_the_dataset(self):
        artifact = _audit(_frame("created_at", _repeat("2024-01-31")))
        semantic = artifact.semantic_dict()
        rolled = copy.deepcopy(semantic)
        rolled["schema_version"] = "1.3"
        del rolled["date_inference"]
        assert "date_inference" not in rolled
        assert rolled["schema_version"] == "1.3"
        assert rolled["dataset"]["fingerprint"] == semantic["dataset"]["fingerprint"]
        assert rolled["verdict"] == semantic["verdict"]
        assert rolled["columns"] == semantic["columns"]
        assert config_fingerprint(rolled) != artifact.semantic_fingerprint

    def test_the_report_does_not_call_the_column_converted(self):
        artifact = _audit(_frame("created_at", _repeat("2024-01-31T13:45:00Z")))
        page = render_report(artifact.to_dict())
        assert "Date inference" in page
        assert "inferred semantic type" in page
        assert "strict_iso_full_population_v1" in page
        assert "Full population scanned: True" in page
        assert "8/8" in page
        assert "utc_z" in page
        assert "Conversion performed: no" in page
        assert "detected kind unchanged" in page
        assert "not converted" in page
        assert "not datetime-profiled" in page
        assert "categorical" in page

    def test_a_chunked_artifact_does_not_claim_date_inference(self, tmp_path: Path):
        path = tmp_path / "people.csv"
        path.write_text(
            "created_at,age\n" + "".join(f"2024-01-31,{index}\n" for index in range(8)),
            encoding="utf-8",
        )
        artifact = AuditBuilder().build_chunked(profile_delimited_chunks(path, chunk_rows=2))
        assert artifact.date_inference is None, "chunked mode claims full date inference"
        assert artifact.semantic_dict()["date_inference"] is None
        assert artifact.verdict.value == "blocked"


class TestIsolation:
    def test_values_and_dtypes_stay(self):
        frame = _frame("created_at", _repeat("2024-01-31"))
        before = frame.copy()
        before_dtype = str(frame["created_at"].dtype)
        artifact = _audit(frame)
        pd.testing.assert_frame_equal(frame, before)
        assert str(frame["created_at"].dtype) == before_dtype
        assert artifact.dataset.fingerprint == dataset_fingerprint(before), (
            "dataset fingerprint changes without conversion"
        )

    def test_the_dataset_fingerprint_is_the_loaded_table(self):
        frame = _frame("created_at", _repeat("2024-01-31T13:45:00Z"))
        before = dataset_fingerprint(frame)
        artifact = _audit(frame)
        assert artifact.dataset.fingerprint == before, (
            "dataset fingerprint changes without conversion"
        )
        assert list(frame["created_at"]) == _repeat("2024-01-31T13:45:00Z")

    def test_inference_does_not_change_kind_or_verdict(self):
        frame = _frame("created_at", _repeat("2024-01-31", 20))
        profile = DataProfiler().profile(frame)
        quality = DataQualityInspector().inspect(frame, profile=profile)
        artifact = AuditBuilder().build(frame, profile=profile, quality=quality)
        assert artifact.columns[0].detected_kind == "categorical", (
            "inferred evidence changes detected_kind or verdict"
        )
        assert artifact.columns[0].detected_kind == profile.column_profiles[0].detected_kind.value
        expected, reasons = decide_verdict(
            tuple(artifact.findings), tuple(artifact.decisions)
        )
        assert artifact.verdict == expected, "inferred evidence changes detected_kind or verdict"
        assert artifact.verdict_reasons == reasons

    def test_quality_task_plan_lineage_and_config_stay(self):
        frame = pd.DataFrame(
            {
                "created_at": _repeat("2024-01-31", 40),
                "city": ["a", "b"] * 20,
                "amount": [1.0, 2.0, 3.0, 4.0] * 10,
                "label": [0, 1] * 20,
            }
        )
        config = KitConfig()
        profile = DataProfiler(config).profile(frame)
        quality = DataQualityInspector(config).inspect(frame, profile=profile, target="label")
        target = TaskDetector(config).detect(frame["label"], target_name="label")
        registration = ModelFactory.registration("logistic_regression")
        plan = PreprocessingPlanner().plan(
            frame,
            profile,
            registration.capabilities.preprocessing_profile(),
            quality=quality,
            target="label",
        )
        features = frame.drop(columns=["label"])
        preprocessor = PreprocessorBuilder().build(plan, features)
        preprocessor.fit(features)
        lineage = preprocessor.lineage()
        lineage_before = {key: tuple(steps) for key, steps in lineage.items()}
        quality_before = quality.to_dict()
        plan_before = plan.to_dict()
        settings = {"task_hint": None}
        artifact = AuditBuilder().build(
            frame,
            profile=profile,
            quality=quality,
            target=target,
            plan=plan,
            lineage=lineage,
            model=registration,
            kit_config=config,
            settings=settings,
        )
        assert quality.to_dict() == quality_before
        assert plan.to_dict() == plan_before
        assert artifact.plan_fingerprint == plan.fingerprint
        assert artifact.target.name.name == "label"
        assert {key: tuple(steps) for key, steps in lineage.items()} == lineage_before
        actions = {entry.source.name: entry.action for entry in artifact.lineage}
        assert set(lineage_before) <= set(actions)
        assert actions["created_at"] in {"exclude", "hold", "include"}
        assert "strict_iso" not in json.dumps(artifact.config.settings)
        assert "date_inference" not in artifact.config.settings
        assert artifact.config.fingerprint == config_fingerprint(artifact.config.settings)
        semantic = artifact.semantic_dict()
        rolled = copy.deepcopy(semantic)
        rolled["schema_version"] = "1.3"
        del rolled["date_inference"]
        assert semantic["dataset"] == rolled["dataset"]
        assert semantic["verdict"] == rolled["verdict"]
        assert semantic["columns"] == rolled["columns"]
        assert semantic["findings"] == rolled["findings"]
        assert semantic["decisions"] == rolled["decisions"]
        assert semantic["lineage"] == rolled["lineage"]
        assert semantic["config"] == rolled["config"]
        assert set(semantic) - set(rolled) == {"date_inference"}

    def test_numeric_text_quality_is_unchanged(self):
        frame = _frame("code", _repeat("42"))
        profile = DataProfiler().profile(frame)
        quality = DataQualityInspector().inspect(frame, profile=profile)
        artifact = _audit(frame, profile=profile, quality=quality)
        assert any(issue.code == "possible_numeric_stored_as_text" for issue in quality.issues)
        assert _names(artifact) == []
        assert artifact.columns[0].detected_kind == "categorical"

    def test_datetime64_stays_on_the_datetime_path(self):
        frame = pd.DataFrame({"created_at": pd.date_range("2024-01-01", periods=8)})
        profile = DataProfiler().profile(frame)
        artifact = _audit(frame, profile=profile)
        assert profile.column_profiles[0].detected_kind.value == "datetime"
        assert artifact.columns[0].detected_kind == "datetime"
        assert _names(artifact) == []

    def test_timezone_aware_datetime64_stays_on_the_datetime_path(self):
        frame = pd.DataFrame(
            {"created_at": pd.date_range("2024-01-01", periods=8, tz="UTC")}
        )
        profile = DataProfiler().profile(frame)
        artifact = _audit(frame, profile=profile)
        assert artifact.columns[0].detected_kind == "datetime"
        assert _names(artifact) == []

    def test_python_date_objects_are_not_textual_candidates(self):
        frame = _frame("created_at", [date(2024, 1, 31) for _ in range(8)])
        profile = DataProfiler().profile(frame)
        artifact = _audit(frame, profile=profile)
        assert artifact.columns[0].detected_kind == "categorical"
        assert _names(artifact) == []
        assert not isinstance(frame["created_at"].iloc[0], str)


class TestDeterminism:
    def test_the_host_locale_is_not_consulted(self, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("host locale changes the decision")

        monkeypatch.setattr("locale.getlocale", boom)
        monkeypatch.setattr("locale.setlocale", boom)
        facts = infer_strict_iso(_frame("created_at", _repeat("2024-01-31")))
        assert facts.columns[0].inferred_semantic_type == "date"

    def test_the_host_timezone_is_not_consulted(self, monkeypatch):
        import aidatasetkit.profiling.date_inference as recognition

        real = recognition._datetime

        class _ZoneGuard:
            def __call__(self, *args, **kwargs):
                return real(*args, **kwargs)

            def now(self, *args, **kwargs):
                raise AssertionError("host timezone changes the decision")

            def __getattr__(self, name):
                return getattr(real, name)

        monkeypatch.setattr(recognition, "_datetime", _ZoneGuard())
        monkeypatch.setenv("TZ", "Pacific/Kiritimati")
        facts = infer_strict_iso(_frame("created_at", _repeat("2024-01-31T13:45:00Z")))
        assert facts.columns[0].timezone_policy == "utc_z"
        assert facts.columns[0].inferred_semantic_type == "datetime"

    def test_paths_and_clocks_are_not_semantic_facts(self):
        frame = _frame("created_at", _repeat("2024-01-31"))
        one = AuditBuilder(dataset_name="a.csv").build(
            frame,
            profile=DataProfiler().profile(frame),
            created_at="2020-01-01T00:00:00Z",
        )
        two = AuditBuilder(dataset_name="b.csv").build(
            frame,
            profile=DataProfiler().profile(frame),
            created_at="2030-01-01T00:00:00Z",
        )
        assert one.semantic_fingerprint == two.semantic_fingerprint
        assert "a.csv" not in canonical_json(one.semantic_dict())


class TestRegressions:
    def test_no_new_date_dependency(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        for name in ("dateparser", "pendulum", "python-dateutil"):
            assert name not in text

    def test_an_excel_selector_is_still_recorded(self, tmp_path: Path):
        openpyxl = pytest.importorskip("openpyxl")
        path = tmp_path / "book.xlsx"
        workbook = openpyxl.Workbook()
        workbook.active.title = "Data"
        workbook.active.append(["n"])
        workbook.active.append([1])
        workbook.save(path)
        loaded = load_table(path, sheet="Data")
        artifact = AuditBuilder().build(
            loaded.frame,
            profile=DataProfiler().profile(loaded.frame),
            ingestion=loaded.metadata,
            source_selector=loaded.selector,
        )
        assert artifact.source_selector.to_dict() == {"kind": "sheet", "name": "Data"}
        assert artifact.schema_version == "1.5"

    def test_a_sqlite_selector_is_still_recorded(self, tmp_path: Path):
        path = tmp_path / "one.sqlite"
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE people (n INTEGER)")
        connection.execute("INSERT INTO people VALUES (1)")
        connection.commit()
        connection.close()
        loaded = load_table(path)
        artifact = AuditBuilder().build(
            loaded.frame,
            profile=DataProfiler().profile(loaded.frame),
            ingestion=loaded.metadata,
            source_selector=loaded.selector,
        )
        assert artifact.source_selector.to_dict() == {"kind": "table", "name": "people"}
        assert artifact.schema_version == "1.5"
