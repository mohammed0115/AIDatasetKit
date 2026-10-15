"""G2-W1 native datetime profiling and fixed-cadence gap analysis."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core import KitConfig
from aidatasetkit.core.exceptions import ConfigurationError, DomainError, EvidenceError
from aidatasetkit.evidence import (
    AuditBuilder,
    ColumnEvidence,
    EvidenceSource,
    LabelRef,
    canonical_json,
    dataset_fingerprint,
    render_report,
)
from aidatasetkit.evidence.fingerprint import config_fingerprint
from aidatasetkit.profiling import DataProfiler
from aidatasetkit.profiling import profiler as profiler_module
from aidatasetkit.profiling.date_inference import infer_strict_iso
from aidatasetkit.profiling.chunked import profile_delimited_chunks


def _summary(values, *, cadence: str | None = None, dtype=None):
    series = pd.Series(values, dtype=dtype)
    config = KitConfig(expected_datetime_cadence=cadence)
    return DataProfiler(config).profile(pd.DataFrame({"when": series})).column("when").datetime


class TestEligibilityAndCounts:
    def test_naive_datetime_counts_range_and_timezone(self):
        summary = _summary(
            ["2024-01-03", None, "2024-01-01", "2024-01-03"],
            dtype="datetime64[ns]",
        )
        assert summary.total_count == 4
        assert summary.valid_count == 3
        assert summary.missing_count == 1
        assert summary.distinct_valid_count == 2
        assert summary.duplicate_timestamp_count == 1
        assert summary.minimum == "2024-01-01T00:00:00"
        assert summary.maximum == "2024-01-03T00:00:00"
        assert summary.timezone is None
        assert summary.resolution == "ns"

    def test_timezone_aware_datetime_preserves_zone_and_endpoint_offsets(self):
        values = pd.DatetimeIndex(
            ["2024-03-31 00:00", "2024-03-31 02:00"],
            tz="UTC",
        ).tz_convert("Europe/London")
        summary = _summary(values)
        assert summary.timezone == "Europe/London"
        assert summary.minimum == "2024-03-31T00:00:00+00:00"
        assert summary.maximum == "2024-03-31T03:00:00+01:00"

    def test_all_nat_has_no_range_or_frequency(self):
        summary = _summary([pd.NaT, pd.NaT], dtype="datetime64[ns]")
        assert summary.total_count == summary.missing_count == 2
        assert summary.valid_count == summary.distinct_valid_count == 0
        assert summary.minimum is summary.maximum is None
        assert summary.interval_count == 0
        assert summary.is_regular is None
        assert summary.observed_cadence is None

    def test_non_datetime_text_is_not_parsed(self):
        frame = pd.DataFrame({"when": ["2024-01-01", "2024-01-02"]})
        column = DataProfiler().profile(frame).column("when")
        assert column.datetime is None
        assert str(frame["when"].dtype) in {"object", "str"}


class TestObservedSpacing:
    def test_one_valid_timestamp_has_unknown_cadence(self):
        summary = _summary(["2024-01-01", None], dtype="datetime64[ns]")
        assert summary.distinct_valid_count == 1
        assert summary.interval_count == 0
        assert summary.is_regular is None
        assert summary.observed_cadence is None

    def test_two_distinct_timestamps_recognize_the_only_exact_spacing(self):
        summary = _summary(["2024-01-01", "2024-01-03"], dtype="datetime64[ns]")
        assert summary.interval_count == 1
        assert summary.distinct_spacing_count == 1
        assert summary.is_regular is True
        assert summary.observed_cadence == "P2DT0H0M0S"

    def test_unsorted_duplicates_do_not_change_frequency(self):
        summary = _summary(
            ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-02"],
            dtype="datetime64[ns]",
        )
        assert summary.duplicate_timestamp_count == 1
        assert summary.interval_count == 2
        assert summary.is_regular is True
        assert summary.observed_cadence == "P1DT0H0M0S"

    def test_irregular_spacing_does_not_invent_a_cadence(self):
        summary = _summary(
            ["2024-01-01", "2024-01-02", "2024-01-04"],
            dtype="datetime64[ns]",
        )
        assert summary.is_regular is False
        assert summary.distinct_spacing_count == 2
        assert summary.minimum_spacing == "P1DT0H0M0S"
        assert summary.maximum_spacing == "P2DT0H0M0S"
        assert summary.observed_cadence is None
        assert summary.missing_expected_observation_count is None
        assert summary.off_grid_observation_count is None


class TestExpectedCadence:
    def test_missing_expected_observations_are_counted_without_materialization(self):
        summary = _summary(
            ["2024-01-01", "2024-01-04"],
            cadence="1 day",
            dtype="datetime64[ns]",
        )
        assert summary.expected_cadence == "P1DT0H0M0S"
        assert summary.missing_expected_observation_count == 2
        assert summary.gap_count == 1
        assert summary.off_grid_observation_count == 0

    def test_off_grid_observations_are_separate_from_gaps_and_duplicates(self):
        summary = _summary(
            [
                "2024-01-01 00:00",
                "2024-01-01 01:30",
                "2024-01-01 03:00",
                "2024-01-01 03:00",
            ],
            cadence="1 hour",
            dtype="datetime64[ns]",
        )
        assert summary.duplicate_timestamp_count == 1
        assert summary.off_grid_observation_count == 1
        assert summary.missing_expected_observation_count == 2
        assert summary.gap_count == 1

    @pytest.mark.parametrize(
        "value",
        ["", "NaT", "nat", "nan", "0s", "-1s", "not-a-duration", "1M", 1, True],
    )
    def test_invalid_cadences_are_rejected(self, value):
        with pytest.raises(ConfigurationError, match="expected_datetime_cadence"):
            KitConfig(expected_datetime_cadence=value)

    def test_cadence_finer_than_column_resolution_is_rejected(self):
        with pytest.raises(ConfigurationError, match="exactly representable"):
            _summary(
                np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[s]"),
                cadence="500ms",
            )

    def test_default_does_not_classify_gaps(self):
        summary = _summary(
            ["2024-01-01", "2024-01-04"], dtype="datetime64[ns]"
        )
        assert summary.expected_cadence is None
        assert summary.missing_expected_observation_count is None
        assert summary.gap_count is None
        assert summary.off_grid_observation_count is None

    def test_cadence_has_a_stable_canonical_config_representation(self):
        left = KitConfig(expected_datetime_cadence="60 minutes")
        right = KitConfig(expected_datetime_cadence="1 hour")
        assert left.expected_datetime_cadence == right.expected_datetime_cadence
        assert left.to_dict() == right.to_dict()
        assert config_fingerprint(left.to_dict()) == config_fingerprint(right.to_dict())

    def test_empty_and_all_nat_series_have_consistent_counts_with_cadence(self):
        empty = _summary([], cadence="1 hour", dtype="datetime64[ns]")
        all_nat = _summary([pd.NaT, pd.NaT], cadence="1 hour", dtype="datetime64[ns]")
        for summary, total in ((empty, 0), (all_nat, 2)):
            assert summary.total_count == total
            assert summary.valid_count == summary.distinct_valid_count == 0
            assert summary.missing_count == total
            assert summary.minimum is summary.maximum is None
            assert summary.missing_expected_observation_count == 0
            assert summary.gap_count == summary.off_grid_observation_count == 0

    def test_duplicate_only_valid_series_has_no_observed_spacing(self):
        summary = _summary(
            ["2024-01-01"] * 4 + [None], dtype="datetime64[ns]"
        )
        assert summary.valid_count == 4
        assert summary.distinct_valid_count == 1
        assert summary.duplicate_timestamp_count == 3
        assert summary.interval_count == 0
        assert summary.is_regular is None
        assert summary.observed_cadence is None

    def test_off_grid_duplicate_rows_count_once_as_off_grid(self):
        summary = _summary(
            ["2024-01-01 00:00", "2024-01-01 01:30", "2024-01-01 01:30"],
            cadence="1 hour",
            dtype="datetime64[ns]",
        )
        assert summary.duplicate_timestamp_count == 1
        assert summary.off_grid_observation_count == 1
        assert summary.missing_expected_observation_count == 1
        assert summary.gap_count == 1

    def test_grid_covers_trailing_off_grid_observed_window(self):
        summary = _summary(
            ["2024-01-01 00:00", "2024-01-01 01:30"],
            cadence="1 hour",
            dtype="datetime64[ns]",
        )
        assert summary.missing_expected_observation_count == 1
        assert summary.gap_count == 1
        assert summary.off_grid_observation_count == 1

    def test_expected_grid_is_anchored_at_the_minimum_timestamp(self):
        summary = _summary(
            ["2024-01-01 00:10", "2024-01-01 02:10"],
            cadence="1 hour",
            dtype="datetime64[ns]",
        )
        assert summary.missing_expected_observation_count == 1
        assert summary.gap_count == 1
        assert summary.off_grid_observation_count == 0

    def test_gap_count_is_contiguous_runs_not_missing_positions(self):
        summary = _summary(
            [
                "2024-01-01 00:00",
                "2024-01-01 02:00",
                "2024-01-01 06:00",
            ],
            cadence="1 hour",
            dtype="datetime64[ns]",
        )
        assert summary.missing_expected_observation_count == 4
        assert summary.gap_count == 2

    def test_empty_input_has_no_temporal_values_and_does_not_crash(self):
        summary = _summary([], dtype="datetime64[ns]")
        assert summary.total_count == summary.valid_count == summary.missing_count == 0
        assert summary.minimum is summary.maximum is None


class TestTimezoneAndResolution:
    def test_dst_spacing_uses_elapsed_instants_and_preserves_timezone(self):
        values = pd.DatetimeIndex(
            ["2024-03-31 00:00", "2024-03-31 01:00", "2024-03-31 02:00"],
            tz="UTC",
        ).tz_convert("Europe/London")
        summary = _summary(values, cadence="1 hour")
        assert summary.timezone == "Europe/London"
        assert summary.is_regular is True
        assert summary.missing_expected_observation_count == 0
        assert summary.off_grid_observation_count == 0
        assert summary.maximum.endswith("+01:00")

    def test_fall_back_repeated_wall_clock_times_are_distinct_elapsed_instants(self):
        values = pd.date_range(
            "2024-10-27 00:00", periods=3, freq="h", tz="UTC"
        ).tz_convert("Europe/London")
        summary = _summary(values, cadence="1 hour")
        assert summary.minimum == "2024-10-27T01:00:00+01:00"
        assert summary.maximum == "2024-10-27T02:00:00+00:00"
        assert summary.is_regular is True
        assert summary.missing_expected_observation_count == 0

    def test_fixed_offset_timezone_is_preserved(self):
        values = pd.DatetimeIndex(
            ["2024-01-01 00:00", "2024-01-01 01:00"], tz="UTC+05:30"
        )
        summary = _summary(values)
        assert summary.timezone == "UTC+05:30"
        assert summary.minimum.endswith("+05:30")

    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_each_supported_resolution_is_recorded(self, unit):
        values = np.array(["2000-01-01", "2000-01-02"], dtype=f"datetime64[{unit}]")
        assert _summary(values).resolution == unit

    def test_second_resolution_outside_nanosecond_range_does_not_overflow(self):
        values = np.array(["2500-01-01", "2500-01-03"], dtype="datetime64[s]")
        summary = _summary(values)
        assert summary.minimum == "2500-01-01T00:00:00"
        assert summary.maximum == "2500-01-03T00:00:00"
        assert summary.observed_cadence == "P2DT0H0M0S"

    @pytest.mark.parametrize(
        "coordinates",
        [
            [np.iinfo("int64").min + 1, np.iinfo("int64").min + 2],
            [np.iinfo("int64").max - 1, np.iinfo("int64").max],
        ],
        ids=["minimum", "maximum"],
    )
    def test_supported_signed_range_boundaries_do_not_wrap(self, coordinates):
        values = np.array(coordinates, dtype="int64").view("datetime64[s]")
        summary = _summary(values)
        assert summary.minimum < summary.maximum
        assert summary.observed_cadence == "P0DT0H0M1S"

    def test_unrepresentable_full_range_spacing_raises_an_explicit_error(self):
        info = np.iinfo("int64")
        values = np.array([info.min + 1, info.max], dtype="int64").view("datetime64[s]")
        with pytest.raises(DomainError, match="without overflow"):
            _summary(values)

    def test_aware_datetime_boundaries_keep_offsets_and_do_not_overflow(self):
        values = np.array(
            [np.iinfo("int64").max - 1, np.iinfo("int64").max],
            dtype="int64",
        ).view("datetime64[s]")
        aware = pd.Series(values).dt.tz_localize("UTC")
        summary = _summary(aware)
        assert summary.minimum.endswith("+00:00")
        assert summary.maximum.endswith("+00:00")
        assert summary.observed_cadence == "P0DT0H0M1S"

    def test_large_gap_count_is_arithmetic_not_timestamp_materialization(self, monkeypatch):
        def forbidden(*args, **kwargs):
            raise AssertionError("expected timestamps were materialized")

        monkeypatch.setattr(pd, "date_range", forbidden)
        summary = _summary(
            ["1900-01-01", "2090-01-01"],
            cadence="1 hour",
            dtype="datetime64[ns]",
        )
        assert summary.missing_expected_observation_count > 1_000_000
        assert summary.gap_count == 1

    @pytest.mark.parametrize("timezone", ["UTC", "America/Los_Angeles"])
    def test_serialization_is_independent_of_host_timezone(self, timezone):
        env = dict(os.environ)
        env["TZ"] = timezone
        code = (
            "import pandas as pd; "
            "from aidatasetkit.core import KitConfig; "
            "from aidatasetkit.profiling import DataProfiler; "
            "s=pd.Series(pd.date_range('2024-01-01', periods=2, freq='h', tz='UTC')); "
            "print(DataProfiler(KitConfig()).profile(pd.DataFrame({'when':s}))."
            "column('when').datetime.to_dict())"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        assert "+00:00" in result.stdout

    def test_serialization_is_independent_of_host_locale(self):
        code = (
            "import pandas as pd; "
            "from aidatasetkit.profiling import DataProfiler; "
            "s=pd.Series(pd.date_range('2024-01-01', periods=2, freq='h')); "
            "print(DataProfiler().profile(pd.DataFrame({'when':s}))."
            "column('when').datetime.to_dict())"
        )
        outputs = []
        for locale in ("C", "C.UTF-8"):
            env = dict(os.environ)
            env["LC_ALL"] = locale
            env["LANG"] = locale
            result = subprocess.run(
                [sys.executable, "-c", code],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            outputs.append(result.stdout)
        assert outputs[0] == outputs[1]

    def test_unsupported_datetime_extension_does_not_crash_or_summarize(self):
        pyarrow = pytest.importorskip("pyarrow")
        dtype = pd.ArrowDtype(pyarrow.timestamp("ns"))
        frame = pd.DataFrame({"when": pd.Series([None], dtype=dtype)})
        column = DataProfiler().profile(frame).column("when")
        assert column.detected_kind.value == "datetime"
        assert column.datetime is None


class TestEvidenceAndIsolation:
    def test_artifact_serialization_is_deterministic_and_bounded(self):
        frame = pd.DataFrame(
            {
                "when": pd.date_range(
                    "2024-01-01", periods=10_000, freq="h", tz="Asia/Riyadh"
                )
            }
        )
        config = KitConfig(expected_datetime_cadence="1 hour")
        profile = DataProfiler(config).profile(frame)
        first = AuditBuilder().build(frame, profile=profile, kit_config=config)
        second = AuditBuilder().build(frame, profile=profile, kit_config=config)
        left = first.semantic_dict()
        right = second.semantic_dict()
        assert left == right
        assert first.semantic_fingerprint == second.semantic_fingerprint
        payload = canonical_json(left)
        assert len(json.loads(payload)["columns"][0]["datetime"]) == 19
        assert '"timestamps"' not in payload
        assert '"gaps"' not in payload
        assert "2024-01-02T" not in payload

    def test_datetime_evidence_changes_semantic_but_not_dataset_identity(self):
        frame = pd.DataFrame(
            {"when": pd.date_range("2024-01-01", periods=3, freq="2h")}
        )
        plain = KitConfig()
        scheduled = KitConfig(expected_datetime_cadence="1 hour")
        plain_artifact = AuditBuilder().build(
            frame,
            profile=DataProfiler(plain).profile(frame),
            kit_config=plain,
        )
        scheduled_artifact = AuditBuilder().build(
            frame,
            profile=DataProfiler(scheduled).profile(frame),
            kit_config=scheduled,
        )
        assert plain_artifact.dataset.fingerprint == dataset_fingerprint(frame)
        assert scheduled_artifact.dataset.fingerprint == dataset_fingerprint(frame)
        assert plain_artifact.dataset.fingerprint == scheduled_artifact.dataset.fingerprint
        assert plain_artifact.config.fingerprint != scheduled_artifact.config.fingerprint
        assert plain_artifact.semantic_fingerprint != scheduled_artifact.semantic_fingerprint

    def test_caller_dataframe_is_unchanged(self):
        frame = pd.DataFrame(
            {
                "when": pd.DatetimeIndex(
                    ["2024-10-27 00:00", None, "2024-10-27 02:00"], tz="UTC"
                ).tz_convert("Europe/London")
            }
        )
        before = frame.copy(deep=True)
        DataProfiler(KitConfig(expected_datetime_cadence="1 hour")).profile(frame)
        pd.testing.assert_frame_equal(frame, before)

    def test_only_bounded_endpoints_are_published(self):
        secret_middle = "2024-01-02T12:34:56"
        frame = pd.DataFrame(
            {
                "when": pd.to_datetime(
                    ["2024-01-01", secret_middle, "2024-01-04"],
                    format="mixed",
                )
            }
        )
        profile = DataProfiler().profile(frame)
        payload = canonical_json(
            AuditBuilder().build(frame, profile=profile).semantic_dict()
        )
        assert secret_middle not in payload
        assert "2024-01-01T00:00:00" in payload
        assert "2024-01-04T00:00:00" in payload

    def test_strict_iso_text_recognition_is_unchanged_and_separate(self):
        values = [f"2024-01-{day:02d}" for day in range(1, 9)]
        frame = pd.DataFrame({"text": values})
        facts = infer_strict_iso(frame)
        profile = DataProfiler().profile(frame)
        assert len(facts.columns) == 1
        assert profile.column("text").datetime is None

    def test_datetime_report_shows_all_measurements_and_escapes_column_name(self):
        frame = pd.DataFrame(
            {"<time&>": pd.to_datetime(["2024-01-01", "2024-01-03"])}
        )
        profile = DataProfiler(KitConfig(expected_datetime_cadence="1 day")).profile(frame)
        artifact = AuditBuilder().build(
            frame, profile=profile, kit_config=KitConfig(expected_datetime_cadence="1 day")
        )
        page = render_report(artifact.to_dict())
        for expected in (
            "Datetime profiling",
            "resolution",
            "distinct spacings",
            "expected cadence",
            "closed observed window",
            "do not affect findings or the verdict",
            "P1DT0H0M0S",
            "&lt;time&amp;&gt;",
        ):
            assert expected in page
        assert "<time&>" not in page

    def test_datetime_evidence_rejects_inconsistent_counts(self):
        from dataclasses import replace

        frame = pd.DataFrame(
            {"when": pd.to_datetime(["2024-01-01", "2024-01-02"])}
        )
        profile = DataProfiler().profile(frame)
        artifact = AuditBuilder().build(frame, profile=profile)
        column = artifact.columns[0]
        invalid = dict(column.datetime)
        invalid["duplicate_timestamp_count"] = 3
        with pytest.raises(EvidenceError, match="duplicate count"):
            replace(column, datetime=invalid)

    def test_column_evidence_positional_source_argument_is_preserved(self):
        values = [
            LabelRef.of("column"),
            "datetime",
            "datetime64[ns]",
            1,
            0,
            0.0,
            1,
            1.0,
            True,
            False,
            False,
            False,
            0,
            None,
            None,
            None,
            EvidenceSource.QUALITY,
        ]
        column = ColumnEvidence(*values)
        assert column.source is EvidenceSource.QUALITY
        assert column.datetime is None

    def test_datetime_profiling_is_deterministic_across_fresh_runs(self):
        frame = pd.DataFrame(
            {"when": pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")}
        )
        first = DataProfiler().profile(frame)
        second = DataProfiler().profile(frame)
        assert first.column("when").datetime == second.column("when").datetime

    def test_chunked_builder_does_not_call_datetime_summarizer(self, tmp_path, monkeypatch):
        path = tmp_path / "values.csv"
        path.write_text("when\n2024-01-01\n", encoding="utf-8")
        monkeypatch.setattr(
            profiler_module,
            "summarize_datetime",
            lambda *args, **kwargs: pytest.fail("chunked mode invoked datetime profiling"),
        )
        chunked = profile_delimited_chunks(path, chunk_rows=1)
        artifact = AuditBuilder().build_chunked(chunked)
        assert artifact.date_inference is None
        assert artifact.columns == ()
