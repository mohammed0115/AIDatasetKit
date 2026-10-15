"""Dedicated semantic witnesses for the CTO G2-W1 adversarial matrix."""

from dataclasses import replace
import json

import pandas as pd
import pytest

from aidatasetkit.core import KitConfig
from aidatasetkit.core.exceptions import ConfigurationError, DomainError, EvidenceError
from aidatasetkit.evidence import (
    AuditBuilder, ColumnEvidence, EvidenceSource, LabelRef, canonical_json,
)
from aidatasetkit.evidence import builder as builder_module
from aidatasetkit.profiling import DataProfiler
from aidatasetkit.profiling.datetime import summarize_datetime


def _summary(values, cadence=None):
    return summarize_datetime(
        pd.Series(values, dtype="datetime64[s]"), expected_cadence=cadence
    )


def test_missing_cadence_is_rejected():
    try:
        KitConfig(expected_datetime_cadence="NaT")
    except ConfigurationError:
        return
    pytest.fail("missing cadence acceptance")


def test_non_fixed_cadence_is_rejected():
    try:
        KitConfig(expected_datetime_cadence="1 month")
    except ConfigurationError:
        return
    pytest.fail("non-fixed cadence acceptance")


def test_unsupported_extension_is_safely_unsummarized():
    arrow = pytest.importorskip("pyarrow")
    series = pd.Series([pd.Timestamp("2024-01-01"), None],
                       dtype=pd.ArrowDtype(arrow.timestamp("ns")))
    before = series.copy(deep=True)
    try:
        summary = summarize_datetime(series, expected_cadence=None)
    except (AttributeError, TypeError, ValueError) as error:
        pytest.fail(f"unsupported datetime eligibility guard failed: {error}")
    assert summary is None, "unsupported datetime eligibility guard failed"
    frame = pd.DataFrame({"when": series})
    assert DataProfiler().profile(frame).column("when").datetime is None
    pd.testing.assert_series_equal(series, before)


def test_profiling_preserves_source_identity_values_index_dtype_order_timezone():
    frame = pd.DataFrame({
        "when": pd.DatetimeIndex(
            ["2024-10-27 02:00", None, "2024-10-27 00:00"], tz="UTC"
        ).tz_convert("Europe/London")
    }, index=pd.Index([7, 2, 9], name="original"))
    source = frame["when"]
    before = frame.copy(deep=True)
    frame_id, source_id, array_id = id(frame), id(source), id(source.array)
    DataProfiler(KitConfig(expected_datetime_cadence="1 hour")).profile(frame)
    try:
        pd.testing.assert_frame_equal(frame, before, check_exact=True)
        pd.testing.assert_series_equal(source, before["when"], check_exact=True)
        assert (id(frame), id(source), id(source.array)) == (
            frame_id, source_id, array_id
        )
        assert str(source.dt.tz) == "Europe/London"
    except AssertionError as error:
        pytest.fail(f"source datetime mutation or reordering: {error}")


def test_duplicates_are_excluded_from_spacing_coordinates():
    summary = _summary(["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-02"])
    assert (
        summary.distinct_valid_count, summary.interval_count,
        summary.distinct_spacing_count, summary.minimum_spacing,
        summary.observed_cadence
    ) == (3, 2, 1, "P1DT0H0M0S", "P1DT0H0M0S"), "duplicate spacing coordinates"


def test_nat_is_excluded_from_duplicate_and_spacing_coordinates():
    try:
        summary = _summary(["1970-01-01", None, "1970-01-02", None, "1970-01-02"])
    except DomainError as error:
        pytest.fail(f"NaT entered temporal coordinates: {error}")
    assert (
        summary.valid_count, summary.missing_count, summary.distinct_valid_count,
        summary.duplicate_timestamp_count, summary.interval_count,
        summary.observed_cadence
    ) == (3, 2, 2, 1, 1, "P1DT0H0M0S"), "NaT entered temporal coordinates"


def test_two_observations_do_not_imply_expected_schedule():
    summary = _summary(["2024-01-01", "2024-01-04"])
    assert summary.observed_cadence == "P3DT0H0M0S"
    assert (
        summary.expected_cadence, summary.missing_expected_observation_count,
        summary.gap_count, summary.off_grid_observation_count
    ) == (None, None, None, None), "observed spacing became expected schedule"


def test_repeated_off_grid_observations_are_counted_distinctly():
    summary = _summary(
        ["2024-01-01 00:00", "2024-01-01 01:30", "2024-01-01 01:30",
         "2024-01-01 03:00"], "P0DT1H0M0S"
    )
    assert summary.off_grid_observation_count == 1, "off-grid duplicates inflated count"
    assert summary.duplicate_timestamp_count == 1


def test_missing_positions_and_contiguous_gaps_are_different_counts():
    summary = _summary(["2024-01-01", "2024-01-05"], "P1DT0H0M0S")
    assert (
        summary.missing_expected_observation_count, summary.gap_count
    ) == (3, 1), "missing positions conflated with gap runs"


def test_canonical_datetime_evidence_keeps_endpoint_and_cadence_context():
    frame = pd.DataFrame({"when": pd.DatetimeIndex(
        ["2024-01-01", "2024-01-03"], tz="Asia/Riyadh"
    )})
    config = KitConfig(expected_datetime_cadence="1 day")
    profile = DataProfiler(config).profile(frame)
    artifact = AuditBuilder().build(frame, profile=profile, kit_config=config)
    payload = json.loads(canonical_json(artifact.to_dict()))
    temporal = payload["columns"][0].get("datetime")
    assert temporal is not None, "canonical datetime context omitted"
    expected = {
        "minimum": "2024-01-01T00:00:00+03:00",
        "maximum": "2024-01-03T00:00:00+03:00",
        "timezone": "Asia/Riyadh",
        "resolution": str(frame["when"].array.unit),
        "expected_cadence": "P1DT0H0M0S",
        "observed_cadence": "P2DT0H0M0S",
    }
    assert {key: temporal.get(key) for key in expected} == expected, (
        "canonical datetime context omitted"
    )


def test_datetime_measurements_are_isolated_from_verdict_inputs(monkeypatch):
    frame = pd.DataFrame({"when": pd.to_datetime(["2024-01-01", "2024-01-04"])})
    plain = DataProfiler().profile(frame)
    scheduled = DataProfiler(KitConfig(expected_datetime_cadence="1 day")).profile(frame)
    calls = []
    original = builder_module.decide_verdict

    def observe(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(builder_module, "decide_verdict", observe)
    artifacts = [AuditBuilder().build(frame, profile=p) for p in (plain, scheduled)]
    assert calls[0] == calls[1], "datetime measurements entered verdict inputs"
    left, right = (artifact.to_dict() for artifact in artifacts)
    for key in ("verdict", "verdict_reasons", "findings", "decisions", "target",
                "model", "plan_fingerprint", "lineage"):
        assert left[key] == right[key], "datetime measurements changed decision evidence"


def test_positional_column_source_remains_compatible():
    try:
        column = ColumnEvidence(
            LabelRef.of("when"), "datetime", "datetime64[ns]",
            1, 0, 0.0, 1, 1.0, True, False, False, False, 0,
            None, None, None, EvidenceSource.QUALITY
        )
    except EvidenceError as error:
        pytest.fail(f"positional source argument displaced: {error}")
    assert column.source is EvidenceSource.QUALITY and column.datetime is None, (
        "positional source argument displaced"
    )


def test_inconsistent_datetime_evidence_requires_evidence_error():
    frame = pd.DataFrame({"when": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    column = AuditBuilder().build(frame, profile=DataProfiler().profile(frame)).columns[0]
    invalid = dict(column.datetime)
    invalid["duplicate_timestamp_count"] = 3
    try:
        replace(column, datetime=invalid)
    except EvidenceError as error:
        assert "duplicate count" in str(error)
        return
    pytest.fail("inconsistent datetime evidence accepted without EvidenceError")


def test_missing_timestamps_are_never_materialized(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("missing timestamps were explicitly materialized")

    monkeypatch.setattr(pd, "date_range", forbidden)
    summary = _summary(["1900-01-01", "2090-01-01"], "P0DT1H0M0S")
    assert summary.missing_expected_observation_count > 1_000_000
