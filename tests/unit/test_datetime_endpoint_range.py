"""Lossless native datetime endpoints through the public evidence pipeline."""

from dataclasses import replace
import datetime
from html import escape
import json

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core.exceptions import EvidenceError
from aidatasetkit.evidence import (
    AuditBuilder, canonical_json, dataset_fingerprint, publish_run, read_current,
    render_report,
)
from aidatasetkit.profiling import DataProfiler


def _frame(unit="us", timezone=None, coordinates=None):
    values = (
        np.array(coordinates, dtype="int64").view(f"datetime64[{unit}]")
        if coordinates is not None
        else np.array(["10000-01-01", "10000-01-01T00:00:01"],
                      dtype=f"datetime64[{unit}]")
    )
    series = pd.Series(values)
    if timezone is not None:
        series = series.dt.tz_localize(timezone)
    return pd.DataFrame({"<when&>": series})


def _build(frame, reason):
    try:
        return AuditBuilder().build(frame, profile=DataProfiler().profile(frame))
    except (EvidenceError, ValueError, OverflowError) as error:
        pytest.fail(f"{reason}: {error}")


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("timezone", [None, "UTC"])
@pytest.mark.parametrize("boundary", ["lower", "upper", "ordinary"])
def test_native_range_roundtrips_evidence_json_and_html(unit, timezone, boundary):
    info = np.iinfo("int64")
    coordinates = {
        "lower": [info.min + 1, info.min + 2],
        "upper": [info.max - 1, info.max],
        "ordinary": [0, 1],
    }[boundary]
    frame = _frame(unit, timezone, coordinates)
    before = frame.copy(deep=True)
    expected = {
        key: value.isoformat()
        for key, value in (("minimum", frame.iloc[0, 0]), ("maximum", frame.iloc[1, 0]))
    }
    fingerprint = dataset_fingerprint(frame)
    artifact = _build(frame, "native boundary evidence rejected")
    temporal = artifact.columns[0].datetime
    decoded = json.loads(canonical_json(artifact.to_dict()))
    restored = replace(artifact.columns[0], datetime=decoded["columns"][0]["datetime"])
    for record in (temporal, decoded["columns"][0]["datetime"], restored.datetime):
        assert {key: record[key] for key in expected} == expected
        assert record["resolution"] == unit
        assert record["timezone"] == timezone
    page = render_report(decoded)
    assert all(escape(value) in page for value in expected.values())
    assert "&lt;when&amp;&gt;" in page and "<when&>" not in page
    assert artifact.dataset.fingerprint == fingerprint == dataset_fingerprint(frame)
    control = _build(_frame(unit, timezone, [0, 1]), "ordinary control rejected")
    left, right = artifact.to_dict(), control.to_dict()
    for key in ("verdict", "verdict_reasons", "findings", "decisions", "target",
                "model", "plan_fingerprint", "lineage", "config", "date_inference"):
        assert left[key] == right[key], f"boundary endpoint affected {key}"
    pd.testing.assert_frame_equal(frame, before, check_exact=True)


@pytest.mark.parametrize("unit", ["s", "ms", "us"])
@pytest.mark.parametrize("timezone", [None, "UTC"])
def test_year_10000_six_combinations(unit, timezone):
    frame = _frame(unit, timezone)
    artifact = _build(frame, "Python datetime range restriction restored")
    temporal = json.loads(canonical_json(artifact.to_dict()))["columns"][0]["datetime"]
    suffix = "+00:00" if timezone else ""
    assert temporal["minimum"] == "10000-01-01T00:00:00" + suffix
    assert temporal["maximum"] == "10000-01-01T00:00:01" + suffix


def test_expanded_endpoint_is_not_clipped():
    artifact = _build(_frame(), "expanded endpoint clipped")
    assert artifact.columns[0].datetime["minimum"] == "10000-01-01T00:00:00", (
        "expanded endpoint clipped"
    )


def test_expanded_endpoint_keeps_timezone():
    artifact = _build(_frame(timezone="UTC"), "expanded endpoint timezone lost")
    temporal = artifact.columns[0].datetime
    assert temporal["timezone"] == "UTC" and temporal["minimum"].endswith("+00:00"), (
        "expanded endpoint timezone lost"
    )


def test_expanded_endpoint_keeps_resolution():
    artifact = _build(_frame(), "expanded endpoint resolution lost")
    assert artifact.columns[0].datetime["resolution"] == "us", (
        "expanded endpoint resolution lost"
    )


def test_expanded_endpoint_serializes_losslessly():
    artifact = _build(_frame(), "expanded endpoint serialization lost")
    decoded = json.loads(canonical_json(artifact.to_dict()))
    assert decoded["columns"][0]["datetime"]["minimum"] == "10000-01-01T00:00:00", (
        "expanded endpoint serialization lost"
    )


def test_expanded_endpoint_deserializes_losslessly(tmp_path):
    artifact = _build(_frame(), "expanded endpoint deserialization lost")
    files = {
        "audit.json": canonical_json(artifact.to_dict()),
        "lineage.json": canonical_json(artifact.lineage_dict()),
        "report.html": render_report(artifact.to_dict()),
    }
    publish_run(tmp_path, files)
    decoded = read_current(tmp_path).json("audit.json")
    assert decoded["columns"][0]["datetime"]["minimum"] == "10000-01-01T00:00:00", (
        "expanded endpoint deserialization lost"
    )
    restored = replace(artifact.columns[0], datetime=decoded["columns"][0]["datetime"])
    assert restored.datetime == artifact.columns[0].datetime


def test_expanded_endpoint_renders_losslessly():
    artifact = _build(_frame(), "expanded endpoint HTML lost")
    page = render_report(json.loads(canonical_json(artifact.to_dict())))
    assert "<td>10000-01-01T00:00:00</td>" in page, "expanded endpoint HTML lost"


def _reject_endpoint(value, reason):
    column = _build(_frame(), reason).columns[0]
    time_part = value.split("T")[-1] if isinstance(value, str) else ""
    timezone = "UTC" if "+" in time_part or "-" in time_part or time_part.endswith("Z") else None
    invalid = dict(column.datetime, minimum=value, maximum=value, timezone=timezone)
    try:
        replace(column, datetime=invalid)
    except EvidenceError:
        return
    pytest.fail(reason)


@pytest.mark.parametrize("value", [
    "arbitrary", "", "NaT", "10000-01-01", "10000-01-01 00:00:00",
    " 10000-01-01T00:00:00", "10000-01-01T00:00:00\n",
    "10000-01-01T00:00:00Z", "10000-01-01T00:00:00 UTC",
    "10000-01-01T00:00:00+0000", "10000-01-01T00:00:00+00",
    "10000-01-01T00:00:00[UTC]", "10000-01-01T00:00:00junk",
    "+10000-01-01T00:00:00", "010000-01-01T00:00:00",
    "-0001-01-01T00:00:00", "-000-01-01T00:00:00",
    "10000-01-01T00:00:00.1", "10000-01-01T00:00:00.000000",
    "10000-01-01T00:00:00.123456000", "10000-01-01T00:00:00-00:00",
    "10000-01-01T00:00:00+00:00:00",
    "10000-01-01T00:00:00+00:00:01.000000",
    "999999999999-01-01T00:00:00", 42,
])
def test_noncanonical_endpoint_is_rejected(value):
    _reject_endpoint(value, "arbitrary endpoint text accepted")


@pytest.mark.parametrize("value", [
    "10000-02-30T00:00:00", "10001-02-29T00:00:00", "1900-02-29T00:00:00",
    "10000-00-01T00:00:00", "10000-13-01T00:00:00",
    "10000-01-00T00:00:00", "10000-04-31T00:00:00",
    "10000-01-01T24:00:00", "10000-01-01T00:60:00", "10000-01-01T00:00:60",
    "10000-01-01T00:00:00+24:00", "10000-01-01T00:00:00+00:60",
    "10000-01-01T00:00:00+00:00:60",
])
def test_impossible_endpoint_is_rejected(value):
    _reject_endpoint(value, "impossible calendar or time accepted")


@pytest.mark.parametrize("year", [-10000, -100, -1, 0, 1, 999, 10000])
def test_signed_years_and_leap_dates_are_preserved(year):
    text = f"{year:04d}-03-01T00:00:00"
    values = np.array([text, text], dtype="datetime64[s]")
    frame = pd.DataFrame({"when": values})
    artifact = _build(frame, "canonical signed year rejected")
    assert artifact.columns[0].datetime["minimum"] == text


@pytest.mark.parametrize("text", [
    "-400-02-29T00:00:00", "0000-02-29T00:00:00", "2000-02-29T23:59:59",
    "10000-02-29T00:00:00", "10000-01-01T00:00:00.000001",
    "2024-01-01T00:00:00.000000001",
])
def test_canonical_calendar_and_fraction_controls(text):
    unit = "ns" if len(text.split(".")[-1]) == 9 else "us"
    frame = pd.DataFrame({"when": np.array([text, text], dtype=f"datetime64[{unit}]")})
    artifact = _build(frame, "canonical calendar or fraction rejected")
    assert artifact.columns[0].datetime["minimum"] == text


@pytest.mark.parametrize("offset", [
    datetime.timedelta(hours=5, minutes=30), datetime.timedelta(seconds=-30),
    datetime.timedelta(seconds=30), datetime.timedelta(microseconds=1),
])
def test_fixed_offset_syntax_is_preserved(offset):
    frame = _frame(timezone=datetime.timezone(offset))
    expected = frame.iloc[0, 0].isoformat()
    artifact = _build(frame, "canonical offset rejected")
    assert artifact.columns[0].datetime["minimum"] == expected


def test_endpoint_order_compares_instants_not_text_or_truncated_microseconds():
    column = _build(_frame(), "endpoint ordering control rejected").columns[0]
    for minimum, maximum, timezone in (
        ("-001-01-01T00:00:00", "-002-01-01T00:00:00", None),
        ("2024-01-01T00:00:00.000000002", "2024-01-01T00:00:00.000000001", None),
        ("2024-10-27T01:30:00+00:00", "2024-10-27T01:45:00+01:00", "Europe/London"),
    ):
        with pytest.raises(EvidenceError, match="minimum is later"):
            replace(column, datetime=dict(column.datetime, minimum=minimum,
                                          maximum=maximum, timezone=timezone))
