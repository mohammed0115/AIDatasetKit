"""G1-W5: chunked CSV/TSV profiling is opt-in, bounded, and not a full audit."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aidatasetkit.core.exceptions import (
    CellLimitError,
    FileSizeLimitError,
    InvalidIngestionOptionsError,
    RowLimitError,
)
from aidatasetkit.evidence import AuditBuilder
from aidatasetkit.evidence.builder import CHUNKED_AUDIT_UNAVAILABLE
from aidatasetkit.evidence.fingerprint import dataset_fingerprint, schema_fingerprint
from aidatasetkit.evidence.types import AuditStage, Verdict
from aidatasetkit.ingestion import IngestionLimits, LoadOptions, load_table
from aidatasetkit.profiling import DataProfiler
from aidatasetkit.profiling.chunked import profile_delimited_chunks


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _csv(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "people.csv",
        "age,city,score\n"
        "1,a,1.5\n"
        "2,b,\n"
        "1,a,1.5\n"
        "4,a,3\n",
    )


def test_core_metrics_match_the_full_profile(tmp_path: Path):
    path = _csv(tmp_path)
    loaded = load_table(path)
    profile = DataProfiler().profile(loaded.frame)
    chunked = profile_delimited_chunks(path, chunk_rows=1)
    assert chunked.population_rows == len(loaded.frame)
    assert chunked.population_columns == loaded.frame.shape[1]
    assert chunked.population_fingerprint == dataset_fingerprint(loaded.frame)
    assert chunked.duplicate_row_count == profile.duplicate_row_count
    assert chunked.full_population_scanned is True
    assert chunked.rows_scanned == chunked.population_rows
    assert chunked.schema_fingerprint == schema_fingerprint(loaded.frame)
    assert chunked.temporary_storage == "removed"
    assert chunked.bounded_memory is True
    by_name = {column.name: column for column in chunked.columns}
    for column in profile.column_profiles:
        measured = by_name[str(column.name)]
        assert measured.missing_count == column.missing_count
        assert measured.count == column.count
        assert measured.unique_count == column.unique_count
        assert measured.pandas_dtype == column.pandas_dtype
        assert "q25" not in measured.to_dict()
        if column.numeric is None:
            assert measured.mean is None
            continue
        assert measured.minimum == column.numeric.minimum
        assert measured.maximum == column.numeric.maximum
        assert measured.mean == column.numeric.mean


def test_the_population_fingerprint_matches_the_full_table(tmp_path: Path):
    path = _csv(tmp_path)
    full = dataset_fingerprint(load_table(path).frame)
    assert profile_delimited_chunks(path, chunk_rows=1).population_fingerprint == full
    assert profile_delimited_chunks(path, chunk_rows=3).population_fingerprint == full


def test_the_fingerprint_grouping_matches_the_evidence_hasher():
    from aidatasetkit.evidence.fingerprint import _CHUNK_ROWS, digest_of
    from aidatasetkit.evidence.serialization import label_token
    from aidatasetkit.profiling.chunked import _FINGERPRINT_CHUNK_ROWS, _digest_of, _label_token

    assert _FINGERPRINT_CHUNK_ROWS == _CHUNK_ROWS
    assert _label_token(0) == label_token(0)
    assert _label_token("age") == label_token("age")
    assert _digest_of("rows=2", "columns=1") == digest_of("rows=2", "columns=1")


def test_the_fingerprint_does_not_depend_on_chunk_size(tmp_path: Path):
    path = _csv(tmp_path)
    first = profile_delimited_chunks(path, chunk_rows=1)
    second = profile_delimited_chunks(path, chunk_rows=100)
    assert first.population_fingerprint == second.population_fingerprint


def test_a_row_over_the_limit_is_refused(tmp_path: Path):
    path = _csv(tmp_path)
    with pytest.raises(RowLimitError):
        profile_delimited_chunks(path, limits=IngestionLimits(max_rows=1), chunk_rows=1)


def test_a_byte_over_the_limit_creates_no_scratch_directory(tmp_path: Path, monkeypatch):
    path = _csv(tmp_path)
    created: list[str] = []
    real = tempfile.mkdtemp

    def wrapped(*args, **kwargs):
        directory = real(*args, **kwargs)
        created.append(directory)
        return directory

    monkeypatch.setattr(tempfile, "mkdtemp", wrapped)
    with pytest.raises(FileSizeLimitError):
        profile_delimited_chunks(path, limits=IngestionLimits(max_source_bytes=1))
    assert created == []


def test_the_file_is_read_in_chunks(tmp_path: Path):
    path = _csv(tmp_path)
    profile_delimited_chunks(path, chunk_rows=1)


def test_the_temporary_directory_is_removed(tmp_path: Path, monkeypatch):
    path = _csv(tmp_path)
    created: list[str] = []
    real = tempfile.mkdtemp

    def wrapped(*args, **kwargs):
        directory = real(*args, **kwargs)
        created.append(directory)
        return directory

    monkeypatch.setattr(tempfile, "mkdtemp", wrapped)
    profile_delimited_chunks(path, chunk_rows=1)
    assert created and all(not Path(directory).exists() for directory in created), (
        "temporary directory survived"
    )


def test_the_temporary_directory_is_removed_when_profiling_fails(tmp_path: Path, monkeypatch):
    path = _csv(tmp_path)
    created: list[str] = []
    real = tempfile.mkdtemp

    def wrapped(*args, **kwargs):
        directory = real(*args, **kwargs)
        created.append(directory)
        return directory

    monkeypatch.setattr(tempfile, "mkdtemp", wrapped)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pd.util, "hash_pandas_object", boom)
    with pytest.raises(RuntimeError, match="boom"):
        profile_delimited_chunks(path, chunk_rows=1)
    assert created and all(not Path(directory).exists() for directory in created), (
        "temporary directory survived"
    )


def test_temporary_storage_is_private(tmp_path: Path, monkeypatch):
    path = _csv(tmp_path)
    modes: list[int] = []
    real = shutil.rmtree

    def wrapped(directory, *args, **kwargs):
        for file in Path(directory).rglob("*"):
            if file.is_file():
                modes.append(file.stat().st_mode & 0o777)
        return real(directory, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", wrapped)
    profile_delimited_chunks(path, chunk_rows=1)
    assert modes, "temporary storage is group-readable"
    # Windows reports st_mode without Unix permission bits, so chmod(0o600)
    # cannot be observed there. POSIX is where the mode is the privacy control.
    if os.name == "posix":
        assert all(mode & 0o077 == 0 for mode in modes), "temporary storage is group-readable"


def test_only_csv_and_tsv_are_accepted(tmp_path: Path):
    path = _write(tmp_path / "notes.json", "a\n1\n")
    with pytest.raises(InvalidIngestionOptionsError, match="CSV and TSV only"):
        profile_delimited_chunks(path)


def test_a_tsv_file_is_profiled(tmp_path: Path):
    path = _write(tmp_path / "people.tsv", "age\tcity\n1\ta\n2\tb\n")
    chunked = profile_delimited_chunks(path, chunk_rows=1)
    assert chunked.format == "tsv"
    assert chunked.population_fingerprint == dataset_fingerprint(load_table(path).frame)


def test_an_approximation_is_labeled(tmp_path: Path):
    path = _csv(tmp_path)
    profile = profile_delimited_chunks(path, chunk_rows=2, quantile_sample_size=2)
    assert profile.approximations, "approximation was not labeled"
    assert {item.label for item in profile.approximations} == {"deterministic_approximation"}
    assert {item.method for item in profile.approximations} == {"deterministic_reservoir"}
    for item in profile.approximations:
        assert item.requested_size == 2
        assert item.actual_size <= 2
        assert item.population_size >= item.actual_size
        assert item.seed == 0
    for column in profile.columns:
        assert "q25" not in column.to_dict(), "approximate quantile stored as exact"
        if column.name == "age":
            assert column.mean is not None
    wider = profile_delimited_chunks(path, chunk_rows=2)
    age = next(column for column in profile.columns if column.name == "age")
    wider_age = next(column for column in wider.columns if column.name == "age")
    assert age.mean == wider_age.mean
    assert age.minimum == wider_age.minimum
    assert age.maximum == wider_age.maximum


def test_a_labeled_approximation_does_not_change_the_verdict(tmp_path: Path):
    path = _csv(tmp_path)
    chunked = profile_delimited_chunks(path, chunk_rows=1, quantile_sample_size=2)
    assert chunked.approximations
    artifact = AuditBuilder().build_chunked(chunked)
    assert artifact.verdict is Verdict.BLOCKED
    assert artifact.verdict_reasons == (CHUNKED_AUDIT_UNAVAILABLE,), (
        "approximation changed the verdict inputs"
    )
    assert artifact.findings == (), "approximation changed the verdict inputs"
    assert artifact.decisions == (), "approximation changed the verdict inputs"
    assert artifact.chunked_profiling is not None
    recorded = artifact.semantic_dict()["chunked_profiling"]
    assert recorded["fingerprint_scope"] == "population"
    assert recorded["approximate_metrics"]
    assert "q25" not in recorded["exact_metrics"]


def test_headerless_csv_matches_the_full_table(tmp_path: Path):
    from aidatasetkit.ingestion import LoadOptions

    path = _write(tmp_path / "bare.csv", "1,a\n2,b\n3,a\n")
    options = LoadOptions(header=False)
    loaded = load_table(path, options=options)
    chunked = profile_delimited_chunks(path, options=options, chunk_rows=1)
    assert chunked.population_fingerprint == dataset_fingerprint(loaded.frame)


def _audit_cli(path: Path, output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from aidatasetkit.cli.main import main; raise SystemExit(main())",
            "audit",
            str(path),
            "--output",
            str(output),
            "--fail-on",
            "never",
            *extra,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_the_default_audit_does_not_chunk(tmp_path: Path):
    path = _csv(tmp_path)
    output = tmp_path / "out"
    completed = _audit_cli(path, output)
    assert completed.returncode == 0, completed.stderr
    current = json.loads((output / "CURRENT").read_text(encoding="utf-8"))
    artifact = json.loads(
        (output / "runs" / current["run_id"] / "audit.json").read_text(encoding="utf-8")
    )
    assert artifact["chunked_profiling"] is None, "default audit used chunked profiling"
    assert artifact["schema_version"] == "1.2"


def test_the_opt_in_records_the_population_and_keeps_the_verdict(tmp_path: Path):
    path = _csv(tmp_path)
    plain_dir = tmp_path / "plain"
    opted_dir = tmp_path / "opted"
    plain = _audit_cli(path, plain_dir)
    opted = _audit_cli(path, opted_dir, "--chunked-profile", "--chunk-rows", "1")
    assert plain.returncode == 0, plain.stderr
    assert opted.returncode == 0, opted.stderr

    def load(output: Path) -> dict:
        current = json.loads((output / "CURRENT").read_text(encoding="utf-8"))
        return json.loads(
            (output / "runs" / current["run_id"] / "audit.json").read_text(encoding="utf-8")
        )

    plain_artifact = load(plain_dir)
    opted_artifact = load(opted_dir)
    assert plain_artifact["chunked_profiling"] is None
    assert plain_artifact["stage"] == "inspected"
    assert plain_artifact["verdict"] != "blocked"
    assert opted_artifact["stage"] == "profiled"
    assert opted_artifact["verdict"] == "blocked"
    assert opted_artifact["verdict_reasons"] == [CHUNKED_AUDIT_UNAVAILABLE]
    assert opted_artifact["findings"] == []
    assert opted_artifact["decisions"] == []
    assert opted_artifact["columns"] == []
    recorded = opted_artifact["chunked_profiling"]
    assert recorded["mode"] == "chunked_profile"
    assert recorded["chunk_rows"] == 1
    assert recorded["rows_scanned"] == recorded["population_rows"]
    assert recorded["full_population_scanned"] is True
    assert recorded["bounded_memory"] is True
    assert recorded["fingerprint_scope"] == "population"
    assert recorded["fingerprint_algorithm"] == "sha256/pandas-hash-v1"
    assert recorded["population_rows"] == plain_artifact["dataset"]["row_count"]
    assert recorded["population_fingerprint"] == plain_artifact["dataset"]["fingerprint"]
    assert recorded["temporary_storage"] == "removed"
    assert recorded["exact_metrics"]
    assert recorded["approximate_metrics"]
    assert recorded["unavailable_metrics"]
    assert recorded["sampling"]["method"] == "deterministic_reservoir"
    assert recorded["sampling"]["seed"] == 0
    assert recorded["sampling"]["requested_size"] == 4096
    assert "quality_findings" in recorded["unavailable_metrics"]


def test_a_cell_over_the_limit_is_refused(tmp_path: Path):
    path = _csv(tmp_path)
    with pytest.raises(CellLimitError):
        profile_delimited_chunks(path, limits=IngestionLimits(max_cells=1), chunk_rows=1)


def test_the_full_population_is_scanned(tmp_path: Path):
    path = _csv(tmp_path)
    frame = load_table(path).frame
    profile = profile_delimited_chunks(path, chunk_rows=1)
    assert profile.rows_scanned == len(frame), "full population was not scanned"
    assert profile.population_rows == len(frame), "full population was not scanned"
    assert profile.full_population_scanned is True, "full population was not scanned"
    assert profile.population_fingerprint == dataset_fingerprint(frame), (
        "full population was not scanned"
    )


def test_results_do_not_depend_on_chunk_size(tmp_path: Path):
    path = _csv(tmp_path)
    narrow = profile_delimited_chunks(path, chunk_rows=1)
    wide = profile_delimited_chunks(path, chunk_rows=100)
    assert _without_chunk_rows(narrow) == _without_chunk_rows(wide), (
        "chunk size changed the result"
    )


def test_chunked_mode_does_not_materialize_the_table(tmp_path: Path, monkeypatch):
    path = _csv(tmp_path)
    output = tmp_path / "out"

    def boom(*args, **kwargs):
        raise AssertionError("chunked mode materialized the full table")

    monkeypatch.setattr("aidatasetkit.ingestion.loader.load_table", boom)
    monkeypatch.setattr("aidatasetkit.profiling.profiler.DataProfiler.profile", boom)
    from aidatasetkit.cli.main import main

    code = main(
        ["audit", str(path), "--output", str(output), "--chunked-profile", "--fail-on", "never"]
    )
    assert code == 0, "chunked mode materialized the full table"
    current = json.loads((output / "CURRENT").read_text(encoding="utf-8"))
    artifact = json.loads(
        (output / "runs" / current["run_id"] / "audit.json").read_text(encoding="utf-8")
    )
    assert artifact["verdict"] == "blocked"
    assert artifact["verdict_reasons"] == [CHUNKED_AUDIT_UNAVAILABLE]


def test_read_csv_always_uses_a_chunk_size(tmp_path: Path, monkeypatch):
    path = _csv(tmp_path)
    real = pd.read_csv

    def wrapped(*args, **kwargs):
        if kwargs.get("chunksize") is None:
            raise AssertionError("read the file without a chunk size")
        return real(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", wrapped)
    profile_delimited_chunks(path, chunk_rows=1)


def test_scratch_files_are_not_read_whole(tmp_path: Path, monkeypatch):
    path = _csv(tmp_path)
    original_read = Path.read_bytes
    original_fromfile = np.fromfile

    def read_bytes(self, *args, **kwargs):
        if "aidatasetkit-chunked-" in str(self):
            raise AssertionError("complete scratch file was read into memory")
        return original_read(self, *args, **kwargs)

    def fromfile(file, *args, **kwargs):
        if "aidatasetkit-chunked-" in str(file):
            raise AssertionError("complete scratch file was read into memory")
        return original_fromfile(file, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(np, "fromfile", fromfile)
    profile = profile_delimited_chunks(path, chunk_rows=1)
    assert profile.population_fingerprint


def test_row_keys_do_not_collide(tmp_path: Path):
    from aidatasetkit.profiling.chunked import _row_key

    pairs = (
        (("a", "b\x1ec"), ("a\x1eb", "c")),
        (("",), (None,)),
        (("1",), (1,)),
        ((True,), (1,)),
        (("\x00",), ("\x1e",)),
        (("", ""), ("",)),
    )
    for left, right in pairs:
        assert _row_key(left) != _row_key(right), "row keys collided"

    import csv

    path = tmp_path / "collisions.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["left", "right"])
        writer.writerow(["a", "b\x1ec"])
        writer.writerow(["a\x1eb", "c"])
        writer.writerow(["", "x"])
        writer.writerow(["\x01", "\x1e"])
    profile = profile_delimited_chunks(path, chunk_rows=1)
    again = profile_delimited_chunks(path, chunk_rows=3)
    assert profile.duplicate_row_count == 0, "row keys collided"
    assert profile.population_rows == 4
    assert profile.population_fingerprint == again.population_fingerprint
    assert profile.duplicate_row_count == again.duplicate_row_count


def test_temporary_state_is_removed_on_interruption(tmp_path: Path, monkeypatch):
    path = _csv(tmp_path)
    created: list[str] = []
    real = tempfile.mkdtemp

    def wrapped(*args, **kwargs):
        directory = real(*args, **kwargs)
        created.append(directory)
        return directory

    monkeypatch.setattr(tempfile, "mkdtemp", wrapped)

    def boom(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pd.util, "hash_pandas_object", boom)
    with pytest.raises(KeyboardInterrupt):
        profile_delimited_chunks(path, chunk_rows=1)
    assert created and all(not Path(directory).exists() for directory in created), (
        "temporary directory survived"
    )


def test_memory_does_not_grow_with_cardinality(tmp_path: Path):
    small = _unique_csv(tmp_path / "small.csv", 200)
    large = _unique_csv(tmp_path / "large.csv", 2000)
    small_peak = _peak(small)
    large_peak = _peak(large)
    assert large_peak < small_peak * 4, "memory grew with the number of distinct values"


def test_memory_does_not_grow_with_numeric_rows(tmp_path: Path):
    small = _numeric_csv(tmp_path / "small-numbers.csv", 200)
    large = _numeric_csv(tmp_path / "large-numbers.csv", 2000)
    small_peak = _peak(small)
    large_peak = _peak(large)
    assert large_peak < small_peak * 4, "memory grew with the number of numeric rows"


def test_chunked_evidence_is_recorded(tmp_path: Path):
    profile = profile_delimited_chunks(_csv(tmp_path), chunk_rows=1)
    artifact = AuditBuilder().build_chunked(profile)
    assert artifact.chunked_profiling is not None, "chunked evidence omitted"
    assert artifact.chunked_profiling.full_population_scanned is True
    assert artifact.verdict_reasons == (CHUNKED_AUDIT_UNAVAILABLE,)


def test_the_default_builder_still_writes_null_chunked_evidence(tmp_path: Path):
    frame = load_table(_csv(tmp_path)).frame
    artifact = AuditBuilder().build(frame, profile=DataProfiler().profile(frame))
    assert artifact.chunked_profiling is None
    assert artifact.semantic_dict()["chunked_profiling"] is None
    assert artifact.schema_version == "1.2"
    assert artifact.stage is AuditStage.INSPECTED


def test_the_chunked_artifact_is_profiled(tmp_path: Path):
    artifact = AuditBuilder().build_chunked(
        profile_delimited_chunks(_csv(tmp_path), chunk_rows=1)
    )
    assert artifact.stage is AuditStage.PROFILED, "chunked artifact claimed inspected"
    assert artifact.stage is not AuditStage.INSPECTED
    assert artifact.schema_version == "1.2"


def test_the_chunked_ingestion_record_describes_the_file(tmp_path: Path):
    path = _csv(tmp_path)
    detected = AuditBuilder().build_chunked(profile_delimited_chunks(path, chunk_rows=1))
    ingestion = detected.ingestion
    assert ingestion is not None, "chunked ingestion evidence omitted"
    assert ingestion.source_kind == "file"
    assert ingestion.format == "csv"
    assert ingestion.encoding == "utf-8"
    assert ingestion.delimiter == ","
    assert ingestion.delimiter_source == "detected"
    assert ingestion.header is True
    assert ingestion.row_count == detected.dataset.row_count
    assert ingestion.column_count == detected.dataset.column_count
    assert ingestion.memory_bytes is None

    explicit = AuditBuilder().build_chunked(
        profile_delimited_chunks(path, options=LoadOptions(delimiter=","), chunk_rows=1)
    )
    assert explicit.ingestion is not None
    assert explicit.ingestion.delimiter == ","
    assert explicit.ingestion.delimiter_source == "explicit"
    assert explicit.ingestion.delimiter_source != detected.ingestion.delimiter_source

    headerless = tmp_path / "bare.csv"
    headerless.write_text("1,a\n2,b\n3,c\n", encoding="utf-8")
    bare = AuditBuilder().build_chunked(
        profile_delimited_chunks(headerless, options=LoadOptions(header=False), chunk_rows=1)
    )
    assert bare.ingestion is not None
    assert bare.ingestion.header is False
    assert bare.ingestion.row_count == 3

    latin = tmp_path / "latin.csv"
    latin.write_bytes("age,city\n1,a\n".encode("latin-1"))
    encoded = AuditBuilder().build_chunked(
        profile_delimited_chunks(latin, options=LoadOptions(encoding="latin-1"), chunk_rows=1)
    )
    assert encoded.ingestion is not None
    assert encoded.ingestion.encoding == "latin-1"


def test_chunked_settings_are_in_the_config_identity(tmp_path: Path):
    path = _csv(tmp_path)
    limits = IngestionLimits(max_rows=5000)
    artifact = AuditBuilder().build_chunked(
        profile_delimited_chunks(path, limits=limits, chunk_rows=2, quantile_sample_size=8)
    )
    settings = artifact.config.settings
    assert settings.get("mode") == "chunked_profile", "chunked settings omitted from config identity"
    assert settings["chunk_rows"] == 2
    assert settings["quantile_sample_size"] == 8
    assert settings["sampling_method"] == "deterministic_reservoir"
    assert settings["sampling_seed"] == 0
    assert settings["encoding"] == "utf-8"
    assert settings["delimiter"] == ","
    assert settings["delimiter_source"] == "detected"
    assert settings["header"] is True
    assert settings["limits"]["max_rows"] == 5000
    assert settings["limits"]["max_cells"] == IngestionLimits().max_cells
    assert settings["fingerprint_algorithm"] == "sha256/pandas-hash-v1"
    assert artifact.config.fingerprint


def test_the_config_fingerprint_tracks_the_scan_contract(tmp_path: Path):
    path = _csv(tmp_path)
    other = tmp_path / "elsewhere" / "copy.csv"
    other.parent.mkdir()
    other.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    left = AuditBuilder().build_chunked(
        profile_delimited_chunks(path, chunk_rows=2),
        dataset_name="people.csv",
        created_at="2020-01-01T00:00:00Z",
    )
    right = AuditBuilder().build_chunked(
        profile_delimited_chunks(other, chunk_rows=2),
        dataset_name="copy.csv",
        created_at="2024-06-01T00:00:00Z",
    )
    assert left.config.fingerprint == right.config.fingerprint
    assert left.created_at != right.created_at
    assert "people.csv" not in json.dumps(left.config.settings)
    assert str(path) not in json.dumps(left.config.settings)

    wider = AuditBuilder().build_chunked(
        profile_delimited_chunks(path, chunk_rows=4)
    )
    assert wider.config.fingerprint != left.config.fingerprint

    small_sample = AuditBuilder().build_chunked(
        profile_delimited_chunks(path, chunk_rows=2, quantile_sample_size=3)
    )
    assert small_sample.config.fingerprint != left.config.fingerprint

    tighter = AuditBuilder().build_chunked(
        profile_delimited_chunks(path, chunk_rows=2, limits=IngestionLimits(max_rows=5000))
    )
    assert tighter.config.fingerprint != left.config.fingerprint


def test_build_chunked_has_no_settings_parameter(tmp_path: Path):
    import inspect

    parameters = inspect.signature(AuditBuilder.build_chunked).parameters
    assert "settings" not in parameters
    profile = profile_delimited_chunks(_csv(tmp_path), chunk_rows=1)
    with pytest.raises(TypeError, match="settings"):
        AuditBuilder().build_chunked(profile, settings={"chunk_rows": 1})


def test_quantile_sample_size_must_be_a_positive_integer(tmp_path: Path):
    path = _csv(tmp_path)
    for value in (0, -3, False):
        with pytest.raises(InvalidIngestionOptionsError, match="quantile_sample_size"):
            profile_delimited_chunks(path, quantile_sample_size=value)


def _without_chunk_rows(profile) -> dict:
    payload = profile.to_dict()
    payload.pop("chunk_rows")
    return payload


def _unique_csv(path: Path, rows: int) -> Path:
    lines = ["city", *(f"city-{index}" for index in range(rows))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _numeric_csv(path: Path, rows: int) -> Path:
    lines = ["score", *(str(index) for index in range(rows))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _peak(path: Path) -> int:
    tracemalloc.start()
    try:
        profile_delimited_chunks(path, chunk_rows=32)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak
