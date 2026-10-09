"""G1-W5: chunked CSV/TSV profiling is opt-in, exact, and kept out of the verdict."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from aidatasetkit.core.exceptions import (
    FileSizeLimitError,
    InvalidIngestionOptionsError,
    RowLimitError,
)
from aidatasetkit.evidence import AuditBuilder
from aidatasetkit.evidence.fingerprint import dataset_fingerprint
from aidatasetkit.ingestion import IngestionLimits, load_table
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
    assert chunked.approximations == ()
    assert chunked.temporary_storage == "removed"
    by_name = {column.name: column for column in chunked.columns}
    for column in profile.column_profiles:
        measured = by_name[str(column.name)]
        assert measured.missing_count == column.missing_count
        assert measured.count == column.count
        assert measured.unique_count == column.unique_count
        assert measured.pandas_dtype == column.pandas_dtype
        if column.numeric is None:
            assert measured.mean is None
            continue
        assert measured.minimum == column.numeric.minimum
        assert measured.maximum == column.numeric.maximum
        assert measured.mean == column.numeric.mean
        assert measured.q25 == column.numeric.q25
        assert measured.median == column.numeric.median
        assert measured.q75 == column.numeric.q75


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
    profile = profile_delimited_chunks(path, chunk_rows=2, approximate_quantiles_above=2)
    assert profile.approximations, "approximation was not labeled"
    assert {item.label for item in profile.approximations} == {"deterministic_approximation"}
    assert {item.method for item in profile.approximations} == {"deterministic_even_stride"}
    approximated = {item.column for item in profile.approximations}
    for column in profile.columns:
        if column.name in approximated:
            assert column.q25 is None, "approximate quantile stored as exact"
            assert column.median is None
            assert column.q75 is None
            assert column.mean is not None
    exact = profile_delimited_chunks(path, chunk_rows=2)
    age = next(column for column in profile.columns if column.name == "age")
    exact_age = next(column for column in exact.columns if column.name == "age")
    assert age.mean == exact_age.mean
    assert age.minimum == exact_age.minimum
    assert age.maximum == exact_age.maximum


def test_a_labeled_approximation_does_not_change_the_verdict(tmp_path: Path):
    path = _csv(tmp_path)
    frame = load_table(path).frame
    measured = DataProfiler().profile(frame)
    chunked = profile_delimited_chunks(path, chunk_rows=1, approximate_quantiles_above=2)
    assert chunked.approximations
    builder = AuditBuilder()
    plain = builder.build(frame, profile=measured)
    marked = builder.build(frame, profile=measured, chunked_profiling=chunked)
    assert plain.verdict == marked.verdict, "approximation changed the verdict inputs"
    assert plain.verdict_reasons == marked.verdict_reasons, "approximation changed the verdict inputs"
    assert plain.findings == marked.findings, "approximation changed the verdict inputs"
    assert plain.chunked_profiling is None
    assert marked.chunked_profiling is not None
    assert marked.semantic_dict()["chunked_profiling"]["fingerprint_scope"] == "population"


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
    assert plain_artifact["verdict"] == opted_artifact["verdict"]
    assert plain_artifact["findings"] == opted_artifact["findings"]
    recorded = opted_artifact["chunked_profiling"]
    assert recorded["fingerprint_scope"] == "population"
    assert recorded["population_rows"] == plain_artifact["dataset"]["row_count"]
    assert recorded["population_fingerprint"] == plain_artifact["dataset"]["fingerprint"]
    assert recorded["temporary_storage"] == "removed"
    assert recorded["approximations"] == []
