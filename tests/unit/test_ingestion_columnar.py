"""G1-W3: a Parquet or Feather file is its table, or a structured refusal.

pyarrow is an optional dependency, so this whole module is skipped where the
``parquet`` extra is not installed; the absence itself is tested in
``tests/integration/test_ingestion_without_pyarrow.py``. Fixtures are written
by each test into its own temporary directory.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pyarrow = pytest.importorskip("pyarrow", reason="the parquet extra is not installed")
import pyarrow.parquet as pq

from aidatasetkit.core.exceptions import (
    CellLimitError,
    ColumnLimitError,
    DuplicateHeadersError,
    EmptyInputError,
    FileSizeLimitError,
    InvalidIngestionOptionsError,
    MalformedInputError,
    RowLimitError,
)
from aidatasetkit.ingestion import (
    IngestionLimits,
    LoadOptions,
    SourceKind,
    TableFormat,
    load_table,
    resolve_format,
)

FRAME = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "ok": [True, False, True]})


def write_parquet(tmp_path: Path, frame: pd.DataFrame = FRAME, name: str = "data.parquet") -> Path:
    path = tmp_path / name
    frame.to_parquet(path, index=False)
    return path


def write_feather(tmp_path: Path, frame: pd.DataFrame = FRAME, name: str = "data.feather") -> Path:
    path = tmp_path / name
    frame.to_feather(path)
    return path


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


class TestColumnarReading:
    @pytest.mark.parametrize("writer", [write_parquet, write_feather], ids=["parquet", "feather"])
    def test_a_round_trip_preserves_the_frame(self, tmp_path, writer):
        loaded = load_table(writer(tmp_path))
        assert loaded.frame.shape == FRAME.shape
        assert list(loaded.frame.columns) == list(FRAME.columns)
        assert loaded.frame["a"].tolist() == [1, 2, 3]
        assert loaded.frame["ok"].tolist() == [True, False, True]

    @pytest.mark.parametrize(
        ("writer", "fmt"), [(write_parquet, TableFormat.PARQUET), (write_feather, TableFormat.FEATHER)]
    )
    def test_metadata_describes_the_file(self, tmp_path, writer, fmt):
        metadata = load_table(writer(tmp_path)).metadata
        assert metadata.source_kind is SourceKind.FILE
        assert metadata.format is fmt
        assert metadata.encoding is None
        assert metadata.delimiter is None and metadata.delimiter_source is None
        assert metadata.header is None
        assert (metadata.row_count, metadata.column_count) == (3, 3)
        assert metadata.memory_bytes > 0

    def test_arrow_is_feather(self, tmp_path):
        path = write_feather(tmp_path, name="data.arrow")
        assert resolve_format(path) is TableFormat.FEATHER
        assert load_table(path).frame.shape == FRAME.shape

    def test_a_compressed_parquet_round_trips(self, tmp_path):
        path = tmp_path / "snappy.parquet"
        FRAME.to_parquet(path, index=False, compression="gzip")
        assert load_table(path).frame["b"].tolist() == ["x", "y", "z"]


# --------------------------------------------------------------------------- #
# Resource limits, enforced from the Parquet footer before any data is read
# --------------------------------------------------------------------------- #


class TestColumnarLimits:
    @pytest.mark.parametrize("writer", [write_parquet, write_feather], ids=["parquet", "feather"])
    @pytest.mark.parametrize(
        ("limit", "exact"), [("max_rows", 3), ("max_columns", 3), ("max_cells", 9)]
    )
    def test_exactly_at_each_limit_is_accepted_and_one_under_is_refused(
        self, tmp_path, writer, limit, exact
    ):
        path = writer(tmp_path)
        assert load_table(path, limits=IngestionLimits(**{limit: exact})).frame.shape == (3, 3)
        with pytest.raises((RowLimitError, ColumnLimitError, CellLimitError)) as caught:
            load_table(path, limits=IngestionLimits(**{limit: exact - 1}))
        assert caught.value.limit_name == limit

    def test_a_parquet_over_the_row_limit_is_refused_without_reading_data(self, tmp_path, monkeypatch):
        """The footer alone condemns it; materializing the columns would be waste."""
        path = write_parquet(tmp_path)

        def data_read(*args, **kwargs):
            raise AssertionError("an over-limit parquet was materialized")

        monkeypatch.setattr(pq, "read_table", data_read)
        with pytest.raises(RowLimitError) as caught:
            load_table(path, limits=IngestionLimits(max_rows=2))
        assert caught.value.observed_value == 3

    def test_a_feather_over_the_column_limit_is_refused_without_reading_data(self, tmp_path, monkeypatch):
        path = write_feather(tmp_path)

        def data_read(self, *args, **kwargs):
            raise AssertionError("an over-column-limit feather was materialized")

        monkeypatch.setattr(pyarrow.ipc.RecordBatchFileReader, "read_all", data_read)
        with pytest.raises(ColumnLimitError):
            load_table(path, limits=IngestionLimits(max_columns=2))

    def test_a_feather_over_the_row_limit_is_refused_before_the_pandas_conversion(self, tmp_path):
        """Feather's footer has no row count, so the rows are condemned on the
        Arrow table -- the frame is never built."""
        path = write_feather(tmp_path)
        with pytest.raises(RowLimitError) as caught:
            load_table(path, limits=IngestionLimits(max_rows=2))
        assert caught.value.observed_value == 3

    def test_the_byte_preflight_applies_before_pyarrow_is_called(self, tmp_path, monkeypatch):
        path = write_parquet(tmp_path)

        def reader_reached(*args, **kwargs):
            raise AssertionError("oversized input reached the columnar reader")

        monkeypatch.setattr("aidatasetkit.ingestion.loader.read_columnar", reader_reached)
        with pytest.raises(FileSizeLimitError):
            load_table(path, limits=IngestionLimits(max_source_bytes=path.stat().st_size - 1))


# --------------------------------------------------------------------------- #
# Structured refusals
# --------------------------------------------------------------------------- #


class TestColumnarRefusals:
    @pytest.mark.parametrize("writer", [write_parquet, write_feather], ids=["parquet", "feather"])
    def test_an_empty_table_is_empty_input(self, tmp_path, writer):
        with pytest.raises(EmptyInputError):
            load_table(writer(tmp_path, FRAME.iloc[0:0]))

    @pytest.mark.parametrize("name", ["broken.parquet", "broken.feather"])
    def test_a_corrupt_file_is_malformed_input_never_a_bare_pyarrow_error(self, tmp_path, name):
        path = tmp_path / name
        path.write_bytes(b"this is not a columnar file at all" * 8)
        with pytest.raises(MalformedInputError, match="not a readable"):
            load_table(path)

    def test_duplicate_column_labels_are_refused(self, tmp_path):
        table = pyarrow.Table.from_arrays(
            [pyarrow.array([1, 2]), pyarrow.array([3, 4])], names=["a", "a"]
        )
        path = tmp_path / "dup.parquet"
        pq.write_table(table, path)
        with pytest.raises(DuplicateHeadersError, match="duplicate column labels"):
            load_table(path)

    @pytest.mark.parametrize("writer", [write_parquet, write_feather], ids=["parquet", "feather"])
    def test_text_options_are_refused_for_binary_formats(self, tmp_path, writer):
        with pytest.raises(InvalidIngestionOptionsError, match="LoadOptions"):
            load_table(writer(tmp_path), options=LoadOptions(encoding="latin-1"))
