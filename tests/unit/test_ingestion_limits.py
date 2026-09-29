from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aidatasetkit.cli.main import build_parser, main
from aidatasetkit.core.exceptions import (
    CellLimitError,
    ColumnLimitError,
    FileSizeLimitError,
    FieldLengthLimitError,
    InvalidIngestionOptionsError,
    KeyLimitError,
    RecordLimitError,
    RowLimitError,
)
from aidatasetkit.ingestion import IngestionLimits, load_table
from aidatasetkit.facade import AIDataFacade


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class TestIngestionLimits:
    def test_defaults_are_finite_and_immutable(self):
        limits = IngestionLimits()
        assert limits.max_source_bytes == 64 * 1024 * 1024
        with pytest.raises(InvalidIngestionOptionsError):
            IngestionLimits(max_rows=0)
        with pytest.raises(InvalidIngestionOptionsError):
            IngestionLimits(max_columns=-1)
        with pytest.raises(InvalidIngestionOptionsError):
            IngestionLimits(max_cells=True)
        with pytest.raises(AttributeError):
            limits.max_rows = 1

    def test_file_bytes_are_rejected_before_parsing(self, tmp_path):
        path = write(tmp_path / "data.csv", "a,b\n1,2\n")
        with pytest.raises(FileSizeLimitError) as caught:
            load_table(path, limits=IngestionLimits(max_source_bytes=path.stat().st_size - 1))
        assert caught.value.limit_name == "max_source_bytes"
        assert caught.value.input_kind == "file"
        assert str(path) not in str(caught.value)

    def test_rows_columns_cells_and_field_length_are_guarded(self, tmp_path):
        path = write(tmp_path / "data.csv", "a,b\n1,2\n3,4\n")
        with pytest.raises(RowLimitError):
            load_table(path, limits=IngestionLimits(max_rows=1))
        with pytest.raises(ColumnLimitError):
            load_table(path, limits=IngestionLimits(max_columns=1))
        with pytest.raises(CellLimitError):
            load_table(path, limits=IngestionLimits(max_cells=3))
        with pytest.raises(FieldLengthLimitError):
            load_table(write(tmp_path / "long.csv", "a\n\u0623\u062d\u0645\u062f\n"), limits=IngestionLimits(max_field_length=2))

    def test_dataframe_is_not_copied_or_mutated(self):
        frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        before = frame.copy(deep=True)
        with pytest.raises(CellLimitError):
            load_table(frame, limits=IngestionLimits(max_cells=3))
        pd.testing.assert_frame_equal(frame, before)
        assert load_table(frame, limits=IngestionLimits(max_cells=4)).frame is frame

    def test_records_are_rejected_before_dataframe_materialization(self):
        records = [{"a": 1}, {"a": 2}]
        with pytest.raises(RecordLimitError):
            load_table(records, limits=IngestionLimits(max_records=1))
        with pytest.raises(KeyLimitError):
            load_table([{"a": 1, "b": 2}], limits=IngestionLimits(max_keys_per_record=1))
        assert records == [{"a": 1}, {"a": 2}]

    def test_facade_checks_limits_before_storing_the_frame(self):
        facade = AIDataFacade(target="label", task="classification")
        frame = pd.DataFrame({"a": [1, 2], "label": [0, 1]})
        with pytest.raises(CellLimitError):
            facade.load(frame, limits=IngestionLimits(max_cells=3))
        assert facade.stage.value == "empty"

    def test_cli_help_and_invalid_limit(self, capsys):
        with pytest.raises(SystemExit) as help_exit:
            main(["audit", "--help"])
        assert help_exit.value.code == 0
        assert "--max-input-bytes" in capsys.readouterr().out
        with pytest.raises(SystemExit) as invalid_exit:
            main(["audit", "missing.csv", "--max-rows", "0"])
        assert invalid_exit.value.code == 1
        assert "positive integer" in capsys.readouterr().err

    def test_cli_defaults_match_library_defaults(self):
        args = build_parser().parse_args(["audit", "input.csv"])
        defaults = IngestionLimits()
        assert args.max_source_bytes == defaults.max_source_bytes
        assert args.max_rows == defaults.max_rows
        assert args.max_columns == defaults.max_columns
        assert args.max_cells == defaults.max_cells
        assert args.max_field_length == defaults.max_field_length
