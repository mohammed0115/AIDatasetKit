"""G1-W6: one ordinary SQLite table, read-only, or a structured refusal.

Not certified. G1-14 stays MISSING_PENDING_CERTIFICATION until a later review.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from aidatasetkit.core.exceptions import (
    CellLimitError,
    ColumnLimitError,
    EmptyInputError,
    FieldLengthLimitError,
    FileSizeLimitError,
    InputNotFoundError,
    InvalidIngestionOptionsError,
    MalformedInputError,
    RowLimitError,
    UnsupportedFormatError,
)
from aidatasetkit.evidence import AuditBuilder, config_fingerprint
from aidatasetkit.ingestion import load_table
from aidatasetkit.ingestion.sqlite import quote_identifier, read_sqlite
from aidatasetkit.ingestion.types import IngestionLimits, LoadOptions, TableFormat
from aidatasetkit.profiling import DataProfiler

import aidatasetkit.ingestion.sqlite as sqlite_reader


def _write(path: Path, statements: str) -> Path:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(statements)
        connection.commit()
    finally:
        connection.close()
    return path


def _people(path: Path) -> Path:
    return _write(
        path,
        "CREATE TABLE people (n INTEGER, label TEXT, amount REAL, blob BLOB);"
        "INSERT INTO people VALUES (1, 'a', 1.5, X'0001');"
        "INSERT INTO people VALUES (NULL, NULL, NULL, NULL);",
    )


class _ConnectionLog:
    """A stand-in that records calls and forwards them. sqlite3.Connection is immutable."""

    def __init__(self, inner: sqlite3.Connection, statements: list[str], configs: list[tuple]):
        self._inner = inner
        self.statements = statements
        self.configs = configs

    def execute(self, sql, *args, **kwargs):
        if isinstance(sql, str):
            self.statements.append(sql)
        return self._inner.execute(sql, *args, **kwargs)

    def executemany(self, *args, **kwargs):
        raise AssertionError("arbitrary/user SQL accepted")

    def executescript(self, *args, **kwargs):
        raise AssertionError("arbitrary/user SQL accepted")

    def enable_load_extension(self, enabled):
        raise AssertionError("extensions are never enabled")

    def setconfig(self, operation, value):
        self.configs.append((operation, value))
        return self._inner.setconfig(operation, value)

    def close(self):
        return self._inner.close()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _spy_connect(monkeypatch: pytest.MonkeyPatch):
    statements: list[str] = []
    configs: list[tuple] = []
    uris: list[str] = []
    real = sqlite3.connect

    def spy(database, *args, **kwargs):
        uris.append(str(database))
        return _ConnectionLog(real(database, *args, **kwargs), statements, configs)

    monkeypatch.setattr(sqlite3, "connect", spy)
    return uris, statements, configs


def _artifact(path: Path, table: str | None = None, created_at: str = "2020-01-01T00:00:00Z"):
    loaded = load_table(path, table=table)
    profile = DataProfiler().profile(loaded.frame)
    return AuditBuilder(dataset_name="dataset").build(
        loaded.frame,
        profile=profile,
        ingestion=loaded.metadata,
        source_selector=loaded.selector,
        settings={"task_hint": None},
        created_at=created_at,
    )


class TestSelection:
    def test_one_table_is_selected(self, tmp_path: Path):
        loaded = load_table(_people(tmp_path / "one.sqlite"))
        assert loaded.selector is not None
        assert loaded.selector.to_dict() == {"kind": "table", "name": "people"}
        assert list(loaded.frame.columns) == ["n", "label", "amount", "blob"]
        assert len(loaded.frame) == 2

    def test_multiple_tables_require_an_explicit_name(self, tmp_path: Path):
        path = _write(
            tmp_path / "many.sqlite",
            "CREATE TABLE a (n INTEGER); INSERT INTO a VALUES (1);"
            "CREATE TABLE b (n INTEGER); INSERT INTO b VALUES (2);",
        )
        try:
            load_table(path)
        except MalformedInputError as error:
            assert "would be a guess" in str(error)
            return
        raise AssertionError("multiple tables silently select one")

    def test_an_explicit_table_is_the_one_read(self, tmp_path: Path):
        path = _write(
            tmp_path / "named.sqlite",
            "CREATE TABLE a (n INTEGER); INSERT INTO a VALUES (1);"
            "CREATE TABLE b (n INTEGER); INSERT INTO b VALUES (2);",
        )
        loaded = load_table(path, table="b")
        assert loaded.selector is not None and loaded.selector.name == "b"
        assert loaded.frame["n"].tolist() == [2]

    def test_a_missing_table_is_refused(self, tmp_path: Path):
        path = _people(tmp_path / "missing.sqlite")
        with pytest.raises(MalformedInputError, match="no ordinary table"):
            load_table(path, table="absent")

    def test_zero_ordinary_tables_are_empty(self, tmp_path: Path):
        path = _write(tmp_path / "view.sqlite", "CREATE VIEW only_view AS SELECT 1 AS n;")
        with pytest.raises(EmptyInputError, match="no ordinary table"):
            load_table(path)

    def test_an_empty_table_keeps_its_columns(self, tmp_path: Path):
        path = _write(tmp_path / "empty-table.sqlite", "CREATE TABLE people (n INTEGER, label TEXT);")
        loaded = load_table(path)
        assert list(loaded.frame.columns) == ["n", "label"]
        assert len(loaded.frame) == 0

    def test_an_unusual_identifier_is_data(self, tmp_path: Path):
        name = 'odd" name'
        path = tmp_path / "odd.sqlite"
        connection = sqlite3.connect(path)
        try:
            connection.execute(f"CREATE TABLE {quote_identifier(name)} (n INTEGER)")
            connection.execute(f"INSERT INTO {quote_identifier(name)} VALUES (7)")
            connection.commit()
        finally:
            connection.close()
        loaded = load_table(path)
        assert loaded.selector is not None and loaded.selector.name == name
        assert loaded.frame["n"].tolist() == [7]

    def test_an_injection_shaped_name_is_not_executed_as_sql(self, tmp_path: Path):
        name = 'x"; DROP TABLE x; --'
        path = tmp_path / "inject.sqlite"
        connection = sqlite3.connect(path)
        try:
            connection.execute(f"CREATE TABLE {quote_identifier(name)} (n INTEGER)")
            connection.execute(f"INSERT INTO {quote_identifier(name)} VALUES (4)")
            connection.commit()
        finally:
            connection.close()
        monkeypatch = pytest.MonkeyPatch()
        _uris, statements, _configs = _spy_connect(monkeypatch)
        failure = None
        try:
            loaded = load_table(path)
        except Exception as error:
            failure = error
            loaded = None
        finally:
            monkeypatch.undo()
        assert statements, "arbitrary/user SQL accepted"
        for sql in statements:
            assert sql.split(None, 1)[0].upper() in {"PRAGMA", "SELECT"}, (
                "arbitrary/user SQL accepted"
            )
            assert not sql.upper().startswith("ATTACH"), "arbitrary/user SQL accepted"
        assert any(sql.upper().startswith("SELECT *") for sql in statements), (
            "arbitrary/user SQL accepted"
        )
        assert failure is None, "arbitrary/user SQL accepted"
        assert loaded is not None and loaded.frame["n"].tolist() == [4]

    def test_an_internal_table_is_excluded(self, tmp_path: Path):
        path = _write(
            tmp_path / "seq.sqlite",
            "CREATE TABLE people (id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT);"
            "INSERT INTO people (label) VALUES ('a');",
        )
        try:
            loaded = load_table(path)
        except Exception as error:
            raise AssertionError("sqlite_* internal table accepted") from error
        assert loaded.selector is not None and loaded.selector.name == "people", (
            "sqlite_* internal table accepted"
        )

    def test_a_view_is_excluded(self, tmp_path: Path):
        path = _write(tmp_path / "only-view.sqlite", "CREATE VIEW people AS SELECT 1 AS n;")
        try:
            load_table(path)
        except EmptyInputError:
            return
        raise AssertionError("view accepted")

    def test_a_virtual_table_is_excluded(self, tmp_path: Path):
        path = tmp_path / "virtual.sqlite"
        connection = sqlite3.connect(path)
        try:
            connection.execute("CREATE VIRTUAL TABLE docs USING fts5(body)")
            connection.commit()
        finally:
            connection.close()
        try:
            load_table(path, table="docs")
        except MalformedInputError:
            return
        raise AssertionError("virtual table accepted")

    def test_sqlite_and_sqlite3_suffixes_are_read(self, tmp_path: Path):
        for suffix in (".sqlite", ".sqlite3"):
            loaded = load_table(_people(tmp_path / f"data{suffix}"))
            assert loaded.metadata.format is TableFormat.SQLITE
            assert len(loaded.frame) == 2

    def test_a_db_suffix_is_refused(self, tmp_path: Path):
        path = _people(tmp_path / "data.db")
        try:
            load_table(path)
        except UnsupportedFormatError:
            return
        raise AssertionError(".db accepted")

    def test_table_is_refused_for_other_formats(self, tmp_path: Path):
        path = tmp_path / "rows.csv"
        path.write_text("n\n1\n", encoding="utf-8")
        try:
            load_table(path, table="people")
        except InvalidIngestionOptionsError:
            return
        raise AssertionError("table= accepted for a non-SQLite format")


class TestSecurity:
    def test_mode_ro_is_used(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        path = _people(tmp_path / "ro.sqlite")
        uris, _statements, _configs = _spy_connect(monkeypatch)
        load_table(path)
        assert any("mode=ro" in uri for uri in uris), "mode=ro"

    def test_query_only_is_enabled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        path = _people(tmp_path / "qo.sqlite")
        _uris, statements, _configs = _spy_connect(monkeypatch)
        load_table(path)
        assert any(sql.replace(" ", "").upper() == "PRAGMAQUERY_ONLY=ON" for sql in statements), (
            "query_only"
        )

    def test_extensions_are_never_enabled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        path = _people(tmp_path / "ext.sqlite")
        _spy_connect(monkeypatch)
        load_table(path)

    def test_a_write_attempt_fails(self, tmp_path: Path):
        path = _people(tmp_path / "write.sqlite")
        before = path.read_bytes()
        connection = sqlite_reader._connect(path)
        try:
            with pytest.raises(sqlite3.OperationalError):
                connection.execute("CREATE TABLE extra (n INTEGER)")
        finally:
            connection.close()
        assert path.read_bytes() == before

    def test_a_missing_database_is_not_created(self, tmp_path: Path):
        path = tmp_path / "absent.sqlite"
        try:
            read_sqlite(path, TableFormat.SQLITE, LoadOptions(), IngestionLimits())
        except InputNotFoundError:
            pass
        except Exception:
            if path.exists():
                raise AssertionError("missing path creates a database") from None
            raise
        if path.exists():
            raise AssertionError("missing path creates a database")

    def test_the_database_bytes_are_unchanged(self, tmp_path: Path):
        path = _people(tmp_path / "bytes.sqlite")
        before = path.read_bytes()
        siblings = set(tmp_path.iterdir())
        load_table(path)
        assert path.read_bytes() == before
        assert set(tmp_path.iterdir()) == siblings

    def test_defensive_config_is_used_when_available(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        if not hasattr(sqlite3, "SQLITE_DBCONFIG_DEFENSIVE"):
            pytest.skip("this Python build has no SQLITE_DBCONFIG_DEFENSIVE")
        path = _people(tmp_path / "defensive.sqlite")
        _uris, _statements, configs = _spy_connect(monkeypatch)
        load_table(path)
        assert any(operation == sqlite3.SQLITE_DBCONFIG_DEFENSIVE for operation, _value in configs), (
            "defensive setconfig"
        )

    def test_a_corrupt_database_does_not_leak_the_driver_error(self, tmp_path: Path):
        path = tmp_path / "bad.sqlite"
        path.write_bytes(b"this is not a database")
        try:
            load_table(path)
        except MalformedInputError as error:
            text = str(error)
            assert str(path) not in text, "raw path"
            assert "SELECT" not in text, "raw path"
            return
        except Exception as error:
            raise AssertionError(f"unwrapped {type(error).__name__}: raw path") from None
        raise AssertionError("unwrapped: raw path")


class TestLimits:
    def test_source_bytes_are_refused_before_opening(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        path = _people(tmp_path / "big.sqlite")
        def spy(*args, **kwargs):
            raise AssertionError("opened")
        monkeypatch.setattr(sqlite3, "connect", spy)
        with pytest.raises(FileSizeLimitError):
            load_table(path, limits=IngestionLimits(max_source_bytes=1))

    def test_too_many_columns_are_refused(self, tmp_path: Path):
        path = _write(tmp_path / "wide.sqlite", "CREATE TABLE t (a INTEGER, b INTEGER);")
        with pytest.raises(ColumnLimitError):
            load_table(path, limits=IngestionLimits(max_columns=1))

    def test_too_many_rows_are_refused_before_fetch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        path = _people(tmp_path / "rows.sqlite")

        def fetch(*args, **kwargs):
            raise AssertionError("row limit checked after materialization")

        monkeypatch.setattr(sqlite_reader, "_fetch_rows", fetch)
        with pytest.raises(RowLimitError):
            load_table(path, limits=IngestionLimits(max_rows=1))

    def test_too_many_cells_are_refused(self, tmp_path: Path):
        path = _people(tmp_path / "cells.sqlite")
        try:
            load_table(path, limits=IngestionLimits(max_cells=1))
        except CellLimitError:
            return
        raise AssertionError("max_cells ignored")

    def test_an_oversized_text_value_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        path = _write(
            tmp_path / "text.sqlite",
            "CREATE TABLE t (label TEXT); INSERT INTO t VALUES ('abcdefghij');",
        )

        def fetch(*args, **kwargs):
            raise AssertionError("materialized")

        monkeypatch.setattr(sqlite_reader, "_fetch_rows", fetch)
        with pytest.raises(FieldLengthLimitError):
            load_table(path, limits=IngestionLimits(max_field_length=4))

    def test_an_oversized_blob_is_refused(self, tmp_path: Path):
        path = tmp_path / "blob.sqlite"
        connection = sqlite3.connect(path)
        try:
            connection.execute("CREATE TABLE t (blob BLOB)")
            connection.execute("INSERT INTO t VALUES (?)", (b"abcdefghij",))
            connection.commit()
        finally:
            connection.close()
        try:
            load_table(path, limits=IngestionLimits(max_field_length=4))
        except FieldLengthLimitError:
            return
        raise AssertionError("max_field_length ignores BLOB")

    def test_text_length_is_its_own_check(self, tmp_path: Path):
        path = _write(
            tmp_path / "text-only.sqlite",
            "CREATE TABLE t (label TEXT); INSERT INTO t VALUES ('abcdefghij');",
        )
        try:
            load_table(path, limits=IngestionLimits(max_field_length=4))
        except FieldLengthLimitError:
            return
        raise AssertionError("max_field_length ignores TEXT")

    def test_a_refusal_publishes_nothing(self, tmp_path: Path):
        from aidatasetkit.cli.main import main

        path = _people(tmp_path / "refuse.sqlite")
        output = tmp_path / "out"
        code = main(["audit", str(path), "--output", str(output), "--max-rows", "1"])
        assert code == 1
        assert not (output / "CURRENT").exists(), "partial artifact"


class TestValues:
    def test_sqlite_values_round_trip(self, tmp_path: Path):
        loaded = load_table(_people(tmp_path / "types.sqlite"))
        first = loaded.frame.iloc[0].tolist()
        assert first == [1, "a", 1.5, b"\x00\x01"]
        assert [type(value) for value in first] == [int, str, float, bytes]
        assert loaded.frame.iloc[1].tolist() == [None, None, None, None]

    def test_column_order_follows_the_catalog(self, tmp_path: Path):
        path = _write(
            tmp_path / "order.sqlite",
            "CREATE TABLE t (z INTEGER, a TEXT); INSERT INTO t VALUES (1, 'q');",
        )
        assert list(load_table(path).frame.columns) == ["z", "a"]

    def test_a_zero_byte_file_is_empty(self, tmp_path: Path):
        path = tmp_path / "zero.sqlite"
        path.write_bytes(b"")
        with pytest.raises(EmptyInputError, match="0 bytes"):
            load_table(path)


class TestEvidence:
    def test_the_artifact_records_the_table(self, tmp_path: Path):
        artifact = _artifact(_people(tmp_path / "ev.sqlite"))
        assert artifact.schema_version == "1.3", "schema incorrectly remains 1.2"
        assert artifact.ingestion is not None
        assert artifact.ingestion.source_kind == "file"
        assert artifact.ingestion.format == "sqlite"
        assert artifact.source_selector is not None, "table selector omitted from artifact evidence"
        assert artifact.source_selector.to_dict() == {"kind": "table", "name": "people"}
        assert "SELECT" not in str(artifact.semantic_dict())
        recorded = artifact.semantic_dict()["source_selector"]
        assert recorded == {"kind": "table", "name": "people"}
        assert artifact.config.settings["source_selector"] == recorded

    def test_a_different_table_changes_the_config_fingerprint(self, tmp_path: Path):
        path = _write(
            tmp_path / "two.sqlite",
            "CREATE TABLE a (n INTEGER); INSERT INTO a VALUES (1);"
            "CREATE TABLE b (n INTEGER); INSERT INTO b VALUES (1);",
        )
        left = _artifact(path, table="a")
        right = _artifact(path, table="b", created_at="2030-01-01T00:00:00Z")
        assert left.config.fingerprint != right.config.fingerprint, (
            "table selector omitted from config fingerprint"
        )
        assert left.config.fingerprint == config_fingerprint(left.config.settings)

    def test_the_same_table_matches_across_paths_and_timestamps(self, tmp_path: Path):
        first = _people(tmp_path / "one.sqlite")
        second = tmp_path / "nested" / "two.sqlite"
        second.parent.mkdir()
        second.write_bytes(first.read_bytes())
        left = _artifact(first, created_at="2020-01-01T00:00:00Z")
        right = _artifact(second, created_at="2030-05-05T00:00:00Z")
        assert left.config.fingerprint == right.config.fingerprint
        assert "one.sqlite" not in str(left.config.settings)
        assert left.created_at != right.created_at

    def test_a_non_sqlite_artifact_has_a_null_selector(self, tmp_path: Path):
        path = tmp_path / "rows.csv"
        path.write_text("n\n1\n", encoding="utf-8")
        loaded = load_table(path)
        artifact = AuditBuilder().build(
            loaded.frame,
            profile=DataProfiler().profile(loaded.frame),
            ingestion=loaded.metadata,
        )
        assert artifact.source_selector is None
        assert artifact.semantic_dict()["source_selector"] is None
        assert "source_selector" not in artifact.config.settings
        assert artifact.schema_version == "1.3"


class TestTheCommand:
    def test_the_cli_forwards_the_table(self, tmp_path: Path):
        from aidatasetkit.cli.main import main

        path = _write(
            tmp_path / "cli.sqlite",
            "CREATE TABLE a (n INTEGER); INSERT INTO a VALUES (1);"
            "CREATE TABLE b (n INTEGER); INSERT INTO b VALUES (2);",
        )
        output = tmp_path / "out"
        code = main(["audit", str(path), "--table", "b", "--output", str(output), "--fail-on", "never"])
        assert code == 0, output
        import json
        from aidatasetkit.evidence import read_current

        artifact = json.loads(read_current(output).text("audit.json"))
        assert artifact["source_selector"] == {"kind": "table", "name": "b"}
        assert artifact["ingestion"]["format"] == "sqlite"
        assert artifact["schema_version"] == "1.3"

    def test_chunked_profiling_refuses_sqlite(self, tmp_path: Path):
        from aidatasetkit.cli.main import main

        path = _people(tmp_path / "chunk.sqlite")
        code = main(["audit", str(path), "--chunked-profile", "--output", str(tmp_path / "out")])
        assert code != 0

    def test_help_mentions_sqlite(self):
        import argparse

        from aidatasetkit.cli.main import build_parser

        parser = build_parser()
        action = next(item for item in parser._actions if isinstance(item, argparse._SubParsersAction))
        help_text = action.choices["audit"].format_help()
        assert "--table" in help_text
        assert ".sqlite" in help_text
