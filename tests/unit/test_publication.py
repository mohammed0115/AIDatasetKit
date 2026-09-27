"""A reader sees the previous complete run or the new complete run. Nothing else.

Every test ends with the same question, asked by :func:`_generation`: read the
current run, check that every file in it came from the same publish, and say
which one. The invariant under test is that, after any failure at any step --
an exception, a killed process, a concurrent writer, a tampered file -- that
question has only two possible answers: the old generation, whole, or the new
generation, whole. A reader never sees a truncated file, a mixture of two
publishes, or a set that disagrees with its manifest; it gets a refusal instead.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from aidatasetkit.core.exceptions import (
    CorruptPublicationError,
    NoPublishedRunError,
    PublicationError,
)
from aidatasetkit.evidence import publication
from aidatasetkit.evidence.publication import (
    CURRENT_NAME,
    MANIFEST_NAME,
    publish_run,
    read_current,
)

NAMES = ("audit.json", "lineage.json", "report.html")
ROOT = Path(__file__).resolve().parents[2]


def _files(generation: str, size: int = 0) -> dict[str, str]:
    """Three files that all carry one generation marker, optionally padded."""
    pad = "x" * size
    return {
        "audit.json": json.dumps({"generation": generation, "pad": pad}),
        "lineage.json": json.dumps({"generation": generation}),
        "report.html": f"<p data-generation='{generation}'>— {pad}</p>",
    }


def _generation(output: Path) -> str | None:
    """The current generation, asserting every file agrees; ``None`` if none."""
    try:
        run = read_current(output)
    except NoPublishedRunError:
        return None
    audit = run.json("audit.json")["generation"]
    assert run.json("lineage.json")["generation"] == audit
    assert f"data-generation='{audit}'" in run.text("report.html")
    assert set(run.contents) == set(NAMES)
    return audit


def _no_debris(output: Path) -> None:
    """A failed in-process publish removes its staging files and pointer draft."""
    staging = output / publication.STAGING_DIRECTORY
    assert not staging.exists() or list(staging.iterdir()) == []
    assert not list(output.glob(f"{CURRENT_NAME}.*.tmp"))


class _FailOn:
    """Replace one primitive so that its ``n``-th call raises ``error``."""

    def __init__(self, monkeypatch, name: str, n: int, error: BaseException):
        self.calls = 0
        original = getattr(publication, name)

        def wrapper(*args, **kwargs):
            self.calls += 1
            if self.calls == n:
                raise error
            return original(*args, **kwargs)

        monkeypatch.setattr(publication, name, wrapper)


# --------------------------------------------------------------------------- #


class TestPublishing:
    def test_a_first_run_becomes_current(self, tmp_path):
        run = publish_run(tmp_path, _files("g1"))
        assert _generation(tmp_path) == "g1"
        assert read_current(tmp_path).run_id == run.run_id

    def test_a_second_run_replaces_it_and_keeps_the_first(self, tmp_path):
        first = publish_run(tmp_path, _files("g1"))
        second = publish_run(tmp_path, _files("g2"))
        assert _generation(tmp_path) == "g2"
        assert (tmp_path / "runs" / first.run_id / "audit.json").exists()
        assert first.run_id < second.run_id

    def test_the_manifest_describes_every_file(self, tmp_path):
        run = read_current(publish_run(tmp_path, _files("g1")).directory.parents[1])
        manifest = run.manifest
        assert manifest["publication_schema_version"] == "1.0"
        assert manifest["run_id"] == run.run_id
        assert set(manifest["files"]) == set(NAMES)
        for name, entry in manifest["files"].items():
            assert entry["bytes"] == len(run.contents[name])

    def test_created_at_is_recorded(self, tmp_path):
        publish_run(tmp_path, _files("g1"), created_at="2025-02-01T00:00:00Z")
        assert read_current(tmp_path).created_at == "2025-02-01T00:00:00Z"

    def test_text_is_utf8(self, tmp_path):
        publish_run(tmp_path, _files("g1"))
        assert "—" in read_current(tmp_path).text("report.html")

    def test_no_run_yet_is_its_own_error(self, tmp_path):
        with pytest.raises(NoPublishedRunError):
            read_current(tmp_path)

    def test_every_run_id_is_distinct(self, tmp_path):
        ids = {publish_run(tmp_path, _files(f"g{i}")).run_id for i in range(20)}
        assert len(ids) == 20

    @pytest.mark.parametrize(
        "name", ["../escape", "a/b", "a\\b", ".hidden", "", MANIFEST_NAME, 7]
    )
    def test_a_name_that_is_not_one_plain_file_is_refused(self, tmp_path, name):
        with pytest.raises(PublicationError):
            publish_run(tmp_path, {name: "x"})
        assert not (tmp_path / CURRENT_NAME).exists()

    def test_an_empty_set_is_refused(self, tmp_path):
        with pytest.raises(PublicationError):
            publish_run(tmp_path, {})


class TestFailureInjection:
    """Each step fails once, first with no previous run and then with one."""

    #: Three files, the manifest, then the pointer draft.
    WRITES = 5

    @pytest.fixture(params=[False, True], ids=["first_run", "with_previous"])
    @staticmethod
    def output(request, tmp_path):
        if request.param:
            publish_run(tmp_path, _files("old"))
        return tmp_path

    @staticmethod
    def _before(output):
        return _generation(output)

    @pytest.mark.parametrize("n", range(1, WRITES + 1))
    def test_a_failed_write(self, monkeypatch, output, n):
        before = self._before(output)
        _FailOn(monkeypatch, "_write_file", n, OSError("injected write failure"))
        with pytest.raises(PublicationError):
            publish_run(output, _files("new"))
        assert _generation(output) == before
        _no_debris(output)

    @pytest.mark.parametrize("n", range(1, WRITES + 1))
    def test_a_failed_fsync(self, monkeypatch, output, n):
        before = self._before(output)
        _FailOn(monkeypatch, "_fsync_file", n, OSError("injected fsync failure"))
        with pytest.raises(PublicationError):
            publish_run(output, _files("new"))
        assert _generation(output) == before
        _no_debris(output)

    def test_a_failed_directory_sync_before_the_move(self, monkeypatch, output):
        before = self._before(output)
        _FailOn(monkeypatch, "_fsync_directory", 1, OSError("injected dir sync failure"))
        with pytest.raises(PublicationError):
            publish_run(output, _files("new"))
        assert _generation(output) == before
        _no_debris(output)

    def test_a_failed_move_out_of_staging(self, monkeypatch, output):
        before = self._before(output)
        _FailOn(monkeypatch, "_rename", 1, OSError("injected rename failure"))
        with pytest.raises(PublicationError):
            publish_run(output, _files("new"))
        assert _generation(output) == before
        _no_debris(output)

    def test_a_failed_replace_of_current(self, monkeypatch, output):
        before = self._before(output)
        _FailOn(monkeypatch, "_replace", 1, OSError("injected replace failure"))
        with pytest.raises(PublicationError):
            publish_run(output, _files("new"))
        assert _generation(output) == before
        _no_debris(output)

    def test_a_failed_sync_after_the_replace_still_publishes(self, monkeypatch, output):
        """The replace is the commit; a sync failure after it cannot undo it."""
        calls = {"n": 0}
        original = publication._fsync_directory

        def fail_last(path):
            calls["n"] += 1
            if Path(path) == Path(output):
                raise OSError("injected late sync failure")
            return original(path)

        monkeypatch.setattr(publication, "_fsync_directory", fail_last)
        publish_run(output, _files("new"))
        assert _generation(output) == "new"

    def test_a_staged_file_that_does_not_match_is_never_published(self, monkeypatch, output):
        """Verification before the move: the bytes on disk must be the bytes meant."""
        before = self._before(output)
        original = publication._write_file

        def corrupting(path, data):
            if Path(path).name == "lineage.json":
                data = data[:-1] + b"!"
            return original(path, data)

        monkeypatch.setattr(publication, "_write_file", corrupting)
        with pytest.raises(PublicationError, match="did not match its own manifest"):
            publish_run(output, _files("new"))
        assert _generation(output) == before
        _no_debris(output)


class TestWindowsContention:
    def test_a_briefly_locked_current_is_retried(self, monkeypatch, tmp_path):
        publish_run(tmp_path, _files("old"))
        state = {"left": 3}
        original = publication._replace

        def locked_three_times(source, target):
            if state["left"]:
                state["left"] -= 1
                raise PermissionError(13, "The process cannot access the file")
            return original(source, target)

        monkeypatch.setattr(publication, "_replace", locked_three_times)
        publish_run(tmp_path, _files("new"))
        assert _generation(tmp_path) == "new"

    def test_a_current_that_stays_locked_fails_cleanly(self, monkeypatch, tmp_path):
        publish_run(tmp_path, _files("old"))
        monkeypatch.setattr(publication, "_RETRY_SECONDS", 0.2)

        def always_locked(source, target):
            raise PermissionError(13, "The process cannot access the file")

        monkeypatch.setattr(publication, "_replace", always_locked)
        with pytest.raises(PublicationError, match="stayed locked"):
            publish_run(tmp_path, _files("new"))
        assert _generation(tmp_path) == "old"
        _no_debris(tmp_path)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows file-sharing semantics")
    def test_a_reader_really_holding_current_open(self, tmp_path):
        """Not simulated: this process opens CURRENT, Windows refuses the replace."""
        publish_run(tmp_path, _files("old"))
        handle = open(tmp_path / CURRENT_NAME, "rb")
        released = threading.Event()

        def release_later():
            time.sleep(0.4)
            handle.close()
            released.set()

        threading.Thread(target=release_later).start()
        started = time.monotonic()
        publish_run(tmp_path, _files("new"))
        assert released.is_set(), "the publish should have waited for the reader"
        assert time.monotonic() - started >= 0.35
        assert _generation(tmp_path) == "new"


class TestTampering:
    @pytest.fixture
    @staticmethod
    def published(tmp_path):
        run = publish_run(tmp_path, _files("g1"))
        return tmp_path, run.directory

    def test_an_altered_byte_is_refused(self, published):
        output, directory = published
        path = directory / "lineage.json"
        data = bytearray(path.read_bytes())
        data[2] ^= 0x01
        path.write_bytes(bytes(data))
        with pytest.raises(CorruptPublicationError, match="digest"):
            read_current(output)

    def test_a_truncated_file_is_refused(self, published):
        output, directory = published
        path = directory / "audit.json"
        path.write_bytes(path.read_bytes()[:5])
        with pytest.raises(CorruptPublicationError, match="bytes"):
            read_current(output)

    def test_a_missing_file_is_refused(self, published):
        output, directory = published
        (directory / "report.html").unlink()
        with pytest.raises(CorruptPublicationError, match="missing"):
            read_current(output)

    def test_an_extra_file_is_refused(self, published):
        output, directory = published
        (directory / "extra.txt").write_text("x", encoding="utf-8")
        with pytest.raises(CorruptPublicationError, match="unexpected"):
            read_current(output)

    def test_an_edited_manifest_is_refused(self, published):
        output, directory = published
        path = directory / MANIFEST_NAME
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["created_at"] = "1999-01-01T00:00:00Z"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(CorruptPublicationError, match="not the one"):
            read_current(output)

    def test_a_file_and_manifest_forged_together_are_refused(self, published):
        """Recomputing the manifest after editing a file is caught by the pointer."""
        output, directory = published
        (directory / "audit.json").write_text('{"generation": "forged"}', encoding="utf-8")
        manifest_path = directory / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        forged = (directory / "audit.json").read_bytes()
        manifest["files"]["audit.json"] = {
            "bytes": len(forged),
            "sha256": publication._sha256(forged),
        }
        manifest_path.write_bytes(publication._canonical(manifest))
        with pytest.raises(CorruptPublicationError, match="not the one"):
            read_current(output)

    @pytest.mark.parametrize(
        "pointer",
        [b"", b"not json", b"[]", b'{"run_id": "../../outside", "manifest_sha256": "0"}',
         b'{"run_id": "20260101T000000000000Z-' + b"0" * 32 + b'", "manifest_sha256": "0"}',
         b'{"manifest_sha256": "0"}'],
        ids=["empty", "garbage", "array", "traversal", "absent_run", "no_run_id"],
    )
    def test_an_invalid_current_is_refused(self, published, pointer):
        output, _ = published
        (output / CURRENT_NAME).write_bytes(pointer)
        with pytest.raises(CorruptPublicationError):
            read_current(output)

    def test_a_pointer_without_a_digest_is_refused(self, published):
        output, directory = published
        (output / CURRENT_NAME).write_text(
            json.dumps({"run_id": directory.name}), encoding="utf-8"
        )
        with pytest.raises(CorruptPublicationError, match="digest"):
            read_current(output)

    def test_a_run_moved_under_another_name_is_refused(self, published):
        output, directory = published
        pointer = json.loads((output / CURRENT_NAME).read_text(encoding="utf-8"))
        other = "20000101T000000000000Z-" + "a" * 32
        directory.rename(directory.parent / other)
        pointer["run_id"] = other
        (output / CURRENT_NAME).write_text(json.dumps(pointer), encoding="utf-8")
        with pytest.raises(CorruptPublicationError, match="describes run"):
            read_current(output)

    def test_what_is_returned_is_what_was_verified(self, published):
        """Changing the file after reading cannot change the verified contents."""
        output, directory = published
        run = read_current(output)
        (directory / "audit.json").write_text("changed", encoding="utf-8")
        assert run.json("audit.json")["generation"] == "g1"


class TestLeftovers:
    def test_a_stale_staging_directory_is_ignored(self, tmp_path):
        publish_run(tmp_path, _files("g1"))
        stale = tmp_path / publication.STAGING_DIRECTORY / ("20000101T000000000000Z-" + "b" * 32)
        stale.mkdir()
        (stale / "audit.json").write_text('{"generation": "half', encoding="utf-8")
        assert _generation(tmp_path) == "g1"
        publish_run(tmp_path, _files("g2"))
        assert _generation(tmp_path) == "g2"
        assert stale.exists(), "another writer's staging is never deleted"

    def test_a_leftover_pointer_draft_is_ignored(self, tmp_path):
        publish_run(tmp_path, _files("g1"))
        (tmp_path / f"{CURRENT_NAME}.junk.tmp").write_text("{", encoding="utf-8")
        assert _generation(tmp_path) == "g1"

    def test_a_complete_run_nobody_pointed_at_is_ignored(self, monkeypatch, tmp_path):
        publish_run(tmp_path, _files("g1"))
        _FailOn(monkeypatch, "_replace", 1, OSError("injected"))
        with pytest.raises(PublicationError):
            publish_run(tmp_path, _files("orphan"))
        assert len(list((tmp_path / "runs").iterdir())) == 2
        assert _generation(tmp_path) == "g1"


# --------------------------------------------------------------------------- #
# Real processes
# --------------------------------------------------------------------------- #

_CHILD = """
import os, sys, json
from pathlib import Path
from aidatasetkit.evidence import publication
output, mode, generation, size = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
pad = "x" * size
files = {
    "audit.json": json.dumps({"generation": generation, "pad": pad}),
    "lineage.json": json.dumps({"generation": generation}),
    "report.html": f"<p data-generation='{generation}'>— {pad}</p>",
}
if mode == "die_mid_write":
    original, calls = publication._write_file, []
    def dying(path, data):
        calls.append(path)
        if len(calls) == 2:
            with open(path, "xb") as handle:
                handle.write(data[: len(data) // 2])
            os._exit(3)
        return original(path, data)
    publication._write_file = dying
elif mode == "die_before_commit":
    def dying(source, target):
        os._exit(3)
    publication._replace = dying
if mode == "loop":
    for i in range(10**6):
        publication.publish_run(output, {k: v.replace(generation, f"{generation}-{i}") for k, v in files.items()})
else:
    publication.publish_run(output, files)
"""


def _child(output: Path, mode: str, generation: str, size: int = 0) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(output), mode, generation, str(size)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


class TestProcessesThatDie:
    def test_a_process_killed_mid_write(self, tmp_path):
        publish_run(tmp_path, _files("old"))
        child = _child(tmp_path, "die_mid_write", "new", size=100_000)
        assert child.wait(timeout=120) == 3, child.stderr.read().decode()
        assert _generation(tmp_path) == "old"

    def test_a_process_killed_before_the_commit(self, tmp_path):
        publish_run(tmp_path, _files("old"))
        child = _child(tmp_path, "die_before_commit", "new")
        assert child.wait(timeout=120) == 3, child.stderr.read().decode()
        assert _generation(tmp_path) == "old"
        # The finished-but-uncommitted run exists and is ignored.
        assert len(list((tmp_path / "runs").iterdir())) == 2

    @pytest.mark.parametrize("delay", [0.9, 1.4, 2.1, 2.8])
    def test_a_process_killed_at_an_arbitrary_moment(self, tmp_path, delay):
        """SIGKILL / TerminateProcess while publishing in a loop, 300 KB per file."""
        publish_run(tmp_path, _files("seed"))
        child = _child(tmp_path, "loop", "k", size=300_000)
        time.sleep(delay)
        child.kill()
        child.wait(timeout=60)
        generation = _generation(tmp_path)
        assert generation == "seed" or generation.startswith("k-")


class TestConcurrentWriters:
    def test_writers_racing_never_produce_a_mixed_set(self, tmp_path):
        """Four processes publish into one directory while this one keeps reading."""
        publish_run(tmp_path, _files("seed"))
        writers = [_child(tmp_path, "loop", f"w{i}", size=20_000) for i in range(4)]
        seen: set[str | None] = set()
        reads = 0

        def writers_seen() -> set[str]:
            return {g.split("-")[0] for g in seen if g and g != "seed"}

        # Read until at least three writers have each become current at least
        # once -- proof they overlapped -- and then for a further stretch, with a
        # ceiling generous enough for slow interpreter start-up on a loaded box.
        deadline = time.monotonic() + 120.0
        settle: float | None = None
        try:
            while time.monotonic() < deadline:
                seen.add(_generation(tmp_path))
                reads += 1
                if settle is None and len(writers_seen()) >= 3:
                    settle = time.monotonic() + 3.0
                if settle is not None and time.monotonic() > settle:
                    break
        finally:
            for writer in writers:
                writer.kill()
                writer.wait(timeout=60)
        final = _generation(tmp_path)
        assert final is not None
        assert len(writers_seen()) >= 3, f"the writers never overlapped: {seen}"
        assert reads > 50
        # Every committed run is complete in its own right, not only the current one.
        for directory in (tmp_path / "runs").iterdir():
            manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
            assert set(manifest["files"]) == set(NAMES)
            for name, entry in manifest["files"].items():
                assert publication._sha256((directory / name).read_bytes()) == entry["sha256"]
