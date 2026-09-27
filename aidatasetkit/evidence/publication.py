"""Publishing a run's artifacts as one set, and reading back only complete sets.

Writing ``audit.json``, ``lineage.json`` and ``report.html`` one after another
into a directory cannot be made safe file by file. ``os.replace`` makes each
*file* atomic, but a failure after the first leaves a new ``audit.json`` beside
an old ``report.html``, and two audits into the same directory interleave. A
reader then holds a set that no run ever produced. So the unit of publication
is the run, and there is exactly one commit point:

.. code-block:: text

    output/
      CURRENT                      <- the only thing a reader trusts first
      runs/<run_id>/
        audit.json  lineage.json  report.html
        manifest.json              <- run id, created_at, name/size/sha256 of each file
      .staging/<run_id>/           <- private to one writer until it is renamed

**The writer** creates a staging directory no other run can name, writes every
file in full with ``flush`` and ``fsync``, writes the manifest the same way,
re-reads every staged file against the manifest, renames the staging directory to
``runs/<run_id>``, and only then points ``CURRENT`` at it by writing a new
pointer beside it and ``os.replace``-ing it into place. ``CURRENT`` names the run
*and the sha256 of its manifest*. Every step before that replace can fail and the
previous ``CURRENT`` is untouched; the replace itself either happens or does not.

**The reader** reads ``CURRENT``, reads the manifest it names and checks its
digest, then checks that the run holds exactly the files the manifest lists, each
of the listed size and digest. Anything else -- a missing, extra, resized or
altered file, a manifest that is not the one ``CURRENT`` names, a pointer it
cannot parse -- is refused whole with :class:`CorruptPublicationError`. The bytes
it returns are the bytes it verified, held in memory, so nothing can change
between the check and the use.

**Concurrency is "the last complete run wins", without a lock.** Each writer has
its own staging directory and its own run directory, so two writers never touch
the same file until the final replace, and that replace swaps one complete
pointer for another. A lock would add a failure mode -- a writer killed while
holding it -- and remove none: there is no state a reader could see that the
lock would prevent. On Windows ``os.replace`` fails with ``PermissionError``
while another process has ``CURRENT`` open; the writer retries for a bounded
time and then fails, leaving the previous run current.

What is deliberately not done: finished runs are never deleted (a reader may be
holding one), and staging directories left by a killed writer are not swept
(one may belong to a writer that is still running). Neither is ever read.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aidatasetkit.core.exceptions import (
    CorruptPublicationError,
    NoPublishedRunError,
    PublicationError,
)

__all__ = [
    "CURRENT_NAME",
    "MANIFEST_NAME",
    "PUBLICATION_SCHEMA_VERSION",
    "PublishedRun",
    "publish_run",
    "read_current",
]

#: Version of the pointer-and-manifest layout, separate from the artifact schema.
PUBLICATION_SCHEMA_VERSION = "1.0"

CURRENT_NAME = "CURRENT"
MANIFEST_NAME = "manifest.json"
RUNS_DIRECTORY = "runs"
STAGING_DIRECTORY = ".staging"

#: ``20260927T101530123456Z-<32 hex>``: sortable by time, unique by UUID4.
_RUN_ID = re.compile(r"^\d{8}T\d{12}Z-[0-9a-f]{32}$")

#: A published file name: one path component, no separators, no leading dot.
_FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: How long a writer keeps retrying ``os.replace`` on ``PermissionError``, and a
#: reader keeps retrying opening ``CURRENT``. Windows reports a file that another
#: process holds open this way; the holder is a reader that is done in
#: milliseconds, so a few seconds separates contention from a real refusal.
_RETRY_SECONDS = 5.0
_RETRY_INTERVAL = 0.02


@dataclass(frozen=True)
class PublishedRun:
    """One complete, verified run.

    Attributes:
        run_id: The run's identifier, also the name of its directory.
        directory: ``runs/<run_id>``.
        manifest: The parsed manifest, as verified.
        contents: Every published file's bytes, exactly as verified against the
            manifest. Read these rather than re-opening the files: they are the
            ones that were checked.
    """

    run_id: str
    directory: Path
    manifest: dict[str, Any]
    contents: Mapping[str, bytes] = field(repr=False)

    @property
    def created_at(self) -> str:
        return str(self.manifest["created_at"])

    def path(self, name: str) -> Path:
        """Where a published file lives. For display; read :meth:`text` instead."""
        if name not in self.contents:
            raise KeyError(f"{name!r} is not part of run {self.run_id}.")
        return self.directory / name

    def text(self, name: str) -> str:
        """A published file's verified contents, decoded as UTF-8."""
        return self.contents[name].decode("utf-8")

    def json(self, name: str) -> Any:
        return json.loads(self.text(name))


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def publish_run(
    output: Path,
    files: Mapping[str, str | bytes],
    *,
    created_at: str | None = None,
) -> PublishedRun:
    """Publish ``files`` as one run under ``output`` and make it current.

    Args:
        output: The output directory. Created if missing.
        files: Published name to content. Text is encoded as UTF-8.
        created_at: Recorded in the manifest; defaults to now, in UTC.

    Returns:
        The run as a reader would see it.

    Raises:
        PublicationError: If the run could not be published. The run that was
            current before the call is still current, unchanged.
    """
    payload = _encoded(files)
    run_id = _new_run_id()
    output = Path(output)
    staging = output / STAGING_DIRECTORY / run_id
    final = output / RUNS_DIRECTORY / run_id

    try:
        (output / STAGING_DIRECTORY).mkdir(parents=True, exist_ok=True)
        (output / RUNS_DIRECTORY).mkdir(parents=True, exist_ok=True)
        staging.mkdir()

        entries: dict[str, dict[str, Any]] = {}
        for name, data in payload.items():
            _write_file(staging / name, data)
            entries[name] = {"bytes": len(data), "sha256": _sha256(data)}

        manifest = {
            "publication_schema_version": PUBLICATION_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": created_at or _utc_now(),
            "files": entries,
        }
        manifest_bytes = _canonical(manifest)
        _write_file(staging / MANIFEST_NAME, manifest_bytes)

        _verify_directory(staging, manifest, manifest_bytes)
        _fsync_directory(staging)
        _rename(staging, final)
        _fsync_directory(final.parent)
    except OSError as error:
        _discard(staging)
        raise PublicationError(
            f"Run {run_id} could not be written to {output}: {error}. Nothing was "
            "published; the previous run, if any, is still current."
        ) from error
    except CorruptPublicationError as error:
        _discard(staging)
        raise PublicationError(
            f"Run {run_id} did not match its own manifest after writing: {error} "
            "Nothing was published; the previous run, if any, is still current."
        ) from error

    pointer = _canonical(
        {
            "publication_schema_version": PUBLICATION_SCHEMA_VERSION,
            "run_id": run_id,
            "manifest_sha256": _sha256(manifest_bytes),
        }
    )
    _commit(output, run_id, pointer)
    return PublishedRun(
        run_id=run_id, directory=final, manifest=manifest, contents=dict(payload)
    )


def _commit(output: Path, run_id: str, pointer: bytes) -> None:
    """Swap ``CURRENT`` for a pointer to ``run_id``. The single commit point."""
    candidate = output / f"{CURRENT_NAME}.{run_id}.tmp"
    try:
        _write_file(candidate, pointer)
    except OSError as error:
        _remove(candidate)
        raise PublicationError(
            f"Run {run_id} was written but could not be made current: {error}. "
            "The previous run is still current."
        ) from error

    deadline = time.monotonic() + _RETRY_SECONDS
    while True:
        try:
            _replace(candidate, output / CURRENT_NAME)
            break
        except PermissionError as error:
            if time.monotonic() >= deadline:
                _remove(candidate)
                raise PublicationError(
                    f"Run {run_id} was written but {CURRENT_NAME} stayed locked "
                    f"for {_RETRY_SECONDS:g}s: {error}. The previous run is still "
                    "current."
                ) from error
            time.sleep(_RETRY_INTERVAL)
        except OSError as error:
            _remove(candidate)
            raise PublicationError(
                f"Run {run_id} was written but could not be made current: {error}. "
                "The previous run is still current."
            ) from error
    try:
        _fsync_directory(output)
    except OSError:
        # The replace has happened. A failed directory sync can lose it on a power
        # cut, which restores the previous complete run -- never a partial one.
        pass


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def read_current(output: Path) -> PublishedRun:
    """Return the current run of ``output``, verified whole, or refuse it.

    Raises:
        NoPublishedRunError: If ``output`` holds no ``CURRENT``.
        CorruptPublicationError: If anything about the current run disagrees with
            its manifest or its pointer.
    """
    output = Path(output)
    pointer_bytes = _read_pointer(output / CURRENT_NAME)
    pointer = _parse(pointer_bytes, CURRENT_NAME)
    run_id = pointer.get("run_id")
    expected_manifest = pointer.get("manifest_sha256")
    if not isinstance(run_id, str) or not _RUN_ID.match(run_id):
        raise CorruptPublicationError(
            f"{CURRENT_NAME} names {run_id!r}, which is not a run identifier."
        )
    if not isinstance(expected_manifest, str):
        raise CorruptPublicationError(f"{CURRENT_NAME} carries no manifest digest.")

    directory = output / RUNS_DIRECTORY / run_id
    try:
        manifest_bytes = (directory / MANIFEST_NAME).read_bytes()
    except OSError as error:
        raise CorruptPublicationError(
            f"{CURRENT_NAME} names run {run_id}, whose manifest cannot be read: {error}"
        ) from error
    if _sha256(manifest_bytes) != expected_manifest:
        raise CorruptPublicationError(
            f"The manifest of run {run_id} is not the one {CURRENT_NAME} names."
        )
    manifest = _parse(manifest_bytes, MANIFEST_NAME)
    if manifest.get("run_id") != run_id:
        raise CorruptPublicationError(
            f"The manifest in {directory} describes run {manifest.get('run_id')!r}."
        )
    contents = _verify_directory(directory, manifest, manifest_bytes)
    return PublishedRun(
        run_id=run_id, directory=directory, manifest=manifest, contents=contents
    )


def _read_pointer(path: Path) -> bytes:
    deadline = time.monotonic() + _RETRY_SECONDS
    while True:
        try:
            return path.read_bytes()
        except FileNotFoundError as error:
            raise NoPublishedRunError(
                f"{path.parent} holds no completed run: {CURRENT_NAME} is absent."
            ) from error
        except PermissionError as error:
            # Windows, while a writer's replace is in flight.
            if time.monotonic() >= deadline:
                raise CorruptPublicationError(
                    f"{path} stayed unreadable for {_RETRY_SECONDS:g}s: {error}"
                ) from error
            time.sleep(_RETRY_INTERVAL)
        except OSError as error:
            raise CorruptPublicationError(f"{path} cannot be read: {error}") from error


def _verify_directory(
    directory: Path, manifest: Mapping[str, Any], manifest_bytes: bytes
) -> dict[str, bytes]:
    """Check a run directory against its manifest and return what was checked."""
    entries = manifest.get("files")
    if not isinstance(entries, dict) or not entries:
        raise CorruptPublicationError(f"The manifest in {directory} lists no files.")
    try:
        present = {entry.name for entry in directory.iterdir()}
    except OSError as error:
        raise CorruptPublicationError(f"{directory} cannot be listed: {error}") from error
    expected = set(entries) | {MANIFEST_NAME}
    if present != expected:
        missing, extra = sorted(expected - present), sorted(present - expected)
        raise CorruptPublicationError(
            f"{directory} does not hold exactly the manifest's files "
            f"(missing {missing}, unexpected {extra})."
        )
    contents: dict[str, bytes] = {}
    for name, entry in entries.items():
        try:
            data = (directory / name).read_bytes()
        except OSError as error:
            raise CorruptPublicationError(f"{name} in {directory} cannot be read: {error}") from error
        if len(data) != entry.get("bytes"):
            raise CorruptPublicationError(
                f"{name} in {directory} is {len(data)} bytes; the manifest says "
                f"{entry.get('bytes')}."
            )
        if _sha256(data) != entry.get("sha256"):
            raise CorruptPublicationError(
                f"{name} in {directory} does not match the digest in its manifest."
            )
        contents[name] = data
    return contents


# --------------------------------------------------------------------------- #
# Primitives. Each is a separate function so that tests can make exactly one
# step fail and assert what a reader sees afterwards.
# --------------------------------------------------------------------------- #


def _write_file(path: Path, data: bytes) -> None:
    """Create ``path`` (it must not exist), write ``data`` in full, flush, fsync."""
    with open(path, "xb") as handle:
        handle.write(data)
        handle.flush()
        _fsync_file(handle.fileno())


def _fsync_file(descriptor: int) -> None:
    os.fsync(descriptor)


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry. POSIX only: Windows cannot open a directory."""
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename(source: Path, target: Path) -> None:
    os.rename(source, target)


def _replace(source: Path, target: Path) -> None:
    os.replace(source, target)


def _discard(staging: Path) -> None:
    """Best-effort removal of a staging directory that will never be renamed."""
    if not staging.exists():
        return
    for entry in staging.iterdir():
        _remove(entry)
    try:
        staging.rmdir()
    except OSError:
        pass


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _encoded(files: Mapping[str, str | bytes]) -> dict[str, bytes]:
    if not files:
        raise PublicationError("A run needs at least one file to publish.")
    encoded: dict[str, bytes] = {}
    for name, content in files.items():
        if not isinstance(name, str) or not _FILE_NAME.match(name) or name == MANIFEST_NAME:
            raise PublicationError(
                f"{name!r} cannot be published: a name must be a single path "
                f"component without a leading dot, and {MANIFEST_NAME!r} is reserved."
            )
        encoded[name] = content.encode("utf-8") if isinstance(content, str) else bytes(content)
    return encoded


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{uuid.uuid4().hex}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse(data: bytes, what: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise CorruptPublicationError(f"{what} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise CorruptPublicationError(f"{what} is not a JSON object.")
    return value
