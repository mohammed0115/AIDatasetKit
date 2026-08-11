"""Identity for a dataset, a configuration, and a run.

A fingerprint answers one question -- *is this the same thing I audited before?*
-- without storing the thing. That constraint is what shapes every decision here:
no dataset values are kept, only a digest of them, and the digest is computed in
a way that stays linear in the size of the frame.

Dataset identity is defined precisely, because a rule left implicit is a rule
nobody can rely on:

=========================  ==========================================
Change                     Same fingerprint?
=========================  ==========================================
A cell value               **No**
A dtype, values unchanged  **No** -- ``1`` and ``1.0`` are not the same
A value's *type* inside
an ``object`` column       **No** -- ``1`` and ``"1"`` are not the same
Column order               **No**
Row order                  **No**
A column label             **No**
Row count                  **No**
The index                  **Yes** -- an index is addressing, not data
Column *labels* only cased differently  **No**
=========================  ==========================================

Row and column order both count, and the reason is the same in each case: a
frame is an ordered table, reordering it produces a different artifact for
anything position-dependent, and pretending otherwise would let a reordered file
claim an audit it never had. A caller who wants order-insensitive comparison can
sort before auditing; the reverse -- recovering order sensitivity from a
fingerprint that threw it away -- is impossible.

The index is deliberately excluded. Reading the same CSV with and without
``index_col`` should not change what the data *is*.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import pandas as pd

from aidatasetkit.evidence.serialization import canonical_json, label_token

__all__ = [
    "ALGORITHM",
    "config_fingerprint",
    "dataset_fingerprint",
    "digest_of",
    "schema_fingerprint",
]

#: Named in the artifact so a future version can change it without ambiguity.
ALGORITHM = "sha256/pandas-hash-v1"

#: How many rows are hashed at a time. Bounded so a very tall frame never builds
#: a large intermediate array, while staying big enough that the per-chunk
#: overhead is irrelevant.
_CHUNK_ROWS = 100_000

_PREFIX_LENGTH = 16


def digest_of(*parts: str) -> str:
    """Return a stable digest over ``parts``, length-delimited.

    Length prefixes rather than a separator: joining ``"ab"`` and ``"c"`` with a
    separator collides with joining ``"a"`` and ``"bc"`` for some separator
    choice, and identity is the one place where a collision cannot be shrugged
    off.
    """
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
    return digest.hexdigest()


def schema_fingerprint(frame: pd.DataFrame) -> str:
    """Return a digest of the frame's shape and column declarations.

    Values are not read. Two frames sharing a schema fingerprint hold the same
    columns, in the same order, with the same dtypes and the same number of rows.
    """
    parts = [f"rows={len(frame)}", f"columns={frame.shape[1]}"]
    for position, label in enumerate(frame.columns):
        parts.append(f"{position}|{label_token(label)}|{frame[label].dtype!s}")
    return digest_of(*parts)


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """Return a digest identifying this frame's schema *and* its contents.

    Values are hashed column by column with pandas' own row hasher, in chunks, so
    the work is linear in the number of cells and no copy of the frame is ever
    built. The per-row digests are folded into one running SHA-256, which keeps
    the result independent of chunk size while remaining sensitive to row order.

    Args:
        frame: The frame to identify. It is read, never modified.

    Returns:
        A hex digest. Prefix it with :data:`ALGORITHM` when recording it.
    """
    digest = hashlib.sha256()
    digest.update(schema_fingerprint(frame).encode("ascii"))

    row_count = len(frame)
    for label in frame.columns:
        column = frame[label]
        is_object = column.dtype == object
        digest.update(b"\x00")
        digest.update(label_token(label).encode("utf-8"))
        digest.update(b"\x00")
        for start in range(0, max(row_count, 1), _CHUNK_ROWS):
            chunk = column.iloc[start : start + _CHUNK_ROWS]
            if chunk.empty:
                continue
            # index=False: an index is addressing, not data. Without it, reading
            # the same file with and without index_col would change identity.
            hashed = pd.util.hash_pandas_object(chunk, index=False)
            digest.update(hashed.to_numpy(dtype="uint64").tobytes())
            if is_object:
                # pandas hashes an object column by each value's text, so the
                # integer 1 and the string "1" land on the same digest -- and so
                # do True and "True". They are different data and a fingerprint
                # that conflates them answers the one question it exists for
                # wrongly. Hashing the per-element type names alongside the
                # values separates them, and costs nothing on the typed columns
                # that make up almost every frame.
                types = "\x1f".join(type(value).__name__ for value in chunk)
                digest.update(types.encode("utf-8"))
    return digest.hexdigest()


def config_fingerprint(settings: Mapping[str, Any]) -> str:
    """Return a digest of everything that steered the run.

    The input is canonicalised first, so the digest depends on the *meaning* of
    the settings rather than on dict ordering or on any object's ``repr``. A
    setting that cannot be canonicalised raises rather than being hashed as text
    nobody can interpret later.
    """
    return hashlib.sha256(canonical_json(settings, indent=None).encode("utf-8")).hexdigest()


def short(fingerprint: str) -> str:
    """Return the readable prefix of a digest, for terminal and report display."""
    return fingerprint[:_PREFIX_LENGTH]
