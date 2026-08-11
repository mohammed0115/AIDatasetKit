"""Canonical JSON for audit artifacts.

Two rules make this module different from
:func:`aidatasetkit.core.types.jsonable`, and both matter enough to justify not
reusing it.

**Nothing is coerced by ``repr``.** ``jsonable`` renders an unrecognised object
as ``repr(value)``, which is right for provenance -- a model parameter holding an
estimator should not abort a training run -- and wrong here. An audit artifact
that silently absorbs ``DecisionTreeClassifier(random_state=42)`` as a string, or
a numpy array as ``"array([...])"``, has leaked an object into a document that is
meant to be a stable, machine-readable record. Anything not explicitly allowed
raises :class:`~aidatasetkit.core.exceptions.SerializationError` instead.

**Ordering is fixed.** Mappings are emitted with sorted keys and floats are
normalised, so two runs that mean the same thing produce byte-identical JSON.
That is what makes an artifact diffable in git and hashable into a fingerprint.
"""

from __future__ import annotations

import datetime as _datetime
import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from aidatasetkit.core.exceptions import SerializationError

__all__ = [
    "canonical",
    "canonical_json",
    "label_token",
    "type_name",
]

#: Types that may appear verbatim in an artifact.
_SCALARS = (bool, int, float, str)

#: Attributes that identify an object we must never serialise, checked before the
#: generic fallbacks so the error names the real problem.
_FORBIDDEN_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    ("fit", "a fitted or unfitted estimator/transformer"),
    ("to_numpy", "a pandas Series or DataFrame"),
    ("__array__", "a numpy array"),
)


def type_name(value: Any) -> str:
    """Return the plain type name of ``value``, for label disambiguation."""
    return type(value).__name__


def label_token(label: Any) -> str:
    """Return a collision-free text form of a column label.

    ``0`` and ``"0"`` are different columns and a naive ``str`` makes them the
    same one. The type prefix keeps them apart wherever a label has to become
    text -- a JSON object key, a fingerprint input -- without anyone downstream
    having to remember the distinction.

    Examples:
        >>> label_token("age")
        'str:age'
        >>> label_token(0)
        'int:0'
    """
    return f"{type_name(label)}:{label}"


def canonical(value: Any, *, path: str = "$") -> Any:
    """Return ``value`` as JSON-safe data, refusing anything that could leak.

    Args:
        value: The value to convert.
        path: Where this value sits in the artifact, used to make an error
            actionable rather than merely refusing.

    Returns:
        ``None``, a bool, int, float, str, list, or dict of the same.

    Raises:
        SerializationError: If ``value`` is an object an audit artifact must not
            contain, or a type this module does not know how to represent.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Enum):
        return canonical(value.value, path=path)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _canonical_float(float(value), path)
    if isinstance(value, (_datetime.datetime, _datetime.date)):
        return value.isoformat()

    # Before the refusals: a numpy scalar exposes __array__ like an array does,
    # and it is a number. Zero dimensions is what separates the two, and a value
    # with no dimensions cannot be hiding a dataset.
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "ndim", None) == 0:
        return canonical(item(), path=path)

    for attribute, description in _FORBIDDEN_ATTRIBUTES:
        if hasattr(value, attribute):
            raise SerializationError(
                f"{path} holds {description} ({type_name(value)}). Audit artifacts "
                "record facts about a run, not the objects that produced it. Record "
                "the values you need from it instead."
            )

    if callable(value):
        raise SerializationError(
            f"{path} holds a callable ({type_name(value)}), which has no meaning in a "
            "stored artifact. Record what it did, not the function itself."
        )

    if isinstance(value, Mapping):
        return {
            str(key): canonical(item, path=f"{path}.{key}")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return [canonical(item, path=f"{path}[]") for item in sorted(value, key=str)]
    if isinstance(value, Sequence):
        return [
            canonical(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        ]

    raise SerializationError(
        f"{path} holds a {type_name(value)}, which this module has no canonical form "
        "for. Convert it to a number, a string, or a mapping before recording it."
    )


def _canonical_float(value: float, path: str) -> float | None:
    """Normalise a float, refusing the two values JSON cannot represent.

    ``json.dumps`` writes ``NaN`` and ``Infinity``, which are not JSON and which
    other parsers reject. A missing measurement is recorded as ``null``; an
    infinity is a real value that has to be named rather than smuggled.
    """
    if math.isnan(value):
        return None
    if math.isinf(value):
        raise SerializationError(
            f"{path} is {'+' if value > 0 else '-'}infinity, which JSON cannot "
            "represent. Record it as a string or a count instead."
        )
    return value


def canonical_json(value: Any, *, indent: int | None = 2) -> str:
    """Return the canonical JSON text for ``value``.

    Keys are sorted and separators are fixed, so the same evidence always
    produces the same bytes -- which is what lets an artifact be diffed in git
    and hashed into a fingerprint.
    """
    return json.dumps(
        canonical(value),
        indent=indent,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ": ") if indent is not None else (",", ":"),
    )
