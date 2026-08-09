"""Environment capture for reproducibility.

This is deliberately not experiment tracking. It answers exactly one question --
"which library versions produced this result?" -- and nothing more. The captured
record is a plain frozen dataclass that serialises to JSON-compatible types.
"""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

__all__ = ["EnvironmentVersions", "capture_environment", "package_version"]

_UNKNOWN = "unknown"

#: Distributions recorded on every run, mapped to their attribute name.
_TRACKED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("aidatasetkit", "aidatasetkit"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "scikit_learn"),
)


def package_version(distribution: str) -> str:
    """Return the installed version of ``distribution``, or ``"unknown"``.

    Never raises: a missing distribution is a reporting gap, not a failure that
    should abort a training run.
    """
    try:
        return version(distribution)
    except PackageNotFoundError:
        return _UNKNOWN
    except Exception:  # pragma: no cover - defensive, metadata backends vary
        return _UNKNOWN


@dataclass(frozen=True, slots=True)
class EnvironmentVersions:
    """Versions of the library and its numerical stack."""

    aidatasetkit: str
    python: str
    numpy: str
    pandas: str
    scipy: str
    scikit_learn: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable mapping of the captured versions."""
        return asdict(self)


@lru_cache(maxsize=1)
def capture_environment() -> EnvironmentVersions:
    """Capture the current environment versions.

    Cached, because versions cannot change within a process and this is called
    once per training run and once per evaluation report.
    """
    versions = {attr: package_version(dist) for dist, attr in _TRACKED_PACKAGES}
    return EnvironmentVersions(python=platform.python_version(), **versions)
