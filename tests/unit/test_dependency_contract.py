"""What pyproject.toml declares is exactly what was run.

The floors in ``pyproject.toml`` are a claim -- "this works on these versions" --
and ``constraints/minimum.txt`` is the environment the claim was tested in. If
they drift apart, one of them is false. The same holds for the reference set and
the verified-environment table in the release notes.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)\s*(.*)$")


def _pins(name: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in (ROOT / "constraints" / name).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        package, _, version = line.partition("==")
        assert version, f"{name}: {line!r} is not an exact pin"
        pins[package.lower()] = version
    return pins


def _declared() -> dict[str, str]:
    """Every dependency floor in pyproject.toml, core and extras, by package."""
    with open(ROOT / "pyproject.toml", "rb") as handle:
        project = tomllib.load(handle)["project"]
    requirements = list(project["dependencies"])
    for group, items in project["optional-dependencies"].items():
        requirements += [item for item in items if not item.startswith("aidatasetkit")]
    floors: dict[str, str] = {}
    for requirement in requirements:
        package, spec = _REQUIREMENT.match(requirement).groups()
        floors[package.lower()] = spec
    return floors


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


class TestTheFloorsAreTheTestedMinimum:
    @pytest.mark.parametrize("package", sorted(_pins("minimum.txt")))
    def test_each_pinned_package_is_declared_with_exactly_that_floor(self, package):
        assert _declared()[package] == f">={_pins('minimum.txt')[package]}"

    def test_every_library_dependency_is_pinned_in_the_minimum(self):
        """build and twine are release tools, exercised by the smoke test."""
        tools = {"build", "twine"}
        assert set(_declared()) - tools == set(_pins("minimum.txt"))

    def test_there_are_no_upper_bounds(self):
        for package, spec in _declared().items():
            assert "<" not in spec and "!=" not in spec and "~=" not in spec, package


class TestTheReferenceSet:
    def test_it_pins_the_same_packages(self):
        assert set(_pins("reference.txt")) == set(_pins("minimum.txt"))

    @pytest.mark.parametrize("package", sorted(_pins("reference.txt")))
    def test_it_is_not_older_than_the_minimum(self, package):
        assert _version(_pins("reference.txt")[package]) >= _version(
            _pins("minimum.txt")[package]
        )

    @pytest.mark.parametrize("package", ["numpy", "pandas", "scipy", "scikit-learn"])
    def test_the_release_notes_name_both_sets(self, package):
        notes = (ROOT / "RELEASE_NOTES_0.1.0a1.md").read_text(encoding="utf-8")
        assert _pins("minimum.txt")[package] in notes
        assert _pins("reference.txt")[package] in notes

    def test_the_reference_fixture_is_generated_on_the_reference_pandas_major(self):
        from tests.golden import REFERENCE_PANDAS_MAJOR

        assert _version(_pins("reference.txt")["pandas"])[0] == REFERENCE_PANDAS_MAJOR
        assert _version(_pins("minimum.txt")["pandas"])[0] == 2
