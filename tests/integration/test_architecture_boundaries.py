"""Executable enforcement of the layered architecture.

The dependency direction agreed for this library is a rule, not a convention, so
it is checked by parsing every module's imports rather than by review. A module
may import from a strictly lower layer and from its own package. Anything else --
an upward import, a sideways import between two layer-3 packages, a cycle -- is a
failure here.

This test is written to grow: each new package is added to ``LAYERS`` when it is
created, and the rule applies to it from that moment on.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

import aidatasetkit

PACKAGE_ROOT = Path(aidatasetkit.__file__).parent
PACKAGE_NAME = aidatasetkit.__name__

#: Layer index of every top-level module or subpackage.
#:
#: A module may only import from a package with a strictly smaller index. Equal
#: indexes are forbidden precisely because that is where accidental coupling
#: appears: ``models`` must not reach into ``preprocessing``, and ``evaluation``
#: must not reach into ``training``.
LAYERS: dict[str, int] = {
    "core": 0,
    "statistics": 1,
    "datasets": 1,
    "profiling": 2,
    "preprocessing": 3,
    "models": 3,
    "training": 4,
    "evaluation": 4,
    "prediction": 4,
    "facade": 5,
}


def _module_files() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _component_of(path: Path) -> str:
    """Return the top-level package or module name a file belongs to."""
    relative = path.relative_to(PACKAGE_ROOT)
    return relative.parts[0] if len(relative.parts) > 1 else relative.stem


def _internal_imports(path: Path) -> set[str]:
    """Return the ``aidatasetkit`` components that ``path`` imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(f"{PACKAGE_NAME}."):
                    imported.add(alias.name.split(".")[1])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise AssertionError(
                    f"{path} uses a relative import; the project uses absolute imports only."
                )
            if node.module and node.module.startswith(f"{PACKAGE_NAME}."):
                imported.add(node.module.split(".")[1])

    return imported


def _layer_of(component: str) -> int:
    if component not in LAYERS:
        raise AssertionError(
            f"Component {component!r} has no declared layer. Add it to LAYERS in this test "
            "as a deliberate architectural decision."
        )
    return LAYERS[component]


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: p.name)
def test_module_belongs_to_a_declared_layer(path: Path) -> None:
    component = _component_of(path)
    if component == "__init__":
        return
    _layer_of(component)


def test_no_module_imports_from_its_own_layer_or_above() -> None:
    violations: list[str] = []

    for path in _module_files():
        component = _component_of(path)
        if component == "__init__":
            continue
        source_layer = _layer_of(component)

        for imported in _internal_imports(path):
            if imported == component:
                continue
            if _layer_of(imported) >= source_layer:
                violations.append(
                    f"{path.relative_to(PACKAGE_ROOT)} (layer {source_layer}) "
                    f"imports {imported} (layer {_layer_of(imported)})"
                )

    assert not violations, "Dependency direction violated:\n  " + "\n  ".join(violations)


def test_the_package_root_is_the_only_module_that_may_import_everything() -> None:
    """``aidatasetkit/__init__.py`` is the assembly point and is exempt by design."""
    root_init = PACKAGE_ROOT / "__init__.py"
    assert root_init.exists()
    assert _internal_imports(root_init) <= set(LAYERS)


def test_core_depends_on_nothing_internal_except_core() -> None:
    for path in (PACKAGE_ROOT / "core").glob("*.py"):
        assert _internal_imports(path) <= {"core"}, f"{path.name} reaches outside core"


def test_core_has_no_internal_import_cycles() -> None:
    graph: dict[str, set[str]] = defaultdict(set)

    for path in (PACKAGE_ROOT / "core").glob("*.py"):
        if path.stem == "__init__":
            continue
        source = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(source):
            if isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                if len(parts) >= 3 and parts[:2] == [PACKAGE_NAME, "core"]:
                    graph[path.stem].add(parts[2])

    visiting: set[str] = set()
    settled: set[str] = set()

    def walk(module: str, trail: tuple[str, ...]) -> None:
        if module in settled:
            return
        if module in visiting:
            raise AssertionError(f"Import cycle in core: {' -> '.join(trail + (module,))}")
        visiting.add(module)
        for dependency in sorted(graph[module]):
            walk(dependency, trail + (module,))
        visiting.discard(module)
        settled.add(module)

    for module in sorted(graph):
        walk(module, ())
