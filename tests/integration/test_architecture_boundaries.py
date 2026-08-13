"""Executable enforcement of the layered architecture.

The rule this file protects is *dependency direction*, not module isolation.
Modules inside one bounded package are expected to collaborate --
``statistics.engine`` builds on ``statistics.moments``, ``evaluation.comparison``
builds on ``evaluation.metrics`` -- and nothing here discourages that. What is
forbidden is a package reaching upward into a layer that is meant to depend on
it, and any import cycle at all.

Three checks implement that:

1. Every module belongs to a package with a declared layer.
2. A cross-package import must target a layer at or below the importer's.
   Same-package imports are always allowed.
3. The module-level import graph is acyclic.

The test grows with the library: a new package is added to ``LAYERS`` when it is
created, which makes the layer assignment a deliberate decision rather than an
accident.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

import aidatasetkit

PACKAGE_ROOT = Path(aidatasetkit.__file__).parent
PACKAGE_NAME = aidatasetkit.__name__

#: Layer index of every top-level package or module.
#:
#: A package may import from its own layer or below, never above. Equal-layer
#: imports across packages are permitted because they are sometimes the honest
#: design -- contorting the structure purely to avoid them produces worse code
#: than allowing them and relying on the cycle check.
LAYERS: dict[str, int] = {
    "core": 0,
    "statistics": 1,
    "datasets": 1,
    "profiling": 2,
    "preprocessing": 3,
    "models": 3,
    "visualization": 3,
    # Evidence reads every layer below it and is read by none of them. That is
    # what lets an audit record be reshaped without touching a measurement.
    "evidence": 4,
    "training": 4,
    "evaluation": 4,
    "prediction": 4,
    "cli": 6,
    "facade": 5,
}

#: Pairs that the layer numbers alone would permit but the agreed architecture
#: forbids. Kept explicit so the intent is readable rather than encoded in
#: carefully chosen integers.
FORBIDDEN_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("models", "preprocessing"),
        ("preprocessing", "models"),
        # A model describes what it needs; it does not inspect data to find out.
        # The layer numbers permit this import, so it is named explicitly.
        ("models", "profiling"),
        ("models", "statistics"),
        ("preprocessing", "profiling"),
        # Exploratory analysis must stay useful before any model exists, so the
        # visualization layer knows nothing about models or preprocessing.
        ("visualization", "models"),
        ("visualization", "preprocessing"),
        # Evidence aggregates; nothing may depend on it. A measurement layer that
        # imported the audit record could not be changed without changing the
        # record's format, which is the cycle this whole layer exists to avoid.
        ("core", "evidence"),
        ("statistics", "evidence"),
        ("profiling", "evidence"),
        ("preprocessing", "evidence"),
        ("models", "evidence"),
        ("visualization", "evidence"),
        # Training and evaluation sit on evidence's layer, so the numbers alone
        # permit them to import it. They must not: evidence aggregates what other
        # layers established, and a training run that wrote its own audit record
        # would make the record's shape a dependency of the orchestration.
        ("training", "evidence"),
        ("evaluation", "evidence"),
        # Evaluation computes metrics from arrays. It has no business knowing
        # which models exist, how data is prepared, or how a run is orchestrated.
        ("evaluation", "models"),
        ("evaluation", "preprocessing"),
        ("evaluation", "profiling"),
        ("evaluation", "training"),
        ("evaluation", "visualization"),
        # Nothing below the orchestration layer may reach back up into it.
        ("core", "training"),
        ("statistics", "training"),
        ("profiling", "training"),
        ("preprocessing", "training"),
        ("models", "training"),
        ("visualization", "training"),
        ("evidence", "training"),
        ("core", "evaluation"),
        ("statistics", "evaluation"),
        ("profiling", "evaluation"),
        ("preprocessing", "evaluation"),
        ("models", "evaluation"),
        ("visualization", "evaluation"),
        ("evidence", "evaluation"),
        # Prediction applies a fitted model and learns nothing. It has no reason
        # to know how data is prepared, how a run is orchestrated, or what an
        # audit record looks like.
        ("prediction", "preprocessing"),
        ("prediction", "profiling"),
        ("prediction", "evidence"),
        ("prediction", "evaluation"),
        ("prediction", "visualization"),
        ("core", "prediction"),
        ("statistics", "prediction"),
        ("profiling", "prediction"),
        ("preprocessing", "prediction"),
        ("models", "prediction"),
        ("visualization", "prediction"),
        ("evidence", "prediction"),
        ("training", "prediction"),
        ("evaluation", "prediction"),
        # The facade is the top of the library. It may reach anything below it;
        # nothing below may reach back. A domain layer that imported the
        # convenience wrapper would make the wrapper load-bearing, and the
        # architecture it exists to present would start depending on it.
        ("core", "facade"),
        ("statistics", "facade"),
        ("datasets", "facade"),
        ("profiling", "facade"),
        ("preprocessing", "facade"),
        ("models", "facade"),
        ("visualization", "facade"),
        ("evidence", "facade"),
        ("training", "facade"),
        ("evaluation", "facade"),
        ("prediction", "facade"),
        ("cli", "facade"),
        # The CLI orchestrates. Nothing imports it.
        ("core", "cli"),
        ("profiling", "cli"),
        ("preprocessing", "cli"),
        ("models", "cli"),
        ("evidence", "cli"),
        ("models", "visualization"),
        ("preprocessing", "visualization"),
    }
)


def _module_files() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _module_name(path: Path) -> str:
    """Return the dotted module name for a file inside the package."""
    relative = path.relative_to(PACKAGE_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join([PACKAGE_NAME, *parts])


def _component_of(path: Path) -> str:
    """Return the top-level package or module a file belongs to."""
    relative = path.relative_to(PACKAGE_ROOT)
    return relative.parts[0] if len(relative.parts) > 1 else relative.stem


def _imported_modules(path: Path) -> set[str]:
    """Return the dotted ``aidatasetkit`` modules that ``path`` imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PACKAGE_NAME or alias.name.startswith(f"{PACKAGE_NAME}."):
                    imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise AssertionError(
                    f"{path} uses a relative import. The project uses absolute imports "
                    "only, which is what makes this analysis exact."
                )
            if node.module == PACKAGE_NAME:
                # `from aidatasetkit import models` reaches a sibling component
                # just as surely as importing its module path, but records only
                # the root as the module, so the names carry the components.
                #
                # `from aidatasetkit import __version__` is a different thing: a
                # string constant, not a layer. Only names that are really
                # modules count, which is what separates a cross-layer import
                # from reading a symbol off the package root.
                for alias in node.names:
                    if _is_submodule(alias.name):
                        imported.add(f"{PACKAGE_NAME}.{alias.name}")
            elif node.module and node.module.startswith(f"{PACKAGE_NAME}."):
                imported.add(node.module)

    return imported


def _is_submodule(name: str) -> bool:
    """Whether ``name`` is a module or package inside ``aidatasetkit``.

    Asked of the filesystem rather than guessed from the spelling: a private
    name convention would be a rule about style, and this test is about
    structure.
    """
    return (PACKAGE_ROOT / name).is_dir() or (PACKAGE_ROOT / f"{name}.py").is_file()


def _component_from_module(module: str) -> str | None:
    """Return the component a dotted module belongs to, or None for the root."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else None


def _layer_of(component: str) -> int:
    if component not in LAYERS:
        raise AssertionError(
            f"Component {component!r} has no declared layer. Add it to LAYERS in this "
            "test as a deliberate architectural decision."
        )
    return LAYERS[component]


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: p.name)
def test_module_belongs_to_a_declared_layer(path: Path) -> None:
    component = _component_of(path)
    if component == "__init__":
        return
    _layer_of(component)


def test_dependency_direction_is_never_reversed() -> None:
    """A package may import from its own layer or below, never above."""
    violations: list[str] = []

    for path in _module_files():
        component = _component_of(path)
        if component == "__init__":
            continue
        source_layer = _layer_of(component)

        for module in _imported_modules(path):
            target = _component_from_module(module)
            if target is None or target == component:
                continue
            if _layer_of(target) > source_layer:
                violations.append(
                    f"{path.relative_to(PACKAGE_ROOT)} (layer {source_layer}) "
                    f"imports {module} (layer {_layer_of(target)})"
                )

    assert not violations, "Dependency direction reversed:\n  " + "\n  ".join(violations)


def test_explicitly_forbidden_package_pairs_stay_separate() -> None:
    """Some same-layer pairs are separated on purpose, not by layer arithmetic.

    ``models`` and ``preprocessing`` must not know about each other: they exchange
    information only through ``core`` types, which is what allows one preprocessor
    to be shared by every model with the same capability profile.
    """
    violations: list[str] = []

    for path in _module_files():
        component = _component_of(path)
        if component == "__init__":
            continue
        for module in _imported_modules(path):
            target = _component_from_module(module)
            if target and (component, target) in FORBIDDEN_PAIRS:
                violations.append(f"{path.relative_to(PACKAGE_ROOT)} imports {module}")

    assert not violations, "Forbidden package coupling:\n  " + "\n  ".join(violations)


def test_sibling_collaboration_inside_a_package_is_allowed() -> None:
    """Guards the rule itself: same-package imports must never be reported.

    Without this, a future tightening of the direction check could quietly ban
    the modular decomposition the architecture depends on.
    """
    for importer, imported in [
        ("statistics", f"{PACKAGE_NAME}.statistics.moments"),
        ("evaluation", f"{PACKAGE_NAME}.evaluation.metrics"),
        ("models", f"{PACKAGE_NAME}.models.registry"),
    ]:
        target = _component_from_module(imported)
        assert target == importer
        assert (importer, target) not in FORBIDDEN_PAIRS


def test_the_package_root_is_the_only_module_that_may_import_everything() -> None:
    """``aidatasetkit/__init__.py`` is the assembly point and is exempt by design."""
    root_init = PACKAGE_ROOT / "__init__.py"
    assert root_init.exists()
    components = {
        _component_from_module(module) for module in _imported_modules(root_init)
    }
    assert components - {None} <= set(LAYERS)


def test_core_depends_on_nothing_internal_except_core() -> None:
    for path in (PACKAGE_ROOT / "core").glob("*.py"):
        targets = {_component_from_module(module) for module in _imported_modules(path)}
        assert targets - {None} <= {"core"}, f"{path.name} reaches outside core"


def test_the_import_graph_is_acyclic() -> None:
    """No cycles anywhere, including between siblings in the same package."""
    known_modules = {_module_name(path) for path in _module_files()}
    graph: dict[str, set[str]] = defaultdict(set)

    for path in _module_files():
        source = _module_name(path)
        graph[source] = {
            module for module in _imported_modules(path) if module in known_modules
        } - {source}

    visiting: set[str] = set()
    settled: set[str] = set()

    def walk(module: str, trail: tuple[str, ...]) -> None:
        if module in settled:
            return
        if module in visiting:
            cycle = " -> ".join(trail[trail.index(module) :] + (module,))
            raise AssertionError(f"Import cycle: {cycle}")
        visiting.add(module)
        for dependency in sorted(graph[module]):
            walk(dependency, trail + (module,))
        visiting.discard(module)
        settled.add(module)

    for module in sorted(graph):
        walk(module, ())
