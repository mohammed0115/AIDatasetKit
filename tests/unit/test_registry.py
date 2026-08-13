"""Tests for the model registry.

Every test that registers anything works against a scratch registry created in a
fixture. Nothing here touches the built-in registry, so test order cannot change
a result and a failure cannot leak into the next test.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import pytest

from aidatasetkit.core.exceptions import (
    AmbiguousModelAliasError,
    DuplicateModelError,
    IncompatibleModelError,
    MissingDependencyError,
    UnknownModelError,
    ValidationError,
)
from aidatasetkit.core.types import Backend, Interpretability, TaskType
from tests.conftest import (
    BUILT_IN_CLASSIFIERS,
    BUILT_IN_CLUSTERERS,
    BUILT_IN_REGRESSORS,
)

from aidatasetkit.models import (
    ModelCapabilities,
    ModelRegistry,
    ModelStrategy,
    default_registry,
    register_model,
)


def make_capabilities(task: TaskType = TaskType.CLASSIFICATION, **overrides):
    defaults: dict[str, Any] = {
        "task_type": task,
        "backend": Backend.SKLEARN,
        "supports_predict_proba": task is TaskType.CLASSIFICATION,
        "requires_scaling": False,
        "supports_sparse_input": True,
        "supports_multiclass": task is TaskType.CLASSIFICATION,
        "handles_missing_values": False,
        "interpretability_level": Interpretability.HIGH,
    }
    return ModelCapabilities(**{**defaults, **overrides})


def make_strategy(
    name: str, task: TaskType = TaskType.CLASSIFICATION, **capability_overrides
) -> type[ModelStrategy]:
    """Build a throwaway strategy class for registry tests."""

    class Fake(ModelStrategy):
        capabilities: ClassVar[ModelCapabilities] = make_capabilities(
            task, **capability_overrides
        )

        def default_params(self) -> dict[str, Any]:
            return {"alpha": 1.0}

        def build(self, **params: Any):
            raise NotImplementedError("registry tests never construct estimators")

    Fake.name = name
    Fake.__name__ = f"Fake_{name}"
    return Fake


@pytest.fixture
def registry() -> ModelRegistry:
    """An isolated registry. The built-in one is never touched by these tests."""
    return ModelRegistry()


class TestRegistration:
    def test_a_strategy_can_be_registered_and_found(self, registry):
        strategy = make_strategy("thing_classifier")
        entry = registry.register(strategy)
        assert entry.canonical_name == "thing_classifier"
        assert registry.resolve("thing_classifier").strategy_type is strategy
        assert "thing_classifier" in registry
        assert len(registry) == 1

    def test_the_decorator_registers_into_a_chosen_registry(self, registry):
        @register_model(aliases=("thing",), registry=registry)
        class Thing(ModelStrategy):
            name: ClassVar[str] = "thing_classifier"
            capabilities: ClassVar[ModelCapabilities] = make_capabilities()

            def build(self, **params: Any):
                raise NotImplementedError

        assert registry.resolve("thing").strategy_type is Thing
        assert "thing_classifier" not in default_registry()

    def test_iteration_yields_the_registrations(self, registry):
        registry.register(make_strategy("a_classifier"))
        registry.register(make_strategy("b_classifier"))
        assert {entry.canonical_name for entry in registry} == {
            "a_classifier",
            "b_classifier",
        }

    def test_repr_reports_the_size(self, registry):
        registry.register(make_strategy("a_classifier"))
        assert "1 models" in repr(registry)


class TestRegistrationIsRefusedWhenUnsafe:
    def test_a_duplicate_canonical_name_is_refused(self, registry):
        registry.register(make_strategy("thing_classifier"))
        with pytest.raises(DuplicateModelError, match="already registered"):
            registry.register(make_strategy("thing_classifier"))

    def test_the_first_registration_survives_a_refused_duplicate(self, registry):
        first = make_strategy("thing_classifier")
        registry.register(first)
        with pytest.raises(DuplicateModelError):
            registry.register(make_strategy("thing_classifier"))
        assert registry.resolve("thing_classifier").strategy_type is first

    def test_an_alias_colliding_with_a_canonical_name_is_refused(self, registry):
        registry.register(make_strategy("thing_classifier"))
        with pytest.raises(DuplicateModelError, match="canonical name of another"):
            registry.register(make_strategy("other_classifier"), aliases=("thing_classifier",))

    def test_a_canonical_name_colliding_with_an_existing_alias_is_refused(self, registry):
        registry.register(make_strategy("thing_classifier"), aliases=("shortcut",))
        with pytest.raises(DuplicateModelError, match="already used as an alias"):
            registry.register(make_strategy("shortcut"))

    def test_an_alias_repeated_within_one_task_is_refused(self, registry):
        registry.register(make_strategy("first_classifier"), aliases=("shared",))
        with pytest.raises(DuplicateModelError, match="same task"):
            registry.register(make_strategy("second_classifier"), aliases=("shared",))

    def test_an_alias_shared_across_tasks_is_allowed(self, registry):
        """This is the "random_forest" case the design has to support."""
        registry.register(make_strategy("forest_classifier"), aliases=("forest",))
        registry.register(
            make_strategy("forest_regressor", TaskType.REGRESSION), aliases=("forest",)
        )
        assert len(registry) == 2

    def test_an_alias_equal_to_its_own_canonical_name_is_refused(self, registry):
        with pytest.raises(ValidationError, match="already the canonical name"):
            registry.register(
                make_strategy("thing_classifier"), aliases=("thing_classifier",)
            )

    @pytest.mark.parametrize("alias", ["", "   ", 42, None])
    def test_an_unusable_alias_is_refused(self, registry, alias):
        with pytest.raises(ValidationError, match="non-empty strings"):
            registry.register(make_strategy("thing_classifier"), aliases=(alias,))

    def test_a_bare_string_of_aliases_is_refused(self, registry):
        """It is an iterable, so it would silently become one alias per character."""
        with pytest.raises(ValidationError, match="not a single string"):
            registry.register(make_strategy("thing_classifier"), aliases="logreg")

    def test_a_canonical_name_with_whitespace_is_refused(self, registry):
        """Lookups strip, so such a name would register and then be unreachable."""
        with pytest.raises(ValidationError, match="whitespace"):
            registry.register(make_strategy(" spaced_classifier "))

    def test_whitespace_cannot_be_used_to_shadow_an_existing_name(self, registry):
        registry.register(make_strategy("thing_classifier"))
        with pytest.raises(ValidationError, match="whitespace"):
            registry.register(make_strategy(" thing_classifier"))
        assert registry.available() == ("thing_classifier",)

    def test_an_alias_with_whitespace_is_refused(self, registry):
        with pytest.raises(ValidationError, match="whitespace"):
            registry.register(make_strategy("thing_classifier"), aliases=(" thing",))

    def test_a_repeated_alias_in_one_call_is_stored_once(self, registry):
        registry.register(make_strategy("thing_classifier"), aliases=("x", "x", "y"))
        entry = registry.resolve("thing_classifier")
        assert entry.aliases == ("x", "y")
        assert registry.resolve("x").canonical_name == "thing_classifier"

    def test_a_refused_registration_leaves_no_partial_state(self, registry):
        """The second alias clashes; the first must not survive the refusal."""
        registry.register(make_strategy("first_classifier"), aliases=("taken",))
        with pytest.raises(DuplicateModelError):
            registry.register(
                make_strategy("second_classifier"), aliases=("fresh", "taken")
            )
        assert len(registry) == 1
        assert "second_classifier" not in registry
        with pytest.raises(UnknownModelError):
            registry.resolve("fresh")

    def test_a_non_strategy_is_refused(self, registry):
        with pytest.raises(ValidationError, match="ModelStrategy subclasses"):
            registry.register(dict)

    def test_an_abstract_strategy_is_refused(self, registry):
        class Abstract(ModelStrategy):
            name: ClassVar[str] = "abstract_classifier"
            capabilities: ClassVar[ModelCapabilities] = make_capabilities()

        with pytest.raises(ValidationError, match="abstract"):
            registry.register(Abstract)

    def test_a_strategy_without_a_name_is_refused(self, registry):
        class Nameless(ModelStrategy):
            capabilities: ClassVar[ModelCapabilities] = make_capabilities()

            def build(self, **params: Any):
                raise NotImplementedError

        with pytest.raises(ValidationError, match="canonical 'name'"):
            registry.register(Nameless)

    def test_a_strategy_without_capabilities_is_refused(self, registry):
        class Undeclared(ModelStrategy):
            name: ClassVar[str] = "undeclared_classifier"

            def build(self, **params: Any):
                raise NotImplementedError

        with pytest.raises(ValidationError, match="'capabilities'"):
            registry.register(Undeclared)


class TestResolution:
    def test_a_canonical_name_resolves(self, registry):
        registry.register(make_strategy("thing_classifier"), aliases=("thing",))
        assert registry.resolve("thing_classifier").canonical_name == "thing_classifier"

    def test_an_alias_resolves(self, registry):
        registry.register(make_strategy("thing_classifier"), aliases=("thing", "th"))
        assert registry.resolve("th").canonical_name == "thing_classifier"

    def test_surrounding_whitespace_is_tolerated(self, registry):
        registry.register(make_strategy("thing_classifier"))
        assert registry.resolve("  thing_classifier ").canonical_name == "thing_classifier"

    def test_an_ambiguous_alias_is_refused_and_names_the_candidates(self, registry):
        registry.register(make_strategy("forest_classifier"), aliases=("forest",))
        registry.register(
            make_strategy("forest_regressor", TaskType.REGRESSION), aliases=("forest",)
        )
        with pytest.raises(AmbiguousModelAliasError) as error:
            registry.resolve("forest")
        message = str(error.value)
        assert "forest_classifier" in message and "forest_regressor" in message

    def test_task_context_disambiguates_an_alias(self, registry):
        registry.register(make_strategy("forest_classifier"), aliases=("forest",))
        registry.register(
            make_strategy("forest_regressor", TaskType.REGRESSION), aliases=("forest",)
        )
        assert (
            registry.resolve("forest", task=TaskType.CLASSIFICATION).canonical_name
            == "forest_classifier"
        )
        assert registry.resolve("forest", task="regression").canonical_name == (
            "forest_regressor"
        )

    def test_an_alias_that_serves_no_such_task_is_reported(self, registry):
        registry.register(make_strategy("forest_classifier"), aliases=("forest",))
        with pytest.raises(IncompatibleModelError, match="none of which serves regression"):
            registry.resolve("forest", task="regression")

    def test_a_canonical_name_from_another_task_is_contradicted_not_ignored(self, registry):
        """The wrong-task request must fail here, not later inside fit()."""
        registry.register(make_strategy("forest_classifier"))
        with pytest.raises(IncompatibleModelError, match="serves classification"):
            registry.resolve("forest_classifier", task="regression")

    def test_an_invalid_task_is_rejected_on_the_canonical_path_too(self, registry):
        """It used to be validated only when the name happened to be an alias."""
        registry.register(make_strategy("forest_classifier"))
        with pytest.raises(ValidationError, match="not a valid TaskType"):
            registry.resolve("forest_classifier", task="not_a_task")

    def test_a_matching_task_is_accepted_for_a_canonical_name(self, registry):
        registry.register(make_strategy("forest_classifier"))
        assert registry.resolve("forest_classifier", task="classification") is not None

    def test_membership_agrees_with_resolution(self, registry):
        registry.register(make_strategy("thing_classifier"), aliases=("thing",))
        for name in ("thing_classifier", "thing", "  thing  "):
            assert name in registry
            assert registry.resolve(name) is not None
        assert "absent" not in registry
        assert 42 not in registry

    def test_an_unknown_name_lists_what_is_registered(self, registry):
        registry.register(make_strategy("thing_classifier"))
        with pytest.raises(UnknownModelError) as error:
            registry.resolve("nonexistent")
        assert "thing_classifier" in str(error.value)

    def test_a_near_miss_gets_a_suggestion(self, registry):
        registry.register(make_strategy("logistic_regression"))
        with pytest.raises(UnknownModelError, match="Did you mean"):
            registry.resolve("logistic_regresion")

    def test_a_non_string_name_is_refused(self, registry):
        with pytest.raises(ValidationError, match="must be a string"):
            registry.resolve(42)

    def test_an_empty_registry_reports_nothing_registered(self, registry):
        with pytest.raises(UnknownModelError):
            registry.resolve("anything")


class TestOptionalDependencies:
    def test_a_model_with_a_missing_package_stays_visible(self, registry):
        registry.register(
            make_strategy("fictional_classifier"), requires_package="not_a_real_package"
        )
        entry = registry.resolve("fictional_classifier")
        assert not entry.is_available
        assert "fictional_classifier" in registry.available(include_unavailable=True)
        assert "fictional_classifier" not in registry.available()

    def test_requesting_an_unavailable_model_explains_how_to_install_it(self, registry):
        registry.register(make_strategy("boost_classifier"), requires_package="xgboost")
        entry = registry.resolve("boost_classifier")
        if entry.is_available:
            pytest.skip("xgboost is installed in this environment")
        with pytest.raises(MissingDependencyError, match=r"pip install aidatasetkit\[boosting\]"):
            entry.require_available()

    def test_every_suggested_extra_actually_installs_the_package_it_names(self):
        """An install command that does not fix the problem is worse than none.

        Checked against pyproject directly, because the test above skips wherever
        the backend happens to be installed -- which is where a wrong mapping
        would ship unnoticed.
        """
        import tomllib
        from pathlib import Path

        import aidatasetkit
        from aidatasetkit.models.registry import _INSTALL_EXTRAS

        root = Path(aidatasetkit.__file__).resolve().parent.parent
        with open(root / "pyproject.toml", "rb") as handle:
            extras = tomllib.load(handle)["project"]["optional-dependencies"]

        for package, extra in _INSTALL_EXTRAS.items():
            assert extra in extras, f"{extra!r} is not declared in pyproject"
            declared = " ".join(extras[extra])
            assert package in declared, (
                f"installing aidatasetkit[{extra}] would not provide {package!r}"
            )

    def test_availability_is_re_probed_so_an_in_session_install_is_noticed(self, registry):
        """A cached negative would keep printing an install command already run."""
        from aidatasetkit.models import registry as registry_module

        registry.register(
            make_strategy("late_classifier"), requires_package="not_yet_installed"
        )
        entry = registry.resolve("late_classifier")
        assert not entry.is_available

        registry_module._KNOWN_INSTALLED.add("not_yet_installed")
        try:
            assert entry.is_available
        finally:
            registry_module._KNOWN_INSTALLED.discard("not_yet_installed")

    def test_an_unrecognised_package_gets_a_plain_install_command(self, registry):
        registry.register(
            make_strategy("fictional_classifier"), requires_package="not_a_real_package"
        )
        with pytest.raises(MissingDependencyError, match="pip install not_a_real_package"):
            registry.resolve("fictional_classifier").require_available()

    def test_an_available_model_passes_the_check(self, registry):
        registry.register(make_strategy("thing_classifier"), requires_package="numpy")
        entry = registry.resolve("thing_classifier")
        assert entry.is_available
        entry.require_available()

    def test_a_model_without_a_package_requirement_is_always_available(self, registry):
        registry.register(make_strategy("thing_classifier"))
        assert registry.resolve("thing_classifier").is_available


class TestDiscovery:
    @pytest.fixture
    def populated(self, registry) -> ModelRegistry:
        registry.register(make_strategy("b_classifier"), aliases=("b",))
        registry.register(make_strategy("a_classifier"), aliases=("a",))
        registry.register(make_strategy("z_regressor", TaskType.REGRESSION))
        registry.register(
            make_strategy("gone_classifier"), requires_package="not_a_real_package"
        )
        return registry

    def test_available_is_sorted(self, populated):
        assert populated.available() == ("a_classifier", "b_classifier", "z_regressor")

    def test_available_excludes_uninstallable_models_by_default(self, populated):
        assert "gone_classifier" not in populated.available()
        assert "gone_classifier" in populated.available(include_unavailable=True)

    def test_available_filters_by_task(self, populated):
        assert populated.available(task=TaskType.REGRESSION) == ("z_regressor",)
        assert populated.available(task="classification") == ("a_classifier", "b_classifier")

    def test_the_catalog_includes_uninstallable_models_by_default(self, populated):
        names = {entry.canonical_name for entry in populated.catalog()}
        assert "gone_classifier" in names

    def test_aliases_can_be_looked_up(self, populated):
        assert populated.aliases_of("a_classifier") == ("a",)


class TestSerialisation:
    def test_a_registration_round_trips_through_json(self, registry):
        registry.register(make_strategy("thing_classifier"), aliases=("thing",))
        payload = json.loads(json.dumps(registry.resolve("thing_classifier").to_dict()))
        assert payload["canonical_name"] == "thing_classifier"
        assert payload["aliases"] == ["thing"]
        assert payload["task_type"] == "classification"
        assert payload["backend"] == "sklearn"
        assert payload["available"] is True
        assert payload["capabilities"]["supports_predict_proba"] is True
        assert payload["default_params"] == {"alpha": 1.0}

    def test_no_estimator_object_leaks_into_the_catalog(self, registry):
        registry.register(make_strategy("thing_classifier"))
        payload = registry.resolve("thing_classifier").to_dict()
        assert "strategy_type" not in payload
        json.dumps(payload)

    def test_one_uninstallable_backend_does_not_take_the_catalog_down(self, registry):
        """An optional adapter imports its library lazily, inside strategy code."""

        class LazyAdapter(ModelStrategy):
            name: ClassVar[str] = "lazy_classifier"
            capabilities: ClassVar[ModelCapabilities] = make_capabilities()

            def default_params(self) -> dict[str, Any]:
                import definitely_not_installed  # noqa: F401

                return {}

            def build(self, **params: Any):
                raise NotImplementedError

        registry.register(make_strategy("healthy_classifier"))
        registry.register(LazyAdapter, requires_package="definitely_not_installed")

        payloads = {entry.to_dict()["canonical_name"]: entry.to_dict() for entry in registry}
        assert payloads["lazy_classifier"]["available"] is False
        assert payloads["lazy_classifier"]["default_params"] is None
        assert payloads["healthy_classifier"]["default_params"] == {"alpha": 1.0}
        json.dumps(list(payloads.values()))

    def test_enums_serialise_as_their_values(self, registry):
        registry.register(make_strategy("thing_classifier"))
        payload = registry.resolve("thing_classifier").to_dict()
        assert isinstance(payload["task_type"], str)
        assert isinstance(payload["interpretability_level"], str)


class TestIsolationFromTheBuiltInRegistry:
    def test_the_built_in_registry_holds_the_expected_models(self):
        assert (
            set(default_registry().available())
            == set(BUILT_IN_CLASSIFIERS)
            | set(BUILT_IN_REGRESSORS)
            | set(BUILT_IN_CLUSTERERS)
        )

    def test_the_three_families_coexist_without_a_canonical_name_collision(self):
        """Twenty-four registrations, twenty-four distinct canonical names.

        Registration refuses a duplicate outright, so a reused name would have
        aborted import rather than failed here -- which is the point: asserting
        the count says which invariant kept that from happening.
        """
        names = [entry.canonical_name for entry in default_registry()]
        assert len(names) == len(set(names)) == 24

    def test_every_shared_alias_maps_to_exactly_one_model_per_family(self):
        """The alias namespace stayed resolvable as the second family arrived."""
        seen: dict[tuple[str, TaskType], str] = {}
        for entry in default_registry():
            for alias in entry.aliases:
                key = (alias, entry.task_type)
                assert key not in seen, f"{alias} is ambiguous within {entry.task_type}"
                seen[key] = entry.canonical_name
        assert seen

    #: The eight aliases S6 made ambiguous, named once so the number is a fact
    #: rather than a count nobody checked.
    SHARED_ALIASES = (
        "baseline",
        "decision_tree",
        "dummy",
        "extra_trees",
        "gradient_boosting",
        "hist_gradient_boosting",
        "knn",
        "random_forest",
    )

    def test_the_shared_aliases_are_exactly_these_eight(self):
        """Derived from the registry, so the documented list cannot drift from it.

        docs/getting-started.md enumerates these for users, and an earlier
        version of that list had seven -- ``extra_trees`` was missing, so a
        reader would have believed it was safe to pass bare.
        """
        by_alias: dict[str, set[TaskType]] = {}
        for entry in default_registry():
            for alias in entry.aliases:
                by_alias.setdefault(alias, set()).add(entry.task_type)
        shared = tuple(sorted(a for a, tasks in by_alias.items() if len(tasks) > 1))
        assert shared == self.SHARED_ALIASES

    def test_the_documented_list_names_every_one_of_them(self):
        """The disclosure has to be complete or it misleads worse than silence."""
        from pathlib import Path

        docs = Path(__file__).resolve().parents[2] / "docs" / "getting-started.md"
        text = docs.read_text(encoding="utf-8")
        for alias in self.SHARED_ALIASES:
            assert f"`{alias}`" in text, f"getting-started.md does not mention {alias}"

    def test_and_the_changelog_records_the_break(self):
        from pathlib import Path

        changelog = (
            Path(__file__).resolve().parents[2] / "CHANGELOG.md"
        ).read_text(encoding="utf-8")
        assert "AmbiguousModelAliasError" in changelog

    @pytest.mark.parametrize("alias", SHARED_ALIASES)
    def test_membership_is_true_while_resolution_refuses(self, alias):
        """The invariant S6 changed, pinned rather than left to be discovered.

        ``__contains__`` answers "is this name registered", and it is. ``resolve``
        answers "which model is it", and without a task there is no single
        answer. The docstring on ``__contains__`` says so; this proves the two
        really do behave as it describes.
        """
        registry = default_registry()
        assert alias in registry
        with pytest.raises(AmbiguousModelAliasError):
            registry.resolve(alias)
        assert registry.resolve(alias, task=TaskType.CLASSIFICATION) is not None
        assert registry.resolve(alias, task=TaskType.REGRESSION) is not None

    def test_an_unshared_alias_still_satisfies_the_old_guard_pattern(self):
        """Only the eight are affected; every other name behaves as before."""
        registry = default_registry()
        for alias in ("logistic", "logreg", "gnb", "ridge"):
            assert alias in registry
            assert registry.resolve(alias) is not None

    def test_a_scratch_registry_starts_empty(self):
        assert len(ModelRegistry()) == 0

    def test_registering_in_a_scratch_registry_leaves_the_built_in_one_alone(self):
        before = default_registry().available()
        scratch = ModelRegistry()
        scratch.register(make_strategy("temporary_classifier"))
        assert default_registry().available() == before
        assert "temporary_classifier" not in default_registry()

    def test_re_running_a_registration_is_refused_rather_than_silently_applied(self):
        """The real re-execution path, not the import cache that normally hides it.

        ``importlib.import_module`` on a loaded module returns it from
        ``sys.modules`` without re-running the body, so asserting on that proves a
        CPython guarantee rather than anything about this registry. Registering
        the same strategy again is what actually exercises the protection.
        """
        from aidatasetkit.models.classification.dummy import DummyClassifierStrategy

        before = default_registry().available()
        with pytest.raises(DuplicateModelError):
            default_registry().register(DummyClassifierStrategy)
        assert default_registry().available() == before
