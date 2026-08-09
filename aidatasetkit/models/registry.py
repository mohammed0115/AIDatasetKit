"""What models AIDatasetKit knows about.

The registry is the single source of truth. The factory, the comparator, the
facade, and the documentation all ask it rather than keeping their own lists,
which is what stops those lists from drifting apart.

Two design points carry most of the weight:

**Nothing is ever silently replaced.** Registering a canonical name twice, or an
alias that could never be resolved unambiguously, is refused at registration
time. A registry that accepted the second registration would hand back a
different model than the one the caller's code was written against.

**A model whose package is missing is still listed.** Optional backends are
declared with ``requires_package`` and remain visible in the catalog, marked
unavailable. Asking for one raises
:class:`~aidatasetkit.core.exceptions.MissingDependencyError` with an install
command. Silently hiding it would leave the user unable to tell a typo from a
missing dependency.
"""

from __future__ import annotations

import difflib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

from aidatasetkit.core.exceptions import (
    AmbiguousModelAliasError,
    DuplicateModelError,
    IncompatibleModelError,
    MissingDependencyError,
    UnknownModelError,
    ValidationError,
)
from aidatasetkit.core.types import Backend, TaskType, jsonable
from aidatasetkit.models.base import ModelStrategy
from aidatasetkit.models.capabilities import ModelCapabilities

__all__ = [
    "ModelRegistration",
    "ModelRegistry",
    "default_registry",
    "register_model",
]

_logger = logging.getLogger(__name__)

#: Extras that install each optional backend, used to build install messages.
#:
#: Keys are *import* names, because that is what ``find_spec`` resolves. Every
#: value here must name an extra that actually installs the package -- an install
#: command that does not fix the problem is worse than no suggestion at all, so
#: this mapping is checked against ``pyproject.toml`` by the test suite.
_INSTALL_EXTRAS: dict[str, str] = {
    "xgboost": "boosting",
    "lightgbm": "boosting",
    "catboost": "boosting",
}


#: Packages already found. Only positive answers are remembered, deliberately.
_KNOWN_INSTALLED: set[str] = set()


def _is_installed(package: str) -> bool:
    """Whether ``package`` can be imported, without importing it.

    Positive answers are cached forever; negative ones are re-probed every time.
    The asymmetry is the point. A package cannot stop being importable inside a
    running process, but it can *start*: an analyst who hits
    :class:`~aidatasetkit.core.exceptions.MissingDependencyError` in a notebook
    runs the install command it printed and carries on in the same kernel. A
    cached negative would keep raising that error and telling them to do what
    they have just done.

    Re-probing costs a ``find_spec`` call, and only for models that declare an
    optional package and do not have it. Models without ``requires_package``
    never reach this function at all.
    """
    if package in _KNOWN_INSTALLED:
        return True
    try:
        found = find_spec(package) is not None
    except (ImportError, ValueError):
        found = False
    if found:
        _KNOWN_INSTALLED.add(package)
    return found


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    """One entry in the registry."""

    canonical_name: str
    strategy_type: type[ModelStrategy]
    aliases: tuple[str, ...] = ()
    requires_package: str | None = None

    @property
    def capabilities(self) -> ModelCapabilities:
        """The registered model's declared capabilities."""
        return self.strategy_type.capabilities

    @property
    def task_type(self) -> TaskType:
        """The task family this model serves."""
        return self.capabilities.task_type

    @property
    def backend(self) -> Backend:
        """The library implementing this model."""
        return self.capabilities.backend

    @property
    def is_available(self) -> bool:
        """Whether this model can be constructed in the current environment."""
        return self.requires_package is None or _is_installed(self.requires_package)

    def require_available(self) -> None:
        """Raise unless the model's optional package is installed.

        Raises:
            MissingDependencyError: With the command that would fix it.
        """
        if self.is_available:
            return
        package = self.requires_package
        extra = _INSTALL_EXTRAS.get(str(package))
        install = (
            f"pip install aidatasetkit[{extra}]" if extra else f"pip install {package}"
        )
        raise MissingDependencyError(
            f"Model {self.canonical_name!r} requires the {package!r} package, which "
            f"is not installed. Install it with: {install}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of this entry.

        Deliberately free of estimator objects: the catalog is metadata, and a
        live estimator in it would neither serialise nor mean anything.
        """
        return {
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "task_type": self.task_type.value,
            "backend": self.backend.value,
            "available": self.is_available,
            "requires_package": self.requires_package,
            "is_baseline": self.capabilities.is_baseline,
            "interpretability_level": self.capabilities.interpretability_level.value,
            "capabilities": self.capabilities.to_dict(),
            "default_params": self._describe_default_params(),
        }

    def _describe_default_params(self) -> dict[str, Any] | None:
        """Read the strategy's defaults, or report that they could not be read.

        Instantiating the strategy runs its code. An optional-backend adapter
        imports its library lazily -- a module-level import would abort
        registration and make the model invisible, which is exactly what
        ``requires_package`` exists to avoid -- so that import lands here. One
        uninstalled backend must not take the whole catalog down with it.
        """
        if not self.is_available:
            return None
        try:
            return {
                str(key): jsonable(value)
                for key, value in self.strategy_type().default_params().items()
            }
        except Exception:
            _logger.warning(
                "Default parameters for %r could not be read; the catalog reports "
                "them as unavailable.",
                self.canonical_name,
                exc_info=True,
            )
            return None


class ModelRegistry:
    """A collection of registered model strategies.

    Instantiable rather than global-only, so that tests and experiments can work
    against an isolated registry without touching the built-in one.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ModelRegistration] = {}
        self._aliases: dict[str, list[str]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        """Whether ``name`` resolves, by canonical name or alias.

        Deliberately matches :meth:`resolve`: a membership test that disagreed
        with it could not be used to guard a lookup.
        """
        if not isinstance(name, str):
            return False
        key = name.strip()
        return key in self._entries or key in self._aliases

    def __iter__(self) -> Iterator[ModelRegistration]:
        return iter(self._entries.values())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self._entries)} models)"

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register(
        self,
        strategy_type: type[ModelStrategy],
        *,
        aliases: tuple[str, ...] | list[str] = (),
        requires_package: str | None = None,
    ) -> ModelRegistration:
        """Add a strategy to this registry.

        Args:
            strategy_type: A concrete :class:`ModelStrategy` subclass declaring
                ``name`` and ``capabilities``.
            aliases: Short names that resolve to this model. An alias may be
                shared by models in different task families and disambiguated by
                task; it may not be shared within one family, because that could
                never be resolved.
            requires_package: *Import* name of the package needed to construct
                this model, for optional backends. It is the name ``find_spec``
                resolves, which is not always the distribution name --
                ``scikit-learn`` is imported as ``sklearn``.

        Returns:
            The stored :class:`ModelRegistration`.

        Raises:
            ValidationError: If the strategy does not declare what it must.
            DuplicateModelError: If the canonical name is taken, or an alias
                would clash irrecoverably.
        """
        self._validate_strategy_type(strategy_type)
        canonical = strategy_type.name
        alias_tuple = self._normalise_aliases(aliases, canonical)

        if canonical in self._entries:
            raise DuplicateModelError(
                f"A model is already registered under the canonical name "
                f"{canonical!r} ({self._entries[canonical].strategy_type.__name__}). "
                "Registration never replaces silently; choose another name or use a "
                "separate registry."
            )
        if canonical in self._aliases:
            raise DuplicateModelError(
                f"The canonical name {canonical!r} is already used as an alias of "
                f"{self._aliases[canonical]}."
            )

        task = strategy_type.capabilities.task_type
        for alias in alias_tuple:
            self._validate_alias(alias, canonical, task)

        entry = ModelRegistration(
            canonical_name=canonical,
            strategy_type=strategy_type,
            aliases=alias_tuple,
            requires_package=requires_package,
        )
        self._entries[canonical] = entry
        for alias in alias_tuple:
            self._aliases.setdefault(alias, []).append(canonical)
        return entry

    def _validate_strategy_type(self, strategy_type: type[ModelStrategy]) -> None:
        """Reject anything that is not a usable, fully declared strategy."""
        if not isinstance(strategy_type, type) or not issubclass(
            strategy_type, ModelStrategy
        ):
            raise ValidationError(
                f"Only ModelStrategy subclasses can be registered, got "
                f"{strategy_type!r}."
            )
        if getattr(strategy_type, "__abstractmethods__", None):
            raise ValidationError(
                f"{strategy_type.__name__} is abstract; it does not implement "
                f"{sorted(strategy_type.__abstractmethods__)}."
            )
        name = getattr(strategy_type, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValidationError(
                f"{strategy_type.__name__} must declare a non-empty canonical "
                "'name' class attribute."
            )
        if name != name.strip():
            # Lookups strip their input, so a stored name carrying whitespace
            # could never be reached, while still occupying the namespace and
            # shadowing the name a user would actually type.
            raise ValidationError(
                f"{strategy_type.__name__} declares the canonical name {name!r}, "
                "which has leading or trailing whitespace. Names are looked up "
                "stripped, so this one could never be resolved."
            )
        if not isinstance(
            getattr(strategy_type, "capabilities", None), ModelCapabilities
        ):
            raise ValidationError(
                f"{strategy_type.__name__} must declare a 'capabilities' class "
                "attribute of type ModelCapabilities."
            )

    @staticmethod
    def _normalise_aliases(
        aliases: tuple[str, ...] | list[str], canonical: str
    ) -> tuple[str, ...]:
        """Validate the alias container itself, then de-duplicate.

        A bare string is refused explicitly. ``dict.fromkeys`` accepts any
        iterable, and a string is one, so ``aliases="logreg"`` would otherwise
        register six single-character aliases without complaint -- wrong in
        itself, and it squats on six entries of the alias namespace.
        """
        if isinstance(aliases, str):
            raise ValidationError(
                f"aliases for {canonical!r} must be a sequence of strings, not a "
                f"single string. Pass ({aliases!r},) rather than {aliases!r}, which "
                "would register one alias per character."
            )
        return tuple(dict.fromkeys(aliases))

    def _validate_alias(self, alias: str, canonical: str, task: TaskType) -> None:
        """Reject an alias that could not be resolved unambiguously."""
        if not isinstance(alias, str) or not alias.strip():
            raise ValidationError(
                f"Aliases of {canonical!r} must be non-empty strings, got {alias!r}."
            )
        if alias != alias.strip():
            raise ValidationError(
                f"Alias {alias!r} of {canonical!r} has leading or trailing "
                "whitespace. Aliases are looked up stripped, so it could never "
                "be resolved."
            )
        if alias == canonical:
            raise ValidationError(
                f"{alias!r} is already the canonical name of this model; listing it "
                "as an alias adds nothing."
            )
        if alias in self._entries:
            raise DuplicateModelError(
                f"Alias {alias!r} of {canonical!r} is already the canonical name of "
                "another model. Canonical names always win, so this alias could "
                "never resolve."
            )
        for existing in self._aliases.get(alias, ()):
            if self._entries[existing].task_type is task:
                raise DuplicateModelError(
                    f"Alias {alias!r} is already registered for {existing!r}, which "
                    f"serves the same task ({task.value}). The two could never be "
                    "told apart, so the alias is refused."
                )

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #

    def resolve(
        self, name: str, *, task: TaskType | str | None = None
    ) -> ModelRegistration:
        """Find the model registered under a canonical name or alias.

        Args:
            name: A canonical name or an alias.
            task: Restricts the result to one task family. It narrows an alias
                shared across families, and it is *checked* against a canonical
                name rather than ignored -- asking for a classifier under
                ``task="regression"`` is a contradiction, and answering it
                silently would defer the failure to ``fit``.

        Returns:
            The matching :class:`ModelRegistration`.

        Raises:
            ValidationError: If ``name`` is not a string, or ``task`` does not
                name a task family.
            UnknownModelError: If nothing matches.
            IncompatibleModelError: If the name resolves but serves another task.
            AmbiguousModelAliasError: If an alias matches several models and
                ``task`` does not narrow it to one.
        """
        if not isinstance(name, str):
            raise ValidationError(f"A model name must be a string, got {name!r}.")

        # Coerced up front so that an invalid task is rejected on every path,
        # not only when the name happens to be an alias.
        wanted = TaskType.coerce(task) if task is not None else None
        key = name.strip()

        if key in self._entries:
            entry = self._entries[key]
            if wanted is not None and entry.task_type is not wanted:
                raise IncompatibleModelError(
                    f"Model {key!r} serves {entry.task_type.value} tasks, but "
                    f"{wanted.value} was requested."
                )
            return entry

        candidates = [self._entries[c] for c in self._aliases.get(key, ())]
        if wanted is not None:
            narrowed = [entry for entry in candidates if entry.task_type is wanted]
            if not narrowed and candidates:
                raise IncompatibleModelError(
                    f"Alias {key!r} matches "
                    f"{sorted(e.canonical_name for e in candidates)}, none of which "
                    f"serves {wanted.value} tasks."
                )
            candidates = narrowed

        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise AmbiguousModelAliasError(
                f"Alias {key!r} matches several models: "
                f"{sorted(entry.canonical_name for entry in candidates)}. Pass "
                "task= to narrow it, or use a canonical name. Nothing is chosen "
                "arbitrarily."
            )
        raise UnknownModelError(self._unknown_message(key, task))

    def _unknown_message(self, name: str, task: TaskType | str | None) -> str:
        """Build an error that helps rather than only refusing."""
        known = sorted(set(self._entries) | set(self._aliases))
        suggestions = difflib.get_close_matches(name, known, n=3, cutoff=0.6)
        message = f"No model is registered under {name!r}."
        if suggestions:
            message += f" Did you mean {suggestions}?"
        scope = f" for {TaskType.coerce(task).value}" if task is not None else ""
        message += f" Registered models{scope}: {sorted(self.available(task=task, include_unavailable=True))}."
        return message

    def available(
        self,
        *,
        task: TaskType | str | None = None,
        include_unavailable: bool = False,
    ) -> tuple[str, ...]:
        """Return canonical names, sorted.

        Args:
            task: Restrict to one task family.
            include_unavailable: Include models whose optional package is not
                installed. False by default, so the result lists what can
                actually be constructed right now.
        """
        return tuple(
            entry.canonical_name
            for entry in self.catalog(task=task, include_unavailable=include_unavailable)
        )

    def catalog(
        self,
        *,
        task: TaskType | str | None = None,
        include_unavailable: bool = True,
    ) -> tuple[ModelRegistration, ...]:
        """Return the registrations, sorted by canonical name.

        Args:
            task: Restrict to one task family.
            include_unavailable: Include models whose optional package is
                missing. True by default, because a catalog exists to show what
                the library knows about, not only what is installed.
        """
        wanted = TaskType.coerce(task) if task is not None else None
        entries = [
            entry
            for entry in self._entries.values()
            if (wanted is None or entry.task_type is wanted)
            and (include_unavailable or entry.is_available)
        ]
        return tuple(sorted(entries, key=lambda entry: entry.canonical_name))

    def aliases_of(self, canonical_name: str) -> tuple[str, ...]:
        """Return the aliases registered for one canonical name."""
        return self.resolve(canonical_name).aliases


_DEFAULT_REGISTRY = ModelRegistry()


def default_registry() -> ModelRegistry:
    """Return the registry holding the library's built-in models."""
    return _DEFAULT_REGISTRY


def register_model(
    *,
    aliases: tuple[str, ...] | list[str] = (),
    requires_package: str | None = None,
    registry: ModelRegistry | None = None,
):
    """Class decorator registering a strategy in a registry.

    Built-in strategies use this at module import, and
    :mod:`aidatasetkit.models` imports every strategy module explicitly. Whether
    a built-in model exists therefore never depends on which module a user
    happened to import first.

    Example:
        >>> from aidatasetkit.models import ModelRegistry, register_model
        >>> scratch = ModelRegistry()
        >>> # @register_model(aliases=("mine",), registry=scratch)
        >>> # class MyStrategy(ModelStrategy): ...
    """

    def decorate(strategy_type: type[ModelStrategy]) -> type[ModelStrategy]:
        target = registry if registry is not None else _DEFAULT_REGISTRY
        target.register(
            strategy_type, aliases=aliases, requires_package=requires_package
        )
        return strategy_type

    return decorate
