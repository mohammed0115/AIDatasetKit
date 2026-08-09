"""How a requested model becomes a fresh estimator.

The factory performs six steps and owns none of them:

1. Resolve the name or alias through the registry.
2. Refuse a model whose optional package is missing.
3. Refuse a model that cannot serve the resolved target.
4. Instantiate the strategy.
5. Ask it to build a new estimator.
6. Return it.

There is no conditional over model names anywhere in this module. Model discovery
belongs to the registry, and adding a model means registering it, not editing a
branch here.

This is not model selection. The factory constructs exactly what was asked for
and never ranks, scores, or chooses.
"""

from __future__ import annotations

from typing import Any

from aidatasetkit.core.config import KitConfig
from aidatasetkit.core.types import Estimator, TargetProfile, TaskType
from aidatasetkit.models.base import ModelStrategy
from aidatasetkit.models.registry import (
    ModelRegistration,
    ModelRegistry,
    default_registry,
)

__all__ = ["ModelFactory"]


class ModelFactory:
    """Constructs estimators from registered model strategies.

    Every method is a classmethod taking an optional ``registry``, so the common
    call reads as in the documentation while tests can still work against an
    isolated registry.

    Note:
        ``name``, ``task``, ``target``, ``config``, and ``registry`` are reserved
        by these signatures; everything else in ``**params`` reaches the
        estimator. An estimator parameter sharing one of those names cannot be
        passed as a keyword -- none of the models this library registers has one,
        and the collision would surface as an ordinary ``TypeError`` from Python
        rather than silently changing behaviour.

    Example:
        >>> from aidatasetkit.models import ModelFactory
        >>> estimator = ModelFactory.create("logistic_regression", C=0.5)
        >>> estimator.get_params()["C"]
        0.5
    """

    @classmethod
    def create(
        cls,
        name: str,
        *,
        task: TaskType | str | None = None,
        target: TargetProfile | None = None,
        config: KitConfig | None = None,
        registry: ModelRegistry | None = None,
        **params: Any,
    ) -> Estimator:
        """Build a new, unfitted estimator for the named model.

        Args:
            name: Canonical name or alias.
            task: Restricts resolution to one task family. Narrows an alias
                shared across families, and contradicts a canonical name from
                another family rather than being ignored. Defaults to
                ``target.task_type`` when a target is supplied.
            target: When given, the model is checked against it before anything
                is constructed, so an incompatible choice fails here rather than
                inside ``fit`` several steps later.
            config: Supplies shared defaults such as ``random_state``.
            registry: Registry to resolve against. Defaults to the built-in one.
            **params: Parameters for the underlying estimator.

        Returns:
            A fresh, unfitted estimator.

        Raises:
            UnknownModelError: If nothing is registered under ``name``.
            AmbiguousModelAliasError: If an alias matches several models.
            MissingDependencyError: If the model's optional package is missing.
            IncompatibleModelError: If the model cannot serve ``target``.
            InvalidModelParameterError: If a parameter is not accepted.
        """
        strategy = cls.strategy(
            name, task=task, target=target, config=config, registry=registry
        )
        return strategy.build(**params)

    @classmethod
    def strategy(
        cls,
        name: str,
        *,
        task: TaskType | str | None = None,
        target: TargetProfile | None = None,
        config: KitConfig | None = None,
        registry: ModelRegistry | None = None,
    ) -> ModelStrategy:
        """Resolve and instantiate a strategy without building an estimator.

        Useful when the caller needs the model's capabilities -- to assemble a
        preprocessor, say -- before deciding to construct anything.
        """
        # A caller who supplied a target has already said which task family they
        # mean. Making them repeat it as task= merely to disambiguate an alias
        # would be demanding information they have already given.
        if task is None and target is not None:
            task = target.task_type
        entry = cls.registration(name, task=task, registry=registry)
        entry.require_available()
        strategy = entry.strategy_type(config=config)
        if target is not None:
            strategy.validate_for(target)
        return strategy

    @classmethod
    def registration(
        cls,
        name: str,
        *,
        task: TaskType | str | None = None,
        registry: ModelRegistry | None = None,
    ) -> ModelRegistration:
        """Resolve a name to its registry entry, without constructing anything."""
        return cls._registry(registry).resolve(name, task=task)

    @classmethod
    def available(
        cls,
        *,
        task: TaskType | str | None = None,
        include_unavailable: bool = False,
        registry: ModelRegistry | None = None,
    ) -> tuple[str, ...]:
        """Return the canonical names this factory can construct."""
        return cls._registry(registry).available(
            task=task, include_unavailable=include_unavailable
        )

    @classmethod
    def catalog(
        cls,
        *,
        task: TaskType | str | None = None,
        include_unavailable: bool = True,
        registry: ModelRegistry | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return serialisable metadata for every known model.

        Suitable for a notebook table, a CLI listing, or generated documentation.
        Contains no estimator objects.
        """
        return tuple(
            entry.to_dict()
            for entry in cls._registry(registry).catalog(
                task=task, include_unavailable=include_unavailable
            )
        )

    @classmethod
    def compatible_with(
        cls,
        target: TargetProfile,
        *,
        include_unavailable: bool = False,
        registry: ModelRegistry | None = None,
    ) -> tuple[str, ...]:
        """Return the models that can serve ``target``.

        A filter, not a recommendation: the names come back in alphabetical
        order, with nothing ranked and nothing preferred.
        """
        return tuple(
            entry.canonical_name
            for entry in cls._registry(registry).catalog(
                task=target.task_type, include_unavailable=include_unavailable
            )
            if entry.capabilities.is_compatible_with(target)
        )

    @staticmethod
    def _registry(registry: ModelRegistry | None) -> ModelRegistry:
        return registry if registry is not None else default_registry()
