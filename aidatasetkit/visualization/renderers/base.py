"""The renderer seam.

A renderer turns a :class:`~aidatasetkit.visualization.types.PreparedChart` into
whatever its library draws with. The planning layer knows only this protocol, so
adding a backend means adding a module and one registry entry -- no advisor, no
scoring rule, and no domain type changes.

**Nothing here imports a plotting library.** Renderer modules are resolved by
name, at the moment a render is actually requested. That is what lets
recommendation, scoring, and serialisation run in an environment where matplotlib
is not installed, which is the common case on a server that only produces plans.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, runtime_checkable

from aidatasetkit.core.exceptions import MissingDependencyError, UnsupportedChartError
from aidatasetkit.visualization.types import ChartType, PreparedChart

__all__ = [
    "VisualizationRenderer",
    "RendererRegistration",
    "RENDERERS",
    "get_renderer",
    "available_renderers",
]


@runtime_checkable
class VisualizationRenderer(Protocol):
    """What a rendering backend must provide.

    Structural, like the estimator contract in :mod:`aidatasetkit.core.types`: a
    backend satisfies it by having the methods, not by inheriting anything.
    """

    name: str

    def supports(self, chart_type: ChartType) -> bool:
        """Whether this backend can draw ``chart_type``."""
        ...

    def render(self, prepared: PreparedChart, figure: Any = None) -> Any:
        """Draw the chart and return the backend's own figure object.

        A caller may supply ``figure`` to have the chart drawn into a figure it
        already owns, which is how display works without the renderer reaching
        into the plotting library's global state.
        """
        ...


@dataclass(frozen=True, slots=True)
class RendererRegistration:
    """How to reach one renderer, and what it needs installed."""

    name: str
    module: str
    attribute: str
    requires_package: str
    extra: str

    def install_hint(self) -> str:
        """The command that would make this renderer usable."""
        return f"pip install aidatasetkit[{self.extra}]"


#: Known renderers. The module is imported only when one is requested.
RENDERERS: dict[str, RendererRegistration] = {
    "matplotlib": RendererRegistration(
        name="matplotlib",
        module="aidatasetkit.visualization.renderers.matplotlib_renderer",
        attribute="MatplotlibRenderer",
        requires_package="matplotlib",
        extra="viz",
    )
}


def available_renderers() -> tuple[str, ...]:
    """Return the names of every known renderer, installed or not."""
    return tuple(sorted(RENDERERS))


def get_renderer(name: str) -> VisualizationRenderer:
    """Construct the renderer registered under ``name``.

    Args:
        name: A key of :data:`RENDERERS`.

    Returns:
        A ready renderer instance.

    Raises:
        UnsupportedChartError: If no renderer is registered under that name.
        MissingDependencyError: If its plotting library is not installed, with the
            command that would fix it.
    """
    registration = RENDERERS.get(name)
    if registration is None:
        raise UnsupportedChartError(
            f"No renderer named {name!r} is registered. Available renderers: "
            f"{list(available_renderers())}."
        )

    try:
        module = import_module(registration.module)
    except ImportError as error:
        raise MissingDependencyError(
            f"The {registration.name!r} renderer requires the "
            f"{registration.requires_package!r} package, which is not installed. "
            f"Install it with: {registration.install_hint()}. Recommendation and "
            "planning do not need it."
        ) from error

    return getattr(module, registration.attribute)()
