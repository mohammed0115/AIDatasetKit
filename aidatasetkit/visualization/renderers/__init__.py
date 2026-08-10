"""Rendering backends.

Importing this package pulls in the protocol and the registry, and no plotting
library. Concrete renderer modules are imported by
:func:`~aidatasetkit.visualization.renderers.base.get_renderer` at the moment one
is asked for, so a missing optional dependency becomes an actionable error at
render time rather than an ``ImportError`` at import time.
"""

from aidatasetkit.visualization.renderers.base import (
    RENDERERS,
    RendererRegistration,
    VisualizationRenderer,
    available_renderers,
    get_renderer,
)

__all__ = [
    "RENDERERS",
    "RendererRegistration",
    "VisualizationRenderer",
    "available_renderers",
    "get_renderer",
]
