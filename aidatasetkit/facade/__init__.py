"""Facade: one guided path over the whole library.

:class:`AIDataFacade` is the high-level entry point -- load, look, prepare,
compare, choose, train, measure, predict. Every step delegates to the layer that
owns it, and the facade adds only sequence and state.

It is optional. Everything underneath is public and composable, and an analyst
who wants three preprocessors side by side, or the plan without the fit, should
reach for those components directly. The facade shortens the syntax of the common
path; it does not replace the architecture, and nothing in the library depends on
it.
"""

from aidatasetkit.facade.facade import AIDataFacade
from aidatasetkit.facade.state import Stage

__all__ = ["AIDataFacade", "Stage"]
