"""The model abstraction.

A :class:`ModelStrategy` answers exactly one question: *what estimator should be
constructed for this model?* It does not load, profile, clean, split, fit,
evaluate, compare, or predict. Those belong to other layers, and keeping them out
is what lets one strategy serve training, cross-validation, and comparison
without knowing any of them exist.

The contract is deliberately not tied to scikit-learn. ``build`` returns anything
satisfying :class:`~aidatasetkit.core.types.Estimator` -- ``fit``, ``predict``,
``get_params``, ``set_params`` -- so a future PyTorch or TensorFlow strategy can
return a thin adapter without a single change to the facade, the comparator, or
the evaluator. scikit-learn is today's implementation; it is not the abstraction.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from aidatasetkit.core.config import KitConfig
from aidatasetkit.core.exceptions import InvalidModelParameterError
from aidatasetkit.core.types import Estimator, TargetProfile
from aidatasetkit.models.capabilities import ModelCapabilities

__all__ = ["ModelStrategy"]


class ModelStrategy(ABC):
    """How AIDatasetKit constructs one model.

    Subclasses declare two class attributes and implement one method:

    ``name``
        The globally unique canonical name, such as ``"logistic_regression"``.
    ``capabilities``
        Verified facts about the algorithm, as a :class:`ModelCapabilities`.
    ``build(**params)``
        Returns a *new* estimator each time it is called.

    Args:
        config: Supplies shared defaults, currently ``random_state``. Strategies
            read the seed from here rather than hard-coding one, so a single
            configuration change makes every model reproducible together.
    """

    #: Globally unique canonical name. Declared by every concrete subclass.
    name: ClassVar[str]

    #: Verified facts about the algorithm. Declared by every concrete subclass.
    capabilities: ClassVar[ModelCapabilities]

    def __init__(self, config: KitConfig | None = None) -> None:
        self._config = config if config is not None else KitConfig()

    @property
    def config(self) -> KitConfig:
        """The configuration this strategy takes its defaults from."""
        return self._config

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"

    def default_params(self) -> dict[str, Any]:
        """Return the parameters used when the caller supplies none.

        Defaults are conservative and deterministic. Nothing here is tuned, and
        nothing is chosen by looking at the data.
        """
        return {}

    def resolve_params(self, **overrides: Any) -> dict[str, Any]:
        """Merge caller overrides over the defaults."""
        return {**self.default_params(), **overrides}

    def validate_for(self, target: TargetProfile) -> None:
        """Check this model against a resolved target, before any fitting.

        Raises:
            IncompatibleModelError: If the model cannot serve this target.
        """
        self.capabilities.validate_for(target)

    @abstractmethod
    def build(self, **params: Any) -> Estimator:
        """Construct a new, unfitted estimator.

        Every call returns a fresh instance. Returning a shared object would let
        one run's fitted state leak into the next, which is the kind of defect
        that shows up as an inexplicably good validation score.

        Args:
            **params: Parameters for the underlying estimator, overriding
                :meth:`default_params`.

        Returns:
            An unfitted object satisfying the
            :class:`~aidatasetkit.core.types.Estimator` protocol.

        Raises:
            InvalidModelParameterError: If a parameter is not accepted by the
                underlying estimator.
        """

    @staticmethod
    def _construct(estimator_type: type, params: dict[str, Any], model_name: str) -> Any:
        """Instantiate an estimator, turning an unknown parameter into a clear error.

        scikit-learn answers an unknown keyword with a bare ``TypeError`` naming
        one argument; listing what the estimator does accept turns a guessing game
        into a correction.

        The unknown names are identified from the signature *before* construction
        rather than by catching ``TypeError`` afterwards. A constructor raises
        ``TypeError`` for other reasons too -- a missing required argument, most
        often -- and reporting one of those as "does not accept the parameter(s)
        []" would name no parameter, blame the caller for the wrong thing, and
        discard the real cause. Anything not attributable to a bad parameter name
        is left to propagate untouched.

        Invalid *values* are the estimator's business: scikit-learn checks them at
        fit time against its own constraints, and duplicating that here would only
        let the two disagree.

        Note:
            An estimator whose ``__init__`` ends in ``**kwargs`` accepts any name,
            so no name can be called unknown and a typo cannot be detected here.
            Neither built-in model has such a signature.
        """
        accepted = inspect.signature(estimator_type).parameters
        takes_any = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in accepted.values()
        )
        unknown = sorted(set(params) - set(accepted))
        if unknown and not takes_any:
            raise InvalidModelParameterError(
                f"{model_name!r} does not accept the parameter(s) {unknown}. "
                f"{estimator_type.__name__} accepts: {', '.join(sorted(accepted))}."
            )
        return estimator_type(**params)
