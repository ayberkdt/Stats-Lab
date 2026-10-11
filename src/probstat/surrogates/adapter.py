"""Adapters that let fitted surrogates stand in for an expensive forward model.

A fitted surrogate predicts one quantity of interest from a physical input
vector. :class:`SurrogateModel` bundles one surrogate per QoI behind the
:class:`probstat.uq.ForwardModel` call contract, and
:func:`surrogate_probabilistic_model` wraps it in a new
:class:`probstat.modeling.ProbabilisticModel` with the *same* joint input law.
Propagation, reliability, and sensitivity code therefore runs unchanged on the
emulator; only the surrogate's approximation error is added, and that error
must be judged separately (for example by its leave-one-out estimate).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from probstat.foundations.validation import require_finite_real
from probstat.modeling import ProbabilisticModel


@runtime_checkable
class Surrogate(Protocol):
    """Structural interface of a fitted scalar surrogate.

    References:
        [REF-SUDRET-2008], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

    """

    def predict(self, x: Sequence[float]) -> float:
        """Return the surrogate prediction at a physical input vector.

        References:
            [REF-SUDRET-2008]. No equation is evaluated by the protocol.

        """
        ...


@dataclass(frozen=True)
class SurrogateModel:
    """A forward model whose outputs come from fitted surrogates.

    Args:
        input_names: Input names in the order each surrogate expects.
        surrogates: Mapping from QoI name to fitted surrogate.

    References:
        [REF-SMITH-UQ-2024], [REF-SUDRET-2008].

    """

    input_names: tuple[str, ...]
    surrogates: Mapping[str, Surrogate]

    def __post_init__(self) -> None:
        """Validate names and surrogate interfaces.

        Raises:
            TypeError: If a surrogate lacks ``predict``.
            ValueError: If names are empty or repeated, or no surrogate is
                given.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        names = tuple(self.input_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("input_names must be non-empty and distinct")
        surrogates = dict(self.surrogates)
        if not surrogates:
            raise ValueError("at least one surrogate is required")
        for key, surrogate in surrogates.items():
            if not isinstance(surrogate, Surrogate):
                raise TypeError(f"surrogate for {key!r} lacks predict(x)")
        object.__setattr__(self, "input_names", names)
        object.__setattr__(self, "surrogates", MappingProxyType(surrogates))

    def __call__(self, parameters: Mapping[str, float]) -> Mapping[str, float]:
        """Evaluate every surrogate at one named input realization.

        Extra parameters (for example a model's fixed values) are ignored;
        the surrogates were trained with those values held fixed.

        Args:
            parameters: Mapping containing every input name.

        Returns:
            QoI-name mapping of surrogate predictions.

        Raises:
            KeyError: If an input name is missing.

        Equation:
            [EQ-UQ-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        x = tuple(
            require_finite_real(parameters[name], name=name)
            for name in self.input_names
        )
        return {
            name: surrogate.predict(x) for name, surrogate in self.surrogates.items()
        }


def surrogate_probabilistic_model(
    model: ProbabilisticModel,
    surrogates: Mapping[str, Surrogate],
    *,
    name: str | None = None,
) -> ProbabilisticModel:
    """Return a probabilistic model that evaluates surrogates instead of ``g``.

    The joint input law and the QoI declarations are shared with ``model``;
    only QoIs that have a surrogate are kept. Fixed parameters are not copied
    because the surrogates already embed them.

    Args:
        model: Original probabilistic model.
        surrogates: Mapping from QoI name to fitted surrogate.
        name: Optional label; defaults to ``"<model name>-surrogate"``.

    Returns:
        Probabilistic model usable by every workflow that accepts one.

    Raises:
        KeyError: If a surrogate names an unknown QoI.

    Equation:
        [EQ-UQ-001].

    References:
        [REF-SMITH-UQ-2024], [REF-SUDRET-2008].

    """
    quantities = tuple(model.quantity(key) for key in surrogates)
    adapter = SurrogateModel(model.inputs.names, surrogates)
    return ProbabilisticModel(
        model.inputs,
        adapter,
        quantities,
        name=name if name is not None else f"{model.name}-surrogate",
    )
