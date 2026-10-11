"""The probabilistic model: one object shared by every UQ workflow.

A :class:`ProbabilisticModel` binds a joint input law, a named forward model
``Y = g(X)``, and the quantities of interest. Propagation, sensitivity,
reliability, calibration, surrogate fitting, and risk analysis all accept this
object, so a study defines its model exactly once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from probstat.linalg import Vector, as_vector
from probstat.uq import ForwardModel, QuantityOfInterest, evaluate_model

from .inputs import JointInput


@dataclass
class EvaluationCounter:
    """Mutable count of complete forward-model evaluations.

    Args:
        count: Number of evaluations recorded so far.

    References:
        [REF-SMITH-UQ-2024, Chapter 3].

    """

    count: int = 0

    def add(self, amount: int = 1) -> None:
        """Increase the evaluation count.

        Args:
            amount: Non-negative number of new evaluations.

        Raises:
            ValueError: If ``amount`` is negative.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        if amount < 0:
            raise ValueError("amount must be non-negative")
        self.count += amount


@dataclass(frozen=True)
class ProbabilisticModel:
    """Joint input law, forward model, and quantities of interest.

    Args:
        inputs: Joint law of the named model inputs.
        model: Deterministic named-input/named-output forward map.
        quantities: Non-empty distinctly named quantities of interest.
        name: Short study label.
        fixed: Optional deterministic parameters passed to every call.

    References:
        [REF-SMITH-UQ-2024, Chapters 1-4], [REF-SUDRET-2007, Chapter 1].

    """

    inputs: JointInput
    model: ForwardModel
    quantities: tuple[QuantityOfInterest, ...]
    name: str = "model"
    fixed: Mapping[str, float] = field(default_factory=dict)
    counter: EvaluationCounter = field(
        default_factory=EvaluationCounter, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        """Validate the model contract once at construction.

        Raises:
            TypeError: If components have invalid types.
            ValueError: If quantities are empty or repeated, or a fixed
                parameter shadows a random input.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        if not isinstance(self.inputs, JointInput):
            raise TypeError("inputs must be a JointInput")
        if not callable(self.model):
            raise TypeError("model must be callable")
        quantities = tuple(self.quantities)
        if not quantities:
            raise ValueError("at least one quantity of interest is required")
        for index, quantity in enumerate(quantities):
            if not isinstance(quantity, QuantityOfInterest):
                raise TypeError(f"quantities[{index}] must be a QuantityOfInterest")
        names = [quantity.name for quantity in quantities]
        if len(set(names)) != len(names):
            raise ValueError("quantity names must be distinct")
        fixed = dict(self.fixed)
        overlap = set(fixed) & set(self.inputs.names)
        if overlap:
            raise ValueError(
                f"fixed parameters shadow random inputs: {sorted(overlap)}"
            )
        object.__setattr__(self, "quantities", quantities)
        object.__setattr__(self, "fixed", MappingProxyType(fixed))

    @property
    def dimension(self) -> int:
        """Return the number of random inputs.

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return self.inputs.dimension

    @property
    def quantity_names(self) -> tuple[str, ...]:
        """Return QoI names in declaration order.

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return tuple(quantity.name for quantity in self.quantities)

    @property
    def evaluations(self) -> int:
        """Return the number of forward-model calls made through this object.

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return self.counter.count

    def quantity(self, name: str) -> QuantityOfInterest:
        """Return the declared quantity of interest with a given name.

        Args:
            name: QoI name.

        Returns:
            Matching quantity.

        Raises:
            KeyError: If no quantity has that name.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        for quantity in self.quantities:
            if quantity.name == name:
                return quantity
        raise KeyError(f"unknown quantity of interest {name!r}")

    def evaluate(self, x: Sequence[float] | Mapping[str, float]) -> Mapping[str, float]:
        """Evaluate every QoI at one physical input realization.

        Args:
            x: Physical input vector in model order or a name mapping.

        Returns:
            Read-only QoI-name mapping.

        Equation:
            [EQ-UQ-001].

        References:
            [REF-SMITH-UQ-2024, Chapters 1-3].

        """
        values: dict[str, float] = (
            dict(x) if isinstance(x, Mapping) else dict(self.inputs.as_mapping(x))
        )
        values.update(self.fixed)
        self.counter.add()
        return evaluate_model(self.model, values, self.quantities)

    def evaluate_many(
        self, points: Iterable[Sequence[float]]
    ) -> Mapping[str, tuple[float, ...]]:
        """Evaluate every QoI at many physical input vectors.

        Args:
            points: Physical input vectors.

        Returns:
            Read-only mapping from QoI name to ordered outputs.

        Equation:
            [EQ-UQ-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        collected: dict[str, list[float]] = {name: [] for name in self.quantity_names}
        for point in points:
            outputs = self.evaluate(point)
            for name in self.quantity_names:
                collected[name].append(outputs[name])
        return MappingProxyType({k: tuple(v) for k, v in collected.items()})

    def scalar_function(self, quantity: str) -> Callable[[Sequence[float]], float]:
        """Return ``x -> g_q(x)`` on physical input vectors.

        Args:
            quantity: QoI name.

        Returns:
            Scalar callable that counts evaluations through this model.

        Raises:
            KeyError: If the quantity is unknown.

        Equation:
            [EQ-UQ-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        self.quantity(quantity)

        def function(x: Sequence[float]) -> float:
            """Evaluate one QoI at a physical input vector.

            References:
                [EQ-UQ-001], [REF-SMITH-UQ-2024, Chapter 3].

            """
            return self.evaluate(as_vector(x, name="x"))[quantity]

        return function

    def standard_space_function(
        self, quantity: str
    ) -> Callable[[Sequence[float]], float]:
        """Return ``u -> g_q(T^-1(u))`` on independent standard normals.

        Args:
            quantity: QoI name.

        Returns:
            Scalar callable used by FORM, SORM, and subset simulation.

        Equation:
            [EQ-UQ-001], [EQ-MOD-NATAF-001].

        References:
            [REF-MELCHERS-BECK-2018, Chapter 5].

        """
        physical = self.scalar_function(quantity)

        def function(u: Sequence[float]) -> float:
            """Evaluate one QoI at a standard-normal point.

            References:
                [EQ-MOD-NATAF-001], [REF-MELCHERS-BECK-2018, Chapter 5].

            """
            return physical(self.inputs.from_standard_normal(u))

        return function

    def unit_function(self, quantity: str) -> Callable[[Sequence[float]], float]:
        """Return ``p -> g_q(F^-1(p))`` on the unit hypercube.

        Args:
            quantity: QoI name.

        Returns:
            Scalar callable on ``(0, 1)^d`` used by pick-freeze Sobol designs.

        Equation:
            [EQ-UQ-001], [EQ-SAMP-INV-001].

        References:
            [REF-SALTELLI-2010].

        """
        physical = self.scalar_function(quantity)

        def function(point: Sequence[float]) -> float:
            """Evaluate one QoI at a unit-hypercube point.

            References:
                [EQ-SAMP-INV-001], [REF-SALTELLI-2010].

            """
            return physical(self.inputs.from_unit(point))

        return function

    def nominal(self) -> Vector:
        """Return the vector of marginal means.

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return self.inputs.means()
