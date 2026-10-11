"""Random-field inputs: a truncated KL expansion as named model inputs.

A Gaussian random field enters a forward model through its truncated
Karhunen-Loeve coefficients ``xi_1..xi_M``, which are independent standard
normals. :class:`RandomFieldInput` declares those coefficients as ordinary
:class:`~probstat.modeling.RandomInput` objects named ``<prefix>_1`` ..
``<prefix>_M`` and maps a parameter mapping back to field values, optionally
through a pointwise transform (``math.exp`` gives a log-normal field). A user
forward model therefore consumes a whole random field, and the resulting
:class:`~probstat.modeling.ProbabilisticModel` works unchanged with
propagation, reliability, and sensitivity analysis.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from probstat.distributions import Normal
from probstat.linalg import Vector
from probstat.processes import KarhunenLoeveExpansion

from .inputs import JointInput, RandomInput


@dataclass(frozen=True)
class RandomFieldInput:
    """Standard-normal KL coefficients of a field, exposed as model inputs.

    Args:
        expansion: Truncated KL expansion of the underlying Gaussian field.
        prefix: Non-empty name prefix; inputs are ``<prefix>_1`` ..
            ``<prefix>_M``.
        transform: Optional pointwise map applied to the Gaussian field
            (``math.exp`` for a log-normal field); ``None`` keeps it Gaussian.
        description: Optional physical meaning of the field.

    References:
        [REF-GHANEM-SPANOS-1991, Chapter 2],
        [REF-SUDRET-DERKIUREGHIAN-2000, Chapter 2].

    """

    expansion: KarhunenLoeveExpansion
    prefix: str = "xi"
    transform: Callable[[float], float] | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the expansion, prefix, and transform.

        Raises:
            TypeError: If a component has the wrong type.
            ValueError: If the prefix is empty.

        References:
            Project validation policy. No equation is evaluated.

        """
        if not isinstance(self.expansion, KarhunenLoeveExpansion):
            raise TypeError("expansion must be a KarhunenLoeveExpansion")
        if not isinstance(self.prefix, str) or not self.prefix.strip():
            raise ValueError("prefix must be a non-empty string")
        if self.transform is not None and not callable(self.transform):
            raise TypeError("transform must be callable or None")

    @property
    def names(self) -> tuple[str, ...]:
        """Return the coefficient input names ``<prefix>_1..<prefix>_M``.

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return tuple(f"{self.prefix}_{k + 1}" for k in range(self.expansion.terms))

    @property
    def dimension(self) -> int:
        """Return the number of KL coefficients ``M``.

        References:
            [REF-GHANEM-SPANOS-1991, Chapter 2].

        """
        return self.expansion.terms

    @property
    def inputs(self) -> tuple[RandomInput, ...]:
        """Return one independent standard-normal input per KL coefficient.

        References:
            [REF-GHANEM-SPANOS-1991, Chapter 2].

        """
        label = self.description or "random field"
        return tuple(
            RandomInput(name, Normal(0.0, 1.0), f"KL coefficient {k + 1} of {label}")
            for k, name in enumerate(self.names)
        )

    def joint_input(self, *others: RandomInput) -> JointInput:
        """Return the independent joint law of the coefficients and others.

        Args:
            *others: Additional independent scalar inputs, placed after the
                KL coefficients.

        Returns:
            Joint input with identity copula.

        References:
            [REF-NELSEN-2006, Section 2.10].

        """
        return JointInput.independent(*self.inputs, *others)

    def coefficients(self, parameters: Mapping[str, float]) -> Vector:
        """Extract ``xi_1..xi_M`` from a forward-model parameter mapping.

        Args:
            parameters: Mapping containing every coefficient name.

        Returns:
            Coefficients in mode order.

        Raises:
            KeyError: If a coefficient is missing.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        missing = [name for name in self.names if name not in parameters]
        if missing:
            raise KeyError(f"missing KL coefficients: {missing}")
        return tuple(float(parameters[name]) for name in self.names)

    def _apply(self, value: float) -> float:
        """Apply the optional pointwise transform.

        Args:
            value: Gaussian field value.

        Returns:
            Transformed value.

        Raises:
            ValueError: If the transform returns a non-finite value.

        Equation:
            [EQ-MOD-RF-001].

        References:
            [REF-GHANEM-SPANOS-1991, Chapter 2].

        """
        if self.transform is None:
            return value
        result = float(self.transform(value))
        if not math.isfinite(result):
            raise ValueError("field transform produced a non-finite value")
        return result

    def field(self, parameters: Mapping[str, float]) -> Vector:
        """Return the (transformed) field at the expansion nodes.

        Args:
            parameters: Mapping containing every coefficient name.

        Returns:
            ``T(m + sum_k sqrt(lambda_k) xi_k phi_k(x_i))`` per node.

        Equation:
            [EQ-MOD-RF-001], [EQ-PROC-KL-001].

        References:
            [REF-GHANEM-SPANOS-1991, Chapter 2].

        """
        gaussian = self.expansion.realize(self.coefficients(parameters))
        return tuple(self._apply(value) for value in gaussian)

    def field_at(
        self, parameters: Mapping[str, float], location: float | Sequence[float]
    ) -> float:
        """Return the (transformed) field at an arbitrary location.

        Args:
            parameters: Mapping containing every coefficient name.
            location: Scalar (1-D) or coordinate tuple.

        Returns:
            Field value from Nystrom-interpolated eigenfunctions.

        Equation:
            [EQ-MOD-RF-001], [EQ-PROC-KL-005].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Section 4.3].

        """
        coefficients = self.coefficients(parameters)
        return self._apply(self.expansion.realize_at(coefficients, location))


def karhunen_loeve_inputs(
    expansion: KarhunenLoeveExpansion,
    prefix: str = "xi",
    *,
    transform: Callable[[float], float] | None = None,
) -> tuple[tuple[RandomInput, ...], Callable[[Mapping[str, float]], Vector]]:
    """Return KL coefficient inputs and the parameter-to-field map.

    Args:
        expansion: Truncated KL expansion.
        prefix: Coefficient name prefix.
        transform: Optional pointwise field transform.

    Returns:
        Standard-normal inputs ``<prefix>_1..<prefix>_M`` and a function
        mapping a parameter mapping to field values at the nodes.

    Equation:
        [EQ-MOD-RF-001].

    References:
        [REF-GHANEM-SPANOS-1991, Chapter 2].

    """
    field = RandomFieldInput(expansion, prefix, transform)
    return field.inputs, field.field
