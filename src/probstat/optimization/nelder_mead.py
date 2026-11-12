"""Derivative-free Nelder-Mead simplex minimization."""

from __future__ import annotations

import math
from collections.abc import Sequence

from probstat.linalg import Vector, as_vector, norm

from ._common import ScalarFunction
from .bounds import Bounds, BoxTransform
from .results import OptimizationResult


def minimize_nelder_mead(
    function: ScalarFunction,
    x0: Sequence[float],
    *,
    initial_step: float = 0.1,
    tolerance: float = 1e-10,
    max_evaluations: int = 20_000,
    bounds: Bounds | None = None,
) -> OptimizationResult:
    """Minimize a function without derivatives by the Nelder-Mead simplex.

    With ``bounds`` the simplex moves in the unconstrained coordinates of
    :class:`BoxTransform`, so every evaluated point lies inside the box; the
    returned ``x`` is mapped back to the original coordinates.

    Args:
        function: Scalar objective; non-finite values are treated as ``+inf``.
        x0: Finite starting point.
        initial_step: Relative edge length of the starting simplex.
        tolerance: Stop when the simplex value spread falls below this value.
        max_evaluations: Maximum objective evaluations.
        bounds: Optional ``(low, high)`` per coordinate (``None`` or infinite
            for a missing bound).

    Returns:
        Minimization result; ``gradient_norm`` is ``nan`` (not computed).

    Raises:
        ValueError: If ``x0`` is empty or outside the bounds.

    Equation:
        [EQ-OPT-NM-001], [EQ-OPT-BOX-001].

    References:
        [REF-NELDER-MEAD-1965], [REF-NOCEDAL-WRIGHT-2006, Section 9.5],
        [REF-JAMES-ROOS-1975].

    """
    start = as_vector(x0, name="x0")
    if not start:
        raise ValueError("x0 must be non-empty")
    if bounds is not None:
        transform = BoxTransform.from_bounds(bounds, len(start))
        inner = minimize_nelder_mead(
            lambda z: function(transform.to_outer(z)),
            transform.to_inner(start),
            initial_step=initial_step,
            tolerance=tolerance,
            max_evaluations=max_evaluations,
        )
        return OptimizationResult(
            transform.to_outer(inner.x),
            inner.value,
            inner.gradient_norm,
            inner.iterations,
            inner.evaluations,
            inner.converged,
            inner.message,
        )
    evaluations = 0

    def f(point: Vector) -> float:
        """Evaluate the objective, mapping failures to ``+inf``.

        References:
            [REF-NELDER-MEAD-1965]. No equation is evaluated.

        """
        nonlocal evaluations
        evaluations += 1
        try:
            value = float(function(point))
        except (ValueError, OverflowError, ZeroDivisionError):
            return math.inf
        return value if math.isfinite(value) else math.inf

    size = len(start)
    simplex = [start]
    for index in range(size):
        vertex = list(start)
        vertex[index] += initial_step * max(1.0, abs(start[index]))
        simplex.append(tuple(vertex))
    values = [f(vertex) for vertex in simplex]
    iterations = 0
    while evaluations < max_evaluations:
        order = sorted(range(size + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        if abs(values[-1] - values[0]) <= tolerance * max(1.0, abs(values[0])):
            spread = max(
                norm(tuple(a - b for a, b in zip(v, simplex[0], strict=True)))
                for v in simplex
            )
            if spread <= math.sqrt(tolerance) * max(1.0, norm(simplex[0])):
                return OptimizationResult(
                    simplex[0],
                    values[0],
                    math.nan,
                    iterations,
                    evaluations,
                    True,
                    "simplex tolerance met",
                )
        iterations += 1
        centroid = tuple(
            math.fsum(vertex[j] for vertex in simplex[:-1]) / size for j in range(size)
        )

        worst = simplex[-1]

        def along(
            coefficient: float,
            centroid: Vector = centroid,
            worst: Vector = worst,
        ) -> Vector:
            """Return ``centroid + coefficient (centroid - worst)``.

            References:
                [EQ-OPT-NM-001], [REF-NELDER-MEAD-1965].

            """
            return tuple(
                c + coefficient * (c - w) for c, w in zip(centroid, worst, strict=True)
            )

        reflected = along(1.0)
        reflected_value = f(reflected)
        if values[0] <= reflected_value < values[-2]:
            simplex[-1], values[-1] = reflected, reflected_value
            continue
        if reflected_value < values[0]:
            expanded = along(2.0)
            expanded_value = f(expanded)
            if expanded_value < reflected_value:
                simplex[-1], values[-1] = expanded, expanded_value
            else:
                simplex[-1], values[-1] = reflected, reflected_value
            continue
        contracted = along(0.5 if reflected_value < values[-1] else -0.5)
        contracted_value = f(contracted)
        if contracted_value < min(reflected_value, values[-1]):
            simplex[-1], values[-1] = contracted, contracted_value
            continue
        best = simplex[0]
        simplex = [best] + [
            tuple(b + 0.5 * (v - b) for b, v in zip(best, vertex, strict=True))
            for vertex in simplex[1:]
        ]
        values = [values[0]] + [f(vertex) for vertex in simplex[1:]]
    best_index = min(range(size + 1), key=lambda i: values[i])
    return OptimizationResult(
        simplex[best_index],
        values[best_index],
        math.nan,
        iterations,
        evaluations,
        False,
        "evaluation limit reached",
    )
