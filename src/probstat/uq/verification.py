"""Numerical verification primitives for refinement and error accounting.

Besides the error budget and the uniform-ratio observed order, the module
implements solution verification from systematic grid refinement:

* :func:`grid_convergence_index` — the three-grid procedure of Celik et al.
  (2008), including non-uniform refinement ratios (fixed-point iteration for
  the observed order), oscillatory-convergence detection, and Roache's
  asymptotic-range check;
* :func:`least_squares_convergence` — a simplified version of the
  least-squares observed-order fit of Eca and Hoekstra (2014) for four or more
  grids; and
* :class:`SolutionVerification` / :func:`verify_solution` — a record that
  turns either analysis into a numerical-uncertainty statement ``phi_1 +/- U``
  that validation and prediction code can consume as ``u_num``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_finite_real,
)
from probstat.optimization import brent_root


@dataclass(frozen=True, slots=True)
class NumericalErrorBudget:
    """Non-negative absolute bounds for major numerical error sources.

    The total is an additive conservative bound, not a root-sum-square estimate;
    no independence assumption is silently introduced.

    Args:
        truncation: Bound attributed to finite-series or iteration truncation.
        roundoff: Bound attributed to finite-precision arithmetic.
        discretization: Bound attributed to spatial or temporal discretization.
        iterative: Bound attributed to incomplete nonlinear/linear solves.

    References:
        [REF-OBERKAMPF-ROY-2025, Chapters 6-8].

    """

    truncation: float = 0.0
    roundoff: float = 0.0
    discretization: float = 0.0
    iterative: float = 0.0

    def __post_init__(self) -> None:
        """Validate and normalize every absolute error bound.

        Raises:
            TypeError: If a component is not real.
            ValueError: If a component is negative or non-finite.

        References:
            [EQ-VV-003], [REF-OBERKAMPF-ROY-2025, Chapters 6-8].

        """
        for name in ("truncation", "roundoff", "discretization", "iterative"):
            value = require_non_negative_finite_real(getattr(self, name), name=name)
            object.__setattr__(self, name, value)

    @property
    def total_bound(self) -> float:
        """Return the conservative sum of componentwise absolute bounds.

        Equation:
            [EQ-VV-003].

        References:
            [REF-OBERKAMPF-ROY-2025, Chapters 6-8].

        """
        return math.fsum(
            (self.truncation, self.roundoff, self.discretization, self.iterative)
        )


def observed_order_of_accuracy(
    coarse: float,
    medium: float,
    fine: float,
    *,
    refinement_ratio: float = 2.0,
) -> float:
    """Estimate convergence order from three uniformly refined solutions.

    Args:
        coarse: QoI evaluated on the coarsest resolution.
        medium: QoI evaluated after one uniform refinement.
        fine: QoI evaluated after a second uniform refinement.
        refinement_ratio: Resolution ratio ``r>1`` shared by both refinements.

    Returns:
        Observed asymptotic order ``p``.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument is non-finite, ``r<=1``, or consecutive
            solution differences are zero.

    Equation:
        [EQ-VV-001].

    References:
        [REF-ROACHE-1998, Chapters 4-5].

    """
    coarse_value = require_finite_real(coarse, name="coarse")
    medium_value = require_finite_real(medium, name="medium")
    fine_value = require_finite_real(fine, name="fine")
    ratio = require_finite_real(refinement_ratio, name="refinement_ratio")
    if ratio <= 1.0:
        raise ValueError("refinement_ratio must be greater than one")
    coarse_difference = abs(coarse_value - medium_value)
    fine_difference = abs(medium_value - fine_value)
    if coarse_difference == 0.0 or fine_difference == 0.0:
        raise ValueError("consecutive refinement differences must be non-zero")
    return math.log(coarse_difference / fine_difference) / math.log(ratio)


def richardson_extrapolation(
    medium: float,
    fine: float,
    *,
    refinement_ratio: float,
    order: float,
) -> float:
    """Extrapolate a refined QoI toward zero discretization size.

    Args:
        medium: QoI at resolution ``h``.
        fine: QoI at resolution ``h/refinement_ratio``.
        refinement_ratio: Uniform refinement ratio ``r>1``.
        order: Positive observed or theoretical convergence order.

    Returns:
        Richardson-extrapolated zero-size estimate.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument is non-finite, ``r<=1``, or ``order<=0``.
        OverflowError: If exponentiation or the extrapolated result is non-finite.

    Equation:
        [EQ-VV-002].

    References:
        [REF-ROACHE-1998, Chapters 4-5].

    """
    medium_value = require_finite_real(medium, name="medium")
    fine_value = require_finite_real(fine, name="fine")
    ratio = require_finite_real(refinement_ratio, name="refinement_ratio")
    checked_order = require_finite_real(order, name="order")
    if ratio <= 1.0:
        raise ValueError("refinement_ratio must be greater than one")
    if checked_order <= 0.0:
        raise ValueError("order must be positive")
    try:
        denominator = math.pow(ratio, checked_order) - 1.0
    except OverflowError as error:
        raise OverflowError("refinement factor overflowed finite arithmetic") from error
    if not math.isfinite(denominator):
        raise OverflowError("refinement factor overflowed finite arithmetic")
    result = fine_value + (fine_value - medium_value) / denominator
    if not math.isfinite(result):
        raise OverflowError("Richardson extrapolation is not finite")
    return result


DEFAULT_SAFETY_FACTOR = 1.25
"""Roache's factor of safety for three or more grids with an observed order."""

UNRELIABLE_SAFETY_FACTOR = 3.0
"""Roache's conservative factor of safety when the order estimate is unreliable."""

_ORDER_EXPONENT_LIMIT = 700.0
_NORMAL_95 = 1.959963984540054


class ConvergenceType(str, Enum):
    """Qualitative behaviour of a QoI under systematic grid refinement.

    With grids ordered fine (1), medium (2), coarse (3) and differences
    ``eps21 = phi2 - phi1`` and ``eps32 = phi3 - phi2``, the ratio
    ``R = eps32 / eps21`` classifies the sequence ([EQ-VV-GCI-006]).

    References:
        [REF-CELIK-ETAL-2008], [REF-ROY-2005].

    """

    MONOTONE = "monotone"
    OSCILLATORY = "oscillatory"
    DIVERGENT = "divergent"


def representative_spacing(
    cells: float, *, dimension: int, volume: float = 1.0
) -> float:
    """Return the representative grid size ``h = (V / N)^(1/d)``.

    Args:
        cells: Positive number of cells (or degrees of freedom) ``N``.
        dimension: Spatial dimension ``d`` in ``{1, 2, 3}``.
        volume: Positive domain length, area, or volume ``V``.

    Returns:
        Representative spacing ``h``.

    Raises:
        TypeError: If an argument is not real or ``dimension`` is not an int.
        ValueError: If ``cells`` or ``volume`` is not positive or the
            dimension is outside ``{1, 2, 3}``.

    Equation:
        [EQ-VV-GCI-001].

    References:
        [REF-CELIK-ETAL-2008].

    """
    count = require_finite_real(cells, name="cells")
    size = require_finite_real(volume, name="volume")
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError("dimension must be an integer")
    if dimension not in (1, 2, 3):
        raise ValueError("dimension must be 1, 2, or 3")
    if count <= 0.0 or size <= 0.0:
        raise ValueError("cells and volume must be positive")
    return math.pow(size / count, 1.0 / dimension)


def _sorted_grids(
    spacings: Sequence[float], values: Sequence[float], *, minimum: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate a refinement study and order it from fine to coarse.

    Args:
        spacings: Representative grid sizes ``h_i > 0``, all distinct.
        values: QoI value on each grid, in the same order as ``spacings``.
        minimum: Minimum number of grids required.

    Returns:
        Spacings and values sorted by increasing spacing (finest first).

    Raises:
        TypeError: If an entry is not real.
        ValueError: If lengths differ, too few grids are given, a spacing is
            not positive, or spacings repeat.

    References:
        [REF-CELIK-ETAL-2008]. No equation is evaluated.

    """
    if isinstance(spacings, str | bytes) or isinstance(values, str | bytes):
        raise TypeError("spacings and values must be sequences of real numbers")
    h_values = tuple(
        require_finite_real(value, name=f"spacings[{index}]")
        for index, value in enumerate(spacings)
    )
    phi_values = tuple(
        require_finite_real(value, name=f"values[{index}]")
        for index, value in enumerate(values)
    )
    if len(h_values) != len(phi_values):
        raise ValueError("spacings and values must have the same length")
    if len(h_values) < minimum:
        raise ValueError(f"at least {minimum} grids are required")
    if any(value <= 0.0 for value in h_values):
        raise ValueError("every spacing must be positive")
    if len(set(h_values)) != len(h_values):
        raise ValueError("spacings must be distinct")
    order = sorted(range(len(h_values)), key=h_values.__getitem__)
    return (
        tuple(h_values[i] for i in order),
        tuple(phi_values[i] for i in order),
    )


def _classify(ratio: float, r21: float, r32: float) -> ConvergenceType:
    """Classify a three-grid sequence from ``R = eps32 / eps21``.

    Args:
        ratio: Non-zero difference ratio ``R``.
        r21: Refinement ratio ``h2 / h1 > 1``.
        r32: Refinement ratio ``h3 / h2 > 1``.

    Returns:
        Oscillatory for ``R < 0``; monotone when ``R`` exceeds the zero-order
        limit ``ln r32 / ln r21`` (so a positive order exists); otherwise
        divergent.

    Equation:
        [EQ-VV-GCI-006].

    References:
        [REF-CELIK-ETAL-2008], [REF-ROY-2005].

    """
    if ratio < 0.0:
        return ConvergenceType.OSCILLATORY
    if ratio > math.log(r32) / math.log(r21):
        return ConvergenceType.MONOTONE
    return ConvergenceType.DIVERGENT


def classify_convergence(
    spacings: Sequence[float], values: Sequence[float]
) -> ConvergenceType:
    """Classify three refined solutions as monotone, oscillatory, or divergent.

    Args:
        spacings: Three distinct positive representative grid sizes.
        values: QoI on each grid (same order as ``spacings``).

    Returns:
        Convergence type of the sequence.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If not exactly three grids are given or a consecutive
            difference is zero (the ratio is then undefined).

    Equation:
        [EQ-VV-GCI-006].

    References:
        [REF-CELIK-ETAL-2008], [REF-ROY-2005].

    """
    h, phi = _sorted_grids(spacings, values, minimum=3)
    if len(h) != 3:
        raise ValueError("exactly three grids are required")
    eps21 = phi[1] - phi[0]
    eps32 = phi[2] - phi[1]
    if eps21 == 0.0 or eps32 == 0.0:
        raise ValueError(
            "consecutive solution differences must be non-zero to classify "
            "convergence"
        )
    return _classify(eps32 / eps21, h[1] / h[0], h[2] / h[1])


def _monotone_order_equation(
    order: float, ratio: float, r21: float, r32: float
) -> float:
    """Return ``ln f(p) - ln R`` for the monotone three-grid order equation.

    ``f(p) = r21^p (r32^p - 1) / (r21^p - 1)`` is strictly increasing in ``p``
    with ``f(0+) = ln r32 / ln r21``; its root is the observed order.

    Args:
        order: Trial order ``p > 0``.
        ratio: Positive difference ratio ``R = eps32 / eps21``.
        r21: Fine refinement ratio ``> 1``.
        r32: Coarse refinement ratio ``> 1``.

    Returns:
        Signed residual of the order equation.

    Equation:
        [EQ-VV-GCI-002].

    References:
        [REF-CELIK-ETAL-2008].

    """
    log_r21 = math.log(r21)
    return (
        order * log_r21
        + math.log(math.expm1(order * math.log(r32)))
        - math.log(math.expm1(order * log_r21))
        - math.log(ratio)
    )


def _observed_order(
    ratio: float,
    r21: float,
    r32: float,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[float, int, str]:
    """Solve the three-grid observed-order equation of Celik et al. (2008).

    The fixed point ``p = |ln|R| + q(p)| / ln r21`` with
    ``q(p) = ln((r21^p - s) / (r32^p - s))`` and ``s = sign(R)`` is iterated
    from ``q = 0`` (exact in one step for a uniform ratio). If the iteration
    fails for a monotone sequence, the strictly monotone signed equation is
    solved by Brent's method instead.

    Args:
        ratio: Non-zero difference ratio ``R = eps32 / eps21``.
        r21: Fine refinement ratio ``> 1``.
        r32: Coarse refinement ratio ``> 1``.
        tolerance: Relative convergence tolerance of the iteration.
        max_iterations: Maximum fixed-point iterations.

    Returns:
        Observed order, iterations used, and the solver name.

    Raises:
        RuntimeError: If no positive finite order can be found.

    Equation:
        [EQ-VV-GCI-002].

    References:
        [REF-CELIK-ETAL-2008], [REF-BRENT-1973, Chapter 4].

    """
    sign = 1.0 if ratio > 0.0 else -1.0
    log_ratio = math.log(abs(ratio))
    log_r21 = math.log(r21)
    log_r32 = math.log(r32)
    order = abs(log_ratio) / log_r21
    for iteration in range(1, max_iterations + 1):
        if order <= 0.0 or order * max(log_r21, log_r32) > _ORDER_EXPONENT_LIMIT:
            break
        numerator = math.exp(order * log_r21) - sign
        denominator = math.exp(order * log_r32) - sign
        if numerator <= 0.0 or denominator <= 0.0:
            break
        updated = abs(log_ratio + math.log(numerator / denominator)) / log_r21
        if abs(updated - order) <= tolerance * max(1.0, abs(updated)):
            if updated > 0.0:
                return updated, iteration, "fixed-point"
            break
        order = updated
    if sign < 0.0:
        raise RuntimeError(
            "the oscillatory observed-order iteration did not converge; "
            "no order can be reported"
        )
    upper_limit = _ORDER_EXPONENT_LIMIT / max(log_r21, log_r32)
    lower = 1.0e-8
    upper = 1.0
    while _monotone_order_equation(upper, ratio, r21, r32) < 0.0:
        upper *= 2.0
        if upper > upper_limit:
            raise RuntimeError("observed order exceeds the representable range")
    result = brent_root(
        lambda p: _monotone_order_equation(p, ratio, r21, r32),
        lower,
        upper,
        tolerance=1.0e-14,
    )
    if not result.converged or result.root <= 0.0:
        raise RuntimeError("observed-order root search did not converge")
    return result.root, result.iterations, "brent"


@dataclass(frozen=True, slots=True)
class GridConvergenceResult:
    """Three-grid Grid Convergence Index study (Celik et al. 2008 procedure).

    Grids are ordered fine (1), medium (2), coarse (3). Absolute GCIs are in
    QoI units; relative ones are fractions of the fine solution and are
    ``None`` when that solution is zero.

    Args:
        spacings: ``(h1, h2, h3)`` with ``h1 < h2 < h3``.
        values: ``(phi1, phi2, phi3)``.
        refinement_ratios: ``(r21, r32) = (h2/h1, h3/h2)``.
        convergence: Classification from ``R = eps32 / eps21``.
        difference_ratio: ``R``.
        order: Observed order ``p``.
        order_iterations: Iterations used by the order solver.
        order_solver: ``"fixed-point"`` or ``"brent"``.
        extrapolated: ``phi_ext^21`` from Richardson extrapolation.
        approximate_relative_error: ``e_a^21 = |(phi1 - phi2)/phi1|``.
        extrapolated_relative_error: ``e_ext^21 = |(phi_ext - phi1)/phi_ext|``.
        safety_factor: ``F_s``.
        gci_fine: Absolute fine-grid GCI ``F_s |eps21| / (r21^p - 1)``.
        gci_fine_relative: Relative fine-grid GCI ``F_s e_a^21 / (r21^p - 1)``.
        gci_coarse: Absolute GCI of the coarse pair ``F_s |eps32|/(r32^p - 1)``.
        asymptotic_ratio: Roache's ratio ``GCI_23 / (r21^p GCI_12)`` with
            relative GCIs (``GCI_23`` relative to ``phi2``, ``GCI_12`` to
            ``phi1``), ``None`` if either solution is zero. With the observed
            order of the same three grids it equals ``|phi1 / phi2|``
            identically, so it only shows that the relative change is small;
            :meth:`asymptotic_ratio_at` with the theoretical order is the
            informative check.

    References:
        [REF-CELIK-ETAL-2008], [REF-ROACHE-1998, Chapters 4-5].

    """

    spacings: tuple[float, float, float]
    values: tuple[float, float, float]
    refinement_ratios: tuple[float, float]
    convergence: ConvergenceType
    difference_ratio: float
    order: float
    order_iterations: int
    order_solver: str
    extrapolated: float
    approximate_relative_error: float | None
    extrapolated_relative_error: float | None
    safety_factor: float
    gci_fine: float
    gci_fine_relative: float | None
    gci_coarse: float
    asymptotic_ratio: float | None

    @property
    def oscillatory(self) -> bool:
        """Return whether the three solutions oscillate under refinement.

        Equation:
            [EQ-VV-GCI-006].

        References:
            [REF-CELIK-ETAL-2008].

        """
        return self.convergence is ConvergenceType.OSCILLATORY

    def asymptotic_ratio_at(self, order: float) -> float:
        """Return the absolute-GCI ratio ``GCI_23 / (r21^p GCI_12)`` at order ``p``.

        Algebraically the ratio is ``R / f(p)`` with
        ``f(p) = r21^p (r32^p - 1) / (r21^p - 1)``. At the observed order it
        is exactly one; at the theoretical order it measures how far the
        three solutions are from the formal asymptotic range (for a uniform
        ratio ``r`` it equals ``r^(p_obs - p_th)``).

        Args:
            order: Positive reference order (typically the theoretical one).

        Returns:
            Asymptotic-range ratio; ``~1`` inside the asymptotic range.

        Raises:
            ValueError: If ``order`` is not positive.

        Equation:
            [EQ-VV-GCI-005].

        References:
            [REF-ROACHE-1998, Chapters 4-5], [REF-CELIK-ETAL-2008].

        """
        reference = require_finite_real(order, name="order")
        if reference <= 0.0:
            raise ValueError("order must be positive")
        r21, r32 = self.refinement_ratios
        eps21 = self.values[1] - self.values[0]
        eps32 = self.values[2] - self.values[1]
        gci12 = self.safety_factor * abs(eps21) / math.expm1(reference * math.log(r21))
        gci23 = self.safety_factor * abs(eps32) / math.expm1(reference * math.log(r32))
        return gci23 / (math.pow(r21, reference) * gci12)

    def in_asymptotic_range(
        self, tolerance: float = 0.1, *, order: float | None = None
    ) -> bool:
        """Return whether the asymptotic-range ratio is within ``1 +/- tolerance``.

        Args:
            tolerance: Non-negative allowed deviation of the ratio from one.
            order: Reference order for :meth:`asymptotic_ratio_at`; ``None``
                uses Roache's relative ratio :attr:`asymptotic_ratio` (which
                fails when it is undefined).

        Returns:
            ``True`` when the asymptotic-range check passes.

        Raises:
            ValueError: If ``tolerance`` or ``order`` is invalid.

        Equation:
            [EQ-VV-GCI-005].

        References:
            [REF-ROACHE-1998, Chapters 4-5].

        """
        allowed = require_non_negative_finite_real(tolerance, name="tolerance")
        if order is None:
            if self.asymptotic_ratio is None:
                return False
            return abs(self.asymptotic_ratio - 1.0) <= allowed
        return abs(self.asymptotic_ratio_at(order) - 1.0) <= allowed


def grid_convergence_index(
    spacings: Sequence[float],
    values: Sequence[float],
    *,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    tolerance: float = 1.0e-12,
    max_iterations: int = 500,
) -> GridConvergenceResult:
    """Compute the three-grid GCI with non-uniform refinement ratios.

    Steps (Celik et al. 2008): order the grids fine to coarse, form
    ``r21 = h2/h1``, ``r32 = h3/h2``, ``eps21 = phi2 - phi1``,
    ``eps32 = phi3 - phi2``; solve ``p = |ln|eps32/eps21| + q(p)| / ln r21``
    by fixed-point iteration; extrapolate ``phi_ext^21``; report the
    approximate and extrapolated relative errors and the fine-grid GCI with
    factor of safety ``F_s``; and evaluate Roache's asymptotic-range ratio.

    Oscillatory sequences (``R < 0``) are computed with ``s = -1`` as in the
    published procedure but are flagged; :func:`verify_solution` does not use
    their GCI. Divergent sequences have no positive order and are rejected.

    Args:
        spacings: Three distinct positive representative grid sizes in any
            order (see :func:`representative_spacing`).
        values: QoI value on each grid, in the order of ``spacings``.
        safety_factor: ``F_s >= 1``; 1.25 for three-grid studies with an
            observed order, 3.0 when the order is not trusted.
        tolerance: Relative tolerance of the fixed-point iteration.
        max_iterations: Iteration limit of the fixed-point solver.

    Returns:
        Grid convergence record.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If not exactly three grids are given, spacings repeat or
            are non-positive, a difference is zero, the sequence is divergent,
            or ``safety_factor < 1``.
        RuntimeError: If the observed-order equation cannot be solved.

    Equation:
        [EQ-VV-GCI-002], [EQ-VV-GCI-003], [EQ-VV-GCI-004], [EQ-VV-GCI-005],
        [EQ-VV-GCI-006].

    References:
        [REF-CELIK-ETAL-2008], [REF-ROACHE-1998, Chapters 4-5],
        [REF-ROY-2005].

    """
    h, phi = _sorted_grids(spacings, values, minimum=3)
    if len(h) != 3:
        raise ValueError("exactly three grids are required")
    factor = require_finite_real(safety_factor, name="safety_factor")
    if factor < 1.0:
        raise ValueError("safety_factor must be at least one")
    tol = require_finite_real(tolerance, name="tolerance")
    if tol <= 0.0:
        raise ValueError("tolerance must be positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    r21 = h[1] / h[0]
    r32 = h[2] / h[1]
    eps21 = phi[1] - phi[0]
    eps32 = phi[2] - phi[1]
    if eps21 == 0.0 or eps32 == 0.0:
        raise ValueError(
            "consecutive solution differences must be non-zero; the QoI may "
            "already be converged to round-off on these grids"
        )
    ratio = eps32 / eps21
    convergence = _classify(ratio, r21, r32)
    if convergence is ConvergenceType.DIVERGENT:
        raise ValueError(
            "the sequence is divergent (no positive observed order exists); "
            "no grid convergence index can be formed"
        )
    order, iterations, solver = _observed_order(
        ratio, r21, r32, tolerance=tol, max_iterations=max_iterations
    )
    fine_factor = math.expm1(order * math.log(r21))
    coarse_factor = math.expm1(order * math.log(r32))
    extrapolated = phi[0] + (phi[0] - phi[1]) / fine_factor
    approximate = None if phi[0] == 0.0 else abs(eps21 / phi[0])
    extrapolated_error = (
        None if extrapolated == 0.0 else abs((extrapolated - phi[0]) / extrapolated)
    )
    gci_fine = factor * abs(eps21) / fine_factor
    gci_coarse = factor * abs(eps32) / coarse_factor
    asymptotic = (
        None
        if phi[0] == 0.0 or phi[1] == 0.0
        else (gci_coarse / abs(phi[1]))
        / (math.pow(r21, order) * gci_fine / abs(phi[0]))
    )
    return GridConvergenceResult(
        (h[0], h[1], h[2]),
        (phi[0], phi[1], phi[2]),
        (r21, r32),
        convergence,
        ratio,
        order,
        iterations,
        solver,
        extrapolated,
        approximate,
        extrapolated_error,
        factor,
        gci_fine,
        None if approximate is None else factor * approximate / fine_factor,
        gci_coarse,
        asymptotic,
    )


@dataclass(frozen=True, slots=True)
class LeastSquaresConvergence:
    """Least-squares observed order from four or more grids (simplified).

    The model ``phi_i = phi_0 + alpha (h_i / h_1)^p`` is fitted by least
    squares in ``(phi_0, alpha, p)``; ``h_1`` is the finest spacing, so
    ``alpha`` is the fitted discretization error ``delta_RE`` of the finest
    solution. The uncertainty rule is a documented simplification of Eca and
    Hoekstra (2014), not their complete procedure.

    Args:
        spacings: Spacings sorted fine to coarse.
        values: QoI values in the same order.
        order: Fitted observed order ``p``.
        extrapolated: Fitted zero-spacing estimate ``phi_0``.
        fine_error: ``delta_RE = alpha`` at the finest grid.
        residual_standard_deviation: ``U_s = sqrt(S_min / (n - 3))``.
        order_at_bound: Whether ``p`` sits on a search bound.
        theoretical_order: Formal order ``p_th`` of the discretization.
        credible_order: ``0.5 <= p < p_th + 0.1`` and not on a bound.
        fixed_order_error: ``alpha_th`` of the refit with ``p = p_th``.
        fixed_order_residual_standard_deviation: ``sqrt(S_th / (n - 2))``.
        safety_factor: 1.25 when the order is credible, otherwise 3.0.
        uncertainty: ``U`` from [EQ-VV-LSQ-002].

    References:
        [REF-ECA-HOEKSTRA-2014], [REF-ROACHE-1998, Chapters 4-5].

    """

    spacings: tuple[float, ...]
    values: tuple[float, ...]
    order: float
    extrapolated: float
    fine_error: float
    residual_standard_deviation: float
    order_at_bound: bool
    theoretical_order: float
    credible_order: bool
    fixed_order_error: float
    fixed_order_residual_standard_deviation: float
    safety_factor: float
    uncertainty: float

    @property
    def grid_count(self) -> int:
        """Return the number of grids in the fit.

        References:
            [REF-ECA-HOEKSTRA-2014].

        """
        return len(self.spacings)


def _fixed_order_fit(
    ratios: Sequence[float], values: Sequence[float], order: float
) -> tuple[float, float, float]:
    """Fit ``phi = phi_0 + alpha x^p`` by linear least squares at fixed ``p``.

    Args:
        ratios: ``h_i / h_1 >= 1``, at least two distinct.
        values: QoI values.
        order: Fixed positive order ``p``.

    Returns:
        ``(phi_0, alpha, S)`` with ``S`` the residual sum of squares.

    Equation:
        [EQ-VV-LSQ-001].

    References:
        [REF-ECA-HOEKSTRA-2014].

    """
    xs = [math.pow(ratio, order) for ratio in ratios]
    count = len(xs)
    x_mean = math.fsum(xs) / count
    y_mean = math.fsum(values) / count
    sxx = math.fsum((x - x_mean) ** 2 for x in xs)
    sxy = math.fsum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, values, strict=True)
    )
    alpha = sxy / sxx
    phi0 = y_mean - alpha * x_mean
    residual = math.fsum(
        (y - phi0 - alpha * x) ** 2 for x, y in zip(xs, values, strict=True)
    )
    return phi0, alpha, residual


def least_squares_convergence(
    spacings: Sequence[float],
    values: Sequence[float],
    *,
    theoretical_order: float = 2.0,
    order_bounds: tuple[float, float] = (0.05, 10.0),
) -> LeastSquaresConvergence:
    """Fit the observed order from four or more grids by least squares.

    For fixed ``p`` the fit is linear in ``(phi_0, alpha)``; the profile
    residual ``S(p)`` is scanned on 400 log-spaced orders inside
    ``order_bounds`` and refined by golden-section search around the best
    scan point. The numerical uncertainty of the finest solution is
    ([EQ-VV-LSQ-002]):

    * credible order ``0.5 <= p < p_th + 0.1``: ``U = 1.25 |delta_RE| + U_s``;
    * otherwise: refit with ``p = p_th`` and use
      ``U = 3 max(|delta_RE|, |delta_th|) + max(U_s, U_s,th)``.

    This keeps the structure of Eca and Hoekstra (2014) — least-squares
    order, data-scatter term ``U_s``, ``F_s = 1.25`` or ``3`` — but omits
    their weighted fits, their alternative fixed-order expansions, and their
    data-range term.

    Args:
        spacings: Four or more distinct positive grid sizes.
        values: QoI on each grid.
        theoretical_order: Positive formal order ``p_th`` of the scheme.
        order_bounds: ``0 < p_min < p_max`` search interval for ``p``.

    Returns:
        Least-squares convergence record.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If fewer than four grids are given, spacings repeat or are
            non-positive, all values are equal, or bounds are invalid.

    Equation:
        [EQ-VV-LSQ-001], [EQ-VV-LSQ-002].

    References:
        [REF-ECA-HOEKSTRA-2014], [REF-BRENT-1973, Chapter 5].

    """
    h, phi = _sorted_grids(spacings, values, minimum=4)
    p_th = require_finite_real(theoretical_order, name="theoretical_order")
    if p_th <= 0.0:
        raise ValueError("theoretical_order must be positive")
    low = require_finite_real(order_bounds[0], name="order_bounds[0]")
    high = require_finite_real(order_bounds[1], name="order_bounds[1]")
    if not 0.0 < low < high:
        raise ValueError("order_bounds must satisfy 0 < lower < upper")
    if max(phi) == min(phi):
        raise ValueError("all solutions are equal; the observed order is undefined")
    ratios = tuple(value / h[0] for value in h)
    if high * math.log(ratios[-1]) > _ORDER_EXPONENT_LIMIT:
        raise ValueError("order_bounds upper limit overflows (h_max/h_min)^p")
    count = len(h)

    def profile(order: float) -> float:
        """Return the residual sum of squares at a fixed order.

        Args:
            order: Fixed positive order.

        Returns:
            Profile residual ``S(p)``.

        Equation:
            [EQ-VV-LSQ-001].

        References:
            [REF-ECA-HOEKSTRA-2014].

        """
        return _fixed_order_fit(ratios, phi, order)[2]

    scan = 400
    log_low, log_high = math.log(low), math.log(high)
    grid = [
        math.exp(log_low + (log_high - log_low) * k / (scan - 1)) for k in range(scan)
    ]
    residuals = [profile(order) for order in grid]
    best = min(range(scan), key=residuals.__getitem__)
    a = grid[max(best - 1, 0)]
    b = grid[min(best + 1, scan - 1)]
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - golden * (b - a)
    d = a + golden * (b - a)
    fc, fd = profile(c), profile(d)
    for _ in range(200):
        if abs(b - a) <= 1.0e-12 * max(1.0, abs(c)):
            break
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - golden * (b - a)
            fc = profile(c)
        else:
            a, c, fc = c, d, fd
            d = a + golden * (b - a)
            fd = profile(d)
    _, order = min((fc, c), (fd, d), (residuals[best], grid[best]))
    phi0, alpha, residual = _fixed_order_fit(ratios, phi, order)
    spread = math.sqrt(max(residual, 0.0) / (count - 3))
    at_bound = best in (0, scan - 1)
    credible = 0.5 <= order < p_th + 0.1 and not at_bound
    _, alpha_th, residual_th = _fixed_order_fit(ratios, phi, p_th)
    spread_th = math.sqrt(max(residual_th, 0.0) / (count - 2))
    if credible:
        factor = DEFAULT_SAFETY_FACTOR
        uncertainty = factor * abs(alpha) + spread
    else:
        factor = UNRELIABLE_SAFETY_FACTOR
        uncertainty = factor * max(abs(alpha), abs(alpha_th)) + max(
            spread, spread_th
        )
    return LeastSquaresConvergence(
        h,
        phi,
        order,
        phi0,
        alpha,
        spread,
        at_bound,
        p_th,
        credible,
        alpha_th,
        spread_th,
        factor,
        uncertainty,
    )


NumericalErrorDistribution = Literal["uniform", "normal95"]
"""How an expanded numerical uncertainty ``U`` is converted to ``u_num``."""


@dataclass(frozen=True, slots=True)
class SolutionVerification:
    """Numerical-uncertainty statement for one QoI from a refinement study.

    The statement is ``phi_fine +/- U``: the QoI computed on the finest grid
    with expanded numerical uncertainty ``U`` (a GCI, a least-squares
    uncertainty, or an oscillation bound). Validation consumes it through
    :meth:`standard_uncertainty` or :meth:`error_budget`; the conversion is
    always explicit.

    Args:
        quantity: QoI name.
        method: How ``U`` was obtained.
        fine_spacing: Finest representative spacing ``h_1``.
        fine_value: QoI on the finest grid ``phi_1``.
        expanded_uncertainty: ``U >= 0`` in QoI units.
        convergence: Observed convergence type.
        observed_order: Observed order, or ``None`` when not meaningful.
        extrapolated: Zero-spacing estimate, or ``None`` when not meaningful.
        safety_factor: Factor of safety used in ``U``.
        grid_count: Number of grids in the study.
        warnings: Conditions that weaken the estimate.
        study: The underlying GCI or least-squares record (``None`` for the
            oscillation bound).

    References:
        [REF-ROACHE-1998, Chapters 4-5], [REF-CELIK-ETAL-2008],
        [REF-ECA-HOEKSTRA-2014], [REF-OBERKAMPF-ROY-2025, Chapters 6-8].

    """

    quantity: str
    method: str
    fine_spacing: float
    fine_value: float
    expanded_uncertainty: float
    convergence: ConvergenceType
    observed_order: float | None
    extrapolated: float | None
    safety_factor: float
    grid_count: int
    warnings: tuple[str, ...] = ()
    study: GridConvergenceResult | LeastSquaresConvergence | None = None

    def __post_init__(self) -> None:
        """Validate the statement fields.

        Raises:
            TypeError: If the convergence type has the wrong type.
            ValueError: If text is empty or a number is invalid.

        References:
            [REF-ROACHE-1998, Chapters 4-5]. No equation is evaluated.

        """
        for name in ("quantity", "method"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.convergence, ConvergenceType):
            raise TypeError("convergence must be a ConvergenceType")
        spacing = require_finite_real(self.fine_spacing, name="fine_spacing")
        if spacing <= 0.0:
            raise ValueError("fine_spacing must be positive")
        object.__setattr__(self, "fine_spacing", spacing)
        object.__setattr__(
            self, "fine_value", require_finite_real(self.fine_value, name="fine_value")
        )
        object.__setattr__(
            self,
            "expanded_uncertainty",
            require_non_negative_finite_real(
                self.expanded_uncertainty, name="expanded_uncertainty"
            ),
        )
        factor = require_finite_real(self.safety_factor, name="safety_factor")
        if factor < 1.0:
            raise ValueError("safety_factor must be at least one")
        object.__setattr__(self, "safety_factor", factor)
        object.__setattr__(self, "warnings", tuple(str(w) for w in self.warnings))

    @property
    def interval(self) -> tuple[float, float]:
        """Return ``(phi_1 - U, phi_1 + U)``.

        Equation:
            [EQ-VV-UNUM-001].

        References:
            [REF-ROACHE-1998, Chapters 4-5].

        """
        return (
            self.fine_value - self.expanded_uncertainty,
            self.fine_value + self.expanded_uncertainty,
        )

    @property
    def reliable(self) -> bool:
        """Return ``True`` for a monotone study without warnings.

        References:
            [REF-CELIK-ETAL-2008], [REF-ROY-2005].

        """
        return self.convergence is ConvergenceType.MONOTONE and not self.warnings

    def standard_uncertainty(
        self, distribution: NumericalErrorDistribution = "uniform"
    ) -> float:
        """Convert the expanded uncertainty ``U`` to a standard uncertainty.

        Args:
            distribution: ``"uniform"`` treats ``[-U, U]`` as the support of a
                rectangular error (``u = U / sqrt(3)``, the same conversion
                ``validation.budget_standard_uncertainty`` applies to error
                bounds); ``"normal95"`` treats ``U`` as a 95 % expanded
                uncertainty of a Gaussian error (``u = U / 1.959964``).

        Returns:
            Standard numerical uncertainty ``u_num``.

        Raises:
            ValueError: If the distribution name is unknown.

        Equation:
            [EQ-VV-UNUM-001].

        References:
            [REF-JCGM-100-2008, Sections 4.3.7 and 6.2],
            [REF-ROACHE-1998, Chapters 4-5].

        """
        if distribution == "uniform":
            return self.expanded_uncertainty / math.sqrt(3.0)
        if distribution == "normal95":
            return self.expanded_uncertainty / _NORMAL_95
        raise ValueError("distribution must be 'uniform' or 'normal95'")

    def error_budget(self) -> NumericalErrorBudget:
        """Return ``U`` as the discretization bound of an error budget.

        Returns:
            Budget with ``discretization = U`` and every other term zero.

        Equation:
            [EQ-VV-003].

        References:
            [REF-OBERKAMPF-ROY-2025, Chapters 6-8].

        """
        return NumericalErrorBudget(discretization=self.expanded_uncertainty)

    def summary(self) -> str:
        """Return a one-paragraph description of the statement.

        Returns:
            Text with value, interval, method, order, and warnings.

        References:
            [REF-ROACHE-1998, Chapters 4-5]. No equation is evaluated.

        """
        order = (
            "no meaningful observed order"
            if self.observed_order is None
            else f"observed order {self.observed_order:.4g}"
        )
        text = (
            f"{self.quantity} = {self.fine_value:.8g} +/- "
            f"{self.expanded_uncertainty:.4g} (numerical, h={self.fine_spacing:.4g}, "
            f"{self.grid_count} grids, {self.convergence.value} convergence, "
            f"{order}, F_s={self.safety_factor:g}; {self.method})"
        )
        if self.warnings:
            text += " Warnings: " + "; ".join(self.warnings) + "."
        return text


def verify_solution(
    quantity: str,
    spacings: Sequence[float],
    values: Sequence[float],
    *,
    theoretical_order: float = 2.0,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    asymptotic_tolerance: float = 0.1,
    oscillatory_safety_factor: float = UNRELIABLE_SAFETY_FACTOR,
) -> SolutionVerification:
    """Turn a grid-refinement study into a numerical-uncertainty statement.

    Rules (every one is reported in the result):

    * three grids, monotone: ``U = GCI_fine`` (absolute, [EQ-VV-GCI-004]);
      a warning when the asymptotic-range ratio evaluated at the theoretical
      order deviates from one by more than ``asymptotic_tolerance`` (at the
      observed order the ratio is identically one and tests nothing);
    * three grids, oscillatory: no Richardson estimate is used;
      ``U = F_s,osc (phi_max - phi_min) / 2`` ([EQ-VV-OSC-001]);
    * three grids, divergent: rejected (no uncertainty can be estimated);
    * four or more grids: :func:`least_squares_convergence`; non-monotone
      sequences are labelled oscillatory and keep the scatter term ``U_s``.

    Args:
        quantity: QoI name.
        spacings: Representative grid sizes (three or more, distinct).
        values: QoI on each grid.
        theoretical_order: Formal order of the scheme.
        safety_factor: ``F_s`` of the three-grid GCI.
        asymptotic_tolerance: Allowed deviation of the asymptotic ratio.
        oscillatory_safety_factor: ``F_s`` of the oscillation bound.

    Returns:
        Solution-verification statement.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If fewer than three grids are given, the data are
            degenerate, or a three-grid sequence is divergent.
        RuntimeError: If the observed order cannot be solved.

    Equation:
        [EQ-VV-GCI-004], [EQ-VV-GCI-005], [EQ-VV-LSQ-002], [EQ-VV-OSC-001].

    References:
        [REF-CELIK-ETAL-2008], [REF-ECA-HOEKSTRA-2014], [REF-ROY-2005],
        [REF-ROACHE-1998, Chapters 4-5].

    """
    if not isinstance(quantity, str) or not quantity.strip():
        raise ValueError("quantity must be a non-empty string")
    h, phi = _sorted_grids(spacings, values, minimum=3)
    p_th = require_finite_real(theoretical_order, name="theoretical_order")
    if p_th <= 0.0:
        raise ValueError("theoretical_order must be positive")
    tolerance = require_non_negative_finite_real(
        asymptotic_tolerance, name="asymptotic_tolerance"
    )
    osc_factor = require_finite_real(
        oscillatory_safety_factor, name="oscillatory_safety_factor"
    )
    if osc_factor < 1.0:
        raise ValueError("oscillatory_safety_factor must be at least one")
    if len(h) >= 4:
        fit = least_squares_convergence(h, phi, theoretical_order=p_th)
        differences = [b - a for a, b in zip(phi, phi[1:], strict=False)]
        monotone = all(d > 0.0 for d in differences) or all(
            d < 0.0 for d in differences
        )
        warnings: list[str] = []
        if not monotone:
            warnings.append(
                "non-monotone sequence; the least-squares scatter term U_s "
                "carries the oscillation"
            )
        if not fit.credible_order:
            warnings.append(
                f"observed order {fit.order:.3g} outside the credible range "
                f"[0.5, {p_th + 0.1:.3g}); conservative F_s=3 rule applied"
            )
        return SolutionVerification(
            quantity,
            "least-squares observed order (simplified Eca-Hoekstra 2014)",
            h[0],
            phi[0],
            fit.uncertainty,
            ConvergenceType.MONOTONE if monotone else ConvergenceType.OSCILLATORY,
            fit.order,
            fit.extrapolated,
            fit.safety_factor,
            len(h),
            tuple(warnings),
            fit,
        )
    convergence = classify_convergence(h, phi)
    if convergence is ConvergenceType.OSCILLATORY:
        bound = osc_factor * 0.5 * (max(phi) - min(phi))
        return SolutionVerification(
            quantity,
            "oscillatory three-grid study: half-range bound times F_s",
            h[0],
            phi[0],
            bound,
            convergence,
            None,
            None,
            osc_factor,
            3,
            (
                "oscillatory convergence; Richardson extrapolation and the GCI "
                "are not used",
            ),
            None,
        )
    if convergence is ConvergenceType.DIVERGENT:
        raise ValueError(
            "the three-grid sequence is divergent; no numerical uncertainty "
            "can be estimated from it"
        )
    gci = grid_convergence_index(h, phi, safety_factor=safety_factor)
    notes: list[str] = []
    ratio_at_theory = gci.asymptotic_ratio_at(p_th)
    if abs(ratio_at_theory - 1.0) > tolerance:
        notes.append(
            f"asymptotic-range ratio at the theoretical order {p_th:g} is "
            f"{ratio_at_theory:.4g} (observed order {gci.order:.3g}); outside "
            f"1 +/- {tolerance:g}, so the grids may not be in the asymptotic range"
        )
    return SolutionVerification(
        quantity,
        "three-grid GCI (Celik et al. 2008)",
        h[0],
        phi[0],
        gci.gci_fine,
        convergence,
        gci.order,
        gci.extrapolated,
        gci.safety_factor,
        3,
        tuple(notes),
        gci,
    )
