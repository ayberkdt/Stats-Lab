"""Probabilistic buckling of thin-walled circular cylinders in axial compression.

A launch-vehicle tank or interstage class problem: a thin isotropic circular
cylinder (radius ``R``, wall thickness ``t``, Young modulus ``E``, Poisson
ratio ``nu``) loaded by an axial force ``P``. All quantities are SI (Pa, m, N).

Building blocks, in the order of the module:

- **Classical buckling** [EQ-AERO-BUCK-001]: ``sigma_cl = E t / (R sqrt(3
  (1 - nu^2)))`` and ``P_cl = 2 pi R t sigma_cl`` (small-deflection linear
  theory of the perfect shell).
- **Imperfection sensitivity** [EQ-AERO-BUCK-002, EQ-AERO-BUCK-003]: Koiter's
  asymptotic law for an axisymmetric imperfection of amplitude ``delta``
  (in wall thicknesses, ``d = delta / t``),
  ``(1 - lambda)^2 = c lambda d`` with the knockdown ``lambda = P_cr / P_cl``
  and one structural constant ``c = 4 |a|`` (``a`` is the initial slope of
  the asymmetric post-buckling path). It is derived below, solved in closed
  form (``lambda`` is the smaller root of ``lambda^2 - (2 + c d) lambda + 1``),
  and inverted exactly (``d = (1 - lambda)^2 / (c lambda)``).
- **Empirical design knockdown** [EQ-AERO-BUCK-004]: the NASA SP-8007 lower
  bound ``gamma = 1 - 0.901 (1 - exp(-phi))``, ``phi = sqrt(R/t) / 16``, for
  ``R/t < 1500`` (Eqs. 9 and 10 of NASA/SP-8007-2020/REV 2, read from the
  report by the maintainers).
- **Probabilistic knockdown** [EQ-AERO-BUCK-005]: the knockdown ``k_p`` with
  ``P(lambda < k_p) = p`` for a random imperfection amplitude, exact because
  ``lambda`` decreases monotonically in ``d``.
- **Probabilistic model** [EQ-AERO-BUCK-006]: :func:`shell_buckling_model`
  returns a :class:`~probstat.modeling.ProbabilisticModel` of ``E``, ``t``,
  ``R``, imperfection amplitude and applied load with outputs buckling load,
  knockdown and margin ``P_cr - P_applied``; failure is ``margin <= 0``.
- **Design** [EQ-AERO-BUCK-007]: the thickness that attains a target failure
  probability (FORM, monotone Brent root), against the deterministic SP-8007
  design thickness.

Derivation of the Koiter relation. Along the bifurcation path of the perfect
shell the load ratio is ``lambda = 1 + a xi`` (asymmetric bifurcation, ``a <
0`` unstable), ``xi`` the amplitude of the buckling mode. For the imperfect
shell with initial imperfection ``xi_bar`` (affine to the buckling mode, in
the same units) the potential energy per unit of the critical stiffness is
``Phi = (1 - lambda) xi^2 / 2 + a xi^3 / 3 - lambda xi_bar xi``, so equilibrium
is ``(1 - lambda) xi + a xi^2 = lambda xi_bar``. The limit load is the
maximum of ``lambda`` along this path, ``d(lambda)/d(xi) = 0``, i.e. ``(1 -
lambda) + 2 a xi = 0``. Eliminating ``xi = (1 - lambda) / (2 |a|)`` gives
``(1 - lambda)^2 = 4 |a| lambda xi_bar``, the relation above with ``c = 4 |a|``
and ``d = xi_bar``. For small ``d``, ``lambda ~ 1 - sqrt(c d)``: the square-root
imperfection sensitivity that makes the shell so much weaker than the perfect
one. Assumptions: small imperfection, a single (axisymmetric, critical-mode
shaped) imperfection, asymptotic validity of the cubic potential (the law is
an extrapolation for ``d`` of order one, and returns ``lambda -> 0`` only as
``d -> infinity``).

Cylinder constant: NASA/SP-8007-2020/REV 2 Eqs. 138-139 give ``gamma = 1 +
alpha eps - sqrt(alpha eps (2 + alpha eps))`` with ``alpha = (3/4) sqrt(3 (1 -
nu^2))`` and ``eps`` the imperfection amplitude over the wall thickness, "based
on the curve presented by Koiter" (Figure 4-22, nu = 0.3). Solving the relation
above gives exactly that form with ``y = c d / 2``, so
``c = 2 alpha = (3/2) sqrt(3 (1 - nu^2))`` (about 2.45 at ``nu = 0.33``). That
is the default of :func:`koiter_coefficient`; the original paper of Koiter was
not read. Every function accepts an explicit ``coefficient`` to override it.

Limitations: isotropic, axial compression only; no internal-pressure
stabilization; no stiffeners, cutouts or load eccentricity; no plasticity; the
knockdown is for one imperfection shape. The SP-8007 relation is an empirical
lower bound for test data of the 1930s to 1960s and is only defined for
``R/t < 1500``. All numbers in this library's cases are synthetic
(aluminium-alloy class) and are not data from any real vehicle.

Experimental (added in 0.28).

References:
    [REF-NASA-SP-8007-2020, Eqs. 138-139], [REF-NASA-SP-8007-1968].

"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from probstat.distributions import LogNormal, Normal, Weibull
from probstat.foundations.validation import require_finite_real
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.modeling.inputs import ContinuousMarginal, Marginal
from probstat.optimization import brent_root
from probstat.reliability import ModelLimitState, form_hlrf
from probstat.uq import FailurePolicy, QuantityOfInterest

__all__ = [
    "SP8007_COEFFICIENT",
    "SP8007_MAX_RADIUS_TO_THICKNESS",
    "SP8007_PHI_DIVISOR",
    "KnockdownComparison",
    "ThicknessDesign",
    "classical_buckling_load",
    "classical_buckling_stress",
    "compare_knockdowns",
    "design_thickness_for_reliability",
    "koiter_coefficient",
    "koiter_imperfection_for_knockdown",
    "koiter_knockdown",
    "koiter_knockdown_derivative",
    "lognormal_from_mean_cov",
    "probabilistic_knockdown",
    "shell_buckling_model",
    "sp8007_axial_knockdown",
    "sp8007_buckling_load",
    "sp8007_design_thickness",
    "weibull_from_mean_cov",
]

SP8007_COEFFICIENT = 0.901
"""Coefficient of the SP-8007 knockdown, Eq. 9 of NASA/SP-8007-2020/REV 2."""

SP8007_PHI_DIVISOR = 16.0
"""Divisor of ``phi = sqrt(R/t) / 16``, Eq. 10 of NASA/SP-8007-2020/REV 2."""

SP8007_MAX_RADIUS_TO_THICKNESS = 1500.0
"""Upper (exclusive) limit of ``R/t`` of the SP-8007 relation."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _positive(value: object, name: str) -> float:
    """Return a finite strictly positive real.

    Args:
        value: Candidate number.
        name: Argument name for messages.

    Returns:
        The value as a float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not finite and positive.

    References:
        [REF-PYTHON-DATAMODEL]. A software validator; no equation.

    """
    number = require_finite_real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _non_negative(value: object, name: str) -> float:
    """Return a finite non-negative real.

    Args:
        value: Candidate number.
        name: Argument name for messages.

    Returns:
        The value as a float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not finite or is negative.

    References:
        [REF-PYTHON-DATAMODEL]. A software validator; no equation.

    """
    number = require_finite_real(value, name=name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _poisson(value: object) -> float:
    """Return a Poisson ratio in ``[0, 0.5)``.

    Args:
        value: Candidate Poisson ratio.

    Returns:
        The value as a float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is outside ``[0, 0.5)``.

    References:
        [REF-PYTHON-DATAMODEL]. A software validator; no equation.

    """
    number = require_finite_real(value, name="poisson_ratio")
    if not 0.0 <= number < 0.5:
        raise ValueError("poisson_ratio must lie in [0, 0.5)")
    return number


# ---------------------------------------------------------------------------
# Classical buckling
# ---------------------------------------------------------------------------


def classical_buckling_stress(
    youngs_modulus: float,
    thickness: float,
    radius: float,
    poisson_ratio: float = 0.33,
) -> float:
    """Return the classical axial buckling stress of a perfect cylinder.

    ``sigma_cl = E t / (R sqrt(3 (1 - nu^2)))`` in Pa, valid for thin shells
    (``t / R`` small), isotropic material, linear small-deflection theory.

    Args:
        youngs_modulus: ``E`` (Pa), positive.
        thickness: Wall thickness ``t`` (m), positive.
        radius: Mid-surface radius ``R`` (m), positive.
        poisson_ratio: ``nu`` in ``[0, 0.5)``.

    Returns:
        ``sigma_cl`` (Pa).

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is outside its domain.

    Equation:
        [EQ-AERO-BUCK-001].

    References:
        [REF-NASA-SP-8007-2020, Eq. 7 with gamma = 1].

    """
    modulus = _positive(youngs_modulus, "youngs_modulus")
    wall = _positive(thickness, "thickness")
    mean_radius = _positive(radius, "radius")
    nu = _poisson(poisson_ratio)
    return modulus * wall / (mean_radius * math.sqrt(3.0 * (1.0 - nu * nu)))


def classical_buckling_load(
    youngs_modulus: float,
    thickness: float,
    radius: float,
    poisson_ratio: float = 0.33,
) -> float:
    """Return the classical axial buckling load ``P_cl = 2 pi R t sigma_cl``.

    Args:
        youngs_modulus: ``E`` (Pa), positive.
        thickness: Wall thickness ``t`` (m), positive.
        radius: Mid-surface radius ``R`` (m), positive.
        poisson_ratio: ``nu`` in ``[0, 0.5)``.

    Returns:
        ``P_cl`` (N).

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is outside its domain.

    Equation:
        [EQ-AERO-BUCK-001].

    References:
        [REF-NASA-SP-8007-2020, Eq. 7].

    """
    stress = classical_buckling_stress(youngs_modulus, thickness, radius, poisson_ratio)
    return 2.0 * math.pi * float(radius) * float(thickness) * stress


# ---------------------------------------------------------------------------
# Koiter imperfection law
# ---------------------------------------------------------------------------


def koiter_coefficient(poisson_ratio: float = 0.33) -> float:
    """Return the Koiter constant ``c = (3/2) sqrt(3 (1 - nu^2))``.

    With ``c = 2 alpha`` the closed form of :func:`koiter_knockdown` equals
    Eq. 138 of NASA/SP-8007-2020/REV 2, ``gamma = 1 + alpha eps - sqrt(alpha
    eps (2 + alpha eps))`` with ``alpha = (3/4) sqrt(3 (1 - nu^2))`` and ``eps``
    the imperfection amplitude over the wall thickness (Eqs. 138-139, based on
    the Koiter curve of the report's Figure 4-22 for ``nu = 0.3``). Pass an
    explicit ``coefficient`` to the other functions to override it.

    Args:
        poisson_ratio: ``nu`` in ``[0, 0.5)``.

    Returns:
        ``c`` (dimensionless).

    Raises:
        TypeError: If ``poisson_ratio`` is not a real number.
        ValueError: If ``poisson_ratio`` is outside ``[0, 0.5)``.

    Equation:
        [EQ-AERO-BUCK-002].

    References:
        [REF-NASA-SP-8007-2020, Eqs. 138-139].

    """
    nu = _poisson(poisson_ratio)
    return 1.5 * math.sqrt(3.0 * (1.0 - nu * nu))


def koiter_knockdown(imperfection_ratio: float, coefficient: float) -> float:
    """Solve ``(1 - lambda)^2 = c lambda d`` for the knockdown ``lambda``.

    ``lambda`` is the smaller root of ``lambda^2 - (2 + x) lambda + 1`` with
    ``x = c d``, evaluated as ``2 / (2 + x + sqrt(x^2 + 4 x))`` (no
    cancellation for any ``x``). It equals 1 at ``d = 0`` and decreases
    strictly to 0 as ``d -> infinity``; it is trustworthy only for small ``d``.

    Args:
        imperfection_ratio: ``d = delta / t >= 0``.
        coefficient: Structural constant ``c > 0``.

    Returns:
        ``lambda`` in ``(0, 1]``.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If ``d < 0`` or ``c <= 0``.

    Equation:
        [EQ-AERO-BUCK-002], [EQ-AERO-BUCK-003].

    References:
        [REF-NASA-SP-8007-2020].

    """
    ratio = _non_negative(imperfection_ratio, "imperfection_ratio")
    constant = _positive(coefficient, "coefficient")
    x = constant * ratio
    return 2.0 / (2.0 + x + math.sqrt(x * x + 4.0 * x))


def koiter_imperfection_for_knockdown(knockdown: float, coefficient: float) -> float:
    """Return the imperfection ratio ``d = (1 - lambda)^2 / (c lambda)``.

    Exact inverse of :func:`koiter_knockdown`.

    Args:
        knockdown: ``lambda`` in ``(0, 1]``.
        coefficient: Structural constant ``c > 0``.

    Returns:
        ``d >= 0``.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If ``lambda`` is outside ``(0, 1]`` or ``c <= 0``.

    Equation:
        [EQ-AERO-BUCK-003].

    References:
        [REF-NASA-SP-8007-2020].

    """
    lam = require_finite_real(knockdown, name="knockdown")
    if not 0.0 < lam <= 1.0:
        raise ValueError("knockdown must lie in (0, 1]")
    constant = _positive(coefficient, "coefficient")
    return (1.0 - lam) ** 2 / (constant * lam)


def koiter_knockdown_derivative(imperfection_ratio: float, coefficient: float) -> float:
    """Return ``d lambda / d d`` of the Koiter law (implicit differentiation).

    From ``(1 - lambda)^2 = c lambda d``:
    ``lambda' = -c lambda / (2 (1 - lambda) + c d)``. For small ``d`` it
    behaves as ``-sqrt(c) / (2 sqrt(d))``, the asymptotic slope.

    Args:
        imperfection_ratio: ``d > 0`` (the slope is infinite at 0).
        coefficient: Structural constant ``c > 0``.

    Returns:
        The (negative) derivative.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If ``d <= 0`` or ``c <= 0``.

    Equation:
        [EQ-AERO-BUCK-003].

    References:
        [REF-NASA-SP-8007-2020].

    """
    ratio = _positive(imperfection_ratio, "imperfection_ratio")
    constant = _positive(coefficient, "coefficient")
    lam = koiter_knockdown(ratio, constant)
    return -constant * lam / (2.0 * (1.0 - lam) + constant * ratio)


# ---------------------------------------------------------------------------
# SP-8007 empirical knockdown
# ---------------------------------------------------------------------------


def sp8007_axial_knockdown(radius: float, thickness: float) -> float:
    """Return the NASA SP-8007 axial-compression knockdown factor.

    ``gamma = 1 - 0.901 (1 - exp(-phi))``, ``phi = sqrt(R/t) / 16``, valid for
    ``R/t < 1500`` (Eqs. 9 and 10 of NASA/SP-8007-2020/REV 2, which describes
    it as a good lower bound for most test data compiled from the 1930s to
    the 1960s; the report takes it from its reference [101]).

    Args:
        radius: ``R`` (m), positive.
        thickness: ``t`` (m), positive.

    Returns:
        ``gamma`` in ``(0.099, 1)``.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is not positive or ``R/t >= 1500``.

    Equation:
        [EQ-AERO-BUCK-004].

    References:
        [REF-NASA-SP-8007-2020, Eqs. 9-10], [REF-NASA-SP-8007-1968].

    """
    mean_radius = _positive(radius, "radius")
    wall = _positive(thickness, "thickness")
    slenderness = mean_radius / wall
    if slenderness >= SP8007_MAX_RADIUS_TO_THICKNESS:
        raise ValueError("SP-8007 knockdown is defined for R/t < 1500")
    phi = math.sqrt(slenderness) / SP8007_PHI_DIVISOR
    return 1.0 - SP8007_COEFFICIENT * (1.0 - math.exp(-phi))


def sp8007_buckling_load(
    youngs_modulus: float,
    thickness: float,
    radius: float,
    poisson_ratio: float = 0.33,
) -> float:
    """Return the SP-8007 design buckling load ``gamma P_cl``.

    Args:
        youngs_modulus: ``E`` (Pa), positive.
        thickness: ``t`` (m), positive.
        radius: ``R`` (m), positive.
        poisson_ratio: ``nu`` in ``[0, 0.5)``.

    Returns:
        ``gamma P_cl`` (N).

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is outside its domain or ``R/t >= 1500``.

    Equation:
        [EQ-AERO-BUCK-004].

    References:
        [REF-NASA-SP-8007-2020, Eqs. 7 and 9].

    """
    gamma = sp8007_axial_knockdown(radius, thickness)
    return gamma * classical_buckling_load(
        youngs_modulus, thickness, radius, poisson_ratio
    )


def sp8007_design_thickness(
    applied_load: float,
    youngs_modulus: float,
    radius: float,
    poisson_ratio: float = 0.33,
    *,
    safety_factor: float = 1.0,
) -> float:
    """Return the thickness with ``gamma P_cl = safety_factor * P_applied``.

    ``gamma(R/t) P_cl(t)`` increases strictly with ``t`` (thicker walls have a
    smaller ``R/t`` and a larger ``gamma``), so the root is unique; it is
    bracketed in ``R/t`` between 1 and 1500 and found by Brent's method.

    Args:
        applied_load: ``P`` (N), positive.
        youngs_modulus: ``E`` (Pa), positive.
        radius: ``R`` (m), positive.
        poisson_ratio: ``nu`` in ``[0, 0.5)``.
        safety_factor: Factor ``>= 1`` applied to the load.

    Returns:
        The thickness (m).

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is outside its domain or no thickness with
            ``R/t`` in ``(1, 1500)`` carries the load.

    Equation:
        [EQ-AERO-BUCK-004].

    References:
        [REF-NASA-SP-8007-2020, Eqs. 7 and 9], [REF-BRENT-1973].

    """
    load = _positive(applied_load, "applied_load")
    factor = _positive(safety_factor, "safety_factor")
    if factor < 1.0:
        raise ValueError("safety_factor must be at least 1")
    mean_radius = _positive(radius, "radius")
    target = load * factor

    def residual(wall: float) -> float:
        """Return ``gamma P_cl - target`` (N) at wall thickness ``wall``.

        Args:
            wall: Thickness (m).

        Returns:
            The load residual.

        References:
            [EQ-AERO-BUCK-004].

        """
        return sp8007_buckling_load(
            youngs_modulus, wall, mean_radius, poisson_ratio
        ) - (target)

    thin = mean_radius / (SP8007_MAX_RADIUS_TO_THICKNESS - 1e-9)
    thick = mean_radius
    if residual(thin) > 0.0:
        raise ValueError("load is carried even at R/t = 1500; thickness is below range")
    if residual(thick) < 0.0:
        raise ValueError("no thickness with R/t > 1 carries the load")
    root = brent_root(residual, thin, thick, tolerance=1e-14 * mean_radius)
    return float(root.root)


# ---------------------------------------------------------------------------
# Imperfection laws and the probabilistic knockdown
# ---------------------------------------------------------------------------


def lognormal_from_mean_cov(mean: float, cov: float) -> LogNormal:
    """Return the lognormal law with the given mean and coefficient of variation.

    ``sigma^2 = ln(1 + cov^2)``, ``mu = ln(mean) - sigma^2 / 2``.

    Args:
        mean: Mean (positive).
        cov: Coefficient of variation (positive).

    Returns:
        A :class:`~probstat.distributions.LogNormal`.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is not positive.

    Equation:
        [EQ-AERO-BUCK-006].

    References:
        [REF-MELCHERS-BECK-2018].

    """
    m = _positive(mean, "mean")
    c = _positive(cov, "cov")
    variance = math.log1p(c * c)
    return LogNormal(math.log(m) - 0.5 * variance, math.sqrt(variance))


def weibull_from_mean_cov(mean: float, cov: float) -> Weibull:
    """Return the two-parameter Weibull law with the given mean and c.o.v.

    The shape ``k`` solves ``Gamma(1 + 2/k) / Gamma(1 + 1/k)^2 - 1 = cov^2``
    (strictly decreasing in ``k``) by Brent's method; the scale is
    ``mean / Gamma(1 + 1/k)``.

    Args:
        mean: Mean (positive).
        cov: Coefficient of variation in ``[0.02, 20]``.

    Returns:
        A :class:`~probstat.distributions.Weibull`.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-BUCK-006].

    References:
        [REF-NIST-DISTRIBUTIONS].

    """
    m = _positive(mean, "mean")
    c = _positive(cov, "cov")
    if not 0.02 <= c <= 20.0:
        raise ValueError("cov must lie in [0.02, 20]")

    def residual(shape: float) -> float:
        """Return the squared-c.o.v. mismatch at Weibull shape ``shape``.

        Args:
            shape: Weibull shape ``k``.

        Returns:
            ``cov(k)^2 - cov^2``.

        References:
            [EQ-AERO-BUCK-006].

        """
        ratio = math.exp(
            math.lgamma(1.0 + 2.0 / shape) - 2.0 * math.lgamma(1.0 + 1.0 / shape)
        )
        return ratio - 1.0 - c * c

    root = brent_root(residual, 0.05, 200.0, tolerance=1e-12)
    shape = float(root.root)
    return Weibull(shape, m / math.exp(math.lgamma(1.0 + 1.0 / shape)))


def probabilistic_knockdown(
    imperfection: ContinuousMarginal,
    thickness: float,
    probability: float,
    coefficient: float,
) -> float:
    """Return ``k_p`` with ``P(lambda < k_p) = p`` for a random imperfection.

    ``lambda`` decreases strictly in ``d = delta / t``, so
    ``P(lambda < k) = P(delta > t d(k))`` and ``k_p = lambda(q_(1-p) / t)``
    with ``q_(1-p)`` the ``(1 - p)``-quantile of the amplitude law. The result
    depends entirely on the assumed imperfection law and on the Koiter
    coefficient.

    Args:
        imperfection: Law of the imperfection amplitude ``delta`` (m).
        thickness: Wall thickness ``t`` (m), positive.
        probability: ``p`` in ``(0, 1)``.
        coefficient: Koiter constant ``c > 0``.

    Returns:
        ``k_p`` in ``(0, 1]``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``thickness``, ``probability`` or ``coefficient`` is
            out of range.

    Equation:
        [EQ-AERO-BUCK-005].

    References:
        [REF-MELCHERS-BECK-2018].

    """
    wall = _positive(thickness, "thickness")
    p = require_finite_real(probability, name="probability")
    if not 0.0 < p < 1.0:
        raise ValueError("probability must lie in (0, 1)")
    survival_inverse = getattr(imperfection, "isf", None)
    if callable(survival_inverse):
        quantile = float(survival_inverse(p))
    else:
        quantile = float(imperfection.quantile(1.0 - p))
    return koiter_knockdown(max(quantile, 0.0) / wall, coefficient)


@dataclass(frozen=True, slots=True)
class KnockdownComparison:
    """Probabilistic Koiter knockdown against SP-8007 over a range of ``R/t``.

    Args:
        radius_to_thickness: The ``R/t`` values.
        probabilistic: ``k_p`` of the imperfection law at each ``R/t``.
        sp8007: SP-8007 ``gamma`` at each ``R/t``.
        probability: The target ``p``.

    Attributes:
        sp8007_conservative: Per point, whether ``gamma <= k_p``.

    References:
        [EQ-AERO-BUCK-005], [REF-NASA-SP-8007-2020].

    """

    radius_to_thickness: tuple[float, ...]
    probabilistic: tuple[float, ...]
    sp8007: tuple[float, ...]
    probability: float

    @property
    def sp8007_conservative(self) -> tuple[bool, ...]:
        """Return, per point, whether the SP-8007 value is not above ``k_p``.

        References:
            [EQ-AERO-BUCK-005].

        """
        return tuple(
            g <= k for g, k in zip(self.sp8007, self.probabilistic, strict=True)
        )


def compare_knockdowns(
    imperfection: ContinuousMarginal,
    radius: float,
    radius_to_thickness: tuple[float, ...],
    probability: float,
    coefficient: float,
) -> KnockdownComparison:
    """Compare ``k_p`` with SP-8007 at fixed radius for several ``R/t``.

    The imperfection law is an absolute amplitude (m) held fixed while the
    thickness ``t = R / (R/t)`` varies, so thinner shells have larger
    ``delta / t``. The statement "SP-8007 is conservative" is therefore
    conditional on that law.

    Args:
        imperfection: Law of the imperfection amplitude (m).
        radius: ``R`` (m), positive.
        radius_to_thickness: Values of ``R/t``, each in ``(1, 1500)``.
        probability: ``p`` in ``(0, 1)``.
        coefficient: Koiter constant ``c > 0``.

    Returns:
        The comparison table.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range or the list is empty.

    Equation:
        [EQ-AERO-BUCK-005].

    References:
        [REF-NASA-SP-8007-2020, Eqs. 9-10].

    """
    mean_radius = _positive(radius, "radius")
    ratios = tuple(_positive(r, "radius_to_thickness") for r in radius_to_thickness)
    if not ratios:
        raise ValueError("radius_to_thickness must not be empty")
    k_p = []
    gamma = []
    for ratio in ratios:
        wall = mean_radius / ratio
        k_p.append(
            probabilistic_knockdown(imperfection, wall, probability, coefficient)
        )
        gamma.append(sp8007_axial_knockdown(mean_radius, wall))
    return KnockdownComparison(ratios, tuple(k_p), tuple(gamma), float(probability))


# ---------------------------------------------------------------------------
# Probabilistic model
# ---------------------------------------------------------------------------

_UNITS = {
    "youngs_modulus": "Pa",
    "thickness": "m",
    "radius": "m",
    "imperfection": "m",
    "applied_load": "N",
}

_QUANTITIES = (
    QuantityOfInterest("buckling_load", "Imperfect-shell buckling load", "N"),
    QuantityOfInterest("knockdown", "Koiter knockdown P_cr / P_cl", "-"),
    QuantityOfInterest("classical_load", "Classical buckling load", "N"),
    QuantityOfInterest("margin", "Buckling load minus applied load", "N"),
)


@dataclass(frozen=True, slots=True)
class _BucklingForward:
    """Picklable forward map of :func:`shell_buckling_model`.

    Args:
        poisson_ratio: Fixed ``nu``.
        coefficient: Koiter constant ``c``.

    References:
        [EQ-AERO-BUCK-006].

    """

    poisson_ratio: float
    coefficient: float

    def __call__(self, parameters: Mapping[str, float]) -> dict[str, float]:
        """Evaluate the outputs for one realization.

        Args:
            parameters: Values of the five model inputs.

        Returns:
            Buckling load, knockdown, classical load and margin.

        Raises:
            ValueError: If a draw is non-physical.

        Equation:
            [EQ-AERO-BUCK-006].

        References:
            [REF-NASA-SP-8007-2020].

        """
        wall = parameters["thickness"]
        classical = classical_buckling_load(
            parameters["youngs_modulus"], wall, parameters["radius"], self.poisson_ratio
        )
        lam = koiter_knockdown(parameters["imperfection"] / wall, self.coefficient)
        load = _positive(parameters["applied_load"], "applied_load")
        return {
            "buckling_load": lam * classical,
            "knockdown": lam,
            "classical_load": classical,
            "margin": lam * classical - load,
        }


def shell_buckling_model(
    *,
    youngs_modulus: Marginal | float,
    thickness: Marginal | float,
    radius: Marginal | float,
    imperfection: Marginal | float,
    applied_load: Marginal | float,
    poisson_ratio: float = 0.33,
    coefficient: float | None = None,
    name: str = "shell-buckling",
    on_failure: FailurePolicy = "raise",
) -> ProbabilisticModel:
    """Build a probabilistic axial-buckling model of a thin cylinder.

    Each argument is a distribution (independent
    :class:`~probstat.modeling.RandomInput`) or a plain number (fixed). Units:
    ``youngs_modulus`` Pa, ``thickness`` and ``radius`` m, ``imperfection``
    (amplitude ``delta``, m, non-negative; a lognormal or Weibull law is the
    usual choice, see :func:`lognormal_from_mean_cov` and
    :func:`weibull_from_mean_cov`), ``applied_load`` N. Outputs:
    ``buckling_load = lambda P_cl`` (N), ``knockdown = lambda``,
    ``classical_load`` (N) and ``margin = P_cr - P_applied`` (N); failure is
    ``margin <= 0``. A non-physical draw fails the run (see ``on_failure``),
    so use bounded or positive laws. Inputs are independent. See the module
    docstring for the Koiter constant and the limitations.

    Args:
        youngs_modulus: ``E``.
        thickness: ``t``.
        radius: ``R``.
        imperfection: ``delta``.
        applied_load: ``P``.
        poisson_ratio: Fixed ``nu``.
        coefficient: Koiter constant; ``None`` uses :func:`koiter_coefficient`.
        name: Model label.
        on_failure: Failure policy.

    Returns:
        A :class:`~probstat.modeling.ProbabilisticModel`.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If a fixed value is outside its domain or no input is
            random.

    Equation:
        [EQ-AERO-BUCK-006].

    References:
        [REF-NASA-SP-8007-2020].

    """
    nu = _poisson(poisson_ratio)
    constant = koiter_coefficient(nu) if coefficient is None else coefficient
    constant = _positive(constant, "coefficient")
    given: dict[str, Marginal | float] = {
        "youngs_modulus": youngs_modulus,
        "thickness": thickness,
        "radius": radius,
        "imperfection": imperfection,
        "applied_load": applied_load,
    }
    random_inputs: list[RandomInput] = []
    fixed: dict[str, float] = {}
    for key, value in given.items():
        if isinstance(value, bool):
            raise TypeError(f"{key} must be a number or a distribution")
        if isinstance(value, int | float):
            fixed[key] = (
                _non_negative(value, key)
                if key == "imperfection"
                else _positive(value, key)
            )
        else:
            random_inputs.append(RandomInput(key, value, unit=_UNITS[key]))
    if not random_inputs:
        raise ValueError("at least one input must be uncertain")
    return ProbabilisticModel(
        JointInput.independent(*random_inputs),
        _BucklingForward(nu, constant),
        _QUANTITIES,
        name=name,
        fixed=fixed,
        on_failure=on_failure,
    )


# ---------------------------------------------------------------------------
# Reliability-based thickness
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThicknessDesign:
    """Thickness meeting a target failure probability by FORM.

    Args:
        thickness: Nominal (mean) thickness (m).
        probability: FORM failure probability at ``thickness``.
        reliability_index: FORM index ``beta`` at ``thickness``.
        target_probability: The requested probability.
        iterations: Brent iterations.
        converged: Whether the root search converged.

    References:
        [EQ-AERO-BUCK-007].

    """

    thickness: float
    probability: float
    reliability_index: float
    target_probability: float
    iterations: int
    converged: bool


def design_thickness_for_reliability(
    *,
    target_probability: float,
    youngs_modulus: Marginal | float,
    radius: Marginal | float,
    imperfection: Marginal | float,
    applied_load: Marginal | float,
    thickness_cov: float = 0.03,
    poisson_ratio: float = 0.33,
    coefficient: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> ThicknessDesign:
    """Find the mean thickness whose FORM failure probability is the target.

    The wall thickness is lognormal with mean ``t`` and coefficient of
    variation ``thickness_cov``. ``beta(t)`` increases with ``t`` (a thicker
    wall has both a higher ``P_cl`` and a smaller ``delta / t``), so
    ``beta(t) - beta_target`` is solved by Brent's method on a bracket
    ``[lower, upper]`` (default ``[R / 1000, R / 10]`` for a fixed radius; give
    them when ``radius`` is random). FORM must converge at both ends: far
    inside the failure domain it may not, and a ``RuntimeError`` then asks for
    a tighter bracket.

    Args:
        target_probability: Target failure probability in ``(0, 0.5)``.
        youngs_modulus: ``E`` (Pa) law or number.
        radius: ``R`` (m) law or number.
        imperfection: ``delta`` (m) law or number.
        applied_load: ``P`` (N) law or number.
        thickness_cov: Coefficient of variation of the thickness.
        poisson_ratio: Fixed ``nu``.
        coefficient: Koiter constant; ``None`` uses :func:`koiter_coefficient`.
        lower: Lower bracket (m); default ``R / 1000`` for a fixed radius.
        upper: Upper bracket (m); default ``R / 10`` for a fixed radius.

    Returns:
        The design and its achieved FORM probability.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range, the bracket is missing or
            does not change sign.
        RuntimeError: If a FORM search does not converge.

    Equation:
        [EQ-AERO-BUCK-007].

    References:
        [REF-HASOFER-LIND-1974], [REF-MELCHERS-BECK-2018].

    """
    p = require_finite_real(target_probability, name="target_probability")
    if not 0.0 < p < 0.5:
        raise ValueError("target_probability must lie in (0, 0.5)")
    cov = _positive(thickness_cov, "thickness_cov")
    target_beta = -Normal(0.0, 1.0).quantile(p)
    if lower is None or upper is None:
        if not isinstance(radius, int | float) or isinstance(radius, bool):
            raise ValueError("lower and upper are required when radius is random")
        lower = float(radius) / 1000.0 if lower is None else lower
        upper = float(radius) / 10.0 if upper is None else upper
    lo = _positive(lower, "lower")
    hi = _positive(upper, "upper")
    if lo >= hi:
        raise ValueError("lower must be below upper")

    def form_at(wall: float) -> tuple[float, float]:
        """Return the FORM ``(beta, probability)`` at mean thickness ``wall``.

        Args:
            wall: Mean thickness (m).

        Returns:
            The reliability index and failure probability.

        Raises:
            RuntimeError: If FORM does not converge.

        References:
            [EQ-AERO-BUCK-007].

        """
        model = shell_buckling_model(
            youngs_modulus=youngs_modulus,
            thickness=lognormal_from_mean_cov(wall, cov),
            radius=radius,
            imperfection=imperfection,
            applied_load=applied_load,
            poisson_ratio=poisson_ratio,
            coefficient=coefficient,
        )
        result = form_hlrf(ModelLimitState(model, "margin", 0.0, "below"))
        if not result.converged:
            raise RuntimeError(f"FORM did not converge at t = {wall:g} m")
        return result.beta, result.probability

    def residual(wall: float) -> float:
        """Return ``beta(t) - beta_target``.

        Args:
            wall: Mean thickness (m).

        Returns:
            The index residual.

        References:
            [EQ-AERO-BUCK-007].

        """
        return form_at(wall)[0] - target_beta

    root = brent_root(residual, lo, hi, tolerance=1e-10 * hi)
    beta, probability = form_at(float(root.root))
    return ThicknessDesign(
        float(root.root), probability, beta, p, root.iterations, bool(root.converged)
    )
