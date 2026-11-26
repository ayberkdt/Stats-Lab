"""Synthetic spherical-harmonic gravity-field inversion with altitude attenuation.

Experimental (added in 0.28).

This module is a SYNTHETIC TEST BED. It supports no claim about any real
planet, mission or interior: the coefficients, the orbit geometry and the data
are generated locally from a stated model and seed. It is NOT real
satellite-tracking data and no real data set is used or imitated. External
gravity alone does not determine an interior density distribution (Newton's
theorem and the equivalent-source ambiguity, see
:mod:`probstat.inverse.benchmark_gravity`); only the exterior field
coefficients are estimated here.

Conventions (fixed explicitly, nothing is left to a reader's guess). Units are
canonical: the reference radius is ``R = 1``, the gravitational parameter is
``GM = 1``, coefficients are dimensionless and gravity is measured in
``GM / R^2``. The exterior potential at radius ``r >= R``, colatitude
``theta`` and longitude ``phi`` is [EQ-INV-APP-GRAV-007]::

    V = (GM / r) sum_l (R / r)^l sum_{m=0}^{l}
            (C_lm cos(m phi) + S_lm sin(m phi)) Pbar_lm(cos theta),

with REAL, FULLY NORMALIZED (4 pi, geodesy) associated Legendre functions
``Pbar_lm = sqrt((2 - delta_m0)(2 l + 1)(l - m)! / (l + m)!) P_lm`` WITHOUT the
Condon-Shortley phase, so that ``(1 / 4 pi) integral (Pbar_lm cos m phi)^2
d Omega = 1`` and the addition theorem ``sum_m Pbar_lm(t)^2 = 2 l + 1`` holds
[EQ-INV-APP-GRAV-006]. They are computed by the standard forward-column
recursion::

    Pbar_00 = 1, Pbar_11 = sqrt(3) sin(theta),
    Pbar_mm = sqrt((2m + 1) / (2m)) sin(theta) Pbar_{m-1,m-1}   (m >= 2),
    Pbar_{m+1,m} = sqrt(2m + 3) cos(theta) Pbar_mm,
    Pbar_lm = a_lm cos(theta) Pbar_{l-1,m} - b_lm Pbar_{l-2,m},
    a_lm = sqrt((4 l^2 - 1) / (l^2 - m^2)),
    b_lm = sqrt((2 l + 1)(l + m - 1)(l - m - 1) / ((2 l - 3)(l^2 - m^2))),

which involves only square roots of ratios of small integers and no
factorials; it is stable at the degrees used here (the recursion in ``l`` at
fixed ``m`` is the stable direction). Degrees above 100 are rejected because
the ``sin^m(theta)`` seed underflows near the poles (the Holmes-Featherstone
scaling needed beyond that is not implemented). The orthonormality of the
table is verified by Gauss-Legendre quadrature in
:func:`legendre_orthonormality_error`.

Observations. The radial component of the gravitational acceleration directed
towards the centre, ``g = -dV/dr``, at satellite radius ``r = R + h``, is linear
in the coefficients [EQ-INV-APP-GRAV-007]::

    g = (GM / r^2) sum_l (l + 1) (R / r)^l sum_m (C_lm cos m phi
            + S_lm sin m phi) Pbar_lm(cos theta)
      = (GM / R^2) sum_l (l + 1) (R / r)^(l + 2) ... .

The factor ``(R / r)^(l + 2)`` is the DEGREE-DEPENDENT ATTENUATION: at altitude
``h`` the degree-``l`` signal is reduced by ``(1 + h / R)^-(l + 2)`` and the
singular values of the design matrix fall accordingly. For a quasi-uniform
design with ``m_obs`` points the design columns are nearly orthogonal and the
``2 l + 1`` singular values of degree ``l`` cluster near ``sqrt(m_obs) (l + 1)
(R / r)^(l + 2)`` [EQ-INV-APP-GRAV-011]. The factor ``l + 1`` from the radial
derivative competes with the attenuation: at ``h = 0.1 R`` the plain spectrum
is nearly flat up to degree about ten (``(l + 1) 1.1^-(l + 2)`` peaks at
``l = 9``), whereas at ``h = 0.5 R`` it falls by a factor of seven between
degrees 2 and 10; the loss of high-degree information at low altitude is
carried mainly by the Kaula prior in the whitened spectrum. Degrees ``0`` and
``1`` (the central term and the centre-of-mass offset) are not estimated: the
unknown vector holds the degrees ``l_min = 2`` to ``L``.

Prior. A Kaula-type degree-variance prior with independent coefficients
``C_lm, S_lm ~ N(0, s_l^2)``, ``s_l = kappa / l^p`` (``p = 2`` is the classical
Kaula rule of thumb; ``kappa`` is a declared scale, here a synthetic value)
[EQ-INV-APP-GRAV-008]. The degree variance of the observed signal is
``(2 l + 1)(l + 1)^2 (R / r)^(2 l + 4) s_l^2``.

Posterior. For ``y = A c + e``, ``e ~ N(0, sigma^2 I)``, the exact Gaussian
posterior has precision ``A'A / sigma^2 + Lambda^-1``; the resolution matrix
is ``I - Sigma Lambda^-1`` and its diagonal is the fraction of prior variance
removed by the data. A degree is called RESOLVED when the mean resolved
fraction of its coefficients is at least a declared threshold (default one
half), and the resolved degree is the largest ``l`` such that every degree up
to ``l`` is resolved [EQ-INV-APP-GRAV-009]. This is a posterior-variance
criterion under the declared prior and noise; it is not a statement about any
real field.

No inverse crime. The truth has a HIGHER maximum degree than the solver, so
the data contain an omission signal the solver cannot represent
(``truth_degree > solver_degree``, ``same_discretization = False``); the
omission error is reported and its expected size is available in closed form
from the prior [EQ-INV-APP-GRAV-010]. The ``"mascon"`` truth is the field of
point masses in closed form (independent of any harmonic expansion) with the
degree 0 and 1 parts removed. With ``truth_degree == solver_degree`` the
benchmark is algebraic verification only and is flagged.

References:
    [REF-KAULA-1966], [REF-HEISKANEN-MORITZ-1967],
    [REF-HOLMES-FEATHERSTONE-2002], [REF-BLAKELY-1995],
    [REF-KAIPIO-SOMERSALO-2007], [REF-HANSEN-2008].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import resolve_stream
from probstat.linalg import (
    Matrix,
    Vector,
    inverse_spd,
    log_det_spd,
    singular_value_decomposition,
    symmetric_eigen,
)

SYNTHETIC_NOTICE = (
    "SYNTHETIC: coefficients, orbit geometry and data are generated locally "
    "from a declared model and seed. No real planet, mission or tracking "
    "data is represented, and nothing here constrains any interior."
)

_MAX_DEGREE = 100
_TRUTHS = ("kaula", "mascon")


def _finite(value: object, name: str) -> float:
    """Return ``value`` as a finite float.

    References:
        [EQ-INV-APP-GRAV-007]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive(value: object, name: str) -> float:
    """Return ``value`` as a finite positive float.

    References:
        [EQ-INV-APP-GRAV-007]. No equation is evaluated.

    """
    number = _finite(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _integer(value: object, name: str, minimum: int) -> int:
    """Return ``value`` as an integer at least ``minimum``.

    References:
        [EQ-INV-APP-GRAV-007]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _degree(value: object, name: str, minimum: int = 0) -> int:
    """Return a validated harmonic degree.

    References:
        [EQ-INV-APP-GRAV-006]. No equation is evaluated.

    """
    degree = _integer(value, name, minimum)
    if degree > _MAX_DEGREE:
        raise ValueError(
            f"{name} must not exceed {_MAX_DEGREE}: the sin^m seed of the "
            "recursion underflows near the poles beyond that"
        )
    return degree


# ---------------------------------------------------------------------------
# Fully normalized associated Legendre functions
# ---------------------------------------------------------------------------


def normalized_legendre_table(max_degree: int, colatitude: float) -> list[list[float]]:
    """Return ``Pbar_lm(cos theta)`` for ``0 <= m <= l <= max_degree``.

    Fully normalized (4 pi), no Condon-Shortley phase, by the forward-column
    recursion of the module docstring [EQ-INV-APP-GRAV-006]. ``table[l][m]``
    is ``Pbar_lm``.

    Args:
        max_degree: Largest degree ``L`` (``0`` to ``100``).
        colatitude: ``theta`` in ``[0, pi]`` (radians).

    Returns:
        The triangular table (row ``l`` has ``l + 1`` entries).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``max_degree`` is out of range or ``theta`` is not in
            ``[0, pi]``.

    References:
        [EQ-INV-APP-GRAV-006], [REF-HOLMES-FEATHERSTONE-2002],
        [REF-HEISKANEN-MORITZ-1967].

    """
    top = _degree(max_degree, "max_degree")
    theta = _finite(colatitude, "colatitude")
    if not 0.0 <= theta <= math.pi:
        raise ValueError("colatitude must lie in [0, pi]")
    t = math.cos(theta)
    s = math.sin(theta)
    table = [[0.0] * (ell + 1) for ell in range(top + 1)]
    table[0][0] = 1.0
    for m in range(1, top + 1):
        factor = math.sqrt(3.0) if m == 1 else math.sqrt((2 * m + 1) / (2 * m))
        table[m][m] = factor * s * table[m - 1][m - 1]
    for m in range(top):
        table[m + 1][m] = math.sqrt(2 * m + 3) * t * table[m][m]
    for m in range(top + 1):
        for ell in range(m + 2, top + 1):
            denominator = ell * ell - m * m
            a = math.sqrt((4 * ell * ell - 1) / denominator)
            b = math.sqrt(
                (2 * ell + 1)
                * (ell + m - 1)
                * (ell - m - 1)
                / ((2 * ell - 3) * denominator)
            )
            table[ell][m] = a * t * table[ell - 1][m] - b * table[ell - 2][m]
    return table


def _gauss_legendre(order: int) -> tuple[Vector, Vector]:
    """Return Gauss-Legendre nodes and weights on ``[-1, 1]`` (Newton).

    References:
        [EQ-INV-APP-GRAV-006]. No equation is evaluated.

    """
    nodes: list[float] = []
    weights: list[float] = []
    for i in range(order):
        x = math.cos(math.pi * (i + 0.75) / (order + 0.5))
        derivative = 1.0
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, order + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            derivative = order * (x * p1 - p0) / (x * x - 1.0)
            step = p1 / derivative
            x -= step
            if abs(step) < 1e-15:
                break
        nodes.append(x)
        weights.append(2.0 / ((1.0 - x * x) * derivative * derivative))
    pairs = sorted(zip(nodes, weights, strict=True))
    return tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)


def legendre_orthonormality_error(max_degree: int) -> float:
    """Return the worst deviation from orthonormality of the table.

    For every order ``m`` and degrees ``l, l' >= m`` the quantity
    ``(1 / 2) int_-1^1 Pbar_lm Pbar_l'm dt`` must equal ``(2 - delta_m0)
    delta_ll'`` (the longitude integrals of the trigonometric factors are
    exact and supply the remaining ``pi (1 + delta_m0)``, giving the 4 pi
    normalization). The integral is evaluated by ``L + 1``-point
    Gauss-Legendre quadrature, which is exact because the integrand is a
    polynomial in ``t`` of degree at most ``2 L`` [EQ-INV-APP-GRAV-006].

    Args:
        max_degree: Largest degree ``L`` (``0`` to ``100``).

    Returns:
        The largest absolute deviation over all ``(m, l, l')``.

    Raises:
        TypeError: If ``max_degree`` is not an integer.
        ValueError: If ``max_degree`` is out of range.

    References:
        [EQ-INV-APP-GRAV-006], [REF-HEISKANEN-MORITZ-1967].

    """
    top = _degree(max_degree, "max_degree")
    nodes, weights = _gauss_legendre(top + 1)
    tables = [normalized_legendre_table(top, math.acos(x)) for x in nodes]
    worst = 0.0
    for m in range(top + 1):
        for ell in range(m, top + 1):
            for other in range(ell, top + 1):
                value = 0.5 * math.fsum(
                    w * tab[ell][m] * tab[other][m]
                    for w, tab in zip(weights, tables, strict=True)
                )
                target = (1.0 if m == 0 else 2.0) if ell == other else 0.0
                worst = max(worst, abs(value - target))
    return worst


# ---------------------------------------------------------------------------
# Coefficients, design and observation geometry
# ---------------------------------------------------------------------------


def coefficient_labels(
    max_degree: int, min_degree: int = 2
) -> tuple[tuple[int, int, str], ...]:
    """Return the ordered labels ``(l, m, "C" or "S")`` of the unknowns.

    Order: by degree, then all ``C_lm`` (``m = 0..l``), then all ``S_lm``
    (``m = 1..l``); ``S_l0`` does not exist. The count is
    ``(L + 1)^2 - l_min^2`` [EQ-INV-APP-GRAV-007].

    Args:
        max_degree: Largest degree ``L``.
        min_degree: Smallest degree ``l_min`` (``0`` to ``L``).

    Returns:
        The labels.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If the degrees are out of range.

    References:
        [EQ-INV-APP-GRAV-007].

    """
    top = _degree(max_degree, "max_degree")
    bottom = _degree(min_degree, "min_degree")
    if bottom > top:
        raise ValueError("min_degree must not exceed max_degree")
    labels: list[tuple[int, int, str]] = []
    for ell in range(bottom, top + 1):
        labels += [(ell, m, "C") for m in range(ell + 1)]
        labels += [(ell, m, "S") for m in range(1, ell + 1)]
    return tuple(labels)


def sphere_design(
    count: int, *, cap_colatitude: float = math.pi
) -> tuple[Vector, Vector]:
    """Return a quasi-uniform Fibonacci point set on a spherical cap.

    With ``u_i = 1 - (1 - cos c)(2 i + 1) / (2 N)`` and
    ``phi_i = 2 pi i / golden_ratio`` (mod ``2 pi``), the colatitudes are
    ``acos(u_i)``; ``cap_colatitude = pi`` covers the whole sphere and a
    smaller value a polar cap (a deliberately poor design used for the
    failure study) [EQ-INV-APP-GRAV-007].

    Args:
        count: Number of points (``>= 1``).
        cap_colatitude: Largest colatitude ``c`` in ``(0, pi]``.

    Returns:
        ``(colatitudes, longitudes)``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``count < 1`` or the cap is out of range.

    References:
        [EQ-INV-APP-GRAV-007].

    """
    n = _integer(count, "count", 1)
    cap = _finite(cap_colatitude, "cap_colatitude")
    if not 0.0 < cap <= math.pi:
        raise ValueError("cap_colatitude must lie in (0, pi]")
    golden = 0.5 * (1.0 + math.sqrt(5.0))
    span = 1.0 - math.cos(cap)
    colatitudes = []
    longitudes = []
    for i in range(n):
        u = 1.0 - span * (2 * i + 1) / (2.0 * n)
        colatitudes.append(math.acos(max(-1.0, min(1.0, u))))
        longitudes.append((2.0 * math.pi * i / golden) % (2.0 * math.pi))
    return tuple(colatitudes), tuple(longitudes)


@dataclass(frozen=True, slots=True)
class GravityDesign:
    """Linear radial-gravity forward operator in the harmonic coefficients.

    Attributes:
        colatitudes: Observation colatitudes (radians).
        longitudes: Observation longitudes (radians).
        radii: Observation radii in units of ``R`` (``>= 1``).
        max_degree: Largest degree ``L`` of the unknowns.
        min_degree: Smallest degree (default 2).
        labels: ``(l, m, "C"/"S")`` per column.
        matrix: ``A`` (``m_obs x n``), ``A_ij = (l + 1) r_i^-(l + 2)
            Pbar_lm(cos theta_i) {cos, sin}(m phi_i)`` in units of
            ``GM / R^2``.

    References:
        [EQ-INV-APP-GRAV-007].

    """

    colatitudes: Vector
    longitudes: Vector
    radii: Vector
    max_degree: int
    min_degree: int
    labels: tuple[tuple[int, int, str], ...]
    matrix: Matrix

    @property
    def observations(self) -> int:
        """Return the number of observations.

        Returns:
            ``m_obs``.

        References:
            [EQ-INV-APP-GRAV-007].

        """
        return len(self.matrix)

    @property
    def unknowns(self) -> int:
        """Return the number of coefficients.

        Returns:
            ``n``.

        References:
            [EQ-INV-APP-GRAV-007].

        """
        return len(self.labels)


def gravity_design(
    colatitudes: Sequence[float],
    longitudes: Sequence[float],
    radii: Sequence[float] | float,
    max_degree: int,
    *,
    min_degree: int = 2,
) -> GravityDesign:
    """Build the radial-gravity design matrix of a set of observations.

    Row ``i``, column ``(l, m, C)``: ``(l + 1) r_i^-(l + 2) Pbar_lm(cos
    theta_i) cos(m phi_i)``; column ``(l, m, S)`` uses ``sin(m phi_i)``. The
    observable is ``-dV/dr`` in units of ``GM / R^2`` [EQ-INV-APP-GRAV-007].

    Args:
        colatitudes: ``theta_i`` in ``[0, pi]``.
        longitudes: ``phi_i`` (any real value).
        radii: One radius in units of ``R`` for all points, or one per
            point (each ``>= 1``).
        max_degree: Largest degree ``L`` (``0`` to ``100``).
        min_degree: Smallest degree.

    Returns:
        :class:`GravityDesign`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If lengths differ, a coordinate is out of range or a
            radius is below ``1``.

    References:
        [EQ-INV-APP-GRAV-007], [REF-HEISKANEN-MORITZ-1967].

    """
    thetas = tuple(_finite(v, "colatitude") for v in colatitudes)
    phis = tuple(_finite(v, "longitude") for v in longitudes)
    if not thetas or len(thetas) != len(phis):
        raise ValueError("colatitudes and longitudes must be non-empty and equal")
    rs: Vector
    if isinstance(radii, (int, float)) and not isinstance(radii, bool):
        rs = (_finite(radii, "radius"),) * len(thetas)
    elif isinstance(radii, bool):
        raise TypeError("radii must be a real number or a sequence")
    else:
        rs = tuple(_finite(v, "radius") for v in radii)
    if len(rs) != len(thetas):
        raise ValueError("radii must be one value or one per observation")
    if min(rs) < 1.0:
        raise ValueError("radii must be at least 1 (the reference radius)")
    labels = coefficient_labels(max_degree, min_degree)
    top = _degree(max_degree, "max_degree")
    rows = []
    for theta, phi, radius in zip(thetas, phis, rs, strict=True):
        table = normalized_legendre_table(top, theta)
        trig = [(math.cos(m * phi), math.sin(m * phi)) for m in range(top + 1)]
        row = []
        for ell, m, kind in labels:
            scale = (ell + 1) * radius ** (-(ell + 2))
            angular = trig[m][0] if kind == "C" else trig[m][1]
            row.append(scale * table[ell][m] * angular)
        rows.append(tuple(row))
    return GravityDesign(
        thetas, phis, rs, top, _degree(min_degree, "min_degree"), labels, tuple(rows)
    )


def potential(
    colatitude: float,
    longitude: float,
    radius: float,
    coefficients: Sequence[float],
    max_degree: int,
    *,
    min_degree: int = 2,
) -> float:
    """Return the anomalous potential ``V`` of a coefficient vector.

    Evaluates the series of the module docstring for the degrees
    ``min_degree`` to ``max_degree`` (``GM = R = 1``); it is the independent
    object whose radial derivative the design matrix must reproduce
    [EQ-INV-APP-GRAV-007].

    Args:
        colatitude: ``theta``.
        longitude: ``phi``.
        radius: ``r / R`` (``>= 1``).
        coefficients: Values in the order of :func:`coefficient_labels`.
        max_degree: Largest degree.
        min_degree: Smallest degree.

    Returns:
        ``V`` in units of ``GM / R``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the coefficient count is wrong or a coordinate is out
            of range.

    References:
        [EQ-INV-APP-GRAV-007].

    """
    labels = coefficient_labels(max_degree, min_degree)
    values = tuple(_finite(v, "coefficient") for v in coefficients)
    if len(values) != len(labels):
        raise ValueError(f"expected {len(labels)} coefficients")
    r = _finite(radius, "radius")
    if r < 1.0:
        raise ValueError("radius must be at least 1")
    phi = _finite(longitude, "longitude")
    table = normalized_legendre_table(max_degree, colatitude)
    total = 0.0
    for (ell, m, kind), c in zip(labels, values, strict=True):
        angular = math.cos(m * phi) if kind == "C" else math.sin(m * phi)
        total += r ** (-(ell + 1)) * c * table[ell][m] * angular
    return total


def point_mass_coefficients(
    colatitude: float,
    longitude: float,
    radius: float,
    mass: float,
    max_degree: int,
    *,
    min_degree: int = 2,
) -> Vector:
    """Return the harmonic coefficients of an interior point mass.

    For a mass ``q`` (in units of ``GM``) at radius ``r_s < R`` the addition
    theorem ``P_l(cos gamma) = (1 / (2 l + 1)) sum_m Pbar_lm Pbar_lm'
    cos(m (phi - phi'))`` gives ``C_lm = q r_s^l Pbar_lm(cos theta_s) cos(m
    phi_s) / (2 l + 1)`` and the same with ``sin`` for ``S_lm``
    [EQ-INV-APP-GRAV-007]. This is the independent analytic case of the
    tests: the series must sum to the closed-form attraction of the point.

    Args:
        colatitude: ``theta_s``.
        longitude: ``phi_s``.
        radius: ``r_s`` in units of ``R`` (in ``(0, 1)``).
        mass: ``q`` (any finite real).
        max_degree: Largest degree.
        min_degree: Smallest degree.

    Returns:
        Coefficients in the order of :func:`coefficient_labels`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the radius is not in ``(0, 1)`` or a degree is invalid.

    References:
        [EQ-INV-APP-GRAV-007], [REF-HEISKANEN-MORITZ-1967].

    """
    rs = _positive(radius, "radius")
    if rs >= 1.0:
        raise ValueError("the point mass must lie inside the reference sphere")
    q = _finite(mass, "mass")
    phi = _finite(longitude, "longitude")
    table = normalized_legendre_table(max_degree, colatitude)
    out = []
    for ell, m, kind in coefficient_labels(max_degree, min_degree):
        angular = math.cos(m * phi) if kind == "C" else math.sin(m * phi)
        out.append(q * rs**ell * table[ell][m] * angular / (2 * ell + 1))
    return tuple(out)


def point_mass_radial_gravity(
    colatitude: float,
    longitude: float,
    radius: float,
    source: tuple[float, float, float, float],
) -> float:
    """Return ``-dV/dr`` of a point mass in closed form (no harmonics).

    With the source ``(theta_s, phi_s, r_s, q)`` and ``V = q / rho``,
    ``rho = |x - x_s|``: ``-dV/dr = q (r - r_s cos gamma) / rho^3`` where
    ``cos gamma = x . x_s / (r r_s)`` [EQ-INV-APP-GRAV-007].

    Args:
        colatitude: Observation ``theta``.
        longitude: Observation ``phi``.
        radius: Observation ``r`` (``> r_s``).
        source: ``(theta_s, phi_s, r_s, q)``.

    Returns:
        The inward-positive radial attraction in units of ``GM / R^2``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the observation is not outside the source radius.

    References:
        [EQ-INV-APP-GRAV-007].

    """
    theta_s, phi_s, r_s, q = (_finite(v, "source entry") for v in source)
    theta = _finite(colatitude, "colatitude")
    phi = _finite(longitude, "longitude")
    r = _positive(radius, "radius")
    if r <= r_s:
        raise ValueError("the observation must be outside the source radius")
    cos_gamma = math.cos(theta) * math.cos(theta_s) + math.sin(theta) * math.sin(
        theta_s
    ) * math.cos(phi - phi_s)
    rho2 = r * r + r_s * r_s - 2.0 * r * r_s * cos_gamma
    return float(q * (r - r_s * cos_gamma) / rho2**1.5)


# ---------------------------------------------------------------------------
# Prior, spectrum and exact Gaussian posterior
# ---------------------------------------------------------------------------


def kaula_prior_sd(
    labels: Sequence[tuple[int, int, str]],
    *,
    kappa: float = 1e-5,
    exponent: float = 2.0,
) -> Vector:
    """Return the Kaula-type prior standard deviation of every coefficient.

    ``s_l = kappa / l^p`` (``p = 2`` is the classical rule of thumb for
    degree ``l >= 1``), independent for ``C_lm`` and ``S_lm``. ``kappa`` is a
    declared scale of the SYNTHETIC field; the value ``1e-5`` is the order of
    magnitude of the Kaula rule for a terrestrial-sized field and carries no
    claim about any body [EQ-INV-APP-GRAV-008].

    Args:
        labels: Coefficient labels.
        kappa: Scale (``> 0``).
        exponent: Decay exponent ``p`` (``>= 0``).

    Returns:
        One standard deviation per label.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a label has degree ``0`` or a parameter is invalid.

    References:
        [EQ-INV-APP-GRAV-008], [REF-KAULA-1966].

    """
    scale = _positive(kappa, "kappa")
    p = _finite(exponent, "exponent")
    if p < 0.0:
        raise ValueError("exponent must be non-negative")
    out = []
    for ell, _, _ in labels:
        if ell < 1:
            raise ValueError("the Kaula rule needs degrees of at least 1")
        out.append(scale / ell**p)
    return tuple(out)


def attenuation(degree: int, radius: float) -> float:
    """Return the degree-``l`` observation scale ``(l + 1) r^-(l + 2)``.

    This is the multiplier of the unit-normalized harmonic in the radial
    gravity at radius ``r`` (units of ``R``), the visible altitude
    attenuation ``(R / r)^(l + 2)`` times ``l + 1`` [EQ-INV-APP-GRAV-007].

    Args:
        degree: ``l`` (``>= 0``).
        radius: ``r / R`` (``>= 1``).

    Returns:
        ``(l + 1) r^-(l + 2)``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If out of range.

    References:
        [EQ-INV-APP-GRAV-007].

    """
    ell = _degree(degree, "degree")
    r = _finite(radius, "radius")
    if r < 1.0:
        raise ValueError("radius must be at least 1")
    return (ell + 1) * r ** (-(ell + 2))


def degree_signal_variance(
    degree: int, radius: float, *, kappa: float = 1e-5, exponent: float = 2.0
) -> float:
    """Return the prior variance of the degree-``l`` signal at one point.

    ``(2 l + 1)(l + 1)^2 r^-(2 l + 4) s_l^2``: the ``2 l + 1`` coefficients of
    the degree each contribute ``s_l^2`` times the mean square of a fully
    normalized harmonic, which is ``1`` [EQ-INV-APP-GRAV-008].

    Args:
        degree: ``l`` (``>= 1``).
        radius: ``r / R`` (``>= 1``).
        kappa: Kaula scale.
        exponent: Kaula exponent.

    Returns:
        The variance in units of ``(GM / R^2)^2``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If out of range.

    References:
        [EQ-INV-APP-GRAV-008], [REF-KAULA-1966].

    """
    ell = _degree(degree, "degree", 1)
    s = kaula_prior_sd(((ell, 0, "C"),), kappa=kappa, exponent=exponent)[0]
    return (2 * ell + 1) * attenuation(ell, radius) ** 2 * s * s


def omission_variance(
    solver_degree: int,
    truth_degree: int,
    radius: float,
    *,
    kappa: float = 1e-5,
    exponent: float = 2.0,
) -> float:
    """Return the expected data variance of the omitted degrees.

    ``sum_{l = L_s + 1}^{L_t} (2 l + 1)(l + 1)^2 r^-(2 l + 4) s_l^2``: what a
    solver of degree ``L_s`` cannot represent when the truth extends to
    ``L_t``; compare with ``sigma^2`` to see whether the omission error is
    below the noise [EQ-INV-APP-GRAV-010].

    Args:
        solver_degree: ``L_s`` (``>= 1``).
        truth_degree: ``L_t`` (``>= L_s``).
        radius: ``r / R``.
        kappa: Kaula scale.
        exponent: Kaula exponent.

    Returns:
        The summed degree variances.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``L_t < L_s`` or a value is out of range.

    References:
        [EQ-INV-APP-GRAV-010], [REF-KAULA-1966].

    """
    low = _degree(solver_degree, "solver_degree", 1)
    high = _degree(truth_degree, "truth_degree", low)
    return math.fsum(
        degree_signal_variance(ell, radius, kappa=kappa, exponent=exponent)
        for ell in range(low + 1, high + 1)
    )


@dataclass(frozen=True, slots=True)
class GravitySpectrum:
    """Singular spectrum of the radial-gravity design matrix.

    Attributes:
        singular_values: Descending singular values of ``A``.
        whitened_singular_values: Singular values of ``A diag(s) / sigma``
            (prior-whitened, noise-normalized); ``> 1`` means data-dominated.
        resolved_modes: Number of whitened singular values above one.
        condition_number: Ratio of the largest to the smallest singular value
            of ``A`` (``inf`` if the smallest is zero).
        degree_predictions: Per degree, ``(degree, predicted, measured_min,
            measured_max)`` with ``predicted = sqrt(m_obs) (l + 1) r^-(l + 2)``
            and the measured range over the ``2 l + 1`` singular values
            nearest to it [EQ-INV-APP-GRAV-011].

    References:
        [EQ-INV-APP-GRAV-011].

    """

    singular_values: Vector
    whitened_singular_values: Vector
    resolved_modes: int
    condition_number: float
    degree_predictions: tuple[tuple[int, float, float, float], ...]


def _singular_values(matrix: Matrix, method: str) -> Vector:
    """Return descending singular values by SVD or by the Gram eigenvalues.

    References:
        [EQ-INV-APP-GRAV-011]. No equation is evaluated.

    """
    if method == "svd":
        return singular_value_decomposition(matrix).singular_values
    columns = len(matrix[0])
    gram = tuple(
        tuple(math.fsum(row[j] * row[k] for row in matrix) for k in range(columns))
        for j in range(columns)
    )
    eigenvalues, _ = symmetric_eigen(gram)
    return tuple(math.sqrt(max(v, 0.0)) for v in reversed(eigenvalues))


def gravity_spectrum(
    design: GravityDesign,
    sigma: float,
    prior_sd: Sequence[float],
    *,
    method: str = "svd",
) -> GravitySpectrum:
    """Return the singular spectrum and the cluster prediction by degree.

    The plain spectrum is the thin SVD of ``A``; the whitened one is the SVD of
    ``A diag(prior_sd) / sigma``. The cluster prediction of degree ``l`` is
    ``sqrt(m_obs) (l + 1) r^-(l + 2)`` for a mean altitude radius
    [EQ-INV-APP-GRAV-011]; it is exact only for exactly orthogonal columns, so
    the measured spread around it is reported rather than assumed.

    Args:
        design: The design.
        sigma: Noise standard deviation (``> 0``).
        prior_sd: Prior standard deviations of the coefficients.
        method: ``"svd"`` (one-sided Jacobi SVD of the data matrix, accurate
            for small singular values) or ``"gram"`` (square roots of the
            eigenvalues of ``A'A``, much faster for many unknowns but
            accurate to about ``sqrt(eps) s_1`` in the small singular values).

    Returns:
        :class:`GravitySpectrum`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If dimensions disagree, ``sigma`` is not positive or the
            method is unknown.

    References:
        [EQ-INV-APP-GRAV-011], [REF-HANSEN-2008].

    """
    if not isinstance(design, GravityDesign):
        raise TypeError("design must be a GravityDesign")
    s = _positive(sigma, "sigma")
    sd = tuple(_positive(v, "prior_sd entry") for v in prior_sd)
    if len(sd) != design.unknowns:
        raise ValueError("prior_sd must have one entry per coefficient")
    if method not in ("svd", "gram"):
        raise ValueError("method must be 'svd' or 'gram'")
    plain = _singular_values(design.matrix, method)
    scaled = tuple(
        tuple(a * sd[j] / s for j, a in enumerate(row)) for row in design.matrix
    )
    whitened = _singular_values(scaled, method)
    smallest = plain[-1]
    mean_radius = math.fsum(design.radii) / len(design.radii)
    root = math.sqrt(design.observations)
    predictions = []
    for ell in range(design.min_degree, design.max_degree + 1):
        predicted = root * attenuation(ell, mean_radius)
        nearest = sorted((abs(v - predicted), v) for v in plain)[: 2 * ell + 1]
        values = [v for _, v in nearest]
        predictions.append((ell, predicted, min(values), max(values)))
    return GravitySpectrum(
        plain,
        whitened,
        sum(1 for v in whitened if v > 1.0),
        math.inf if smallest <= 0.0 else plain[0] / smallest,
        tuple(predictions),
    )


@dataclass(frozen=True, slots=True)
class GravityPosterior:
    """Exact Gaussian posterior of the coefficients.

    Attributes:
        mean: Posterior mean (zeros when no data were given).
        covariance: Posterior covariance (dense).
        sd: Marginal posterior standard deviations.
        prior_sd: Prior standard deviations.
        resolution_diagonal: ``1 - Sigma_jj / s_j^2`` per coefficient, the
            fraction of prior variance removed by the data.
        log_evidence: ``log p(y)`` (NaN when no data were given).
        sigma: Noise standard deviation.

    References:
        [EQ-INV-APP-GRAV-009].

    """

    mean: Vector
    covariance: Matrix
    sd: Vector
    prior_sd: Vector
    resolution_diagonal: Vector
    log_evidence: float
    sigma: float


def gravity_posterior(
    design: GravityDesign,
    sigma: float,
    prior_sd: Sequence[float],
    data: Sequence[float] | None = None,
) -> GravityPosterior:
    """Return the exact posterior of ``y = A c + e``, ``c ~ N(0, Lambda)``.

    Precision ``Q = A'A / sigma^2 + Lambda^-1`` (Cholesky, no jitter), mean
    ``Q^-1 A'y / sigma^2``, resolution diagonal ``1 - (Q^-1)_jj / s_j^2`` and
    ``log p(y) = -(1/2)(y'y / sigma^2 - b'Q^-1 b) - (1/2) (m log sigma^2 +
    log det Lambda + log det Q) - (m / 2) log 2 pi`` with ``b = A'y /
    sigma^2`` (matrix-determinant lemma) [EQ-INV-APP-GRAV-009]. With
    ``data=None`` only the covariance quantities are returned, which do not
    depend on the data.

    Args:
        design: The design.
        sigma: Noise standard deviation (``> 0``).
        prior_sd: Prior standard deviations (one per coefficient).
        data: Observations of length ``m_obs`` or ``None``.

    Returns:
        :class:`GravityPosterior`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If dimensions disagree or the precision is not positive
            definite.

    References:
        [EQ-INV-APP-GRAV-009], [REF-KAIPIO-SOMERSALO-2007].

    """
    if not isinstance(design, GravityDesign):
        raise TypeError("design must be a GravityDesign")
    s = _positive(sigma, "sigma")
    sd = tuple(_positive(v, "prior_sd entry") for v in prior_sd)
    n = design.unknowns
    m = design.observations
    if len(sd) != n:
        raise ValueError("prior_sd must have one entry per coefficient")
    a = design.matrix
    columns = [tuple(a[i][j] for i in range(m)) for j in range(n)]
    precision = [
        [
            math.fsum(x * y for x, y in zip(columns[j], columns[k], strict=True))
            / (s * s)
            for k in range(n)
        ]
        for j in range(n)
    ]
    for j in range(n):
        precision[j][j] += 1.0 / (sd[j] * sd[j])
    q = tuple(tuple(r) for r in precision)
    covariance = inverse_spd(q)
    resolution = tuple(1.0 - covariance[j][j] / (sd[j] * sd[j]) for j in range(n))
    marginal = tuple(math.sqrt(covariance[j][j]) for j in range(n))
    if data is None:
        return GravityPosterior(
            (0.0,) * n, covariance, marginal, sd, resolution, math.nan, s
        )
    y = tuple(_finite(v, "data entry") for v in data)
    if len(y) != m:
        raise ValueError("data must have one entry per observation")
    b = tuple(
        math.fsum(c * v for c, v in zip(columns[j], y, strict=True)) / (s * s)
        for j in range(n)
    )
    mean = tuple(math.fsum(covariance[j][k] * b[k] for k in range(n)) for j in range(n))
    quad = math.fsum(v * v for v in y) / (s * s) - math.fsum(
        b[j] * mean[j] for j in range(n)
    )
    log_det = (
        m * math.log(s * s) + math.fsum(math.log(v * v) for v in sd) + log_det_spd(q)
    )
    evidence = -0.5 * (quad + log_det + m * math.log(2.0 * math.pi))
    return GravityPosterior(mean, covariance, marginal, sd, resolution, evidence, s)


@dataclass(frozen=True, slots=True)
class DegreeResolution:
    """Resolution of one degree.

    Attributes:
        degree: ``l``.
        attenuation: ``(l + 1) r^-(l + 2)`` at the mean radius.
        prior_sd: Prior standard deviation of the degree's coefficients.
        posterior_rms_sd: Root-mean-square posterior standard deviation.
        resolved_fraction: Mean of the resolution diagonal over the degree.
        expected_fraction: Orthogonal-design prediction ``tau / (1 + tau)``,
            ``tau = m_obs (attenuation s_l / sigma)^2``.
        resolved: ``resolved_fraction >= threshold``.

    References:
        [EQ-INV-APP-GRAV-009].

    """

    degree: int
    attenuation: float
    prior_sd: float
    posterior_rms_sd: float
    resolved_fraction: float
    expected_fraction: float
    resolved: bool


def degree_resolution(
    design: GravityDesign,
    posterior: GravityPosterior,
    *,
    threshold: float = 0.5,
) -> tuple[DegreeResolution, ...]:
    """Summarize the resolution by degree and flag the resolved ones.

    The expected fraction uses the near-orthogonality of the quasi-uniform
    design: each coefficient of degree ``l`` is observed with information
    ``m_obs (attenuation_l)^2 / sigma^2``, so the fraction of prior variance
    removed is ``tau / (1 + tau)`` with ``tau = m_obs attenuation_l^2 s_l^2 /
    sigma^2``. It is a prediction to be compared with the exact diagonal, not
    a substitute for it [EQ-INV-APP-GRAV-009].

    Args:
        design: The design.
        posterior: The posterior of the same design.
        threshold: Resolved fraction at or above which a degree is called
            resolved, in ``(0, 1)``.

    Returns:
        One :class:`DegreeResolution` per degree.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If sizes disagree or the threshold is out of range.

    References:
        [EQ-INV-APP-GRAV-009].

    """
    if not isinstance(design, GravityDesign) or not isinstance(
        posterior, GravityPosterior
    ):
        raise TypeError("design and posterior must be GravityDesign/GravityPosterior")
    level = _finite(threshold, "threshold")
    if not 0.0 < level < 1.0:
        raise ValueError("threshold must lie in (0, 1)")
    if len(posterior.sd) != design.unknowns:
        raise ValueError("posterior and design disagree")
    mean_radius = math.fsum(design.radii) / len(design.radii)
    rows = []
    for ell in range(design.min_degree, design.max_degree + 1):
        index = [j for j, (degree, _, _) in enumerate(design.labels) if degree == ell]
        fraction = math.fsum(posterior.resolution_diagonal[j] for j in index) / len(
            index
        )
        rms = math.sqrt(math.fsum(posterior.sd[j] ** 2 for j in index) / len(index))
        s_l = posterior.prior_sd[index[0]]
        att = attenuation(ell, mean_radius)
        tau = design.observations * (att * s_l / posterior.sigma) ** 2
        rows.append(
            DegreeResolution(
                ell, att, s_l, rms, fraction, tau / (1.0 + tau), fraction >= level
            )
        )
    return tuple(rows)


def resolved_degree(resolution: Sequence[DegreeResolution]) -> int:
    """Return the largest ``l`` with every degree up to ``l`` resolved.

    Args:
        resolution: Output of :func:`degree_resolution` (ascending degrees).

    Returns:
        The contiguous resolved degree, or ``min_degree - 1`` (the degree
        before the first one) when the first degree is not resolved.

    Raises:
        ValueError: If ``resolution`` is empty.

    References:
        [EQ-INV-APP-GRAV-009].

    """
    rows = tuple(resolution)
    if not rows:
        raise ValueError("resolution must be non-empty")
    best = rows[0].degree - 1
    for row in rows:
        if not row.resolved:
            break
        best = row.degree
    return best


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GravitySphereBenchmark:
    """SYNTHETIC spherical-harmonic gravity benchmark (truth withheld).

    Attributes:
        design: The solver design (degrees ``2..solver_degree``).
        data: Noisy radial-gravity observations.
        sigma: Noise standard deviation (``GM / R^2``).
        realized_noise_norm: ``||e||`` of the drawn noise.
        prior_sd: Kaula prior standard deviations of the solver unknowns.
        kappa: Kaula scale.
        exponent: Kaula exponent.
        truth_kind: ``"kaula"`` or ``"mascon"``.
        truth_coefficients: Truth coefficients on the SOLVER's coefficient
            list (the part the solver can represent); scoring only.
        solver_degree: Solver maximum degree.
        truth_degree: Truth maximum degree (``> solver_degree`` unless the
            benchmark is the flagged algebraic one).
        omission_rms: Root-mean-square over the observations of the signal
            from degrees above ``solver_degree`` (what the solver cannot
            represent).
        same_discretization: ``True`` only when ``truth_degree ==
            solver_degree`` (inverse crime; algebraic verification).
        seed: Root seed.
        notice: :data:`SYNTHETIC_NOTICE`.

    References:
        [EQ-INV-APP-GRAV-010].

    """

    design: GravityDesign
    data: Vector
    sigma: float
    realized_noise_norm: float
    prior_sd: Vector
    kappa: float
    exponent: float
    truth_kind: str
    truth_coefficients: Vector
    solver_degree: int
    truth_degree: int
    omission_rms: float
    same_discretization: bool
    seed: int
    notice: str


def gravity_sphere_benchmark(
    *,
    truth: str = "kaula",
    solver_degree: int = 6,
    truth_degree: int = 12,
    count: int = 150,
    altitude: float = 0.1,
    sigma: float = 1e-5,
    kappa: float = 1e-5,
    exponent: float = 2.0,
    cap_colatitude: float = math.pi,
    seed: int = 7,
) -> GravitySphereBenchmark:
    """Return a SYNTHETIC gravity benchmark with a higher-degree truth.

    ``truth="kaula"`` draws coefficients of degrees ``2..truth_degree`` from the
    Kaula prior. ``truth="mascon"`` places three interior point masses of
    declared positions and uses the CLOSED-FORM attraction (no harmonic series)
    minus the exact degree 0 and 1 parts, so the data do not come from the
    solver's representation. The solver estimates the degrees
    ``2..solver_degree``; the remainder is an omission error that is reported,
    not hidden [EQ-INV-APP-GRAV-010]. ``truth_degree == solver_degree`` is the
    inverse crime and is flagged (``same_discretization = True``).

    Args:
        truth: ``"kaula"`` or ``"mascon"``.
        solver_degree: Solver degree (``>= 2``).
        truth_degree: Truth degree (``>= solver_degree``; for ``"mascon"`` the
            degrees above the solver's are implicit in the closed form and
            ``truth_degree`` only sets the reported flag).
        count: Number of observations.
        altitude: ``h / R`` (``>= 0``); observation radius ``1 + altitude``.
        sigma: Noise standard deviation (``> 0``, units ``GM / R^2``).
        kappa: Kaula scale.
        exponent: Kaula exponent.
        cap_colatitude: Coverage cap of the design (``pi`` = global).
        seed: Root seed.

    Returns:
        :class:`GravitySphereBenchmark`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: On an unknown truth or invalid sizes.

    References:
        [EQ-INV-APP-GRAV-010], [REF-KAULA-1966], [REF-KAIPIO-SOMERSALO-2007].

    """
    if truth not in _TRUTHS:
        raise ValueError(f"truth must be one of {_TRUTHS}")
    ls = _degree(solver_degree, "solver_degree", 2)
    lt = _degree(truth_degree, "truth_degree", ls)
    h = _finite(altitude, "altitude")
    if h < 0.0:
        raise ValueError("altitude must be non-negative")
    level = _positive(sigma, "sigma")
    thetas, phis = sphere_design(count, cap_colatitude=cap_colatitude)
    radius = 1.0 + h
    design = gravity_design(thetas, phis, radius, ls)
    prior_sd = kaula_prior_sd(design.labels, kappa=kappa, exponent=exponent)
    stream = resolve_stream(seed).child("gravity-sphere", truth)
    omission_rms: float
    if truth == "kaula":
        full = gravity_design(thetas, phis, radius, lt)
        full_sd = kaula_prior_sd(full.labels, kappa=kappa, exponent=exponent)
        generator = stream.child("coefficients").generator()
        coefficients = tuple(generator.gauss(0.0, s) for s in full_sd)
        exact = tuple(
            math.fsum(a * c for a, c in zip(row, coefficients, strict=True))
            for row in full.matrix
        )
        index = {label: k for k, label in enumerate(full.labels)}
        kept = tuple(coefficients[index[label]] for label in design.labels)
        low = tuple(
            math.fsum(a * c for a, c in zip(row, kept, strict=True))
            for row in design.matrix
        )
        residual = [e - v for e, v in zip(exact, low, strict=True)]
        omission_rms = math.sqrt(math.fsum(v * v for v in residual) / len(residual))
    else:
        sources = (
            (0.9, 0.4, 0.8, 4e-6),
            (2.1, 3.5, 0.7, -3e-6),
            (1.3, 5.0, 0.85, 2e-6),
        )
        exact_full = tuple(
            math.fsum(point_mass_radial_gravity(th, ph, radius, src) for src in sources)
            for th, ph in zip(thetas, phis, strict=True)
        )
        low_design = gravity_design(thetas, phis, radius, 1, min_degree=0)
        low_coefficients = [0.0] * low_design.unknowns
        for src in sources:
            part = point_mass_coefficients(
                src[0], src[1], src[2], src[3], 1, min_degree=0
            )
            low_coefficients = [
                a + b for a, b in zip(low_coefficients, part, strict=True)
            ]
        low_signal = tuple(
            math.fsum(a * c for a, c in zip(row, low_coefficients, strict=True))
            for row in low_design.matrix
        )
        exact = tuple(e - v for e, v in zip(exact_full, low_signal, strict=True))
        parts = [
            point_mass_coefficients(src[0], src[1], src[2], src[3], ls, min_degree=2)
            for src in sources
        ]
        coefficients = tuple(math.fsum(col) for col in zip(*parts, strict=True))
        kept = coefficients
        represented = tuple(
            math.fsum(a * c for a, c in zip(row, kept, strict=True))
            for row in design.matrix
        )
        residual = [e - v for e, v in zip(exact, represented, strict=True)]
        omission_rms = math.sqrt(math.fsum(v * v for v in residual) / len(residual))
    noise_generator = stream.child("noise").generator()
    draws = [noise_generator.gauss(0.0, 1.0) for _ in exact]
    data = tuple(e + level * d for e, d in zip(exact, draws, strict=True))
    return GravitySphereBenchmark(
        design,
        data,
        level,
        level * math.sqrt(math.fsum(d * d for d in draws)),
        prior_sd,
        _positive(kappa, "kappa"),
        exponent,
        truth,
        kept,
        ls,
        lt,
        omission_rms,
        lt == ls and truth == "kaula",
        seed,
        SYNTHETIC_NOTICE,
    )


@dataclass(frozen=True, slots=True)
class ResolutionSweepRow:
    """Resolved degree at one noise level.

    Attributes:
        sigma: Noise standard deviation.
        resolved_degree: Contiguous resolved degree (``min_degree - 1`` when
            none).
        resolved_fractions: Resolved fraction per degree.

    References:
        [EQ-INV-APP-GRAV-009].

    """

    sigma: float
    resolved_degree: int
    resolved_fractions: tuple[float, ...]


def resolution_sweep(
    design: GravityDesign,
    prior_sd: Sequence[float],
    sigmas: Sequence[float],
    *,
    threshold: float = 0.5,
) -> tuple[ResolutionSweepRow, ...]:
    """Return the resolved degree for several noise levels.

    Only covariances are needed (no data), so the sweep is exact and
    deterministic [EQ-INV-APP-GRAV-009].

    Args:
        design: The design.
        prior_sd: Prior standard deviations.
        sigmas: Positive noise levels.
        threshold: Resolved-fraction threshold.

    Returns:
        One row per noise level.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a noise level is not positive.

    References:
        [EQ-INV-APP-GRAV-009].

    """
    rows = []
    for level in sigmas:
        posterior = gravity_posterior(design, level, prior_sd)
        table = degree_resolution(design, posterior, threshold=threshold)
        rows.append(
            ResolutionSweepRow(
                float(level),
                resolved_degree(table),
                tuple(r.resolved_fraction for r in table),
            )
        )
    return tuple(rows)
