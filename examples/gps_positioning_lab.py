"""GPS positioning laboratory: the law of a receiver position from five ranges.

Five satellites ``S_1, ..., S_5`` broadcast signals; the receiver reads the
pseudo-ranges ``D_k = c tau + ||S_k - R|| + e_k`` and must deduce its own
position ``R``. The laboratory has two parts.

Part 1 follows a classical uncertainty-quantification teaching example.
Three sources of uncertainty enter every reading: a small clock shift
``tau`` common to all satellites, the satellite positions (known only to lie
in small spheres around their broadcast centres), and the range reading
itself. The laboratory computes the density of ``R`` given one set of
readings,

    f_R(r) proportional to E_{tau, S}[ prod_k f_e(D_k - c tau - ||r - S_k||) ],

its mean ``E[R]`` and the radius of the sphere around ``E[R]`` that holds 90%
of the probability, by self-normalized importance sampling with the inner
expectation evaluated in closed form (satellite offsets) and by Gauss-Legendre
quadrature (clock shift). It then propagates the same three sources through
a nonlinear least-squares fix (one solve per draw), splits the position
error by source, checks the linearization through the geometry matrix, and
shows how the satellite geometry scales the error.

Part 2 is the standard single-point receiver: the clock offset is large and
unknown, so it is estimated as a fourth unknown next to the position. The
laboratory recovers position and clock from exact readings, computes the
dilution-of-precision family (GDOP, PDOP, HDOP, VDOP, TDOP), compares the
ordinary and the weighted least-squares fix under elevation-dependent
errors, and uses the one redundant reading for a residual-based fault
detection test (receiver autonomous integrity monitoring, RAIM).

Definitions:
    local frame: East, North, Up coordinates in metres with the true receiver
        ``R0`` at the origin on a spherical Earth of radius 6371 km; every
        satellite sits on a 26 560 km orbit radius, 20 000 to 25 000 km
        away, at a stated azimuth and elevation.
    Part 1 sources: clock shift ``tau ~ Uniform(-1e-9, 1e-9)`` s (so
        ``c tau`` is within ``b = 0.30`` m); satellite position uniform in a
        ball of radius ``a = 2`` m around its broadcast centre; range error
        ``e_k ~ Uniform(-h, h)`` with ``h = 10`` m, independent across
        satellites.
    geometry matrix: rows ``g_k = (R0 - S_k) / ||R0 - S_k||`` (unit lines of
        sight, clock known) or ``(g_k, 1)`` (clock unknown);
        ``A = (G'G)^-1 G'`` maps range errors to fix errors to first order.
    dilution of precision: with ``H = (G'G)^-1`` for the four-column ``G``,
        ``GDOP = sqrt(trace H)``, ``PDOP = sqrt(H_EE + H_NN + H_UU)``,
        ``HDOP = sqrt(H_EE + H_NN)``, ``VDOP = sqrt(H_UU)``,
        ``TDOP = sqrt(H_bb)``; the known-clock PDOP uses the three-column
        ``G`` of Part 1.
    90% radius: the radius ``rho`` of the sphere around ``E[R]`` (posterior)
        or around ``R0`` (least-squares fix) holding probability 0.9.
    Part 2 errors: ``e_k ~ Normal(0, sigma_k^2)`` with
        ``sigma_k = sigma_0 / sin(elevation_k)``, ``sigma_0 = 2`` m; receiver
        clock offset 0.25 ms (75 km of range), unknown to the receiver.
    RAIM statistic: ``T = sum_k (r_k / sigma_k)^2`` from the weighted fix
        residuals; an alarm is raised when ``T`` exceeds the
        ``1 - P_FA`` quantile of ``chi^2(n - 4)``, ``P_FA = 0.01``.
    minimal detectable bias: the range fault on one satellite that the test
        detects with probability 0.9.

Assumptions:
    - Part 1: the three sources are independent; the clock shift is the same
      for all five readings; the receiver knows the broadcast centres, not
      the true satellite positions. The least-squares fix solves for the
      three position coordinates only and treats ``c tau`` (at most 0.3 m
      against 10 m range errors) as zero; the posterior integrates it out.
    - The posterior uses a flat prior on ``R``. Its likelihood replaces
      ``||r - S_k||`` by ``||r - C_k|| + u_k'(S_k - C_k)`` (line-of-sight
      component of the offset), an error below ``a^2 / ||r - C_k||``, about
      2e-7 m.
    - Part 2: errors are Gaussian, independent, and of known variance; the
      fault test assumes a single faulty satellite. With five satellites and
      four unknowns there is one redundant reading: a fault can be detected
      but not attributed to a satellite (no exclusion).
    - Signal travel time, Earth rotation, and atmospheric delays are left out;
      ``P_FA = 0.01`` is a teaching value, far above aviation integrity
      requirements.

Equations:
    EQ-LAB-GPS-001: pseudorange likelihood with ball-uniform satellite offset,
        uniform reading error, and uniform clock shift.
    EQ-LAB-GPS-002: first-order covariance of a pseudorange fix and the
        dilution of precision.
    EQ-LAB-GPS-003: residual fault-detection test and its detection
        probability.
    EQ-OPT-LM-001: Levenberg-Marquardt solve of each least-squares fix.
    EQ-OPT-LM-002: Gauss-Newton covariance ``s^2 (J'J)^-1`` of the fix.
    EQ-PROC-QUAD-001: Gauss-Legendre rule over the clock shift.
    EQ-FILT-PF-002: effective sample size of the importance weights.
    EQ-STAT-QUANT-001: sample quantiles of the radial error.
    EQ-DIST-CHI2-001: chi-square threshold of the fault test.
    EQ-OPT-BRENT-001: root search for the minimal detectable bias.

Diagnostics:
    - Exact readings recover the true position (and, in Part 2, the clock
      offset) to below a micrometre.
    - The linearized covariance agrees with the Monte Carlo covariance of the
      fix within its standard error; the three single-source runs add up to
      the all-source run; the library Gauss-Newton covariance equals the
      geometry-matrix formula.
    - Importance sampling reports its effective sample size, the share of
      proposals with positive weight, whether any weighted proposal touches
      the edge of the proposal box (none may), and the largest relative
      change of the likelihood when the clock quadrature goes from 8 to 16
      nodes per panel (exact rules: rounding only).
    - Each 90% radius and posterior mean carries a batch standard error from
      ``BATCHES`` independent batches of proposals or draws.
    - The fault test's simulated false-alarm rate and detection rates are
      compared with ``P_FA`` and with the closed-form detection probability.

Cost and error:
    - About 1 to 1.5 minutes in pure Python; the run prints its wall-clock
      time.
      Most of it goes to the 27 500 least-squares fixes (21 000 in Part 1;
      4 500 weighted and 2 000 ordinary in Part 2) at about 2 ms each; the
      120 000 importance-sampling proposals take about 3 seconds.
    - The posterior likelihood carries no Monte Carlo error of its own (the
      inner expectation is exact up to the quadrature check above); only the
      importance sampling over ``R`` is random.
    - Monte Carlo standard errors are printed next to every simulated
      number; differences smaller than two standard errors are not findings.

References:
    [REF-SMITH-UQ-2024], [REF-MISRA-ENGE-2006],
    [REF-TEUNISSEN-MONTENBRUCK-2017], [REF-BROWN-1992], [REF-OWEN-2013],
    [REF-NOCEDAL-WRIGHT-2006], [REF-SEBER-WILD-2003],
    [REF-DOUCET-JOHANSEN-2011], [REF-HYNDMAN-FAN-1996],
    [REF-GOLUB-WELSCH-1969], [REF-MATPLOTLIB].

"""

from __future__ import annotations

import functools
import itertools
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from probstat.distributions import ChiSquare
from probstat.filtering import effective_sample_size
from probstat.inference import standard_normal_cdf
from probstat.linalg import (
    Matrix,
    cholesky,
    inverse_spd,
    lu_solve,
    matmul,
    symmetric_eigen,
    transpose,
)
from probstat.optimization import (
    LeastSquaresResult,
    brent_root,
    gauss_newton_covariance,
    least_squares,
)
from probstat.processes import gauss_legendre
from probstat.statistics import covariance, quantile
from probstat.visualization import save_figure, theme

# Seed of every random stream in the laboratory.
SEED = 2031

SPEED_OF_LIGHT = 299_792_458.0  # m/s
CLOCK_HALF_WIDTH = 1e-9  # s
CLOCK_RANGE = SPEED_OF_LIGHT * CLOCK_HALF_WIDTH  # m, half-width b of c tau
SATELLITE_RADIUS = 2.0  # m
RANGE_HALF_WIDTH = 10.0  # m
EARTH_RADIUS = 6_371_000.0  # m
ORBIT_RADIUS = 26_560_000.0  # m
LEVEL = 0.9
ORIGIN: tuple[float, float, float] = (0.0, 0.0, 0.0)

# Azimuth and elevation in degrees: a well-spread sky and a clustered one.
GOOD_SKY = ((20.0, 78.0), (75.0, 28.0), (160.0, 42.0), (245.0, 22.0), (320.0, 50.0))
POOR_SKY = ((60.0, 35.0), (85.0, 48.0), (100.0, 30.0), (120.0, 55.0), (140.0, 38.0))

FIX_DRAWS = 3000
GEOMETRY_DRAWS = 500
RANDOM_GEOMETRIES = 10
PROPOSALS = 120_000
BOX_MARGIN = 1.05
CLOCK_NODES = 8
CHECK_NODES = 16
BATCHES = 10

# Part 2: the standard receiver with an unknown clock offset.
RECEIVER_CLOCK_OFFSET = 2.5e-4  # s
CLOCK_BIAS = SPEED_OF_LIGHT * RECEIVER_CLOCK_OFFSET  # m
ZENITH_SIGMA = 2.0  # m
CLOCK_FIX_DRAWS = 2000
FALSE_ALARM = 0.01
DETECTION_LEVEL = 0.9
DETECTION_DRAWS = 250

Vector3 = tuple[float, float, float]
SOURCES = ("clock", "satellite", "range")


# --------------------------------------------------------------------------
# Geometry and sampling
# --------------------------------------------------------------------------


def satellite_centre(azimuth: float, elevation: float) -> Vector3:
    """Return the East-North-Up position of a satellite seen at a direction.

    The satellite lies on the orbit sphere of radius ``ORBIT_RADIUS`` around
    the Earth centre ``(0, 0, -EARTH_RADIUS)``; its range ``rho`` along the
    unit direction ``u`` solves
    ``||rho u + (0, 0, EARTH_RADIUS)|| = ORBIT_RADIUS``.

    Args:
        azimuth: Degrees clockwise from North.
        elevation: Degrees above the horizon.

    Returns:
        Satellite centre in metres.

    References:
        [REF-MISRA-ENGE-2006, Chapter 4]. No equation is evaluated.

    """
    az, el = math.radians(azimuth), math.radians(elevation)
    u = (math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el))
    b = EARTH_RADIUS * math.sin(el)
    rho = -b + math.sqrt(b * b + ORBIT_RADIUS**2 - EARTH_RADIUS**2)
    return (rho * u[0], rho * u[1], rho * u[2])


def constellation(sky: Sequence[tuple[float, float]]) -> tuple[Vector3, ...]:
    """Return the satellite centres of a sky plan.

    Args:
        sky: Azimuth and elevation pairs in degrees.

    Returns:
        Satellite centres in metres.

    References:
        [REF-MISRA-ENGE-2006, Chapter 4]. No equation is evaluated.

    """
    return tuple(satellite_centre(az, el) for az, el in sky)


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the Euclidean distance between two points of space.

    References:
        [REF-STRANG-2016]. No equation is evaluated.

    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def ball_point(rng: random.Random, radius: float) -> Vector3:
    """Return a point uniform in a ball centred at the origin.

    Direction from three normal coordinates, radius ``radius U^(1/3)``.

    References:
        [REF-OWEN-2013]. No equation is evaluated.

    """
    while True:
        x, y, z = rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)
        size = math.sqrt(x * x + y * y + z * z)
        if size > 0.0:
            break
    scale = radius * rng.random() ** (1.0 / 3.0) / size
    return (x * scale, y * scale, z * scale)


@dataclass(frozen=True)
class SourceDraw:
    """One draw of the three uncertainty sources of Part 1.

    Attributes:
        clock_range: ``c tau`` in metres.
        offsets: Displacement of each satellite from its centre.
        errors: Range-reading error of each satellite.

    """

    clock_range: float
    offsets: tuple[Vector3, ...]
    errors: tuple[float, ...]


def draw_sources(
    rng: random.Random, count: int, active: Sequence[str] = SOURCES
) -> SourceDraw:
    """Draw the clock shift, satellite offsets, and range errors.

    Every source is drawn whether active or not, so switching a source off
    leaves the random stream of the others unchanged.

    Args:
        rng: Random stream.
        count: Number of satellites.
        active: Sources switched on; the others are set to zero.

    Returns:
        One draw of the sources.

    References:
        [REF-SMITH-UQ-2024]. No equation is evaluated.

    """
    clock = SPEED_OF_LIGHT * rng.uniform(-CLOCK_HALF_WIDTH, CLOCK_HALF_WIDTH)
    offsets = tuple(ball_point(rng, SATELLITE_RADIUS) for _ in range(count))
    errors = tuple(
        rng.uniform(-RANGE_HALF_WIDTH, RANGE_HALF_WIDTH) for _ in range(count)
    )
    return SourceDraw(
        clock if "clock" in active else 0.0,
        offsets if "satellite" in active else tuple((0.0, 0.0, 0.0) for _ in offsets),
        errors if "range" in active else tuple(0.0 for _ in errors),
    )


def readings(
    centres: Sequence[Vector3], receiver: Vector3, sources: SourceDraw
) -> tuple[float, ...]:
    """Return the pseudo-ranges ``D_k = c tau + ||S_k - R|| + e_k``.

    References:
        [REF-SMITH-UQ-2024], [REF-MISRA-ENGE-2006, Chapter 5].
        No equation is evaluated.

    """
    return tuple(
        sources.clock_range
        + distance(tuple(c + o for c, o in zip(centre, offset, strict=True)), receiver)
        + error
        for centre, offset, error in zip(
            centres, sources.offsets, sources.errors, strict=True
        )
    )


# --------------------------------------------------------------------------
# Least-squares fix with a known clock and its linearization
# --------------------------------------------------------------------------


def solve_fix(
    centres: Sequence[Vector3],
    observed: Sequence[float],
    start: Sequence[float] = ORIGIN,
) -> LeastSquaresResult:
    """Return the three-unknown least-squares position fix from pseudo-ranges.

    Minimizes ``sum_k (||r - S_k|| - D_k)^2`` over ``r`` with the broadcast
    centres ``S_k`` and the analytic Jacobian (rows: unit vectors from the
    satellites to ``r``).

    Args:
        centres: Broadcast satellite centres.
        observed: Pseudo-ranges in metres.
        start: Starting position.

    Returns:
        The :class:`probstat.optimization.LeastSquaresResult`.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 10.3].

    """

    def residual(r: Sequence[float]) -> tuple[float, ...]:
        """Return range residuals at a candidate position."""
        return tuple(
            distance(r, centre) - d for centre, d in zip(centres, observed, strict=True)
        )

    def jacobian(r: Sequence[float]) -> tuple[tuple[float, ...], ...]:
        """Return the unit lines of sight at a candidate position."""
        rows = []
        for centre in centres:
            size = distance(r, centre)
            rows.append(tuple((r[i] - centre[i]) / size for i in range(3)))
        return tuple(rows)

    return least_squares(residual, start, jacobian=jacobian, tolerance=1e-12)


def geometry_matrix(centres: Sequence[Vector3], receiver: Vector3) -> Matrix:
    """Return the unit lines of sight ``g_k = (R - S_k) / ||R - S_k||``.

    References:
        [REF-MISRA-ENGE-2006, Chapter 6]. No equation is evaluated.

    """
    rows = []
    for centre in centres:
        size = distance(receiver, centre)
        rows.append(tuple((receiver[i] - centre[i]) / size for i in range(3)))
    return tuple(rows)


def error_map(g: Matrix) -> Matrix:
    """Return ``A = (G'G)^-1 G'``, range errors to fix errors (``p x k``).

    References:
        [EQ-LAB-GPS-002], [REF-SEBER-WILD-2003, Section 2.1].

    """
    gt = transpose(g)
    return matmul(inverse_spd(matmul(gt, g)), gt)


def pdop_known_clock(centres: Sequence[Vector3], receiver: Vector3 = ORIGIN) -> float:
    """Return the known-clock position dilution ``sqrt(trace((G'G)^-1))``.

    Uses the three-column geometry matrix of Part 1. It is smaller than the
    standard PDOP of :func:`dilution_of_precision`, which also estimates the
    clock.

    References:
        [EQ-LAB-GPS-002], [REF-MISRA-ENGE-2006, Chapter 6].

    """
    g = geometry_matrix(centres, receiver)
    inverse = inverse_spd(matmul(transpose(g), g))
    return math.sqrt(inverse[0][0] + inverse[1][1] + inverse[2][2])


def source_variances() -> dict[str, float]:
    """Return the per-reading variance each source contributes, in m^2.

    ``b^2 / 3`` for the clock (shared by all readings), ``a^2 / 5`` for the
    line-of-sight component of a point uniform in a ball of radius ``a``,
    and ``h^2 / 3`` for a range error uniform on ``[-h, h]``.

    References:
        [REF-SMITH-UQ-2024]. No equation is evaluated.

    """
    return {
        "clock": CLOCK_RANGE**2 / 3.0,
        "satellite": SATELLITE_RADIUS**2 / 5.0,
        "range": RANGE_HALF_WIDTH**2 / 3.0,
    }


def linearized_covariance(
    centres: Sequence[Vector3],
    receiver: Vector3 = ORIGIN,
    active: Sequence[str] = SOURCES,
) -> list[list[float]]:
    """Return the first-order covariance ``A Sigma_e A'`` of the position fix.

    ``Sigma_e = (s_sat^2 + s_range^2) I + s_clock^2 1 1'``: the satellite and
    range terms are independent across readings, the clock term is common.
    The independent part equals the Gauss-Newton covariance
    ``s^2 (G'G)^-1`` of :func:`probstat.optimization.gauss_newton_covariance`.

    Args:
        centres: Broadcast satellite centres.
        receiver: Linearization point.
        active: Sources included.

    Returns:
        3 x 3 covariance in m^2.

    References:
        [EQ-LAB-GPS-002], [REF-SEBER-WILD-2003, Section 2.1].

    """
    variances = source_variances()
    independent = sum(
        variances[name] for name in ("satellite", "range") if name in active
    )
    common = variances["clock"] if "clock" in active else 0.0
    a = error_map(geometry_matrix(centres, receiver))
    k = len(centres)
    row_sums = [sum(a[i]) for i in range(3)]
    return [
        [
            independent * sum(a[i][n] * a[j][n] for n in range(k))
            + common * row_sums[i] * row_sums[j]
            for j in range(3)
        ]
        for i in range(3)
    ]


# --------------------------------------------------------------------------
# Monte Carlo propagation through the least-squares fix
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FixStudy:
    """Monte Carlo errors of the least-squares fix for one scenario.

    Attributes:
        errors: Position error ``R_hat - R0`` of every draw.
        radius: 90% quantile of ``||R_hat - R0||``.
        radius_se: Batch standard error of ``radius``.
        mean_square: Mean of ``||R_hat - R0||^2`` (trace of the second moment).
        mean_square_se: Its standard error.

    """

    errors: tuple[Vector3, ...]
    radius: float
    radius_se: float
    mean_square: float
    mean_square_se: float


def _batch_se(values: Sequence[float]) -> float:
    """Return the standard error of the mean of batch estimates."""
    m = len(values)
    centre = math.fsum(values) / m
    return math.sqrt(math.fsum((v - centre) ** 2 for v in values) / (m - 1) / m)


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    """Return the sample mean and its standard error."""
    n = len(values)
    centre = math.fsum(values) / n
    spread = math.fsum((v - centre) ** 2 for v in values) / (n - 1)
    return centre, math.sqrt(spread / n)


def study_fix(
    centres: Sequence[Vector3],
    draws: int,
    seed: int,
    active: Sequence[str] = SOURCES,
) -> FixStudy:
    """Propagate the sources through the least-squares fix, one solve per draw.

    Args:
        centres: Broadcast satellite centres.
        draws: Number of draws.
        seed: Seed of the source draws.
        active: Sources switched on.

    Returns:
        The scenario's errors, 90% radius, and mean squared error.

    References:
        [REF-OWEN-2013, Chapter 2], [REF-HYNDMAN-FAN-1996].

    """
    rng = random.Random(seed)
    errors = []
    for _ in range(draws):
        observed = readings(centres, ORIGIN, draw_sources(rng, len(centres), active))
        fix = solve_fix(centres, observed).x
        errors.append((fix[0], fix[1], fix[2]))
    norms = [math.sqrt(e[0] ** 2 + e[1] ** 2 + e[2] ** 2) for e in errors]
    size = draws // BATCHES
    batches = [norms[b * size : (b + 1) * size] for b in range(BATCHES)]
    mean_square, mean_square_se = _mean_se([n * n for n in norms])
    return FixStudy(
        tuple(errors),
        quantile(norms, LEVEL),
        _batch_se([quantile(batch, LEVEL) for batch in batches]),
        mean_square,
        mean_square_se,
    )


def sample_covariance(points: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return the unbiased sample covariance matrix of vector draws.

    References:
        [REF-OWEN-2013, Chapter 2]. No equation is evaluated here; the
        entries come from :func:`probstat.statistics.covariance`.

    """
    size = len(points[0])
    columns = [[p[i] for p in points] for i in range(size)]
    return [
        [covariance(columns[i], columns[j], ddof=1) for j in range(size)]
        for i in range(size)
    ]


def _covariance_se(points: Sequence[Vector3]) -> list[list[float]]:
    """Return the standard error of each sample-covariance entry."""
    n = len(points)
    means = [math.fsum(p[i] for p in points) / n for i in range(3)]
    return [
        [
            _mean_se([(p[i] - means[i]) * (p[j] - means[j]) for p in points])[1]
            for j in range(3)
        ]
        for i in range(3)
    ]


def _linear_radius(covariance_matrix: Sequence[Sequence[float]], seed: int) -> float:
    """Return the 90% radius of a centred normal law with this covariance.

    Estimated from 20 000 normal draws (standard error near 0.4% of the
    radius).
    """
    factor = cholesky(covariance_matrix)
    rng = random.Random(seed)
    norms = []
    for _ in range(20000):
        z = (rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0))
        x = [sum(factor[i][j] * z[j] for j in range(i + 1)) for i in range(3)]
        norms.append(math.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2))
    return quantile(norms, LEVEL)


# --------------------------------------------------------------------------
# Posterior density of R by importance sampling
# --------------------------------------------------------------------------


def offset_cdf(x: float) -> float:
    """Return the CDF of one coordinate of a point uniform in the satellite ball.

    ``G(x) = 1/2 + 3x/(4a) - x^3/(4a^3)`` on ``[-a, a]``: the projection of a
    uniform ball of radius ``a`` has density ``3 (a^2 - x^2) / (4 a^3)``.

    References:
        [EQ-LAB-GPS-001].

    """
    a = SATELLITE_RADIUS
    if x <= -a:
        return 0.0
    if x >= a:
        return 1.0
    t = x / a
    return 0.5 + 0.75 * t - 0.25 * t * t * t


def residual_density(z: float) -> float:
    """Return the density of ``X + e`` (line-of-sight offset plus range error).

    ``f_z(z) = (G(z + h) - G(z - h)) / (2h)``; it equals ``1 / (2h)`` for
    ``|z| <= h - a`` and vanishes for ``|z| >= h + a``.

    References:
        [EQ-LAB-GPS-001].

    """
    h = RANGE_HALF_WIDTH
    return (offset_cdf(z + h) - offset_cdf(z - h)) / (2.0 * h)


# Largest |D_k - ||r - C_k||| with positive likelihood.
REACH = RANGE_HALF_WIDTH + SATELLITE_RADIUS + CLOCK_RANGE


_KINKS = (
    -RANGE_HALF_WIDTH - SATELLITE_RADIUS,
    -RANGE_HALF_WIDTH + SATELLITE_RADIUS,
    RANGE_HALF_WIDTH - SATELLITE_RADIUS,
    RANGE_HALF_WIDTH + SATELLITE_RADIUS,
)


@functools.cache
def _unit_rule(nodes: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the Gauss-Legendre rule on ``[-1, 1]`` (computed once per size)."""
    points, weights = gauss_legendre(nodes)
    return tuple(points), tuple(weights)


def marginal_likelihood(
    r: Sequence[float],
    centres: Sequence[Vector3],
    observed: Sequence[float],
    nodes: int = CLOCK_NODES,
) -> float:
    """Return ``E_{tau, S}[prod_k f_e(D_k - c tau - ||r - S_k||)]``.

    The satellite offsets are integrated in closed form
    (:func:`residual_density`), the clock shift ``beta = c tau`` over
    ``[-b, b]``: ``L(r) = (1/2b) int prod_k f_z(D_k - beta - d_k) dbeta``.
    Each ``f_z`` is a polynomial of degree at most 3 between its kinks
    ``z = +-h +- a``, so the integrand is a polynomial of degree at most
    ``3k = 15`` between the kinks that fall inside ``[-b, b]``; a
    Gauss-Legendre rule with ``nodes >= 8`` on every such panel is exact up
    to rounding.

    Args:
        r: Candidate receiver position.
        centres: Broadcast satellite centres.
        observed: Pseudo-ranges.
        nodes: Gauss-Legendre nodes per panel.

    Returns:
        Likelihood of ``r`` in m^-k.

    References:
        [EQ-LAB-GPS-001], [REF-SMITH-UQ-2024], [REF-GOLUB-WELSCH-1969].

    """
    gaps = []
    for centre, d in zip(centres, observed, strict=True):
        gap = d - distance(r, centre)
        if abs(gap) >= REACH:
            return 0.0
        gaps.append(gap)
    breaks = sorted(
        {-CLOCK_RANGE, CLOCK_RANGE}
        | {
            gap - kink
            for gap in gaps
            for kink in _KINKS
            if -CLOCK_RANGE < gap - kink < CLOCK_RANGE
        }
    )
    unit_nodes, unit_weights = _unit_rule(nodes)
    total = 0.0
    for low, high in zip(breaks[:-1], breaks[1:], strict=True):
        middle, half = 0.5 * (low + high), 0.5 * (high - low)
        for t, weight in zip(unit_nodes, unit_weights, strict=True):
            beta = middle + half * t
            product = half * weight
            for gap in gaps:
                product *= residual_density(gap - beta)
                if product == 0.0:
                    break
            total += product
    return total / (2.0 * CLOCK_RANGE)


@dataclass(frozen=True)
class PosteriorStudy:
    """Self-normalized importance-sampling summary of ``f_R``.

    Attributes:
        observed: The pseudo-ranges conditioned on.
        fix: Least-squares fix from the same readings.
        points: Proposals with positive weight.
        weights: Their normalized weights.
        mean: Posterior mean ``E[R]``.
        mean_se: Batch standard error of each coordinate of ``mean``.
        radius: 90% radius around ``E[R]``.
        radius_se: Batch standard error of ``radius``.
        ess: Effective sample size of the weights.
        proposals: Number of proposals.
        positive_share: Share of proposals with positive weight.
        edge_hits: Weighted proposals within 2% of the box edge.
        quadrature_gap: Largest relative change of the likelihood from
            ``CLOCK_NODES`` to ``CHECK_NODES`` clock nodes.
        half_widths: Half-widths of the (rotated) proposal box.

    """

    observed: tuple[float, ...]
    fix: Vector3
    points: tuple[Vector3, ...]
    weights: tuple[float, ...]
    mean: Vector3
    mean_se: Vector3
    radius: float
    radius_se: float
    ess: float
    proposals: int
    positive_share: float
    edge_hits: int
    quadrature_gap: float
    half_widths: Vector3


def _weighted_quantile(
    values: Sequence[float], weights: Sequence[float], level: float
) -> float:
    """Return the smallest value whose weighted CDF reaches ``level``."""
    order = sorted(range(len(values)), key=values.__getitem__)
    total = math.fsum(weights)
    running = 0.0
    for index in order:
        running += weights[index]
        if running >= level * total:
            return values[index]
    return values[order[-1]]


def _weighted_radius(
    points: Sequence[Vector3], weights: Sequence[float]
) -> tuple[Vector3, float]:
    """Return the weighted mean and the 90% radius around it."""
    total = math.fsum(weights)
    mean = tuple(
        math.fsum(w * p[i] for w, p in zip(weights, points, strict=True)) / total
        for i in range(3)
    )
    radii = [distance(p, mean) for p in points]
    return (mean[0], mean[1], mean[2]), _weighted_quantile(radii, weights, LEVEL)


@dataclass(frozen=True)
class _BoxSample:
    """Proposals with positive likelihood from one uniform box."""

    points: list[Vector3]
    likelihoods: list[float]
    batches: list[int]
    edge_hits: int


def _sample_box(
    rng: random.Random,
    count: int,
    centre: Sequence[float],
    axes: Sequence[Sequence[float]],
    half_widths: Sequence[float],
    likelihood: Callable[[Vector3], float],
) -> _BoxSample:
    """Draw uniform proposals ``centre + sum_i w_i u_i q_i`` and keep L > 0.

    Args:
        rng: Random stream.
        count: Number of proposals.
        centre: Box centre.
        axes: Orthonormal box axes ``q_i``.
        half_widths: Half-widths ``w_i``.
        likelihood: Likelihood of a position.

    Returns:
        The kept points, their likelihoods, the batch of each proposal, and
        the number of kept points within 2% of the box edge.

    """
    kept = _BoxSample([], [], [], 0)
    hits = 0
    for index in range(count):
        u = [rng.uniform(-1.0, 1.0) for _ in range(3)]
        r = tuple(
            centre[i] + sum(half_widths[j] * u[j] * axes[j][i] for j in range(3))
            for i in range(3)
        )
        point: Vector3 = (r[0], r[1], r[2])
        value = likelihood(point)
        if value > 0.0:
            kept.points.append(point)
            kept.likelihoods.append(value)
            kept.batches.append(index * BATCHES // count)
            if max(abs(v) for v in u) > 0.98:
                hits += 1
    return _BoxSample(kept.points, kept.likelihoods, kept.batches, hits)


def support_vertices(
    centres: Sequence[Vector3], observed: Sequence[float], fix: Vector3
) -> list[Vector3]:
    """Return the vertices of the linearized support of the likelihood.

    ``L(r) > 0`` requires ``|D_k - ||r - C_k||| < h + a + b`` for every
    satellite. With ``||r - C_k|| ~ d_k + u_k'(r - fix)`` (``u_k`` the unit
    vector from ``C_k`` to the fix) this is a polytope of ``2k`` half-spaces;
    its vertices are the feasible intersections of three bounding planes.
    The linearization error is about ``||r - fix||^2 / ||C_k||``, near
    1e-5 m here.

    Args:
        centres: Broadcast satellite centres.
        observed: Pseudo-ranges.
        fix: Linearization point (the least-squares fix).

    Returns:
        The polytope vertices.

    References:
        [EQ-LAB-GPS-001], [REF-STRANG-2016].

    """
    planes = []
    for centre, d in zip(centres, observed, strict=True):
        size = distance(fix, centre)
        u = tuple((fix[i] - centre[i]) / size for i in range(3))
        gap = d - size
        planes.append((u, gap + REACH))
        planes.append((tuple(-v for v in u), REACH - gap))
    vertices: list[Vector3] = []
    for i, j, k in itertools.combinations(range(len(planes)), 3):
        rows = (planes[i][0], planes[j][0], planes[k][0])
        try:
            delta = lu_solve(rows, (planes[i][1], planes[j][1], planes[k][1]))
        except ValueError:
            continue
        if all(sum(n[m] * delta[m] for m in range(3)) <= c + 1e-9 for n, c in planes):
            vertices.append((fix[0] + delta[0], fix[1] + delta[1], fix[2] + delta[2]))
    return vertices


def study_posterior(
    centres: Sequence[Vector3],
    seed: int,
    proposals: int = PROPOSALS,
    nodes: int = CLOCK_NODES,
) -> PosteriorStudy:
    """Estimate ``f_R``, ``E[R]``, and the 90% radius from one set of readings.

    The readings come from one draw of the three sources at ``R0``. The
    likelihood vanishes outside the polytope of :func:`support_vertices`;
    proposals are uniform on the box aligned with the principal axes of its
    vertices, spanning their extent enlarged by ``BOX_MARGIN``. The
    proposal density is constant on the box, so the weight of a proposal
    is its likelihood (:func:`marginal_likelihood`).

    Args:
        centres: Broadcast satellite centres.
        seed: Seed of the readings and the proposals.
        proposals: Number of proposals.
        nodes: Gauss-Legendre nodes over the clock shift.

    Returns:
        The importance-sampling summary.

    References:
        [REF-OWEN-2013, Chapter 9], [REF-DOUCET-JOHANSEN-2011].

    """
    rng = random.Random(seed)
    observed = readings(centres, ORIGIN, draw_sources(rng, len(centres)))
    fit = solve_fix(centres, observed).x
    fix: Vector3 = (fit[0], fit[1], fit[2])

    def likelihood(r: Vector3) -> float:
        """Return the likelihood of a candidate position."""
        return marginal_likelihood(r, centres, observed, nodes)

    vertices = support_vertices(centres, observed, fix)
    centroid = [math.fsum(v[i] for v in vertices) / len(vertices) for i in range(3)]
    scatter = [
        [
            math.fsum((v[i] - centroid[i]) * (v[j] - centroid[j]) for v in vertices)
            for j in range(3)
        ]
        for i in range(3)
    ]
    _, vectors = symmetric_eigen(scatter)
    axes = [tuple(vectors[i][j] for i in range(3)) for j in range(3)]
    centre = list(centroid)
    half = []
    for axis in axes:
        projections = [
            sum((v[i] - centroid[i]) * axis[i] for i in range(3)) for v in vertices
        ]
        low, high = min(projections), max(projections)
        for i in range(3):
            centre[i] += 0.5 * (low + high) * axis[i]
        half.append(BOX_MARGIN * 0.5 * (high - low))
    sample = _sample_box(rng, proposals, centre, axes, half, likelihood)
    # Safety net: if a weighted proposal touches the box edge (the support
    # reaches beyond the linearized polytope), the box grows by half and
    # the sample is redrawn, so the estimates always come from one box.
    for _ in range(4):
        if sample.edge_hits == 0:
            break
        half = [1.5 * w for w in half]
        sample = _sample_box(rng, proposals, centre, axes, half, likelihood)
    main = sample

    total = math.fsum(main.likelihoods)
    normalized = tuple(w / total for w in main.likelihoods)
    mean, radius = _weighted_radius(main.points, normalized)
    batch_means: list[Vector3] = []
    batch_radii: list[float] = []
    for b in range(BATCHES):
        chosen = [i for i, batch in enumerate(main.batches) if batch == b]
        batch_mean, batch_radius = _weighted_radius(
            [main.points[i] for i in chosen], [normalized[i] for i in chosen]
        )
        batch_means.append(batch_mean)
        batch_radii.append(batch_radius)
    mean_se = tuple(_batch_se([m[i] for m in batch_means]) for i in range(3))
    step = max(1, len(main.points) // 400)
    gap = max(
        abs(
            marginal_likelihood(main.points[i], centres, observed, CHECK_NODES)
            / main.likelihoods[i]
            - 1.0
        )
        for i in range(0, len(main.points), step)
    )
    return PosteriorStudy(
        observed,
        fix,
        tuple(main.points),
        normalized,
        mean,
        (mean_se[0], mean_se[1], mean_se[2]),
        radius,
        _batch_se(batch_radii),
        effective_sample_size(normalized),
        proposals,
        len(main.points) / proposals,
        main.edge_hits,
        gap,
        (half[0], half[1], half[2]),
    )


# --------------------------------------------------------------------------
# Part 2: unknown receiver clock, dilution of precision, weighted fix, RAIM
# --------------------------------------------------------------------------


def elevation_sigmas(sky: Sequence[tuple[float, float]]) -> tuple[float, ...]:
    """Return the range-error standard deviations ``sigma_0 / sin(elevation)``.

    Propagation errors grow at low elevation, where the signal crosses more
    atmosphere; the ``1 / sin`` form is a modelling choice of this
    laboratory.

    References:
        [REF-MISRA-ENGE-2006, Chapter 5]. No equation is evaluated.

    """
    return tuple(ZENITH_SIGMA / math.sin(math.radians(el)) for _, el in sky)


def clock_geometry_matrix(centres: Sequence[Vector3], receiver: Vector3) -> Matrix:
    """Return the four-column geometry matrix with rows ``(g_k, 1)``.

    References:
        [EQ-LAB-GPS-002], [REF-MISRA-ENGE-2006, Chapter 6].

    """
    return tuple((*row, 1.0) for row in geometry_matrix(centres, receiver))


def solve_clock_fix(
    centres: Sequence[Vector3],
    observed: Sequence[float],
    sigmas: Sequence[float] | None = None,
    start: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
) -> LeastSquaresResult:
    """Return the four-unknown (position and clock) least-squares fix.

    Minimizes ``sum_k ((||r - S_k|| + beta - D_k) / sigma_k)^2`` over the
    position ``r`` and the clock range ``beta``; ``sigmas=None`` gives the
    ordinary (unweighted) fix. The residuals are whitened, so
    ``2 * result.cost`` is the fault-test statistic ``T`` of the weighted fix
    and ``gauss_newton_covariance(result, noise_variance=1)`` its covariance
    ``(G' W G)^-1``.

    Args:
        centres: Satellite centres.
        observed: Pseudo-ranges in metres.
        sigmas: Range-error standard deviations, or ``None``.
        start: Starting ``(E, N, U, beta)``.

    Returns:
        The :class:`probstat.optimization.LeastSquaresResult`.

    References:
        [REF-MISRA-ENGE-2006, Chapter 6], [REF-NOCEDAL-WRIGHT-2006,
        Section 10.3].

    """
    scales = [1.0] * len(centres) if sigmas is None else list(sigmas)

    def residual(x: Sequence[float]) -> tuple[float, ...]:
        """Return whitened pseudorange residuals."""
        return tuple(
            (distance(x, centre) + x[3] - d) / s
            for centre, d, s in zip(centres, observed, scales, strict=True)
        )

    def jacobian(x: Sequence[float]) -> tuple[tuple[float, ...], ...]:
        """Return whitened rows ``(g_k, 1) / sigma_k``."""
        rows = []
        for centre, s in zip(centres, scales, strict=True):
            size = distance(x, centre)
            rows.append((*((x[i] - centre[i]) / (size * s) for i in range(3)), 1.0 / s))
        return tuple(rows)

    return least_squares(residual, start, jacobian=jacobian, tolerance=1e-12)


@dataclass(frozen=True)
class DilutionOfPrecision:
    """Dilution-of-precision family of one constellation (clock unknown)."""

    gdop: float
    pdop: float
    hdop: float
    vdop: float
    tdop: float


def dilution_of_precision(
    centres: Sequence[Vector3], receiver: Vector3 = ORIGIN
) -> DilutionOfPrecision:
    """Return GDOP, PDOP, HDOP, VDOP, and TDOP from ``H = (G'G)^-1``.

    References:
        [EQ-LAB-GPS-002], [REF-MISRA-ENGE-2006, Chapter 6].

    """
    g = clock_geometry_matrix(centres, receiver)
    h = inverse_spd(matmul(transpose(g), g))
    return DilutionOfPrecision(
        math.sqrt(h[0][0] + h[1][1] + h[2][2] + h[3][3]),
        math.sqrt(h[0][0] + h[1][1] + h[2][2]),
        math.sqrt(h[0][0] + h[1][1]),
        math.sqrt(h[2][2]),
        math.sqrt(h[3][3]),
    )


def _whitened(centres: Sequence[Vector3], sigmas: Sequence[float]) -> Matrix:
    """Return ``W^(1/2) G`` for the four-column geometry matrix."""
    g = clock_geometry_matrix(centres, ORIGIN)
    return tuple(tuple(v / s for v in row) for row, s in zip(g, sigmas, strict=True))


def clock_fix_covariance(
    centres: Sequence[Vector3], sigmas: Sequence[float], *, weighted: bool
) -> Matrix:
    """Return the first-order covariance of the four-unknown fix.

    Weighted: ``(G' W G)^-1`` with ``W = diag(sigma_k^-2)``. Ordinary:
    ``A Sigma A'`` with ``A = (G'G)^-1 G'`` and ``Sigma = W^-1``.

    References:
        [EQ-LAB-GPS-002], [REF-TEUNISSEN-MONTENBRUCK-2017, Chapter 22],
        [REF-SEBER-WILD-2003, Section 2.1].

    """
    if weighted:
        gw = _whitened(centres, sigmas)
        return inverse_spd(matmul(transpose(gw), gw))
    a = error_map(clock_geometry_matrix(centres, ORIGIN))
    return tuple(
        tuple(
            sum(a[i][n] * a[j][n] * sigmas[n] ** 2 for n in range(len(sigmas)))
            for j in range(4)
        )
        for i in range(4)
    )


def residual_projector(centres: Sequence[Vector3], sigmas: Sequence[float]) -> Matrix:
    """Return ``S = I - G~ (G~'G~)^-1 G~'`` with ``G~ = W^(1/2) G``.

    The whitened residuals of the weighted fix are ``S W^(1/2) e`` to first
    order; ``S`` has rank ``n - 4``.

    References:
        [EQ-LAB-GPS-003], [REF-BROWN-1992].

    """
    gw = _whitened(centres, sigmas)
    hat = matmul(matmul(gw, inverse_spd(matmul(transpose(gw), gw))), transpose(gw))
    n = len(centres)
    return tuple(
        tuple((1.0 if i == j else 0.0) - hat[i][j] for j in range(n)) for i in range(n)
    )


def raim_threshold(probability: float = FALSE_ALARM, degrees: int = 1) -> float:
    """Return the alarm threshold, the ``1 - P_FA`` quantile of ``chi^2(n - 4)``.

    References:
        [EQ-LAB-GPS-003], [REF-BROWN-1992].

    """
    return ChiSquare(degrees).quantile(1.0 - probability)


def detection_probability(noncentrality: float, threshold: float) -> float:
    """Return the detection probability ``P(chi^2_1(lambda) > T)``.

    ``Phi(sqrt(lambda) - sqrt(T)) + Phi(-sqrt(lambda) - sqrt(T))``.

    With one redundant reading the statistic is ``(Z + sqrt(lambda))^2``.

    References:
        [EQ-LAB-GPS-003].

    """
    root, cut = math.sqrt(noncentrality), math.sqrt(threshold)
    return standard_normal_cdf(root - cut) + standard_normal_cdf(-root - cut)


def fault_noncentrality(
    projector: Matrix, sigmas: Sequence[float], index: int, bias: float
) -> float:
    """Return ``lambda = (bias / sigma_k)^2 S_kk`` of a fault on satellite ``k``.

    References:
        [EQ-LAB-GPS-003].

    """
    return (bias / sigmas[index]) ** 2 * projector[index][index]


def minimal_detectable_bias(
    projector: Matrix,
    sigmas: Sequence[float],
    index: int,
    threshold: float,
    level: float = DETECTION_LEVEL,
) -> float:
    """Return the fault on satellite ``k`` detected with probability ``level``.

    Solves ``detection_probability(lambda, T) = level`` for ``sqrt(lambda)``
    with :func:`probstat.optimization.brent_root`, then
    ``bias = sigma_k sqrt(lambda / S_kk)``.

    References:
        [EQ-LAB-GPS-003], [REF-BROWN-1992].

    """
    root = brent_root(
        lambda s: detection_probability(s * s, threshold) - level, 0.0, 40.0
    ).root
    return sigmas[index] * root / math.sqrt(projector[index][index])


def fault_position_shift(
    centres: Sequence[Vector3], sigmas: Sequence[float], index: int, bias: float
) -> tuple[float, float, float, float]:
    """Return the change of the weighted fix caused by a fault on satellite ``k``.

    ``bias (G'WG)^-1 (g_k, 1)' / sigma_k^2`` (East, North, Up, clock range).

    References:
        [EQ-LAB-GPS-002], [REF-BROWN-1992].

    """
    cov = clock_fix_covariance(centres, sigmas, weighted=True)
    row = clock_geometry_matrix(centres, ORIGIN)[index]
    shift = [
        bias * sum(cov[i][j] * row[j] for j in range(4)) / sigmas[index] ** 2
        for i in range(4)
    ]
    return (shift[0], shift[1], shift[2], shift[3])


def clock_readings(
    centres: Sequence[Vector3],
    sigmas: Sequence[float],
    rng: random.Random,
    fault: tuple[int, float] | None = None,
) -> tuple[float, ...]:
    """Return pseudo-ranges ``||S_k|| + c dt + e_k`` with Gaussian errors.

    Args:
        centres: Satellite centres.
        sigmas: Error standard deviations.
        rng: Random stream.
        fault: Optional ``(satellite index, bias in m)``.

    References:
        [REF-MISRA-ENGE-2006, Chapter 6]. No equation is evaluated.

    """
    values = [
        distance(centre, ORIGIN) + CLOCK_BIAS + rng.gauss(0.0, s)
        for centre, s in zip(centres, sigmas, strict=True)
    ]
    if fault is not None:
        values[fault[0]] += fault[1]
    return tuple(values)


@dataclass(frozen=True)
class ClockFixStudy:
    """Monte Carlo errors of the ordinary and weighted four-unknown fixes.

    Attributes:
        ols_errors: ``(E, N, U, beta - c dt)`` errors of the ordinary fix.
        wls_errors: The same for the weighted fix.
        statistics: Fault-test statistic ``T`` of every fault-free draw.
        false_alarm: Share of fault-free draws with ``T`` above threshold.
        false_alarm_se: Its standard error.

    """

    ols_errors: tuple[tuple[float, ...], ...]
    wls_errors: tuple[tuple[float, ...], ...]
    statistics: tuple[float, ...]
    false_alarm: float
    false_alarm_se: float


def study_clock_fix(
    centres: Sequence[Vector3],
    sigmas: Sequence[float],
    draws: int,
    seed: int,
    threshold: float,
) -> ClockFixStudy:
    """Simulate fault-free readings and solve the ordinary and weighted fixes.

    References:
        [REF-TEUNISSEN-MONTENBRUCK-2017, Chapter 22], [REF-BROWN-1992].

    """
    rng = random.Random(seed)
    truth = (0.0, 0.0, 0.0, CLOCK_BIAS)
    ols, wls, stats = [], [], []
    for _ in range(draws):
        observed = clock_readings(centres, sigmas, rng)
        plain = solve_clock_fix(centres, observed).x
        weighted = solve_clock_fix(centres, observed, sigmas)
        ols.append(tuple(v - t for v, t in zip(plain, truth, strict=True)))
        wls.append(tuple(v - t for v, t in zip(weighted.x, truth, strict=True)))
        stats.append(2.0 * weighted.cost)
    alarms = [1.0 if t > threshold else 0.0 for t in stats]
    rate, rate_se = _mean_se(alarms)
    return ClockFixStudy(tuple(ols), tuple(wls), tuple(stats), rate, rate_se)


def detection_rate(
    centres: Sequence[Vector3],
    sigmas: Sequence[float],
    fault: tuple[int, float],
    draws: int,
    seed: int,
    threshold: float,
) -> tuple[float, float]:
    """Return the simulated detection rate of a fault and its standard error.

    References:
        [REF-BROWN-1992]. No equation is evaluated.

    """
    rng = random.Random(seed)
    alarms = []
    for _ in range(draws):
        observed = clock_readings(centres, sigmas, rng, fault)
        statistic = 2.0 * solve_clock_fix(centres, observed, sigmas).cost
        alarms.append(1.0 if statistic > threshold else 0.0)
    return _mean_se(alarms)


@dataclass(frozen=True)
class RaimRow:
    """Fault-detection summary of one satellite.

    Attributes:
        index: Satellite index (0-based).
        elevation: Elevation in degrees.
        sigma: Range-error standard deviation.
        redundancy: Diagonal entry ``S_kk`` of the residual projector.
        bias: Minimal detectable bias at ``DETECTION_LEVEL``.
        horizontal_error: Horizontal fix shift that this bias causes.
        checks: ``(bias, closed form, simulated rate, standard error)`` at
            the biases detected with probability 0.5 and 0.9.

    """

    index: int
    elevation: float
    sigma: float
    redundancy: float
    bias: float
    horizontal_error: float
    checks: tuple[tuple[float, float, float, float], ...]


# --------------------------------------------------------------------------
# Whole analysis
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GeometryPoint:
    """Known-clock PDOP and simulated 90% radius of one constellation."""

    label: str
    pdop: float
    radius: float
    radius_se: float
    linear_radius: float


@dataclass(frozen=True)
class LabResults:
    """Everything the report and the figures need."""

    centres: tuple[Vector3, ...]
    noise_free_error: float
    posterior: PosteriorStudy
    fix_all: FixStudy
    fix_by_source: dict[str, FixStudy]
    linear_by_source: dict[str, float]
    linear_covariance: list[list[float]]
    mc_covariance: list[list[float]]
    covariance_se: list[list[float]]
    no_range_radius: float
    no_range_radius_se: float
    geometries: tuple[GeometryPoint, ...]
    sigmas: tuple[float, ...]
    clock_noise_free: tuple[float, float]
    three_unknown_error: float
    dops: dict[str, DilutionOfPrecision]
    known_clock_pdops: dict[str, float]
    clock_study: ClockFixStudy
    ols_covariance: Matrix
    wls_covariance: Matrix
    threshold: float
    raim: tuple[RaimRow, ...]
    seconds: float


def run_part_two(seed: int, *, draws: int = CLOCK_FIX_DRAWS) -> dict[str, Any]:
    """Run the unknown-clock fix, the DOP family, and the fault-detection test.

    Args:
        seed: Base seed of Part 2.
        draws: Fault-free draws of the ordinary and weighted fixes.

    Returns:
        The Part 2 fields of :class:`LabResults`.

    References:
        [REF-MISRA-ENGE-2006, Chapter 6], [REF-BROWN-1992].

    """
    centres = constellation(GOOD_SKY)
    sigmas = elevation_sigmas(GOOD_SKY)
    exact = tuple(distance(c, ORIGIN) + CLOCK_BIAS for c in centres)
    solved = solve_clock_fix(centres, exact).x
    noise_free = (distance(solved, ORIGIN), abs(solved[3] - CLOCK_BIAS))
    three = distance(solve_fix(centres, exact).x, ORIGIN)
    threshold = raim_threshold()
    study = study_clock_fix(centres, sigmas, draws, seed, threshold)
    projector = residual_projector(centres, sigmas)
    rows = []
    for k in range(len(centres)):
        checks = []
        for level in (0.5, DETECTION_LEVEL):
            bias = minimal_detectable_bias(projector, sigmas, k, threshold, level)
            rate, rate_se = detection_rate(
                centres,
                sigmas,
                (k, bias),
                DETECTION_DRAWS,
                seed + 1 + 2 * k + (level > 0.5),
                threshold,
            )
            checks.append((bias, level, rate, rate_se))
        mdb = checks[-1][0]
        shift = fault_position_shift(centres, sigmas, k, mdb)
        rows.append(
            RaimRow(
                k,
                GOOD_SKY[k][1],
                sigmas[k],
                projector[k][k],
                mdb,
                math.hypot(shift[0], shift[1]),
                tuple(checks),
            )
        )
    skies = {"well spread": GOOD_SKY, "clustered": POOR_SKY}
    return {
        "sigmas": sigmas,
        "clock_noise_free": noise_free,
        "three_unknown_error": three,
        "dops": {
            name: dilution_of_precision(constellation(sky))
            for name, sky in skies.items()
        },
        "known_clock_pdops": {
            name: pdop_known_clock(constellation(sky)) for name, sky in skies.items()
        },
        "clock_study": study,
        "ols_covariance": clock_fix_covariance(centres, sigmas, weighted=False),
        "wls_covariance": clock_fix_covariance(centres, sigmas, weighted=True),
        "threshold": threshold,
        "raim": tuple(rows),
    }


def run_analysis(seed: int = SEED, *, fix_draws: int = FIX_DRAWS) -> LabResults:
    """Run every part of the laboratory.

    Args:
        seed: Base seed; each part uses its own offset stream.
        fix_draws: Draws of each Part 1 least-squares scenario.

    Returns:
        The laboratory results.

    References:
        [REF-SMITH-UQ-2024], [REF-OWEN-2013].

    """
    started = time.perf_counter()
    centres = constellation(GOOD_SKY)
    exact = readings(centres, ORIGIN, draw_sources(random.Random(seed), 5, ()))
    noise_free = solve_fix(centres, exact, start=(3000.0, -4000.0, 1500.0)).x
    posterior = study_posterior(centres, seed + 1)
    fix_all = study_fix(centres, fix_draws, seed + 2)
    by_source = {
        name: study_fix(centres, fix_draws, seed + 3 + index, (name,))
        for index, name in enumerate(SOURCES)
    }
    linear_by_source = {
        name: sum(
            linearized_covariance(centres, active=(name,))[i][i] for i in range(3)
        )
        for name in SOURCES
    }
    no_range = study_fix(centres, fix_draws, seed + 7, ("clock", "satellite"))
    geometries = []
    rng = random.Random(seed + 8)
    skies = [("well spread", GOOD_SKY), ("clustered", POOR_SKY)]
    for index in range(RANDOM_GEOMETRIES):
        sky = tuple(
            (rng.uniform(0.0, 360.0), rng.uniform(15.0, 80.0)) for _ in range(5)
        )
        skies.append((f"random {index + 1}", sky))
    for index, (label, sky) in enumerate(skies):
        sats = constellation(sky)
        study = study_fix(sats, GEOMETRY_DRAWS, seed + 20 + index)
        geometries.append(
            GeometryPoint(
                label,
                pdop_known_clock(sats),
                study.radius,
                study.radius_se,
                _linear_radius(linearized_covariance(sats), seed + 60 + index),
            )
        )
    part_two = run_part_two(seed + 100)
    return LabResults(
        centres,
        distance(noise_free, ORIGIN),
        posterior,
        fix_all,
        by_source,
        linear_by_source,
        linearized_covariance(centres),
        sample_covariance(fix_all.errors),
        _covariance_se(fix_all.errors),
        no_range.radius,
        no_range.radius_se,
        tuple(geometries),
        seconds=time.perf_counter() - started,
        **part_two,
    )


def gauss_newton_check(centres: Sequence[Vector3]) -> float:
    """Return the largest gap between the library and the lab covariance.

    The Gauss-Newton covariance ``s^2 (J'J)^-1`` of a noise-free fix, with
    ``s^2`` the independent per-reading variance, must equal the lab's
    ``A Sigma_e A'`` without the clock term.

    References:
        [REF-SEBER-WILD-2003, Section 2.1].

    """
    exact = readings(centres, ORIGIN, draw_sources(random.Random(0), 5, ()))
    result = solve_fix(centres, exact)
    variances = source_variances()
    library = gauss_newton_covariance(
        result, noise_variance=variances["satellite"] + variances["range"]
    )
    lab = linearized_covariance(centres, active=("satellite", "range"))
    return max(abs(library[i][j] - lab[i][j]) for i in range(3) for j in range(3))


def weighted_covariance_check(
    centres: Sequence[Vector3], sigmas: Sequence[float]
) -> float:
    """Return the largest gap between the library and the lab ``(G'WG)^-1``.

    References:
        [REF-SEBER-WILD-2003, Section 2.1].

    """
    exact = tuple(distance(c, ORIGIN) + CLOCK_BIAS for c in centres)
    result = solve_clock_fix(centres, exact, sigmas)
    library = gauss_newton_covariance(result, noise_variance=1.0)
    lab = clock_fix_covariance(centres, sigmas, weighted=True)
    return max(abs(library[i][j] - lab[i][j]) for i in range(4) for j in range(4))


def _trace3(matrix: Sequence[Sequence[float]]) -> float:
    """Return the trace of the leading 3 x 3 block."""
    return matrix[0][0] + matrix[1][1] + matrix[2][2]


def print_report(results: LabResults) -> None:
    """Print the laboratory's numbers with their Monte Carlo errors.

    References:
        [REF-SMITH-UQ-2024], [REF-MISRA-ENGE-2006].

    """
    post = results.posterior
    print("Satellites (East, North, Up) [km] and range to the receiver [km]:")
    for centre in results.centres:
        print(
            f"  ({centre[0] / 1e3:9.1f}, {centre[1] / 1e3:9.1f}, "
            f"{centre[2] / 1e3:9.1f})"
            f"  {distance(centre, ORIGIN) / 1e3:8.1f}"
        )
    print("\n=== Part 1: known clock, three uncertainty sources ===")
    print(
        "Known-clock PDOP of the well-spread sky: "
        f"{pdop_known_clock(results.centres):.2f}"
    )
    print(f"Noise-free fix error: {results.noise_free_error:.2e} m")
    print(
        "Gauss-Newton covariance vs A Sigma_e A' (no clock), largest gap: "
        f"{gauss_newton_check(results.centres):.1e} m^2"
    )
    print("\nPosterior f_R by importance sampling (one set of readings):")
    print(
        f"  E[R] = ({post.mean[0]:.2f}, {post.mean[1]:.2f}, {post.mean[2]:.2f}) m, "
        f"batch SE ({post.mean_se[0]:.2f}, {post.mean_se[1]:.2f}, "
        f"{post.mean_se[2]:.2f})"
    )
    print(
        f"  least-squares fix = ({post.fix[0]:.2f}, {post.fix[1]:.2f}, "
        f"{post.fix[2]:.2f}) m"
    )
    print(f"  ||E[R] - R0|| = {distance(post.mean, ORIGIN):.2f} m")
    print(
        f"  90% radius around E[R]: {post.radius:.2f} m (batch SE {post.radius_se:.2f})"
    )
    print(
        f"  ESS {post.ess:.0f} of {post.proposals} proposals; positive "
        f"weight {100 * post.positive_share:.1f}%; edge hits {post.edge_hits}"
    )
    print(
        f"  clock quadrature {CLOCK_NODES} vs {CHECK_NODES} nodes, largest relative "
        f"likelihood change: {post.quadrature_gap:.1e}"
    )
    fix = results.fix_all
    print(f"\nLeast-squares fix, {FIX_DRAWS} draws of all sources:")
    print(
        f"  90% radius of ||R_hat - R0||: {fix.radius:.2f} m "
        f"(batch SE {fix.radius_se:.2f})"
    )
    print(
        f"  mean squared error: {fix.mean_square:.2f} m^2 (SE {fix.mean_square_se:.2f})"
    )
    print("  covariance entry: Monte Carlo (SE) vs linearized [m^2]")
    for i, a in enumerate("ENU"):
        for j in range(i, 3):
            print(
                f"    {a}{'ENU'[j]}: {results.mc_covariance[i][j]:7.2f} "
                f"({results.covariance_se[i][j]:.2f})  "
                f"{results.linear_covariance[i][j]:7.2f}"
            )
    print("\nSensitivity split of the mean squared error [m^2]:")
    total = sum(results.linear_by_source.values())
    for name in SOURCES:
        study = results.fix_by_source[name]
        print(
            f"  {name:9s}: Monte Carlo {study.mean_square:8.3f} "
            f"(SE {study.mean_square_se:.3f})"
            f"  linearized {results.linear_by_source[name]:8.3f}"
            f"  share {100 * results.linear_by_source[name] / total:5.1f}%"
        )
    added = sum(results.fix_by_source[name].mean_square for name in SOURCES)
    added_se = math.sqrt(
        sum(results.fix_by_source[name].mean_square_se ** 2 for name in SOURCES)
    )
    print(
        f"  sum of single-source runs {added:.2f} (SE {added_se:.2f}) vs all-source "
        f"run {fix.mean_square:.2f} (SE {fix.mean_square_se:.2f})"
    )
    print(
        "  90% radius without the range error (clock and satellites only): "
        f"{results.no_range_radius:.2f} m (SE {results.no_range_radius_se:.2f})"
    )
    print(
        "\nGeometry: known-clock PDOP and 90% radius "
        "(Monte Carlo, SE, linearized normal) [m]:"
    )
    for point in results.geometries:
        print(
            f"  {point.label:12s} PDOP {point.pdop:5.2f}  {point.radius:6.2f} "
            f"({point.radius_se:.2f})  {point.linear_radius:6.2f}"
        )

    print("\n=== Part 2: unknown receiver clock, weighted fix, fault detection ===")
    position_gap, clock_gap = results.clock_noise_free
    print(
        f"Clock offset {RECEIVER_CLOCK_OFFSET * 1e3:.2f} ms = "
        f"{CLOCK_BIAS / 1e3:.1f} km "
        "of range."
    )
    print(
        f"  four-unknown fix from exact readings: position error {position_gap:.1e} m, "
        f"clock error {clock_gap:.1e} m"
    )
    print(
        "  three-unknown fix from the same readings (clock ignored): position "
        f"error {results.three_unknown_error / 1e3:.1f} km"
    )
    print("Dilution of precision (clock unknown; last column: known-clock PDOP):")
    print("  sky          GDOP  PDOP  HDOP  VDOP  TDOP   PDOP|clock")
    for name, dop in results.dops.items():
        print(
            f"  {name:12s} {dop.gdop:5.2f} {dop.pdop:5.2f} {dop.hdop:5.2f} "
            f"{dop.vdop:5.2f} {dop.tdop:5.2f}   {results.known_clock_pdops[name]:5.2f}"
        )
    sigma_text = ", ".join(f"{s:.2f}" for s in results.sigmas)
    print(f"Range-error SD sigma_0 / sin(elevation) [m]: {sigma_text}")
    study = results.clock_study
    print(
        "  weighted (G'WG)^-1 vs library Gauss-Newton covariance, largest gap: "
        f"{weighted_covariance_check(results.centres, results.sigmas):.1e}"
    )
    print(
        f"  {CLOCK_FIX_DRAWS} draws: Monte Carlo (SE) vs first order "
        "[position m^2 | clock m]"
    )
    for label, errors, cov in (
        ("ordinary", study.ols_errors, results.ols_covariance),
        ("weighted", study.wls_errors, results.wls_covariance),
    ):
        mse, mse_se = _mean_se([e[0] ** 2 + e[1] ** 2 + e[2] ** 2 for e in errors])
        clock_values = [e[3] for e in errors]
        clock_sd = math.sqrt(covariance(clock_values, clock_values, ddof=1))
        print(
            f"  {label:9s} position MSE {mse:6.2f} ({mse_se:.2f}) vs "
            f"{_trace3(cov):6.2f} | clock SD {clock_sd:5.2f} vs "
            f"{math.sqrt(cov[3][3]):5.2f}"
        )
    mean_t, mean_t_se = _mean_se(study.statistics)
    print(
        f"Fault test: threshold chi2(1) quantile {1 - FALSE_ALARM:.2f} = "
        f"{results.threshold:.2f}; mean T {mean_t:.3f} (SE {mean_t_se:.3f}, "
        "expected 1)"
    )
    print(
        f"  false-alarm rate {100 * study.false_alarm:.2f}% "
        f"(SE {100 * study.false_alarm_se:.2f}) vs {100 * FALSE_ALARM:.0f}%"
    )
    print(
        "  sat  elev  sigma   S_kk   MDB(90%)  horizontal shift   "
        "detection at MDB: simulated (SE)"
    )
    for row in results.raim:
        bias, level, rate, rate_se = row.checks[-1]
        print(
            f"  S{row.index + 1}  {row.elevation:4.0f}  {row.sigma:5.2f}  "
            f"{row.redundancy:5.3f}  {row.bias:7.1f} m  {row.horizontal_error:7.1f} m"
            f"          {rate:.3f} ({rate_se:.3f}) vs {level:.1f}"
        )
    for row in results.raim:
        bias, level, rate, rate_se = row.checks[0]
        print(
            f"  S{row.index + 1} at {bias:5.1f} m: simulated {rate:.3f} "
            f"({rate_se:.3f}) "
            f"vs {level:.1f}"
        )


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _axes(figsize: tuple[float, float] = (7.2, 4.4)) -> tuple[Any, Any]:
    """Return a themed figure and axes."""
    plt = theme.pyplot()
    figure, ax = plt.subplots(figsize=figsize)
    theme.style_axes(ax)
    return figure, ax


def _label(ax: Any, x: float, y: float, text: str, colour: str, **kwargs: Any) -> None:
    """Write a direct label next to a point."""
    options: dict[str, Any] = {
        "xytext": (6.0, 0.0),
        "textcoords": "offset points",
        "ha": "left",
        "va": "center",
        "fontsize": theme.ANNOTATION_SIZE,
        "fontweight": "bold",
        "color": colour,
    }
    options.update(kwargs)
    ax.annotate(text, (x, y), **options)


def figure_posterior_cloud(results: LabResults) -> Any:
    """Draw the posterior sample of ``R`` in the East-North plane.

    References:
        [REF-MATPLOTLIB].

    """
    palette = theme.active_palette()
    post = results.posterior
    figure, ax = _axes((6.4, 5.6))
    rng = random.Random(SEED)
    picks = rng.choices(range(len(post.points)), weights=post.weights, k=2500)
    ax.scatter(
        [post.points[i][0] for i in picks],
        [post.points[i][1] for i in picks],
        s=5,
        color=palette.observed,
        alpha=0.35,
        linewidths=0,
    )
    angles = [2 * math.pi * t / 200 for t in range(201)]
    ax.plot(
        [post.mean[0] + post.radius * math.cos(t) for t in angles],
        [post.mean[1] + post.radius * math.sin(t) for t in angles],
        color=palette.ink,
        linewidth=theme.REFERENCE_WIDTH,
    )
    ax.plot([0.0], [0.0], marker="+", markersize=12, color=palette.model, mew=2.0)
    ax.plot([post.mean[0]], [post.mean[1]], marker="o", markersize=6, color=palette.ink)
    _label(
        ax,
        0.0,
        0.0,
        "True position",
        palette.model,
        xytext=(-8.0, 8.0),
        ha="right",
        va="bottom",
    )
    _label(ax, post.mean[0], post.mean[1], "E[R]", palette.ink, xytext=(8.0, 8.0))
    _label(
        ax,
        post.mean[0] + post.radius * math.cos(0.8),
        post.mean[1] + post.radius * math.sin(0.8),
        f"90% sphere, radius {post.radius:.1f} m",
        palette.ink,
        xytext=(6.0, 6.0),
        va="bottom",
    )
    ax.set_aspect("equal", adjustable="datalim")
    theme.axis_labels(ax, "East", "North", x_unit="m", y_unit="m")
    theme.title_block(
        ax,
        f"Five ranges fix the receiver within {post.radius:.0f} m",
        "Posterior draws of R (importance resampled), East-North projection",
    )
    return figure


def figure_radial_cdf(results: LabResults) -> Any:
    """Draw the CDF of the radial error for the posterior and the fix.

    References:
        [REF-MATPLOTLIB].

    """
    palette = theme.active_palette()
    post = results.posterior
    figure, ax = _axes()
    radii = [distance(p, post.mean) for p in post.points]
    order = sorted(range(len(radii)), key=radii.__getitem__)
    running, xs, ys = 0.0, [], []
    for index in order:
        running += post.weights[index]
        xs.append(radii[index])
        ys.append(running)
    ax.plot(xs, ys, color=palette.observed, linewidth=theme.LINE_WIDTH)
    fix_norms = sorted(math.sqrt(sum(v * v for v in e)) for e in results.fix_all.errors)
    n = len(fix_norms)
    ax.plot(
        fix_norms,
        [(i + 1) / n for i in range(n)],
        color=palette.model,
        linewidth=theme.LINE_WIDTH,
    )
    ax.axhline(
        LEVEL, color=palette.muted, linewidth=theme.REFERENCE_WIDTH, linestyle="--"
    )
    _label(
        ax,
        post.radius,
        LEVEL,
        f"Posterior: {post.radius:.1f} m",
        palette.observed,
        xytext=(-6.0, 10.0),
        ha="right",
    )
    fix = results.fix_all
    _label(
        ax,
        fix.radius,
        LEVEL,
        f"Least-squares fix: {fix.radius:.1f} m",
        palette.model,
        xytext=(6.0, -12.0),
    )
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0.0, max(fix_norms[-1], xs[-1]) * 1.02)
    theme.axis_labels(ax, "Distance from centre", "Probability within", x_unit="m")
    theme.title_block(
        ax,
        f"90% radius: {post.radius:.1f} m for the posterior, "
        f"{fix.radius:.1f} m for the fix",
        "Posterior: ||R - E[R]|| for one set of readings; "
        "fix: ||R_hat - R0|| over draws",
    )
    return figure


def figure_sensitivity(results: LabResults) -> Any:
    """Draw the mean squared position error contributed by each source.

    References:
        [REF-MATPLOTLIB].

    """
    from matplotlib.ticker import FuncFormatter, NullFormatter

    palette = theme.active_palette()
    figure, ax = _axes((7.2, 3.6))
    names = list(SOURCES)
    labels = {
        "clock": "Clock shift",
        "satellite": "Satellite position",
        "range": "Range reading",
    }
    for row, name in enumerate(names):
        study = results.fix_by_source[name]
        low = max(study.mean_square - 2 * study.mean_square_se, 1e-6)
        ax.plot(
            [low, study.mean_square + 2 * study.mean_square_se],
            [row, row],
            color=palette.observed,
            linewidth=3.0,
            solid_capstyle="butt",
        )
        ax.plot(
            [study.mean_square], [row], marker="o", markersize=7, color=palette.observed
        )
        ax.plot(
            [results.linear_by_source[name]],
            [row],
            marker="|",
            markersize=16,
            mew=2.0,
            color=palette.model,
        )
    ax.set_xscale("log")
    ax.set_yticks(range(len(names)), [labels[n] for n in names])
    ax.set_ylim(-0.6, len(names) - 0.2)
    top = len(names) - 1
    _label(
        ax,
        results.fix_by_source[names[top]].mean_square,
        top,
        "Monte Carlo ± 2 SE",
        palette.observed,
        xytext=(0.0, 12.0),
        ha="right",
        va="bottom",
    )
    _label(
        ax,
        results.linear_by_source[names[top]],
        top,
        "Linearized",
        palette.model,
        xytext=(0.0, -14.0),
        ha="center",
        va="top",
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", color=palette.rule, linewidth=0.7)
    ax.grid(axis="y", visible=False)
    total = sum(results.linear_by_source.values())
    share = 100 * results.linear_by_source["range"] / total
    theme.axis_labels(ax, "Mean squared position error", None, x_unit="m²")
    theme.title_block(
        ax,
        f"The range reading carries {share:.0f}% of the position error",
        "One source switched on at a time; least-squares fix",
    )
    return figure


def figure_geometry(results: LabResults) -> Any:
    """Draw the 90% radius against the known-clock position dilution.

    References:
        [REF-MATPLOTLIB].

    """
    palette = theme.active_palette()
    figure, ax = _axes()
    points = sorted(results.geometries, key=lambda p: p.pdop)
    slope = math.fsum(p.linear_radius * p.pdop for p in points) / math.fsum(
        p.pdop**2 for p in points
    )
    top = points[-1].pdop * 1.05
    ax.plot(
        [0.0, top],
        [0.0, slope * top],
        color=palette.model,
        linewidth=theme.REFERENCE_WIDTH,
    )
    for p in points:
        ax.plot(
            [p.pdop, p.pdop],
            [p.radius - 2 * p.radius_se, p.radius + 2 * p.radius_se],
            color=palette.observed,
            linewidth=1.5,
        )
    ax.plot(
        [p.pdop for p in points],
        [p.radius for p in points],
        "o",
        color=palette.observed,
        markersize=6,
    )
    for p in points:
        if p.label in {"well spread", "clustered"}:
            _label(
                ax,
                p.pdop,
                p.radius,
                p.label.capitalize(),
                palette.ink,
                xytext=(8.0, -8.0) if p.label == "clustered" else (8.0, -10.0),
            )
    anchor = 0.7 * top
    _label(
        ax,
        anchor,
        slope * anchor,
        f"Linearized: {slope:.1f} m per PDOP unit",
        palette.model,
        xytext=(10.0, -10.0),
        ha="left",
        va="top",
    )
    ax.set_xlim(0.0, top * 1.02)
    ax.set_ylim(0.0, max(p.radius + 2 * p.radius_se for p in points) * 1.08)
    theme.axis_labels(
        ax, "Position dilution of precision, clock known", "90% radius", y_unit="m"
    )
    theme.title_block(
        ax,
        "Error radius grows in proportion to the dilution of precision",
        f"Least-squares fix, {GEOMETRY_DRAWS} draws per constellation, bars ± 2 SE",
    )
    return figure


def figure_raim(results: LabResults) -> Any:
    """Draw the detection probability of a range fault against its size.

    References:
        [REF-MATPLOTLIB].

    """
    palette = theme.active_palette()
    figure, ax = _axes()
    centres = results.centres
    sigmas = results.sigmas
    projector = residual_projector(centres, sigmas)
    hardest = max(results.raim, key=lambda row: row.bias)
    top = hardest.bias * 1.15
    grid = [top * i / 200 for i in range(201)]
    for row in results.raim:
        colour = palette.accent if row is hardest else palette.muted
        width = theme.LINE_WIDTH if row is hardest else theme.REFERENCE_WIDTH
        curve = [
            detection_probability(
                fault_noncentrality(projector, sigmas, row.index, b), results.threshold
            )
            for b in grid
        ]
        ax.plot(grid, curve, color=colour, linewidth=width)
        for bias, _, rate, rate_se in row.checks:
            ax.plot(
                [bias, bias],
                [rate - 2 * rate_se, rate + 2 * rate_se],
                color=palette.observed,
                linewidth=1.2,
            )
            ax.plot([bias], [rate], "o", color=palette.observed, markersize=4)
    ax.axhline(
        DETECTION_LEVEL,
        color=palette.rule,
        linewidth=theme.REFERENCE_WIDTH,
        linestyle="--",
    )
    ax.set_xlim(0.0, top * 1.12)
    ax.set_ylim(0.0, 1.04)
    others = [row for row in results.raim if row is not hardest]
    widest = max(others, key=lambda row: row.bias)
    names = ", ".join(f"S{row.index + 1}" for row in others)
    _label(
        ax,
        widest.bias,
        0.62,
        f"{names}:\n90% at {min(row.bias for row in others):.0f} to "
        f"{widest.bias:.0f} m",
        palette.ink,
        xytext=(6.0, 0.0),
    )
    half_bias = hardest.checks[0][0]
    _label(
        ax,
        half_bias,
        detection_probability(
            fault_noncentrality(projector, sigmas, hardest.index, half_bias),
            results.threshold,
        ),
        f"S{hardest.index + 1} ({hardest.elevation:.0f}° elevation)",
        palette.accent,
        xytext=(14.0, -8.0),
        va="top",
    )
    theme.axis_labels(
        ax, "Range fault on one satellite", "Detection probability", x_unit="m"
    )
    theme.title_block(
        ax,
        f"A {hardest.bias:.0f} m fault on S{hardest.index + 1} is needed "
        "for 90% detection",
        f"Closed-form detection probability per satellite, P_FA = {FALSE_ALARM:g}; "
        "dots: simulation ± 2 SE",
    )
    return figure


FIGURES: tuple[tuple[str, Callable[[LabResults], Any]], ...] = (
    ("gps_posterior_cloud.png", figure_posterior_cloud),
    ("gps_radial_cdf.png", figure_radial_cdf),
    ("gps_sensitivity.png", figure_sensitivity),
    ("gps_geometry.png", figure_geometry),
    ("gps_raim_detection.png", figure_raim),
)


def write_figures(results: LabResults, output: Path) -> list[Path]:
    """Draw and save every figure of the laboratory.

    Args:
        results: Output of :func:`run_analysis`.
        output: Destination folder.

    Returns:
        The written paths.

    References:
        [REF-MATPLOTLIB].

    """
    plt = theme.pyplot()
    written = []
    with theme.chart_theme(theme.active_palette()):
        for name, draw in FIGURES:
            figure = draw(results)
            written.append(save_figure(figure, output / name))
            plt.close(figure)
    return written


def main() -> None:
    """Run the laboratory, print its report, and save its figures.

    References:
        [REF-SMITH-UQ-2024], [REF-MATPLOTLIB].

    """
    started = time.perf_counter()
    print("Simulating five-satellite position fixes (about a minute)...")
    results = run_analysis()
    print_report(results)
    output = Path(__file__).parent / "generated"
    written = write_figures(results, output)
    post = results.posterior
    hardest = max(results.raim, key=lambda row: row.bias)
    print(
        f"\nResult: E[R] lies {distance(post.mean, ORIGIN):.1f} m from the true "
        f"position; 90% of f_R lies within {post.radius:.1f} m of E[R]. With the "
        f"clock unknown, a fault must reach {hardest.bias:.0f} m on "
        f"S{hardest.index + 1} to be detected 90% of the time."
    )
    folder = output.relative_to(Path(__file__).parent.parent)
    print(
        f"Wrote {len(written)} figures in {time.perf_counter() - started:.0f} s. "
        f"Next: open {folder / FIGURES[0][0]}"
    )


if __name__ == "__main__":
    main()
