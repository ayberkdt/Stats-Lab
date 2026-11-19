"""GPS positioning laboratory: the law of a receiver position from five ranges.

Five satellites ``S_1, ..., S_5`` broadcast signals; the receiver reads the
pseudo-ranges ``D_k = c tau + ||S_k - R|| + e_k`` and must deduce its own
position ``R``. Three sources of uncertainty enter every reading: an
unknown clock shift ``tau`` common to all satellites, the satellite
positions (known only to lie in small spheres around their broadcast
centres), and the range reading itself. Following a classical V&V-UQ
teaching example, the laboratory computes the density of ``R`` given one set
of readings,

    f_R(r) proportional to E_{tau, S}[ prod_k f_e(D_k - c tau - ||r - S_k||) ],

its mean ``E[R]`` and the radius of the sphere around ``E[R]`` that holds 90%
of the probability, by self-normalized importance sampling. It then
propagates the same three sources through a nonlinear least-squares fix (one
solve per draw), splits the position error by source, checks the
linearization through the geometry matrix, and shows how the satellite
geometry (dilution of precision) scales the error.

Definitions:
    local frame: East, North, Up coordinates in metres with the true receiver
        ``R0`` at the origin on a spherical Earth of radius 6371 km; every
        satellite sits on a 26 560 km orbit radius, 20 000 to 25 000 km
        away, at a stated azimuth and elevation.
    sources: clock shift ``tau ~ Uniform(-1e-9, 1e-9)`` s (so ``c tau`` is
        within 0.30 m); satellite position uniform in a ball of radius 2 m
        around its broadcast centre; range error ``e_k ~ Uniform(-10, 10)`` m,
        independent across satellites.
    geometry matrix ``G``: rows ``g_k = (R0 - S_k) / ||R0 - S_k||``, the unit
        lines of sight; ``A = (G'G)^-1 G'`` maps range errors to position
        errors to first order.
    position dilution of precision ``PDOP = sqrt(trace((G'G)^-1))``.
    90% radius: the radius ``rho`` of the sphere around ``E[R]`` (posterior)
        or around ``R0`` (least-squares fix) holding probability 0.9.

Assumptions:
    - The three sources are independent; the clock shift is the same for all
      five readings; the receiver knows the broadcast centres, not the true
      satellite positions.
    - The fix solves for the three position coordinates only; the clock
      shift is a nuisance integrated out in the posterior and ignored (set to
      zero) by the least-squares fix, which is defensible because ``c tau``
      is at most 0.3 m against 10 m range errors.
    - The posterior uses a flat prior on ``R``; the inner expectation over
      ``(tau, S)`` uses ``INNER_DRAWS`` common draws for every candidate
      position (common random numbers).
    - Signal travel time, Earth rotation, and atmospheric delays are left out.

Equations:
    EQ-OPT-LM-001: Levenberg-Marquardt solve of each least-squares fix.
    EQ-OPT-LM-002: Gauss-Newton covariance ``s^2 (J'J)^-1`` of the fix.
    EQ-FILT-PF-002: effective sample size of the importance weights.
    EQ-STAT-QUANT-001: sample quantiles of the radial error.

Diagnostics:
    - Noise-free readings recover the true position to below a micrometre.
    - The linearized covariance ``A Sigma_e A'`` agrees with the Monte Carlo
      covariance of the fix within its standard error; the three
      single-source runs add up to the all-source run.
    - Importance sampling reports its effective sample size, the share of
      proposals with positive weight, and whether any weighted proposal
      touches the edge of the proposal box (it must not).
    - Each 90% radius carries a batch standard error from ``BATCHES``
      independent batches.

Cost and error:
    - About 30 to 60 seconds in pure Python; the run prints its wall-clock
      time. Most of it goes to ``FIX_DRAWS`` least-squares fixes per scenario
      and ``PROPOSALS x INNER_DRAWS`` likelihood terms.
    - Monte Carlo standard errors are printed next to every simulated
      number; differences smaller than two standard errors are not findings.

References:
    [REF-SMITH-UQ-2024], [REF-OWEN-2013], [REF-NOCEDAL-WRIGHT-2006],
    [REF-SEBER-WILD-2003], [REF-DOUCET-JOHANSEN-2011], [REF-HYNDMAN-FAN-1996],
    [REF-MATPLOTLIB].

"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from probstat.filtering import effective_sample_size
from probstat.optimization import gauss_newton_covariance, least_squares
from probstat.statistics import quantile
from probstat.visualization import save_figure, theme

# Seed of every random stream in the laboratory.
SEED = 2031

SPEED_OF_LIGHT = 299_792_458.0  # m/s
CLOCK_HALF_WIDTH = 1e-9  # s
SATELLITE_RADIUS = 2.0  # m
RANGE_HALF_WIDTH = 10.0  # m
EARTH_RADIUS = 6_371_000.0  # m
ORBIT_RADIUS = 26_560_000.0  # m
LEVEL = 0.9

# Azimuth and elevation in degrees: a well-spread sky and a clustered one.
GOOD_SKY = ((20.0, 78.0), (75.0, 28.0), (160.0, 42.0), (245.0, 22.0), (320.0, 50.0))
POOR_SKY = ((60.0, 35.0), (85.0, 48.0), (100.0, 30.0), (120.0, 55.0), (140.0, 38.0))

FIX_DRAWS = 4000
GEOMETRY_DRAWS = 600
RANDOM_GEOMETRIES = 10
PROPOSALS = 30000
INNER_DRAWS = 48
BATCHES = 10
PILOT_MINIMUM = 400

Vector3 = tuple[float, float, float]
SOURCES = ("clock", "satellite", "range")


# --------------------------------------------------------------------------
# Geometry and sampling
# --------------------------------------------------------------------------


def satellite_centre(azimuth: float, elevation: float) -> Vector3:
    """Return the East-North-Up position of a satellite seen at a direction.

    The satellite lies on the orbit sphere of radius ``ORBIT_RADIUS`` around
    the Earth centre ``(0, 0, -EARTH_RADIUS)``; its range ``rho`` along the
    unit direction ``u`` solves ``||rho u + (0, 0, EARTH_RADIUS)|| = ORBIT_RADIUS``.

    Args:
        azimuth: Degrees clockwise from North.
        elevation: Degrees above the horizon.

    Returns:
        Satellite centre in metres.

    References:
        [REF-SMITH-UQ-2024]. No equation is evaluated.

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
        [REF-SMITH-UQ-2024]. No equation is evaluated.

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
    """One draw of the three uncertainty sources.

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
    errors = tuple(rng.uniform(-RANGE_HALF_WIDTH, RANGE_HALF_WIDTH) for _ in range(count))
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
        [REF-SMITH-UQ-2024]. No equation is evaluated.

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
# Least-squares fix and its linearization
# --------------------------------------------------------------------------


def solve_fix(
    centres: Sequence[Vector3],
    observed: Sequence[float],
    start: Sequence[float] = (0.0, 0.0, 0.0),
) -> Any:
    """Return the least-squares position fix from pseudo-ranges.

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


def geometry_matrix(centres: Sequence[Vector3], receiver: Vector3) -> list[list[float]]:
    """Return the unit lines of sight ``g_k = (R - S_k) / ||R - S_k||``.

    References:
        [REF-SEBER-WILD-2003, Section 2.1]. No equation is evaluated.

    """
    rows = []
    for centre in centres:
        size = distance(receiver, centre)
        rows.append([(receiver[i] - centre[i]) / size for i in range(3)])
    return rows


def _inverse3(m: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return the inverse of a 3 x 3 matrix by cofactors."""
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ]


def error_map(g: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return ``A = (G'G)^-1 G'`` (3 x k), range errors to position errors.

    References:
        [REF-SEBER-WILD-2003, Section 2.1]. No equation is evaluated.

    """
    k = len(g)
    gram = [[sum(g[n][i] * g[n][j] for n in range(k)) for j in range(3)] for i in range(3)]
    inverse = _inverse3(gram)
    return [[sum(inverse[i][j] * g[n][j] for j in range(3)) for n in range(k)] for i in range(3)]


def pdop(centres: Sequence[Vector3], receiver: Vector3 = (0.0, 0.0, 0.0)) -> float:
    """Return the position dilution of precision ``sqrt(trace((G'G)^-1))``.

    References:
        [REF-SEBER-WILD-2003, Section 2.1]. No equation is evaluated.

    """
    g = geometry_matrix(centres, receiver)
    gram = [[sum(row[i] * row[j] for row in g) for j in range(3)] for i in range(3)]
    inverse = _inverse3(gram)
    return math.sqrt(inverse[0][0] + inverse[1][1] + inverse[2][2])


def source_variances() -> dict[str, float]:
    """Return the per-reading variance each source contributes, in m^2.

    ``(c a)^2 / 3`` for the clock (shared by all readings), ``r^2 / 5`` for
    the line-of-sight component of a point uniform in a ball of radius
    ``r``, and ``h^2 / 3`` for a range error uniform on ``[-h, h]``.

    References:
        [REF-SMITH-UQ-2024]. No equation is evaluated.

    """
    return {
        "clock": (SPEED_OF_LIGHT * CLOCK_HALF_WIDTH) ** 2 / 3.0,
        "satellite": SATELLITE_RADIUS**2 / 5.0,
        "range": RANGE_HALF_WIDTH**2 / 3.0,
    }


def linearized_covariance(
    centres: Sequence[Vector3],
    receiver: Vector3 = (0.0, 0.0, 0.0),
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
        [REF-SEBER-WILD-2003, Section 2.1].

    """
    variances = source_variances()
    independent = sum(variances[name] for name in ("satellite", "range") if name in active)
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
    receiver: Vector3 = (0.0, 0.0, 0.0)
    errors = []
    for _ in range(draws):
        observed = readings(centres, receiver, draw_sources(rng, len(centres), active))
        fix = solve_fix(centres, observed).x
        errors.append((fix[0], fix[1], fix[2]))
    norms = [math.sqrt(e[0] ** 2 + e[1] ** 2 + e[2] ** 2) for e in errors]
    squares = [n * n for n in norms]
    size = draws // BATCHES
    batches = [norms[b * size : (b + 1) * size] for b in range(BATCHES)]
    mean_square = math.fsum(squares) / draws
    spread = math.fsum((s - mean_square) ** 2 for s in squares) / (draws - 1)
    return FixStudy(
        tuple(errors),
        quantile(norms, LEVEL),
        _batch_se([quantile(batch, LEVEL) for batch in batches]),
        mean_square,
        math.sqrt(spread / draws),
    )


def sample_covariance(points: Sequence[Vector3]) -> list[list[float]]:
    """Return the unbiased sample covariance of three-dimensional points.

    References:
        [REF-OWEN-2013, Chapter 2]. No equation is evaluated.

    """
    n = len(points)
    means = [math.fsum(p[i] for p in points) / n for i in range(3)]
    return [
        [
            math.fsum((p[i] - means[i]) * (p[j] - means[j]) for p in points) / (n - 1)
            for j in range(3)
        ]
        for i in range(3)
    ]


# --------------------------------------------------------------------------
# Posterior density of R by importance sampling (the course formula)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PosteriorStudy:
    """Self-normalized importance-sampling summary of ``f_R``.

    Attributes:
        observed: The pseudo-ranges conditioned on.
        fix: Least-squares fix from the same readings.
        points: Proposals with positive weight.
        weights: Their normalized weights.
        mean: Posterior mean ``E[R]``.
        radius: 90% radius around ``E[R]``.
        radius_se: Batch standard error of ``radius``.
        ess: Effective sample size of the weights.
        positive_share: Share of proposals with positive weight.
        edge_hits: Weighted proposals within 2% of the box edge.
        box: Half-widths of the uniform proposal box.

    """

    observed: tuple[float, ...]
    fix: Vector3
    points: tuple[Vector3, ...]
    weights: tuple[float, ...]
    mean: Vector3
    radius: float
    radius_se: float
    ess: float
    positive_share: float
    edge_hits: int
    box: Vector3


def _weighted_quantile(values: Sequence[float], weights: Sequence[float], level: float) -> float:
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
    mean = tuple(math.fsum(w * p[i] for w, p in zip(weights, points, strict=True)) / total for i in range(3))
    radii = [distance(p, mean) for p in points]
    return (mean[0], mean[1], mean[2]), _weighted_quantile(radii, weights, LEVEL)


def study_posterior(
    centres: Sequence[Vector3],
    seed: int,
    proposals: int = PROPOSALS,
    inner_draws: int = INNER_DRAWS,
) -> PosteriorStudy:
    """Estimate ``f_R``, ``E[R]``, and the 90% radius from one set of readings.

    The readings come from one draw of the three sources at ``R0``. The
    proposal is uniform on a box around the least-squares fix whose
    half-widths ``2 (h + c a + r) sum_k |A_ik|`` contain, to first order,
    every position compatible with the readings. The weight of a proposal
    is the inner average over ``inner_draws`` common draws of ``(tau, S)``
    of ``prod_k f_e(D_k - c tau - ||r - S_k||)``.

    Args:
        centres: Broadcast satellite centres.
        seed: Seed of the readings, the inner draws, and the proposals.
        proposals: Number of proposals.
        inner_draws: Number of common inner draws.

    Returns:
        The importance-sampling summary.

    References:
        [REF-OWEN-2013, Chapter 9], [REF-DOUCET-JOHANSEN-2011].

    """
    rng = random.Random(seed)
    k = len(centres)
    observed = readings(centres, (0.0, 0.0, 0.0), draw_sources(rng, k))
    fit = solve_fix(centres, observed).x
    fix: Vector3 = (fit[0], fit[1], fit[2])
    a = error_map(geometry_matrix(centres, fix))
    bound = 2.0 * (RANGE_HALF_WIDTH + SPEED_OF_LIGHT * CLOCK_HALF_WIDTH + SATELLITE_RADIUS)
    box = tuple(bound * sum(abs(v) for v in a[i]) for i in range(3))
    inner = []
    for _ in range(inner_draws):
        source = draw_sources(rng, k)
        moved = tuple(
            tuple(c + o for c, o in zip(centre, offset, strict=True))
            for centre, offset in zip(centres, source.offsets, strict=True)
        )
        inner.append((source.clock_range, moved))
    density = 1.0 / (2.0 * RANGE_HALF_WIDTH)

    def sample(
        count: int, half: Sequence[float]
    ) -> tuple[list[Vector3], list[float], int]:
        """Draw uniform proposals in a box and keep those with positive weight."""
        kept: list[Vector3] = []
        kept_weights: list[float] = []
        hits = 0
        for _ in range(count):
            u = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
            r = (fix[0] + half[0] * u[0], fix[1] + half[1] * u[1], fix[2] + half[2] * u[2])
            accepted = 0
            for clock, moved in inner:
                for satellite, d in zip(moved, observed, strict=True):
                    if abs(d - clock - distance(r, satellite)) > RANGE_HALF_WIDTH:
                        break
                else:
                    accepted += 1
            if accepted:
                kept.append(r)
                kept_weights.append(accepted * density**k / inner_draws)
                if max(abs(v) for v in u) > 0.98:
                    hits += 1
        return kept, kept_weights, hits

    # Pilot stage on the first-order box; the main stage uses the box that
    # bounds the pilot's support, widened by half, when the pilot kept at
    # least PILOT_MINIMUM points (it never grows past the first-order box).
    # Only the main stage enters the estimates, so the weights stay those of
    # one uniform proposal.
    pilot, _, _ = sample(proposals // 4, box)
    if len(pilot) >= PILOT_MINIMUM:
        box = tuple(
            min(box[i], 1.5 * max(abs(p[i] - fix[i]) for p in pilot)) for i in range(3)
        )
    points, weights, edge_hits = sample(proposals - proposals // 4, box)
    proposals -= proposals // 4
    total = math.fsum(weights)
    normalized = tuple(w / total for w in weights)
    mean, radius = _weighted_radius(points, normalized)
    size = len(points) // BATCHES
    # Proposals are i.i.d., so batches of consecutive accepted points are
    # independent replicates of the estimator.
    batch_radii = [
        _weighted_radius(points[b * size : (b + 1) * size], normalized[b * size : (b + 1) * size])[1]
        for b in range(BATCHES)
    ]
    return PosteriorStudy(
        observed,
        fix,
        tuple(points),
        normalized,
        mean,
        radius,
        _batch_se(batch_radii),
        effective_sample_size(normalized),
        len(points) / proposals,
        edge_hits,
        (box[0], box[1], box[2]),
    )


# --------------------------------------------------------------------------
# Whole analysis
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GeometryPoint:
    """Dilution of precision and simulated 90% radius of one constellation."""

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
    seconds: float


def _covariance_se(points: Sequence[Vector3]) -> list[list[float]]:
    """Return the standard error of each sample-covariance entry."""
    n = len(points)
    means = [math.fsum(p[i] for p in points) / n for i in range(3)]
    out = []
    for i in range(3):
        row = []
        for j in range(3):
            products = [(p[i] - means[i]) * (p[j] - means[j]) for p in points]
            centre = math.fsum(products) / n
            spread = math.fsum((v - centre) ** 2 for v in products) / (n - 1)
            row.append(math.sqrt(spread / n))
        out.append(row)
    return out


def _linear_radius(covariance: Sequence[Sequence[float]], seed: int) -> float:
    """Return the 90% radius of a centred normal law with this covariance."""
    l11 = math.sqrt(covariance[0][0])
    l21 = covariance[1][0] / l11
    l22 = math.sqrt(covariance[1][1] - l21 * l21)
    l31 = covariance[2][0] / l11
    l32 = (covariance[2][1] - l31 * l21) / l22
    l33 = math.sqrt(covariance[2][2] - l31 * l31 - l32 * l32)
    rng = random.Random(seed)
    norms = []
    for _ in range(20000):
        z = (rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0))
        x = l11 * z[0]
        y = l21 * z[0] + l22 * z[1]
        w = l31 * z[0] + l32 * z[1] + l33 * z[2]
        norms.append(math.sqrt(x * x + y * y + w * w))
    return quantile(norms, LEVEL)


def run_analysis(seed: int = SEED, *, fix_draws: int = FIX_DRAWS) -> LabResults:
    """Run every part of the laboratory.

    Args:
        seed: Base seed; each part uses its own offset stream.
        fix_draws: Draws of each least-squares scenario.

    Returns:
        The laboratory results.

    References:
        [REF-SMITH-UQ-2024], [REF-OWEN-2013].

    """
    started = time.perf_counter()
    centres = constellation(GOOD_SKY)
    exact = readings(centres, (0.0, 0.0, 0.0), draw_sources(random.Random(seed), 5, ()))
    noise_free = solve_fix(centres, exact, start=(3000.0, -4000.0, 1500.0)).x
    posterior = study_posterior(centres, seed + 1)
    fix_all = study_fix(centres, fix_draws, seed + 2)
    by_source = {
        name: study_fix(centres, fix_draws, seed + 3 + index, (name,))
        for index, name in enumerate(SOURCES)
    }
    linear_by_source = {
        name: sum(linearized_covariance(centres, active=(name,))[i][i] for i in range(3))
        for name in SOURCES
    }
    no_range = study_fix(centres, fix_draws, seed + 7, ("clock", "satellite"))
    geometries = []
    rng = random.Random(seed + 8)
    skies = [("well spread", GOOD_SKY), ("clustered", POOR_SKY)]
    for index in range(RANDOM_GEOMETRIES):
        sky = tuple((rng.uniform(0.0, 360.0), rng.uniform(15.0, 80.0)) for _ in range(5))
        skies.append((f"random {index + 1}", sky))
    for index, (label, sky) in enumerate(skies):
        sats = constellation(sky)
        study = study_fix(sats, GEOMETRY_DRAWS, seed + 20 + index)
        geometries.append(
            GeometryPoint(
                label,
                pdop(sats),
                study.radius,
                study.radius_se,
                _linear_radius(linearized_covariance(sats), seed + 60 + index),
            )
        )
    return LabResults(
        centres,
        distance(noise_free, (0.0, 0.0, 0.0)),
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
        time.perf_counter() - started,
    )


def gauss_newton_check(centres: Sequence[Vector3]) -> float:
    """Return the largest gap between the library and the lab covariance.

    The Gauss-Newton covariance ``s^2 (J'J)^-1`` of a noise-free fix, with
    ``s^2`` the independent per-reading variance, must equal the lab's
    ``A Sigma_e A'`` without the clock term.

    References:
        [REF-SEBER-WILD-2003, Section 2.1].

    """
    exact = readings(centres, (0.0, 0.0, 0.0), draw_sources(random.Random(0), 5, ()))
    result = solve_fix(centres, exact)
    variances = source_variances()
    library = gauss_newton_covariance(
        result, noise_variance=variances["satellite"] + variances["range"]
    )
    lab = linearized_covariance(centres, active=("satellite", "range"))
    return max(abs(library[i][j] - lab[i][j]) for i in range(3) for j in range(3))


def print_report(results: LabResults) -> None:
    """Print the laboratory's numbers with their Monte Carlo errors.

    References:
        [REF-SMITH-UQ-2024].

    """
    post = results.posterior
    print("Satellites (East, North, Up) [km] and range to the receiver [km]:")
    for centre in results.centres:
        print(
            f"  ({centre[0] / 1e3:9.1f}, {centre[1] / 1e3:9.1f}, {centre[2] / 1e3:9.1f})"
            f"  {distance(centre, (0.0, 0.0, 0.0)) / 1e3:8.1f}"
        )
    print(f"PDOP of the well-spread sky: {pdop(results.centres):.2f}")
    print(f"\nNoise-free fix error: {results.noise_free_error:.2e} m")
    print(
        "Gauss-Newton covariance vs A Sigma_e A' (no clock), largest gap: "
        f"{gauss_newton_check(results.centres):.1e} m^2"
    )
    print("\nPosterior f_R by importance sampling (one set of readings):")
    print(f"  E[R] = ({post.mean[0]:.2f}, {post.mean[1]:.2f}, {post.mean[2]:.2f}) m")
    print(f"  least-squares fix = ({post.fix[0]:.2f}, {post.fix[1]:.2f}, {post.fix[2]:.2f}) m")
    print(f"  ||E[R] - R0|| = {distance(post.mean, (0.0, 0.0, 0.0)):.2f} m")
    print(f"  90% radius around E[R]: {post.radius:.2f} m (batch SE {post.radius_se:.2f})")
    print(
        f"  ESS {post.ess:.0f} of {PROPOSALS - PROPOSALS // 4} main-stage proposals; positive weight "
        f"{100 * post.positive_share:.1f}%; edge hits {post.edge_hits}"
    )
    fix = results.fix_all
    print(f"\nLeast-squares fix, {FIX_DRAWS} draws of all sources:")
    print(f"  90% radius of ||R_hat - R0||: {fix.radius:.2f} m (batch SE {fix.radius_se:.2f})")
    print(f"  mean squared error: {fix.mean_square:.2f} m^2 (SE {fix.mean_square_se:.2f})")
    print("  covariance entry: Monte Carlo (SE) vs linearized [m^2]")
    for i, a in enumerate("ENU"):
        for j in range(i, 3):
            print(
                f"    {a}{'ENU'[j]}: {results.mc_covariance[i][j]:7.2f} "
                f"({results.covariance_se[i][j]:.2f})  {results.linear_covariance[i][j]:7.2f}"
            )
    print("\nSensitivity split of the mean squared error [m^2]:")
    total = sum(results.linear_by_source.values())
    for name in SOURCES:
        study = results.fix_by_source[name]
        print(
            f"  {name:9s}: Monte Carlo {study.mean_square:8.3f} (SE {study.mean_square_se:.3f})"
            f"  linearized {results.linear_by_source[name]:8.3f}"
            f"  share {100 * results.linear_by_source[name] / total:5.1f}%"
        )
    added = sum(results.fix_by_source[name].mean_square for name in SOURCES)
    print(f"  sum of single-source runs {added:.2f} vs all-source run {fix.mean_square:.2f}")
    print(
        "  90% radius without the range error (clock and satellites only): "
        f"{results.no_range_radius:.2f} m (SE {results.no_range_radius_se:.2f})"
    )
    print("\nGeometry: PDOP and 90% radius (Monte Carlo, SE, linearized normal) [m]:")
    for point in results.geometries:
        print(
            f"  {point.label:12s} PDOP {point.pdop:5.2f}  {point.radius:6.2f} "
            f"({point.radius_se:.2f})  {point.linear_radius:6.2f}"
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
    _label(ax, 0.0, 0.0, "True position", palette.model, xytext=(8.0, -10.0))
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
    ax.plot(fix_norms, [(i + 1) / n for i in range(n)], color=palette.model, linewidth=theme.LINE_WIDTH)
    ax.axhline(LEVEL, color=palette.muted, linewidth=theme.REFERENCE_WIDTH, linestyle="--")
    _label(ax, post.radius, LEVEL, f"Posterior: {post.radius:.1f} m", palette.observed,
           xytext=(-6.0, 10.0), ha="right")
    fix = results.fix_all
    _label(ax, fix.radius, LEVEL, f"Least-squares fix: {fix.radius:.1f} m", palette.model,
           xytext=(6.0, -12.0))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0.0, max(fix_norms[-1], xs[-1]) * 1.02)
    theme.axis_labels(ax, "Distance from centre", "Probability within", x_unit="m")
    theme.title_block(
        ax,
        f"90% radius: {post.radius:.1f} m for the posterior, {fix.radius:.1f} m for the fix",
        "Posterior: ||R - E[R]|| for one set of readings; fix: ||R_hat - R0|| over draws",
    )
    return figure


def figure_sensitivity(results: LabResults) -> Any:
    """Draw the mean squared position error contributed by each source.

    References:
        [REF-MATPLOTLIB].

    """
    palette = theme.active_palette()
    figure, ax = _axes((7.2, 3.6))
    names = list(SOURCES)
    labels = {"clock": "Clock shift", "satellite": "Satellite position", "range": "Range reading"}
    for row, name in enumerate(names):
        study = results.fix_by_source[name]
        low = max(study.mean_square - 2 * study.mean_square_se, 1e-6)
        ax.plot([low, study.mean_square + 2 * study.mean_square_se], [row, row],
                color=palette.observed, linewidth=3.0, solid_capstyle="butt")
        ax.plot([study.mean_square], [row], marker="o", markersize=7, color=palette.observed)
        ax.plot([results.linear_by_source[name]], [row], marker="|", markersize=16,
                mew=2.0, color=palette.model)
    ax.set_xscale("log")
    ax.set_yticks(range(len(names)), [labels[n] for n in names])
    ax.set_ylim(-0.6, len(names) - 0.2)
    top = len(names) - 1
    _label(ax, results.fix_by_source[names[top]].mean_square, top,
           "Monte Carlo Â± 2 SE", palette.observed, xytext=(0.0, 12.0), ha="right", va="bottom")
    _label(ax, results.linear_by_source[names[top]], top,
           "Linearized", palette.model, xytext=(0.0, -14.0), ha="center", va="top")
    from matplotlib.ticker import FuncFormatter, NullFormatter

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", color=palette.rule, linewidth=0.7)
    ax.grid(axis="y", visible=False)
    total = sum(results.linear_by_source.values())
    share = 100 * results.linear_by_source["range"] / total
    theme.axis_labels(ax, "Mean squared position error", None, x_unit="mÂ²")
    theme.title_block(
        ax,
        f"The range reading carries {share:.0f}% of the position error",
        "One source switched on at a time; least-squares fix",
    )
    return figure


def figure_geometry(results: LabResults) -> Any:
    """Draw the 90% radius against the position dilution of precision.

    References:
        [REF-MATPLOTLIB].

    """
    palette = theme.active_palette()
    figure, ax = _axes()
    points = sorted(results.geometries, key=lambda p: p.pdop)
    slope = math.fsum(p.linear_radius * p.pdop for p in points) / math.fsum(p.pdop**2 for p in points)
    top = points[-1].pdop * 1.05
    ax.plot([0.0, top], [0.0, slope * top], color=palette.model, linewidth=theme.REFERENCE_WIDTH)
    for p in points:
        ax.plot([p.pdop, p.pdop], [p.radius - 2 * p.radius_se, p.radius + 2 * p.radius_se],
                color=palette.observed, linewidth=1.5)
    ax.plot([p.pdop for p in points], [p.radius for p in points], "o",
            color=palette.observed, markersize=6)
    for p in points:
        if p.label in {"well spread", "clustered"}:
            _label(ax, p.pdop, p.radius, p.label.capitalize(), palette.ink,
                   xytext=(8.0, -8.0) if p.label == "clustered" else (8.0, -10.0))
    _label(ax, top, slope * top, f"Linearized: {slope:.1f} m per PDOP unit",
           palette.model, xytext=(-4.0, 8.0), ha="right", va="bottom")
    ax.set_xlim(0.0, top * 1.02)
    ax.set_ylim(0.0, max(p.radius + 2 * p.radius_se for p in points) * 1.08)
    theme.axis_labels(ax, "Position dilution of precision (PDOP)", "90% radius", y_unit="m")
    theme.title_block(
        ax,
        "Error radius grows in proportion to the dilution of precision",
        f"Least-squares fix, {GEOMETRY_DRAWS} draws per constellation, bars Â± 2 SE",
    )
    return figure


FIGURES: tuple[tuple[str, Callable[[LabResults], Any]], ...] = (
    ("gps_posterior_cloud.png", figure_posterior_cloud),
    ("gps_radial_cdf.png", figure_radial_cdf),
    ("gps_sensitivity.png", figure_sensitivity),
    ("gps_geometry.png", figure_geometry),
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
    print("Propagating clock, satellite, and range uncertainty (under a minute)...")
    results = run_analysis()
    print_report(results)
    output = Path(__file__).parent / "generated"
    written = write_figures(results, output)
    post = results.posterior
    print(
        f"\nResult: E[R] lies {distance(post.mean, (0.0, 0.0, 0.0)):.1f} m from the true "
        f"position; 90% of f_R lies within {post.radius:.1f} m of E[R]."
    )
    folder = output.relative_to(Path(__file__).parent.parent)
    print(
        f"Wrote {len(written)} figures in {time.perf_counter() - started:.0f} s. "
        f"Next: open {folder / FIGURES[0][0]}"
    )


if __name__ == "__main__":
    main()
