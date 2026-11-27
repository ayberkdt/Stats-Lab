"""Short-term-encounter conjunction assessment: collision probability and its limits.

Operational collision avoidance screens a close approach of two space objects
with the *short-term encounter* model (Foster and Estes; Akella and Alfriend;
the NASA CARA handbook): during the encounter the relative motion is a straight
line at constant speed, and the position uncertainty of both objects is a
constant Gaussian. The relative position at the time of closest approach
(TCA) then has a Gaussian law in the plane perpendicular to the relative
velocity (the encounter or B-plane), and the probability of collision ``Pc`` is
the probability mass of that law inside the hard-body disc, whose radius is the
sum of the two object radii. All lengths are SI metres, speeds m/s, times s.

Contents:

- :func:`encounter_plane_geometry`: the 2-D frame, projected combined
  covariance and miss vector from 3-D relative state and covariances;
  :func:`plane_encounter` builds the same container from 2-D data.
- :func:`collision_probability`: ``Pc`` by the exact rotated quadrature
  (:func:`probstat.multivariate.gaussian_disc_probability`), by Foster-style
  polar numerical integration (:func:`foster_probability`, adaptive node
  doubling with a stated convergence flag), by Chan's series for the
  equal-variance case (:func:`chan_series_probability`, an independent
  oracle) and by the small-object approximation.
- :func:`dilution_analysis`: ``Pc`` as a function of a covariance scale factor,
  its maximum and the dilution flag; :func:`collision_probability_sensitivity`
  for derivatives with respect to the hard-body radius and the scale.
- :func:`pc_under_scale_uncertainty`: distribution of ``Pc`` when the scale is
  log-normal (Gauss-Legendre quadrature in the normal score, optional Monte
  Carlo quantiles).
- :func:`monte_carlo_collision_probability`: crude Monte Carlo from sampled 3-D
  relative positions with a Wilson interval, independent of the encounter-plane
  projection, for validation.
- :func:`conjunction_realism`: covariance-realism check of residuals with
  :func:`probstat.filtering.covariance_realism` and the range of ``Pc`` over
  the confidence interval of the covariance scale.

Assumptions and validity. Rectilinear relative motion with constant relative
speed, constant covariances, Gaussian errors, spherical hard bodies and no
correlation between the two objects' errors. These hold for high-speed
encounters of short duration; they fail for slow relative speeds (co-orbital
or low-speed encounters), for long-duration encounters and for strongly
non-Gaussian position errors, where the model must not be used. ``Pc`` here is
only as good as the covariances: realism of the covariance is checked, not
assumed (:func:`conjunction_realism`). Every case in the tests is synthetic;
no result is a statement about a real conjunction.

Experimental (added in 0.28).

References:
    [REF-FOSTER-ESTES-1992], [REF-NASA-CARA-HANDBOOK, Appendix N],
    [REF-CHAN-2008], [REF-AKELLA-ALFRIEND-2000].

"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

from probstat.filtering import CovarianceRealismResult, covariance_realism
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.inference.confidence import wilson_score_interval
from probstat.linalg import cholesky
from probstat.linalg.dense import as_vector, symmetrize
from probstat.multivariate import (
    gaussian_disc_probability,
    maximum_disc_probability,
    small_disc_probability,
)

PC_METHODS: tuple[str, ...] = ("exact", "foster", "chan", "small-object")

_MAX_SERIES_TERMS = 2_000_000
_DEFAULT_FOSTER_START = 16


@dataclass(frozen=True, slots=True)
class EncounterGeometry:
    """Encounter-plane description of a conjunction.

    Args:
        miss_vector: Mean of the relative position in the plane ``(x, y)`` in
            metres. For a geometry built from 3-D data ``y = 0`` and ``x`` is
            the miss distance.
        covariance: Symmetric positive-definite ``2 x 2`` covariance (m^2) of
            the relative position in the plane.
        hard_body_radius: Sum of the two object radii (m), positive.
        miss_distance: ``||miss_vector||`` in metres.
        relative_speed: ``||v||`` in m/s, or ``None`` for a geometry built
            directly from plane data.
        time_to_closest_approach: ``-(r . v) / (v . v)`` in seconds measured
            from the epoch of the supplied state (negative: the closest
            approach is in the past), or ``None``.
        along_track_offset: ``r . v / ||v||`` in metres, the component of the
            supplied relative position along the relative velocity that the
            projection removes (zero if the state is at TCA), or ``None``.
        basis: The rows ``(e1, e2, v_hat)`` of the right-handed frame in the
            input coordinates (``e1`` along the perpendicular miss vector,
            ``e3 = v_hat`` along the relative velocity), or ``None``.

    Equation:
        [EQ-AERO-PC-001].

    References:
        [REF-FOSTER-ESTES-1992], [REF-NASA-CARA-HANDBOOK, Appendix N].

    """

    miss_vector: tuple[float, float]
    covariance: tuple[tuple[float, float], tuple[float, float]]
    hard_body_radius: float
    miss_distance: float
    relative_speed: float | None
    time_to_closest_approach: float | None
    along_track_offset: float | None
    basis: (
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
        | None
    )


@dataclass(frozen=True, slots=True)
class ConjunctionProbability:
    """Probability of collision with an error statement.

    Args:
        probability: ``Pc`` in ``[0, 1]``.
        error_estimate: Absolute error estimate of ``probability`` (an
            estimate, not a bound, unless stated by the method).
        method: ``"exact"``, ``"foster"``, ``"chan"`` or ``"small-object"``.
        converged: Whether the stopping rule of the method was met.

    Equation:
        [EQ-AERO-PC-002].

    References:
        [REF-FOSTER-ESTES-1992], [REF-NASA-CARA-HANDBOOK, Appendix N].

    """

    probability: float
    error_estimate: float
    method: str
    converged: bool


@dataclass(frozen=True, slots=True)
class DilutionAnalysis:
    """Collision probability against a covariance scale factor.

    Args:
        nominal_scale: Operating scale ``c0`` (1 means the covariance as
            given).
        nominal_probability: ``Pc`` at ``c0``.
        dilution_possible: Whether the miss distance exceeds the hard-body
            radius (otherwise ``Pc`` is largest as the covariance shrinks and
            no finite maximizer exists).
        maximum_scale: Scale ``c*`` that maximises ``Pc``, or ``None``.
        maximum_probability: ``Pc`` at ``c*``, or ``None``.
        in_dilution_region: ``c0 > c*``: increasing the uncertainty from the
            operating point lowers ``Pc`` (the probability is *diluted*), so a
            small ``Pc`` there is not conservative.
        maximum_ratio: ``maximum_probability / nominal_probability`` or
            ``None`` (also ``None`` when the nominal probability is zero).
        curve: ``(scale, Pc)`` pairs on the requested scales.
        evaluations: Disc-probability evaluations spent on the maximum.

    Equation:
        [EQ-AERO-PC-005].

    References:
        [REF-ALFANO-2005B], [REF-NASA-CARA-HANDBOOK, Section M.4].

    """

    nominal_scale: float
    nominal_probability: float
    dilution_possible: bool
    maximum_scale: float | None
    maximum_probability: float | None
    in_dilution_region: bool
    maximum_ratio: float | None
    curve: tuple[tuple[float, float], ...]
    evaluations: int


@dataclass(frozen=True, slots=True)
class PcSensitivity:
    """Derivatives of ``Pc`` with respect to hard-body radius and covariance scale.

    Args:
        probability: ``Pc`` at the operating point.
        d_radius: ``dPc/dR`` in 1/m.
        d_log_scale: ``dPc / d ln c`` (dimensionless).
        radius_elasticity: ``d ln Pc / d ln R`` (``None`` if ``Pc`` is zero).
        scale_elasticity: ``d ln Pc / d ln c`` (``None`` if ``Pc`` is zero).

    Equation:
        [EQ-AERO-PC-006].

    References:
        [REF-NASA-CARA-HANDBOOK, Appendix N].

    """

    probability: float
    d_radius: float
    d_log_scale: float
    radius_elasticity: float | None
    scale_elasticity: float | None


@dataclass(frozen=True, slots=True)
class ScaleUncertaintyResult:
    """Distribution of ``Pc`` when the covariance scale is log-normal.

    Args:
        median_scale: Median ``c_med`` of the scale ``c = c_med exp(s Z)``.
        log_sigma: Standard deviation ``s`` of ``ln c``.
        nominal_probability: ``Pc`` at ``c = c_med``.
        mean_probability: ``E[Pc(c)]`` by Gauss-Legendre quadrature in ``Z``.
        std_probability: Standard deviation of ``Pc(c)`` (same quadrature).
        mean_error_estimate: Absolute difference of the mean between the
            two quadrature orders (an estimate, not a bound).
        converged: Whether ``mean_error_estimate`` is below ``1e-8`` of the
            mean plus ``1e-14``.
        quantile_levels: Requested quantile levels (empty without sampling).
        quantile_probabilities: Empirical quantiles of ``Pc`` over the
            sampled scales (empty without sampling).
        samples: Number of Monte Carlo scales (zero: no sampling).
        sample_mean: Monte Carlo mean of ``Pc`` (``None`` without sampling).
        sample_standard_error: Monte Carlo standard error of that mean.

    Equation:
        [EQ-AERO-PC-007].

    References:
        [REF-DAVIS-RABINOWITZ-1984], [REF-NASA-CARA-HANDBOOK, Appendix N].

    """

    median_scale: float
    log_sigma: float
    nominal_probability: float
    mean_probability: float
    std_probability: float
    mean_error_estimate: float
    converged: bool
    quantile_levels: tuple[float, ...]
    quantile_probabilities: tuple[float, ...]
    samples: int
    sample_mean: float | None
    sample_standard_error: float | None


@dataclass(frozen=True, slots=True)
class MonteCarloProbability:
    """Crude Monte Carlo collision probability with a Wilson interval.

    Args:
        probability: ``hits / samples``.
        standard_error: ``sqrt(p (1 - p) / samples)`` (zero when ``p`` is 0
            or 1; use the interval then).
        lower: Lower Wilson endpoint.
        upper: Upper Wilson endpoint.
        confidence: Nominal confidence level of the interval.
        hits: Samples whose straight-line path passes within the radius.
        samples: Number of samples.
        method: ``"crude-wilson"``.

    Equation:
        [EQ-AERO-PC-008].

    References:
        [REF-WILSON-1927].

    """

    probability: float
    standard_error: float
    lower: float
    upper: float
    confidence: float
    hits: int
    samples: int
    method: str


@dataclass(frozen=True, slots=True)
class ConjunctionRealism:
    """Covariance-realism verdict and its effect on ``Pc``.

    Args:
        realism: Result of :func:`probstat.filtering.covariance_realism`.
        nominal_probability: ``Pc`` with the covariance as given.
        scaled_probability: ``Pc`` with the covariance multiplied by the
            estimated scale factor.
        probability_range: ``(min, max)`` of ``Pc`` over the confidence
            interval of the scale factor (endpoints and a log-spaced scan).

    Equation:
        [EQ-AERO-PC-009].

    References:
        [REF-HORWOOD-ETAL-2014], [REF-NASA-CARA-HANDBOOK, Appendix I].

    """

    realism: CovarianceRealismResult
    nominal_probability: float
    scaled_probability: float
    probability_range: tuple[float, float]


# ---------------------------------------------------------------------------
# Validation and geometry
# ---------------------------------------------------------------------------


def _vector3(values: Iterable[float], name: str) -> tuple[float, float, float]:
    """Return a validated 3-vector.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the vector is not of length three or not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation.

    """
    vector = as_vector(values, name=name)
    if len(vector) != 3:
        raise ValueError(f"{name} must have three components")
    return (vector[0], vector[1], vector[2])


def _matrix3(
    values: Iterable[Iterable[float]], name: str
) -> tuple[tuple[float, ...], ...]:
    """Return a validated symmetric 3 x 3 matrix.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the matrix is not symmetric ``3 x 3``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation.

    """
    matrix = symmetrize(values, name=name)
    if len(matrix) != 3:
        raise ValueError(f"{name} must be 3 x 3")
    return matrix


def _positive(value: float, name: str) -> float:
    """Return a positive finite real.

    Raises:
        TypeError: If ``value`` is not real.
        ValueError: If it is not finite and positive.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation.

    """
    number = require_finite_real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the Euclidean inner product ``a . b``.

    References:
        [REF-GOLUB-VANLOAN-2013, Chapter 1]. No equation.

    """
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def _cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Return the cross product ``a x b``.

    References:
        [REF-GOLUB-VANLOAN-2013, Chapter 1]. No equation.

    """
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _unit(v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return ``v / ||v||`` (``v`` non-zero).

    References:
        [REF-GOLUB-VANLOAN-2013, Chapter 1]. No equation.

    """
    norm = math.sqrt(_dot(v, v))
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def _quadratic(
    matrix: tuple[tuple[float, ...], ...],
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    """Return the bilinear form ``a' M b``.

    References:
        [REF-GOLUB-VANLOAN-2013, Chapter 1]. No equation.

    """
    return math.fsum(a[i] * matrix[i][j] * b[j] for i in range(3) for j in range(3))


def _require_conditioned(matrix: tuple[tuple[float, ...], ...]) -> None:
    """Reject a ``2 x 2`` covariance whose condition number exceeds ``1e10``.

    Rounding can leave a rank-deficient projection with a tiny positive
    Cholesky pivot; this check refuses it rather than integrating a
    degenerate density.

    Raises:
        ValueError: If the smaller eigenvalue is below ``1e-10`` times the
            larger one.

    References:
        [REF-GOLUB-VANLOAN-2013, Chapter 2]. No equation.

    """
    mean = 0.5 * (matrix[0][0] + matrix[1][1])
    radius = math.hypot(0.5 * (matrix[0][0] - matrix[1][1]), matrix[0][1])
    if mean - radius <= 1.0e-10 * (mean + radius):
        raise ValueError("covariance is numerically singular")


def _validated_plane(
    miss_vector: Iterable[float],
    covariance: Iterable[Iterable[float]],
    radius: float,
) -> tuple[tuple[float, float], tuple[tuple[float, float], tuple[float, float]], float]:
    """Return validated plane data (miss, SPD covariance, radius).

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes are wrong, the covariance is not symmetric
            positive definite, or the radius is not positive.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation.

    """
    miss = as_vector(miss_vector, name="miss_vector")
    if len(miss) != 2:
        raise ValueError("miss_vector must have two components")
    matrix = symmetrize(covariance, name="covariance")
    if len(matrix) != 2:
        raise ValueError("covariance must be 2 x 2")
    try:
        cholesky(matrix)
    except ValueError as error:
        raise ValueError("covariance must be positive definite") from error
    _require_conditioned(matrix)
    radius_value = _positive(radius, "hard_body_radius")
    return (
        (miss[0], miss[1]),
        ((matrix[0][0], matrix[0][1]), (matrix[1][0], matrix[1][1])),
        radius_value,
    )


def plane_encounter(
    miss_vector: Iterable[float],
    covariance: Iterable[Iterable[float]],
    hard_body_radius: float,
) -> EncounterGeometry:
    """Build an :class:`EncounterGeometry` from data already in the plane.

    Args:
        miss_vector: Mean relative position ``(x, y)`` in the encounter plane
            (m).
        covariance: Symmetric positive-definite ``2 x 2`` covariance (m^2).
        hard_body_radius: Combined hard-body radius ``R`` (m), positive.

    Returns:
        A geometry whose three-dimensional fields are ``None``.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes are wrong, the covariance is not symmetric
            positive definite, or the radius is not positive.

    Equation:
        [EQ-AERO-PC-001].

    References:
        [REF-NASA-CARA-HANDBOOK, Appendix N].

    """
    miss, matrix, radius = _validated_plane(miss_vector, covariance, hard_body_radius)
    return EncounterGeometry(
        miss, matrix, radius, math.hypot(*miss), None, None, None, None
    )


def encounter_plane_geometry(
    relative_position: Iterable[float],
    relative_velocity: Iterable[float],
    covariance_primary: Iterable[Iterable[float]],
    covariance_secondary: Iterable[Iterable[float]],
    radius_primary: float,
    radius_secondary: float,
) -> EncounterGeometry:
    """Project a 3-D conjunction onto the encounter (B-) plane.

    With the relative state ``r = r_2 - r_1`` (m), ``v = v_2 - v_1`` (m/s) and
    the unit vector ``w = v / ||v||``, the straight-line path passes the
    primary at the perpendicular offset ``r_perp = r - (r . w) w``. The frame
    is ``e1 = r_perp / ||r_perp||`` (any perpendicular if the miss is zero),
    ``e2 = w x e1``; the miss vector is ``(||r_perp||, 0)`` and the plane
    covariance is ``[e1 e2]' (C_1 + C_2) [e1 e2]``. The hard-body radius is
    ``R = R_1 + R_2``. If the supplied ``r`` is not at TCA the component
    ``r . w`` along the path is removed (reported in ``along_track_offset``),
    which is exact for rectilinear motion.

    Args:
        relative_position: Relative position ``r`` (3 reals, m).
        relative_velocity: Relative velocity ``v`` (3 reals, m/s), non-zero.
        covariance_primary: Symmetric ``3 x 3`` position covariance of the
            primary (m^2), in the same coordinates.
        covariance_secondary: Same for the secondary.
        radius_primary: Primary hard-body radius (m), positive.
        radius_secondary: Secondary hard-body radius (m), positive.

    Returns:
        The encounter geometry.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If a vector is not three-dimensional, a covariance is not
            symmetric ``3 x 3``, the relative velocity is zero (the encounter
            plane is undefined), a radius is not positive, or the projected
            combined covariance is not positive definite (singular in the
            plane).

    Equation:
        [EQ-AERO-PC-001].

    References:
        [REF-FOSTER-ESTES-1992], [REF-NASA-CARA-HANDBOOK, Appendix N],
        [REF-AKELLA-ALFRIEND-2000].

    """
    r = _vector3(relative_position, "relative_position")
    v = _vector3(relative_velocity, "relative_velocity")
    c1 = _matrix3(covariance_primary, "covariance_primary")
    c2 = _matrix3(covariance_secondary, "covariance_secondary")
    radius = _positive(radius_primary, "radius_primary") + _positive(
        radius_secondary, "radius_secondary"
    )
    speed = math.sqrt(_dot(v, v))
    if speed <= 0.0 or not math.isfinite(speed):
        raise ValueError("relative_velocity must be non-zero and finite")
    w = _unit(v)
    along = _dot(r, w)
    perp = (r[0] - along * w[0], r[1] - along * w[1], r[2] - along * w[2])
    miss = math.sqrt(_dot(perp, perp))
    if miss > 0.0:
        e1 = _unit(perp)
    else:
        axis = [0.0, 0.0, 0.0]
        axis[min(range(3), key=lambda k: abs(w[k]))] = 1.0
        keep = _dot(axis, w)
        e1 = _unit(
            (axis[0] - keep * w[0], axis[1] - keep * w[1], axis[2] - keep * w[2])
        )
    e2 = _cross(w, e1)
    combined = tuple(tuple(c1[i][j] + c2[i][j] for j in range(3)) for i in range(3))
    plane = (
        (_quadratic(combined, e1, e1), _quadratic(combined, e1, e2)),
        (_quadratic(combined, e2, e1), _quadratic(combined, e2, e2)),
    )
    plane = (
        (plane[0][0], 0.5 * (plane[0][1] + plane[1][0])),
        (0.5 * (plane[0][1] + plane[1][0]), plane[1][1]),
    )
    try:
        cholesky(plane)
        _require_conditioned(plane)
    except ValueError as error:
        raise ValueError(
            "the combined covariance projected onto the encounter plane must be "
            "positive definite and not numerically singular"
        ) from error
    return EncounterGeometry(
        (miss, 0.0),
        plane,
        radius,
        miss,
        speed,
        -_dot(r, v) / (speed * speed),
        along,
        (e1, e2, w),
    )


# ---------------------------------------------------------------------------
# Collision probability
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _gauss_legendre(order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return Gauss-Legendre nodes and weights on ``[-1, 1]`` (Newton iteration).

    Args:
        order: Number of nodes, at least one.

    Returns:
        Ascending nodes and positive weights summing to two.

    References:
        [REF-DAVIS-RABINOWITZ-1984]. Used by [EQ-AERO-PC-002] and
        [EQ-AERO-PC-007].

    """
    nodes: list[float] = []
    weights: list[float] = []
    half = (order + 1) // 2
    for i in range(1, half + 1):
        x = math.cos(math.pi * (i - 0.25) / (order + 0.5))
        for _ in range(100):
            p_prev, p_cur = 1.0, x
            for k in range(2, order + 1):
                p_prev, p_cur = p_cur, ((2 * k - 1) * x * p_cur - (k - 1) * p_prev) / k
            derivative = order * (x * p_cur - p_prev) / (x * x - 1.0)
            step = p_cur / derivative
            x -= step
            if abs(step) < 1.0e-15:
                break
        p_prev, p_cur = 1.0, x
        for k in range(2, order + 1):
            p_prev, p_cur = p_cur, ((2 * k - 1) * x * p_cur - (k - 1) * p_prev) / k
        derivative = order * (x * p_cur - p_prev) / (x * x - 1.0)
        weight = 2.0 / ((1.0 - x * x) * derivative * derivative)
        if order % 2 == 1 and i == half:
            nodes.append(0.0)
            weights.append(weight)
        else:
            nodes.extend((-x, x))
            weights.extend((weight, weight))
    pairs = sorted(zip(nodes, weights, strict=True))
    return tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)


def foster_probability(
    miss_vector: Iterable[float],
    covariance: Iterable[Iterable[float]],
    hard_body_radius: float,
    *,
    tolerance: float = 1.0e-8,
    max_nodes: int = 256,
) -> ConjunctionProbability:
    """Integrate the planar Gaussian density over the disc in polar coordinates.

    ``Pc = int_0^R int_0^(2 pi) f(r cos t, r sin t) r dt dr`` with the bivariate
    normal density ``f`` of the relative position (mean ``miss_vector``): the
    Foster-Estes two-dimensional numerical integral. The radial integral is
    Gauss-Legendre with ``n`` nodes and the angular integral the periodic
    trapezoidal rule with ``2 n`` points (spectrally accurate for the smooth
    periodic integrand). ``n`` doubles from 16 until two successive values
    differ by at most ``tolerance`` times the value, or ``max_nodes`` is
    reached; the result is then flagged as not converged. The method resolves
    features down to about ``R / n``; it is unreliable when the covariance is
    much smaller than the disc and the miss is far outside it.

    Args:
        miss_vector: Mean ``(x, y)`` of the relative position in the plane (m).
        covariance: Symmetric positive-definite ``2 x 2`` covariance (m^2).
        hard_body_radius: Disc radius ``R`` (m), positive.
        tolerance: Relative difference of successive doublings, in ``(0, 1)``.
        max_nodes: Largest radial node count, at least 16.

    Returns:
        ``Pc`` with the last doubling difference as ``error_estimate``.

    Raises:
        TypeError: If an entry is not real or ``max_nodes`` is not an integer.
        ValueError: If shapes are wrong, the covariance is not symmetric
            positive definite, the radius is not positive, ``tolerance`` is
            outside ``(0, 1)`` or ``max_nodes`` is below 16.

    Equation:
        [EQ-AERO-PC-002].

    References:
        [REF-FOSTER-ESTES-1992], [REF-NASA-CARA-HANDBOOK, Appendix N],
        [REF-DAVIS-RABINOWITZ-1984].

    """
    miss, cov, radius = _validated_plane(miss_vector, covariance, hard_body_radius)
    tol = require_finite_real(tolerance, name="tolerance")
    if not 0.0 < tol < 1.0:
        raise ValueError("tolerance must lie in (0, 1)")
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int):
        raise TypeError("max_nodes must be an integer")
    if max_nodes < _DEFAULT_FOSTER_START:
        raise ValueError("max_nodes must be at least 16")
    (a, b), (_, d) = cov
    det = a * d - b * b
    p00, p01, p11 = d / det, -b / det, a / det
    norm = 1.0 / (2.0 * math.pi * math.sqrt(det))

    def level(order: int) -> float:
        """Return the polar quadrature with ``order`` radial nodes.

        References:
            [REF-FOSTER-ESTES-1992]. Evaluates [EQ-AERO-PC-002].

        """
        x, w = _gauss_legendre(order)
        count = 2 * order
        cos_t = [math.cos(2.0 * math.pi * j / count) for j in range(count)]
        sin_t = [math.sin(2.0 * math.pi * j / count) for j in range(count)]
        total = 0.0
        for xi, wi in zip(x, w, strict=True):
            r = 0.5 * radius * (xi + 1.0)
            ring = 0.0
            for c, s in zip(cos_t, sin_t, strict=True):
                dx = r * c - miss[0]
                dy = r * s - miss[1]
                ring += math.exp(
                    -0.5 * (p00 * dx * dx + 2.0 * p01 * dx * dy + p11 * dy * dy)
                )
            total += 0.5 * radius * wi * r * ring * (2.0 * math.pi / count)
        return norm * total

    order = _DEFAULT_FOSTER_START
    previous = level(order)
    while True:
        order *= 2
        current = level(order)
        difference = abs(current - previous)
        if difference <= tol * current:
            return ConjunctionProbability(min(1.0, current), difference, "foster", True)
        if order >= max_nodes:
            return ConjunctionProbability(
                min(1.0, current), difference, "foster", False
            )
        previous = current


def chan_series_probability(
    miss_distance: float, sigma: float, hard_body_radius: float
) -> ConjunctionProbability:
    """Return ``Pc`` for an isotropic planar covariance by Chan's series.

    For ``X ~ N(mu, sigma^2 I_2)`` with ``||mu|| = d``, ``||X||^2 / sigma^2``
    is noncentral chi-square with two degrees of freedom, a Poisson mixture
    of central ones, which gives the series

    ``Pc = sum_m e^(-v) v^m / m! * (1 - e^(-a) sum_(k<=m) a^k / k!)``,
    ``v = d^2 / (2 sigma^2)``, ``a = R^2 / (2 sigma^2)``,

    the equal-variance form of Chan's series. The outer weights are summed
    over the window ``v +/- (40 sqrt(v) + 40)`` (neglected mass below
    ``1e-300``) and the inner tails by a reverse cumulative sum of Poisson
    probabilities, so a small ``Pc`` keeps its relative accuracy. This routine
    shares no code with :func:`probstat.multivariate.gaussian_disc_probability`
    and serves as its independent check.

    Args:
        miss_distance: Miss distance ``d`` (m), non-negative.
        sigma: Common standard deviation (m), positive.
        hard_body_radius: Disc radius ``R`` (m), positive.

    Returns:
        ``Pc``; ``error_estimate`` is a rounding-scale allowance
        ``sqrt(terms) * 2.2e-16`` (an estimate), ``converged`` is true.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If ``miss_distance`` is negative, ``sigma`` or the radius
            is not positive, or the series window exceeds two million terms
            (``R / sigma`` or ``d / sigma`` too large for this routine).

    Equation:
        [EQ-AERO-PC-003].

    References:
        [REF-CHAN-2008], [REF-NASA-CARA-HANDBOOK, Appendix N].

    """
    distance = require_finite_real(miss_distance, name="miss_distance")
    if distance < 0.0:
        raise ValueError("miss_distance must be non-negative")
    s = _positive(sigma, "sigma")
    radius = _positive(hard_body_radius, "hard_body_radius")
    lam = distance * distance / (2.0 * s * s)
    a = radius * radius / (2.0 * s * s)
    if lam == 0.0:
        low = high = 0
    else:
        spread = 40.0 * math.sqrt(lam) + 40.0
        low = max(0, math.floor(lam - spread))
        high = math.ceil(lam + spread)
    kmax = max(high + 1, math.ceil(a + 40.0 * math.sqrt(a) + 40.0))
    if kmax - low > _MAX_SERIES_TERMS:
        raise ValueError("the series window is too long for this routine")
    log_a = math.log(a)
    # tail[k - low] = sum_{j = k + 1}^{kmax} Poisson(j; a), for k in [low, kmax].
    tail = [0.0] * (kmax - low + 1)
    running = 0.0
    for k in range(kmax, low, -1):
        running += math.exp(-a + k * log_a - math.lgamma(k + 1.0))
        tail[k - 1 - low] = running
    tail[kmax - low] = 0.0
    terms = []
    for m in range(low, high + 1):
        if lam == 0.0:
            weight = 1.0
        else:
            weight = math.exp(-lam + m * math.log(lam) - math.lgamma(m + 1.0))
        terms.append(weight * tail[m - low])
    probability = min(1.0, math.fsum(terms))
    return ConjunctionProbability(
        probability, math.sqrt(len(terms)) * 2.2e-16, "chan", True
    )


def _plane_probability(
    miss: tuple[float, float],
    cov: tuple[tuple[float, float], tuple[float, float]],
    radius: float,
    method: str,
    tolerance: float,
) -> ConjunctionProbability:
    """Evaluate ``Pc`` for validated plane data with the chosen method.

    Raises:
        ValueError: For an unknown method or a non-isotropic covariance with
            ``method="chan"``.

    References:
        [REF-NASA-CARA-HANDBOOK, Appendix N]. Evaluates [EQ-AERO-PC-002].

    """
    if method == "exact":
        result = gaussian_disc_probability(miss, cov, radius, tolerance=tolerance)
        return ConjunctionProbability(
            min(1.0, max(0.0, result.probability)),
            result.error_estimate,
            "exact",
            result.converged,
        )
    if method == "foster":
        return foster_probability(miss, cov, radius)
    if method == "chan":
        scale = max(abs(cov[0][0]), abs(cov[1][1]))
        if (
            abs(cov[0][0] - cov[1][1]) > 1.0e-9 * scale
            or abs(cov[0][1]) > 1.0e-9 * scale
        ):
            raise ValueError(
                "method 'chan' needs an isotropic covariance (equal variances, "
                "no correlation)"
            )
        return chan_series_probability(
            math.hypot(*miss), math.sqrt(0.5 * (cov[0][0] + cov[1][1])), radius
        )
    if method == "small-object":
        result = small_disc_probability(miss, cov, radius)
        return ConjunctionProbability(
            min(1.0, result.probability), result.error_estimate, "small-object", True
        )
    raise ValueError(f"method must be one of {PC_METHODS}")


def collision_probability(
    geometry: EncounterGeometry,
    *,
    method: str = "exact",
    tolerance: float = 1.0e-11,
) -> ConjunctionProbability:
    """Return the probability of collision of an encounter geometry.

    ``Pc = int_(||x|| <= R) N(x; m, P) dx`` for the planar miss vector ``m``,
    plane covariance ``P`` and combined hard-body radius ``R``.

    Methods: ``"exact"`` (rotated one-dimensional adaptive quadrature,
    :func:`probstat.multivariate.gaussian_disc_probability`), ``"foster"``
    (polar numerical integration, :func:`foster_probability`), ``"chan"``
    (series for an isotropic covariance only, :func:`chan_series_probability`)
    and ``"small-object"`` (``pi R^2 f(0)``, accurate only for ``R`` small
    against the covariance length scale; its first neglected term is
    returned as the error estimate).

    Args:
        geometry: Encounter geometry from :func:`encounter_plane_geometry` or
            :func:`plane_encounter`.
        method: One of :data:`PC_METHODS`.
        tolerance: Quadrature tolerance of ``"exact"`` (relative, in
            ``(0, 1)``).

    Returns:
        ``Pc`` with an error statement and a convergence flag.

    Raises:
        TypeError: If ``geometry`` is not an :class:`EncounterGeometry`.
        ValueError: If ``method`` is unknown, ``tolerance`` is outside
            ``(0, 1)``, or ``"chan"`` is requested for a non-isotropic
            covariance.

    Equation:
        [EQ-AERO-PC-002], [EQ-AERO-PC-003], [EQ-AERO-PC-004].

    References:
        [REF-FOSTER-ESTES-1992], [REF-CHAN-2008],
        [REF-NASA-CARA-HANDBOOK, Appendix N].

    """
    if not isinstance(geometry, EncounterGeometry):
        raise TypeError("geometry must be an EncounterGeometry")
    return _plane_probability(
        geometry.miss_vector,
        geometry.covariance,
        geometry.hard_body_radius,
        method,
        tolerance,
    )


# ---------------------------------------------------------------------------
# Dilution, sensitivity, uncertain scale
# ---------------------------------------------------------------------------


def _scaled(geometry: EncounterGeometry, scale: float) -> EncounterGeometry:
    """Return ``geometry`` with its covariance multiplied by ``scale``.

    References:
        [REF-ALFANO-2005B]. Used by [EQ-AERO-PC-005].

    """
    c = geometry.covariance
    return replace(
        geometry,
        covariance=(
            (scale * c[0][0], scale * c[0][1]),
            (scale * c[1][0], scale * c[1][1]),
        ),
    )


def _exact_pc(geometry: EncounterGeometry, scale: float = 1.0) -> float:
    """Return the exact ``Pc`` of ``geometry`` with a scaled covariance.

    References:
        [REF-NASA-CARA-HANDBOOK, Appendix N]. Evaluates [EQ-AERO-PC-002].

    """
    scaled = _scaled(geometry, scale)
    return collision_probability(scaled).probability


def dilution_analysis(
    geometry: EncounterGeometry,
    *,
    scale: float = 1.0,
    scales: Iterable[float] = (),
) -> DilutionAnalysis:
    """Analyse ``Pc`` as a function of a covariance scale factor.

    For ``X ~ N(m, c P)`` the probability ``p(c)`` of the hard-body disc
    tends to zero as ``c -> 0`` and as ``c -> infinity`` when the miss
    distance exceeds ``R``, so it has a maximum at ``c*``
    (:func:`probstat.multivariate.maximum_disc_probability`, scan plus
    golden-section search). The operating scale ``c0`` lies in the *dilution
    region* when ``c0 > c*``: more uncertainty then lowers ``Pc``, and a small
    ``Pc`` is not a conservative statement about the miss. The maximum is
    the largest ``Pc`` attainable by changing only the covariance scale.

    Args:
        geometry: Encounter geometry.
        scale: Operating scale ``c0`` (positive).
        scales: Extra scales at which to tabulate ``Pc`` (positive).

    Returns:
        The analysis; ``maximum_scale`` and ``maximum_probability`` are
        ``None`` when the miss distance does not exceed the radius (then
        ``Pc`` increases as the covariance shrinks and has no finite
        maximizer, so ``in_dilution_region`` is false).

    Raises:
        TypeError: If ``geometry`` is not an :class:`EncounterGeometry`.
        ValueError: If a scale is not positive, or the maximum lies outside
            the scanned range of the underlying search.

    Equation:
        [EQ-AERO-PC-005].

    References:
        [REF-ALFANO-2005B], [REF-NASA-CARA-HANDBOOK, Section M.4].

    """
    if not isinstance(geometry, EncounterGeometry):
        raise TypeError("geometry must be an EncounterGeometry")
    c0 = _positive(scale, "scale")
    grid = tuple(_positive(s, "scales entry") for s in scales)
    nominal = _exact_pc(geometry, c0)
    curve = tuple((s, _exact_pc(geometry, s)) for s in grid)
    if geometry.miss_distance <= geometry.hard_body_radius:
        return DilutionAnalysis(c0, nominal, False, None, None, False, None, curve, 0)
    best = maximum_disc_probability(
        geometry.miss_vector, geometry.covariance, geometry.hard_body_radius
    )
    ratio = best.probability / nominal if nominal > 0.0 else None
    return DilutionAnalysis(
        c0,
        nominal,
        True,
        best.scale,
        best.probability,
        c0 > best.scale,
        ratio,
        curve,
        best.evaluations,
    )


def collision_probability_sensitivity(
    geometry: EncounterGeometry, *, relative_step: float = 1.0e-3
) -> PcSensitivity:
    """Return ``dPc/dR`` and ``dPc/d ln c`` by Richardson-extrapolated differences.

    Central differences with steps ``h`` and ``h / 2`` (relative to ``R`` and
    in ``ln c``) are combined as ``(4 D(h/2) - D(h)) / 3`` (error ``O(h^4)``).
    The exact quadrature underlies every evaluation. For a small object
    ``Pc`` is proportional to ``R^2`` and the radius elasticity tends to 2.

    Args:
        geometry: Encounter geometry.
        relative_step: Step ``h``, in ``(0, 0.1]``.

    Returns:
        Derivatives and elasticities (``None`` when ``Pc`` is zero).

    Raises:
        TypeError: If ``geometry`` is not an :class:`EncounterGeometry`.
        ValueError: If ``relative_step`` is outside ``(0, 0.1]``.

    Equation:
        [EQ-AERO-PC-006].

    References:
        [REF-NASA-CARA-HANDBOOK, Appendix N],
        [REF-GOLUB-VANLOAN-2013, Chapter 1].

    """
    if not isinstance(geometry, EncounterGeometry):
        raise TypeError("geometry must be an EncounterGeometry")
    h = require_finite_real(relative_step, name="relative_step")
    if not 0.0 < h <= 0.1:
        raise ValueError("relative_step must lie in (0, 0.1]")
    radius = geometry.hard_body_radius
    base = _exact_pc(geometry)

    def radius_difference(step: float) -> float:
        """Return the central difference in ``R`` with relative step ``step``.

        References:
            [REF-NASA-CARA-HANDBOOK, Appendix N]. Part of [EQ-AERO-PC-006].

        """
        delta = step * radius
        up = _exact_pc(replace(geometry, hard_body_radius=radius + delta))
        down = _exact_pc(replace(geometry, hard_body_radius=radius - delta))
        return (up - down) / (2.0 * delta)

    def scale_difference(step: float) -> float:
        """Return the central difference in ``ln c`` with step ``step``.

        References:
            [REF-NASA-CARA-HANDBOOK, Appendix N]. Part of [EQ-AERO-PC-006].

        """
        up = _exact_pc(geometry, math.exp(step))
        down = _exact_pc(geometry, math.exp(-step))
        return (up - down) / (2.0 * step)

    d_radius = (4.0 * radius_difference(h / 2.0) - radius_difference(h)) / 3.0
    d_scale = (4.0 * scale_difference(h / 2.0) - scale_difference(h)) / 3.0
    if base > 0.0:
        radius_elasticity: float | None = radius * d_radius / base
        scale_elasticity: float | None = d_scale / base
    else:
        radius_elasticity = scale_elasticity = None
    return PcSensitivity(base, d_radius, d_scale, radius_elasticity, scale_elasticity)


def pc_under_scale_uncertainty(
    geometry: EncounterGeometry,
    log_sigma: float,
    *,
    median_scale: float = 1.0,
    nodes: int = 64,
    samples: int = 0,
    quantile_levels: Iterable[float] = (0.05, 0.5, 0.95),
    seed: int | RandomStream | None = None,
) -> ScaleUncertaintyResult:
    """Return the law of ``Pc`` when the covariance scale is log-normal.

    The scale is ``c = c_med exp(s Z)`` with ``Z ~ N(0, 1)`` and
    ``s = log_sigma``; ``Pc(c)`` is the disc probability for the covariance
    ``c P``. The mean and second moment of ``Pc(c)`` are Gauss-Legendre
    integrals of ``Pc(c_med e^(s z)) phi(z)`` over ``[-9, 9]`` (the neglected
    normal mass is about ``1e-19``) with ``nodes`` and ``2 nodes`` points, whose
    difference is the error estimate. Because ``Pc(c)`` is not monotone in
    ``c`` (dilution), quantiles of ``Pc`` are not images of scale quantiles;
    they come from ``samples`` Monte Carlo scales when requested.

    Args:
        geometry: Encounter geometry.
        log_sigma: Standard deviation of ``ln c`` (positive).
        median_scale: Median scale ``c_med`` (positive).
        nodes: Gauss-Legendre order (at least 8); the check uses ``2 nodes``.
        samples: Monte Carlo scales for quantiles (``0``: none).
        quantile_levels: Levels in ``(0, 1)``.
        seed: Seed or stream of the sampling.

    Returns:
        Mean, standard deviation, error estimate and optional quantiles.

    Raises:
        TypeError: If ``geometry`` is not an :class:`EncounterGeometry` or an
            integer argument is not an integer.
        ValueError: If ``log_sigma`` or ``median_scale`` is not positive,
            ``nodes`` is below 8, ``samples`` is negative, or a level is
            outside ``(0, 1)``.

    Equation:
        [EQ-AERO-PC-007].

    References:
        [REF-NASA-CARA-HANDBOOK, Appendix N], [REF-DAVIS-RABINOWITZ-1984].

    """
    if not isinstance(geometry, EncounterGeometry):
        raise TypeError("geometry must be an EncounterGeometry")
    s = _positive(log_sigma, "log_sigma")
    c_med = _positive(median_scale, "median_scale")
    for name, value in (("nodes", nodes), ("samples", samples)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if nodes < 8:
        raise ValueError("nodes must be at least 8")
    if samples < 0:
        raise ValueError("samples must be non-negative")
    levels = tuple(
        require_finite_real(q, name="quantile level") for q in quantile_levels
    )
    if any(not 0.0 < q < 1.0 for q in levels):
        raise ValueError("quantile levels must lie in (0, 1)")
    half_range = 9.0

    def moments(order: int) -> tuple[float, float]:
        """Return the quadrature mean and second moment of ``Pc(c)``.

        References:
            [REF-DAVIS-RABINOWITZ-1984]. Evaluates [EQ-AERO-PC-007].

        """
        x, w = _gauss_legendre(order)
        first = 0.0
        second = 0.0
        for xi, wi in zip(x, w, strict=True):
            z = half_range * xi
            density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
            value = _exact_pc(geometry, c_med * math.exp(s * z))
            first += half_range * wi * density * value
            second += half_range * wi * density * value * value
        return first, second

    mean, second_moment = moments(nodes)
    mean_check, _ = moments(2 * nodes)
    variance = max(0.0, second_moment - mean * mean)
    error = abs(mean_check - mean)
    levels_out: tuple[float, ...] = ()
    values_out: tuple[float, ...] = ()
    sample_mean: float | None = None
    sample_se: float | None = None
    if samples > 0:
        rng = resolve_stream(seed).generator()
        draws = sorted(
            _exact_pc(geometry, c_med * math.exp(s * rng.gauss(0.0, 1.0)))
            for _ in range(samples)
        )
        sample_mean = math.fsum(draws) / samples
        spread = math.fsum((d - sample_mean) ** 2 for d in draws) / max(1, samples - 1)
        sample_se = math.sqrt(spread / samples)
        levels_out = levels
        picked: list[float] = []
        for q in levels:
            position = q * (samples - 1)
            lo = math.floor(position)
            hi = min(lo + 1, samples - 1)
            picked.append(draws[lo] + (position - lo) * (draws[hi] - draws[lo]))
        values_out = tuple(picked)
    return ScaleUncertaintyResult(
        c_med,
        s,
        _exact_pc(geometry, c_med),
        mean_check,
        math.sqrt(variance),
        error,
        error <= 1.0e-8 * mean_check + 1.0e-14,
        levels_out,
        values_out,
        samples,
        sample_mean,
        sample_se,
    )


# ---------------------------------------------------------------------------
# Monte Carlo and covariance realism
# ---------------------------------------------------------------------------


def monte_carlo_collision_probability(
    relative_position: Iterable[float],
    relative_velocity: Iterable[float],
    covariance_primary: Iterable[Iterable[float]],
    covariance_secondary: Iterable[Iterable[float]],
    radius_primary: float,
    radius_secondary: float,
    *,
    samples: int = 100_000,
    confidence: float = 0.95,
    seed: int | RandomStream | None = None,
) -> MonteCarloProbability:
    """Estimate ``Pc`` by sampling the three-dimensional relative position.

    The relative position is drawn from ``N(r, C_1 + C_2)`` in three
    dimensions and the sample is a hit when its perpendicular distance to
    the straight path through the primary along ``v`` is at most ``R_1 + R_2``,
    ``||y||^2 - (y . w)^2 <= R^2`` with ``w = v / ||v||``. This does not
    use the encounter-plane projection, so it validates it as well as the
    planar integrals. The estimator is crude Monte Carlo; the interval is the
    Wilson score interval of the hit fraction, which stays informative when
    the number of hits is small (but a rare event of probability ``p``
    needs about ``100 / p`` samples for a 10 % relative error).

    Args:
        relative_position: Relative position ``r`` (3 reals, m).
        relative_velocity: Relative velocity ``v`` (3 reals, m/s), non-zero.
        covariance_primary: Symmetric positive-definite ``3 x 3`` covariance.
        covariance_secondary: Same for the secondary; the sum must be
            positive definite in three dimensions.
        radius_primary: Primary radius (m), positive.
        radius_secondary: Secondary radius (m), positive.
        samples: Number of samples (positive).
        confidence: Confidence level of the Wilson interval.
        seed: Seed or stream (``None`` draws OS entropy).

    Returns:
        Estimate, standard error, Wilson interval and counts.

    Raises:
        TypeError: If an entry is not real or ``samples`` is not an integer.
        ValueError: If a shape is wrong, the relative velocity is zero, a
            radius is not positive, the combined covariance is not positive
            definite, ``samples`` is not positive or ``confidence`` is
            outside ``(0, 1)``.

    Equation:
        [EQ-AERO-PC-008].

    References:
        [REF-WILSON-1927], [REF-FOSTER-ESTES-1992].

    """
    r = _vector3(relative_position, "relative_position")
    v = _vector3(relative_velocity, "relative_velocity")
    c1 = _matrix3(covariance_primary, "covariance_primary")
    c2 = _matrix3(covariance_secondary, "covariance_secondary")
    radius = _positive(radius_primary, "radius_primary") + _positive(
        radius_secondary, "radius_secondary"
    )
    if isinstance(samples, bool) or not isinstance(samples, int):
        raise TypeError("samples must be an integer")
    if samples < 1:
        raise ValueError("samples must be positive")
    speed = math.sqrt(_dot(v, v))
    if speed <= 0.0 or not math.isfinite(speed):
        raise ValueError("relative_velocity must be non-zero and finite")
    w = _unit(v)
    combined = tuple(tuple(c1[i][j] + c2[i][j] for j in range(3)) for i in range(3))
    try:
        lower = cholesky(combined)
    except ValueError as error:
        raise ValueError(
            "the combined 3-D covariance must be positive definite"
        ) from error
    rng = resolve_stream(seed).generator()
    radius_sq = radius * radius
    hits = 0
    for _ in range(samples):
        z0, z1, z2 = rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)
        y0 = r[0] + lower[0][0] * z0
        y1 = r[1] + lower[1][0] * z0 + lower[1][1] * z1
        y2 = r[2] + lower[2][0] * z0 + lower[2][1] * z1 + lower[2][2] * z2
        along = y0 * w[0] + y1 * w[1] + y2 * w[2]
        if y0 * y0 + y1 * y1 + y2 * y2 - along * along <= radius_sq:
            hits += 1
    interval = wilson_score_interval(hits, samples, confidence=confidence)
    p = hits / samples
    return MonteCarloProbability(
        p,
        math.sqrt(p * (1.0 - p) / samples),
        interval.lower,
        interval.upper,
        interval.confidence,
        hits,
        samples,
        "crude-wilson",
    )


def conjunction_realism(
    geometry: EncounterGeometry,
    errors: Sequence[Iterable[float]],
    covariances: Sequence[Iterable[Iterable[float]]],
    *,
    confidence: float = 0.95,
    test: str = "cramer-von-mises",
    scan_points: int = 33,
) -> ConjunctionRealism:
    """Check covariance realism of predicted-versus-observed residuals.

    The residuals ``e_i`` (observed minus predicted position, from a set of
    comparable ephemeris pairs) are compared with their covariances through
    the squared Mahalanobis distance ``e' P^-1 e ~ chi2(n)``
    (:func:`probstat.filtering.covariance_realism`, which uses the NEES of
    :func:`probstat.filtering.nees`). Its scale factor ``s`` (covariances
    should be multiplied by ``s``) and the chi-square confidence interval of
    ``s`` are then pushed through the collision probability: ``Pc`` at
    ``c = 1``, at ``c = s`` and the minimum and maximum of ``Pc(c)`` over the
    interval (scan of ``scan_points`` log-spaced scales plus the endpoints).
    The range is wide when the geometry is near the dilution maximum or when
    the interval is wide. Independence of the residual pairs is assumed by
    the interval, as documented there.

    Args:
        geometry: Encounter geometry whose covariance is the one under test.
        errors: ``N`` residual vectors of equal dimension.
        covariances: The ``N`` covariances the residuals should follow.
        confidence: Confidence level of the scale interval.
        test: Goodness-of-fit statistic (default Cramer-von Mises, the
            statistic the NASA handbook prefers).
        scan_points: Scales scanned inside the interval (at least 3).

    Returns:
        The realism result and the ``Pc`` values.

    Raises:
        TypeError: If ``geometry`` is not an :class:`EncounterGeometry` or
            ``scan_points`` is not an integer.
        ValueError: If the inputs are rejected by
            :func:`probstat.filtering.covariance_realism` or ``scan_points``
            is below three.

    Equation:
        [EQ-AERO-PC-009].

    References:
        [REF-HORWOOD-ETAL-2014], [REF-NASA-CARA-HANDBOOK, Appendix I],
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001].

    """
    if not isinstance(geometry, EncounterGeometry):
        raise TypeError("geometry must be an EncounterGeometry")
    if isinstance(scan_points, bool) or not isinstance(scan_points, int):
        raise TypeError("scan_points must be an integer")
    if scan_points < 3:
        raise ValueError("scan_points must be at least 3")
    result = covariance_realism(errors, covariances, confidence=confidence, test=test)
    low, high = result.scale_interval
    log_low, log_high = math.log(low), math.log(high)
    scales = [
        math.exp(log_low + (log_high - log_low) * i / (scan_points - 1))
        for i in range(scan_points)
    ]
    values = [_exact_pc(geometry, c) for c in scales]
    return ConjunctionRealism(
        result,
        _exact_pc(geometry),
        _exact_pc(geometry, result.scale_factor),
        (min(values), max(values)),
    )


__all__ = [
    "PC_METHODS",
    "ConjunctionProbability",
    "ConjunctionRealism",
    "DilutionAnalysis",
    "EncounterGeometry",
    "MonteCarloProbability",
    "PcSensitivity",
    "ScaleUncertaintyResult",
    "chan_series_probability",
    "collision_probability",
    "collision_probability_sensitivity",
    "conjunction_realism",
    "dilution_analysis",
    "encounter_plane_geometry",
    "foster_probability",
    "monte_carlo_collision_probability",
    "pc_under_scale_uncertainty",
    "plane_encounter",
]
