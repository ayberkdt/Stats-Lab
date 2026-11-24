"""Probability that a Gaussian vector falls inside a ball.

For ``X ~ N(mu, Sigma)`` in ``n`` dimensions this module computes
``P(||X|| <= R)``, the probability content of the ball of radius ``R``
centred at the origin. In conjunction assessment ``X`` is the relative
position (miss vector) of two objects at closest approach and ``R`` the sum
of their hard-body radii; the same number answers any question of the form
"how likely is a Gaussian error to land within a given distance of a point".
The ball is centred at the origin; to test another centre, subtract it from
the mean.

Four methods are provided and they cross-check one another:

- :func:`gaussian_disc_probability` (``n = 2``): rotate to the principal axes
  of ``Sigma`` and integrate over one axis with the inner integral in closed
  form through the normal distribution function. The outer variable is
  substituted ``u = R sin(theta)`` so that the square-root edge of the chord
  disappears, and the integration range is split where the chord length
  equals the absolute offset along the other axis. The error is the
  quadrature estimate (not a guarantee).
- :func:`isotropic_gaussian_ball_probability`: for ``Sigma = s^2 I`` the
  squared radius over ``s^2`` is noncentral chi-square, so the probability is
  a Poisson mixture of central chi-square distribution functions (for
  ``n = 2``, one minus the Marcum Q function of the Rice law). The neglected
  Poisson tails give a rigorous bound.
- :func:`gaussian_ball_probability` (any ``n``): the distribution of the
  positive-definite quadratic form ``sum_i lambda_i (z_i + b_i)^2`` as a
  non-negative mixture of central chi-square laws (Ruben's series). The
  weights are non-negative and sum to one for any scale
  ``beta <= lambda_min``, which makes the truncation error bounded by the
  unused weight times one chi-square value; the bound is rigorous up to
  rounding of the weight sum.
- :func:`small_disc_probability`: density at the centre times the area, with
  the first correction as its stated error.

:func:`maximum_disc_probability` answers the "dilution" question: how large
can the probability be when the covariance is multiplied by an unknown
factor (the miss vector lies outside the disc).

Assumptions: ``Sigma`` is symmetric positive definite; the ball is centred at
the origin; arithmetic is in ``float`` and relative accuracy is lost for
probabilities below about ``1e-300``. The series of
:func:`gaussian_ball_probability` needs terms in proportion to the condition
number of ``Sigma`` and to the squared Mahalanobis offset; when
``max_terms`` runs out the result says ``converged=False`` and reports the
remaining bound. For ``n = 2`` the quadrature is the robust choice for
elongated covariances.

References:
    [REF-RUBEN-1962], [REF-SHEIL-OMUIRCHEARTAIGH-1977], [REF-IMHOF-1961],
    [REF-JOHNSON-KOTZ-BALAKRISHNAN-1995], [REF-FOSTER-ESTES-1992],
    [REF-ALFANO-2005], [REF-ALFANO-2005B], [REF-CHAN-2008],
    [REF-NASA-CARA-HANDBOOK].

"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from probstat.distributions import ChiSquare
from probstat.foundations.validation import require_finite_real
from probstat.linalg import symmetric_eigen
from probstat.numerics import gauss_kronrod

from .normal import MultivariateNormal

_SQRT2 = math.sqrt(2.0)
_WINDOW_SIGMAS = 40.0
_RESCALE_LIMIT = 1.0e150


@dataclass(frozen=True, slots=True)
class BallProbability:
    """Probability content of a ball with an error statement.

    Args:
        probability: ``P(||X|| <= R)`` in ``[0, 1]``.
        error_estimate: Absolute error of ``probability`` (a bound when
            ``rigorous`` is true, an estimate otherwise).
        method: ``"disc-quadrature"``, ``"isotropic-poisson"``,
            ``"ruben-series"`` or ``"small-disc"``.
        rigorous: Whether ``error_estimate`` is a mathematical bound (up to
            floating-point rounding of the weight sums) rather than a
            quadrature or truncation-order estimate.
        converged: Whether the stopping rule of the method was met before its
            term or panel limit.
        terms: Series terms or quadrature evaluations used.

    References:
        [REF-RUBEN-1962], [REF-FOSTER-ESTES-1992].

    """

    probability: float
    error_estimate: float
    method: str
    rigorous: bool
    converged: bool
    terms: int


@dataclass(frozen=True, slots=True)
class MaximumBallProbability:
    """Largest ball probability over a covariance scale factor.

    Args:
        scale: Factor ``c`` multiplying the covariance at the maximum.
        probability: ``P(||X|| <= R)`` for ``X ~ N(mu, c Sigma)``.
        evaluations: Disc probabilities evaluated.

    References:
        [REF-ALFANO-2005B].

    """

    scale: float
    probability: float
    evaluations: int


def _validated_gaussian(
    mean: Iterable[float],
    covariance: Iterable[Iterable[float]],
    radius: float,
    *,
    dimension: int | None = None,
) -> tuple[MultivariateNormal, float]:
    """Return the validated Gaussian law and a positive radius.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree, the covariance is not symmetric
            positive definite, ``dimension`` is not matched, or the radius is
            not positive and finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No mathematical
        equation is evaluated.

    """
    law = MultivariateNormal(tuple(mean), tuple(tuple(row) for row in covariance))
    if dimension is not None and len(law.mean) != dimension:
        raise ValueError(f"mean must have {dimension} components")
    radius_value = require_finite_real(radius, name="radius")
    if radius_value <= 0.0:
        raise ValueError("radius must be positive")
    return law, radius_value


def _normal_interval(lower: float, upper: float) -> float:
    """Return ``Phi(upper) - Phi(lower)`` without cancellation in the tails.

    The tail of the side that is farther from zero is evaluated with
    ``erfc`` so that an interval deep in a tail keeps full relative accuracy.

    References:
        [REF-JOHNSON-KOTZ-BALAKRISHNAN-1995]. Evaluates [EQ-MVN-DISC-001].

    """
    if upper <= lower:
        return 0.0
    if lower > 0.0:
        return 0.5 * (math.erfc(lower / _SQRT2) - math.erfc(upper / _SQRT2))
    if upper < 0.0:
        return 0.5 * (math.erfc(-upper / _SQRT2) - math.erfc(-lower / _SQRT2))
    return 1.0 - 0.5 * math.erfc(upper / _SQRT2) - 0.5 * math.erfc(-lower / _SQRT2)


def gaussian_disc_probability(
    mean: Iterable[float],
    covariance: Iterable[Iterable[float]],
    radius: float,
    *,
    tolerance: float = 1.0e-11,
) -> BallProbability:
    """Return ``P(||X|| <= R)`` for a two-dimensional Gaussian ``X``.

    With the spectral decomposition ``Sigma = V diag(s1^2, s2^2) V'`` and
    ``a = V'mu``, the coordinates ``u = V'X`` are independent normal, and

    ``P = int_(-R)^(R) phi((u - a1)/s1)/s1 *
    [Phi((h(u) - a2)/s2) - Phi((-h(u) - a2)/s2)] du``, ``h(u) = sqrt(R^2 - u^2)``.

    The larger standard deviation is the outer variable. The substitution
    ``u = R sin(theta)`` removes the square-root edge of ``h``; the range is
    cut at the mode ``a1`` and where ``h(u) = |a2|``; and it is restricted to
    ``a1 +/- 40 s1``, outside which the density is below ``1e-347``. This
    is the short-encounter collision probability when ``X`` is the miss
    vector in the encounter plane and ``R`` the combined hard-body radius: the
    integral of the Gaussian density over the hard-body disc (Foster and
    Estes), which the tests also compare with the one-dimensional
    error-function form given in the NASA handbook.

    Args:
        mean: Mean of ``X`` (two finite reals).
        covariance: Symmetric positive-definite ``2 x 2`` covariance.
        radius: Positive finite radius ``R``.
        tolerance: Relative tolerance of each adaptive Gauss-Kronrod panel
            set.

    Returns:
        The probability with the summed quadrature error estimate (not a
        bound).

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the mean is not two-dimensional, the covariance is not
            symmetric positive definite, the radius is not positive, or the
            tolerance is not in ``(0, 1)``.

    Equation:
        [EQ-MVN-DISC-001].

    References:
        [REF-FOSTER-ESTES-1992], [REF-NASA-CARA-HANDBOOK, Appendix N],
        [REF-ALFANO-2005].

    """
    law, radius_value = _validated_gaussian(mean, covariance, radius, dimension=2)
    tol = require_finite_real(tolerance, name="tolerance")
    if not 0.0 < tol < 1.0:
        raise ValueError("tolerance must lie in (0, 1)")
    values, vectors = symmetric_eigen(law.covariance)
    # Outer axis: the larger variance (index 1, eigenvalues are ascending).
    s1 = math.sqrt(values[1])
    s2 = math.sqrt(values[0])
    a1 = vectors[0][1] * law.mean[0] + vectors[1][1] * law.mean[1]
    a2 = vectors[0][0] * law.mean[0] + vectors[1][0] * law.mean[1]

    def integrand(theta: float) -> float:
        """Return the chord probability times the density in ``theta``.

        References:
            [REF-FOSTER-ESTES-1992]. Evaluates [EQ-MVN-DISC-001].

        """
        height = radius_value * math.cos(theta)
        u = radius_value * math.sin(theta)
        scaled = (u - a1) / s1
        density = math.exp(-0.5 * scaled * scaled) / (s1 * math.sqrt(2.0 * math.pi))
        chord = _normal_interval((-height - a2) / s2, (height - a2) / s2)
        return density * chord * height

    window_low = max(-radius_value, a1 - _WINDOW_SIGMAS * s1)
    window_high = min(radius_value, a1 + _WINDOW_SIGMAS * s1)
    if window_low >= window_high:
        return BallProbability(0.0, 0.0, "disc-quadrature", False, True, 0)
    theta_low = math.asin(max(-1.0, min(1.0, window_low / radius_value)))
    theta_high = math.asin(max(-1.0, min(1.0, window_high / radius_value)))
    cuts = {math.asin(max(-1.0, min(1.0, a1 / radius_value)))}
    if abs(a2) < radius_value:
        angle = math.acos(abs(a2) / radius_value)
        cuts.update((angle, -angle))
    edges = sorted(
        {theta_low, theta_high} | {c for c in cuts if theta_low < c < theta_high}
    )
    total = 0.0
    error = 0.0
    evaluations = 0
    converged = True
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        piece = gauss_kronrod(integrand, left, right, rel_tol=tol, limit=400)
        total += piece.value
        error += piece.error_estimate
        evaluations += piece.evaluations
        converged = converged and piece.converged
    probability = min(1.0, max(0.0, total))
    return BallProbability(
        probability, error, "disc-quadrature", False, converged, evaluations
    )


def _poisson_chi_square(
    x: float, dimension: int, noncentrality: float, tolerance: float
) -> tuple[float, float, int]:
    """Return the noncentral chi-square CDF, a truncation bound, and the terms.

    ``F(x; k, lambda) = sum_j Pois(j; lambda/2) F_(k+2j)(x)``. The sum runs
    outward from the Poisson mode; the neglected part is bounded by the
    Poisson tail mass (exact, from incomplete gamma functions) times the
    largest neglected chi-square value.

    References:
        [REF-JOHNSON-KOTZ-BALAKRISHNAN-1995]. Evaluates [EQ-MVN-RICE-001].

    """
    mu = 0.5 * noncentrality
    if mu == 0.0:
        return float(ChiSquare(float(dimension)).cdf(x)), 0.0, 1
    log_mu = math.log(mu)
    mode = int(mu)

    def term(j: int) -> float:
        """Return the Poisson weight times the chi-square CDF of term ``j``.

        References:
            [REF-JOHNSON-KOTZ-BALAKRISHNAN-1995]. Evaluates [EQ-MVN-RICE-001].

        """
        log_weight = -mu + j * log_mu - math.lgamma(j + 1.0)
        return math.exp(log_weight) * float(ChiSquare(float(dimension + 2 * j)).cdf(x))

    total = term(mode)
    terms = 1
    up = mode + 1
    down = mode - 1
    bound_up = 0.0
    bound_down = 0.0
    done_up = False
    done_down = down < 0
    while not (done_up and done_down) and terms < 100_000:
        if not done_up:
            piece = term(up)
            total += piece
            terms += 1
            above = float(ChiSquare(2.0 * (up + 1)).cdf(2.0 * mu))
            bound_up = above * float(ChiSquare(float(dimension + 2 * (up + 1))).cdf(x))
            up += 1
            done_up = bound_up <= tolerance * total
        if not done_down:
            piece = term(down)
            total += piece
            terms += 1
            below = float(ChiSquare(2.0 * down).sf(2.0 * mu)) if down > 0 else 0.0
            bound_down = below * float(ChiSquare(float(dimension)).cdf(x))
            down -= 1
            done_down = down < 0 or bound_down <= tolerance * total
    return min(1.0, total), bound_up + bound_down, terms


def noncentral_chi_square_cdf(
    x: float, degrees_of_freedom: int, noncentrality: float
) -> float:
    """Return the noncentral chi-square distribution function.

    ``P(sum_(i=1)^k (z_i + d_i)^2 <= x)`` with ``z_i`` standard normal and
    ``lambda = sum d_i^2``, as the Poisson mixture
    ``sum_j exp(-lambda/2) (lambda/2)^j / j! * P(chi2_(k+2j) <= x)``. For
    ``k = 2`` this is one minus the Marcum Q function of the Rice law.

    Args:
        x: Argument; non-positive values give zero.
        degrees_of_freedom: Positive integer ``k``.
        noncentrality: Non-negative ``lambda``.

    Returns:
        The probability, accurate to about ``1e-13`` relative.

    Raises:
        TypeError: If an argument is not real or ``k`` is not an integer.
        ValueError: If ``k < 1`` or ``lambda < 0``.

    Equation:
        [EQ-MVN-RICE-001].

    References:
        [REF-JOHNSON-KOTZ-BALAKRISHNAN-1995].

    """
    if isinstance(degrees_of_freedom, bool) or not isinstance(degrees_of_freedom, int):
        raise TypeError("degrees_of_freedom must be an integer")
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be at least one")
    argument = require_finite_real(x, name="x")
    delta = require_finite_real(noncentrality, name="noncentrality")
    if delta < 0.0:
        raise ValueError("noncentrality must be non-negative")
    if argument <= 0.0:
        return 0.0
    return _poisson_chi_square(argument, degrees_of_freedom, delta, 1.0e-14)[0]


def isotropic_gaussian_ball_probability(
    offset: float, sigma: float, radius: float, *, dimension: int = 2
) -> BallProbability:
    """Return ``P(||X|| <= R)`` for ``X ~ N(mu, sigma^2 I)`` with ``||mu|| = offset``.

    ``||X||^2 / sigma^2`` is noncentral chi-square with ``n`` degrees of
    freedom and noncentrality ``(offset/sigma)^2``; for ``n = 2`` this is
    ``1 - Q_1(offset/sigma, R/sigma)`` with the Marcum function ``Q_1`` (the
    Rice law of ``||X||``). The error is the neglected Poisson tail mass, a
    rigorous bound.

    Args:
        offset: Distance ``||mu||`` from the ball centre to the mean.
        sigma: Positive standard deviation per coordinate.
        radius: Positive radius.
        dimension: Dimension ``n`` of the space.

    Returns:
        The probability and the truncation bound.

    Raises:
        TypeError: If an argument is not real or ``dimension`` is not an
            integer.
        ValueError: If ``offset < 0``, ``sigma <= 0``, ``radius <= 0`` or
            ``dimension < 1``.

    Equation:
        [EQ-MVN-RICE-001].

    References:
        [REF-JOHNSON-KOTZ-BALAKRISHNAN-1995], [REF-CHAN-2008].

    """
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError("dimension must be an integer")
    if dimension < 1:
        raise ValueError("dimension must be at least one")
    distance = require_finite_real(offset, name="offset")
    scale = require_finite_real(sigma, name="sigma")
    radius_value = require_finite_real(radius, name="radius")
    if distance < 0.0:
        raise ValueError("offset must be non-negative")
    if scale <= 0.0 or radius_value <= 0.0:
        raise ValueError("sigma and radius must be positive")
    value, bound, terms = _poisson_chi_square(
        (radius_value / scale) ** 2, dimension, (distance / scale) ** 2, 1.0e-14
    )
    return BallProbability(value, bound, "isotropic-poisson", True, True, terms)


def gaussian_ball_probability(
    mean: Iterable[float],
    covariance: Iterable[Iterable[float]],
    radius: float,
    *,
    tolerance: float = 1.0e-10,
    max_terms: int = 4000,
) -> BallProbability:
    """Return ``P(||X|| <= R)`` for ``X ~ N(mu, Sigma)`` in ``n`` dimensions.

    With ``Sigma = V diag(lambda) V'`` the squared norm is the quadratic form
    ``Q = sum_i lambda_i (z_i + b_i)^2`` with ``b_i = (V'mu)_i / sqrt(lambda_i)``.
    For a scale ``beta <= lambda_min`` and ``r_i = 1 - beta/lambda_i`` in
    ``[0, 1)``, the Laplace transform of ``Q`` equals
    ``sum_k a_k (1 + 2 s beta)^(-(n+2k)/2)`` with

    ``a_0 = prod_i sqrt(beta/lambda_i) e^(-b_i^2/2)``,
    ``a_k = (1/k) sum_(j=1)^k d_j a_(k-j)``,
    ``d_j = sum_i [ r_i^j/2 + j b_i^2 (1 - r_i) r_i^(j-1)/2 ]``,

    where every ``a_k >= 0`` (the generating function is a product of
    power series with non-negative coefficients) and ``sum a_k = 1``. Hence
    ``Q`` is a mixture of ``beta chi2_(n+2k)`` and
    ``P(Q <= R^2) = sum_k a_k P(chi2_(n+2k) <= R^2/beta)``. Truncating after
    ``K`` terms leaves at most ``(1 - sum_(k<K) a_k) P(chi2_(n+2K) <= R^2/beta)``
    because the chi-square distribution function decreases in its degrees of
    freedom. Here ``beta = lambda_min``. All recursion terms are
    non-negative, so there is no cancellation; the weights are rescaled
    during the recursion to survive large offsets.

    Args:
        mean: Mean of ``X`` (``n >= 1`` finite reals).
        covariance: Symmetric positive-definite ``n x n`` covariance.
        radius: Positive radius.
        tolerance: Stop when the truncation bound is below
            ``tolerance`` times the probability accumulated so far.
        max_terms: Largest number of series terms.

    Returns:
        The probability and the truncation bound (rigorous up to rounding of
        the weight sum, about ``1e-16`` times the chi-square value);
        ``converged`` is false when ``max_terms`` was reached first.

    Raises:
        TypeError: If an entry is not real or ``max_terms`` is not an integer.
        ValueError: If the covariance is not symmetric positive definite, the
            radius is not positive, the tolerance is not in ``(0, 1)``, or
            ``max_terms < 1``.

    Equation:
        [EQ-MVN-BALL-001].

    References:
        [REF-RUBEN-1962], [REF-SHEIL-OMUIRCHEARTAIGH-1977], [REF-IMHOF-1961].

    """
    law, radius_value = _validated_gaussian(mean, covariance, radius)
    tol = require_finite_real(tolerance, name="tolerance")
    if not 0.0 < tol < 1.0:
        raise ValueError("tolerance must lie in (0, 1)")
    if isinstance(max_terms, bool) or not isinstance(max_terms, int):
        raise TypeError("max_terms must be an integer")
    if max_terms < 1:
        raise ValueError("max_terms must be at least one")
    values, vectors = symmetric_eigen(law.covariance)
    size = len(values)
    rotated = tuple(
        math.fsum(vectors[row][col] * law.mean[row] for row in range(size))
        for col in range(size)
    )
    shifts_sq = tuple(a * a / lam for a, lam in zip(rotated, values, strict=True))
    beta = values[0]
    ratios = tuple(max(0.0, 1.0 - beta / lam) for lam in values)
    log_a0 = 0.5 * math.fsum(math.log(beta / lam) for lam in values) - 0.5 * math.fsum(
        shifts_sq
    )
    argument = radius_value * radius_value / beta
    powers = [1.0] * size  # r_i^(j-1)
    scaled = [1.0]  # a_k / a_0, rescaled by exp(-log_shift)
    log_shift = 0.0
    weight_sum = 0.0
    probability = 0.0
    bound = 1.0
    d_terms: list[float] = []
    converged = False
    count = 0
    for k in range(max_terms):
        if k > 0:
            powers_next = [p * r for p, r in zip(powers, ratios, strict=True)]
            d_k = math.fsum(
                0.5 * pn + 0.5 * k * s * (1.0 - r) * p
                for pn, p, r, s in zip(
                    powers_next, powers, ratios, shifts_sq, strict=True
                )
            )
            powers = powers_next
            d_terms.append(d_k)
            value = (
                math.fsum(d_terms[j - 1] * scaled[k - j] for j in range(1, k + 1)) / k
            )
            scaled.append(value)
            if value > _RESCALE_LIMIT:
                scaled = [v / _RESCALE_LIMIT for v in scaled]
                log_shift += math.log(_RESCALE_LIMIT)
        weight = (
            math.exp(math.log(scaled[k]) + log_a0 + log_shift)
            if scaled[k] > 0.0
            else 0.0
        )
        weight_sum += weight
        probability += weight * float(ChiSquare(float(size + 2 * k)).cdf(argument))
        count = k + 1
        remaining = max(0.0, 1.0 - weight_sum)
        bound = remaining * float(ChiSquare(float(size + 2 * k + 2)).cdf(argument))
        if bound <= tol * probability or (probability == 0.0 and bound == 0.0):
            converged = True
            break
    return BallProbability(
        min(1.0, probability), bound, "ruben-series", True, converged, count
    )


def small_disc_probability(
    mean: Iterable[float],
    covariance: Iterable[Iterable[float]],
    radius: float,
) -> BallProbability:
    """Return the small-disc approximation ``pi R^2 f(0)`` and its error.

    For a density ``f`` that is smooth on the disc,
    ``int_(||x||<=R) f = pi R^2 f(0) + (pi R^4 / 8) Laplacian f(0) + O(R^6)``
    (the mean value of a smooth function over a disc). For the Gaussian
    ``f(0) = exp(-m'S^-1 m / 2) / (2 pi sqrt(det S))`` and
    ``Laplacian f(0) = f(0) (||S^-1 m||^2 - tr S^-1)``, so the leading error
    is the second term, returned as ``error_estimate`` (its magnitude; an
    estimate, not a bound). The approximation is good when ``R`` is small
    against every length on which ``f`` varies: ``R`` much smaller than
    ``sqrt(lambda_min)`` and than ``lambda_min / ||m||`` when the mean is
    far from the origin.

    Args:
        mean: Mean of ``X`` (two finite reals).
        covariance: Symmetric positive-definite ``2 x 2`` covariance.
        radius: Positive radius.

    Returns:
        ``pi R^2 f(0)`` with the magnitude of the next term as the error
        estimate (``rigorous`` is false).

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the mean is not two-dimensional, the covariance is not
            symmetric positive definite, or the radius is not positive.

    Equation:
        [EQ-MVN-SMALLDISC-001].

    References:
        [REF-CHAN-2008], [REF-ALFANO-2005], [REF-FOSTER-ESTES-1992].

    """
    law, radius_value = _validated_gaussian(mean, covariance, radius, dimension=2)
    (a, b), (_, d) = law.covariance
    determinant = a * d - b * b
    inverse = ((d / determinant, -b / determinant), (-b / determinant, a / determinant))
    mx, my = law.mean
    precision_mean = (
        inverse[0][0] * mx + inverse[0][1] * my,
        inverse[1][0] * mx + inverse[1][1] * my,
    )
    quadratic = mx * precision_mean[0] + my * precision_mean[1]
    density = math.exp(-0.5 * quadratic) / (2.0 * math.pi * math.sqrt(determinant))
    laplacian = density * (
        precision_mean[0] ** 2
        + precision_mean[1] ** 2
        - (inverse[0][0] + inverse[1][1])
    )
    value = math.pi * radius_value**2 * density
    correction = math.pi * radius_value**4 / 8.0 * laplacian
    return BallProbability(value, abs(correction), "small-disc", False, True, 0)


def maximum_disc_probability(
    mean: Iterable[float],
    covariance: Iterable[Iterable[float]],
    radius: float,
    *,
    tolerance: float = 1.0e-9,
) -> MaximumBallProbability:
    """Return the largest disc probability over a covariance scale factor.

    For ``X ~ N(mu, c Sigma)`` the probability ``p(c)`` of the disc of radius
    ``R`` is tends to zero as ``c -> 0`` and as ``c -> infinity`` when the
    mean lies outside the disc (``||mu|| > R``), so it has a maximum
    ("dilution": a larger covariance can raise the probability). The
    maximum is located by a scan of ``log c`` over a factor of about
    ``1e6`` around ``||mu||^2 / tr(Sigma)`` followed by golden-section
    refinement of the bracket around the best scan point. For a small disc
    and an isotropic covariance the maximum is ``R^2 / (e ||mu||^2)`` at
    ``c s^2 = ||mu||^2 / 2``. Uniqueness of the maximum is not proven for a
    strongly anisotropic covariance; the scan guards against a missed
    secondary peak at the resolution of its grid.

    Args:
        mean: Mean of ``X`` (two finite reals) with ``||mean|| > radius``.
        covariance: Symmetric positive-definite ``2 x 2`` covariance.
        radius: Positive radius.
        tolerance: Relative width of the final golden-section bracket in
            ``log c``.

    Returns:
        The maximizing scale, the probability there and the number of disc
        probabilities evaluated.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the mean is not two-dimensional, the covariance is not
            symmetric positive definite, the radius is not positive, the mean
            lies inside or on the disc (the supremum is then one, or one half
            on the boundary, and is not attained at a positive scale), or the
            maximum lies at the edge of the scanned range.

    Equation:
        [EQ-MVN-DILUTION-001].

    References:
        [REF-ALFANO-2005B], [REF-NASA-CARA-HANDBOOK, Section M.4].

    """
    law, radius_value = _validated_gaussian(mean, covariance, radius, dimension=2)
    tol = require_finite_real(tolerance, name="tolerance")
    if not 0.0 < tol < 1.0:
        raise ValueError("tolerance must lie in (0, 1)")
    distance = math.hypot(*law.mean)
    if distance <= radius_value:
        raise ValueError("the mean must lie outside the disc for a finite maximizer")
    base = distance * distance / (law.covariance[0][0] + law.covariance[1][1])
    log_base = math.log(base)
    evaluations = 0

    def probability_at(log_scale: float) -> float:
        """Return the disc probability for the scale ``exp(log_scale)``.

        References:
            [REF-ALFANO-2005B]. Evaluates [EQ-MVN-DILUTION-001].

        """
        nonlocal evaluations
        evaluations += 1
        factor = math.exp(log_scale)
        scaled = tuple(tuple(factor * v for v in row) for row in law.covariance)
        return gaussian_disc_probability(law.mean, scaled, radius_value).probability

    grid = [log_base + 7.0 * (-1.0 + i / 30.0) for i in range(61)]
    scan = [probability_at(point) for point in grid]
    best = max(range(len(grid)), key=scan.__getitem__)
    if best in (0, len(grid) - 1):
        raise ValueError("the maximum lies outside the scanned range of scales")
    low, high = grid[best - 1], grid[best + 1]
    inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = high - inverse_phi * (high - low)
    x2 = low + inverse_phi * (high - low)
    f1, f2 = probability_at(x1), probability_at(x2)
    while high - low > tol:
        if f1 < f2:
            low, x1, f1 = x1, x2, f2
            x2 = low + inverse_phi * (high - low)
            f2 = probability_at(x2)
        else:
            high, x2, f2 = x2, x1, f1
            x1 = high - inverse_phi * (high - low)
            f1 = probability_at(x1)
    point = 0.5 * (low + high)
    return MaximumBallProbability(math.exp(point), probability_at(point), evaluations)


__all__ = [
    "BallProbability",
    "MaximumBallProbability",
    "gaussian_ball_probability",
    "gaussian_disc_probability",
    "isotropic_gaussian_ball_probability",
    "maximum_disc_probability",
    "noncentral_chi_square_cdf",
    "small_disc_probability",
]
