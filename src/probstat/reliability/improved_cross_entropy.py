"""Improved cross-entropy importance sampling with smoothed failure indicators.

Classical cross-entropy importance sampling (``adaptive``) fits its proposal
to the elite points below a sample quantile, so each level uses a tenth of
its sample. The improved method replaces the indicator ``I(g <= 0)`` by the
smooth approximation ``Phi(-g / sigma)`` and lets ``sigma`` decrease from
level to level; every sample then carries a weight in the parameter update.

With the standard normal density ``phi_n`` the intermediate targets are
``h_k(u)`` proportional to ``Phi(-g(u) / sigma_k) phi_n(u)``. They approach
the optimal importance-sampling density ``I(g <= 0) phi_n / P_f`` as
``sigma_k -> 0`` and equal ``phi_n`` for ``sigma = inf``.

* :func:`smoothing_scale` chooses the next ``sigma`` so that the importance
  weights towards the next target have a prescribed coefficient of variation.
* :func:`improved_cross_entropy` runs the adaptation with a single Gaussian
  or a Gaussian mixture fitted by weighted EM and returns the
  importance-sampling estimate of the last sample.

The von Mises-Fisher-Nakagami mixture of the source paper is not implemented.
"""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.distributions import Normal
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector, cholesky, dot, symmetric_eigen

from .adaptive import (
    _floor_eigenvalues,
    _gaussian_draw,
    _kish_effective_size,
    _mean_and_error,
    _require_count,
    _require_limit_state,
)
from .model_based import ModelLimitState

ICE_FAMILIES = ("gaussian", "mixture")
"""Parametric families of :func:`improved_cross_entropy`."""

ICE_TARGET_CV = 1.5
"""Default target coefficient of variation of the level weights."""

ICE_COVARIANCE_FLOOR = 0.55
"""Default lower bound on the eigenvalues of a fitted covariance."""

_RIDGE = 1e-10
_SCAN_FACTOR = 0.7
_SCALE_SPAN = 1e12
_EM_TOLERANCE = 1e-6
_MIN_COMPONENT_WEIGHT = 1e-6
_STANDARD = Normal(0.0, 1.0)
_SQRT_TWO = math.sqrt(2.0)
_LOG_HALF = math.log(0.5)


def _log_smooth_indicator(value: float, sigma: float) -> float:
    """Return ``log Phi(-g / sigma)``; ``log(1/2)`` for ``sigma = inf``.

    For ``sigma = 0`` the limit is used: ``0`` for ``g < 0``, ``log(1/2)``
    for ``g = 0``, and ``-inf`` for ``g > 0``.

    References:
        [EQ-REL-ICE-001], [REF-PAPAIOANNOU-GEYER-STRAUB-2019].

    """
    if math.isinf(sigma):
        return _LOG_HALF
    if sigma <= 0.0:
        if value < 0.0:
            return 0.0
        return _LOG_HALF if value == 0.0 else -math.inf
    z = -value / sigma
    if z > -30.0:
        return math.log(0.5 * math.erfc(-z / _SQRT_TWO))
    if z < -1e150:
        return -math.inf
    return _STANDARD.log_cdf(z)


def _log_cv(log_weights: Sequence[float]) -> float:
    """Return the coefficient of variation of ``exp(log_weights)``.

    The population standard deviation divided by the mean, computed after
    subtracting the largest log-weight; ``inf`` when every weight is zero.

    References:
        [EQ-REL-ICE-002], [REF-OWEN-2013, Chapter 9].

    """
    shift = max(log_weights)
    if shift == -math.inf:
        return math.inf
    scaled = [math.exp(item - shift) for item in log_weights]
    count = len(scaled)
    mean = math.fsum(scaled) / count
    second = math.fsum(item * item for item in scaled) / count
    return math.sqrt(max(0.0, second / (mean * mean) - 1.0))


def _stopping_cv(values: Sequence[float], sigma: float) -> float:
    """Return the c.o.v. of ``I(g <= 0) / Phi(-g / sigma)`` over a sample.

    ``inf`` when no sample failed.

    References:
        [EQ-REL-ICE-004], [REF-PAPAIOANNOU-GEYER-STRAUB-2019].

    """
    return _log_cv(
        [
            -_log_smooth_indicator(value, sigma) if value <= 0.0 else -math.inf
            for value in values
        ]
    )


def _scale_cv(values: Sequence[float], base: Sequence[float], sigma: float) -> float:
    """Return the c.o.v. of ``Phi(-g_i / sigma) exp(base_i)``.

    References:
        [EQ-REL-ICE-002], [REF-PAPAIOANNOU-GEYER-STRAUB-2019].

    """
    return _log_cv(
        [
            _log_smooth_indicator(value, sigma) + offset
            for value, offset in zip(values, base, strict=True)
        ]
    )


def smoothing_scale(
    values: Sequence[float],
    log_base_weights: Sequence[float] | None = None,
    *,
    previous: float = math.inf,
    target_cv: float = ICE_TARGET_CV,
) -> tuple[float, float]:
    """Choose the next smoothing scale from a target weight variation.

    With limit-state values ``g_i`` of the current sample and base weights
    ``b_i`` the weights towards the target with scale ``sigma`` are
    ``W_i(sigma) = Phi(-g_i / sigma) b_i``. The function returns the largest
    ``sigma < previous`` found at which their coefficient of variation
    (population standard deviation over mean) equals ``target_cv``.

    Search: ``sigma`` is decreased geometrically (factor 0.7) from
    ``previous`` (from ``100 max|g_i|`` when ``previous`` is infinite) until
    the c.o.v. exceeds the target, and the crossing is then located by
    bisection on ``log sigma``. The c.o.v. need not be monotone in
    ``sigma``; the scan makes the result the first crossing below
    ``previous`` on the scan grid.

    Two boundary cases are returned without a crossing: if the c.o.v. at the
    upper end already exceeds the target, the upper end is returned (for a
    finite ``previous`` the scale does not decrease); if the c.o.v. stays
    below the target down to ``1e-12`` of the starting value, that smallest
    scale is returned (the indicator itself is then within the target).

    Args:
        values: Limit-state values ``g_i`` of the sample (at least two).
        log_base_weights: Logarithms of the base weights ``b_i``
            (``-inf`` allowed); ``None`` means equal weights.
        previous: Previous scale, positive or ``inf``.
        target_cv: Positive target coefficient of variation.

    Returns:
        The pair ``(sigma, cv)`` with the c.o.v. attained at ``sigma``.

    Raises:
        ValueError: If the inputs are inconsistent, not finite, or every
            base weight is zero.

    Equation:
        [EQ-REL-ICE-002].

    References:
        [REF-PAPAIOANNOU-GEYER-STRAUB-2019],
        [REF-PAPAIOANNOU-PAPADIMITRIOU-STRAUB-2016].

    """
    points = [require_finite_real(value, name="values") for value in values]
    if len(points) < 2:
        raise ValueError("values must contain at least two limit-state values")
    if log_base_weights is None:
        base = [0.0] * len(points)
    else:
        base = [float(item) for item in log_base_weights]
        if len(base) != len(points):
            raise ValueError("log_base_weights must have the length of values")
        if any(math.isnan(item) or item == math.inf for item in base):
            raise ValueError("log_base_weights must be finite or -inf")
        if max(base) == -math.inf:
            raise ValueError("every base weight is zero")
    target = require_finite_real(target_cv, name="target_cv")
    if target <= 0.0:
        raise ValueError("target_cv must be positive")
    if isinstance(previous, bool) or not isinstance(previous, (int, float)):
        raise ValueError("previous must be a positive number or inf")
    upper = float(previous)
    if math.isnan(upper) or upper <= 0.0:
        raise ValueError("previous must be a positive number or inf")
    spread = max(abs(value) for value in points)
    if spread == 0.0:
        spread = 1.0
    if math.isinf(upper):
        upper = 100.0 * spread
    high_cv = _scale_cv(points, base, upper)
    if high_cv >= target:
        return upper, high_cv
    floor = upper / _SCALE_SPAN
    low = upper
    low_cv = high_cv
    while low_cv < target:
        if low <= floor:
            return low, low_cv
        upper, high_cv = low, low_cv
        low = low * _SCAN_FACTOR
        low_cv = _scale_cv(points, base, low)
    log_low, log_high = math.log(low), math.log(upper)
    for _ in range(60):
        if log_high - log_low < 1e-10:
            break
        middle = 0.5 * (log_low + log_high)
        if _scale_cv(points, base, math.exp(middle)) < target:
            log_high = middle
        else:
            log_low = middle
    sigma = math.exp(0.5 * (log_low + log_high))
    return sigma, _scale_cv(points, base, sigma)


class _Mixture:
    """Gaussian mixture ``sum_j pi_j N(mu_j, L_j L_j')`` with Cholesky factors.

    Args:
        weights: Mixing proportions ``pi_j`` summing to one.
        means: Component means.
        covariances: Component covariances.

    References:
        [EQ-GMM-001], [REF-MCLACHLAN-PEEL-2000]. No equation is evaluated.

    """

    __slots__ = ("covariances", "log_dets", "lowers", "means", "weights")

    def __init__(
        self,
        weights: Sequence[float],
        means: Sequence[Vector],
        covariances: Sequence[Matrix],
    ) -> None:
        """Store the parameters and factor every covariance.

        Raises:
            ValueError: If a covariance is not positive definite.

        References:
            [EQ-DIST-MVN-001], [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        self.weights = tuple(weights)
        self.means = tuple(means)
        self.covariances = tuple(covariances)
        self.lowers = tuple(cholesky(item) for item in self.covariances)
        self.log_dets = tuple(
            math.fsum(math.log(lower[i][i]) for i in range(len(lower)))
            for lower in self.lowers
        )

    def component_log_terms(self, u: Vector) -> list[float]:
        """Return ``log pi_j + log N(u; mu_j, Sigma_j) + (d/2) log 2 pi``.

        References:
            [EQ-GMM-001], [EQ-DIST-MVN-001].

        """
        terms: list[float] = []
        dimension = len(u)
        for weight, mean, lower, log_det in zip(
            self.weights, self.means, self.lowers, self.log_dets, strict=True
        ):
            z: list[float] = []
            for i in range(dimension):
                row = lower[i]
                residual = u[i] - mean[i]
                for k in range(i):
                    residual -= row[k] * z[k]
                z.append(residual / row[i])
            terms.append(math.log(weight) - 0.5 * math.fsum(v * v for v in z) - log_det)
        return terms

    def log_weight(self, u: Vector) -> float:
        """Return ``log phi_n(u) - log q(u)``; the ``2 pi`` constants cancel.

        References:
            [EQ-REL-ICE-005], [EQ-DIST-MVN-001].

        """
        terms = self.component_log_terms(u)
        shift = max(terms)
        log_density = shift + math.log(
            math.fsum(math.exp(item - shift) for item in terms)
        )
        return -0.5 * dot(u, u) - log_density

    def draw(self, generator: random.Random) -> Vector:
        """Draw one point: a component by its proportion, then a Gaussian.

        References:
            [EQ-DIST-MVN-005], [REF-MCLACHLAN-PEEL-2000].

        """
        index = 0
        if len(self.weights) > 1:
            uniform = generator.random()
            cumulative = 0.0
            index = len(self.weights) - 1
            for position, weight in enumerate(self.weights):
                cumulative += weight
                if uniform < cumulative:
                    index = position
                    break

        def gauss() -> float:
            """Return one standard normal variate.

            References:
                [REF-MATSUMOTO-NISHIMURA-1998].

            """
            return generator.gauss(0.0, 1.0)

        return _gaussian_draw(self.means[index], self.lowers[index], gauss)[0]


def _weighted_moments(
    points: Sequence[Vector], weights: Sequence[float], floor: float | None
) -> tuple[float, Vector, Matrix]:
    """Return the weight sum, weighted mean, and weighted covariance.

    The covariance is ``sum w (u - mu)(u - mu)' / sum w`` plus a ridge of
    ``1e-10``, with its eigenvalues raised to ``floor`` when one is given.

    References:
        [EQ-REL-ICE-003], [EQ-REL-CE-003], [REF-PAPAIOANNOU-GEYER-STRAUB-2019].

    """
    dimension = len(points[0])
    total = math.fsum(weights)
    mean = tuple(
        math.fsum(w * u[i] for u, w in zip(points, weights, strict=True)) / total
        for i in range(dimension)
    )
    centered = [tuple(u[i] - mean[i] for i in range(dimension)) for u in points]
    rows: list[list[float]] = [[0.0] * dimension for _ in range(dimension)]
    for i in range(dimension):
        for j in range(i + 1):
            value = (
                math.fsum(
                    w * c[i] * c[j] for c, w in zip(centered, weights, strict=True)
                )
                / total
            )
            rows[i][j] = value
            rows[j][i] = value
        rows[i][i] += _RIDGE
    covariance: Matrix = tuple(tuple(row) for row in rows)
    if floor is not None:
        covariance = _floor_eigenvalues(covariance, floor)
    return total, mean, covariance


def _seed_centers(
    points: Sequence[Vector],
    weights: Sequence[float],
    components: int,
    generator: random.Random,
) -> list[Vector]:
    """Return weighted k-means++ seeds.

    The first centre is drawn with probability proportional to the weights,
    each later one with probability proportional to the weight times the
    squared distance to the nearest centre chosen so far.

    References:
        [EQ-GMM-003], [REF-ARTHUR-VASSILVITSKII-2007].

    """

    def pick(scores: Sequence[float]) -> int:
        """Return an index drawn with probability proportional to ``scores``.

        References:
            [EQ-GMM-003]. No further equation is evaluated.

        """
        total = math.fsum(scores)
        if not total > 0.0:
            return generator.randrange(len(scores))
        level = generator.random() * total
        cumulative = 0.0
        for index, score in enumerate(scores):
            cumulative += score
            if level < cumulative:
                return index
        return len(scores) - 1

    centers = [points[pick(weights)]]
    distances = [math.inf] * len(points)
    while len(centers) < components:
        latest = centers[-1]
        for index, point in enumerate(points):
            gap = math.fsum((a - b) ** 2 for a, b in zip(point, latest, strict=True))
            if gap < distances[index]:
                distances[index] = gap
        centers.append(
            points[pick([w * d for w, d in zip(weights, distances, strict=True)])]
        )
    return centers


def _fit_mixture(
    points: Sequence[Vector],
    weights: Sequence[float],
    components: int,
    floor: float | None,
    generator: random.Random,
    iterations: int,
) -> _Mixture:
    """Fit a Gaussian mixture to weighted points by the EM algorithm.

    E-step: ``gamma_ij`` proportional to ``pi_j N(u_i; mu_j, Sigma_j)``.
    M-step with sample weights ``W_i``: ``pi_j = sum_i W_i gamma_ij / sum_i
    W_i`` and the mean and covariance of component ``j`` are the moments
    weighted by ``W_i gamma_ij``. The start is a hard assignment to weighted
    k-means++ seeds. A component whose proportion falls below ``1e-6`` or
    whose covariance cannot be factored is removed; the iteration stops when
    the weighted log-likelihood changes by less than ``1e-6`` relative.

    References:
        [EQ-REL-ICE-003], [EQ-GMM-001], [REF-PAPAIOANNOU-GEYER-STRAUB-2019],
        [REF-DEMPSTER-LAIRD-RUBIN-1977].

    """
    count = len(points)
    total = math.fsum(weights)
    centers = _seed_centers(points, weights, components, generator)
    responsibilities: list[list[float]] = []
    for point in points:
        gaps = [
            math.fsum((a - b) ** 2 for a, b in zip(point, center, strict=True))
            for center in centers
        ]
        nearest = gaps.index(min(gaps))
        responsibilities.append(
            [1.0 if j == nearest else 0.0 for j in range(len(centers))]
        )
    mixture: _Mixture | None = None
    previous = -math.inf
    dimension = len(points[0])
    for _ in range(iterations):
        proportions: list[float] = []
        means: list[Vector] = []
        covariances: list[Matrix] = []
        for j in range(len(responsibilities[0])):
            joint = [weights[i] * responsibilities[i][j] for i in range(count)]
            mass = math.fsum(joint)
            support = sum(1 for item in joint if item > 0.0)
            if mass / total < _MIN_COMPONENT_WEIGHT or support <= dimension:
                continue
            _, mean, covariance = _weighted_moments(points, joint, floor)
            try:
                cholesky(covariance)
            except ValueError:
                continue
            proportions.append(mass / total)
            means.append(mean)
            covariances.append(covariance)
        if not proportions:
            _, mean, covariance = _weighted_moments(points, weights, floor)
            return _Mixture((1.0,), (mean,), (covariance,))
        scale = math.fsum(proportions)
        mixture = _Mixture([item / scale for item in proportions], means, covariances)
        likelihood = 0.0
        responsibilities = []
        for point, weight in zip(points, weights, strict=True):
            terms = mixture.component_log_terms(point)
            shift = max(terms)
            parts = [math.exp(item - shift) for item in terms]
            norm = math.fsum(parts)
            responsibilities.append([item / norm for item in parts])
            likelihood += weight * (shift + math.log(norm))
        likelihood /= total
        if abs(likelihood - previous) <= _EM_TOLERANCE * (1.0 + abs(likelihood)):
            break
        previous = likelihood
    assert mixture is not None
    return mixture


@dataclass(frozen=True, slots=True)
class ImprovedCrossEntropyResult:
    """Improved cross-entropy importance-sampling estimate.

    Args:
        limit_state: Label of the analyzed limit state.
        probability: Importance-sampling estimate of ``P_f`` from the last
            sample.
        standard_error: Standard error of the weighted mean; it assumes
            weights of finite variance (see ``weights_heavy_tailed``).
        coefficient_of_variation: ``standard_error / probability`` (``inf``
            when no sample failed).
        family: ``"gaussian"`` or ``"mixture"``.
        scales: Smoothing scales ``sigma_1 > sigma_2 > ...`` of the fitted
            levels (one per parameter update).
        weight_coefficients_of_variation: C.o.v. of the level weights
            attained at each scale (the target unless a boundary case of
            :func:`smoothing_scale` occurred).
        stopping_coefficients_of_variation: C.o.v. of
            ``I(g <= 0) / Phi(-g / sigma_k)`` for every sample drawn, the
            initial standard-normal sample first (``sigma_0 = inf``).
        level_failures: Failed points in every sample drawn.
        mixture_weights: Proportions of the final proposal (``(1,)`` for a
            single Gaussian).
        means: Component means of the final proposal.
        covariances: Component covariances of the final proposal.
        levels: Number of parameter updates.
        samples_per_level: Samples per level ``N``.
        failures: Failed points in the last sample.
        effective_sample_size: Kish effective sample size of the last
            sample's likelihood ratios ``phi_n / q`` (all points).
        failure_effective_sample_size: Kish effective sample size of the
            likelihood ratios of the failed points, the terms of the
            estimate.
        evaluations: Model evaluations, ``N (levels + 1)``.
        converged: Whether the stopping c.o.v. reached the target within
            ``max_levels`` updates. An unconverged result is the
            importance-sampling estimate of the last proposal and is
            unreliable.
        min_covariance_eigenvalue: Smallest eigenvalue over the component
            covariances of the final proposal.
        weights_heavy_tailed: ``True`` when ``min_covariance_eigenvalue <=
            1/2``: the likelihood ratio then has infinite variance
            [EQ-REL-CE-003] and the standard error can understate the error.

    References:
        [REF-PAPAIOANNOU-GEYER-STRAUB-2019].

    """

    limit_state: str
    probability: float
    standard_error: float
    coefficient_of_variation: float
    family: str
    scales: Vector
    weight_coefficients_of_variation: Vector
    stopping_coefficients_of_variation: Vector
    level_failures: tuple[int, ...]
    mixture_weights: Vector
    means: tuple[Vector, ...]
    covariances: tuple[Matrix, ...]
    levels: int
    samples_per_level: int
    failures: int
    effective_sample_size: float
    failure_effective_sample_size: float
    evaluations: int
    converged: bool
    min_covariance_eigenvalue: float
    weights_heavy_tailed: bool


def improved_cross_entropy(
    limit_state: ModelLimitState,
    *,
    samples_per_level: int = 1000,
    target_cv: float = ICE_TARGET_CV,
    family: str = "gaussian",
    components: int = 1,
    covariance_floor: float | None = ICE_COVARIANCE_FLOOR,
    max_levels: int = 50,
    em_iterations: int = 100,
    seed: int | RandomStream | None = None,
) -> ImprovedCrossEntropyResult:
    """Estimate ``P_f`` by improved cross-entropy importance sampling.

    The proposal ``q_0`` is the standard normal law. For ``k = 0, 1, ...``:

    1. Draw ``N`` points ``u_i ~ q_k``, evaluate ``g_i``, and form the
       likelihood ratios ``w_i = phi_n(u_i) / q_k(u_i)``.
    2. Stop if the coefficient of variation of ``I(g_i <= 0) /
       Phi(-g_i / sigma_k)`` over the sample is at most ``target_cv``
       (``sigma_0 = inf``) [EQ-REL-ICE-004]. This ratio is, up to a
       constant, the density ratio of the optimal importance-sampling
       density and the current smooth target, so a small value means that
       the smooth target is close to the optimal density.
    3. Otherwise choose ``sigma_{k+1} < sigma_k`` by
       :func:`smoothing_scale` so that the weights ``W_i =
       Phi(-g_i / sigma_{k+1}) w_i`` have c.o.v. ``target_cv``
       [EQ-REL-ICE-002], and fit ``q_{k+1}`` by weighted maximum likelihood
       with the weights ``W_i`` [EQ-REL-ICE-003]: the weighted mean and
       covariance for ``family="gaussian"``, weighted EM for
       ``family="mixture"``.

    The estimate is ``P_f = (1/N) sum_i I(g_i <= 0) w_i`` over the sample at
    which the iteration stopped [EQ-REL-ICE-005]. The proposal was fitted to
    earlier samples only and its exact density (after the covariance guard)
    enters ``w_i``, so a sample from a fixed proposal would give an unbiased
    estimate. The stopping rule looks at that same sample, however, which
    leaves a small upward bias for small ``N``: the mean of
    ``estimate / exact`` on a linear limit state with ``beta = 3`` in two
    dimensions was 1.037 for ``N = 30`` (800 runs), 1.023 for ``N = 100``
    (600 runs, Monte Carlo standard error 0.007) and 1.004 for
    ``N = 1000`` (100 runs, standard error 0.005, not distinguishable from
    one).

    Covariance guard: a fitted covariance follows the failure region, whose
    spread along the important direction is far below one for a small
    ``P_f``. The likelihood ratio ``phi_n / q`` has infinite variance once
    an eigenvalue of a component covariance is ``1/2`` or less
    [EQ-REL-CE-003]. By default the eigenvalues of every fitted covariance
    are therefore raised to ``covariance_floor = 0.55``, the same guard as
    in :func:`cross_entropy_importance_sampling`. This is a choice of this
    library, not part of the source method; ``covariance_floor=None`` gives
    the unguarded fit, which usually has a smaller reported c.o.v. that
    understates the error (``weights_heavy_tailed`` is then set).

    Dimension: a full covariance has ``d (d + 3) / 2`` parameters and each
    fit rests on about ``N / (1 + target_cv^2)`` effective samples. When
    that is too few the likelihood ratios of the fitted proposal alone
    exceed the target c.o.v., :func:`smoothing_scale` returns the previous
    scale (see ``weight_coefficients_of_variation`` above the target and
    constant ``scales``), and the run stalls. For a linear limit state with
    ``beta = 3`` in 50 dimensions, ``N = 2000`` failed in 20 of 21 runs
    (30 levels; estimates between ``1e-10`` and 1.09 times the exact
    value, median 0.01, with reported c.o.v. of 0.5 to 1.0 and Kish
    effective sample sizes of a few points; the one run that met the
    stopping rule returned 0.34 times the exact value with a reported
    c.o.v. of 0.84), while ``N = 6000`` converged in 50 of 50 runs in 4 or
    5 levels with mean ``estimate / exact`` of 1.008 and replicate c.o.v.
    0.033 (reported 0.035). The standard error of an unconverged run is not
    a measure of its error; check ``converged`` and
    ``effective_sample_size``. The covariance floor does not help here:
    an eigenvalue just above ``1/2`` gives a finite but very large weight
    variance when many directions sit at the floor, and
    ``weights_heavy_tailed`` stays ``False``.

    A single Gaussian follows one failure region. For several separated
    regions it either locks on one of them and underestimates ``P_f`` or
    covers them with one wide density and a large c.o.v.; use
    ``family="mixture"`` with at least as many components as regions.

    Args:
        limit_state: Model limit state.
        samples_per_level: Samples ``N`` per level (at least 10).
        target_cv: Target c.o.v. of the level weights and of the stopping
            ratio (positive). The default is ``ICE_TARGET_CV = 1.5``.
        family: ``"gaussian"`` or ``"mixture"`` (``ICE_FAMILIES``).
        components: Number of mixture components (at least 1; must be 1 for
            ``family="gaussian"``). Components that receive no weight are
            removed, so the final proposal can have fewer.
        covariance_floor: Lower bound on the eigenvalues of every fitted
            covariance (must exceed ``0.5``) or ``None`` for no guard.
        max_levels: Maximum number of parameter updates.
        em_iterations: Maximum EM iterations per mixture fit.
        seed: Integer seed, random stream, or ``None`` for fresh entropy.

    Returns:
        Estimate, standard error, c.o.v., level history, final proposal,
        weight diagnostics, evaluation count, and convergence flag.

    Raises:
        TypeError: If ``limit_state`` has the wrong type.
        ValueError: If an option is invalid, a limit-state value is not
            finite, or a fitted covariance is not positive definite.

    Warns:
        UserWarning: If the run did not converge or the final proposal has a
            covariance eigenvalue of ``1/2`` or less.

    Equation:
        [EQ-REL-ICE-001], [EQ-REL-ICE-002], [EQ-REL-ICE-003],
        [EQ-REL-ICE-004], [EQ-REL-ICE-005].

    References:
        [REF-PAPAIOANNOU-GEYER-STRAUB-2019].

    """
    state = _require_limit_state(limit_state)
    count = _require_count(samples_per_level, "samples_per_level", 10)
    target = require_finite_real(target_cv, name="target_cv")
    if target <= 0.0:
        raise ValueError("target_cv must be positive")
    if not isinstance(family, str) or family not in ICE_FAMILIES:
        raise ValueError(f"family must be one of {ICE_FAMILIES}")
    parts = _require_count(components, "components", 1)
    if family == "gaussian" and parts != 1:
        raise ValueError("components must be 1 for family='gaussian'")
    floor: float | None = None
    if covariance_floor is not None:
        floor = require_finite_real(covariance_floor, name="covariance_floor")
        if floor <= 0.5:
            raise ValueError("covariance_floor must exceed 0.5")
    levels_limit = _require_count(max_levels, "max_levels", 1)
    em_limit = _require_count(em_iterations, "em_iterations", 1)
    dimension = state.dimension
    if count <= dimension + 1:
        raise ValueError("samples_per_level must exceed the dimension plus one")
    generator = resolve_stream(seed).child("improved-cross-entropy").generator()
    identity: Matrix = tuple(
        tuple(1.0 if i == j else 0.0 for j in range(dimension))
        for i in range(dimension)
    )
    proposal = _Mixture((1.0,), (tuple(0.0 for _ in range(dimension)),), (identity,))
    sigma = math.inf
    scales: list[float] = []
    weight_cvs: list[float] = []
    stopping: list[float] = []
    level_failures: list[int] = []
    before = state.model.evaluations
    converged = False
    while True:
        points = [proposal.draw(generator) for _ in range(count)]
        values = [state.standard(u) for u in points]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("the limit state returned a non-finite value")
        log_ratios = [proposal.log_weight(u) for u in points]
        level_failures.append(sum(1 for value in values if value <= 0.0))
        stop = _stopping_cv(values, sigma)
        stopping.append(stop)
        if stop <= target:
            converged = True
            break
        if len(scales) >= levels_limit:
            break
        sigma, attained = smoothing_scale(
            values, log_ratios, previous=sigma, target_cv=target
        )
        scales.append(sigma)
        weight_cvs.append(attained)
        log_weights = [
            _log_smooth_indicator(value, sigma) + ratio
            for value, ratio in zip(values, log_ratios, strict=True)
        ]
        shift = max(log_weights)
        fit_weights = [math.exp(item - shift) for item in log_weights]
        try:
            if family == "gaussian":
                _, mean, covariance = _weighted_moments(points, fit_weights, floor)
                proposal = _Mixture((1.0,), (mean,), (covariance,))
            else:
                proposal = _fit_mixture(
                    points, fit_weights, parts, floor, generator, em_limit
                )
        except ValueError as error:
            raise ValueError(
                "a fitted covariance is not positive definite; increase "
                "samples_per_level or set covariance_floor"
            ) from error
    ratios = [math.exp(item) for item in log_ratios]
    contributions = [
        ratio if value <= 0.0 else 0.0
        for ratio, value in zip(ratios, values, strict=True)
    ]
    probability, standard_error = _mean_and_error(contributions)
    failed = [item for item in contributions if item > 0.0]
    min_eigenvalue = min(min(symmetric_eigen(item)[0]) for item in proposal.covariances)
    heavy = bool(scales) and min_eigenvalue <= 0.5
    if not converged:
        warnings.warn(
            "improved cross-entropy did not reach the stopping criterion within "
            f"{levels_limit} levels; the estimate is unreliable",
            stacklevel=2,
        )
    if heavy:
        warnings.warn(
            "the final proposal has a covariance eigenvalue of 1/2 or less; the "
            "importance weights have infinite variance and the standard error "
            "can understate the error",
            stacklevel=2,
        )
    return ImprovedCrossEntropyResult(
        limit_state=state.label,
        probability=probability,
        standard_error=standard_error,
        coefficient_of_variation=(
            standard_error / probability if probability > 0.0 else math.inf
        ),
        family=family,
        scales=tuple(scales),
        weight_coefficients_of_variation=tuple(weight_cvs),
        stopping_coefficients_of_variation=tuple(stopping),
        level_failures=tuple(level_failures),
        mixture_weights=proposal.weights,
        means=proposal.means,
        covariances=proposal.covariances,
        levels=len(scales),
        samples_per_level=count,
        failures=level_failures[-1],
        effective_sample_size=_kish_effective_size(ratios),
        failure_effective_sample_size=_kish_effective_size(failed),
        evaluations=state.model.evaluations - before,
        converged=converged,
        min_covariance_eigenvalue=min_eigenvalue,
        weights_heavy_tailed=heavy,
    )
