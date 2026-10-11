"""Adaptive rare-event estimators: cross-entropy importance sampling and lines.

Both estimators work on a :class:`ModelLimitState` in independent
standard-normal space ``u`` (the Nataf image of the model's joint input law)
and report their model-evaluation cost.

* :func:`cross_entropy_importance_sampling` adapts a Gaussian proposal
  ``N(mu, Sigma)`` through a sequence of intermediate failure thresholds given
  by sample quantiles, then estimates ``P_f`` by importance sampling.
* :func:`line_sampling` draws standard-normal points, projects them onto the
  hyperplane orthogonal to an important direction ``alpha``, and integrates the
  remaining one-dimensional probability exactly along each line after locating
  the limit-state crossing by Brent's method.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    dot,
    norm,
    symmetric_eigen,
)
from probstat.modeling import standard_normal_cdf
from probstat.optimization import brent_root

from .model_based import FORMResult, ModelLimitState, form_hlrf

_RIDGE = 1e-10


def _require_limit_state(limit_state: object) -> ModelLimitState:
    """Return the argument when it is a :class:`ModelLimitState`.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not isinstance(limit_state, ModelLimitState):
        raise TypeError("limit_state must be a ModelLimitState")
    return limit_state


def _require_count(value: object, name: str, minimum: int) -> int:
    """Validate an integer option with a lower limit.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _mean_and_error(values: Sequence[float]) -> tuple[float, float]:
    """Return the sample mean and the standard error of the mean.

    References:
        [EQ-MC-MEAN-001], [EQ-MC-SE-001], [REF-ROBERT-CASELLA-2004, Chapter 3].

    """
    count = len(values)
    mean = math.fsum(values) / count
    if count < 2:
        return mean, math.inf
    variance = math.fsum((v - mean) ** 2 for v in values) / (count - 1)
    return mean, math.sqrt(variance / count)


@dataclass(frozen=True, slots=True)
class CrossEntropyResult:
    """Cross-entropy adaptive importance-sampling estimate.

    Args:
        limit_state: Label of the analyzed limit state.
        probability: Final importance-sampling estimate of ``P_f``.
        standard_error: Standard error of the weighted mean.
        coefficient_of_variation: ``standard_error / probability``.
        thresholds: Intermediate levels ``gamma_1 > ... > gamma_m = 0``.
        mean: Final proposal mean in standard-normal space.
        covariance: Final proposal covariance (identity unless adapted).
        levels: Number of adaptation levels run.
        samples_per_level: Samples per adaptation level.
        final_samples: Samples of the final importance-sampling stage.
        failures: Final-stage samples in the failure domain.
        effective_sample_size: Kish effective sample size of all final
            weights (including the zero-indicator samples). For a rare event
            this is small even for a healthy proposal; use
            ``failure_effective_sample_size`` to judge the estimate.
        evaluations: Model evaluations (adaptation plus final stage).
        converged: Whether an adaptation level reached threshold ``0``.
        failure_effective_sample_size: Kish effective sample size
            ``(sum w)^2 / sum w^2`` of the weights of the final-stage failure
            samples, the terms that make up the estimate.
        weights_degenerate: ``True`` when ``failure_effective_sample_size``
            is below ``CE_MIN_FAILURE_ESS_FRACTION`` times ``final_samples``
            with at least one failure (a few samples carry the estimate), or
            when ``min_covariance_eigenvalue <= 0.5`` (the weights have
            infinite variance). The reported standard error is then unreliable; it
            typically understates the error.
        min_covariance_eigenvalue: Smallest eigenvalue of the final proposal
            covariance (``1`` for the mean-shift default).

    References:
        [REF-RUBINSTEIN-KROESE-2004], [REF-KURTZ-SONG-2013].

    """

    limit_state: str
    probability: float
    standard_error: float
    coefficient_of_variation: float
    thresholds: Vector
    mean: Vector
    covariance: Matrix
    levels: int
    samples_per_level: int
    final_samples: int
    failures: int
    effective_sample_size: float
    evaluations: int
    converged: bool
    failure_effective_sample_size: float = math.nan
    weights_degenerate: bool = False
    min_covariance_eigenvalue: float = 1.0


CE_MIN_FAILURE_ESS_FRACTION = 0.01
"""Failure-weight ESS fraction below which a CE-IS result is flagged."""

CE_COVARIANCE_FLOOR = 0.55
"""Default lower bound on the eigenvalues of an adapted CE proposal."""


def _kish_effective_size(weights: Sequence[float]) -> float:
    """Return Kish's effective sample size ``(sum w)^2 / sum w^2``.

    References:
        [EQ-MC-ESS-001], [REF-OWEN-2013, Chapter 9].

    """
    squared = math.fsum(w * w for w in weights)
    return math.fsum(weights) ** 2 / squared if squared > 0.0 else 0.0


def _floor_eigenvalues(matrix: Matrix, floor: float) -> Matrix:
    """Return ``V diag(max(lambda_i, floor)) V'`` for a symmetric matrix.

    References:
        [EQ-REL-CE-003], [EQ-LINALG-005], [REF-GOLUB-VANLOAN-2013, Section 8.5].

    """
    values, vectors = symmetric_eigen(matrix)
    if min(values) >= floor:
        return matrix
    size = len(matrix)
    clipped = [max(value, floor) for value in values]
    rebuilt = [
        [
            math.fsum(vectors[i][k] * clipped[k] * vectors[j][k] for k in range(size))
            for j in range(size)
        ]
        for i in range(size)
    ]
    return tuple(
        tuple(0.5 * (rebuilt[i][j] + rebuilt[j][i]) for j in range(size))
        for i in range(size)
    )


def _gaussian_draw(
    mean: Vector, lower: Matrix, generator_gauss: Callable[[], float]
) -> tuple[Vector, Vector]:
    """Return ``(u, z)`` with ``u = mean + L z`` and ``z`` standard normal.

    References:
        [EQ-DIST-MVN-005], [REF-RUBINSTEIN-KROESE-2004].

    """
    dimension = len(mean)
    z = tuple(generator_gauss() for _ in range(dimension))
    u = tuple(
        mean[i] + math.fsum(lower[i][k] * z[k] for k in range(i + 1))
        for i in range(dimension)
    )
    return u, z


def _log_weight(u: Vector, z: Vector, log_det_lower: float) -> float:
    """Return ``log phi_n(u) - log N(u; mu, L L')`` from ``u`` and ``z``.

    ``log N(u; mu, LL') = -||z||^2 / 2 - log det L - (n/2) log 2 pi`` with
    ``z = L^-1 (u - mu)``, so the normalizing constants cancel.

    References:
        [EQ-REL-CE-002], [EQ-DIST-MVN-001].

    """
    return -0.5 * dot(u, u) + 0.5 * dot(z, z) + log_det_lower


def cross_entropy_importance_sampling(
    limit_state: ModelLimitState,
    *,
    samples_per_level: int = 1000,
    level_quantile: float = 0.1,
    final_samples: int | None = None,
    adapt_covariance: bool = False,
    max_levels: int = 20,
    initial_mean: Sequence[float] | None = None,
    covariance_floor: float | None = CE_COVARIANCE_FLOOR,
    seed: int | RandomStream | None = None,
) -> CrossEntropyResult:
    """Estimate ``P_f`` by cross-entropy adaptive importance sampling.

    Start from ``h_0 = N(mu_0, I)`` (``mu_0 = 0`` by default). At level ``t``
    draw ``N`` points from ``h_{t-1}``, set the intermediate threshold
    ``gamma_t = max(q_rho, 0)`` where ``q_rho`` is the ``rho`` sample quantile
    of ``G``, and update the proposal by the weighted moments of the elite
    points ``G <= gamma_t`` with likelihood ratios ``W = phi_n(u) /
    h_{t-1}(u)``::

        mu_t = sum W u / sum W,   Sigma_t = sum W (u - mu_t)(u - mu_t)' / sum W

    (the covariance is kept at ``I`` unless ``adapt_covariance``). Once
    ``gamma_t = 0`` the last update uses failure points and a fresh sample of
    ``final_samples`` points from ``h_t`` gives
    ``P_f = mean(I(G <= 0) phi_n(u) / h_t(u))``.

    Weight-variance guard: with ``h = N(mu, Sigma)`` the second moment
    ``E_h[w^2] = int phi_n^2 / h`` is finite only when every eigenvalue of
    ``Sigma`` exceeds ``1/2``. The elite covariance of a near-planar failure
    domain collapses along the important direction, which makes the weights
    heavy-tailed: the estimate is then biased low in practice and the c.o.v.
    understates the error several-fold. Adapted covariances are therefore
    projected onto ``lambda_min >= covariance_floor`` (default ``0.55``) at
    every level. Independently of the guard, a :class:`UserWarning` is
    emitted and ``weights_degenerate`` set when the final proposal has an
    eigenvalue ``<= 1/2`` (possible only with ``covariance_floor=None``) or
    when the failure-weight effective sample size falls below
    ``CE_MIN_FAILURE_ESS_FRACTION = 0.01`` of ``final_samples``. A Kish
    effective sample size alone cannot detect the collapse: a heavy weight
    tail that was never sampled leaves the observed weights looking healthy.

    Args:
        limit_state: Model limit state.
        samples_per_level: Samples ``N`` per adaptation level (at least 10).
        level_quantile: Quantile ``rho`` in ``(0, 0.5]`` defining the levels.
        final_samples: Final importance-sampling size (default ``N``).
        adapt_covariance: Whether to adapt the full covariance as well as the
            mean; needs at least ``d + 1`` elite points per level. Caveat: the
            elite covariance shrinks along the important direction, so on
            near-planar limit states the levels can stall and the final
            weights can be heavy-tailed, so the c.o.v. may understate the
            error; see the weight-variance guard above. The mean-shift
            default is the robust choice.
        max_levels: Maximum adaptation levels before stopping unconverged.
        initial_mean: Optional starting proposal mean.
        covariance_floor: Lower bound on the eigenvalues of the adapted
            covariance (must exceed ``0.5``), or ``None`` to disable the guard
            (unguarded adaptation, kept only to study the failure mode).
            Ignored unless ``adapt_covariance``.
        seed: Integer seed, random stream, or ``None`` for fresh entropy.

    Returns:
        Estimate, standard error, c.o.v., thresholds, final proposal, weight
        diagnostics, evaluation count, and convergence flag. An unconverged
        run still performs the final stage with the last proposal.

    Raises:
        TypeError: If ``limit_state`` has the wrong type.
        ValueError: If options are invalid or all elite weights underflow.

    Warns:
        UserWarning: If the final proposal covariance has an eigenvalue
            ``<= 1/2`` or the final failure-weight effective sample size is
            below ``CE_MIN_FAILURE_ESS_FRACTION * final_samples``.

    Equation:
        [EQ-REL-CE-001], [EQ-REL-CE-002], [EQ-REL-CE-003], [EQ-MC-ESS-001].

    References:
        [REF-RUBINSTEIN-KROESE-2004], [REF-KURTZ-SONG-2013].

    """
    state = _require_limit_state(limit_state)
    count = _require_count(samples_per_level, "samples_per_level", 10)
    rho = require_finite_real(level_quantile, name="level_quantile")
    if not 0.0 < rho <= 0.5:
        raise ValueError("level_quantile must lie in (0, 0.5]")
    final_count = _require_count(
        count if final_samples is None else final_samples, "final_samples", 2
    )
    levels_limit = _require_count(max_levels, "max_levels", 1)
    floor: float | None = None
    if covariance_floor is not None:
        floor = require_finite_real(covariance_floor, name="covariance_floor")
        if floor <= 0.5:
            raise ValueError("covariance_floor must exceed 0.5")
    dimension = state.dimension
    elite_count = max(1, math.ceil(rho * count))
    if adapt_covariance and elite_count < dimension + 1:
        raise ValueError(
            "adapt_covariance needs ceil(rho N) >= d + 1 elite points per level"
        )
    if initial_mean is None:
        mean: Vector = tuple(0.0 for _ in range(dimension))
    else:
        mean = as_vector(initial_mean, name="initial_mean")
        if len(mean) != dimension:
            raise ValueError("initial_mean length must equal the input dimension")
    covariance: Matrix = tuple(
        tuple(1.0 if i == j else 0.0 for j in range(dimension))
        for i in range(dimension)
    )
    lower: Matrix = covariance
    log_det_lower = 0.0
    generator = resolve_stream(seed).child("cross-entropy").generator()

    def gauss() -> float:
        """Return one standard normal variate from the run's generator.

        References:
            [REF-MATSUMOTO-NISHIMURA-1998].

        """
        return generator.gauss(0.0, 1.0)

    before = state.model.evaluations
    thresholds: list[float] = []
    converged = False
    levels = 0
    for _ in range(levels_limit):
        levels += 1
        draws = [_gaussian_draw(mean, lower, gauss) for _ in range(count)]
        values = [state.standard(u) for u, _ in draws]
        ordered = sorted(values)
        threshold = max(ordered[elite_count - 1], 0.0)
        thresholds.append(threshold)
        elite = [
            (u, _log_weight(u, z, log_det_lower))
            for (u, z), value in zip(draws, values, strict=True)
            if value <= threshold
        ]
        shift = max(log_w for _, log_w in elite)
        weights = [math.exp(log_w - shift) for _, log_w in elite]
        total = math.fsum(weights)
        if not total > 0.0:
            raise ValueError("cross-entropy weights underflowed at every elite point")
        updated: Vector = tuple(
            math.fsum(w * u[i] for (u, _), w in zip(elite, weights, strict=True))
            / total
            for i in range(dimension)
        )
        if adapt_covariance:
            covariance = tuple(
                tuple(
                    math.fsum(
                        w * (u[i] - updated[i]) * (u[j] - updated[j])
                        for (u, _), w in zip(elite, weights, strict=True)
                    )
                    / total
                    + (_RIDGE if i == j else 0.0)
                    for j in range(dimension)
                )
                for i in range(dimension)
            )
            if floor is not None:
                covariance = _floor_eigenvalues(covariance, floor)
            try:
                lower = cholesky(covariance)
            except ValueError as error:
                raise ValueError(
                    "adapted cross-entropy covariance is not positive definite"
                ) from error
            log_det_lower = math.fsum(math.log(lower[i][i]) for i in range(dimension))
        mean = updated
        if threshold == 0.0:
            converged = True
            break
    contributions: list[float] = []
    weights_all: list[float] = []
    failures = 0
    for _ in range(final_count):
        u, z = _gaussian_draw(mean, lower, gauss)
        weight = math.exp(_log_weight(u, z, log_det_lower))
        weights_all.append(weight)
        if state.standard(u) <= 0.0:
            failures += 1
            contributions.append(weight)
        else:
            contributions.append(0.0)
    probability, standard_error = _mean_and_error(contributions)
    failure_ess = _kish_effective_size([w for w in contributions if w > 0.0])
    smallest = min(symmetric_eigen(covariance)[0]) if adapt_covariance else 1.0
    reasons: list[str] = []
    if smallest <= 0.5:
        reasons.append(
            f"proposal eigenvalue {smallest:.3g} <= 0.5 gives infinite weight variance"
        )
    if failures > 0 and failure_ess < CE_MIN_FAILURE_ESS_FRACTION * final_count:
        reasons.append(
            f"failure-weight effective sample size {failure_ess:.1f} is below "
            f"{CE_MIN_FAILURE_ESS_FRACTION:g} of {final_count} final samples"
        )
    degenerate = bool(reasons)
    if degenerate:
        warnings.warn(
            "CE-IS weights are degenerate ("
            + "; ".join(reasons)
            + "); the estimate and its standard error are unreliable",
            UserWarning,
            stacklevel=2,
        )
    return CrossEntropyResult(
        limit_state=state.label,
        probability=probability,
        standard_error=standard_error,
        coefficient_of_variation=(
            standard_error / probability if probability > 0.0 else math.inf
        ),
        thresholds=tuple(thresholds),
        mean=mean,
        covariance=covariance,
        levels=levels,
        samples_per_level=count,
        final_samples=final_count,
        failures=failures,
        effective_sample_size=_kish_effective_size(weights_all),
        evaluations=state.model.evaluations - before,
        converged=converged,
        failure_effective_sample_size=failure_ess,
        weights_degenerate=degenerate,
        min_covariance_eigenvalue=smallest,
    )


@dataclass(frozen=True, slots=True)
class LineSamplingResult:
    """Line-sampling estimate of a failure probability.

    Args:
        limit_state: Label of the analyzed limit state.
        probability: Mean of the per-line probabilities.
        standard_error: ``sqrt(sum (p_k - P)^2 / (N (N - 1)))``.
        coefficient_of_variation: ``standard_error / probability``.
        direction: Unit important direction ``alpha`` in standard-normal space.
        line_probabilities: Per-line conditional probabilities ``p_k``.
        distances: Signed crossing distance ``c_k`` per line (``inf`` when
            the line stays safe within ``max_distance``, ``-inf`` when it fails
            everywhere searched).
        lines: Number of lines ``N``.
        evaluations: Model evaluations for the line searches.
        form_evaluations: Model evaluations spent on FORM (zero when a
            direction or FORM result was supplied).

    References:
        [REF-KOUTSOURELAKIS-PRADLWARTER-SCHUELLER-2004].

    """

    limit_state: str
    probability: float
    standard_error: float
    coefficient_of_variation: float
    direction: Vector
    line_probabilities: Vector
    distances: Vector
    lines: int
    evaluations: int
    form_evaluations: int


def _line_probability(
    function: Callable[[float], float],
    start: float,
    max_distance: float,
    tolerance: float,
) -> tuple[float, float]:
    """Locate the crossing on one line and return ``(p_k, c_k)``.

    The line ``c -> G(u_perp + c alpha)`` is assumed to cross zero at most
    once with failure on the side of increasing ``c``. A bracket is grown from
    ``start`` in doubling steps; Brent's method refines it. Values are cached,
    so Brent's endpoint evaluations are free. A line that is still safe at
    ``c = max_distance`` returns ``(0.0, inf)`` and one that already fails at
    ``c = -max_distance`` returns ``(1.0, -inf)``: exactly 0 or 1, not
    ``Phi(-+max_distance)``.

    References:
        [EQ-REL-LINE-001], [EQ-OPT-BRENT-001],
        [REF-KOUTSOURELAKIS-PRADLWARTER-SCHUELLER-2004], [REF-BRENT-1973].

    """
    cache: dict[float, float] = {}

    def cached(c: float) -> float:
        """Evaluate the line function once per abscissa.

        References:
            [REF-BRENT-1973]. No equation is evaluated.

        """
        if c not in cache:
            cache[c] = function(c)
        return cache[c]

    begin = min(max(start, -max_distance), max_distance)
    value = cached(begin)
    step = 0.5
    previous = begin
    if value > 0.0:
        while True:
            candidate = min(previous + step, max_distance)
            if cached(candidate) <= 0.0:
                lower_end, upper_end = previous, candidate
                break
            if candidate >= max_distance:
                return 0.0, math.inf
            previous, step = candidate, 2.0 * step
    else:
        while True:
            candidate = max(previous - step, -max_distance)
            if cached(candidate) > 0.0:
                lower_end, upper_end = candidate, previous
                break
            if candidate <= -max_distance:
                return 1.0, -math.inf
            previous, step = candidate, 2.0 * step
    root = brent_root(cached, lower_end, upper_end, tolerance=tolerance).root
    return standard_normal_cdf(-root), root


def line_sampling(
    limit_state: ModelLimitState,
    lines: int = 100,
    *,
    direction: Sequence[float] | None = None,
    form_result: FORMResult | None = None,
    max_distance: float = 10.0,
    tolerance: float = 1e-6,
    seed: int | RandomStream | None = None,
) -> LineSamplingResult:
    """Estimate ``P_f`` by line sampling along an important direction.

    For each line draw ``u ~ N(0, I)``, project ``u_perp = u - (alpha . u)
    alpha``, solve ``G(u_perp + c_k alpha) = 0`` for ``c_k``, and set
    ``p_k = Phi(-c_k)``. Then ``P_f = mean(p_k)`` with variance
    ``sum (p_k - P_f)^2 / (N (N - 1))``. The direction defaults to the FORM
    ``alpha`` (``-grad G / ||grad G||`` at the design point), which points
    into the failure domain.

    Args:
        limit_state: Model limit state.
        lines: Number of lines ``N`` (at least two).
        direction: Optional important direction (normalized internally).
        form_result: Optional FORM result supplying ``alpha`` and the starting
            distance ``beta``.
        max_distance: Search half-width along each line. A line with no
            crossing in ``|c| <= max_distance`` contributes exactly ``0`` when
            it is safe at ``c = max_distance`` (distance recorded as ``inf``)
            and exactly ``1`` when it fails at ``c = -max_distance`` (distance
            ``-inf``); the neglected mass is at most ``Phi(-max_distance)``
            per line (about ``7.6e-24`` for the default ``10``).
        tolerance: Absolute Brent tolerance on ``c_k``.
        seed: Integer seed, random stream, or ``None`` for fresh entropy.

    Returns:
        Estimate, standard error, c.o.v., direction, per-line values, and
        evaluation counts.

    Raises:
        TypeError: If arguments have the wrong type.
        ValueError: If options are invalid or the direction is zero or has
            the wrong length.

    Equation:
        [EQ-REL-LINE-001], [EQ-OPT-BRENT-001].

    References:
        [REF-KOUTSOURELAKIS-PRADLWARTER-SCHUELLER-2004], [REF-BRENT-1973].

    """
    state = _require_limit_state(limit_state)
    count = _require_count(lines, "lines", 2)
    reach = require_finite_real(max_distance, name="max_distance")
    if reach <= 0.0:
        raise ValueError("max_distance must be positive")
    tol = require_finite_real(tolerance, name="tolerance")
    if tol <= 0.0:
        raise ValueError("tolerance must be positive")
    form_evaluations = 0
    start = 3.0
    if direction is not None:
        raw = as_vector(direction, name="direction")
    else:
        if form_result is None:
            form_result = form_hlrf(state)
            form_evaluations = form_result.evaluations
        elif not isinstance(form_result, FORMResult):
            raise TypeError("form_result must be a FORMResult")
        raw = form_result.alpha
        start = form_result.beta
    if len(raw) != state.dimension:
        raise ValueError("direction length must equal the input dimension")
    size = norm(raw)
    if not size > 0.0:
        raise ValueError("direction must be non-zero")
    alpha = tuple(value / size for value in raw)
    generator = resolve_stream(seed).child("line-sampling").generator()
    before = state.model.evaluations
    probabilities: list[float] = []
    distances: list[float] = []
    for _ in range(count):
        u = tuple(generator.gauss(0.0, 1.0) for _ in alpha)
        projection = dot(alpha, u)
        perpendicular = tuple(
            ui - projection * ai for ui, ai in zip(u, alpha, strict=True)
        )

        def along(c: float, base: Vector = perpendicular) -> float:
            """Evaluate ``G(u_perp + c alpha)``.

            References:
                [EQ-REL-LINE-001], [REF-KOUTSOURELAKIS-PRADLWARTER-SCHUELLER-2004].

            """
            return state.standard(
                tuple(b + c * a for b, a in zip(base, alpha, strict=True))
            )

        probability, distance = _line_probability(along, start, reach, tol)
        probabilities.append(probability)
        distances.append(distance)
        if math.isfinite(distance):
            start = distance
    estimate = math.fsum(probabilities) / count
    variance = math.fsum((p - estimate) ** 2 for p in probabilities) / (
        count * (count - 1)
    )
    standard_error = math.sqrt(variance)
    return LineSamplingResult(
        limit_state=state.label,
        probability=estimate,
        standard_error=standard_error,
        coefficient_of_variation=(
            standard_error / estimate if estimate > 0.0 else math.inf
        ),
        direction=alpha,
        line_probabilities=tuple(probabilities),
        distances=tuple(distances),
        lines=count,
        evaluations=state.model.evaluations - before,
        form_evaluations=form_evaluations,
    )
