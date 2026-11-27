"""Local-approximation MCMC for expensive forward models.

Experimental (added in 0.28).

Setting
    Bayesian inverse problem ``y = G(theta) + eps``, ``eps ~ N(0, diag(s^2))``,
    with a cheap prior density ``p0`` and an expensive forward map ``G``.
    A Metropolis chain needs ``G`` at every proposal. Conrad, Marzouk, Pillai
    and Smith (2016) replace ``G`` by a *local* regression fitted to the
    stored true evaluations nearest to the query point and refine the store
    during the run, so the chain stays asymptotically exact while the number
    of true evaluations grows far more slowly than the chain length.

Local regression ([EQ-INV-LOCAL-001])
    For a query ``x`` take the ``k`` stored points ``x_i`` nearest to ``x`` in
    the metric ``||(x_i - x) / s||`` (``s`` positive length scales; ties are
    broken by store order). With ``h = 1.1 max_i d_i`` and ``d_i`` the
    distances, the weights are the tricube kernel
    ``w_i = (1 - (d_i / h)^3)^3 > 0``. The model is a polynomial of order 1
    (local linear, ``p = 1 + n`` terms) or 2 (local quadratic,
    ``p = (n + 1)(n + 2) / 2`` terms) in the local coordinates
    ``u_i = (x_i - x) / (s h)``; the weighted least-squares fit of every output
    component shares one design ``B = diag(sqrt w) A``. The value at ``x`` is
    the intercept, a linear smoother ``sum_i l_i G(x_i)``.

    Minimum neighbours: ``k >= p + 1``, so that the fit has at least one
    residual degree of freedom and leave-one-out is defined. Until the store
    holds ``k`` points no approximation exists and the point is refined.
    Well-posedness: the fit is accepted only if ``cond(B) = sqrt(lambda_max /
    lambda_min)`` of ``B' B`` is finite and at most ``condition_limit``;
    otherwise the point is refined. No jitter, ridge or pseudo-inverse is used.

Cross-validation indicator ([EQ-INV-LOCAL-002])
    With ``C = (B' B)^-1``, hat diagonal ``H_jj = b_j' C b_j`` and weighted
    residual ``r_j = sqrt(w_j) (G(x_j) - a_j' beta)``, deleting neighbour
    ``j`` moves the fitted value by
    ``delta_j = (c_0' b_j) r_j / (1 - H_jj)`` (exact leave-one-out update of
    weighted least squares with fixed weights; ``c_0`` is the first row of
    ``C``). The indicator is
    ``eps(x) = max_j |l(G_hat) - l(G_hat - delta_j)|`` with ``l`` the
    log-likelihood: the largest change of the log-acceptance factor caused by
    dropping one neighbour. If ``H_jj`` is within ``1e-12`` of 1 the
    indicator is ``+inf`` (the fit depends on a single point).

Refinement ([EQ-INV-LOCAL-003]); every refinement is a TRUE evaluation
    * *Forced*: no approximation exists (too few points or ill-conditioned
      design) at the proposal or the current state.
    * *Random*: at iteration ``t`` (``t = 1, 2, ...``) the proposal is
      evaluated with probability ``beta_t = min(1, beta_0 t^-gamma)``.
    * *Cross-validation*: if ``eps(x') + eps(x) > cv_tolerance`` the point
      with the larger indicator is evaluated; at most two refinements per
      iteration (after both points are stored both are exact and
      the indicator is zero).
    Stored points are used exactly (no regression at a stored point).

Acceptance ([EQ-INV-LOCAL-004])
    Symmetric Gaussian random walk; accept with probability
    ``min(1, exp[log p0(x') + l_hat(x') - log p0(x) - l_hat(x)])`` where both
    ``l_hat`` use the store *after* the refinements of this iteration.

Exactness: what is and is not guaranteed
    The chain is adaptive: the surrogate changes with the history, so the
    kernel is not time homogeneous and the detailed-balance argument for a
    fixed surrogate does not apply. The argument of Conrad et al. (2016) is
    asymptotic: (i) refinement never stops in the limit, which holds if
    ``sum_t beta_t = infinity``, i.e. ``beta_0 > 0`` and ``gamma <= 1``
    (:attr:`LocalApproximationChainResult.refinement_never_stops`); (ii) the
    store becomes dense on the region visited by the chain, so the local
    error and with it the change of the kernel (diminishing adaptation)
    vanishes; (iii) the exact kernel is ergodic. I have not re-derived the
    theorem's exact hypotheses in this session; they are not checked by the
    code. NOT guaranteed at finite length: the chain's law differs from the
    posterior by an unquantified surrogate error, the cross-validation
    indicator can be small by accident (a smooth wrong fit), and with
    ``beta_0 = 0``, ``gamma > 1`` or a frozen store the limit is biased
    (measured in the tests). Mixing, burn-in and the chain length needed are
    the user's responsibility.

Delayed acceptance ([EQ-INV-LOCAL-005])
    :func:`local_approximation_delayed_acceptance` freezes a local
    approximation built from a design plus the start point and uses it as the
    cheap stage of :func:`probstat.inference.delayed_acceptance_metropolis`;
    the second stage evaluates ``G`` for every stage-1 acceptance. Exact for
    any fixed surrogate whose support contains the target's. A frozen local
    surrogate must be a deterministic function of ``x``: where the quadratic
    fit is ill-posed it uses the linear fit, then the nearest stored value;
    these calls are counted. Surrogate error only costs efficiency.

Accounting ([EQ-INV-LOCAL-006])
    ``true_evaluations`` counts every call of ``G`` in the run. The ESS
    per true evaluation is the smallest per-coordinate ESS of the kept draws
    divided by it.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.inference.delayed_acceptance import (
    DelayedAcceptanceResult,
    delayed_acceptance_metropolis,
)
from probstat.inference.mcmc import effective_sample_size
from probstat.linalg import Vector, symmetric_eigen

ForwardMap = Callable[[Vector], Sequence[float]]
LogPrior = Callable[[Vector], float]

__all__ = [
    "LocalApproximationChainResult",
    "LocalDelayedAcceptanceResult",
    "LocalFit",
    "RefinementEvent",
    "local_approximation_delayed_acceptance",
    "local_approximation_metropolis",
    "local_weighted_fit",
]

_LEVERAGE_TOLERANCE = 1e-12
_BANDWIDTH_FACTOR = 1.1


@dataclass(frozen=True, slots=True)
class LocalFit:
    """Result of one local weighted least-squares fit.

    Attributes:
        status: ``"ok"``, ``"insufficient_points"`` or ``"ill_conditioned"``.
        value: Fitted output at the query (``None`` unless ``status == "ok"``).
        loo_shifts: For each neighbour ``j``, the change ``delta_j`` of the
            fitted value caused by deleting ``j`` (empty unless ok). A
            leverage-one neighbour yields ``inf`` entries.
        condition: ``cond`` of the weighted design (``nan`` if too few points).
        neighbours: Indices of the neighbours used, nearest first.
        weights: Tricube weights of those neighbours.
        order: Polynomial order (1 or 2).

    Equation:
        [EQ-INV-LOCAL-001], [EQ-INV-LOCAL-002].

    References:
        [REF-CONRAD-MARZOUK-PILLAI-SMITH-2016].

    """

    status: str
    value: Vector | None
    loo_shifts: tuple[Vector, ...]
    condition: float
    neighbours: tuple[int, ...]
    weights: tuple[float, ...]
    order: int


@dataclass(frozen=True, slots=True)
class RefinementEvent:
    """One true evaluation added to the store during a chain.

    Attributes:
        iteration: 1-based iteration at which it happened.
        reason: ``"initial_design"``-free labels ``"insufficient_points"``,
            ``"ill_conditioned"``, ``"random"`` or ``"cross_validation"``.
        point: Evaluated parameter vector.
        indicator: Cross-validation indicator that triggered it (``nan`` if
            not applicable).

    Equation:
        [EQ-INV-LOCAL-003].

    References:
        [REF-CONRAD-MARZOUK-PILLAI-SMITH-2016].

    """

    iteration: int
    reason: str
    point: Vector
    indicator: float


@dataclass(frozen=True, slots=True)
class LocalApproximationChainResult:
    """Output of :func:`local_approximation_metropolis`.

    Attributes:
        samples: Kept draws after burn-in.
        acceptance_rate: Fraction of all iterations ending in a move.
        iterations: Burn-in plus kept iterations.
        true_evaluations: Calls of ``G`` in the whole run (design included).
        evaluations_per_iteration: ``true_evaluations / iterations``; plain
            Metropolis costs about 1.
        cumulative_true_evaluations: ``G`` calls made up to each kept draw.
        store_size: Number of stored true evaluations at the end.
        store_points: The stored parameter vectors.
        refinement_log: Refinements made during the iterations.
        refinement_counts: Count per reason.
        effective_sample_size: Per-coordinate ESS of the kept draws.
        ess_per_true_evaluation: Smallest finite ESS over ``true_evaluations``.
        refinement_never_stops: ``beta_0 > 0 and gamma <= 1``; necessary for
            the asymptotic-exactness argument (not sufficient, not verified).
        settings: Mapping of the numerical settings used.
        seed: Seed of the random stream.

    Equation:
        [EQ-INV-LOCAL-003], [EQ-INV-LOCAL-006].

    References:
        [REF-CONRAD-MARZOUK-PILLAI-SMITH-2016].

    """

    samples: tuple[Vector, ...]
    acceptance_rate: float
    iterations: int
    true_evaluations: int
    evaluations_per_iteration: float
    cumulative_true_evaluations: tuple[int, ...]
    store_size: int
    store_points: tuple[Vector, ...]
    refinement_log: tuple[RefinementEvent, ...]
    refinement_counts: dict[str, int]
    effective_sample_size: tuple[float, ...]
    ess_per_true_evaluation: float
    refinement_never_stops: bool
    settings: dict[str, object]
    seed: int | None


@dataclass(frozen=True, slots=True)
class LocalDelayedAcceptanceResult:
    """Output of :func:`local_approximation_delayed_acceptance`.

    Attributes:
        chain: The underlying :class:`DelayedAcceptanceResult`.
        design_evaluations: True evaluations spent on the frozen store.
        true_evaluations: Design evaluations plus second-stage evaluations.
        store_size: Stored true evaluations used by the surrogate.
        fallback_calls: Surrogate calls that used a lower-order fit
            (ill-posed higher-order fit) as a dict ``{"linear": n,
            "nearest": n}``.
        ess_per_true_evaluation: Smallest finite ESS over ``true_evaluations``.
        exact_target: ``True``: exact for a fixed surrogate (see module).

    Equation:
        [EQ-INV-LOCAL-005], [EQ-INV-LOCAL-006].

    References:
        [REF-CHRISTEN-FOX-2005].

    """

    chain: DelayedAcceptanceResult
    design_evaluations: int
    true_evaluations: int
    store_size: int
    fallback_calls: dict[str, int]
    ess_per_true_evaluation: float
    exact_target: bool = True


def _basis_size(dimension: int, order: int) -> int:
    """Return the number of polynomial terms ``p``.

    References:
        [EQ-INV-LOCAL-001].

    """
    if order == 1:
        return 1 + dimension
    return (dimension + 1) * (dimension + 2) // 2


def _basis(local: Sequence[float], order: int) -> list[float]:
    """Return ``[1, u, u_j u_l (j <= l)]`` for local coordinates ``u``.

    References:
        [EQ-INV-LOCAL-001].

    """
    row = [1.0, *local]
    if order == 2:
        n = len(local)
        for j in range(n):
            for ell in range(j, n):
                row.append(local[j] * local[ell])
    return row


def _require_order(order: object) -> int:
    """Validate the polynomial order.

    References:
        [EQ-INV-LOCAL-001].

    """
    if isinstance(order, bool) or order not in (1, 2):
        raise ValueError("order must be 1 (local linear) or 2 (local quadratic)")
    return int(order)


def _positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    """Validate an integer ``>= minimum``.

    References:
        [EQ-INV-LOCAL-006].

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _scales(length_scales: float | Sequence[float] | None, n: int) -> tuple[float, ...]:
    """Return positive per-coordinate length scales.

    References:
        [EQ-INV-LOCAL-001].

    """
    if length_scales is None:
        return (1.0,) * n
    if isinstance(length_scales, Sequence):
        values = tuple(
            require_finite_real(s, name="length_scales") for s in length_scales
        )
        if len(values) != n:
            raise ValueError(
                "length_scales must be a scalar or have one entry per axis"
            )
    else:
        values = (require_finite_real(length_scales, name="length_scales"),) * n
    if any(s <= 0.0 for s in values):
        raise ValueError("length_scales must be positive")
    return values


def _vector(values: Sequence[float], name: str) -> Vector:
    """Return a finite float tuple.

    References:
        [EQ-INV-LOCAL-001].

    """
    result = tuple(require_finite_real(v, name=name) for v in values)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def local_weighted_fit(
    points: Sequence[Sequence[float]],
    values: Sequence[Sequence[float]],
    query: Sequence[float],
    *,
    order: int = 2,
    neighbours: int | None = None,
    length_scales: float | Sequence[float] | None = None,
    condition_limit: float = 1e6,
) -> LocalFit:
    """Fit a local polynomial to the nearest stored evaluations.

    Implements the tricube-weighted local regression and the exact
    leave-one-out shifts of the module docstring. The query is treated as a
    new point: if it coincides with a stored point the distance is zero and
    the weight is one (the fit is not forced to interpolate).

    Args:
        points: Stored parameter vectors (at least one, equal dimension).
        values: Stored outputs ``G(x_i)``, one vector per point, equal length.
        query: Query point.
        order: ``1`` (local linear) or ``2`` (local quadratic).
        neighbours: Number ``k`` of nearest points; default ``2 p``. Must be
            at least ``p + 1``.
        length_scales: Positive scalar or per-axis scales of the metric.
        condition_limit: Largest accepted ``cond`` of the weighted design.

    Returns:
        A :class:`LocalFit`. ``status`` is ``"insufficient_points"`` if fewer
        than ``k`` points are stored and ``"ill_conditioned"`` if the design
        is rank deficient or ``cond`` exceeds ``condition_limit``.

    Raises:
        TypeError: For non-integer ``neighbours``.
        ValueError: For inconsistent dimensions, non-finite entries, invalid
            ``order``, ``neighbours < p + 1``, non-positive scales or
            ``condition_limit <= 1``.

    Equation:
        [EQ-INV-LOCAL-001], [EQ-INV-LOCAL-002].

    References:
        [REF-CONRAD-MARZOUK-PILLAI-SMITH-2016].

    """
    order_value = _require_order(order)
    x = _vector(query, "query")
    n = len(x)
    if len(points) != len(values):
        raise ValueError("points and values must have the same length")
    if not points:
        raise ValueError("points must be non-empty")
    pts = [_vector(p, "points") for p in points]
    vals = [_vector(v, "values") for v in values]
    if any(len(p) != n for p in pts):
        raise ValueError("every stored point must match the query dimension")
    if len({len(v) for v in vals}) != 1:
        raise ValueError("every stored value must have the same length")
    basis_size = _basis_size(n, order_value)
    k = (
        2 * basis_size
        if neighbours is None
        else _positive_int(neighbours, "neighbours")
    )
    if k < basis_size + 1:
        raise ValueError(
            f"neighbours must be at least p + 1 = {basis_size + 1} for order "
            f"{order_value} in dimension {n}"
        )
    limit = require_finite_real(condition_limit, name="condition_limit")
    if limit <= 1.0:
        raise ValueError("condition_limit must exceed 1")
    scale = _scales(length_scales, n)
    return _fit(pts, vals, x, order_value, k, scale, limit)


def _fit(
    pts: Sequence[Vector],
    vals: Sequence[Vector],
    x: Vector,
    order: int,
    k: int,
    scale: Sequence[float],
    limit: float,
) -> LocalFit:
    """Fit after validation; the worker of :func:`local_weighted_fit`.

    References:
        [EQ-INV-LOCAL-001], [EQ-INV-LOCAL-002].

    """
    if len(pts) < k:
        return LocalFit("insufficient_points", None, (), math.nan, (), (), order)
    n = len(x)
    dist = [
        math.sqrt(
            math.fsum(((a - b) / s) ** 2 for a, b, s in zip(p, x, scale, strict=True))
        )
        for p in pts
    ]
    chosen = heapq.nsmallest(k, range(len(pts)), key=lambda i: (dist[i], i))
    d = [dist[i] for i in chosen]
    bandwidth = _BANDWIDTH_FACTOR * max(d)
    if bandwidth <= 0.0:
        return LocalFit("ill_conditioned", None, (), math.inf, tuple(chosen), (), order)
    w = [(1.0 - (di / bandwidth) ** 3) ** 3 for di in d]
    root = [math.sqrt(wi) for wi in w]
    rows: list[list[float]] = []
    for idx, rt in zip(chosen, root, strict=True):
        local = [
            (a - b) / (s * bandwidth)
            for a, b, s in zip(pts[idx], x, scale, strict=True)
        ]
        rows.append([rt * entry for entry in _basis(local, order)])
    p = len(rows[0])
    gram = [[math.fsum(r[a] * r[b] for r in rows) for b in range(p)] for a in range(p)]
    for a in range(p):
        for b in range(a):
            gram[a][b] = gram[b][a] = 0.5 * (gram[a][b] + gram[b][a])
    eigenvalues, vectors = symmetric_eigen(tuple(tuple(r) for r in gram))
    smallest, largest = eigenvalues[0], eigenvalues[-1]
    if smallest <= 0.0 or largest <= 0.0:
        return LocalFit(
            "ill_conditioned", None, (), math.inf, tuple(chosen), tuple(w), order
        )
    condition = math.sqrt(largest / smallest)
    if not condition <= limit:
        return LocalFit(
            "ill_conditioned", None, (), condition, tuple(chosen), tuple(w), order
        )
    inverse = [
        [
            math.fsum(vectors[a][q] * vectors[b][q] / eigenvalues[q] for q in range(p))
            for b in range(p)
        ]
        for a in range(p)
    ]
    outputs = len(vals[0])
    # beta_o = C sum_i b_i sqrt(w_i) g_io ; intercept = beta_o[0].
    rhs = [
        [
            math.fsum(rows[i][a] * root[i] * vals[chosen[i]][o] for i in range(k))
            for a in range(p)
        ]
        for o in range(outputs)
    ]
    beta = [
        [math.fsum(inverse[a][b] * rhs[o][b] for b in range(p)) for a in range(p)]
        for o in range(outputs)
    ]
    value = tuple(beta[o][0] for o in range(outputs))
    shifts: list[Vector] = []
    for i in range(k):
        c_b = [
            math.fsum(inverse[a][b] * rows[i][b] for b in range(p)) for a in range(p)
        ]
        leverage = math.fsum(rows[i][a] * c_b[a] for a in range(p))
        gain = c_b[0]
        if 1.0 - leverage <= _LEVERAGE_TOLERANCE:
            shifts.append((math.inf,) * outputs)
            continue
        entries = []
        for o in range(outputs):
            fitted = math.fsum(rows[i][a] * beta[o][a] for a in range(p))
            residual = root[i] * vals[chosen[i]][o] - fitted
            entries.append(gain * residual / (1.0 - leverage))
        shifts.append(tuple(entries))
    del n
    return LocalFit(
        "ok", value, tuple(shifts), condition, tuple(chosen), tuple(w), order
    )


class _Problem:
    """Gaussian-noise log-likelihood and counted true forward map.

    References:
        [EQ-INV-LOCAL-004], [EQ-INV-LOCAL-006].

    """

    __slots__ = ("calls", "data", "forward", "log_prior", "noise")

    def __init__(
        self,
        forward: ForwardMap,
        data: Sequence[float],
        noise_std: float | Sequence[float],
        log_prior: LogPrior,
    ) -> None:
        """Validate and store the problem.

        Args:
            forward: Expensive forward map.
            data: Observation of length ``m``.
            noise_std: Positive scalar or length-``m`` standard deviations.
            log_prior: Cheap log-prior (``-inf`` allowed, ``nan``/``+inf`` not).

        References:
            [EQ-INV-LOCAL-004].

        """
        if not callable(forward):
            raise TypeError("forward must be callable")
        if not callable(log_prior):
            raise TypeError("log_prior must be callable")
        self.forward = forward
        self.log_prior = log_prior
        self.data = _vector(data, "data")
        m = len(self.data)
        if isinstance(noise_std, Sequence):
            self.noise = tuple(
                require_finite_real(s, name="noise_std") for s in noise_std
            )
            if len(self.noise) != m:
                raise ValueError("noise_std must be a scalar or match the data length")
        else:
            self.noise = (require_finite_real(noise_std, name="noise_std"),) * m
        if any(s <= 0.0 for s in self.noise):
            raise ValueError("noise_std must be positive")
        self.calls = 0

    def true(self, point: Vector) -> Vector:
        """Evaluate and count the true forward map.

        References:
            [EQ-INV-LOCAL-006].

        """
        out = self.forward(point)
        self.calls += 1
        result = tuple(float(v) for v in out)
        if len(result) != len(self.data):
            raise ValueError("forward output length must match the data length")
        if not all(math.isfinite(v) for v in result):
            raise ValueError("forward returned a non-finite value")
        return result

    def loglik(self, output: Sequence[float]) -> float:
        """Return the Gaussian log-likelihood up to an additive constant.

        References:
            [EQ-INV-LOCAL-004].

        """
        return -0.5 * math.fsum(
            ((y - g) / s) ** 2
            for y, g, s in zip(self.data, output, self.noise, strict=True)
        )

    def prior(self, point: Vector) -> float:
        """Return the validated log-prior.

        References:
            [EQ-INV-LOCAL-004].

        """
        value = float(self.log_prior(point))
        if math.isnan(value) or value == math.inf:
            raise ValueError("log_prior returned NaN or +inf")
        return value

    def indicator(self, fit: LocalFit) -> float:
        """Return the cross-validation indicator of an ``ok`` fit.

        References:
            [EQ-INV-LOCAL-002].

        """
        assert fit.value is not None
        base = self.loglik(fit.value)
        worst = 0.0
        for shift in fit.loo_shifts:
            if any(math.isinf(s) for s in shift):
                return math.inf
            moved = self.loglik(
                tuple(v - s for v, s in zip(fit.value, shift, strict=True))
            )
            worst = max(worst, abs(base - moved))
        return worst


def _check_common(
    samples: object, burn_in: object, step: float | Sequence[float], n: int
) -> tuple[float, ...]:
    """Validate chain length arguments and return the step scales.

    References:
        [EQ-INV-LOCAL-004].

    """
    _positive_int(samples, "samples")
    _positive_int(burn_in, "burn_in", minimum=0)
    return _scales(step, n)


def local_approximation_metropolis(
    forward: ForwardMap,
    data: Sequence[float],
    *,
    noise_std: float | Sequence[float],
    log_prior: LogPrior,
    initial: Sequence[float],
    step: float | Sequence[float],
    samples: int,
    burn_in: int = 0,
    order: int = 2,
    neighbours: int | None = None,
    length_scales: float | Sequence[float] | None = None,
    condition_limit: float = 1e6,
    beta0: float = 1.0,
    gamma: float = 0.5,
    cv_tolerance: float | None = 0.1,
    initial_design: Sequence[Sequence[float]] = (),
    seed: int | RandomStream | None = None,
) -> LocalApproximationChainResult:
    """Run an adaptive Metropolis chain on a local approximation of ``G``.

    See the module docstring for the regression, the refinement rules, the
    acceptance rule and the limits of the exactness claim. ``forward`` is
    called only for stored points: the initial point, the initial design and
    every refinement; the call count is reported.

    Args:
        forward: Expensive map ``G`` from a parameter vector to ``m`` outputs.
        data: Observation ``y`` (length ``m``).
        noise_std: Positive scalar or per-output noise standard deviations.
        log_prior: Cheap log-prior of the parameters.
        initial: Start point; ``log_prior`` must be finite there.
        step: Positive random-walk scale (scalar or per coordinate).
        samples: Kept draws (positive).
        burn_in: Discarded iterations (non-negative).
        order: ``1`` or ``2``.
        neighbours: ``k`` nearest points; default ``2 p``; at least ``p + 1``.
        length_scales: Metric scales of the neighbour search (default 1).
        condition_limit: Largest accepted design condition number.
        beta0: Random refinement probability scale in ``[0, 1]``.
        gamma: Decay exponent ``>= 0`` of ``beta_t = beta0 t^-gamma``.
        cv_tolerance: Refine when the indicator sum exceeds it; ``None``
            disables cross-validation refinement.
        initial_design: Extra points evaluated truly before the chain starts.
        seed: Integer seed or stream.

    Returns:
        A :class:`LocalApproximationChainResult`.

    Raises:
        TypeError: For non-callable inputs or non-integer counts.
        ValueError: For invalid shapes, scales, probabilities, a start with
            ``-inf`` prior, duplicated design points, or forward outputs that
            are non-finite or of the wrong length.

    Equation:
        [EQ-INV-LOCAL-003], [EQ-INV-LOCAL-004], [EQ-INV-LOCAL-006].

    References:
        [REF-CONRAD-MARZOUK-PILLAI-SMITH-2016].

    """
    problem = _Problem(forward, data, noise_std, log_prior)
    start = _vector(initial, "initial")
    n = len(start)
    scales_step = _check_common(samples, burn_in, step, n)
    order_value = _require_order(order)
    basis_size = _basis_size(n, order_value)
    k = (
        2 * basis_size
        if neighbours is None
        else _positive_int(neighbours, "neighbours")
    )
    if k < basis_size + 1:
        raise ValueError(f"neighbours must be at least p + 1 = {basis_size + 1}")
    metric = _scales(length_scales, n)
    limit = require_finite_real(condition_limit, name="condition_limit")
    if limit <= 1.0:
        raise ValueError("condition_limit must exceed 1")
    b0 = require_finite_real(beta0, name="beta0")
    g = require_finite_real(gamma, name="gamma")
    if not 0.0 <= b0 <= 1.0:
        raise ValueError("beta0 must lie in [0, 1]")
    if g < 0.0:
        raise ValueError("gamma must be non-negative")
    tolerance: float | None = None
    if cv_tolerance is not None:
        tolerance = require_finite_real(cv_tolerance, name="cv_tolerance")
        if tolerance <= 0.0:
            raise ValueError("cv_tolerance must be positive")
    stream = resolve_stream(seed)
    rng = stream.generator()

    store_points: list[Vector] = []
    store_values: list[Vector] = []
    lookup: dict[Vector, Vector] = {}

    def add(point: Vector) -> Vector:
        """Evaluate ``G`` truly at ``point`` and store the result.

        References:
            [EQ-INV-LOCAL-003].

        """
        value = problem.true(point)
        lookup[point] = value
        store_points.append(point)
        store_values.append(value)
        return value

    if not math.isfinite(problem.prior(start)):
        raise ValueError("log_prior must be finite at the initial point")
    for design_point in initial_design:
        dp = _vector(design_point, "initial_design")
        if len(dp) != n:
            raise ValueError("initial_design points must match the dimension")
        if dp in lookup:
            raise ValueError("initial_design contains a duplicated point")
        if math.isfinite(problem.prior(dp)):
            add(dp)
    if start not in lookup:
        add(start)

    cache: dict[Vector, tuple[int, float, float, str]] = {}

    def surrogate(point: Vector) -> tuple[float, float, str]:
        """Return ``(l_hat, indicator, status)`` using the current store.

        References:
            [EQ-INV-LOCAL-002], [EQ-INV-LOCAL-004].

        """
        if point in lookup:
            return problem.loglik(lookup[point]), 0.0, "exact"
        version = len(store_points)
        hit = cache.get(point)
        if hit is not None and hit[0] == version:
            return hit[1], hit[2], hit[3]
        fit = _fit(store_points, store_values, point, order_value, k, metric, limit)
        if fit.status != "ok":
            result = (math.nan, math.nan, fit.status)
        else:
            assert fit.value is not None
            result = (problem.loglik(fit.value), problem.indicator(fit), "ok")
        cache[point] = (version, *result)
        if len(cache) > 8:
            cache.pop(next(iter(cache)))
        return result

    state = start
    log_prior_state = problem.prior(state)
    log: list[RefinementEvent] = []
    counts = {
        "insufficient_points": 0,
        "ill_conditioned": 0,
        "random": 0,
        "cross_validation": 0,
    }
    kept: list[Vector] = []
    cumulative: list[int] = []
    accepted = 0
    total = burn_in + samples
    for iteration in range(1, total + 1):
        moved = tuple(
            x + s * rng.gauss(0.0, 1.0) for x, s in zip(state, scales_step, strict=True)
        )
        refine_draw = rng.random()
        accept_draw = math.log(1.0 - rng.random())
        log_prior_moved = problem.prior(moved)
        if log_prior_moved > -math.inf:
            beta_t = min(1.0, b0 * iteration ** (-g))
            if refine_draw < beta_t and moved not in lookup:
                add(moved)
                counts["random"] += 1
                log.append(RefinementEvent(iteration, "random", moved, math.nan))
            for _ in range(3):
                l_cur, e_cur, s_cur = surrogate(state)
                l_new, e_new, s_new = surrogate(moved)
                target: Vector | None = None
                reason = ""
                indicator = math.nan
                if s_new not in ("ok", "exact"):
                    target, reason = moved, s_new
                elif s_cur not in ("ok", "exact"):
                    target, reason = state, s_cur
                elif tolerance is not None and e_cur + e_new > tolerance:
                    target = moved if e_new >= e_cur else state
                    reason, indicator = "cross_validation", max(e_new, e_cur)
                if target is None:
                    break
                add(target)
                counts[reason] += 1
                log.append(RefinementEvent(iteration, reason, target, indicator))
            else:  # pragma: no cover - at most two distinct points exist
                raise RuntimeError("refinement did not settle")
            log_ratio = (log_prior_moved + l_new) - (log_prior_state + l_cur)
            if accept_draw < log_ratio:
                state, log_prior_state = moved, log_prior_moved
                accepted += 1
        if iteration > burn_in:
            kept.append(state)
            cumulative.append(problem.calls)
    ess = tuple(
        effective_sample_size([[draw[j] for draw in kept]])
        if samples >= 4
        else math.nan
        for j in range(n)
    )
    finite = [e for e in ess if math.isfinite(e)]
    return LocalApproximationChainResult(
        samples=tuple(kept),
        acceptance_rate=accepted / total,
        iterations=total,
        true_evaluations=problem.calls,
        evaluations_per_iteration=problem.calls / total,
        cumulative_true_evaluations=tuple(cumulative),
        store_size=len(store_points),
        store_points=tuple(store_points),
        refinement_log=tuple(log),
        refinement_counts=counts,
        effective_sample_size=ess,
        ess_per_true_evaluation=min(finite) / problem.calls if finite else math.nan,
        refinement_never_stops=b0 > 0.0 and g <= 1.0,
        settings={
            "order": order_value,
            "neighbours": k,
            "condition_limit": limit,
            "beta0": b0,
            "gamma": g,
            "cv_tolerance": tolerance,
            "burn_in": burn_in,
            "step": scales_step,
        },
        seed=stream.seed,
    )


def local_approximation_delayed_acceptance(
    forward: ForwardMap,
    data: Sequence[float],
    *,
    noise_std: float | Sequence[float],
    log_prior: LogPrior,
    initial: Sequence[float],
    design: Sequence[Sequence[float]],
    step: float | Sequence[float],
    samples: int,
    burn_in: int = 0,
    order: int = 2,
    neighbours: int | None = None,
    length_scales: float | Sequence[float] | None = None,
    condition_limit: float = 1e6,
    seed: int | RandomStream | None = None,
) -> LocalDelayedAcceptanceResult:
    """Sample the exact posterior with a frozen local surrogate as stage 1.

    The store is the true evaluations of ``design`` and ``initial``; it is
    frozen, so the stage-1 surrogate is a fixed deterministic function and
    :func:`probstat.inference.delayed_acceptance_metropolis` is exact. Where
    the order-``order`` fit is ill-posed the surrogate falls back to a local
    linear fit and then to the nearest stored value (counted in the result).

    Args:
        forward: Expensive map ``G``.
        data: Observation ``y``.
        noise_std: Positive scalar or per-output noise standard deviations.
        log_prior: Cheap log-prior (finite at ``initial``).
        initial: Start point.
        design: Points evaluated truly to build the frozen store (finite prior).
        step: Random-walk scale.
        samples: Kept draws.
        burn_in: Discarded iterations.
        order: ``1`` or ``2``.
        neighbours: ``k`` nearest points (default ``2 p``).
        length_scales: Metric scales.
        condition_limit: Largest accepted design condition number.
        seed: Integer seed or stream.

    Returns:
        A :class:`LocalDelayedAcceptanceResult`.

    Raises:
        TypeError: For non-callables or non-integer counts.
        ValueError: For invalid settings, an empty store, or a start with
            non-finite prior.

    Equation:
        [EQ-INV-LOCAL-005], [EQ-INV-LOCAL-006].

    References:
        [REF-CHRISTEN-FOX-2005].

    """
    problem = _Problem(forward, data, noise_std, log_prior)
    start = _vector(initial, "initial")
    n = len(start)
    _check_common(samples, burn_in, step, n)
    order_value = _require_order(order)
    basis_size = _basis_size(n, order_value)
    k = (
        2 * basis_size
        if neighbours is None
        else _positive_int(neighbours, "neighbours")
    )
    if k < basis_size + 1:
        raise ValueError(f"neighbours must be at least p + 1 = {basis_size + 1}")
    metric = _scales(length_scales, n)
    limit = require_finite_real(condition_limit, name="condition_limit")
    if limit <= 1.0:
        raise ValueError("condition_limit must exceed 1")
    if not math.isfinite(problem.prior(start)):
        raise ValueError("log_prior must be finite at the initial point")
    points: list[Vector] = []
    values: list[Vector] = []
    seen: set[Vector] = set()
    for raw in design:
        point = _vector(raw, "design")
        if len(point) != n:
            raise ValueError("design points must match the dimension")
        if point in seen:
            raise ValueError("design contains a duplicated point")
        seen.add(point)
        if math.isfinite(problem.prior(point)):
            points.append(point)
            values.append(problem.true(point))
    if start not in seen:
        points.append(start)
        values.append(problem.true(start))
    design_calls = problem.calls
    if len(points) < 2:
        raise ValueError("the frozen store needs at least two points")
    fallbacks = {"linear": 0, "nearest": 0}

    def cheap(point: Vector) -> float:
        """Return the frozen-surrogate log-posterior (prior plus likelihood).

        References:
            [EQ-INV-LOCAL-005].

        """
        prior = problem.prior(point)
        if prior == -math.inf:
            return -math.inf
        fit = _fit(points, values, point, order_value, k, metric, limit)
        if fit.status != "ok" and order_value == 2:
            fit = _fit(points, values, point, 1, max(k, n + 2), metric, limit)
            if fit.status == "ok":
                fallbacks["linear"] += 1
        if fit.status == "ok":
            assert fit.value is not None
            return prior + problem.loglik(fit.value)
        fallbacks["nearest"] += 1
        nearest = min(
            range(len(points)),
            key=lambda i: (
                math.fsum(
                    ((a - b) / s) ** 2
                    for a, b, s in zip(points[i], point, metric, strict=True)
                ),
                i,
            ),
        )
        return prior + problem.loglik(values[nearest])

    def exact(point: Vector) -> float:
        """Return the exact log-posterior (one true forward call).

        References:
            [EQ-INV-LOCAL-005].

        """
        prior = problem.prior(point)
        if prior == -math.inf:
            return -math.inf
        return prior + problem.loglik(problem.true(point))

    chain = delayed_acceptance_metropolis(
        exact,
        cheap,
        start,
        proposal="random_walk",
        step=step,
        samples=samples,
        burn_in=burn_in,
        seed=seed,
    )
    ess = [e for e in chain.effective_sample_size if math.isfinite(e)]
    return LocalDelayedAcceptanceResult(
        chain=chain,
        design_evaluations=design_calls,
        true_evaluations=problem.calls,
        store_size=len(points),
        fallback_calls=fallbacks,
        ess_per_true_evaluation=min(ess) / problem.calls if ess else math.nan,
    )
