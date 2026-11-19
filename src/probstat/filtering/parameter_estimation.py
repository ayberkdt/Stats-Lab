"""Parameter estimation in state-space models.

Three estimators of static parameters ``theta`` of a state-space model:

* :func:`fit_state_space_mle` maximizes the filter log-likelihood
  ``log p(y_1:T | theta)`` (exact for the Kalman filter, approximate for the
  EKF and UKF) by BFGS and reports an observed-information covariance from a
  central-difference Hessian;
* :func:`linear_gaussian_em` runs the EM algorithm of Shumway and Stoffer
  for linear-Gaussian models: the E-step is the RTS smoother with lag-one
  covariances, the M-step is closed form, and the log-likelihood never
  decreases;
* :func:`particle_marginal_metropolis_hastings` samples ``p(theta | y_1:T)``
  with a particle-filter likelihood estimate. Because the estimate is
  unbiased and the current state's estimate is stored and reused, the chain
  targets the exact posterior (Andrieu, Doucet, and Holenstein).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_non_negative_integer
from probstat.inference import (
    ConvergenceWarning,
    MarkovChain,
    MCMCResult,
    convergence_warnings,
)
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    cholesky_solve,
    inverse_spd,
    matvec,
    symmetrize,
)
from probstat.optimization import OptimizationResult, central_hessian, minimize_bfgs
from probstat.optimization.bounds import Bounds

from .extended import extended_kalman_filter, unscented_kalman_filter
from .kalman import (
    KalmanFilterResult,
    Observation,
    SmootherResult,
    _observation_vector,
    kalman_filter,
    rts_smoother,
)
from .particle import ParticleFilterResult, bootstrap_particle_filter
from .particle_proposals import (
    auxiliary_particle_filter,
    optimal_proposal_particle_filter,
)
from .state_space import LinearGaussianStateSpace, StateSpaceModel

ModelBuilder = Callable[[Vector], StateSpaceModel]

_FILTERS = ("kalman", "ekf", "ukf")
_PARTICLE_FILTERS = ("bootstrap", "auxiliary", "optimal")
_EM_BLOCKS = (
    "transition",
    "observation",
    "process_covariance",
    "observation_covariance",
    "initial",
)


# ---------------------------------------------------------------------------
# Maximum likelihood
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StateSpaceFit:
    """Maximum-likelihood fit of state-space model parameters.

    Args:
        parameters: Estimate ``theta_hat``.
        names: Parameter names.
        log_likelihood: Filter log-likelihood at ``theta_hat``.
        covariance: Inverse observed information ``(-d^2 l / d theta^2)^-1``
            at ``theta_hat``, or ``None`` when it is not available.
        standard_errors: Square roots of the covariance diagonal, or ``None``.
        observed_information: Central-difference Hessian of ``-l``, or
            ``None`` when it could not be evaluated.
        model: Model built at ``theta_hat``.
        filter_result: Filter run at ``theta_hat``.
        optimization: BFGS result (on the negative log-likelihood).
        filter: Filter that defined the likelihood (``"kalman"``, ``"ekf"``,
            or ``"ukf"``).
        message: Why ``covariance`` is ``None``, or ``"ok"``.

    References:
        [REF-DURBIN-KOOPMAN-2012, Chapter 7], [REF-SARKKA-2013, Chapter 12].

    """

    parameters: Vector
    names: tuple[str, ...]
    log_likelihood: float
    covariance: Matrix | None
    standard_errors: Vector | None
    observed_information: Matrix | None
    model: StateSpaceModel
    filter_result: KalmanFilterResult
    optimization: OptimizationResult
    filter: str
    message: str

    @property
    def converged(self) -> bool:
        """Return whether the optimizer met a stopping tolerance.

        References:
            [REF-NOCEDAL-WRIGHT-2006, Chapter 6]. No equation is evaluated.

        """
        return self.optimization.converged


def _run_gaussian_filter(
    model: StateSpaceModel,
    observations: Sequence[Observation],
    filter: str,
    controls: Sequence[Sequence[float] | None] | None,
) -> KalmanFilterResult:
    """Run the named Gaussian filter.

    Args:
        model: State-space model.
        observations: Observations ``y_1..y_T``.
        filter: ``"kalman"``, ``"ekf"``, or ``"ukf"``.
        controls: Optional controls.

    Returns:
        Filter result.

    Raises:
        TypeError: If ``"kalman"`` is asked for a nonlinear model.

    References:
        [REF-SARKKA-2013, Chapters 4, 5, and 12]. No new equation is evaluated.

    """
    if filter == "kalman":
        if not isinstance(model, LinearGaussianStateSpace):
            raise TypeError(
                "filter='kalman' needs the builder to return a "
                "LinearGaussianStateSpace; use filter='ekf' or 'ukf'"
            )
        return kalman_filter(model, observations, controls=controls)
    if filter == "ekf":
        return extended_kalman_filter(model, observations, controls=controls)
    return unscented_kalman_filter(model, observations, controls=controls)


def fit_state_space_mle(
    builder: ModelBuilder,
    initial_parameters: Sequence[float],
    observations: Sequence[Observation],
    *,
    bounds: Bounds | None = None,
    filter: str = "kalman",
    controls: Sequence[Sequence[float] | None] | None = None,
    names: Sequence[str] | None = None,
    gradient_tolerance: float = 1e-6,
    max_iterations: int = 200,
    typical_scale: float | Sequence[float] | None = None,
    hessian_step: float = 1e-4,
) -> StateSpaceFit:
    """Maximize the filter log-likelihood of a parametrized state-space model.

    ``theta_hat = argmax_theta l(theta)`` with
    ``l(theta) = sum_k log N(v_k; 0, S_k)`` from the Kalman filter (exact) or
    the EKF/UKF (a Gaussian approximation), minimized as ``-l`` by
    :func:`probstat.optimization.minimize_bfgs` with central-difference
    gradients. Constraints are handled by ``bounds`` (a smooth box
    transform) or, often better, by a builder that takes unconstrained
    parameters such as log-variances. The covariance is the inverse of the
    observed information ``-d^2 l / d theta^2`` from
    :func:`probstat.optimization.central_hessian`; it is reported as ``None``
    with a reason when the Hessian cannot be evaluated (for example a step
    leaves the builder's domain at an estimate on a bound) or is not
    positive definite. Nothing is regularized.

    Args:
        builder: Map ``theta -> model``; it may raise ``ValueError`` for
            invalid ``theta``, which makes the line search shorten its step.
        initial_parameters: Start ``theta_0``.
        observations: Observations ``y_1..y_T`` (``None`` if missing).
        bounds: Optional ``(low, high)`` per parameter.
        filter: ``"kalman"`` (builder must return a
            ``LinearGaussianStateSpace``), ``"ekf"``, or ``"ukf"``.
        controls: Optional controls passed to the filter.
        names: Optional parameter names (default ``theta0, theta1, ...``).
        gradient_tolerance: Absolute gradient-norm tolerance of BFGS.
        max_iterations: BFGS iteration limit.
        typical_scale: Typical parameter magnitudes for BFGS.
        hessian_step: Relative step of the central-difference Hessian.

    Returns:
        Estimate, log-likelihood, observed-information covariance, fitted
        model, filter run, and the optimizer record.

    Raises:
        ValueError: If ``filter`` is unknown, names do not match, or the
            start has a non-finite likelihood.
        TypeError: If ``filter="kalman"`` meets a nonlinear model.

    Equation:
        [EQ-FILT-KF-004], [EQ-FILT-MLE-001], [EQ-OPT-FD-002].

    References:
        [REF-DURBIN-KOOPMAN-2012, Chapter 7], [REF-SARKKA-2013, Chapter 12],
        [REF-NOCEDAL-WRIGHT-2006, Chapter 6].

    """
    if filter not in _FILTERS:
        raise ValueError(f"filter must be one of {_FILTERS}")
    start = as_vector(initial_parameters, name="initial_parameters")
    if not start:
        raise ValueError("initial_parameters must be non-empty")
    labels = (
        tuple(f"theta{i}" for i in range(len(start))) if names is None else tuple(names)
    )
    if len(labels) != len(start) or len(set(labels)) != len(labels):
        raise ValueError("names must be distinct with one name per parameter")
    data = tuple(observations)
    if not data:
        raise ValueError("observations must be non-empty")
    plan = None if controls is None else tuple(controls)

    def negative_log_likelihood(theta: Vector) -> float:
        """Return ``-l(theta)`` from the chosen filter.

        References:
            [EQ-FILT-MLE-001], [REF-SARKKA-2013, Chapter 12].

        """
        return -_run_gaussian_filter(builder(theta), data, filter, plan).log_likelihood

    first = negative_log_likelihood(start)
    if not math.isfinite(first):
        raise ValueError("the log-likelihood is not finite at initial_parameters")
    optimum = minimize_bfgs(
        negative_log_likelihood,
        start,
        gradient_tolerance=gradient_tolerance,
        max_iterations=max_iterations,
        bounds=bounds,
        typical_scale=typical_scale,
    )
    estimate = optimum.x
    model = builder(estimate)
    result = _run_gaussian_filter(model, data, filter, plan)
    covariance: Matrix | None = None
    errors: Vector | None = None
    information: Matrix | None = None
    message = "ok"
    try:
        information, _ = central_hessian(
            negative_log_likelihood, estimate, relative_step=hessian_step
        )
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        message = f"observed information could not be evaluated: {error}"
    if information is not None:
        try:
            cholesky(information)
        except ValueError:
            message = "observed information is not positive definite"
        else:
            covariance = inverse_spd(information)
            errors = tuple(math.sqrt(covariance[i][i]) for i in range(len(estimate)))
    return StateSpaceFit(
        estimate,
        labels,
        result.log_likelihood,
        covariance,
        errors,
        information,
        model,
        result,
        optimum,
        filter,
        message,
    )


# ---------------------------------------------------------------------------
# Expectation maximization
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EMResult:
    """Outcome of EM estimation of a linear-Gaussian state-space model.

    Args:
        model: Model after the last M-step.
        log_likelihood_trace: Exact log-likelihood of the start model and of
            the model after each M-step; the last entry belongs to ``model``.
        iterations: Number of M-steps performed.
        converged: Whether the last change met the tolerance.
        estimated: Blocks that were updated.

    References:
        [REF-SHUMWAY-STOFFER-1982], [REF-SARKKA-2013, Chapter 12].

    """

    model: LinearGaussianStateSpace
    log_likelihood_trace: tuple[float, ...]
    iterations: int
    converged: bool
    estimated: tuple[str, ...]

    @property
    def log_likelihood(self) -> float:
        """Return the log-likelihood of the returned model.

        References:
            [EQ-FILT-KF-004], [REF-SARKKA-2013, Chapter 12].

        """
        return self.log_likelihood_trace[-1]


def smoothed_lag_one_covariances(smoother: SmootherResult) -> tuple[Matrix, ...]:
    """Return ``Cov(x_k, x_(k-1) | y_1:T)`` for ``k = 1..T``.

    For the RTS smoother, ``Cov(x_k, x_(k-1) | y_1:T) = P^s_k G_(k-1)'`` with
    the smoother gain ``G_(k-1)``; this equals the lag-one recursion of
    Shumway and Stoffer.

    Args:
        smoother: Result of :func:`probstat.filtering.rts_smoother`.

    Returns:
        ``T`` lag-one smoothed cross-covariances.

    Equation:
        [EQ-FILT-EM-001].

    References:
        [REF-SARKKA-2013, Chapter 12], [REF-SHUMWAY-STOFFER-1982].

    """
    smoothed = smoother.smoothed
    result: list[Matrix] = []
    for k in range(1, len(smoothed)):
        covariance = smoothed[k].covariance
        gain = smoother.gains[k - 1]
        size = len(covariance)
        result.append(
            tuple(
                tuple(
                    math.fsum(covariance[i][t] * gain[j][t] for t in range(size))
                    for j in range(size)
                )
                for i in range(size)
            )
        )
    return tuple(result)


def _zeros(rows: int, columns: int) -> list[list[float]]:
    """Return a mutable zero matrix.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No equation is evaluated.

    """
    return [[0.0] * columns for _ in range(rows)]


def _accumulate(
    target: list[list[float]],
    left: Sequence[float],
    right: Sequence[float],
    extra: Matrix | None = None,
) -> None:
    """Add ``left right' (+ extra)`` to a mutable matrix in place.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No equation is evaluated.

    """
    for i, a in enumerate(left):
        row = target[i]
        for j, b in enumerate(right):
            row[j] += a * b + (0.0 if extra is None else extra[i][j])


def _scaled(matrix: Sequence[Sequence[float]], factor: float) -> Matrix:
    """Return ``factor * matrix`` as a tuple matrix.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No equation is evaluated.

    """
    return tuple(tuple(factor * value for value in row) for row in matrix)


def _right_solve(numerator: Matrix, denominator: Matrix, name: str) -> Matrix:
    """Return ``A D^-1`` for symmetric positive-definite ``D`` by Cholesky solves.

    Raises:
        ValueError: If ``D`` is not positive definite.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

    """
    try:
        lower = cholesky(denominator)
    except ValueError as error:
        raise ValueError(
            f"the smoothed second moment for the {name} update is not positive definite"
        ) from error
    return tuple(cholesky_solve(lower, row) for row in numerator)


def _residual_covariance(
    second: Matrix, cross: Matrix, previous: Matrix, operator: Matrix
) -> Matrix:
    """Return ``S - C A' - A C' + A Phi A'`` symmetrized.

    Args:
        second: ``S = E[z z']`` (averaged).
        cross: ``C = E[z u']`` (averaged).
        previous: ``Phi = E[u u']`` (averaged).
        operator: ``A`` in the regression ``z = A u + e``.

    Returns:
        ``E[(z - A u)(z - A u)']``.

    Equation:
        [EQ-FILT-EM-001].

    References:
        [REF-SHUMWAY-STOFFER-1982]. No other equation is evaluated.

    """
    rows = len(second)
    inner = len(previous)
    a_phi = [
        [
            math.fsum(operator[i][t] * previous[t][j] for t in range(inner))
            for j in range(inner)
        ]
        for i in range(rows)
    ]
    raw = [
        [
            second[i][j]
            - math.fsum(cross[i][t] * operator[j][t] for t in range(inner))
            - math.fsum(operator[i][t] * cross[j][t] for t in range(inner))
            + math.fsum(a_phi[i][t] * operator[j][t] for t in range(inner))
            for j in range(rows)
        ]
        for i in range(rows)
    ]
    return tuple(
        tuple(0.5 * (raw[i][j] + raw[j][i]) for j in range(rows)) for i in range(rows)
    )


def _em_step(
    model: LinearGaussianStateSpace,
    data: tuple[Observation, ...],
    smoother: SmootherResult,
    blocks: frozenset[str],
) -> LinearGaussianStateSpace:
    """Return the model after one closed-form M-step.

    Args:
        model: Current model.
        data: Observations.
        smoother: RTS smoother of the current model.
        blocks: Blocks to update.

    Returns:
        Updated model.

    Raises:
        ValueError: If a second-moment matrix is singular or an updated
            covariance is not valid for the model.

    Equation:
        [EQ-FILT-EM-001].

    References:
        [REF-SHUMWAY-STOFFER-1982], [REF-SARKKA-2013, Chapter 12].

    """
    smoothed = smoother.smoothed
    steps = len(data)
    size = model.state_dimension
    rows = model.observation_dimension
    lag_one = smoothed_lag_one_covariances(smoother)
    current = _zeros(size, size)
    previous = _zeros(size, size)
    cross = _zeros(size, size)
    for k in range(1, steps + 1):
        now, before = smoothed[k], smoothed[k - 1]
        _accumulate(current, now.mean, now.mean, now.covariance)
        _accumulate(previous, before.mean, before.mean, before.covariance)
        _accumulate(cross, now.mean, before.mean, lag_one[k - 1])
    sigma = _scaled(current, 1.0 / steps)
    phi = _scaled(previous, 1.0 / steps)
    c_matrix = _scaled(cross, 1.0 / steps)
    transition = model.transition_matrix
    if "transition" in blocks:
        transition = _right_solve(c_matrix, phi, "transition")
    process = model.process_covariance
    if "process_covariance" in blocks:
        process = _residual_covariance(sigma, c_matrix, phi, transition)
    observation = model.observation_matrix
    noise = model.observation_covariance
    if "observation" in blocks or "observation_covariance" in blocks:
        observed = [k for k in range(1, steps + 1) if data[k - 1] is not None]
        if not observed:
            raise ValueError("observation blocks need at least one observation")
        state_moment = _zeros(size, size)
        data_cross = _zeros(rows, size)
        data_moment = _zeros(rows, rows)
        for k in observed:
            raw = data[k - 1]
            assert raw is not None
            y = _observation_vector(raw, rows)
            state = smoothed[k]
            _accumulate(state_moment, state.mean, state.mean, state.covariance)
            _accumulate(data_cross, y, state.mean)
            _accumulate(data_moment, y, y)
        scale = 1.0 / len(observed)
        moment = _scaled(state_moment, scale)
        b_matrix = _scaled(data_cross, scale)
        if "observation" in blocks:
            observation = _right_solve(b_matrix, moment, "observation")
        if "observation_covariance" in blocks:
            noise = _residual_covariance(
                _scaled(data_moment, scale), b_matrix, moment, observation
            )
    initial_mean = model.initial_mean
    initial_covariance = model.initial_covariance
    if "initial" in blocks:
        initial_mean = smoothed[0].mean
        initial_covariance = symmetrize(smoothed[0].covariance)
    try:
        return LinearGaussianStateSpace(
            transition,
            process,
            observation,
            noise,
            initial_mean,
            initial_covariance,
        )
    except ValueError as error:
        raise ValueError(f"the M-step produced an invalid model: {error}") from error


def linear_gaussian_em(
    model: LinearGaussianStateSpace,
    observations: Sequence[Observation],
    *,
    iterations: int = 500,
    tolerance: float = 1e-8,
    estimate: Sequence[str] = _EM_BLOCKS,
) -> EMResult:
    """Estimate linear-Gaussian model matrices by expectation maximization.

    E-step: Kalman filter and RTS smoother of the current model, with
    ``E[x_k x_k'] = P^s_k + m^s_k m^s_k'`` and
    ``E[x_k x_(k-1)'] = P^s_k G_(k-1)' + m^s_k m^s_(k-1)'``. M-step (closed
    form, averages over ``k = 1..T``, or over observed ``k`` for the
    observation blocks): ``F = C Phi^-1``,
    ``Q = Sigma - C F' - F C' + F Phi F'``, ``H = B Sigma_o^-1``,
    ``R = D - H B' - B H' + H Sigma_o H'``, ``m_0 = m^s_0``, ``P_0 = P^s_0``.
    A block not in ``estimate`` stays fixed and the others maximize the
    expected complete-data log-likelihood given it, so the exact
    log-likelihood is non-decreasing. Iteration stops when the change is at
    most ``tolerance * max(1, |l|)``.

    Args:
        model: Start model (no control matrix).
        observations: Observations ``y_1..y_T`` (``None`` if missing).
        iterations: Maximum number of M-steps.
        tolerance: Relative log-likelihood change that declares convergence.
        estimate: Blocks to update, a subset of ``"transition"``,
            ``"observation"``, ``"process_covariance"``,
            ``"observation_covariance"``, and ``"initial"`` (mean and
            covariance of ``x_0``).

    Returns:
        Final model, log-likelihood trace, and convergence record.

    Raises:
        TypeError: If ``model`` is not a ``LinearGaussianStateSpace``.
        ValueError: If arguments are invalid, the model has controls, or an
            M-step produces an invalid covariance.

    Equation:
        [EQ-FILT-EM-001], [EQ-FILT-KF-004], [EQ-FILT-RTS-001].

    References:
        [REF-SHUMWAY-STOFFER-1982], [REF-SARKKA-2013, Chapter 12].

    """
    if not isinstance(model, LinearGaussianStateSpace):
        raise TypeError("model must be a LinearGaussianStateSpace")
    if model.control_matrix is not None:
        raise ValueError("linear_gaussian_em does not support controlled models")
    count = require_non_negative_integer(iterations, name="iterations")
    if not tolerance > 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")
    requested = tuple(estimate)
    unknown = [name for name in requested if name not in _EM_BLOCKS]
    if unknown or not requested:
        raise ValueError(f"estimate must be a non-empty subset of {_EM_BLOCKS}")
    blocks = frozenset(requested)
    data = tuple(observations)
    if not data:
        raise ValueError("observations must be non-empty")
    current = model
    result = kalman_filter(current, data)
    trace = [result.log_likelihood]
    converged = False
    performed = 0
    for _ in range(count):
        smoother = rts_smoother(result)
        current = _em_step(current, data, smoother, blocks)
        result = kalman_filter(current, data)
        trace.append(result.log_likelihood)
        performed += 1
        change = trace[-1] - trace[-2]
        if abs(change) <= tolerance * max(1.0, abs(trace[-1])):
            converged = True
            break
    return EMResult(
        current,
        tuple(trace),
        performed,
        converged,
        tuple(name for name in _EM_BLOCKS if name in blocks),
    )


# ---------------------------------------------------------------------------
# Particle marginal Metropolis-Hastings
# ---------------------------------------------------------------------------


def _particle_filter(
    name: str,
    model: StateSpaceModel,
    data: tuple[Observation, ...],
    particles: int,
    threshold: float,
    stream: RandomStream,
    controls: Sequence[Sequence[float] | None] | None,
) -> ParticleFilterResult:
    """Run the named particle filter.

    References:
        [REF-ANDRIEU-DOUCET-HOLENSTEIN-2010]. No new equation is evaluated.

    """
    if name == "bootstrap":
        return bootstrap_particle_filter(
            model,
            data,
            particles=particles,
            resample_threshold=threshold,
            seed=stream,
            controls=controls,
        )
    if name == "auxiliary":
        return auxiliary_particle_filter(
            model, data, particles=particles, seed=stream, controls=controls
        )
    return optimal_proposal_particle_filter(
        model,
        data,
        particles=particles,
        resample_threshold=threshold,
        seed=stream,
        controls=controls,
    )


def particle_marginal_metropolis_hastings(
    builder: ModelBuilder,
    log_prior: Callable[[Vector], float],
    initial: Sequence[float],
    observations: Sequence[Observation],
    *,
    particles: int,
    draws: int,
    proposal_covariance: Sequence[Sequence[float]],
    warmup: int = 0,
    chains: int = 1,
    thin: int = 1,
    seed: int | RandomStream | None = None,
    particle_filter: str = "bootstrap",
    resample_threshold: float = 0.5,
    controls: Sequence[Sequence[float] | None] | None = None,
    names: Sequence[str] | None = None,
    check_convergence: bool = True,
) -> MCMCResult:
    """Sample ``p(theta | y_1:T)`` by particle marginal Metropolis-Hastings.

    Each iteration proposes ``theta' = theta + L z`` (``L L'`` the proposal
    covariance), runs a fresh particle filter to obtain the unbiased estimate
    ``p_hat(y_1:T | theta')``, and accepts with probability
    ``min(1, p(theta') p_hat(y | theta') / (p(theta) p_hat(y | theta)))``.
    The estimate of the current state is the one stored when that state was
    accepted; it is never recomputed. With that rule the chain is an exact MH
    chain on the extended space of parameters and filter randomness whose
    ``theta``-marginal is the exact posterior, for any particle count (fewer
    particles only mix worse). Proposals with ``log_prior = -inf`` are
    rejected without running a filter; a filter whose weights all underflow
    returns the estimate zero and the proposal is rejected.

    Args:
        builder: Map ``theta -> model``.
        log_prior: Log prior density (up to a constant); ``-inf`` outside the
            support.
        initial: Start ``theta_0`` shared by all chains; its prior and its
            likelihood estimate must be finite.
        observations: Observations ``y_1..y_T`` (``None`` if missing).
        particles: Particles per likelihood estimate.
        draws: Kept draws per chain.
        proposal_covariance: Symmetric positive-definite random-walk
            covariance.
        warmup: Discarded iterations per chain (no adaptation).
        chains: Independent chains (streams ``"pmmh-chain"`` by index).
        thin: Keep every ``thin``-th post-warm-up state.
        seed: Integer seed or stream; per chain the children ``"proposal"``
            and ``("likelihood", i)`` drive proposals and the filter run of
            iteration ``i`` (``i = 0`` is the start).
        particle_filter: ``"bootstrap"``, ``"auxiliary"``, or ``"optimal"``.
        resample_threshold: ESS fraction for the bootstrap and optimal filters.
        controls: Optional controls passed to the filter.
        names: Optional parameter names.
        check_convergence: Run the convergence diagnostics of
            :func:`probstat.inference.convergence_warnings` and emit each
            finding as a ``ConvergenceWarning``.

    Returns:
        :class:`probstat.inference.MCMCResult` whose ``log_density`` traces
        hold ``log p(theta) + log p_hat(y | theta)`` of each kept state and
        whose ``evaluations`` count particle-filter runs.

    Raises:
        TypeError: If ``log_prior`` is not callable.
        ValueError: For invalid arguments, a start with a non-finite target
            estimate, or a log prior that returns NaN or ``+inf``.

    Equation:
        [EQ-FILT-PMMH-001], [EQ-FILT-PF-001].

    References:
        [REF-ANDRIEU-DOUCET-HOLENSTEIN-2010], [REF-METROPOLIS-ETAL-1953],
        [REF-HASTINGS-1970].

    """
    if not callable(log_prior):
        raise TypeError("log_prior must be callable")
    if particle_filter not in _PARTICLE_FILTERS:
        raise ValueError(f"particle_filter must be one of {_PARTICLE_FILTERS}")
    if isinstance(particles, bool) or not isinstance(particles, int) or particles < 2:
        raise ValueError("particles must be an integer of at least two")
    kept = require_non_negative_integer(draws, name="draws")
    burn = require_non_negative_integer(warmup, name="warmup")
    chain_count = require_non_negative_integer(chains, name="chains")
    step = require_non_negative_integer(thin, name="thin")
    if kept == 0 or chain_count == 0 or step == 0:
        raise ValueError("draws, chains, and thin must be positive")
    start = as_vector(initial, name="initial")
    if not start:
        raise ValueError("initial must be non-empty")
    dimension = len(start)
    covariance = symmetrize(proposal_covariance, name="proposal_covariance")
    if len(covariance) != dimension:
        raise ValueError("proposal_covariance must match the parameter dimension")
    try:
        lower = cholesky(covariance)
    except ValueError as error:
        raise ValueError("proposal_covariance must be positive definite") from error
    labels = (
        tuple(f"theta{i}" for i in range(dimension)) if names is None else tuple(names)
    )
    data = tuple(observations)
    if not data:
        raise ValueError("observations must be non-empty")
    plan = None if controls is None else tuple(controls)

    def prior(theta: Vector) -> float:
        """Evaluate and validate the log prior.

        References:
            [REF-ROBERT-CASELLA-2004, Section 7.3]. No equation is evaluated.

        """
        value = float(log_prior(theta))
        if math.isnan(value) or value == math.inf:
            raise ValueError("log_prior returned NaN or +inf")
        return value

    def run_chain(stream: RandomStream) -> MarkovChain:
        """Run one PMMH chain.

        References:
            [EQ-FILT-PMMH-001], [REF-ANDRIEU-DOUCET-HOLENSTEIN-2010].

        """
        generator = stream.child("proposal").generator()
        likelihood_streams = stream.child("likelihood")
        runs = 0

        def target(theta: Vector, index: int) -> float:
            """Return ``log p(theta) + log p_hat(y | theta)`` (one filter run).

            References:
                [EQ-FILT-PMMH-001], [EQ-FILT-PF-001].

            """
            nonlocal runs
            log_p = prior(theta)
            if log_p == -math.inf:
                return -math.inf
            model = builder(theta)
            runs += 1
            try:
                estimate = _particle_filter(
                    particle_filter,
                    model,
                    data,
                    particles,
                    resample_threshold,
                    likelihood_streams.child(index),
                    plan,
                ).log_likelihood
            except ValueError as error:
                if "underflowed" not in str(error):
                    raise
                return -math.inf
            return log_p + estimate

        state = start
        current = target(state, 0)
        if not math.isfinite(current):
            raise ValueError("initial must have a finite prior and likelihood estimate")
        warm_accepted = 0
        kept_accepted = 0
        samples: list[Vector] = []
        trace: list[float] = []
        for iteration in range(burn + kept * step):
            z = tuple(generator.gauss(0.0, 1.0) for _ in range(dimension))
            proposal = tuple(
                a + b for a, b in zip(state, matvec(lower, z), strict=True)
            )
            proposed = target(proposal, iteration + 1)
            accepted = math.log(1.0 - generator.random()) < proposed - current
            if accepted:
                # The accepted estimate is stored and reused until the next
                # acceptance: re-estimating the current state would break
                # exactness.
                state, current = proposal, proposed
            if iteration < burn:
                warm_accepted += accepted
            else:
                kept_accepted += accepted
                if (iteration - burn + 1) % step == 0:
                    samples.append(state)
                    trace.append(current)
        return MarkovChain(
            tuple(samples),
            tuple(trace),
            kept_accepted / (kept * step),
            warm_accepted / burn if burn else math.nan,
            covariance,
            runs,
            stream.seed,
        )

    streams = resolve_stream(seed).spawn(chain_count, label="pmmh-chain")
    result = MCMCResult(
        tuple(run_chain(stream) for stream in streams),
        labels,
        "particle marginal Metropolis-Hastings",
        burn,
        step,
    )
    if not check_convergence:
        return result
    messages = convergence_warnings(result)
    for message in messages:
        warnings.warn(message, ConvergenceWarning, stacklevel=2)
    return replace(result, warnings=messages)
