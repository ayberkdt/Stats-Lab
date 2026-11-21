"""Bayesian optimization of an expensive function over a box.

A Gaussian-process surrogate of the objective is refitted after every
evaluation and an *acquisition function* built from its predictive mean
``mu(x)`` and standard deviation ``s(x)`` selects the next point. This module
provides the acquisition functions as plain functions of ``(mu, s)`` and a
sequential loop on top of :func:`probstat.surrogates.fit_gaussian_process`:

* expected improvement ``EI = s h(z)``, ``h(z) = phi(z) + z Phi(z)``
  [EQ-DES-BO-001], and its logarithm, evaluated without underflow for very
  negative ``z`` [EQ-DES-BO-002];
* upper and lower confidence bounds with a user ``beta`` and the finite-domain
  GP-UCB schedule [EQ-DES-BO-003];
* constrained expected improvement with independent GP models of the
  constraints [EQ-DES-BO-004];
* the knowledge gradient for independent normal beliefs over a finite set of
  alternatives [EQ-DES-BO-005];
* Thompson sampling on a finite candidate set through one joint posterior
  draw [EQ-DES-BO-006].

Direction is always explicit: every function takes ``maximize`` (default
``False``, minimization) and the improvement is ``f* - f`` for minimization
and ``f - f*`` for maximization.

The loop is a heuristic global optimizer. It carries no convergence
certificate: the result reports the best evaluated point, the evaluation
counts, and the history, never a claim that the global optimum was found.

``design`` imports its layer-5 sibling ``surrogates`` here (ADR-0125);
``surrogates`` does not import ``design``, so the import graph stays
acyclic.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.distributions import Normal
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    cholesky_semidefinite,
    solve_lower,
)
from probstat.optimization import minimize_nelder_mead
from probstat.sampling import latin_hypercube, maximin_latin_hypercube
from probstat.surrogates import (
    KERNELS,
    GaussianProcessRegression,
    GaussianProcessRegularization,
    fit_gaussian_process,
    kernel_value,
)

ACQUISITIONS: tuple[str, ...] = ("log_ei", "ei", "ucb", "thompson")
"""Acquisition names accepted by :func:`bayesian_optimization`."""

INITIAL_DESIGNS: tuple[str, ...] = ("lhs", "maximin")
"""Initial-design names accepted by :func:`bayesian_optimization`."""

_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_SQRT_TWO = math.sqrt(2.0)
_STANDARD_NORMAL = Normal(0.0, 1.0)
_DEFAULT_REGULARIZATION = GaussianProcessRegularization()


def _density(z: float) -> float:
    """Return the standard-normal density ``phi(z)``.

    References:
        [EQ-DIST-NORM-001], [REF-NIST-DLMF, Section 7.2].

    """
    return math.exp(-0.5 * z * z - _LOG_SQRT_TWO_PI)


def _cdf(z: float) -> float:
    """Return the standard-normal CDF ``Phi(z) = erfc(-z / sqrt 2) / 2``.

    References:
        [EQ-DIST-NORM-002], [REF-NIST-DLMF, Equation 7.2.1].

    """
    return 0.5 * math.erfc(-z / _SQRT_TWO)


def _tail_fraction(t: float) -> float:
    """Return ``c(t) = 1 / (t + 2 / (t + 3 / (t + ...)))`` for ``t >= 1``.

    The Mills ratio ``R(t) = (1 - Phi(t)) / phi(t)`` has the continued
    fraction ``R(t) = 1 / (t + 1 / (t + 2 / (t + ...))) = 1 / (t + c(t))``,
    so ``1 - t R(t) = c(t) / (t + c(t))`` without subtraction. The fraction
    is evaluated bottom-up with ``30 + 300 / t^2`` terms; the resulting
    ``log h`` agreed with 50-digit reference arithmetic to ``1e-15``
    relative on ``-100 <= z <= -1`` in the tests.

    References:
        [EQ-DES-BO-002], [REF-NIST-DLMF, Section 7.9].

    """
    terms = 30 + int(min(300.0, 300.0 / (t * t)))
    value = 0.0
    for index in range(terms, 1, -1):
        value = index / (t + value)
    return 1.0 / (t + value)


def log_improvement_function(z: float) -> float:
    """Return ``log h(z)`` with ``h(z) = phi(z) + z Phi(z)``, stable in the tail.

    ``h`` is positive and increasing with ``h(z) ~ phi(z) / z^2`` as
    ``z -> -inf``, where ``phi(z) + z Phi(z)`` cancels and then underflows.
    For ``z > -1`` the sum is evaluated directly (it loses at most a factor
    of three in relative accuracy at ``z = -1``). For ``z <= -1``, with
    ``t = -z`` and the Mills ratio ``R(t) = 1 / (t + c(t))`` written as a
    continued fraction,

    ``h(z) = phi(t) (1 - t R(t)) = phi(t) c(t) / (t + c(t))``,

    so ``log h = -t^2/2 - log sqrt(2 pi) + log c - log(t + c)`` has no
    cancellation and no underflow: ``log h(-40) = -808.2986...``.

    Args:
        z: Standardized improvement; ``-inf`` and ``inf`` are accepted.

    Returns:
        ``log h(z)`` (``-inf`` at ``z = -inf`` and when ``z^2`` overflows).

    Raises:
        ValueError: If ``z`` is NaN.

    Equation:
        [EQ-DES-BO-002].

    References:
        [REF-AMENT-ETAL-2023] (the need for a stable ``log h`` and the
        tail behaviour), [REF-NIST-DLMF, Section 7.9] (continued fraction
        of the complementary error function).

    """
    if math.isnan(z):
        raise ValueError("z must not be NaN")
    if z == math.inf:
        return math.inf
    if z > -1.0:
        return math.log(_density(z) + z * _cdf(z))
    t = -z
    if t > 1e150:
        return -math.inf
    tail = _tail_fraction(t)
    return -0.5 * t * t - _LOG_SQRT_TWO_PI + math.log(tail) - math.log(t + tail)


def improvement_function(z: float) -> float:
    """Return ``h(z) = phi(z) + z Phi(z) = E[max(z + Z, 0)]``, ``Z ~ N(0, 1)``.

    For ``z <= -1`` the value is ``exp(log h(z))`` from
    :func:`log_improvement_function`, which keeps full relative accuracy
    until the result underflows to zero near ``z = -38.5``.

    Args:
        z: Standardized improvement.

    Returns:
        ``h(z) >= 0``.

    Raises:
        ValueError: If ``z`` is NaN.

    Equation:
        [EQ-DES-BO-001], [EQ-DES-BO-002].

    References:
        [REF-JONES-SCHONLAU-WELCH-1998].

    """
    if math.isnan(z):
        raise ValueError("z must not be NaN")
    if z > -1.0:
        return z if z == math.inf else _density(z) + z * _cdf(z)
    return math.exp(log_improvement_function(z))


def _moments(mean: float, standard_deviation: float) -> tuple[float, float]:
    """Validate a predictive mean and standard deviation.

    Raises:
        ValueError: If a value is not finite or the deviation is negative.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    mu = require_finite_real(mean, name="mean")
    sd = require_finite_real(standard_deviation, name="standard_deviation")
    if sd < 0.0:
        raise ValueError("standard_deviation must be non-negative")
    return mu, sd


def expected_improvement(
    mean: float,
    standard_deviation: float,
    best: float,
    *,
    maximize: bool = False,
) -> float:
    """Return the expected improvement over the incumbent ``best``.

    For ``F ~ N(mean, s^2)`` the improvement is ``I = max(best - F, 0)``
    (minimization) or ``max(F - best, 0)`` (maximization) and
    ``E[I] = s h(z)`` with ``z = (best - mean) / s`` or
    ``z = (mean - best) / s``. With ``s = 0`` the value is the deterministic
    improvement.

    Args:
        mean: Predictive mean ``mu``.
        standard_deviation: Predictive standard deviation ``s >= 0``.
        best: Incumbent value ``f*`` (best observed so far).
        maximize: Direction of the optimization.

    Returns:
        ``E[I] >= 0``.

    Raises:
        ValueError: If an argument is not finite or ``s < 0``.

    Equation:
        [EQ-DES-BO-001].

    References:
        [REF-JONES-SCHONLAU-WELCH-1998].

    """
    mu, sd = _moments(mean, standard_deviation)
    incumbent = require_finite_real(best, name="best")
    gain = mu - incumbent if maximize else incumbent - mu
    if sd == 0.0:
        return max(gain, 0.0)
    return sd * improvement_function(gain / sd)


def log_expected_improvement(
    mean: float,
    standard_deviation: float,
    best: float,
    *,
    maximize: bool = False,
) -> float:
    """Return ``log E[I] = log s + log h(z)``, finite where ``E[I]`` underflows.

    Args:
        mean: Predictive mean ``mu``.
        standard_deviation: Predictive standard deviation ``s >= 0``.
        best: Incumbent value ``f*``.
        maximize: Direction of the optimization.

    Returns:
        Logarithm of the expected improvement; with ``s = 0`` the logarithm
        of the deterministic improvement (``-inf`` when there is none).

    Raises:
        ValueError: If an argument is not finite or ``s < 0``.

    Equation:
        [EQ-DES-BO-001], [EQ-DES-BO-002].

    References:
        [REF-AMENT-ETAL-2023], [REF-JONES-SCHONLAU-WELCH-1998].

    """
    mu, sd = _moments(mean, standard_deviation)
    incumbent = require_finite_real(best, name="best")
    gain = mu - incumbent if maximize else incumbent - mu
    if sd == 0.0:
        return math.log(gain) if gain > 0.0 else -math.inf
    return math.log(sd) + log_improvement_function(gain / sd)


def confidence_bound(
    mean: float,
    standard_deviation: float,
    beta: float,
    *,
    maximize: bool = False,
) -> float:
    """Return the optimistic confidence bound ``mu +/- sqrt(beta) s``.

    The upper bound ``mu + sqrt(beta) s`` is returned for maximization (the
    next point maximizes it) and the lower bound ``mu - sqrt(beta) s`` for
    minimization (the next point minimizes it).

    Args:
        mean: Predictive mean ``mu``.
        standard_deviation: Predictive standard deviation ``s >= 0``.
        beta: Exploration weight ``beta >= 0`` (the bound uses its root).
        maximize: Direction of the optimization.

    Returns:
        Upper confidence bound (``maximize``) or lower confidence bound.

    Raises:
        ValueError: If an argument is not finite, ``s < 0``, or
            ``beta < 0``.

    Equation:
        [EQ-DES-BO-003].

    References:
        [REF-SRINIVAS-ETAL-2010].

    """
    mu, sd = _moments(mean, standard_deviation)
    weight = require_finite_real(beta, name="beta")
    if weight < 0.0:
        raise ValueError("beta must be non-negative")
    width = math.sqrt(weight) * sd
    return mu + width if maximize else mu - width


def gp_ucb_beta(iteration: int, candidates: int, delta: float = 0.1) -> float:
    """Return the GP-UCB weight ``2 log(|D| t^2 pi^2 / (6 delta))``.

    This is the schedule of Srinivas et al. for a *finite* decision set
    ``D``; their regret bound further assumes a GP prior with known kernel
    and known noise variance. Inside :func:`bayesian_optimization`, where
    the kernel is re-estimated and the acquisition is refined off the
    candidate set, the schedule is a heuristic choice of ``beta``.

    Args:
        iteration: Round ``t >= 1``.
        candidates: Size ``|D| >= 1`` of the decision set.
        delta: Failure probability ``delta`` in ``(0, 1)``.

    Returns:
        ``beta_t > 0``.

    Raises:
        ValueError: If an argument is out of range.

    Equation:
        [EQ-DES-BO-003].

    References:
        [REF-SRINIVAS-ETAL-2010].

    """
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
        raise ValueError("iteration must be a positive integer")
    if isinstance(candidates, bool) or not isinstance(candidates, int):
        raise ValueError("candidates must be a positive integer")
    if candidates < 1:
        raise ValueError("candidates must be a positive integer")
    level = require_finite_real(delta, name="delta")
    if not 0.0 < level < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    return 2.0 * math.log(candidates * iteration**2 * math.pi**2 / (6.0 * level))


def probability_of_feasibility(
    mean: float, standard_deviation: float, threshold: float = 0.0
) -> float:
    """Return ``P(C <= threshold)`` for ``C ~ N(mean, s^2)``.

    Args:
        mean: Predictive mean of the constraint function.
        standard_deviation: Predictive standard deviation ``s >= 0``.
        threshold: Feasibility threshold (feasible when ``c(x) <= threshold``).

    Returns:
        ``Phi((threshold - mean) / s)``; ``0`` or ``1`` when ``s = 0``.

    Raises:
        ValueError: If an argument is not finite or ``s < 0``.

    Equation:
        [EQ-DES-BO-004].

    References:
        [REF-GARDNER-ETAL-2014].

    """
    mu, sd = _moments(mean, standard_deviation)
    limit = require_finite_real(threshold, name="threshold")
    if sd == 0.0:
        return 1.0 if mu <= limit else 0.0
    return _cdf((limit - mu) / sd)


def _log_feasibility(mean: float, standard_deviation: float) -> float:
    """Return ``log P(C <= 0)`` without underflow.

    References:
        [EQ-DES-BO-004], [EQ-DIST-NORM-007], [REF-NIST-DLMF, Section 7.9].

    """
    if standard_deviation == 0.0:
        return 0.0 if mean <= 0.0 else -math.inf
    return _STANDARD_NORMAL.log_cdf(-mean / standard_deviation)


def constrained_expected_improvement(
    mean: float,
    standard_deviation: float,
    best: float,
    constraint_means: Sequence[float],
    constraint_standard_deviations: Sequence[float],
    *,
    thresholds: Sequence[float] | None = None,
    maximize: bool = False,
) -> float:
    """Return expected improvement times the probability of feasibility.

    With independent Gaussian predictive laws of the objective and of the
    constraint functions ``c_k`` (feasible when ``c_k(x) <= threshold_k``),
    ``cEI = EI * prod_k Phi((threshold_k - mu_k) / s_k)``; ``best`` must be
    the best *feasible* value observed so far.

    Args:
        mean: Predictive mean of the objective.
        standard_deviation: Predictive standard deviation of the objective.
        best: Best feasible objective value so far.
        constraint_means: Predictive means of the constraint functions.
        constraint_standard_deviations: Their predictive standard deviations.
        thresholds: Feasibility thresholds (default all zero).
        maximize: Direction of the optimization.

    Returns:
        Constrained expected improvement ``>= 0``.

    Raises:
        ValueError: If the lengths disagree or a value is invalid.

    Equation:
        [EQ-DES-BO-004].

    References:
        [REF-GARDNER-ETAL-2014], [REF-JONES-SCHONLAU-WELCH-1998].

    """
    means = as_vector(constraint_means, name="constraint_means")
    deviations = as_vector(
        constraint_standard_deviations, name="constraint_standard_deviations"
    )
    limits = (
        (0.0,) * len(means)
        if thresholds is None
        else as_vector(thresholds, name="thresholds")
    )
    if len(deviations) != len(means) or len(limits) != len(means):
        raise ValueError("need one mean, deviation, and threshold per constraint")
    value = expected_improvement(mean, standard_deviation, best, maximize=maximize)
    for mu, sd, limit in zip(means, deviations, limits, strict=True):
        value *= probability_of_feasibility(mu, sd, limit)
    return value


def knowledge_gradient(
    means: Sequence[float],
    standard_deviations: Sequence[float],
    noise_variance: float | Sequence[float],
    *,
    maximize: bool = True,
) -> Vector:
    """Return the knowledge gradient of one measurement of each alternative.

    Beliefs about ``M >= 2`` alternatives are independent,
    ``theta_x ~ N(mu_x, s_x^2)``, and a measurement of ``x`` has noise
    variance ``lambda_x``. One measurement changes the posterior mean of
    ``x`` by ``st_x Z`` with ``st_x = s_x^2 / sqrt(s_x^2 + lambda_x)`` and
    ``Z ~ N(0, 1)``, so the expected increase of the best posterior mean is

    ``KG_x = E[max_y mu'_y] - max_y mu_y = st_x h(zeta_x)``,
    ``zeta_x = -|mu_x - max_(y != x) mu_y| / st_x``.

    For minimization the same formula is applied to ``-mu`` (the expected
    decrease of the smallest posterior mean).

    Args:
        means: Posterior means ``mu_x``.
        standard_deviations: Posterior standard deviations ``s_x >= 0``.
        noise_variance: Measurement noise variance ``lambda >= 0``, common
            or one per alternative.
        maximize: Whether the best alternative has the largest mean
            (default) or the smallest.

    Returns:
        ``KG_x >= 0`` for every alternative (``0`` where ``s_x = 0``).

    Raises:
        ValueError: If fewer than two alternatives are given or a value is
            invalid.

    Equation:
        [EQ-DES-BO-005].

    References:
        [REF-FRAZIER-POWELL-DAYANIK-2008].

    """
    mu = as_vector(means, name="means")
    sd = as_vector(standard_deviations, name="standard_deviations")
    count = len(mu)
    if count < 2 or len(sd) != count:
        raise ValueError("need at least two alternatives with one deviation each")
    if any(value < 0.0 for value in sd):
        raise ValueError("standard_deviations must be non-negative")
    if isinstance(noise_variance, int | float):
        noise: Vector = (require_finite_real(noise_variance, name="noise"),) * count
    else:
        noise = as_vector(noise_variance, name="noise_variance")
    if len(noise) != count or any(value < 0.0 for value in noise):
        raise ValueError("noise_variance must be non-negative, one per alternative")
    signed = mu if maximize else tuple(-value for value in mu)
    order = sorted(range(count), key=lambda index: signed[index], reverse=True)
    top, second = signed[order[0]], signed[order[1]]
    result = []
    for index in range(count):
        variance = sd[index] * sd[index]
        if variance == 0.0:
            result.append(0.0)
            continue
        change = variance / math.sqrt(variance + noise[index])
        rival = second if index == order[0] else top
        result.append(
            change * improvement_function(-abs(signed[index] - rival) / change)
        )
    return tuple(result)


def gaussian_process_joint_posterior(
    model: GaussianProcessRegression, points: Sequence[Sequence[float]]
) -> tuple[Vector, Matrix]:
    """Return the joint posterior mean and covariance of a GP at ``points``.

    ``mean_i = mu(x_i)`` and
    ``C_ij = k(x_i, x_j) - k_i' K^-1 k_j`` with ``K`` the training kernel
    matrix including its noise diagonal (latent function, no observation
    noise added). The kernel matrix is rebuilt from the model's
    hyperparameters in original units and factorized once.

    Args:
        model: Fitted Gaussian process.
        points: Points ``x_1..x_m``.

    Returns:
        Mean vector and symmetric ``m x m`` covariance matrix.

    Raises:
        ValueError: If no point is given or a point has the wrong dimension.

    Equation:
        [EQ-DES-BO-006], [EQ-SUR-GP-003].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 2.2].

    """
    x = tuple(as_vector(point, name="point") for point in points)
    dimension = len(model.points[0])
    if not x or any(len(point) != dimension for point in x):
        raise ValueError("points must be non-empty with the training dimension")
    hyper = model.hyperparameters
    scales, signal = hyper.length_scales, hyper.signal_variance
    rows = []
    for i, left in enumerate(model.points):
        row = [
            kernel_value(model.kernel, left, right, scales, signal)
            for right in model.points
        ]
        row[i] += hyper.noise_variance
        rows.append(tuple(row))
    lower = cholesky(tuple(rows))
    whitened = [
        solve_lower(
            lower,
            tuple(
                kernel_value(model.kernel, point, train, scales, signal)
                for train in model.points
            ),
        )
        for point in x
    ]
    count = len(x)
    covariance = [[0.0] * count for _ in range(count)]
    for i in range(count):
        for j in range(i + 1):
            value = kernel_value(model.kernel, x[i], x[j], scales, signal)
            value -= math.fsum(
                a * b for a, b in zip(whitened[i], whitened[j], strict=True)
            )
            covariance[i][j] = value
            covariance[j][i] = value
    means = tuple(model.predict(point) for point in x)
    return means, tuple(tuple(row) for row in covariance)


@dataclass(frozen=True, slots=True)
class ThompsonDraw:
    """One joint posterior draw on a candidate set and its optimizer.

    Args:
        index: Index of the candidate that optimizes the draw.
        point: That candidate.
        values: The sampled function values at all candidates.

    Equation:
        [EQ-DES-BO-006].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 2.2].

    """

    index: int
    point: Vector
    values: Vector


def thompson_sample(
    model: GaussianProcessRegression,
    candidates: Sequence[Sequence[float]],
    *,
    maximize: bool = False,
    seed: int | RandomStream | None = None,
) -> ThompsonDraw:
    """Draw the GP posterior jointly on a finite set and return its optimizer.

    ``f = mean + L z`` with ``L L' = C + eps I`` the (semidefinite) Cholesky
    factor of the joint posterior covariance of
    :func:`gaussian_process_joint_posterior` and ``z`` standard normal;
    ``eps = 1e-10`` times the largest diagonal entry absorbs rounding in a
    nearly singular ``C``. The selected candidate is the argmin
    (minimization) or argmax (maximization) of the draw, so each candidate
    is chosen with its posterior probability of being the best *of the
    candidate set*. The cost is ``O(m^3)`` in the number of candidates.

    Args:
        model: Fitted Gaussian process.
        candidates: Finite candidate set.
        maximize: Direction of the optimization.
        seed: Integer seed or stream (child ``"thompson"``).

    Returns:
        The optimizer of the draw and the sampled values.

    Raises:
        ValueError: If the candidate set is empty or has the wrong
            dimension.

    Equation:
        [EQ-DES-BO-006].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 2.2].

    """
    x = tuple(as_vector(point, name="candidate") for point in candidates)
    means, covariance = gaussian_process_joint_posterior(model, x)
    count = len(x)
    largest = max(max(covariance[i][i] for i in range(count)), 0.0)
    jitter = 1e-10 * largest
    shifted = tuple(
        tuple(covariance[i][j] + (jitter if i == j else 0.0) for j in range(count))
        for i in range(count)
    )
    factor = cholesky_semidefinite(shifted, tolerance=1e-8)
    generator = resolve_stream(seed).child("thompson").generator()
    normal = [generator.gauss(0.0, 1.0) for _ in range(count)]
    values = tuple(
        means[i] + math.fsum(factor[i][j] * normal[j] for j in range(i + 1))
        for i in range(count)
    )
    chooser = max if maximize else min
    index = chooser(range(count), key=lambda item: values[item])
    return ThompsonDraw(index, x[index], values)


@dataclass(frozen=True, slots=True)
class BayesianOptimizationResult:
    """History and outcome of a Bayesian optimization run.

    Args:
        best_point: Best evaluated feasible point (``None`` if no feasible
            point was found).
        best_value: Its objective value (``nan`` if none).
        points: Every evaluated point in order (initial design first).
        values: Objective values at ``points``.
        constraint_values: Per point, the values of the constraint
            functions (empty tuples without constraints).
        feasible: Per point, whether every constraint is ``<= 0``.
        best_history: Best feasible objective value after each evaluation
            (``nan`` until a feasible point exists).
        acquisition_values: Acquisition value of each sequentially chosen
            point (``log_ei``: log expected improvement; ``ei``: expected
            improvement; ``ucb``: the optimistic bound, negated for
            minimization so that larger is better; ``thompson``: the
            sampled value; ``nan`` for a fallback iteration).
        acquisition: Acquisition name.
        maximize: Direction of the optimization.
        initial_size: Size of the initial space-filling design.
        evaluations: Objective evaluations (equal to the budget).
        constraint_evaluations: Constraint-function evaluations.
        surrogate_fits: Gaussian-process fits (objective and constraints).
        likelihood_evaluations: Kernel factorizations spent in those fits.
        acquisition_evaluations: Surrogate predictions spent on maximizing
            the acquisition.
        fallback_iterations: Iterations in which the surrogate could not be
            fitted (for example constant responses) and a random candidate
            was evaluated instead.
        unconverged_fits: Surrogate fits whose optimizer did not meet its
            tolerance.
        seed: Seed of the root random stream.

    Equation:
        [EQ-DES-BO-001], [EQ-DES-BO-003], [EQ-DES-BO-004], [EQ-DES-BO-006].

    References:
        [REF-JONES-SCHONLAU-WELCH-1998].

    """

    best_point: Vector | None
    best_value: float
    points: tuple[Vector, ...]
    values: Vector
    constraint_values: tuple[Vector, ...]
    feasible: tuple[bool, ...]
    best_history: Vector
    acquisition_values: Vector
    acquisition: str
    maximize: bool
    initial_size: int
    evaluations: int
    constraint_evaluations: int
    surrogate_fits: int
    likelihood_evaluations: int
    acquisition_evaluations: int
    fallback_iterations: int
    unconverged_fits: int
    seed: int

    @property
    def feasible_found(self) -> bool:
        """Return whether at least one evaluated point is feasible.

        References:
            [REF-GARDNER-ETAL-2014]. No equation is evaluated.

        """
        return self.best_point is not None

    def summary(self) -> str:
        """Return a one-sentence account of the run without overclaiming.

        References:
            [REF-JONES-SCHONLAU-WELCH-1998]. No equation is evaluated.

        """
        direction = "maximum" if self.maximize else "minimum"
        if self.best_point is None:
            return (
                f"No feasible point in {self.evaluations} evaluations "
                f"({self.acquisition})."
            )
        return (
            f"Best evaluated {direction} {self.best_value:.6g} after "
            f"{self.evaluations} evaluations ({self.acquisition}, "
            f"{self.fallback_iterations} fallback iterations); a heuristic "
            "search result, not a certified global optimum."
        )


def _positive_integer(value: object, name: str) -> int:
    """Return ``value`` if it is a positive integer.

    Raises:
        ValueError: Otherwise.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def bayesian_optimization(
    objective: Callable[[Vector], float],
    bounds: Sequence[tuple[float, float]],
    *,
    budget: int,
    initial_size: int | None = None,
    acquisition: str = "log_ei",
    maximize: bool = False,
    constraints: Sequence[Callable[[Vector], float]] = (),
    beta: float | None = None,
    delta: float = 0.1,
    kernel: str = "matern52",
    nugget: float = 1e-10,
    regularization: GaussianProcessRegularization | None = _DEFAULT_REGULARIZATION,
    restarts: int = 2,
    candidates: int | None = None,
    local_starts: int = 3,
    local_evaluations: int = 60,
    initial_design: str = "lhs",
    seed: int | RandomStream | None = None,
) -> BayesianOptimizationResult:
    """Optimize an expensive function over a box by Bayesian optimization.

    1. An initial Latin hypercube design of ``initial_size`` points is
       evaluated (stream child ``"initial"``).
    2. In each of the remaining ``budget - initial_size`` iterations a GP is
       fitted to all objective values (child ``("fit", n)``; one more GP per
       constraint function), the acquisition is evaluated on a fresh Latin
       hypercube candidate set (child ``("candidates", n)``), the
       ``local_starts`` best candidates are each refined by a bounded
       Nelder-Mead search of at most ``local_evaluations`` acquisition
       evaluations, and the best point found is evaluated. A point that
       coincides with an evaluated one is replaced by the best unevaluated
       candidate.

    Acquisitions: ``"log_ei"`` (default) and ``"ei"`` use the expected
    improvement over the best feasible value; with constraints (each
    feasible when ``c_k(x) <= 0``) the expected improvement is multiplied by
    the probabilities of feasibility of independent GP models of the
    ``c_k`` [EQ-DES-BO-004], and while no feasible point is known the
    acquisition is the probability of feasibility alone (a library choice).
    ``"ucb"`` uses the optimistic bound with ``beta``; ``beta=None`` uses
    :func:`gp_ucb_beta` with the iteration number and the candidate count,
    a heuristic here. ``"thompson"`` takes the optimizer of one joint
    posterior draw on the candidate set, without local refinement. ``"ucb"``
    and ``"thompson"`` do not support constraints.

    If a surrogate cannot be fitted (for example all responses equal), the
    iteration evaluates the first candidate instead and is counted in
    ``fallback_iterations``. This also applies to the constraint surrogates:
    while a constraint function has returned the same value at every
    evaluated point (for example a constraint that is satisfied with a
    constant margin), every iteration is a fallback and the run is a random
    search; check ``fallback_iterations``.

    The objective and the constraints are assumed deterministic. The
    surrogate interpolates the data up to the nugget and the incumbent is
    the smallest (largest) *observed* value, so with a noisy objective the
    reported ``best_value`` is biased towards the favourable tail of the
    noise and the expected improvement is taken over a noisy incumbent: for
    an objective that is pure ``N(0, 1)`` noise a run of 25 evaluations
    reported a minimum of -2.0. Evaluated points are never repeated exactly
    (a proposal equal to an evaluated point is replaced), but proposals
    arbitrarily close to evaluated points are possible.

    Args:
        objective: Function of a point (a tuple of floats) returning a
            finite float.
        bounds: Finite ``(low, high)`` per coordinate with ``low < high``.
        budget: Total number of objective evaluations.
        initial_size: Size of the initial design (default
            ``min(budget, max(4, 3 d))``; at least two).
        acquisition: One of :data:`ACQUISITIONS`.
        maximize: Maximize instead of minimize.
        constraints: Constraint functions, feasible when ``<= 0``; each is
            evaluated at every point.
        beta: Confidence-bound weight for ``"ucb"``.
        delta: Failure probability of the GP-UCB schedule.
        kernel: GP kernel name.
        nugget: GP nugget in standardized output units.
        regularization: GP hyperparameter regularization (default
            ``GaussianProcessRegularization()``; ``None`` for the plain
            maximum-likelihood fit).
        restarts: Optimizer starts of each GP fit.
        candidates: Candidate-set size (default ``100 d`` capped at 400;
            ``50 d`` capped at 100 for ``"thompson"``, whose cost is cubic
            in it).
        local_starts: Number of candidates refined locally.
        local_evaluations: Acquisition evaluations per local refinement.
        initial_design: ``"lhs"`` or ``"maximin"`` (maximin Latin
            hypercube).
        seed: Integer seed or stream; ``None`` is not reproducible.

    Returns:
        The evaluated points, the best feasible one, and the cost counters.

    Raises:
        ValueError: If an option is invalid, a bound is not finite, or the
            objective or a constraint returns a non-finite value.

    Equation:
        [EQ-DES-BO-001], [EQ-DES-BO-002], [EQ-DES-BO-003], [EQ-DES-BO-004],
        [EQ-DES-BO-006].

    References:
        [REF-JONES-SCHONLAU-WELCH-1998], [REF-AMENT-ETAL-2023],
        [REF-SRINIVAS-ETAL-2010], [REF-GARDNER-ETAL-2014].

    """
    box = []
    for pair in bounds:
        low = require_finite_real(pair[0], name="lower bound")
        high = require_finite_real(pair[1], name="upper bound")
        if not low < high:
            raise ValueError("each bound must satisfy low < high")
        box.append((low, high))
    dimension = len(box)
    if dimension == 0:
        raise ValueError("bounds must be non-empty")
    total = _positive_integer(budget, "budget")
    if acquisition not in ACQUISITIONS:
        raise ValueError(f"acquisition must be one of {ACQUISITIONS}")
    if initial_design not in INITIAL_DESIGNS:
        raise ValueError(f"initial_design must be one of {INITIAL_DESIGNS}")
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    constraint_functions = tuple(constraints)
    if constraint_functions and acquisition in ("ucb", "thompson"):
        raise ValueError("constraints need the 'ei' or 'log_ei' acquisition")
    if initial_size is None:
        start_size = min(total, max(4, 3 * dimension))
    else:
        start_size = _positive_integer(initial_size, "initial_size")
    if start_size < 2 or start_size > total:
        raise ValueError("initial_size must lie between 2 and the budget")
    if candidates is None:
        if acquisition == "thompson":
            pool = min(100, 50 * dimension)
        else:
            pool = min(400, 100 * dimension)
    else:
        pool = _positive_integer(candidates, "candidates")
    starts = _positive_integer(local_starts, "local_starts")
    local_budget = _positive_integer(local_evaluations, "local_evaluations")
    _positive_integer(restarts, "restarts")
    weight: float | None = None
    if beta is not None:
        weight = require_finite_real(beta, name="beta")
        if weight < 0.0:
            raise ValueError("beta must be non-negative")
    level = require_finite_real(delta, name="delta")
    if not 0.0 < level < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    root = resolve_stream(seed)

    def scale(unit: Sequence[float]) -> Vector:
        """Map a point of the unit cube to the box.

        References:
            [REF-MCKAY-BECKMAN-CONOVER-1979]. No equation is evaluated.

        """
        return tuple(
            low + (high - low) * u for u, (low, high) in zip(unit, box, strict=True)
        )

    points: list[Vector] = []
    values: list[float] = []
    constraint_values: list[Vector] = []
    feasible: list[bool] = []
    history: list[float] = []
    incumbent: list[float] = [math.nan]
    incumbent_point: list[Vector | None] = [None]

    def evaluate(point: Vector) -> None:
        """Run the objective and the constraints at one point and record it.

        Raises:
            ValueError: If a returned value is not finite.

        References:
            [REF-JONES-SCHONLAU-WELCH-1998]. No equation is evaluated.

        """
        value = require_finite_real(objective(point), name="objective value")
        limits = tuple(
            require_finite_real(function(point), name="constraint value")
            for function in constraint_functions
        )
        allowed = all(limit <= 0.0 for limit in limits)
        points.append(point)
        values.append(value)
        constraint_values.append(limits)
        feasible.append(allowed)
        current = incumbent[0]
        better = math.isnan(current) or (
            value > current if maximize else value < current
        )
        if allowed and better:
            incumbent[0] = value
            incumbent_point[0] = point
        history.append(incumbent[0])

    if initial_design == "maximin":
        start = maximin_latin_hypercube(
            start_size, dimension, seed=root.child("initial")
        )
    else:
        start = latin_hypercube(start_size, dimension, seed=root.child("initial"))
    for unit in start.points:
        evaluate(scale(unit))

    fits = 0
    factorizations = 0
    unconverged = 0
    predictions = 0
    fallbacks = 0
    chosen: list[float] = []
    for iteration in range(1, total - start_size + 1):
        count = len(points)
        pool_points = tuple(
            scale(unit)
            for unit in latin_hypercube(
                pool, dimension, seed=root.child("candidates", count)
            ).points
        )
        seen = set(points)
        try:
            model = fit_gaussian_process(
                points,
                values,
                kernel=kernel,
                nugget=nugget,
                restarts=restarts,
                seed=root.child("fit", count),
                regularization=regularization,
            )
            constraint_models = tuple(
                fit_gaussian_process(
                    points,
                    [row[k] for row in constraint_values],
                    kernel=kernel,
                    nugget=nugget,
                    restarts=restarts,
                    seed=root.child("fit-constraint", k, count),
                    regularization=regularization,
                )
                for k in range(len(constraint_functions))
            )
        except ValueError:
            fallbacks += 1
            chosen.append(math.nan)
            evaluate(next(p for p in pool_points if p not in seen))
            continue
        for fitted in (model, *constraint_models):
            fits += 1
            factorizations += fitted.evaluations
            unconverged += 0 if fitted.converged else 1
        if acquisition == "thompson":
            draw = thompson_sample(
                model,
                pool_points,
                maximize=maximize,
                seed=root.child("draw", count),
            )
            predictions += pool
            chosen.append(draw.values[draw.index])
            evaluate(draw.point)
            continue
        best = incumbent[0]
        if acquisition == "ucb":
            ucb_weight = (
                gp_ucb_beta(iteration, pool, level) if weight is None else weight
            )
        else:
            ucb_weight = 0.0

        def score(
            point: Vector,
            *,
            model: GaussianProcessRegression = model,
            constraint_models: tuple[
                GaussianProcessRegression, ...
            ] = constraint_models,
            best: float = best,
            ucb_weight: float = ucb_weight,
        ) -> float:
            """Return the acquisition at ``point`` (larger is better).

            The value is on the log scale for ``"log_ei"`` and whenever no
            feasible point is known yet (log probability of feasibility).

            References:
                [EQ-DES-BO-001], [EQ-DES-BO-002], [EQ-DES-BO-003],
                [EQ-DES-BO-004].

            """
            mean, variance = model.predict_with_variance(point)
            deviation = math.sqrt(variance)
            if acquisition == "ucb":
                bound = confidence_bound(mean, deviation, ucb_weight, maximize=maximize)
                return bound if maximize else -bound
            log_feasible = 0.0
            for constraint_model in constraint_models:
                c_mean, c_variance = constraint_model.predict_with_variance(point)
                log_feasible += _log_feasibility(c_mean, math.sqrt(c_variance))
            if math.isnan(best):
                return log_feasible
            if acquisition == "log_ei":
                return log_feasible + log_expected_improvement(
                    mean, deviation, best, maximize=maximize
                )
            return math.exp(log_feasible) * expected_improvement(
                mean, deviation, best, maximize=maximize
            )

        scored = sorted(
            ((score(point), point) for point in pool_points),
            key=lambda item: item[0],
            reverse=True,
        )
        predictions += pool
        best_score, best_point = scored[0]
        for start_score, start_point in scored[:starts]:
            if not math.isfinite(start_score):
                continue
            local = minimize_nelder_mead(
                lambda z: -score(z),
                start_point,
                initial_step=0.05,
                tolerance=1e-9,
                max_evaluations=local_budget,
                bounds=box,
            )
            predictions += local.evaluations
            if -local.value > best_score:
                best_score, best_point = -local.value, local.x
        if best_point in seen:
            best_score, best_point = next(
                item for item in scored if item[1] not in seen
            )
        chosen.append(best_score)
        evaluate(best_point)

    return BayesianOptimizationResult(
        incumbent_point[0],
        incumbent[0],
        tuple(points),
        tuple(values),
        tuple(constraint_values),
        tuple(feasible),
        tuple(history),
        tuple(chosen),
        acquisition,
        maximize,
        start_size,
        len(points),
        len(points) * len(constraint_functions),
        fits,
        factorizations,
        predictions,
        fallbacks,
        unconverged,
        root.seed,
    )
