"""Reliability sensitivity: FORM, score-function, and failure-indicator indices.

Experimental (added in 0.28).

Three complementary answers to "how does the failure probability respond?":

* **FORM sensitivities** (:func:`form_sensitivities`): for a parametrised
  standard-normal limit state ``G(u; theta)`` the derivative of the
  reliability index follows from the design point,
  ``d beta / d theta = (dG/dtheta)(u*) / ||grad_u G(u*)||``, and
  ``d P_f / d theta = -phi(beta) d beta / d theta`` to first order. It covers
  distribution parameters (through the isoprobabilistic map) and deterministic
  design parameters of the limit state alike, and is checked against central
  finite differences of repeated FORM runs
  (:func:`form_sensitivities_finite_difference`).
* **Score-function (likelihood-ratio) estimators**
  (:func:`score_function_monte_carlo`, :func:`score_function_subset`): for
  independent normal inputs ``X_i ~ N(mu_i, sigma_i^2)``,
  ``d P_f / d theta = E[1{g(X) <= 0} s(X; theta)]`` with the score
  ``s = d ln f / d theta``. The Monte Carlo version averages
  ``1_F s`` over crude samples; the subset version averages ``1_F s`` over the
  last conditional level and multiplies by the product of the intermediate
  conditional probabilities. Standard errors are reported.
* **Failure-indicator Sobol indices** (:func:`failure_indicator_sobol`):
  first-order indices ``Var(E[1_F | X_i]) / Var(1_F)`` estimated from
  independent samples by equal-count binning with bias correction.

Assumptions: independent normal inputs for the score estimators; failure is
``g <= 0``. Score-function estimators differentiate with respect to
distribution parameters only; a parameter of ``g`` itself (a design parameter)
requires the FORM route or a surface-integral estimator. All estimators are
unbiased or consistent but rare-event variance can be large; see each result's
standard errors.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector, as_vector, dot, norm
from probstat.modeling import standard_normal_cdf
from probstat.optimization import central_gradient

from .model_based import _hlrf_iterate
from .simulation import _level_cov

ParametricLimitState = Callable[[Vector, Vector], float]
ScalarLimitState = Callable[[Vector], float]

_INV_SQRT_TWO_PI = 1.0 / math.sqrt(2.0 * math.pi)


def _positive(value: object, name: str) -> float:
    """Return a finite strictly positive float.

    References:
        [EQ-REL-RSENS-001], [REF-HOHENBICHLER-RACKWITZ-1986].
        No equation is evaluated.

    """
    number = require_finite_real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _int_at_least(value: object, name: str, minimum: int) -> int:
    """Return an integer at least ``minimum``.

    References:
        [EQ-REL-RSENS-001], [REF-HOHENBICHLER-RACKWITZ-1986].
        No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def normal_parameter_names(inputs: int, design: int = 0) -> tuple[str, ...]:
    """Return the names of the parameter vector of a normal-input problem.

    The order is ``mu_1..mu_n, sigma_1..sigma_n, d_1..d_m`` (means, standard
    deviations, design parameters), the layout of
    :func:`normal_input_limit_state`.

    Args:
        inputs: Number of normal inputs ``n``.
        design: Number of deterministic design parameters ``m``.

    Returns:
        Tuple of ``2 n + m`` names.

    Raises:
        TypeError: If an argument is not an integer.
        ValueError: If ``inputs`` is not positive or ``design`` negative.

    Equation:
        [EQ-REL-RSENS-001].

    References:
        [REF-DITLEVSEN-MADSEN-1996].

    """
    n = _int_at_least(inputs, "inputs", 1)
    m = _int_at_least(design, "design", 0)
    return (
        tuple(f"mu_{i + 1}" for i in range(n))
        + tuple(f"sigma_{i + 1}" for i in range(n))
        + tuple(f"d_{j + 1}" for j in range(m))
    )


def normal_input_limit_state(
    limit_state: Callable[..., float], inputs: int, design: int = 0
) -> ParametricLimitState:
    """Return ``G(u; theta) = g(mu + sigma u [, d])`` for independent normals.

    The parameter vector is ``theta = (mu, sigma, d)`` (see
    :func:`normal_parameter_names`). With ``design = 0`` the physical limit
    state is called as ``g(x)``, otherwise as ``g(x, d)``.

    Args:
        limit_state: Physical limit state ``g`` with failure ``g <= 0``.
        inputs: Number of normal inputs.
        design: Number of design parameters.

    Returns:
        Callable ``G(u, theta)`` in standard-normal space.

    Raises:
        TypeError: If ``limit_state`` is not callable.
        ValueError: If the theta length is wrong at call time or a standard
            deviation is not positive.

    Equation:
        [EQ-REL-RSENS-001].

    References:
        [REF-DITLEVSEN-MADSEN-1996], [REF-HOHENBICHLER-RACKWITZ-1986].

    """
    if not callable(limit_state):
        raise TypeError("limit_state must be callable")
    n = _int_at_least(inputs, "inputs", 1)
    m = _int_at_least(design, "design", 0)

    def standard(u: Vector, theta: Vector) -> float:
        """Evaluate ``g(mu + sigma u, d)``.

        References:
            [EQ-REL-RSENS-001]. No equation is evaluated.

        """
        if len(theta) != 2 * n + m or len(u) != n:
            raise ValueError("u or theta has the wrong length")
        x = tuple(theta[i] + theta[n + i] * u[i] for i in range(n))
        if any(theta[n + i] <= 0.0 for i in range(n)):
            raise ValueError("standard deviations must be positive")
        if m == 0:
            return float(limit_state(x))
        return float(limit_state(x, tuple(theta[2 * n :])))

    return standard


@dataclass(frozen=True, slots=True)
class FORMSensitivityResult:
    """FORM design point and parameter sensitivities.

    Args:
        beta: Signed reliability index.
        probability: First-order failure probability ``Phi(-beta)``.
        design_point: Design point ``u*``.
        alpha: Unit vector ``-grad G / ||grad G||``.
        gradient_norm: ``||grad_u G(u*)||``.
        dbeta: ``d beta / d theta_k``.
        dprobability: ``d Phi(-beta) / d theta_k = -phi(beta) d beta / d theta_k``.
        parameters: Parameter vector ``theta``.
        evaluations: Limit-state evaluations (search plus parameter partials).
        converged: Whether the FORM search met its tolerance.
        stop_reason: HL-RF termination message.

    Equation:
        [EQ-REL-RSENS-001], [EQ-REL-RSENS-002].

    References:
        [REF-HOHENBICHLER-RACKWITZ-1986], [REF-DITLEVSEN-MADSEN-1996].

    """

    beta: float
    probability: float
    design_point: Vector
    alpha: Vector
    gradient_norm: float
    dbeta: Vector
    dprobability: Vector
    parameters: Vector
    evaluations: int
    converged: bool
    stop_reason: str


def _form_search(
    limit_state: ParametricLimitState,
    theta: Vector,
    dimension: int,
    start: Vector,
    gradient_u: Callable[[Vector, Vector], Sequence[float]] | None,
    max_iterations: int,
    tolerance: float,
) -> tuple[Vector, Vector, bool, str, int]:
    """Return ``(u*, grad, converged, message, evaluations)`` of one FORM run.

    References:
        [REF-HASOFER-LIND-1974], [REF-DITLEVSEN-MADSEN-1996].
        No equation is evaluated.

    """
    count = 0

    def function(u: Vector) -> float:
        """Evaluate ``G(u; theta)`` and count the call.

        References:
            [EQ-REL-RSENS-001]. No equation is evaluated.

        """
        nonlocal count
        count += 1
        return float(limit_state(u, theta))

    def gradient(u: Vector) -> Vector:
        """Return the analytic or central-difference gradient in ``u``.

        References:
            [EQ-REL-RSENS-001]. No equation is evaluated.

        """
        nonlocal count
        if gradient_u is not None:
            return as_vector(gradient_u(u, theta), name="gradient")
        grad, calls = central_gradient(function, u, relative_step=1e-6)
        count += calls
        return grad

    u, _, grad, _, ok, message = _hlrf_iterate(
        function, start, gradient, True, max_iterations, tolerance
    )
    if len(u) != dimension:
        raise ValueError("design point has the wrong dimension")
    return u, grad, ok, message, count


def form_sensitivities(
    limit_state: ParametricLimitState,
    parameters: Sequence[float],
    dimension: int,
    *,
    start: Sequence[float] | None = None,
    gradient_u: Callable[[Vector, Vector], Sequence[float]] | None = None,
    parameter_step: float = 1e-6,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> FORMSensitivityResult:
    """Return the FORM index and its derivatives with respect to parameters.

    Let ``beta(theta) = min ||u||`` subject to ``G(u; theta) = 0`` with the
    design point ``u*``. The envelope theorem gives

    ``d beta / d theta_k = (dG/dtheta_k)(u*) / ||grad_u G(u*)||``

    and ``d P_f1 / d theta_k = -phi(beta) d beta / d theta_k``. The partials
    ``dG/dtheta_k`` are central differences of ``G`` at fixed ``u*`` with step
    ``parameter_step * max(1, |theta_k|)``, so the cost is ``2 |theta|``
    evaluations in addition to the search. Distribution parameters and design
    parameters enter in the same way, because both are columns of ``theta``
    in ``G(u; theta)``.

    Args:
        limit_state: ``G(u, theta)`` with failure ``G <= 0``.
        parameters: Parameter vector ``theta``.
        dimension: Dimension of ``u``.
        start: Optional starting point of the search (default the origin).
        gradient_u: Optional analytic ``grad_u G(u, theta)``.
        parameter_step: Relative central-difference step for ``dG/dtheta``.
        max_iterations: Maximum HL-RF iterations.
        tolerance: Relative HL-RF tolerance.

    Returns:
        Index, design point, derivatives, and the convergence flag.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If lengths are inconsistent or the gradient vanishes.

    Equation:
        [EQ-REL-RSENS-001], [EQ-REL-RSENS-002].

    References:
        [REF-HOHENBICHLER-RACKWITZ-1986],
        [REF-DITLEVSEN-MADSEN-1996].

    """
    if not callable(limit_state):
        raise TypeError("limit_state must be callable")
    theta = as_vector(parameters, name="parameters")
    n = _int_at_least(dimension, "dimension", 1)
    u0: Vector = (0.0,) * n if start is None else as_vector(start, name="start")
    if len(u0) != n:
        raise ValueError("start length must equal dimension")
    step = _positive(parameter_step, "parameter_step")
    limit = _int_at_least(max_iterations, "max_iterations", 1)
    tol = _positive(tolerance, "tolerance")
    u, grad, ok, message, count = _form_search(
        limit_state, theta, n, u0, gradient_u, limit, tol
    )
    length = norm(grad)
    alpha = tuple(-g / length for g in grad)
    beta = dot(alpha, u)

    def at_fixed_point(values: Vector) -> float:
        """Evaluate ``G(u*; theta)`` as a function of ``theta``.

        References:
            [EQ-REL-RSENS-001]. No equation is evaluated.

        """
        return float(limit_state(u, values))

    steps = tuple(step * max(1.0, abs(t)) for t in theta)
    partials, calls = central_gradient(at_fixed_point, theta, step=steps)
    dbeta = tuple(p / length for p in partials)
    density = _INV_SQRT_TWO_PI * math.exp(-0.5 * beta * beta)
    return FORMSensitivityResult(
        beta=beta,
        probability=standard_normal_cdf(-beta),
        design_point=u,
        alpha=alpha,
        gradient_norm=length,
        dbeta=dbeta,
        dprobability=tuple(-density * d for d in dbeta),
        parameters=theta,
        evaluations=count + calls,
        converged=ok,
        stop_reason=message,
    )


def form_sensitivities_finite_difference(
    limit_state: ParametricLimitState,
    parameters: Sequence[float],
    dimension: int,
    *,
    relative_step: float = 1e-3,
    start: Sequence[float] | None = None,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
) -> Vector:
    """Return ``d beta / d theta`` by central differences of full FORM runs.

    Every parameter is perturbed by ``+-h`` with ``h = relative_step
    max(1, |theta_k|)`` and FORM is re-run from the same start point. This
    is the independent check of :func:`form_sensitivities`; it costs two
    searches per parameter.

    Args:
        limit_state: ``G(u, theta)`` with failure ``G <= 0``.
        parameters: Parameter vector.
        dimension: Dimension of ``u``.
        relative_step: Relative perturbation.
        start: Optional start of the base search.
        max_iterations: Maximum HL-RF iterations per search.
        tolerance: Relative HL-RF tolerance of every search.

    Returns:
        ``(beta(theta + h e_k) - beta(theta - h e_k)) / (2 h)`` per parameter.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a search does not converge or arguments are invalid.

    Equation:
        [EQ-REL-RSENS-001].

    References:
        [REF-HOHENBICHLER-RACKWITZ-1986], [REF-HASOFER-LIND-1974].

    """
    theta = as_vector(parameters, name="parameters")
    n = _int_at_least(dimension, "dimension", 1)
    rel = _positive(relative_step, "relative_step")
    limit = _int_at_least(max_iterations, "max_iterations", 1)
    tol = _positive(tolerance, "tolerance")
    u0: Vector = (0.0,) * n if start is None else as_vector(start, name="start")
    _, _, ok, message, _ = _form_search(limit_state, theta, n, u0, None, limit, tol)
    if not ok:
        raise ValueError(f"base FORM search failed: {message}")

    def beta_at(values: Vector) -> float:
        """Return the signed index at parameters ``values``.

        References:
            [EQ-REL-RSENS-001]. No equation is evaluated.

        """
        u, grad, converged, text, _ = _form_search(
            limit_state, values, n, u0, None, limit, tol
        )
        if not converged:
            raise ValueError(f"perturbed FORM search failed: {text}")
        return -dot(grad, u) / norm(grad)

    out: list[float] = []
    for k, t in enumerate(theta):
        h = rel * max(1.0, abs(t))
        plus = theta[:k] + (t + h,) + theta[k + 1 :]
        minus = theta[:k] + (t - h,) + theta[k + 1 :]
        out.append((beta_at(plus) - beta_at(minus)) / (2.0 * h))
    return tuple(out)


def normal_scores(
    x: Sequence[float], mean: Sequence[float], std: Sequence[float]
) -> Vector:
    """Return the score of independent normals with respect to ``(mu, sigma)``.

    ``d ln f / d mu_i = (x_i - mu_i) / sigma_i^2`` and
    ``d ln f / d sigma_i = ((x_i - mu_i)^2 / sigma_i^2 - 1) / sigma_i``,
    ordered ``mu_1..mu_n, sigma_1..sigma_n``.

    Args:
        x: Point ``x``.
        mean: Means ``mu``.
        std: Standard deviations ``sigma`` (positive).

    Returns:
        Vector of length ``2 n``.

    Raises:
        ValueError: If lengths differ or a standard deviation is not positive.

    Equation:
        [EQ-REL-RSENS-003].

    References:
        [REF-RUBINSTEIN-KROESE-2004], [REF-PAPAIOANNOU-BREITUNG-STRAUB-2018].

    """
    if not len(x) == len(mean) == len(std):
        raise ValueError("x, mean, and std must have equal length")
    if any(s <= 0.0 for s in std):
        raise ValueError("standard deviations must be positive")
    z = [(xi - m) / s for xi, m, s in zip(x, mean, std, strict=True)]
    return tuple(zi / s for zi, s in zip(z, std, strict=True)) + tuple(
        (zi * zi - 1.0) / s for zi, s in zip(z, std, strict=True)
    )


def score_function_estimator(
    failed: Sequence[bool], scores: Sequence[Sequence[float]]
) -> tuple[Vector, Vector]:
    """Return ``mean(1_F s)`` and its standard error from independent samples.

    Args:
        failed: Failure indicators ``1{g(x_j) <= 0}``.
        scores: Score vectors ``s(x_j; theta)`` of equal length.

    Returns:
        ``(estimates, standard_errors)`` with ``se = sd / sqrt(N)``.

    Raises:
        ValueError: If fewer than two samples are given or lengths differ.

    Equation:
        [EQ-REL-RSENS-004].

    References:
        [REF-RUBINSTEIN-KROESE-2004], [REF-PAPAIOANNOU-BREITUNG-STRAUB-2018].

    """
    n = len(failed)
    if n < 2 or len(scores) != n:
        raise ValueError("need at least two samples with one score per sample")
    width = len(scores[0])
    sums = [0.0] * width
    squares = [0.0] * width
    for flag, s in zip(failed, scores, strict=True):
        if len(s) != width:
            raise ValueError("score vectors must have equal length")
        if flag:
            for k in range(width):
                sums[k] += s[k]
                squares[k] += s[k] * s[k]
    mean = [v / n for v in sums]
    se = [
        math.sqrt(max(0.0, (squares[k] / n - mean[k] ** 2) / (n - 1)))
        for k in range(width)
    ]
    return tuple(mean), tuple(se)


@dataclass(frozen=True, slots=True)
class ScoreSensitivityResult:
    """Score-function estimate of ``dP_f / d(mu, sigma)``.

    Args:
        method: ``"monte-carlo"`` or ``"subset-simulation"``.
        probability: Failure-probability estimate from the same samples.
        probability_standard_error: Its standard error (c.o.v. of Au and Beck
            for subset simulation).
        derivatives: ``dP_f / d theta_k``, ordered ``mu_1..mu_n,
            sigma_1..sigma_n``.
        standard_errors: Standard errors of ``derivatives``. For subset
            simulation they combine the batch-means error over chains with the
            level c.o.v. of the conditioning probability, assuming the two are
            uncorrelated (an approximation).
        parameter_names: Names matching ``derivatives``.
        samples: Samples (per level for subset simulation).
        failures: Failed samples in the final level.
        levels: Number of conditional levels (one for Monte Carlo).
        evaluations: Limit-state evaluations.
        converged: ``False`` if no failure was observed in the final level.
        stop_reason: Human-readable status.

    Equation:
        [EQ-REL-RSENS-004], [EQ-REL-RSENS-005].

    References:
        [REF-RUBINSTEIN-KROESE-2004], [REF-SONG-LU-QIAO-2009],
        [REF-PAPAIOANNOU-BREITUNG-STRAUB-2018].

    """

    method: str
    probability: float
    probability_standard_error: float
    derivatives: Vector
    standard_errors: Vector
    parameter_names: tuple[str, ...]
    samples: int
    failures: int
    levels: int
    evaluations: int
    converged: bool
    stop_reason: str


def _check_normal_inputs(
    mean: Sequence[float], std: Sequence[float]
) -> tuple[Vector, Vector]:
    """Return validated mean and standard-deviation vectors.

    References:
        [EQ-REL-RSENS-003], [REF-RUBINSTEIN-KROESE-2004].
        No equation is evaluated.

    """
    m = as_vector(mean, name="mean")
    s = as_vector(std, name="std")
    if len(m) != len(s) or not m:
        raise ValueError("mean and std must be non-empty and of equal length")
    if any(v <= 0.0 for v in s):
        raise ValueError("std must be positive")
    return m, s


def score_function_monte_carlo(
    limit_state: ScalarLimitState,
    mean: Sequence[float],
    std: Sequence[float],
    *,
    samples: int,
    seed: int | RandomStream | None = None,
) -> ScoreSensitivityResult:
    """Estimate ``dP_f/d(mu, sigma)`` from crude Monte Carlo samples.

    ``dP_f/d theta = E[1{g(X) <= 0} s(X; theta)]`` is unbiased: the same
    ``N`` independent draws of ``X_i ~ N(mu_i, sigma_i^2)`` give ``P_f`` and
    all ``2 n`` derivatives. The standard error is ``sd(1_F s_k) / sqrt(N)``.

    Args:
        limit_state: Physical limit state ``g(x)`` with failure ``g <= 0``.
        mean: Input means.
        std: Input standard deviations.
        samples: Number of samples (at least two).
        seed: Integer seed, stream, or ``None``.

    Returns:
        Derivatives with standard errors; ``converged`` is ``False`` when no
        sample failed.

    Raises:
        TypeError: If ``limit_state`` is not callable or ``samples`` is not an
            integer.
        ValueError: If inputs are invalid.

    Equation:
        [EQ-REL-RSENS-003], [EQ-REL-RSENS-004].

    References:
        [REF-RUBINSTEIN-KROESE-2004], [REF-PAPAIOANNOU-BREITUNG-STRAUB-2018].

    """
    if not callable(limit_state):
        raise TypeError("limit_state must be callable")
    m, s = _check_normal_inputs(mean, std)
    n = _int_at_least(samples, "samples", 2)
    generator = resolve_stream(seed).child("score-mc").generator()
    failed: list[bool] = []
    scores: list[Vector] = []
    for _ in range(n):
        x = tuple(
            mu + sig * generator.gauss(0.0, 1.0) for mu, sig in zip(m, s, strict=True)
        )
        flag = limit_state(x) <= 0.0
        failed.append(flag)
        scores.append(normal_scores(x, m, s) if flag else (0.0,) * (2 * len(m)))
    count = sum(failed)
    estimate, se = score_function_estimator(failed, scores)
    p = count / n
    return ScoreSensitivityResult(
        method="monte-carlo",
        probability=p,
        probability_standard_error=math.sqrt(p * (1.0 - p) / n),
        derivatives=estimate,
        standard_errors=se,
        parameter_names=normal_parameter_names(len(m))[: 2 * len(m)],
        samples=n,
        failures=count,
        levels=1,
        evaluations=n,
        converged=count > 0,
        stop_reason="converged" if count else "no failure observed",
    )


def score_function_subset(
    limit_state: ScalarLimitState,
    mean: Sequence[float],
    std: Sequence[float],
    *,
    samples_per_level: int = 1000,
    level_probability: float = 0.1,
    correlation: float = 0.8,
    max_levels: int = 20,
    seed: int | RandomStream | None = None,
) -> ScoreSensitivityResult:
    """Estimate ``dP_f/d(mu, sigma)`` from subset-simulation samples.

    Subset simulation runs in standard-normal space ``x = mu + sigma u`` with
    conditional-sampling chains (correlation ``rho``). Let ``F_{m-1}`` be the
    last intermediate set (``F_0`` is the whole space) with
    ``P(F_{m-1}) = p0^{m-1}``. Because ``F subset F_{m-1}``,

    ``dP_f/d theta = P(F_{m-1}) E[1_F s | F_{m-1}]``

    and the final-level samples are draws from ``f(. | F_{m-1})``, so the
    estimator is ``p0^{m-1} (1/N) sum_j 1_F(x_j) s(x_j)``. Intermediate
    thresholds are treated as fixed sets, which leaves a small
    data-dependence bias of the Au-Beck type. The standard error combines
    the batch-means error over chains with the Au-Beck c.o.v. of the
    conditioning probability, treated as independent.

    Args:
        limit_state: Physical limit state ``g(x)`` with failure ``g <= 0``.
        mean: Input means.
        std: Input standard deviations.
        samples_per_level: Samples ``N`` per level; ``N p0`` must be a positive
            integer dividing ``N``.
        level_probability: Conditional probability ``p0`` in ``(0, 0.5]``.
        correlation: Conditional-sampling correlation in ``[0, 1)``.
        max_levels: Maximum number of levels.
        seed: Integer seed, stream, or ``None``.

    Returns:
        Probability, derivatives, and standard errors.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If options are inconsistent or limit-state values tie at
            an intermediate threshold.

    Equation:
        [EQ-REL-RSENS-003], [EQ-REL-RSENS-005].

    References:
        [REF-SONG-LU-QIAO-2009], [REF-AU-BECK-2001],
        [REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015].

    """
    if not callable(limit_state):
        raise TypeError("limit_state must be callable")
    m, s = _check_normal_inputs(mean, std)
    n_inputs = len(m)
    total = _int_at_least(samples_per_level, "samples_per_level", 10)
    p0 = require_finite_real(level_probability, name="level_probability")
    if not 0.0 < p0 <= 0.5:
        raise ValueError("level_probability must lie in (0, 0.5]")
    seeds_count = round(total * p0)
    if seeds_count < 1 or abs(seeds_count - total * p0) > 1e-9:
        raise ValueError("samples_per_level * level_probability must be an integer")
    if total % seeds_count != 0:
        raise ValueError("samples_per_level * level_probability must divide N")
    rho = require_finite_real(correlation, name="correlation")
    if not 0.0 <= rho < 1.0:
        raise ValueError("correlation must lie in [0, 1)")
    levels_limit = _int_at_least(max_levels, "max_levels", 1)
    chain_length = total // seeds_count
    sigma_step = math.sqrt(1.0 - rho * rho)
    generator: random.Random = resolve_stream(seed).child("score-subset").generator()

    def physical(u: Vector) -> Vector:
        """Map standard-normal ``u`` to physical ``x``.

        References:
            [EQ-REL-RSENS-003]. No equation is evaluated.

        """
        return tuple(mu + sig * ui for mu, sig, ui in zip(m, s, u, strict=True))

    def value(u: Vector) -> float:
        """Evaluate ``g(mu + sigma u)``.

        References:
            [EQ-REL-RSENS-003]. No equation is evaluated.

        """
        return float(limit_state(physical(u)))

    points: list[Vector] = [
        tuple(generator.gauss(0.0, 1.0) for _ in range(n_inputs)) for _ in range(total)
    ]
    values = [value(u) for u in points]
    evaluations = total
    covs_sq = 0.0
    level = 0
    while True:
        failures = sum(1 for v in values if v <= 0.0)
        if failures >= seeds_count or level + 1 >= levels_limit:
            break
        order = sorted(range(total), key=values.__getitem__)
        low, high = values[order[seeds_count - 1]], values[order[seeds_count]]
        if low == high and level == 0:
            raise ValueError("limit-state values tie at an intermediate threshold")
        threshold = 0.5 * (low + high)
        covs_sq += (
            _level_cov(
                [v <= threshold for v in values],
                seeds_count,
                p0,
                level > 0,
                [chain_length] * seeds_count if level > 0 else None,
            )
            ** 2
        )
        new_points: list[Vector] = []
        new_values: list[float] = []
        for index in order[:seeds_count]:
            current, current_value = points[index], values[index]
            new_points.append(current)
            new_values.append(current_value)
            for _ in range(chain_length - 1):
                proposal = tuple(
                    rho * c + sigma_step * generator.gauss(0.0, 1.0) for c in current
                )
                proposal_value = value(proposal)
                evaluations += 1
                if proposal_value <= threshold:
                    current, current_value = proposal, proposal_value
                new_points.append(current)
                new_values.append(current_value)
        points, values = new_points, new_values
        level += 1
    flags = [v <= 0.0 for v in values]
    count = sum(flags)
    conditioning = p0**level
    width = 2 * n_inputs
    names = normal_parameter_names(n_inputs)[:width]
    if count == 0:
        return ScoreSensitivityResult(
            "subset-simulation",
            0.0,
            math.nan,
            (0.0,) * width,
            (math.nan,) * width,
            names,
            total,
            0,
            level + 1,
            evaluations,
            False,
            "no failure in the final level (max_levels reached)",
        )
    contributions = [
        normal_scores(physical(u), m, s) if flag else (0.0,) * width
        for u, flag in zip(points, flags, strict=True)
    ]
    chains = seeds_count if level > 0 else total
    size = total // chains
    pf = conditioning * count / total
    final_cov = (
        _level_cov(flags, chains, count / total, level > 0, None)
        if 0 < count < total
        else 0.0
    )
    derivatives: list[float] = []
    errors: list[float] = []
    for k in range(width):
        chain_means = [
            math.fsum(contributions[c * size + j][k] for j in range(size)) / size
            for c in range(chains)
        ]
        grand = math.fsum(chain_means) / chains
        spread = math.fsum((v - grand) ** 2 for v in chain_means) / (chains - 1)
        batch_se = math.sqrt(spread / chains)
        derivative = conditioning * grand
        derivatives.append(derivative)
        errors.append(
            math.hypot(conditioning * batch_se, math.sqrt(covs_sq) * derivative)
        )
    return ScoreSensitivityResult(
        method="subset-simulation",
        probability=pf,
        probability_standard_error=pf * math.sqrt(covs_sq + final_cov**2),
        derivatives=tuple(derivatives),
        standard_errors=tuple(errors),
        parameter_names=names,
        samples=total,
        failures=count,
        levels=level + 1,
        evaluations=evaluations,
        converged=count >= seeds_count,
        stop_reason="converged"
        if count >= seeds_count
        else "maximum levels reached before p0 N failures",
    )


@dataclass(frozen=True, slots=True)
class FailureSobolResult:
    """First-order Sobol-type indices of the failure indicator.

    Args:
        first_order: ``S_i = Var(E[1_F | X_i]) / Var(1_F)`` per input; the
            bias-corrected binned estimate (may be slightly negative from
            sampling noise, not clipped).
        standard_errors: Bootstrap standard errors (nan if no bootstrap).
        failure_probability: Sample failure fraction.
        samples: Number of samples.
        bins: Equal-count bins per input.

    Equation:
        [EQ-REL-RSENS-006].

    References:
        [REF-SOBOL-1993], [REF-SALTELLI-2008].

    """

    first_order: Vector
    standard_errors: Vector
    failure_probability: float
    samples: int
    bins: int


def _binned_indices(
    columns: Sequence[Sequence[float]], failed: Sequence[bool], bins: int
) -> Vector:
    """Return bias-corrected binned first-order indices.

    References:
        [EQ-REL-RSENS-006], [REF-SALTELLI-2008]. No equation is evaluated.

    """
    n = len(failed)
    p = sum(failed) / n
    variance = p * (1.0 - p)
    out: list[float] = []
    for column in columns:
        order = sorted(range(n), key=column.__getitem__)
        between = 0.0
        correction = 0.0
        for b in range(bins):
            lo, hi = b * n // bins, (b + 1) * n // bins
            size = hi - lo
            if size < 2:
                continue
            ybar = sum(failed[order[j]] for j in range(lo, hi)) / size
            weight = size / n
            between += weight * (ybar - p) ** 2
            correction += weight * ybar * (1.0 - ybar) / (size - 1)
        out.append((between - correction) / variance if variance > 0.0 else 0.0)
    return tuple(out)


def failure_indicator_sobol(
    samples: Sequence[Sequence[float]],
    failed: Sequence[bool],
    *,
    bins: int | None = None,
    n_bootstrap: int = 50,
    seed: int | RandomStream | None = None,
) -> FailureSobolResult:
    """Estimate first-order indices of ``1{g <= 0}`` from independent samples.

    ``S_i = Var(E[Y | X_i]) / Var(Y)`` with ``Y = 1_F``. Each input is sorted
    and split into ``B`` equal-count bins; the between-bin variance of the
    bin means is corrected by the within-bin sampling variance
    ``ybar_b (1 - ybar_b) / (n_b - 1)``, which removes the leading ``O(B / N)``
    bias of the plug-in estimate. Binning smooths ``E[Y | X_i]`` and therefore
    biases ``S_i`` slightly downward when the conditional mean varies strongly
    inside a bin. Samples must be i.i.d. draws of the input law (crude Monte
    Carlo); subset-simulation samples are dependent and not valid here.

    Args:
        samples: Input vectors ``x_j`` (rows), i.i.d. from the input law.
        failed: Failure indicators ``1{g(x_j) <= 0}``.
        bins: Bins per input; default ``max(2, min(20, F / 10))`` with ``F``
            the number of failures.
        n_bootstrap: Bootstrap replicates for the standard errors (0 skips).
        seed: Integer seed, stream, or ``None`` (bootstrap resampling).

    Returns:
        Indices, bootstrap standard errors, and sample statistics.

    Raises:
        TypeError: If integer arguments have the wrong type.
        ValueError: If shapes are inconsistent, fewer than 20 samples are
            given, or no failure and no success occurred.

    Equation:
        [EQ-REL-RSENS-006].

    References:
        [REF-SOBOL-1993], [REF-SALTELLI-2008].

    """
    n = len(samples)
    if n < 20 or len(failed) != n:
        raise ValueError("need at least 20 samples and one indicator per sample")
    width = len(samples[0])
    if width < 1 or any(len(row) != width for row in samples):
        raise ValueError("samples must be rows of equal positive length")
    flags = [bool(f) for f in failed]
    count = sum(flags)
    if count == 0 or count == n:
        raise ValueError("indicator is constant; indices are undefined")
    boot = _int_at_least(n_bootstrap, "n_bootstrap", 0)
    if bins is None:
        b = max(2, min(20, count // 10))
    else:
        b = _int_at_least(bins, "bins", 2)
    if n < 2 * b:
        raise ValueError("need at least two samples per bin")
    columns = [[row[i] for row in samples] for i in range(width)]
    estimate = _binned_indices(columns, flags, b)
    errors: tuple[float, ...] = (math.nan,) * width
    if boot > 1:
        generator = resolve_stream(seed).child("failure-sobol").generator()
        replicates: list[Vector] = []
        for _ in range(boot):
            pick = [generator.randrange(n) for _ in range(n)]
            resampled_flags = [flags[j] for j in pick]
            if all(resampled_flags) or not any(resampled_flags):
                continue
            replicates.append(
                _binned_indices(
                    [[col[j] for j in pick] for col in columns], resampled_flags, b
                )
            )
        if len(replicates) > 1:
            errors = tuple(
                math.sqrt(
                    math.fsum(
                        (r[i] - sum(q[i] for q in replicates) / len(replicates)) ** 2
                        for r in replicates
                    )
                    / (len(replicates) - 1)
                )
                for i in range(width)
            )
    return FailureSobolResult(
        first_order=estimate,
        standard_errors=errors,
        failure_probability=count / n,
        samples=n,
        bins=b,
    )
