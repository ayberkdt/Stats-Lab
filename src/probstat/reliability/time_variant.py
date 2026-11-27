"""Time-variant reliability: outcrossing rates, PHI2, and first passage.

Experimental (added in 0.28).

A time-variant limit state ``G(u, t)`` in independent standard-normal space
fails at time ``t`` when ``G(u, t) <= 0``. The *first-passage* probability
over ``[0, T]`` is the probability that failure occurs at least once,
``P_f(0, T) = P(exists t in [0, T]: G(U, t) <= 0)``. This module provides

* the **Rice formula** for the mean upcrossing rate of a Gaussian process
  over a (moving) barrier, in closed form, as an oracle
  (:func:`rice_upcrossing_rate`);
* the **PHI2 method** of Andrieu-Renaud, Sudret and Lemaire: two FORM
  analyses at ``t`` and ``t + dt`` and a bivariate normal probability give
  the outcrossing rate ``nu(t)``; the integral bounds the first-passage
  probability from above (:func:`phi2_time_variant`,
  :func:`phi2_outcrossing_rate`);
* trajectory-space estimators of the first-passage probability by crude Monte
  Carlo and by subset simulation of the library
  (:func:`first_passage_monte_carlo`,
  :func:`first_passage_subset_simulation`) on stationary Gaussian processes
  with an exact discretisation (:class:`Matern32Process`,
  :class:`HarmonicProcess`);
* a side-by-side study of the bounds and estimates over a threshold sweep
  (:func:`first_passage_comparison`).

Assumptions and units: time and length scales share one (arbitrary) time
unit, thresholds share the unit of the process. Gaussian processes are
sampled on the grid ``t_k = k h``, ``h = T / steps``. A grid-based failure
indicator misses crossings between grid points, so grid estimates lie below
the continuous-time first-passage probability by an amount that vanishes as
``h -> 0``; the Rice and PHI2 bounds refer to continuous time (the grid
union bound uses ``dt = h``).

Limitations: the upper bound ``P_f(0) + integral nu dt`` can exceed one and is
loose when crossings cluster (low thresholds, long horizons); the Poisson
approximation assumes independent crossings and fails for the same reasons.
FORM-based PHI2 is exact only for limit states that are linear in ``u`` for
each ``t``.
"""

from __future__ import annotations

import functools
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from probstat.distributions import Normal
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector, as_vector, dot, norm
from probstat.modeling import (
    JointInput,
    ProbabilisticModel,
    RandomInput,
    standard_normal_cdf,
)
from probstat.optimization import central_gradient
from probstat.uq import QuantityOfInterest

from .model_based import ModelLimitState, _hlrf_iterate
from .simulation import subset_simulation
from .systems import bivariate_normal_cdf

TimeVariantLimitState = Callable[[Vector, float], float]

_INV_SQRT_TWO_PI = 1.0 / math.sqrt(2.0 * math.pi)
_MIN_MATERN_STEP = 1e-3


def _pdf(x: float) -> float:
    """Return the standard normal density.

    References:
        [EQ-REL-TV-001], [REF-DITLEVSEN-MADSEN-1996].

    """
    return _INV_SQRT_TWO_PI * math.exp(-0.5 * x * x)


def _positive(value: object, name: str) -> float:
    """Return a finite strictly positive float.

    References:
        [EQ-REL-TV-001], [REF-DITLEVSEN-MADSEN-1996]. No equation is evaluated.

    """
    number = require_finite_real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _positive_int(value: object, name: str, minimum: int = 1) -> int:
    """Return an integer at least ``minimum``.

    References:
        [EQ-REL-TV-001], [REF-DITLEVSEN-MADSEN-1996]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@functools.lru_cache(maxsize=16)
def _harmonic_table(
    amplitudes: tuple[float, ...],
    frequencies: tuple[float, ...],
    step: float,
    steps: int,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Return ``(s_k cos(w_k t_j), s_k sin(w_k t_j))`` for every grid time.

    References:
        [EQ-REL-TV-006], [REF-RICE-1945].

    """
    return tuple(
        tuple(
            (a * math.cos(w * j * step), a * math.sin(w * j * step))
            for a, w in zip(amplitudes, frequencies, strict=True)
        )
        for j in range(steps + 1)
    )


def rice_upcrossing_rate(
    *,
    sigma: float,
    sigma_rate: float,
    threshold: float,
    threshold_rate: float = 0.0,
    mean: float = 0.0,
) -> float:
    """Return the Rice mean upcrossing rate of a Gaussian process over a barrier.

    For a differentiable stationary Gaussian process ``X`` with mean ``m``,
    standard deviation ``sigma``, and derivative standard deviation
    ``sigma_rate`` (``X(t)`` and ``X'(t)`` are independent at equal time), the
    mean number of upcrossings per unit time of the barrier ``b(t)`` is

    ``nu = (sigma_rate / sigma) phi((b - m) / sigma) psi(-b' / sigma_rate)``

    with ``psi(z) = phi(z) + z Phi(z)``. For a constant barrier ``psi(0) =
    phi(0)`` and ``nu = sigma_rate exp(-(b - m)^2 / (2 sigma^2)) /
    (2 pi sigma)``, Rice's classical result.

    Args:
        sigma: Process standard deviation (positive).
        sigma_rate: Standard deviation of ``X'`` (positive); for a spectral
            density ``S`` it is the square root of the second spectral moment.
        threshold: Barrier level ``b`` (finite).
        threshold_rate: Barrier slope ``b'`` (default zero).
        mean: Process mean ``m``.

    Returns:
        Mean upcrossing rate in ``1 / time``.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If ``sigma`` or ``sigma_rate`` is not positive or any
            argument is not finite.

    Equation:
        [EQ-REL-TV-001].

    References:
        [REF-RICE-1944], [REF-RICE-1945], [REF-CRAMER-LEADBETTER-1967].

    """
    s = _positive(sigma, "sigma")
    s_dot = _positive(sigma_rate, "sigma_rate")
    b = require_finite_real(threshold, name="threshold")
    b_dot = require_finite_real(threshold_rate, name="threshold_rate")
    m = require_finite_real(mean, name="mean")
    z = (b - m) / s
    q = -b_dot / s_dot
    psi = _pdf(q) + q * standard_normal_cdf(q)
    return (s_dot / s) * _pdf(z) * psi


def phi2_outcrossing_probability(
    beta_now: float, beta_next: float, correlation: float
) -> float:
    """Return the PHI2 probability of a safe-to-failed transition in one step.

    With linearised failure events ``E_1 = {alpha_1 . U >= beta_1}`` at ``t``
    and ``E_2 = {alpha_2 . U >= beta_2}`` at ``t + dt`` and correlation
    ``rho = alpha_1 . alpha_2``,

    ``P(not E_1 and E_2) = Phi(-beta_2) - Phi_2(-beta_1, -beta_2; rho)``.

    Args:
        beta_now: Reliability index at ``t``.
        beta_next: Reliability index at ``t + dt``.
        correlation: ``rho`` in ``[-1, 1]``.

    Returns:
        Outcrossing probability, clipped at zero against rounding.

    Raises:
        ValueError: If an argument is not finite or ``rho`` is outside
            ``[-1, 1]``.

    Equation:
        [EQ-REL-TV-003].

    References:
        [REF-ANDRIEU-RENAUD-SUDRET-LEMAIRE-2004],
        [REF-DITLEVSEN-MADSEN-1996].

    """
    b1 = require_finite_real(beta_now, name="beta_now")
    b2 = require_finite_real(beta_next, name="beta_next")
    rho = require_finite_real(correlation, name="correlation")
    both = bivariate_normal_cdf(-b1, -b2, rho)
    return max(0.0, standard_normal_cdf(-b2) - both)


def phi2_outcrossing_rate(
    beta_now: float, beta_next: float, correlation: float, dt: float
) -> float:
    """Return the PHI2 outcrossing rate ``P(not E_1 and E_2) / dt``.

    The rate converges to the exact outcrossing rate of a linear Gaussian
    problem as ``dt -> 0`` with relative error ``O(dt)``; very small ``dt``
    loses digits because the probability is a difference of two numbers of
    size ``Phi(-beta)``.

    Args:
        beta_now: Reliability index at ``t``.
        beta_next: Reliability index at ``t + dt``.
        correlation: ``rho = alpha(t) . alpha(t + dt)``.
        dt: Positive time increment.

    Returns:
        Outcrossing rate in ``1 / time``.

    Raises:
        ValueError: If ``dt`` is not positive or other arguments are invalid.

    Equation:
        [EQ-REL-TV-002], [EQ-REL-TV-003].

    References:
        [REF-ANDRIEU-RENAUD-SUDRET-LEMAIRE-2004].

    """
    step = _positive(dt, "dt")
    return phi2_outcrossing_probability(beta_now, beta_next, correlation) / step


@dataclass(frozen=True, slots=True)
class PHI2Result:
    """PHI2 time-variant reliability over ``[0, T]``.

    Args:
        times: Grid times ``t_0 = 0 < ... < t_n = T``.
        betas: Signed FORM index ``beta(t_k)``.
        rates: PHI2 outcrossing rate ``nu(t_k)``.
        correlations: ``alpha(t_k) . alpha(t_k + dt)``.
        initial_probability: ``Phi(-beta(0))``.
        expected_outcrossings: Trapezoidal ``integral nu dt``.
        upper_bound: ``P_f(0) + integral nu dt``; may exceed one.
        poisson_probability: ``1 - (1 - P_f(0)) exp(-integral nu dt)``.
        dt: Increment used in the finite-difference outcrossing rate.
        evaluations: Limit-state evaluations of all FORM searches.
        converged: Whether every FORM search met its tolerance.
        stop_reason: ``"converged"`` or the first failing search.

    Equation:
        [EQ-REL-TV-002], [EQ-REL-TV-004].

    References:
        [REF-ANDRIEU-RENAUD-SUDRET-LEMAIRE-2004].

    """

    times: Vector
    betas: Vector
    rates: Vector
    correlations: Vector
    initial_probability: float
    expected_outcrossings: float
    upper_bound: float
    poisson_probability: float
    dt: float
    evaluations: int
    converged: bool
    stop_reason: str


def phi2_time_variant(
    limit_state: TimeVariantLimitState,
    dimension: int,
    horizon: float,
    *,
    steps: int = 20,
    dt: float | None = None,
    start: Sequence[float] | None = None,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> PHI2Result:
    """Bound the first-passage probability by the PHI2 method.

    At each grid time ``t_k`` the FORM design points at ``t_k`` and
    ``t_k + dt`` give ``beta_1``, ``beta_2``, and ``rho = alpha_1 . alpha_2``,
    so ``nu(t_k) = [Phi(-beta_2) - Phi_2(-beta_1, -beta_2; rho)] / dt``. The
    trapezoid rule integrates ``nu``; ``P_f(0, T) <= P_f(0) + integral nu dt``
    (Rice's union bound). FORM searches are warm-started from the preceding
    design point.

    Args:
        limit_state: ``G(u, t)`` in standard-normal space with failure
            ``G <= 0``.
        dimension: Number of standard-normal variables.
        horizon: Positive horizon ``T``.
        steps: Number of grid intervals (at least one).
        dt: Outcrossing increment; default ``1e-3 horizon / steps``.
        start: Optional first FORM starting point (default the origin).
        max_iterations: Maximum HL-RF iterations per FORM search.
        tolerance: Relative HL-RF tolerance.

    Returns:
        Indices, rates, bound, Poisson estimate, and convergence status.

    Raises:
        TypeError: If arguments have the wrong type.
        ValueError: If arguments are invalid or a gradient vanishes.

    Equation:
        [EQ-REL-TV-002], [EQ-REL-TV-003], [EQ-REL-TV-004].

    References:
        [REF-ANDRIEU-RENAUD-SUDRET-LEMAIRE-2004], [REF-RICE-1944],
        [REF-HASOFER-LIND-1974].

    """
    if not callable(limit_state):
        raise TypeError("limit_state must be callable")
    n = _positive_int(dimension, "dimension")
    horizon_value = _positive(horizon, "horizon")
    count = _positive_int(steps, "steps")
    increment = 1e-3 * horizon_value / count if dt is None else _positive(dt, "dt")
    limit = _positive_int(max_iterations, "max_iterations")
    tol = _positive(tolerance, "tolerance")
    u0: Vector = (0.0,) * n if start is None else as_vector(start, name="start")
    if len(u0) != n:
        raise ValueError("start length must equal dimension")
    evaluations = 0
    failures: list[str] = []

    def form_at(t: float, guess: Vector) -> tuple[Vector, float, Vector]:
        """Run FORM at time ``t`` and return ``(u*, beta, alpha)``.

        References:
            [REF-ANDRIEU-RENAUD-SUDRET-LEMAIRE-2004]. No equation is evaluated.

        """

        def function(u: Vector) -> float:
            """Evaluate ``G(u, t)`` and count the call.

            References:
                [EQ-REL-TV-002]. No equation is evaluated.

            """
            nonlocal evaluations
            evaluations += 1
            return float(limit_state(u, t))

        def gradient(u: Vector) -> Vector:
            """Return a central-difference gradient of ``G(., t)``.

            References:
                [EQ-REL-TV-002]. No equation is evaluated.

            """
            nonlocal evaluations
            grad, calls = central_gradient(function, u, relative_step=1e-6)
            evaluations += calls
            return grad

        u, _, grad, _, ok, message = _hlrf_iterate(
            function, guess, gradient, True, limit, tol
        )
        if not ok:
            failures.append(f"FORM at t={t:g}: {message}")
        length = norm(grad)
        alpha = tuple(-component / length for component in grad)
        return u, dot(alpha, u), alpha

    times = tuple(horizon_value * k / count for k in range(count + 1))
    betas: list[float] = []
    rates: list[float] = []
    rhos: list[float] = []
    guess = u0
    for t in times:
        u1, beta1, alpha1 = form_at(t, guess)
        u2, beta2, alpha2 = form_at(t + increment, u1)
        rho = max(-1.0, min(1.0, dot(alpha1, alpha2)))
        betas.append(beta1)
        rhos.append(rho)
        rates.append(phi2_outcrossing_rate(beta1, beta2, rho, increment))
        guess = u1
    integral = math.fsum(
        0.5 * (rates[k] + rates[k + 1]) * (times[k + 1] - times[k])
        for k in range(count)
    )
    initial = standard_normal_cdf(-betas[0])
    return PHI2Result(
        times=times,
        betas=tuple(betas),
        rates=tuple(rates),
        correlations=tuple(rhos),
        initial_probability=initial,
        expected_outcrossings=integral,
        upper_bound=initial + integral,
        poisson_probability=1.0 - (1.0 - initial) * math.exp(-integral),
        dt=increment,
        evaluations=evaluations,
        converged=not failures,
        stop_reason="converged" if not failures else failures[0],
    )


@runtime_checkable
class GaussianProcessModel(Protocol):
    """A stationary Gaussian process with an exact grid representation.

    Implementations expose the moments needed by the Rice formula and map a
    vector of independent standard-normal variables to the process values on
    the grid ``t_k = k h``.

    References:
        [REF-RICE-1944], [REF-CRAMER-LEADBETTER-1967].

    """

    mean: float

    @property
    def sigma(self) -> float:
        """Return the marginal standard deviation.

        References:
            [EQ-REL-TV-001].

        """
        ...

    @property
    def sigma_rate(self) -> float:
        """Return the standard deviation of the derivative process.

        References:
            [EQ-REL-TV-001].

        """
        ...

    def correlation(self, lag: float) -> float:
        """Return the autocorrelation at a time lag.

        Args:
            lag: Time lag.

        References:
            [EQ-REL-TV-005].

        """
        ...

    def noise_dimension(self, steps: int) -> int:
        """Return the number of standard-normal inputs for ``steps`` steps.

        Args:
            steps: Number of grid intervals.

        References:
            [EQ-REL-TV-007].

        """
        ...

    def path(self, z: Sequence[float], step: float, steps: int) -> tuple[float, ...]:
        """Return the process at ``t_k = k step`` for ``k = 0..steps``.

        Args:
            z: Independent standard-normal inputs.
            step: Grid spacing.
            steps: Number of grid intervals.

        References:
            [EQ-REL-TV-005].

        """
        ...


@dataclass(frozen=True, slots=True)
class Matern32Process:
    """Stationary Matern-3/2 Gaussian process with exact grid sampling.

    Autocorrelation ``rho(tau) = (1 + lam |tau|) exp(-lam |tau|)`` with
    ``lam = sqrt(3) / length_scale``. The process is once mean-square
    differentiable with ``sigma_rate = lam sigma``, so the Rice rate is
    finite. The pair ``(X, X')`` is a Markov process with transition
    ``e^{-lam h} [[1 + lam h, h], [-lam^2 h, 1 - lam h]]``, so a path costs
    ``O(steps)`` and has exactly the target law on the grid.

    Args:
        sigma_value: Marginal standard deviation (positive).
        length_scale: Length scale ``ell`` in time units (positive).
        mean: Constant mean.

    Raises:
        ValueError: If the standard deviation or length scale is not positive.

    Equation:
        [EQ-REL-TV-005].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006], [REF-RICE-1945].

    """

    sigma_value: float
    length_scale: float
    mean: float = 0.0

    def __post_init__(self) -> None:
        """Validate parameters.

        References:
            [EQ-REL-TV-005], [REF-RASMUSSEN-WILLIAMS-2006].

        """
        _positive(self.sigma_value, "sigma_value")
        _positive(self.length_scale, "length_scale")
        require_finite_real(self.mean, name="mean")

    @property
    def sigma(self) -> float:
        """Return the marginal standard deviation.

        References:
            [EQ-REL-TV-005].

        """
        return self.sigma_value

    @property
    def sigma_rate(self) -> float:
        """Return ``sqrt(3) sigma / ell``, the derivative standard deviation.

        References:
            [EQ-REL-TV-005].

        """
        return math.sqrt(3.0) / self.length_scale * self.sigma_value

    def correlation(self, lag: float) -> float:
        """Return ``(1 + lam |tau|) exp(-lam |tau|)``.

        Args:
            lag: Time lag.

        Returns:
            Autocorrelation in ``(0, 1]``.

        References:
            [EQ-REL-TV-005].

        """
        x = math.sqrt(3.0) * abs(lag) / self.length_scale
        return (1.0 + x) * math.exp(-x)

    def noise_dimension(self, steps: int) -> int:
        """Return ``2 (steps + 1)``: initial state and one innovation per step.

        Args:
            steps: Number of grid intervals.

        Returns:
            Dimension of the standard-normal input.

        References:
            [EQ-REL-TV-005].

        """
        return 2 * (_positive_int(steps, "steps") + 1)

    def path(self, z: Sequence[float], step: float, steps: int) -> tuple[float, ...]:
        """Return the process values on the grid from standard-normal inputs.

        Args:
            z: ``noise_dimension(steps)`` independent standard normals.
            step: Grid spacing ``h`` (positive).
            steps: Number of grid intervals.

        Returns:
            Values at ``t_k = k h``, ``k = 0..steps``.

        Raises:
            ValueError: If ``z`` has the wrong length or the innovation
                covariance is numerically singular (``lam h`` too small).

        References:
            [EQ-REL-TV-005], [REF-RASMUSSEN-WILLIAMS-2006].

        """
        h = _positive(step, "step")
        if len(z) != self.noise_dimension(steps):
            raise ValueError("z has the wrong length")
        lam = math.sqrt(3.0) / self.length_scale
        x_scale = self.sigma_value
        decay = math.exp(-lam * h)
        a11, a12 = decay * (1.0 + lam * h), decay * h
        a21, a22 = -decay * lam * lam * h, decay * (1.0 - lam * h)
        p11, p22 = x_scale * x_scale, lam * lam * x_scale * x_scale
        q11 = p11 - (a11 * a11 * p11 + a12 * a12 * p22)
        q12 = -(a11 * a21 * p11 + a12 * a22 * p22)
        q22 = p22 - (a21 * a21 * p11 + a22 * a22 * p22)
        if lam * h < _MIN_MATERN_STEP:
            raise ValueError(
                "step too small for the closed-form innovation covariance "
                f"(need sqrt(3) step / length_scale >= {_MIN_MATERN_STEP:g})"
            )
        if q11 <= 0.0:
            raise ValueError("innovation covariance is singular; step too small")
        l11 = math.sqrt(q11)
        l21 = q12 / l11
        rest = q22 - l21 * l21
        if rest <= 0.0:
            raise ValueError("innovation covariance is singular; step too small")
        l22 = math.sqrt(rest)
        x = x_scale * z[0]
        v = lam * x_scale * z[1]
        out = [self.mean + x]
        for k in range(1, steps + 1):
            e1, e2 = z[2 * k], z[2 * k + 1]
            x, v = (
                a11 * x + a12 * v + l11 * e1,
                a21 * x + a22 * v + l21 * e1 + l22 * e2,
            )
            out.append(self.mean + x)
        return tuple(out)


@dataclass(frozen=True, slots=True)
class HarmonicProcess:
    """Stationary Gaussian process as a finite sum of random harmonics.

    ``X(t) = m + sum_k s_k (z_{2k} cos w_k t + z_{2k+1} sin w_k t)`` with
    independent standard normals ``z``. It is stationary with variance
    ``sum s_k^2``, autocorrelation ``sum s_k^2 cos(w_k tau) / sum s_k^2``, and
    derivative variance ``sum s_k^2 w_k^2``. Because it is exact in
    continuous time and linear in ``z``, it is the closed-form test bed for
    PHI2 (FORM is exact) and the Rice formula.

    Args:
        amplitudes: Standard deviations ``s_k`` (positive).
        frequencies: Angular frequencies ``w_k`` in rad per time unit
            (positive, same length).
        mean: Constant mean.

    Raises:
        ValueError: If lists are empty, differ in length, or are not positive.

    Equation:
        [EQ-REL-TV-006].

    References:
        [REF-RICE-1945], [REF-CRAMER-LEADBETTER-1967].

    """

    amplitudes: tuple[float, ...]
    frequencies: tuple[float, ...]
    mean: float = 0.0

    def __post_init__(self) -> None:
        """Validate amplitudes and frequencies.

        References:
            [EQ-REL-TV-006], [REF-RICE-1945].

        """
        if not self.amplitudes or len(self.amplitudes) != len(self.frequencies):
            raise ValueError("amplitudes and frequencies must be equal, non-empty")
        for a in self.amplitudes:
            _positive(a, "amplitude")
        for w in self.frequencies:
            _positive(w, "frequency")
        require_finite_real(self.mean, name="mean")

    @property
    def sigma(self) -> float:
        """Return ``sqrt(sum s_k^2)``.

        References:
            [EQ-REL-TV-006].

        """
        return math.sqrt(math.fsum(a * a for a in self.amplitudes))

    @property
    def sigma_rate(self) -> float:
        """Return ``sqrt(sum s_k^2 w_k^2)``.

        References:
            [EQ-REL-TV-006].

        """
        return math.sqrt(
            math.fsum(
                (a * w) ** 2
                for a, w in zip(self.amplitudes, self.frequencies, strict=True)
            )
        )

    def correlation(self, lag: float) -> float:
        """Return ``sum s_k^2 cos(w_k tau) / sum s_k^2``.

        Args:
            lag: Time lag.

        Returns:
            Autocorrelation in ``[-1, 1]``.

        References:
            [EQ-REL-TV-006].

        """
        total = math.fsum(a * a for a in self.amplitudes)
        return (
            math.fsum(
                a * a * math.cos(w * lag)
                for a, w in zip(self.amplitudes, self.frequencies, strict=True)
            )
            / total
        )

    def noise_dimension(self, steps: int) -> int:
        """Return ``2 K`` (independent of ``steps``).

        Args:
            steps: Number of grid intervals (validated only).

        Returns:
            Number of standard-normal inputs.

        References:
            [EQ-REL-TV-006].

        """
        _positive_int(steps, "steps")
        return 2 * len(self.amplitudes)

    def value(self, z: Sequence[float], t: float) -> float:
        """Return ``X(t)`` for standard-normal inputs ``z``.

        Args:
            z: ``2 K`` independent standard normals.
            t: Time.

        Returns:
            Process value.

        Raises:
            ValueError: If ``z`` has the wrong length.

        References:
            [EQ-REL-TV-006].

        """
        if len(z) != 2 * len(self.amplitudes):
            raise ValueError("z has the wrong length")
        return self.mean + math.fsum(
            a * (z[2 * k] * math.cos(w * t) + z[2 * k + 1] * math.sin(w * t))
            for k, (a, w) in enumerate(
                zip(self.amplitudes, self.frequencies, strict=True)
            )
        )

    def path(self, z: Sequence[float], step: float, steps: int) -> tuple[float, ...]:
        """Return the process values on the grid.

        Args:
            z: ``2 K`` independent standard normals.
            step: Grid spacing.
            steps: Number of grid intervals.

        Returns:
            Values at ``t_k = k step``.

        References:
            [EQ-REL-TV-006].

        """
        h = _positive(step, "step")
        count = _positive_int(steps, "steps")
        if len(z) != 2 * len(self.amplitudes):
            raise ValueError("z has the wrong length")
        table = _harmonic_table(self.amplitudes, self.frequencies, h, count)
        return tuple(
            self.mean
            + math.fsum(z[2 * k] * c + z[2 * k + 1] * s for k, (c, s) in enumerate(row))
            for row in table
        )

    def limit_state(self, threshold: float) -> TimeVariantLimitState:
        """Return ``G(u, t) = b - X(t)`` for PHI2 (failure when ``X >= b``).

        Args:
            threshold: Constant barrier ``b``.

        Returns:
            Callable ``G(u, t)`` in standard-normal space.

        References:
            [EQ-REL-TV-002], [EQ-REL-TV-006].

        """
        b = require_finite_real(threshold, name="threshold")

        def g(u: Vector, t: float) -> float:
            """Evaluate ``b - X(t)``.

            References:
                [EQ-REL-TV-006]. No equation is evaluated.

            """
            return b - self.value(u, t)

        return g


@dataclass(frozen=True, slots=True)
class FirstPassageResult:
    """Estimate of the grid first-passage probability of a process.

    Args:
        method: ``"monte-carlo"`` or ``"subset-simulation"``.
        probability: Estimated ``P(max_k (X_k - b_k) >= 0)``.
        standard_error: Standard error (nan when no failure was observed).
        coefficient_of_variation: ``standard_error / probability`` (nan when
            the probability is zero).
        samples: Trajectories (Monte Carlo) or samples per level.
        evaluations: Trajectory evaluations.
        steps: Grid intervals.
        step: Grid spacing ``h``.
        converged: ``False`` when the estimate is not trustworthy (no
            failure, or subset simulation reached ``max_levels``).
        stop_reason: Human-readable status.

    Equation:
        [EQ-REL-TV-007].

    References:
        [REF-AU-BECK-2001], [REF-MELCHERS-BECK-2018].

    """

    method: str
    probability: float
    standard_error: float
    coefficient_of_variation: float
    samples: int
    evaluations: int
    steps: int
    step: float
    converged: bool
    stop_reason: str


def _barrier(threshold: float | Sequence[float], steps: int) -> tuple[float, ...]:
    """Return the barrier on ``steps + 1`` grid points.

    References:
        [EQ-REL-TV-007], [REF-MELCHERS-BECK-2018]. No equation is evaluated.

    """
    if isinstance(threshold, bool):
        raise TypeError("threshold must be a number or a sequence")
    if isinstance(threshold, (int, float)):
        return (require_finite_real(threshold, name="threshold"),) * (steps + 1)
    values = tuple(float(v) for v in threshold)
    if len(values) != steps + 1 or not all(math.isfinite(v) for v in values):
        raise ValueError("threshold sequence must be finite with steps + 1 entries")
    return values


def _require_process(process: object) -> GaussianProcessModel:
    """Return ``process`` if it satisfies the process protocol.

    References:
        [EQ-REL-TV-007], [REF-RICE-1944]. No equation is evaluated.

    """
    if not isinstance(process, GaussianProcessModel):
        raise TypeError("process must implement GaussianProcessModel")
    return process


def first_passage_monte_carlo(
    process: GaussianProcessModel,
    threshold: float | Sequence[float],
    *,
    horizon: float,
    steps: int,
    samples: int,
    seed: int | RandomStream | None = None,
) -> FirstPassageResult:
    """Estimate the grid first-passage probability by crude Monte Carlo.

    Draws ``samples`` independent trajectories on ``t_k = k T / steps`` and
    counts those with ``X_k >= b_k`` for some ``k`` (including ``k = 0``).
    The estimator is binomial, ``se = sqrt(p (1 - p) / N)``.

    Args:
        process: Gaussian process with exact grid sampling.
        threshold: Constant barrier or one value per grid point.
        horizon: Horizon ``T`` (positive).
        steps: Number of grid intervals.
        samples: Number of trajectories (at least two).
        seed: Integer seed, stream, or ``None``.

    Returns:
        First-passage estimate; ``converged`` is ``False`` when no trajectory
        failed.

    Raises:
        TypeError: If ``process`` or an integer argument has the wrong type.
        ValueError: If arguments are invalid.

    Equation:
        [EQ-REL-TV-007].

    References:
        [REF-MELCHERS-BECK-2018], [REF-RUBINSTEIN-KROESE-2004].

    """
    model = _require_process(process)
    count = _positive_int(steps, "steps")
    n = _positive_int(samples, "samples", minimum=2)
    h = _positive(horizon, "horizon") / count
    barrier = _barrier(threshold, count)
    generator: random.Random = (
        resolve_stream(seed).child("first-passage-mc").generator()
    )
    dim = model.noise_dimension(count)
    failures = 0
    for _ in range(n):
        z = [generator.gauss(0.0, 1.0) for _ in range(dim)]
        trajectory = model.path(z, h, count)
        if any(x >= b for x, b in zip(trajectory, barrier, strict=True)):
            failures += 1
    p = failures / n
    if failures == 0:
        return FirstPassageResult(
            "monte-carlo",
            0.0,
            math.nan,
            math.nan,
            n,
            n,
            count,
            h,
            False,
            "no failure observed; probability below about 3/N",
        )
    se = math.sqrt(p * (1.0 - p) / n)
    return FirstPassageResult(
        "monte-carlo", p, se, se / p, n, n, count, h, True, "converged"
    )


def first_passage_subset_simulation(
    process: GaussianProcessModel,
    threshold: float | Sequence[float],
    *,
    horizon: float,
    steps: int,
    samples_per_level: int = 1000,
    level_probability: float = 0.1,
    sampler: str = "acs",
    max_levels: int = 30,
    seed: int | RandomStream | None = None,
) -> FirstPassageResult:
    """Estimate the grid first-passage probability by subset simulation.

    The standard-normal innovations of the process are the random variables
    (trajectory space). With ``y(z) = max_k (X_k(z) - b_k)`` the failure event
    is ``y >= 0``; the library's :func:`subset_simulation` is run on this
    limit state, so the estimate and its c.o.v. follow Au and Beck.

    Args:
        process: Gaussian process with exact grid sampling.
        threshold: Constant barrier or one value per grid point.
        horizon: Horizon ``T`` (positive).
        steps: Number of grid intervals.
        samples_per_level: Samples per level.
        level_probability: Conditional level probability ``p0``.
        sampler: Subset-simulation kernel (``"cs"``, ``"acs"``, ``"hmc"`` or
            ``"mma"``).
        max_levels: Maximum number of levels.
        seed: Integer seed, stream, or ``None``.

    Returns:
        First-passage estimate with the subset-simulation c.o.v.

    Raises:
        TypeError: If ``process`` or an integer argument has the wrong type.
        ValueError: If arguments are invalid.

    Equation:
        [EQ-REL-TV-007].

    References:
        [REF-AU-BECK-2001], [REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015].

    """
    model = _require_process(process)
    count = _positive_int(steps, "steps")
    h = _positive(horizon, "horizon") / count
    barrier = _barrier(threshold, count)
    dim = model.noise_dimension(count)
    names = [f"z{k}" for k in range(dim)]
    inputs = JointInput.independent(*(RandomInput(n, Normal(0.0, 1.0)) for n in names))

    def forward(parameters: Mapping[str, float]) -> Mapping[str, float]:
        """Return the maximal barrier exceedance of the trajectory.

        References:
            [EQ-REL-TV-007]. No equation is evaluated.

        """
        trajectory = model.path([parameters[n] for n in names], h, count)
        return {"y": max(x - b for x, b in zip(trajectory, barrier, strict=True))}

    pm = ProbabilisticModel(
        inputs, forward, (QuantityOfInterest("y", "maximal barrier exceedance"),)
    )
    result = subset_simulation(
        ModelLimitState(pm, "y", 0.0, "above"),
        samples_per_level=samples_per_level,
        level_probability=level_probability,
        sampler=sampler,
        max_levels=max_levels,
        seed=seed,
    )
    p = result.probability
    cov = result.coefficient_of_variation
    return FirstPassageResult(
        "subset-simulation",
        p,
        p * cov,
        cov,
        result.samples_per_level,
        result.evaluations,
        count,
        h,
        result.converged,
        "converged" if result.converged else "maximum levels reached",
    )


@dataclass(frozen=True, slots=True)
class FirstPassageRow:
    """One threshold of the first-passage comparison.

    Args:
        threshold: Barrier ``b``.
        standardized_threshold: ``(b - m) / sigma``.
        rice_rate: Continuous Rice rate ``nu``.
        rice_bound: ``P_f(0) + nu T`` (continuous-time bound).
        rice_poisson: ``1 - (1 - P_f(0)) exp(-nu T)``.
        phi2_bound: ``P_f(0) + nu_PHI2 T`` with ``dt = phi2_dt``.
        grid_union_bound: ``P_f(0) + n [Phi(-beta) - Phi_2(-beta, -beta;
            rho(h))]``, an exact upper bound for the grid event.
        monte_carlo: Crude Monte Carlo grid estimate.
        subset: Subset-simulation grid estimate.

    Equation:
        [EQ-REL-TV-004], [EQ-REL-TV-007].

    References:
        [REF-ANDRIEU-RENAUD-SUDRET-LEMAIRE-2004], [REF-RICE-1944].

    """

    threshold: float
    standardized_threshold: float
    rice_rate: float
    rice_bound: float
    rice_poisson: float
    phi2_bound: float
    grid_union_bound: float
    monte_carlo: FirstPassageResult
    subset: FirstPassageResult


def first_passage_comparison(
    process: GaussianProcessModel,
    thresholds: Sequence[float],
    *,
    horizon: float,
    steps: int,
    mc_samples: int = 2000,
    subset_samples: int = 500,
    phi2_dt: float | None = None,
    seed: int | RandomStream | None = None,
) -> tuple[FirstPassageRow, ...]:
    """Compare Rice, PHI2, Poisson, Monte Carlo, and subset simulation.

    For a stationary Gaussian process with a constant barrier all FORM
    problems are linear (``beta = (b - m) / sigma`` for every ``t``,
    ``rho = rho(dt)``), so PHI2 is exact up to the finite ``dt`` and
    needs no search; it is evaluated directly from the process correlation.

    Args:
        process: Gaussian process with exact grid sampling.
        thresholds: Constant barriers ``b`` to sweep.
        horizon: Horizon ``T``.
        steps: Grid intervals of the trajectory estimators.
        mc_samples: Trajectories of crude Monte Carlo.
        subset_samples: Samples per level of subset simulation.
        phi2_dt: Outcrossing increment of PHI2 (default ``h / 10``).
        seed: Integer seed, stream, or ``None``; every threshold and method
            uses its own child stream.

    Returns:
        One :class:`FirstPassageRow` per threshold.

    Raises:
        TypeError: If ``process`` has the wrong type.
        ValueError: If arguments are invalid.

    Equation:
        [EQ-REL-TV-002], [EQ-REL-TV-004], [EQ-REL-TV-007].

    References:
        [REF-ANDRIEU-RENAUD-SUDRET-LEMAIRE-2004], [REF-RICE-1944],
        [REF-AU-BECK-2001].

    """
    model = _require_process(process)
    count = _positive_int(steps, "steps")
    big_t = _positive(horizon, "horizon")
    h = big_t / count
    dt = h / 10.0 if phi2_dt is None else _positive(phi2_dt, "phi2_dt")
    stream = resolve_stream(seed)
    rows: list[FirstPassageRow] = []
    for index, b in enumerate(thresholds):
        level = require_finite_real(b, name="threshold")
        beta = (level - model.mean) / model.sigma
        p0 = standard_normal_cdf(-beta)
        rate = rice_upcrossing_rate(
            sigma=model.sigma,
            sigma_rate=model.sigma_rate,
            threshold=level,
            mean=model.mean,
        )
        nu_phi2 = phi2_outcrossing_rate(beta, beta, model.correlation(dt), dt)
        per_step = phi2_outcrossing_probability(beta, beta, model.correlation(h))
        child = stream.child("comparison", index)
        rows.append(
            FirstPassageRow(
                threshold=level,
                standardized_threshold=beta,
                rice_rate=rate,
                rice_bound=p0 + rate * big_t,
                rice_poisson=1.0 - (1.0 - p0) * math.exp(-rate * big_t),
                phi2_bound=p0 + nu_phi2 * big_t,
                grid_union_bound=p0 + count * per_step,
                monte_carlo=first_passage_monte_carlo(
                    model,
                    level,
                    horizon=big_t,
                    steps=count,
                    samples=mc_samples,
                    seed=child.child("mc"),
                ),
                subset=first_passage_subset_simulation(
                    model,
                    level,
                    horizon=big_t,
                    steps=count,
                    samples_per_level=subset_samples,
                    seed=child.child("ss"),
                ),
            )
        )
    return tuple(rows)
