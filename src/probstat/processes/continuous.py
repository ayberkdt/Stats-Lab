"""Continuous-time processes with exact simulation.

Counting processes: the homogeneous Poisson process and the compound Poisson
process. Diffusions with closed-form transition laws: Brownian motion with
drift, geometric Brownian motion, and the Ornstein-Uhlenbeck process. Every
sampler here is exact in distribution at the requested times, so these
processes also serve as references for the discretization schemes in
:mod:`probstat.processes.sde`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)


def _positive(value: object, *, name: str) -> float:
    """Validate a strictly positive finite real.

    Args:
        value: Candidate value.
        name: Parameter name for error messages.

    Returns:
        The value as a float.

    Raises:
        TypeError: If the value is not real.
        ValueError: If the value is non-finite or not positive.

    References:
        Project validation policy. No equation is evaluated.

    """
    number = require_finite_real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _non_negative(value: object, *, name: str) -> float:
    """Validate a non-negative finite real.

    Args:
        value: Candidate value.
        name: Parameter name for error messages.

    Returns:
        The value as a float.

    Raises:
        TypeError: If the value is not real.
        ValueError: If the value is non-finite or negative.

    References:
        Project validation policy. No equation is evaluated.

    """
    number = require_finite_real(value, name=name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _positive_integer(value: object, *, name: str) -> int:
    """Validate a strictly positive integer.

    Args:
        value: Candidate value.
        name: Parameter name for error messages.

    Returns:
        The integer.

    Raises:
        TypeError: If the value is not an integer.
        ValueError: If the value is not positive.

    References:
        Project validation policy. No equation is evaluated.

    """
    number = require_non_negative_integer(value, name=name)
    if number == 0:
        raise ValueError(f"{name} must be positive")
    return number


@dataclass(frozen=True, slots=True)
class SamplePath:
    """One realization of a scalar process on an increasing time grid.

    Args:
        times: Increasing observation times starting at the initial time.
        values: Process values at those times.

    References:
        [REF-GLASSERMAN-2003, Chapter 3].

    """

    times: tuple[float, ...]
    values: tuple[float, ...]

    @property
    def terminal(self) -> float:
        """Return the value at the final time.

        References:
            [REF-GLASSERMAN-2003, Chapter 3].

        """
        return self.values[-1]


def uniform_grid(horizon: float, steps: int) -> tuple[float, ...]:
    """Return ``steps + 1`` equally spaced times on ``[0, horizon]``.

    Args:
        horizon: Positive final time ``T``.
        steps: Positive number of intervals.

    Returns:
        Times ``k T / steps``.

    Raises:
        ValueError: If ``horizon`` or ``steps`` is not positive.

    References:
        [REF-GLASSERMAN-2003, Section 3.1]. No named equation is needed.

    """
    end = _positive(horizon, name="horizon")
    count = _positive_integer(steps, name="steps")
    return tuple(end * k / count for k in range(count)) + (end,)


@dataclass(frozen=True, slots=True)
class PoissonProcess:
    """Homogeneous Poisson counting process ``N(t)`` with rate ``lambda``.

    Args:
        rate: Positive intensity ``lambda`` (expected events per unit time).

    References:
        [REF-NORRIS-1997, Section 2.4].

    """

    rate: float

    def __post_init__(self) -> None:
        """Validate the rate.

        Raises:
            ValueError: If the rate is not positive.

        References:
            [REF-NORRIS-1997, Section 2.4]. No equation is evaluated.

        """
        object.__setattr__(self, "rate", _positive(self.rate, name="rate"))

    def mean_count(self, time: float) -> float:
        """Return ``E[N(t)] = lambda t``.

        Args:
            time: Non-negative time.

        Returns:
            Expected count.

        Equation:
            [EQ-PROC-POIS-001].

        References:
            [REF-NORRIS-1997, Section 2.4].

        """
        return self.rate * _non_negative(time, name="time")

    def variance_count(self, time: float) -> float:
        """Return ``Var[N(t)] = lambda t``.

        Args:
            time: Non-negative time.

        Returns:
            Count variance.

        Equation:
            [EQ-PROC-POIS-001].

        References:
            [REF-NORRIS-1997, Section 2.4].

        """
        return self.rate * _non_negative(time, name="time")

    def count_pmf(self, count: int, time: float) -> float:
        """Return ``Pr(N(t) = k) = e^(-lambda t) (lambda t)^k / k!``.

        Args:
            count: Non-negative integer ``k``.
            time: Non-negative time.

        Returns:
            Probability mass.

        Equation:
            [EQ-PROC-POIS-001].

        References:
            [REF-NORRIS-1997, Section 2.4].

        """
        k = require_non_negative_integer(count, name="count")
        mean = self.mean_count(time)
        if mean == 0.0:
            return 1.0 if k == 0 else 0.0
        return math.exp(k * math.log(mean) - mean - math.lgamma(k + 1.0))

    def event_times(
        self, horizon: float, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Simulate the event times in ``(0, horizon]``.

        Inter-arrival times are independent ``Exponential(lambda)`` draws.

        Args:
            horizon: Positive final time.
            seed: Integer seed or stream.

        Returns:
            Increasing event times.

        Equation:
            [EQ-PROC-POIS-002].

        References:
            [REF-NORRIS-1997, Section 2.4], [REF-DEVROYE-1986, Chapter VI].

        """
        end = _positive(horizon, name="horizon")
        generator = resolve_stream(seed).child("poisson_process").generator()
        times: list[float] = []
        current = generator.expovariate(self.rate)
        while current <= end:
            times.append(current)
            current += generator.expovariate(self.rate)
        return tuple(times)

    def sample_counts(
        self,
        horizon: float,
        size: int,
        *,
        seed: int | RandomStream | None = None,
    ) -> tuple[int, ...]:
        """Simulate ``size`` independent counts ``N(horizon)``.

        Args:
            horizon: Positive final time.
            size: Positive number of replications.
            seed: Integer seed or stream; replication ``r`` uses child ``r``.

        Returns:
            Event counts.

        Equation:
            [EQ-PROC-POIS-002].

        References:
            [REF-NORRIS-1997, Section 2.4].

        """
        count = _positive_integer(size, name="size")
        streams = resolve_stream(seed).spawn(count, label="poisson_counts")
        return tuple(len(self.event_times(horizon, seed=s)) for s in streams)


class JumpLaw(Protocol):
    """Structural interface for the jump-size law of a compound process.

    References:
        [REF-ROSS-2019, Chapter 7].

    """

    def quantile(self, probability: float) -> float:
        """Return the quantile at ``probability``.

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        ...

    @property
    def mean(self) -> float:
        """Return the jump mean.

        References:
            [REF-ROSS-2019, Chapter 7].

        """
        ...

    @property
    def variance(self) -> float:
        """Return the jump variance.

        References:
            [REF-ROSS-2019, Chapter 7].

        """
        ...


@dataclass(frozen=True, slots=True)
class CompoundPoissonProcess:
    """Compound Poisson process ``S(t) = sum_(i <= N(t)) J_i``.

    Args:
        rate: Positive intensity of the driving Poisson process.
        jump: Jump-size law with ``quantile``, ``mean``, and ``variance``;
            jumps are drawn by inversion and are independent of ``N``.

    References:
        [REF-ROSS-2019, Chapter 7], [REF-NORRIS-1997, Section 2.4].

    """

    rate: float
    jump: JumpLaw

    def __post_init__(self) -> None:
        """Validate the rate and the jump-law interface.

        Raises:
            TypeError: If the jump law lacks a required member.
            ValueError: If the rate is not positive.

        References:
            Project validation policy. No equation is evaluated.

        """
        object.__setattr__(self, "rate", _positive(self.rate, name="rate"))
        for attribute in ("quantile", "mean", "variance"):
            if not hasattr(self.jump, attribute):
                raise TypeError(f"jump law lacks required {attribute!r}")

    def mean(self, time: float) -> float:
        """Return ``E[S(t)] = lambda t E[J]``.

        Args:
            time: Non-negative time.

        Returns:
            Expected cumulative jump.

        Equation:
            [EQ-PROC-CPOIS-001].

        References:
            [REF-ROSS-2019, Chapter 7].

        """
        return self.rate * _non_negative(time, name="time") * float(self.jump.mean)

    def variance(self, time: float) -> float:
        """Return ``Var[S(t)] = lambda t E[J^2]``.

        Args:
            time: Non-negative time.

        Returns:
            Variance of the cumulative jump.

        Equation:
            [EQ-PROC-CPOIS-001].

        References:
            [REF-ROSS-2019, Chapter 7].

        """
        second_moment = float(self.jump.variance) + float(self.jump.mean) ** 2
        return self.rate * _non_negative(time, name="time") * second_moment

    def path(
        self, horizon: float, *, seed: int | RandomStream | None = None
    ) -> SamplePath:
        """Simulate one right-continuous path on ``[0, horizon]``.

        Args:
            horizon: Positive final time.
            seed: Integer seed or stream.

        Returns:
            Path with the time origin, every jump time, and the horizon.

        Equation:
            [EQ-PROC-POIS-002], [EQ-PROC-CPOIS-001].

        References:
            [REF-ROSS-2019, Chapter 7], [REF-DEVROYE-1986, Chapter II].

        """
        stream = resolve_stream(seed)
        events = PoissonProcess(self.rate).event_times(
            horizon, seed=stream.child("events")
        )
        jumps = stream.child("jumps").generator()
        times = [0.0]
        values = [0.0]
        total = 0.0
        for event in events:
            total += float(self.jump.quantile(_open_unit(jumps.random())))
            times.append(event)
            values.append(total)
        times.append(float(horizon))
        values.append(total)
        return SamplePath(tuple(times), tuple(values))

    def sample_values(
        self,
        horizon: float,
        size: int,
        *,
        seed: int | RandomStream | None = None,
    ) -> tuple[float, ...]:
        """Simulate ``size`` independent values of ``S(horizon)``.

        Args:
            horizon: Positive final time.
            size: Positive number of replications.
            seed: Integer seed or stream.

        Returns:
            Terminal cumulative jumps.

        Equation:
            [EQ-PROC-CPOIS-001].

        References:
            [REF-ROSS-2019, Chapter 7].

        """
        count = _positive_integer(size, name="size")
        streams = resolve_stream(seed).spawn(count, label="compound_poisson")
        return tuple(self.path(horizon, seed=s).terminal for s in streams)


def _open_unit(value: float) -> float:
    """Map a draw from ``[0, 1)`` into the open interval ``(0, 1)``.

    Args:
        value: Uniform draw.

    Returns:
        ``value`` or the smallest positive double when the draw is zero.

    References:
        [REF-DEVROYE-1986, Chapter II]. No named equation is needed.

    """
    return value if value > 0.0 else 5e-324


def _normal_draws(stream: RandomStream, count: int) -> tuple[float, ...]:
    """Return ``count`` standard-normal draws from one stream.

    Args:
        stream: Random stream.
        count: Number of draws.

    Returns:
        Standard-normal variates.

    References:
        [REF-DEVROYE-1986, Chapter V]. No named equation is needed.

    """
    generator = stream.generator()
    return tuple(generator.gauss(0.0, 1.0) for _ in range(count))


@dataclass(frozen=True, slots=True)
class BrownianMotion:
    """Brownian motion with drift ``X_t = x_0 + mu t + sigma W_t``.

    Args:
        drift: Finite drift ``mu``.
        volatility: Positive diffusion coefficient ``sigma``.

    References:
        [REF-GLASSERMAN-2003, Section 3.1], [REF-HIGHAM-2001].

    """

    drift: float = 0.0
    volatility: float = 1.0

    def __post_init__(self) -> None:
        """Validate the coefficients.

        Raises:
            ValueError: If the volatility is not positive.

        References:
            Project validation policy. No equation is evaluated.

        """
        object.__setattr__(self, "drift", require_finite_real(self.drift, name="drift"))
        object.__setattr__(
            self, "volatility", _positive(self.volatility, name="volatility")
        )

    def mean(self, time: float, initial: float = 0.0) -> float:
        """Return ``E[X_t] = x_0 + mu t``.

        Args:
            time: Non-negative time.
            initial: Initial value ``x_0``.

        Returns:
            Expected value.

        Equation:
            [EQ-PROC-BM-001].

        References:
            [REF-GLASSERMAN-2003, Section 3.1].

        """
        return float(initial) + self.drift * _non_negative(time, name="time")

    def variance(self, time: float) -> float:
        """Return ``Var[X_t] = sigma^2 t``.

        Args:
            time: Non-negative time.

        Returns:
            Variance.

        Equation:
            [EQ-PROC-BM-001].

        References:
            [REF-GLASSERMAN-2003, Section 3.1].

        """
        return self.volatility**2 * _non_negative(time, name="time")

    def path(
        self,
        horizon: float,
        steps: int,
        *,
        initial: float = 0.0,
        seed: int | RandomStream | None = None,
    ) -> SamplePath:
        """Simulate the process exactly on a uniform grid.

        Args:
            horizon: Positive final time.
            steps: Positive number of intervals.
            initial: Initial value.
            seed: Integer seed or stream.

        Returns:
            Sample path with independent Gaussian increments.

        Equation:
            [EQ-PROC-BM-001].

        References:
            [REF-GLASSERMAN-2003, Section 3.1].

        """
        times = uniform_grid(horizon, steps)
        h = times[-1] / steps
        draws = _normal_draws(resolve_stream(seed).child("brownian"), steps)
        step_scale = self.volatility * math.sqrt(h)
        values = [require_finite_real(initial, name="initial")]
        for z in draws:
            values.append(values[-1] + self.drift * h + step_scale * z)
        return SamplePath(times, tuple(values))


def brownian_increments(
    horizon: float, steps: int, *, seed: int | RandomStream | None = None
) -> tuple[float, ...]:
    """Return the increments of standard Brownian motion on a uniform grid.

    Args:
        horizon: Positive final time.
        steps: Positive number of intervals.
        seed: Integer seed or stream.

    Returns:
        ``steps`` independent ``N(0, horizon / steps)`` increments.

    Equation:
        [EQ-PROC-BM-001].

    References:
        [REF-HIGHAM-2001], [REF-GLASSERMAN-2003, Section 3.1].

    """
    end = _positive(horizon, name="horizon")
    count = _positive_integer(steps, name="steps")
    scale = math.sqrt(end / count)
    draws = _normal_draws(resolve_stream(seed).child("brownian"), count)
    return tuple(scale * z for z in draws)


@dataclass(frozen=True, slots=True)
class GeometricBrownianMotion:
    """Geometric Brownian motion ``dS = mu S dt + sigma S dW``.

    Args:
        drift: Finite drift ``mu``.
        volatility: Positive volatility ``sigma``.

    References:
        [REF-GLASSERMAN-2003, Section 3.2], [REF-HIGHAM-2001].

    """

    drift: float
    volatility: float

    def __post_init__(self) -> None:
        """Validate the coefficients.

        Raises:
            ValueError: If the volatility is not positive.

        References:
            Project validation policy. No equation is evaluated.

        """
        object.__setattr__(self, "drift", require_finite_real(self.drift, name="drift"))
        object.__setattr__(
            self, "volatility", _positive(self.volatility, name="volatility")
        )

    def exact(self, initial: float, time: float, brownian: float) -> float:
        """Return ``S_t = S_0 exp((mu - sigma^2/2) t + sigma W_t)``.

        Args:
            initial: Positive initial value ``S_0``.
            time: Non-negative time ``t``.
            brownian: Value of the driving Brownian motion ``W_t``.

        Returns:
            Exact solution for that Brownian value.

        Equation:
            [EQ-PROC-GBM-001].

        References:
            [REF-GLASSERMAN-2003, Section 3.2], [REF-HIGHAM-2001].

        """
        t = _non_negative(time, name="time")
        return _positive(initial, name="initial") * math.exp(
            (self.drift - 0.5 * self.volatility**2) * t + self.volatility * brownian
        )

    def mean(self, initial: float, time: float) -> float:
        """Return ``E[S_t] = S_0 e^(mu t)``.

        Args:
            initial: Positive initial value.
            time: Non-negative time.

        Returns:
            Expected value.

        Equation:
            [EQ-PROC-GBM-002].

        References:
            [REF-GLASSERMAN-2003, Section 3.2].

        """
        return _positive(initial, name="initial") * math.exp(
            self.drift * _non_negative(time, name="time")
        )

    def variance(self, initial: float, time: float) -> float:
        """Return ``Var[S_t] = S_0^2 e^(2 mu t) (e^(sigma^2 t) - 1)``.

        Args:
            initial: Positive initial value.
            time: Non-negative time.

        Returns:
            Variance.

        Equation:
            [EQ-PROC-GBM-002].

        References:
            [REF-GLASSERMAN-2003, Section 3.2].

        """
        s0 = _positive(initial, name="initial")
        t = _non_negative(time, name="time")
        return (
            s0
            * s0
            * math.exp(2.0 * self.drift * t)
            * math.expm1(self.volatility**2 * t)
        )

    def path(
        self,
        horizon: float,
        steps: int,
        *,
        initial: float = 1.0,
        seed: int | RandomStream | None = None,
    ) -> SamplePath:
        """Simulate the process exactly on a uniform grid.

        Args:
            horizon: Positive final time.
            steps: Positive number of intervals.
            initial: Positive initial value.
            seed: Integer seed or stream.

        Returns:
            Exact sample path (no discretization error at grid times).

        Equation:
            [EQ-PROC-GBM-001].

        References:
            [REF-GLASSERMAN-2003, Section 3.2].

        """
        times = uniform_grid(horizon, steps)
        h = times[-1] / steps
        draws = _normal_draws(resolve_stream(seed).child("gbm"), steps)
        growth = (self.drift - 0.5 * self.volatility**2) * h
        scale = self.volatility * math.sqrt(h)
        values = [_positive(initial, name="initial")]
        for z in draws:
            values.append(values[-1] * math.exp(growth + scale * z))
        return SamplePath(times, tuple(values))

    def sample_terminal(
        self,
        initial: float,
        time: float,
        size: int,
        *,
        seed: int | RandomStream | None = None,
    ) -> tuple[float, ...]:
        """Draw ``size`` exact independent values of ``S_t``.

        Args:
            initial: Positive initial value.
            time: Positive time.
            size: Positive number of draws.
            seed: Integer seed or stream.

        Returns:
            Log-normal draws of ``S_t``.

        Equation:
            [EQ-PROC-GBM-001].

        References:
            [REF-GLASSERMAN-2003, Section 3.2].

        """
        t = _positive(time, name="time")
        count = _positive_integer(size, name="size")
        draws = _normal_draws(resolve_stream(seed).child("gbm_terminal"), count)
        return tuple(self.exact(initial, t, math.sqrt(t) * z) for z in draws)


@dataclass(frozen=True, slots=True)
class OrnsteinUhlenbeck:
    """Ornstein-Uhlenbeck process ``dX = theta (mu - X) dt + sigma dW``.

    Args:
        reversion: Positive mean-reversion rate ``theta``.
        level: Finite long-run mean ``mu``.
        volatility: Positive diffusion coefficient ``sigma``.

    References:
        [REF-GILLESPIE-1996], [REF-GLASSERMAN-2003, Section 3.3].

    """

    reversion: float
    level: float
    volatility: float

    def __post_init__(self) -> None:
        """Validate the coefficients.

        Raises:
            ValueError: If the reversion rate or volatility is not positive.

        References:
            Project validation policy. No equation is evaluated.

        """
        object.__setattr__(
            self, "reversion", _positive(self.reversion, name="reversion")
        )
        object.__setattr__(self, "level", require_finite_real(self.level, name="level"))
        object.__setattr__(
            self, "volatility", _positive(self.volatility, name="volatility")
        )

    def mean(self, initial: float, time: float) -> float:
        """Return ``E[X_t | X_0] = mu + (X_0 - mu) e^(-theta t)``.

        Args:
            initial: Initial value.
            time: Non-negative time.

        Returns:
            Conditional mean.

        Equation:
            [EQ-PROC-OU-001].

        References:
            [REF-GILLESPIE-1996].

        """
        t = _non_negative(time, name="time")
        x0 = require_finite_real(initial, name="initial")
        return self.level + (x0 - self.level) * math.exp(-self.reversion * t)

    def variance(self, time: float) -> float:
        """Return ``Var[X_t | X_0] = sigma^2 (1 - e^(-2 theta t)) / (2 theta)``.

        Args:
            time: Non-negative time.

        Returns:
            Conditional variance.

        Equation:
            [EQ-PROC-OU-001].

        References:
            [REF-GILLESPIE-1996].

        """
        t = _non_negative(time, name="time")
        return (
            self.volatility**2
            * -math.expm1(-2.0 * self.reversion * t)
            / (2.0 * self.reversion)
        )

    @property
    def stationary_variance(self) -> float:
        """Return the stationary variance ``sigma^2 / (2 theta)``.

        Equation:
            [EQ-PROC-OU-002].

        References:
            [REF-GILLESPIE-1996].

        """
        return self.volatility**2 / (2.0 * self.reversion)

    def stationary_covariance(self, lag: float) -> float:
        """Return ``Cov(X_t, X_(t+s)) = sigma^2 e^(-theta |s|) / (2 theta)``.

        Args:
            lag: Time lag ``s``.

        Returns:
            Stationary autocovariance (an exponential kernel).

        Equation:
            [EQ-PROC-OU-002].

        References:
            [REF-GILLESPIE-1996], [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        s = abs(require_finite_real(lag, name="lag"))
        return self.stationary_variance * math.exp(-self.reversion * s)

    def path(
        self,
        horizon: float,
        steps: int,
        *,
        initial: float | None = None,
        seed: int | RandomStream | None = None,
    ) -> SamplePath:
        """Simulate the process exactly on a uniform grid.

        Args:
            horizon: Positive final time.
            steps: Positive number of intervals.
            initial: Initial value; ``None`` draws it from the stationary law.
            seed: Integer seed or stream.

        Returns:
            Exact sample path from the Gaussian transition law.

        Equation:
            [EQ-PROC-OU-001], [EQ-PROC-OU-002].

        References:
            [REF-GILLESPIE-1996].

        """
        times = uniform_grid(horizon, steps)
        h = times[-1] / steps
        stream = resolve_stream(seed).child("ornstein_uhlenbeck")
        draws = _normal_draws(stream, steps + 1)
        if initial is None:
            start = self.level + math.sqrt(self.stationary_variance) * draws[0]
        else:
            start = require_finite_real(initial, name="initial")
        decay = math.exp(-self.reversion * h)
        scale = math.sqrt(self.variance(h))
        values = [start]
        for z in draws[1:]:
            values.append(self.level + (values[-1] - self.level) * decay + scale * z)
        return SamplePath(times, tuple(values))

    def sample_terminal(
        self,
        initial: float,
        time: float,
        size: int,
        *,
        seed: int | RandomStream | None = None,
    ) -> tuple[float, ...]:
        """Draw ``size`` exact independent values of ``X_t`` given ``X_0``.

        Args:
            initial: Initial value.
            time: Positive time.
            size: Positive number of draws.
            seed: Integer seed or stream.

        Returns:
            Gaussian draws from the transition law.

        Equation:
            [EQ-PROC-OU-001].

        References:
            [REF-GILLESPIE-1996].

        """
        t = _positive(time, name="time")
        count = _positive_integer(size, name="size")
        mean = self.mean(initial, t)
        scale = math.sqrt(self.variance(t))
        draws = _normal_draws(resolve_stream(seed).child("ou_terminal"), count)
        return tuple(mean + scale * z for z in draws)
