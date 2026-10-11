"""Time-stepping schemes for scalar Ito stochastic differential equations.

For ``dX = a(t, X) dt + b(t, X) dW`` on a uniform grid with step ``h`` and
Brownian increments ``dW_n ~ N(0, h)``:

* Euler-Maruyama: ``X_(n+1) = X_n + a h + b dW_n`` (strong order 1/2, weak
  order 1 under the usual Lipschitz and growth conditions);
* Milstein: adds ``(1/2) b b_x (dW_n^2 - h)`` (strong order 1).

The convergence studies drive every step size with the SAME fine Brownian
path (coarse increments are block sums), so strong errors compare the scheme
and the exact solution pathwise, as in Higham (2001).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real

from .continuous import SamplePath, _positive, _positive_integer, uniform_grid

SCHEMES = ("euler_maruyama", "milstein")


@dataclass(frozen=True, slots=True)
class ScalarSDE:
    """Scalar Ito SDE ``dX = a(t, X) dt + b(t, X) dW``.

    Args:
        drift: Drift coefficient ``a(t, x)``.
        diffusion: Diffusion coefficient ``b(t, x)``.
        diffusion_derivative: Optional ``db/dx (t, x)``, required by Milstein.

    References:
        [REF-KLOEDEN-PLATEN-1992, Chapter 4], [REF-HIGHAM-2001].

    """

    drift: Callable[[float, float], float]
    diffusion: Callable[[float, float], float]
    diffusion_derivative: Callable[[float, float], float] | None = None

    def __post_init__(self) -> None:
        """Validate that the coefficients are callable.

        Raises:
            TypeError: If a coefficient is not callable.

        References:
            Project validation policy. No equation is evaluated.

        """
        if not callable(self.drift) or not callable(self.diffusion):
            raise TypeError("drift and diffusion must be callable")
        if self.diffusion_derivative is not None and not callable(
            self.diffusion_derivative
        ):
            raise TypeError("diffusion_derivative must be callable or None")

    @classmethod
    def geometric_brownian_motion(cls, drift: float, volatility: float) -> ScalarSDE:
        """Return ``dX = mu X dt + sigma X dW`` with its diffusion derivative.

        Args:
            drift: Finite drift ``mu``.
            volatility: Finite volatility ``sigma``.

        Returns:
            Linear SDE with ``a = mu x``, ``b = sigma x``, ``b_x = sigma``.

        Equation:
            [EQ-PROC-GBM-001].

        References:
            [REF-HIGHAM-2001].

        """
        mu = require_finite_real(drift, name="drift")
        sigma = require_finite_real(volatility, name="volatility")

        def a(t: float, x: float) -> float:
            """Return the GBM drift ``mu x``.

            References:
                [REF-HIGHAM-2001].

            """
            return mu * x

        def b(t: float, x: float) -> float:
            """Return the GBM diffusion ``sigma x``.

            References:
                [REF-HIGHAM-2001].

            """
            return sigma * x

        def b_x(t: float, x: float) -> float:
            """Return the GBM diffusion derivative ``sigma``.

            References:
                [REF-HIGHAM-2001].

            """
            return sigma

        return cls(a, b, b_x)

    @classmethod
    def ornstein_uhlenbeck(
        cls, reversion: float, level: float, volatility: float
    ) -> ScalarSDE:
        """Return ``dX = theta (mu - X) dt + sigma dW`` (additive noise).

        Args:
            reversion: Finite reversion rate ``theta``.
            level: Finite long-run mean ``mu``.
            volatility: Finite diffusion coefficient ``sigma``.

        Returns:
            SDE with constant diffusion, for which Milstein equals
            Euler-Maruyama.

        Equation:
            [EQ-PROC-OU-001].

        References:
            [REF-GILLESPIE-1996], [REF-KLOEDEN-PLATEN-1992, Chapter 10].

        """
        theta = require_finite_real(reversion, name="reversion")
        mu = require_finite_real(level, name="level")
        sigma = require_finite_real(volatility, name="volatility")

        def a(t: float, x: float) -> float:
            """Return the OU drift ``theta (mu - x)``.

            References:
                [REF-GILLESPIE-1996].

            """
            return theta * (mu - x)

        def b(t: float, x: float) -> float:
            """Return the constant OU diffusion ``sigma``.

            References:
                [REF-GILLESPIE-1996].

            """
            return sigma

        def b_x(t: float, x: float) -> float:
            """Return the zero derivative of a constant diffusion.

            References:
                [REF-GILLESPIE-1996].

            """
            return 0.0

        return cls(a, b, b_x)


def _increments(
    horizon: float,
    steps: int,
    seed: int | RandomStream | None,
    increments: Sequence[float] | None,
) -> tuple[float, ...]:
    """Return supplied increments or draw ``N(0, h)`` increments.

    Args:
        horizon: Positive final time.
        steps: Positive number of steps.
        seed: Seed used when ``increments`` is ``None``.
        increments: Optional prescribed Brownian increments.

    Returns:
        ``steps`` Brownian increments.

    Raises:
        ValueError: If the supplied increments have the wrong length.

    References:
        [REF-HIGHAM-2001]. No named equation is needed.

    """
    if increments is not None:
        values = tuple(
            require_finite_real(v, name=f"increments[{i}]")
            for i, v in enumerate(increments)
        )
        if len(values) != steps:
            raise ValueError("increments length must equal steps")
        return values
    generator = resolve_stream(seed).child("sde_increments").generator()
    scale = math.sqrt(horizon / steps)
    return tuple(scale * generator.gauss(0.0, 1.0) for _ in range(steps))


def _step_function(
    sde: ScalarSDE, scheme: str
) -> Callable[[float, float, float, float], float]:
    """Return the one-step map ``(t, x, h, dW) -> x_next`` of a scheme.

    Args:
        sde: Equation to integrate.
        scheme: ``"euler_maruyama"`` or ``"milstein"``.

    Returns:
        One-step update.

    Raises:
        ValueError: If the scheme is unknown or Milstein lacks ``b_x``.

    Equation:
        [EQ-PROC-SDE-001], [EQ-PROC-SDE-002].

    References:
        [REF-KLOEDEN-PLATEN-1992, Sections 10.2-10.3], [REF-HIGHAM-2001].

    """
    a, b = sde.drift, sde.diffusion
    if scheme == "euler_maruyama":

        def euler(t: float, x: float, h: float, dw: float) -> float:
            """Apply one Euler-Maruyama step.

            References:
                [EQ-PROC-SDE-001], [REF-HIGHAM-2001].

            """
            return x + a(t, x) * h + b(t, x) * dw

        return euler
    if scheme == "milstein":
        derivative = sde.diffusion_derivative
        if derivative is None:
            raise ValueError("milstein requires diffusion_derivative")

        def milstein_step(t: float, x: float, h: float, dw: float) -> float:
            """Apply one Milstein step.

            References:
                [EQ-PROC-SDE-002], [REF-HIGHAM-2001].

            """
            bx = b(t, x)
            return (
                x + a(t, x) * h + bx * dw + 0.5 * bx * derivative(t, x) * (dw * dw - h)
            )

        return milstein_step
    raise ValueError(f"scheme must be one of {SCHEMES}")


def _integrate(
    step: Callable[[float, float, float, float], float],
    initial: float,
    horizon: float,
    increments: Sequence[float],
) -> list[float]:
    """Apply a one-step map along prescribed increments.

    Args:
        step: One-step update.
        initial: Initial value.
        horizon: Final time.
        increments: Brownian increments on a uniform grid.

    Returns:
        Values at every grid time, including the initial value.

    Raises:
        ValueError: If the iteration produces a non-finite value.

    References:
        [REF-KLOEDEN-PLATEN-1992, Chapter 10]. No additional equation.

    """
    h = horizon / len(increments)
    values = [initial]
    x = initial
    for n, dw in enumerate(increments):
        x = step(n * h, x, h, dw)
        if not math.isfinite(x):
            raise ValueError("scheme produced a non-finite value; reduce the step")
        values.append(x)
    return values


def solve_sde(
    sde: ScalarSDE,
    initial: float,
    horizon: float,
    steps: int,
    *,
    scheme: str = "euler_maruyama",
    seed: int | RandomStream | None = None,
    increments: Sequence[float] | None = None,
) -> SamplePath:
    """Integrate a scalar SDE on a uniform grid.

    Args:
        sde: Equation to integrate.
        initial: Finite initial value.
        horizon: Positive final time.
        steps: Positive number of steps.
        scheme: ``"euler_maruyama"`` or ``"milstein"``.
        seed: Seed for the Brownian increments when none are supplied.
        increments: Optional prescribed Brownian increments of length
            ``steps`` (each nominally ``N(0, horizon / steps)``).

    Returns:
        Approximate sample path.

    Raises:
        ValueError: If inputs are invalid or the iteration diverges.

    Equation:
        [EQ-PROC-SDE-001], [EQ-PROC-SDE-002].

    References:
        [REF-KLOEDEN-PLATEN-1992, Sections 10.2-10.3], [REF-HIGHAM-2001].

    """
    end = _positive(horizon, name="horizon")
    count = _positive_integer(steps, name="steps")
    start = require_finite_real(initial, name="initial")
    step = _step_function(sde, scheme)
    path = _integrate(step, start, end, _increments(end, count, seed, increments))
    return SamplePath(uniform_grid(end, count), tuple(path))


def euler_maruyama(
    sde: ScalarSDE,
    initial: float,
    horizon: float,
    steps: int,
    *,
    seed: int | RandomStream | None = None,
    increments: Sequence[float] | None = None,
) -> SamplePath:
    """Integrate a scalar SDE by the Euler-Maruyama scheme.

    Args:
        sde: Equation to integrate.
        initial: Finite initial value.
        horizon: Positive final time.
        steps: Positive number of steps.
        seed: Seed for the Brownian increments when none are supplied.
        increments: Optional prescribed Brownian increments.

    Returns:
        Approximate sample path.

    Equation:
        [EQ-PROC-SDE-001].

    References:
        [REF-KLOEDEN-PLATEN-1992, Section 10.2], [REF-HIGHAM-2001].

    """
    return solve_sde(
        sde,
        initial,
        horizon,
        steps,
        scheme="euler_maruyama",
        seed=seed,
        increments=increments,
    )


def milstein(
    sde: ScalarSDE,
    initial: float,
    horizon: float,
    steps: int,
    *,
    seed: int | RandomStream | None = None,
    increments: Sequence[float] | None = None,
) -> SamplePath:
    """Integrate a scalar SDE by the Milstein scheme.

    Args:
        sde: Equation with ``diffusion_derivative``.
        initial: Finite initial value.
        horizon: Positive final time.
        steps: Positive number of steps.
        seed: Seed for the Brownian increments when none are supplied.
        increments: Optional prescribed Brownian increments.

    Returns:
        Approximate sample path.

    Raises:
        ValueError: If the SDE has no diffusion derivative.

    Equation:
        [EQ-PROC-SDE-002].

    References:
        [REF-KLOEDEN-PLATEN-1992, Section 10.3], [REF-HIGHAM-2001].

    """
    return solve_sde(
        sde,
        initial,
        horizon,
        steps,
        scheme="milstein",
        seed=seed,
        increments=increments,
    )


@dataclass(frozen=True, slots=True)
class ConvergenceStudy:
    """Monte Carlo estimate of strong or weak discretization error.

    Args:
        scheme: Scheme name.
        kind: ``"strong"`` or ``"weak"``.
        step_sizes: Step sizes ``h`` in decreasing order.
        errors: Estimated errors, one per step size.
        standard_errors: Monte Carlo standard errors of those estimates.
        order: Least-squares slope of ``log error`` against ``log h``.
        paths: Number of Brownian paths.

    References:
        [REF-KLOEDEN-PLATEN-1992, Section 9.6], [REF-HIGHAM-2001].

    """

    scheme: str
    kind: str
    step_sizes: tuple[float, ...]
    errors: tuple[float, ...]
    standard_errors: tuple[float, ...]
    order: float
    paths: int


def empirical_order(step_sizes: Sequence[float], errors: Sequence[float]) -> float:
    """Return the least-squares slope ``q`` in ``log e = log C + q log h``.

    Args:
        step_sizes: At least two distinct positive step sizes.
        errors: Positive errors of matching length.

    Returns:
        Fitted convergence order.

    Raises:
        ValueError: If lengths differ, fewer than two points are given, or a
            value is not positive.

    Equation:
        [EQ-PROC-SDE-004].

    References:
        [REF-HIGHAM-2001], [REF-MONTGOMERY-PECK-VINING-2021, Chapter 2].

    """
    if len(step_sizes) != len(errors) or len(errors) < 2:
        raise ValueError("need at least two matching step sizes and errors")
    if any(h <= 0.0 for h in step_sizes) or any(e <= 0.0 for e in errors):
        raise ValueError("step sizes and errors must be positive")
    xs = [math.log(h) for h in step_sizes]
    ys = [math.log(e) for e in errors]
    x_bar = math.fsum(xs) / len(xs)
    y_bar = math.fsum(ys) / len(ys)
    sxx = math.fsum((x - x_bar) ** 2 for x in xs)
    if sxx == 0.0:
        raise ValueError("step sizes must not all be equal")
    sxy = math.fsum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys, strict=True))
    return sxy / sxx


def _validated_counts(step_counts: Sequence[int]) -> tuple[int, ...]:
    """Validate step counts that all divide the finest one.

    Args:
        step_counts: At least two distinct positive step counts.

    Returns:
        Sorted increasing step counts.

    Raises:
        ValueError: If fewer than two distinct counts are given or a count
            does not divide the finest count.

    References:
        [REF-HIGHAM-2001]. No equation is evaluated.

    """
    counts = tuple(
        sorted({_positive_integer(c, name="step_counts[i]") for c in step_counts})
    )
    if len(counts) < 2:
        raise ValueError("at least two distinct step counts are required")
    finest = counts[-1]
    if any(finest % c for c in counts):
        raise ValueError("every step count must divide the finest step count")
    return counts


def _coarsen(fine: Sequence[float], count: int) -> tuple[float, ...]:
    """Sum consecutive blocks of fine increments into ``count`` increments.

    Args:
        fine: Fine Brownian increments.
        count: Coarse step count dividing ``len(fine)``.

    Returns:
        Coarse increments on the same Brownian path.

    Equation:
        [EQ-PROC-BM-001].

    References:
        [REF-HIGHAM-2001].

    """
    block = len(fine) // count
    return tuple(math.fsum(fine[k * block : (k + 1) * block]) for k in range(count))


def _mean_and_standard_error(values: Sequence[float]) -> tuple[float, float]:
    """Return the sample mean and its standard error.

    Args:
        values: At least two values.

    Returns:
        ``(mean, standard error)``.

    Equation:
        [EQ-MC-MEAN-001].

    References:
        [REF-OWEN-2013, Chapter 2].

    """
    count = len(values)
    mean = math.fsum(values) / count
    variance = math.fsum((v - mean) ** 2 for v in values) / (count - 1)
    return mean, math.sqrt(variance / count)


def _simulate_terminals(
    sde: ScalarSDE,
    initial: float,
    horizon: float,
    counts: tuple[int, ...],
    paths: int,
    scheme: str,
    seed: int | RandomStream | None,
) -> tuple[list[list[float]], list[float]]:
    """Return terminal values for every step count plus each path's ``W_T``.

    Args:
        sde: Equation.
        initial: Initial value.
        horizon: Final time.
        counts: Validated step counts.
        paths: Number of Brownian paths.
        scheme: Scheme name.
        seed: Root seed; path ``r`` uses child ``("convergence", r)``.

    Returns:
        ``terminals[c][r]`` for count index ``c`` and path ``r``, and the
        terminal Brownian values ``W_T`` per path.

    Equation:
        [EQ-PROC-SDE-001], [EQ-PROC-SDE-002].

    References:
        [REF-HIGHAM-2001].

    """
    step = _step_function(sde, scheme)
    finest = counts[-1]
    scale = math.sqrt(horizon / finest)
    root = resolve_stream(seed)
    terminals: list[list[float]] = [[] for _ in counts]
    brownian: list[float] = []
    for r in range(paths):
        generator = root.child("convergence", r).generator()
        fine = tuple(scale * generator.gauss(0.0, 1.0) for _ in range(finest))
        brownian.append(math.fsum(fine))
        for index, count in enumerate(counts):
            coarse = fine if count == finest else _coarsen(fine, count)
            terminals[index].append(_integrate(step, initial, horizon, coarse)[-1])
    return terminals, brownian


def strong_convergence_study(
    sde: ScalarSDE,
    initial: float,
    horizon: float,
    exact: Callable[[float], float],
    *,
    step_counts: Sequence[int] = (16, 32, 64, 128),
    paths: int = 500,
    scheme: str = "euler_maruyama",
    seed: int | RandomStream | None = None,
) -> ConvergenceStudy:
    """Estimate the strong error ``E|X_N - X(T)|`` against an exact solution.

    Args:
        sde: Equation.
        initial: Finite initial value.
        horizon: Positive final time ``T``.
        exact: Map from the terminal Brownian value ``W_T`` to the exact
            ``X(T)`` (available for linear SDEs such as GBM).
        step_counts: At least two step counts dividing the finest one.
        paths: Number of Brownian paths, at least two.
        scheme: ``"euler_maruyama"`` or ``"milstein"``.
        seed: Integer seed or stream.

    Returns:
        Convergence study whose ``order`` estimates the strong order.

    Raises:
        ValueError: If inputs are invalid.

    Equation:
        [EQ-PROC-SDE-003], [EQ-PROC-SDE-004].

    References:
        [REF-KLOEDEN-PLATEN-1992, Section 9.6], [REF-HIGHAM-2001].

    """
    end = _positive(horizon, name="horizon")
    start = require_finite_real(initial, name="initial")
    counts = _validated_counts(step_counts)
    replications = _positive_integer(paths, name="paths")
    if replications < 2:
        raise ValueError("paths must be at least two")
    terminals, brownian = _simulate_terminals(
        sde, start, end, counts, replications, scheme, seed
    )
    reference = [float(exact(w)) for w in brownian]
    errors: list[float] = []
    standard_errors: list[float] = []
    for values in terminals:
        mean, error = _mean_and_standard_error(
            [abs(v - x) for v, x in zip(values, reference, strict=True)]
        )
        errors.append(mean)
        standard_errors.append(error)
    sizes = tuple(end / c for c in counts)
    return ConvergenceStudy(
        scheme,
        "strong",
        sizes,
        tuple(errors),
        tuple(standard_errors),
        empirical_order(sizes, errors),
        replications,
    )


def weak_convergence_study(
    sde: ScalarSDE,
    initial: float,
    horizon: float,
    exact_expectation: float,
    *,
    functional: Callable[[float], float] | None = None,
    step_counts: Sequence[int] = (8, 16, 32, 64),
    paths: int = 2000,
    scheme: str = "euler_maruyama",
    seed: int | RandomStream | None = None,
) -> ConvergenceStudy:
    """Estimate the weak error ``|E f(X_N) - E f(X(T))|``.

    The Monte Carlo estimate of ``E f(X_N)`` carries sampling error of order
    ``1 / sqrt(paths)``; the study is meaningful only while the bias at the
    finest step clearly exceeds the reported standard errors.

    Args:
        sde: Equation.
        initial: Finite initial value.
        horizon: Positive final time.
        exact_expectation: Exact ``E f(X(T))``.
        functional: Test function ``f``; identity when ``None``.
        step_counts: At least two step counts dividing the finest one.
        paths: Number of Brownian paths, at least two.
        scheme: ``"euler_maruyama"`` or ``"milstein"``.
        seed: Integer seed or stream.

    Returns:
        Convergence study whose ``order`` estimates the weak order.

    Raises:
        ValueError: If inputs are invalid or an estimated error is zero.

    Equation:
        [EQ-PROC-SDE-003], [EQ-PROC-SDE-004].

    References:
        [REF-KLOEDEN-PLATEN-1992, Section 9.7], [REF-HIGHAM-2001].

    """
    end = _positive(horizon, name="horizon")
    start = require_finite_real(initial, name="initial")
    target = require_finite_real(exact_expectation, name="exact_expectation")
    counts = _validated_counts(step_counts)
    replications = _positive_integer(paths, name="paths")
    if replications < 2:
        raise ValueError("paths must be at least two")
    f = functional if functional is not None else (lambda value: value)
    terminals, _ = _simulate_terminals(
        sde, start, end, counts, replications, scheme, seed
    )
    errors: list[float] = []
    standard_errors: list[float] = []
    for values in terminals:
        mean, error = _mean_and_standard_error([float(f(v)) for v in values])
        errors.append(abs(mean - target))
        standard_errors.append(error)
    sizes = tuple(end / c for c in counts)
    return ConvergenceStudy(
        scheme,
        "weak",
        sizes,
        tuple(errors),
        tuple(standard_errors),
        empirical_order(sizes, errors),
        replications,
    )
