"""Gaussian-process anomaly detection for synthetic spacecraft telemetry.

Experimental (added in 0.28).

A telemetry channel (a temperature, a bus current) is modelled as a Gaussian
process (GP) with an orbital periodic component plus a smooth trend plus white
observation noise, fitted on a nominal window. New samples are scored against
the one-step-ahead predictive law of a *sliding-window* GP of bounded cost
``O(w^2)`` per sample (rank-one Cholesky update and downdate; no state-space
form is used, so the cost is bounded by the window ``w`` rather than ``O(1)``).
All channels in this module are synthetic; no real vehicle data are involved.

Scores (per sample ``t`` with observation ``y_t``, predictive mean ``mu_t`` and
observation variance ``s_t^2 = sigma_f^2 + sigma_n^2``):

* standardized residual ``z_t = (y_t - mu_t) / s_t`` [EQ-SUR-GPANOM-001];
* negative log predictive density, Gaussian or Student-t [EQ-SUR-GPANOM-002];
* two-sided CUSUM of ``z`` [EQ-SUR-GPANOM-003].

Thresholds for a target per-sample false-alarm rate ``alpha``: the analytic
Gaussian or Student-t quantile of ``|z|`` [EQ-SUR-GPANOM-005], or the split
conformal quantile of the score on held-out nominal data
[EQ-SUR-GPANOM-004]. The conformal guarantee is marginal and needs
exchangeable scores; the one-step-ahead standardized residuals of a correct
model are approximately independent, but a time series is not exchangeable in
general and the guarantee is then only approximate (the rate is measured on
fresh nominal data by :func:`false_alarm_rate`, never assumed). A Student-t
observation model fitted by EM down-weights outliers in the nominal training
data [EQ-SUR-GPANOM-006]. Synthetic faults (spike, step, drift, variance
increase) and a rolling mean/standard-deviation baseline support operating
characteristics [EQ-SUR-GPANOM-008].

Assumptions:
    Time stamps are strictly increasing; the nominal training window is fault
    free (or contaminated only mildly when ``robust=True``); calibration data
    are disjoint from the training window.

Common pitfalls:
    - A sliding window that keeps absorbing a slow drift can hide it; use
      ``gate`` or ``robust_updates`` in :func:`sliding_window_scores` and a
      CUSUM on ``z``.
    - The CUSUM threshold calibrated here is an empirical alarm-rate match on
      nominal data, not a conformal guarantee.
    - The periodic kernel's likelihood is multimodal in the period; keep
      ``period_bounds`` tight around the orbital period.

References:
    [REF-RASMUSSEN-WILLIAMS-2006], [REF-PAGE-1954],
    [REF-VOVK-GAMMERMAN-SHAFER-2005], [REF-LEI-ETAL-2018],
    [REF-LANGE-LITTLE-TAYLOR-1989], [REF-SHAH-WILSON-GHAHRAMANI-2014].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from operator import mul

from probstat.distributions import Normal, StudentT
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.linalg import Vector, as_vector, cholesky, solve_spd

from .gp_advanced import (
    Kernel,
    PeriodicKernel,
    StationaryKernel,
    SumKernel,
    WhiteKernel,
    fit_kernel_gp,
)

__all__ = [
    "FAULT_KINDS",
    "ChannelSpec",
    "CusumResult",
    "DetectionStudy",
    "FaultedSeries",
    "NominalModel",
    "OperatingPoint",
    "ScoreSeries",
    "SlidingWindowGP",
    "SyntheticChannel",
    "batch_window_predict",
    "calibrate_cusum_threshold",
    "conformal_threshold",
    "contaminate_values",
    "cusum",
    "detection_study",
    "false_alarm_rate",
    "fit_nominal_model",
    "gaussian_threshold",
    "inject_fault",
    "orbital_kernel",
    "rolling_zscore",
    "simulate_telemetry",
    "sliding_window_scores",
    "student_t_threshold",
]

FAULT_KINDS = ("spike", "step", "drift", "variance")
_PERIOD_GRID = 24
_METHODS = ("gp_abs_z", "gp_cusum", "baseline_abs_z", "baseline_cusum")


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the inner product of two equal-length sequences.

    References:
        No equation is evaluated; plain summation [REF-GOLUB-VANLOAN-2013].

    """
    return float(sum(map(mul, left, right)))


def _forward(rows: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Solve ``L w = b`` for a ragged lower-triangular factor.

    Equation:
        [EQ-SUR-GPANOM-007].

    References:
        [REF-GOLUB-VANLOAN-2013].

    """
    out: list[float] = []
    for i, row in enumerate(rows):
        out.append((rhs[i] - _dot(row[:i], out)) / row[i])
    return out


def _chol_update(rows: list[list[float]], x: list[float]) -> None:
    """Replace ``L`` by the factor of ``L L' + x x'`` in place.

    Equation:
        [EQ-SUR-GPANOM-007].

    References:
        [REF-GOLUB-VANLOAN-2013].

    """
    size = len(rows)
    for k in range(size):
        lkk = rows[k][k]
        radius = math.hypot(lkk, x[k])
        c = radius / lkk
        s = x[k] / lkk
        rows[k][k] = radius
        for i in range(k + 1, size):
            updated = (rows[i][k] + s * x[i]) / c
            x[i] = c * x[i] - s * updated
            rows[i][k] = updated


def _check_series(
    times: Sequence[float], values: Sequence[float], *, minimum: int = 1
) -> tuple[Vector, Vector]:
    """Validate strictly increasing times and finite values of equal length.

    Raises:
        ValueError: On a length mismatch, too few points or unordered times.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    t = as_vector(times, name="times")
    y = as_vector(values, name="values")
    if len(t) != len(y):
        raise ValueError("times and values must have the same length")
    if len(t) < minimum:
        raise ValueError(f"need at least {minimum} samples")
    if any(b <= a for a, b in zip(t, t[1:], strict=False)):
        raise ValueError("times must be strictly increasing")
    return t, y


def _check_alpha(alpha: float) -> float:
    """Return ``alpha`` after checking ``0 < alpha < 1``.

    Raises:
        ValueError: If ``alpha`` is not a finite number in ``(0, 1)``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(alpha, bool) or not isinstance(alpha, int | float):
        raise TypeError("alpha must be a real number")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return float(alpha)


# ---------------------------------------------------------------------------
# synthetic channel
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """Parameters of a synthetic telemetry channel (arbitrary units, minutes).

    The noise-free signal is ``sum_j a_j sin(2 pi j t / P + phi_j)`` plus a
    Matern-3/2 slow component (exact state-space sampling) with standard
    deviation ``trend_sd`` and length scale ``trend_length_scale``; the
    observation noise is white, Gaussian or Student-t with ``noise_df``
    degrees of freedom scaled to standard deviation ``noise_sd``.

    Args:
        period: Orbital period ``P > 0`` (minutes).
        dt: Sampling interval ``> 0`` (minutes).
        amplitudes: Harmonic amplitudes ``a_1, a_2, ...``.
        trend_sd: Standard deviation of the slow component (``>= 0``).
        trend_length_scale: Matern-3/2 length scale (minutes, ``> 0``).
        noise_sd: Observation noise standard deviation ``> 0``.
        noise: ``"gaussian"`` or ``"student_t"``.
        noise_df: Degrees of freedom ``> 2`` for ``"student_t"`` noise.

    Raises:
        ValueError: On an invalid value.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006]. A synthetic generator, not vehicle data.

    """

    period: float = 96.0
    dt: float = 4.0
    amplitudes: tuple[float, ...] = (5.0, 1.5)
    trend_sd: float = 1.0
    trend_length_scale: float = 300.0
    noise_sd: float = 0.3
    noise: str = "gaussian"
    noise_df: float = 3.0

    def __post_init__(self) -> None:
        """Validate the specification.

        Raises:
            ValueError: If a value is out of range.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        for name in ("period", "dt", "trend_length_scale", "noise_sd"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if not (math.isfinite(self.trend_sd) and self.trend_sd >= 0.0):
            raise ValueError("trend_sd must be finite and non-negative")
        if self.noise not in ("gaussian", "student_t"):
            raise ValueError("noise must be 'gaussian' or 'student_t'")
        if self.noise == "student_t" and not (
            math.isfinite(self.noise_df) and self.noise_df > 2.0
        ):
            raise ValueError("noise_df must be finite and exceed 2")
        if any(not math.isfinite(a) for a in self.amplitudes):
            raise ValueError("amplitudes must be finite")


@dataclass(frozen=True, slots=True)
class SyntheticChannel:
    """A simulated channel and its noise-free signal.

    Attributes:
        times: Sample times (minutes).
        values: Observations (signal plus noise).
        signal: Noise-free signal (harmonics plus slow component).
        spec: The generating :class:`ChannelSpec`.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006].

    """

    times: Vector
    values: Vector
    signal: Vector
    spec: ChannelSpec


def simulate_telemetry(
    count: int,
    spec: ChannelSpec | None = None,
    *,
    seed: int | RandomStream | None = None,
    start: float = 0.0,
) -> SyntheticChannel:
    """Simulate a synthetic nominal telemetry channel.

    The slow component is a Matern-3/2 process sampled exactly through its
    state-space form with ``lambda = sqrt(3) / ell``, transition ``A =
    e^{-lambda dt} [[1 + lambda dt, dt], [-lambda^2 dt, 1 - lambda dt]]`` and
    process covariance ``P_inf - A P_inf A'`` [EQ-SUR-GPANOM-008].

    Args:
        count: Number of samples (at least one).
        spec: Channel parameters (defaults to :class:`ChannelSpec`).
        seed: Integer seed or stream.
        start: Time of the first sample (minutes).

    Returns:
        The simulated :class:`SyntheticChannel`.

    Raises:
        ValueError: If ``count < 1`` or ``start`` is not finite.

    Equation:
        [EQ-SUR-GPANOM-008].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4],
        [REF-SARKKA-ALVAREZ-LAWRENCE-2019].

    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not math.isfinite(start):
        raise ValueError("start must be finite")
    chan = spec if spec is not None else ChannelSpec()
    rng = resolve_stream(seed).child("gpanom-channel").generator()
    times = tuple(start + i * chan.dt for i in range(count))
    phases = [rng.uniform(0.0, 2.0 * math.pi) for _ in chan.amplitudes]
    harmonic = [
        math.fsum(
            a * math.sin(2.0 * math.pi * (j + 1) * t / chan.period + phases[j])
            for j, a in enumerate(chan.amplitudes)
        )
        for t in times
    ]
    slow = [0.0] * count
    if chan.trend_sd > 0.0:
        lam = math.sqrt(3.0) / chan.trend_length_scale
        e = math.exp(-lam * chan.dt)
        a11, a12 = e * (1.0 + lam * chan.dt), e * chan.dt
        a21, a22 = -e * lam**2 * chan.dt, e * (1.0 - lam * chan.dt)
        p1, p2 = chan.trend_sd**2, lam**2 * chan.trend_sd**2
        q11 = p1 - (a11 * a11 * p1 + a12 * a12 * p2)
        q12 = -(a11 * a21 * p1 + a12 * a22 * p2)
        q22 = p2 - (a21 * a21 * p1 + a22 * a22 * p2)
        l11 = math.sqrt(q11)
        l21 = q12 / l11
        l22 = math.sqrt(q22 - l21 * l21)
        f = rng.gauss(0.0, chan.trend_sd)
        d = rng.gauss(0.0, math.sqrt(p2))
        for i in range(count):
            slow[i] = f
            u, v = rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)
            f, d = a11 * f + a12 * d + l11 * u, a21 * f + a22 * d + l21 * u + l22 * v
    noise: list[float] = []
    for _ in range(count):
        if chan.noise == "gaussian":
            noise.append(chan.noise_sd * rng.gauss(0.0, 1.0))
        else:
            nu = chan.noise_df
            draw = rng.gauss(0.0, 1.0) / math.sqrt(rng.gammavariate(nu / 2.0, 2.0) / nu)
            noise.append(chan.noise_sd * draw / math.sqrt(nu / (nu - 2.0)))
    signal = tuple(h + s for h, s in zip(harmonic, slow, strict=True))
    values = tuple(s + n for s, n in zip(signal, noise, strict=True))
    return SyntheticChannel(times, values, signal, chan)


def contaminate_values(
    values: Sequence[float],
    fraction: float,
    magnitude: float,
    *,
    seed: int | RandomStream | None = None,
) -> Vector:
    """Add ``+/- magnitude`` outliers to a random fraction of samples.

    Args:
        values: Series to contaminate.
        fraction: Share of samples changed, in ``[0, 1)``.
        magnitude: Absolute size of every outlier (data units, ``> 0``).
        seed: Integer seed or stream.

    Returns:
        The contaminated series (same length).

    Raises:
        ValueError: On an invalid ``fraction`` or ``magnitude``.

    References:
        [REF-LANGE-LITTLE-TAYLOR-1989]. Test-data helper; no equation.

    """
    y = as_vector(values, name="values")
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must lie in [0, 1)")
    if not (math.isfinite(magnitude) and magnitude > 0.0):
        raise ValueError("magnitude must be finite and positive")
    rng = resolve_stream(seed).child("gpanom-contaminate").generator()
    count = round(fraction * len(y))
    chosen = set(rng.sample(range(len(y)), count))
    return tuple(
        v + (magnitude if rng.random() < 0.5 else -magnitude) if i in chosen else v
        for i, v in enumerate(y)
    )


# ---------------------------------------------------------------------------
# nominal model
# ---------------------------------------------------------------------------


def orbital_kernel(
    period: float,
    *,
    trend: str = "matern32",
    trend_variance: float = 1.0,
    trend_length_scale: float = 100.0,
    periodic_variance: float = 1.0,
    periodic_length_scale: float = 1.0,
    noise_variance: float = 0.01,
) -> Kernel:
    """Return ``k_trend + k_periodic + white`` for an orbital channel.

    ``k(t, t') = sigma_t^2 phi(|t - t'| / ell_t) + sigma_p^2 exp(-2 sin^2(pi
    |t - t'| / P) / ell_p^2) + sigma_n^2 [t = t']`` [EQ-SUR-GPANOM-001].

    Args:
        period: Orbital period ``P`` (input units).
        trend: ``"matern32"``, ``"matern52"`` or ``"squared_exponential"``.
        trend_variance: ``sigma_t^2``.
        trend_length_scale: ``ell_t``.
        periodic_variance: ``sigma_p^2``.
        periodic_length_scale: ``ell_p`` of the exp-sine-squared kernel.
        noise_variance: White variance ``sigma_n^2``.

    Returns:
        A :class:`Kernel` with the layout ``Sum(Sum(trend, periodic), white)``.

    Raises:
        ValueError: If a parameter is not positive or ``trend`` is unknown.

    Equation:
        [EQ-SUR-GPANOM-001].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

    """
    return SumKernel(
        SumKernel(
            StationaryKernel(trend, trend_variance, (trend_length_scale,)),
            PeriodicKernel(periodic_variance, periodic_length_scale, period),
        ),
        WhiteKernel(noise_variance),
    )


@dataclass(frozen=True, slots=True)
class NominalModel:
    """A GP model of nominal channel behaviour with fixed hyperparameters.

    Attributes:
        kernel: Noise-free kernel ``Sum(trend, periodic)``.
        noise_variance: Observation noise variance ``sigma_n^2`` (data units
            squared). For a Student-t model this is the robust scale squared.
        output_mean: Constant prior mean subtracted from the data.
        nu: Student-t degrees of freedom of the observation model, ``inf`` for
            the Gaussian model.
        period: Fitted orbital period (input units).
        times: Training times.
        values: Training observations.
        weights: EM weights per training sample (all ones when Gaussian);
            small weights flag samples treated as outliers.
        log_marginal_likelihood: Gaussian log evidence of the fit, ``nan`` for
            the robust model.
        converged: Optimizer convergence of the final hyperparameter fit.
        stop_reason: Optimizer message of the final fit.

    Equation:
        [EQ-SUR-GPANOM-001], [EQ-SUR-GPANOM-006].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006], [REF-LANGE-LITTLE-TAYLOR-1989].

    """

    kernel: Kernel
    noise_variance: float
    output_mean: float
    nu: float
    period: float
    times: Vector
    values: Vector
    weights: Vector
    log_marginal_likelihood: float
    converged: bool
    stop_reason: str


def _replace_period(kernel: Kernel, period: float) -> Kernel:
    """Return the orbital kernel with its periodic period replaced.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

    """
    core, white, _ = _split_fitted(kernel)
    if not (isinstance(core, SumKernel) and isinstance(core.right, PeriodicKernel)):
        raise RuntimeError("unexpected kernel layout")  # pragma: no cover
    swapped = replace(core.right, period=period)
    return SumKernel(SumKernel(core.left, swapped), WhiteKernel(white))


def _log_bound(value: float) -> float:
    """Return ``log(value)`` clipped into the optimizer box ``|theta| < 15``.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

    """
    return min(max(math.log(value), -14.9), 14.9)


def _split_fitted(kernel: Kernel) -> tuple[Kernel, float, float]:
    """Return ``(trend + periodic, white variance, period)`` of a fitted kernel.

    Raises:
        RuntimeError: If the kernel does not have the orbital layout.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006].

    """
    if (
        isinstance(kernel, SumKernel)
        and isinstance(kernel.left, SumKernel)
        and isinstance(kernel.left.right, PeriodicKernel)
        and isinstance(kernel.right, WhiteKernel)
    ):
        return kernel.left, kernel.right.variance, kernel.left.right.period
    raise RuntimeError("unexpected kernel layout")  # pragma: no cover


def fit_nominal_model(
    times: Sequence[float],
    values: Sequence[float],
    *,
    period: float,
    period_bounds: tuple[float, float] | None = None,
    trend: str = "matern32",
    robust: bool = False,
    nu: float = 4.0,
    em_iterations: int = 8,
    start_from: NominalModel | None = None,
    restarts: int = 2,
    nugget: float = 1e-8,
    max_iterations: int = 100,
    seed: int | RandomStream | None = None,
) -> NominalModel:
    """Fit the nominal-behaviour GP (orbital periodic plus trend plus noise).

    Gaussian fit: type-II maximum likelihood of ``Sum(trend, periodic, white)``
    with :func:`probstat.surrogates.gp_advanced.fit_kernel_gp` inside explicit
    log bounds. The exp-sine-squared likelihood is multimodal in the period, so
    the period is confined to ``period_bounds`` and first scanned on a grid of
    24 values (marginal likelihood at the starting amplitudes); the best grid
    value starts the optimizer. On 8 synthetic seeds this recovered the period
    to within 0.5%; it is a heuristic, not a global-optimality guarantee.

    Robust fit (``robust=True``): the observation noise is Student-t with
    ``nu`` degrees of freedom, written as a Gaussian scale mixture. EM
    alternates the E-step ``w_i = (nu + 1) / (nu + e_i / sigma^2)`` with
    ``e_i = (y_i - m_i)^2 + v_i`` (posterior mean and latent variance at the
    training point) and the M-step ``sigma^2 = mean(w_i e_i)``, the GP being
    refitted with per-sample noise ``sigma^2 / w_i`` [EQ-SUR-GPANOM-006]. After
    ``em_iterations`` steps with fixed kernel parameters the kernel is
    re-optimized once with the weights frozen, followed by two more EM steps.
    The data are median-centered before fitting.

    Args:
        times: Strictly increasing training times spanning at least two
            periods.
        values: Nominal training observations (not constant).
        period: Nominal orbital period (input units, ``> 0``); the scan and
            the optimizer move the fitted period inside ``period_bounds``.
        period_bounds: ``(low, high)`` for the fitted period; default
            ``(0.7 period, 1.3 period)``.
        trend: Trend kernel name (``"matern32"``, ``"matern52"``,
            ``"squared_exponential"``).
        robust: Use the Student-t observation model.
        nu: Student-t degrees of freedom ``> 2`` (robust fit).
        em_iterations: EM steps before the weighted re-optimization.
        start_from: Robust fit only: a Gaussian :class:`NominalModel` of the
            same data whose kernel and noise start the EM, skipping the first
            likelihood maximization (a warm start that saves one fit).
        restarts: Optimizer starts of each likelihood maximization.
        nugget: Explicit tiny diagonal variance (data units squared).
        max_iterations: BFGS iteration limit.
        seed: Integer seed or stream.

    Returns:
        The :class:`NominalModel`.

    Raises:
        ValueError: On invalid data, bounds or a non-positive-definite matrix.

    Equation:
        [EQ-SUR-GPANOM-001], [EQ-SUR-GPANOM-006].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4],
        [REF-LANGE-LITTLE-TAYLOR-1989], [REF-DEMPSTER-LAIRD-RUBIN-1977].

    """
    t, y = _check_series(times, values, minimum=8)
    if not (math.isfinite(period) and period > 0.0):
        raise ValueError("period must be finite and positive")
    if t[-1] - t[0] < 2.0 * period:
        raise ValueError("the training window must span at least two periods")
    if robust and not (math.isfinite(nu) and nu > 2.0):
        raise ValueError("nu must be finite and exceed 2")
    if not isinstance(em_iterations, int) or em_iterations < 1:
        raise ValueError("em_iterations must be a positive integer")
    low, high = (
        period_bounds if period_bounds is not None else (0.7 * period, 1.3 * period)
    )
    if not 0.0 < low < period < high:
        raise ValueError("period_bounds must satisfy 0 < low < period < high")
    if max(y) == min(y):
        raise ValueError("training values are constant")
    count = len(y)
    mean = math.fsum(y) / count
    var = math.fsum((v - mean) ** 2 for v in y) / (count - 1)
    spacing = sorted(b - a for a, b in zip(t, t[1:], strict=False))[count // 2]
    span = t[-1] - t[0]
    centre = sorted(y)[count // 2] if robust else 0.0
    data = tuple(v - centre for v in y)
    sf_lo, sf_hi = _log_bound(var * 1e-4), _log_bound(var * 1e2)
    ell_lo, ell_hi = _log_bound(2.0 * spacing), _log_bound(5.0 * span)
    bounds_core = [
        (sf_lo, sf_hi),
        (ell_lo, ell_hi),
        (sf_lo, sf_hi),
        (math.log(0.3), math.log(5.0)),
        (math.log(low), math.log(high)),
    ]
    start = orbital_kernel(
        period,
        trend=trend,
        trend_variance=0.3 * var,
        trend_length_scale=span / 3.0,
        periodic_variance=0.7 * var,
        periodic_length_scale=1.0,
        noise_variance=0.01 * var,
    )
    points = tuple((v,) for v in t)
    rng = resolve_stream(seed)
    if start_from is None:
        # The exp-sine-squared likelihood is multimodal in the period: scan the
        # marginal likelihood on a grid at the starting amplitudes, then refine.
        best_period, best_value = period, -math.inf
        for index in range(1, _PERIOD_GRID + 1):
            candidate = low + (high - low) * index / (_PERIOD_GRID + 1)
            try:
                value = fit_kernel_gp(
                    points,
                    data,
                    _replace_period(start, candidate),
                    nugget=nugget,
                    optimize=False,
                ).log_marginal_likelihood
            except ValueError:
                continue
            if value > best_value:
                best_period, best_value = candidate, value
        start = _replace_period(start, best_period)
    if start_from is not None:
        if not robust:
            raise ValueError("start_from is only used with robust=True")
        if not math.isinf(start_from.nu):
            raise ValueError("start_from must be a Gaussian model")
        core = start_from.kernel
        white = max(start_from.noise_variance - nugget, 1e-12 * var)
    else:
        gaussian = fit_kernel_gp(
            points,
            data,
            start,
            nugget=nugget,
            restarts=restarts,
            seed=rng,
            max_iterations=max_iterations,
            log_bounds=[*bounds_core, (_log_bound(var * 1e-7), _log_bound(var))],
        )
        core, white, fitted_period = _split_fitted(gaussian.kernel)
    if not robust:
        return NominalModel(
            core,
            white + nugget,
            gaussian.output_mean,
            math.inf,
            fitted_period,
            t,
            y,
            (1.0,) * count,
            gaussian.log_marginal_likelihood,
            gaussian.converged,
            gaussian.stop_reason,
        )

    def em_steps(
        kernel: Kernel, weights: list[float], sigma2: float, steps: int
    ) -> tuple[list[float], float]:
        """Run EM steps with fixed kernel parameters.

        Equation:
            [EQ-SUR-GPANOM-006].

        References:
            [REF-LANGE-LITTLE-TAYLOR-1989], [REF-DEMPSTER-LAIRD-RUBIN-1977].

        """
        for _ in range(steps):
            gp = fit_kernel_gp(
                points,
                data,
                kernel,
                nugget=nugget,
                noise_variances=[sigma2 / w for w in weights],
                optimize=False,
            )
            expected = []
            for point, observed in zip(points, data, strict=True):
                m, v = gp.predict_with_variance(point)
                expected.append((observed - m) ** 2 + v)
            weights = [(nu + 1.0) / (nu + e / sigma2) for e in expected]
            sigma2 = math.fsum(w * e for w, e in zip(weights, expected, strict=True))
            sigma2 /= count
        return weights, sigma2

    weights, sigma2 = em_steps(core, [1.0] * count, white, em_iterations)
    refit = fit_kernel_gp(
        points,
        data,
        core,
        nugget=nugget,
        noise_variances=[sigma2 / w for w in weights],
        restarts=1,
        seed=rng,
        max_iterations=max_iterations,
        log_bounds=bounds_core,
    )
    weights, sigma2 = em_steps(refit.kernel, weights, sigma2, 2)
    final = fit_kernel_gp(
        points,
        data,
        refit.kernel,
        nugget=nugget,
        noise_variances=[sigma2 / w for w in weights],
        optimize=False,
    )
    periodic = refit.kernel.right if isinstance(refit.kernel, SumKernel) else None
    if not isinstance(periodic, PeriodicKernel):
        raise RuntimeError("unexpected kernel layout")  # pragma: no cover
    return NominalModel(
        refit.kernel,
        sigma2 + nugget,
        centre + final.output_mean,
        float(nu),
        periodic.period,
        t,
        y,
        tuple(weights),
        math.nan,
        refit.converged,
        refit.stop_reason,
    )


# ---------------------------------------------------------------------------
# sliding-window GP
# ---------------------------------------------------------------------------


class SlidingWindowGP:
    """GP over the last ``w`` samples with ``O(w^2)`` cost per update.

    The window keeps the Cholesky factor ``L`` of ``K + diag(noise)``. A new
    sample appends a row (one forward solve); when the window is full the
    oldest sample is dropped by deleting its row and column and a rank-one
    Cholesky *update* with the removed column, ``L_22 L_22' + l_21 l_21' =
    chol(K_22)`` [EQ-SUR-GPANOM-007]. The predictive law equals the batch GP on
    the window with the same prior mean [EQ-SUR-GPANOM-001], which
    :func:`batch_window_predict` recomputes densely.

    Args:
        kernel: Noise-free kernel on scalar times.
        output_mean: Constant prior mean.
        window: Maximum number of samples kept (at least two).

    Raises:
        ValueError: If ``window < 2``.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1], [REF-GOLUB-VANLOAN-2013].

    """

    def __init__(self, kernel: Kernel, output_mean: float, window: int) -> None:
        """Create an empty window.

        Raises:
            ValueError: If ``window < 2`` or ``output_mean`` is not finite.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        if isinstance(window, bool) or not isinstance(window, int) or window < 2:
            raise ValueError("window must be an integer of at least 2")
        if not math.isfinite(output_mean):
            raise ValueError("output_mean must be finite")
        self._kernel = kernel
        self._mean = float(output_mean)
        self._window = window
        self._times: list[float] = []
        self._values: list[float] = []
        self._noise: list[float] = []
        self._rows: list[list[float]] = []
        self._pending: tuple[float, list[float]] | None = None

    @property
    def size(self) -> int:
        """Return the number of samples currently in the window.

        References:
            No equation is evaluated [REF-RASMUSSEN-WILLIAMS-2006].

        """
        return len(self._times)

    @property
    def times(self) -> Vector:
        """Return the window times.

        References:
            No equation is evaluated [REF-RASMUSSEN-WILLIAMS-2006].

        """
        return tuple(self._times)

    @property
    def values(self) -> Vector:
        """Return the window observations.

        References:
            No equation is evaluated [REF-RASMUSSEN-WILLIAMS-2006].

        """
        return tuple(self._values)

    def _cross(self, t: float) -> list[float]:
        """Return ``k(t, t_i)`` over the window.

        References:
            [EQ-SUR-GPANOM-001], [REF-RASMUSSEN-WILLIAMS-2006].

        """
        return [self._kernel.evaluate((t,), (s,)) for s in self._times]

    def predict(self, t: float) -> tuple[float, float]:
        """Return the predictive mean and latent variance at time ``t``.

        ``mu = m + k_*' (K + N)^-1 (y - m)`` and ``sigma_f^2 = k(t, t) -
        k_*' (K + N)^-1 k_*`` computed from ``L`` by two forward solves
        [EQ-SUR-GPANOM-001].

        Args:
            t: Query time.

        Returns:
            ``(mean, latent_variance)`` in data units; the variance is clipped
            at zero only for rounding noise below zero.

        Raises:
            ValueError: If ``t`` is not finite.

        Equation:
            [EQ-SUR-GPANOM-001].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

        """
        if not math.isfinite(t):
            raise ValueError("t must be finite")
        prior = self._kernel.evaluate((t,), (t,))
        if not self._times:
            return self._mean, prior
        cross = self._cross(t)
        self._pending = (t, cross)
        v = _forward(self._rows, cross)
        z = _forward(self._rows, [y - self._mean for y in self._values])
        return self._mean + _dot(v, z), max(prior - _dot(v, v), 0.0)

    def update(self, t: float, y: float, noise_variance: float) -> None:
        """Append the sample ``(t, y)``, dropping the oldest if the window is full.

        Args:
            t: Time, greater than the last window time.
            y: Observation.
            noise_variance: Noise variance of this sample (``> 0``).

        Raises:
            ValueError: On unordered time, non-finite input or a window matrix
                that is not positive definite (the state is then unchanged).

        Equation:
            [EQ-SUR-GPANOM-007].

        References:
            [REF-GOLUB-VANLOAN-2013], [REF-RASMUSSEN-WILLIAMS-2006].

        """
        if not (math.isfinite(t) and math.isfinite(y)):
            raise ValueError("t and y must be finite")
        if not (math.isfinite(noise_variance) and noise_variance > 0.0):
            raise ValueError("noise_variance must be finite and positive")
        if self._times and t <= self._times[-1]:
            raise ValueError("times must be strictly increasing")
        rows = self._rows
        times, values, noise = self._times, self._values, self._noise
        pending = self._pending
        self._pending = None
        cross = pending[1] if pending is not None and pending[0] == t else None
        if len(times) == self._window:
            column = [row[0] for row in rows[1:]]
            reduced = [row[1:] for row in rows[1:]]
            _chol_update(reduced, column)
            rows = reduced
            times, values, noise = times[1:], values[1:], noise[1:]
            cross = cross[1:] if cross is not None else None
        if cross is None:
            cross = [self._kernel.evaluate((t,), (s,)) for s in times]
        v = _forward(rows, cross)
        diagonal = self._kernel.evaluate((t,), (t,)) + noise_variance - _dot(v, v)
        if not diagonal > 0.0:
            raise ValueError("window matrix is not positive definite")
        self._rows = [*rows, [*v, math.sqrt(diagonal)]]
        self._times = [*times, t]
        self._values = [*values, y]
        self._noise = [*noise, noise_variance]


def batch_window_predict(
    kernel: Kernel,
    output_mean: float,
    times: Sequence[float],
    values: Sequence[float],
    noise_variances: Sequence[float],
    t: float,
) -> tuple[float, float]:
    """Return the batch GP predictive mean and latent variance on a window.

    Dense reference for :class:`SlidingWindowGP`: ``mu = m + k_*' A^-1 (y -
    m)`` and ``sigma_f^2 = k(t, t) - k_*' A^-1 k_*`` with ``A = K +
    diag(noise)`` solved by ``probstat.linalg.solve_spd`` [EQ-SUR-GPANOM-001].

    Args:
        kernel: Noise-free kernel on scalar times.
        output_mean: Constant prior mean.
        times: Window times (strictly increasing).
        values: Window observations.
        noise_variances: Per-sample noise variances (``> 0``).
        t: Query time.

    Returns:
        ``(mean, latent_variance)`` in data units.

    Raises:
        ValueError: On inconsistent lengths or a non-positive-definite matrix.

    Equation:
        [EQ-SUR-GPANOM-001].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

    """
    ts, ys = _check_series(times, values, minimum=1)
    noise = as_vector(noise_variances, name="noise_variances")
    if len(noise) != len(ts) or any(v <= 0.0 for v in noise):
        raise ValueError("noise_variances must be positive, one per sample")
    matrix = tuple(
        tuple(
            kernel.evaluate((a,), (b,)) + (noise[i] if i == j else 0.0)
            for j, b in enumerate(ts)
        )
        for i, a in enumerate(ts)
    )
    cholesky(matrix)
    cross = tuple(kernel.evaluate((t,), (s,)) for s in ts)
    alpha = solve_spd(matrix, tuple(y - output_mean for y in ys))
    gain = solve_spd(matrix, cross)
    return (
        output_mean + _dot(cross, alpha),
        max(kernel.evaluate((t,), (t,)) - _dot(cross, gain), 0.0),
    )


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoreSeries:
    """One-step-ahead scores of a scored series.

    Attributes:
        times: Scored times.
        values: Observations.
        mean: Predictive mean ``mu_t``.
        variance: Predictive observation variance ``s_t^2`` (latent variance
            plus noise); for a Student-t model ``s_t`` is the *scale*.
        z: Standardized residuals ``(y_t - mu_t) / s_t``.
        nlpd: Negative log predictive density of ``y_t``.
        nu: Degrees of freedom of the predictive law (``inf`` if Gaussian).

    Equation:
        [EQ-SUR-GPANOM-001], [EQ-SUR-GPANOM-002].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006].

    """

    times: Vector
    values: Vector
    mean: Vector
    variance: Vector
    z: Vector
    nlpd: Vector
    nu: float

    @property
    def abs_z(self) -> Vector:
        """Return ``|z_t|``.

        References:
            [EQ-SUR-GPANOM-001], [REF-RASMUSSEN-WILLIAMS-2006].

        """
        return tuple(abs(v) for v in self.z)


def _nlpd(z: float, variance: float, nu: float) -> float:
    """Return the negative log predictive density at standardized residual ``z``.

    Equation:
        [EQ-SUR-GPANOM-002].

    References:
        [REF-SHAH-WILSON-GHAHRAMANI-2014], [REF-RASMUSSEN-WILLIAMS-2006].

    """
    if math.isinf(nu):
        return 0.5 * math.log(2.0 * math.pi * variance) + 0.5 * z * z
    return -(
        math.lgamma(0.5 * (nu + 1.0))
        - math.lgamma(0.5 * nu)
        - 0.5 * math.log(nu * math.pi)
        - 0.5 * math.log(variance)
        - 0.5 * (nu + 1.0) * math.log1p(z * z / nu)
    )


def sliding_window_scores(
    model: NominalModel,
    times: Sequence[float],
    values: Sequence[float],
    *,
    window: int = 36,
    history: tuple[Sequence[float], Sequence[float]] | None = None,
    gate: float | None = None,
    robust_updates: bool = False,
) -> ScoreSeries:
    """Score a series one step ahead with a sliding-window GP.

    The window starts from the last ``window`` samples of ``history`` (default
    the training data of ``model``). For each new sample the predictive mean
    and observation variance ``s^2 = sigma_f^2 + sigma_n^2`` give ``z`` and the
    NLPD [EQ-SUR-GPANOM-001], [EQ-SUR-GPANOM-002]; then the sample enters the
    window unless ``|z| > gate`` (the window is then left as it was, so a step
    fault is not absorbed). With ``robust_updates`` the sample enters with the
    inflated noise ``sigma_n^2 / w``, ``w = min(1, (nu + 1) / (nu + z^2))``, a
    one-step Student-t down-weighting (requires a Student-t model).

    Args:
        model: Fitted :class:`NominalModel`.
        times: Strictly increasing times after the history.
        values: Observations.
        window: Window length (at least two).
        history: Optional ``(times, values)`` warm-up before ``times``.
        gate: Optional ``|z|`` above which a sample is kept out of the window.
        robust_updates: Down-weight outliers when they enter the window.

    Returns:
        The :class:`ScoreSeries`.

    Raises:
        ValueError: On invalid input, a non-positive ``gate``,
            ``robust_updates`` with a Gaussian model, or times not after the
            history.

    Equation:
        [EQ-SUR-GPANOM-001], [EQ-SUR-GPANOM-002], [EQ-SUR-GPANOM-007].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006], [REF-SHAH-WILSON-GHAHRAMANI-2014].

    """
    t, y = _check_series(times, values, minimum=1)
    if gate is not None and not (math.isfinite(gate) and gate > 0.0):
        raise ValueError("gate must be finite and positive")
    if robust_updates and math.isinf(model.nu):
        raise ValueError("robust_updates needs a Student-t model (finite nu)")
    ht, hy = (
        _check_series(*history, minimum=1)
        if history is not None
        else (model.times, model.values)
    )
    if t[0] <= ht[-1]:
        raise ValueError("times must come after the history")
    gp = SlidingWindowGP(model.kernel, model.output_mean, window)
    for a, b in zip(ht[-window:], hy[-window:], strict=True):
        gp.update(a, b, model.noise_variance)
    mean: list[float] = []
    variance: list[float] = []
    z: list[float] = []
    nlpd: list[float] = []
    for a, b in zip(t, y, strict=True):
        m, latent = gp.predict(a)
        s2 = latent + model.noise_variance
        score = (b - m) / math.sqrt(s2)
        mean.append(m)
        variance.append(s2)
        z.append(score)
        nlpd.append(_nlpd(score, s2, model.nu))
        if gate is not None and abs(score) > gate:
            continue
        noise = model.noise_variance
        if robust_updates:
            noise /= min(1.0, (model.nu + 1.0) / (model.nu + score * score))
        gp.update(a, b, noise)
    return ScoreSeries(
        t, y, tuple(mean), tuple(variance), tuple(z), tuple(nlpd), model.nu
    )


def rolling_zscore(
    values: Sequence[float],
    window: int,
    history: Sequence[float] = (),
) -> Vector:
    """Return the rolling mean/standard-deviation baseline score.

    ``z_t = (y_t - mean(y_{t-w:t-1})) / sd(y_{t-w:t-1})`` with the sample
    standard deviation of the previous ``w`` observations (``history`` supplies
    the first ones) [EQ-SUR-GPANOM-008].

    Args:
        values: Series to score.
        window: Number of previous samples (at least three).
        history: Observations preceding ``values``; with fewer than ``window``
            of them in total at some step the call raises.

    Returns:
        One score per value.

    Raises:
        ValueError: If fewer than ``window`` previous samples exist or the
            window has zero spread.

    Equation:
        [EQ-SUR-GPANOM-008].

    References:
        [REF-PAGE-1954]. Textbook baseline; no sharper source is claimed.

    """
    y = as_vector(values, name="values")
    past = list(as_vector(history, name="history"))
    if isinstance(window, bool) or not isinstance(window, int) or window < 3:
        raise ValueError("window must be an integer of at least 3")
    out: list[float] = []
    for value in y:
        if len(past) < window:
            raise ValueError("not enough history for the rolling window")
        recent = past[-window:]
        mean = math.fsum(recent) / window
        sd = math.sqrt(math.fsum((v - mean) ** 2 for v in recent) / (window - 1))
        if sd == 0.0:
            raise ValueError("rolling window has zero spread")
        out.append((value - mean) / sd)
        past.append(value)
    return tuple(out)


# ---------------------------------------------------------------------------
# CUSUM and thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CusumResult:
    """Two-sided CUSUM statistic and its alarm indices.

    Attributes:
        statistic: ``max(S+_t, S-_t)`` before any reset, per sample.
        alarms: Indices ``t`` where the statistic exceeded ``h`` (the CUSUM was
            reset to zero after each); empty when no ``h`` was given.

    Equation:
        [EQ-SUR-GPANOM-003].

    References:
        [REF-PAGE-1954].

    """

    statistic: Vector
    alarms: tuple[int, ...]


def cusum(z: Sequence[float], *, k: float = 0.5, h: float | None = None) -> CusumResult:
    """Return the two-sided CUSUM of standardized residuals.

    ``S+_t = max(0, S+_{t-1} + z_t - k)``, ``S-_t = max(0, S-_{t-1} - z_t -
    k)``; an alarm is raised when ``max(S+, S-) > h`` and both sums restart
    from zero [EQ-SUR-GPANOM-003].

    Args:
        z: Standardized residuals.
        k: Reference value (allowance, in units of ``z``); ``k = delta / 2``
            is tuned to a shift ``delta``.
        h: Decision threshold, or ``None`` for the raw statistic only.

    Returns:
        The :class:`CusumResult`.

    Raises:
        ValueError: If ``k`` is negative or ``h`` not positive.

    Equation:
        [EQ-SUR-GPANOM-003].

    References:
        [REF-PAGE-1954].

    """
    scores = as_vector(z, name="z")
    if not (math.isfinite(k) and k >= 0.0):
        raise ValueError("k must be finite and non-negative")
    if h is not None and not (math.isfinite(h) and h > 0.0):
        raise ValueError("h must be finite and positive")
    up = down = 0.0
    stat: list[float] = []
    alarms: list[int] = []
    for index, score in enumerate(scores):
        up = max(0.0, up + score - k)
        down = max(0.0, down - score - k)
        level = max(up, down)
        stat.append(level)
        if h is not None and level > h:
            alarms.append(index)
            up = down = 0.0
    return CusumResult(tuple(stat), tuple(alarms))


def calibrate_cusum_threshold(
    z: Sequence[float], alarm_rate: float, *, k: float = 0.5
) -> float:
    """Return the smallest tested ``h`` whose CUSUM alarm rate on ``z`` is ``<= rate``.

    Bisection on ``h`` of the empirical rate (alarms per sample, CUSUM reset
    after each alarm) on a nominal ``z`` series. This matches an alarm rate on
    the given data; it is **not** a distribution-free guarantee, because the
    CUSUM is path dependent [EQ-SUR-GPANOM-003].

    Args:
        z: Nominal standardized residuals.
        alarm_rate: Target alarms per sample in ``(0, 1)``.
        k: CUSUM reference value.

    Returns:
        A threshold ``h`` with measured rate at most ``alarm_rate``.

    Raises:
        ValueError: On invalid arguments or fewer than ``1 / alarm_rate``
            samples (the rate could not be resolved).

    Equation:
        [EQ-SUR-GPANOM-003].

    References:
        [REF-PAGE-1954].

    """
    scores = as_vector(z, name="z")
    rate = _check_alpha(alarm_rate)
    if len(scores) < 1.0 / rate:
        raise ValueError("too few samples to resolve the alarm rate")
    high = max(cusum(scores, k=k).statistic) + 1.0
    low = 0.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if len(cusum(scores, k=k, h=mid).alarms) / len(scores) <= rate:
            high = mid
        else:
            low = mid
    return high


def gaussian_threshold(alpha: float) -> float:
    """Return the two-sided Gaussian threshold for ``|z|`` at rate ``alpha``.

    ``tau = Phi^-1(1 - alpha / 2)`` so that ``P(|z| > tau) = alpha`` when
    ``z ~ N(0, 1)`` [EQ-SUR-GPANOM-005].

    Args:
        alpha: Per-sample false-alarm rate in ``(0, 1)``.

    Returns:
        The threshold.

    Raises:
        ValueError: If ``alpha`` is not in ``(0, 1)``.

    Equation:
        [EQ-SUR-GPANOM-005].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006].

    """
    return Normal().quantile(1.0 - 0.5 * _check_alpha(alpha))


def student_t_threshold(alpha: float, nu: float) -> float:
    """Return the two-sided Student-t threshold for ``|z|`` at rate ``alpha``.

    ``tau = t_nu^-1(1 - alpha / 2)``, valid when ``z`` is a standard Student-t
    variable with ``nu`` degrees of freedom (the *scale* convention of
    :class:`ScoreSeries` for a Student-t model) [EQ-SUR-GPANOM-005].

    Args:
        alpha: Per-sample false-alarm rate in ``(0, 1)``.
        nu: Degrees of freedom ``> 0``.

    Returns:
        The threshold.

    Raises:
        ValueError: If ``alpha`` or ``nu`` is invalid.

    Equation:
        [EQ-SUR-GPANOM-005].

    References:
        [REF-SHAH-WILSON-GHAHRAMANI-2014].

    """
    if not (math.isfinite(nu) and nu > 0.0):
        raise ValueError("nu must be finite and positive")
    return StudentT(nu).quantile(1.0 - 0.5 * _check_alpha(alpha))


def conformal_threshold(scores: Sequence[float], alpha: float) -> float:
    """Return the split-conformal threshold of a nonconformity score.

    With ``n`` calibration scores from held-out nominal data, the threshold is
    the ``ceil((1 - alpha)(n + 1))``-th smallest score; for exchangeable scores
    ``P(S_new > q) <= alpha`` marginally over the calibration draw, and for
    continuous scores ``>= alpha - 1/(n+1)`` [EQ-SUR-GPANOM-004]. Time-series
    scores are not exchangeable in general; the guarantee is then approximate.
    The library's ``probstat.conformal`` is not imported (it is a higher
    layer); the rule is re-implemented here with exact rational arithmetic.

    Args:
        scores: Calibration scores (larger is more anomalous).
        alpha: Target rate in ``(0, 1)``.

    Returns:
        The threshold; alarm when a new score is strictly larger.

    Raises:
        ValueError: If ``ceil((1 - alpha)(n + 1)) > n`` (too few calibration
            scores; ``+inf`` is never returned silently).

    Equation:
        [EQ-SUR-GPANOM-004].

    References:
        [REF-VOVK-GAMMERMAN-SHAFER-2005], [REF-LEI-ETAL-2018].

    """
    s = sorted(as_vector(scores, name="scores"))
    rate = Fraction(repr(_check_alpha(alpha)))
    rank = math.ceil((1 - rate) * (len(s) + 1))
    if rank > len(s):
        raise ValueError(
            f"need at least {math.ceil(1 / rate) - 1} calibration scores for "
            f"alpha={alpha}"
        )
    return s[rank - 1]


def false_alarm_rate(statistic: Sequence[float], threshold: float) -> float:
    """Return the share of samples with ``statistic > threshold``.

    Args:
        statistic: Alarm statistic on fresh nominal data.
        threshold: Alarm threshold.

    Returns:
        The realized per-sample false-alarm rate.

    Raises:
        ValueError: If ``statistic`` is empty or ``threshold`` is not a
            number.

    Equation:
        [EQ-SUR-GPANOM-004].

    References:
        [REF-LEI-ETAL-2018].

    """
    s = as_vector(statistic, name="statistic")
    if not s:
        raise ValueError("statistic must not be empty")
    if math.isnan(threshold):
        raise ValueError("threshold must not be NaN")
    return sum(1 for v in s if v > threshold) / len(s)


# ---------------------------------------------------------------------------
# faults and study
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FaultedSeries:
    """A series with an injected fault.

    Attributes:
        values: Faulted observations.
        kind: One of :data:`FAULT_KINDS`.
        size: Fault size in units of ``scale``.
        onset: Index of the first affected sample.
        affected: Flag per sample of whether the fault acts on it.

    References:
        [EQ-SUR-GPANOM-008].

    """

    values: Vector
    kind: str
    size: float
    onset: int
    affected: tuple[bool, ...]


def inject_fault(
    values: Sequence[float],
    kind: str,
    *,
    onset: int,
    size: float,
    scale: float,
    duration: int = 1,
    drift_length: int = 30,
    seed: int | RandomStream | None = None,
) -> FaultedSeries:
    """Inject a synthetic fault into a series.

    With ``a = size * scale`` and ``j = i - onset``: ``spike`` adds ``a`` for
    ``duration`` samples; ``step`` adds ``a`` from the onset on; ``drift`` adds
    ``a (j + 1) / drift_length`` from the onset on (linear ramp, reaching ``a``
    after ``drift_length`` samples and continuing); ``variance`` adds
    independent ``N(0, a^2)`` noise from the onset on [EQ-SUR-GPANOM-008].

    Args:
        values: Nominal series.
        kind: One of :data:`FAULT_KINDS`.
        onset: First affected index, ``0 <= onset < len(values)``.
        size: Fault size in units of ``scale`` (``> 0``).
        scale: Nominal observation standard deviation (data units, ``> 0``).
        duration: Spike length in samples.
        drift_length: Ramp length of the drift fault.
        seed: Integer seed or stream (used by ``variance``).

    Returns:
        The :class:`FaultedSeries`.

    Raises:
        ValueError: On an unknown kind or out-of-range argument.

    Equation:
        [EQ-SUR-GPANOM-008].

    References:
        [REF-PAGE-1954]. Synthetic fault models, not vehicle failure data.

    """
    y = as_vector(values, name="values")
    if kind not in FAULT_KINDS:
        raise ValueError(f"kind must be one of {FAULT_KINDS}")
    if isinstance(onset, bool) or not isinstance(onset, int):
        raise ValueError("onset must be an integer")
    if not 0 <= onset < len(y):
        raise ValueError("onset must index the series")
    if not (
        math.isfinite(size) and size > 0.0 and math.isfinite(scale) and scale > 0.0
    ):
        raise ValueError("size and scale must be finite and positive")
    if duration < 1 or drift_length < 1:
        raise ValueError("duration and drift_length must be at least 1")
    amplitude = size * scale
    rng = resolve_stream(seed).child("gpanom-fault", kind).generator()
    out: list[float] = []
    flags: list[bool] = []
    for i, v in enumerate(y):
        j = i - onset
        if j < 0:
            out.append(v)
            flags.append(False)
            continue
        if kind == "spike":
            hit = j < duration
            out.append(v + (amplitude if hit else 0.0))
            flags.append(hit)
            continue
        flags.append(True)
        if kind == "step":
            out.append(v + amplitude)
        elif kind == "drift":
            out.append(v + amplitude * (j + 1) / drift_length)
        else:
            out.append(v + rng.gauss(0.0, amplitude))
    return FaultedSeries(tuple(out), kind, float(size), onset, tuple(flags))


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """Detection performance of one method at one fault kind and size.

    Attributes:
        fault: Fault kind.
        size: Fault size in units of the nominal noise standard deviation.
        method: ``gp_abs_z``, ``gp_cusum``, ``baseline_abs_z`` or
            ``baseline_cusum``.
        detection_rate: Share of replicates alarmed inside the detection window.
        mean_delay: Mean alarm delay in samples over detected replicates
            (``nan`` if none was detected).
        pre_onset_alarm_rate: Alarms per sample before the onset, pooled.
        replicates: Number of replicates.

    References:
        [EQ-SUR-GPANOM-008].

    """

    fault: str
    size: float
    method: str
    detection_rate: float
    mean_delay: float
    pre_onset_alarm_rate: float
    replicates: int


@dataclass(frozen=True, slots=True)
class DetectionStudy:
    """Operating characteristics of the GP detector and the baseline.

    Attributes:
        points: All :class:`OperatingPoint` rows.
        alpha: Target per-sample false-alarm rate used for every threshold.
        thresholds: Thresholds by method (``|z|`` level or CUSUM ``h``).

    References:
        [EQ-SUR-GPANOM-004], [EQ-SUR-GPANOM-008].

    """

    points: tuple[OperatingPoint, ...]
    alpha: float
    thresholds: dict[str, float]

    def get(self, fault: str, size: float, method: str) -> OperatingPoint:
        """Return the row for ``(fault, size, method)``.

        Args:
            fault: Fault kind.
            size: Fault size.
            method: Method name.

        Returns:
            The matching :class:`OperatingPoint`.

        Raises:
            KeyError: If no such row exists.

        References:
            [EQ-SUR-GPANOM-008].

        """
        for point in self.points:
            if point.fault == fault and point.size == size and point.method == method:
                return point
        raise KeyError((fault, size, method))


def _first_in(indices: Sequence[int], start: int, stop: int) -> int | None:
    """Return the first index in ``[start, stop)`` or ``None``.

    References:
        [EQ-SUR-GPANOM-008].

    """
    return next((i for i in indices if start <= i < stop), None)


def detection_study(
    model: NominalModel,
    spec: ChannelSpec,
    *,
    alpha: float = 0.01,
    faults: Sequence[str] = FAULT_KINDS,
    sizes: Sequence[float] = (2.0, 4.0, 8.0),
    replicates: int = 6,
    calibration_series: int = 4,
    series_length: int = 90,
    onset: int = 40,
    window: int = 36,
    cusum_k: float = 0.5,
    drift_length: int = 30,
    horizon: int = 12,
    spike_duration: int = 1,
    gate: float | None = None,
    seed: int | RandomStream | None = None,
) -> DetectionStudy:
    """Measure detection rate and delay of the GP detector against a baseline.

    Thresholds of all four methods are set on the same nominal calibration
    series for the same per-sample rate ``alpha``: the split-conformal quantile
    of ``|z|`` [EQ-SUR-GPANOM-004] and the empirical CUSUM threshold
    [EQ-SUR-GPANOM-003] for both the GP score and the rolling z-score baseline.
    Each replicate is an independent nominal channel from ``spec`` whose first
    ``window`` samples are the warm-up; a fault of the given kind and size (in
    units of ``spec.noise_sd``) is injected at index ``onset`` of the scored
    part. ``model`` supplies only hyperparameters.

    Args:
        model: Fitted nominal model for the channel family of ``spec``.
        spec: Channel generator.
        alpha: Per-sample false-alarm target in ``(0, 1)``.
        faults: Fault kinds to study.
        sizes: Fault sizes in units of ``spec.noise_sd``.
        replicates: Independent faulted series per fault and size.
        calibration_series: Independent nominal series pooled for thresholds.
        series_length: Scored samples per series.
        onset: Fault onset within the scored samples.
        window: Sliding-window length and baseline window.
        cusum_k: CUSUM reference value.
        drift_length: Ramp length of the drift fault.
        horizon: A fault counts as detected when an alarm falls in
            ``[onset, onset + horizon)``; delay is measured from the onset.
            A chance alarm in the window is counted, at about ``horizon *
            alpha`` probability (see ``pre_onset_alarm_rate``).
        spike_duration: Spike length; the detection window of a spike is its
            duration, not ``horizon``.
        gate: Optional ``|z|`` gate of :func:`sliding_window_scores` for the GP
            score (samples above it stay out of the window).
        seed: Integer seed or stream.

    Returns:
        The :class:`DetectionStudy`.

    Raises:
        ValueError: On invalid arguments (including too few calibration scores
            for ``alpha``).

    Equation:
        [EQ-SUR-GPANOM-003], [EQ-SUR-GPANOM-004], [EQ-SUR-GPANOM-008].

    References:
        [REF-PAGE-1954], [REF-VOVK-GAMMERMAN-SHAFER-2005], [REF-LEI-ETAL-2018].

    """
    _check_alpha(alpha)
    if not 0 < onset < series_length:
        raise ValueError("onset must lie inside the scored series")
    if horizon < 1 or spike_duration < 1:
        raise ValueError("horizon and spike_duration must be at least 1")
    if replicates < 1 or calibration_series < 1:
        raise ValueError("replicates and calibration_series must be positive")
    for kind in faults:
        if kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault kind {kind!r}")
    stream = resolve_stream(seed)

    def scored(
        series_values: Sequence[float], series_times: Sequence[float]
    ) -> tuple[Vector, Vector]:
        """Return GP and baseline ``z`` for a series with a warm-up prefix.

        References:
            [EQ-SUR-GPANOM-001], [EQ-SUR-GPANOM-008].

        """
        gp = sliding_window_scores(
            model,
            series_times[window:],
            series_values[window:],
            window=window,
            history=(series_times[:window], series_values[:window]),
            gate=gate,
        )
        base = rolling_zscore(series_values[window:], window, series_values[:window])
        return gp.z, base

    gp_z: list[float] = []
    base_z: list[float] = []
    for r in range(calibration_series):
        chan = simulate_telemetry(
            window + series_length, spec, seed=stream.child("calibration", r)
        )
        g, b = scored(chan.values, chan.times)
        gp_z.extend(g)
        base_z.extend(b)
    thresholds = {
        "gp_abs_z": conformal_threshold([abs(v) for v in gp_z], alpha),
        "baseline_abs_z": conformal_threshold([abs(v) for v in base_z], alpha),
        "gp_cusum": calibrate_cusum_threshold(gp_z, alpha, k=cusum_k),
        "baseline_cusum": calibrate_cusum_threshold(base_z, alpha, k=cusum_k),
    }
    rows: list[OperatingPoint] = []
    for kind in faults:
        for size in sizes:
            stats: dict[str, list[tuple[int | None, int]]] = {m: [] for m in _METHODS}
            for r in range(replicates):
                label = ("fault", kind, repr(size), r)
                chan = simulate_telemetry(
                    window + series_length, spec, seed=stream.child(*label)
                )
                faulted = inject_fault(
                    chan.values[window:],
                    kind,
                    onset=onset,
                    size=size,
                    scale=spec.noise_sd,
                    duration=spike_duration,
                    drift_length=drift_length,
                    seed=stream.child(*label, "fault"),
                )
                series = (*chan.values[:window], *faulted.values)
                g, b = scored(series, chan.times)
                stop = onset + (spike_duration if kind == "spike" else horizon)
                for name, z in (("gp", g), ("baseline", b)):
                    level = thresholds[f"{name}_abs_z"]
                    hits = [i for i, v in enumerate(z) if abs(v) > level]
                    stats[f"{name}_abs_z"].append(
                        (_first_in(hits, onset, stop), sum(i < onset for i in hits))
                    )
                    alarms = cusum(z, k=cusum_k, h=thresholds[f"{name}_cusum"]).alarms
                    stats[f"{name}_cusum"].append(
                        (
                            _first_in(alarms, onset, stop),
                            sum(i < onset for i in alarms),
                        )
                    )
            for method in _METHODS:
                found = [(f, c) for f, c in stats[method]]
                delays = [f - onset for f, _ in found if f is not None]
                rows.append(
                    OperatingPoint(
                        kind,
                        float(size),
                        method,
                        len(delays) / replicates,
                        math.fsum(delays) / len(delays) if delays else math.nan,
                        sum(c for _, c in found) / (replicates * onset),
                        replicates,
                    )
                )
    return DetectionStudy(tuple(rows), alpha, thresholds)
