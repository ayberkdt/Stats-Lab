"""Stochastic kriging for simulators with replicated noisy output.

A stochastic simulation run at design point ``x_i`` returns
``Y_j(x_i) = b0 + M(x_i) + e_j(x_i)``: an unknown constant, a Gaussian random
field ``M`` with covariance ``S_M`` (the *extrinsic* uncertainty about the
response surface), and simulation noise ``e_j`` with variance ``V(x_i)`` that
may change over the design space (the *intrinsic* uncertainty). With ``n_i``
independent replications the sample means ``Ybar_i`` have noise variance
``V(x_i) / n_i``, so that ``Cov(Ybar) = S_M + S_e`` with
``S_e = diag(V(x_i) / n_i)``, and the best linear predictor of the mean
response is [EQ-SUR-SK-001]

    ``Yhat(x0) = b0 + S_M(x0, .)' (S_M + S_e)^-1 (Ybar - b0 1)``.

Following Ankenman, Nelson and Staum, ``V(x_i)`` is replaced by the sample
variance of the replications at ``x_i`` and then treated as known; the
reported mean squared error does not include the error of that plug-in. The
sample variances need enough replications to be useful: two per design
point is the minimum accepted here, and variances estimated from a handful
of replications are themselves very noisy.
Independent replications are assumed: common random numbers across design
points would make ``S_e`` non-diagonal and are not supported.

Kernels are the library's stationary ARD kernels
(:func:`~probstat.surrogates.kernel_value`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector, as_vector, cholesky, cholesky_solve
from probstat.optimization import multi_start

from .gaussian_process import KERNELS, kernel_value

_LOG_TWO_PI = math.log(2.0 * math.pi)


def _covariance(
    kernel: str,
    points: Sequence[Vector],
    signal: float,
    scales: Sequence[float],
    intrinsic: Sequence[float],
) -> Matrix:
    """Return ``S_M + S_e`` on the design.

    References:
        [EQ-SUR-SK-001], [REF-ANKENMAN-NELSON-STAUM-2010].

    """
    rows = []
    for i, left in enumerate(points):
        row = [kernel_value(kernel, left, right, scales, signal) for right in points]
        row[i] += intrinsic[i]
        rows.append(tuple(row))
    return tuple(rows)


def _profile(
    kernel: str,
    points: Sequence[Vector],
    means: Vector,
    intrinsic: Sequence[float],
    signal: float,
    scales: Sequence[float],
    trend: float | None,
) -> tuple[float, float, Matrix, Vector, Vector, float]:
    """Factorize ``Sigma = S_M + S_e`` and return the likelihood pieces.

    Returns:
        ``(log-likelihood, b0, L, Sigma^-1 (Ybar - b0 1), Sigma^-1 1,
        1' Sigma^-1 1)`` with ``b0`` the given trend or its generalized
        least-squares estimate ``1' Sigma^-1 Ybar / 1' Sigma^-1 1``.

    Raises:
        ValueError: If ``Sigma`` is not positive definite.

    References:
        [EQ-SUR-SK-001], [EQ-SUR-SK-003], [REF-ANKENMAN-NELSON-STAUM-2010].

    """
    count = len(points)
    lower = cholesky(_covariance(kernel, points, signal, scales, intrinsic))
    solved_ones = cholesky_solve(lower, (1.0,) * count)
    total = math.fsum(solved_ones)
    if trend is None:
        constant = math.fsum(a * b for a, b in zip(solved_ones, means, strict=True))
        constant /= total
    else:
        constant = trend
    residuals = tuple(value - constant for value in means)
    alpha = cholesky_solve(lower, residuals)
    log_det = 2.0 * math.fsum(math.log(lower[i][i]) for i in range(count))
    quadratic = math.fsum(a * b for a, b in zip(residuals, alpha, strict=True))
    likelihood = -0.5 * (log_det + quadratic + count * _LOG_TWO_PI)
    return likelihood, constant, lower, alpha, solved_ones, total


@dataclass(frozen=True, slots=True)
class StochasticKriging:
    """A fitted stochastic kriging metamodel of a noisy simulator.

    Args:
        kernel: Kernel name of the extrinsic covariance ``S_M``.
        points: Design points ``x_1..x_k``.
        means: Sample means ``Ybar_i`` of the replications.
        intrinsic_variances: Diagonal of ``S_e``: the variances
            ``V(x_i) / n_i`` of the sample means.
        replications: Replication counts ``n_i`` (``0`` when the model was
            built from given variances of the means).
        signal_variance: Extrinsic process variance ``tau^2 = S_M(x, x)``.
        length_scales: Length scales of ``S_M``.
        trend: The constant ``b0``.
        trend_estimated: Whether ``b0`` is the generalized least-squares
            estimate (its variance then enters the mean squared error).
        log_likelihood: Gaussian log-likelihood of the sample means at the
            reported parameters (profiled over ``b0`` when it is estimated).
        converged: Whether the best optimizer run met its tolerance
            (``True`` when the hyperparameters were given).
        restarts: Optimizer starts (``0`` when the hyperparameters were
            given).
        evaluations: Likelihood evaluations (factorizations of ``Sigma``).
        at_bound: Hyperparameters within ``0.01`` (log units) of a bound of
            the search box, as ``"length_scale[0]:upper"`` and similar.

    Raises:
        ValueError: If the pieces are inconsistent or ``S_M + S_e`` is not
            positive definite.

    Equation:
        [EQ-SUR-SK-001], [EQ-SUR-SK-002].

    References:
        [REF-ANKENMAN-NELSON-STAUM-2010].

    """

    kernel: str
    points: tuple[Vector, ...]
    means: Vector
    intrinsic_variances: Vector
    replications: tuple[int, ...]
    signal_variance: float
    length_scales: Vector
    trend: float
    trend_estimated: bool
    log_likelihood: float
    converged: bool = True
    restarts: int = 0
    evaluations: int = 1
    at_bound: tuple[str, ...] = ()
    _lower: Matrix = field(init=False, repr=False, compare=False)
    _alpha: Vector = field(init=False, repr=False, compare=False)
    _solved_ones: Vector = field(init=False, repr=False, compare=False)
    _ones_total: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Factorize ``S_M + S_e`` once for all predictions.

        Raises:
            ValueError: If the stored pieces are inconsistent or the
                covariance matrix is not positive definite.

        References:
            [EQ-SUR-SK-001], [REF-ANKENMAN-NELSON-STAUM-2010].

        """
        if self.kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}")
        count = len(self.points)
        if count < 1 or len(self.means) != count:
            raise ValueError("need one sample mean per design point")
        if len(self.intrinsic_variances) != count:
            raise ValueError("need one intrinsic variance per design point")
        _, _, lower, alpha, solved_ones, total = _profile(
            self.kernel,
            self.points,
            self.means,
            self.intrinsic_variances,
            self.signal_variance,
            self.length_scales,
            self.trend,
        )
        object.__setattr__(self, "_lower", lower)
        object.__setattr__(self, "_alpha", alpha)
        object.__setattr__(self, "_solved_ones", solved_ones)
        object.__setattr__(self, "_ones_total", total)

    def predict_with_variance(self, x: Sequence[float]) -> tuple[float, float]:
        """Return the predictor of the mean response and its MSE at ``x``.

        With ``c = S_M(x, .)`` and ``Sigma = S_M + S_e``,
        ``Yhat = b0 + c' Sigma^-1 (Ybar - b0 1)`` and
        ``MSE = tau^2 - c' Sigma^-1 c + (1 - 1' Sigma^-1 c)^2 / (1' Sigma^-1 1)``;
        the last term is the variance contribution of the generalized
        least-squares estimate of ``b0`` and is omitted when ``b0`` was
        given. The MSE refers to the mean response surface (no simulation
        noise of a future run is added) and treats ``S_e`` and the kernel
        hyperparameters as known. Tiny negative values from cancellation are
        reported as zero.

        Args:
            x: Prediction point.

        Returns:
            ``(Yhat(x), MSE(x))``.

        Raises:
            ValueError: If ``x`` has the wrong dimension.

        Equation:
            [EQ-SUR-SK-001], [EQ-SUR-SK-002].

        References:
            [REF-ANKENMAN-NELSON-STAUM-2010].

        """
        point = as_vector(x, name="x")
        if len(point) != len(self.points[0]):
            raise ValueError("x must have the design dimension")
        cross = tuple(
            kernel_value(
                self.kernel, point, train, self.length_scales, self.signal_variance
            )
            for train in self.points
        )
        mean = self.trend + math.fsum(
            a * b for a, b in zip(cross, self._alpha, strict=True)
        )
        solved = cholesky_solve(self._lower, cross)
        variance = self.signal_variance - math.fsum(
            a * b for a, b in zip(cross, solved, strict=True)
        )
        if self.trend_estimated:
            gap = 1.0 - math.fsum(
                a * b for a, b in zip(cross, self._solved_ones, strict=True)
            )
            variance += gap * gap / self._ones_total
        return mean, max(variance, 0.0)

    def predict(self, x: Sequence[float]) -> float:
        """Return the stochastic kriging predictor ``Yhat(x)``.

        Args:
            x: Prediction point.

        Returns:
            Predicted mean response.

        Equation:
            [EQ-SUR-SK-001].

        References:
            [REF-ANKENMAN-NELSON-STAUM-2010].

        """
        return self.predict_with_variance(x)[0]


def _bound_hits(
    theta: Sequence[float], bounds: Sequence[tuple[float, float]]
) -> tuple[str, ...]:
    """Name the log hyperparameters within ``0.01`` of a bound.

    References:
        [EQ-SUR-SK-003]. No new equation.

    """
    names = ("signal_variance", *(f"length_scale[{i}]" for i in range(len(theta) - 1)))
    hits = []
    for name, value, (low, high) in zip(names, theta, bounds, strict=True):
        if value - low <= 0.01:
            hits.append(f"{name}:lower")
        elif high - value <= 0.01:
            hits.append(f"{name}:upper")
    return tuple(hits)


def stochastic_kriging_from_summaries(
    points: Sequence[Sequence[float]],
    means: Sequence[float],
    mean_variances: Sequence[float],
    *,
    kernel: str = "squared_exponential",
    signal_variance: float | None = None,
    length_scales: Sequence[float] | None = None,
    trend: float | None = None,
    restarts: int = 3,
    seed: int | RandomStream | None = None,
    replications: Sequence[int] | None = None,
) -> StochasticKriging:
    """Fit stochastic kriging from sample means and their noise variances.

    ``mean_variances`` is the diagonal of ``S_e``, the variance of each
    sample mean (``V(x_i) / n_i``). With both ``signal_variance`` and
    ``length_scales`` given nothing is optimized. Otherwise the missing
    hyperparameters maximize the Gaussian likelihood of the sample means
    with ``S_e`` held fixed and ``b0`` profiled out by generalized least
    squares [EQ-SUR-SK-003]: BFGS with central differences on
    ``(log tau^2, log ell_1..d)`` inside the box
    ``tau^2 in [1e-6, 1e3] s^2`` (``s^2`` the sample variance of the means)
    and ``ell_i in [0.01, 20] E_i`` (``E_i`` the extent of input ``i``), from
    ``tau^2 = s^2``, ``ell_i = E_i / 2`` and ``restarts - 1`` uniformly drawn
    starts. The box factors are computational choices.

    Args:
        points: Design points (at least two distinct points to estimate
            hyperparameters).
        means: Sample means ``Ybar_i``.
        mean_variances: Non-negative variances of the sample means.
        kernel: ``"squared_exponential"``, ``"matern52"``, or ``"matern32"``.
        signal_variance: Fixed ``tau^2``, or ``None`` to estimate it.
        length_scales: Fixed length scales, or ``None`` to estimate them.
        trend: Fixed ``b0``, or ``None`` for the generalized least-squares
            estimate.
        restarts: Optimizer starts (at least one).
        seed: Integer seed or stream for the random starts.
        replications: Optional replication counts recorded in the result.

    Returns:
        Fitted stochastic kriging metamodel.

    Raises:
        ValueError: If inputs are invalid, the means are constant while
            ``tau^2`` must be estimated, an input is constant while its
            length scale must be estimated, or no start gives a
            positive-definite covariance.

    Equation:
        [EQ-SUR-SK-001], [EQ-SUR-SK-003].

    References:
        [REF-ANKENMAN-NELSON-STAUM-2010].

    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    x = tuple(as_vector(point, name="point") for point in points)
    ybar = as_vector(means, name="means")
    intrinsic = as_vector(mean_variances, name="mean_variances")
    count = len(x)
    if count < 1 or len(ybar) != count or len(intrinsic) != count:
        raise ValueError("need one mean and one variance per design point")
    dimension = len(x[0])
    if dimension == 0 or any(len(point) != dimension for point in x):
        raise ValueError("points must be non-empty and equally long")
    if any(value < 0.0 for value in intrinsic):
        raise ValueError("mean_variances must be non-negative")
    if isinstance(restarts, bool) or not isinstance(restarts, int) or restarts < 1:
        raise ValueError("restarts must be a positive integer")
    counts = (0,) * count if replications is None else tuple(replications)
    if len(counts) != count:
        raise ValueError("need one replication count per design point")
    constant = None if trend is None else require_finite_real(trend, name="trend")
    fixed_signal: float | None = None
    if signal_variance is not None:
        fixed_signal = require_finite_real(signal_variance, name="signal_variance")
        if fixed_signal <= 0.0:
            raise ValueError("signal_variance must be positive")
    fixed_scales: Vector | None = None
    if length_scales is not None:
        fixed_scales = as_vector(length_scales, name="length_scales")
        if len(fixed_scales) != dimension or any(s <= 0.0 for s in fixed_scales):
            raise ValueError("length_scales must be positive, one per input")
    if fixed_signal is not None and fixed_scales is not None:
        likelihood, b0, *_ = _profile(
            kernel, x, ybar, intrinsic, fixed_signal, fixed_scales, constant
        )
        return StochasticKriging(
            kernel,
            x,
            ybar,
            intrinsic,
            counts,
            fixed_signal,
            fixed_scales,
            b0,
            constant is None,
            likelihood,
        )
    bounds: list[tuple[float, float]] = []
    start: list[float] = []
    if fixed_signal is None:
        centre = math.fsum(ybar) / count
        spread = (
            math.fsum((value - centre) ** 2 for value in ybar) / (count - 1)
            if count > 1
            else 0.0
        )
        if spread <= 0.0:
            raise ValueError("sample means are constant; tau^2 cannot be estimated")
        bounds.append((math.log(1e-6 * spread), math.log(1e3 * spread)))
        start.append(math.log(spread))
    if fixed_scales is None:
        for axis in range(dimension):
            column = [point[axis] for point in x]
            extent = max(column) - min(column)
            if extent <= 0.0:
                raise ValueError(f"input {axis} is constant across the design")
            bounds.append((math.log(0.01 * extent), math.log(20.0 * extent)))
            start.append(math.log(0.5 * extent))
    evaluations = [0]

    def decode(theta: Sequence[float]) -> tuple[float, Vector]:
        """Split the optimizer vector into ``tau^2`` and the length scales.

        References:
            [EQ-SUR-SK-003]. No new equation.

        """
        values = list(theta)
        signal = math.exp(values.pop(0)) if fixed_signal is None else fixed_signal
        scales = (
            tuple(math.exp(value) for value in values)
            if fixed_scales is None
            else fixed_scales
        )
        return signal, scales

    def negative(theta: Vector) -> float:
        """Return minus the profile log-likelihood of the sample means.

        Raises:
            ValueError: If the covariance is not positive definite.

        References:
            [EQ-SUR-SK-003], [REF-ANKENMAN-NELSON-STAUM-2010].

        """
        evaluations[0] += 1
        signal, scales = decode(theta)
        return -_profile(kernel, x, ybar, intrinsic, signal, scales, constant)[0]

    result = multi_start(
        negative,
        bounds,
        starts=restarts,
        seed=resolve_stream(seed).child("stochastic-kriging"),
        method="bfgs",
        x0=start,
    )
    best = result.best
    signal, scales = decode(best.x)
    likelihood, b0, *_ = _profile(kernel, x, ybar, intrinsic, signal, scales, constant)
    return StochasticKriging(
        kernel,
        x,
        ybar,
        intrinsic,
        counts,
        signal,
        scales,
        b0,
        constant is None,
        likelihood,
        converged=best.converged,
        restarts=restarts,
        evaluations=evaluations[0],
        at_bound=_bound_hits(best.x, bounds) if fixed_signal is None else (),
    )


def fit_stochastic_kriging(
    points: Sequence[Sequence[float]],
    replicates: Sequence[Sequence[float]],
    *,
    kernel: str = "squared_exponential",
    signal_variance: float | None = None,
    length_scales: Sequence[float] | None = None,
    trend: float | None = None,
    restarts: int = 3,
    seed: int | RandomStream | None = None,
) -> StochasticKriging:
    """Fit stochastic kriging to replicated simulation output.

    For each design point the ``n_i >= 2`` replications are reduced to the
    sample mean ``Ybar_i`` and the unbiased sample variance ``V_i``; the
    intrinsic covariance is ``S_e = diag(V_i / n_i)`` and the fit proceeds
    as in :func:`stochastic_kriging_from_summaries`.

    Args:
        points: Design points.
        replicates: One sequence of at least two outputs per design point
            (the counts may differ).
        kernel: Kernel of the extrinsic covariance.
        signal_variance: Fixed ``tau^2``, or ``None`` to estimate it.
        length_scales: Fixed length scales, or ``None`` to estimate them.
        trend: Fixed ``b0``, or ``None`` for the generalized least-squares
            estimate.
        restarts: Optimizer starts.
        seed: Integer seed or stream for the random starts.

    Returns:
        Fitted stochastic kriging metamodel with ``replications`` set.

    Raises:
        ValueError: If a design point has fewer than two replications or
            the inputs are otherwise invalid.

    Equation:
        [EQ-SUR-SK-001], [EQ-SUR-SK-003].

    References:
        [REF-ANKENMAN-NELSON-STAUM-2010].

    """
    if len(replicates) != len(points):
        raise ValueError("need one sequence of replications per design point")
    means: list[float] = []
    variances: list[float] = []
    counts: list[int] = []
    for outputs in replicates:
        sample = as_vector(outputs, name="replicates")
        size = len(sample)
        if size < 2:
            raise ValueError("every design point needs at least two replications")
        centre = math.fsum(sample) / size
        variance = math.fsum((value - centre) ** 2 for value in sample) / (size - 1)
        means.append(centre)
        variances.append(variance / size)
        counts.append(size)
    return stochastic_kriging_from_summaries(
        points,
        means,
        variances,
        kernel=kernel,
        signal_variance=signal_variance,
        length_scales=length_scales,
        trend=trend,
        restarts=restarts,
        seed=seed,
        replications=counts,
    )
