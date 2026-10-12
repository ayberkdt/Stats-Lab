"""Gaussian-process regression with ARD kernels and marginal-likelihood fitting.

A Gaussian process (GP) prior ``f ~ GP(0, k)`` conditioned on noisy training
data gives a Gaussian predictive law whose mean interpolates (or smooths) the
data and whose variance grows away from it. Outputs are centered and scaled
to unit sample variance before fitting; the reported hyperparameters and
predictions are transformed back to the original units.

The ``nugget`` is an explicit noise variance (in standardized output units)
added to the kernel diagonal. It regularizes nearly singular kernel matrices
for deterministic simulators and is never added silently; a nugget that is
too small for the data raises a clear error instead of being enlarged.

Hyperparameter regularization is opt-in (``regularization=``, see
:class:`GaussianProcessRegularization` and ADR-0054): it bounds the log length
scales relative to the spacing and extent of the training inputs, bounds the
signal variance, optionally adds a log-normal prior on the length scales, and
keeps the diagonal at least a fixed fraction of the signal variance. Without
it the fit is the plain type-II maximum likelihood estimate in the box
``|theta| <= 25``. Both report which parameters ended on a bound.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    cholesky_solve,
    solve_lower,
)
from probstat.modeling import ProbabilisticModel
from probstat.optimization import minimize_bfgs
from probstat.propagation import make_design

KERNELS = ("squared_exponential", "matern52")
_SQRT5 = math.sqrt(5.0)
_LOG_TWO_PI = math.log(2.0 * math.pi)
_LOG_PARAMETER_BOUND = 25.0
_BOUND_TOLERANCE = 0.01


@dataclass(frozen=True, slots=True)
class GaussianProcessRegularization:
    """Bounds, prior, and diagonal floor for GP hyperparameter estimation.

    With per-input training extent ``E_i = max x_i - min x_i`` and smallest
    positive gap ``h_i`` between sorted distinct coordinates of input ``i``,
    the log parameters are confined to

    * ``log ell_i in [log(a h_i), log(b E_i)]``: below the smallest gap all
      off-diagonal correlations vanish and the likelihood is flat in
      ``ell_i`` (a white-noise fit with no information about ``ell_i``);
      far above the extent the kernel matrix tends to a singular
      low-order-polynomial limit whose likelihood keeps rising while the
      posterior variance falls below rounding resolution;
    * ``log sigma_f^2 in [log s_lo, log s_hi]`` in standardized output units
      (the data have unit sample variance);
    * ``log sigma_n^2 in [-25, log s_hi]`` when noise is estimated.

    The diagonal of the kernel matrix is ``sigma_f^2 (1 + tau) + nugget
    [+ sigma_n^2]`` with ``tau = min_relative_nugget``: a noise floor
    proportional to the signal variance bounds the condition number of the
    correlation part by ``1 + n / tau`` whatever the length scales. With
    ``length_scale_prior_sd = s`` the objective adds the log-normal prior
    ``log ell_i ~ N(log E_i, s^2)`` (a maximum a posteriori fit). The
    optimizer works on a smooth periodic reparameterization of the box, so a
    maximum on a bound is approached smoothly and is reported in
    :attr:`GaussianProcessRegression.at_bound`.

    The default factors are computational choices, not published constants:
    ``a = 0.1`` keeps the full range the data can inform, ``b = 20`` admits
    near-linear trends across the design, ``[1e-3, 1e3]`` allows signal
    variances three orders of magnitude around the sample variance, and
    ``tau = 1e-8`` (a relative noise standard deviation of ``1e-4``) keeps
    ``1 + n / tau`` below about ``1e11`` for the tens to hundreds of points
    a pure-Python GP handles.

    Args:
        lower_spacing_factor: ``a > 0``.
        upper_extent_factor: ``b > 0``.
        signal_variance_bounds: ``(s_lo, s_hi)`` with ``0 < s_lo < s_hi``.
        min_relative_nugget: ``tau >= 0``.
        length_scale_prior_sd: Log-scale prior standard deviation ``s > 0``,
            or ``None`` for no prior (bounded maximum likelihood).

    Raises:
        ValueError: If a field is out of range.

    Equation:
        [EQ-SUR-GP-006].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Sections 5.2 and 5.4],
        [REF-ANDRIANAKIS-CHALLENOR-2012].

    """

    lower_spacing_factor: float = 0.1
    upper_extent_factor: float = 20.0
    signal_variance_bounds: tuple[float, float] = (1e-3, 1e3)
    min_relative_nugget: float = 1e-8
    length_scale_prior_sd: float | None = None

    def __post_init__(self) -> None:
        """Validate the regularization settings.

        Raises:
            ValueError: If a field is out of range.

        References:
            Project validation policy. No equation is evaluated.

        """
        for name in ("lower_spacing_factor", "upper_extent_factor"):
            value = require_finite_real(getattr(self, name), name=name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        low, high = (
            require_finite_real(value, name="signal_variance_bounds")
            for value in self.signal_variance_bounds
        )
        if not 0.0 < low < high:
            raise ValueError("signal_variance_bounds must satisfy 0 < low < high")
        tau = require_finite_real(self.min_relative_nugget, name="min_relative_nugget")
        if tau < 0.0:
            raise ValueError("min_relative_nugget must be non-negative")
        if self.length_scale_prior_sd is not None:
            sd = require_finite_real(
                self.length_scale_prior_sd, name="length_scale_prior_sd"
            )
            if sd <= 0.0:
                raise ValueError("length_scale_prior_sd must be positive")


def gaussian_process_log_parameter_bounds(
    points: Sequence[Sequence[float]],
    regularization: GaussianProcessRegularization,
    *,
    estimate_noise: bool = False,
) -> tuple[tuple[float, float], ...]:
    """Return the box for ``theta = (log sigma_f^2, log ell_1..d[, log sigma_n^2])``.

    Length-scale bounds are ``[log(a h_i), log(b E_i)]`` with ``h_i`` the
    smallest positive gap between sorted coordinates of input ``i`` and
    ``E_i`` its extent; all bounds are clipped to ``|theta| <= 25``.

    Args:
        points: Training inputs.
        regularization: Regularization settings.
        estimate_noise: Whether ``theta`` ends with ``log sigma_n^2``.

    Returns:
        One ``(lower, upper)`` pair per log parameter.

    Raises:
        ValueError: If an input is constant across the points.

    Equation:
        [EQ-SUR-GP-006].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

    """
    x = tuple(as_vector(point, name="point") for point in points)
    cap = _LOG_PARAMETER_BOUND
    low, high = regularization.signal_variance_bounds
    bounds = [(max(math.log(low), -cap), min(math.log(high), cap))]
    for axis in range(len(x[0])):
        column = sorted({point[axis] for point in x})
        if len(column) < 2:
            raise ValueError(f"input {axis} is constant across the training points")
        gap = min(b - a for a, b in zip(column, column[1:], strict=False))
        extent = column[-1] - column[0]
        lower = math.log(regularization.lower_spacing_factor * gap)
        upper = math.log(regularization.upper_extent_factor * extent)
        lower, upper = max(lower, -cap), min(upper, cap)
        if lower >= upper:
            raise ValueError(f"length-scale bounds of input {axis} are empty")
        bounds.append((lower, upper))
    if estimate_noise:
        bounds.append((-cap, min(math.log(high), cap)))
    return tuple(bounds)


def _bound_hits(
    theta: Sequence[float], bounds: Sequence[tuple[float, float]], dimension: int
) -> tuple[str, ...]:
    """Name the log parameters within ``0.01`` of a bound.

    References:
        [EQ-SUR-GP-006]. No new equation.

    """
    names = (
        "signal_variance",
        *(f"length_scale[{axis}]" for axis in range(dimension)),
        "noise_variance",
    )
    hits: list[str] = []
    for name, value, (lower, upper) in zip(names, theta, bounds, strict=False):
        if value - lower <= _BOUND_TOLERANCE:
            hits.append(f"{name}:lower")
        elif upper - value <= _BOUND_TOLERANCE:
            hits.append(f"{name}:upper")
    return tuple(hits)


@dataclass(frozen=True, slots=True)
class GaussianProcessHyperparameters:
    """Kernel hyperparameters in original output and input units.

    Args:
        signal_variance: Prior process variance ``sigma_f^2``.
        length_scales: ARD length scales ``ell_i``, one per input.
        noise_variance: Diagonal noise variance (nugget plus any estimated
            noise).

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapters 2 and 5].

    """

    signal_variance: float
    length_scales: Vector
    noise_variance: float


def kernel_value(
    kernel: str,
    left: Sequence[float],
    right: Sequence[float],
    length_scales: Sequence[float],
    signal_variance: float,
) -> float:
    """Evaluate a stationary ARD covariance kernel.

    Squared exponential: ``sigma_f^2 exp(-r^2 / 2)``; Matern 5/2:
    ``sigma_f^2 (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r)``, with
    ``r^2 = sum_i ((x_i - x'_i) / ell_i)^2``.

    Args:
        kernel: ``"squared_exponential"`` or ``"matern52"``.
        left: First input vector.
        right: Second input vector.
        length_scales: Positive length scales.
        signal_variance: Positive signal variance.

    Returns:
        Covariance ``k(x, x')``.

    Raises:
        ValueError: If the kernel name is unknown.

    Equation:
        [EQ-SUR-GP-001], [EQ-SUR-GP-002].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

    """
    squared = math.fsum(
        ((a - b) / ell) ** 2
        for a, b, ell in zip(left, right, length_scales, strict=True)
    )
    return signal_variance * _correlation(kernel, squared)


def _correlation(kernel: str, squared: float) -> float:
    """Return the unit-variance kernel as a function of ``r^2``.

    Raises:
        ValueError: If the kernel name is unknown.

    Equation:
        [EQ-SUR-GP-001], [EQ-SUR-GP-002].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

    """
    if kernel == "squared_exponential":
        return math.exp(-0.5 * squared)
    if kernel == "matern52":
        r = math.sqrt(squared)
        return (1.0 + _SQRT5 * r + 5.0 * squared / 3.0) * math.exp(-_SQRT5 * r)
    raise ValueError(f"kernel must be one of {KERNELS}")


def _length_scale_factor(kernel: str, squared: float) -> float:
    """Return ``(dk/d log ell_i) / (sigma_f^2 (Delta_i/ell_i)^2)``.

    Squared exponential: ``exp(-r^2/2)``. Matern 5/2:
    ``(5/3)(1 + sqrt(5) r) exp(-sqrt(5) r)``. Both are finite at ``r = 0``.

    Equation:
        [EQ-SUR-GP-004].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Sections 4.2 and 5.4].

    """
    if kernel == "squared_exponential":
        return math.exp(-0.5 * squared)
    r = math.sqrt(squared)
    return (5.0 / 3.0) * (1.0 + _SQRT5 * r) * math.exp(-_SQRT5 * r)


def _split_parameters(
    log_parameters: Sequence[float], dimension: int, estimate_noise: bool
) -> tuple[float, Vector, float]:
    """Decode ``(log sigma_f^2, log ell_1..d[, log sigma_n^2])``.

    Raises:
        ValueError: If the vector has the wrong length or leaves the
            documented box ``|theta_j| <= 25``.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4]. No equation is evaluated.

    """
    theta = as_vector(log_parameters, name="log_parameters")
    expected = 1 + dimension + (1 if estimate_noise else 0)
    if len(theta) != expected:
        raise ValueError(f"log_parameters must have length {expected}")
    if any(abs(value) > _LOG_PARAMETER_BOUND for value in theta):
        raise ValueError("log hyperparameters left the box |theta| <= 25")
    signal = math.exp(theta[0])
    scales = tuple(math.exp(value) for value in theta[1 : 1 + dimension])
    extra = math.exp(theta[-1]) if estimate_noise else 0.0
    return signal, scales, extra


def gaussian_process_log_marginal_likelihood(
    points: Sequence[Sequence[float]],
    targets: Sequence[float],
    log_parameters: Sequence[float],
    *,
    kernel: str = "matern52",
    nugget: float = 1e-10,
    estimate_noise: bool = False,
    relative_nugget: float = 0.0,
) -> tuple[float, Vector]:
    """Return the GP log marginal likelihood and its exact gradient.

    ``log p(y | theta) = -1/2 y' K^-1 y - 1/2 log det K - n/2 log 2 pi`` with
    ``K = K_f + (nugget + tau sigma_f^2 + sigma_n^2) I`` (``tau`` the
    relative nugget, ``0`` by default) and gradient
    ``d/d theta_j = 1/2 tr((alpha alpha' - K^-1) dK/d theta_j)``,
    ``alpha = K^-1 y``. Parameters are ``theta = (log sigma_f^2,
    log ell_1..d[, log sigma_n^2])``; targets are used as given (callers
    standardize them).

    Args:
        points: Training inputs.
        targets: Training targets.
        log_parameters: Log hyperparameter vector ``theta``.
        kernel: Kernel name.
        nugget: Fixed positive diagonal variance.
        estimate_noise: Whether ``theta`` ends with ``log sigma_n^2``.
        relative_nugget: Diagonal floor ``tau >= 0`` as a fraction of the
            signal variance (see :class:`GaussianProcessRegularization`).

    Returns:
        Log marginal likelihood and gradient with respect to ``theta``.

    Raises:
        ValueError: If the kernel matrix is not positive definite or the
            parameters leave the documented box.

    Equation:
        [EQ-SUR-GP-004].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

    """
    x = tuple(as_vector(point, name="point") for point in points)
    y = as_vector(targets, name="targets")
    dimension = len(x[0])
    signal, scales, extra = _split_parameters(log_parameters, dimension, estimate_noise)
    tau = require_finite_real(relative_nugget, name="relative_nugget")
    if tau < 0.0:
        raise ValueError("relative_nugget must be non-negative")
    count = len(x)
    noise = nugget + extra
    if tau > 0.0:
        noise += tau * signal
    scaled_sq: list[list[tuple[float, ...]]] = []
    correlation = [[0.0] * count for _ in range(count)]
    for i in range(count):
        row: list[tuple[float, ...]] = []
        for j in range(count):
            parts = tuple(
                ((a - b) / ell) ** 2
                for a, b, ell in zip(x[i], x[j], scales, strict=True)
            )
            row.append(parts)
            correlation[i][j] = _correlation(kernel, math.fsum(parts))
        scaled_sq.append(row)
    matrix = tuple(
        tuple(
            signal * correlation[i][j] + (noise if i == j else 0.0)
            for j in range(count)
        )
        for i in range(count)
    )
    try:
        lower = cholesky(matrix)
    except ValueError as error:
        raise ValueError(
            "kernel matrix is not positive definite; increase the nugget"
        ) from error
    alpha = cholesky_solve(lower, y)
    log_det = 2.0 * math.fsum(math.log(lower[i][i]) for i in range(count))
    value = -0.5 * math.fsum(a * b for a, b in zip(y, alpha, strict=True))
    value += -0.5 * log_det - 0.5 * count * _LOG_TWO_PI
    inverse_columns = [
        cholesky_solve(lower, tuple(1.0 if r == c else 0.0 for r in range(count)))
        for c in range(count)
    ]
    weight = [
        [alpha[i] * alpha[j] - inverse_columns[j][i] for j in range(count)]
        for i in range(count)
    ]
    gradient = [
        0.5
        * math.fsum(
            weight[i][j] * signal * correlation[i][j]
            for i in range(count)
            for j in range(count)
        )
    ]
    if tau > 0.0:
        gradient[0] += 0.5 * tau * signal * math.fsum(
            weight[i][i] for i in range(count)
        )
    for axis in range(dimension):
        gradient.append(
            0.5
            * math.fsum(
                weight[i][j]
                * signal
                * _length_scale_factor(kernel, math.fsum(scaled_sq[i][j]))
                * scaled_sq[i][j][axis]
                for i in range(count)
                for j in range(count)
                if i != j
            )
        )
    if estimate_noise:
        gradient.append(0.5 * extra * math.fsum(weight[i][i] for i in range(count)))
    return value, tuple(gradient)


@dataclass(frozen=True, slots=True)
class GaussianProcessLOO:
    """Closed-form leave-one-out diagnostics of a fitted GP.

    Args:
        means: LOO predictive means ``mu_(-i)``.
        variances: LOO predictive variances ``sigma^2_(-i)`` (with noise).
        residuals: ``y_i - mu_(-i)``.
        relative_error: ``mean(residual^2) / Var(y)``.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

    """

    means: Vector
    variances: Vector
    residuals: Vector
    relative_error: float


@dataclass(frozen=True)
class GaussianProcessRegression:
    """A fitted Gaussian-process surrogate.

    Args:
        kernel: Kernel name.
        points: Training inputs.
        values: Training outputs in original units.
        hyperparameters: Fitted hyperparameters in original units.
        log_parameters: Fitted ``theta`` in standardized output units.
        nugget: Fixed diagonal variance in standardized output units.
        estimate_noise: Whether a noise variance was estimated.
        output_mean: Training-output sample mean used for centering.
        output_scale: Training-output sample standard deviation used for
            scaling.
        log_marginal_likelihood: ``log p(y | theta)`` in original units.
        converged: Whether the best restart met the optimizer tolerance.
        restarts: Number of optimizer starts attempted.
        evaluations: Total likelihood evaluations over all restarts.
        relative_nugget: Diagonal floor ``tau`` as a fraction of the
            standardized signal variance (``0`` without regularization).
        regularization: Settings used, or ``None`` for the unregularized
            maximum likelihood fit.
        log_parameter_bounds: The box searched for ``theta`` (``|theta| <=
            25`` without regularization).
        at_bound: Parameters whose estimate lies within ``0.01`` (log
            units) of a bound, as ``"length_scale[0]:upper"`` and similar;
            empty when the optimum is interior. A length scale on its upper
            bound signals a nearly polynomial (or constant) trend, one on its
            lower bound a fit with almost no correlation between training
            points.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapters 2 and 5].

    """

    kernel: str
    points: tuple[Vector, ...]
    values: Vector
    hyperparameters: GaussianProcessHyperparameters
    log_parameters: Vector
    nugget: float
    estimate_noise: bool
    output_mean: float
    output_scale: float
    log_marginal_likelihood: float
    converged: bool
    restarts: int
    evaluations: int
    relative_nugget: float = 0.0
    regularization: GaussianProcessRegularization | None = None
    log_parameter_bounds: tuple[tuple[float, float], ...] = ()
    at_bound: tuple[str, ...] = ()
    _factorization: tuple[float, Vector, float, Matrix, Vector] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Factorize the training kernel matrix once for all predictions.

        Raises:
            ValueError: If the kernel matrix is not positive definite.

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1]. No new equation.

        """
        object.__setattr__(self, "_factorization", self._standardized())

    def _standardized(self) -> tuple[float, Vector, float, Matrix, Vector]:
        """Return signal, scales, noise, Cholesky factor, and ``alpha``.

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1]. No new equation.

        """
        dimension = len(self.points[0])
        signal, scales, extra = _split_parameters(
            self.log_parameters, dimension, self.estimate_noise
        )
        noise = self.nugget + extra
        if self.relative_nugget > 0.0:
            noise += self.relative_nugget * signal
        count = len(self.points)
        matrix = tuple(
            tuple(
                kernel_value(
                    self.kernel, self.points[i], self.points[j], scales, signal
                )
                + (noise if i == j else 0.0)
                for j in range(count)
            )
            for i in range(count)
        )
        lower = cholesky(matrix)
        targets = tuple(
            (value - self.output_mean) / self.output_scale for value in self.values
        )
        return signal, scales, noise, lower, cholesky_solve(lower, targets)

    @property
    def standardized_noise_variance(self) -> float:
        """Diagonal noise variance of the kernel matrix in standardized units.

        ``nugget + tau sigma_f^2 [+ sigma_n^2]``, exactly as factorized.

        References:
            [EQ-SUR-GP-006], [REF-RASMUSSEN-WILLIAMS-2006, Section 2.2].

        """
        return self._factorization[2]

    def predict_with_variance(
        self, x: Sequence[float], *, include_noise: bool = False
    ) -> tuple[float, float]:
        """Return the predictive mean and variance at one input.

        ``mu = m_y + s_y k_*' alpha`` and
        ``sigma^2 = s_y^2 (k_** - v'v [+ noise])`` with ``v = L^-1 k_*``.
        Tiny negative variances from cancellation are reported as zero. The
        subtraction cannot resolve variances below about ``1e-16 k_**``; with
        very long length scales (nearly flat posteriors) an exact zero means
        "below resolution", so variance-normalized conformal scores need an
        explicit offset (``conformal.conformalize_surrogate(scale_offset=...)``).

        Args:
            x: Input vector.
            include_noise: Add the diagonal noise variance (prediction of a
                new observation rather than of the latent function).

        Returns:
            Predictive mean and variance in original units.

        Equation:
            [EQ-SUR-GP-003].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

        """
        point = as_vector(x, name="x")
        if len(point) != len(self.points[0]):
            raise ValueError("x must have the training input dimension")
        signal, scales, noise, lower, alpha = self._factorization
        cross = tuple(
            kernel_value(self.kernel, point, train, scales, signal)
            for train in self.points
        )
        mean = math.fsum(a * b for a, b in zip(cross, alpha, strict=True))
        whitened = solve_lower(lower, cross)
        variance = signal - math.fsum(v * v for v in whitened)
        if include_noise:
            variance += noise
        return (
            self.output_mean + self.output_scale * mean,
            self.output_scale**2 * max(variance, 0.0),
        )

    def predict(self, x: Sequence[float]) -> float:
        """Return the predictive mean at one input.

        Args:
            x: Input vector.

        Returns:
            Predictive mean in original units.

        Equation:
            [EQ-SUR-GP-003].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

        """
        return self.predict_with_variance(x)[0]

    def leave_one_out(self) -> GaussianProcessLOO:
        """Return closed-form LOO predictions at fixed hyperparameters.

        ``mu_(-i) = y_i - [K^-1 y]_i / [K^-1]_ii`` and
        ``sigma^2_(-i) = 1 / [K^-1]_ii`` (standardized units).

        Returns:
            LOO means, variances, residuals, and relative error.

        Equation:
            [EQ-SUR-GP-005].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

        """
        _, _, _, lower, alpha = self._factorization
        count = len(self.points)
        means: list[float] = []
        variances: list[float] = []
        residuals: list[float] = []
        for i in range(count):
            column = cholesky_solve(
                lower, tuple(1.0 if r == i else 0.0 for r in range(count))
            )
            diagonal = column[i]
            standardized = (self.values[i] - self.output_mean) / self.output_scale
            loo_mean = standardized - alpha[i] / diagonal
            means.append(self.output_mean + self.output_scale * loo_mean)
            variances.append(self.output_scale**2 / diagonal)
            residuals.append(self.values[i] - means[-1])
        center = math.fsum(self.values) / count
        spread = math.fsum((v - center) ** 2 for v in self.values) / (count - 1)
        return GaussianProcessLOO(
            tuple(means),
            tuple(variances),
            tuple(residuals),
            math.fsum(r * r for r in residuals) / count / spread,
        )


def fit_gaussian_process(
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    *,
    kernel: str = "matern52",
    nugget: float = 1e-10,
    estimate_noise: bool = False,
    optimize: bool = True,
    restarts: int = 3,
    seed: int | RandomStream | None = None,
    initial_log_parameters: Sequence[float] | None = None,
    regularization: GaussianProcessRegularization | None = None,
) -> GaussianProcessRegression:
    """Fit a zero-mean GP to standardized outputs by maximum marginal likelihood.

    The first start uses ``sigma_f^2 = 1`` (standardized units), length scales
    equal to half of each input's range, and ``sigma_n^2 = 1e-4`` when noise is
    estimated (or ``initial_log_parameters``). Further starts perturb the log
    length scales and log signal variance uniformly by up to ``+/- 1.5``. Each
    start runs BFGS with the exact gradient on ``-log p(y | theta)``; the best
    finite optimum is kept.

    With ``regularization`` the search is confined to the box of
    :func:`gaussian_process_log_parameter_bounds` through the map
    ``theta_j = l_j + (u_j - l_j) sin^2(phi_j / 2)`` (BFGS runs on ``phi``;
    starts are clipped to the inner 98 % of each interval). A maximum on a
    bound is then an ordinary stationary point ``phi_j = 0`` or ``pi``,
    reached at the usual BFGS rate, where a logistic map would push
    ``phi_j`` towards infinity with a vanishing gradient. The BFGS gradient
    tolerance is ``1e-5`` (relative) on this path instead of ``1e-6``: at
    an optimum on a bound of a nearly singular direction, rounding in the
    likelihood gradient kept its norm near ``3e-6`` and the search took 200
    iterations without further progress. The
    diagonal gains ``tau sigma_f^2`` with ``tau = min_relative_nugget``, and
    an optional log-normal length-scale prior is added to the objective
    [EQ-SUR-GP-006]. ``log_marginal_likelihood`` always reports the
    likelihood alone. Parameters that end within ``0.01`` of a bound are
    listed in ``at_bound`` in both modes.

    Args:
        points: Training inputs (at least two distinct points).
        values: Training outputs (not all equal).
        kernel: ``"squared_exponential"`` or ``"matern52"``.
        nugget: Explicit positive diagonal variance in standardized units.
        estimate_noise: Also estimate a noise variance above the nugget.
        optimize: Maximize the likelihood; ``False`` keeps the first start.
        restarts: Total number of optimizer starts (at least one).
        seed: Integer seed or stream for the restart perturbations.
        initial_log_parameters: Optional first start ``theta``.
        regularization: Optional :class:`GaussianProcessRegularization`;
            ``None`` (default) gives the unregularized fit, unchanged from
            earlier releases (ADR-0054).

    Returns:
        Fitted regression surrogate.

    Raises:
        TypeError: If ``regularization`` has the wrong type.
        ValueError: If inputs are invalid or no start yields a positive
            definite kernel matrix.

    Equation:
        [EQ-SUR-GP-001], [EQ-SUR-GP-002], [EQ-SUR-GP-004], [EQ-SUR-GP-006].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapters 2, 4, and 5],
        [REF-NOCEDAL-WRIGHT-2006, Section 6.1].

    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    if regularization is not None and not isinstance(
        regularization, GaussianProcessRegularization
    ):
        raise TypeError("regularization must be a GaussianProcessRegularization")
    x = tuple(as_vector(point, name="point") for point in points)
    y = as_vector(values, name="values")
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need at least two points with one value each")
    dimension = len(x[0])
    if dimension == 0 or any(len(point) != dimension for point in x):
        raise ValueError("points must be non-empty and equally long")
    jitter = require_finite_real(nugget, name="nugget")
    if jitter <= 0.0:
        raise ValueError("nugget must be positive")
    if isinstance(restarts, bool) or not isinstance(restarts, int) or restarts < 1:
        raise ValueError("restarts must be a positive integer")
    count = len(y)
    mean = math.fsum(y) / count
    scale = math.sqrt(math.fsum((v - mean) ** 2 for v in y) / (count - 1))
    if scale <= 0.0:
        raise ValueError("values are constant; nothing to regress")
    targets = tuple((v - mean) / scale for v in y)
    ranges = []
    for axis in range(dimension):
        column = [point[axis] for point in x]
        width = max(column) - min(column)
        if width <= 0.0:
            raise ValueError(f"input {axis} is constant across the training points")
        ranges.append(width)
    if initial_log_parameters is not None:
        first = as_vector(initial_log_parameters, name="initial_log_parameters")
    else:
        first = (0.0, *(math.log(0.5 * width) for width in ranges)) + (
            (math.log(1e-4),) if estimate_noise else ()
        )
    _split_parameters(first, dimension, estimate_noise)
    generator = resolve_stream(seed).child("gp-restarts").generator()
    starts = [first]
    for _ in range(restarts - 1 if optimize else 0):
        perturbed = [first[0] + generator.uniform(-1.5, 1.5)]
        perturbed.extend(
            first[1 + axis] + generator.uniform(-1.5, 1.5) for axis in range(dimension)
        )
        if estimate_noise:
            perturbed.append(first[-1])
        starts.append(tuple(perturbed))
    if regularization is None:
        tau = 0.0
        prior_sd: float | None = None
        bounds = tuple(
            (-_LOG_PARAMETER_BOUND, _LOG_PARAMETER_BOUND) for _ in range(len(first))
        )
    else:
        tau = regularization.min_relative_nugget
        prior_sd = regularization.length_scale_prior_sd
        bounds = gaussian_process_log_parameter_bounds(
            x, regularization, estimate_noise=estimate_noise
        )
    centres = tuple(math.log(width) for width in ranges)

    def log_prior(theta: Vector) -> tuple[float, list[float]]:
        """Return the log-normal length-scale log prior and its gradient.

        References:
            [EQ-SUR-GP-006], [REF-RASMUSSEN-WILLIAMS-2006, Section 5.2].

        """
        gradient = [0.0] * len(theta)
        if prior_sd is None:
            return 0.0, gradient
        value = 0.0
        for axis, centre in enumerate(centres):
            offset = (theta[1 + axis] - centre) / prior_sd
            value -= 0.5 * offset * offset
            gradient[1 + axis] = -offset / prior_sd
        return value, gradient

    def objective(theta: Vector) -> tuple[float, list[float]]:
        """Return ``-(log p(y | theta) + log prior)`` and its gradient.

        References:
            [EQ-SUR-GP-004], [EQ-SUR-GP-006],
            [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

        """
        value, gradient = gaussian_process_log_marginal_likelihood(
            x,
            targets,
            theta,
            kernel=kernel,
            nugget=jitter,
            estimate_noise=estimate_noise,
            relative_nugget=tau,
        )
        prior, prior_gradient = log_prior(theta)
        return -(value + prior), [
            -(g + p) for g, p in zip(gradient, prior_gradient, strict=True)
        ]

    def to_theta(phi: Vector) -> Vector:
        """Map unconstrained ``phi`` to ``theta`` inside the box.

        References:
            [EQ-SUR-GP-006]. No new equation.

        """
        return tuple(
            min(max(low + (high - low) * math.sin(0.5 * p) ** 2, low), high)
            for p, (low, high) in zip(phi, bounds, strict=True)
        )

    def to_phi(theta: Vector) -> Vector:
        """Map ``theta`` (clipped to the inner 98 % of the box) to ``phi``.

        References:
            [EQ-SUR-GP-006]. No new equation.

        """
        phi = []
        for value, (low, high) in zip(theta, bounds, strict=True):
            fraction = min(max((value - low) / (high - low), 0.01), 0.99)
            phi.append(2.0 * math.asin(math.sqrt(fraction)))
        return tuple(phi)

    def negative(phi: Vector) -> float:
        """Return the objective in the optimizer's coordinates.

        References:
            [EQ-SUR-GP-004], [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

        """
        theta = phi if regularization is None else to_theta(phi)
        return objective(theta)[0]

    def negative_gradient(phi: Vector) -> Vector:
        """Return the objective gradient in the optimizer's coordinates.

        References:
            [EQ-SUR-GP-004], [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

        """
        if regularization is None:
            return tuple(objective(phi)[1])
        gradient = objective(to_theta(phi))[1]
        return tuple(
            0.5 * g * (high - low) * math.sin(p)
            for g, p, (low, high) in zip(gradient, phi, bounds, strict=True)
        )

    best: tuple[float, Vector, bool] | None = None
    evaluations = 0
    for start in starts:
        begin = start if regularization is None else to_phi(start)
        try:
            if optimize:
                result = minimize_bfgs(
                    negative,
                    begin,
                    gradient=negative_gradient,
                    gradient_tolerance=1e-6 if regularization is None else 1e-5,
                    max_iterations=200,
                )
                candidate = (result.value, result.x, result.converged)
                evaluations += result.evaluations
            else:
                candidate = (negative(begin), begin, True)
                evaluations += 1
        except ValueError:
            continue
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError("no start produced a positive-definite kernel matrix")
    value, optimum, converged = best
    theta = optimum if regularization is None else to_theta(optimum)
    if prior_sd is not None:
        value += log_prior(theta)[0]
    signal, scales, extra = _split_parameters(theta, dimension, estimate_noise)
    hyperparameters = GaussianProcessHyperparameters(
        signal * scale * scale, scales, (jitter + tau * signal + extra) * scale * scale
    )
    return GaussianProcessRegression(
        kernel,
        x,
        y,
        hyperparameters,
        theta,
        jitter,
        estimate_noise,
        mean,
        scale,
        -value - count * math.log(scale),
        converged,
        len(starts),
        evaluations,
        relative_nugget=tau,
        regularization=regularization,
        log_parameter_bounds=bounds,
        at_bound=_bound_hits(theta, bounds, dimension),
    )


def fit_gaussian_process_to_model(
    model: ProbabilisticModel,
    quantity: str | None = None,
    *,
    size: int,
    design: str = "lhs",
    seed: int | RandomStream | None = None,
    kernel: str = "matern52",
    nugget: float = 1e-10,
    estimate_noise: bool = False,
    restarts: int = 3,
    regularization: GaussianProcessRegularization | None = None,
) -> GaussianProcessRegression:
    """Train a GP surrogate of one QoI on a design drawn from the input law.

    Args:
        model: Probabilistic model.
        quantity: QoI name; defaults to the first declared quantity.
        size: Number of training evaluations.
        design: ``"mc"``, ``"lhs"``, ``"halton"``, or ``"sobol"``.
        seed: Root seed or stream (children ``"gp-design"`` and restarts).
        kernel: Kernel name.
        nugget: Explicit diagonal variance in standardized units.
        estimate_noise: Also estimate a noise variance.
        restarts: Optimizer starts.
        regularization: Optional hyperparameter regularization (see
            :func:`fit_gaussian_process`).

    Returns:
        Fitted regression on physical inputs; ``model.evaluations`` grows by
        ``size``.

    Equation:
        [EQ-SUR-GP-003], [EQ-SUR-GP-004], [EQ-SAMP-INV-001].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapters 2 and 5].

    """
    name = model.quantity_names[0] if quantity is None else quantity
    model.quantity(name)
    root = resolve_stream(seed)
    unit = make_design(design, size, model.dimension, seed=root.child("gp-design"))
    points = model.inputs.sample_design(unit)
    values = model.evaluate_many(points)[name]
    return fit_gaussian_process(
        points,
        values,
        kernel=kernel,
        nugget=nugget,
        estimate_noise=estimate_noise,
        restarts=restarts,
        seed=root,
        regularization=regularization,
    )
