"""Bayesian Gaussian-process emulators: hyperparameter posteriors and extensions.

Experimental (added in 0.28). The API may change before it is declared stable.

Type-II maximum likelihood (:func:`probstat.surrogates.fit_gaussian_process`)
plugs one hyperparameter vector into the predictive law and therefore ignores
the uncertainty of the hyperparameters, which is large when the design is
small. This module collects the tools that keep that uncertainty, validate an
emulator, and relax the Gaussian assumptions.

Fully Bayesian hyperparameters (:func:`fit_bayesian_gp`, [EQ-SUR-GPB-001])
    The vector ``theta = (log sigma_f^2, log ell_1..d, log sigma_n^2)`` (the
    layout of :func:`gaussian_process_log_marginal_likelihood` with
    ``estimate_noise=True``) receives independent Gaussian priors
    (:class:`GPHyperPrior`) on the standardized outputs; the posterior
    ``p(theta | y) ~ p(y | theta) p(theta)`` is sampled by coordinate-wise slice
    sampling from :mod:`probstat.inference`. The predictive law is the
    mixture over draws; its mean is the mean of the component means and its
    variance is ``mean(variances) + var(means)`` [EQ-SUR-GPB-002] (law of total
    variance).

Emulator validation (:func:`validate_emulator_moments`)
    Individual standardized errors, the Mahalanobis distance with its
    chi-square or scaled-F reference law, pivoted-Cholesky errors, and
    credible-interval coverage on held-out runs [EQ-SUR-GPB-003],
    [EQ-SUR-GPB-004], [EQ-SUR-GPB-005]. Every failed criterion is listed in
    ``flags``.

GP-based global sensitivity (:func:`gp_sobol_posterior`)
    Joint posterior sample paths on a Saltelli design give one Sobol estimate
    per path, so each first-order and total index has a distribution that
    reflects the emulator uncertainty [EQ-SUR-GPB-006].

GP classification (:func:`fit_laplace_gp_classifier`)
    Logistic likelihood, Laplace approximation of the latent posterior and the
    predictive class probability [EQ-SUR-GPB-007], [EQ-SUR-GPB-008].

Student-t process (:func:`fit_student_t_process`)
    Heavier-tailed process whose predictive variance depends on the observed
    data through ``beta = y' K^-1 y`` [EQ-SUR-GPB-009], [EQ-SUR-GPB-010]; it
    tends to the GP as the degrees of freedom grow.

Output-warped GP (:func:`fit_warped_gp`)
    Box-Cox warp of a positive response with the Jacobian in the likelihood and
    predictive quantiles through the inverse warp [EQ-SUR-GPB-011],
    [EQ-SUR-GPB-012].

Numerical policy: the nugget of each model is an explicit argument and is never
enlarged silently; a kernel matrix that is not positive definite raises
``ValueError`` (inside the sampler it is a zero-probability state and is
counted in ``failed_factorizations``). Randomness comes from
:func:`probstat.foundations.random_streams.resolve_stream` only.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from operator import mul

from probstat.distributions import ChiSquare, FisherF, Normal, StudentT
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.inference import MCMCResult, slice_sampler
from probstat.linalg import Vector, as_vector
from probstat.optimization import minimize_nelder_mead

from .gaussian_process import (
    KERNELS,
    GaussianProcessRegression,
    fit_gaussian_process,
    kernel_value,
)

__all__ = [
    "BayesianGPPrediction",
    "BayesianGPRegression",
    "EmulatorValidation",
    "GPHyperPrior",
    "GPSobolResult",
    "LaplaceGPClassifier",
    "StudentTProcessRegression",
    "WarpedGPRegression",
    "box_cox",
    "default_hyper_prior",
    "fit_bayesian_gp",
    "fit_laplace_gp_classifier",
    "fit_student_t_process",
    "fit_warped_gp",
    "gp_predictive_moments",
    "gp_sobol_posterior",
    "inverse_box_cox",
    "sobol_estimates",
    "student_t_process_log_marginal_likelihood",
    "validate_emulator_moments",
    "validate_gp_emulator",
]

_LOG_BOUND = 12.0
_LOG_TWO_PI = math.log(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)
_NORMAL = Normal(0.0, 1.0)
_NU_GRID = (2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 25.0, 50.0, 100.0, 1e3, 1e6)
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


# ---------------------------------------------------------------------------
# Dense helpers (small lower-triangular algebra on ragged row lists)
# ---------------------------------------------------------------------------


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the compensated inner product of two equal-length sequences.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 2.1]. No equation is evaluated.

    """
    return math.fsum(map(mul, left, right))


def _cholesky(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return the lower Cholesky factor as ragged rows (row ``i`` has ``i+1``).

    Raises:
        ValueError: If a pivot is not strictly positive and finite.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

    """
    lower: list[list[float]] = []
    for i, source in enumerate(matrix):
        row: list[float] = []
        for j in range(i):
            row.append((source[j] - _dot(row, lower[j][:j])) / lower[j][j])
        pivot = source[i] - _dot(row, row)
        if not math.isfinite(pivot) or pivot <= 0.0:
            raise ValueError("matrix must be positive definite")
        row.append(math.sqrt(pivot))
        lower.append(row)
    return lower


def _forward(lower: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Solve ``L x = b`` for ragged lower-triangular ``L``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.1]. No equation is evaluated.

    """
    solution: list[float] = []
    for i, row in enumerate(lower):
        solution.append((rhs[i] - _dot(row[:i], solution)) / row[i])
    return solution


def _backward(lower: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Solve ``L' x = b`` for ragged lower-triangular ``L``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.1]. No equation is evaluated.

    """
    size = len(lower)
    solution = [0.0] * size
    for i in range(size - 1, -1, -1):
        tail = math.fsum(lower[k][i] * solution[k] for k in range(i + 1, size))
        solution[i] = (rhs[i] - tail) / lower[i][i]
    return solution


def _standardize(values: Sequence[float]) -> tuple[float, float, Vector]:
    """Return mean, sample standard deviation and the standardized values.

    Raises:
        ValueError: For fewer than three values or a constant response.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2]. No equation is evaluated.

    """
    data = as_vector(values, name="values")
    count = len(data)
    if count < 3:
        raise ValueError("at least three observations are required")
    mean = math.fsum(data) / count
    spread = math.sqrt(math.fsum((v - mean) ** 2 for v in data) / (count - 1))
    if spread <= 0.0:
        raise ValueError("values must not be constant")
    return mean, spread, tuple((v - mean) / spread for v in data)


def _check_points(points: Sequence[Sequence[float]], count: int) -> tuple[Vector, ...]:
    """Validate training inputs and return them as vectors.

    Raises:
        ValueError: If the number or dimension of inputs is inconsistent.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2]. No equation is evaluated.

    """
    x = tuple(as_vector(point, name="point") for point in points)
    if len(x) != count:
        raise ValueError("points and values must have the same length")
    if not x or not x[0]:
        raise ValueError("points must be non-empty vectors")
    if any(len(point) != len(x[0]) for point in x):
        raise ValueError("all points must have the same dimension")
    return x


def _check_kernel(kernel: str) -> None:
    """Raise ``ValueError`` for an unknown kernel name.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2]. No equation is evaluated.

    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")


@dataclass(frozen=True, slots=True)
class _Conditioned:
    """A zero-mean GP conditioned on standardized targets at fixed parameters.

    Attributes:
        kernel: Kernel name.
        points: Training inputs.
        signal: Signal variance.
        scales: Length scales.
        noise: Total diagonal variance.
        lower: Ragged Cholesky factor of ``K + noise I``.
        alpha: ``K^-1 y``.
        half_log_det: ``log det(L)`` (half the log-determinant of ``K``).
        quad: ``y' K^-1 y``.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

    """

    kernel: str
    points: tuple[Vector, ...]
    signal: float
    scales: Vector
    noise: float
    lower: list[list[float]]
    alpha: list[float]
    half_log_det: float
    quad: float


def _condition(
    kernel: str,
    points: tuple[Vector, ...],
    targets: Sequence[float],
    signal: float,
    scales: Vector,
    noise: float,
) -> _Conditioned:
    """Factorize ``K + noise I`` and solve for the targets.

    Raises:
        ValueError: If the matrix is not positive definite.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

    """
    count = len(points)
    rows = [
        [kernel_value(kernel, points[i], points[j], scales, signal) for j in range(i)]
        + [signal + noise]
        for i in range(count)
    ]
    # ``_cholesky`` reads only the lower triangle (``source[j]`` for ``j <= i``).
    lower = _cholesky(rows)
    whitened = _forward(lower, targets)
    alpha = _backward(lower, whitened)
    return _Conditioned(
        kernel,
        points,
        signal,
        scales,
        noise,
        lower,
        alpha,
        math.fsum(math.log(row[-1]) for row in lower),
        _dot(whitened, whitened),
    )


def _predict_point(core: _Conditioned, x: Sequence[float]) -> tuple[float, float]:
    """Return the latent predictive mean and variance in standardized units.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

    """
    cross = [
        kernel_value(core.kernel, x, train, core.scales, core.signal)
        for train in core.points
    ]
    mean = _dot(cross, core.alpha)
    whitened = _forward(core.lower, cross)
    return mean, max(core.signal - _dot(whitened, whitened), 0.0)


def _predict_joint(
    core: _Conditioned, xs: Sequence[Vector], *, include_noise: bool
) -> tuple[list[float], list[list[float]]]:
    """Return the joint latent predictive mean and covariance (standardized).

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Eq. 2.19].

    """
    cross = [
        [
            kernel_value(core.kernel, x, train, core.scales, core.signal)
            for train in core.points
        ]
        for x in xs
    ]
    mean = [_dot(row, core.alpha) for row in cross]
    whitened = [_forward(core.lower, row) for row in cross]
    size = len(xs)
    covariance = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            value = kernel_value(
                core.kernel, xs[i], xs[j], core.scales, core.signal
            ) - _dot(whitened[i], whitened[j])
            covariance[i][j] = value
            covariance[j][i] = value
        if include_noise:
            covariance[i][i] += core.noise
    return mean, covariance


def _gp_log_likelihood(core: _Conditioned) -> float:
    """Return ``log N(y | 0, K)`` of a conditioned GP.

    Equation:
        [EQ-SUR-GPB-001].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Eq. 2.30].

    """
    count = len(core.points)
    return -0.5 * core.quad - core.half_log_det - 0.5 * count * _LOG_TWO_PI


def _normal_cdf(z: float) -> float:
    """Return the standard normal CDF through ``erfc``.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 3]. No equation is evaluated.

    """
    return 0.5 * math.erfc(-z / _SQRT2)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Return the linearly interpolated percentile of sorted values.

    References:
        [REF-SALTELLI-2008, Chapter 4]. No equation is evaluated.

    """
    position = probability * (len(sorted_values) - 1)
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


# ---------------------------------------------------------------------------
# 1. Fully Bayesian hyperparameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GPHyperPrior:
    """Independent Gaussian priors on the log hyperparameters (standardized).

    Args:
        log_signal_mean: Prior mean of ``log sigma_f^2``.
        log_signal_sd: Prior standard deviation of ``log sigma_f^2``.
        log_length_means: Prior means of ``log ell_i``, one per input.
        log_length_sds: Prior standard deviations of ``log ell_i``.
        log_noise_mean: Prior mean of ``log sigma_n^2``.
        log_noise_sd: Prior standard deviation of ``log sigma_n^2``.

    Raises:
        ValueError: For non-finite values, non-positive standard deviations or
            mismatched length-scale tuples.

    Equation:
        [EQ-SUR-GPB-001].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 5.2].

    """

    log_signal_mean: float
    log_signal_sd: float
    log_length_means: Vector
    log_length_sds: Vector
    log_noise_mean: float
    log_noise_sd: float

    def __post_init__(self) -> None:
        """Validate the prior parameters.

        Raises:
            ValueError: For invalid parameters.

        References:
            [EQ-SUR-GPB-001].

        """
        means = as_vector(self.log_length_means, name="log_length_means")
        sds = as_vector(self.log_length_sds, name="log_length_sds")
        if not means or len(means) != len(sds):
            raise ValueError("length-scale priors need one mean and sd per input")
        scalars = (
            self.log_signal_mean,
            self.log_signal_sd,
            self.log_noise_mean,
            self.log_noise_sd,
        )
        if not all(math.isfinite(v) for v in scalars):
            raise ValueError("prior parameters must be finite")
        if min((self.log_signal_sd, self.log_noise_sd, *sds)) <= 0.0:
            raise ValueError("prior standard deviations must be positive")
        object.__setattr__(self, "log_length_means", means)
        object.__setattr__(self, "log_length_sds", sds)

    @property
    def dimension(self) -> int:
        """Return the input dimension the prior was built for.

        References:
            [EQ-SUR-GPB-001].

        """
        return len(self.log_length_means)

    def mean_vector(self) -> Vector:
        """Return the prior mean of ``theta``.

        Returns:
            ``(log sigma_f^2, log ell_1..d, log sigma_n^2)`` prior means.

        References:
            [EQ-SUR-GPB-001].

        """
        return (self.log_signal_mean, *self.log_length_means, self.log_noise_mean)

    def sd_vector(self) -> Vector:
        """Return the prior standard deviations of ``theta``.

        Returns:
            Standard deviations in the layout of :meth:`mean_vector`.

        References:
            [EQ-SUR-GPB-001].

        """
        return (self.log_signal_sd, *self.log_length_sds, self.log_noise_sd)

    def log_density(self, theta: Sequence[float]) -> float:
        """Return the log prior density of ``theta`` (with constants).

        Args:
            theta: ``(log sigma_f^2, log ell_1..d, log sigma_n^2)``.

        Returns:
            ``sum_j log N(theta_j; m_j, s_j^2)``.

        Raises:
            ValueError: If ``theta`` has the wrong length.

        Equation:
            [EQ-SUR-GPB-001].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Section 5.2].

        """
        mean = self.mean_vector()
        sd = self.sd_vector()
        if len(theta) != len(mean):
            raise ValueError(f"theta must have length {len(mean)}")
        return math.fsum(
            -0.5 * ((t - m) / s) ** 2 - math.log(s) - 0.5 * _LOG_TWO_PI
            for t, m, s in zip(theta, mean, sd, strict=True)
        )

    def sample(self, stream: RandomStream) -> Vector:
        """Draw one hyperparameter vector from the prior.

        Args:
            stream: Random stream.

        Returns:
            A draw of ``theta``.

        Equation:
            [EQ-SUR-GPB-001].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Section 5.2].

        """
        generator = stream.generator()
        return tuple(
            generator.gauss(m, s)
            for m, s in zip(self.mean_vector(), self.sd_vector(), strict=True)
        )


def default_hyper_prior(
    points: Sequence[Sequence[float]],
    *,
    length_fraction: float = 0.25,
    length_sd: float = 1.0,
    signal_sd: float = 1.0,
    noise_mean: float = 1e-3,
    noise_sd: float = 1.5,
) -> GPHyperPrior:
    """Build a weakly informative prior from the input ranges.

    The median length scale is ``length_fraction`` times the range of each
    input; the signal variance is centred at one (outputs are standardized to
    unit sample variance) and the noise variance at ``noise_mean``. All priors
    are log-normal.

    Args:
        points: Training inputs (at least two distinct values per coordinate).
        length_fraction: Median length scale as a fraction of the input range.
        length_sd: Prior standard deviation of ``log ell_i``.
        signal_sd: Prior standard deviation of ``log sigma_f^2``.
        noise_mean: Prior median of the noise variance (standardized units).
        noise_sd: Prior standard deviation of ``log sigma_n^2``.

    Returns:
        The prior.

    Raises:
        ValueError: If an input coordinate is constant or a parameter is not
            positive.

    Equation:
        [EQ-SUR-GPB-001].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 5.2].

    """
    if min(length_fraction, length_sd, signal_sd, noise_mean, noise_sd) <= 0.0:
        raise ValueError("prior parameters must be positive")
    x = tuple(as_vector(point, name="point") for point in points)
    if len(x) < 2 or not x[0]:
        raise ValueError("at least two non-empty points are required")
    dimension = len(x[0])
    means: list[float] = []
    for axis in range(dimension):
        column = [point[axis] for point in x]
        span = max(column) - min(column)
        if span <= 0.0:
            raise ValueError("every input coordinate must vary")
        means.append(math.log(length_fraction * span))
    return GPHyperPrior(
        0.0,
        signal_sd,
        tuple(means),
        (length_sd,) * dimension,
        math.log(noise_mean),
        noise_sd,
    )


class _LogPosterior:
    """Log posterior of ``theta`` for the slice sampler (value only).

    Equation:
        [EQ-SUR-GPB-001].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 5].

    """

    def __init__(
        self,
        kernel: str,
        points: tuple[Vector, ...],
        targets: Vector,
        prior: GPHyperPrior,
        nugget: float,
    ) -> None:
        """Store the data, prior and nugget.

        References:
            [EQ-SUR-GPB-001].

        """
        self.kernel = kernel
        self.points = points
        self.targets = targets
        self.prior = prior
        self.nugget = nugget
        self.dimension = len(points[0])
        self.failures = 0

    def __call__(self, theta: Sequence[float]) -> float:
        """Return ``log p(y | theta) + log p(theta)`` or ``-inf`` outside the box.

        A non-positive-definite kernel matrix has zero numerical posterior mass
        and increments ``failures``.

        References:
            [EQ-SUR-GPB-001].

        """
        if any(abs(value) > _LOG_BOUND for value in theta):
            return -math.inf
        signal = math.exp(theta[0])
        scales = tuple(math.exp(v) for v in theta[1 : 1 + self.dimension])
        noise = self.nugget + math.exp(theta[-1])
        try:
            core = _condition(
                self.kernel, self.points, self.targets, signal, scales, noise
            )
        except ValueError:
            self.failures += 1
            return -math.inf
        return _gp_log_likelihood(core) + self.prior.log_density(theta)


@dataclass(frozen=True, slots=True)
class BayesianGPPrediction:
    """Predictive mixture moments at one input (original units).

    Args:
        mean: Mixture mean (mean of the component means).
        variance: Total variance ``mean_of_variances + variance_of_means``.
        mean_of_variances: Average conditional (aleatory plus GP) variance.
        variance_of_means: Variance of the component means (hyperparameter
            uncertainty).
        component_means: Mean for each posterior draw.
        component_variances: Variance for each posterior draw.

    Equation:
        [EQ-SUR-GPB-002].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 5].

    """

    mean: float
    variance: float
    mean_of_variances: float
    variance_of_means: float
    component_means: Vector
    component_variances: Vector


@dataclass(frozen=True, slots=True)
class BayesianGPRegression:
    """GP regression with posterior-sampled hyperparameters.

    Build it with :func:`fit_bayesian_gp`.

    Args:
        kernel: Kernel name.
        points: Training inputs.
        values: Training outputs in original units.
        prior: Hyperparameter prior (standardized units).
        draws: Posterior draws of ``theta`` (pooled over chains).
        output_mean: Training-output mean used for centering.
        output_scale: Training-output standard deviation used for scaling.
        nugget: Fixed diagonal variance in standardized units.
        mcmc: The sampler result (per-chain draws and diagnostics).
        failed_factorizations: Number of log-posterior evaluations whose kernel
            matrix was not positive definite (treated as zero density).

    Equation:
        [EQ-SUR-GPB-001], [EQ-SUR-GPB-002].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 5], [REF-NEAL-2003].

    """

    kernel: str
    points: tuple[Vector, ...]
    values: Vector
    prior: GPHyperPrior
    draws: tuple[Vector, ...]
    output_mean: float
    output_scale: float
    nugget: float
    mcmc: MCMCResult
    failed_factorizations: int
    _cores: tuple[_Conditioned, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Condition one GP per posterior draw.

        Raises:
            ValueError: If a draw does not give a positive-definite matrix.

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

        """
        dimension = len(self.points[0])
        targets = tuple((v - self.output_mean) / self.output_scale for v in self.values)
        cores = tuple(
            _condition(
                self.kernel,
                self.points,
                targets,
                math.exp(theta[0]),
                tuple(math.exp(v) for v in theta[1 : 1 + dimension]),
                self.nugget + math.exp(theta[-1]),
            )
            for theta in self.draws
        )
        object.__setattr__(self, "_cores", cores)

    def predict(
        self, x: Sequence[float], *, include_noise: bool = False
    ) -> BayesianGPPrediction:
        """Return the predictive mixture moments at one input.

        ``E = mean_s mu_s`` and ``Var = mean_s sigma_s^2 + var_s mu_s`` (the
        variance of the component means uses the population divisor ``S``).

        Args:
            x: Input vector.
            include_noise: Add each draw's noise variance (a new observation
                rather than the latent function).

        Returns:
            Mixture moments and the component means and variances.

        Raises:
            ValueError: If ``x`` has the wrong dimension.

        Equation:
            [EQ-SUR-GPB-002].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

        """
        point = as_vector(x, name="x")
        if len(point) != len(self.points[0]):
            raise ValueError("x must have the training input dimension")
        means: list[float] = []
        variances: list[float] = []
        for core in self._cores:
            mean, variance = _predict_point(core, point)
            if include_noise:
                variance += core.noise
            means.append(self.output_mean + self.output_scale * mean)
            variances.append(self.output_scale**2 * variance)
        count = len(means)
        center = math.fsum(means) / count
        mean_of_variances = math.fsum(variances) / count
        variance_of_means = math.fsum((m - center) ** 2 for m in means) / count
        return BayesianGPPrediction(
            center,
            mean_of_variances + variance_of_means,
            mean_of_variances,
            variance_of_means,
            tuple(means),
            tuple(variances),
        )

    def predictive_cdf(
        self, x: Sequence[float], value: float, *, include_noise: bool = False
    ) -> float:
        """Return the mixture CDF ``P(f(x) <= value)``.

        Args:
            x: Input vector.
            value: Evaluation point.
            include_noise: Predict a new noisy observation.

        Returns:
            Average of the component normal CDFs.

        Equation:
            [EQ-SUR-GPB-002].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Chapter 5].

        """
        prediction = self.predict(x, include_noise=include_noise)
        return self._cdf(prediction, value)

    @staticmethod
    def _cdf(prediction: BayesianGPPrediction, value: float) -> float:
        """Return the mixture CDF of a prediction at ``value``.

        References:
            [EQ-SUR-GPB-002].

        """
        total = 0.0
        for mean, variance in zip(
            prediction.component_means, prediction.component_variances, strict=True
        ):
            if variance <= 0.0:
                total += 1.0 if value >= mean else 0.0
            else:
                total += _normal_cdf((value - mean) / math.sqrt(variance))
        return total / len(prediction.component_means)

    def predictive_quantile(
        self, x: Sequence[float], probability: float, *, include_noise: bool = False
    ) -> float:
        """Return the mixture quantile by bisection on the mixture CDF.

        Args:
            x: Input vector.
            probability: Probability in ``(0, 1)``.
            include_noise: Predict a new noisy observation.

        Returns:
            ``q`` with ``P(f(x) <= q) = probability`` to relative ``1e-12``.

        Raises:
            ValueError: If ``probability`` is not in ``(0, 1)``.

        Equation:
            [EQ-SUR-GPB-002].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Chapter 5].

        """
        if not 0.0 < probability < 1.0:
            raise ValueError("probability must be in (0, 1)")
        prediction = self.predict(x, include_noise=include_noise)
        spreads = [math.sqrt(v) for v in prediction.component_variances]
        low = min(
            m - 12.0 * s
            for m, s in zip(prediction.component_means, spreads, strict=True)
        )
        high = max(
            m + 12.0 * s
            for m, s in zip(prediction.component_means, spreads, strict=True)
        )
        if high - low <= 0.0:
            return low
        for _ in range(200):
            mid = 0.5 * (low + high)
            if self._cdf(prediction, mid) < probability:
                low = mid
            else:
                high = mid
            if high - low <= 1e-13 * max(1.0, abs(mid)):
                break
        return 0.5 * (low + high)

    def predictive_interval(
        self, x: Sequence[float], level: float = 0.95, *, include_noise: bool = False
    ) -> tuple[float, float]:
        """Return the central ``level`` credible interval of the mixture.

        Args:
            x: Input vector.
            level: Credible level in ``(0, 1)``.
            include_noise: Predict a new noisy observation.

        Returns:
            Lower and upper mixture quantiles.

        Raises:
            ValueError: If ``level`` is not in ``(0, 1)``.

        Equation:
            [EQ-SUR-GPB-002].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Chapter 5].

        """
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        tail = 0.5 * (1.0 - level)
        return (
            self.predictive_quantile(x, tail, include_noise=include_noise),
            self.predictive_quantile(x, 1.0 - tail, include_noise=include_noise),
        )


def fit_bayesian_gp(
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    *,
    kernel: str = "matern52",
    prior: GPHyperPrior | None = None,
    draws: int = 200,
    warmup: int = 100,
    chains: int = 1,
    thin: int = 1,
    widths: float = 1.0,
    nugget: float = 1e-8,
    seed: int | RandomStream | None = None,
) -> BayesianGPRegression:
    """Sample the hyperparameter posterior of a GP by slice sampling.

    The standardized outputs ``y`` have likelihood ``N(0, K(theta) + (nugget +
    sigma_n^2) I)`` and ``theta = (log sigma_f^2, log ell_1..d, log
    sigma_n^2)`` has the independent Gaussian prior ``prior``; the posterior
    is sampled by :func:`probstat.inference.slice_sampler` started at the prior
    mean. The sampler adapts nothing during warm-up; inspect
    ``result.mcmc`` for convergence diagnostics before trusting the draws.

    Args:
        points: Training inputs.
        values: Training outputs (not constant, at least three).
        kernel: ``"squared_exponential"``, ``"matern52"`` or ``"matern32"``.
        prior: Hyperparameter prior; ``None`` uses :func:`default_hyper_prior`.
        draws: Kept draws per chain.
        warmup: Discarded iterations per chain.
        chains: Number of chains (pooled in ``draws``).
        thin: Thinning interval.
        widths: Initial slice width in log units.
        nugget: Fixed extra diagonal variance in standardized units.
        seed: Integer seed or stream.

    Returns:
        The fitted Bayesian GP.

    Raises:
        ValueError: For invalid data, kernel, prior dimension or nugget, or a
            prior mean whose kernel matrix is not positive definite.

    Equation:
        [EQ-SUR-GPB-001], [EQ-SUR-GPB-002].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 5], [REF-NEAL-2003].

    """
    _check_kernel(kernel)
    mean, spread, targets = _standardize(values)
    x = _check_points(points, len(targets))
    if nugget <= 0.0 or not math.isfinite(nugget):
        raise ValueError("nugget must be positive and finite")
    chosen = prior if prior is not None else default_hyper_prior(x)
    if chosen.dimension != len(x[0]):
        raise ValueError("prior dimension must match the input dimension")
    target = _LogPosterior(kernel, x, targets, chosen, nugget)
    result = slice_sampler(
        target,
        chosen.mean_vector(),
        draws=draws,
        warmup=warmup,
        chains=chains,
        thin=thin,
        widths=widths,
        seed=seed,
        check_convergence=False,
    )
    pooled = tuple(tuple(s) for chain in result.chains for s in chain.samples)
    return BayesianGPRegression(
        kernel,
        x,
        as_vector(values, name="values"),
        chosen,
        pooled,
        mean,
        spread,
        nugget,
        result,
        target.failures,
    )


# ---------------------------------------------------------------------------
# 2. Emulator validation (Bastos and O'Hagan)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EmulatorValidation:
    """Result of the emulator diagnostics on a validation set.

    Args:
        standardized_errors: ``D_i = (y_i - m_i) / sqrt(V_ii)``.
        mahalanobis: ``MD = (y - m)' V^-1 (y - m)``.
        mahalanobis_expected: ``E[MD] = m`` (number of validation points).
        mahalanobis_p_value: Two-sided tail probability of ``MD`` under its
            reference law.
        mahalanobis_reference: ``"chi_square(m)"`` or ``"scaled_F(m, nu)"``.
        pivot_order: Validation indices in the order chosen by the pivoted
            Cholesky factorization.
        pivoted_errors: Pivoted-Cholesky errors, in pivot order (length
            ``rank``).
        full_rank: Whether the factorization reached every validation point.
        coverage: Fraction of validation outputs inside the central credible
            interval of level ``coverage_level``.
        coverage_level: Nominal level of that interval.
        flags: Human-readable description of every failed criterion.
        passed: ``True`` when ``flags`` is empty.

    Equation:
        [EQ-SUR-GPB-003], [EQ-SUR-GPB-004], [EQ-SUR-GPB-005].

    References:
        [REF-BASTOS-OHAGAN-2009].

    """

    standardized_errors: Vector
    mahalanobis: float
    mahalanobis_expected: float
    mahalanobis_p_value: float
    mahalanobis_reference: str
    pivot_order: tuple[int, ...]
    pivoted_errors: Vector
    full_rank: bool
    coverage: float
    coverage_level: float
    flags: tuple[str, ...]
    passed: bool


def _pivoted_errors(
    residual: Sequence[float], covariance: Sequence[Sequence[float]]
) -> tuple[tuple[int, ...], Vector, bool]:
    """Return pivot order, pivoted-Cholesky errors and a full-rank flag.

    Greedy pivoting on the largest remaining conditional variance; the
    ``k``-th error is ``(r_p - E[r_p | earlier]) / sd(r_p | earlier)``. The
    factorization stops, and ``full_rank`` is ``False``, when the largest
    remaining conditional variance drops below ``1e-12`` of the largest
    marginal variance.

    Equation:
        [EQ-SUR-GPB-004].

    References:
        [REF-BASTOS-OHAGAN-2009].

    """
    size = len(residual)
    remaining = [covariance[i][i] for i in range(size)]
    floor = 1e-12 * max(remaining)
    factor: list[list[float]] = [[] for _ in range(size)]
    errors: list[float] = []
    order: list[int] = []
    chosen: set[int] = set()
    for step in range(size):
        pivot = max(
            (i for i in range(size) if i not in chosen), key=remaining.__getitem__
        )
        if remaining[pivot] <= floor:
            return tuple(order), tuple(errors), False
        root = math.sqrt(remaining[pivot])
        for i in range(size):
            if i in chosen or i == pivot:
                continue
            factor[i].append(
                (covariance[i][pivot] - _dot(factor[i][:step], factor[pivot][:step]))
                / root
            )
            remaining[i] -= factor[i][-1] ** 2
        factor[pivot].append(root)
        errors.append(
            (residual[pivot] - _dot(factor[pivot][:step], errors[:step])) / root
        )
        order.append(pivot)
        chosen.add(pivot)
    return tuple(order), tuple(errors), True


def validate_emulator_moments(
    observed: Sequence[float],
    mean: Sequence[float],
    covariance: Sequence[Sequence[float]],
    *,
    level: float = 0.95,
    degrees_of_freedom: float | None = None,
    alpha: float = 0.05,
) -> EmulatorValidation:
    """Run the Bastos-O'Hagan diagnostics on validation outputs.

    ``mean`` and ``covariance`` are the emulator's joint predictive moments at
    the validation inputs (include the noise variance on the diagonal when the
    outputs are noisy). Under a correct emulator with known hyperparameters

    * ``D_i = (y_i - m_i) / sqrt(V_ii)`` is standard normal,
    * ``MD = (y - m)' V^-1 (y - m)`` is chi-square with ``m`` degrees of
      freedom (``m`` validation points), or, with a variance estimated on
      ``nu`` degrees of freedom, ``(nu - 2) MD / (m nu)`` follows
      ``F(m, nu)`` [EQ-SUR-GPB-003],
    * the pivoted-Cholesky errors are independent standard normal and satisfy
      ``sum e_k^2 = MD`` [EQ-SUR-GPB-004],
    * a fraction ``level`` of the outputs falls in their central credible
      intervals [EQ-SUR-GPB-005].

    With ``degrees_of_freedom`` the individual and coverage quantiles use the
    Student-t law instead of the normal law.

    Failure rules (all listed in ``flags``): any ``|D_i| > 3`` or any
    ``|e_k| > 3``; more than ``expected + 3 sd`` of the ``|D_i|`` (or
    ``|e_k|``) above 2 with a binomial count (``p = 0.0455``); a two-sided
    Mahalanobis p-value below ``alpha``; coverage outside ``level`` plus or
    minus three binomial standard deviations; a rank-deficient predictive
    covariance. Validation errors of a GP are correlated, so these rules are
    screening thresholds, not exact tests.

    Args:
        observed: Validation outputs.
        mean: Predictive means.
        covariance: Predictive covariance (symmetric positive definite).
        level: Nominal credible level in ``(0, 1)``.
        degrees_of_freedom: Degrees of freedom ``nu > 2`` of an estimated
            variance, or ``None`` for known hyperparameters.
        alpha: Significance level of the Mahalanobis check.

    Returns:
        The diagnostics.

    Raises:
        ValueError: For inconsistent sizes, non-positive variances, a
            covariance that is not positive definite, or invalid levels.

    Equation:
        [EQ-SUR-GPB-003], [EQ-SUR-GPB-004], [EQ-SUR-GPB-005].

    References:
        [REF-BASTOS-OHAGAN-2009].

    """
    y = as_vector(observed, name="observed")
    m = as_vector(mean, name="mean")
    size = len(y)
    if size < 2 or len(m) != size or len(covariance) != size:
        raise ValueError("observed, mean and covariance must have the same size >= 2")
    if any(len(row) != size for row in covariance):
        raise ValueError("covariance must be square")
    if not (0.0 < level < 1.0 and 0.0 < alpha < 1.0):
        raise ValueError("level and alpha must be in (0, 1)")
    if degrees_of_freedom is not None and not degrees_of_freedom > 2.0:
        raise ValueError("degrees_of_freedom must exceed 2")
    if any(covariance[i][i] <= 0.0 for i in range(size)):
        raise ValueError("predictive variances must be positive")
    residual = [a - b for a, b in zip(y, m, strict=True)]
    errors = tuple(r / math.sqrt(covariance[i][i]) for i, r in enumerate(residual))
    lower = _cholesky(covariance)
    whitened = _forward(lower, residual)
    distance = _dot(whitened, whitened)
    if degrees_of_freedom is None:
        cdf = ChiSquare(float(size)).cdf(distance)
        reference = f"chi_square({size})"
    else:
        scaled = (degrees_of_freedom - 2.0) * distance / (size * degrees_of_freedom)
        cdf = FisherF(float(size), degrees_of_freedom).cdf(scaled)
        reference = f"scaled_F({size}, {degrees_of_freedom:g})"
    p_value = min(1.0, 2.0 * min(cdf, 1.0 - cdf))
    order, pivoted, full_rank = _pivoted_errors(residual, covariance)
    critical = (
        _NORMAL.quantile(0.5 * (1.0 + level))
        if degrees_of_freedom is None
        else StudentT(degrees_of_freedom).quantile(0.5 * (1.0 + level))
    )
    coverage = sum(1 for e in errors if abs(e) <= critical) / size
    flags: list[str] = []
    for label, values in (("individual", errors), ("pivoted", pivoted)):
        flags.extend(_error_flags(label, values))
    if p_value < alpha:
        direction = "too large" if cdf > 0.5 else "too small"
        flags.append(
            f"Mahalanobis distance {distance:.4g} is {direction} for "
            f"{reference} (two-sided p = {p_value:.3g})"
        )
    band = 3.0 * math.sqrt(level * (1.0 - level) / size)
    if abs(coverage - level) > band:
        flags.append(
            f"coverage {coverage:.3g} differs from level {level:g} by more than "
            f"{band:.3g}"
        )
    if not full_rank:
        flags.append(
            f"predictive covariance is numerically rank deficient (rank {len(order)})"
        )
    return EmulatorValidation(
        errors,
        distance,
        float(size),
        p_value,
        reference,
        order,
        pivoted,
        full_rank,
        coverage,
        level,
        tuple(flags),
        not flags,
    )


def _error_flags(label: str, errors: Sequence[float]) -> list[str]:
    """Return the failure messages of a vector of standardized errors.

    References:
        [REF-BASTOS-OHAGAN-2009]. Screening thresholds, see
        :func:`validate_emulator_moments`.

    """
    flags: list[str] = []
    size = len(errors)
    if size == 0:
        return flags
    largest = max(abs(e) for e in errors)
    if largest > 3.0:
        flags.append(f"{label} errors: |error| = {largest:.3g} exceeds 3")
    probability = 2.0 * (1.0 - _normal_cdf(2.0))
    exceed = sum(1 for e in errors if abs(e) > 2.0)
    limit = size * probability + 3.0 * math.sqrt(size * probability * (1 - probability))
    if exceed > limit:
        flags.append(f"{label} errors: {exceed} of {size} exceed 2 (limit {limit:.3g})")
    return flags


def gp_predictive_moments(
    gp: GaussianProcessRegression,
    points: Sequence[Sequence[float]],
    *,
    include_noise: bool = False,
) -> tuple[Vector, tuple[Vector, ...]]:
    """Return the joint predictive mean and covariance of a fitted GP.

    ``m = mu + s k_*' alpha`` and ``V = s^2 (K_** - K_*' K^-1 K_*)`` in original
    units, with the fitted hyperparameters of ``gp``; the diagonal noise
    ``s^2 * gp.standardized_noise_variance`` is added to the diagonal with
    ``include_noise``.

    Args:
        gp: A fitted :class:`GaussianProcessRegression`.
        points: Prediction inputs.
        include_noise: Add the noise variance (predict new observations).

    Returns:
        The mean vector and the covariance matrix.

    Raises:
        TypeError: If ``gp`` is not a :class:`GaussianProcessRegression`.
        ValueError: If a point has the wrong dimension.

    Equation:
        [EQ-SUR-GPB-003].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Eq. 2.19].

    """
    if not isinstance(gp, GaussianProcessRegression):
        raise TypeError("gp must be a GaussianProcessRegression")
    xs = tuple(as_vector(p, name="point") for p in points)
    dimension = len(gp.points[0])
    if not xs or any(len(p) != dimension for p in xs):
        raise ValueError("points must be non-empty with the training dimension")
    theta = gp.log_parameters
    targets = tuple((v - gp.output_mean) / gp.output_scale for v in gp.values)
    core = _condition(
        gp.kernel,
        gp.points,
        targets,
        math.exp(theta[0]),
        tuple(math.exp(v) for v in theta[1 : 1 + dimension]),
        gp.standardized_noise_variance,
    )
    mean, covariance = _predict_joint(core, xs, include_noise=include_noise)
    scale = gp.output_scale
    return (
        tuple(gp.output_mean + scale * v for v in mean),
        tuple(tuple(scale**2 * v for v in row) for row in covariance),
    )


def validate_gp_emulator(
    gp: GaussianProcessRegression,
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    *,
    include_noise: bool = False,
    level: float = 0.95,
    alpha: float = 0.05,
) -> EmulatorValidation:
    """Validate a fitted GP on held-out runs.

    Args:
        gp: A fitted :class:`GaussianProcessRegression`.
        points: Validation inputs.
        values: Validation outputs.
        include_noise: Treat validation outputs as noisy observations.
        level: Nominal credible level.
        alpha: Significance level of the Mahalanobis check.

    Returns:
        See :func:`validate_emulator_moments`; the hyperparameters are plug-in
        values, so the chi-square reference is approximate.

    Raises:
        ValueError: If sizes disagree (further checks as in
            :func:`validate_emulator_moments`).

    Equation:
        [EQ-SUR-GPB-003], [EQ-SUR-GPB-004], [EQ-SUR-GPB-005].

    References:
        [REF-BASTOS-OHAGAN-2009].

    """
    if len(points) != len(values):
        raise ValueError("points and values must have the same length")
    mean, covariance = gp_predictive_moments(gp, points, include_noise=include_noise)
    return validate_emulator_moments(values, mean, covariance, level=level, alpha=alpha)


# ---------------------------------------------------------------------------
# 3. GP-based Sobol indices with emulator uncertainty
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GPSobolResult:
    """Distribution of Sobol indices over GP posterior sample paths.

    Args:
        first_order_samples: One first-order index vector per path.
        total_samples: One total index vector per path.
        first_order_mean: Mean over paths of each first-order index.
        first_order_std: Standard deviation over paths.
        first_order_interval: Central 95 % percentile interval per input.
        total_mean: Mean over paths of each total index.
        total_std: Standard deviation over paths.
        total_interval: Central 95 % percentile interval per input.
        mean_path_first_order: Estimator applied to the posterior mean surface.
        mean_path_total: Same for total indices.
        evaluation_points: Number of design points per path (``N (d + 2)``).
        design: The design points in the stacking order ``A, B, AB_1..d``;
            evaluating the true function there and applying
            :func:`sobol_estimates` isolates the emulator error from the
            Monte Carlo error of the design.

    Equation:
        [EQ-SUR-GPB-006].

    References:
        [REF-OAKLEY-OHAGAN-2004], [REF-MARREL-ETAL-2009].

    """

    first_order_samples: tuple[Vector, ...]
    total_samples: tuple[Vector, ...]
    first_order_mean: Vector
    first_order_std: Vector
    first_order_interval: tuple[tuple[float, float], ...]
    total_mean: Vector
    total_std: Vector
    total_interval: tuple[tuple[float, float], ...]
    mean_path_first_order: Vector
    mean_path_total: Vector
    evaluation_points: int
    design: tuple[Vector, ...]


def sobol_estimates(
    values: Sequence[float], count: int, dimension: int
) -> tuple[Vector, Vector]:
    """Return Saltelli first-order and Jansen total indices for one function.

    ``values`` stacks ``f(A)``, ``f(B)`` and ``f(AB_1..d)``, each block of
    ``count`` rows (the order of :attr:`GPSobolResult.design`).

    Args:
        values: Function values on the stacked design.
        count: Rows ``N`` of ``A`` and ``B``.
        dimension: Number of inputs ``d``.

    Returns:
        First-order indices (Saltelli et al. 2010) and total indices (Jansen
        1999).

    Raises:
        ValueError: If the sample variance is not positive.

    Equation:
        [EQ-SUR-GPB-006].

    References:
        [REF-SALTELLI-2010], [REF-JANSEN-1999].

    """
    f_a = values[:count]
    f_b = values[count : 2 * count]
    pooled = [*f_a, *f_b]
    center = math.fsum(pooled) / len(pooled)
    variance = math.fsum((v - center) ** 2 for v in pooled) / len(pooled)
    if variance <= 0.0:
        raise ValueError("sample path has zero variance")
    first: list[float] = []
    total: list[float] = []
    for index in range(dimension):
        start = (2 + index) * count
        f_ab = values[start : start + count]
        first.append(
            math.fsum(b * (ab - a) for a, b, ab in zip(f_a, f_b, f_ab, strict=True))
            / count
            / variance
        )
        total.append(
            math.fsum((a - ab) ** 2 for a, ab in zip(f_a, f_ab, strict=True))
            / (2.0 * count)
            / variance
        )
    return tuple(first), tuple(total)


def _summarize(
    samples: Sequence[Vector],
) -> tuple[Vector, Vector, tuple[tuple[float, float], ...]]:
    """Return per-coordinate mean, standard deviation and 95 % interval.

    References:
        [REF-SALTELLI-2008, Chapter 4]. No equation is evaluated.

    """
    paths = len(samples)
    means: list[float] = []
    stds: list[float] = []
    intervals: list[tuple[float, float]] = []
    for column in zip(*samples, strict=True):
        center = math.fsum(column) / paths
        means.append(center)
        stds.append(
            math.sqrt(math.fsum((c - center) ** 2 for c in column) / (paths - 1))
        )
        ordered = sorted(column)
        intervals.append((_percentile(ordered, 0.025), _percentile(ordered, 0.975)))
    return tuple(means), tuple(stds), tuple(intervals)


def _latin_hypercube(
    count: int, dimension: int, generator: random.Random
) -> list[list[float]]:
    """Return a random Latin hypercube sample in the unit cube.

    References:
        [REF-SALTELLI-2008, Chapter 4]. No equation is evaluated.

    """
    columns: list[list[float]] = []
    for _ in range(dimension):
        strata = list(range(count))
        generator.shuffle(strata)
        columns.append([(k + generator.random()) / count for k in strata])
    return [[column[i] for column in columns] for i in range(count)]


def gp_sobol_posterior(
    gp: GaussianProcessRegression,
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    base_samples: int = 48,
    paths: int = 100,
    relative_jitter: float = 1e-8,
    max_points: int = 600,
    seed: int | RandomStream | None = None,
) -> GPSobolResult:
    """Sobol indices of a GP emulator including the emulator uncertainty.

    Inputs are independent and uniform on the box ``[lower, upper]``. A
    Saltelli design of two independent Latin hypercube matrices ``A``, ``B``
    (``N`` rows) and
    the ``d`` hybrids ``AB_i`` (``A`` with column ``i`` from ``B``) gives
    ``N (d + 2)`` points. For each of ``paths`` joint draws of the GP posterior
    at these points (``f = m + L z`` with ``L L' = V + jitter I``) the
    estimators

    ``S_i = mean(f(B)(f(AB_i) - f(A))) / Var(f)`` (Saltelli et al. 2010) and
    ``ST_i = mean((f(A) - f(AB_i))^2) / (2 Var(f))`` (Jansen 1999)

    are evaluated, with ``Var`` the variance of the pooled ``f(A)``, ``f(B)``
    [EQ-SUR-GPB-006]. The design is shared by all paths, so the spread of each
    index across paths is the emulator uncertainty at this design; the Monte
    Carlo error of the design itself is common to all paths and shrinks with
    ``base_samples``.

    Args:
        gp: A fitted :class:`GaussianProcessRegression`.
        lower: Lower bounds of the input box.
        upper: Upper bounds (strictly above ``lower``).
        base_samples: Rows ``N`` of ``A`` and ``B``.
        paths: Number of posterior sample paths.
        relative_jitter: Explicit diagonal jitter of the posterior covariance,
            relative to ``s^2 sigma_f^2`` (the posterior covariance is only
            positive semi-definite).
        max_points: Largest allowed ``N (d + 2)`` (the joint Cholesky factor is
            dense).
        seed: Integer seed or stream.

    Returns:
        Per-path samples and summaries of both index families.

    Raises:
        TypeError: If ``gp`` is not a :class:`GaussianProcessRegression`.
        ValueError: For invalid bounds or sizes, a design larger than
            ``max_points``, or a joint covariance that is not positive definite
            even with the jitter (increase ``relative_jitter``).

    Equation:
        [EQ-SUR-GPB-006].

    References:
        [REF-OAKLEY-OHAGAN-2004], [REF-MARREL-ETAL-2009],
        [REF-SALTELLI-2010], [REF-JANSEN-1999].

    """
    if not isinstance(gp, GaussianProcessRegression):
        raise TypeError("gp must be a GaussianProcessRegression")
    low = as_vector(lower, name="lower")
    high = as_vector(upper, name="upper")
    dimension = len(gp.points[0])
    if len(low) != dimension or len(high) != dimension:
        raise ValueError("bounds must have the training input dimension")
    if any(b <= a for a, b in zip(low, high, strict=True)):
        raise ValueError("upper must exceed lower in every coordinate")
    if base_samples < 8 or paths < 2:
        raise ValueError("base_samples must be >= 8 and paths >= 2")
    if relative_jitter <= 0.0:
        raise ValueError("relative_jitter must be positive")
    total_points = base_samples * (dimension + 2)
    if total_points > max_points:
        raise ValueError(f"design has {total_points} points; max_points={max_points}")
    generator = resolve_stream(seed).generator()
    width = [b - a for a, b in zip(low, high, strict=True)]
    matrix_a = _latin_hypercube(base_samples, dimension, generator)
    matrix_b = _latin_hypercube(base_samples, dimension, generator)
    unit = [*matrix_a, *matrix_b]
    for index in range(dimension):
        for row_a, row_b in zip(matrix_a, matrix_b, strict=True):
            hybrid = list(row_a)
            hybrid[index] = row_b[index]
            unit.append(hybrid)
    design = [
        tuple(a + w * u for a, w, u in zip(low, width, row, strict=True))
        for row in unit
    ]
    mean, covariance = gp_predictive_moments(gp, design)
    signal_scale = gp.output_scale**2 * math.exp(gp.log_parameters[0])
    shifted = [list(row) for row in covariance]
    for i in range(total_points):
        shifted[i][i] += relative_jitter * signal_scale
    factor = _cholesky(shifted)
    first_samples: list[Vector] = []
    total_samples: list[Vector] = []
    for _ in range(paths):
        noise = [generator.gauss(0.0, 1.0) for _ in range(total_points)]
        path = [
            m + _dot(row, noise[: len(row)])
            for m, row in zip(mean, factor, strict=True)
        ]
        first, total = sobol_estimates(path, base_samples, dimension)
        first_samples.append(first)
        total_samples.append(total)
    first_mean, first_std, first_interval = _summarize(first_samples)
    total_mean, total_std, total_interval = _summarize(total_samples)
    mean_first, mean_total = sobol_estimates(mean, base_samples, dimension)
    return GPSobolResult(
        tuple(first_samples),
        tuple(total_samples),
        first_mean,
        first_std,
        first_interval,
        total_mean,
        total_std,
        total_interval,
        mean_first,
        mean_total,
        total_points,
        tuple(design),
    )


# ---------------------------------------------------------------------------
# 4. GP classification (Laplace approximation)
# ---------------------------------------------------------------------------


def _log_sigmoid(value: float) -> float:
    """Return ``log(1 / (1 + exp(-value)))`` without overflow.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 3.4]. No equation is evaluated.

    """
    if value >= 0.0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def _sigmoid(value: float) -> float:
    """Return the logistic function without overflow.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 3.4]. No equation is evaluated.

    """
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


@dataclass(frozen=True, slots=True)
class _LaplaceMode:
    """Newton result of the Laplace approximation (internal).

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 3.1].

    """

    latent: list[float]
    alpha: list[float]
    lower: list[list[float]]
    sqrt_weights: list[float]
    log_marginal: float
    iterations: int
    converged: bool


def _laplace_mode(
    kernel_matrix: Sequence[Sequence[float]],
    signs: Sequence[float],
    tolerance: float,
    max_iterations: int,
) -> _LaplaceMode:
    """Run Newton's method of Rasmussen and Williams Algorithm 3.1.

    Equation:
        [EQ-SUR-GPB-007].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 3.1].

    """
    count = len(signs)
    targets = [(s + 1.0) / 2.0 for s in signs]
    latent = [0.0] * count
    alpha = [0.0] * count
    objective = -sum(math.log(2.0) for _ in range(count))
    iterations = 0
    converged = False
    sqrt_weights = [0.0] * count
    lower: list[list[float]] = []
    while iterations < max_iterations:
        iterations += 1
        probabilities = [_sigmoid(f) for f in latent]
        weights = [p * (1.0 - p) for p in probabilities]
        sqrt_weights = [math.sqrt(w) for w in weights]
        system = [
            [
                (1.0 if i == j else 0.0)
                + sqrt_weights[i] * kernel_matrix[i][j] * sqrt_weights[j]
                for j in range(i + 1)
            ]
            for i in range(count)
        ]
        lower = _cholesky(system)
        shift = [
            w * f + (t - p)
            for w, f, t, p in zip(weights, latent, targets, probabilities, strict=True)
        ]
        scaled = [sqrt_weights[i] * _dot(kernel_matrix[i], shift) for i in range(count)]
        solved = _backward(lower, _forward(lower, scaled))
        alpha = [
            b - sw * s for b, sw, s in zip(shift, sqrt_weights, solved, strict=True)
        ]
        latent = [_dot(kernel_matrix[i], alpha) for i in range(count)]
        new_objective = -0.5 * _dot(alpha, latent) + math.fsum(
            _log_sigmoid(s * f) for s, f in zip(signs, latent, strict=True)
        )
        change = abs(new_objective - objective)
        objective = new_objective
        if change < tolerance:
            converged = True
            break
    probabilities = [_sigmoid(f) for f in latent]
    weights = [p * (1.0 - p) for p in probabilities]
    sqrt_weights = [math.sqrt(w) for w in weights]
    system = [
        [
            (1.0 if i == j else 0.0)
            + sqrt_weights[i] * kernel_matrix[i][j] * sqrt_weights[j]
            for j in range(i + 1)
        ]
        for i in range(count)
    ]
    lower = _cholesky(system)
    log_marginal = objective - math.fsum(math.log(row[-1]) for row in lower)
    return _LaplaceMode(
        latent, alpha, lower, sqrt_weights, log_marginal, iterations, converged
    )


def _kernel_matrix(
    kernel: str,
    points: Sequence[Vector],
    scales: Vector,
    signal: float,
    nugget: float,
) -> list[list[float]]:
    """Return the full symmetric kernel matrix plus ``nugget`` on the diagonal.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2]. No equation is evaluated.

    """
    count = len(points)
    matrix = [[0.0] * count for _ in range(count)]
    for i in range(count):
        for j in range(i + 1):
            value = kernel_value(kernel, points[i], points[j], scales, signal)
            matrix[i][j] = value
            matrix[j][i] = value
        matrix[i][i] += nugget
    return matrix


@dataclass(frozen=True, slots=True)
class LaplaceGPClassifier:
    """Binary GP classifier with a logistic likelihood (Laplace approximation).

    Build it with :func:`fit_laplace_gp_classifier`.

    Args:
        kernel: Kernel name.
        points: Training inputs.
        labels: Training labels in ``{0, 1}``.
        length_scales: ARD length scales used.
        signal_variance: Signal variance used.
        nugget: Explicit diagonal variance added to the kernel matrix.
        latent_mode: Posterior mode ``f_hat`` of the latent values.
        log_marginal_likelihood: Laplace approximation of ``log p(y | X)``.
        newton_iterations: Newton iterations used.
        converged: Whether the objective change fell below the tolerance.

    Equation:
        [EQ-SUR-GPB-007], [EQ-SUR-GPB-008].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 3].

    """

    kernel: str
    points: tuple[Vector, ...]
    labels: tuple[int, ...]
    length_scales: Vector
    signal_variance: float
    nugget: float
    latent_mode: Vector
    log_marginal_likelihood: float
    newton_iterations: int
    converged: bool
    _lower: list[list[float]] = field(init=False, repr=False, compare=False)
    _sqrt_weights: list[float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Rebuild the factor ``B = I + W^1/2 K W^1/2`` at the stored mode.

        Raises:
            ValueError: If ``B`` is not positive definite.

        Equation:
            [EQ-SUR-GPB-008].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 3.2].

        """
        kernel_matrix = _kernel_matrix(
            self.kernel,
            self.points,
            self.length_scales,
            self.signal_variance,
            self.nugget,
        )
        weights = [_sigmoid(f) * (1.0 - _sigmoid(f)) for f in self.latent_mode]
        roots = [math.sqrt(w) for w in weights]
        system = [
            [
                (1.0 if i == j else 0.0) + roots[i] * kernel_matrix[i][j] * roots[j]
                for j in range(i + 1)
            ]
            for i in range(len(roots))
        ]
        object.__setattr__(self, "_lower", _cholesky(system))
        object.__setattr__(self, "_sqrt_weights", roots)

    def predict_latent(self, x: Sequence[float]) -> tuple[float, float]:
        """Return the Gaussian predictive mean and variance of the latent value.

        ``f_bar = k_*' (t - pi_hat)`` and ``v = k_** - (L^-1 W^1/2 k_*)'
        (L^-1 W^1/2 k_*)`` (Algorithm 3.2); the kernel nugget is not added to
        ``k_**``.

        Args:
            x: Input vector.

        Returns:
            Latent predictive mean and variance.

        Raises:
            ValueError: If ``x`` has the wrong dimension.

        Equation:
            [EQ-SUR-GPB-008].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 3.2].

        """
        point = as_vector(x, name="x")
        if len(point) != len(self.points[0]):
            raise ValueError("x must have the training input dimension")
        cross = [
            kernel_value(
                self.kernel, point, train, self.length_scales, self.signal_variance
            )
            for train in self.points
        ]
        gradient = [
            label - _sigmoid(f)
            for label, f in zip(self.labels, self.latent_mode, strict=True)
        ]
        mean = _dot(cross, gradient)
        whitened = _forward(
            self._lower, [r * c for r, c in zip(self._sqrt_weights, cross, strict=True)]
        )
        return mean, max(self.signal_variance - _dot(whitened, whitened), 0.0)

    def predict_proba(self, x: Sequence[float]) -> float:
        """Return the predictive probability of class one at ``x``.

        ``pi_* = E[sigmoid(f_*)]`` with ``f_* ~ N(f_bar, v)`` evaluated by
        composite Simpson quadrature over ``f_bar +/- 9 sqrt(v)`` with 600
        intervals (absolute error below ``1e-9`` for ``v <= 100``).

        Args:
            x: Input vector.

        Returns:
            Probability in ``(0, 1)``.

        Equation:
            [EQ-SUR-GPB-008].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 3.2].

        """
        mean, variance = self.predict_latent(x)
        if variance <= 0.0:
            return _sigmoid(mean)
        spread = math.sqrt(variance)
        intervals = 600
        step = 18.0 * spread / intervals
        total = 0.0
        for k in range(intervals + 1):
            z = -9.0 + 18.0 * k / intervals
            weight = 1.0 if k in (0, intervals) else (4.0 if k % 2 else 2.0)
            total += weight * _sigmoid(mean + spread * z) * math.exp(-0.5 * z * z)
        return total * (step / spread) / 3.0 / math.sqrt(2.0 * math.pi)

    def predict_class(self, x: Sequence[float], threshold: float = 0.5) -> int:
        """Return ``1`` when the predictive probability reaches ``threshold``.

        Args:
            x: Input vector.
            threshold: Probability cut in ``(0, 1)``.

        Returns:
            The class label ``0`` or ``1``.

        Raises:
            ValueError: If ``threshold`` is not in ``(0, 1)``.

        Equation:
            [EQ-SUR-GPB-008].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Section 3.4].

        """
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        return 1 if self.predict_proba(x) >= threshold else 0


def fit_laplace_gp_classifier(
    points: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    kernel: str = "squared_exponential",
    length_scales: Sequence[float] | None = None,
    signal_variance: float = 1.0,
    optimize: bool = False,
    nugget: float = 1e-8,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> LaplaceGPClassifier:
    """Fit a binary GP classifier with a logistic likelihood.

    The latent posterior ``p(f | X, y) ~ p(y | f) N(f; 0, K)`` is replaced by
    the Gaussian centred at its mode, found by the Newton iteration of
    Algorithm 3.1 (the logistic log-likelihood is concave in ``f``, so the
    Newton step is well defined). With ``optimize=True`` the log length scales
    and log signal variance maximize the Laplace marginal likelihood
    [EQ-SUR-GPB-007] by Nelder-Mead inside ``|log| <= 5``.

    Args:
        points: Training inputs.
        labels: Class labels, ``0`` or ``1`` (``-1`` is read as ``0``); both
            classes must occur.
        kernel: Kernel name.
        length_scales: Length scales (one per input); default ``1`` each, or
            the starting point when ``optimize`` is true.
        signal_variance: Signal variance (starting value when optimizing).
        optimize: Maximize the approximate marginal likelihood.
        nugget: Explicit diagonal variance of the kernel matrix.
        tolerance: Newton stop on the change of the objective.
        max_iterations: Maximum Newton iterations.

    Returns:
        The fitted classifier; ``converged`` is ``False`` when the iteration
        limit was hit.

    Raises:
        ValueError: For invalid labels (a single class), non-positive scales or
            nugget, or a non-positive-definite system.

    Equation:
        [EQ-SUR-GPB-007], [EQ-SUR-GPB-008].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 3].

    """
    _check_kernel(kernel)
    x = _check_points(points, len(labels))
    codes: list[int] = []
    for label in labels:
        if label not in (-1, 0, 1):
            raise ValueError("labels must be 0/1 (or -1/+1)")
        codes.append(1 if label == 1 else 0)
    if len(set(codes)) < 2:
        raise ValueError("both classes must be present")
    if nugget <= 0.0 or signal_variance <= 0.0 or tolerance <= 0.0:
        raise ValueError("nugget, signal_variance and tolerance must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    dimension = len(x[0])
    scales = (
        (1.0,) * dimension
        if length_scales is None
        else as_vector(length_scales, name="length_scales")
    )
    if len(scales) != dimension or min(scales) <= 0.0:
        raise ValueError("length_scales must be positive, one per input")
    signs = [2.0 * c - 1.0 for c in codes]

    if optimize:
        start = (*(math.log(s) for s in scales), math.log(signal_variance))
        objective = _ClassifierObjective(
            kernel, x, signs, dimension, nugget, tolerance, max_iterations
        )
        result = minimize_nelder_mead(
            objective,
            start,
            initial_step=0.3,
            tolerance=1e-8,
            max_evaluations=600,
            bounds=((-5.0, 5.0),) * (dimension + 1),
        )
        scales = tuple(math.exp(v) for v in result.x[:dimension])
        signal_variance = math.exp(result.x[dimension])
    matrix = _kernel_matrix(kernel, x, scales, signal_variance, nugget)
    mode = _laplace_mode(matrix, signs, tolerance, max_iterations)
    return LaplaceGPClassifier(
        kernel,
        x,
        tuple(codes),
        scales,
        signal_variance,
        nugget,
        tuple(mode.latent),
        mode.log_marginal,
        mode.iterations,
        mode.converged,
    )


class _ClassifierObjective:
    """Negative Laplace log marginal likelihood in log hyperparameters.

    Equation:
        [EQ-SUR-GPB-007].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Eq. 3.32].

    """

    def __init__(
        self,
        kernel: str,
        points: tuple[Vector, ...],
        signs: list[float],
        dimension: int,
        nugget: float,
        tolerance: float,
        max_iterations: int,
    ) -> None:
        """Store the problem data.

        References:
            [EQ-SUR-GPB-007].

        """
        self.kernel = kernel
        self.points = points
        self.signs = signs
        self.dimension = dimension
        self.nugget = nugget
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def __call__(self, theta: Sequence[float]) -> float:
        """Return ``-log q(y | X, theta)``; ``+inf`` if the system fails.

        References:
            [EQ-SUR-GPB-007].

        """
        scales = tuple(math.exp(v) for v in theta[: self.dimension])
        matrix = _kernel_matrix(
            self.kernel,
            self.points,
            scales,
            math.exp(theta[self.dimension]),
            self.nugget,
        )
        try:
            mode = _laplace_mode(
                matrix, self.signs, self.tolerance, self.max_iterations
            )
        except ValueError:
            return math.inf
        return -mode.log_marginal


# ---------------------------------------------------------------------------
# 5. Student-t process regression
# ---------------------------------------------------------------------------


def _student_t_log_likelihood(
    count: int, half_log_det: float, quad: float, nu: float
) -> float:
    """Return the multivariate Student-t log density (covariance form).

    ``log Gamma((nu+n)/2) - log Gamma(nu/2) - n/2 log((nu-2) pi) - 1/2 log|K|
    - (nu+n)/2 log(1 + beta/(nu-2))``.

    Equation:
        [EQ-SUR-GPB-009].

    References:
        [REF-SHAH-WILSON-GHAHRAMANI-2014].

    """
    return (
        math.lgamma(0.5 * (nu + count))
        - math.lgamma(0.5 * nu)
        - 0.5 * count * math.log((nu - 2.0) * math.pi)
        - half_log_det
        - 0.5 * (nu + count) * math.log1p(quad / (nu - 2.0))
    )


def student_t_process_log_marginal_likelihood(
    points: Sequence[Sequence[float]],
    targets: Sequence[float],
    *,
    nu: float,
    signal_variance: float,
    length_scales: Sequence[float],
    noise_variance: float,
    kernel: str = "matern52",
) -> float:
    """Return the Student-t process log marginal likelihood of the targets.

    ``y ~ MVT_n(nu, 0, K)`` with ``K`` the covariance (including the noise
    variance on the diagonal): ``log p(y) = log Gamma((nu+n)/2) - log
    Gamma(nu/2) - (n/2) log((nu-2) pi) - (1/2) log|K| - ((nu+n)/2) log(1 +
    beta/(nu-2))`` with ``beta = y' K^-1 y`` [EQ-SUR-GPB-009]. As ``nu`` grows
    it tends to the Gaussian log density.

    Args:
        points: Inputs.
        targets: Targets (used as given; no standardization).
        nu: Degrees of freedom, ``nu > 2``.
        signal_variance: Kernel signal variance.
        length_scales: ARD length scales.
        noise_variance: Diagonal variance added to the kernel matrix.
        kernel: Kernel name.

    Returns:
        The log density.

    Raises:
        ValueError: For ``nu <= 2``, invalid sizes or parameters, or a matrix
            that is not positive definite.

    Equation:
        [EQ-SUR-GPB-009].

    References:
        [REF-SHAH-WILSON-GHAHRAMANI-2014].

    """
    _check_kernel(kernel)
    y = as_vector(targets, name="targets")
    x = _check_points(points, len(y))
    scales = as_vector(length_scales, name="length_scales")
    if not nu > 2.0 or not math.isfinite(nu):
        raise ValueError("nu must be finite and greater than 2")
    if len(scales) != len(x[0]) or min(scales) <= 0.0:
        raise ValueError("length_scales must be positive, one per input")
    if signal_variance <= 0.0 or noise_variance <= 0.0:
        raise ValueError("signal_variance and noise_variance must be positive")
    core = _condition(kernel, x, y, signal_variance, scales, noise_variance)
    return _student_t_log_likelihood(len(y), core.half_log_det, core.quad, nu)


@dataclass(frozen=True, slots=True)
class StudentTProcessRegression:
    """Student-t process regression with plug-in kernel hyperparameters.

    Build it with :func:`fit_student_t_process`. Outputs are standardized by
    their sample mean and standard deviation before the likelihood is formed.

    Args:
        kernel: Kernel name.
        points: Training inputs.
        values: Training outputs (original units).
        nu: Degrees of freedom ``nu > 2``.
        log_parameters: ``theta = (log sigma_f^2, log ell_1..d, log
            sigma_n^2)`` in standardized units.
        nugget: Explicit diagonal variance (standardized units).
        output_mean: Centering mean.
        output_scale: Scale (sample standard deviation).
        log_likelihood: Student-t log marginal likelihood in original units.
        beta: ``y' K^-1 y`` of the standardized targets.

    Equation:
        [EQ-SUR-GPB-009], [EQ-SUR-GPB-010].

    References:
        [REF-SHAH-WILSON-GHAHRAMANI-2014].

    """

    kernel: str
    points: tuple[Vector, ...]
    values: Vector
    nu: float
    log_parameters: Vector
    nugget: float
    output_mean: float
    output_scale: float
    log_likelihood: float
    beta: float
    _core: _Conditioned = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Condition the underlying Gaussian system at the stored parameters.

        Raises:
            ValueError: If the kernel matrix is not positive definite.

        References:
            [REF-SHAH-WILSON-GHAHRAMANI-2014].

        """
        dimension = len(self.points[0])
        theta = self.log_parameters
        targets = tuple((v - self.output_mean) / self.output_scale for v in self.values)
        core = _condition(
            self.kernel,
            self.points,
            targets,
            math.exp(theta[0]),
            tuple(math.exp(v) for v in theta[1 : 1 + dimension]),
            self.nugget + math.exp(theta[-1]),
        )
        object.__setattr__(self, "_core", core)

    def variance_scaling(self) -> float:
        """Return the data-dependent factor ``(nu + beta - 2) / (nu + n - 2)``.

        Returns:
            The multiplier of the Gaussian-process predictive variance.

        Equation:
            [EQ-SUR-GPB-010].

        References:
            [REF-SHAH-WILSON-GHAHRAMANI-2014].

        """
        count = len(self.points)
        return (self.nu + self.beta - 2.0) / (self.nu + count - 2.0)

    def predict_with_variance(
        self, x: Sequence[float], *, include_noise: bool = False
    ) -> tuple[float, float]:
        """Return the predictive mean and covariance-form variance at ``x``.

        The predictive law is ``MVT(nu + n, mu_*, (nu + beta - 2) / (nu + n - 2)
        Sigma_*)`` with the Gaussian-process moments ``mu_*``, ``Sigma_*``
        (original units after the scaling by ``s``).

        Args:
            x: Input vector.
            include_noise: Add the noise variance to ``Sigma_*``.

        Returns:
            Mean and variance in original units.

        Raises:
            ValueError: If ``x`` has the wrong dimension.

        Equation:
            [EQ-SUR-GPB-010].

        References:
            [REF-SHAH-WILSON-GHAHRAMANI-2014].

        """
        point = as_vector(x, name="x")
        if len(point) != len(self.points[0]):
            raise ValueError("x must have the training input dimension")
        mean, variance = _predict_point(self._core, point)
        if include_noise:
            variance += self._core.noise
        return (
            self.output_mean + self.output_scale * mean,
            self.output_scale**2 * self.variance_scaling() * variance,
        )

    def predictive_interval(
        self, x: Sequence[float], level: float = 0.95, *, include_noise: bool = False
    ) -> tuple[float, float]:
        """Return the central ``level`` interval from the Student-t quantile.

        The scale of the ``nu + n`` degrees-of-freedom law is
        ``sqrt(variance (nu + n - 2) / (nu + n))``.

        Args:
            x: Input vector.
            level: Credible level in ``(0, 1)``.
            include_noise: Predict a new noisy observation.

        Returns:
            Lower and upper limits.

        Raises:
            ValueError: If ``level`` is not in ``(0, 1)``.

        Equation:
            [EQ-SUR-GPB-010].

        References:
            [REF-SHAH-WILSON-GHAHRAMANI-2014].

        """
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        mean, variance = self.predict_with_variance(x, include_noise=include_noise)
        dof = self.nu + len(self.points)
        scale = math.sqrt(variance * (dof - 2.0) / dof)
        quantile = StudentT(dof).quantile(0.5 * (1.0 + level))
        return mean - scale * quantile, mean + scale * quantile


def fit_student_t_process(
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    *,
    kernel: str = "matern52",
    nu: float | None = None,
    log_parameters: Sequence[float] | None = None,
    nugget: float = 1e-8,
    seed: int | RandomStream | None = None,
) -> StudentTProcessRegression:
    """Fit a Student-t process with Gaussian-process plug-in hyperparameters.

    The kernel hyperparameters are those of the type-II ML Gaussian-process fit
    (with estimated noise) unless ``log_parameters`` is given; ``nu`` is chosen
    on a fixed grid ``(2.5, 3, 4, 5, 7, 10, 15, 25, 50, 100, 1e3, 1e6)`` by the
    Student-t marginal likelihood unless it is given. This is a profile in
    ``nu`` at fixed kernel parameters, not a joint maximum.

    Args:
        points: Training inputs.
        values: Training outputs (not constant, at least three).
        kernel: Kernel name.
        nu: Degrees of freedom ``> 2``, or ``None`` to select from the grid.
        log_parameters: Optional ``theta`` in standardized units.
        nugget: Explicit diagonal variance (standardized units).
        seed: Seed for the Gaussian-process fit restarts.

    Returns:
        The fitted regression.

    Raises:
        ValueError: For invalid data, ``nu <= 2``, or a kernel matrix that is
            not positive definite.

    Equation:
        [EQ-SUR-GPB-009], [EQ-SUR-GPB-010].

    References:
        [REF-SHAH-WILSON-GHAHRAMANI-2014].

    """
    _check_kernel(kernel)
    mean, spread, targets = _standardize(values)
    x = _check_points(points, len(targets))
    if nu is not None and (not nu > 2.0 or not math.isfinite(nu)):
        raise ValueError("nu must be finite and greater than 2")
    if log_parameters is None:
        gp = fit_gaussian_process(
            x,
            values,
            kernel=kernel,
            nugget=nugget,
            estimate_noise=True,
            seed=seed,
        )
        theta = gp.log_parameters
    else:
        theta = as_vector(log_parameters, name="log_parameters")
        if len(theta) != len(x[0]) + 2:
            raise ValueError("log_parameters must have length d + 2")
    dimension = len(x[0])
    core = _condition(
        kernel,
        x,
        targets,
        math.exp(theta[0]),
        tuple(math.exp(v) for v in theta[1 : 1 + dimension]),
        nugget + math.exp(theta[-1]),
    )
    count = len(x)
    if nu is None:
        nu = max(
            _NU_GRID,
            key=lambda n: _student_t_log_likelihood(
                count, core.half_log_det, core.quad, n
            ),
        )
    log_likelihood = _student_t_log_likelihood(
        count, core.half_log_det, core.quad, nu
    ) - count * math.log(spread)
    return StudentTProcessRegression(
        kernel,
        x,
        as_vector(values, name="values"),
        float(nu),
        tuple(theta),
        nugget,
        mean,
        spread,
        log_likelihood,
        core.quad,
    )


# ---------------------------------------------------------------------------
# 6. Output-warped GP (Box-Cox)
# ---------------------------------------------------------------------------


def box_cox(value: float, warp_lambda: float) -> float:
    """Return the Box-Cox transform ``(y^lambda - 1) / lambda`` (``log y`` at 0).

    Args:
        value: Positive response.
        warp_lambda: Exponent ``lambda``.

    Returns:
        The warped value ``z``.

    Raises:
        ValueError: If ``value`` is not positive and finite.

    Equation:
        [EQ-SUR-GPB-011].

    References:
        [REF-BOX-COX-1964], [REF-SNELSON-RASMUSSEN-GHAHRAMANI-2004].

    """
    if not value > 0.0 or not math.isfinite(value):
        raise ValueError("value must be positive and finite")
    if abs(warp_lambda) < 1e-12:
        return math.log(value)
    return math.expm1(warp_lambda * math.log(value)) / warp_lambda


def inverse_box_cox(warped: float, warp_lambda: float) -> float:
    """Return the inverse Box-Cox transform ``(lambda z + 1)^(1/lambda)``.

    Args:
        warped: Warped value ``z``.
        warp_lambda: Exponent ``lambda``.

    Returns:
        The response ``y > 0``.

    Raises:
        ValueError: If ``lambda z + 1 <= 0``, i.e. ``z`` lies outside the image
            of the warp (no positive ``y`` maps to it).

    Equation:
        [EQ-SUR-GPB-012].

    References:
        [REF-BOX-COX-1964], [REF-SNELSON-RASMUSSEN-GHAHRAMANI-2004].

    """
    if abs(warp_lambda) < 1e-12:
        return math.exp(warped)
    base = warp_lambda * warped + 1.0
    if base <= 0.0:
        raise ValueError("warped value lies outside the image of the Box-Cox map")
    return math.exp(math.log(base) / warp_lambda)


@dataclass(frozen=True, slots=True)
class WarpedGPRegression:
    """GP on Box-Cox warped outputs; predictions are mapped back to ``y``.

    Build it with :func:`fit_warped_gp`.

    Args:
        warp_lambda: Box-Cox exponent.
        gp: Gaussian process fitted to ``z = BC(y; lambda)``.
        values: Positive training outputs.
        log_likelihood: ``log p(y)`` including the Jacobian ``sum (lambda - 1)
            log y_i`` [EQ-SUR-GPB-011].
        lambda_searched: Whether ``lambda`` was optimized (``False`` if given).

    Equation:
        [EQ-SUR-GPB-011], [EQ-SUR-GPB-012].

    References:
        [REF-SNELSON-RASMUSSEN-GHAHRAMANI-2004], [REF-BOX-COX-1964].

    """

    warp_lambda: float
    gp: GaussianProcessRegression
    values: Vector
    log_likelihood: float
    lambda_searched: bool

    def predict_latent(
        self, x: Sequence[float], *, include_noise: bool = False
    ) -> tuple[float, float]:
        """Return the Gaussian predictive mean and variance of ``z`` at ``x``.

        Args:
            x: Input vector.
            include_noise: Predict a new noisy observation in warped space.

        Returns:
            Mean and variance of the warped response.

        Equation:
            [EQ-SUR-GPB-012].

        References:
            [REF-SNELSON-RASMUSSEN-GHAHRAMANI-2004].

        """
        return self.gp.predict_with_variance(x, include_noise=include_noise)

    def quantile(
        self, x: Sequence[float], probability: float, *, include_noise: bool = False
    ) -> float:
        """Return the predictive quantile of ``y`` through the inverse warp.

        Because the warp is increasing, ``q_y(p) = BC^-1(mu + sigma
        Phi^-1(p))`` [EQ-SUR-GPB-012].

        Args:
            x: Input vector.
            probability: Probability in ``(0, 1)``.
            include_noise: Predict a new noisy observation.

        Returns:
            The quantile of the positive response.

        Raises:
            ValueError: If ``probability`` is not in ``(0, 1)`` or the warped
                quantile lies outside the image of the warp (see
                :func:`inverse_box_cox`).

        Equation:
            [EQ-SUR-GPB-012].

        References:
            [REF-SNELSON-RASMUSSEN-GHAHRAMANI-2004].

        """
        if not 0.0 < probability < 1.0:
            raise ValueError("probability must be in (0, 1)")
        mean, variance = self.predict_latent(x, include_noise=include_noise)
        z = mean + math.sqrt(variance) * _NORMAL.quantile(probability)
        return inverse_box_cox(z, self.warp_lambda)

    def median(self, x: Sequence[float]) -> float:
        """Return the predictive median ``BC^-1(mu)``.

        Args:
            x: Input vector.

        Returns:
            The median of the positive response.

        Equation:
            [EQ-SUR-GPB-012].

        References:
            [REF-SNELSON-RASMUSSEN-GHAHRAMANI-2004].

        """
        return self.quantile(x, 0.5)


def _golden_maximize(
    objective: _WarpObjective, low: float, high: float, iterations: int
) -> float:
    """Return the maximizer of a unimodal function by golden-section search.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 9.5]. No equation is evaluated.

    """
    a, b = low, high
    c = b - _GOLDEN * (b - a)
    d = a + _GOLDEN * (b - a)
    fc, fd = objective(c), objective(d)
    for _ in range(iterations):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN * (b - a)
            fd = objective(d)
    return 0.5 * (a + b)


class _WarpObjective:
    """Profile log-likelihood of the Box-Cox exponent (with Jacobian).

    Equation:
        [EQ-SUR-GPB-011].

    References:
        [REF-SNELSON-RASMUSSEN-GHAHRAMANI-2004].

    """

    def __init__(
        self,
        points: tuple[Vector, ...],
        values: Vector,
        kernel: str,
        nugget: float,
        seed: int | RandomStream | None,
    ) -> None:
        """Store the data.

        References:
            [EQ-SUR-GPB-011].

        """
        self.points = points
        self.values = values
        self.kernel = kernel
        self.nugget = nugget
        self.seed = seed
        self.log_sum = math.fsum(math.log(v) for v in values)

    def fit(self, warp_lambda: float) -> GaussianProcessRegression:
        """Fit the GP to the warped outputs at a fixed exponent.

        References:
            [EQ-SUR-GPB-011].

        """
        warped = [box_cox(v, warp_lambda) for v in self.values]
        return fit_gaussian_process(
            self.points,
            warped,
            kernel=self.kernel,
            nugget=self.nugget,
            estimate_noise=True,
            restarts=2,
            seed=self.seed,
        )

    def total(self, warp_lambda: float, gp: GaussianProcessRegression) -> float:
        """Return GP log marginal likelihood plus the Jacobian term.

        References:
            [EQ-SUR-GPB-011].

        """
        return gp.log_marginal_likelihood + (warp_lambda - 1.0) * self.log_sum

    def __call__(self, warp_lambda: float) -> float:
        """Return the profile log-likelihood at ``warp_lambda``.

        References:
            [EQ-SUR-GPB-011].

        """
        return self.total(warp_lambda, self.fit(warp_lambda))


def fit_warped_gp(
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    *,
    kernel: str = "matern52",
    warp_lambda: float | None = None,
    lambda_bounds: tuple[float, float] = (-2.0, 2.0),
    search_iterations: int = 14,
    nugget: float = 1e-8,
    seed: int | RandomStream | None = None,
) -> WarpedGPRegression:
    """Fit a GP to Box-Cox warped positive outputs.

    With ``z_i = BC(y_i; lambda)`` and ``z ~ N(0-mean GP)``, the density of the
    original outputs carries the Jacobian ``dz/dy = y^(lambda - 1)``:
    ``log p(y) = log p_GP(z) + sum_i (lambda - 1) log y_i`` [EQ-SUR-GPB-011].
    Without a given ``warp_lambda`` the exponent maximizes this profile by
    golden-section search on ``lambda_bounds`` (``search_iterations`` steps,
    assuming a unimodal profile), refitting the GP at each trial.

    Args:
        points: Training inputs.
        values: Strictly positive training outputs.
        kernel: Kernel name.
        warp_lambda: Fixed exponent, or ``None`` to search.
        lambda_bounds: Search interval ``(low, high)``.
        search_iterations: Golden-section iterations.
        nugget: Explicit diagonal variance (standardized units).
        seed: Seed for the Gaussian-process fit restarts.

    Returns:
        The fitted warped GP.

    Raises:
        ValueError: For non-positive outputs, invalid bounds or iterations, or
            kernel failures from the underlying fit.

    Equation:
        [EQ-SUR-GPB-011], [EQ-SUR-GPB-012].

    References:
        [REF-SNELSON-RASMUSSEN-GHAHRAMANI-2004], [REF-BOX-COX-1964].

    """
    _check_kernel(kernel)
    y = as_vector(values, name="values")
    x = _check_points(points, len(y))
    if any(v <= 0.0 for v in y):
        raise ValueError("values must be strictly positive")
    low, high = lambda_bounds
    if not low < high or search_iterations < 1:
        raise ValueError("lambda_bounds must satisfy low < high and iterations >= 1")
    objective = _WarpObjective(x, y, kernel, nugget, seed)
    searched = warp_lambda is None
    exponent = (
        _golden_maximize(objective, low, high, search_iterations)
        if warp_lambda is None
        else float(warp_lambda)
    )
    gp = objective.fit(exponent)
    return WarpedGPRegression(exponent, gp, y, objective.total(exponent, gp), searched)
