"""Kennedy-O'Hagan calibration with a Gaussian-process model discrepancy.

The observation model of :class:`DiscrepancyCalibration` is

``y_i = eta(c_i, theta) + delta(x_i) + eps_i``,
``delta ~ GP(0, k_psi)``, ``eps_i ~ N(0, sigma_i^2)`` independent,

where ``c_i`` are the experimental conditions of experiment ``i``, ``x_i`` the
subset of conditions the discrepancy depends on, ``eta`` the simulator (a
:class:`~probstat.modeling.ProbabilisticModel` whose input law is the prior of
``theta``), and ``psi = (sigma_delta, ell_1..ell_p)`` the discrepancy
hyperparameters. Marginalizing ``delta`` gives the Gaussian likelihood

``y | theta, psi, sigma ~ N(eta(C, theta), K_psi + diag(sigma_i^2))``,

evaluated by one Cholesky factorization per call (``O(n^3)``, so intended for
tens of observations, not thousands). The noise is either known per
observation (:class:`KnownNoise`) or log-linear in covariates,
``sigma_i = exp(a + b' g(c_i))`` (:class:`LogLinearNoise`), estimated jointly.

The simulator is evaluated directly (no emulator), there is no simulator
scale factor ``rho`` and the discrepancy has zero prior mean. These are
simplifications of the full Kennedy-O'Hagan framework, which also emulates
the simulator with a second Gaussian process.

**Identifiability.** ``theta`` and ``delta`` are confounded: a change of
``theta`` can be absorbed by ``delta``. Bias-corrected predictions inside the
data range stay accurate, but ``theta`` itself is learned only if the prior
on ``delta`` is realistic (Brynjarsdottir and O'Hagan 2014). Use
:meth:`DiscrepancyCalibration.identifiability` to see the posterior
correlation between ``theta`` and ``psi`` and how far the posterior of
``theta`` contracted from its prior.

Parameter vectors are ``(theta, log sigma_delta, log ell, a, b)`` where only
hyperparameters declared with a prior are free; see
:attr:`DiscrepancyCalibration.vector_names`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.inference import (
    HamiltonianResult,
    LaplaceApproximation,
    MCMCResult,
    ParameterSummary,
    adaptive_metropolis,
    hamiltonian_monte_carlo,
    laplace_approximation,
    no_u_turn_sampler,
)
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    cholesky_solve,
    inverse_spd,
    matvec,
    scale_matrix,
    solve_lower,
)
from probstat.modeling import (
    ContinuousMarginal,
    ProbabilisticModel,
    RandomInput,
    standard_normal_cdf,
)
from probstat.optimization import OptimizationResult, central_hessian, minimize_bfgs
from probstat.statistics import quantile

from .nonlinear import Experiment

DISCREPANCY_KERNELS = ("squared_exponential", "matern52")
_SQRT5 = math.sqrt(5.0)
_LOG_TWO_PI = math.log(2.0 * math.pi)
_EXP_LIMIT = 709.0


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


def _correlation(kernel: str, squared: float) -> float:
    """Return the unit-variance stationary correlation as a function of ``r^2``.

    Raises:
        ValueError: If the kernel name is unknown.

    Equation:
        [EQ-KOH-KERNEL-001].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

    """
    if kernel == "squared_exponential":
        return math.exp(-0.5 * squared)
    if kernel == "matern52":
        r = math.sqrt(squared)
        return (1.0 + _SQRT5 * r + 5.0 * squared / 3.0) * math.exp(-_SQRT5 * r)
    raise ValueError(f"kernel must be one of {DISCREPANCY_KERNELS}")


def discrepancy_kernel(
    kernel: str,
    left: Sequence[float],
    right: Sequence[float],
    length_scales: Sequence[float],
    variance: float,
) -> float:
    """Evaluate the discrepancy covariance ``k_psi(x, x')``.

    Squared exponential ``sigma_delta^2 exp(-r^2/2)`` or Matern 5/2
    ``sigma_delta^2 (1 + sqrt(5) r + 5 r^2/3) exp(-sqrt(5) r)`` with
    ``r^2 = sum_i ((x_i - x'_i) / ell_i)^2``.

    Args:
        kernel: ``"squared_exponential"`` or ``"matern52"``.
        left: First discrepancy input vector.
        right: Second discrepancy input vector.
        length_scales: Positive length scales, one per input.
        variance: Non-negative discrepancy variance ``sigma_delta^2``.

    Returns:
        Covariance value.

    Raises:
        ValueError: For an unknown kernel or mismatched lengths.

    Equation:
        [EQ-KOH-KERNEL-001].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2], [REF-KENNEDY-OHAGAN-2001].

    """
    if not (len(left) == len(right) == len(length_scales)):
        raise ValueError("inputs and length scales must have equal lengths")
    squared = math.fsum(
        ((a - b) / ell) ** 2
        for a, b, ell in zip(left, right, length_scales, strict=True)
    )
    return variance * _correlation(kernel, squared)


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


def _check_prior(name: str, law: ContinuousMarginal) -> RandomInput:
    """Wrap a hyperparameter prior law after a structural check.

    References:
        [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

    """
    return RandomInput(name, law)


@dataclass(frozen=True, slots=True)
class DiscrepancyProcess:
    """Gaussian-process prior of the additive model discrepancy ``delta(x)``.

    Each hyperparameter is either fixed or given a prior on ``(0, inf)``;
    free hyperparameters are estimated on the log scale. A fixed
    ``standard_deviation`` of zero switches the discrepancy off, which makes
    the problem identical to :class:`~probstat.inverse.CalibrationProblem`
    with known Gaussian noise.

    Args:
        inputs: Condition names the discrepancy depends on (``x``).
        kernel: ``"squared_exponential"`` (default) or ``"matern52"``.
        standard_deviation: Fixed ``sigma_delta >= 0``.
        standard_deviation_prior: Prior law of an unknown ``sigma_delta``.
        length_scales: Fixed positive length scales, one per input.
        length_scale_priors: Prior laws of unknown length scales.

    Raises:
        ValueError: For an empty or repeated input list, unknown kernel, or
            an invalid fixed/prior combination.

    References:
        [REF-KENNEDY-OHAGAN-2001], [REF-BRYNJARSDOTTIR-OHAGAN-2014],
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

    """

    inputs: tuple[str, ...]
    kernel: str = "squared_exponential"
    standard_deviation: float | None = None
    standard_deviation_prior: ContinuousMarginal | None = None
    length_scales: tuple[float, ...] | None = None
    length_scale_priors: tuple[ContinuousMarginal, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the declaration.

        References:
            [EQ-KOH-KERNEL-001]. No equation is evaluated.

        """
        names = tuple(self.inputs) if not isinstance(self.inputs, str) else ()
        if not names or any(
            not isinstance(name, str) or not name.strip() for name in names
        ):
            raise ValueError("inputs must be a non-empty sequence of names")
        if len(set(names)) != len(names):
            raise ValueError("inputs must be distinct")
        object.__setattr__(self, "inputs", names)
        if self.kernel not in DISCREPANCY_KERNELS:
            raise ValueError(f"kernel must be one of {DISCREPANCY_KERNELS}")
        if (self.standard_deviation is None) == (self.standard_deviation_prior is None):
            raise ValueError(
                "give exactly one of standard_deviation or standard_deviation_prior"
            )
        if self.standard_deviation is not None:
            value = require_finite_real(
                self.standard_deviation, name="standard_deviation"
            )
            if value < 0.0:
                raise ValueError("standard_deviation must be non-negative")
            object.__setattr__(self, "standard_deviation", value)
        else:
            assert self.standard_deviation_prior is not None
            _check_prior("sigma_delta", self.standard_deviation_prior)
        if (self.length_scales is None) == (self.length_scale_priors is None):
            raise ValueError("give exactly one of length_scales or length_scale_priors")
        if self.length_scales is not None:
            scales = as_vector(self.length_scales, name="length_scales")
            if len(scales) != len(names) or any(s <= 0.0 for s in scales):
                raise ValueError("length_scales must be positive, one per input")
            object.__setattr__(self, "length_scales", scales)
        else:
            assert self.length_scale_priors is not None
            priors = tuple(self.length_scale_priors)
            if len(priors) != len(names):
                raise ValueError("length_scale_priors must have one law per input")
            for name, law in zip(names, priors, strict=True):
                _check_prior(f"length[{name}]", law)
            object.__setattr__(self, "length_scale_priors", priors)

    @property
    def is_active(self) -> bool:
        """Return whether the discrepancy can be non-zero.

        References:
            [REF-KENNEDY-OHAGAN-2001]. No equation is evaluated.

        """
        return self.standard_deviation is None or self.standard_deviation > 0.0

    @property
    def free_names(self) -> tuple[str, ...]:
        """Return vector names of the free (log-scale) hyperparameters.

        References:
            Project API policy. No equation is evaluated.

        """
        names: tuple[str, ...] = ()
        if self.standard_deviation_prior is not None:
            names += ("log_sigma_delta",)
        if self.length_scale_priors is not None:
            names += tuple(f"log_length[{name}]" for name in self.inputs)
        return names


@dataclass(frozen=True, slots=True)
class KnownNoise:
    """Known Gaussian observation standard deviations.

    Args:
        standard_deviations: One positive value shared by all observations,
            or one value per experiment (heteroscedastic, known).

    Raises:
        ValueError: If a value is not finite and positive.

    References:
        [REF-GELMAN-ETAL-2013, Section 14.7].

    """

    standard_deviations: float | tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate and freeze the standard deviations.

        References:
            [EQ-KOH-MARG-001]. No equation is evaluated.

        """
        if isinstance(self.standard_deviations, int | float):
            value = require_finite_real(
                self.standard_deviations, name="standard_deviations"
            )
            if value <= 0.0:
                raise ValueError("standard_deviations must be positive")
            object.__setattr__(self, "standard_deviations", value)
            return
        values = as_vector(self.standard_deviations, name="standard_deviations")
        if not values or any(v <= 0.0 for v in values):
            raise ValueError("standard_deviations must be positive and non-empty")
        object.__setattr__(self, "standard_deviations", values)

    def expand(self, count: int) -> Vector:
        """Return one standard deviation per observation.

        Args:
            count: Number of observations.

        Returns:
            Length-``count`` vector.

        Raises:
            ValueError: If a per-observation tuple has the wrong length.

        References:
            [EQ-KOH-MARG-001]. No equation is evaluated.

        """
        if isinstance(self.standard_deviations, float):
            return tuple(self.standard_deviations for _ in range(count))
        if len(self.standard_deviations) != count:
            raise ValueError(
                f"KnownNoise needs {count} standard deviations, "
                f"got {len(self.standard_deviations)}"
            )
        return self.standard_deviations


@dataclass(frozen=True, slots=True)
class LogLinearNoise:
    """Parametric heteroscedastic noise ``sigma(c) = exp(a + b' g(c))``.

    ``g`` maps the merged conditions of an experiment (``model.fixed``
    overridden by ``Experiment.conditions``) to ``p`` covariates; ``a`` and
    ``b`` are unconstrained and estimated jointly with ``theta``. With no
    slopes the model is an unknown constant noise level ``exp(a)``.

    Args:
        covariates: Map from conditions to ``p`` finite covariates.
        intercept_prior: Prior law of ``a`` (log noise scale at ``g = 0``).
        slope_priors: Prior laws of ``b_1..b_p``.
        covariate_names: Optional names of the covariates.

    Raises:
        TypeError: If ``covariates`` is not callable.
        ValueError: If the names do not match the slope priors.

    References:
        [REF-HARVEY-1976], [REF-GELMAN-ETAL-2013, Section 14.7].

    """

    covariates: Callable[[Mapping[str, float]], Sequence[float]]
    intercept_prior: ContinuousMarginal
    slope_priors: tuple[ContinuousMarginal, ...] = ()
    covariate_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the declaration.

        References:
            [EQ-KOH-HET-001]. No equation is evaluated.

        """
        if not callable(self.covariates):
            raise TypeError("covariates must be callable")
        _check_prior("noise_a", self.intercept_prior)
        priors = tuple(self.slope_priors)
        for index, law in enumerate(priors):
            _check_prior(f"noise_b[{index}]", law)
        object.__setattr__(self, "slope_priors", priors)
        if self.covariate_names is None:
            names = tuple(f"g{index + 1}" for index in range(len(priors)))
        else:
            names = tuple(str(name) for name in self.covariate_names)
        if len(names) != len(priors) or len(set(names)) != len(names):
            raise ValueError("covariate_names must be distinct, one per slope prior")
        object.__setattr__(self, "covariate_names", names)

    @property
    def names(self) -> tuple[str, ...]:
        """Return vector names ``noise_a`` and ``noise_b[name]``.

        References:
            Project API policy. No equation is evaluated.

        """
        assert self.covariate_names is not None
        return ("noise_a",) + tuple(f"noise_b[{n}]" for n in self.covariate_names)

    def evaluate(self, conditions: Mapping[str, float]) -> Vector:
        """Return validated covariates ``g(c)``.

        Raises:
            ValueError: If the covariate count is wrong or a value is not
                finite.

        References:
            [EQ-KOH-HET-001]. No equation is evaluated.

        """
        values = as_vector(self.covariates(MappingProxyType(dict(conditions))))
        if len(values) != len(self.slope_priors):
            raise ValueError(
                f"covariates must return {len(self.slope_priors)} values, "
                f"got {len(values)}"
            )
        return values

    def standard_deviation(
        self, coefficients: Sequence[float], covariates: Sequence[float]
    ) -> float:
        """Return ``exp(a + b' g)`` (``inf`` on overflow).

        Equation:
            [EQ-KOH-HET-001].

        References:
            [REF-HARVEY-1976].

        """
        exponent = coefficients[0] + math.fsum(
            b * g for b, g in zip(coefficients[1:], covariates, strict=True)
        )
        return math.exp(exponent) if exponent < _EXP_LIMIT else math.inf


NoiseVarianceModel = KnownNoise | LogLinearNoise


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def _correlation_matrix(covariance: Matrix) -> Matrix:
    """Return the correlation matrix of a covariance (``nan`` for zero variance).

    Equation:
        [EQ-KOH-IDENT-001].

    References:
        [REF-CASELLA-BERGER-2002, Section 4.5].

    """
    size = len(covariance)
    sd = tuple(math.sqrt(max(covariance[i][i], 0.0)) for i in range(size))
    return tuple(
        tuple(
            covariance[i][j] / (sd[i] * sd[j])
            if sd[i] > 0.0 and sd[j] > 0.0
            else math.nan
            for j in range(size)
        )
        for i in range(size)
    )


@dataclass(frozen=True, slots=True)
class DiscrepancyFit:
    """Point estimate of ``(theta, log psi, a, b)`` with a Hessian covariance.

    Args:
        names: Vector names (:attr:`DiscrepancyCalibration.vector_names`).
        estimate: Maximizer of the log posterior or log likelihood.
        covariance: Inverse finite-difference Hessian of the negative
            objective at the estimate, or ``None`` if it was not positive
            definite (for example a variance drifting to zero).
        log_value: Objective value (log posterior or log likelihood).
        objective: ``"posterior"`` (MAP) or ``"likelihood"`` (MLE).
        converged: Whether BFGS met its tolerance.
        evaluations: Objective evaluations including restarts and Hessian.
        optimization: Raw BFGS result of the best start.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 6], [REF-TIERNEY-KADANE-1986].

    """

    names: tuple[str, ...]
    estimate: Vector
    covariance: Matrix | None
    log_value: float
    objective: str
    converged: bool
    evaluations: int
    optimization: OptimizationResult

    def as_mapping(self) -> Mapping[str, float]:
        """Return the estimate keyed by vector name.

        References:
            Project API policy. No equation is evaluated.

        """
        return MappingProxyType(dict(zip(self.names, self.estimate, strict=True)))

    @property
    def standard_errors(self) -> Vector:
        """Return ``sqrt(diag(covariance))``.

        Raises:
            ValueError: If no covariance is available.

        Equation:
            [EQ-LAPLACE-001].

        References:
            [REF-TIERNEY-KADANE-1986].

        """
        if self.covariance is None:
            raise ValueError("the Hessian was not positive definite")
        return tuple(math.sqrt(self.covariance[i][i]) for i in range(len(self.names)))

    def correlation(self) -> Matrix:
        """Return the approximate posterior (or sampling) correlation matrix.

        Raises:
            ValueError: If no covariance is available.

        Equation:
            [EQ-KOH-IDENT-001].

        References:
            [REF-CASELLA-BERGER-2002, Section 4.5].

        """
        if self.covariance is None:
            raise ValueError("the Hessian was not positive definite")
        return _correlation_matrix(self.covariance)


@dataclass(frozen=True, slots=True)
class DiscrepancyPosterior:
    """MCMC posterior of a Kennedy-O'Hagan calibration.

    Args:
        vector: Chains in vector coordinates ``(theta, log psi, a, b)``.
        mcmc: Chains on the physical scale ``(theta, sigma_delta, ell, a, b)``.
        standard_space: Raw chains in sampling coordinates
            ``(u, log psi, a, b)`` with ``theta = T^-1(u)``.
        laplace: Laplace approximation used to start the chains, or ``None``.
        sampler: HMC/NUTS diagnostics, ``None`` for adaptive Metropolis.

    References:
        [REF-KENNEDY-OHAGAN-2001], [REF-HIGDON-ETAL-2004],
        [REF-GELMAN-ETAL-2013, Chapter 11].

    """

    vector: MCMCResult
    mcmc: MCMCResult
    standard_space: MCMCResult
    laplace: LaplaceApproximation | None
    sampler: HamiltonianResult | None = None

    def mean(self) -> Mapping[str, float]:
        """Return pooled physical posterior means keyed by name.

        Equation:
            [EQ-MC-MEAN-001].

        References:
            [REF-ROBERT-CASELLA-2004, Chapter 7].

        """
        return MappingProxyType(
            dict(zip(self.mcmc.names, self.mcmc.mean(), strict=True))
        )

    def summary(
        self, probabilities: Sequence[float] = (0.05, 0.5, 0.95)
    ) -> tuple[ParameterSummary, ...]:
        """Summarize every physical coordinate with convergence diagnostics.

        Equation:
            [EQ-MCMC-RHAT-002], [EQ-MCMC-ESS-001], [EQ-MCMC-MCSE-001].

        References:
            [REF-VEHTARI-ETAL-2021].

        """
        return self.mcmc.summary(probabilities)


def _mixture_quantile(
    means: Sequence[float], variances: Sequence[float], probability: float
) -> float:
    """Return a quantile of an equally weighted Gaussian mixture.

    Components with zero variance are point masses. When every component is
    a point mass the type-7 sample quantile is returned; otherwise the
    mixture CDF ``F(q) = S^-1 sum_s Phi((q - m_s)/s_s)`` is inverted by
    bisection to ``1e-12`` relative width.

    Equation:
        [EQ-KOH-PRED-001], [EQ-STAT-QUANT-001].

    References:
        [REF-GELMAN-ETAL-2013, Section 6.3], [REF-HYNDMAN-FAN-1996].

    """
    if all(v <= 0.0 for v in variances):
        return quantile(sorted(means), probability)
    sds = tuple(math.sqrt(max(v, 0.0)) for v in variances)
    count = len(means)

    def cdf(point: float) -> float:
        """Return the mixture CDF at one point.

        References:
            [EQ-KOH-PRED-001]. No new equation is introduced.

        """
        total = 0.0
        for m, s in zip(means, sds, strict=True):
            if s > 0.0:
                total += standard_normal_cdf((point - m) / s)
            elif point >= m:
                total += 1.0
        return total / count

    low = min(m - 12.0 * s for m, s in zip(means, sds, strict=True))
    high = max(m + 12.0 * s for m, s in zip(means, sds, strict=True))
    scale = max(1.0, abs(low), abs(high))
    for _ in range(200):
        if high - low <= 1e-12 * scale:
            break
        middle = 0.5 * (low + high)
        if cdf(middle) < probability:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _sample_variance(values: Sequence[float]) -> float:
    """Return the unbiased sample variance (zero for one value).

    Equation:
        [EQ-MC-VAR-001].

    References:
        [REF-ROBERT-CASELLA-2004, Chapter 3].

    """
    if len(values) < 2:
        return 0.0
    center = math.fsum(values) / len(values)
    return math.fsum((v - center) ** 2 for v in values) / (len(values) - 1)


@dataclass(frozen=True, slots=True)
class DiscrepancyPrediction:
    """Posterior prediction at one new condition, simulator versus corrected.

    For each parameter draw ``s``: the simulator value ``eta_s``, the
    conditional discrepancy mean ``m_s`` and variance ``v_s`` given the data,
    and (when available) the observation-noise variance ``n_s``. The
    bias-corrected latent process ``eta + delta`` is the Gaussian mixture
    ``S^-1 sum_s N(eta_s + m_s, v_s)``; a new observation adds ``n_s``.

    Args:
        condition: Merged prediction conditions.
        simulator: ``eta(c*, theta_s)`` per draw.
        discrepancy_means: ``m_s`` per draw.
        discrepancy_variances: ``v_s`` per draw.
        noise_variances: ``n_s`` per draw, or ``None`` when unknown.

    References:
        [REF-KENNEDY-OHAGAN-2001], [REF-RASMUSSEN-WILLIAMS-2006, Section 2.2].

    """

    condition: Mapping[str, float]
    simulator: Vector
    discrepancy_means: Vector
    discrepancy_variances: Vector
    noise_variances: Vector | None

    @property
    def simulator_mean(self) -> float:
        """Return the posterior mean of the simulator alone.

        Equation:
            [EQ-KOH-PRED-001].

        References:
            [REF-GELMAN-ETAL-2013, Section 6.3].

        """
        return math.fsum(self.simulator) / len(self.simulator)

    @property
    def simulator_variance(self) -> float:
        """Return the parameter-uncertainty variance of the simulator.

        Equation:
            [EQ-KOH-PRED-001], [EQ-MC-VAR-001].

        References:
            [REF-GELMAN-ETAL-2013, Section 6.3].

        """
        return _sample_variance(self.simulator)

    @property
    def discrepancy_mean(self) -> float:
        """Return the posterior mean of ``delta(x*)``.

        Equation:
            [EQ-KOH-PRED-001].

        References:
            [REF-KENNEDY-OHAGAN-2001].

        """
        return math.fsum(self.discrepancy_means) / len(self.discrepancy_means)

    def _corrected_centers(self) -> Vector:
        """Return ``eta_s + m_s``.

        References:
            [EQ-KOH-PRED-001]. No new equation is introduced.

        """
        return tuple(
            e + m for e, m in zip(self.simulator, self.discrepancy_means, strict=True)
        )

    @property
    def corrected_mean(self) -> float:
        """Return the posterior mean of ``eta(x*, theta) + delta(x*)``.

        Equation:
            [EQ-KOH-PRED-001].

        References:
            [REF-KENNEDY-OHAGAN-2001].

        """
        centers = self._corrected_centers()
        return math.fsum(centers) / len(centers)

    @property
    def corrected_variance(self) -> float:
        """Return ``Var_s(eta_s + m_s) + mean_s(v_s)`` (law of total variance).

        Equation:
            [EQ-KOH-PRED-001].

        References:
            [REF-CASELLA-BERGER-2002, Section 4.4].

        """
        values = self.discrepancy_variances
        return _sample_variance(self._corrected_centers()) + math.fsum(values) / len(
            values
        )

    def interval(
        self, probability: float = 0.95, *, kind: str = "corrected"
    ) -> tuple[float, float]:
        """Return a central predictive interval.

        Args:
            probability: Central probability in ``(0, 1)``.
            kind: ``"corrected"`` (``eta + delta``), ``"simulator"``
                (``eta`` alone, parameter uncertainty only; type-7 sample
                quantiles), or ``"observed"`` (``eta + delta + eps``).

        Returns:
            Lower and upper limits.

        Raises:
            ValueError: For an invalid level or kind, or ``"observed"``
                without noise variances.

        Equation:
            [EQ-KOH-PRED-001], [EQ-STAT-QUANT-001].

        References:
            [REF-GELMAN-ETAL-2013, Section 6.3], [REF-HYNDMAN-FAN-1996].

        """
        level = require_finite_real(probability, name="probability")
        if not 0.0 < level < 1.0:
            raise ValueError("probability must lie strictly between zero and one")
        tails = (0.5 * (1.0 - level), 0.5 * (1.0 + level))
        if kind == "simulator":
            ordered = sorted(self.simulator)
            return quantile(ordered, tails[0]), quantile(ordered, tails[1])
        if kind == "corrected":
            variances = self.discrepancy_variances
        elif kind == "observed":
            if self.noise_variances is None:
                raise ValueError("observation noise variances are not available")
            variances = tuple(
                v + n
                for v, n in zip(
                    self.discrepancy_variances, self.noise_variances, strict=True
                )
            )
        else:
            raise ValueError("kind must be 'corrected', 'simulator', or 'observed'")
        centers = self._corrected_centers()
        return (
            _mixture_quantile(centers, variances, tails[0]),
            _mixture_quantile(centers, variances, tails[1]),
        )


@dataclass(frozen=True, slots=True)
class DiscrepancyIdentifiability:
    """Theta-discrepancy confounding report.

    Args:
        names: Vector names analysed.
        correlation: Posterior correlation matrix in vector coordinates.
        cross_correlations: ``theta name -> hyperparameter name -> rho``.
        max_abs_cross_correlation: Largest ``|rho|`` (``0`` if none).
        prior_standard_deviations: Prior standard deviation of each ``theta``.
        posterior_standard_deviations: Posterior standard deviation of each
            ``theta``.
        standard_deviation_ratios: Posterior over prior standard deviation;
            values near one mean the data barely inform ``theta``.
        source: ``"mcmc"`` or ``"laplace"``.
        warning: Human-readable warning or ``None``.

    References:
        [REF-BRYNJARSDOTTIR-OHAGAN-2014], [REF-KENNEDY-OHAGAN-2001].

    """

    names: tuple[str, ...]
    correlation: Matrix
    cross_correlations: Mapping[str, Mapping[str, float]]
    max_abs_cross_correlation: float
    prior_standard_deviations: Mapping[str, float]
    posterior_standard_deviations: Mapping[str, float]
    standard_deviation_ratios: Mapping[str, float]
    source: str
    warning: str | None


# ---------------------------------------------------------------------------
# The calibration problem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscrepancyCalibration:
    """Kennedy-O'Hagan calibration of one observed quantity.

    Args:
        model: Simulator with the prior of ``theta`` as its input law.
        experiments: Experiments; each must observe ``quantity`` (other
            observed quantities are ignored).
        quantity: Observed quantity of interest.
        noise: :class:`KnownNoise` or :class:`LogLinearNoise`.
        discrepancy: Gaussian-process discrepancy, or ``None`` for none.

    Raises:
        TypeError: If components have the wrong types.
        ValueError: If the quantity is undeclared or unobserved, a condition
            shadows a parameter, or a discrepancy input is missing.

    References:
        [REF-KENNEDY-OHAGAN-2001], [REF-HIGDON-ETAL-2004],
        [REF-BRYNJARSDOTTIR-OHAGAN-2014].

    """

    model: ProbabilisticModel
    experiments: tuple[Experiment, ...]
    quantity: str
    noise: NoiseVarianceModel
    discrepancy: DiscrepancyProcess | None = None
    _bases: tuple[Mapping[str, float], ...] = field(
        init=False, repr=False, compare=False
    )
    _observed: Vector = field(init=False, repr=False, compare=False)
    _points: tuple[Vector, ...] = field(init=False, repr=False, compare=False)
    _known: Vector | None = field(init=False, repr=False, compare=False)
    _covariates: tuple[Vector, ...] = field(init=False, repr=False, compare=False)
    _hyper_priors: tuple[RandomInput, ...] = field(
        init=False, repr=False, compare=False
    )
    _noise_priors: tuple[RandomInput, ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate the declarations and cache the data layout.

        References:
            [EQ-KOH-MODEL-001]. No equation is evaluated.

        """
        if not isinstance(self.model, ProbabilisticModel):
            raise TypeError("model must be a ProbabilisticModel")
        if not isinstance(self.noise, KnownNoise | LogLinearNoise):
            raise TypeError("noise must be KnownNoise or LogLinearNoise")
        if self.discrepancy is not None and not isinstance(
            self.discrepancy, DiscrepancyProcess
        ):
            raise TypeError("discrepancy must be a DiscrepancyProcess or None")
        if self.quantity not in self.model.quantity_names:
            raise ValueError(f"{self.quantity!r} is not a model quantity")
        experiments = tuple(self.experiments)
        if not experiments:
            raise ValueError("at least one experiment is required")
        parameters = set(self.model.inputs.names)
        bases: list[Mapping[str, float]] = []
        observed: list[float] = []
        for index, experiment in enumerate(experiments):
            if not isinstance(experiment, Experiment):
                raise TypeError(f"experiments[{index}] must be an Experiment")
            if self.quantity not in experiment.observations:
                raise ValueError(f"experiment {index} does not observe the quantity")
            shadow = parameters & set(experiment.conditions)
            if shadow:
                raise ValueError(
                    f"experiment {index} conditions shadow parameters {sorted(shadow)}"
                )
            base = dict(self.model.fixed)
            base.update(experiment.conditions)
            bases.append(MappingProxyType(base))
            observed.append(experiment.observations[self.quantity])
        points: tuple[Vector, ...] = ()
        hyper: list[RandomInput] = []
        if self.discrepancy is not None:
            names = self.discrepancy.inputs
            clash = parameters & set(names)
            if clash:
                raise ValueError(f"discrepancy inputs name parameters {sorted(clash)}")
            for index, merged in enumerate(bases):
                missing = [name for name in names if name not in merged]
                if missing:
                    raise ValueError(
                        f"experiment {index} lacks discrepancy inputs {missing}"
                    )
            points = tuple(tuple(base[name] for name in names) for base in bases)
            if self.discrepancy.standard_deviation_prior is not None:
                hyper.append(
                    RandomInput(
                        "sigma_delta", self.discrepancy.standard_deviation_prior
                    )
                )
            if self.discrepancy.length_scale_priors is not None:
                hyper.extend(
                    RandomInput(f"length[{name}]", law)
                    for name, law in zip(
                        names, self.discrepancy.length_scale_priors, strict=True
                    )
                )
        known: Vector | None = None
        covariates: tuple[Vector, ...] = ()
        noise_priors: tuple[RandomInput, ...] = ()
        if isinstance(self.noise, KnownNoise):
            known = self.noise.expand(len(experiments))
        else:
            covariates = tuple(self.noise.evaluate(base) for base in bases)
            noise_priors = (
                RandomInput("noise_a", self.noise.intercept_prior),
            ) + tuple(
                RandomInput(name, law)
                for name, law in zip(
                    self.noise.names[1:], self.noise.slope_priors, strict=True
                )
            )
        object.__setattr__(self, "experiments", experiments)
        object.__setattr__(self, "_bases", tuple(bases))
        object.__setattr__(self, "_observed", tuple(observed))
        object.__setattr__(self, "_points", points)
        object.__setattr__(self, "_known", known)
        object.__setattr__(self, "_covariates", covariates)
        object.__setattr__(self, "_hyper_priors", tuple(hyper))
        object.__setattr__(self, "_noise_priors", noise_priors)

    # -- layout -------------------------------------------------------------

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return calibration-parameter names.

        References:
            [REF-KENNEDY-OHAGAN-2001]. No equation is evaluated.

        """
        return self.model.inputs.names

    @property
    def hyperparameter_names(self) -> tuple[str, ...]:
        """Return vector names of free discrepancy hyperparameters.

        References:
            [REF-KENNEDY-OHAGAN-2001]. No equation is evaluated.

        """
        return () if self.discrepancy is None else self.discrepancy.free_names

    @property
    def noise_names(self) -> tuple[str, ...]:
        """Return vector names of free noise coefficients.

        References:
            [REF-HARVEY-1976]. No equation is evaluated.

        """
        return self.noise.names if isinstance(self.noise, LogLinearNoise) else ()

    @property
    def vector_names(self) -> tuple[str, ...]:
        """Return names of ``(theta, log psi, a, b)``.

        References:
            Project API policy. No equation is evaluated.

        """
        return self.parameter_names + self.hyperparameter_names + self.noise_names

    @property
    def physical_names(self) -> tuple[str, ...]:
        """Return names of ``(theta, sigma_delta, ell, a, b)``.

        References:
            Project API policy. No equation is evaluated.

        """
        hyper = tuple(
            name[4:] if name.startswith("log_") else name
            for name in self.hyperparameter_names
        )
        return self.parameter_names + hyper + self.noise_names

    @property
    def dimension(self) -> int:
        """Return the parameter-vector length.

        References:
            Project API policy. No equation is evaluated.

        """
        return len(self.vector_names)

    @property
    def observation_count(self) -> int:
        """Return the number of observations.

        References:
            [EQ-KOH-MODEL-001]. No equation is evaluated.

        """
        return len(self._observed)

    @property
    def observed(self) -> Vector:
        """Return the observations in experiment order.

        References:
            [EQ-KOH-MODEL-001]. No equation is evaluated.

        """
        return self._observed

    @property
    def is_dense(self) -> bool:
        """Return whether the likelihood needs the dense Cholesky path.

        Without an active discrepancy the covariance is diagonal and the
        likelihood is a sum of independent Gaussian terms (``O(n)``).

        References:
            [EQ-KOH-MARG-001]. No equation is evaluated.

        """
        return self.discrepancy is not None and self.discrepancy.is_active

    def _unpack(
        self, parameters: Sequence[float]
    ) -> tuple[Vector, float, Vector, Vector]:
        """Split a vector into ``theta``, ``sigma_delta``, ``ell``, and ``(a, b)``.

        Raises:
            ValueError: If the length is wrong.

        References:
            [EQ-KOH-POST-001]. No new equation is introduced.

        """
        vector = as_vector(parameters, name="parameters")
        if len(vector) != self.dimension:
            raise ValueError(f"parameters must have length {self.dimension}")
        d = self.model.dimension
        theta = vector[:d]
        offset = d
        sigma_delta = 0.0
        lengths: Vector = ()
        if self.discrepancy is not None:
            if self.discrepancy.standard_deviation is not None:
                sigma_delta = self.discrepancy.standard_deviation
            else:
                sigma_delta = _safe_exp(vector[offset])
                offset += 1
            if self.discrepancy.length_scales is not None:
                lengths = self.discrepancy.length_scales
            else:
                count = len(self.discrepancy.inputs)
                lengths = tuple(_safe_exp(v) for v in vector[offset : offset + count])
                offset += count
        return theta, sigma_delta, lengths, vector[offset:]

    def noise_standard_deviations(self, parameters: Sequence[float]) -> Vector:
        """Return the observation-noise standard deviation of every experiment.

        Equation:
            [EQ-KOH-HET-001].

        References:
            [REF-HARVEY-1976].

        """
        _, _, _, coefficients = self._unpack(parameters)
        return self._noise_sds(coefficients)

    def _noise_sds(self, coefficients: Vector) -> Vector:
        """Return noise standard deviations for split coefficients.

        References:
            [EQ-KOH-HET-001]. No new equation is introduced.

        """
        if self._known is not None:
            return self._known
        assert isinstance(self.noise, LogLinearNoise)
        noise = self.noise
        return tuple(
            noise.standard_deviation(coefficients, g) for g in self._covariates
        )

    # -- forward evaluation -------------------------------------------------

    def _simulate(self, base: Mapping[str, float], theta: Vector) -> float:
        """Evaluate the simulator quantity under merged conditions.

        References:
            [EQ-UQ-001], [EQ-KOH-MODEL-001].

        """
        values = dict(base)
        values.update(zip(self.model.inputs.names, theta, strict=True))
        self.model.counter.add()
        outputs = self.model.model(MappingProxyType(values))
        if not isinstance(outputs, Mapping):
            raise TypeError("model output must be a mapping")
        if self.quantity not in outputs:
            raise ValueError(f"model output is missing QoI {self.quantity!r}")
        return require_finite_real(outputs[self.quantity], name="model output")

    def predictions(self, theta: Sequence[float]) -> Vector:
        """Return simulator predictions ``eta(c_i, theta)`` in experiment order.

        Raises:
            ValueError: If ``theta`` has the wrong length.

        Equation:
            [EQ-KOH-MODEL-001].

        References:
            [REF-KENNEDY-OHAGAN-2001].

        """
        vector = as_vector(theta, name="theta")
        if len(vector) != self.model.dimension:
            raise ValueError("theta length must equal the number of parameters")
        return tuple(self._simulate(base, vector) for base in self._bases)

    def discrepancy_covariance(
        self, sigma_delta: float, length_scales: Sequence[float]
    ) -> Matrix:
        """Return the discrepancy kernel matrix ``K_psi`` at the data inputs.

        Raises:
            ValueError: If the problem has no discrepancy.

        Equation:
            [EQ-KOH-KERNEL-001].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        if self.discrepancy is None:
            raise ValueError("the problem has no discrepancy process")
        kernel = self.discrepancy.kernel
        variance = sigma_delta * sigma_delta
        count = len(self._points)
        rows = [[0.0] * count for _ in range(count)]
        for i in range(count):
            rows[i][i] = variance
            for j in range(i):
                value = discrepancy_kernel(
                    kernel, self._points[i], self._points[j], length_scales, variance
                )
                rows[i][j] = value
                rows[j][i] = value
        return tuple(tuple(row) for row in rows)

    def marginal_covariance(self, parameters: Sequence[float]) -> Matrix:
        """Return ``K_psi + diag(sigma_i^2)`` for a full parameter vector.

        Equation:
            [EQ-KOH-MARG-001].

        References:
            [REF-KENNEDY-OHAGAN-2001], [REF-RASMUSSEN-WILLIAMS-2006, Sec. 2.2].

        """
        _, sigma_delta, lengths, coefficients = self._unpack(parameters)
        return self._covariance(sigma_delta, lengths, self._noise_sds(coefficients))

    def _covariance(
        self, sigma_delta: float, lengths: Vector, noise_sds: Vector
    ) -> Matrix:
        """Assemble the marginal covariance from split parameters.

        References:
            [EQ-KOH-MARG-001]. No new equation is introduced.

        """
        count = self.observation_count
        if self.is_dense:
            base = self.discrepancy_covariance(sigma_delta, lengths)
        else:
            base = tuple(tuple(0.0 for _ in range(count)) for _ in range(count))
        return tuple(
            tuple(
                base[i][j] + (noise_sds[i] ** 2 if i == j else 0.0)
                for j in range(count)
            )
            for i in range(count)
        )

    # -- densities ------------------------------------------------------------

    def log_likelihood(self, parameters: Sequence[float]) -> float:
        """Return the marginal log-likelihood ``log N(y; eta(C, theta), C_psi)``.

        ``C_psi = K_psi + diag(sigma_i^2) = L L'`` and
        ``log p = -1/2 |L^-1 r|^2 - sum_i log L_ii - (n/2) log 2 pi`` with
        ``r = y - eta(C, theta)``. Without an active discrepancy the
        covariance is diagonal and the sum of independent Gaussian
        log-densities is returned. A covariance that is not numerically
        positive definite gives ``-inf``.

        Args:
            parameters: Vector ``(theta, log psi, a, b)``.

        Returns:
            Log marginal likelihood.

        Equation:
            [EQ-KOH-MARG-001].

        References:
            [REF-KENNEDY-OHAGAN-2001], [REF-RASMUSSEN-WILLIAMS-2006, Sec. 2.2].

        """
        theta, sigma_delta, lengths, coefficients = self._unpack(parameters)
        return self._log_likelihood(theta, sigma_delta, lengths, coefficients)

    def _log_likelihood(
        self,
        theta: Vector,
        sigma_delta: float,
        lengths: Vector,
        coefficients: Vector,
    ) -> float:
        """Evaluate the marginal likelihood for split parameters.

        References:
            [EQ-KOH-MARG-001], [REF-RASMUSSEN-WILLIAMS-2006, Section 2.2].

        """
        noise_sds = self._noise_sds(coefficients)
        if any(not math.isfinite(s) or s <= 0.0 for s in noise_sds):
            return -math.inf
        if not math.isfinite(sigma_delta) or any(
            not math.isfinite(ell) or ell <= 0.0 for ell in lengths
        ):
            return -math.inf
        residuals = tuple(
            y - p for y, p in zip(self._observed, self.predictions(theta), strict=True)
        )
        count = len(residuals)
        if not self.is_dense:
            return math.fsum(
                -0.5 * (_LOG_TWO_PI + 2.0 * math.log(s) + (r / s) ** 2)
                for r, s in zip(residuals, noise_sds, strict=True)
            )
        try:
            lower = cholesky(self._covariance(sigma_delta, lengths, noise_sds))
        except ValueError:
            return -math.inf
        whitened = solve_lower(lower, residuals)
        return (
            -0.5 * math.fsum(w * w for w in whitened)
            - math.fsum(math.log(lower[i][i]) for i in range(count))
            - 0.5 * count * _LOG_TWO_PI
        )

    def _log_extra_prior(self, values: Sequence[float]) -> float:
        """Return priors of log hyperparameters (with Jacobian) and noise terms.

        References:
            [EQ-KOH-POST-001], [REF-CASELLA-BERGER-2002, Section 4.3].

        """
        hyper = len(self._hyper_priors)
        total = 0.0
        for prior, eta in zip(self._hyper_priors, values[:hyper], strict=True):
            if eta >= _EXP_LIMIT:
                return -math.inf
            value = math.exp(eta)
            if value <= 0.0:
                return -math.inf
            density = prior.log_pdf(value)
            if not math.isfinite(density):
                return -math.inf
            total += density + eta
        for prior, coefficient in zip(self._noise_priors, values[hyper:], strict=True):
            density = prior.log_pdf(coefficient)
            if not math.isfinite(density):
                return -math.inf
            total += density
        return total

    def log_prior(self, parameters: Sequence[float]) -> float:
        """Return the prior log-density of ``(theta, log psi, a, b)``.

        Equation:
            [EQ-KOH-POST-001].

        References:
            [REF-CASELLA-BERGER-2002, Section 4.3], [REF-KENNEDY-OHAGAN-2001].

        """
        vector = as_vector(parameters, name="parameters")
        if len(vector) != self.dimension:
            raise ValueError(f"parameters must have length {self.dimension}")
        d = self.model.dimension
        prior = self.model.inputs.log_pdf(vector[:d])
        if not math.isfinite(prior):
            return -math.inf
        return prior + self._log_extra_prior(vector[d:])

    def log_posterior(self, parameters: Sequence[float]) -> float:
        """Return the unnormalized log posterior of ``(theta, log psi, a, b)``.

        Equation:
            [EQ-KOH-POST-001], [EQ-BAYES-RULE-001].

        References:
            [REF-KENNEDY-OHAGAN-2001], [REF-GELMAN-ETAL-2013, Section 1.3].

        """
        prior = self.log_prior(parameters)
        if prior == -math.inf:
            return -math.inf
        return prior + self.log_likelihood(parameters)

    def standard_log_posterior(self, coordinates: Sequence[float]) -> float:
        """Return the log posterior in sampling coordinates ``(u, log psi, a, b)``.

        ``theta = T^-1(u)`` (Nataf map of the prior), so the prior of
        ``theta`` becomes ``-|u|^2/2`` and needs no Jacobian.

        Equation:
            [EQ-KOH-POST-001], [EQ-CAL-STD-001], [EQ-MOD-NATAF-001].

        References:
            [REF-LIU-DERKIUREGHIAN-1986], [REF-KENNEDY-OHAGAN-2001].

        """
        vector = as_vector(coordinates, name="coordinates")
        if len(vector) != self.dimension:
            raise ValueError(f"coordinates must have length {self.dimension}")
        d = self.model.dimension
        u = vector[:d]
        extra = self._log_extra_prior(vector[d:])
        if extra == -math.inf:
            return -math.inf
        theta = self.model.inputs.from_standard_normal(u)
        _, sigma_delta, lengths, coefficients = self._unpack(theta + vector[d:])
        return (
            self._log_likelihood(theta, sigma_delta, lengths, coefficients)
            - 0.5 * math.fsum(value * value for value in u)
            + extra
        )

    def to_vector(self, coordinates: Sequence[float]) -> Vector:
        """Map sampling coordinates ``(u, ...)`` to vector coordinates ``(theta, ...)``.

        Equation:
            [EQ-MOD-NATAF-001].

        References:
            [REF-LIU-DERKIUREGHIAN-1986].

        """
        vector = as_vector(coordinates, name="coordinates")
        d = self.model.dimension
        return self.model.inputs.from_standard_normal(vector[:d]) + vector[d:]

    def to_physical(self, parameters: Sequence[float]) -> Vector:
        """Map vector coordinates to ``(theta, sigma_delta, ell, a, b)``.

        References:
            [EQ-KOH-POST-001]. No new equation is introduced.

        """
        vector = as_vector(parameters, name="parameters")
        d = self.model.dimension
        hyper = len(self._hyper_priors)
        return (
            vector[:d]
            + tuple(_safe_exp(v) for v in vector[d : d + hyper])
            + vector[d + hyper :]
        )

    def _default_extra(self) -> Vector:
        """Return log prior medians of hyperparameters and noise prior medians.

        References:
            [REF-CASELLA-BERGER-2002, Section 4.3]. No equation is evaluated.

        """
        return tuple(
            math.log(float(prior.distribution.quantile(0.5)))
            for prior in self._hyper_priors
        ) + tuple(
            float(prior.distribution.quantile(0.5)) for prior in self._noise_priors
        )

    def default_start(self) -> Vector:
        """Return prior means of ``theta`` with prior medians of the rest.

        References:
            [REF-CASELLA-BERGER-2002, Section 4.3]. No equation is evaluated.

        """
        return self.model.nominal() + self._default_extra()

    # -- point estimation -----------------------------------------------------

    def fit(
        self,
        *,
        objective: str = "posterior",
        initial: Sequence[float] | None = None,
        restarts: int = 0,
        seed: int | RandomStream | None = None,
        max_iterations: int = 500,
    ) -> DiscrepancyFit:
        """Maximize the log posterior (MAP) or log likelihood (MLE) by BFGS.

        Coordinates are the vector coordinates ``(theta, log psi, a, b)``.
        ``restarts`` extra starts are drawn from the prior (``theta`` via the
        Nataf map, hyperparameters and noise coefficients by inverse CDF) and
        the best optimum is kept. The covariance is the inverse central
        finite-difference Hessian of the negative objective; for the MLE it
        is the observed-information covariance. For MAP the hyperparameter
        mode includes the log-scale Jacobian, as in
        :meth:`CalibrationProblem.laplace_posterior`.

        Args:
            objective: ``"posterior"`` or ``"likelihood"``.
            initial: Starting vector; defaults to :meth:`default_start`.
            restarts: Number of additional prior-drawn starts.
            seed: Seed or stream for the restarts.
            max_iterations: BFGS iteration limit per start.

        Returns:
            Point estimate, covariance (or ``None``), and diagnostics.

        Raises:
            ValueError: For an unknown objective, a negative restart count,
                or a starting point with non-finite objective.

        Equation:
            [EQ-KOH-POST-001], [EQ-KOH-MARG-001], [EQ-OPT-BFGS-001],
            [EQ-LAPLACE-001].

        References:
            [REF-NOCEDAL-WRIGHT-2006, Chapter 6], [REF-HARVEY-1976],
            [REF-KENNEDY-OHAGAN-2001].

        """
        if objective not in ("posterior", "likelihood"):
            raise ValueError("objective must be 'posterior' or 'likelihood'")
        extra_starts = require_non_negative_integer(restarts, name="restarts")
        target = self.log_posterior if objective == "posterior" else self.log_likelihood

        def negative(point: Vector) -> float:
            """Return the negative objective (``+inf`` raises inside BFGS).

            References:
                [EQ-KOH-POST-001]. No new equation is introduced.

            """
            return -target(point)

        start = self.default_start() if initial is None else as_vector(initial)
        if len(start) != self.dimension:
            raise ValueError(f"initial must have length {self.dimension}")
        if not math.isfinite(negative(start)):
            raise ValueError("initial point has a non-finite objective")
        starts = [start]
        if extra_starts:
            generator = resolve_stream(seed).child("koh-restarts").generator()
            for _ in range(extra_starts):
                u = tuple(
                    generator.gauss(0.0, 1.0) for _ in range(self.model.dimension)
                )
                candidate = self.model.inputs.from_standard_normal(u)
                candidate += tuple(
                    math.log(float(prior.distribution.quantile(generator.random())))
                    for prior in self._hyper_priors
                )
                candidate += tuple(
                    float(prior.distribution.quantile(generator.random()))
                    for prior in self._noise_priors
                )
                if math.isfinite(negative(candidate)):
                    starts.append(candidate)
        best: OptimizationResult | None = None
        evaluations = 0
        for point in starts:
            result = minimize_bfgs(negative, point, max_iterations=max_iterations)
            evaluations += result.evaluations
            if best is None or result.value < best.value:
                best = result
        assert best is not None
        covariance: Matrix | None
        try:
            hessian, used = central_hessian(negative, best.x)
            evaluations += used
            covariance = inverse_spd(hessian)
            covariance = tuple(
                tuple(
                    0.5 * (covariance[i][j] + covariance[j][i])
                    for j in range(len(hessian))
                )
                for i in range(len(hessian))
            )
        except ValueError:
            covariance = None
        return DiscrepancyFit(
            self.vector_names,
            best.x,
            covariance,
            -best.value,
            objective,
            best.converged,
            evaluations,
            best,
        )

    # -- sampling -------------------------------------------------------------

    def sample_posterior(
        self,
        *,
        draws: int = 2000,
        warmup: int = 1000,
        chains: int = 4,
        thin: int = 1,
        dispersion: float = 2.0,
        seed: int | RandomStream | None = None,
        method: str = "adaptive-metropolis",
    ) -> DiscrepancyPosterior:
        """Sample ``(theta, log psi, a, b)`` by MCMC in standard-normal ``theta`` space.

        The procedure mirrors :meth:`CalibrationProblem.sample_posterior`:
        Laplace approximation of :meth:`standard_log_posterior` from ``u = 0``
        and prior medians, over-dispersed starts, then adaptive Metropolis
        (default) or HMC/NUTS with central-difference gradients. Without free
        hyperparameters and with known noise it reproduces the
        ``CalibrationProblem`` chains for the same seed up to rounding.

        Args:
            draws: Kept draws per chain.
            warmup: Warm-up iterations per chain.
            chains: Number of chains.
            thin: Thinning interval.
            dispersion: Start spread relative to the Laplace scale.
            seed: Integer seed or stream.
            method: ``"adaptive-metropolis"``, ``"hmc"``, or ``"nuts"``.

        Returns:
            Posterior in vector, physical, and sampling coordinates.

        Raises:
            ValueError: For invalid run arguments or an unknown method.

        Equation:
            [EQ-KOH-POST-001], [EQ-MCMC-AM-001], [EQ-HMC-LEAPFROG-001],
            [EQ-NUTS-SLICE-001].

        References:
            [REF-HAARIO-SAKSMAN-TAMMINEN-2001], [REF-HOFFMAN-GELMAN-2014],
            [REF-KENNEDY-OHAGAN-2001], [REF-HIGDON-ETAL-2004].

        """
        if method not in ("adaptive-metropolis", "hmc", "nuts"):
            raise ValueError("method must be 'adaptive-metropolis', 'hmc', or 'nuts'")
        spread = require_finite_real(dispersion, name="dispersion")
        if spread <= 0.0:
            raise ValueError("dispersion must be positive")
        chain_count = require_non_negative_integer(chains, name="chains")
        if chain_count == 0:
            raise ValueError("chains must be positive")
        stream = resolve_stream(seed)
        size = self.dimension
        origin = tuple(0.0 for _ in range(self.model.dimension)) + self._default_extra()
        laplace: LaplaceApproximation | None
        try:
            laplace = laplace_approximation(self.standard_log_posterior, origin)
        except ValueError:
            laplace = None
        if laplace is not None:
            center, covariance = laplace.mode, laplace.covariance
        else:
            center = origin
            covariance = tuple(
                tuple(0.01 if i == j else 0.0 for j in range(size)) for i in range(size)
            )
        lower = cholesky(covariance)
        generator = stream.child("starts").generator()
        starts: list[Vector] = []
        for _ in range(chain_count):
            start = center
            for _attempt in range(100):
                z = tuple(generator.gauss(0.0, spread) for _ in range(size))
                candidate = tuple(
                    c + s for c, s in zip(center, matvec(lower, z), strict=True)
                )
                if math.isfinite(self.standard_log_posterior(candidate)):
                    start = candidate
                    break
            starts.append(start)
        d = self.model.dimension
        names = tuple(f"u[{name}]" for name in self.parameter_names)
        names += self.vector_names[d:]
        sampler: HamiltonianResult | None = None
        if method == "adaptive-metropolis":
            sampled = adaptive_metropolis(
                self.standard_log_posterior,
                tuple(starts),
                draws=draws,
                warmup=warmup,
                chains=chains,
                thin=thin,
                initial_covariance=scale_matrix(covariance, 2.38**2 / size),
                seed=stream.child("chains"),
                names=names,
            )
        else:
            run = no_u_turn_sampler if method == "nuts" else hamiltonian_monte_carlo
            sampler = run(
                self.standard_log_posterior,
                tuple(starts),
                draws=draws,
                warmup=warmup,
                chains=chains,
                thin=thin,
                metric="diag",
                initial_inverse_metric=covariance,
                seed=stream.child("chains"),
                names=names,
            )
            sampled = sampler.mcmc
        vector = sampled.transformed(self.to_vector, self.vector_names)
        physical = vector.transformed(self.to_physical, self.physical_names)
        return DiscrepancyPosterior(vector, physical, sampled, laplace, sampler)

    # -- prediction -----------------------------------------------------------

    def predict(
        self,
        source: DiscrepancyPosterior | DiscrepancyFit | Sequence[Sequence[float]],
        conditions: Sequence[Mapping[str, float]],
        *,
        max_draws: int | None = 1000,
        observation_standard_deviations: Sequence[float] | None = None,
    ) -> tuple[DiscrepancyPrediction, ...]:
        """Predict the simulator and the bias-corrected process at new conditions.

        For every parameter draw (posterior draws, the point estimate of a
        fit, or explicit vectors) with residuals ``r = y - eta(C, theta)``
        and ``C_psi = L L'``: ``m = k*' C_psi^-1 r`` and
        ``v = k(x*, x*) - |L^-1 k*|^2`` are the Gaussian conditional moments
        of ``delta(x*)`` given the data. The simulator alone is
        ``eta(c*, theta)``. Noise variances at ``c*`` come from the
        log-linear model or, for known noise, from
        ``observation_standard_deviations``.

        Args:
            source: Posterior, fit, or vectors ``(theta, log psi, a, b)``.
            conditions: New condition mappings (merged over ``model.fixed``).
            max_draws: Use at most this many evenly spaced posterior draws
                (``None`` for all).
            observation_standard_deviations: Known noise at each new
                condition (only for :class:`KnownNoise`).

        Returns:
            One :class:`DiscrepancyPrediction` per condition.

        Raises:
            ValueError: For empty inputs, missing discrepancy inputs, a
                parameter shadowed by a condition, or mismatched lengths.

        Equation:
            [EQ-KOH-PRED-001], [EQ-KOH-HET-001].

        References:
            [REF-KENNEDY-OHAGAN-2001], [REF-RASMUSSEN-WILLIAMS-2006, Sec. 2.2].

        """
        if isinstance(source, DiscrepancyPosterior):
            vectors: tuple[Vector, ...] = source.vector.pooled()
        elif isinstance(source, DiscrepancyFit):
            vectors = (source.estimate,)
        else:
            vectors = tuple(as_vector(row, name="parameter draw") for row in source)
        if not vectors:
            raise ValueError("at least one parameter draw is required")
        if max_draws is not None:
            limit = require_non_negative_integer(max_draws, name="max_draws")
            if limit == 0:
                raise ValueError("max_draws must be positive")
            if len(vectors) > limit:
                step = len(vectors) / limit
                vectors = tuple(vectors[int(i * step)] for i in range(limit))
        new = tuple(conditions)
        if not new:
            raise ValueError("at least one prediction condition is required")
        parameters = set(self.model.inputs.names)
        bases: list[Mapping[str, float]] = []
        for index, condition in enumerate(new):
            checked = {
                str(k): require_finite_real(v, name=f"conditions[{index}][{k!r}]")
                for k, v in dict(condition).items()
            }
            shadow = parameters & set(checked)
            if shadow:
                raise ValueError(
                    f"conditions {index} shadow parameters {sorted(shadow)}"
                )
            base = dict(self.model.fixed)
            base.update(checked)
            bases.append(MappingProxyType(base))
        points: tuple[Vector, ...] = ()
        if self.discrepancy is not None:
            names = self.discrepancy.inputs
            for index, merged in enumerate(bases):
                missing = [name for name in names if name not in merged]
                if missing:
                    raise ValueError(
                        f"conditions {index} lack discrepancy inputs {missing}"
                    )
            points = tuple(tuple(base[name] for name in names) for base in bases)
        known_new: Vector | None = None
        new_covariates: tuple[Vector, ...] = ()
        if isinstance(self.noise, LogLinearNoise):
            new_covariates = tuple(self.noise.evaluate(base) for base in bases)
        elif observation_standard_deviations is not None:
            known_new = as_vector(
                observation_standard_deviations, name="observation_standard_deviations"
            )
            if len(known_new) != len(bases) or any(s <= 0.0 for s in known_new):
                raise ValueError(
                    "observation_standard_deviations must be positive, "
                    "one per condition"
                )
        count = len(bases)
        simulator: list[list[float]] = [[] for _ in range(count)]
        means: list[list[float]] = [[] for _ in range(count)]
        variances: list[list[float]] = [[] for _ in range(count)]
        noise: list[list[float]] = [[] for _ in range(count)]
        for vector in vectors:
            theta, sigma_delta, lengths, coefficients = self._unpack(vector)
            alpha: Vector = ()
            lower: Matrix = ()
            if self.is_dense:
                residuals = tuple(
                    y - p
                    for y, p in zip(
                        self._observed, self.predictions(theta), strict=True
                    )
                )
                lower = cholesky(
                    self._covariance(
                        sigma_delta, lengths, self._noise_sds(coefficients)
                    )
                )
                alpha = cholesky_solve(lower, residuals)
            for j, merged in enumerate(bases):
                simulator[j].append(self._simulate(merged, theta))
                if self.is_dense:
                    assert self.discrepancy is not None
                    kernel = self.discrepancy.kernel
                    variance = sigma_delta * sigma_delta
                    cross = tuple(
                        discrepancy_kernel(kernel, points[j], x, lengths, variance)
                        for x in self._points
                    )
                    v = solve_lower(lower, cross)
                    means[j].append(
                        math.fsum(a * k for a, k in zip(alpha, cross, strict=True))
                    )
                    variances[j].append(
                        max(variance - math.fsum(t * t for t in v), 0.0)
                    )
                else:
                    means[j].append(0.0)
                    variances[j].append(0.0)
                if isinstance(self.noise, LogLinearNoise):
                    sd = self.noise.standard_deviation(coefficients, new_covariates[j])
                    noise[j].append(sd * sd)
                elif known_new is not None:
                    noise[j].append(known_new[j] ** 2)
        has_noise = isinstance(self.noise, LogLinearNoise) or known_new is not None
        return tuple(
            DiscrepancyPrediction(
                bases[j],
                tuple(simulator[j]),
                tuple(means[j]),
                tuple(variances[j]),
                tuple(noise[j]) if has_noise else None,
            )
            for j in range(count)
        )

    # -- diagnostics ----------------------------------------------------------

    def identifiability(
        self,
        result: DiscrepancyPosterior | DiscrepancyFit,
        *,
        correlation_warning: float = 0.5,
        ratio_warning: float = 0.8,
    ) -> DiscrepancyIdentifiability:
        """Report theta-discrepancy confounding from a posterior or a fit.

        Two symptoms are reported: (1) the posterior correlation between each
        ``theta_j`` and each free discrepancy hyperparameter (in vector
        coordinates, i.e. ``log sigma_delta`` and ``log ell``), and (2) the
        ratio of posterior to prior standard deviation of each ``theta_j``.
        A large ``|rho|`` means the data trade ``theta`` against the
        discrepancy; a ratio near one means the data barely inform ``theta``
        once ``delta`` is allowed. Neither proves identifiability: the
        confounding is between ``theta`` and the discrepancy *function*,
        which is marginalized. Brynjarsdottir and O'Hagan (2014) show that
        only realistic prior information on ``delta`` recovers physical
        parameters.

        Args:
            result: MCMC posterior (sample correlation) or fit (Hessian
                correlation; prior and posterior sds as ``theta`` scales).
            correlation_warning: ``|rho|`` threshold in ``(0, 1)``.
            ratio_warning: Standard-deviation ratio threshold in ``(0, 1]``.

        Returns:
            Correlations, contraction ratios, and a warning text.

        Raises:
            ValueError: For invalid thresholds or a fit without covariance.

        Equation:
            [EQ-KOH-IDENT-001].

        References:
            [REF-BRYNJARSDOTTIR-OHAGAN-2014], [REF-KENNEDY-OHAGAN-2001].

        """
        rho_limit = require_finite_real(correlation_warning, name="correlation_warning")
        if not 0.0 < rho_limit < 1.0:
            raise ValueError("correlation_warning must lie in (0, 1)")
        ratio_limit = require_finite_real(ratio_warning, name="ratio_warning")
        if not 0.0 < ratio_limit <= 1.0:
            raise ValueError("ratio_warning must lie in (0, 1]")
        if isinstance(result, DiscrepancyPosterior):
            covariance = result.vector.covariance()
            source = "mcmc"
        else:
            if result.covariance is None:
                raise ValueError("the fit has no covariance")
            covariance = result.covariance
            source = "laplace"
        names = self.vector_names
        correlation = _correlation_matrix(covariance)
        d = self.model.dimension
        hyper = self.hyperparameter_names
        cross: dict[str, Mapping[str, float]] = {}
        largest = 0.0
        for i, theta_name in enumerate(self.parameter_names):
            row = {name: correlation[i][d + k] for k, name in enumerate(hyper)}
            cross[theta_name] = MappingProxyType(row)
            for value in row.values():
                if math.isfinite(value):
                    largest = max(largest, abs(value))
        prior_sds = self.model.inputs.standard_deviations()
        post_sds = tuple(math.sqrt(max(covariance[i][i], 0.0)) for i in range(d))
        ratios = tuple(
            p / q if q > 0.0 else math.inf
            for p, q in zip(post_sds, prior_sds, strict=True)
        )
        messages: list[str] = []
        if largest >= rho_limit:
            messages.append(
                f"theta and the discrepancy hyperparameters are correlated a "
                f"posteriori (max |rho| = {largest:.2f}): the data trade the "
                "simulator parameters against the discrepancy."
            )
        weak = [
            f"{name} ({ratio:.2f})"
            for name, ratio in zip(self.parameter_names, ratios, strict=True)
            if ratio >= ratio_limit
        ]
        if weak:
            messages.append(
                "posterior/prior standard-deviation ratios near one for "
                + ", ".join(weak)
                + ": the data barely inform these parameters."
            )
        warning = None
        if messages:
            warning = " ".join(messages) + (
                " Theta and delta are confounded; physical parameters are learned "
                "only with realistic prior information on the discrepancy "
                "(Brynjarsdottir and O'Hagan 2014). Bias-corrected predictions "
                "inside the data range remain usable."
            )
        return DiscrepancyIdentifiability(
            names,
            correlation,
            MappingProxyType(cross),
            largest,
            MappingProxyType(dict(zip(self.parameter_names, prior_sds, strict=True))),
            MappingProxyType(dict(zip(self.parameter_names, post_sds, strict=True))),
            MappingProxyType(dict(zip(self.parameter_names, ratios, strict=True))),
            source,
            warning,
        )


def _safe_exp(value: float) -> float:
    """Return ``exp(value)`` or ``inf`` on overflow.

    References:
        [REF-KENNEDY-OHAGAN-2001]. No equation is evaluated.

    """
    return math.exp(value) if value < _EXP_LIMIT else math.inf
