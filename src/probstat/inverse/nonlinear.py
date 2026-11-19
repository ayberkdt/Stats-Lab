"""Bayesian and least-squares calibration of general nonlinear models.

A :class:`CalibrationProblem` couples three declarations:

* a :class:`~probstat.modeling.ProbabilisticModel` whose joint input law is the
  **prior** over the calibration parameters ``theta``;
* a list of :class:`Experiment` records, each with fixed experimental
  conditions ``c_j`` (merged over ``model.fixed``) and observed values of some
  named quantities of interest; and
* one :class:`NoiseModel` per observed quantity: a known Gaussian standard
  deviation, or an unknown one with its own prior, optionally a Student-t
  (robust) likelihood, and an optional additive discrepancy variance.

The observation model is ``y_jq = g_q(theta; c_j) + delta_jq + eps_jq`` with
``Var(delta) = sigma_delta_q^2`` and ``eps`` of scale ``sigma_q``.

Parameter vectors used by :meth:`CalibrationProblem.log_posterior` are
``(theta_1..theta_d, log sigma_q for every unknown-noise quantity)``. The
log-scale makes noise parameters unconstrained; the prior density is
transformed with its Jacobian ``|d sigma / d log sigma| = sigma``.

Posterior sampling (:meth:`CalibrationProblem.sample_posterior`) runs adaptive
Metropolis (default), static HMC, NUTS, preconditioned Crank-Nicolson, or
tempered sequential Monte Carlo in the Nataf standard-normal space
``u = T(theta)`` of the prior.
Because ``T`` pushes the prior forward to ``N(0, I)``, the posterior density in
``u`` is ``L(T^-1(u)) phi(u)`` exactly; the prior Jacobian is absorbed by the
transformation and no determinant needs to be computed. Bounded or skewed
priors become unbounded and roughly isotropic, which conditions the sampler.

Fully standardized coordinates (``method="pcn"`` or ``"smc"``)
    Preconditioned Crank-Nicolson and the evidence need a prior that is
    exactly ``N(0, I)`` in *every* sampled coordinate. The log-noise
    coordinates ``eta = log sigma`` of the default samplers are not: their
    prior is ``p_sigma(e^eta) e^eta``. Those two methods therefore also map
    each unknown noise scale through its own prior quantile transform,
    ``sigma = F_sigma^-1(Phi(w))`` with ``w ~ N(0, 1)``, and work in
    ``v = (u, w)`` with target ``L(T^-1(u), sigma(w)) N(v; 0, I)``
    (:meth:`CalibrationProblem.standard_log_likelihood`). The draws are
    mapped back to ``(u, log sigma)`` for ``standard_space`` and to
    ``(theta, sigma)`` for the physical chains, so the result has the same
    layout for every method. Because the change of variables is exact,
    ``integral L(T^-1(u), sigma(w)) N(v; 0, I) dv`` equals the marginal
    likelihood ``p(y)`` of the original problem, which is what
    :meth:`CalibrationProblem.log_evidence` estimates.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

from probstat.distributions._autodiff import admath as _admath
from probstat.distributions._autodiff import any_variable, primal, real_argument
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.inference import (
    HamiltonianResult,
    LaplaceApproximation,
    MarkovChain,
    MCMCResult,
    ParameterSummary,
    SMCResult,
    adaptive_metropolis,
    hamiltonian_monte_carlo,
    laplace_approximation,
    no_u_turn_sampler,
    preconditioned_crank_nicolson,
    sequential_monte_carlo,
)
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    matmul,
    matvec,
    scale_matrix,
    solve_lower,
    symmetric_eigen,
    transpose,
)
from probstat.modeling import (
    ContinuousMarginal,
    JointInput,
    ProbabilisticModel,
    RandomInput,
)
from probstat.numerics import AutodiffUnsupportedError, reverse_value_and_gradient
from probstat.optimization import (
    LeastSquaresResult,
    central_jacobian,
    gauss_newton_covariance,
    least_squares,
)
from probstat.uq import evaluate_model

from . import _traced
from .error_correlation import ErrorCorrelation, error_correlation_kernel

_LOG_TWO_PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Experiment:
    """Observed quantities of interest under one set of experimental conditions.

    Args:
        observations: Mapping from QoI name to its measured value.
        conditions: Deterministic model inputs for this experiment (load,
            time, temperature, or an index encoded as a float). They override
            ``model.fixed`` entries with the same name and must not name a
            calibration parameter.
        label: Optional human-readable identifier.

    Raises:
        ValueError: If no observation is given or a value is not finite.

    References:
        [REF-KENNEDY-OHAGAN-2001], [REF-SMITH-UQ-2024, Chapter 8].

    """

    observations: Mapping[str, float]
    conditions: Mapping[str, float] = field(default_factory=dict)
    label: str = ""

    def __post_init__(self) -> None:
        """Validate and freeze the observation and condition mappings.

        References:
            [EQ-CAL-NL-001]. No equation is evaluated.

        """
        observed = {
            str(name): require_finite_real(value, name=f"observations[{name!r}]")
            for name, value in dict(self.observations).items()
        }
        if not observed:
            raise ValueError("an experiment needs at least one observation")
        conditions = {
            str(name): require_finite_real(value, name=f"conditions[{name!r}]")
            for name, value in dict(self.conditions).items()
        }
        object.__setattr__(self, "observations", MappingProxyType(observed))
        object.__setattr__(self, "conditions", MappingProxyType(conditions))


@dataclass(frozen=True, slots=True)
class NoiseModel:
    """Observation-error law for one quantity of interest.

    Exactly one of ``standard_deviation`` (known noise) and
    ``standard_deviation_prior`` (unknown noise, sampled on the log scale) must
    be given. With ``degrees_of_freedom = nu`` the error is Student-t with
    scale ``s`` (standard deviation ``s sqrt(nu/(nu-2))`` for ``nu > 2``),
    which down-weights outliers. ``discrepancy_variance`` adds a declared
    independent Gaussian model-inadequacy variance, so the total scale is
    ``s^2 = sigma^2 + sigma_delta^2`` (same semantics as
    ``validation.AdditiveGaussianDiscrepancy.variance``).

    With ``correlation`` the errors of this quantity are jointly Gaussian
    across the experiments that observe it, with covariance
    ``sigma^2 M + sigma_delta^2 I`` (:class:`ErrorCorrelation`,
    [EQ-CAL-CORR-002]). Correlated errors are Gaussian only. Without
    ``correlation`` (the default) every number is that of independent noise.

    Args:
        quantity: Observed QoI name.
        standard_deviation: Known positive noise scale.
        standard_deviation_prior: Continuous prior law of an unknown positive
            noise scale.
        degrees_of_freedom: Optional positive Student-t degrees of freedom.
        discrepancy_variance: Non-negative additive discrepancy variance.
        correlation: Optional correlation of the errors across experiments.

    Raises:
        TypeError: If ``correlation`` is not an :class:`ErrorCorrelation`.
        ValueError: For an invalid combination or non-positive values, or a
            Student-t law combined with correlated errors.

    References:
        [REF-KENNEDY-OHAGAN-2001], [REF-LANGE-LITTLE-TAYLOR-1989],
        [REF-SEBER-WILD-2003, Chapter 6].

    """

    quantity: str
    standard_deviation: float | None = None
    standard_deviation_prior: ContinuousMarginal | None = None
    degrees_of_freedom: float | None = None
    discrepancy_variance: float = 0.0
    correlation: ErrorCorrelation | None = None

    def __post_init__(self) -> None:
        """Validate the noise declaration.

        References:
            [EQ-CAL-LIK-001]. No equation is evaluated.

        """
        if not isinstance(self.quantity, str) or not self.quantity.strip():
            raise ValueError("quantity must be a non-empty string")
        known = self.standard_deviation is not None
        unknown = self.standard_deviation_prior is not None
        if known == unknown:
            raise ValueError(
                "give exactly one of standard_deviation or standard_deviation_prior"
            )
        if self.standard_deviation is not None:
            value = require_finite_real(
                self.standard_deviation, name="standard_deviation"
            )
            if value <= 0.0:
                raise ValueError("standard_deviation must be positive")
            object.__setattr__(self, "standard_deviation", value)
        if self.standard_deviation_prior is not None:
            RandomInput(f"sigma[{self.quantity}]", self.standard_deviation_prior)
        if self.degrees_of_freedom is not None:
            nu = require_finite_real(self.degrees_of_freedom, name="degrees_of_freedom")
            if nu <= 0.0:
                raise ValueError("degrees_of_freedom must be positive")
            object.__setattr__(self, "degrees_of_freedom", nu)
        variance = require_finite_real(
            self.discrepancy_variance, name="discrepancy_variance"
        )
        if variance < 0.0:
            raise ValueError("discrepancy_variance must be non-negative")
        object.__setattr__(self, "discrepancy_variance", variance)
        if self.correlation is not None:
            if not isinstance(self.correlation, ErrorCorrelation):
                raise TypeError("correlation must be an ErrorCorrelation")
            if self.degrees_of_freedom is not None:
                raise ValueError(
                    "correlated measurement errors are Gaussian only; a "
                    "Student-t law (degrees_of_freedom) cannot be combined "
                    "with correlation"
                )

    @property
    def is_known(self) -> bool:
        """Return whether the noise scale is a declared constant.

        References:
            [EQ-CAL-LIK-001]. No equation is evaluated.

        """
        return self.standard_deviation is not None

    def log_density(self, residual: float, sigma: float) -> float:
        """Return the log-density of one residual under this noise law.

        Args:
            residual: Observed minus predicted value.
            sigma: Noise scale (known or current unknown value).

        For a correlated quantity this is the marginal density of one
        error; the joint density of the correlated errors is evaluated by
        :meth:`CalibrationProblem.log_likelihood`.

        Returns:
            Gaussian ``-0.5(log(2 pi s^2) + r^2/s^2)`` or Student-t
            ``log G((nu+1)/2) - log G(nu/2) - 0.5 log(nu pi s^2)
            - (nu+1)/2 log(1 + r^2/(nu s^2))`` with ``s^2 = sigma^2 +
            sigma_delta^2``.

        Equation:
            [EQ-CAL-LIK-001].

        References:
            [REF-GELMAN-ETAL-2013, Section 17.2], [REF-LANGE-LITTLE-TAYLOR-1989].

        """
        total = sigma * sigma + self.discrepancy_variance
        if self.degrees_of_freedom is None:
            return -0.5 * (_LOG_TWO_PI + math.log(total) + residual * residual / total)
        nu = self.degrees_of_freedom
        return (
            math.lgamma(0.5 * (nu + 1.0))
            - math.lgamma(0.5 * nu)
            - 0.5 * math.log(nu * math.pi * total)
            - 0.5 * (nu + 1.0) * math.log1p(residual * residual / (nu * total))
        )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FisherIdentifiability:
    """Eigen-analysis of the Gauss-Newton (Fisher) information ``J'J``.

    Args:
        names: Parameter names.
        eigenvalues: Ascending eigenvalues of the (optionally scaled) matrix.
        eigenvectors: Unit eigenvectors, one per eigenvalue.
        condition_number: ``lambda_max / lambda_min`` (``inf`` if singular).
        rank: Number of eigenvalues above ``rank_tolerance * lambda_max``.
        identifiable: Whether the information matrix has full rank.
        weak_directions: ``(eigenvalue, {name: component})`` for eigenvalues
            below ``lambda_max / condition_warning``, weakest first.
        warning: Human-readable warning or ``None``.

    References:
        [REF-SEBER-WILD-2003, Section 3.4], [REF-TARANTOLA-2005, Chapter 3].

    """

    names: tuple[str, ...]
    eigenvalues: Vector
    eigenvectors: tuple[Vector, ...]
    condition_number: float
    rank: int
    identifiable: bool
    weak_directions: tuple[tuple[float, Mapping[str, float]], ...]
    warning: str | None


@dataclass(frozen=True, slots=True)
class LeastSquaresCalibration:
    """Weighted nonlinear least-squares estimate with Gauss-Newton covariance.

    Args:
        names: Calibration-parameter names.
        estimate: Least-squares parameter vector.
        covariance: Gauss-Newton covariance ``(J_w'J_w)^-1`` of the weighted
            problem.
        noise_standard_deviations: Known or estimated noise scale per QoI.
        residuals: Raw observed-minus-predicted residuals.
        weighted_cost: ``0.5 sum (r_i / s_i)^2`` at the estimate.
        identifiability: Fisher eigen-analysis at the estimate.
        converged: Whether Levenberg-Marquardt met its tolerance.
        evaluations: Forward-model evaluations used.
        solver: Raw solver result for the final weighted problem.

    References:
        [REF-SEBER-WILD-2003, Chapter 2], [REF-NOCEDAL-WRIGHT-2006, Ch. 10].

    """

    names: tuple[str, ...]
    estimate: Vector
    covariance: Matrix
    noise_standard_deviations: Mapping[str, float]
    residuals: Vector
    weighted_cost: float
    identifiability: FisherIdentifiability
    converged: bool
    evaluations: int
    solver: LeastSquaresResult

    @property
    def standard_errors(self) -> Vector:
        """Return asymptotic standard errors ``sqrt(diag(covariance))``.

        Equation:
            [EQ-OPT-LM-002].

        References:
            [REF-SEBER-WILD-2003, Section 2.1].

        """
        return tuple(math.sqrt(self.covariance[i][i]) for i in range(len(self.names)))

    def as_mapping(self) -> Mapping[str, float]:
        """Return the estimate keyed by parameter name.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return MappingProxyType(dict(zip(self.names, self.estimate, strict=True)))


@dataclass(frozen=True, slots=True)
class CalibrationPosterior:
    """MCMC posterior of calibration parameters and unknown noise scales.

    Args:
        parameter_names: Calibration-parameter names.
        noise_names: ``sigma[q]`` names of unknown noise scales.
        mcmc: Chains on the physical scale ``(theta, sigma)``.
        standard_space: Raw chains in the sampling coordinates
            ``(u, log sigma)``.
        laplace: Laplace approximation in sampling coordinates used to start
            the chains (``None`` if the Hessian was not positive definite).
        sampler: HMC/NUTS diagnostics (divergences, tree depths, step sizes)
            in sampling coordinates; ``None`` for the other methods.
        smc: Full sequential Monte Carlo result (temperatures, acceptance
            rates, log evidence) in the fully standardized coordinates
            ``(u, w)``; ``None`` unless ``method="smc"``. For SMC, ``mcmc``
            and ``standard_space`` hold the final particles as one pseudo
            chain: means and quantiles are valid, but the chain diagnostics
            (``R-hat``, ESS) of :meth:`summary` do not apply to particles.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 11], [REF-KENNEDY-OHAGAN-2001],
        [REF-HOFFMAN-GELMAN-2014], [REF-DELMORAL-DOUCET-JASRA-2006].

    """

    parameter_names: tuple[str, ...]
    noise_names: tuple[str, ...]
    mcmc: MCMCResult
    standard_space: MCMCResult
    laplace: LaplaceApproximation | None
    sampler: HamiltonianResult | None = None
    smc: SMCResult | None = None

    @property
    def evaluations(self) -> int:
        """Return log-posterior evaluations used by the chains.

        References:
            [REF-ROBERT-CASELLA-2004, Chapter 7]. No equation is evaluated.

        """
        return self.mcmc.evaluations

    def draws(self) -> tuple[Mapping[str, float], ...]:
        """Return pooled physical draws keyed by name.

        References:
            [REF-GELMAN-ETAL-2013, Section 11.4]. No equation is evaluated.

        """
        names = self.mcmc.names
        return tuple(
            MappingProxyType(dict(zip(names, sample, strict=True)))
            for sample in self.mcmc.pooled()
        )

    def mean(self) -> Mapping[str, float]:
        """Return the pooled posterior mean keyed by name.

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
        """Summarize each physical coordinate with diagnostics.

        Equation:
            [EQ-MCMC-RHAT-002], [EQ-MCMC-ESS-001], [EQ-MCMC-MCSE-001].

        References:
            [REF-VEHTARI-ETAL-2021].

        """
        return self.mcmc.summary(probabilities)


# ---------------------------------------------------------------------------
# Identifiability
# ---------------------------------------------------------------------------


def fisher_identifiability(
    jacobian: Sequence[Sequence[float]],
    names: Sequence[str],
    *,
    scales: Sequence[float] | None = None,
    condition_warning: float = 1.0e8,
    rank_tolerance: float = 1.0e-10,
) -> FisherIdentifiability:
    """Diagnose local identifiability from a weighted residual Jacobian.

    For Gaussian errors with unit-variance weighted residuals, ``F = J'J`` is
    the Fisher information (Gauss-Newton Hessian). With ``D = diag(scales)``
    the matrix ``D F D`` is analysed instead; with prior standard deviations as
    scales its eigenvalues compare data information to prior information
    (``>> 1``: data-dominated, ``<< 1``: prior-dominated direction). Small
    eigenvalues identify weakly informed parameter **combinations** (their
    eigenvectors), not merely weak individual parameters. The result is local:
    it holds at the Jacobian's linearization point.

    Args:
        jacobian: ``m`` by ``d`` Jacobian of weighted residuals or predictions.
        names: ``d`` parameter names.
        scales: Optional positive parameter scales.
        condition_warning: Threshold above one for the warning.
        rank_tolerance: Relative eigenvalue threshold for numerical rank.

    Returns:
        Eigenvalues, eigenvectors, condition number, rank, and weak directions.

    Raises:
        ValueError: If shapes disagree or thresholds are invalid.

    Equation:
        [EQ-CAL-FISHER-001], [EQ-LINALG-005].

    References:
        [REF-SEBER-WILD-2003, Section 3.4], [REF-TARANTOLA-2005, Chapter 3],
        [REF-GOLUB-VANLOAN-2013, Section 8.5].

    """
    rows = tuple(as_vector(row, name="jacobian row") for row in jacobian)
    labels = tuple(names)
    if not rows or any(len(row) != len(labels) for row in rows) or not labels:
        raise ValueError("jacobian must have one column per parameter name")
    threshold = require_finite_real(condition_warning, name="condition_warning")
    if threshold <= 1.0:
        raise ValueError("condition_warning must be greater than one")
    tolerance = require_finite_real(rank_tolerance, name="rank_tolerance")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("rank_tolerance must lie in (0, 1)")
    information = matmul(transpose(rows), rows)
    if scales is not None:
        factors = as_vector(scales, name="scales")
        if len(factors) != len(labels) or any(f <= 0.0 for f in factors):
            raise ValueError("scales must be positive and match the names")
        information = tuple(
            tuple(
                information[i][j] * factors[i] * factors[j] for j in range(len(labels))
            )
            for i in range(len(labels))
        )
    values, vectors = symmetric_eigen(information)
    size = len(labels)
    directions = tuple(
        tuple(vectors[row][column] for row in range(size)) for column in range(size)
    )
    largest = max(values)
    if largest <= 0.0:
        rank = 0
    else:
        rank = sum(value > tolerance * largest for value in values)
    smallest = values[0]
    condition = largest / smallest if smallest > 0.0 and rank == size else math.inf
    weak = tuple(
        (
            value,
            MappingProxyType(dict(zip(labels, direction, strict=True))),
        )
        for value, direction in zip(values, directions, strict=True)
        if largest <= 0.0 or value < largest / threshold
    )
    identifiable = rank == size
    if not identifiable:
        warning: str | None = (
            "The information matrix is rank deficient: some parameter "
            "combinations are not identifiable from these data."
        )
    elif condition >= threshold:
        warning = (
            "The information matrix is full rank but ill-conditioned: some "
            "parameter combinations are only weakly identified."
        )
    else:
        warning = None
    return FisherIdentifiability(
        labels,
        values,
        directions,
        condition,
        rank,
        identifiable,
        weak,
        warning,
    )


# ---------------------------------------------------------------------------
# Correlated measurement errors
# ---------------------------------------------------------------------------

CALIBRATION_GRADIENTS = ("central", "reverse")
"""Gradient options of ``laplace_posterior`` and HMC/NUTS ``sample_posterior``."""


def _check_gradient_option(gradient: str) -> None:
    """Raise for a gradient option other than ``"central"`` or ``"reverse"``.

    References:
        [EQ-AD-REVERSE-001], [EQ-OPT-FD-001]. No equation is evaluated.

    """
    if gradient not in CALIBRATION_GRADIENTS:
        raise ValueError("gradient must be 'central' or 'reverse'")


@dataclass(frozen=True, slots=True)
class _ErrorGroup:
    """Observations of one quantity whose errors are jointly Gaussian.

    ``positions`` index the problem's observation list (experiment order);
    ``points`` are the condition values of a kernel structure. For a known
    structure ``shape`` is the matrix ``M``; with no discrepancy variance
    also its Cholesky factor and log-determinant are cached, so the
    likelihood costs one triangular solve per evaluation.

    References:
        [EQ-CAL-CORR-001], [EQ-CAL-CORR-002], [REF-SEBER-WILD-2003, Chapter 6].

    """

    quantity: str
    positions: tuple[int, ...]
    points: tuple[float, ...]
    structure: ErrorCorrelation
    discrepancy_variance: float
    shape: Matrix | None
    shape_lower: Matrix | None
    shape_log_det: float

    @property
    def size(self) -> int:
        """Return the number of correlated observations.

        References:
            [EQ-CAL-CORR-002]. No equation is evaluated.

        """
        return len(self.positions)

    def matrix(self, length: float | None) -> Matrix:
        """Return the shape matrix ``M`` (cached when the structure is known).

        References:
            [EQ-CAL-CORR-001]. No new equation is introduced.

        """
        if self.shape is not None:
            return self.shape
        return self.structure.correlation_matrix(self.points, length)

    def covariance(self, sigma: float, length: float | None) -> Matrix:
        """Return ``Sigma = sigma^2 M + sigma_delta^2 I``.

        References:
            [EQ-CAL-CORR-002]. No new equation is introduced.

        """
        shape = self.matrix(length)
        variance = sigma * sigma
        return tuple(
            tuple(
                variance * shape[i][j] + (self.discrepancy_variance if i == j else 0.0)
                for j in range(self.size)
            )
            for i in range(self.size)
        )

    def factor(self, sigma: float, length: float | None) -> tuple[Matrix, float]:
        """Return a lower factor ``F`` and divisor ``c`` with ``Sigma = c^2 F F'``.

        Raises:
            ValueError: If ``Sigma`` is not numerically positive definite.

        References:
            [EQ-CAL-CORR-003], [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        if self.discrepancy_variance == 0.0 and self.shape_lower is not None:
            return self.shape_lower, sigma
        return cholesky(self.covariance(sigma, length)), 1.0

    def log_density(
        self, residuals: Sequence[float], sigma: float, length: float | None
    ) -> float:
        """Return the joint Gaussian log-density of the group's residuals.

        ``-0.5 (n log 2 pi + log det Sigma + r' Sigma^-1 r)`` by Cholesky; a
        covariance that is not numerically positive definite (possible only
        for a sampled correlation length) gives ``-inf``.

        Equation:
            [EQ-CAL-CORR-002].

        References:
            [REF-SEBER-WILD-2003, Chapter 6], [REF-GOLUB-VANLOAN-2013,
            Section 4.2].

        """
        count = self.size
        if self.discrepancy_variance == 0.0 and self.shape_lower is not None:
            whitened = solve_lower(self.shape_lower, tuple(residuals))
            variance = sigma * sigma
            quadratic = math.fsum(w * w for w in whitened) / variance
            log_det = count * math.log(variance) + self.shape_log_det
        else:
            try:
                lower = cholesky(self.covariance(sigma, length))
            except ValueError:
                return -math.inf
            whitened = solve_lower(lower, tuple(residuals))
            quadratic = math.fsum(w * w for w in whitened)
            log_det = 2.0 * math.fsum(math.log(lower[i][i]) for i in range(count))
        return -0.5 * (count * _LOG_TWO_PI + log_det + quadratic)

    def traced_log_density(
        self, residuals: Sequence[float], sigma: float, length: float | None
    ) -> float:
        """Return :meth:`log_density` for reverse-mode variables.

        Equation:
            [EQ-CAL-CORR-002], [EQ-AD-REVERSE-001].

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        count = self.size
        if self.discrepancy_variance == 0.0 and self.shape_lower is not None:
            whitened = _traced.solve_lower(self.shape_lower, residuals)
            variance = sigma * sigma
            quadratic = _admath.fsum(w * w for w in whitened) / variance
            log_det = count * _admath.log(variance) + self.shape_log_det
        else:
            if self.shape is not None:
                shape: Sequence[Sequence[float]] = self.shape
            else:
                if length is None:
                    raise ValueError("a correlation length is required")
                kernel = self.structure.kernel
                shape = [
                    [
                        1.0
                        if i == j
                        else error_correlation_kernel(kernel, abs(a - b) / length)
                        for j, b in enumerate(self.points)
                    ]
                    for i, a in enumerate(self.points)
                ]
            variance = sigma * sigma
            covariance = [
                [
                    variance * shape[i][j]
                    + (self.discrepancy_variance if i == j else 0.0)
                    for j in range(count)
                ]
                for i in range(count)
            ]
            lower = _traced.cholesky_lower(covariance)
            if lower is None:
                return -math.inf
            whitened = _traced.solve_lower(lower, residuals)
            quadratic = _admath.fsum(w * w for w in whitened)
            log_det = 2.0 * _admath.fsum(_admath.log(lower[i][i]) for i in range(count))
        return -0.5 * (count * _LOG_TWO_PI + log_det + quadratic)


def _error_group(
    item: NoiseModel,
    positions: tuple[int, ...],
    bases: Sequence[Mapping[str, float]],
    pairs: Sequence[tuple[int, str, float]],
    parameters: set[str],
) -> _ErrorGroup:
    """Validate a correlated quantity against the experiments and cache ``M``.

    Raises:
        ValueError: If a declared matrix has the wrong size, the condition is
            missing or names a parameter, condition values repeat without a
            discrepancy variance, or a known ``M`` is not positive definite.

    References:
        [EQ-CAL-CORR-001], [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    structure = item.correlation
    assert structure is not None
    quantity = item.quantity
    points: tuple[float, ...] = ()
    if structure.condition is not None:
        name = structure.condition
        if name in parameters:
            raise ValueError(
                f"the error correlation of {quantity!r} uses {name!r}, which is a "
                "calibration parameter, not an experimental condition"
            )
        values: list[float] = []
        for position in positions:
            experiment = pairs[position][0]
            if name not in bases[experiment]:
                raise ValueError(
                    f"experiment {experiment} lacks the condition {name!r} used by "
                    f"the error correlation of {quantity!r}"
                )
            values.append(float(bases[experiment][name]))
        points = tuple(values)
        if item.discrepancy_variance == 0.0 and len(set(points)) < len(points):
            raise ValueError(
                f"repeated values of {name!r} make the error correlation of "
                f"{quantity!r} singular; declare a discrepancy_variance (nugget) "
                "or distinct conditions"
            )
    elif structure.matrix is not None and len(structure.matrix) != len(positions):
        raise ValueError(
            f"the error correlation matrix of {quantity!r} has "
            f"{len(structure.matrix)} rows but {len(positions)} experiments "
            "observe it"
        )
    shape: Matrix | None = None
    lower: Matrix | None = None
    log_det = 0.0
    if structure.is_known:
        shape = structure.correlation_matrix(points or tuple(0.0 for _ in positions))
        if item.discrepancy_variance == 0.0:
            # With a nugget the covariance sigma^2 M + sigma_delta^2 I is
            # positive definite for the positive-semidefinite kernel matrix M.
            try:
                factor = cholesky(shape)
            except ValueError as error:
                raise ValueError(
                    f"the error correlation matrix of {quantity!r} is not "
                    "numerically positive definite (a correlation length that is "
                    "long compared with the spacing of the conditions); declare a "
                    "discrepancy_variance (nugget) or a shorter length"
                ) from error
            lower = factor
            log_det = 2.0 * math.fsum(
                math.log(factor[i][i]) for i in range(len(factor))
            )
    return _ErrorGroup(
        quantity,
        positions,
        points,
        structure,
        item.discrepancy_variance,
        shape,
        lower,
        log_det,
    )


# ---------------------------------------------------------------------------
# The calibration problem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationProblem:
    """Prior, forward model, experiments, and noise model of one calibration.

    Args:
        model: Probabilistic model whose ``inputs`` are the prior over the
            calibration parameters.
        experiments: Observed experiments.
        noise: One noise model per observed quantity.

    Raises:
        TypeError: If components have the wrong types.
        ValueError: If a quantity is undeclared, lacks a noise model, is
            declared twice, or a condition shadows a calibration parameter.

    References:
        [REF-KENNEDY-OHAGAN-2001], [REF-GELMAN-ETAL-2013, Chapter 1],
        [REF-TARANTOLA-2005, Chapter 1].

    """

    model: ProbabilisticModel
    experiments: tuple[Experiment, ...]
    noise: tuple[NoiseModel, ...]
    _pairs: tuple[tuple[int, str, float], ...] = field(
        init=False, repr=False, compare=False
    )
    _noise_by_name: Mapping[str, NoiseModel] = field(
        init=False, repr=False, compare=False
    )
    _unknown: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _sigma_priors: tuple[RandomInput, ...] = field(
        init=False, repr=False, compare=False
    )
    _bases: tuple[Mapping[str, float], ...] = field(
        init=False, repr=False, compare=False
    )
    _noise_inputs: JointInput | None = field(init=False, repr=False, compare=False)
    _groups: tuple[_ErrorGroup, ...] = field(init=False, repr=False, compare=False)
    _independent: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _unknown_lengths: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _prior_lower: Matrix | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the declarations and cache the observation layout.

        References:
            [EQ-CAL-NL-001]. No equation is evaluated.

        """
        if not isinstance(self.model, ProbabilisticModel):
            raise TypeError("model must be a ProbabilisticModel")
        experiments = tuple(self.experiments)
        noise = tuple(self.noise)
        if not experiments:
            raise ValueError("at least one experiment is required")
        for index, experiment in enumerate(experiments):
            if not isinstance(experiment, Experiment):
                raise TypeError(f"experiments[{index}] must be an Experiment")
        for index, item in enumerate(noise):
            if not isinstance(item, NoiseModel):
                raise TypeError(f"noise[{index}] must be a NoiseModel")
        by_name: dict[str, NoiseModel] = {}
        declared = set(self.model.quantity_names)
        for item in noise:
            if item.quantity in by_name:
                raise ValueError(f"duplicate noise model for {item.quantity!r}")
            if item.quantity not in declared:
                raise ValueError(f"{item.quantity!r} is not a model quantity")
            by_name[item.quantity] = item
        parameters = set(self.model.inputs.names)
        pairs: list[tuple[int, str, float]] = []
        for index, experiment in enumerate(experiments):
            shadow = parameters & set(experiment.conditions)
            if shadow:
                raise ValueError(
                    f"experiment {index} conditions shadow parameters {sorted(shadow)}"
                )
            for quantity, value in experiment.observations.items():
                if quantity not in by_name:
                    raise ValueError(f"no noise model for observed {quantity!r}")
                pairs.append((index, quantity, value))
        unknown = tuple(item.quantity for item in noise if not item.is_known)
        priors = tuple(
            RandomInput(f"sigma[{item.quantity}]", item.standard_deviation_prior)
            for item in noise
            if item.standard_deviation_prior is not None
        )
        lengths = tuple(
            item.quantity
            for item in noise
            if item.correlation is not None
            and item.correlation.length_prior is not None
        )
        priors = priors + tuple(
            RandomInput(f"length[{item.quantity}]", item.correlation.length_prior)
            for item in noise
            if item.correlation is not None
            and item.correlation.length_prior is not None
        )
        object.__setattr__(self, "experiments", experiments)
        object.__setattr__(self, "noise", noise)
        object.__setattr__(self, "_pairs", tuple(pairs))
        object.__setattr__(self, "_noise_by_name", MappingProxyType(by_name))
        object.__setattr__(self, "_unknown", unknown)
        object.__setattr__(self, "_sigma_priors", priors)
        object.__setattr__(
            self, "_noise_inputs", JointInput.independent(*priors) if priors else None
        )
        bases = []
        for experiment in experiments:
            base = dict(self.model.fixed)
            base.update(experiment.conditions)
            bases.append(MappingProxyType(base))
        object.__setattr__(self, "_bases", tuple(bases))
        groups: list[_ErrorGroup] = []
        grouped: set[int] = set()
        for item in noise:
            if item.correlation is None:
                continue
            positions = tuple(
                position
                for position, (_, quantity, _) in enumerate(pairs)
                if quantity == item.quantity
            )
            if not positions:
                continue
            groups.append(_error_group(item, positions, bases, pairs, parameters))
            grouped.update(positions)
        object.__setattr__(self, "_groups", tuple(groups))
        object.__setattr__(
            self,
            "_independent",
            tuple(
                position for position in range(len(pairs)) if position not in grouped
            ),
        )
        object.__setattr__(self, "_unknown_lengths", lengths)
        object.__setattr__(self, "_prior_lower", _traced.prior_lower(self.model.inputs))

    # -- layout -------------------------------------------------------------

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return calibration-parameter names (the prior's input names).

        References:
            [REF-SMITH-UQ-2024, Chapter 8]. No equation is evaluated.

        """
        return self.model.inputs.names

    @property
    def noise_names(self) -> tuple[str, ...]:
        """Return names of the uncertain noise parameters.

        ``sigma[q]`` for every unknown noise scale, followed by ``length[q]``
        for every uncertain error-correlation length (none unless a
        :class:`ErrorCorrelation` declares a ``length_prior``).

        References:
            [REF-GELMAN-ETAL-2013, Section 14.2]. No equation is evaluated.

        """
        return tuple(f"sigma[{q}]" for q in self._unknown) + tuple(
            f"length[{q}]" for q in self._unknown_lengths
        )

    @property
    def vector_names(self) -> tuple[str, ...]:
        """Return names of the full vector ``(theta, log sigma, log length)``.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return (
            self.parameter_names
            + tuple(f"log_sigma[{q}]" for q in self._unknown)
            + tuple(f"log_length[{q}]" for q in self._unknown_lengths)
        )

    @property
    def dimension(self) -> int:
        """Return the length of the full parameter vector.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return self.model.dimension + len(self._unknown) + len(self._unknown_lengths)

    @property
    def has_correlated_errors(self) -> bool:
        """Return whether any observed quantity has correlated errors.

        References:
            [EQ-CAL-CORR-002]. No equation is evaluated.

        """
        return bool(self._groups)

    @property
    def observation_count(self) -> int:
        """Return the number of scalar observations.

        References:
            [REF-SMITH-UQ-2024, Chapter 8]. No equation is evaluated.

        """
        return len(self._pairs)

    @property
    def observed(self) -> Vector:
        """Return observations in experiment-then-quantity order.

        References:
            [EQ-CAL-NL-001]. No equation is evaluated.

        """
        return tuple(value for _, _, value in self._pairs)

    def split(self, parameters: Sequence[float]) -> tuple[Vector, Mapping[str, float]]:
        """Split a full vector into ``theta`` and noise scales per QoI.

        Args:
            parameters: ``(theta, log sigma_unknown)``.

        Returns:
            ``theta`` and a mapping from every observed QoI to its noise scale.

        Raises:
            ValueError: If the length is wrong.

        References:
            [EQ-CAL-POST-001]. No new equation is introduced.

        """
        vector = as_vector(parameters, name="parameters")
        if len(vector) != self.dimension:
            raise ValueError(f"parameters must have length {self.dimension}")
        d = self.model.dimension
        sigmas: dict[str, float] = {}
        for item in self.noise:
            if item.standard_deviation is not None:
                sigmas[item.quantity] = item.standard_deviation
        for offset, quantity in enumerate(self._unknown):
            eta = vector[d + offset]
            sigmas[quantity] = math.exp(eta) if eta < 709.0 else math.inf
        return vector[:d], MappingProxyType(sigmas)

    def correlation_lengths(self, parameters: Sequence[float]) -> Mapping[str, float]:
        """Return the error-correlation length of every kernel-correlated QoI.

        Args:
            parameters: Full vector ``(theta, log sigma, log length)``.

        Returns:
            Known lengths and ``exp(log length)`` of uncertain ones, keyed by
            quantity; empty without kernel correlations.

        Raises:
            ValueError: If the length is wrong.

        References:
            [EQ-CAL-CORR-001]. No new equation is introduced.

        """
        vector = as_vector(parameters, name="parameters")
        if len(vector) != self.dimension:
            raise ValueError(f"parameters must have length {self.dimension}")
        return self._lengths(vector[self.model.dimension + len(self._unknown) :])

    def _known_lengths(self) -> dict[str, float]:
        """Return the declared (known) error-correlation lengths.

        References:
            [EQ-CAL-CORR-001]. No equation is evaluated.

        """
        return {
            item.quantity: item.correlation.length
            for item in self.noise
            if item.correlation is not None and item.correlation.length is not None
        }

    def _lengths(self, log_lengths: Sequence[float]) -> Mapping[str, float]:
        """Map log-length coordinates to the lengths of every kernel structure.

        References:
            [EQ-CAL-CORR-001]. No new equation is introduced.

        """
        lengths = self._known_lengths()
        for quantity, eta in zip(self._unknown_lengths, log_lengths, strict=True):
            lengths[quantity] = _admath.exp(eta) if primal(eta) < 709.0 else math.inf
        return MappingProxyType(lengths)

    # -- forward evaluation -------------------------------------------------

    def evaluate(
        self, theta: Sequence[float], conditions: Mapping[str, float]
    ) -> Mapping[str, float]:
        """Evaluate every model QoI at ``theta`` under given conditions.

        Inputs are merged as ``fixed``, then ``conditions``, then ``theta``.

        Args:
            theta: Calibration-parameter vector in model order.
            conditions: Experimental conditions.

        Returns:
            Read-only QoI mapping.

        Equation:
            [EQ-UQ-001], [EQ-CAL-NL-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 8].

        """
        values: dict[str, float] = dict(self.model.fixed)
        values.update(conditions)
        values.update(self.model.inputs.as_mapping(theta))
        self.model.counter.add()
        return evaluate_model(self.model.model, values, self.model.quantities)

    def _call(self, values: dict[str, float]) -> Mapping[str, float]:
        """Call the forward model with a lean output check.

        Parameters were validated once when ``theta`` was materialized, so
        only the output contract of :func:`probstat.uq.evaluate_model` is
        rechecked here (mapping type, presence, finiteness). This keeps MCMC,
        which calls the model thousands of times, affordable in pure Python.

        References:
            [EQ-UQ-001], [REF-SMITH-UQ-2024, Chapter 3].

        """
        self.model.counter.add()
        outputs = self.model.model(MappingProxyType(values))
        if not isinstance(outputs, Mapping):
            raise TypeError("model output must be a mapping")
        return outputs

    def predictions(self, theta: Sequence[float]) -> Vector:
        """Return model predictions aligned with :attr:`observed`.

        Equation:
            [EQ-CAL-NL-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 8].

        """
        vector = as_vector(theta, name="theta")
        if len(vector) != self.model.dimension:
            raise ValueError("theta length must equal the number of parameters")
        named = dict(zip(self.model.inputs.names, vector, strict=True))
        outputs: list[Mapping[str, float]] = []
        for base in self._bases:
            values = dict(base)
            values.update(named)
            outputs.append(self._call(values))
        predicted: list[float] = []
        for index, quantity, _ in self._pairs:
            if quantity not in outputs[index]:
                raise ValueError(f"model output is missing QoI {quantity!r}")
            predicted.append(
                require_finite_real(
                    outputs[index][quantity], name=f"output[{quantity!r}]"
                )
            )
        return tuple(predicted)

    def residuals(self, theta: Sequence[float]) -> Vector:
        """Return observed-minus-predicted residuals.

        Equation:
            [EQ-INV-RESID-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 8].

        """
        predicted = self.predictions(theta)
        return tuple(
            value - center
            for (_, _, value), center in zip(self._pairs, predicted, strict=True)
        )

    # -- densities ------------------------------------------------------------

    def log_likelihood(self, parameters: Sequence[float]) -> float:
        """Return ``log p(y | theta, sigma)`` for a full parameter vector.

        Without correlated errors this is the sum of independent Gaussian or
        Student-t log-densities. Each quantity with an
        :class:`ErrorCorrelation` contributes one joint Gaussian log-density
        ``-0.5 (n log 2 pi + log det Sigma_q + r_q' Sigma_q^-1 r_q)``
        evaluated by Cholesky; a sampled correlation length whose covariance
        is not numerically positive definite gives ``-inf``.

        Args:
            parameters: ``(theta, log sigma_unknown, log length_unknown)``.

        Returns:
            Log-likelihood.

        Equation:
            [EQ-CAL-LIK-001], [EQ-CAL-CORR-002].

        References:
            [REF-GELMAN-ETAL-2013, Section 3.2], [REF-LANGE-LITTLE-TAYLOR-1989],
            [REF-SEBER-WILD-2003, Chapter 6].

        """
        theta, sigmas = self.split(parameters)
        if not self._groups:
            return self._log_likelihood(theta, sigmas)
        return self._log_likelihood(theta, sigmas, self.correlation_lengths(parameters))

    def _log_likelihood(
        self,
        theta: Vector,
        sigmas: Mapping[str, float],
        lengths: Mapping[str, float] | None = None,
    ) -> float:
        """Evaluate the likelihood for split parameters.

        References:
            [EQ-CAL-LIK-001], [EQ-CAL-CORR-002], [REF-GELMAN-ETAL-2013,
            Section 3.2].

        """
        residuals = self.residuals(theta)
        if not self._groups:
            return math.fsum(
                self._noise_by_name[quantity].log_density(residual, sigmas[quantity])
                for (_, quantity, _), residual in zip(
                    self._pairs, residuals, strict=True
                )
            )
        scales = {} if lengths is None else lengths
        terms = [
            self._noise_by_name[self._pairs[i][1]].log_density(
                residuals[i], sigmas[self._pairs[i][1]]
            )
            for i in self._independent
        ]
        for group in self._groups:
            sigma = sigmas[group.quantity]
            if not 0.0 < sigma < math.inf:
                return -math.inf
            terms.append(
                group.log_density(
                    tuple(residuals[i] for i in group.positions),
                    sigma,
                    scales.get(group.quantity),
                )
            )
        return math.fsum(terms)

    def _traced_log_likelihood(
        self,
        theta: Sequence[float],
        sigmas: Mapping[str, float],
        lengths: Mapping[str, float],
    ) -> float:
        """Evaluate the likelihood on reverse-mode variables.

        The forward model is called with variables in place of the
        parameters; a model that converts them to ``float`` (``math.exp``
        and similar) raises :class:`~probstat.numerics.AutodiffUnsupportedError`.

        References:
            [EQ-CAL-LIK-001], [EQ-CAL-CORR-002], [EQ-AD-REVERSE-001],
            [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        named = dict(zip(self.model.inputs.names, theta, strict=True))
        outputs: list[Mapping[str, float]] = []
        for base in self._bases:
            values = dict(base)
            values.update(named)
            self.model.counter.add()
            try:
                result = self.model.model(MappingProxyType(values))
            except TypeError as error:
                raise AutodiffUnsupportedError(
                    "the forward model does not accept reverse-mode variables; "
                    "write it with operators and probstat.numerics.admath "
                    f"instead of math ({type(error).__name__}: {error})"
                ) from error
            if not isinstance(result, Mapping):
                raise TypeError("model output must be a mapping")
            outputs.append(result)
        residuals: list[float] = []
        for index, quantity, value in self._pairs:
            if quantity not in outputs[index]:
                raise ValueError(f"model output is missing QoI {quantity!r}")
            center = real_argument(
                outputs[index][quantity], name=f"output[{quantity!r}]"
            )
            residuals.append(value - center)
        terms: list[float] = []
        for i in self._independent:
            item = self._noise_by_name[self._pairs[i][1]]
            terms.append(
                _traced.observation_log_density(
                    residuals[i],
                    sigmas[item.quantity],
                    item.discrepancy_variance,
                    item.degrees_of_freedom,
                )
            )
        for group in self._groups:
            sigma = sigmas[group.quantity]
            if not 0.0 < primal(sigma) < math.inf:
                return -math.inf
            length = lengths.get(group.quantity)
            if length is not None and not 0.0 < primal(length) < math.inf:
                return -math.inf
            terms.append(
                group.traced_log_density(
                    [residuals[i] for i in group.positions], sigma, length
                )
            )
        return _admath.fsum(terms)

    def _traced_noise(
        self, coordinates: Sequence[float]
    ) -> tuple[float, dict[str, float], Mapping[str, float]]:
        """Return the noise prior, scales and lengths of log-noise coordinates.

        The prior is ``sum log p(e^eta) + eta`` written with ``admath``;
        ``-inf`` where the float version is ``-inf``.

        References:
            [EQ-CAL-POST-001], [EQ-AD-REVERSE-001].

        """
        total: float = 0.0
        for prior, eta in zip(self._sigma_priors, coordinates, strict=True):
            if primal(eta) >= 709.0:
                return -math.inf, {}, {}
            scale = _admath.exp(eta)
            if not 0.0 < primal(scale) < math.inf:
                return -math.inf, {}, {}
            law: Any = prior.distribution
            method = getattr(law, "log_pdf", None)
            if method is None:
                raise AutodiffUnsupportedError(
                    f"the prior of {prior.name} has no log_pdf to differentiate"
                )
            try:
                density = method(scale)
            except ValueError:
                return -math.inf, {}, {}
            if not math.isfinite(primal(density)):
                return -math.inf, {}, {}
            total = total + density + eta
        sigmas: dict[str, float] = {}
        for item in self.noise:
            if item.standard_deviation is not None:
                sigmas[item.quantity] = item.standard_deviation
        count = len(self._unknown)
        for quantity, eta in zip(self._unknown, coordinates[:count], strict=True):
            sigmas[quantity] = _admath.exp(eta)
        return total, sigmas, self._lengths(coordinates[count:])

    def _traced_standard_log_posterior(self, coordinates: Sequence[float]) -> float:
        """Return :meth:`standard_log_posterior`, traceable in reverse mode.

        For floats it is :meth:`standard_log_posterior` itself.

        References:
            [EQ-CAL-STD-001], [EQ-CAL-AD-001], [EQ-AD-REVERSE-001].

        """
        vector = tuple(coordinates)
        if not any_variable(*vector):
            return self.standard_log_posterior(vector)
        if len(vector) != self.dimension:
            raise ValueError(f"coordinates must have length {self.dimension}")
        d = self.model.dimension
        u = vector[:d]
        noise_prior, sigmas, lengths = self._traced_noise(vector[d:])
        if primal(noise_prior) == -math.inf:
            return -math.inf
        theta = _traced.from_standard_normal(self.model.inputs, self._prior_lower, u)
        return (
            self._traced_log_likelihood(theta, sigmas, lengths)
            - 0.5 * _admath.fsum(value * value for value in u)
            + noise_prior
        )

    def _traced_log_posterior(self, parameters: Sequence[float]) -> float:
        """Return :meth:`log_posterior`, traceable in reverse mode.

        For floats it is :meth:`log_posterior` itself.

        References:
            [EQ-CAL-POST-001], [EQ-CAL-AD-001], [EQ-AD-REVERSE-001].

        """
        vector = tuple(parameters)
        if not any_variable(*vector):
            return self.log_posterior(vector)
        if len(vector) != self.dimension:
            raise ValueError(f"parameters must have length {self.dimension}")
        d = self.model.dimension
        prior = _traced.log_prior(self.model.inputs, vector[:d])
        if primal(prior) == -math.inf:
            return -math.inf
        noise_prior, sigmas, lengths = self._traced_noise(vector[d:])
        if primal(noise_prior) == -math.inf:
            return -math.inf
        return (
            prior
            + noise_prior
            + self._traced_log_likelihood(vector[:d], sigmas, lengths)
        )

    def _reverse(
        self, function: Callable[[Sequence[float]], float], point: Sequence[float]
    ) -> tuple[float, Vector]:
        """Return value and reverse-mode gradient with a calibration-level error.

        Raises:
            AutodiffUnsupportedError: If the prior, the noise priors, or the
                forward model cannot be traced.

        References:
            [EQ-AD-REVERSE-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        reason = _traced.supports_prior_map(self.model.inputs)
        if reason is not None:
            raise AutodiffUnsupportedError(
                f"gradient='reverse' is unavailable: {reason}"
            )
        try:
            return reverse_value_and_gradient(cast(Any, function), point)
        except TypeError as error:
            raise AutodiffUnsupportedError(
                "gradient='reverse' cannot trace this calibration problem "
                f"(no finite-difference fallback is used): {error}"
            ) from error

    def standard_log_posterior_and_gradient(
        self, coordinates: Sequence[float]
    ) -> tuple[float, Vector]:
        """Return :meth:`standard_log_posterior` and its reverse-mode gradient.

        One recording of the forward model on reverse-mode variables (one run
        per experiment) and one backward sweep give the whole gradient,
        instead of the ``2 n`` extra posterior evaluations of central
        differences (``n`` = :attr:`dimension`). The prior map
        ``T^-1(u)`` takes its derivative from the inverse-function rule
        ``phi(z_i) / f_i(x_i)`` ([EQ-CAL-AD-001]); the forward model must be
        written with operators and :class:`~probstat.numerics.admath`.

        Args:
            coordinates: ``(u, log sigma, log length)`` of length
                :attr:`dimension`.

        Returns:
            ``(log posterior, gradient)`` in sampling coordinates.

        Raises:
            AutodiffUnsupportedError: If the prior (discrete inputs, a
                non-Gaussian copula), a noise or length prior, or the forward
                model cannot be traced. There is no silent fallback.
            NonDifferentiableError: Where the prior map has an unbounded
                derivative.

        Equation:
            [EQ-CAL-STD-001], [EQ-CAL-AD-001], [EQ-AD-REVERSE-001].

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3], [REF-BAYDIN-ETAL-2018,
            Section 3.2], [REF-LIU-DERKIUREGHIAN-1986].

        """
        point = as_vector(coordinates, name="coordinates")
        if len(point) != self.dimension:
            raise ValueError(f"coordinates must have length {self.dimension}")
        return self._reverse(self._traced_standard_log_posterior, point)

    def log_posterior_and_gradient(
        self, parameters: Sequence[float]
    ) -> tuple[float, Vector]:
        """Return :meth:`log_posterior` and its reverse-mode gradient.

        The prior is differentiated through the marginal ``log_pdf`` methods
        (and, for a Gaussian copula, the normal scores with the derivative
        ``f_i / phi(z_i)``); see :meth:`standard_log_posterior_and_gradient`
        for the forward-model requirements.

        Args:
            parameters: ``(theta, log sigma, log length)``.

        Returns:
            ``(log posterior, gradient)`` in physical coordinates.

        Raises:
            AutodiffUnsupportedError: If the problem cannot be traced.

        Equation:
            [EQ-CAL-POST-001], [EQ-CAL-AD-001], [EQ-AD-REVERSE-001].

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3], [REF-NELSEN-2006,
            Section 2.3].

        """
        point = as_vector(parameters, name="parameters")
        if len(point) != self.dimension:
            raise ValueError(f"parameters must have length {self.dimension}")
        return self._reverse(self._traced_log_posterior, point)

    def _log_noise_prior(self, log_sigmas: Sequence[float]) -> float:
        """Return the log-scale prior of unknown noise scales with Jacobian.

        References:
            [EQ-CAL-POST-001], [REF-CASELLA-BERGER-2002, Section 4.3].

        """
        total = 0.0
        for prior, eta in zip(self._sigma_priors, log_sigmas, strict=True):
            if eta >= 709.0:
                # exp would overflow: such a scale has zero prior density for
                # every proper prior, so reject instead of aborting a chain.
                return -math.inf
            sigma = math.exp(eta)
            if sigma <= 0.0 or not math.isfinite(sigma):
                return -math.inf
            total += prior.log_pdf(sigma) + eta
        return total

    def log_prior(self, parameters: Sequence[float]) -> float:
        """Return the prior log-density of ``(theta, log sigma)``.

        ``log pi(theta)`` comes from ``JointInput.log_pdf``; each unknown
        ``eta = log sigma`` contributes ``log p_sigma(e^eta) + eta``.

        Equation:
            [EQ-CAL-POST-001], [EQ-MOD-COPULA-001].

        References:
            [REF-CASELLA-BERGER-2002, Section 4.3], [REF-NELSEN-2006, Section 2.3].

        """
        vector = as_vector(parameters, name="parameters")
        if len(vector) != self.dimension:
            raise ValueError(f"parameters must have length {self.dimension}")
        d = self.model.dimension
        prior = self.model.inputs.log_pdf(vector[:d])
        if not math.isfinite(prior):
            return -math.inf
        return prior + self._log_noise_prior(vector[d:])

    def log_posterior(self, parameters: Sequence[float]) -> float:
        """Return the unnormalized log posterior of ``(theta, log sigma)``.

        The forward model is not evaluated when the prior density is zero.

        Equation:
            [EQ-CAL-POST-001], [EQ-BAYES-RULE-001].

        References:
            [REF-GELMAN-ETAL-2013, Section 1.3].

        """
        prior = self.log_prior(parameters)
        if prior == -math.inf:
            return -math.inf
        return prior + self.log_likelihood(parameters)

    def standard_log_posterior(self, coordinates: Sequence[float]) -> float:
        """Return the unnormalized log posterior in sampling coordinates.

        For ``v = (u, eta)`` with ``theta = T^-1(u)`` (Nataf map of the prior)
        and ``sigma = e^eta``:
        ``log p(v | y) = log L(T^-1(u), e^eta) - |u|^2/2
        + sum_q [log p_sigma(e^eta_q) + eta_q] + const``.

        Args:
            coordinates: ``(u, log sigma_unknown)``.

        Returns:
            Log posterior up to an additive constant.

        Equation:
            [EQ-CAL-STD-001], [EQ-MOD-NATAF-001].

        References:
            [REF-LIU-DERKIUREGHIAN-1986], [REF-CASELLA-BERGER-2002, Section 4.3].

        """
        vector = as_vector(coordinates, name="coordinates")
        if len(vector) != self.dimension:
            raise ValueError(f"coordinates must have length {self.dimension}")
        d = self.model.dimension
        u = vector[:d]
        noise_prior = self._log_noise_prior(vector[d:])
        if noise_prior == -math.inf:
            return -math.inf
        theta = self.model.inputs.from_standard_normal(u)
        _, sigmas = self.split(theta + vector[d:])
        if self._groups:
            likelihood = self._log_likelihood(
                theta, sigmas, self._lengths(vector[d + len(self._unknown) :])
            )
        else:
            likelihood = self._log_likelihood(theta, sigmas)
        return likelihood - 0.5 * math.fsum(value * value for value in u) + noise_prior

    def to_physical(self, coordinates: Sequence[float]) -> Vector:
        """Map sampling coordinates ``(u, log sigma)`` to ``(theta, sigma)``.

        Equation:
            [EQ-MOD-NATAF-001].

        References:
            [REF-LIU-DERKIUREGHIAN-1986].

        """
        vector = as_vector(coordinates, name="coordinates")
        d = self.model.dimension
        theta = self.model.inputs.from_standard_normal(vector[:d])
        return theta + tuple(
            math.exp(eta) if eta < 709.0 else math.inf for eta in vector[d:]
        )

    def _noise_from_whitened(self, whitened: Sequence[float]) -> Vector:
        """Map standard-normal noise coordinates ``w`` to noise scales.

        References:
            [EQ-MOD-NATAF-001], [REF-LIU-DERKIUREGHIAN-1986].

        """
        if self._noise_inputs is None:
            return ()
        return self._noise_inputs.from_standard_normal(whitened)

    def standard_log_likelihood(self, coordinates: Sequence[float]) -> float:
        """Return the log-likelihood in fully standardized coordinates.

        For ``v = (u, w)`` with ``theta = T^-1(u)`` (Nataf map of the prior)
        and ``sigma_q = F_q^-1(Phi(w_q))`` (quantile map of each unknown
        noise scale's prior), the prior of ``v`` is exactly ``N(0, I)`` and
        the posterior is ``L(T^-1(u), sigma(w)) N(v; 0, I) / p(y)``. This is
        the likelihood used by ``method="pcn"`` and ``"smc"`` and by
        :meth:`log_evidence`. It carries every normalizing constant of the
        Gaussian or Student-t noise density. Without unknown noise scales
        ``v = u``.

        Args:
            coordinates: ``(u, w)`` of length :attr:`dimension`.

        Returns:
            ``log p(y | theta, sigma)``; ``-inf`` if a mapped noise scale is
            zero or infinite.

        Raises:
            ValueError: If the length is wrong.

        Equation:
            [EQ-CAL-LIK-001], [EQ-MOD-NATAF-001], [EQ-CAL-STD-002].

        References:
            [REF-LIU-DERKIUREGHIAN-1986], [REF-COTTER-ROBERTS-STUART-WHITE-2013].

        """
        vector = as_vector(coordinates, name="coordinates")
        if len(vector) != self.dimension:
            raise ValueError(f"coordinates must have length {self.dimension}")
        d = self.model.dimension
        theta = self.model.inputs.from_standard_normal(vector[:d])
        scales = self._noise_from_whitened(vector[d:])
        if any(not 0.0 < value < math.inf for value in scales):
            return -math.inf
        sigmas: dict[str, float] = {}
        for item in self.noise:
            if item.standard_deviation is not None:
                sigmas[item.quantity] = item.standard_deviation
        count = len(self._unknown)
        for quantity, value in zip(self._unknown, scales[:count], strict=True):
            sigmas[quantity] = value
        if not self._groups:
            return self._log_likelihood(theta, MappingProxyType(sigmas))
        lengths = self._known_lengths()
        lengths.update(zip(self._unknown_lengths, scales[count:], strict=True))
        return self._log_likelihood(theta, MappingProxyType(sigmas), lengths)

    def _whitened_names(self) -> tuple[str, ...]:
        """Return names of the fully standardized coordinates ``(u, w)``.

        References:
            [EQ-CAL-STD-002]. No equation is evaluated.

        """
        return (
            tuple(f"u[{name}]" for name in self.parameter_names)
            + tuple(f"w[{q}]" for q in self._unknown)
            + tuple(f"w[length[{q}]]" for q in self._unknown_lengths)
        )

    def _whitened_to_sampling(self, coordinates: Sequence[float]) -> Vector:
        """Map ``(u, w)`` to the sampling coordinates ``(u, log sigma)``.

        References:
            [EQ-CAL-STD-002], [REF-LIU-DERKIUREGHIAN-1986].

        """
        vector = as_vector(coordinates, name="coordinates")
        d = self.model.dimension
        return vector[:d] + tuple(
            math.log(value) for value in self._noise_from_whitened(vector[d:])
        )

    def _sampling_to_whitened(self, coordinates: Sequence[float]) -> Vector:
        """Map ``(u, log sigma)`` to fully standardized ``(u, w)``.

        References:
            [EQ-CAL-STD-002], [REF-LIU-DERKIUREGHIAN-1986].

        """
        vector = as_vector(coordinates, name="coordinates")
        d = self.model.dimension
        if self._noise_inputs is None:
            return vector
        return vector[:d] + self._noise_inputs.to_standard_normal(
            tuple(math.exp(eta) for eta in vector[d:])
        )

    def _whitened_to_physical(self, coordinates: Sequence[float]) -> Vector:
        """Map ``(u, w)`` to ``(theta, sigma)``.

        References:
            [EQ-CAL-STD-002], [EQ-MOD-NATAF-001].

        """
        vector = as_vector(coordinates, name="coordinates")
        d = self.model.dimension
        return self.model.inputs.from_standard_normal(
            vector[:d]
        ) + self._noise_from_whitened(vector[d:])

    def _run_smc(
        self,
        *,
        particles: int,
        mutation_steps: int,
        ess_fraction: float,
        stream: RandomStream,
    ) -> SMCResult:
        """Run tempered SMC on :meth:`standard_log_likelihood`.

        The prior of ``(u, w)`` is ``N(0, I)``; particles are mutated by
        random-walk Metropolis with the particle covariance, which adapts to
        a concentrated, anisotropic posterior better than an isotropic pCN
        move.

        References:
            [EQ-SMC-TEMPER-001], [EQ-SMC-EVID-001],
            [REF-DELMORAL-DOUCET-JASRA-2006].

        """
        size = self.dimension

        def prior_draw(generator: random.Random) -> Vector:
            """Draw one ``N(0, I)`` prior point.

            References:
                [EQ-CAL-STD-002]. No new equation is introduced.

            """
            return tuple(generator.gauss(0.0, 1.0) for _ in range(size))

        def standard_normal_log_prior(point: Vector) -> float:
            """Return ``-|v|^2 / 2``, the ``N(0, I)`` log-density up to a constant.

            References:
                [EQ-CAL-STD-002]. No new equation is introduced.

            """
            return -0.5 * math.fsum(value * value for value in point)

        names = self._whitened_names()
        return sequential_monte_carlo(
            prior_draw,
            standard_normal_log_prior,
            self.standard_log_likelihood,
            particles=particles,
            mutation="rwm",
            mutation_steps=mutation_steps,
            ess_fraction=ess_fraction,
            seed=stream,
            names=names,
        )

    def log_evidence(
        self,
        *,
        particles: int = 1000,
        mutation_steps: int = 10,
        ess_fraction: float = 0.5,
        seed: int | RandomStream | None = None,
    ) -> SMCResult:
        """Estimate the marginal likelihood ``log p(y)`` by tempered SMC.

        ``p(y) = integral p(y | theta, sigma) p(theta) p(sigma) d theta
        d sigma`` is computed in the fully standardized coordinates of
        :meth:`standard_log_likelihood`, where the prior is ``N(0, I)`` and
        the change of variables leaves the integral unchanged. The noise
        densities include all normalizing constants, so the estimate is the
        density of the observations themselves; ratios between problems with
        the same observations are Bayes factors.

        Args:
            particles: Number of SMC particles (at least two).
            mutation_steps: Random-walk Metropolis moves per particle and
                stage.
            ess_fraction: ESS target fraction of the adaptive tempering.
            seed: Integer seed or stream.

        Returns:
            The SMC result; ``log_evidence`` is ``log p(y)`` and
            ``log_evidence_standard_error`` its Lee-Whiteley standard error
            (see :func:`probstat.inference.sequential_monte_carlo` for its
            limitations).

        Raises:
            ValueError: For invalid SMC options.

        Equation:
            [EQ-SMC-EVID-001], [EQ-SMC-EVID-002], [EQ-CAL-STD-002].

        References:
            [REF-DELMORAL-DOUCET-JASRA-2006], [REF-LEE-WHITELEY-2018],
            [REF-GELMAN-ETAL-2013, Chapter 7].

        """
        return self._run_smc(
            particles=particles,
            mutation_steps=mutation_steps,
            ess_fraction=ess_fraction,
            stream=resolve_stream(seed),
        )

    def _default_log_sigmas(self) -> Vector:
        """Return log prior medians of unknown noise scales.

        References:
            [REF-CASELLA-BERGER-2002, Section 4.3]. No equation is evaluated.

        """
        return tuple(
            math.log(float(prior.distribution.quantile(0.5)))
            for prior in self._sigma_priors
        )

    # -- estimation -----------------------------------------------------------

    def _whitener(
        self,
        scales: Mapping[str, float],
        lengths: Mapping[str, float] | None = None,
    ) -> Callable[[Vector], Vector]:
        """Return the linear map ``v -> W v`` with ``W' W = Sigma^-1``.

        Independent observations are divided by their total scale
        ``sqrt(sigma^2 + sigma_delta^2)``; the observations of a correlated
        quantity are replaced by ``L^-1 v_q`` with ``Sigma_q = L L'``
        (placed at the positions of that quantity), so the sum of squares of
        whitened residuals is the generalized residual sum of squares.

        Raises:
            ValueError: If a correlated covariance is not positive definite.

        Equation:
            [EQ-CAL-CORR-003], [EQ-CAL-WLS-001].

        References:
            [REF-SEBER-WILD-2003, Chapters 2 and 6].

        """
        weights = tuple(
            math.sqrt(scales[q] ** 2 + self._noise_by_name[q].discrepancy_variance)
            for _, q, _ in self._pairs
        )
        if not self._groups:

            def independent(values: Vector) -> Vector:
                """Divide each entry by its total noise scale.

                References:
                    [EQ-CAL-WLS-001]. No new equation is introduced.

                """
                return tuple(v / w for v, w in zip(values, weights, strict=True))

            return independent
        available = {} if lengths is None else lengths
        factors = tuple(
            group.factor(scales[group.quantity], available.get(group.quantity))
            for group in self._groups
        )

        def correlated(values: Vector) -> Vector:
            """Whiten independent entries and correlated groups.

            References:
                [EQ-CAL-CORR-003]. No new equation is introduced.

            """
            output = [v / w for v, w in zip(values, weights, strict=True)]
            for group, (lower, divisor) in zip(self._groups, factors, strict=True):
                solved = solve_lower(lower, tuple(values[i] for i in group.positions))
                for position, value in zip(group.positions, solved, strict=True):
                    output[position] = value / divisor
            return tuple(output)

        return correlated

    def whitened_residuals(self, parameters: Sequence[float]) -> Vector:
        """Return residuals whitened by the full noise covariance.

        For independent noise ``r_i / s_i``; for a correlated quantity
        ``L_q^-1 r_q`` with ``Sigma_q = L_q L_q'`` ([EQ-CAL-CORR-003]). Their
        sum of squares is ``r' Sigma^-1 r``, the Gaussian quadratic form of
        the log-likelihood.

        Args:
            parameters: Full vector ``(theta, log sigma, log length)``.

        Returns:
            Whitened residuals aligned with :attr:`observed` (for a
            correlated quantity the entries are the components of
            ``L_q^-1 r_q``, not per-observation values).

        Raises:
            ValueError: If the length is wrong or a covariance is not
                positive definite.

        Equation:
            [EQ-CAL-CORR-003].

        References:
            [REF-SEBER-WILD-2003, Chapters 2 and 6], [REF-GOLUB-VANLOAN-2013,
            Section 4.2].

        """
        theta, sigmas = self.split(parameters)
        lengths = self.correlation_lengths(parameters)
        return self._whitener(sigmas, lengths)(self.residuals(theta))

    def _weighted_jacobian(
        self,
        theta: Vector,
        scales: Mapping[str, float],
        lengths: Mapping[str, float] | None = None,
    ) -> tuple[Matrix, int]:
        """Return the Jacobian of whitened predictions.

        References:
            [EQ-CAL-FISHER-001], [EQ-CAL-CORR-003], [EQ-OPT-FD-001].

        """
        whiten = self._whitener(scales, lengths)

        def weighted(point: Vector) -> Vector:
            """Return weighted predictions.

            References:
                [EQ-CAL-WLS-001]. No new equation is introduced.

            """
            return whiten(self.predictions(point))

        return central_jacobian(weighted, theta)

    def identifiability(
        self,
        theta: Sequence[float] | None = None,
        *,
        prior_scaled: bool = True,
        condition_warning: float = 1.0e8,
    ) -> FisherIdentifiability:
        """Diagnose local identifiability at ``theta`` (default: prior means).

        Unknown noise scales (and uncertain error-correlation lengths) are
        replaced by their prior medians; correlated errors enter through the
        whitened Jacobian ``L^-1 J`` ([EQ-CAL-CORR-003]).

        Args:
            theta: Linearization point.
            prior_scaled: Scale by prior standard deviations (dimensionless,
                data-versus-prior information); otherwise raw units.
            condition_warning: Warning threshold for the condition number.

        Returns:
            Fisher eigen-analysis.

        Equation:
            [EQ-CAL-FISHER-001].

        References:
            [REF-SEBER-WILD-2003, Section 3.4], [REF-TARANTOLA-2005, Chapter 3].

        """
        point = (
            self.model.nominal() if theta is None else as_vector(theta, name="theta")
        )
        full = point + self._default_log_sigmas()
        _, sigmas = self.split(full)
        if self._groups:
            jacobian, _ = self._weighted_jacobian(
                point, sigmas, self.correlation_lengths(full)
            )
        else:
            jacobian, _ = self._weighted_jacobian(point, sigmas)
        scales = self.model.inputs.standard_deviations() if prior_scaled else None
        return fisher_identifiability(
            jacobian,
            self.parameter_names,
            scales=scales,
            condition_warning=condition_warning,
        )

    def fit_least_squares(
        self,
        initial: Sequence[float] | None = None,
        *,
        tolerance: float = 1e-10,
        max_iterations: int = 200,
        max_reweights: int = 10,
    ) -> LeastSquaresCalibration:
        """Fit ``theta`` by weighted nonlinear least squares (prior ignored).

        Minimizes ``0.5 sum_i (r_i / s_i)^2`` by Levenberg-Marquardt. Known
        noise gives fixed weights ``s_i^2 = sigma_q^2 + sigma_delta_q^2``.
        Unknown noise scales are estimated by iterative reweighting with
        ``s_q^2 = (RSS_q / n_q) n / (n - d)``; for one unknown QoI this is the
        usual ``RSS / (n - d)``. The covariance is ``(J_w'J_w)^-1``. Student-t
        noise is fitted with Gaussian weights here; use the posterior methods
        for robust inference. This is the frequentist estimate: the prior
        enters only through the starting point.

        Correlated errors are whitened: the objective is
        ``0.5 r' Sigma^-1 r`` (generalized least squares) through the
        Cholesky factor of each correlated quantity ([EQ-CAL-CORR-003]), and
        an unknown scale of such a quantity is re-estimated from the
        generalized residual sum of squares ``r_q' M_q^-1 r_q``.

        Args:
            initial: Starting ``theta``; defaults to the prior means.
            tolerance: Levenberg-Marquardt tolerance.
            max_iterations: Levenberg-Marquardt iteration limit.
            max_reweights: Reweighting passes for unknown noise.

        Returns:
            Estimate, covariance, noise scales, residuals, and identifiability.

        Raises:
            ValueError: If there are no more observations than parameters when
                noise must be estimated, or ``J'J`` is singular; if an
                error-correlation length is uncertain (least squares does not
                estimate it); or if an unknown scale of a correlated quantity
                is combined with a discrepancy variance (no closed-form
                scale update).

        Equation:
            [EQ-CAL-WLS-001], [EQ-CAL-CORR-003], [EQ-OPT-LM-001],
            [EQ-OPT-LM-002].

        References:
            [REF-SEBER-WILD-2003, Chapter 2], [REF-MORE-1978].

        """
        start_count = self.model.evaluations
        theta = self.model.nominal() if initial is None else as_vector(initial)
        if len(theta) != self.model.dimension:
            raise ValueError("initial must match the number of parameters")
        n, d = self.observation_count, self.model.dimension
        if self._unknown and n <= d:
            raise ValueError("estimating noise needs more observations than parameters")
        groups = {group.quantity: group for group in self._groups}
        if self._unknown_lengths:
            raise ValueError(
                "fit_least_squares does not estimate error-correlation lengths; "
                "declare a known length or use laplace_posterior or "
                "sample_posterior"
            )
        for quantity in self._unknown:
            if quantity in groups and groups[quantity].discrepancy_variance > 0.0:
                raise ValueError(
                    f"the unknown noise scale of the correlated quantity "
                    f"{quantity!r} cannot be re-estimated in closed form with a "
                    "discrepancy variance; use laplace_posterior or "
                    "sample_posterior"
                )
        lengths = self._known_lengths()
        scales: dict[str, float] = {
            item.quantity: (
                item.standard_deviation if item.standard_deviation is not None else 1.0
            )
            for item in self.noise
        }
        result: LeastSquaresResult | None = None
        passes = 1 + (max_reweights if self._unknown else 0)
        for _ in range(passes):
            weights = tuple(
                math.sqrt(scales[q] ** 2 + self._noise_by_name[q].discrepancy_variance)
                for _, q, _ in self._pairs
            )

            def weighted_residuals(
                point: Vector, weights: tuple[float, ...] = weights
            ) -> Vector:
                """Return residuals divided by their total noise scale.

                References:
                    [EQ-CAL-WLS-001]. No new equation is introduced.

                """
                return tuple(
                    r / w for r, w in zip(self.residuals(point), weights, strict=True)
                )

            objective: Callable[[Vector], Vector] = weighted_residuals
            if self._groups:
                whiten = self._whitener(scales, lengths)

                def whitened_residuals(
                    point: Vector, whiten: Callable[[Vector], Vector] = whiten
                ) -> Vector:
                    """Return residuals whitened by the noise covariance.

                    References:
                        [EQ-CAL-CORR-003]. No new equation is introduced.

                    """
                    return whiten(self.residuals(point))

                objective = whitened_residuals
            result = least_squares(
                objective,
                theta,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
            theta = result.x
            if not self._unknown:
                break
            raw = self.residuals(theta)
            changed = 0.0
            for quantity in self._unknown:
                values = [
                    r
                    for (_, q, _), r in zip(self._pairs, raw, strict=True)
                    if q == quantity
                ]
                if quantity in groups:
                    shape_lower = groups[quantity].shape_lower
                    assert shape_lower is not None
                    values = list(solve_lower(shape_lower, tuple(values)))
                total = math.fsum(r * r for r in values) / len(values) * n / (n - d)
                sigma2 = max(
                    total - self._noise_by_name[quantity].discrepancy_variance,
                    1e-300,
                )
                new = math.sqrt(sigma2)
                changed = max(changed, abs(new - scales[quantity]) / new)
                scales[quantity] = new
            if changed <= 1e-8:
                break
        assert result is not None
        covariance = gauss_newton_covariance(result, noise_variance=1.0)
        prior_scales = self.model.inputs.standard_deviations()
        identifiability = fisher_identifiability(
            result.jacobian, self.parameter_names, scales=prior_scales
        )
        residuals = self.residuals(theta)
        return LeastSquaresCalibration(
            self.parameter_names,
            theta,
            covariance,
            MappingProxyType(scales),
            residuals,
            result.cost,
            identifiability,
            result.converged,
            self.model.evaluations - start_count,
            result,
        )

    def laplace_posterior(
        self, initial: Sequence[float] | None = None, *, gradient: str = "central"
    ) -> LaplaceApproximation:
        """Return the Laplace approximation of the posterior of ``(theta, log sigma)``.

        The MAP point maximizes :meth:`log_posterior` by BFGS in the physical
        ``theta`` (and log-noise) coordinates; the covariance is the inverse
        finite-difference Hessian there. Names follow :attr:`vector_names`.
        The approximation is exact for linear-Gaussian problems with Gaussian
        priors and known noise (independent or correlated).

        With ``gradient="reverse"`` BFGS receives value and gradient of
        :meth:`log_posterior` from one reverse-mode recording
        (:meth:`log_posterior_and_gradient`, one forward-model run per
        experiment) instead of ``2 n`` extra evaluations per gradient, and
        the Hessian is forward over reverse when the prior is independent;
        with a Gaussian-copula prior, whose normal scores are differentiated
        to first order only, it falls back to central differences of the
        log posterior (recorded in ``fallback_reason``). The default
        ``"central"`` reproduces earlier results exactly.

        Args:
            initial: Full starting vector; defaults to prior means and log
                prior medians of unknown noise scales.
            gradient: ``"central"`` (default) or ``"reverse"``.

        Returns:
            Laplace approximation (mode = MAP estimate).

        Raises:
            ValueError: If the start has zero density, the Hessian is not
                positive definite, or ``gradient`` is unknown.
            AutodiffUnsupportedError: With ``gradient="reverse"`` for a
                problem that cannot be traced (no silent fallback).

        Equation:
            [EQ-LAPLACE-001], [EQ-CAL-POST-001], [EQ-AD-REVERSE-001].

        References:
            [REF-TIERNEY-KADANE-1986], [REF-GELMAN-ETAL-2013, Section 4.1],
            [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        _check_gradient_option(gradient)
        start = (
            self.model.nominal() + self._default_log_sigmas()
            if initial is None
            else as_vector(initial, name="initial")
        )
        if gradient == "central":
            return laplace_approximation(self.log_posterior, start)
        self.log_posterior_and_gradient(start)
        return laplace_approximation(
            self._traced_log_posterior, start, gradient="reverse"
        )

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
        particles: int = 1000,
        mutation_steps: int = 10,
        gradient: str = "central",
    ) -> CalibrationPosterior:
        """Sample the posterior by MCMC or SMC in standard-normal space.

        Steps: (1) Laplace approximation of :meth:`standard_log_posterior`
        from ``u = 0`` and log prior-median noise; (2) chain starts drawn from
        that Gaussian with standard deviations inflated by ``dispersion``
        (over-dispersed starts make ``R-hat`` meaningful); (3) adaptive
        Metropolis with initial proposal ``2.38^2/d`` times the Laplace
        covariance, adaptation only during warm-up; (4) draws mapped back to
        ``(theta, sigma)``. If the Laplace Hessian is not positive definite,
        chains start near ``u = 0`` with a small isotropic proposal.

        With ``method="hmc"`` or ``"nuts"`` step (3) is replaced by static
        Hamiltonian Monte Carlo or the No-U-Turn Sampler with a diagonal
        metric initialized from the Laplace variances and adapted in warm-up
        windows. Gradients of :meth:`standard_log_posterior` are central
        differences, so every leapfrog step costs ``2 n + 1`` posterior
        evaluations (``n`` = :attr:`dimension`), i.e. ``2 n + 1`` forward-model
        runs per experiment; divergences are reported in ``sampler``. The
        default reproduces the adaptive-Metropolis results of earlier
        versions exactly.

        ``gradient="reverse"`` (HMC and NUTS only) differentiates
        :meth:`standard_log_posterior` by reverse-mode automatic
        differentiation instead: one recording of the forward model (one run
        per experiment) gives value and gradient of a leapfrog step, and
        BFGS of step (1) uses the same gradient. It needs a forward model
        written with operators and :class:`~probstat.numerics.admath` and a
        continuous prior with independence or a Gaussian copula; otherwise
        :class:`~probstat.numerics.AutodiffUnsupportedError` is raised before
        sampling (no silent fallback). The default ``"central"`` reproduces
        earlier results exactly.

        With ``method="pcn"`` step (3) is preconditioned Crank-Nicolson on
        :meth:`standard_log_likelihood` in the fully standardized coordinates
        ``(u, w)`` (module docstring), started from the same over-dispersed
        points mapped to ``w``; ``beta`` is adapted in warm-up. pCN pays off
        when the data inform few directions of a high-dimensional prior; for
        a posterior much narrower than the prior in every direction its
        isotropic proposal mixes slower than adaptive Metropolis.

        With ``method="smc"`` steps (1) to (3) are replaced by tempered
        sequential Monte Carlo from the ``N(0, I)`` prior of ``(u, w)`` with
        ``particles`` particles and ``mutation_steps`` random-walk moves per
        stage (``draws``, ``warmup``, ``chains``, ``thin`` and ``dispersion``
        are not used). The final particles form one pseudo chain of
        ``mcmc`` and ``standard_space``; the full result, including the log
        evidence, is in ``smc``.

        Args:
            draws: Kept draws per chain.
            warmup: Adaptation and burn-in iterations per chain.
            chains: Number of independent chains.
            thin: Thinning interval.
            dispersion: Start-point spread relative to the Laplace scale.
            seed: Integer seed or stream.
            method: ``"adaptive-metropolis"`` (default), ``"hmc"``,
                ``"nuts"``, ``"pcn"``, or ``"smc"``.
            particles: SMC particles (``method="smc"`` only).
            mutation_steps: SMC moves per particle and stage
                (``method="smc"`` only).
            gradient: ``"central"`` (default) or ``"reverse"`` (HMC and NUTS
                only).

        Returns:
            Posterior with physical and sampling-space chains.

        Raises:
            ValueError: For invalid run arguments, an unknown ``method`` or
                ``gradient``, or ``gradient="reverse"`` with a method other
                than HMC or NUTS.
            AutodiffUnsupportedError: With ``gradient="reverse"`` for a
                problem that cannot be traced.

        Equation:
            [EQ-CAL-STD-001], [EQ-CAL-STD-002], [EQ-MCMC-AM-001],
            [EQ-MCMC-MH-001], [EQ-HMC-LEAPFROG-001], [EQ-NUTS-SLICE-001],
            [EQ-MCMC-PCN-001], [EQ-SMC-TEMPER-001], [EQ-AD-REVERSE-001].

        References:
            [REF-HAARIO-SAKSMAN-TAMMINEN-2001], [REF-GELMAN-RUBIN-1992],
            [REF-LIU-DERKIUREGHIAN-1986], [REF-HOFFMAN-GELMAN-2014],
            [REF-NEAL-2011], [REF-COTTER-ROBERTS-STUART-WHITE-2013],
            [REF-DELMORAL-DOUCET-JASRA-2006], [REF-GRIEWANK-WALTHER-2008,
            Chapter 3].

        """
        if method not in ("adaptive-metropolis", "hmc", "nuts", "pcn", "smc"):
            raise ValueError(
                "method must be 'adaptive-metropolis', 'hmc', 'nuts', 'pcn', or 'smc'"
            )
        _check_gradient_option(gradient)
        reverse = gradient == "reverse"
        if reverse and method not in ("hmc", "nuts"):
            raise ValueError("gradient='reverse' applies to method='hmc' or 'nuts'")
        stream = resolve_stream(seed)
        sampling_names = (
            tuple(f"u[{name}]" for name in self.parameter_names)
            + tuple(f"log_sigma[{q}]" for q in self._unknown)
            + tuple(f"log_length[{q}]" for q in self._unknown_lengths)
        )
        physical_names = self.parameter_names + self.noise_names
        if method == "smc":
            return self._smc_posterior(
                particles=particles,
                mutation_steps=mutation_steps,
                stream=stream,
                sampling_names=sampling_names,
            )
        spread = require_finite_real(dispersion, name="dispersion")
        if spread <= 0.0:
            raise ValueError("dispersion must be positive")
        chain_count = require_non_negative_integer(chains, name="chains")
        if chain_count == 0:
            raise ValueError("chains must be positive")
        size = self.dimension
        origin = tuple(0.0 for _ in range(self.model.dimension))
        origin = origin + self._default_log_sigmas()
        laplace: LaplaceApproximation | None
        if reverse:
            self.standard_log_posterior_and_gradient(origin)
        try:
            if reverse:
                laplace = laplace_approximation(
                    self._traced_standard_log_posterior, origin, gradient="reverse"
                )
            else:
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
        sampler: HamiltonianResult | None = None
        if method == "pcn":
            whitened = preconditioned_crank_nicolson(
                self.standard_log_likelihood,
                tuple(self._sampling_to_whitened(start) for start in starts),
                draws=draws,
                warmup=warmup,
                chains=chains,
                thin=thin,
                seed=stream.child("chains"),
                names=self._whitened_names(),
            )
            return CalibrationPosterior(
                self.parameter_names,
                self.noise_names,
                whitened.transformed(self._whitened_to_physical, physical_names),
                whitened.transformed(self._whitened_to_sampling, sampling_names),
                laplace,
            )
        if method == "adaptive-metropolis":
            proposal = scale_matrix(covariance, 2.38**2 / size)
            sampled = adaptive_metropolis(
                self.standard_log_posterior,
                tuple(starts),
                draws=draws,
                warmup=warmup,
                chains=chains,
                thin=thin,
                initial_covariance=proposal,
                seed=stream.child("chains"),
                names=sampling_names,
            )
        elif reverse:
            run = no_u_turn_sampler if method == "nuts" else hamiltonian_monte_carlo
            sampler = run(
                self._traced_standard_log_posterior,
                tuple(starts),
                gradient="reverse",
                draws=draws,
                warmup=warmup,
                chains=chains,
                thin=thin,
                metric="diag",
                initial_inverse_metric=covariance,
                seed=stream.child("chains"),
                names=sampling_names,
            )
            sampled = sampler.mcmc
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
                names=sampling_names,
            )
            sampled = sampler.mcmc
        physical = sampled.transformed(self.to_physical, physical_names)
        return CalibrationPosterior(
            self.parameter_names,
            self.noise_names,
            physical,
            sampled,
            laplace,
            sampler,
        )

    def _smc_posterior(
        self,
        *,
        particles: int,
        mutation_steps: int,
        stream: RandomStream,
        sampling_names: tuple[str, ...],
    ) -> CalibrationPosterior:
        """Run SMC and package its particles as a one-chain posterior.

        The ``log_density`` trace holds ``log L - |v|^2 / 2`` and the
        ``proposal_covariance`` field the particle covariance (no single
        proposal produced the particles).

        References:
            [EQ-SMC-TEMPER-001], [REF-DELMORAL-DOUCET-JASRA-2006].

        """
        result = self._run_smc(
            particles=particles,
            mutation_steps=mutation_steps,
            ess_fraction=0.5,
            stream=stream.child("smc"),
        )
        trace = tuple(
            value - 0.5 * math.fsum(x * x for x in point)
            for value, point in zip(
                result.log_likelihoods, result.particles, strict=True
            )
        )
        chain = MarkovChain(
            result.particles,
            trace,
            result.acceptance_rates[-1],
            math.nan,
            result.covariance(),
            result.evaluations,
            stream.seed,
        )
        pseudo = MCMCResult(
            (chain,),
            result.names,
            "sequential Monte Carlo (particles, not a chain)",
            0,
            1,
        )
        return CalibrationPosterior(
            self.parameter_names,
            self.noise_names,
            pseudo.transformed(
                self._whitened_to_physical, self.parameter_names + self.noise_names
            ),
            pseudo.transformed(self._whitened_to_sampling, sampling_names),
            None,
            None,
            result,
        )
