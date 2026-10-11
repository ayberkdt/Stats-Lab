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
Metropolis (default), static HMC, or NUTS in the Nataf standard-normal space
``u = T(theta)`` of the prior.
Because ``T`` pushes the prior forward to ``N(0, I)``, the posterior density in
``u`` is ``L(T^-1(u)) phi(u)`` exactly; the prior Jacobian is absorbed by the
transformation and no determinant needs to be computed. Bounded or skewed
priors become unbounded and roughly isotropic, which conditions the sampler.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
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
    matmul,
    matvec,
    scale_matrix,
    symmetric_eigen,
    transpose,
)
from probstat.modeling import ContinuousMarginal, ProbabilisticModel, RandomInput
from probstat.optimization import (
    LeastSquaresResult,
    central_jacobian,
    gauss_newton_covariance,
    least_squares,
)
from probstat.uq import evaluate_model

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

    Args:
        quantity: Observed QoI name.
        standard_deviation: Known positive noise scale.
        standard_deviation_prior: Continuous prior law of an unknown positive
            noise scale.
        degrees_of_freedom: Optional positive Student-t degrees of freedom.
        discrepancy_variance: Non-negative additive discrepancy variance.

    Raises:
        ValueError: For an invalid combination or non-positive values.

    References:
        [REF-KENNEDY-OHAGAN-2001], [REF-LANGE-LITTLE-TAYLOR-1989].

    """

    quantity: str
    standard_deviation: float | None = None
    standard_deviation_prior: ContinuousMarginal | None = None
    degrees_of_freedom: float | None = None
    discrepancy_variance: float = 0.0

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
            Project API policy. No equation is evaluated.

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
            in sampling coordinates; ``None`` for adaptive Metropolis.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 11], [REF-KENNEDY-OHAGAN-2001],
        [REF-HOFFMAN-GELMAN-2014].

    """

    parameter_names: tuple[str, ...]
    noise_names: tuple[str, ...]
    mcmc: MCMCResult
    standard_space: MCMCResult
    laplace: LaplaceApproximation | None
    sampler: HamiltonianResult | None = None

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
        object.__setattr__(self, "experiments", experiments)
        object.__setattr__(self, "noise", noise)
        object.__setattr__(self, "_pairs", tuple(pairs))
        object.__setattr__(self, "_noise_by_name", MappingProxyType(by_name))
        object.__setattr__(self, "_unknown", unknown)
        object.__setattr__(self, "_sigma_priors", priors)
        bases = []
        for experiment in experiments:
            base = dict(self.model.fixed)
            base.update(experiment.conditions)
            bases.append(MappingProxyType(base))
        object.__setattr__(self, "_bases", tuple(bases))

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
        """Return ``sigma[q]`` names of unknown noise scales.

        References:
            [REF-GELMAN-ETAL-2013, Section 14.2]. No equation is evaluated.

        """
        return tuple(f"sigma[{q}]" for q in self._unknown)

    @property
    def vector_names(self) -> tuple[str, ...]:
        """Return names of the full vector ``(theta, log sigma)``.

        References:
            Project API policy. No equation is evaluated.

        """
        return self.parameter_names + tuple(f"log_sigma[{q}]" for q in self._unknown)

    @property
    def dimension(self) -> int:
        """Return the length of the full parameter vector.

        References:
            Project API policy. No equation is evaluated.

        """
        return self.model.dimension + len(self._unknown)

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

        Args:
            parameters: ``(theta, log sigma_unknown)``.

        Returns:
            Sum of independent Gaussian or Student-t log-densities.

        Equation:
            [EQ-CAL-LIK-001].

        References:
            [REF-GELMAN-ETAL-2013, Section 3.2], [REF-LANGE-LITTLE-TAYLOR-1989].

        """
        theta, sigmas = self.split(parameters)
        return self._log_likelihood(theta, sigmas)

    def _log_likelihood(self, theta: Vector, sigmas: Mapping[str, float]) -> float:
        """Evaluate the likelihood for split parameters.

        References:
            [EQ-CAL-LIK-001], [REF-GELMAN-ETAL-2013, Section 3.2].

        """
        residuals = self.residuals(theta)
        return math.fsum(
            self._noise_by_name[quantity].log_density(residual, sigmas[quantity])
            for (_, quantity, _), residual in zip(self._pairs, residuals, strict=True)
        )

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
        return (
            self._log_likelihood(theta, sigmas)
            - 0.5 * math.fsum(value * value for value in u)
            + noise_prior
        )

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

    def _weighted_jacobian(
        self, theta: Vector, scales: Mapping[str, float]
    ) -> tuple[Matrix, int]:
        """Return the Jacobian of predictions divided by total noise scales.

        References:
            [EQ-CAL-FISHER-001], [EQ-OPT-FD-001].

        """
        weights = tuple(
            math.sqrt(scales[q] ** 2 + self._noise_by_name[q].discrepancy_variance)
            for _, q, _ in self._pairs
        )

        def weighted(point: Vector) -> Vector:
            """Return weighted predictions.

            References:
                [EQ-CAL-WLS-001]. No new equation is introduced.

            """
            return tuple(
                p / w for p, w in zip(self.predictions(point), weights, strict=True)
            )

        return central_jacobian(weighted, theta)

    def identifiability(
        self,
        theta: Sequence[float] | None = None,
        *,
        prior_scaled: bool = True,
        condition_warning: float = 1.0e8,
    ) -> FisherIdentifiability:
        """Diagnose local identifiability at ``theta`` (default: prior means).

        Unknown noise scales are replaced by their prior medians.

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
        _, sigmas = self.split(point + self._default_log_sigmas())
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

        Args:
            initial: Starting ``theta``; defaults to the prior means.
            tolerance: Levenberg-Marquardt tolerance.
            max_iterations: Levenberg-Marquardt iteration limit.
            max_reweights: Reweighting passes for unknown noise.

        Returns:
            Estimate, covariance, noise scales, residuals, and identifiability.

        Raises:
            ValueError: If there are no more observations than parameters when
                noise must be estimated, or ``J'J`` is singular.

        Equation:
            [EQ-CAL-WLS-001], [EQ-OPT-LM-001], [EQ-OPT-LM-002].

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

            result = least_squares(
                weighted_residuals,
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
        self, initial: Sequence[float] | None = None
    ) -> LaplaceApproximation:
        """Return the Laplace approximation of the posterior of ``(theta, log sigma)``.

        The MAP point maximizes :meth:`log_posterior` by BFGS in the physical
        ``theta`` (and log-noise) coordinates; the covariance is the inverse
        finite-difference Hessian there. Names follow :attr:`vector_names`.
        The approximation is exact for linear-Gaussian problems with Gaussian
        priors and known noise.

        Args:
            initial: Full starting vector; defaults to prior means and log
                prior medians of unknown noise scales.

        Returns:
            Laplace approximation (mode = MAP estimate).

        Raises:
            ValueError: If the start has zero density or the Hessian is not
                positive definite.

        Equation:
            [EQ-LAPLACE-001], [EQ-CAL-POST-001].

        References:
            [REF-TIERNEY-KADANE-1986], [REF-GELMAN-ETAL-2013, Section 4.1].

        """
        start = (
            self.model.nominal() + self._default_log_sigmas()
            if initial is None
            else as_vector(initial, name="initial")
        )
        return laplace_approximation(self.log_posterior, start)

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
    ) -> CalibrationPosterior:
        """Sample the posterior by MCMC in standard-normal space.

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

        Args:
            draws: Kept draws per chain.
            warmup: Adaptation and burn-in iterations per chain.
            chains: Number of independent chains.
            thin: Thinning interval.
            dispersion: Start-point spread relative to the Laplace scale.
            seed: Integer seed or stream.
            method: ``"adaptive-metropolis"`` (default), ``"hmc"``, or
                ``"nuts"``.

        Returns:
            Posterior with physical and sampling-space chains.

        Raises:
            ValueError: For invalid run arguments or an unknown ``method``.

        Equation:
            [EQ-CAL-STD-001], [EQ-MCMC-AM-001], [EQ-MCMC-MH-001],
            [EQ-HMC-LEAPFROG-001], [EQ-NUTS-SLICE-001].

        References:
            [REF-HAARIO-SAKSMAN-TAMMINEN-2001], [REF-GELMAN-RUBIN-1992],
            [REF-LIU-DERKIUREGHIAN-1986], [REF-HOFFMAN-GELMAN-2014],
            [REF-NEAL-2011].

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
        origin = tuple(0.0 for _ in range(self.model.dimension))
        origin = origin + self._default_log_sigmas()
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
        names = tuple(f"u[{name}]" for name in self.parameter_names) + tuple(
            f"log_sigma[{q}]" for q in self._unknown
        )
        sampler: HamiltonianResult | None = None
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
        physical = sampled.transformed(
            self.to_physical, self.parameter_names + self.noise_names
        )
        return CalibrationPosterior(
            self.parameter_names,
            self.noise_names,
            physical,
            sampled,
            laplace,
            sampler,
        )
