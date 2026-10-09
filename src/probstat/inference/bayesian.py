"""Conjugate linear-Gaussian inference and predictive diagnostics."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field

from probstat.foundations import coerce_finite_sample
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.multivariate import MultivariateNormal

Vector = tuple[float, ...]
Matrix = tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class GaussianLinearPosterior:
    """Posterior parameter law for a known-variance linear-Gaussian model.

    References:
        [REF-GELMAN-ETAL-2013, Sections 3.1-3.3].

    """

    mean: Vector
    covariance: Matrix
    observation_variance: float
    _law: MultivariateNormal = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the posterior through the multivariate-normal contract.

        References:
            [EQ-BAYES-LIN-POST-001], [REF-GELMAN-ETAL-2013, Section 3.3].

        """
        law = MultivariateNormal(self.mean, self.covariance)
        variance = require_finite_real(
            self.observation_variance, name="observation_variance"
        )
        if variance <= 0.0:
            raise ValueError("observation_variance must be positive")
        object.__setattr__(self, "mean", law.mean)
        object.__setattr__(self, "covariance", law.covariance)
        object.__setattr__(self, "observation_variance", variance)
        object.__setattr__(self, "_law", law)

    @property
    def dimension(self) -> int:
        """Return the number of calibrated parameters.

        References:
            [REF-GELMAN-ETAL-2013, Section 3.3]. No equation is evaluated.

        """
        return len(self.mean)

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[Vector, ...]:
        """Draw reproducible parameter samples from the posterior.

        Args:
            size: Non-negative number of draws.
            seed: Optional call-local pseudo-random seed.

        Returns:
            Parameter vectors distributed according to the posterior law.

        Equation:
            [EQ-BAYES-POST-SAMPLE-001].

        References:
            [REF-DEVROYE-1986, Chapter XI].

        """
        return self._law.sample(size=size, seed=seed)


@dataclass(frozen=True, slots=True)
class PosteriorPredictive:
    """Joint Gaussian predictive law at a declared set of design rows.

    References:
        [REF-GELMAN-ETAL-2013, Section 6.3].

    """

    mean: Vector
    covariance: Matrix
    parameter_variance: Vector
    observation_variance: float
    discrepancy_variance: float
    _cholesky: Matrix = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate components and factor a positive-semidefinite covariance.

        References:
            [EQ-BAYES-PRED-001], [REF-GELMAN-ETAL-2013, Section 6.3].

        """
        mean = _finite_vector(self.mean, name="mean")
        if not mean:
            raise ValueError("mean must contain at least one component")
        covariance = _symmetric_covariance(
            self.covariance, dimension=len(mean), name="covariance"
        )
        cholesky = _cholesky_semidefinite(covariance)
        parameter_variance = _finite_vector(
            self.parameter_variance, name="parameter_variance"
        )
        if len(parameter_variance) != len(mean):
            raise ValueError("parameter_variance must match predictive dimension")
        if any(value < 0.0 for value in parameter_variance):
            raise ValueError("parameter_variance must be non-negative")
        observation_variance = require_finite_real(
            self.observation_variance, name="observation_variance"
        )
        discrepancy_variance = require_finite_real(
            self.discrepancy_variance, name="discrepancy_variance"
        )
        if observation_variance < 0.0 or discrepancy_variance < 0.0:
            raise ValueError("predictive variance components must be non-negative")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "parameter_variance", parameter_variance)
        object.__setattr__(self, "observation_variance", observation_variance)
        object.__setattr__(self, "discrepancy_variance", discrepancy_variance)
        object.__setattr__(self, "_cholesky", cholesky)

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[Vector, ...]:
        """Draw reproducible observations from the predictive distribution.

        Args:
            size: Non-negative number of predictive replicates.
            seed: Optional call-local pseudo-random seed.

        Returns:
            Joint predictive vectors.

        Equation:
            [EQ-BAYES-PRED-001].

        References:
            [REF-GELMAN-ETAL-2013, Section 6.3].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = random.Random(seed)
        samples: list[Vector] = []
        for _ in range(checked_size):
            independent = tuple(generator.gauss(0.0, 1.0) for _ in self.mean)
            sample = tuple(
                self.mean[row]
                + math.fsum(
                    self._cholesky[row][column] * independent[column]
                    for column in range(row + 1)
                )
                for row in range(len(self.mean))
            )
            if not all(math.isfinite(value) for value in sample):
                raise OverflowError("posterior-predictive sample is not finite")
            samples.append(sample)
        return tuple(samples)


@dataclass(frozen=True, slots=True)
class PosteriorPredictiveCheck:
    """A simulation-based posterior-predictive discrepancy comparison.

    References:
        [REF-GELMAN-ETAL-2013, Section 6.3].

    """

    observed_discrepancy: float
    replicated_discrepancies: tuple[float, ...]
    tail_probability: float
    draws: int


def gaussian_log_likelihood(
    observations: Iterable[object],
    predictions: Iterable[object],
    *,
    variance: object,
) -> float:
    """Evaluate an independent homoscedastic Gaussian log-likelihood.

    Args:
        observations: Finite measured responses.
        predictions: Finite model predictions paired with observations.
        variance: Known positive total error variance.

    Returns:
        Sum of pointwise Gaussian log densities.

    Equation:
        [EQ-BAYES-LIKE-001].

    References:
        [REF-GELMAN-ETAL-2013, Section 3.2].

    """
    observed = coerce_finite_sample(observations, name="observations")
    predicted = coerce_finite_sample(predictions, name="predictions")
    if len(observed) != len(predicted):
        raise ValueError("observations and predictions must have equal length")
    checked_variance = require_finite_real(variance, name="variance")
    if checked_variance <= 0.0:
        raise ValueError("variance must be positive")
    squared_error = math.fsum(
        (value - center) ** 2
        for value, center in zip(observed, predicted, strict=True)
    )
    return -0.5 * (
        len(observed) * math.log(2.0 * math.pi * checked_variance)
        + squared_error / checked_variance
    )


def condition_linear_gaussian(
    prior: MultivariateNormal,
    design_matrix: Iterable[Iterable[object]],
    observations: Iterable[object],
    *,
    observation_variance: object,
) -> GaussianLinearPosterior:
    """Condition a Gaussian prior under a known-variance linear observation model.

    Args:
        prior: Positive-definite multivariate Gaussian parameter prior.
        design_matrix: Non-empty rows in ``y = H*theta + error``.
        observations: Finite measured responses, one per design row.
        observation_variance: Known positive independent noise variance.

    Returns:
        Analytic Gaussian posterior for the parameter vector.

    Raises:
        TypeError: If inputs contain non-real values.
        ValueError: If shapes disagree or variance is not positive.

    Equation:
        [EQ-BAYES-RULE-001], [EQ-BAYES-LIN-POST-001].

    References:
        [REF-GELMAN-ETAL-2013, Section 3.3],
        [REF-BISHOP-2006, Section 3.3].

    """
    if not isinstance(prior, MultivariateNormal):
        raise TypeError("prior must be a MultivariateNormal")
    design = _finite_matrix(design_matrix, name="design_matrix")
    observed = coerce_finite_sample(observations, name="observations")
    if len(design) != len(observed):
        raise ValueError("design rows and observations must have equal length")
    if any(len(row) != prior.dimension for row in design):
        raise ValueError("every design row must match prior dimension")
    variance = require_finite_real(
        observation_variance, name="observation_variance"
    )
    if variance <= 0.0:
        raise ValueError("observation_variance must be positive")

    prior_precision = _inverse_spd(prior.covariance)
    gram = _transpose_product(design, design)
    posterior_precision = tuple(
        tuple(
            prior_precision[row][column] + gram[row][column] / variance
            for column in range(prior.dimension)
        )
        for row in range(prior.dimension)
    )
    posterior_covariance = _inverse_spd(posterior_precision)
    prior_information = _matrix_vector(prior_precision, prior.mean)
    data_information = _transpose_vector_product(design, observed)
    information = tuple(
        prior_information[index] + data_information[index] / variance
        for index in range(prior.dimension)
    )
    posterior_mean = _matrix_vector(posterior_covariance, information)
    return GaussianLinearPosterior(
        posterior_mean, posterior_covariance, variance
    )


def posterior_predictive(
    posterior: GaussianLinearPosterior,
    design_matrix: Iterable[Iterable[object]],
    *,
    include_observation_noise: bool = True,
    discrepancy_variance: object = 0.0,
) -> PosteriorPredictive:
    """Construct the Gaussian posterior-predictive law at new design rows.

    Args:
        posterior: Conditioned parameter distribution.
        design_matrix: New rows mapping parameters to outputs.
        include_observation_noise: Whether the target is a future observation;
            false targets the latent physical mean instead.
        discrepancy_variance: Non-negative additive model-discrepancy variance.

    Returns:
        Joint predictive mean, covariance, and separated variance components.

    Equation:
        [EQ-BAYES-PRED-001], [EQ-CAL-DISC-001].

    References:
        [REF-GELMAN-ETAL-2013, Section 6.3],
        [REF-KENNEDY-OHAGAN-2001].

    """
    if not isinstance(posterior, GaussianLinearPosterior):
        raise TypeError("posterior must be a GaussianLinearPosterior")
    if not isinstance(include_observation_noise, bool):
        raise TypeError("include_observation_noise must be a bool")
    design = _finite_matrix(design_matrix, name="design_matrix")
    if any(len(row) != posterior.dimension for row in design):
        raise ValueError("every design row must match posterior dimension")
    discrepancy = require_finite_real(
        discrepancy_variance, name="discrepancy_variance"
    )
    if discrepancy < 0.0:
        raise ValueError("discrepancy_variance must be non-negative")
    parameter_covariance = _sandwich(design, posterior.covariance)
    observation = posterior.observation_variance if include_observation_noise else 0.0
    additive = observation + discrepancy
    covariance = tuple(
        tuple(
            value + (additive if row == column else 0.0)
            for column, value in enumerate(matrix_row)
        )
        for row, matrix_row in enumerate(parameter_covariance)
    )
    mean = tuple(
        math.fsum(
            value * weight
            for value, weight in zip(row, posterior.mean, strict=True)
        )
        for row in design
    )
    parameter_variance = tuple(
        parameter_covariance[index][index] for index in range(len(design))
    )
    return PosteriorPredictive(
        mean,
        covariance,
        parameter_variance,
        observation,
        discrepancy,
    )


def posterior_predictive_check(
    predictive: PosteriorPredictive,
    observations: Iterable[object],
    *,
    draws: int = 2_000,
    seed: int | None = None,
) -> PosteriorPredictiveCheck:
    """Compare observed and replicated standardized quadratic discrepancies.

    Args:
        predictive: Joint predictive law for the observation sites.
        observations: Held-out or calibration observations at those sites.
        draws: Positive number of predictive replicates.
        seed: Optional call-local pseudo-random seed.

    Returns:
        Observed statistic, replicate statistics, and upper-tail proportion.

    Equation:
        [EQ-BAYES-PPC-001].

    References:
        [REF-GELMAN-ETAL-2013, Section 6.3].

    """
    if not isinstance(predictive, PosteriorPredictive):
        raise TypeError("predictive must be a PosteriorPredictive")
    observed = coerce_finite_sample(observations, name="observations")
    if len(observed) != len(predictive.mean):
        raise ValueError("observations must match predictive dimension")
    checked_draws = require_non_negative_integer(draws, name="draws")
    if checked_draws == 0:
        raise ValueError("draws must be positive")
    try:
        diagnostic_law = MultivariateNormal(
            predictive.mean, predictive.covariance
        )
    except ValueError as error:
        raise ValueError(
            "posterior predictive check requires positive-definite covariance"
        ) from error
    observed_discrepancy = diagnostic_law.mahalanobis_squared(observed)
    replicated = predictive.sample(checked_draws, seed=seed)
    replicated_discrepancies = tuple(
        diagnostic_law.mahalanobis_squared(sample) for sample in replicated
    )
    extreme = sum(
        value >= observed_discrepancy for value in replicated_discrepancies
    )
    return PosteriorPredictiveCheck(
        observed_discrepancy,
        replicated_discrepancies,
        extreme / checked_draws,
        checked_draws,
    )


def _finite_vector(values: Iterable[object], *, name: str) -> Vector:
    """Materialize a possibly empty finite vector.

    References:
        Project validation policy. No mathematical equation is evaluated.

    """
    try:
        materialized = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be iterable") from error
    return tuple(
        require_finite_real(value, name=f"{name}[{index}]")
        for index, value in enumerate(materialized)
    )


def _finite_matrix(values: Iterable[Iterable[object]], *, name: str) -> Matrix:
    """Materialize a non-empty rectangular finite matrix.

    References:
        [REF-GOLUB-VANLOAN-2013, Chapter 1]. No equation is evaluated.

    """
    try:
        rows = tuple(
            _finite_vector(row, name=f"{name}[{index}]")
            for index, row in enumerate(values)
        )
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of rows") from error
    if not rows or not rows[0]:
        raise ValueError(f"{name} must be non-empty")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{name} must be rectangular")
    return rows


def _cholesky(matrix: Matrix) -> Matrix:
    """Compute the lower Cholesky factor of a positive-definite matrix.

    References:
        [EQ-LINALG-001], [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            residual = matrix[row][column] - math.fsum(
                lower[row][index] * lower[column][index]
                for index in range(column)
            )
            if row == column:
                if residual <= 0.0:
                    raise ValueError("matrix must be positive definite")
                lower[row][column] = math.sqrt(residual)
            else:
                lower[row][column] = residual / lower[column][column]
    return tuple(tuple(row) for row in lower)


def _symmetric_covariance(
    values: Iterable[Iterable[object]], *, dimension: int, name: str
) -> Matrix:
    """Validate a finite symmetric covariance matrix of known dimension.

    References:
        [REF-GOLUB-VANLOAN-2013, Chapter 4]. No equation is evaluated.

    """
    matrix = _finite_matrix(values, name=name)
    if len(matrix) != dimension or any(len(row) != dimension for row in matrix):
        raise ValueError(f"{name} must be square with mean dimension")
    symmetric: list[tuple[float, ...]] = []
    for row, values_row in enumerate(matrix):
        result_row: list[float] = []
        for column, value in enumerate(values_row):
            mirror = matrix[column][row]
            tolerance = 1e-12 * max(1.0, abs(value), abs(mirror))
            if not math.isclose(value, mirror, rel_tol=0.0, abs_tol=tolerance):
                raise ValueError(f"{name} must be symmetric")
            result_row.append(0.5 * (value + mirror))
        symmetric.append(tuple(result_row))
    return tuple(symmetric)


def _cholesky_semidefinite(matrix: Matrix) -> Matrix:
    """Compute a lower square root for a positive-semidefinite covariance.

    References:
        [EQ-LINALG-001], [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    size = len(matrix)
    scale = max(max(abs(value) for value in row) for row in matrix)
    tolerance = 1e-12 * max(1.0, scale)
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            residual = matrix[row][column] - math.fsum(
                lower[row][index] * lower[column][index]
                for index in range(column)
            )
            if row == column:
                if residual < -tolerance:
                    raise ValueError("covariance must be positive semidefinite")
                lower[row][column] = math.sqrt(max(0.0, residual))
            elif lower[column][column] > tolerance:
                lower[row][column] = residual / lower[column][column]
            elif abs(residual) > tolerance:
                raise ValueError("covariance must be positive semidefinite")
    return tuple(tuple(row) for row in lower)


def _solve_spd(matrix: Matrix, right_hand_side: Vector) -> Vector:
    """Solve a symmetric positive-definite linear system by Cholesky factors.

    References:
        [EQ-LINALG-002], [REF-GOLUB-VANLOAN-2013, Sections 3.1 and 4.2].

    """
    lower = _cholesky(matrix)
    size = len(matrix)
    forward: list[float] = []
    for row in range(size):
        residual = right_hand_side[row] - math.fsum(
            lower[row][column] * forward[column] for column in range(row)
        )
        forward.append(residual / lower[row][row])
    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        residual = forward[row] - math.fsum(
            lower[column][row] * solution[column]
            for column in range(row + 1, size)
        )
        solution[row] = residual / lower[row][row]
    return tuple(solution)


def _inverse_spd(matrix: Matrix) -> Matrix:
    """Invert a symmetric positive-definite matrix by repeated solves.

    References:
        [EQ-LINALG-002], [REF-GOLUB-VANLOAN-2013, Sections 3.1 and 4.2].

    """
    size = len(matrix)
    columns = tuple(
        _solve_spd(
            matrix,
            tuple(1.0 if row == column else 0.0 for row in range(size)),
        )
        for column in range(size)
    )
    return tuple(
        tuple(columns[column][row] for column in range(size))
        for row in range(size)
    )


def _matrix_vector(matrix: Matrix, vector: Vector) -> Vector:
    """Multiply a dense matrix by a compatible vector.

    References:
        [REF-GOLUB-VANLOAN-2013, Chapter 1]. No named equation is needed.

    """
    return tuple(
        math.fsum(value * weight for value, weight in zip(row, vector, strict=True))
        for row in matrix
    )


def _transpose_product(left: Matrix, right: Matrix) -> Matrix:
    """Return ``left.T @ right`` for equally shaped matrices.

    References:
        [REF-GOLUB-VANLOAN-2013, Chapter 1]. No named equation is needed.

    """
    columns = len(left[0])
    return tuple(
        tuple(
            math.fsum(left[row][first] * right[row][second] for row in range(len(left)))
            for second in range(columns)
        )
        for first in range(columns)
    )


def _transpose_vector_product(matrix: Matrix, vector: Vector) -> Vector:
    """Return ``matrix.T @ vector``.

    References:
        [REF-GOLUB-VANLOAN-2013, Chapter 1]. No named equation is needed.

    """
    return tuple(
        math.fsum(matrix[row][column] * vector[row] for row in range(len(matrix)))
        for column in range(len(matrix[0]))
    )


def _sandwich(design: Matrix, covariance: Matrix) -> Matrix:
    """Return ``design @ covariance @ design.T``.

    References:
        [EQ-BAYES-PRED-001], [REF-BISHOP-2006, Section 3.3].

    """
    transformed = tuple(_matrix_vector(covariance, row) for row in design)
    return tuple(
        tuple(
            math.fsum(
                left_value * right_value
                for left_value, right_value in zip(
                    left, transformed[column], strict=True
                )
            )
            for column in range(len(design))
        )
        for left in design
    )
