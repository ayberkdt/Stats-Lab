"""Conjugate linear-Gaussian inference and predictive diagnostics.

Known-variance updates use a Gaussian prior; unknown-variance linear
regression uses the Normal-inverse-gamma conjugate family with Student-t
marginal and predictive laws.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field

from probstat.distributions._special import regularized_beta, regularized_gamma_p
from probstat.foundations import coerce_finite_sample
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import (
    Matrix,
    Vector,
    as_matrix,
    as_vector,
    cholesky,
    cholesky_semidefinite,
    inverse_spd,
    matmul,
    matvec,
    quadratic_form,
    sandwich,
    scale_matrix,
    symmetrize,
    transpose,
)
from probstat.multivariate import MultivariateNormal
from probstat.optimization import brent_root


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
        mean = as_vector(self.mean, name="mean")
        if not mean:
            raise ValueError("mean must contain at least one component")
        covariance = _symmetric_covariance(
            self.covariance, dimension=len(mean), name="covariance"
        )
        cholesky = cholesky_semidefinite(covariance)
        parameter_variance = as_vector(
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
    design = as_matrix(design_matrix, name="design_matrix")
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

    prior_precision = inverse_spd(prior.covariance)
    gram = matmul(transpose(design), design)
    posterior_precision = tuple(
        tuple(
            prior_precision[row][column] + gram[row][column] / variance
            for column in range(prior.dimension)
        )
        for row in range(prior.dimension)
    )
    posterior_covariance = inverse_spd(posterior_precision)
    prior_information = matvec(prior_precision, prior.mean)
    data_information = matvec(transpose(design), observed)
    information = tuple(
        prior_information[index] + data_information[index] / variance
        for index in range(prior.dimension)
    )
    posterior_mean = matvec(posterior_covariance, information)
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
    design = as_matrix(design_matrix, name="design_matrix")
    if any(len(row) != posterior.dimension for row in design):
        raise ValueError("every design row must match posterior dimension")
    discrepancy = require_finite_real(
        discrepancy_variance, name="discrepancy_variance"
    )
    if discrepancy < 0.0:
        raise ValueError("discrepancy_variance must be non-negative")
    parameter_covariance = sandwich(design, posterior.covariance)
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


def _symmetric_covariance(
    values: Iterable[Iterable[object]], *, dimension: int, name: str
) -> Matrix:
    """Validate a finite symmetric covariance matrix of known dimension.

    References:
        [REF-GOLUB-VANLOAN-2013, Chapter 4]. No equation is evaluated.

    """
    matrix = as_matrix(values, name=name)
    if len(matrix) != dimension or any(len(row) != dimension for row in matrix):
        raise ValueError(f"{name} must be square with mean dimension")
    return symmetrize(matrix, name=name)


# ---------------------------------------------------------------------------
# Linear regression with unknown noise variance: Normal-inverse-gamma
# ---------------------------------------------------------------------------


def student_t_cdf(value: float, degrees_of_freedom: float) -> float:
    """Return the standard Student-t CDF through the incomplete beta function.

    Args:
        value: Finite evaluation point.
        degrees_of_freedom: Positive degrees of freedom ``nu``.

    Returns:
        ``F_nu(t) = 1 - I_{nu/(nu+t^2)}(nu/2, 1/2) / 2`` for ``t >= 0``,
        extended by symmetry.

    Raises:
        ValueError: If ``degrees_of_freedom`` is not positive.

    Equation:
        [EQ-DIST-T-001], [EQ-NUM-BETA-001].

    References:
        [REF-NIST-DLMF-BETA, Section 8.17], [REF-CASELLA-BERGER-2002, Ch. 5].

    """
    t = require_finite_real(value, name="value")
    nu = require_finite_real(degrees_of_freedom, name="degrees_of_freedom")
    if nu <= 0.0:
        raise ValueError("degrees_of_freedom must be positive")
    if t == 0.0:
        return 0.5
    tail = 0.5 * regularized_beta(nu / (nu + t * t), 0.5 * nu, 0.5)
    return 1.0 - tail if t > 0.0 else tail


def student_t_quantile(probability: float, degrees_of_freedom: float) -> float:
    """Return the standard Student-t quantile by bracketed Brent inversion.

    Args:
        probability: Level strictly inside ``(0, 1)``.
        degrees_of_freedom: Positive degrees of freedom.

    Returns:
        ``t`` with ``F_nu(t) = probability``.

    Raises:
        ValueError: If the probability or degrees of freedom are invalid.

    Equation:
        [EQ-DIST-T-001], [EQ-OPT-BRENT-001].

    References:
        [REF-BRENT-1973, Chapter 4], [REF-NIST-DLMF-BETA, Section 8.17].

    """
    p = require_finite_real(probability, name="probability")
    if not 0.0 < p < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    nu = require_finite_real(degrees_of_freedom, name="degrees_of_freedom")
    if nu <= 0.0:
        raise ValueError("degrees_of_freedom must be positive")
    if p == 0.5:
        return 0.0
    target = max(p, 1.0 - p)
    upper = 1.0
    while student_t_cdf(upper, nu) < target:
        upper *= 2.0
        if upper > 1e300:
            raise ArithmeticError("Student-t quantile bracket overflowed")

    def excess(t: float) -> float:
        """Return ``F_nu(t) - target`` for the root search.

        References:
            [EQ-DIST-T-001]. No new equation is introduced.

        """
        return student_t_cdf(t, nu) - target

    root = brent_root(excess, 0.0, upper, tolerance=1e-13 * max(1.0, upper)).root
    return root if p > 0.5 else -root


def _gamma_quantile(probability: float, shape: float) -> float:
    """Return the unit-rate gamma quantile by bracketed Brent inversion.

    References:
        [EQ-NUM-GAMMA-001], [EQ-OPT-BRENT-001], [REF-NIST-DLMF-GAMMA, 8.2].

    """
    upper = max(1.0, shape)
    while regularized_gamma_p(shape, upper) < probability:
        upper *= 2.0
        if upper > 1e300:
            raise ArithmeticError("gamma quantile bracket overflowed")

    def excess(x: float) -> float:
        """Return ``P(shape, x) - probability`` for the root search.

        References:
            [EQ-NUM-GAMMA-001]. No new equation is introduced.

        """
        return regularized_gamma_p(shape, x) - probability

    return brent_root(excess, 0.0, upper, tolerance=1e-14 * max(1.0, upper)).root


def _interval_tails(probability: object) -> tuple[float, float]:
    """Validate a central-interval probability and return its tail levels.

    References:
        [REF-GELMAN-ETAL-2013, Section 2.3]. No equation is evaluated.

    """
    level = require_finite_real(probability, name="probability")
    if not 0.0 < level < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    return 0.5 * (1.0 - level), 0.5 * (1.0 + level)


@dataclass(frozen=True, slots=True)
class StudentTPredictive:
    """Multivariate Student-t law ``t_nu(location, scale_matrix)``.

    Every component ``i`` is a univariate ``t_nu`` with location
    ``location[i]`` and scale ``sqrt(scale_matrix[i][i])``; the covariance is
    ``nu / (nu - 2) * scale_matrix`` when ``nu > 2``.

    References:
        [REF-GELMAN-ETAL-2013, Section 14.2], [REF-MURPHY-2012, Section 7.6.3].

    """

    location: Vector
    scale_matrix: Matrix
    degrees_of_freedom: float

    @property
    def variances(self) -> Vector:
        """Return component variances ``nu/(nu-2) S_ii``.

        Raises:
            ValueError: If ``nu <= 2`` (infinite variance).

        Equation:
            [EQ-BAYES-NIG-003].

        References:
            [REF-MURPHY-2012, Section 7.6.3].

        """
        nu = self.degrees_of_freedom
        if nu <= 2.0:
            raise ValueError("variance is infinite unless degrees_of_freedom > 2")
        return tuple(
            nu / (nu - 2.0) * self.scale_matrix[i][i]
            for i in range(len(self.location))
        )

    def interval(self, index: int, probability: float = 0.95) -> tuple[float, float]:
        """Return the equal-tailed central interval of one component.

        Args:
            index: Component index.
            probability: Central probability in ``(0, 1)``.

        Returns:
            Lower and upper interval bounds.

        Equation:
            [EQ-BAYES-NIG-003], [EQ-DIST-T-001].

        References:
            [REF-MURPHY-2012, Section 7.6.3].

        """
        _, upper = _interval_tails(probability)
        half = student_t_quantile(upper, self.degrees_of_freedom) * math.sqrt(
            self.scale_matrix[index][index]
        )
        return self.location[index] - half, self.location[index] + half


@dataclass(frozen=True, slots=True)
class NormalInverseGamma:
    """Normal-inverse-gamma law for linear coefficients and noise variance.

    ``sigma^2 ~ InvGamma(shape, rate)`` with density proportional to
    ``(sigma^2)^(-shape-1) exp(-rate / sigma^2)``, and
    ``beta | sigma^2 ~ N(mean, sigma^2 scale_matrix)``. It is both the
    conjugate prior and the posterior of ``y = X beta + e``,
    ``e ~ N(0, sigma^2 I)`` with ``sigma^2`` unknown.

    Args:
        mean: Coefficient location ``m``.
        scale_matrix: Symmetric positive-definite ``V``.
        shape: Positive inverse-gamma shape ``a``.
        rate: Positive inverse-gamma scale ``b``.

    Raises:
        ValueError: If shapes disagree, ``V`` is not positive definite, or
            ``a``, ``b`` are not positive.

    References:
        [REF-GELMAN-ETAL-2013, Section 14.2], [REF-MURPHY-2012, Section 7.6.3].

    """

    mean: Vector
    scale_matrix: Matrix
    shape: float
    rate: float
    _lower: Matrix = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate parameters and cache the Cholesky factor of ``V``.

        References:
            [EQ-BAYES-NIG-001], [REF-MURPHY-2012, Section 7.6.3].

        """
        mean = as_vector(self.mean, name="mean")
        if not mean:
            raise ValueError("mean must be non-empty")
        scale = _symmetric_covariance(
            self.scale_matrix, dimension=len(mean), name="scale_matrix"
        )
        try:
            lower = cholesky(scale)
        except ValueError as error:
            raise ValueError("scale_matrix must be positive definite") from error
        shape = require_finite_real(self.shape, name="shape")
        rate = require_finite_real(self.rate, name="rate")
        if shape <= 0.0 or rate <= 0.0:
            raise ValueError("shape and rate must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale_matrix", scale)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "_lower", lower)

    @property
    def dimension(self) -> int:
        """Return the number of regression coefficients.

        References:
            [REF-MURPHY-2012, Section 7.6.3]. No equation is evaluated.

        """
        return len(self.mean)

    @property
    def degrees_of_freedom(self) -> float:
        """Return the Student-t degrees of freedom ``2 a``.

        Equation:
            [EQ-BAYES-NIG-002].

        References:
            [REF-MURPHY-2012, Section 7.6.3].

        """
        return 2.0 * self.shape

    @property
    def noise_variance_mean(self) -> float:
        """Return ``E[sigma^2] = b / (a - 1)``.

        Raises:
            ValueError: If ``a <= 1`` (infinite mean).

        Equation:
            [EQ-BAYES-NIG-002].

        References:
            [REF-GELMAN-ETAL-2013, Appendix A].

        """
        if self.shape <= 1.0:
            raise ValueError("E[sigma^2] is infinite unless shape > 1")
        return self.rate / (self.shape - 1.0)

    @property
    def noise_variance_mode(self) -> float:
        """Return the inverse-gamma mode ``b / (a + 1)``.

        Equation:
            [EQ-BAYES-NIG-002].

        References:
            [REF-GELMAN-ETAL-2013, Appendix A].

        """
        return self.rate / (self.shape + 1.0)

    def coefficient_marginal(self) -> StudentTPredictive:
        """Return the marginal ``beta ~ t_{2a}(m, (b/a) V)``.

        Equation:
            [EQ-BAYES-NIG-002].

        References:
            [REF-MURPHY-2012, Section 7.6.3].

        """
        factor = self.rate / self.shape
        return StudentTPredictive(
            self.mean,
            scale_matrix(self.scale_matrix, factor),
            self.degrees_of_freedom,
        )

    def coefficient_interval(
        self, index: int, probability: float = 0.95
    ) -> tuple[float, float]:
        """Return an equal-tailed marginal credible interval for one coefficient.

        Args:
            index: Coefficient index.
            probability: Central probability in ``(0, 1)``.

        Returns:
            Lower and upper bounds from the ``t_{2a}`` marginal.

        Equation:
            [EQ-BAYES-NIG-002], [EQ-DIST-T-001].

        References:
            [REF-MURPHY-2012, Section 7.6.3].

        """
        return self.coefficient_marginal().interval(index, probability)

    def noise_variance_interval(self, probability: float = 0.95) -> tuple[float, float]:
        """Return an equal-tailed credible interval for ``sigma^2``.

        Args:
            probability: Central probability in ``(0, 1)``.

        Returns:
            ``(b / G_upper, b / G_lower)`` with unit-rate gamma quantiles.

        Equation:
            [EQ-BAYES-NIG-002], [EQ-NUM-GAMMA-001].

        References:
            [REF-GELMAN-ETAL-2013, Appendix A].

        """
        lower_tail, upper_tail = _interval_tails(probability)
        return (
            self.rate / _gamma_quantile(upper_tail, self.shape),
            self.rate / _gamma_quantile(lower_tail, self.shape),
        )

    def predictive(
        self, design_matrix: Iterable[Iterable[object]]
    ) -> StudentTPredictive:
        """Return the posterior predictive of future observations at new rows.

        Args:
            design_matrix: Rows ``X*`` with one column per coefficient.

        Returns:
            ``t_{2a}(X* m, (b/a)(I + X* V X*'))``.

        Raises:
            ValueError: If a row length differs from the dimension.

        Equation:
            [EQ-BAYES-NIG-003].

        References:
            [REF-MURPHY-2012, Section 7.6.3], [REF-GELMAN-ETAL-2013, 14.2].

        """
        design = as_matrix(design_matrix, name="design_matrix")
        if any(len(row) != self.dimension for row in design):
            raise ValueError("every design row must match the dimension")
        factor = self.rate / self.shape
        spread = sandwich(design, self.scale_matrix)
        scale = tuple(
            tuple(
                factor * (value + (1.0 if i == j else 0.0))
                for j, value in enumerate(row)
            )
            for i, row in enumerate(spread)
        )
        location = matvec(design, self.mean)
        return StudentTPredictive(location, scale, self.degrees_of_freedom)

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[tuple[Vector, float], ...]:
        """Draw ``(beta, sigma^2)`` pairs from the joint law.

        Args:
            size: Non-negative number of draws.
            seed: Integer seed or stream.

        Returns:
            Pairs with ``sigma^2 = b / G``, ``G ~ Gamma(a, 1)``, and
            ``beta = m + sigma L z``.

        Equation:
            [EQ-BAYES-NIG-002].

        References:
            [REF-DEVROYE-1986, Chapter IX], [REF-MURPHY-2012, Section 7.6.3].

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        draws: list[tuple[Vector, float]] = []
        for _ in range(count):
            variance = self.rate / generator.gammavariate(self.shape, 1.0)
            z = tuple(generator.gauss(0.0, 1.0) for _ in range(self.dimension))
            step = matvec(self._lower, z)
            root = math.sqrt(variance)
            beta = tuple(m + root * s for m, s in zip(self.mean, step, strict=True))
            draws.append((beta, variance))
        return tuple(draws)


def condition_linear_normal_inverse_gamma(
    prior: NormalInverseGamma,
    design_matrix: Iterable[Iterable[object]],
    observations: Iterable[object],
) -> NormalInverseGamma:
    """Update a Normal-inverse-gamma prior with linear-Gaussian data.

    ``V_n = (V_0^-1 + X'X)^-1``, ``m_n = V_n (V_0^-1 m_0 + X'y)``,
    ``a_n = a_0 + n/2`` and
    ``b_n = b_0 + ((y - X m_n)'(y - X m_n) + (m_n - m_0)' V_0^-1 (m_n - m_0))/2``.
    The residual form of ``b_n`` avoids cancellation in
    ``y'y + m_0'V_0^-1 m_0 - m_n'V_n^-1 m_n``.

    Args:
        prior: Proper Normal-inverse-gamma prior.
        design_matrix: Rows of ``X`` with one column per coefficient.
        observations: Finite responses ``y``, one per row.

    Returns:
        Posterior Normal-inverse-gamma law.

    Raises:
        TypeError: If ``prior`` has the wrong type.
        ValueError: If shapes disagree.

    Equation:
        [EQ-BAYES-NIG-001].

    References:
        [REF-MURPHY-2012, Section 7.6.3], [REF-GELMAN-ETAL-2013, Section 14.2].

    """
    if not isinstance(prior, NormalInverseGamma):
        raise TypeError("prior must be a NormalInverseGamma")
    design = as_matrix(design_matrix, name="design_matrix")
    observed = coerce_finite_sample(observations, name="observations")
    if len(design) != len(observed):
        raise ValueError("design rows and observations must have equal length")
    if any(len(row) != prior.dimension for row in design):
        raise ValueError("every design row must match prior dimension")
    prior_precision = inverse_spd(prior.scale_matrix)
    gram = matmul(transpose(design), design)
    size = prior.dimension
    posterior_precision = tuple(
        tuple(prior_precision[i][j] + gram[i][j] for j in range(size))
        for i in range(size)
    )
    raw_scale = inverse_spd(posterior_precision)
    scale = tuple(
        tuple(0.5 * (raw_scale[i][j] + raw_scale[j][i]) for j in range(size))
        for i in range(size)
    )
    information = tuple(
        a + b
        for a, b in zip(
            matvec(prior_precision, prior.mean),
            matvec(transpose(design), observed),
            strict=True,
        )
    )
    mean = matvec(scale, information)
    residuals = tuple(
        y - fitted for y, fitted in zip(observed, matvec(design, mean), strict=True)
    )
    shift = tuple(a - b for a, b in zip(mean, prior.mean, strict=True))
    rate = prior.rate + 0.5 * (
        math.fsum(r * r for r in residuals) + quadratic_form(prior_precision, shift)
    )
    return NormalInverseGamma(mean, scale, prior.shape + 0.5 * len(observed), rate)
