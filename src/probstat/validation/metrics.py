"""Validation metrics comparing model predictions with experimental data.

Every metric takes predictive *samples* (from :func:`probstat.propagation.
propagate` at the experiment's conditions, from a calibrated posterior push-
forward, or supplied by the user) and observed values. Each metric answers a
different question, so none is promoted to a universal score:

* :func:`area_metric` — how far apart are the predictive CDF and the data
  ECDF, in QoI units (Ferson-Oberkampf-Ginzburg area metric);
* :func:`u_pooling` — are observations from many experiments distributed like
  draws from their respective predictive distributions (probability integral
  transform pooled against ``Uniform(0, 1)``);
* :func:`mahalanobis_metric` — standardized residuals for Gaussian predictive
  summaries with a chi-square reference;
* :func:`reliability_metric` — probability that prediction and observation
  differ by less than a stated tolerance;
* :func:`asme_validation_comparison` — ASME V&V 20 comparison error and
  validation uncertainty (an estimate of model error, not a pass/fail test);
* :func:`validation_bayes_factor` — evidence for the model's predictive density
  against an explicitly declared broader alternative.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.distributions import Uniform
from probstat.distributions._special import regularized_gamma_q
from probstat.foundations.validation import (
    coerce_finite_sample,
    require_finite_real,
    require_non_negative_finite_real,
    require_probability,
)
from probstat.inference import KolmogorovSmirnovResult, kolmogorov_smirnov_test
from probstat.linalg import Matrix, cholesky, solve_lower
from probstat.statistics import quantile
from probstat.uq import NumericalErrorBudget

_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Area metric and probability integral transform
# ---------------------------------------------------------------------------


def area_metric(predictions: Sequence[float], observations: Sequence[float]) -> float:
    """Return the area between the predictive CDF and the data ECDF.

    Both arguments define right-continuous step CDFs; the integral of their
    absolute difference is computed exactly by summing rectangle areas between
    consecutive merged breakpoints. If either sample is a point mass at ``c``,
    the result equals the mean absolute distance of the other sample from
    ``c``.

    Args:
        predictions: Non-empty finite predictive samples.
        observations: Non-empty finite observed values.

    Returns:
        Non-negative area in the unit of the quantity of interest.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a sample is empty or contains a non-finite value.

    Equation:
        [EQ-VAL-AREA-001].

    References:
        [REF-FERSON-OBERKAMPF-GINZBURG-2008], [REF-OBERKAMPF-ROY-2025,
        Chapter 12].

    """
    predicted = sorted(coerce_finite_sample(predictions, name="predictions"))
    observed = sorted(coerce_finite_sample(observations, name="observations"))
    breakpoints = sorted(set(predicted) | set(observed))
    size_p, size_o = len(predicted), len(observed)
    pieces: list[float] = []
    index_p = index_o = 0
    for left, right in zip(breakpoints, breakpoints[1:], strict=False):
        while index_p < size_p and predicted[index_p] <= left:
            index_p += 1
        while index_o < size_o and observed[index_o] <= left:
            index_o += 1
        gap = abs(index_p / size_p - index_o / size_o)
        if gap > 0.0:
            pieces.append(gap * (right - left))
    return math.fsum(pieces)


def probability_integral_transform(
    predictions: Sequence[float], observation: float
) -> float:
    """Return ``u = F_S(y)``, the predictive ECDF evaluated at an observation.

    Args:
        predictions: Non-empty finite predictive samples defining ``F_S``.
        observation: Finite observed value ``y``.

    Returns:
        Fraction of predictive samples ``<= y``, in ``[0, 1]``.

    Raises:
        TypeError: If a value is not real.
        ValueError: If the sample is empty or a value is non-finite.

    Equation:
        [EQ-VAL-UPOOL-001].

    References:
        [REF-FERSON-OBERKAMPF-GINZBURG-2008].

    """
    predicted = sorted(coerce_finite_sample(predictions, name="predictions"))
    value = require_finite_real(observation, name="observation")
    return bisect.bisect_right(predicted, value) / len(predicted)


def uniform_area(u_values: Sequence[float]) -> float:
    """Return the exact area between the ECDF of ``u`` values and ``U(0, 1)``.

    Args:
        u_values: Non-empty values in ``[0, 1]``.

    Returns:
        Area in ``[0, 1/2]``; zero only in the limit of perfectly uniform ``u``.

    Raises:
        TypeError: If a value is not real.
        ValueError: If the sample is empty or a value lies outside ``[0, 1]``.

    Equation:
        [EQ-VAL-UPOOL-002].

    References:
        [REF-FERSON-OBERKAMPF-GINZBURG-2008].

    """
    values = sorted(
        require_probability(value, name="u_values")
        for value in coerce_finite_sample(u_values, name="u_values")
    )
    count = len(values)
    edges = [0.0, *values, 1.0]
    pieces: list[float] = []
    for k in range(count + 1):
        left, right = edges[k], edges[k + 1]
        if right <= left:
            continue
        level = k / count
        if level <= left:
            pieces.append(0.5 * ((right - level) ** 2 - (left - level) ** 2))
        elif level >= right:
            pieces.append(0.5 * ((level - left) ** 2 - (level - right) ** 2))
        else:
            pieces.append(0.5 * ((level - left) ** 2 + (right - level) ** 2))
    return math.fsum(pieces)


@dataclass(frozen=True, slots=True)
class UPoolingResult:
    """Pooled probability-integral-transform evidence across experiments.

    Args:
        u_values: ``u = F_i(y)`` for every observation of every experiment.
        area: Exact area between the ECDF of ``u_values`` and ``U(0, 1)``.
        ks: One-sample Kolmogorov-Smirnov test of ``u_values`` against
            ``U(0, 1)`` (asymptotic p-value; ignores the discreteness of a
            finite predictive sample).

    References:
        [REF-FERSON-OBERKAMPF-GINZBURG-2008], [REF-NIST-GOF].

    """

    u_values: tuple[float, ...]
    area: float
    ks: KolmogorovSmirnovResult

    @property
    def count(self) -> int:
        """Return the number of pooled observations.

        References:
            [REF-FERSON-OBERKAMPF-GINZBURG-2008].

        """
        return len(self.u_values)


def u_pooling(
    pairs: Sequence[tuple[Sequence[float], Sequence[float]]],
) -> UPoolingResult:
    """Pool observations from many experiments through their predictive CDFs.

    Each pair is ``(predictions_i, observations_i)`` for experiment ``i``.
    Every observation ``y_ij`` is transformed to ``u_ij = F_i(y_ij)``. If the
    model's predictive distributions are correct, the pooled ``u`` values are
    approximately ``Uniform(0, 1)`` regardless of how the experiments differ.

    Args:
        pairs: Non-empty sequence of (predictive samples, observed values).

    Returns:
        Pooled ``u`` values, their area metric against ``U(0, 1)``, and a KS
        test.

    Raises:
        TypeError: If a value is not real.
        ValueError: If no pair is given or a sample is empty or non-finite.

    Equation:
        [EQ-VAL-UPOOL-001], [EQ-VAL-UPOOL-002], [EQ-INF-GOF-KS-001].

    References:
        [REF-FERSON-OBERKAMPF-GINZBURG-2008], [REF-NIST-GOF].

    """
    items = tuple(pairs)
    if not items:
        raise ValueError("u-pooling needs at least one experiment")
    u_values: list[float] = []
    for predictions, observations in items:
        predicted = sorted(coerce_finite_sample(predictions, name="predictions"))
        for value in coerce_finite_sample(observations, name="observations"):
            u_values.append(bisect.bisect_right(predicted, value) / len(predicted))
    pooled = tuple(u_values)
    return UPoolingResult(
        pooled, uniform_area(pooled), kolmogorov_smirnov_test(pooled, Uniform(0.0, 1.0))
    )


# ---------------------------------------------------------------------------
# Gaussian summaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MahalanobisResult:
    """Standardized residuals and Mahalanobis distance with a chi-square tail.

    Args:
        residuals: ``y - mu`` for each observation.
        standardized_residuals: ``(y_i - mu_i) / sqrt(C_ii)``.
        squared_distance: ``D^2 = r' C^-1 r``.
        degrees_of_freedom: Number of residuals.
        p_value: ``P(chi2_n >= D^2)`` under a correct Gaussian predictive law.

    References:
        [REF-MARDIA-KENT-BIBBY-1979, Chapters 1-3], [REF-LIU-CHEN-ARENDT-HUANG-2011].

    """

    residuals: tuple[float, ...]
    standardized_residuals: tuple[float, ...]
    squared_distance: float
    degrees_of_freedom: int
    p_value: float


def mahalanobis_metric(
    observations: Sequence[float],
    predictive_means: Sequence[float],
    covariance: Sequence[Sequence[float]],
) -> MahalanobisResult:
    """Compare observations with a Gaussian predictive summary.

    The covariance must include every variance the comparison should allow
    for: predictive (input) variance, measurement variance, and any declared
    discrepancy variance. The chi-square p-value is exact only if the
    predictive law is Gaussian with this covariance.

    Args:
        observations: Observed values ``y`` (length ``n >= 1``).
        predictive_means: Predictive means ``mu`` (length ``n``).
        covariance: Symmetric positive-definite ``n x n`` matrix ``C``.

    Returns:
        Residuals, standardized residuals, ``D^2``, and the chi-square tail.

    Raises:
        TypeError: If a value is not real.
        ValueError: If lengths disagree, a value is non-finite, or ``C`` is not
            symmetric positive definite.

    Equation:
        [EQ-VAL-MAHAL-001].

    References:
        [REF-MARDIA-KENT-BIBBY-1979, Chapters 1-3], [REF-NIST-DLMF-GAMMA].

    """
    observed = coerce_finite_sample(observations, name="observations")
    means = coerce_finite_sample(predictive_means, name="predictive_means")
    size = len(observed)
    if len(means) != size:
        raise ValueError("predictive_means must match observations in length")
    matrix: Matrix = tuple(
        tuple(require_finite_real(v, name="covariance") for v in row)
        for row in covariance
    )
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("covariance must be n x n")
    for i in range(size):
        for j in range(i):
            scale = max(abs(matrix[i][j]), abs(matrix[j][i]), 1.0)
            if abs(matrix[i][j] - matrix[j][i]) > 1.0e-12 * scale:
                raise ValueError("covariance must be symmetric")
    residuals = tuple(y - mu for y, mu in zip(observed, means, strict=True))
    diagonal = all(
        matrix[i][j] == 0.0 for i in range(size) for j in range(size) if i != j
    )
    if diagonal:
        if any(matrix[i][i] <= 0.0 for i in range(size)):
            raise ValueError("matrix must be positive definite")
        whitened = tuple(r / math.sqrt(matrix[i][i]) for i, r in enumerate(residuals))
    else:
        whitened = solve_lower(cholesky(matrix), residuals)
    squared = math.fsum(value * value for value in whitened)
    standardized = tuple(r / math.sqrt(matrix[i][i]) for i, r in enumerate(residuals))
    p_value = regularized_gamma_q(0.5 * size, 0.5 * squared)
    return MahalanobisResult(
        residuals, standardized, squared, size, min(1.0, max(0.0, p_value))
    )


# ---------------------------------------------------------------------------
# Reliability metric
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReliabilityMetricResult:
    """Probability that prediction and observation agree within a tolerance.

    Args:
        tolerance: Absolute agreement tolerance ``epsilon > 0``.
        estimate: ``r = P(|Y_pred - Y_obs| < epsilon)`` averaged over the
            empirical observation distribution.
        standard_error: Monte Carlo standard error of ``estimate`` conditional
            on the observations (predictive samples treated as independent).
        sample_size: Number of predictive samples.

    References:
        [REF-REBBA-MAHADEVAN-2008], [REF-LIU-CHEN-ARENDT-HUANG-2011].

    """

    tolerance: float
    estimate: float
    standard_error: float
    sample_size: int


def reliability_metric(
    predictions: Sequence[float],
    observations: Sequence[float],
    tolerance: float,
) -> ReliabilityMetricResult:
    """Estimate ``P(|Y_pred - Y_obs| < epsilon)`` from samples.

    Args:
        predictions: Predictive samples (at least two).
        observations: Observed values (at least one).
        tolerance: Positive agreement tolerance in QoI units.

    Returns:
        Estimate with a conditional Monte Carlo standard error.

    Raises:
        TypeError: If a value is not real.
        ValueError: If samples are too small, non-finite, or ``tolerance<=0``.

    Equation:
        [EQ-VAL-REL-001].

    References:
        [REF-REBBA-MAHADEVAN-2008].

    """
    predicted = coerce_finite_sample(predictions, name="predictions")
    observed = coerce_finite_sample(observations, name="observations")
    epsilon = require_finite_real(tolerance, name="tolerance")
    if epsilon <= 0.0:
        raise ValueError("tolerance must be positive")
    if len(predicted) < 2:
        raise ValueError("at least two predictive samples are required")
    count = len(observed)
    per_sample = [
        sum(1 for y in observed if abs(value - y) < epsilon) / count
        for value in predicted
    ]
    size = len(per_sample)
    mean = math.fsum(per_sample) / size
    variance = math.fsum((v - mean) ** 2 for v in per_sample) / (size - 1)
    return ReliabilityMetricResult(epsilon, mean, math.sqrt(variance / size), size)


# ---------------------------------------------------------------------------
# ASME V&V 20 comparison
# ---------------------------------------------------------------------------


def budget_standard_uncertainty(budget: NumericalErrorBudget) -> float:
    """Convert an absolute numerical error bound to a standard uncertainty.

    The bound ``B`` is treated as the half-width of a rectangular distribution
    (a GUM Type B evaluation), giving ``u_num = B / sqrt(3)``. This is an
    explicit modeling assumption, not a property of the bound.

    Args:
        budget: Verification error budget.

    Returns:
        Standard numerical uncertainty.

    Raises:
        TypeError: If ``budget`` is not a :class:`NumericalErrorBudget`.

    Equation:
        [EQ-VAL-ASME-002], [EQ-VV-003].

    References:
        [REF-JCGM-100-2008, Section 4.3.7], [REF-ASME-VV20-2009].

    """
    if not isinstance(budget, NumericalErrorBudget):
        raise TypeError("budget must be a NumericalErrorBudget")
    return budget.total_bound / math.sqrt(3.0)


@dataclass(frozen=True, slots=True)
class AsmeComparison:
    """ASME V&V 20 comparison error, validation uncertainty, and statement.

    Args:
        simulation: Simulation value ``S``.
        data: Experimental value ``D``.
        comparison_error: ``E = S - D``.
        numerical_uncertainty: ``u_num``.
        input_uncertainty: ``u_input``.
        data_uncertainty: ``u_D``.
        validation_uncertainty: ``u_val = sqrt(u_num^2 + u_input^2 + u_D^2)``.
        coverage_factor: ``k`` used for the model-error interval.
        model_error_interval: ``(E - k u_val, E + k u_val)``.
        numerical_uncertainty_source: How ``u_num`` was obtained.

    References:
        [REF-ASME-VV20-2009], [REF-OBERKAMPF-ROY-2025, Chapter 12].

    """

    simulation: float
    data: float
    comparison_error: float
    numerical_uncertainty: float
    input_uncertainty: float
    data_uncertainty: float
    validation_uncertainty: float
    coverage_factor: float
    model_error_interval: tuple[float, float]
    numerical_uncertainty_source: str

    @property
    def error_resolved(self) -> bool:
        """Return whether ``|E| > k u_val`` (model error sign is resolved).

        Equation:
            [EQ-VAL-ASME-001].

        References:
            [REF-ASME-VV20-2009].

        """
        return abs(self.comparison_error) > (
            self.coverage_factor * self.validation_uncertainty
        )

    def statement(self) -> str:
        """Return the V&V 20 interval statement for the model error.

        Returns:
            Text stating the interval for ``delta_model`` and whether the
            comparison error is resolved beyond the validation uncertainty.

        References:
            [REF-ASME-VV20-2009]. No equation beyond [EQ-VAL-ASME-001].

        """
        low, high = self.model_error_interval
        interval = (
            f"delta_model in [{low:.4g}, {high:.4g}] "
            f"(E = {self.comparison_error:.4g}, u_val = "
            f"{self.validation_uncertainty:.4g}, k = {self.coverage_factor:g}; "
            "assumes independent error sources)"
        )
        if self.error_resolved:
            return (
                interval + ": |E| > k u_val, so the model error is resolved and "
                f"approximately E = {self.comparison_error:.4g}"
            )
        return (
            interval + ": |E| <= k u_val, so the model error is not resolved "
            "beyond the validation uncertainty"
        )


def asme_validation_comparison(
    simulation: float,
    data: float,
    *,
    numerical_uncertainty: float | NumericalErrorBudget = 0.0,
    input_uncertainty: float = 0.0,
    data_uncertainty: float = 0.0,
    coverage_factor: float = 2.0,
) -> AsmeComparison:
    """Compute the ASME V&V 20 comparison error and validation uncertainty.

    V&V 20 treats validation as an *estimate of model error*,
    ``delta_model in [E - k u_val, E + k u_val]``, not as a pass/fail test.

    Args:
        simulation: Simulation value ``S``.
        data: Experimental value ``D``.
        numerical_uncertainty: Standard uncertainty ``u_num`` or a
            :class:`~probstat.uq.NumericalErrorBudget` converted by
            :func:`budget_standard_uncertainty`.
        input_uncertainty: Standard uncertainty ``u_input`` of ``S`` due to
            uncertain inputs (e.g. the predictive standard deviation).
        data_uncertainty: Standard uncertainty ``u_D`` of ``D``.
        coverage_factor: Positive factor ``k`` for the interval.

    Returns:
        Comparison record with interval statement.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If a value is non-finite, an uncertainty is negative, or
            ``k <= 0``.

    Equation:
        [EQ-VAL-ASME-001], [EQ-VAL-ASME-002].

    References:
        [REF-ASME-VV20-2009], [REF-OBERKAMPF-ROY-2025, Chapter 12].

    """
    s_value = require_finite_real(simulation, name="simulation")
    d_value = require_finite_real(data, name="data")
    if isinstance(numerical_uncertainty, NumericalErrorBudget):
        u_num = budget_standard_uncertainty(numerical_uncertainty)
        source = (
            f"NumericalErrorBudget bound {numerical_uncertainty.total_bound:.4g} "
            "/ sqrt(3) (rectangular Type B assumption)"
        )
    else:
        u_num = require_non_negative_finite_real(
            numerical_uncertainty, name="numerical_uncertainty"
        )
        source = "declared standard uncertainty"
    u_input = require_non_negative_finite_real(
        input_uncertainty, name="input_uncertainty"
    )
    u_data = require_non_negative_finite_real(data_uncertainty, name="data_uncertainty")
    k = require_finite_real(coverage_factor, name="coverage_factor")
    if k <= 0.0:
        raise ValueError("coverage_factor must be positive")
    error = s_value - d_value
    u_val = math.sqrt(math.fsum((u_num * u_num, u_input * u_input, u_data * u_data)))
    return AsmeComparison(
        s_value,
        d_value,
        error,
        u_num,
        u_input,
        u_data,
        u_val,
        k,
        (error - k * u_val, error + k * u_val),
        source,
    )


# ---------------------------------------------------------------------------
# Bayesian validation
# ---------------------------------------------------------------------------

_ALTERNATIVE_KINDS = ("uniform", "normal", "inflated")


@dataclass(frozen=True, slots=True)
class ValidationAlternative:
    """An explicitly declared alternative hypothesis for Bayesian validation.

    Use one of the constructors:

    * :meth:`uniform` — each observation is only known to lie in
      ``[lower, upper]``;
    * :meth:`normal` — observations follow ``N(mean, sd^2)`` independent of the
      model;
    * :meth:`inflated` — observations follow a normal law centered at the
      model's predictive mean with standard deviation ``factor`` times the
      predictive standard deviation of an observation.

    Bayes factors depend strongly on how broad the alternative is (the
    Jeffreys-Lindley effect); the alternative is therefore always reported.

    Args:
        kind: ``"uniform"``, ``"normal"``, or ``"inflated"``.
        first: Lower bound, mean, or inflation factor.
        second: Upper bound, standard deviation, or unused (``0``).
        description: Human-readable statement of the alternative.

    References:
        [REF-KASS-RAFTERY-1995], [REF-REBBA-MAHADEVAN-2008].

    """

    kind: str
    first: float
    second: float
    description: str

    def __post_init__(self) -> None:
        """Validate the alternative parameters.

        Raises:
            ValueError: If the kind is unknown or parameters are invalid.

        References:
            [EQ-VAL-BF-001]. No equation is evaluated.

        """
        if self.kind not in _ALTERNATIVE_KINDS:
            raise ValueError(f"kind must be one of {_ALTERNATIVE_KINDS}")
        first = require_finite_real(self.first, name="first")
        second = require_finite_real(self.second, name="second")
        if self.kind == "uniform" and not second > first:
            raise ValueError("uniform alternative needs upper > lower")
        if self.kind == "normal" and second <= 0.0:
            raise ValueError("normal alternative needs a positive standard deviation")
        if self.kind == "inflated" and first <= 1.0:
            raise ValueError("inflation factor must exceed one")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        object.__setattr__(self, "first", first)
        object.__setattr__(self, "second", second)

    @classmethod
    def uniform(cls, lower: float, upper: float) -> ValidationAlternative:
        """Return a uniform alternative on ``[lower, upper]``.

        Args:
            lower: Lower plausible bound.
            upper: Upper plausible bound (``> lower``).

        Returns:
            Declared alternative.

        References:
            [REF-KASS-RAFTERY-1995]. No equation is evaluated.

        """
        return cls(
            "uniform",
            lower,
            upper,
            f"each observation uniform on [{lower:.6g}, {upper:.6g}]",
        )

    @classmethod
    def normal(cls, mean: float, standard_deviation: float) -> ValidationAlternative:
        """Return a model-independent normal alternative.

        Args:
            mean: Alternative mean.
            standard_deviation: Positive alternative standard deviation.

        Returns:
            Declared alternative.

        References:
            [REF-KASS-RAFTERY-1995]. No equation is evaluated.

        """
        return cls(
            "normal",
            mean,
            standard_deviation,
            f"each observation N({mean:.6g}, {standard_deviation:.6g}^2)",
        )

    @classmethod
    def inflated(cls, factor: float = 5.0) -> ValidationAlternative:
        """Return a model-centered alternative with inflated spread.

        Args:
            factor: Inflation factor ``> 1`` applied to the predictive standard
                deviation of an observation.

        Returns:
            Declared alternative.

        References:
            [REF-KASS-RAFTERY-1995]. No equation is evaluated.

        """
        return cls(
            "inflated",
            factor,
            0.0,
            f"each observation N(predictive mean, ({factor:g} x predictive sd)^2)",
        )

    def log_density(self, value: float, *, center: float, scale: float) -> float:
        """Return the alternative log density of one observation.

        Args:
            value: Observation.
            center: Predictive mean (used only by ``"inflated"``).
            scale: Predictive standard deviation of an observation (used only
                by ``"inflated"``; must be positive there).

        Returns:
            Log density (``-inf`` outside a uniform support).

        Raises:
            ValueError: If ``scale <= 0`` for an inflated alternative.

        Equation:
            [EQ-VAL-BF-001].

        References:
            [REF-KASS-RAFTERY-1995].

        """
        if self.kind == "uniform":
            if self.first <= value <= self.second:
                return -math.log(self.second - self.first)
            return -math.inf
        if self.kind == "normal":
            return _normal_log_pdf(value, self.first, self.second)
        if scale <= 0.0:
            raise ValueError("inflated alternative needs a positive predictive scale")
        return _normal_log_pdf(value, center, self.first * scale)


@dataclass(frozen=True, slots=True)
class BayesFactorResult:
    """Bayes factor of the model predictive density against an alternative.

    Args:
        log_evidence_model: ``sum_i log p_M(y_i)``.
        log_evidence_alternative: ``sum_i log p_A(y_i)``.
        log_bayes_factor: ``log B = log p_M - log p_A`` (positive favors the
            model's predictive density).
        kernel_standard_deviation: Gaussian kernel width used to turn
            predictive samples into a density.
        kernel_source: ``"measurement"`` or ``"silverman"``.
        alternative: Declared alternative.

    References:
        [REF-KASS-RAFTERY-1995], [REF-REBBA-MAHADEVAN-2008].

    """

    log_evidence_model: float
    log_evidence_alternative: float
    log_bayes_factor: float
    kernel_standard_deviation: float
    kernel_source: str
    alternative: ValidationAlternative

    @property
    def bayes_factor(self) -> float:
        """Return ``B = exp(log B)`` (``inf``/``0`` on overflow/underflow).

        Equation:
            [EQ-VAL-BF-001].

        References:
            [REF-KASS-RAFTERY-1995].

        """
        if self.log_bayes_factor > 700.0:
            return math.inf
        return math.exp(self.log_bayes_factor)

    @property
    def interpretation(self) -> str:
        """Return the Kass-Raftery verbal category of ``2 log B``.

        References:
            [REF-KASS-RAFTERY-1995, Section 3.2].

        """
        return interpret_log_bayes_factor(self.log_bayes_factor)


def interpret_log_bayes_factor(log_bayes_factor: float) -> str:
    """Map ``log B`` to the Kass-Raftery evidence scale on ``2 log B``.

    Args:
        log_bayes_factor: Natural-log Bayes factor (model vs alternative).

    Returns:
        Text naming the favored hypothesis and the evidence category.

    References:
        [REF-KASS-RAFTERY-1995, Section 3.2]. No equation is evaluated.

    """
    if math.isnan(log_bayes_factor):
        return "undefined"
    favored = "model" if log_bayes_factor >= 0.0 else "alternative"
    magnitude = 2.0 * abs(log_bayes_factor)
    if magnitude < 2.0:
        category = "not worth more than a bare mention"
    elif magnitude < 6.0:
        category = "positive"
    elif magnitude < 10.0:
        category = "strong"
    else:
        category = "very strong"
    return f"{category} evidence for the {favored}"


def silverman_bandwidth(samples: Sequence[float]) -> float:
    """Return Silverman's rule-of-thumb Gaussian kernel bandwidth.

    Args:
        samples: At least two finite samples.

    Returns:
        ``0.9 min(s, IQR/1.34) n^(-1/5)`` (falls back to ``s`` if the IQR is
        zero); zero for a degenerate sample.

    Raises:
        ValueError: If fewer than two samples are given.

    Equation:
        [EQ-VAL-BF-001].

    References:
        [REF-SILVERMAN-1986, Section 3.4.2].

    """
    values = sorted(coerce_finite_sample(samples, name="samples"))
    size = len(values)
    if size < 2:
        raise ValueError("at least two samples are required")
    mean = math.fsum(values) / size
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (size - 1))
    iqr = quantile(values, 0.75) - quantile(values, 0.25)
    spread = min(sd, iqr / 1.34) if iqr > 0.0 else sd
    return 0.9 * spread * math.pow(size, -0.2)


def validation_bayes_factor(
    predictions: Sequence[float],
    observations: Sequence[float],
    *,
    alternative: ValidationAlternative,
    noise_standard_deviation: float = 0.0,
) -> BayesFactorResult:
    """Compute a Bayes factor of model predictive density vs an alternative.

    The model predictive density of one observation is the Gaussian-kernel
    mixture ``p_M(y) = (1/S) sum_s N(y; y_s, h^2)`` over predictive samples
    ``y_s``. With known Gaussian measurement noise ``h = sigma_m`` this is the
    Monte Carlo estimate of the exact predictive density of a noisy
    observation; with ``sigma_m = 0`` the Silverman bandwidth is used and the
    result is a kernel-density approximation. Observations are treated as
    conditionally independent given the predictive law.

    Args:
        predictions: Predictive samples of the latent QoI (at least two).
        observations: Observed values (at least one).
        alternative: Explicit alternative hypothesis.
        noise_standard_deviation: Measurement standard deviation ``>= 0``.

    Returns:
        Log evidences, log Bayes factor, kernel width, and alternative.

    Raises:
        TypeError: If ``alternative`` has the wrong type or a value is not real.
        ValueError: If samples are too small or degenerate with zero noise.

    Equation:
        [EQ-VAL-BF-001].

    References:
        [REF-KASS-RAFTERY-1995], [REF-REBBA-MAHADEVAN-2008],
        [REF-SILVERMAN-1986, Section 3.4.2].

    """
    if not isinstance(alternative, ValidationAlternative):
        raise TypeError("alternative must be a ValidationAlternative")
    predicted = coerce_finite_sample(predictions, name="predictions")
    observed = coerce_finite_sample(observations, name="observations")
    if len(predicted) < 2:
        raise ValueError("at least two predictive samples are required")
    sigma = require_non_negative_finite_real(
        noise_standard_deviation, name="noise_standard_deviation"
    )
    if sigma > 0.0:
        kernel, source = sigma, "measurement"
    else:
        kernel, source = silverman_bandwidth(predicted), "silverman"
        if kernel <= 0.0:
            raise ValueError(
                "predictive samples are degenerate and no measurement noise is given"
            )
    size = len(predicted)
    center = math.fsum(predicted) / size
    latent_var = math.fsum((v - center) ** 2 for v in predicted) / (size - 1)
    scale = math.sqrt(latent_var + kernel * kernel)
    log_model: list[float] = []
    log_alt: list[float] = []
    for y in observed:
        terms = [_normal_log_pdf(y, value, kernel) for value in predicted]
        log_model.append(_log_sum_exp(terms) - math.log(size))
        log_alt.append(alternative.log_density(y, center=center, scale=scale))
    evidence_model = math.fsum(log_model)
    evidence_alt = (
        -math.inf if any(v == -math.inf for v in log_alt) else math.fsum(log_alt)
    )
    return BayesFactorResult(
        evidence_model,
        evidence_alt,
        evidence_model - evidence_alt,
        kernel,
        source,
        alternative,
    )


def _normal_log_pdf(value: float, mean: float, sd: float) -> float:
    """Return the normal log density.

    Args:
        value: Evaluation point.
        mean: Mean.
        sd: Positive standard deviation.

    Returns:
        ``log N(value; mean, sd^2)``.

    References:
        [REF-CASELLA-BERGER-2002, Section 3.3]. No named equation is needed.

    """
    z = (value - mean) / sd
    return -0.5 * z * z - math.log(sd) - _LOG_SQRT_TWO_PI


def _log_sum_exp(values: Sequence[float]) -> float:
    """Return ``log(sum(exp(v)))`` without overflow.

    Args:
        values: Non-empty log terms.

    Returns:
        Stable log-sum-exp.

    References:
        [REF-ROBERT-CASELLA-2004]. No named equation is needed.

    """
    peak = max(values)
    if peak == -math.inf:
        return -math.inf
    return peak + math.log(math.fsum(math.exp(v - peak) for v in values))
