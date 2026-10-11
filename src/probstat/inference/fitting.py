"""Generic maximum-likelihood fitting, calibrated goodness of fit, and selection.

A :class:`DistributionFamily` describes a parametric family by its constructor,
its parameter kinds (``location``, ``positive``, ``real``), a starting rule, and
an optional closed-form maximum-likelihood estimator. :func:`fit_distribution`
maximizes the log-likelihood built from the family's ``log_pdf`` with BFGS in
unconstrained coordinates and reports observed-information standard errors.
:func:`bootstrap_goodness_of_fit` calibrates Kolmogorov-Smirnov and
Anderson-Darling statistics when parameters are estimated from the tested data
(the Lilliefors/Stephens problem) by a seeded parametric bootstrap that refits
every replicate. :func:`select_distribution` ranks candidate families by
information criteria, Akaike weights, and calibrated goodness of fit, and
warns when even the best model is rejected.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from probstat.distributions import (
    Beta,
    Exponential,
    Gamma,
    GeneralizedExtremeValue,
    Gumbel,
    Logistic,
    LogNormal,
    Normal,
    StudentT,
    Uniform,
    Weibull,
)
from probstat.foundations import coerce_finite_sample
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_non_negative_integer,
    require_probability,
)
from probstat.linalg import Matrix, Vector, inverse_spd
from probstat.optimization import (
    OptimizationResult,
    central_hessian,
    minimize_bfgs,
    minimize_nelder_mead,
)
from probstat.statistics import median, median_absolute_deviation

from .estimation import (
    fit_gamma_mle,
    fit_lognormal_mle,
    fit_normal_mle,
    fit_weibull_mle,
)
from .goodness_of_fit import anderson_darling_test, kolmogorov_smirnov_test

ParameterKind = Literal["location", "positive", "real"]
SampleDomain = Literal["real", "positive", "nonnegative", "unit"]
FitMethod = Literal["auto", "closed-form", "numerical"]
Criterion = Literal["aic", "aicc", "bic"]

_PARAMETER_KINDS = ("location", "positive", "real")
_SAMPLE_DOMAINS = ("real", "positive", "nonnegative", "unit")
_FIT_METHODS = ("auto", "closed-form", "numerical")
_CRITERIA = ("aic", "aicc", "bic")
_EULER_GAMMA = 0.5772156649015329
_FIT_FAILURES = (ValueError, ArithmeticError)
STUDENT_T_MAX_NU = 1000.0
"""Upper bound on the fitted Student-t degrees of freedom (normal-like limit)."""


class FittableDistribution(Protocol):
    """Interface a fitted law must expose for likelihood and diagnostics.

    References:
        [REF-CASELLA-BERGER-2002, Section 7.2].

    """

    def log_pdf(self, x: float) -> float:
        """Return the natural log-density at ``x`` (``-inf`` off support).

        References:
            [REF-CASELLA-BERGER-2002, Section 7.2]. No equation is evaluated.

        """
        ...

    def cdf(self, x: float) -> float:
        """Return the cumulative probability through ``x``.

        References:
            [REF-ROSS-2019, Chapter 5]. No equation is evaluated.

        """
        ...

    def quantile(self, probability: float) -> float:
        """Return the generalized inverse CDF at ``probability``.

        References:
            [REF-ROSS-2019, Chapter 5]. No equation is evaluated.

        """
        ...

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[float, ...]:
        """Draw reproducible observations with a call-local seed.

        References:
            [REF-DEVROYE-1986, Chapter 2]. No equation is evaluated.

        """
        ...


class ModelSelectionWarning(UserWarning):
    """Raised when model selection finds no adequate or no converged candidate.

    References:
        [REF-BURNHAM-ANDERSON-2002, Chapter 2].

    """


@dataclass(frozen=True)
class DistributionFamily:
    """A parametric family prepared for generic maximum-likelihood fitting.

    Args:
        name: Non-empty family label used in reports.
        parameter_names: Constructor parameter names in positional order.
        parameter_kinds: ``"location"`` (real, scaled by the data spread),
            ``"positive"`` (optimized on the log scale), or ``"real"``
            (dimensionless real, e.g. a GEV shape) for each parameter.
        build: Constructor mapping a natural parameter vector to a law.
        initial: Rule mapping the validated sample to natural start values.
        domain: Admissible sample domain: ``"real"``, ``"positive"``
            (``x > 0``), ``"nonnegative"`` (``x >= 0``), or ``"unit"``
            (``0 < x < 1``).
        closed_form: Optional exact MLE mapping a sample to natural values.
        regular: ``False`` for families whose support depends on parameters
            (e.g. Uniform); observed-information errors are then undefined.
        upper_bounds: Optional per-parameter upper bounds (``None`` entries
            for unbounded parameters). Only ``positive`` parameters may be
            bounded; they are then mapped by a logistic transform onto
            ``(0, bound)``. Used to stop a Student-t ``nu`` from drifting to
            infinity on light-tailed data.

    References:
        [REF-CASELLA-BERGER-2002, Sections 7.2 and 10.1].

    """

    name: str
    parameter_names: tuple[str, ...]
    parameter_kinds: tuple[ParameterKind, ...]
    build: Callable[[Vector], FittableDistribution]
    initial: Callable[[Vector], Vector]
    domain: SampleDomain = "real"
    closed_form: Callable[[Vector], Vector] | None = None
    regular: bool = True
    upper_bounds: tuple[float | None, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the family description.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If names are empty or repeated, lengths disagree, or a
                kind or domain label is unknown.

        References:
            Project validation policy. No equation is evaluated.

        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("family name must be a non-empty string")
        names = tuple(self.parameter_names)
        kinds = tuple(self.parameter_kinds)
        if not names:
            raise ValueError("a family needs at least one parameter")
        if len(names) != len(kinds):
            raise ValueError("parameter_names and parameter_kinds lengths differ")
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be distinct")
        for kind in kinds:
            if kind not in _PARAMETER_KINDS:
                raise ValueError(f"unknown parameter kind {kind!r}")
        if self.domain not in _SAMPLE_DOMAINS:
            raise ValueError(f"unknown sample domain {self.domain!r}")
        if not callable(self.build) or not callable(self.initial):
            raise TypeError("build and initial must be callable")
        if self.closed_form is not None and not callable(self.closed_form):
            raise TypeError("closed_form must be callable or None")
        if self.upper_bounds is not None:
            bounds = tuple(self.upper_bounds)
            if len(bounds) != len(names):
                raise ValueError("upper_bounds length must match the parameters")
            for kind, bound in zip(kinds, bounds, strict=True):
                if bound is None:
                    continue
                if kind != "positive":
                    raise ValueError("only positive parameters may be bounded")
                if not (math.isfinite(bound) and bound > 0.0):
                    raise ValueError("upper bounds must be finite and positive")
            object.__setattr__(self, "upper_bounds", bounds)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "parameter_kinds", kinds)

    def bound(self, index: int) -> float | None:
        """Return the upper bound of parameter ``index``, if any.

        Args:
            index: Zero-based parameter position.

        Returns:
            Finite positive bound or ``None``.

        References:
            Project API policy. No equation is evaluated.

        """
        if self.upper_bounds is None:
            return None
        return self.upper_bounds[index]

    def admits(self, sample: Sequence[float]) -> bool:
        """Return whether every observation lies in the family's sample domain.

        Args:
            sample: Finite observations.

        Returns:
            ``True`` when the domain condition holds for all observations.

        References:
            Project validation policy. No equation is evaluated.

        """
        if self.domain == "positive":
            return all(value > 0.0 for value in sample)
        if self.domain == "nonnegative":
            return all(value >= 0.0 for value in sample)
        if self.domain == "unit":
            return all(0.0 < value < 1.0 for value in sample)
        return True


@dataclass(frozen=True)
class FitResult:
    """Maximum-likelihood fit of one family with information criteria.

    Args:
        family: Family label.
        parameter_names: Parameter names in constructor order.
        estimates: Maximum-likelihood estimates (natural parameters).
        standard_errors: Observed-information standard errors (``nan`` when
            undefined, e.g. non-regular families or a singular Hessian).
        covariance: Asymptotic covariance of the estimates, or ``None``.
        log_likelihood: Maximized log-likelihood.
        sample_size: Number of observations ``n``.
        aic: Akaike information criterion ``2k - 2 log L``.
        aicc: Small-sample corrected AIC (``inf`` when ``n <= k + 1``).
        bic: Bayesian (Schwarz) information criterion ``k log n - 2 log L``.
        converged: Whether the optimizer met a stopping tolerance.
        method: ``"closed-form"``, ``"bfgs"``, or ``"nelder-mead+bfgs"``.
        message: Termination or error-estimate note.
        distribution: Fitted law.

    References:
        [REF-AKAIKE-1974], [REF-SCHWARZ-1978], [REF-HURVICH-TSAI-1989],
        [REF-CASELLA-BERGER-2002, Section 10.1].

    """

    family: str
    parameter_names: tuple[str, ...]
    estimates: Vector
    standard_errors: Vector
    covariance: Matrix | None
    log_likelihood: float
    sample_size: int
    aic: float
    aicc: float
    bic: float
    converged: bool
    method: str
    message: str
    distribution: FittableDistribution

    @property
    def parameter_count(self) -> int:
        """Return the number of estimated parameters ``k``.

        References:
            [REF-AKAIKE-1974]. No equation is evaluated.

        """
        return len(self.estimates)

    @property
    def params(self) -> dict[str, float]:
        """Return estimates keyed by parameter name.

        References:
            [REF-CASELLA-BERGER-2002, Section 7.2]. No equation is evaluated.

        """
        return dict(zip(self.parameter_names, self.estimates, strict=True))

    @property
    def standard_error_map(self) -> dict[str, float]:
        """Return standard errors keyed by parameter name.

        References:
            [REF-CASELLA-BERGER-2002, Section 10.1]. No equation is evaluated.

        """
        return dict(zip(self.parameter_names, self.standard_errors, strict=True))


@dataclass(frozen=True, slots=True)
class BootstrapGoodnessOfFit:
    """Parametric-bootstrap KS and AD p-values with estimated parameters.

    Args:
        family: Family label of the fitted null model.
        ks_statistic: Observed Kolmogorov-Smirnov distance to the fitted law.
        ks_p_value: Bootstrap p-value ``(1 + #{D* >= D}) / (1 + refits)``.
        ad_statistic: Observed Anderson-Darling statistic (may be ``inf``).
        ad_p_value: Bootstrap p-value, ``nan`` when the observed statistic is
            infinite (an observation sits on a fitted support boundary).
        n_bootstrap: Requested number of bootstrap replicates.
        refits: Replicates whose refit succeeded and entered the p-values.
        failed_refits: Replicates whose refit failed and were excluded.
        sample_size: Number of observations.
        method: Calibration label.

    References:
        [REF-STEPHENS-1974], [REF-LILLIEFORS-1967],
        [REF-DAVISON-HINKLEY-1997, Chapter 4].

    """

    family: str
    ks_statistic: float
    ks_p_value: float
    ad_statistic: float
    ad_p_value: float
    n_bootstrap: int
    refits: int
    failed_refits: int
    sample_size: int
    method: str = "parametric bootstrap with refitting on every replicate"

    def rejected(self, alpha: float = 0.05) -> bool:
        """Return whether either calibrated p-value falls below ``alpha``.

        Args:
            alpha: Significance level in ``(0, 1)``.

        Returns:
            ``True`` when the KS or (defined) AD p-value is below ``alpha``.

        Raises:
            ValueError: If ``alpha`` is not strictly inside ``(0, 1)``.

        Equation:
            [EQ-INF-GOF-PB-001].

        References:
            [REF-DAVISON-HINKLEY-1997, Chapter 4].

        """
        level = require_probability(alpha, name="alpha")
        if not 0.0 < level < 1.0:
            raise ValueError("alpha must lie strictly inside (0, 1)")
        ad_rejects = not math.isnan(self.ad_p_value) and self.ad_p_value < level
        return self.ks_p_value < level or ad_rejects


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    """One ranked candidate in a model-selection study.

    Args:
        rank: One-based rank by the selection criterion.
        fit: Maximum-likelihood fit of the candidate.
        delta_aic: ``AIC_i - min AIC``.
        akaike_weight: ``exp(-delta_aic/2)`` normalized over candidates.
        delta_bic: ``BIC_i - min BIC``.
        bic_weight: ``exp(-delta_bic/2)`` normalized over candidates.
        goodness_of_fit: Bootstrap-calibrated GOF, or ``None`` if not run.

    References:
        [REF-BURNHAM-ANDERSON-2002, Chapter 2].

    """

    rank: int
    fit: FitResult
    delta_aic: float
    akaike_weight: float
    delta_bic: float
    bic_weight: float
    goodness_of_fit: BootstrapGoodnessOfFit | None


@dataclass(frozen=True, slots=True)
class ModelSelectionResult:
    """Ranked candidate families with weights, calibrated GOF, and warnings.

    Args:
        criterion: Ranking criterion: ``"aic"``, ``"aicc"``, or ``"bic"``.
        candidates: Successfully fitted candidates in rank order.
        failures: ``(family, reason)`` pairs for candidates that could not be
            fitted or whose domain excludes the sample.
        warnings: Human-readable warnings, also emitted as
            :class:`ModelSelectionWarning`.
        alpha: Significance level used for GOF rejection warnings.
        sample_size: Number of observations.

    References:
        [REF-BURNHAM-ANDERSON-2002, Chapter 2], [REF-AKAIKE-1974].

    """

    criterion: str
    candidates: tuple[CandidateAssessment, ...]
    failures: tuple[tuple[str, str], ...]
    warnings: tuple[str, ...]
    alpha: float
    sample_size: int

    @property
    def best(self) -> CandidateAssessment:
        """Return the top-ranked candidate.

        References:
            [REF-BURNHAM-ANDERSON-2002, Chapter 2]. No equation is evaluated.

        """
        return self.candidates[0]

    def assessment(self, family: str) -> CandidateAssessment:
        """Return the assessment of a named family.

        Args:
            family: Family label.

        Returns:
            The matching candidate assessment.

        Raises:
            KeyError: If the family was not successfully fitted.

        References:
            Project API policy. No equation is evaluated.

        """
        for candidate in self.candidates:
            if candidate.fit.family == family:
                return candidate
        raise KeyError(f"no fitted candidate named {family!r}")


# ---------------------------------------------------------------------------
# Starting values and closed forms for the ready-made families
# ---------------------------------------------------------------------------


def _moments(sample: Vector) -> tuple[float, float]:
    """Return the sample mean and population standard deviation.

    References:
        [REF-NIST-EDA, Section 1.3.5]. No new equation is evaluated.

    """
    count = len(sample)
    center = math.fsum(sample) / count
    spread = math.sqrt(math.fsum((x - center) ** 2 for x in sample) / count)
    return center, spread


def _positive_spread(sample: Vector) -> float:
    """Return a strictly positive robust-or-moment scale for start values.

    References:
        [REF-HAMPEL-1974]. No new equation is evaluated.

    """
    _, spread = _moments(sample)
    if spread > 0.0:
        return spread
    return max(abs(sample[0]), 1.0)


def _normal_closed_form(sample: Vector) -> Vector:
    """Return the closed-form normal MLE ``(mu, sigma)``.

    References:
        [EQ-INF-MLE-NORM-001], [REF-CASELLA-BERGER-2002, Section 7.2].

    """
    law = fit_normal_mle(sample)
    return (law.mu, law.sigma)


def _lognormal_closed_form(sample: Vector) -> Vector:
    """Return the closed-form lognormal MLE ``(mu, sigma)`` of ``log X``.

    References:
        [EQ-INF-MLE-LN-001], [REF-NIST-DISTRIBUTIONS].

    """
    law = fit_lognormal_mle(sample)
    return (law.mu, law.sigma)


def _exponential_closed_form(sample: Vector) -> Vector:
    """Return the closed-form exponential rate MLE ``1 / mean``.

    References:
        [EQ-INF-MLE-EXP-001], [REF-CASELLA-BERGER-2002, Section 7.2].

    """
    center = math.fsum(sample) / len(sample)
    if not center > 0.0:
        raise ValueError("exponential MLE requires a positive sample mean")
    return (1.0 / center,)


def _gamma_closed_form(sample: Vector) -> Vector:
    """Return the profile-likelihood gamma MLE ``(shape, rate)``.

    References:
        [EQ-INF-MLE-GAM-001], [REF-JKB-1994, Chapter 17].

    """
    law = fit_gamma_mle(sample)
    return (law.shape, law.rate)


def _weibull_closed_form(sample: Vector) -> Vector:
    """Return the profile-likelihood Weibull MLE ``(shape, scale)``.

    References:
        [EQ-INF-MLE-WEIB-001], [REF-JKB-1994, Chapter 21].

    """
    law = fit_weibull_mle(sample)
    return (law.shape, law.scale)


def _uniform_closed_form(sample: Vector) -> Vector:
    """Return the uniform MLE ``(min x, max x)``.

    References:
        [EQ-INF-MLE-UNIF-001], [REF-CASELLA-BERGER-2002, Section 7.2].

    """
    lower, upper = min(sample), max(sample)
    if not upper > lower:
        raise ValueError("uniform MLE requires at least two distinct values")
    return (lower, upper)


def _normal_start(sample: Vector) -> Vector:
    """Return moment start values for the normal family.

    References:
        [REF-CASELLA-BERGER-2002, Section 7.2]. No new equation is evaluated.

    """
    center, _ = _moments(sample)
    return (center, _positive_spread(sample))


def _lognormal_start(sample: Vector) -> Vector:
    """Return log-moment start values for the lognormal family.

    References:
        [REF-NIST-DISTRIBUTIONS]. No new equation is evaluated.

    """
    logs = tuple(math.log(x) for x in sample)
    center, _ = _moments(logs)
    return (center, _positive_spread(logs))


def _exponential_start(sample: Vector) -> Vector:
    """Return the reciprocal-mean start for the exponential rate.

    References:
        [REF-CASELLA-BERGER-2002, Section 7.2]. No new equation is evaluated.

    """
    center, _ = _moments(sample)
    return (1.0 / center if center > 0.0 else 1.0,)


def _gamma_start(sample: Vector) -> Vector:
    """Return method-of-moments start values ``(m^2/v, m/v)`` for the gamma.

    References:
        [REF-JKB-1994, Chapter 17]. No new equation is evaluated.

    """
    center, spread = _moments(sample)
    variance = max(spread * spread, 1e-12 * center * center)
    return (center * center / variance, center / variance)


def _weibull_start(sample: Vector) -> Vector:
    """Return Gumbel-moment start values for the Weibull law of ``log X``.

    References:
        [REF-JKB-1994, Chapter 21]. No new equation is evaluated.

    """
    logs = tuple(math.log(x) for x in sample)
    center, spread = _moments(logs)
    shape = math.pi / (math.sqrt(6.0) * spread) if spread > 0.0 else 1.0
    return (shape, math.exp(center + _EULER_GAMMA / shape))


def _gumbel_start(sample: Vector) -> Vector:
    """Return moment start values ``beta = sqrt(6) s / pi``, ``mu = m - gamma beta``.

    References:
        [REF-COLES-2001, Section 3.1]. No new equation is evaluated.

    """
    center, _ = _moments(sample)
    beta = math.sqrt(6.0) * _positive_spread(sample) / math.pi
    return (center - _EULER_GAMMA * beta, beta)


def _gev_start(sample: Vector) -> Vector:
    """Return Gumbel-moment start values with zero shape for the GEV.

    References:
        [REF-COLES-2001, Section 3.3]. No new equation is evaluated.

    """
    mu, beta = _gumbel_start(sample)
    return (mu, beta, 0.0)


def _logistic_start(sample: Vector) -> Vector:
    """Return moment start values ``s = sqrt(3) sd / pi`` for the logistic.

    References:
        [REF-JKB-1995, Chapter 23]. No new equation is evaluated.

    """
    center, _ = _moments(sample)
    return (center, math.sqrt(3.0) * _positive_spread(sample) / math.pi)


def _student_t_start(sample: Vector) -> Vector:
    """Return robust start values ``(nu=5, median, MAD)`` for the Student t.

    References:
        [REF-LANGE-LITTLE-TAYLOR-1989], [REF-HAMPEL-1974].

    """
    scale = median_absolute_deviation(sample) if len(sample) > 1 else 0.0
    if not scale > 0.0:
        scale = _positive_spread(sample)
    return (5.0, median(sample), scale)


def _beta_start(sample: Vector) -> Vector:
    """Return method-of-moments start values for the beta family.

    References:
        [REF-JKB-1995, Chapter 25]. No new equation is evaluated.

    """
    center, spread = _moments(sample)
    variance = spread * spread
    common = center * (1.0 - center) / variance - 1.0 if variance > 0.0 else 0.0
    if not common > 0.0:
        return (1.0, 1.0)
    return (center * common, (1.0 - center) * common)


def _uniform_start(sample: Vector) -> Vector:
    """Return the sample range as uniform start values.

    References:
        [REF-CASELLA-BERGER-2002, Section 7.2]. No new equation is evaluated.

    """
    return _uniform_closed_form(sample)


NORMAL_FAMILY = DistributionFamily(
    "Normal",
    ("mu", "sigma"),
    ("location", "positive"),
    lambda p: Normal(p[0], p[1]),
    _normal_start,
    "real",
    _normal_closed_form,
)
LOGNORMAL_FAMILY = DistributionFamily(
    "LogNormal",
    ("mu", "sigma"),
    ("real", "positive"),
    lambda p: LogNormal(p[0], p[1]),
    _lognormal_start,
    "positive",
    _lognormal_closed_form,
)
EXPONENTIAL_FAMILY = DistributionFamily(
    "Exponential",
    ("rate",),
    ("positive",),
    lambda p: Exponential(p[0]),
    _exponential_start,
    "nonnegative",
    _exponential_closed_form,
)
GAMMA_FAMILY = DistributionFamily(
    "Gamma",
    ("shape", "rate"),
    ("positive", "positive"),
    lambda p: Gamma(p[0], p[1]),
    _gamma_start,
    "positive",
    _gamma_closed_form,
)
WEIBULL_FAMILY = DistributionFamily(
    "Weibull",
    ("shape", "scale"),
    ("positive", "positive"),
    lambda p: Weibull(p[0], p[1]),
    _weibull_start,
    "positive",
    _weibull_closed_form,
)
GUMBEL_FAMILY = DistributionFamily(
    "Gumbel",
    ("mu", "beta"),
    ("location", "positive"),
    lambda p: Gumbel(p[0], p[1]),
    _gumbel_start,
)
GEV_FAMILY = DistributionFamily(
    "GEV",
    ("mu", "sigma", "xi"),
    ("location", "positive", "real"),
    lambda p: GeneralizedExtremeValue(p[0], p[1], p[2]),
    _gev_start,
)
LOGISTIC_FAMILY = DistributionFamily(
    "Logistic",
    ("mu", "s"),
    ("location", "positive"),
    lambda p: Logistic(p[0], p[1]),
    _logistic_start,
)
STUDENT_T_FAMILY = DistributionFamily(
    "StudentT",
    ("nu", "mu", "sigma"),
    ("positive", "location", "positive"),
    lambda p: StudentT(p[0], p[1], p[2]),
    _student_t_start,
    upper_bounds=(STUDENT_T_MAX_NU, None, None),
)
BETA_FAMILY = DistributionFamily(
    "Beta",
    ("alpha", "beta"),
    ("positive", "positive"),
    lambda p: Beta(p[0], p[1]),
    _beta_start,
    "unit",
)
UNIFORM_FAMILY = DistributionFamily(
    "Uniform",
    ("lower", "upper"),
    ("location", "location"),
    lambda p: Uniform(p[0], p[1]),
    _uniform_start,
    "real",
    _uniform_closed_form,
    regular=False,
)

STANDARD_FAMILIES: tuple[DistributionFamily, ...] = (
    NORMAL_FAMILY,
    LOGNORMAL_FAMILY,
    EXPONENTIAL_FAMILY,
    GAMMA_FAMILY,
    WEIBULL_FAMILY,
    GUMBEL_FAMILY,
    GEV_FAMILY,
    LOGISTIC_FAMILY,
    STUDENT_T_FAMILY,
    BETA_FAMILY,
    UNIFORM_FAMILY,
)


def family_by_name(name: str) -> DistributionFamily:
    """Return a ready-made family by its label (case-insensitive).

    Args:
        name: One of ``Normal``, ``LogNormal``, ``Exponential``, ``Gamma``,
            ``Weibull``, ``Gumbel``, ``GEV``, ``Logistic``, ``StudentT``,
            ``Beta``, or ``Uniform``.

    Returns:
        The matching :class:`DistributionFamily`.

    Raises:
        TypeError: If ``name`` is not a string.
        KeyError: If no ready-made family has that label.

    References:
        Project API policy. No equation is evaluated.

    """
    if not isinstance(name, str):
        raise TypeError("family name must be a string")
    for family in STANDARD_FAMILIES:
        if family.name.lower() == name.strip().lower():
            return family
    known = ", ".join(family.name for family in STANDARD_FAMILIES)
    raise KeyError(f"unknown family {name!r}; known families: {known}")


# ---------------------------------------------------------------------------
# Likelihood, reparameterization, and observed information
# ---------------------------------------------------------------------------


def _negative_log_likelihood(
    family: DistributionFamily, sample: Vector, parameters: Vector
) -> float:
    """Return ``-sum log f(x_i; theta)``, or ``inf`` for invalid parameters.

    References:
        [EQ-INF-MLE-GEN-001], [REF-CASELLA-BERGER-2002, Section 7.2].

    """
    try:
        law = family.build(parameters)
        total = math.fsum(law.log_pdf(x) for x in sample)
    except _FIT_FAILURES:
        return math.inf
    if math.isnan(total) or total == math.inf:
        return math.inf
    return -total


def _reference_scales(
    family: DistributionFamily, anchor: Vector, spread: float
) -> Vector:
    """Return per-parameter scales of the unconstrained coordinates.

    ``location`` parameters move in units of the data spread, ``real``
    parameters in unit steps, and ``positive`` parameters multiplicatively.

    References:
        [EQ-INF-MLE-GEN-001], [REF-NOCEDAL-WRIGHT-2006, Section 2.2].

    """
    return tuple(
        spread if kind == "location" else (value if kind == "positive" else 1.0)
        for kind, value in zip(family.parameter_kinds, anchor, strict=True)
    )


def _to_natural(
    family: DistributionFamily, anchor: Vector, scales: Vector, u: Vector
) -> Vector:
    """Map unconstrained coordinates ``u`` to natural parameters.

    ``theta = a + s u`` for location/real parameters, ``theta = a exp(u)``
    for positive parameters, and ``theta = b / (1 + (b/a - 1) exp(-u))`` for
    positive parameters bounded above by ``b``, with anchor ``a`` and scale
    ``s``; ``u = 0`` maps to the anchor.

    References:
        [EQ-INF-MLE-GEN-001], [REF-CASELLA-BERGER-2002, Section 7.2].

    """
    natural: list[float] = []
    for index, (kind, a, s, coordinate) in enumerate(
        zip(family.parameter_kinds, anchor, scales, u, strict=True)
    ):
        bound = family.bound(index)
        if kind == "positive" and bound is not None:
            odds = (bound / a - 1.0) * math.exp(-max(min(coordinate, 700.0), -700.0))
            natural.append(bound / (1.0 + odds))
        elif kind == "positive":
            natural.append(a * math.exp(min(coordinate, 700.0)))
        else:
            natural.append(a + s * coordinate)
    return tuple(natural)


def _check_anchor(family: DistributionFamily, anchor: Vector) -> Vector:
    """Validate natural anchor values for the reparameterization.

    Raises:
        ValueError: If a value is non-finite or a positive parameter is not
            strictly positive.

    References:
        Project validation policy. No equation is evaluated.

    """
    values = tuple(float(value) for value in anchor)
    if len(values) != len(family.parameter_names):
        raise ValueError(f"{family.name}: start has the wrong number of values")
    for name, kind, value in zip(
        family.parameter_names, family.parameter_kinds, values, strict=True
    ):
        if not math.isfinite(value):
            raise ValueError(f"{family.name}: non-finite value for {name}")
        if kind == "positive" and not value > 0.0:
            raise ValueError(f"{family.name}: {name} must be positive")
    return tuple(
        min(value, (1.0 - 1e-9) * bound) if bound is not None else value
        for value, bound in (
            (value, family.bound(index)) for index, value in enumerate(values)
        )
    )


def _numerical_mle(
    family: DistributionFamily, sample: Vector, spread: float
) -> tuple[Vector, bool, str, str]:
    """Maximize the likelihood by BFGS, falling back to Nelder-Mead.

    Returns:
        Estimates, convergence flag, method label, and termination message.

    Raises:
        ValueError: If the starting values have zero likelihood.

    References:
        [EQ-INF-MLE-GEN-001], [EQ-OPT-BFGS-001], [EQ-OPT-NM-001],
        [REF-NOCEDAL-WRIGHT-2006, Chapter 6], [REF-NELDER-MEAD-1965].

    """
    anchor = _check_anchor(family, family.initial(sample))
    scales = _reference_scales(family, anchor, spread)
    size = len(anchor)

    def objective(u: Vector) -> float:
        """Return the negative log-likelihood in unconstrained coordinates.

        References:
            [EQ-INF-MLE-GEN-001], [REF-CASELLA-BERGER-2002, Section 7.2].

        """
        return _negative_log_likelihood(
            family, sample, _to_natural(family, anchor, scales, u)
        )

    origin = (0.0,) * size
    if not math.isfinite(objective(origin)):
        raise ValueError(f"{family.name}: starting values have zero likelihood")
    result: OptimizationResult | None
    try:
        result = minimize_bfgs(objective, origin)
    except ValueError:
        result = None
    if result is not None and result.converged:
        return (
            _to_natural(family, anchor, scales, result.x),
            True,
            "bfgs",
            result.message,
        )
    simplex = minimize_nelder_mead(
        objective,
        result.x if result is not None else origin,
        initial_step=0.25,
        tolerance=1e-12,
    )
    best_x, best_value = simplex.x, simplex.value
    converged, message = simplex.converged, simplex.message
    try:
        polished = minimize_bfgs(objective, simplex.x)
    except ValueError:
        polished = None
    if polished is not None and polished.value <= best_value:
        best_x, best_value = polished.x, polished.value
        converged, message = polished.converged, polished.message
    return (
        _to_natural(family, anchor, scales, best_x),
        converged,
        "nelder-mead+bfgs",
        message,
    )


def _observed_information(
    family: DistributionFamily, sample: Vector, estimates: Vector, spread: float
) -> tuple[Vector, Matrix | None, str]:
    """Return delta-method standard errors from the observed information.

    The Hessian ``H_u`` of the negative log-likelihood is evaluated by central
    differences in the unconstrained coordinates anchored at the estimate,
    where every coordinate has unit natural scale, and mapped back with the
    diagonal Jacobian ``J``: ``Cov(theta) = J H_u^-1 J``. A bounded parameter
    whose estimate sits within ``0.1 %`` of its bound is held fixed: its
    standard error is ``nan`` and the others come from the remaining block.

    Returns:
        Standard errors, covariance (or ``None``), and a note.

    References:
        [EQ-INF-MLE-GEN-002], [EQ-OPT-FD-002],
        [REF-CASELLA-BERGER-2002, Section 10.1].

    """
    size = len(estimates)
    undefined = (math.nan,) * size
    if not family.regular:
        return (
            undefined,
            None,
            "non-regular family: support depends on parameters, so "
            "observed-information standard errors are undefined",
        )
    try:
        anchor = _check_anchor(family, estimates)
    except ValueError as error:
        return undefined, None, f"standard errors unavailable: {error}"
    scales = _reference_scales(family, anchor, spread)
    free: list[int] = []
    for index, value in enumerate(anchor):
        bound = family.bound(index)
        if bound is None or value < (1.0 - 1e-3) * bound:
            free.append(index)
    if not free:
        return undefined, None, "every parameter sits on its bound"

    def objective(v: Vector) -> float:
        """Return the negative log-likelihood over the free coordinates.

        References:
            [EQ-INF-MLE-GEN-001], [REF-CASELLA-BERGER-2002, Section 7.2].

        """
        u = [0.0] * size
        for position, index in enumerate(free):
            u[index] = v[position]
        return _negative_log_likelihood(
            family, sample, _to_natural(family, anchor, scales, tuple(u))
        )

    try:
        hessian, _ = central_hessian(objective, (0.0,) * len(free))
        covariance_v = inverse_spd(hessian)
    except ValueError:
        return (
            undefined,
            None,
            "observed information is not positive definite; "
            "standard errors unavailable",
        )
    jacobian: list[float] = []
    for index in free:
        kind, value = family.parameter_kinds[index], anchor[index]
        bound = family.bound(index)
        if kind == "positive" and bound is not None:
            jacobian.append(value * (1.0 - value / bound))
        elif kind == "positive":
            jacobian.append(value)
        else:
            jacobian.append(scales[index])
    rows = [[math.nan] * size for _ in range(size)]
    for a, i in enumerate(free):
        for b, j in enumerate(free):
            rows[i][j] = jacobian[a] * covariance_v[a][b] * jacobian[b]
    covariance = tuple(tuple(row) for row in rows)
    errors = tuple(
        math.sqrt(max(covariance[i][i], 0.0)) if i in free else math.nan
        for i in range(size)
    )
    note = "observed-information standard errors"
    if len(free) < size:
        fixed = ", ".join(
            family.parameter_names[i] for i in range(size) if i not in free
        )
        note += f" (parameter at its bound held fixed: {fixed})"
    return errors, covariance, note


def information_criteria(
    log_likelihood: float, parameter_count: int, sample_size: int
) -> tuple[float, float, float]:
    """Return ``(AIC, AICc, BIC)`` for a maximized log-likelihood.

    Args:
        log_likelihood: Maximized log-likelihood ``log L``.
        parameter_count: Number of estimated parameters ``k >= 1``.
        sample_size: Number of observations ``n >= 1``.

    Returns:
        ``AIC = 2k - 2 log L``, ``AICc = AIC + 2k(k+1)/(n-k-1)`` (``inf`` when
        ``n <= k + 1``), and ``BIC = k log n - 2 log L``.

    Raises:
        TypeError: If counts are not integers.
        ValueError: If counts are not positive or the log-likelihood is NaN.

    Equation:
        [EQ-INF-IC-001], [EQ-INF-IC-002], [EQ-INF-IC-003].

    References:
        [REF-AKAIKE-1974], [REF-HURVICH-TSAI-1989], [REF-SCHWARZ-1978].

    """
    k = require_non_negative_integer(parameter_count, name="parameter_count")
    n = require_non_negative_integer(sample_size, name="sample_size")
    if k == 0 or n == 0:
        raise ValueError("parameter_count and sample_size must be positive")
    value = float(log_likelihood)
    if math.isnan(value):
        raise ValueError("log_likelihood must not be NaN")
    aic = 2.0 * k - 2.0 * value
    aicc = aic + 2.0 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else math.inf
    bic = k * math.log(n) - 2.0 * value
    return aic, aicc, bic


def akaike_weights(criterion_values: Iterable[float]) -> tuple[float, ...]:
    """Return normalized weights ``exp(-Delta_i / 2) / sum_j exp(-Delta_j / 2)``.

    Args:
        criterion_values: Information-criterion values (AIC, AICc, or BIC);
            ``inf`` entries receive zero weight.

    Returns:
        Weights summing to one.

    Raises:
        ValueError: If the input is empty, contains NaN, or has no finite value.

    Equation:
        [EQ-INF-IC-004].

    References:
        [REF-BURNHAM-ANDERSON-2002, Chapter 2].

    """
    values = tuple(float(value) for value in criterion_values)
    if not values:
        raise ValueError("at least one criterion value is required")
    if any(math.isnan(value) for value in values):
        raise ValueError("criterion values must not be NaN")
    finite = [value for value in values if math.isfinite(value)]
    if not finite or any(value == -math.inf for value in values):
        raise ValueError("criterion values need a finite minimum")
    smallest = min(finite)
    raw = tuple(
        math.exp(-0.5 * (value - smallest)) if math.isfinite(value) else 0.0
        for value in values
    )
    total = math.fsum(raw)
    return tuple(weight / total for weight in raw)


def _validated_fit_sample(
    values: Iterable[object], family: DistributionFamily, method: str
) -> Vector:
    """Validate inputs shared by fitting and bootstrap calibration.

    Raises:
        TypeError: If ``family`` is not a :class:`DistributionFamily`.
        ValueError: If the method is unknown, the sample is outside the
            family domain, or it has fewer than two distinct values.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not isinstance(family, DistributionFamily):
        raise TypeError("family must be a DistributionFamily")
    if method not in _FIT_METHODS:
        raise ValueError(f"method must be one of {_FIT_METHODS}")
    if method == "closed-form" and family.closed_form is None:
        raise ValueError(f"{family.name} has no closed-form MLE")
    if method == "numerical" and not family.regular:
        raise ValueError(
            f"{family.name} is non-regular; its MLE sits on a support boundary "
            "and is not found by smooth optimization"
        )
    sample = coerce_finite_sample(values)
    if len(set(sample)) < 2:
        raise ValueError("fitting requires at least two distinct observations")
    if not family.admits(sample):
        raise ValueError(
            f"sample lies outside the {family.name} domain ({family.domain})"
        )
    return sample


def _estimate(
    family: DistributionFamily, sample: Vector, method: str
) -> tuple[Vector, bool, str, str, float]:
    """Return estimates, convergence, method label, message, and data spread.

    References:
        [EQ-INF-MLE-GEN-001], [REF-CASELLA-BERGER-2002, Section 7.2].

    """
    spread = _positive_spread(sample)
    if family.closed_form is not None and method != "numerical":
        estimates = tuple(float(value) for value in family.closed_form(sample))
        return (
            estimates,
            True,
            "closed-form",
            "closed-form maximum-likelihood estimate",
            spread,
        )
    estimates, converged, label, message = _numerical_mle(family, sample, spread)
    return estimates, converged, label, message, spread


def fit_distribution(
    values: Iterable[object],
    family: DistributionFamily,
    *,
    method: FitMethod = "auto",
) -> FitResult:
    """Fit a parametric family by maximum likelihood with standard errors.

    ``method="auto"`` uses the family's closed-form or profile MLE when one
    exists and otherwise minimizes the negative log-likelihood with BFGS in
    unconstrained coordinates (log scale for positive parameters, data-spread
    units for locations), falling back to Nelder-Mead plus a BFGS polish when
    BFGS fails. ``method="numerical"`` forces the optimizer, which is how the
    closed forms are cross-checked.

    Args:
        values: Finite observations inside the family's sample domain with at
            least two distinct values.
        family: Family description, e.g. :data:`WEIBULL_FAMILY`.
        method: ``"auto"``, ``"closed-form"``, or ``"numerical"``.

    Returns:
        Estimates, observed-information standard errors and covariance,
        log-likelihood, AIC, AICc, BIC, convergence flag, and fitted law.

    Raises:
        TypeError: If inputs have the wrong type.
        ValueError: If the sample is invalid for the family, the method is
            unknown or unavailable, or the start has zero likelihood.
        ArithmeticError: If a closed-form solver fails to converge or the
            maximized likelihood is not finite.

    Equation:
        [EQ-INF-MLE-GEN-001], [EQ-INF-MLE-GEN-002], [EQ-INF-IC-001],
        [EQ-INF-IC-002], [EQ-INF-IC-003].

    References:
        [REF-CASELLA-BERGER-2002, Sections 7.2 and 10.1],
        [REF-NOCEDAL-WRIGHT-2006, Chapter 6], [REF-AKAIKE-1974],
        [REF-SCHWARZ-1978], [REF-HURVICH-TSAI-1989].

    """
    sample = _validated_fit_sample(values, family, method)
    estimates, converged, label, message, spread = _estimate(family, sample, method)
    law = family.build(estimates)
    negative = _negative_log_likelihood(family, sample, estimates)
    if not math.isfinite(negative):
        raise ArithmeticError(f"{family.name}: maximized likelihood is not finite")
    log_likelihood = -negative
    errors, covariance, note = _observed_information(
        family, sample, estimates, spread
    )
    aic, aicc, bic = information_criteria(log_likelihood, len(estimates), len(sample))
    return FitResult(
        family=family.name,
        parameter_names=family.parameter_names,
        estimates=estimates,
        standard_errors=errors,
        covariance=covariance,
        log_likelihood=log_likelihood,
        sample_size=len(sample),
        aic=aic,
        aicc=aicc,
        bic=bic,
        converged=converged,
        method=label,
        message=f"{message}; {note}",
        distribution=law,
    )


# ---------------------------------------------------------------------------
# Goodness of fit with estimated parameters
# ---------------------------------------------------------------------------


def bootstrap_goodness_of_fit(
    values: Iterable[object],
    family: DistributionFamily,
    *,
    n_bootstrap: int = 199,
    seed: int | RandomStream | None = None,
    fit: FitResult | None = None,
    method: FitMethod = "auto",
) -> BootstrapGoodnessOfFit:
    """Calibrate KS and AD p-values when parameters are fitted to the data.

    The ordinary KS and AD null laws assume a fully specified model. When
    parameters are estimated from the same data, the statistics are
    stochastically smaller and the ordinary p-values are conservative
    (Lilliefors, Stephens). This routine fits ``theta_hat``, then for every
    replicate draws ``n`` values from ``F(.; theta_hat)``, refits the family
    with the same method, and recomputes both statistics against the refitted
    law. Replicates whose refit fails are excluded and counted.

    Args:
        values: Finite observations admissible for ``family``.
        family: Null family.
        n_bootstrap: Positive number of parametric-bootstrap replicates.
        seed: Integer seed or :class:`RandomStream`; ``None`` is not
            reproducible.
        fit: Optional existing fit of ``family`` to ``values`` (reused).
        method: Fitting method for the original fit and every refit.

    Returns:
        Observed statistics, bootstrap p-values, and refit accounting.

    Raises:
        TypeError: If inputs have the wrong type.
        ValueError: If the sample or settings are invalid, or ``fit`` belongs
            to another family or sample size.
        ArithmeticError: If every bootstrap refit fails.

    Equation:
        [EQ-INF-GOF-PB-001], [EQ-INF-GOF-KS-001], [EQ-INF-GOF-AD-001].

    References:
        [REF-STEPHENS-1974], [REF-LILLIEFORS-1967],
        [REF-DAVISON-HINKLEY-1997, Chapter 4],
        [REF-EFRON-TIBSHIRANI-1993, Chapter 16].

    """
    sample = _validated_fit_sample(values, family, method)
    replicates = require_non_negative_integer(n_bootstrap, name="n_bootstrap")
    if replicates == 0:
        raise ValueError("n_bootstrap must be positive")
    if fit is None:
        fit = fit_distribution(sample, family, method=method)
    elif not isinstance(fit, FitResult):
        raise TypeError("fit must be a FitResult or None")
    if fit.family != family.name or fit.sample_size != len(sample):
        raise ValueError("fit does not belong to this family and sample")
    refit_method: FitMethod = "numerical" if fit.method != "closed-form" else method
    stream = resolve_stream(seed).child("bootstrap-gof", family.name)
    law = fit.distribution
    ks_observed = kolmogorov_smirnov_test(sample, law).statistic
    ad_observed = anderson_darling_test(sample, law).statistic
    ks_exceed = ad_exceed = refits = failed = 0
    for index in range(replicates):
        try:
            replicate = law.sample(len(sample), seed=stream.child(index).seed)
            if len(set(replicate)) < 2 or not family.admits(replicate):
                raise ValueError("degenerate bootstrap replicate")
            estimates, _, _, _, _ = _estimate(family, replicate, refit_method)
            refitted = family.build(estimates)
            ks_value = kolmogorov_smirnov_test(replicate, refitted).statistic
            ad_value = anderson_darling_test(replicate, refitted).statistic
        except _FIT_FAILURES:
            failed += 1
            continue
        refits += 1
        ks_exceed += ks_value >= ks_observed
        ad_exceed += ad_value >= ad_observed
    if refits == 0:
        raise ArithmeticError("every bootstrap refit failed")
    ks_p = (1 + ks_exceed) / (1 + refits)
    ad_p = math.nan if math.isinf(ad_observed) else (1 + ad_exceed) / (1 + refits)
    return BootstrapGoodnessOfFit(
        family=family.name,
        ks_statistic=ks_observed,
        ks_p_value=ks_p,
        ad_statistic=ad_observed,
        ad_p_value=ad_p,
        n_bootstrap=replicates,
        refits=refits,
        failed_refits=failed,
        sample_size=len(sample),
    )


def qq_points(
    values: Iterable[object], distribution: FittableDistribution
) -> tuple[tuple[float, float], ...]:
    """Return Q-Q pairs ``(F^-1(p_i), x_(i))`` with Hazen positions.

    Args:
        values: Non-empty finite observations.
        distribution: Law exposing ``quantile``.

    Returns:
        Pairs of theoretical and empirical quantiles in ascending order, with
        ``p_i = (i - 1/2) / n``.

    Raises:
        TypeError: If observations are not real or the law lacks ``quantile``.
        ValueError: If the sample is empty or non-finite.

    Equation:
        [EQ-INF-GOF-QQ-001].

    References:
        [REF-WILK-GNANADESIKAN-1968], [REF-NIST-EDA, Section 1.3.3].

    """
    sample = sorted(coerce_finite_sample(values))
    if not callable(getattr(distribution, "quantile", None)):
        raise TypeError("distribution must expose quantile(p)")
    count = len(sample)
    return tuple(
        (float(distribution.quantile((rank - 0.5) / count)), value)
        for rank, value in enumerate(sample, start=1)
    )


def pp_points(
    values: Iterable[object], distribution: FittableDistribution
) -> tuple[tuple[float, float], ...]:
    """Return P-P pairs ``(F(x_(i)), p_i)`` with Hazen positions.

    Args:
        values: Non-empty finite observations.
        distribution: Law exposing ``cdf``.

    Returns:
        Pairs of theoretical and empirical probabilities in ascending order,
        with ``p_i = (i - 1/2) / n``.

    Raises:
        TypeError: If observations are not real or the law lacks ``cdf``.
        ValueError: If the sample is empty, non-finite, or a CDF value is not
            a probability.

    Equation:
        [EQ-INF-GOF-QQ-001].

    References:
        [REF-WILK-GNANADESIKAN-1968], [REF-NIST-EDA, Section 1.3.3].

    """
    sample = sorted(coerce_finite_sample(values))
    if not callable(getattr(distribution, "cdf", None)):
        raise TypeError("distribution must expose cdf(x)")
    count = len(sample)
    return tuple(
        (
            require_probability(distribution.cdf(value), name="distribution.cdf(x)"),
            (rank - 0.5) / count,
        )
        for rank, value in enumerate(sample, start=1)
    )


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def _resolve_candidates(
    candidates: Sequence[DistributionFamily | str] | None,
    sample: Vector,
) -> tuple[list[DistributionFamily], list[tuple[str, str]]]:
    """Resolve candidate families and record domain exclusions.

    References:
        Project API policy. No equation is evaluated.

    """
    if candidates is None:
        pool: list[DistributionFamily] = list(STANDARD_FAMILIES)
    else:
        if isinstance(candidates, str | DistributionFamily):
            raise TypeError("candidates must be a sequence of families or names")
        pool = []
        for item in candidates:
            if isinstance(item, str):
                pool.append(family_by_name(item))
            elif isinstance(item, DistributionFamily):
                pool.append(item)
            else:
                raise TypeError("each candidate must be a DistributionFamily or name")
    if not pool:
        raise ValueError("at least one candidate family is required")
    names = [family.name for family in pool]
    if len(set(names)) != len(names):
        raise ValueError("candidate family names must be distinct")
    admitted: list[DistributionFamily] = []
    excluded: list[tuple[str, str]] = []
    for family in pool:
        if family.admits(sample):
            admitted.append(family)
        else:
            excluded.append(
                (family.name, f"sample lies outside the domain ({family.domain})")
            )
    return admitted, excluded


def select_distribution(
    values: Iterable[object],
    candidates: Sequence[DistributionFamily | str] | None = None,
    *,
    criterion: Criterion = "aic",
    n_bootstrap: int = 99,
    alpha: float = 0.05,
    seed: int | RandomStream | None = None,
) -> ModelSelectionResult:
    """Fit candidate families and rank them by an information criterion.

    Every admissible candidate is fitted by maximum likelihood. Candidates are
    ranked by AIC, AICc, or BIC; Akaike and BIC weights are reported. When
    ``n_bootstrap > 0`` each candidate also receives a bootstrap-calibrated
    KS/AD p-value. A :class:`ModelSelectionWarning` is emitted when the best
    model is still rejected at ``alpha`` (the ranking is relative: the best of
    poor models is still poor), when a fit did not converge, or when a
    candidate failed.

    Args:
        values: Finite observations with at least two distinct values.
        candidates: Families or ready-made names; ``None`` uses every
            ready-made family whose domain admits the sample.
        criterion: ``"aic"``, ``"aicc"``, or ``"bic"``.
        n_bootstrap: Bootstrap replicates per candidate (``0`` skips GOF).
        alpha: Rejection level for GOF warnings, in ``(0, 1)``.
        seed: Integer seed or :class:`RandomStream` for bootstrap streams.

    Returns:
        Ranked assessments, failures, and warnings.

    Raises:
        TypeError: If inputs have the wrong type.
        KeyError: If a candidate name is unknown.
        ValueError: If settings are invalid, candidate names repeat, or no
            candidate could be fitted.

    Equation:
        [EQ-INF-IC-001], [EQ-INF-IC-002], [EQ-INF-IC-003], [EQ-INF-IC-004],
        [EQ-INF-GOF-PB-001].

    References:
        [REF-BURNHAM-ANDERSON-2002, Chapter 2], [REF-AKAIKE-1974],
        [REF-SCHWARZ-1978], [REF-STEPHENS-1974].

    """
    sample = coerce_finite_sample(values)
    if len(set(sample)) < 2:
        raise ValueError("model selection requires at least two distinct values")
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {_CRITERIA}")
    replicates = require_non_negative_integer(n_bootstrap, name="n_bootstrap")
    level = require_probability(alpha, name="alpha")
    if not 0.0 < level < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1)")
    stream = resolve_stream(seed) if replicates > 0 else RandomStream(0)
    families, failures = _resolve_candidates(candidates, sample)
    fits: list[FitResult] = []
    for family in families:
        try:
            fits.append(fit_distribution(sample, family))
        except _FIT_FAILURES as error:
            failures.append((family.name, f"fit failed: {error}"))
    if not fits:
        reasons = "; ".join(f"{name}: {reason}" for name, reason in failures)
        raise ValueError(f"no candidate family could be fitted ({reasons})")
    gof: dict[str, BootstrapGoodnessOfFit | None] = {}
    by_name = {family.name: family for family in families}
    for fitted in fits:
        if replicates == 0:
            gof[fitted.family] = None
            continue
        try:
            gof[fitted.family] = bootstrap_goodness_of_fit(
                sample,
                by_name[fitted.family],
                n_bootstrap=replicates,
                seed=stream.child("candidate", fitted.family),
                fit=fitted,
            )
        except _FIT_FAILURES as error:
            gof[fitted.family] = None
            failures.append((fitted.family, f"bootstrap GOF failed: {error}"))
    aic_weights = akaike_weights(fit.aic for fit in fits)
    bic_weights = akaike_weights(fit.bic for fit in fits)
    min_aic = min(fit.aic for fit in fits)
    min_bic = min(fit.bic for fit in fits)

    def score(item: tuple[int, FitResult]) -> tuple[float, int]:
        """Return the ranking key: criterion value, then candidate order.

        References:
            [REF-BURNHAM-ANDERSON-2002, Chapter 2]. No equation is evaluated.

        """
        index, fitted = item
        return (getattr(fitted, criterion), index)

    order = sorted(enumerate(fits), key=score)
    ranked = tuple(
        CandidateAssessment(
            rank=position,
            fit=fitted,
            delta_aic=fitted.aic - min_aic,
            akaike_weight=aic_weights[index],
            delta_bic=fitted.bic - min_bic,
            bic_weight=bic_weights[index],
            goodness_of_fit=gof[fitted.family],
        )
        for position, (index, fitted) in enumerate(order, start=1)
    )
    messages = _selection_warnings(ranked, failures, level)
    if replicates and 1.0 / (replicates + 1) >= level:
        messages += (
            f"n_bootstrap={replicates} cannot produce a p-value below "
            f"alpha={level:g} (smallest attainable is {1.0 / (replicates + 1):.3g}); "
            "increase n_bootstrap",
        )
    for message in messages:
        warnings.warn(message, ModelSelectionWarning, stacklevel=2)
    return ModelSelectionResult(
        criterion=criterion,
        candidates=ranked,
        failures=tuple(failures),
        warnings=messages,
        alpha=level,
        sample_size=len(sample),
    )


def _selection_warnings(
    ranked: tuple[CandidateAssessment, ...],
    failures: list[tuple[str, str]],
    alpha: float,
) -> tuple[str, ...]:
    """Compose model-selection warnings for rejection and convergence.

    References:
        [REF-BURNHAM-ANDERSON-2002, Chapter 2]. No equation is evaluated.

    """
    messages: list[str] = []
    best = ranked[0]
    check = best.goodness_of_fit
    if check is not None and check.rejected(alpha):
        messages.append(
            f"best model {best.fit.family} is rejected by the bootstrap "
            f"goodness-of-fit test at alpha={alpha:g} (KS p={check.ks_p_value:.3g}, "
            f"AD p={check.ad_p_value:.3g}); the ranking is relative and no "
            "candidate describes the data adequately"
        )
    for candidate in ranked:
        if not candidate.fit.converged:
            messages.append(
                f"{candidate.fit.family} fit did not converge "
                f"({candidate.fit.message}); its criterion values are unreliable"
            )
    for name, reason in failures:
        if reason.startswith(("fit failed", "bootstrap GOF failed")):
            messages.append(f"{name}: {reason}")
    return tuple(messages)
