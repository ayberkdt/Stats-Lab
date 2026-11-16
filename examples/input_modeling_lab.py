"""Input-modeling laboratory: build the law of an uncertain input step by step.

The law of an uncertain input should encode the information that is
available and nothing more. This laboratory walks the full decision on one
seeded data set that resembles wind-speed maxima: 400 block maxima, each the
largest of 144 Weibull(2, 8) speeds (one day of ten-minute means), and the
400 block means of the same speeds. It follows the structure of a classical
input-modeling exercise: describe the data; build a law from expert
statements alone; fit candidate families by likelihood and by moments and
test them with calibrated p-values; propagate parameter uncertainty to the
fitted CDF; estimate the density without a family when data are plentiful;
and compare a design value, the 1-in-100 block maximum, with its exact value
``F^-1(0.99^(1/144))``. The data are simulated by the library, so the exact
law of the maximum, ``F(x)^144``, is known and every answer can be checked.

Definitions:
    block maximum ``M = max(V_1, ..., V_144)`` and block mean
        ``(V_1 + ... + V_144) / 144`` of ``V_j ~ Weibull(shape 2, scale 8)``
        speeds in m/s; 400 independent blocks.
    sample skewness ``g1 = m3 / m2^(3/2)`` and excess kurtosis
        ``g2 = m4 / m2^2 - 3`` with ``m_k = (1/n) sum (x_i - xbar)^k`` (type 1
        of R's ``e1071``); R's default (type 3) divides by the ``n - 1``
        standard deviation: ``b1 = g1 ((n - 1)/n)^(3/2)`` and
        ``b2 = (g2 + 3)(1 - 1/n)^2 - 3``.
    maximum-entropy law: the density of largest entropy on a stated range
        with stated moments.
    candidate family: normal, lognormal, gamma, Weibull, or GEV, fitted by
        maximum likelihood (MLE) and by the method of moments.
    naive p-value: Kolmogorov-Smirnov (KS) or Anderson-Darling (AD) p-value
        computed as if the fitted law had been fixed in advance; bootstrap
        p-value: the same statistic calibrated by refitting every
        parametric-bootstrap replicate.
    parameter-uncertainty band: pointwise band of ``F(x; theta*)`` over
        parameters drawn from the asymptotic normal law of the estimator;
        quantile band, min/max envelope, and delta-method band.
    integrated squared error ``ISE(h) = int (f_h(x) - f(x))^2 dx`` of a
        kernel density estimate ``f_h`` against the exact density ``f``.
    design value: the 0.99 quantile of the block maximum (exceeded by one
        block in 100 on average).

Assumptions:
    - The 400 blocks are independent and identically distributed; within a
      block the 144 speeds are independent. Real ten-minute speeds are
      autocorrelated and seasonal, so the exact law ``F(x)^144`` holds for
      the simulation only.
    - The expert statements (a range of 12 to 32 m/s, a mean of 19 m/s, a
      standard deviation of 2 m/s, and optionally the skewness of a Gumbel
      law) are round values chosen before looking at the data.
    - Parameter bands use the asymptotic normal law of the estimators; they
      describe parameter uncertainty within a family, not the uncertainty
      of the choice of family.
    - The course data file of 400 wind maxima is not used: its licence is
      unknown, so every number here comes from library-generated data.

Equations:
    EQ-DIST-WEIB-003: Weibull quantile used to draw the speeds.
    EQ-EVA-BM-001: block maxima of consecutive blocks.
    EQ-LAB-INPUT-001: exact law ``F(x)^m`` of a block maximum, its density
        and quantile.
    EQ-STAT-SHAPE-001: moment skewness and excess kurtosis.
    EQ-LAB-INPUT-002: R ``e1071`` type 3 skewness and kurtosis.
    EQ-STAT-QUANT-001: sample quantiles (type 7).
    EQ-DIST-MAXENT-001: maximum-entropy density and its convex dual.
    EQ-INF-MLE-GEN-001: maximum-likelihood fit of each candidate.
    EQ-INF-MLE-GEN-002: observed-information standard errors.
    EQ-INF-MOM-001: method-of-moments estimator with population moments.
    EQ-INF-MOM-002: closed-form moment estimators (normal, lognormal, gamma).
    EQ-INF-MOM-003: Weibull and GEV moment equations.
    EQ-INF-MOM-004: delta-method covariance of moment estimators.
    EQ-INF-IC-001: Akaike information criterion.
    EQ-INF-IC-004: criterion differences and Akaike weights.
    EQ-INF-GOF-KS-001: Kolmogorov-Smirnov statistic.
    EQ-INF-GOF-AD-001: Anderson-Darling statistic.
    EQ-INF-GOF-AD-003: finite-sample Anderson-Darling p-value (naive).
    EQ-INF-GOF-PB-001: parametric-bootstrap p-values with refits.
    EQ-INF-BAND-001: parameter-draw CDF and quantile bands.
    EQ-INF-BAND-002: delta-method pointwise band.
    EQ-STAT-KDE-001: Gaussian kernel density estimate and Silverman's rule.
    EQ-STAT-KDE-004: leave-one-out likelihood cross-validation.
    EQ-STAT-KDE-005: unbiased cross-validation score.
    EQ-STAT-KDE-006: multivariate kernel estimate of the joint sample.
    EQ-LAB-INPUT-003: integrated squared error of a density estimate.
    EQ-INF-GOF-QQ-001: Hazen plotting positions of the observed maxima.

Diagnostics:
    - Shape: a skewness near zero and an excess kurtosis near zero point to
      a normal law; the maxima are right-skewed, the means are not.
    - Expert law: the constraints are checked after the solve (residuals
      below ``1e-10``); the density at the range ends shows whether a
      high moment piles mass against an arbitrary bound.
    - Fit: AIC ranks the candidates; a bootstrap p-value below
      ``ALPHA = 0.05`` rejects a candidate whatever its rank; the naive
      p-values are printed next to the bootstrap ones to show how much too
      large they are after fitting.
    - Bands: the quantile band settles as the number of draws grows, the
      min/max envelope keeps widening, and the delta band agrees with the
      quantile band where the CDF is close to linear in the parameters.
    - Kernel estimates: the ISE against the exact density ranks the three
      bandwidths; a cross-validation optimum on the end of its search
      interval is reported and the interval is widened.
    - Design value: each band is checked against the exact quantile.

Cost and error:
    - About one minute in pure Python, most of it in the
      ``BOOTSTRAP_REPLICATES = 199`` refits of each of the five candidates
      (numerical GEV refits dominate); the run prints its wall-clock time.
    - A bootstrap p-value ``p`` from ``B = 199`` refits has resolution
      ``1/200`` and Monte Carlo standard error ``sqrt(p (1 - p) / 199)``, at
      most 0.035; ``0.005`` means "at or below the floor".
    - Each band uses ``BAND_DRAWS = 2000`` parameter draws; a 95% band
      limit then carries a Monte Carlo error of a few per cent of its
      width. Standard errors are asymptotic.
    - ISE values come from adaptive Gauss-Kronrod quadrature with relative
      tolerance ``1e-8``; the exact quantities need no simulation.

References:
    REF-COLES-2001: block maxima and the extremal types theorem.
    REF-CASELLA-BERGER-2002: the central limit theorem and the delta method.
    REF-JOANES-GILL-1998: sample skewness and kurtosis estimators.
    REF-R-E1071: R's ``skewness`` and ``kurtosis`` conventions.
    REF-JAYNES-1957: maximum-entropy laws.
    REF-DELIGNETTE-MULLER-DUTANG-2015: moment and likelihood fits in R.
    REF-AKAIKE-1974: the Akaike information criterion.
    REF-STEPHENS-1974: EDF goodness-of-fit statistics.
    REF-SILVERMAN-1986: kernel density estimation.
    REF-BOWMAN-1984: integrated squared error and least-squares
        cross-validation.
    REF-MATPLOTLIB: figures.

"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from probstat.distributions import Gumbel, MaximumEntropy, Weibull, maximum_entropy
from probstat.inference import (
    GAMMA_FAMILY,
    GEV_FAMILY,
    LOGNORMAL_FAMILY,
    NORMAL_FAMILY,
    WEIBULL_FAMILY,
    BootstrapGoodnessOfFit,
    FitResult,
    MomentFitResult,
    ParameterUncertaintyBand,
    akaike_weights,
    anderson_darling_test,
    bootstrap_goodness_of_fit,
    fit_distribution,
    fit_distribution_moments,
    kolmogorov_smirnov_test,
    parameter_uncertainty_band,
)
from probstat.numerics import gauss_kronrod
from probstat.risk import block_maxima
from probstat.statistics import (
    excess_kurtosis,
    gaussian_kde,
    gaussian_kde_multivariate,
    kde_bandwidth,
    mean,
    median,
    oversmoothed_bandwidth,
    pearson_correlation,
    quantile,
    select_kde_bandwidth,
    skewness,
    standard_deviation,
)
from probstat.visualization import save_figure, theme

# Seed of the speeds, the bootstrap refits, and the parameter draws.
SEED = 2026
BLOCK_COUNT = 400
BLOCK_SIZE = 144
PARENT = Weibull(2.0, 8.0)
DESIGN_PROBABILITY = 0.99
ALPHA = 0.05
BOOTSTRAP_REPLICATES = 199
BAND_DRAWS = 2000
BAND_LEVEL = 0.95
ENVELOPE_DRAWS = (50, 100, 200, 500, 1000, 2000, 5000, 10000)
PROBE_SPEED = 24.0
EXPERT_RANGE = (12.0, 32.0)
EXPERT_MEAN = 19.0
EXPERT_STANDARD_DEVIATION = 2.0
EXPERT_UPPER_ENDS = (30.0, 32.0, 35.0, 40.0)
GUMBEL_SKEWNESS = Gumbel(0.0, 1.0).skewness
FAMILIES = {
    "Normal": NORMAL_FAMILY,
    "LogNormal": LOGNORMAL_FAMILY,
    "Gamma": GAMMA_FAMILY,
    "Weibull": WEIBULL_FAMILY,
    "GEV": GEV_FAMILY,
}
CLOSED_FORM_MOMENTS = ("Normal", "LogNormal", "Gamma")
ISE_RANGE = (8.0, 48.0)
EXPERT_LABELS = ("range, mean, sd", "range, mean, sd, skewness")


# ---------------------------------------------------------------------------
# Data and the exact law of the maximum
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindData:
    """Simulated speeds with their block maxima and block means.

    Attributes:
        speeds: ``BLOCK_COUNT * BLOCK_SIZE`` Weibull speeds in m/s.
        maxima: One maximum per block.
        means: One mean per block.

    References:
        [EQ-EVA-BM-001], [REF-COLES-2001, Section 3.1].

    """

    speeds: tuple[float, ...]
    maxima: tuple[float, ...]
    means: tuple[float, ...]


def simulate_wind_data(seed: int = SEED) -> WindData:
    """Draw the speeds and form block maxima and block means.

    Args:
        seed: Seed of the Weibull speeds.

    Returns:
        :class:`WindData`.

    Equation:
        [EQ-DIST-WEIB-003], [EQ-EVA-BM-001].

    References:
        [REF-COLES-2001, Section 3.1].

    """
    speeds = tuple(PARENT.sample(BLOCK_COUNT * BLOCK_SIZE, seed=seed))
    blocks = [
        speeds[index * BLOCK_SIZE : (index + 1) * BLOCK_SIZE]
        for index in range(BLOCK_COUNT)
    ]
    return WindData(
        speeds=speeds,
        maxima=tuple(block_maxima(speeds, BLOCK_SIZE)),
        means=tuple(mean(block) for block in blocks),
    )


def exact_maximum_cdf(x: float, block_size: int = BLOCK_SIZE) -> float:
    """Return ``P(M <= x) = F(x)^m`` for the maximum of ``m`` parent speeds.

    Args:
        x: Speed in m/s.
        block_size: Number ``m`` of independent speeds per block.

    Returns:
        Exact CDF of the block maximum.

    Equation:
        [EQ-LAB-INPUT-001].

    References:
        [REF-COLES-2001, Section 3.1].

    """
    return float(PARENT.cdf(x)) ** block_size


def exact_maximum_pdf(x: float, block_size: int = BLOCK_SIZE) -> float:
    """Return the exact density ``m F(x)^(m-1) f(x)`` of the block maximum.

    Args:
        x: Speed in m/s.
        block_size: Number ``m`` of independent speeds per block.

    Returns:
        Exact density of the block maximum.

    Equation:
        [EQ-LAB-INPUT-001].

    References:
        [REF-COLES-2001, Section 3.1].

    """
    if x <= 0.0:
        return 0.0
    return block_size * float(PARENT.cdf(x)) ** (block_size - 1) * float(PARENT.pdf(x))


def exact_maximum_quantile(probability: float, block_size: int = BLOCK_SIZE) -> float:
    """Return the exact quantile ``F^-1(p^(1/m))`` of the block maximum.

    Args:
        probability: Probability in ``(0, 1)``.
        block_size: Number ``m`` of independent speeds per block.

    Returns:
        Exact quantile in m/s.

    Equation:
        [EQ-LAB-INPUT-001], [EQ-DIST-WEIB-003].

    References:
        [REF-COLES-2001, Section 3.1].

    """
    return float(PARENT.quantile(probability ** (1.0 / block_size)))


def exact_maximum_moments() -> tuple[float, float, float]:
    """Return the mean, standard deviation, and skewness of ``F(x)^144``.

    Returns:
        ``(mean, standard deviation, skewness)`` by adaptive quadrature.

    Equation:
        [EQ-LAB-INPUT-001].

    References:
        [REF-COLES-2001, Section 3.1].

    """
    low, high = ISE_RANGE

    def moment(power: Callable[[float], float]) -> float:
        """Integrate ``power(x) f(x)`` over the effective support.

        References:
            [EQ-LAB-INPUT-001].

        """
        return gauss_kronrod(
            lambda x: power(x) * exact_maximum_pdf(x), low, high, rel_tol=1e-11
        ).value

    centre = moment(lambda x: x)
    second = moment(lambda x: (x - centre) ** 2)
    third = moment(lambda x: (x - centre) ** 3)
    return centre, math.sqrt(second), third / second**1.5


# ---------------------------------------------------------------------------
# 1. Description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShapeSummary:
    """Location, spread, and shape of one sample.

    Attributes:
        size: Number of values.
        mean: Sample mean.
        standard_deviation: Standard deviation with divisor ``n - 1``.
        minimum: Smallest value.
        first_quartile: Type 7 quantile at 0.25.
        median: Sample median.
        third_quartile: Type 7 quantile at 0.75.
        maximum: Largest value.
        skewness: ``g1`` (``e1071`` type 1).
        excess_kurtosis: ``g2`` (``e1071`` type 1).
        skewness_type3: ``b1`` (``e1071`` default, type 3).
        excess_kurtosis_type3: ``b2`` (``e1071`` default, type 3).

    References:
        [EQ-STAT-SHAPE-001], [EQ-LAB-INPUT-002], [REF-JOANES-GILL-1998].

    """

    size: int
    mean: float
    standard_deviation: float
    minimum: float
    first_quartile: float
    median: float
    third_quartile: float
    maximum: float
    skewness: float
    excess_kurtosis: float
    skewness_type3: float
    excess_kurtosis_type3: float


def describe(values: Sequence[float]) -> ShapeSummary:
    """Summarize a sample and convert its shape to R's default convention.

    Args:
        values: At least two distinct finite values.

    Returns:
        :class:`ShapeSummary`.

    Equation:
        [EQ-STAT-SHAPE-001], [EQ-LAB-INPUT-002], [EQ-STAT-QUANT-001].

    References:
        [REF-JOANES-GILL-1998], [REF-R-E1071].

    """
    n = len(values)
    g1 = skewness(values)
    g2 = excess_kurtosis(values)
    return ShapeSummary(
        size=n,
        mean=mean(values),
        standard_deviation=standard_deviation(values, ddof=1),
        minimum=min(values),
        first_quartile=quantile(values, 0.25),
        median=median(values),
        third_quartile=quantile(values, 0.75),
        maximum=max(values),
        skewness=g1,
        excess_kurtosis=g2,
        skewness_type3=g1 * ((n - 1) / n) ** 1.5,
        excess_kurtosis_type3=(g2 + 3.0) * (1.0 - 1.0 / n) ** 2 - 3.0,
    )


# ---------------------------------------------------------------------------
# 2. Expert statements only
# ---------------------------------------------------------------------------


def expert_laws(upper: float = EXPERT_RANGE[1]) -> dict[str, MaximumEntropy]:
    """Return the maximum-entropy laws of the expert statements.

    Args:
        upper: Upper end of the stated range in m/s.

    Returns:
        Mapping from the constraint set to its law: range, mean, and standard
        deviation; and the same with the skewness of a Gumbel law.

    Equation:
        [EQ-DIST-MAXENT-001].

    References:
        [REF-JAYNES-1957], [REF-COLES-2001, Section 3.1].

    """
    lower = EXPERT_RANGE[0]
    variance = EXPERT_STANDARD_DEVIATION**2
    return {
        EXPERT_LABELS[0]: maximum_entropy(
            lower, upper, mean=EXPERT_MEAN, variance=variance
        ),
        EXPERT_LABELS[1]: maximum_entropy(
            lower,
            upper,
            mean=EXPERT_MEAN,
            variance=variance,
            skewness=GUMBEL_SKEWNESS,
        ),
    }


# ---------------------------------------------------------------------------
# 3. Candidate families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateResult:
    """Likelihood and moment fits of one family with its goodness of fit.

    Attributes:
        family: Family name.
        fit: Maximum-likelihood fit.
        moments: Method-of-moments fit.
        akaike_weight: Akaike weight within the candidate set.
        naive_ks_p: KS p-value as if the fitted law were fixed in advance.
        naive_ad_p: AD p-value as if the fitted law were fixed in advance.
        bootstrap: Bootstrap-calibrated p-values, or ``None`` if not run.

    References:
        [EQ-INF-MLE-GEN-001], [EQ-INF-MOM-001], [EQ-INF-GOF-PB-001].

    """

    family: str
    fit: FitResult
    moments: MomentFitResult
    akaike_weight: float
    naive_ks_p: float
    naive_ad_p: float
    bootstrap: BootstrapGoodnessOfFit | None

    def rejected(self, alpha: float = ALPHA) -> bool:
        """Return whether a bootstrap p-value falls below ``alpha``.

        References:
            [EQ-INF-GOF-PB-001].

        """
        if self.bootstrap is None:
            return False
        return min(self.bootstrap.ks_p_value, self.bootstrap.ad_p_value) < alpha


def fit_candidates(
    values: Sequence[float],
    *,
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
    bootstrap_families: Sequence[str] | None = None,
) -> tuple[CandidateResult, ...]:
    """Fit every candidate by likelihood and moments, ranked by AIC.

    Args:
        values: Positive sample.
        n_bootstrap: Bootstrap refits per family (0 skips the bootstrap).
        seed: Seed of the bootstrap refits.
        bootstrap_families: Families to calibrate (default: all).

    Returns:
        One :class:`CandidateResult` per family, best AIC first.

    Equation:
        [EQ-INF-MLE-GEN-001], [EQ-INF-MOM-001], [EQ-INF-IC-001],
        [EQ-INF-IC-004], [EQ-INF-GOF-KS-001], [EQ-INF-GOF-AD-001],
        [EQ-INF-GOF-PB-001].

    References:
        [REF-AKAIKE-1974], [REF-STEPHENS-1974],
        [REF-DELIGNETTE-MULLER-DUTANG-2015].

    """
    chosen = set(FAMILIES if bootstrap_families is None else bootstrap_families)
    fits = {name: fit_distribution(values, family) for name, family in FAMILIES.items()}
    weights = dict(
        zip(fits, akaike_weights(fit.aic for fit in fits.values()), strict=True)
    )
    results = []
    for name, family in FAMILIES.items():
        fit = fits[name]
        calibrated = None
        if n_bootstrap > 0 and name in chosen:
            calibrated = bootstrap_goodness_of_fit(
                values, family, n_bootstrap=n_bootstrap, seed=seed, fit=fit
            )
        results.append(
            CandidateResult(
                family=name,
                fit=fit,
                moments=fit_distribution_moments(values, family),
                akaike_weight=weights[name],
                naive_ks_p=kolmogorov_smirnov_test(values, fit.distribution).p_value,
                naive_ad_p=anderson_darling_test(values, fit.distribution).p_value,
                bootstrap=calibrated,
            )
        )
    return tuple(sorted(results, key=lambda result: result.fit.aic))


def choose_law(candidates: Sequence[CandidateResult]) -> CandidateResult:
    """Return the best-AIC candidate that the bootstrap does not reject.

    Args:
        candidates: Results of :func:`fit_candidates`, best AIC first.

    Returns:
        The chosen candidate (the best AIC if every candidate is rejected).

    References:
        [EQ-INF-IC-004], [EQ-INF-GOF-PB-001], [REF-AKAIKE-1974].

    """
    for candidate in candidates:
        if not candidate.rejected():
            return candidate
    return candidates[0]


# ---------------------------------------------------------------------------
# 4. Parameter uncertainty
# ---------------------------------------------------------------------------


def band_grid() -> tuple[float, ...]:
    """Return the speeds at which the fitted CDFs are banded.

    References:
        [EQ-INF-BAND-001].

    """
    return tuple(13.0 + 0.1 * k for k in range(141))


def fitted_band(
    fit: FitResult | MomentFitResult,
    grid: Sequence[float],
    *,
    draws: int = BAND_DRAWS,
    seed: int = SEED,
) -> ParameterUncertaintyBand:
    """Return the CDF bands of a fit with its 0.99-quantile band.

    Args:
        fit: Likelihood or moment fit.
        grid: Speeds in m/s.
        draws: Number of parameter draws.
        seed: Seed of the parameter draws.

    Returns:
        :class:`ParameterUncertaintyBand` at level ``BAND_LEVEL``.

    Equation:
        [EQ-INF-BAND-001], [EQ-INF-BAND-002].

    References:
        [REF-CASELLA-BERGER-2002, Section 5.5.4].

    """
    return parameter_uncertainty_band(
        fit,
        grid,
        draws=draws,
        level=BAND_LEVEL,
        seed=seed,
        quantile_probabilities=(DESIGN_PROBABILITY,),
    )


def envelope_widths(
    fit: FitResult, x: float = PROBE_SPEED, seed: int = SEED
) -> tuple[tuple[int, float, float, float], ...]:
    """Return the widths of the three exceedance bands at one speed.

    Args:
        fit: Likelihood fit.
        x: Speed in m/s.
        seed: Seed of the parameter draws (nested runs share their draws).

    Returns:
        ``(draws, quantile band, envelope, delta band)`` widths per entry of
        ``ENVELOPE_DRAWS``.

    Equation:
        [EQ-INF-BAND-001], [EQ-INF-BAND-002].

    References:
        [REF-CASELLA-BERGER-2002, Section 5.5.4].

    """
    rows = []
    for draws in ENVELOPE_DRAWS:
        band = parameter_uncertainty_band(
            fit, (x,), draws=draws, level=BAND_LEVEL, seed=seed
        )
        rows.append(
            (
                draws,
                band.upper[0] - band.lower[0],
                band.envelope_upper[0] - band.envelope_lower[0],
                band.delta_upper[0] - band.delta_lower[0],
            )
        )
    return tuple(rows)


# ---------------------------------------------------------------------------
# 5. Kernel density estimates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelResult:
    """One kernel density estimate and its error against the exact law.

    Attributes:
        method: Bandwidth rule or criterion.
        bandwidth: Kernel standard deviation in m/s.
        ise: Integrated squared error against ``exact_maximum_pdf``.
        design_value: 0.99 quantile of the estimate.
        note: Remark on the search (empty when nothing to report).

    References:
        [EQ-STAT-KDE-001], [EQ-LAB-INPUT-003].

    """

    method: str
    bandwidth: float
    ise: float
    design_value: float
    note: str


def integrated_squared_error(density: Callable[[float], float]) -> float:
    """Return ``int (f_h - f)^2 dx`` against the exact density of the maximum.

    Args:
        density: Estimated density.

    Returns:
        Integrated squared error over ``ISE_RANGE``, which holds all but a
        negligible part of both densities.

    Equation:
        [EQ-LAB-INPUT-003], [EQ-LAB-INPUT-001].

    References:
        [REF-BOWMAN-1984].

    """
    low, high = ISE_RANGE
    return gauss_kronrod(
        lambda x: (density(x) - exact_maximum_pdf(x)) ** 2, low, high, rel_tol=1e-8
    ).value


def kernel_estimates(values: Sequence[float]) -> tuple[KernelResult, ...]:
    """Fit Gaussian kernel estimates with three bandwidth selectors.

    Silverman's rule, leave-one-out likelihood, and unbiased
    cross-validation. When a cross-validation optimum lies on the end of
    the default search interval ``[0.1 h_os, h_os]``, the search is
    repeated on ``[0.1 h_os, 3 h_os]`` and the note says so.

    Args:
        values: Sample of block maxima.

    Returns:
        One :class:`KernelResult` per selector, in increasing bandwidth.

    Equation:
        [EQ-STAT-KDE-001], [EQ-STAT-KDE-004], [EQ-STAT-KDE-005],
        [EQ-LAB-INPUT-003].

    References:
        [REF-SILVERMAN-1986, Chapter 3], [REF-BOWMAN-1984].

    """
    reference = oversmoothed_bandwidth(values)
    chosen: list[tuple[str, float, str]] = [
        ("Silverman", kde_bandwidth(values, rule="silverman"), "")
    ]
    for label, method in (("LOO likelihood", "likelihood"), ("UCV", "ucv")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            selection = select_kde_bandwidth(values, method=method)
            note = ""
            if selection.at_boundary:
                end = selection.bandwidth
                selection = select_kde_bandwidth(
                    values, method=method, upper=3.0 * reference
                )
                note = (
                    f"default search ended at its bound {end:.3f}; "
                    f"searched again on [0.1, 3] h_os"
                )
        chosen.append((label, selection.bandwidth, note))
    results = []
    for label, bandwidth, note in chosen:
        estimate = gaussian_kde(values, bandwidth=bandwidth)
        results.append(
            KernelResult(
                method=label,
                bandwidth=bandwidth,
                ise=integrated_squared_error(estimate.pdf),
                design_value=estimate.quantile(DESIGN_PROBABILITY),
                note=note,
            )
        )
    return tuple(sorted(results, key=lambda result: result.bandwidth))


def joint_factors(
    means: Sequence[float], maxima: Sequence[float]
) -> tuple[float, float, float]:
    """Return the correlation and two kernel factors of (mean, maximum).

    Args:
        means: Block means.
        maxima: Block maxima of the same blocks.

    Returns:
        ``(Pearson correlation, Silverman factor, LOO-likelihood factor)``.

    Equation:
        [EQ-STAT-KDE-006].

    References:
        [REF-SILVERMAN-1986, Chapter 4].

    """
    points = list(zip(means, maxima, strict=True))
    return (
        pearson_correlation(means, maxima),
        gaussian_kde_multivariate(points).factor,
        gaussian_kde_multivariate(points, bandwidth="likelihood").factor,
    )


# ---------------------------------------------------------------------------
# The full analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DesignRow:
    """One estimate of the 1-in-100 block maximum.

    Attributes:
        label: Input law and estimator.
        estimate: 0.99 quantile in m/s.
        lower: Lower end of its 95% band (``nan`` without a band).
        upper: Upper end of its 95% band (``nan`` without a band).

    References:
        [EQ-INF-BAND-001], [EQ-STAT-QUANT-001].

    """

    label: str
    estimate: float
    lower: float
    upper: float

    def covers(self, value: float) -> bool:
        """Return whether the band contains ``value``.

        References:
            [EQ-INF-BAND-001].

        """
        return self.lower <= value <= self.upper


@dataclass(frozen=True)
class LabResults:
    """Every number the laboratory prints and draws.

    References:
        [REF-COLES-2001, Chapter 3], [REF-SILVERMAN-1986, Chapter 3].

    """

    data: WindData
    maxima_summary: ShapeSummary
    means_summary: ShapeSummary
    exact_moments: tuple[float, float, float]
    exact_design_value: float
    experts: dict[str, MaximumEntropy]
    expert_upper_ends: tuple[tuple[float, float, float], ...]
    candidates: tuple[CandidateResult, ...]
    means_candidates: tuple[CandidateResult, ...]
    chosen: CandidateResult
    grid: tuple[float, ...]
    chosen_band: ParameterUncertaintyBand
    normal_band: ParameterUncertaintyBand
    widths: tuple[tuple[int, float, float, float], ...]
    kernels: tuple[KernelResult, ...]
    joint: tuple[float, float, float]
    design: tuple[DesignRow, ...]


def run_analysis(
    seed: int = SEED, n_bootstrap: int = BOOTSTRAP_REPLICATES
) -> LabResults:
    """Run all six steps and collect their results.

    Args:
        seed: Seed of the data, the bootstrap refits, and the band draws.
        n_bootstrap: Bootstrap refits per candidate family.

    Returns:
        :class:`LabResults`.

    Equation:
        [EQ-LAB-INPUT-001], [EQ-INF-MLE-GEN-001], [EQ-INF-BAND-001].

    References:
        [REF-COLES-2001, Chapter 3], [REF-DELIGNETTE-MULLER-DUTANG-2015].

    """
    data = simulate_wind_data(seed)
    experts = expert_laws()
    upper_ends = []
    for upper in EXPERT_UPPER_ENDS:
        laws = expert_laws(upper)
        upper_ends.append(
            (
                upper,
                laws[EXPERT_LABELS[0]].quantile(DESIGN_PROBABILITY),
                laws[EXPERT_LABELS[1]].quantile(DESIGN_PROBABILITY),
            )
        )
    candidates = fit_candidates(data.maxima, n_bootstrap=n_bootstrap, seed=seed)
    means_candidates = fit_candidates(
        data.means, n_bootstrap=n_bootstrap, seed=seed, bootstrap_families=("Normal",)
    )
    chosen = choose_law(candidates)
    normal = next(item for item in candidates if item.family == "Normal")
    grid = band_grid()
    chosen_band = fitted_band(chosen.fit, grid, seed=seed)
    normal_band = fitted_band(normal.fit, grid, seed=seed)
    design = []
    for candidate in candidates:
        band = (
            chosen_band
            if candidate is chosen
            else normal_band
            if candidate is normal
            else fitted_band(candidate.fit, (PROBE_SPEED,), seed=seed)
        )
        design.append(
            DesignRow(
                f"{candidate.family} (likelihood)",
                band.quantile_estimate[0],
                band.quantile_lower[0],
                band.quantile_upper[0],
            )
        )
    moment_band = fitted_band(chosen.moments, (PROBE_SPEED,), seed=seed)
    design.append(
        DesignRow(
            f"{chosen.family} (moments)",
            moment_band.quantile_estimate[0],
            moment_band.quantile_lower[0],
            moment_band.quantile_upper[0],
        )
    )
    for label, law in experts.items():
        design.append(
            DesignRow(
                f"Maximum entropy ({label})",
                law.quantile(DESIGN_PROBABILITY),
                math.nan,
                math.nan,
            )
        )
    kernels = kernel_estimates(data.maxima)
    for kernel in kernels:
        design.append(
            DesignRow(f"KDE ({kernel.method})", kernel.design_value, math.nan, math.nan)
        )
    design.append(
        DesignRow(
            "Empirical quantile (type 7)",
            quantile(data.maxima, DESIGN_PROBABILITY),
            math.nan,
            math.nan,
        )
    )
    return LabResults(
        data=data,
        maxima_summary=describe(data.maxima),
        means_summary=describe(data.means),
        exact_moments=exact_maximum_moments(),
        exact_design_value=exact_maximum_quantile(DESIGN_PROBABILITY),
        experts=experts,
        expert_upper_ends=tuple(upper_ends),
        candidates=candidates,
        means_candidates=means_candidates,
        chosen=chosen,
        grid=grid,
        chosen_band=chosen_band,
        normal_band=normal_band,
        widths=envelope_widths(chosen.fit, seed=seed),
        kernels=kernels,
        joint=joint_factors(data.means, data.maxima),
        design=tuple(design),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _parameters(fit: FitResult | MomentFitResult) -> str:
    """Format estimates with their standard errors.

    References:
        [EQ-INF-MLE-GEN-002], [EQ-INF-MOM-004].

    """
    return ", ".join(
        f"{name} {value:.4g} ({error:.2g})"
        for name, value, error in zip(
            fit.parameter_names, fit.estimates, fit.standard_errors, strict=True
        )
    )


def _probe_index(grid: Sequence[float], x: float = PROBE_SPEED) -> int:
    """Return the index of the grid point closest to ``x``.

    References:
        [EQ-INF-BAND-001].

    """
    return min(range(len(grid)), key=lambda index: abs(grid[index] - x))


def print_report(results: LabResults) -> None:
    """Print the six sections of the laboratory.

    Args:
        results: Output of :func:`run_analysis`.

    References:
        [REF-COLES-2001, Chapter 3], [REF-CASELLA-BERGER-2002, Section 5.5.3].

    """
    n = BLOCK_COUNT
    print(
        f"Data: {n} blocks of {BLOCK_SIZE} Weibull(2, 8) speeds [m/s], seed {SEED};"
        "\nblock maxima and block means of the same speeds."
    )

    print("\n1. Describe")
    maxima, means = results.maxima_summary, results.means_summary
    rows = (
        ("mean", "mean", 3),
        ("sd (divisor n - 1)", "standard_deviation", 3),
        ("min", "minimum", 2),
        ("Q1 (type 7)", "first_quartile", 2),
        ("median", "median", 2),
        ("Q3 (type 7)", "third_quartile", 2),
        ("max", "maximum", 2),
        ("skewness g1 (type 1)", "skewness", 3),
        ("skewness b1 (type 3)", "skewness_type3", 3),
        ("excess kurtosis g2 (type 1)", "excess_kurtosis", 3),
        ("excess kurtosis b2 (type 3)", "excess_kurtosis_type3", 3),
    )
    print(f"  {'':<28}{'maxima':>9}{'means':>9}")
    for label, field, digits in rows:
        print(
            f"  {label:<28}{getattr(maxima, field):>9.{digits}f}"
            f"{getattr(means, field):>9.{digits}f}"
        )
    print(
        "g1 = m3/m2^1.5 and g2 = m4/m2^2 - 3 use divisor n (R's e1071 type 1).\n"
        "R's e1071::skewness and kurtosis default to type 3: b1 = g1 ((n-1)/n)^1.5,"
        "\nb2 = (g2 + 3)(1 - 1/n)^2 - 3; at n = 400 the conventions differ by "
        "less than 0.02."
    )
    exact_mean, exact_sd, exact_skew = results.exact_moments
    print(
        f"Exact law F(x)^{BLOCK_SIZE} of a maximum: mean {exact_mean:.3f}, "
        f"sd {exact_sd:.3f}, skewness {exact_skew:.3f}."
    )

    print("\n2. Expert statements only (maximum entropy)")
    low, high = EXPERT_RANGE
    print(
        f"Range [{low:g}, {high:g}] m/s, mean {EXPERT_MEAN:g}, sd "
        f"{EXPERT_STANDARD_DEVIATION:g}; optionally the Gumbel skewness "
        f"{GUMBEL_SKEWNESS:.4f}."
    )
    for label, law in results.experts.items():
        residual = max(abs(value) for value in law.residuals)
        print(
            f"  {label:<26} q0.99 {law.quantile(DESIGN_PROBABILITY):6.2f}  "
            f"pdf({low:g}) {law.pdf(low):.1e}  pdf({high:g}) "
            f"{law.pdf(high):.1e}  residual {residual:.0e}"
        )
    print(
        "  Upper end of the range:  "
        + "  ".join(f"{upper:g}" for upper, _, _ in results.expert_upper_ends)
    )
    print(
        "    q0.99 without skewness "
        + "  ".join(f"{value:.2f}" for _, value, _ in results.expert_upper_ends)
    )
    print(
        "    q0.99 with skewness    "
        + "  ".join(f"{value:.2f}" for _, _, value in results.expert_upper_ends)
    )
    print(
        f"Exact q0.99 {results.exact_design_value:.2f}. A mean and a variance on "
        "a range give a truncated normal shape:\nsymmetric, light-tailed, and "
        "insensitive to the range. A skewness constraint on a bounded range\n"
        "raises the density again near the upper end, so its tail quantiles "
        "follow the arbitrary bound.\nThe law encodes the statements, not the "
        "physics of maxima."
    )

    print("\n3. Some data: candidate families for the maxima")
    best = results.candidates[0].fit.aic
    print(f"Maximum-likelihood fits; best AIC {best:.1f}; KS and AD p-values")
    print(
        f"{'Family':<10}{'dAIC':>7}{'weight':>8}{'KS naive':>10}"
        f"{'KS boot':>9}{'AD naive':>10}{'AD boot':>9}"
    )
    for candidate in results.candidates:
        boot = candidate.bootstrap
        print(
            f"{candidate.family:<10}"
            f"{candidate.fit.aic - best:>7.1f}{candidate.akaike_weight:>8.3f}"
            f"{candidate.naive_ks_p:>10.3f}"
            f"{boot.ks_p_value if boot else math.nan:>9.3f}"
            f"{candidate.naive_ad_p:>10.3f}"
            f"{boot.ad_p_value if boot else math.nan:>9.3f}"
        )
    refits = results.chosen.bootstrap.n_bootstrap if results.chosen.bootstrap else 0
    print(
        f"(bootstrap: {refits} refits per family; the naive p-values\n"
        "treat the fitted law as fixed in advance and are too large)"
    )
    print("Estimates (standard error) by likelihood and by moments:")
    for candidate in results.candidates:
        kind = "closed form" if candidate.family in CLOSED_FORM_MOMENTS else "solved"
        print(f"  {candidate.family:<10}likelihood  {_parameters(candidate.fit)}")
        print(f"  {'':<10}moments     {_parameters(candidate.moments)} ({kind})")
    chosen = results.chosen
    runner = next(item for item in results.candidates if item is not chosen)
    rejected = [item.family for item in results.candidates if item.rejected()]
    print(
        f"Chosen: {chosen.family}. It has the best AIC among the laws that no "
        f"bootstrap test rejects\nat alpha = {ALPHA} (weight "
        f"{chosen.akaike_weight:.2f}), and the extremal types theorem "
        "anticipates a GEV law\nfor maxima of many values. Rejected: "
        + (", ".join(rejected) or "none")
        + f".\n{runner.family} is {abs(runner.fit.aic - chosen.fit.aic):.1f} AIC "
        f"units away and "
        + ("is rejected" if runner.rejected() else "is not rejected either")
        + ";\nthe two laws differ mostly in the upper tail (section 6)."
    )
    means_best = results.means_candidates[0]
    means_normal = next(
        item for item in results.means_candidates if item.family == "Normal"
    )
    boot = means_normal.bootstrap
    assert boot is not None
    print(
        "Block means: "
        + ", ".join(
            f"{item.family} {item.fit.aic - means_best.fit.aic:.1f}"
            for item in results.means_candidates
        )
        + " (dAIC);"
    )
    print(
        f"  normal bootstrap KS p {boot.ks_p_value:.3f}, AD p "
        f"{boot.ad_p_value:.3f}: a mean of {BLOCK_SIZE} values is near-normal\n"
        "  (central limit theorem), a maximum is not. Both results could be\n"
        "  anticipated from how the values were formed."
    )

    print("\n4. Parameter uncertainty")
    index = _probe_index(results.grid)
    print(
        f"Exceedance probability at {PROBE_SPEED:g} m/s [%] "
        f"({BAND_DRAWS} draws, {BAND_LEVEL:.0%} pointwise; observed "
        f"{sum(value > PROBE_SPEED for value in results.data.maxima)} of {n}):"
    )
    for label, band in (
        (results.chosen.family, results.chosen_band),
        ("Normal", results.normal_band),
    ):
        print(
            f"  {label:<7} estimate {100 * (1 - band.cdf_estimate[index]):.2f}  "
            f"band {100 * (1 - band.upper[index]):.2f}-"
            f"{100 * (1 - band.lower[index]):.2f}  envelope "
            f"{100 * (1 - band.envelope_upper[index]):.2f}-"
            f"{100 * (1 - band.envelope_lower[index]):.2f}  delta "
            f"{100 * (1 - band.delta_upper[index]):.2f}-"
            f"{100 * (1 - band.delta_lower[index]):.2f}  rejected "
            f"{band.rejected_draws}"
        )
    print(f"Band widths at {PROBE_SPEED:g} m/s for {results.chosen.family} [points]:")
    print(f"  {'draws':>6}{'quantile':>10}{'envelope':>10}{'delta':>8}")
    for draws, width, envelope, delta in results.widths:
        print(
            f"  {draws:>6}{100 * width:>10.2f}{100 * envelope:>10.2f}"
            f"{100 * delta:>8.2f}"
        )
    print(
        "The quantile band settles and agrees with the delta band; the "
        "envelope never narrows\nand widens whenever a more extreme draw "
        "arrives, so it measures the number of draws,\nnot the uncertainty."
    )

    print("\n5. Many data: kernel density estimates of the maxima")
    print(f"  {'bandwidth':<16}{'h [m/s]':>8}{'ISE':>10}{'q0.99':>8}")
    for kernel in results.kernels:
        print(
            f"  {kernel.method:<16}{kernel.bandwidth:>8.3f}{kernel.ise:>10.5f}"
            f"{kernel.design_value:>8.2f}"
        )
        if kernel.note:
            print(f"    {kernel.method}: {kernel.note}")
    corr, silverman, likelihood = results.joint
    print(
        "ISE against the exact density m F^(m-1) f. For this smooth unimodal "
        "law and this sample,\ncross-validation does not beat the "
        "normal-reference rule; its value is for shapes that rule\nmisses, "
        "such as several modes."
    )
    print(
        f"Joint (block mean, block maximum): correlation {corr:.2f}; "
        f"2-D kernel factor\n  Silverman {silverman:.3f}, LOO likelihood "
        f"{likelihood:.3f}. The two inputs are dependent:\n  a joint input "
        "model must keep their dependence."
    )

    print("\n6. Design value: the 1-in-100 block maximum (0.99 quantile)")
    exact = results.exact_design_value
    print(f"  {'input law':<44}{'q0.99':>7}{'95% band':>17}  contains exact")
    for row in results.design:
        band = (
            f"{row.lower:>8.2f}-{row.upper:<7.2f}"
            if math.isfinite(row.lower)
            else f"{'-':>16}"
        )
        covered = (
            ("yes" if row.covers(exact) else "no") if math.isfinite(row.lower) else ""
        )
        print(f"  {row.label:<44}{row.estimate:>7.2f}{band:>17}  {covered}")
    print(
        f"  Exact value F^-1(0.99^(1/{BLOCK_SIZE})) = {exact:.2f} m/s. The bands "
        "are conditional on their family:\n  a narrow band around the wrong "
        "family is confidently wrong in the tail."
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _legible(colour: str, minimum: float = 4.5) -> str:
    """Mix a series colour toward the ink until text in it is legible.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib.colors import to_hex, to_rgb

    palette = theme.active_palette()

    def luminance(rgb: Sequence[float]) -> float:
        """Return the relative luminance of an sRGB colour.

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        linear = [
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    canvas = luminance(to_rgb(palette.canvas))
    base, ink = to_rgb(colour), to_rgb(palette.ink)
    for step in range(21):
        weight = step / 20
        mixed = [(1 - weight) * b + weight * i for b, i in zip(base, ink, strict=True)]
        value = luminance(mixed)
        light, dark = max(value, canvas), min(value, canvas)
        if (light + 0.05) / (dark + 0.05) >= minimum:
            return str(to_hex(mixed))
    return palette.ink


def _label(
    ax: Any,
    x: float,
    y: float,
    text: str,
    colour: str,
    *,
    offset: tuple[float, float] = (6.0, 0.0),
    ha: str = "left",
    va: str = "center",
) -> None:
    """Write a direct label in a legible tone of its series colour.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    ax.annotate(
        text,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=theme.ANNOTATION_SIZE,
        fontweight="bold",
        color=_legible(colour),
        annotation_clip=False,
    )


def _plain_log(axis: Any, *, percent: bool = False) -> None:
    """Write plain numbers or percentages on a logarithmic axis.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator

    def plain(value: float, scale: float) -> str:
        """Return a positional decimal such as ``0.00001`` (no E notation).

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        return f"{scale * value:.12f}".rstrip("0").rstrip(".")

    if percent:
        axis.set_major_formatter(FuncFormatter(lambda v, _: plain(v, 100.0) + "%"))
    else:
        axis.set_major_formatter(FuncFormatter(lambda v, _: plain(v, 1.0)))
    axis.set_minor_formatter(NullFormatter())
    axis.set_minor_locator(NullLocator())


def _panels(count: int = 1, *, height: float = 4.3) -> tuple[Any, list[Any]]:
    """Return a themed figure with ``count`` axes side by side.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    plt = theme.pyplot()
    figure, grid = plt.subplots(1, count, figsize=(7.2, height), squeeze=False)
    axes = list(grid[0])
    for ax in axes:
        theme.style_axes(ax)
    if count > 1:
        figure.subplots_adjust(wspace=0.32)
    return figure, axes


def draw_description(results: LabResults) -> Any:
    """Draw standardized histograms of the maxima and the means.

    References:
        [EQ-STAT-SHAPE-001], [REF-COLES-2001, Section 3.1].

    """
    figure, (ax,) = _panels()
    edges = [-3.5 + 0.5 * k for k in range(17)]
    colours = {
        "block maxima": theme.role_colour("observed"),
        "block means": theme.role_colour("series-3"),
    }
    for (label, values), summary in zip(
        (("block maxima", results.data.maxima), ("block means", results.data.means)),
        (results.maxima_summary, results.means_summary),
        strict=True,
    ):
        scores = [(v - summary.mean) / summary.standard_deviation for v in values]
        ax.hist(
            scores,
            bins=edges,
            density=True,
            histtype="step",
            linewidth=theme.LINE_WIDTH,
            color=colours[label],
        )
    grid = [-3.5 + 0.02 * k for k in range(401)]
    normal = [math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) for z in grid]
    reference = theme.role_colour("reference")
    theme.line(
        ax,
        grid,
        normal,
        color=reference,
        dashes=theme.REFERENCE,
        width=theme.REFERENCE_WIDTH,
    )
    ax.set_xlim(-3.5, 4.5)
    ax.set_ylim(0.0, 0.62)
    _label(ax, 2.5, 0.055, "block maxima", colours["block maxima"],
           offset=(6.0, 10.0), va="bottom")  # fmt: skip
    _label(ax, 1.5, 0.225, "block means", colours["block means"],
           offset=(6.0, 0.0))  # fmt: skip
    _label(ax, 0.0, 0.43, "standard normal", reference, offset=(0.0, 14.0),
           ha="center", va="bottom")  # fmt: skip
    theme.axis_labels(
        ax, "Standardized value (x - mean) / sd", theme.axis_title("density")
    )
    theme.title_block(
        ax,
        f"Block maxima lean right (skewness {results.maxima_summary.skewness:.2f}); "
        f"block means do not ({results.means_summary.skewness:.2f})",
        f"Steps: density histograms of {BLOCK_COUNT} standardized values. "
        "Dashed: standard normal density",
    )
    return figure


def draw_expert_laws(results: LabResults) -> Any:
    """Draw the two maximum-entropy densities against the exact density.

    References:
        [EQ-DIST-MAXENT-001], [EQ-LAB-INPUT-001], [REF-JAYNES-1957].

    """
    figure, (ax,) = _panels()
    low, high = EXPERT_RANGE
    grid = [low + 0.05 * k for k in range(int(round((high - low) / 0.05)) + 1)]
    colours = (theme.role_colour("series-3"), theme.role_colour("model"))
    for law, colour in zip(results.experts.values(), colours, strict=True):
        theme.line(ax, grid, [max(law.pdf(x), 1e-12) for x in grid], color=colour)
    truth = [exact_maximum_pdf(x) for x in grid]
    theme.line(ax, grid, truth, entity="truth")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 0.5)
    ax.set_xlim(low, high + 1.0)
    ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 0.1])
    _plain_log(ax.yaxis)
    first, second = (law for law in results.experts.values())
    # The light-tailed law crosses 1e-4 on its right flank; its label sits
    # in the empty region to the left of that point.
    crossing = next(x for x in grid if x > EXPERT_MEAN and first.pdf(x) < 1e-4)
    _label(ax, crossing, 1e-4, "range, mean, sd", colours[0], offset=(-8.0, 0.0),
           ha="right")  # fmt: skip
    _label(ax, high, second.pdf(high), "+ Gumbel skewness", colours[1],
           offset=(0.0, 8.0), ha="right", va="bottom")  # fmt: skip
    _label(ax, 30.2, exact_maximum_pdf(30.2), "exact law",
           theme.role_colour("reference"), offset=(4.0, 10.0),
           va="bottom")  # fmt: skip
    theme.axis_labels(
        ax,
        "Block maximum of the wind speed",
        theme.axis_title("density"),
        x_unit="m/s",
    )
    q_first = first.quantile(DESIGN_PROBABILITY)
    q_second = second.quantile(DESIGN_PROBABILITY)
    theme.title_block(
        ax,
        f"Range, mean and sd alone put the 0.99 quantile at {q_first:.2f} m/s; a "
        f"Gumbel skewness moves it to {q_second:.2f} (exact "
        f"{results.exact_design_value:.2f})",
        f"Maximum-entropy densities on [{low:g}, {high:g}] m/s; the skewed one "
        f"rises again at {high:g}. Dashed: exact density of the maximum",
    )
    return figure


def draw_candidates(results: LabResults) -> Any:
    """Draw the fitted exceedance curves against the observed maxima.

    References:
        [EQ-INF-MLE-GEN-001], [EQ-STAT-005], [REF-COLES-2001, Chapter 3].

    """
    figure, (ax,) = _panels()
    sorted_maxima = sorted(results.data.maxima)
    n = len(sorted_maxima)
    grid = [13.0 + 0.05 * k for k in range(271)]
    floor = 1e-3
    colours = {
        results.chosen.family: theme.role_colour("model"),
        "Normal": theme.role_colour("series-3"),
    }
    ends = []
    for candidate in reversed(results.candidates):
        law = candidate.fit.distribution
        colour = colours.get(candidate.family, theme.role_colour("context"))
        exceed = [1.0 - law.cdf(x) for x in grid]
        visible = [value if value >= floor else math.nan for value in exceed]
        width = theme.LINE_WIDTH if candidate.family in colours else theme.CONTEXT_WIDTH
        theme.line(ax, grid, visible, color=colour, dashes=theme.FITTED, width=width)
        last = max(k for k, value in enumerate(visible) if not math.isnan(value))
        if candidate.family in colours:
            ends.append((grid[last], candidate.family, colour))
    theme.points(
        ax,
        sorted_maxima,
        [1.0 - (i + 0.5) / n for i in range(n)],
        entity="observations",
        size=14.0,
    )
    ax.set_yscale("log")
    ax.set_ylim(4.5e-4, 1.2)
    ax.set_xlim(13.0, 28.0)
    ax.set_yticks([0.001, 0.01, 0.1, 1.0])
    _plain_log(ax.yaxis, percent=True)
    # Each highlighted curve stops at the floor; its name hangs below its end.
    for x_end, name, colour in ends:
        _label(ax, x_end, 6.5e-4, name, colour, offset=(0.0, 0.0), ha="center")
    others = [item.family for item in results.candidates if item.family not in colours]
    theme.axis_labels(
        ax,
        "Block maximum of the wind speed",
        theme.axis_title("exceedance_probability"),
        x_unit="m/s",
    )
    best = results.candidates[0].fit.aic
    normal = next(item for item in results.candidates if item.family == "Normal")
    theme.title_block(
        ax,
        f"{results.chosen.family} ranks first by AIC; the normal law, "
        f"{normal.fit.aic - best:.1f} AIC units behind, falls below the "
        "largest maxima",
        f"Dots: {n} maxima at Hazen positions. Dashed: maximum-likelihood fits; "
        "grey: " + ", ".join(others),
    )
    return figure


def draw_parameter_bands(results: LabResults) -> Any:
    """Draw the exceedance bands and the growth of the envelope with draws.

    References:
        [EQ-INF-BAND-001], [EQ-INF-BAND-002],
        [REF-CASELLA-BERGER-2002, Section 5.5.4].

    """
    figure, (left, right) = _panels(2, height=4.4)
    floor = 1e-3
    colours = {
        results.chosen.family: theme.role_colour("model"),
        "Normal": theme.role_colour("series-3"),
    }
    grid = results.grid
    for name, band in (
        (results.chosen.family, results.chosen_band),
        ("Normal", results.normal_band),
    ):
        lower = [max(1.0 - value, floor) for value in band.upper]
        upper = [max(1.0 - value, floor) for value in band.lower]
        theme.band(left, grid, lower, upper, color=colours[name])
        estimate = [
            1.0 - value if 1.0 - value >= floor else math.nan
            for value in band.cdf_estimate
        ]
        theme.line(left, grid, estimate, color=colours[name])
    n = len(results.data.maxima)
    theme.points(
        left,
        sorted(results.data.maxima),
        [1.0 - (i + 0.5) / n for i in range(n)],
        entity="observations",
        size=10.0,
    )
    left.set_yscale("log")
    left.set_ylim(floor, 1.2)
    left.set_xlim(16.0, 27.0)
    left.set_yticks([0.001, 0.01, 0.1, 1.0])
    _plain_log(left.yaxis, percent=True)
    _label(left, 25.2, 1.0 - results.chosen.fit.distribution.cdf(25.2),
           results.chosen.family, colours[results.chosen.family],
           offset=(4.0, 10.0))  # fmt: skip
    _label(left, 23.0, 1.0 - results.normal_band.cdf_estimate[_probe_index(grid, 23.0)],
           "normal", colours["Normal"], offset=(-6.0, -8.0), ha="right",
           va="top")  # fmt: skip
    theme.axis_labels(
        left, "Block maximum", theme.axis_title("exceedance_probability"), x_unit="m/s"
    )
    draws = [row[0] for row in results.widths]
    quantile_width = [100.0 * row[1] for row in results.widths]
    envelope_width = [100.0 * row[2] for row in results.widths]
    delta_width = [100.0 * row[3] for row in results.widths]
    theme.line(right, draws, envelope_width, color=theme.role_colour("context"))
    theme.points(right, draws, envelope_width, color=theme.role_colour("context"),
                 size=18.0)  # fmt: skip
    theme.line(right, draws, quantile_width, color=colours[results.chosen.family])
    theme.points(right, draws, quantile_width, color=colours[results.chosen.family],
                 size=18.0)  # fmt: skip
    theme.line(right, draws, delta_width, color=theme.role_colour("reference"),
               dashes=theme.REFERENCE, width=theme.REFERENCE_WIDTH)  # fmt: skip
    right.set_xscale("log")
    _plain_log(right.xaxis)
    right.set_xticks([100, 1000, 10000])
    right.set_ylim(0.0, 1.15 * max(envelope_width))
    _label(right, draws[-1], envelope_width[-1], "min/max envelope",
           theme.role_colour("context"), offset=(-4.0, 10.0), ha="right",
           va="bottom")  # fmt: skip
    _label(right, draws[-1], quantile_width[-1], f"{results.chosen.family} 95% band",
           colours[results.chosen.family], offset=(-4.0, -12.0), ha="right",
           va="top")  # fmt: skip
    _label(right, draws[0], delta_width[0], "delta band",
           theme.role_colour("reference"), offset=(2.0, -12.0), ha="left",
           va="top")  # fmt: skip
    theme.axis_labels(right, "Parameter draws", f"Width at {PROBE_SPEED:g} m/s",
                      y_unit="points")  # fmt: skip
    first, last = results.widths[0], results.widths[-1]
    theme.title_block(
        left,
        f"At {PROBE_SPEED:g} m/s the {results.chosen.family} 95% band stays near "
        f"{100 * last[1]:.1f} points; the envelope grows from {100 * first[2]:.1f} "
        f"to {100 * last[2]:.1f}",
        f"Left: fits with pointwise 95% bands, {BAND_DRAWS} draws. Right: "
        f"{results.chosen.family} band widths against the number of draws",
    )
    return figure


def draw_kernel_estimates(results: LabResults) -> Any:
    """Draw the kernel estimates and their pointwise errors.

    References:
        [EQ-STAT-KDE-001], [EQ-LAB-INPUT-003], [REF-SILVERMAN-1986, Chapter 3].

    """
    figure, (left, right) = _panels(2, height=4.4)
    grid = [12.0 + 0.05 * k for k in range(321)]
    truth = [exact_maximum_pdf(x) for x in grid]
    # The selectors are methods, not ordered members: one categorical hue each.
    colours = [theme.role_colour(role) for role in ("model", "series-3", "context")][
        : len(results.kernels)
    ]
    theme.line(left, grid, truth, entity="truth")
    theme.reference_line(right, y=0.0)
    # Drawn widest first, so the narrowest bandwidth lies on top.
    for kernel, colour in reversed(list(zip(results.kernels, colours, strict=True))):
        estimate = gaussian_kde(results.data.maxima, bandwidth=kernel.bandwidth)
        values = [estimate.pdf(x) for x in grid]
        theme.line(left, grid, values, color=colour)
        theme.line(
            right,
            grid,
            [v - t for v, t in zip(values, truth, strict=True)],
            color=colour,
        )
    left.set_xlim(12.0, 28.0)
    left.set_ylim(0.0, 0.24)
    right.set_xlim(12.0, 28.0)
    _label(left, 21.5, exact_maximum_pdf(21.5), "exact", theme.role_colour("reference"),
           offset=(6.0, 6.0))  # fmt: skip
    for index, (kernel, colour) in enumerate(
        zip(results.kernels, colours, strict=True)
    ):
        right.annotate(
            f"{kernel.method}, h = {kernel.bandwidth:.2f}",
            xy=(1.0, 0.22 - 0.075 * index),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=theme.ANNOTATION_SIZE,
            fontweight="bold",
            color=_legible(colour),
        )
    theme.axis_labels(left, "Block maximum", theme.axis_title("density"), x_unit="m/s")
    theme.axis_labels(right, "Block maximum", "Estimate minus exact density",
                      x_unit="m/s")  # fmt: skip
    best = min(results.kernels, key=lambda kernel: kernel.ise)
    others = [kernel for kernel in results.kernels if kernel is not best]
    theme.title_block(
        left,
        f"The {best.method} bandwidth ({best.bandwidth:.2f} m/s) gives the smallest "
        f"integrated squared error, {best.ise:.5f}, against "
        + " and ".join(f"{kernel.ise:.5f}" for kernel in others),
        f"Gaussian kernel estimates of {BLOCK_COUNT} maxima. Left: densities; "
        "dashed: exact density. Right: pointwise error",
    )
    return figure


def draw_design_values(results: LabResults) -> Any:
    """Draw every 0.99-quantile estimate with its band against the exact value.

    References:
        [EQ-INF-BAND-001], [EQ-LAB-INPUT-001], [REF-COLES-2001, Section 3.3].

    """
    rows = list(reversed(results.design))
    figure, (ax,) = _panels(height=5.0)
    exact = results.exact_design_value
    theme.reference_line(ax, x=exact, entity="truth")
    for position, row in enumerate(rows):
        banded = math.isfinite(row.lower)
        colour = (
            theme.role_colour("model")
            if banded and row.covers(exact)
            else theme.role_colour("series-3")
            if banded
            else theme.role_colour("context")
        )
        if banded:
            ax.hlines(position, row.lower, row.upper, color=colour,
                      linewidth=theme.LINE_WIDTH, zorder=3)  # fmt: skip
        theme.points(ax, [row.estimate], [position], color=colour, size=36.0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row.label for row in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True, color=theme.active_palette().rule, linewidth=0.7)
    ends = [row.estimate for row in rows] + [
        value
        for row in rows
        for value in (row.lower, row.upper)
        if math.isfinite(value)
    ]
    low, high = min(ends), max(ends)
    ax.set_xlim(math.floor(low - 0.3), math.ceil(high + 0.3))
    _label(ax, exact, len(rows) - 0.5, f"exact {exact:.2f}",
           theme.role_colour("reference"), offset=(4.0, 0.0))  # fmt: skip
    theme.axis_labels(ax, "1-in-100 block maximum", "Input law", x_unit="m/s")
    covering = [
        row
        for row in results.design
        if math.isfinite(row.lower)
        and row.covers(exact)
        and row.label.endswith("(likelihood)")
    ]
    if len(covering) == 1:
        lead = (
            f"Only the {covering[0].label.split(' ')[0]} band "
            f"({covering[0].lower:.2f} to {covering[0].upper:.2f} m/s) contains the "
            f"exact 1-in-100 block maximum, {exact:.2f} m/s"
        )
    else:
        lead = (
            f"{len(covering)} likelihood bands contain the exact 1-in-100 block "
            f"maximum, {exact:.2f} m/s"
        )
    theme.title_block(
        ax,
        lead,
        "Dots: 0.99 quantile of each input law. Bars: 95% parameter bands of "
        "the fitted families; highlighted when they contain the exact value",
    )
    return figure


FIGURES: tuple[tuple[str, Callable[[LabResults], Any]], ...] = (
    ("input-modeling-description.png", draw_description),
    ("input-modeling-expert-laws.png", draw_expert_laws),
    ("input-modeling-candidates.png", draw_candidates),
    ("input-modeling-parameter-bands.png", draw_parameter_bands),
    ("input-modeling-kernel-estimates.png", draw_kernel_estimates),
    ("input-modeling-design-values.png", draw_design_values),
)


def write_figures(results: LabResults, output: Path) -> list[Path]:
    """Draw and save every figure of the laboratory.

    Args:
        results: Output of :func:`run_analysis`.
        output: Destination folder.

    Returns:
        The written paths.

    References:
        [REF-MATPLOTLIB].

    """
    plt = theme.pyplot()
    written = []
    with theme.chart_theme(theme.active_palette()):
        for name, draw in FIGURES:
            figure = draw(results)
            written.append(save_figure(figure, output / name))
            plt.close(figure)
    return written


def main() -> None:
    """Run the laboratory, print its report, and save its figures.

    References:
        [REF-COLES-2001, Chapter 3], [REF-MATPLOTLIB].

    """
    started = time.perf_counter()
    print(
        f"Fitting five families with {BOOTSTRAP_REPLICATES} bootstrap refits each "
        "(takes about a minute)..."
    )
    results = run_analysis()
    print_report(results)
    output = Path(__file__).parent / "generated"
    written = write_figures(results, output)
    chosen = next(
        row for row in results.design if row.label.startswith(results.chosen.family)
    )
    folder = output.relative_to(Path(__file__).parent.parent)
    print(
        f"\nResult: {results.chosen.family} chosen; its 95% band of the 1-in-100 "
        f"block maximum,\n{chosen.lower:.2f} to {chosen.upper:.2f} m/s, "
        f"{'contains' if chosen.covers(results.exact_design_value) else 'misses'} "
        f"the exact {results.exact_design_value:.2f} m/s."
    )
    print(
        f"Wrote {len(written)} figures in {time.perf_counter() - started:.0f} s. "
        f"Next: open {folder / FIGURES[-1][0]}"
    )


if __name__ == "__main__":
    main()
