"""Parity of probstat with R on fixed public datasets.

``tools/r_oracles/parity.R`` evaluates R 4.6.2 (base, recommended packages,
fitdistrplus, ismev, e1071) on literal copies of classic R datasets and writes
``tests/data/r_parity.json``: the R version, the package versions, the
datasets, and one record per case with the exact R call, the R values, the
tolerance, and its reason. This module repeats every case in probstat and
requires ``|probstat - R| <= absolute + relative |R|`` for every value. R is
not needed to run it; regenerate the file with
``Rscript tools/r_oracles/parity.R`` from the repository root.

References:
    [REF-R-CORE-TEAM], [REF-HYNDMAN-FAN-1996]. No new equation; each case
    names the probstat function whose equations it checks.

"""

from __future__ import annotations

import json
import math
import re
import unittest
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from probstat import distributions
from probstat.inference import (
    GAMMA_FAMILY,
    LOGNORMAL_FAMILY,
    WEIBULL_FAMILY,
    SmallExpectedCountWarning,
    adjust_p_values,
    bartlett_test,
    binomial_test,
    brown_forsythe_test,
    chi_square_goodness_of_fit,
    chi_square_independence_test,
    fisher_exact_test,
    fit_distribution,
    fit_distribution_moments,
    fit_glm,
    fit_linear_mixed_model,
    fit_linear_model,
    fit_negative_binomial_regression,
    fit_robust_linear_model,
    friedman_test,
    kolmogorov_smirnov_test,
    kolmogorov_smirnov_two_sample_test,
    kruskal_wallis_test,
    levene_test,
    linear_model_anova,
    mann_whitney_u_test,
    mcnemar_test,
    noncentral_t_cdf,
    one_sample_t_test,
    one_way_anova,
    paired_t_test,
    random_effect,
    shapiro_wilk_test,
    studentized_range_cdf,
    t_test_power,
    tukey_hsd_test,
    two_sample_t_test,
    wilcoxon_signed_rank_test,
    wilson_score_interval,
)
from probstat.inference.survival import (
    fit_cox_model,
    kaplan_meier,
    log_rank_test,
    survival_data,
)
from probstat.modeling import kendall_tau
from probstat.optimization import gauss_newton_covariance, least_squares
from probstat.risk import fit_gev
from probstat.statistics import (
    STUDENT_SLEEP,
    excess_kurtosis,
    interquartile_range,
    kde_bandwidth,
    median_absolute_deviation,
    pearson_correlation,
    quantiles,
    select_kde_bandwidth,
    skewness,
    spearman_correlation,
    standard_deviation,
    trimmed_mean,
    variance,
)
from probstat.timeseries import acf, fit_arima, pacf, yule_walker

REPOSITORY_ROOT = Path(__file__).parents[1]
PARITY_FILE = REPOSITORY_ROOT / "tests" / "data" / "r_parity.json"
GENERATOR = REPOSITORY_ROOT / "tools" / "r_oracles" / "parity.R"
GUIDE = REPOSITORY_ROOT / "docs" / "foundations" / "hypothesis-testing-and-bootstrap.md"
DOCUMENT: dict[str, Any] = json.loads(PARITY_FILE.read_text(encoding="utf-8"))
DATA: dict[str, Any] = DOCUMENT["datasets"]
CASES: list[dict[str, Any]] = DOCUMENT["cases"]

Values = Mapping[str, float | Sequence[float]]
Adapter = Callable[[dict[str, Any]], Values]
ADAPTERS: dict[str, Adapter] = {}


def adapter(*case_ids: str) -> Callable[[Adapter], Adapter]:
    """Register a function that computes the probstat side of some cases.

    Args:
        *case_ids: Ids of the JSON cases the function answers.

    Returns:
        A decorator that records the function under every id.

    References:
        [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

    """

    def register(function: Adapter) -> Adapter:
        for case_id in case_ids:
            if case_id in ADAPTERS:
                raise ValueError(f"duplicate adapter for {case_id}")
            ADAPTERS[case_id] = function
        return function

    return register


def column(dataset: str, name: str) -> list[float]:
    """Return a numeric column of an embedded dataset.

    References:
        [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

    """
    return [float(v) for v in DATA[dataset][name]]


def labels(dataset: str, name: str) -> list[str]:
    """Return a factor column of an embedded dataset as strings.

    References:
        [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

    """
    return [str(v) for v in DATA[dataset][name]]


def vector(dataset: str) -> list[float]:
    """Return an embedded numeric vector.

    References:
        [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

    """
    return [float(v) for v in DATA[dataset]]


def table(dataset: str) -> list[list[int]]:
    """Return an embedded contingency table as integer rows.

    References:
        [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

    """
    return [[int(v) for v in row] for row in DATA[dataset]]


def split(dataset: str, value: str, factor: str) -> dict[str, list[float]]:
    """Split a numeric column by a factor, keeping the order of first levels.

    References:
        [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

    """
    groups: dict[str, list[float]] = {}
    for y, g in zip(column(dataset, value), labels(dataset, factor), strict=True):
        groups.setdefault(g, []).append(y)
    return dict(sorted(groups.items()))


def indicator(values: Sequence[str], level: str) -> list[float]:
    """Return the treatment-contrast dummy of one factor level.

    References:
        [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

    """
    return [1.0 if v == level else 0.0 for v in values]


def rows(*columns: Sequence[float]) -> list[list[float]]:
    """Return design rows with a leading intercept column.

    References:
        [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

    """
    return [[1.0, *values] for values in zip(*columns, strict=True)]


def pair(interval: Any) -> list[float]:
    """Return ``[lower, upper]`` of a confidence interval object.

    References:
        [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

    """
    return [interval.lower, interval.upper]


# ------------------------------------------------------------ distributions


def law_values(case: dict[str, Any]) -> Values:
    """Evaluate a probstat law's method at the case's points.

    References:
        [REF-R-CORE-TEAM]. No new equation; the law's own equations apply.

    """
    spec = case["probstat"]
    law = getattr(distributions, spec["law"])(**spec["parameters"])
    method = getattr(law, spec["method"])
    discrete = spec["method"] == "pmf" or spec["law"] in {
        "Binomial", "Poisson", "NegativeBinomial", "Hypergeometric", "Geometric",
    }  # fmt: skip
    points = spec["points"]
    if discrete and spec["method"] != "quantile":
        return {"values": [method(int(x)) for x in points]}
    return {"values": [float(method(x)) for x in points]}


for _case in CASES:
    if _case["area"] == "distributions" and "law" in _case.get("probstat", {}):
        adapter(_case["id"])(law_values)


@adapter("dist.noncentral_t.cdf")
def _noncentral_t(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``dist.noncentral_t.cdf``."""
    spec = case["probstat"]
    return {
        "values": [
            noncentral_t_cdf(x, spec["df"], spec["noncentrality"])
            for x in spec["points"]
        ]
    }


@adapter("dist.studentized_range.cdf")
def _studentized_range(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``dist.studentized_range.cdf``."""
    spec = case["probstat"]
    return {
        "values": [
            studentized_range_cdf(q, spec["groups"], spec["df"]) for q in spec["points"]
        ]
    }


# -------------------------------------------------------------- descriptive


@adapter("descriptive.quantile.type1", "descriptive.quantile.type7")
def _quantile(case: dict[str, Any]) -> Values:
    """Compute the probstat side of the cases named in its decorator."""
    spec = case["probstat"]
    values = quantiles(
        column("sleep", "extra"), spec["probabilities"], method=spec["method"]
    )
    return {"values": values}


@adapter("descriptive.variance")
def _variance(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``descriptive.variance``."""
    mpg = column("mtcars", "mpg")
    return {
        "variance": variance(mpg, ddof=1),
        "standard_deviation": standard_deviation(mpg, ddof=1),
    }


@adapter("descriptive.mad")
def _mad(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``descriptive.mad``."""
    x = column("faithful", "eruptions")
    return {
        "raw": median_absolute_deviation(x, normal_consistent=False),
        "normal": median_absolute_deviation(x),
    }


@adapter("descriptive.iqr")
def _iqr(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``descriptive.iqr``."""
    return {"iqr": interquartile_range(column("faithful", "waiting"))}


@adapter("descriptive.trimmed_mean")
def _trimmed_mean(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``descriptive.trimmed_mean``."""
    return {"trimmed_mean": trimmed_mean(column("chickwts", "weight"), 0.1)}


@adapter("descriptive.cor.pearson")
def _pearson(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``descriptive.cor.pearson``."""
    return {
        "correlation": pearson_correlation(
            column("mtcars", "mpg"), column("mtcars", "wt")
        )
    }


@adapter("descriptive.cor.spearman")
def _spearman(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``descriptive.cor.spearman``."""
    return {
        "correlation": spearman_correlation(
            column("mtcars", "mpg"), column("mtcars", "hp")
        )
    }


@adapter("descriptive.cor.kendall")
def _kendall(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``descriptive.cor.kendall``."""
    return {"correlation": kendall_tau(column("mtcars", "mpg"), column("mtcars", "hp"))}


@adapter("descriptive.skewness.type1")
def _skewness(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``descriptive.skewness.type1``."""
    return {"skewness": skewness(column("faithful", "eruptions"))}


@adapter("descriptive.kurtosis.type1")
def _kurtosis(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``descriptive.kurtosis.type1``."""
    return {"excess_kurtosis": excess_kurtosis(column("faithful", "eruptions"))}


@adapter("descriptive.bandwidth.nrd0", "descriptive.bandwidth.nrd")
def _bandwidth_rule(case: dict[str, Any]) -> Values:
    """Compute the probstat side of the cases named in its decorator."""
    rule = case["probstat"]["rule"]
    return {"bandwidth": kde_bandwidth(column("faithful", "eruptions"), rule=rule)}


@adapter("descriptive.bandwidth.ucv", "descriptive.bandwidth.bcv")
def _bandwidth_cv(case: dict[str, Any]) -> Values:
    """Compute the probstat side of the cases named in its decorator."""
    selection = select_kde_bandwidth(
        column("faithful", "eruptions"), method=case["probstat"]["method"]
    )
    if selection.at_boundary:
        raise AssertionError("an end-point minimum would only compare search bounds")
    return {"bandwidth": selection.bandwidth}


# -------------------------------------------------------------------- tests


def sleep_group(level: str) -> list[float]:
    """Return the extra sleep of one drug group of ``sleep``.

    References:
        [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

    """
    return split("sleep", "extra", "group")[level]


def t_values(result: Any) -> dict[str, Any]:
    """Return the R-named numeric parts of a t-test result.

    References:
        [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

    """
    return {
        "statistic": result.statistic,
        "df": result.df,
        "p_value": result.p_value,
        "conf_int": pair(result.confidence_interval),
        "standard_error": result.standard_error,
    }


@adapter("tests.t.one_sample")
def _t_one(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.t.one_sample``."""
    return t_values(one_sample_t_test(sleep_group("2"), null_mean=1.0))


@adapter("tests.t.welch", "tests.t.pooled")
def _t_two(case: dict[str, Any]) -> Values:
    """Compute the probstat side of the cases named in its decorator."""
    groups = split("ToothGrowth", "len", "supp")
    pooled = case["id"] == "tests.t.pooled"
    return t_values(
        two_sample_t_test(groups["OJ"], groups["VC"], equal_variances=pooled)
    )


@adapter("tests.t.paired")
def _t_paired(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.t.paired``."""
    return t_values(paired_t_test(sleep_group("1"), sleep_group("2")))


@adapter("tests.t.power")
def _t_power(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.t.power``."""
    return {
        "one_sample": t_test_power(0.6, 20),
        "two_sample": t_test_power(0.8, 15, design="two-sample"),
    }


@adapter("tests.wilcoxon.signed_rank_exact")
def _signed_rank_exact(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.wilcoxon.signed_rank_exact``."""
    result = wilcoxon_signed_rank_test(
        vector("depression_x"),
        vector("depression_y"),
        alternative="greater",
        method="exact",
    )
    return {"statistic": result.statistic, "p_value": result.p_value}


@adapter("tests.wilcoxon.signed_rank_normal")
def _signed_rank_normal(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.wilcoxon.signed_rank_normal``."""
    result = wilcoxon_signed_rank_test(
        sleep_group("2"), sleep_group("1"), method="normal", continuity_correction=True
    )
    return {"statistic": result.statistic, "p_value": result.p_value}


@adapter("tests.wilcoxon.rank_sum_exact")
def _rank_sum_exact(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.wilcoxon.rank_sum_exact``."""
    result = mann_whitney_u_test(
        vector("permeability_x"),
        vector("permeability_y"),
        alternative="greater",
        method="exact",
    )
    return {"statistic": result.statistic, "p_value": result.p_value}


@adapter("tests.wilcoxon.rank_sum_normal")
def _rank_sum_normal(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.wilcoxon.rank_sum_normal``."""
    groups = split("InsectSprays", "count", "spray")
    result = mann_whitney_u_test(
        groups["C"], groups["D"], method="normal", continuity_correction=True
    )
    return {"statistic": result.statistic, "p_value": result.p_value}


@adapter("tests.binomial")
def _binomial(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.binomial``."""
    result = binomial_test(682, 925, null_probability=0.75)
    return {"p_value": result.p_value, "conf_int": pair(result.confidence_interval)}


@adapter("tests.wilson_interval")
def _wilson(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.wilson_interval``."""
    return {"conf_int": pair(wilson_score_interval(15, 40))}


@adapter("tests.fisher.greater")
def _fisher_greater(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.fisher.greater``."""
    return {
        "p_value": fisher_exact_test(
            table("tea_tasting"), alternative="greater"
        ).p_value
    }


@adapter("tests.fisher.two_sided")
def _fisher_two_sided(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.fisher.two_sided``."""
    return {"p_value": fisher_exact_test(table("convictions")).p_value}


def chi_values(result: Any) -> dict[str, Any]:
    """Return the R-named numeric parts of a chi-square-type result.

    References:
        [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

    """
    return {"statistic": result.statistic, "df": result.df, "p_value": result.p_value}


@adapter("tests.chisq.independence")
def _chisq_independence(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.chisq.independence``."""
    return chi_values(chi_square_independence_test(table("party_by_gender")))


@adapter("tests.chisq.yates")
def _chisq_yates(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.chisq.yates``."""
    return chi_values(
        chi_square_independence_test(table("ucb_admissions"), yates_correction=True)
    )


@adapter("tests.chisq.goodness_of_fit")
def _chisq_gof(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.chisq.goodness_of_fit``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SmallExpectedCountWarning)
        result = chi_square_goodness_of_fit(
            [89, 37, 30, 28, 2], [0.40, 0.20, 0.20, 0.19, 0.01]
        )
    return chi_values(result)


@adapter("tests.mcnemar")
def _mcnemar(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.mcnemar``."""
    result = mcnemar_test(table("performance"), exact=False, continuity_correction=True)
    return {"statistic": result.statistic, "p_value": result.p_value}


@adapter("tests.ks.one_sample_exact")
def _ks_exact(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.ks.one_sample_exact``."""
    result = kolmogorov_smirnov_test(
        vector("depression_x"), distributions.Normal(1.5, 0.6)
    )
    return {"statistic": result.statistic, "p_value": result.p_value}


@adapter("tests.ks.one_sample_statistic")
def _ks_statistic(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.ks.one_sample_statistic``."""
    result = kolmogorov_smirnov_test(
        column("faithful", "eruptions"), distributions.Normal(3.5, 1.1)
    )
    return {"statistic": result.statistic}


@adapter("tests.ks.two_sample_exact")
def _ks_two(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.ks.two_sample_exact``."""
    result = kolmogorov_smirnov_two_sample_test(
        vector("permeability_x"), vector("permeability_y"), exact=True
    )
    return {"statistic": result.statistic, "p_value": result.p_value}


@adapter("tests.shapiro.small", "tests.shapiro.large")
def _shapiro(case: dict[str, Any]) -> Values:
    """Compute the probstat side of the cases named in its decorator."""
    if case["id"] == "tests.shapiro.small":
        values = column("PlantGrowth", "weight")
    else:
        values = column("faithful", "waiting")
    result = shapiro_wilk_test(values)
    return {"statistic": result.statistic, "p_value": result.p_value}


@adapter("tests.kruskal.chickwts", "tests.kruskal.insectsprays")
def _kruskal(case: dict[str, Any]) -> Values:
    """Compute the probstat side of the cases named in its decorator."""
    if case["id"] == "tests.kruskal.chickwts":
        groups = split("chickwts", "weight", "feed")
    else:
        groups = split("InsectSprays", "count", "spray")
    return chi_values(kruskal_wallis_test(*groups.values(), method="chi-square"))


@adapter("tests.friedman")
def _friedman(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.friedman``."""
    blocks = [[float(v) for v in row] for row in DATA["rounding_times"]]
    return chi_values(friedman_test(blocks))


@adapter("tests.bartlett.plantgrowth", "tests.bartlett.chickwts")
def _bartlett(case: dict[str, Any]) -> Values:
    """Compute the probstat side of the cases named in its decorator."""
    if case["id"] == "tests.bartlett.plantgrowth":
        groups = split("PlantGrowth", "weight", "group")
    else:
        groups = split("chickwts", "weight", "feed")
    return chi_values(bartlett_test(*groups.values()))


@adapter("tests.levene.mean")
def _levene(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.levene.mean``."""
    groups = split("PlantGrowth", "weight", "group")
    return chi_values(levene_test(*groups.values(), center="mean"))


@adapter("tests.levene.median")
def _brown_forsythe(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.levene.median``."""
    groups = split("chickwts", "weight", "feed")
    return chi_values(brown_forsythe_test(*groups.values()))


@adapter("tests.anova.one_way")
def _anova(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.anova.one_way``."""
    result = one_way_anova(*split("PlantGrowth", "weight", "group").values())
    return {
        "statistic": result.statistic,
        "df": [result.df_between, result.df_within],
        "p_value": result.p_value,
    }


def tukey_rows() -> list[Any]:
    """Return probstat's Tukey comparisons in R's order (2-1, 3-1, 3-2).

    References:
        [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

    """
    result = tukey_hsd_test(*split("PlantGrowth", "weight", "group").values())
    by_pair = {(c.first, c.second): c for c in result.comparisons}
    return [by_pair[(0, 1)], by_pair[(0, 2)], by_pair[(1, 2)]]


@adapter("tests.tukey.difference_and_p")
def _tukey(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.tukey.difference_and_p``."""
    comparisons = tukey_rows()
    return {
        "difference": [-c.difference for c in comparisons],
        "p_value": [c.p_value for c in comparisons],
    }


@adapter("tests.tukey.interval")
def _tukey_interval(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``tests.tukey.interval``."""
    comparisons = tukey_rows()
    return {
        "lower": [-c.confidence_interval.upper for c in comparisons],
        "upper": [-c.confidence_interval.lower for c in comparisons],
    }


@adapter(
    "tests.p_adjust.bonferroni", "tests.p_adjust.holm", "tests.p_adjust.bh",
    "tests.p_adjust.by",
)  # fmt: skip
def _p_adjust(case: dict[str, Any]) -> Values:
    """Compute the probstat side of the cases named in its decorator."""
    method = case["probstat"]["method"]
    result = adjust_p_values(vector("p_values"), method=method)
    return {"adjusted": result.adjusted_p_values}


# --------------------------------------------------------------- regression


def mtcars_fit() -> Any:
    """Return ``lm(mpg ~ wt + hp, mtcars)`` in probstat.

    References:
        [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

    """
    design = rows(column("mtcars", "wt"), column("mtcars", "hp"))
    return fit_linear_model(design, column("mtcars", "mpg"))


@adapter("regression.lm.coefficients")
def _lm_coefficients(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``regression.lm.coefficients``."""
    fit = mtcars_fit()
    return {
        "coefficients": fit.coefficients,
        "standard_errors": fit.standard_errors,
        "t_values": fit.t_values,
        "p_values": fit.p_values,
    }


@adapter("regression.lm.summary")
def _lm_summary(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``regression.lm.summary``."""
    fit = mtcars_fit()
    return {
        "r_squared": fit.r_squared,
        "adjusted_r_squared": fit.adjusted_r_squared,
        "f_statistic": fit.f_statistic,
        "sigma": fit.residual_standard_error,
    }


@adapter("regression.lm.information")
def _lm_information(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``regression.lm.information``."""
    fit = mtcars_fit()
    return {"log_likelihood": fit.log_likelihood, "aic": fit.aic, "bic": fit.bic}


@adapter("regression.lm.hatvalues")
def _lm_leverage(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``regression.lm.hatvalues``."""
    return {"leverage": mtcars_fit().leverage}


@adapter("regression.lm.residuals")
def _lm_residuals(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``regression.lm.residuals``."""
    fit = mtcars_fit()
    return {
        "standardized": fit.standardized_residuals,
        "studentized": fit.studentized_residuals,
        "cooks_distance": fit.cooks_distance,
    }


@adapter("regression.anova.sequential")
def _anova_sequential(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``regression.anova.sequential``."""
    cyl = [str(int(v)) for v in column("mtcars", "cyl")]
    design = rows(column("mtcars", "wt"), indicator(cyl, "6"), indicator(cyl, "8"))
    anova = linear_model_anova(
        design,
        column("mtcars", "mpg"),
        {"wt": [1], "factor(cyl)": [2, 3]},
        kind="I",
    )
    terms = [row for row in anova.rows if row.term != "Residuals"]
    residual = next(row for row in anova.rows if row.term == "Residuals")
    return {
        "df": [*(row.df for row in terms), residual.df],
        "sum_squares": [*(row.sum_squares for row in terms), residual.sum_squares],
        "f_statistic": [row.f_statistic for row in terms],
        "p_value": [row.p_value for row in terms],
    }


def glm_values(fit: Any) -> dict[str, Any]:
    """Return the R-named numeric parts of a GLM fit.

    References:
        [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

    """
    return {
        "coefficients": fit.coefficients,
        "standard_errors": fit.standard_errors,
        "deviance": fit.deviance,
        "null_deviance": fit.null_deviance,
        "dispersion": fit.dispersion,
        "aic": fit.aic,
    }


@adapter("regression.glm.binomial")
def _glm_binomial(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``regression.glm.binomial``."""
    fit = fit_glm(
        rows(column("mtcars", "wt")), column("mtcars", "am"), family="binomial"
    )
    return glm_values(fit)


@adapter("regression.glm.binomial_trials")
def _glm_binomial_trials(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``regression.glm.binomial_trials``."""
    cases = column("esoph", "ncases")
    totals = [a + b for a, b in zip(cases, column("esoph", "ncontrols"), strict=True)]
    fit = fit_glm(
        rows(column("esoph", "agegp"), column("esoph", "alcgp")),
        [a / n for a, n in zip(cases, totals, strict=True)],
        family="binomial",
        weights=totals,
    )
    return glm_values(fit)


def warpbreaks_design() -> list[list[float]]:
    """Return the treatment-contrast design of ``breaks ~ wool + tension``.

    References:
        [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

    """
    wool = labels("warpbreaks", "wool")
    tension = labels("warpbreaks", "tension")
    return rows(indicator(wool, "B"), indicator(tension, "M"), indicator(tension, "H"))


@adapter("regression.glm.poisson")
def _glm_poisson(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``regression.glm.poisson``."""
    fit = fit_glm(warpbreaks_design(), column("warpbreaks", "breaks"), family="poisson")
    return glm_values(fit)


@adapter("regression.glm.gamma_log", "regression.glm.gamma_inverse")
def _glm_gamma(case: dict[str, Any]) -> Values:
    """Compute the probstat side of the cases named in its decorator."""
    link = "log" if case["id"] == "regression.glm.gamma_log" else "inverse"
    fit = fit_glm(
        rows(column("airquality", "Temp")),
        column("airquality", "Ozone"),
        family="gamma",
        link=link,
    )
    return glm_values(fit)


@adapter("regression.glm_nb", "regression.glm_nb.standard_errors")
def _glm_nb(case: dict[str, Any]) -> Values:
    """Compute the probstat side of the cases named in its decorator."""
    fit = fit_negative_binomial_regression(
        warpbreaks_design(), column("warpbreaks", "breaks")
    )
    return {
        "coefficients": fit.coefficients,
        "theta": fit.theta,
        "log_likelihood": fit.log_likelihood,
        "standard_errors": fit.standard_errors,
        "theta_standard_error": fit.theta_standard_error,
    }


@adapter("regression.rlm_huber")
def _rlm(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``regression.rlm_huber``."""
    design = rows(
        column("stackloss", "Air.Flow"),
        column("stackloss", "Water.Temp"),
        column("stackloss", "Acid.Conc."),
    )
    fit = fit_robust_linear_model(
        design, column("stackloss", "stack.loss"), psi="huber"
    )
    return {"coefficients": fit.coefficients, "scale": fit.scale}


@adapter("regression.nls")
def _nls(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``regression.nls``."""
    conc = column("Puromycin", "conc")
    rate = column("Puromycin", "rate")

    def residual(theta: Sequence[float]) -> list[float]:
        vm, k = theta
        return [r - vm * c / (k + c) for c, r in zip(conc, rate, strict=True)]

    def jacobian(theta: Sequence[float]) -> list[list[float]]:
        vm, k = theta
        return [[-c / (k + c), vm * c / (k + c) ** 2] for c in conc]

    result = least_squares(residual, [200.0, 0.05], jacobian=jacobian, tolerance=1e-15)
    covariance = gauss_newton_covariance(result)
    rss = sum(r * r for r in result.residuals)
    return {
        "estimates": result.x,
        "standard_errors": [math.sqrt(covariance[i][i]) for i in range(2)],
        "sigma": math.sqrt(rss / (len(conc) - 2)),
    }


@adapter("regression.lme_reml", "regression.lme_reml.closed_form")
def _lme(case: dict[str, Any]) -> Values:
    """Compute the probstat side of the cases named in its decorator."""
    group = labels("sleep", "group")
    fit = fit_linear_mixed_model(
        rows(indicator(group, "2")),
        column("sleep", "extra"),
        random_effect(labels("sleep", "ID"), name="ID"),
        method="REML",
    )
    return {
        "coefficients": fit.coefficients,
        "standard_errors": fit.standard_errors,
        "residual_variance": fit.residual_variance,
        "intercept_variance": fit.variance_components[0].covariance[0][0],
        "log_likelihood": fit.log_likelihood,
    }


# ----------------------------------------------------------------- survival


def aml_data(level: str | None = None) -> Any:
    """Return the ``aml`` sample, optionally one maintenance arm.

    References:
        [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

    """
    keep = [level is None or x == level for x in labels("aml", "x")]
    times = [t for t, k in zip(column("aml", "time"), keep, strict=True) if k]
    events = [int(s) for s, k in zip(column("aml", "status"), keep, strict=True) if k]
    return survival_data(times, events)


@adapter("survival.kaplan_meier", "survival.kaplan_meier.interval")
def _kaplan_meier(case: dict[str, Any]) -> Values:
    """Compute the probstat side of the cases named in its decorator."""
    estimate = kaplan_meier(aml_data())
    return {
        "times": estimate.times,
        "at_risk": estimate.at_risk,
        "events": estimate.events,
        "survival": estimate.survival,
        "standard_error": [math.sqrt(v) for v in estimate.variance],
        "lower": estimate.lower,
        "upper": estimate.upper,
    }


@adapter("survival.log_rank")
def _log_rank(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``survival.log_rank``."""
    result = log_rank_test(aml_data("Maintained"), aml_data("Nonmaintained"))
    return {
        "statistic": result.statistic,
        "observed": result.observed,
        "expected": result.expected,
        "p_value": result.p_value,
    }


@adapter("survival.cox.efron", "survival.cox.breslow")
def _cox(case: dict[str, Any]) -> Values:
    """Compute the probstat side of the cases named in its decorator."""
    ties = "efron" if case["id"] == "survival.cox.efron" else "breslow"
    covariates = [[v] for v in indicator(labels("aml", "x"), "Nonmaintained")]
    fit = fit_cox_model(aml_data(), covariates, ties=ties)
    return {
        "coefficients": fit.coefficients,
        "standard_errors": fit.standard_errors,
        "log_partial_likelihood": fit.log_partial_likelihood,
        "null_log_partial_likelihood": fit.null_log_partial_likelihood,
        "score_statistic": fit.score_statistic,
    }


# -------------------------------------------------------------- time series


@adapter("timeseries.acf")
def _acf(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``timeseries.acf``."""
    return {"values": acf(vector("lh"), 10)}


@adapter("timeseries.pacf")
def _pacf(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``timeseries.pacf``."""
    return {"values": pacf(vector("lh"), 10)}


@adapter("timeseries.yule_walker")
def _yule_walker(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``timeseries.yule_walker``."""
    return {"coefficients": yule_walker(vector("lh"), 3).coefficients}


ARIMA_MODELS = {
    "ar1": ("lh", (1, 0, 0)),
    "arma11": ("LakeHuron", (1, 0, 1)),
    "ima11": ("LakeHuron", (0, 1, 1)),
}


@adapter(
    *(f"timeseries.arima.{name}" for name in ARIMA_MODELS),
    *(f"timeseries.arima.{name}.log_likelihood" for name in ARIMA_MODELS),
)
def _arima(case: dict[str, Any]) -> Values:
    """Compute the probstat side of the cases named in its decorator."""
    series, order = ARIMA_MODELS[case["id"].split(".")[2]]
    fit = fit_arima(vector(series), order)
    coefficients = [*fit.ar, *fit.ma]
    if order[1] == 0:
        coefficients.append(fit.mean)
    return {
        "coefficients": coefficients,
        "sigma2": fit.variance,
        "log_likelihood": fit.log_likelihood,
    }


# ------------------------------------------------------- distribution fitting

FAMILIES = {"gamma": GAMMA_FAMILY, "weibull": WEIBULL_FAMILY, "lnorm": LOGNORMAL_FAMILY}


@adapter(
    "fitting.gamma.mle", "fitting.gamma.mle.profile",
    "fitting.gamma.mle.standard_errors", "fitting.weibull.mle",
    "fitting.weibull.mle.profile", "fitting.lnorm.mle",
    "fitting.lnorm.mle.standard_errors",
)  # fmt: skip
def _fit_mle(case: dict[str, Any]) -> Values:
    """Compute the probstat side of the cases named in its decorator."""
    family = FAMILIES[case["id"].split(".")[1]]
    fit = fit_distribution(column("airquality", "Ozone"), family)
    return {
        "estimates": fit.estimates,
        "standard_errors": fit.standard_errors,
        "log_likelihood": fit.log_likelihood,
    }


@adapter("fitting.gamma.mme")
def _fit_mme(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``fitting.gamma.mme``."""
    return {
        "estimates": fit_distribution_moments(
            column("airquality", "Ozone"), GAMMA_FAMILY
        ).estimates
    }


@adapter("fitting.gev.mle")
def _fit_gev(case: dict[str, Any]) -> Values:
    """Compute the probstat side of case ``fitting.gev.mle``."""
    fit = fit_gev(vector("portpirie")).fit
    return {
        "estimates": fit.estimates,
        "standard_errors": fit.standard_errors,
        "log_likelihood": fit.log_likelihood,
    }


# --------------------------------------------------------------------- tests


def as_list(value: float | Sequence[float]) -> list[float]:
    """Return a scalar or sequence as a list of floats.

    References:
        [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

    """
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(v) for v in value]


class RParityTests(unittest.TestCase):
    """Every recorded R value is reproduced within its stated tolerance."""

    def test_every_case_matches_r(self) -> None:
        """``|probstat - R| <= absolute + relative |R|`` for every value.

        References:
            [REF-R-CORE-TEAM]. Each case checks its target's own equations.

        """
        for case in CASES:
            with self.subTest(case=case["id"]):
                actual = ADAPTERS[case["id"]](case)
                relative = case["tolerance"]["relative"]
                absolute = case["tolerance"]["absolute"]
                for key, expected in case["r_value"].items():
                    got = as_list(actual[key])
                    self.assertEqual(len(got), len(expected), key)
                    for index, (p, r) in enumerate(zip(got, expected, strict=True)):
                        self.assertLessEqual(
                            abs(p - r),
                            absolute + relative * abs(r),
                            f"{key}[{index}]: probstat {p!r}, R {r!r}",
                        )

    def test_cases_and_adapters_correspond(self) -> None:
        """Every case has exactly one adapter and every adapter has a case.

        References:
            [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

        """
        ids = [case["id"] for case in CASES]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sorted(ADAPTERS), sorted(ids))

    def test_records_are_complete_and_tolerances_justified(self) -> None:
        """Each record names its target, R call, tolerance, and reason.

        References:
            [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

        """
        self.assertEqual(DOCUMENT["case_count"], len(CASES))
        self.assertTrue(DOCUMENT["r_version"].startswith("R version "))
        self.assertTrue(GENERATOR.is_file())
        for case in CASES:
            with self.subTest(case=case["id"]):
                self.assertTrue(case["target"].startswith("probstat."))
                self.assertTrue(case["r_call"].strip())
                self.assertGreater(len(case["tolerance_reason"]), 20)
                tolerance = case["tolerance"]
                self.assertGreaterEqual(tolerance["relative"], 0.0)
                self.assertGreaterEqual(tolerance["absolute"], 0.0)
                self.assertLessEqual(tolerance["relative"], 1e-3)
                self.assertLessEqual(tolerance["absolute"], 1e-4)
                self.assertTrue(case["r_value"])
        for entry in DOCUMENT["skipped"]:
            self.assertGreater(len(entry["reason"]), 20)

    def test_embedded_sleep_data_matches_the_library_copy(self) -> None:
        """R's ``sleep`` equals ``probstat.statistics.STUDENT_SLEEP``.

        References:
            [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

        """
        columns = {c.name: c.values for c in STUDENT_SLEEP.columns}
        self.assertEqual(tuple(sleep_group("1")), tuple(columns["drug_1"]))
        self.assertEqual(tuple(sleep_group("2")), tuple(columns["drug_2"]))

    def test_targets_resolve(self) -> None:
        """Every recorded probstat target names an importable object.

        References:
            [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

        """
        import importlib

        for case in CASES:
            with self.subTest(case=case["id"]):
                parts = case["target"].split(".")
                for split_at in range(len(parts) - 1, 0, -1):
                    try:
                        obj: Any = importlib.import_module(".".join(parts[:split_at]))
                    except ModuleNotFoundError:
                        continue
                    for name in parts[split_at:]:
                        obj = getattr(obj, name)
                    break
                else:
                    self.fail(case["target"])

    def test_guide_quotes_the_number_of_cases(self) -> None:
        """The "Agreement with R" section states the current case count.

        References:
            [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

        """
        text = GUIDE.read_text(encoding="utf-8")
        section = text.partition("## Agreement with R")[2].partition("\n## ")[0]
        counts = re.findall(r"(\d+) parity cases", section)
        self.assertEqual(counts, [str(len(CASES))])

    def test_readme_quotes_the_number_of_cases(self) -> None:
        """The README trust table states the current case count.

        References:
            [REF-R-CORE-TEAM]. No mathematical equation is evaluated.

        """
        readme = (GUIDE.parents[2] / "README.md").read_text(encoding="utf-8")
        self.assertEqual(
            re.findall(r"(\d+) cases checked against R", readme), [str(len(CASES))]
        )


if __name__ == "__main__":
    unittest.main()
