"""Real-data fitting laboratory: five historical data sets fitted end to end.

Every other fitting laboratory draws seeded synthetic data. This one fits the
bundled historical records of :mod:`probstat.statistics.datasets`: deaths by
horse kick (Bortkiewicz 1898), alpha-particle counts (Rutherford and Geiger
1910), Newcomb's light passage times (Stigler 1977, Table 5), ball-bearing
endurance (Lieblein and Zelen 1956), and Student's sleep data (1908). Each
case states its candidate families, ranks them by AIC and BIC, calibrates
goodness of fit for estimated parameters, and reports confidence intervals.

Definitions:
    Count data are non-negative integers per unit (corps-year, 7.5 s
    interval); a frequency table lists how many units showed each count.
    The negative binomial has mean ``mu`` and variance ``mu + mu^2 / r``;
    ``r -> infinity`` is the Poisson law. A hanging rootogram hangs bars of
    length ``sqrt(observed)`` from the curve ``sqrt(expected)``. A calibrated
    p-value accounts for the parameters estimated from the tested data.

Assumptions:
    Units are independent and identically distributed within each data set
    (the corps-years of Bortkiewicz's ten pooled corps, the successive
    intervals of Rutherford and Geiger, Newcomb's 66 repetitions, the 23
    bearings of one test group, Student's ten patients). Newcomb's values are
    deviations in nanoseconds from 24800 ns; 33.02 is the deviation implied by
    the accepted speed of light (Stigler 1977). The sleep data are paired by
    patient.

Equations:
    [EQ-INF-COUNT-001] Poisson maximum likelihood; [EQ-INF-COUNT-002]
    negative binomial profile likelihood and its Poisson boundary;
    [EQ-INF-COUNT-003] Garwood exact interval; [EQ-INF-CHI2-GOF-001]
    Pearson statistic on pooled cells; [EQ-INF-GOF-PB-001] parametric
    bootstrap p-value; [EQ-INF-MLE-GEN-001], [EQ-INF-MLE-GEN-002] numerical
    maximum likelihood and observed-information errors; [EQ-INF-IC-001],
    [EQ-INF-IC-003], [EQ-INF-IC-004] AIC, BIC, Akaike weights;
    [EQ-INF-GOF-NORM-001] fitted-normal Lilliefors p-value;
    [EQ-INF-CI-TMEAN-001], [EQ-INF-T-001] Student-t interval and paired t;
    [EQ-STAT-TRIM-001], [EQ-STAT-QUANT-001], [EQ-STAT-MAD-001] robust
    location and scale; [EQ-VIZ-ROOT-001], [EQ-INF-GOF-QQ-001] rootogram and
    Q-Q coordinates.

Diagnostics:
    Count fits: the variance-to-mean ratio, the negative binomial size (or
    its boundary), Pearson ``X^2`` with cells pooled to at least five
    expected units, its asymptotic and bootstrap p-values, and a hanging
    rootogram. Continuous fits: AIC, BIC, Akaike weights, Kolmogorov-Smirnov
    and Anderson-Darling p-values calibrated by refitting every bootstrap
    replicate (closed-form Lilliefors calibration for the normal family), a
    Q-Q plot and a fitted CDF. Light speed: classical against robust location
    estimates and their distance from 33.02.

Cost and error:
    One run takes a few seconds: ``COUNT_BOOTSTRAP = 999`` refits for each
    count model and ``CONTINUOUS_BOOTSTRAP = 199`` refits for each continuous
    family. A bootstrap p-value ``p`` has Monte Carlo standard error
    ``sqrt(p (1 - p) / B)``, at most 0.016 for ``B = 999`` and 0.035 for
    ``B = 199``; every number is reproducible from ``SEED``. The counts are
    exact; the measurements carry the rounding of their sources (1 ns for
    Newcomb, 0.01 million revolutions for the bearings, 0.1 h for sleep).

References:
    [REF-BORTKIEWICZ-1898], [REF-RUTHERFORD-GEIGER-1910],
    [REF-NEWCOMB-VELOCITY-OF-LIGHT], [REF-STIGLER-1977],
    [REF-LIEBLEIN-ZELEN-1956], [REF-STUDENT-1908], [REF-CUSHNY-PEEBLES-1905],
    [REF-GARWOOD-1936], [REF-LEVIN-REEDS-1977], [REF-KLEIBER-ZEILEIS-2016],
    [REF-STEPHENS-1974], [REF-BURNHAM-ANDERSON-2002], [REF-MATPLOTLIB].

"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from probstat.foundations import RandomStream
from probstat.inference import (
    GAMMA_FAMILY,
    LOGNORMAL_FAMILY,
    NORMAL_FAMILY,
    STUDENT_T_FAMILY,
    WEIBULL_FAMILY,
    BootstrapGoodnessOfFit,
    ConfidenceInterval,
    CountFit,
    CountGoodnessOfFit,
    DistributionFamily,
    FitResult,
    NormalityTests,
    TTestResult,
    akaike_weights,
    bootstrap_goodness_of_fit,
    count_goodness_of_fit,
    fit_distribution,
    fit_negative_binomial_counts,
    fit_poisson_counts,
    normality_tests,
    paired_t_test,
    poisson_mean_interval,
    student_t_mean_confidence_interval,
)
from probstat.statistics import (
    BORTKIEWICZ_HORSE_KICKS,
    LIEBLEIN_ZELEN_BALL_BEARINGS,
    NEWCOMB_PASSAGE_TIMES,
    RUTHERFORD_GEIGER_SCINTILLATIONS,
    STUDENT_SLEEP,
    DataSet,
    mean,
    median,
    median_absolute_deviation,
    standard_deviation,
    trimmed_mean,
    variance,
)

SEED = 20261111
"""Root seed of every bootstrap stream in this laboratory."""

COUNT_BOOTSTRAP = 999
"""Parametric-bootstrap replicates per count model."""

CONTINUOUS_BOOTSTRAP = 199
"""Parametric-bootstrap replicates per continuous family."""

LIGHT_SPEED_REFERENCE = 33.02
"""Newcomb deviation (ns) implied by the accepted speed of light (Stigler 1977)."""

DOC_BEGIN = "<!-- real-data-case-studies:begin -->"
DOC_END = "<!-- real-data-case-studies:end -->"


@dataclass(frozen=True)
class CountCaseStudy:
    """Poisson against negative binomial for one frequency table.

    Args:
        dataset: Source record.
        counts: One count per unit, expanded from the frequency table.
        poisson: Poisson fit.
        negative_binomial: Negative binomial fit (possibly on the boundary).
        interval: Garwood 95% interval for the mean count.
        goodness_of_fit: Pearson test of the Poisson fit with bootstrap.

    References:
        [EQ-INF-COUNT-001], [EQ-INF-COUNT-002], [EQ-INF-COUNT-003].

    """

    dataset: DataSet
    counts: tuple[int, ...]
    poisson: CountFit
    negative_binomial: CountFit
    interval: ConfidenceInterval
    goodness_of_fit: CountGoodnessOfFit

    @property
    def dispersion_ratio(self) -> float:
        """Return the divisor-``n`` variance over the mean.

        The negative binomial maximum is on the Poisson boundary exactly when
        this ratio does not exceed one.

        References:
            [EQ-STAT-001], [EQ-STAT-002], [EQ-INF-COUNT-002].

        """
        return variance(self.counts, ddof=0) / mean(self.counts)


@dataclass(frozen=True)
class ContinuousCandidate:
    """One fitted continuous family with its calibrated goodness of fit.

    Args:
        fit: Maximum-likelihood fit.
        goodness_of_fit: Parametric-bootstrap KS and AD p-values.
        akaike_weight: Akaike weight among the candidates of the case.

    References:
        [EQ-INF-MLE-GEN-001], [EQ-INF-GOF-PB-001], [EQ-INF-IC-004].

    """

    fit: FitResult
    goodness_of_fit: BootstrapGoodnessOfFit
    akaike_weight: float


@dataclass(frozen=True)
class LightSpeedCaseStudy:
    """Normal against Student-t for Newcomb's passage times.

    Args:
        candidates: Normal and Student-t candidates.
        normality: Closed-form fitted-normal EDF tests.
        mean_interval: Student-t 95% interval for the mean.
        estimates: ``(label, estimate)`` location estimates, classical first.
        scales: ``(label, estimate)`` scale estimates, classical first.

    References:
        [EQ-INF-GOF-NORM-001], [EQ-STAT-TRIM-001], [REF-STIGLER-1977].

    """

    candidates: tuple[ContinuousCandidate, ...]
    normality: NormalityTests
    mean_interval: ConfidenceInterval
    estimates: tuple[tuple[str, float], ...]
    scales: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class CaseStudies:
    """All results of one laboratory run.

    Args:
        horse_kicks: Bortkiewicz count case.
        scintillations: Rutherford-Geiger count case.
        light_speed: Newcomb case.
        bearings: Lieblein-Zelen candidates in AIC order.
        sleep: Paired t test of Student's sleep data.
        sleep_normality: Fitted-normal tests of the paired differences.
        seconds: Wall-clock time of the fits.

    References:
        [EQ-INF-IC-001], [EQ-INF-T-001].

    """

    horse_kicks: CountCaseStudy
    scintillations: CountCaseStudy
    light_speed: LightSpeedCaseStudy
    bearings: tuple[ContinuousCandidate, ...]
    sleep: TTestResult
    sleep_normality: NormalityTests
    seconds: float


def count_case_study(
    dataset: DataSet, value: str, frequency: str, stream: RandomStream
) -> CountCaseStudy:
    """Fit Poisson and negative binomial laws to one frequency table.

    Args:
        dataset: Count data set.
        value: Column with the count ``k``.
        frequency: Column with the number of units showing ``k``.
        stream: Bootstrap stream.

    Returns:
        Fits, the Garwood interval, and the bootstrap Pearson test.

    References:
        [EQ-INF-COUNT-001], [EQ-INF-COUNT-002], [EQ-INF-COUNT-003],
        [EQ-INF-CHI2-GOF-001], [EQ-INF-GOF-PB-001].

    """
    counts = dataset.expand_frequencies(value, frequency)
    poisson = fit_poisson_counts(counts)
    return CountCaseStudy(
        dataset=dataset,
        counts=counts,
        poisson=poisson,
        negative_binomial=fit_negative_binomial_counts(counts),
        interval=poisson_mean_interval(counts),
        goodness_of_fit=count_goodness_of_fit(
            counts, poisson, n_bootstrap=COUNT_BOOTSTRAP, seed=stream
        ),
    )


def continuous_candidates(
    values: tuple[float, ...],
    families: tuple[DistributionFamily, ...],
    stream: RandomStream,
) -> tuple[ContinuousCandidate, ...]:
    """Fit families by maximum likelihood and calibrate each fit.

    Args:
        values: Observations.
        families: Candidate families.
        stream: Bootstrap stream (one child per family).

    Returns:
        Candidates sorted by AIC with Akaike weights.

    References:
        [EQ-INF-MLE-GEN-001], [EQ-INF-IC-001], [EQ-INF-IC-004],
        [EQ-INF-GOF-PB-001], [REF-BURNHAM-ANDERSON-2002, Chapter 2].

    """
    fits = tuple(fit_distribution(values, family) for family in families)
    weights = akaike_weights(fit.aic for fit in fits)
    candidates = tuple(
        ContinuousCandidate(
            fit=fit,
            goodness_of_fit=bootstrap_goodness_of_fit(
                values,
                family,
                n_bootstrap=CONTINUOUS_BOOTSTRAP,
                seed=stream.child(family.name),
                fit=fit,
            ),
            akaike_weight=weight,
        )
        for family, fit, weight in zip(families, fits, weights, strict=True)
    )
    return tuple(sorted(candidates, key=lambda candidate: candidate.fit.aic))


def light_speed_case_study(stream: RandomStream) -> LightSpeedCaseStudy:
    """Fit normal and Student-t laws to Newcomb's data and compare locations.

    Args:
        stream: Bootstrap stream.

    Returns:
        Candidates, normality tests, the t interval, and location estimates.

    References:
        [EQ-INF-GOF-NORM-001], [EQ-INF-CI-TMEAN-001], [EQ-STAT-TRIM-001],
        [EQ-STAT-QUANT-001], [REF-STIGLER-1977].

    """
    values = NEWCOMB_PASSAGE_TIMES.values()
    candidates = continuous_candidates(
        values, (NORMAL_FAMILY, STUDENT_T_FAMILY), stream
    )
    student = next(c.fit for c in candidates if c.fit.family == "StudentT")
    kept = tuple(value for value in values if value >= 0.0)
    estimates = (
        ("mean", mean(values)),
        ("median", median(values)),
        ("10% trimmed mean", trimmed_mean(values, 0.1)),
        ("20% trimmed mean", trimmed_mean(values, 0.2)),
        ("Student-t location (MLE)", student.params["mu"]),
        ("mean without the two negative values", mean(kept)),
    )
    return LightSpeedCaseStudy(
        candidates=candidates,
        normality=normality_tests(values),
        mean_interval=student_t_mean_confidence_interval(values),
        estimates=estimates,
        scales=(
            ("standard deviation", standard_deviation(values, ddof=1)),
            ("normal-consistent MAD", median_absolute_deviation(values)),
            ("Student-t scale (MLE)", student.params["sigma"]),
        ),
    )


def run_case_studies(seed: int = SEED) -> CaseStudies:
    """Run every case study from one seed.

    Args:
        seed: Root seed of the bootstrap streams.

    Returns:
        All fits and diagnostics.

    References:
        [EQ-INF-GOF-PB-001], [EQ-INF-T-001].

    """
    started = time.perf_counter()
    root = RandomStream(seed)
    horse_kicks = count_case_study(
        BORTKIEWICZ_HORSE_KICKS, "deaths", "corps_years", root.child("horse-kicks")
    )
    scintillations = count_case_study(
        RUTHERFORD_GEIGER_SCINTILLATIONS,
        "particles",
        "intervals",
        root.child("scintillations"),
    )
    light_speed = light_speed_case_study(root.child("light-speed"))
    bearings = continuous_candidates(
        LIEBLEIN_ZELEN_BALL_BEARINGS.values(),
        (WEIBULL_FAMILY, LOGNORMAL_FAMILY, GAMMA_FAMILY),
        root.child("bearings"),
    )
    differences = tuple(
        b - a
        for a, b in zip(
            STUDENT_SLEEP.values("drug_1"), STUDENT_SLEEP.values("drug_2"), strict=True
        )
    )
    return CaseStudies(
        horse_kicks=horse_kicks,
        scintillations=scintillations,
        light_speed=light_speed,
        bearings=bearings,
        sleep=paired_t_test(
            STUDENT_SLEEP.values("drug_2"), STUDENT_SLEEP.values("drug_1")
        ),
        sleep_normality=normality_tests(differences),
        seconds=time.perf_counter() - started,
    )


def _p(value: float) -> str:
    """Format a p-value with three decimals, or ``< 0.001``.

    References:
        [EQ-INF-GOF-PB-001]. No equation is evaluated.

    """
    if math.isnan(value):
        return "n/a"
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def _parameters(fit: FitResult) -> str:
    """Format fitted parameters with observed-information standard errors.

    References:
        [EQ-INF-MLE-GEN-002]. No equation is evaluated.

    """
    return ", ".join(
        f"{name} = {value:.4g} ({error:.2g})"
        for name, value, error in zip(
            fit.parameter_names, fit.estimates, fit.standard_errors, strict=True
        )
    )


def _count_row(case: CountCaseStudy) -> str:
    """Return the Markdown row of one count case.

    References:
        [EQ-INF-COUNT-001], [EQ-INF-COUNT-002], [EQ-INF-COUNT-003].

    """
    nb = case.negative_binomial
    size = "inf (boundary)" if nb.at_boundary else f"{nb.estimates[0]:.3f}"
    gof = case.goodness_of_fit
    return (
        f"| {case.dataset.key} | {len(case.counts)} | {sum(case.counts)} "
        f"| {case.poisson.mean:.4f} | [{case.interval.lower:.4f}, "
        f"{case.interval.upper:.4f}] | {case.dispersion_ratio:.4f} | {size} "
        f"| {case.poisson.aic:.2f} | {nb.aic:.2f} | {gof.statistic:.3f} "
        f"({len(gof.cells)} cells, {gof.degrees_of_freedom} df) "
        f"| {_p(gof.asymptotic_p_value)} | {_p(gof.bootstrap_p_value)} |"
    )


def _expected_line(case: CountCaseStudy) -> str:
    """Return the fitted Poisson frequencies of one count case as text.

    References:
        [EQ-INF-COUNT-001].

    """
    values = case.poisson.expected_frequencies(max(case.counts))
    listed = ", ".join(f"{value:.1f}" for value in values[:-1])
    return (
        f"- `{case.dataset.key}`: n p(k) for k = 0..{max(case.counts)} is "
        f"{listed}; the tail k > {max(case.counts)} adds {values[-1]:.2f}."
    )


def results_markdown(results: CaseStudies) -> str:
    """Render the case-study tables quoted in the fitting guide.

    Args:
        results: Output of :func:`run_case_studies`.

    Returns:
        Markdown between the ``DOC_BEGIN`` and ``DOC_END`` markers.

    References:
        [EQ-INF-IC-001], [EQ-INF-IC-003], [EQ-INF-GOF-PB-001].

    """
    lines = [
        DOC_BEGIN,
        "",
        "**Counts: Poisson against negative binomial** (Garwood 95% interval "
        "for the mean; Pearson X^2 on cells pooled to at least five expected "
        f"units; bootstrap with {COUNT_BOOTSTRAP} refits, seed {SEED})",
        "",
        "| Data set | Units | Total | Poisson mean | 95% interval "
        "| Variance (divisor n) / mean | NB size | Poisson AIC | NB AIC | X^2 "
        "| Asymptotic p | Bootstrap p |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        _count_row(results.horse_kicks),
        _count_row(results.scintillations),
        "",
        "Fitted Poisson frequencies:",
        "",
        _expected_line(results.horse_kicks),
        _expected_line(results.scintillations),
        "",
        "**Newcomb's passage times: normal against Student-t** (KS and AD "
        f"p-values from {CONTINUOUS_BOOTSTRAP} bootstrap refits)",
        "",
        "| Family | Parameters (SE) | log L | AIC | BIC | Akaike weight "
        "| KS p | AD p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for candidate in results.light_speed.candidates:
        fit, gof = candidate.fit, candidate.goodness_of_fit
        lines.append(
            f"| {fit.family} | {_parameters(fit)} | {fit.log_likelihood:.2f} "
            f"| {fit.aic:.2f} | {fit.bic:.2f} | {candidate.akaike_weight:.4f} "
            f"| {_p(gof.ks_p_value)} | {_p(gof.ad_p_value)} |"
        )
    normality = results.light_speed.normality
    interval = results.light_speed.mean_interval
    lines += [
        "",
        "Closed-form fitted-normal calibration: Lilliefors p "
        f"{_p(normality.lilliefors.p_value)}, Cramer-von Mises p "
        f"{_p(normality.cramer_von_mises.p_value)}, Anderson-Darling p "
        f"{_p(normality.anderson_darling.p_value)}. Student-t 95% interval "
        f"for the mean: [{interval.lower:.3f}, {interval.upper:.3f}].",
        "",
        f"| Location estimate | Value (ns) | Below {LIGHT_SPEED_REFERENCE} by |",
        "|---|---|---|",
    ]
    for label, value in results.light_speed.estimates:
        lines.append(
            f"| {label} | {value:.3f} | {LIGHT_SPEED_REFERENCE - value:.3f} |"
        )
    scales = "; ".join(
        f"{label} {value:.3f}" for label, value in results.light_speed.scales
    )
    lines += ["", f"Scale estimates (ns): {scales}."]
    lines += [
        "",
        "**Ball-bearing endurance: Weibull, lognormal, gamma** (KS and AD "
        f"p-values from {CONTINUOUS_BOOTSTRAP} bootstrap refits)",
        "",
        "| Family | Parameters (SE) | log L | AIC | BIC | Akaike weight "
        "| KS p | AD p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for candidate in results.bearings:
        fit, gof = candidate.fit, candidate.goodness_of_fit
        lines.append(
            f"| {fit.family} | {_parameters(fit)} | {fit.log_likelihood:.2f} "
            f"| {fit.aic:.2f} | {fit.bic:.2f} | {candidate.akaike_weight:.4f} "
            f"| {_p(gof.ks_p_value)} | {_p(gof.ad_p_value)} |"
        )
    sleep = results.sleep
    interval = sleep.confidence_interval
    assert interval is not None
    lines += [
        "",
        "**Student's sleep data: paired t test** (drug_2 - drug_1)",
        "",
        "| Mean difference (h) | SE | t | df | Two-sided p | 95% interval "
        "| Lilliefors p of the differences |",
        "|---|---|---|---|---|---|---|",
        f"| {sleep.estimate:.3f} | {sleep.standard_error:.4f} "
        f"| {sleep.statistic:.4f} | {sleep.df:.0f} | {sleep.p_value:.5f} "
        f"| [{interval.lower:.3f}, {interval.upper:.3f}] "
        f"| {_p(results.sleep_normality.lilliefors.p_value)} |",
        "",
        DOC_END,
    ]
    return "\n".join(lines)


def _frame(axes: Any) -> None:
    """Reserve figure margins for the title block and the direct labels.

    Args:
        axes: Matplotlib axes of a single-panel figure.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    figure = axes.figure
    figure.set_size_inches(7.6, 5.0)
    figure.subplots_adjust(left=0.1, right=0.8, bottom=0.13, top=0.8)


def render_figures(results: CaseStudies, output: Path) -> tuple[Path, ...]:
    """Save a rootogram per count case, a Newcomb Q-Q plot, and a bearing CDF.

    Args:
        results: Output of :func:`run_case_studies`.
        output: Directory for the PNG files.

    Returns:
        Written paths.

    References:
        [EQ-VIZ-ROOT-001], [EQ-INF-GOF-QQ-001], [EQ-STAT-005],
        [REF-KLEIBER-ZEILEIS-2016], [REF-MATPLOTLIB].

    """
    import matplotlib.pyplot as plt

    from probstat.visualization import plot_fitted_cdf, save_figure
    from probstat.visualization.fit_plots import plot_qq, plot_rootogram

    written: list[Path] = []
    for case, stem, label in (
        (
            results.horse_kicks,
            "real-data-horse-kick-rootogram",
            "Deaths per corps-year",
        ),
        (
            results.scintillations,
            "real-data-alpha-particle-rootogram",
            "Alpha particles per 7.5 s interval",
        ),
    ):
        top = max(case.counts)
        observed = [case.counts.count(k) for k in range(top + 1)]
        expected = case.poisson.expected_frequencies(top)[:-1]
        axes = plot_rootogram(
            observed,
            expected,
            count_label=label,
            model_label=f"Poisson fit (mean {case.poisson.mean:.3f})",
        )
        _frame(axes)
        written.append(save_figure(axes.figure, output / f"{stem}.png"))
        plt.close(axes.figure)
    normal = next(
        c.fit for c in results.light_speed.candidates if c.fit.family == "Normal"
    )
    axes = plot_qq(
        NEWCOMB_PASSAGE_TIMES.values(),
        normal.distribution,
        model_label="normal",
        value_label="Passage time - 24800 ns",
        quantile_label="Fitted normal quantile - 24800 ns",
        unit="ns",
    )
    _frame(axes)
    written.append(save_figure(axes.figure, output / "real-data-newcomb-qq.png"))
    plt.close(axes.figure)
    best = results.bearings[0].fit
    axes = plot_fitted_cdf(LIEBLEIN_ZELEN_BALL_BEARINGS.values(), best.distribution)
    axes.set_xlabel("Endurance (millions of revolutions)")
    _frame(axes)
    written.append(save_figure(axes.figure, output / "real-data-bearing-cdf.png"))
    plt.close(axes.figure)
    return tuple(written)


def main() -> None:
    """Run the case studies, print the guide tables, and save four figures.

    References:
        [EQ-INF-GOF-PB-001], [REF-MATPLOTLIB].

    """
    print(
        f"Fitting five historical data sets (seed {SEED}; "
        f"{COUNT_BOOTSTRAP} and {CONTINUOUS_BOOTSTRAP} bootstrap refits)..."
    )
    results = run_case_studies()
    print(results_markdown(results))
    print(f"(fits and bootstrap calibration: {results.seconds:.1f} s)")
    output = Path(__file__).parent / "generated"
    for path in render_figures(results, output):
        print(f"saved {path.name}")


if __name__ == "__main__":
    main()
