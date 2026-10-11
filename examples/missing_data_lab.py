"""Missing-data laboratory: complete cases, mean imputation, EM, and MI.

Four seeded experiments, printed as numbers (no figures):

1. Missingness pattern and Little's MCAR test on one MAR data set.
2. Mean, standard error, variance, and correlation of the incomplete
   variable under five analyses: the full data before deletion (oracle),
   complete cases, mean imputation, EM, and multiple imputation.
3. A regression coefficient pooled across imputations by Rubin's rules.
4. Interval coverage over repeated MAR samples.

Data: ``(x0, x1, x2) ~ N(mu, Sigma)`` with ``mu = (1, 2, -1)``; ``x0`` is
always observed, ``x1`` is missing with probability
``logistic(-0.5 + 2 (x0 - 1))`` (missing at random given ``x0``), and ``x2``
is missing completely at random with probability 0.2.
"""

from __future__ import annotations

import math
import random
import time

from probstat.foundations.random_streams import RandomStream
from probstat.inference import (
    complete_cases,
    fit_incomplete_multivariate_normal,
    fit_linear_model,
    impute_multivariate_normal,
    little_mcar_test,
    mean_impute,
    missing_pattern,
    pool_estimates,
    student_t_quantile,
)
from probstat.multivariate import MultivariateNormal

TRUE_MEAN = (1.0, 2.0, -1.0)
TRUE_COVARIANCE = ((1.0, 0.6, 0.3), (0.6, 1.5, 0.4), (0.3, 0.4, 0.8))
TRUE_SLOPE = TRUE_COVARIANCE[0][1] / TRUE_COVARIANCE[1][1]  # x0 on x1: 0.4


def simulate(
    n: int, seed: int
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float | None, ...], ...]]:
    """Return the full rows and the same rows with MAR/MCAR deletions.

    References:
        [REF-LITTLE-RUBIN-2019]. No equation is evaluated.

    """
    full = MultivariateNormal(TRUE_MEAN, TRUE_COVARIANCE).sample(n, seed=seed)
    rng = random.Random(seed + 10_000)
    rows: list[tuple[float | None, ...]] = []
    for x in full:
        drop_1 = rng.random() < 1.0 / (1.0 + math.exp(0.5 - 2.0 * (x[0] - 1.0)))
        drop_2 = rng.random() < 0.2
        rows.append((x[0], None if drop_1 else x[1], None if drop_2 else x[2]))
    return full, tuple(rows)


def moments(rows: tuple[tuple[float, ...], ...]) -> tuple[float, float, float, float]:
    """Return mean(x1), naive SE of that mean, var(x1), and corr(x0, x1).

    References:
        [REF-CASELLA-BERGER-2002, Chapter 5]. No registered equation.

    """
    n = len(rows)
    m0 = math.fsum(r[0] for r in rows) / n
    m1 = math.fsum(r[1] for r in rows) / n
    v0 = math.fsum((r[0] - m0) ** 2 for r in rows) / (n - 1)
    v1 = math.fsum((r[1] - m1) ** 2 for r in rows) / (n - 1)
    c01 = math.fsum((r[0] - m0) * (r[1] - m1) for r in rows) / (n - 1)
    return m1, math.sqrt(v1 / n), v1, c01 / math.sqrt(v0 * v1)


def row(label: str, used: str, values: tuple[float, float, float, float]) -> None:
    """Print one line of the comparison table.

    References:
        Project laboratory policy. No equation is evaluated.

    """
    mean, se, variance, corr = values
    print(f"   {label:<22}{used:>6}{mean:>9.3f}{se:>8.3f}{variance:>9.3f}{corr:>9.3f}")


def patterns_and_little(rows: tuple[tuple[float | None, ...], ...]) -> None:
    """Experiment 1: missingness summary and Little's test.

    References:
        [EQ-MISS-004], [REF-LITTLE-1988].

    """
    print("1. Missingness pattern and Little's MCAR test (n = 400, seed 31)")
    summary = missing_pattern(rows)
    fractions = ", ".join(f"{f:.3f}" for f in summary.missing_fraction)
    print(
        f"   missing fraction per variable: ({fractions}); "
        f"complete cases {summary.complete_cases}/{summary.n_observations}"
    )
    for flags, count in zip(summary.patterns, summary.pattern_counts, strict=True):
        mask = "".join("o" if flag else "." for flag in flags)
        print(f"   pattern {mask}  rows {count}")
    test = little_mcar_test(rows)
    print(
        f"   Little d^2 = {test.statistic:.2f} on {test.degrees_of_freedom} df, "
        f"p = {test.p_value:.2e} -> MCAR rejected (MAR on x0 by construction)"
    )


def compare_methods(
    full: tuple[tuple[float, ...], ...],
    rows: tuple[tuple[float | None, ...], ...],
) -> tuple[float, float]:
    """Experiment 2: five analyses of the incomplete variable x1.

    References:
        [EQ-MISS-002], [EQ-MISS-003], [EQ-MISS-007], [EQ-MISS-009],
        [REF-LITTLE-RUBIN-2019].

    """
    print("\n2. Variable x1 (true mean 2.000, variance 1.500, corr(x0, x1) 0.490)")
    print(f"   {'analysis':<22}{'rows':>6}{'mean':>9}{'SE':>8}{'var':>9}{'corr':>9}")
    n = len(rows)
    row("full data (oracle)", str(n), moments(full))
    cases = tuple((r[0], r[1]) for r in complete_cases(rows))
    row("complete cases*", str(len(cases)), moments(cases))
    row("mean imputation*", str(n), moments(mean_impute(rows)))
    fit = fit_incomplete_multivariate_normal(rows)
    assert fit.mean_standard_errors is not None
    em_corr = fit.covariance[0][1] / math.sqrt(
        fit.covariance[0][0] * fit.covariance[1][1]
    )
    row(
        "EM (ML, divisor n)",
        str(n),
        (fit.mean[1], fit.mean_standard_errors[1], fit.covariance[1][1], em_corr),
    )
    imputed = impute_multivariate_normal(rows, 20, burn_in=50, start=fit, seed=31)
    per_set = [moments(dataset) for dataset in imputed.datasets]
    pooled = pool_estimates(
        [m[0] for m in per_set], [m[1] ** 2 for m in per_set], complete_data_df=n - 1
    )
    row(
        "MI (m = 20, Rubin)",
        str(n),
        (
            pooled.estimate,
            pooled.standard_error,
            math.fsum(m[2] for m in per_set) / 20,
            math.fsum(m[3] for m in per_set) / 20,
        ),
    )
    print("   * biased under MAR: complete cases shift the mean; mean imputation")
    print("     shrinks the variance and correlation and understates the SE.")
    print("   One sample: EM and MI sit 1.2 SE above the truth here; experiment 4")
    print("   shows the behaviour over repeated samples.")
    print(
        f"   EM: {fit.em.iterations} iterations, monotone = {fit.em.monotone}, "
        f"log-likelihood {fit.log_likelihood:.2f}"
    )
    print(
        f"   MI pooling: r = {pooled.relative_increase:.3f}, "
        f"lambda = {pooled.missing_variance_fraction:.3f}, "
        f"FMI = {pooled.fraction_missing_information:.3f}, "
        f"df = {pooled.degrees_of_freedom:.1f}"
    )
    return fit.mean[1], pooled.estimate


def pooled_regression(rows: tuple[tuple[float | None, ...], ...]) -> float:
    """Experiment 3: slope of x0 on x1 pooled across imputations.

    References:
        [EQ-MISS-007], [EQ-MISS-008], [REF-RUBIN-1987], [REF-BARNARD-RUBIN-1999].

    """
    print(f"\n3. Regression slope of x0 on x1 (true {TRUE_SLOPE:.3f})")
    cases = complete_cases(rows)
    cc = fit_linear_model([(1.0, r[1]) for r in cases], [r[0] for r in cases])
    print(
        f"   complete cases*: {cc.coefficients[1]:.3f} "
        f"(SE {cc.standard_errors[1]:.3f}, n = {len(cases)})"
    )
    imputed = impute_multivariate_normal(rows, 20, burn_in=50, seed=32)
    estimates: list[float] = []
    variances: list[float] = []
    for dataset in imputed.datasets:
        model = fit_linear_model(
            [(1.0, r[1]) for r in dataset], [r[0] for r in dataset]
        )
        estimates.append(model.coefficients[1])
        variances.append(model.standard_errors[1] ** 2)
    df_com = len(rows) - 2
    pooled = pool_estimates(estimates, variances, complete_data_df=df_com)
    print(
        f"   MI (m = 20):     {pooled.estimate:.3f} (SE {pooled.standard_error:.3f}, "
        f"95% CI {pooled.lower:.3f} to {pooled.upper:.3f})"
    )
    print(
        f"   Rubin: Ubar = {pooled.within_variance:.5f}, "
        f"B = {pooled.between_variance:.5f}, T = {pooled.total_variance:.5f}, "
        f"Barnard-Rubin df = {pooled.degrees_of_freedom:.1f} "
        f"(complete-data df {df_com})"
    )
    return pooled.estimate


def coverage(reps: int, n: int, m: int) -> tuple[float, float, float]:
    """Experiment 4: 95% interval coverage of E[x1] over repeated samples.

    References:
        [EQ-MISS-003], [EQ-MISS-007], [REF-RUBIN-1987].

    """
    print(
        f"\n4. Coverage of nominal 95% intervals for E[x1] "
        f"({reps} samples, n = {n}, m = {m}, burn-in 20)"
    )
    hits = {"complete cases*": 0, "mean imputation*": 0, "EM Wald": 0, "MI Rubin": 0}
    errors: dict[str, list[float]] = {label: [] for label in hits}
    for rep in range(reps):
        if rep and rep % 40 == 0:
            print(f"   ... {rep}/{reps} samples", flush=True)
        _, rows = simulate(n, 5000 + rep)
        cases = tuple((r[0], r[1]) for r in complete_cases(rows))
        for label, data in (
            ("complete cases*", cases),
            ("mean imputation*", mean_impute(rows)),
        ):
            mean, se, _, _ = moments(data)
            half = student_t_quantile(0.975, len(data) - 1) * se
            hits[label] += abs(mean - TRUE_MEAN[1]) <= half
            errors[label].append(mean - TRUE_MEAN[1])
        fit = fit_incomplete_multivariate_normal(rows)
        assert fit.mean_standard_errors is not None
        hits["EM Wald"] += abs(fit.mean[1] - TRUE_MEAN[1]) <= (
            1.959964 * fit.mean_standard_errors[1]
        )
        errors["EM Wald"].append(fit.mean[1] - TRUE_MEAN[1])
        imputed = impute_multivariate_normal(
            rows, m, burn_in=20, start=fit, seed=RandomStream(rep)
        )
        per_set = [moments(dataset) for dataset in imputed.datasets]
        pooled = pool_estimates(
            [v[0] for v in per_set],
            [v[1] ** 2 for v in per_set],
            complete_data_df=n - 1,
        )
        hits["MI Rubin"] += pooled.lower <= TRUE_MEAN[1] <= pooled.upper
        errors["MI Rubin"].append(pooled.estimate - TRUE_MEAN[1])
    print(f"   {'interval':<18}{'coverage':>9}{'mean error':>12}")
    for label, count in hits.items():
        bias = math.fsum(errors[label]) / reps
        print(f"   {label:<18}{count / reps:>9.3f}{bias:>12.3f}")
    return (
        hits["complete cases*"] / reps,
        hits["EM Wald"] / reps,
        hits["MI Rubin"] / reps,
    )


def main() -> None:
    """Run the four experiments and print a one-line conclusion.

    References:
        Project laboratory policy. No equation is evaluated.

    """
    started = time.perf_counter()
    full, rows = simulate(400, 31)
    patterns_and_little(rows)
    compare_methods(full, rows)
    pooled_regression(rows)
    cc_cov, em_cov, mi_cov = coverage(120, 80, 10)
    print(
        f"\nDone in {time.perf_counter() - started:.1f} s: coverage complete cases "
        f"{cc_cov:.3f}, EM {em_cov:.3f}, MI {mi_cov:.3f} (nominal 0.95). "
        "Next: docs/foundations/missing-data.md"
    )


if __name__ == "__main__":
    main()
