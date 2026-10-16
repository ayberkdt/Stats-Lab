# R Bridge

This guide maps common R calls to their `probstat` equivalents and spells out
where the two disagree on parameterization, defaults, or reported quantities.
It is written for readers who learned statistics in R [REF-R-CORE-TEAM].

## Provenance of the numbers

- R was **not** run while writing this guide. R defaults quoted below were
  read from the `stats` package R sources (development trunk, files such as
  `t.test.R`, `wilcox.test.R`, `fisher.test.R`, `chisq.test.R`,
  `binom.test.R`, `p.adjust.R`, `quantile.R`, `optim.R`, `nlm.R`, `mad.R`) on
  2026-10-11.
- Reference values in the "R value" columns were computed independently of
  `probstat` with mpmath 1.3.0 (30 significant digits, closed forms where they
  exist) and SciPy 1.14.1, then rounded to 7 significant digits, which is R's
  default print precision (`getOption("digits")`). They are what the R
  formulas give; they are not transcripts of an R session.
- Every Python snippet was executed; the block after it is its exact output.
  `tests/test_prerequisite_guides.py` re-runs the snippets and
  `tests/test_r_bridge_examples.py` re-checks the quoted values, several of
  them against a second, standard-library-only computation.

## 1. Distribution functions: `d`, `p`, `q`, `r`

| R | `probstat` | Notes |
|---|---|---|
| `dfoo(x, ...)` | `Foo(...).pdf(x)` / `.pmf(k)` | densities for continuous, masses for discrete families |
| `dfoo(x, ..., log = TRUE)` | `.log_pdf(x)` / `.log_pmf(k)` | |
| `pfoo(q, ...)` | `.cdf(q)` | |
| `pfoo(q, ..., lower.tail = FALSE)` | `.sf(q)` | computed directly, accurate in the upper tail |
| `qfoo(p, ...)` | `.quantile(p)` | discrete: smallest support point with `F(x) >= p`, as in R |
| `qfoo(p, ..., lower.tail = FALSE)` | `.isf(p)` | continuous families |
| `pfoo(..., log.p = TRUE)` | none | take `math.log` of `cdf`/`sf` |
| `rfoo(n, ...)` | `.sample(n, seed=...)` | streams differ from R even for equal seeds (section 5) |

| R family | `probstat` class | Parameterization notes |
|---|---|---|
| `norm(mean, sd)` | `Normal(mu, sigma)` | standard deviation, not variance |
| `lnorm(meanlog, sdlog)` | `LogNormal(mu, sigma)` | parameters of `log X` |
| `unif(min, max)` | `Uniform(lower, upper)` | |
| `exp(rate)` | `Exponential(rate)` | rate in both; mean is `1/rate` |
| `gamma(shape, rate = 1, scale = 1/rate)` | `Gamma(shape, rate)`, `Gamma.from_scale(shape, scale)` | R's third positional argument is `rate`; write `scale =` explicitly when porting |
| `weibull(shape, scale = 1)` | `Weibull(shape, scale)` | same order and meaning |
| `beta(shape1, shape2)` | `Beta(alpha, beta)` | no non-centrality |
| `t(df)` | `StudentT(nu, mu=0, sigma=1)` | adds location and scale; R's `ncp` CDF is `inference.noncentral_t_cdf` |
| `chisq(df)` | `ChiSquare(nu)` | no non-centrality |
| `f(df1, df2)` | `FisherF(nu1, nu2)` | no non-centrality |
| `logis(location, scale)` | `Logistic(mu, s)` | |
| `binom(size, prob)` | `Binomial(n, p)` | |
| `pois(lambda)` | `Poisson(lam)` | |
| `geom(prob)` | `Geometric(p)` | both count failures before the first success (support `0, 1, ...`) |
| `nbinom(size, prob)` | `NegativeBinomial(r, p)` | failures before the `r`-th success; `r` must be an integer here, R also accepts real `size` and a `mu` form |

```python
import math
from probstat.distributions import (
    Normal, StudentT, ChiSquare, FisherF, Gamma, Exponential, Weibull,
    LogNormal, Beta, Binomial, Poisson, Geometric, NegativeBinomial, Logistic,
)

z = Normal(0.0, 1.0)
print(f"pnorm(1.96)            {z.cdf(1.96):.7f}")
print(f"qnorm(0.975)           {z.quantile(0.975):.6f}")
print(f"pnorm(-3)              {z.cdf(-3.0):.6e}")
print(f"qt(0.975, 10)          {StudentT(10).quantile(0.975):.6f}")
print(f"pt(2, 5)               {StudentT(5).cdf(2.0):.7f}")
print(f"qchisq(0.95, 1)        {ChiSquare(1).quantile(0.95):.6f}")
print(f"pchisq(3, 4)           {ChiSquare(4).cdf(3.0):.7f}")
print(f"pf(3, 2, 10)           {FisherF(2, 10).cdf(3.0):.7f}")
print(f"qf(0.95, 2, 10)        {FisherF(2, 10).quantile(0.95):.6f}")
print(f"qgamma(0.5, 2, 0.5)    {Gamma(2.0, rate=0.5).quantile(0.5):.6f}")
print(f"pexp(1, rate = 2)      {Exponential(rate=2.0).cdf(1.0):.7f}")
print(f"qweibull(0.5, 1.5, 2)  {Weibull(1.5, 2.0).quantile(0.5):.6f}")
print(f"plnorm(2)              {LogNormal(0.0, 1.0).cdf(2.0):.7f}")
print(f"pbeta(0.3, 2, 5)       {Beta(2.0, 5.0).cdf(0.3):.6f}")
print(f"plogis(1)              {Logistic(0.0, 1.0).cdf(1.0):.7f}")
print(f"dbinom(3, 10, 0.5)     {Binomial(10, 0.5).pmf(3):.7f}")
print(f"pbinom(3, 10, 0.5)     {Binomial(10, 0.5).cdf(3):.6f}")
print(f"ppois(2, 3)            {Poisson(3.0).cdf(2):.7f}")
print(f"dgeom(2, 0.3)          {Geometric(0.3).pmf(2):.3f}")
print(f"dnbinom(3, 2, 0.4)     {NegativeBinomial(2, 0.4).pmf(3):.5f}")
```

```text
pnorm(1.96)            0.9750021
qnorm(0.975)           1.959964
pnorm(-3)              1.349898e-03
qt(0.975, 10)          2.228139
pt(2, 5)               0.9490303
qchisq(0.95, 1)        3.841459
pchisq(3, 4)           0.4421746
pf(3, 2, 10)           0.9046326
qf(0.95, 2, 10)        4.102821
qgamma(0.5, 2, 0.5)    3.356694
pexp(1, rate = 2)      0.8646647
qweibull(0.5, 1.5, 2)  1.566440
plnorm(2)              0.7558914
pbeta(0.3, 2, 5)       0.579825
plogis(1)              0.7310586
dbinom(3, 10, 0.5)     0.1171875
pbinom(3, 10, 0.5)     0.171875
ppois(2, 3)            0.4231901
dgeom(2, 0.3)          0.147
dnbinom(3, 2, 0.4)     0.13824
```

| R call | R value | Independent check |
|---|---|---|
| `pnorm(1.96)` | 0.9750021 | mpmath `ncdf`; `statistics.NormalDist` |
| `qnorm(0.975)` | 1.959964 | mpmath `erfinv`; `statistics.NormalDist` |
| `pnorm(-3)` | 0.001349898 | mpmath `ncdf` |
| `qt(0.975, 10)` | 2.228139 | mpmath root of the incomplete-beta CDF; Simpson integral of the density |
| `pt(2, 5)` | 0.9490303 | mpmath incomplete beta; Simpson integral |
| `qchisq(0.95, 1)` | 3.841459 | `qnorm(0.975)^2` |
| `pchisq(3, 4)` | 0.4421746 | closed form `1 - e^(-1.5) (1 + 1.5)` |
| `pf(3, 2, 10)` | 0.9046326 | closed form `1 - 1.6^(-5)` |
| `qf(0.95, 2, 10)` | 4.102821 | closed form `5 (0.05^(-1/5) - 1)` |
| `qgamma(0.5, 2, 0.5)` | 3.356694 | mpmath root of `1 - e^(-x/2)(1 + x/2) = 1/2` |
| `pexp(1, rate = 2)` | 0.8646647 | `1 - e^(-2)` |
| `qweibull(0.5, 1.5, 2)` | 1.566440 | `2 (log 2)^(2/3)`; R prints `1.56644` |
| `plnorm(2)` | 0.7558914 | `pnorm(log 2)` |
| `pbeta(0.3, 2, 5)` | 0.579825 | binomial-sum identity for integer shapes |
| `plogis(1)` | 0.7310586 | `1 / (1 + e^(-1))` |
| `dbinom(3, 10, 0.5)` | 0.1171875 | exact `120/1024` |
| `pbinom(3, 10, 0.5)` | 0.171875 | exact `176/1024` |
| `ppois(2, 3)` | 0.4231901 | `e^(-3) (1 + 3 + 4.5)` |
| `dgeom(2, 0.3)` | 0.147 | exact `0.7^2 0.3` |
| `dnbinom(3, 2, 0.4)` | 0.13824 | exact `4 (0.4^2)(0.6^3)` |

## 2. Descriptive statistics

| R | `probstat.statistics` | Notes |
|---|---|---|
| `mean(x)` | `mean(x)` | |
| `var(x)`, `sd(x)` | `variance(x, ddof=1)`, `standard_deviation(x, ddof=1)` | **default `ddof=0` differs from R's `n - 1`** |
| `cov(x, y)` | `covariance(x, y, ddof=1)` | same caveat |
| `cor(x, y)` | `pearson_correlation(x, y)` | |
| `cor(x, y, method = "spearman")` | `spearman_correlation(x, y)` | |
| `median(x)` | `median(x)` | |
| `quantile(x, p)` (type 7) | `quantile(x, p)`, `quantiles(x, ps)` | only Hyndman-Fan types 7 (default) and 1 (`method=1`) [REF-HYNDMAN-FAN-1996] |
| `IQR(x)` | `interquartile_range(x)` | type 7 in both |
| `mad(x)` | `median_absolute_deviation(x)` | R multiplies by `constant = 1.4826`; `probstat` uses `1/qnorm(0.75) = 1.482602...`, a relative difference of about `1.5e-6` |
| `scale(x)` | `z_scores(x, ddof=1)` | R centers and divides by `sd`, i.e. `n - 1` |
| `ecdf(x)(t)` | `empirical_cdf(x, t)` | right-continuous step function |

```python
from probstat.statistics import (
    variance, standard_deviation, quantiles, interquartile_range,
    median_absolute_deviation, empirical_cdf,
)

x = (2.1, 3.4, 1.9, 5.6, 4.4)
print(f"var(x) = {variance(x, ddof=1):.6f}   (ddof=0 gives {variance(x):.6f})")
print(f"quantile(x, c(.1, .25, .5, .9)) = {tuple(round(q, 6) for q in quantiles(x, (0.1, 0.25, 0.5, 0.9)))}")
print(f"IQR(x) = {interquartile_range(x):.6f}")
print(f"mad(x) = {median_absolute_deviation(x):.6f}")
print(f"ecdf(x)(3.4) = {empirical_cdf(x, 3.4):.1f}")
```

```text
var(x) = 2.437000   (ddof=0 gives 1.949600)
quantile(x, c(.1, .25, .5, .9)) = (1.98, 2.1, 3.4, 5.12)
IQR(x) = 2.300000
mad(x) = 1.927383
ecdf(x)(3.4) = 0.6
```

The R formulas give `var(x) = 2.437`, `quantile(x, c(.1, .25, .5, .9)) =
1.98, 2.10, 3.40, 5.12`, and `IQR(x) = 2.3` (type-7 interpolation
`h = (n - 1) p + 1`, checked by hand and with NumPy's default `linear`
method), and `ecdf(x)(3.4) = 3/5`. R's `mad(x) = 1.4826 * 1.3 = 1.92738`
differs from the value above by about `3e-6` because of the rounded constant.

## 3. Classical tests

| R | `probstat.inference` | Notes |
|---|---|---|
| `t.test(x, mu = m)` | `one_sample_t_test(x, null_mean=m)` | |
| `t.test(x, y)` | `two_sample_t_test(x, y)` | Welch is the default in both (`var.equal = FALSE`, `equal_variances=False`) |
| `t.test(x, y, var.equal = TRUE)` | `two_sample_t_test(x, y, equal_variances=True)` | |
| `t.test(x, y, paired = TRUE)` | `paired_t_test(x, y)` | |
| `alternative = "two.sided"/"greater"/"less"` | `alternative="two-sided"/"greater"/"less"` | hyphen instead of dot |
| `chisq.test(tab)` | `chi_square_independence_test(tab, yates_correction=...)` | **R applies Yates' correction to 2 x 2 tables by default (`correct = TRUE`); `probstat` defaults to `False`.** Pass `yates_correction=True` to match R on 2 x 2 tables |
| `chisq.test(x, p = p)` | `chi_square_goodness_of_fit(x, p)` | `estimated_parameters=` reduces the degrees of freedom; R has no such argument |
| `fisher.test(tab)` | `fisher_exact_test(tab)` | same two-sided rule (sum of tables no more probable than observed, relative tolerance `1e-7`); **R reports the conditional MLE of the odds ratio, `probstat` the sample odds ratio `ad/(bc)`**; `probstat` handles 2 x 2 tables only |
| `binom.test(x, n, p)` | `binomial_test(x, n, null_probability=p)` | same two-sided rule (outcomes with mass at most the observed mass, relative tolerance `1e-7`) |
| `wilcox.test(x, y, paired = TRUE)` / `wilcox.test(x, mu = m)` | `wilcoxon_signed_rank_test(x, y)` / `(x, null_value=m)` | statistic `V` = sum of positive ranks in both; zero differences are dropped |
| `wilcox.test(x, y)` | `mann_whitney_u_test(x, y)` | R's `W` equals `U` of the first sample (rank sum minus `n_x (n_x + 1)/2`) |
| `p.adjust(p, "bonferroni"/"holm"/"BH"/"BY")` | `adjust_p_values(p, method="bonferroni"/"holm"/"bh"/"by")` | R's `"fdr"` is `"bh"`; `"hochberg"` and `"hommel"` are not available |

Rank tests choose between exact and normal-approximation p-values
differently. R's default (`exact = NULL`) uses the exact route when
`n < 50` (signed rank) or `n_x < 50 && n_y < 50` (rank sum). `probstat`'s
`method="auto"` uses it when the sizes are at most 50 and there are no
ties. How R handles ties and zeros on the exact route has changed between R
versions (the current development source computes exact p-values with ties),
so pass `method="normal"` or `method="exact"` explicitly, and compare against
R's `exact =` argument, when ties are present. Both continuity corrections
default to on. Two-sided exact p-values are twice the smaller tail, capped
at one, in both (for the signed-rank statistic R picks the tail by comparing
`V` with the centre of its null distribution, which is the same thing when
there are no zeros or ties).

```python
from probstat.inference import (
    one_sample_t_test, two_sample_t_test, chi_square_independence_test,
    fisher_exact_test, binomial_test, wilcoxon_signed_rank_test,
    mann_whitney_u_test, adjust_p_values,
)

x = (5.1, 4.8, 6.0, 5.7, 5.3, 6.2, 5.9, 5.5)
y = (4.6, 5.0, 4.4, 5.2, 4.9, 4.3, 4.7)
welch = two_sample_t_test(x, y)
print(f"Welch: t = {welch.statistic:.4f}, df = {welch.df:.3f}, p = {welch.p_value:.6f}")
pooled = two_sample_t_test(x, y, equal_variances=True)
print(f"pooled: t = {pooled.statistic:.4f}, df = {pooled.df:.0f}, p = {pooled.p_value:.6f}")
print(f"one-sample (mu = 5): p = {one_sample_t_test(x, null_mean=5.0).p_value:.5f}")

table = ((12, 5), (7, 15))
yates = chi_square_independence_test(table, yates_correction=True)
plain = chi_square_independence_test(table)
print(f"chisq.test(tab): X2 = {yates.statistic:.4f}, p = {yates.p_value:.5f}")
print(f"chisq.test(tab, correct = FALSE): X2 = {plain.statistic:.4f}, p = {plain.p_value:.5f}")

tea = ((3, 1), (1, 3))
fisher = fisher_exact_test(tea)
print(f"fisher: p = {fisher.p_value:.7f}, sample odds ratio = {fisher.odds_ratio:.1f}")
print(f"fisher greater: p = {fisher_exact_test(tea, alternative='greater').p_value:.7f}")

print(f"binom.test(10, 20, 0.3): p = {binomial_test(10, 20, null_probability=0.3).p_value:.7f}")
print(f"binom.test(7, 20): p = {binomial_test(7, 20).p_value:.7f}")
```

```text
Welch: t = 3.9915, df = 12.337, p = 0.001697
pooled: t = 3.8878, df = 13, p = 0.001869
one-sample (mu = 5): p = 0.01261
chisq.test(tab): X2 = 4.3221, p = 0.03762
chisq.test(tab, correct = FALSE): X2 = 5.7696, p = 0.01631
fisher: p = 0.4857143, sample odds ratio = 9.0
fisher greater: p = 0.2428571
binom.test(10, 20, 0.3): p = 0.0834450
binom.test(7, 20): p = 0.2631760
```

Rank tests and multiple-testing adjustment:

```python
before = (1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30)
after = (0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29)
signed = wilcoxon_signed_rank_test(before, after)
print(f"signed rank: V = {signed.statistic:.0f}, p = {signed.p_value:.7f}, exact = {signed.exact}")
a = (0.80, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46)
b = (1.15, 0.88, 0.90, 0.74, 1.21)
rank_sum = mann_whitney_u_test(a, b)
print(f"rank sum: W = {rank_sum.statistic:.0f}, p = {rank_sum.p_value:.6f}")

p = (0.01, 0.04, 0.03, 0.005, 0.2)
for method in ("bonferroni", "holm", "bh", "by"):
    adjusted = adjust_p_values(p, method=method).adjusted_p_values
    print(method, tuple(round(v, 6) for v in adjusted))
```

```text
signed rank: V = 40, p = 0.0390625, exact = True
rank sum: W = 35, p = 0.254412
bonferroni (0.05, 0.2, 0.15, 0.025, 1.0)
holm (0.04, 0.09, 0.09, 0.025, 0.2)
bh (0.025, 0.05, 0.05, 0.025, 0.2)
by (0.057083, 0.114167, 0.114167, 0.057083, 0.456667)
```

| R call | R value | Independent check |
|---|---|---|
| `t.test(x, y)` | `t = 3.9915, df = 12.337, p-value = 0.001697` | SciPy `ttest_ind(equal_var=False)` |
| `t.test(x, y, var.equal = TRUE)` | `t = 3.8878, df = 13, p-value = 0.001869` | SciPy `ttest_ind` |
| `t.test(x, mu = 5)` | `p-value = 0.01261` | SciPy `ttest_1samp` |
| `chisq.test(tab)` | `X-squared = 4.3221, p-value = 0.03762` | SciPy `chi2_contingency(correction=True)` |
| `chisq.test(tab, correct = FALSE)` | `X-squared = 5.7696, p-value = 0.01631` | SciPy `chi2_contingency(correction=False)` |
| `fisher.test(tea)` | `p-value = 0.4857`, odds ratio about `6.408` | exact `34/70`; SciPy conditional MLE `6.40832` (R finds the MLE with `uniroot` at its default tolerance, so trailing digits can differ) |
| `fisher.test(tea, alternative = "greater")` | `p-value = 0.2429` | exact `17/70` |
| `binom.test(10, 20, 0.3)` | `p-value = 0.08345` | exact rational sum; SciPy `binomtest` |
| `binom.test(7, 20)` | `p-value = 0.2632` | exact `2 P(X <= 7)` |
| `wilcox.test(before, after, paired = TRUE)` | `V = 40, p-value = 0.03906` | exact `20/512` by enumeration |
| `wilcox.test(a, b)` | `W = 35, p-value = 0.2544` | SciPy `mannwhitneyu(method="exact")` |
| `p.adjust(p, "holm")` | `0.040 0.090 0.090 0.025 0.200` | R's `cummax((n + 1 - i) p_(i))` by hand |
| `p.adjust(p, "BH")` | `0.025 0.050 0.050 0.025 0.200` | R's `cummin(n / i p_(i))` by hand |
| `p.adjust(p, "BY")` | `0.05708333 0.11416667 0.11416667 0.05708333 0.45666667` | BH times `sum(1/i) = 2.283333` |

## 4. Optimization, roots, and regression

| R | `probstat` | Notes |
|---|---|---|
| `optim(par, fn)` | `optimization.minimize_nelder_mead(fn, par)` | Nelder-Mead is R's default method |
| `optim(par, fn, gr, method = "BFGS")` | `optimization.minimize_bfgs(fn, par, gradient=gr)` | central differences when no gradient is given, as in R |
| `$par`, `$value`, `$counts`, `$convergence == 0` | `.x`, `.value`, `.evaluations`, `.converged` | `probstat` counts finite-difference calls in `evaluations` |
| `control = list(fnscale = -1)` | minimize `lambda x: -fn(x)` | |
| `uniroot(f, c(a, b))` | `optimization.brent_root(f, a, b)` | R's default `tol = .Machine$double.eps^0.25` (about `1.2e-4`); `probstat` defaults to `1e-12`, so roots agree only to R's tolerance |
| `nls(...)` | `optimization.least_squares(residual, x0)` | Levenberg-Marquardt (closer to `minpack.lm::nlsLM`); R's `nls` default is Gauss-Newton |
| `vcov(nls_fit)` | `optimization.gauss_newton_covariance(fit)` | `s^2 (J'J)^-1` |
| `lm(y ~ x)` | `inference.fit_simple_linear_regression(x, y)` | one predictor; `.intercept`, `.slope`, `.residuals`, `.r_squared`, `.residual_variance` (= `sigma(fit)^2`) |

```python
from probstat.optimization import brent_root, minimize_bfgs, minimize_nelder_mead
from probstat.inference import fit_simple_linear_regression

root = brent_root(lambda t: t ** 3 - t - 1.0, 1.0, 2.0)
print(f"uniroot(function(x) x^3 - x - 1, c(1, 2))$root ~ {root.root:.12f}")
quadratic = lambda v: (v[0] - 3.0) ** 2 + (v[1] + 1.0) ** 2 + 1.0
for result in (minimize_nelder_mead(quadratic, (0.0, 0.0)), minimize_bfgs(quadratic, (0.0, 0.0))):
    print(tuple(round(c, 6) for c in result.x), round(result.value, 10), result.converged)
fit = fit_simple_linear_regression((1, 2, 3, 4, 5), (2.1, 3.9, 6.2, 7.8, 10.1))
print(f"coef(lm) = ({fit.intercept:.4f}, {fit.slope:.4f}), R^2 = {fit.r_squared:.6f}, sigma^2 = {fit.residual_variance:.6f}")
```

```text
uniroot(function(x) x^3 - x - 1, c(1, 2))$root ~ 1.324717957245
(2.999996, -0.999998) 1.0 True
(3.0, -1.0) 1.0 True
coef(lm) = (0.0500, 1.9900), R^2 = 0.997305, sigma^2 = 0.035667
```

The real root of `x^3 = x + 1` is `1.324717957244746...` (mpmath); R's
`uniroot` returns it only to about four decimals at its default tolerance.
`lm(y ~ x)` on these data gives coefficients `0.05` and `1.99` (closed form
`S_xy / S_xx = 19.9 / 10`), `R^2 = 0.9973` and `sigma = 0.1889`.

## 5. Random numbers: `set.seed` versus `RandomStream`

`set.seed(42); rnorm(3)` and `Normal().sample(3, seed=42)` do **not** produce
the same numbers. Python's `random` module and R both default to the Mersenne
Twister, but the seeding procedures and the normal-variate algorithms differ.
Reproducibility therefore holds within `probstat`, not across languages; port
simulation studies by comparing distributions or Monte Carlo summaries, never
individual draws.

`foundations.RandomStream` gives each component of a study its own
reproducible, named substream (`root.child("inputs")`, `root.spawn(4)`). It
serves the purpose that `RNGkind("L'Ecuyer-CMRG")` with
`parallel::nextRNGStream()` serves in R: independent streams without manual
seed bookkeeping. Distribution `sample` methods take an integer seed (use
`stream.integer_seed()`); higher-level functions such as `inference.bootstrap`
and `sensitivity.sobol_analysis` accept a stream directly.

```python
from probstat.foundations import RandomStream

root = RandomStream(42)
first = Normal().sample(3, seed=root.child("inputs").integer_seed())
again = Normal().sample(3, seed=root.child("inputs").integer_seed())
other = Normal().sample(3, seed=root.child("noise").integer_seed())
print(first == again, first == other)
print(tuple(round(v, 6) for v in Normal().sample(3, seed=42)))
```

```text
True False
(-0.14409, -0.172904, -0.111316)
```

## 6. Bootstrap: `boot::boot` and `boot::boot.ci`

| R (`boot` package [REF-DAVISON-HINKLEY-1997]) | `probstat.inference` |
|---|---|
| `boot(data, function(d, i) mean(d[i]), R = 2000)` | `bootstrap(lambda sample: mean(sample), data, replicates=2000, seed=...)` |
| `boot.ci(b, type = "perc" / "basic" / "norm" / "stud" / "bca")` | `bootstrap_confidence_interval(stat, data, method="percentile" / "basic" / "normal" / "studentized" / "bca")` |
| `t0`, `t`, bias, std. error | `.estimate`, `.replicates`, `.bias`, `.standard_error` |

The statistic receives the resampled data itself, not an index vector.
Bootstrap results agree with `boot` only up to Monte Carlo error and each
package's quantile-interpolation rules, because the random streams differ.

```python
from probstat.inference import bootstrap, bootstrap_confidence_interval
from probstat.statistics import mean

data = (2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 2.8, 6.1, 3.9, 4.0)
result = bootstrap(mean, data, replicates=2000, seed=7)
print(f"t0 = {result.estimate:.4f}, bias = {result.bias:.4f}, se = {result.standard_error:.4f}")
interval = bootstrap_confidence_interval(mean, data, method="percentile", replicates=2000, seed=7)
print(f"percentile 95% interval = ({interval.lower:.4f}, {interval.upper:.4f})")
# The bootstrap estimates the plug-in standard error, which uses ddof=0.
print(f"plug-in se = {standard_deviation(data) / math.sqrt(len(data)):.4f}")
```

```text
t0 = 3.7500, bias = -0.0209, se = 0.4153
percentile 95% interval = (2.9398, 4.5700)
plug-in se = 0.4102
```

## 7. Sobol indices: `sensitivity::sobol`, `soboljansen`

| R (`sensitivity` package) | `probstat.sensitivity` |
|---|---|
| `soboljansen(model, X1, X2)` with two base samples | `estimate_sobol_indices(model, sample_a, sample_b)` |
| `sobolSalt`, `sobol2007`, ... on a model with declared input laws | `sobol_analysis(model, quantity, size, seed=...)` on a `modeling.ProbabilisticModel` |

`estimate_sobol_indices` uses a Saltelli pick-freeze estimator for first-order
indices and the Jansen estimator for total effects [EQ-SENS-FIRST-001,
EQ-SENS-TOTAL-001]; R's `sensitivity` package offers several estimators under
different function names, so match the estimator, not only the function, when
comparing. Indices are not clipped to `[0, 1]`.

```python
import random
from probstat.sensitivity import estimate_sobol_indices

def ishigami(v):
    return math.sin(v[0]) + 7.0 * math.sin(v[1]) ** 2 + 0.1 * v[2] ** 4 * math.sin(v[0])

generator = random.Random(2026)
draw = lambda: tuple(generator.uniform(-math.pi, math.pi) for _ in range(3))
sample_a = tuple(draw() for _ in range(4000))
sample_b = tuple(draw() for _ in range(4000))
sobol = estimate_sobol_indices(ishigami, sample_a, sample_b)
print("first", tuple(round(s, 2) for s in sobol.first_order))
print("total", tuple(round(s, 2) for s in sobol.total_effect))
print("analytic first (0.31, 0.44, 0.00), total (0.56, 0.44, 0.24)")
```

```text
first (0.31, 0.44, 0.02)
total (0.56, 0.43, 0.24)
analytic first (0.31, 0.44, 0.00), total (0.56, 0.44, 0.24)
```

The analytic Ishigami indices for `a = 7`, `b = 0.1` are
`S = (0.3139, 0.4424, 0)` and `S_T = (0.5576, 0.4424, 0.2437)`
[REF-SALTELLI-2008, Chapter 4]; with 4 000 base points the estimates carry
Monte Carlo error in the second decimal.

## 8. Multivariate normal: `mvtnorm`

| R (`mvtnorm` [REF-GENZ-BRETZ-2009]) | `probstat` |
|---|---|
| `dmvnorm(x, mean, sigma)` | `multivariate.MultivariateNormal(mean, sigma).pdf(x)` (`log_pdf` for `log = TRUE`) |
| `rmvnorm(n, mean, sigma)` | `MultivariateNormal(mean, sigma).sample(n, seed=...)` |
| `pmvnorm(upper = c(h, k), corr = R)` in two dimensions | `reliability.bivariate_normal_cdf(h, k, rho)` (standardized margins) |
| `pmvnorm` in three or more dimensions | not available |
| conditional laws (e.g. `condMVNorm::condMVN`) | `MultivariateNormal.conditional(indices, values)` |
| marginal of a subset | `MultivariateNormal.marginal_subset(indices)` |

```python
from probstat.multivariate import MultivariateNormal
from probstat.reliability import bivariate_normal_cdf

law = MultivariateNormal((0.0, 0.0), ((1.0, 0.5), (0.5, 1.0)))
print(f"dmvnorm(c(0.5, -0.2), sigma = S) = {law.pdf((0.5, -0.2)):.7f}")
print(f"pmvnorm(upper = c(1, 1), corr = S) = {bivariate_normal_cdf(1.0, 1.0, 0.5):.7f}")
print(law.conditional([1], [1.0]))
```

```text
dmvnorm(c(0.5, -0.2), sigma = S) = 0.1417010
pmvnorm(upper = c(1, 1), corr = S) = 0.7452036
MultivariateNormal(mean=(0.5,), covariance=((0.75,),))
```

| R call | R value | Independent check |
|---|---|---|
| `dmvnorm(c(0.5, -0.2), sigma = S)` | 0.141701 | SciPy `multivariate_normal.pdf` |
| `pmvnorm(upper = c(1, 1), corr = S)` | 0.7452036 | mpmath quadrature of Plackett's identity; SciPy `multivariate_normal.cdf`. `pmvnorm` itself is a randomized quasi-Monte Carlo integrator in general and reports an error estimate |

## Summary of traps when porting

1. `var`, `sd`, `cov`, `scale`: pass `ddof=1`.
2. `chisq.test` on 2 x 2 tables: pass `yates_correction=True`.
3. `fisher.test` estimate: conditional MLE in R, sample odds ratio here.
4. `dgamma(x, 2, 3)` means rate 3 in R; spell out `rate=` or `from_scale`.
5. `alternative = "two.sided"` becomes `"two-sided"`.
6. `uniroot` and `optim` tolerances differ; compare to the looser one.
7. Same seed, different numbers: compare distributions, not draws.

Previous: [probability and statistics refresher](probability-and-statistics-refresher.md).
