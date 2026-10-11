# Hypothesis Testing, Confidence Intervals, and the Bootstrap

`probstat.inference` provides classical tests, exact and resampling-based
confidence procedures, power analysis, and multiple-testing control. Every
procedure returns a frozen result that names its method, so a reader can see
which assumptions were used. All numbers below come from the library and are
also asserted in `tests/test_classical_resampling_inference.py`.

## 0. What the numbers mean

- **p-value.** Assume the null hypothesis and the test's model. The p-value is
  the probability of a test statistic at least as extreme as the observed one,
  in the direction of the alternative. It is **not** the probability that the
  null hypothesis is true. A large p-value is not evidence that the null holds.
- **Confidence level.** A 95% interval comes from a procedure that covers the
  fixed true value in 95% of repeated samples. A single computed interval
  either covers the value or does not; 95% is not a probability about that
  interval.
- **Power.** The probability of rejecting the null when a stated effect is
  true. Power is a design quantity. Computing it from the observed effect
  ("post-hoc power") adds nothing to the p-value.

## 1. Sampling distributions

When observations are independent and normal, the t, chi-square, and F laws
are the exact distributions of the classical test statistics. Each upper tail
is computed directly so that small p-values keep their accuracy:

| Function | Formula | Equation |
|---|---|---|
| `student_t_sf`, `student_t_two_sided_p` | `I_{nu/(nu+t^2)}(nu/2, 1/2)` | EQ-DIST-T-001 |
| `chi_square_sf` | `Q(k/2, x/2)` | EQ-INF-DIST-CHI2-001 |
| `f_sf` | `I_{d2/(d2+d1 f)}(d2/2, d1/2)` | EQ-INF-DIST-F-001 |
| `noncentral_t_cdf` | `E[Phi(t S - delta)]`, `S = sqrt(chi^2_nu / nu)` | EQ-INF-DIST-NCT-001 |

`noncentral_t_cdf` integrates the exact mixture representation with
2000-interval Simpson quadrature. It agrees with an independent reference
implementation to about `1e-11` on the tested points.

## 2. t tests and the Student-t interval

Assumptions: independent observations, normal data (or normal paired
differences), and, for the pooled test only, equal variances.

| Function | Statistic | df |
|---|---|---|
| `one_sample_t_test` | `(x_bar - mu_0)/(s/sqrt(n))` | `n - 1` |
| `paired_t_test` | same, on the differences | `n - 1` |
| `two_sample_t_test(equal_variances=True)` | pooled `s_p` | `n1 + n2 - 2` |
| `two_sample_t_test()` (Welch, default) | `s1^2/n1 + s2^2/n2` | Welch-Satterthwaite |

Worked example: the Cushny-Peebles sleep data used by Student (1908), ten
patients under two drugs.

- Welch test: `t = -1.8608`, `df = 17.776`, two-sided `p = 0.0794`.
- Paired test, the correct analysis because each patient received both drugs:
  `t = -4.0621`, `df = 9`, `p = 0.00283`. The 95% interval for the mean
  difference is `(-2.460, -0.700)`.

Ignoring the pairing hides a clear effect: the between-patient variation
swamps the within-patient difference.

`student_t_mean_confidence_interval` adds the exact
`x_bar +- t_{n-1,1-alpha/2} s/sqrt(n)` interval. The older
`normal_mean_confidence_interval` is unchanged and still labels its
estimated-sigma branch a large-sample approximation. A seeded simulation in
the test suite (2000 normal samples of size `n = 6`, seed 97) gives these
coverages:

- t interval: **0.953**.
- z interval with `s`: only **0.891**.

Every `TTestResult` reports a two-sided interval at `confidence`, whatever
alternative was tested, together with Cohen's `d` (pooled for two samples,
`d_z` for pairs).

## 3. Count data

- `chi_square_goodness_of_fit` (EQ-INF-CHI2-GOF-001). For Mendel's
  `315/108/101/32` peas against `9:3:3:1`: `X^2 = 0.470`, `df = 3`,
  `p = 0.925`. A small `X^2` means the data are consistent with the model; it
  does not prove the model.
- `chi_square_independence_test` (EQ-INF-CHI2-IND-001). For the
  `2 x 3` gender-by-party table `[[762, 327, 468], [484, 239, 477]]`:
  `X^2 = 30.07`, `df = 2`, `p = 2.95e-7`, Cramer's `V = 0.104`. The p-value
  is tiny but the association is weak: significance is not magnitude.
- Expected counts below five emit `SmallExpectedCountWarning` (Cochran's
  rule) and set `small_expected_counts = True`. In that case, use an exact
  test.
- `fisher_exact_test` (EQ-INF-FISHER-001). Fisher's tea-tasting table
  `[[3, 1], [1, 3]]` gives two-sided `p = 34/70 = 0.486` and one-sided
  `p = 17/70`.
- `binomial_test` (EQ-INF-BINOM-001). Seven successes in 20 trials against
  `p_0 = 0.3` give two-sided `p = 0.629`. The two-sided rule sums every
  outcome no more probable than the observed one. The result includes the
  Clopper-Pearson 95% interval `(0.154, 0.592)` (EQ-INF-CI-CP-001), whose
  coverage is never below the nominal level.

## 4. Rank tests

`mann_whitney_u_test` and `wilcoxon_signed_rank_test` compute exact null laws
from integer generating functions (EQ-INF-MWU-001, EQ-INF-WSR-001). The
automatic method chooses between two branches:

- **Exact law:** when there are no ties and sizes are at most 50.
- **Normal approximation:** used otherwise, with the tie-corrected variance
  and a continuity correction.

In the Wilcoxon test, zero differences are discarded. In the tests, exact
p-values are checked against brute-force enumeration of all `C(m+n, m)` rank
sets or all `2^n` sign vectors. Example: for the Hamilton depression-scale
pairs, the exact one-sided test gives `W+ = 40` and `p = 0.01953 = 10/512`.
Each result reports a rank-biserial effect size.

## 5. Permutation tests

`permutation_test` compares `mean(first) - mean(second)` with its
distribution over relabelings of the pooled data (EQ-INF-PERM-001). The test
is valid whenever the observations are exchangeable under the null. It runs
in one of two modes:

- **Exact enumeration:** used when there are at most `max_exact` relabelings.
  The five-versus-six example in the tests has 462 relabelings and a one-sided
  `p = 7/462 = 0.0152`.
- **Monte Carlo:** used otherwise. It reports `(b + 1)/(M + 1)`, which is
  never zero and keeps the test's level exact (Phipson and Smyth). It is
  seeded through `RandomStream`.

## 6. Bootstrap and jackknife

`bootstrap(statistic, data, replicates, seed)` returns the replicates, the
bootstrap bias `mean(theta*) - theta`, and the standard error (EQ-INF-BOOT-001).
Observations may be any objects, for example pairs.

`jackknife` gives the deterministic leave-one-out bias and standard error
(EQ-INF-JACK-001). Two exact identities are tested:

- For the mean, the jackknife standard error equals `s/sqrt(n)`.
- For the plug-in variance, the bias-corrected estimate equals the unbiased
  variance.

`bootstrap_confidence_interval` offers five methods:

| Method | Interval | Accuracy |
|---|---|---|
| `percentile` | `[q_{alpha/2}, q_{1-alpha/2}]` | first order |
| `basic` | `[2 theta - q_{1-alpha/2}, 2 theta - q_{alpha/2}]` | first order |
| `normal` | `theta - bias +- z se` | first order |
| `studentized` | bootstrap-t with supplied or jackknife SE | second order |
| `bca` | bias correction `z_0`, jackknife acceleration `a` | second order |

Worked example: `n = 200` normal observations, seed 20261011, 2000
replicates, seed 7.

| Interval | Lower | Upper | Diagnostics |
|---|---|---|---|
| BCa | 9.710 | 10.307 | `z_0 = -0.010`, `a = 0.0018` |
| Student-t | 9.714 | 10.314 | |

For a smooth statistic on well-behaved data, the bootstrap reproduces the
analytic answer, as it should. BCa raises an error in two cases, rather than
returning a meaningless interval:

- every replicate lies on one side of the estimate;
- the jackknife values are constant.

## 7. Power, effect sizes, sample size

- `cohens_d` and `hedges_g` (EQ-INF-ES-001). `g` applies Hedges' exact
  factor `J(m) = Gamma(m/2) / (sqrt(m/2) Gamma((m-1)/2))`.
- `z_test_power` (EQ-INF-POWER-001) and `t_test_power` (EQ-INF-POWER-002).
  t power uses the **exact noncentral t law** (quadrature), with no normal
  approximation and no simulation. A seeded simulation in the tests agrees
  within Monte Carlo error.
- `required_sample_size` (EQ-INF-POWER-003). Values for `d = 0.5`,
  `alpha = 0.05` two-sided, and target power 0.8:

| Design and test | Required `n` | Achieved power |
|---|---|---|
| Two-sample t | 64 per group | 0.8015 |
| One-sample or paired t | 34 | 0.8078 |
| Two-sample z | 63 per group | 0.8013 |

## 8. Multiple testing

`adjust_p_values` (EQ-INF-MT-001, EQ-INF-MT-002) returns adjusted p-values
and rejection sets. For the vector `p = (0.01, 0.04, 0.03, 0.005)`:

| Method | Adjusted p-values | Controls |
|---|---|---|
| Bonferroni | 0.04, 0.16, 0.12, 0.02 | FWER, any dependence |
| Holm | 0.03, 0.06, 0.06, 0.02 | FWER, any dependence |
| BH | 0.02, 0.04, 0.04, 0.02 | FDR, independence or PRDS |
| BY | 0.0417, 0.0833, 0.0833, 0.0417 | FDR, any dependence |

At `alpha = 0.05`, the methods reject:

- Holm: the first and fourth hypotheses;
- BH: all four.

## 9. Equivalence (TOST)

`tost_equivalence_test` (EQ-INF-TOST-001) makes equivalence the alternative
hypothesis. It runs two one-sided t tests at the margins, and its p-value is
the larger of the two.

Example: ten differences with mean 0.02 and margins `(-0.2, 0.2)`:

- p-value: `p = 5.1e-6`;
- the 90% interval `(-0.017, 0.057)` lies inside the margins, so equivalence
  is declared.

A non-significant difference test is never a substitute for TOST.

## Limitations

- Exact rank and permutation branches are pure-Python enumerations. They are
  limited to moderate sizes (`max(m, n) <= 50` for automatic exact rank
  tests, `max_exact` relabelings for permutations).
- The two-sided binomial and Fisher tests use the minimum-likelihood
  convention. Other conventions, such as doubling the smaller tail, give
  different p-values.
- Power functions cover one-sample, paired, and equal-size two-sample mean
  tests only.
- Studentized bootstrap with the default jackknife SE costs `n` statistic
  evaluations per replicate.

## Sources

[REF-STUDENT-1908], [REF-WELCH-1947], [REF-SATTERTHWAITE-1946],
[REF-CASELLA-BERGER-2002], [REF-AGRESTI-2013], [REF-COCHRAN-1954],
[REF-CLOPPER-PEARSON-1934], [REF-MANN-WHITNEY-1947], [REF-WILCOXON-1945],
[REF-HOLLANDER-WOLFE-CHICKEN-2014], [REF-GOOD-2005], [REF-PHIPSON-SMYTH-2010],
[REF-EFRON-1979], [REF-EFRON-1987], [REF-EFRON-TIBSHIRANI-1993],
[REF-DAVISON-HINKLEY-1997], [REF-COHEN-1988], [REF-HEDGES-1981],
[REF-JOHNSON-KOTZ-BALAKRISHNAN-1995], [REF-HOLM-1979],
[REF-BENJAMINI-HOCHBERG-1995], [REF-BENJAMINI-YEKUTIELI-2001],
[REF-LEHMANN-ROMANO-2005], [REF-SCHUIRMANN-1987].
