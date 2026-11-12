# ADR-0091: Defect review of inference, statistics, information, causal, design, and conformal

- Status: Accepted
- Date: 2026-11-12

## Context

`probstat.inference` (without `inference.ppl`), `probstat.statistics`,
`probstat.information`, `probstat.causal`, `probstat.design`, and
`probstat.conformal` hold about 40,000 lines and several hundred public
callables. This review checked the public functions module by module
against independent oracles: SciPy 1.14 and mpmath 1.3 evaluated offline
(never imported by the tests), closed forms, exact rational arithmetic,
Marsaglia-Tsang-Wang matrix powers at 80 digits, union and Harris-FKG
bounds, hand counts, and seeded simulations whose tolerance comes from the
binomial standard error. Edge cases covered `n = 1, 2`, ties, zero variance,
empty groups, boundary probabilities, extreme levels, and NaN or infinite
inputs.

## Decision

Five defects were demonstrated by a failing test and fixed; a sixth is a
documentation gap. The regression tests are in
`tests/test_review_inference_statistics.py`.

1. **Type-1 sample quantile overshoots by one order statistic**
   (`statistics.quantile(..., method=1)`, `quantiles`). The index
   `ceil(n p)` was computed in floating point: `100 * 0.07` is
   `7.000000000000001`, so `Q(0.07)` of `1..100` returned 8 although
   `F_n(7) = 0.07` already reaches `p` ([REF-HYNDMAN-FAN-1996], type 1 is
   the inverse empirical CDF). Of the levels `k/100` and sizes up to 200,
   22 pairs were affected. The index is now corrected against `j / n >= p`
   evaluated on the same floats as `empirical_cdf`, and agrees with
   `ceil(n Fraction(p))` on all 30,300 pairs with `n <= 300` and `p = k/100`.
2. **Trimmed mean trims one value too few** (`statistics.trimmed_mean`).
   `floor(proportion n)` in floating point gave 28 for `0.29 * 100 =
   28.999999999999996` (also `n = 180, 0.35` and `n = 200, 0.29`). The count
   is now the largest `g` with `g / n <= proportion`, which equals
   `floor(n Fraction(proportion))` on every case tested ([EQ-STAT-TRIM-001]).
3. **Exact one-sample Kolmogorov tail loses probability for larger `n`**
   (`inference.goodness_of_fit.kolmogorov_exact_sf`, `_kolmogorov_noe_cdf`).
   The Noé recursion propagated `Q_m(c)`, proportional to `t^c / c!`, which
   falls by hundreds of orders of magnitude across the admissible band
   `n t +/- n d`. Counts above the lower boundary underflowed to zero and
   their probability was lost: `kolmogorov_exact_sf(700, 0.1)` returned
   0.00288 (exact 1.51e-6), `(800, 0.0972)` 0.0437 (exact 4.92e-7), and
   `(5000, 0.01)` 0.838, above the union bound `2 P(D_n^+ >= d) = 0.731`.
   The gaps are now scaled by `n`, so `Q_m` is a Poisson law centred in the
   band, and `P(D_n < d) = n! n^-n Q_M(n)` ([REF-NOE-1972],
   [EQ-INF-GOF-KS-004]). `kolmogorov_smirnov_test` calls the recursion only
   for `n <= 100`, where it was already correct, so its p-values do not
   change beyond `1e-13`. The docstring now states the accuracy and run
   time that were measured instead of "about `1e-14`, practical up to a few
   hundred".
4. **Adaptive conformal rank differs from split conformal**
   (`conformal.AdaptiveConformalPredictor.radius`). It computed
   `ceil((1 - alpha_t)(m + 1))` in binary floating point, while
   `conformal_rank` uses exact rational arithmetic on the decimal level
   ([REF-GIBBS-CANDES-2021]). With `gamma = 0`, which is documented to
   reproduce split conformal prediction, `alpha = 0.7` and nine scores gave
   rank 4 instead of 3; 61 of the `(alpha, n)` pairs scanned disagreed.
   `radius` now calls `conformal_rank`.
5. **Entropy of a point mass is `-0.0`** (`information.entropy`,
   `joint_entropy`, `conditional_entropy`). `-(1 log 1)` is `-0.0`, and the
   clamp `max(-0.0, 0.0)` keeps the negative zero, so reports printed
   `-0.0 bits`. The sums now add `0.0`, which maps `-0.0` to `+0.0` and
   leaves every other value unchanged.
6. **Efficient rounding of a raw grid design wastes runs**
   (`design.efficient_rounding`, documentation). Pukelsheim-Rieder rounding
   gives each support point at least one run ([REF-PUKELSHEIM-RIEDER-1992]);
   the grid algorithms of `d_optimal_design` leave weights of `4e-5` next to
   the optimum, so seven runs of the quadratic D-optimal design gave counts
   `(2, 1, 1, 1, 2)` on five points. The docstring now says so and points to
   `ApproximateDesign.condensed`, which gives `(3, 2, 2)` on `{-1, 0, 1}`.

Candidates that were examined and rejected (the code was right, or the
behaviour is documented):

- `effective_sample_size` differs by up to a third, for antithetic chains,
  from a transcription of the Stan C++ estimator written for this review;
  it follows the R code of `posterior` ([REF-BURKNER-ETAL-POSTERIOR]) step
  for step, which is the cited reference. Split and rank-normalized R-hat
  agree with an independent implementation of [REF-VEHTARI-ETAL-2021] to
  `1e-12`.
- The two-sample Kolmogorov-Smirnov p-value beyond `m n = 250000` is the
  limiting Kolmogorov law (0.367 against an exact 0.353 at `600 x 500`); the
  method label says so.
- `wilcoxon_signed_rank_test` drops zero differences and stays exact, while
  SciPy switches to the normal approximation; the convention is documented.
- `student_t` fits that run to `nu = 1000` report the bound in the message;
  `poisson` and `negative-binomial` count fits refuse all-zero data;
  `normality_tests` requires eight observations; `fit_gaussian_mixture`
  with one point per component returns the variance floor set by
  `regularization`. All are documented choices.
- `renyi_divergence_gaussian` returns `inf` when `alpha S1 + (1 - alpha) S0`
  is not positive definite, which is the correct value.
- `hierarchical_normal_posterior` means and quantiles agree with mpmath
  quadrature to `1e-10` and `4e-6`; SciPy's `quad` was the less accurate
  oracle.
- SciPy's `kstwo` is off by up to `1.4e-6` for `n = 200 .. 1000`; the
  library's values matched the 80-digit matrix formula there.
- `mcnemar_test` reports the corrected chi-square statistic next to the
  exact p-value; the result docstring defines it that way.
- The P² streaming quantile is inaccurate for heavy tails at `n = 200`
  (0.80 against 1.22 for a cubed normal) and accurate at `n = 20000`; its
  docstring states that it has no error bound.

## Consequences

- Type-1 quantiles and trimmed means now match the textbook definitions at
  every level whose product with `n` is an integer; results at other levels
  are unchanged.
- `kolmogorov_exact_sf` returns correct values for large samples; at
  `n = 5000, d = 0.01` it gives 0.69545572563 (SciPy 0.69545572539, union
  bound 0.731) in about four minutes. The one-sample KS test is unchanged.
- Adaptive conformal intervals at a frozen level are identical to split
  conformal intervals; with `gamma > 0` the rank can shift by one at levels
  whose binary and decimal products straddle an integer. The long-run
  bound of Gibbs and Candès holds for any fixed rank rule.
- No seeded numeric result changes except through these fixes.

## Verification

- Hypothesis tests against SciPy on 20 random sample pairs and three
  alternatives each: one-sample, pooled and Welch t (180 comparisons), exact
  Mann-Whitney and signed-rank (120), one-way ANOVA (20), tie-corrected
  normal approximations with and without continuity correction (120),
  Fisher exact, chi-square with and without Yates, and the exact binomial
  test (160): all within `1e-7` relative.
- Sampling distributions against SciPy at 200 random points each: Student
  t survival and quantile, chi-square and F survival (within `1e-9`), and
  the noncentral t CDF for `df >= 1` (within `1e-7`).
- Intervals and adjustments: Clopper-Pearson, Wilson, and Student-t mean
  intervals at 60 random settings up to level `0.999999`; Bonferroni, Holm,
  BH, and BY on 30 vectors including ties; the Garwood Poisson interval;
  KS and AD statistics, KS exact p-values for `n <= 40`, and two-sample KS
  for samples up to 400: all matched.
- Seeded size checks (3000 normal samples, standard error 0.004): the
  Lilliefors, Cramér-von Mises, and Anderson-Darling tests reject at
  0.045-0.050 for `n = 8, 20, 100`. On discrete data (1000 samples,
  standard error 0.007) the two-sample KS, Mann-Whitney, and signed-rank
  tests reject at 0.031, 0.051, and 0.036.
- Regression and GLMs against NumPy and SciPy optimizers: weighted least
  squares (coefficients, standard errors, `R^2`, log-likelihood, AIC, BIC,
  F test), weighted binomial and Poisson GLMs with offsets, a gamma GLM with
  the R dispersion and AIC conventions, elastic net (objective minimized by
  Powell's method), ridge with its effective degrees of freedom, and the
  lasso `lambda_max`: all within `1e-7`.
- Survival against SciPy and direct likelihood maximization: Kaplan-Meier
  with log-log intervals, Nelson-Aalen, the log-rank test, Cox partial
  likelihood with Breslow and Efron ties, and exponential, Weibull, and
  lognormal AFT fits.
- Count models, beta-binomial empirical Bayes, Rubin's rules (with the
  Barnard-Rubin degrees of freedom), conjugate Gaussian and
  normal-inverse-gamma updates, the posterior predictive, the Laplace
  approximation of a Gaussian, and the estimator-risk closed forms: all
  matched their formulas.
- Information measures: the discrete entropies, cross-entropy, KL,
  Jensen-Shannon, Rényi (four orders), alpha (three orders), Hellinger,
  Bhattacharyya, total variation, and mutual information; the Gaussian KL,
  Bhattacharyya, Hellinger, Rényi, Wasserstein, Bures, and total-variation
  closed forms; twelve differential entropies; five Fisher information
  matrices; and seven Fisher-Rao distances, against SciPy, NumPy, or
  quadrature.
- Design: D-optimal designs of degree 1-3 on a 41-point grid recover the
  known supports and weights, and the equivalence theorem holds within
  `1e-3`; D, A, I, and E criteria match NumPy.
- Causal: the E-value, bias factor, IPW, stabilized IPW (influence
  function of the Hajek form), AIPW, and the backdoor, minimal-adjustment,
  and frontdoor graph criteria were read against [REF-VANDERWEELE-DING-2017]
  and [REF-PEARL-2009]; no numeric oracle was run for this package.
- Defect 3: the corrected recursion agrees with the Marsaglia-Tsang-Wang
  formula at 80 digits within `4e-13` at eight points with
  `n = 100 .. 1000`, and every value tested for `n = 700, 800` lies within
  `2e-12` of the bracket `[2 p+ - p+^2, 2 p+]`.
- Eight of the ten new tests failed on the unfixed code and pass after the
  fixes (the other two pin the documented rounding workflow and an 80-digit
  matrix-formula value);
  `tests/test_documentation_contract.py`, `tests/test_architecture.py`, and
  the existing tests of every touched package pass; ruff and strict mypy
  report no issue.
