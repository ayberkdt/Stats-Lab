# ADR-0101: Robust estimation, quantile regression, and the missing classical tests

- Status: Accepted
- Date: 2026-11-12

## Context

`probstat.inference` covered two-sample location tests (t, Welch,
Mann-Whitney, Wilcoxon signed-rank), the one-way ANOVA, count-data tests,
EDF normality tests, the bootstrap, and multiple-testing adjustment
(`adjust_p_values`), but an inventory of the `inference` and `statistics`
exports found no Kruskal-Wallis, Friedman, sign, Levene, Brown-Forsythe,
Bartlett, Shapiro-Wilk, Cochran's Q, Tukey HSD, or Dunn test, no
studentized range distribution, no robust M-estimator of location or
regression, no Theil-Sen slope, and no quantile regression. The only
quantile-regression code was the conformal correction in
`conformal.conformalized_quantile_regression`, which wraps user-supplied
quantile predictors and does not fit any. `statistics` already provided
`median_absolute_deviation` and `ranks`, and `inference` the binomial test,
the one-way ANOVA, and the chi-square, normal, and t tails that the new
tests reuse.

## Decision

Four new modules, each one concept; no existing public name changes.

1. **`inference.robust`** (EQ-INF-ROB-001 to 004). `robust_location` solves
   `sum psi((x - mu)/s) = 0` by IRLS from the median with `s` the normalized
   MAD (fixed); `fit_robust_linear_model` iterates
   `s = 1.4826 median|r|`, weights `psi(u)/u`, and a weighted QR solve (the
   `MASS::rlm` scheme with `scale.est = "MAD"`) to the fixed point
   `sum psi(r_i/s) x_i = 0`. Huber (`k = 1.345`) and Tukey biweight
   (`c = 4.685`) scores, both 95% efficient at the normal law. The biweight
   starts from median regression by default because its root depends on
   the start. Standard errors are the empirical sandwich
   `s^2 n/(n-p) A^-1 B A^-1`. `theil_sen_slope` returns the median of
   pairwise slopes with Sen's interval (Kendall variance corrected for ties
   in `x`). MM- and S-estimators are out of scope.
2. **`inference.quantile_regression`** (EQ-INF-QR-001, 002). An exact
   simplex walk on elemental bases in the manner of Barrodale and Roberts:
   at basis `h` the `2p` edge derivatives of the check loss are computed
   from `c_i = x_i' X_h^-1`, the steepest descending edge is followed by an
   exact weighted-median line search (several vertices per step), and every
   step strictly lowers the objective, so the walk terminates without an
   anti-cycling rule. At a degenerate vertex the edges of every elemental
   basis of the zero-residual set (one copy of each repeated design row)
   are examined, which certifies `0 in dR(beta)`; beyond 20000 bases the
   result is flagged `optimality_certified=False` with a warning.
   Intervals come from the pairs bootstrap (percentile limits). An
   interior-point method and rank-inversion intervals were not added: the
   simplex is exact and fast enough for a few thousand rows, and the
   bootstrap needs no sparsity estimate.
3. **`inference.studentized_range`** (EQ-INF-SRANGE-001). The distribution
   function as the double integral `int f_nu(s) P_k(q s) ds`: fixed
   16-point Gauss-Legendre on nine width-two panels of `[-9, 9]` for the
   inner range law (with `Phi(z)` precomputed at the nodes) and adaptive
   Gauss-Kronrod (`numerics.gauss_kronrod`) over the bulk of the law of
   `S`; quantiles by Brent's method.
4. **`inference.classical_tests`** (EQ-INF-KW-001, FRIEDMAN-001, SIGN-001,
   LEVENE-001, BARTLETT-001, SW-001, COCHRANQ-001, TUKEY-001, DUNN-001).
   Kruskal-Wallis enumerates all `N!/prod n_g!` assignments of the observed
   mid-ranks when there are at most `max_exact = 100000` (exact integer
   comparison in doubled ranks), otherwise chi-square; Levene and
   Brown-Forsythe reuse `one_way_anova` on absolute deviations; the sign
   test reuses `binomial_test`; Dunn's test reuses `adjust_p_values`;
   Shapiro-Wilk implements AS R94 with the published polynomial constants.
   Results share one record, `ClassicalTestResult`; post hoc procedures
   return `PairwiseComparisonResult`.

The guide `docs/foundations/robust-and-quantile-regression.md` (learning
step S16) and section 10 of `docs/foundations/hypothesis-testing-and-bootstrap.md`
document the methods with a which-test-when table and pitfalls.

## Consequences

- 29 public names are added to `probstat.inference`; 16 equation records
  and 24 bibliography entries are appended. The README equation count and
  the generated API and mathematical references must be regenerated.
- `inference.robust` imports `inference.quantile_regression` for the
  biweight start; the reverse import does not exist.
- Each studentized-range evaluation costs about 40 ms and a quantile 0.4 to
  0.7 s, so a Tukey HSD with `k` groups costs roughly
  `0.5 s + 0.04 k(k-1)/2 s`.
- Bootstrap resamples repeat rows, which makes degenerate vertices common;
  deduplicating repeated design rows in the degeneracy search cut the
  bootstrap reproducibility test (two runs of 100 stack-loss resamples)
  from 1.97 s to 0.43 s.

## Verification

Oracles were computed offline with SciPy 1.14.1 and NumPy 2.2.6, which are
not test dependencies; R was not available, so no `MASS::rlm`,
`robustbase`, `quantreg::rq`, or `ptukey` value is quoted.

| Method | Oracle | Agreement |
|---|---|---|
| Huber location | `scipy.optimize.brentq` on the estimating equation | `1e-12` |
| Biweight location | same | `1e-12` |
| Location on symmetric data | closed form (the centre) | `1e-12` |
| Huber regression (stack loss) | `scipy.optimize.fsolve` on `X' psi(r/s(b)) = 0` | coefficients `1e-7` relative, scale `1e-8` |
| Biweight regression (stack loss) | independent NumPy IRLS from least squares | `1e-7` relative |
| Sandwich standard errors | NumPy recomputation at the oracle root | `1e-6` relative |
| Location and slope intervals | seeded coverage, 400 and 300 replications | within 3 binomial SE of 0.95 |
| Theil-Sen slope and interval | `scipy.stats.theilslopes(method="joint")`, ties in `x` | `1e-14` |
| Quantile regression (stack loss, `tau = 0.1 ... 0.9`) | `scipy.optimize.linprog(method="highs")` objective | `1e-9` |
| Quantile regression (median) | enumeration of all 5985 elemental bases | `1e-9` |
| 40 random small designs, integer ties included | enumeration of all elemental bases | `1e-9` |
| 300 bootstrap-style resamples (prototype) and 200 random designs | `linprog` objective | no mismatch above `1e-8` |
| Intercept-only model | sample quantile `x_(ceil(n tau))` | exact |
| Bootstrap interval | seeded coverage, 60 data sets | within 3 binomial SE of 0.90 |
| Studentized range cdf | `scipy.stats.studentized_range.cdf` on 630 points (`k <= 20`, `nu = 1 ... 1000, inf`) | max `6.4e-13` |
| Studentized range, `k = 2` | `2 Phi(q/sqrt 2) - 1`, `P(abs(T_nu) <= q/sqrt 2)` | `1e-12` |
| Studentized range quantile | `studentized_range.ppf(0.95)` | `1e-9` |
| Tukey HSD | `scipy.stats.tukey_hsd` p-values and 95% intervals | `1e-10` and `1e-9` |
| Kruskal-Wallis (chi-square) | `scipy.stats.kruskal` | `1e-12` |
| Kruskal-Wallis (exact) | brute-force relabelling, with and without ties; 84/1680 for `H >= 5.6` | exact |
| Friedman | `scipy.stats.friedmanchisquare` | `1e-12` |
| Levene, Brown-Forsythe, Bartlett | `scipy.stats.levene`, `bartlett` | `1e-11` |
| Shapiro-Wilk, `n = 4 ... 1000` | `scipy.stats.shapiro` (Fortran `swilk`) | `W` `1e-8`, p `1e-6` relative |
| Shapiro-Wilk, `n = 3` | exact arcsine law, `W = 27/28` | `1e-14` |
| Sign test | `22/1024`, `11/1024` | `1e-15` |
| Cochran's Q | McNemar without correction (`k = 2`), hand computation | `1e-14` |
| Dunn | exact rational recomputation, `adjust_p_values` | `1e-13` |

The Shapiro-Wilk constants `C1` to `C6` and `G` were compared digit by digit
with the StatLib listing of AS R94 [REF-STATLIB-AS-R94]; the `n = 3`
constant uses `sqrt(1/2)` and `6/pi` in full precision instead of the
listing's single-precision literals. The stack-loss values were checked
against the Rdatasets export of R's `stackloss` [REF-R-DATASETS-STACKLOSS].

Tests: `tests/test_robust_estimation.py`, `tests/test_quantile_regression.py`,
`tests/test_classical_tests.py`.

References: [REF-HUBER-1964], [REF-HUBER-1981],
[REF-MARONNA-MARTIN-YOHAI-2006], [REF-BEATON-TUKEY-1974], [REF-SEN-1968],
[REF-KOENKER-BASSETT-1978], [REF-KOENKER-2005], [REF-BARRODALE-ROBERTS-1973],
[REF-KRUSKAL-WALLIS-1952], [REF-FRIEDMAN-1937], [REF-BROWN-FORSYTHE-1974],
[REF-BARTLETT-1937], [REF-SHAPIRO-WILK-1965], [REF-ROYSTON-1992],
[REF-ROYSTON-1995], [REF-COCHRAN-1950], [REF-KRAMER-1956],
[REF-COPENHAVER-HOLLAND-1988], [REF-LUND-LUND-1983],
[REF-HOCHBERG-TAMHANE-1987], [REF-DUNN-1964], [REF-DIXON-MOOD-1946].
