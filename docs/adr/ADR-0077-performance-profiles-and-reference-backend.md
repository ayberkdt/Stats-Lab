# ADR-0077: Performance profiles and a multiple-precision reference backend

- Status: Accepted
- Date: 2026-11-11

## Context

The cross-cutting roadmap item "Performance profiles and
multiple-precision/reference backends" names two gaps.

1. The benchmark suite (ADR-0049) reports, per problem, each method's error,
   calibration, and cost, and "What the numbers say" names the cheapest
   accurate method per problem. Nothing aggregates cost-to-accuracy over a
   *set* of problems, so a reader cannot see which method is reliably cheap
   across a category, how far behind the others are, or how robust a ranking
   is to one problem.
2. The float special-function kernels (`distributions._special`, the normal
   distribution, `math.lgamma`) are tested against tolerances and a few
   closed forms. There is no independent high-precision reference and no
   statement of their accuracy in units in the last place (ULPs), so a
   regression of a few hundred ULPs, or a tail where accuracy degrades with
   the argument, is invisible.

Constraints: standard library only; strict mypy; every definition documented
with source keys; the report tables have at most seven columns; plotting is
optional and must not break headless runs.

## Decision

1. **Profiles live in `numerics.profiles` (layer 0).** Performance ratios
   `r_(p,s) = t_(p,s) / min_s t_(p,s)` and profiles
   `rho_s(tau) = |{p : r_(p,s) <= tau}| / n_p` follow Dolan and Moré
   (2002, Section 2) [EQ-NUM-PROFILE-001, EQ-NUM-PROFILE-002]; data profiles
   `d_s(kappa) = |{p : t_(p,s)/(n_p + 1) <= kappa}| / |P|` and the
   convergence test `f(x_0) - f(x) >= (1 - tau)(f(x_0) - f_L)` follow Moré
   and Wild (2009) [EQ-NUM-PROFILE-004]. They are pure functions of a cost
   table with no model, estimator, or validation semantics, so they belong
   with the other deterministic numerical diagnostics rather than in
   `validation` (layer 5), whose subject is model-versus-data evidence;
   placing them in layer 0 also lets `optimization`, `monte_carlo`, or any
   higher package compare its own solvers without an upward import.
   Public names: `performance_ratios`, `performance_profile`,
   `PerformanceProfile` (`value`, `wins`, `solved_fraction`, `tau_at`,
   `area`, `breakpoints`, `max_ratio`), `data_profile`, `DataProfile`
   (`value`, `kappa_at`, `breakpoints`), and `first_converged_evaluation`.
2. **Failures have ratio infinity.** `None`, `inf`, or `NaN` in the cost
   table marks a failure; a problem no solver solves has infinite ratios for
   all. This is the source's `r_M` convention with `r_M` sent to infinity,
   which changes no profile value below `r_M`. Ties give every tied solver a
   ratio of 1, so `rho_s(1)` is the fraction of wins counting ties.
3. **Area summary.** `PerformanceProfile.area` is the mean of `rho_s` over
   `log2 tau` in `[0, log2 tau_max]`, computed exactly from the steps as
   `(1/(n_p L)) sum (L - log2 r)` [EQ-NUM-PROFILE-003], with `tau_max`
   defaulting to the largest finite ratio of any solver so that one table
   shares an axis. It is a library summary, not a quantity of the source;
   the profile itself remains the primary output.
4. **Benchmark report.** `benchmarks/report.py` ends each category with a
   "Performance profiles" subsection. A problem instance is a (problem,
   quantity group) pair as scored by "What the numbers say" (calibration:
   the posterior mean only). A method solves an instance when its median
   error meets the *accurate* threshold of that section (5 % relative, 0.02
   absolute for indices, 0.05 posterior sd for calibration means), at a cost
   of its mean model evaluations; inaccurate, failed, skipped, and
   inapplicable runs are failures. Methods that record no model evaluations
   (closed forms) cannot enter a ratio and are listed under the table. Two
   tables follow: performance profile (accurate on, `rho(1)`, `rho(4)`,
   `rho(16)`, `tau` for 50 %, area) and data profile (`d(kappa)` at
   `kappa = 10, 100, 1000, 10^4` simplex gradients and `kappa` for 50 %),
   both within the seven-column limit, rows sorted by area. The threshold
   table `SOLVED_ERROR` is now shared by "What the numbers say".
5. **Optional figure.** `visualization.profile_plots.plot_performance_profile`
   draws the profiles as right-continuous step lines on a `log2 tau` axis
   with a fixed `[0, 1]` share axis in the semantic theme (ADR-0024): the up
   to four leading distinct profiles (by area) take series hues 0, 3, 4, 5
   with direct labels, solvers with identical ratios share one line and are
   named in the subtitle, the rest are thin muted context lines, and the
   title is computed from the data. Leading lines are wider and drawn below
   later ones, so a coinciding segment shows as a stripe. Labels are set at
   the label size and moved toward the ink, keeping their hue, until they
   reach 4.5:1 contrast with the canvas. `benchmarks/run.py --figures DIR`
   writes a light (`PORCELAIN`) and a dark (`GRAPHITE`) PNG per category and
   embeds them with a `<picture>` element that follows the reader's colour
   scheme; without the flag, or without Matplotlib, nothing is drawn, so
   headless and dependency-free runs are unchanged. The committed report is
   generated without figures.
6. **Reference backend in `numerics.reference` (layer 0).** Functions return
   a `decimal.Decimal` rounded to `digits` significant digits (default: the
   active decimal context's precision) and evaluate at `digits + 10` guard
   digits:
   - `reference_pi` (arcsine series, tail below a third of the last term)
     [EQ-NUM-REFERENCE-001]; `reference_exp`, `reference_log` (the correctly
     rounded `Decimal.exp` and `Decimal.ln`);
   - `reference_erf`, `reference_erfc`, `reference_normal_cdf`: the
     positive-term series DLMF 7.6.2 below `max(3, sqrt(w)/2)` and Laplace's
     continued fraction DLMF 7.9.1 above; the series stops when its term
     ratio is at most 1/2 and the term is below tolerance (geometric tail
     bound), the fraction when successive convergents agree (they bracket
     the value because all elements are positive) [EQ-NUM-REFERENCE-002];
   - `reference_normal_quantile`: Newton steps on `log Phi` from the float
     quantile or a tail asymptote, upper tail by exact `1 - p`
     [EQ-NUM-REFERENCE-004];
   - `reference_log_gamma`: Stirling's series at `z = x + m >= 0.37 w + 10`
     with Bernoulli numbers from DLMF 24.5.3, stopped before the first term
     below tolerance, which bounds the remainder for real `z` (DLMF
     5.11(ii)); no Lanczos or Spouge coefficients [EQ-NUM-REFERENCE-005];
   - `reference_regularized_gamma_p` / `_q`: the series DLMF 8.7.1 for
     `x < a + 1` and the continued fraction DLMF 8.9.2 otherwise
     [EQ-NUM-REFERENCE-006]; `reference_regularized_beta`: the continued
     fraction DLMF 8.17.22-23 below `x = (a+1)/(a+b+2)` and the symmetry
     8.17.4 above [EQ-NUM-REFERENCE-007]; both fractions use the modified
     Lentz method [EQ-NUM-REFERENCE-003]. These fractions can have negative
     elements, so their stopping test is not a proven bound; the tests
     cross-check them against the series and against exact forms.
   - `exact_regularized_beta`: the binomial sum DLMF 8.17.5 in `Fraction`
     arithmetic for integer shapes.
   Error control beyond truncation: complements `1 - v` are recomputed with
   `-log10(1 - v)` extra digits; `exp(E)` prefactors carry `log10 |E|`
   extra digits; `log Gamma` near its zeros carries
   `log10(|log Gamma(z)| / |log Gamma(x)|)` extra digits; `Phi(x)` carries
   `log10(1 + x^2)` for the rounding of `x/sqrt 2`.
7. **Verification harness.** `ulp_error(value, reference)` measures
   `|value - reference| / ulp(fl(reference))` [EQ-NUM-REFERENCE-008];
   `compare_to_reference(function, reference, points, digits=40)` evaluates
   the reference inside a decimal context of `digits` digits and returns a
   `ReferenceComparison` (per-point ULP and relative errors, `max_ulp`,
   `mean_ulp`, `worst_point`, `within(budget)`).

## Consequences

- Benchmark readers get a cross-problem ranking of cost-to-accuracy per
  category; with quick-mode budgets and four to six instances per category
  the profiles are coarse, and a method run on only one problem of a
  category (System MC, System IS) is a failure on the others by
  construction.
- The ULP budgets below are now regression tests; a kernel change that
  degrades accuracy beyond them fails `tests/test_profiles_and_reference.py`.
- The normal CDF/survival error grows like `x^2` ULPs in the tails. This is
  the conditioning of `Phi` (relative condition number about `x^2`) applied
  to the rounding of the argument `(x - mu)/(sigma sqrt 2)`, not an error of
  `math.erfc`; removing it would need the argument in extended precision.
  It is recorded, not changed.
- The incomplete gamma and beta kernels lose accuracy roughly in proportion
  to the size of the `exp(a log x - x - log Gamma(a))` prefactor exponent
  for large shapes (hundreds to about 1300 ULPs, relative error up to
  about `3e-13`); this is recorded, not changed.
- The reference functions are slow by design (pure-Python decimal
  arithmetic, milliseconds per value at 40 digits) and are meant for tests
  and audits, not for inner loops.
- `docs/api-reference.md` must be regenerated for the new public names.

## Verification

`tests/test_profiles_and_reference.py` (25 tests, about 8 s):

- Profiles on a hand-made table: exact ratios with ties and failures, exact
  `rho` values at and between breakpoints, `rho(1)` equal to an
  independently counted win fraction, `tau_at`, `solved_fraction`, the area
  equal to its closed form (0.5, 0.875, 0, and 0.625 at `tau_max = 16`), and
  `area(.., 1) = rho(1)`; data profiles with `n_p + 1` divisors, exact
  values, and monotonicity over `kappa = 2^-2 .. 2^11`; the Moré-Wild test;
  rejection of empty, ragged, non-positive, and non-real tables.
- The benchmark section on synthetic rows (inaccurate runs become
  failures; zero-cost methods are listed, not profiled; seven-column limit)
  and the figure in both palettes with the Agg backend.
- References against constants and exact identities: `pi` to 100 digits,
  `e` to 50, `ln 2` to 40; `erf(1)` to 100 digits against OEIS A099286;
  `erfc(x) = Q(1/2, x^2)` to 43 digits for `x` up to 27 (two independent
  algorithms); `Phi(-10)` against `Q(1/2, 50)/2` to 38 digits and against
  `math.erfc` to 13; normal-quantile round trips from `p = 10^-300` to
  `1 - 2^-50` to 40 digits; `log Gamma(1/2) = log(pi)/2`,
  `log Gamma(100.5) = log(200!/(4^100 100!)) + log(pi)/2`,
  `log Gamma(50) = log 49!` to 60 digits; `log Gamma(1 + 2^-40)` against
  `-gamma e + (pi^2/12) e^2` (curvature within `1e-10`); `Q(n, x)` for
  integer `n` against the Poisson sum to 39 digits including `Q(3, 100)`;
  series-complement and continued-fraction branches of the incomplete gamma
  agree to 45 digits; `I_x(a, b)` for integer shapes against the exact
  binomial fraction to 39 digits, `I_(1/2)(a, a) = 1/2`, `I_x(a, 1) = x^a`.
- Invalid inputs (NaN, infinities, strings, booleans, out-of-domain
  arguments, `digits` out of range) raise.

Measured float-kernel errors (40-digit references; Windows 11, Python
3.12.1; the test grids) and the budgets the tests enforce:

| Kernel | Grid | Max error | Budget |
|---|---|---|---|
| `Normal.cdf` | `x` in [-37.5, 8], 93 points | 1228.5 ULP at x = -37.0; max ULP/(1 + x^2) = 1.44; 2.29 ULP for \|x\| <= 2 | 2 (1 + x^2) ULP; 4 ULP for \|x\| <= 2 |
| `Normal.sf` | `x` in [5, 37.5] | 1093 ULP at x = 35; max ULP/(1 + x^2) = 0.91 | 2 (1 + x^2) ULP |
| `Normal.quantile` | 25 central points, `p = 10^-2 .. 10^-296`, `1 - 10^-3, -8, -15` | 2.86 ULP (mean 0.86) | 5 ULP |
| `Normal.isf` | `q = 10^-2 .. 10^-296` | 2.74 ULP | 5 ULP |
| `math.lgamma` | (0, 0.8] and [3, 1e15] | 16.65 ULP at x = 0.8 | 32 ULP |
| `math.lgamma` | [0.8, 3] (zeros at 1, 2) | 6.4e-16 absolute | 2e-15 absolute |
| `log_beta_function` | 20 shape pairs up to 400 | 85.7 ULP at (1.5, 0.7) | 256 ULP |
| `regularized_gamma_p` | `a <= 10` / `10 < a <= 200` | 30.5 / 460.8 ULP | 256 / 4096 ULP |
| `regularized_gamma_q` | `a <= 10` / `10 < a <= 200` | 60.9 / 1330.3 ULP at (200, 605) | 256 / 4096 ULP |
| `regularized_beta` | shapes <= 10 / up to 400, `x` in [0.001, 0.999] | 64.7 / 769.5 ULP at (0.05, 300, 400) | 256 / 2048 ULP |

The benchmark report was regenerated with `python benchmarks/run.py
--quick` (Windows 11, Python 3.12.1, 45.1 s) instead of the previous Linux,
Python 3.13 run. Besides the new sections, timings, and platform line,
results differ in the last bits (the `linear-2d` reference
`0.5 erfc(3/sqrt 2)` itself differs by 2e-19), which moves a few displayed
numbers: the parabolic Multi-point FORM and Mixture IS evaluation counts
(361 to 329, 1,361 to 1,329), the lognormal SORM errors (8e-06 % to
3e-06 %), and the NUTS and static HMC rows (errors, evaluation counts, and
diagnostics; static HMC mean error 0.100 to 0.146 posterior sd). The
figures were
checked in both palettes with an automated figure audit (no errors; the
remaining warnings are a tritan-simulation proximity of two hues, mitigated
by direct labels) and by eye.
