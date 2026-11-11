# ADR-0059: Numerical core, distribution contract, sampling designs, surrogate numerics

- Status: Accepted (workstream W2 of the 2026-10-16 core-mechanics programme)
- Date: 2026-10-16

## Context

The 2026-10-16 core-mechanics gap audit
(`memory/sessions/2026-10-16-core-mechanics-gap-audit.md`) found defects in
the numerical layer every analysis depends on: a BFGS stopping rule that
scales with `|f|` (0.7), finite-difference steps floored at `1e-6` with no
fallback at domain boundaries (0.8), no bounds or multi-start (P1.11), an
incomplete beta that loses up to three digits for a large first shape near
`x = 1` (0.13), no common distribution contract (P1.8, 0.14), samplers that
call `random.Random(seed)` directly and accept any seed type, Sobol points
limited to 16 dimensions (0.6), a biased Halton randomization, PCE fitted by
normal equations (0.15), and a GP likelihood that factorizes twice and forms
`K^-1` column by column (P2).

## Decision

1. **BFGS (0.7).** `minimize_bfgs` keeps its signature (new keyword-only
   `step_tolerance`, `bounds`, `typical_scale`). Stopping: absolute
   `||g||_2 <= gradient_tolerance`; relative step
   `max |s_i| / max(|x_i|, typ_i) <= eps^(2/3)` (Dennis and Schnabel);
   everything else is `converged=False` except two floating-point
   resolution limits that are reported in the message: (a) with a
   finite-difference gradient whose rounding error `||(eps |f| / h_i)||`
   exceeds the tolerance, `||g|| <= ` that resolution; (b) a failed line
   search whose predicted quasi-Newton decrease `-g'p/2` is below
   `16 eps |f|`. Neither is a tolerance on `f`; they say that the
   requested accuracy cannot be represented. The line search is strong Wolfe
   (`c1 = 1e-4`, `c2 = 0.9`, Nocedal and Wright Algorithms 3.5-3.6, gradient
   evaluated only where sufficient decrease holds), the identity is scaled by
   `s'y / y'y` before the first update (eq. 6.20), and the first trial step
   is `min(1, 1/||g||)`.
2. **Finite differences (0.8).** Default step `h_i = c max(|x_i|, typ_i)`,
   `c = eps^(1/3)` (central) or `sqrt(eps)` (new `forward_gradient`), rounded
   so `x + h` is representable. `typ_i` defaults to 1 for the standalone
   functions (Dennis and Schnabel's default; a pure `|x|` rule gives rounding
   garbage when a unit-scale parameter passes near zero) and to `|x0_i|`
   (1 where `x0_i = 0`) inside BFGS. So the floor is no longer a hard
   `1e-6`, but a caller-controlled typical scale; parameters on a `1e-6`
   scale need `typical_scale=1e-6` (or a start on that scale in BFGS). An
   explicit `relative_step` keeps the old rule exactly (FORM, HMC, filters
   and the design package pass it), an explicit `step` is used as given. A
   side that raises `ValueError`/`ArithmeticError` or is non-finite falls
   back to the second-order one-sided formula (first order if `x + 2h` also
   fails); both sides failing raises.
3. **Bounds and multi-start (P1.11).** `BoxTransform` implements the MINUIT
   maps (`sin` for two bounds, `sqrt(z^2 + 1)` for one), used by
   `minimize_bfgs(bounds=)` and `minimize_nelder_mead(bounds=)`; objective
   evaluations never leave the box. Start points on a bound move inward by
   `1e-3` of the width because `dx/dz = 0` there. `multi_start` draws
   uniform starts from `resolve_stream(seed).child("multi-start")` and keeps
   the best converged run (`MultiStartResult`).
4. **Incomplete beta (0.13).** For `max(a, b) >= 1e4`, `-log(1 - t) <= 1`
   on the small-shape side and `|s - 1| s (s + 1) <= 24 N^2`, a new
   convergent series in regularized incomplete gamma functions is used
   (EQ-NUM-BETA-002; derived here from the substitution `v = -log(1 - u)`
   and the expansion of `(sinh(v/2)/(v/2))^(s-1)`, the convergent
   counterpart of DiDonato and Morris's BGRAT). The smaller tail is summed
   directly (`P` series, non-negative terms) or by the asymptotic `Q`
   series truncated at its smallest term. The continued fraction is
   unchanged elsewhere. Diagnosis: near the split point the Lentz test
   `|delta - 1| <= 8 eps` is met while ~`sqrt(a)` factors `1 + O(1/a)`
   remain.
5. **Distribution contract (P1.8, 0.14).** New mixins
   `ContinuousDistribution` and `DiscreteDistribution`
   (`distributions/_base.py`), inherited by every family. They wrap the
   family's `cdf/sf/pdf/log_pdf/log_cdf/log_sf` (and `pmf/log_pmf`) once at
   class creation so `+-inf` returns the limit; NaN still raises. Defaults:
   `log_cdf/log_sf` (via `log1p` of the complement above 1/2), `isf`,
   `median`, `std`, `entropy`, `skewness`, excess `kurtosis` (quadrature in
   probability space, `isf` used above the median, `nan` if it does not
   converge; discrete laws sum the mass function). One moment convention:
   divergent integrals are `inf`, undefined ones (`inf - inf`, central
   moments about an undefined mean) `nan`; properties never raise
   (`StudentT(nu <= 1).mean` is now `nan`, GEV overflow returns `+-inf`).
   Discrete `pmf` accepts integral floats (so `pmf(quantile(p))` works);
   non-integral floats remain a `TypeError`. Discrete `isf(q) =
   min{k: S(k) <= q}` searches with `S`. Closed forms were added for
   Student t (skewness, kurtosis, entropy), Triangular, and the Normal log
   tails. `RandomInput("x", StudentT(1))` and every new family work as model
   inputs.
6. **New families (P2).** `Cauchy`, `Laplace`, `Rayleigh`, `Pareto` (type I),
   `PERT` (beta-PERT with `lam = 4` default, `alpha = 1 + lam (m - a)/(b - a)`,
   mean `(a + lam m + b)/(lam + 2)`), `DiscreteUniform`, `Hypergeometric`
   (exact rational masses for `N <= 5000`), and wrappers `Affine` (negative
   scales reflect), `Truncated` (one-sided allowed; upper-tail intervals
   use survival probabilities), `Mixture` (quantile by Brent in
   `[min Q_i(p), max Q_i(p)]`), `Empirical` (`method="linear"`: continuous
   law with the type-7 quantile; `"step"`: ECDF with type-1 quantile and no
   density), plus `KernelDensity.sf/quantile/mean/variance/draw`. All
   satisfy `ContinuousMarginal`. **MLE for the new families is deferred.**
7. **Seeds.** Every `sample` in `distributions` and
   `multivariate.MultivariateNormal.sample` takes `int | RandomStream |
   None` through `resolve_stream`: negative integers raise `ValueError`,
   `bool`/`float`/`str` raise `TypeError`, `None` is OS entropy. An integer
   seed gives the same stream as before (`RandomStream(s).generator()` is
   `random.Random(s)`), so seeded results are unchanged.
8. **Surrogates.** PCE coefficients come from `householder_qr` of the design
   matrix (`c = R^-1 Q' y`, leverages `||q_i||^2`); rank deficiency raises
   with the dependent multi-indices named. The GP likelihood uses one
   factorization per `theta` for value and gradient (cached in the fit, so
   the optimizer's separate value/gradient calls share it), forms the needed
   `K^-1` entries from one triangular inversion of the same factor over the
   upper triangle (an exact trace needs every entry because each `dK/dtheta`
   is dense; "no explicit `K^-1`" is therefore read as "no `n` unit-vector
   solves and no second factorization"), and keeps the Cholesky
   bit-identical to `linalg.cholesky` (`fsum` of the same products) because
   AK-MCS's incremental posterior reproduces the fit's factor bit for bit.
   `adaptive_nugget=True` retries a fit in which no start is positive
   definite with the nugget times 10 (up to `1e-4`), warns
   (`GaussianProcessJitterWarning`) and reports the nugget used. The
   1-norm condition number is computed (`condition_number`) and above
   `1e12` emits `GaussianProcessConditioningWarning`. Among restarts a
   converged run is preferred to a lower non-converged one when the gap is
   below the estimated objective noise `kappa eps (n + |f|)`.
9. **Sampling (0.6, P2).** The Joe-Kuo `new-joe-kuo-6.21201` initial
   direction numbers for dimensions 2-1024 were extracted from SciPy's
   copy and embedded in `sampling/_joe_kuo.py` (48 KB, with the citation and
   the authors' BSD-style licence notice; the notice was reproduced from
   memory of the authors' licence text and should be checked against the
   file at https://web.maths.unsw.edu.au/~fkuo/sobol/ before a release).
   `SOBOL_MAX_DIMENSION = 1024`, so `sobol_analysis` no longer falls back to
   Halton silently for 9-512 inputs. A size that is not a power of two warns
   (`SobolBalanceWarning`). Unscrambled points are centred in their finest
   dyadic cell by default (`centered=False` restores the raw sequence), so
   the origin never reaches an inverse CDF and the net property is kept.
   `scramble="owen"` adds Owen's nested uniform scrambling (lazy random bits
   per `(digit, prefix)`, independent uniform low bits); `True`/`"shift"`
   remains the digital shift, so default study results are unchanged.
   Halton `scramble=True` now permutes every digit position independently,
   zero included (with leading zeros down to `2^-53` and a final jitter),
   which makes each point uniform; Halton supports 1024 dimensions.
   `maximin_latin_hypercube` (Morris-Mitchell `phi_15`, seeded column
   swaps) and `iman_conover` (rank-correlation induction that keeps every
   marginal) were added.

## Evidence

- Incomplete beta: 545 grid points (1040 tail values), `a` in
  `1e2..5e13`, `b` in `0.5..1e3`, `y = 1 - x` from `1e-3` to `30` times the
  mean `b/(a+b)`, against 80-digit mpmath (all-positive hypergeometric
  series; 60-digit Lentz continued fraction for deep lower tails; the
  mpmath quadrature first tried was found inaccurate at 1e-9 and
  discarded). Worst relative error before/after: `a = 5e13` 1.2e-3 / 4.8e-15
  (b = 0.5), `a = 1e10` 9.8e-7 / 2.4e-15, `a = 1e8` 6.3e-9 / 3.7e-15,
  `a = 1e6` 4.9e-11 / 9.3e-15, `b = 1e3` rows 1.6e-5 / 6.3e-13; every point
  now below 1e-12. `StudentT(1e14).sf(3)` = 1.3498980316304233e-3
  (reference 1.3498980316304269e-3; 1.34928e-3 before). 28 Student-t
  (`nu` 30..1e14) and 9 FisherF (`d2` 1e4..1e10) values agree to 1e-12.
  Cost about 3.6 ms per call on the series path.
- BFGS: `(x-1)^2 + 10(y+2)^2 + C` with `C = 1e9` and `-1e12` reaches
  `(1, -2)` to 1e-8 with an analytic gradient (before: start point returned,
  `converged=True`); with central differences it reports the
  finite-difference resolution (about 1e-2 at `C = 1e9`). An inconsistent
  gradient and a ripple objective are reported `converged=False`.
  Accepted steps satisfy both strong Wolfe inequalities.
- Finite differences: `d/dx exp(x/1e-6)` at `2e-6` with
  `typical_scale=1e-6` to 1e-9 (17.5 % error before); `sqrt`/`log` next to
  zero through the one-sided path to 1e-9 (Jacobian).
- Distributions: new families against scipy 1.18 (pdf, logpdf, cdf, sf to
  1e-12; ppf, isf to 1e-9; moments, entropy to 1e-10); numerical default
  skewness/kurtosis/entropy of 14 families against scipy closed forms to
  1e-8/1e-7/1e-9; discrete `isf` at `q = 1e-15` exact for four families.
  scipy's `truncnorm.isf(1e-9)` was found off by 2.6e-9 relative and
  replaced by a 40-digit mpmath bisection as the oracle.
- Sobol: SHA-256 of `round(2^m x)` equals SciPy's unscrambled points for
  `(d, n) = (50, 32), (1024, 32), (50, 1024)`. Owen scrambling keeps the
  `(0, 8, 2)`-net; over 120 seeds, `int exp(x + y)` with 256 points has
  standard deviation 5.2e-4 (Owen) versus 4.8e-3 (digital shift).
- Halton: mean of the first scrambled point over 2000 seeds 0.4956
  (standard error 0.0065; 0.445 with the old zero-fixing permutation).
- PCE: degree-20 Hermite fit of `exp(x/2)` with 60 LHS points succeeds for
  all 40 seeds (33/40 failed with normal equations per the audit); mean and
  variance within 1e-6 and 1e-5 of the lognormal closed forms.
- GP (`d = 2`, Matern 5/2, one value+gradient evaluation, this container):
  n = 100: 0.128 s -> 0.017 s (7.4x); n = 200: 0.738 s -> 0.108 s (6.8x).
  The old fit evaluated twice per BFGS point, so per point 14.8x and 13.7x.
  A full single-start fit at n = 100: 12.0 s -> 2.2 s with the same
  optimum (log p = 304.570414). Values agree with the old implementation
  to 2.5e-12 (n = 100) and 1.0e-11 (n = 200) relative, and with a dense
  explicit-inverse reference to 1e-11 (value) and 1e-7 (gradient).

## Consequences

- Positive: MLE, MAP, Laplace and calibration no longer stop early on large
  negative log-likelihoods; heavy-tailed and data-defined laws are model
  inputs; Sobol designs work up to 1024 dimensions; randomized QMC is
  unbiased; PCE and GP fits are more robust and faster.
- Changed behaviour (tests updated): `cdf(inf)` returns 1 instead of
  raising; `StudentT(nu <= 1).mean/variance` are `nan`; the default
  unscrambled Sobol point set is cell-centred; scrambled Halton points
  differ; `minimize_bfgs` evaluation counts and some unregularized GP
  optima changed (the AK-MCS white-noise regression now escapes to a higher
  likelihood; that test asserts the improvement). `GaussianProcessRegression.
  evaluations` now counts factorizations.
- Limitations: the BFGS resolution rules involve `eps |f|` (unavoidable
  floating-point limits, reported); the GP noise estimate for restart
  choice is a heuristic; the incomplete-beta series covers
  `min(a, b) <~ (24 max(a,b)^2)^(1/3)`, so both-large shapes (for example
  `a = b = 1e6` away from the existing tests) still use the continued
  fraction; numerical moment defaults return `nan` when quadrature fails
  (closed forms exist only where listed); Hypergeometric CDFs are `O(n)`
  sums; `Empirical(method="step")` has no density.
- Deferred: MLE for the new families; `Categorical` CDF/quantile/mean;
  stream seeds in `monte_carlo`, `inference/bayesian.py` and
  `reliability/analysis.py` (outside W2 ownership); LMS (Matoušek) Sobol
  scrambling; switching the study default from digital shift to Owen
  scrambling (changes README numbers, integrator's decision).
