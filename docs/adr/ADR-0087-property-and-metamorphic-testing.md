# ADR-0087: Property-based and metamorphic testing of the public catalogue

- Status: Accepted
- Date: 2026-11-11

## Context

The distribution families were tested family by family with hand-picked
parameter values, and the boundary, property and metamorphic checks that
existed were scattered over individual test modules. Nothing guaranteed that
a new family received the same checks, that identities such as `S = 1 - F`
held away from the chosen values, or that estimators respected the
invariances that theory gives them. The roadmap item "Boundary/property/
metamorphic testing" was therefore only partly met.

## Decision

1. **Harness** (`tests/_properties.py`, test-side, standard library only).
   Each family has a `FamilySpec` with a seeded generator of valid parameters
   that encodes its constructor constraints over a well-conditioned operating
   range, an optional generator of extreme parameters, the parameter kinds
   used to test rejection of invalid values, density kinks (quadrature
   breakpoints), a sample size and, where the survival function is a
   difference of tails, its cancellation factor. Every case seed is a CRC-32
   of `(PROPERTY_SEED, property, family, index)`, so runs are deterministic
   and a reported failure (property, family, case index, case seed,
   parameters, message) is reproduced by `reproduce`. There is no shrinking.
   Cheap properties run 50 cases, quadrature and sampling properties 10;
   `PROBSTAT_PROPERTY_CASES` raises the counts and `PROBSTAT_PROPERTY_SEED`
   selects a different case set for longer local runs.
2. **Catalogue properties** (`tests/test_distribution_properties.py`). The
   test discovers every law exported by `probstat.distributions` and fails
   when one has no generator. For random valid parameters it checks: `F`
   non-decreasing in `[0, 1]` with the limits and `F = 0`/`S = 0` at closed
   support ends; `F + S = 1`; `Q(F(x)) = x` and `isf(S(x)) = x` with the
   tolerance set by the forward error `delta F / f`; `f >= 0` and
   `log_pdf = log pdf`; `int_a^b f = F(b) - F(a)` on pieces and unit total
   mass by `probstat.numerics.integrate`; `S(x)` against the quadrature of
   the tail at `isf(1e-6)` and `isf(1e-10)`; the entropy against
   `-int f log f`; sample means and variances within six standard errors;
   `Q(0)`, `Q(1)`, `isf(0)`, `isf(1)` at the support ends and rejection of
   invalid arguments and parameters; consistency at extreme parameters. For
   integer laws: step-function `F`, `S` against the summed tail, the
   generalized inverse `Q(p) = min{k : F(k) >= p}` and its survival analogue,
   masses that sum to one and accumulate to `F`, the entropy as
   `-sum p log p`, and sampling moments. Multivariate laws are checked
   against their univariate reductions (normal marginals, `Dirichlet(a, b)`
   against `Beta(a, b)`, two-category multinomial against binomial), unit
   total mass of the multinomial and Dirichlet-multinomial, and sampling
   means of the normal, Student-t and Wishart laws.
3. **Metamorphic relations** (`tests/test_metamorphic_relations.py`), each a
   theorem for the estimator concerned [REF-CHEN-ETAL-2018]: affine
   equivariance of sample quantiles (types 1 and 7, reflection for type 7),
   of location-scale and scale-family maximum-likelihood fits and of family
   quantiles; permutation invariance of descriptive statistics, correlations
   under joint permutation, and of KS, Anderson-Darling, two-sample KS and
   Mann-Whitney tests; invariance of the KS statistic under a joint
   increasing map of data and law; sufficiency of the Poisson-gamma,
   Bernoulli-beta and normal-inverse-gamma posteriors (raw-data posterior
   against the conjugate form; data sets with the same sufficient statistic,
   including a Householder reflection that fixes the sum and the sum of
   squares); nesting of confidence intervals in the level and narrowing with
   replicated data; Sobol indices invariant under bijective input
   reparameterization (model level), positive output scaling, monotone input
   warps and input permutation (estimator level); FORM index and design
   point invariant to positive scaling of the limit-state function, crude
   Monte Carlo failure counts identical under it; BDD system reliability
   equal for random variable orders, after sifting, and to full state
   enumeration; and unbiasedness over seeds of plain, antithetic, stratified
   and importance-sampling Monte Carlo means with calibrated standard errors.
4. **Tolerances** follow forward-error analysis [REF-HIGHAM-2002]. Where a
   tolerance is wider than a few rounding errors the test states why: the
   condition number of `exp` at a log-prefactor (special functions, summed
   discrete masses), the cancellation of a truncated survival function, or
   the mass that floating-point nodes cannot resolve next to a singular
   support end away from zero.
5. **Every failure was investigated.** Library defects were fixed with a
   regression test in `tests/test_property_regressions.py`; test-side
   misconceptions were corrected in the test, not by widening a tolerance.

## Defects found and fixed

| # | Defect | Fix |
|---|--------|-----|
| 1 | `GeneralizedExtremeValue` and `GeneralizedPareto`: at the reported closed support end `mu - sigma/xi`, `1 + xi z` rounded to a tiny positive number, so `S(end)` was up to `1e-90` instead of `0` (a third of random GPD laws with `xi < 0`). | The reduced variate compares `x` with the reported endpoint. |
| 2 | `pdf` raised `OverflowError` where the density exceeds the float range (shape below one near zero for `Gamma`, `FisherF`; scales near the smallest normal float for every location-scale family) while `log_pdf` was finite. | `density_from_log` saturates to `inf` ([EQ-DIST-BASE-005]); used by 17 families. |
| 3 | `Binomial.cdf`/`sf` summed `O(n)` masses per call; `quantile` at `n ~ 10^7` took minutes. | Regularized incomplete beta form above `n = 256` ([EQ-DIST-BIN-002] updated). |
| 4 | `Binomial.quantile(1)` returned the first count where the rounded CDF reaches one (7 for `n = 185`, `p = 9.5e-5`) instead of `n`. | `Q(1)` is the largest point with positive mass. |
| 5 | `Binomial.sf` could exceed one by rounding (`S(0) = 1 + 2.4e-14`), which made `isf(1)` return 5 instead of 0. | Clamped to `[0, 1]` like the CDF. |
| 6 | `Hypergeometric.quantile(1)` returned 24 for support `{4, ..., 29}`. | Returns the upper support end. |
| 7 | `Hypergeometric.cdf`/`sf` summed every mass from the support end; `N = 10^5` to `10^6` took seconds per call. | Log-concave tail sums stop once the remainder is below `1e-20` of the total. |
| 8 | Discrete `isf(0)` returned the nominal support end for degenerate laws (`Poisson(0)`, `Geometric(1)`, `NegativeBinomial(r, 1)`, `Bernoulli(0)`, `Binomial(n, 0)`), disagreeing with `quantile(1) = 0`. | `isf(0)` is `quantile(1)` ([EQ-DIST-BASE-003]). |
| 9 | `Uniform.isf(1)` and `quantile(1)` were `upper - (upper - lower)` and `lower + width`, which need not round to the ends. | Exact ends, clamped interior values. |
| 10 | `Affine` with a negative scale gave `F(lower) = 3.3e-26` at its own support end. | Support ends cached in `x`; `cdf`/`sf` compare with them. |
| 11 | `Truncated.isf(1)` inverted the base survival function at `S(lower)` and returned `49.66686163951595` for `lower = 49.66686163951593`. | `isf(1)` is `quantile(0)`, the lower end. |
| 12 | `Empirical.entropy` (interpolated law) returned `nan` from the probability-space quadrature for samples with very unequal gaps. | Closed form ([EQ-DIST-EMPIRICAL-003]); ties give `-inf`, the step law `nan`. |
| 13 | `Beta.entropy` used the generic quadrature: `nan` for a shape below one with a large partner, and about 1 s per call. | Closed form shared with `information.beta_entropy` ([EQ-INFO-DH-003]). |
| 14 | `Mixture.skewness`/`kurtosis` integrated over the mixture quantile (a root solve per node, up to 10 s), and `Mixture.entropy` returned `nan` when a component density is singular inside the mixture support. | Moments from component moments; entropy by `x`-space quadrature split at component ends and medians ([EQ-DIST-MIXTURE-002]). |

No defect was found by the metamorphic relations: fitting, statistics,
interval, Sobol, FORM, BDD and Monte Carlo code satisfied every relation on
the first run.

## Consequences

- A new exported law fails `test_every_exported_law_has_a_generator` until it
  receives a `FamilySpec`, and then inherits all catalogue properties.
- `Binomial` probabilities for `n > 256` change in the last digits (relative
  differences below `1e-12` against exact rational sums); `Mixture`,
  `Beta` and `Empirical` entropies and mixture moments become cheaper and
  finite where they were `nan`.
- The added tests run in about 65 s (properties about 56 s, metamorphic
  relations about 6 s, regressions about 3 s) on the reference machine.
- Not covered: shrinking of failing cases; the probabilistic-programming
  layer; laws without a common interface (`LKJ` laws are checked in their own
  module).

## Verification

- `tests/test_distribution_properties.py`: 33 univariate families (25
  continuous, 7 integer-valued, 1 categorical) and 6 multivariate laws;
  11 continuous and 8 discrete properties plus the categorical and invalid
  parameter checks give 347 (property, family) pairs and about 10,900
  generated cases per run; all pass for `PROPERTY_SEED = 20261111` and for
  the alternative seeds 7, 99 and 12345.
- `tests/test_metamorphic_relations.py`: 20 relations (about 990 subtests),
  all passing.
- `tests/test_property_regressions.py`: 21 regression tests, one or more per
  defect above.
- The alternative seeds found defect 11 and three test-side errors that were
  corrected in the tests: the variance check is limited to excess kurtosis at
  most 10 (a Pareto law of shape 4.7 exceeded six standard errors), the
  standard error of `s^2` uses the exact finite-`n` term (a two-point law has
  `kappa + 2 = 0`), and tail quadratures use a purely relative target and
  split at density kinks.
- Full suite, `ruff check` and `mypy src/probstat` pass.
