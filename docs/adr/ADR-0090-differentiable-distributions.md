# ADR-0090: Differentiable distribution families, and the split of the composite-law module

- Status: Accepted
- Date: 2026-11-12

## Context

The probabilistic-programming layer (`probstat.inference.ppl`) takes the
gradient of a program by reverse-mode automatic differentiation when the
program is traceable (ADR-0069) and by central differences otherwise. Its
traceability stopped at the library's own laws: the constructors of
`probstat.distributions` validated their parameters with float-only
validators, so `Normal(mu, 1.0)` with a latent `mu` raised
`AutodiffUnsupportedError` ("mu must be a real number") and the run fell
back to finite differences. ADR-0065 (Decision 5) recorded this as future
work in `probstat.distributions`, and ADR-0082 recorded the same for
`multivariate.Multinomial` with a latent probability vector. A latent site
with a library prior such as `Gamma(2, 1)` fell back as well, because
`Gamma.log_pdf` validated its point with `float()`. A user who writes a
model with the library laws reasonably expects the gradient-based samplers
to be exact; instead each gradient cost `2d + 1` model executions and
carried an `O(h^2)` truncation error.

ADR-0073 (follow-up 1) also found that `distributions/wrappers.py` grouped
three unrelated constructions, `Affine`, `Mixture`, and `Truncated`, under
a generic name.

## Decision

1. **Variables pass validation.** A private module
   `distributions._autodiff` holds the helpers. `is_variable` recognizes a
   `numerics.Dual` or `numerics.ReverseVariable`, `primal` returns its
   innermost float through the existing `numerics.autodiff._primal`, and
   `real_argument` validates the primal value of a variable and returns the
   variable itself (floats go through `require_finite_real` exactly as
   before). `RealIntervalConstraint.validate_parameter` and
   `ProbabilitySimplexConstraint.validate_parameter` do the same for
   parameters; the simplex version normalizes variable entries by their
   differentiable sum. Integer constraints are unchanged. The families call
   `validate_parameter` in `__post_init__`; the float-only `validate` keeps
   its behaviour for every other caller, including the special functions.
2. **Log-densities with `admath`.** `log_pdf` (continuous families, whose
   point may also be a variable) and `log_pmf` (discrete families, integer
   points) are written with the `numerics.admath` functions, through a
   float-typed view `distributions._autodiff.admath` so the float-annotated
   code type-checks. For a float argument every `admath` function is the
   `math` function itself, so the operations, their order and the results
   are those of the float-only code. The families covered are Bernoulli,
   Beta, Binomial, Categorical, Cauchy, ChiSquare, Exponential, FisherF,
   Gamma, GeneralizedExtremeValue, GeneralizedPareto, Geometric, Gumbel,
   Laplace, Logistic, LogNormal, NegativeBinomial (probability), Normal,
   Pareto, PERT, Poisson, Rayleigh, StudentT, Triangular, TruncatedNormal,
   Uniform and Weibull, together with `multivariate.Multinomial`
   (probabilities, through `multivariate._support.probability_vector`) and
   `multivariate.MultivariateNormal` (mean and point; the covariance stays
   a float matrix and its Cholesky factor is applied by forward
   substitution written with operators).
3. **Accurate values, closed-form derivatives.** Where a family evaluates
   its log-density with a float-only kernel (the Stirling and
   cancellation-free forms of Gamma, Beta, FisherF and StudentT above
   `LARGE_SHAPE = 100`, and the interval mass of TruncatedNormal), the
   variable result takes its value from that kernel at the primal values
   and its derivatives from the closed form:
   `l~ = v + (g - g_0)` ([EQ-DIST-AD-001]). The value is therefore the float
   log-density bit for bit, and the derivatives, which do not suffer the
   cancellation of the value, come from `lgamma`, `digamma` and `erfc`
   ([REF-GRIEWANK-WALTHER-2008, Chapter 3]). At `xi = 0` the GEV and GPD
   reduced variates use `z (1 - w/2 + w^2/3)` for variables so that the
   derivatives with respect to `xi` survive the branch that returns `z`.
4. **Which methods require floats.** Only the log-densities are
   differentiable by contract. `cdf`, `sf`, `log_cdf`, `log_sf`,
   `quantile`, `isf`, `hazard`, `entropy`, `support` and the moments are not
   edited; with a variable they raise a `TypeError` (usually
   `AutodiffUnsupportedError`) from `math` or a float validator, or they
   return a variable built with operators, never a value whose derivative
   was silently dropped. `support` builds a `RealSupport`, which validates
   its bounds as floats, so a latent site whose support bound depends on
   another latent value (`Uniform(0, t)` with a latent `t`) still raises
   and falls back to finite differences. Integer parameters, `Empirical`,
   `DiscreteUniform`, `Hypergeometric`, the composite laws, and the
   multivariate laws other than the two above (`Dirichlet`,
   `DirichletMultinomial`, `MultivariateStudentT`, `LKJ*`, `Wishart`,
   `InverseWishart`) take float parameters. The PPL evaluates Dirichlet,
   LKJ, Wishart and inverse-Wishart *priors* with constant parameters from
   the transform's by-products itself (ADR-0082), so those remain
   traceable.
5. **PPL.** The inlined normal density of `ppl._program._log_density` uses
   `admath.log(sigma)`, so a latent scale is differentiated. With the
   families above a program written with the library laws is traceable,
   and `sample`, `laplace` and `fit_advi` use reverse mode (forward over
   reverse for the Laplace Hessian) with `gradient_method == "reverse"` and
   no fallback reason. The fallback and its recorded reason remain for
   programs that are not traceable.
6. **A defect found on the way.** Running the Laplace approximation of the
   Dirichlet-multinomial model in reverse mode exposed that the
   stick-breaking softplus, written as `max(t, 0) + log1p(exp(-|t|))`, has
   derivative 0 instead of 1/2 at `t = 0` under automatic differentiation
   (the derivative of `|t|` at its kink is defined as 0). The last stick
   has offset `log 1 = 0`, so at the PPL start `u = 0` the gradient of
   every Dirichlet site lost a coordinate, and BFGS stopped at a point with
   a singular Hessian. `_softplus` in `ppl._structured` (and, for
   uniformity, in `ppl._program`, whose interval Jacobian was correct by
   symmetry) now uses the smooth branches `t + log1p(exp(-t))` for `t > 0`
   and `log1p(exp(t))` otherwise; float values are unchanged.
7. **Module split** (ADR-0073 follow-up 1). `distributions/wrappers.py` is
   replaced by `affine.py` (`Affine`), `mixture.py` (`Mixture` and its
   bracketing helper), `truncated.py` (`Truncated`), and the private
   `_wrapped.py` with the helpers the three share (structural interface
   check, `sf`/`isf`/`log_pdf` dispatch, sampling of a base law). Every
   definition moved unchanged; `probstat.distributions.__all__` is
   unchanged. As in ADR-0073, the old module path is removed without a
   shim: it was absent from every `__all__` and from the API reference,
   and only `distributions/__init__.py` imported it. The equation registry
   needed no change, because its `- Code:` entries name the public paths
   `distributions.Affine`, `distributions.Mixture`, `distributions.Truncated`
   and `distributions.Mixture._central_moment`, which still resolve.

## Consequences

- Float results are unchanged (Verification), so existing seeded results
  of code that does not differentiate through a library law are unchanged.
- Programs written with library laws now take exact reverse-mode
  gradients: one model execution per gradient instead of `2d + 1`, no
  truncation error. Seeded NUTS, HMC, ADVI and Laplace results of such
  programs change because the gradient changes (the purpose of the
  change), as do seeded runs on simplex sites, whose start gradient is now
  correct. The Dirichlet-multinomial run of the foundations guide (seed 3)
  moves from 0.5766, 0.2579, 0.1655 to 0.5772, 0.2571, 0.1658.
- For one- and two-dimensional programs the wall-clock time is similar
  (taping costs about what the extra `2d` float executions cost); the
  saving grows with the dimension (ADR-0069, Verification).
- `Normal(Dual(...), 1.0)` and the like no longer raise at construction;
  code that relied on that error to detect variables must test the
  arguments itself. Tests that encoded the old fallback,
  `test_reverse_autodiff.test_library_distributions_fall_back` and
  `test_mcmc_diagnostics_autodiff.test_standard_distributions_fall_back`,
  now assert reverse and forward mode with the closed-form gradient; the
  fallback path is kept under test by a latent support bound
  (`test_latent_support_bound_falls_back`).
- Each composite law reports its own module
  (`probstat.distributions.affine`, `.mixture`, `.truncated`); pickles of
  those three classes written by earlier versions reference the removed
  module and no longer load.
- Remaining limits: latent support bounds, latent Dirichlet concentrations
  and latent covariance matrices of `MultivariateNormal` fall back to
  finite differences; in the far tails of a truncated normal (both bounds
  beyond about 37 standard deviations on the same side) the closed-form
  derivative of the interval mass underflows.

## Verification

- **Float behaviour is bit-identical.** A comparison script drew seeded
  random parameters for every family exported by `probstat.distributions`
  (25 laws each, including shapes above `LARGE_SHAPE`, `xi = 0` and
  endpoint cases), the three composite laws, and Multinomial,
  DirichletMultinomial, Dirichlet, MultivariateNormal,
  MultivariateStudentT, Wishart, InverseWishart and LKJCorrelation, and
  recorded `float.hex` of every `log_pdf`, `log_pmf`, `pdf`, `pmf`, `cdf`,
  `sf`, `log_cdf`, `log_sf`, `hazard`, `quantile`, moment, entropy and
  seeded `sample` result (or the exception type and message), plus PPL
  log-densities and central-difference gradients of three programs:
  160,774 values. The output files before and after the change, and after
  the module split, are byte-identical.
- **Gradients.** `tests/test_differentiable_distributions.py` checks 36
  cases covering every differentiable family (with large-shape cases for
  Beta, FisherF, Gamma and StudentT and the three truncation forms of
  TruncatedNormal): reverse mode equals forward mode within `1e-12`
  (relative) and central differences within `1e-7` (relative; observed
  `<= 6.3e-10`). The value carried by a variable equals the float
  log-density exactly. Closed forms: the normal scores `z/sigma` and
  `(z^2 - 1)/sigma`, the gamma scores `log(lambda x) - psi(k)` and
  `k/lambda - x`, the beta scores `log x - psi(a) + psi(a + b)`, the
  Poisson, binomial and multinomial scores, all within `1e-12`; the
  forward-over-reverse second derivative of the gamma log-density in `k`
  equals `-psi'(k)` for `k = 2.5` and `k = 250` within `1e-12` relative.
  Validation reads the primal value (`Normal(0, Dual(-1))` raises
  `ValueError`), and `cdf`/`sf` with a variable raise a `TypeError`.
- **Programs.** Six programs written with `distributions.Normal`, `Gamma`,
  `Beta`, `StudentT`, `Poisson`, `Binomial` and `multivariate.Multinomial`
  with a `Dirichlet` prior trace with `gradient_method == "reverse"` and no
  fallback reason, and their gradients equal central differences within
  `1e-7`; the Poisson-gamma gradient on `log lambda` equals
  `(3 + sum k) - 5 lambda` within `1e-12`. Two chains of 300 + 400 draws
  (seed 11; means with MCSE against the conjugate answer):

  | Program | Engine | Posterior mean (MCSE) | Exact |
  |---|---|---|---|
  | Normal mean, N(0, 10^2) prior | NUTS | 1.2567 (0.0222) | 1.2575 |
  | Normal precision, Gamma(2, 1) prior | HMC | 2.2326 (0.0339) | 2.1898 |
  | Poisson rate, Gamma(3, 1) prior | NUTS | 3.3421 (0.0469) | 3.4000 |
  | Binomial probability, Beta(2, 3) prior | HMC | 0.4238 (0.0036) | 0.4286 |
  | Multinomial, Dirichlet(2, 1.5, 1) prior, `p[0]` | NUTS | 0.5785 (0.0039) | 0.5714 |
  | `p[1]` | | 0.2591 (0.0036) | 0.2653 |
  | `p[2]` | | 0.1624 (0.0036) | 0.1633 |
  | Student-t(4) location, N(0, 10^2) prior | NUTS | 1.2779 (0.0241) | 1.2572 |

  The Student-t reference is a trapezoidal quadrature of the posterior on
  20,001 points. The test requires every mean within 4 MCSE; it also passed
  with seeds 1, 2, 3 and 5. Laplace on the normal-mean program records
  reverse gradients and Hessians and gives the exact posterior variance
  `1/(0.01 + 5)` within `1e-12`; on the Poisson-gamma program its variance
  of `log lambda` equals `1/(3 + sum k)` within `1e-6`.
- **Cost.** The Dirichlet-multinomial run of the foundations guide (seed 3,
  two chains of 300 + 400 draws) needed 5,616 model executions in reverse
  mode (one per gradient) against 5,551 gradients, 27,755 executions, with
  central differences; it took 3.6 s against 5.6 s on the development
  machine.
- **Softplus defect.** `test_simplex_gradient_is_exact_at_the_origin`
  failed before the fix (reverse gradient 0.0 against 0.25 by central
  differences for a prior-only `Dirichlet(2, 1.5, 1)` site at `u = 0`) and
  passes after it; the existing Laplace check of the Dirichlet-multinomial
  test passes again.
- **Split.** An AST comparison of every top-level definition of the old
  `wrappers.py` with the definitions in `affine.py`, `mixture.py`,
  `truncated.py` and `_wrapped.py` finds the same twelve names and no
  difference; `probstat.distributions.__all__` is unchanged.
- The documentation contract, architecture, distribution, multivariate,
  automatic-differentiation, Hamiltonian-sampler and PPL test modules pass;
  `ruff` and strict `mypy` report no issue.
