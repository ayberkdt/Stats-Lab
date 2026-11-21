# ADR-0125 — Multi-fidelity surrogates, stochastic kriging, Bayesian quadrature, and Bayesian optimization

- Status: Accepted
- Date: 2026-11-21
- Extends: ADR-0019 (surrogate models), ADR-0054 (GP hyperparameter
  regularization), ADR-0107 (surrogate posterior protocol)

## Context

`surrogates` had Gaussian-process regression for one deterministic code and
universal kriging with a regression trend. Four routine uses of a GP were
missing:

- combining many runs of a cheap code with a few runs of an expensive one;
- emulating a *stochastic* simulator whose noise level changes over the
  design space and is estimated from replications;
- integrating a function with an error indication from very few evaluations;
- choosing the next evaluation of an expensive function in order to optimize
  it. `design` covered optimal designs for regression and expected
  information gain, but no sequential optimization.

All four are exact Gaussian conditioning plus a small amount of closed-form
calculus, so they can be built on the existing kernels, likelihood, and
prediction code.

## Decision

1. **`surrogates/multifidelity.py`: recursive AR(1) co-kriging.**
   `fit_recursive_cokriging` takes nested designs, cheapest first, and
   returns `RecursiveCoKriging`, a tuple of `GaussianProcessRegression`
   levels and the scale factors. Prediction is the recursion
   [EQ-SUR-MF-001]. Each level above the first is fitted by the existing
   `fit_universal_kriging` with the regressors `(1, z_(t-1)(x))`, which
   gives the generalized least-squares `(beta, rho)`, the profiled process
   variance, and the concentrated-likelihood length scales [EQ-SUR-MF-002];
   the result is stored as a GP with known mean through the new
   `gaussian_process_from_hyperparameters`. No GP algebra is duplicated.
   Designs must be nested by exact coordinates; a non-nested design raises.
   The predictive variance is plug-in (simple kriging per level): the
   uncertainty of `rho`, the trend constants, and the hyperparameters is not
   propagated, and the docstrings say so.
2. **Default of ten optimizer starts per level.** With four expensive runs
   the concentrated likelihood of the test pair is bimodal: a short length
   scale with `rho = 0.95` (log-likelihood -9.49) against a long one with
   `rho = 2.00` (-7.27), the right answer. Three starts found the lower mode
   for seed 1; ten found the higher one for seeds 0 to 7.
3. **`surrogates/stochastic_kriging.py`.** `fit_stochastic_kriging` reduces
   replications to sample means and `V_i / n_i`;
   `stochastic_kriging_from_summaries` accepts those directly. The predictor
   and its mean squared error are [EQ-SUR-SK-001] and [EQ-SUR-SK-002]; the
   extrinsic variance and length scales maximize the profile likelihood
   [EQ-SUR-SK-003] by `optimization.multi_start` (BFGS with central
   differences) in a box. The intrinsic variances are plugged in as known,
   as in the source. The library GP has one noise variance, so the
   heteroscedastic factorization lives in this module (through
   `linalg.cholesky`); kernels come from `kernel_value`.
4. **`surrogates/bayesian_quadrature.py`.** `bayesian_quadrature` returns
   the posterior mean, variance, and weights [EQ-SUR-BQ-001] for a
   squared-exponential kernel against a product of `GaussianMeasure` and
   `UniformMeasure` factors, with closed-form kernel integrals
   [EQ-SUR-BQ-002], [EQ-SUR-BQ-003]. Hyperparameters are given, or fitted by
   `fit_gaussian_process` when `length_scales=None`.
5. **`design/bayesian_optimization.py`.** Acquisition functions are plain
   functions of the predictive mean and standard deviation:
   `expected_improvement`, `log_expected_improvement`, `confidence_bound`,
   `gp_ucb_beta`, `probability_of_feasibility`,
   `constrained_expected_improvement`, `knowledge_gradient`, and
   `thompson_sample` on top of `gaussian_process_joint_posterior`. The
   direction is an explicit `maximize` flag everywhere.
6. **Stable `log h`.** For `z <= -1` the improvement function is written as
   `h(z) = phi(t) c(t) / (t + c(t))`, `t = -z`, with `c` the tail of the
   continued fraction of the Mills ratio [EQ-DES-BO-002]. This form has no
   subtraction and needs no scaled complementary error function, which the
   standard library lacks. Ament et al. motivate the logarithm; the
   continued-fraction evaluation is this library's derivation and is
   verified against 50-digit arithmetic.
7. **The loop.** `bayesian_optimization` evaluates a Latin hypercube start,
   then per iteration refits the GP (default
   `GaussianProcessRegularization()`, two starts), scores a fresh Latin
   hypercube candidate set, refines the best three candidates by bounded
   Nelder-Mead, and evaluates the winner. Constraints (`c_k(x) <= 0`) get
   independent GPs and multiply the expected improvement by the probability
   of feasibility [EQ-DES-BO-004]; while no feasible point is known the
   acquisition is the probability of feasibility alone (a library choice).
   A surrogate that cannot be fitted (constant responses) makes the
   iteration evaluate a candidate and is counted in `fallback_iterations`.
   The result is a frozen record of points, values, feasibility, running
   best, acquisition values, and cost counters; its `summary()` says it is
   not a certified optimum.
8. **Sibling import inside layer 5.** `design.bayesian_optimization`
   imports `probstat.surrogates` (both layer 5), exactly as
   `reliability.active_learning` already does. `tests/test_architecture.py`
   has no allow-list: it requires imports to point to the same or a lower
   layer and the module graph to be acyclic. `surrogates` imports nothing
   from `design` (checked by grep and by the acyclicity test), so the layer
   map is unchanged. The alternative, a copy of the GP in `design`, would
   duplicate the kernel and likelihood code.

## Consequences

- Co-kriging oracle: for three levels in two dimensions with fixed
  hyperparameters the recursive mean and variance agree with the joint
  Gaussian conditional built from the explicit block AR(1) covariance to
  `1.5e-13`.
- Two-fidelity pair on `[0, 1]` (eleven cheap runs, four expensive runs):
  estimated `rho = 1.996` (true value 2); high-fidelity root mean squared
  error `0.185` against `5.47` for a GP on the four expensive runs alone.
- Stochastic kriging equals dense GP algebra with a heteroscedastic diagonal
  to `1e-12` and the library GP in the equal-variance case to `1e-11`.
- Bayesian quadrature with thirteen nodes agrees with a 40-point
  Gauss-Hermite rule to `3e-4` on four smooth integrands while reporting a
  posterior standard deviation of `5e-5` for a fixed length scale: the
  variance is conditional on the kernel and is not an error bound.
- `log h` agrees with 50-digit arithmetic to `9e-16` relative on
  `[-100, 8]`; `log h(-40) = -808.2986` where the direct formula returns 0.
- Loop, one dimension, budget 15, seeds 0 to 4: largest regret `1.2e-6`
  against a mean regret `0.82` of random search with the same budget. Two
  dimensions (library Rastrigin benchmark, maximization, budget 25, seeds 0
  to 2): regrets `2.3e-3`, `3.1e-4`, `1.3e-4` against `0.034`, `0.54`,
  `1.05` for random search.
- Cost: a 25-evaluation run in two dimensions takes about one second; every
  iteration is a full GP fit, so the loop is meant for budgets of tens of
  evaluations, not thousands.
- Thompson sampling factorizes an `m x m` covariance per iteration; its
  default candidate set is capped at 100 points.

## Rejected alternatives

- **Alternating a GP fit of the discrepancy with a least-squares update of
  `rho`.** Implemented first; on the test pair it stalled at `rho = 0.93`
  because it follows the short-length-scale mode. The concentrated
  likelihood of universal kriging treats `rho` and the length scales
  together.
- **Universal co-kriging variance with estimated trend and `rho`.** It needs
  the lower-level posterior inside the regression of the next level; the
  plug-in variance is simpler and its limits are documented.
- **Non-nested designs** through imputation of the missing lower-fidelity
  runs by the lower-level posterior mean. The factorization that makes the
  recursion exact no longer holds; refused with an error instead.
- **Adding heteroscedastic noise to `GaussianProcessRegression`.** It would
  change the likelihood, gradient, leave-one-out formulas, and AK-MCS's
  incremental updates for one consumer.
- **Evaluating `log h` through `log(1 - t R(t))` with `erfc`.** `math.erfc`
  underflows near `t = 38` and the subtraction loses `2 log10 t` digits.
- **A grid or pure candidate-set maximization of the acquisition.** Cheap
  but limited to the candidate resolution; the local refinement costs about
  180 extra surrogate predictions per iteration.
- **Knowledge gradient for correlated beliefs or inside the loop.** Only the
  independent-normal closed form of the cited paper is implemented.

## Verification

`tests/test_multifidelity_surrogates.py` (25 tests, about 2 s): the joint
block-covariance oracle, level selection, fitted co-kriging against a
single-fidelity GP, fixed scale factors, an exactly linear level relation,
nesting validation; stochastic kriging against dense algebra, the library
GP, and explicit best-linear-unbiased weights, interpolation of noise-free
points, calibration of the reported mean squared error over eight seeded
replicates; kernel integrals against `numerics.integrate`, exact integration
of kernel sections, monotone variance, Gauss-Hermite agreement, a
two-dimensional product measure, and validation.

`tests/test_bayesian_optimization.py` (25 tests, about 8 s): `h` and `log h`
against the reference backend, the underflow case, expected improvement
against quadrature in both directions, the GP-UCB schedule, constrained
expected improvement limits, the knowledge gradient against Monte Carlo
(within four standard errors) and its variance identity by simulation of the
conjugate update, the joint posterior against marginal predictions, Thompson
draw moments, the loop on a one-dimensional function and on the library
Rastrigin benchmark against random search, a constrained problem, an
infeasible problem, bookkeeping, the fallback path, and validation.

## References

[REF-KENNEDY-OHAGAN-2000], [REF-LE-GRATIET-GARNIER-2014],
[REF-ANKENMAN-NELSON-STAUM-2010], [REF-OHAGAN-1991],
[REF-JONES-SCHONLAU-WELCH-1998], [REF-AMENT-ETAL-2023],
[REF-SRINIVAS-ETAL-2010], [REF-GARDNER-ETAL-2014],
[REF-FRAZIER-POWELL-DAYANIK-2008], [REF-RASMUSSEN-WILLIAMS-2006],
[REF-NIST-DLMF].
