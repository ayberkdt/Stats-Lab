# ADR-0082: Structured and discrete latent sites in the probabilistic-programming layer

- Status: Accepted
- Date: 2026-11-11

## Context

`probstat.inference.ppl` (ADR-0053, ADR-0069) accepted only scalar and flat
vector latent sites of continuous laws on intervals. The roadmap (Phase 6)
and the evidence audit of ADR-0079 list what remained: structured latent
supports (simplex, correlation and covariance matrices), discrete latent
sites, and dynamic site structure. Without the first two, a mixture model,
a multinomial model with a Dirichlet prior, or a model with an unknown
covariance matrix had to be marginalized and transformed by hand, which is
what the interface exists to avoid. `probstat.multivariate` already provides
the Dirichlet, Wishart, and inverse-Wishart laws (ADR-0067) but no law on
correlation matrices.

## Decision

1. **Structured supports.** A new `ppl.StructuredSupport(kind, dimension,
   lower=None)` holds the transforms of the Stan Reference Manual 2.36,
   chapter "Constraint Transforms" ([REF-STAN-REFERENCE-MANUAL-2-36]):
   stick-breaking onto the simplex ([EQ-PPL-TRANSFORM-003]), cumulative
   exponentials onto ordered vectors ([EQ-PPL-TRANSFORM-004]), tanh of
   canonical partial correlations onto Cholesky factors of correlation
   matrices and, through `L L'`, onto correlation matrices
   ([EQ-PPL-TRANSFORM-005]), and a Cholesky factor with a log diagonal onto
   covariance matrices ([EQ-PPL-TRANSFORM-006]). Version 2.36 is cited
   because version 2.37 replaced the stick-breaking simplex map by a softmax
   of a sum-to-zero vector. Each transform returns the value, the log
   absolute Jacobian onto the free coordinates on which the site's density
   is defined, and the logarithms it computed on the way (simplex entries,
   Cholesky diagonal). Every operation uses operators and `admath`, so the
   maps are transparent to forward and reverse mode.
2. **Laws on structured supports.** `m.sample` recognizes
   `multivariate.Dirichlet` (simplex), `multivariate.LKJCholesky` and
   `multivariate.LKJCorrelation` (new), `multivariate.Wishart` and
   `InverseWishart` (covariance), and `ppl.OrderedVector(base, n)` (new; the
   law of the sorted vector of `n` i.i.d. draws, density `n! prod f(x_i)`).
   A user law opts in by setting `support` to a `StructuredSupport`. The
   densities of the library laws are evaluated inside the PPL from the
   by-products of the transform with `admath`, so priors with constant
   parameters are traceable; other laws are evaluated by their own
   `log_pdf`. `m.observe` with a law that has a `dimension` attribute adds one
   joint term for a vector or matrix observation.
3. **The LKJ law.** `multivariate.LKJCorrelation(d, eta)` and
   `multivariate.LKJCholesky(d, eta)` implement the density
   `det(Omega)^(eta - 1) / c_d(eta)` and its Cholesky form
   `prod_k L_kk^(d - k + 2 eta - 2) / c_d(eta)` ([EQ-DIST-LKJ-001],
   [EQ-DIST-LKJ-002]; [REF-LEWANDOWSKI-KUROWICKA-JOE-2009],
   [REF-STAN-FUNCTIONS-REFERENCE-2-36]), with
   `multivariate.lkj_log_normalizer` for `log c_d`, and draw by independent
   Beta partial correlations ([EQ-DIST-LKJ-003]). They are laws, so they live
   in `multivariate` beside the Wishart family.
4. **Discrete sites.** `m.sample(name, law, marginalize=True)` declares a
   scalar discrete site with a finite support. It has no unconstrained
   coordinate; the density handed to the engines is the marginal over the
   discrete sites, computed by enumeration with a log-sum-exp
   ([EQ-PPL-ENUM-001]; [REF-STAN-USERS-GUIDE-2-36]). Sites outside plates
   are enumerated jointly by re-running the model per configuration.
   `m.plate(name, n)` returns an iterator over `0..n-1`; in density mode an
   iteration that declares discrete sites is re-run per configuration of its
   own sites, after the handle is rewound to the iteration start, and adds
   the log-sum-exp of the terms that follow its first discrete site. This is
   exact under the conditional independence a plate declares
   ([REF-OBERMEYER-ETAL-2019]) and costs `sum_i |Z_i|` body executions
   instead of `prod_i |Z_i|`. Unbounded supports are refused with a message
   pointing to `ppl.TruncatedDiscrete(base, lower, upper)`; more than 4096
   configurations in one scope are refused. Opting in with `marginalize`
   keeps the earlier behavior for unmarked discrete sites (a specification
   error) and makes the enumeration cost explicit.
5. **Order rule.** In each scope (the whole model for global sites, one
   plate iteration for local ones), discrete sites follow every continuous
   latent and deterministic site; only observed and discrete sites may come
   after them. The unconstrained coordinates, the bijections, and the
   reported values are then identical in every configuration, and the
   enumeration needs only the terms after the first discrete site, which are
   accumulated in a separate sink rather than recovered by subtraction.
6. **Recovery.** After any engine, `ModelIR.discrete_conditional(u)` re-runs
   the enumeration for each continuous draw and records the unnormalized log
   weights; `PosteriorSamples.discrete` (a `ppl.DiscreteDraws`) holds one
   joint draw per continuous draw (global configuration, then each plate
   iteration, from the child stream `"ppl-discrete"`) and the exact
   conditional probabilities ([EQ-PPL-ENUM-002]).
   `posterior.discrete_probabilities(name)` averages them (a
   Rao-Blackwellized estimate) and `posterior.mean(name)` uses them;
   `posterior.draws()` and `posterior_predictive` include the drawn values.
7. **Engines unchanged.** NUTS, HMC, adaptive Metropolis, ADVI, and Laplace
   receive the marginal density on `R^d` as before; results report
   constrained values (vectors and matrices) by site name.
8. **Layering.** `inference.ppl` imports `multivariate` (both layer 2).
   `multivariate` imports nothing from `inference` (ADR-0067), and
   `inference.bayesian` and `inference.estimation` already import
   `multivariate`, so the graph stays acyclic and no code moves to a lower
   layer. Inside the subpackage, `ModelSpecificationError` moved to a private
   `_errors` module so that the new `_structured` and `_discrete` modules
   and `_program` do not import each other in a cycle; the public name is
   unchanged.
9. **Not done.** Dynamic site structure (stochastic control flow that adds
   or removes sites) remains unsupported, as do nested plates, vector-valued
   discrete sites, and variable elimination across plate iterations (hidden
   Markov chains).

## Consequences

- Mixtures, latent class models, change points, and models with simplex,
  correlation, or covariance parameters are written directly; a traceable
  mixture (Dirichlet weights, ordered means, `admath` likelihood) is
  differentiated in reverse mode through the enumeration.
- Each density evaluation costs one model execution per global
  configuration, times the per-iteration branch count inside plates. The
  30-observation mixture took about 0.9 ms per log-density and 2.1 ms per
  reverse-mode gradient on the development machine; recovery adds one
  recorded evaluation per draw.
- The plate body is executed several times per evaluation, so side effects
  in it are repeated; this is documented.
- `SiteInfo` gained `structure`, `enumeration`, and `plate` (defaulted), the
  site kind `"discrete"`, and the property `free_length`; `PosteriorSamples`
  gained `discrete`. Existing models follow the previous code paths; the
  fixed-seed results of `tests/test_probabilistic_programming.py` and
  `tests/test_mcmc_diagnostics_autodiff.py` are unchanged.
- New public names: `inference.ppl.StructuredSupport`, `OrderedVector`,
  `TruncatedDiscrete`, `DiscreteDraws`; `multivariate.LKJCholesky`,
  `LKJCorrelation`, `lkj_log_normalizer`. `docs/api-reference.md` must be
  regenerated.

## Verification

`tests/test_ppl_structured_sites.py` (20 tests, about 52-60 s on the
development machine):

- **Jacobians.** For the simplex (`K = 4`), ordered vectors (with and
  without a lower bound), Cholesky correlation factors (`d = 3`),
  correlation matrices (`d = 3, 4`), and covariance matrices (`d = 3`), the
  log-Jacobian agrees with the log-determinant of a central-difference
  Jacobian of the free coordinates; the largest difference was
  `8.5e-11`. Inverse and forward maps round-trip to `1e-12`, and images lie
  in the support for random points.
- **LKJ constant.** `c_3(eta)` against nested Gauss-Kronrod quadrature of
  `det^(eta - 1)` over the 3 x 3 correlation matrices: 4.934802200544679
  against 4.93480220054468 (`eta = 1`, equal to `pi^2 / 2`),
  1.8505508252042595 against 1.8505508252042546 (`eta = 2`),
  1.0280837917801355 against 1.0280837917801415 (`eta = 3`). For `d = 2`
  the constant matches quadrature for `eta = 0.5, 1, 2, 3.7` to ten decimal
  places.
- **LKJ(1), 2 x 2.** The density is `1/2` for every correlation; the PPL
  density of the site is `log(1/2) + log(1 - tanh^2 u)` to twelve digits;
  4000 draws have mean and variance of `r` within 4 standard errors of `0`
  and `1/3`; an HMC run (reverse mode) has a posterior mean of `r` within
  4 MCSE of zero.
- **LKJ Cholesky factorization.** The PPL log-density of an
  `LKJCholesky(4, 1.7)` site equals the sum over partial correlations of the
  `2 Beta(b_j, b_j) - 1` log-densities plus `log(1 - tanh^2)` to eleven
  digits; the variance of sampled correlations of LKJ(2), `d = 4`, matches
  `1 / (2 eta + d - 1)`.
- **Dirichlet-multinomial.** Dirichlet(2, 1.5, 1) prior, counts
  (12, 5, 3); NUTS (seed 3, two chains of 300 + 400 draws, finite-difference
  gradient because `Multinomial` validates float parameters): 0.5766,
  0.2579, 0.1655 (MCSE 0.0034, 0.0033, 0.0025) against the conjugate means
  0.5714, 0.2653, 0.1633 from `multivariate.dirichlet_posterior`; Laplace and
  ADVI means within 0.05.
- **Inverse-Wishart covariance.** Twelve bivariate observations,
  IW(Psi, 6) prior, NUTS (seed 4): 1.2458, 0.8585, 0.9645 (MCSE 0.043,
  0.033, 0.031) against the conjugate 1.1695, 0.8114, 0.9209.
- **Gradients.** Reverse mode, forward mode, and central differences agree
  on a model with all six structured laws.
- **Mixture.** The marginalized log-density of a two-component mixture with
  30 observations equals the hand-written mixture density to ten digits.
  At fixed parameters the recovered `P(z_i = 1 | theta, y_i)` equals exact
  Bayes to twelve digits for every observation. NUTS (seed 5, two chains of
  200 + 200 draws, reverse mode, about 17 s) against a 31^3 Simpson grid of
  the exact posterior: `w[1]` 0.5960 (MCSE 0.0067) against 0.6059, `mu[0]`
  -1.9921 (0.0262) against -1.9873, `mu[1]` 1.9493 (0.0116) against 1.9408.
  The Rao-Blackwellized indicator probabilities equal the average of the
  exact Bayes formula over the posterior draws to twelve digits, and the
  sampled indicator frequencies agree within five binomial standard errors.
- **Global enumeration and truncation.** A `TruncatedDiscrete(Poisson(6),
  0, 20)` number of trials with a binomial count of 4 and a Beta(2, 2)
  probability: the log-density matches the explicit sum to eleven digits,
  the conditional weights match exact Bayes, and the Rao-Blackwellized
  posterior mean of `n` from NUTS (seed 6) is 6.438 against the closed form
  6.451 (standard error about 0.073). A global `Categorical` site with
  string labels works with `terms` and `laplace`.
- **Invalid models.** Unbounded discrete supports, `marginalize` on a
  continuous law, latent or deterministic sites after a discrete site
  (globally and inside a plate), nested plates, more than 4096 global
  configurations, vector discrete sites, shapes on structured sites,
  ordered vectors of bounded laws, and unmarked discrete sites raise
  `ModelSpecificationError`.
- The existing suites `tests/test_probabilistic_programming.py`,
  `tests/test_mcmc_diagnostics_autodiff.py`, `tests/test_hmc_nuts_advi.py`,
  and `tests/test_reverse_autodiff.py` pass unchanged.
