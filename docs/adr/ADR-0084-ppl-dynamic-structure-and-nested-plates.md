# ADR-0084: Dynamic site structure, nested plates, and vector-valued discrete sites in the probabilistic-programming layer

- Status: Accepted
- Date: 2026-11-11

## Context

ADR-0082 added structured and discrete latent sites to
`probstat.inference.ppl` and left three items open, which the roadmap
(Phase 6) lists as the last open PPL work: dynamic site structure, nested
plates, and vector-valued discrete sites. Concretely:

- `trace` ran the model once; every later execution had to declare the
  same sites, and a continuous latent or deterministic site after a
  discrete site was refused. A model whose number of components is chosen
  by a discrete site could therefore not be written.
- `m.plate` inside `m.plate` raised `ModelSpecificationError`.
- `marginalize=True` accepted only scalar sites; a vector of indicators had
  to be written as several sites, and a multinomial count vector could not
  be a latent site.

Structure that depends on observed data (a loop over the groups of the
data, a plate whose size is a data length) was already static from the
point of view of `trace`, because the data are fixed when the model is
traced.

## Decision

1. **Configurations are traced one by one.** `trace` executes the model
   once per configuration of the discrete sites outside plates. Each
   execution takes a prefix of support indices; later discrete sites take
   index `0`, and the next configuration increments the deepest site that
   has a further support value. The configurations form a tree (a site may
   exist only for some values of earlier sites) whose leaves are visited in
   lexicographic order; for a static structure this is the order of
   `itertools.product`, so static models trace and evaluate exactly as
   before. Each configuration is recorded as a `ppl.TracedStructure`
   (`choices`, `assignment`, `sites`, `padded`); `ModelIR.structures` holds
   them, `ModelIR.sites` is their union in order of first appearance, and
   `ModelIR.dynamic` says whether they differ. A site name that changes
   kind, size, structured support, reparameterization, or discrete support
   between configurations is refused.
2. **Padding with the sites' own priors.** The unconstrained vector holds
   every continuous latent site of the union. For configuration `c`, the
   log-density is that of the model under `c` plus, for each continuous site
   that `c` does not declare, the prior and log-Jacobian the site has in the
   first configuration that declares it; the density on `u` is the
   log-sum-exp over the configurations ([EQ-PPL-DYNAMIC-001]). Each
   pseudo-prior is a proper density on its coordinates given the
   coordinates before it, so it integrates to one, and the marginal
   probability of every configuration is that of the model. This is the
   product-space construction of Carlin and Chib ([REF-CARLIN-CHIB-1995])
   with the prior as pseudo-prior, the composite-space view of
   [REF-GODSILL-2001]. Pseudo-priors are not tuned, so the engines sample a
   mixture of each configuration's posterior with the pseudo-priors of the
   sites it lacks.
3. **Per-configuration evaluation.** When the structure is dynamic, or a
   continuous latent or deterministic site follows a discrete site outside
   plates (now allowed), `ModelIR.per_configuration` is true: every
   configuration is evaluated in full (continuous terms included), latent
   coordinates are read by site name, and a numerical failure in one
   configuration gives it zero weight. Other models keep the earlier
   evaluation path unchanged. The ADR-0082 order rule still holds inside
   plate iterations.
4. **Recovery under the drawn configuration.** For each draw, the global
   configuration `c^(s)` is drawn from `p(c | u^(s), y)` and the reported
   values are those of `c^(s)`; sites it does not declare keep their
   pseudo-prior values in `posterior.constrained`, are `None` in
   `posterior[name]`, and are omitted from `posterior.draws()`.
   `PosteriorSamples.present(name)` gives the per-draw mask,
   `presence_probability(name)` the Rao-Blackwellized probability that a
   site exists, and `mean(name)` the mean given that it exists.
   `DiscreteDraws` gained `structures` and `structure_probabilities`;
   recovered discrete sites that a configuration lacks are `None`.
   Predictive simulation accepts varying structure and pads absent sites
   with `None`.
5. **Nested plates.** Active plates form a stack. Each iteration that
   declares discrete sites is re-run per configuration of its own sites;
   the log-sum-exp an inner iteration contributes enters the sink of the
   innermost enclosing iteration that has started one, and the checkpoint of
   a re-run keeps the sinks of all enclosing plates
   ([EQ-PPL-ENUM-004]; [REF-OBERMEYER-ETAL-2019]). The enumeration record of
   an iteration holds, per branch, the records of its nested iterations
   (`EnumerationGroup.children`), and recovery descends that tree. Inner
   plate sizes may depend on data and on enclosing indices; a plate may not
   reuse the name of an enclosing plate. `SiteInfo.plates` records the plate
   path.
6. **Vector-valued discrete sites.** `m.sample(name, law, shape=n,
   marginalize=True)`, a list of laws, or a `multivariate.Multinomial(n, p)`
   declare one discrete site whose value is a tuple. The joint support is
   the Cartesian product of the element supports or the
   `C(n + K - 1, K - 1)` count vectors, at most 4096 values
   ([EQ-PPL-ENUM-003]); the log-mass is the sum of the element log-masses or
   the multinomial log-mass. `discrete_probabilities` is keyed by the joint
   value and `mean` is elementwise.
7. **Refused structures.** Each raises `ModelSpecificationError` with a
   message naming the rule:
   - structure that depends on continuous values (checked at every
     evaluation against the traced structure of the configuration);
   - inside a plate iteration, sites that depend on the iteration's own
     discrete sites (checked at trace time by one log-density evaluation at
     `u = 0` when the model has local discrete sites and complete data), and
     continuous or deterministic sites after a local discrete site;
   - unbounded discrete supports (as before), more than 4096 traced
     configurations, and more than 64 discrete sites outside plates in one
     execution (an unbounded chain, whose trace cost grows with the square
     of its depth).

## Consequences

- Models that choose a number of components, a regime-specific set of
  parameters, or whether an effect exists are written directly, and the
  posterior probability of each structure is the Rao-Blackwellized average
  of `p(c | u^(s), y)`.
- Tracing costs one model execution per global configuration (before:
  one), plus one log-density evaluation for models with local discrete
  sites. Each log-density evaluation of a per-configuration model runs every
  configuration in full; for the one-or-two-component model of the tests
  (15 observations) that is about 0.9 ms per log-density and 1.6 ms per
  reverse-mode gradient on the development machine.
- The padded sites keep their priors as pseudo-priors. When a structure's
  posterior is far from the pseudo-prior of a site it lacks, the sampled
  density is a mixture of separated components and mixes slowly; tuned
  pseudo-priors are not provided.
- Behavior changes: nested plates, vector discrete sites, and continuous
  sites after a global discrete site are now accepted; they previously
  raised. `ModelIR` gained the field `structures` and the members
  `dynamic`, `per_configuration`, `presence`, and `recovery_record`;
  `PosteriorSamples` gained `present` and `presence_probability`. The
  fixed-seed results of the existing suites are unchanged: a mixture NUTS
  run, a truncated-Poisson run, a normal-model run, and a log-density of
  `tests/test_ppl_structured_sites.py` were compared bit for bit with the
  previous sources.
- New public name: `inference.ppl.TracedStructure`; `docs/api-reference.md`
  must be regenerated.

## Verification

`tests/test_ppl_dynamic_structure.py` (12 tests, about 30 s on the
development machine):

- **One or two components.** `k` in {1, 2} with equal prior mass,
  `mu[0] ~ N(0, 2^2)`, and, only for `k = 2`, `mu[1] ~ N(3, 1)` and one
  marginalized indicator per observation (weight 0.2), 15 observations.
  The PPL log-density equals the hand-written padded mixture at three points
  to `7e-15`, and its reverse-mode gradient agrees with central differences.
  Simpson quadrature of the hand-written marginal likelihoods gives
  `p(y | k = 1) = 2.3476e-12`, `p(y | k = 2) = 6.2786e-12`, so
  `p(k = 2 | y) = 0.727853`. An 81 x 81 Simpson integral of the PPL
  configuration weights over `u` gives `int p(u) du` equal to
  `sum_k p(k) p(y | k)` within a relative `2.1e-5` and `p(k = 2 | y) =
  0.727865`, which checks that the pseudo-prior integrates out. NUTS (seed
  7, two chains of 300 + 300 draws, reverse mode, about 11 s) gives the
  Rao-Blackwellized `p(k = 2 | y) = 0.7196` with a standard error of 0.0129
  (ESS 296 of the per-draw probabilities); `mu[1]` is present in 441 of 600
  draws and `presence_probability("mu[1]")` equals the model probability.
  Prior and posterior predictive simulation pad absent sites with `None`.
- **Continuous site after a global discrete site.** `z ~ Bernoulli(0.3)`,
  `x ~ N(2z, 1)`, `y = 1.4 ~ N(x, 0.5^2)`: the log-density equals the
  two-term mixture to twelve digits; NUTS (seed 4) gives
  `p(z = 1 | y) = 0.4454` (standard error 0.0122) against the closed form
  0.4484, and the deterministic site is reported under each draw's
  configuration.
- **Nested plates.** Three groups of sizes 2, 3, 4 (sizes from the data),
  a per-group label `c_g` and a per-observation outlier indicator: the
  enumerated log-likelihood equals the brute-force log-sum-exp over all
  4096 joint configurations within `3.6e-15`, and the hand-written nested
  formula; the reverse-mode gradient agrees with central differences; the
  recovered `p(c_g = 1 | theta, y)`, the per-branch `p(o_gj = 1 | theta,
  c_g, y)`, and the marginal `p(o_gj = 1 | theta, y)` equal exact Bayes to
  twelve digits, and 4000 recovered draws match the marginal within five
  binomial standard errors. A model with indicators two plates deep and no
  outer discrete site matches the closed form.
- **Vector discrete sites.** Three independent indicators as one site
  (8 joint values) and a `Multinomial(3, (0.2, 0.3, 0.5))` count vector
  (10 joint values), enumerated jointly (80 configurations): the
  log-density equals the hand-written double sum within `9e-16`, and the
  recovered joint conditional probabilities equal exact Bayes to twelve
  digits. With Laplace draws, `discrete_probabilities("z")` and the
  elementwise `mean("z")` equal the averages of the exact formulas over the
  draws to twelve digits. A `shape=2` vector inside a plate matches the
  closed form.
- **Structure from data and refused structures.** Plate counts and sizes
  taken from the supplied data give the expected dimensions. Value-dependent
  structure, local structure controlled by a plate's discrete site, a
  continuous site after a local discrete site, a Poisson number of
  components, an unbounded chain of discrete sites (both the
  many-configuration and the endless single-execution forms), a site whose
  kind changes between configurations, a nested plate with the name of its
  parent, and too-large vector supports raise `ModelSpecificationError`.
- `tests/test_ppl_structured_sites.py` (with three cases of
  `test_invalid_models_raise` removed because those models are now
  supported), `tests/test_probabilistic_programming.py`,
  `tests/test_mcmc_diagnostics_autodiff.py`, `tests/test_hmc_nuts_advi.py`,
  and `tests/test_reverse_autodiff.py` pass.

What remains: dynamic structure controlled by discrete sites inside plate
iterations (a branch of an iteration that declares different sites), tuned
pseudo-priors or reversible-jump moves for structures whose posteriors are
far from the priors, and variable elimination across plate iterations
(hidden Markov chains), which still require joint enumeration within the
4096-configuration limit.
