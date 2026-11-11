# ADR-0088: Markov-chain elimination, structure inside plate iterations, and tuned pseudo-priors in the probabilistic-programming layer

- Status: Accepted
- Date: 2026-11-12

## Context

ADR-0084 left three items of `probstat.inference.ppl` open:

- discrete sites inside plate iterations could not control the site
  structure of the iteration (a branch that declares a further continuous
  site or a different observation was refused);
- the pseudo-priors of padded sites were their priors and could not be
  tuned, and there were no reversible-jump moves;
- there was no variable elimination across plate iterations, so a hidden
  Markov chain of `n` binary states had to be enumerated jointly (`2^n`
  configurations, refused above 4096).

No other module of the library implements discrete-state hidden Markov
models: the filtering package holds Kalman, extended Kalman, and particle
filters for continuous states, and `processes.MarkovChain` describes an
observed chain. The new code is therefore checked against brute-force
enumeration and an independent scaled forward recursion written in the
tests.

## Decision

1. **`m.markov(name, n)`.** A new handle method yields `(t, previous)` for
   the steps of a first-order chain; `previous` is `None` at `t = 0` and
   otherwise the value of the discrete site of step `t - 1` (a tuple when the
   step declares several). The discrete sites (`marginalize=True`) and
   observations of step `t` may depend on `previous`. In density mode the
   step body is executed once per pair (state of step `t - 1`, joint state of
   step `t`); the sum of every term of that execution is `l_t(i, j)`, and the
   forward variables `alpha_t(j) = log sum_i exp(alpha_(t-1)(i) + l_t(i, j))`
   are carried in log space, with `log sum_j exp(alpha_(n-1)(j))` added to
   the enclosing scope ([EQ-PPL-MARKOV-001]; [REF-RABINER-1989],
   [REF-BAUM-ETAL-1970]). The cost is `K_0 + sum_(t >= 1) K_(t-1) K_t` body
   executions, `O(n K^2)`, instead of `K^n`. The recursion is transparent to
   reverse-mode variables, so NUTS uses the exact gradient. Each step may
   enumerate at most 4096 joint states; a step may contain nested plates,
   whose discrete sites are enumerated per pair; a chain may sit inside a
   plate iteration (several independent sequences). Refused, with
   `ModelSpecificationError`: continuous latent and deterministic sites
   inside a step, a chain nested in a chain, and step structure that depends
   on the step's states. Discrete sites in a chain record `SiteInfo.markov`.
2. **Recovery along chains.** The enumeration record of a chain is a
   `MarkovGroup` (initial log-weights and the transition tables of every
   step, and the records of nested iterations per pair). Recovery computes
   the smoothed marginals `exp(alpha_t + beta_t - log L)` and the pairwise
   probabilities from the backward recursion, and draws the path by forward
   filtering and backward sampling ([EQ-PPL-MARKOV-002]; [REF-CHIB-1996]).
   `ModelIR.most_probable_discrete(u)` returns the jointly most probable
   values of every discrete site by max-product elimination: the Viterbi
   recursion along chains and, for the record as a whole, the gain
   `max - logsumexp` of every group added to the weight that contains it
   ([EQ-PPL-MARKOV-003]; [REF-VITERBI-1967]).
3. **Structure controlled by discrete sites inside plate iterations.** The
   branches of a plate iteration may declare different observed and
   continuous latent sites. `trace` now runs every branch of every plate
   iteration; the first branch defines the checked site sequence
   (`TracedStructure.layout` when it differs from the union), a branch whose
   sites differ is recorded by its branch path in
   `TracedStructure.branches`, and continuous sites declared after a local
   discrete site are listed in `TracedStructure.enumerated`. In density
   mode a continuous site after a local discrete site contributes its prior
   and log-Jacobian to the branch's enumerated terms, and every branch that
   lacks a site of the iteration's union adds the site's pseudo-prior (its
   prior and log-Jacobian in the first branch that declares it), so the
   iteration contributes
   `log sum_z exp(l_i(z, u) + sum_(s absent in z) log q_s(u_s))`
   ([EQ-PPL-LOCAL-001]). The union of a nested iteration is declared by the
   enclosing branch, so an outer branch that lacks an inner iteration pads
   its sites too. Such models are evaluated configuration by configuration
   and read coordinates by name (`ModelIR.per_configuration`). The record of
   the iteration keeps, per branch, the sites it lacks
   (`EnumerationGroup.absent`); `DiscreteDraws.absent` and
   `DiscreteDraws.absence` carry, per draw, the sites missing from the drawn
   branches and their conditional absence probabilities, so
   `PosteriorSamples.present`, `presence_probability`, `__getitem__`, `mean`,
   and `draws` treat these sites as they treat sites absent from a global
   configuration. Kept limits: at most 4096 branches per iteration; the
   discrete sites of an iteration and their supports must be the same in
   every branch; a name declared in several branches must have the same
   kind, size, support, and transform in each; deterministic sites must
   precede the iteration's discrete sites; the structure of a branch must
   not depend on continuous values (checked against the traced branch at
   every evaluation).
4. **Tuned pseudo-priors.** `ppl.PseudoPrior(mean, scale)` is an
   independent normal density on a site's unconstrained coordinates.
   `ppl.tune_pseudo_priors(pilot)` moment-matches one per site that some
   configuration or plate branch lacks, with the Rao-Blackwellized weights
   `w_s = P(site declared | u^(s), y)` ([EQ-PPL-PSEUDO-001]); since
   `E[u 1(declared)] = E[u P(declared | u, y)]`, the weighted moments
   estimate the conditional posterior given that the site exists, the
   choice that [REF-CARLIN-CHIB-1995] recommend. `ppl.sample(...,
   pseudo_priors=...)` and `ModelIR.with_pseudo_priors` use them in the
   global padding of ADR-0084 and in the per-iteration padding above. The
   default (`None`) keeps each site's prior, so seeded results are
   unchanged.
5. **No reversible-jump moves.** The product-space (composite-space)
   method with tuned pseudo-priors is the transdimensional sampler of the
   layer; it is an accepted alternative to reversible jump
   ([REF-CARLIN-CHIB-1995], [REF-GODSILL-2001]). Reversible jump is not
   needed for the supported structure class: the structures are a finite
   set traced before sampling (at most 4096 global configurations and 4096
   branches per plate iteration; unbounded structures are refused), so the
   union of their continuous sites has a fixed, known dimension and the
   padded density on that space is exact for any proper pseudo-priors.
   [REF-GODSILL-2001] shows that both methods are samplers on one composite
   space and that reversible jump avoids carrying the parameters of unused
   models; that saving matters for unbounded or very large model sets,
   which the layer does not accept. Reversible jump would also need a
   dimension-matching proposal and its Jacobian for every pair of
   structures, which cannot be derived from model code, whereas the
   gradient-based engines need a fixed-dimension differentiable target,
   which the padded density is.

## Consequences

- Hidden Markov models, regime-switching models, and several independent
  sequences are written with `m.markov`; the marginal of the continuous
  parameters is exact and differentiable. For the 40-step two-state chain
  of the tests one log-density costs 158 step executions, about 2.6 ms, and
  one reverse-mode gradient about 5.4 ms on the development machine.
- Spike-and-slab and "an effect exists for this unit" models with
  per-unit parameters are written directly inside `m.plate`.
- `trace` executes every branch of every plate iteration (before: the
  first branch only), which costs as much as one log-density evaluation;
  for models without branch-dependent structure the traced sites, the
  layouts, and every log-density are as before.
- Behavior changes: continuous latent sites after a local discrete site and
  branch-dependent observed sites inside plate iterations are now accepted;
  they previously raised. `local_branch` and `local_continuous` were removed
  from the refused cases of `tests/test_ppl_dynamic_structure.py`;
  deterministic sites after a local discrete site are still refused.
- New public names: `ModelHandle.markov`, `ModelHandle.unconstrained`,
  `ModelIR.most_probable_discrete`, `ModelIR.with_pseudo_priors`,
  `ModelIR.pseudo_priors`, `ppl.PseudoPrior`, `ppl.tune_pseudo_priors`, the
  `pseudo_priors` argument of `ppl.sample`, `SiteInfo.markov`,
  `TracedStructure.layout`, `TracedStructure.branches`,
  `TracedStructure.enumerated`, `DiscreteDraws.absent`, and
  `DiscreteDraws.absence`; `docs/api-reference.md` and
  `docs/math-reference/` are regenerated.
- Fixed-seed results of the existing suites are unchanged: the
  one-or-two-component NUTS run (seed 7, `p(k = 2 | y) = 0.719644886043826`),
  its log-density and reverse-mode gradient, the nested-plate log-density
  and gradient, and the mixture NUTS run of
  `tests/test_ppl_structured_sites.py` (sum of the draws and the per-draw
  probabilities of its first indicators) were compared digit for digit with
  the previous sources.

## Verification

`tests/test_ppl_markov.py` (9 tests, about 37 s on the development
machine):

- **Forward algorithm.** Two-state chain with initial law `(0.6, 0.4)`,
  transitions `((0.85, 0.15), (0.25, 0.75))`, and unit-variance Gaussian
  emissions: for `n` in {1, 2, 5, 10} and three emission-mean pairs, the
  log-likelihood equals the log-sum-exp over all `2^n` paths within
  `3.6e-15` and the scaled forward recursion within `1e-12`. A counter in
  the step body shows `2 + 4 (n - 1)` executions per log-density; the
  reverse-mode gradient agrees with central differences. A three-state
  chain with a second indicator per step (6 joint states per step, 7776
  joint paths for 5 steps) and three sequences in a plate match brute
  force within `1e-12`; a chain whose steps hold a plate of two outlier
  indicators matches the sum over all 4096 joint configurations within
  `1e-12`, its recovered state and indicator probabilities equal exact
  Bayes to twelve digits, and the joint mode equals the brute-force
  maximum.
- **Recovery at fixed parameters** (`n = 10`): the recovered marginals
  equal the exact smoothed probabilities to twelve digits; 4000 forward
  filtering, backward sampling draws match every marginal and the pairwise
  law of `(z_3, z_4)` within five binomial standard errors; the Viterbi
  path equals the most probable of the 1024 paths.
- **NUTS** on the 40-step chain with ordered emission means
  (`OrderedVector(Normal(0, 3), 2)`, seed 3, two chains of 250 + 250 draws,
  reverse mode, about 32 s): posterior means `-0.6850` (MCSE 0.0146) and
  `1.8186` (MCSE 0.0143) against `-0.6805` and `1.8132` from a 101 x 101
  Simpson quadrature of the exact marginal; the true means `(-1.0, 1.5)` lie
  1.6 and 1.0 posterior standard deviations away. With two unordered
  normal priors both chains of the same run settled in the label-swapped
  local mode, which is why the test uses the ordered prior.
- **Refused structures**: a continuous site in a step, a nested chain, an
  observation present for one state only, and a chain of length zero.
  Prior predictive simulation passes the drawn state to the next step.

`tests/test_ppl_dynamic_structure.py` (new classes `LocalStructureTests`,
`PseudoPriorTuningTests`):

- **In-plate structure.** Three observations, `z_i ~ Bernoulli(0.4)`, and
  `theta_i ~ N(mu, 1.5^2)`, `y_i ~ N(theta_i, 0.5^2)` only when `z_i = 1`
  (else `y_i ~ N(mu, 1)`): the log-density equals the hand-written padded
  sum within `1e-12` at two points, each iteration's branch weights equal
  the hand formula to twelve digits, and the reverse-mode gradient agrees
  with central differences. A 2001-node Simpson integral of each branch
  weight over `theta_i` equals `0.6 N(y_i; mu, 1)` and
  `0.4 N(y_i; mu, 2.5)` to nine digits, which checks that the pseudo-prior
  integrates out. NUTS (seed 11, two chains of 300 + 300) gives
  `p(z_i = 1 | y) = 0.3114, 0.5826, 0.3835` (standard errors 0.0164, 0.0368,
  0.0213) against `0.3215, 0.5864, 0.3808` by quadrature over `mu`;
  `theta[i]` is present exactly in the draws with `z_i = 1`, and its
  presence probability equals the Rao-Blackwellized `p(z_i = 1 | y)` to
  twelve digits. A two-level model in which an outer label decides whether
  inner indicators and offsets exist matches the hand formula within
  `1e-12`, with recovered label and presence probabilities exact to twelve
  digits. A branch-only discrete site, a site whose support changes
  between branches, and a branch whose sites depend on a continuous value
  are refused.
- **Tuned pseudo-priors** on the one-or-two-component model of ADR-0084: a
  pilot NUTS run (seed 1, two chains of 300 + 300, about 9 s) gives the
  pseudo-prior `N(2.786, 0.626^2)` for `mu[1]` (its prior is `N(3, 1)`). An
  81 x 81 Simpson integral of the tuned configuration weights still gives
  the evidence within a relative `4.2e-6` and `p(k = 2 | y) = 0.727852`
  (exact 0.727853). Effective sample sizes over 600 draws, default against tuned,
  for seeds 7, 8, 9, 10: per-draw `p(k = 2 | u)` 296, 350, 424, 320 against
  510, 484, 524, 523; the drawn indicator 530, 542, 518, 521 against 545,
  564, 608, 574. The standard error of the Rao-Blackwellized estimate falls
  from 0.0114-0.0129 to 0.0093-0.0103, and the estimates move from 0.7196,
  0.7159, 0.7131, 0.7575 to 0.7263, 0.7248, 0.7260, 0.7358. The test
  asserts the ESS gain for seed 7 and that the default run reproduces
  `0.719644886043826`.

`tests/test_ppl_structured_sites.py`, `tests/test_probabilistic_programming.py`,
and `tests/test_mcmc_diagnostics_autodiff.py` pass unchanged.

What remains: a Markov step is first-order, may not declare continuous
latent or deterministic sites, may not change its sites with its states,
and holds at most 4096 joint states; coupled chains must be written as one
chain on the joint state, and elimination is provided for plates and
chains only, not for general discrete graphs. Inside a plate iteration the
discrete sites must be the same in every branch and deterministic sites
must precede them. Pseudo-priors are independent normals on the
unconstrained coordinates, tuned from a separate pilot run rather than
adapted during sampling; there are no reversible-jump moves.
