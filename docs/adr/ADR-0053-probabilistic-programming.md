# ADR-0053 — A small probabilistic-programming interface (`inference.ppl`)

- Status: Accepted
- Date: 2026-10-12

## Context

`probstat.inference` already had the engines a Bayesian workflow needs:
adaptive Metropolis, HMC and NUTS, ADVI, Laplace, and posterior predictive
checks. All of them take a log-density on `R^d`. Users therefore had to write
the log-density by hand, which meant:

- choosing transforms to unconstrained space for positive or bounded
  parameters;
- adding the log-Jacobians;
- mapping the draws back.

`hierarchical.non_centered_log_density` is one such hand-written density.
Roadmap Phase 6 asks for a probabilistic-programming intermediate
representation and an inference API on top of it.

## Decision

1. **Location.** The interface is the subpackage `probstat.inference.ppl`,
   layer 2, beside `inference.survival`. It is not a new top-level package.
   It is not re-exported from `probstat.inference`, because its natural
   names (`sample`, `laplace`, `posterior_predictive`) would clash with
   existing ones. As a subpackage with `__all__`, it gets its own section in
   `docs/api-reference.md`.
2. **Models are plain functions.** A model is `model(m)`, and the handle `m`
   offers three methods:
   - `sample(name, dist, shape=None, reparam=None)`;
   - `observe(name, dist, value=None, shape=None)`;
   - `deterministic(name, value)`.

   A distribution is any object with `support`, `log_pdf` (or `log_pmf`),
   and `sample(size, seed=)`. Library distributions qualify as they are, and
   so do small user classes such as the half-Cauchy in the lab. A
   list/tuple of distributions gives one law per element. `elementwise`
   builds such a list by broadcasting, because the library distributions
   take scalar parameters only. `condition(model, data)` turns `sample`
   sites into observed sites by name, or replaces the data of `observe`
   sites; the outermost condition wins.
3. **One handle, three modes.**
   - `trace`: deterministic. Every latent site takes `T(0)`.
   - `density`: reads an unconstrained vector.
   - `generate`: latents are fixed by name or drawn from the prior, and
     observed sites are drawn.

   The traced site list is the intermediate representation (`ModelIR`). Its
   structure must be static: every later execution is checked against it,
   and a change raises `ModelSpecificationError`.
4. **Bijections from the live support.** Each site's bijection is chosen
   from the support of the distribution active at that evaluation. The
   choices are identity, `a + exp(u)`, `b - exp(u)`, and the scaled logit.
   Because the support is read each time, bounds that depend on earlier
   latents (`Uniform(0, theta)`) are handled, and the Jacobian is still the
   sum of the per-site terms. Discrete latent sites are rejected; the user
   must marginalize them out. Errors raised inside the model by
   distribution constructors, and non-finite results, become `-inf`.
   `ModelSpecificationError` always propagates.
5. **Gradients by central differences.** Each gradient costs `2d` model
   executions. The library distributions convert their parameters to
   `float` during validation, so forward-mode dual numbers would be
   stripped at the first constructor. Exact gradients would need changes
   across `distributions`, and that is out of scope. The leapfrog
   integrator stays volume-preserving with an approximate force, so this
   costs efficiency but not correctness.
6. **Inference API.**
   - `sample(method="nuts"|"hmc"|"metropolis")`, `fit_advi`, and `laplace`
     all return `PosteriorSamples`. It holds draws by site name on the
     natural scale (`constrained`, an `MCMCResult` that keeps chain
     structure), the engine's unconstrained draws, and the raw engine result.
     For NUTS/HMC, `.hamiltonian` re-attaches the per-chain divergence and
     tree-depth statistics to the constrained draws.
   - `prior_predictive` and `posterior_predictive` re-run the model in
     `generate` mode.
   - `predictive_pvalues` wraps `posterior_predictive_pvalues` for one
     observed site.
7. **Non-centred option.** `reparam="noncentered"` is allowed for `Normal`
   sites only. The sampled coordinate is `z ~ N(0, 1)` and the value is
   `loc + scale z` (Papaspiliopoulos, Roberts and Sköld 2007).
8. **Speed.** In `density` mode the handle creates no `SiteInfo` records,
   and it caches bijections per support. `Normal` log-densities are
   inlined, using the same formula without argument validation.
   `foundations.validation.require_finite_real` gained a fast path for exact
   `float` inputs. Its behaviour is unchanged, but the `numbers.Real` ABC
   check was most of the cost of building a distribution. Together these
   took the eight-schools log-density from about 163 µs to about 75 µs per
   evaluation on the development machine.

## Consequences

- Eight schools written as a PPL model matches the exact grid posterior
  (`tests/test_probabilistic_programming.py`, seed 1, two chains of 150
  draws):
  - mu: 4.591 (MCSE 0.198) against exact 4.441;
  - tau: 3.257 (MCSE 0.163) against exact 3.290.
- The Beta-Binomial unconstrained density integrates to the exact evidence
  (0.0474308300) to ten digits, which checks the logit transform and its
  Jacobian.
- Cost is the main limitation. That NUTS run needed 5 610 gradients and
  117 810 model executions, about 7-11 s. Models with tens of parameters
  are practical; hundreds are not.
- Vector sites are flat tuples. There are no matrix-shaped sites, plates,
  or mixture enumeration.
- `PosteriorSamples.point` from `laplace` and `fit_advi` is located on the
  unconstrained scale. It is not the posterior mode on the natural scale.

## Rejected alternatives

- **A new top-level `probstat.ppl` package.** The task asks for an
  inference module, and nothing above layer 2 is needed.
- **Dual numbers threaded through the model.** Distribution validation
  converts parameters to `float`, so the derivatives would be lost silently.
- **Vectorized distribution classes.** They would duplicate the
  distribution catalogue. `elementwise` plus per-element laws reuse it
  unchanged.
