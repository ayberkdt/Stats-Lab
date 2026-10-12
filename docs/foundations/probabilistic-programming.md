# Probabilistic Programming

`probstat.inference.ppl` lets you write a Bayesian model as an ordinary
Python function. You then hand that function to the existing engines:

- NUTS, HMC, and adaptive Metropolis;
- ADVI;
- the Laplace approximation;
- posterior predictive checks.

You do not write a log-density by hand.

Every number below was printed by `examples/ppl_lab.py`, by
`tests/test_probabilistic_programming.py`, or by short scripts that used the
same models and seeds. They are the library's own outputs. None is copied
from the literature.

## 1. A model is a function

```python
from probstat.distributions import Normal, TruncatedNormal
from probstat.inference import ppl

y = (28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0)
s = (15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0)

def eight_schools(m):
    mu = m.sample("mu", Normal(0.0, 5.0))
    tau = m.sample("tau", TruncatedNormal(0.0, 5.0, lower=0.0))  # half-normal
    theta = m.sample("theta", Normal(mu, tau), shape=8, reparam="noncentered")
    m.observe("y", ppl.elementwise(Normal, theta, s), y)

posterior = ppl.sample(eight_schools, draws=1000, warmup=1000, chains=4, seed=1)
posterior.mean("tau"); posterior["theta"]; posterior.summary()
```

The handle `m` has three methods:

- `m.sample(name, dist, shape=None, reparam=None)` declares a latent
  variable. It returns a float for a scalar site, or a tuple for a site
  with `shape=n` or a list of distributions.
- `m.observe(name, dist, value)` adds `log p(value | ...)` to the
  likelihood. The value may be omitted if you supply it later with
  `ppl.condition(model, {"y": data})`. Conditioning also turns a `sample`
  site into an observed site.
- `m.deterministic(name, value)` records a derived quantity. It is reported
  with the draws and recomputed in predictive simulation.

A distribution is any object with three members:

- `support`, a `RealSupport` for latent sites;
- `log_pdf` (or `log_pmf`);
- `sample(size, seed=)`.

Every continuous law in `probstat.distributions` works as is. The lab's
half-Cauchy is a user class of about twenty lines.

The library distributions take scalar parameters. To give each element its
own law, pass a list. `ppl.elementwise(Normal, theta, s)` builds that list
by broadcasting scalars and equal-length sequences.

## 2. The intermediate representation

`ppl.trace(model)` runs the model once and returns a `ModelIR`. Each latent
site takes the value `T(0)`, so tracing uses no randomness. The `ModelIR`
holds the ordered sites with their sizes, supports, and bijections:

| Support | Bijection `x = T(u)` | log-Jacobian |
|---|---|---|
| `R` | `u` | `0` |
| `(a, inf)` | `a + exp(u)` | `u` |
| `(-inf, b)` | `b - exp(u)` | `u` |
| `(a, b)` | `a + (b - a) / (1 + exp(-u))` | `log(b - a) - softplus(u) - softplus(-u)` |

The table rows are [EQ-PPL-TRANSFORM-001] and [EQ-PPL-TRANSFORM-002]. The
joint log-density on `R^d` ([EQ-PPL-JOINT-001]) is

`log p(u) = sum_k log p(x_k | x_<k) + log p(y | x) + sum_k log |dT_k/du_k|`.

`ModelIR.terms(u)` returns the three parts separately. For eight schools,
`unconstrained_names` begins `mu, log(tau), theta_z[0]` and `d = 10`.

**The bijection is chosen from the live support.** It is re-derived at
every evaluation from the distribution that is active then. So
`m.sample("x", Uniform(0.0, theta))` works with a latent `theta`. The map
stays lower triangular in site order, so the Jacobian is still the sum of
the per-site terms. `tests/test_probabilistic_programming.py` checks this
case against a hand-derived log-density to ten digits.

**Rules the IR enforces.** Each of these raises `ModelSpecificationError`:

- a site name used twice;
- a latent site whose support is not a continuous `RealSupport` (for
  example `Binomial`; discrete latents must be marginalized by hand);
- a distribution without `support`;
- `reparam="noncentered"` on a non-normal site;
- an observed site without data;
- a `condition` name that matches no site;
- a site structure that changes with the parameter values. Every later
  execution is checked against the trace.

Numerical failures inside the model, such as `Normal(0, 0)` built from a
latent value, are mapped to `-inf`. Every sampler reads `-inf` as "outside
the support". `terms(u)` lets the exception through, for debugging.

**Gradients.** `ModelIR.gradient(u)` uses central differences
([EQ-OPT-FD-001]): two model executions per coordinate. NUTS and HMC compute
the same approximation internally.

Exact gradients would need dual numbers to survive the distribution
constructors, which convert parameters to `float`. So exact gradients are
not offered (ADR-0053). Leapfrog stays volume-preserving and reversible
with an approximate force, so the approximation costs efficiency, not
correctness. Cost is the practical limit. The eight-schools test run
(two chains of 150 warm-up and 150 kept draws) needed 5 610 gradients,
which is 117 810 model executions. That took 7-11 s on the development
machine, at about 75 µs per execution.

## 3. Inference

All functions return `PosteriorSamples`:

- `posterior[name]` gives the pooled draws on the natural scale.
- `posterior.mean(name)` and `posterior.summary()` give means, quantiles,
  ESS, and rank-normalized R-hat.
- `posterior.constrained` is an `MCMCResult` named `mu, tau, theta[0], ...`.
- `posterior.unconstrained` holds the draws on the sampling scale.
- `posterior.engine` is the raw engine result.
- `posterior.hamiltonian` (NUTS/HMC only) re-attaches the divergence and
  tree-depth statistics to the constrained draws.

The constrained draws are deterministic functions of the chain, so every
existing diagnostic applies to them unchanged.

| Function | Engine | Notes |
|---|---|---|
| `sample(model, method="nuts")` | `no_u_turn_sampler` | default |
| `sample(model, method="hmc")` | `hamiltonian_monte_carlo` | step-size jitter 0.2 by default |
| `sample(model, method="metropolis")` | `adaptive_metropolis` | adaptation during warm-up only |
| `fit_advi(model)` | `advi` | Gaussian on `R^d`: log-normal or logit-normal on the natural scale |
| `laplace(model)` | `laplace_approximation` | Gaussian at the unconstrained mode |

Starting points are drawn uniformly from `[-2, 2]^d` on the unconstrained
scale (`init_radius`), with separate child streams of `seed` per chain.

The Laplace `point`, like the ADVI mean, is located on the unconstrained
scale. Because of the Jacobian, it is not the posterior mode of the natural
parameter. For `tau = exp(u)` it maximizes `tau p(tau | y)`.

### Eight schools against the exact grid

The grid posterior of `hierarchical_normal_posterior` integrates `theta`
and `mu` out exactly (see [hierarchical models](hierarchical-models.md)).
It is the reference here.

*Test, half-normal(5) prior on tau, normal(0, 5) prior on mu.* Seed 1, two
chains of 150 warm-up and 150 kept draws:

| | exact | NUTS (MCSE) |
|---|---|---|
| E[mu] | 4.441 | 4.591 (0.198) |
| E[tau] | 3.290 | 3.257 (0.163) |

The run had no divergences and a mean tree depth of 2.9 and 3.1. The test
requires agreement within 4 MCSE and rank-normalized R-hat < 1.05.

*Lab, half-Cauchy(5) prior on tau.* The half-Cauchy is a user-defined
class. Seed 2026, two chains of 300 + 300 draws:

| | exact | NUTS (MCSE) | ESS |
|---|---|---|---|
| E[mu] | 4.397 | 4.301 (0.250) | 235 |
| E[tau] | 3.598 | 3.858 (0.179) | 285 |
| E[theta_1] | 6.212 | 6.366 (0.301) | |

The non-centred run had 0 divergences. The same model written centred
(tree depth capped at 6, 150 + 150 draws) had 3 divergences and an ESS of
only 9 for tau. Its tau mean of 3.408 happens to be close, but with an ESS
of 9 that number cannot be trusted. This is the funnel that
`reparam="noncentered"` removes ([EQ-PPL-NONCENTERED-001]).

### Conjugate checks

**Normal-normal.** The model is `mu ~ N(0, 3^2)` with ten observations
`y_i ~ N(mu, 1.2^2)`. The data are supplied by `condition`. The closed form
gives posterior mean 4.8425 and sd 0.3765.

Results from the lab (seeds 1-5):

| Method | mean | sd |
|---|---|---|
| Laplace (2 000 draws) | 4.8598 | 0.3804 |
| NUTS | 4.8400 | 0.3724 |
| HMC | 4.8391 | 0.3762 |
| Metropolis | 4.8503 | 0.3808 |
| ADVI | 4.8220 | 0.3674 |

The Laplace mode itself is 4.842520, exact to six digits. In the test, the
Laplace sd is 0.376473, also exact to six digits. The Laplace mean in the
table is a sample mean of draws, so it carries Monte Carlo error.

**Beta-binomial.** The model is `p ~ Beta(2, 3)` with 13 successes in 20
trials, so the exact posterior is `Beta(15, 10)` (mean 0.6000, sd 0.0961).

- The lab integrates `exp(log p(u))` over `u` by the trapezoid rule and
  gets 0.0474308300. The exact evidence `p(k)` is also 0.0474308300. This
  checks the logit bijection and its Jacobian together.
- The test checks the same thing pointwise: `log p(u)` minus
  `log Beta(15, 10)(sigmoid(u))` minus `log J(u)` equals the log evidence
  to ten decimal places at five values of `u`.
- NUTS gives mean 0.5996 and sd 0.0915 (lab, seed 7), or 0.6075 (MCSE
  0.0048) and sd 0.0957 (test, seed 8).
- Every draw lies inside (0, 1).

## 4. Predictive simulation

- `prior_predictive(model, draws, seed)` draws latent sites from their
  priors in execution order, then draws observed sites.
- `posterior_predictive(model, posterior, seed)` re-runs the model with
  each latent site fixed, by name, to its posterior value. It draws the
  observed sites, so `y_rep ~ p(y | theta^(s))` ([EQ-PPL-PREDICTIVE-001]).

Both return `PredictiveSamples` keyed by site name. The same seed gives the
same draws.

`predictive_pvalues(model, posterior, "y", statistics=...)` passes the
constrained `MCMCResult` and a simulator that re-runs the model to
`posterior_predictive_pvalues` ([EQ-PPC-001]). Because the chain structure
is kept, the p-value MCSE uses the effective sample size of the indicator
chains.

For eight schools (lab, half-Cauchy prior):

- The prior predictive range `max y - min y` has median 43.3 and 90 %
  quantile 99.6. The observed range is 31.0.
- The posterior predictive p-values are 0.327 for `max` (MCSE 0.020),
  0.088 for `min` (0.011), and 0.750 for `sd` (0.019).

## 5. Limitations

- Gradients come from finite differences, so each gradient costs `2d + 1`
  model executions. The interface suits models with up to a few tens of
  parameters.
- Sites are scalars or flat vectors. There are no matrix-shaped sites,
  plates, discrete-latent enumeration, or mixture marginalization.
- The site structure must be static.
- `reparam="noncentered"` applies to `Normal` sites only.
- Starting values cannot be supplied on the natural scale, because the
  inverse of a non-centred site depends on its parents.
- Posterior predictive p-values reuse the data and are not uniform under
  the true model (see [hierarchical models](hierarchical-models.md)).

## References

- [REF-CARPENTER-ETAL-2017] Stan: transforms of constrained parameters and
  the unconstrained log-density with Jacobian.
- [REF-BINGHAM-ETAL-2019] Pyro: models as Python functions with named
  `sample` sites and conditioning.
- [REF-PAPASPILIOPOULOS-ROBERTS-SKOLD-2007] Centred and non-centred
  parameterizations.
- [REF-KUCUKELBIR-ETAL-2017] ADVI on the unconstrained space.
- [REF-GELMAN-ETAL-2013] BDA3, Chapters 5 (eight schools) and 6 (predictive
  checks).
