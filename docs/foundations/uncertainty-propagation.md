# Uncertainty Propagation, Quadrature, and Basic Monte Carlo

## From uncertain inputs to uncertain outputs

For a forward model `Y = g(X)`, uncertainty propagation asks for properties of
the output distribution induced by uncertain `X`. `propagate_samples` evaluates
explicit named parameter realizations through the validated forward-model
contract and retains both raw QoI observations and immutable summaries.

The implementation reports three quantities that must not be conflated:

- **Output variance** describes variability in simulated `Y` values.
- **Monte Carlo variance/standard error** describes sampling uncertainty in an
  estimator such as the output mean.
- **Numerical-model error** describes approximation error inside a model or
  quadrature rule and belongs in the numerical verification budget.

More model samples reduce standard error but do not automatically reduce model
bias, discretization error, model-form error, or input-distribution error.

## Expectation integrals and quadrature

For a scalar density, `E[g(X)] = integral g(x)f_X(x)dx`. The library provides
composite trapezoid and Simpson rules with finite-value validation and exact
function-evaluation counts. Simpson requires an even number of equal
subintervals.

`expectation_by_quadrature` accepts explicit finite bounds. When a distribution
has unbounded support, the result is a truncated expectation unless omitted
tail mass is bounded or corrected. A dense grid is not itself an error
estimate; refinement studies should accompany high-stakes results.

## Monte Carlo estimators

`estimate_mean` returns the sample mean, unbiased output sample variance,
estimated standard error, an explicitly normal-approximation confidence
interval, sample size, and evaluation count. Constant output is represented by
a zero-width interval rather than rejected.

`estimate_probability` accepts exact Boolean indicators and uses a Wilson score
interval. This avoids reporting the fragile uncorrected Wald interval,
especially when no events or all events are observed.

`estimate_uniform_integral` implements
`integral_a^b f(x)dx = (b-a)E[f(U)]`. Its contribution variance includes the
interval-width scaling, so the reported standard error has integral units.

## Convergence and cost

For independent, identically distributed finite-variance contributions, ideal
Monte Carlo standard deviation scales as `N^(-1/2)`. Four times as many samples
are therefore needed to halve standard error. `convergence_trace` computes
nested-prefix estimates at explicit checkpoints, and
`plot_monte_carlo_convergence` displays estimates and their reported intervals.

This rate is an assumption-backed asymptotic statement. Dependence, heavy
tails, nonstationarity, rare events, or biased sampling can invalidate its
ordinary interpretation. A visually stable trace is evidence, not proof.

<picture><source media="(prefers-color-scheme: dark)" srcset="../images/guides/propagation-design-convergence-dark.png"><img alt="Log-log plot of the spread of replicate mean estimates against sample size for Monte Carlo, Latin hypercube, and Sobol designs" src="../images/guides/propagation-design-convergence.png"></picture>

*For the laboratory's projectile range, the spread of plain Monte Carlo means follows the N^-1/2 guide (fitted slope -0.53), so four times the samples halve the error; Latin hypercube and randomized Sobol points fall faster, and at N = 2,048 their means are 28 and 34 times less variable.*

Every result carries an evaluation count. In forward propagation, one model
call may produce several QoIs, so `model_evaluations` counts complete calls
rather than multiplying by the number of outputs.

## Threshold probabilities

For a threshold `S`, strict exceedance is estimated as
`P(Y>S)=E[I(Y>S)]`. Equality is deliberately not counted. The threshold,
success count, trial count, standard error, Wilson interval, and model-call cost
remain inspectable.

## Reproducible laboratory

`examples/basic_monte_carlo_lab.py` includes:

1. Monte Carlo estimation of pi with a Wilson interval;
2. Simpson versus Monte Carlo integration of `x^2`;
3. propagation of uncertain speed and angle through an idealized projectile
   model; and
4. an output threshold probability and convergence plot.

All random streams are seeded locally. The projectile model is pedagogical and
does not include drag, Earth curvature, or numerical trajectory integration.

## Guaranteeing a quantile with Wilks' order statistics

The type-7 sample quantile of `n` model runs is a point estimate: for large
`n` it falls below the true quantile `q_a` about half of the time. A safety
argument needs a value that exceeds `q_a` with a stated probability `beta`.
Sort the outputs, `Y_(1) <= ... <= Y_(n)`. The number of runs below `q_a`
is `Binomial(n, a)` whatever the output law, so
`P(Y_(k) >= q_a) = P(Binomial(n, a) <= k - 1)` [EQ-PROP-WILKS-001]. The
Wilks rank is the smallest `k` for which this probability reaches `beta`;
it is exact for a continuous output law and conservative when the law has
atoms.

- With `n = 1000` runs and `a = beta = 0.95`, `propagation.wilks_rank`
  returns rank 962, the 39th largest output, with exact confidence 0.9567.
- The smallest study for a 95 %/95 % upper bound has 59 runs, and the bound
  is then the maximum (confidence `1 - 0.95^59 = 0.9515`). Using the second,
  third, or fourth largest value instead needs 93, 124, or 153 runs
  [EQ-PROP-WILKS-002]; these sizes equal
  `reliability.success_run_sample_size` with one, two, or three allowed
  failures.
- A bound on the median with 95 % confidence needs 5 runs.
- `propagation.wilks_tolerance_interval` gives `[Y_(m), Y_(n-m+1)]`, which
  contains at least a fraction `gamma` of the output law with confidence
  `beta` [EQ-PROP-WILKS-003]. Its content follows `Beta(n - 2m + 1, 2m)`, so
  the first-order 95 %/95 % interval `[min, max]` needs 93 runs, the same as
  a one-sided second-order bound. From 1000 runs it drops 18 values at each
  end (`m = 19`, confidence 0.9693).

A Wilks bound is one-sided: the whole risk `1 - beta` is in one tail.
`propagation.order_statistic_interval` is a two-sided, equal-tailed
confidence interval for `q_a` with `(1 - confidence)/2` in each tail, so its
upper rank at confidence 0.90 coincides with the Wilks rank at
`beta = 0.95`. Both statements need independent runs; Latin hypercube and
quasi-random designs do not provide them.

```python
from probstat.propagation import wilks_bound, wilks_rank, wilks_sample_size

rank = wilks_rank(1000, 0.95, 0.95)
rank.rank, rank.order                 # (962, 39)
round(rank.achieved_confidence, 4)    # 0.9567
[wilks_sample_size(0.95, 0.95, order=m) for m in (1, 2, 3, 4)]  # [59, 93, 124, 153]
wilks_sample_size(0.95, 0.95, sided="two-sided")   # 93
wilks_sample_size(0.5, 0.95)                       # 5
bound = wilks_bound([float(i) for i in range(1, 60)], 0.95, 0.95)
bound.bound, round(bound.empirical_quantile, 1)    # (59.0, 56.1)
```

<picture><source media="(prefers-color-scheme: dark)" srcset="../images/guides/propagation-wilks-quantiles-dark.png"><img alt="Empirical CDFs over replicates of the empirical 0.95-quantile and of the Wilks upper bound, with the true quantile" src="../images/guides/propagation-wilks-quantiles.png"></picture>

*Each of 1,000 replicates draws 1000 standard normal values and computes both the type-7 empirical 0.95-quantile and the Wilks bound, the order statistic of rank 962. The empirical quantile is centred on the true value 1.645 and falls below it in 52.2% of the replicates, as a point estimate should; the Wilks bound falls below it in only 3.9%, within the designed risk of 1 - 0.9567. The price of the guarantee is the horizontal gap between the two curves: the bound is a conservative value, not an estimate of the quantile.*

`examples/risk_analysis_lab.py` runs this replicate study after its
reliability estimates.

## Lattice rules and the median of randomized rules

`probstat.sampling` also integrates directly over the unit cube. A rank-1
lattice `x_k = frac(k z / n)` [EQ-SAMP-LATTICE-001] is defined by one integer
vector `z`. `probstat.sampling.cbc_generating_vector` chooses it for a prime
`n`, one component at a time, by minimizing the shift-averaged squared
worst-case error `e^2(z)` in a weighted Sobolev space
[EQ-SAMP-LATTICE-002]; the search is exhaustive in two dimensions and greedy
beyond. `probstat.sampling.korobov_generating_vector` searches the vectors
`(1, a, a^2, ...)` instead. An unshifted lattice contains the origin; a
random shift `frac(x_k + Delta)` makes every point uniform.

Three estimators combine `R` independent randomizations at a cost of `R n`
evaluations:

- `probstat.sampling.rqmc_mean` averages the replicates and reports their
  standard error with a Student-t interval [EQ-SAMP-RQMC-002]. The estimate
  is unbiased. The interval is approximate, because `R` is small and a
  replicate is not normally distributed: in the library's tests nominal 95 %
  intervals covered between 92 % and 96 % of the time.
- `probstat.sampling.rqmc_median` returns the median of the replicates
  [EQ-SAMP-RQMC-003]. It has no standard error and no interval, and none
  with guaranteed coverage is provided: the median of the replicates
  estimates the median of their distribution, which is the integral only if
  that distribution is symmetric about it.
- `probstat.sampling.median_random_lattice` draws a fresh random generating
  vector in every replicate instead of constructing one
  [EQ-SAMP-RQMC-004]. Most random vectors are good and a few are very bad;
  the median ignores the bad ones, the mean does not.

Lattice rules pay off for smooth periodic integrands. For
`f = prod_j (1 + B_2(x_j))` in three dimensions, eleven shifted CBC lattices
of 509 points have a root mean squared error of `4.5e-7`, plain Monte Carlo
at the same cost `1.5e-3`. For the non-periodic `prod_j x_j e^(x_j)` the
gain over Monte Carlo is only a factor of 3 to 5.

```python
import math
import statistics

from probstat.sampling import (
    cbc_generating_vector,
    median_random_lattice,
    rqmc_mean,
    rqmc_median,
)

def smooth(x):
    return math.prod(1.0 + v * v - v + 1.0 / 6.0 for v in x)  # integral 1

rule = cbc_generating_vector(127, 3)
rule.generating_vector             # (1, 29, 24)
round(rule.worst_case_error, 5)    # 0.01156

mean = rqmc_mean(smooth, 3, 127, replicates=11, seed=1)
mean.evaluations                           # 1397
round(mean.estimate, 6)                    # 1.0
round(mean.standard_error * 1e6, 2)        # 5.28
rqmc_median(smooth, 3, 127, replicates=11, seed=1).interval is None  # True

random_rules = median_random_lattice(smooth, 3, 509, replicates=11, seed=1)
round(random_rules.estimate, 6)                                # 1.0
round(statistics.fmean(random_rules.replicate_estimates), 6)   # 1.000105
```

In the last two lines one of the eleven random generating vectors is bad
(error `1.2e-3`); it moves the mean of the replicates to `1.000105` and
leaves the median at `1.0000004`.

## Common pitfalls

- **Different defaults for the same step.** `propagation.propagate` defaults
  to `design="mc"` and `seed=None` (fresh entropy, not reproducible);
  `UQStudy.propagate` defaults to `design="sobol"` on a child of the study
  seed.
- **Error bars for LHS and quasi-random designs.** Without `replicates`, the
  mean and exceedance intervals treat the points as independent draws. For
  LHS and Sobol designs they are conservative references; pass
  `replicates=r` for a Student-t interval over `r` randomizations, at cost
  `r * size`.
- **Strict exceedance.** `exceedance(t)` counts `Y > t`; an output exactly
  equal to `t` is not an exceedance, which matters for discrete or clipped
  outputs.
- **Output spread is not estimator error.** `sample_variance` describes the
  outputs; `standard_error` describes the mean estimate and shrinks like
  `n^(-1/2)`.
- **Truncated quadrature.** `expectation_by_quadrature` integrates only over
  the given bounds; Simpson's rule needs an even number of intervals.

```python
import inspect

from probstat.distributions import Normal
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.propagation import propagate
from probstat.uq import QuantityOfInterest
from probstat.workflow import UQStudy

inspect.signature(propagate).parameters["design"].default          # 'mc'
inspect.signature(UQStudy.propagate).parameters["design"].default  # 'sobol'

step = ProbabilisticModel(
    JointInput((RandomInput("x", Normal(0.0, 1.0)),)),
    lambda p: {"y": float(p["x"] > 0.0)},
    (QuantityOfInterest("y", "indicator of x > 0"),),
)
y = propagate(step, 200, seed=3)["y"]
y.exceedance(1.0).probability   # 0.0, although about half the outputs equal 1
plain = propagate(step, 64, design="sobol", seed=1)["y"]
plain.mean_interval().method    # 'normal approximation with estimated standard deviation'
replicated = propagate(step, 64, design="sobol", seed=1, replicates=4)["y"]
replicated.mean_interval().method  # 'Student-t interval over 4 randomized replicates of the mean'
```

## Sources

- [REF-SMITH-UQ-2024, Chapters 3-4].
- [REF-ROBERT-CASELLA-2004, Chapters 3-4].
- [REF-DAVIS-RABINOWITZ-1984, Chapters 2-3].
- [REF-WILSON-1927].
- [REF-WILKS-1941], [REF-WILKS-1942], [REF-NUTT-WALLIS-2004].
- [REF-CRANLEY-PATTERSON-1976], [REF-GODA-LECUYER-2022], [REF-PAN-OWEN-2023],
  [REF-LECUYER-NAKAYAMA-OWEN-TUFFIN-2023].
