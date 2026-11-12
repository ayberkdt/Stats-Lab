# Frequentist and Bayesian Interpretation

Both schools use the same probability calculus, but they attach
probabilities to different things. A frequentist statement is about a
*procedure* applied to data that could have come out differently, with the
parameter held fixed. A Bayesian statement is about the *parameter*, given
the data that did come out and a stated prior. The library keeps the two
explicit: every interval, test, or posterior it returns documents which
statement it supports, and this guide puts the two side by side on one data
set.

Every number in this guide is printed by the code blocks below, which call
existing `probstat` functions; `tests/test_measure_theory.py` executes the
blocks and checks that each quoted value is reproduced.

## 1. Two meanings of one model

The model is `X ~ Binomial(n, p)`. In measure-theoretic terms
([measure-theoretic probability](measure-theoretic-probability.md)), the
frequentist works with the family of measures `P_p` on the sample space,
one for each fixed `p`; the Bayesian works with one joint measure
`pi(dp) P_p(dx)` on parameter space times sample space and conditions it on
the observed `x` ([EQ-MEAS-008]).

| | Frequentist | Bayesian |
|---|---|---|
| What is random | the data `X`; `p` is fixed and unknown | `p`, through the prior; the observed `x` is fixed |
| Probability means | long-run frequency over repeated samples | conditional probability given prior and data |
| Interval | confidence: `P_p(L(X) <= p <= U(X)) >= 1 - alpha` for every `p` | credible: `P(l <= p <= u \| x) = 1 - alpha` |
| Test of `p = p_0` | p-value `P_(p_0)(T(X) at least as extreme as T(x))` | posterior `P(H_0 \| x)` and Bayes factor `B_01` |
| Calibration claim | coverage for every fixed `p` | coverage averaged over the prior |
| Depends on | the sampling plan (what else could have been observed) | the prior (and the likelihood of the observed data only) |

Confidence intervals and tests follow Casella and Berger
[REF-CASELLA-BERGER-2002, Chapter 9] and Wasserman
[REF-WASSERMAN-2004, Chapter 6]; posterior inference follows Gelman et al.
[REF-GELMAN-ETAL-2013, Chapter 2] and Wasserman
[REF-WASSERMAN-2004, Chapter 11].

## 2. One data set, two intervals

Fourteen successes in twenty trials. The frequentist reports the Wilson
score interval [REF-WILSON-1927] and the Clopper-Pearson interval
[REF-CLOPPER-PEARSON-1934], [REF-AGRESTI-2013, Chapter 1]; the Bayesian
with a uniform `Beta(1, 1)` prior reports the `Beta(1 + x, 1 + n - x)`
posterior [REF-GELMAN-ETAL-2013, Chapter 2] and its equal-tailed credible
interval.

```python
import math

from probstat.distributions import Beta, Binomial
from probstat.inference import (
    beta_binomial_log_likelihood,
    binomial_test,
    clopper_pearson_interval,
    evaluate_estimator,
    simulation_based_calibration,
    wilson_score_interval,
)

n, x = 20, 14
wilson = wilson_score_interval(x, n, confidence=0.95)
exact = clopper_pearson_interval(x, n, confidence=0.95)
print(f"estimate        {x / n:.4f}")
print(f"Wilson          [{wilson.lower:.4f}, {wilson.upper:.4f}]")
print(f"Clopper-Pearson [{exact.lower:.4f}, {exact.upper:.4f}]")

posterior = Beta(1 + x, 1 + n - x)
print(f"posterior mean  {posterior.mean:.4f}")
print(
    f"credible        [{posterior.quantile(0.025):.4f}, "
    f"{posterior.quantile(0.975):.4f}]"
)
print(f"P(p > 0.5 | x)  {posterior.sf(0.5):.4f}")
```

```text
estimate        0.7000
Wilson          [0.4810, 0.8545]
Clopper-Pearson [0.4572, 0.8811]
posterior mean  0.6818
credible        [0.4782, 0.8541]
P(p > 0.5 | x)  0.9608
```

| Statement | Frequentist reading | Bayesian reading |
|---|---|---|
| `[0.4810, 0.8545]` (Wilson) | a procedure that covers the true `p` in about 95% of repeated samples produced this interval | none: it is not a probability statement about `p` |
| `[0.4782, 0.8541]` (credible) | none without a coverage calculation (Section 4) | `P(0.4782 <= p <= 0.8541 \| x) = 0.95` under the uniform prior |
| `0.9608` | not defined: `p` is not random | posterior probability that `p > 0.5` |

The Wilson and credible intervals nearly coincide here. That is a property
of this sample size and this flat prior, not an equivalence: the two
intervals answer different questions and diverge for small counts
(Section 4) or informative priors.

## 3. Testing `p = 0.5`: p-value, posterior probability, Bayes factor

The exact binomial test reports the null probability of a count at least as
extreme as the observed one; it is not the probability that the null is
true. The Bayesian compares the point null `H_0: p = 0.5` with
`H_1: p ~ Uniform(0, 1)` through the Bayes factor
`B_01 = P(x | H_0) / P(x | H_1)` [REF-KASS-RAFTERY-1995]; the marginal
`P(x | H_1)` is the beta-binomial likelihood.

```python
two_sided = binomial_test(x, n, null_probability=0.5)
greater = binomial_test(x, n, null_probability=0.5, alternative="greater")
print(f"two-sided p-value     {two_sided.p_value:.4f}")
print(f"one-sided p-value     {greater.p_value:.4f}")
print(f"P(p <= 0.5 | x)       {posterior.cdf(0.5):.4f}")

null_mass = Binomial(n, 0.5).pmf(x)
marginal = math.exp(beta_binomial_log_likelihood([x], [n], 1.0, 1.0))
bf01 = null_mass / marginal
print(f"P(x | H0)             {null_mass:.4f}")
print(f"P(x | H1)             {marginal:.4f}")
print(f"Bayes factor B01      {bf01:.4f}")
print(f"Bayes factor B10      {1 / bf01:.4f}")
print(f"P(H0 | x), prior 1/2  {bf01 / (1 + bf01):.4f}")
```

```text
two-sided p-value     0.1153
one-sided p-value     0.0577
P(p <= 0.5 | x)       0.0392
P(x | H0)             0.0370
P(x | H1)             0.0476
Bayes factor B01      0.7763
Bayes factor B10      1.2882
P(H0 | x), prior 1/2  0.4370
```

- The one-sided p-value `0.0577` and the posterior probability
  `P(p <= 0.5 | x) = 0.0392` are different quantities: the first averages
  over unobserved counts `X >= 14` with `p = 0.5` fixed; the second
  averages over `p` with the observed count fixed.
- With equal prior odds the point null keeps posterior probability
  `0.4370`, far above the two-sided p-value `0.1153`. A p-value is not
  `P(H_0 | x)`, and for point nulls the two can disagree sharply
  [REF-BERGER-SELLKE-1987].
- `B_10 = 1.2882` lies between 1 and 3, the range Kass and Raftery describe
  as "not worth more than a bare mention" [REF-KASS-RAFTERY-1995,
  Section 3.2]. The Bayes factor depends on the prior under `H_1`; the
  p-value depends on the sampling plan. Neither dependence can be removed
  by switching schools.

## 4. Frequentist calibration: coverage at a fixed `p`

A confidence procedure is calibrated when its coverage
`P_p(L(X) <= p <= U(X))` is at least nominal for every `p`. For a binomial
model the coverage is a finite sum of binomial masses, so it can be
computed exactly. The credible interval, used as a frequentist procedure,
has no such guarantee.

```python
def credible(k):
    law = Beta(1 + k, 1 + n - k)
    return law.quantile(0.025), law.quantile(0.975)


def wilson_bounds(k):
    interval = wilson_score_interval(k, n)
    return interval.lower, interval.upper


def exact_bounds(k):
    interval = clopper_pearson_interval(k, n)
    return interval.lower, interval.upper


def coverage(bounds, p):
    law = Binomial(n, p)
    return math.fsum(
        law.pmf(k) for k in range(n + 1) if bounds(k)[0] <= p <= bounds(k)[1]
    )


for p in (0.01, 0.05, 0.5):
    print(
        f"p = {p:<4}  Wilson {coverage(wilson_bounds, p):.4f}  "
        f"Clopper-Pearson {coverage(exact_bounds, p):.4f}  "
        f"credible {coverage(credible, p):.4f}"
    )
```

```text
p = 0.01  Wilson 0.9831  Clopper-Pearson 0.9831  credible 0.8179
p = 0.05  Wilson 0.9245  Clopper-Pearson 0.9841  credible 0.9245
p = 0.5   Wilson 0.9586  Clopper-Pearson 0.9586  credible 0.9586
```

Clopper-Pearson never falls below 0.95 (it inverts exact tests and is
conservative); Wilson undercovers at `p = 0.05` (`0.9245`); the credible
interval undercovers badly near the boundary (`0.8179` at `p = 0.01`).

The same coverage can be estimated by repeated sampling with
`evaluate_estimator`, which reports a Monte Carlo standard error with every
quantity:

```python
def bernoulli_sampler(size, rng, p=0.05):
    return [1.0 if rng.random() < p else 0.0 for _ in range(size)]


def wilson_from_sample(sample):
    interval = wilson_score_interval(int(sum(sample)), len(sample))
    return interval.lower, interval.upper


result = evaluate_estimator(
    lambda sample: sum(sample) / len(sample),
    bernoulli_sampler,
    0.05,
    n,
    4000,
    seed=20261111,
    interval=wilson_from_sample,
)
print(f"simulated Wilson coverage at p = 0.05: {result.coverage:.4f}")
print(f"Monte Carlo standard error:           {result.coverage_se:.4f}")
```

```text
simulated Wilson coverage at p = 0.05: 0.9193
Monte Carlo standard error:           0.0043
```

The simulated `0.9193` differs from the exact `0.9245` by about 1.2
standard errors (`0.0043`): the simulation agrees with the exact sum and
adds only Monte Carlo noise.

## 5. Bayesian calibration: coverage averaged over the prior

A credible interval carries a different guarantee. If `p` is drawn from the
prior and `x` from the model, the posterior `95%` interval contains `p`
with probability exactly `0.95`, because averaging the posterior over the
prior predictive returns the prior. With a uniform prior every count has
marginal probability `1 / (n + 1)`, so the prior-averaged coverage of any
interval rule is a finite sum of posterior probabilities.

```python
def prior_averaged(bounds):
    total = []
    for k in range(n + 1):
        lower, upper = bounds(k)
        law = Beta(1 + k, 1 + n - k)
        total.append((law.cdf(upper) - law.cdf(lower)) / (n + 1))
    return math.fsum(total)


print(f"prior-averaged coverage, credible:        {prior_averaged(credible):.4f}")
print(f"prior-averaged coverage, Wilson:          {prior_averaged(wilson_bounds):.4f}")
print(f"prior-averaged coverage, Clopper-Pearson: {prior_averaged(exact_bounds):.4f}")
```

```text
prior-averaged coverage, credible:        0.9500
prior-averaged coverage, Wilson:          0.9530
prior-averaged coverage, Clopper-Pearson: 0.9770
```

The credible interval is calibrated on average over the prior (`0.9500`),
yet Section 4 shows it can undercover at a particular `p`. Clopper-Pearson
is calibrated at every `p`, and therefore over-covers on average
(`0.9770`). Each guarantee is real; they are guarantees about different
experiments.

Simulation-based calibration [REF-TALTS-ETAL-2018] checks the Bayesian
guarantee by repeated sampling from the joint model: the rank of the true
`p` among posterior draws is uniform when the posterior is correct.

```python
def prior_sampler(generator):
    return [generator.random()]


def simulator(theta, generator):
    return sum(1 for _ in range(n) if generator.random() < theta[0])


def posterior_sampler(count, stream):
    draws = Beta(1 + count, 1 + n - count).sample(99, seed=stream)
    return [[draw] for draw in draws]


sbc = simulation_based_calibration(
    prior_sampler,
    simulator,
    posterior_sampler,
    replicates=400,
    bins=10,
    seed=20261111,
    names=["p"],
)
print(f"SBC chi-square ({sbc.bins - 1} df): {sbc.chi_square[0]:.2f}")
print(f"SBC p-value:             {sbc.p_values[0]:.4f}")
print(f"calibrated at 0.01:      {sbc.calibrated(0.01)}")
```

```text
SBC chi-square (9 df): 3.70
SBC p-value:             0.9300
calibrated at 0.01:      True
```

Note the layering: simulation-based calibration is a frequentist test
(the p-value `0.9300` refers to repeated SBC runs) of a Bayesian
computation. Using a tool from one school to check the other is legitimate
as long as each statement is labelled with the experiment it refers to.

## 6. Reporting rules

- Do not describe a confidence interval as containing `p` with 95%
  probability. Describe the procedure's coverage.
- Do not describe a p-value as the probability that the null hypothesis is
  true. `BinomialTestResult.p_value` and the other test results document
  this in their docstrings.
- Report the prior with every credible interval, posterior probability, or
  Bayes factor; the likelihood alone does not determine them.
- Report the sampling plan and the reference set with every p-value and
  confidence interval.
- When a single number is meaningful in both schools (for example the
  coverage of a rule), state which experiment it averages over: fixed `p`,
  or `p` drawn from the prior.

| Frequentist procedures | Bayesian procedures |
|---|---|
| `inference.wilson_score_interval`, `inference.clopper_pearson_interval`, `inference.student_t_mean_confidence_interval` | `inference.condition_linear_gaussian`, `inference.posterior_predictive` |
| `inference.binomial_test`, `inference.fisher_exact_test` | `inference.hierarchical_normal_posterior` |
| `inference.bootstrap_confidence_interval` | `inference.posterior_predictive_check`, `inference.posterior_predictive_pvalues` |
| `inference.evaluate_estimator` (coverage, bias, MSE) | `inference.simulation_based_calibration` |

## Common pitfalls

- **Two-sided is not twice one-sided.** `binomial_test` sums every outcome
  no more probable than the observed one. For an asymmetric null this
  differs from doubling the smaller one-sided p-value (`0.6295` against
  `0.7840` for 7 of 20 at `p_0 = 0.3`).
- **Wilson is not guaranteed.** `wilson_score_interval` undercovers at some
  `p` (`0.9245` at `p = 0.05` in Section 4); use
  `clopper_pearson_interval` when coverage must hold for every `p`.
- **Beta shape parameters.** `Beta(a, b)` takes the two shape parameters,
  so the uniform-prior posterior is `Beta(1 + x, 1 + n - x)` with mean
  `(1 + x) / (2 + n)`, not `x / n`.
- **Simulated coverage carries Monte Carlo error.** `evaluate_estimator`
  reports `coverage_se` next to `coverage`; a difference of one or two
  standard errors from an exact value is noise (Section 4).

```python
seven = binomial_test(7, 20, null_probability=0.3)
upper_tail = binomial_test(7, 20, null_probability=0.3, alternative="greater")
lower_tail = binomial_test(7, 20, null_probability=0.3, alternative="less")
print(f"two-sided p-value          {seven.p_value:.4f}")
print(f"doubled smaller tail       {2 * min(upper_tail.p_value, lower_tail.p_value):.4f}")
print(f"posterior mean (1+x)/(2+n) {Beta(1 + x, 1 + n - x).mean:.4f}")
print(f"sample proportion x/n      {x / n:.4f}")
```

```text
two-sided p-value          0.6295
doubled smaller tail       0.7840
posterior mean (1+x)/(2+n) 0.6818
sample proportion x/n      0.7000
```

## References

- [REF-CASELLA-BERGER-2002] Chapter 9 (interval estimation).
- [REF-WASSERMAN-2004] Chapters 6 and 11.
- [REF-GELMAN-ETAL-2013] Chapter 2 (binomial model, posterior summaries).
- [REF-WILSON-1927], [REF-CLOPPER-PEARSON-1934], [REF-AGRESTI-2013,
  Chapter 1].
- [REF-KASS-RAFTERY-1995] Section 3.2 (Bayes factor scale).
- [REF-BERGER-SELLKE-1987] (p-values and posterior probabilities of point
  nulls).
- [REF-TALTS-ETAL-2018] (simulation-based calibration).
