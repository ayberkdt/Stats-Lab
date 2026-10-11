# Probability and Statistics Refresher

This refresher maps the probability and statistics a VV-UQ course assumes to
the objects that implement them. It is a map, not a textbook: each concept
links to the package that computes it and, where useful, to the foundation
guide that explains it in depth.

Every snippet below was executed; the block after it is its exact output, and
`tests/test_prerequisite_guides.py` re-runs all of them. Snippets share one
session, top to bottom.

Background reading: [REF-CASELLA-BERGER-2002], [REF-WASSERMAN-2004],
[REF-ROSS-2019].

## Concept map

| Concept | Where it lives | Deeper guide |
|---|---|---|
| Sample space, events, axioms | `probability.FiniteSampleSpace`, `FiniteEvent`, `FiniteProbabilityMeasure` | [probability](../foundations/probability.md) |
| Counting | `probability.factorial`, `permutations`, `combinations`, `multiplication_rule` | [probability](../foundations/probability.md) |
| Conditional probability, independence, Bayes | `FiniteProbabilityMeasure.conditional_probability`, `are_independent`, `total_probability`, `bayes_probability` | [probability](../foundations/probability.md) |
| Random variable, PMF, CDF, moments | `probability.FiniteRandomVariable` | [random variables](../foundations/random-variables.md) |
| Joint, marginal, conditional PMF, covariance | `probability.FiniteJointDistribution` | [random variables](../foundations/random-variables.md) |
| Parametric families (PMF/PDF, CDF, survival, quantile, sampling) | `distributions.*` (Normal, StudentT, ChiSquare, FisherF, Gamma, Beta, Binomial, Poisson, ...) | [continuous catalog](../foundations/continuous-distribution-catalog.md), [discrete families](../foundations/discrete-distributions.md) |
| Multivariate normal, marginals, conditionals | `multivariate.MultivariateNormal` | [multivariable calculus and conditioning](multivariable-calculus-and-conditioning.md) |
| Sample summaries | `statistics.mean`, `variance(ddof=)`, `quantile(method=7)`, `empirical_cdf`, `pearson_correlation`, `spearman_correlation` | [descriptive statistics](../foundations/descriptive-statistics.md) |
| Estimation (MLE), confidence intervals | `inference.fit_normal_mle`, `fit_gamma_mle`, `student_t_mean_confidence_interval`, `wilson_score_interval` | [uncertainty modeling and fitting](../foundations/uncertainty-modeling-and-fitting.md) |
| Hypothesis tests, multiple testing, bootstrap | `inference.one_sample_t_test`, `chi_square_independence_test`, `adjust_p_values`, `bootstrap_confidence_interval` | [hypothesis testing and bootstrap](../foundations/hypothesis-testing-and-bootstrap.md) |
| Law of large numbers, Monte Carlo error | `monte_carlo.estimate_mean`, `convergence_trace` | [uncertainty propagation](../foundations/uncertainty-propagation.md) |
| Reproducible randomness | `foundations.RandomStream`, `seed=` arguments | [probabilistic modeling](../foundations/probabilistic-modeling.md) |

## 1. Events, conditioning, and Bayes' rule

A diagnostic test with 1 % prevalence, 95 % sensitivity, and 90 % specificity
is the standard reminder that `P(disease | positive)` is not the sensitivity.

```python
from probstat.probability import FiniteProbabilityMeasure, FiniteSampleSpace

outcomes = (("ill", "+"), ("ill", "-"), ("well", "+"), ("well", "-"))
space = FiniteSampleSpace(outcomes)
weights = {
    ("ill", "+"): 0.01 * 0.95, ("ill", "-"): 0.01 * 0.05,
    ("well", "+"): 0.99 * 0.10, ("well", "-"): 0.99 * 0.90,
}
measure = FiniteProbabilityMeasure(space, weights)
ill = space.event(o for o in outcomes if o[0] == "ill")
positive = space.event(o for o in outcomes if o[1] == "+")
print(f"P(+) = {measure.probability(positive):.4f}")
print(f"P(ill | +) = {measure.bayes_probability(ill, positive):.4f}")
print(f"P(+ | ill) = {measure.conditional_probability(positive, ill):.4f}")
print(measure.are_independent(ill, positive))
```

```text
P(+) = 0.1085
P(ill | +) = 0.0876
P(+ | ill) = 0.9500
False
```

## 2. Random variables and moments

A random variable is a function on outcomes; its PMF, CDF, and moments are
induced by the measure.

```python
from probstat.probability import combinations

die = FiniteSampleSpace(range(1, 7))
fair = FiniteProbabilityMeasure.uniform(die)
from probstat.probability import FiniteRandomVariable
X = FiniteRandomVariable.from_function(fair, lambda face: face, name="X")
Y = X.transform(lambda value: (value - 3.5) ** 2, name="Y")
print(f"E[X]={X.mean:.4f} Var[X]={X.variance:.4f} E[Y]={Y.mean:.4f}")
print(f"P(X <= 4) = {X.cdf(4):.4f}")
print(f"C(10, 3) = {combinations(10, 3)}")
```

```text
E[X]=3.5000 Var[X]=2.9167 E[Y]=2.9167
P(X <= 4) = 0.6667
C(10, 3) = 120
```

`E[(X - EX)^2] = Var X`, as the last two numbers of the first line confirm.

## 3. Parametric distributions

Every continuous family exposes `pdf`, `log_pdf`, `cdf`, `sf` (survival
`1 - F`, computed directly for accuracy in the upper tail), `quantile`, `isf`,
`mean` and `variance` properties, and `sample(size, seed=)`. Discrete families expose `pmf`
and `log_pmf` instead of densities. Parameters follow the conventions spelled
out in each class docstring; the [R bridge](r-bridge.md) lists them next to
R's.

```python
from probstat.distributions import Normal, StudentT, Gamma, Binomial, Poisson

z = Normal(0.0, 1.0)
print(f"{z.cdf(1.96):.7f} {z.quantile(0.975):.6f} {z.sf(6.0):.6e}")
print(f"{StudentT(10).quantile(0.975):.6f}")
g = Gamma(shape=2.0, rate=0.5)
print(f"{g.mean:.4f} {g.variance:.4f} {Gamma.from_scale(2.0, 2.0) == g}")
print(f"{Binomial(10, 0.5).pmf(3):.7f} {Poisson(3.0).cdf(2):.7f}")
```

```text
0.9750021 1.959964 9.865876e-10
2.228139
4.0000 8.0000 True
0.1171875 0.4231901
```

## 4. Sample statistics and the empirical distribution

`statistics.variance` defaults to `ddof=0` (the variance of the finite values
themselves); pass `ddof=1` for the unbiased sample variance. Quantiles default
to Hyndman-Fan type 7 [REF-HYNDMAN-FAN-1996].

```python
from probstat.statistics import (
    mean, variance, standard_deviation, quantile, median, empirical_cdf,
    pearson_correlation, spearman_correlation,
)

data = (2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 2.8, 6.1, 3.9, 4.0)
print(f"mean={mean(data):.4f} var0={variance(data):.4f} var1={variance(data, ddof=1):.4f}")
print(f"sd1={standard_deviation(data, ddof=1):.4f} median={median(data):.4f}")
print(f"q(0.1)={quantile(data, 0.1):.4f} q(0.9)={quantile(data, 0.9):.4f}")
print(f"ecdf(3.4)={empirical_cdf(data, 3.4):.2f}")
other = tuple(x * x for x in data)
print(f"pearson={pearson_correlation(data, other):.4f} spearman={spearman_correlation(data, other):.4f}")
```

```text
mean=3.7500 var0=1.6825 var1=1.8694
sd1=1.3673 median=3.6500
q(0.1)=2.0800 q(0.9)=5.6500
ecdf(3.4)=0.50
pearson=0.9860 spearman=1.0000
```

A monotone transformation leaves Spearman's rank correlation at exactly one
while Pearson's linear correlation drops below one.

## 5. Law of large numbers and the central limit theorem

The sample mean of `n` draws has standard error `sigma / sqrt(n)`, so the
error shrinks by a factor of ten for every hundredfold increase in `n`.

```python
from probstat.distributions import Exponential

law = Exponential(rate=2.0)  # mean 0.5, sd 0.5
for n in (100, 10_000):
    draws = law.sample(n, seed=7)
    m = mean(draws)
    print(f"n={n:>6} mean={m:.4f} error={abs(m - 0.5):.4f} se={0.5 / n ** 0.5:.4f}")

means = [mean(law.sample(30, seed=1000 + k)) for k in range(2000)]
inside = sum(abs(m - 0.5) <= 1.96 * 0.5 / 30 ** 0.5 for m in means) / len(means)
print(f"fraction of 30-sample means within 1.96 se: {inside:.3f}")
```

```text
n=   100 mean=0.4508 error=0.0492 se=0.0500
n= 10000 mean=0.4981 error=0.0019 se=0.0050
fraction of 30-sample means within 1.96 se: 0.963
```

Even for a skewed exponential law, means of 30 draws fall within the normal
95 % band close to 95 % of the time (here 96.3 % of 2000 repetitions).

## 6. From statistics to inference

```python
from probstat.inference import one_sample_t_test, student_t_mean_confidence_interval

test = one_sample_t_test(data, null_mean=3.0)
print(f"t={test.statistic:.4f} df={test.df:.0f} p={test.p_value:.4f}")
interval = student_t_mean_confidence_interval(data, confidence=0.95)
print(f"[{interval.lower:.4f}, {interval.upper:.4f}] {interval.method}")
```

```text
t=1.7346 df=9 p=0.1168
[2.7719, 4.7281] Student-t interval for a normal mean
```

The 95 % interval contains `3.0` exactly when the two-sided p-value exceeds
`0.05`. The [hypothesis testing and bootstrap](../foundations/hypothesis-testing-and-bootstrap.md)
guide continues from here.

Previous: [multivariable calculus and conditioning](multivariable-calculus-and-conditioning.md).
Next: [R bridge](r-bridge.md).
