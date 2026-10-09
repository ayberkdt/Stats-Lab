# Waiting Counts, Tails, Quantiles, and Continuous Foundations

This chapter connects two ways to describe waiting: a discrete number of failed
trials and a continuous amount of elapsed time. It also introduces three
operations needed throughout inference and survival analysis: survival
probability, quantiles, and hazard.

## State the parameterization before using a name

“Geometric” and “negative binomial” do not have universal support conventions.
This library uses:

- `Geometric(p)`: failures before the first success, support `{0,1,...}`;
- `NegativeBinomial(r,p)`: failures before the `r`-th success, support
  `{0,1,...}`.

Under a trials-until-success convention, both variables would be shifted. The
class docstrings, equation records, plot labels, and tests repeat the chosen
convention so a correct formula cannot be paired with the wrong interpretation.

## Geometric waiting counts

For independent Bernoulli trials with success probability `p`, exactly `k`
failures before the first success has probability

`P(X=k)=p(1-p)^k`. See [EQ-DIST-GEOM-001].

The direct upper tail is especially simple:

`P(X>k)=(1-p)^(k+1)`. See [EQ-DIST-GEOM-002].

The implementation uses `log1p(-p)` and `expm1` to preserve small differences
near zero. `p=1` is treated as an exact degenerate law at zero.

```python
from probstat.distributions import Geometric

attempts = Geometric(p=0.2)
print(attempts.pmf(3))       # three failures, then success
print(attempts.sf(10))       # more than ten failures
print(attempts.quantile(0.9))
```

## Negative binomial waiting counts

For a positive integer target `r`, the probability of `k` failures before the
`r`-th success is

`P(X=k)=C(k+r-1,k) p^r (1-p)^k`. See [EQ-DIST-NBIN-001].

Setting `r=1` recovers the library's Geometric distribution. Means and variances
also add because the law is the sum of `r` independent geometric waiting
counts.

The CDF and survival function are regularized incomplete-beta functions:

`F(k)=I_p(r,k+1)` and `S(k)=I_(1-p)(k+1,r)`.

The two tails are evaluated directly rather than forming `1-F` when `F` is near
one. The shared incomplete-beta implementation follows [EQ-NUM-BETA-001], uses
a continued fraction, and selects its symmetry branch according to the DLMF
convergence boundary.

```python
from probstat.distributions import NegativeBinomial

failures = NegativeBinomial(r=5, p=0.4)
print(failures.mean)
print(failures.cdf(6), failures.sf(6))
```

## Survival is not merely `1-CDF` in floating-point arithmetic

Mathematically, `S(x)=P(X>x)=1-F(x)`. Numerically, subtracting a CDF such as
`0.9999999999999999` can lose most useful digits in the tail. A direct `sf`
method is therefore a separate narrow protocol, not a convenience alias.

This release provides direct survival functions for Bernoulli, Binomial,
Geometric, Negative Binomial, Poisson, Normal, Uniform, and Exponential laws.
The algorithms differ: finite sums, powers, incomplete beta/gamma functions,
`erfc`, or a closed-form exponential tail.

## Quantiles are generalized inverses

The library defines

`Q(u)=inf{x: F(x)>=u}` for `0<=u<=1`.

For discrete laws, quantiles return supported count values even though the
method's common return type is `float`; positive infinity represents the
`u=1` endpoint of non-degenerate distributions with unbounded support. Finite
support uses binary search, unbounded count support first brackets the target,
and closed-form laws invert directly.

Endpoint conventions are explicit:

- finite support returns its lower or upper endpoint;
- Normal returns negative/positive infinity at zero/one;
- positive unbounded laws return positive infinity at one;
- degenerate laws return their sole supported value.

## Continuous Uniform

`Uniform(lower,upper)` has constant density `1/(upper-lower)` on its finite
interval. Its CDF, survival, and quantile functions are linear. Its hazard

`h(x)=1/(upper-x)`

increases toward the fixed upper endpoint and is undefined after survival has
reached zero. The implementation raises instead of silently returning a
meaningless `0/0` value.

## Exponential waiting time

`Exponential(rate)` uses a rate, not a scale, parameter:

`f(x)=rate exp(-rate x)`, `S(x)=exp(-rate x)`, and `E[X]=1/rate`.

The constant hazard `h(x)=rate` expresses memorylessness. The CDF uses `expm1`
for accuracy near zero; the quantile and sampler use `log1p`:

`Q(u)=-log(1-u)/rate`. See [EQ-DIST-EXP-001] through
[EQ-DIST-EXP-006].

```python
from probstat.distributions import Exponential

waiting_time = Exponential(rate=2.0)
print(waiting_time.sf(1.5))
print(waiting_time.hazard(1.5))
print(waiting_time.quantile(0.95))
```

## Reading the paired visualizations

The waiting-count plot compares Geometric and Negative Binomial PMFs under the
same zero-based convention. Its finite horizontal window does not truncate the
laws mathematically.

The continuous-functions plot uses separate vertical axes: density on the left,
probability on the right. A density may exceed one, while CDF and survival
remain in `[0,1]`; placing all three on a falsely shared “probability” axis would
be easier to draw but conceptually wrong.

## Sources and next steps

Core distribution identities follow [REF-ROSS-2019] and
[REF-CASELLA-BERGER-2002]. The regularized beta definition, symmetry, binomial
identity, and continued fraction follow [REF-NIST-DLMF-BETA]. Hazard conventions
follow [REF-KLEIN-MOESCHBERGER-2003].

The next family should build Gamma and Beta laws on this special-function
foundation, then derive Chi-square and Student-t relationships. Stable inverse
special functions, explicit error budgets, and reference-backend comparisons
remain required before those APIs can be considered mature.
