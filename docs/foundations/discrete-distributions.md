# Discrete Distribution Foundations

A discrete distribution assigns probability to separate outcomes. This sounds
simple, but a useful implementation must keep four ideas distinct: parameters,
support, probability mass, and numerical representation.

## The common contract

For a discrete random variable `X`, the probability mass function (PMF) is

`p_X(x) = P(X=x)`, with `p_X(x) >= 0` and `sum_x p_X(x) = 1`.

`ProbabilityMassFunction` captures only this operation. The larger
`DiscreteProbabilityLaw` also requires a CDF, finite mean, and finite variance.
This separation is intentional: arbitrary category labels have probability
mass, but no canonical order for a CDF and no canonical numeric encoding for
moments.

All implemented discrete families also expose `log_pmf`. Log probabilities turn
products into sums and retain information when a tiny probability would
underflow to zero. Outside-support and zero-mass points use `-inf`, matching
`log(0)` by limit.

## Support is mathematical metadata

`IntegerSupport(lower, upper)` represents an inclusive bounded or upper-
unbounded integer support. `FiniteSupport(values)` preserves the declared order
of distinct labels for deterministic displays without claiming that the labels
have mathematical order. `RealSupport` provides the corresponding interval
description for continuous laws.

Support objects answer membership questions; parameter constraints validate
the model itself. Keeping these roles separate prevents a valid observation
range from being confused with valid parameter values.

## Binomial distribution

`Binomial(n, p)` models the number of successes in `n` independent trials with
common success probability `p`:

`P(X=k) = C(n,k) p^k (1-p)^(n-k)`. See [EQ-DIST-BIN-001].

Its support is `{0,...,n}`, mean is `np`, and variance is `np(1-p)`. The code
computes the binomial coefficient in log space through `lgamma`, uses `log1p`
for `log(1-p)`, and treats `p=0` and `p=1` as exact limiting cases. For
`n <= 256` the CDF sums the lower tail below the mean and complements the upper
tail above it. For larger `n` it uses the regularized incomplete beta function,
`F(k) = I_(1-p)(n - k, k + 1)` [EQ-NUM-BETA-001], which selects the smaller tail
before any complement and needs a bounded number of continued-fraction terms
instead of a sum of `O(n)` masses. Both are more stable than blindly forming
factorials and powers.

```python
from probstat.distributions import Binomial

defects = Binomial(n=20, p=0.03)
print(defects.pmf(0))
print(defects.cdf(1))
```

## Categorical distribution

`Categorical({label: probability, ...})` models one draw from a finite set of
named outcomes. Probabilities must lie on the unit simplex. See
[EQ-DIST-CAT-001].

The law exposes event probability, all tied modes, entropy, and sampling. It
deliberately does not expose a CDF, mean, or variance. Assigning `red=0` and
`blue=1` would create a user-specific encoding, not a property of colors.

```python
from probstat.distributions import Categorical

weather = Categorical({"sun": 0.60, "cloud": 0.25, "rain": 0.15})
print(weather.probability_of(("cloud", "rain")))
print(weather.entropy)  # nats; see EQ-DIST-CAT-002
```

## Poisson distribution

`Poisson(lam)` models a non-negative count under the constant-rate Poisson
assumptions:

`P(X=k) = exp(-lam) lam^k / k!`. See [EQ-DIST-POIS-001].

Its support is unbounded, and both its mean and variance equal `lam`. That
equality is a model implication: overdispersion or underdispersion in observed
data is evidence to inspect, not something the implementation should erase.

The CDF and survival function use complementary regularized incomplete-gamma
functions [EQ-DIST-POIS-002; EQ-NUM-GAMMA-001], so a tiny tail is computed
directly rather than by subtracting two nearly equal numbers. Sampling uses
Knuth's exact product algorithm for small rates and Hörmann's transformed-
rejection method for larger rates.

```python
from probstat.distributions import Poisson

arrivals = Poisson(lam=4.0)
print(arrivals.cdf(2))
print(arrivals.sf(8))
```

## Reading the plots correctly

`plot_integer_pmf` requires an explicit display window. For an unbounded law,
bars outside that window still have probability mass. `plot_categorical_pmf`
uses positions only for display and labels the axis as categories, not numeric
values. These constraints make the visuals easier to read without changing the
mathematics they represent.

## Common pitfalls

- **Masses need integer arguments; CDFs do not.** `pmf` and `log_pmf` accept
  an integer or an integral float such as `2.0`, and raise `TypeError` for
  `2.5`. `cdf(2.7)` is `cdf(2)`, because the CDF is a right-continuous step
  function of a real argument.
- **Outside the support the mass is zero, not an error.** `pmf` returns
  `0.0` and `log_pmf` returns `-inf` there, also for a `Categorical` label
  that was never declared, so a misspelled label passes silently.
- **`Categorical` has no CDF, mean, or variance.** Map the labels to
  numbers explicitly, for example with a `probability.FiniteRandomVariable`,
  when a numeric summary is meant.
- **Equal mean and variance is a Poisson assumption.** `Poisson(lam)` has
  `mean == variance == lam`; check observed counts for over- or
  underdispersion instead of assuming it.

```python
def raises(call):
    try:
        call()
    except (TypeError, ValueError) as error:
        return type(error).__name__
    return "no error"


five = Binomial(5, 0.5)
five.pmf(2.0), raises(lambda: five.pmf(2.5))  # (0.3125, 'TypeError')
five.cdf(2.7) == five.cdf(2)  # True
Poisson(2.0).pmf(-1), five.log_pmf(7) == float("-inf")  # (0.0, True)
weather.pmf("snow")  # 0.0
hasattr(weather, "cdf"), hasattr(weather, "mean")  # (False, False)
arrivals.mean, arrivals.variance  # (4.0, 4.0)
```

## Sources and next steps

Definitions and elementary identities follow [REF-ROSS-2019]. Incomplete-gamma
definitions and expansions follow [REF-NIST-DLMF-GAMMA]. Entropy follows
[REF-COVER-THOMAS-2006], and transformed-rejection sampling follows
[REF-HORMANN-1993]. Full records are in the equation registry and bibliography.

Geometric and negative binomial waiting counts, survival functions,
quantiles, and hazards continue in
[waiting counts and continuous foundations](waiting-and-continuous-distributions.md);
the continuous families are listed in the
[continuous distribution catalog](continuous-distribution-catalog.md).
