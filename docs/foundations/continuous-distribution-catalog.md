# Continuous Distribution Catalog for UQ and Reliability

Input uncertainty in a UQ study is only as credible as the marginal laws
chosen for it. This guide lists the continuous families in
`probstat.distributions`, states each parameterization exactly, describes how
fast each tail decays, and says when each family is a sensible choice. Every
family follows one contract: frozen dataclass, validated parameters,
`support`, `log_pdf`, `pdf`, `cdf`, `sf`, `quantile`, `isf`, `mean`,
`variance`, and `sample(size, *, seed)` with a call-local generator. All
numbers below come from the library with the stated arguments.

## Parameterizations

Names hide incompatible conventions, so check this table before entering
values taken from a paper, a standard, or another library.

| Class | Parameters | Support | Mean | Variance |
|---|---|---|---|---|
| `Normal(mu, sigma)` | location, scale | real line | `mu` | `sigma^2` |
| `LogNormal(mu, sigma)` | moments of `log X` | `x > 0` | `exp(mu + sigma^2/2)` | see [EQ-DIST-LN-004] |
| `Uniform(lower, upper)` | endpoints | `[a, b]` | `(a+b)/2` | `(b-a)^2/12` |
| `Triangular(lower, mode, upper)` | endpoints and mode | `[a, b]` | `(a+b+c)/3` | [EQ-DIST-TRI-004] |
| `TruncatedNormal(mu, sigma, lower, upper)` | parent normal plus bounds (`None` = open side) | `[l, u]` | [EQ-DIST-TN-003] | [EQ-DIST-TN-003] |
| `Beta(alpha, beta)` | two shapes | `[0, 1]` | `a/(a+b)` | [EQ-DIST-BETA-003] |
| `Exponential(rate)` | rate `lambda` | `x >= 0` | `1/lambda` | `1/lambda^2` |
| `Gamma(shape, rate)` | shape `k`, rate `lambda` (`Gamma.from_scale` for `theta`) | `x >= 0` | `k/lambda` | `k/lambda^2` |
| `Weibull(shape, scale)` | shape `k`, scale `lambda` | `x >= 0` | [EQ-DIST-WEIB-004] | [EQ-DIST-WEIB-005] |
| `ChiSquare(nu)` | degrees of freedom | `x >= 0` | `nu` | `2 nu` |
| `FisherF(nu1, nu2)` | numerator, denominator df | `x >= 0` | `nu2/(nu2-2)` | [EQ-DIST-F-003] |
| `StudentT(nu, mu, sigma)` | df, location, scale | real line | `mu` (`nu > 1`) | `sigma^2 nu/(nu-2)` |
| `Logistic(mu, s)` | location, scale | real line | `mu` | `s^2 pi^2/3` |
| `Gumbel(mu, beta, kind)` | location, scale, `"max"` or `"min"` | real line | `mu +/- beta gamma_E` | `pi^2 beta^2/6` |
| `GeneralizedExtremeValue(mu, sigma, xi)` | location, scale, shape | `1 + xi z > 0` | [EQ-DIST-GEV-003] | [EQ-DIST-GEV-003] |

Conventions to watch:

- `Gamma` uses the **rate**. A source quoting `Gamma(k, theta = 2)` means
  `Gamma(k, 0.5)` here, or `Gamma.from_scale(k, 2.0)`.
- `GeneralizedExtremeValue` uses the hydrology/climate sign of `xi`:
  `xi > 0` is the heavy-tailed Frechet type. Some libraries pass `c = -xi`.
- `Gumbel(kind="min")` models smallest values (strength, minimum
  temperature); it is the mirror image of the default maximum law.
- Missing moments: `StudentT.mean` and `.variance` raise `ValueError` for
  `nu <= 1` (undefined); variances that diverge (`1 < nu <= 2`, `FisherF`
  with `nu2 <= 4`, GEV with `xi >= 1/2`) return `inf`.

## Tail behaviour

Tail weight usually decides a reliability result more than the bulk of the
law does. Four laws with mean 0 and variance 1 give these probabilities of
exceeding 5 (`sf(5.0)`):

| Law (mean 0, variance 1) | `P(X > 5)` | Tail class |
|---|---|---|
| `Normal(0, 1)` | `2.87e-7` | Gaussian, `exp(-x^2/2)` |
| `Logistic(0, sqrt(3)/pi)` | `1.15e-4` | exponential |
| `Gumbel(-gamma_E sqrt(6)/pi, sqrt(6)/pi)` | `9.21e-4` | exponential (upper) |
| `StudentT(5, 0, sqrt(3/5))` | `6.64e-4` | power law, `x^-5` |

Matching the first two moments does not determine the tail: the
probabilities above span more than three orders of magnitude. The t law and
the Frechet GEV (`xi > 0`) have power-law tails, and moments of order
`>= nu` (or `>= 1/xi`) do not exist. The reversed-Weibull GEV (`xi < 0`),
Beta, Triangular, Uniform, and two-sided TruncatedNormal laws are bounded.

### Numerical tail accuracy

Every family evaluates `sf` directly rather than as `1 - cdf`, so upper tails
far below `1e-16` stay accurate. Families without closed-form inverses
(`Gamma`, `Beta`, `StudentT`, `ChiSquare`, `FisherF`, `TruncatedNormal`)
invert whichever tail carries the smaller probability, in log space, on a
coordinate (`log x`, `logit x`, `asinh z`) that flattens the tail
[EQ-DIST-INV-001]. For example, `StudentT(3).isf(1e-6) = 103.30`, whereas the
normal value is `4.75`. `TruncatedNormal` works with log masses, Mills-ratio
continued fractions [EQ-DIST-TN-004], and the distance to a truncation point
[EQ-DIST-TN-005], so `TruncatedNormal(0, 1, lower=40)` is well defined.

Two limits remain:

- Near a **finite support end**, a probability can only be as precise as the
  spacing of floats next to that end allows. A quantile `1e-20` away from
  `upper = 3.0` cannot be represented.
- Incomplete gamma and beta prefactors are formed from `lgamma` differences.
  For shapes or degrees of freedom far above about `1e4`, their relative
  accuracy degrades roughly in proportion to the shape.

## When to use which family in UQ

- **Physically positive, skewed quantities** (loads, permeabilities,
  concentrations): `LogNormal` when the uncertainty is multiplicative;
  `Gamma` for sums of exponential-type contributions and as the conjugate law
  for rates; `Weibull` for strengths and lifetimes (weakest-link arguments).
- **Fractions, probabilities, efficiencies**: `Beta`, or a `TruncatedNormal`
  on `[0, 1]` when the evidence is a mean and standard deviation.
- **Expert judgement with bounds and a most likely value**: `Triangular`. It
  is simple but has thin tails by construction; prefer `Beta` when the shape
  matters.
- **Measured quantities that cannot be negative but are nearly normal**:
  `TruncatedNormal(lower=0)`. For example, `TruncatedNormal(10, 4, lower=0)`
  has mean 10.07, standard deviation 3.91, and `P(X <= 0.5) = 0.00258`.
- **Extremes** (annual maximum wind, flood, or load): `Gumbel` as the light
  tail model, `GeneralizedExtremeValue` when the tail shape must be
  estimated. The 100-year return level `isf(0.01)` is 53.0 for
  `Gumbel(30, 5)` and 59.2 for `GeneralizedExtremeValue(30, 5, 0.1)`, so a
  modest positive `xi` changes the design value by 12%.
- **Robust or heavy-tailed errors**: `StudentT` with small `nu`, `Logistic`
  for a moderately heavier tail than the normal.
- **Test statistics and variance ratios**: `ChiSquare`, `FisherF`, `StudentT`.

All families can be marginals of `modeling.RandomInput` inside a
`modeling.JointInput` (the Nataf/Gaussian-copula map uses `isf` and `sf` for
upper-tail accuracy). The exception is a law whose mean is undefined
(`StudentT` with `nu <= 1`): `RandomInput` requires `mean`, and that property
raises for such a law.

## Fitting gamma and Weibull laws

`inference.fit_gamma_mle` and `inference.fit_weibull_mle` profile out the
rate or scale and solve the remaining one-dimensional shape equation by
safeguarded Newton iteration [EQ-INF-MLE-GAM-001], [EQ-INF-MLE-WEIB-001].
Both equations are monotone in the shape, so the root is unique.

```python
from probstat.distributions import Gamma, Weibull
from probstat.inference import fit_gamma_mle, fit_weibull_mle

fit_gamma_mle(Gamma(2.5, 0.5).sample(2000, seed=1))
# Gamma(shape=2.5038, rate=0.4996)
fit_weibull_mle(Weibull(1.8, 120.0).sample(2000, seed=1))
# Weibull(shape=1.8019, scale=120.38)
```

The Weibull fit centres the logarithms and uses a log-sum-exp shift, so
rescaling the data by `1e6` rescales the fitted scale by exactly that factor.
A maximum-likelihood point estimate says nothing about its own uncertainty.
In a UQ study, report the fitting uncertainty separately (for example by
bootstrap or a Bayesian posterior).

## Sources

[REF-JKB-1994], [REF-JKB-1995], [REF-NIST-DISTRIBUTIONS], [REF-COLES-2001],
[REF-KOTZ-VANDORP-2004], [REF-MARSAGLIA-TSANG-2000], [REF-NIST-DLMF-GAMMA],
[REF-NIST-DLMF-BETA], [REF-NIST-DLMF], [REF-BRENT-1973].
