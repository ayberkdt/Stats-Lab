# Multivariate Laws and Conjugate Updates

This guide covers the vector- and matrix-valued laws in
`probstat.multivariate` beyond the multivariate normal, and the two
conjugate families built on them:

- `Dirichlet` on the probability simplex;
- `Multinomial` and `DirichletMultinomial` for category counts;
- `MultivariateStudentT`;
- `Wishart` and `InverseWishart` on positive-definite matrices;
- `dirichlet_posterior` (categorical or multinomial data) and
  `normal_inverse_wishart_posterior` (multivariate normal data with unknown
  mean and covariance).

Every number below was printed by the library with the inputs shown. The
design decisions are recorded in ADR-0067.

## 1. The Dirichlet law

`X ~ Dirichlet(alpha)`, `alpha = (alpha_1, ..., alpha_K)`, `K >= 2`, lives
on the simplex `{x : x_i >= 0, sum_i x_i = 1}`. Its density with respect to
Lebesgue measure on the first `K - 1` coordinates is
`f(x) = prod_i x_i^(alpha_i - 1) / B(alpha)` with
`log B(alpha) = sum_i log Gamma(alpha_i) - log Gamma(alpha_0)` and
`alpha_0 = sum_i alpha_i` ([EQ-DIST-DIR-001]; [REF-BISHOP-2006, Appendix B]).
`log_pdf` rejects a vector that is not on the hyperplane `sum x_i = 1`
(tolerance `1e-10`) and returns `-inf` for a negative component.

- mean `m = alpha / alpha_0` and covariance
  `(diag(m) - m m') / (alpha_0 + 1)` ([EQ-DIST-DIR-002]);
- mode `(alpha_i - 1) / (alpha_0 - K)` when every `alpha_i > 1`, otherwise
  `nan` ([EQ-DIST-DIR-003]);
- entropy `log B(alpha) + (alpha_0 - K) psi(alpha_0)
  - sum_i (alpha_i - 1) psi(alpha_i)` ([EQ-DIST-DIR-004]);
- marginal `X_i ~ Beta(alpha_i, alpha_0 - alpha_i)` ([EQ-DIST-DIR-005]).

Merging categories keeps the family: sums of coordinates over a partition
are Dirichlet with the summed concentrations (`aggregate`, [EQ-DIST-DIR-005];
[REF-KOTZ-BALAKRISHNAN-JOHNSON-2000, Chapter 49]). Samples normalize
independent `Gamma(alpha_i, 1)` variates; the variates are formed in log
space so that concentrations such as `1e-3` still give points of the simplex
([EQ-DIST-DIR-006]).

## 2. Counts: multinomial and Dirichlet-multinomial

`Multinomial(n, p)` has mass `n!/prod x_i! prod p_i^(x_i)` on the count
vectors with total `n` ([EQ-DIST-MULT-001];
[REF-JOHNSON-KOTZ-BALAKRISHNAN-1997, Chapter 35]). For `n <= 1000` the
coefficient is an exact integer, so only one logarithm is rounded. Its mean
is `n p`, its covariance `n (diag(p) - p p')` ([EQ-DIST-MULT-002]), and each
count is `Binomial(n, p_i)` ([EQ-DIST-MULT-003]). Sampling uses sequential
conditional binomials ([EQ-DIST-MULT-004]).

`DirichletMultinomial(n, alpha)` integrates the multinomial over
`p ~ Dirichlet(alpha)` ([EQ-DIST-DIRMULT-001]; [REF-MOSIMANN-1962]):
`P(x) = n!/prod x_i! * Gamma(A)/Gamma(n + A) * prod Gamma(x_i + alpha_i)/Gamma(alpha_i)`,
`A = sum alpha_i`. Its mean is `n alpha / A` and its covariance is the
multinomial covariance inflated by `(n + A)/(1 + A)` (`overdispersion`,
[EQ-DIST-DIRMULT-002]). As `A -> inf` with `alpha / A` fixed it tends to the
multinomial; the gamma ratios use a Stirling-series difference
([EQ-NUM-STIRLING-001]) so this limit is reached without loss of digits.

## 3. The multivariate Student t

`MultivariateStudentT(location, scale_matrix, dof)` is
`X = mu + Z / sqrt(W / nu)` with `Z ~ N(0, Sigma)` and `W ~ chi^2(nu)`
([EQ-DIST-MVT-004]). Its log-density is
`log Gamma((nu+p)/2) - log Gamma(nu/2) - (p/2) log(nu pi) - (1/2) log|Sigma|
- ((nu+p)/2) log(1 + delta/nu)` with the squared Mahalanobis distance
`delta` ([EQ-DIST-MVT-001]; [REF-KOTZ-NADARAJAH-2004, Chapter 1]). The mean
is `mu` for `nu > 1` and the covariance `nu/(nu - 2) Sigma` for `nu > 2`
([EQ-DIST-MVT-002]); `Sigma` is a scale, not a covariance. Marginals of any
index subset keep `nu` ([EQ-DIST-MVT-003]).

## 4. Wishart and inverse Wishart

`Wishart(S, nu)` (real `nu > p - 1`) has log-density
`((nu-p-1)/2) log|X| - tr(S^-1 X)/2 - (nu p/2) log 2 - (nu/2) log|S|
- log Gamma_p(nu/2)` ([EQ-DIST-WISH-001]; [REF-ANDERSON-2003, Section 7.2]),
where `log Gamma_p` is the multivariate log-gamma function
(`multivariate_log_gamma`, [EQ-NUM-MVLGAMMA-001];
[REF-NIST-DLMF-MATRIX-ARGUMENT, Section 35.3]). The mean is `nu S` and the
mode `(nu - p - 1) S` ([EQ-DIST-WISH-002]). Draws use the Bartlett
decomposition `W = (L A)(L A)'` ([EQ-DIST-WISH-003]; [REF-SMITH-HOCKING-1972]).

`InverseWishart(Psi, nu)` is the law of `Sigma` with
`Sigma^-1 ~ Wishart(Psi^-1, nu)` ([EQ-DIST-IWISH-001];
[REF-ANDERSON-2003, Section 7.7]). Its mean is `Psi/(nu - p - 1)` for
`nu > p + 1` and its mode `Psi/(nu + p + 1)` ([EQ-DIST-IWISH-002]). Draws
invert the Bartlett factor, a triangular inverse only ([EQ-DIST-IWISH-003]).

## 5. Dirichlet-categorical updating

With prior `Dirichlet(alpha)` and observed counts `x` the posterior is
`Dirichlet(alpha + x)` ([EQ-CONJ-DIR-001]; [REF-GELMAN-ETAL-2013,
Section 3.4]). The next category has posterior predictive probability
`(alpha_i + x_i)/(A + n)` and the counts of `m` future trials are
`DirichletMultinomial(m, alpha + x)` ([EQ-CONJ-DIR-002]).

```python
from probstat.multivariate import Dirichlet, dirichlet_posterior

result = dirichlet_posterior(Dirichlet((1.0, 1.0, 1.0)), (12, 5, 3))
result.posterior.alpha             # (13.0, 6.0, 4.0)
result.predictive_probabilities    # (0.5652, 0.2609, 0.1739) to 4 decimals
result.log_marginal_likelihood     # -21.2116 (this ordered sequence)
result.predictive(10).overdispersion  # 1.375
```

The log marginal likelihood refers to one ordered sequence with these
counts; adding `log(n!/prod x_i!)` gives the Dirichlet-multinomial mass of
the unordered counts.

## 6. Normal-inverse-Wishart updating

`NormalInverseWishart(m, kappa, Psi, nu)` is the prior
`Sigma ~ InverseWishart(Psi, nu)`, `mu | Sigma ~ N(m, Sigma/kappa)`
([EQ-CONJ-NIW-001]; [REF-MURPHY-2007, Section "Normal-inverse-Wishart prior"]).
For `n` observations with mean `xbar` and scatter
`S = sum (x_i - xbar)(x_i - xbar)'` ([EQ-CONJ-NIW-002];
[REF-GELMAN-ETAL-2013, Section 3.6]):

```text
kappa_n = kappa_0 + n        nu_n = nu_0 + n
m_n     = (kappa_0 m_0 + n xbar) / kappa_n
Psi_n   = Psi_0 + S + (kappa_0 n / kappa_n) (xbar - m_0)(xbar - m_0)'
```

Integrating out `Sigma` gives Student-t laws with `nu - p + 1` degrees of
freedom ([EQ-CONJ-NIW-003]): `mu ~ t(m, Psi/(kappa (nu - p + 1)))`, and one
new observation `x ~ t(m, Psi (kappa + 1)/(kappa (nu - p + 1)))`. The log
marginal likelihood is
`-(n p/2) log pi + log Gamma_p(nu_n/2) - log Gamma_p(nu_0/2)
+ (nu_0/2) log|Psi_0| - (nu_n/2) log|Psi_n| + (p/2) log(kappa_0/kappa_n)`
([EQ-CONJ-NIW-004]; [REF-MURPHY-2012, Section 4.6.3]).

```python
from probstat.multivariate import NormalInverseWishart, normal_inverse_wishart_posterior

prior = NormalInverseWishart((0.0, 0.0), 1.0, ((1.0, 0.0), (0.0, 1.0)), 4.0)
data = ((1.0, -0.5), (0.2, -1.4), (1.6, 0.1), (0.9, -0.8))
post = normal_inverse_wishart_posterior(prior, data)
post.posterior.location        # (0.74, -0.52); kappa_n = 5, nu_n = 8
post.posterior.scale           # ((2.672, 0.584), (0.584, 2.508))
post.log_marginal_likelihood   # -10.4752
post.predictive.dof            # 7.0
post.predictive.covariance     # ((0.6413, 0.1402), (0.1402, 0.6019)) to 4 decimals
```

Two checks that the tests run: for `n = 1` the log marginal likelihood
equals the prior predictive log-density at the observation, and for any `n`
it equals the sum of the sequential one-step predictive log-densities.

`inference.NormalInverseWishartPrior` (missing-data augmentation, ADR-0047)
holds hyperparameters for a sampler and allows an improper default;
`multivariate.NormalInverseWishart` is a proper law with a density and a
closed-form update.
