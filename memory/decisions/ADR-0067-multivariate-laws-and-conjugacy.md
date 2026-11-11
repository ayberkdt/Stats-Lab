# ADR-0067: Multivariate probability laws and conjugate updates

- Status: Accepted
- Date: 2026-11-11

## Context

`probstat.multivariate` offered only the multivariate normal and change-of-
variable tools. The Phase 2 roadmap lists the Dirichlet and multinomial laws
as missing, and Bayesian work in higher layers needs the standard conjugate
families for categorical data and for a normal vector with unknown mean and
covariance. `inference.missing` already draws inverse-Wishart matrices and
normal-inverse-Wishart posteriors for data augmentation (ADR-0047,
[EQ-MISS-005], [EQ-MISS-006]), but only as private steps of one algorithm and
in a sibling package of the same layer, so it cannot be the shared home.

## Decision

1. **New laws in layer 2.** `Dirichlet`, `Multinomial`,
   `DirichletMultinomial`, `MultivariateStudentT`, `Wishart` and
   `InverseWishart` are frozen dataclasses in `probstat.multivariate`
   (modules `dirichlet`, `multinomial`, `student_t`, `wishart`). They import
   only `foundations`, `linalg`, `numerics` and `distributions` (layers 0-1);
   nothing in `multivariate` imports `inference`. Method names follow
   `MultivariateNormal`: `log_pdf`/`pdf` for densities, `log_pmf`/`pmf` for
   mass functions, `sample(size, *, seed)` through `resolve_stream`
   (ADR-0059 seed contract), moments as properties.
2. **Supports and conventions.** A Dirichlet point must lie on the hyperplane
   `sum x_i = 1` (tolerance `1e-10`, otherwise `ValueError`); a negative
   component is outside the support (`-inf`); on a face the limiting value is
   returned, and `nan` when zero and infinite factors meet. Count vectors
   accept integers and integral floats, as univariate discrete laws do; a
   wrong total or a negative count is `-inf`. A symmetric matrix that is not
   positive definite has Wishart and inverse-Wishart log-density `-inf`; a
   non-symmetric or misshapen matrix raises. Undefined moments follow
   ADR-0059: the Dirichlet mode with some `alpha_i <= 1` and the Wishart mode
   with `nu < p + 1` are `nan`; the Student-t mean is `nan` for `nu <= 1`;
   divergent second moments (Student t with `1 < nu <= 2`, inverse-Wishart
   mean with `nu <= p + 1`) have `inf` diagonal and `nan` off-diagonal
   entries. Properties never raise.
3. **Numerics.** Cholesky factors, triangular solves and log-determinants
   come from `linalg`; `tr(A^-1 B)` is formed column by column from the
   factor and no full inverse is computed. The multinomial coefficient is the
   exact integer `prod C(.,.)` up to `n = 1000` and the log-gamma form above.
   Gamma ratios in the Dirichlet-multinomial mass, the Dirichlet evidence and
   the Student-t normalizer use `distributions._special.log_gamma_ratio`
   (Stirling-series difference, [EQ-NUM-STIRLING-001]); other layer-1 and
   layer-2 packages already import that module. The multivariate log-gamma
   function is public as `multivariate_log_gamma` ([EQ-NUM-MVLGAMMA-001]).
4. **Sampling.** Dirichlet: normalized gamma variates in log space, with the
   boost `G_a = G_(a+1) U^(1/a)` for `a < 1` ([EQ-DIST-DIR-006]).
   Multinomial: sequential conditional binomials drawn as Bernoulli sums
   (`O(n)` per vector, as `distributions.Binomial.sample`). Student t:
   `mu + L z / sqrt(W/nu)`. Wishart: Bartlett decomposition. Inverse
   Wishart: the inverted Bartlett factor `Sigma = (C A^-T)(C A^-T)'`, the
   construction of [EQ-MISS-005], reimplemented here because `inference` is
   a sibling package. Gamma and chi-square variates come from
   `random.Random.gammavariate`, as elsewhere in the library.
5. **Conjugacy as pure functions.** `dirichlet_posterior(prior, counts)`
   returns a frozen `DirichletPosterior` (posterior Dirichlet, log evidence
   of the observed sequence, `predictive(m)` as `DirichletMultinomial`).
   `NormalInverseWishart(location, kappa, scale, dof)` is the prior law
   (`log_pdf`, `sample`, `covariance_marginal`, `mean_marginal`,
   `predictive`), and `normal_inverse_wishart_posterior(prior, observations)`
   returns a frozen `NormalInverseWishartPosterior` with the posterior,
   sufficient statistics, log marginal likelihood and the posterior
   predictive `MultivariateStudentT`. The prior must be proper
   (`kappa > 0`, `nu > p - 1`); the improper default of
   `inference.NormalInverseWishartPrior` is not offered here.

## Consequences

- Positive: categorical and multivariate-normal Bayesian updates have closed
  forms with evidence and predictive laws; the Dirichlet-multinomial gives an
  overdispersed count model; the Wishart family is available as laws, not
  only as a sampler.
- Two normal-inverse-Wishart representations now exist:
  `inference.NormalInverseWishartPrior` (hyperparameters for data
  augmentation, improper default allowed) and
  `multivariate.NormalInverseWishart` (a proper law with density and
  conjugate update). Unifying them would require `inference` to import
  `multivariate`, which the layer map allows; it is left to a later change
  so that `inference.missing` results do not move.
- Limitations: the Dirichlet log-density and entropy subtract `lgamma`
  values, so for concentrations near `1e9` the absolute error is about
  `1e-6`; multinomial sampling is `O(n)` per vector (about 0.37 s for 1000
  vectors with `n = 1000` in pure Python); Wishart variances, conditional
  Student-t laws and the matrix-normal family are not implemented.

## Verification

`tests/test_multivariate_laws.py` (44 tests) uses oracles independent of the
code under test:

- reductions: two-category Dirichlet against `distributions.Beta` (density,
  entropy), two-category multinomial against `Binomial`, `p = 1` Student t
  against `StudentT`, `p = 1` Wishart against `Gamma(nu/2, 1/(2 s))`,
  inverse Wishart against the inverted Wishart with the `|X|^-(p+1)`
  Jacobian, `p = 1` normal-inverse-Wishart predictive against `StudentT`;
- exact arithmetic: multinomial masses for every composition of `n = 9`
  against `fractions.Fraction` (relative `4e-15`, exact rational total 1);
  multinomial moments by exact enumeration; Dirichlet-multinomial with
  integer `alpha` against the Polya urn in rationals (exact total 1);
  Dirichlet evidence against the sequential chain rule in rationals;
  normal-inverse-Wishart posterior scale against the raw-moment route in
  rationals;
- quadrature: Dirichlet density integrates to 1 over the 2-simplex (nested
  adaptive quadrature, `1e-8`); Dirichlet-multinomial masses against the
  compound integral for `K = 2` (all `k`, `1e-9`) and `K = 3` (`1e-8`);
- reference values computed with scipy 1.14.1 outside the suite (Dirichlet
  log-density and entropy, multivariate t in three and two dimensions,
  Wishart and inverse Wishart in two and three dimensions, Dirichlet-
  multinomial mass, `multigammaln`), all within `1e-13` relative;
- limits: the Student t with `nu = 1e10` matches the normal log-density to
  `1e-8` and the Dirichlet-multinomial with `alpha = 1e9 p` matches the
  multinomial to `1e-6`. With plain `lgamma` differences these limits failed
  (errors `1.4e-5` and `4.4e-6`); with the Stirling-series ratio the
  differences are `1.4e-10` (`nu = 1e10`) and `2.8e-8` (`alpha_0 = 1e9`),
  the size of the true `O(1/nu)` and `O(1/alpha_0)` model differences;
- sampling: means and variances within four standard errors (variance
  errors from bounded support or the known fourth moments) for Dirichlet,
  multinomial, Dirichlet-multinomial, Student t, Wishart, inverted inverse
  Wishart, and the normal-inverse-Wishart mean and covariance marginals;
  reproducibility for integer and `RandomStream` seeds; finite simplex points
  for `alpha = 1e-3`;
- chain rule: the normal-inverse-Wishart evidence for `n = 1` equals the
  prior predictive Student-t density, and for `n = 4` the sum of sequential
  predictive log-densities (`1e-12`); sequential and batch posteriors agree;
- invalid inputs and boundaries for every constructor and argument check.
