# Equation Registry

This file is the authoritative catalog of equations currently evaluated by
code. Symbols are local to each record. Source keys resolve in
`docs/references.md`.

## Probability and counting

### EQ-PROB-001 — Probability-measure bounds and axioms

`P(A) in [0, 1]`, `P(Omega) = 1`, and for pairwise disjoint `A_i`,
`P(union_i A_i) = sum_i P(A_i)`.

- Domain: `(Omega, F, P)` is a probability space.
- Meaning: valid event probabilities are normalized and countably additive.
- Code: `foundations.validation.require_probability`; distribution parameter
  validation.
- Source: [REF-KOLMOGOROV-1956, Chapter I, Section 2].

### EQ-PROB-002 — Probability of an event in a finite weighted space

`P(A) = sum_{omega in A} p(omega)`, where `p(omega) >= 0` and
`sum_{omega in Omega} p(omega) = 1`.

- Domain: non-empty finite `Omega`, event `A` contained in `Omega`.
- Meaning: event probability is the total mass of its member outcomes.
- Code: `probability.finite.FiniteProbabilityMeasure` and finite-measure plot.
- Source: [REF-KOLMOGOROV-1956, Chapter I], [REF-ROSS-2019, Chapter 2].

### EQ-PROB-003 — Complement rule

`P(A^c) = 1 - P(A)`; in particular, `P(empty set) = 0`.

- Domain: event `A` in a probability space.
- Meaning: exactly one of an event and its complement occurs.
- Code: `probability.finite.FiniteEvent.complement`; verified against finite
  probability measures in tests.
- Source: [REF-ROSS-2019, Chapter 2].

### EQ-PROB-004 — Two-event inclusion-exclusion rule

`P(A union B) = P(A) + P(B) - P(A intersection B)`.

- Domain: events `A` and `B` in the same probability space.
- Meaning: subtract the overlap counted once by each marginal probability.
- Code: `FiniteEvent.union` and `FiniteEvent.intersection`; the probability
  identity is verified in tests.
- Source: [REF-ROSS-2019, Chapter 2].

### EQ-PROB-005 — Conditional probability

`P(A | B) = P(A intersection B) / P(B)`.

- Domain: events `A`, `B` with `P(B) > 0`.
- Meaning: renormalize probability after restricting the sample space to `B`.
- Code: `FiniteProbabilityMeasure.conditional_probability`.
- Source: [REF-ROSS-2019, Chapter 3].

### EQ-PROB-006 — Independence criterion

`A independent B` if and only if `P(A intersection B) = P(A)P(B)`.

- Domain: events `A` and `B` in the same probability space.
- Meaning: knowledge that either event occurred does not alter the probability
  of the other; the product criterion remains valid for zero-probability events.
- Code: `FiniteProbabilityMeasure.are_independent`.
- Source: [REF-ROSS-2019, Chapter 3].

### EQ-PROB-007 — Law of total probability

For a positive-probability partition `{B_i}_{i=1}^m` of `Omega`,
`P(A) = sum_{i=1}^m P(A | B_i) P(B_i)`.

- Domain: finite partition with pairwise-disjoint cells, exhaustive union, and
  `P(B_i) > 0` for every cell.
- Meaning: reconstruct an event probability by conditioning on exhaustive cases.
- Code: `FiniteProbabilityMeasure.total_probability`.
- Source: [REF-ROSS-2019, Chapter 3].

### EQ-PROB-008 — Bayes' rule for two events

`P(H | E) = P(E | H) P(H) / P(E)`.

- Domain: events `H`, `E` with `P(H) > 0` and `P(E) > 0` for this factorized
  implementation.
- Meaning: update a prior probability using evidence likelihood and normalize by
  evidence probability.
- Code: `FiniteProbabilityMeasure.bayes_probability`.
- Source: [REF-ROSS-2019, Chapter 3].

### EQ-PROB-009 — Expectation on a finite probability space

`E[g(X)] = sum_{omega in Omega} g(omega) p(omega)`.

- Domain: non-empty finite `Omega`, normalized mass `p`, and finite real-valued
  function `g`.
- Meaning: probability-weighted average of a quantity attached to outcomes.
- Code: `FiniteProbabilityMeasure.expectation`.
- Source: [REF-ROSS-2019, Chapter 4].

### EQ-COMB-001 — Factorial

`n! = product_{j=1}^n j`, with `0! = 1`.

- Domain: integer `n >= 0`.
- Meaning: orderings of `n` distinct objects.
- Code: `probability.counting.factorial`.
- Source: [REF-ROSS-2019, Section 1.2].

### EQ-COMB-002 — Permutations

`P(n, r) = n! / (n-r)!`.

- Domain: integers `0 <= r <= n`.
- Meaning: ordered selections of `r` distinct objects from `n`.
- Code: `probability.counting.permutations`.
- Source: [REF-ROSS-2019, Section 1.4].

### EQ-COMB-003 — Combinations

`C(n, r) = n! / (r!(n-r)!)`.

- Domain: integers `0 <= r <= n`.
- Meaning: unordered selections of `r` distinct objects from `n`.
- Code: `probability.counting.combinations`.
- Source: [REF-ROSS-2019, Section 1.5].

### EQ-COMB-004 — Multiplication rule

`N = product_{j=1}^m n_j`.

- Domain: finite stage counts `n_j >= 0`, integer `m >= 1`.
- Meaning: count of paths through sequential choice stages when each partial
  path exposes the stated number of next choices.
- Code: `probability.counting.multiplication_rule`.
- Source: [REF-ROSS-2019, Section 1.1].

## Finite random variables and joint laws

### EQ-RV-001 — Induced probability mass function

`p_X(x) = P({omega in Omega : X(omega) = x})`.

- Domain: finite probability space and hashable-valued mapping `X`.
- Meaning: collect the probability of every source outcome mapped to `x`.
- Code: `FiniteRandomVariable`, `preimage`, `pmf`, and `distribution`.
- Source: [REF-ROSS-2019, Chapters 4-5].

### EQ-RV-002 — Transformation of a random variable

`Y(omega) = g(X(omega))`.

- Domain: finite random variable `X` and deterministic hashable-valued `g`.
- Meaning: composition changes values while retaining the underlying probability
  measure and outcome ownership.
- Code: `FiniteRandomVariable.transform`.
- Source: [REF-ROSS-2019, Chapter 5].

### EQ-RV-003 — CDF of a finite real-valued random variable

`F_X(x) = P(X <= x) = sum_{u <= x} p_X(u)`.

- Domain: finite real-valued `X` and finite real evaluation point `x`.
- Meaning: accumulated probability mass through `x`.
- Code: `FiniteRandomVariable.cdf`.
- Source: [REF-ROSS-2019, Chapters 4-5].

### EQ-RV-004 — Expectation of a finite random variable

`E[X] = sum_x x p_X(x) = sum_{omega in Omega} X(omega)p(omega)`.

- Domain: finite real-valued `X`.
- Meaning: probability-weighted center, whether summed over source outcomes or
  induced values.
- Code: `FiniteRandomVariable.expectation`.
- Source: [REF-ROSS-2019, Chapters 4-5].

### EQ-RV-005 — Raw moment

`mu'_k = E[X^k]`.

- Domain: finite real-valued `X`, integer `k >= 0`.
- Meaning: expectation of the `k`th raw power.
- Code: `FiniteRandomVariable.raw_moment`.
- Source: [REF-ROSS-2019, Chapter 4].

### EQ-RV-006 — Central moment

`mu_k = E[(X - E[X])^k]`.

- Domain: finite real-valued `X`, integer `k >= 0`.
- Meaning: moment of deviations from the expected value.
- Code: `FiniteRandomVariable.central_moment`.
- Source: [REF-ROSS-2019, Chapter 4].

### EQ-RV-007 — Variance and standard deviation

`Var(X) = E[(X-E[X])^2]` and `sigma_X = sqrt(Var(X))`.

- Domain: finite real-valued `X`.
- Meaning: squared spread and its original-unit scale.
- Code: `FiniteRandomVariable.variance` and `standard_deviation`.
- Source: [REF-ROSS-2019, Chapter 4].

### EQ-RV-008 — Covariance of two finite random variables

`Cov(X,Y) = E[(X-E[X])(Y-E[Y])]`.

- Domain: finite real-valued `X` and `Y` on the same probability measure.
- Meaning: probability-weighted joint centered movement.
- Code: `FiniteJointDistribution.covariance`.
- Source: [REF-ROSS-2019, Chapter 6].

### EQ-RV-009 — Correlation of two finite random variables

`Corr(X,Y) = Cov(X,Y) / (sigma_X sigma_Y)`.

- Domain: [EQ-RV-008] with both variances strictly positive.
- Meaning: dimensionless linear association.
- Code: `FiniteJointDistribution.correlation`.
- Source: [REF-ROSS-2019, Chapter 6].

### EQ-RV-010 — Joint PMF

`p_{X,Y}(x,y) = P(X=x, Y=y)`.

- Domain: finite random variables `X` and `Y` sharing one measure object.
- Meaning: mass assigned to a pair of simultaneous mapped values.
- Code: `FiniteJointDistribution`, `joint_pmf`, and `distribution`.
- Source: [REF-ROSS-2019, Chapter 6].

### EQ-RV-011 — Marginalization of a joint PMF

`p_X(x) = sum_y p_{X,Y}(x,y)` and `p_Y(y) = sum_x p_{X,Y}(x,y)`.

- Domain: finite joint PMF.
- Meaning: recover either one-variable law by summing out the other value.
- Code: `FiniteJointDistribution.first_marginal` and `second_marginal`.
- Source: [REF-ROSS-2019, Chapter 6].

### EQ-RV-012 — Conditional PMF from a joint PMF

`p_{X|Y}(x|y) = p_{X,Y}(x,y) / p_Y(y)`.

- Domain: finite joint PMF with `p_Y(y) > 0`.
- Meaning: renormalize the joint slice at the observed value `y`.
- Code: `FiniteJointDistribution.conditional_pmf_first_given_second`.
- Source: [REF-ROSS-2019, Chapter 6].

### EQ-RV-013 — Random-variable independence

`X independent Y` if and only if
`p_{X,Y}(x,y) = p_X(x)p_Y(y)` for every pair `(x,y)`.

- Domain: finite random variables sharing one probability measure.
- Meaning: the complete joint law factorizes into its marginal laws.
- Code: `FiniteJointDistribution.are_independent`.
- Source: [REF-ROSS-2019, Chapter 6].

## Generating and characteristic functions

### EQ-GF-001 — Probability-generating function

`G_X(z) = E[z^X] = sum_{k=0}^infinity p_X(k) z^k`.

- Domain: non-negative integer-valued `X`; this finite implementation gives a
  polynomial at every finite complex `z`.
- Meaning: encodes the full discrete law in polynomial coefficients and exposes
  factorial moments through derivatives at one.
- Code: `FiniteRandomVariable.probability_generating_function`.
- Source: [REF-ROSS-2019, Section 4.7].

### EQ-GF-002 — Moment-generating function

`M_X(t) = E[exp(tX)]`.

- Domain: finite real-valued `X` and finite real `t`; the mathematical finite-
  support sum exists everywhere although machine exponentials can overflow.
- Meaning: exponential transform of the law; derivatives at zero recover raw
  moments when differentiation is justified.
- Code: `FiniteRandomVariable.moment_generating_function`.
- Source: [REF-ROSS-2019, Section 4.7].

### EQ-GF-003 — Characteristic function

`phi_X(t) = E[exp(i t X)] = E[cos(tX)] + i E[sin(tX)]`.

- Domain: finite real-valued `X` and finite real `t`.
- Meaning: bounded Fourier transform of the probability law; unlike an MGF, a
  characteristic function exists for every real-valued probability law.
- Code: `FiniteRandomVariable.characteristic_function`; characteristic plot.
- Source: [REF-ROSS-2019, Section 6.5].

## Descriptive statistics

### EQ-STAT-001 — Arithmetic mean

`x_bar = (1/n) sum_{i=1}^n x_i`.

- Domain: finite real sample, integer `n >= 1`.
- Meaning: additive center of observed values.
- Code: `statistics.descriptive.mean`.
- Source: [REF-NIST-EDA, Section 1.3.5.1].

### EQ-STAT-002 — Variance with degrees-of-freedom offset

`s_d^2 = sum_{i=1}^n (x_i - x_bar)^2 / (n-d)`.

- Domain: finite real sample and integer `0 <= d < n`.
- Meaning: average squared distance from the sample mean under an explicit
  denominator convention.
- Code: `statistics.descriptive.variance`.
- Source: [REF-NIST-EDA, Sections 1.3.5.1 and 1.3.5.2].

### EQ-STAT-003 — Standard deviation

`s_d = sqrt(s_d^2)`.

- Domain: the variance in [EQ-STAT-002].
- Meaning: non-negative spread in the original measurement units.
- Code: `statistics.descriptive.standard_deviation`.
- Source: [REF-NIST-EDA, Section 1.3.5.2].

### EQ-STAT-004 — Z-score

`z_i = (x_i - x_bar) / s_d`.

- Domain: [EQ-STAT-002] with `s_d > 0`.
- Meaning: signed centered distance in standard-deviation units.
- Code: `statistics.descriptive.z_scores`.
- Source: [REF-NIST-EDA, Section 1.3.5.17].

### EQ-STAT-005 — Empirical cumulative distribution function

`F_n(x) = (1/n) sum_{i=1}^n 1{x_i <= x}`.

- Domain: non-empty finite sample and finite real `x`.
- Meaning: observed fraction at or below `x`.
- Code: `statistics.descriptive.empirical_cdf` and
  `visualization.foundations.plot_empirical_cdf`.
- Source: [REF-WASSERMAN-2004, Section 7.3].

### EQ-STAT-006 — Covariance with degrees-of-freedom offset

`cov_d(x,y) = sum_{i=1}^n (x_i-x_bar)(y_i-y_bar) / (n-d)`.

- Domain: paired finite real samples of equal positive length; `0 <= d < n`.
- Meaning: average joint centered movement.
- Code: `statistics.descriptive.covariance`.
- Source: [REF-NIST-EDA, Section 1.3.5.12].

### EQ-STAT-007 — Pearson correlation

`r_xy = cov_0(x,y) / (s_{x,0} s_{y,0})`.

- Domain: paired samples with both population standard deviations positive.
- Meaning: dimensionless linear association.
- Code: `statistics.descriptive.pearson_correlation`.
- Source: [REF-NIST-EDA, Section 1.3.5.12].

## Distributions

### EQ-DIST-BER-001 — Bernoulli PMF

`P(X=k) = p^k (1-p)^(1-k)` for `k in {0,1}`, and zero otherwise.

- Domain: `0 <= p <= 1`, integer `k`.
- Meaning: mass of one binary trial.
- Code: `distributions.Bernoulli.pmf`; Bernoulli PMF plot.
- Source: [REF-ROSS-2019, Section 4.2].

### EQ-DIST-BER-002 — Bernoulli CDF

`F(x)=0` for `x<0`; `F(x)=1-p` for `0<=x<1`; `F(x)=1` for `x>=1`.

- Domain: `0 <= p <= 1`, finite real `x`.
- Meaning: accumulated Bernoulli mass through `x`.
- Code: `distributions.Bernoulli.cdf`.
- Source: [REF-ROSS-2019, Sections 4.1-4.2].

### EQ-DIST-BER-003 — Bernoulli mean

`E[X] = p`.

- Domain: `X ~ Bernoulli(p)`.
- Meaning: long-run average success indicator under repeated sampling.
- Code: `distributions.Bernoulli.mean`.
- Source: [REF-ROSS-2019, Section 4.2].

### EQ-DIST-BER-004 — Bernoulli variance

`Var(X) = p(1-p)`.

- Domain: `X ~ Bernoulli(p)`.
- Meaning: binary-trial spread, maximal at `p=1/2`.
- Code: `distributions.Bernoulli.variance`.
- Source: [REF-ROSS-2019, Section 4.2].

### EQ-DIST-NORM-001 — Normal PDF

`f(x) = exp(-(x-mu)^2/(2 sigma^2)) / (sigma sqrt(2 pi))`.

- Domain: finite real `x` and `mu`; finite `sigma > 0`.
- Meaning: density of `Normal(mu, sigma^2)`; point density is not probability.
- Code: `distributions.Normal.pdf`; normal PDF plot.
- Source: [REF-CASELLA-BERGER-2002, Section 3.3].

### EQ-DIST-NORM-002 — Normal CDF through the error function

`F(x) = (1/2)[1 + erf((x-mu)/(sigma sqrt(2)))]`.

- Domain: the domain of [EQ-DIST-NORM-001].
- Meaning: accumulated normal probability through `x`.
- Code: `distributions.Normal.cdf`.
- Source: [REF-NIST-DLMF, Equation 7.2.1].

### EQ-DIST-NORM-003 — Normal mean

`E[X] = mu`.

- Domain: `X ~ Normal(mu, sigma^2)`.
- Meaning: distribution center.
- Code: `distributions.Normal.mean`.
- Source: [REF-CASELLA-BERGER-2002, Section 3.3].

### EQ-DIST-NORM-004 — Normal variance

`Var(X) = sigma^2`.

- Domain: `X ~ Normal(mu, sigma^2)`.
- Meaning: squared scale of the distribution.
- Code: `distributions.Normal.variance`.
- Source: [REF-CASELLA-BERGER-2002, Section 3.3].

## Numerical functions

### EQ-NUM-GAMMA-001 — Regularized incomplete gamma functions

`P(a,x) = gamma(a,x) / Gamma(a)` and
`Q(a,x) = Gamma(a,x) / Gamma(a) = 1 - P(a,x)`.

- Domain: finite `a > 0` and finite `x >= 0`.
- Meaning: normalized lower and upper incomplete-gamma integrals. The code uses
  a convergent series for the lower region and a stabilized continued fraction
  for the upper region, avoiding a numerically inappropriate single formula.
- Code: `distributions._special.regularized_gamma_p` and
  `distributions._special.regularized_gamma_q`.
- Source: [REF-NIST-DLMF-GAMMA, Sections 8.2 and 8.9].

## Discrete distribution families

### EQ-DIST-BIN-001 — Binomial PMF and log-PMF

`P(X=k) = C(n,k) p^k (1-p)^(n-k)` for `k in {0,...,n}`.

The implementation evaluates the logarithmic form with `lgamma` and `log1p`
before exponentiation, with explicit `p=0` and `p=1` limits.

- Domain: integer `n >= 0`, `0 <= p <= 1`, integer `k`.
- Meaning: probability of exactly `k` successes in `n` independent Bernoulli
  trials with common success probability `p`.
- Code: `distributions.Binomial.pmf` and `Binomial.log_pmf`.
- Source: [REF-ROSS-2019, Section 4.3].

### EQ-DIST-BIN-002 — Binomial CDF

`F(x) = sum_{k=0}^{floor(x)} C(n,k) p^k (1-p)^(n-k)` with support clipping.

- Domain: [EQ-DIST-BIN-001] and finite real `x`.
- Meaning: probability of at most `floor(x)` successes. The implementation
  sums the lower tail below the mean and the complementary upper tail above it
  to reduce cancellation in small-tail regimes.
- Code: `distributions.Binomial.cdf`.
- Source: [REF-ROSS-2019, Section 4.3].

### EQ-DIST-BIN-003 — Binomial mean

`E[X] = n p`.

- Domain: `X ~ Binomial(n,p)`.
- Meaning: expected success count.
- Code: `distributions.Binomial.mean`.
- Source: [REF-ROSS-2019, Section 4.3].

### EQ-DIST-BIN-004 — Binomial variance

`Var(X) = n p (1-p)`.

- Domain: `X ~ Binomial(n,p)`.
- Meaning: success-count spread implied by independent binary trials.
- Code: `distributions.Binomial.variance`.
- Source: [REF-ROSS-2019, Section 4.3].

### EQ-DIST-CAT-001 — Categorical probability mass

`P(X=c_j) = p_j`, where `p_j >= 0` and `sum_j p_j = 1`.

- Domain: a non-empty finite set of distinct hashable labels and a matching
  probability vector on the unit simplex.
- Meaning: one draw from named, unordered categories. Labels have no canonical
  numeric value or order, so this law does not manufacture a CDF or moments.
- Code: `distributions.Categorical`, `ProbabilitySimplexConstraint`, and
  `visualization.foundations.plot_categorical_pmf`.
- Source: [REF-ROSS-2019, Section 4.6].

### EQ-DIST-CAT-002 — Categorical Shannon entropy

`H(X) = -sum_j p_j log(p_j)`, with `0 log(0) = 0`.

- Domain: [EQ-DIST-CAT-001]; natural logarithms give entropy in nats.
- Meaning: expected self-information and uncertainty in the category draw.
- Code: `distributions.Categorical.entropy`.
- Source: [REF-COVER-THOMAS-2006, Section 2.1].

### EQ-DIST-POIS-001 — Poisson PMF and log-PMF

`P(X=k) = exp(-lambda) lambda^k / k!` for integers `k >= 0`.

- Domain: finite `lambda >= 0`; integer `k`, with the `lambda=0` distribution
  concentrated at zero.
- Meaning: count law for events occurring at a constant rate under the Poisson
  model assumptions.
- Code: `distributions.Poisson.pmf` and `Poisson.log_pmf`.
- Source: [REF-ROSS-2019, Section 4.8].

### EQ-DIST-POIS-002 — Poisson CDF and survival function

`P(X <= k) = Q(k+1, lambda)` and `P(X > k) = P(k+1, lambda)`.

- Domain: [EQ-DIST-POIS-001], finite real threshold `x`, and
  `k=floor(x) >= 0`.
- Meaning: lower count probability and its complementary upper tail, evaluated
  through the numerically appropriate regularized incomplete-gamma branch.
- Code: `distributions.Poisson.cdf` and `Poisson.sf`.
- Source: [REF-NIST-DLMF-GAMMA, Equation 8.4.10].

### EQ-DIST-POIS-003 — Poisson mean

`E[X] = lambda`.

- Domain: `X ~ Poisson(lambda)`.
- Meaning: expected count in the modeled interval.
- Code: `distributions.Poisson.mean`.
- Source: [REF-ROSS-2019, Section 4.8].

### EQ-DIST-POIS-004 — Poisson variance

`Var(X) = lambda`.

- Domain: `X ~ Poisson(lambda)`.
- Meaning: the Poisson model equates count variance and mean; empirical
  disagreement is a model-diagnostic signal rather than a fact to conceal.
- Code: `distributions.Poisson.variance`.
- Source: [REF-ROSS-2019, Section 4.8].

