# Equation Registry

This file is the authoritative catalog of equations currently evaluated by
code. Symbols are local to each record. Source keys resolve in
`docs/references.md`.

## Numerical integration and Monte Carlo propagation

### EQ-NUM-TRAP-001 — Composite trapezoid rule

`integral_a^b f(x) dx ~= h[f(x_0)/2 + sum_(i=1)^(n-1)f(x_i) + f(x_n)/2]`,
where `h=(b-a)/n`.

- Domain: finite `a<b`, positive integer `n`, equally spaced nodes, and finite
  real integrand values.
- Meaning: integrates the piecewise-linear interpolant and charges `n+1`
  evaluations.
- Code: `numerics.composite_trapezoid`.
- Source: [REF-DAVIS-RABINOWITZ-1984, Chapter 2].

### EQ-NUM-SIMP-001 — Composite Simpson rule

`integral_a^b f(x) dx ~= (h/3)[f(x_0)+4 sum_odd f(x_i)+
2 sum_even f(x_i)+f(x_n)]`.

- Domain: [EQ-NUM-TRAP-001] with a positive even number of subintervals.
- Meaning: integrates local quadratic interpolants and is exact for cubics in
  exact arithmetic.
- Code: `numerics.composite_simpson`.
- Source: [REF-DAVIS-RABINOWITZ-1984, Chapter 2].

### EQ-PROP-EXP-001 — Expectation integral

`E[g(X)] = integral g(x) f_X(x) dx`.

- Domain: measurable transform and density with an absolutely integrable
  product; the implemented rule integrates only explicitly supplied finite
  bounds.
- Meaning: deterministic uncertainty propagation of a scalar transform through
  its input density.
- Code: `propagation.expectation_by_quadrature`.
- Source: [REF-SMITH-UQ-2024, Chapter 3].

### EQ-MC-MEAN-001 — Monte Carlo sample-mean estimator

`mu_hat_N=(1/N) sum_(i=1)^N Y_i`.

- Domain: at least two finite sampled contributions; consistency requires the
  relevant law-of-large-numbers assumptions.
- Meaning: estimates an expectation from independent representative draws.
- Code: `monte_carlo.estimate_mean` and `propagation.propagate_samples`.
- Source: [REF-ROBERT-CASELLA-2004, Chapter 3].

### EQ-MC-VAR-001 — Unbiased sample variance

`s_N^2=(1/(N-1)) sum_(i=1)^N (Y_i-mu_hat_N)^2`.

- Domain: at least two finite observations.
- Meaning: estimates output variance; it is distinct from variance of the
  sample-mean estimator.
- Code: `monte_carlo.estimate_mean` and `propagation.OutputSummary.variance`.
- Source: [REF-ROBERT-CASELLA-2004, Chapter 3].

### EQ-MC-SE-001 — Estimated Monte Carlo standard error

`SE(mu_hat_N)=s_N/sqrt(N)`.

- Domain: [EQ-MC-VAR-001], with independent identically distributed sampling
  for the usual interpretation.
- Meaning: estimates sampling variability of the mean rather than physical
  output variability.
- Code: `monte_carlo.estimate_mean`.
- Source: [REF-ROBERT-CASELLA-2004, Chapters 3-4].

### EQ-MC-PROB-001 — Monte Carlo event-probability estimator

`p_hat_N=(1/N) sum_(i=1)^N I(A_i)` and
`SE(p_hat_N)=sqrt(p_hat_N(1-p_hat_N)/N)`.

- Domain: non-empty exact Boolean event indicators; the plug-in standard error
  may be zero at observed boundaries while the Wilson interval remains
  non-degenerate.
- Meaning: estimates threshold, failure, or other event probabilities.
- Code: `monte_carlo.estimate_probability`.
- Source: [REF-ROBERT-CASELLA-2004, Chapter 3], [REF-WILSON-1927].

### EQ-MC-INT-001 — Uniform Monte Carlo integration

`integral_a^b f(x) dx = (b-a) E[f(U)]`, where `U~Uniform(a,b)`.

- Domain: finite `a<b`, integrable `f`, and independent uniform draws.
- Meaning: converts a finite-interval integral into a sample-mean problem.
- Code: `monte_carlo.estimate_uniform_integral`.
- Source: [REF-ROBERT-CASELLA-2004, Chapter 3].

### EQ-MC-ROOTN-001 — Root-sample Monte Carlo scaling

`SD(mu_hat_N)=sigma/sqrt(N)` for independent equal-distribution draws with
finite variance `sigma^2`.

- Domain: independent identically distributed finite-variance contributions.
- Meaning: multiplying sample count by four is needed to halve idealized
  standard error; a trace alone does not prove asymptotic behavior.
- Code: `monte_carlo.convergence_trace` exposes nested-prefix estimates and
  standard errors.
- Source: [REF-ROBERT-CASELLA-2004, Chapters 3-4].

### EQ-MC-ANTI-001 — Antithetic-pair estimator

`Z_i=[f(U_i)+f(1-U_i)]/2` and `mu_hat_A=(1/m) sum_i Z_i`, where
`U_i~Uniform(0,1)` independently across pairs.

- Domain: integrable unit-interval `f`, at least two independent pairs, and
  finite evaluated values; gains require favorable negative covariance within
  a pair and are not guaranteed for arbitrary functions.
- Meaning: preserves `E[f(U)]`, charges two function evaluations per independent
  pair, and estimates uncertainty from the pair averages rather than treating
  the negatively dependent evaluations as independent.
- Code: `monte_carlo.estimate_antithetic_uniform`.
- Source: [REF-OWEN-2013, Section 8.2].

### EQ-MC-CONTROL-001 — Control-variate adjustment

`Y_i^(cv)=Y_i-beta(C_i-mu_C)` with
`beta*=Cov(Y,C)/Var(C)` when `Var(C)>0`.

- Domain: paired finite target/control observations and a genuinely known
  control mean `mu_C`; the automatic plug-in coefficient also requires
  positive sample control variance.
- Meaning: the adjustment preserves the target expectation for a fixed
  coefficient and can reduce variance when the control is correlated with the
  target. A coefficient fitted on the same sample makes the reported standard
  error a plug-in diagnostic rather than an exact finite-sample result.
- Code: `monte_carlo.estimate_control_variate`.
- Source: [REF-OWEN-2013, Section 8.9].

### EQ-MC-COND-001 — Conditional Monte Carlo identity

`E[Y]=E[E[Y|Z]]` and
`Var(Y)=E[Var(Y|Z)]+Var(E[Y|Z])`.

- Domain: integrable `Y`; the variance decomposition requires finite second
  moments and an actual conditional expectation under the intended model.
- Meaning: sampling `E[Y|Z]` instead of raw `Y` preserves the expectation and
  cannot have larger population variance.
- Code: `monte_carlo.estimate_conditional_expectation`.
- Source: [REF-OWEN-2013, Section 8.1].

### EQ-MC-STRAT-001 — Stratified mean and variance estimator

`mu_hat_st=sum_h W_h mu_hat_h` and
`Var_hat(mu_hat_st)=sum_h W_h^2 s_h^2/n_h`.

- Domain: disjoint exhaustive strata with known weights `W_h`, independent
  within-stratum samples, and at least two samples per stratum for each `s_h^2`.
  The implementation uses equal-width unit-interval strata, so `W_h=1/H`.
- Meaning: combines local means and accounts for heterogeneous allocation
  without pretending all sampled outputs are identically distributed.
- Code: `monte_carlo.estimate_stratified_uniform`.
- Source: [REF-OWEN-2013, Section 8.4].

### EQ-MC-IS-001 — Ordinary importance-sampling identity

`E_p[g(X)]=E_q[g(X) w(X)]`, where `w(x)=p(x)/q(x)`.

- Domain: the proposal `q` is positive wherever `|g|p` contributes, the target
  and proposal densities are evaluated consistently, and the weighted
  integrand has the moments needed for the requested error summary.
- Meaning: draws come from a convenient proposal while density-ratio weights
  preserve the target expectation. The implementation is ordinary, not
  self-normalized, importance sampling.
- Code: `monte_carlo.estimate_importance_sampling`.
- Source: [REF-OWEN-2013, Chapter 9].

### EQ-MC-ESS-001 — Weight-only effective sample size diagnostic

`ESS_w=(sum_i w_i)^2/(sum_i w_i^2)`.

- Domain: non-negative finite weights with at least one positive weight.
- Meaning: reports weight concentration on a scale from one to `N`; it is a
  heuristic diagnostic and is not the estimator's actual independent-sample
  count or a replacement for its standard error.
- Code: `monte_carlo.estimate_importance_sampling`.
- Source: [REF-OWEN-2013, Section 9.3].

### EQ-MC-NEYMAN-001 — Stratified simulation-budget allocation

`n_h proportional to W_h sigma_h` for equal per-sample costs.

- Domain: fixed total budget, known or pilot-estimated non-negative stratum
  standard deviations, and the classical independent stratified estimator.
- Meaning: the unconstrained Neyman rule minimizes stratified estimator
  variance. The implementation first enforces an integer minimum in every
  equal-width stratum, then apportions the remaining budget proportionally by
  largest remainders; this lower-bounded construction is transparent and
  deterministic but need not equal the exact constrained optimum.
- Code: `monte_carlo.allocate_stratified_budget`.
- Source: [REF-OWEN-2013, Section 8.4].

### EQ-MC-EFF-001 — Cost-normalized estimator efficiency

`score=SE(mu_hat)^2 C` and
`gain_(base->cand)=score_base/score_candidate`, where `C` is charged work.

- Domain: comparable estimands, non-negative finite estimated standard errors,
  and positive evaluation counts; both-zero scores are undefined.
- Meaning: under the usual inverse-cost variance scaling, a gain above one
  means the candidate delivers more precision per evaluation. It is an
  empirical comparison, not a universal ranking independent of model and
  integrand.
- Code: `monte_carlo.compare_efficiency` and
  `visualization.plot_efficiency_comparison`.
- Source: [REF-OWEN-2013, Chapters 2 and 8].

### EQ-PROP-THRESH-001 — Strict exceedance probability

`P(Y>S)=E[I(Y>S)]`.

- Domain: finite threshold `S` and propagated finite output samples.
- Meaning: turns a decision threshold into a Bernoulli Monte Carlo estimator;
  equality is not counted as exceedance.
- Code: `propagation.propagate_samples` and
  `monte_carlo.estimate_probability`.
- Source: [REF-SMITH-UQ-2024, Chapter 4].

### EQ-LAB-PROJECTILE-001 — Idealized projectile range

`R=v^2 sin(2 theta)/g`.

- Domain: constant gravitational acceleration, level launch/landing height, no
  drag, non-negative speed, and launch angle interpreted in radians.
- Meaning: transparent nonlinear teaching model for propagating uncertain
  speed and angle; it is not an orbital or high-fidelity trajectory model.
- Code: `examples.basic_monte_carlo_lab.projectile_model`.
- Source: [REF-HALLIDAY-RESNICK-WALKER-2014, projectile motion].

## UQ-1 continuous and multivariate distributions

### EQ-DIST-LN-001 — Lognormal PDF and log-PDF

`f(x) = exp(-(log(x)-mu)^2/(2 sigma^2)) / (x sigma sqrt(2 pi))` for
`x>0`, and zero otherwise.

- Domain: finite `mu`, finite `sigma>0`, and finite real `x`.
- Meaning: `log(X)` is normal with location `mu` and scale `sigma`; these are
  not the mean and standard deviation of `X` itself.
- Code: `distributions.LogNormal.pdf` and `LogNormal.log_pdf`.
- Source: [REF-NIST-DISTRIBUTIONS, Lognormal Distribution].

### EQ-DIST-LN-002 — Lognormal CDF and survival function

`F(x)=Phi((log(x)-mu)/sigma)` and `S(x)=1-F(x)` for `x>0`.

- Domain: [EQ-DIST-LN-001], with `F(x)=0` and `S(x)=1` for `x<=0`.
- Meaning: lower and upper probabilities are transformed normal tails; direct
  error-function branches avoid near-one subtraction.
- Code: `distributions.LogNormal.cdf` and `LogNormal.sf`.
- Source: [REF-NIST-DISTRIBUTIONS], [REF-NIST-DLMF, Section 7.2].

### EQ-DIST-LN-003 — Lognormal quantile and inverse sampling

`Q(u)=exp(mu + sigma Phi^-1(u))`.

- Domain: `0<=u<=1`, with `Q(0)=0` and `Q(1)=+infinity`.
- Meaning: exponentiating a normal quantile gives the lognormal generalized
  inverse and inverse-transform sampler.
- Code: `distributions.LogNormal.quantile` and `LogNormal.sample`.
- Source: [REF-NIST-DISTRIBUTIONS], [REF-DEVROYE-1986, Chapter IX].

### EQ-DIST-LN-004 — Lognormal moments

`E[X]=exp(mu+sigma^2/2)` and
`Var(X)=(exp(sigma^2)-1) exp(2 mu+sigma^2)`.

- Domain: [EQ-DIST-LN-001]; values must fit finite-precision range to be
  returned as floats.
- Meaning: original-scale moments are nonlinear functions of log-space
  parameters.
- Code: `distributions.LogNormal.mean` and `LogNormal.variance`.
- Source: [REF-NIST-DISTRIBUTIONS, Lognormal Distribution].

### EQ-DIST-WEIB-001 — Weibull PDF and log-PDF

`f(x)=(k/lambda)(x/lambda)^(k-1) exp(-(x/lambda)^k)` for `x>=0`.

- Domain: finite shape `k>0`, scale `lambda>0`, and finite real `x`; the
  endpoint density is infinite for `k<1`, `1/lambda` for `k=1`, and zero for
  `k>1`.
- Meaning: shape controls decreasing, constant, or increasing failure rates.
- Code: `distributions.Weibull.pdf` and `Weibull.log_pdf`.
- Source: [REF-NIST-DISTRIBUTIONS, Weibull Distribution].

### EQ-DIST-WEIB-002 — Weibull CDF and survival function

`F(x)=1-exp(-(x/lambda)^k)` and `S(x)=exp(-(x/lambda)^k)` for `x>=0`.

- Domain: [EQ-DIST-WEIB-001], with boundary clipping below support.
- Meaning: the survival form is evaluated directly and the CDF uses `expm1`
  near zero.
- Code: `distributions.Weibull.cdf` and `Weibull.sf`.
- Source: [REF-NIST-DISTRIBUTIONS, Weibull Distribution].

### EQ-DIST-WEIB-003 — Weibull quantile and inverse sampling

`Q(u)=lambda[-log(1-u)]^(1/k)`.

- Domain: `0<=u<1`, with `Q(1)=+infinity`.
- Meaning: generalized inverse used for quantiles and reproducible
  inverse-transform samples.
- Code: `distributions.Weibull.quantile` and `Weibull.sample`.
- Source: [REF-NIST-DISTRIBUTIONS], [REF-DEVROYE-1986, Chapter II].

### EQ-DIST-WEIB-004 — Weibull mean

`E[X]=lambda Gamma(1+1/k)`.

- Domain: positive finite shape and scale, subject to float range.
- Meaning: expected lifetime under the selected shape-scale convention.
- Code: `distributions.Weibull.mean`.
- Source: [REF-NIST-DISTRIBUTIONS, Weibull Distribution].

### EQ-DIST-WEIB-005 — Weibull variance

`Var(X)=lambda^2[Gamma(1+2/k)-Gamma(1+1/k)^2]`.

- Domain: positive finite shape and scale, subject to float range.
- Meaning: lifetime dispersion on the original scale.
- Code: `distributions.Weibull.variance`.
- Source: [REF-NIST-DISTRIBUTIONS, Weibull Distribution].

### EQ-DIST-WEIB-006 — Weibull hazard

`h(x)=(k/lambda)(x/lambda)^(k-1)`.

- Domain: `x>=0`; the endpoint limit follows the cases in
  [EQ-DIST-WEIB-001].
- Meaning: hazard decreases for `k<1`, is constant for `k=1`, and increases for
  `k>1`.
- Code: `distributions.Weibull.hazard`.
- Source: [REF-NIST-DISTRIBUTIONS, Weibull Distribution].

### EQ-DIST-MVN-001 — Multivariate normal density

`f(x)=(2 pi)^(-d/2)|Sigma|^(-1/2)
exp(-(x-mu)'Sigma^-1(x-mu)/2)`.

- Domain: finite `d`-vectors and symmetric positive-definite covariance
  `Sigma`; singular Gaussian measures are excluded.
- Meaning: joint Gaussian density with dependence encoded by covariance.
- Code: `multivariate.MultivariateNormal.pdf` and `log_pdf`.
- Source: [REF-MARDIA-KENT-BIBBY-1979, Chapter 1].

### EQ-DIST-MVN-002 — Squared Mahalanobis distance

`D_M^2=(x-mu)'Sigma^-1(x-mu)=||L^-1(x-mu)||_2^2`, where `Sigma=LL'`.

- Domain: [EQ-DIST-MVN-001].
- Meaning: covariance-standardized squared distance evaluated without forming
  an explicit matrix inverse.
- Code: `multivariate.MultivariateNormal.mahalanobis_squared` and `log_pdf`.
- Source: [REF-MARDIA-KENT-BIBBY-1979, Chapter 1].

### EQ-DIST-MVN-003 — Univariate Gaussian marginal

`X_i ~ Normal(mu_i, sqrt(Sigma_ii))`.

- Domain: any component of a positive-definite multivariate normal.
- Meaning: selecting one coordinate preserves normality.
- Code: `multivariate.MultivariateNormal.marginal`.
- Source: [REF-MARDIA-KENT-BIBBY-1979, Chapter 3].

### EQ-DIST-MVN-004 — Component correlation

`rho_ij=Sigma_ij/sqrt(Sigma_ii Sigma_jj)`.

- Domain: positive marginal variances guaranteed by positive definiteness.
- Meaning: standardized linear dependence between two Gaussian components.
- Code: `multivariate.MultivariateNormal.correlation`.
- Source: [REF-MARDIA-KENT-BIBBY-1979, Chapter 1].

### EQ-DIST-MVN-005 — Cholesky Gaussian sampling

`X=mu+LZ`, where `LL'=Sigma` and `Z~Normal(0,I)`.

- Domain: [EQ-DIST-MVN-001] and independent standard-normal pseudo-random
  components.
- Meaning: a linear map transfers identity covariance to `Sigma`.
- Code: `multivariate.MultivariateNormal.sample`.
- Source: [REF-DEVROYE-1986, Chapter XI].

## Linear algebra primitives

### EQ-LINALG-001 — Cholesky factorization

`A=LL'` for symmetric positive-definite `A` and lower-triangular `L` with a
positive diagonal.

- Domain: finite symmetric positive-definite square matrix.
- Meaning: exposes determinant, solves, and Gaussian transforms without an
  explicit inverse.
- Code: `multivariate.normal._cholesky_factor`.
- Source: [REF-GOLUB-VANLOAN-2013, Section 4.2].

### EQ-LINALG-002 — Forward substitution

`x_i=(b_i-sum_(j<i) L_ij x_j)/L_ii`.

- Domain: lower-triangular `L` with nonzero diagonal and matching finite `b`.
- Meaning: solves `Lx=b` one component at a time.
- Code: `multivariate.normal._solve_lower`.
- Source: [REF-GOLUB-VANLOAN-2013, Section 3.1].

## UQ-1 estimation and goodness of fit

### EQ-INF-MLE-NORM-001 — Normal maximum-likelihood estimators

`mu_hat=(1/n)sum x_i` and
`sigma_hat^2=(1/n)sum (x_i-mu_hat)^2`.

- Domain: non-empty finite real sample with positive empirical MLE variance.
- Meaning: likelihood maximizers; the variance divisor is `n`, not the
  unbiased-estimator divisor `n-1`.
- Code: `inference.fit_normal_mle`.
- Source: [REF-CASELLA-BERGER-2002, Sections 7.2-7.3].

### EQ-INF-MLE-LN-001 — Lognormal maximum-likelihood estimators

`mu_hat=mean(log x_i)` and
`sigma_hat^2=(1/n)sum(log x_i-mu_hat)^2`.

- Domain: strictly positive finite observations with nonzero log variance.
- Meaning: fitting the lognormal is equivalent to fitting a normal law after
  the natural-log transform.
- Code: `inference.fit_lognormal_mle`.
- Source: [REF-NIST-DISTRIBUTIONS], [REF-CASELLA-BERGER-2002, Section 7.2].

### EQ-INF-GOF-KS-001 — One-sample Kolmogorov-Smirnov statistic

`D_n=max_i{i/n-F(x_(i)), F(x_(i))-(i-1)/n}`.

- Domain: ordered finite sample and a fully specified continuous null CDF `F`.
- Meaning: maximum vertical distance between empirical and null CDFs.
- Code: `inference.kolmogorov_smirnov_test`.
- Source: [REF-NIST-GOF, Kolmogorov-Smirnov Test].

### EQ-INF-GOF-KS-002 — Limiting Kolmogorov tail probability

`P(K>z)=2 sum_(j>=1)(-1)^(j-1) exp(-2j^2z^2)`, with a transformed small-`z`
series and Stephens finite-sample argument correction.

- Domain: non-negative corrected statistic; asymptotic one-sample calibration
  for a fully specified continuous null.
- Meaning: approximate KS p-value; it is invalid without correction when null
  parameters are estimated from the tested data.
- Code: `inference.goodness_of_fit._kolmogorov_survival`.
- Source: [REF-NIST-GOF, Kolmogorov-Smirnov Test].

### EQ-INF-GOF-AD-001 — Anderson-Darling statistic

`A^2=-n-(1/n)sum_i(2i-1)[log F(x_(i))+log(1-F(x_(n+1-i)))]`.

- Domain: ordered finite sample and a fully specified continuous null CDF;
  exact boundary tail probability zero gives infinite discrepancy.
- Meaning: empirical-CDF discrepancy with additional tail weight; critical
  values depend on null family and parameter-estimation procedure.
- Code: `inference.anderson_darling_test`.
- Source: [REF-NIST-GOF, Anderson-Darling Test].

### EQ-INF-CI-NMEAN-001 — Normal z interval for a mean

`x_bar +- z_(1-alpha/2) sigma/sqrt(n)`.

- Domain: independent normal observations with known positive `sigma`; using
  sample standard deviation instead is explicitly labeled a large-sample
  approximation.
- Meaning: repeated-sampling confidence procedure, not a posterior probability
  statement about a fixed mean.
- Code: `inference.normal_mean_confidence_interval`.
- Source: [REF-CASELLA-BERGER-2002, Sections 9.2-9.3].

### EQ-INF-CI-WILSON-001 — Wilson score interval

`center=(p_hat+z^2/(2n))/(1+z^2/n)` and
`half=z sqrt(p_hat(1-p_hat)/n+z^2/(4n^2))/(1+z^2/n)`.

- Domain: `0<=successes<=n`, positive integer `n`, and confidence in `(0,1)`.
- Meaning: score-based binomial-proportion interval with better boundary
  behavior than the uncorrected Wald interval.
- Code: `inference.wilson_score_interval`.
- Source: [REF-WILSON-1927].

## Verification and uncertainty quantification

### EQ-UQ-001 — Named forward-model map

`Y = g(X)`, with requested components `Y_q = g_q(X)` for quantities of
interest `q`.

- Domain: `X` is a finite named parameter realization, `g` is deterministic
  for that realization, and every requested output is a finite real scalar.
- Meaning: uncertainty propagation starts from an explicit input-to-output
  contract; the quantity of interest determines which model response is used.
- Code: `uq.taxonomy.ForwardModel` and `uq.taxonomy.evaluate_model`.
- Source: [REF-SMITH-UQ-2024, Chapters 1-3].

### EQ-VV-001 — Observed order under uniform refinement

`p = log(|phi_coarse - phi_medium| / |phi_medium - phi_fine|) / log(r)`.

- Domain: three finite solutions from uniform refinement ratio `r>1`, non-zero
  consecutive differences, and an asymptotic single-term error regime.
- Meaning: estimates how rapidly the selected quantity converges as resolution
  is refined; a value alone does not prove asymptotic convergence.
- Code: `uq.verification.observed_order_of_accuracy`.
- Source: [REF-ROACHE-1998, Chapters 4-5].

### EQ-VV-002 — Richardson extrapolation

`phi_ext = phi_fine + (phi_fine - phi_medium)/(r^p - 1)`.

- Domain: finite medium and fine solutions, uniform `r>1`, positive order
  `p`, and the same asymptotic leading-error assumption as [EQ-VV-001].
- Meaning: cancels the estimated leading discretization-error term to predict
  the zero-step-size quantity.
- Code: `uq.verification.richardson_extrapolation`.
- Source: [REF-ROACHE-1998, Chapters 4-5].

### EQ-VV-003 — Conservative numerical-error budget

`epsilon_total <= epsilon_truncation + epsilon_roundoff +
epsilon_discretization + epsilon_iterative`.

- Domain: each term is a non-negative absolute error bound for the same
  quantity and units.
- Meaning: the triangle inequality gives a conservative bound without silently
  assuming independence; it is not a root-sum-square uncertainty estimate.
- Code: `uq.verification.NumericalErrorBudget.total_bound`.
- Source: [REF-OBERKAMPF-ROY-2025, Chapters 6-8].

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

## Extended distribution functions

### EQ-NUM-BETA-001 — Regularized incomplete beta function

`I_x(a,b) = B_x(a,b) / B(a,b)` and
`I_x(a,b) = 1 - I_(1-x)(b,a)`.

- Domain: finite `a>0`, `b>0`, and `0<=x<=1`.
- Meaning: normalized beta integral. The implementation uses a continued
  fraction and the symmetry branch selected around
  `x=(a+1)/(a+b+2)` to evaluate the numerically smaller tail first.
- Code: `distributions._special.regularized_beta`.
- Source: [REF-NIST-DLMF-BETA, Equations 8.17.2, 8.17.4, and 8.17.22].

### EQ-DIST-BER-005 — Bernoulli quantile

`Q(u)=0` for `u<=1-p`, and `Q(u)=1` otherwise.

- Domain: `X~Bernoulli(p)` and `0<=u<=1`.
- Meaning: generalized inverse of [EQ-DIST-BER-002] on binary support.
- Code: `distributions.Bernoulli.quantile`.
- Source: [REF-CASELLA-BERGER-2002, Chapter 2].

### EQ-DIST-BIN-005 — Binomial quantile

`Q(u) = min{k in {0,...,n}: F(k)>=u}`.

- Domain: `X~Binomial(n,p)` and `0<=u<=1`.
- Meaning: generalized inverse CDF found by monotone binary search.
- Code: `distributions.Binomial.quantile`.
- Source: [REF-CASELLA-BERGER-2002, Chapter 2].

### EQ-DIST-POIS-005 — Poisson quantile

`Q(u) = min{k in nonnegative integers: F(k)>=u}`.

- Domain: `X~Poisson(lambda)` and `0<=u<=1`; for `lambda>0`, `Q(1)=+infinity`.
- Meaning: generalized inverse CDF found by adaptive bracketing and bisection.
- Code: `distributions.Poisson.quantile`.
- Source: [REF-CASELLA-BERGER-2002, Chapter 2].

### EQ-DIST-NORM-005 — Normal survival function

`S(x) = (1/2) erfc((x-mu)/(sigma sqrt(2)))`.

- Domain: finite real `x`, `mu`, and finite `sigma>0`.
- Meaning: direct upper-tail evaluation that avoids subtracting a near-one CDF.
- Code: `distributions.Normal.sf`; continuous-functions plot.
- Source: [REF-NIST-DLMF, Equation 7.2.2].

### EQ-DIST-NORM-006 — Normal quantile

`Q(u) = mu + sigma Phi^(-1)(u)`.

- Domain: `0<=u<=1`; endpoint values are negative and positive infinity.
- Meaning: generalized inverse normal CDF.
- Code: `distributions.Normal.quantile` using `statistics.NormalDist.inv_cdf`.
- Source: [REF-PYTHON-STATISTICS].

## Waiting-count distributions

### EQ-DIST-GEOM-001 — Geometric PMF and log-PMF

`P(X=k) = p(1-p)^k` for integers `k>=0`.

- Domain: `0<p<=1` and integer `k`; `X` counts failures before first success.
- Meaning: zero-based geometric waiting count, distinct from the convention
  that counts the successful trial.
- Code: `distributions.Geometric.pmf`, `Geometric.log_pmf`, and waiting-count
  PMF plot.
- Source: [REF-ROSS-2019, Section 4.5].

### EQ-DIST-GEOM-002 — Geometric CDF and survival function

`F(x)=1-(1-p)^(floor(x)+1)` and `S(x)=(1-p)^(floor(x)+1)` for `x>=0`.

- Domain: [EQ-DIST-GEOM-001] and finite real `x`, with boundary clipping below
  support.
- Meaning: accumulated and direct upper-tail waiting-count probabilities.
- Code: `distributions.Geometric.cdf` and `Geometric.sf`.
- Source: [REF-ROSS-2019, Section 4.5].

### EQ-DIST-GEOM-003 — Geometric mean

`E[X] = (1-p)/p`.

- Domain: the failures-before-first-success convention.
- Meaning: expected failures before success.
- Code: `distributions.Geometric.mean`.
- Source: [REF-ROSS-2019, Section 4.5].

### EQ-DIST-GEOM-004 — Geometric variance

`Var(X) = (1-p)/p^2`.

- Domain: the failures-before-first-success convention.
- Meaning: spread of the zero-based waiting count.
- Code: `distributions.Geometric.variance`.
- Source: [REF-ROSS-2019, Section 4.5].

### EQ-DIST-GEOM-005 — Geometric quantile

`Q(u) = ceil(log(1-u)/log(1-p) - 1)`, clipped at zero.

- Domain: `0<p<1`, `0<=u<1`; `Q(1)=+infinity`. For `p=1`, `Q(u)=0`.
- Meaning: generalized inverse of [EQ-DIST-GEOM-002].
- Code: `distributions.Geometric.quantile`.
- Source: [REF-CASELLA-BERGER-2002, Chapter 2].

### EQ-DIST-NBIN-001 — Negative binomial PMF and log-PMF

`P(X=k) = C(k+r-1,k) p^r (1-p)^k` for integers `k>=0`.

- Domain: integer `r>=1`, `0<p<=1`; `X` counts failures before the `r`-th
  success.
- Meaning: sum of `r` independent zero-based geometric waiting counts.
- Code: `distributions.NegativeBinomial.pmf`, `NegativeBinomial.log_pmf`, and
  waiting-count PMF plot.
- Source: [REF-ROSS-2019, Section 4.5].

### EQ-DIST-NBIN-002 — Negative binomial CDF and survival function

`F(k)=I_p(r,k+1)` and `S(k)=I_(1-p)(k+1,r)`.

- Domain: [EQ-DIST-NBIN-001] and integer cutoff `k>=0`, with real thresholds
  mapped through `floor`.
- Meaning: complementary count tails evaluated through direct incomplete-beta
  branches rather than near-one subtraction.
- Code: `distributions.NegativeBinomial.cdf` and `NegativeBinomial.sf`.
- Source: [REF-NIST-DLMF-BETA, Equations 8.17.4 and 8.17.5].

### EQ-DIST-NBIN-003 — Negative binomial mean

`E[X] = r(1-p)/p`.

- Domain: failures before the `r`-th success.
- Meaning: expected total of `r` independent geometric failure counts.
- Code: `distributions.NegativeBinomial.mean`.
- Source: [REF-ROSS-2019, Section 4.5].

### EQ-DIST-NBIN-004 — Negative binomial variance

`Var(X) = r(1-p)/p^2`.

- Domain: failures before the `r`-th success.
- Meaning: variance sum of independent geometric failure counts.
- Code: `distributions.NegativeBinomial.variance`.
- Source: [REF-ROSS-2019, Section 4.5].

### EQ-DIST-NBIN-005 — Negative binomial quantile

`Q(u) = min{k in nonnegative integers: F(k)>=u}`.

- Domain: [EQ-DIST-NBIN-001] and `0<=u<=1`; a non-degenerate law has
  `Q(1)=+infinity`.
- Meaning: generalized inverse found by adaptive bracketing and bisection.
- Code: `distributions.NegativeBinomial.quantile`.
- Source: [REF-CASELLA-BERGER-2002, Chapter 2].

## Foundational continuous distributions

### EQ-DIST-UNIF-001 — Continuous uniform PDF and log-PDF

`f(x)=1/(b-a)` for `a<=x<=b`, and zero otherwise.

- Domain: finite `a<b` and finite real `x`; endpoint density values do not
  affect probabilities.
- Meaning: constant density over a finite interval.
- Code: `distributions.Uniform.pdf` and `Uniform.log_pdf`; continuous-functions
  plot.
- Source: [REF-CASELLA-BERGER-2002, Section 3.3].

### EQ-DIST-UNIF-002 — Continuous uniform CDF and survival function

`F(x)=(x-a)/(b-a)` and `S(x)=(b-x)/(b-a)` inside `[a,b]`, with boundary
clipping outside support.

- Domain: [EQ-DIST-UNIF-001].
- Meaning: linear lower and upper interval proportions.
- Code: `distributions.Uniform.cdf` and `Uniform.sf`; continuous-functions plot.
- Source: [REF-CASELLA-BERGER-2002, Section 3.3].

### EQ-DIST-UNIF-003 — Continuous uniform mean

`E[X]=(a+b)/2`.

- Domain: `X~Uniform(a,b)` with `a<b`.
- Meaning: interval midpoint.
- Code: `distributions.Uniform.mean`.
- Source: [REF-CASELLA-BERGER-2002, Section 3.3].

### EQ-DIST-UNIF-004 — Continuous uniform variance

`Var(X)=(b-a)^2/12`.

- Domain: `X~Uniform(a,b)` with `a<b`.
- Meaning: squared interval-width dispersion.
- Code: `distributions.Uniform.variance`.
- Source: [REF-CASELLA-BERGER-2002, Section 3.3].

### EQ-DIST-UNIF-005 — Continuous uniform quantile

`Q(u)=a+u(b-a)`.

- Domain: `0<=u<=1` and finite `a<b`.
- Meaning: linear generalized inverse CDF.
- Code: `distributions.Uniform.quantile`.
- Source: [REF-CASELLA-BERGER-2002, Chapter 2].

### EQ-DIST-UNIF-006 — Continuous uniform hazard

`h(x)=1/(b-x)` for `a<=x<b`, and zero below support.

- Domain: finite `x<b`; hazard is undefined after survival reaches zero.
- Meaning: conditional instantaneous rate increases toward the fixed endpoint.
- Code: `distributions.Uniform.hazard`.
- Source: [REF-KLEIN-MOESCHBERGER-2003, Chapter 2].

### EQ-DIST-EXP-001 — Exponential PDF and log-PDF

`f(x)=lambda exp(-lambda x)` for `x>=0`, and zero otherwise.

- Domain: finite `lambda>0` and finite real `x`.
- Meaning: rate-parameterized continuous waiting-time density.
- Code: `distributions.Exponential.pdf` and `Exponential.log_pdf`;
  continuous-functions plot.
- Source: [REF-ROSS-2019, Chapter 5].

### EQ-DIST-EXP-002 — Exponential CDF and survival function

`F(x)=1-exp(-lambda x)` and `S(x)=exp(-lambda x)` for `x>=0`.

- Domain: [EQ-DIST-EXP-001], with `F(x)=0` and `S(x)=1` below support.
- Meaning: complementary waiting-time probabilities; the CDF uses `expm1` for
  small arguments while survival is evaluated directly.
- Code: `distributions.Exponential.cdf` and `Exponential.sf`;
  continuous-functions plot.
- Source: [REF-ROSS-2019, Chapter 5].

### EQ-DIST-EXP-003 — Exponential mean

`E[X]=1/lambda`.

- Domain: `X~Exponential(lambda)`, `lambda>0`.
- Meaning: expected waiting time, reciprocal to the event rate.
- Code: `distributions.Exponential.mean`.
- Source: [REF-ROSS-2019, Chapter 5].

### EQ-DIST-EXP-004 — Exponential variance

`Var(X)=1/lambda^2`.

- Domain: `X~Exponential(lambda)`, `lambda>0`.
- Meaning: squared waiting-time scale.
- Code: `distributions.Exponential.variance`.
- Source: [REF-ROSS-2019, Chapter 5].

### EQ-DIST-EXP-005 — Exponential quantile

`Q(u)=-log(1-u)/lambda`.

- Domain: `0<=u<1`, with `Q(1)=+infinity`.
- Meaning: generalized inverse CDF and inverse-transform sampling equation.
- Code: `distributions.Exponential.quantile` and `Exponential.sample`.
- Source: [REF-DEVROYE-1986, Chapter II].

### EQ-DIST-EXP-006 — Exponential hazard

`h(x)=f(x)/S(x)=lambda` for `x>=0`.

- Domain: `X~Exponential(lambda)` on non-negative support.
- Meaning: constant instantaneous rate, equivalent to memorylessness.
- Code: `distributions.Exponential.hazard`.
- Source: [REF-KLEIN-MOESCHBERGER-2003, Chapter 2].

