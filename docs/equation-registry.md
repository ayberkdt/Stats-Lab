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

## Local and global sensitivity analysis

### EQ-LAB-ISHIGAMI-001 — Ishigami sensitivity benchmark

`Y=sin(X_1)+7 sin(X_2)^2+0.1 X_3^4 sin(X_1)`, with independent
`X_i~Uniform(-pi,pi)`.

- Domain: three independent unit-density uniform coordinates on `[-pi,pi]`.
- Meaning: teaching benchmark whose third input acts through interaction with
  the first and therefore distinguishes first-order from total-effect ranking.
- Code: `examples.sensitivity_lab.ishigami`.
- Source: [REF-SALTELLI-2008, Chapter 4].

### EQ-SENS-LOCAL-001 — Central finite-difference sensitivity

`partial_i f(x) ~= [f(x+h_i e_i)-f(x-h_i e_i)]/(2 h_i)`.

- Domain: positive finite step, finite model evaluations, and sufficient local
  smoothness for the usual second-order truncation interpretation.
- Meaning: measures local slope at one point; it does not average over the
  uncertain input distribution and charges two model calls per coordinate.
- Code: `sensitivity.central_difference_gradient`.
- Source: [REF-FORNBERG-1988].

### EQ-SENS-ANOVA-001 — Functional ANOVA variance decomposition

`f(X)=f_0+sum_i f_i(X_i)+sum_(i<j) f_ij(X_i,X_j)+...` and
`Var(Y)=sum_(u nonempty) V_u`.

- Domain: square-integrable scalar model response and mutually independent
  input variables under the product measure.
- Meaning: orthogonal component variances separate main and interaction
  contributions; dependence requires different sensitivity definitions.
- Code: `sensitivity.estimate_sobol_indices` estimates normalized components.
- Source: [REF-SOBOL-1993], [REF-SALTELLI-2008, Chapter 4].

### EQ-SENS-FIRST-001 — First-order Sobol index

`S_i=Var(E[Y|X_i])/Var(Y)`.

- Domain: [EQ-SENS-ANOVA-001] with positive output variance.
- Meaning: fraction of output variance explained by input `i` alone.
- Code: `sensitivity.estimate_sobol_indices` uses a Saltelli pick-freeze
  estimator and retains finite-sample values without clipping.
- Source: [REF-SOBOL-1993], [REF-SALTELLI-2010].

### EQ-SENS-SECOND-001 — Second-order Sobol interaction

`S_ij=V_ij/Var(Y)` for `i != j`, after subtracting both main effects from the
closed two-input contribution.

- Domain: [EQ-SENS-ANOVA-001].
- Meaning: variance caused by the pair beyond their separate main effects.
- Code: `sensitivity.estimate_sobol_indices` evaluates pair hybrids explicitly.
- Source: [REF-SOBOL-1993], [REF-SALTELLI-2008, Chapter 4].

### EQ-SENS-TOTAL-001 — Total-effect Sobol index

`S_Ti=E[Var(Y|X_(-i))]/Var(Y)=1-Var(E[Y|X_(-i)])/Var(Y)`.

- Domain: [EQ-SENS-ANOVA-001].
- Meaning: all variance involving input `i`, including every interaction; the
  implementation uses the Jansen squared-difference estimator.
- Code: `sensitivity.estimate_sobol_indices`, `SobolResult.ranking`, and
  `SobolResult.negligible_inputs`.
- Source: [REF-SALTELLI-2010].

### EQ-SENS-DECISION-001 — Total-effect screening rule

`fix_candidates={i:S_Ti<=tau}` and remaining inputs are prioritized by
descending estimated `S_Ti`, for a declared non-negative threshold `tau`.

- Domain: an interpreted Sobol study with a user-chosen screening threshold.
- Meaning: translates global importance into a transparent dimension-reduction
  and measurement-priority suggestion; it is not a causal or economic optimum.
- Code: `sensitivity.SobolResult.decision`.
- Source: [REF-SALTELLI-2008, Chapters 1 and 6].

## Reliability and rare-event analysis

### EQ-REL-LIMIT-001 — Limit-state failure convention

`failure = {x : g(x) <= 0}` and `safe = {x : g(x) > 0}`.

- Domain: a finite deterministic scalar limit-state evaluation on each input.
- Meaning: zero belongs to the closed failure domain, so every reliability
  method uses one explicit and consistent boundary convention.
- Code: `reliability.LimitState.evaluate` and `LimitState.failed`.
- Source: [REF-MELCHERS-BECK-2018, Chapter 3].

### EQ-REL-PF-001 — Failure probability and reliability

`P_f=P(g(X)<=0)=E[I(g(X)<=0)]` and `R=1-P_f`.

- Domain: a probabilistic input model and measurable failure event.
- Meaning: direct Monte Carlo estimates the Bernoulli event and retains a
  Wilson interval, including a nonzero upper bound after zero observed failures.
- Code: `reliability.estimate_failure_probability`.
- Source: [REF-MELCHERS-BECK-2018, Chapters 3 and 5].

### EQ-REL-IS-001 — Importance-weighted failure estimator

`P_f=E_q[I(g(X)<=0) p(X)/q(X)]`.

- Domain: proposal support covers the target failure contribution, likelihood
  ratios are correct and non-negative, and weighted contributions have finite
  variance for the usual error interpretation.
- Meaning: proposal draws can deliberately visit a rare failure region while
  ordinary importance weights preserve the target probability.
- Code: `reliability.estimate_weighted_failure_probability`.
- Source: [REF-MELCHERS-BECK-2018, Chapter 6], [REF-OWEN-2013, Chapter 9].

### EQ-REL-FORM-001 — Affine standard-normal FORM geometry

For `g(u)=b+a^T u`, `U~N(0,I)`,
`beta=b/||a||`, `u*=-beta a/||a||`, and `P_f=Phi(-beta)`.

- Domain: nonzero finite coefficient vector and independent standard-normal
  coordinates. For affine boundaries the result is exact; general nonlinear
  FORM additionally requires transformation, gradients, and iterative
  constrained optimization not claimed by this implementation.
- Meaning: `u*` solves the nearest-boundary problem and signed `beta` records
  whether the origin lies in the safe or failed half-space.
- Code: `reliability.linear_normal_form`.
- Source: [REF-MELCHERS-BECK-2018, Chapters 4 and 5].

### EQ-REL-DIR-001 — Directional standard-normal probability

`P_f=E_A[Q(d/2, r(A)^2/2)]`, where `A` is a uniform unit direction,
`R^2~chi-square_d`, and failure begins at radial boundary `r(A)`.

- Domain: independent standard-normal space and a failure set that is
  star-shaped from the origin along sampled rays; a ray without failure
  contributes zero.
- Meaning: analytically integrates the radial chi tail and Monte Carlo samples
  only direction, often reducing variance for suitable reliability geometry.
- Code: `reliability.estimate_directional_failure_probability`.
- Source: [REF-MELCHERS-BECK-2018, Chapter 6].

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
- Code: `linalg.cholesky`, `linalg.cholesky_semidefinite`, and every
  Gaussian consumer (`multivariate`, `inference`, `filtering`).
- Source: [REF-GOLUB-VANLOAN-2013, Section 4.2].

### EQ-LINALG-002 — Forward substitution

`x_i=(b_i-sum_(j<i) L_ij x_j)/L_ii`.

- Domain: lower-triangular `L` with nonzero diagonal and matching finite `b`.
- Meaning: solves `Lx=b` one component at a time.
- Code: `linalg.solve_lower` and `linalg.cholesky_solve`.
- Source: [REF-GOLUB-VANLOAN-2013, Section 3.1].

### EQ-LINALG-003 — Back substitution

`x_i=(b_i-sum_(j>i) U_ij x_j)/U_ii`, evaluated from `i=n` down to `1`.

- Domain: upper-triangular `U` with nonzero diagonal and matching finite `b`.
- Meaning: solves `Ux=b`; paired with [EQ-LINALG-002] it solves `LL'x=b`.
- Code: `linalg.solve_upper`, `linalg.cholesky_solve`, `linalg.solve_spd`.
- Source: [REF-GOLUB-VANLOAN-2013, Section 3.1].

### EQ-LINALG-004 — Gaussian elimination with partial pivoting

`PA=LU`, where each elimination step swaps in the row with the largest
remaining pivot magnitude before subtracting multiples of it.

- Domain: square finite non-singular `A`; a pivot below `1e-14 max|A_ij|` is
  reported as singular rather than perturbed.
- Meaning: general dense solve used where symmetry or definiteness is not
  guaranteed, such as Newton steps on nonsymmetric Jacobians.
- Code: `linalg.lu_solve`.
- Source: [REF-GOLUB-VANLOAN-2013, Section 3.4].

### EQ-LINALG-005 — Cyclic Jacobi eigenvalue iteration

`A_(k+1)=J_k' A_k J_k`, where each plane rotation `J_k(p,q,theta)` zeroes
`A_pq`; `V=prod J_k` accumulates the eigenvectors.

- Domain: real symmetric finite `A`; iteration stops when the off-diagonal
  Frobenius norm falls below a relative tolerance.
- Meaning: returns `A=V diag(lambda) V'` with orthonormal `V`, used for
  curvature analysis (SORM), spectral conditioning, and covariance geometry.
- Code: `linalg.symmetric_eigen`.
- Source: [REF-GOLUB-VANLOAN-2013, Section 8.5].

### EQ-LINALG-006 — Log-determinant from a Cholesky factor

`log det A = 2 sum_i log L_ii` for `A=LL'`.

- Domain: symmetric positive-definite `A`.
- Meaning: stable Gaussian normalizing constants without forming `det A`.
- Code: `linalg.log_det_spd`.
- Source: [REF-GOLUB-VANLOAN-2013, Section 4.2].

### EQ-LINALG-007 — Linear covariance propagation

`Cov(HX)=H Sigma H'` for a linear map `H` and `Cov(X)=Sigma`.

- Domain: finite conformable `H` and symmetric `Sigma`; the result is
  symmetrized to remove rounding asymmetry.
- Meaning: exact second-moment propagation through linear maps; the basis of
  Gaussian predictive covariance and Kalman prediction.
- Code: `linalg.sandwich`.
- Source: [REF-GOLUB-VANLOAN-2013, Section 1.1] and
  [REF-BISHOP-2006, Section 3.3].

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
  for a fully specified continuous null, used for `n > 100` (smaller samples
  use the exact law of [EQ-INF-GOF-KS-003] and [EQ-INF-GOF-KS-004]).
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

`F(x) = (1/2)[1 + erf((x-mu)/(sigma sqrt(2)))] = (1/2) erfc(-(x-mu)/(sigma sqrt(2)))`.

The code evaluates the `erfc` form, which avoids cancellation in the lower
tail (measured round-trip relative error below `1e-13` down to `F(x)=1e-300`).

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

## Inverse problems and Bayesian calibration

### EQ-REG-LINE-001 — Straight-line response model

`E[Y | x] = alpha + beta x`.

- Domain: finite scalar predictor `x` and finite coefficients `alpha`, `beta`.
- Meaning: conditional response mean for simple linear calibration.
- Code: `inference.LinearRegressionResult.predict` and
  `inverse.LinearCalibrationResult.design_rows`.
- Source: [REF-MONTGOMERY-PECK-VINING-2021, Chapter 1].

### EQ-REG-OLS-001 — Simple linear least-squares estimates

`beta_hat = sum((x_i-x_bar)(y_i-y_bar)) / sum((x_i-x_bar)^2)` and
`alpha_hat = y_bar - beta_hat x_bar`.

- Domain: at least three paired finite observations with non-constant
  predictors.
- Meaning: the unique intercept and slope minimizing residual sum of squares.
- Code: `inference.fit_simple_linear_regression`.
- Source: [REF-MONTGOMERY-PECK-VINING-2021, Chapters 1-2].

### EQ-REG-DIAG-001 — Regression residual variance and coefficient of determination

`s^2 = SSE/(n-2)` and `R^2 = 1 - SSE/SST`.

- Domain: [EQ-REG-OLS-001]; `R^2=1` by convention for a constant response
  fitted exactly.
- Meaning: residual scale estimate and in-sample variance-explanation summary.
- Code: `inference.fit_simple_linear_regression`.
- Source: [REF-MONTGOMERY-PECK-VINING-2021, Chapter 2].

### EQ-INV-OBS-001 — Additive observation model

`y_i = f(x_i, theta) + delta_i + epsilon_i`, with
`Var(delta_i + epsilon_i) = sigma_delta^2 + sigma_epsilon^2` under independence.

- Domain: paired finite inputs and responses; positive measurement variance;
  non-negative discrepancy variance; independent centered Gaussian terms for
  the implemented likelihood.
- Meaning: measurement noise and model discrepancy combine numerically but
  remain distinct uncertainty sources.
- Code: `inverse.GaussianObservationModel`.
- Source: [REF-KENNEDY-OHAGAN-2001].

### EQ-INV-RESID-001 — Calibration residual

`r_i(theta) = y_i - f(x_i, theta)`.

- Domain: [EQ-INV-OBS-001].
- Meaning: signed observed-minus-modeled response difference.
- Code: `inverse.GaussianObservationModel.residuals`.
- Source: [REF-SMITH-UQ-2024, Chapter 8].

### EQ-INV-RMSE-001 — Root mean squared calibration residual

`RMSE(theta) = sqrt((1/n) sum_i r_i(theta)^2)`.

- Domain: a non-empty residual vector from [EQ-INV-RESID-001].
- Meaning: response-scale descriptive fit error, not a validation certificate.
- Code: `inverse.GaussianObservationModel.root_mean_squared_residual`.
- Source: [REF-SMITH-UQ-2024, Chapter 8].

### EQ-INV-IDENT-001 — Linear-design identifiability and conditioning

`rank(H)=p` and `kappa_2(H^T H)=lambda_max/lambda_min`.

- Domain: finite design matrix; the implemented diagnostic specializes to the
  two-column intercept/slope design.
- Meaning: full column rank is structural identifiability; a large condition
  number warns of weak practical separation.
- Code: `inverse.diagnose_linear_identifiability`.
- Source: [REF-TARANTOLA-2005, Chapter 3] and
  [REF-GOLUB-VANLOAN-2013, Section 8.1].

### EQ-BAYES-RULE-001 — Bayes' rule for parameter inference

`p(theta | y) = p(y | theta) p(theta) / p(y)`.

- Domain: a proper prior and finite positive evidence for the observed data.
- Meaning: posterior uncertainty is conditional on prior, likelihood, data,
  and model assumptions.
- Code: `inference.condition_linear_gaussian`.
- Source: [REF-GELMAN-ETAL-2013, Chapter 1].

### EQ-BAYES-LIKE-001 — Independent Gaussian log-likelihood

`log p(y | mu,sigma^2) = -(n/2)log(2 pi sigma^2)
 - sum_i (y_i-mu_i)^2/(2 sigma^2)`.

- Domain: paired finite observations and predictions with known
  `sigma^2 > 0`.
- Meaning: log density of independent homoscedastic Gaussian measurement
  errors.
- Code: `inference.gaussian_log_likelihood` and
  `inverse.GaussianObservationModel.log_likelihood`.
- Source: [REF-GELMAN-ETAL-2013, Section 3.2].

### EQ-BAYES-LIN-POST-001 — Linear-Gaussian posterior conditioning

`Sigma_n = (Sigma_0^-1 + H^T H/sigma^2)^-1` and
`mu_n = Sigma_n(Sigma_0^-1 mu_0 + H^T y/sigma^2)`.

- Domain: positive-definite Gaussian prior covariance, finite compatible
  design and observations, and known `sigma^2 > 0`.
- Meaning: exact conjugate posterior mean and covariance for linear parameters.
- Code: `inference.condition_linear_gaussian` and
  `inverse.calibrate_linear_gaussian`.
- Source: [REF-GELMAN-ETAL-2013, Section 3.3] and
  [REF-BISHOP-2006, Section 3.3].

### EQ-BAYES-POST-SAMPLE-001 — Gaussian posterior sampling

`theta^(s) = mu_n + L z^(s)`, where `L L^T = Sigma_n` and `z^(s) ~ N(0,I)`.

- Domain: positive-definite posterior covariance and a controlled random stream.
- Meaning: correlated parameter draws preserving posterior covariance.
- Code: `inference.GaussianLinearPosterior.sample`.
- Source: [REF-DEVROYE-1986, Chapter XI].

### EQ-BAYES-PRED-001 — Linear posterior-predictive distribution

`y_* | y ~ N(H_* mu_n, H_* Sigma_n H_*^T + sigma_obs^2 I + sigma_delta^2 I)`.

- Domain: [EQ-BAYES-LIN-POST-001], compatible prediction design, and
  non-negative included variance components.
- Meaning: separates parameter, future-observation, and model-discrepancy
  uncertainty; latent prediction omits `sigma_obs^2`.
- Code: `inference.posterior_predictive` and `PosteriorPredictive.sample`.
- Source: [REF-GELMAN-ETAL-2013, Section 6.3] and
  [REF-KENNEDY-OHAGAN-2001].

### EQ-BAYES-PPC-001 — Posterior-predictive tail diagnostic

`p_B = (1/S) sum_s 1[T(y_rep^s) >= T(y_obs)]`, with
`T(y)=(y-mu_pred)^T Sigma_pred^-1 (y-mu_pred)`.

- Domain: positive-definite joint predictive covariance and `S>0` seeded
  replicates.
- Meaning: simulation-based extremeness diagnostic; it is not a frequentist
  p-value or a model-validation certificate.
- Code: `inference.posterior_predictive_check`.
- Source: [REF-GELMAN-ETAL-2013, Section 6.3].

### EQ-CAL-DISC-001 — Additive Gaussian model-discrepancy variance

`delta ~ N(0, sigma_delta^2)` and
`Var(model + delta + epsilon) = Var(model) + sigma_delta^2 + sigma_epsilon^2`
under independence.

- Domain: declared finite `sigma_delta >= 0`, independent additive centered
  discrepancy, and a stated applicability domain.
- Meaning: records model inadequacy separately from parameter and observation
  uncertainty while acknowledging possible confounding.
- Code: `validation.AdditiveGaussianDiscrepancy`,
  `inverse.GaussianObservationModel`, and `inference.posterior_predictive`.
- Source: [REF-KENNEDY-OHAGAN-2001] and
  [REF-BRYNJARSDOTTIR-OHAGAN-2014].

### EQ-LAB-CAL-001 — Calibration-laboratory data generator

`Y_i = 1.25 + 0.82 x_i + epsilon_i`, where `epsilon_i ~ N(0,0.35^2)`.

- Domain: finite teaching inputs and a call-local pseudo-random stream.
- Meaning: reproducible synthetic truth for demonstrating calibration without
  implying that generated data validate the fitted model.
- Code: `examples/bayesian_calibration_lab.py`.
- Source: [REF-GELMAN-ETAL-2013, Chapter 3].


## Central probabilistic modeling

### EQ-MOD-NATAF-001 — Nataf isoprobabilistic transformation

`z_i = Phi^-1(F_i(x_i))`, `u = L^-1 z`, with `R = LL'`; inversely
`x_i = F_i^-1(Phi((Lu)_i))`.

- Domain: continuous strictly increasing marginal CDFs `F_i` on their open
  supports and a symmetric positive-definite latent correlation `R` with unit
  diagonal. `R = I` gives the independent (Rosenblatt) case.
- Meaning: maps physical inputs to independent standard normals so that FORM,
  SORM, subset simulation, and every sampler share one joint law.
- Code: `modeling.JointInput.to_latent`, `to_standard_normal`,
  `from_standard_normal`, `from_unit`.
- Source: [REF-LIU-DERKIUREGHIAN-1986] and
  [REF-MELCHERS-BECK-2018, Chapter 5].

### EQ-MOD-NATAF-002 — Nataf latent-correlation integral equation

`rho_X,ij = integral integral h_i(z_1) h_j(z_2) phi_2(z_1,z_2; rho_Z,ij) dz_1 dz_2`,
with `h_k(z) = (F_k^-1(Phi(z)) - mu_k)/sigma_k`.

- Domain: marginals with finite variance and a target `rho_X` inside the
  attainable range for that marginal pair; otherwise the solver reports that
  the target cannot be reached.
- Meaning: finds the Gaussian-copula parameter that reproduces a prescribed
  physical Pearson correlation. The integral uses a tensor Gauss-Hermite rule
  ([EQ-MOD-GH-001]) and Brent root finding ([EQ-OPT-BRENT-001]).
- Code: `modeling.nataf_correlation`, `modeling.JointInput.from_pearson`.
- Source: [REF-LIU-DERKIUREGHIAN-1986].

### EQ-MOD-COPULA-001 — Gaussian-copula joint density

`log f(x) = sum_i log f_i(x_i) - (1/2) z'(R^-1 - I) z - (1/2) log det R`,
with `z` from [EQ-MOD-NATAF-001].

- Domain: as [EQ-MOD-NATAF-001]; `-inf` outside the product support.
- Meaning: the joint log-density used by MCMC priors, importance weights,
  and likelihood-based diagnostics.
- Code: `modeling.JointInput.log_pdf`.
- Source: [REF-NELSEN-2006, Section 2.3] and [REF-LIU-DERKIUREGHIAN-1986].

### EQ-MOD-GH-001 — Probabilists' Gauss-Hermite quadrature (Golub-Welsch)

`E[g(Z)] ~= sum_k w_k g(z_k)` for `Z ~ N(0,1)`, where `z_k` are eigenvalues of
the symmetric tridiagonal Jacobi matrix `J` with `J_(k,k+1) = sqrt(k)` and
`w_k` is the squared first component of the k-th normalized eigenvector.

- Domain: positive order `n <= 80`; exact for polynomials of degree `<= 2n-1`.
- Meaning: Gaussian expectations without random sampling error.
- Code: `modeling.gauss_hermite`.
- Source: [REF-GOLUB-WELSCH-1969].

## Sampling designs and random streams

### EQ-RNG-SPLITMIX-001 — SplitMix64 seed derivation

`z <- (s + 0x9E3779B97F4A7C15) mod 2^64`,
`z <- (z xor (z >> 30)) * 0xBF58476D1CE4E5B9`,
`z <- (z xor (z >> 27)) * 0x94D049BB133111EB`, output `z xor (z >> 31)`; a
child seed is the mix of the parent seed XOR the mixed key.

- Domain: integer states modulo `2^64`; integer or UTF-8 text keys.
- Meaning: reproducible, order-independent named random streams that do not
  depend on Python's salted string hashing.
- Code: `foundations.splitmix64`, `foundations.derive_seed`,
  `foundations.RandomStream`.
- Source: [REF-STEELE-LEA-FLOOD-2014].

### EQ-SAMP-INV-001 — Inverse-transform sampling

`X = F^-1(U)` with `U ~ Uniform(0,1)` has CDF `F`.

- Domain: generalized inverse of a univariate CDF; design coordinates are
  kept strictly inside `(0,1)`.
- Meaning: lets any unit-hypercube design (MC, LHS, QMC) induce samples from
  arbitrary marginals while keeping its stratification.
- Code: `modeling.JointInput.from_unit`, `sample_design`, `sample`.
- Source: [REF-DEVROYE-1986, Section II.2].

### EQ-SAMP-LHS-001 — Latin hypercube sampling

`X_ij = (pi_j(i) + V_ij)/n`, where `pi_j` are independent random permutations
of `{0,...,n-1}` and `V_ij ~ Uniform(0,1)` (or `1/2` when centered).

- Domain: positive `n` and `d`.
- Meaning: every one-dimensional projection has exactly one point per
  stratum; the mean estimator is unbiased with variance never much worse than
  Monte Carlo and much smaller for additive integrands.
- Code: `sampling.latin_hypercube`.
- Source: [REF-MCKAY-BECKMAN-CONOVER-1979] and
  [REF-OWEN-2013, Section 10.3].

### EQ-SAMP-HALTON-001 — Radical inverse and Halton sequence

`phi_b(i) = sum_k d_k(i) b^-(k+1)` for base-`b` digits `d_k(i)`; the Halton
point is `(phi_(p_1)(i), ..., phi_(p_d)(i))` for the first `d` primes. A
scrambled version replaces `d_k` by `pi(d_k)` with `pi(0) = 0`.

- Domain: non-negative integer index, dimension `<= 50`.
- Meaning: deterministic low-discrepancy points; quality degrades in high
  dimension, which scrambling partly mitigates.
- Code: `sampling.radical_inverse`, `sampling.halton`.
- Source: [REF-HALTON-1960] and [REF-OWEN-2013, Section 15.4].

### EQ-SAMP-SOBOL-001 — Sobol digital sequence (Gray-code construction)

`m_k = 2a_1 m_(k-1) xor 4a_2 m_(k-2) xor ... xor 2^s m_(k-s) xor m_(k-s)`,
`v_k = m_k / 2^k`, and `x_(i+1) = x_i xor v_(c(i))`, where `c(i)` is the index
of the lowest zero bit of `i` and `a_j` are coefficients of a primitive
polynomial of degree `s`.

- Domain: dimension `<= 16` with the bundled primitive polynomials and
  initial direction numbers; `2^m` points give the full net property.
- Meaning: a `(t,m,d)`-net whose one-dimensional projections are perfectly
  stratified at every power of two.
- Code: `sampling.sobol`.
- Source: [REF-SOBOL-1967], [REF-BRATLEY-FOX-1988], and [REF-JOE-KUO-2008].

### EQ-SAMP-RQMC-001 — Randomized quasi-Monte Carlo

Cranley-Patterson shift `x' = (x + Delta) mod 1` with `Delta ~ U(0,1)^d`, or
digital shift `x' = x xor Delta` for base-2 nets.

- Domain: any deterministic point set on `[0,1)^d`.
- Meaning: each randomized point is uniform, so the estimator is unbiased;
  replicates give an honest error estimate while low discrepancy is kept.
- Code: `sampling.halton(shift=True)`, `sampling.sobol(scramble=True)`,
  `propagation.compare_designs`.
- Source: [REF-CRANLEY-PATTERSON-1976] and [REF-L-ECUYER-2018].

### EQ-SAMP-DISC-001 — Grid lower bound on star discrepancy

`D*_grid = max_(t in G) | #{x_i in [0,t)}/n - prod_j t_j |` over anchored boxes
whose corners lie on a regular grid `G`.

- Domain: dimension `<= 4` for tractable enumeration.
- Meaning: a lower bound on the star discrepancy that ranks designs of equal
  size; it is not the exact discrepancy.
- Code: `sampling.star_discrepancy_bound`.
- Source: [REF-OWEN-2013, Section 15.2].

## Optimization and numerical differentiation

### EQ-OPT-FD-001 — Central difference derivative

`df/dx_i ~= [f(x + h e_i) - f(x - h e_i)] / (2h)`, `h = eps_rel max(1,|x_i|)`.

- Domain: `f` continuously differentiable near `x`; truncation error
  `O(h^2)` plus rounding error `O(eps/h)`.
- Meaning: gradient and Jacobian approximations for optimization, FORM, and
  Laplace approximations; each costs `2n` evaluations.
- Code: `optimization.central_gradient`, `optimization.central_jacobian`.
- Source: [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

### EQ-OPT-FD-002 — Central difference Hessian

`H_ii ~= [f(x+h_i e_i) - 2f(x) + f(x-h_i e_i)]/h_i^2` and
`H_ij ~= [f(++) - f(+-) - f(-+) + f(--)]/(4 h_i h_j)`.

- Domain: `f` twice continuously differentiable near `x`.
- Meaning: curvature for SORM, Laplace posterior covariances, and
  diagnostics; costs `1 + 2n + 4n(n-1)/2` evaluations.
- Code: `optimization.central_hessian`.
- Source: [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

### EQ-OPT-ARMIJO-001 — Armijo sufficient decrease

Accept step `alpha` when `f(x + alpha p) <= f(x) + c alpha grad f(x)' p` with
`c = 1e-4`, halving `alpha` otherwise.

- Domain: descent direction `grad f(x)' p < 0`.
- Meaning: guarantees monotone objective decrease in line-search methods.
- Code: `optimization.minimize_bfgs`.
- Source: [REF-NOCEDAL-WRIGHT-2006, Section 3.1].

### EQ-OPT-BFGS-001 — BFGS inverse-Hessian update

`H_(k+1) = (I - rho s y') H_k (I - rho y s') + rho s s'`, `rho = 1/(y's)`,
`s = x_(k+1) - x_k`, `y = grad_(k+1) - grad_k`.

- Domain: curvature condition `y's > 0`; otherwise the update is skipped.
- Meaning: superlinearly convergent quasi-Newton minimization of smooth
  objectives.
- Code: `optimization.minimize_bfgs`.
- Source: [REF-NOCEDAL-WRIGHT-2006, Section 6.1].

### EQ-OPT-NM-001 — Nelder-Mead simplex moves

With centroid `c` of the best `n` vertices and worst vertex `w`, trial points
are `c + t(c - w)` for reflection `t=1`, expansion `t=2`, outside/inside
contraction `t=+-1/2`, followed by shrinkage toward the best vertex.

- Domain: any objective; no convergence guarantee in general.
- Meaning: derivative-free minimization for noisy or non-smooth objectives.
- Code: `optimization.minimize_nelder_mead`.
- Source: [REF-NELDER-MEAD-1965].

### EQ-OPT-LM-001 — Levenberg-Marquardt step

`(J'J + lambda diag(J'J)) delta = -J'r`; accept when the cost
`0.5||r||^2` decreases, then shrink `lambda`, otherwise enlarge it.

- Domain: residual vector with at least as many components as parameters.
- Meaning: interpolates between Gauss-Newton and scaled gradient descent for
  nonlinear least squares and calibration.
- Code: `optimization.least_squares`.
- Source: [REF-MORE-1978] and [REF-NOCEDAL-WRIGHT-2006, Section 10.3].

### EQ-OPT-LM-002 — Gauss-Newton asymptotic covariance

`Cov(theta_hat) ~= s^2 (J'J)^-1`, with `s^2 = ||r||^2/(m-n)` or a known noise
variance.

- Domain: local linearity near the optimum, homoscedastic independent
  residuals, and full-rank `J`.
- Meaning: first-order parameter uncertainty of nonlinear least squares.
- Code: `optimization.gauss_newton_covariance`.
- Source: [REF-SEBER-WILD-2003, Section 2.1].

### EQ-OPT-BRENT-001 — Brent root finding

Combine inverse quadratic interpolation, secant steps, and bisection while
keeping a sign-changing bracket `[a,b]`.

- Domain: continuous `f` with `f(a)f(b) <= 0`.
- Meaning: guaranteed bracketed convergence with superlinear local speed.
- Code: `optimization.brent_root`.
- Source: [REF-BRENT-1973, Chapter 4].

## Order statistics and shape

### EQ-STAT-QUANT-001 — Sample quantile (Hyndman-Fan types 1 and 7)

Type 7: `Q(p) = x_(floor(h)) + (h - floor(h))(x_(floor(h)+1) - x_(floor(h)))`
with zero-based `h = (n-1)p`. Type 1: `Q(p) = x_(ceil(np))` (inverse ECDF).

- Domain: non-empty finite sample and `p in [0,1]`.
- Meaning: explicit, reproducible quantile convention; type 7 matches the
  default of R and NumPy.
- Code: `statistics.quantile`, `statistics.quantiles`, `statistics.median`,
  `propagation.OutputDistribution.quantile`.
- Source: [REF-HYNDMAN-FAN-1996].

### EQ-STAT-MAD-001 — Median absolute deviation

`MAD = c median_i |x_i - median(x)|`, with `c = 1/Phi^-1(3/4) ~= 1.4826` for
normal consistency.

- Domain: non-empty finite sample.
- Meaning: robust scale with a 50% breakdown point.
- Code: `statistics.median_absolute_deviation`.
- Source: [REF-HAMPEL-1974].

### EQ-STAT-SHAPE-001 — Moment skewness and excess kurtosis

`g1 = m3 / m2^(3/2)`, `g2 = m4 / m2^2 - 3`, with central moments
`m_k = n^-1 sum (x_i - xbar)^k`.

- Domain: at least two observations with positive `m2`.
- Meaning: asymmetry and tail weight relative to a normal law; these are the
  biased moment estimators.
- Code: `statistics.skewness`, `statistics.excess_kurtosis`.
- Source: [REF-JOANES-GILL-1998].

### EQ-STAT-RANK-001 — Mid-ranks and Spearman correlation

`r_i` = one-based position in the sorted sample, ties averaged;
`rho_S = Pearson(r(x), r(y))`.

- Domain: paired finite samples of equal length with non-constant ranks.
- Meaning: monotone association robust to marginal transformations; used for
  input-output screening after propagation.
- Code: `statistics.ranks`, `statistics.spearman_correlation`,
  `propagation.PropagationStudy.rank_correlations`.
- Source: [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 8].

## Model-based sensitivity analysis

### EQ-SENS-BOOT-001 — Bootstrap percentile intervals for Sobol indices

Resample the design rows `k = 1..N` with replacement `B` times, recompute
[EQ-SENS-FIRST-001] and [EQ-SENS-TOTAL-001] on each resample from the stored
outputs `f(A)`, `f(B)`, `f(A_B^(i))`, and report the `(1-c)/2` and `(1+c)/2`
quantiles of the replicates.

- Domain: independent design rows; percentile intervals assume the bootstrap
  distribution approximates the sampling distribution, which is weakest for
  indices near zero and for strongly skewed outputs.
- Meaning: quantifies Monte Carlo error of the indices at no extra model cost.
- Code: `sensitivity.sobol_analysis`.
- Source: [REF-ARCHER-SALTELLI-SOBOL-1997] and [REF-EFRON-TIBSHIRANI-1993].

### EQ-SENS-MORRIS-001 — Morris elementary effects

`EE_i = [f(x + Delta e_i) - f(x)] / Delta`, `mu*_i = mean |EE_i|`,
`sigma_i = sd(EE_i)` over `r` one-at-a-time trajectories on a `p`-level grid
with `Delta = p / (2(p-1))` in the unit (probability) scale.

- Domain: independent inputs; effects are measured on the probability scale
  of each input.
- Meaning: cheap screening (`r(d+1)` runs); large `mu*` means influential,
  large `sigma` means nonlinear or interacting.
- Code: `sensitivity.morris_screening`.
- Source: [REF-MORRIS-1991] and [REF-CAMPOLONGO-CARIBONI-SALTELLI-2007].

## Quantitative risk analysis

### EQ-RISK-DEF-001 — Risk as scenarios, frequencies, and consequences

`R = {<s_i, f_i, c_i>}`; the scalar summary is expected consequence
`sum_i f_i c_i`.

- Domain: exhaustive, mutually exclusive scenarios `s_i` with non-negative
  frequencies (or probabilities) `f_i` and finite consequences `c_i`.
- Meaning: risk answers "what can happen, how likely is it, and what are the
  consequences"; the expected value discards the shape that the exceedance
  curve retains.
- Code: `risk.EventSequence.risk`, `risk.EventTree.total_risk`,
  `risk.EventTree.ranked_sequences`.
- Source: [REF-KAPLAN-GARRICK-1981].

### EQ-RISK-EL-001 — Expected loss

`E[L] = sum_i w_i L_i` with normalized weights `w_i >= 0`, `sum_i w_i = 1`
(equal weights `1/n` for plain Monte Carlo).

- Domain: non-empty finite losses; weights non-negative with positive sum.
- Meaning: risk-neutral summary of the loss distribution.
- Code: `risk.expected_loss`, `risk.assess_risk`, `risk.bootstrap_risk_interval`.
- Source: [REF-KAPLAN-GARRICK-1981], [REF-MCNEIL-FREY-EMBRECHTS-2015,
  Chapter 2].

### EQ-RISK-VAR-001 — Value-at-risk (lower quantile)

`VaR_alpha(L) = inf{l : F_L(l) >= alpha}`; for a weighted sample sorted
ascending, the smallest `L_(k)` with `sum_{j<=k} w_(j) >= alpha`; for equal
weights `L_(ceil(n alpha))` (Hyndman-Fan type 1). Cumulative sums are compared
with a `1e-10` tolerance.

- Domain: `0 < alpha < 1`; losses oriented so larger is worse.
- Meaning: loss level not exceeded with probability at least `alpha`; not
  subadditive in general and silent about the size of tail losses.
- Code: `risk.value_at_risk`, `risk.assess_risk`, `risk.bootstrap_risk_interval`.
- Source: [REF-ROCKAFELLAR-URYASEV-2002], [REF-HYNDMAN-FAN-1996].

### EQ-RISK-CVAR-001 — Conditional value-at-risk (expected shortfall)

`CVaR_alpha(L) = min_c {c + E[(L - c)^+] / (1 - alpha)}
= VaR_alpha + E[(L - VaR_alpha)^+] / (1 - alpha)`.

- Domain: `0 < alpha < 1`, `E|L| < infinity`; valid for general laws,
  including atoms and finite samples with ties, where it equals
  `[(F(VaR) - alpha) VaR + E[L 1{L > VaR}]] / (1 - alpha)`.
- Meaning: average of the worst `1 - alpha` probability mass; a coherent
  risk measure.
- Code: `risk.conditional_value_at_risk`, `risk.expected_shortfall`,
  `risk.assess_risk`, `risk.bootstrap_risk_interval`.
- Source: [REF-ROCKAFELLAR-URYASEV-2000], [REF-ROCKAFELLAR-URYASEV-2002],
  [REF-ACERBI-TASCHE-2002], [REF-ARTZNER-ETAL-1999].

### EQ-RISK-EXC-001 — Loss exceedance curve

`G(l) = P(L > l) = sum_i w_i 1{L_i > l}`.

- Domain: finite losses and levels; equality is not an exceedance.
- Meaning: complete tail picture of the loss distribution; for event trees,
  the frequency of consequences above `l` (an F-N style curve).
- Code: `risk.exceedance_probability`, `risk.loss_exceedance_curve`,
  `risk.RiskAssessment.exceedance_probability`,
  `risk.EventTree.consequence_exceedance_frequency`.
- Source: [REF-KAPLAN-GARRICK-1981], [REF-BAKER-BRADLEY-STAFFORD-2021].

### EQ-RISK-RP-001 — Return period and Poisson exceedance probability

`T = 1 / lambda`; `P(N(t) >= 1) = 1 - exp(-lambda t)`.

- Domain: mean annual rate `lambda > 0` (for `T`), exposure `t >= 0`, events
  forming a homogeneous Poisson process.
- Meaning: mean recurrence interval and probability of at least one event in
  `t` years; `T` is not a guaranteed waiting time.
- Code: `risk.return_period`, `risk.poisson_exceedance_probability`,
  `risk.HazardConvolution.return_period`.
- Source: [REF-BAKER-BRADLEY-STAFFORD-2021].

### EQ-RISK-BOOT-001 — Percentile bootstrap interval

Draw `B` resamples `L*_b` of size `n` with replacement; the interval is the
type-7 empirical `(1 - gamma)/2` and `(1 + gamma)/2` quantiles of `{T(L*_b)}`.

- Domain: i.i.d. equally weighted losses, `n >= 2`, `B >= 20`.
- Meaning: nonparametric sampling uncertainty of VaR, CVaR, or expected loss;
  unreliable when `n (1 - alpha)` is small.
- Code: `risk.bootstrap_risk_interval`.
- Source: [REF-EFRON-TIBSHIRANI-1993, Chapter 13].

### EQ-RISK-NORMAL-001 — Normal VaR and CVaR

`VaR_alpha = mu + sigma z_alpha`,
`CVaR_alpha = mu + sigma phi(z_alpha) / (1 - alpha)`, `z_alpha = Phi^-1(alpha)`.

- Domain: `L ~ Normal(mu, sigma^2)`, `sigma > 0`, `0 < alpha < 1`.
- Meaning: closed-form oracle for sample estimators.
- Code: `risk.normal_value_at_risk`, `risk.normal_conditional_value_at_risk`.
- Source: [REF-MCNEIL-FREY-EMBRECHTS-2015, Chapter 2],
  [REF-ROCKAFELLAR-URYASEV-2000].

### EQ-RISK-LOGNORMAL-001 — Lognormal VaR and CVaR

`VaR_alpha = exp(mu + sigma z_alpha)`,
`CVaR_alpha = exp(mu + sigma^2/2) Phi(sigma - z_alpha) / (1 - alpha)`.

- Domain: `ln L ~ Normal(mu, sigma^2)`, `sigma > 0`, `0 < alpha < 1`.
- Meaning: closed-form oracle for right-skewed losses; follows from the
  lognormal partial expectation `E[L 1{L > v}]`.
- Code: `risk.lognormal_value_at_risk`,
  `risk.lognormal_conditional_value_at_risk`.
- Source: [REF-MCNEIL-FREY-EMBRECHTS-2015, Chapter 2].

### EQ-RISK-LOSS-001 — Consequence (loss) functions

Step `c 1{y > t}`; hinge `min(cap, r max(0, y - t))`; piecewise linear
interpolation between knots `(y_k, l_k)` with constant extrapolation; or a
user callable `c(y)`. The loss sample is `L_i = c(g(X_i))`.

- Domain: finite QoI values; finite losses.
- Meaning: maps a physical QoI to the decision-relevant consequence before
  any risk measure is applied.
- Code: `risk.LossFunction`, `risk.step_loss`, `risk.linear_loss`,
  `risk.piecewise_linear_loss`, `risk.custom_loss`, `risk.assess_risk`.
- Source: [REF-KAPLAN-GARRICK-1981].

### EQ-RISK-FT-GATE-001 — Gate probabilities under independence

AND: `prod_j p_j`; OR: `1 - prod_j (1 - p_j)`; k-out-of-n:
`P(sum_j B_j >= k)` for independent Bernoulli `B_j ~ Bern(p_j)` (Poisson-binomial
tail by dynamic programming).

- Domain: statistically independent gate inputs.
- Meaning: bottom-up quantification of a tree without repeated events.
- Code: `risk.and_gate`, `risk.or_gate`, `risk.k_of_n_gate`,
  `risk.FaultTree.top_event_probability`.
- Source: [REF-VESELY-ETAL-1981], [REF-RAUSAND-HOYLAND-2004].

### EQ-RISK-FT-MOCUS-001 — MOCUS minimal cut sets

Start from the row `{TOP}`; replace an AND gate in a row by all its inputs, an
OR gate by one new row per input, and a k-out-of-n gate by one row per
`k`-subset; when rows contain only basic events, remove duplicates and every
set that contains another set (absorption `A + AB = A`).

- Domain: coherent static fault trees (AND, OR, k-out-of-n gates).
- Meaning: the minimal combinations of basic events that cause the top event.
- Code: `risk.FaultTree.minimal_cut_sets`.
- Source: [REF-VESELY-ETAL-1981], [REF-RAUSAND-HOYLAND-2004].

### EQ-RISK-FT-SHANNON-001 — Shannon decomposition on repeated events

`P(TOP) = sum_{x in {0,1}^r} prod_{i<=r} p_i^{x_i} (1 - p_i)^{1 - x_i}
P(TOP | E_R = x)`, where `E_R` are the `r` repeated basic events and the
conditional probability is evaluated bottom-up with EQ-RISK-FT-GATE-001.

- Domain: independent basic events; `r` not above the configured guard.
- Meaning: exact top-event probability; conditioning removes every shared
  event, so remaining sibling subtrees are independent.
- Code: `risk.FaultTree.top_event_probability`, `risk.FaultTree.importance`.
- Source: [REF-RAUSAND-HOYLAND-2004].

### EQ-RISK-FT-IE-001 — Inclusion-exclusion over minimal cut sets

`P(union_j C_j) = sum_{nonempty S} (-1)^{|S|+1} prod_{i in union_{j in S} C_j} p_i`.

- Domain: independent basic events; at most 15 cut sets in this
  implementation.
- Meaning: exact top-event probability from cut sets; cross-check for the
  Shannon method.
- Code: `risk.FaultTree.top_event_probability`.
- Source: [REF-RAUSAND-HOYLAND-2004].

### EQ-RISK-FT-BOUND-001 — Rare-event approximation and min-cut upper bound

`Q_REA = sum_j P(C_j)`, `Q_MCUB = 1 - prod_j (1 - P(C_j))`,
`P(C_j) = prod_{i in C_j} p_i`; for coherent trees
`Q <= Q_MCUB <= Q_REA`.

- Domain: independent basic events, coherent tree.
- Meaning: conservative cut-set approximations, accurate when cut-set
  probabilities are small.
- Code: `risk.FaultTree.rare_event_approximation`,
  `risk.FaultTree.min_cut_upper_bound`, `risk.FaultTree.cut_set_probability`.
- Source: [REF-VESELY-ETAL-1981], [REF-RAUSAND-HOYLAND-2004].

### EQ-RISK-FT-IMP-001 — Component importance measures

Birnbaum `I_B = Q(1_i) - Q(0_i)`; Fussell-Vesely
`I_FV = P(union_{C_j ∋ i} C_j) / Q`; risk achievement worth
`RAW = Q(1_i) / Q`; risk reduction worth `RRW = Q / Q(0_i)`.

- Domain: `Q > 0` for ratios (otherwise reported as `inf` or `nan`).
- Meaning: sensitivity of top risk to an event (Birnbaum), share of risk
  involving it (FV), and risk increase or decrease if it is certain to fail or
  made perfect (RAW, RRW).
- Code: `risk.FaultTree.importance`, `risk.ImportanceMeasures`.
- Source: [REF-BIRNBAUM-1969], [REF-RAUSAND-HOYLAND-2004].

### EQ-RISK-ET-001 — Event-tree sequence frequency

`f_seq = f_IE prod_{k in path} P(branch_k | path before k)`;
`sum_seq f_seq = f_IE`.

- Domain: branch probabilities at each heading sum to one and are conditional
  on the path so far.
- Meaning: frequency of each accident sequence from an initiating event.
- Code: `risk.EventTree.sequences`, `risk.EventTree.total_frequency`,
  `risk.EventTree.outcome_frequencies`, `risk.binary_heading`.
- Source: [REF-RAUSAND-HOYLAND-2004].

### EQ-RISK-FRAG-001 — Lognormal fragility function

`P(F | IM = a) = Phi(ln(a / theta) / beta)` for `a > 0`.

- Domain: median capacity `theta > 0`, dispersion `beta > 0`.
- Meaning: conditional failure probability at a given intensity.
- Code: `risk.LognormalFragility`.
- Source: [REF-SHINOZUKA-ETAL-2000], [REF-BAKER-2015].

### EQ-RISK-FRAG-MLE-001 — Binomial likelihood for fragility fitting

`ln L(theta, beta) = sum_j [z_j ln p_j + (n_j - z_j) ln(1 - p_j)]`,
`p_j = Phi(ln(a_j / theta) / beta)`, maximized over `(ln theta, ln beta)`.

- Domain: positive intensities, at least one failure and one survival, no
  complete separation by intensity, and at least two distinct intensities.
- Meaning: maximum-likelihood fragility parameters from binary or
  multiple-stripe data.
- Code: `risk.fit_lognormal_fragility`.
- Source: [REF-BAKER-2015], [REF-SHINOZUKA-ETAL-2000].

### EQ-RISK-CONV-001 — Risk convolution with a hazard curve

`lambda_F = int P(F | a) |d lambda(a)|
~= sum_i P(F | sqrt(a_i a_{i+1})) (lambda(a_i) - lambda(a_{i+1}))
+ P(F | a_max) lambda(a_max)`.

- Domain: hazard levels strictly increasing and positive; rates non-negative
  and non-increasing; fragility negligible below the first level.
- Meaning: mean annual failure frequency combining hazard and vulnerability;
  the summands disaggregate it by intensity.
- Code: `risk.annual_failure_frequency`, `risk.HazardConvolution`.
- Source: [REF-BAKER-BRADLEY-STAFFORD-2021], [REF-CORNELL-ETAL-2002].

### EQ-RISK-CONV-002 — Closed form for power-law hazard

`lambda(a) = k0 a^-k` and a lognormal fragility give
`lambda_F = k0 theta^-k exp(k^2 beta^2 / 2)`.

- Domain: `k0 > 0`, `k > 0`, power-law hazard over the whole intensity range.
- Meaning: analytic oracle for the numerical convolution.
- Code: `risk.power_law_failure_frequency`.
- Source: [REF-CORNELL-ETAL-2002].

### EQ-RISK-DEC-EU-001 — Expected payoff and Bayes decision

`U(d) = sum_s p_s u(d, s)`; `d* = argmax_d U(d)` (or `argmin` for losses).

- Domain: finite decisions and states, prior probabilities summing to one,
  finite payoffs.
- Meaning: decision that is best on average under the stated beliefs.
- Code: `risk.DecisionProblem.expected_values`,
  `risk.DecisionProblem.optimal_decision`,
  `risk.DecisionProblem.optimal_value`.
- Source: [REF-RAIFFA-SCHLAIFER-1961, Chapter 1].

### EQ-RISK-DEC-EVPI-001 — Expected value of perfect information

`EVPI = sum_s p_s max_d u(d, s) - max_d sum_s p_s u(d, s)` (signs reversed
for losses); `EVPI >= 0`.

- Domain: as for EQ-RISK-DEC-EU-001.
- Meaning: maximum worth of resolving the uncertainty before deciding.
- Code: `risk.DecisionProblem.value_with_perfect_information`,
  `risk.DecisionProblem.expected_value_of_perfect_information`.
- Source: [REF-RAIFFA-SCHLAIFER-1961, Chapter 4].

### EQ-RISK-DEC-EVSI-001 — Preposterior analysis and EVSI

`P(o) = sum_s P(o | s) p_s`, `p(s | o) = P(o | s) p_s / P(o)`,
`EVSI = sum_o P(o) max_d sum_s p(s | o) u(d, s) - max_d sum_s p_s u(d, s)`;
`0 <= EVSI <= EVPI`.

- Domain: discrete experiment with likelihood rows summing to one.
- Meaning: expected gain from observing the experiment before deciding;
  compare with the experiment's cost.
- Code: `risk.DecisionProblem.preposterior`, `risk.PreposteriorAnalysis`.
- Source: [REF-RAIFFA-SCHLAIFER-1961, Chapters 1 and 4].

### EQ-RISK-MATRIX-001 — Risk-matrix classification

`row = #{t in T_L : t <= likelihood}`, `column = #{t in T_S : t <= severity}`,
category `= M[row][column]` for interior edge sets `T_L`, `T_S`.

- Domain: strictly increasing finite edges; a value on an edge falls into the
  higher bin.
- Meaning: qualitative communication only; can rank risks inconsistently
  with their quantitative values.
- Code: `risk.RiskMatrix.cell`, `risk.RiskMatrix.classify`.
- Source: [REF-COX-2008].

## Model-based reliability: FORM, SORM, subset simulation, and systems

### EQ-REL-MLS-001 — Model limit state from a quantity of interest

For `y = g_q(x)` from a probabilistic model and threshold `t`:
`g(x) = t - y` when failure means `y >= t` ("above"), and `g(x) = y - t` when
failure means `y <= t` ("below"). In standard-normal space
`G(u) = g(T^-1(u))` with the Nataf map of [EQ-MOD-NATAF-001].

- Domain: a finite threshold and a model QoI that is finite at every
  evaluated input.
- Meaning: one model and one joint input law define the limit state for every
  reliability method, always with the closed failure convention `g <= 0`
  [EQ-REL-LIMIT-001].
- Code: `reliability.ModelLimitState.margin`, `physical`, `standard`,
  `output_from_margin`.
- Source: [REF-MELCHERS-BECK-2018, Chapter 3].

### EQ-REL-HLRF-001 — Hasofer-Lind-Rackwitz-Fiessler recursion

`u_hlrf = [(grad G(u_k) . u_k - G(u_k)) / ||grad G(u_k)||^2] grad G(u_k)`,
`d_k = u_hlrf - u_k`, `u_{k+1} = u_k + lambda_k d_k`.
Convergence: `|G(u)| <= eps |G(u_0)|` and
`||u - (alpha . u) alpha|| <= eps max(1, ||u||)`.

- Domain: `G` continuously differentiable near the iterates with non-zero
  gradient; plain HL-RF uses `lambda_k = 1`. Gradients are central differences
  [EQ-OPT-FD-001] unless an analytic gradient is supplied.
- Meaning: each step jumps to the design point of the limit state linearized at
  `u_k`; at a fixed point `u` is parallel to the gradient and on `G = 0`, the
  first-order optimality conditions of `min ||u|| s.t. G(u) = 0`.
- Code: `reliability.form_hlrf`.
- Source: [REF-HASOFER-LIND-1974], [REF-RACKWITZ-FIESSLER-1978].

### EQ-REL-HLRF-002 — iHL-RF merit function and Armijo step

`m(u) = ||u||^2 / 2 + c |G(u)|` with
`c = 2 max(||u_k||, ||u_hlrf||) / ||grad G(u_k)||`; `lambda_k` is the largest
`2^-j` with `m(u_k + lambda d_k) <= m(u_k) + 10^-4 lambda (u_k . d_k - c |G(u_k)|)`.

- Domain: as [EQ-REL-HLRF-001]. Because `grad G . d_k = -G(u_k)`, the
  directional derivative of `m` along `d_k` is `u_k . d_k - c|G(u_k)|`, which is
  negative whenever `c > ||u_k|| / ||grad G||` and `u_k` is not a solution.
- Meaning: the HL-RF direction becomes a descent direction of a merit function,
  so step halving removes the oscillation of plain HL-RF on strongly nonlinear
  limit states. After 30 halvings the smallest trial step is accepted.
- Code: `reliability.form_hlrf` with `line_search=True`.
- Source: [REF-ZHANG-DERKIUREGHIAN-1995], [REF-DERKIUREGHIAN-2005].

### EQ-REL-FORM-002 — General FORM index, probability, and importance

At the converged design point `u*`: `alpha = -grad G(u*) / ||grad G(u*)||`,
`beta = alpha . u*`, `P_f ~= Phi(-beta)`, `x* = T^-1(u*)`, importance
factors `alpha_i^2` (summing to one), and for the Nataf map `u = L^-1 z`
the correlated importance vector `gamma = L^-T alpha / ||L^-T alpha||`.

- Domain: a unique dominant design point and a limit surface well
  approximated by its tangent hyperplane; the sign of `beta` is negative when
  the origin lies in the failure domain. Under correlation `u_i` is the i-th
  Cholesky coordinate, so `alpha_i^2` depends on input order while `gamma`
  refers to the physical inputs.
- Meaning: generalizes [EQ-REL-FORM-001] to nonlinear, non-normal, and
  correlated inputs; exact whenever `G = 0` is a hyperplane in `u` (for example
  a lognormal `R - S` margin).
- Code: `reliability.form_hlrf`, `reliability.FORMResult`,
  `reliability.correlated_importance`.
- Source: [REF-HASOFER-LIND-1974], [REF-DERKIUREGHIAN-2005],
  [REF-MELCHERS-BECK-2018, Chapter 5].

### EQ-REL-SORM-001 — Principal curvatures at the design point

Let `R` be orthonormal with last row `alpha` and `T` its first `n-1` rows.
The principal curvatures are the eigenvalues of
`A = T H T' / ||grad G(u*)||`, where `H` is the Hessian of `G` at `u*`.

- Domain: `G` twice differentiable at `u*` with non-zero gradient; `H` is a
  central-difference Hessian [EQ-OPT-FD-002] (`2n^2 + 1` model calls) and the
  eigenvalues come from cyclic Jacobi [EQ-LINALG-005].
- Meaning: near `u*` the limit surface is the paraboloid
  `v_n = beta + (1/2) sum_i kappa_i v_i^2` in rotated coordinates; positive
  `kappa_i` bend it away from the origin and shrink the failure domain.
- Code: `reliability.principal_curvatures`, `reliability.sorm`.
- Source: [REF-BREITUNG-1984], [REF-DERKIUREGHIAN-2005].

### EQ-REL-SORM-002 — Breitung asymptotic SORM formula

`P_f ~= Phi(-beta) prod_{i=1}^{n-1} (1 + beta kappa_i)^(-1/2)`.

- Domain: `beta > 0` and `1 + beta kappa_i > 0` for every curvature;
  otherwise the code raises instead of returning a meaningless value. The
  formula is asymptotically exact as `beta -> infinity` with `beta kappa_i`
  fixed.
- Meaning: corrects the FORM hyperplane by the Gaussian mass of the
  osculating paraboloid.
- Code: `reliability.second_order_probabilities`, `reliability.sorm`.
- Source: [REF-BREITUNG-1984].

### EQ-REL-SORM-003 — Hohenbichler-Rackwitz SORM improvement

`P_f ~= Phi(-beta) prod_i (1 + psi(beta) kappa_i)^(-1/2)`, with
`psi(beta) = phi(beta) / Phi(-beta)`.

- Domain: `beta > 0` and `1 + psi kappa_i > 0`. Since `psi(beta) > beta`, it
  departs further from FORM than [EQ-REL-SORM-002] for moderate `beta`.
- Meaning: replaces `beta` by the inverse Mills ratio, improving accuracy for
  moderate reliability indices.
- Code: `reliability.second_order_probabilities`, `reliability.sorm`.
- Source: [REF-HOHENBICHLER-RACKWITZ-1988].

### EQ-REL-ISDP-001 — Importance sampling centered at the design point

Proposal `h(u) = phi_n(u - c)` with `c = u*` (or a supplied center);
`w(u) = phi_n(u) / h(u) = exp(||c||^2 / 2 - u . c)` and
`P_f ~= N^-1 sum_i I(G(u_i) <= 0) w(u_i)`. For a system proposal, the equal
mixture `h(u) = m^-1 sum_k phi_n(u - c_k)` over component design points gives
`w(u) = phi_n(u) / h(u)`.

- Domain: the proposal covers the failure contribution (always true for a
  Gaussian proposal); efficiency requires the failure mass to concentrate near
  the centers. Distant secondary failure regions are under-sampled and inflate
  the variance.
- Meaning: the specialization of [EQ-REL-IS-001] that reuses FORM to turn a
  rare event into a frequent one under the proposal.
- Code: `reliability.design_point_importance_sampling`,
  `reliability.system_failure_probability` with `method="is"`.
- Source: [REF-MELCHERS-BECK-2018, Chapter 6], [REF-OWEN-2013, Chapter 9].

### EQ-REL-SUBSET-001 — Subset-simulation product of conditional levels

`F_1 ⊃ F_2 ⊃ ... ⊃ F_m = F` with `F_j = {G(u) <= b_j}` and
`P_f = P(F_1) prod_{j=2}^{m} P(F_j | F_{j-1}) ~= p0^(m-1) N_F / N`. The
threshold `b_j` is the midpoint of the `N p0`-th and `(N p0 + 1)`-th smallest
values of `G` at the current level; the run stops when at least `N p0`
samples satisfy `G <= 0`.

- Domain: `N p0` a positive integer dividing `N`; a continuous response near
  the intermediate thresholds (level-0 ties are rejected).
- Meaning: a rare probability becomes a product of frequent conditional
  probabilities, each estimated from about `N` samples.
- Code: `reliability.subset_simulation`.
- Source: [REF-AU-BECK-2001].

### EQ-REL-SUBSET-002 — Component-wise modified Metropolis step

For each coordinate, `xi_k = u_k + s Z` with `Z ~ N(0, 1)` is accepted with
probability `min(1, phi(xi_k) / phi(u_k))`; the candidate vector replaces the
current state only if `G(xi) <= b_j`, otherwise the chain repeats its state.

- Domain: independent standard-normal target restricted to `F_j`; the
  proposal spread `s > 0` controls acceptance and correlation.
- Meaning: generates conditional samples given `F_j` whose stationary law is
  `phi_n(u) I(u in F_j) / P(F_j)`, with acceptance rates that do not collapse
  in high dimension as a full-vector Metropolis step would.
- Code: `reliability.subset_simulation`.
- Source: [REF-AU-BECK-2001].

### EQ-REL-SUBSET-003 — Subset-simulation coefficient of variation

`delta_j^2 = (1 - P_j) / (N P_j) (1 + gamma_j)`,
`gamma_j = 2 sum_{k=1}^{N_s - 1} (1 - k/N_s) rho_j(k)`,
`rho_j(k) = R_j(k)/R_j(0)` with
`R_j(k) = (N - k N_c)^-1 sum_chains sum_l I_l I_{l+k} - P_j^2`;
reported `delta ~= sqrt(sum_j delta_j^2)` and upper value `sum_j delta_j`.

- Domain: chains of length `N_s = 1/p0` started from `N_c = N p0` seeds;
  level 0 has `gamma = 0`. The root-sum-square value assumes uncorrelated
  level estimators and typically understates the true c.o.v.; the sum is the
  fully correlated value.
- Meaning: an approximate a-posteriori error measure that accounts for
  within-chain correlation of the indicator samples.
- Code: `reliability.subset_simulation` (`_level_cov`).
- Source: [REF-AU-BECK-2001].

### EQ-REL-SYS-001 — Series and parallel system failure events

Series: `F_sys = union_k {g_k <= 0} = {min_k g_k <= 0}`.
Parallel: `F_sys = intersection_k {g_k <= 0} = {max_k g_k <= 0}`.
`P_sys = E[I(x in F_sys)]` estimated by crude Monte Carlo [EQ-MC-PROB-001]
or by the mixture importance sampler of [EQ-REL-ISDP-001].

- Domain: all components share one joint input law; each distinct model is
  evaluated once per sample.
- Meaning: the system event is a deterministic Boolean combination of
  component events on the same random input.
- Code: `reliability.system_failure_probability`.
- Source: [REF-MELCHERS-BECK-2018, Chapter 5].

### EQ-REL-SYS-002 — Uni-modal series-system bounds

`max_i P_i <= P_sys <= 1 - prod_i (1 - P_i)`.

- Domain: the upper bound requires positively dependent component events
  (reported through `positively_correlated`, i.e. all `rho_ij >= 0` in the
  first-order model); the lower bound always holds.
- Meaning: brackets a series system from component probabilities alone; wide
  when several components have similar probabilities.
- Code: `reliability.series_system_bounds`.
- Source: [REF-CORNELL-1967], [REF-MELCHERS-BECK-2018, Chapter 5].

### EQ-REL-SYS-003 — Ditlevsen bi-modal series-system bounds

With components ordered by decreasing `P_i`, `P_i = Phi(-beta_i)`,
`rho_ij = alpha_i . alpha_j`, and `P_ij = Phi_2(-beta_i, -beta_j; rho_ij)`:
`P_1 + sum_{i>=2} max(0, P_i - sum_{j<i} P_ij) <= P_sys <=
sum_i P_i - sum_{i>=2} max_{j<i} P_ij`.

- Domain: valid for any dependence given exact `P_i` and `P_ij`; here both
  come from linearized (FORM) components, so the bounds bracket the
  first-order system probability. Bounds depend on ordering.
- Meaning: second-order inclusion-exclusion bounds that are usually much
  narrower than [EQ-REL-SYS-002]; exact for two components.
- Code: `reliability.series_system_bounds`.
- Source: [REF-DITLEVSEN-1979], [REF-MELCHERS-BECK-2018, Chapter 5].

### EQ-REL-BVN-001 — Bivariate standard normal CDF

`Phi_2(h, k; rho) = Phi(h) Phi(k) + (2 pi)^-1 integral_0^{asin rho}
exp(-(h^2 + k^2 - 2 h k sin t) / (2 cos^2 t)) dt` for `|rho| < 0.925`; for
`|rho| >= 0.925` the integral is rewritten in `sqrt(1 - rho^2)` with an
asymptotic expansion of the singular part, and `rho = +-1` is handled
exactly. Special value: `Phi_2(0, 0; rho) = 1/4 + asin(rho) / (2 pi)`.

- Domain: real limits (infinite limits handled explicitly), `rho in [-1, 1]`.
- Meaning: joint failure probability of two linearized components; computed
  with 6, 12, or 20 Gauss-Legendre nodes [EQ-REL-BVN-002] to about `1e-15`
  absolute error.
- Code: `reliability.bivariate_normal_cdf`.
- Source: [REF-GENZ-2004], [REF-DREZNER-WESOLOWSKY-1990].

### EQ-REL-BVN-002 — Gauss-Legendre quadrature by Golub-Welsch

`integral_{-1}^{1} f(t) dt ~= sum_i w_i f(t_i)`, where `t_i` are the
eigenvalues of the symmetric tridiagonal matrix with off-diagonal
`k / sqrt(4k^2 - 1)` and `w_i = 2 v_{1i}^2`.

- Domain: order `1 <= n <= 64`; exact for polynomials of degree `<= 2n - 1`.
- Meaning: deterministic quadrature nodes for the bivariate normal integral,
  computed rather than tabulated.
- Code: `reliability.gauss_legendre`.
- Source: [REF-GOLUB-WELSCH-1969].

### EQ-REL-INV-001 — Inverse FORM threshold on the target sphere

For target `beta_t > 0`, iterate `u_{k+1} = beta_t alpha(u_k)` with
`alpha = -grad G / ||grad G||` from `u_0 = 0` until
`||u_{k+1} - u_k|| <= eps beta_t`; the threshold is `t* = y(u*)`.

- Domain: the threshold enters `g` additively, so the minimizer of `G` on the
  sphere `||u|| = beta_t` does not depend on `t`; convergence of the fixed
  point is reported, not assumed (it can oscillate on concave responses).
- Meaning: the performance-measure form of target reliability: the quantity
  level whose first-order failure index equals `beta_t`.
- Code: `reliability.inverse_form`.
- Source: [REF-DERKIUREGHIAN-ZHANG-LI-1994], [REF-TU-CHOI-PARK-1999].

### EQ-REL-RBDO-001 — Design parameter for a target reliability index

Find `theta` in a sign-changing bracket with
`beta_FORM(theta) - beta_t = 0`, where `theta` is a deterministic entry of
`model.fixed`, using Brent's method [EQ-OPT-BRENT-001].

- Domain: `beta_FORM` continuous in `theta` on the bracket and FORM
  converging at every trial value (otherwise an explicit error is raised).
- Meaning: the reliability-index (RIA) view of a one-parameter
  reliability-based design problem.
- Code: `reliability.design_for_target_beta`.
- Source: [REF-TU-CHOI-PARK-1999], [REF-BRENT-1973, Chapter 4].

### EQ-REL-BETA-001 — Generalized (equivalent) reliability index

`beta_g = -Phi^-1(P_f)`, with `beta_g = +inf` for `P_f = 0` and `-inf` for
`P_f = 1`.

- Domain: any failure probability in `[0, 1]`.
- Meaning: expresses estimates from SORM, sampling, or subset simulation on
  the same scale as the FORM index for comparison.
- Code: `reliability.generalized_reliability_index`,
  `reliability.compare_reliability_methods`.
- Source: [REF-DITLEVSEN-MADSEN-1996].

## Tail-accurate inverse survival functions

### EQ-DIST-ISF-001 — Inverse survival function

`S^-1(q) = F^-1(1-q)`, evaluated without forming `1-q`: Normal
`mu - sigma Phi^-1(q)`; LogNormal `exp(mu - sigma Phi^-1(q))`; Exponential
`-ln(q)/lambda`; Weibull `scale (-ln q)^(1/k)`; Uniform `b - q(b-a)`.

- Domain: `q in [0,1]`; `q = 0` maps to the upper support end.
- Meaning: upper-tail quantiles with full relative accuracy, so the Nataf
  map stays injective for large latent values (FORM at reliability indices
  beyond about 8, where `Phi(z)` rounds to one).
- Code: `distributions.Normal.isf`, `LogNormal.isf`, `Exponential.isf`,
  `Weibull.isf`, `Uniform.isf`, and `modeling.JointInput.from_standard_normal`.
- Source: [REF-NIST-DLMF, Section 7.17] and [REF-PYTHON-STATISTICS].

## Nonlinear calibration and Markov chain Monte Carlo

### EQ-MCMC-MH-001 — Random-walk Metropolis acceptance

Propose `x' = x + L z` with `z ~ N(0, I)` and `L L' = C`; accept with
probability `alpha = min(1, p(x') / p(x))`, i.e. when
`log U < log p(x') - log p(x)` for `U ~ Uniform(0, 1)`.

- Domain: target density `p` known up to a constant; symmetric proposal;
  `log p = -inf` marks states outside the support (always rejected).
- Meaning: the kernel satisfies detailed balance with respect to `p`, so `p`
  is its stationary law; the Hastings ratio reduces to the density ratio
  because the proposal is symmetric.
- Code: `inference.random_walk_metropolis`, `inference.adaptive_metropolis`,
  `inference.mcmc._run_chain`.
- Source: [REF-METROPOLIS-ETAL-1953], [REF-HASTINGS-1970], and
  [REF-ROBERT-CASELLA-2004, Section 7.5].

### EQ-MCMC-AM-001 — Adaptive Metropolis proposal covariance

`C_t = s_d (Cov(x_0, ..., x_t) + eps I_d)` with `s_d = 2.38^2 / d`, the
running covariance updated by Welford's recursion, and `C_t` frozen at the
end of warm-up.

- Domain: `t >= t_0` warm-up iterations, `eps > 0`; only warm-up draws are
  used for adaptation.
- Meaning: learns the scale and orientation of the target; `2.38^2/d` is the
  asymptotically optimal random-walk scaling for Gaussian-like targets.
  Freezing the covariance keeps the retained chain time-homogeneous, so it is
  an ordinary Metropolis chain.
- Code: `inference.adaptive_metropolis`,
  `inverse.CalibrationProblem.sample_posterior`.
- Source: [REF-HAARIO-SAKSMAN-TAMMINEN-2001] and
  [REF-ROBERTS-ROSENTHAL-2001].

### EQ-MCMC-RHAT-001 — Split potential scale reduction factor

Split each of `M` chains into halves (`m = 2M` sequences of length `n`);
`W = mean_j s_j^2`, `B = n/(m-1) sum_j (xbar_j - xbar)^2`,
`var+ = (n-1)/n W + B/n`, `R-hat = sqrt(var+ / W)`.

- Domain: at least four draws per chain; `W > 0`.
- Meaning: compares between- and within-sequence variance; values near one
  are necessary for convergence, values above about 1.01-1.1 signal
  non-mixing or drift.
- Code: `inference.split_rhat`, `inference.summarize_draws`.
- Source: [REF-GELMAN-RUBIN-1992] and [REF-GELMAN-ETAL-2013, Section 11.4].

### EQ-MCMC-RHAT-002 — Rank-normalized and folded split R-hat

Replace pooled split draws by `z = Phi^-1((r - 3/8)/(S + 1/4))` with average
ranks `r`; compute [EQ-MCMC-RHAT-001] on `z` (bulk) and on the rank-normalized
folded draws `|x - median(x)|` (tail); report the maximum.

- Domain: as [EQ-MCMC-RHAT-001]; any continuous or discrete draws.
- Meaning: robust to heavy tails and infinite variance and sensitive to
  chains that differ in scale but not location.
- Code: `inference.rank_normalized_rhat`, `inference.summarize_draws`.
- Source: [REF-VEHTARI-ETAL-2021].

### EQ-MCMC-ESS-001 — Effective sample size by Geyer's initial monotone sequence

`rho_t = 1 - (W - mean_m c_{t,m}) / var+` with biased within-chain
autocovariances `c_{t,m}` and `rho_0 = 1`;
`Gamma_k = rho_{2k} + rho_{2k+1}`, truncated at the first non-positive
`Gamma_k` (lags `t < n - 5`, stopping lag `T`) and made non-increasing;
`tau = max(-1 + 2 sum_{t < T} rho_t + rho_T^+, 1 / log10(M n))` and
`ESS = M n / tau`, with `rho_T^+` the even autocorrelation at the stopping
lag when positive (the improved estimate of the stan-dev `posterior`
package, [REF-BURKNER-ETAL-POSTERIOR]). The lower bound on `tau` keeps the
ESS positive for strongly antithetic chains.

- Domain: `M` equal-length chains of at least four draws with positive
  pooled variance.
- Meaning: number of independent draws carrying the same information about
  the mean; for AR(1) with coefficient `phi` it tends to
  `M n (1 - phi)/(1 + phi)`.
- Code: `inference.effective_sample_size`, `inference.summarize_draws`.
- Source: [REF-GEYER-1992], [REF-VEHTARI-ETAL-2021],
  [REF-BURKNER-ETAL-POSTERIOR], and [REF-GELMAN-ETAL-2013, Section 11.5].

### EQ-MCMC-MCSE-001 — Monte Carlo standard error of a posterior mean

`MCSE(xbar) = s / sqrt(ESS)`, with pooled sample standard deviation `s`.

- Domain: [EQ-MCMC-ESS-001] with finite posterior variance.
- Meaning: simulation error of the posterior-mean estimate, not posterior
  uncertainty.
- Code: `inference.monte_carlo_standard_error`, `inference.summarize_draws`.
- Source: [REF-GEYER-1992] and [REF-VEHTARI-ETAL-2021].

### EQ-LAPLACE-001 — Laplace approximation

`x* = argmax log p(x)`, `H = -grad^2 log p(x*)`, `p(x) ~= N(x*, H^-1)`, and
`log integral p ~= log p(x*) + (d/2) log(2 pi) - (1/2) log det H`.

- Domain: twice-differentiable `log p` with a strict interior local maximum
  (`H` positive definite).
- Meaning: second-order Gaussian approximation of a posterior and of its
  normalizing constant; exact for Gaussian targets.
- Code: `inference.laplace_approximation`, `inference.LaplaceApproximation`,
  `inverse.CalibrationProblem.laplace_posterior`.
- Source: [REF-TIERNEY-KADANE-1986] and [REF-GELMAN-ETAL-2013, Section 4.1].

### EQ-DIST-T-001 — Student-t distribution function

For `t >= 0`, `F_nu(t) = 1 - I_{nu/(nu+t^2)}(nu/2, 1/2) / 2`, and
`F_nu(-t) = 1 - F_nu(t)`; quantiles invert `F_nu` by Brent's method.

- Domain: real `t`, `nu > 0`.
- Meaning: marginal and predictive law of regression quantities when the
  noise variance is integrated out.
- Code: `inference.student_t_cdf`, `inference.student_t_quantile`.
- Source: [REF-NIST-DLMF-BETA, Section 8.17] and
  [REF-CASELLA-BERGER-2002, Chapter 5].

### EQ-BAYES-NIG-001 — Normal-inverse-gamma conjugate update

Prior `sigma^2 ~ IG(a_0, b_0)`, `beta | sigma^2 ~ N(m_0, sigma^2 V_0)`; data
`y = X beta + e`, `e ~ N(0, sigma^2 I_n)`. Posterior
`V_n = (V_0^-1 + X'X)^-1`, `m_n = V_n (V_0^-1 m_0 + X'y)`, `a_n = a_0 + n/2`,
`b_n = b_0 + ((y - X m_n)'(y - X m_n) + (m_n - m_0)' V_0^-1 (m_n - m_0)) / 2`.

- Domain: positive-definite `V_0`, `a_0, b_0 > 0`, finite data.
- Meaning: exact joint posterior of coefficients and unknown noise variance.
- Code: `inference.condition_linear_normal_inverse_gamma`,
  `inference.NormalInverseGamma`.
- Source: [REF-MURPHY-2012, Section 7.6.3] and
  [REF-GELMAN-ETAL-2013, Section 14.2].

### EQ-BAYES-NIG-002 — Normal-inverse-gamma marginals

`beta ~ t_{2a}(m, (b/a) V)`; `sigma^2 ~ IG(a, b)` with
`E[sigma^2] = b/(a-1)` for `a > 1`, mode `b/(a+1)`, and quantiles
`b / G_{1-p}` for unit-rate gamma quantiles `G`.

- Domain: [EQ-BAYES-NIG-001].
- Meaning: coefficient uncertainty inflated (heavier tails) by not knowing
  the noise level.
- Code: `inference.NormalInverseGamma.coefficient_marginal`,
  `inference.NormalInverseGamma.coefficient_interval`,
  `inference.NormalInverseGamma.noise_variance_interval`,
  `inference.NormalInverseGamma.sample`.
- Source: [REF-MURPHY-2012, Section 7.6.3] and
  [REF-GELMAN-ETAL-2013, Appendix A].

### EQ-BAYES-NIG-003 — Normal-inverse-gamma posterior predictive

`y* | y ~ t_{2a}(X* m, (b/a)(I + X* V X*'))`, with component variance
`nu/(nu-2)` times the scale for `nu = 2a > 2`.

- Domain: [EQ-BAYES-NIG-001]; new design rows `X*`.
- Meaning: prediction of future noisy observations with both parameter and
  noise uncertainty.
- Code: `inference.NormalInverseGamma.predictive`,
  `inference.StudentTPredictive`.
- Source: [REF-MURPHY-2012, Section 7.6.3] and
  [REF-GELMAN-ETAL-2013, Section 14.2].

### EQ-CAL-NL-001 — Nonlinear observation model with experimental conditions

`y_jq = g_q(theta; c_j) + delta_jq + eps_jq`, where `c_j` are fixed
experimental conditions merged over the model's fixed inputs.

- Domain: deterministic forward model; independent errors across
  observations; conditions that do not overlap calibration parameters.
- Meaning: one calibration problem may combine many experiments and several
  observed quantities of interest.
- Code: `inverse.Experiment`, `inverse.CalibrationProblem.predictions`,
  `inverse.CalibrationProblem.evaluate`.
- Source: [REF-KENNEDY-OHAGAN-2001] and [REF-SMITH-UQ-2024, Chapter 8].

### EQ-CAL-LIK-001 — Gaussian and Student-t calibration likelihood

With total scale `s^2 = sigma_q^2 + sigma_delta_q^2` and residual `r`:
Gaussian `log p = -0.5 (log(2 pi s^2) + r^2/s^2)`; Student-t
`log p = log G((nu+1)/2) - log G(nu/2) - 0.5 log(nu pi s^2)
- (nu+1)/2 log(1 + r^2/(nu s^2))`; the log-likelihood is the sum over
observations.

- Domain: `sigma_q > 0`, `sigma_delta_q^2 >= 0`, `nu > 0`.
- Meaning: Gaussian noise or a heavy-tailed robust alternative; the
  discrepancy variance is a declared additive model-inadequacy term.
- Code: `inverse.NoiseModel.log_density`,
  `inverse.CalibrationProblem.log_likelihood`.
- Source: [REF-GELMAN-ETAL-2013, Section 17.2],
  [REF-LANGE-LITTLE-TAYLOR-1989], and [REF-KENNEDY-OHAGAN-2001].

### EQ-CAL-POST-001 — Calibration posterior with log-noise parameters

`log p(theta, eta | y) = log L(theta, e^eta) + log pi(theta)
+ sum_q [log p_sigma(e^eta_q) + eta_q] + const`, with `eta_q = log sigma_q`.

- Domain: proper prior `pi` (a `JointInput`), proper noise priors on
  `(0, inf)`.
- Meaning: the `+ eta_q` term is the Jacobian `|d sigma / d eta|`, so the
  log-scale density is the same prior expressed in unconstrained coordinates.
- Code: `inverse.CalibrationProblem.log_prior`,
  `inverse.CalibrationProblem.log_posterior`.
- Source: [REF-GELMAN-ETAL-2013, Section 1.3] and
  [REF-CASELLA-BERGER-2002, Section 4.3].

### EQ-CAL-STD-001 — Posterior in the prior's standard-normal space

For the Nataf map `u = T(theta)` of the prior,
`pi(T^-1(u)) |det dT^-1/du| = phi_d(u)`, so
`p(u, eta | y) proportional to L(T^-1(u), e^eta) phi_d(u)
prod_q p_sigma(e^eta_q) e^eta_q`.

- Domain: continuous prior marginals with a Gaussian copula; `u` in `R^d`.
- Meaning: sampling in `u` needs no explicit prior Jacobian; bounded or skewed
  priors become unbounded and roughly isotropic, which conditions the
  random-walk sampler.
- Code: `inverse.CalibrationProblem.standard_log_posterior`,
  `inverse.CalibrationProblem.to_physical`,
  `inverse.CalibrationProblem.sample_posterior`.
- Source: [REF-LIU-DERKIUREGHIAN-1986] and
  [REF-CASELLA-BERGER-2002, Section 4.3].

### EQ-CAL-WLS-001 — Weighted calibration least squares

`theta_hat = argmin 0.5 sum_i (r_i(theta) / s_i)^2`, `s_i` the total noise
scale of observation `i`; unknown scales are re-estimated as
`s_q^2 = (RSS_q / n_q) n / (n - d)`.

- Domain: more observations than parameters when scales are estimated.
- Meaning: Gaussian maximum-likelihood estimate of `theta` (prior ignored)
  with Gauss-Newton covariance [EQ-OPT-LM-002] at unit weighted variance.
- Code: `inverse.CalibrationProblem.fit_least_squares`.
- Source: [REF-SEBER-WILD-2003, Chapter 2] and [REF-MORE-1978].

### EQ-CAL-FISHER-001 — Fisher-information identifiability spectrum

`F = J_w' J_w` with the Jacobian `J_w` of weighted predictions; analyse
`D F D = Q Lambda Q'` with `D = diag(scales)` (prior standard deviations by
default); `kappa = lambda_max / lambda_min`; eigenvectors with small
`lambda` are weakly identified parameter combinations.

- Domain: local linearization at a parameter point; Gaussian errors.
- Meaning: rank deficiency means non-identifiability; a large `kappa` means
  practical (weak) identifiability; prior-scaled eigenvalues compare data
  information with prior information.
- Code: `inverse.fisher_identifiability`,
  `inverse.CalibrationProblem.identifiability`,
  `inverse.CalibrationProblem.fit_least_squares`.
- Source: [REF-SEBER-WILD-2003, Section 3.4], [REF-TARANTOLA-2005, Chapter 3],
  and [REF-GOLUB-VANLOAN-2013, Section 8.5].

### EQ-CAL-PRED-001 — Posterior push-forward prediction

Latent: `y*_s = g_q(theta^(s); c*)`; observed:
`y*_s + e_s`, `e_s ~ s_q^(s) N(0,1)` or `s_q^(s) t_nu`, with
`s_q^(s)^2 = sigma_q^(s)^2 + sigma_delta_q^2`; quantiles of the samples give
predictive intervals.

- Domain: posterior draws `theta^(s)` (and `sigma^(s)` when calibrated) and
  declared prediction conditions `c*`.
- Meaning: separates parameter uncertainty (latent) from measurement noise
  (observed), closing the calibration-to-prediction loop; extrapolation in
  `c*` is computable but not validated by the calibration data.
- Code: `inverse.predict`, `inverse.PredictedQuantity`.
- Source: [REF-GELMAN-ETAL-2013, Section 6.3] and
  [REF-SMITH-UQ-2024, Chapter 8].

## State estimation, data assimilation, and surrogate models

### EQ-FILT-SSM-001 — Linear-Gaussian state-space model

`x_k = F x_(k-1) + B u_k + w_k`, `w_k ~ N(0,Q)`; `y_k = H x_k + v_k`,
`v_k ~ N(0,R)`; `x_0 ~ N(m_0,P_0)`, with `w`, `v`, and `x_0` mutually
independent.

- Domain: `Q` and `P_0` symmetric positive semidefinite, `R` symmetric
  positive definite, conformable `F`, `B`, `H`; observations `y_1..y_T`
  follow the transitions.
- Meaning: `Q` is process (dynamics) uncertainty that inflates predictions;
  `R` is measurement uncertainty that limits how far one observation can move
  the estimate.
- Code: `filtering.LinearGaussianStateSpace`, `filtering.simulate`.
- Source: [REF-KALMAN-1960] and [REF-SARKKA-2013, Chapter 4].

### EQ-FILT-SSM-002 — Nonlinear additive-noise state-space model

`x_k = f(x_(k-1)) + w_k`, `y_k = h(x_k) + v_k`, with Gaussian `w_k`, `v_k`,
and `x_0` as in [EQ-FILT-SSM-001].

- Domain: measurable `f` and `h`; differentiable where an extended filter
  linearizes them.
- Meaning: the common model consumed by extended, unscented, ensemble, and
  particle methods.
- Code: `filtering.NonlinearStateSpace`, `filtering.simulate`,
  `data_assimilation.AdditiveGaussianModel`.
- Source: [REF-SARKKA-2013, Chapters 4-5].

### EQ-FILT-BAYES-001 — Sequential Bayesian filtering recursion

Prediction (Chapman-Kolmogorov):
`p(x_k|y_1:k-1) = integral p(x_k|x_(k-1)) p(x_(k-1)|y_1:k-1) dx_(k-1)`;
update (Bayes): `p(x_k|y_1:k) ∝ p(y_k|x_k) p(x_k|y_1:k-1)`.

- Domain: Markov states and observations conditionally independent given the
  current state.
- Meaning: every filter in the library is an exact or approximate
  implementation of this two-step recursion; smoothing additionally conditions
  on future data.
- Code: `filtering.kalman_filter`, `filtering.bootstrap_particle_filter`,
  `data_assimilation.ensemble_kalman_filter`.
- Source: [REF-SARKKA-2013, Chapter 4].

### EQ-FILT-KF-001 — Kalman prediction (covariance propagation)

`m_k^- = F m_(k-1) + B u_k`, `P_k^- = F P_(k-1) F' + Q`.

- Domain: [EQ-FILT-SSM-001].
- Meaning: the Gaussian belief is pushed through the linear dynamics; the
  congruence `F P F'` propagates covariance and `Q` adds new uncertainty.
- Code: `filtering.kalman_predict`, `filtering.kalman_filter`.
- Source: [REF-KALMAN-1960] and [REF-SARKKA-2013, Chapter 4].

### EQ-FILT-KF-002 — Kalman measurement update

`v_k = y_k - H m_k^-`, `S_k = H P_k^- H' + R`, `K_k = P_k^- H' S_k^-1`,
`m_k = m_k^- + K_k v_k`.

- Domain: [EQ-FILT-SSM-001] with positive-definite `S_k`.
- Meaning: Gaussian conditioning on one new observation; the gain weighs the
  innovation by the ratio of prediction to innovation uncertainty. The gain
  is computed by Cholesky solves, never by explicit inversion.
- Code: `filtering.kalman_update`, `filtering.linear_gaussian_update`,
  `filtering.gain_from_cross_covariance`.
- Source: [REF-KALMAN-1960] and [REF-SARKKA-2013, Chapter 4].

### EQ-FILT-KF-003 — Joseph-form covariance update

`P_k = (I - K_k H) P_k^- (I - K_k H)' + K_k R K_k'`.

- Domain: any gain `K_k`; equals `(I - K_k H) P_k^-` for the optimal gain.
- Meaning: a sum of congruences that stays symmetric positive semidefinite
  under rounding and under suboptimal gains.
- Code: `filtering.linear_gaussian_update`.
- Source: [REF-SIMON-2006, Chapter 5] and
  [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Chapter 5].

### EQ-FILT-KF-004 — Innovation-form log marginal likelihood

`log p(y_1:T) = sum_k [-(m/2) log 2pi - (1/2) log det S_k - (1/2) v_k' S_k^-1 v_k]`.

- Domain: [EQ-FILT-SSM-001]; missing observations contribute zero.
- Meaning: the prediction-error decomposition used for parameter estimation
  and model comparison; exact for linear-Gaussian models and a linearized
  approximation for EKF/UKF.
- Code: `filtering.KalmanFilterResult.log_likelihood`,
  `filtering.gaussian_log_density`.
- Source: [REF-SARKKA-2013, Chapter 12].

### EQ-FILT-KF-005 — Discrete Riccati steady state

`P^- = F (P^- - P^- H' (H P^- H' + R)^-1 H P^-) F' + Q`, iterated from
`F P_0 F' + Q`; `K = P^- H' (H P^- H' + R)^-1`. For the scalar random walk
(`F = H = 1`) the fixed point is `P^- = (q + sqrt(q^2 + 4qr))/2`.

- Domain: time-invariant model satisfying the usual detectability and
  stabilizability conditions.
- Meaning: the gain and covariance a long-running filter settles to;
  non-convergence is reported, not raised.
- Code: `filtering.steady_state_kalman`.
- Source: [REF-SIMON-2006, Chapter 7] and
  [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Chapter 5].

### EQ-FILT-RTS-001 — Rauch-Tung-Striebel smoother

`G_k = C_(k+1) (P_(k+1)^-)^-1` with `C_(k+1) = Cov(x_k, x_(k+1)|y_1:k)`
(`= P_k F'` for linear dynamics), `m_k^s = m_k + G_k (m_(k+1)^s - m_(k+1)^-)`,
`P_k^s = P_k + G_k (P_(k+1)^s - P_(k+1)^-) G_k'`.

- Domain: positive-definite predicted covariances.
- Meaning: the backward pass that yields `p(x_k|y_1:T)`; it uses future data
  and so never has larger marginal variance than the filter. The same
  recursion smooths EKF (Jacobian) and UKF (sigma-point) outputs.
- Code: `filtering.rts_smoother`.
- Source: [REF-RAUCH-TUNG-STRIEBEL-1965] and [REF-SARKKA-2013, Chapters 8-9].

### EQ-FILT-EKF-001 — Extended Kalman filter linearization

`m_k^- = f(m_(k-1))`, `P_k^- = F_x P_(k-1) F_x' + Q` with
`F_x = df/dx(m_(k-1))`; update [EQ-FILT-KF-002]/[EQ-FILT-KF-003] with
`y_hat = h(m_k^-)` and `H_x = dh/dx(m_k^-)`.

- Domain: [EQ-FILT-SSM-002] with differentiable maps; Jacobians analytic or by
  central differences [EQ-OPT-FD-001].
- Meaning: first-order Taylor approximation of the nonlinear model; exact for
  linear models, biased when curvature matters on the scale of `P`.
- Code: `filtering.extended_kalman_filter`,
  `filtering.NonlinearStateSpace.transition_jacobian`,
  `filtering.NonlinearStateSpace.observation_jacobian`.
- Source: [REF-SARKKA-2013, Chapter 5] and [REF-SIMON-2006, Chapter 13].

### EQ-FILT-UKF-001 — Scaled unscented transform

`lambda = alpha^2 (n + kappa) - n`; sigma points `chi_0 = m`,
`chi_(i), chi_(n+i) = m +/- sqrt(n + lambda) L_i` with `P = L L'`;
`W_0^m = lambda/(n+lambda)`, `W_0^c = W_0^m + 1 - alpha^2 + beta`,
`W_i^m = W_i^c = 1/(2(n+lambda))`; `mu = sum W_i^m g(chi_i)`,
`Sigma = sum W_i^c (g(chi_i)-mu)(g(chi_i)-mu)'`,
`C = sum W_i^c (chi_i - m)(g(chi_i)-mu)'`.

- Domain: `n + lambda > 0` and positive-semidefinite `P`.
- Meaning: deterministic moment matching; exact mean and covariance for linear
  `g`, exact mean for quadratic `g`.
- Code: `filtering.UnscentedWeights`, `filtering.unscented_transform`.
- Source: [REF-JULIER-UHLMANN-2004] and [REF-SARKKA-2013, Chapter 5].

### EQ-FILT-UKF-002 — Unscented Kalman filter step

Prediction: [EQ-FILT-UKF-001] on `f` plus `Q`. Update: [EQ-FILT-UKF-001] on
`h` gives `mu`, `S = Sigma + R`, `C`; `K = C S^-1`, `m = m^- + K(y - mu)`,
`P = P^- - K S K'`.

- Domain: additive noise; negative `W_0^c` can make covariances indefinite,
  which raises instead of being repaired.
- Meaning: a derivative-free Gaussian filter for nonlinear models.
- Code: `filtering.unscented_kalman_filter`, `filtering.unscented_update`.
- Source: [REF-JULIER-UHLMANN-2004] and [REF-SARKKA-2013, Chapter 5].

### EQ-FILT-PF-001 — Bootstrap particle filter

`x_k^i = f(x_(k-1)^i) + w_k^i`, `w_k^i ~ N(0,Q)`;
`W_k^i ∝ W_(k-1)^i N(y_k; h(x_k^i), R)`;
`p_hat(y_k|y_1:k-1) = sum_i W_(k-1)^i N(y_k; h(x_k^i), R)` with normalized
previous weights.

- Domain: [EQ-FILT-SSM-002]; any posterior shape.
- Meaning: sequential importance sampling with the transition prior as the
  proposal; the product of the increments is an unbiased likelihood estimate
  whose logarithm is biased downward.
- Code: `filtering.bootstrap_particle_filter`,
  `filtering.ParticleFilterResult.log_likelihood`.
- Source: [REF-GORDON-SALMOND-SMITH-1993] and [REF-DOUCET-JOHANSEN-2011].

### EQ-FILT-PF-002 — Effective sample size

`ESS = 1 / sum_i (W^i)^2` for normalized weights.

- Domain: non-negative weights with positive sum; `1 <= ESS <= N`.
- Meaning: weight-degeneracy diagnostic; resampling runs when
  `ESS < threshold * N`.
- Code: `filtering.effective_sample_size`.
- Source: [REF-DOUCET-JOHANSEN-2011].

### EQ-FILT-PF-003 — Systematic resampling

With one `U ~ Uniform[0,1)`, ancestor `j` is the smallest `i` with
`sum_(l<=i) W^l > (U + j)/N`, `j = 0..N-1`.

- Domain: normalized weights.
- Meaning: each particle receives `floor(N W^i)` or `ceil(N W^i)` copies with
  expected count `N W^i`, with less resampling noise than multinomial draws.
- Code: `filtering.systematic_resample`.
- Source: [REF-KITAGAWA-1996] and [REF-DOUCET-JOHANSEN-2011].

### EQ-DA-ENKF-001 — Stochastic ensemble Kalman analysis

`C_xy = (N-1)^-1 sum_i (x_i^f - xbar)(h(x_i^f) - hbar)'`,
`C_yy = (N-1)^-1 sum_i (h(x_i^f) - hbar)(h(x_i^f) - hbar)'`,
`K = C_xy (C_yy + R)^-1`, `x_i^a = x_i^f + K (y + e_i - h(x_i^f))`,
`e_i ~ N(0,R)`.

- Domain: at least two members; `C_yy + R` positive definite.
- Meaning: Monte Carlo Kalman update; perturbing the observations keeps the
  analysis spread consistent with the Kalman covariance in expectation. For
  linear `h` and `N -> infinity` it converges to the Kalman filter.
- Code: `data_assimilation.ensemble_analysis`,
  `data_assimilation.ensemble_forecast`,
  `data_assimilation.ensemble_kalman_filter`.
- Source: [REF-EVENSEN-1994], [REF-BURGERS-VANLEEUWEN-EVENSEN-1998], and
  [REF-EVENSEN-2003].

### EQ-DA-ENKF-002 — Ensemble moments and multiplicative inflation

`xbar = N^-1 sum_i x_i`, `P_e = (N-1)^-1 sum_i (x_i - xbar)(x_i - xbar)'`;
inflation `x_i <- xbar + sqrt(lambda)(x_i - xbar)` gives
`P_e <- lambda P_e`, `lambda >= 1`.

- Domain: at least two members.
- Meaning: `P_e` has rank at most `N - 1` and relative error of order
  `N^-1/2`; inflation is an explicit, user-declared counter-measure against the
  resulting under-dispersion.
- Code: `data_assimilation.Ensemble.mean`, `data_assimilation.Ensemble.covariance`,
  `data_assimilation.inflate`.
- Source: [REF-EVENSEN-2003].

### EQ-DA-3DVAR-001 — Three-dimensional variational cost

`J(x) = (1/2)(x - x_b)' B^-1 (x - x_b) + (1/2)(y - h(x))' R^-1 (y - h(x))`,
`grad J = B^-1 (x - x_b) - H_x' R^-1 (y - h(x))`; analysis covariance
`(B^-1 + H_x' R^-1 H_x)^-1`.

- Domain: positive-definite `B` and `R`; differentiable `h`.
- Meaning: the MAP state of a Gaussian background and Gaussian observation
  error; equals one Kalman update for linear `h`, but `B` is a static input
  rather than a propagated covariance.
- Code: `data_assimilation.three_dimensional_variational`.
- Source: [REF-LORENC-1986].

### EQ-SUR-PCE-001 — Polynomial chaos expansion

`Y ~= sum_(alpha in A) c_alpha Psi_alpha(xi)`,
`Psi_alpha(xi) = prod_i psi_(alpha_i)(xi_i)`, with
`E[Psi_alpha Psi_beta] = delta_(alpha beta)` under independent `xi`.
`xi = u` (Nataf standard normals, [EQ-MOD-NATAF-001]) for the Hermite basis;
`xi_i = (2 x_i - a_i - b_i)/(b_i - a_i)` for independent `Uniform(a_i,b_i)`
inputs and the Legendre basis.

- Domain: square-integrable output; Legendre requires independent uniform
  inputs.
- Meaning: spectral surrogate whose coefficients encode moments and variance
  decomposition.
- Code: `surrogates.PolynomialChaosExpansion.predict`,
  `surrogates.PolynomialChaosExpansion.basis_values`.
- Source: [REF-XIU-KARNIADAKIS-2002] and [REF-SUDRET-2008].

### EQ-SUR-PCE-002 — Orthonormal probabilists' Hermite recurrence

`psi_0 = 1`, `psi_1 = x`, `psi_(n+1) = (x psi_n - sqrt(n) psi_(n-1))/sqrt(n+1)`,
i.e. `psi_n = He_n / sqrt(n!)`.

- Domain: real `x`; orthonormal under `N(0,1)`.
- Meaning: Wiener-Askey basis for Gaussian variables.
- Code: `surrogates.hermite_orthonormal`.
- Source: [REF-XIU-KARNIADAKIS-2002] and [REF-NIST-DLMF-ORTHOGONAL].

### EQ-SUR-PCE-003 — Orthonormal Legendre recurrence

`(n+1) P_(n+1) = (2n+1) x P_n - n P_(n-1)`, `psi_n = sqrt(2n+1) P_n`.

- Domain: `x in [-1,1]`; orthonormal under `Uniform(-1,1)`.
- Meaning: Wiener-Askey basis for uniform variables.
- Code: `surrogates.legendre_orthonormal`.
- Source: [REF-XIU-KARNIADAKIS-2002] and [REF-NIST-DLMF-ORTHOGONAL].

### EQ-SUR-PCE-004 — Total-degree and hyperbolic truncation

`A_(p,q) = {alpha in N^d : (sum_i alpha_i^q)^(1/q) <= p}`, `0 < q <= 1`;
`|A_(p,1)| = C(d+p, p)`.

- Domain: `d >= 1`, `p >= 0`.
- Meaning: `q = 1` keeps all terms up to total degree `p`; `q < 1` drops
  high-order interactions first (sparsity-of-effects).
- Code: `surrogates.multi_indices`.
- Source: [REF-BLATMAN-SUDRET-2011].

### EQ-SUR-PCE-005 — Least-squares PCE coefficients

`c = argmin sum_j (y_j - Psi(x_j)'c)^2 = (Psi' Psi)^-1 Psi' y`, solved by a
Cholesky factorization of the information matrix `Psi' Psi`.

- Domain: more design points than basis terms and a positive-definite
  information matrix; `Psi' Psi / N` near the identity indicates a
  well-conditioned design.
- Meaning: non-intrusive regression fit on an experimental design drawn from
  the input law.
- Code: `surrogates.polynomial_chaos_from_samples`,
  `surrogates.fit_polynomial_chaos`, `surrogates.normalized_information_matrix`.
- Source: [REF-BLATMAN-SUDRET-2011] and [REF-SUDRET-2008].

### EQ-SUR-PCE-006 — Closed-form leave-one-out error

`Delta_j = (y_j - y_hat_j)/(1 - h_j)`, `h_j = psi_j' (Psi' Psi)^-1 psi_j`;
`epsilon_LOO = N^-1 sum_j Delta_j^2 / Var_hat(y)`.

- Domain: [EQ-SUR-PCE-005] with `h_j < 1`.
- Meaning: generalization-error estimate from one fit, no extra model runs.
- Code: `surrogates.PolynomialChaosExpansion.loo_error`.
- Source: [REF-BLATMAN-SUDRET-2011].

### EQ-SUR-PCE-007 — PCE moments

`E[Y] = c_0`, `Var[Y] = sum_(alpha != 0) c_alpha^2`.

- Domain: orthonormal basis with `Psi_0 = 1`.
- Meaning: moments of the surrogate obtained without sampling.
- Code: `surrogates.PolynomialChaosExpansion.mean`,
  `surrogates.PolynomialChaosExpansion.variance`.
- Source: [REF-SUDRET-2008].

### EQ-SUR-PCE-008 — Sobol indices from PCE coefficients

`S_u = sum_(alpha in A_u) c_alpha^2 / Var[Y]` with
`A_u = {alpha : alpha_i > 0 iff i in u}`;
`S_Ti = sum_(alpha : alpha_i > 0) c_alpha^2 / Var[Y]`.

- Domain: independent inputs (the PCE terms are then the ANOVA components).
- Meaning: analytic first-, second-, and total-order indices of the surrogate
  at no extra model cost.
- Code: `surrogates.PolynomialChaosExpansion.sobol_indices`.
- Source: [REF-SUDRET-2008].

### EQ-SUR-GP-001 — Squared-exponential ARD kernel

`k(x,x') = sigma_f^2 exp(-r^2/2)`, `r^2 = sum_i ((x_i - x'_i)/ell_i)^2`.

- Domain: `sigma_f^2 > 0`, `ell_i > 0`.
- Meaning: infinitely smooth stationary prior with one length scale per input.
- Code: `surrogates.kernel_value`.
- Source: [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

### EQ-SUR-GP-002 — Matern-5/2 ARD kernel

`k(x,x') = sigma_f^2 (1 + sqrt(5) r + 5 r^2/3) exp(-sqrt(5) r)`.

- Domain: as [EQ-SUR-GP-001].
- Meaning: twice mean-square differentiable prior, often more realistic than
  the squared exponential for simulator outputs.
- Code: `surrogates.kernel_value`.
- Source: [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

### EQ-SUR-GP-003 — GP predictive mean and variance

`K = K_f + sigma_n^2 I = L L'`, `alpha = K^-1 y`;
`mu(x*) = k*' alpha`, `sigma^2(x*) = k(x*,x*) - v'v`, `v = L^-1 k*`
(plus `sigma_n^2` for a new noisy observation); outputs are standardized
before and de-standardized after.

- Domain: positive-definite `K`; `sigma_n^2` includes the explicit nugget.
- Meaning: posterior Gaussian law of the latent function at a new input.
- Code: `surrogates.GaussianProcessRegression.predict_with_variance`,
  `surrogates.GaussianProcessRegression.predict`.
- Source: [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

### EQ-SUR-GP-004 — GP log marginal likelihood and gradient

`log p(y|theta) = -(1/2) y'K^-1 y - (1/2) log det K - (n/2) log 2pi`;
`d/d theta_j = (1/2) tr((alpha alpha' - K^-1) dK/d theta_j)` with
`theta = (log sigma_f^2, log ell_1..d[, log sigma_n^2])`;
`dk/d log ell_i = sigma_f^2 (Delta_i/ell_i)^2 exp(-r^2/2)` (squared
exponential) or `(5/3) sigma_f^2 (Delta_i/ell_i)^2 (1 + sqrt(5) r) exp(-sqrt(5) r)`
(Matern 5/2).

- Domain: positive-definite `K`; log parameters restricted to `|theta_j| <= 25`.
- Meaning: type-II maximum likelihood objective balancing data fit and
  complexity; maximized by BFGS with seeded restarts.
- Code: `surrogates.gaussian_process_log_marginal_likelihood`,
  `surrogates.fit_gaussian_process`.
- Source: [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

### EQ-SUR-GP-005 — GP closed-form leave-one-out

`mu_(-i) = y_i - [K^-1 y]_i / [K^-1]_ii`, `sigma_(-i)^2 = 1/[K^-1]_ii`.

- Domain: fixed hyperparameters and positive-definite `K`.
- Meaning: cross-validation residuals without refitting.
- Code: `surrogates.GaussianProcessRegression.leave_one_out`.
- Source: [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

## Model validation metrics and prediction credibility

### EQ-VAL-AREA-001 — Area validation metric

`d(F, S_n) = integral_{-inf}^{inf} |F(y) - S_n(y)| dy`, where `F` is the
predictive CDF and `S_n` the empirical CDF of the observations. For two step
CDFs with merged breakpoints `z_1 < ... < z_K`,
`d = sum_k |F(z_k) - S_n(z_k)| (z_{k+1} - z_k)` exactly.

- Domain: non-empty finite predictive samples and observations; both CDFs are
  right-continuous step functions.
- Meaning: mismatch between prediction and data in QoI units. If either law is
  a point mass at `c`, `d` equals the mean absolute distance of the other
  sample from `c`; `d = 0` only for identical distributions.
- Code: `validation.metrics.area_metric`.
- Source: [REF-FERSON-OBERKAMPF-GINZBURG-2008], [REF-OBERKAMPF-ROY-2025,
  Chapter 12].

### EQ-VAL-UPOOL-001 — Probability integral transform for u-pooling

`u_ij = F_i(y_ij)`, with `F_i(y) = #{s : y_s^(i) <= y} / S` the predictive
ECDF of experiment `i`.

- Domain: each experiment has its own predictive sample `y_s^(i)`.
- Meaning: maps observations from experiments with different conditions onto
  one common probability scale; under a correct predictive law the pooled `u`
  are approximately `Uniform(0, 1)`.
- Code: `validation.metrics.probability_integral_transform`,
  `validation.metrics.u_pooling`.
- Source: [REF-FERSON-OBERKAMPF-GINZBURG-2008].

### EQ-VAL-UPOOL-002 — Area between pooled u values and the uniform CDF

`d_u = integral_0^1 |S_n(u) - u| du = sum_k integral_{u_(k)}^{u_(k+1)}
|k/n - u| du`, with `u_(0) = 0`, `u_(n+1) = 1`, each piece evaluated in closed
form.

- Domain: `n >= 1` values in `[0, 1]`.
- Meaning: pooled calibration mismatch on the probability scale; `0 <= d_u <=
  1/2`. It is reported together with the KS statistic of [EQ-INF-GOF-KS-001].
- Code: `validation.metrics.uniform_area`, `validation.metrics.u_pooling`.
- Source: [REF-FERSON-OBERKAMPF-GINZBURG-2008].

### EQ-VAL-MAHAL-001 — Standardized residuals and Mahalanobis distance

`r = y - mu`, `z_i = r_i / sqrt(C_ii)`, `D^2 = r' C^-1 r`,
`p = P(chi^2_n >= D^2) = Q(n/2, D^2/2)`.

- Domain: symmetric positive-definite `C` that includes predictive, measurement,
  and any discrepancy variance.
- Meaning: Gaussian-summary consistency check; the chi-square tail is exact
  only for a Gaussian predictive law with covariance `C`. In
  `assess_predictions`, `C = diag(s_i^2 + sigma_m,i^2)` with observations
  treated as conditionally independent.
- Code: `validation.metrics.mahalanobis_metric`,
  `validation.assessment.assess_predictions`.
- Source: [REF-MARDIA-KENT-BIBBY-1979, Chapters 1-3], [REF-NIST-DLMF-GAMMA].

### EQ-VAL-REL-001 — Model reliability metric

`r = P(|Y_pred - Y_obs| < epsilon)`, estimated by
`r_hat = (1/S) sum_s (1/n) sum_i 1(|y_s - y_i| < epsilon)` with conditional
standard error `sd(h_s)/sqrt(S)`, `h_s = (1/n) sum_i 1(|y_s - y_i| < epsilon)`.

- Domain: `epsilon > 0`, at least two predictive samples, at least one
  observation.
- Meaning: probability that model and experiment agree within a decision-
  relevant tolerance; the standard error is conditional on the observations.
- Code: `validation.metrics.reliability_metric`.
- Source: [REF-REBBA-MAHADEVAN-2008], [REF-LIU-CHEN-ARENDT-HUANG-2011].

### EQ-VAL-ASME-001 — V&V 20 comparison error and validation uncertainty

`E = S - D`, `u_val = sqrt(u_num^2 + u_input^2 + u_D^2)`,
`delta_model in [E - k u_val, E + k u_val]`.

- Domain: independent numerical, input, and data error sources expressed as
  standard uncertainties; coverage factor `k > 0`.
- Meaning: an estimate of the model-form error, not a pass/fail test. If
  `|E| <= k u_val` the model error is not resolved beyond the validation
  uncertainty; if `|E| >> u_val`, `delta_model ~ E`.
- Code: `validation.metrics.asme_validation_comparison`,
  `validation.metrics.AsmeComparison`,
  `validation.records.ValidationExperiment.data_standard_uncertainty`.
- Source: [REF-ASME-VV20-2009], [REF-OBERKAMPF-ROY-2025, Chapter 12].

### EQ-VAL-ASME-002 — Numerical bound to standard uncertainty

`u_num = B / sqrt(3)` for an absolute error bound `B` from [EQ-VV-003].

- Domain: the numerical error is modeled as rectangular on `[-B, B]`.
- Meaning: a GUM Type B conversion that lets a conservative verification bound
  enter the V&V 20 root-sum-square; the rectangular assumption is explicit.
- Code: `validation.metrics.budget_standard_uncertainty`,
  `validation.credibility.VerificationEvidence.standard_uncertainty`.
- Source: [REF-JCGM-100-2008, Section 4.3.7], [REF-ASME-VV20-2009].

### EQ-VAL-BF-001 — Validation Bayes factor against a declared alternative

`p_M(y) = (1/S) sum_s N(y; y_s, h^2)`, `log B = sum_i log p_M(y_i) -
sum_i log p_A(y_i)`, with `h = sigma_m` when measurement noise is known and
`h = 0.9 min(s, IQR/1.34) S^(-1/5)` (Silverman) otherwise. The alternative
`p_A` is uniform on `[a, b]`, `N(m, t^2)`, or the inflated
`N(mean(y_s), c^2 (s^2 + h^2))` with `c > 1`.

- Domain: at least two predictive samples; observations conditionally
  independent given the predictive law; the alternative is declared by the
  user and reported.
- Meaning: positive `log B` favors the model's predictive density over the
  alternative; magnitudes are read on the Kass-Raftery `2 log B` scale. The
  value depends on the breadth of the alternative.
- Code: `validation.metrics.validation_bayes_factor`,
  `validation.metrics.ValidationAlternative.log_density`,
  `validation.metrics.silverman_bandwidth`.
- Source: [REF-KASS-RAFTERY-1995], [REF-REBBA-MAHADEVAN-2008],
  [REF-SILVERMAN-1986, Section 3.4.2].

### EQ-VAL-DOM-001 — Validation-domain membership and scaled distance

`x_tilde_j = (x_j - l_j)/(u_j - l_j)`; box: `l <= x <= u`; hull: feasible
`lambda >= 0`, `sum lambda_i = 1`, `sum lambda_i x_tilde_i = x_tilde`
(phase-one simplex optimum zero); distance `delta = min_i ||x_tilde -
x_tilde_i||_2`; extrapolation if outside the hull or `delta > tau`.

- Domain: validation conditions share names; zero-range dimensions require
  equality with the single validated value.
- Meaning: validation evidence applies directly only near tested conditions;
  outside the convex hull a prediction is extrapolation.
- Code: `validation.records.ValidationDomain.check`,
  `validation.records._convex_hull_residual`.
- Source: [REF-OBERKAMPF-ROY-2025, Part IV], [REF-NOCEDAL-WRIGHT-2006,
  Chapter 13].

## Stochastic processes and random fields

### EQ-PROC-MC-001 — Finite Markov chain and transition matrix

`Pr(X_(t+1) = j | X_t = i, X_(t-1), ..., X_0) = P_ij`, with `P_ij >= 0` and
`sum_j P_ij = 1` for every row `i`.

- Domain: finite state space `{0..n-1}`; time-homogeneous transitions; rows
  checked to sum to one within `1e-10`.
- Meaning: the next state depends on the past only through the present
  state; a path is simulated by drawing each next state from row `X_t`.
- Code: `processes.MarkovChain`, `processes.MarkovChain.simulate`.
- Source: [REF-NORRIS-1997, Section 1.1].

### EQ-PROC-MC-002 — Chapman-Kolmogorov and n-step distributions

`P^(m+n) = P^m P^n`; `Pr(X_n = j | X_0 = i) = (P^n)_ij`;
`pi_n = pi_0 P^n` for the row vector of state probabilities.

- Domain: as [EQ-PROC-MC-001]; `n >= 0` with `P^0 = I`.
- Meaning: the law after `n` steps is obtained by repeated matrix
  multiplication; repeated squaring needs `O(log n)` products.
- Code: `processes.MarkovChain.n_step_matrix`,
  `processes.MarkovChain.distribution_after`.
- Source: [REF-NORRIS-1997, Section 1.1].

### EQ-PROC-MC-003 — Stationary distribution

`pi = pi P`, `pi_i >= 0`, `sum_i pi_i = 1`; solved as `(P' - I) pi = 0` with
one balance equation replaced by the normalization.

- Domain: unique exactly when the chain has a single closed communicating
  class; transient states then get `pi_i = 0`.
- Meaning: the law left invariant by one step; for an irreducible aperiodic
  chain it is also the limit of `pi_0 P^n` and the long-run fraction of time
  spent in each state.
- Code: `processes.MarkovChain.stationary_distribution`.
- Source: [REF-NORRIS-1997, Sections 1.7-1.8].

### EQ-PROC-MC-004 — Class structure and period

`i <-> j` iff `(P^m)_ij > 0` and `(P^n)_ji > 0` for some `m, n >= 0`; a class
`C` is closed iff `P_ij = 0` for `i in C`, `j not in C`;
`d(i) = gcd{n >= 1 : (P^n)_ii > 0}`, computed as the gcd of
`level(u) + 1 - level(v)` over class edges `u -> v` from breadth-first levels.

- Domain: finite chains; in a finite chain the closed classes are exactly the
  recurrent classes. The period is a class property; `0` marks a state that
  cannot return to itself.
- Meaning: irreducibility (one class) and aperiodicity (`d = 1`) decide
  whether the stationary law is unique and whether `pi_0 P^n` converges.
- Code: `processes.MarkovChain.communicating_classes`,
  `processes.MarkovChain.closed_classes`, `processes.MarkovChain.period`.
- Source: [REF-NORRIS-1997, Sections 1.2, 1.5, and 1.8].

### EQ-PROC-MC-005 — Fundamental matrix of an absorbing chain

Canonical form `P = [[Q, R], [0, I]]` (transient block `Q`); `N = (I - Q)^-1`,
`B = N R`, `t = N 1`.

- Domain: at least one absorbing state, every closed class a single absorbing
  state, and at least one transient state (so `I - Q` is non-singular).
- Meaning: `N_ij` is the expected number of visits to transient `j` from
  transient `i`; `B_ik` is the probability of absorption in `k`; `t_i` is the
  expected number of steps before absorption.
- Code: `processes.MarkovChain.absorption`, `processes.AbsorptionAnalysis`.
- Source: [REF-KEMENY-SNELL-1976, Chapter III] and [REF-NORRIS-1997,
  Section 1.3].

### EQ-PROC-MC-006 — Mean first passage and mean return times

`m_ij = 1 + sum_(k != j) P_ik m_kj` for `i != j`; `m_jj = 1 / pi_j`.

- Domain: irreducible finite chain.
- Meaning: expected number of steps to reach `j` for the first time from `i`;
  the diagonal is Kac's mean recurrence time.
- Code: `processes.MarkovChain.mean_first_passage_times`.
- Source: [REF-NORRIS-1997, Sections 1.3 and 1.7].

### EQ-PROC-MC-007 — Gambler's ruin (validation reference)

Walk on `{0..N}` with up-probability `p`, `q = 1 - p`, `r = q / p`:
`Pr(hit N | X_0 = i) = (1 - r^i) / (1 - r^N)` and
`E[T | X_0 = i] = i / (q - p) - (N / (q - p)) (1 - r^i) / (1 - r^N)` for
`p != 1/2`; `i / N` and `i (N - i)` for `p = 1/2`.

- Domain: `0 < p < 1`, absorbing barriers `0` and `N`.
- Meaning: closed-form absorption probabilities and expected durations that
  validate [EQ-PROC-MC-005].
- Code: `tests/test_stochastic_processes.py` (validation only).
- Source: [REF-ROSS-2019, Chapter 3] and [REF-NORRIS-1997, Section 1.3].

### EQ-PROC-POIS-001 — Poisson counting process

`N(t) ~ Poisson(lambda t)`: `Pr(N(t) = k) = e^(-lambda t) (lambda t)^k / k!`;
`E[N(t)] = Var[N(t)] = lambda t`; increments over disjoint intervals are
independent.

- Domain: rate `lambda > 0`, `t >= 0`.
- Meaning: events that occur singly, independently, and at a constant
  average rate; equal mean and variance is its signature.
- Code: `processes.PoissonProcess.mean_count`,
  `processes.PoissonProcess.variance_count`, `processes.PoissonProcess.count_pmf`.
- Source: [REF-NORRIS-1997, Section 2.4].

### EQ-PROC-POIS-002 — Exponential inter-arrival construction

`T_k = E_1 + ... + E_k` with independent `E_i ~ Exponential(lambda)`;
`N(t) = max{k : T_k <= t}`.

- Domain: as [EQ-PROC-POIS-001].
- Meaning: exact simulation of event times and counts.
- Code: `processes.PoissonProcess.event_times`,
  `processes.PoissonProcess.sample_counts`, `processes.CompoundPoissonProcess.path`.
- Source: [REF-NORRIS-1997, Section 2.4] and [REF-DEVROYE-1986, Chapter VI].

### EQ-PROC-CPOIS-001 — Compound Poisson moments

`S(t) = sum_(i=1)^(N(t)) J_i`; `E[S(t)] = lambda t E[J]`,
`Var[S(t)] = lambda t E[J^2]`.

- Domain: i.i.d. jumps with finite second moment, independent of `N`.
- Meaning: aggregate loss or damage from randomly arriving shocks; follows
  from the conditional expectation and conditional variance formulas for a
  random sum.
- Code: `processes.CompoundPoissonProcess.mean`,
  `processes.CompoundPoissonProcess.variance`,
  `processes.CompoundPoissonProcess.sample_values`.
- Source: [REF-ROSS-2019, Chapter 7].

### EQ-PROC-BM-001 — Brownian motion with drift

`X_t = x_0 + mu t + sigma W_t`, `W_0 = 0`, independent increments
`W_(t+h) - W_t ~ N(0, h)`; `E[X_t] = x_0 + mu t`, `Var[X_t] = sigma^2 t`.

- Domain: `sigma > 0`, `h > 0`.
- Meaning: exact simulation on a grid sums independent Gaussian increments;
  coarse increments of the same path are block sums of fine ones.
- Code: `processes.BrownianMotion`, `processes.brownian_increments`,
  `processes.sde` convergence studies.
- Source: [REF-GLASSERMAN-2003, Section 3.1] and [REF-HIGHAM-2001].

### EQ-PROC-GBM-001 — Exact solution of geometric Brownian motion

`dS = mu S dt + sigma S dW` has
`S_t = S_0 exp((mu - sigma^2 / 2) t + sigma W_t)`.

- Domain: `S_0 > 0`, `sigma > 0`.
- Meaning: log-normal law at every time; simulation on a grid by multiplying
  exact log-normal factors has no discretization error and gives the
  pathwise reference for strong-error studies.
- Code: `processes.GeometricBrownianMotion.exact`,
  `processes.GeometricBrownianMotion.path`,
  `processes.GeometricBrownianMotion.sample_terminal`,
  `processes.ScalarSDE.geometric_brownian_motion`.
- Source: [REF-GLASSERMAN-2003, Section 3.2] and [REF-HIGHAM-2001].

### EQ-PROC-GBM-002 — Moments of geometric Brownian motion

`E[S_t] = S_0 e^(mu t)`, `Var[S_t] = S_0^2 e^(2 mu t) (e^(sigma^2 t) - 1)`.

- Domain: as [EQ-PROC-GBM-001].
- Meaning: exact targets for sample moments and for weak-error studies.
- Code: `processes.GeometricBrownianMotion.mean`,
  `processes.GeometricBrownianMotion.variance`.
- Source: [REF-GLASSERMAN-2003, Section 3.2].

### EQ-PROC-OU-001 — Exact Ornstein-Uhlenbeck transition

`dX = theta (mu - X) dt + sigma dW`;
`X_(t+h) | X_t ~ N(mu + (X_t - mu) e^(-theta h), sigma^2 (1 - e^(-2 theta h)) / (2 theta))`.

- Domain: `theta > 0`, `sigma > 0`, `h >= 0`.
- Meaning: Gaussian mean-reverting process; the exact update has no
  time-step error at any `h`.
- Code: `processes.OrnsteinUhlenbeck.mean`, `processes.OrnsteinUhlenbeck.variance`,
  `processes.OrnsteinUhlenbeck.path`, `processes.OrnsteinUhlenbeck.sample_terminal`,
  `processes.ScalarSDE.ornstein_uhlenbeck`.
- Source: [REF-GILLESPIE-1996] and [REF-GLASSERMAN-2003, Section 3.3].

### EQ-PROC-OU-002 — Stationary Ornstein-Uhlenbeck law

`X_infinity ~ N(mu, sigma^2 / (2 theta))`,
`Cov(X_t, X_(t+s)) = sigma^2 e^(-theta |s|) / (2 theta)`.

- Domain: as [EQ-PROC-OU-001].
- Meaning: the stationary covariance is the exponential kernel
  [EQ-PROC-KER-002] with length scale `1 / theta`.
- Code: `processes.OrnsteinUhlenbeck.stationary_variance`,
  `processes.OrnsteinUhlenbeck.stationary_covariance`.
- Source: [REF-GILLESPIE-1996].

### EQ-PROC-SDE-001 — Euler-Maruyama scheme

`X_(n+1) = X_n + a(t_n, X_n) h + b(t_n, X_n) dW_n`, `dW_n ~ N(0, h)`.

- Domain: scalar Ito SDE `dX = a dt + b dW`; uniform step `h = T / N`;
  Lipschitz and linear-growth coefficients for the convergence results.
- Meaning: strong order 1/2 and weak order 1.
- Code: `processes.euler_maruyama`, `processes.solve_sde`.
- Source: [REF-KLOEDEN-PLATEN-1992, Section 10.2] and [REF-HIGHAM-2001].

### EQ-PROC-SDE-002 — Milstein scheme

`X_(n+1) = X_n + a h + b dW_n + (1/2) b (db/dx) (dW_n^2 - h)`.

- Domain: as [EQ-PROC-SDE-001] with differentiable `b`.
- Meaning: the Ito-Taylor correction raises the strong order to 1; for
  additive noise (`db/dx = 0`) it coincides with Euler-Maruyama.
- Code: `processes.milstein`, `processes.solve_sde`.
- Source: [REF-KLOEDEN-PLATEN-1992, Section 10.3] and [REF-HIGHAM-2001].

### EQ-PROC-SDE-003 — Strong and weak errors

Strong: `e_s(h) = E|X_N - X(T)| <= C h^gamma`; weak:
`e_w(h) = |E f(X_N) - E f(X(T))| <= C h^beta` for smooth test functions `f`.

- Domain: `X_N` and `X(T)` driven by the same Brownian path for the strong
  error; estimated by Monte Carlo averages with standard errors.
- Meaning: strong error measures pathwise accuracy, weak error accuracy of
  expectations; a weak study is valid only while the bias dominates the
  Monte Carlo standard error.
- Code: `processes.strong_convergence_study`,
  `processes.weak_convergence_study`, `processes.ConvergenceStudy`.
- Source: [REF-KLOEDEN-PLATEN-1992, Sections 9.6-9.7] and [REF-HIGHAM-2001].

### EQ-PROC-SDE-004 — Empirical convergence order

`log e(h) = log C + q log h`; `q` is the ordinary least-squares slope over the
studied step sizes.

- Domain: at least two distinct positive step sizes with positive errors.
- Meaning: observed order reported next to the theoretical order.
- Code: `processes.empirical_order`.
- Source: [REF-HIGHAM-2001] and [REF-MONTGOMERY-PECK-VINING-2021, Chapter 2].

### EQ-PROC-KER-001 — Squared-exponential covariance kernel

`C(x, y) = sigma^2 exp(-r^2 / (2 ell^2))`, `r = ||x - y||`.

- Domain: `sigma^2 > 0`, `ell > 0`, points in any dimension.
- Meaning: infinitely smooth stationary field; very fast KL eigenvalue decay.
- Code: `processes.CovarianceKernel`, `processes.covariance_matrix`.
- Source: [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

### EQ-PROC-KER-002 — Exponential covariance kernel

`C(x, y) = sigma^2 exp(-r / ell)` (Matern with `nu = 1/2`).

- Domain: as [EQ-PROC-KER-001].
- Meaning: continuous but nowhere mean-square differentiable field (the
  Ornstein-Uhlenbeck covariance in 1-D); slow KL eigenvalue decay.
- Code: `processes.CovarianceKernel`, `processes.exponential_kernel_kl_modes`.
- Source: [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

### EQ-PROC-KER-003 — Matern 3/2 covariance kernel

`C(x, y) = sigma^2 (1 + sqrt(3) r / ell) exp(-sqrt(3) r / ell)`.

- Domain: as [EQ-PROC-KER-001].
- Meaning: once mean-square differentiable field.
- Code: `processes.CovarianceKernel`.
- Source: [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

### EQ-PROC-KER-004 — Matern 5/2 covariance kernel

`C(x, y) = sigma^2 (1 + sqrt(5) r / ell + 5 r^2 / (3 ell^2)) exp(-sqrt(5) r / ell)`.

- Domain: as [EQ-PROC-KER-001].
- Meaning: twice mean-square differentiable field.
- Code: `processes.CovarianceKernel`.
- Source: [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

### EQ-PROC-KL-001 — Karhunen-Loeve expansion of a Gaussian field

`H(x) = m + sum_(k>=1) sqrt(lambda_k) xi_k phi_k(x)`, where
`integral_D C(x, y) phi_k(y) dy = lambda_k phi_k(x)`,
`integral_D phi_k phi_l = delta_kl`, and `xi_k` are independent `N(0, 1)`;
the truncated covariance is `C_M(x, y) = sum_(k<=M) lambda_k phi_k(x) phi_k(y)`.

- Domain: continuous covariance on a bounded domain `D` (Mercer's theorem);
  Gaussian field so the uncorrelated coefficients are independent.
- Meaning: the mean-square optimal representation of the field by `M`
  uncorrelated standard variables.
- Code: `processes.KarhunenLoeveExpansion.realize`,
  `processes.KarhunenLoeveExpansion.sample`,
  `processes.KarhunenLoeveExpansion.covariance`, `modeling.RandomFieldInput.field`.
- Source: [REF-GHANEM-SPANOS-1991, Chapter 2] and [REF-LEMAITRE-KNIO-2010,
  Chapter 2].

### EQ-PROC-KL-002 — Nystrom discretization of the KL eigenproblem

`sum_j w_j C(x_i, x_j) phi_k(x_j) = lambda_k phi_k(x_i)`, symmetrized as
`(W^(1/2) C W^(1/2)) psi_k = lambda_k psi_k`, `phi_k = W^(-1/2) psi_k`, so
`sum_i w_i phi_k(x_i)^2 = 1`.

- Domain: quadrature nodes `x_i` with positive weights `w_i`; unit weights
  give the discrete KL (principal components) of the covariance matrix.
- Meaning: replaces the integral operator by a quadrature rule; eigenvalues
  converge to the continuous ones as the rule is refined (only algebraically
  for the non-smooth exponential kernel).
- Code: `processes.karhunen_loeve`, `processes.karhunen_loeve_interval`,
  `processes.KarhunenLoeveExpansion.eigenfunction`.
- Source: [REF-RASMUSSEN-WILLIAMS-2006, Section 4.3] and
  [REF-SUDRET-DERKIUREGHIAN-2000, Chapter 2].

### EQ-PROC-KL-003 — Truncation by explained variance

`integral_D Var H = sum_k lambda_k` (discretely `sum_i w_i C(x_i, x_i)`);
`M = min{m : sum_(k<=m) lambda_k >= alpha sum_k lambda_k}`; pointwise truncated
variance `sum_(k<=M) lambda_k phi_k(x)^2 <= C(x, x)`.

- Domain: `0 < alpha <= 1`; optional cap on `M`; non-positive discrete
  eigenvalues are never retained.
- Meaning: fraction of the integrated field variance represented by the
  retained modes; truncation always under-represents pointwise variance.
- Code: `processes.karhunen_loeve`,
  `processes.KarhunenLoeveExpansion.explained_variance_fraction`,
  `processes.KarhunenLoeveExpansion.total_variance`,
  `processes.KarhunenLoeveExpansion.pointwise_variance`.
- Source: [REF-SUDRET-DERKIUREGHIAN-2000, Chapter 2].

### EQ-PROC-KL-004 — Exact KL eigenpairs of the exponential kernel on [-a, a]

For `C(x, y) = sigma^2 exp(-|x - y| / b)`, `c = 1 / b`:
`lambda = 2 c sigma^2 / (omega^2 + c^2)`; even modes
`phi(x) = cos(omega x) / sqrt(a + sin(2 omega a) / (2 omega))` with
`c - omega tan(omega a) = 0`; odd modes
`phi(x) = sin(omega x) / sqrt(a - sin(2 omega a) / (2 omega))` with
`omega + c tan(omega a) = 0`.

- Domain: `sigma^2, b, a > 0`; even roots `omega a in (k pi, k pi + pi/2)`,
  odd roots `omega a in (k pi - pi/2, k pi)`, solved in tangent-free form.
- Meaning: analytic benchmark for discrete KL solvers; for `a = b = 1`,
  `omega_1 = 0.860334`, `lambda_1 = 1.149310`.
- Code: `processes.exponential_kernel_kl_modes`, `processes.ExponentialKLMode`.
- Source: [REF-GHANEM-SPANOS-1991, Chapter 2].

### EQ-PROC-KL-005 — Nystrom interpolation of eigenfunctions

`phi_k(x) = (1 / lambda_k) sum_i w_i C(x, x_i) phi_k(x_i)`.

- Domain: `lambda_k > 0`; any location `x` in the domain.
- Meaning: extends discrete eigenvectors to continuous eigenfunctions,
  exactly reproducing the nodal values.
- Code: `processes.KarhunenLoeveExpansion.eigenfunction_at`,
  `processes.KarhunenLoeveExpansion.realize_at`, `modeling.RandomFieldInput.field_at`.
- Source: [REF-RASMUSSEN-WILLIAMS-2006, Section 4.3].

### EQ-PROC-QUAD-001 — Interval quadrature rules

`integral_a^b f ~ sum_i w_i f(x_i)`. Gauss-Legendre: nodes are eigenvalues of
the Jacobi matrix with off-diagonal `k / sqrt(4 k^2 - 1)`, weights
`(b - a) v_(0,i)^2` after the affine map; midpoint and trapezoid rules use
uniform nodes; rectangles use tensor products.

- Domain: `b > a`; Gauss-Legendre exact for polynomials of degree `2n - 1`.
- Meaning: weights of the Nystrom KL discretization and of integrals of
  field-dependent model outputs.
- Code: `processes.gauss_legendre`, `processes.interval_rule`,
  `processes.rectangle_rule`.
- Source: [REF-GOLUB-WELSCH-1969] and [REF-DAVIS-RABINOWITZ-1984, Chapters 2
  and 5].

### EQ-MOD-RF-001 — Random-field input map

`theta -> xi = (theta[prefix_1], ..., theta[prefix_M]) -> T(H_M(x_i; xi))`,
with `xi_k ~ N(0, 1)` independent model inputs and optional pointwise
transform `T` (for example `T = exp` for a log-normal field).

- Domain: truncated KL expansion [EQ-PROC-KL-001]; `T` finite on the field
  values.
- Meaning: a forward model consumes a whole random field through `M`
  standard-normal scalar inputs, so the model plugs into propagation,
  reliability, and sensitivity workflows unchanged.
- Code: `modeling.RandomFieldInput`, `modeling.karhunen_loeve_inputs`.
- Source: [REF-GHANEM-SPANOS-1991, Chapter 2] and
  [REF-SUDRET-DERKIUREGHIAN-2000, Chapter 2].

## Classical and resampling inference

### EQ-INF-DIST-NORM-001 — Standard normal distribution and survival functions

`Phi(z) = erfc(-z/sqrt(2))/2`, `1 - Phi(z) = erfc(z/sqrt(2))/2`.

- Domain: finite real `z`.
- Meaning: reference law of z statistics; the complementary error function
  keeps both tails accurate without `1 - Phi` cancellation.
- Code: `inference.standard_normal_cdf`, `inference.standard_normal_sf`.
- Source: [REF-NIST-DLMF, Section 7.2] and [REF-CASELLA-BERGER-2002, Chapter 3].

### EQ-INF-DIST-CHI2-001 — Chi-square upper tail

`P(X >= x) = Q(k/2, x/2)` for `X ~ chi^2_k`.

- Domain: `k > 0`, `x >= 0`.
- Meaning: null reference law of Pearson statistics; the upper regularized
  incomplete gamma function returns small p-values directly.
- Code: `inference.chi_square_sf`.
- Source: [REF-NIST-DLMF-GAMMA, Section 8.2] and
  [REF-CASELLA-BERGER-2002, Chapter 5].

### EQ-INF-DIST-F-001 — F upper tail

`P(F >= f) = I_{d2/(d2 + d1 f)}(d2/2, d1/2)` for `F ~ F(d1, d2)`.

- Domain: `d1, d2 > 0`, `f >= 0`.
- Meaning: null reference law of the ANOVA ratio of mean squares.
- Code: `inference.f_sf`, `inference.one_way_anova`.
- Source: [REF-NIST-DLMF-BETA, Section 8.17] and
  [REF-CASELLA-BERGER-2002, Chapter 5].

### EQ-INF-DIST-NCT-001 — Noncentral t distribution function

`T' = (Z + delta)/sqrt(V/nu)`; with `S = sqrt(V/nu)`,
`P(T' <= t) = integral_0^inf Phi(t s - delta) g_nu(s) ds`,
`g_nu(s) = 2 (nu/2)^(nu/2) s^(nu-1) exp(-nu s^2/2) / Gamma(nu/2)`;
integrated by adaptive Gauss-Legendre quadrature [EQ-INF-DIST-NCT-002] on
`[max(0, 1 - 12/sqrt(nu)), 1 + 12/sqrt(nu)]`.

- Domain: `nu >= 1`, finite `t` and `delta`; `delta = 0` returns the central
  CDF exactly.
- Meaning: law of the t statistic under a shifted mean; drives t-test power.
- Code: `inference.noncentral_t_cdf`.
- Source: [REF-JOHNSON-KOTZ-BALAKRISHNAN-1995] (noncentral t chapter) and
  [REF-DAVIS-RABINOWITZ-1984, Chapter 2].

### EQ-INF-CI-TMEAN-001 — Student-t interval for a normal mean

`x_bar +- t_(n-1, 1-alpha/2) s / sqrt(n)`.

- Domain: `n >= 2` independent normal observations, `s > 0`.
- Meaning: exact `1 - alpha` repeated-sampling coverage for any `n`; closes
  the large-sample caveat of `EQ-INF-CI-NMEAN-001` when `sigma` is estimated.
- Code: `inference.student_t_mean_confidence_interval`,
  `inference.TTestResult.confidence_interval`.
- Source: [REF-STUDENT-1908] and [REF-CASELLA-BERGER-2002, Chapter 9].

### EQ-INF-CI-CP-001 — Clopper-Pearson binomial interval

`[B(alpha/2; x, n - x + 1), B(1 - alpha/2; x + 1, n - x)]`, with beta
quantiles `B` and endpoints `0` at `x = 0`, `1` at `x = n`.

- Domain: integers `0 <= x <= n`, `n >= 1`.
- Meaning: inverts two one-sided exact binomial tests; coverage is at least
  `1 - alpha` (conservative).
- Code: `inference.clopper_pearson_interval`, `inference.binomial_test`.
- Source: [REF-CLOPPER-PEARSON-1934] and [REF-AGRESTI-2013, Chapter 1].

### EQ-INF-T-001 — One-sample and paired t statistic

`t = (x_bar - mu_0)/(s/sqrt(n))`, `df = n - 1`; paired: apply to
`d_i = x_i - y_i`.

- Domain: `n >= 2`, `s > 0`, independent normal observations (differences).
- Meaning: standardized distance of the sample mean from the null value.
- Code: `inference.one_sample_t_test`, `inference.paired_t_test`,
  `inference.tost_equivalence_test`.
- Source: [REF-STUDENT-1908] and [REF-CASELLA-BERGER-2002, Chapter 8].

### EQ-INF-T-002 — Pooled two-sample t statistic

`s_p^2 = ((n1-1)s1^2 + (n2-1)s2^2)/(n1+n2-2)`,
`t = (x_bar_1 - x_bar_2 - Delta_0)/(s_p sqrt(1/n1 + 1/n2))`, `df = n1+n2-2`.

- Domain: independent normal samples with a common variance, `n1, n2 >= 2`.
- Meaning: exact Student test when variances are equal.
- Code: `inference.two_sample_t_test(equal_variances=True)`,
  `inference.pooled_standard_deviation`.
- Source: [REF-STUDENT-1908] and [REF-CASELLA-BERGER-2002, Chapter 8].

### EQ-INF-T-003 — Welch statistic and Welch-Satterthwaite degrees of freedom

`t = (x_bar_1 - x_bar_2 - Delta_0)/sqrt(s1^2/n1 + s2^2/n2)`,
`nu = (s1^2/n1 + s2^2/n2)^2 / ((s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1))`.

- Domain: independent normal samples, `n1, n2 >= 2`, not both constant.
- Meaning: approximate t reference law without assuming equal variances;
  `min(n1, n2) - 1 <= nu <= n1 + n2 - 2`.
- Code: `inference.two_sample_t_test`, `inference.welch_satterthwaite_df`.
- Source: [REF-WELCH-1947] and [REF-SATTERTHWAITE-1946].

### EQ-INF-TOST-001 — Two one-sided tests for equivalence

`t_L = (theta_hat - L)/se` (reject `theta <= L` if `P(T >= t_L) < alpha`),
`t_U = (theta_hat - U)/se` (reject `theta >= U` if `P(T <= t_U) < alpha`);
`p = max(p_L, p_U)`; equivalently the `1 - 2 alpha` t interval lies in
`(L, U)`.

- Domain: `L < U`, `0 < alpha < 1/2`, t-test assumptions.
- Meaning: equivalence is the alternative hypothesis, so a non-significant
  difference test is never used as evidence of equivalence.
- Code: `inference.tost_equivalence_test`.
- Source: [REF-SCHUIRMANN-1987].

### EQ-INF-CHI2-GOF-001 — Pearson goodness-of-fit statistic

`X^2 = sum_i (O_i - n p_i)^2 / (n p_i)`, `df = k - 1 - q` for `q` estimated
parameters; Cohen's `w = sqrt(X^2/n)`.

- Domain: `k >= 2` categories, `p_i > 0`, `sum p_i = 1`.
- Meaning: asymptotically `chi^2_df` under the null; a warning is emitted
  when an expected count is below five.
- Code: `inference.chi_square_goodness_of_fit`.
- Source: [REF-AGRESTI-2013, Chapter 1] and [REF-COCHRAN-1954].

### EQ-INF-CHI2-IND-001 — Pearson independence statistic, Yates, Cramer's V

`E_ij = R_i C_j / n`, `X^2 = sum (O_ij - E_ij)^2 / E_ij`,
`df = (r - 1)(c - 1)`; Yates (2 x 2): `sum (max(0, |O - E| - 1/2))^2 / E`;
`V = sqrt(X^2 / (n (min(r, c) - 1)))`.

- Domain: `r, c >= 2`, all margins positive.
- Meaning: asymptotic test of row/column independence and its effect size.
- Code: `inference.chi_square_independence_test`.
- Source: [REF-AGRESTI-2013, Chapter 3] and [REF-COCHRAN-1954].

### EQ-INF-FISHER-001 — Fisher exact test for a 2 x 2 table

`P(A = a) = C(R_1, a) C(R_2, C_1 - a) / C(n, C_1)` given margins;
two-sided `p = sum_{P(k) <= P(a)(1 + 1e-7)} P(k)`; one-sided tails `A >= a`
or `A <= a`.

- Domain: non-negative integer cells, positive margins.
- Meaning: exact conditional test of independence for small counts.
- Code: `inference.fisher_exact_test`.
- Source: [REF-AGRESTI-2013, Chapter 3].

### EQ-INF-BINOM-001 — Exact binomial test

`P(X = k) = C(n, k) p_0^k (1 - p_0)^(n-k)`; upper `sum_{k >= x}`, lower
`sum_{k <= x}`, two-sided `sum_{P(k) <= P(x)(1 + 1e-7)} P(k)`.

- Domain: `0 <= x <= n`, `0 < p_0 < 1`.
- Meaning: exact null tail of the observed success count.
- Code: `inference.binomial_test`.
- Source: [REF-AGRESTI-2013, Chapter 1] and [REF-CLOPPER-PEARSON-1934].

### EQ-INF-MWU-001 — Mann-Whitney U statistic and null law

`U = R_1 - m(m+1)/2` (mid-ranks); exact counts are the coefficients of
`prod_{i=1}^{m} (1 - q^(n+i))/(1 - q^i)`; normal approximation
`E U = mn/2`, `Var U = (mn/12)(N + 1 - sum(t^3 - t)/(N(N - 1)))`, with a
half-unit continuity correction; rank-biserial `r = 2U/(mn) - 1`.

- Domain: `m, n >= 1`; exact law only without ties.
- Meaning: `U/(mn)` estimates `P(X > Y) + P(X = Y)/2`.
- Code: `inference.mann_whitney_u_test`, `inference.mann_whitney_u_distribution`.
- Source: [REF-MANN-WHITNEY-1947] and
  [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 4].

### EQ-INF-WSR-001 — Wilcoxon signed-rank statistic and null law

Zero differences dropped; `W+ = sum_{d_i > 0} rank(|d_i|)`; exact counts are
the coefficients of `prod_{k=1}^{n} (1 + q^k)`; normal approximation
`E W+ = n(n+1)/4`, `Var W+ = n(n+1)(2n+1)/24 - sum(t^3 - t)/48`, with a
half-unit continuity correction; rank-biserial `(W+ - W-)/(W+ + W-)`.

- Domain: `n >= 1` non-zero differences; exact law only without ties.
- Meaning: tests symmetry of differences about the null location.
- Code: `inference.wilcoxon_signed_rank_test`,
  `inference.signed_rank_distribution`.
- Source: [REF-WILCOXON-1945] and [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 3].

### EQ-INF-ANOVA-001 — One-way ANOVA F statistic

`SS_B = sum_i n_i (x_bar_i - x_bar)^2`, `SS_W = sum_i sum_j (x_ij - x_bar_i)^2`,
`F = (SS_B/(k - 1)) / (SS_W/(N - k))`; `eta^2 = SS_B/(SS_B + SS_W)`.

- Domain: `k >= 2` groups, `N > k`, `SS_W > 0`.
- Meaning: `F ~ F(k - 1, N - k)` under equal normal means with common
  variance.
- Code: `inference.one_way_anova`.
- Source: [REF-CASELLA-BERGER-2002, Chapter 11].

### EQ-INF-BOOT-001 — Nonparametric bootstrap bias and standard error

Resample `x*_b` of size `n` with replacement, `theta*_b = s(x*_b)`;
`bias = mean(theta*) - theta_hat`, `se = sd(theta*)` (divisor `B - 1`).

- Domain: `n >= 2` observations, `B >= 2` replicates.
- Meaning: plug-in estimates of the sampling bias and spread of a statistic.
- Code: `inference.bootstrap`, `inference.BootstrapResult`.
- Source: [REF-EFRON-1979] and [REF-EFRON-TIBSHIRANI-1993, Chapters 6 and 10].

### EQ-INF-BOOT-002 — Percentile, basic, and normal bootstrap intervals

Percentile `[q_(alpha/2), q_(1-alpha/2)]`; basic
`[2 theta_hat - q_(1-alpha/2), 2 theta_hat - q_(alpha/2)]`; normal
`theta_hat - bias +- z_(1-alpha/2) se`, with type-7 replicate quantiles `q`.

- Domain: bootstrap replicates of a smooth statistic.
- Meaning: first-order accurate approximate confidence procedures.
- Code: `inference.bootstrap_confidence_interval`.
- Source: [REF-EFRON-TIBSHIRANI-1993, Chapter 13] and
  [REF-DAVISON-HINKLEY-1997, Chapter 5].

### EQ-INF-BOOT-003 — Studentized (bootstrap-t) interval

`t*_b = (theta*_b - theta_hat)/se*_b`;
`[theta_hat - t*_(1-alpha/2) se, theta_hat - t*_(alpha/2) se]`, with `se`
from a supplied function or the jackknife.

- Domain: positive standard errors for the data and every resample.
- Meaning: second-order accurate for approximately pivotal statistics.
- Code: `inference.bootstrap_confidence_interval(method="studentized")`.
- Source: [REF-EFRON-TIBSHIRANI-1993, Chapter 12] and
  [REF-DAVISON-HINKLEY-1997, Chapter 5].

### EQ-INF-BOOT-004 — Bias-corrected and accelerated (BCa) interval

`z_0 = Phi^-1((#{theta* < theta_hat} + #{theta* = theta_hat}/2)/B)`;
`a = sum (m - theta_(i))^3 / (6 (sum (m - theta_(i))^2)^(3/2))` from
jackknife values `theta_(i)` with mean `m`;
`alpha_j = Phi(z_0 + (z_0 + z_j)/(1 - a(z_0 + z_j)))`, `z_j = Phi^-1` of
`alpha/2` and `1 - alpha/2`; interval `[q_(alpha_1), q_(alpha_2)]`.

- Domain: `0 < share < 1`, non-constant jackknife values,
  `1 - a(z_0 + z_j) > 0`.
- Meaning: transformation-respecting, second-order accurate correction of
  the percentile interval for bias and skewness.
- Code: `inference.bootstrap_confidence_interval(method="bca")`.
- Source: [REF-EFRON-1987] and [REF-EFRON-TIBSHIRANI-1993, Chapter 14].

### EQ-INF-JACK-001 — Jackknife bias and standard error

`theta_(i) = s(x without x_i)`, `theta_(.) = mean theta_(i)`;
`bias = (n - 1)(theta_(.) - theta_hat)`,
`se = sqrt((n - 1)/n sum (theta_(i) - theta_(.))^2)`.

- Domain: `n >= 2`; smooth statistics (unreliable for the median).
- Meaning: deterministic leave-one-out estimates of bias and spread.
- Code: `inference.jackknife`.
- Source: [REF-EFRON-TIBSHIRANI-1993, Chapter 11].

### EQ-INF-PERM-001 — Permutation p-value for a difference in means

`T = mean(x) - mean(y)`; exact `p = #{g : T_g extreme} / C(m+n, m)` over all
relabelings `g`; Monte Carlo `p = (b + 1)/(M + 1)` for `b` of `M` random
relabelings at least as extreme (two-sided uses `|T|`).

- Domain: exchangeable pooled observations under the null.
- Meaning: exact-level test without distributional assumptions; the `+1`
  terms keep the Monte Carlo test valid and the p-value positive.
- Code: `inference.permutation_test`.
- Source: [REF-GOOD-2005] and [REF-PHIPSON-SMYTH-2010].

### EQ-INF-ES-001 — Cohen's d and Hedges' g

`d = (x_bar_1 - x_bar_2)/s_p` (two samples), `(x_bar - mu_0)/s` (one
sample), `d_bar/s_d` (paired); `g = J(m) d`,
`J(m) = Gamma(m/2)/(sqrt(m/2) Gamma((m - 1)/2)) ~ 1 - 3/(4m - 1)`.

- Domain: positive standard deviation; `m > 1` degrees of freedom.
- Meaning: scale-free effect magnitude; `g` removes the small-sample upward
  bias of `|d|`.
- Code: `inference.cohens_d`, `inference.hedges_g`,
  `inference.hedges_correction`.
- Source: [REF-COHEN-1988, Chapter 2] and [REF-HEDGES-1981].

### EQ-INF-POWER-001 — Power of z tests

`delta = d sqrt(n)` (one-sample/paired) or `d sqrt(n/2)` (two groups of `n`);
two-sided `Phi(delta - z_(1-alpha/2)) + Phi(-delta - z_(1-alpha/2))`,
one-sided `Phi(+-delta - z_(1-alpha))`.

- Domain: known variance, standardized effect `d`.
- Meaning: probability of rejecting when the stated effect is true.
- Code: `inference.z_test_power`.
- Source: [REF-COHEN-1988, Chapter 2] and [REF-LEHMANN-ROMANO-2005, Chapter 3].

### EQ-INF-POWER-002 — Exact power of t tests

`df = n - 1` or `2n - 2`; two-sided
`1 - F'_(df,delta)(t_c) + F'_(df,delta)(-t_c)` with `t_c = t_(df,1-alpha/2)`;
one-sided analogues with `t_(df,1-alpha)`; `F'` from `EQ-INF-DIST-NCT-001`.

- Domain: normal observations, `n >= 2`.
- Meaning: exact design power under the noncentral t law (no normal
  approximation, no simulation).
- Code: `inference.t_test_power`.
- Source: [REF-COHEN-1988, Chapter 2] and
  [REF-JOHNSON-KOTZ-BALAKRISHNAN-1995].

### EQ-INF-POWER-003 — Required sample size

`n* = min{n : power(n) >= 1 - beta}`, searched from the z start
`k (z_(1-alpha') + z_(1-beta))^2 / d^2` (`k = 2` for two groups).

- Domain: nonzero `d` whose sign agrees with a one-sided alternative,
  `alpha < 1 - beta < 1`.
- Meaning: smallest design achieving the target power.
- Code: `inference.required_sample_size`.
- Source: [REF-COHEN-1988, Chapter 2].

### EQ-INF-MT-001 — Bonferroni and Holm adjusted p-values

Bonferroni `min(1, m p_i)`; Holm
`p~_(i) = max_{j <= i} min(1, (m - j + 1) p_(j))` for sorted `p_(1) <= ...`.

- Domain: `m >= 1` p-values in `[0, 1]`, arbitrary dependence.
- Meaning: control of the family-wise error rate at `alpha` when rejecting
  `p~ <= alpha`; Holm is uniformly more powerful than Bonferroni.
- Code: `inference.adjust_p_values`.
- Source: [REF-HOLM-1979] and [REF-LEHMANN-ROMANO-2005, Chapter 9].

### EQ-INF-MT-002 — Benjamini-Hochberg and Benjamini-Yekutieli adjusted p-values

BH `p~_(i) = min_{j >= i} min(1, m p_(j)/j)`; BY multiplies by
`c(m) = sum_{k=1}^{m} 1/k`.

- Domain: BH for independent or positively regression-dependent p-values;
  BY for arbitrary dependence.
- Meaning: control of the false discovery rate at `alpha` when rejecting
  `p~ <= alpha`.
- Code: `inference.adjust_p_values`.
- Source: [REF-BENJAMINI-HOCHBERG-1995] and [REF-BENJAMINI-YEKUTIELI-2001].

## Continuous distribution catalog: UQ and reliability families

### EQ-DIST-INV-001 — Tail-aware quantile inversion

For `p <= 1/2` solve `log F(x(t)) = log p`; otherwise solve
`log S(x(t)) = log(1-p)` (and symmetrically for `isf`). The coordinate map
`x(t)` is `exp(t)` on `(0, inf)`, `expit(t)` on `(0, 1)`, `mu + sigma sinh(t)`
on the real line, and an endpoint-anchored map for truncated supports. The
increasing residual is bracketed by geometric expansion and refined by
Brent's zeroin method.

- Domain: continuous strictly increasing CDF on the support interior;
  `0 < p < 1`.
- Meaning: quantiles keep relative accuracy in the tail that carries the
  smaller probability, without forming `1 - p` near one; the coordinate
  maps let far quantiles (for example `1e-120` or `1e24`) be bracketed in a
  few dozen steps. Accuracy is ultimately limited by the spacing of floats
  near a finite support end.
- Code: `distributions._continuous.solve_tail_quantile`,
  `distributions._continuous.invert_increasing`,
  `distributions._continuous.brent_increasing`; used by `Gamma`, `Beta`,
  `StudentT`, `ChiSquare`, `FisherF`, and `TruncatedNormal`.
- Source: [REF-BRENT-1973, Chapter 4].

### EQ-DIST-GAM-001 — Gamma density (shape-rate)

`f(x) = lambda^k x^(k-1) exp(-lambda x) / Gamma(k)` for `x >= 0`;
`log f = k log lambda + (k-1) log x - lambda x - lgamma(k)`.

- Domain: `k > 0`, `lambda > 0`; at `x = 0` the density is `inf` for
  `k < 1`, `lambda` for `k = 1`, and `0` for `k > 1`.
- Meaning: waiting time to the `k`-th Poisson event and the conjugate law
  for rates; `k = 1` is `Exponential(lambda)` and `Gamma(nu/2, 1/2)` is
  `chi2(nu)`.
- Code: `distributions.Gamma.log_pdf`, `distributions.Gamma.pdf`.
- Source: [REF-JKB-1994, Chapter 17] and [REF-NIST-DISTRIBUTIONS].

### EQ-DIST-GAM-002 — Gamma CDF and survival function

`F(x) = P(k, lambda x)` and `S(x) = Q(k, lambda x)`.

- Domain: `x >= 0`.
- Meaning: both tails are regularized incomplete gamma functions evaluated
  through the branch that computes the smaller tail directly.
- Code: `distributions.Gamma.cdf`, `distributions.Gamma.sf`.
- Source: [REF-NIST-DLMF-GAMMA, Section 8.2].

### EQ-DIST-GAM-003 — Gamma quantile

`x_p = F^-1(p)` solved by [EQ-DIST-INV-001] on `t = log x`.

- Domain: `0 < p < 1`; `x_0 = 0`, `x_1 = inf`.
- Meaning: no closed form exists; inversion of `log P` or `log Q` keeps far
  tails accurate.
- Code: `distributions.Gamma.quantile`, `distributions.Gamma.isf`,
  `distributions.ChiSquare.quantile`, `distributions.ChiSquare.isf`.
- Source: [REF-NIST-DLMF-GAMMA, Section 8.10] and [REF-BRENT-1973, Chapter 4].

### EQ-DIST-GAM-004 — Gamma moments

`E[X] = k / lambda` and `Var(X) = k / lambda^2`.

- Domain: `k > 0`, `lambda > 0`.
- Meaning: first two moments in the shape-rate parameterization.
- Code: `distributions.Gamma.mean`, `distributions.Gamma.variance`.
- Source: [REF-JKB-1994, Chapter 17].

### EQ-DIST-GAM-005 — Marsaglia-Tsang gamma variates

For `a >= 1`: `d = a - 1/3`, `c = 1/sqrt(9d)`, `v = (1 + c Z)^3`; accept
`d v` when `U < 1 - 0.0331 Z^4` or
`log U < Z^2/2 + d (1 - v + log v)`. For `a < 1`:
`G_a = G_(a+1) U^(1/a)`, applied in log space.

- Domain: `a > 0`; `Z ~ N(0,1)`, `U ~ Uniform(0,1)` independent.
- Meaning: exact rejection sampler for `Gamma(a, 1)`; scaled by `1/lambda`
  and reused for beta, chi-square, Student-t, and F variates.
- Code: `distributions._continuous.log_standard_gamma_variate`,
  `distributions.Gamma.sample`.
- Source: [REF-MARSAGLIA-TSANG-2000].

### EQ-DIST-BETA-001 — Beta density

`f(x) = x^(a-1) (1-x)^(b-1) / B(a, b)` on `[0, 1]`, with
`log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a+b)`.

- Domain: `a > 0`, `b > 0`; endpoint density is `inf`, `1/B`, or `0` as
  the matching shape is below, equal to, or above one.
- Meaning: law of a bounded fraction or probability; `Beta(1,1)` is
  `Uniform(0,1)`.
- Code: `distributions.Beta.log_pdf`, `distributions.Beta.pdf`.
- Source: [REF-JKB-1995, Chapter 25] and [REF-NIST-DISTRIBUTIONS].

### EQ-DIST-BETA-002 — Beta CDF and survival function

`F(x) = I_x(a, b)` and `S(x) = I_(1-x)(b, a)`.

- Domain: `0 <= x <= 1`.
- Meaning: the survival function uses the symmetry relation directly, so
  small upper tails are not formed as `1 - F`.
- Code: `distributions.Beta.cdf`, `distributions.Beta.sf`; quantiles use
  [EQ-DIST-INV-001] on `t = logit x`.
- Source: [REF-NIST-DLMF-BETA, Section 8.17].

### EQ-DIST-BETA-003 — Beta moments

`E[X] = a/(a+b)` and `Var(X) = ab / ((a+b)^2 (a+b+1))`.

- Domain: `a > 0`, `b > 0`.
- Meaning: mean and variance of the bounded law.
- Code: `distributions.Beta.mean`, `distributions.Beta.variance`.
- Source: [REF-JKB-1995, Chapter 25].

### EQ-DIST-BETA-004 — Beta variates from gamma ratios

`X = G_a / (G_a + G_b) = expit(log G_a - log G_b)` with independent
`G_a ~ Gamma(a,1)`, `G_b ~ Gamma(b,1)`.

- Domain: `a > 0`, `b > 0`.
- Meaning: exact transformation; the log form avoids `0/0` for tiny shapes.
- Code: `distributions.Beta.sample`.
- Source: [REF-DEVROYE-1986, Chapter IX].

### EQ-DIST-STT-001 — Location-scale Student-t density

`f(x) = Gamma((nu+1)/2) / (sqrt(nu pi) sigma Gamma(nu/2))
(1 + z^2/nu)^(-(nu+1)/2)`, `z = (x - mu)/sigma`.

- Domain: `nu > 0`, `sigma > 0`, real `x`.
- Meaning: heavy-tailed symmetric law; `nu = 1` is Cauchy and
  `nu -> inf` is `Normal(mu, sigma)`.
- Code: `distributions.StudentT.log_pdf`, `distributions.StudentT.pdf`.
- Source: [REF-JKB-1995, Chapter 28] and [REF-NIST-DISTRIBUTIONS].

### EQ-DIST-STT-002 — Student-t tails through the incomplete beta function

`P(T > |z|) = I_(nu/(nu+z^2))(nu/2, 1/2) / 2`, evaluated as
`(1 - I_(z^2/(nu+z^2))(1/2, nu/2)) / 2` when `nu/(nu+z^2) >= 1/2`;
`F(x) = P(T > -z)` for `z < 0` and `S(x) = P(T > z)` for `z > 0`.

- Domain: `nu > 0`, real `z` with finite `z^2`.
- Meaning: both incomplete-beta arguments are formed without subtraction
  from one, so the centre and both tails keep relative accuracy.
- Code: `distributions.StudentT.cdf`, `distributions.StudentT.sf`; quantiles
  use [EQ-DIST-INV-001] on `t = asinh(z)`.
- Source: [REF-NIST-DLMF-BETA, Section 8.17] and [REF-JKB-1995, Chapter 28].

### EQ-DIST-STT-003 — Student-t moments

`E[X] = mu` for `nu > 1`; `Var(X) = sigma^2 nu/(nu-2)` for `nu > 2` and
`inf` for `1 < nu <= 2`.

- Domain: undefined mean and variance for `nu <= 1` (raised as errors).
- Meaning: heavy tails remove moments of order `>= nu`.
- Code: `distributions.StudentT.mean`, `distributions.StudentT.variance`.
- Source: [REF-JKB-1995, Chapter 28].

### EQ-DIST-STT-004 — Student-t variates

`T = Z / sqrt(V/nu)` with `Z ~ N(0,1)` and `V = 2 G`, `G ~ Gamma(nu/2, 1)`.

- Domain: `nu > 0`.
- Meaning: exact normal/chi construction of the t law.
- Code: `distributions.StudentT.sample`.
- Source: [REF-DEVROYE-1986, Chapter IX].

### EQ-DIST-CHI2-001 — Chi-square as a gamma law

`chi2(nu) = Gamma(nu/2, 1/2)`: `F(x) = P(nu/2, x/2)`,
`S(x) = Q(nu/2, x/2)`, and variates `2 G`, `G ~ Gamma(nu/2, 1)`.

- Domain: `nu > 0`, `x >= 0`.
- Meaning: law of a sum of `nu` squared standard normals (integer `nu`);
  the reference law for variance and goodness-of-fit statistics.
- Code: `distributions.ChiSquare`.
- Source: [REF-JKB-1994, Chapter 18] and [REF-NIST-DISTRIBUTIONS].

### EQ-DIST-CHI2-002 — Chi-square moments

`E[X] = nu` and `Var(X) = 2 nu`.

- Domain: `nu > 0`.
- Meaning: first two moments.
- Code: `distributions.ChiSquare.mean`, `distributions.ChiSquare.variance`.
- Source: [REF-JKB-1994, Chapter 18].

### EQ-DIST-F-001 — F density

`f(x) = y^(nu1/2) (1-y)^(nu2/2) / (x B(nu1/2, nu2/2))`,
`y = nu1 x / (nu1 x + nu2)`.

- Domain: `nu1 > 0`, `nu2 > 0`, `x > 0`; `f(0) = inf, 1, 0` for
  `nu1 < 2, = 2, > 2`.
- Meaning: law of a ratio of independent scaled chi-squares (variance
  ratios, ANOVA).
- Code: `distributions.FisherF.log_pdf`, `distributions.FisherF.pdf`.
- Source: [REF-JKB-1995, Chapter 27] and [REF-NIST-DISTRIBUTIONS].

### EQ-DIST-F-002 — F CDF and survival function

`F(x) = I_y(nu1/2, nu2/2)` and `S(x) = I_(1-y)(nu2/2, nu1/2)` with
`1 - y = nu2 / (nu1 x + nu2)` formed directly.

- Domain: `x >= 0`.
- Meaning: both tails without subtraction; quantiles use
  [EQ-DIST-INV-001] on `t = log x`.
- Code: `distributions.FisherF.cdf`, `distributions.FisherF.sf`,
  `distributions.FisherF.quantile`, `distributions.FisherF.isf`.
- Source: [REF-NIST-DLMF-BETA, Section 8.17].

### EQ-DIST-F-003 — F moments

`E[X] = nu2/(nu2-2)` for `nu2 > 2` (else `inf`);
`Var(X) = 2 nu2^2 (nu1+nu2-2) / (nu1 (nu2-2)^2 (nu2-4))` for `nu2 > 4`
(else `inf`).

- Domain: `nu1 > 0`, `nu2 > 0`.
- Meaning: positive support makes missing moments infinite rather than
  undefined.
- Code: `distributions.FisherF.mean`, `distributions.FisherF.variance`.
- Source: [REF-JKB-1995, Chapter 27].

### EQ-DIST-F-004 — F variates

`X = (G_1/(nu1/2)) / (G_2/(nu2/2))` with independent `G_i ~ Gamma(nu_i/2, 1)`.

- Domain: `nu1 > 0`, `nu2 > 0`.
- Meaning: exact chi-square ratio construction, evaluated in log space.
- Code: `distributions.FisherF.sample`.
- Source: [REF-DEVROYE-1986, Chapter IX].

### EQ-DIST-LOGIS-001 — Logistic density

`f(x) = exp(-z) / (s (1 + exp(-z))^2)`, evaluated as
`log f = -|z| - log s - 2 log1p(exp(-|z|))`, `z = (x - mu)/s`.

- Domain: `s > 0`, real `x`.
- Meaning: symmetric law with exponential tails heavier than the normal.
- Code: `distributions.Logistic.log_pdf`, `distributions.Logistic.pdf`.
- Source: [REF-JKB-1995, Chapter 23].

### EQ-DIST-LOGIS-002 — Logistic CDF and survival function

`F(x) = expit(z) = 1/(1 + exp(-z))` and `S(x) = expit(-z)`; `expit(t)` is
evaluated as `exp(t)/(1 + exp(t))` for `t < 0`.

- Domain: real `z`.
- Meaning: both tails with relative accuracy and no overflow.
- Code: `distributions.Logistic.cdf`, `distributions.Logistic.sf`,
  `distributions._continuous.expit`.
- Source: [REF-JKB-1995, Chapter 23].

### EQ-DIST-LOGIS-003 — Logistic quantile

`x_p = mu + s (log p - log1p(-p))` and
`S^-1(q) = mu + s (log1p(-q) - log q)`.

- Domain: `0 < p, q < 1`.
- Meaning: closed-form inverse; also the inverse-transform sampler.
- Code: `distributions.Logistic.quantile`, `distributions.Logistic.isf`,
  `distributions.Logistic.sample`.
- Source: [REF-JKB-1995, Chapter 23].

### EQ-DIST-LOGIS-004 — Logistic moments

`E[X] = mu` and `Var(X) = s^2 pi^2 / 3`.

- Domain: `s > 0`.
- Meaning: first two moments.
- Code: `distributions.Logistic.mean`, `distributions.Logistic.variance`.
- Source: [REF-JKB-1995, Chapter 23].

### EQ-DIST-GUM-001 — Gumbel laws for maxima and minima

Maximum: `F(x) = exp(-exp(-z))`, `S(x) = -expm1(-exp(-z))`,
`log f = -log beta - z - exp(-z)`. Minimum: the reflection `w = -z`, so
`F(x) = -expm1(-exp(z))` and `S(x) = exp(-exp(z))`.

- Domain: `beta > 0`, real `x`, `z = (x - mu)/beta`.
- Meaning: limiting law of block maxima (or minima) of light-tailed
  variables; the GEV law with `xi = 0`.
- Code: `distributions.Gumbel.log_pdf`, `distributions.Gumbel.pdf`,
  `distributions.Gumbel.cdf`, `distributions.Gumbel.sf`.
- Source: [REF-JKB-1995, Chapter 22], [REF-NIST-DISTRIBUTIONS], and
  [REF-COLES-2001, Chapter 3].

### EQ-DIST-GUM-002 — Gumbel quantile

Maximum: `x_p = mu - beta log(-log p)`, with `-log p` replaced by
`-log1p(-q)` in the upper tail; minimum: `x_p = mu + beta log(-log1p(-p))`.

- Domain: `0 < p < 1`.
- Meaning: closed-form inverse used for quantiles, `isf`, and sampling.
- Code: `distributions.Gumbel.quantile`, `distributions.Gumbel.isf`,
  `distributions.Gumbel.sample`.
- Source: [REF-JKB-1995, Chapter 22].

### EQ-DIST-GUM-003 — Gumbel moments

`E[X] = mu +/- beta gamma_E` (plus for maxima, minus for minima) and
`Var(X) = pi^2 beta^2 / 6`, with `gamma_E` the Euler-Mascheroni constant.

- Domain: `beta > 0`.
- Meaning: first two moments.
- Code: `distributions.Gumbel.mean`, `distributions.Gumbel.variance`.
- Source: [REF-JKB-1995, Chapter 22].

### EQ-DIST-TRI-001 — Triangular density

`f(x) = 2(x-a)/((b-a)(c-a))` on `[a, c)`, `2/(b-a)` at `c`, and
`2(b-x)/((b-a)(b-c))` on `(c, b]`.

- Domain: `a < b`, `a <= c <= b`.
- Meaning: expert-elicited bounded law with lower, most likely, and upper
  values.
- Code: `distributions.Triangular.pdf`, `distributions.Triangular.log_pdf`.
- Source: [REF-KOTZ-VANDORP-2004, Chapter 1].

### EQ-DIST-TRI-002 — Triangular CDF and survival function

`F(x) = (x-a)^2/((b-a)(c-a))` for `a < x <= c`;
`S(x) = (b-x)^2/((b-a)(b-c))` for `c <= x < b`; each function uses the
complement only on the far side of the mode.

- Domain: `a <= x <= b`.
- Meaning: piecewise-quadratic tails evaluated from the nearer endpoint.
- Code: `distributions.Triangular.cdf`, `distributions.Triangular.sf`.
- Source: [REF-KOTZ-VANDORP-2004, Chapter 1].

### EQ-DIST-TRI-003 — Triangular quantile

`x_p = a + sqrt(p (b-a)(c-a))` for `p <= (c-a)/(b-a)`, otherwise
`x_p = b - sqrt(q (b-a)(b-c))`, `q = 1 - p`.

- Domain: `0 <= p <= 1`.
- Meaning: closed-form inverse used for quantiles, `isf`, and sampling.
- Code: `distributions.Triangular.quantile`, `distributions.Triangular.isf`,
  `distributions.Triangular.sample`.
- Source: [REF-KOTZ-VANDORP-2004, Chapter 1].

### EQ-DIST-TRI-004 — Triangular moments

`E[X] = (a+b+c)/3` and
`Var(X) = (a^2+b^2+c^2-ab-ac-bc)/18 = ((b-a)^2 - (c-a)(b-c))/18`.

- Domain: `a < b`, `a <= c <= b`.
- Meaning: the second form avoids cancellation far from the origin.
- Code: `distributions.Triangular.mean`, `distributions.Triangular.variance`.
- Source: [REF-KOTZ-VANDORP-2004, Chapter 1].

### EQ-DIST-TN-001 — Truncated normal density

`f(x) = phi(z) / (sigma Z)` for `l <= x <= u`, `z = (x - mu)/sigma`,
`Z = Phi(beta) - Phi(alpha)`, `alpha = (l - mu)/sigma`,
`beta = (u - mu)/sigma`.

- Domain: `sigma > 0`, `l < u` with either end possibly infinite.
- Meaning: normal law conditioned on a physical interval (positive loads,
  bounded tolerances).
- Code: `distributions.TruncatedNormal.log_pdf`,
  `distributions.TruncatedNormal.pdf`.
- Source: [REF-JKB-1994, Chapter 13].

### EQ-DIST-TN-002 — Truncated normal CDF and survival function

`F(x) = (Phi(z) - Phi(alpha)) / Z` and `S(x) = (Phi(beta) - Phi(z)) / Z`,
with every interval mass `log(Phi(b) - Phi(a))` evaluated as a log-space
difference of upper tails (`a >= 0`), lower tails (`b <= 0`), or as
`(erf(b/sqrt 2) - erf(a/sqrt 2))/2` across zero.

- Domain: `alpha <= z <= beta`.
- Meaning: probabilities stay finite and accurate for truncation regions
  deep in a normal tail; quantiles use [EQ-DIST-INV-001].
- Code: `distributions.TruncatedNormal.cdf`,
  `distributions.TruncatedNormal.sf`,
  `distributions._continuous.log_normal_interval_mass`.
- Source: [REF-JKB-1994, Chapter 13] and [REF-NIST-DLMF, Sections 7.2 and 7.9].

### EQ-DIST-TN-003 — Truncated normal moments

`E[X] = mu + sigma (phi(alpha) - phi(beta)) / Z` and
`Var(X) = sigma^2 [1 + (alpha phi(alpha) - beta phi(beta))/Z
- ((phi(alpha) - phi(beta))/Z)^2]`, infinite ends contributing zero.

- Domain: `sigma > 0`, `l < u`.
- Meaning: mean shift and variance reduction caused by truncation; the
  variance bracket loses relative accuracy as a finite end moves far into a
  tail.
- Code: `distributions.TruncatedNormal.mean`,
  `distributions.TruncatedNormal.variance`.
- Source: [REF-JKB-1994, Chapter 13].

### EQ-DIST-TN-004 — Log standard-normal tail beyond erfc underflow

`log(1 - Phi(z)) = log(erfc(z/sqrt 2)/2)` for `z < 35`; otherwise
`-z^2/2 - log sqrt(2 pi) + log R(z)` with Mills ratio
`R(z) = 1/(z + 1/(z + 2/(z + 3/(z + ...))))`; `log Phi(z) = log(1 - Phi(-z))`.

- Domain: real `z`, including `+/- inf`.
- Meaning: finite log tail probabilities where `erfc` underflows to zero.
- Code: `distributions._continuous.log_standard_normal_sf`,
  `distributions._continuous.log_standard_normal_cdf`.
- Source: [REF-NIST-DLMF, Sections 7.2 and 7.9].

### EQ-DIST-GEV-001 — Generalized extreme value distribution

`F(x) = exp(-t)`, `t = exp(-y)`, `y = log1p(xi z)/xi` (`y = z` when
`xi z` is zero to working precision), `z = (x - mu)/sigma`;
`S(x) = -expm1(-t)` and `log f = -log sigma - (1 + xi) y - t` on
`1 + xi z > 0`.

- Domain: `sigma > 0`, real `xi`; lower endpoint `mu - sigma/xi` for
  `xi > 0`, upper endpoint `mu - sigma/xi` for `xi < 0`.
- Meaning: the three extreme-value types in one family (Frechet `xi > 0`,
  Gumbel `xi = 0`, reversed Weibull `xi < 0`); the `log1p` form is
  continuous through `xi = 0`.
- Code: `distributions.GeneralizedExtremeValue.log_pdf`,
  `distributions.GeneralizedExtremeValue.pdf`,
  `distributions.GeneralizedExtremeValue.cdf`,
  `distributions.GeneralizedExtremeValue.sf`.
- Source: [REF-COLES-2001, Section 3.1].

### EQ-DIST-GEV-002 — GEV quantile (return level)

`x_p = mu + sigma expm1(-xi log L)/xi` with `L = -log p` (or `-log1p(-q)`
in the upper tail), and `x_p = mu - sigma log L` when `xi log L` is zero to
working precision.

- Domain: `0 < p < 1`.
- Meaning: return level exceeded with probability `q = 1 - p` per block;
  used for `isf` and inverse-transform sampling.
- Code: `distributions.GeneralizedExtremeValue.quantile`,
  `distributions.GeneralizedExtremeValue.isf`,
  `distributions.GeneralizedExtremeValue.sample`.
- Source: [REF-COLES-2001, Section 3.1].

### EQ-DIST-GEV-003 — GEV moments with a stable Gumbel limit

`E[X] = mu + sigma (Gamma(1-xi) - 1)/xi` for `xi < 1` (`inf` otherwise) and
`Var(X) = sigma^2 (Gamma(1-2xi) - Gamma(1-xi)^2)/xi^2` for `xi < 1/2`
(`inf` otherwise). For `|xi| < 0.01` the series
`log Gamma(1-xi) = gamma_E xi + sum_(k>=2) zeta(k) xi^k / k = xi S_1` gives
`(Gamma(1-xi)-1)/xi = expm1(xi S_1)/xi` and the variance factor
`exp(2 xi S_1) S_2 expm1(xi^2 S_2)/(xi^2 S_2)` with
`S_2 = sum_(k>=2) zeta(k) (2^k - 2) xi^(k-2) / k`.

- Domain: `sigma > 0`, real `xi`; series truncated at `k = 12`.
- Meaning: moments converge continuously to the Gumbel values
  `mu + sigma gamma_E` and `sigma^2 pi^2/6` without cancellation.
- Code: `distributions.GeneralizedExtremeValue.mean`,
  `distributions.GeneralizedExtremeValue.variance`.
- Source: [REF-COLES-2001, Section 3.1] and [REF-NIST-DLMF-GAMMA, Section 5.7].

### EQ-NUM-DIGAMMA-001 — Digamma and trigamma functions

`psi(x) = psi(x+1) - 1/x`, `psi'(x) = psi'(x+1) + 1/x^2`;
for `x >= 10`, `log x - psi(x) = 1/(2x) + sum_(k=1..6) B_(2k)/(2k x^(2k))`
and `psi'(x) = 1/x + 1/(2x^2) + sum_(k=1..6) B_(2k)/x^(2k+1)`.

- Domain: finite `x > 0`.
- Meaning: derivatives of `log Gamma` needed by the gamma likelihood
  equations; `log x - psi(x)` is returned without cancellation.
- Code: `numerics.digamma`, `numerics.trigamma`, `numerics.log_minus_digamma`.
- Source: [REF-NIST-DLMF-GAMMA, Sections 5.5, 5.11, and 5.15].

### EQ-INF-MLE-GAM-001 — Gamma maximum-likelihood equations

`lambda_hat = k_hat / mean(x)` and
`log k_hat - psi(k_hat) = s = log(mean(x)) - mean(log x)`; Newton's method on
the decreasing left side, safeguarded by a bracket, from
`k_0 = (3 + sqrt(9 + 12 s))/(12 s)`.

- Domain: positive observations with at least two distinct values
  (`s > 0`).
- Meaning: profile likelihood reduces the two-parameter fit to one
  monotone scalar equation.
- Code: `inference.fit_gamma_mle`.
- Source: [REF-JKB-1994, Chapter 17] and [REF-CASELLA-BERGER-2002, Section 7.2].

### EQ-INF-MLE-WEIB-001 — Weibull maximum-likelihood equations

`sum x_i^k log x_i / sum x_i^k - 1/k - mean(log x) = 0` and
`lambda_hat = (mean(x^k_hat))^(1/k_hat)`; logarithms are centred and power
sums use a log-sum-exp shift; Newton's method with bracket safeguard from
`k_0 = pi / (sqrt 6 sd(log x))`.

- Domain: positive observations with at least two distinct values.
- Meaning: the profile equation is strictly increasing in `k`, so its root
  is the unique shape MLE.
- Code: `inference.fit_weibull_mle`.
- Source: [REF-JKB-1994, Chapter 21] and [REF-NIST-DISTRIBUTIONS].

### EQ-DIST-TN-005 — Short-interval normal mass from the interval width

`Phi(a + w) - Phi(a) = phi(a) sum_(n>=0) c_n w^(n+1)/(n+1)` with `c_0 = 1`,
`c_1 = -a`, and `(n+1) c_(n+1) = -a c_n - c_(n-1)` (the Taylor coefficients
of `exp(-a s - s^2/2)`).

- Domain: finite `a`, `w > 0`; used when `w (|a| + w) <= 1/2`.
- Meaning: probability next to a truncation point computed from the physical
  distance `w = (x - l)/sigma` instead of a difference of nearly equal
  standardized points, so `F(x)` keeps relative accuracy as `x -> l`.
- Code: `distributions._continuous.log_normal_short_mass`,
  `distributions.TruncatedNormal.cdf`, `distributions.TruncatedNormal.sf`.
- Source: [REF-NIST-DLMF, Section 7.2] and [REF-JKB-1994, Chapter 13].

## Multilevel Monte Carlo, adaptive rare-event sampling, and Shapley effects

### EQ-MC-MLMC-001 — Multilevel telescoping estimator

`E[P_L] = E[P_0] + sum_(l=1..L) E[P_l - P_(l-1)]`, estimated by
`Y_hat = sum_(l=0..L) (1/N_l) sum_(n=1..N_l) (P_l^(n,l) - P_(l-1)^(n,l))` with
`P_(-1) = 0`; `Var(Y_hat) = sum_l V_l / N_l`, `V_l = Var(P_l - P_(l-1))`.

- Domain: square-integrable approximations; the fine and coarse values of
  one sample `(n, l)` share randomness, different levels are independent.
- Meaning: the bias is that of the finest level `L`, while most samples are
  spent on cheap coarse levels because coupling makes `V_l` small.
- Code: `monte_carlo.multilevel_monte_carlo`,
  `monte_carlo.adaptive_multilevel_monte_carlo`,
  `monte_carlo.MultilevelResult.standard_error`.
- Source: [REF-GILES-2008] and [REF-GILES-2015].

### EQ-MC-MLMC-002 — Cost-optimal multilevel allocation

Minimizing `sum_l N_l C_l` subject to `sum_l V_l / N_l = v` gives
`N_l = ceil(sqrt(V_l / C_l) sum_k sqrt(V_k C_k) / v)` and total cost
`(sum_k sqrt(V_k C_k))^2 / v`. A coupled sample costs
`C_l = c(l) + c(l-1)` (`C_0 = c(0)`) for fine-path cost `c(l)`; plain Monte
Carlo on level `L` with the same variance costs `Var(P_L) c(L) / v`.

- Domain: `V_l >= 0`, `C_l > 0`, `v > 0`.
- Meaning: Lagrange-multiplier allocation; samples go where variance per
  unit cost is largest.
- Code: `monte_carlo.optimal_level_allocation`,
  `monte_carlo.MultilevelResult.single_level_cost`,
  `monte_carlo.MultilevelResult.cost_ratio`.
- Source: [REF-GILES-2008] and [REF-GILES-2015].

### EQ-MC-MLMC-003 — Rate regression, bias extrapolation, and MSE split

`|E[Y_l]| ~ M^(-alpha l)`, `V_l ~ M^(-beta l)`, `C_l ~ M^(gamma l)`, fitted by
least squares of `log_M` values on `l >= 1`. Remaining bias
`|E[P - P_L]| ~ max_(j=0,1,2) m_(L-j) M^(-j alpha) / (M^alpha - 1)`. With
`Var(Y_hat) <= eps^2/2` and bias `<= eps/sqrt 2`,
`MSE = Var(Y_hat) + bias^2 <= eps^2`.

- Domain: geometric refinement factor `M >= 2`; `alpha > 0` for the
  extrapolation (the adaptive driver uses `max(alpha, 1/2)`).
- Meaning: Giles' complexity theorem gives total cost `O(eps^-2)` when
  `beta > gamma`, `O(eps^-2 (log eps)^2)` when `beta = gamma`, and
  `O(eps^(-2-(gamma-beta)/alpha))` when `beta < gamma`; the adaptive driver
  adds levels until the extrapolated bias passes.
- Code: `monte_carlo.convergence_rates`,
  `monte_carlo.adaptive_multilevel_monte_carlo`,
  `monte_carlo.MultilevelResult.rmse_estimate`.
- Source: [REF-GILES-2008] and [REF-GILES-2015].

### EQ-REL-CE-001 — Cross-entropy levels and Gaussian parameter update

`gamma_t = max(q_rho(G(u_1..u_N)), 0)` for `u_k ~ h_(t-1) = N(mu, Sigma)`;
with elite set `E_t = {k : G(u_k) <= gamma_t}` and likelihood ratios
`W_k = phi_n(u_k) / h_(t-1)(u_k)`,
`mu_t = sum_(E_t) W_k u_k / sum_(E_t) W_k` and
`Sigma_t = sum_(E_t) W_k (u_k - mu_t)(u_k - mu_t)' / sum_(E_t) W_k`.

- Domain: standard-normal space of the Nataf map; `0 < rho <= 1/2`;
  covariance adaptation needs at least `d + 1` elite points.
- Meaning: weighted maximum likelihood of the Gaussian family for the
  conditional law `phi_n(u) I(G(u) <= gamma_t)`, which minimizes the
  Kullback-Leibler (cross-entropy) distance to it; the levels end at
  `gamma = 0`.
- Code: `reliability.cross_entropy_importance_sampling`.
- Source: [REF-RUBINSTEIN-KROESE-2004] and [REF-KURTZ-SONG-2013].

### EQ-REL-CE-002 — Gaussian-proposal importance-sampling estimate

`P_f_hat = (1/N) sum_k I(G(u_k) <= 0) w(u_k)` with `u_k ~ N(mu, LL')` and
`log w(u) = -||u||^2/2 + ||z||^2/2 + log det L`, `z = L^-1 (u - mu)`;
`c.o.v. = SE(P_f_hat) / P_f_hat`.

- Domain: proposal with full support (positive-definite `LL'`).
- Meaning: unbiased for any such proposal; the normalizing constants of
  both Gaussians cancel in the weight.
- Code: `reliability.cross_entropy_importance_sampling`,
  `reliability.compare_reliability_methods` (row `"CE-IS"`).
- Source: [REF-RUBINSTEIN-KROESE-2004] and [REF-OWEN-2013, Chapter 9].

### EQ-REL-LINE-001 — Line-sampling estimator and variance

For unit `alpha` and `u_k ~ N(0, I)`, `u_perp = u_k - (alpha . u_k) alpha`,
`G(u_perp + c_k alpha) = 0`, `p_k = Phi(-c_k)`;
`P_f_hat = (1/N) sum_k p_k` and
`Var(P_f_hat) = sum_k (p_k - P_f_hat)^2 / (N (N - 1))`.

- Domain: each line crosses the limit state at most once, with failure on
  the side of increasing `c`; lines without a crossing in `|c| <= c_max`
  contribute 0 (always safe) or 1 (always failed).
- Meaning: conditional Monte Carlo that integrates the coordinate along
  `alpha` exactly; for a planar limit state orthogonal to `alpha` every
  `p_k = Phi(-beta)` and the variance is zero.
- Code: `reliability.line_sampling`,
  `reliability.compare_reliability_methods` (row `"Line"`).
- Source: [REF-KOUTSOURELAKIS-PRADLWARTER-SCHUELLER-2004].

### EQ-SENS-SHAPLEY-001 — Shapley effect

`Sh_i = (1/Var Y) sum_(J subset D\{i}) [|J|! (d-|J|-1)! / d!] (c(J u {i}) - c(J))`
`= (1/(d! Var Y)) sum_(pi) [c(P_i(pi) u {i}) - c(P_i(pi))]`, where `P_i(pi)` is
the set of inputs preceding `i` in permutation `pi`; `c(empty) = 0`,
`c(D) = Var Y`, so `sum_i Sh_i = 1`.

- Domain: square-integrable output; any joint input law.
- Meaning: the Shapley value of the variance game; with independent inputs
  and no interactions it equals the first-order and total Sobol indices.
- Code: `sensitivity.exact_shapley_values`, `sensitivity.shapley_effects`,
  `sensitivity.linear_gaussian_shapley`.
- Source: [REF-OWEN-2014] and [REF-SONG-NELSON-STAUM-2016].

### EQ-SENS-SHAPLEY-002 — Conditional-variance cost and its estimator

`c(J) = E[Var(Y | X_(-J))]` gives the same Shapley values as
`Var(E[Y | X_J])`. Estimate:
`c_hat(J) = (1/N_O) sum_(o) s^2(Y(X_J^(o,1..N_I), X_(-J)^(o)))` with outer
draws `X_(-J)^(o)` from their marginal, inner draws
`X_J^(o,i) ~ X_J | X_(-J)^(o)`, and unbiased inner sample variance `s^2`.

- Domain: `N_I >= 2`; conditional sampling of `X_J` given `X_(-J)` must be
  available.
- Meaning: an unbiased estimator of the cost that needs only `N_I` inner
  draws; exact enumeration costs `N_V + (2^d - 2) N_O N_I` evaluations and
  random permutations `N_V + m (d - 1) N_O N_I`.
- Code: `sensitivity.shapley_effects`.
- Source: [REF-SONG-NELSON-STAUM-2016].

### EQ-SENS-SHAPLEY-003 — Gaussian conditioning by the Schur complement

For latent `z ~ N(0, R)`:
`z_J | z_K ~ N(R_JK R_KK^-1 z_K, R_JJ - R_JK R_KK^-1 R_KJ)`, then
`x_i = F_i^-1(Phi(z_i))`.

- Domain: positive-definite latent correlation `R`; Gaussian-copula inputs.
- Meaning: conditional laws of Nataf inputs are exact in latent space.
- Code: `sensitivity.shapley._ConditionalSampler`,
  `sensitivity.shapley_effects`.
- Source: [REF-RASMUSSEN-WILLIAMS-2006, Appendix A.2] and
  [REF-LIU-DERKIUREGHIAN-1986].

### EQ-SENS-SHAPLEY-004 — Shapley effects of a linear Gaussian model

For `Y = b'X`, `X ~ N(mu, Sigma)`, `s = Sigma b`:
`Var(E[Y | X_J]) = s_J' Sigma_JJ^-1 s_J`, `Var Y = b' Sigma b`. For `d = 2`
with standard deviations `sigma_1, sigma_2` and correlation `rho`,
`Sh_1 = (b1^2 sigma_1^2 (1 - rho^2/2) + rho b1 b2 sigma_1 sigma_2 +
b2^2 sigma_2^2 rho^2/2) / Var Y`.

- Domain: positive-definite `Sigma`, `Var Y > 0`.
- Meaning: the correlated share `2 rho b1 b2 sigma_1 sigma_2` is split
  equally and each own share loses `rho^2/2` of itself to the partner;
  the two-dimensional formula was derived and checked numerically
  against the subset enumeration for this release.
- Code: `sensitivity.linear_gaussian_shapley`.
- Source: [REF-OWEN-PRIEUR-2017] and [REF-IOOSS-PRIEUR-2019].

## Generic fitting, goodness-of-fit calibration, and model selection

### EQ-INF-MLE-EXP-001 — Exponential rate maximum-likelihood estimator

`lambda_hat = n / sum x_i = 1 / mean(x)`.

- Domain: non-negative finite observations with positive mean.
- Meaning: the score `n/lambda - sum x_i` vanishes at the reciprocal mean;
  observed information `n / lambda^2` gives `se(lambda_hat) = lambda_hat/sqrt(n)`.
- Code: `inference.fitting.EXPONENTIAL_FAMILY`, `inference.fit_distribution`.
- Source: [REF-CASELLA-BERGER-2002, Section 7.2].

### EQ-INF-MLE-UNIF-001 — Uniform endpoint maximum-likelihood estimator

`(a_hat, b_hat) = (min x_i, max x_i)`, `log L = -n log(b_hat - a_hat)`.

- Domain: finite real observations with at least two distinct values.
- Meaning: the likelihood `(b - a)^-n` on `a <= min x, b >= max x` is
  maximized at the sample range. The support depends on the parameters, so
  the model is non-regular: observed-information errors are undefined and AIC
  is used only as a descriptive score.
- Code: `inference.fitting.UNIFORM_FAMILY`.
- Source: [REF-CASELLA-BERGER-2002, Section 7.2].

### EQ-INF-MLE-GEN-001 — Generic likelihood maximization in unconstrained coordinates

`theta_hat = argmin_theta -sum_i log f(x_i; theta)`, searched over `u` with
`theta_j = a_j + s_j u_j` (location: `s_j` = data spread; real: `s_j = 1`),
`theta_j = a_j exp(u_j)` (positive), or
`theta_j = b_j / (1 + (b_j/a_j - 1) exp(-u_j))` (positive, bounded by `b_j`),
anchored at start values `a`, by BFGS with Armijo backtracking and a
Nelder-Mead fallback.

- Domain: `log_pdf` defined for every observation; invalid parameters or zero
  likelihood map to `+inf`.
- Meaning: transforms remove positivity constraints and equalize scales, so a
  single quasi-Newton routine fits every family; the MLE is invariant to the
  reparameterization.
- Code: `inference.fit_distribution`, `inference.fitting._numerical_mle`,
  `modeling.fit_joint_input`.
- Source: [REF-CASELLA-BERGER-2002, Section 7.2] and
  [REF-NOCEDAL-WRIGHT-2006, Chapter 6].

### EQ-INF-MLE-GEN-002 — Observed-information covariance by the delta method

`Cov(theta_hat) ~= J H_u^-1 J`, with `H_u` the central-difference Hessian of
the negative log-likelihood in `u` at the estimate and diagonal Jacobian
`J = diag(d theta_j / d u_j)` (`s_j`, `theta_j`, or
`theta_j (1 - theta_j/b_j)`); `se_j = sqrt(Cov_jj)`.

- Domain: regular interior maximum with positive-definite `H_u`; parameters
  within `0.1 %` of an upper bound are held fixed and get `se = nan`.
- Meaning: inverse observed Fisher information gives large-sample standard
  errors; at a stationary point the Hessian transforms exactly by `J`.
- Code: `inference.fitting._observed_information`, `inference.fit_distribution`.
- Source: [REF-CASELLA-BERGER-2002, Section 10.1].

### EQ-INF-IC-001 — Akaike information criterion

`AIC = 2k - 2 log L(theta_hat)`.

- Domain: maximized log-likelihood with `k` estimated parameters.
- Meaning: estimate of relative expected Kullback-Leibler information loss;
  only differences between models fitted to the same data are meaningful.
- Code: `inference.information_criteria`, `inference.FitResult`.
- Source: [REF-AKAIKE-1974] and [REF-BURNHAM-ANDERSON-2002, Chapter 2].

### EQ-INF-IC-002 — Small-sample corrected AIC

`AICc = AIC + 2k(k+1)/(n-k-1)`, `inf` when `n <= k + 1`.

- Domain: `n > k + 1`.
- Meaning: bias correction of AIC that matters when `n/k` is small.
- Code: `inference.information_criteria`.
- Source: [REF-HURVICH-TSAI-1989] and [REF-BURNHAM-ANDERSON-2002, Chapter 2].

### EQ-INF-IC-003 — Bayesian (Schwarz) information criterion

`BIC = k log n - 2 log L(theta_hat)`.

- Domain: maximized log-likelihood, `n` independent observations.
- Meaning: leading terms of the log marginal likelihood; penalizes
  parameters more strongly than AIC for `n >= 8`.
- Code: `inference.information_criteria`.
- Source: [REF-SCHWARZ-1978].

### EQ-INF-IC-004 — Akaike weights

`Delta_i = IC_i - min_j IC_j`, `w_i = exp(-Delta_i/2) / sum_j exp(-Delta_j/2)`.

- Domain: finite criterion values over one candidate set and one data set.
- Meaning: relative support of each candidate within the set; they do not
  measure absolute adequacy.
- Code: `inference.akaike_weights`, `inference.select_distribution`.
- Source: [REF-BURNHAM-ANDERSON-2002, Chapter 2].

### EQ-INF-GOF-PB-001 — Parametric-bootstrap p-value with estimated parameters

`T = T(x, F(.; theta_hat(x)))`; for `b = 1..B`: `x*_b ~ F(.; theta_hat(x))`,
`T*_b = T(x*_b, F(.; theta_hat(x*_b)))`;
`p = (1 + #{T*_b >= T}) / (1 + B_ok)`, with `B_ok` successful refits.

- Domain: EDF statistic `T` (KS `D` or AD `A^2`); the refit uses the same
  estimator as the original fit; `p = nan` for an infinite observed `A^2`.
- Meaning: reproduces the null law of the statistic when parameters are
  estimated from the tested data, which the fully specified KS/AD tables do
  not (the Lilliefors/Stephens problem); the smallest attainable p-value is
  `1/(B+1)`.
- Code: `inference.bootstrap_goodness_of_fit`, `inference.select_distribution`.
- Source: [REF-STEPHENS-1974], [REF-LILLIEFORS-1967],
  [REF-DAVISON-HINKLEY-1997, Chapter 4], and [REF-EFRON-TIBSHIRANI-1993,
  Chapter 16].

### EQ-INF-GOF-QQ-001 — Q-Q and P-P plotting positions

`p_i = (i - 1/2)/n`; Q-Q pairs `(F^-1(p_i), x_(i))`; P-P pairs
`(F(x_(i)), p_i)`.

- Domain: ordered finite sample and a continuous model law.
- Meaning: points near the diagonal indicate agreement; Q-Q emphasizes tails,
  P-P the centre.
- Code: `inference.qq_points`, `inference.pp_points`.
- Source: [REF-WILK-GNANADESIKAN-1968] and [REF-NIST-EDA, Section 1.3.3].

### EQ-INF-MLE-MVN-001 — Multivariate normal maximum-likelihood estimators

`mu_hat = (1/n) sum x_i`,
`Sigma_hat = (1/n) sum (x_i - mu_hat)(x_i - mu_hat)'`.

- Domain: `n > d` observations with non-singular sample covariance.
- Meaning: likelihood maximizers; divisor `n`, not `n - 1`.
- Code: `inference.fit_multivariate_normal_mle`.
- Source: [REF-MARDIA-KENT-BIBBY-1979, Chapter 4].

### EQ-MOD-COPULA-FIT-001 — Spearman-to-latent Gaussian-copula correlation

`r_s = (6/pi) arcsin(rho/2)`, inverted as `rho = 2 sin(pi r_s / 6)`.

- Domain: bivariate Gaussian copula; `r_s` in `[-1, 1]`.
- Meaning: Spearman's rho is a copula invariant, so it converts to the latent
  normal correlation regardless of the marginals.
- Code: `modeling.spearman_to_latent_correlation`, `modeling.fit_gaussian_copula`.
- Source: [REF-KRUSKAL-1958] and [REF-MCNEIL-FREY-EMBRECHTS-2015].

### EQ-MOD-COPULA-FIT-002 — Normal scores of rank pseudo-observations

`U_i = R_i/(n+1)`, `z_i = Phi^-1(U_i)`; `rho_jk = corr(z_j, z_k)`.

- Domain: at least two observations per column; ties take average ranks.
- Meaning: rank pseudo-observations remove marginal effects; the Pearson
  correlation of their normal scores (van der Waerden) estimates the latent
  Gaussian-copula correlation.
- Code: `modeling.normal_scores`, `modeling.fit_gaussian_copula`.
- Source: [REF-GENEST-GHOUDI-RIVEST-1995] and
  [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 8].

## Prerequisite bridges: determinants, Gaussian conditioning, change of variables

### EQ-LINALG-008 — Determinant from a partial-pivot LU factorization

`PA=LU` with unit lower-triangular `L` gives `det A = (-1)^s prod_i U_ii` and
`log|det A| = sum_i log|U_ii|`, where `s` is the number of row interchanges.

- Domain: square finite `A`. Only an exactly zero pivot column is reported as
  singular (determinant `0`, log-determinant error), so uniformly tiny but
  non-singular matrices keep their determinant.
- Meaning: volume scaling of the linear map `x -> Ax`; the log form is the
  Jacobian term of a change of variables and does not overflow.
- Code: `linalg.determinant`, `linalg.log_abs_determinant`,
  `multivariate.TransformedDensity.log_abs_det_jacobian`.
- Source: [REF-GOLUB-VANLOAN-2013, Section 3.4] and
  [REF-STRANG-2016, Chapter 5].

### EQ-DIST-MVN-006 — Gaussian subset marginal

`X_S ~ Normal(mu_S, Sigma_SS)` for an ordered list `S` of distinct indices.

- Domain: [EQ-DIST-MVN-001]; any principal submatrix of a positive-definite
  matrix is positive definite.
- Meaning: integrating out the other components of a Gaussian vector leaves
  a Gaussian whose parameters are simply read off the joint ones.
- Code: `multivariate.MultivariateNormal.marginal_subset`.
- Source: [REF-BISHOP-2006, Section 2.3.2] and
  [REF-MARDIA-KENT-BIBBY-1979, Chapter 3].

### EQ-DIST-MVN-007 — Gaussian conditioning by the Schur complement

`X_a | X_b = x_b ~ Normal(mu_a + Sigma_ab Sigma_bb^-1 (x_b - mu_b),
Sigma_aa - Sigma_ab Sigma_bb^-1 Sigma_ba)`; evaluated as
`mu_a + W'L^-1(x_b - mu_b)` and `Sigma_aa - W'W` with `Sigma_bb = LL'` and
`W = L^-1 Sigma_ba`.

- Domain: positive-definite joint covariance; `a` and `b` partition the
  indices and `a` is non-empty.
- Meaning: the conditional mean is linear in the observation (the population
  regression of `X_a` on `X_b`) and the conditional covariance does not depend
  on the observed value.
- Code: `multivariate.MultivariateNormal.conditional`.
- Source: [REF-BISHOP-2006, Section 2.3.1] and
  [REF-MARDIA-KENT-BIBBY-1979, Chapter 3].

### EQ-MVN-JAC-001 — Change of variables for densities

`p_Y(y) = p_X(g^-1(y)) |det J_(g^-1)(y)|`, i.e.
`log p_Y(y) = log p_X(g^-1(y)) + log|det d g^-1/dy|`.

- Domain: `g` a continuously differentiable bijection between open subsets of
  `R^n` with differentiable inverse; `X` has a density `p_X`.
- Meaning: probability mass is conserved while volume elements are stretched
  by the Jacobian determinant of the inverse map; analytic or
  central-difference Jacobians ([EQ-OPT-FD-001]) are accepted.
- Code: `multivariate.transformed_density`, `multivariate.TransformedDensity`.
- Source: [REF-CASELLA-BERGER-2002, Sections 2.1, 4.3, and 4.6] and
  [REF-MAGNUS-NEUDECKER-2019, Chapter 9].

### EQ-MVN-DELTA-001 — First-order (delta-method) moment propagation

`E[g(X)] ~= g(mu)` and `Cov[g(X)] ~= J Sigma J'`, where `J = dg/dx` at
`mu = E[X]` and `Sigma = Cov[X]`.

- Domain: `g` differentiable near `mu`; exact for affine `g`; otherwise the
  neglected mean bias is `tr(H_k Sigma)/2` per output with Hessian `H_k`.
- Meaning: the linearization behind standard errors of transformed
  estimators and local uncertainty propagation; the covariance is the
  congruence [EQ-LINALG-007].
- Code: `multivariate.delta_method`, `multivariate.DeltaMethodResult`.
- Source: [REF-CASELLA-BERGER-2002, Section 5.5.4] and
  [REF-MAGNUS-NEUDECKER-2019, Chapter 5].

## Solution verification, prediction components, and the UQ workflow

### EQ-VV-GCI-001 — Representative grid size

`h = (V / N)^(1/d)` for `N` cells (or degrees of freedom) filling a domain of
length, area, or volume `V` in `d` dimensions.

- Domain: `N > 0`, `V > 0`, `d in {1, 2, 3}`; meshes of one study should be
  geometrically similar so that one `h` characterizes each.
- Meaning: a single spacing per grid that makes refinement ratios
  `r = h_coarse / h_fine` well defined for unstructured or non-uniform meshes.
- Code: `uq.verification.representative_spacing`.
- Source: [REF-CELIK-ETAL-2008].

### EQ-VV-GCI-002 — Three-grid observed order with non-uniform ratios

Grids fine (1), medium (2), coarse (3): `r21 = h2/h1`, `r32 = h3/h2`,
`eps21 = phi2 - phi1`, `eps32 = phi3 - phi2`, `s = sign(eps32/eps21)`,
`p = |ln|eps32/eps21| + q(p)| / ln r21`,
`q(p) = ln((r21^p - s)/(r32^p - s))`, solved by fixed-point iteration from
`q = 0`. For monotone data the equivalent strictly increasing equation
`eps32/eps21 = r21^p (r32^p - 1)/(r21^p - 1)` is solved by Brent's method when
the iteration fails.

- Domain: `h1 < h2 < h3`, non-zero differences; for a uniform ratio `q = 0`
  and the formula reduces to [EQ-VV-001].
- Meaning: the rate at which the QoI converges, estimated from three grids
  without requiring a constant refinement ratio.
- Code: `uq.verification._observed_order`,
  `uq.verification._monotone_order_equation`,
  `uq.verification.grid_convergence_index`.
- Source: [REF-CELIK-ETAL-2008], [REF-BRENT-1973, Chapter 4].

### EQ-VV-GCI-003 — Extrapolated value and relative errors

`phi_ext^21 = (r21^p phi1 - phi2)/(r21^p - 1) = phi1 + (phi1 - phi2)/(r21^p - 1)`,
`e_a^21 = |(phi1 - phi2)/phi1|`, `e_ext^21 = |(phi_ext^21 - phi1)/phi_ext^21|`.

- Domain: monotone asymptotic convergence with order `p > 0`; relative errors
  are undefined (reported as `None`) when the denominator is zero.
- Meaning: Richardson extrapolation with the observed order and the size of
  the last refinement step relative to the solution.
- Code: `uq.verification.grid_convergence_index`,
  `uq.verification.GridConvergenceResult`.
- Source: [REF-CELIK-ETAL-2008], [REF-ROACHE-1998, Chapters 4-5].

### EQ-VV-GCI-004 — Fine-grid Grid Convergence Index

Absolute: `GCI_fine^21 = F_s |phi1 - phi2| / (r21^p - 1)`; relative:
`GCI_fine,rel^21 = F_s e_a^21 / (r21^p - 1)`; coarse pair:
`GCI^32 = F_s |phi2 - phi3| / (r32^p - 1)`. `F_s = 1.25` for three or more
grids with an observed order; `F_s = 3` when the order is not trusted.

- Domain: as [EQ-VV-GCI-003].
- Meaning: an error band `phi1 +/- GCI` intended to bracket the
  discretization error with high (about 95 %) confidence; with the exact order
  it equals `F_s` times the true fine-grid error.
- Code: `uq.verification.grid_convergence_index`,
  `uq.verification.verify_solution`.
- Source: [REF-ROACHE-1998, Chapters 4-5], [REF-CELIK-ETAL-2008].

### EQ-VV-GCI-005 — Asymptotic-range check

`GCI^32 / (r21^p GCI^21) ~ 1`. With absolute GCIs at a reference order `p`
the ratio equals `(eps32/eps21) / f(p)`, `f(p) = r21^p (r32^p - 1)/(r21^p - 1)`;
with relative GCIs (Roache's form, `GCI^32` relative to `phi2`) it is
additionally multiplied by `|phi1/phi2|`.

- Domain: monotone three-grid data; reference order positive.
- Meaning: whether the grids behave as a single leading error term predicts.
  At the observed order of the same grids the absolute ratio is identically
  one (an algebraic consequence of [EQ-VV-GCI-002]); evaluated at the
  theoretical order it tests the asymptotic range (for a uniform ratio it
  equals `r^(p_obs - p_th)`).
- Code: `uq.verification.GridConvergenceResult.asymptotic_ratio_at`,
  `uq.verification.GridConvergenceResult.in_asymptotic_range`,
  `uq.verification.grid_convergence_index`, `uq.verification.verify_solution`.
- Source: [REF-ROACHE-1998, Chapters 4-5], [REF-CELIK-ETAL-2008].

### EQ-VV-GCI-006 — Convergence classification

`R = eps32 / eps21`: oscillatory if `R < 0`; monotone if
`R > ln r32 / ln r21` (the `p -> 0` limit of `f(p)`, which is one for a
uniform ratio); otherwise divergent.

- Domain: non-zero differences.
- Meaning: Richardson-based estimates require monotone convergence;
  oscillatory sequences are flagged and divergent ones admit no positive
  order. With the fine-over-coarse ratio `eps21/eps32` used by Roy (2005), a
  uniform ratio gives monotone convergence for values in `(0, 1)`.
- Code: `uq.verification.classify_convergence`, `uq.verification._classify`,
  `uq.verification.ConvergenceType`.
- Source: [REF-CELIK-ETAL-2008], [REF-ROY-2005].

### EQ-VV-LSQ-001 — Least-squares observed order

`S(phi_0, alpha, p) = sum_i (phi_i - phi_0 - alpha (h_i/h_1)^p)^2`; for fixed
`p` the minimizer in `(phi_0, alpha)` is ordinary linear regression on
`x_i = (h_i/h_1)^p`; `p` minimizes the profile `S(p)` (400-point log scan plus
golden-section refinement); `U_s = sqrt(S_min/(n - 3))`.

- Domain: `n >= 4` distinct spacings, non-constant data, `p` inside declared
  bounds.
- Meaning: with more grids than unknowns the order is fitted rather than
  solved, so scatter in the data is visible in `U_s` instead of corrupting
  `p`; `alpha` is the fitted error of the finest solution.
- Code: `uq.verification.least_squares_convergence`,
  `uq.verification._fixed_order_fit`.
- Source: [REF-ECA-HOEKSTRA-2014], [REF-BRENT-1973, Chapter 5].

### EQ-VV-LSQ-002 — Simplified least-squares numerical uncertainty

If `0.5 <= p < p_th + 0.1` (credible order): `U = 1.25 |alpha| + U_s`;
otherwise, with `alpha_th`, `U_s,th = sqrt(S_th/(n - 2))` from the refit at
`p = p_th`: `U = 3 max(|alpha|, |alpha_th|) + max(U_s, U_s,th)`.

- Domain: as [EQ-VV-LSQ-001]; `p_th > 0` declared by the user.
- Meaning: a project simplification that keeps the structure of the
  Eca-Hoekstra procedure (fitted order, scatter term, factor of safety 1.25 or
  3) but omits their weighted fits, alternative fixed-order expansions, and
  data-range term. It is not their complete procedure.
- Code: `uq.verification.least_squares_convergence`,
  `uq.verification.verify_solution`.
- Source: [REF-ECA-HOEKSTRA-2014] (structure); simplification documented in
  this registry.

### EQ-VV-OSC-001 — Oscillatory three-grid bound

`U = F_s,osc (phi_max - phi_min)/2` with `F_s,osc = 3` by default.

- Domain: three grids with `eps32/eps21 < 0`.
- Meaning: a conservative project rule that does not use Richardson
  extrapolation when the sequence oscillates: half the oscillation range,
  multiplied by the factor of safety for untrusted estimates.
- Code: `uq.verification.verify_solution`.
- Source: [REF-ROY-2005] (oscillatory convergence invalidates Richardson
  estimates), [REF-ROACHE-1998, Chapters 4-5] (factor of safety 3); the rule
  itself is a project choice.

### EQ-VV-UNUM-001 — Numerical-uncertainty statement and standard uncertainty

Statement `phi_1 +/- U` with interval `[phi_1 - U, phi_1 + U]`;
`u_num = U / sqrt(3)` (rectangular, "uniform") or `u_num = U / 1.959964`
(Gaussian with `U` a 95 % expanded uncertainty, "normal95").

- Domain: `U >= 0` from [EQ-VV-GCI-004], [EQ-VV-LSQ-002], or [EQ-VV-OSC-001];
  the distribution is a declared assumption.
- Meaning: turns a verification result into the `u_num` term used by the
  V&V 20 comparison [EQ-VAL-ASME-001] and prediction [EQ-VAL-PRED-001].
- Code: `uq.verification.SolutionVerification.interval`,
  `uq.verification.SolutionVerification.standard_uncertainty`,
  `validation.credibility.VerificationEvidence.from_solution_verification`,
  `validation.prediction.predict_with_uncertainty_components`.
- Source: [REF-JCGM-100-2008, Sections 4.3.7 and 6.2],
  [REF-ROACHE-1998, Chapters 4-5].

### EQ-VAL-DISC-001 — Distance-based discrepancy-variance extrapolation

`sigma_delta^2(x) = sigma_delta,0^2 (1 + kappa d(x))`, with `d(x)` the scaled
distance from `x` to the nearest validation condition ([EQ-VAL-DOM-001]) and
declared `kappa >= 0`.

- Domain: finite `d(x)` (the condition may not leave a validation dimension
  of zero range); `sigma_delta,0` estimated at validation points.
- Meaning: model-form uncertainty equals its validation estimate at the
  validation points and grows linearly with distance from them; `kappa` is a
  modelling assumption, not an estimate.
- Code: `validation.prediction.DiscrepancyExtrapolation.at`,
  `validation.prediction.DiscrepancyAtCondition.variance`.
- Source: [REF-ROY-OBERKAMPF-2011] (extrapolating model-form uncertainty
  beyond the validation domain); the linear rule is a project choice.

### EQ-VAL-DISC-002 — Moment estimate of the discrepancy variance

`sigma_delta,0^2 = max(0, mean_i(E_i^2) - mean_i(u_val,i^2))` with
`E_i = S_i - D_i` and `u_val,i` from [EQ-VAL-ASME-001].

- Domain: at least one comparison; errors modeled as zero-mean discrepancy
  plus independent validation uncertainty.
- Meaning: the part of the observed comparison error not explained by
  numerical, input, and data uncertainty; a systematic bias is absorbed into
  the variance and reported separately as `mean(E_i)`.
- Code: `validation.prediction.estimate_discrepancy`.
- Source: [REF-ASME-VV20-2009], [REF-KENNEDY-OHAGAN-2001].

### EQ-VAL-PRED-001 — Probabilistic combination of prediction components

`z_s = y_s + e_s + d_s` with parametric samples `y_s`, independent
`d_s ~ N(0, sigma_delta^2(x))`, and `e_s ~ Uniform(-U, U)` or
`N(0, (U/1.96)^2)`; `Var(Z) = Var(Y) + u_num^2 + sigma_delta^2(x)`; the
combined interval is the equal-tailed sample interval of `z`.

- Domain: the three components are independent; the treatment of numerical
  error is declared.
- Meaning: one interval for decision use, while the component variances and
  their shares stay reported separately.
- Code: `validation.prediction.predict_with_uncertainty_components`,
  `validation.prediction.ComponentPrediction.variance_shares`.
- Source: [REF-ROY-OBERKAMPF-2011], [REF-OBERKAMPF-ROY-2025, Part IV].

### EQ-VAL-PRED-002 — Interval treatment of numerical uncertainty

`[q_(alpha/2)(Y + delta) - U, q_(1-alpha/2)(Y + delta) + U]`.

- Domain: numerical error known only to lie in `[-U, U]` (epistemic, no
  distribution).
- Meaning: the union of the probabilistic interval shifted anywhere in the
  numerical-error interval; numerical uncertainty widens the bounds instead of
  being averaged into a variance.
- Code: `validation.prediction.predict_with_uncertainty_components`.
- Source: [REF-ROY-OBERKAMPF-2011].

## Uncertainty visualization

### EQ-STAT-DKW-001 — Dvoretzky-Kiefer-Wolfowitz confidence band

`P(sup_x |F_n(x) - F(x)| > eps) <= 2 exp(-2 n eps^2)`, so
`eps = sqrt(ln(2/alpha) / (2n))` gives a simultaneous `1 - alpha` band
`[F_n - eps, F_n + eps]` clipped to `[0, 1]`.

- Domain: independent identically distributed observations from any
  continuous or discrete law; Massart's constant 2 is tight.
- Meaning: a distribution-free band for the whole CDF, wider than pointwise
  intervals because it holds for every `x` at once.
- Code: `visualization.uncertainty.dkw_epsilon`,
  `visualization.uncertainty.plot_ecdf_band`.
- Source: [REF-DVORETZKY-KIEFER-WOLFOWITZ-1956] and [REF-MASSART-1990].

## Hamiltonian Monte Carlo, NUTS, and Gaussian variational inference

### EQ-HMC-H-001 — Hamiltonian with a Gaussian momentum

`H(x, p) = U(x) + K(p)` with potential `U(x) = -log p(x)` and kinetic energy
`K(p) = p' M^-1 p / 2` for momentum `p ~ N(0, M)`; `dx/dt = M^-1 p`,
`dp/dt = grad log p(x)`. The augmented density is `exp(-H)`, whose
`x`-marginal is the target.

- Domain: `log p` differentiable on `R^d`; `M` symmetric positive definite
  (the code stores `M^-1`, diagonal or dense).
- Meaning: Hamiltonian flow conserves `H` and phase-space volume, so it can
  move far along the target's level sets without being rejected; `M^-1`
  plays the role of a preconditioner (ideally the posterior covariance).
- Code: `inference.hamiltonian_energy`, `inference.hamiltonian._Metric`,
  `inference.hamiltonian._joint`.
- Source: [REF-NEAL-2011] and [REF-BETANCOURT-2017].

### EQ-HMC-LEAPFROG-001 — Leapfrog (Stormer-Verlet) integrator

`p_{1/2} = p + (eps/2) grad log p(x)`, `x' = x + eps M^-1 p_{1/2}`,
`p' = p_{1/2} + (eps/2) grad log p(x')`.

- Domain: step size `eps > 0` (negative integrates backwards); any
  position-dependent force field, including a finite-difference gradient.
- Meaning: each half-step is a shear, so the map is volume-preserving and
  reversible under momentum negation for any force; the energy error over a
  fixed integration time is `O(eps^2)` (the library measures
  `7.52e-4, 1.88e-4, 4.68e-5` for `eps = 0.1, 0.05, 0.025`).
- Code: `inference.leapfrog`, `inference.hamiltonian._leapfrog_step`.
- Source: [REF-NEAL-2011] and [REF-BETANCOURT-2017].

### EQ-HMC-ACCEPT-001 — Static HMC Metropolis correction

After `L` leapfrog steps from `(x, p)` to `(x*, p*)`, accept `x*` with
probability `min(1, exp(H(x, p) - H(x*, p*)))`.

- Domain: fixed `L` and `eps` (optionally jittered independently of the
  state); `H(x*, p*) = +inf` outside the support (rejected).
- Meaning: corrects the integrator's energy error exactly, so the chain is
  reversible with respect to `p(x)` regardless of `eps`; the acceptance
  probability is the statistic targeted by dual averaging.
- Code: `inference.hamiltonian_monte_carlo`,
  `inference.hamiltonian._hmc_transition`.
- Source: [REF-NEAL-2011].

### EQ-HMC-DA-001 — Dual-averaging step-size adaptation

`Hbar_m = (1 - 1/(m + t0)) Hbar_{m-1} + (delta - alpha_m)/(m + t0)`,
`log eps_m = mu - sqrt(m) Hbar_m / gamma`,
`log epsbar_m = m^-kappa log eps_m + (1 - m^-kappa) log epsbar_{m-1}`, with
`mu = log(10 eps_0)`, `gamma = 0.05`, `t0 = 10`, `kappa = 0.75`, and target
acceptance `delta = 0.8`.

- Domain: warm-up iterations only; `alpha_m` is the iteration's mean
  acceptance statistic; restarted after every metric update.
- Meaning: drives the average acceptance to `delta`; `epsbar` (frozen after
  warm-up) averages out the noisy late iterates.
- Code: `inference.hamiltonian._DualAveraging`,
  `inference.hamiltonian._run_hamiltonian_chain`.
- Source: [REF-HOFFMAN-GELMAN-2014, Algorithm 5] and [REF-NESTEROV-2009].

### EQ-HMC-METRIC-001 — Windowed, shrunk inverse-metric estimate

`M^-1 = n/(n + 5) C_w + 10^-3 * 5/(n + 5) I`, where `C_w` is the unbiased
sample covariance (or its diagonal) of the `n` draws of one slow warm-up
window; windows double in length between a 75-iteration initial and a
50-iteration terminal fast buffer (15 % / 75 % / 10 % below 150 warm-up
iterations).

- Domain: `n >= 3` window draws; the update is skipped if the estimate is not
  positive definite.
- Meaning: whitens the target so one step size fits every direction; the
  shrinkage keeps short windows from producing a singular metric. The
  constants are a project choice modeled on common practice and documented in
  ADR-0032.
- Code: `inference.adaptation_windows`,
  `inference.hamiltonian._window_metric`.
- Source: [REF-BETANCOURT-2017] and [REF-HOFFMAN-GELMAN-2014].

### EQ-HMC-BFMI-001 — Energy Bayesian fraction of missing information

`E-BFMI = sum_{n>=2} (E_n - E_{n-1})^2 / sum_n (E_n - Ebar)^2` over the
post-warm-up energies `E_n = H(x_n, p_n)` after momentum refresh.

- Domain: at least two energies with positive spread.
- Meaning: how well momentum resampling explores the marginal energy
  distribution; values below about 0.3 indicate heavy tails or funnels that
  HMC explores slowly.
- Code: `inference.HamiltonianChainStatistics.e_bfmi`.
- Source: [REF-BETANCOURT-2017].

### EQ-NUTS-SLICE-001 — NUTS slice variable and progressive sampling

Draw `u ~ Uniform(0, exp(-H(x, p_0)))`; the candidate set is the leapfrog
states with `-H >= log u`. Within a subtree the second half's candidate is
chosen with probability `n''/(n' + n'')`; a new top-level subtree's candidate
replaces the current one with probability `min(1, n'/n)`. A leaf with
`-H < log u - Delta_max` (`Delta_max = 1000`) is a divergence and stops the
doubling.

- Domain: trajectory built by repeated doubling in a random direction up to
  the maximum tree depth.
- Meaning: the slice-sampling version of efficient NUTS (Algorithm 3 with the
  dual-averaging statistics of Algorithm 6); it leaves the target invariant
  and favors states far from the start. The multinomial variant is not
  implemented.
- Code: `inference.no_u_turn_sampler`,
  `inference.hamiltonian._nuts_transition`.
- Source: [REF-HOFFMAN-GELMAN-2014, Algorithms 3 and 6].

### EQ-NUTS-UTURN-001 — No-U-turn termination criterion

Stop doubling when `(x+ - x-) . M^-1 p- < 0` or `(x+ - x-) . M^-1 p+ < 0` for
the whole trajectory or any balanced subtree.

- Domain: trajectory end points `(x-, p-)` and `(x+, p+)`; `M^-1 p` is the
  velocity under the chosen metric.
- Meaning: continuing would start to bring the ends closer again, so further
  integration wastes gradients; checking every subtree preserves
  reversibility.
- Code: `inference.hamiltonian._no_u_turn`.
- Source: [REF-HOFFMAN-GELMAN-2014] and [REF-BETANCOURT-2017].

### EQ-VI-ELBO-001 — Evidence lower bound

`ELBO(q) = E_q[log p(x)] + H[q] = log Z - KL(q || p/Z)`; for
`q = N(mu, L L')`, `H[q] = sum_i log L_ii + (d/2)(1 + log 2 pi)`.

- Domain: unnormalized `log p` on `R^d` integrable under `q`.
- Meaning: maximizing the ELBO minimizes the reverse KL divergence; the gap
  to `log Z` is the KL divergence (for a mean-field fit of a Gaussian,
  `(1/2)(sum_i log Lambda_ii - log det Lambda)`).
- Code: `inference.advi`, `inference.variational._entropy`.
- Source: [REF-BLEI-KUCUKELBIR-MCAULIFFE-2017] and
  [REF-KUCUKELBIR-ETAL-2017].

### EQ-VI-REPARAM-001 — Reparameterization gradient

With `x = mu + L eta`, `eta ~ N(0, I)`:
`grad_mu ELBO = E[g]`, `grad_{L_ij} ELBO = E[g_i eta_j]` (`i > j`),
`grad_{lambda_i} ELBO = E[g_i eta_i] L_ii + 1` for `L_ii = exp(lambda_i)`,
where `g = grad log p(x)`.

- Domain: differentiable `log p`; estimated by averaging over a few `eta`
  draws per iteration.
- Meaning: unbiased low-variance stochastic gradients of the ELBO from
  gradients of `log p` alone; the `+1` is the entropy gradient.
- Code: `inference.advi`, `inference.variational._transform`.
- Source: [REF-KUCUKELBIR-ETAL-2017].

### EQ-VI-ADVI-001 — Gaussian variational families

Mean-field: `L = diag(exp(omega))`; full-rank: `L` lower triangular with
`L_ii = exp(lambda_i)` and free strictly-lower entries; covariance `L L'`.

- Domain: unconstrained parameters on `R^d` (constrained ones transformed by
  the caller with the Jacobian in `log p`).
- Meaning: mean-field ignores correlations and, for a Gaussian target,
  recovers the mean and the variances `1/Lambda_ii` (too small); full-rank
  can represent the target exactly.
- Code: `inference.advi`, `inference.VariationalResult.covariance`.
- Source: [REF-KUCUKELBIR-ETAL-2017].

### EQ-OPT-ADAM-001 — Adam update with Polyak-Ruppert tail averaging

`m_t = b1 m_{t-1} + (1 - b1) g_t`, `v_t = b2 v_{t-1} + (1 - b2) g_t^2`,
`theta_t = theta_{t-1} + alpha (m_t / (1 - b1^t)) / (sqrt(v_t / (1 - b2^t)) + eps)`
(ascent); the reported parameters average the iterates over the most recent
half of the iterations.

- Domain: stochastic gradients `g_t`; `0 <= b1, b2 < 1`, `alpha, eps > 0`.
- Meaning: per-coordinate step sizes normalized by a running gradient scale;
  averaging the tail removes most of the step noise that a constant `alpha`
  leaves around the optimum.
- Code: `inference.advi`.
- Source: [REF-KINGMA-BA-2015, Algorithm 1] and [REF-POLYAK-JUDITSKY-1992].

## Time series: correlograms, ARIMA, and exponential smoothing

### EQ-TS-ACF-001 — Sample autocovariance and autocorrelation

`gamma_hat(h) = (1/n) sum_{t=1}^{n-h} (x_t - x_bar)(x_{t+h} - x_bar)`,
`rho_hat(h) = gamma_hat(h) / gamma_hat(0)`.

- Domain: `n >= 2` finite observations, non-constant, `0 <= h < n`.
- Meaning: the `1/n` divisor (not `1/(n-h)`) keeps the sample autocovariance
  matrix positive semidefinite, so the derived PACF stays in `[-1, 1]`.
- Code: `timeseries.autocovariance`, `timeseries.acf`.
- Source: [REF-BROCKWELL-DAVIS-2016, Chapter 1], [REF-BOX-JENKINS-2015,
  Chapter 2].

### EQ-TS-PACF-001 — Durbin-Levinson recursion

`phi_kk = (rho(k) - sum_{j=1}^{k-1} phi_(k-1),j rho(k-j)) / v_(k-1)`,
`phi_kj = phi_(k-1),j - phi_kk phi_(k-1),(k-j)`,
`v_k = v_(k-1) (1 - phi_kk^2)`, `v_0 = 1`.

- Domain: a positive-definite autocorrelation sequence.
- Meaning: solves the Yule-Walker equations of every order `1..K` in
  `O(K^2)`; `phi_kk` is the partial autocorrelation at lag `k`, which cuts
  off after lag `p` for an AR(p).
- Code: `timeseries.durbin_levinson`, `timeseries.pacf`.
- Source: [REF-BROCKWELL-DAVIS-2016, Chapters 2 and 3], [REF-BOX-JENKINS-2015,
  Chapter 3].

### EQ-TS-BAND-001 — White-noise correlogram band

`+/- z_(1-alpha/2) / sqrt(n)`.

- Domain: i.i.d. noise with finite variance; pointwise, large `n`.
- Meaning: each sample autocorrelation at a nonzero lag is approximately
  `N(0, 1/n)`; about `alpha` of the lags fall outside by chance.
- Code: `timeseries.white_noise_band`.
- Source: [REF-BROCKWELL-DAVIS-2016, Chapter 1], [REF-BOX-JENKINS-2015,
  Chapter 2].

### EQ-TS-BAND-002 — Bartlett band for an MA(k-1) null

`Var(rho_hat(k)) ~ (1/n) (1 + 2 sum_{j=1}^{k-1} rho(j)^2)`, half-width
`z_(1-alpha/2) sqrt(Var)`.

- Domain: the autocorrelations beyond lag `k - 1` are zero; sample values
  replace `rho(j)`.
- Meaning: the band widens with the lag because earlier autocorrelations add
  sampling variance; used to judge where an ACF cuts off.
- Code: `timeseries.bartlett_band`.
- Source: [REF-BOX-JENKINS-2015, Chapter 2], [REF-BROCKWELL-DAVIS-2016,
  Chapter 2].

### EQ-TS-LB-001 — Ljung-Box portmanteau statistic

`Q = n (n + 2) sum_{k=1}^{h} rho_hat(k)^2 / (n - k)`, compared with
`chi^2(h - m)` where `m` is the number of fitted ARMA parameters.

- Domain: `1 <= h < n`, `h > m`; asymptotic.
- Meaning: tests that the first `h` autocorrelations are jointly zero; the
  `(n + 2)/(n - k)` factor improves the chi-square approximation of the
  Box-Pierce statistic in small samples.
- Code: `timeseries.ljung_box`, `timeseries.ARIMAFit.diagnostics`.
- Source: [REF-LJUNG-BOX-1978], [REF-BOX-JENKINS-2015, Chapter 8].

### EQ-TS-DIFF-001 — Ordinary and seasonal differencing

`(1 - B^s)^D (1 - B)^d y_t`; inverse
`y_t = x_t - sum_{k=1}^{d} binom(d, k) (-1)^k y_(t-k)`.

- Domain: `n > d + D s`; the inverse needs the first `d` observations.
- Meaning: removes polynomial trends (`d`) and fixed seasonal patterns
  (`D`, period `s`).
- Code: `timeseries.difference`, `timeseries.seasonal_difference`,
  `timeseries.undifference`, `timeseries.simulate_arima`.
- Source: [REF-BOX-JENKINS-2015, Chapters 4 and 9],
  [REF-BROCKWELL-DAVIS-2016, Chapter 6].

### EQ-TS-DECOMP-001 — Classical moving-average decomposition

Trend `T_t` = centred moving average (equal weights over `s` points for odd
`s`; the `2 x s` average with half weights at both ends for even `s`).
Additive: `S_j = mean_t(y_t - T_t | t = j mod s)` shifted to sum to zero,
`R_t = y_t - T_t - S_t`. Multiplicative: ratios `y_t / T_t`, indices scaled to
average one, `R_t = y_t / (T_t S_t)`.

- Domain: at least two full periods; positive data for the multiplicative
  form. The trend is undefined for the first and last `floor(s/2)` points.
- Meaning: a descriptive split of level, seasonality, and irregular
  variation; it does not model uncertainty.
- Code: `timeseries.centered_moving_average`,
  `timeseries.classical_decomposition`.
- Source: [REF-BROCKWELL-DAVIS-2016, Chapter 1].

### EQ-TS-ARMA-001 — ARMA model, causality, and invertibility

`phi(B)(x_t - mu) = theta(B) e_t`, `phi(z) = 1 - sum_{j=1}^{p} phi_j z^j`,
`theta(z) = 1 + sum_{j=1}^{q} theta_j z^j`, `e_t ~ N(0, sigma^2)` i.i.d.
Causal (stationary) iff all roots of `phi` satisfy `|z| > 1`; invertible iff
all roots of `theta` do.

- Domain: real coefficients; `phi` and `theta` without common roots.
- Meaning: the root conditions make `x_t` a convergent linear filter of past
  shocks and make the shocks recoverable from past observations.
- Code: `timeseries.ar_root_check`, `timeseries.ma_root_check`,
  `timeseries.arma_state_space`.
- Source: [REF-BROCKWELL-DAVIS-2016, Chapter 3], [REF-HAMILTON-1994,
  Chapter 3].

### EQ-TS-ARMA-002 — MA(infinity) weights

`psi_0 = 1`, `psi_j = theta_j + sum_{i=1}^{min(j,p)} phi_i psi_(j-i)`
(`theta_j = 0` for `j > q`); `gamma(k) = sigma^2 sum_j psi_j psi_(j+k)`.

- Domain: causal ARMA.
- Meaning: the impulse response of `theta(z)/phi(z)`; it gives the
  autocovariances and the classical forecast-error variances.
- Code: `timeseries.psi_weights`, `timeseries.arma_autocovariance`.
- Source: [REF-BROCKWELL-DAVIS-2016, Chapter 3], [REF-BOX-JENKINS-2015,
  Chapter 3].

### EQ-TS-ROOT-001 — Durand-Kerner simultaneous root iteration

`z_k <- z_k - p(z_k) / prod_{j != k} (z_k - z_j)` for a monic polynomial,
starting on a circle of radius `|a_0|^(1/n)`, followed by accepted Newton
steps `z <- z - p(z)/p'(z)`.

- Domain: real or complex coefficients, nonzero leading coefficient.
- Meaning: finds all complex roots at once (conjugate pairs included);
  simple roots converge quadratically, repeated roots linearly.
- Code: `timeseries.polynomial_roots`.
- Source: [REF-KERNER-1966], [REF-GOLUB-VANLOAN-2013, Section 7.4].

### EQ-TS-ROOT-002 — Schur-Cohn step-down stationarity test

`r_k = phi_kk`, `phi_(k-1),j = (phi_kj + r_k phi_k,(k-j)) / (1 - r_k^2)`;
`phi(z)` has all roots outside the unit circle iff `|r_k| < 1` for all `k`.

- Domain: real AR coefficients.
- Meaning: an exact algebraic stationarity decision that needs no root
  finding; it is the inverse of the Durbin-Levinson step-up.
- Code: `timeseries.coefficients_to_pacf`, `timeseries.is_stationary`,
  `timeseries.is_invertible`.
- Source: [REF-MONAHAN-1984], [REF-BROCKWELL-DAVIS-2016, Chapter 2].

### EQ-TS-PARAM-001 — Partial-autocorrelation reparameterization

`r_k = tanh(u_k)`, `phi^(k)_k = r_k`,
`phi^(k)_j = phi^(k-1)_j - r_k phi^(k-1)_(k-j)`; MA coefficients use
`theta_j = -phi_j(r)`.

- Domain: unconstrained `u` in `R^p`.
- Meaning: a smooth bijection from `R^p` onto the stationary (or invertible)
  region, so an unconstrained optimizer cannot leave it.
- Code: `timeseries.pacf_to_coefficients`, `timeseries.fit_arima`.
- Source: [REF-JONES-1980], [REF-MONAHAN-1984].

### EQ-TS-SSM-001 — Harvey state-space form of ARMA

`alpha_t = T alpha_(t-1) + R e_t`, `x_t - mu = Z alpha_t`, `r = max(p, q+1)`,
`T = [phi | I_(r-1); 0]` (first column `phi_1..phi_r`, ones above the
diagonal), `R = (1, theta_1, ..., theta_(r-1))'`, `Z = e_1'`; stationary
`P = T P T' + sigma^2 R R'`, `gamma(k) = Z T^k P Z'`.

- Domain: causal ARMA; no measurement noise.
- Meaning: the first state element is the observation; the Lyapunov solution
  gives the exact stationary start for the Kalman filter.
- Code: `timeseries.arma_state_space`, `timeseries.ARMAStateSpace`,
  `timeseries.arma_autocovariance`.
- Source: [REF-DURBIN-KOOPMAN-2012, Chapters 3 and 5], [REF-HAMILTON-1994,
  Chapter 13].

### EQ-TS-LIK-001 — Exact likelihood by prediction-error decomposition

`log L = -(1/2) sum_t [log 2 pi + log(sigma^2 F_t) + v_t^2 / (sigma^2 F_t)]`
with Kalman innovations `v_t` and unit-scale variances `F_t`;
`sigma_hat^2 = (1/n) sum_t v_t^2 / F_t` and the concentrated
`log L_c = -(n/2)(log 2 pi + 1 + log sigma_hat^2) - (1/2) sum_t log F_t`.

- Domain: [EQ-TS-SSM-001] started at its stationary distribution.
- Meaning: the exact Gaussian likelihood in `O(n r^2)` without forming the
  `n x n` covariance; concentrating `sigma^2` removes one optimization
  dimension.
- Code: `timeseries.arma_log_likelihood`, `timeseries.ARMAStateSpace.kalman`,
  `timeseries.fit_arima`.
- Source: [REF-DURBIN-KOOPMAN-2012, Chapter 7], [REF-HAMILTON-1994,
  Chapters 5 and 13].

### EQ-TS-LIK-002 — Closed-form exact AR(1) likelihood

`log L = -(1/2) log(2 pi sigma^2/(1 - phi^2)) - (1 - phi^2)(y_1 - mu)^2/(2 sigma^2)
- ((n-1)/2) log(2 pi sigma^2)
- sum_{t=2}^{n} (y_t - mu - phi (y_(t-1) - mu))^2 / (2 sigma^2)`.

- Domain: `|phi| < 1`.
- Meaning: independent reference value against which the Kalman likelihood
  is tested.
- Code: `tests/test_timeseries.py::ar1_closed_form_log_likelihood`.
- Source: [REF-HAMILTON-1994, Chapter 5].

### EQ-TS-YW-001 — Yule-Walker estimator

`Gamma_hat_p phi_hat = gamma_hat_p`, `sigma_hat^2 = gamma_hat(0) - phi_hat'
gamma_hat_p`, asymptotic `Cov(phi_hat) ~ sigma^2 Gamma_p^-1 / n`.

- Domain: AR(p), `p < n`, biased sample autocovariances.
- Meaning: a method-of-moments estimate that is always stationary; used as
  the starting value for maximum likelihood.
- Code: `timeseries.yule_walker`.
- Source: [REF-BROCKWELL-DAVIS-2016, Chapter 5], [REF-BOX-JENKINS-2015,
  Chapter 7].

### EQ-TS-SE-001 — Observed-information standard errors

`Cov(theta_hat) ~ H^-1`, `H = -nabla^2 log L` at the MLE in
`(phi, theta, mu, log sigma^2)`; `se(sigma^2) = sigma^2 se(log sigma^2)`.

- Domain: interior MLE, positive-definite Hessian; asymptotic.
- Meaning: Wald standard errors; reported as `None` when the Hessian is not
  positive definite or a difference step leaves the stationary region.
- Code: `timeseries.fit_arima`.
- Source: [REF-HAMILTON-1994, Chapter 5], [REF-BOX-JENKINS-2015, Chapter 7].

### EQ-TS-FC-001 — Exact h-step ARIMA forecast and interval

`x_hat_(n+j) = mu + Z T^(j-1) a_(n+1)`,
`P_(n+j+1|n) = T P_(n+j|n) T' + sigma^2 R R'`,
`Cov(x_(n+i), x_(n+j) | y) = Z T^(j-i) P_(n+i|n) Z'` (`i <= j`);
`y_(n+h) = c_h + sum_{j=1}^{h} binom(h-j+d-1, d-1) x_(n+j)`, so
`Var(y_(n+h) | y) = w_h' C w_h` and the interval is
`y_hat +/- z_(1-alpha/2) sqrt(Var)`.

- Domain: fitted parameters treated as known; first `d` observations treated
  as fixed (diffuse) starting values.
- Meaning: finite-sample conditional forecasts; for ARIMA(0,1,0) the
  variance is `h sigma^2` and the interval widens as `sqrt(h)`; for large `n`
  it reduces to `sigma^2 sum_{j<h} psi*_j^2`.
- Code: `timeseries.forecast_arima`, `timeseries.ARIMAFit.forecast`.
- Source: [REF-DURBIN-KOOPMAN-2012, Chapter 4], [REF-BOX-JENKINS-2015,
  Chapter 5], [REF-BROCKWELL-DAVIS-2016, Chapter 6].

### EQ-TS-SIM-001 — ARMA simulation recursion

`w_t = sum_{i=1}^{p} phi_i w_(t-i) + e_t + sum_{j=1}^{q} theta_j e_(t-j)`,
zero start, the first `B` values discarded, `x_t = mu + w_t`.

- Domain: causal ARMA; `B` large relative to the slowest root decay.
- Meaning: the burn-in lets the zero start forget itself so the returned
  path is approximately stationary.
- Code: `timeseries.simulate_arma`, `timeseries.simulate_arima`.
- Source: [REF-BROCKWELL-DAVIS-2016, Chapter 3], [REF-BOX-JENKINS-2015,
  Chapter 3].

### EQ-TS-ES-001 — Simple exponential smoothing

`y_hat_(t|t-1) = l_(t-1)`, `l_t = l_(t-1) + alpha (y_t - l_(t-1))`,
`0 < alpha < 1`; `alpha` minimizes `sum_t (y_t - y_hat_(t|t-1))^2` with `l_0`
chosen by least squares.

- Domain: level without trend or season.
- Meaning: an exponentially weighted average of the past; the forecast is
  flat at `l_n`. Concentrating `l_0` is a project choice.
- Code: `timeseries.simple_exponential_smoothing`.
- Source: [REF-BROCKWELL-DAVIS-2016, Chapter 10], [REF-HOLT-2004].

### EQ-TS-ES-002 — Holt's linear trend method

`y_hat_(t|t-1) = l_(t-1) + b_(t-1)`, `l_t = alpha y_t + (1 - alpha)(l_(t-1) +
b_(t-1))`, `b_t = beta (l_t - l_(t-1)) + (1 - beta) b_(t-1)`; forecast
`l_n + h b_n`.

- Domain: `0 < alpha, beta < 1`; locally linear trend.
- Meaning: smooths the level and the slope separately; the parameters
  minimize the one-step squared error with `(l_0, b_0)` chosen by least
  squares (a project choice).
- Code: `timeseries.holt_linear`.
- Source: [REF-HOLT-2004], [REF-BROCKWELL-DAVIS-2016, Chapter 10].

## Survival analysis and conformal prediction

### EQ-SURV-KM-001 — Kaplan-Meier product-limit estimator

`S_hat(t) = prod_{t_j <= t} (1 - d_j / n_j)` over the distinct event times
`t_j`, with `n_j = #{i : t_i >= t_j}` at risk and `d_j` events at `t_j`.

- Domain: independent right-censored observations `(t_i, delta_i)`;
  censoring independent of the event time (non-informative); a subject
  censored at `t_j` counts in `n_j`.
- Meaning: the nonparametric MLE of the survival function; a right-continuous
  step function that drops only at event times and is undefined beyond the
  last observed time when that time is censored.
- Code: `inference.survival.nonparametric.kaplan_meier`,
  `inference.survival.nonparametric.KaplanMeierEstimate.survival_at`,
  `inference.survival.data.SurvivalData.risk_table`.
- Source: [REF-KAPLAN-MEIER-1958], [REF-KLEIN-MOESCHBERGER-2003, Chapter 4].

### EQ-SURV-KM-002 — Greenwood variance

`Var_hat(S_hat(t)) = S_hat(t)^2 sum_{t_j <= t} d_j / (n_j (n_j - d_j))`.

- Domain: as EQ-SURV-KM-001; large-sample (delta-method) approximation. When
  `n_j = d_j` the curve reaches zero and the project reports variance `0`.
- Meaning: sampling variance of the product-limit estimate at a fixed `t`.
- Code: `inference.survival.nonparametric.kaplan_meier`,
  `inference.survival.nonparametric.KaplanMeierEstimate.standard_errors`.
- Source: [REF-GREENWOOD-1926], [REF-KLEIN-MOESCHBERGER-2003, Chapter 4].

### EQ-SURV-KM-003 — Log-log pointwise confidence interval

`[S_hat^{exp(z w)}, S_hat^{exp(-z w)}]` with
`w = sqrt(sum_{t_j <= t} d_j / (n_j (n_j - d_j))) / |log S_hat(t)|` and
`z = Phi^-1((1 + confidence) / 2)`.

- Domain: `0 < S_hat(t) < 1`; the project returns `(1, 1)` before the first
  event and `(0, 0)` once `S_hat = 0`.
- Meaning: a delta-method interval on the `log(-log S)` scale; it always lies
  inside `(0, 1)`, unlike the plain `S_hat +/- z se` interval.
- Code: `inference.survival.nonparametric.kaplan_meier`,
  `inference.survival.nonparametric.KaplanMeierEstimate.interval_at`.
- Source: [REF-KLEIN-MOESCHBERGER-2003, Chapter 4].

### EQ-SURV-NA-001 — Nelson-Aalen cumulative hazard

`H_hat(t) = sum_{t_j <= t} d_j / n_j`, `Var_hat(H_hat(t)) = sum_{t_j <= t}
d_j / n_j^2`, log-transformed interval `H_hat exp(+/- z sqrt(Var) / H_hat)`,
and survival estimate `exp(-H_hat(t))`.

- Domain: as EQ-SURV-KM-001.
- Meaning: the nonparametric estimate of the integrated hazard; `exp(-H_hat)`
  is never below the Kaplan-Meier value.
- Code: `inference.survival.nonparametric.nelson_aalen`,
  `inference.survival.nonparametric.NelsonAalenEstimate.cumulative_hazard_at`,
  `inference.survival.nonparametric.NelsonAalenEstimate.survival_at`.
- Source: [REF-NELSON-1972], [REF-AALEN-1978],
  [REF-KLEIN-MOESCHBERGER-2003, Chapter 4].

### EQ-SURV-MED-001 — Survival quantiles and band-inversion interval

`t_p = min{t_j : S_hat(t_j) <= 1 - p}` (median: `p = 1/2`); the interval is
`[min{t_j : S_L(t_j) <= 1 - p}, min{t_j : S_U(t_j) <= 1 - p}]` with the
log-log pointwise bands `S_L <= S_hat <= S_U` of EQ-SURV-KM-003.

- Domain: `0 < p < 1`; a limit is `+inf` when the corresponding curve never
  reaches `1 - p` during follow-up. If a curve equals `1 - p` exactly on a
  horizontal segment `[t_j, t_{j+1})`, the midpoint `(t_j + t_{j+1}) / 2` is
  used (the convention of `quantile.survfit` in the R survival package),
  which gives the ordinary sample median for uncensored data.
- Meaning: the set of times at which `1 - p` lies inside the pointwise band.
- Code: `inference.survival.nonparametric.KaplanMeierEstimate.quantile`,
  `inference.survival.nonparametric.KaplanMeierEstimate.median`.
- Source: [REF-KLEIN-MOESCHBERGER-2003, Chapter 4].

### EQ-SURV-RMST-001 — Restricted mean survival time

`mu_hat(tau) = int_0^tau S_hat(u) du` with
`Var_hat = sum_{t_j <= tau} A_j^2 d_j / (n_j (n_j - d_j))`,
`A_j = int_{t_j}^tau S_hat(u) du`, and a normal interval
`mu_hat +/- z sqrt(Var_hat)`.

- Domain: `tau > 0`, not beyond the last observed time while `S_hat > 0`.
- Meaning: the expected event-free time up to `tau`, `E[min(T, tau)]`; it is
  identified even when the median is not.
- Code: `inference.survival.nonparametric.KaplanMeierEstimate.restricted_mean`.
- Source: [REF-KLEIN-MOESCHBERGER-2003, Chapter 4].

### EQ-SURV-LR-001 — k-sample log-rank statistic

`U_g = sum_j (d_gj - d_j n_gj / n_j)`,
`V_gh = sum_j d_j (n_j - d_j) / (n_j - 1) (n_gj / n_j) (1[g = h] - n_hj / n_j)`,
`X^2 = U' V^-1 U` over the first `k - 1` groups, referred to chi-square with
`k - 1` degrees of freedom. For two groups `X^2 = (O_1 - E_1)^2 / V_11`.

- Domain: independent groups with non-informative censoring; large-sample
  chi-square approximation; terms with `n_j = 1` contribute no variance.
- Meaning: compares observed with expected events under equal hazards using
  the hypergeometric moments of each risk-set table; most powerful against
  proportional-hazards alternatives.
- Code: `inference.survival.nonparametric.log_rank_test`.
- Source: [REF-MANTEL-1966], [REF-KLEIN-MOESCHBERGER-2003, Chapter 7].

### EQ-SURV-AFT-001 — Censored accelerated-failure-time log-likelihood

`log T_i = x_i' beta + sigma W`, `z_i = (log t_i - x_i' beta) / sigma`,
`l = sum_i delta_i [log f_W(z_i) - log sigma - log t_i] + (1 - delta_i)
log S_W(z_i)`, with `f_W(z) = exp(z - e^z)`, `S_W(z) = exp(-e^z)` (Weibull;
`sigma = 1` is exponential) or `f_W = phi`, `S_W = 1 - Phi` (lognormal).
Weibull shape `= 1/sigma`, scale `= exp(x' beta)`; exponential rate
`= exp(-x' beta)`.

- Domain: strictly positive times, non-informative censoring, at least one
  event; observed-information standard errors are large-sample.
- Meaning: covariates multiply survival time by `exp(beta_k)` per unit; the
  likelihood uses the density for events and the survival function for
  censored subjects.
- Code: `inference.survival.parametric.fit_parametric_survival`,
  `inference.survival.parametric.ParametricSurvivalFit.distribution`,
  `inference.survival.parametric.ParametricSurvivalFit.weibull_parameters`.
- Source: [REF-KLEIN-MOESCHBERGER-2003, Chapter 12].

### EQ-SURV-COX-001 — Cox partial likelihood with Breslow ties

`l(beta) = sum_j [sum_{i in D_j} x_i' beta - d_j log S0(R_j)]`,
`S0(A) = sum_{i in A} exp(x_i' beta)`; score
`U = sum_j [sum_{D_j} x_i - d_j S1/S0]`, information
`I = sum_j d_j [S2/S0 - (S1/S0)(S1/S0)']` over `R_j = {i : t_i >= t_j}`.

- Domain: proportional hazards `h(t | x) = h_0(t) exp(x' beta)`;
  non-informative censoring; covariates fixed in time.
- Meaning: Cox's partial likelihood; with ties every tied event is compared
  with the full risk set, which biases `beta` toward zero when ties are many.
- Code: `inference.survival.cox.cox_partial_likelihood`,
  `inference.survival.cox.fit_cox_model`.
- Source: [REF-COX-1972], [REF-BRESLOW-1974],
  [REF-KLEIN-MOESCHBERGER-2003, Chapter 8].

### EQ-SURV-COX-002 — Efron tie correction

`l(beta) = sum_j [sum_{i in D_j} x_i' beta - sum_{r=0}^{d_j - 1}
log(S0(R_j) - (r/d_j) S0(D_j))]`, with the score and information obtained by
replacing `S_k(R_j)` by `S_k(R_j) - (r/d_j) S_k(D_j)` term by term.

- Domain: as EQ-SURV-COX-001; identical to it when there are no ties.
- Meaning: the tied events leave the risk set in equal fractions, a close
  approximation to the exact (discrete-time) partial likelihood.
- Code: `inference.survival.cox.cox_partial_likelihood`,
  `inference.survival.cox.fit_cox_model`.
- Source: [REF-EFRON-1977], [REF-KLEIN-MOESCHBERGER-2003, Chapter 8].

### EQ-SURV-COX-003 — Wald hazard-ratio interval and global tests

`HR_k = exp(beta_k)`, interval `exp(beta_k +/- z se_k)` with
`se_k = sqrt([I(beta_hat)^-1]_kk)`; likelihood ratio `2[l(beta_hat) - l(0)]`
and score `U(0)' I(0)^-1 U(0)`, both chi-square with `p` degrees of freedom.

- Domain: large-sample normality of `beta_hat`; finite MLE (no monotone
  likelihood).
- Meaning: multiplicative change in the hazard per unit covariate. With one
  binary covariate, untied times, and Breslow ties, the score statistic equals
  the two-sample log-rank statistic.
- Code: `inference.survival.cox.fit_cox_model`.
- Source: [REF-COX-1972], [REF-KLEIN-MOESCHBERGER-2003, Chapter 8].

### EQ-SURV-COX-004 — Breslow baseline cumulative hazard

`H_0_hat(t) = sum_{t_j <= t} d_j / sum_{i in R_j} exp(x_i' beta_hat)` and
`S_hat(t | x) = exp(-H_0_hat(t) exp(x' beta_hat))`.

- Domain: as EQ-SURV-COX-001; reduces to Nelson-Aalen at `beta = 0`.
- Meaning: the baseline needed to turn hazard ratios into survival curves.
- Code: `inference.survival.cox.fit_cox_model`,
  `inference.survival.cox.CoxModelResult.survival`.
- Source: [REF-BRESLOW-1974], [REF-KLEIN-MOESCHBERGER-2003, Chapter 8].

### EQ-CONF-001 — Finite-sample conformal quantile

`k = ceil((1 - alpha)(n + 1))`, `q_hat = s_(k)` (the `k`-th smallest of `n`
calibration scores), `q_hat = +inf` when `k = n + 1`; equivalently the
empirical `(1 - alpha)(n + 1)/n` quantile.

- Domain: `0 < alpha < 1`, `n >= 1`; computed exactly from the decimal text of
  `alpha`.
- Meaning: the `(n + 1)/n` inflation makes room for the exchangeable test
  point; `alpha < 1/(n + 1)` forces infinite sets.
- Code: `conformal.quantile.conformal_rank`,
  `conformal.quantile.conformal_quantile`.
- Source: [REF-VOVK-GAMMERMAN-SHAFER-2005], [REF-LEI-ETAL-2018],
  [REF-ANGELOPOULOS-BATES-2023].

### EQ-CONF-002 — Split conformal regression intervals

Absolute score `s_i = |y_i - mu(x_i)|`, interval `mu(x) +/- q_hat`;
normalized (locally weighted) score `s_i = |y_i - mu(x_i)| / sigma(x_i)`,
interval `mu(x) +/- q_hat sigma(x)`, with `mu` and `sigma > 0` fixed before
seeing the calibration data.

- Domain: calibration and test pairs exchangeable; `mu`, `sigma` independent
  of the calibration set.
- Meaning: wraps any predictor (including a surrogate of a probabilistic
  model) with marginally valid error bars; normalization adapts the width to
  `sigma(x)` without changing the guarantee.
- Code: `conformal.regression.split_conformal_regression`,
  `conformal.regression.SplitConformalRegressor.interval`,
  `conformal.surrogate.conformalize_surrogate`.
- Source: [REF-LEI-ETAL-2018], [REF-VOVK-GAMMERMAN-SHAFER-2005].

### EQ-CONF-003 — Conformalized quantile regression

`E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))`, interval
`[q_lo(x) - q_hat, q_hi(x) + q_hat]` with `q_hat` from EQ-CONF-001 applied to
the `E_i`.

- Domain: as EQ-CONF-002; `q_lo`, `q_hi` any functions fitted without the
  calibration data.
- Meaning: shifts estimated conditional-quantile bands outward (or inward when
  `q_hat < 0`) until they are marginally valid, keeping their adaptive width.
- Code: `conformal.regression.conformalized_quantile_regression`,
  `conformal.regression.ConformalizedQuantileRegressor.interval`.
- Source: [REF-ROMANO-PATTERSON-CANDES-2019].

### EQ-CONF-004 — Jackknife+ interval

`R_i = |y_i - mu_{-i}(x_i)|`; lower limit the `floor(alpha (n + 1))`-th
smallest of `mu_{-i}(x) - R_i` (`-inf` if that index is 0), upper limit the
`ceil((1 - alpha)(n + 1))`-th smallest of `mu_{-i}(x) + R_i` (`+inf` if it
exceeds `n`).

- Domain: exchangeable training and test data; a fitting algorithm that is
  symmetric in its training points.
- Meaning: uses every point for both fitting and calibration at the cost of
  `n` refits; marginal coverage is at least `1 - 2 alpha` and typically near
  `1 - alpha`.
- Code: `conformal.regression.jackknife_plus_regression`,
  `conformal.regression.JackknifePlusRegressor.interval`,
  `conformal.quantile.lower_conformal_rank`.
- Source: [REF-BARBER-ETAL-2021].

### EQ-CONF-005 — Split conformal marginal coverage

`1 - alpha <= k / (n + 1) <= P(Y_{n+1} in C(X_{n+1}))`, with equality
`P = k/(n + 1) < 1 - alpha + 1/(n + 1)` when scores are almost surely distinct.

- Domain: exchangeable calibration and test points; the probability is over
  both the calibration set and the test point (marginal, not conditional on
  `x` or on the calibration set).
- Meaning: the exact target against which empirical coverage is checked; a
  coverage count on held-out data is binomial and its Clopper-Pearson interval
  shows how precisely coverage has been measured.
- Code: `conformal.quantile.split_conformal_coverage`,
  `conformal.coverage.empirical_coverage`,
  `conformal.regression.SplitConformalRegressor.guaranteed_coverage`.
- Source: [REF-LEI-ETAL-2018], [REF-ANGELOPOULOS-BATES-2023].

## Multiple linear regression, generalized linear models, and regularization

### EQ-LINALG-009 — Householder QR factorization and least squares

`H_k = I - 2 v_k v_k'` with `v_k` proportional to `x - alpha e_1`,
`alpha = -sign(x_1) ||x||`, applied to the trailing part of column `k`; then
`A = Q R` with `Q = H_0 ... H_(p-1)` (thin, `n x p`) and upper-triangular
`R`. The least-squares solution is `x = R^-1 Q'b`, `(A'A)^-1 = R^-1 R^-T`, and
the leverages are `h_i = sum_j Q_ij^2`.

- Domain: finite `n x p` matrix with `n >= p`; without pivoting, column `j`
  is flagged rank deficient when `|R_jj| <= tol ||a_j||` (default
  `tol = 1e-10`), and the solve then refuses instead of regularizing.
- Meaning: least squares without forming `A'A`, so the solve sees `kappa(A)`
  rather than `kappa(A)^2`.
- Code: `linalg.householder_qr`, `linalg.HouseholderQR.solve`,
  `linalg.HouseholderQR.gram_inverse`, `linalg.HouseholderQR.hat_diagonal`.
- Source: [REF-HOUSEHOLDER-1958], [REF-GOLUB-VANLOAN-2013, Sections 5.1-5.3].

### EQ-LM-DESIGN-001 — Treatment (reference-level) coding

For a factor with sorted levels `L_1 < ... < L_m` and reference `L_r`, the
design receives the indicators `1{x_i = L_k}` for `k != r`; without an
intercept the first factor keeps all `m` indicators.

- Domain: finite numeric columns; categorical values compared as text; new
  data may only contain levels seen when the design was built.
- Meaning: the intercept is the reference-level mean (at zero numeric
  inputs) and each indicator coefficient is a contrast with the reference;
  dropping one level avoids the dummy-variable trap.
- Code: `inference.design_matrix`, `inference.DesignMatrix.transform`,
  `inference.ColumnEncoding.column_names`.
- Source: [REF-MONTGOMERY-PECK-VINING-2021, Chapter 8].

### EQ-LM-OLS-001 — Ordinary and weighted least squares

`beta_hat = argmin sum_i w_i (y_i - x_i'beta)^2`, the solution of
`X'WX beta = X'Wy`, computed as `beta_hat = R^-1 Q' W^(1/2) y` from the QR
factorization of `W^(1/2) X`.

- Domain: `n > p`, full column rank, strictly positive finite weights
  (inverse relative variances); unit weights give OLS.
- Meaning: best linear unbiased estimate under `Var(e_i) = sigma^2 / w_i`.
- Code: `inference.fit_linear_model`.
- Source: [REF-MONTGOMERY-PECK-VINING-2021, Chapters 3 and 5].

### EQ-LM-INF-001 — Coefficient covariance and t tests

`sigma_hat^2 = RSS/(n-p)`, `RSS = sum w_i e_i^2`,
`Cov(beta_hat) = sigma_hat^2 (X'WX)^-1`, `t_j = beta_hat_j / se_j` with
`t_j ~ t_(n-p)` under `beta_j = 0`, and intervals
`beta_hat_j +- t_(1-a/2, n-p) se_j`.

- Domain: [EQ-LM-OLS-001] with independent normal errors for exact t laws.
- Meaning: classical inference for individual coefficients.
- Code: `inference.fit_linear_model`,
  `inference.LinearModelResult.coefficient_intervals`.
- Source: [REF-MONTGOMERY-PECK-VINING-2021, Chapter 3].

### EQ-LM-FIT-001 — R squared, adjusted R squared, and the overall F test

`R^2 = 1 - RSS/TSS`, `R^2_adj = 1 - (1 - R^2)(n - k0)/(n - p)`,
`F = ((TSS - RSS)/(p - k0)) / (RSS/(n - p)) ~ F_(p-k0, n-p)`, where `k0 = 1`
and `TSS = sum w_i (y_i - y_bar_w)^2` with an intercept, and `k0 = 0` with the
uncentered `TSS = sum w_i y_i^2` without one.

- Domain: [EQ-LM-INF-001]; `F` is undefined (`nan`) for an intercept-only fit.
- Meaning: in-sample share of variation explained, its degrees-of-freedom
  adjustment, and the test that every non-intercept coefficient is zero.
- Code: `inference.fit_linear_model`.
- Source: [REF-MONTGOMERY-PECK-VINING-2021, Chapter 3].

### EQ-LM-PRED-001 — Confidence and prediction intervals at new rows

`y_hat_0 = x_0'beta_hat`; mean interval
`y_hat_0 +- t_(1-a/2, n-p) sigma_hat sqrt(x_0'(X'WX)^-1 x_0)`; prediction
interval `y_hat_0 +- t_(1-a/2, n-p) sigma_hat sqrt(1/w_0 + x_0'(X'WX)^-1 x_0)`.

- Domain: [EQ-LM-INF-001]; the new observation is independent of the fitting
  data with variance `sigma^2 / w_0`.
- Meaning: uncertainty of the regression mean versus that of one future
  observation, which also carries the noise.
- Code: `inference.LinearModelResult.predict`,
  `inference.LinearModelResult.interval`.
- Source: [REF-MONTGOMERY-PECK-VINING-2021, Chapter 3].

### EQ-LM-DIAG-001 — Leverage

`h_i = [H]_ii` for `H = X*(X*'X*)^-1 X*' = QQ'`, `X* = W^(1/2) X`, so
`0 <= h_i <= 1` and `sum_i h_i = p`.

- Domain: full-rank design.
- Meaning: how strongly observation `i` pulls its own fitted value; large
  leverage flags remote design points.
- Code: `inference.fit_linear_model`, `linalg.HouseholderQR.hat_diagonal`.
- Source: [REF-MONTGOMERY-PECK-VINING-2021, Chapter 6].

### EQ-LM-DIAG-002 — Standardized and studentized residuals

`r_i = e*_i / (sigma_hat sqrt(1 - h_i))` (internal) and
`t_i = e*_i / (sigma_(i) sqrt(1 - h_i))` (external), with
`sigma_(i)^2 = (RSS - e*_i^2/(1 - h_i)) / (n - p - 1)` and
`e*_i = sqrt(w_i) e_i`.

- Domain: `h_i < 1`; external residuals need `n - p > 1`; `t_i ~ t_(n-p-1)`
  under normal errors.
- Meaning: residuals on a common scale; the external version uses the
  variance estimated without observation `i` and is the outlier statistic.
- Code: `inference.fit_linear_model`.
- Source: [REF-MONTGOMERY-PECK-VINING-2021, Chapter 4].

### EQ-LM-DIAG-003 — Cook's distance

`D_i = r_i^2 h_i / (p (1 - h_i))`, equivalently
`(beta_hat_(i) - beta_hat)'X*'X*(beta_hat_(i) - beta_hat) / (p sigma_hat^2)`.

- Domain: [EQ-LM-DIAG-002].
- Meaning: the change in all fitted values when observation `i` is deleted,
  combining residual size and leverage.
- Code: `inference.fit_linear_model`.
- Source: [REF-COOK-1977], [REF-MONTGOMERY-PECK-VINING-2021, Chapter 6].

### EQ-LM-HC-001 — Heteroskedasticity-consistent covariance (HC0-HC3)

`V_HC = (X*'X*)^-1 X*' diag(omega) X* (X*'X*)^-1` with
`omega_i = e*_i^2` (HC0), `n/(n-p) e*_i^2` (HC1), `e*_i^2/(1-h_i)` (HC2), and
`e*_i^2/(1-h_i)^2` (HC3).

- Domain: independent errors with possibly unequal variances; `h_i < 1` for
  HC2/HC3.
- Meaning: coefficient standard errors that stay consistent without the
  constant-variance assumption; HC2 and HC3 correct the downward bias of
  HC0 at high-leverage points in small samples.
- Code: `inference.LinearModelResult.robust_covariance`,
  `inference.LinearModelResult.robust_standard_errors`.
- Source: [REF-WHITE-1980], [REF-MACKINNON-WHITE-1985].

### EQ-LM-VIF-001 — Variance inflation factor

`VIF_j = 1/(1 - R_j^2)`, where `R_j^2` is the centered coefficient of
determination from regressing column `j` on an intercept and the other
non-constant columns.

- Domain: at least one non-constant column and `n` larger than the number of
  predictors; an exactly explained column gets `VIF = inf`.
- Meaning: the factor by which collinearity inflates `Var(beta_hat_j)`
  relative to orthogonal predictors.
- Code: `inference.variance_inflation_factors`,
  `inference.LinearModelResult.variance_inflation_factors`.
- Source: [REF-MONTGOMERY-PECK-VINING-2021, Chapter 9].

### EQ-LM-LL-001 — Gaussian log-likelihood of a linear model

`log L = -(n/2)(log(2 pi RSS/n) + 1) + (1/2) sum_i log w_i`, the profile
log-likelihood at `sigma^2_MLE = RSS/n`; `AIC` and `BIC` count `p + 1`
parameters.

- Domain: [EQ-LM-OLS-001] with `RSS > 0`.
- Meaning: likelihood-based comparison of linear models with the same
  response, consistent with the Gaussian GLM.
- Code: `inference.fit_linear_model`.
- Source: [REF-AKAIKE-1974], [REF-SCHWARZ-1978].

### EQ-GLM-FAM-001 — GLM families, links, and variance functions

`g(mu_i) = eta_i = x_i'beta + o_i`, `Var(y_i) = phi V(mu_i)/w_i`, with
Gaussian `V = 1`, identity link; binomial (proportions, `w` = trials)
`V = mu(1-mu)`, logit `log(mu/(1-mu))` or probit `Phi^-1(mu)`; Poisson
`V = mu`, log link; Gamma `V = mu^2`, inverse `1/mu` or log link.

- Domain: responses in `[0, 1]` (binomial), `[0, inf)` (Poisson),
  `(0, inf)` (Gamma); logit and probit predictors are clipped to `[-30, 30]`
  and `[-8, 8]` when inverted so fitted probabilities stay inside `(0, 1)`.
- Meaning: the mean model, the link that makes it linear, and the
  mean-variance relation that determines the IRLS weights.
- Code: `inference.GLMFamily`, `inference.glm_family`,
  `inference.GLMResult.predict`.
- Source: [REF-NELDER-WEDDERBURN-1972],
  [REF-MCCULLAGH-NELDER-1989, Chapters 2, 4, 6, and 8].

### EQ-GLM-IRLS-001 — Iteratively reweighted least squares (Fisher scoring)

`z_i = eta_i - o_i + (y_i - mu_i)/mu'(eta_i)`,
`W_i = w_i mu'(eta_i)^2 / V(mu_i)`, `beta_new = (X'WX)^-1 X'Wz`, repeated
with step halving when a step leaves the mean space or raises the deviance.

- Domain: full-rank design; convergence when
  `max_j |delta beta_j|/(|beta_j| + 0.1) < tol`; for canonical links Fisher
  scoring equals Newton-Raphson.
- Meaning: maximum-likelihood fitting of a GLM by a sequence of weighted
  least-squares problems.
- Code: `inference.fit_glm`.
- Source: [REF-MCCULLAGH-NELDER-1989, Section 2.5],
  [REF-NELDER-WEDDERBURN-1972].

### EQ-GLM-DEV-001 — Unit deviances and the deviance

`D = sum_i w_i d(y_i, mu_i)` with `d = (y - mu)^2` (Gaussian),
`2[y log(y/mu) + (1-y) log((1-y)/(1-mu))]` (binomial),
`2[y log(y/mu) - (y - mu)]` (Poisson), `2[-log(y/mu) + (y - mu)/mu]`
(Gamma), using `0 log 0 = 0`.

- Domain: [EQ-GLM-FAM-001].
- Meaning: twice the log-likelihood gap to the saturated model (scaled by
  `phi`); differences of deviances of nested models are likelihood-ratio
  statistics. With the canonical log link and an intercept,
  `sum (y - mu) = 0`, so the Poisson deviance reduces to
  `2 sum y log(y/mu)`.
- Code: `inference.GLMFamily.unit_deviance`, `inference.fit_glm`.
- Source: [REF-MCCULLAGH-NELDER-1989, Section 2.3].

### EQ-GLM-PEARSON-001 — Pearson chi-square and dispersion estimate

`X^2 = sum_i w_i (y_i - mu_i)^2 / V(mu_i)`; `phi_hat = X^2/(n - p)` for the
Gaussian and Gamma families, `phi = 1` for binomial and Poisson.

- Domain: [EQ-GLM-FAM-001], `n > p`.
- Meaning: moment estimate of the dispersion; `X^2/(n-p)` much larger than
  one for binomial or Poisson data signals overdispersion.
- Code: `inference.fit_glm`.
- Source: [REF-MCCULLAGH-NELDER-1989, Section 2.5].

### EQ-GLM-WALD-001 — Wald standard errors for GLM coefficients

`Cov(beta_hat) = phi (X'WX)^-1` at the final IRLS weights;
`z_j = beta_hat_j/se_j` compared with `N(0,1)` (known `phi`) or `t_(n-p)`
(estimated `phi`); intervals `beta_hat_j +- q se_j`.

- Domain: large-sample regularity; meaningless under separation.
- Meaning: inverse expected Fisher information as the coefficient
  covariance.
- Code: `inference.fit_glm`, `inference.GLMResult.coefficient_intervals`.
- Source: [REF-MCCULLAGH-NELDER-1989, Section 2.5].

### EQ-GLM-AIC-001 — GLM log-likelihood and AIC

Binomial `sum [log C(w_i, w_i y_i) + w_i y_i log mu_i + w_i(1-y_i) log(1-mu_i)]`;
Poisson `sum w_i [y_i log mu_i - mu_i - log y_i!]`; Gaussian
`-(n/2)(log(2 pi D/n) + 1) + (1/2) sum log w_i`; Gamma
`sum w_i log f_Gamma(y_i; shape 1/phi, scale mu_i phi)` with the plug-in
`phi = D / sum w_i`. `AIC = -2 log L + 2k`, `BIC = -2 log L + k log n`, with
`k = p` plus one for an estimated dispersion.

- Domain: [EQ-GLM-FAM-001]; the Gamma plug-in dispersion is an approximation,
  not the exact shape MLE.
- Meaning: likelihood-based comparison of models for the same response.
- Code: `inference.fit_glm`.
- Source: [REF-AKAIKE-1974], [REF-MCCULLAGH-NELDER-1989, Chapter 2].

### EQ-GLM-SEP-001 — Separation certificate for binomial models

The binomial MLE does not exist when some direction `d` satisfies
`x_i'd >= 0` for `y_i = 1`, `x_i'd <= 0` for `y_i = 0`, and `x_i'd = 0` for
`0 < y_i < 1`, with at least one strict inequality: the log-likelihood is
then non-decreasing along `beta + t d` for every `t > 0`.

- Domain: logit or probit link. The library takes `d` as the last IRLS step,
  requires it to move some `x_i'd` by more than `1e-4`, and checks the
  inequalities to the relative tolerance `1e-6 max_i |x_i'd|`.
- Meaning: complete or quasi-complete separation; the coefficients diverge,
  and the fit is rejected (or flagged with a warning) instead of returning
  meaningless estimates and standard errors.
- Code: `inference.fit_glm`, `inference.SeparationError`,
  `inference.SeparationWarning`.
- Source: [REF-ALBERT-ANDERSON-1984].

### EQ-PEN-STD-001 — Standardization for penalized regression

`x~_ij = (x_ij - x_bar_j)/s_j`, `s_j^2 = (1/n) sum_i (x_ij - x_bar_j)^2`,
`y_c = y - y_bar`; the unpenalized intercept is
`b_0 = y_bar - sum_j b_j x_bar_j` with original-scale slopes
`b_j = b~_j / s_j`.

- Domain: every predictor varies (`s_j > 0`); without an intercept no
  centring is done and `s_j^2 = (1/n) sum x_ij^2`.
- Meaning: the penalty acts on comparable scales, and the intercept is not
  shrunk.
- Code: `inference.fit_ridge`, `inference.fit_elastic_net`,
  `inference.elastic_net_path`, `inference.cross_validate_elastic_net`.
- Source: [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 3.4],
  [REF-FRIEDMAN-HASTIE-TIBSHIRANI-2010, Section 2].

### EQ-PEN-RIDGE-001 — Ridge regression and effective degrees of freedom

`b~ = argmin ||y_c - X~b||^2 + lambda ||b||^2 = (X~'X~ + lambda I)^-1 X~'y_c`,
solved as least squares on `[X~; sqrt(lambda) I]`;
`df(lambda) = tr[X~(X~'X~ + lambda I)^-1 X~'] = sum_j d_j^2/(d_j^2 + lambda)`
with singular values `d_j` of `X~`.

- Domain: `lambda >= 0`; `lambda > 0` makes any design full rank; `lambda = 0`
  is OLS and gives `df = p`.
- Meaning: shrinkage that trades bias for variance; `df` counts the effective
  number of fitted parameters (intercept excluded).
- Code: `inference.fit_ridge`, `inference.RidgeResult.predict`.
- Source: [REF-HOERL-KENNARD-1970],
  [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 3.4.1].

### EQ-PEN-EN-001 — Elastic net and its coordinate-descent update

Minimize `(1/2n)||y_c - X~b||^2 + lambda[(1-alpha)/2 ||b||^2 + alpha ||b||_1]`
by cycling `b_j <- S(rho_j, lambda alpha)/(v_j + lambda(1 - alpha))`, with
`rho_j = (1/n) x~_j'(y_c - X~b) + v_j b_j`, `v_j = (1/n) x~_j'x~_j`, and
`S(z, g) = sign(z) max(|z| - g, 0)`.

- Domain: `lambda >= 0`, `0 <= alpha <= 1`; `alpha = 1` is the lasso and
  `alpha = 0` equals ridge [EQ-PEN-RIDGE-001] with penalty `n lambda`.
- Meaning: exact coordinatewise minimization of a convex objective; the soft
  threshold sets small coefficients exactly to zero.
- Code: `inference.fit_elastic_net`, `inference.elastic_net_path`,
  `inference.ElasticNetResult.predict`.
- Source: [REF-TIBSHIRANI-1996], [REF-ZOU-HASTIE-2005],
  [REF-FRIEDMAN-HASTIE-TIBSHIRANI-2010, Section 2].

### EQ-PEN-KKT-001 — Elastic-net optimality (KKT) conditions and lambda_max

With `g_j = (1/n) x~_j'(y_c - X~b) - lambda(1-alpha) b_j`:
`g_j = lambda alpha sign(b_j)` when `b_j != 0` and `|g_j| <= lambda alpha`
when `b_j = 0`. Hence `b = 0` is optimal iff
`lambda >= lambda_max = max_j |x~_j'y_c|/(n alpha)`.

- Domain: [EQ-PEN-EN-001] with `alpha > 0` for `lambda_max`.
- Meaning: certificate of optimality checked by the tests, and the start of
  the default log-spaced penalty path.
- Code: `inference.elastic_net_path`, `inference.fit_elastic_net`.
- Source: [REF-FRIEDMAN-HASTIE-TIBSHIRANI-2010, Section 2].

### EQ-PEN-CV-001 — K-fold cross-validation and the one-standard-error rule

`CV(lambda) = (1/K) sum_k MSE_k(lambda)`, `SE(lambda) = sd_k(MSE_k)/sqrt(K)`;
`lambda_min = argmin CV` and `lambda_1se` is the largest `lambda` with
`CV(lambda) <= CV(lambda_min) + SE(lambda_min)`. Folds assign the `i`-th
element of a seeded permutation to fold `i mod K`.

- Domain: `2 <= K <= n`; each training part is re-standardized on its own.
- Meaning: an estimate of out-of-sample prediction error along the path and
  a deliberately more parsimonious choice within its noise.
- Code: `inference.cross_validate_elastic_net`, `inference.kfold_assignment`.
- Source: [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 7.10],
  [REF-FRIEDMAN-HASTIE-TIBSHIRANI-2010].

## Numerical regressions, round two

### EQ-NUM-STIRLING-001 — Stirling-difference prefactors for large parameters

`stirling_error(a) = lgamma(a) - (a - 1/2) log a + a - log sqrt(2 pi)
= sum_(k=1..7) B_(2k) / (2k (2k-1) a^(2k-1))` for `a >= 15`;
`log1pmx(t) = log(1 + t) - t = -t y + 2 sum_(k>=1) y^(2k+1)/(2k+1)`,
`y = t/(2+t)`, for `|t| <= 1/2`;
`a log x - x - lgamma(a) = a log1pmx((x-a)/a) + log(a)/2 - log sqrt(2 pi)
- stirling_error(a)`;
`a log x + b log(1-x) - log B(a,b) = a log1pmx(t) + b log1pmx(s)
+ log(ab/(a+b))/2 - log sqrt(2 pi) - stirling_error(a) - stirling_error(b)
+ stirling_error(a+b)` with `1 + t = x(a+b)/a`, `1 + s = (1-x)(a+b)/b`.

- Domain: used when the larger shape is at least `100`; the shared offset
  `x(a+b) - a` is formed in exact rational arithmetic.
- Meaning: removes the cancellation between `lgamma` values of size
  `a log a` that cost relative errors of `1e-5` to `1e-3` in densities and
  incomplete-function prefactors at shapes `1e10` to `1e12`.
- Code: `distributions._special.stirling_error`, `distributions._special.log1pmx`,
  `distributions._special.log_gamma_kernel`,
  `distributions._special.log_beta_kernel`,
  `distributions._special.log_gamma_ratio`,
  `distributions._special.log_beta_function`, `distributions.Gamma.log_pdf`,
  `distributions.Beta.log_pdf`, `distributions.FisherF.log_pdf`,
  `distributions.StudentT.log_pdf`, `inference.noncentral_t_cdf`.
- Source: [REF-NIST-DLMF-GAMMA, Section 5.11] and [REF-DIDONATO-MORRIS-1992].

### EQ-NUM-GAMMA-002 — Temme's uniform expansion of the incomplete gamma ratios

`Q(a,x) = erfc(eta sqrt(a/2))/2 + R`, `P(a,x) = erfc(-eta sqrt(a/2))/2 - R`,
`R = exp(-a eta^2/2)/sqrt(2 pi a) (c_0(eta) + c_1(eta)/a)`,
`eta = sign(lambda - 1) sqrt(2(lambda - 1 - log lambda))`, `lambda = x/a`,
`c_0 = 1/(lambda-1) - 1/eta`,
`c_1 = 1/eta^3 - 1/(lambda-1)^3 - 1/(lambda-1)^2 - 1/(12(lambda-1))`, both
summed from 18-term Taylor series in `eta` with exact rational coefficients.

- Domain: `a >= 1e6` and `|x - a| <= 0.3 a`, where the series and continued
  fraction would need `O(sqrt(a))` terms; the omitted `c_2/a^2` term is below
  `5e-15` relative to `R`.
- Meaning: constant-cost, relatively accurate `P` and `Q` for very large
  shapes (for example chi-square with `2e7` degrees of freedom).
- Code: `distributions._special._gamma_temme`,
  `distributions._special.regularized_gamma_p`,
  `distributions._special.regularized_gamma_q`.
- Source: [REF-TEMME-1979] and [REF-NIST-DLMF-GAMMA, Section 8.12].

### EQ-INF-GOF-KS-003 — Exact one-sided Smirnov tail and the two-sided tail

`P(D_n^+ >= d) = d sum_(j=0)^(floor(n(1-d))) C(n,j) (1-d-j/n)^(n-j)
(d+j/n)^(j-1)`; `P(D_n >= d) = 2 P(D_n^+ >= d)` exactly for `d >= 1/2`, and
`|P(D_n >= d) - 2 P(D_n^+ >= d)| <= P(D_n^+ >= d)^2` for every `d`.

- Domain: continuous fully specified null; `n >= 1`. For `n = 1` and
  `d >= 1/2` the tail is `2 (1 - d)`.
- Meaning: the events `D_n^+ >= d` (decreasing in the observations) and
  `D_n^- >= d` (increasing) are disjoint for `d >= 1/2` and negatively
  correlated in general (Harris-FKG inequality for independent variables),
  so doubling the one-sided tail is exact or within `1e-14` once that tail
  is below `1e-7`.
- Code: `inference.goodness_of_fit.smirnov_one_sided_sf`,
  `inference.goodness_of_fit.kolmogorov_exact_sf`.
- Source: [REF-BIRNBAUM-TINGEY-1951] and [REF-ESARY-PROSCHAN-WALKUP-1967].

### EQ-INF-GOF-KS-004 — Exact finite-n Kolmogorov distribution (Noé recursion)

`D_n < d` iff `N(i/n - d) <= i - 1` and `N((i-1)/n + d) >= i` for all `i`,
with `N` the uniform-sample counting process; over the merged boundary
times `t_m`, `Q_(m+1)(c') = sum_(c<=c') Q_m(c) (t_(m+1)-t_m)^(c'-c)/(c'-c)!`
on admissible counts and `P(D_n < d) = n! Q_M(n)`.

- Domain: `n <= 100` in `inference.kolmogorov_smirnov_test` (larger samples
  keep [EQ-INF-GOF-KS-002]); `1/(2n) < d < 1`.
- Meaning: the exact small-sample p-value `1 - P(D_n < d)`; all terms are
  non-negative, and the result agrees with the Marsaglia-Tsang-Wang matrix
  formula in exact arithmetic to about `1e-15`.
- Code: `inference.goodness_of_fit._kolmogorov_noe_cdf`,
  `inference.goodness_of_fit.kolmogorov_exact_sf`.
- Source: [REF-NOE-1972] and [REF-MARSAGLIA-TSANG-WANG-2003].

### EQ-DIST-TN-006 — Short-interval truncated normal moments

On `[c - h, c + h]` (standardized), `w(u) = exp(-c u - u^2/2)`,
`(n+1) q_(n+1) = -c h q_n - h^2 q_(n-1)`, `q_0 = 1`,
`S_k = sum_(n+k even) q_n/(n+k+1)`; `E[Z] = c + h S_1/S_0`,
`Var[Z] = h^2 (S_2/S_0 - (S_1/S_0)^2)`.

- Domain: both ends finite and `w (|alpha| + w) <= 1/2` (`w = 2h`), the
  same regime as [EQ-DIST-TN-005].
- Meaning: moments of a narrow truncation without subtracting nearly equal
  density ratios (the closed form gave a variance of `5.55e-17` instead of
  `3.33e-17` on `(-1e-8, 1e-8)` and was 38% off on `(10, 10.0001)`).
- Code: `distributions.truncated_normal._short_interval_moments`,
  `distributions.TruncatedNormal.mean`, `distributions.TruncatedNormal.variance`.
- Source: [REF-JKB-1994, Chapter 13] and [REF-NIST-DLMF, Section 7.6].

### EQ-INF-DIST-NCT-002 — Adaptive Gauss-Legendre evaluation of the noncentral t

`P(T' <= t) = integral Phi(t s - delta) g_nu(s) ds` by 10-point
Gauss-Legendre panels, bisected until the two halves agree with the panel
estimate within the panel's share of `1e-13`; breakpoints at
`1 + {-3, 0, 3}/sqrt(2 nu)` and `delta/t + {-8, -1, 0, 1, 8}/|t|`; for
`nu < 2` the variable `u = s^nu` gives the bounded integrand
`Phi(t u^(1/nu) - delta) C exp(-nu u^(2/nu)/2)/nu`.

- Domain: `nu >= 1`, finite `t` and `delta != 0`.
- Meaning: resolves both the chi-scale bump and the normal step of width
  `1/|t|`; a fixed 2000-panel Simpson rule erred by up to `7e-4`.
- Code: `inference.noncentral_t_cdf`,
  `inference.sampling_distributions._adaptive_integral`,
  `inference.sampling_distributions._gauss_legendre_rule`.
- Source: [REF-DAVIS-RABINOWITZ-1984, Chapters 2 and 6].

## Causal inference: graphs, interventions, effect estimation, and E-values

### EQ-CAUSAL-DSEP-001 — d-separation via the moralized ancestral graph

`X _||_ Y | Z` in a DAG `G` iff `Z` separates `X` from `Y` in the undirected
moral graph `(G_An(X u Y u Z))^m`: restrict `G` to the ancestral set of
`X u Y u Z`, join every pair of parents of a common child, drop directions,
and delete `Z`.

- Domain: finite DAG; `X`, `Y` non-empty; `X`, `Y`, `Z` pairwise disjoint.
- Meaning: every path between `X` and `Y` is blocked by `Z` (a non-collider
  in `Z`, or a collider with neither itself nor a descendant in `Z`); then the
  conditional independence holds in every distribution Markov to `G`.
- Code: `causal.CausalGraph.is_d_separated`.
- Source: [REF-PEARL-2009, Section 1.2.3], [REF-LAURITZEN-ETAL-1990].

### EQ-CAUSAL-BD-001 — Backdoor criterion

`Z` satisfies the backdoor criterion relative to `(X, Y)` iff
`Z n De(X) = {}` and `X _||_ Y | Z` in `G_underline(X)` (the graph with all
edges leaving `X` removed). Every minimal such `Z` lies in
`An_{G_underline(X)}(X u Y)`.

- Domain: [EQ-CAUSAL-DSEP-001]; `Z` disjoint from `X u Y`.
- Meaning: `Z` blocks every path into `X` that could carry confounding and
  opens no new path; then [EQ-CAUSAL-ADJ-001] identifies `P(y | do(x))`.
- Code: `causal.satisfies_backdoor`, `causal.minimal_adjustment_sets`.
- Source: [REF-PEARL-2009, Definition 3.3.1 and Theorem 3.3.2],
  [REF-LAURITZEN-ETAL-1990].

### EQ-CAUSAL-FDC-001 — Frontdoor criterion

`M` satisfies the frontdoor criterion relative to `(X, Y)` iff (i) every
directed path `X -> ... -> Y` meets `M`; (ii) `X _||_ M` in
`G_underline(X)`; (iii) `M _||_ Y | X` in `G_underline(M)`.

- Domain: single `X` and `Y`; non-empty `M` disjoint from `{X, Y}`.
- Meaning: the effect is transmitted only through `M`, `X -> M` is
  unconfounded, and `M -> Y` is deconfounded by `X`; then
  [EQ-CAUSAL-FD-001] identifies `P(y | do(x))`.
- Code: `causal.satisfies_frontdoor`.
- Source: [REF-PEARL-2009, Definition 3.3.3 and Theorem 3.3.4].

### EQ-CAUSAL-FACT-001 — Markov factorization of a causal Bayesian network

`P(v_1, ..., v_n) = prod_i P(v_i | pa_i)`.

- Domain: finite variables; one conditional table per variable given its
  parents in a DAG.
- Meaning: the observational joint law implied by the causal graph and its
  conditional tables.
- Code: `causal.DiscreteCausalModel.joint`.
- Source: [REF-PEARL-2009, Section 1.2].

### EQ-CAUSAL-TRUNC-001 — Truncated factorization under intervention

`P(v | do(X = x)) = prod_{i: V_i not in X} P(v_i | pa_i)` if `v` agrees with
`x` on `X`, and `0` otherwise.

- Domain: [EQ-CAUSAL-FACT-001]; `x` in the domain of `X`.
- Meaning: an intervention replaces the mechanisms of `X` by constants and
  leaves every other mechanism unchanged (graph surgery).
- Code: `causal.DiscreteCausalModel.interventional_joint`,
  `causal.DiscreteCausalModel.interventional_distribution`,
  `causal.DiscreteCausalModel.intervene`.
- Source: [REF-PEARL-2009, Section 3.2], [REF-HERNAN-ROBINS-2020, Chapter 6].

### EQ-CAUSAL-ADJ-001 — Backdoor adjustment formula

`P(y | do(x)) = sum_z P(y | x, z) P(z)`.

- Domain: `Z` satisfies [EQ-CAUSAL-BD-001]; positivity `P(x, z) > 0`
  whenever `P(z) > 0`.
- Meaning: standardize the stratum-specific outcome law to the population
  distribution of `Z` instead of its distribution among the treated. The
  naive `P(y | x) = sum_z P(y | x, z) P(z | x)` can reverse sign (Simpson).
- Code: `causal.backdoor_adjustment`, `causal.compare_naive_and_adjusted`.
- Source: [REF-PEARL-2009, Theorem 3.3.2 and Section 6.1],
  [REF-HERNAN-ROBINS-2020, Chapter 2].

### EQ-CAUSAL-FD-001 — Frontdoor adjustment formula

`P(y | do(x)) = sum_m P(m | x) sum_x' P(y | x', m) P(x')`.

- Domain: `M` satisfies [EQ-CAUSAL-FDC-001]; `P(x) > 0` and `P(x', m) > 0`
  whenever `P(m | x) > 0` and `P(x') > 0`.
- Meaning: identifies the effect despite an unobserved treatment-outcome
  confounder by chaining two unconfounded steps.
- Code: `causal.frontdoor_adjustment`.
- Source: [REF-PEARL-2009, Theorem 3.3.4].

### EQ-CAUSAL-PS-001 — Logistic propensity score

`e(x) = P(T = 1 | X = x) = 1 / (1 + exp(-(b_0 + x'b)))`, fitted by maximum
likelihood ([EQ-GLM-IRLS-001]).

- Domain: binary `T`; both arms present; no separation.
- Meaning: the coarsest balancing score; conditional exchangeability given
  `X` implies exchangeability given `e(X)`.
- Code: `causal.fit_propensity_scores`, and internally `causal.estimate_ipw`,
  `causal.estimate_aipw`.
- Source: [REF-ROSENBAUM-RUBIN-1983].

### EQ-CAUSAL-REG-001 — Regression adjustment (standardization)

`tau_reg = (1/n) sum_i [m_1(x_i) - m_0(x_i)]` with `m_t` the least-squares
fit of `Y` on `(1, X)` within arm `t`.

- Domain: each arm has more units than coefficients; conditional
  exchangeability and positivity given `X`.
- Meaning: predicts both potential outcomes for everyone and averages;
  consistent when the outcome models are correct.
- Code: `causal.estimate_regression_adjustment`, and the outcome part of
  `causal.estimate_aipw`.
- Source: [REF-HERNAN-ROBINS-2020, Chapter 13],
  [REF-IMBENS-RUBIN-2015, Chapter 12].

### EQ-CAUSAL-IPW-001 — Inverse-probability-weighted ATE

Horvitz-Thompson: `tau_HT = (1/n) sum_i [T_i Y_i / e_i - (1 - T_i) Y_i /
(1 - e_i)]`. Hajek: `tau_H = sum T Y / e / sum T / e - sum (1-T) Y / (1-e) /
sum (1-T) / (1-e)`. Standard error `sd(psi_i) / sqrt(n)` with the
corresponding influence values, treating `e` as known. Trimming keeps units
with `a <= e_i <= 1 - a`.

- Domain: `0 < e_i < 1`; after trimming the estimand is the ATE of the
  overlap population.
- Meaning: reweighting creates a pseudo-population in which `T` is
  independent of `X`.
- Code: `causal.estimate_ipw`.
- Source: [REF-HERNAN-ROBINS-2020, Chapter 12],
  [REF-IMBENS-RUBIN-2015, Chapter 16].

### EQ-CAUSAL-SW-001 — Stabilized weights

`SW_i = T_i p / e_i + (1 - T_i)(1 - p) / (1 - e_i)` with `p = mean(T)`; the
`SW`-weighted arm means give the Hajek estimator of [EQ-CAUSAL-IPW-001].

- Domain: [EQ-CAUSAL-IPW-001].
- Meaning: the numerator `P(T = t)` keeps the mean weight near one and the
  weights less variable without changing the saturated-model estimate.
- Code: `causal.estimate_ipw` with `stabilized=True`.
- Source: [REF-HERNAN-ROBINS-2020, Chapter 12].

### EQ-CAUSAL-AIPW-001 — Augmented IPW (doubly robust) estimator

`psi_i = m_1(x_i) - m_0(x_i) + T_i (Y_i - m_1(x_i)) / e_i -
(1 - T_i)(Y_i - m_0(x_i)) / (1 - e_i)`, `tau_AIPW = mean(psi)`,
`se = sd(psi) / sqrt(n)`.

- Domain: [EQ-CAUSAL-PS-001] and [EQ-CAUSAL-REG-001].
- Meaning: consistent if either the propensity model or the outcome model
  is correct; locally efficient when both are.
- Code: `causal.estimate_aipw`.
- Source: [REF-ROBINS-ROTNITZKY-ZHAO-1994],
  [REF-HERNAN-ROBINS-2020, Chapter 13].

### EQ-CAUSAL-SMD-001 — Standardized mean difference

`SMD_j = (xbar_1j - xbar_0j) / sqrt((s_1j^2 + s_0j^2) / 2)`, with weighted
arm means when weights are supplied and unweighted arm variances in the
denominator.

- Domain: each arm has at least two units; the pooled variance is positive.
- Meaning: scale-free covariate imbalance; values near zero after weighting
  indicate the weights balance that covariate's mean.
- Code: `causal.standardized_mean_differences`.
- Source: [REF-IMBENS-RUBIN-2015, Chapter 14].

### EQ-CAUSAL-EVALUE-001 — E-value

For `RR >= 1`: `E = RR + sqrt(RR (RR - 1))`; for `RR < 1` apply to `1/RR`.
The interval E-value uses the limit closest to one and is `1` if the
interval contains one.

- Domain: `RR > 0` on the risk-ratio scale.
- Meaning: the minimum risk-ratio association an unmeasured confounder must
  have with both treatment and outcome to explain the estimate away.
- Code: `causal.e_value`.
- Source: [REF-VANDERWEELE-DING-2017].

### EQ-CAUSAL-BIAS-001 — Bounding factor for unmeasured confounding

`B = RR_EU RR_UD / (RR_EU + RR_UD - 1)`; `B(E, E) = RR` for the E-value of
[EQ-CAUSAL-EVALUE-001].

- Domain: `RR_EU >= 1`, `RR_UD >= 1`.
- Meaning: the largest factor by which a confounder of the given strengths
  can inflate a risk ratio.
- Code: `causal.confounding_bias_factor`.
- Source: [REF-VANDERWEELE-DING-2017].

### EQ-CAUSAL-RRAPPROX-001 — Approximate conversions to the risk-ratio scale

`RR ~ OR` (rare outcome), `RR ~ sqrt(OR)` (common outcome), and
`RR ~ exp(0.91 d)` for a standardized mean difference `d`.

- Domain: approximations used only to place other effect measures on the
  E-value scale.
- Meaning: lets [EQ-CAUSAL-EVALUE-001] be applied to odds ratios and
  continuous outcomes.
- Code: `causal.risk_ratio_from_odds_ratio`,
  `causal.risk_ratio_from_standardized_difference`.
- Source: [REF-VANDERWEELE-DING-2017].

## Optimal and Bayesian experimental design

### EQ-OED-INFO-001 — Information matrix of an approximate design

`M(xi) = sum_i w_i f(x_i) f(x_i)'` for the design
`xi = {(x_i, w_i)}`, `w_i >= 0`, `sum_i w_i = 1`; an exact design with
`n_i` runs at `x_i` has `w_i = n_i / n` and Fisher information
`n M(xi) / sigma^2`.

- Domain: regression model `E[y|x] = f(x)' beta` with independent
  homoscedastic errors; `f: X -> R^p`.
- Meaning: the per-run information about `beta`; `sigma^2 M^-1 / n` is the
  covariance of the least-squares estimate.
- Code: `design.information_matrix`, `design.ApproximateDesign`,
  `design.ExactDesign.as_approximate`.
- Source: [REF-ATKINSON-DONEV-TOBIAS-2007], [REF-FEDOROV-1972].

### EQ-OED-CRIT-001 — D-, A-, E-, and I-optimality criteria

`D: max log det M`; `A: min tr M^-1`; `E: max lambda_min(M)`;
`I: min (1/K) sum_k f(z_k)' M^-1 f(z_k) = tr(M^-1 M_Z)` over region points
`z_k`. D-efficiency `(det M(xi) / det M(xi_ref))^(1/p)`.

- Domain: [EQ-OED-INFO-001]; singular `M` gives `D = -inf`, `A = I = inf`.
- Meaning: D shrinks the confidence-ellipsoid volume, A the average
  parameter variance, E the worst-direction variance, I the average
  prediction variance over the region.
- Code: `design.design_criterion`, `design.d_efficiency`.
- Source: [REF-ATKINSON-DONEV-TOBIAS-2007], [REF-FEDOROV-1972].

### EQ-OED-KW-001 — Kiefer-Wolfowitz equivalence theorem

`d(x, xi) = f(x)' M(xi)^-1 f(x)`; `sum_i w_i d(x_i, xi) = p`, so
`max_x d(x, xi) >= p`, with equality iff `xi` is D-optimal (equivalently
G-optimal); then `d(x_i, xi) = p` on the support. Concavity of `log det`
gives `log det M* - log det M(xi) <= max_x d(x, xi) - p`, i.e. D-efficiency
`>= exp(1 - max_x d / p)`.

- Domain: [EQ-OED-INFO-001] with nonsingular `M(xi)`; the maximum is over
  the design space (here a finite candidate set).
- Meaning: a computable certificate of optimality and an efficiency bound
  for any design.
- Code: `design.equivalence_check`, `design.standardized_variances`,
  `design.OptimalDesignResult.equivalence`.
- Source: [REF-KIEFER-WOLFOWITZ-1960], [REF-ATKINSON-DONEV-TOBIAS-2007].

### EQ-OED-MULT-001 — Multiplicative algorithm for D-optimal weights

`w_i^(t+1) = w_i^(t) d(x_i, w^(t)) / p` on a finite candidate set; stop when
`max_i d(x_i, w) <= p (1 + tol)`.

- Domain: all starting weights positive with nonsingular `M`.
- Meaning: weights grow where the prediction variance exceeds `p` and shrink
  elsewhere; they stay normalized because `sum_i w_i d_i = p`, and
  `log det M` increases monotonically.
- Code: `design.d_optimal_design(method="multiplicative")`.
- Source: [REF-SILVEY-TITTERINGTON-TORSNEY-1978],
  [REF-ATKINSON-DONEV-TOBIAS-2007].

### EQ-OED-FW-001 — Fedorov-Wynn vertex-direction step

`x* = argmax_x d(x, xi_t)`, `xi_(t+1) = (1 - alpha) xi_t + alpha delta_x*`,
`alpha = (d(x*) - p) / ((d(x*) - 1) p)`, so
`M_(t+1) = (1 - alpha) M_t + alpha f(x*) f(x*)'`.

- Domain: [EQ-OED-INFO-001], nonsingular `M_t`, `d(x*) > p`.
- Meaning: move mass toward the point of largest prediction variance; the
  step is the exact maximizer of `log det` along that direction (Wynn's
  sequential version uses `alpha_t = 1/(t+1)`).
- Code: `design.d_optimal_design(method="fedorov-wynn")`.
- Source: [REF-FEDOROV-1972], [REF-WYNN-1970].

### EQ-OED-ROUND-001 — Efficient rounding to an exact design

With `l` support points start at `n_i = ceil((n - l/2) w_i)`; while
`sum n_i < n` increment `argmin_j n_j / w_j`; while `sum n_i > n`
decrement `argmax_j (n_j - 1) / w_j`.

- Domain: approximate design with positive weights; integer `n >= 1`.
- Meaning: the apportionment with the smallest efficiency loss for a wide
  class of criteria.
- Code: `design.efficient_rounding`.
- Source: [REF-PUKELSHEIM-RIEDER-1992].

### EQ-OED-EXCH-001 — Fedorov exchange for exact designs

Replace one run `x_i` by a candidate `x` when
`log det M(xi_n - delta_x_i/n + delta_x/n) > log det M(xi_n)`; apply the
best exchange of each pass until none improves.

- Domain: exact `n`-run design with nonsingular `M`; finite candidates.
- Meaning: monotone local search among exact designs; it stops at a design
  no single exchange can improve.
- Code: `design.exchange_exact_design`.
- Source: [REF-FEDOROV-1972], [REF-ATKINSON-DONEV-TOBIAS-2007].

### EQ-OED-LOCAL-001 — Local design basis of a nonlinear model

`f(x) = d eta(x, theta) / d theta |_(theta_0)`. For
`eta = A exp(-k t)` on `t in [0, T]`, `T >= 1/k`, two points with weights
`1/2` give `det M = A^2 exp(-2k(t1 + t2)) (t2 - t1)^2 / 4`, maximized by
`t1 = 0`, `t2 = 1/k`; the equivalence theorem confirms global D-optimality.

- Domain: smooth `eta`; the design is optimal only at the nominal
  `theta_0`, and here it does not depend on `A`.
- Meaning: the asymptotic information of a nonlinear least-squares fit at
  `theta_0` equals that of the linearized model.
- Code: `design.LocalLinearization`.
- Source: [REF-BOX-LUCAS-1959], [REF-ATKINSON-DONEV-TOBIAS-2007].

### EQ-OED-EIG-001 — Expected information gain

`EIG(D) = E_(theta, y | D)[log p(y | theta, D) - log p(y | D)]
= I(theta; y | D) = E_y[KL(p(theta | y, D) || p(theta))]`.

- Domain: proper prior, likelihood with a density; invariant under
  bijective reparameterization of `theta`.
- Meaning: the expected reduction of entropy about `theta` from the
  experiment `D`; the Bayesian analogue of D-optimality.
- Code: `design.expected_information_gain`, `design.greedy_design`,
  `design.nested_monte_carlo_information_gain`.
- Source: [REF-CHALONER-VERDINELLI-1995], [REF-COVER-THOMAS-2006, Chapter 2].

### EQ-OED-EIG-LIN-001 — Closed-form EIG of a linear-Gaussian experiment

For `y = J theta + e`, `theta ~ N(m, Sigma)`, `e ~ N(0, R)`:
`EIG = 1/2 log det(I + Sigma J' R^-1 J) = 1/2 log det(I + B'B)`,
`B = L_R^-1 J L_Sigma`.

- Domain: SPD `Sigma`, `R`; independent of `m` and of the data.
- Meaning: Bayesian D-optimality; prior entropy minus posterior entropy.
- Code: `design.linear_gaussian_information_gain`,
  `design.linearized_information_gain`.
- Source: [REF-CHALONER-VERDINELLI-1995], [REF-COVER-THOMAS-2006, Chapter 8].

### EQ-OED-LAP-001 — Laplace (Gauss-Newton) EIG in prior standard space

With `u ~ N(0, I)`, `theta = T^-1(u)` (Nataf map) and whitened Jacobian
`J_u = d[g(T^-1(u)) / s] / du`:
`EIG ~= E_u[1/2 log det(I + J_u' J_u)]`, estimated by Monte Carlo over `u`.

- Domain: known Gaussian noise; posterior approximately Gaussian with mode
  near the data-generating `u`.
- Meaning: prior entropy minus the expected Laplace posterior entropy;
  exact for linear models with Gaussian priors.
- Code: `design.laplace_information_gain`,
  `design.linearized_information_gain`, `design.BayesianDesignProblem.standard_jacobian`.
- Source: [REF-CHALONER-VERDINELLI-1995], [REF-TIERNEY-KADANE-1986].

### EQ-OED-NMC-001 — Nested Monte Carlo EIG estimator

`EIG_hat = (1/N) sum_n [log p(y_n | theta_n)
- log((1/M) sum_m p(y_n | theta_(n,m)))]`, `theta_n ~ p(theta)`,
`y_n ~ p(y | theta_n)`. Fresh inner draws: `E[EIG_hat] >= EIG`, bias
`O(1/M)`, variance `O(1/N)`. Reusing the outer sample as inner sample
(`theta_(n,m) = theta_m`, `M = N`): each term is a prior contrastive bound,
`E[EIG_hat] <= EIG` and `EIG_hat <= log N`.

- Domain: likelihood densities evaluable pointwise; Gaussian constants
  cancel.
- Meaning: consistent as `N, M -> inf`; reuse cuts model evaluations from
  `N (1 + M)` to `N` per design condition at the price of a downward bias.
- Code: `design.nested_monte_carlo_information_gain`.
- Source: [REF-RYAN-2003], [REF-HUAN-MARZOUK-2013],
  [REF-RAINFORTH-ETAL-2018], [REF-FOSTER-ETAL-2020].

### EQ-OED-GREEDY-001 — Greedy sequential selection by EIG

`c_(k+1) = argmax_(c in C) EIG(B + {c_1..c_k} + {c})`, and by the chain rule
`EIG(D + {c}) - EIG(D) = I(theta; y_c | y_D)`.

- Domain: finite candidate set `C`, baseline conditions `B`.
- Meaning: each step buys the largest conditional information; myopic, so
  not guaranteed optimal as a batch.
- Code: `design.greedy_design`.
- Source: [REF-CHALONER-VERDINELLI-1995], [REF-COVER-THOMAS-2006, Chapter 2].

### EQ-DIST-GPD-001 — Generalized Pareto distribution

`S(x) = exp(-y)`, `y = log1p(xi z)/xi` (`y = z` when `xi z` is zero to
working precision), `z = (x - mu)/sigma >= 0`; `F(x) = -expm1(-y)`,
`log f = -log sigma - (1 + xi) y` and hazard `1/(sigma (1 + xi z))` on
`1 + xi z > 0`.

- Domain: `sigma > 0`, real `xi`; support `[mu, inf)` for `xi >= 0` and
  `[mu, mu - sigma/xi]` for `xi < 0` (`xi = -1` is uniform).
- Meaning: the Pickands-Balkema-de Haan limit law of excesses over a high
  threshold `mu`; `xi = 0` is the exponential law with mean `sigma`, reached
  continuously through the `log1p` form.
- Code: `distributions.GeneralizedPareto.log_pdf`,
  `distributions.GeneralizedPareto.pdf`, `distributions.GeneralizedPareto.cdf`,
  `distributions.GeneralizedPareto.sf`, `distributions.GeneralizedPareto.log_sf`,
  `distributions.GeneralizedPareto.hazard`.
- Source: [REF-PICKANDS-1975], [REF-BALKEMA-DEHAAN-1974],
  [REF-COLES-2001, Section 4.2].

### EQ-DIST-GPD-002 — GPD quantile and inverse survival

`x = mu + sigma expm1(xi L)/xi` with `L = -log1p(-p)` (quantile) or
`L = -log q` (inverse survival), and `x = mu + sigma L` when `xi L` is zero to
working precision.

- Domain: `0 < p < 1`, `0 < q < 1`.
- Meaning: closed-form inverse used for `quantile`, `isf`, and inverse-
  transform sampling; upper-tail levels keep relative accuracy in `q`.
- Code: `distributions.GeneralizedPareto.quantile`,
  `distributions.GeneralizedPareto.isf`, `distributions.GeneralizedPareto.sample`.
- Source: [REF-COLES-2001, Section 4.2].

### EQ-DIST-GPD-003 — GPD moments and entropy

`E[X] = mu + sigma/(1 - xi)` (`xi < 1`), `Var X = sigma^2/((1 - xi)^2
(1 - 2 xi))` (`xi < 1/2`), and `h(X) = log sigma + xi + 1` for every `xi`
(because `y = -log S(X)` is standard exponential).

- Domain: `sigma > 0`; infinite mean/variance outside the stated ranges.
- Meaning: moments for checking fits; the entropy is linear in `xi`.
- Code: `distributions.GeneralizedPareto.mean`,
  `distributions.GeneralizedPareto.variance`,
  `distributions.GeneralizedPareto.entropy`.
- Source: [REF-HOSKING-WALLIS-1987], [REF-COVER-THOMAS-2006, Chapter 8].

### EQ-EVA-BM-001 — Block maxima and the GEV likelihood

`M_k = max(x_((k-1)b+1), ..., x_(kb))` for complete blocks of size `b`;
`l(mu, sigma, xi) = sum_k log f_GEV(M_k)` with `f_GEV` from
[EQ-DIST-GEV-001].

- Domain: `b >= 1`; an incomplete final block is discarded; the MLE is
  regular for `xi > -1/2`.
- Meaning: by the extremal types theorem block maxima are approximately GEV;
  the maximum-likelihood fit reuses [EQ-INF-MLE-GEN-001].
- Code: `risk.block_maxima`, `risk.fit_gev`.
- Source: [REF-COLES-2001, Sections 3.1 and 3.3], [REF-SMITH-1985].

### EQ-EVA-RL-001 — GEV return level

`z_T = mu - sigma r(-xi, log y_T)` with `y_T = -log(1 - 1/T)` and
`r(a, t) = expm1(a t)/a` (`r(0, t) = t`); equivalently
`z_T = mu - (sigma/xi)(1 - y_T^(-xi))`, and `mu - sigma log y_T` at `xi = 0`.

- Domain: `T > 1` blocks.
- Meaning: the level exceeded by one block maximum with probability `1/T`
  (the `T`-block return level), `z_T = F^-1(1 - 1/T)`.
- Code: `risk.gev_return_level`, `risk.GEVFit.return_level`.
- Source: [REF-COLES-2001, Section 3.3].

### EQ-EVA-RL-002 — Delta-method return-level interval

`grad z_T = (1, -r(-xi, l), sigma r_a(-xi, l))` with `l = log y_T` and
`r_a(a, t) = d r/da = sum_(k>=2) (k-1) t^k a^(k-2)/k!` (series for
`|a t| < 0.05`, otherwise `(a t e^(a t) - expm1(a t))/a^2`);
`Var z_T ~ grad' V grad`, interval `z_T +/- z_(1-alpha/2) sqrt(Var z_T)`.

- Domain: `V` the observed-information covariance of the MLE.
- Meaning: symmetric Wald interval; it ignores the skewness of the
  likelihood for long return periods (compare [EQ-EVA-PROF-001]).
- Code: `risk.gev_return_level_gradient`, `risk.GEVFit.return_level`,
  `risk.PeaksOverThresholdFit.return_level`.
- Source: [REF-COLES-2001, Sections 2.6 and 3.3].

### EQ-EVA-PROF-001 — Profile-likelihood interval for a return level

With `mu = z_T + sigma r(-xi, log y_T)`,
`l_p(z_T) = max_(sigma, xi) l(mu(z_T, sigma, xi), sigma, xi)` and the interval
is `{z : 2 (l_max - l_p(z)) <= chi2_1(1 - alpha)}`,
`chi2_1(c) = Phi^-1((1 + c)/2)^2`.

- Domain: regular GEV likelihood; endpoints bracketed by doubling steps of
  the delta-method standard error and refined by Brent's method.
- Meaning: asymmetric interval that follows the likelihood; typically
  longer upward than the Wald interval for heavy tails.
- Code: `risk.GEVFit.profile_log_likelihood`,
  `risk.GEVFit.profile_return_level`.
- Source: [REF-COLES-2001, Sections 2.6 and 3.3], [REF-BRENT-1973].

### EQ-EVA-GPD-001 — Threshold excesses and the GPD likelihood

`Y = X - u | X > u` is approximately `GPD(sigma_u, xi)` for high `u`;
`l(sigma, xi) = -k log sigma - (1 + 1/xi) sum log(1 + xi y_i/sigma)`
(`-k log sigma - sum y_i/sigma` at `xi = 0`).

- Domain: `y_i >= 0`, `1 + xi y_i/sigma > 0`; regular for `xi > -1/2`,
  unbounded likelihood for `xi < -1`.
- Meaning: the peaks-over-threshold model for the far tail; fitted through
  [EQ-INF-MLE-GEN-001] with a feasible PWM start.
- Code: `risk.threshold_excesses`, `risk.fit_gpd`, `risk.GPD_FAMILY`.
- Source: [REF-PICKANDS-1975], [REF-BALKEMA-DEHAAN-1974],
  [REF-COLES-2001, Section 4.3], [REF-SMITH-1985].

### EQ-EVA-LMOM-001 — Sample L-moments

`b_r = n^-1 sum_j [C(j-1, r)/C(n-1, r)] x_(j)` (unbiased probability-weighted
moments `beta_r = E[X F^r]`); `l1 = b0`, `l2 = 2b1 - b0`,
`l3 = 6b2 - 6b1 + b0`, `l4 = 20b3 - 30b2 + 12b1 - b0`.

- Domain: `n >= 4` ordered observations `x_(1) <= ... <= x_(n)`.
- Meaning: linear analogues of moments; robust and defined whenever the
  mean exists.
- Code: `risk.sample_l_moments`.
- Source: [REF-HOSKING-1990].

### EQ-EVA-PWM-001 — PWM / L-moment estimator of the GPD

With `alpha_s = E[X (1 - F)^s] = sigma/((s+1)(s+1-xi))`:
`xi = 2 - alpha_0/(alpha_0 - 2 alpha_1)`,
`sigma = 2 alpha_0 alpha_1/(alpha_0 - 2 alpha_1)`, estimated with
`a_0 = b_0`, `a_1 = b_0 - b_1`; equivalently `xi = 2 - l1/l2`,
`sigma = l1 (l1/l2 - 1)`.

- Domain: `xi < 1` (finite mean); Hosking-Wallis write `k = -xi`.
- Meaning: closed-form estimator that is competitive with the MLE for small
  samples and `xi` near zero; also used as the MLE start.
- Code: `risk.fit_gpd` (`method="pwm"`).
- Source: [REF-HOSKING-WALLIS-1987], [REF-HOSKING-1990].

### EQ-EVA-MRL-001 — Mean residual life

`e(u) = E[X - u | X > u]`, estimated by the mean of the `n_u` excesses with
limits `e(u) +/- z_(1-alpha/2) s_u/sqrt(n_u)`; under a GPD above `u_0`,
`e(u) = (sigma_(u_0) + xi (u - u_0))/(1 - xi)` is linear in `u > u_0`.

- Domain: `xi < 1`; thresholds with fewer than `min_exceedances` skipped.
- Meaning: threshold choice: the lowest `u` above which the plot is
  approximately linear.
- Code: `risk.mean_residual_life`.
- Source: [REF-COLES-2001, Section 4.3].

### EQ-EVA-STAB-001 — Threshold stability of GPD parameters

`sigma*(u) = sigma_u - xi u` and `xi` are constant for `u > u_0`;
`Var sigma* = V_ss - 2u V_sx + u^2 V_xx` (delta method).

- Domain: GPD MLE at each threshold.
- Meaning: threshold choice: the lowest `u` above which `xi` and `sigma*`
  are constant within their intervals.
- Code: `risk.threshold_stability`.
- Source: [REF-COLES-2001, Section 4.3].

### EQ-EVA-RATE-001 — Exceedance-rate model

`zeta_u = k/n`, `Var zeta_u = zeta_u (1 - zeta_u)/n` (binomial);
annual rate `lambda = n_y zeta_u` with standard error
`n_y sqrt(zeta_u (1 - zeta_u)/n)`; `P(N_t >= 1) = 1 - exp(-lambda t)` under
a Poisson count.

- Domain: `0 < k <= n`, `n_y > 0` observations per year.
- Meaning: frequency part of the POT model, combined with the GPD severity.
- Code: `risk.ExceedanceRate`, `risk.exceedance_rate`.
- Source: [REF-COLES-2001, Section 4.3].

### EQ-EVA-POT-001 — POT return level

`x_N = u + sigma r(xi, log(m zeta_u))`, `m = N n_y`,
`r(a, t) = expm1(a t)/a`; i.e. `u + (sigma/xi)((m zeta_u)^xi - 1)`, and
`u + sigma log(m zeta_u)` at `xi = 0`. The yearly exceedance rate of a level
`x >= u` is `lambda S_GPD(x)`.

- Domain: `m zeta_u > 1` (otherwise the level lies below `u`).
- Meaning: the level exceeded on average once every `N` years.
- Code: `risk.pot_return_level`, `risk.PeaksOverThresholdFit.return_level`,
  `risk.PeaksOverThresholdFit.annual_exceedance_rate`.
- Source: [REF-COLES-2001, Section 4.3].

### EQ-EVA-POT-002 — Delta-method POT return-level interval

`grad x_N = (sigma e^(xi l)/zeta_u, r(xi, l), sigma r_a(xi, l))` in
`(zeta_u, sigma, xi)` with `l = log(m zeta_u)`; `V` is block diagonal with
`Var zeta_u` and the GPD observed-information covariance.

- Domain: as [EQ-EVA-POT-001]; unavailable for PWM fits (no covariance).
- Meaning: Wald interval for the `N`-year level.
- Code: `risk.PeaksOverThresholdFit.return_level`.
- Source: [REF-COLES-2001, Section 4.3].

### EQ-EVA-DECL-001 — Runs declustering and the extremal index

A cluster starts at an exceedance of `u` and ends after `r` consecutive
non-exceedances; one peak (cluster maximum) is kept per cluster. The runs
estimate of the extremal index is `theta = n_clusters/n_exceedances`.

- Domain: run length `r >= 1`; results depend on `(u, r)`.
- Meaning: removes short-range dependence before fitting the GPD to
  (approximately independent) cluster peaks.
- Code: `risk.decluster_runs`, `risk.DeclusteredExceedances.extremal_index`,
  `risk.fit_pot`.
- Source: [REF-COLES-2001, Section 5.3].

### EQ-INFO-H-001 — Shannon entropy and information units

`H(X) = -sum_x p(x) log p(x)` with `0 log 0 = 0`; one bit equals `log 2`
nats.

- Domain: finite PMF summing to one within `1e-9`.
- Meaning: average surprise; `0 <= H <= log |support|`.
- Code: `information.entropy`, `information._units.unit_factor`.
- Source: [REF-COVER-THOMAS-2006, Section 2.1].

### EQ-INFO-H-002 — Joint and conditional entropy (chain rule)

`H(X, Y) = -sum p(x, y) log p(x, y)`,
`H(Y | X) = -sum p(x, y) log(p(x, y)/p(x))`, and
`H(X, Y) = H(X) + H(Y | X)`.

- Domain: rectangular joint PMF with rows indexing `X`.
- Meaning: uncertainty left in `Y` once `X` is known.
- Code: `information.joint_entropy`, `information.conditional_entropy`,
  `information.marginals`.
- Source: [REF-COVER-THOMAS-2006, Section 2.2].

### EQ-INFO-MI-001 — Mutual information

`I(X; Y) = sum p(x, y) log(p(x, y)/(p(x) p(y))) = H(X) + H(Y) - H(X, Y)
= D(p_XY || p_X p_Y) >= 0`, with equality iff `X` and `Y` are independent.

- Domain: joint PMF; rounding-level negative sums are returned as zero.
- Meaning: reduction of uncertainty in one variable from observing the
  other.
- Code: `information.mutual_information`.
- Source: [REF-COVER-THOMAS-2006, Section 2.4].

### EQ-INFO-KL-001 — Kullback-Leibler divergence and cross-entropy

`D(p || q) = sum p log(p/q) >= 0` (Gibbs), `inf` if `q_i = 0 < p_i`;
`H(p, q) = -sum p log q = H(p) + D(p || q)`.

- Domain: two PMFs on one finite support.
- Meaning: expected log-likelihood ratio; extra code length when coding `p`
  with a code built for `q`. Not symmetric.
- Code: `information.kl_divergence`, `information.cross_entropy`.
- Source: [REF-COVER-THOMAS-2006, Section 2.3].

### EQ-INFO-JS-001 — Jensen-Shannon divergence

`JS(p, q) = D(p || m)/2 + D(q || m)/2`, `m = (p + q)/2`;
`0 <= JS <= log 2`.

- Domain: two PMFs on one finite support.
- Meaning: symmetric, always finite divergence; its square root is a metric.
- Code: `information.jensen_shannon_divergence`.
- Source: [REF-LIN-1991].

### EQ-INFO-DH-001 — Gaussian differential entropy

`h(N(mu, sigma^2)) = log(sigma sqrt(2 pi e))`;
`h(N_d(mu, Sigma)) = (d log(2 pi e) + log det Sigma)/2`.

- Domain: `sigma > 0`; `Sigma` symmetric positive definite.
- Meaning: the maximum entropy among laws with a given covariance.
- Code: `information.normal_entropy`,
  `information.multivariate_normal_entropy`,
  `information.differential_entropy`.
- Source: [REF-COVER-THOMAS-2006, Sections 8.1 and 8.4].

### EQ-INFO-DH-002 — Uniform and exponential differential entropy

`h(U(a, b)) = log(b - a)`; `h(Exp(lambda)) = 1 - log lambda`.

- Domain: `b > a`; `lambda > 0`.
- Meaning: differential entropy can be negative (`b - a < 1`).
- Code: `information.uniform_entropy`, `information.exponential_entropy`.
- Source: [REF-COVER-THOMAS-2006, Section 8.1].

### EQ-INFO-DH-003 — Gamma and beta differential entropy

`h(Gamma(k, lambda)) = k - log lambda + log Gamma(k) + (1 - k) psi(k)`;
`h(Beta(a, b)) = log B(a, b) - (a-1) psi(a) - (b-1) psi(b)
+ (a+b-2) psi(a+b)`.

- Domain: positive shapes and rate; `psi` is the digamma function.
- Meaning: from `E[log X] = psi(k) - log lambda` (gamma) and
  `E[log X] = psi(a) - psi(a+b)` (beta).
- Code: `information.gamma_entropy`, `information.beta_entropy`.
- Source: [REF-COVER-THOMAS-2006, Chapter 8], [REF-NIST-DLMF-GAMMA,
  Section 5.2].

### EQ-INFO-DH-004 — Lognormal differential entropy

`h(X) = mu + log(sigma sqrt(2 pi e))` for `log X ~ N(mu, sigma^2)`.

- Domain: `sigma > 0`.
- Meaning: `h(e^Y) = h(Y) + E[Y]` (change of variables).
- Code: `information.lognormal_entropy`.
- Source: [REF-COVER-THOMAS-2006, Chapter 8].

### EQ-INFO-DH-005 — GEV and Gumbel differential entropy

`h(GEV(mu, sigma, xi)) = log sigma + gamma_E (1 + xi) + 1`; the Gumbel case
is `xi = 0`.

- Domain: `sigma > 0`, real `xi`.
- Meaning: with `T = exp(-y)` standard exponential, `E[y] = gamma_E` and
  `E[T] = 1` in `-E log f = log sigma + (1 + xi) E[y] + E[T]`.
- Code: `information.gev_entropy`, `information.differential_entropy`.
- Source: [REF-COLES-2001, Section 3.1], [REF-COVER-THOMAS-2006, Chapter 8].

### EQ-INFO-KLG-001 — KL divergence between univariate normals

`D(N(m0, s0^2) || N(m1, s1^2)) = log(s1/s0) + (s0^2 + (m0 - m1)^2)/(2 s1^2)
- 1/2`.

- Domain: `s0, s1 > 0`.
- Meaning: closed form used to check numerical integration and estimators.
- Code: `information.kl_normal`.
- Source: [REF-COVER-THOMAS-2006, Section 8.5].

### EQ-INFO-KLG-002 — KL divergence between multivariate normals

`D = (tr(S1^-1 S0) + (m1 - m0)' S1^-1 (m1 - m0) - d + log det S1
- log det S0)/2`, evaluated with Cholesky factors.

- Domain: symmetric positive-definite `S0`, `S1` of one dimension `d`.
- Meaning: reduces to a sum of [EQ-INFO-KLG-001] terms for diagonal
  covariances.
- Code: `information.kl_multivariate_normal`.
- Source: [REF-COVER-THOMAS-2006, Section 8.5].

### EQ-INFO-KLE-001 — Kozachenko-Leonenko k-NN entropy estimator

`h = psi(n) - psi(k) + log V_d + (d/n) sum_i log r_i`, `r_i` the distance from
point `i` to its `k`-th nearest neighbour, `V_d = pi^(d/2)/Gamma(d/2 + 1)`
(Euclidean) or `2^d` (max norm).

- Domain: `1 <= k < n`, continuous data without duplicated points;
  exhaustive `O(n^2 d)` neighbour search.
- Meaning: local uniform-density approximation inside each `k`-NN ball;
  consistent, with bias decaying in `n`.
- Code: `information.kozachenko_leonenko_entropy`.
- Source: [REF-KOZACHENKO-LEONENKO-1987],
  [REF-KRASKOV-STOGBAUER-GRASSBERGER-2004].

### EQ-INFO-KSG-001 — KSG mutual-information estimator

`I = psi(k) + psi(n) - n^-1 sum_i [psi(n_x(i) + 1) + psi(n_y(i) + 1)]`, where
`eps_i` is the max-norm distance to the `k`-th joint neighbour and
`n_x(i)`, `n_y(i)` count other points strictly within `eps_i` in each
marginal space.

- Domain: `1 <= k < n`, paired continuous samples without duplicated pairs;
  `O(n^2)` search.
- Meaning: algorithm 1 of Kraskov et al.; using one radius for joint and
  marginal spaces makes the k-NN biases largely cancel.
- Code: `information.ksg_mutual_information`.
- Source: [REF-KRASKOV-STOGBAUER-GRASSBERGER-2004].

## Kernel density estimation

### EQ-STAT-KDE-001 — Gaussian kernel density estimate and rule-of-thumb bandwidths

`f_h(x) = (1/(n h)) sum_i phi((x - x_i)/h)` and
`F_h(x) = (1/n) sum_i Phi((x - x_i)/h)`, with bandwidth
`h = 0.9 min(s, IQR/1.34) n^(-1/5)` (Silverman's robust rule) or
`h = 1.06 s n^(-1/5)` (normal reference rule).

- Domain: at least two finite observations, not all equal; `s` is the sample
  standard deviation and `IQR` the type-7 interquartile range (a zero IQR
  falls back to `s`).
- Meaning: a smooth, everywhere-positive estimate of the sampling density;
  the normal rule minimizes the asymptotic mean integrated squared error for
  normal data and oversmooths multimodal data, which the robust rule
  mitigates.
- Code: `statistics.density.gaussian_kde`, `statistics.density.kde_bandwidth`,
  `statistics.density.KernelDensity`.
- Source: [REF-SILVERMAN-1986, Sections 3.3 and 3.4.2].

## Model discrepancy and heteroscedastic calibration (Kennedy-O'Hagan)

### EQ-KOH-MODEL-001 — Kennedy-O'Hagan observation model

`y_i = eta(c_i, theta) + delta(x_i) + eps_i`, `delta ~ GP(0, k_psi)`,
`eps_i ~ N(0, sigma_i^2)` independent of each other and of `delta`; `x_i` is
the subset of the conditions `c_i` the discrepancy depends on.

- Domain: one scalar observed quantity; simulator evaluated directly (no
  emulator); zero-mean discrepancy; no simulator scale factor `rho`.
- Meaning: the simulator at the best-fitting `theta` differs from reality by
  a smooth unknown function plus measurement error.
- Code: `inverse.DiscrepancyCalibration`,
  `inverse.DiscrepancyCalibration.predictions`.
- Source: [REF-KENNEDY-OHAGAN-2001] and [REF-HIGDON-ETAL-2004].

### EQ-KOH-KERNEL-001 — Discrepancy covariance kernels

`k_psi(x, x') = sigma_delta^2 exp(-r^2/2)` (squared exponential) or
`sigma_delta^2 (1 + sqrt(5) r + 5 r^2/3) exp(-sqrt(5) r)` (Matern 5/2), with
`r^2 = sum_i ((x_i - x'_i)/ell_i)^2` and `psi = (sigma_delta, ell_1..ell_p)`.

- Domain: `sigma_delta >= 0`, `ell_i > 0`.
- Meaning: stationary prior on the discrepancy; `sigma_delta` sets its size,
  `ell_i` its smoothness scale along input `i`.
- Code: `inverse.discrepancy_kernel`,
  `inverse.DiscrepancyCalibration.discrepancy_covariance`.
- Source: [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

### EQ-KOH-MARG-001 — Marginal likelihood with the discrepancy integrated out

`C_psi = K_psi + diag(sigma_i^2) = L L'`, `r = y - eta(C, theta)`,
`log p(y | theta, psi, sigma) = -(1/2)|L^-1 r|^2 - sum_i log L_ii
- (n/2) log 2pi`. Without an active discrepancy `C_psi` is diagonal and the
value is `sum_i -(1/2)(log 2pi + 2 log sigma_i + r_i^2/sigma_i^2)`.

- Domain: positive-definite `C_psi` (guaranteed in exact arithmetic when every
  `sigma_i > 0`); a failed factorization returns `-inf`. Cost `O(n^3)`.
- Meaning: `delta` is Gaussian and enters linearly, so it can be integrated
  out exactly; the remaining parameters are `(theta, psi, sigma)`.
- Code: `inverse.DiscrepancyCalibration.log_likelihood`,
  `inverse.DiscrepancyCalibration.marginal_covariance`.
- Source: [REF-KENNEDY-OHAGAN-2001] and
  [REF-RASMUSSEN-WILLIAMS-2006, Section 2.2].

### EQ-KOH-POST-001 — Posterior of parameters and log hyperparameters

`log p(theta, eta, beta | y) = log L + log pi(theta)
+ sum_k [log p_k(e^eta_k) + eta_k] + sum_j log p_j(beta_j) + const`, with
`eta = (log sigma_delta, log ell)` for free hyperparameters and `beta =
(a, b)` the noise coefficients. In sampling coordinates `theta = T^-1(u)` and
`log pi(theta)` is replaced by `-|u|^2/2`.

- Domain: proper priors; hyperparameter priors on `(0, inf)`.
- Meaning: `+ eta_k` is the Jacobian of the log transform; maximizing gives
  MAP (or, without priors, the MLE) and MCMC samples the joint posterior.
- Code: `inverse.DiscrepancyCalibration.log_prior`,
  `inverse.DiscrepancyCalibration.log_posterior`,
  `inverse.DiscrepancyCalibration.standard_log_posterior`,
  `inverse.DiscrepancyCalibration.fit`,
  `inverse.DiscrepancyCalibration.sample_posterior`.
- Source: [REF-KENNEDY-OHAGAN-2001], [REF-HIGDON-ETAL-2004], and
  [REF-GELMAN-ETAL-2013, Section 1.3].

### EQ-KOH-PRED-001 — Bias-corrected and simulator-only prediction

Per draw `s`: `m_s = k*' C_psi^-1 r_s`, `v_s = k(x*, x*) - |L^-1 k*|^2`; the
bias-corrected process is `S^-1 sum_s N(eta(c*, theta_s) + m_s, v_s)` with
mean `mean_s(eta_s + m_s)` and variance `Var_s(eta_s + m_s) + mean_s(v_s)`;
the simulator alone is the sample `eta(c*, theta_s)`; a new observation adds
`sigma(c*)^2`. Interval limits invert the mixture CDF by bisection.

- Domain: positive-definite `C_psi` at every draw; marginal (pointwise)
  predictions only.
- Meaning: Gaussian conditioning of `delta(x*)` on the residuals, averaged
  over parameter uncertainty; the gap between the two predictions is the
  estimated discrepancy contribution.
- Code: `inverse.DiscrepancyCalibration.predict`,
  `inverse.DiscrepancyPrediction.interval`.
- Source: [REF-KENNEDY-OHAGAN-2001] and
  [REF-RASMUSSEN-WILLIAMS-2006, Section 2.2].

### EQ-KOH-HET-001 — Log-linear heteroscedastic noise

`sigma(c) = exp(a + b' g(c))`, with user covariates `g(c)` and unconstrained
coefficients `(a, b)` estimated jointly with `theta`.

- Domain: finite covariates; `b = ()` gives an unknown constant noise level.
- Meaning: multiplicative heteroscedasticity; the log link keeps every
  standard deviation positive.
- Code: `inverse.LogLinearNoise`,
  `inverse.DiscrepancyCalibration.noise_standard_deviations`.
- Source: [REF-HARVEY-1976].

### EQ-KOH-IDENT-001 — Theta-discrepancy confounding diagnostics

`rho_jk = Cov(theta_j, eta_k) / (sd(theta_j) sd(eta_k))` from posterior draws
or from the Hessian covariance, and `ratio_j = sd_post(theta_j) /
sd_prior(theta_j)`.

- Domain: at least two posterior draws, or a positive-definite Hessian.
- Meaning: large `|rho|` or a ratio near one signals that the data cannot
  separate `theta` from `delta`; small values do not prove identifiability.
- Code: `inverse.DiscrepancyCalibration.identifiability`,
  `inverse.DiscrepancyFit.correlation`.
- Source: [REF-BRYNJARSDOTTIR-OHAGAN-2014] and [REF-KENNEDY-OHAGAN-2001].

## Visualization diagnostics

### EQ-VIZ-QQBAND-001 — Pointwise Q-Q band from uniform order statistics

`U_(i) ~ Beta(i, n - i + 1)` for the `i`-th smallest of `n` independent
uniforms; the pointwise band for the `i`-th ordered observation is
`(F^-1(b_(alpha/2)), F^-1(b_(1 - alpha/2)))` with `b_q` the `q`-quantile of that
beta law and `F` the model CDF.

- Domain: continuous model law with a quantile function; `1 <= i <= n`;
  `0 < alpha < 1`. Limits are pointwise, not simultaneous.
- Meaning: an ordered observation outside its band is unusual for its rank
  under the model; about `alpha n` points fall outside by chance.
- Code: `visualization.extremes_plots.order_statistic_band`,
  `visualization.extremes_plots.plot_tail_quantiles`.
- Source: [REF-CASELLA-BERGER-2002, Section 5.4].

### EQ-EM-001 — EM iteration and its monotonicity

`theta_{t+1} = argmax_theta Q(theta | theta_t)`,
`Q(theta | theta_t) = E[log p(y, Z | theta) | y, theta_t]`, and
`l(theta_{t+1}) >= l(theta_t)` for the observed-data log-likelihood `l`.

- Domain: an exact E-step and an M-step that maximizes `Q` (or a penalized
  `Q` whose penalty is fixed across iterations, in which case the inequality
  holds for the penalized objective).
- Meaning: `Q(theta | theta_t) - l(theta)` is maximized at `theta_t`
  (Jensen), so each step cannot decrease `l`; limit points are stationary
  points under Wu's conditions, not necessarily global maxima. The driver
  flags a decrease larger than `slack (1 + |l_t|)`.
- Code: `inference.expectation_maximization`.
- Source: [REF-DEMPSTER-LAIRD-RUBIN-1977], [REF-WU-1983],
  [REF-MCLACHLAN-KRISHNAN-2008].

### EQ-EM-CENS-001 — Observed-data log-likelihood of censored normal data

`l(mu, sigma) = sum_exact [-log sigma - log sqrt(2 pi) - z_i^2/2]
+ sum_censored log(Phi(b_i) - Phi(a_i))`, with `z_i = (y_i - mu)/sigma`,
`a_i = (l_i - mu)/sigma`, `b_i = (u_i - mu)/sigma`.

- Domain: `sigma > 0`; each censored value lies in `[l_i, u_i]` with
  `l_i = -inf` (left), `u_i = inf` (right), or both finite (interval).
- Meaning: exact values contribute densities, censored values contribute
  interval probabilities, evaluated in log space without tail cancellation.
- Code: `inference.censored_normal_log_likelihood`.
- Source: [REF-DEMPSTER-LAIRD-RUBIN-1977], [REF-NIST-DLMF, Section 7.2].

### EQ-EM-CENS-002 — EM steps for censored normal data

E-step: `E[Y_i] = m_i`, `E[Y_i^2] = v_i + m_i^2`, where `m_i`, `v_i` are the
mean and variance of `N(mu_t, sigma_t^2)` truncated to `[l_i, u_i]` (exact
values: `m_i = y_i`, `v_i = 0`). M-step: `mu = (1/n) sum E[Y_i]`,
`sigma^2 = (1/n) sum E[Y_i^2] - mu^2`.

- Domain: at least two finite values to start; positive completed variance.
- Meaning: the complete-data normal MLE applied to completed sufficient
  statistics; truncated moments from [EQ-DIST-TN-003].
- Code: `inference.fit_censored_normal`.
- Source: [REF-DEMPSTER-LAIRD-RUBIN-1977], [REF-MCLACHLAN-KRISHNAN-2008],
  [REF-JKB-1994, Chapter 13].

### EQ-GMM-001 — Gaussian-mixture log-likelihood and responsibilities

`l = sum_i log sum_k w_k N(x_i | m_k, S_k)` and
`r_ik = w_k N(x_i | m_k, S_k) / sum_l w_l N(x_i | m_l, S_l)`, computed with
`log-sum-exp` and Cholesky factors of `S_k`.

- Domain: weights on the simplex, symmetric positive-definite `S_k`.
- Meaning: `r_ik` is the posterior probability that point `i` came from
  component `k`; it is the E-step of mixture EM.
- Code: `inference.GaussianMixture.log_pdf`,
  `inference.GaussianMixture.responsibilities`, `inference.fit_gaussian_mixture`.
- Source: [REF-MCLACHLAN-PEEL-2000], [REF-BISHOP-2006, Section 9.2].

### EQ-GMM-002 — Penalized M-step with a fixed covariance floor

Objective `l_pen = l - (1/2) sum_k tr(Lambda S_k^-1)`,
`Lambda = diag(lambda_j)`, `lambda_j = c var_j`. M-step:
`N_k = sum_i r_ik`, `w_k = N_k/n`, `m_k = sum_i r_ik x_i / N_k`,
`S_k = (sum_i r_ik (x_i - m_k)(x_i - m_k)' + Lambda) / N_k` (diagonal part only
for diagonal covariances).

- Domain: `c >= 0`; with `c > 0` every `S_k` is positive definite and
  `l_pen` is bounded above.
- Meaning: the exact maximizer of `Q - (1/2) sum_k tr(Lambda S_k^-1)`; since
  `Lambda` does not change between iterations, EM is monotone for `l_pen`
  ([EQ-EM-001]). Information criteria use the unpenalized `l`.
- Code: `inference.fit_gaussian_mixture`.
- Source: [REF-MCLACHLAN-PEEL-2000], [REF-DEMPSTER-LAIRD-RUBIN-1977].

### EQ-GMM-003 — k-means++ seeding

First center uniform over the data; each next center is `x_i` with
probability `D(x_i)^2 / sum_j D(x_j)^2`, `D(x)` the distance from `x` to the
nearest chosen center.

- Domain: `1 <= K <= n`; if all distances are zero the choice is uniform.
- Meaning: the `D^2` weighting of Arthur and Vassilvitskii; the library
  follows it with up to 20 Lloyd iterations before forming EM starting values.
- Code: `inference.kmeans_plus_plus`, `inference.fit_gaussian_mixture`.
- Source: [REF-ARTHUR-VASSILVITSKII-2007].

### EQ-GMM-004 — Free parameters of a Gaussian mixture

`p = (K - 1) + K d + K d (d + 1)/2` (full covariances) or
`p = (K - 1) + 2 K d` (diagonal covariances); `AIC = 2p - 2 l`,
`BIC = p log n - 2 l`.

- Domain: maximized unpenalized log-likelihood `l`, `n` data points.
- Meaning: BIC penalizes each extra component by its parameter count; used to
  choose `K` ([EQ-INF-IC-001], [EQ-INF-IC-003]).
- Code: `inference.GaussianMixture.parameter_count`,
  `inference.select_gaussian_mixture`.
- Source: [REF-MCLACHLAN-PEEL-2000], [REF-SCHWARZ-1978], [REF-AKAIKE-1974].

### EQ-HIER-001 — Conditional posteriors of the hierarchical normal model

`mu | tau, y ~ N(mu_hat, V_mu)` with
`V_mu^-1 = [1/s_0^2] + sum_j 1/(s_j^2 + tau^2)`,
`mu_hat = V_mu ([m_0/s_0^2] + sum_j y_j/(s_j^2 + tau^2))` (bracketed terms
only for a `N(m_0, s_0^2)` prior); `theta_j | mu, tau, y ~
N((1 - B_j) y_j + B_j mu, (1 - B_j) s_j^2)`.

- Domain: known `s_j > 0`, `tau >= 0` (at `tau = 0`, `theta_j = mu`).
- Meaning: normal-normal conjugacy given `tau`; integrating `mu` gives
  `E[theta_j | tau, y] = (1 - B_j) y_j + B_j mu_hat`,
  `Var = (1 - B_j) s_j^2 + B_j^2 V_mu`.
- Code: `inference.HierarchicalNormalPosterior.conditional_mu`,
  `inference.HierarchicalNormalPosterior.conditional_theta`,
  `inference.HierarchicalNormalPosterior.sample`.
- Source: [REF-GELMAN-ETAL-2013, Chapter 5], [REF-RUBIN-1981].

### EQ-HIER-002 — Marginal posterior of the population scale tau

`p(tau | y) ∝ p(tau) V_mu^(1/2) prod_j (s_j^2 + tau^2)^(-1/2)
exp(-(y_j - mu_hat)^2 / (2 (s_j^2 + tau^2))) [x exp(-(mu_hat - m_0)^2 /
(2 s_0^2))]`.

- Domain: `tau >= 0`; with a flat `mu` prior and uniform `p(tau)` the tail
  decays like `tau^-(J-1)`, so at least three groups are needed for
  propriety. The library checks the tail slope before integrating.
- Meaning: `theta` and `mu` integrated analytically; evaluated on a grid
  uniform in `u = log(1 + tau/c)` with Simpson weights times the Jacobian
  `tau + c`; draws use the piecewise-linear CDF in `u`.
- Code: `inference.hierarchical_normal_posterior`.
- Source: [REF-GELMAN-ETAL-2013, Chapter 5].

### EQ-HIER-003 — Non-centered parameterization

`x = (mu, eta, z_1..z_J)`, `tau = exp(eta)`, `theta_j = mu + tau z_j`;
`log p(x | y) = log p(mu) + log p(tau) + eta - (1/2) sum_j z_j^2
- (1/2) sum_j (y_j - mu - tau z_j)^2 / s_j^2 + const`.

- Domain: all of `R^(J+2)`; `eta` is the log-Jacobian of `tau = exp(eta)`.
- Meaning: the prior on `z` no longer depends on `tau`, removing the funnel
  between `tau` and `theta` that the centered form has when the data are
  weakly informative; sampled with NUTS and an analytic gradient.
- Code: `inference.non_centered_log_density`,
  `inference.sample_hierarchical_normal`.
- Source: [REF-BETANCOURT-GIROLAMI-2013], [REF-GELMAN-ETAL-2013, Chapter 5].

### EQ-HIER-004 — Partial-pooling shrinkage factor

`B_j(tau) = s_j^2 / (s_j^2 + tau^2)`, `E[B_j | y] = integral B_j(tau)
p(tau | y) dtau`.

- Domain: `s_j > 0`, `tau >= 0`.
- Meaning: weight that `E[theta_j | tau, y]` puts on the pooled mean;
  `B_j = 1` is complete pooling, `B_j = 0` no pooling. Noisier groups shrink
  more.
- Code: `inference.shrinkage_factors`,
  `inference.HierarchicalNormalPosterior.expected_shrinkage`.
- Source: [REF-GELMAN-ETAL-2013, Chapter 5], [REF-MORRIS-1983].

### EQ-HIER-BB-001 — Beta-binomial marginal likelihood

`l(alpha, beta) = sum_j [log C(n_j, y_j) + log B(y_j + alpha, n_j - y_j +
beta) - log B(alpha, beta)]`; posterior mean
`(y_j + alpha)/(n_j + alpha + beta) = (1 - b_j) y_j/n_j + b_j alpha/(alpha +
beta)` with `b_j = (alpha + beta)/(n_j + alpha + beta)`.

- Domain: `alpha, beta > 0`, `0 <= y_j <= n_j`, `n_j > 0`; maximized over
  `(logit m, log kappa)`.
- Meaning: type-II maximum likelihood (empirical Bayes); the plug-in posterior
  ignores uncertainty in `(alpha, beta)`. Without over-dispersion the
  maximum is at `kappa -> inf`.
- Code: `inference.beta_binomial_log_likelihood`, `inference.fit_beta_binomial`.
- Source: [REF-GELMAN-ETAL-2013, Chapter 5], [REF-MORRIS-1983].

### EQ-PPC-001 — Simulation-based posterior predictive p-value

`p_B = (1/S) sum_s 1[T(y_rep^s, theta^s) >= T(y, theta^s)]`,
`y_rep^s ~ p(y | theta^s)`; lower tail with `<=`; Monte Carlo standard error
`sqrt(p_B (1 - p_B) / S_eff)` with `S_eff` the effective sample size of the
indicator chains (or `S` for independent draws).

- Domain: posterior draws `theta^s`, a simulator, data-only statistics
  `T(y)` or realized discrepancies `T(y, theta)`.
- Meaning: tail probability of the observed test quantity under the
  posterior predictive law. Not uniform under a correct model in general
  (conservative toward 0.5); uniform when `T`'s sampling law is free of the
  unknown parameters.
- Code: `inference.posterior_predictive_pvalues`.
- Source: [REF-RUBIN-1984], [REF-GELMAN-MENG-STERN-1996],
  [REF-GELMAN-ETAL-2013, Chapter 6].

### EQ-PPC-002 — Probability integral transform from replicates

`PIT_i = (#{s : y_rep,i^s < y_i} + c_i #{s : y_rep,i^s = y_i}) / S`, with
`c_i = 1/2` or `c_i ~ Uniform(0, 1)` (randomized PIT).

- Domain: `S >= 1` replicated datasets aligned with the observations.
- Meaning: uniform for a calibrated predictive law of independent outcomes.
  When the replicates come from a posterior fitted to the same data (not
  leave-one-out), PIT values are pulled toward `1/2` and a KS test of
  uniformity is descriptive only.
- Code: `inference.pit_values`, `inference.pit_uniformity`.
- Source: [REF-DAWID-1984], [REF-GNEITING-BALABDAOUI-RAFTERY-2007].

### EQ-REL-CE-003 — Finite importance-weight variance of a Gaussian proposal

For `h = N(mu, Sigma)` and `w = phi_n / h`,
`E_h[w^2] = int phi_n(u)^2 / h(u) du` is finite if and only if
`I - Sigma^-1 / 2` is positive definite, i.e. every eigenvalue of `Sigma`
exceeds `1/2`. The CE guard replaces `Sigma = V diag(lambda) V'` by
`V diag(max(lambda_i, lambda_min)) V'` with `lambda_min > 1/2`
(default `0.55`).

- Domain: unbounded failure domains in standard-normal space; the guard is
  applied to every adapted CE covariance.
- Meaning: `phi_n^2 / h` has the quadratic exponent
  `-u'(I - Sigma^-1 / 2) u + O(u)`; when it is not negative definite the
  weights have infinite variance, so the sample variance (and the reported
  c.o.v.) cannot represent the error. Elite covariances of near-planar
  failure domains collapse along the important direction and violate the
  condition.
- Code: `reliability.cross_entropy_importance_sampling`
  (`covariance_floor`, `min_covariance_eigenvalue`, `weights_degenerate`).
- Source: [REF-OWEN-2013, Chapter 9] (importance-sampling variance and
  light-tailed proposals); the eigenvalue condition is a direct Gaussian
  integral.

### EQ-REL-MDP-001 — Bulged limit state for successive design points

`G_b(u) = G(u) + sum_j B_j(u)` with
`B_j(u) = h_j (1 - ||u - u_j*||^2 / r_j^2)^2` for `||u - u_j*|| < r_j` and
`0` otherwise; `r_j = gamma beta_j`, `h_j = delta G(0)`
(defaults `gamma = 1.1`, `delta = 0.75`).

- Domain: safe origin (`G(0) > 0`); known design points with `beta_j > 0`.
- Meaning: each bulge is continuously differentiable, equals `h_j` at the
  known point and vanishes with its gradient on the sphere of radius `r_j`,
  so the known point becomes safe for `G_b` and a new HL-RF search is
  deflected elsewhere; candidates are refined on the original `G`. The
  quartic profile and the `G(0)` height scaling are this library's choices.
- Code: `reliability.bulge_value`, `reliability.find_design_points`.
- Source: [REF-DERKIUREGHIAN-DAKESSIAN-1998] (bulging idea).

### EQ-REL-MDP-002 — Multi-point FORM series approximation

With design points `u_i*`, indices `beta_i`, and directions `alpha_i`,
`P_f ~= P(union_i {alpha_i . u >= beta_i}) = 1 - Phi_m(beta; R)`,
`R_ij = alpha_i . alpha_j`; for `m = 2`,
`P_f ~= Phi(-beta_1) + Phi(-beta_2) - Phi_2(-beta_1, -beta_2; R_12)`.

- Domain: converged design points in one standard-normal space.
- Meaning: each mode is linearized at its own design point and the modes
  are combined as a series system; exact when `G` is the minimum of the
  linear functions.
- Code: `reliability.multi_point_form`.
- Source: [REF-DERKIUREGHIAN-DAKESSIAN-1998] and
  [REF-MELCHERS-BECK-2018, Chapter 5].

### EQ-REL-MDP-003 — Mixture importance sampling over design points

`h(u) = sum_k pi_k phi_n(u - u_k*)`,
`w(u) = phi_n(u) / h(u) = 1 / sum_k pi_k exp(u . u_k* - ||u_k*||^2 / 2)`,
`P_f_hat = N^-1 sum_i I(G(u_i) <= 0) w(u_i)`; `pi_k = 1/m` or
`pi_k proportional to Phi(-beta_k)`.

- Domain: any centres; every component has unit covariance, so `h > 0`.
- Meaning: unbiased importance sampling whose proposal covers all known
  failure modes; on `u . u_k* >= ||u_k*||^2` the weight is at most
  `exp(-||u_k*||^2 / 2) / pi_k`.
- Code: `reliability.mixture_importance_sampling`.
- Source: [REF-OWEN-2013, Chapter 9] and [REF-MELCHERS-BECK-2018, Chapter 6].

### EQ-REL-MVN-001 — Separation-of-variables multivariate normal probability

For `V ~ N(0, R)`, `R = C C'` (lower `C`):
`P(V <= b) = E[prod_i e_i(w)]`, `e_i = Phi((b_i - sum_(j<i) c_ij y_j) / c_ii)`,
`y_j = Phi^-1(w_j e_j)`, `w_j ~ U(0, 1)`; the union probability is
averaged as `1 - prod_i e_i = -expm1(sum_i log1p(-Phi(-t_i)))`.

- Domain: positive semidefinite `R`. A row `i` with a zero pivot satisfies
  `V_i = sum_(k<=j) c_ik Z_k` with last non-zero column `j`; its constraint
  is folded into step `j` as an upper (`c_ij > 0`) or lower (`c_ij < 0`)
  bound, so `e_j = Phi(t_hi) - Phi(t_lo)` and
  `y_j = Phi^-1(Phi(t_lo) + w_j e_j)`.
- Meaning: Genz's transformation of the rectangle probability into an
  integral over the unit cube, estimated by Monte Carlo with a standard
  error; averaging the complement keeps relative precision for small union
  probabilities.
- Code: `reliability.multi_point_form` (three or more points).
- Source: [REF-GENZ-BRETZ-2009, Chapter 4].

### EQ-REL-AKMCS-001 — AK-MCS learning function and stopping rule

`U(u) = |mu_G(u)| / sigma_G(u)` from the GP posterior of `G`; next point
`u* = argmin_(i not in T) U(u_i)` over the Monte Carlo population; stop when
`min_i U(u_i) >= 2`.

- Domain: GP surrogate of `G` in standard-normal space; fixed population.
- Meaning: `Phi(-U)` is the posterior probability that the predicted sign
  of `G(u_i)` is wrong, so `min U >= 2` bounds every point's misclassification
  probability by `Phi(-2) ~ 0.023`. Between hyperparameter fits the
  posterior is updated by exact rank-one Cholesky updates.
- Code: `reliability.ak_mcs`.
- Source: [REF-ECHARD-GAYTON-LEMAIRE-2011] and
  [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

### EQ-REL-AKMCS-002 — AK-MCS population estimate and coefficient of variation

`P_f_hat = N_f / N` with `N_f = #{i : mu_G(u_i) <= 0}`;
`c.o.v. = sqrt((1 - P_f_hat) / (N P_f_hat))`; the population is enlarged
while the c.o.v. exceeds the target.

- Domain: `N_f > 0`; the c.o.v. is the Monte Carlo error of the population
  and excludes surrogate misclassification.
- Meaning: crude Monte Carlo on the population with the surrogate's sign in
  place of the true indicator.
- Code: `reliability.ak_mcs`.
- Source: [REF-ECHARD-GAYTON-LEMAIRE-2011] and [REF-OWEN-2013, Chapter 2].

## Proper scoring rules, forecast calibration, and distribution shift

### EQ-VAL-SCORE-CRPS-001 — Continuous ranked probability score

`CRPS(F, y) = int (F(x) - 1{x >= y})^2 dx = E|X - y| - (1/2) E|X - X'|`,
with `X, X'` independent draws from `F`.

- Domain: `F` with finite first moment; `y` real. Negatively oriented.
- Meaning: strictly proper score that rewards calibration and sharpness
  jointly; it reduces to the absolute error for a point forecast.
- Code: `validation.scoring.crps_ensemble`,
  `validation.scoring.crps_distribution`.
- Source: [REF-MATHESON-WINKLER-1976] and [REF-GNEITING-RAFTERY-2007].

### EQ-VAL-SCORE-CRPS-002 — Ensemble CRPS by sorting

For members `x_1..x_m` with order statistics `x_(k)`:
`CRPS(F_m, y) = (1/m) sum |x_i - y| - (1/m^2) sum_k (2k - m - 1) x_(k)`,
using `sum_i sum_j |x_i - x_j| = 2 sum_k (2k - m - 1) x_(k)`.

- Domain: at least one finite member; members shifted by `y` first.
- Meaning: exact CRPS of the empirical CDF in `O(m log m)`; equals the
  integral of EQ-VAL-SCORE-CRPS-001 for the step CDF `F_m`.
- Code: `validation.scoring.crps_ensemble`,
  `validation.ExperimentAssessment.crps`.
- Source: [REF-HERSBACH-2000] and [REF-GNEITING-RAFTERY-2007].

### EQ-VAL-SCORE-CRPS-003 — Fair ensemble CRPS

`fCRPS(x, y) = (1/m) sum |x_i - y| - (1/(2 m (m - 1))) sum_i sum_j |x_i - x_j|`.

- Domain: `m >= 2` members.
- Meaning: unbiased for the CRPS of the law the members are drawn from; its
  expectation does not depend on `m`.
- Code: `validation.scoring.crps_ensemble(fair=True)`.
- Source: [REF-FERRO-2014].

### EQ-VAL-SCORE-CRPS-004 — CRPS of a normal law

`CRPS(N(mu, sigma^2), y) = sigma [z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)]`,
`z = (y - mu) / sigma`.

- Domain: `sigma > 0`.
- Meaning: closed form of EQ-VAL-SCORE-CRPS-001 for a Gaussian forecast.
- Code: `validation.scoring.crps_normal`.
- Source: [REF-GNEITING-RAFTERY-WESTVELD-GOLDMAN-2005] and
  [REF-GNEITING-RAFTERY-2007].

### EQ-VAL-SCORE-CRPS-005 — CRPS of a log-normal law

For `y > 0`, `z = (ln y - mu)/sigma`:
`CRPS = y (2 Phi(z) - 1) - 2 exp(mu + sigma^2/2) [Phi(z - sigma) + Phi(sigma/sqrt 2) - 1]`;
for `y <= 0`, `CRPS = -y + 2 exp(mu + sigma^2/2) [1 - Phi(sigma/sqrt 2)]`.

- Domain: `sigma > 0`; any real `y`.
- Meaning: closed form for a log-normal forecast; the `y <= 0` branch is the
  limit `Phi(z) = Phi(z - sigma) = 0` of the first.
- Code: `validation.scoring.crps_lognormal`.
- Source: [REF-BARAN-LERCH-2015] and [REF-JORDAN-KRUGER-LERCH-2019]; checked
  against quadrature of EQ-VAL-SCORE-CRPS-001 in the tests.

### EQ-VAL-SCORE-CRPS-006 — CRPS of a logistic law

`CRPS(Logistic(mu, s), y) = s [z - 2 ln F(z) - 1]`, `z = (y - mu)/s`, with the
standard logistic CDF `F`.

- Domain: `s > 0`.
- Meaning: closed form for a logistic forecast.
- Code: `validation.scoring.crps_logistic`.
- Source: [REF-JORDAN-KRUGER-LERCH-2019]; checked against quadrature of
  EQ-VAL-SCORE-CRPS-001 in the tests.

### EQ-VAL-SCORE-CRPS-007 — CRPS of an exponential law

`CRPS(Exp(lambda), y) = y + 2 exp(-lambda y)/lambda - 3/(2 lambda)` for
`y >= 0` and `|y| + 1/(2 lambda)` for `y < 0`.

- Domain: `lambda > 0`.
- Meaning: from `E|X - y| - E|X - X'|/2` with `E|X - X'| = 1/lambda`.
- Code: `validation.scoring.crps_exponential`.
- Source: derived from EQ-VAL-SCORE-CRPS-001 ([REF-GNEITING-RAFTERY-2007]);
  same expression as [REF-JORDAN-KRUGER-LERCH-2019]; checked by quadrature.

### EQ-VAL-SCORE-CRPS-008 — CRPS of a uniform law

With `w = b - a`: `CRPS = ((y - a)^2 + (b - y)^2)/(2w) - w/6` for
`a <= y <= b` and `|y - (a + b)/2| - w/6` otherwise.

- Domain: `a < b`.
- Meaning: from `E|X - y| - E|X - X'|/2` with `E|X - X'| = w/3`.
- Code: `validation.scoring.crps_uniform`.
- Source: derived from EQ-VAL-SCORE-CRPS-001 ([REF-GNEITING-RAFTERY-2007]);
  checked by quadrature.

### EQ-VAL-SCORE-CRPS-009 — CRPS as an integral of quantile scores

`CRPS(F, y) = 2 int_0^1 QS_u(F^-1(u), y) du
= 2 [int_0^(F(y)) u (y - Q(u)) du + int_(F(y))^1 (1 - u)(Q(u) - y) du]`.

- Domain: continuous `F` with finite mean.
- Meaning: generic CRPS for any law with `cdf` and `quantile`; the split at
  `F(y)` leaves two smooth integrands for tanh-sinh quadrature.
- Code: `validation.scoring.crps_quadrature`.
- Source: [REF-GNEITING-RAFTERY-2007].

### EQ-VAL-SCORE-LOG-001 — Logarithmic score

`LogS(F, y) = -log f(y)` (density or mass function).

- Domain: `f(y) >= 0`; `+inf` when `f(y) = 0`.
- Meaning: strictly proper local score; in nats.
- Code: `validation.scoring.log_score`.
- Source: [REF-GNEITING-RAFTERY-2007].

### EQ-VAL-SCORE-DSS-001 — Dawid-Sebastiani score

`DSS(F, y) = (y - mu_F)^2 / sigma_F^2 + ln sigma_F^2`.

- Domain: `sigma_F^2 > 0`.
- Meaning: proper with respect to the first two moments; twice the Gaussian
  log score up to a constant.
- Code: `validation.scoring.dawid_sebastiani_score`.
- Source: [REF-DAWID-SEBASTIANI-1999] and [REF-GNEITING-RAFTERY-2007].

### EQ-VAL-SCORE-ES-001 — Energy score

`ES(F_m, y) = (1/m) sum ||x_i - y|| - (1/(2 m^2)) sum_i sum_j ||x_i - x_j||`
(Euclidean norm; fair version `2 m (m - 1)`).

- Domain: `m` vectors of the dimension of `y`.
- Meaning: multivariate generalization of EQ-VAL-SCORE-CRPS-002; equal to it
  in one dimension.
- Code: `validation.scoring.energy_score`.
- Source: [REF-GNEITING-RAFTERY-2007] and [REF-GNEITING-STANBERRY-ETAL-2008].

### EQ-VAL-SCORE-VS-001 — Variogram score of order p

`VS_p = sum_(i<j) w_ij (|y_i - y_j|^p - (1/m) sum_k |x_ki - x_kj|^p)^2`.

- Domain: dimension `d >= 2`, `p > 0`, `w_ij >= 0`.
- Meaning: compares pairwise component differences and so targets the
  dependence structure; proper but not strictly proper. Summing each pair
  once is half of the full double sum with symmetric weights.
- Code: `validation.scoring.variogram_score`.
- Source: [REF-SCHEUERER-HAMILL-2015].

### EQ-VAL-SCORE-QS-001 — Pinball (quantile) loss

`QS_tau(q, y) = (1{y < q} - tau)(q - y)`.

- Domain: `tau in [0, 1]`.
- Meaning: proper for the `tau`-quantile.
- Code: `validation.scoring.pinball_loss`.
- Source: [REF-GNEITING-RAFTERY-2007].

### EQ-VAL-SCORE-IS-001 — Interval score

`IS_alpha(l, u; y) = (u - l) + (2/alpha)(l - y) 1{y < l} + (2/alpha)(y - u) 1{y > u}`.

- Domain: `l <= u`, `alpha in (0, 1)`.
- Meaning: width plus miss penalty for a central `(1 - alpha)` interval;
  `(alpha/2) IS_alpha = QS_(alpha/2)(l, y) + QS_(1 - alpha/2)(u, y)`.
- Code: `validation.scoring.interval_score`.
- Source: [REF-GNEITING-RAFTERY-2007] and
  [REF-BRACHER-RAY-GNEITING-REICH-2021].

### EQ-VAL-SCORE-WIS-001 — Weighted interval score

`WIS = (1/(K + 1/2)) [ (1/2)|y - m| + sum_k (alpha_k/2) IS_(alpha_k)(l_k, u_k; y) ]`,
split into dispersion `sum (alpha_k/2)(u_k - l_k)` and over-/underprediction
parts.

- Domain: `K >= 0` central intervals and a median `m`.
- Meaning: twice the mean pinball loss over the `2K + 1` quantile levels; an
  approximation of the CRPS from a quantile forecast.
- Code: `validation.scoring.weighted_interval_score`.
- Source: [REF-BRACHER-RAY-GNEITING-REICH-2021].

### EQ-VAL-SCORE-BS-001 — Brier score

`BS = (1/N) sum (f_i - o_i)^2`, `o_i in {0, 1}`.

- Domain: `f_i in [0, 1]`.
- Meaning: strictly proper score for binary events.
- Code: `validation.scoring.brier_score`.
- Source: [REF-GNEITING-RAFTERY-2007] and [REF-MURPHY-1973].

### EQ-VAL-SCORE-BSDEC-001 — Murphy decomposition

For bins `k` with `n_k` forecasts, mean forecast `fbar_k`, event frequency
`obar_k` and climatology `obar`:
`REL = (1/N) sum n_k (fbar_k - obar_k)^2`, `RES = (1/N) sum n_k (obar_k - obar)^2`,
`UNC = obar (1 - obar)`, and `BS = REL - RES + UNC` when every bin holds a
single forecast value.

- Domain: binary outcomes; grouping by distinct forecast values.
- Meaning: miscalibration minus discrimination plus base-rate uncertainty.
- Code: `validation.scoring.brier_decomposition(bins=None)`.
- Source: [REF-MURPHY-1973].

### EQ-VAL-SCORE-BSDEC-002 — Within-bin terms of the Brier decomposition

`BS = REL - RES + UNC + WBV - WBC` with
`WBV = (1/N) sum_k sum_(i in k) (f_i - fbar_k)^2` and
`WBC = (2/N) sum_k sum_(i in k) (f_i - fbar_k)(o_i - obar_k)`.

- Domain: any binning of the forecasts.
- Meaning: restores the exact identity when forecasts inside a bin differ;
  follows from expanding `f_i - o_i = (fbar_k - obar_k) + (f_i - fbar_k) -
  (o_i - obar_k)` inside each bin.
- Code: `validation.scoring.brier_decomposition`,
  `validation.BrierDecomposition.identity_residual`.
- Source: [REF-STEPHENSON-COELHO-JOLLIFFE-2008].

### EQ-VAL-CAL-PIT-001 — Probability integral transform

`u_i = F_i(y_i)`.

- Domain: continuous forecast laws.
- Meaning: `u ~ Uniform(0, 1)` across cases for a probabilistically
  calibrated forecast.
- Code: `validation.calibration.pit_values`.
- Source: [REF-GNEITING-BALABDAOUI-RAFTERY-2007].

### EQ-VAL-CAL-PIT-002 — Randomized ensemble PIT

`u = (r + V (s + 1)) / (m + 1)`, `r = #{x_k < y}`, `s = #{x_k = y}`,
`V ~ Uniform(0, 1)`.

- Domain: ensemble of `m` members.
- Meaning: if `y` is exchangeable with the members, its rank among the
  `m + 1` values (ties broken at random) is uniform on `0..m`, so `u` is
  exactly uniform.
- Code: `validation.calibration.ensemble_pit`.
- Source: [REF-HAMILL-2001], [REF-GNEITING-BALABDAOUI-RAFTERY-2007], and
  [REF-CZADO-GNEITING-HELD-2009] (randomization for discrete forecasts).

### EQ-VAL-CAL-HIST-001 — PIT histogram uniformity test

Counts `c_1..c_K` of `u` in `K` equal bins; `X^2 = sum (c_k - N/K)^2 / (N/K)`
against `chi^2_(K-1)`; edge ratio `(c_1 + c_K)/(2N/K)`.

- Domain: independent cases; `N/K >= 5` for the chi-square approximation.
- Meaning: a U shape (edge ratio above one) indicates overconfident
  forecasts, a hump underconfident ones, a slope bias.
- Code: `validation.calibration.pit_histogram`, `validation.PitHistogram`.
- Source: [REF-GNEITING-BALABDAOUI-RAFTERY-2007] and [REF-HAMILL-2001].

### EQ-VAL-CAL-COV-001 — Central-interval coverage from PIT values

`y in [F^-1((1-beta)/2), F^-1((1+beta)/2)]` iff `(1-beta)/2 <= u <= (1+beta)/2`;
empirical coverage `c/N` with a Clopper-Pearson or Wilson interval.

- Domain: continuous forecasts; independent cases for the binomial interval.
- Meaning: reads off coverage at every nominal level from one PIT sample.
- Code: `validation.calibration.pit_coverage`,
  `validation.calibration.interval_coverage`.
- Source: [REF-GNEITING-BALABDAOUI-RAFTERY-2007],
  [REF-CLOPPER-PEARSON-1934], and [REF-WILSON-1927].

### EQ-VAL-CAL-REL-001 — Reliability diagram and consistency bands

Per bin: `fbar_k`, `obar_k`, `n_k`; band = central quantiles of
`(1/n_k) sum_(i in k) B_i`, `B_i ~ Bernoulli(f_i)`, over resamples.

- Domain: binary outcomes; bins `[e_k, e_(k+1))`, the last closed.
- Meaning: under reliability `obar_k` scatters inside the band; the band
  keeps the bin's forecasts fixed (a simplification of Bröcker-Smith
  consistency resampling, which also resamples the forecasts).
- Code: `validation.calibration.reliability_diagram`.
- Source: [REF-BROCKER-SMITH-2007] and [REF-MURPHY-1973].

### EQ-VAL-CAL-SHARP-001 — Sharpness summaries

Mean and median width `F^-1((1+beta)/2) - F^-1((1-beta)/2)` per level, and
mean predictive standard deviation.

- Domain: forecasts with quantiles (ensembles: type-7 sample quantiles).
- Meaning: concentration of the forecasts, to be maximized subject to
  calibration.
- Code: `validation.calibration.sharpness_summary`.
- Source: [REF-GNEITING-BALABDAOUI-RAFTERY-2007].

### EQ-VAL-SHIFT-ENERGY-001 — Energy distance and its test statistic

`E_(m,n) = (2/(mn)) sum_ij |x_i - y_j| - (1/m^2) sum_ij |x_i - x_j| - (1/n^2) sum_ij |y_i - y_j|`,
test statistic `T = (m n/(m + n)) E_(m,n)`; in 1-D `E = 2 int (F_m - G_n)^2 dx`
and within-sample sums use `sum_(i<j) |z_i - z_j| = sum_k (2k - K - 1) z_(k)`.

- Domain: samples of equal dimension.
- Meaning: zero iff the two (empirical) distributions coincide.
- Code: `validation.shift.energy_distance`,
  `validation.shift.energy_distance_test`.
- Source: [REF-SZEKELY-RIZZO-2013].

### EQ-VAL-SHIFT-MMD-001 — Unbiased squared MMD with a Gaussian kernel

`MMD_u^2 = (1/(m(m-1))) sum_(i!=j) k(x_i, x_j) + (1/(n(n-1))) sum_(i!=j) k(y_i, y_j) - (2/(mn)) sum_ij k(x_i, y_j)`,
`k(a, b) = exp(-||a - b||^2/(2 h^2))`, `h` = pooled median distance by default.

- Domain: `m, n >= 2`; `h > 0`.
- Meaning: unbiased estimate of the squared RKHS distance between the mean
  embeddings; zero in population iff the laws coincide (characteristic
  kernel).
- Code: `validation.shift.mmd_squared`, `validation.shift.mmd_test`.
- Source: [REF-GRETTON-ETAL-2012].

### EQ-VAL-SHIFT-PERM-001 — Two-sample permutation p-value

`p = (1 + #{b : T(pi_b) >= T_obs}) / (1 + B)`; pair sums per relabeling via
`S_XY = sum_(i in X) R_i - 2 S_XX`, `S_YY = T - S_XX - S_XY` (off-diagonal row
sums `R_i`, unordered total `T`).

- Domain: exchangeable pooled sample under the null.
- Meaning: valid p-value for any `B`; only the within-first block is summed
  per relabeling.
- Code: `validation.shift.energy_distance_test`, `validation.shift.mmd_test`.
- Source: [REF-PHIPSON-SMYTH-2010] and [REF-GOOD-2005].

### EQ-VAL-SHIFT-W-001 — Wasserstein distance between 1-D samples

`W_p^p = int_0^1 |F_m^-1(u) - G_n^-1(u)|^p du`, summed exactly over the merged
breakpoints `k/m`, `l/n`; `W_1 = int |F_m - G_n| dx`.

- Domain: `p >= 1`.
- Meaning: optimal-transport cost between the empirical laws.
- Code: `validation.shift.wasserstein_distance`.
- Source: [REF-PANARETOS-ZEMEL-2019].

### EQ-VAL-SHIFT-W-002 — Wasserstein distance between 1-D laws

`W_p^p = int_0^1 |Q_1(u) - Q_2(u)|^p du` by tanh-sinh quadrature.

- Domain: laws with `quantile`; finite `p`-th moments.
- Meaning: same quantity for distribution objects.
- Code: `validation.shift.distribution_wasserstein_distance`.
- Source: [REF-PANARETOS-ZEMEL-2019].

### EQ-VAL-SHIFT-W-003 — W2 between two univariate normals

`W_2^2(N(mu_1, s_1^2), N(mu_2, s_2^2)) = (mu_1 - mu_2)^2 + (s_1 - s_2)^2`.

- Domain: `s_1, s_2 > 0`.
- Meaning: quantile functions differ by `(mu_1 - mu_2) + (s_1 - s_2) z`.
- Code: `validation.shift.normal_wasserstein_distance`.
- Source: [REF-PANARETOS-ZEMEL-2019].

### EQ-VAL-QUAD-DE-001 — Tanh-sinh quadrature on (0, 1)

`int_a^b g(u) du = (b - a) int w(t) g(a + (b - a) x(t)) dt`,
`x(t) = 1/(1 + exp(-pi sinh t))`, `w(t) = pi cosh(t) x (1 - x)`, trapezoidal
rule with `h = 2^-k`, nodes kept while `min(x, 1 - x) >= 1e-300`. The error
estimate is `|T_k - T_{k-1}|` plus the truncation term `|T(t_e)| / r` on each
side, with `T(t_e)` the weighted integrand at the outermost node and `r` its
decay rate towards the edge. Pieces split at breakpoints `(u, 1 - u)` (with
exact complements near one) are converged when `sum E_i <= max(tol |sum I_i|,
abs_tol)`.

- Domain: integrands smooth inside, integrable endpoint singularities.
- Meaning: double-exponential decay of the transformed integrand gives fast
  convergence for quantile-scale integrals.
- Code: `numerics.double_exponential.tanh_sinh` (re-exported as
  `numerics.tanh_sinh`; used by `validation.scoring`, `validation.shift`,
  and `information.wasserstein.quantile_wasserstein`).
- Source: [REF-TAKAHASI-MORI-1974].

### EQ-INF-GOF-KS2-001 — Two-sample Kolmogorov-Smirnov statistic

`D = max_z |F_m(z) - G_n(z)|`, kept as the integer `K = m n D = max |i n - j m|`
over the ECDF step points.

- Domain: two finite samples; comparisons only at ends of tied blocks.
- Meaning: largest vertical gap between the two empirical CDFs.
- Code: `inference.kolmogorov_smirnov_two_sample_test`.
- Source: [REF-HODGES-1958] and [REF-HOLLANDER-WOLFE-CHICKEN-2014].

### EQ-INF-GOF-KS2-002 — Exact two-sample KS tail by lattice paths

`P(D >= d) = 1 - N_inside / C(m + n, m)` where `N_inside` counts monotone
lattice paths `(0,0) -> (m,n)` with `|i n - j m| < K` at every checked step.

- Domain: exchangeable pooled sample; exact conditionally on the tie pattern.
- Meaning: exact permutation p-value computed with integer counts.
- Code: `inference.kolmogorov_smirnov_two_sample_test`.
- Source: [REF-HODGES-1958].

### EQ-INF-GOF-KS2-003 — Asymptotic two-sample KS tail

`P(sqrt(m n/(m + n)) D > t) -> 2 sum_(k>=1) (-1)^(k-1) exp(-2 k^2 t^2)`.

- Domain: large `m`, `n`, continuous data; no small-sample correction.
- Meaning: limiting Kolmogorov law for the scaled statistic.
- Code: `inference.kolmogorov_smirnov_two_sample_test(exact=False)`.
- Source: [REF-HOLLANDER-WOLFE-CHICKEN-2014] and [REF-NIST-GOF].

### EQ-OT-COST-001 — Ground costs for discrete transport

`C_ij = |x_i - y_j|^p` with the Euclidean norm; `p = 2` gives the squared
Euclidean cost. For `p >= 1`, `W_p(alpha, beta) = (min_P <P, C>)^(1/p)`.

- Domain: point clouds of one common dimension; `p > 0` for the matrix,
  `p >= 1` for the distance.
- Meaning: price of moving unit mass from `x_i` to `y_j`.
- Code: `information.power_cost_matrix`,
  `information.squared_euclidean_cost_matrix`,
  `information.discrete_wasserstein`.
- Source: [REF-PEYRE-CUTURI-2019, Section 2.4] and
  [REF-VILLANI-2009, Chapter 6].

### EQ-OT-LP-001 — Kantorovich linear program, dual, and certificate

Primal `min sum_ij C_ij P_ij` over `P >= 0`, `P 1 = a`, `P' 1 = b`; dual
`max sum_i a_i u_i + sum_j b_j v_j` subject to `u_i + v_j <= C_ij`. A
primal-feasible `P` and dual-feasible `(u, v)` with
`P_ij (C_ij - u_i - v_j) = 0` for all `(i, j)` are both optimal, and then
`sum C P = sum a u + sum b v`.

- Domain: non-negative weights with equal total mass; finite costs.
- Meaning: complementary slackness is a checkable optimality certificate;
  the certificate reports the duality gap, marginal error, negativity, dual
  infeasibility, and the largest slackness product.
- Code: `information.exact_transport`, `information.transport_certificate`,
  `information.TransportPlan.certificate`.
- Source: [REF-PEYRE-CUTURI-2019, Sections 3.1-3.3].

### EQ-OT-SIMPLEX-001 — Transportation simplex

Start from the north-west corner plan (exactly `m + n - 1` basic cells
forming a spanning tree). Repeat: solve `u_i + v_j = C_ij` on the tree with
`u_0 = 0`; if `min_ij (C_ij - u_i - v_j) >= -tol` stop; otherwise add the
most negative cell, shift `theta = min` of the cells with sign `-` around
the unique tree cycle (signs alternate `+, -, +, ...` from the entering
cell), and remove a cell that reaches zero. After `m + n` consecutive
degenerate pivots the entering and leaving cells are chosen by smallest
index (Bland).

- Domain: balanced problems; dense pure-Python implementation for small
  `m n`.
- Meaning: the simplex method specialized to the transportation polytope;
  every iterate is a vertex, and the final potentials certify optimality.
- Code: `information.exact_transport`.
- Source: [REF-DANTZIG-1951], [REF-BLAND-1977], and
  [REF-PEYRE-CUTURI-2019, Sections 3.4-3.5].

### EQ-OT-SINK-001 — Log-domain Sinkhorn iterations

For unit-mass `a, b` and `eps > 0`,
`OT_eps = min_P <P, C> + eps KL(P || a b')`, with
`P_ij = a_i b_j exp((f_i + g_j - C_ij)/eps)` and alternating soft
c-transforms `f_i = -eps log sum_j b_j exp((g_j - C_ij)/eps)`,
`g_j = -eps log sum_i a_i exp((f_i - C_ij)/eps)` computed by log-sum-exp.
After a `g` update the columns of `P` are exact; the stopping error is
`sum_i a_i |exp((f_i - f_i^new)/eps) - 1| = |P 1 - a|_1`. The reported
regularized cost is the dual value
`<a, f> + <b, g> - eps (<a b', exp((f + g - C)/eps)> - 1)`.

- Domain: balanced non-negative weights (normalized internally), finite
  costs, `eps > 0`.
- Meaning: a strictly convex smoothing of the transport LP; `<P, C>`
  approaches the exact optimum as `eps -> 0`, but the number of iterations
  grows quickly when `eps` is small relative to the spread of `C`.
- Code: `information.sinkhorn`, `information.SinkhornResult`.
- Source: [REF-CUTURI-2013] and [REF-PEYRE-CUTURI-2019, Sections 4.2 and
  4.4].

### EQ-OT-SINK-002 — Sinkhorn divergence

`S_eps(alpha, beta) = OT_eps(alpha, beta) - OT_eps(alpha, alpha)/2
- OT_eps(beta, beta)/2`. The symmetric terms use the averaged map
`f <- (f + T(f))/2`, `T(f)_i = -eps log sum_j a_j exp((f_j - C_ij)/eps)`,
and `OT_eps(alpha, alpha) = 2 <a, f>` at the fixed point.

- Domain: two weighted point clouds with equal mass; cost `|x - y|^p`.
- Meaning: removes the entropic bias, so `S_eps(alpha, alpha) = 0`; it
  interpolates between the transport cost (`eps -> 0`) and an MMD
  (`eps -> infinity`).
- Code: `information.sinkhorn_divergence`.
- Source: [REF-FEYDY-ETAL-2019].

### EQ-OT-W1D-001 — One-dimensional Wasserstein distance by quantiles

`W_p(mu, nu)^p = int_0^1 |Q_mu(u) - Q_nu(u)|^p du`, evaluated as
`int |Q_mu(Phi(z)) - Q_nu(Phi(z))|^p phi(z) dz` by composite Simpson
quadrature on `[-z_max, z_max]`; upper-tail nodes use `isf(Phi(-z))`.

- Domain: laws on the real line with finite `p`-th moments and a quantile
  function; truncation omits probability `2 Phi(-z_max)` (about `1.2e-15`
  for `z_max = 8`).
- Meaning: on the line the optimal coupling is monotone, so `W_p` is the
  `L^p` distance between quantile functions.
- Code: `information.quantile_wasserstein`,
  `information._quantile_quadrature`.
- Source: [REF-PEYRE-CUTURI-2019, Remark 2.30] and
  [REF-DAVIS-RABINOWITZ-1984, Chapter 2].

### EQ-OT-GAUSS-001 — Wasserstein-2 distance between Gaussians (Bures)

`W_2(N(m0, S0), N(m1, S1))^2 = |m0 - m1|^2 + B(S0, S1)^2` with
`B(S0, S1)^2 = tr S0 + tr S1 - 2 tr (S0^(1/2) S1 S0^(1/2))^(1/2)`. In one
dimension `W_2 = sqrt((mu0 - mu1)^2 + (sigma0 - sigma1)^2)`. Matrix square
roots are `V diag(lambda^(1/2)) V'` from the Jacobi eigendecomposition, with
eigenvalues above `-1e-10 max(1, |lambda|)` clipped to zero.

- Domain: symmetric positive-semidefinite covariances of equal size.
- Meaning: the optimal map between Gaussians is affine; `B` is the Bures
  metric on covariance matrices.
- Code: `information.gaussian_wasserstein`, `information.bures_distance`,
  `information.normal_wasserstein`.
- Source: [REF-DOWSON-LANDAU-1982], [REF-OLKIN-PUKELSHEIM-1982], and
  [REF-PEYRE-CUTURI-2019, Remark 2.31].

### EQ-OT-BARY-001 — Wasserstein-2 barycenter of 1-D laws

`argmin_nu sum_k w_k W_2(nu, mu_k)^2` has quantile function
`Q(u) = sum_k w_k Q_k(u)` and mean `sum_k w_k E[X_k]`; two normal laws give
`N(sum w_k mu_k, (sum w_k sigma_k)^2)`.

- Domain: laws on the line with finite second moments; weights on the
  simplex.
- Meaning: the line is flat in `W_2` (quantile functions form a convex
  cone of `L^2`), so barycenters average quantiles.
- Code: `information.quantile_barycenter`, `information.QuantileBarycenter`.
- Source: [REF-PEYRE-CUTURI-2019, Section 9.2] and
  [REF-ALVAREZ-ESTEBAN-ETAL-2016].

### EQ-OT-BARY-002 — Fixed-point iteration for the Gaussian barycenter

The barycenter of `N(m_k, S_k)` with weights `w_k` is `N(sum w_k m_k, S)`
where `S = sum_k w_k (S^(1/2) S_k S^(1/2))^(1/2)`. Iterate
`S <- S^(-1/2) (sum_k w_k (S^(1/2) S_k S^(1/2))^(1/2))^2 S^(-1/2)` from
`S = sum w_k S_k` until the relative Frobenius step is below the tolerance.

- Domain: positive-definite covariances of one dimension.
- Meaning: for commuting covariances the fixed point is
  `(sum_k w_k S_k^(1/2))^2`.
- Code: `information.gaussian_barycenter`, `information.GaussianBarycenter`.
- Source: [REF-ALVAREZ-ESTEBAN-ETAL-2016].

### EQ-IG-FISHER-001 — Fisher information and its numerical evaluation

`I(theta) = E[s s'] = -E[d^2 log f(X; theta) / d theta d theta']` with
`s = d log f / d theta`. Numerically, derivatives are central differences
in `theta`; the expectation is Simpson quadrature over `x = Q(Phi(z))`
(continuous), a sum over the integer support truncated at upper-tail mass
`1e-14` (discrete), or a Monte Carlo mean.

- Domain: regular families whose support does not depend on `theta`.
- Meaning: the Fisher metric `ds^2 = d theta' I(theta) d theta`; the two
  forms agree under the usual regularity conditions.
- Code: `information.numerical_fisher_information`.
- Source: [REF-CASELLA-BERGER-2002, Section 7.3],
  [REF-MIYAMOTO-ETAL-2024, Section 1], and [REF-AMARI-2016].

### EQ-IG-FISHER-002 — Closed-form Fisher information matrices

`N(mu, sigma)`: `diag(1/sigma^2, 2/sigma^2)`. `Exponential(lambda)`:
`1/lambda^2`. `Gamma(k, lambda)` (rate): `[[psi'(k), -1/lambda],
[-1/lambda, k/lambda^2]]`. `Beta(a, b)`: `[[psi'(a) - psi'(a + b),
-psi'(a + b)], [-psi'(a + b), psi'(b) - psi'(a + b)]]`. `Bernoulli(p)`:
`1/(p(1 - p))`. `Binomial(n, p)`: `n/(p(1 - p))`. `Poisson(lambda)`:
`1/lambda`. Categorical in `(p_1, ..., p_{k-1})`: `delta_ij/p_i + 1/p_k`.
Mean of `N(m, S)`: `S^-1`.

- Domain: parameters in the open parameter space.
- Meaning: gamma and beta entries are minus the expected Hessian of
  `k log lambda - log Gamma(k) + (k - 1) log x - lambda x` and of
  `(a - 1) log x + (b - 1) log(1 - x) - log B(a, b)`; `psi'` is the
  trigamma function.
- Code: `information.normal_fisher_information`,
  `information.exponential_fisher_information`,
  `information.gamma_fisher_information`,
  `information.beta_fisher_information`,
  `information.bernoulli_fisher_information`,
  `information.binomial_fisher_information`,
  `information.poisson_fisher_information`,
  `information.categorical_fisher_information`,
  `information.multivariate_normal_mean_fisher_information`,
  `information.fisher_information`.
- Source: [REF-MIYAMOTO-ETAL-2024, Sections 3.1, 3.2, 3.5, 4.1 and 4.4.1],
  [REF-CASELLA-BERGER-2002, Section 7.3], and
  [REF-NIST-DLMF-GAMMA, Section 5.15].

### EQ-IG-FR-001 — Fisher-Rao distance between univariate normals

`d = sqrt 2 arccosh(1 + ((mu0 - mu1)^2/2 + (sigma0 - sigma1)^2) /
(2 sigma0 sigma1)) = 2 sqrt 2 asinh(sqrt(delta / (4 sigma0 sigma1)))` with
`delta = (mu0 - mu1)^2/2 + (sigma0 - sigma1)^2`.

- Domain: positive standard deviations.
- Meaning: `(d mu^2 + 2 d sigma^2)/sigma^2` is the Poincaré half-plane
  metric scaled by two in the coordinates `(mu/sqrt 2, sigma)`; the `asinh`
  form keeps precision for nearby laws.
- Code: `information.fisher_rao_normal`.
- Source: [REF-MIYAMOTO-ETAL-2024, Lemma 2.1 and Section 4.4.1] and
  [REF-ATKINSON-MITCHELL-1981].

### EQ-IG-FR-002 — Fisher-Rao distance on the simplex (sphere)

Categorical: `d(p, q) = 2 arccos(sum_i sqrt(p_i q_i)) =
4 asin(|sqrt p - sqrt q|_2 / 2)`. Binomial with `n` known:
`2 sqrt(n) |asin sqrt(p0) - asin sqrt(p1)|`; Bernoulli is `n = 1`.

- Domain: PMFs on one finite support (boundary allowed); probabilities in
  `[0, 1]`.
- Meaning: `p -> 2 sqrt(p)` is an isometry onto the positive orthant of the
  sphere of radius two; distances are great-circle arcs.
- Code: `information.fisher_rao_categorical`,
  `information.fisher_rao_bernoulli`, `information.fisher_rao_binomial`.
- Source: [REF-MIYAMOTO-ETAL-2024, Sections 3.1 and 3.5].

### EQ-IG-FR-003 — Fisher-Rao distances for one-parameter and flat models

Poisson: `2 |sqrt(lambda0) - sqrt(lambda1)|`. Exponential (rate):
`|log lambda0 - log lambda1|`. Normal mean with known covariance `S`:
`sqrt((m0 - m1)' S^-1 (m0 - m1))`.

- Domain: non-negative Poisson means, positive rates, positive-definite
  `S`.
- Meaning: one-parameter models give `d = |int sqrt(I(theta)) d theta|`;
  a constant metric `S^-1` is flat, so the distance is Mahalanobis.
- Code: `information.fisher_rao_poisson`,
  `information.fisher_rao_exponential`,
  `information.fisher_rao_normal_mean`.
- Source: [REF-MIYAMOTO-ETAL-2024, Sections 1, 3.2 and 4.1].

### EQ-DIV-TV-001 — Total variation distance for finite laws

`TV(p, q) = sum_i |p_i - q_i| / 2 = max_A |P(A) - Q(A)|`.

- Domain: two PMFs on one finite support.
- Meaning: the largest difference in probability that the two laws assign
  to an event; `TV` lies in `[0, 1]`.
- Code: `information.total_variation_distance`.
- Source: [REF-TSYBAKOV-2009, Chapter 2].

### EQ-DIV-HELL-001 — Hellinger and Bhattacharyya for finite laws

`BC = sum_i sqrt(p_i q_i)`, `D_B = -log BC`, and
`H^2 = sum_i (sqrt p_i - sqrt q_i)^2 / 2 = 1 - BC`.

- Domain: two PMFs on one finite support; `D_B = inf` for disjoint
  supports.
- Meaning: `H` lies in `[0, 1]`; texts that omit the factor `1/2` report
  `sqrt 2 H`. `D_B = D_{1/2}/2` in Rényi terms.
- Code: `information.bhattacharyya_coefficient`,
  `information.bhattacharyya_distance`, `information.hellinger_distance`.
- Source: [REF-KAILATH-1967] and [REF-TSYBAKOV-2009, Chapter 2].

### EQ-DIV-RENYI-001 — Rényi divergence for finite laws

`D_alpha(p || q) = log(sum_i p_i^alpha q_i^(1 - alpha)) / (alpha - 1)` for
`alpha` in `(0, 1) U (1, inf)`; `D_0 = -log Q(p > 0)`, `D_1 = KL(p || q)`,
`D_inf = log max_i p_i / q_i`. Conventions: a term with `q_i = 0 < p_i` is
infinite for `alpha > 1` and zero for `alpha < 1`.

- Domain: two PMFs on one finite support, `alpha` in `[0, inf]`.
- Meaning: non-decreasing in `alpha`, continuous at `alpha = 1` where it
  equals the KL divergence.
- Code: `information.renyi_divergence`.
- Source: [REF-VAN-ERVEN-HARREMOES-2014].

### EQ-DIV-ALPHA-001 — Alpha-divergence

`D_A^(alpha)(p || q) = (sum_i p_i^alpha q_i^(1 - alpha) - 1) /
(alpha (alpha - 1))` for normalized `p, q` and `alpha` not in `{0, 1}`;
`D_A^(1) = KL(p || q)`, `D_A^(0) = KL(q || p)`, and `D_A^(1/2) = 4 H^2`.
Rényi: `D_alpha = log(1 + alpha (alpha - 1) D_A^(alpha)) / (alpha - 1)`.

- Domain: two PMFs on one finite support or two Gaussian laws; any real
  `alpha`.
- Meaning: a one-parameter family of f-divergences through both KL
  directions.
- Code: `information.alpha_divergence`,
  `information.alpha_divergence_gaussian`.
- Source: [REF-CICHOCKI-CRUCES-AMARI-2011, Equation 25] (the general
  positive-measure form reduces to this for normalized laws).

### EQ-DIV-GAUSS-001 — Gaussian affinity, Rényi, Bhattacharyya, Hellinger

With `S_alpha = alpha S1 + (1 - alpha) S0` and `d = m0 - m1`,
`log int N0^alpha N1^(1 - alpha) = -alpha (1 - alpha) d' S_alpha^-1 d / 2
- (log det S_alpha - (1 - alpha) log det S0 - alpha log det S1) / 2`,
finite iff `S_alpha` is positive definite. Then
`D_alpha = log A_alpha / (alpha - 1)`,
`D_B = d' S^-1 d / 8 + (log det S - (log det S0 + log det S1)/2) / 2` with
`S = (S0 + S1)/2`, and `H = sqrt(1 - exp(-D_B))`.

- Domain: positive-definite covariances of one dimension; for `alpha > 1`
  the divergence is infinite when `S_alpha` is not positive definite.
- Meaning: completing the square in the exponent of `N0^alpha N1^(1-alpha)`.
- Code: `information.renyi_divergence_gaussian`,
  `information.bhattacharyya_distance_gaussian`,
  `information.hellinger_distance_gaussian`,
  `information.alpha_divergence_gaussian`.
- Source: [REF-GIL-ALAJAJI-LINDER-2013] and [REF-KAILATH-1967]; checked
  against numerical integration in `tests/test_transport_and_geometry.py`.

### EQ-DIV-GAUSS-002 — Total variation between Gaussians

Univariate: with `r1 < r2` the roots of `log p0(x) = log p1(x)` (a
quadratic when `sigma0 != sigma1`),
`TV = |(F0(r2) - F0(r1)) - (F1(r2) - F1(r1))|`; for `sigma0 = sigma1 =
sigma`, `TV = erf(|mu0 - mu1| / (2 sqrt 2 sigma))`. Shared covariance `S`:
`TV = erf(Delta / (2 sqrt 2))` with Mahalanobis `Delta`.

- Domain: positive standard deviations; positive-definite shared `S`.
- Meaning: Scheffé's identity `TV = int_{p0 > p1} (p0 - p1)` evaluated on
  the region where one density dominates.
- Code: `information.total_variation_normal`,
  `information.total_variation_gaussian_shared_covariance`.
- Source: [REF-TSYBAKOV-2009, Chapter 2].

### EQ-VIZ-MIXQ-001 — Central interval of a weighted Gaussian mixture

`F(q) = sum_s w_s Phi((q - m_s) / sd_s)` with `w_s >= 0`, `sum_s w_s = 1` (a
component with `sd_s = 0` is a point mass at `m_s`); the central
`(1 - alpha)` interval is `(F^-1(alpha/2), F^-1(1 - alpha/2))`, found by
bisection because `F` is continuous and non-decreasing.

- Domain: finite means, non-negative variances and weights, `0 < alpha < 1`.
- Meaning: the exact marginal interval of a quantity that is Gaussian given a
  parameter and mixed over posterior draws (equal weights) or quadrature nodes
  (quadrature weights), such as `delta(x*)` in Kennedy-O'Hagan prediction or
  `theta_j` of the hierarchical normal model.
- Code: `visualization.calibration_plots.gaussian_mixture_interval`,
  `visualization.hierarchical_plots.theta_intervals`.
- Source: [REF-GELMAN-ETAL-2013, Section 6.3] and [REF-NIST-DLMF].

### EQ-VIZ-PITBAND-001 — Pointwise binomial band of a PIT histogram

If `n` PIT values are independent and uniform on `[0, 1]`, the count in each of
`B` equal-width bins is `Binomial(n, 1/B)`; the pointwise band is
`(k_lo, k_hi)` with `k_lo` the smallest `k` such that
`P(K <= k) >= alpha/2` and `k_hi` the smallest `k` such that
`P(K <= k) >= 1 - alpha/2`, around the expected count `n/B`.

- Domain: `n >= 1`, `B >= 2`, `0 < alpha < 1`. Limits are pointwise, not
  simultaneous across bins; bin counts are dependent (they sum to `n`).
- Meaning: a bin outside its band is unusual for a calibrated predictive
  distribution; posterior (not leave-one-out) PIT values are pulled toward
  `1/2`, so a central hump is expected for in-sample checks.
- Code: `visualization.hierarchical_plots.binomial_band`,
  `visualization.hierarchical_plots.plot_pit_histogram`.
- Source: [REF-CASELLA-BERGER-2002, Section 3.2] and
  [REF-GNEITING-BALABDAOUI-RAFTERY-2007].

## Contingency tables

### EQ-INF-CT-001 — Two-by-two effect measures and their intervals

With `p1 = a/(a+b)`, `p2 = c/(c+d)`: risk difference `RD = p1 - p2` with the
Newcombe hybrid score interval `[RD - sqrt((p1 - l1)^2 + (u2 - p2)^2),
RD + sqrt((u1 - p1)^2 + (p2 - l2)^2)]` from Wilson limits `(l_i, u_i)`; risk
ratio `RR = p1/p2` with `log RR +/- z sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))`
(Katz); odds ratio `OR = ad/(bc)` with `log OR +/- z sqrt(1/a + 1/b + 1/c +
1/d)` (Woolf), adding 0.5 to every cell when one is zero.

- Domain: independent binomial rows with positive totals; the log intervals
  need `a, c > 0` (RR) or the corrected table (OR).
- Meaning: absolute, relative, and odds-scale contrasts of two risks; the
  Newcombe interval keeps nominal coverage near 0 and 1 where the Wald
  interval fails.
- Code: `inference.contingency.two_by_two_measures`.
- Source: [REF-NEWCOMBE-1998], [REF-KATZ-ETAL-1978], [REF-WOOLF-1955],
  [REF-AGRESTI-2013, Chapter 3].

### EQ-INF-CT-002 — McNemar test for paired proportions

With discordant counts `b` and `c`, `b ~ Binomial(b + c, 1/2)` under marginal
homogeneity; the asymptotic statistic is `(|b - c| - k)^2 / (b + c)` with
`k = 1` (continuity correction) or `0`, compared with chi-square on one
degree of freedom.

- Domain: paired binary outcomes; concordant pairs carry no information.
- Meaning: tests whether the probability of a positive outcome changed
  between the two measurements; `b/c` is the conditional odds ratio.
- Code: `inference.contingency.mcnemar_test`.
- Source: [REF-MCNEMAR-1947], [REF-AGRESTI-2013, Chapter 11].

### EQ-INF-CT-003 — Pearson and adjusted residuals under independence

`E_ij = R_i C_j / n`, Pearson residual `(O_ij - E_ij)/sqrt(E_ij)`, adjusted
residual `(O_ij - E_ij)/sqrt(E_ij (1 - R_i/n)(1 - C_j/n))`.

- Domain: an `r x c` table with no empty row or column.
- Meaning: adjusted residuals are approximately standard normal under
  independence, so cells with `|residual| > 2` show where the association
  lives after a significant chi-square test.
- Code: `inference.contingency.contingency_residuals`.
- Source: [REF-HABERMAN-1973], [REF-AGRESTI-2013, Chapter 2].

## Missing data

### EQ-MISS-001 — Observed-data multivariate normal log-likelihood

`l(mu, Sigma) = sum_i [-(p_i/2) log(2 pi) - (1/2) log|Sigma_(oo,i)|
- (1/2) (x_(o,i) - mu_(o,i))' Sigma_(oo,i)^-1 (x_(o,i) - mu_(o,i))]`, where
`o_i` is the set of `p_i` observed variables of row `i`; rows with `p_i = 0`
contribute zero.

- Domain: `Sigma` symmetric positive definite; missing entries coded as `None`
  or `nan`; evaluated per pattern with one Cholesky factor of `Sigma_oo`.
- Meaning: the marginal likelihood of what was observed. Under MAR with
  parameters distinct from those of the missingness mechanism, likelihood
  and Bayesian inference may ignore the mechanism and use this function.
- Code: `inference.incomplete_normal_log_likelihood`,
  `inference.fit_incomplete_multivariate_normal`.
- Source: [REF-LITTLE-RUBIN-2019], [REF-SCHAFER-1997].

### EQ-MISS-002 — EM steps for the incomplete multivariate normal

E-step for row `i` with observed block `o` and missing block `m`:
`xhat_m = mu_m + B (x_o - mu_o)`, `B = Sigma_mo Sigma_oo^-1`,
`C_i = Sigma_mm - Sigma_mo Sigma_oo^-1 Sigma_om` (Schur complement; zero for
observed coordinates). M-step: `mu = (1/n) sum_i xhat_i`,
`Sigma = (1/n) sum_i [(xhat_i - mu)(xhat_i - mu)' + C_i]`.

- Domain: every variable observed at least twice with positive spread; the
  completed `Sigma` must stay positive definite.
- Meaning: the complete-data MLE applied to conditionally expected
  sufficient statistics; monotone in [EQ-MISS-001] by [EQ-EM-001]. With no
  missing values the first M-step equals the sample mean and the
  divisor-`n` covariance computed on centred values.
- Code: `inference.fit_incomplete_multivariate_normal`.
- Source: [REF-DEMPSTER-LAIRD-RUBIN-1977], [REF-LITTLE-RUBIN-2019],
  [REF-SCHAFER-1997].

### EQ-MISS-003 — Observed-information standard errors of the mean

`I(theta) = -d^2 l / d theta d theta'` at the EM estimate, with
`theta = (mu, vech Sigma)`; `Cov(mu_hat) ~ [I^-1]_(mu, mu)`. The library
evaluates the Hessian by central differences in standardized parameters
`mu_j = mu_hat_j + s_j t_j`, `Sigma_jk = Sigma_hat_jk + s_j s_k u_jk`,
`s_j = sqrt(Sigma_hat_jj)`.

- Domain: an interior maximum where `I` is positive definite; otherwise no
  standard errors are reported.
- Meaning: inverting the full observed information keeps the
  `mu`-`Sigma` cross information, which is non-zero under MAR (the expected
  information is block diagonal and can give inconsistent standard errors
  when data are MAR but not MCAR). With complete data it reduces to
  `Sigma_hat / n`.
- Code: `inference.fit_incomplete_multivariate_normal`.
- Source: [REF-LITTLE-RUBIN-2019], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

### EQ-MISS-004 — Little's MCAR test

`d^2 = sum_j n_j (ybar_j - mu_hat_(o_j))' Sigma_hat_(o_j o_j)^-1
(ybar_j - mu_hat_(o_j))`, summed over missingness patterns `j` with `n_j`
rows, observed set `o_j` of size `p_j >= 1` and observed-variable mean
`ybar_j`, using the EM estimates `(mu_hat, Sigma_hat)`. Under MCAR,
`d^2` is asymptotically chi-square on `sum_j p_j - p` degrees of freedom.

- Domain: at least two patterns with observed values. The library uses the
  maximum-likelihood `Sigma_hat` (divisor `n`).
- Meaning: one global test of whether pattern-wise means differ from the
  pooled estimate; a small p-value is evidence against MCAR. It cannot
  distinguish MAR from MNAR and is sensitive to non-normality.
- Code: `inference.little_mcar_test`.
- Source: [REF-LITTLE-1988], [REF-LITTLE-RUBIN-2019].

### EQ-MISS-005 — Inverse-Wishart draws by the Bartlett decomposition

`A` lower triangular with `A_ii^2 ~ chi^2(nu - i + 1)` (`i = 1..p`) and
`A_ij ~ N(0, 1)` for `i > j`, all independent, so `A A' ~ W_p(nu, I)`. With
`Psi = C C'` (Cholesky), `Sigma = C A^-T A^-1 C'` satisfies
`Sigma^-1 ~ W_p(nu, Psi^-1)`, i.e. `Sigma ~ IW_p(Psi, nu)`, and
`E[Sigma] = Psi / (nu - p - 1)` for `nu > p + 1`.

- Domain: `Psi` symmetric positive definite; real `nu > p - 1`.
- Meaning: an inverse-Wishart sampler that needs only chi-square and normal
  variates and triangular solves.
- Code: `inference.sample_inverse_wishart`,
  `inference.impute_multivariate_normal`.
- Source: [REF-SMITH-HOCKING-1972], [REF-MARDIA-KENT-BIBBY-1979, Chapter 3],
  [REF-GELMAN-ETAL-2013, Section 3.6].

### EQ-MISS-006 — Data augmentation for the multivariate normal

I-step: `x_m ~ N(mu_m + B (x_o - mu_o), Sigma_mm.o)` for every row, with `B`
and `Sigma_mm.o` as in [EQ-MISS-002]. P-step under the normal-inverse-Wishart
prior `(mu_0, kappa_0, Psi_0, nu_0)`: with completed mean `ybar` and centred
cross-products `A`, `kappa_n = kappa_0 + n`, `nu_n = nu_0 + n`,
`mu_n = (kappa_0 mu_0 + n ybar)/kappa_n`,
`Psi_n = Psi_0 + A + (kappa_0 n / kappa_n)(ybar - mu_0)(ybar - mu_0)'`;
draw `Sigma ~ IW(Psi_n, nu_n)`, then `mu ~ N(mu_n, Sigma / kappa_n)`.

- Domain: `nu_n > p - 1`, `Psi_n` positive definite. The default
  `kappa_0 = 0`, `Psi_0 = 0`, `nu_0 = -1` is the improper prior
  `p(mu, Sigma) ∝ |Sigma|^-(p+1)/2`, giving `Sigma | Y ~ IW(A, n - 1)` and
  `mu | Sigma, Y ~ N(ybar, Sigma / n)`.
- Meaning: a Gibbs sampler on (missing values, parameters); after burn-in the
  I-step draws come from the posterior predictive distribution of the
  missing values, which makes the imputations proper. Each imputation runs
  its own chain from the EM estimate on a child random stream.
- Code: `inference.impute_multivariate_normal`.
- Source: [REF-TANNER-WONG-1987], [REF-SCHAFER-1997],
  [REF-GELMAN-ETAL-2013, Section 3.6].

### EQ-MISS-007 — Rubin's combining rules

`Qbar = (1/m) sum_k Q_k`, `Ubar = (1/m) sum_k U_k`,
`B = (1/(m-1)) sum_k (Q_k - Qbar)^2`, `T = Ubar + (1 + 1/m) B`,
`r = (1 + 1/m) B / Ubar`, `lambda = (1 + 1/m) B / T`,
`nu_old = (m - 1)/lambda^2 = (m - 1)(1 + 1/r)^2`, and
`gamma = ((nu + 1) lambda + 2)/(nu + 3)` with `nu` the degrees of freedom
used; the interval is `Qbar +/- t_nu(1 - alpha/2) sqrt(T)`.

- Domain: `m >= 2`, `U_k >= 0`, `Ubar > 0`; scalar estimands whose
  completed-data estimates are approximately normal. With `B = 0`,
  `nu_old = inf` and a normal quantile is used.
- Meaning: total variance adds the between-imputation spread to the average
  completed-data variance; `r` is the relative increase in variance due to
  missingness, `lambda` the share of total variance it causes, and `gamma`
  the estimated fraction of missing information. With `nu = nu_old`,
  `gamma = (r + 2/(nu_old + 3))/(1 + r)`.
- Code: `inference.pool_estimates`.
- Source: [REF-RUBIN-1987], [REF-SCHAFER-1997].

### EQ-MISS-008 — Barnard-Rubin small-sample degrees of freedom

`nu_obs = ((nu_com + 1)/(nu_com + 3)) nu_com (1 - lambda)` and
`nu_BR = (1/nu_old + 1/nu_obs)^-1`, with `nu_com` the complete-data degrees
of freedom.

- Domain: `nu_com > 0`; when `B = 0`, `nu_BR = nu_obs`.
- Meaning: `nu_BR <= nu_com` always, unlike `nu_old`, which can exceed the
  complete-data value when the fraction of missing information is small.
- Code: `inference.pool_estimates`.
- Source: [REF-BARNARD-RUBIN-1999].

### EQ-MISS-009 — Biased baselines: listwise deletion and mean imputation

Complete-case analysis keeps rows with `R_i = 1` (all observed); unconditional
mean imputation sets `x_ij = ybar_j^obs` for missing entries, so the completed
variance is `((n_j^obs - 1) s_j^2)/(n - 1) < s_j^2` and covariances shrink
toward zero.

- Domain: any incomplete rectangular data.
- Meaning: complete cases are a random subsample only under MCAR; under MAR
  their means are biased. Mean imputation inherits the observed-mean bias,
  understates variances by about the observed fraction, attenuates
  correlations, and treats imputed values as data, so its standard errors
  are too small.
- Code: `inference.complete_cases`, `inference.mean_impute`.
- Source: [REF-LITTLE-RUBIN-2019].

### EQ-NUM-GK15-001 — Gauss-Kronrod G7-K15 panel and QUADPACK local error

On `[a, b]` with centre `c` and half-width `h`,
`K15 = h sum_{i=1}^{15} w_i f(c + h x_i)` and `G7 = h sum_{j=1}^{7} v_j f(c + h y_j)`,
the seven `y_j` being Gauss-Legendre nodes contained in the fifteen Kronrod
nodes `x_i`. With `I_abs = h sum w_i |f_i|` and
`I_asc = h sum w_i |f_i - K15/(2h)|`, the local error estimate is
`E = max(50 eps I_abs, I_asc min(1, (200 |K15 - G7| / I_asc)^1.5))`.

- Domain: finite panel; `f` finite at the 15 interior nodes (never at `a`, `b`).
- Meaning: K15 is exact for polynomials of degree at most 22 and G7 for degree
  at most 13; the scaled difference is QUADPACK's empirical, deliberately
  pessimistic error estimate.
- Code: `numerics.gauss_kronrod.kronrod_panel`, `numerics.gauss_kronrod`.
- Source: [REF-PIESSENS-QUADPACK-1983], [REF-KRONROD-1965], [REF-LAURIE-1997].

### EQ-NUM-QAG-001 — Globally adaptive bisection and the QAGI substitution

Keep a partition `{P_k}` with estimates `(I_k, E_k)` from [EQ-NUM-GK15-001];
bisect `argmax_k E_k` until `sum E_k <= max(abs_tol, rel_tol |sum I_k|)` or
the panel count reaches `limit`. Infinite ranges use
`int_a^inf f(x) dx = int_0^1 f(a + (1 - t)/t) t^-2 dt` (and the folded
`f(x) + f(-x)` on the whole line).

- Domain: `abs_tol >= 0`, `rel_tol >= 50 eps` when `abs_tol = 0`, `limit >= 1`.
- Meaning: refinement concentrates where the local error is largest; reaching
  `limit` or an unbisectable panel is reported as non-convergence.
- Code: `numerics.gauss_kronrod`, `numerics.integrate`.
- Source: [REF-PIESSENS-QUADPACK-1983].

### EQ-NUM-DE-001 — Double-exponential rules on general ranges

`int f(x) dx = int f(phi(t)) phi'(t) dt ~ h sum_k f(phi(kh)) phi'(kh)` with
tanh-sinh `phi(t) = a + (b - a)/(1 + exp(-pi sinh t))` on `[a, b]`, exp-sinh
`phi(t) = a + exp((pi/2) sinh t)` on `[a, inf)`, and sinh-sinh
`phi(t) = sinh((pi/2) sinh t)` on the line; `h = 2^-k` is halved, reusing
nodes, until `E = max(|T_k - T_{k-1}|, 50 eps h sum |f phi'|) +
h sum' |f phi'| min(1, 2 alpha ulp(x)/d) + sum_sides |f phi'|(t_e) / r <=
max(abs_tol, rel_tol |T_k|)`, `d` being the distance to the nearest finite
end, `sum'` running over nodes with `ulp(x)/d >= 1e-4`, `alpha` the local
log-log slope of `|f|` against `d`, and `t_e` the outermost node on each side
with decay rate `r` of `|f phi'|` there.

- Domain: `f` analytic in a strip around the open range, integrable end
  singularities at a zero end, decay faster than `1/|x|` on infinite ranges.
- Meaning: the transformed integrand decays double-exponentially, so the
  trapezoidal error falls like `exp(-c N / log N)`; the rounding term guards
  against singularities at a non-zero end that floating point cannot resolve
  (it vanishes for integrands that are flat at the ulp scale, `alpha ~ 0`),
  and the truncation term against mass beyond the outermost nodes (slow
  algebraic decay such as `x^-1.025` at infinity or `x^-0.97` at zero).
- Code: `numerics.tanh_sinh_interval`, `numerics.exp_sinh`,
  `numerics.sinh_sinh`, `numerics.integrate`.
- Source: [REF-TAKAHASI-MORI-1974], [REF-MORI-SUGIHARA-2001].

## Limit theorems and estimator properties

### EQ-LIM-001 — Berry-Esseen bound with Shevtsova's constant

`sup_z |P(Z_n <= z) - Phi(z)| <= C rho / (sigma^3 sqrt(n))`, with
`Z_n = sqrt(n) (Xbar_n - mu)/sigma`, `rho = E|X - mu|^3` and
`C = 0.4748`.

- Domain: i.i.d. summands with `sigma > 0` and `rho < inf`; `n >= 1`.
- Meaning: a uniform, non-asymptotic bound on the CLT error. The value is
  not capped at one, so for small `n` or very skewed summands it can be
  vacuous.
- Code: `probability.berry_esseen_bound`, `probability.clt_error_profile`.
- Source: [REF-SHEVTSOVA-2011].

### EQ-LIM-002 — Exact laws of the standardized mean

Bernoulli: `P(Z_n <= z) = P(S <= floor(n p + z sqrt(n p q)))`,
`S ~ Binomial(n, p)`. Exponential(`lambda`): `P(Z_n <= z) = P(n, n + z
sqrt(n))` (regularized lower incomplete gamma, rate-free). Normal:
`P(Z_n <= z) = Phi(z)`. Tails of `Xbar_n - mu` follow from the same laws:
`Q(n, n (1 + lambda t))` and `P(n, n (1 - lambda t))` for the exponential.

- Domain: `0 < p < 1`, `lambda > 0`, `sigma > 0`, `n >= 1`.
- Meaning: closed forms that make CLT errors and tail probabilities exact
  rather than simulated.
- Code: `probability.standardized_mean_cdf`,
  `probability.sample_mean_upper_tail`,
  `probability.sample_mean_two_sided_tail`,
  `probability.standard_normal_cdf`. Binomial masses use Loader's
  saddle-point form [REF-LOADER-2000], accurate to rounding for any `n`.
- Source: [REF-CASELLA-BERGER-2002], [REF-NIST-DLMF-GAMMA],
  [REF-LOADER-2000].

### EQ-LIM-003 — Kolmogorov distance to the normal law

`K_n = sup_z |F_n(z) - A_n(z)|`, with `A_n = Phi` or the Edgeworth `G_n`.
For a lattice `F_n` and monotone `Phi`, the supremum is attained at an
atom `z_k`, so `K_n = max_k max(|F_n(z_k^-) - Phi(z_k)|, |F_n(z_k) -
Phi(z_k)|)`.

- Domain: any `n >= 1`. For continuous `F_n` the supremum is located on a
  1601-node grid over `[max(-sqrt(n) - 8, -40), 10]` (plus the support edge
  when inside) and refined by golden-section search; both laws are compared
  at the standardized value of the rounded sum `n + z sqrt(n)`.
- Meaning: the exact uniform CLT error. The lattice formula is exact; the
  continuous maximization is numerical, and it agrees with a 40001-point
  scan to `1e-6` in the tests.
- Code: `probability.kolmogorov_distance_to_normal`.
- Source: [REF-CASELLA-BERGER-2002], [REF-BRENT-1973].

### EQ-LIM-004 — One-term Edgeworth expansion

`G_n(z) = Phi(z) - phi(z) gamma (z^2 - 1) / (6 sqrt(n))`, with
`gamma = E(X - mu)^3 / sigma^3`.

- Domain: non-lattice summands with a finite third moment. Lattice
  summands need an additional discontinuous correction, which is not
  implemented, so the code raises.
- Meaning: `F_n - G_n = o(n^-1/2)` uniformly. It also gives
  `sqrt(n) K_n -> |gamma| phi(0)/6`, which is `1/(3 sqrt(2 pi))` for
  exponential summands.
- Code: `probability.edgeworth_cdf`,
  `probability.kolmogorov_distance_to_normal`.
- Source: [REF-HALL-1992].

### EQ-LIM-005 — Chebyshev bound for the sample mean

`P(|Xbar_n - mu| >= t) <= sigma^2 / (n t^2)`.

- Domain: finite variance, `t > 0`. The result is capped at one.
- Meaning: the weakest bound; it uses only the variance.
- Code: `probability.chebyshev_bound`, `probability.compare_tail_bounds`.
- Source: [REF-CASELLA-BERGER-2002].

### EQ-LIM-006 — Hoeffding bound

`P(Xbar_n - mu >= t) <= exp(-2 n t^2 / (b - a)^2)` for independent
`X_i in [a, b]`. The two-sided bound doubles this.

- Domain: bounded support, `t > 0`.
- Meaning: exponential concentration that uses only the range.
- Code: `probability.hoeffding_bound`, `probability.compare_tail_bounds`.
- Source: [REF-HOEFFDING-1963], [REF-WASSERMAN-2004].

### EQ-LIM-007 — Bernstein bound

`P(Xbar_n - mu >= t) <= exp(-n t^2 / (2 (sigma^2 + c t / 3)))` for
independent summands with `|X_i - mu| <= c`. This is the bounded case of
the sub-gamma inequality with `v = n sigma^2`, scale `c/3`, deviation
`n t`.

- Domain: `|X - mu| <= c` almost surely, `t > 0`.
- Meaning: tighter than Hoeffding when `sigma^2` is small relative to the
  range.
- Code: `probability.bernstein_bound`, `probability.compare_tail_bounds`.
- Source: [REF-BOUCHERON-LUGOSI-MASSART-2013].

### EQ-LIM-008 — Delta method and exact small-sample variances

Delta: `Var g(Xbar_n) ~ g'(mu)^2 sigma^2 / n`. Exact comparisons:

- normal `Xbar^2`: `4 mu^2 sigma^2/n + 2 sigma^4/n^2`;
- normal `exp(Xbar)`: `(e^{s^2} - 1) e^{2 mu + s^2}`, with
  `s^2 = sigma^2/n`;
- exponential `1/Xbar`: `(n lambda)^2 / ((n-1)^2 (n-2))`;
- Bernoulli `g(Xbar)`: a finite binomial sum.

- Domain: `g` differentiable at `mu`; `n >= 3` for `1/Xbar`.
- Meaning: the first-order approximation's relative error at small `n`.
  It fails completely when `g'(mu) = 0`.
- Code: `probability.delta_method_variance`,
  `probability.delta_method_normal_square`,
  `probability.delta_method_normal_exp`,
  `probability.delta_method_exponential_reciprocal`,
  `probability.delta_method_bernoulli`.
- Source: [REF-CASELLA-BERGER-2002].

### EQ-LIM-009 — Simulation estimates of convergence

There are three estimators:

- weak law: `p_hat_n = R^-1 sum_r 1{|Xbar_n^(r) - mu| > eps}`, with standard
  error `sqrt(p_hat (1 - p_hat)/R)`;
- strong law, finite horizon: `R^-1 sum_r 1{max_{m <= k <= N}
  |Xbar_k^(r) - mu| > eps}`;
- distribution: `D_R = sup_z |F_R(z) - Phi(z)|` for `R` simulated
  standardized means.

- Domain: `R` independent replications on child streams.
- Meaning: illustrations with Monte Carlo error. `D_R` adds noise of order
  `R^-1/2` (DKW) to the true Kolmogorov distance. The finite-horizon
  frequency lower-bounds the infinite-horizon probability.
- Code: `monte_carlo.simulate_weak_law`,
  `monte_carlo.simulate_running_means`,
  `monte_carlo.simulate_clt_distance`.
- Source: [REF-CASELLA-BERGER-2002], [REF-WASSERMAN-2004],
  [REF-DVORETZKY-KIEFER-WOLFOWITZ-1956], [REF-MASSART-1990].

### EQ-LIM-010 — Third central moments of the summands

- Bernoulli: `E(X - p)^3 = p q (1 - 2p)` and `E|X - p|^3 = p q (p^2 +
  q^2)`.
- Exponential(`lambda`): `E(X - mu)^3 = 2/lambda^3` and
  `E|X - mu|^3 = (12/e - 2)/lambda^3`.
- Normal: `0` and `2 sqrt(2/pi) sigma^3`.

- Domain: the parameter domains of the families.
- Meaning: the inputs of the Berry-Esseen bound (`rho`) and of the
  Edgeworth term (skewness).
- Code: `probability.bernoulli_summand`, `probability.exponential_summand`,
  `probability.normal_summand`.
- Source: [REF-CASELLA-BERGER-2002].

### EQ-EST-001 — Monte Carlo bias, variance, and MSE with standard errors

From estimates `T_1..T_R`:

- `bias = Tbar - theta`, with SE `s/sqrt(R)`;
- variance `s^2`, with SE `sqrt((m4 - s^4 (R-3)/(R-1))/R)`;
- `MSE = R^-1 sum (T_r - theta)^2`, with SE `sd((T - theta)^2)/sqrt(R)`.

`MSE = bias^2 + Var` for exact risks.

- Domain: `R >= 4` independent replications with finite estimates.
- Meaning: each simulated risk component comes with its own sampling error.
- Code: `inference.evaluate_estimator`, `inference.EstimatorRisk`.
- Source: [REF-CASELLA-BERGER-2002], [REF-ROBERT-CASELLA-2004].

### EQ-EST-002 — Paired relative efficiency

`RE = MSE(reference)/MSE(T)` on the same data sets. The standard error is
`sqrt((V_a - 2 RE C_ab + RE^2 V_b)/R) / MSE(T)`, where `a`, `b` are the
squared errors of the reference and of `T`.

- Domain: paired replications, `MSE(T) > 0`.
- Meaning: values above one favour `T`. The delta-method SE accounts for
  the positive correlation created by sharing data sets.
- Code: `inference.evaluate_estimator`.
- Source: [REF-CASELLA-BERGER-2002], [REF-ROBERT-CASELLA-2004].

### EQ-EST-003 — Interval coverage

`c = R^-1 sum 1{L_r <= theta <= U_r}`, with SE `sqrt(c (1 - c)/R)`.

- Domain: an interval procedure that returns finite ordered bounds.
- Meaning: the frequentist coverage of a procedure at one `theta`.
- Code: `inference.evaluate_estimator`.
- Source: [REF-CASELLA-BERGER-2002].

### EQ-EST-004 — Cramer-Rao lower bound

For an estimator with `E T = psi(theta)`, `Var T >= grad psi' I(theta)^-1
grad psi / n`, with `I` the per-observation Fisher information. If `T` is
unbiased for `tau(theta)`, `psi = tau`.

- Domain: a regular family (differentiation under the integral), `I`
  positive definite.
- Meaning: an unbiased estimator cannot beat the bound, but a biased one
  can beat the *unbiased* bound in MSE (normal-variance MLE). Efficiency is
  reported as `crlb / MSE`.
- Code: `inference.cramer_rao_bound`, `inference.mle_efficiency`.
- Source: [REF-CASELLA-BERGER-2002], [REF-LEHMANN-CASELLA-1998].

### EQ-EST-005 — Normal-variance divisor family

`SS = sum (X_i - Xbar)^2 ~ sigma^2 chi^2_{n-1}`, so
`MSE(SS/d) = sigma^4 (2 (n-1) + (n-1-d)^2)/d^2`. The minimum is at
`d = n + 1`, with value `2 sigma^4/(n + 1)`.

- Domain: i.i.d. normal data, `n >= 2`, `d > 0`.
- Meaning: the unbiased `d = n - 1` is not the minimum-MSE choice.
- Code: `inference.normal_variance_estimator_risk`,
  `inference.minimum_mse_variance_divisor`.
- Source: [REF-CASELLA-BERGER-2002].

### EQ-EST-006 — Uniform(0, theta) maximum

With `M = max X_i`: `E M = n theta/(n+1)` and
`Var M = n theta^2/((n+1)^2 (n+2))`.

- MLE `M`: `MSE = 2 theta^2/((n+1)(n+2))`.
- UMVUE `(n+1) M/n`: variance `theta^2/(n (n+2))`.
- `(n+2) M/(n+1)`: minimum MSE, `theta^2/(n+1)^2`.
- `2 Xbar`: variance `theta^2/(3n)`.

- Domain: `theta > 0`, `n >= 1`.
- Meaning: the sufficient-statistic estimators converge at rate `1/n`
  rather than `1/sqrt(n)`, and the unbiased choice is not the minimum-MSE
  choice.
- Code: `inference.uniform_maximum_risks`.
- Source: [REF-CASELLA-BERGER-2002], [REF-LEHMANN-CASELLA-1998].

### EQ-EST-007 — Rao-Blackwellization of the Poisson zero probability

`E[1{X_1 = 0} | S] = ((n - 1)/n)^S`, with `S ~ Poisson(n lambda)`. Using
`E a^S = exp(n lambda (a - 1))`:

- the crude estimator has variance `e^{-lambda}(1 - e^{-lambda})`;
- the Rao-Blackwellized estimator has variance
  `e^{-2 lambda}(e^{lambda/n} - 1)`;
- the plug-in `exp(-Xbar)` has mean `exp(n lambda (e^{-1/n} - 1))`.

The bound is `lambda e^{-2 lambda}/n`.

- Domain: `lambda > 0`, `n >= 1`.
- Meaning: conditioning an unbiased estimator on a sufficient statistic
  keeps it unbiased and cannot increase its variance.
- Code: `inference.poisson_zero_indicator`,
  `inference.poisson_zero_rao_blackwell`,
  `inference.poisson_zero_probability_risks`.
- Source: [REF-CASELLA-BERGER-2002], [REF-LEHMANN-CASELLA-1998].

### EQ-EST-008 — Factorization check for sufficiency

If `L(theta; x) = g(T(x), theta) h(x)`, then `T(a) = T(b)` implies that
`log L(theta; a) - log L(theta; b)` is constant in `theta`. For an
exponential family `h(x) exp(eta(theta)' T(x) - A(theta))`,
`sum T(x_i)` is sufficient.

- Domain: two samples with equal statistics and a grid of at least two
  parameter values.
- Meaning: a non-constant ratio refutes sufficiency. A constant ratio on a
  finite grid is necessary evidence, not a proof.
- Code: `inference.check_sufficiency`.
- Source: [REF-CASELLA-BERGER-2002], [REF-LEHMANN-CASELLA-1998].

### EQ-EST-009 — Consistency from vanishing MSE

`P(|T_n - theta| >= eps) <= MSE(T_n)/eps^2`, by Markov's inequality on
`(T_n - theta)^2`.

- Domain: finite second moments, `eps > 0`.
- Meaning: if bias and variance both tend to zero, `T_n` is consistent.
- Code: `inference.consistency_profile`.
- Source: [REF-CASELLA-BERGER-2002], [REF-WASSERMAN-2004].

### EQ-EST-010 — Exact moments of the exponential-rate MLE

`Xbar ~ Gamma(n, n lambda)`, so for `lambda_hat = 1/Xbar`:
`E lambda_hat = n lambda/(n-1)` and
`Var lambda_hat = n^2 lambda^2/((n-1)^2 (n-2))`. The biased-estimator
bound is `n lambda^2/(n-1)^2`, and the efficiency is
`crlb/MSE = (n-1)(n-2)/(n (n+2))`.

- Domain: `n >= 3`.
- Meaning: the MLE is biased and never attains the bound at finite `n`,
  but its efficiency tends to one.
- Code: `inference.mle_efficiency`,
  `probability.delta_method_exponential_reciprocal`.
- Source: [REF-CASELLA-BERGER-2002].

### EQ-VIZ-PITBAND-002 — Simultaneous (Bonferroni) band of a PIT histogram

With `B` equal-width bins of `n` PIT values, the simultaneous band is the
pointwise band of [EQ-VIZ-PITBAND-001] at level `1 - alpha/B` in every bin:
`k_lo` the smallest `k` with `P(K <= k) >= alpha/(2B)` and `k_hi` the smallest
`k` with `P(K <= k) >= 1 - alpha/(2B)`, `K ~ Binomial(n, 1/B)`.

- Domain: `n >= 1`, `B >= 2`, `0 < alpha < 1`.
- Meaning: each pointwise band leaves at most `alpha/B` probability outside,
  so by the union bound (Bonferroni, [EQ-INF-MT-001]) all `B` counts of
  independent uniform PIT values stay inside with probability at least
  `1 - alpha`, whatever the dependence between counts (they sum to `n`).
  The pointwise band is exceeded by about `B alpha` bins of a calibrated
  forecast; with `n = 400`, `B = 20`, `alpha = 0.05` some bin leaves it in
  about half of all calibrated samples, the simultaneous band in under 5%.
- Code: `visualization.hierarchical_plots.simultaneous_binomial_band`,
  `visualization.hierarchical_plots.plot_pit_histogram`.
- Source: [REF-LEHMANN-ROMANO-2005, Chapter 9],
  [REF-GNEITING-BALABDAOUI-RAFTERY-2007].

### EQ-PPL-TRANSFORM-001 — Log bijections for half-bounded supports

For a support `(a, inf)`: `x = a + exp(u)`, `log |dx/du| = u`,
`u = log(x - a)`. For `(-inf, b)`: `x = b - exp(u)`, `log |dx/du| = u`,
`u = log(b - x)`. For `R`: `x = u` with zero log-Jacobian.

- Domain: `u` in `R`; `exp(u)` overflows for `u > ~709` (mapped to a
  log-density of `-inf`).
- Meaning: a smooth bijection from `R` onto the interior of the support, so
  samplers and optimizers work on an unconstrained space; the log-Jacobian
  turns the density of `x` into a density of `u`.
- Code: `inference.ppl.Bijection`, `inference.ppl.bijection_for`.
- Source: [REF-CARPENTER-ETAL-2017], [REF-KUCUKELBIR-ETAL-2017].

### EQ-PPL-TRANSFORM-002 — Scaled logit bijection for an interval

For a support `(a, b)`: `x = a + (b - a) s(u)` with `s(u) = 1/(1 + exp(-u))`,
`log |dx/du| = log(b - a) + log s(u) + log(1 - s(u))
= log(b - a) - softplus(u) - softplus(-u)`, `softplus(t) = log(1 + exp(t))`,
and `u = log(x - a) - log(b - x)`.

- Domain: `u` in `R`, `a < b` finite.
- Meaning: maps `R` onto `(a, b)`; the softplus form avoids overflow for
  large `|u|`. Closed endpoints have probability zero under a continuous law
  and are not reached.
- Code: `inference.ppl.Bijection`, `inference.ppl.bijection_for`.
- Source: [REF-CARPENTER-ETAL-2017], [REF-KUCUKELBIR-ETAL-2017].

### EQ-PPL-JOINT-001 — Joint log-density on the unconstrained space

With latent elements `x_k = T_k(u_k; x_1..x_{k-1})` in execution order and
observed data `y`,
`log p(u) = sum_k log p(x_k | x_1..x_{k-1}) + sum_i log p(y_i | x)
+ sum_k log |dT_k/du_k|`.

- Domain: `u` in `R^d`; the bijection `T_k` may depend on earlier latent
  values (bounds that are parameters), which keeps the map lower triangular
  so its log-determinant is the sum of the diagonal terms.
- Meaning: the density that the samplers, ADVI, and the Laplace
  approximation target; draws of `u` are mapped back through `T`.
- Code: `inference.ppl.ModelIR.log_density`, `inference.ppl.ModelIR.terms`.
- Source: [REF-CARPENTER-ETAL-2017], [REF-BINGHAM-ETAL-2019].

### EQ-PPL-NONCENTERED-001 — Non-centred reparameterization of a normal site

`theta = loc + scale z`, `z ~ N(0, 1)`; the site contributes
`log N(z; 0, 1)` to the joint log-density and no Jacobian term, because `z`
is the sampled coordinate.

- Domain: `scale > 0`; `loc` and `scale` may depend on earlier latent sites.
- Meaning: the prior of the sampled coordinate no longer depends on `scale`,
  which removes the funnel between a group scale and the group effects when
  the data are weakly informative.
- Code: `inference.ppl.ModelHandle.sample` (`reparam="noncentered"`).
- Source: [REF-PAPASPILIOPOULOS-ROBERTS-SKOLD-2007],
  [REF-BETANCOURT-GIROLAMI-2013].

### EQ-PPL-PREDICTIVE-001 — Prior and posterior predictive simulation

Prior predictive: `x^(s) ~ p(x)` site by site, then `y^(s) ~ p(y | x^(s))`.
Posterior predictive: `y_rep^(s) ~ p(y | x^(s))` for posterior draws
`x^(s) ~ p(x | y)`, so `y_rep` has law `int p(y | x) p(x | y) dx`.

- Domain: any model whose observed sites can be sampled.
- Meaning: simulated data under the prior or the fitted model, the input of
  prior predictive checks and of [EQ-PPC-001].
- Code: `inference.ppl.prior_predictive`, `inference.ppl.posterior_predictive`,
  `inference.ppl.predictive_pvalues`.
- Source: [REF-GELMAN-ETAL-2013, Chapter 6], [REF-RUBIN-1984].

### EQ-SUR-GP-006 — Regularized GP hyperparameter estimation

With per-input training extent `E_i` and smallest positive coordinate gap
`h_i`, maximize `log p(y | theta) + log pi(theta)` over the box
`log ell_i in [log(a h_i), log(b E_i)]`, `log sigma_f^2 in [log s_lo, log s_hi]`
(`log sigma_n^2 in [-25, log s_hi]` when estimated), all clipped to
`|theta_j| <= 25`, with kernel matrix
`K = sigma_f^2 (R + tau I) + (nugget + sigma_n^2) I` and optional log-normal
prior `log pi = -(1/2) sum_i ((log ell_i - log E_i) / s)^2`. The box is
searched through `theta_j = l_j + (u_j - l_j) sin^2(phi_j / 2)`, so
`d theta_j / d phi_j = (1/2)(u_j - l_j) sin(phi_j)`; a parameter within
`0.01` of a bound is reported. The relative nugget adds
`(1/2) tau sigma_f^2 tr(alpha alpha' - K^-1)` to `d log p / d log sigma_f^2`.

- Domain: `a, b > 0`, `0 < s_lo < s_hi`, `tau >= 0`, `s > 0`; defaults
  `a = 0.1`, `b = 20`, `[s_lo, s_hi] = [1e-3, 1e3]` (standardized output
  units), `tau = 1e-8`, no prior. These values are computational choices.
- Meaning: below the smallest gap all correlations vanish and the likelihood
  is flat in `ell_i` (white-noise optimum); far above the extent `K` tends to
  a singular polynomial limit whose likelihood keeps rising while the
  posterior variance falls below rounding. `tau` bounds the condition number
  of `R + tau I` by `1 + n / tau`, i.e. caps the signal-to-noise ratio at
  `1 / tau`.
- Code: `surrogates.GaussianProcessRegularization`,
  `surrogates.gaussian_process_log_parameter_bounds`,
  `surrogates.fit_gaussian_process`,
  `surrogates.gaussian_process_log_marginal_likelihood`,
  `reliability.ak_mcs`.
- Source: [REF-RASMUSSEN-WILLIAMS-2006, Sections 5.2 and 5.4],
  [REF-ANDRIANAKIS-CHALLENOR-2012].

### EQ-REL-CE-004 — Coverage of the first-level elite set by a CE proposal

With `E_1` the elite points of the first cross-entropy level (drawn from the
starting density) and `mu` the final proposal mean,
`f = #{u in E_1 : u . mu / ||mu|| <= 0} / |E_1|`, computed only when
`||mu|| >= 1`; CE-IS warns when `f > 0.1`.

- Domain: `||mu|| >= 1` (`f = nan` otherwise); standard-normal space.
- Meaning: one convex or near-planar failure region `{u . alpha >= beta}`
  has all elite points at `u . alpha >= q > 0`, so `f` is near `0`; elite
  points behind the origin as seen from the proposal belong to a failure
  region the single Gaussian proposal does not sample, whose probability
  is then missed (four-branch system: `f` about `0.5`, estimate about a
  quarter to a third of the exact value).
- Code: `reliability.cross_entropy_importance_sampling`.
- Source: [REF-RUBINSTEIN-KROESE-2004], [REF-KURTZ-SONG-2013].

### EQ-STAT-STREAM-001 — Shifted Welford update of the mean and second central sum

With shift `K = x_1`, `y_i = x_i - K`, running mean `a_n` of the `y_i`, and
`M2_n = sum_{i<=n} (y_i - a_n)^2`: `delta = y_n - a_{n-1}`,
`a_n = a_{n-1} + delta / n`, `M2_n = M2_{n-1} + delta^2 (n - 1) / n`; the
reported mean is `K + a_n` and the variance `M2_n / (n - d)`.

- Domain: finite real observations, `n >= 1`; `0 <= d < n` for the variance.
- Meaning: one pass, constant memory. Central sums are shift invariant, and
  for data with a large common offset `x_i - K` is exact, so the recurrence
  runs on well-scaled numbers (2000 values `1e9 + Exp(1)`: relative variance
  error `3.5e-16`, unshifted Welford `2.0e-8`).
- Code: `statistics.RunningMoments`, `statistics.RunningCovariance`,
  `monte_carlo.estimate_mean_streaming`.
- Source: [REF-WELFORD-1962], [REF-CHAN-GOLUB-LEVEQUE-1983] (shifted data).

### EQ-STAT-STREAM-002 — Pairwise merge of means, second central sums, and co-moments

For disjoint chunks `A`, `B` with counts `n_A`, `n_B`, `n = n_A + n_B`, and
`delta = mean_B - mean_A`: `mean = mean_A + delta n_B / n`,
`M2 = M2_A + M2_B + delta^2 n_A n_B / n`, and for vectors
`C_jk = C_A,jk + C_B,jk + delta_j delta_k n_A n_B / n`.

- Domain: `n_A, n_B >= 1`; an empty chunk is the identity.
- Meaning: summaries of chunks combine into the summary of the concatenated
  data, so the computation parallelizes; the single-value update of
  [EQ-STAT-STREAM-001] is the case `n_B = 1`.
- Code: `statistics.RunningMoments.merge`, `statistics.RunningCovariance`.
- Source: [REF-CHAN-GOLUB-LEVEQUE-1983], [REF-PEBAY-2008].

### EQ-STAT-STREAM-003 — Pairwise merge of third and fourth central sums

With the notation of [EQ-STAT-STREAM-002]:
`M3 = M3_A + M3_B + delta^3 n_A n_B (n_A - n_B) / n^2
+ 3 delta (n_A M2_B - n_B M2_A) / n` and
`M4 = M4_A + M4_B + delta^4 n_A n_B (n_A^2 - n_A n_B + n_B^2) / n^3
+ 6 delta^2 (n_A^2 M2_B + n_B^2 M2_A) / n^2 + 4 delta (n_A M3_B - n_B M3_A) / n`.
Then `g1 = sqrt(n) M3 / M2^(3/2)` and `g2 = n M4 / M2^2 - 3`.

- Domain: as [EQ-STAT-STREAM-002]; `M2 > 0` for `g1`, `g2`.
- Meaning: one-pass, mergeable skewness and kurtosis equal to
  [EQ-STAT-SHAPE-001]; with `n_B = 1` and `M2_B = M3_B = M4_B = 0` this is the
  streaming update.
- Code: `statistics.RunningMoments`.
- Source: [REF-PEBAY-2008].

### EQ-STAT-STREAM-004 — West's weighted update and weighted-variance conventions

For weight `w > 0`: `W' = W + w`, `delta = x - mean`, `R = delta w / W'`,
`mean' = mean + R`, `S' = S + W delta R`; merge
`S = S_A + S_B + delta^2 W_A W_B / W`. Variances: population `S / W`,
frequency `S / (W - 1)`, reliability `S / (W - V / W)` with `V = sum w_i^2`;
effective sample size `W^2 / V`.

- Domain: finite values, finite non-negative weights, `W > 0`; each
  denominator must be positive.
- Meaning: `S = sum w_i (x_i - mean)^2` and `mean = sum w_i x_i / W` in one
  pass; the convention states whether weights are repeat counts or relative
  precisions.
- Code: `statistics.WeightedRunningMoments`.
- Source: [REF-WEST-1979], [REF-CHAN-GOLUB-LEVEQUE-1983].

### EQ-STAT-STREAM-005 — P² marker adjustment

Markers at heights `q_1..q_5` and integer positions `n_1..n_5` with desired
positions `n'_i` advancing by `(0, p/2, p, (1+p)/2, 1)` per observation. If
`d = n'_i - n_i` satisfies `d >= 1, n_{i+1} - n_i > 1` or
`d <= -1, n_{i-1} - n_i < -1`, set `s = sign(d)`,
`q_i <- q_i + s / (n_{i+1} - n_{i-1}) [ (n_i - n_{i-1} + s)(q_{i+1} - q_i) /
(n_{i+1} - n_i) + (n_{i+1} - n_i - s)(q_i - q_{i-1}) / (n_i - n_{i-1}) ]`
when this stays strictly between `q_{i-1}` and `q_{i+1}`, else the linear
`q_i + s (q_{i+s} - q_i) / (n_{i+s} - n_i)`; then `n_i <- n_i + s`.

- Domain: `0 < p < 1`, at least six observations (exact type-7 quantile
  before).
- Meaning: five-number fixed-memory estimate `q_3` of the `p`-quantile; a
  heuristic without a worst-case error bound and not mergeable.
- Code: `statistics.P2Quantile`.
- Source: [REF-JAIN-CHLAMTAC-1985].

### EQ-STAT-STREAM-006 — Reservoir sampling (Algorithm R)

Keep the first `k` items; for item `t > k` draw `J` uniform on
`{1, ..., t}` and replace slot `J` when `J <= k`. After `n` items each item is
retained with probability `min(1, k / n)` and every `k`-subset is equally
likely.

- Domain: capacity `k >= 1`; a uniform integer generator.
- Meaning: an exact uniform random subsample of fixed size from a stream of
  unknown length.
- Code: `statistics.ReservoirSampler`.
- Source: [REF-VITTER-1985].

### EQ-STAT-STREAM-007 — Relative-standard-error stopping rule

`r_n = s_n / (sqrt(n) |mean_n|)` with `s_n^2 = M2_n / (n - 1)`; stop at the
first check `n in {n_0, n_0 + h, n_0 + 2h, ...}` with `r_n <= epsilon`, or at
`n = N_max`.

- Domain: `n >= 2`, `epsilon > 0`, `n_0 >= 2`, `h >= 1`; `r_n = inf` when
  `mean_n = 0` and `s_n > 0`.
- Meaning: a fixed relative precision target for a running Monte Carlo mean,
  with constant memory; data-dependent stopping makes `r_n` slightly
  optimistic.
- Code: `statistics.run_until`, `statistics.RunningMoments`,
  `monte_carlo.estimate_mean_streaming`.
- Source: [REF-ROBERT-CASELLA-2004, Chapters 3-4] ([EQ-MC-SE-001]).

## W7 — Input-law expressiveness: copulas, discrete inputs, epistemic separation (ADR-0064)

### EQ-MOD-COP-SKLAR-001 — Sklar decomposition of a joint law

`F(x) = C(F_1(x_1), ..., F_d(x_d))`, `f(x) = c(u) prod_i f_i(x_i)` with
`u_i = F_i(x_i)` and `c = d^d C / du_1 ... du_d`.

- Domain: continuous marginals (the copula is then unique); for discrete
  marginals `C` is unique only on the range of `(F_1, ..., F_d)`.
- Meaning: separates the marginal laws from the dependence structure; every
  copula boundary condition `C(..., 0, ...) = 0` and `C(..., 1, ...)` = margin
  is used by `Copula.cdf`.
- Code: `modeling.Copula.cdf`, `modeling.Copula.log_density`,
  `modeling.JointInput.log_pdf`.
- Source: [REF-NELSEN-2006, Section 2.3].

### EQ-MOD-COP-ROSEN-001 — Rosenblatt transformation

`w_1 = u_1`, `w_k = C(u_k | u_1, ..., u_{k-1}) = P(U_k <= u_k | U_<k = u_<k)`;
standard space `u'_k = Phi^-1(w_k)`. The inverse solves each conditional for
`u_k` in turn (conditional-inverse construction).

- Domain: `u` in `(0, 1)^d`; depends on the variable order. For the Gaussian
  copula it equals the Nataf map `Phi(L^-1 Phi^-1(u))`, whose FORM results do
  not depend on the order.
- Meaning: an exact isoprobabilistic map to independent uniforms (normals);
  FORM linearizes the limit state in this space, so FORM/SORM approximations
  can change with the order for non-Gaussian copulas.
- Code: `modeling.Copula.rosenblatt`, `modeling.Copula.inverse_rosenblatt`,
  `modeling.JointInput.to_standard_normal`,
  `modeling.JointInput.from_standard_normal`, `modeling.JointInput.from_unit`.
- Source: [REF-ROSENBLATT-1952], [REF-HOHENBICHLER-RACKWITZ-1981],
  [REF-LEBRUN-DUTFOY-2009].

### EQ-MOD-COP-TAU-001 — Population Kendall's tau of a copula

`tau = 4 E[C(U, V)] - 1`; elliptical copulas: `tau = (2/pi) arcsin(rho)`;
Archimedean: `tau = 1 + 4 int_0^1 phi(t)/phi'(t) dt`.

- Domain: bivariate margins of continuous laws.
- Meaning: a margin-free concordance measure in closed form for every family
  implemented here (the Archimedean integral is used as an independent check
  in the tests).
- Code: `modeling.Copula.kendall_tau`.
- Source: [REF-NELSEN-2006, Chapter 5], [REF-MCNEIL-FREY-EMBRECHTS-2015].

### EQ-MOD-COP-TAIL-001 — Tail-dependence coefficients

`lambda_L = lim_{q->0+} C(q, q) / q`,
`lambda_U = lim_{q->1-} (1 - 2q + C(q, q)) / (1 - q)`.

- Domain: bivariate margins.
- Meaning: probability of a joint extreme given one extreme; zero for the
  Gaussian and Frank copulas.
- Code: `modeling.Copula.tail_dependence`.
- Source: [REF-NELSEN-2006, Chapter 5], [REF-JOE-2014].

### EQ-MOD-COP-GAUSS-001 — Bivariate normal CDF by Plackett's identity

`Phi_2(a, b; rho) = Phi(a) Phi(b) + int_0^rho phi_2(a, b; r) dr` with
`phi_2(a, b; r) = exp(-(a^2 - 2abr + b^2) / (2(1 - r^2))) / (2 pi sqrt(1 - r^2))`.

- Domain: `|rho| < 1`, finite `a, b`; the integral is evaluated by adaptive
  Gauss-Kronrod quadrature (relative tolerance `1e-12`).
- Meaning: the bivariate Gaussian-copula CDF `C(u, v) = Phi_2(Phi^-1(u),
  Phi^-1(v); rho)`, derived from `d Phi_2 / d rho = phi_2`.
- Code: `modeling.GaussianCopula.cdf`.
- Source: [REF-PLACKETT-1954].

### EQ-MOD-COP-T-001 — Student-t copula density and stochastic representation

`log c(u) = lgamma((nu+d)/2) + (d-1) lgamma(nu/2) - d lgamma((nu+1)/2)
- (1/2) log det R - ((nu+d)/2) log(1 + x'R^-1 x / nu)
+ ((nu+1)/2) sum_i log(1 + x_i^2 / nu)`, `x_i = T_nu^-1(u_i)`;
samples `U = T_nu(L Z / sqrt(W / nu))`, `Z ~ N(0, I)`, `W ~ chi^2_nu`.

- Domain: `nu > 0` (not necessarily integer), `R` positive-definite
  correlation.
- Meaning: elliptical copula with symmetric tail dependence.
- Code: `modeling.StudentTCopula.log_density`, `modeling.StudentTCopula.sample`.
- Source: [REF-DEMARTA-MCNEIL-2005], [REF-MCNEIL-FREY-EMBRECHTS-2015].

### EQ-MOD-COP-T-002 — Sequential conditionals of the multivariate t

With `x = L y`, `q_k = sum_{j<k} y_j^2`:
`T_k = y_k sqrt((nu + k) / (nu + q_k)) ~ t_{nu+k}`, independent of
`y_<k`, so `w_k = T_{nu+k}(T_k)` (0-based `k`).

- Domain: as [EQ-MOD-COP-T-001].
- Meaning: the conditional law of a multivariate t given earlier
  coordinates is t with `nu + k` degrees of freedom and scale inflated by
  `(nu + q_k)/(nu + k)`; this gives the t-copula Rosenblatt map in closed form.
- Code: `modeling.StudentTCopula.rosenblatt`,
  `modeling.StudentTCopula.inverse_rosenblatt`.
- Source: [REF-DEMARTA-MCNEIL-2005], [REF-ROSENBLATT-1952].

### EQ-MOD-COP-T-003 — t-copula tail dependence

`lambda_L = lambda_U = 2 T_{nu+1}(-sqrt((nu + 1)(1 - rho)/(1 + rho)))`.

- Domain: `|rho| < 1`, `nu > 0`.
- Meaning: positive for every finite `nu` even when `rho = 0`.
- Code: `modeling.StudentTCopula.tail_dependence`.
- Source: [REF-DEMARTA-MCNEIL-2005].

### EQ-MOD-COP-ARCH-001 — Archimedean copula, density

`C(u) = psi(sum_i phi(u_i))`, `phi = psi^-1`,
`c(u) = |psi^(d)(t)| prod_i |phi'(u_i)|` with `t = sum_i phi(u_i)`.

- Domain: `psi` d-monotone (completely monotone for every `d`: Clayton
  `theta > 0`, Gumbel `theta >= 1`, Frank `theta > 0`; bivariate Frank also
  `theta < 0`).
- Meaning: exchangeable dependence from one generator; the CDF is closed
  form in every dimension.
- Code: `modeling.ArchimedeanCopula.cdf`,
  `modeling.ArchimedeanCopula.log_density`.
- Source: [REF-NELSEN-2006, Chapter 4], [REF-MCNEIL-FREY-EMBRECHTS-2015].

### EQ-MOD-COP-ARCH-002 — Archimedean conditional distributions

`C(u_k | u_1, ..., u_{k-1}) = psi^(k-1)(t_k) / psi^(k-1)(t_{k-1})`,
`t_k = sum_{i<=k} phi(u_i)`. The inverse solves
`log|psi^(k-1)(t_{k-1} + s)| - log|psi^(k-1)(t_{k-1})| = log w` for
`s = phi(u_k)` (closed form for Clayton, Brent's method on `log s` otherwise).

- Domain: as [EQ-MOD-COP-ARCH-001]; the left side decreases from 0 to
  `-inf` in `s`, so the root exists and is unique.
- Meaning: the Archimedean Rosenblatt map; a conditional probability within
  `epsilon` of one keeps only absolute accuracy `2^-53` in its complement.
- Code: `modeling.ArchimedeanCopula.rosenblatt`,
  `modeling.ArchimedeanCopula.inverse_rosenblatt`.
- Source: [REF-HOFERT-2008], [REF-ROSENBLATT-1952], [REF-BRENT-1973, Chapter 4].

### EQ-MOD-COP-ARCH-003 — Marshall-Olkin frailty sampling

`V ~ F` with Laplace transform `psi`, `E_i ~ Exp(1)` independent,
`U_i = psi(E_i / V)`.

- Domain: `psi` a Laplace transform (completely monotone generator).
- Meaning: an exact sampler using `d + 1` (or more) random numbers; used by
  `sample` and as an independent check of the conditional-inverse sampler
  that `JointInput.from_unit` uses.
- Code: `modeling.ArchimedeanCopula.sample`.
- Source: [REF-MARSHALL-OLKIN-1988], [REF-HOFERT-2008].

### EQ-MOD-COP-CLAYTON-001 — Clayton generator and properties

`psi(t) = (1 + t)^(-1/theta)`, `phi(u) = u^-theta - 1`,
`|psi^(k)(t)| = prod_{j<k} (1/theta + j) (1 + t)^(-1/theta - k)`,
conditional inverse `s = (1 + t_{k-1}) expm1(-log w / (1/theta + k - 1))`;
`tau = theta / (theta + 2)`, `lambda_L = 2^(-1/theta)`, `lambda_U = 0`;
frailty `V ~ Gamma(1/theta, 1)`.

- Domain: `theta > 0`, any dimension.
- Meaning: lower-tail dependence.
- Code: `modeling.ClaytonCopula`.
- Source: [REF-NELSEN-2006, Chapters 4-5], [REF-HOFERT-2008].

### EQ-MOD-COP-GUMBEL-001 — Gumbel generator, derivatives, and properties

`psi(t) = exp(-t^a)`, `a = 1/theta`, `phi(u) = (-log u)^theta`;
`|psi^(k)(t)| = e^(-t^a) sum_m a_km t^(m a - k)` with `a_00 = 1`,
`a_{k+1,m+1} += a a_km`, `a_{k+1,m} += (k - m a) a_km` (all terms
non-negative for `a <= 1`); `tau = 1 - 1/theta`, `lambda_U = 2 - 2^(1/theta)`,
`lambda_L = 0`.

- Domain: `theta >= 1` (`theta = 1` is independence), any dimension.
- Meaning: upper-tail dependence. The coefficient recursion follows from
  differentiating `e^(-t^a) t^(m a - k)` (derived for this library and
  verified against mpmath derivatives of the CDF).
- Code: `modeling.GumbelCopula`.
- Source: [REF-NELSEN-2006, Chapters 4-5], [REF-HOFERT-MACHLER-MCNEIL-2012].

### EQ-MOD-COP-GUMBEL-002 — Positive stable frailty (Kanter's representation)

`V = sin(a U) sin((1 - a) U)^((1-a)/a) / (sin(U)^(1/a) E^((1-a)/a))` with
`U ~ Uniform(0, pi)`, `E ~ Exp(1)`, has `E[exp(-t V)] = exp(-t^a)`.

- Domain: `0 < a < 1` (`a = 1` gives `V = 1`).
- Meaning: the Gumbel frailty for [EQ-MOD-COP-ARCH-003].
- Code: `modeling.GumbelCopula.sample`.
- Source: [REF-KANTER-1975], [REF-CHAMBERS-MALLOWS-STUCK-1976],
  [REF-HOFERT-2008].

### EQ-MOD-COP-FRANK-001 — Frank generator, derivatives, and properties

`psi(t) = -log(1 - (1 - e^-theta) e^-t) / theta`,
`phi(u) = -log(expm1(-theta u) / expm1(-theta))`;
`psi^(k)(t) = (-1)^k Li_{1-k}(x) / theta`, `x = (1 - e^-theta) e^-t`, with
`Li_0(x) = x/(1-x)` and `Li_{-n}(x) = x P_n(x) / (1 - x)^(n+1)`, `P_n` the
Eulerian polynomial `sum_j A(n, j) x^j`; `tau = 1 + 4 (D_1(theta) - 1)/theta`,
no tail dependence.

- Domain: `theta != 0` for `d = 2`, `theta > 0` for `d >= 3`; `|theta| <= 700`.
- Meaning: symmetric, tail-independent dependence of either sign (bivariate).
- Code: `modeling.FrankCopula`, `modeling.frank_tau`.
- Source: [REF-NELSEN-2006, Chapters 4-5], [REF-NIST-DLMF, Sections 25.12
  and 26.14].

### EQ-MOD-COP-FRANK-002 — Logarithmic frailty (Kemp's algorithm LK)

`P(V = k) = p^k / (-k log(1 - p))`, `p = 1 - e^-theta`; with `U, V'` uniform:
return 1 if `V' >= p`; else `q = 1 - (1 - p)^U` and return
`floor(1 + log V' / log q)` if `V' < q^2`, 1 if `V' > q`, else 2.

- Domain: `0 < p < 1`.
- Meaning: the Frank frailty for [EQ-MOD-COP-ARCH-003]
  (`E[e^(-tV)] = psi(t)`).
- Code: `modeling.FrankCopula.sample`.
- Source: [REF-KEMP-1981], [REF-DEVROYE-1986, Chapter X].

### EQ-MOD-COP-DEBYE-001 — First Debye function

`D_1(x) = (1/x) int_0^x t / (e^t - 1) dt`; for `|x| < 2`
`(D_1(x) - 1)/x = -1/4 + sum_{k>=1} B_2k x^(2k-1) / ((2k + 1)(2k)!)`;
for `x >= 2` `int_0^x = pi^2/6 - sum_{k>=1} e^(-kx)(x/k + 1/k^2)`;
`D_1(-x) = D_1(x) + x/2`.

- Domain: real `x`; Bernoulli numbers `B_n` (with `B_1 = -1/2`) from
  `sum_{j<=m} C(m+1, j) B_j = 0` in exact rational arithmetic.
- Meaning: Kendall's tau of the Frank copula without cancellation near
  `theta = 0` (`tau ~ theta/9`).
- Code: `modeling.frank_tau`, `modeling.FrankCopula.kendall_tau`.
- Source: [REF-NIST-DLMF, Sections 24.2 and 24.5], [REF-NELSEN-2006, Chapter 5].

### EQ-MOD-COP-NCM-001 — Nearest correlation matrix (alternating projections)

`Delta S_0 = 0`, `Y_0 = A`; repeat `R_k = Y_k - Delta S_k`,
`X_k = P_S(R_k)`, `Delta S_{k+1} = X_k - R_k`, `Y_{k+1} = P_U(X_k)`, where
`P_S` clips eigenvalues below `delta` and `P_U` sets the diagonal to one.

- Domain: symmetric `A`; `delta` in `[0, 1)`; stop when
  `||Y_{k+1} - Y_k||_F <= tol ||Y_{k+1}||_F`.
- Meaning: Dykstra-corrected projections converge to the Frobenius-nearest
  unit-diagonal matrix with eigenvalues at least `delta`; used to repair
  pairwise latent correlations that are not jointly positive definite.
- Code: `modeling.nearest_correlation_matrix`, `modeling.JointInput.from_pearson`,
  `modeling.JointInput.from_spearman`, `modeling.JointInput.from_kendall`,
  `modeling.fit_copula`.
- Source: [REF-HIGHAM-2002].

### EQ-MOD-COPULA-FIT-003 — Sample Kendall's tau-b

`tau_b = (n_c - n_d) / sqrt((n_0 - n_1)(n_0 - n_2))`, `n_0 = n(n-1)/2`.

- Domain: `n >= 2`, neither sample constant; direct `O(n^2)` pair count.
- Meaning: rank concordance with a tie correction; equals tau-a without ties.
- Code: `modeling.kendall_tau`.
- Source: [REF-KENDALL-1938], [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 8].

### EQ-MOD-COPULA-FIT-004 — Archimedean tau inversion

Clayton `theta = 2 tau / (1 - tau)`; Gumbel `theta = 1 / (1 - tau)`; Frank
solves `1 + 4 (D_1(theta) - 1)/theta = tau`; multivariate exchangeable fits
use the average pairwise sample tau.

- Domain: Clayton `tau > 0`, Gumbel `tau >= 0`, Frank `tau != 0`
  (`tau > 0` for `d >= 3`).
- Meaning: method-of-moments estimate of the dependence parameter from ranks.
- Code: `modeling.archimedean_theta_from_tau`, `modeling.fit_copula`.
- Source: [REF-GENEST-RIVEST-1993], [REF-NELSEN-2006, Chapter 5].

### EQ-MOD-COPULA-FIT-005 — Kendall's tau to elliptical latent correlation

`rho = sin(pi tau / 2)`.

- Domain: `tau` in `[-1, 1]`; any elliptical copula (Gaussian, t).
- Meaning: inverts `tau = (2/pi) arcsin(rho)`.
- Code: `modeling.kendall_to_latent_correlation`, `modeling.JointInput.from_kendall`,
  `modeling.fit_copula`.
- Source: [REF-MCNEIL-FREY-EMBRECHTS-2015], [REF-DEMARTA-MCNEIL-2005].

### EQ-MOD-COPULA-FIT-006 — Maximum pseudo-likelihood

`theta_hat = argmax_theta sum_i log c(u_i; theta)`,
`u_ij = rank_ij / (n + 1)`.

- Domain: one-parameter families (Clayton, Gumbel, Frank; t-copula `nu`
  with `R` fixed from tau); grid scan then golden-section search over a
  bounded transformed range.
- Meaning: the rank-based (canonical maximum likelihood) estimator, consistent
  and asymptotically normal; the inverse pseudo-information understates its
  variance, so no standard error is reported.
- Code: `modeling.fit_copula`.
- Source: [REF-GENEST-GHOUDI-RIVEST-1995], [REF-BRENT-1973, Chapter 5],
  [REF-KIEFER-1953].

### EQ-MOD-DISC-001 — Distributional transform of a discrete input

`U = F(X-) + V (F(X) - F(X-))`, `V ~ Uniform(0, 1)` independent of `X`
(`V = 1/2` without a seed); `U ~ Uniform(0, 1)` and `F^-1(U) = X` a.s.

- Domain: integer-valued `X`; `F(X-) = F(X - 1)`.
- Meaning: a randomized probability integral transform; the latent map is not
  one-to-one (the whole interval `(F(x-), F(x)]` maps back to `x`), so
  gradient-based reliability methods are refused for discrete inputs.
- Code: `modeling.JointInput.to_latent`, `modeling.JointInput.to_standard_normal`.
- Source: [REF-RUSCHENDORF-2009].

### EQ-MOD-DISC-002 — Joint mass of discrete inputs under a copula

`P(X = x) = sum_{e in {0,1}^d} (-1)^{|e|} C(F_1(x_1 - e_1), ..., F_d(x_d - e_d))`.

- Domain: integer-valued margins, a copula CDF available (Archimedean any
  `d`, Gaussian `d <= 2`, independence); at most 16 inputs.
- Meaning: rectangle inclusion-exclusion over the copula CDF; mixed
  discrete/continuous inputs under a dependent copula need partial
  derivatives of `C` and are not implemented.
- Code: `modeling.JointInput.log_pdf`.
- Source: [REF-GENEST-NESLEHOVA-2007], [REF-NELSEN-2006, Section 2.10].

### EQ-PROP-MIXED-001 — Nested (double-loop) propagation

For outer epistemic values `e_j` (Latin hypercube over the epistemic
dimensions, plus interval corners and optional compass-search points):
`F_j(y) = (1/n) sum_i 1{g(x_i, e_j) <= y}` with `x_i` the aleatory inner
sample (common to every `j` with common random numbers); statistics
`s(e_j)` = mean, `P(Y > t)`, type-7 quantiles; bounds
`[min_j s(e_j), max_j s(e_j)]`.

- Domain: aleatory and epistemic inputs independent; at least one of each.
- Meaning: separates variability (inner CDF) from lack of knowledge (outer
  spread). The bounds over a finite design are inner approximations of the
  true bounds (exact at the corners for responses monotone in each interval
  input); cost `(outer points) x n` model evaluations.
- Code: `propagation.nested_propagation`.
- Source: [REF-ROY-OBERKAMPF-2011], [REF-OBERKAMPF-ROY-2010].

### EQ-PROP-PBOX-001 — Probability box from a family of CDFs

`F_low(y) = min_j F_j(y)`, `F_up(y) = max_j F_j(y)`;
`P(Y > t)` in `[1 - F_up(t), 1 - F_low(t)]`.

- Domain: the outer points of [EQ-PROP-MIXED-001].
- Meaning: the pointwise envelope of the output CDFs (a p-box); for
  `Y = X + E`, `E` in `[a, b]`, it is `[Phi(y - b), Phi(y - a)]`.
- Code: `propagation.ProbabilityBox.bounds`,
  `propagation.ProbabilityBox.exceedance_bounds`.
- Source: [REF-FERSON-ET-AL-2003].
