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
`Gamma_k` and made non-increasing; `ESS = M n / (-1 + 2 sum_k Gamma_k)`.

- Domain: `M` equal-length chains of at least four draws with positive
  pooled variance.
- Meaning: number of independent draws carrying the same information about
  the mean; for AR(1) with coefficient `phi` it tends to
  `M n (1 - phi)/(1 + phi)`.
- Code: `inference.effective_sample_size`, `inference.summarize_draws`.
- Source: [REF-GEYER-1992], [REF-VEHTARI-ETAL-2021], and
  [REF-GELMAN-ETAL-2013, Section 11.5].

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
