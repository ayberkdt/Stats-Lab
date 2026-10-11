# Model-Based Reliability: FORM, SORM, Subset Simulation, and Systems

[Reliability and rare-event analysis](reliability-analysis.md) introduced the
failure convention `g <= 0` and estimators that work on explicit vectors. This
guide covers reliability methods that run on one
`probstat.modeling.ProbabilisticModel`. The model owns the joint input law, so
FORM, SORM, importance sampling, subset simulation, system analysis, and
inverse design all use the same marginals and the same Gaussian copula.

## 1. The limit state comes from the model

`ModelLimitState(model, quantity, threshold, failure)` turns one quantity of
interest `y` into a limit state [EQ-REL-MLS-001]:

| `failure` | limit state | failure event |
| --- | --- | --- |
| `"above"` | `g = t - y` | `y >= t` |
| `"below"` | `g = y - t` | `y <= t` |

Methods work in independent standard-normal space `u`. The Nataf map
[EQ-MOD-NATAF-001] links `u` to the physical inputs: `z = L u`,
`x_i = F_i^-1(Phi(z_i))`. `G(u) = g(T^-1(u))` is the same limit state written
in `u`. Every method reports how many model evaluations it used. These counts
come from the model's own counter.

## 2. FORM by HL-RF with a merit line search

The design point `u*` is the point of `G(u) = 0` closest to the origin. The
reliability index is `beta = alpha . u*`, where
`alpha = -grad G / ||grad G||`, and `P_f ~= Phi(-beta)` [EQ-REL-FORM-002].
`form_hlrf` repeats the HL-RF step [EQ-REL-HLRF-001]. By default the step
length comes from an Armijo search on the iHL-RF merit function
`||u||^2/2 + c|G|` [EQ-REL-HLRF-002]. Plain HL-RF (`line_search=False`) can
cycle on strongly nonlinear limit states. With the merit function, every HL-RF
direction lowers the merit, so the search cannot cycle.

Outputs:

- `beta` carries a sign. It is negative when the median input already fails.
- The design point is given in both `u` and `x`.
- `importance_factors` are the values `alpha_i^2`. They sum to one.
- `gamma` is the importance vector for correlated inputs.

With correlation, `u_i` is a Cholesky coordinate, so `alpha_i^2` depends on
the input order. `gamma = L^-T alpha / ||L^-T alpha||` refers to the physical
inputs instead.

FORM fails or misleads in these cases:

- The limit surface is strongly curved (use SORM, or sampling).
- There are several design points of similar distance (FORM finds one, and
  which one depends on the start point).
- The gradient vanishes. The code raises an error.
- The search does not converge within `max_iterations`. The result then has
  `converged=False`, and SORM refuses to use it.

**Upper-tail precision limit.** `JointInput.from_standard_normal` evaluates
`F^-1(Phi(z))`. For latent values `z > 6`, `Phi(z)` is within `1e-9` of one,
and double precision rounds it. Beyond `z ~ 8.1` the input stops changing.
Finite-difference gradients then become noisy, and FORM can stall when a
design-point coordinate is in that region. This was observed for
`beta ~ 6.8` and `8.5` on a lognormal margin. Treat indices above about 6 with
care until the transformation evaluates the upper tail through `Phi(-z)`.

## 3. SORM from principal curvatures

`sorm` takes a central-difference Hessian of `G` at `u*` (`2n^2 + 1` calls). It
rotates the Hessian so that the last axis is `alpha`. The principal curvatures
`kappa_i` are the eigenvalues of the tangent block divided by `||grad G||`
[EQ-REL-SORM-001]. A positive curvature bends the surface away from the origin
and lowers `P_f`. Two corrections are reported:

- Breitung: `Phi(-beta) prod (1 + beta kappa_i)^(-1/2)` [EQ-REL-SORM-002].
- Hohenbichler-Rackwitz: `beta` in the product is replaced by
  `psi = phi(beta)/Phi(-beta)` [EQ-REL-SORM-003].

Both corrections require `beta > 0` and `1 + beta kappa_i > 0`. If either
condition fails, the code raises an error. When `1 + beta kappa_i <= 0`, the
paraboloid does not approximate the failure domain, and continuing would give
a wrong number.

## 4. Sampling around the design point and subset simulation

`design_point_importance_sampling` draws from `N(u*, I)` and weights each draw
by `w = exp(||u*||^2/2 - u . u*)` [EQ-REL-ISDP-001]. The method is efficient
when the failure mass is concentrated near `u*`. It misses failure regions far
from `u*`, and high dimension makes the weights degenerate.

`subset_simulation` [EQ-REL-SUBSET-001] works without a design point. It
writes a small `P_f` as a product of conditional probabilities near `p0 = 0.1`.
Each intermediate threshold is a quantile of the current level. A
component-wise modified Metropolis sampler [EQ-REL-SUBSET-002] repopulates
each level.

The reported c.o.v. [EQ-REL-SUBSET-003] accounts for correlation inside each
chain. It treats the levels as independent, so it tends to understate the
error. `coefficient_of_variation_upper` is the value for fully correlated
levels.

Evidence from runs: for `G = 4 - (u1 + u2)/sqrt 2` (`P_f = 3.17e-5`) with
`N = 1000`, six seeds gave about 4150-4180 calls each. The ratio of estimate to
exact value ranged from 0.33 to 2.40, with mean 1.01. The independence c.o.v.
was 0.33-0.36 and the upper c.o.v. was 0.70-0.78. The observed spread matches
the upper value better.

## 5. Systems of limit states

`system_failure_probability` estimates series systems (`min_k g_k <= 0`) and
parallel systems (`max_k g_k <= 0`) [EQ-REL-SYS-001] on the shared joint
input. Each distinct model is called once per sample. Two methods are
available:

- Crude Monte Carlo.
- Importance sampling from an equal mixture centered at the component design
  points. It suits series systems. It is a poor proposal for parallel systems
  whose joint design point is far from every component design point.

`series_system_bounds` linearizes each component with
`rho_ij = alpha_i . alpha_j` and returns two kinds of bounds:

- Uni-modal bounds [EQ-REL-SYS-002]. The upper bound is valid only for
  positively correlated components, which the `positively_correlated` field
  reports.
- Ditlevsen bi-modal bounds [EQ-REL-SYS-003]. They use
  `P_ij = Phi_2(-beta_i, -beta_j; rho_ij)`.

`bivariate_normal_cdf` implements Genz's version of the Drezner-Wesolowsky
Gauss-Legendre scheme [EQ-REL-BVN-001] [EQ-REL-BVN-002]. It matches an
independent quadrature to `4e-15` over a grid of 714 points, including
`|rho| >= 0.925`.

## 6. Inverse FORM and target-reliability design

- `inverse_form(state, beta_t)` returns the threshold whose FORM index is
  `beta_t`. It runs the fixed-point iteration `u = beta_t alpha(u)` on the
  sphere `||u|| = beta_t` [EQ-REL-INV-001].
- `design_for_target_beta(state, name, beta_t, lower, upper)` finds a
  deterministic parameter in `model.fixed`. It applies Brent's method to
  `beta(theta) - beta_t` [EQ-REL-RBDO-001].

Both report whether they converged. A bracket without a sign change raises an
error.

## 7. Worked example (numbers computed with this release)

Inputs: `R ~ LogNormal(1.2, 0.15)` and `S ~ LogNormal(0.3, 0.25)`, with latent
correlation `0.4`. The margin is `y = R - S`, and failure is `y <= 0`. The
surface `G = 0` is a plane in `u`, so FORM is exact. The exact index is
`beta = 0.9 / sqrt(0.15^2 + 0.25^2 - 2(0.4)(0.15)(0.25)) = 3.837613`.

FORM results: `beta = 3.8376128944` after 6 iterations and 35 calls. The design
point is `u* = (-0.818, 3.749)` and `x* = (2.9367, 2.9367)`. The importance
factors are `alpha^2 = (0.045, 0.955)`, and the correlated importance vector is
`gamma = (-0.514, 0.857)`. Most of the importance falls on `S`.

`compare_reliability_methods(state, monte_carlo_samples=20000, seed=2026)`:

| Method | `P_f` | equivalent `beta` | calls | c.o.v. |
| --- | --- | --- | --- | --- |
| FORM | 6.2118e-5 | 3.8376 | 35 | — |
| SORM-Breitung | 6.2118e-5 | 3.8376 | 44 | — |
| SORM-HR | 6.2119e-5 | 3.8376 | 44 | — |
| IS (2000 draws) | 6.490e-5 | 3.8268 | 2035 | 0.046 |
| Subset (N = 1000) | 6.19e-5 | 3.8385 | 4191 | 0.34 |
| Crude MC | 1.0e-4 | 3.7190 | 20000 | 0.71 |

Crude MC observed two failures in 20000 draws. With the same number of calls,
it is far less precise than IS around the design point. Inverse FORM with
`beta_t = 3` returns the threshold `y = 0.5224`, after 10 iterations and 41
calls.

The curvature correction matters for the parabolic limit state
`G = 3 - u2 + kappa u1^2 / 2`. Its exact value comes from Gauss-Hermite
quadrature of `E[Phi(-(3 + kappa U^2/2))]`:

| `kappa` | exact | FORM | Breitung | Hohenbichler-Rackwitz |
| --- | --- | --- | --- | --- |
| +0.2 | 1.0436e-3 | 1.3499e-3 | 1.0672e-3 | 1.0488e-3 |
| -0.2 | 2.1257e-3 | 1.3499e-3 | 2.1344e-3 | 2.3036e-3 |

The series system has three linear components with `beta = (2.5, 2.7, 2.6)`
and correlations `(0.6, 0.8, 0)`. Its bounds and estimate:

- Uni-modal bounds: `[0.00621, 0.01427]`.
- Ditlevsen bounds: `[0.011813, 0.011829]`.
- Mixture importance sampling (20000 draws): `0.01195 +/- 0.00014`.

## Sources

- [REF-HASOFER-LIND-1974], [REF-RACKWITZ-FIESSLER-1978],
  [REF-ZHANG-DERKIUREGHIAN-1995], [REF-DERKIUREGHIAN-2005].
- [REF-BREITUNG-1984], [REF-HOHENBICHLER-RACKWITZ-1988].
- [REF-AU-BECK-2001].
- [REF-CORNELL-1967], [REF-DITLEVSEN-1979], [REF-GENZ-2004],
  [REF-DREZNER-WESOLOWSKY-1990].
- [REF-DERKIUREGHIAN-ZHANG-LI-1994], [REF-TU-CHOI-PARK-1999].
- [REF-MELCHERS-BECK-2018, Chapters 5 and 6], [REF-DITLEVSEN-MADSEN-1996].

## 8. Cross-entropy importance sampling and line sampling

Two further estimators in `probstat.reliability.adaptive` work on the same
`ModelLimitState` in standard-normal space. Both are opt-in rows of
`compare_reliability_methods`: `methods=(..., "ce", "line")` adds the rows
`"CE-IS"` and `"Line"`. The default `METHODS` tuple is unchanged, so existing
comparisons and the study workflow give the same numbers as before;
`AVAILABLE_METHODS` lists all seven keys.

### Cross-entropy adaptive importance sampling

`cross_entropy_importance_sampling` needs no design point. It starts from
`N(0, I)`, draws `N` points per level, and sets the intermediate threshold to
the `rho = 0.1` sample quantile of `G` (clipped at 0). The proposal mean (and,
optionally, covariance) is refitted to the elite points `G <= gamma_t` with
likelihood-ratio weights `phi_n(u) / h(u)` [EQ-REL-CE-001]. When a level
reaches `gamma = 0`, a fresh sample from the final proposal gives the
importance-sampling estimate and its c.o.v. [EQ-REL-CE-002]. The result
reports the thresholds, final mean and covariance, the Kish effective sample
size, the number of levels, and every model evaluation (levels plus final
stage).

The default adapts the **mean only**. `adapt_covariance=True` also fits the
covariance, but the elite covariance shrinks along the important direction.
On `G = 4 - (u1 + u2)/sqrt 2` (`P_f = 3.17e-5`) with `N = 500` and seeds 1-5:

| seed | mean-shift ratio to exact | c.o.v. | covariance-adapted ratio | c.o.v. |
| --- | --- | --- | --- | --- |
| 1 | 1.063 | 0.094 | 0.429 | 0.35 |
| 2 | 1.007 | 0.094 | 1.009 | 0.045 |
| 3 | 0.975 | 0.096 | 0 (stalled, 20 levels) | — |
| 4 | 0.856 | 0.104 | 0 (stalled, 20 levels) | — |
| 5 | 1.046 | 0.094 | 0 (stalled, 20 levels) | — |

Each mean-shift run took 3 levels and 2000 calls. With covariance adaptation
the levels stall on this planar limit state, and a finished run can still be
far off (seed 1: 0.43 of the exact value). In a trial with `rho = 0.2`, seed 1
returned 0.73 of the exact value with a reported c.o.v. of 0.04, so the
reported c.o.v. can understate the error badly. Gaussian-mixture proposals
[REF-KURTZ-SONG-2013] address multiple failure regions and are not implemented.

The table above shows the unguarded adaptation, still available as
`covariance_floor=None`. Adapted covariances are now floored at eigenvalue
`0.55` by default (the weight variance is infinite once an eigenvalue drops to
`1/2` or below [EQ-REL-CE-003]), and a collapsed or degenerate run warns and
sets `weights_degenerate`. See
[Adaptive reliability](adaptive-reliability.md) for the guarded numbers.

### Line sampling

`line_sampling` uses an important direction `alpha`: the FORM `alpha` by
default (FORM runs once unless a `FORMResult` or a `direction` is supplied).
Each line draws `u ~ N(0, I)`, projects it onto the hyperplane orthogonal to
`alpha`, and finds the crossing `c_k` of `G(u_perp + c alpha) = 0` by bracket
doubling from the previous root followed by Brent's method (cached, so the
bracket ends are not re-evaluated). The line contributes `p_k = Phi(-c_k)`, and
`P_f = mean(p_k)` with variance `sum (p_k - P_f)^2 / (N (N - 1))`
[EQ-REL-LINE-001]. A line that does not cross within `|c| <= max_distance`
contributes 0 (safe) or 1 (failed). The estimator assumes at most one crossing
per line with failure beyond it.

For a planar limit state every `p_k` equals `Phi(-beta)`, so line sampling is
exact with zero variance. For curved limit states it is conditional Monte
Carlo along `alpha` and is far more precise per call than crude Monte Carlo.

### Numbers computed with this release

Correlated lognormal `R - S` example of Section 7,
`compare_reliability_methods(state, methods=("form", "is", "ce", "line"),
seed=2026)` (defaults: 2000 IS draws, 1000 CE samples per level, 100 lines):

| Method | `P_f` | equivalent `beta` | calls | c.o.v. |
| --- | --- | --- | --- | --- |
| FORM | 6.2118e-5 | 3.8376 | 35 | — |
| IS | 6.490e-5 | 3.8268 | 2035 | 0.046 |
| CE-IS | 6.002e-5 | 3.8460 | 4000 | 0.064 |
| Line | 6.2118e-5 | 3.8376 | 335 | 0 (planar) |

Parabolic limit state `G = 3 - u2 + kappa u1^2 / 2` (seed 1; exact values by
Gauss-Hermite quadrature):

| `kappa` | exact | CE-IS (`N = 500`) | calls | Line (100 lines) | calls |
| --- | --- | --- | --- | --- | --- |
| +0.2 | 1.0436e-3 | 9.86e-4 (c.o.v. 0.091) | 2000 | 1.0153e-3 (c.o.v. 0.034) | 402 + 10 FORM |
| -0.2 | 2.1257e-3 | 1.90e-3 (c.o.v. 0.097) | 1500 | 2.1733e-3 (c.o.v. 0.069) | 401 + 10 FORM |

Line sampling needs about four calls per line here. CE-IS spends most of its
calls on adaptation but needs neither a gradient nor a design point.

Sources: [REF-RUBINSTEIN-KROESE-2004], [REF-KURTZ-SONG-2013],
[REF-KOUTSOURELAKIS-PRADLWARTER-SCHUELLER-2004], [REF-BRENT-1973].
