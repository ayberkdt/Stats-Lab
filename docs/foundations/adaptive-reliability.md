# Adaptive Reliability: Lines, Multiple Design Points, and AK-MCS

This guide extends [model-based reliability](advanced-reliability.md). All
methods work on a `ModelLimitState` in independent standard-normal space `u`,
and every number below comes from runs of this release (the lab
`examples/adaptive_reliability_lab.py` prints most of them).

## 1. A guard for covariance-adapted CE-IS

`cross_entropy_importance_sampling(adapt_covariance=True)` fits the elite
covariance at every level. For a near-planar failure domain the elite points
are a thin slab, so the variance along the important direction collapses. With
a Gaussian proposal `h = N(mu, Sigma)` the weight second moment
`int phi_n^2 / h` is finite only when every eigenvalue of `Sigma` exceeds `1/2`
[EQ-REL-CE-003]. Below that the weights are heavy-tailed. The sample variance
cannot see a tail it never sampled, so the estimate is biased low and its
c.o.v. looks small.

Two changes follow:

- Adapted covariances are projected onto `lambda_min >= covariance_floor`
  (default `0.55`) at every level. `covariance_floor=None` keeps the old
  behaviour for study.
- The result reports `failure_effective_sample_size` (Kish ESS of the failure
  weights), `min_covariance_eigenvalue`, and `weights_degenerate`. A
  `UserWarning` fires when an eigenvalue is `<= 1/2` or when the failure ESS
  is below `0.01 * final_samples`.

The Kish ESS alone does not detect the collapse. In the runs below several
unguarded estimates are 35-62% low while their failure ESS is 60-402. The
eigenvalue test catches every one of them.

`G = 4 - (u1 + u2)/sqrt 2`, exact `P_f = 3.1671e-5`, `N = 1000` per level:

| seed | floor | `P_f` | ratio | c.o.v. | z | `lambda_min` | flagged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | none | 1.2129e-5 | 0.383 | 0.125 | -12.89 | 0.0005 | yes |
| 1 | 0.55 | 3.2865e-5 | 1.038 | 0.055 | 0.66 | 0.55 | no |
| 2 | none | 2.0614e-5 | 0.651 | 0.039 | -13.90 | 0.0035 | yes |
| 2 | 0.55 | 3.1232e-5 | 0.986 | 0.058 | -0.24 | 0.55 | no |
| 3 | none | 0 | 0 | inf | — | 0.0000 | yes |
| 3 | 0.55 | 3.3558e-5 | 1.060 | 0.055 | 1.03 | 0.55 | no |

With `N = 500` and seeds 1-8, the guarded ratios to the exact value were
1.013, 1.080, 1.089, 0.912, 0.965, 0.966, 0.784, and 0.949. The z-scores were
0.17, 0.95, 1.10, -1.17, -0.45, -0.45, -3.16, and -0.64, so one of eight runs
was just outside 3 SE. All eight unguarded runs were flagged. One of them
(seed 2, ratio 1.009) was accurate by luck: the flag is conservative. The
mean-shift default never builds a covariance, so it gives the same numbers as
before. At `N = 1000`, seeds 1-5 gave 3.357e-5, 3.175e-5, 2.954e-5, 3.313e-5,
and 2.827e-5 both before and after the change.

**Several failure regions.** One Gaussian proposal follows one failure
region. The others get almost no samples, their weights are never observed,
and the weight diagnostics above stay healthy. On the four-branch system
(exact `4.457e-3`, mean-shift default, `N = 500`, integer seeds 1-10), seven
runs converged on one region and returned `0.95e-3` to `1.76e-3` with
`z = -7` to `-32` and failure ESS 21-126. Until ADR-0054 none of them warned.
The first adaptation level is drawn from the starting density before any
adaptation, so its elite points show where `G` is small in every direction.
`uncovered_elite_fraction` is the share of them behind the origin as seen from
the final proposal mean (`u . mu / ||mu|| <= 0`) [EQ-REL-CE-004]. Above
`CE_MAX_UNCOVERED_ELITE_FRACTION = 0.1` a `UserWarning` recommends
`find_design_points` with `mixture_importance_sampling`, or
`subset_simulation`, and `multiple_failure_regions_suspected` is set. The
seven runs gave 0.48-0.58. In the other three the mean stalled near the
origin (`||mu|| < 1`, 20 levels without converging). The check is skipped
there, and the existing failure-ESS warning fired (ESS 2-4). On the
single-region benchmark problems (both linear cases, the parabola, the
lognormal capacity-demand case; 10 seeds each) the fraction was 0.00 in 39 of
40 runs and 0.02 in one, so none warned. With `adapt_covariance=True` and
`N = 1000` the four-branch proposal stayed centred near the origin with a wide
covariance and covered all four regions (`|z| <= 1.2`, check skipped). The
check costs no model calls. It cannot see a region with no first-level elite
point, and an `initial_mean` weakens it.

## 2. Line sampling

`line_sampling` draws `u ~ N(0, I)`, removes its component along the
important direction `alpha` (the FORM `alpha` by default), and finds the
crossing `c_k` of `G(u_perp + c alpha) = 0` by bracket doubling and Brent's
method. The line contributes `p_k = Phi(-c_k)` [EQ-REL-LINE-001]. A line with
no crossing in `|c| <= max_distance` contributes exactly 0 if it is safe at
`+max_distance` (distance `+inf`) and exactly 1 if it fails at
`-max_distance` (distance `-inf`). The neglected mass is at most
`Phi(-max_distance)` per line. Before this release the public docstring said
such lines contribute `Phi(-max_distance)`, while the code returned exactly 0
or 1. The docstrings now state the code's behaviour, which also matches the
equation registry.

| case | `P_f` | reference | SE / c.o.v. | calls |
| --- | --- | --- | --- | --- |
| plane, `d = 3`, `beta = 3.5`, 30 lines | 2.326291e-4 | `Phi(-3.5)` = 2.326291e-4 | c.o.v. 0 | 90 + 14 (FORM) |
| `G = 3 - u2 + 0.15 u1^2`, 200 lines | 9.77749e-4 | quadrature 9.50811e-4 | SE 2.78e-5 (z = 0.97) | 826 |
| safe band (`G = 5 + u2^2` for `abs(u1) >= 1`), 400 lines | 9.58428e-4 | 9.21561e-4 | SE 3.07e-5 | — |

In the band case 116 of 400 lines never cross and contribute exactly 0.

## 3. Multiple design points

`find_design_points` follows the bulging idea of Der Kiureghian and Dakessian
[REF-DERKIUREGHIAN-DAKESSIAN-1998]. After the FORM point `u_1`, HL-RF runs on

`G_b(u) = G(u) + sum_j h_j (1 - ||u - u_j||^2 / r_j^2)^2_+`

with `r_j = 1.1 beta_j` and `h_j = 0.75 G(0)` [EQ-REL-MDP-001]. Each point
found on the bulged surface is refined on the original `G` and accepted only
if it is new. The quartic profile, the scaling by `G(0)`, and the defaults are
this library's choices. The exact profile used in the paper was not checked
(the full text was not available).

HL-RF is local, and the bulge gradient at the origin points along `u_1`. A
single restart from the origin is therefore drawn back towards `u_1` whenever
`grad G(0)` points there. On `G = min(3 - u1, 3.2 - (-0.6 u1 + 0.8 u2))` the
origin-only rule (`starts="origin"`) finds one point and stops. The default
`starts="axes"` also starts from `+-beta_1 e_i` and keeps the nearest new
point. It finds both points (`beta = 3.0, 3.2`) in 504 calls.

`multi_point_form` treats the modes as a series system:
`P_f ~= 1 - Phi_m(beta; R)` with `R_ij = alpha_i . alpha_j` [EQ-REL-MDP-002].
One or two points use closed forms (the second via the existing bivariate
normal CDF). Three or more use Genz's separation of variables with a standard
error [EQ-REL-MVN-001]. Rank-deficient rows (`rho = +-1`) are folded into
two-sided bounds, so antipodal modes cost no sampling noise.
`mixture_importance_sampling` samples `sum_k pi_k N(u_k*, I)` and is unbiased
[EQ-REL-MDP-003].

| problem | exact / reference | FORM (one point) | multi-point FORM | mixture IS |
| --- | --- | --- | --- | --- |
| two planes | 2.037036e-3 (bivariate) | 1.349898e-3 | 2.037036e-3 | 1.9027e-3 +- 8.6e-5 (2000) |
| four-branch, `k = 6` | crude MC 4.4750e-3 +- 1.5e-4 (2e5) | 1.3499e-3 | 5.3923e-3 (4 points, SE 3e-15) | 4.6069e-3 +- 1.4e-4 (4000) |

The four-branch function has four design points, all with `beta = 3` (1160
calls with the axes rule). Multi-point FORM equals the linearized value
`1 - (1 - 2 Phi(-3))^2`, which overestimates because two branches curve away
from the origin. Mixture IS agrees with crude Monte Carlo. The four-branch
`G` has a kink at the origin, where the central-difference gradient vanishes,
so the first FORM run is started at `(0.3, 0.1)`.

## 4. AK-MCS

`ak_mcs` follows Echard, Gayton, and Lemaire
[REF-ECHARD-GAYTON-LEMAIRE-2011]:

1. Draw a Monte Carlo population in `u` space.
2. Fit a GP to `G` on a small space-filling design.
3. Repeatedly add the population point with the smallest
   `U = |mu| / sigma`.
4. Stop when `min U >= 2` [EQ-REL-AKMCS-001].
5. Estimate `P_f = N_f / N` from the predicted signs, and enlarge the
   population while its c.o.v. exceeds the target [EQ-REL-AKMCS-002].

`evaluations` is read from the model's evaluation counter.
`compare_with_monte_carlo=True` evaluates the true `G` on the final population
and reports the same-population crude MC value and the number of
misclassified signs. These calls are counted separately.

Implementation choices that are not part of the published method:

- The GP is a library `matern52` GP in `u` space. Its hyperparameters are
  re-estimated (warm-started) when the training set has grown by a factor of
  `1.5`, and again before learning may stop. Between fits the posterior over
  the population is updated exactly by rank-one Cholesky updates, at `O(n)`
  per point per step.
- The initial design is an LHS in the box `[-5, 5]^d` (`design_bound`).
  Echard et al. draw it from the population. With a design mapped through
  `Phi^-1` (`design_bound=None`), seed 3 stopped at once: 12 points, all
  inside `abs(u) < 2`, gave a confidently wrong GP with `min U >= 2` and no
  predicted failure. It returned `P_f = 0` after enlarging the population to the 200 000 cap, where crude MC on the same points gave 4.175e-3.
  Such a run is now reported as not converged. The GP regularization below
  does not change this run.
- GP hyperparameters are fitted with `GaussianProcessRegularization()` by
  default (ADR-0054, [EQ-SUR-GP-006]). Length scales stay between a tenth of
  the smallest coordinate gap and 20 times the extent of the training set,
  and the diagonal keeps at least `1e-8 sigma_f^2`. `regularization=None`
  restores plain maximum likelihood. Without it, `G = 3 + u1^2/4 - u2` gave
  length scales `1e3`-`2e5`, a signal-to-noise ratio of about `1e18`, and
  `min U` of 114-2320 right after the 12-point design (10 seeds,
  `N = 2000`). With populations up to `5 * 10^4` (`target_cov = 0.1`, seeds
  1-5) those surrogates misclassified 4 points in total although `min U` was
  up to 639. Regularized, the same runs added 3-7 points, misclassified none,
  and stopped at `min U` 2.05-4.6. On the four-branch function
  (`N = 2000`, seeds 1-10), seeds 5 and 9 kept a white-noise optimum
  (length scales `2e-3` and `5e-6`). They ran 198 and 128 additions and
  misclassified 7 and 11 points; seed 9 predicted no failure at all.
  Regularized, all ten runs used 20-31 calls and misclassified nothing.

Four-branch system (`k = 6`), `N = 10^4`, `target_cov` not binding:

| seed | calls | added | `P_f` AK-MCS | `P_f` crude MC, same population | misclassified |
| --- | --- | --- | --- | --- | --- |
| 1 | 35 | 23 | 4.5000e-3 | 4.5000e-3 | 0 |
| 2 | 34 | 22 | 3.7000e-3 | 3.7000e-3 | 0 |
| 3 | 43 | 31 | 5.1000e-3 | 5.1000e-3 | 0 |
| 4 | 37 | 25 | 4.8000e-3 | 4.8000e-3 | 0 |
| 5 | 41 | 29 | 3.8000e-3 | 3.9000e-3 | 1 |

Starting from `N = 5000` with `target_cov = 0.15`, the population grew to
10 000 and the run used 36 calls (`P_f = 4.5e-3`, the same as crude MC on the
same points, c.o.v. 0.149).

Seeds 1-4 and the growing-population run are identical to the unregularized
fit. Seed 5 used 41 calls instead of 48 and misclassified one point
(`min U = 2.19`). The `min U >= 2` rule bounds each point's sign error by
about 2.3%, not the count over `10^4` points.

The kernel matters. With `squared_exponential`, `N = 5000`, and the
unregularized fit, seed 1 exhausted `max_added = 200` without reaching
`min U >= 2`. It ended with 24 misclassified points (`P_f = 1.0e-3` against
5.8e-3), while seed 2 matched crude MC with 37 calls. With the regularized
default, seed 1 needed 34 calls and misclassified nothing. Seed 2 needed 35
calls and misclassified 3 (`P_f = 3.6e-3` against 4.2e-3). Use the Matérn
default.

## Limitations

- Bulging can only find modes that some start reaches. The axes rule costs
  about `2d + 1` HL-RF searches per round, and secondary points inside a
  bulge ball are hidden.
- Separation-of-variables sampling uses no variable reordering. For three
  equicorrelated modes (`rho = 0.5`, `beta = 3`), 4000 samples gave a relative
  SE of 2.5%.
- AK-MCS cost grows with `N` times the training size (posterior updates) plus
  the GP likelihood fits. It is meant for expensive models and populations up
  to about `10^5` in pure Python. Its c.o.v. is the Monte Carlo error of the
  population and excludes misclassification. Very small `P_f` needs importance
  or subset variants of AK-MCS, which are not implemented.

## Sources

[REF-ECHARD-GAYTON-LEMAIRE-2011], [REF-DERKIUREGHIAN-DAKESSIAN-1998],
[REF-KOUTSOURELAKIS-PRADLWARTER-SCHUELLER-2004], [REF-GENZ-BRETZ-2009],
[REF-OWEN-2013], [REF-RASMUSSEN-WILLIAMS-2006].
