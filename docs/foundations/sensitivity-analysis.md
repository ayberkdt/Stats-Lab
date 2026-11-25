# Local and Global Sensitivity Analysis

Sensitivity analysis asks how model output changes when inputs change. Local and
global answers are different and must not be conflated.

## Local derivatives

`central_difference_gradient` approximates each partial derivative at one
declared point with symmetric perturbations [EQ-SENS-LOCAL-001]. It charges two
model evaluations per coordinate. The result depends on point and step size;
large derivatives do not by themselves imply large variance contributions over
an uncertain input distribution.

## Functional ANOVA and Sobol indices

For mutually independent inputs and a square-integrable scalar response,
functional ANOVA decomposes output variance into orthogonal main and interaction
components [EQ-SENS-ANOVA-001]. First-order indices measure main effects,
second-order indices isolate pair interactions, and total-effect indices include
every term involving an input [EQ-SENS-FIRST-001], [EQ-SENS-SECOND-001],
[EQ-SENS-TOTAL-001].

`estimate_sobol_indices` takes two explicitly independent base matrices and
constructs pick-freeze hybrids. It reports all model calls. Monte Carlo estimates
are intentionally not clipped to `[0,1]`: small negative or above-one estimates
are finite-sample diagnostics, while large violations signal inadequate sample
size, dependence, numerical noise, or a broken sampling design.

The current estimator assumes independent inputs. For dependent inputs use the
Shapley effects described below. Replicated stochastic simulators are
handled by `stochastic_sobol_indices`, which evaluates every design point
several times with independent seeds. Streaming evaluation, with indices
updated as batches arrive, is not implemented.

## Choosing a Sobol estimator

A pick-freeze design produces three blocks of outputs on `N` rows:
`y_A = f(A)`, `y_B = f(B)`, and `y_i = f(A_B^(i))`, where `A_B^(i)` is `A`
with column `i` taken from `B`. The pair `(y_B, y_i)` shares only `x_i` and
carries the first-order index; the pair `(y_A, y_i)` shares every input
except `x_i` and carries the total index. Every estimator below reads the
same `N (d + 2)` runs, so the choice costs no model evaluations and several
estimators can be compared on one design. `estimate_sobol_indices`,
`sobol_analysis`, and `sobol_analysis_all` take `estimator=`, one of
`SOBOL_ESTIMATORS`; bootstrap and replicate intervals work for all of them.

With `V_A` the sample variance of `y_A` (denominator `N - 1`):

| `estimator` | first order `S_i` | total `S_Ti` | equation |
| --- | --- | --- | --- |
| `"saltelli2010"` (default) | `mean((y_B - m)(y_i - y_A)) / V`, `m`, `V` from the pooled `y_A, y_B` | `mean((y_A - y_i)^2) / (2 V)` | [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001] |
| `"saltelli2002"` | `[sum y_B y_i / (N-1) - mean(y_A y_B)] / V_A` | `1 - [sum y_A y_i / (N-1) - mean(y_A)^2] / V_A` | [EQ-SENS-EST-001] |
| `"sobol2007"` | `sum y_B (y_i - y_A) / ((N-1) V_A)` | `sum y_A (y_A - y_i) / ((N-1) V_A)` | [EQ-SENS-EST-002] |
| `"jansen"` | `1 - sum (y_B - y_i)^2 / ((2N-1) V_A)` | `sum (y_A - y_i)^2 / ((2N-1) V_A)` | [EQ-SENS-EST-003] |
| `"martinez"` | `corr(y_B, y_i)` | `1 - corr(y_A, y_i)` | [EQ-SENS-EST-004] |
| `"janon"` | `J(y_B, y_i)` | `1 - J(y_A, y_i)` | [EQ-SENS-EST-006] |

Here `J(y, y') = [mean(y y') - c^2] / [mean((y^2 + y'^2) / 2) - c^2]` with
`c = mean((y + y') / 2)`. The non-default formulas follow the functions
`sobol2002`, `sobol2007`, `soboljansen`, `sobolmartinez`, and `sobolEff` of
the R package `sensitivity` [REF-R-SENSITIVITY], including their `N - 1` and
`2N - 1` divisors, so results can be checked against analyses done in R.

When each one helps:

- `"saltelli2010"` centres `y_B` on the pooled mean, which removes the
  dependence of the error on the output mean. It is the default and keeps
  the numbers of earlier releases.
- `"saltelli2002"` [REF-SALTELLI-2002] and `"sobol2007"`
  [REF-SOBOL-ETAL-2007] multiply raw outputs. Their error grows with the
  squared ratio of the output mean to its standard deviation; adding
  `10^4` to Ishigami moves their first-order estimates by more than `0.05`
  on 512 rows while the four other estimators do not move. Use them to
  reproduce a published R analysis, not as a first choice.
- `"jansen"` [REF-JANSEN-1999] uses squared differences and is a good total
  estimator. Its first-order estimate subtracts a residual from the
  variance, so its error does not shrink with the index; it had the widest
  first-order intervals in the comparison below.
- `"martinez"` [REF-BAUDIN-ETAL-2016] writes both indices as correlations.
  It is invariant to the output's location and scale and accurate for small
  indices. With `asymptotic=True` on a `"mc"` design it reports Fisher-z
  intervals [EQ-SENS-EST-005] without resampling; they are exact only for a
  Gaussian output.
- `"janon"` [REF-JANON-ETAL-2014] has the smallest asymptotic variance among
  regular first-order estimators built from the pair. The total index is
  one minus the same estimator applied to the complementary pair.
  `asymptotic=True` reports intervals from its central limit theorem
  [EQ-SENS-EST-007].

On the Ishigami function (`a = 7`, `b = 0.1`; analytic
`S = (0.3139, 0.4424, 0)` and `S_T = (0.5576, 0.4424, 0.2437)`), the same
runs give first-order estimates of `x1` from 0.290 to 0.351 depending on the
formula, while the Martinez and Janon rows agree to 0.001. One Monte Carlo
design of `N = 4096` rows with seed 2026 (20480 runs shared by all six
estimators):

| `estimator` | `S_1` | `S_2` | `S_3` | `S_T1` | `S_T2` | `S_T3` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `"saltelli2010"` | 0.331 | 0.461 | -0.016 | 0.570 | 0.462 | 0.251 |
| `"saltelli2002"` | 0.351 | 0.487 | -0.025 | 0.533 | 0.436 | 0.247 |
| `"sobol2007"` | 0.351 | 0.487 | -0.025 | 0.533 | 0.436 | 0.247 |
| `"jansen"` | 0.290 | 0.437 | -0.052 | 0.570 | 0.463 | 0.251 |
| `"martinez"` | 0.304 | 0.438 | -0.037 | 0.560 | 0.462 | 0.248 |
| `"janon"` | 0.303 | 0.438 | -0.037 | 0.560 | 0.462 | 0.248 |

The spread between rows is the Monte Carlo error of each formula on the same
runs. On a scrambled Sobol design of the same size split into 8 independent
replicates (seed 2026), the 95% half-width of `S_1` was 0.008 for
`"martinez"` and `"janon"`, 0.011 for `"saltelli2002"` and `"sobol2007"`,
0.013 for `"saltelli2010"`, and 0.019 for `"jansen"`; for `S_T3` the
half-widths ranged from 0.0035 (`"jansen"`) to 0.0059 (`"saltelli2010"`).
The Fisher-z interval of `S_3` on the Monte Carlo design is
`[-0.067, -0.006]`, which misses the true zero: the Ishigami output is far
from Gaussian, so prefer replicate intervals when that assumption is doubtful.

```python
from probstat.modeling import ishigami_model
from probstat.sensitivity import SOBOL_ESTIMATORS, sobol_analysis

ishigami = ishigami_model()
by_estimator = {
    name: sobol_analysis(ishigami, "y", 4096, design="mc", seed=2026, estimator=name)
    for name in SOBOL_ESTIMATORS
}
by_estimator["saltelli2010"].result.first_order   # (0.331, 0.461, -0.016)
by_estimator["martinez"].result.first_order   # (0.304, 0.438, -0.037)
by_estimator["martinez"].evaluations   # 20480
fisher = sobol_analysis(
    ishigami, "y", 4096, design="mc", seed=2026, estimator="martinez", asymptotic=True
)
(fisher.first_order[0].lower, fisher.first_order[0].upper)   # (0.276, 0.331)
fisher.interval_method   # 'asymptotic'
```

## Quadratic sum: Taylor importance factors

Linearizing the model around a representative point `x0`,
`g(X) ~= g(x0) + sum_i dg/dx_i(x0) (X_i - x0_i)`, gives for independent
inputs the first-order Taylor variance `V_T = sum_i (dg/dx_i(x0))^2 sigma_i^2`
and the importance factors

`eta_i^2 = (dg/dx_i(x0))^2 sigma_i^2 / V_T`   [EQ-SENS-TAYLOR-001].

`taylor_importance_factors(model, quantity)` expands around the input means
with the standard deviations of the input law; `point=` and
`standard_deviations=` accept vectors or mappings by input name.
Derivatives come from central differences with step `h_i = c |x0_i|`
(`relative_step=c`, default `0.01`, that is a 1% perturbation; `h_i = c`
where `x0_i = 0`), at a cost of `2d` runs, or from `gradient=`, either the
values at `x0` or a callable. `quadratic_sum_importance` does the same for a
plain function of a tuple.

The factors are local. They see only the slope at `x0` and are faithful
when the model is nearly linear over the range in which the inputs vary.
Sobol indices average over the whole input law; a gap between `eta_i^2` and
`S_i` measures the curvature and the interactions the linearization misses.

For the flood model the overflow is `S = Zv + H - Hd - Cb` with
`H = (Q / (Ks B sqrt((Zm - Zv) / L)))^0.6` [EQ-MOD-FLOOD-001]. Its exact
derivatives are `dS/dQ = 0.6 H / Q`, `dS/dKs = -0.6 H / Ks`,
`dS/dZv = 1 + 0.3 H / (Zm - Zv)`, `dS/dZm = -0.3 H / (Zm - Zv)`,
`dS/dHd = dS/dCb = -1`, `dS/dL = 0.3 H / L`, and `dS/dB = -0.6 H / B`. At
`FLOOD_NOMINAL` (`H = 2.525` m) with `FLOOD_NOMINAL_STANDARD_DEVIATIONS`,
the 1% finite differences reproduce the exact importance factors to within
`2e-4`, and both rank the inputs as the Sobol indices do (rows in
decreasing `eta^2`):

| input | `dS/dx` exact | 1% FD relative error | `eta^2` (1% FD) | `eta^2` (exact) | `eta^2` (law means) | Sobol `S_i` | Sobol `S_Ti` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Q` | 1.118e-3 | 9.3e-6 | 0.3482 | 0.3483 | 0.3470 | 0.3440 | 0.3523 |
| `Hd` | -1.000e+0 | 2.1e-14 | 0.2962 | 0.2963 | 0.2960 | 0.2844 | 0.2843 |
| `Zv` | 1.152e+0 | 6.6e-4 | 0.1984 | 0.1982 | 0.1964 | 0.1897 | 0.1902 |
| `Ks` | -4.968e-2 | 6.9e-5 | 0.1163 | 0.1163 | 0.1201 | 0.1338 | 0.1420 |
| `Cb` | -1.000e+0 | 5.6e-16 | 0.0373 | 0.0374 | 0.0370 | 0.0361 | 0.0355 |
| `Zm` | -1.515e-1 | 6.1e-3 | 0.0035 | 0.0034 | 0.0034 | 0.0030 | 0.0038 |
| `B` | -5.051e-3 | 6.9e-5 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| `L` | 1.515e-4 | 2.0e-5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

The Sobol columns come from `sobol_analysis(flood_model(), "overflow", 4096,
seed=2026, replicates=8)` on the full input law; their 95% half-widths are
at most 0.003. Three points stand out:

- **Finite differences.** In `Q`, `Ks`, `L`, and `B` the overflow is a power
  law `x^p`, for which the 1% central quotient equals the derivative times
  `((1.01)^p - (0.99)^p) / (0.02 p)`: a relative error of `9.3e-6` for
  `p = 0.6` and `6.9e-5` for `p = -0.6`. The largest error, 0.6% for `Zm`,
  arises because 1% of `Zm = 55` m is a step of 0.55 m against a gap
  `Zm - Zv` of only 5 m. Even there the factor moves by less than `1e-4`, so a
  1% step is adequate for this model.
- **Agreement with Sobol.** The overflow is close to linear over the input
  ranges: the factors rank the inputs as the Sobol indices do
  (`Q`, `Hd`, `Zv`, `Ks`, `Cb`, then the negligible `Zm`, `B`, `L`), the
  first-order Sobol indices sum to 0.991, and `V_T = 1.125` against a
  sampled variance of 1.172.
- **Where linearization falls short.** `Ks` has a coefficient of variation
  of about 24% and enters through `Ks^-0.6`, a convex function: its Sobol
  index (0.134, total 0.142) exceeds its Taylor factor (0.116 to 0.120).
  `V_T` also falls about 4% short of the sampled variance; a first-order
  expansion has no term for such curvature.

The locality is plain on Ishigami. At the input means `(0, 0, 0)` the
derivatives are `(1, 0, 0)`, so the quadratic sum assigns all variance to
`x1`, whereas `x2` has the largest first-order Sobol index (0.4424) and `x3`
a total index of 0.2437. Expanding at a single point cannot see a response
whose slope vanishes there or changes sign across the input range.

```python
import math

from probstat.modeling import (
    FLOOD_NOMINAL,
    FLOOD_NOMINAL_STANDARD_DEVIATIONS,
    flood_model,
    ishigami_model,
)
from probstat.sensitivity import taylor_importance_factors

def overflow_gradient(x):
    q, ks, zv, zm, hd, cb, length, width = x
    height = (q / (ks * width * math.sqrt((zm - zv) / length))) ** 0.6
    gap = zm - zv
    return (0.6 * height / q, -0.6 * height / ks, 1 + 0.3 * height / gap,
            -0.3 * height / gap, -1.0, -1.0, 0.3 * height / length, -0.6 * height / width)

flood = flood_model()
nominal = {"point": FLOOD_NOMINAL, "standard_deviations": FLOOD_NOMINAL_STANDARD_DEVIATIONS}
by_differences = taylor_importance_factors(flood, "overflow", **nominal)
by_differences.ranking()   # ('Q', 'Hd', 'Zv', 'Ks', 'Cb', 'Zm', 'B', 'L')
by_differences.importance_factors[:3]   # (0.348, 0.116, 0.198)
(by_differences.taylor_variance, by_differences.evaluations)   # (1.125, 16)
exact = taylor_importance_factors(flood, "overflow", gradient=overflow_gradient, **nominal)
exact.gradient[3], by_differences.gradient[3]   # (-0.1515, -0.1524)
taylor_importance_factors(ishigami_model(), "y").importance_factors   # (1.0, 0.0, 0.0)
```

## Ranking and decisions

Inputs are ranked by total effect because it includes interactions. The declared
threshold rule [EQ-SENS-DECISION-001] separates uncertainty-reduction priorities
from candidates for fixing during dimension reduction. This is a screening
recommendation, not a causal statement or cost-benefit optimum; physical,
epistemic, safety, and measurement-cost information must still be considered.

## Reproducible laboratory

Run `python examples/sensitivity_lab.py`. The Ishigami benchmark contains a
strong interaction: its third input has essentially no first-order effect but a
substantial total effect.

`python examples/flood_sensitivity_lab.py` applies every measure of this
guide to the flood model: Taylor importance factors with finite differences
checked against exact derivatives, Morris screening, regression and rank
regression, Sobol indices with four estimators, and HSIC, including target
HSIC of the top overflow decile, for both the overflow and the cost.

<picture><source media="(prefers-color-scheme: dark)" srcset="../images/guides/ishigami-sobol-indices-dark.png"><img alt="Dumbbell chart of first-order and total Sobol indices of the three Ishigami inputs on a common zero to one scale" src="../images/guides/ishigami-sobol-indices.png"></picture>

*With two 50,000-row pick-freeze designs, x3 has a first-order index of 0.01 but a total index of 0.25 (analytic values 0 and 0.2437): ranking by main effects alone would mark it for fixing, while the total effect, which includes its interaction with x1, keeps it.*

## Sources

- [REF-SOBOL-1993].
- [REF-SALTELLI-2008, Chapters 1, 4, and 6].
- [REF-SALTELLI-2010].
- [REF-FORNBERG-1988].

## Shapley effects for dependent inputs

Sobol indices need independent inputs: under correlation the functional ANOVA
terms are no longer orthogonal and first-order and total indices stop adding
up. Shapley effects share the output variance among inputs by the Shapley
value of the variance game [EQ-SENS-SHAPLEY-001]:

`Sh_i = (1/Var Y) sum_(J not containing i) [|J|! (d-|J|-1)! / d!] (c(J u {i}) - c(J))`.

They are non-negative in the population and sum to one under any joint law.
For independent inputs without interactions they equal the first-order and
total Sobol indices; with interactions each interaction share is split equally
among its inputs [REF-OWEN-2014].

`shapley_effects(model, quantity)` follows Song, Nelson, and Staum
[EQ-SENS-SHAPLEY-002]. The cost `c(J) = E[Var(Y | X_-J)]` (equivalent to the
closed index `Var(E[Y | X_J])`) is estimated with `N_O` outer draws of `X_-J`
and `N_I = 3` inner draws of `X_J | X_-J`. `Var Y` comes from `N_V` separate
joint draws, so the effects telescope to exactly one.

- `method="exact"` (default for `d <= 6`) estimates each of the `2^d - 2`
  proper subsets once and uses the subset formula; cost
  `N_V + (2^d - 2) N_O N_I`.
- `method="random"` averages `m` random permutations with `N_O = 1` by
  default; cost `N_V + m (d - 1) N_O N_I`.

Conditional sampling uses the model's joint law. For Gaussian-copula inputs
`z_J | z_K` is Gaussian with mean `R_JK R_KK^-1 z_K` and Schur-complement
covariance `R_JJ - R_JK R_KK^-1 R_KJ`, and `x_i = F_i^-1(Phi(z_i))`
[EQ-SENS-SHAPLEY-003]. The standard errors are delta-method approximations
that ignore the noise of the normalizing `Var Y`.

`linear_gaussian_shapley(b, Sigma)` gives exact effects of `Y = b'X` for
Gaussian `X` from `Var(E[Y | X_J]) = s_J' Sigma_JJ^-1 s_J`, `s = Sigma b`
[EQ-SENS-SHAPLEY-004]. In two dimensions:

`Sh_1 = (b1^2 s1^2 (1 - rho^2/2) + rho b1 b2 s1 s2 + b2^2 s2^2 rho^2/2) / V`.

Expanding `Sh_1 = (1/2)[c({1}) + V - c({2})]` with
`c({1}) = (b1 s1 + rho b2 s2)^2` and `c({2}) = (b2 s2 + rho b1 s1)^2` gives
this formula. `tests/test_mlmc_ce_line_shapley.py` checks it against
`linear_gaussian_shapley` on three parameter sets to twelve decimal places.

### Numbers computed with this release

`Y = X1 + X2` with `X1 ~ N(0, 1)`, `X2 ~ N(0, 2^2)`, and correlation `0.6`
(`Var Y = 7.4`):

| quantity | `X1` | `X2` |
| --- | --- | --- |
| closed first-order index `Var(E[Y | X_i]) / V` | 0.654 | 0.914 |
| total index `1 - Var(E[Y | X_-i]) / V` | 0.086 | 0.346 |
| exact Shapley effect | 0.3703 | 0.6297 |
| `shapley_effects`, exact, `N_O = 500`, `N_V = 2000`, seed 1 (5000 calls) | 0.3755 +/- 0.0084 | 0.6245 +/- 0.0084 |
| `shapley_effects`, 400 random permutations, `N_V = 1000`, seed 1 (2200 calls) | 0.3831 +/- 0.0198 | 0.6169 +/- 0.0198 |

The first-order indices sum to 1.57 and the total indices to 0.43. Only the
Shapley effects give a variance budget that sums to one. The estimated rows
are reproduced by:

```python
from probstat.distributions import Normal
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.sensitivity import linear_gaussian_shapley, shapley_effects
from probstat.uq import QuantityOfInterest

sum_qoi = (QuantityOfInterest("y", "sum"),)
pair_inputs = JointInput(
    (RandomInput("x1", Normal(0.0, 1.0)), RandomInput("x2", Normal(0.0, 2.0))),
    ((1.0, 0.6), (0.6, 1.0)),
)
pair = ProbabilisticModel(pair_inputs, lambda p: {"y": p["x1"] + p["x2"]}, sum_qoi)
analytic = linear_gaussian_shapley((1.0, 1.0), ((1.0, 1.2), (1.2, 4.0)))
round(analytic[0], 4)   # 0.3703
subsets = shapley_effects(pair, "y", outer_samples=500, variance_samples=2000, seed=1)
round(subsets.effects[0], 4), round(subsets.standard_errors[0], 4)   # (0.3755, 0.0084)
subsets.evaluations   # 5000
permuted = shapley_effects(pair, "y", method="random", permutations=400, seed=1)
round(permuted.effects[0], 4), round(permuted.standard_errors[0], 4)   # (0.3831, 0.0198)
permuted.evaluations   # 2200
```

For the independent additive model `Y = X0 + 2 X1 + 3 X2` (standard normal
inputs) the exact values are `(1, 4, 9)/14 = (0.071, 0.286, 0.643)`. With
`N_O = 400` and seed 4, `shapley_effects` returned `(0.088, 0.263, 0.649)`,
with standard errors of about `0.02` (0.022, 0.018 and 0.021), in 8200 calls.

Sources: [REF-OWEN-2014], [REF-SONG-NELSON-STAUM-2016],
[REF-OWEN-PRIEUR-2017], [REF-IOOSS-PRIEUR-2019].

## Sampling-based measures: regression coefficients and HSIC

A forward propagation already provides an i.i.d. sample `(x_k, y_k)` of
inputs and outputs. The measures below are computed from that sample alone,
without the structured designs of Sobol or Morris, and accept dependent
inputs (they then describe total association, not causal shares).

**Look first.** `visualization.plot_input_scatter` draws the output against
each input, one panel per input with a shared output axis and a lowess
smoother (local linear fit on two thirds of the points). A flat smoother
marks an input without first-order effect; a curved one, an effect that a
linear index will understate. `visualization.plot_cobweb` draws every run as
a polyline across the inputs (scaled ranks in `[0, 1]`) and colours the runs
whose output exceeds a threshold over grey context: a coloured bundle that
crosses one end of an axis names an input that drives the exceedance, and
crossing patterns between neighbouring axes hint at interactions, which the
marginal scatterplots cannot show.

**Linear regression.** `sensitivity.regression_sensitivity` fits
`y = b_0 + sum_i b_i x_i + e` by least squares and reports:

- the standardized regression coefficient `SRC_i = b_i s_i / s_y`
  ([EQ-SENS-SRC-001]); for a linear model with independent inputs,
  `SRC_i^2 = b_i^2 Var(X_i) / Var(Y)` is the share of variance due to `X_i`
  and `sum_i SRC_i^2 = R^2 = 1`;
- the partial correlation coefficient `PCC_i = t_i / sqrt(t_i^2 + n - d - 1)`,
  the correlation of `y` and `x_i` once both are cleared of the other inputs
  ([EQ-SENS-PCC-001]). It ranks inputs but is not a variance share: an input
  with a small spread and a precise effect has a small `SRC` and a large
  `PCC`;
- with `rank=True`, the same quantities on the mid-ranks of every column
  (SRRC, PRCC, rank `R^2`, [EQ-SENS-RANK-001]), which measure monotone
  rather than linear association;
- the Student test of each slope (`H0: b_i^2 Var(X_i) = 0`) and the overall
  Fisher test, both exact only for normal errors of constant variance.

Two numbers decide whether the indices can be read as shares. `R^2` is the
fraction of output variance that the linear fit explains: the result adds a
note when `R^2 < 0.7`, because nonlinear effects and interactions are then
missing from the indices. `sum_i SRC_i^2` differs from `R^2` by
`2 sum_{i<j} SRC_i SRC_j r_ij`, where `r_ij` are the sample correlations of
the inputs: with independent inputs this term is noise of order
`1/sqrt(n)`, with correlated inputs the squared coefficients overlap.
A rank `R^2` well above `R^2` signals a monotone nonlinear response, for
which SRRC and PRCC are the better summaries.

**HSIC.** `sensitivity.hsic_indices` estimates the Hilbert-Schmidt
independence criterion between each input and the output with Gaussian
kernels `exp(-(x - x')^2 / (2 h^2))`; the default bandwidth `h` is the
sample standard deviation, and `bandwidth="median"` uses the median
pairwise distance instead. HSIC is zero if and only if `X_i` and `Y` are
independent, so it detects nonmonotone effects: for `Y = X^2` with a
symmetric `X` the correlation is zero but HSIC is large. The biased estimate
is `tr(K H L H) / n^2` ([EQ-SENS-HSIC-001]); `estimator="unbiased"` gives the
U-statistic ([EQ-SENS-HSIC-002]), centred at zero under independence. Raw
HSIC values depend on the kernels; compare inputs through
`R2_HSIC = HSIC(X, Y) / sqrt(HSIC(X, X) HSIC(Y, Y))` and through the
p-values of `H0: X_i independent of Y`, obtained either by permuting the
output ([EQ-SENS-HSIC-004], exact level, `O(n^2)` per permutation and input,
about 6 s per input for 200 permutations at `n = 1000`) or by the Gamma
approximation to the null law of `n HSIC_b` ([EQ-SENS-HSIC-003], about 0.7 s
per input at `n = 1000`). `target=c` replaces `Y` by `1{Y > c}`
([EQ-SENS-HSIC-005]) and measures the influence of each input on the
occurrence of the critical event rather than on the whole output
distribution. `model_regression_sensitivity` and `model_hsic_indices` draw
the sample from a `ProbabilisticModel` first (plain Monte Carlo by default,
which keeps the rows i.i.d. as both tests assume).

### Flood model: numbers computed with this release

`modeling.flood_model()`, 1000 plain Monte Carlo runs, seed 2026 (the
regression and the HSIC wrappers draw their samples from separate child
streams of the seed). Overflow `S` (m) and maintenance cost `C` (million
euros); HSIC uses the biased estimator and the standard-deviation
bandwidth, permutation p-values use 200 permutations (smallest attainable
value `1/201 = 0.005`).

| input | `S`: `SRC^2` | `S`: `R2_HSIC` | `S`: HSIC p (Gamma / perm.) | `C`: `SRC^2` | `C`: `R2_HSIC` | `C`: HSIC p (Gamma / perm.) |
| --- | --- | --- | --- | --- | --- | --- |
| `Q` | 0.330 | 0.200 | 5e-72 / 0.005 | 0.324 | 0.226 | 9e-63 / 0.005 |
| `Ks` | 0.127 | 0.057 | 1e-25 / 0.005 | 0.151 | 0.082 | 1e-35 / 0.005 |
| `Zv` | 0.198 | 0.091 | 8e-40 / 0.005 | 0.173 | 0.102 | 6e-42 / 0.005 |
| `Zm` | 0.003 | 0.003 | 0.23 / 0.19 | 0.007 | 0.002 | 0.50 / 0.41 |
| `Hd` | 0.291 | 0.236 | 5e-95 / 0.005 | 0.005 | 0.128 | 1e-54 / 0.005 |
| `Cb` | 0.035 | 0.023 | 4e-10 / 0.005 | 0.029 | 0.015 | 5e-06 / 0.005 |
| `L` | 0.000 | 0.001 | 0.78 / 0.83 | 0.000 | 0.001 | 0.71 / 0.68 |
| `B` | 0.000 | 0.003 | 0.38 / 0.29 | 0.001 | 0.001 | 0.90 / 0.92 |

Fit summaries: for `S`, `R^2 = 0.980`, `sum SRC^2 = 0.984` and rank
`R^2 = 0.948`; for `C`, `R^2 = 0.687`, `sum SRC^2 = 0.690` and rank
`R^2 = 0.725`. Base R `lm` on the same `S` and `C` samples gives the same
`R^2`, SRC and partial correlations (checked to nine decimal places in
`tests/test_regression_and_hsic_sensitivity.py`).

Reading the table:

- The overflow is close to linear in its inputs over their ranges, so the
  squared coefficients are variance shares: `Q`, `Hd`, `Zv` and `Ks` carry
  about 95 % of the variance, `L` and `B` almost none.
- The cost fit explains only 69 % of the variance and the result carries the
  low-`R^2` note. The dike height shows why: its `SRC^2` is 0.005, yet it is
  the second input by `R2_HSIC` (0.128). Over the range of `Hd` the cost
  first falls and then rises, a V shape that a straight line averages away
  and that the scatterplot of `C` against `Hd` shows directly.
- `Zm` illustrates the opposite limit. Its effect on `S` is linear and
  small (`SRC^2 = 0.003`); the Student test detects it (`p = 2e-35`)
  because the residual variance of the near-perfect fit is tiny, while the
  marginal HSIC test, which does not remove the effect of the other inputs,
  does not (`p = 0.19`).
- Goal-oriented HSIC of the event `S > -9.64` (the 0.9-quantile of the HSIC
  sample, 10 % of the runs) keeps the same leading inputs but changes their
  weights: `R2_HSIC` is 0.143 for `Q` and 0.048, 0.048 and 0.040 for `Ks`,
  `Hd` and `Zv`, with Gamma p-values below `1e-8`; `L` and `B` stay at
  `p > 0.79`.

```python
from probstat.modeling import flood_model
from probstat.sensitivity import model_hsic_indices, model_regression_sensitivity

overflow = model_regression_sensitivity(flood_model(), "overflow", 1000, seed=2026, rank=True)
round(overflow.r_squared, 3)   # 0.98
round(overflow.sum_src_squared, 3)   # 0.984
overflow.ranking()   # ('Q', 'Hd', 'Zv', 'Ks', 'Cb', 'Zm', 'B', 'L')
[round(value, 3) for value in overflow.src_squared]   # [0.33, 0.127, 0.198, 0.003, 0.291, 0.035, 0.0, 0.0]
cost = model_regression_sensitivity(flood_model(), "cost", 1000, seed=2026)
round(cost.r_squared, 3)   # 0.687
round(cost.src_squared[4], 4)   # 0.0052
len(cost.notes)   # 1
hsic = model_hsic_indices(flood_model(), "cost", 1000, seed=2026, test="gamma")
hsic.ranking()[:4]   # ('Q', 'Hd', 'Zv', 'Ks')
round(hsic.r2_hsic[4], 3)   # 0.128
hsic.independent(0.05)   # ('Zm', 'L', 'B')
```

The two figures below come from `examples/flood_sensitivity_lab.py`, which
draws one plain Monte Carlo sample of 1000 runs (seed 2026) and evaluates
both outputs on it, so its regression and HSIC values differ from the table
above, which uses the separate samples of the two wrappers, by the sampling
error of 1000 runs.

<picture><source media="(prefers-color-scheme: dark)" srcset="../images/guides/sensitivity-flood-top-decile-dark.png"><img alt="Eight scatterplots of the flood-model overflow against each input for 1,000 runs, with the top decile of the overflow highlighted" src="../images/guides/sensitivity-flood-top-decile.png"></picture>

*The 100 runs above the 0.9-quantile of the overflow, S > -9.48 m, are coloured. Their median rank departs from the middle by more than 0.15 for four inputs, and the title names them: they come from low Ks, high Q, low Hd and high Zv. On `Zm`, `Cb`, `L` and `B` the coloured runs spread over the whole range, in line with the target HSIC p-values of these inputs. The laboratory also draws the same runs as a cobweb plot (`visualization.plot_cobweb`), where the coloured bundle crosses the low end of the `Ks` and `Hd` axes.*

<picture><source media="(prefers-color-scheme: dark)" srcset="../images/guides/sensitivity-flood-cost-measures-dark.png"><img alt="Four panels of Taylor importance factors, squared regression coefficients, first-order Sobol indices, and HSIC indices of the flood-model cost for eight inputs" src="../images/guides/sensitivity-flood-cost-measures.png"></picture>

*Four measures of the cost `C`, with the inputs ordered by their total Sobol index. The linear fit explains R^2 = 0.73 of the variance and gives the dike height `Hd` an `SRC^2` of only 0.006, while the first-order Sobol index (0.12) and `R2_HSIC` (0.11) keep it among the four leading inputs. A low dike lets the river overflow and a high dike costs more to maintain, so the cost first falls and then rises with `Hd`; a straight line through that V is nearly flat. The quadratic sum gives 0.04 because its expansion point, the nominal `Hd = 8` m, lies near the bottom of the V, where the slope is small. Morris screening flags the same effect through a signed mean `mu` far below `mu*`. For the overflow, which is nearly linear, the laboratory's same four panels agree.*

Sources: [REF-SALTELLI-2008, Chapter 1], [REF-HELTON-ETAL-2006],
[REF-IMAN-CONOVER-1979], [REF-GRETTON-ETAL-2005], [REF-GRETTON-ETAL-2008],
[REF-SONG-ETAL-2012], [REF-DA-VEIGA-2015], [REF-MARREL-CHABRIDON-2021],
[REF-INSELBERG-1985], [REF-CLEVELAND-1979].

## Rank statistics, distribution-based indices, kernels, and active subspaces

The measures of this section answer questions that a variance share does not
(ADR-0123). The first four need only an existing i.i.d. sample or one
pick-freeze design; the last one needs gradients.

**Rank-based first-order Sobol indices.** Sort the sample by `x_i` and read
the output in that order, `y_(1), ..., y_(n)`. Neighbouring rows have almost
the same `x_i` and unrelated values of the other inputs, so they act as the
pick-freeze pair of `E[E[Y | X_i]^2]`, and

`S_i = [(1/n) sum_j y_(j) y_(j+1) - ybar^2] / [(1/n) sum_j y_j^2 - ybar^2]`

with `y_(n+1) = y_(1)` ([EQ-SENS-RANKSOBOL-001]; Gamboa, Gremaud, Klein and
Lagnoux 2022). `sensitivity.rank_sobol_indices` returns all `d` indices from
the same `n` runs, without a class count or a bandwidth to choose;
`sensitivity.model_rank_sobol_indices` draws the sample from a model. On the
Ishigami function sixteen replicates give a root-mean-square error of 0.046
at `n = 500` and 0.017 at `n = 4000`, the `sqrt(n)` rate. Do not bootstrap
this estimator: a resample repeats rows, repeated rows are neighbours after
sorting, and their equal outputs inflate the lag-one product (the index of
the inert Ishigami input rises from about 0 to above 0.25 on a resampled
table). `blocks=B` instead recomputes the estimator on `B` disjoint blocks
of rows and reports `sd / sqrt(B)` ([EQ-SENS-RANKSOBOL-002]); over 24
replicates at `n = 2000` this standard error averaged 0.020, 0.022 and 0.030
against replicate standard deviations of 0.021, 0.019 and 0.028. It is a
diagnostic with `B - 1` degrees of freedom; the asymptotic variance of the
published central limit theorem is not implemented.

**Chatterjee's rank correlation.** `sensitivity.chatterjee_indices` applies
the same neighbour idea to the ranks `r_j` of the output:
`xi_n = 1 - 3 sum_j |r_(j+1) - r_(j)| / (n^2 - 1)` ([EQ-SENS-XI-001], with
the tie-aware form [EQ-SENS-XI-002]). Its limit is 0 if and only if `X_i`
and `Y` are independent and 1 if and only if `Y` is a function of `X_i`; a
strictly monotone relation gives exactly `1 - 3 / (n + 1)`. Under
independence and a continuous output `sqrt(n) xi_n -> N(0, 2/5)`
([EQ-SENS-XI-003]), which gives a p-value without resampling;
`test="permutation"` handles outputs with ties. The coefficient measures
how well `Y` is a *function* of `X_i`, so it is small for an input that acts
only through interactions, and its test has little power against weak
smooth effects.

**Moment-independent indices.** Borgonovo's
`delta_i = (1/2) E[int |f_Y(y) - f_{Y | X_i}(y)| dy]` is the expected
total-variation distance between the conditional and the unconditional
output density; PAWN replaces it by the Kolmogorov-Smirnov distance between
the two distribution functions, summarized over the conditioning values by
the median, mean or maximum. `sensitivity.borgonovo_delta` and
`sensitivity.pawn_indices` cut the sample into equal-count classes of `x_i`
and compare each class with the whole sample ([EQ-SENS-DELTA-001],
[EQ-SENS-PAWN-001]). With the default histogram on bins of equal pooled
frequency, and for PAWN always, the result depends on the output only
through its order, so a strictly increasing transformation of `Y` changes
nothing. Both estimators are positive for an input without any effect,
because two estimated distributions never coincide. Set `permutations` to
obtain the `null_level` (the mean of the index after random reassignment of
the outputs to the classes, [EQ-SENS-DELTA-002]), a permutation p-value, and
`corrected = index - null_level`. The correction is a heuristic: for
`Y = 2 X_1 + X_2` with standard normal inputs the exact values are
`delta = (0.5055, 0.1604)`; at `n = 4000` the raw histogram estimates are
`(0.482, 0.180)` and the corrected ones `(0.419, 0.117)`, with an inert
third input at `0.059` against a null level of `0.063`. Class averaging
biases the estimate of a strong input downwards, the noise floor biases
that of a weak input upwards.

**Kernel indices.** A kernel `k` on the output turns the comparison of
distributions into a comparison of mean embeddings. With `Y^i` sharing only
`X_i` with `Y` and `Y'` an independent copy,

`S_i^k = (E[k(Y, Y^i)] - E[k(Y, Y')]) / (E[k(Y, Y)] - E[k(Y, Y')])`

is the expected squared maximum mean discrepancy between `P_{Y | X_i}` and
`P_Y`, normalized ([EQ-SENS-KERNEL-001]; da Veiga 2021).
`sensitivity.kernel_sobol_indices` estimates it from the pick-freeze outputs
`y_B` and `y_i = f(A_B^(i))` with `N (d + 1)` runs ([EQ-SENS-KERNEL-002]).
`kernel="linear"` reproduces the first-order Sobol estimate of
`estimate_sobol_indices(..., estimator="janon")` to rounding error (a
difference of `2e-16` on 300 Ishigami rows), which is the check that the
estimator is the right generalization. The default Gaussian kernel, with the
median pairwise distance as bandwidth, reacts to any change of the output
distribution: for `Y = X_1 X_2` with standard normal inputs the first-order
Sobol indices are zero while the Gaussian-kernel indices are about 0.13.
Kernel indices depend on the bandwidth and are not variance shares.

**Active subspaces.** With gradients available, the matrix
`C = E[grad f grad f'] = W Lambda W'` ([EQ-SENS-ACTIVE-001]; Constantine,
Dow and Wang 2014) shows along which *directions*, not only along which
coordinates, the function changes: `lambda_j` is the mean squared
directional derivative along `w_j`. `f = h(a'x)` gives a rank-one `C` with
`w_1 = a / |a|`, and `f = x'Ax/2` with standard normal inputs gives
`C = A^2`. `sensitivity.active_subspace` samples a `ProbabilisticModel` in
its standard-normal space, where every coordinate has unit variance
(gradients by automatic differentiation with the chain rule
[EQ-SENS-ACTIVE-004], or by central differences at `2 d + 1` runs per
point); `sensitivity.active_subspace_from_gradients` takes gradients
directly. The result reports the eigenvalue gaps, the active dimension (the
largest gap unless given, a heuristic), activity scores
`alpha_i = sum_{j <= k} lambda_j w_ij^2` ([EQ-SENS-ACTIVE-002]), bootstrap
percentile intervals of the eigenvalues, and the mean bootstrap distance
`|W_1 W_1' - W_1* W_1*'|_2` for each candidate dimension
([EQ-SENS-ACTIVE-003]). `sufficient_summary()` returns the pairs
`(w_1'x, f(x))`: plotted, a thin curve supports a one-dimensional active
subspace.

```python
import math
import random

from probstat.modeling.benchmarks import ishigami
from probstat.sensitivity import (
    active_subspace_from_gradients,
    borgonovo_delta,
    chatterjee_indices,
    kernel_sobol_indices,
    pawn_indices,
    rank_sobol_indices,
)

rng = random.Random(7)
rows = [[rng.uniform(-math.pi, math.pi) for _ in range(3)] for _ in range(4000)]
y = [ishigami(*row) for row in rows]

rank = rank_sobol_indices(rows, y, blocks=10)
rank.first_order   # (0.295, 0.423, -0.023)
rank.standard_errors   # (0.018, 0.016, 0.026)
xi = chatterjee_indices(rows, y)
xi.xi   # (0.169, 0.353, 0.012)
xi.dependent()   # ('x0', 'x1')
delta = borgonovo_delta(rows, y, permutations=50, seed=1)
delta.delta   # (0.252, 0.377, 0.187)
round(delta.null_level, 3)   # 0.065
pawn = pawn_indices(rows, y, permutations=200, seed=1)
pawn.index   # (0.241, 0.387, 0.093)
round(pawn.null_level, 3)   # 0.039
kernel = kernel_sobol_indices(lambda row: ishigami(*row), rows[:400], rows[400:800])
kernel.indices   # (0.166, 0.313, 0.053)

gradients = [
    [
        math.cos(x1) * (1.0 + 0.1 * x3**4),
        14.0 * math.sin(x2) * math.cos(x2),
        0.4 * x3**3 * math.sin(x1),
    ]
    for x1, x2, x3 in rows[:1000]
]
space = active_subspace_from_gradients(gradients, bootstrap=100, seed=1)
space.eigenvalues   # (24.14, 11.47, 6.85)
space.eigenvectors[0]   # (-0.029, 0.999, 0.019)
space.subspace_errors   # (0.043, 0.07)
```

The third Ishigami input has a first-order Sobol index of exactly zero, and
neither the rank estimate nor `xi` separates it from zero. The
distribution-based indices do: `delta = 0.187` against a null level of
0.065 and PAWN `0.093` against 0.039, because `x3` changes the spread of
the output through its interaction with `x1`. The Ishigami function has no
low-dimensional active subspace in these coordinates: the three eigenvalues
are of the same order and the leading direction is simply `x2`.

Sources: [REF-GAMBOA-ETAL-2022], [REF-CHATTERJEE-2021], [REF-BORGONOVO-2007],
[REF-PLISCHKE-BORGONOVO-SMITH-2013], [REF-PIANOSI-WAGENER-2015],
[REF-DA-VEIGA-2021], [REF-JANON-ETAL-2014], [REF-CONSTANTINE-DOW-WANG-2014].

## Common pitfalls

- **Dependent inputs.** `sensitivity.sobol_analysis` and
  `sensitivity.morris_screening` raise `ValueError` for a model with a
  non-identity copula. Use `shapley_effects` rather than dropping the
  correlation.
- **No interval by default.** `sobol_analysis` defaults to `bootstrap=0` and
  reports `interval_method == "none"` with `nan` bounds; `UQStudy.sensitivity`
  defaults to `bootstrap=200`, which on a Sobol design gives replicate
  intervals over 8 randomizations. Rows of a quasi-random or LHS design are
  never bootstrapped.
- **Cost.** One analysis costs `size (d + 2)` model runs; use
  `sobol_analysis_all` to share them across QoIs.
- **Rank by total effect.** Estimates are not clipped to `[0, 1]`, and an
  input that acts only through interactions has a first-order index near
  zero.
- **Resampling rank statistics.** Never bootstrap `rank_sobol_indices` or
  `chatterjee_indices`; use `blocks` or the built-in tests.
- **Noise floor of distribution-based indices.** `borgonovo_delta` and
  `pawn_indices` are positive for an inert input; compare with
  `null_level`, not with zero.
- **Scaling of active subspaces.** Eigenvectors change when an input is
  rescaled; gradients passed to `active_subspace_from_gradients` must be in
  coordinates of comparable scale (`active_subspace` uses the
  standard-normal space).
- **Morris on periodic responses.** One Morris step moves an input by half
  of its quantile range, so `sin(x)**2` with `x ~ Uniform(-pi, pi)` gives
  `mu* = 0` although `x` matters.
- **Reading a tornado chart.** One-at-a-time swings show neither
  interactions nor dependence and say nothing between the two ends:
  `tornado_analysis` refuses dependent inputs, and an input flagged in
  `nonmonotone_inputs()` has a swing that understates its effect (zero for
  `sin(x)**2` with a symmetric range).
- **FAST sizes and noise.** `fast_analysis` needs an odd `samples` of at
  least `fast_sample_size(d, M)`; its error comes from the truncation at `M`
  harmonics as well as from the phases, so a longer curve at fixed `M` does
  not help (raise `M`). The raw `rbd_fast_analysis` estimate is biased
  upwards by `(2 M / N)(1 - S_i)`; keep `bias_correction=True`.

```python
import math

from probstat.distributions import Normal, Uniform
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.sensitivity import morris_screening, sobol_analysis
from probstat.uq import QuantityOfInterest
from probstat.workflow import UQStudy

def total(p):
    return {"y": p["a"] + p["b"]}

laws = (RandomInput("a", Normal(0.0, 1.0)), RandomInput("b", Normal(0.0, 2.0)))
qoi = (QuantityOfInterest("y", "sum"),)
independent = ProbabilisticModel(JointInput(laws), total, qoi)
plain = sobol_analysis(independent, "y", 128, seed=1)
(plain.interval_method, plain.evaluations)    # ('none', 512); cost 128 * (2 + 2)
staged = UQStudy(independent, seed=1).sensitivity("y", 128)
(staged.interval_method, staged.replicates)   # ('replicates', 8)
correlated = ProbabilisticModel(JointInput(laws, ((1.0, 0.6), (0.6, 1.0))), total, qoi)
try:
    sobol_analysis(correlated, "y", 64, seed=1)
except ValueError as error:
    refused = str(error).split(";")[0]
refused                                       # 'variance-based Sobol indices require independent inputs'

def periodic(p):
    return {"y": math.sin(p["x"]) ** 2 + p["z"]}

wave = ProbabilisticModel(
    JointInput((RandomInput("x", Uniform(-math.pi, math.pi)), RandomInput("z", Uniform(0.0, 1.0)))),
    periodic,
    (QuantityOfInterest("y", "periodic in x"),),
)
morris_screening(wave, "y", 10, seed=1).mu_star  # (0.0, 1.0)
```

## One-at-a-time swings and Fourier amplitude sensitivity testing

A *tornado chart* answers a design-review question cheaply: with every input
at a baseline, how far does the output move when one input alone goes to a low
and to a high value? `tornado_analysis(model, quantity)` moves each input to
its 0.1 and 0.9 quantile (`low` and `high`) around the median (`baseline`),
costs `1 + 2 d` runs, and returns the swing `y_high - y_low`, its sign, and
the share of the sum of squared half-swings for each input, sorted by absolute
swing [EQ-SENS-TORNADO-001]. Inputs known only as ranges go in through
`values={"name": (low, high)}` or through `tornado_function`, which takes a
callable and a `low / base / high` triple per input.
`visualization.plot_tornado(result)` draws it.

What the chart cannot show: **interactions** (inputs act one at a time, and an
input that acts only through an interaction with an input sitting at zero has
swing zero), **dependence** (a model whose joint law has a copula is refused,
because moves along the axes through a baseline point are not consistent with
the law), and **non-monotone responses** (two ends say nothing about the
interior). The cheap check `nonmonotone_inputs()` compares the baseline output
with the two end outputs and flags an extremum inside the range
[EQ-SENS-TORNADO-002]; it does not catch every non-monotone shape.

Fourier amplitude sensitivity testing (FAST) estimates variance-based indices
from one structured design: the model is run along a periodic search curve
that gives each input its own integer frequency, and the power of the output
spectrum at the harmonics of an input's frequency is its partial variance
[EQ-SENS-FAST-001], [EQ-SENS-FAST-002]. `fast_analysis` is the extended
method: for each input a curve with a high frequency for that input and a set
of low frequencies for the others yields the first-order index and, from the
spectrum below half the high frequency, the total-effect index. `harmonics` is
the interference factor `M`; `samples` must be odd and at least
`fast_sample_size(d, M)` [EQ-SENS-FAST-003]; the `resamples` random phase
shifts give a standard error for each index. `rbd_fast_analysis` is the random
balance design variant: one sample of `N` points gives all first-order
indices, with the Tissot-Prieur correction of the positive bias
`(2 M / N)(1 - S_i)` [EQ-SENS-RBD-001], [EQ-SENS-RBD-002]. Both need
independent inputs; the plain-callable entries are `tornado_function`,
`fast_indices` and `rbd_fast_indices`.

```python
from probstat.modeling.benchmarks import ishigami_model
from probstat.sensitivity import (
    fast_analysis,
    fast_sample_size,
    rbd_fast_analysis,
    tornado_analysis,
)

ishigami = ishigami_model()
swings = tornado_analysis(ishigami, "y")
swings.ranking()   # ('x1', 'x2', 'x3')
swings.nonmonotone_inputs()   # ('x2',)
round(swings.as_mapping()["x2"], 6)   # 0.0
fast = fast_analysis(ishigami, "y", fast_sample_size(3, 8), harmonics=8, seed=1)
fast.ranking()   # ('x1', 'x2', 'x3')
[round(v, 1) for v in fast.first_order]   # [0.3, 0.4, 0.0]
[round(v, 1) for v in fast.total_effect]   # [0.6, 0.4, 0.2]
abs(fast.total_effect[0] - 0.5576) < 0.03   # True
rbd = rbd_fast_analysis(ishigami, "y", 2001, harmonics=8, seed=1)
rbd.ranking()[:2]   # ('x2', 'x1')
abs(rbd.first_order[0] - 0.3139) < 0.03   # True
```

The tornado ranks `x2` last although it carries 44% of the variance: its
response `sin(x2)**2` is symmetric about the baseline, so the swing is zero
and only the flag reveals it. FAST recovers the analytic indices of the
Ishigami function (first order 0.314, 0.442, 0; total 0.558, 0.442, 0.244)
within about 0.01 at `M = 8`, with 11,535 runs for the extended method and
10,005 for the random balance design.

Sources: [REF-HOWARD-1988], [REF-CUKIER-1973],
[REF-SALTELLI-TARANTOLA-CHAN-1999], [REF-TARANTOLA-GATELLI-MARA-2006],
[REF-TISSOT-PRIEUR-2012].
