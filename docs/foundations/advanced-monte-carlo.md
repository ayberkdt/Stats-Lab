# Advanced Monte Carlo and Variance Reduction

Basic Monte Carlo spends independent samples and accepts the resulting
root-`N` error. Variance reduction changes the random contribution being
averaged while preserving the estimand. The useful question is therefore not
“which method has the smallest variance?” in isolation, but “which valid method
has the smallest estimator variance for the available model-evaluation cost?”

This guide covers the UQ-3 implementation. Formula identifiers resolve in the
[equation registry](../equation-registry.md), and source keys resolve in the
[bibliography](../references.md).

## One comparison contract

Every implemented strategy reports an estimate, a standard error, and charged
function evaluations. The cost-normalized score is

`SE(estimate)^2 * evaluations` [EQ-MC-EFF-001].

For two methods targeting the same quantity, a candidate-to-baseline gain above
one favors the candidate. This comparison is local to the model, integrand,
sample budget, and diagnostics used. It does not prove that one strategy is
universally superior.

## Antithetic variates

For `U~Uniform(0,1)`, both `U` and `1-U` have the same distribution. The pair
average

`[f(U)+f(1-U)]/2`

therefore has the same expectation as `f(U)` [EQ-MC-ANTI-001]. If the two
evaluations move in opposite directions, their covariance is negative and the
pair average is less variable. Monotone functions are the familiar favorable
case; arbitrary or oscillatory functions need not benefit.

`estimate_antithetic_uniform` correctly treats each pair average as one
independent contribution while charging two function evaluations. Counting the
two paired values as independent would understate uncertainty.

## Control variates

Suppose a cheap control `C` has a known expectation `mu_C` and is correlated
with target output `Y`. The adjusted contribution is

`Y - beta(C-mu_C)` [EQ-MC-CONTROL-001].

The expectation is unchanged for fixed `beta`. The optimal population
coefficient is `Cov(Y,C)/Var(C)`; the implementation can estimate it from the
sample or accept a fixed coefficient. A sample-fitted coefficient is useful,
but its reported interval is a plug-in approximation because coefficient
estimation uncertainty is not separately propagated. A control with an
incorrectly claimed “known” mean can introduce bias.

## Conditional Monte Carlo

The tower property gives `E[Y]=E[E[Y|Z]]`, while the law of total variance
shows that `E[Y|Z]` is no more variable than `Y` [EQ-MC-COND-001]. The library
accepts already-computed conditional expectations so the mathematical boundary
is explicit: it does not guess or approximate a conditional model silently.

Example: for independent unit-uniform `X,Y`,
`P(X+Y>1 | X)=X`. Averaging sampled `X` values estimates the same probability
as raw indicators, with lower population variance.

## Stratified sampling and allocation

Stratification partitions the input domain, samples within every part, and
weights the local means [EQ-MC-STRAT-001]. The current API uses equal-width
strata on `[0,1]`, but allocations may differ. Its standard error uses the
design-based sum `sum W_h^2 s_h^2/n_h`, not the ordinary IID formula.

When pilot standard deviations are available, Neyman allocation assigns work
in proportion to `W_h sigma_h` [EQ-MC-NEYMAN-001]. The implemented allocator
enforces at least two samples per stratum and apportions the remaining integer
budget by largest remainders. Pilot error, unequal evaluation costs, adaptive
reallocation, and exact constrained optimization remain future extensions.

## Importance sampling

Importance sampling draws from proposal density `q` and weights by `p/q` to
recover an expectation under target density `p` [EQ-MC-IS-001]. It can focus
work where `|g|p` matters, but validity requires proposal support wherever the
target integrand contributes. A poor proposal can produce extreme weights and
variance much worse than crude Monte Carlo.

The implementation reports both the ordinary weighted-contribution standard
error and the weight-only effective-sample-size diagnostic [EQ-MC-ESS-001].
ESS describes weight concentration; it is not an actual sample count and must
not replace the estimator's standard error or convergence checks.

## Reproducible laboratory

Run:

```powershell
python examples/advanced_monte_carlo_lab.py
```

The lab estimates `E[U^2]=1/3` with crude, antithetic, control-variate,
stratified, and importance-sampling strategies under the same 20,000-evaluation
budget. It prints errors, allocation, weight diagnostics, and efficiency gains,
then writes `examples/generated/advanced-monte-carlo-efficiency.png`. A second
experiment contrasts raw-indicator and conditional Monte Carlo estimators.

## Deliberate limits and next steps

- Confidence intervals are large-sample normal approximations.
- Automatic control fitting uses the same sample and reports a plug-in error.
- Stratification is presently one-dimensional with equal-width strata.
- Importance sampling is ordinary rather than self-normalized; adaptive and
  multiple importance sampling are not yet implemented.
- Randomized quasi-Monte Carlo, Latin hypercube sampling, control
  functionals, and rare-event proposal design remain roadmap work here;
  multilevel Monte Carlo is covered in the section below.

The next ATHENS slice builds reliability and rare-event analysis on these
auditable probability, cost, and importance-weight foundations.

## Sources

- [REF-OWEN-2013, Chapters 2, 8, and 9].
- [REF-ROBERT-CASELLA-2004, Chapters 3-4].

## Multilevel Monte Carlo

Many quantities are only computable through an approximation `P_l` whose
accuracy and cost grow with a level `l`: an SDE path with `M^l` Euler steps, a
PDE solve on a mesh refined `l` times. Plain Monte Carlo on the finest level
pays the full cost for every sample. Multilevel Monte Carlo (MLMC) rewrites the
finest expectation as a telescoping sum [EQ-MC-MLMC-001]

`E[P_L] = E[P_0] + sum_(l=1..L) E[P_l - P_(l-1)]`

and estimates every term independently. The trick is *coupling*: the fine and
coarse values in one correction sample use the same randomness (for an SDE the
coarse path sums pairs of the fine Brownian increments), so
`V_l = Var(P_l - P_(l-1))` shrinks with `l` and only a few expensive samples are
needed on fine levels.

The user supplies the hierarchy as one function,
`level_sampler(level, n, stream) -> (fine, coarse)`, returning `n` coupled
values; at level 0 `coarse` is ignored. Every batch receives its own child
`RandomStream`, so runs are reproducible. A fine path at level `l` costs
`path_cost(l)` (default `M^l`), and one correction sample costs
`C_l = c(l) + c(l-1)`.

- `multilevel_monte_carlo(sampler, (N_0, ..., N_L))` uses a fixed allocation.
- `optimal_level_allocation(V, C, v)` returns the cost-optimal
  `N_l = ceil(sqrt(V_l/C_l) sum_k sqrt(V_k C_k) / v)` [EQ-MC-MLMC-002].
- `convergence_rates(means, variances, costs)` fits `alpha` (weak rate),
  `beta` (variance rate), and `gamma` (cost rate) by log-linear least squares
  on levels `l >= 1` [EQ-MC-MLMC-003].
- `adaptive_multilevel_monte_carlo(sampler, eps)` runs Giles' algorithm: start
  with levels `0..2`, allocate for sampling variance `eps^2/2`, and add levels
  until the extrapolated bias `max_j m_(L-j) M^(-j alpha) / (M^alpha - 1)` is
  below `eps/sqrt 2`, so that `MSE <= eps^2`.

`MultilevelResult` reports the estimate, sampling variance, extrapolated bias,
an RMSE estimate, per-level statistics, the three rates, the total cost
`sum N_l C_l`, and `single_level_cost`: plain Monte Carlo on the finest level
with the same sampling variance, `Var(P_L) c(L) / Var(Y_hat)`.

### Worked example (numbers computed with this release)

Geometric Brownian motion `dS = S dt + 0.5 S dW`, `S_0 = 1`, `T = 1`, Euler-
Maruyama with `2^l` steps on level `l`; the exact answer is
`E[S_T] = e = 2.718282`. `adaptive_multilevel_monte_carlo(sampler, 0.02,
seed=1)` (the sampler is `GeometricBrownianMotion` in
`tests/test_mlmc_ce_line_shapley.py`):

| level | `N_l` | mean of `Y_l` | `V_l` | `C_l` |
| --- | --- | --- | --- | --- |
| 0 | 15494 | 2.0031 | 0.2532 | 1 |
| 1 | 5037 | 0.25639 | 0.07964 | 3 |
| 2 | 3426 | 0.18452 | 0.07675 | 6 |
| 3 | 1916 | 0.11324 | 0.04595 | 12 |
| 4 | 1034 | 0.076085 | 0.02643 | 24 |
| 5 | 483 | 0.048253 | 0.01164 | 48 |
| 6 | 240 | 0.019291 | 0.005682 | 96 |
| 7 | 120 | 0.011723 | 0.002531 | 192 |
| 8 | 59 | 0.0028269 | 0.001121 | 384 |

- Estimate `2.71547` (error `-0.0028`), standard error `0.0139`,
  extrapolated bias `0.0081` (the true Euler bias at `L = 8` is `0.0053`), RMSE
  estimate `0.0160 <= 0.02`.
- Rates `alpha = 0.88`, `beta = 0.93`, `gamma = 1.00` (theory for Euler on
  GBM: `alpha = beta = gamma = 1`).
- Total cost `190,889` Euler steps, exactly the number of steps the sampler
  took. Plain Monte Carlo at `L = 8` with the exact `Var(S_T) = 2.0987` needs
  `2 Var(S_T) 2^8 / eps^2 = 2.69e6` steps, a ratio of `14.1`; the in-run
  estimate `single_level_cost` gives `1.58e6` (ratio `8.3`) because
  `Var(P_8)` is estimated from only 59 samples.
- With `eps = 0.04` and `eps = 0.01` the run also stops at `L = 8`, costing
  `47,332` and `745,338` steps, ratios `14.2` and `14.4` against the exact
  single-level cost. Since `beta ~ gamma`, MLMC cost grows like
  `eps^-2 (log eps)^2` instead of `eps^-3` for plain Monte Carlo.

### Limits

- The rate regression and the bias extrapolation assume geometric decay from
  level 1 on; very coarse levels outside the asymptotic regime bias both.
- The bias test uses sample means of the finest corrections, so it is itself
  random; Giles' flooring of `m_l` and `V_l` (at half the value extrapolated
  from level `l - 1`) is applied to keep it stable.
- Cost is the user's model `c(l)`, not wall-clock time.
- Multi-index MC, randomized (unbiased) MLMC, and MLQMC are not implemented.

Sources: [REF-GILES-2008], [REF-GILES-2015].
