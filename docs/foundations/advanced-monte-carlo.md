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
- Randomized quasi-Monte Carlo, Latin hypercube sampling, multilevel Monte
  Carlo, control functionals, and rare-event proposal design remain roadmap
  work.

The next ATHENS slice builds reliability and rare-event analysis on these
auditable probability, cost, and importance-weight foundations.

## Sources

- [REF-OWEN-2013, Chapters 2, 8, and 9].
- [REF-ROBERT-CASELLA-2004, Chapters 3-4].
