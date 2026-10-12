# Descriptive Statistics

Let `x_1, ..., x_n` be a non-empty finite sample of real observations.

## Center and spread

The arithmetic mean [EQ-STAT-001] locates the balance point. Variance
[EQ-STAT-002] averages squared deviations from that center with denominator
`n - d`, where `d` is the explicit degrees-of-freedom offset. Standard
deviation [EQ-STAT-003] returns spread to the original units.

Use `d = 0` for the variance of the finite values as the complete population.
Use `d = 1` when the familiar sample variance is intended as an unbiased
estimator under the independent, identically distributed finite-variance
model. The software never guesses which interpretation the user means.

## Standardization

A z-score [EQ-STAT-004] reports signed distance from the mean in standard
deviation units. It is undefined when all values are identical because the
scale is zero.

Standardization changes location and scale, not distributional shape. A
z-score is not automatically a normal probability.

## Empirical distribution

The empirical CDF [EQ-STAT-005] assigns mass `1/n` to every observation. It is
a right-continuous step function and is defined without assuming a parametric
distribution. It is the bridge from observed data to distributional thinking.

## Paired variation

Covariance [EQ-STAT-006] measures joint centered movement in the product of the
input units. Pearson correlation [EQ-STAT-007] divides covariance by both
standard deviations, producing a dimensionless measure of linear association.

Correlation does not establish causation, can miss nonlinear dependence, and
is sensitive to outliers. A scatter plot and data-generating assumptions belong
beside the coefficient.

## Numerical note

The implementation uses `math.fsum` and a centered two-pass variance. This is
more stable than the algebraically equivalent one-pass expression
`mean(x^2) - mean(x)^2`, which can suffer catastrophic cancellation when values
are large and tightly clustered.

References: [REF-NIST-EDA], [REF-WASSERMAN-2004, Chapter 7].

## Streaming (one-pass) summaries

`probstat.statistics.streaming` summarizes data that arrive one value at a
time, or that are too long to store, in constant memory.

| Accumulator | Keeps | Mergeable | Equation |
| --- | --- | --- | --- |
| `RunningMoments` | count, mean, `M2`, `M3`, `M4` | yes | [EQ-STAT-STREAM-001], [EQ-STAT-STREAM-003] |
| `WeightedRunningMoments` | `W`, `sum w^2`, mean, `S` | yes | [EQ-STAT-STREAM-004] |
| `RunningCovariance(d)` | count, mean vector, co-moment matrix | yes | [EQ-STAT-STREAM-002] |
| `ExceedanceCounter` | trials, events | yes | [EQ-INF-CI-WILSON-001] |
| `P2Quantile(p)` | five markers | no | [EQ-STAT-STREAM-005] |
| `ReservoirSampler(k)` | uniform subset of size `k` | no | [EQ-STAT-STREAM-006] |

`merge` combines the summaries of two disjoint chunks into the summary of
their concatenation (Chan, Golub, and LeVeque; Pébay), so chunks can be
summarized in parallel and combined in any order.

### Large offsets

Each moment accumulator subtracts its first observation `K` before updating.
For tightly clustered values far from zero, `x - K` is exact and the
recurrence works on numbers of order one. On 2000 values `1e9 + Exp(1)`
(seed 7), relative errors against an exact rational computation were:

| Method | variance | skewness |
| --- | --- | --- |
| `RunningMoments` (shifted Welford/Pébay) | 3.5e-16 | 3.9e-16 |
| `RunningMoments`, 7 merged chunks | 1.2e-16 | 2.6e-16 |
| two-pass `variance` / `skewness` | 2.4e-15 | 8.8e-8 |
| unshifted Welford | 2.0e-8 | — |
| `mean(x^2) - mean(x)^2` | returns `-128` (true value 0.956) | — |

The two-pass skewness error comes from the rounded mean: at `1e9` the mean
can only be stored to about `1e-7`, and the third moment is sensitive to the
center to first order.

### Streaming quantiles

`P2Quantile` (Jain and Chlamtac) keeps five markers and moves the three
interior ones by a piecewise-parabolic prediction. Over five seeds per case,
the largest errors were:

| Law, `p` | `n` | max rank error | max absolute error |
| --- | --- | --- | --- |
| N(0,1), 0.5 | 1e4 / 1e5 | 0.0014 / 0.0004 | 0.016 / 0.006 |
| N(0,1), 0.9 | 1e4 / 1e5 | 0.0014 / 0.0002 | 0.021 / 0.015 |
| N(0,1), 0.99 | 1e4 / 1e5 | 0.0008 / 0.0001 | 0.062 / 0.023 |
| Exp(1), 0.5 | 1e4 / 1e5 | 0.0016 / 0.0003 | 0.017 / 0.003 |
| Exp(1), 0.9 | 1e4 / 1e5 | 0.0009 / 0.0008 | 0.080 / 0.023 |
| Exp(1), 0.99 | 1e4 / 1e5 | 0.0006 / 0.0002 | 0.084 / 0.026 |

The rank error is `abs(F_n(q) - p)`, with `F_n` the empirical CDF of the same
stream: the error of P² as a substitute for the sample quantile. Most of the
absolute error `abs(q - q_true)` is sampling error that a stored sample quantile has too. Accuracy limits: P²
has no worst-case bound, depends on arrival order (sorted or drifting
streams are the worst case), is weakest for extreme `p` and heavy tails, and
cannot be merged. For mergeable or exact summaries use `ReservoirSampler`
(Vitter's Algorithm R), which keeps a uniform random subset of fixed size and
is reproducible from a seed or `RandomStream`.

### Exceedance probabilities and stopping

`ExceedanceCounter(threshold)` counts strict exceedances `x > t` (or Boolean
events) and reports the Wilson interval with the same formula as
`inference.wilson_score_interval`. `monte_carlo.estimate_probability` now
counts its indicators in one pass.

`run_until(sampler, target_relative_error=..., max_samples=...)` draws into a
`RunningMoments` until `SE / |mean| <= target` [EQ-STAT-STREAM-007]. For
`Exp(1)` with a 1 % target (seed 17) it stopped after 10 200 draws with mean
0.9918, standard error 0.0099, and relative error 0.00999; theory predicts
about `1 / 0.01^2 = 10 000` draws. Data-dependent stopping makes the reported
error slightly optimistic, and a mean near zero makes a relative target
unreachable. `monte_carlo.estimate_mean_streaming` wraps the same rule and
returns a `MonteCarloEstimate` whose work count is the number of draws.

References: [REF-WELFORD-1962], [REF-WEST-1979],
[REF-CHAN-GOLUB-LEVEQUE-1983], [REF-PEBAY-2008], [REF-JAIN-CHLAMTAC-1985],
[REF-VITTER-1985], [REF-WILSON-1927].
