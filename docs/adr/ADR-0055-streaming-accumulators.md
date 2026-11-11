# ADR-0055: Streaming (one-pass, constant-memory) accumulators

- Status: Accepted (proposed during the 2026-10-12 session)
- Date: 2026-10-12

## Context

The roadmap line "streaming accumulation" (UQ-2) was open. Every summary in
`probstat.statistics` and `monte_carlo.estimate_mean` / `estimate_probability`
materialized the whole sample. Long Monte Carlo runs, chunked or parallel
evaluation, and "sample until the error is small enough" loops need
summaries that see each value once, use fixed memory, and combine across
chunks.

## Decision

1. New module `probstat.statistics.streaming` (layer 1; it imports only
   `foundations` and `statistics.order`). Public names: `RunningMoments`,
   `WeightedRunningMoments`, `RunningCovariance`, `P2Quantile`,
   `ReservoirSampler`, `ExceedanceCounter`, `run_until`,
   `StreamingRunResult`.
2. **Moments.** Welford's update extended to `M3`, `M4` with Pébay's
   formulas; `merge` uses the Chan-Golub-LeVeque / Pébay pairwise formulas
   and returns a new object (operands unchanged). Weighted moments follow
   West (1979) with three explicit variance conventions (`population`,
   `frequency`, `reliability`) instead of one guessed default. Vector
   co-moments are updated as a symmetric matrix (the upper triangle is
   mirrored so the result is exactly symmetric).
3. **Shifted coordinates.** Every moment accumulator subtracts its first
   observation and adds it back only when reporting the mean. Plain Welford
   gave a relative variance error of `2.0e-8` on 2000 values `1e9 + Exp(1)`;
   the shifted version gives `3.5e-16` against an exact rational reference,
   which met the 1e-12 requirement. Merging accumulators with different
   shifts uses `(K_B - K_A) + mean_B`, exact when the shifts are close.
4. **Accumulators are mutable** (`push`, `extend`) with `__slots__`; results
   (`StreamingRunResult`) are frozen dataclasses, as elsewhere in the
   library.
5. **P²** returns the exact type-7 quantile for `n <= 5`. It has no merge,
   and its docstring and the foundations note state its limits (no
   worst-case bound, order dependence, weak for extreme `p`).
   `ReservoirSampler` (Algorithm R) uses `RandomStream` seeding so that it
   is reproducible.
6. **Wilson interval computed locally.** `inference.wilson_score_interval`
   is in layer 2, so `ExceedanceCounter.wilson_interval` evaluates the same
   formula ([EQ-INF-CI-WILSON-001]) and returns a `(lower, upper)` tuple; a
   test asserts agreement with the inference helper to 1e-15.
7. **Stopping helper.** `run_until(source, target_relative_error,
   max_samples, min_samples=100, check_every=100)` checks
   `SE / |mean|` after `min_samples` draws and then every `check_every`
   draws, not after every draw, to limit the optimism of data-dependent
   stopping. `source` is a zero-argument sampler or an iterable.
8. **Light Monte Carlo integration.** `monte_carlo.estimate_probability`
   now counts indicators through `ExceedanceCounter` (same results and
   errors, no stored list). New `monte_carlo.estimate_mean_streaming`
   returns a `MonteCarloEstimate` from a sampler or iterable, with a fixed
   budget or a relative-error target; its interval is `mean +- z SE`.
   `estimate_mean` is unchanged (it still uses the stored-sample interval
   helper).

## Consequences

- Equations EQ-STAT-STREAM-001 to 007 and six references (Welford 1962,
  West 1979, Chan-Golub-LeVeque 1983, Pébay 2008, Jain-Chlamtac 1985,
  Vitter 1985) were added.
- Found while testing: the two-pass `statistics.skewness` has a relative
  error of `8.8e-8` at offset `1e9`, because the rounded mean enters the
  third moment to first order. The streaming class is more accurate there.
  The two-pass functions were not changed in this ADR.
- Not done: merging P² or reservoir summaries (a weighted reservoir merge is
  possible); higher than fourth moments; propagation of numerical or
  model-form uncertainty (the rest of that roadmap line); streaming in
  `propagation`.
