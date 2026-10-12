# ADR-0056: Error-targeted propagation stage `UQStudy.propagate_until`

- Status: Accepted (proposed during the 2026-10-12 session)
- Date: 2026-10-12

## Context

`UQStudy.propagate(size)` needs the sample size up front and stores every
output. ADR-0055 added constant-memory accumulators (`RunningMoments`,
`ExceedanceCounter`, `P2Quantile`, `run_until`), but no study stage used them.
A common question is "sample until the mean (or a failure probability) is
known to 1%". Without a stage for it, users guessed a size and ran the model
again when it was too small.

## Decision

1. New stage `UQStudy.propagate_until(quantity, *, target_relative_error=0.01,
   threshold=None, failure="above", batch=256, max_samples=200_000,
   quantiles=(0.05, 0.5, 0.95))`. Its stage kind is `sequential_propagation`,
   mapped to `WorkflowStep.PROPAGATION` in `WORKFLOW_GUIDE` (calls list gets
   `probstat.workflow.UQStudy.propagate_until`).
2. **Sampling.** The k-th batch is `model.inputs.sample(m, seed=stream.child(
   "batch", k))` on the stage stream `self.stream("sequential_propagation")`.
   This is plain Monte Carlo, the sampling API `JointInput` already has.
   Randomized QMC is left out because its error estimate needs replicates,
   not one running variance. A study's results depend only on its root seed
   and the run index of the stage.
3. **Accumulators.** Outputs are evaluated one at a time with
   `model.evaluate` and pushed into `RunningMoments`, one `P2Quantile` per
   level, and (with a threshold) an `ExceedanceCounter` of Boolean
   indicators. No output is stored. Memory is O(batch) for the input
   vectors of the current batch, independent of the total `n`.
4. **Failure convention.** `failure="above"` counts `Y >= t` and `"below"`
   counts `Y <= t`. This matches `ModelLimitState` (`g <= 0` is failure),
   not `ExceedanceCounter`'s strict `x > t`. For continuous QoIs the two
   are the same.
5. **Stopping rule.** The check runs after every batch, so `batch` plays the
   role of `check_every`/`min_samples` in `run_until`. Mean mode uses
   `RunningMoments.relative_standard_error()` [EQ-STAT-STREAM-007].
   Probability mode uses `sqrt((1-p)/(n p))` from [EQ-MC-PROB-001], and the
   error is `inf` while no failure has been seen. The last batch is cut so
   that `n <= max_samples`. Reaching the budget gives `converged=False` and
   `"no (max_samples reached)"` in the summary. `run_until` is not called
   directly because it controls only a mean and takes a scalar source. The
   stage must also feed the quantile and exceedance accumulators from the
   same draws.
6. **Result.** Frozen `TargetedPropagation` (exported from
   `probstat.workflow`): controlled estimate, target and achieved relative
   error, converged flag, sample size, batches, mean, SE, P² quantile pairs
   with a `quantile(p)` lookup, threshold/failure, probability, plug-in SE,
   95% Wilson interval, and event count. The stage charges exactly
   `sample_size` evaluations.

## Evidence (seeded runs, `Y = a + b ~ N(10, 2)`, cv 0.2)

- Mean, target 1%, `batch=50`: theory `(0.2/0.01)^2 = 400`. Seeds 1/2/3
  stopped at 500/400/400 runs with relative errors 0.00947/0.00996/0.00990.
- `P[Y >= 12] = 0.15866`, target 5%, `batch=100`: theory 2121. Seeds 1/2/3
  stopped at 1800/2200/2100. Seed 2 gave `p = 0.1559`, Wilson (0.1414,
  0.1717), which covers the truth and a direct 20000-run estimate of 0.1631.
  Seed 1's interval (0.1667, 0.2025) misses the truth. That is consistent
  with 95% coverage, and early stopping on a high estimate also pushes
  that way.
- `P[Y <= 8]`, target 10%: 11 of 12 seeds covered the truth. The runs took
  500 to 700 samples against a theory of 530.
- Budget 300 with target 1e-4: 3 batches (128, 128, 44), `converged=False`.

## Consequences

- Positive: one call answers "how many runs do I need for 1%?" in constant
  memory. The stage appears in the study report and the workflow record like
  every other stage.
- Limitations: (a) data-dependent stopping biases the achieved error
  slightly low and the interval coverage slightly below nominal. Checking
  only once per batch limits this but does not remove it. (b) A mean near
  zero, or a rare event with no observed failures, cannot meet a relative
  target, and the stage will spend the whole budget. (c) P² quantiles are
  approximate and come without an error statement. (d) Only one QoI is
  controlled per call. (e) Only plain Monte Carlo is available; there is no
  QMC or importance sampling in this stage.
