# ADR-0143 — Schedule risk analysis and joint cost-schedule confidence

- Status: Accepted
- Date: 2026-11-24
- Extends: ADR-0048 (API stability)

## Context

`probstat.cost.cost_risk` computes a cost S-curve and a joint confidence level
(JCL) from paired cost and schedule samples, but nothing in the library
produced the schedule samples: there was no activity network, no critical
path, and no link between duration and cost. A cost estimate whose schedule
side is a hand-typed distribution cannot show that a slip drives cost through
the burn rate, which is the main reason cost and schedule are positively
dependent.

## Decision

1. **Layer-6 module in `probstat.cost`**: `schedule_risk.py`. It imports
   `modeling` (Gaussian copula via `JointInput`), `propagation`
   (`QuantileInterval`), `inference` (Clopper-Pearson), `risk` (`Risk`),
   `statistics`, and `cost.cost_risk`; no new import cycle.
2. **Network.** `Activity(name, duration, predecessors)` with a number or a
   law; predecessors are names (finish-to-start) or `Precedence(activity,
   kind, lag)` with kind `FS`, `SS`, `FF`. Start-to-finish is left out: it is
   rare and the backward pass for it was not worth an untested branch. The
   network validates names and acyclicity; a cycle is reported as
   `a -> b -> c -> a`, rotated to start at the earliest-listed activity.
3. **CPM** ([EQ-COST-CPM-001]) as one forward and one backward pass over a
   cached topological order; the same pass runs inside the simulation.
   Critical means total float within `1e-9 * max(1, T)`; critical paths are
   chains of critical activities joined by zero-slack relations, enumerated
   up to `max_paths`. Free float is the smallest slack of the outgoing
   relations.
4. **PERT** ([EQ-COST-PERT-001]) as a named function restricted to
   finish-to-start networks; with several critical paths it uses the one of
   largest variance. Its merge bias is stated in the docstring, in every
   result's `notes`, and demonstrated in a test against simulation (two
   parallel Exp(1): PERT mean 1, true mean 3/2).
5. **Simulation** ([EQ-COST-SRA-001..003]). Durations are drawn through
   `JointInput`: the default correlation is the latent correlation of the
   normal scores (the Gaussian copula parameter), and
   `correlation_kind="pearson"` solves for the latent value that matches
   stated Pearson correlations (Nataf). Only random activities enter the
   copula; a fixed activity named in a correlation is an error. A negative
   draw raises an error naming the activity rather than being clipped, which
   would silently change the law. Risk events wrap `risk.Risk` (probability
   and impact are the record's; the affected activity is the only addition),
   so no second risk type exists; impacts are added to the activity duration
   and therefore reach the critical path and the burn cost. Each event has
   its own derived stream, so adding an event does not change the duration
   draws.
6. **Measures.** Criticality index (Van Slyke) with a Clopper-Pearson
   interval; cruciality as the Pearson and Spearman correlation of the
   duration with the completion time (Williams); a schedule sensitivity index
   defined as criticality times `sd(duration) / sd(completion)`, a
   convention of schedule risk software that is stated, not attributed to a
   source. Deadline probability with an exact Clopper-Pearson interval.
7. **Quantile intervals.** The result type is `propagation.QuantileInterval`
   and the ranks are those of `order_statistic_interval`. The ranks are found
   by bisection on the `Binomial` distribution function instead of calling
   that function, because its table construction is quadratic in the sample
   size (about 20 s for 8 000 draws); a test checks both give identical
   ranks. The cure for the duplication is to make the table linear in
   `propagation`, which is outside this change.
8. **Cost loading** ([EQ-COST-LOAD-001]): time-independent cost per
   activity, burn rate times sampled duration per activity, project rate times
   the sampled completion, and a constant; costs and rates given as laws are
   drawn independently per sample from their own derived streams. The paired
   samples go to the existing `joint_confidence_level` and
   `cost_for_joint_confidence` (reused, not reimplemented);
   `cost_schedule_frontier` lists, per deadline, the smallest cost reaching a
   stated joint confidence.
9. **Joint confidence is below both marginals** is stated in the module,
   package and guide docstrings and shown in the guide example (0.7 and 0.7
   marginals, 0.49 joint when independent).

## Sources

Read in this work: the GAO Schedule Assessment Guide GAO-16-89G (Best
Practice 8: correlation between durations, risk criticality, sensitivity
indexes, the cumulative distribution as the probability of finishing by a
date). Confirmed bibliographically only (formulas re-derived, nothing
copied): Kelley and Walker (1959), Malcolm et al. (1959, already registered),
Van Slyke (1963), MacCrimmon and Ryavec (1964), Williams (1992), Elmaghraby
(2000). The NASA Cost Estimating Handbook appendix on the JCL could not be
opened (the retrieval returned an error), so the JCL is defined here as the
probability of meeting both targets and is not attributed to a page of the
handbook.

## Consequences

- Positive: a cost-risk study can now carry its own schedule model, show
  which activities drive the completion time, and give a defensible JCL and
  frontier.
- Positive: one risk-record type (`Risk`) serves cost and schedule.
- Negative: simulation is pure Python; 8 000 draws of a two-activity network
  took about 2 s when measured, most of it input validation in the correlation
  statistics, so large networks take proportionally longer.
- Left out deliberately: start-to-finish relations, calendars, resource
  leveling, tail-dependent copulas, conditional (event-chain) risks,
  probabilistic branching, and Elmaghraby's analytic sensitivity of the mean
  completion time to an activity mean (a finite-difference or
  likelihood-ratio estimator would be needed to do it right).
