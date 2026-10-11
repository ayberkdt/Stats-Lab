# ADR-0016 — Quantitative risk-analysis layer

- Status: Accepted
- Date: 2026-10-11

## Context

Reliability analysis answers "how likely is failure?" but not "how bad is it,
and is it worth paying to reduce?". Users need the full Kaplan-Garrick view of
risk (scenarios, likelihoods, consequences): tail risk measures on loss
samples, consequence models attached to the shared `ProbabilisticModel`,
logic models for system failure (fault and event trees), fragility-hazard
convolution, and decision analysis with value of information. These tools
consume propagation, optimization, statistics, and modeling, so they cannot
live in any existing package without an upward import.

## Decision

1. Add `probstat.risk` at architecture layer 6. It may import layers 0-5
   (foundations, linalg, optimization, statistics, modeling, monte_carlo,
   propagation) and never `filtering`, `surrogates`, or `workflow`.
   `visualization` moves to layer 9 so future layers 6-8 fit beneath it.
2. Losses are oriented so larger is worse. Every sample measure accepts
   optional non-negative weights normalized internally.
3. VaR is the lower quantile `inf{l : F(l) >= alpha}` (Hyndman-Fan type 1 for
   equal weights), never interpolated; cumulative weights are compared with a
   `1e-10` tolerance. CVaR uses the Rockafellar-Uryasev formula
   `VaR + E[(L - VaR)^+]/(1 - alpha)`, which is exact for atoms and ties.
   Levels must lie strictly inside `(0, 1)`.
4. `assess_risk` always propagates through `probstat.propagation.propagate`,
   so designs, seeds, and evaluation counts follow the shared model contract.
   Loss functions are explicit objects (`step`, `linear`, `piecewise_linear`,
   `custom`) and must return finite values.
5. Fault trees are static, coherent, and use independent basic events; equal
   names denote one shared event and conflicting definitions are errors.
   Minimal cut sets come from MOCUS with absorption. The default top-event
   probability is exact by Shannon decomposition on repeated events with a
   `2^r` guard (`r <= 16` by default); inclusion-exclusion over cut sets is a
   guarded cross-check (at most 15 cut sets). Fussell-Vesely uses the exact
   cut-set-union definition rather than the `1 - Q(0_i)/Q` shortcut.
6. Event-tree branches carry conditional probabilities that must sum to one
   per heading. Dependence between headings is the caller's responsibility;
   fault-tree linking with shared events is out of scope.
7. Fragility fitting maximizes the binomial likelihood over log-parameters
   with `optimization.minimize_bfgs`, and refuses data for which no finite MLE
   exists (single outcome type, one distinct intensity, complete separation).
   Hazard convolution uses geometric-midpoint decrements plus an optional tail
   term and is checked against the power-law closed form.
8. Decision analysis supports finite decision/state tables, EVPI, and EVSI for
   discrete experiments by preposterior analysis. A risk matrix is provided
   only as a communication aid and documents Cox (2008) limitations.

## Consequences

Risk studies reuse the single model definition, inherit reproducible seeding,
and report model-evaluation cost. Every numeric path has a hand-computed or
closed-form test oracle. Exact fault-tree quantification is exponential in the
number of repeated events, which is acceptable for teaching-scale trees but
not for plant-scale PSA models; BDD-based quantification is the natural
extension. The fragility fit reports no parameter uncertainty, EVSI is limited
to discrete experiments, and percentile bootstrap intervals are unreliable
when few losses lie in the tail.

## Rejected alternatives

- Interpolated (type-7) VaR: it can return a value no scenario produces and
  breaks the exact CVaR identity for finite samples.
- Averaging all losses at or above VaR as CVaR: it overcounts tied atoms and
  is not coherent for discrete laws.
- Rare-event approximation as the default top-event probability: it can
  exceed one and silently overstates risk for non-rare events.
- Putting risk measures in `statistics` or fault trees in `reliability`:
  both would require upward imports of propagation or modeling.
- Using a risk matrix to rank or select mitigations: matrix categories can
  contradict quantitative risk orderings.
