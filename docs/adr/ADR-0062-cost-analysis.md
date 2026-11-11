# ADR-0062: Cost-analysis package `probstat.cost`

- Status: Accepted (proposed during the 2026-10-16 session, workstream W5)
- Date: 2026-10-16

## Context

Systems engineers need life-cycle cost (LCC), cost-risk, and maintenance-cost
analysis that sit on the same probabilistic model as the rest of the library.
`probstat` had risk measures and decision analysis, but no discounting, no
cost breakdown structure, no S-curves or reserves, and no maintenance or
spares economics.

## Decision

1. **New package `probstat.cost` at layer 6** (same as `risk`). It may import
   `modeling`, `distributions`, `numerics`, `optimization`, `monte_carlo`,
   `uq`, and `risk`. Only `workflow` and `visualization` may import it. It is
   added to `LAYER` (tests/test_architecture.py), to `GROUPS` as a new
   "Cost analysis" group (tools/generate_api_reference.py), and to
   `PACKAGE_STABILITY` as `experimental`.
2. **Modules.**
   - `discounting`: discrete and continuous discount factors, NPV, annuity
     and capital-recovery factors, EAC, the exact Fisher relation, and
     escalation.
   - `life_cycle`: `CostElement` (fixed amount or continuous law, a
     `(time, multiplier)` profile, a category from `COST_CATEGORIES`) and
     `LifeCycleCostModel` (breakdowns by category, time, and element; PV
     coefficients; exact moments; `joint_input`; `as_probabilistic_model`).
   - `moments`: the exact mean and variance of `c_0 + sum a_i X_i` under
     Pearson correlation, together with its independent counterpart.
   - `cost_risk`: a method-of-moments S-curve with a moment-matched lognormal
     or normal total, and a Monte Carlo S-curve through
     `JointInput.from_pearson` (Gaussian copula, Nataf). The confidence-level
     table includes the correlation-ignored column, contingency, and notes.
     Also the JCL helpers.
   - `learning`: Wright and Crawford theories with exact lot sums, and the
     midpoint-integral lot approximation, which is returned with the exact
     value and its relative error.
   - `maintenance`: age replacement (grid plus Brent; reports no finite
     optimum), block replacement (Xie RS renewal-equation solver with a
     step-halving error estimate; parabolic refinement on the grid),
     power-law minimal repair (closed form plus a general numeric optimiser),
     and expected corrective cost.
   - `spares`: Poisson protection level, minimum stock by bisection, exact
     EBO identity, and the plan-cost optimum by critical fractile.
   - `unreliability`: cost of unreliability (Campbell), CER/ICER, ranking of
     alternatives, and `reliability_decision_problem`, which returns a
     `risk.DecisionProblem` for the EVPI hook.
3. **Linearity is exploited and stated.** Every LCC total is
   `sum a_i X_i`, so the deterministic evaluation at the means is the exact
   expectation and the variance is exact. Only the shape of the analytic
   S-curve is approximate, and the result notes say so.
4. **Private Brent minimiser** (`cost/_scalar.py`). `optimization` has no 1-D
   minimiser. A private one avoids touching a shared stable package during
   parallel work. It can be promoted to `optimization` later.
5. **Correlation default.** No correlation means independence, with a note
   that this understates the spread. No default correlation value is
   imposed, because the handbook's guidance on default values was not
   available verbatim.

## Evidence (`tests/test_cost_analysis.py`, 26 tests)

- NPV, annuity, and Fisher identities checked against direct sums.
- LCC breakdown against a hand-built cash-flow sum; moments against
  `a' Sigma a`; probabilistic-model QoIs against `evaluate`.
- `UQStudy` on an LCC model: the propagated mean is within 1e-3, and the
  first-order Sobol indices match `a_i^2 sigma_i^2 / Var` within 0.05.
- Method-of-moments variance 1415 (hand computed) and 1025 independent.
  Normal elements: analytic quantiles equal `scipy.stats.norm` to 1e-6, and
  Monte Carlo quantiles lie within 4 quantile standard errors. Correlated
  lognormals: the Monte Carlo P90 and standard deviation exceed the
  independent run (sd ratio > 1.3). Moment-matched lognormal P80 equals scipy
  (380.842143741).
- Learning curves: exact sums equal numpy sums to 1e-5. The midpoint error is
  below 1e-3 for units 11-30 and larger for the first lot.
- Age replacement, Weibull(2, 1), c_p = 1, c_f = 10: `T* = 0.33645119`
  against scipy brentq on the optimality condition (0.336451191255), and
  `C* = 6.0561214426`. Weibull(3, 1000): `T* = 382.4555`. Exponential: no
  finite optimum, and the closed-form rate matches.
- Renewal function: Erlang-2 closed form and `M(t) = t` within 1e-5 and
  1e-4; the error estimate bounds the actual error. Block-replacement
  Erlang-2 optimum within one grid step of scipy (0.688210679771).
- Minimal repair: the closed form equals the numeric optimiser to 1e-4 in `T`
  and 1e-9 in the rate.
- Spares: CDF and EBO equal `scipy.stats.poisson` and a brute-force sum to
  1e-12. The minimum stock equals `poisson.ppf` (including `mu = 1000`).
- Trade-off ranking and EVPI hand-computed (EVPI = 35).

## Consequences

- Positive: cost becomes a first-class QoI. Correlation effects and reserve
  levels are reported with their assumptions. The maintenance optima carry
  honest flags for the case where no finite optimum exists.
- Limitations: (a) only linear cost totals are built in; nonlinear CERs need
  a custom model. (b) The Gaussian copula has no tail dependence. (c) The
  renewal solver is `O(n^2)` in pure Python (about 0.1 s for 800 steps), and
  the block-replacement optimum is resolved only to the grid step. (d) The
  spares model is single-echelon. (e) Crawford theory is cited through the
  NASA handbook and Yelle (1979); Crawford's 1944 work is not formally
  published. The lot-midpoint formula is derived here as the midpoint rule
  and verified numerically, not copied from a table.
