# Cost Analysis: Life-Cycle Cost, Cost Risk, and Maintenance Economics

`probstat.cost` covers the cost side of systems engineering: discounting,
life-cycle cost (LCC) models, cost-risk analysis with S-curves and reserves,
learning curves, preventive-maintenance optimisation, spare provisioning, and
the cost of unreliability. The package plugs into the rest of the library:
an LCC model becomes a `ProbabilisticModel`, so `UQStudy` can propagate it and
rank the cost drivers by Sobol indices. Reliability parameters estimated in
`probstat.reliability.data` can enter as epistemic inputs.

All numbers below were produced by the library and, where marked, are checked
in `tests/test_cost_analysis.py`. Independent oracles (SciPy, closed forms,
brute-force sums) are listed in ADR-0062.

## 1. Time value of money

| Quantity | Formula | Function |
|---|---|---|
| Discount factor | `(1 + r)^-t` or `exp(-r t)` [EQ-COST-DISC-001] | `discount_factor` |
| NPV | `sum_k CF_k DF(t_k)` [EQ-COST-DISC-002] | `net_present_value` |
| Annuity factor | `(1 - (1 + r)^-n) / r` [EQ-COST-ANN-001] | `annuity_factor` |
| Capital recovery, EAC | `PV r / (1 - (1 + r)^-n)` | `capital_recovery_factor`, `equivalent_annual_cost` |
| Fisher | `(1 + n) = (1 + r)(1 + i)` [EQ-COST-FISHER-001] | `real_rate`, `nominal_rate` |
| Escalation | `A_0 (1 + e)^t` [EQ-COST-ESC-001] | `escalate` |

**Real or nominal, never both.** Discount constant-currency (base-year)
amounts with a real rate. Then-year amounts that include inflation are
discounted with the nominal rate. The Fisher relation is applied exactly;
`nominal - inflation` is only a first-order approximation. Equivalent annual
cost compares alternatives with unequal lives if each is renewed like for
like.

## 2. Life-cycle cost model

A `CostElement` is one line of the cost breakdown structure. Its value `X` is
fixed or has a continuous law, and its *profile* `((t, m), ...)` says when it
is paid. Helper constructors build the usual shapes:

```python
from probstat.cost import CostElement, LifeCycleCostModel
from probstat.distributions import LogNormal, Normal

model = LifeCycleCostModel(
    (
        CostElement.spread("development", Normal(1000, 100), {0: 0.4, 1: 0.6},
                           category="research_development"),
        CostElement.one_time("production", LogNormal(6.0, 0.2), 2, category="acquisition"),
        CostElement.recurring("operations", Normal(50, 5), 3, 7, escalation=0.02,
                              category="operation"),
        CostElement.one_time("disposal", 80.0, 8, category="disposal"),
    ),
    discount_rate=0.05,
)
breakdown = model.deterministic()
```

Present value 1615.34 (undiscounted 1767.71). By category: research and
development 971.43, acquisition 373.31, operation 216.45, disposal 54.15
(tested against a hand-built cash-flow sum).

**Linearity.** Every total equals `sum_i a_i X_i` with
`a_i = sum_k m_ik DF(t_ik)` [EQ-COST-LCC-001]. Evaluating at the element means
therefore gives the **exact expected LCC**, not a percentile. The variance
of the total is exactly `a' Sigma a`, which `model.moments(rho)` returns.

**Integration.** `model.as_probabilistic_model(rho)` returns a
`ProbabilisticModel` with QoIs `lcc_present_value`, `lcc_undiscounted` and
`pv_<category>`. Fixed elements become fixed parameters. Correlations are
Pearson by default; the Nataf transform finds the matching Gaussian-copula
matrix. Pass `kind="latent"` to give the copula matrix directly. In the test
suite, `UQStudy.sensitivity` on such a model recovers the exact first-order
Sobol indices `a_i^2 sigma_i^2 / Var` of the linear total.

## 3. Cost-risk analysis (NASA CEH v4.0 practice)

`cost_risk_analysis(elements, rho)` (or `life_cycle_cost_risk(model, rho)`)
reports two S-curves:

1. **Method of moments.** The mean `sum a_i mu_i` and the variance
   `sum_ij a_i a_j rho_ij sigma_i sigma_j` are exact [EQ-COST-MOM-001]. A
   lognormal (default) or normal total with these moments is *assumed*
   [EQ-COST-MOM-002]. Its percentiles are an approximation.
2. **Monte Carlo.** The elements are sampled jointly through `JointInput`,
   with their own marginals and a Gaussian copula calibrated to the same
   Pearson correlations. Percentiles are sample quantiles
   [EQ-COST-SCURVE-001].

The result also gives the *independent* counterpart and the contingency
`quantile - point estimate` at each level.

Worked example (the model above, `rho = [[1, .5, 0], [.5, 1, .3], [0, .3, 1]]`,
20 000 samples, seed 2026, point estimate = deterministic 1615.34):

| Level | Analytic (lognormal) | Analytic, correlation ignored | Monte Carlo | Contingency (MC) |
|---|---|---|---|---|
| P50 | 1607.99 | 1610.54 | 1609.53 | -5.81 |
| P70 | 1690.56 | 1677.06 | 1689.39 | 74.05 |
| P80 | 1742.55 | 1718.63 | 1740.44 | 125.10 |
| P90 | 1817.30 | 1778.00 | 1810.59 | 195.25 |

The standard deviation is 154.59 with correlation and 124.87 without. Ignoring
correlation would understate the spread by 19.2 % and the P80 reserve by about
24. With five identical lognormal elements (mean 21.76 each) and pairwise
correlation 0.3, the standard deviation is 30.06 against 20.27 independent,
and the Monte Carlo P80 is 132.20 against 125.01. **Omitting correlation
understates the reserve.** The library therefore warns whenever no correlation
is supplied.

For normal elements the total is exactly normal. The test suite checks that
the analytic quantiles equal `scipy.stats.norm` to 1e-6. It also checks that
the Monte Carlo quantiles lie within four quantile standard errors.

**Joint cost and schedule confidence (JCL).**
`joint_confidence_level(costs, schedules, C, S)` estimates
`P(cost <= C and schedule <= S)` from paired samples [EQ-COST-JCL-001].
`cost_for_joint_confidence` returns the smallest budget that reaches a JCL
for a given schedule. It returns `None` when the schedule alone makes the
level unreachable. The samples must come from one joint simulation, so that
the cost-schedule dependence is kept.

## 4. Learning curves

The power law `Y = T_1 x^b` uses `b = log2(slope)` [EQ-COST-LEARN-001]:

- **Crawford (unit)**: `Y` is the unit cost; lot costs are exact sums.
- **Wright (cumulative average)**: `Y` is the average of the first `n` units.
  The total is `T_1 n^(b+1)` and lot costs are differences of totals
  [EQ-COST-LEARN-002].

For an 85 % curve with `T_1 = 1000`: units 1-100 cost 43 753.87 under unit
theory and 33 967.96 under cumulative-average theory. The midpoint
approximation [EQ-COST-LEARN-003] gives 9974.99 for the lot of units 11-30
(exact 9974.60, relative error 3.9e-5, lot midpoint 19.433). For the first lot
(units 1-10) the error rises to 2.6e-3. The approximation is always returned
next to the exact sum and labelled as an approximation.

## 5. Maintenance optimisation

All policies minimise the long-run expected cost per unit time from the
renewal-reward theorem (Barlow and Proschan 1965). Results report
`finite_optimum`, the interval, the cost rate, the run-to-failure rate
`c_f / MTTF`, and the savings.

| Policy | Cost rate | Function |
|---|---|---|
| Age replacement | `(c_p R(T) + c_f F(T)) / int_0^T R` [EQ-COST-AGE-001] | `optimize_age_replacement` |
| Block replacement | `(c_p + c_f M(T)) / T` [EQ-COST-BLOCK-001] | `optimize_block_replacement` |
| Minimal repair + overhaul | `(c_p + c_m Lambda(T)) / T` [EQ-COST-MINREP-001] | `minimal_repair_overhaul`, `optimize_periodic_overhaul` |
| Corrective cost | `c int_0^H w(t) e^{-rt} dt` [EQ-COST-CORR-001] | `expected_corrective_cost` |

- **Age replacement** first scans a log grid, then refines with Brent's
  method. For Weibull(shape 2, scale 1) with `c_p = 1`, `c_f = 10` it gives
  `T* = 0.3364512` and `C* = 6.0561214`. SciPy gives `0.336451191` from the
  optimality condition `h(T) int_0^T R - F(T) = c_p / (c_f - c_p)`. With an
  **exponential** lifetime the rate decreases monotonically, and the result
  reports *no finite optimum*. The same holds for any non-increasing hazard.
- **Block replacement** solves the renewal equation numerically with Xie's
  trapezoidal Riemann-Stieltjes scheme [EQ-COST-RENEW-001]. The result
  carries a step-halving error estimate. Checks: `M(t) = lambda t` for the
  exponential and `M(t) = t/2 - 1/4 + exp(-2t)/4` for Erlang-2, both to about
  1e-5. The optimum sits on the renewal grid and is refined by a parabola;
  `resolution` reports the grid step.
- **Minimal repair** with the power-law process gives
  `T* = eta (c_p / ((beta - 1) c_m))^(1/beta)`. For `beta = 2.5`,
  `eta = 100`, `c_p = 5000` and `c_m = 800`: `T* = 176.976`,
  `C* = 47.087`. The general numeric optimiser reproduces this.

Weibull(2.5, 1000) with `c_p = 1000` and `c_f = 10 000`: age replacement at
354.6 h costs 4.751 per hour, block replacement at about 349 h costs 4.868 per
hour, and run-to-failure costs 11.271 per hour.

## 6. Spare parts

Demand in a period is `Poisson(lambda n T)` [EQ-COST-SPARE-001]. The
expected backorders use the exact identity
`EBO(s) = mu P(D >= s) - s P(D >= s + 1)` [EQ-COST-SPARE-002]. The plan
`s c_u + c_b EBO(s)` is convex, and its optimum is the critical fractile
`P(D <= s) >= 1 - c_u / c_b` [EQ-COST-SPARE-003].

Example: 8 units, `lambda = 1e-4 /h`, 4000 h, so `mu = 3.2`. A 95 %
protection level needs 6 spares (protection 0.9554, EBO 0.0696). With
`c_u = 100` and `c_b = 2000` the optimum is also 6, at total cost 739.18.
These values match `scipy.stats.poisson` to 1e-12.

## 7. Cost of unreliability and trade-offs

`cost_of_unreliability` gives the present value of the expected cost of
failure modes with constant frequencies, `sum lambda_i C_i (1 - e^{-rH})/r`.
This is exact by Campbell's theorem [EQ-COST-UNREL-001].
`compare_alternatives` ranks designs by LCC plus risk cost.
`reliability_decision_problem` builds a `risk.DecisionProblem` across
failure-rate scenarios. Its EVPI is the most it is worth paying to learn
which scenario holds, for example through a test programme.

## Assumptions and limitations

- The cost totals are linear in the element values. Nonlinear cost models,
  such as cost-estimating relationships with uncertain exponents, should be
  wrapped as a custom `ProbabilisticModel` and propagated.
- The analytic S-curve is a moment-matched approximation. Use Monte Carlo
  for the tails. The Gaussian copula has no tail dependence.
- A Pearson correlation matrix must be attainable for the chosen marginals
  (the Nataf transform raises an error otherwise), and the implied latent
  matrix must be positive definite.
- The maintenance models assume negligible repair times, constant costs, a
  known lifetime law, and a single component. The block-replacement optimum
  is resolved only to the renewal-grid step.
- The spares model is single-item, single-echelon, and has no repair within
  the period.
