# Risk Registers, FMECA, Hazard Matrices, F-N Curves, and Reliability Data

This guide covers the engineering risk-management additions to
`probstat.risk` and the data and allocation tools in `probstat.reliability`.
All of them are *experimental* (ADR-0063). The tests are in
`tests/test_risk_register_fmeca.py` and
`tests/test_reliability_data_allocation.py`.

## 1. Quantitative risk register

Each `Risk` occurs with probability `p` and then has impact `I`. The impact
can be a number, a `DiscreteImpact`, or any law with `quantile`, `mean` and
`variance`. The exposure is `X = B I` with `B ~ Bernoulli(p)` independent of
`I` [EQ-RISK-REG-001]:

    E[X] = p E[I],    Var[X] = p E[I^2] - p^2 E[I]^2.

For **mutually independent** risks the total's mean and variance are the sums
[EQ-RISK-REG-002]. When every impact is discrete, `exposure_distribution()`
convolves the risks exactly [EQ-RISK-REG-004]. Support values are added as
`Fraction`s, so equal totals merge exactly. `simulate()` handles continuous
impacts by Monte Carlo.

Worked example (also a test). The register holds three risks:

- supplier: `p = 0.3`, impact 100, mitigation to `p = 0.1` for 5;
- test: `p = 0.2`, impact 50 or 200 with equal probability, mitigation to a
  fixed 60 for 10;
- late: `p = 0.5`, impact 40.

| Quantity | Value |
|---|---|
| Expected exposure | 75.00 (mitigated 42.00) |
| Standard deviation | 78.26 |
| P90 of the total | 190 (mitigated 100) |
| `P(T > 200)` | 0.0650 |

The exact distribution equals brute-force enumeration of all occurrence and
impact combinations in rational arithmetic.

**Mitigation.** The risk reduction is the drop in expected exposure, and the
ROI is the reduction divided by the mitigation cost [EQ-RISK-REG-003]. The
supplier mitigation gives 20 for 5 (ROI 4.0); the test mitigation gives 13
for 10 (ROI 1.3). `mitigation_waterfall()` orders the steps by ROI: 75 → 55 →
42 at a cumulative cost of 15. For independent risks the reductions add
exactly.

**Independence is an assumption.** Common causes make risks co-occur and
fatten the tail. When that matters, model the drivers in a
`ProbabilisticModel` with a copula instead.

## 2. FMECA (MIL-STD-1629A, Task 102)

The failure-mode criticality number is `C_m = beta alpha lambda_p t`
[EQ-RISK-FMECA-001]:

- `lambda_p`: item failure rate;
- `alpha`: failure-mode ratio;
- `beta`: failure-effect probability;
- `t`: operating time per mission.

The item criticality per severity class is `C_r = sum C_m`
[EQ-RISK-FMECA-002]. Severity classes are I Catastrophic, II Critical,
III Marginal and IV Minor. `criticality_matrix(edges)` bins items or modes by
severity and by `C`, using edges that **the analyst supplies**. The
standard's qualitative probability levels and its guidance values for
`beta` are not encoded, because they must come from the standard or from
programme tailoring, not from this library. `mode_ratio_sums()` and
`inconsistent_items()` flag items whose `alpha` values do not sum to one.

**RPN and its flaws.** `RPN = S x O x D` [EQ-RISK-FMECA-003] multiplies
ordinal ranks. Bowles (2003) shows the problems: the product of ordinal
scales has no quantitative meaning, equal RPNs hide different risks, and
only 120 of the integers 1-1000 can occur. The test suite confirms the 120 by
enumerating all 1000 triples. The worked example shows the effect:

| Mode | Class | `C_m` | RPN |
|---|---|---|---|
| pump: seizure | II | 0.0080 | 96 |
| valve: fails closed | II | 0.0035 | 96 |
| pump: seal leak | III | 0.0060 | 90 |

RPN ties two class-II modes whose criticality numbers differ by a factor of
2.3. Use the criticality numbers, or the quantitative measures in
`probstat.risk`, for decisions.

**AIAG-VDA Action Priority.** The 2019 AIAG-VDA FMEA Handbook replaces RPN
with Action Priority (high, medium or low) looked up from tables of S, O and
D. Those tables are not reproduced here: this library does not encode tables
it cannot quote faithfully. Apply them from the handbook.

## 3. MIL-STD-882E risk assessment matrix

`mil_std_882e_risk_matrix()` returns a `RiskMatrix` that encodes Table III of
MIL-STD-882E. `hazard_risk_level(matrix, severity, level)` classifies a
hazard:

| Probability | 1 Catastrophic | 2 Critical | 3 Marginal | 4 Negligible |
|---|---|---|---|---|
| A Frequent | High | High | Serious | Medium |
| B Probable | High | High | Serious | Medium |
| C Occasional | High | Serious | Medium | Low |
| D Remote | Serious | Medium | Medium | Low |
| E Improbable | Medium | Medium | Medium | Low |
| F Eliminated | Eliminated | Eliminated | Eliminated | Eliminated |

A tailored table can be passed instead. The quantitative definitions of the
probability levels and severity categories (Tables I and II) are tailorable
and are not encoded. Risk matrices carry the limitations described by
Cox (2008).

## 4. F-N curves

`fn_curve(scenarios, criterion)` computes
`F(N) = sum_{N_i >= N} f_i` [EQ-RISK-FN-001] from `(frequency, N)` pairs.
`fn_curve_from_event_trees` builds the pairs from event-tree sequences. The
optional criterion line is `F_c(N) = C (N_0 / N)^a` [EQ-RISK-FN-002].
`F` is a step function and `F_c` decreases, so checking at the distinct `N`
values decides compliance exactly. Example: scenarios
`(1e-3, 1), (1e-4, 10), (2e-4, 10), (1e-5, 100)` against `C = 1e-3`, `a = 2`
give `F(1) = 1.31e-3`, `F(10) = 3.1e-4`, `F(100) = 1e-5`. The largest ratio
to the criterion is 100 at `N = 100`, so the case is not compliant.

## 5. Reliability data (NUREG/CR-6823 practice)

| Data | Bayesian | Classical |
|---|---|---|
| `x` failures in `n` demands | `Beta(a + x, b + n - x)`; Jeffreys `a = b = 1/2` [EQ-REL-DATA-001] | Clopper-Pearson |
| `r` failures in exposure `T` | `Gamma(alpha + r, beta + T)`; Jeffreys `Gamma(r + 1/2, T)` [EQ-REL-DATA-002] | `chi2_{alpha/2}(2r)/(2T)`, `chi2_{1-alpha/2}(2r+2)/(2T)` [EQ-REL-DATA-003] |

The default level is 90 % (5 % and 95 % bounds). For 2 failures in 50
demands the Jeffreys posterior mean is 0.0490, the 90 % credible interval is
[0.01156, 0.10633] and the Clopper-Pearson interval is [0.00715, 0.12061].
For 3 failures in 12 000 h the posterior is Gamma(3.5, 12 000) with mean
2.92e-4. The 90 % credible interval is [9.03e-5, 5.86e-4], and the
time-terminated chi-square bounds are [6.81e-5, 6.46e-4]. A
failure-terminated test uses `2r` degrees of freedom in the upper bound.
Every posterior has `.random_input(name)`, so it can enter a system model as
an epistemic input.

**Demonstration testing** [EQ-REL-DATA-004]:

- Success-run sample size `n = ceil(ln(1 - C) / ln R)`: 22 trials for
  `R = 0.9` at `C = 0.9`. With failures allowed, the size comes from a
  binomial search (105 trials for `R = 0.95`, `C = 0.9`, two failures).
- Exponential test time `T = chi2_C(2r + 2)/(2 lambda)`: 2302.6 h to
  demonstrate `lambda = 1e-3` at 90 % with zero failures.

**Reliability growth (Crow-AMSAA)** [EQ-REL-DATA-005], [EQ-REL-DATA-006]. The
MLE is computed for time-terminated and failure-terminated data, together
with the unbiased shape `(n - 1)/n beta` or `(n - 2)/n beta` and the exact
conditional chi-square interval on `beta`. On the 14-failure example in the
tests (time-terminated at 5000 h): `beta = 0.5531` (unbiased 0.5136),
`lambda = 0.1259`, and the instantaneous MTBF is 645.7 h. The MLE agrees with
direct numerical maximisation of the likelihood. The fit assumes one system
and does not give MTBF confidence bounds.

## 6. Reliability allocation

For series systems, `equal_apportionment`, `arinc_allocation`,
`agree_allocation` (exact and approximate forms) and
`feasibility_of_objectives` [EQ-REL-ALLOC-001..003] return unit reliabilities
and failure rates. Each result reports the achieved system reliability, so
you can see that the approximate AGREE form misses the target slightly
(0.95016 against 0.95 in the test).

**Cost-optimal allocation** minimises `sum c_i(R_i)` subject to
`R_sys >= R*`. The cost model is Mettas's
`c(R) = exp((1 - f)(R - R_min)/(R_max - R))` [EQ-REL-ALLOC-004].

- `cost_optimal_series_allocation` solves the KKT conditions
  `R_i c_i'(R_i) = mu` exactly, in log space to avoid overflow
  [EQ-REL-ALLOC-005].
- `cost_optimal_allocation` accepts any structure function and uses an
  augmented Lagrangian with BFGS on a bounded reparametrisation
  [EQ-REL-ALLOC-006]. A final feasibility repair guarantees
  `R_sys >= R*`.

On a series system the general solver matches the exact Lagrange solution to
1e-6. On `R1 (1 - (1 - R2)(1 - R3)) >= 0.95` it matches SciPy SLSQP to 1e-5.
SLSQP's slightly lower cost comes from ending marginally infeasible
(`R_sys = 0.94999996`).

## Limitations

- The register assumes independent risks and independence between occurrence
  and impact.
- The FMECA criticality numbers are only as good as `alpha`, `beta` and
  `lambda_p`. Propagate their uncertainty, for example with the Gamma
  posteriors above, if decisions depend on the ranking.
- The weighting allocation methods assume constant failure rates and series
  logic. `cost_optimal_allocation` finds a local optimum; try several
  `start` points for non-convex problems.
