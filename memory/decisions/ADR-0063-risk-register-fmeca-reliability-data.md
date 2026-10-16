# ADR-0063: Risk register, FMECA, MIL-STD-882E matrix, F-N curves, reliability data and allocation

- Status: Accepted (proposed during the 2026-10-16 session, workstream W5)
- Date: 2026-10-16

## Context

`probstat.risk` had risk measures, fault and event trees, and a generic
`RiskMatrix`. It had no register, no FMECA, and no societal-risk curves.
`probstat.reliability` covered structural reliability, but not estimation
from test and field data or allocation of targets to subsystems.

## Decision

1. **`risk.fmeca`.** `FailureMode` and `Fmeca` compute the MIL-STD-1629A
   numbers `C_m = beta alpha lambda_p t` and `C_r` per severity class. The
   criticality matrix uses bin edges that the user supplies. The 1629A
   qualitative probability levels and `beta` guidance are **not encoded**:
   I could not verify every threshold, and the protocol forbids fabricated
   tables. RPN is provided with a documented caveat (Bowles 2003). The
   test enumerates all 1000 triples and confirms that only 120 distinct
   values occur. The AIAG-VDA Action Priority tables are described in the
   documentation only and not encoded.
2. **`risk.register`.** `Risk`, `DiscreteImpact`, `RiskRegister`, and
   `ExposureDistribution`. The independent-risk mean and variance are exact.
   The exact total uses convolution with `Fraction` support keys and a
   support cap. Monte Carlo draws the impacts by inverse transform, using
   common random numbers between the mitigated and unmitigated runs (each
   risk always consumes two uniforms). Ranking keys, ROI, and the ROI-ordered
   waterfall are included.
3. **`risk.hazard_matrix`.** `mil_std_882e_risk_matrix()` encodes MIL-STD-882E
   Table III (risk levels per RAC). I am confident of every cell: High = 1A,
   1B, 1C, 2A, 2B; Serious = 1D, 2C, 3A, 3B; Medium = 1E, 2D, 2E, 3C, 3D,
   3E, 4A, 4B; Low = 4C, 4D, 4E; Eliminated = F. A tailored table can be
   passed instead. The numeric definitions in Tables I and II are not
   encoded. The integrator should still compare the table with the standard
   before the API is promoted.
4. **`risk.fn_curve`.** `F(N) = sum_{N_i >= N} f_i` from scenarios or event
   trees, and a criterion line `C (N_0/N)^a`. The check at step ends is
   exact for non-increasing criteria.
5. **`reliability.data`** (layer 5).
   - Beta posteriors (Jeffreys, uniform, user prior) with Clopper-Pearson
     intervals, reusing `inference.confidence`.
   - Gamma posteriors (Jeffreys `Gamma(r + 1/2, T)` or a conjugate user
     prior), and chi-square bounds for time- and failure-terminated tests,
     one- or two-sided.
   - Success-run and exponential demonstration-test sizes, including `r`
     allowed failures.
   - The Crow-AMSAA MLE for both terminations, with the unbiased shape and
     the exact conditional chi-square interval.
   - Every posterior offers `.random_input(name)` for epistemic studies.
6. **`reliability.allocation`.**
   - Equal, ARINC, AGREE (exact and approximate forms, reporting the
     achieved reliability), and feasibility of objectives.
   - `MettasCost`, with value, derivative, and log-derivative.
   - The exact series KKT solution in log space (Brent for each unit, Brent
     for the multiplier), and a general augmented Lagrangian (BFGS on a
     sigmoid reparametrisation) with a feasibility repair along the segment
     to the upper bounds.
7. All new risk and reliability modules are marked *experimental* in their
   docstrings. Their packages stay *provisional*, because labels are per
   package (ADR-0048).

## Derivations I made myself (verified numerically)

- Crow-AMSAA unbiased factors: `(n - 1)/n` (time) and `(n - 2)/n`
  (failure). They follow from `2 n beta / beta_hat ~ chi2(2n)` or
  `chi2(2n - 2)`, because the `ln(T/t_i)` are iid exponential given `n`.
  A simulation test checks the mean.
- The AGREE "exact" form is chosen so that
  `prod [1 - E_i (1 - R_i)] = R*` holds exactly. The approximate form is
  the classical rate formula.
- The KKT condition `R_i c_i'(R_i) = mu` comes from the constraint
  `sum ln R_i >= ln R*`.

## Evidence

`tests/test_risk_register_fmeca.py` (11 tests) and
`tests/test_reliability_data_allocation.py` (16 tests):

- Register convolution equals brute-force rational enumeration of every
  occurrence and impact combination to 1e-15. The moments are hand
  computed. Monte Carlo agrees with the exact moments for continuous
  impacts.
- FMECA `C_m` and `C_r` hand computed; the RPN set has 120 elements; the
  882E matrix cells round-trip; the F-N curve is hand computed.
- Beta and Gamma posterior quantiles, Clopper-Pearson, and chi-square bounds
  equal `scipy.stats` to about 1e-11. The success-run sizes equal a scipy
  binomial search; the demonstration time equals `chi2.ppf`.
- Crow-AMSAA on a 14-failure dataset constructed for the test: closed-form
  values equal numpy, and the MLE equals Nelder-Mead maximisation of the
  likelihood (`beta = 0.5531288`).
- Allocation: the products meet the targets. Series Lagrange equals scipy
  SLSQP to 1e-6 and satisfies `R_i c_i'(R_i) = const` to 1e-8. The augmented
  Lagrangian equals Lagrange to 1e-6 on series systems and SLSQP to 1e-5 on
  a series-parallel system. The constraint is always satisfied; SLSQP ends
  slightly infeasible.

## Consequences

- Positive: register, FMECA, hazard, and data outputs connect directly to
  the quantitative tools (posteriors as `RandomInput`, event trees to F-N,
  costs to `DecisionProblem`).
- Limitations: the register assumes independence. FMECA probability-level
  tables are not encoded. Crow-AMSAA covers one system, with no MTBF
  confidence bounds and no goodness-of-fit test. Weighting allocations
  assume series logic and constant rates. The general cost-optimal
  allocation finds a local optimum and assumes a coherent (monotone)
  structure for the repair step.
