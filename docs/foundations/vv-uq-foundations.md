# Verification, Validation, and UQ Foundations

## The three questions

Verification, validation, and uncertainty quantification answer different
questions and must not be collapsed into a single accuracy score.

- **Solution verification:** Are the chosen equations being solved correctly?
  It studies discretization, iteration, truncation, and roundoff error.
- **Model validation:** Are those equations sufficiently representative of the
  real system for a stated use? It compares model predictions with relevant
  observations while accounting for uncertainty on both sides.
- **Uncertainty quantification:** Given uncertain inputs, numerical error, data,
  and model limitations, what uncertainty should be assigned to a quantity of
  interest?

Validation is conditional on intended use, operating regime, observations, and
quantities of interest. It never proves that a model is universally true.

## Forward model and quantity of interest

The first software contract is the named deterministic map

`Y = g(X)` ([EQ-UQ-001]),

where `X` is one finite parameter realization and `Y` is a set of model
responses. A **quantity of interest (QoI)** is a named component or derived
summary used for a scientific or decision purpose. `ForwardModel`,
`QuantityOfInterest`, and `evaluate_model` make names and finite-value checks
explicit. A stochastic simulator should expose its random stream as an input
or present each seeded realization as a deterministic call.

A forward problem predicts `Y` from `X`. An inverse problem uses observations
of outputs to learn inputs, latent states, or model parameters. Inversion needs
an observation model and assumptions about noise; identifiability is not
guaranteed merely because a forward function exists.

## Two independent uncertainty classifications

`UncertaintyNature` records interpretation:

- **Aleatory:** variability represented as inherent under the chosen model.
- **Epistemic:** incomplete knowledge that may be reduced by information.

`UncertaintySourceKind` records engineering origin:

- parametric,
- model form,
- numerical, or
- observation.

These axes are deliberately separate. A material parameter may describe
population variability (aleatory) or imperfectly known calibration
(epistemic). The label depends on the modeled system and decision context, not
the variable name alone.

## Numerical verification

For three solutions obtained with a common uniform refinement ratio `r`, the
observed order is estimated by [EQ-VV-001]. Under a single leading error term
and asymptotic convergence, Richardson extrapolation [EQ-VV-002] predicts the
zero-step-size value.

Those assumptions are substantive. Oscillatory convergence, changing meshes,
mixed error terms, iterative error, noisy simulators, or a result outside the
asymptotic range can make the estimates misleading. At least three resolutions
and a convergence study are evidence, not a ritual.

`NumericalErrorBudget` records non-negative absolute bounds for truncation,
roundoff, discretization, and iterative error. Its sum [EQ-VV-003] is a
conservative triangle-inequality bound. It is not a probabilistic standard
deviation and introduces no independence assumption.

## Simulation as a computational experiment

A simulation run is documented the way a physical experiment is:
`run_computational_experiment` records the model identity, the physical
inputs, the **numerical settings** (grid spacing, time step, tolerance, solver
seed) that play the role of the apparatus, and the QoIs, in a frozen
`ComputationalExperiment`. Settings may not shadow physical inputs, so a
change of mesh can never be mistaken for a change of physics. A
grid-refinement study is then a designed computational experiment in which
only one setting varies; `refinement_study` runs it and hands the QoI sequence
to `verify_solution`.

## Solution verification: the Grid Convergence Index

`grid_convergence_index` implements the three-grid procedure of Celik et al.
[REF-CELIK-ETAL-2008]. With grids ordered fine (1), medium (2), coarse (3)
and representative sizes `h = (V/N)^(1/d)` ([EQ-VV-GCI-001]):

1. `r21 = h2/h1`, `r32 = h3/h2`, `eps21 = phi2 - phi1`, `eps32 = phi3 - phi2`;
2. the ratio `R = eps32/eps21` classifies the sequence ([EQ-VV-GCI-006]):
   oscillatory if `R < 0`, monotone if `R > ln r32 / ln r21`, otherwise
   divergent (no positive order exists and no GCI is formed);
3. the observed order solves `p = |ln|R| + q(p)| / ln r21`,
   `q(p) = ln((r21^p - s)/(r32^p - s))`, by fixed-point iteration
   ([EQ-VV-GCI-002]); for a uniform ratio `q = 0` and one step suffices. If the
   iteration is not contractive (very different ratios) the equivalent
   strictly increasing equation is solved by Brent's method, and the solver
   used is reported;
4. Richardson extrapolation and relative errors ([EQ-VV-GCI-003]);
5. the fine-grid GCI with factor of safety `F_s = 1.25` ([EQ-VV-GCI-004]),
   reported in QoI units and relative to `phi1`.

**Worked numbers (trapezoid rule, non-uniform refinement).** The composite
trapezoid integral of `exp(x)` on `[0, 1]` with `N = 8, 12, 20` intervals
(`h = 0.125, 0.0833, 0.05`; `r21 = 1.667`, `r32 = 1.5`) gives
`phi1 = 1.7186398`, `phi2 = 1.7192761`, `phi3 = 1.7205186`. The fixed point
converges in 15 iterations to `p = 1.99951` (formal order 2);
`phi_ext = 1.71828173` (error `-1.0e-7` against `e - 1`);
`e_a^21 = 3.70e-4`, `e_ext^21 = 2.08e-4`; `GCI_fine = 4.48e-4`
(relative `2.60e-4`), which brackets the true fine-grid error `3.58e-4` by the
factor `1.25` almost exactly. `tests/test_uq0_solution_verification.py`
asserts the order, the bracketing, and the extrapolation accuracy.

**The asymptotic-range check needs care.** Roache's ratio
`GCI_23 / (r21^p GCI_12)` ([EQ-VV-GCI-005]) is reported (`0.99963` above) but,
evaluated at the observed order of the same three grids, the absolute-GCI
ratio is identically one — it is the order equation rearranged — and the
relative-GCI form only adds the factor `|phi1/phi2|`. The library therefore
also evaluates the ratio at the **theoretical** order
(`asymptotic_ratio_at(p_th)`, `0.99978` above; for a uniform ratio it equals
`r^(p_obs - p_th)`) and `verify_solution` warns when it leaves `1 +/- 0.1`.

**Oscillation.** `phi = (0.99, 1.02, 1.0)` on `h = (0.025, 0.05, 0.1)` has
`R < 0`. The published procedure still produces a number, but Richardson
extrapolation assumes monotone convergence [REF-ROY-2005]. `verify_solution`
therefore discards it and uses the conservative project rule
`U = 3 (phi_max - phi_min)/2 = 0.045` ([EQ-VV-OSC-001]), marks the statement
unreliable, and reports no order.

## Least-squares observed order from four or more grids

With more grids than unknowns, `least_squares_convergence` fits
`phi_i = phi_0 + alpha (h_i/h_1)^p` by least squares ([EQ-VV-LSQ-001]): linear
in `(phi_0, alpha)` for fixed `p`, then a profile scan and golden-section
search in `p`. The residual standard deviation `U_s` exposes scatter that would
corrupt a three-grid order. The uncertainty rule ([EQ-VV-LSQ-002]) is a
**simplification** of Eca and Hoekstra [REF-ECA-HOEKSTRA-2014]: credible
orders `0.5 <= p < p_th + 0.1` use `U = 1.25 |alpha| + U_s`; otherwise the data
are refitted at `p = p_th` and `U = 3 max(|alpha|, |alpha_th|) + max(U_s,
U_s,th)`. Their weighted fits, alternative fixed-order expansions, and
data-range term are not implemented.

For the trapezoid integral with `N = 8, 12, 16, 24, 32`, the fit gives
`p = 1.99959`, `phi_0 = 1.71828177`, `U_s = 1.1e-8`, `alpha = 1.3988e-4`
(true fine-grid error `1.3983e-4`), and `U = 1.749e-4`.

## From verification to `u_num`

`verify_solution` returns a `SolutionVerification` statement
`phi_1 +/- U` ([EQ-VV-UNUM-001]) with the method, convergence type, order,
factor of safety, and warnings. Its use downstream is always explicit:

- `standard_uncertainty("uniform")` gives `U/sqrt(3)` (rectangular, the same
  conversion `budget_standard_uncertainty` applies to error bounds);
  `standard_uncertainty("normal95")` treats `U` as a Gaussian 95 % expanded
  uncertainty (`U/1.96`). For the oscillatory example these are `0.0260` and
  `0.0230`;
- `error_budget()` returns a `NumericalErrorBudget` with
  `discretization = U`;
- `VerificationEvidence.from_solution_verification` places the statement in a
  `CredibilityRecord`, and `assess_predictions(numerical_uncertainty=...)`
  uses `u_num` in the ASME V&V 20 comparison.

The statement applies to the QoI, parameters, and settings of the refinement
study. Using it at other parameter values is an assumption and should be
stated as such.

## Prediction with separate uncertainty components

`predict_with_uncertainty_components` (in `probstat.validation`, layer 5,
because the model-form part needs the validation domain and validation
comparisons that layer-4 `propagation` may not import) reports three
components separately [REF-ROY-OBERKAMPF-2011]:

| Component | Input | Treatment |
|---|---|---|
| parametric | propagated or posterior-predictive samples `y_s` | sample mean, sd, equal-tailed interval |
| numerical | `SolutionVerification` (`U_num`) | `"interval"` (default, no distribution), `"uniform"`, or `"normal"` |
| model form | `AdditiveGaussianDiscrepancy` or `DiscrepancyExtrapolation` | `N(0, sigma_delta(x)^2)` |

The discrepancy variance at validation points comes from the ASME comparison
errors, `sigma_delta,0^2 = max(0, mean(E^2) - mean(u_val^2))`
([EQ-VAL-DISC-002], `estimate_discrepancy`); the mean error is reported because
a systematic bias is absorbed into the variance. Away from the validation
points it is extrapolated by the declared rule
`sigma_delta^2(x) = sigma_delta,0^2 (1 + kappa d(x))` ([EQ-VAL-DISC-001]), with
`d` the scaled distance to the nearest validation condition. The combined
interval is the sample interval of `y + e + d` under independence
([EQ-VAL-PRED-001]); with the interval treatment the numerical error gets no
distribution and the interval of `y + d` is widened by `U_num` on each side
([EQ-VAL-PRED-002]). The result keeps every component, the variance budget and
shares, the list of excluded components, and the exact combination rule.

## The UQ workflow map

`UQWorkflowRecord` holds one entry per step with a status (`pending`,
`in_progress`, `complete`, `not_applicable`) and evidence references. A step
cannot be complete without evidence or not applicable without a note, and
`gaps` reports unsettled steps and steps completed before their prerequisites
(validation before verification, prediction before verification and
validation, prediction without propagation or calibration).
`record_stages(study.records)` attaches `probstat.workflow.UQStudy` stage
records (stage, label, stream seed, evaluations) as evidence.

| Step | Question | Direction | Library calls | `UQStudy` stage |
|---|---|---|---|---|
| QoI definition | Which output supports the decision? | evidence | `uq.QuantityOfInterest`, `uq.evaluate_model`, `modeling.ProbabilisticModel` | — |
| Input characterization | What is uncertain, of what nature, with what law? | evidence | `uq.UncertaintySource`, `modeling.RandomInput`, `modeling.JointInput` | — |
| Verification | Are the equations solved correctly; how large is `u_num`? | evidence | `uq.refinement_study`, `uq.verify_solution`, `uq.grid_convergence_index`, `uq.least_squares_convergence`, `validation.VerificationEvidence` | — |
| Validation | Do predictions agree with independent data in the domain? | evidence | `validation.assess_validation`, `validation.assess_predictions`, `validation.estimate_discrepancy` | `validate` |
| Propagation | What output uncertainty does the input law induce? | forward | `propagation.propagate` | `propagate`, `sensitivity`, `screening`, `reliability`, `form`, `risk`, `emulate` |
| Calibration | What do observed outputs say about the inputs? | inverse | `inverse.CalibrationProblem`, `inverse.calibrate_linear_gaussian` | `calibrate` |
| Prediction | What is claimed at the intended use? | forward | `inverse.predict`, `validation.predict_with_uncertainty_components`, `validation.PredictionStatement` | `predict` |

The same map is available in code as `probstat.uq.WORKFLOW_GUIDE`; a test
imports every listed call.

## Forward versus inverse: a worked example

`examples/uq_workflow_lab.py` (seed 2026; asserted by
`tests/test_uq0_prediction_workflow.py`) solves `dy/dt = -k y`, `y(0) = A`,
with Heun's second-order method at step `h = 0.25`. "Reality" is
`5 exp(-0.3 t) + 0.1`: the model lacks the offset, so it has genuine
model-form error. The QoI is `y` at `t = 10`.

1. **QoI and inputs.** `A ~ Uniform(3, 7)`, `k ~ Uniform(0.1, 0.6)`, both
   recorded as epistemic parametric sources.
2. **Verification.** `refinement_study` at the prior means (`A = 5`,
   `k = 0.35`) on `h = 1, 0.5, 0.25` gives
   `y = 0.15170876 +/- 0.00078` with observed order `2.28`; the asymptotic
   ratio at the formal order is `1.21`, so the statement carries a warning
   that the coarsest grid is not in the asymptotic range. (The exact value
   `5 e^-3.5 = 0.150987` lies inside the band.)
3. **Forward problem.** `UQStudy.propagate(1024, design="sobol")` pushes the
   prior through the model: mean `0.366`, 5-95 % interval
   `[0.015, 1.434]`.
4. **Inverse problem.** `UQStudy.calibrate` with seven noisy observations at
   `t = 0, ..., 6` (noise sd `0.05`): posterior `A = 5.125 +/- 0.045`,
   `k = 0.2944 +/- 0.0046` (rank-normalized R-hat `1.026` and `1.010`). The
   calibration absorbs part of the missing offset into `A` and `k`.
5. **Validation.** Separate data at `t = 7, 8, 9` (four replicates each)
   against posterior-predictive samples: the calibrated model is
   **inconsistent** with these data (u-pooling rejects), and
   `estimate_discrepancy` gives `sigma_delta,0 = 0.058` with mean error
   `-0.055` (the model under-predicts late times).
6. **Prediction.** At `t = 10` (scaled distance `d = 0.5` beyond the
   validation domain `t in [7, 9]`, `kappa = 1`):

| Component | Value |
|---|---|
| parametric (posterior push-forward) | mean `0.2708`, sd `0.0110`, 95 % `[0.2478, 0.2943]` |
| numerical | `+/- 0.00078` (interval treatment) |
| model form | `sigma_delta(10) = 0.0714` (97.7 % of the probabilistic variance) |
| combined 95 % | `[0.1329, 0.4063]` |
| truth | `0.3489` |

The parametric interval alone — the answer the inverse problem gives — misses
the truth; the combined interval covers it because validation exposed the
model-form error and the prediction carries it, inflated for extrapolation.
The forward problem answered "what does the prior imply" (5-95 % width
`1.42`); the inverse problem narrowed the parametric 95 % interval to width
`0.047`; neither alone says how wrong the model equations are. The final
`UQWorkflowRecord` has every step complete, no gaps, and directions
`forward` and `inverse`.

## Scope boundary

Experimental validation metrics, the scoped validation verdict, and the
separation of verification, validation, and prediction evidence are covered in
[Model Validation and Prediction Credibility](model-validation.md). Solution
verification here covers discretization error from systematic refinement; it
does not estimate iterative or round-off error (record those in a
`NumericalErrorBudget`), and code verification by the method of manufactured
solutions is only exercised in the tests. The model-form component is an
additive Gaussian discrepancy with a declared extrapolation rule, not a
Gaussian-process discrepancy; probability-box propagation is not implemented.
Other outcomes are individually tracked in the
[ATHENS coverage matrix](../curriculum/athens-2026-coverage.md); a tracked topic
is not represented as completed until tested implementation evidence exists.

## Sources

- [REF-SMITH-UQ-2024, Chapters 1-4 and 8].
- [REF-OBERKAMPF-ROY-2025, Chapters 1-2, 6-8, and Part IV].
- [REF-ROACHE-1998, Chapters 4-5] — GCI and factor of safety.
- [REF-CELIK-ETAL-2008] — three-grid procedure with non-uniform ratios.
- [REF-ROY-2005] — convergence behaviour and Richardson-based estimates.
- [REF-ECA-HOEKSTRA-2014] — least-squares observed order (simplified here).
- [REF-ROY-OBERKAMPF-2011] — separate numerical, model-form, and parametric
  uncertainty in predictions.
- [REF-JCGM-100-2008, Sections 4.3.7 and 6.2] — Type B conversions.
