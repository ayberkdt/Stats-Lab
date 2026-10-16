# The central probabilistic model

Every uncertainty-quantification workflow in this library asks questions about
the same object: uncertain inputs `X`, a forward model `Y = g(X)`, and the
quantities of interest (QoIs) that carry decisions. Before this layer existed,
propagation took named mappings, reliability took vectors in a
limit-state function, sensitivity took a bare callable, and calibration only
knew a straight line. A study therefore had to restate its model for every
method. `probstat.modeling.ProbabilisticModel` removes that duplication.

```python
import math

from probstat.distributions import LogNormal, Normal, Uniform
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.uq import QuantityOfInterest

inputs = JointInput.from_pearson(
    (
        RandomInput("load", LogNormal(0.0, 0.25), unit="kN"),
        RandomInput("strength", Normal(4.0, 0.4), unit="kN"),
        RandomInput("angle", Uniform(0.0, math.pi / 6), unit="rad"),
    ),
    ((1.0, 0.3, 0.0), (0.3, 1.0, 0.0), (0.0, 0.0, 1.0)),
)

def bracket(p):
    return {"margin": p["strength"] - p["load"] / math.cos(p["angle"])}

model = ProbabilisticModel(
    inputs, bracket, (QuantityOfInterest("margin", "capacity minus demand", "kN"),)
)
```

The same `model` is now accepted by `propagation.propagate`,
`sensitivity.sobol_analysis`, `sensitivity.morris_screening`, the model-based
reliability methods, the risk layer, nonlinear calibration (as a prior and a
forward map), and surrogate fitting.

## Inputs: marginals plus a copula

`RandomInput` names one uncertain input and its marginal law. Any law with
`pdf`, `cdf`, `quantile`, `mean`, and `variance` qualifies.

`JointInput` adds dependence through a **Gaussian copula** with latent
correlation `R`. Independence is `R = I`. The isoprobabilistic **Nataf
transformation** [EQ-MOD-NATAF-001] maps physical inputs to independent
standard normals:

```text
z_i = Phi^-1(F_i(x_i)),   u = L^-1 z,   R = L L'
```

Reliability methods work in `u`; samplers push unit-hypercube designs back
through the inverse map; MCMC priors use the copula density
[EQ-MOD-COPULA-001]. One law, one transformation, every method.

**Latent versus physical correlation.** `R` is the correlation of the latent
normals, not of the physical inputs. `JointInput.from_pearson` solves the
Nataf integral equation [EQ-MOD-NATAF-002] pairwise so that the *physical*
Pearson correlations match a target. For two lognormal inputs with log-scale
standard deviations 0.5 and 0.8 and a target Pearson correlation of 0.6, the
solver returns `rho_Z = 0.6612152317384`; the closed form
`ln(1 + 0.6 sqrt((e^0.25-1)(e^0.64-1))) / 0.4` gives `0.6612152317385`.
A 30 000-draw sample reproduces a physical correlation of about 0.59–0.60.
Some targets are unattainable for skewed marginals (for example a strong
negative correlation between two heavy lognormals); the solver reports that
instead of returning a wrong matrix.

## Designs: how inputs are sampled

`probstat.sampling` produces points in the open unit cube `(0,1)^d`:

| Design | Function | Property |
|---|---|---|
| Monte Carlo | `monte_carlo_design` | IID; error `O(N^-1/2)` |
| Latin hypercube | `latin_hypercube` | one point per marginal stratum [EQ-SAMP-LHS-001] |
| Halton | `halton(scramble=..., shift=...)` | low discrepancy, prime bases [EQ-SAMP-HALTON-001] |
| Sobol | `sobol(scramble=...)` | base-2 digital net, 16 dimensions [EQ-SAMP-SOBOL-001] |

Randomizing a quasi-random design (digital shift, Cranley-Patterson shift)
keeps its uniformity while making the estimator unbiased [EQ-SAMP-RQMC-001],
so independent replicates give an honest error estimate.

For the Ishigami function (three uniform inputs on `[-pi, pi]`, exact mean
3.5) and 256 points per estimate, eight independent replicates gave these
standard deviations of the mean estimate (`propagation.compare_designs`,
seed 1):

| Design | Spread of the mean |
|---|---|
| Monte Carlo | 0.227 |
| Latin hypercube | 0.100 |
| scrambled Halton | 0.048 |
| scrambled Sobol | 0.0037 |

Same model, same cost, about 60 times less error with scrambled Sobol. The
advantage shrinks for non-smooth or very high-dimensional integrands, which is
why the design is a parameter and not a hidden default.

## Forward propagation on the model

```python
from probstat.propagation import propagate

study = propagate(model, 4096, design="sobol", seed=7)
margin = study["margin"]
margin.mean, margin.standard_deviation
margin.quantile(0.05)          # Hyndman-Fan type 7 [EQ-STAT-QUANT-001]
margin.exceedance(0.0)          # P(Y > 0) with a Wilson interval
margin.exceedance_curve(50)     # (level, P(Y > level)) pairs
study.rank_correlations("margin")  # Spearman screening per input
```

For Ishigami with 4096 Sobol points this gives a mean of 3.500004 and a
standard deviation of 3.7213 against the exact 3.7208.

When the sample size is not known in advance, `UQStudy.propagate_until(q,
target_relative_error=0.01, threshold=None, batch=256, max_samples=200_000)`
(stage kind `sequential_propagation`, ADR-0056) draws plain Monte Carlo
batches and stops at the first batch whose relative standard error meets the
target [EQ-STAT-STREAM-007]: `s / (sqrt(n) |mean|)` for the mean, or
`sqrt((1 - p) / (n p))` for `P[Y > t]` (the same strict event as
`OutputDistribution.exceedance`; `failure="below"` counts the complement
`Y <= t`) when a threshold is given [EQ-MC-PROB-001]. Mean, variance, P²
quantiles, and the exceedance count with its Wilson interval are streaming
accumulators, so memory does not grow with `n`. For `Y ~ N(10, 2)`
(coefficient of variation 0.2) the theory `n = (cv / target)^2` gives 400 runs
at a 1% target; seeds 1, 2, 3 with `batch=50` stopped at 500, 400, and 400.

Stopping when the running estimate looks precise is data-dependent, so the
estimates of the sequential sample itself are biased and their intervals
under-cover (optional stopping). Since ADR-0058 the sequential run is only a
pilot that chooses `n`; with `final="fresh"` (default) every reported number
comes from a fresh, independent sample of size `n` [EQ-WF-SEQ-FRESH-001],
which is an ordinary fixed-size sample given `n`. In 300 seeded runs of
`Y ~ N(1, 0.5)` reusing the pilot over-estimated `P(Y > 1.64) = 0.100` by
4.9% (4 standard errors) and the mean by 1.0%; the fresh estimates were
unbiased within one standard error. `final="reuse"` keeps the old,
cheaper behaviour with a warning. `confidence=` sets the level of the
Student-t mean interval and the Wilson interval. If `max_samples` is reached
first the stage reports `converged = no`.

Numerical (discretization) uncertainty enters the study through
`UQStudy.verify(q, grid_values, refinement_ratio)` (or `grid_sizes=`; stage
kind `verification`, ADR-0057). It runs `probstat.uq.verify_solution` on QoI
values the user computed on three or more grids: the three-grid GCI
[EQ-VV-GCI-004] or the oscillation bound [EQ-VV-OSC-001], and least squares
for four or more grids [EQ-VV-LSQ-002]. It records the observed order, the
extrapolated value, `U`, `u_num` (`U / sqrt(3)` by default [EQ-VV-UNUM-001]),
and the asymptotic-range ratio at the theoretical order [EQ-VV-GCI-005]. No
model evaluations are charged. `validate(..., numerical_uncertainty=
"verified")` then adds that `u_num` to each ASME V&V 20 comparison,
`u_val = sqrt(u_num^2 + u_input^2 + u_D^2)` [EQ-VAL-ASME-001]. The area
verdict, u-pooling, CRPS, and PIT do not change. For
`phi(h) = 2 + 0.03 h^2 + 0.004 h^3` on `h = 1, 2, 4` (values 2.034, 2.152,
2.736) the stage gives observed order 2.307, extrapolated 2.00412,
`GCI_fine = 0.03735`, and `u_num = 0.02156`. The ratio at order 2 is 1.237,
so the stage warns that the grids are outside the asymptotic range. In the
spring validation of `tests/test_workflow_verify.py` (seed 6, 400 samples),
`u_val` grows from 0.1024 to 0.1046 at load 1 and from 0.2958 to 0.2966 at
load 3. Input uncertainty dominates there. `predict` does not add `u_num`;
pass the `SolutionVerification` that `verify` returns as `numerical=` to
`probstat.validation.predict_with_uncertainty_components` explicitly.

## Seeds and streams

`foundations.RandomStream` derives independent child seeds by SplitMix64
mixing [EQ-RNG-SPLITMIX-001]. Workflows ask for named children
(`"propagation"`, `"sobol-design"`, `"sobol-bootstrap"`, ...), so one root seed
reproduces an entire study while different stages never share a stream by
accident. Hand-picked seeds such as `310` and `311` are no longer necessary.

## Cost accounting

`ProbabilisticModel.evaluations` counts every *completed* forward-model call
made through the model, and each workflow reports the calls it charged. For
expensive simulators this number, not wall time, is the honest measure of
cost. Failed attempts (`model.failed_evaluations`) and cache hits
(`model.cache_hits`) are counted separately; `dataclasses.replace(model,
...)` gives the new model its own counter, linked to the source counter so the
source still accounts for every run of the shared simulator.

## Failed runs, batch evaluation, and study provenance (ADR-0058)

- **Failure policy.** `ProbabilisticModel(..., on_failure="raise")` keeps the
  original behaviour; `"skip"` drops a failed run (exception, missing QoI,
  NaN or infinity) and logs its error text; `"record"` also keeps its
  inputs. `propagate` and the study stages keep the successful runs, list the
  failures, and report the failure fraction. Dropping failures conditions
  every estimate on success: it is unbiased only when failing is unrelated to
  the output.
- **Strict inputs.** `model.evaluate(mapping)` rejects missing, unknown, and
  fixed-parameter names. `fixed=` may hold strings, integers, Booleans, and
  tuples; they reach the model unchanged.
- **Engine.** `model.evaluate_many(points, executor=..., batch=...,
  cache=...)` runs in a `concurrent.futures` executor (results keep input
  order), calls a model's `evaluate_batch(list_of_mappings)` hook when it
  has one, and reuses runs from an `EvaluationCache`. `UQStudy(...,
  executor=..., cache=True)` installs both for every stage.
- **Accuracy.** `propagate(..., replicates=r)` gives Student-t intervals over
  `r` independent randomizations [EQ-PROP-RQMC-CI-001], the honest error bar
  for LHS and QMC; `quantile_interval` is the distribution-free
  order-statistic interval [EQ-PROP-QCI-001]; `variance_interval` is the
  chi-square interval (normal outputs only) or a bootstrap
  [EQ-PROP-VARCI-001].
- **Chaining.** `study.adopt_posterior()` replaces the input law by a
  marginals-plus-Gaussian-copula approximation of the latest calibration
  posterior; `study.with_inputs(law)` sets any law. Every stage record states
  the law it used (label and hash).
- **Budget, persistence.** `UQStudy(..., evaluation_budget=N)` refuses
  stages with a known cost above the remainder and stops the others at the
  limit. `study.to_json()` and `study.save_json(path)` write every stage with
  its settings, warnings, cost, and the library, Python, platform, time, and
  root-seed provenance; models, results, and studies pickle.

## Tails, information, and design inside one `UQStudy`

`workflow.UQStudy` (ADR-0025) runs every analysis on one model from one root
seed. Three more stages (ADR-0038) put extreme-value analysis, information
theory, and experimental design in the same study, the same report, and the
same evaluation count. Each one calls a single domain function. Each draws
from its own child stream `root.child(stage, k)`.

| Stage method | Stage kind | Delegates to | New runs |
|---|---|---|---|
| `extremes(q, tail=..., level=..., return_periods=...)` | `extremes` | `risk.fit_pot`, `PeaksOverThresholdFit.return_level` | `size` (default 4096) fresh Monte Carlo runs; 0 with `reuse=True` after a plain Monte Carlo propagation |
| `information_sensitivity(q, max_points=512)` | `information` | `information.ksg_mutual_information` per input | `size` (default `max_points`) fresh Monte Carlo runs; 0 with `reuse=True` after a plain Monte Carlo propagation |
| `design(candidates, size, noise=None)` | `design` | `design.greedy_design`, `design.expected_information_gain` | EIG cost |

- **`extremes`** fits a peaks-over-threshold (POT) model to the propagated
  sample [EQ-EVA-GPD-001] [EQ-EVA-POT-001].
  - Each model run counts as one observation, so a return period of `T`
    means `T` runs. The `T`-run level has tail probability `1/T`.
  - `tail="lower"` analyzes `-Y`, then maps results back to QoI units.
  - Given a `level`, the summary puts the POT tail probability next to the
    empirical fraction and its Wilson interval.
  - The binomial rate model assumes independent runs, so the stage never
    reuses a Sobol or LHS sample (ADR-0058): results do not depend on
    which stages ran before.
- **`information_sensitivity`** estimates `I(X_i; Y)` in nats with the KSG
  estimator [EQ-INFO-KSG-001].
  - It is zero only under independence and does not change under monotone
    transforms. That makes it a moment-independent complement to Sobol
    indices. It is not a variance share, and the values do not sum to one.
  - The estimator costs `O(n^2)` per input, so the stage uses the first
    `max_points` runs.
- **`design`** reuses the study's calibration problem: the prior, the
  forward model, and the known Gaussian noise.
  - The calibration experiments become the baseline. Conditions are then
    added greedily by expected information gain (EIG) [EQ-OED-GREEDY-001].
  - The full design is re-estimated on an independent stream, so the
    report gives the EIG with a standard error.
  - The EIG is a preposterior quantity under the prior. It does not
    condition on the values that were observed.
  - With no calibration, `design` needs `noise=`; otherwise it raises
    `ValueError`.

Results from `examples/uq_study_extensions_lab.py` (seed 2026, about 12 s):

- **Cantilever, 20 000 MC runs, threshold at the 95% quantile.**
  - Fit: `xi = -0.057 ± 0.033`.
  - `P(deflection > 0.075 m)`: POT `7.67e-3`, empirical `7.80e-3`
    (Wilson 95% `[6.67e-3, 9.12e-3]`).
  - 10 000-run return level: `99.3 mm` (`[94.4, 104.3]`), against an
    empirical `96.5 mm`.
  - The stage made no new model runs.
- **Ishigami, 1024-row Sobol design plus 800 MC runs for KSG (`k = 3`).**

  | Input | `S` | `ST` | MI (nats) |
  |---|---|---|---|
  | `x1` | 0.305 | 0.555 | 0.351 |
  | `x2` | 0.443 | 0.441 | 0.446 |
  | `x3` | -0.001 | 0.245 | 0.142 |

  `x3` acts on `Y` only through its interaction with `x1`. Its first-order
  Sobol index is zero, but its mutual information is clearly positive.
- **Decay `A exp(-k t)`, calibrated on two observations at `t = 0, 1`.**
  - Laplace EIG with 64 prior draws picks `t = 3`, then `t = 2`, from eight
    candidates.
  - Final EIG of the four-condition design: `6.50 ± 0.14` nats.
  - The greedy path reports `6.81` for the same design. That value is the
    maximum over candidates scored with one stream, so it is biased upward.
    The independent re-estimate is the honest figure.

## Discrepancy, scoring, and figures in the same study

ADR-0045 closes the remaining gaps, so every analysis of this release runs
from one `UQStudy` and appears in one report.

| Addition | What it records |
|---|---|
| `calibrate_with_discrepancy(experiments, q, noise, discrepancy)` (stage kind `discrepancy_calibration`) | Kennedy-O'Hagan posterior of `(theta, sigma_delta, ell)` [EQ-KOH-POST-001], the identifiability report [EQ-KOH-IDENT-001], and simulator-only and bias-corrected predictions on a grid for the figure |
| `predict` after it | simulator-only, bias-corrected, and (with `include_noise`) observed intervals per condition [EQ-KOH-PRED-001] |
| `validate` | CRPS per experiment and overall [EQ-VAL-SCORE-CRPS-002], randomized ensemble PIT mean [EQ-VAL-CAL-PIT-002], coverage at 50% and 90% with Clopper-Pearson intervals [EQ-VAL-CAL-COV-001]; the verdict criteria are unchanged |
| `TailAnalysis.exceedances`, `DesignRecommendation.candidates` | the data the tail and greedy-design figures need |

The plain `calibrate` is unchanged and bit-identical. The study uses the
latest calibration of either kind for `predict` and `design`. After a
discrepancy calibration, `design` needs explicit `noise=` models, because the
Bayesian design problem has no discrepancy term.

Results from `examples/uq_study_complete_lab.py` (seed 2026, 17 s,
139 887 model runs). The model is `y = A exp(-k x)` with `A ~ N(4, 0.4)` and
`k ~ LogNormal(log 0.5, 0.25)`. The data come from
`4.2 exp(-0.45 x) + 0.15 sin(2x)` plus noise with sd 0.03 at 11 points on
`[0, 4]`.

- **Two design points** of `|y(1) - 4 e^-0.5| >= 0.9`, at `beta = 2.38` and
  `2.28`.
  - Multi-point FORM: `2.00e-2`, 2.3 times the single-point value
    `8.70e-3`.
  - Mixture importance sampling: `2.05e-2` (c.o.v. 0.04).
  - AK-MCS: `2.08e-2` from 26 true-model runs, equal to crude Monte Carlo on
    the same 4000 points.
- **Plain calibration** is confident and wrong.
  - It gives `k = 0.455 ± 0.004`.
  - At `x = 2` its 95% interval `[1.694, 1.738]` misses the truth `1.594`.
- **Kennedy-O'Hagan calibration** gives `k = 0.420 ± 0.037` and
  `sigma_delta = 0.20`.
  - Max `|rho(theta, log psi)|` is 0.37. Posterior/prior sd ratios are 0.45
    for `A` and 0.29 for `k`, so no confounding is flagged.
  - At `x = 2` the simulator-only interval is `[1.428, 1.993]`. The
    bias-corrected interval is `[1.548, 1.632]` and covers the truth.
  - At `x = 5`, outside the data, the bias-corrected interval widens to
    `[0.281, 0.935]`. It still covers the truth `0.361`.
- **Validation** of the prior model at four conditions (5 replicates each)
  is inconsistent.
  - Mean CRPS is 0.165.
  - PIT mean is 0.74: observations sit high in the predictive laws.
  - 90% coverage is 20/20 (`[0.832, 1.000]`), so the intervals are too wide.
    50% coverage is 7/20.

`visualization.study` now also draws:

- the standard-normal plane of the design points, with their FORM
  hyperplanes and the smallest-`beta` circle;
- the AK-MCS estimate and `min U` against training runs;
- the posterior marginals of a calibration over their prior densities;
- the two discrepancy figures of ADR-0044;
- the greedy candidate curves of a one-factor design.

## What the model does not decide

- It does not validate the forward model. Validation remains a separate
  evidence question (`probstat.validation`).
- A Gaussian copula cannot represent tail dependence. If joint extremes
  matter, record that limitation.
- Variance-based Sobol indices are refused for dependent inputs, because the
  functional ANOVA decomposition behind them assumes independence.
