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

## Seeds and streams

`foundations.RandomStream` derives independent child seeds by SplitMix64
mixing [EQ-RNG-SPLITMIX-001]. Workflows ask for named children
(`"propagation"`, `"sobol-design"`, `"sobol-bootstrap"`, ...), so one root seed
reproduces an entire study while different stages never share a stream by
accident. Hand-picked seeds such as `310` and `311` are no longer necessary.

## Cost accounting

`ProbabilisticModel.evaluations` counts every forward-model call made through
the model, and each workflow reports the calls it charged. For expensive
simulators this number, not wall time, is the honest measure of cost.

## Tails, information, and design inside one `UQStudy`

`workflow.UQStudy` (ADR-0025) runs every analysis on one model from one root
seed. Three more stages (ADR-0038) put extreme-value analysis, information
theory, and experimental design in the same study, the same report, and the
same evaluation count. Each one calls a single domain function. Each draws
from its own child stream `root.child(stage, k)`.

| Stage method | Stage kind | Delegates to | New runs |
|---|---|---|---|
| `extremes(q, tail=..., level=..., return_periods=...)` | `extremes` | `risk.fit_pot`, `PeaksOverThresholdFit.return_level` | 0 if it reuses the latest propagation |
| `information_sensitivity(q, max_points=512)` | `information` | `information.ksg_mutual_information` per input | 0 if it reuses the latest propagation |
| `design(candidates, size, noise=None)` | `design` | `design.greedy_design`, `design.expected_information_gain` | EIG cost |

- **`extremes`** fits a peaks-over-threshold (POT) model to the propagated
  sample [EQ-EVA-GPD-001] [EQ-EVA-POT-001].
  - Each model run counts as one observation, so a return period of `T`
    means `T` runs. The `T`-run level has tail probability `1/T`.
  - `tail="lower"` analyzes `-Y`, then maps results back to QoI units.
  - Given a `level`, the summary puts the POT tail probability next to the
    empirical fraction and its Wilson interval.
  - The binomial rate model assumes independent runs. For a Sobol or LHS
    sample, treat its intervals as references only.
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
