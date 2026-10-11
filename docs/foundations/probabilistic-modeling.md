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

## What the model does not decide

- It does not validate the forward model. Validation remains a separate
  evidence question (`probstat.validation`).
- A Gaussian copula cannot represent tail dependence. If joint extremes
  matter, record that limitation.
- Variance-based Sobol indices are refused for dependent inputs, because the
  functional ANOVA decomposition behind them assumes independence.
