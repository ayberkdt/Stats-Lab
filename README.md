# Probability and Statistics Library

An English-language, citation-first Python library that builds probability and
statistics from auditable foundations toward modern inference and frontier
probabilistic systems.

The project has two equally important outputs:

1. trustworthy, modular software; and
2. explanations that make every implemented equation traceable and teachable.

## Foundation releases

Release `0.8.0` adds:

- Lognormal, Weibull, and positive-definite multivariate Normal models with
  documented parameterizations and reproducible sampling;
- Normal and Lognormal MLE, KS/AD goodness-of-fit diagnostics, and explicit z,
  large-sample, and Wilson confidence procedures;
- fitted-CDF visualization and a seeded statistical-fitting laboratory; and
- dependency-free Cholesky algebra with singular-covariance rejection instead
  of hidden numerical jitter.

Release `0.7.0` added:

- a test-enforced matrix for all 110 supplied ATHENS 2026 VV-UQ topics, with
  honest `Implemented`, `Partial`, and `Planned` states;
- separate aleatory/epistemic and engineering-source classifications;
- immutable quantities of interest and a validated `Y = g(X)` forward-model
  contract; and
- numerical-error budgets, observed convergence order, and Richardson
  extrapolation with cited assumptions and boundary tests.

Earlier foundation releases provide:

- zero-based Geometric and Negative Binomial waiting-count laws with explicit
  parameterization contracts;
- continuous Uniform and rate-parameterized Exponential laws;
- direct survival, generalized quantile, density/log-density, and hazard
  protocols, with existing ordered laws extended where supported;
- a symmetry-aware regularized incomplete-beta implementation for stable
  Negative Binomial tails;
- coordinated waiting-count and PDF/CDF/survival teaching plots;

- reusable parameter constraints and explicit finite, integer, and real support
  descriptors;
- stable Binomial, Categorical, and Poisson distributions with log-PMFs,
  moments where mathematically defined, reproducible sampling, and documented
  boundary behavior;
- regularized incomplete-gamma tails for stable Poisson CDF and survival
  evaluation;
- narrow PMF/log-PMF protocols that preserve unordered categorical semantics;
- count-window and categorical teaching plots;

- probability-generating, moment-generating, and characteristic functions with
  explicit mathematical and floating-point domains;
- dependency-neutral runtime-checkable protocols shared by analytic and induced
  discrete/continuous laws;
- immutable finite random variables with induced PMFs and CDFs;
- exact transformations, expectation, raw/central moments, and variance;
- joint PMFs, marginalization, conditional PMFs, random-variable independence,
  covariance, and correlation;
- immutable finite sample spaces and owned event algebra;
- arbitrary finite probability measures and uniform measures;
- conditional probability, event independence, total probability, Bayes' rule,
  and finite expectation;
- validated counting primitives;
- descriptive statistics for finite samples;
- Bernoulli and normal distributions;
- optional plotting helpers for PMFs, PDFs, and empirical CDFs;
- an equation registry and a bibliography with stable citation keys;
- automated checks for dependency cycles, missing docstrings, missing equation
  records, and missing references; and
- a roadmap, decision log, and session memory.

## Install and verify

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
```

Plotting support is optional:

```powershell
python -m pip install -e ".[visualization]"
python examples/generate_foundation_figures.py
```

## Small example

```python
from probstat.probability import FiniteProbabilityMeasure, FiniteSampleSpace
from probstat.probability import FiniteRandomVariable

die = FiniteSampleSpace((1, 2, 3, 4, 5, 6))
fair = FiniteProbabilityMeasure.uniform(die)
even = die.event((2, 4, 6))

print(fair.probability(even))  # 0.5
print(fair.expectation(float))  # 3.5

roll = FiniteRandomVariable.from_function(fair, float, name="Roll")
print(roll.expectation)  # 3.5
print(roll.variance)  # 2.916666...
```

Each public function links its formula to an ID such as `EQ-STAT-001`. Look up
that ID in [the equation registry](docs/equation-registry.md), then follow its
citation key to [the bibliography](docs/references.md).

## Read in this order

1. [Learning path](docs/learning-path.md)
2. [VV-UQ foundations](docs/foundations/vv-uq-foundations.md)
3. [Uncertainty modeling and fitting](docs/foundations/uncertainty-modeling-and-fitting.md)
4. [ATHENS 2026 coverage matrix](docs/curriculum/athens-2026-coverage.md)
5. [Architecture](docs/architecture.md)
6. [Equation registry](docs/equation-registry.md)
7. [Contribution contract](CONTRIBUTING.md)
8. [Roadmap](memory/ROADMAP.md)

## Design promise

Ease of explanation never removes mathematical scope. Concepts are introduced
in layers, with definitions, assumptions, equations, examples, and visuals,
while the roadmap preserves the path to measure-theoretic probability,
inference, stochastic processes, probabilistic programming, and frontier
generative systems.

