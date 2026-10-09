# Probability and Statistics Library

An English-language, citation-first Python library that builds probability and
statistics from auditable foundations toward modern inference and frontier
probabilistic systems.

The project has two equally important outputs:

1. trustworthy, modular software; and
2. explanations that make every implemented equation traceable and teachable.

## Foundation releases

Release `0.2.0` provides:

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

die = FiniteSampleSpace((1, 2, 3, 4, 5, 6))
fair = FiniteProbabilityMeasure.uniform(die)
even = die.event((2, 4, 6))

print(fair.probability(even))  # 0.5
print(fair.expectation(float))  # 3.5
```

Each public function links its formula to an ID such as `EQ-STAT-001`. Look up
that ID in [the equation registry](docs/equation-registry.md), then follow its
citation key to [the bibliography](docs/references.md).

## Read in this order

1. [Learning path](docs/learning-path.md)
2. [Architecture](docs/architecture.md)
3. [Equation registry](docs/equation-registry.md)
4. [Contribution contract](CONTRIBUTING.md)
5. [Roadmap](memory/ROADMAP.md)

## Design promise

Ease of explanation never removes mathematical scope. Concepts are introduced
in layers, with definitions, assumptions, equations, examples, and visuals,
while the roadmap preserves the path to measure-theoretic probability,
inference, stochastic processes, probabilistic programming, and frontier
generative systems.

