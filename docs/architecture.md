# Architecture

## Dependency direction

The repository uses a directed acyclic package graph. An arrow means "may
depend on."

```text
probstat.visualization ──> probstat.distributions ──> probstat.foundations
          │                                           ^
          └────────────> probstat.statistics ──────────┤
                                                      │
probstat.probability ──────────────────────────────────┘
```

`probstat.__init__` exposes version metadata only. This deliberate choice keeps
domain ownership visible and avoids a root import hub that could become a
cycle.

## Layer responsibilities

| Layer | Owns | Must not own |
|---|---|---|
| `foundations` | Validation, scalar numerical primitives, shared protocols | Domain distributions or estimators |
| `probability` | Sets, counting, probability rules, random-variable algebra | Inference or plotting |
| `statistics` | Descriptive statistics, estimators, diagnostics | Distribution implementations or plotting |
| `distributions` | Distribution objects, density/mass/CDF behavior, sampling | Educational rendering |
| `visualization` | Optional plots and explanatory annotations | Mathematical truth or validation rules |

Sibling domain packages do not import one another unless an ADR introduces a
clear higher-level layer. Cross-cutting code moves down only when it is truly
domain-independent.

## Planned layer growth

```text
frontier systems
  └─ probabilistic programming / generative systems / causal systems
      └─ stochastic processes / Bayesian computation / asymptotics
          └─ inference / multivariate analysis / information theory
              └─ distributions / statistics / probability
                  └─ measure and numerical foundations
```

Each new layer may depend only on layers below it. Mature public interfaces are
promoted deliberately; experimental frontier work begins under an explicit
`experimental` namespace and cannot leak into foundations.

## Enforcement

`tests/test_architecture.py` builds a module-level import graph from Python's
AST. It rejects both upward imports and cycles. `tests/test_documentation_contract.py`
checks docstrings and all `EQ-*` and `REF-*` identifiers used by source code.

This is an initial structural guard, not a formal proof. Future rounds will add
API stability tiers and generated dependency diagrams.

