# Architecture

## Dependency direction

The repository uses a directed acyclic package graph. An arrow means "may
depend on."

```text
visualization
  ├─> propagation ─> monte_carlo ─> inference ─┐
  │        ├─> uq ──────────────────────────────┤
  │        └─> numerics ────────────────────────┤
  ├─> multivariate ─> distributions ────────────┤
  ├─> statistics ───────────────────────────────┤
  └─> probability ──────────────────────────────┤
                                               v
                                          foundations
```

`probstat.__init__` exposes version metadata only. This deliberate choice keeps
domain ownership visible and avoids a root import hub that could become a
cycle.

## Layer responsibilities

| Layer | Owns | Must not own |
|---|---|---|
| `foundations` | Validation, scalar numerical primitives, shared protocols | Domain distributions or estimators |
| `numerics` | Domain-neutral deterministic algorithms, work counts, and numerical diagnostics | Probability models or UQ workflows |
| `probability` | Sets, counting, probability rules, random-variable algebra | Inference or plotting |
| `statistics` | Descriptive statistics, estimators, diagnostics | Distribution implementations or plotting |
| `distributions` | Distribution objects, density/mass/CDF behavior, sampling | Educational rendering |
| `uq` | Cross-cutting uncertainty vocabulary, QoI contracts, numerical verification | Monte Carlo, reliability, inference, or domain-specific solvers |
| `multivariate` | Vector-valued distributions and covariance-aware probability operations | Inference workflows or plotting |
| `inference` | Estimation, confidence procedures, tests, and diagnostics | Rendering, simulation orchestration, or application models |
| `monte_carlo` | Sample estimators, sampling-error diagnostics, and evaluation accounting | Forward-model semantics or plotting |
| `propagation` | QoI sample propagation, expectation integration, output summaries, thresholds | Distribution implementations or rendering |
| `visualization` | Optional plots and explanatory annotations | Mathematical truth or validation rules |

Structural interfaces such as `DiscreteProbabilityLaw` and
`ContinuousProbabilityLaw` live in `foundations.protocols`. Both analytic
distribution objects and induced finite laws conform by structure; neither
domain package imports the other. Protocol conformance does not replace each
implementation's equation, domain, or numerical documentation.

Narrow `ProbabilityMassFunction` and `LogProbabilityMassFunction` protocols let
unordered categorical laws participate in generic mass-based algorithms without
claiming a CDF or numeric moments. Parameter constraints, support descriptors,
and special-function algorithms stay inside `distributions`; they may depend on
foundation validators but never on probability or statistics domain packages.

Density, log-density, survival, quantile, and hazard are also independent narrow
protocols. A type implements only capabilities that are mathematically defined
and numerically supported. Direct survival functions are not replaced by a
generic `1-CDF` adapter because that would erase tail-accuracy guarantees.

Sibling domain packages do not import one another unless an ADR introduces a
clear higher-level layer. Cross-cutting code moves down only when it is truly
domain-independent.

## Planned layer growth

```text
frontier / probabilistic programming / causal systems
  └─ filtering / smoothing / data assimilation
      └─ reliability / sensitivity / inverse problems
          └─ propagation / Monte Carlo / inference
              └─ UQ contracts / multivariate distributions
                  └─ probability / statistics / distributions
                      └─ numerical / linear algebra / optimization foundations
```

Each new layer may depend only on layers below it. Mature public interfaces are
promoted deliberately; experimental frontier work begins under an explicit
`experimental` namespace and cannot leak into foundations.

The `uq` package stays narrow: it defines concepts shared across workflows and
solution-verification primitives. `monte_carlo` owns estimator behavior without
knowing forward-model semantics; `propagation` combines those estimators with
QoIs and explicit model calls. Dedicated packages will likewise own
reliability, sensitivity, inverse problems, filtering, and data assimilation,
so `uq` does not become a dependency hub. The complete target graph and its
delivery order are recorded in ADR-0007 and the ATHENS coverage matrix.

## Enforcement

`tests/test_architecture.py` builds a module-level import graph from Python's
AST. It rejects both upward imports and cycles. `tests/test_documentation_contract.py`
checks docstrings and all `EQ-*` and `REF-*` identifiers used by source code.
`tests/test_athens_coverage.py` prevents curriculum rows from disappearing or
claiming completion without repository evidence.

This is an initial structural guard, not a formal proof. Future rounds will add
API stability tiers and generated dependency diagrams.

