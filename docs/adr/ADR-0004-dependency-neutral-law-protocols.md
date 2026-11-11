# ADR-0004: Dependency-Neutral Probability-Law Protocols

- Status: Accepted
- Date: 2026-10-09

## Context

Analytic objects such as `Bernoulli` and `Normal` and induced finite random-
variable laws share useful behavior. Importing either domain package from the
other would blur ownership and create a future circular-dependency route. A
nominal base class would also force unrelated implementations into inheritance.

## Decision

Define minimal runtime-checkable structural protocols in
`probstat.foundations.protocols`: cumulative distribution, finite mean/variance,
discrete probability law, and continuous probability law. Domain objects conform
by implementing methods and properties; they do not inherit from or import one
another.

Protocols specify shape, not mathematical validity. Concrete implementations
retain responsibility for parameter constraints, equation IDs, numerical
methods, support semantics, and error behavior.

## Consequences

- Generic consumers can accept analytic and induced laws through one interface.
- Package dependencies remain downward and acyclic.
- Runtime protocol checks establish attribute availability, not semantic proof.
- Future multivariate, sampleable, quantile, survival, and log-probability
  protocols must remain narrow rather than forming one oversized interface.

