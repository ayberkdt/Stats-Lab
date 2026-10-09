# ADR-0012: Sensitivity estimator boundaries

- Status: accepted
- Date: 2026-10-09

## Decision

Local derivatives retain point and perturbation metadata. Variance-based global
analysis requires two independent, equally shaped base matrices and explicitly
assumes mutually independent inputs. Sobol estimates remain unclipped so Monte
Carlo error is visible. Pair interactions are evaluated rather than inferred
from first and total effects. Ranking and screening use total effects, with a
declared threshold and an explicit warning that the result is not causal or an
economic optimum.

## Consequences

The API cannot silently apply independent-input Sobol semantics to correlated
inputs. Model-call costs include base, single-column, and pair-hybrid runs.
Bootstrap intervals, Shapley effects, and stochastic-model replication remain
separate future extensions.
