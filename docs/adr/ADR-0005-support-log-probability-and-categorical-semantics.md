# ADR-0005: Support, Log Probability, and Categorical Semantics

- Status: Accepted
- Date: 2026-10-09

## Context

Distribution growth needs reusable parameter checks and support descriptions.
It also needs stable log-probability evaluation. A single large distribution
interface would incorrectly require unordered categorical labels to provide a
CDF and numeric moments.

## Decision

Keep immutable parameter constraints and support descriptors inside the
`distributions` layer, where their semantics are currently owned. Expose narrow
dependency-neutral PMF and log-PMF protocols from `foundations`. Build the
larger discrete-law protocol from PMF, CDF, and finite-moment behaviors.

Treat category labels as unordered and non-numeric unless a caller explicitly
supplies a separate encoding. Therefore `Categorical` provides PMF, log-PMF,
event probability, modes, entropy, and sampling, but not CDF, mean, or variance.

Use direct log-space formulas and complementary tail functions when these avoid
overflow, underflow, or cancellation. Support descriptors do not silently
truncate infinite support for display or computation.

## Consequences

- Binomial, Categorical, and Poisson share validation and support vocabulary.
- Generic PMF consumers accept categorical laws without claiming they are
  ordered numeric laws.
- Plots of infinite-support laws require an explicit window.
- A future measure-theoretic foundation may generalize these support objects;
  promotion requires a separate ADR rather than an upward dependency.
- Large-scale binomial tails and wider special-function coverage remain future
  numerical work.
