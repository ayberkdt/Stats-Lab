# Probability Foundations

## Definitions

- An **experiment** is a repeatable process with an uncertain outcome.
- The **sample space** `Omega` is the set of all possible outcomes.
- An **event** is a member of a chosen sigma-algebra `F` of subsets of `Omega`.
- A **probability measure** `P` maps events to the interval `[0, 1]`.

For a probability space `(Omega, F, P)`, the axioms are:

1. non-negativity: `P(A) >= 0` for every `A in F`;
2. normalization: `P(Omega) = 1`; and
3. countable additivity: for pairwise disjoint events `A_i`,
   `P(union_i A_i) = sum_i P(A_i)`.

Together these form [EQ-PROB-001]. They are structural assumptions from which
derived rules follow; they are not empirical estimates. See
[REF-KOLMOGOROV-1956, Chapter I, Section 2].

## Immediate consequences

- `P(empty set) = 0` and `P(A complement) = 1 - P(A)` [EQ-PROB-003].
- If `A` is a subset of `B`, then `P(A) <= P(B)`.
- `P(A union B) = P(A) + P(B) - P(A intersection B)` [EQ-PROB-004].

The event-algebra implementation represents every event as an immutable subset
owned by one sample space. Operations across different spaces are rejected;
this prevents set expressions that look valid in Python but have no declared
probabilistic universe.

## Finite weighted spaces

For a finite sample space, assign every outcome `omega` a non-negative mass
`p(omega)` whose total is one. Event probability is the sum of its outcome
masses [EQ-PROB-002]. Equal weights are a special case, not a requirement.

```python
from probstat.probability import FiniteProbabilityMeasure, FiniteSampleSpace

die = FiniteSampleSpace((1, 2, 3, 4, 5, 6))
fair = FiniteProbabilityMeasure.uniform(die)
even = die.event((2, 4, 6))
assert fair.probability(even) == 0.5
```

The full power set is used as the event sigma-algebra in this finite module.
The later measure-theory layer will make measurable-space selection explicit.

## Finite equally likely models

When a finite sample space has equally likely outcomes,
`P(A) = |A| / |Omega|`. Counting is therefore useful, but this ratio is a
special model—not the definition of probability in general.

The code includes factorials, permutations, combinations, and the multiplication
rule. Their exact equations are registered as `EQ-COMB-001` through
`EQ-COMB-004`.

## Conditioning and independence

Conditional probability [EQ-PROB-005] restricts attention to a positive-
probability event and renormalizes. Conditioning on a zero-probability event is
not defined by this elementary ratio and raises an error.

Independence is the symmetric product criterion [EQ-PROB-006]. Disjointness is
not independence: two disjoint positive-probability events cannot be independent
because their intersection has probability zero while the product of their
probabilities is positive.

## Exhaustive cases and Bayesian reversal

The law of total probability [EQ-PROB-007] combines conditional models across a
partition. Bayes' rule [EQ-PROB-008] reverses the conditioning direction. The
implementation validates partition disjointness, coverage, ownership, and
positive cell probability rather than accepting a merely list-shaped input.

For example, a rare condition can still have a modest posterior probability
after a positive test when false positives are common relative to the base rate.
This is why Bayes' rule must retain `P(H)` and `P(E)` rather than comparing only
the likelihood `P(E | H)`.

## Finite expectation bridge

Expectation [EQ-PROB-009] is available for finite real-valued outcome functions.
This provides a mathematically useful bridge, while the next round will add an
explicit immutable random-variable abstraction, transformations, moments, and
joint behavior.

## Common misunderstanding

A probability density may exceed one. Only its integral over a measurable set
is a probability and must lie in `[0, 1]`. In contrast, a discrete PMF value is
itself a point probability.

References: [REF-KOLMOGOROV-1956, Chapter I, Section 2],
[REF-ROSS-2019, Chapters 2-4].

