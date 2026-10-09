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

- `P(empty set) = 0`.
- `P(A complement) = 1 - P(A)`.
- If `A` is a subset of `B`, then `P(A) <= P(B)`.
- `P(A union B) = P(A) + P(B) - P(A intersection B)`.

These consequences will receive code and separate registry IDs when the event
algebra module is implemented. They are included now so the conceptual base is
not reduced to the currently available functions.

## Finite equally likely models

When a finite sample space has equally likely outcomes,
`P(A) = |A| / |Omega|`. Counting is therefore useful, but this ratio is a
special model—not the definition of probability in general.

The code starts with factorials, permutations, combinations, and the
multiplication rule. Their exact equations are registered as `EQ-COMB-001`
through `EQ-COMB-004`.

## Common misunderstanding

A probability density may exceed one. Only its integral over a measurable set
is a probability and must lie in `[0, 1]`. In contrast, a discrete PMF value is
itself a point probability.

