# ADR-0071 — Finite measure bridge and explicit frequentist/Bayesian interpretation

- Status: Accepted
- Date: 2026-11-11
- Addresses (roadmap): Phase 1 "Measure/integration bridge: sigma-algebras,
  measurable maps, Lebesgue expectation, Radon-Nikodym derivatives, and
  product measures"; Phase 4 "Frequentist/Bayesian interpretations kept
  explicit and separate"; the Phase 4 note that completeness was discussed
  but not computed (ADR-0051, decision 10).

## Context

ADR-0002 fixed the full power set as the sigma-algebra of every
`FiniteSampleSpace` and deferred restricted sigma-algebras to a later
measure-theory layer. Without that layer the library could not express
information (a sub-sigma-algebra), measurability of a map, conditional
expectation given a sigma-algebra, or a density between two measures, and
the guides had to describe these ideas without executable counterparts.
Completeness of a statistic was documented only, because ADR-0051 judged a
numerical check to need a function-space search. Finally, intervals, tests,
and posteriors were each documented correctly in isolation, but no guide
set the two interpretations of the same numbers side by side.

## Decision

1. **One new layer-1 module, `probability.measure`.** It imports only
   `probability.finite` and the standard library. No new package or layer.
2. **Sigma-algebras are represented by atoms.** `FiniteMeasurableSpace`
   generates `sigma(G)` from any collection of subsets by grouping outcomes
   with equal generator-membership signatures. Events are unions of atoms,
   so `|sigma(G)| = 2^k` is never materialized unless `events()` is called.
   Constructors cover the power set, the trivial algebra, an explicit
   partition, and `sigma(f)` for a map `f`; `is_sigma_algebra` checks the
   closure axioms of an arbitrary family directly.
3. **Spaces compare by value, not identity.** A sub-sigma-algebra is by
   definition a second structure on the same set, so identity-based
   ownership (ADR-0002) cannot express it. Two `FiniteMeasurableSpace`
   objects are equal when their outcome sets and atoms are equal.
   `FiniteSampleSpace` keeps its identity semantics unchanged.
4. **Measures are atom masses.** `FiniteMeasure` stores one non-negative
   mass per atom; additivity holds by construction and subsets that split
   an atom are rejected rather than assigned a value. `restrict`,
   `pushforward` (with a measurability check against the target space),
   `product`, `with_density`, and `from_probability_measure` (the bridge
   from ADR-0002 objects) all return new measures.
5. **Exact arithmetic by input type.** `int` and `Fraction` inputs are kept
   as `Fraction`, so every identity in the module (Fubini, the
   Radon-Nikodym reconstruction, the chain rule, the tower property) holds
   with `==`. `float` inputs are accepted; results that depend on them are
   floats and comparisons use an explicit tolerance (default `1e-12`).
6. **Almost-everywhere objects use an explicit convention.** The
   Radon-Nikodym derivative and the conditional expectation are unique only
   almost everywhere; on null atoms both return `0` and say so in their
   docstrings. Checks (`verify_tower_property`,
   `verify_taking_out_what_is_known`) compare on positive-mass atoms only.
7. **Fubini is a check, not a theorem prover.** `fubini_check` returns the
   product integral and both iterated integrals (`FubiniCheck`). On a finite
   space every measurable function is integrable, so disagreement can only
   indicate a defect.
8. **Completeness is a rank condition.** For a finite parameter set and a
   finite sample space, `T` is complete iff the matrix
   `A[theta, t] = P_theta(T = t)` over charged values `t` has full column
   rank. `check_completeness` computes the rank by exact Gauss-Jordan
   elimination for fractions (partial pivoting with a relative tolerance for
   floats) and returns a null vector as an explicit unbiased estimator of
   zero when `T` is not complete. This replaces the function-space search
   that ADR-0051 rejected: on finite families the search is a linear system.
   The verdict is relative to the parameter set given; the binomial count
   with `n` trials is complete for any `n + 1` distinct `p` and not for `n`.
9. **Interpretation guide uses existing functions only.** The frequentist
   and Bayesian guide quotes Wilson and Clopper-Pearson intervals, the exact
   binomial test, the Beta posterior, a beta-binomial Bayes factor, exact
   and simulated coverage (`evaluate_estimator`), prior-averaged coverage,
   and simulation-based calibration. No new inference function is added;
   the guide's code blocks are executed by the test suite and every quoted
   decimal must appear in their output.

## Consequences

- Seven public names are added to `probstat.probability`:
  `FiniteMeasurableSpace`, `FiniteMeasure`, `FubiniCheck`,
  `CompletenessCheck`, `check_completeness`, `is_complete_statistic`, and
  `is_sigma_algebra`. The generated API reference is stale until it is
  regenerated.
- The equation registry gains EQ-MEAS-001..010. The bibliography gains
  Billingsley 1995, Durrett 2019, and Berger-Sellke 1987.
- Section locators in Billingsley (Sections 2, 10, 13, 15-16, 18, 32, 34),
  Durrett (Sections 1.1-1.7, 4.1), Lehmann-Casella (Section 1.6), and
  Casella-Berger (Section 6.2.4) follow the books' standard tables of
  contents; they were not re-checked against the printed books.
- Only finite spaces are covered. Countable and continuous spaces,
  Lebesgue measure on the line, and convergence theorems for integrals
  remain described in prose only.
- `events()` is exponential in the number of atoms and is intended for
  small examples and exhaustive tests.

## Verification

- `tests/test_measure_theory.py` (29 tests): generated sigma-algebra sizes
  `2^k` and agreement with a brute-force closure; closure-check rejections;
  non-measurable events, maps, and functions rejected; pushforward equals
  `mu(f^-1(B))` on every target event; integrals equal exact finite sums
  and agree across simple-function representations; Fubini equality on
  three functions, exactly; `nu(A) = integral_A (d nu / d mu) dmu` on all
  64 events and the chain rule pointwise; the defining property of
  `E[X | G]` on every `G`-event, the tower property, and taking out what is
  known, all exact; total variation equal to an exhaustive supremum;
  binomial count complete for `n + 1` and incomplete for `n` parameter
  values (`n = 1..4`) with witnesses of exact zero expectation; the full
  sample incomplete; invalid inputs raising `TypeError` or `ValueError`.
- The same file executes both new guides and compares every printed block,
  and checks that every decimal of three or more digits quoted in the
  interpretation guide's prose is printed by its code, plus the derived
  claims (simulation within 1.2 standard errors, `1 < B_10 < 3`,
  Clopper-Pearson coverage at least 0.95, prior-averaged credible coverage
  0.95).
- `tests/test_documentation_contract.py`, `tests/test_architecture.py`,
  `tests/test_finite_probability.py`, `tests/test_prerequisite_guides.py`,
  `ruff check`, and strict `mypy` pass.
