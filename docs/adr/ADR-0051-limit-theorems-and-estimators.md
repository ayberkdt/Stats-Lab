# ADR-0051 — Limit theorems and properties of estimators

- Status: Accepted
- Date: 2026-10-12
- Addresses (roadmap): Phase 2 "Modes of convergence, laws of large numbers,
  and central limit theorems" and Phase 4 "Estimands, estimators, bias,
  risk, consistency, sufficiency, completeness". Completeness is documented
  only.

## Context

The library had sampling distributions, Fisher information, MLE fitting, and
Monte Carlo estimators, but no layer that answered limit-theorem questions
or evaluated estimators as procedures. A simulation-only treatment would
teach the wrong habit: it cannot tell a true CLT error from Monte Carlo
noise.

## Decision

1. **Exact results and simulations live in different packages.**
   - `probability.limit_theorems` (layer 1) holds only closed forms and
     proven inequalities. It imports `foundations` and the private
     `distributions._special` incomplete-gamma functions. The same private
     import is already used by `inference.sampling_distributions`.
   - `monte_carlo.limit_simulation` (layer 3) holds the RandomStream-driven
     demonstrations. Each of its docstrings points to the exact counterpart.
2. **Exact laws only where they are closed-form.**
   - The summand families are Bernoulli (binomial sum), exponential (gamma
     sum, rate-free standardized CDF `P(n, n + z sqrt(n))`), and normal. They
     are described by a `SummandLaw` record carrying `mu`, `sigma^2`, the
     third central moment, and `rho`.
   - Rejected: a generic convolution engine. It would need numerical
     characteristic-function inversion, whose error would contaminate the
     very distances being measured.
3. **Kolmogorov distance.**
   - Lattice case: exact evaluation of both one-sided limits at every atom.
   - Continuous case: a deterministic 1601-node grid plus golden-section
     refinement of the five largest local maxima, with the support edge as a
     node. It agrees with a 40001-point scan to `1e-6`.
4. **Berry-Esseen constant.**
   - `C = 0.4748` (Shevtsova 2011, arXiv:1111.6554). The arXiv abstract was
     read in this session.
   - The bound is not capped at one, so its vacuity for skewed summands is
     visible.
5. **Edgeworth.** One term, non-lattice only. For a Bernoulli summand the
   comparison raises rather than return a number that omits the lattice
   correction.
6. **Tail bounds.**
   - Chebyshev is two-sided.
   - Hoeffding and Bernstein are one-sided, bounded-support forms, with
     two-sided doubling available. Bernstein follows the bounded case of
     Boucheron-Lugosi-Massart Corollary 2.11.
   - For unbounded summands, `compare_tail_bounds` reports `None` rather
     than a bound whose hypothesis fails.
7. **Estimator harness.**
   - `inference.evaluate_estimator` simulates data set `r` from
     `seed.child("replication", r)`. The estimator, a reference estimator,
     and an interval procedure therefore share data (a paired design), and
     results are independent of evaluation order.
   - Every reported quantity has a Monte Carlo standard error:
     - variance: from the fourth central moment;
     - relative efficiency: delta-method ratio standard error including the
       covariance;
     - coverage: binomial.
   - The standard errors were checked against the spread over 60 runs (all
     ratios within 0.99 to 1.11).
8. **Cramer-Rao reuses `information.geometry`.**
   - `inference` and `information` are both layer 2. `information` does not
     import `inference`, so the new import keeps the graph acyclic. It
     avoids re-deriving Fisher information.
   - The bound takes a gradient, so one function covers transformed
     estimands (`sigma^2` from the `(mu, sigma)` information) and biased
     estimators (`grad E[T]`).
9. **Worked examples are exact functions, not prose.** These are:
   - the variance-divisor risks (minimum at `n + 1`);
   - the uniform-maximum risks (MLE, UMVUE, minimum-MSE multiple, moments);
   - the Poisson zero-probability Rao-Blackwellization (crude,
     `((n-1)/n)^S`, plug-in, CRLB);
   - the exact MLE efficiencies for five family/estimand pairs.

   Each is verified against simulation within four Monte Carlo standard
   errors in `tests/test_estimator_properties.py`.
10. **Sufficiency is checked, not proved.**
    - `check_sufficiency` tests the factorization consequence that equal
      `T` implies a `theta`-free likelihood ratio. It can refute sufficiency
      (the Laplace mean) but only supplies evidence for it.
    - Completeness is left to the documentation. A numerical check would
      need a function-space search with no deterministic answer.

## Consequences

- There are 47 new public names: 24 in `probability`, 6 in `monte_carlo`,
  and 17 in `inference`. The API reference is regenerated, and no package
  or layer is added.
- The equation registry gains EQ-LIM-001..010 and EQ-EST-001..010.
- Five references are added: Shevtsova 2011, Hoeffding 1963,
  Boucheron-Lugosi-Massart 2013, Hall 1992, and Lehmann-Casella 1998.
- The section numbers cited inside Casella-Berger and Lehmann-Casella come
  from the books' standard tables of contents and were not checked against
  the books in this session. Hall 1992 and Boucheron et al. are cited at
  chapter or corollary level for that reason.
- The two new test files run in about 3 s and 5 s. The lab runs in about
  2 s and prints numbers only, because no existing visualization helper
  fits and `visualization/` was out of scope.
