# ADR-0023 — Stochastic processes and random fields

- Status: Accepted
- Date: 2026-10-11

## Context

UQ studies need random quantities that vary in time (degradation, arrivals,
diffusions) and in space (material properties, loads). Spatial variability
enters forward models as random fields. To be usable by propagation,
reliability, and sensitivity analysis, a field must be reduced to a finite
set of scalar inputs of the central `ProbabilisticModel`. The library also
lacked Markov chains and SDE schemes, which the curriculum needs and which
serve as references for filtering and MCMC.

## Decision

1. New layer-2 package `probstat.processes`. It may import `foundations`,
   `linalg`, and `optimization` (Brent roots). It does not import `uq`,
   `inference`, `multivariate`, or any layer of 3 or higher. Modules:
   - `markov`: finite discrete-time chains.
   - `continuous`: Poisson and compound Poisson processes, Brownian motion,
     GBM, and OU, all with exact samplers.
   - `sde`: Euler-Maruyama, Milstein, and strong/weak convergence studies.
   - `random_fields`: kernels, quadrature rules, Nystrom KL, and the analytic
     exponential-kernel benchmark.
2. Markov chains:
   - Classification uses the exact zero pattern of `P`: an entry is an edge
     iff it is `> 0`.
   - The stationary distribution, fundamental matrix, and first-passage times
     come from direct Gaussian elimination (`linalg.lu_solve`). No power
     iteration is used.
   - The stationary distribution raises unless there is exactly one closed
     class.
   - Absorption analysis requires every closed class to be a single
     absorbing state. Recurrent classes must be lumped first; the method
     never guesses.
3. Exact samplers are preferred wherever a closed-form transition law exists
   (GBM, OU, Brownian motion). SDE schemes exist to approximate equations
   without one, and to teach discretization error.
4. Convergence studies drive every step size with the same fine Brownian
   path (block-summed increments). The strong error is then a pathwise
   comparison. Every study reports Monte Carlo standard errors next to the
   errors, and the fitted order is a plain log-log least-squares slope.
5. KL expansion:
   - Nystrom discretization with explicit quadrature weights, symmetrized as
     `W^(1/2) C W^(1/2)` and solved by `linalg.symmetric_eigen` (Jacobi).
     Unit weights give the plain covariance-matrix PCA on a grid.
   - Truncation keeps the smallest `M` that reaches a target fraction of the
     discrete integrated variance, optionally capped by `max_terms`.
   - Materially negative discrete eigenvalues raise; round-off negatives are
     clipped to zero and never retained.
   - Eigenvector signs are fixed so that the component of largest magnitude
     is positive. Results are therefore reproducible, but signs can differ
     from textbook conventions.
6. Random-field adapter in `probstat.modeling.random_field`. Layer 3 may
   import layer 2.
   - `RandomFieldInput` exposes `<prefix>_1..<prefix>_M` as independent
     `Normal(0, 1)` `RandomInput`s.
   - It maps a parameter mapping to nodal field values, with an optional
     pointwise transform (for example `exp` for log-normal fields).
   - `karhunen_loeve_inputs` returns `(inputs, field_function)` for users who
     assemble the `JointInput` themselves.
   - The field enters `ProbabilisticModel` through ordinary inputs, so no
     downstream workflow changes.

## Consequences

- Validation results are in `tests/test_stochastic_processes.py` and
  `docs/foundations/stochastic-processes-and-random-fields.md`:
  - The KL solver reproduces the Ghanem-Spanos exponential-kernel
    eigenvalues: within 0.05 % for mode 1 and within 1 % for modes 1-4, with
    40 midpoint nodes.
  - EM and Milstein reach strong orders 0.55 and 0.94 on Higham's GBM
    problem.
  - The weak EM error tracks the exact bias `e^2 - (1 + 2h)^N` within four
    standard errors.
  - The random-stiffness bar propagated through `propagate` matches the
    closed-form truncated variance within 0.4 % and a 10,000-draw reference
    within 2 %.
- Pure-Python Jacobi limits KL grids to roughly 150 nodes. Large 2-D or 3-D
  fields need a future iterative (Lanczos) or low-rank eigen-solver.
- Sobol indices of a field model attribute variance to KL modes rather than
  to locations. Users must interpret them as such.

## Rejected alternatives

- Computing the stationary distribution by power iteration: it converges
  slowly or not at all for periodic and slowly mixing chains, and it hides
  non-uniqueness.
- Silently lumping recurrent classes in absorption analysis: it changes the
  question the user asked.
- Putting the random-field adapter inside `processes`: it would need
  `modeling` (layer 3) from layer 2, an upward import.
- Embedding the KL basis into `surrogates`: random-field discretization is
  input modeling, not emulation.

## Unverified details

The section and chapter numbers cited for Norris (1997), Kloeden-Platen
(1992), Glasserman (2003), Davis-Rabinowitz (1984), and Rasmussen-Williams
(2006, Section 4.3) were recalled from memory. They were not checked against
the books in this session. The bibliographic records themselves are standard
and, to the best of our knowledge, correct.
