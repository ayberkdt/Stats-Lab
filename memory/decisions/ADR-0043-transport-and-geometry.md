# ADR-0043 — Optimal transport and information geometry

- Status: Accepted
- Date: 2026-10-12

## Context

Roadmap Phase 6 lists "Information geometry and optimal transport".
`probstat.information` (layer 2, ADR-0036) already had entropies, KL, JS, and
mutual information. It also had closed-form differential entropies, Gaussian
KL, and nearest-neighbour estimators. It had no transport plans, Wasserstein
distances, Fisher information, or non-KL divergences.

At the same time, another work stream is adding 1-D sample Wasserstein
distances and energy/MMD two-sample tests to the validation layer
(layer 5) for forecast calibration.

The library is pure Python with no runtime dependencies. Its layer map is
enforced by `tests/test_architecture.py`.

## Decision

1. **Four new modules inside `probstat.information`, no new package.**
   - `transport` holds discrete OT.
   - `wasserstein` holds closed forms and barycenters.
   - `geometry` holds Fisher information and Fisher-Rao distances.
   - `divergences` holds TV, Hellinger, Bhattacharyya, Rényi, and alpha.
   - A shared private helper, `_quantile_quadrature`, does quadrature in
     the normal variable.

   The modules import only layers 0-1: `linalg`, `numerics`, `foundations`,
   `distributions`, and `optimization`. This is the same boundary as
   ADR-0036, so the layer map, `GROUPS`, and the architecture test are
   unchanged. A separate `transport` package was rejected because
   Wasserstein distances, f-divergences, and the Fisher metric are compared
   with one another constantly, and they share validation helpers
   (`_spd`, `_pmf`).

2. **No hypothesis tests here.** Two-sample tests and sample-based 1-D
   Wasserstein distances belong to the validation work stream. This
   package supplies distances between *given* measures or laws. The 1-D
   function is named `quantile_wasserstein` (between distribution objects),
   which keeps it clearly separate from sample-based functions.

3. **Exact solver: the transportation simplex.** The alternatives were the
   Hungarian algorithm, which covers only assignment, and a general network
   simplex.
   - The north-west corner rule gives a spanning-tree basis; tree
     potentials and cycle pivots follow.
   - It handles rectangular problems and zero-mass atoms, and its final
     potentials are the dual certificate.
   - Anti-cycling: Dantzig pricing, switching to Bland's rule after `m + n`
     consecutive degenerate pivots.
   - `transport_certificate` checks a plan and potentials independently.
     It reports the duality gap, marginal error, dual infeasibility, and
     complementary slackness, so a caller can verify any claimed optimum.

4. **Sinkhorn in the log domain with the KL-to-`a b'` regularizer.**
   - The potentials are soft c-transforms.
   - The stopping rule is the `L1` row-marginal error after an exact column
     update; the full per-iteration trace is kept.
   - The reported `regularized_cost` is the dual value. Under the
     KL-to-`a b'` convention this is the quantity whose debiasing gives the
     Feydy et al. Sinkhorn divergence.
   - Self terms of the divergence use the averaged symmetric map, because
     plain alternation on `(alpha, alpha)` was measured to converge very
     slowly.
   - `epsilon_scaling` is optional and off by default; the measured benefit
     was small.
   - `sinkhorn` returns `converged=False` rather than raising.
     `sinkhorn_divergence` raises `ArithmeticError`, because a composite of
     unconverged terms is meaningless.

5. **1-D `W_p` and Fisher expectations by normal-substitution Simpson.**
   With `u = Phi(z)`, unbounded quantile functions lose their endpoint
   singularities. The lower-tail probability is computed with `erfc`, and
   `isf` is used for upper-tail nodes. `numerics` has only fixed-step
   Simpson and trapezoid rules, so no adaptive quadrature was added.

6. **Matrix square roots by `linalg.symmetric_eigen`.**
   - Eigenvalues above `-1e-10 * scale` are clipped, so the Bures distance
     accepts PSD covariances.
   - Negative powers require PD.
   - No `sqrtm` was added to `linalg`, which is owned by other work. The
     helper is private to `wasserstein`.

7. **Fisher information as functions.**
   - Closed forms return matrices (`1 x 1` for scalar families) in the
     distribution objects' own parameterizations: Gamma by rate, Normal by
     `(mu, sigma)`.
   - `fisher_information(law)` dispatches by type, as
     `differential_entropy` does (ADR-0036 decision 8).
   - `numerical_fisher_information` uses `optimization.central_gradient` and
     `central_hessian` in `theta`. It works with any family callable that
     returns an object with `log_pdf` or `log_pmf`.

8. **Fisher-Rao distances only where a closed form is standard.**
   - Normal: hyperbolic half-plane, evaluated in an `asinh` form that stays
     accurate for nearby laws.
   - Categorical, Bernoulli, binomial: the sphere, evaluated as a chord
     `asin`.
   - Poisson, exponential, and MVN mean with known covariance (Mahalanobis).
   - The full MVN and other non-closed-form models need geodesic shooting,
     which is deferred.

9. **Hellinger convention: `H^2 = 1 - BC` (range `[0, 1]`).** The docstring
   says that texts omitting the `1/2` report `sqrt 2 H`.
   - Rényi follows van Erven and Harremoës, for `alpha` in `[0, inf]`.
   - The alpha-divergence follows Cichocki, Cruces and Amari (2011, eq. 25),
     normalized: `KL(p||q)` at `alpha = 1`, `KL(q||p)` at `alpha = 0`.
   - Gaussian versions come from one log-affinity function. That function
     returns `+inf` when `alpha S1 + (1 - alpha) S0` is not PD.

## Consequences

- 48 new public names in `probstat.information`. The API reference was
  regenerated.
- `tests/test_transport_and_geometry.py` has 26 tests and runs in about 3 s
  under `unittest`. Exact OT matches brute force; certificates hold;
  Sinkhorn converges to the LP optimum as `eps -> 0` and meets its marginal
  tolerance; closed forms match independent quadrature; Fisher closed forms
  match numerical expectations to `1e-6`; Fisher-Rao distances are metrics
  with the Fisher local expansion; Rényi tends to KL.
- `examples/optimal_transport_lab.py` prints five tables in about 2 s.
  `docs/foundations/optimal-transport.md` documents them.
- Known limits: dense pure-Python OT, which is slow for small `eps`
  relative to the cost spread; Fisher information for regular families only
  (support independent of `theta`); no Fisher-Rao for the full MVN; no
  Gaussian TV beyond 1-D or a shared covariance.
- Citation status. Bibliographic details were checked by web search in this
  session for all new references. The formula content was read directly
  for:
  - Miyamoto et al. 2024 (arXiv v3);
  - Cichocki-Cruces-Amari 2011 (eq. 25);
  - Peyré-Cuturi 2019 (section list, Remarks 2.30-2.31, Section 3.3).

  Content not read in this session: Atkinson-Mitchell 1981, Kailath 1967,
  Gil-Alajaji-Linder 2013 (only the abstract was seen), Tsybakov 2009, and
  Dowson-Landau / Olkin-Pukelsheim (only the abstracts were seen). The
  formulas attributed to them are verified numerically in the tests.
