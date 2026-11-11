# ADR-0015 — Model-based reliability on the shared probabilistic model

- Status: Accepted
- Date: 2026-10-11

## Context

ADR-0011 limited FORM to affine standard-normal limit states and left general
FORM, curvature corrections, multiple design points, and systems as named
extensions. The central `ProbabilisticModel` now owns the joint input law and
an auditable Nataf map to independent standard normals. Reliability methods
must use that map instead of each defining their own input transformation.
They must also keep the closed failure convention `g <= 0`.

## Decision

1. `ModelLimitState(model, quantity, threshold, failure)` is the single
   reliability view of a model. It sets `g = t - y` for failure `"above"` and
   `g = y - t` for failure `"below"`. It evaluates `g` in physical space and in
   standard-normal space through `model.inputs.from_standard_normal`, and it
   charges every call to the model's own evaluation counter.
2. FORM uses HL-RF in `u`, with the iHL-RF merit-function Armijo search on by
   default. Gradients are central differences from `probstat.optimization`
   unless an analytic gradient is supplied. Reaching the iteration limit
   returns `converged=False` and is never reported as success. A vanishing
   gradient raises an error.
3. SORM rotates a finite-difference Hessian so that the last axis is `alpha`
   and diagonalizes the tangent block with `linalg.symmetric_eigen`. It
   reports both the Breitung and the Hohenbichler-Rackwitz corrections. It
   refuses `beta <= 0`, any curvature with `1 + beta kappa <= 0` (or
   `1 + psi kappa <= 0`), and FORM results that did not converge.
4. Every sampling method draws in `u` from a `RandomStream` child stream:
   crude Monte Carlo, importance sampling around the design point, subset
   simulation, and system sampling. Each reports its model evaluations and a
   coefficient of variation. The subset-simulation c.o.v. is reported as
   approximate. The root-sum-square value assumes independent levels, so a
   fully correlated upper value is reported alongside it.
5. System reliability requires every component to share one joint input law.
   Series bounds are first-order quantities: uni-modal bounds and Ditlevsen
   bounds built from `rho_ij = alpha_i . alpha_j` and an in-house bivariate
   normal CDF (Genz / Drezner-Wesolowsky with Golub-Welsch Gauss-Legendre
   nodes). The uni-modal upper bound is flagged when correlations are
   negative.
6. Inverse reliability has a deliberately small scope. It covers a threshold
   found by the sphere fixed-point iteration, and one design parameter found
   by Brent's method on `beta(theta)`. General RBDO with several constraints
   and several variables is out of scope.
7. The new code lives in `reliability.model_based`, `reliability.simulation`,
   and `reliability.systems`. The original vector-level API in
   `reliability.analysis` is unchanged.

## Consequences

General nonlinear, non-normal, correlated FORM/SORM, rare-event simulation,
and simple systems now share one model definition and its cost accounting.
Results reproduce exactly for a fixed seed.

Known limits:

- FORM finds one design point. Problems with several design points need
  multiple starts, or sampling methods.
- Mixture importance sampling for parallel systems can be inefficient.
- `JointInput.from_standard_normal` loses resolution in the upper tail
  (latent `z > ~6`, saturating near `z ~ 8.1`), so FORM at very large indices
  is unreliable. Fixing this belongs to the modeling layer, not reliability.

## Rejected alternatives

- Accepting the last iterate of a non-converged FORM silently: it would hide
  failures of the design-point search.
- Clipping invalid SORM curvatures: it would return a number for a geometry
  that the formula does not describe.
- Tabulating Gauss-Legendre constants or importing SciPy: the library has no
  runtime dependencies, and computing the nodes is auditable through the
  existing Jacobi eigen-solver.
- Re-implementing a Nataf transform inside reliability: it would duplicate the
  joint law owned by `probstat.modeling`.
