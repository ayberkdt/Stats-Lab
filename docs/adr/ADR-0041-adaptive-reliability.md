# ADR-0041 — Adaptive reliability: lines, multiple design points, AK-MCS

- Status: Accepted
- Date: 2026-10-12

## Context

The roadmap item "Multiple design points, cross-entropy adaptation, and line
sampling" was partly done. CE-IS and line sampling existed (ADR-0029). A
reviewer then measured covariance-adapted CE-IS on a `beta = 4` plane at 0.38
to 0.65 of the exact value, while it reported c.o.v. values of 4-13%. A second
reviewer noted that the line-sampling docstring disagreed with the code about
lines with no root. Multiple design points and surrogate-based active learning
were missing. `reliability` and `surrogates` are both layer 5. The
architecture test forbids only upward imports and cycles, so a same-layer
import is allowed as long as it stays acyclic.

## Decision

1. **CE covariance guard.** Adapted covariances are projected onto eigenvalues
   `>= covariance_floor` (default `0.55`, which must exceed `1/2`) at every
   level. The weight variance of a Gaussian proposal is finite only when every
   eigenvalue is above `1/2`, and the floor removed the measured bias (seeds
   1-3: ratios 1.038, 0.986, 1.060, all within 3 SE). `covariance_floor=None`
   restores the old behaviour. The result gains
   `failure_effective_sample_size`, `min_covariance_eigenvalue`, and
   `weights_degenerate`. A `UserWarning` fires when an eigenvalue is `<= 1/2`
   or when the failure ESS is below `0.01 N`. The Kish ESS alone missed the
   biased runs, so the eigenvalue test is part of the flag. The mean-shift
   default is numerically unchanged.
2. **Line sampling keeps exact 0/1 contributions** for rootless lines,
   matching the infinite distances and the equation registry. The docstrings
   that said `Phi(-max_distance)` were corrected; the code was not changed.
3. **HL-RF is factored into private `_hlrf_iterate` and `_form_result`**
   helpers in `model_based`, so bulged searches reuse the exact `form_hlrf`
   loop. `form_hlrf`'s behaviour and signature are unchanged. The core gains
   an optional feasibility `scale`, because refinement starts on the surface,
   where `|G(start)|` is almost zero.
4. **Multiple design points by bulging with multi-start** (new module
   `reliability.design_points`). Bulging follows Der Kiureghian and Dakessian
   (1998). The quartic bump, the height `0.75 G(0)`, and the radius
   `1.1 beta_j` are this library's choices; the paper's exact profile was not
   checked. A single restart from the origin cannot leave `u_1` when
   `grad G(0)` points at it (shown by a test), so the default restarts also
   from `+-beta_1 e_i` and accepts the nearest new refined point.
5. **Multi-point FORM** uses closed forms for one or two points (existing
   bivariate normal CDF). For three or more it uses Genz's
   separation-of-variables Monte Carlo with a semidefinite Cholesky
   factorization. Rank-deficient rows are folded into two-sided bounds, so
   antipodal modes add no noise. The complement is averaged directly to keep
   relative precision. Ditlevsen bounds are returned alongside.
6. **Mixture IS** over all design points uses unit-covariance components
   (always unbiased) with equal or FORM-probability weights.
7. **AK-MCS lives in `reliability.active_learning` and imports
   `probstat.surrogates`.** AK-MCS is a reliability method, and `surrogates`
   does not import `reliability`, so the graph stays acyclic. If `surrogates`
   ever needs `reliability`, AK-MCS must move to a higher layer. The GP is
   fitted in `u` space. Hyperparameters are refitted (warm-started) when the
   training set has grown by a factor of 1.5, and before stopping. Between
   fits the population posterior is updated by exact rank-one Cholesky
   updates. The initial LHS is drawn in the box `[-5, 5]^d`, because a design
   mapped through `Phi^-1` once stopped at once with `P_f = 0` on the
   four-branch problem. Runs that predict no failure are not converged. The
   true-call count comes from the model's evaluation counter, and the optional
   same-population crude MC reference is charged separately.

## Consequences

- On the four-branch system with `N = 10^4`, AK-MCS reproduced crude MC on the
  same population exactly for seeds 1-5, with 34-48 model calls and zero
  misclassified points. The squared-exponential kernel failed one of two
  seeds, so `matern52` is the default.
- New tests (`tests/test_adaptive_reliability.py`, about 13 s) cover the CE
  warning and fix, line sampling (exact plane, quadrature parabola, 0/1
  lines), two-mode multi-point FORM against the exact bivariate value, SOV
  against quadrature and antipodal exact values, mixture IS within 3 SE, and
  AK-MCS against same-population MC plus seeded determinism.
- Not implemented: Gaussian-mixture CE proposals, SOV variable reordering,
  and importance or subset variants of AK-MCS for very small `P_f`.
