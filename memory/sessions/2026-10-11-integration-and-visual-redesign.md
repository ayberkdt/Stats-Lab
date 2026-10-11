# Session 2026-10-11 — Integration round (0.14) and visual redesign

## Goal

Implement the review recommendations (central model, forward/backward/risk
workflows, remaining UQ topics) so that analyses no longer run as separate
islands, and replace the default-looking figures with a designed theme based
on the `sunumatik` chart-aesthetics rules.

## What changed

- One `ProbabilisticModel` (ADR-0014) with Nataf joint inputs; shared
  `linalg`, `optimization`, `sampling`, random streams.
- Parallel work streams (separate git worktrees, merged into `main` with
  union resolution of the append-only registry and bibliography):
  reliability (ADR-0015), risk (ADR-0016), nonlinear calibration and MCMC
  (ADR-0017), filtering and data assimilation (ADR-0018), surrogates
  (ADR-0019), continuous distribution catalog (ADR-0020), classical and
  resampling inference (ADR-0021), model validation (ADR-0022), stochastic
  processes and random fields (ADR-0023).
- Semantic chart theme (ADR-0024) and the `UQStudy` workflow (ADR-0025).
- README rebuilt around the study API with light/dark gallery images.

## Verification evidence (recorded at merge time)

- Full suite after the last wave-2 merge: `436 passed, 685 subtests passed`;
  `ruff check .` clean; `mypy src/probstat` strict clean (128 files).
- Every example lab runs; total about three minutes on the development
  machine.
- Spot checks against closed forms: Nataf latent correlation vs lognormal
  closed form (difference 6e-14); copula log-density equals MVN log-density;
  Sobol on Ishigami within bootstrap intervals of the analytic indices; FORM
  converges at beta = 9.5 after the upper-tail fix; emulator exceedance
  0.03833 vs true model 0.03827.

## Bugs found and fixed during the round

- `Normal.cdf` lost relative accuracy in the lower tail (`1 + erf`).
- Nataf upper tail saturated near z = 8 (FORM stalled at beta ~ 6.8).
- `UQStudy.emulate` with a Hermite basis on uniform inputs returned a mean of
  -0.98 instead of 3.5 on large quasi-random designs.
- Direct chart labels read a stale data transform before autoscaling.
- A computed Sobol title asserted the wrong input acted "mostly through
  interactions".

## Open / next safe steps

- Bibliographic details written by the parallel work streams were largely
  cited from memory; their reports list the locators to spot-check.
- Numerical backend boundary for large dense problems (ADR-0008 follow-up).
- Remaining Partial ATHENS rows (UQ-0 verification topics, fitting, labs,
  prerequisites) are being addressed by the next work wave.
