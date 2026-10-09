# Session: UQ-2 Propagation and Basic Monte Carlo

- Date: 2026-10-09
- Release: 0.9.0

## Delivered

- Composite trapezoid and Simpson quadrature with exact evaluation counts.
- Finite-bound expectation integration with density validation.
- Monte Carlo mean, probability, and uniform-integral estimators with sample
  variance, standard error, confidence intervals, and work metadata.
- Nested convergence traces and root-N teaching visualization.
- Named QoI forward propagation with raw outputs, output moments, strict
  threshold probabilities, Wilson intervals, and model-call accounting.
- A seeded lab covering pi estimation, quadrature versus Monte Carlo, and
  projectile-range propagation.

## Explicit limitations

- Confidence intervals use normal/Wilson approximations as labeled.
- Finite quadrature bounds do not include omitted tail mass.
- Input samples are materialized; no streaming or distributed execution yet.
- The projectile model is pedagogical constant-gravity kinematics, not an
  orbital dynamics model.

## Next step

Implement UQ-3 variance reduction: antithetic, control, conditional,
stratified, and importance estimators with common cost/efficiency diagnostics.
