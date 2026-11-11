# ADR-0050 — Adaptive quadrature with error estimates and unbounded ranges

- Status: Accepted
- Date: 2026-10-12
- Closes (roadmap): "Unbounded-domain quadrature transformations, adaptive
  error estimates"

## Context

`probstat.numerics` had fixed-grid trapezoid and Simpson rules (no error
estimate) and `tanh_sinh` restricted to sub-intervals of `(0, 1)` with an
integrand of `(u, 1 - u)`. Callers that needed `[a, b]`, `[a, inf)` or the
whole line had to truncate the range or invent substitutions by hand, and
nothing reported whether a result could be trusted.

## Decision

1. **Four new layer-0 modules in `probstat.numerics`.** They import only
   `foundations.validation`.
   - `integration_result.py`: `IntegrationResult` (value, error estimate,
     evaluations, `converged`, rule name, final panels, DE levels) and the
     shared tolerance checks.
   - `gauss_kronrod.py`: `gauss_kronrod`, QUADPACK QAG without extrapolation
     (G7-K15 panels, bisect the panel with the largest local error, stop at
     `max(abs_tol, rel_tol |I|)`, the panel `limit`, or an unbisectable panel).
     Infinite ranges use the QAGI map `x = a + (1 - t)/t`.
   - `de_transforms.py`: `tanh_sinh_interval`, `exp_sinh`, `sinh_sinh`, with
     the level-halving trapezoid and last-two-levels error estimate of the
     existing `tanh_sinh`.
   - `integration.py`: `integrate(f, a, b, method="auto")` dispatches on the
     finiteness of the limits. `"auto"` runs the double-exponential rule and,
     only if it does not converge, also Gauss-Kronrod; it returns the result
     with the smaller error estimate and counts both runs' evaluations.
2. **Constants.** The G7-K15 nodes and weights are the 33-digit values of
   QUADPACK `dqk15.f` (Netlib). The test suite checks in `Fraction`
   arithmetic that K15 is exact through degree 22 and G7 through degree 13
   (residuals below `1e-32`). The check caught a wrong trailing digit in a
   value recalled from memory (`xgk(5)`, wrong from the 27th digit on).
3. **Error estimates are conservative by design.**
   - Gauss-Kronrod: QUADPACK's `resasc min(1, (200 |K - G| / resasc)^1.5)`,
     floored at `50 eps int |f|`.
   - Double exponential: `max(|T_k - T_{k-1}|, 50 eps int |f|)` plus a
     node-rounding term `h sum |w f| min(1, ulp(x)/d)`, `d` being the
     distance to the nearest finite end. Without the last term,
     `(x - 2)^-1/2` on `[2, 3]` reported `converged=True` with an estimate of
     `1.4e-10` while the true error was `3.1e-8`, because the integrand sees
     rounded nodes closer to 2 than `ulp(2)`. With it the estimate is
     `5.4e-8` and the run is flagged unconverged.
   - Tolerances: `rel_tol < 50 eps` with `abs_tol = 0` is rejected, as in
     QUADPACK.
4. **Evaluation policy.**
   - Gauss-Kronrod raises on any non-finite value. Its nodes are interior,
     so a finite end is never evaluated.
   - The double-exponential rules raise inside the safe range. Outside it
     (relative distance below `1e-15` from a finite end, a node rounded onto
     the end, or `|x| > 1e15` on an infinite range), a failing or non-finite
     evaluation counts as zero, as the existing `tanh_sinh` does beyond its
     safe range.
5. **No existing call site changed.**
   - `validation.scoring`, `validation.shift` and `information.wasserstein`
     already use `tanh_sinh` on `(0, 1)` with complements.
   - The Simpson users (`propagation.forward`, `information` quantile
     quadrature, `inference.hierarchical`) expose `intervals`/`z_max` or
     grid parameters in public signatures, so swapping the rule would change
     their documented behaviour. A clear, test-neutral improvement was not
     available.

## Evidence (tests/test_adaptive_quadrature.py)

- Double exponential, defaults:
  - `int_0^1 x^-1/2`, `int_0^1 log x`, `int_0^inf e^-x`,
    `int_0^inf x^2 e^-x` and `Gamma(1/2)`: exact in 97 to 433 evaluations.
  - `int e^-x^2/2` (error `4.4e-16`, estimate/error `3.6e4`) and
    `int_0^inf 1/(1+x^2)` (error `2.2e-16`, ratio 80).
- Gauss-Kronrod: ratios 20 (`x^-1/2`, 1965 evaluations), 430 (`log x`),
  `1.4e5` (Gaussian) and `5.8e5` (`1/(1+x^2)`); exact results elsewhere.
- Lorentzian peak of half-width `1e-3` at 0.3: 23 panels, widths from
  `9.8e-4` at the peak to `0.25`; with `limit=5` the run reports
  `converged=False`, estimate `8.0e3` against a true error of `1.9e3`.
- Every result tested, converged or not, had an estimate at least as large
  as its true error, except for unconverged tanh-sinh runs at an interior
  kink (`|x - 0.3|`: ratio 0.42). There `auto` falls back to Gauss-Kronrod.

## Consequences

- No epsilon-algorithm extrapolation (QAGS) and no QUADPACK roundoff
  detection, so endpoint singularities cost more Gauss-Kronrod evaluations
  than in QUADPACK; tanh-sinh is the default for finite ranges for this
  reason.
- No oscillatory rules (QAWO/QAWF). `sin(x)/x` on `[0, inf)` is reported as
  unconverged, not solved.
- A singularity at a non-zero finite end must be moved to zero by the
  caller (or `tanh_sinh` with complements used on `(0, 1)`).
- Results are not vectorized; a call costs about 100 to 7000 Python
  evaluations (under 40 ms for every case in the test file).
