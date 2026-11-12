# ADR-0093: Defect review of systems, processes, risk, cost, surrogates, filtering, data assimilation, and time series

- Status: Accepted
- Date: 2026-11-12

## Context

The packages `probstat.systems`, `probstat.processes`, `probstat.risk`,
`probstat.cost`, `probstat.surrogates`, `probstat.filtering`,
`probstat.data_assimilation`, and `probstat.timeseries` (about 40 000 lines)
were reviewed for results that are wrong, misleading, or poorly explained.
Each public function was checked against its cited formula and convention,
at edge cases, against exact answers, and for the honesty of its reported
error estimates. A defect counts only when an independent oracle (closed
form, enumeration, quadrature, or a published value) demonstrates it.

## Decision

Two defects were found and fixed.

1. **`filtering.steady_state_kalman` declared convergence by an absolute
   test.** The stopping rule compared the change of `P^-` with
   `tolerance * max(1, max|P^-|)`, so for covariances far below one the
   test was absolute. For a scalar random walk with `q = r = 1e-12` the
   function returned after one iteration with `converged=True`, a
   predicted covariance 7.3 % below the Riccati fixed point
   `(q + sqrt(q^2 + 4 q r)) / 2` and a gain of 0.600 instead of 0.618; at
   `q = r = 1e-6` the relative error was `1.6e-8` instead of the requested
   `1e-12`. The test is now relative to the largest entry of `P^-`
   (`change <= tolerance * max|P^-|`), as the docstring already stated,
   so the iteration count and accuracy do not depend on the units of the
   state. A zero fixed point (no process noise, zero prior) still
   converges on the first step.
2. **`SystemStructure.reliability_moments(shared_moments=...)` documented
   the wrong variable.** The docstring asked for raw moments `E[q^k]`, and
   the user guide uses `q` for the unreliability, but the code integrates
   the shared *reliability* `p` whose mean and variance are given in
   `means` and `variances`. Following the documentation raised
   "raw moments disagree with mean/variance" whenever `E[p] != 1/2`, and
   for a law with mean one half and matching variance it would silently
   use the moments of the wrong variable. The docstrings of
   `reliability_moments` and of the raw-moment helper and
   `docs/foundations/system-reliability.md` now say `E[p^k]` of the
   reliability, and the error message names the unreliability when the
   supplied moments match `1 - p`.

Examined and rejected (behaviour agrees with the oracle or is documented):

- `timeseries`: ARMA(1,1)/MA(2) autocovariances, psi weights, root and
  reflection-coefficient checks, ACF/PACF/Durbin-Levinson, Bartlett and
  white-noise bands, Ljung-Box degrees of freedom, differencing and
  undifferencing, exact AR(1) and MA(1) likelihoods against the dense
  Gaussian density, AR(1), ARIMA(0,1,0), and ARIMA(0,2,0) forecast means
  and standard errors, simple and Holt smoothing recursions, classical
  decomposition; the frozen steady state of the unit-variance ARMA filter
  is safe because its innovation variance is at least one.
- `filtering`, `data_assimilation`: Kalman filter and log-likelihood on a
  scalar random walk, RTS smoother gain, Joseph update, unscented weights
  and sigma points, EKF cross covariances, bootstrap particle filter
  increments and resampling rule, stochastic EnKF gain and inflation,
  3D-Var gradient and Gauss-Newton covariance.
- `systems`: series, parallel, 2-out-of-3 and bridge reliabilities,
  Birnbaum, criticality, Fussell-Vesely, RAW and RRW against brute force;
  beta-factor, MGL, and alpha-factor coefficients (NUREG/CR-5485 forms,
  including staggered testing) and their normalisation; repairable unit
  availability, expected failures, interval availability and downtime with
  their bounds; cold and warm standby, load sharing, parallel pair MTTF,
  repairable k-out-of-n steady state; `SystemModel.mttf` over rate scales
  from `1e-9` to `1e3`; beta-factor hazard split; periodically tested
  unavailability series; exact epistemic moments of a bridge by
  enumeration; epistemic CCF moments (beta and MGL) against quadrature;
  PAND, cold and warm spare, FDEP, and SEQ gates of dynamic fault trees
  against closed forms; phased-mission error accumulation.
- `processes`: uniformization windows and Chernoff tail bounds, series and
  squaring paths on stiff two-state chains (`q t` up to `5e4`) with every
  reported bound containing the exact error, occupancy bounds,
  hypoexponential absorption, Markov-chain period, absorption and mean
  first passage times, compound Poisson variance, OU stationary
  covariance, exponential-kernel KL roots and normalisation, Matérn
  kernels. The semi-Markov Richardson estimate can understate the error by
  about 15 % when a sojourn CDF has a kink inside the horizon (for example
  `Uniform(1, 2)`); it is documented as an estimate, not a bound, so it was
  left unchanged.
- `risk`: lower-quantile VaR and Rockafellar-Uryasev CVaR with ties,
  normal and lognormal VaR/CVaR, GEV and POT return levels and gradients,
  GPD maximum-likelihood standard errors against the asymptotic
  covariance, L-moments, runs declustering, hazard convolution and its
  power-law oracle, fault-tree bounds, F-N curve step convention, EVPI and
  EVSI, MIL-STD-882E Table III, Thompson sampling.
- `cost`: discounting, annuity and capital recovery factors, Fisher
  relation, escalation, learning curves (unit and cumulative average),
  age replacement against a fine grid, block replacement, renewal
  function for exponential and gamma lifetimes with its error estimate,
  power-law overhaul optimum, spare provisioning and expected backorders,
  lognormal moment matching, joint confidence level.
- `surrogates`: SE and Matérn 5/2 kernels and their length-scale
  derivatives, GP prediction and closed-form LOO, orthonormal Hermite and
  Legendre recursions, PCE moments and Sobol indices.

## Consequences

- `steady_state_kalman` now needs the same number of iterations (15 for the
  scalar random walk at `tolerance = 1e-12`) at every scale of the
  covariances; problems whose covariances were far below one may take more
  iterations than before and now report `converged=False` honestly when
  the limit is reached. No seeded numeric result changes.
- `reliability_moments` behaves as before; only its documentation and the
  diagnostic message change.

## Verification

- `tests/test_review_systems_processes_risk.py`: 6 tests (5 sub-tests).
  Against the code before the fixes, 6 checks fail (3 scale sub-tests, the
  unit-invariance test, the unreliability-moment diagnostic, the
  docstring check); after the fixes all pass.
- Existing tests of every touched package (`tests/test_uq7_filtering.py`,
  `tests/test_systems_structure.py`, `tests/test_systems_bdd.py`,
  `tests/test_systems_importance.py`), `tests/test_documentation_contract.py`,
  and `tests/test_architecture.py` pass; ruff and strict mypy are clean.
- About 120 independent oracle comparisons across the eight packages were
  run during the review; the two defects above are the only failures.
