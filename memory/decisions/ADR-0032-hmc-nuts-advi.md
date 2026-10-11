# ADR-0032 — Hamiltonian Monte Carlo, NUTS, and Gaussian ADVI

- Status: Accepted
- Date: 2026-10-12
- Lifts: the "gradient-based samplers are not implemented" boundary of
  ADR-0017 and of the nonlinear-calibration guide

## Context

ADR-0017 gave the library random-walk and adaptive Metropolis with
diagnostics. Random-walk proposals mix slowly in more than about ten
dimensions and on curved posteriors. The library has no automatic
differentiation and no runtime dependencies, so any gradient-based method
must accept a user gradient or fall back to finite differences. It must also
keep reusing the existing `MCMCResult` diagnostics.

## Decision

1. New modules `probstat.inference.hamiltonian` and
   `probstat.inference.variational` (layer 2). They depend only on
   `foundations`, `linalg`, `optimization.central_gradient`, and
   `inference.mcmc`, so the layer graph is unchanged.
2. **Results reuse `MCMCResult`.** `HamiltonianResult` wraps an `MCMCResult`
   (`.mcmc`) plus one `HamiltonianChainStatistics` per chain. Those records
   hold the step size, `M^-1`, divergent iteration indices, acceptance
   statistics, tree depths, leapfrog counts, energies, E-BFMI, and gradient
   and density counts. In each `MarkovChain`, `acceptance_rate` holds the mean
   acceptance statistic and `proposal_covariance` holds the adapted `M^-1`.
3. **NUTS is the slice-sampling version** of Hoffman and Gelman (2014),
   Algorithms 3 and 6, with `Delta_max = 1000` and the U-turn criterion
   applied to the velocity `M^-1 p`. The multinomial variant used by current
   Stan is not implemented. Slice NUTS is the published, fully specified
   algorithm and is simpler to audit; multinomial NUTS is a possible later
   addition.
4. **Warm-up is simplified Stan-style windowing.** It has a 75-iteration
   initial fast buffer, doubling slow windows from 25 iterations, and a
   50-iteration terminal buffer. Below 150 iterations it uses 15 % / 75 % /
   10 % with one slow window, and below 20 iterations it adapts the step size
   only. At each window end:
   - `M^-1` is set to `n/(n+5) C + 1e-3 * 5/(n+5) I`;
   - the step size is re-initialized by Algorithm 4;
   - dual averaging restarts (`gamma = 0.05`, `t0 = 10`, `kappa = 0.75`,
     `delta = 0.8`).

   The shrinkage constants follow the defaults commonly used in Stan. They
   were **not** checked against Stan's source in this session and are
   recorded as a project choice. Both the step size and the metric freeze at
   the end of warm-up, as in ADR-0017, so retained draws come from one
   time-homogeneous kernel.
5. **Divergences are energy-error events.** A trajectory diverges when
   `H - H_0 > max_energy_error`, or when the log-density or gradient is
   non-finite or raises `ArithmeticError` or `ValueError` at a trajectory
   point. The initial point is still validated strictly. Static HMC stops and
   rejects a divergent trajectory. NUTS stops doubling, as in the paper.
6. **Gradients** come from a user callable or from central differences
   (`2d` extra evaluations per leapfrog step, documented and tested). Because
   leapfrog stays volume-preserving and reversible for any position-dependent
   force, finite-difference gradients cost efficiency, not correctness.
7. **Static HMC** uses a fixed `L` with optional uniform step-size jitter. The
   default jitter is 0, so a run with fixed settings is deterministic; the
   docs warn that a fixed `L eps` can resonate with the target's period.
8. **ADVI** uses Gaussian mean-field and full-rank families. The full-rank
   Cholesky diagonal is log-parameterized. Gradients are seeded Monte Carlo
   reparameterization gradients, and parameters are updated by Adam. Three
   project choices:
   - the ELBO is monitored on a fixed set of draws (common random numbers);
   - the run stops when the mean or median of recent relative ELBO changes
     (denominator `max(1, |ELBO|)`) falls below `tolerance`;
   - the reported parameters are the Polyak-Ruppert average of the most
     recent half of the iterates.

   Constant-step Adam still leaves the variances slightly low even after
   averaging. In exploratory runs (5 seeds, 4 gradient samples, tolerance
   `1e-4`, 2-D correlated Gaussian) the mean variance error was about -1.5 %
   at learning rate 0.05 and about -0.6 % at 0.02, so the default learning
   rate is 0.02. ADVI targets unconstrained spaces only; users transform
   constrained parameters themselves.
9. `inverse.CalibrationProblem.sample_posterior` gains
   `method="adaptive-metropolis" | "hmc" | "nuts"`. The default keeps the
   previous code path and random streams, so existing results are reproduced
   bit for bit. HMC and NUTS run on `standard_log_posterior` with
   finite-difference gradients and a diagonal metric started from the Laplace
   variances. `CalibrationPosterior` gains an optional trailing field,
   `sampler`, which defaults to `None`.

## Consequences

- Users get gradient-based MCMC with per-chain divergence and tree-depth
  reporting, and every existing diagnostic still applies.
- With finite-difference gradients, NUTS on cheap, low-dimensional
  calibrations costs far more model runs than adaptive Metropolis (33 340
  against 1 002 evaluations on the 2-parameter line problem). The guide says
  so explicitly.
- Not implemented: multinomial NUTS, Riemannian metrics, automatic
  differentiation, PSIS/Pareto-k diagnostics for ADVI, normalizing-flow
  families, and ADVI inside `sample_posterior`.

## Verification

`tests/test_hmc_nuts_advi.py` (21 tests, about 12 s) checks:

- leapfrog energy error ratios of 4.0 when the step is halved, and
  reversibility to 10 decimals;
- the window schedule;
- NUTS (diagonal and dense metrics) and HMC recover a correlated Gaussian
  within MCSE, with zero divergences;
- acceptance above 0.999 at a tiny step;
- NUTS ESS per gradient more than twice the RWM ESS per evaluation on a 10-D
  Gaussian (measured 0.124 against 0.038);
- divergences on the funnel at a fixed step of 1.0;
- the finite-difference cost identity;
- reproducibility for a fixed seed;
- ADVI full-rank and mean-field optima, and the ELBO rise and gap;
- the calibration default is unchanged, and NUTS agrees with the conjugate
  posterior.

## References

[REF-NEAL-2011], [REF-HOFFMAN-GELMAN-2014], [REF-BETANCOURT-2017],
[REF-NESTEROV-2009], [REF-KUCUKELBIR-ETAL-2017],
[REF-BLEI-KUCUKELBIR-MCAULIFFE-2017], [REF-KINGMA-BA-2015],
[REF-POLYAK-JUDITSKY-1992].
