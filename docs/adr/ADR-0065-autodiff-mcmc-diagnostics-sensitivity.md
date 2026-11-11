# ADR-0065: Forward-mode autodiff, MCMC diagnostics, and honest sensitivity error bars

- Status: Accepted (workstream W8 of the 2026-10-16 core-mechanics audit)
- Date: 2026-10-16

## Context

The 2026-10-16 gap audit found that every gradient in the library is a
central difference (P1.7: NUTS/HMC, ADVI, the PPL, FORM, Laplace), that the
MCMC diagnostics lack the split/bulk/tail ESS, quantile MCSE and automatic
warnings of Vehtari et al. (2021) (P2), and that the Sobol error bars are
wrong under quasi-random designs (0.10: the row bootstrap of a scrambled
Sobol design gave coverage 1.00), that second-order indices that were never
computed were reported as 0.0 (P1.10), that every QoI re-ran a full
pick-freeze design (P1.5), and that there was no DGSM or given-data
estimator (P2). `probstat.optimization` was being changed in parallel by W2
and was not touched.

## Decision

1. **`probstat.numerics.autodiff`** (exported from `probstat.numerics`):
   `Dual(value, tangent)` with vector tangents and nesting (`order`),
   operators `+ - * / **`, comparisons/`bool`/`hash` on the innermost value,
   `abs`; `admath` (a namespace class mirroring `math`: exp, expm1, log
   [base], log1p, log2, log10, sqrt, pow, sin, cos, tan, asin, acos, atan,
   atan2, sinh, cosh, tanh, erf, erfc, lgamma, digamma, trigamma [floats
   only], hypot, fabs, fsum, isfinite/isnan/isinf, constants);
   `value_and_gradient`, `gradient`, `derivative`, `jacobian`,
   `value_gradient_and_hessian`, `hessian` (forward over forward,
   symmetrized). `float(dual)`, `int`, `round`, `floor`, `//`, `%`, indexing
   and therefore every `math.*` function and `math.fsum` raise
   `AutodiffUnsupportedError` (a `TypeError`) with a message naming
   `admath` — a dual is never silently converted. `Dual` is deliberately
   *not* registered as `numbers.Real`, so float validators reject it.
   Kinks: derivative 0 at `abs(0)` and `hypot(0, 0)`; poles raise as in
   `math` (`0 ** 0.5` slope, `sqrt(0)` slope). Third derivatives of
   `lgamma` (trigamma of a dual) raise.
2. **`GradientEvaluator`** (numerics) is the shared oracle: option
   `"auto" | "autodiff" | "finite-difference" | callable`. `"auto"` tries
   duals; on a `TypeError` whose float evaluation succeeds it switches
   permanently to the consumer's finite-difference callable and records
   `fallback_reason`. Errors that also occur on floats propagate unchanged.
   The finite-difference callable is injected by each consumer
   (`optimization.central_gradient`), because numerics (layer 0) cannot
   import optimization.
3. **Wiring.** `hamiltonian_monte_carlo`, `no_u_turn_sampler`, `advi`,
   `laplace_approximation` and `ppl.sample/fit_advi/laplace` take
   `gradient="auto"` by default (`None` means the same). Results record the
   method: `HamiltonianChainStatistics.gradient_method /
   gradient_fallback_reason` (+ `HamiltonianResult.gradient_method`,
   `"mixed"` if chains differ), `VariationalResult.gradient_method`,
   `LaplaceApproximation.gradient_method / hessian_method /
   fallback_reason`, `PosteriorSamples.gradient_method /
   gradient_fallback_reason / warnings`. Laplace with duals passes the
   exact gradient to BFGS through the existing `gradient=` parameter and
   takes the exact nested-dual Hessian (one call) instead of the `2 d^2 + 1`
   finite-difference Hessian. Every call of the user log-density (float or
   dual) is counted once; an `"auto"` fallback costs one failed dual call
   per chain.
4. **FORM** (`form_hlrf`): new option `gradient="auto" | "autodiff" |
   "finite-difference" | callable`; autodiff calls the raw forward model
   once with dual inputs at `x = T^-1(u)` and applies the exact Nataf
   Jacobian `grad_u G = L' D grad_x g`, `D_ii = phi(z_i) / f_i(x_i)`
   ([EQ-REL-FORM-AD-001]). **The default stays `"finite-difference"`**
   (`None` keeps its old meaning): on the benchmark four-branch system,
   `g = min(branches)` has a symmetric kink at the start point; central
   differences average the branches, the gradient vanishes and FORM stops
   with the recorded known limitation, whereas autodiff silently follows
   the first active branch and reports one branch's design point
   (`tests/test_benchmarks.py` pins the failure). `FORMResult` gains
   `gradient_method` and `gradient_fallback_reason`; `gradient_evaluations`
   now also counts autodiff gradients. `inverse_form` is unchanged.
5. **PPL.** Density mode is dual-transparent (bijections, log-Jacobians,
   the inlined normal density, non-centred sites, deterministic sites,
   `log_pdf` results are no longer forced to `float`). `ModelIR.gradient`
   gains `method=` and `ModelIR.gradient_evaluator()` is new. *Limit:* the
   classes of `probstat.distributions` validate their parameters as reals,
   so `Normal(mu, 1.0)` with a latent `mu` rejects duals and the PPL falls
   back to finite differences (recorded, reason "mu must be a real
   number"). Models built from dual-aware duck-typed distributions are
   differentiated exactly. Making the distribution constructors accept
   duals is a follow-up for the distributions owner.
6. **MCMC diagnostics** (`inference.mcmc`): `effective_sample_size` splits
   chains by default (`split=False` gives the old whole-chain value);
   new `bulk_effective_sample_size`, `quantile_effective_sample_size`,
   `tail_effective_sample_size`, `mcse_quantile` ([EQ-MCMC-ESS-002],
   [EQ-MCMC-MCSE-002]); `ParameterSummary` gains
   `bulk_effective_sample_size`, `tail_effective_sample_size`,
   `mcse_quantiles` (defaults `None`/`()` for old positional construction).
   `convergence_warnings(result)` applies Vehtari et al.'s rules
   (rank-normalized R-hat > 1.01; bulk- or tail-ESS < 100 x chains; stuck
   coordinates; fewer than 4 draws); `hamiltonian_warnings` adds any
   post-warm-up divergence, E-BFMI < 0.3 per chain, and tree-depth
   saturation. All samplers (`random_walk_metropolis`,
   `adaptive_metropolis`, HMC, NUTS) run the checks after sampling unless
   `check_convergence=False`, emit each message as `ConvergenceWarning`
   (stack level pointing at the sampler's caller) and store them in
   `MCMCResult.warnings` / `HamiltonianResult.warnings`.
7. **Multinomial NUTS** is the new default (`variant="multinomial"`):
   uniform progressive sampling inside subtrees, biased progressive
   sampling at the top level, divergence `H - H_0 > Delta_max`
   (Betancourt 2017, Appendix A.3). `variant="slice"` keeps the previous
   sampler. Both use the Hoffman-Gelman end-point U-turn criterion, not
   Stan's generalized criterion.
8. **SBC** (`inference.sbc`): `simulation_based_calibration(prior_sampler,
   simulator, posterior_sampler, replicates=, bins=, seed=)` returns
   `SBCResult` (ranks, histograms with exact expected counts for any bin
   count, Pearson statistics, p-values, `calibrated(level)` with a
   Bonferroni split); `rank_statistic` breaks ties at random.
9. **Sensitivity.**
   - `sobol_analysis` labels its intervals (`interval_method`):
     `"bootstrap"` only for `design="mc"`; for `sobol`, `halton`, `lhs` any
     interval request (`bootstrap > 0` or `replicates=R`) splits the `size`
     rows into `R` independent randomizations (default 8) and reports
     `mean +- t_(R-1) s / sqrt(R)` ([EQ-SENS-RQMC-001]); the cost stays
     `size (d + 2)`. `ModelSobolResult` gains `interval_method` and
     `replicates`; `bootstrap` is the number of row resamples actually used.
   - `SobolResult.second_order` is `None` when not computed (was a matrix
     of 0.0); `estimate_sobol_indices` still computes it.
   - `sobol_analysis_all(model, size, quantities=None, ...)` evaluates one
     pick-freeze design for every QoI (one model call returns all QoIs) and
     returns `MultiQuantitySobolResult`; each entry equals the
     single-QoI call with the same seed. `UQStudy.sensitivity` (workflow,
     not owned) still calls `sobol_analysis` per QoI.
   - `dgsm(model, quantity, size, design=, gradient="auto")` estimates
     `nu_i = E[(df/dx_i)^2]` with autodiff through the forward model (one
     call per point) or central differences (`2 d + 1`), and the bound
     `S_Ti <= C(mu_i) nu_i / D` with the optimal Poincare constants
     `(b-a)^2/pi^2` (uniform; `nu_i / (pi^2 D)` on `[0, 1]`) and `sigma^2`
     (normal); other laws get `None`, not a guessed constant.
   - `given_data_first_order(inputs, outputs, partitions=None)`: equal-count
     class partition of the sorted sample (Plischke, Borgonovo & Smith
     2013), raw `SSB/SST` and Kelley's bias-corrected
     `1 - (n-1)/(n-M) SSW/SST`; `M = max(2, min(50, floor(sqrt n)))`. The
     raw bias `~(M-1)(1-S)/(n-1)` and the `O(1/M^2)` class-averaging bias are
     documented; dependent inputs are allowed.

## Evidence

Autodiff (`tests/test_autodiff.py`): value, gradient and Hessian of a
13-term function using every `admath` function equal mpmath 1.4
`mpmath.diff` (50 digits) to 1e-13 (gradient) and 1e-12 (Hessian); gradient
equals the complex-step derivative (h = 1e-30); closed-form derivatives of
each function; quadratic-form Hessian exact; failure modes raise.

Evaluation savings (`tests/test_mcmc_diagnostics_autodiff.py`, scratch
benchmark): NUTS on a 10-D Gaussian with scales 0.5..2.75, 4 chains,
1000 warm-up + 1000 draws, seed 11:

| gradient | variant | gradient evals | log-density calls | wall time |
|---|---|---|---|---|
| autodiff | multinomial | 53,121 | 53,121 | 4.7 s |
| finite difference | multinomial | 52,619 | 1,105,003 | 4.3 s |
| autodiff | slice | 50,812 | 50,812 | 3.5 s |
| finite difference | slice | 51,628 | 1,084,192 | 4.3 s |

Autodiff makes 21x (= 2d + 1) fewer calls of the user function. In this
pure-Python implementation a dual evaluation costs about `d` float
evaluations, so wall time is similar for a cheap density; the saving is real
whenever the log-density is expensive relative to arithmetic (simulators,
data loops), and the gradient is exact. FORM on a lognormal/normal limit
state: 12 vs 30 model calls (independent), 14 vs 35 (correlated, rho = 0.4),
same beta to 1e-12, same 5/6 iterations.

Diagnostics: split, bulk, tail and quantile ESS and the quantile MCSE equal
a numpy/scipy port of stan-dev/posterior to 1e-12 on two chain sets (the
port reproduces the earlier unsplit oracle 825.6296706351059); bulk-ESS of
AR(1) chains matches `n (1-phi)/(1+phi)` within 15 %; the quantile MCSE of
i.i.d. normal draws averages 1.08 (p = 0.5) and 0.93 (p = 0.9) times
`sqrt(p(1-p)/S)/f(q_p)` over 8 data sets.

Multinomial NUTS: one-transition invariance test — 2000 exact draws of
`log Gamma(3)`, one transition each at a fixed step 0.9 (with divergences),
Pearson statistic on 20 CDF bins below the 0.1 % point for both variants
(scratch values 11.7 / 6.9 at step 1.2 and 19.9 / 9.1 at 0.3 with 4000
draws); 10-D Gaussian moments within 4 MCSE and sd within 12 % for both
variants; z-scores of `E[x^2]` across seeds 1-3 were all within +-2.2.

SBC: 400 replicates, L = 99, conjugate normal posterior passes; a posterior
with half the correct sd gives p < 1e-6 and a U-shaped histogram.

Sobol intervals (40 seeds, Ishigami, 512 rows): replicate intervals
(8 x 64) cover the six analytic indices in 93.8 % of cases (nominal 95 %)
with mean half-width / (1.96 x seed-to-seed sd) between 1.00 and 1.30. The
former row bootstrap on one 512-row scrambled Sobol design covered 100 %
with half-widths 2.6-4.7 times too large. Splitting into replicates costs
accuracy: the seed-to-seed sd of `S_1` is 0.025 with 8 x 64 rows vs 0.016
with one 512-row design.

DGSM: Ishigami `nu = (0.5 (1 + 0.04 pi^4 + 0.01 pi^8/9), 24.5,
0.08 pi^6/7)` within 3 % at 4096 Sobol points, autodiff and finite
differences agree to 1e-6; the bound is attained (1.00 +- 0.01) for
`cos(pi x)` on U(0,1) and for a linear map of normals. Given-data indices
recover the Ishigami first-order indices within 0.02 (n = 20000) and
`S_2 = rho^2` for correlated inputs.

## Consequences

- Default NUTS output changes (multinomial variant, exact gradients when
  the density accepts duals); seeded draws differ from 0.15. All existing
  tests pass; two tests that asserted the finite-difference cost now pass
  `gradient="finite-difference"` explicitly, and the posterior-oracle ESS
  test passes `split=False`.
- Samplers now emit `ConvergenceWarning`s; short test or teaching runs will
  show them (silence with `check_convergence=False` or a warnings filter).
- `effective_sample_size` defaults to split chains, so ESS and
  `mcse_mean` values in summaries are slightly different (more
  conservative under drift).
- The workflow (`UQStudy.sensitivity`) still passes `bootstrap=200` with the
  Sobol design; it now gets 8 replicate intervals at the same cost.
  Switching it to `sobol_analysis_all` is left to the workflow owner.
- Stability: `numerics` is labelled stable package-wide (ADR-0048); the
  autodiff module states in its docstring that it is experimental within
  that package until the next minor release.

## Not done

- Reverse mode; sparse tangents; `gamma`, `erfinv`, `trigamma` of duals.
- Dual support in `probstat.distributions` and `ProbabilisticModel.evaluate`
  (both validate reals), which would let the PPL and calibration
  log-densities use autodiff.
- Stan's generalized (momentum-sum) U-turn criterion; PSIS-LOO/WAIC.
- Poincare constants for laws other than uniform and normal.
