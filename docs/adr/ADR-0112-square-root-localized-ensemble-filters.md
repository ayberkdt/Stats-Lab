# ADR-0112 — Square-root, localized, and smoothing ensemble filters

- Status: Accepted
- Date: 2026-11-19

## Context

ADR-0018 shipped one ensemble method: the stochastic (perturbed-observation)
EnKF with multiplicative inflation, and it stated that square-root variants
and localization were not implemented. That is enough to show that an
ensemble approaches the Kalman filter as `N` grows, but not to show why
operational ensemble filters look the way they do: with `N = 20` members on
the 40-variable Lorenz-96 model the stochastic EnKF diverges, and a
localized square-root filter with relaxation tracks the truth well below
the observation error. The package also had no chaotic test systems and no
ensemble smoother, and the ODE integrator of ADR-0109 only stopped at
events, so it could not serve as a model transition or be differentiated
for a later 4D-Var.

## Decision

1. **Fixed-step trajectories in `numerics`.**
   `integrate_fixed_step(rhs, t0, y0, *, step, steps, method="rk4")` reuses
   the explicit steps of ADR-0109 ([EQ-NUM-ODE-001]) over a fixed horizon
   and returns every grid state ([EQ-NUM-ODE-003]). It does nothing but
   arithmetic on the state entries and checks only plain floats for
   finiteness, so `reverse_gradient` and forward-mode duals pass through it
   and give the exact derivative of the discrete Runge-Kutta map.
2. **Lorenz test systems in `filtering.benchmark_models`.** `lorenz63` and
   `lorenz96` return `NonlinearStateSpace` objects whose transition is
   `substeps` RK4 steps of size `step` ([EQ-FILT-LORENZ-001],
   [EQ-FILT-LORENZ-002]); the defaults are the published settings (Lorenz
   1963: `sigma = 10`, `rho = 28`, `beta = 8/3`, step `0.01`; Lorenz &
   Emanuel 1998: `J = 40`, `F = 8`, step `0.05`). The model is perfect by
   default (`Q = 0`), `R` and `P_0` default to the identity, the
   transition Jacobian is the forward-mode derivative of the discrete map,
   and the raw tendencies are public.
3. **Every analysis is an ensemble transform.** The ETKF in the symmetric
   square-root form of Hunt et al. ([EQ-DA-ETKF-001]), the serial EAKF in
   Anderson's two-step form ([EQ-DA-EAKF-001]), the LETKF with
   R-localization ([EQ-DA-LETKF-001]), and the stochastic analysis with or
   without Schur-product localization ([EQ-DA-LOC-002]) all return, or can
   return, the matrices `T_j` with `E^a_j = E^f_j T_j` (one shared matrix
   for a global analysis, one per state component for a local one). This
   one representation is what the ensemble smoother needs. The transforms
   are built only on request (`transforms=True`, or by the smoother): they
   cost `O(N^2)` per component, while the ETKF members are updated from
   factored weights in `O(N p)`.
4. **One filter function with options, not one function per variant.**
   `ensemble_kalman_filter` gains `analysis="stochastic" | "etkf" | "eakf"
   | "letkf"`, `localization=`, and `relaxation=`, all keyword-only with
   defaults that reproduce the earlier stochastic filter bit for bit (a
   regression test pins seeded numbers from release 0.25.0). Separate
   public functions would each repeat the stream names, the time indexing,
   the missing-observation rule, the inflation, and the recorded moments of
   ADR-0018, and the smoother would need a variant of each; the options
   are few, orthogonal, and validated together (a global analysis refuses
   a localization, the LETKF requires one, the EAKF requires a diagonal
   `R`). The single-step analyses stay public (`etkf_analysis`,
   `eakf_analysis`, `letkf_analysis`, `localized_ensemble_analysis`). The
   result records the analysis, localization, and relaxation it used.
   To keep the package graph acyclic, the filter loop and its result type
   moved from `enkf` to a new `ensemble_filter` module; the public import
   path `probstat.data_assimilation` is unchanged.
5. **Localization from coordinates.** `Localization` holds one coordinate
   point per state and per observation component, the Gaspari-Cohn
   half-width `c` (weights vanish beyond `2c`, [EQ-DA-LOC-001]), and an
   optional period for cyclic domains such as Lorenz-96.
   `half_width=math.inf` gives weights of exactly one, so the localized
   stochastic analysis then equals the plain one and the LETKF equals the
   ETKF bit for bit (both tested).
6. **Inflation and relaxation stay explicit.** The multiplicative
   `inflation=` keeps its meaning (applied to the forecast anomalies,
   default 1). RTPP ([EQ-DA-RELAX-001]) and RTPS ([EQ-DA-RELAX-002]) are a
   `Relaxation(method, factor)` applied after each analysis, relative to
   the (inflated) forecast that entered it. Adaptive inflation is not
   implemented.
7. **ETKF weights in the smaller space.** The ETKF diagonalizes, with
   `linalg.symmetric_eigen`, either the `N` by `N` matrix
   `(N - 1) I + Yhat' Yhat` (when there are at least `N` observations) or
   the `p` by `p` Gram matrix `Yhat Yhat'` (fewer observations) and forms
   the same symmetric square root through the Woodbury identity, written
   without cancellation for small eigenvalues. The pure-Python Jacobi
   iteration costs about 20 ms for `N = 20`; a LETKF on Lorenz-96 with a
   few local observations per grid point would otherwise spend most of its
   time there. Both branches are tested against the exact Kalman update.
8. **Fixed-lag EnKS on the filter cycle.** `ensemble_kalman_smoother`
   iterates the same private cycle generator as the filter (same streams,
   so its filter part is identical) and multiplies the stored ensembles of
   the last `lag` times by each analysis' transforms ([EQ-DA-ENKS-001]).
   With a localized analysis each lagged component is updated with the
   transform of its own component, and a relaxation acts on the augmented
   state: each updated lagged ensemble is relaxed towards itself before the
   update, as the current analysis is relaxed towards its forecast. (Folding
   RTPS into the lagged transforms instead multiplies the lagged spread by
   the current spread ratio at every step and was rejected after it
   inflated the smoothed variance above the filtered one in a test.) Both
   choices reduce to the published EnKS without localization and
   relaxation.
9. **Conventions of ADR-0018 unchanged.** Time indexing, `None` as a missing
   observation, Cholesky solves for every `R^-1` (whitening by the factor
   of the local block of `R`), `ValueError` for indefinite matrices
   without jitter, and the stream names `initial`, `("forecast", k)`,
   `("analysis", k)`; the deterministic analyses draw no random numbers.

## Consequences

- `tests/test_data_assimilation_advanced.py` checks the ETKF (both
  branches, correlated `R`) and the EAKF against the exact Kalman update of
  the ensemble moments to `1e-10`, the transforms against the members, the
  LETKF with infinite radius against the ETKF exactly, Gaspari-Cohn values,
  the localized filters and the EnKS against Kalman and RTS moments within
  Monte Carlo bands, RTPP/RTPS identities, RK4 order and reverse-mode
  gradients through the trajectory, Lorenz fixed points and shift
  equivariance, and a Lorenz-96 twin experiment (`J = 40`, `N = 20`) that
  asserts only wide inequalities.
- On that twin experiment (seeds 1-4, 40 cycles) the LETKF with `c = 2` and
  RTPS 0.5 reaches a time-averaged analysis RMSE of 0.31-0.36 against an
  observation error of 1, while the non-localized stochastic EnKF has
  2.0-2.4. One LETKF cycle takes about 0.15 s in pure Python.
- 4D-Var, adaptive inflation (Anderson 2007, 2009), localization of the
  serial EAKF, and model-error schemes other than additive `Q` are not
  implemented. ADR-0109's event integrator is unchanged; this ADR extends
  its fixed-step methods to fixed horizons.

## Rejected alternatives

- Separate filter functions per variant: duplicated loop logic, and the
  smoother would need one variant per filter.
- A Cholesky (non-symmetric) square root in the ETKF: it does not keep the
  analysis anomalies centred and makes neighbouring LETKF analyses
  inconsistent.
- Always diagonalizing the `N` by `N` matrix: correct but about ten times
  slower for local analyses with few observations.
- Silent localization or relaxation defaults: they would change the
  numbers of existing code and hide the choice from the user.

## References

[REF-BISHOP-ETHERTON-MAJUMDAR-2001], [REF-HUNT-KOSTELICH-SZUNYOGH-2007],
[REF-ANDERSON-2001], [REF-ANDERSON-JL-2003], [REF-GASPARI-COHN-1999],
[REF-HOUTEKAMER-MITCHELL-2001], [REF-ZHANG-SNYDER-SUN-2004],
[REF-WHITAKER-HAMILL-2012], [REF-EVENSEN-VANLEEUWEN-2000],
[REF-EVENSEN-2003], [REF-LORENZ-1963], [REF-LORENZ-2006],
[REF-LORENZ-EMANUEL-1998], [REF-HAIRER-NORSETT-WANNER-1993],
[REF-GRIEWANK-WALTHER-2008]. Related: ADR-0018, ADR-0109.
