# ADR-0145: Aerospace propulsion models, advanced Gaussian processes, hybrid physics-ML and the research inverse layer

- Status: Accepted (experimental APIs)
- Date: 2026-11-27

## Context

The library could propagate, rank and bound the uncertainty of any
`ProbabilisticModel`, but it offered no engineering model to run those tools
on, and three method families that space-propulsion and inverse-problem
groups use daily were missing or thin: Gaussian-process emulators beyond a
single ARD squared-exponential kernel with point hyperparameters; reduced and
hybrid physics-data models; and the posterior samplers, transport maps and
design criteria that make a nonlinear PDE inversion credible. The
uncertainty work of agencies and launch providers on engines and vehicles
leans on exactly these pieces: subset simulation and transitional MCMC
(Au-Beck, Ching-Chen), FORM sensitivities, outcrossing methods, emulator
calibration with a discrepancy (Kennedy-O'Hagan), and demonstration test
planning.

Constraints carried over: no new runtime dependency; the layer map of
`tests/test_architecture.py`; every formula traced to an `EQ-` record and a
source; existing defaults and seeded results unchanged; new names
experimental.

## Decision

1. **New package `probstat.aerospace` (layer 7, experimental).** Six modules
   build `ProbabilisticModel`s or deterministic solvers with explicit units
   and validity ranges: `liquid_engine` (ideal-rocket `c*`, `CF`, `Isp`,
   area-Mach inversion, divergence, the Summerfield separation flag,
   mixture-ratio tables, the turbine-pump balance) [EQ-AERO-LRE-001]-[EQ-AERO-LRE-010];
   `solid_motor` (0D ballistics, Vieille burn rate, tubular, end-burning and
   BATES grains, MEOP and burst margin) [EQ-AERO-SRM-001]-[EQ-AERO-SRM-007]; `combustion_instability` (`n`-`tau`
   modes with a Marble-Candel choked nozzle, the Rayleigh index, eigenvalue
   sensitivities) [EQ-AERO-CI-001]-[EQ-AERO-CI-007]; `ascent` (planar ascent through the 1976 standard
   atmosphere, staging, insertion, dispersion intervals, ellipses and an RSS
   budget) [EQ-AERO-ASC-001]-[EQ-AERO-ASC-008]; `engine_reliability`
   (classical and Bayesian demonstration plans, heritage priors, Weibull
   test-time equivalence, growth, clusters with beta-factor common cause,
   launch event trees) [EQ-AERO-REL-001]-[EQ-AERO-REL-012]; `conjunction` (exact, Foster, Chan and small-object
   collision probability, dilution, scale uncertainty)
   [EQ-AERO-PC-001]-[EQ-AERO-PC-009]; `entry` (planar entry with optional
   lift, the Allen-Eggers closed form, Sutton-Graves heating with the air
   coefficient `K = 0.1113 kg/(s m^1.5 atm^0.5)` of NASA TR R-376 converted
   to `k = 1.7483e-4 kg^0.5/m`, ablative heat-shield sizing, entry
   corridors) [EQ-AERO-ENTRY-001]-[EQ-AERO-ENTRY-011]; `nozzle_cooling`
   (Bartz gas-side heat transfer, the wall resistance network, Dittus-Boelter
   coolant convection with an exact energy balance)
   [EQ-AERO-COOL-001]-[EQ-AERO-COOL-009]; `debris_risk` (Poisson penetration
   risk, parameterized ballistic limits whose published coefficients the
   user supplies, epistemic factors, Gamma-Poisson updating)
   [EQ-AERO-MMOD-001]-[EQ-AERO-MMOD-011]; `shell_buckling` (classical axial
   buckling, the SP-8007 knockdown of its equations 7-10, Koiter's
   imperfection law with the constant `(3/2) sqrt(3(1 - nu^2))` that
   equations 138-139 of SP-8007 revision 2 imply, a probabilistic knockdown)
   [EQ-AERO-BUCK-001]-[EQ-AERO-BUCK-007]; `health_monitoring` (a
   chamber-pressure digital twin with joint efficiency estimation, CUSUM
   and GLR detection, inverse-Gaussian remaining-useful-life prognostics)
   [EQ-AERO-PHM-001]-[EQ-AERO-PHM-007]; `turbopump` (suction head and
   cavitation probability) [EQ-AERO-PUMP-001]-[EQ-AERO-PUMP-009];
   `entry_reconstruction` (density profile and ballistic coefficient from
   entry data) [EQ-AERO-RECON-001]-[EQ-AERO-RECON-007]; `orbital_decay`
   (lifetime under drag and disposal compliance)
   [EQ-AERO-DECAY-001]-[EQ-AERO-DECAY-008]; `wind_loads` (q-alpha loads
   through a synthetic wind field) [EQ-AERO-WIND-001]-[EQ-AERO-WIND-006];
   `attitude_estimation` (gyro-stellar filter and pointing budget)
   [EQ-AERO-ATT-001]-[EQ-AERO-ATT-009]; `power_budget` (end-of-life energy
   balance) [EQ-AERO-POWER-001]-[EQ-AERO-POWER-010]; `reentry_casualty`
   (fragment demise and casualty expectation)
   [EQ-AERO-CASUALTY-001]-[EQ-AERO-CASUALTY-009]. All
   example vehicles
   are synthetic.
2. **Reliability and inference methods beside the existing ones.**
   `inference.transitional_mcmc` (TMCMC with the log evidence, model-class
   selection, model averaging, hierarchical calibration);
   `reliability.time_variant` (Rice and PHI2 outcrossing, first passage by
   Monte Carlo and subset simulation); `reliability.reliability_sensitivity`
   (FORM and score-function sensitivities, failure-indicator Sobol indices).
3. **Gaussian processes.** `surrogates.gp_advanced` (kernel algebra,
   derivative observations, sparse DTC/FITC/VFE, heteroscedastic,
   multi-output LMC and operator-constrained GPs) and `surrogates.gp_bayes`
   (fully Bayesian hyperparameters, Bastos-O'Hagan diagnostics, GP Sobol
   indices with posterior uncertainty, Laplace classification, Student-t and
   warped processes), and `surrogates.nonlinear_multifidelity` (NARGP)
   [EQ-SUR-NARGP-001]-[EQ-SUR-NARGP-007], and `surrogates.gp_anomaly` (GP
   telemetry anomaly detection) [EQ-SUR-GPANOM-001]-[EQ-SUR-GPANOM-008].
   `validation.koh_emulator` adds Kennedy-O'Hagan
   calibration of an expensive simulator through a GP emulator in the
   modular form, with the exact block covariance and the predictive
   variance split into emulator, discrepancy, cross, parameter and noise
   parts [EQ-VAL-KOHEM-001]-[EQ-VAL-KOHEM-007]. `validation.functional_calibration`
   calibrates a field-valued simulator through a principal-component basis
   with GP emulators of its weights and a basis discrepancy, on the projected
   data [EQ-VAL-FUNCEM-001]-[EQ-VAL-FUNCEM-008].
4. **Hybrid physics-ML.** `surrogates.reduced_order` (POD, operator
   inference, DMD, EDMD, SINDy and ensemble SINDy) and
   `surrogates.physics_informed` (PINNs with exact jet derivatives, hybrid
   ODEs, residual hybrids with split-conformal intervals, DeepONet), and
   `surrogates.latent_force` (a linear ODE driven by a GP force; closed-form
   first-order covariances and an exact Kalman form for second-order
   channels) [EQ-SUR-LFM-001]-[EQ-SUR-LFM-008].
5. **Research inverse layer.** `inverse.laplace_nonlinear` (Gauss-Newton
   Laplace with a low-rank Hessian, RTO, RML, pCN, gpCN, DILI);
   `inverse.heat_conduction_inverse` (the inverse heat conduction problem
   of a slab with embedded thermocouples, exact discrete adjoint, Beck's
   sequential function specification, Tikhonov, CGLS and a Gaussian-process
   posterior) [EQ-INV-IHCP-001]-[EQ-INV-IHCP-012];
   `inverse.inertia_identification` (inertia tensor from gyro data)
   [EQ-INV-INERTIA-001]-[EQ-INV-INERTIA-009];
   `inverse.thermal_correlation` (thermal model correlation)
   [EQ-INV-THERM-001]-[EQ-INV-THERM-008];
   `inverse.modal_updating` (structural model updating from modal data)
   [EQ-INV-MODAL-001]-[EQ-INV-MODAL-009];
   `inverse.local_approximation_mcmc` (Metropolis-Hastings with a local
   regression surrogate refined by true runs, asymptotically exact when the
   refinement never stops, and delayed acceptance with it)
   [EQ-INV-LOCAL-001]-[EQ-INV-LOCAL-006];
   `inverse.transport_maps` (triangular maps from samples or a density,
   SVGD, comparison with a reference); `inverse.benchmark_aristoff_bangerth`
   (the published 64-parameter elliptic benchmark with an exact discrete
   adjoint) [EQ-INV-ABBENCH-001]-[EQ-INV-ABBENCH-005];
   `design.ihcp_sensor_design` (thermocouple depths for heat-flux
   reconstruction) [EQ-DES-IHCPD-001]-[EQ-DES-IHCPD-007];
   `design.nonlinear_optimal` (nested Monte Carlo, contrastive and Laplace
   EIG); `workflow.inverse_study` (`InverseStudy`, one seeded, counted
   report from spectrum to comparison) [EQ-WF-INVSTUDY-001]-[EQ-WF-INVSTUDY-005].
6. **Hot paths, bit-identical.** Inner loops of dense linear algebra, the GP
   likelihood, input transforms, quantiles, subset simulation, random-walk
   Metropolis and the Sobol and Latin-hypercube generators were rewritten
   with hoisted locals and `math.fsum(map(operator.mul, a, b))`. Every
   change reproduces the old floating-point bits (`float.hex`) and
   exceptions; `math.sumprod` is not used because it rounds differently
   and is missing on Python 3.11.

## Consequences

- Speed-ups measured on 2026-11-27 against the committed code (median of
  three, machine shared with other jobs): Cholesky 1.8-2.7x, SPD solve
  2.0-2.3x, SPD inverse 2.4-2.8x, Householder QR 1.5x, subset simulation
  2.2-2.9x, crude Monte Carlo 1.5x, random-walk Metropolis 1.5x, Sobol
  sequence 1.7x, Latin hypercube 1.8x; SVD, LSQR, Sobol indices and the GP
  log-likelihood gradient unchanged within noise. A GP fit of 60 points in
  three dimensions: 1.25x. A third pass (input validation, Sobol-index
  estimation, LSQR) measured Sobol indices on the Ishigami function 1.37x
  and LSQR 1.27x faster in a second run against the committed code
  (median of three), bit-identical; SVD (1.05x) was left as it was
  because no safe gain was found. A fourth pass on the operators and the
  tridiagonal solves measured, against the committed code on a quiet
  machine: LSQR on a convolution 1.86x, first differences 2.42x, image
  gradients 1.46x, `materialize` 2.76x, the heat-source operator 1.50x, the
  slab-flux operator 1.16x and the elliptic solve 1.76x. Its convolution
  kernels are generated from the tap count alone (at most 24 taps) and
  cached; no user data reaches the generated source.
- A fifth pass validates each counted operator application once instead of
  twice and forms the LSMR iterate lazily. All passes together, measured
  against the committed code (median of three): counted convolution
  applications 2.70x, LSQR on a convolution 2.04x (damped 2.00x), LSQR on a
  dense operator 1.80x, CGLS 1.67x, conjugate gradients 1.61x, LSMR 1.35x;
  `lanczos` stayed flat (1.07x), bounded by the dense eigensolver.
- A sixth pass on the GP: fitting is bound by the exact summations that
  bit-identity requires, so a 60-point fit in three dimensions stays near
  1.14-1.23x over the committed code; one likelihood-and-gradient call takes
  about 11 ms against 14-15 ms (in-process median over 300 calls), and
  prediction with variance at 500 points runs 1.82x faster. This pass also
  found that the first pass's kernel assembly raised a different overflow
  error than the committed code when two entries of one row overflowed in
  different ways; it now replays a failing row in the original order, and a
  test covers the case.
- Shell buckling: with the SP-8007-implied Koiter constant and a lognormal
  imperfection amplitude of mean 0.3 mm (c.o.v. 0.5) on a 1.5 m radius, the
  1 % probabilistic knockdown
  is above the SP-8007 value at every `R/t` from 200 to 1450 (0.602 against
  0.471 at 200; 0.275 against 0.182 at 1450), so SP-8007 is conservative for
  that assumed law; a mean of 1.5 mm reverses it (0.196 against 0.322 at
  `R/t = 500`). An earlier draft used a constant twice as large; reading the
  report caught it.
- Field-output calibration: over 20 seeds a biased simulator without a
  discrepancy covered the true parameter in 0 cases and with a discrepancy
  in 19; the field posterior was 2.6-2.9 times tighter than a calibration at
  one location; the field band covered the noise-free response at new
  controls in 0.84 of cases (nominal 0.95).
- Health monitoring: CUSUM run lengths from Siegmund's approximation agree
  with the Brook-Evans integral equation within 3.6 %; on filter
  innovations the in-control run length at threshold 5 was 466 against 469
  predicted; the 80 % remaining-useful-life interval covered the true life
  in 0.82 of 300 synthetic runs.
- Thermocouple placement: under linear, independent sensor noise the best
  sensors were the shallowest candidates in every one of 108 settings
  (greedy equal to exhaustive), and halving one sensor's noise beat adding a
  second sensor on all four criteria.
- The hot-fire laboratory: with a model-form error in the synthetic engine,
  the 95 % predictive band of vacuum `Isp` covered 30 held-out operating
  points in 100 % of cases with a discrepancy and 13 % without, and the
  posterior sd of the efficiency was 17 times smaller without the
  discrepancy, which is overconfidence, not information.
- Latent force models: on three coupled synthetic thermal channels the
  extrapolation error of a plain multi-output GP was a median 6.9 times
  that of the latent force model (10 seeds); with every parameter unknown
  only ratios of the decays were identified, and the force band covered
  the truth in 0.43 of cases on average.
- Entry reconstruction: without lift and with a known entry state the
  trajectory depends on density over ballistic coefficient only (a test
  scales both and reproduces the trajectory to rounding); radar fixes
  sharpened that ratio about 5 times but cut the separate uncertainties by
  only about 5 %, while a prior on the ballistic coefficient or one density
  anchor separated them. Over 40 synthetic flights the batch 95 % band
  covered the true density at 0.95 of points and the filter band at 0.92.
- Turbopump: in the synthetic case the scatter of the required suction
  specific speed carried 0.78 of the variance of the cavitation margin and
  the propellant temperature 0.07; the NPSH definitions are standard forms
  derived again, their sources not read.
- Orbital lifetime: with the solar-cycle phase fixed, FORM, subset
  simulation and Monte Carlo agreed on the probability of missing a
  25-year deadline (0.177, 0.167, 0.167 at 640 km); with the phase
  uncertain FORM was unreliable (0.407 against 0.318, and no convergence at
  600 km) because the lifetime is periodic in the phase, so only simulation
  is compared there.
- NARGP: on the nonlinear one-dimensional benchmark its error was 0.033
  against 0.338 for linear co-kriging, but with plug-in hyperparameters its
  95 % intervals covered 0.55-0.83 of test points.
- Modal updating: on a mirror-symmetric structure with frequency data only
  the Laplace fit locks into one of two equally good modes and reports
  near-certainty about which spring is damaged, while transitional MCMC put
  0.47 of its samples in each mode; mode shapes at both points removed the
  ambiguity. With damage in one segment, the 95 % Laplace intervals covered
  the true stiffnesses in 40 of 40 cases.
- Wind loads: without wind the loads reproduce the ascent simulation bit
  for bit; at the synthetic limits FORM, subset simulation, Monte Carlo and
  GEV/GPD tail fits agreed (2e-3 level at 6000 profiles) because the peak
  load always occurs near maximum dynamic pressure.
- Attitude estimation: the derived closed-form steady-state accuracy agreed
  with the Riccati solution to 1e-7 over 300 parameter sets; the
  Allan-variance fit's standard errors were 2.0-2.6 times too small when
  correlation between cluster sizes was ignored, so they are inflated by a
  measured factor of 2.3, stated as such.
- Thermal correlation: steady tests left two conductances in series at
  their prior uncertainty (correlation -0.998); a transient test shrank
  their standard deviations about six times but kept the correlation near
  -1; with an unmodelled coupling the 90 % flight band covered 0.53 of
  predictions without a discrepancy and 0.975 with one.
- Power budget: on the minimum of the power and depth-of-discharge margins
  FORM followed one branch and gave 0.277 against Monte Carlo 0.375 (series
  bounds 0.277-0.488 contain it), a second case where FORM is the wrong
  tool.
- Inertia identification: the finite-difference regression estimator had a
  trace bias of -1.8 (about -4.8 %) that did not shrink from 240 to 480
  samples (errors in variables), while the output-error fit stayed within
  +0.02; with realistic torques, gyroscopic coupling let single-axis
  excitation identify every entry.
- Reentry casualty: the casualty expectation is step-valued (each fragment
  survives or not), so FORM stops with a vanishing gradient on it; on the
  continuous demise margin of the critical fragment FORM and subset
  simulation agree with simulation (about 0.09-0.10).
- GP anomaly detection: at a 1 % target the Gaussian threshold gave a 2.7 %
  false-alarm rate on heavy-tailed noise and the split-conformal threshold
  1.9 % (400 calibration scores, serially dependent data); slow drifts are
  absorbed by the sliding window and poorly detected.
- Local-approximation MCMC on a two-parameter decay problem used 102 true
  evaluations for 1900 iterations and gave 17x the effective samples per
  true evaluation of plain Metropolis (52x on a 15 000-draw run); the
  delayed-acceptance variant was exact but saved nothing on that case.
- Debris risk: the gap between the mean probability of no penetration and
  its value at the mean inputs is not always positive; with an uncertain
  ballistic limit it was negative (-5.8e-4), while `E[PNP] - exp(-E[N])`
  is non-negative by Jensen's inequality, and both are tested.
- Nozzle cooling: the Bartz multiplier carries 0.88 of the variance of the
  peak wall temperature in the synthetic case; FORM gives 0.070 for the
  limit exceedance against 0.0605 by Monte Carlo.
- Inverse heat conduction: with a sensor 2 mm deep and 1 % noise, relative
  errors of 0.05-0.09 (triangular pulse) and 0.10-0.16 (step) across the
  five estimators; errors grow with sensor depth; GCV at 6 mm chose a
  near-zero weight (error about 50) while reporting an interior minimum, so
  the discrepancy rule is the safer default there; Beck's method with one
  future step diverges at 4 mm.
- Limitations recorded where the code lives: the residual hybrid's
  split-conformal coverage falls to 0.07 when it extrapolates (nominal
  0.9); gpCN mixes poorly on some seeds (ESS 193 against 1237 on another
  seed); the KOH posterior depends strongly on the discrepancy-amplitude
  prior; short chains on the Aristoff-Bangerth benchmark miss the
  heavy-tailed means (`theta_0` 3.3 against the published 76.3); subset
  simulation at 500 samples per level has a coefficient of variation near
  0.5 on the motor-burst case; FORM overstates the instability probability
  (0.170 against 0.151 by Monte Carlo) and the vacuum-`Isp` shortfall
  probability (0.0529 against 0.0422), where SORM-Breitung gives 0.0446.
- The planar ascent ignores Earth rotation, J2, lift and wind;
  `max_dynamic_pressure` is the maximum on the integration grid; above
  86 km the atmosphere is an isothermal extension, flagged.

## Verification evidence

- Oracles in the tests: the uniform-gravity vertical burn and Tsiolkovsky
  increment (1e-8 relative); the 1976 atmosphere's closed-form layers
  against a separate RK4 hydrostatic integration (1e-8); Kepler invariants
  over a coast (1e-9); the Aristoff-Bangerth reference forward values (169
  values, max error under 5e-10) and published log-likelihoods (about
  1e-8); the exact linear-Gaussian posterior for `InverseStudy` (MAP 1e-8,
  evidence 1e-7) and a brute-force grid for its quadrature reference; the
  KOH joint covariance entry by entry against an independent block matrix
  and the linear-in-`t` calibration against the exact posterior; the
  bit-for-bit equivalence suites (44 tests, 9804 subtests).
- Measured comparisons: KOH coverage of the true parameter over ten
  predeclared seeds, 9 of 10 with a discrepancy and 0 of 10 without; the
  motor-burst probability by crude Monte Carlo with 40 000 burns,
  0.00333 (standard error 0.00029), against four subset runs of 0.0024 to
  0.0030.
- Bibliography: DOIs checked against Crossref on 2026-11-27 (including
  Aristoff-Bangerth, SIAM Review 65(4) 1074-1105; Bayarri et al.,
  Technometrics 49(2) 138-154; Elman-Silvester-Wathen, 2nd edition);
  arXiv records against DataCite; the 1976 standard atmosphere against the
  NASA Technical Reports Server record 19770009539 (report numbers and
  date), with the layer pressures used in the tests cross-checked against a
  secondary tabulation only; the Curtis textbook against a catalogue record
  that states no edition; Allen-Eggers and Sutton-Graves against their
  NASA Technical Reports Server records, with the Sutton-Graves air
  coefficient read from table II of the report itself; the Beck,
  Blackwell and St. Clair and the Carslaw and Jaeger books against
  catalogue records; Christiansen, Cour-Palais, Whipple and Conrad et al.
  against Crossref; the Bartz note only as far as its DOI resolves (the
  Technical Notes section of Jet Propulsion 27(1), 1957), since no record
  read gives its title or author; NASA SP-8007 (1968) and its revision 2
  (2020) against their NASA Technical Reports Server records, with
  equations 7-10 and 138-139 of revision 2 read from the report; Higdon et
  al. (2008) against Crossref.

References: [REF-SUTTON-BIBLARZ-2017], [REF-AU-BECK-2001],
[REF-ALLEN-EGGERS-1958], [REF-SUTTON-GRAVES-1971],
[REF-BECK-BLACKWELL-STCLAIR-1985], [REF-CARSLAW-JAEGER-1959],
[REF-CONRAD-MARZOUK-PILLAI-SMITH-2016], [REF-CHRISTIANSEN-1993],
[REF-BARTZ-1957], [REF-NASA-SP-8007-2020],
[REF-HIGDON-GATTIKER-WILLIAMS-RIGHTLEY-2008], [REF-BROOK-EVANS-1972],
[REF-SIEGMUND-1979], [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008],
[REF-ALVAREZ-LUENGO-LAWRENCE-2013], [REF-PERDIKARIS-ETAL-2017],
[REF-BECK-KATAFYGIOTIS-1998], [REF-FARRENKOPF-1978],
[REF-MARKLEY-CRASSIDIS-2014],
[REF-KENNEDY-OHAGAN-2001], [REF-BAYARRI-ETAL-2007],
[REF-ARISTOFF-BANGERTH-2023], [REF-US-STANDARD-ATMOSPHERE-1976],
[REF-CURTIS-2014].
