# ADR-0061: Continuous-time Markov chains and analytic Markov reliability

- Status: Accepted (proposed during the 2026-10-16 session, workstream W4)
- Date: 2026-10-16

## Context

The library had a discrete-time `MarkovChain` and exactly simulated
continuous-time processes. It had no continuous-time Markov chain. Users
asked for success and failure probabilities of repairable systems, standby
redundancy and mission phases, computed analytically rather than by Monte
Carlo. The two practical obstacles are these:

- **Stiffness.** Failure rates are about `1e-6`, repair rates about `1e3`,
  and missions last up to `1e6` h. So `q t` reaches `1e9`, and the
  quantities of interest are tiny: unreliabilities of `1e-13` to `1e-9`.
- **Honesty.** A truncated series must not be reported as exact.

## Decision

1. **`probstat.processes.ctmc`** (layer 2) provides
   `ContinuousTimeMarkovChain(states, rates)`. The rates are given either as
   a `{(from, to): rate}` mapping or as a generator matrix. Labels can be any
   hashable values. The diagonal is always recomputed as minus the row sum.
   Results are frozen dataclasses: `TransientSolution`, `OccupancySolution`,
   `CTMCAbsorption`, `CTMCPath` and `BoundedValue` (a value with an absolute
   error bound).
2. **Transient law by uniformization** (Jensen; Grassmann; Reibman-Trivedi).
   - The Poisson window `[L, R]` is chosen from Cramer-Chernoff tail bounds
     `exp(-lambda)(e lambda / k)^k`, evaluated in log space as
     `(k - lambda) - k log1p((k - lambda)/lambda)`. Nothing underflows at
     `lambda = 1e7` or beyond.
   - The weights are generated from the mode outward, in the style of Fox
     and Glynn, and normalized. The L1 truncation error is then at most
     `2 Pr(N outside window) <= tolerance`. This is rigorous.
   - The rounding bound is first order and componentwise **relative**:
     `u [m (n + 3) + 2 span + (R - L) + 4]`. It is relative because every
     operand is non-negative. Both bounds are reported, and so is their sum.
   - Each iterate `v_k P` is rescaled to the initial mass. This is exact in
     real arithmetic because `P` is stochastic, and it removes the
     systematic drift caused by rounded row sums. On the stiff test chain at
     `q t = 1e5`, the L1 error fell from `1.1e-11` to `2.4e-16`.
   - If an iterate repeats bit for bit, the remaining weights are summed in
     closed form. This changes no result.
3. **Scaling and squaring** for large `q t` (`method="auto"` picks the
   cheaper path from an operation count; `"series"` and `"squaring"` force
   one path). The rows of `exp(Q t / 2^s)` and of
   `int_0^(t/2^s) exp(Q u) du` come from the series. Then `M <- M + E M` and
   `E <- E^2` are applied `s` times.
   - The truncation budget is `tolerance / 2^(s+1)`.
   - The bounds propagate as `d <- 2d + d^2` and `d_M <- 2 d_M + d tau_k`.
   - The rows of `E` are rescaled to sum one after each squaring. This is
     exact in real arithmetic, because the truncated rows built from
     normalized weights are exactly stochastic, and so are their products.
4. **Rewards.** `expected_time_in_states` uses
   `(1/q) sum Pr(N > k) pi(0) P^k`. It has a rigorous truncation bound that
   uses `E[(N - R)^+] <= lambda Pr(N >= R)`, and the window is widened until
   this bound is at most `tolerance * t`. `expected_accumulated_reward`
   (rate rewards) and `expected_transition_count` (impulse reward per
   transition) are built on top.
5. **Steady state and absorption without subtraction.**
   - `steady_state` applies GTH state reduction to the unique closed class.
   - `limiting_distribution(initial)` handles several closed classes.
   - `absorption(initial)` uses a GTH-type elimination for `(-Q_TT) X = B`.
     The diagonal is recomputed as the sum of the off-diagonal rates plus
     the exit rate.
   - `embedded_chain()` returns the existing `MarkovChain`.
   - `simulate()` is a seeded cross-check only.
6. **`probstat.systems.markov`** (new package `probstat.systems`, layer 6,
   stability label *experimental*) provides
   `MarkovReliabilityModel(chain, up_states, initial)`.
   - Time measures: point availability and unavailability, `R(t)` from the
     chain with the down states absorbing, the failure intensity `w(t)`,
     interval availability, expected downtime, and expected failures (the
     accumulated up-to-down flux).
   - Steady-state and mean-time measures: MTTF (`inf` when the initial law
     can reach a set of up states that never fails), steady-state
     availability, `w`, `MUT = A/w`, `MDT = (1-A)/w` (the system MTTR) and
     `MTBF = 1/w`.
   - Builders:
     - `single_repairable_unit`;
     - `repairable_k_out_of_n` (birth-death, `c` crews or `None` for one
       crew per unit, `failures_when_down`);
     - `parallel_redundant_pair` (shared first-come-first-served crew, or
       independent repair);
     - `standby_system` (cold, warm or hot, imperfect switching, optional
       repair, `active` units);
     - `load_sharing_pair`;
     - `from_states`.
7. **`probstat.systems.phased_markov`** provides:
   - `MissionPhase` and `MarkovPhasedMission`: one CTMC per phase; the law
     on the success states is carried into the next phase by label or by a
     sub-stochastic `entry_mapping`; with `continuous=True` failure states
     are absorbing, otherwise the requirement is checked only at the end of
     the phase. The mission returns the cumulative success, the conditional
     phase reliabilities and an error bound that sums the per-phase bounds.
   - `discrete_mission_outcome(markov_chain, start, success_states,
     failure_states)` on top of `MarkovChain.absorption()`.
   - `MissionStage` and `MissionStageModel` (success, retry, abort and
     failure per stage).
8. **Standby semantics.** These are design choices and are documented in
   the docstrings:
   - A failed switch-over takes the system down, into
     `"switch failure (j failed)"`.
   - Standby failures are revealed at once.
   - No unit fails while the system is down.
   - With repair, a completed repair returns the system to `"j-1 failed"`.
9. **Coordination with W3.** W3 creates the rest of `probstat.systems`
   (bdd, structure, importance, components, ccf, system, phased) in
   parallel. This branch adds only `markov.py`, `phased_markov.py` and an
   `__init__.py` that exports W4 names; the integrator merges the two
   `__init__.py` files. `"probstat.systems": 6` (LAYER), the GROUPS entry
   ("Reliability and risk") and `PACKAGE_STABILITY["systems"] =
   "experimental"` are added here as well; W3 may add the same lines, and
   the conflict is expected. The equation ids use the `EQ-SYS-MKV-*` and
   `EQ-SYS-MKPM-*` prefixes to avoid colliding with W3's ids.

## Evidence (`tests/test_ctmc.py`, `tests/test_markov_reliability.py`)

- **Transient law.** A 4-state chain at `t = 0.5, 3, 40` agrees with scipy
  `expm` to an L1 error below `1e-13` (series and squaring). Two-state closed
  forms agree to `1e-14` up to `t = 300`.
- **Stiff chain.** Rates from `1e-6` to `1e3` with `q t` up to `1e5`, against
  60-digit mpmath `expm`. The tiny components (`8.5e-16` to `4e-9`) agree to
  `< 1e-11` relative error by both methods. In scratch runs every L1 error
  was below the reported bound by a factor of `2e-3` to `1e-7`.
- **Large `q t`.** For the 1-out-of-2 system with `lambda = 1e-6`,
  `mu = 1e3`:
  - at `t = 100, 1e4, 1e6` (`q t` up to `1e9`, 21 squarings), the
    unreliability matches mpmath (`1.99997999399992e-13`,
    `1.9999997939800012e-11`, `1.99999999e-09`) to `< 1e-9` relative error,
    within the bound;
  - scipy `expm` gives `1.999999977e-9` at `t = 1e6`;
  - the library's `lu_solve` declares the MTTF system singular, while the
    subtraction-free solve gives `(3 lambda + mu)/(2 lambda^2)` to `1e-14`
    relative error.
- **Poisson windows.** For `lambda = 1e3` and `1e5`, the certified tails
  (`<= 2.5e-13`) dominate the scipy tails (`1.2e-14`).
- **Rewards.** The occupancy over `[0, 3]` matches Van Loan's block
  exponential to `1e-13`, and over `[0, 100]` on the stiff chain to `< 1e-9`
  (bound `1e-8`). A signed cost reward and `lambda int A` for one unit agree
  with their closed forms.
- **Steady state.** GTH matches the scipy eigenvector to `1e-15`, and a stiff
  birth-death chain matches the product form to `1e-14` relative error down
  to probabilities of `1e-27`.
- **Absorption.** CTMC absorption matches the embedded `MarkovChain`
  absorption and the gambler's ruin closed form.
- **Monte Carlo.** 3000 seeded paths agree with the transient law and the
  occupancy within 4.5 standard errors.
- **Reliability closed forms.**
  - single unit: `A(t)`, `R(t)`, interval availability, downtime, expected
    failures, MUT and MDT;
  - 1-out-of-2 MTTF `(3 lambda + mu)/(2 lambda^2)` from three builders;
  - pair availability: product formula and shared-crew birth-death;
  - `k`-out-of-`n`: binomial (steady state and `A(t)` with independent
    repair) and the product form with one crew;
  - non-repairable: binomial `R(t)` and `MTTF = sum 1/(j lambda)`;
  - standby: cold Erlang for `s = 1..3`, cold `exp(-lt)(1 + p l t)`, warm
    with one spare, and hot = active parallel;
  - load-sharing pair `R(t)` and MTTF.
- **Repairable warm standby** (2 spares, `p = 0.95`, one crew): `A`, `R`,
  downtime and expected failures match mpmath to `2e-15`; MTTF
  `22.076171875`; `A_inf`, `w`, MUT and MDT match mpmath `lu_solve`.
- **Phased mission.** Three phases match hand-chained scipy `expm` to
  `1e-14`. Identical phases equal the single-chain `R(T1 + T2)`. A mapping
  deficit and end-of-phase checking match closed forms.
- **Stage model.** Matches exact `Fraction` arithmetic. Gambler's ruin
  checks `discrete_mission_outcome`.

## Formulas derived or checked here rather than quoted

- **Warm standby with `s` spares** (`p = 1`, one active unit):
  `R(t) = e^(-lambda t) sum_(k <= s) (a)_k / k! (1 - e^(-lambda_s t))^k`,
  with `a = lambda / lambda_s`. It reduces to the textbook one-spare formula
  and agrees with `expm` to `1e-15` for `s = 1..3`. Its MTTF is the
  hypoexponential mean `sum_j 1/(lambda + j lambda_s)`.
- **One warm spare with switching**: `MTTF = 1/lambda + p/(lambda +
  lambda_s)`, obtained by integrating the closed-form `R(t)`.
- **Occupancy truncation bound**: derived here from
  `E[N 1{N >= j}] = lambda Pr(N >= j - 1)`.

## Consequences

- **Positive.** Repairable and standby systems, and phased missions, are
  solved analytically, with error bounds that hold on stiff models where
  double-precision `expm` loses digits.
- **Limitations.**
  - The rounding bound is a worst-case first-order estimate: it is honest
    but loose, often by about 3 to 7 orders of magnitude.
  - The series cost grows linearly in `q t` (guarded by `max_terms`), and
    squaring costs `O(n^3 s)` with dense rows, so large state spaces with
    large `q t` remain slow in pure Python.
  - Only exponential sojourns are supported (no semi-Markov models), and
    phase durations are deterministic.
  - Per-phase conditional reliabilities carry no separate bound.
  - The steady-state results (GTH) carry no a-posteriori bound: they are
    accurate to a few ulps in componentwise relative terms (O'Cinneide
    1993), but this is not reported numerically.
