# ADR-0076: Semi-Markov reliability and common-cause parameters as epistemic inputs

- Status: Accepted
- Date: 2026-11-11

## Context

The systems-engineering track listed two open items: semi-Markov sojourns
and CCF parameters as epistemic inputs.

- **Sojourns.** `MarkovReliabilityModel` (ADR-0061) and the dynamic fault
  trees (ADR-0070) assume exponential holding times. Repair times are
  usually lognormal and wear-out failures Weibull, so availability and
  reliability of repairable units were modelled with the wrong law or by
  Monte Carlo only. ADR-0061 named "no semi-Markov models" as a limitation.
- **CCF parameters.** Beta-factor, MGL and alpha-factor parameters are
  estimated from a handful of events (Mosleh et al. 1998) and are among the
  most uncertain inputs of a PRA. `SystemModel.as_probabilistic_model`
  varied component parameters only; CCF parameters stayed fixed, and the
  exact BDD moment engine (EQ-SYS-MOM-001, EQ-SYS-MOM-002) had no access to
  them.

## Decision

1. **`probstat.processes.semi_markov`** (layer 2) provides
   `SemiMarkovProcess(states, {(i, j): (p_ij, F_ij)})` with any sojourn law
   that has `cdf` and a finite `mean` and no atom at zero (`sf` is used for
   upper tails, `quantile` for simulation) [EQ-PROC-SMP-001]. A state
   without transitions is absorbing; self-transitions are allowed.
   `from_ctmc` converts a `ContinuousTimeMarkovChain` (exponential
   sojourns).
2. **Markov renewal equation** [EQ-PROC-SMP-002]: `transition_matrix`,
   `transient`, `expected_reward_curve`, `probability_curve`, `probability`
   and `first_passage_curve` solve `u(t) = (1 - H(t)) r + int dQ(s) u(t - s)`
   by the implicit trapezoidal scheme with exact kernel increments
   (Corradi, Janssen and Manca 2004; Linz 1985, Chapters 7-8), once on `n`
   and once on `2n` steps, and report the Richardson value
   `(4 u_2n - u_n)/3` with the estimate `|u_2n - u_n| / 3` of the error of
   the finer solution [EQ-PROC-SMP-003]. Results are
   `SemiMarkovEstimate`, `SemiMarkovTransient`,
   `SemiMarkovTransitionMatrix` and `RenewalCurve`; the error is called an
   *estimate* throughout and never presented as a bound. The convolution
   is evaluated with `math.fsum(map(mul, ...))` per transition, so the cost
   is `O(n^2)` per transition and reward column.
3. **Exact parts.** `absorption` returns `N = (I - P_TT)^-1`, `B = N P_TA`
   and `m = N tau` [EQ-PROC-SMP-004]; `mean_first_passage_time` returns
   `inf` when a closed class outside the targets is reachable;
   `steady_state` returns `nu_j tau_j / sum_k nu_k tau_k` [EQ-PROC-SMP-005].
   The jump chain is held as a CTMC with rates `p_ij` (`i != j`), whose
   generator restricted to the transient states is `-(I - P_TT)`, so the
   existing subtraction-free elimination and GTH reduction are reused
   unchanged. `simulate` and `monte_carlo_probabilities` are seeded
   cross-checks [EQ-PROC-SMP-006].
4. **`probstat.systems.semi_markov`** provides
   `SemiMarkovReliabilityModel(process, up_states, initial)` with
   availability, unavailability (each solved directly), reliability (down
   states absorbing), their curves, the exact MTTF and the steady-state
   availability and unavailability, and `alternating_renewal_unit(up_time,
   down_time)` [EQ-SYS-SMP-001]. **`systems.SemiMarkovComponent`** (in
   `adapters`) exposes such a model through the `ComponentModel` protocol,
   caches per time and keeps the largest error estimate in
   `last_error_estimate`, so a semi-Markov block can be used in
   `SystemModel` and `MissionAssessment` exactly as `MarkovComponent`.
5. **`probstat.systems.epistemic_ccf`** provides
   `ccf_parameter_uncertainty(system, t, {(group, name): law})` and
   `ccf_interval_bounds(system, t, {(group, name): interval},
   distributions=...)`. Names are `"beta"`; the MGL factors `"beta"`,
   `"gamma"`, `"delta"`, ...; `"alpha_k"` (`k >= 2`, `alpha_1` absorbs the
   remainder); and `"total"`, the shared `Q_t` of a group at `t`. Laws are
   probstat laws or `RandomInput`s supported on `[0, 1]` (a lognormal must
   be truncated). `SystemModel` gains the read-only properties
   `ccf_groups` and `ccf_basis`.
   - **Exact path** (probability split with beta-factor or MGL parameters
     and any `"total"`): the coefficients are monomials in `theta` and
     `1 - theta`; the Shannon recursion on the BDD of the fully expanded
     structure with polynomial event probabilities in the mixed-moment basis
     of EQ-SYS-MOM-002 gives `R(theta)` and `1 - R(theta)` with
     non-negative coefficients, and the moments are sums of products of the
     parameters' mixed moments (closed form for `Beta`, adaptive quadrature
     otherwise) [EQ-SYS-ECCF-001]. First-order shares
     `Var(E[R | theta_S]) / Var[R]` come from the cross moment of two
     copies that share `theta_S` [EQ-SYS-ECCF-002]; `ccf_contribution` is
     the share of all CCF parameters jointly, `ccf_variance` its absolute
     value.
   - **Monte Carlo path** (hazard split, alpha factors, or on request):
     seeded inversion sampling, the sample mean with its standard error,
     and the Saltelli (2010) first-order estimator with `f(B)` centred by
     the sample mean [EQ-SYS-ECCF-003].
   - **Intervals**: each parameter is scanned on a grid through the box
     midpoint and classified as increasing, decreasing, constant or
     non-monotone; the unreliability is evaluated at every corner (common
     random numbers for distributed parameters) and the probability box is
     the envelope of the corner distributions, a
     `propagation.ProbabilityBox` [EQ-SYS-ECCF-004].
   The structure is expanded once with events of every order (stand-in
   coefficients), so parameter values that make a coefficient zero need no
   recompilation.
6. **Layering and naming.** `systems` (layer 6) imports `propagation`
   (layer 4) for `ProbabilityBox` and `modeling` for `IntervalInput` and
   `RandomInput`; `processes.semi_markov` imports `distributions` (layer 1)
   for `Exponential`. Equation ids use the prefixes `EQ-PROC-SMP-*`,
   `EQ-SYS-SMP-*` and `EQ-SYS-ECCF-*`.

## Consequences

- Positive: repairable units with Weibull, lognormal, gamma or any other
  sojourn law are solved deterministically, with exact limiting and mean
  measures and a transient solution of fourth-order observed accuracy for
  smooth kernels; they compose with the BDD system model. CCF-parameter
  uncertainty, previously ignored, is propagated exactly where the model
  is polynomial in the parameters, with variance shares that say how much
  of the epistemic spread of `R` is due to CCF data.
- Limitations:
  - The Richardson error estimate is not a rigorous bound; kernels whose
    density is singular or discontinuous (for example Weibull with shape
    below one at `t = 0`) converge more slowly and the estimate may be
    optimistic there.
  - The renewal solver costs `O(n^2)` per transition and column on a
    uniform grid; very long horizons relative to the sojourn scales need
    many steps, and `SystemModel.mttf` (quadrature over `R(t)`) becomes slow
    for semi-Markov blocks; `SemiMarkovReliabilityModel.mean_time_to_failure`
    is exact instead.
  - Sojourn laws must have no atom at zero; deterministic sojourns are not
    supported by the renewal solver (only by simulation through a law
    whose `cdf` jumps after zero).
  - Exact CCF moments need the probability split and beta-factor or MGL
    parameters; the polynomial size grows with the product of the
    per-parameter degrees. Alpha-factor parameters use the remainder
    convention for `alpha_1` and are sampled. Parameters are independent of
    each other and of component parameters.
  - Interval bounds rely on a monotonicity scan along lines through the box
    midpoint; non-monotone responses give inner approximations.

## Verification (`tests/test_semi_markov_and_epistemic_ccf.py`)

- **Exponential sojourns versus the CTMC solver.** On the 4-state chain of
  `tests/test_ctmc.py` with 256 steps the L1 error of the state law is
  `1.1e-13` at `t = 0.5` and `3.1e-10` at `t = 3` (estimates `3.2e-7`,
  `6.6e-6`); repairable 2-out-of-3 (`lambda = 0.4`, `mu = 1.5`): `A(t)` and
  `R(t)` differ by at most `4e-11` at `t = 0.7, 2.5`, the MTTF
  `3.6458333333333` agrees to rounding and the steady-state
  availability exactly; the limiting law matches GTH to `1e-14`; a
  semi-Markov block in series with a fixed component equals the
  `MarkovComponent` system to `1e-8`.
- **Alternating renewal** (Weibull(1.5, 10) up, LogNormal(0, 0.5) down):
  `A_inf = E[U]/(E[U] + E[D]) = 0.8884762416710886` to `1e-15`, MTTF
  `E[U] = 9.0274529295` and `R(t) = P(U > t)` to `1e-12`; `A(3), A(9),
  A(18) = 0.92698, 0.89229, 0.88822` against 6000 seeded paths
  `0.92067, 0.88967, 0.88683` (standard errors `0.0035`-`0.0041`, all
  within 4.5); largest error estimate `1.2e-5` on 180 steps. The block runs
  inside `MissionAssessment` and gives `P(U > 5) exp(-5e-3)` to `1e-10`.
- **Absorption.** On a 3-state example with Weibull, lognormal and gamma
  sojourns the mean times `3.7488886338` and `3.4591591691` equal
  `(I - P_TT)^-1 tau` to `1e-13` relative and agree with 3000 simulated
  absorption times within 4.5 standard errors.
- **Epistemic beta factor** on a 2-out-of-3 group (`Q_t = 0.01`,
  `beta ~ Beta(2, 18)`): exact `E[R] = 0.9987574155957086` and
  `Var[R] = 3.8554904e-7` equal adaptive quadrature over the beta law to
  `1e-13` and `1e-9` relative; a truncated lognormal law (quadrature
  moments) agrees with direct integration to `1e-11`; Monte Carlo agrees
  within 4.5 standard errors. With `Q_t ~ Beta(2, 198)` as well, the exact
  shares are `0.1829` (beta) and `0.7406` (total); the centred Saltelli
  estimates with 2000 samples are within `0.05` and `0.15`. MGL exact and
  Monte Carlo means agree; hazard-split and alpha-factor parameters take
  the Monte Carlo path.
- **Intervals.** For `beta in [0.02, 0.2]` the unreliability is detected as
  increasing, the bounds `[4.862e-4, 2.191e-3]` equal the endpoint values
  and contain 200 random draws; a series system is detected as decreasing.
  With `Q_t` distributed the two corner distributions span a probability
  box with `lower <= upper` everywhere.
- **Guards.** Unnormalised jump probabilities, unknown states, laws with
  mass at zero or without `cdf`, invalid step counts, initial mass on down
  states, unknown CCF parameters, laws outside `[0, 1]`, `method="exact"`
  under the hazard split, systems without CCF groups, reversed intervals
  and keys given both as interval and as law raise.
