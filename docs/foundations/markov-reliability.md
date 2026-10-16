# Markov Reliability and Availability

Repairable systems, standby redundancy, and missions with phases can often be
solved exactly with a continuous-time Markov chain (CTMC). No Monte Carlo is
needed. `probstat.processes.ContinuousTimeMarkovChain` solves the chain.
`probstat.systems` turns a chain with *up* and *down* states into
reliability measures, and it builds the textbook models for you. Every
time-dependent number comes with a rigorous error bound. All numbers below
were produced by the library.

## 1. The model and its assumptions

A CTMC on states `S` has a generator `Q`. For `i != j`, `q_ij >= 0` is the
rate of jumping from `i` to `j`, and `q_ii = -sum_(j != i) q_ij`. The state
law solves `pi'(t) = pi(t) Q`, so `pi(t) = pi(0) exp(Q t)`
[EQ-PROC-CTMC-001].

```python
from probstat.processes import ContinuousTimeMarkovChain

chain = ContinuousTimeMarkovChain(
    ("up", "down"), {("up", "down"): 0.001, ("down", "up"): 0.1}
)
```

The modelling assumptions should be stated whenever you use these results:

- **Exponential sojourns.** Every failure and repair time is exponential with
  a constant rate. So the future depends only on the current state, not on
  how long the system has been in it. Wear-out (an increasing hazard) is
  excluded.
- **Interactions only through the state.** Units interact only through what
  the state records: a shared repair crew, a standby switch, load sharing.
  Otherwise failures and repairs are independent. Common-cause failures need
  extra transitions that fail several units at once.
- **Homogeneity.** Rates are constant in time. Missions whose stresses change
  are modelled as phases, each with its own constant rates (Section 5).

The Markov model is not appropriate in three cases. A deterministic or
lognormal repair time, a fixed inspection interval, or a component that ages
needs a **semi-Markov** model (general sojourn distributions in each state),
a non-homogeneous model, or phase-type expansions, for example an Erlang
stage chain approximating a deterministic delay. Each stage then adds states.
For large systems of independent components, the combinatorial methods
(fault trees and BDDs) are cheaper. Markov models are needed when dependence
such as repair queues, standby or sequencing matters.

## 2. Solving the chain

**Transient law by uniformization** [EQ-PROC-CTMC-002]. With
`q = max_i |q_ii|` and `P = I + Q/q`:

`pi(t) = sum_k Poisson(k; q t) pi(0) P^k`.

Every term is non-negative, so there is no cancellation. The Poisson series
is truncated to a window `[L, R]`. The window is certified by
Cramer-Chernoff tail bounds, evaluated in log space so that `exp(-q t)` never
underflows. The weights are generated from the mode outward, in the style of
Fox and Glynn [EQ-PROC-CTMC-003]. `transient` returns a `TransientSolution`
with these fields:

- `truncation_error_bound`: a rigorous absolute L1 bound, `<= tolerance`.
- `rounding_error_bound`: a first-order bound on the componentwise
  **relative** rounding error. Because everything is non-negative, a tiny
  probability such as an unreliability of `1e-13` keeps its relative accuracy.
- `error_bound`: the sum of the two. `total(states)` and `weighted(values)`
  give the bound for a sum over a set of states.

The series costs about `q t` sparse products. For a stiff chain (repair rate
`1e3`, horizon `1e6`, so `q t = 1e9`) this is too slow. `method="auto"` then
switches to **scaling and squaring of the uniformized matrix**: it builds
`exp(Q t/2^s)` row by row and squares it `s` times [EQ-PROC-CTMC-008]. The
truncation budget is divided by `2^s` up front, and the bound propagation is
reported.

**Rewards and occupancy** [EQ-PROC-CTMC-004]. Let
`L(t) = int_0^t pi(s) ds` be the expected time spent in each state. It is
computed exactly from the integrated series
`(1/q) sum_k Pr(N > k) pi(0) P^k`, with its own truncation bound. From
`L(t)` follow:

- the accumulated reward `L(t) . r` (cost or downtime over a mission);
- the expected number of `i -> j` transitions, `q_ij L_i(t)`.

**Steady state** [EQ-PROC-CTMC-005]. The Grassmann-Taksar-Heyman (GTH)
state reduction uses no subtraction. For a birth-death chain with rates
`1e-6` and `1e3`, the stationary probabilities (down to `1e-27`) agree with
the product formula to `1e-14` relative accuracy. With several closed
classes, `limiting_distribution(initial)` weights each class by the
probability of entering it.

**Absorption** [EQ-PROC-CTMC-006]. `absorption(initial)` returns
`N = (-Q_TT)^-1`, the absorption probabilities `B = N Q_TA` and the mean
times `N 1`. These are computed by a subtraction-free elimination of the GTH
type. For the 1-out-of-2 system with `lambda = 1e-6` and `mu = 1e3`, the MTTF
`(3 lambda + mu)/(2 lambda^2) = 5.000000015e14` is reproduced to `1e-14`
relative accuracy. The library's dense LU with partial pivoting
(`linalg.lu_solve`) declares the same 2x2 system singular to working
precision, because its last pivot `2 lambda^2 / mu` is formed by
cancellation.

`embedded_chain()` returns the jump chain as a discrete `MarkovChain`
[EQ-PROC-CTMC-007]. `simulate(initial, horizon, seed=...)` exists only to
cross-check the analytic answers by Monte Carlo.

## 3. Reliability measures

`MarkovReliabilityModel(chain, up_states, initial)` splits the states into
up states `U` and down states `D` [EQ-SYS-MKV-001] to [EQ-SYS-MKV-004]:

| Measure | Method | Definition |
|---|---|---|
| point availability `A(t)` | `availability(t)` | `sum_(U) pi_i(t)` |
| unavailability | `unavailability(t)` | `sum_(D) pi_i(t)` (no `1 - A` cancellation) |
| reliability `R(t)` | `reliability(t)` | up-mass of the chain with `D` absorbing |
| MTTF | `mean_time_to_failure()` | `alpha (-Q_UU)^-1 1` (`inf` if failure is not certain) |
| steady-state availability | `steady_state_availability()` | `sum_(U) pi_i` (GTH) |
| failure frequency | `failure_frequency(t)`, `steady_state_failure_frequency()` | `sum_(i in U, j in D) pi_i q_ij` |
| MUT, MDT (= system MTTR), MTBF | `mean_up_time()` etc. | `A/w`, `(1 - A)/w`, `1/w` |
| interval availability | `interval_availability(t)` | `(1/t) sum_(U) L_i(t)` |
| expected downtime | `expected_downtime(t)` | `sum_(D) L_i(t)` |
| expected number of failures | `expected_failures(t)` | `sum q_ij L_i(t)` over up-to-down pairs |

Builders return ready models:

- `single_repairable_unit(lambda, mu)` [EQ-SYS-MKV-005];
- `repairable_k_out_of_n(n, k, lambda, mu, repair_crews=c)`, a birth-death
  model with an optional stop of failures while the system is down
  [EQ-SYS-MKV-006];
- `parallel_redundant_pair(l1, l2, m1, m2, shared_repair=...)`
  [EQ-SYS-MKV-008];
- `standby_system(spares, lambda, standby_failure_rate, switch_success_probability, repair_rate)`,
  covering cold, warm and hot standby with imperfect switching
  [EQ-SYS-MKV-007];
- `load_sharing_pair(lambda, loaded_rate)`;
- `from_states(states, rates, up_states, initial)` for any other model.

## 4. Worked examples

**One repairable unit.** Take `lambda = 0.001/h` and `mu = 0.1/h`.

- `A(10) = 0.9937051384` (bound `3.7e-13`), and `A = mu/(lambda+mu) =
  0.9900990099`.
- Over 1000 h the interval availability is `0.99019704`.
- The expected downtime is `9.80296` h (bound `4.8e-10`).
- The expected number of failures is `0.990197`. It equals `lambda` times the
  expected uptime, as it must.

**Repairable 1-out-of-2.** Take `lambda = 1e-4/h`, `mu = 0.05/h` and one
crew.

- `MTTF = (3 lambda + mu)/(2 lambda^2) = 2,515,000` h.
- The steady-state unavailability is `7.968e-6`.
- `w = 3.984e-7` failures/h, so `MUT = 2,510,000` h, `MDT = 20` h, and
  `MTBF = 2,510,020` h.
- Over one year (8760 h): `R = 0.99653081`, unreliability `0.00346919`,
  expected failures `0.00348206`, and expected downtime `0.0695` h.

**Standby with imperfect switching.** One active unit with `lambda = 0.001/h`,
one spare, switch success `p = 0.98`, mission of 1000 h:

| standby rate | `R(1000)` | MTTF (h) |
|---|---|---|
| 0 (cold) | 0.728401 | 1980.0 |
| 0.0005 (warm) | 0.651588 | 1653.3 |
| 0.001 (hot) | 0.595773 | 1490.0 |

The cold value equals `exp(-lambda t)(1 + p lambda t)`. The warm values equal
`exp(-lambda t)[1 + p (lambda/lambda_s)(1 - exp(-lambda_s t))]`, and the MTTF
equals `1/lambda + p/(lambda + lambda_s)`. Adding repair at `mu = 0.05/h` to
the warm case raises `R(1000)` to `0.953959` and the MTTF to `20,992` h.

**A stiff model.** Take the 1-out-of-2 system with `lambda = 1e-6` and
`mu = 1e3` over `t = 1e6` h, so `q t = 1e9`. The `auto` method uses 21
squarings. The unreliability `1.999999990e-9` agrees with a 60-digit mpmath
`expm` to `1e-15` relative accuracy, and the reported bound is `4e-13`.
scipy's double-precision `expm` gives `1.999999977e-9` (relative error
`7e-9`). Its `1 - R` gives `1.5e-8`, because the subtraction cancels.

## 5. Phased missions

A `MarkovPhasedMission` is a list of `MissionPhase(name, chain, duration,
success_states, entry_mapping=None, continuous=True)` [EQ-SYS-MKPM-001].
Each phase has its own generator: its own stresses, repair policy and
configuration. The state law at the end of a phase, restricted to the phase's
success states, enters the next phase in one of two ways:

- by label (the default);
- through a sub-stochastic `entry_mapping`, whose row deficit is a failure at
  the phase change (for example a reconfiguration).

With `continuous=True` the failure states are absorbing, so any entry fails
the mission. With `continuous=False` the requirement is checked only at the
end of the phase, so the system may fail and be repaired within the phase.

Example: a two-unit system.

- Take-off, 2 h: 1-out-of-2 with repair.
- Cruise, 10 h: higher stress, no repair.
- Landing, 1 h: both units must be working at touchdown.

The mission succeeds with probability `0.6458747`. The conditional phase
reliabilities are `0.99971`, `0.83930` and `0.76977`. These agree with
hand-chained matrix exponentials to `1e-14`. When the same chain is split
into two phases, the result equals the single-chain reliability `R(T1 + T2)`.

**Discrete stages.** `MissionStageModel` builds a discrete-time chain from
stages with success, retry, abort and failure probabilities.
`discrete_mission_outcome` analyses any `MarkovChain` with absorbing outcomes
[EQ-SYS-MKPM-002]. Take launch (0.95 / 0.03 / 0.01), orbit (0.9 / 0.05 /
0.04) and return (0.98 / 0 / 0.01). The outcome probabilities are success
`0.909278`, abort `0.060825` and loss `0.029897`, with `2.98969` expected
stage attempts. These match exact rational arithmetic.

## 6. Accuracy and limits

- The error bounds are worst-case and first order. On the stiff examples the
  actual error is about 3 to 7 orders of magnitude below the bound. The truncation
  part is rigorous. The rounding part assumes IEEE double arithmetic and
  ignores second-order terms.
- The cost is `O(q t)` sparse products for the series. Squaring costs
  `O(n^3 log(q t))` and is preferred for few states with large `q t`.
  `max_terms` guards against runaway series.
- The state space must be finite and explicit. For many components the size
  grows as `2^n`. Use `k`-out-of-`n` lumping, or the combinatorial methods
  when the components are independent.
