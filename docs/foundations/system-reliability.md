# Exact System Reliability

A systems engineer usually knows a mission system as a set of subsystems and
components and the logic that connects them. Examples are "two pumps, either
one is enough", "two of three sensors must agree", and "the valve and the
controller are both needed". `probstat.systems` turns that description into
the **exact** probability that the system works, together with the usual
diagnostics: minimal cut sets, bounds, importance measures, common-cause
failures, MTTF, availability, phased missions, and the epistemic uncertainty
of the result. Nothing in this package samples. Each number is a closed-form
evaluation of the structure function on a binary decision diagram (BDD), up
to floating-point rounding. The one exception is the MTTF, which is a
quadrature with its own error estimate.

The numbers in the worked example were produced by the library and are
asserted by `tests/test_systems_model_phased.py`.

## 1. Describing a system

Components are named, and `x_i = 1` means that component `i` works. A
structure expression combines them:

| Builder | Works when | Equation |
|---|---|---|
| `series(a, b, ...)` | every item works | [EQ-SYS-STRUCT-001] |
| `parallel(a, b, ...)` | at least one item works | [EQ-SYS-STRUCT-001] |
| `k_out_of_n(k, a, b, ...)` | at least `k` items work (items may differ and may be sub-structures) | [EQ-SYS-STRUCT-002] |
| `bridge(a, b, c, d, e)` | a path `{a,d}`, `{b,e}`, `{a,c,e}` or `{b,c,d}` works | [EQ-SYS-STRUCT-003] |
| `from_fault_tree(tree)` | the top event of a `risk.FaultTree` does not occur | [EQ-SYS-STRUCT-004] |

A name may appear in several branches. Every occurrence is the same physical
component, called a *repeated* or *shared* event. Repeated events are the
reason why series-parallel reduction formulas are not enough in practice. The
BDD handles them exactly.

```python
from probstat.systems import SystemStructure, series, parallel, k_out_of_n

structure = SystemStructure(
    series(parallel("P1", "P2"), "V", k_out_of_n(2, "S1", "S2", "S3"), "C")
)
structure.reliability({"P1": 0.9, "P2": 0.9, "V": 0.98, "S1": 0.95,
                       "S2": 0.95, "S3": 0.95, "C": 0.99})
```

Every builder is monotone: repairing a component never makes the system
fail. The structures are therefore *semi-coherent*. They are coherent when
every component is relevant, and `SystemStructure.irrelevant_components`
lists the components that are not. For example, `B` is irrelevant in
`parallel(A, series(A, B))`.

## 2. Binary decision diagrams

A reduced ordered BDD [REF-BRYANT-1986] stores a Boolean function as a graph
of Shannon expansions. Each internal node `(v, low, high)` means
`f = (not x_v and f_low) or (x_v and f_high)`. The variable order is fixed,
and no node has `low == high`. A unique table makes the representation
canonical: two equivalent formulas get the same node id.

Under independence, the probability of a node is a convex combination of the
probabilities of its two children:

`P(f_n = 1) = (1 - p_v) P(f_low = 1) + p_v P(f_high = 1)` [EQ-SYS-BDD-001].

The library evaluates `P(f = 1)` and `P(f = 0)` with two separate recursions.
Each is a sum of non-negative terms, so a system unreliability of `1e-27`
keeps full relative accuracy when you pass the component unreliabilities
(`unreliabilities=`) and not only `1 - p`. The tests check a
triple-redundant system with `q = 1e-9`, whose unreliability is `1e-27` to
`1e-14` relative error.

The size of a BDD depends on the variable order. The default order is the
depth-first, left-most order of first appearance in the expression, which
places the components of one module next to each other. Bryant's classic
example `OR_i (x_i and y_i)` shows how much the order matters: it has `2n`
nodes when the variables are interleaved and `2^(n+1) - 2` nodes when they
are separated (tested for `n = 4`). Pass `order=` to `SystemStructure` to
override the default. The engine is iterative, so a 3000-variable chain does
not hit Python's recursion limit. It never garbage-collects nodes, and
`max_nodes` bounds its memory.

**All conditional probabilities in one pass.** A top-down pass computes the
probability `r(n)` that the random path through the diagram reaches node
`n`. Every path crosses the level of variable `v` exactly once, either at a
node labelled `v` or along an edge that skips the level. This gives
`P(f | x_v = 0)` and `P(f | x_v = 1)` for every variable from one top-down
and one bottom-up pass [EQ-SYS-BDD-002]. The sums over skipping edges are
accumulated exactly with Shewchuk's partials. Birnbaum's importance is
`sum_{label(n)=v} r(n) [P(f_high) - P(f_low)]`.

## 3. Minimal cut and path sets, and bounds

For a monotone function, Rauzy's algorithm [REF-RAUZY-1993] builds the
minimal solutions directly on the diagram as a set-family BDD:

`minsol(ite(x, f1, f0)) = x * without(minsol(f1), minsol(f0)) + minsol(f0)`
[EQ-SYS-MINSOL-001].

The minimal path sets are `minsol(phi)`. The minimal cut sets are the
minimal solutions of the dual `phi^D(y) = not phi(not y)`, the failure
function in the variables "component failed". The family diagram counts the
sets without listing them (`count_minimal_cut_sets`). It also gives the
exact probability of any union of cut sets, which the Fussell-Vesely measure
needs. For risk-style fault trees the result equals MOCUS
(`risk.FaultTree.minimal_cut_sets`); the tests check this on trees with
shared events.

`SystemStructure.bounds` returns the exact value together with two pairs of
classical bounds:

- Esary-Proschan [EQ-SYS-BOUND-001]:
  `prod_cuts [1 - prod q_i] <= R <= 1 - prod_paths [1 - prod p_i]`. The
  lower bound is the fault-tree "min-cut upper bound" on the unreliability.
  The bounds hold for associated components, and independent components are
  associated [REF-ESARY-PROSCHAN-WALKUP-1967].
- Min-path and min-cut bounds [EQ-SYS-BOUND-002]:
  `max_paths prod p_i <= R <= min_cuts [1 - prod q_i]`.

The Esary-Proschan bounds are exact for pure series structures (lower bound)
and for pure parallel structures (upper bound).

## 4. Importance measures

`importance_measures(structure, p)` (or `SystemModel.importance(t)`) returns
every measure for every component. All of them are exact. In the table,
`Q(1_i)` is the unreliability given that component `i` has failed, and
`Q(0_i)` is the unreliability given that `i` is perfect.

| Measure | Definition | Reading |
|---|---|---|
| Birnbaum | `Q(1_i) - Q(0_i) = dR/dp_i` | how much `R` moves per unit of `p_i` |
| Criticality (Lambert) | `I_B q_i / Q` | `P(i failed and critical given the system failed)` |
| Fussell-Vesely | `P(some minimal cut set containing i is failed) / Q` | share of the risk in which `i` takes part |
| RAW | `Q(1_i) / Q` | factor by which the risk grows if `i` fails |
| RRW | `Q / Q(0_i)` | factor by which the risk falls if `i` is perfect |
| Improvement potential | `R(1_i) - R = I_B q_i` | reliability gain from a perfect `i` |
| Structural | `I_B` with every `p = 1/2` | fraction of the other components' states in which `i` is critical |

Two facts that PRA codes often blur:

1. Some codes report as "Fussell-Vesely" the *fractional reduction*
   `(Q - Q(0_i)) / Q = 1 - 1/RRW`. Because `Q` is linear in `q_i`, this
   quantity is identical to the criticality importance, so the library
   reports it once, as `criticality`.
2. The cut-set definition of Fussell-Vesely is at least as large as the
   criticality. Its union probability is evaluated exactly on a BDD built
   from the cut-set family that contains `i`, not with the rare-event sum.

For the bridge, the structural importances are `3/8` for the four outer
components and `1/8` for the bridging component (tested).

## 5. Epistemic uncertainty in component reliabilities

Component reliabilities are usually estimated from data, so `p_i` is itself
uncertain. If the `p_i` are independent with means `m_i` and variances
`s_i`, the reliability polynomial `R(p)` is multilinear. Its mean is
therefore `E[R] = R(m)`. Its variance comes from a pairwise recursion on the
BDD [EQ-SYS-MOM-001]:

`C(F, G) = (m^2 + s) C(F1, G1) + (m(1-m) - s) [C(F1, G0) + C(F0, G1)]
+ ((1-m)^2 + s) C(F0, G0) + s dF dG`, with `Var R = C(phi, phi)`.

Every coefficient is non-negative. For monotone structures every covariance
is non-negative too (Harris-FKG), so the sum has no cancellation. The
recursion follows from `E[p^2] = m^2 + s`, `E[p(1-p)] = m(1-m) - s` and
`E[(1-p)^2] = (1-m)^2 + s`; it was derived for this library and is not
taken from a published source.

**State-of-knowledge correlation.** Identical components whose reliability
is estimated from the *same* data share one uncertain value, not
independent copies of it [REF-APOSTOLAKIS-KAPLAN-1981]. Treating them as
independent understates the mean unreliability of redundant designs. For
two redundant units, `E[Q]` is `E[q]^2` if the units are independent but
`E[q]^2 + Var q` if they share `q`. Declare such groups with
`reliability_moments(means, variances, shared={"P1": "pump", "P2": "pump"})`.
`R` is then a polynomial in each shared value, of degree up to the group
size. The library reorders the BDD so that each group occupies consecutive
levels, carries the values of nodes inside a group as polynomials
`sum c_jk q^j (1 - q)^k` with non-negative coefficients, and integrates each
group out with the mixed moments `E[q^j (1 - q)^k]` [EQ-SYS-MOM-002].

The variance needs raw moments of the shared value up to twice the group
size, and a mean and a variance alone do not determine them. Supply them
with `shared_moments=`, or the library assumes the beta law that matches
the mean and variance and reports the group in `beta_assumed_groups`.

For a full propagation, with any input law, sensitivity indices, and
reports, use `SystemModel.as_probabilistic_model` (section 8). The exact
moments then serve as a check on the propagation.

## 6. Component time models

Every model implements `reliability(t)`, `unreliability(t)`,
`availability(t)` and `unavailability(t)`. Each complement is computed
directly (with `expm1`), never as `1 - x`.

| Model | Mission reliability | Point availability |
|---|---|---|
| `FixedProbability(p)` | `p` (probability of *working*) | `p` |
| `DemandComponent(q)` | `1 - q` | `1 - q` |
| `ExponentialComponent(lambda)` | `exp(-lambda t)` | = reliability |
| `WeibullComponent(beta, eta)` | `exp(-(t/eta)^beta)` | = reliability |
| `LifeDistributionComponent(law)` | `law.sf(t)` | = reliability |
| `RepairableComponent(lambda, mu)` | `exp(-lambda t)` | `mu/(lambda+mu) + lambda/(lambda+mu) e^{-(lambda+mu) t}` [EQ-SYS-COMP-002] |
| `PeriodicallyTestedComponent(lambda, tau, first_test)` | `exp(-lambda t)` | `exp(-lambda a(t))`, where `a(t)` is the time since the last test [EQ-SYS-COMP-003] |

For a periodically tested standby component, the exact mean unavailability
over a test interval is `1 - (1 - e^{-lambda tau}) / (lambda tau)`. It is
evaluated by its series for `lambda tau < 0.5`, and the tests check it
against `mpmath` to `1e-14` relative error. The textbook value
`lambda tau / 2` overstates it by about `(lambda tau)^2 / 6`; for
`lambda tau = 0.3` the exact value is `0.13606` and the approximation gives
`0.15`.

## 7. Common-cause failures

A CCF group is a set of `m` similar components. The parametric models of
NUREG/CR-4780 [REF-MOSLEH-ETAL-1988] and NUREG/CR-5485 [REF-MOSLEH-ETAL-1998]
split each member's total failure probability `Q_t` into `Q_k`. `Q_k` is the
probability of the basic event that fails one specific set of `k` members:

| Model | `Q_k` |
|---|---|
| `BetaFactorGroup(members, beta)` | `Q_1 = (1-beta) Q_t`, `Q_m = beta Q_t` [EQ-SYS-CCF-001] |
| `MultipleGreekLetterGroup(members, (beta, gamma, ...))` | `(prod_{i<=k} rho_i)(1 - rho_{k+1}) Q_t / C(m-1, k-1)` [EQ-SYS-CCF-002] |
| `AlphaFactorGroup(members, alphas, "non-staggered")` | `k/C(m-1,k-1) * alpha_k/alpha_t * Q_t`, `alpha_t = sum k alpha_k` [EQ-SYS-CCF-003] |
| `AlphaFactorGroup(..., "staggered")` | `alpha_k Q_t / C(m-1, k-1)` |

Every model satisfies `sum_k C(m-1, k-1) Q_k = Q_t` [EQ-SYS-CCF-004], and
the tests check this for `m = 2..6` with random parameters. The expansion
replaces member `i` by the series of its basic events: `IND[i]` and every
`CCF[S]` with `i` in `S` [EQ-SYS-CCF-005]. All basic events are
independent, so the expanded structure goes into the BDD without
approximation.

**Time dependence** [EQ-SYS-CCF-006]. Applying `Q_k = c_k Q_t(t)` (the
*probability* split) is the NUREG convention for demand probabilities and
unavailabilities. Over a long mission it under-states each member's failure
probability, because the exact union of the member's events is below the
rare-event sum. It even gives `R(t)` a spurious positive limit. For mission
reliability, `SystemModel` therefore splits the member's cumulative hazard
instead: `Q_k = 1 - exp(-c_k H(t))` with `H = -log(1 - Q_t)`. For an
exponential member this is the familiar rate split `lambda_CCF = beta
lambda`, and each member's marginal failure probability is exactly `Q_t(t)`.
A beta-factor redundant pair then has
`R(t) = 2 e^{-lambda t} - e^{-(2-beta) lambda t}` and
`MTTF = 2/lambda - 1/((2-beta) lambda)` (tested). Choose the split with
`ccf_basis="hazard" | "probability" | "auto"`. The two splits agree to first
order in `Q_t`.

## 8. The system model

`SystemModel(structure, components, ccf_groups)` provides the following:

- `reliability(t)` and `unreliability(t)`: mission reliability, with the
  components non-repairable during the mission;
- `availability(t)` and `unavailability(t)`: point availability, assuming
  independent components, each with its own repair or test policy;
- `mttf()`: `int_0^inf R(t) dt` by `numerics.integrate`, which tries a
  double-exponential rule first and falls back to Gauss-Kronrod. The result
  carries the error estimate, the evaluation count and a convergence flag. A
  system whose `R(t)` does not tend to zero raises `ValueError`. The tests
  check the closed forms: series `1/sum lambda`, parallel
  `1/l1 + 1/l2 - 1/(l1+l2)`, 2-out-of-3 `5/(6 lambda)`, and Weibull
  `eta Gamma(1 + 1/beta)`;
- `importance(t)` and `minimal_cut_sets()` over the basic events;
- `as_probabilistic_model(uncertain, times, measures=...)`: a
  `ProbabilisticModel` whose random inputs are uncertain component
  parameters. Examples are a Beta posterior on `p`, or a Gamma or LogNormal
  law on a failure rate. Its quantities of interest are exact system
  measures named `"reliability@<t>"`. Map several components to random
  inputs with the same name to make them share one value. `UQStudy` can then
  propagate the uncertainty, compute sensitivity indices, and report on the
  model.

## 9. Phased missions

A phased mission [REF-ZANG-SUN-TRIVEDI-1999] runs through consecutive phases
`j = 1..J`. Each phase has its own structure, which must work for the whole
phase, and the components are not repaired during the mission. Let
`x_{i,j}` mean "component `i` is alive at the end of phase `j`". Without
repair, a component state can only go from working to failed, so it is
non-increasing in time. Each phase structure is monotone, so the system
state is also non-increasing within a phase, and its minimum over the phase
is its value at the end of the phase. The mission therefore succeeds exactly
when `AND_j phi_j(x_{., j}) = 1`.

The variables `x_{i,1} >= ... >= x_{i,J}` of one component are dependent:
they are fixed by the component's *phase of failure* `K_i`. The library
builds the mission function in one BDD with a component-major variable
order. It then evaluates the probability bottom-up. At a node of component
`i`, it enumerates `K_i` and walks deterministically through `i`'s
consecutive variables to the first node of a later component. The result is
exact because each walk consumes the whole dependent block, and blocks of
different components are independent [EQ-SYS-PMS-001].

A component given one model ages on the cumulative mission time. A component
given a list of models (one per phase) applies each model to its age within
the phase, for example phase-dependent exponential rates. `evaluate()`
returns the mission reliability, the cumulative and conditional phase
reliabilities, and the probability that the mission fails in each phase.

The two-phase closed forms show why a product of separate phase
reliabilities is wrong. If phase 1 is `parallel(A, B)` and phase 2 is
`series(A, B)`, then `R = sA1 sA2 sB1 sB2`. If the order is reversed,
`R = sA1 sB1 [1 - (1 - sA2)(1 - sB2)]`.

## 10. Worked example

Consider a cooling loop with two pumps (either one is enough), a valve,
2-out-of-3 sensors and a controller, all with exponential lives (rates per
hour):

| Component | P1, P2 | V | S1, S2, S3 | C |
|---|---|---|---|---|
| `lambda` | `1e-4` | `2e-5` | `5e-5` | `1e-5` |

```python
from probstat.systems import (BetaFactorGroup, ExponentialComponent,
                              SystemModel, series, parallel, k_out_of_n)

structure = series(parallel("P1", "P2"), "V", k_out_of_n(2, "S1", "S2", "S3"), "C")
rates = {"P1": 1e-4, "P2": 1e-4, "V": 2e-5, "S1": 5e-5, "S2": 5e-5,
         "S3": 5e-5, "C": 1e-5}
components = {name: ExponentialComponent(rate) for name, rate in rates.items()}
plain = SystemModel(structure, components)
ccf = SystemModel(structure, components,
                  ccf_groups=[BetaFactorGroup(("P1", "P2"), 0.05)])
```

| Quantity at `t = 1000 h` | Without CCF | With beta = 0.05 on the pumps |
|---|---|---|
| Reliability | 0.955018267 | 0.951063146 |
| Unreliability | 0.044981733 | 0.048936854 |
| MTTF (h) | 7973.491040 | 7903.806339 |
| Minimal cut sets | `{C}, {V}, {P1,P2}`, three sensor pairs | `{CCF[P1,P2]}` becomes a single-event cut |

The MTTF without CCF has a closed form, which the tests use as a check. The
pump pair contributes `2e^{-lambda_p t} - e^{-2 lambda_p t}` and the sensors
`3e^{-2 lambda_s t} - 2e^{-3 lambda_s t}`. Integrating the product gives
`6/2.3e-4 - 4/2.8e-4 - 3/3.3e-4 + 2/3.8e-4 = 7973.491`.

The importance measures at 1000 h, with CCF:

| Event | Birnbaum | Criticality | FV | RAW | RRW |
|---|---|---|---|---|---|
| V | 0.970 | 0.393 | 0.405 | 20.4 | 1.65 |
| C | 0.961 | 0.195 | 0.203 | 20.4 | 1.24 |
| IND[P1] | 0.087 | 0.161 | 0.168 | 2.62 | 1.19 |
| CCF[P1,P2] | 0.956 | 0.097 | 0.102 | 20.4 | 1.11 |
| S1 | 0.089 | 0.089 | 0.095 | 2.73 | 1.10 |

The valve is the main contributor: it is a single-event cut with the largest
rate. The CCF event has the RAW of a single-event cut, even though its
criticality is small.

## 11. Verification

Every method is tested against an independent oracle (`tests/test_systems_*.py`):

- a direct recursive evaluator of the expression with brute-force
  enumeration of all `2^n` states in exact rational arithmetic. This checks
  reliability, cut and path sets, every importance measure, and the
  conditional probabilities of random monotone and non-monotone formulas;
- the textbook bridge polynomial, binomial and Poisson-binomial sums for
  k-out-of-n, and the consecutive-2-out-of-60:F recursion
  `R(n) = p R(n-1) + q p R(n-2)`. That system has 60 components and every
  inner component is repeated, and it is evaluated in milliseconds;
- `risk.FaultTree`, using both its Shannon method and inclusion-exclusion,
  and its importance measures, on trees with shared events;
- exact rational double enumeration for the epistemic moments, with and
  without shared groups, and a 4-million-draw numpy Monte Carlo run in the
  verification environment;
- `scipy.linalg.expm` for repairable availability, `mpmath` for the
  periodic-test average, and quadrature for the MTTF closed forms;
- for phased missions, enumeration over every combination of phases of
  failure, plus closed forms.

## 12. Assumptions and limitations

- **Independence.** Component states are independent except through
  explicit CCF events. Other dependencies are not modelled: shared support
  systems that are not in the structure, load sharing, standby switching, or
  repair crews. Model them in the structure, or use a Markov model.
- **Coherence.** The builders are monotone. Non-coherent logic (NOT gates,
  "fails if exactly one works") is supported by the `BDD` class, but minimal
  cut sets and the bounds require monotone functions.
- **Static structures and simple repair.** Availability assumes independent
  components with their own repair or test policies. There are no repair
  queues, no sequence-dependent failures (dynamic gates), and no repair
  during phased missions.
- **CCF data.** The parameters `beta`, `gamma` and `alpha_k` are estimated
  from very sparse event data and are among the most uncertain inputs of a
  PRA. Propagate them as uncertain (for example through
  `as_probabilistic_model` with a custom `builders=` entry), or at least
  report how sensitive the results are to them. Group members must be
  symmetric, that is, share `Q_t`.
- **Epistemic moments.** Shared groups need raw moments up to twice the group
  size. Without them a beta law is assumed and reported. For large groups,
  explicit raw moments enter through an alternating sum, which loses
  accuracy.
- **BDD size.** Exact methods are limited by the size of the BDD, and some
  structures are exponential under every variable order. The node guard
  then raises an error instead of returning an approximation.
