# ADR-0060: Exact system reliability engine `probstat.systems`

- Status: Accepted (proposed during the 2026-10-16 session, workstream W3)
- Date: 2026-10-16

## Context

`risk.FaultTree` quantifies static fault trees exactly. It uses Shannon
decomposition on the repeated events, so the cost is `2^r` with a guard of
16. Its minimal cut sets come from MOCUS. `reliability.systems` brackets and
samples series and parallel systems of *limit states*. Neither gives a
systems engineer what they need for a mission system: the exact
probability that a system described by its components works, with large
numbers of repeated events, importance measures, common-cause failures,
component time models (MTTF, availability, periodic tests), phased
missions, and the epistemic uncertainty of the answer.

## Decision

1. **New package `probstat.systems`, layer 6, experimental.** It may import
   lower layers and `risk.fault_tree` (for `from_fault_tree`); `risk` never
   imports it. It is registered in `tests/test_architecture.py` `LAYER`,
   `foundations.stability.PACKAGE_STABILITY` and the API-reference
   `GROUPS` ("Reliability and risk").
2. **BDD engine (`bdd.py`).** A reduced ordered BDD after Bryant (1986), with
   a unique table, a computed cache and an iterative memoized post-order
   evaluator, so there is no recursion limit (a 3000-variable chain is
   tested). It has no garbage collection; `max_nodes` (5e6 by default) raises
   instead of exhausting memory. It provides:
   - probabilities `P(f=1)` and `P(f=0)` as two separate convex-combination
     recursions, with optional exact complements `q`, so a tiny
     unreliability keeps its relative accuracy;
   - all conditional probabilities `P(f | x_v = 0/1)` and Birnbaum measures
     from one top-down reach pass and one bottom-up pass. Edges that skip a
     level are accumulated with exact Shewchuk partials, because difference
     arrays cancel catastrophically;
   - Rauzy's (1993) `minsol` and `without` on zero-suppressed set families
     in the same unique table (`_family_mk` reduces only `high == FALSE`),
     plus the dual, family count, onset and family-to-function operations;
   - exact epistemic moments (decision 5).
3. **Structures (`structure.py`).** Success-space expressions `Component`,
   `Series`, `Parallel` and `KOutOfN` (frozen dataclasses), the builders
   `series`, `parallel`, `k_out_of_n` and `bridge`, and `from_fault_tree`
   by De Morgan duality (OR becomes series, AND becomes parallel, k-of-n
   failure becomes (n-k+1)-of-n success). k-out-of-n is compiled with the
   threshold recursion `T(i,k) = (f_i and T(i+1,k-1)) or T(i+1,k)`. The
   default variable order is the depth-first order of first appearance;
   `order=` overrides it. `SystemStructure` provides exact reliability and
   unreliability, minimal path and cut sets (cut sets are `minsol` of the
   dual), their counts, Esary-Proschan and min-path/min-cut bounds, and
   `reliability_moments`.
4. **Importance (`importance.py`).** Birnbaum, criticality (Lambert),
   cut-set Fussell-Vesely with the union probability computed exactly on
   the BDD of the cut-set family that contains `i`, RAW, RRW, improvement
   potential, and structural importance. The "fractional" FV
   `(Q - Q(0_i))/Q` is *identical* to criticality, because `Q` is linear in
   `q_i`, so it is documented and not duplicated. Undefined ratios are
   `None` and not NaN (audit item 0.17).
5. **Epistemic moments: derivation, stated because no source was
   available.** For independent `p_v` with mean `m` and variance `s`, `R`
   is multilinear, so `E[R] = R(m)`. Expanding `F G` at the top variable
   with `E[p^2] = m^2 + s`, `E[p(1-p)] = m(1-m) - s` and
   `E[(1-p)^2] = (1-m)^2 + s`, and subtracting
   `E[F]E[G] = (m F1 + (1-m) F0)(m G1 + (1-m) G0)`, gives the covariance
   recursion
   `C = (m^2+s) C11 + (m(1-m)-s)(C10 + C01) + ((1-m)^2+s) C00 + s dF dG`.
   Its coefficients are non-negative, and for monotone `f` the covariances
   are non-negative too (Harris-FKG), so the sum has no cancellation. For
   **shared epistemic parameters** (Apostolakis and Kaplan 1981) the
   structure re-compiles with each group on consecutive levels. Inside a
   block, node values are polynomials `sum c_jk q^j (1-q)^k` with
   non-negative coefficients. A child in a deeper block depends on no
   variable of this block (by contiguity), so it enters as its expectation.
   The same scheme on node pairs gives `E[R^2]`. This is exact because the
   blocks are independent and the evaluation is linear in each child. The
   variance subtraction `E[g^2] - E[g]^2` uses whichever of `f` or `not f`
   has the smaller mean. Higher moments of a shared value are not
   determined by its mean and variance: the user supplies raw moments, or
   the beta law that matches the mean and variance is assumed and reported
   in `beta_assumed_groups`. This is an assumption, not
   `NotImplementedError`, and it is visible in the result.
6. **Component models (`components.py`).** These are the `ComponentModel`
   protocol (`reliability`, `unreliability`, `availability`,
   `unavailability`; each complement computed directly) and its
   implementations `FixedProbability(p)` (where `p` is the probability of
   *working*), `DemandComponent(q)`, `ExponentialComponent`,
   `WeibullComponent`, `LifeDistributionComponent` (any law with `sf`),
   `RepairableComponent` (two-state Markov availability), and
   `PeriodicallyTestedComponent`. The last one has a sawtooth point
   unavailability and `first_test` for staggering. Its exact mean
   `1 - (1-e^{-x})/x` is computed by series for `x < 0.5`. The
   approximation `x/2` is exposed separately and documented as overstating
   the mean by about `x^2/6`.
7. **CCF (`ccf.py`).** `BetaFactorGroup`, `MultipleGreekLetterGroup` and
   `AlphaFactorGroup` (non-staggered and staggered) are all linear in
   `Q_t`, so each exposes coefficients `c_k`. `expand_common_cause` replaces
   member `i` by `series(IND[i], CCF[S] for S containing i)` and drops
   events with `c_k = 0`. The BDD then evaluates the union exactly, so a
   member's marginal is `1 - prod (1-Q_k)^C(m-1,k-1)`, slightly below the
   rare-event normalisation `Q_t`.
8. **Time-dependent CCF split (a deviation found by testing).** Applying
   `Q_k = c_k Q_t(t)` over a long mission under-states the members' failure
   probabilities. The worked example showed it: adding a beta factor
   *raised* the MTTF from 7973 h to 8192 h, because `R(inf) > 0`. For
   mission reliability `SystemModel` therefore splits the cumulative hazard
   by default, `Q_k = 1 - exp(c_k log(1 - Q_t))`. This is the standard rate
   split `lambda_CCF = beta lambda` for exponential members, and the
   marginal is then exactly `Q_t`. Availability keeps the NUREG/CR-5485
   probability split. `ccf_basis="hazard" | "probability" | "auto"` selects
   the split, and the two agree to first order.
9. **`SystemModel` (`system.py`).** It provides `reliability/unreliability(t)`,
   `availability/unavailability(t)`, `event_probabilities`, `importance(t)`,
   `minimal_cut_sets()`, and `mttf()`. `mttf()` calls
   `numerics.integrate(R, 0, inf)` and returns `MTTFResult` with the error
   estimate, the convergence flag and the call count. It raises
   `ValueError` when `R(inf) > 0`. Finally there is
   `as_probabilistic_model(uncertain, times, measures, builders)`. By
   default it varies `p`, the demand probability, the failure rate, or the
   Weibull scale, and any other model needs a `builders=` entry.
   Components that share a random-input name share one value. The QoIs are
   named `"<measure>@<t>"`, and CCF parameters stay fixed.
10. **Phased missions (`phased.py`).** `Phase` and `PhasedMission`. The
    variables are `"<component>#<phase>"` = alive at the end of the phase,
    in component-major order. The proof that the end-of-phase state
    suffices (monotone structures, non-repairable components) is in the
    module docstring. Evaluation enumerates each component's phase of
    failure at the first node of its block and walks deterministically
    through the block, which is exact by independence and contiguity. This
    replaces the phase algebra of Zang, Sun and Trivedi (1999). A single
    model ages on cumulative mission time; a per-phase list applies each
    model to the age within its phase. `evaluate()` reports cumulative and
    conditional phase reliabilities and the probability of failing in each
    phase (`P(C_{j-1} and not phi_j)`, computed directly, not by
    subtraction).

## Evidence (`tests/test_systems_*.py`, 67 tests)

- A BDD of 40 random non-monotone formulas over 5 variables matches the
  truth tables for evaluate, negate, dual, restrict and ite. Probabilities
  and all four conditional probabilities plus Birnbaum match exact
  `Fraction` enumeration to 2e-15. Minimal solutions equal brute-force
  minimal true sets, onset families match, and `family_function(minsol(f))
  == f`. Bryant's ordering example gives 8 and 30 nodes for `n = 4`. The
  moments match a rational double enumeration over `4^5` state pairs to
  1e-14, and `grouped_moments` with singleton blocks matches as well.
- Structures: the bridge equals the textbook polynomial (0.97848 at
  `p = 0.9`). 40 random structures with repeated components match the
  enumerated reliability, cut sets and path sets. k-out-of-n matches the
  binomial and Poisson-binomial sums. `from_fault_tree` matches
  `FaultTree.top_event_probability` (Shannon and inclusion-exclusion) and
  MOCUS on shared-event trees. A unreliability of 1e-27 is accurate to
  1e-14 relative.
- Moments: bridge with Beta(8,2) components gives mean 0.91136 and variance
  0.0029948, equal to the rational oracle and inside the 4e6-draw numpy
  Monte Carlo (0.911314 +- 0.000027, variance 0.0030008). With components
  1 and 2 sharing the value: 0.898327 and 0.0055718 (MC 0.898318 +-
  0.000037, 0.0055785). Random structures with a shared group of three
  match the rational Beta-mixed-moment oracle to 1e-12.
- Scaling: consecutive-2-out-of-60:F equals the recursion
  `R(n) = p R(n-1) + q p R(n-2)` to 1e-13 relative. Its BDD has 118 nodes
  and 59 minimal cut sets, and it runs in about 20 ms. A 40-of-60 vote
  matches the binomial sum, and 20 bridges in series (100 components)
  match `0.97848^20`.
- Importance: every measure matches enumeration on the bridge and 25
  random structures. Structural importance of the bridge is 3/8 and 1/8.
  Birnbaum, FV, RAW and RRW equal `risk.FaultTree.importance`.
- Components: `A(t)` of the repairable component equals
  `scipy.linalg.expm` to 2e-16 and satisfies the Markov ODE. The
  periodic-test average equals `mpmath` (50 digits) to 1e-14 relative for
  `lambda tau` from 1e-6 to 2, and equals quadrature of the sawtooth.
- CCF: the normalisation holds to 1e-14 for `m = 2..6` with random
  parameters. The explicit three-member formulas hold, and staggered alpha
  equals MGL under `alpha_k = (prod rho)(1 - rho_{k+1})`. Expanded
  2-out-of-3 and 1-out-of-4 systems equal brute force over all CCF events
  to 1e-17. A beta-factor pair has `MTTF = 2/lambda - 1/((2-beta) lambda)`
  under the hazard split.
- MTTF matches the closed forms (series, parallel, 2-out-of-3 `5/(6
  lambda)`, Weibull `eta Gamma(1+1/beta)`) to 1e-9, and the cooling-loop
  closed form to 1e-10. Phased missions match enumeration over every
  combination of phases of failure, two-phase closed forms, and the
  equivalence between a single exponential model and per-phase models.
  The exact moments agree with `propagate` through
  `as_probabilistic_model` within 5 standard errors.

## Consequences

- Positive: exact answers with honest accuracy, a single BDD shared by
  reliability, cut sets, bounds, every importance measure and the moments,
  and a bridge into `ProbabilisticModel`/`UQStudy` for full epistemic
  propagation and sensitivity analysis.
- Limitations: (a) independence except through explicit CCF events, with
  no load sharing, standby switching, repair queues or dynamic gates;
  (b) the structures, cut sets and bounds require monotone logic (the `BDD`
  class itself handles any logic); (c) BDD size can be exponential for
  some structures, in which case the node guard raises; (d) CCF groups must
  be symmetric, and CCF parameters are not propagated as uncertain by
  `as_probabilistic_model` (users can do it with `builders=` only for
  component parameters); (e) shared-group moments beyond the variance rely
  on supplied raw moments or an assumed (reported) beta law; (f) phased
  missions have no repair, and per-phase models reset the age within each
  phase by construction; (g) no variable reordering (sifting) and no
  garbage collection.
