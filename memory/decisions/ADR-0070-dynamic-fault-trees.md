# ADR-0070: Dynamic fault trees and BDD variable reordering in `probstat.systems`

- Status: Accepted
- Date: 2026-11-11

## Context

`probstat.systems` quantifies static coherent structures exactly on reduced
ordered BDDs (ADR-0060) and repairable or standby blocks on continuous-time
Markov chains (ADR-0061). Two gaps remained on the systems-engineering
track:

- **Dynamic gates.** Order- and use-dependent failure logic (priority AND,
  spare gates with cold, warm or hot spares and shared spare pools,
  functional dependencies, sequence enforcement) could only be modelled by
  hand-built chains such as `standby_system`. Dugan, Bavuso and Boyd (1992)
  and the NASA Fault Tree Handbook (Stamatelatos et al. 2002, Chapter 8)
  define these gates and solve them by conversion to a Markov chain.
- **Variable order.** The BDD keeps the order of first appearance (or a
  user order). A poor order can make the diagram exponentially larger than
  necessary (Bryant 1986); ADR-0060 listed the absence of reordering as a
  limitation.

## Decision

1. **New module `systems.dynamic`.** `DynamicBasicEvent(name, rate,
   dormancy)` is a non-repairable event with an exponential lifetime; the
   dormancy factor `alpha` in `[0, 1]` scales the rate while the event is an
   unused spare. `DynamicGate(name, kind, inputs, k)` covers `and`, `or`,
   `vote`, `pand`, `spare`, `fdep` and `seq`, with the builders `and_gate`,
   `or_gate`, `vote_gate`, `pand_gate`, `spare_gate`, `fdep_gate` and
   `seq_gate`. `DynamicFaultTree(events, gates, top, max_states=10_000)`
   validates the tree and prunes elements that cannot influence the top
   event (a spare gate outside the top event is kept when it shares a unit
   with a kept element, an FDEP when a dependent is kept, a SEQ gate when a
   follower is kept).
2. **Semantics (stated in the module docstring and EQ-SYS-DFT-001).**
   PAND is inclusive: failures at the same instant, which only an FDEP can
   produce, count as in order, and an out-of-order failure blocks the gate
   for ever. A spare gate claims the first listed spare that has neither
   failed nor been claimed by another gate; simultaneous demands are served
   in gate declaration order; a claimed spare runs at its full rate. FDEP
   outputs are not gate inputs and cannot be the top event; dependents are
   basic events. SEQ followers are basic events with rate zero until their
   predecessor fails. Primaries and spares are basic events; an event is
   the primary of at most one gate and never both a primary and a spare.
3. **Exact quantification by state-space generation.** The state is the set
   of failed events, the unit in use per spare gate and the set of PAND
   gates with an order violation. The chain is generated breadth-first from
   the all-working state into the existing `ContinuousTimeMarkovChain`;
   after each failure the spare claims and FDEP cascades are resolved to a
   fixed point at the same instant, and every state with the top event is
   merged into the absorbing state `"failed"`. `unreliability_curve`,
   `unreliability` and `reliability` return `BoundedValue`s from
   uniformization (tolerance `1e-13` by default); `mttf` is the mean time to
   absorption (EQ-SYS-DFT-002) and is `inf` when a working state without
   outgoing transitions is reachable. `state_space()` returns the chain,
   its size, the transition count and those absorbing working states. More
   than `max_states` states raise `ValueError` naming the limit.
4. **Static trees on the BDD.** `is_static` trees compile to a
   `SystemStructure` through the fault-tree duality (EQ-SYS-STRUCT-004) and
   `method="auto"` uses that path; `"markov"` and `"bdd"` force a path.
5. **Monte Carlo cross-check.** `monte_carlo_unreliability(tree, times,
   samples, seed, confidence)` simulates failure times, not chain jumps:
   dormant spares draw `Exp(alpha lambda)` and, once claimed at time `s`,
   `s + Exp(lambda)`; gates are evaluated from failure times (order
   statistics; PAND by non-decreasing input times). Each estimate carries a
   Clopper-Pearson interval (EQ-SYS-DFT-003).
6. **Reordering in `systems.bdd`.** `BDD.sift(roots, max_growth=1.2,
   converge=False)` implements Rudell's (1993) sifting on a private,
   reference-counted copy of the reachable nodes with in-place adjacent
   level swaps (EQ-SYS-SIFT-001). Variables are sifted largest level first;
   each moves towards the nearer end, then the other, abandoning a
   direction above `max_growth` times its starting size, and returns to the
   best position seen, so the size never increases. `BDD.reorder(roots,
   order)` rebuilds the roots for a prescribed order. Both return
   `ReorderResult(manager, roots, order, nodes_before, nodes_after, swaps)`
   with a new manager; the source manager and its node ids are unchanged,
   because `SystemStructure` and other callers hold node ids.
   `BDD.shared_node_count` reports the shared size of several roots.

## Consequences

- Positive: PAND, spare (cold, warm, hot, shared pools), FDEP and SEQ logic
  is quantified exactly with an error bound, on the same chain engine as
  the rest of the package, with a semantically independent simulation for
  verification. Diagram sizes can be reduced without changing any result.
- Limitations: (a) the state space grows exponentially with the number of
  events (no modularization of independent static subtrees and no symmetry
  reduction), so `max_states` guards it and the CTMC constructor's
  reachability closure costs `O(n^3)`; (b) basic events are exponential and
  non-repairable; (c) spare units and SEQ followers must be basic events
  (no spare subtrees); (d) PAND is inclusive by convention; other
  conventions for simultaneous failures are not offered; (e) sifting
  returns a new manager rather than reordering in place, and
  `SystemStructure` does not re-sift automatically.

## Verification (`tests/test_dynamic_fault_trees.py`, `tests/test_bdd_reordering.py`)

- Closed forms (EQ-SYS-DFT-004) to `1e-13` at four times: PAND(A, B) for
  three rate pairs (and the sum of both input orders equals `F_A F_B`); cold
  pair = Erlang(2), three units = Erlang(3); hot pair = `(1 - e^{-lambda
  t})^2` and the static AND on the BDD; warm spares with unequal rates
  (hypoexponential) and their MTTF; a shared spare between two gates under
  an OR top = `Exp((2 + alpha) lambda) + Exp(2 lambda)` with its MTTF for
  `alpha = 0, 0.5, 1`; SEQ(A, B) = cold spare = hypoexponential; FDEP on
  the inputs of AND(A, B) = `T or (A and B)`, equal to the static tree on
  the BDD to `1e-14`. PAND MTTF is `inf`.
- Static trees: 25 random AND/OR/vote trees with shared events give the BDD
  unreliability on the Markov path to `1e-14` at three times; a static
  2-of-3 module under AND with a PAND factorises into the BDD value times
  the PAND closed form to `1e-14`; a 2-of-3 failure vote has MTTF
  `5/(6 lambda)`.
- Monte Carlo (20000 histories, 99.9% Clopper-Pearson) contains the exact
  value at three times for every elementary gate, for two gates sharing
  two warm spares under AND, and for a tree with every gate kind; FDEP with
  a gate trigger and simultaneous PAND inputs agrees as well.
- Guards: the state limit, irrelevant-element pruning, a sharing spare gate
  outside the top event, and fourteen invalid trees raise or behave as
  documented.
- Sifting: Bryant's interleaved `x1 x2 + ... + x(2n-1) x(2n)` in the order
  `x1, x3, ..., x2, x4, ...` drops from `2^(n+1) - 2` to `2n` nodes for
  `n = 2..8` with every pair adjacent; 60 random three-root diagrams never
  grow, keep every truth-table entry, and keep the exact rational
  probability (`fractions.Fraction` Shannon recursion on both managers);
  a poorly ordered series of parallel pairs keeps its reliability and
  unreliability.
