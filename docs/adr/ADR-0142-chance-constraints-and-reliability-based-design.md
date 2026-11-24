# ADR-0142 — Chance constraints and reliability-based design optimisation

- Status: Accepted
- Date: 2026-11-24
- Extends: ADR-0132 (batch evaluation contract), the model-based reliability
  tools of `probstat.reliability`

## Context

Sizing a subsystem is an optimisation with probabilistic requirements:
minimise a cost over a design vector subject to "the margin is positive with
probability 0.99". The library had FORM, SORM, `inverse_form` and
`design_for_target_beta` (one design parameter against one limit state, solved
by Brent's method), but no optimiser over a design vector with reliability
constraints, no constrained optimiser at all (`probstat.optimization` has
BFGS, Nelder-Mead, least squares and Brent), and no tools for chance
constraints that are not tied to a design point.

## Decision

1. **A small constrained solver inside `reliability`.**
   `minimize_constrained` is an augmented Lagrangian for inequality
   constraints `c_j(x) <= 0` ([EQ-REL-AUGLAG-001], Nocedal and Wright
   Section 17.4) around the existing `minimize_bfgs` (finite-difference
   gradients, optional box). It is a local solver, tested on problems with
   known solutions and multipliers. It lives in `design_optimization.py`
   (layer 5) rather than `probstat.optimization` (layer 1) because the brief
   restricts the change to the reliability package; a later move downward is
   a pure relocation.
2. **Scenario guarantees as exact binomial tails.** `scenario_confidence`,
   `scenario_violation_bound` and `scenario_sample_size` implement the bound
   of Campi and Garatti (2008) ([EQ-REL-SCENARIO-001]), summed in log space.
   The explicit sufficient sample size `(2/eps)(d - 1 + ln(1/beta))` is derived
   here from the multiplicative Chernoff bound ([EQ-REL-SCENARIO-002]) and
   is reported next to the exact size. `scenario_posterior_violation_bound`
   implements the wait-and-judge level of Campi and Garatti (2018) from the
   number of support constraints ([EQ-REL-SCENARIO-003]); both papers were
   opened and the formulas were checked against R (`pbinom`, `uniroot`) and
   against the sample sizes of the 2008 paper's Table 2.2 (285, 581, 1171,
   2942 for `d = 10`, `beta = 1e-5`).
3. **A-posteriori certification is independent of the optimiser.**
   `certify_design` draws a fresh seeded validation sample from a child
   stream of the seed (`"validate"`, disjoint from the `"optimise"` child) and
   returns the exact one-sided Clopper-Pearson bound, reusing
   `inference.clopper_pearson_interval` at level `2 c - 1`. It does not
   require convexity.
4. **Sample-average chance constraints without smoothing.**
   `solve_chance_constrained` turns "at most `floor(gamma N)` of `N` samples
   violate" into the single order-statistic constraint
   `h_(q+1)(x) <= 0` with `h_s = max_i g_i(x, xi_s)`
   ([EQ-REL-SAA-001]); joint constraints use the maximum over the rows. The
   constraint is exact for the sample, piecewise smooth and generally
   non-convex, so the answer is local. With `gamma = 0` the program is the
   scenario program: every sample is then a separate smooth constraint (an
   order statistic would put the solution on a kink), and the a-priori bound,
   a leave-one-out count of support constraints and the a-posteriori bound
   are reported. They hold only when the user's program is convex with a
   unique solution, which the function cannot check.
5. **Closed forms as oracles and tools.** `normal_linear_chance_margin`
   (second-order cone, exact, [EQ-REL-CHANCE-NORMAL-001]) and
   `distribution_free_chance_margin` (Cantelli, attained by a two-point law,
   and the more conservative Chebyshev, [EQ-REL-CHANCE-FREE-001]).
6. **RBDO through limit-state factories.** A design enters a limit state
   through a factory `design -> ModelLimitState`, so deterministic variables
   (`model.fixed`) and means of random inputs are handled by the same
   interface and no new model class is needed. `rbdo_sora` is the decoupled
   method: cycle `k` solves the deterministic problem with constraints
   evaluated at `mu(d) - s`, then `inverse_form` gives the most-probable target
   point at the new design and `s = mu(d) - x*` ([EQ-REL-SORA-001]).
   Convergence needs a stationary design and every margin at the target point
   non-negative; the result carries the cycle history, the FORM index at the
   final design, evaluation counts, and a flag with its reason.
   `rbdo_double_loop` ([EQ-REL-RBDO-002]) is the nested reference: one FORM
   run per constraint call, slower by a factor of about eight on the
   three-constraint benchmark of the tests.
7. **Verification.** `verify_design_reliability` re-evaluates every limit state
   at the final design with SORM (Hohenbichler-Rackwitz) or design-point
   importance sampling and reports the index difference from FORM.

## Sources

Opened and read: Campi and Garatti (2008) and (2018) (the bound and the
wait-and-judge level); and, in the source verification pass,
- Campi, Garatti and Prandini (2009) [REF-CAMPI-GARATTI-PRANDINI-2009]:
  Theorem 1 with the explicit size `N >= (2 / epsilon)(ln(1 / beta) + d)`
  (Eq. 4) and the exact condition of Campi and Garatti (Eq. 5). Compared with
  `scenario_sample_size`: the explicit size derived here
  (`(2 / epsilon)(d - 1 + ln(1 / beta))`, EQ-REL-SCENARIO-002) is the same
  Chernoff form with `d - 1` for `d`, hence `2 / epsilon` smaller, and was
  checked never to fall below the exact size. The tutorial's worked example
  (`epsilon = 0.01`, `beta = 1e-10`, three variables, `N = 5206`) is
  reproduced from Eq. (4) and pinned in
  `test_published_tutorial_example_and_explicit_size`; the exact size of the
  library for it is smaller.
- Nemirovski and Shapiro (2006), Optimization Online preprint of December
  2004 (introduction): the only generic case with a provably convex chance
  constraint feasible set for a bilinear constraint is normal coefficients and
  `epsilon <= 1/2`; this matches the domain of `normal_linear_chance_margin`
  (EQ-REL-CHANCE-NORMAL-001). The published version was not opened.
- Luedtke and Ahmed (2008), Optimization Online preprint of 2 May 2008
  (Section 2): the sample approximation `(1/N) #{G(x, xi_i) <= 0} >= 1 - gamma`
  is the constraint of EQ-REL-SAA-001, with `gamma = 0` the scenario program.
  Their sample-size theorems (3, 5, 7) are not used. The published version was
  not opened.

Not opened: Calafiore and Campi (2006) (no open copy found; the tutorial
above credits it), Du and Chen (2004), Aoues and Chateauneuf (2010) (publisher
pages only). The shifting-vector formulation and the classification of RBDO
methods remain re-derived and stated from catalogue descriptions. Tu, Choi and
Park (1999) and Der Kiureghian, Zhang and Li (1994) were already registered.

## Consequences

- The SORA shift is taken in physical space as `mu(d) - x*`; it is exact for
  linear limit states in normal inputs with design-independent standard
  deviations and a fixed-point repair otherwise. Inputs whose spread depends
  on the design converge more slowly.
- Both RBDO methods are first order and local; the nonlinear benchmark agrees
  between them, but a second design-point or a multimodal limit state is not
  detected. The SORM or sampling check is advisory.
- The sampled chance-constrained program inherits the optimiser's curse: the
  sampled optimum is biased low and its violation level on fresh draws is
  above the nominal one; the certificate, not the sample, states the
  violation level.

## Deliberately left out

- Smoothing or relaxation of the indicator (sigmoid, CVaR, Bernstein and
  other convex conservative approximations of Nemirovski and Shapiro): the
  order-statistic form is exact for the sample; the convex approximations are
  not implemented.
- Sample-size results for the sample-average approximation of Luedtke and
  Ahmed (Theorems 3, 5, 7 of their preprint); the scenario sizes cover the
  convex case and the certificate covers the rest.
- Single-loop RBDO methods (SLA, KKT-based) and benchmark problems of Aoues
  and Chateauneuf beyond the one used in the tests.
- Scenario programs with discarded constraints (sampling and discarding), and
  degenerate programs for the a-posteriori bound.
