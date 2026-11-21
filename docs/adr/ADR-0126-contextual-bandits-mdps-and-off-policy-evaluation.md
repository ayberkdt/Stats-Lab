# ADR-0126 — Contextual bandits, Markov decision processes, and off-policy evaluation

- Status: Accepted
- Date: 2026-11-21
- Extends: the Beta-Bernoulli bandit of `risk.bandits` and the one-shot
  decision analysis of `risk.decision`

## Context

`probstat.risk` could analyse one decision under uncertainty
(`DecisionProblem`, EVPI, EVSI) and one kind of repeated decision, the
Beta-Bernoulli bandit with Thompson sampling. The roadmap listed
"Contextual bandits and Markov decision processes" as not implemented. Three
questions were out of reach:

- choosing an action from side information while still learning its effect;
- estimating the value of a policy from data that a different policy
  collected, which is what an analyst has when an experiment is not allowed;
- finding a policy when an action changes the state in which the next
  decision is made.

A fourth, smaller one belongs with them: naming the best of `n` options
with a fixed number of trials, where rewards during the search do not count.

## Decision

1. **Four new modules in `risk`** (layer 6), depending only on
   `foundations` and `linalg`: `contextual_bandits.py`, `off_policy.py`,
   `markov_decision.py`, and `best_arm.py`. The layer map is unchanged.
2. **LinUCB with disjoint linear models** (`LinUCB`, [EQ-RISK-LINUCB-001]):
   per arm `A = I + sum x x'`, `b = sum r x`, `theta = A^-1 b`, score
   `theta'x + alpha sqrt(x'A^-1 x)`. `A^-1` is maintained by the rank-one
   update `A^-1 - u u'/(1 + x'u)`, `u = A^-1 x` ([EQ-RISK-LINUCB-002]); the
   identity is verified by multiplying out, and `A` itself is accumulated
   as well so that the two can be compared. `alpha` is a user parameter.
3. **Linear Thompson sampling** (`LinearThompsonSampling`,
   [EQ-RISK-LINTS-001]) for one shared parameter: draw
   `theta ~ N(B^-1 f, v^2 B^-1)` through the Cholesky factor of `B`
   (`B^-1 f + v L'^-1 z`). `v` is a user parameter; the value in the regret
   theorem of Agrawal and Goyal depends on the horizon and a confidence
   level and is deliberately not reproduced.
4. **A simulation driver** `simulate_linear_bandit` for a known linear
   environment (one parameter row per arm, or one shared vector) with
   Gaussian noise and separate child streams for contexts, noise, and
   uniform play. It reports the pseudo-regret ([EQ-RISK-LINUCB-003]). The
   policies are mutable objects with `select` / `update`, so they can also
   be driven by an external system.
5. **Off-policy evaluation** on a validated `LoggedBanditData`:
   `inverse_propensity_score`, `self_normalized_ips`, `direct_method`, and
   `doubly_robust` ([EQ-RISK-OPE-001] to [EQ-RISK-OPE-004]). The doubly
   robust estimator is written for a stochastic target policy,
   `sum_a pi(a|x) q(x,a) + w (r - q(x,a))`; Dudik, Langford and Li state it
   for a deterministic one, and the general form is confirmed by exact
   enumeration. Every `OffPolicyEstimate` carries a standard error from the
   per-record terms (delta method for the ratio estimator), the effective
   sample size and largest weight, the clip and the clipped fraction, and
   the terms themselves ([EQ-RISK-OPE-005]). Clipping is optional and
   documented as biased.
6. **Finite discounted MDPs** (`FiniteMDP`): `P[a][s][s']`, rewards
   `R[a][s]` or `R[a][s][s']` (reduced to expected rewards), `0 <= gamma <
   1`, every action available in every state. `policy_evaluation` solves
   `(I - gamma P_pi) V = r_pi` with `linalg.lu_solve` for deterministic or
   randomized policies ([EQ-RISK-MDP-001]); `policy_iteration` changes an
   action only when it is better by more than `1e-12` relative, so ties and
   rounding cannot cycle ([EQ-RISK-MDP-004]); `value_iteration` stops when
   the sup-norm change is below `eps (1 - gamma)/(2 gamma)`
   ([EQ-RISK-MDP-003]); `backward_induction` handles a finite horizon and
   accepts `discount = 1` ([EQ-RISK-MDP-005]).
7. **Error bounds are derived, not quoted** ([EQ-RISK-MDP-006]). For any `V`
   with Bellman residual `rho` and greedy policy `pi`:
   `|V - V*| <= rho/(1 - gamma)` and `|V_pi - V*| <= 2 rho/(1 - gamma)`.
   After a value-iteration sweep `rho <= gamma |V_{k+1} - V_k|`, which gives
   the stopping rule. `MDPSolution` reports the residual, both bounds, the
   iteration count, a `converged` flag, and the number of backups and
   linear solves.
8. **Sequential halving** (`sequential_halving`, [EQ-RISK-SH-001]) with
   per-round averages as in the source, a required budget
   `T >= n ceil(log2 n)`, and a per-round record. The error-probability
   bound is described qualitatively only.
9. **Tabular Q-learning is left out.** The bibliography has no verified
   source for it, and the project rule is not to add unverified citations.

## Consequences

- One state with reward 1 and `gamma = 0.9`: policy evaluation and policy
  iteration return 10 to `2e-15`; value iteration with `eps = 1e-9` stops
  after 226 sweeps at `10 - 4.6e-10`.
- On 12 random MDPs (2 to 4 states, 2 or 3 actions, `gamma` in
  `[0.3, 0.95]`) policy iteration equals the best of all enumerated
  deterministic policies exactly and value iteration with `eps = 1e-10`
  differs by at most `4.9e-11`.
- Off-policy enumeration over all 144 two-record logs of a tabular problem
  with value 0.576: bias of IPS `-2e-16`; doubly robust with a wrong model
  and true propensities `0`; with wrong propensities and the true model
  `-2e-16`. The same enumeration gives `-0.316` for IPS with wrong
  propensities, `-0.374` for doubly robust with both wrong, `-0.276` for
  IPS clipped at 2, and `-0.234` for self-normalized IPS at `n = 2` (its
  finite-sample bias; a log with all weights zero counted as 0).
- On 4000 sampled records the standard errors are 0.0230 (IPS), 0.0143
  (self-normalized), 0.0228 (doubly robust, wrong model), and 0.0128
  (doubly robust, true model); the effective sample size is 1139.
- LinUCB (4 arms, `d = 3`, `alpha = 1`, noise 0.5, seed 21): regret per
  round 0.068 after 200 rounds and 0.010 after 2000; total 20.7 against
  1224 for uniform play. Linear Thompson sampling (5 arms, shared
  parameter, `v = 0.5`, seed 22): 0.054 and 0.0066; total 13.2 against
  1408. After 300 updates the maintained inverse differs from the Cholesky
  inverse by at most `7e-18`.
- Sequential halving on eight Bernoulli arms (means 0.6, 0.45, 0.4, 0.4,
  0.35, 0.3, 0.3, 0.2): misidentification in 34.7 %, 12.0 %, and 0 % of 300
  seeded runs with budgets 48, 240, and 1200.
- `LinearThompsonSampling` factors `B` in every round (`O(d^3)`); this is
  adequate for the small dimensions of a pure-Python library.
- Value iteration needs about `log(1/eps) / (1 - gamma)` sweeps; the
  docstring points to policy iteration for `gamma` near one.

## Rejected alternatives

- **Hybrid LinUCB with shared features.** A second algorithm of the source
  with considerably more bookkeeping; the disjoint model covers the stated
  need.
- **Reproducing the theoretical `alpha` and `v`.** The constants depend on
  the horizon, the confidence level, and norm bounds that a user rarely
  knows; quoting them from memory would risk a wrong formula.
- **Normalizing propensities or target probabilities silently.** A target
  policy that does not sum to one is rejected.
- **Fitting the reward model inside `doubly_robust`.** The unbiasedness
  argument needs a model that is independent of the evaluated records;
  leaving the fit to the caller keeps that requirement visible.
- **State-dependent action sets and average-reward or undiscounted
  infinite-horizon criteria.** They need different existence conditions;
  only the discounted and the finite-horizon problems are implemented.
- **Q-learning without a registered source.** See Decision 9.
- **Reusing `processes` Markov-chain classes.** `risk` (layer 6) may import
  layer 2, but the solvers need only a transition tensor and a linear
  solve; a plain validated tensor keeps the module self-contained.

## Verification

`tests/test_sequential_decisions.py` (32 tests, about 2 s) checks: the
geometric-series value; agreement of value iteration, policy iteration, and
enumeration of all deterministic policies on random MDPs; the stopping rule
and both residual bounds against the enumerated optimum; reduction of
per-transition rewards; the Bellman equation of a randomized policy; a
two-stage backward induction computed by hand and its long-horizon limit;
the myopic policy at `gamma = 0`; unconverged flags; exact unbiasedness of
IPS and of both doubly robust cases by enumeration, and the bias of the
wrong combinations and of clipping; equality of the doubly robust and direct
terms for deterministic rewards with the true model; shift equivariance of
self-normalized IPS; the standard-error formulas and the effective sample
size; the scalar LinUCB closed form; the rank-one inverse against direct
inversion; the linear Thompson posterior against
`inference.condition_linear_gaussian` and the moments of 6000 draws;
decreasing regret per round and the comparison with uniform play; the
halving schedule, deterministic arms, the budget constraint, and a
decreasing Monte Carlo error rate; reproducibility; and input validation.

## References

[REF-LI-ETAL-2010], [REF-AGRAWAL-GOYAL-2013],
[REF-DUDIK-LANGFORD-LI-2011], [REF-NORRIS-1997], [REF-PUTERMAN-1994],
[REF-KARNIN-KOREN-SOMEKH-2013], [REF-OWEN-2013], [REF-GOLUB-VANLOAN-2013].
