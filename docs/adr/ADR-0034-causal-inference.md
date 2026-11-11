# ADR-0034 — Causal inference package

- Status: Accepted
- Date: 2026-10-12

## Context

Roadmap Phase 6 lists "causal graphs, interventions, identification,
sensitivity analysis". The library is pure Python with no runtime
dependencies, and the layer map in `tests/test_architecture.py` is enforced.
Effect estimation needs logistic regression and least squares
(`inference.glm`, `inference.linear_models`) and the bootstrap
(`inference.resampling`). All three live at layer 2.

## Decision

1. **New package `probstat.causal` at layer 3.** It imports `inference`,
   `linalg`, and `foundations` and nothing above layer 2. Nothing imports
   it. It is a peer of `modeling` and `monte_carlo` and does not depend on
   either. The package has four modules: `graphs`, `discrete`, `effects`,
   and `sensitivity`.
2. **One graph type.** `CausalGraph` is a frozen dataclass.
   - Nodes are stored in a canonical topological order: Kahn's algorithm
     with alphabetical tie-breaking. Edges are sorted, so equality does not
     depend on input order.
   - Acyclicity is checked at construction and reported as
     `CyclicGraphError`.
3. **d-separation by the moralized ancestral graph** (Lauritzen et al.
   1990), not Bayes-ball. It is short and easy to audit. The tests compare
   it with brute-force path enumeration on 4032 queries over six random
   seeded DAGs.
4. **Backdoor search is exhaustive with one proven pruning step.**
   - If `Z` d-separates `X` and `Y`, then so does `Z n An(X u Y)`. The
     moral graph of an ancestral subset is an induced subgraph. Every
     minimal set therefore lies inside the ancestral set of `X u Y` in the
     graph with the edges leaving `X` cut.
   - Subsets are scanned by size, and supersets of sets already found are
     skipped.
   - `max_candidates=20` bounds the work explicitly rather than silently.
5. **Treatment sets and mediator sets** use the common
   "cut outgoing edges, then test d-separation" form of the backdoor and
   frontdoor conditions. The guide says so.
6. **Discrete SCMs keep their own `JointTable`** instead of
   `probability.FiniteJointDistribution`. That class is bound to a
   `FiniteSampleSpace` and covers exactly two variables, while causal
   adjustment needs named n-way tables with marginals and conditionals.
   - Conditioning on a zero-probability event raises an error.
   - Positivity failures in the adjustment formulas raise errors and are
     never skipped.
7. **The effect estimators reuse the inference layer.**
   - The propensity model is `fit_glm(binomial, logit)`, so separation
     surfaces as `SeparationError`.
   - The outcome models are per-arm `fit_linear_model` fits. Per-arm fits
     allow effect modification, and with saturated covariates they reduce
     exactly to stratification, which the tests check.
   - `bootstrap_effect` resamples unit indices through
     `bootstrap_confidence_interval`, which is seeded by `RandomStream`.
     Only the percentile, basic, and normal methods are allowed: BCa and
     studentized intervals need `n` jackknife refits per call.
8. **Explicit estimands.**
   - Trimming drops units and relabels the estimand
     `"ATE (trimmed population)"`.
   - Scores outside `[0.01, 0.99]` emit a `PositivityWarning`; they are not
     silently clipped.
   - IPW standard errors treat `e` as known, and the docstrings say so.
9. **The E-value has its interval rule.** The interval E-value uses the
   limit nearest the null and is 1 when the interval covers one. The
   odds-ratio and SMD conversions are separate, explicitly approximate
   helpers.

## Verification

- d-separation: chain, fork, collider, collider-descendant, and M-bias
  cases, plus the brute-force comparison above.
- Backdoor: `{Z1, Z3}` and `{Z2, Z3}` are the minimal sets of the Z1/Z2/Z3
  graph. `{Z3}` alone fails. Making nodes latent changes the answer as
  expected. M-bias gives `((),)`.
- Truncated factorization: `P(Y=1 | do(X=1)) = 0.418` and
  `P(Y=1 | do(X=0)) = 0.286` match the hand sums. The backdoor formula, the
  frontdoor formula, and the mutilated model agree to `1e-12`.
- With a latent confounder `U`, the frontdoor formula on `(X, M, Y)` equals
  the truncated factorization.
- Simpson reversal: naive risk difference -0.20, adjusted +0.10, both
  exact. The `from_counts` round trip gives the same result.
- Seeded simulation (`n = 2000`, true ATE 2):
  - the naive difference is 2.794;
  - the IPW, AIPW, and regression-adjustment estimates fall within 3 SE of
    2;
  - with one model misspecified, AIPW gives 2.006 and 1.990, while the
    single-model comparators give 3.29 and 3.31;
  - with both models wrong, AIPW gives 3.30.
- E-value: RR 3.9 gives 7.263; the CI limit 1.8 gives 3.0; `B(E, E) = RR`.
- The causal test file runs in about 9 s.

## Consequences

- The only identification routes are backdoor and frontdoor. A
  do-calculus or complete-ID engine (Shpitser-Pearl 2006) is future work.
- The following citations were written from the author's knowledge and
  were not re-read in this session: Pearl 2009, Hernán-Robins 2020,
  Rosenbaum-Rubin 1983, Robins-Rotnitzky-Zhao 1994, Imbens-Rubin 2015,
  Shpitser-Pearl 2006, VanderWeele-Ding 2017, and Lauritzen et al. 1990.
  - Pearl's definition and theorem numbers (3.3.1-3.3.4) are cited because
    they are widely reproduced.
  - Other locators are given at section or chapter level only.
  - The `exp(0.91 d)` and `sqrt(OR)` conversions are attributed to
    VanderWeele-Ding 2017 from memory.

## Rejected alternatives

- Putting causal code in `inference` (layer 2): the graph and SCM parts are
  not inference, and a dedicated package keeps `inference` from growing.
- Bayes-ball d-separation: it is equivalent but harder to audit than
  moralization.
- Silently clipping extreme propensity scores: clipping changes the
  estimator without telling the user. The library offers explicit trimming
  and a warning instead.
- A `bootstrap_effect` that accepts arbitrary estimator callables: a closed
  set of methods keeps the resampling of models and data consistent.
