# Causal Inference: Graphs, Interventions, and Effect Estimation

`probstat.causal` (layer 3) covers three steps of a causal analysis:

1. **Graphs.** Encode assumptions as a DAG and decide whether an effect is
   identified.
2. **Discrete structural models.** Compute exact interventional
   distributions.
3. **Estimation.** Estimate an average treatment effect (ATE) from data and
   ask how strong unmeasured confounding would have to be to explain it away.

The package is pure Python. It reuses `inference.fit_glm` for logistic
propensity scores, `inference.fit_linear_model` for outcome models, and
`inference.bootstrap_confidence_interval` for seeded bootstrap intervals.

Every number below was produced by running the library. The tests in
`tests/test_causal_inference.py` check the same quantities against
independent computations:

- d-separation by brute-force path enumeration (4032 queries on six random
  DAGs);
- interventional probabilities by hand-written sums of products;
- saturated-covariate estimates by explicit stratification;
- E-values by the closed form.

## 1. Causal graphs and identification

`CausalGraph(edges, nodes=())` validates that the graph is acyclic
(`CyclicGraphError` otherwise). It stores the nodes in a deterministic
topological order.

### d-separation

`is_d_separated(x, y, given)` uses the moralized ancestral graph
(EQ-CAUSAL-DSEP-001):

1. Keep the ancestors of `X u Y u Z`.
2. Join co-parents.
3. Drop directions and delete `Z`.
4. Test reachability.

| Graph | Query | Result |
|---|---|---|
| chain `A -> B -> C` | `A _||_ C` / `A _||_ C | B` | no / yes |
| fork `A <- B -> C` | `A _||_ C` / `A _||_ C | B` | no / yes |
| collider `A -> B <- C`, `B -> D` | `A _||_ C` / `| B` / `| D` | yes / no / no |

### Backdoor criterion and minimal adjustment sets

`satisfies_backdoor(graph, X, Y, Z)` checks two conditions
(EQ-CAUSAL-BD-001):

- no member of `Z` descends from `X`;
- `X _||_ Y | Z` holds in the graph with the edges leaving `X` removed.

`minimal_adjustment_sets` searches subsets exhaustively by increasing size.
It only considers candidates inside the ancestral set of `X u Y` of that
graph, because every minimal set lies there. It raises an error when more
than `max_candidates` (default 20) candidates remain.

Example graph: `Z1 -> Z3 <- Z2`, `Z1 -> X`, `Z3 -> X`, `Z3 -> Y`,
`Z2 -> Y`, `X -> W -> Y`.

| Query | Result |
|---|---|
| `{Z3}` satisfies backdoor? | `False`, because conditioning on the collider `Z3` opens `X <- Z1 -> Z3 <- Z2 -> Y` |
| minimal sets | `(('Z1', 'Z3'), ('Z2', 'Z3'))` |
| minimal sets with `Z1` latent | `(('Z2', 'Z3'),)` |
| minimal sets with `Z3` latent | `()`: no observed backdoor set exists |

**M-bias.** Take `X <- U1 -> M <- U2 -> Y` with `X -> Y`. The empty set is
the only minimal set. Adjusting for `M` alone *fails* the criterion. `{M, U1}`
and `{M, U2}` repair it.

### Frontdoor criterion

`satisfies_frontdoor(graph, X, Y, M)` checks three conditions
(EQ-CAUSAL-FDC-001). For `U -> X`, `U -> Y`, `X -> M -> Y` with `U` latent:

- `{M}` satisfies the criterion, while no observed backdoor set exists.
- Adding `X -> Y` breaks condition (i).
- Adding `U -> M` breaks condition (ii).

### Graph surgery

`graph.do(nodes)` deletes the edges into the intervened nodes.
`graph.remove_outgoing(nodes)` deletes the edges leaving them.

## 2. Discrete structural causal models

`DiscreteCausalModel(domains, parents, tables)` takes one conditional table
per variable. Each row lists probabilities in domain order and must sum to
one; rows must not be missing.

- `joint()` is the Markov factorization (EQ-CAUSAL-FACT-001).
- `interventional_joint({X: x})` and `interventional_distribution(Y, {X: x})`
  use the truncated factorization (EQ-CAUSAL-TRUNC-001).
- `intervene({X: x})` returns the mutilated model. Its `joint()` equals the
  truncated factorization.

### Truncated factorization by hand

The model has binary variables and the edges `Z -> X -> W -> Y` and
`Z -> Y`, with:

- `P(Z=1) = 0.3`;
- `P(X=1 | Z) = 0.4, 0.7`;
- `P(W=1 | X) = 0.2, 0.6`;
- `P(Y=1 | W, Z) = 0.1 (0,0), 0.5 (0,1), 0.4 (1,0), 0.9 (1,1)`.

```text
P(Y=1 | do(X=1)) = 0.7 (0.4*0.1 + 0.6*0.4) + 0.3 (0.4*0.5 + 0.6*0.9) = 0.418
P(Y=1 | do(X=0)) = 0.7 (0.8*0.1 + 0.2*0.4) + 0.3 (0.8*0.5 + 0.2*0.9) = 0.286
```

| Quantity | `X = 1` | `X = 0` |
|---|---|---|
| `P(Y=1 | do(X))`, truncated factorization | 0.418 | 0.286 |
| backdoor formula on the joint, adjusting for `Z` (EQ-CAUSAL-ADJ-001) | 0.418 | — |
| frontdoor formula on `(X, W, Y)` only (EQ-CAUSAL-FD-001) | 0.418 | — |
| naive `P(Y=1 | X)` | 0.477143 | 0.234118 |

The naive contrast is 0.243 and the causal contrast is 0.132.

`backdoor_adjustment` and `frontdoor_adjustment` work on any `JointTable`,
including `JointTable.from_counts(...)` built from data. When positivity fails
(`P(z) > 0` but `P(x, z) = 0`) they raise an error instead of returning a
number.

### Simpson's paradox

The model is `Z -> X`, `Z -> Y`, `X -> Y`, with a binary severity `Z`:

- `P(Z=1) = 0.5`;
- `P(X=1 | Z) = 0.2, 0.8`;
- `P(Y=1 | X, Z)`: `0.8, 0.9` when mild and `0.3, 0.4` when severe.

So treatment raises recovery by 0.10 in each stratum.
`compare_naive_and_adjusted(joint, "X", "Y", ["Z"], outcome_value=1,
treated=1, control=0)` returns:

| | treated | control | difference |
|---|---|---|---|
| naive `P(Y=1 | X)` | 0.50 | 0.70 | -0.20 |
| adjusted `P(Y=1 | do(X))` | 0.65 | 0.55 | +0.10 |

`reversal=True`. The severely ill are mostly treated, so the naive
comparison credits the treatment with their poor prognosis.

## 3. Estimating an ATE from data

The estimators assume:

- conditional exchangeability given `X`;
- positivity, `0 < e(X) < 1`;
- consistency.

They take covariate rows, a 0/1 treatment, and outcomes, and return an
`EffectEstimate` (estimate, standard error, `E[Y(1)]`, `E[Y(0)]`, trimming
counts).

| Function | Method | Standard error |
|---|---|---|
| `estimate_regression_adjustment` | Per-arm OLS outcome models, averaged over everyone (EQ-CAUSAL-REG-001) | `nan`; use the bootstrap |
| `estimate_ipw(..., stabilized=False)` | Horvitz-Thompson (EQ-CAUSAL-IPW-001) | influence function, `e` treated as known |
| `estimate_ipw(..., stabilized=True)` | Stabilized weights, Hajek form (EQ-CAUSAL-SW-001) | same |
| `estimate_aipw` | Doubly robust (EQ-CAUSAL-AIPW-001) | `sd(psi)/sqrt(n)` |

The logistic propensity model (EQ-CAUSAL-PS-001) is fitted by IRLS.
Separation raises `SeparationError`. Scores outside `[0.01, 0.99]` emit a
`PositivityWarning`.

`trim=a` drops units with `e < a` or `e > 1 - a`. That changes the estimand
to the ATE of the overlap population, which the result labels
`"ATE (trimmed population)"`.

### Simulated confounded data

The data come from `simulate(2000, 20261012)` in the test file:

- `X1, X2 ~ N(0, 1)`;
- `logit e = -0.3 + 0.8 X1 - 0.5 X2`;
- `Y = 1 + 2T + 1.5 X1 + X2 + 0.5 T X1 + N(0, 1)`.

The true ATE is 2. The sample has 873 treated units.

| Estimator | Estimate | SE |
|---|---|---|
| naive difference of means | 2.7944 | — |
| regression adjustment | 1.9849 | — |
| IPW, Horvitz-Thompson | 1.9696 | 0.1475 |
| IPW, stabilized | 1.9836 | 0.1230 |
| IPW, stabilized, `trim=0.1` (61 dropped) | 1.9700 | 0.1231 |
| AIPW | 1.9825 | 0.0530 |

The stabilized weights have mean 0.998 and maximum 5.88.

Covariate balance (`standardized_mean_differences`, EQ-CAUSAL-SMD-001) uses
the unweighted pooled SD in both rows, so the two rows share one scale:

| Covariate | SMD raw | SMD after stabilized IPW |
|---|---|---|
| x1 | 0.727 | 0.008 |
| x2 | -0.426 | -0.016 |

### Double robustness: misspecifying one model at a time

The misspecified working model omits the confounder `X1`.

| Outcome model | Propensity model | AIPW (SE) | Single-model comparator |
|---|---|---|---|
| wrong | right | 2.0063 (0.1355) | regression adjustment 3.2925 |
| right | wrong | 1.9897 (0.0488) | IPW 3.3127 (0.1110) |
| wrong | wrong | 3.3043 (0.0925) | — |

AIPW stays within 3 SE of 2 whenever one model is correct. It fails, like
both comparators, when both models are wrong. Double robustness is not
robustness to unmeasured confounding.

### Bootstrap intervals

`bootstrap_effect(..., method="aipw", replicates=40, seed=11)` was run on
`simulate(250, 77)`. It resamples whole units and refits both models in each
replicate. The percentile interval is `[1.861, 2.348]`; the estimate is
2.075 and the bootstrap SE is 0.128. Equal seeds reproduce the replicates
exactly. Use at least 1000 replicates for reporting; the tests use 40 to
stay fast.

## 4. Sensitivity to unmeasured confounding: E-values

`e_value(rr, lower=, upper=)` computes
`E = RR + sqrt(RR (RR - 1))` (EQ-CAUSAL-EVALUE-001). A protective `RR < 1`
is inverted first. The interval E-value uses the limit closest to one, and
is 1 when the interval covers one.

| Input | E (point) | E (CI limit) |
|---|---|---|
| RR 3.9, CI 1.8-8.7 | 7.263 | 3.000 |
| RR 0.5, CI 0.3-0.8 | 3.414 | 1.809 |
| RR 1.5, lower 0.9 | 2.366 | 1.000 |

The RR 3.9 (1.8-8.7) row matches the worked example usually quoted from
VanderWeele and Ding (2017). That match comes from memory; the paper was not
re-read in this session.

`confounding_bias_factor(rr_eu, rr_ud)` (EQ-CAUSAL-BIAS-001) satisfies
`B(E, E) = RR`, and the tests check this identity.
`risk_ratio_from_odds_ratio` and `risk_ratio_from_standardized_difference`
are approximate conversions (EQ-CAUSAL-RRAPPROX-001).

## Limitations

- Identification covers the backdoor and frontdoor criteria only. There is
  no do-calculus engine and no complete ID algorithm
  [REF-SHPITSER-PEARL-2006].
- For treatment *sets*, the backdoor check uses d-separation in the graph
  with the treatments' outgoing edges cut.
- Discrete models enumerate all assignments, so the cost is the product of
  the domain sizes.
- Effect estimation covers a binary treatment and the ATE only, not the ATT
  or ATC. Outcome models are linear and the propensity model is logistic;
  flexible learners and cross-fitting are not provided.
- IPW standard errors ignore propensity estimation. Use `bootstrap_effect`
  for a full accounting.
