# Optimal and Bayesian Experimental Design

`probstat.design` chooses *where to measure*. The classical half
(`design.classical`) optimizes the information matrix of a linear or locally
linearized regression model. The Bayesian half (`design.bayesian`) scores
experimental conditions of a calibration-style problem by their expected
information gain (EIG). Every number below was computed with the library and
is asserted, or recomputed, in `tests/test_experimental_design.py`.

## 1. Approximate designs and the information matrix (EQ-OED-INFO-001)

An approximate design `xi` puts weight `w_i` on the point `x_i`. For the model
`E[y|x] = f(x)' beta`, its per-run information matrix is
`M(xi) = sum_i w_i f(x_i) f(x_i)'`. An exact design with `n_i` runs at `x_i`
has `w_i = n_i / n`. Working with weights turns a combinatorial problem into a
concave optimization over the probability simplex.

```python
from probstat.design import ApproximateDesign, information_matrix, polynomial_basis

xi = ApproximateDesign.uniform((-1.0, 0.0, 1.0))
information_matrix(xi, polynomial_basis(2))
# ((1, 0, 2/3), (0, 2/3, 0), (2/3, 0, 2/3))
```

## 2. Criteria (EQ-OED-CRIT-001)

`design_criterion(xi, basis, "D" | "A" | "E" | "I", region=...)` returns
`log det M`, `tr M^-1`, `lambda_min(M)`, or the average of
`d(x) = f(x)' M^-1 f(x)` over region points. For the three-point design above:

| Criterion | Value | Closed form |
|---|---|---|
| D | `log det M = -1.909543` | `log(4/27)` |
| A | `9` | `3 + 3/2 + 9/2` |
| E | `0.146149` | `(5 - sqrt 17)/6` |
| I (21-point grid on [-1, 1]) | mean of `3 - 4.5x^2 + 4.5x^4` | exact |

A singular `M` gives `D = -inf` and `A = I = inf`, never an exception.

## 3. The equivalence theorem (EQ-OED-KW-001)

For every design `sum_i w_i d(x_i, xi) = tr(M^-1 M) = p`, so the largest
standardized variance is at least `p`. Kiefer and Wolfowitz proved that it
equals `p` exactly when `xi` is D-optimal. `equivalence_check` evaluates
`max_x d(x, xi)` on a candidate set and also reports the bound
`D-efficiency >= exp(1 - max d / p)`, which follows from concavity of
`log det`.

| Design (quadratic, p = 3) | `max d` | Bound | True D-efficiency |
|---|---|---|---|
| `{-1, 0, 1}`, weights 1/3 | `3.000000` | `1.000` | `1` |
| uniform on 21 grid points | `7.482` at `x = -1` | `0.224` | `0.642` |

At the optimum `d(x) = 3 - 4.5x^2 + 4.5x^4` (checked on 2001 points), which
touches `p = 3` only at `-1, 0, 1`.

## 4. Approximate D-optimal designs on a grid (EQ-OED-MULT-001, EQ-OED-FW-001)

`d_optimal_design(candidates, basis, method=...)` stops when
`max d <= p (1 + tolerance)`, i.e. with a certified efficiency of at least
`exp(-tolerance)`.

| Problem | Method | Iterations | Result |
|---|---|---|---|
| Quadratic, grid step 0.1 on [-1, 1], tol `1e-6` | multiplicative | 686 | `{-1, 0, 1}`, weights `1/3 +/- 1.5e-5`; `log det = -1.909543` |
| Same, tol `1e-3` | Fedorov-Wynn | 1989 | efficiency `0.99950` (bound `0.99900`) |
| Straight line, grid on [2, 5] | multiplicative | 113 | `{2, 5}`, weights `1/2` |

Grid algorithms leave tiny weights on neighbours of the true support points.
`ApproximateDesign.condensed(min_weight=..., merge_distance=...)` drops and
pools them. The multiplicative algorithm converges much faster here. The
Fedorov-Wynn steps zig-zag near the optimum, and a `1e-6` tolerance took more
than 20000 iterations.

## 5. Exact designs (EQ-OED-ROUND-001, EQ-OED-EXCH-001)

`efficient_rounding(xi, n)` applies the Pukelsheim-Rieder rule. Weights
`(1/3, 1/3, 1/3)` with `n = 10` give counts `(4, 3, 3)`, and weights
`(1/2, 1/2)` with `n = 7` give `(4, 3)`. `exchange_exact_design` then performs
best-improvement Fedorov exchanges. Starting from runs `(4, 3, 3)` at
`(-0.5, 0.2, 0.9)` with `log det = -4.078`, it made 11 exchanges and reached
`(4, 3, 3)` at `(-1, 0, 1)` with `log det = -1.938`. That is a D-efficiency of
`0.9906` relative to the approximate optimum; the integer constraint costs
the rest.

## 6. Local designs for nonlinear models (EQ-OED-LOCAL-001)

For `eta(t) = A exp(-k t)` the information depends on the unknown `theta`.
`LocalLinearization(response, nominal)` uses the gradient at a nominal value
as the basis. With two points and weights `1/2`,

```text
det M = A^2 exp(-2k(t1 + t2)) (t2 - t1)^2 / 4.
```

This decreases in `t1`, so `t1 = 0`; and `t^2 exp(-2kt)` peaks at `t = 1/k`.
The local D-optimal design is therefore `{0, 1/k}` with equal weights,
whatever the value of `A`. The test confirms this three ways for `k = 0.5`:

- A brute-force search over all pairs of a `0.02` grid on `[0, 6]` finds
  `(0, 2.0)`.
- The multiplicative algorithm on a grid of step `0.25` converges in 766
  iterations to `{0, 2}` with weights `0.50002 / 0.49998`. It used 100
  response evaluations, which is 25 candidates times `2 d`.
- The equivalence check on 601 points gives `max d = 2.000000`, for both
  `A = 3` and `A = 30`.

A local design is only as good as its nominal value. If `k` is uncertain,
use the Bayesian criteria below.

## 7. Expected information gain (EQ-OED-EIG-001)

`EIG(D) = E[log p(y | theta, D) - log p(y | D)]` is the mutual information
between parameters and data. It is the expected KL divergence from the prior
to the posterior, and it does not change under reparameterization.
`BayesianDesignProblem(model, noise)` takes the prior from
`model.inputs` and one known-scale Gaussian `NoiseModel` per observed
quantity. `BayesianDesignProblem.from_calibration(problem)` reuses an existing
`CalibrationProblem`. A *condition* is a mapping of deterministic inputs, as in
`Experiment.conditions`.

### 7.1 Closed form (EQ-OED-EIG-LIN-001)

For `y = J theta + e`, the EIG is `1/2 log det(I + Sigma J' R^-1 J)`. The
example below uses the line `a + b x` with prior `N(0, 1) x N(0, 4)` and
`x = (-1, 0.5, 1)`:

| Noise sd | 2.0 | 1.5 | 1.0 | 0.5 | 0.25 | 0.1 |
|---|---|---|---|---|---|---|
| EIG (nats) | 0.8636 | 1.2198 | 1.8318 | 3.0710 | 4.4159 | 6.2365 |

`linearized_information_gain` (one linearization at the prior median) and
`laplace_information_gain` (EQ-OED-LAP-001) reproduce `1.219830` to `1e-8`.
For a linear model with a Gaussian prior, both are exact.

### 7.2 Nested Monte Carlo (EQ-OED-NMC-001)

| sd | Estimator | N | M | Model evals | Estimate +/- SE | Exact |
|---|---|---|---|---|---|---|
| 1.5 | fresh inner (Ryan) | 300 | 60 | 54 900 | `1.336 +/- 0.096` | 1.220 |
| 1.5 | reuse outer (Huan-Marzouk) | 800 | 800 | 2 400 | `1.266 +/- 0.042` | 1.220 |
| 0.1 | fresh inner | 300 | 20 | 18 900 | `108 +/- 11` | 6.237 |
| 0.1 | reuse outer | 200 | 200 | 600 | `4.972 +/- 0.045` (`<= log 200 = 5.30`) | 6.237 |

The bias has a known sign, and the `sd = 0.1` rows show it clearly:

- **Fresh inner samples.** Jensen's inequality makes the inner log-evidence
  too small, so the estimate is biased **upward**. The bias is `O(1/M)`
  (Ryan; Rainforth et al.). When the EIG is large, the posterior covers a
  fraction of roughly `e^-EIG` of the prior. Almost no inner draw then lands
  near the data, and the estimate blows up. You need `M >> e^EIG`, and
  `M ~ sqrt(N)` once the bias is in its asymptotic regime.
- **Reused samples.** Huan and Marzouk reuse the outer draws as the inner
  sample, including `theta_n` itself. This cuts model evaluations from
  `N (1 + M)` to `N` per condition. Each term is then a prior contrastive
  bound (Foster et al.), so the estimate is biased **downward** and can never
  exceed `log N`. The shared inner sample also correlates the terms, so the
  reported SE is optimistic.

With a moderate EIG (`sd = 1.5`) both estimators land within 4 SE of the
closed form, as the test asserts.

### 7.3 Monotonicity in noise

Halving the noise must increase the EIG. The test checks this for the closed
form, the linearized form, and nested MC with reuse. Nested MC used one seed
for every noise level (common random numbers) and gave
`0.916, 1.883, 3.048, 4.216` for `sd = 2, 1, 0.5, 0.25`.

## 8. Greedy sequential design (EQ-OED-GREEDY-001)

`greedy_design(problem, candidates, size, method=..., baseline=...)` adds the
candidate that maximizes the EIG of the enlarged design. By the chain rule,
that is the candidate with the largest information conditional on the
earlier choices. The baseline and every candidate design at every step are
scored with one common random stream, so *differences* between scores are much
less noisy than the scores. For Laplace the same prior draws are reused, and
each per-draw term can only grow when a condition is added, so the reported
increments are never negative.

- **Slope only** (`y = b x`, `x in {0, 0.25, ..., 1}`). The greedy rule picks
  `x = 1` three times; the EIG `1/2 log 4` is the closed form.
- **Intercept and slope on [-1, 1].** It picks `x = -1` first (a tie with
  `x = 1` goes to the earlier candidate) and then `x = 1`, which is the
  classical D-optimal pair. The EIG equals the closed form for `{-1, 1}`.
- **Decay** (`A ~ N(3, 0.3^2)`, `k ~ LogNormal(log 0.5, 0.2)`,
  `sd = 0.1`, `t in {0, 0.5, ..., 6}`, Laplace with 8 draws).
  - Step 1 picks `t = 0`. Its EIG is `1/2 log(1 + (0.3/0.1)^2) = 1.1513`,
    because `t = 0` informs only `A`.
  - Step 2 picks `t = 2.0 = 1/k`, raising the EIG to `2.062`. The candidate
    scores peak at `t = 2` (`2.062`) against `2.043` at `1.5` and `2.032`
    at `2.5`.
  - This reproduces the local design of Section 6 without fixing `k`. The
    search cost 1248 model evaluations.
  - With `baseline=[{"t": 0}]`, the first added point is `t = 2`.
  - For `{0, 2}`, Laplace gives `2.029 +/- 0.015` and nested MC with reuse
    (`N = 400`) gives `1.903 +/- 0.063`. Laplace ignores posterior skewness;
    nested MC is biased low.

## 9. Limitations

- Classical designs assume homoscedastic, uncorrelated errors and a finite
  candidate grid; continuous refinement of support points is not provided.
- Only the D-criterion is optimized; A, E and I are evaluated only.
- Bayesian design needs known Gaussian noise scales. Unknown scales and
  Student-t noise are rejected.
- The Laplace EIG assumes an approximately Gaussian posterior centred near the
  data-generating parameter. Nested MC bias can dominate when the EIG is
  large.
- Greedy selection is myopic and does not optimize batches jointly.
