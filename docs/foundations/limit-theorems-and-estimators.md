# Limit Theorems and Properties of Estimators

This guide covers three modules:

- `probstat.probability.limit_theorems` gives **exact or proven** answers:
  - closed-form laws of the sample mean;
  - Kolmogorov distances to `Phi` against the Berry-Esseen bound;
  - the one-term Edgeworth expansion;
  - Chebyshev, Hoeffding, and Bernstein tail bounds against exact tails;
  - delta-method variances against exact variances.
- `probstat.monte_carlo.limit_simulation` gives **simulated illustrations**
  of the weak law, the strong law, and the CLT. Every number it returns
  carries Monte Carlo error.
- `probstat.inference.estimator_properties` covers estimators:
  - a Monte Carlo harness for bias, variance, MSE, relative efficiency, and
    coverage, each with its standard error;
  - Cramer-Rao bounds and exact worked examples.

Every number below was printed by `examples/limit_theorems_lab.py` (seed
`20261012`) or by `tests/test_limit_theorems.py` and
`tests/test_estimator_properties.py`. They are the library's own outputs. None
is copied from the literature.

## 1. The standardized mean and its exact law

For i.i.d. summands with mean `mu`, variance `sigma^2`, and
`rho = E|X - mu|^3`, the standardized mean is
`Z_n = sqrt(n) (Xbar_n - mu)/sigma`. Three summand families have sums with
closed-form laws ([EQ-LIM-002]):

| summand            | law of the sum            | `P(Z_n <= z)`                          |
|--------------------|---------------------------|----------------------------------------|
| `Bernoulli(p)`     | `Binomial(n, p)`          | `P(S <= floor(n p + z sqrt(n p q)))`   |
| `Exponential(l)`   | `Gamma(n, l)`             | `P(n, n + z sqrt(n))` (rate-free)      |
| `Normal(mu, s^2)`  | `Normal(n mu, n s^2)`     | `Phi(z)`                               |

`bernoulli_summand`, `exponential_summand`, and `normal_summand` return a
`SummandLaw` with the mean, the variance, the skewness, and
`rho` ([EQ-LIM-010]). For example, `E|X - 1|^3 = 12/e - 2` for
`Exponential(1)`.

## 2. Berry-Esseen: a bound that always holds

`berry_esseen_bound(rho, sigma, n)` returns `C rho / (sigma^3 sqrt(n))` with
`C = 0.4748` ([EQ-LIM-001]). This constant is the bound Shevtsova (2011)
states for identically distributed summands; the arXiv abstract gives
`C_0 < 0.4748`. `kolmogorov_distance_to_normal` computes the exact
`K_n = sup_z |F_n(z) - Phi(z)|` ([EQ-LIM-003]):

- **Lattice summands (Bernoulli).** The supremum is attained at an atom, so
  both one-sided limits at each of the `n + 1` atoms are compared with `Phi`.
  This is exact.
- **Continuous summands (exponential).** The gap is maximized on a
  1601-node grid over `[-sqrt(n) - 8, 10]` and then refined by golden-section
  search. It agrees with a 40001-point scan to within `1e-6` for `n = 5`.

Bernoulli means:

| p    | n    | exact `K_n` | Berry-Esseen | ratio |
|------|------|-------------|--------------|-------|
| 0.50 | 10   | 0.12305     | 0.15014      | 0.820 |
| 0.50 | 100  | 0.03979     | 0.04748      | 0.838 |
| 0.50 | 1000 | 0.01261     | 0.01501      | 0.840 |
| 0.10 | 10   | 0.23610     | 0.41040      | 0.575 |
| 0.10 | 100  | 0.08316     | 0.12978      | 0.641 |
| 0.10 | 1000 | 0.02660     | 0.04104      | 0.648 |

For `p = 1/2` and `n = 10` the distance is exactly `C(10, 5)/2^11 =
0.123046875`, half the central binomial atom. The fair coin comes closest to
the bound. Across `p in {0.5, 0.3, 0.1, 0.01}` and `n in {1, 2, 5, 10, 30,
100, 500}`, the largest ratio of exact distance to bound is between 0.8 and
1.0, and it is never reached. The bound itself is not small: for
`p = 0.01` it exceeds one (and is vacuous) for every `n <= 21`.

## 3. Edgeworth: correcting for skewness

The one-term expansion `G_n(z) = Phi(z) - phi(z) gamma (z^2 - 1)/(6 sqrt(n))`
([EQ-LIM-004]) removes the `n^-1/2` skewness term. For exponential sums
(`gamma = 2`):

| n   | `sup|F_n - Phi|` | `sup|F_n - G_n|` | Berry-Esseen |
|-----|------------------|------------------|--------------|
| 2   | 0.09446          | 0.058301         | 0.81065      |
| 5   | 0.05963          | 0.017862         | 0.51270      |
| 10  | 0.04211          | 0.007975         | 0.36253      |
| 30  | 0.02429          | 0.002397         | 0.20931      |
| 100 | 0.01330          | 0.000678         | 0.11464      |

- The normal error falls like `n^-1/2`. At `n = 1000`, `sqrt(n) K_n =
  0.13298`, which matches the Edgeworth prediction
  `gamma phi(0)/6 = 1/(3 sqrt(2 pi))`.
- The Edgeworth error falls like `n^-1`: from `n = 10` to `n = 100` it drops
  by a factor of about 12.
- For a lattice law the expansion needs an extra discontinuous term. The
  code raises rather than return a misleading number.

## 4. Tail bounds against exact tails

`compare_tail_bounds(summand, n, t)` reports, for `Xbar_n - mu`:

- the exact one-sided tail `P(Xbar_n - mu >= t)` and the exact two-sided
  tail;
- Chebyshev ([EQ-LIM-005], two-sided);
- Hoeffding ([EQ-LIM-006], one-sided, range `[a, b]`);
- Bernstein ([EQ-LIM-007], one-sided, `|X - mu| <= c`).

For `Bernoulli(0.3)` and `n = 100`:

| t    | exact one-sided | Hoeffding | Bernstein | exact two-sided | Chebyshev |
|------|-----------------|-----------|-----------|-----------------|-----------|
| 0.05 | 1.629e-01       | 6.065e-01 | 5.690e-01 | 3.260e-01       | 0.8400    |
| 0.10 | 2.099e-02       | 1.353e-01 | 1.173e-01 | 3.745e-02       | 0.2100    |
| 0.20 | 2.206e-05       | 3.355e-04 | 4.129e-04 | 2.362e-05       | 0.0525    |

- Every bound holds in the test grid: `p in {0.5, 0.3, 0.05}`,
  `n in {10, 50, 200, 1000}`, and `t in {0.02, 0.05, 0.1, 0.2}`.
- The bounds can be loose by an order of magnitude or more. At `t = 0.2`,
  Hoeffding is 15 times the exact tail.
- Bernstein beats Hoeffding at moderate `t`, where the variance term
  dominates. Hoeffding wins at large `t`, where the range term `c t/3` takes
  over.
- For exponential summands, Hoeffding and Bernstein are reported as `None`,
  because their bounded-support hypothesis fails.

## 5. Delta method against exact variances

The first-order approximation is `Var g(Xbar_n) ~ g'(mu)^2 sigma^2/n`
([EQ-LIM-008]). The table shows the exact/delta ratios:

| n   | `Xbar^2`, normal(1, 1) | `1/Xbar`, exponential(1) |
|-----|------------------------|--------------------------|
| 5   | 1.100                  | 2.604                    |
| 20  | 1.025                  | 1.231                    |
| 100 | 1.005                  | 1.041                    |

- The reciprocal, whose exact variance `(n lambda)^2/((n-1)^2 (n-2))` is
  infinite for `n <= 2`, is badly understated at small `n`.
- For `g(p) = p(1 - p)` at `p = 1/2`, the delta variance is zero, but the
  exact variance for `n = 10` is positive.
- Simulating `Var(Xbar^2)` for normal data with `n = 5` (40000 replications)
  agrees with the exact `0.88` within four standard errors.

## 6. Simulation demonstrations (kept separate)

These helpers live in `probstat.monte_carlo.limit_simulation`. Each takes a
one-summand sampler `sampler(rng)` and a seed, and uses one child stream per
replication ([EQ-LIM-009]). All rows below use Exponential(1) summands and
`eps = 0.2`.

| quantity                                      | simulated          | exact  |
|-----------------------------------------------|--------------------|--------|
| `P(|Xbar_10 - 1| > 0.2)`                      | 0.5405 +/- 0.0111  | 0.5258 |
| `P(|Xbar_50 - 1| > 0.2)`                      | 0.1580 +/- 0.0082  | 0.1547 |
| `P(|Xbar_200 - 1| > 0.2)`                     | 0.0065 +/- 0.0018  | 0.0049 |
| KS distance to `Phi`, `n = 5`, 5000 means     | 0.0659             | 0.0596 |
| KS distance to `Phi`, `n = 30`, 5000 means    | 0.0279             | 0.0243 |

The sampling scale of the KS rows is `1/sqrt(5000) = 0.0141`.

- **Weak law.** `simulate_weak_law` estimates the probabilities in the first
  three rows. The exact values are the gamma tails of section 4.
- **Distribution.** `simulate_clt_distance` returns the KS distance of the
  simulated standardized means to `Phi`. It overestimates the exact
  distance in both rows here, by Monte Carlo noise of order `1/sqrt(R)`.
  At `n = 30` that noise scale is more than half the signal. Prefer the
  exact function whenever it applies.
- **Strong law.** `simulate_running_means` records running-mean paths. It
  also estimates `P(max_{m <= k <= 1000} |Xbar_k - 1| > 0.2)`: 0.850 for
  `m = 10`, 0.093 for `m = 100`, and 0.000 for `m = 1000` (300 paths). The
  finite horizon makes these lower bounds for the strong-law probabilities.
  They decrease in `m`, which is what almost-sure convergence predicts.

## 7. Estimators: bias, variance, MSE, efficiency, coverage

`evaluate_estimator(estimator, sampler, true_value, n, replications, seed,
interval=..., reference=...)` simulates `replications` data sets. It uses
`seed.child("replication", r)`, so the estimator, a competing reference
estimator, and an interval procedure all see the same data. It reports
([EQ-EST-001] to [EQ-EST-003]):

- the bias, with standard error `s/sqrt(R)`;
- the variance `s^2`, with standard error from the fourth central moment;
- the MSE, with the standard deviation of squared errors over `sqrt(R)`;
- the paired relative efficiency `MSE(reference)/MSE(estimator)`, with a
  delta-method standard error;
- the coverage, with its binomial standard error.

**Checking the standard errors.** 60 independent runs (sample mean against
median, `n = 10`, 200 replications each) compare the run-to-run standard
deviation of each quantity with the mean reported standard error:

| quantity            | run-to-run SD | mean reported SE | ratio |
|---------------------|---------------|------------------|-------|
| bias                | 0.02429       | 0.02246          | 1.081 |
| variance            | 0.01095       | 0.00987          | 1.109 |
| MSE                 | 0.01082       | 0.00997          | 1.086 |
| coverage            | 0.01583       | 0.01538          | 1.029 |
| relative efficiency | 0.10414       | 0.10473          | 0.994 |

The test requires each ratio to lie in `[0.7, 1.4]`. The mean coverage of
the 95% z interval over the 60 runs was 0.949.

## 8. Cramer-Rao bounds and MLE efficiency

`cramer_rao_bound(information, n, gradient)` evaluates
`g' I^-1 g / n` from the per-observation Fisher information of
`probstat.information` ([EQ-EST-004]). With `g = grad tau`, it bounds an
unbiased estimator of `tau`. With `g = grad E[T]`, it bounds a biased one.
`mle_efficiency` gives the exact finite-sample results ([EQ-EST-010]):

- **Bernoulli, Poisson, and normal-mean MLEs** attain the bound.
  Efficiency `crlb/MSE = 1`.
- **Normal variance, `n = 10`, `sigma = 1`.** The MLE `SS/n` has variance
  0.18 and MSE 0.19, below the unbiased bound 0.20. Its efficiency is
  1.053: a biased estimator can beat the unbiased bound in MSE. Its
  variance still respects the biased-estimator bound.
- **Exponential rate.** The MLE `1/Xbar` has efficiency
  `(n-1)(n-2)/(n(n+2))`: 0.3429 at `n = 5`, 0.9046 at `n = 50`, and 0.9900
  at `n = 500`. It never attains the bound but is asymptotically efficient.

## 9. Worked examples with exact answers

**Variance divisors** (normal data, `n = 10`, `sigma = 1`, [EQ-EST-005]):

| estimator | bias    | variance | MSE    |
|-----------|---------|----------|--------|
| `SS/9`    | 0       | 0.2222   | 0.2222 |
| `SS/10`   | -0.1000 | 0.1800   | 0.1900 |
| `SS/11`   | -0.1818 | 0.1488   | 0.1818 |

The minimum-MSE divisor is exactly `n + 1`. Simulation with 5000
replications gives the MSE of `SS/(n-1)` as 0.2303 +/- 0.0056, against the
exact 0.2222.

**Uniform(0, theta) maximum** (`theta = 1`, `n = 10`, [EQ-EST-006]):

| estimator        | bias    | MSE     |
|------------------|---------|---------|
| `max`            | -0.0909 | 0.01515 |
| `(n+1)/n max`    | 0       | 0.00833 |
| `(n+2)/(n+1) max`| -0.0083 | 0.00826 |
| `2 mean`         | 0       | 0.03333 |

The UMVUE is unbiased, but a small shrinkage lowers the MSE further. The
moment estimator, which ignores the sufficient statistic, is four times
worse.

**Rao-Blackwellization** (`P(X = 0) = e^-1`, Poisson(1), `n = 10`,
[EQ-EST-007]):

| quantity                        | crude `1{X1=0}` | `((n-1)/n)^S` | CRLB    |
|---------------------------------|-----------------|---------------|---------|
| variance (exact)                | 0.23254         | 0.01423       | 0.01353 |

Simulation with 5000 replications gives the Rao-Blackwellized estimator a
bias of +0.00037 +/- 0.00169 and a variance of 0.01425 +/- 0.00033. Its
efficiency relative to the crude estimator is 16.11 +/- 0.38; the exact
ratio is 16.34.

## 10. Sufficiency and consistency

`check_sufficiency(log_likelihood, statistic, a, b, grid)` applies the
factorization criterion ([EQ-EST-008]). If `T(a) = T(b)`, the
log-likelihood ratio must not depend on `theta`.

- For an exponential family `h(x) exp(eta(theta)' T(x) - A(theta))`,
  `sum T(x_i)` passes by construction. In the tests, a normal mean and a
  Poisson rate both pass with samples `(0, 3, 3)` against `(4, 1, 1)` or
  `(2, 2, 2)`.
- The sample mean fails for a Laplace location: with the same two samples,
  the log-ratio spread exceeds 1.
- A passing check on a finite grid is evidence, not a proof.

`consistency_profile(risk, sizes, eps)` tabulates exact bias, variance, and
MSE against `n`, together with the bound `MSE/eps^2` on
`P(|T_n - theta| >= eps)` ([EQ-EST-009]). When bias and variance both
vanish, the bound goes to zero and the estimator is consistent.

## Limitations

- Exact laws are implemented only for Bernoulli, exponential, and normal
  summands. Other families need simulation or a user-supplied CDF.
- The continuous Kolmogorov distance is a deterministic numerical
  maximization, not a closed form.
- The Edgeworth expansion is one term and non-lattice only.
- Hoeffding and Bernstein are given in their bounded-support forms only.
- `check_sufficiency` can refute sufficiency but cannot prove it.
- Completeness is discussed but not checked numerically.
- `mle_efficiency` covers five family/estimand pairs with exact moments.
