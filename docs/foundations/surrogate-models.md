# Surrogate Models: Polynomial Chaos and Gaussian Processes

A surrogate (emulator) is a cheap approximation of an expensive forward model,
fitted to a limited number of model runs. `probstat.surrogates` builds
surrogates directly from a `ProbabilisticModel`, so the joint input law used
for propagation, reliability, and sensitivity also defines the training
design. A fitted surrogate can then replace the model inside a new
`ProbabilisticModel`. All numbers below were produced by the library with the
stated seeds.

## 1. Polynomial chaos expansion

A PCE writes the output as a sum of multivariate polynomials that are
orthonormal under the input law [EQ-SUR-PCE-001]:

`Y ~= sum_alpha c_alpha Psi_alpha(xi)`, with `E[Psi_alpha Psi_beta] = delta`.

Two bases are available:

- `basis="hermite"`: orthonormal probabilists' Hermite polynomials
  [EQ-SUR-PCE-002] in the independent standard-normal variables `u` of the
  model's Nataf transformation [EQ-MOD-NATAF-001]. It works for any marginals
  and any Gaussian copula. Non-Gaussian marginals become non-polynomial
  functions of `u`, though, so convergence is slower.
- `basis="legendre"`: orthonormal Legendre polynomials [EQ-SUR-PCE-003] of
  affinely mapped, independent uniform inputs. This is the Wiener-Askey
  optimal basis for uniform inputs.

**Truncation.** `multi_indices(d, p, q_norm=q)` keeps
`(sum alpha_i^q)^(1/q) <= p` [EQ-SUR-PCE-004]. For `d = 3`, `p = 10`, the
total-degree set has 286 terms; the hyperbolic set with `q = 0.75` keeps 144
by dropping high-order interactions first.

**Fit and validation.** `fit_polynomial_chaos` draws an experimental design
(LHS by default, twice as many points as terms), maps it through the joint
input law, evaluates the model, and solves the least-squares normal equations
by Cholesky factorization [EQ-SUR-PCE-005]. The leave-one-out error comes from
the hat-matrix diagonal without re-running the model [EQ-SUR-PCE-006]. The
mean is `c_0`, the variance is the sum of the other squared coefficients
[EQ-SUR-PCE-007], and Sobol indices are partial sums of squared coefficients
[EQ-SUR-PCE-008].

**Worked example: Ishigami** [EQ-LAB-ISHIGAMI-001]. Legendre basis, LHS,
`seed=1`; analytic values are mean 3.5, variance 13.8446,
`S = (0.3139, 0.4424, 0)`, and `S_T = (0.5576, 0.4424, 0.2437)`:

| degree / `q` | terms | runs | LOO error | mean | variance | `S_1, S_2, S_3` | `S_T1, S_T2, S_T3` |
|---|---|---|---|---|---|---|---|
| 2 / 1.0 | 10 | 20 | 3.70 | 2.980 | 6.405 | 0.403, 0.074, 0.356 | 0.569, 0.095, 0.504 |
| 6 / 1.0 | 84 | 168 | 0.142 | 3.402 | 15.335 | 0.270, 0.429, 0.002 | 0.563, 0.499, 0.259 |
| 10 / 0.75 | 144 | 288 | 0.0018 | 3.492 | 13.836 | 0.3133, 0.4423, 0.0000 | 0.5575, 0.4427, 0.2443 |

The LOO error is the diagnostic to read. At degree 2 it exceeds one: the
expansion predicts held-out points worse than the output mean would, and its
indices are meaningless (for example `S_3 = 0.356`, although `x_3` has no main
effect). At degree 10 the LOO error is 0.18% and every index is within 0.001
of its analytic value. The coefficient-based second-order index
`S_13 = 0.2439` recovers the `x_1`-`x_3` interaction that main-effect ranking
misses.

**Exactness check.** A cubic `1 + 2a - a b^2 + 0.5 b^3` in independent normal
inputs is reproduced at degree 3 to machine precision, with LOO error below
`1e-22`. The Nataf map is affine for normal inputs, so the Hermite expansion
is exact.

**Dependence.** With correlated inputs the Hermite PCE still yields the
correct mean and variance. The tests check this on correlated lognormals
against the exact sum moments. Its Sobol indices, however, would attribute
variance to the independent Nataf variables rather than to the physical
inputs, so `sobol_indices()` raises for dependent inputs instead of returning
misleading numbers.

## 2. Gaussian-process regression

A GP prior `f ~ GP(0, k)` conditioned on training runs gives a Gaussian
predictive law: the mean interpolates the data, and the variance grows away
from it [EQ-SUR-GP-003]. Two ARD kernels are provided: squared exponential
[EQ-SUR-GP-001] and Matérn 5/2 [EQ-SUR-GP-002], with one length scale per
input.

`fit_gaussian_process` standardizes the outputs, then maximizes the log
marginal likelihood [EQ-SUR-GP-004] over log hyperparameters. It uses BFGS
with the exact gradient and several seeded restarts. The **nugget** is an
explicit diagonal variance in standardized units (default `1e-10`). It is
never enlarged automatically: if the kernel matrix is numerically singular, a
clear error asks for a larger nugget. `estimate_noise=True` adds a fitted
noise variance on top of the nugget. Closed-form leave-one-out residuals
[EQ-SUR-GP-005] are available from `leave_one_out()`.

**Worked example.** Twelve equally spaced points on `[0, 6]` of
`sin(x) + 0.3 x`, nugget `1e-10`, `seed=4`:

| kernel | log-lik. at start | log-lik. fitted | `sigma_f^2` | `ell` | max error at 11 midpoints | LOO error |
|---|---|---|---|---|---|---|
| squared exponential | -128.78 | 40.21 | 9.71 | 3.01 | 4.8e-6 | 3.6e-9 |
| Matérn 5/2 | -19.81 | 7.49 | 32.69 | 7.58 | 1.8e-3 | 2.5e-4 |

Both fits interpolate the training data. The squared exponential is more
accurate here because the target is analytic. The predictive variance at
`x = 12`, far outside the data, is 2.79 (squared exponential) and 10.47
(Matérn), compared with about `1e-11` inside the design. The variance reflects
the absence of data, not the true error.

Use GPs with caution. The cost is cubic in the number of runs, and every
likelihood evaluation forms the full inverse kernel matrix, so this
implementation targets tens of points. Grid designs can create degenerate
likelihood optima: a length scale shrinks so that each grid line is fitted
independently. Use space-filling random designs, which
`fit_gaussian_process_to_model` does by default with LHS.

## 3. Surrogates inside the central model

`surrogate_probabilistic_model(model, {"y": surrogate})` wraps any object with
`predict(x)` in a `SurrogateModel` forward model. The result is a new
`ProbabilisticModel` with the same joint input law and QoI declarations.
`propagation.propagate`, `ProbabilisticModel.unit_function` with
`sensitivity.estimate_sobol_indices`, and every other workflow that accepts a
`ProbabilisticModel` run unchanged on it, and none of their calls touch the
expensive model. The tests propagate a PCE emulator of Ishigami with 1024
Sobol points and recover its analytic mean and variance within sampling
error. They also estimate pick-freeze Sobol indices on the emulator that
agree with the PCE's coefficient-based indices.

The emulator's answers inherit the surrogate's approximation error, which
none of these workflows can see. Report the LOO error next to any result
computed on a surrogate.

## Limitations

- PCE: ordinary least squares on a fixed truncation only. There is no sparse
  regression (LARS/OMP), no adaptive degree selection, no arbitrary
  Wiener-Askey families beyond Hermite and Legendre, and no quadrature
  projection.
- GP: constant (centered) mean only, no universal kriging trend, stationary
  kernels, dense `O(n^3)` algebra, and hyperparameters restricted to
  `|log theta| <= 25`.

## Sources

- [REF-XIU-KARNIADAKIS-2002], [REF-SUDRET-2008], [REF-SUDRET-2007],
  [REF-BLATMAN-SUDRET-2011], [REF-NIST-DLMF-ORTHOGONAL].
- [REF-RASMUSSEN-WILLIAMS-2006, Chapters 2, 4, and 5].
- [REF-SALTELLI-2008, Chapter 4] for the Ishigami benchmark.
