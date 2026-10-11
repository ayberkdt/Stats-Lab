# Model Discrepancy and Heteroscedastic Noise

This guide extends [nonlinear calibration and MCMC](nonlinear-calibration-and-mcmc.md)
to simulators that are known to be wrong. Every simulator is a simplification
of the process it represents, so calibrating it as if it were exact usually
gives a parameter estimate that is too confident and predictions that are
biased. The Kennedy-O'Hagan formulation adds an explicit, correlated
**model discrepancy** term to the calibration. This guide also covers
observation noise whose size varies across experiments.

Every number below was computed in this session with
`examples/discrepancy_calibration_lab.py` (seeded; wall time 11-15 s over two runs) or with
the seeds in `tests/test_discrepancy_calibration.py`.

## 1. The model

For experiment `i` with conditions `c_i` ([EQ-KOH-MODEL-001]):

`y_i = eta(c_i, theta) + delta(x_i) + eps_i`,
`delta ~ GP(0, k_psi)`, `eps_i ~ N(0, sigma_i^2)`.

- `eta` is the simulator, declared as a `ProbabilisticModel`. Its input law is
  the prior of `theta`.
- `x_i` is the subset of conditions the discrepancy depends on.
- `delta` is a zero-mean Gaussian process with a squared-exponential or
  Matern-5/2 kernel ([EQ-KOH-KERNEL-001]). Its hyperparameters are
  `psi = (sigma_delta, ell_1..ell_p)`. Each one is either fixed or given a
  prior on `(0, inf)`.
- `eps_i` is measurement error. Its standard deviation is either known per
  observation (`KnownNoise`) or modelled as `sigma(c) = exp(a + b' g(c))`
  (`LogLinearNoise`, [EQ-KOH-HET-001]).

The discrepancy differs from the scalar `NoiseModel.discrepancy_variance` of
`CalibrationProblem` in one key way: it is **correlated across inputs**. A
simulator that is too low at `x = 0.6` is probably too low at `x = 0.7` as
well, and the Gaussian process encodes exactly that.

Because `delta` is Gaussian and enters linearly, it is integrated out
([EQ-KOH-MARG-001]):

`y | theta, psi, sigma ~ N(eta(C, theta), K_psi + diag(sigma_i^2))`.

`DiscrepancyCalibration.log_likelihood` evaluates this with one Cholesky
factorization per call, which costs `O(n^3)`. The method is meant for tens of
field observations. Without an active discrepancy the covariance is diagonal,
and the code uses the `O(n)` sum of independent Gaussian terms.

## 2. Declaring a problem

```python
import math
from probstat.distributions import LogNormal
from probstat.inverse import (
    DiscrepancyCalibration, DiscrepancyProcess, Experiment, KnownNoise,
)

discrepancy = DiscrepancyProcess(
    ("x",),                                  # discrepancy inputs
    kernel="squared_exponential",            # or "matern52"
    standard_deviation_prior=LogNormal(math.log(0.2), 0.75),
    length_scale_priors=(LogNormal(math.log(0.5), 0.5),),
)
problem = DiscrepancyCalibration(model, experiments, "y", KnownNoise(0.03), discrepancy)
```

Parameter vectors are `(theta, log sigma_delta, log ell, a, b)`. Only free
hyperparameters and noise coefficients appear in them; see
`problem.vector_names`. The log scale makes the hyperparameters unconstrained.
Their prior densities carry the Jacobian `+ log(value)` ([EQ-KOH-POST-001]).

## 3. Estimation

- **Point estimate.** `problem.fit()` maximizes the log posterior (MAP) by
  BFGS. `problem.fit(objective="likelihood")` maximizes the likelihood (MLE).
  The covariance is the inverse finite-difference Hessian. It is `None` when
  the Hessian is not positive definite, which is typical when a variance
  drifts towards zero. `restarts=` adds starts drawn from the prior.
- **Full Bayes.** `problem.sample_posterior()` follows the same recipe as
  `CalibrationProblem.sample_posterior`:
  1. Laplace approximation in sampling coordinates `(u, log psi, a, b)`, where
     `theta = T^-1(u)` is the Nataf map of the prior.
  2. Over-dispersed chain starts.
  3. Adaptive Metropolis by default, or `method="hmc"` / `"nuts"` with
     central-difference gradients.

  The result holds chains in vector coordinates (`posterior.vector`), on the
  physical scale (`posterior.mcmc`: `theta, sigma_delta, length[x], ...`) and
  in sampling coordinates.

**Reduction check.** With no discrepancy (`discrepancy=None`, or a fixed
`standard_deviation=0`) and known noise, the method reproduces
`CalibrationProblem`. The test suite checks this on the exponential-decay
problem (21 points, seed 7). Log posteriors agree to 10 decimal places. The
MAP and Laplace covariance agree with `laplace_posterior()`. The seeded
adaptive-Metropolis chains agree draw by draw to within `1e-7`. In an
exploratory run with 2 chains of 500 draws (seed 5), the maximum difference
was `3.1e-11`.

## 4. Prediction: simulator alone versus bias-corrected

`problem.predict(source, conditions)` accepts a posterior, a fit, or explicit
vectors. For each parameter draw `s` it computes the Gaussian conditional
moments of the discrepancy at the new input `x*` ([EQ-KOH-PRED-001]):

`m_s = k*' C_psi^-1 r_s`, `v_s = k(x*, x*) - |L^-1 k*|^2`, with
`r_s = y - eta(C, theta_s)`.

The returned `DiscrepancyPrediction` keeps three things separate:

- `interval(kind="simulator")`: the sample of `eta(c*, theta_s)`, which
  carries parameter uncertainty only.
- `interval(kind="corrected")`: the mixture `S^-1 sum_s N(eta_s + m_s, v_s)`
  for the true process `eta + delta`. The limits invert the mixture CDF
  exactly.
- `interval(kind="observed")`: the corrected mixture plus `sigma(c*)^2` for a
  new measurement.

`discrepancy_mean` is the estimated discrepancy contribution at `x*`.
`corrected_variance = Var_s(eta_s + m_s) + mean_s(v_s)` by the law of total
variance.

### Worked example

The simulator is `eta(x; k) = 4 exp(-k x)`. The truth is
`4 exp(-0.4 x) + 0.25 sin(2.5 x)`, observed at 15 points on `[0, 4]` with
noise sd 0.03 (seed 11). The prior is `k ~ LogNormal(log 0.5, 0.5)`. The run
used 2 chains of 1000 draws after 500 warm-up iterations, with seed 3.

| | `k` |
| --- | --- |
| plain `CalibrationProblem` | 0.3925 +- 0.0023, 90% 0.3887..0.3964 (excludes 0.4) |
| KOH MAP | 0.3930 (SE 0.0377) |
| KOH posterior | mean 0.3952, sd 0.0450, R-hat 1.002, ESS 165 |

Without a discrepancy, the posterior of `k` is 20 times narrower and excludes
the true value. The discrepancy posterior has `sigma_delta` mean 0.289 and
`length[x]` mean 0.655.

| held-out `x` | truth | simulator-only 95% | bias-corrected 95% | `delta` mean |
| --- | --- | --- | --- | --- |
| 0.65 | 3.3338 | [2.901, 3.270], misses | [3.300, 3.388], covers | +0.248 |
| 1.90 | 1.6208 | [1.564, 2.219], covers | [1.601, 1.687], covers | -0.252 |
| 3.14 | 1.3892 | [0.847, 1.510], covers | [1.374, 1.459], covers | +0.248 |
| 4.50 | 0.4192 | [0.433, 0.990], misses | [0.093, 0.720], covers | -0.281 |

Inside the data range, the corrected intervals are narrow and cover the truth.
At `x = 4.5`, beyond the data, the corrected interval widens back towards the
prior discrepancy variance. That width is the honest answer: the data say
nothing about the discrepancy there.

The test suite runs the same problem with 500 draws after 400 warm-up
iterations. At `x = 0.65` it gives a corrected interval [3.300, 3.388] and a
simulator-only interval [2.876, 3.262], which misses the truth.

## 5. Identifiability: the Brynjarsdottir-O'Hagan point

`theta` and `delta` are confounded: a shift in `theta` can be absorbed by
`delta`. Brynjarsdottir and O'Hagan (2014) show that a Gaussian-process
discrepancy keeps predictions inside the data range accurate, but the physical
parameter is recovered only when the prior on the discrepancy is realistic.

`problem.identifiability(result)` reports two symptoms ([EQ-KOH-IDENT-001]):

1. the posterior correlation between each `theta_j` and each free
   hyperparameter (`log sigma_delta`, `log ell`);
2. the posterior/prior standard-deviation ratio of each `theta_j`.

It returns a warning above `|rho| = 0.5` or a ratio of `0.8`. Neither number
proves identifiability, because the confounding is with the discrepancy
*function*, which has been integrated out.

Part 2 of the lab uses a confounded discrepancy: the truth is
`4 exp(-0.4 x) - 0.12 x`, observed at the same 15 points (seed 12).

| analysis | `k` (90% interval) | sd ratio | max `|rho|` | corrected 95% at `x = 2.1` (truth 1.475) |
| --- | --- | --- | --- | --- |
| plain calibration (Laplace) | 0.4924 +- 0.0031 | — | — | — |
| KOH, `sigma_delta ~ LogNormal(0, 0.75)` | 0.4101 (0.3284..0.5077) | 0.204 | 0.468 | [1.446, 1.497] |
| KOH, `sigma_delta ~ LogNormal(log 0.05, 0.3)` | 0.4746 (0.4287..0.5183) | 0.090 | 0.484 | [1.446, 1.502] |

The plain calibration is confidently wrong. The vague discrepancy prior gives
a wide interval for `k` that contains 0.4. The tight prior says, in effect,
"the simulator is nearly right", and this pulls `k` back towards the biased
value, so the true value falls outside its 90% interval. Both KOH analyses
predict the process equally well. The data constrain `eta + delta`, not how it
splits into `eta` and `delta`; that split comes from the prior. In this run
the correlations stayed just below the 0.5 warning threshold. The report is a
screening aid, not a guarantee.

## 6. Heteroscedastic noise

`KnownNoise((s_1, ..., s_n))` declares known per-observation standard
deviations. `LogLinearNoise(g, prior_a, (prior_b1, ...))` estimates
`sigma(c) = exp(a + b' g(c))` jointly with `theta` ([EQ-KOH-HET-001]). This is
the multiplicative heteroscedasticity model of Harvey (1976). Both noise
models work with or without a discrepancy.

Part 3 of the lab fits `y = c0 + c1 x` with `sigma(x) = exp(-2 + 0.5 x)` at
200 points (seed 2026), using the MLE:

| parameter | estimate | SE | truth | z |
| --- | --- | --- | --- | --- |
| c0 | +0.9757 | 0.0312 | 1.00 | -0.78 |
| c1 | +0.6984 | 0.0240 | 0.70 | -0.07 |
| a | -1.8914 | 0.1012 | -2.00 | +1.07 |
| b | +0.4664 | 0.0440 | 0.50 | -0.76 |

A constant-noise least-squares fit of the same data reports
`c1 = 0.7088 (SE 0.0303)` and `sigma = 0.4969`. That single sigma is too large
at small `x` and too small at large `x`. The plug-in 95% observation intervals
of the heteroscedastic fit are [0.952, 1.698] at `x = 0.5` and [1.907, 4.933]
at `x = 3.5`.

## 7. Boundaries

- The simulator is evaluated directly. There is no emulator Gaussian process
  for expensive codes, no scale factor `rho`, and no non-zero discrepancy mean.
  Each of these is part of the full Kennedy-O'Hagan framework.
- One observed quantity per problem. Multivariate discrepancies and
  cross-quantity correlation are not modelled.
- `O(n^3)` per likelihood evaluation in pure Python. Use tens of observations,
  not thousands.
- Predictions are pointwise marginals. Joint draws across several `x*` are not
  provided.
- NUTS and HMC use finite-difference gradients, so each leapfrog step costs
  `2 d + 1` likelihood evaluations.
- The identifiability report shows symptoms of confounding; it is not a test
  of identifiability.

## References

[REF-KENNEDY-OHAGAN-2001], [REF-HIGDON-ETAL-2004],
[REF-BRYNJARSDOTTIR-OHAGAN-2014], [REF-RASMUSSEN-WILLIAMS-2006],
[REF-HARVEY-1976].
