# Nonlinear Calibration and Markov Chain Monte Carlo

This guide extends [inverse problems and Bayesian calibration](inverse-problems-and-calibration.md)
from straight lines with known noise to general nonlinear forward models,
unknown or heavy-tailed noise, and sampled posteriors. Every number quoted
below was computed with this library and the seeds used in
`tests/test_uq6b_nonlinear_calibration.py`, so it can be reproduced exactly.

## 1. What changes when the model is nonlinear

The observation model keeps the same structure as before, but the response is
now any deterministic model `g` evaluated under experimental conditions `c_j`
([EQ-CAL-NL-001]):

`y_jq = g_q(theta; c_j) + delta_jq + eps_jq`.

Three things stop being true:

1. The posterior is no longer Gaussian, so there is no closed-form mean and
   covariance. We either approximate it (Laplace) or sample it (MCMC).
2. Identifiability depends on where you look: the Jacobian, and therefore the
   information in the data, changes with `theta`.
3. Predictions are no longer linear in `theta`, so the predictive law is found
   by pushing posterior draws through the model rather than by a covariance
   sandwich.

## 2. Declaring a calibration problem

A study is declared once as a `ProbabilisticModel`. For calibration its joint
input law is read as the **prior** on the calibration parameters.

```python
import math
from probstat.distributions import LogNormal, Uniform
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.uq import QuantityOfInterest
from probstat.inverse import CalibrationProblem, Experiment, NoiseModel, predict

def decay(p):
    return {"y": p["A"] * math.exp(-p["k"] * p["t"])}

prior = JointInput.independent(
    RandomInput("A", Uniform(1.0, 10.0)),
    RandomInput("k", LogNormal(math.log(0.5), 1.0)),
)
model = ProbabilisticModel(prior, decay, (QuantityOfInterest("y", "signal"),))
experiments = tuple(Experiment({"y": y}, {"t": t}) for t, y in data)
problem = CalibrationProblem(model, experiments, (NoiseModel("y", standard_deviation=0.1),))
```

Each `Experiment` carries its own conditions (here the time `t`). Conditions
override entries of `model.fixed` with the same name and may never name a
calibration parameter; an experiment index can be passed as a numeric
condition. One `NoiseModel` is required per observed quantity.

## 3. Likelihoods and the log-noise parameterization

For a residual `r` and total scale `s^2 = sigma_q^2 + sigma_delta_q^2`
([EQ-CAL-LIK-001]):

- Gaussian: `log p = -0.5 (log(2 pi s^2) + r^2 / s^2)`;
- Student-t with `nu` degrees of freedom: heavier tails, so a single outlier
  cannot dominate the fit.

`sigma_delta_q^2` is the declared model-discrepancy variance. It has the same
meaning as `validation.AdditiveGaussianDiscrepancy.variance`, but the inverse
layer accepts a plain float so it never imports the higher validation layer.

When `sigma_q` is unknown it receives its own prior and is sampled as
`eta = log sigma`. A density does not transform like a function: the prior on
`eta` is `p_sigma(e^eta) e^eta`, and the extra `e^eta` (an additive `+ eta` on
the log scale) is the Jacobian `|d sigma / d eta|` ([EQ-CAL-POST-001]).
Forgetting it silently changes the prior.

**Robustness example.** Twenty points on `1 + 2x` with noise `0.1`, one
observation shifted by `+5`. The Gaussian MAP intercept is `1.268`; with a
Student-t likelihood (`nu = 3`) it is `1.0475`. The truth is `1`.

## 4. Least squares and identifiability

`fit_least_squares` minimizes `0.5 sum (r_i / s_i)^2` by Levenberg-Marquardt
([EQ-CAL-WLS-001]) and reports the Gauss-Newton covariance
([EQ-OPT-LM-002]). It is the frequentist baseline: the prior is not used.

Exponential decay, true `A = 5`, `k = 0.3`, 21 times `t = 0, 0.5, ..., 10`,
noise `0.1`:

| Quantity | Estimate | Standard error |
|---|---:|---:|
| `A` | 5.0031 | 0.0680 |
| `k` | 0.3008 | 0.0064 |

The fit used 819 forward-model runs (finite-difference Jacobians included).

Identifiability is read from the spectrum of the Fisher information
`F = J_w' J_w` ([EQ-CAL-FISHER-001]). Scaling by prior standard deviations,
`D F D`, makes it dimensionless: an eigenvalue of 100 means the data carry 100
times the prior information along that direction. At the least-squares point
the eigenvalues are `1426.6` and `51919.8` (condition number `36.4`), so both
parameter directions are strongly data-informed.

Now take `y = a b x`. Only the product is informed. At `a = b = 1` with
three observations, the prior-scaled eigenvalues are `0` and `2800`: the
rank is one, and the null direction is `(a, b) = (0.7071, -0.7071)`. Moving
along it (raising `a`, lowering `b`) leaves every prediction unchanged to
first order. The diagnostic names the **combination** that is unidentified,
which a per-parameter standard error cannot do. The result is local:
re-evaluate it at the posterior mode, not only at the prior mean.

## 5. Laplace approximation

The Laplace approximation ([EQ-LAPLACE-001]) places a Gaussian at the mode:
`N(x*, H^-1)` with `H` the Hessian of `-log p`. It also approximates the
normalizing constant,
`log Z ~= log p(x*) + (d/2) log(2 pi) - (1/2) log det H`.

For the unnormalized Gaussian with covariance `[[1, 0.8], [0.8, 2]]` the
library returns the mode `(1, -2)`, the covariance exactly, and
`log Z = 1.99161942`, which equals `0.5 log((2 pi)^2 x 1.36)`. Fifty
log-density evaluations were needed (BFGS plus the finite-difference Hessian).

For a straight-line calibration `y = a + b x` (prior `a ~ N(0, 2^2)`,
`b ~ N(1, 1)`, seven points at `x = 0, 0.5, ..., 3`, known noise `0.3`),
`CalibrationProblem.laplace_posterior` reproduces the conjugate result of
`condition_linear_gaussian` to six decimals: mean `(0.641712, 0.787485)`,
variances `0.040994` and `0.012604`, covariance `-0.018846`. That agreement
is the bridge between the linear and nonlinear APIs.

## 6. Markov chain Monte Carlo

### Random-walk and adaptive Metropolis

A random-walk Metropolis step proposes `x' = x + L z` and accepts with
probability `min(1, p(x')/p(x))` ([EQ-MCMC-MH-001]). Only density **ratios**
are needed, so the unknown normalizing constant cancels.

Adaptive Metropolis ([EQ-MCMC-AM-001]) learns the proposal covariance
`s_d (Cov(history) + eps I)` with `s_d = 2.38^2 / d`. Here the adaptation runs
**only during warm-up** and the covariance is then frozen. Perpetual adaptation
can still be ergodic (Haario et al. prove it for their algorithm), but the
retained chain would no longer be a time-homogeneous Markov chain, and the
diagnostics below assume one. Freezing gives the guarantee for free.

On the correlated Gaussian above (4 chains, 600 warm-up, 2000 kept draws):

| Coordinate | Mean (truth) | MCSE | ESS | split R-hat | rank R-hat |
|---|---:|---:|---:|---:|---:|
| `x0` | 0.9953 (1) | 0.0320 | 1030.5 | 1.0048 | 1.0048 |
| `x1` | -1.9792 (-2) | 0.0437 | 1070.6 | 1.0033 | 1.0079 |

The acceptance rate was `0.343`. Both errors are well inside two MCSEs.

### Sampling in the prior's standard-normal space

`sample_posterior` does not sample `theta` directly. It samples `u = T(theta)`,
where `T` is the Nataf map of the prior (EQ-MOD-NATAF-001). Because `T`
pushes the prior forward to `N(0, I)`,
`pi(T^-1(u)) |det dT^-1/du| = phi(u)`, so the posterior in `u` is
`L(T^-1(u)) phi(u)` ([EQ-CAL-STD-001]). No Jacobian determinant has to be
coded: the transformation absorbs it. A test checks this numerically. At
three random points, the physical log posterior plus the log Jacobian, minus
the standard-space log posterior, equals `-log(2 pi)` (the dropped constant
for `d = 2`) to eight decimals.

The practical benefit is conditioning. Under `Uniform(1, 10)` and log-normal
priors, `A` is bounded and `k` positive and skewed. In `u` both are
unbounded and roughly isotropic, which is the geometry a Gaussian random walk
handles best. The sampler starts from the Laplace approximation in `u`, draws
over-dispersed chain starts (twice the Laplace standard deviations) so that
R-hat can detect non-convergence, and maps every draw back to
`(theta, sigma)`.

## 7. Convergence diagnostics

- **Split R-hat** ([EQ-MCMC-RHAT-001]) compares between- and within-chain
  variance after splitting each chain in half.
- **Rank-normalized R-hat** ([EQ-MCMC-RHAT-002]) repeats this on normal
  scores of ranks (bulk) and of folded draws `|x - median|` (tail).
- **ESS** ([EQ-MCMC-ESS-001]) uses Geyer's initial monotone sequence to sum
  autocorrelations without the noisy tail.
- **MCSE** ([EQ-MCMC-MCSE-001]) `= sd / sqrt(ESS)` is the simulation error of a
  posterior mean, not the posterior uncertainty.

Worked numbers:

- Two random-walk chains on a mixture with modes at `+-5`, started in
  different modes with proposal standard deviation `0.5`, never cross. Split
  R-hat is `6.137` and rank R-hat `1.836`, both far above `1.1`.
- Chains `N(0, 1)` and `N(0, 3^2)` have the same mean. Split R-hat is `1.0002`
  and misses the difference; the folded rank R-hat is `1.1716` and flags it.
- Four AR(1) chains, `phi = 0.5`, 5000 draws each: the estimated ESS is
  `6831.2` against the theoretical `4 x 5000 x (1 - phi)/(1 + phi) = 6666.7`.
  The MCSE of the mean is `0.01212` against `1/sqrt(6666.7) = 0.01225`.

R-hat near one is necessary, not sufficient: chains that all miss the same
mode agree with each other.

## 8. Worked example: exponential decay

Same data as section 4, four chains of 800 draws after 400 warm-up
iterations. That is 4804 log-posterior evaluations, each running the model at
21 conditions.

| Parameter | Posterior mean | sd | 5% | 50% | 95% | ESS | rank R-hat |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A` | 5.0073 | 0.0710 | 4.8916 | 5.0037 | 5.1253 | 408 | 1.0175 |
| `k` | 0.3011 | 0.0065 | 0.2907 | 0.3010 | 0.3121 | 445 | 1.0059 |

The posterior standard deviations (`0.0710`, `0.0065`) are close to the
least-squares standard errors (`0.0680`, `0.0064`). That is expected here:
the data dominate a weak prior, and the model is nearly linear over the
posterior region.

**Unknown noise.** Replace the known noise with
`standard_deviation_prior = LogNormal(log 0.3, 1)`. Least-squares
reweighting estimates `sigma = 0.0892`. The posterior (four chains, 800 draws
after 500 warm-up) gives a mean of `sigma` of `0.0981` and a 90% interval of
`(0.0729, 0.1322)`, which covers the true `0.1`. `A` and `k` are essentially
unchanged (`5.0028`, `0.3009`), and their intervals widen slightly because the
noise level is now uncertain.

## 9. Prediction: closing the loop

`predict` pushes posterior draws through the forward model under new
conditions ([EQ-CAL-PRED-001]). Latent predictions carry parameter
uncertainty only. With `include_noise=True` one noise draw is added per
parameter draw: Gaussian or Student-t as declared, with the draw's own `sigma`
when it was calibrated and with the discrepancy variance included.

Using 1600 posterior draws:

| `t` | True mean | Latent 95% interval | Observation 95% interval |
|---:|---:|---|---|
| 12 (extrapolation) | 0.1366 | (0.1172, 0.1549) | (-0.0556, 0.3270) |
| 2 | 2.7441 | (2.6820, 2.8034) | (2.5359, 2.9375) |

At `t = 12` the latent variance is `8.86e-5` and the observation variance is
`9.98e-3`, close to `8.86e-5 + 0.1^2`. A future measurement is dominated by
noise, while the mean response is pinned down well. Reporting only one of
the two would mislead.

`t = 12` lies outside the calibration window `[0, 10]`. The computation is
valid as conditional mathematics, but the data cannot confirm that the model
form still holds there.

## 10. Linear regression with unknown noise variance

For `y = X beta + e`, `e ~ N(0, sigma^2 I)` with `sigma^2` unknown, the
Normal-inverse-gamma family is conjugate ([EQ-BAYES-NIG-001]). Integrating out
`sigma^2` gives Student-t marginals and predictions ([EQ-BAYES-NIG-002],
[EQ-BAYES-NIG-003], [EQ-DIST-T-001]).

Hand example: estimate a mean (`X` a column of ones) from `y = (1, 2, 3)`
with prior `m_0 = 0`, `V_0 = 1`, `a_0 = 2`, `b_0 = 2`:

- `V_n = 1/(1 + 3) = 0.25`, `m_n = 0.25 x 6 = 1.5`, `a_n = 2 + 3/2 = 3.5`;
- `b_n = 2 + ((0.5^2 + 0.5^2 + 1.5^2) + 1.5^2)/2 = 4.5`;
- `E[sigma^2] = 4.5/2.5 = 1.8`.

The library returns exactly these values. The 95% interval for the mean is
`(0.1594, 2.8406)` from `t_7(1.5, (4.5/3.5) 0.25)`. The interval for `sigma^2`
is `(0.5621, 5.3259)`, and the 95% predictive interval for a new observation
is `(-1.4977, 4.4977)`. Conditional on any fixed `sigma^2`, the coefficient
law equals `condition_linear_gaussian` with prior covariance `sigma^2 V_0`,
which the tests check to ten decimals.

## 11. Boundaries

- Random-walk samplers scale poorly beyond roughly ten to twenty parameters
  and with strongly curved posteriors. Gradient-based samplers (HMC/NUTS) are
  not implemented.
- Adaptation stops at the end of warm-up. A poor warm-up gives a valid but
  inefficient chain; check ESS.
- Multimodal posteriors can defeat every diagnostic when no chain starts near
  a mode. Use over-dispersed starts and domain knowledge.
- Noise is independent across observations. Correlated errors and
  input-dependent discrepancy processes (Kennedy-O'Hagan Gaussian processes)
  are not implemented.
- The Fisher diagnostic is local and assumes Gaussian errors.
- The Laplace approximation is computed in physical coordinates. Bounded
  priors with modes on the boundary make it unreliable; prefer
  `sample_posterior`.
