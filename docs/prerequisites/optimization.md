# Optimization Prerequisite

Calibration finds parameters that minimize a misfit, reliability analysis
finds the most probable failure point, and quantiles are roots of `F(x) - p`.
This guide covers the solvers in `probstat.optimization`: finite-difference
derivatives, BFGS, Nelder-Mead, Levenberg-Marquardt least squares with its
Gauss-Newton covariance, and Brent's bracketing root finder. Each returns an
evaluation count and a convergence flag rather than silently handing back the
last iterate.

Every snippet below was executed; the block after it is its exact output, and
`tests/test_prerequisite_guides.py` re-runs all of them. Snippets share one
session, top to bottom.

Background reading: [REF-NOCEDAL-WRIGHT-2006, Chapters 2, 3, 6, 8, 10],
[REF-MORE-1978], and [REF-BRENT-1973, Chapter 4].

## 1. Finite differences: truncation versus rounding

The central difference `(f(x+h) - f(x-h)) / 2h` [EQ-OPT-FD-001] has
truncation error `O(h^2)` and rounding error `O(eps/h)`. The best step is near
`eps^(1/3) ~ 6e-6`, which is why the default relative step is `1e-6`.

```python
import math
from probstat.optimization import central_gradient, central_hessian

# Only +, -, *, / so every platform rounds identically.
f = lambda x: x[0] * x[0] * x[0] * x[1] + x[1] * x[1] / x[0]
point = (1.5, 2.0)
exact = (3 * 1.5 * 1.5 * 2.0 - 4.0 / 2.25, 1.5 ** 3 + 2 * 2.0 / 1.5)
for step in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
    gradient, calls = central_gradient(f, point, relative_step=step)
    error = max(abs(g - e) for g, e in zip(gradient, exact))
    print(f"h={step:.0e} error={error:.1e} calls={calls}")
hessian, calls = central_hessian(f, point)
print(tuple(tuple(round(v, 4) for v in row) for row in hessian), calls)
```

```text
h=1e-02 error=2.7e-04 calls=4
h=1e-04 error=2.7e-08 calls=4
h=1e-06 error=2.9e-10 calls=4
h=1e-08 error=9.6e-08 calls=4
h=1e-10 error=2.0e-06 calls=4
((20.3704, 4.9722), (4.9722, 1.3333)) 9
```

The error first falls like `h^2` and then rises like `eps/h` once rounding
dominates. The analytic Hessian is
`((6 x1 x2 + 2 x2^2/x1^3, 3 x1^2 - 2 x2/x1^2), (., 2/x1))`, which at
`(1.5, 2)` equals `((20.3704, 4.9722), (4.9722, 1.3333))` [EQ-OPT-FD-002].

## 2. BFGS on the Rosenbrock valley

BFGS builds an inverse-Hessian approximation from gradient differences
[EQ-OPT-BFGS-001] and accepts steps by the Armijo rule [EQ-OPT-ARMIJO-001].
Without an analytic gradient it uses central differences, which costs more
evaluations.

```python
from probstat.optimization import minimize_bfgs, minimize_nelder_mead

def rosenbrock(x):
    a, b = 1.0 - x[0], x[1] - x[0] * x[0]
    return a * a + 100.0 * b * b

def rosenbrock_gradient(x):
    b = x[1] - x[0] * x[0]
    return (-2.0 * (1.0 - x[0]) - 400.0 * x[0] * b, 200.0 * b)

analytic = minimize_bfgs(rosenbrock, (-1.2, 1.0), gradient=rosenbrock_gradient)
numeric = minimize_bfgs(rosenbrock, (-1.2, 1.0))
for result in (analytic, numeric):
    print(tuple(round(v, 6) for v in result.x), result.converged, result.evaluations)
simplex = minimize_nelder_mead(rosenbrock, (-1.2, 1.0))
print(tuple(round(v, 4) for v in simplex.x), simplex.converged, simplex.evaluations)
```

```text
(1.0, 1.0) True 91
(1.0, 1.0) True 199
(1.0, 1.0) True 218
```

Nelder-Mead [EQ-OPT-NM-001] needs no derivatives but many more evaluations;
it is the right tool for noisy or non-smooth objectives, not for smooth ones.

## 3. Levenberg-Marquardt least squares

`least_squares` minimizes `0.5 ||r(theta)||^2` with a damped Gauss-Newton step
`(J'J + lambda D) delta = -J'r` [EQ-OPT-LM-001]. At the solution,
`gauss_newton_covariance` returns `s^2 (J'J)^-1` [EQ-OPT-LM-002], the
asymptotic parameter covariance under local linearity.

The data below are generated from `y = 2.5 exp(-0.8 t)` plus a fixed,
deterministic perturbation so the example is reproducible without a seed.

```python
from probstat.optimization import least_squares, gauss_newton_covariance

times = tuple(0.5 * k for k in range(12))
wiggle = (0.03, -0.02, 0.01, 0.02, -0.03, 0.0, 0.01, -0.01, 0.02, -0.02, 0.01, 0.0)
observed = tuple(2.5 * math.exp(-0.8 * t) + w for t, w in zip(times, wiggle))

def residual(theta):
    amplitude, rate = theta
    return tuple(amplitude * math.exp(-rate * t) - y for t, y in zip(times, observed))

fit = least_squares(residual, (1.0, 0.1))
covariance = gauss_newton_covariance(fit)
print(tuple(round(v, 4) for v in fit.x), fit.converged, fit.message)
print(f"cost={fit.cost:.3e}")
print(tuple(round(math.sqrt(covariance[i][i]), 4) for i in range(2)))
```

```text
(2.5193, 0.8071) True relative reduction or step tolerance met
cost=1.663e-03
(0.0163, 0.009)
```

The standard errors use the residual variance estimate `||r||^2 / (m - n)`.
They describe local curvature, not a full posterior; see
[nonlinear calibration and MCMC](../foundations/nonlinear-calibration-and-mcmc.md)
for the Bayesian treatment.

## 4. Brent's method for scalar roots

`brent_root` combines bisection, secant, and inverse quadratic interpolation
inside a sign-changing bracket [EQ-OPT-BRENT-001]. A typical use is inverting
a CDF: the 97.5 % standard normal quantile solves `Phi(x) - 0.975 = 0`.

```python
from probstat.optimization import brent_root
from probstat.distributions import Normal

standard = Normal()
root = brent_root(lambda x: standard.cdf(x) - 0.975, 0.0, 5.0)
print(f"{root.root:.10f} {standard.quantile(0.975):.10f}", root.converged)
try:
    brent_root(lambda x: x * x + 1.0, -1.0, 1.0)
except ValueError as error:
    print(error)
```

```text
1.9599639845 1.9599639845 True
root bracket must change sign
```

## Choosing a solver

| Problem | Solver | Notes |
|---|---|---|
| Smooth scalar objective | `minimize_bfgs` | supply `gradient=` when cheap |
| Non-smooth or noisy objective | `minimize_nelder_mead` | derivative-free, slow in high dimension |
| Sum of squared residuals | `least_squares` | returns the Jacobian for `gauss_newton_covariance` |
| Scalar root with a bracket | `brent_root` | guaranteed convergence once the bracket changes sign |
| Derivatives of a black box | `central_gradient`, `central_jacobian`, `central_hessian` | `2n` (gradient) or `1 + 2n^2` (Hessian) calls |

Next: [multivariable calculus and conditioning](multivariable-calculus-and-conditioning.md).
