# Multivariable Calculus, Jacobians, and Conditional Distributions

Three pieces of multivariable calculus carry most of probabilistic modeling:

1. the Jacobian of a vector map, which linearizes it;
2. the change-of-variables rule, which moves a density through a smooth
   bijection by the absolute Jacobian determinant of the inverse; and
3. the delta method, which moves a mean and covariance through a smooth
   map by its Jacobian.

The fourth topic, conditioning a Gaussian vector, is where linear algebra
meets probability: the conditional law is again Gaussian, with a mean given by
a regression and a covariance given by a Schur complement.

Every snippet below was executed; the block after it is its exact output, and
`tests/test_prerequisite_guides.py` re-runs all of them. Snippets share one
session, top to bottom. Unit tests for the same code live in
`tests/test_prerequisite_bridges.py`.

Background reading: [REF-CASELLA-BERGER-2002, Sections 2.1, 4.3, 4.6, and
5.5.4], [REF-MAGNUS-NEUDECKER-2019], and [REF-BISHOP-2006, Section 2.3].

## 1. Jacobians

For `g: R^n -> R^m`, the Jacobian `J_ij = d g_i / d x_j` is the best linear
approximation `g(x + d) ~= g(x) + J d`. `optimization.central_jacobian`
approximates it with `2n` evaluations [EQ-OPT-FD-001].

```python
import math
from probstat.optimization import central_jacobian
from probstat.linalg import determinant

def polar_to_cartesian(y):
    r, theta = y
    return (r * math.cos(theta), r * math.sin(theta))

r, theta = 2.0, 0.6
J, calls = central_jacobian(polar_to_cartesian, (r, theta))
exact = ((math.cos(theta), -r * math.sin(theta)), (math.sin(theta), r * math.cos(theta)))
print(tuple(tuple(round(v, 6) for v in row) for row in J), calls)
print(max(abs(J[i][j] - exact[i][j]) for i in range(2) for j in range(2)) < 1e-8)
print(f"det J = {determinant(J):.6f}  (= r)")
```

```text
((0.825336, -1.129285), (0.564642, 1.650671)) 4
True
det J = 2.000000  (= r)
```

The determinant `r` is the familiar area element `dx1 dx2 = r dr dtheta`.

## 2. Change of variables

If `Y = g(X)` with `g` a smooth bijection, then [EQ-MVN-JAC-001]

`log p_Y(y) = log p_X(g^-1(y)) + log |det J_(g^-1)(y)|`.

`multivariate.transformed_density(log_pdf, inverse_map, jacobian_of_inverse)`
returns a callable that evaluates exactly this. Omit the Jacobian to use
central differences; the determinant comes from a partial-pivot LU
factorization (`linalg.log_abs_determinant`, [EQ-LINALG-008]). Scalars are
one-component vectors.

### Log-normal from normal

```python
from probstat.distributions import Normal, LogNormal
from probstat.multivariate import transformed_density

base = Normal(0.5, 0.8)
log_y = transformed_density(
    lambda x: base.log_pdf(x[0]),       # log p_X
    lambda y: (math.log(y[0]),),        # g^-1(y) = log y
    lambda y: ((1.0 / y[0],),),         # d g^-1 / dy = 1/y
)
numeric = transformed_density(lambda x: base.log_pdf(x[0]), lambda y: (math.log(y[0]),))
for y in (0.5, 1.0, 3.0):
    print(f"y={y}: {log_y([y]):.8f} {numeric([y]):.8f} {LogNormal(0.5, 0.8).log_pdf(y):.8f}")
```

```text
y=0.5: -1.11483545 -1.11483545 -1.11483545
y=1.0: -0.89110748 -0.89110748 -0.89110748
y=3.0: -2.07435780 -2.07435780 -2.07435780
```

### Polar coordinates of a standard normal pair

With `X ~ Normal(0, I_2)` and `Y = (R, Theta)`, the inverse map is the polar
map above and `|det J| = r`, so `p(r, theta) = r e^(-r^2/2) / (2 pi)`: a
Rayleigh radius independent of a uniform angle.

```python
from probstat.multivariate import MultivariateNormal

standard = MultivariateNormal((0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)))
polar = transformed_density(standard.log_pdf, polar_to_cartesian)
for r, theta in ((0.5, 1.0), (1.0, 3.0), (2.0, 5.0)):
    rayleigh_uniform = math.log(r) - 0.5 * r * r - math.log(2.0 * math.pi)
    print(f"{polar([r, theta]):.7f} {rayleigh_uniform:.7f}")
```

```text
-2.6560242 -2.6560242
-2.3378771 -2.3378771
-3.1447299 -3.1447299
```

### Box-Muller in reverse

Box and Muller [REF-BOX-MULLER-1958] map two independent uniforms to two
independent standard normals:
`Z = sqrt(-2 log U1) (cos 2 pi U2, sin 2 pi U2)`. Pushing the uniform density
(`log p_U = 0` on the unit square) through that map must give the bivariate
standard normal density. The inverse is
`U1 = exp(-|z|^2 / 2)`, `U2 = atan2(z2, z1) / (2 pi) mod 1`.

```python
def uniform_square(u):
    return 0.0 if 0.0 < u[0] < 1.0 and 0.0 < u[1] < 1.0 else -math.inf

def box_muller_inverse(z):
    return (math.exp(-0.5 * (z[0] ** 2 + z[1] ** 2)), (math.atan2(z[1], z[0]) / (2 * math.pi)) % 1.0)

pushed = transformed_density(uniform_square, box_muller_inverse)
for z in ((0.3, 0.8), (-1.2, 0.4), (-0.7, -1.9)):
    print(f"{pushed(z):.6f} {standard.log_pdf(z):.6f}")
```

```text
-2.202877 -2.202877
-2.637877 -2.637877
-3.887877 -3.887877
```

The finite-difference Jacobian is only valid away from the branch cut of
`atan2(...) mod 1` on the positive `z1` axis, where the inverse map jumps.

### A linear map of a Gaussian is Gaussian

For `Y = A X + b`, the rule gives `Normal(A mu + b, A Sigma A')`:

```python
from probstat.linalg import lu_solve, matvec, sandwich

A = ((1.0, 2.0), (-0.5, 1.5))
b = (1.0, -2.0)
X = MultivariateNormal((0.0, 1.0), ((1.0, 0.3), (0.3, 2.0)))
Y = MultivariateNormal(tuple(m + s for m, s in zip(matvec(A, X.mean), b)), sandwich(A, X.covariance))
linear = transformed_density(X.log_pdf, lambda y: lu_solve(A, (y[0] - b[0], y[1] - b[1])))
for y in ((0.0, 0.0), (4.0, -1.0)):
    print(f"{linear(y):.7f} {Y.log_pdf(y):.7f}")
```

```text
-5.5154158 -5.5154158
-3.6012796 -3.6012796
```

## 3. The delta method

For a smooth `g` and `X` with mean `mu` and covariance `Sigma`
[EQ-MVN-DELTA-001]:

`E[g(X)] ~= g(mu)`, `Cov[g(X)] ~= J Sigma J'`, `J = dg/dx (mu)`.

It is exact for affine maps. For curved maps it ignores the bias
`tr(H Sigma)/2`, so check it against simulation when variances are not small
relative to curvature.

```python
from probstat.multivariate import delta_method

law = MultivariateNormal((10.0, 5.0), ((0.04, 0.01), (0.01, 0.09)))
product_ratio = lambda x: (x[0] * x[1], x[0] / x[1])
approx = delta_method(product_ratio, law.mean, law.covariance)
print(tuple(round(v, 6) for v in approx.mean))
print(tuple(tuple(round(v, 6) for v in row) for row in approx.covariance))
print(tuple(round(v, 4) for v in approx.standard_errors), approx.evaluations)

draws = [product_ratio(x) for x in law.sample(20_000, seed=11)]
from probstat.statistics import mean, variance
for k, name in enumerate(("product", "ratio")):
    values = [d[k] for d in draws]
    print(f"{name}: MC mean {mean(values):.4f}  MC var {variance(values, ddof=1):.4f}")
```

```text
(50.0, 2.0)
((11.0, -0.32), (-0.32, 0.0144))
(3.3166, 0.12) 5
product: MC mean 50.0549  MC var 10.8031
ratio: MC mean 2.0061  MC var 0.0147
```

The delta variances `11.0` and `0.0144` agree with the simulated ones to
about 2 %, which is within two Monte Carlo standard errors (a sample variance
from 20 000 normal-like draws has a relative standard error near 1 %). The
simulated ratio mean exceeds `g(mu) = 2` by about `0.006`, close to the
curvature bias `mu1 s2^2 / mu2^3 - Sigma12 / mu2^2 = 0.0068` that a
first-order method cannot see.

## 4. Conditioning a Gaussian vector

Partition `X = (X_a, X_b)`. Then [EQ-DIST-MVN-006, EQ-DIST-MVN-007]

- marginal: `X_b ~ Normal(mu_b, Sigma_bb)`;
- conditional: `X_a | X_b = x_b ~ Normal(mu_a + Sigma_ab Sigma_bb^-1 (x_b - mu_b),
  Sigma_aa - Sigma_ab Sigma_bb^-1 Sigma_ba)`.

`MultivariateNormal.marginal_subset(indices)` and
`MultivariateNormal.conditional(indices, values)` implement both; the
conditional is computed from the Cholesky factor of `Sigma_bb` without an
explicit inverse, and the remaining components keep ascending index order.

```python
joint = MultivariateNormal((1.0, 2.0, 3.0), ((4.0, 1.2, 0.5), (1.2, 2.0, 0.3), (0.5, 0.3, 1.0)))
print(joint.marginal_subset([2, 0]))
given = joint.conditional([2], [3.5])
print(tuple(round(v, 6) for v in given.mean))
print(tuple(tuple(round(v, 6) for v in row) for row in given.covariance))
print(joint.conditional([2], [10.0]).covariance == given.covariance)
```

```text
MultivariateNormal(mean=(3.0, 1.0), covariance=((1.0, 0.5), (0.5, 4.0)))
(1.25, 2.15)
((3.75, 1.05), (1.05, 1.91))
True
```

The conditional covariance does not depend on the observed value: observing
`X3 = 3.5` or `X3 = 10` reduces uncertainty by the same amount.

The joint density factors as marginal times conditional, which is a strong
check on both formulas:

```python
x = (0.3, 2.5, 3.5)
chain = joint.marginal_subset([2]).log_pdf([3.5]) + given.log_pdf([0.3, 2.5])
print(f"{joint.log_pdf(x):.10f} {chain:.10f}")
```

```text
-4.0204097736 -4.0204097736
```

The conditional mean is the population regression of `X_a` on `X_b`, so a
regression fitted to seeded samples recovers it up to sampling error (the
slope's standard error here is about `sqrt(1.36 / 20000) = 0.008`):

```python
pair = MultivariateNormal((0.0, 1.0), ((2.0, 0.8), (0.8, 1.0)))
sample = pair.sample(20_000, seed=2026)
from probstat.inference import fit_simple_linear_regression
fit = fit_simple_linear_regression([s[1] for s in sample], [s[0] for s in sample])
exact = pair.conditional([1], [0.0])
print(f"slope {fit.slope:.4f} vs {0.8 / 1.0:.4f}")
print(f"intercept {fit.intercept:.4f} vs {exact.mean[0]:.4f}")
print(f"residual variance {fit.residual_variance:.4f} vs {exact.covariance[0][0]:.4f}")
```

```text
slope 0.8169 vs 0.8000
intercept -0.8319 vs -0.8000
residual variance 1.3418 vs 1.3600
```

## Where this is used

- `inference.condition_linear_gaussian` and the Kalman filters apply the same
  Schur-complement algebra to observations `y = H x + e`.
- `modeling.JointInput` and the Nataf transformation are changes of variables
  between physical and standard-normal spaces.
- `optimization.gauss_newton_covariance` is the delta method applied to a
  least-squares estimator.

Previous: [linear algebra](linear-algebra.md), [optimization](optimization.md).
Next: [probability and statistics refresher](probability-and-statistics-refresher.md).
