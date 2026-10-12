# Numerical Integration Prerequisite

Moments, entropies, divergences, scoring rules, and marginal likelihoods are
integrals, and most have no closed form. `probstat.numerics` offers three
families of rules:

- `composite_trapezoid` and `composite_simpson`: fixed grids on a finite
  range, with no error estimate;
- `gauss_kronrod`: globally adaptive G7-K15 bisection with QUADPACK's error
  estimate [EQ-NUM-GK15-001], [EQ-NUM-QAG-001];
- `tanh_sinh_interval`, `exp_sinh`, `sinh_sinh`: double-exponential rules
  for finite ranges with end singularities, half-lines, and the whole line
  [EQ-NUM-DE-001].

`integrate(f, a, b)` picks the rule from the limits. Every adaptive rule
returns an `IntegrationResult` with the estimate, an error estimate, the
number of integrand evaluations, the final panels, and a `converged` flag
that is `False` whenever a limit stopped the run.

Every snippet below was executed; the block after it is its exact output, and
`tests/test_prerequisite_guides.py` re-runs all of them. Snippets share one
session, top to bottom. Errors are printed as bounds rather than as digits,
because the last bit of `exp` and `log` can differ between C libraries.

Background reading: [REF-DAVIS-RABINOWITZ-1984, Chapters 2, 5, 6],
[REF-PIESSENS-QUADPACK-1983, Chapter 2], [REF-TAKAHASI-MORI-1974], and
[REF-MORI-SUGIHARA-2001].

## 1. Why fixed grids are not enough

Simpson's rule has error `O(h^4)` only for smooth integrands. For `sqrt(x)`,
whose derivative is unbounded at zero, the error falls only like `h^1.5`
(sixteen times more nodes buy a factor of 64, not 65536), and an integrable
singularity such as `x^-1/2` cannot be used at all because Simpson evaluates
the node `x = 0`. Tanh-sinh never evaluates the ends and clusters its nodes
there.

```python
import math
from probstat.numerics import composite_simpson, gauss_kronrod, integrate

f = lambda x: math.sqrt(x)  # derivative unbounded at 0; exact integral 2/3
for n in (16, 256, 4096):
    s = composite_simpson(f, 0.0, 1.0, intervals=n)
    print(f"Simpson n={n:5d} error={abs(s.value - 2 / 3):.1e}")
result = integrate(f, 0.0, 1.0)
print(result.rule, result.evaluations, abs(result.value - 2 / 3) < 1e-15)
```

```text
Simpson n=   16 error=1.3e-03
Simpson n=  256 error=2.0e-05
Simpson n= 4096 error=3.1e-07
tanh-sinh 97 True
```

## 2. Adaptive Gauss-Kronrod

One G7-K15 panel uses 15 evaluations; the seven Gauss nodes are a subset, so
`|K15 - G7|` comes for free. QUADPACK inflates it to
`I_asc min(1, (200 |K15 - G7| / I_asc)^1.5)` and floors it at
`50 eps int |f|`, which makes the estimate deliberately pessimistic. The
panel with the largest local error is bisected until the summed error meets
`max(abs_tol, rel_tol |I|)`. The constants are QUADPACK's 33-digit values;
the test suite checks in exact rational arithmetic that K15 integrates
`x^0 ... x^22` and G7 integrates `x^0 ... x^13` on `[-1, 1]` with residuals
below `1e-32`.

A Lorentzian of half-width `1e-3` at `x = 0.3` shows the panels piling up at
the peak:

```python
peak = lambda x: 1.0 / ((x - 0.3) ** 2 + 1e-6)
exact = 1e3 * (math.atan(0.7e3) + math.atan(0.3e3))
result = gauss_kronrod(peak, 0.0, 1.0)
widths = sorted(right - left for left, right in result.intervals)
print(result.converged, len(result.intervals), result.evaluations)
print(f"narrowest {widths[0]:.2e} widest {widths[-1]:.2e}")
print(sum(1 for left, right in result.intervals if right > 0.29 and left < 0.31))
error = abs(result.value - exact)
print(error <= result.error_estimate, f"estimate {result.error_estimate:.0e}")
capped = gauss_kronrod(peak, 0.0, 1.0, limit=5)
print(capped.converged, f"{capped.error_estimate:.1e}",
      f"{abs(capped.value - exact):.1e}")
```

```text
True 23 675
narrowest 9.77e-04 widest 2.50e-01
10
True estimate 3e-07
False 8.0e+03 1.9e+03
```

With `limit=5` the run stops after four bisections and says so; its error
estimate still exceeds the true error.

## 3. Double-exponential rules

The substitution `x = phi(t)` makes the integrand decay like
`exp(-c exp|t|)`, after which the trapezoidal rule converges geometrically in
the number of nodes. Halving the step reuses all previous nodes, and the
difference of the last two levels is the error estimate. The node maps are

| Range | Rule | `phi(t)` |
|---|---|---|
| `[a, b]` | tanh-sinh | `a + (b - a) / (1 + exp(-pi sinh t))` |
| `[a, inf)` | exp-sinh | `a + exp((pi/2) sinh t)` |
| `(-inf, inf)` | sinh-sinh | `sinh((pi/2) sinh t)` |

```python
cases = [
    ("x^-1/2 on [0,1]", lambda x: x ** -0.5, 0.0, 1.0, 2.0),
    ("log x on [0,1]", math.log, 0.0, 1.0, -1.0),
    ("exp(-x) on [0,inf)", lambda x: math.exp(-x), 0.0, math.inf, 1.0),
    ("x^2 exp(-x) on [0,inf)", lambda x: x * x * math.exp(-x), 0.0, math.inf, 2.0),
    ("exp(-x^2/2) on R", lambda x: math.exp(-0.5 * x * x),
     -math.inf, math.inf, math.sqrt(2 * math.pi)),
    ("1/(1+x^2) on [0,inf)", lambda x: 1 / (1 + x * x), 0.0, math.inf, math.pi / 2),
    ("x^-1/2 exp(-x) on [0,inf)", lambda x: math.exp(-x) / math.sqrt(x),
     0.0, math.inf, math.sqrt(math.pi)),
]
for name, g, a, b, exact in cases:
    de = integrate(g, a, b)
    gk = integrate(g, a, b, method="gauss-kronrod")
    print(f"{name:26s} {de.rule:9s} n={de.evaluations:4d} "
          f"ok={abs(de.value - exact) <= 4 * math.ulp(exact)} | "
          f"GK n={gk.evaluations:4d} conv={gk.converged} "
          f"bounded={abs(gk.value - exact) <= gk.error_estimate}")
```

```text
x^-1/2 on [0,1]            tanh-sinh n=  97 ok=True | GK n=1965 conv=True bounded=True
log x on [0,1]             tanh-sinh n=  97 ok=True | GK n=1005 conv=True bounded=True
exp(-x) on [0,inf)         exp-sinh  n= 433 ok=True | GK n= 135 conv=True bounded=True
x^2 exp(-x) on [0,inf)     exp-sinh  n= 433 ok=True | GK n= 165 conv=True bounded=True
exp(-x^2/2) on R           sinh-sinh n= 433 ok=True | GK n= 330 conv=True bounded=True
1/(1+x^2) on [0,inf)       exp-sinh  n= 109 ok=True | GK n=  75 conv=True bounded=True
x^-1/2 exp(-x) on [0,inf)  exp-sinh  n= 217 ok=True | GK n=1425 conv=False bounded=True
```

The double-exponential rules reach the correctly rounded value (within four
units in the last place) in 97 to 433 evaluations. Adaptive Gauss-Kronrod
needs 1965 evaluations for `x^-1/2` because, without QUADPACK's epsilon
extrapolation, each bisection next to the singularity only shrinks its error
by `sqrt(2)`; on `x^-1/2 exp(-x)` the QAGI map `x = (1 - t)/t` moves the
singularity to `t = 1`, where panels stop at the bisection floor and the run
honestly reports `converged=False` with error `7e-9`.

In every converged case the reported estimate exceeded the true error. The
estimate-to-error ratios measured for `tests/test_adaptive_quadrature.py`
ranged from 20 (Gauss-Kronrod, `x^-1/2`) to about `6e5`, and many results
were exact to the last bit, so the ratio was infinite: the estimates are
safe, not sharp.

## 4. Where each rule fails

- **Interior kinks or peaks** break the analyticity that tanh-sinh needs.
  `integrate` with the default `method="auto"` then also runs Gauss-Kronrod
  and returns the result whose error estimate is the smaller multiple of its
  own target `max(abs_tol, rel_tol |value|)`. If neither converges and the
  two values differ by more than their error estimates, the
  double-exponential value is kept: Gauss-Kronrod can miss all of the mass
  of a narrow peak far from the origin (`exp(-(x - 50)^2)` on the line gives
  `3e-198` with an error estimate of the same size).
- **Mass beyond the outermost nodes** (`x^-1.025` at infinity, `x^-0.97` at
  zero) is estimated from the weighted integrand at the last node on each
  side and its decay rate; while it is not negligible the run reports
  `converged=False`.
- **Singularities at a non-zero end** `c`: the integrand only sees the
  rounded node, so the mass closer to `c` than `ulp(c)` is lost (about
  `2 sqrt(ulp(c))` for `(x - c)^-1/2`). A node-rounding term in the error
  estimate catches this and the run reports `converged=False`; substitute
  `y = x - c` so the singularity sits at zero.
- **Slowly decaying oscillation** such as `sin(x)/x` on `[0, inf)` is not
  absolutely integrable at the rate the rules need; there is no QAWO-type
  oscillatory rule, and both methods report `converged=False`.

```python
kink = integrate(lambda x: abs(x - 0.3), 0.0, 1.0)
print(kink.rule, abs(kink.value - 0.29) < 1e-12, kink.converged)
upper = integrate(lambda x: (1 - x) ** -0.5, 0.0, 1.0, method="double-exponential")
error = abs(upper.value - 2)
print(upper.converged, f"error {error:.0e}", error <= upper.error_estimate)
moved = integrate(lambda y: y ** -0.5, 0.0, 1.0)
print(moved.converged, abs(moved.value - 2) < 1e-15)
sinc = integrate(lambda x: math.sin(x) / x if x else 1.0, 0.0, math.inf)
print(sinc.converged)
```

```text
gauss-kronrod-15 True True
False error 2e-08 True
True True
False
```

## 5. Evaluation policy

A NaN, infinity, or exception from the integrand raises inside the range
that carries weight. The double-exponential rules tolerate failures only at
nodes that cannot matter: within `1e-15` (relative) of a finite end, rounded
onto that end, or beyond `|x| = 1e15` on an infinite range. That is why
`x * x * exp(-x)` integrates although it is `inf * 0 = nan` at `x = 1e300`,
while `exp(-x) cosh(x/2)` raises at `x ~ 710`, where `cosh` overflows but the
true integrand is still representable.

```python
from probstat.numerics import exp_sinh

tail = exp_sinh(lambda x: x * x * math.exp(-x), 0.0, math.inf)
print(tail.converged, abs(tail.value - 2) < 1e-15)
try:
    exp_sinh(lambda x: math.exp(-x) * math.cosh(0.5 * x), 0.0, math.inf)
except OverflowError as error:
    print("OverflowError:", error)
```

```text
True True
OverflowError: math range error
```

## Choosing a rule

| Integrand | Call | Notes |
|---|---|---|
| Smooth on a finite range | `integrate(f, a, b)` | tanh-sinh, about 100 evaluations |
| End singularity at 0 | `integrate(f, 0, b)` | move other singular ends to 0 first |
| Interior kink, peak, or discontinuity | `gauss_kronrod(f, a, b)` | or split at the point |
| Half-line or whole line | `integrate(f, a, math.inf)` | needs decay faster than `1/x` |
| Zero-valued integral | pass `abs_tol` | a relative target cannot certify zero |
| Quantile-scale integral on `(0, 1)` | `tanh_sinh(g, 0, 1)` | `g(u, 1 - u)` avoids cancellation |

Next: [multivariable calculus and conditioning](multivariable-calculus-and-conditioning.md).
