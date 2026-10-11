# Information Theory

`probstat.information` (layer 2) measures uncertainty and dependence. It has
three parts:

- Shannon quantities for finite PMFs;
- closed-form differential entropies and Gaussian divergences;
- nearest-neighbour estimators that work directly from samples.

Every function takes `unit="nats"` (natural log, the default) or
`unit="bits"` (base 2). One bit is `log 2 = 0.6931` nats. Every number below
was computed with the library, and the identities are checked in
`tests/test_information_theory.py`.

## 1. Entropy of a finite PMF (EQ-INFO-H-001)

`entropy(p) = -sum p log p`, with `0 log 0 = 0`. A PMF must sum to one
within `1e-9`; it is not silently renormalized.

- `entropy((0.5, 0.25, 0.25), unit="bits")` is exactly `1.5` bits. This is
  the average length of the optimal prefix code `0, 10, 11`.
- A uniform PMF on `n` points has the maximum, `log n`. A point mass has 0.

## 2. Joint, conditional, and mutual information (EQ-INFO-H-002, EQ-INFO-MI-001)

A joint PMF is a table with rows indexing `X` and columns indexing `Y`.
Worked example:

| | `y1` | `y2` | `y3` | `p(x)` |
|---|---|---|---|---|
| `x1` | 0.30 | 0.10 | 0.05 | 0.45 |
| `x2` | 0.05 | 0.20 | 0.30 | 0.55 |
| `p(y)` | 0.35 | 0.30 | 0.35 | |

| Quantity | Call | nats | bits |
|---|---|---|---|
| `H(X)` | `entropy(px)` | 0.68814 | 0.99277 |
| `H(Y)` | `entropy(py)` | 1.09607 | 1.58129 |
| `H(X, Y)` | `joint_entropy(J)` | 1.57410 | 2.27095 |
| `H(Y \| X)` | `conditional_entropy(J)` | 0.88596 | 1.27818 |
| `I(X; Y)` | `mutual_information(J)` | 0.21010 | 0.30311 |

The tests confirm three identities to `1e-14`:

- the chain rule `H(X, Y) = H(X) + H(Y | X)`, here
  `1.57410 = 0.68814 + 0.88596`;
- `I = H(X) + H(Y) - H(X, Y)`;
- symmetry of `I` under transposing the table.

For a product table `p(x) p(y)`, `I = 0` and `H(Y | X) = H(Y)`. Mutual
information is never negative. A rounding-level negative sum, which can
occur for an exact product table, is returned as `0`.

## 3. Divergences between PMFs (EQ-INFO-KL-001, EQ-INFO-JS-001)

With `p = (0.5, 0.5)` and `q = (0.9, 0.1)`:

| Quantity | Value (nats) |
|---|---|
| `kl_divergence(p, q)` | 0.51083 |
| `kl_divergence(q, p)` | 0.36806 |
| `cross_entropy(p, q) = H(p) + D(p \|\| q)` | 1.20397 |
| `jensen_shannon_divergence(p, q)` | 0.10175 (0.14679 bits) |

- KL is not symmetric. It is infinite when `q` puts zero mass where `p`
  does not.
- Jensen-Shannon compares both laws to their average. It is symmetric, always
  finite, and at most one bit: `JS((1, 0), (0, 1)) = 1` bit exactly.

## 4. Differential entropy (EQ-INFO-DH-001..005)

For a density, `h(X) = -E[log f(X)]`. Unlike the discrete case, it can be
negative, and it changes under a change of variables.

| Law | Closed form | Value (nats) |
|---|---|---|
| `Normal(mu, 0.7)` | `log(sigma sqrt(2 pi e))` | 1.06226 |
| `Uniform(-1, 0.5)` | `log(b - a)` | 0.40547 |
| `Uniform(0, 0.5)` | `log(b - a)` | -0.69315 |
| `Exponential(2)` | `1 - log lambda` | 0.30685 |
| `Gamma(2.5, rate 1.5)` | `k - log lambda + log Gamma(k) + (1-k) psi(k)` | 1.32448 |
| `Beta(2, 3)` | `log B - (a-1) psi(a) - (b-1) psi(b) + (a+b-2) psi(a+b)` | -0.23491 |
| `LogNormal(0.3, 0.7)` | `mu + log(sigma sqrt(2 pi e))` | 1.36226 |
| `GEV(sigma = 2, xi = -0.2)` | `log sigma + gamma_E (1 + xi) + 1` | 2.15492 |
| `Gumbel(beta = 1.5)` | `log beta + gamma_E + 1` | 1.98268 |
| `GPD(sigma = 1.3, xi = 0.3)` | `log sigma + xi + 1` | 1.56236 |
| `N_2(Sigma = [[1, .5], [.5, 2]])` | `(d log(2 pi e) + log det Sigma)/2` | 3.11768 |

`differential_entropy(law)` accepts the distribution objects listed above.
Each closed form agrees with Simpson integration of `-f log f` to `2e-6`
(`1e-5` for the heavy-tailed GPD). The bivariate normal agrees with a
`0.04`-step grid to `1e-6`.

**Design note.** These are functions in `information`, not new `entropy`
properties on the existing distribution classes. That keeps the change
non-invasive: no existing distribution module was edited. The one exception
is the new `GeneralizedPareto`, which carries an `entropy` property. ADR-0036
records the decision.

## 5. KL divergence between Gaussians (EQ-INFO-KLG-001/002)

```text
D(N(m0, s0^2) || N(m1, s1^2)) = log(s1/s0) + (s0^2 + (m0 - m1)^2)/(2 s1^2) - 1/2
D(N(m0, S0) || N(m1, S1)) = (tr(S1^-1 S0) + (m1-m0)' S1^-1 (m1-m0) - d
                             + log det S1 - log det S0)/2
```

- `kl_normal(0, 1, 1, 2) = log 2 + 2/8 - 1/2 = 0.44315`.
- `kl_normal(2, 0.5, -1, 1.5) = 2.65417`.
- In 2-D, with `m0 = (0, 0)`, `S0 = [[1, .3], [.3, 1]]`, `m1 = (1, -1)` and
  `S1 = [[2, -.4], [-.4, .8]]`: `D(0 || 1) = 0.97948` but
  `D(1 || 0) = 1.86942`.

These match Simpson and grid integration of `p log(p/q)` to `1e-9` (1-D)
and `1e-6` (2-D). With diagonal covariances the multivariate formula reduces
to the sum of univariate terms. The code uses Cholesky solves, not explicit
inverses.

## 6. Estimating entropy and mutual information from samples

When only samples are available, nearest-neighbour distances act as local
density estimates.

**Kozachenko-Leonenko (EQ-INFO-KLE-001).**
`kozachenko_leonenko_entropy(samples, k=3, metric="euclidean")` computes

```text
h = psi(n) - psi(k) + log V_d + (d/n) sum log r_i
```

where `r_i` is the distance from point `i` to its `k`-th neighbour.

**KSG (EQ-INFO-KSG-001).** `ksg_mutual_information(x, y, k=3)` is
algorithm 1 of Kraskov, Stögbauer and Grassberger (2004). It finds the
max-norm radius of the `k`-th joint neighbour, then counts neighbours inside
that radius in each marginal:

```text
I = psi(k) + psi(n) - mean[psi(n_x + 1) + psi(n_y + 1)]
```

Seeded checks against Gaussian closed forms:

| Case | n | Estimate | Closed form |
|---|---|---|---|
| KL entropy, `N(0, 2^2)` | 1000 | 2.0478 | 2.1121 |
| KL entropy, 2-D `rho = 0.6`, `k = 4`, Euclidean | 800 | 2.5559 | 2.6147 |
| KL entropy, same, max norm | 800 | 2.5582 | 2.6147 |
| KSG, `rho = 0.6` | 800 | 0.2294 | `-log(1 - rho^2)/2 = 0.2231` |
| KSG, independent | 800 | -0.0063 | 0 |

Over eight seeds at `n = 400`, both estimators had a mean error below 0.01
nats and a standard deviation of about 0.045 nats. Scaled to the test sample
sizes, the test tolerances (0.12 for entropy, 0.10 and 0.06 for KSG) are 2-4
standard deviations. The seeded estimates above sit well inside them.

**Practical points.**

- The KSG estimate can be slightly negative under independence. It is
  returned as computed, not clipped.
- Duplicated points give a zero radius and raise `ValueError`. These
  estimators are for continuous data; add tiny jitter to discretized data
  only if you understand the bias that adds.
- Both use an exhaustive `O(n^2 d)` neighbour search in pure Python. That
  takes about 0.15 s for `n = 1000` in 1-D on the development machine. A
  k-d tree would be needed for much larger samples.

## 7. Limitations

- No plug-in or bias-corrected (Miller-Madow) entropy estimators for
  discrete *samples*. The discrete functions take PMFs, not counts.
- No Rényi or Tsallis entropies, no f-divergences beyond KL and JS, and no
  transfer entropy.
- Differential entropies are provided for the families listed in section 4
  only. `differential_entropy` raises `TypeError` for any other family, for
  example `Weibull`.
