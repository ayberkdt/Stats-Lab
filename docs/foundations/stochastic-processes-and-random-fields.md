# Stochastic Processes and Random Fields

`probstat.processes` covers random quantities that evolve in time or vary in
space: finite Markov chains, Poisson and compound Poisson processes, Brownian
diffusions with exact samplers, discretization schemes for scalar SDEs, and
Gaussian random fields represented by Karhunen-Loeve (KL) expansions.
`probstat.modeling.RandomFieldInput` turns a truncated KL expansion into
ordinary model inputs, so a forward model driven by a random field runs
through propagation, reliability, and sensitivity like any other
`ProbabilisticModel`. All numbers below were produced by the library with the
stated seeds.

## 1. Discrete-time Markov chains

A `MarkovChain` holds a row-stochastic matrix `P` [EQ-PROC-MC-001]. Rows must
be non-negative and sum to one within `1e-10`; nothing is renormalized
silently.

**Distributions after n steps.** `n_step_matrix(n)` computes `P^n` by
repeated squaring, and `distribution_after(pi0, n)` computes `pi0 P^n`
[EQ-PROC-MC-002]. For the two-state chain `P = [[0.7, 0.3], [0.1, 0.9]]`,
`(P^5)_00 = 0.30832`. This equals the closed form
`b/(a+b) + a/(a+b) (1-a-b)^5` with `a = 0.3` and `b = 0.1`.

**Classification.** `communicating_classes`, `closed_classes`,
`transient_states`, and `period` follow [EQ-PROC-MC-004]. The deterministic
3-cycle has period 3. Its stationary law `(1/3, 1/3, 1/3)` exists, but
`pi0 P^n` oscillates and does not converge to it.

**Stationary distribution.** `stationary_distribution()` solves the balance
equations directly, replacing one of them by `sum(pi) = 1` [EQ-PROC-MC-003].
The answer is unique exactly when the chain has one closed class; with
several closed classes the method raises instead of returning one arbitrary
solution. The two-state chain gives `pi = (0.25, 0.75)`.

**First passage.** `mean_first_passage_times()` returns `m_ij`
[EQ-PROC-MC-006]. The two-state chain gives `m_01 = 1/a = 3.333` and
`m_10 = 1/b = 10`. The diagonal holds Kac's mean return times
`m_00 = 1/pi_0 = 4` and `m_11 = 1/pi_1 = 1.333`.

**Absorption.** `absorption()` writes `P` in canonical form and returns the
fundamental matrix `N = (I - Q)^-1`, the absorption probabilities `B = N R`,
and the expected times to absorption `t = N 1` [EQ-PROC-MC-005]. For gambler's
ruin with `p = 0.4` and target `N = 5`, starting from `i = 1..4`:

| start `i` | `Pr(reach 5)` | expected duration |
|---|---|---|
| 1 | 0.075829 | 3.104265 |
| 2 | 0.189573 | 5.260664 |
| 3 | 0.360190 | 5.995261 |
| 4 | 0.616114 | 4.597156 |

These agree with the closed forms in [EQ-PROC-MC-007] to `1e-11` (see
`tests/test_stochastic_processes.py`). Row 1 of `N` is
`(1.540284, 0.900474, 0.473934, 0.189573)`: starting with one unit, the
gambler holds that same amount 1.54 times on average before the game ends.

`simulate(steps, initial=..., seed=...)` samples each row by inversion. A
20,000-step path of the birth-death chain `[[.5,.5,0],[.25,.5,.25],[0,.5,.5]]`
spends close to `(1/4, 1/2, 1/4)` of its time in the three states.

## 2. Continuous-time processes with exact samplers

| Process | Law used for simulation | Moments | Equation |
|---|---|---|---|
| `PoissonProcess(rate)` | exponential inter-arrivals | `E N(t) = Var N(t) = lambda t` | [EQ-PROC-POIS-001], [EQ-PROC-POIS-002] |
| `CompoundPoissonProcess(rate, jump)` | Poisson times, jumps by inversion | `lambda t E J`, `lambda t E J^2` | [EQ-PROC-CPOIS-001] |
| `BrownianMotion(mu, sigma)` | Gaussian increments | `x0 + mu t`, `sigma^2 t` | [EQ-PROC-BM-001] |
| `GeometricBrownianMotion(mu, sigma)` | exact log-normal factors | `S0 e^(mu t)`, `S0^2 e^(2 mu t)(e^(sigma^2 t) - 1)` | [EQ-PROC-GBM-001], [EQ-PROC-GBM-002] |
| `OrnsteinUhlenbeck(theta, mu, sigma)` | exact Gaussian transition | stationary `N(mu, sigma^2/(2 theta))` | [EQ-PROC-OU-001], [EQ-PROC-OU-002] |

Worked checks:

- **Poisson.** Rate 3, horizon 2, 4,000 replications, seed 21: sample mean
  5.996 and sample variance 5.892, against 6 for both.
- **Compound Poisson.** Rate 2 with `Exponential(0.5)` jumps (`E J = 2`,
  `E J^2 = 8`), horizon 3, 3,000 replications, seed 5: sample mean 11.95 and
  variance 49.5, against 12 and 48.
- **GBM.** `mu = 0.05`, `sigma = 0.2`, `S0 = 100`, `t = 1`, 20,000 exact
  draws, seed 13: sample mean 105.39 and variance 461.2, against 105.127 and
  451.03.
- **OU.** `theta = 2`, `mu = 1`, `sigma = 0.5`, one exact path of 20,000
  steps over `[0, 1000]` started from the stationary law, seed 6: sample mean
  0.990 and variance 0.0632, against 1 and 0.0625. The path is strongly
  autocorrelated (lag-one correlation `e^(-0.1) = 0.905`), so these 20,000
  values carry roughly the information of 1,000 independent draws.

The OU stationary covariance `sigma^2 e^(-theta|s|) / (2 theta)` is the
exponential kernel of Section 4 with length scale `1/theta`. A 1-D
exponential-kernel random field is therefore the stationary OU process
indexed by space.

## 3. SDE schemes and their convergence orders

`ScalarSDE(drift, diffusion, diffusion_derivative)` describes
`dX = a(t, X) dt + b(t, X) dW`. `euler_maruyama` [EQ-PROC-SDE-001] and
`milstein` [EQ-PROC-SDE-002] accept either a seed or prescribed Brownian
increments. With additive noise (`b_x = 0`, for example OU), the two schemes
produce identical paths.

**Strong error** compares the scheme with the exact solution on the *same*
Brownian path [EQ-PROC-SDE-003]. `strong_convergence_study` draws one fine
path per replication and forms the coarse increments as block sums. The
reported order is the least-squares slope of `log e` against `log h`
[EQ-PROC-SDE-004]. Higham's test problem (GBM with `lambda = 2`, `mu = 1`,
`X0 = 1`, `T = 1`; 500 paths, seed 1) gives:

| `h` | EM `E|X_N - X(T)|` | Milstein |
|---|---|---|
| 1/16 | 1.156 | 0.924 |
| 1/32 | 0.818 | 0.495 |
| 1/64 | 0.571 | 0.258 |
| 1/128 | 0.365 | 0.132 |
| fitted order | **0.551** | **0.938** |

The theoretical orders are 1/2 and 1. Monte Carlo standard errors at
`h = 1/128` are 0.039 (EM) and 0.018 (Milstein).

**Weak error** compares `E f(X_N)` with the exact `E f(X(T))`. For GBM with
EM, `E X_N = (1 + lambda h)^N` holds exactly, so the true bias is known. With
`lambda = 2`, `mu = 0.1`, 1,000 paths, seed 2:

| `h` | estimated weak error | exact bias `e^2 - (1 + 2h)^N` | MC s.e. |
|---|---|---|---|
| 1/16 | 0.830 | 0.806 | 0.019 |
| 1/32 | 0.457 | 0.430 | 0.022 |
| 1/64 | 0.251 | 0.223 | 0.023 |
| 1/128 | 0.143 | 0.113 | 0.024 |

The fitted order is 0.849. The order of the exact bias over this range is
0.94; it is not yet 1 because `lambda h` is not small at `h = 1/16`. A weak
study is valid only while the bias clearly exceeds the Monte Carlo standard
error. At `h = 1/128` the bias is only about five standard errors, so finer
steps need many more paths.

## 4. Gaussian random fields and the Karhunen-Loeve expansion

**Kernels.** `CovarianceKernel(family, variance, length_scale)` supports
`squared_exponential`, `exponential`, `matern32`, and `matern52`
[EQ-PROC-KER-001]-[EQ-PROC-KER-004] on points of any dimension.
`covariance_matrix(kernel, points)` accepts 1-D scalars or coordinate
tuples.

**KL expansion.** A Gaussian field is written as
`H(x) = m + sum_k sqrt(lambda_k) xi_k phi_k(x)` with independent
`xi_k ~ N(0, 1)` [EQ-PROC-KL-001]. `karhunen_loeve(kernel, points, weights)`
solves the Nystrom-discretized eigenproblem `W^(1/2) C W^(1/2) psi = lambda psi`
[EQ-PROC-KL-002]. `karhunen_loeve_interval` supplies the Gauss-Legendre,
midpoint, or trapezoid weights on an interval [EQ-PROC-QUAD-001], and
`rectangle_rule` builds a 2-D tensor grid. Without weights the result is the
plain eigen-decomposition (PCA) of the covariance matrix on the grid.

**Validation against the exact solution.** For `C = exp(-|x - y|)` on
`[-1, 1]`, the eigenpairs solve transcendental equations [EQ-PROC-KL-004].
`exponential_kernel_kl_modes` brackets each root and solves it in a
tangent-free form:

| mode | parity | `omega` | exact `lambda` | midpoint, 40 nodes | Gauss-Legendre, 20 nodes |
|---|---|---|---|---|---|
| 1 | even | 0.860334 | 1.149310 | 1.149856 | 1.152185 |
| 2 | odd | 2.028758 | 0.390941 | 0.391468 | 0.392972 |
| 3 | even | 3.425618 | 0.157049 | 0.157522 | 0.159385 |
| 4 | odd | 4.913180 | 0.079557 | 0.080005 | 0.082071 |
| 5 | even | 6.437298 | 0.047127 | 0.047564 | 0.049763 |
| 6 | odd | 7.978666 | 0.030931 | 0.031364 | 0.033679 |

The exponential kernel has a kink on the diagonal, so every rule converges
only algebraically (about `O(h^2)` here). Gauss-Legendre has no advantage
for this kernel, but it converges much faster for the smooth squared
exponential and Matern 5/2 kernels. The nodal eigenvectors of the 40-node
midpoint solution match the analytic eigenfunctions to within `2e-3`.

**Truncation.** `variance_fraction = alpha` keeps the smallest `M` with
`sum_(k<=M) lambda_k >= alpha sum_k lambda_k` [EQ-PROC-KL-003]. `max_terms`
caps `M`. Smoothness controls how fast the eigenvalues decay. On `[0, 1]`
with `ell = 0.3`, 30 Gauss-Legendre nodes, and `alpha = 0.95`, the number of
retained modes is:

| kernel | modes for 95 % |
|---|---|
| exponential | 12 |
| Matern 3/2 | 4 |
| Matern 5/2 | 4 |
| squared exponential | 3 |

The exact exponential-kernel spectrum on `[-1, 1]` (total variance 2)
explains 57.5 %, 77.0 %, 84.9 %, 88.8 %, 91.2 %, and 92.8 % with 1-6 modes.
The pointwise truncated variance `sum_(k<=M) lambda_k phi_k(x)^2` never
exceeds `sigma^2`, so a truncated field is always slightly *less* variable
than the target. This should be reported alongside results that depend on
the field variance.

**Evaluation off the grid.** `eigenfunction_at` and `realize_at` use Nystrom
interpolation [EQ-PROC-KL-005], which reproduces the nodal values exactly.

## 5. Random-field inputs in a probabilistic model

```python
kl = karhunen_loeve_interval(CovarianceKernel("exponential", 0.3**2, 0.5),
                             0.0, 1.0, 30, variance_fraction=0.9)
stiffness = RandomFieldInput(kl, prefix="xi", transform=math.exp)

def bar(p):
    ea = stiffness.field(p)          # EA(x_i) at the KL nodes
    return {"tip": 2.0 * math.fsum(w / k for w, k in zip(kl.weights, ea))}

model = ProbabilisticModel(stiffness.joint_input(), bar,
                           (QuantityOfInterest("tip", "Tip displacement"),))
study = propagate(model, 2000, design="lhs", seed=7)
```

`RandomFieldInput` declares `xi_1..xi_M` as independent `Normal(0, 1)`
`RandomInput`s and maps a parameter mapping back to nodal field values
[EQ-MOD-RF-001]. `karhunen_loeve_inputs(kl, prefix)` returns the same pair
as a tuple of inputs and a function. The bar has length 1, a tip load of 2,
and log-normal axial stiffness `EA = exp(G)`. Its tip displacement is
`u = P int dx / EA(x)`, evaluated with the KL quadrature weights. With these
settings `M = 5` modes explain 91.5 % of the variance.

| quantity | mean | variance |
|---|---|---|
| closed form for the truncated, discretized model | 2.08405 | 0.22893 |
| `propagate`, LHS 2,000, seed 7 | 2.08386 | 0.22814 |
| direct KL sampling, 10,000 draws, seed 11 | 2.08351 | 0.23190 |
| closed form, all 30 modes (untruncated) | 2.09206 | 0.23070 |

The closed form uses `E e^(-G_i) = e^(v_i/2)` and
`Cov(e^(-G_i), e^(-G_j)) = e^((v_i+v_j)/2)(e^(C_ij) - 1)` with the truncated
covariance. The last row shows the truncation bias: keeping 5 of the 30
modes under-states the output variance by 0.8 %.

Because the result is an ordinary `ProbabilisticModel`, FORM, subset
simulation, Sobol indices, and surrogates all apply unchanged. Sobol indices
then rank the KL modes, not physical locations.

## Limitations

- Eigen-decomposition uses the dependency-free Jacobi method, which scales
  as `O(n^3)` in pure Python. It is practical up to about 150 nodes; a
  40-node solve takes 1-2 seconds on a laptop.
- Kernels are stationary and isotropic; anisotropic or non-stationary fields
  require a user-built covariance and `karhunen_loeve` on custom points.
- Only Gaussian fields are expanded. Non-Gaussian fields are produced by a
  pointwise transform (`transform=math.exp`), whose marginal law and
  covariance differ from those of the underlying Gaussian field.
- The SDE schemes are scalar with uniform steps; there is no adaptive step
  control or multilevel Monte Carlo.
