# ADR-0118 — Rank-1 lattice rules and mean/median randomized quasi-Monte Carlo

- Status: Accepted
- Date: 2026-11-21
- Extends: the sampling designs of `probstat.sampling` (Halton and Sobol
  points with their randomizations, [EQ-SAMP-RQMC-001])

## Context

`sampling` offered digital nets and Halton points but no lattice rules, and
no estimator that turns a randomized design into an integral estimate with an
error statement: replicate-based intervals existed only inside
`propagation`, three layers up. Two things were missing.

- **Lattice rules.** A rank-1 lattice needs a generating vector, and a
  generating vector needs a quality criterion and a search.
- **A way to combine randomizations.** The mean of `R` replicates is unbiased
  and has a variance estimate, but its Student-t interval rests on normality
  that randomized quasi-Monte Carlo replicates do not have. The median of the
  replicates is robust to occasional bad randomizations, which makes
  constructing the rule optional (Goda and L'Ecuyer 2022; Pan and Owen 2023),
  but it has no error bar.

## Decision

1. **New module `sampling/lattice.py`**: `rank1_lattice`,
   `lattice_squared_error`, `cbc_generating_vector`,
   `korobov_generating_vector`, `korobov_vector`, and the frozen result
   `LatticeRule`. **New module `sampling/randomized_integration.py`**:
   `rqmc_mean`, `rqmc_median`, `median_random_lattice`, the frozen result
   `RQMCResult`, and the tuple `RQMC_METHODS`. Imports stay within layers
   0-1 (`foundations`, and `distributions.student_t` for the t quantile, as
   `designs.py` already imports `distributions.normal`).
2. **Points** ([EQ-SAMP-LATTICE-001]). Unshifted points are the exact
   floats of `(k z_j mod n) / n` and include the origin; the shifted points
   use one `Delta` from the stream's generator and are clamped into
   `(0, 1)` like the other designs. Components of `z` must be coprime to
   `n`; `n` itself need not be prime for the points or for the criterion.
3. **Criterion** ([EQ-SAMP-LATTICE-002]), derived rather than quoted:
   - Per coordinate take the Hilbert space of absolutely continuous `f` on
     `[0, 1]` with `<f, g> = (int f)(int g) + (1/gamma) int f' g'`. Its
     reproducing kernel is
     `K(x, y) = 1 + gamma (B_1(x) B_1(y) + B_2(|x - y|) / 2)`: then
     `int K(x, y) dx = 1` and
     `d/dx K = gamma (B_1(y) + B_1(|x - y|) sign(x - y))`, and inserting
     this in the inner product gives `f(y)` (integrate `f'` on both sides of
     `y`). The `s`-dimensional space is the tensor product with kernel
     `prod_j K_j`.
   - For points `P` and the equal-weight rule, the squared worst-case error
     over the unit ball is
     `e^2(P) = int int K - (2/n) sum_i int K(x_i, y) dy + n^-2 sum_ik K(x_i, x_k)
     = -1 + n^-2 sum_ik K(x_i, x_k)`.
   - Averaging over a shift replaces `K` by
     `K_sh(x, y) = int K(frac(x + D), frac(y + D)) dD`. With the Fourier
     series `B_1(frac(t)) = -sum_(h != 0) e^(2 pi i h t) / (2 pi i h)` and
     `B_2(frac(t)) = sum_(h != 0) e^(2 pi i h t) / (2 pi^2 h^2)`, the shift
     average of `B_1 B_1` is `B_2(frac(x - y)) / 2`, and
     `B_2(|x - y|) = B_2(frac(x - y))` by the symmetry of `B_2` about `1/2`.
     Hence `K_sh = prod_j (1 + gamma_j B_2(frac(x_j - y_j)))`.
   - On a lattice `x_i - x_k` runs over the lattice itself `n` times, so the
     double sum collapses to the single sum of the record.
   Authors scale the weights of such spaces differently; the library states
   the kernel and inner product instead of naming a convention. Default
   weights are all one.
4. **Component-by-component search** ([EQ-SAMP-LATTICE-003]) for prime `n`:
   `z_1 = 1`, then each component minimizes `e^2` with the earlier ones
   fixed. Only `z <= n/2` is searched because `B_2` is symmetric; ties go to
   the smallest `z`. The implementation is the naive `O(s n^2)` loop (the
   running products are kept, so one candidate costs `n` multiplications);
   results are cached per `(n, weights)`. It is documented as greedy:
   optimal for `s <= 2`, a heuristic beyond.
5. **Korobov search** ([EQ-SAMP-LATTICE-004]): exhaustive over
   `1 <= a <= n/2` for prime `n`.
6. **Mean estimator** ([EQ-SAMP-RQMC-002]): `R >= 2` independent
   randomizations from the child streams `("replicate", r)`; the rule is a
   shifted lattice (CBC vector unless one is supplied), Owen-scrambled or
   digitally shifted Sobol points, or digit-permuted Halton points. The
   result carries the replicate estimates, the standard error, a Student-t
   interval whose `interval_method` text says that coverage is not
   guaranteed, and `evaluations = R n`.
7. **Median estimator** ([EQ-SAMP-RQMC-003]): odd `R >= 3`, same replicates
   as the mean for the same seed. `standard_error`, `interval`, and
   `confidence` are `None`, and `interval_method` states the reason: the
   median of the replicates estimates the median of the replicate law, which
   is the integral only for a law symmetric about it, so neither the
   replicate spread nor an order-statistic interval bounds its error. The
   sample standard deviation of the replicates is reported as a diagnostic.
8. **Random generating vectors** ([EQ-SAMP-RQMC-004]):
   `median_random_lattice` draws `z^(r)` uniformly from `{1..n-1}^s` (prime
   `n >= 3`) per replicate. An independent random shift per replicate is on
   by default; this is a library choice that keeps every replicate unbiased
   for non-periodic integrands, and `shift=False` gives the unshifted rules.

## Consequences

- `cbc_generating_vector(251, 4)` returns `(1, 70, 97, 118)` with
  `e^2 = 1.040e-4`; the best Korobov vector (`a = 61`) has `1.097e-4` and the
  fixed Korobov vector `a = 76` has `1.319e-3`. For `n = 1021`, `s = 4` the
  values are `9.38e-6` (CBC), `9.92e-6` (Korobov), at about 0.3 s for the
  CBC search in pure Python. For `n = 7, 11, 13` and `s <= 3` (weights
  `1, 1/2, 1/4`) CBC attains the brute-force minimum over all vectors; the
  best Korobov vector does not for `n = 11, 13`, `s = 3`.
- Unshifted CBC lattice, `f = prod_j exp(sin(2 pi x_j)) / I_0(1)` in three
  dimensions: absolute errors `4.3e-2`, `6.6e-5`, `4.8e-11` at
  `n = 31, 127, 509`, against `n^-1/2 = 0.18, 0.089, 0.044`.
- Root mean squared error over 40 seeds, `s = 3`, `R = 11`, at equal cost
  `11 n`, for `f = prod_j (1 + B_2(x_j))`:

  | n | mean, CBC lattice | median, CBC lattice | median, random z | mean of the same random-z replicates | plain Monte Carlo |
  |---|---|---|---|---|---|
  | 31 | 5.8e-5 | 7.2e-5 | 2.7e-4 | 4.7e-4 | 7.6e-3 |
  | 127 | 4.1e-6 | 5.7e-6 | 4.5e-5 | 1.8e-4 | 3.6e-3 |
  | 509 | 4.5e-7 | 4.7e-7 | 4.0e-6 | 1.3e-4 | 1.5e-3 |

  For the periodic exponential integrand at `n = 509` the median with random
  vectors has error `7.0e-6` and the mean of the same replicates `1.0e-2`
  (Monte Carlo `1.8e-2`). With a constructed rule the median is no better
  than the mean (ratios 1.0 to 1.8 in these studies): its benefit is
  robustness to bad randomizations, not variance reduction.
- For the non-periodic `f = prod_j x_j e^(x_j)` the shifted lattice is only
  3 to 5 times more accurate than Monte Carlo (`1.3e-2` against `4.0e-2` at
  `n = 127`): lattice rules reward periodic integrands, and no
  periodizing transformation is applied.
- Student-t coverage of nominal 95 % intervals over 1000 seeds, `n = 31`
  (`32` for nets): between 0.935 and 0.955 for shifted lattices, Owen-
  scrambled Sobol and scrambled Halton points on three two-dimensional
  integrands with `R = 5` and `R = 10`; `0.923` for the one-dimensional
  linear integrand with a shifted lattice and `R = 5`, where each replicate
  error is exactly uniform. The docstrings call the interval approximate.
- The mean over 300 seeds of the shifted-lattice estimate differs from the
  integral by less than three standard errors on three integrands.

## Rejected alternatives

- **Attaching an interval to the median** (a t interval over the replicates,
  a bootstrap, or a sign-test interval). Each covers a functional of the
  replicate law that is not the integral, or assumes symmetry that is not
  checked; none has a coverage guarantee, so none is offered.
- **Fast CBC by fast Fourier transform.** It reduces the cost to
  `O(s n log n)` but needs a primitive root and circulant bookkeeping; the
  naive search takes 0.3 s at `n = 1021`, `s = 4`, and the sizes practical
  for a pure-Python integrand are of that order.
- **Citing a named weighted-space convention for the criterion.** The
  constant in front of `B_2` differs between texts; stating the kernel and
  verifying it numerically is unambiguous.
- **Composite `n` in the searches.** The criterion and the points accept any
  `n`; restricting the searches to primes keeps every candidate admissible
  and the statement of the algorithm short.
- **A periodizing (baker or tent) transformation.** It changes the integrand
  and the function space; out of scope here.
- **Returning the mean and the median in one call.** They answer different
  questions and only one has an error bar; `rqmc_median` reuses the
  replicates of `rqmc_mean` for the same seed, so both can be had at the
  cost of one extra pass.

## Verification

`tests/test_lattice_and_median_qmc.py` (39 tests, about 4 s) checks the
points in rational arithmetic, group closure, the shift as one common
rotation, exactness for `exp(2 pi i h.x)` off the dual lattice by
enumeration over `h` in `[-5, 5]^s`, the closed form `gamma / (6 n^2)`, the
single-sum reduction against the double sum, the reproducing property of the
kernel by quadrature, the criterion against a Monte Carlo average over 600
shifts of the kernel worst-case error, CBC and Korobov against brute force,
every greedy step against its own exhaustive minimum, error decay, the
replicate streams, known integrals for all four rules, unbiasedness, mean and
median against Monte Carlo, the median against the mean for random
generating vectors, measured t coverage in a near-nominal and an
under-covering case, and input validation.

## References

[REF-CRANLEY-PATTERSON-1976], [REF-L-ECUYER-2018], [REF-GODA-LECUYER-2022],
[REF-PAN-OWEN-2023], [REF-LECUYER-NAKAYAMA-OWEN-TUFFIN-2023],
[REF-OWEN-1995].
