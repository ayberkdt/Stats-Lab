# ADR-0081: Dense backend boundary for large dense problems

- Status: Accepted
- Date: 2026-11-11

## Context

ADR-0008 introduced a small, cited Cholesky factorization on immutable tuple
matrices and recorded the consequence that "larger dense problems will
eventually require a numerical backend ADR and reference comparisons".
ADR-0001 requires that a large numerical framework enter only through an ADR
that states ownership, precision, array, device, and reproducibility
consequences, and that the base installation stay dependency free.

The consumers whose cost grows as `n^3` are the Gaussian-process likelihood,
fit, and predictor (`surrogates.gaussian_process`) and the multivariate normal
(`multivariate.normal`). The GP path already uses hand-tuned pure-Python
kernels (ragged lower triangles, `math.sumprod`, one triangular inversion for
the gradient traces) whose results the suite pins bit for bit. A GP
likelihood evaluation with gradient takes 1.4 s at `n = 400` on the
reference machine, and a fit repeats it tens to hundreds of times.

Constraints:

- the default results of the whole suite must not change;
- NumPy must stay optional and must not be imported unless selected;
- the public API returns tuples, never arrays;
- `linalg` is layer 0 and may import only `foundations`;
- strict mypy must pass with and without NumPy installed.

## Decision

1. **Boundary in `probstat.linalg.backend` (layer 0).** Two structural
   protocols describe the dense kernels the library needs:
   - `DenseBackend`: `name`, `cholesky(matrix, *, jitter=0.0)`,
     `factorize(matrix, *, jitter=0.0)`, `solve_lower`, `solve_upper`,
     `cholesky_solve`, `solve_spd`, `log_det_spd`, `inverse_spd`, `matmul`,
     `matvec`, `symmetric_eigen`.
   - `CholeskyFactorization`, returned by `factorize`: `lower`, `jitter`,
     `solve_lower`, `solve`, `multiply_lower` (the Gaussian sampling
     transform `L z`), `log_det`, `inverse`. A factorization keeps the
     backend's native storage, so repeated predictions, density
     evaluations, and samples against one matrix do not convert the factor
     again.

   Every method accepts and returns `Vector`/`Matrix` tuples.

2. **Jitter policy.** The diagonal shift `delta` is an explicit,
   non-negative argument, zero by default, recorded on the factorization
   (`CholeskyFactorization.jitter`) [EQ-LINALG-010]. No backend chooses or
   enlarges it; a matrix that is not positive definite after the requested
   shift raises `ValueError("matrix must be positive definite")` in both
   backends. Adaptive escalation stays with the caller that owns the model
   meaning of the shift (the GP's `adaptive_nugget`, which warns).

3. **Reference backend.** `PythonBackend` (registry name `"python"`, the
   default) calls the routine of the same name in `linalg.dense`. Its
   results are therefore bit-identical to the direct routines;
   `inverse_spd` now delegates to a private `_inverse_from_cholesky` that
   `CholeskyFactorization.inverse` shares [EQ-LINALG-011], with unchanged
   arithmetic.

4. **NumPy backend.** `NumPyBackend` (registry name `"numpy"`) is
   constructed only on explicit selection. It imports NumPy through
   `importlib.import_module`, typed `Any`, so mypy never resolves NumPy and
   checks identically with or without it; an absent NumPy raises
   `ImportError` naming the `probstat-library[numerics]` extra, and the
   active backend is left unchanged. Cholesky is
   `numpy.linalg.cholesky` (LAPACK, lower triangle read), eigenvalues
   `numpy.linalg.eigh`, products BLAS [REF-HARRIS-ETAL-2020],
   [REF-LAPACK-1999, Chapter 2]. NumPy has no triangular solver, so forward
   and back substitution run one row at a time with one vector product per
   row: the same backward-stable algorithm as the reference backend
   [REF-HIGHAM-2002, Chapter 8] with `n` interpreter steps instead of
   `n^2`; `inverse` is `L^-T L^-1` from a row-wise inversion of `L`
   [REF-HIGHAM-2002, Chapter 14]. Results are converted with `tolist()`.
   `pyproject.toml` gains the optional extra `numerics = ["numpy>=1.26"]`;
   `dependencies` stays empty.

5. **Selection.** `available_backends()`, `get_backend()`,
   `set_backend(name_or_backend)` (returns the previous backend), and the
   context manager `use_backend(...)`, which restores the previous backend
   on exit and on an exception. Any object satisfying `DenseBackend` is
   accepted. The selection is process-wide; `use_backend` does not isolate
   concurrently running threads.

6. **Consumers.**
   - `multivariate.MultivariateNormal` factorizes its covariance with the
     active backend at construction and keeps the factorization; the
     Mahalanobis solve, log-determinant, conditional law, and sampling
     transform use it. Under `PythonBackend` each step is the same
     arithmetic as before (same `cholesky`, `solve_lower`, `math.fsum`
     sums), so results are bit-identical.
   - `surrogates.gaussian_process`: under `PythonBackend` the specialized
     kernels run unchanged. Under any other backend the likelihood, the
     fitted surrogate, the restart noise estimate, and the condition number
     factorize the full kernel matrix once through `factorize` and take
     `alpha = K^-1 y`, `log det K`, and `K^-1` from it; the kernel assembly
     and the `O(n^2 d)` gradient contraction stay in Python. A fitted
     surrogate keeps the factorization of the backend that built it, so its
     predictions do not depend on the backend active later.
   - Other dense users (`inference`, `inverse`, `filtering`) keep calling
     `linalg.dense` directly; they solve systems of the parameter or state
     dimension, not of the sample size.

Public names added to `probstat.linalg`: `DenseBackend`,
`CholeskyFactorization`, `PythonBackend`, `NumPyBackend`,
`available_backends`, `get_backend`, `set_backend`, `use_backend`.

## Consequences

- The base installation remains dependency free, and importing any
  `probstat` module never imports NumPy (verified in a fresh interpreter
  with the import blocked).
- Default results are unchanged: the reference backend is the existing code,
  and the full suite passes without modification.
- Results under the NumPy backend are not bit-identical to the default:
  BLAS accumulation order differs from `math.fsum`. They agree within the
  backward-error bounds of [EQ-LINALG-012]; eigenvectors may differ in sign.
- Precision is IEEE double in both backends; there is no device or
  single-precision path. Reproducibility under NumPy depends on the BLAS
  build and thread count, which can change the last bits of products.
- The GP speed-up is limited by kernel assembly, which is Python code in
  every backend; the dense kernels themselves speed up by one to two orders
  of magnitude. Vectorized kernel assembly would need array-valued kernels
  and is out of scope.
- A further backend (for example a GPU array library) is one class
  implementing `DenseBackend`; no consumer changes.

## Verification

`tests/test_linalg_backend.py` (19 tests, 48 subtests):

- **Selection.** Default backend, `set_backend` by name (case- and
  space-insensitive) and by object, unknown names (`ValueError`) and types
  (`TypeError`) leaving the state unchanged, nested `use_backend` with
  restoration on normal exit and on an exception, and a user-supplied
  backend driving the GP path.
- **Absent NumPy.** In-process with `sys.modules["numpy"] = None`:
  `NumPyBackend()`, `set_backend("numpy")`, and `use_backend("numpy")`
  raise `ImportError` naming the extra, and the active backend is unchanged.
  In a fresh interpreter with a meta-path finder that blocks NumPy:
  `probstat.linalg`, `multivariate`, and `surrogates` import, an MVN density
  and samples and a GP fit run, selecting `"numpy"` raises, and NumPy is
  absent from `sys.modules`.
- **Reference backend.** Every `PythonBackend` method and factorization
  method equals the wrapped routine exactly (`assertEqual` on tuples); the
  jitter policy (shifted factor equals the factor of `A + delta I`,
  negative and NaN shifts rejected, insufficient shifts raise); the MVN
  log-density and samples equal the direct formulas bit for bit.
- **Reference comparisons** (run when NumPy is importable; CI installs it
  through Matplotlib). Random SPD matrices `A = B B'/n + 0.1 I`,
  `kappa_2(A)` between 10 and 41, against the bounds of [EQ-LINALG-012]
  [REF-HIGHAM-2002, Chapters 3, 7, 8, and 10]:

  | n | factor difference / bound | solve difference / bound | log det difference / bound |
  |---|---|---|---|
  | 5 | 1.4e-16 / 2.9e-14 | 1.1e-16 / 7.7e-14 | 4.4e-16 / 1.4e-13 |
  | 50 | 7.6e-16 / 6.4e-12 | 7.8e-16 / 1.9e-11 | 3.6e-15 / 3.2e-10 |
  | 100 | 1.0e-15 / 2.4e-11 | 8.4e-16 / 7.2e-11 | 7.1e-15 / 2.4e-09 |
  | 200 | 1.7e-15 / 9.7e-11 | 8.4e-16 / 2.9e-10 | 0 / 2.0e-08 |

  Both backends also satisfy the componentwise Cholesky residual bound
  `|A - L L'| <= (gamma_(n+1) + 2u) |L||L'|` up to `n = 200`; products agree
  componentwise within `(gamma_k + 2u)|A||B|`; inverses, back substitution,
  and factor solves agree within the solve bound; eigenvalues agree within
  Weyl's bound [REF-GOLUB-VANLOAN-2013, Section 8.1] and NumPy eigenpairs
  have residuals within it; non-positive-definite, singular, misshapen, and
  NaN inputs raise `ValueError` in both. A 200-dimensional MVN agrees in
  log-density, Mahalanobis distance, samples from the same seed, and
  conditional mean.
- **GP, `n = 300`** (two inputs, Matern 5/2, fixed hyperparameters,
  nugget `1e-3`, `kappa_1(K) = 2.6e5`): log marginal likelihood 290.735,
  difference `6.7e-11`; largest predictive-mean difference over 25 test
  points `2.7e-13`, predictive-variance difference `8.2e-16`; LOO relative
  error difference `1.6e-15`; condition numbers agree to `1.0e-13`
  relative. The test tolerance is `2 kappa_1 n gamma_(3n+1) = 1.6e-5`
  relative, scaled by `n + ||y|| ||alpha||` for the likelihood (the form
  used by the GP restart-noise estimate), by
  `s_y sqrt(n) sigma_f^2 ||alpha||` for means, and by `2 s_y^2 sigma_f^2`
  for variances.
- **GP, `n = 60`, optimized** (estimated noise, two restarts): both
  backends converge to the same optimum within `1e-4` in log
  hyperparameters and `1e-7` relative in the log marginal likelihood; the
  optimizer's gradient tolerance, not rounding, sets this tolerance.

**Speed-up** (`PYTHONPATH=src python tools/time_dense_backend.py`, best of
three wall times; Windows 11, CPython 3.12.1, NumPy 2.2.6 with OpenBLAS
0.3.29). `spd` is `factorize` + `solve` + `log_det` through the backend API
including tuple conversions; `gp` one likelihood-and-gradient evaluation;
`mvn` constructing a multivariate normal (validation included) and one
log-density.

| n | task | python (s) | numpy (s) | speed-up |
|---|---|---|---|---|
| 100 | spd | 0.0328 | 0.0013 | 24x |
| 100 | gp | 0.0347 | 0.0176 | 2.0x |
| 100 | mvn | 0.0333 | 0.0089 | 3.7x |
| 200 | spd | 0.2024 | 0.0057 | 36x |
| 200 | gp | 0.2049 | 0.0660 | 3.1x |
| 200 | mvn | 0.2133 | 0.0392 | 5.4x |
| 400 | spd | 1.4486 | 0.0201 | 72x |
| 400 | gp | 1.3986 | 0.2745 | 5.1x |
| 400 | mvn | 1.5375 | 0.1802 | 8.5x |

At `n = 400` about three quarters of the NumPy GP time is kernel assembly
in Python (profiled), and most of the MVN time is input validation.

The full suite passes unchanged except `tests/test_api_reference.py`, which
reports the new public names until the API reference is regenerated; ruff
and strict mypy pass, the latter both in an environment without NumPy and in
one with NumPy and Matplotlib installed.
