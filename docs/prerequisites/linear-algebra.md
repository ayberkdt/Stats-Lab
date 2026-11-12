# Linear Algebra Prerequisite

Uncertainty quantification is linear algebra on covariance matrices. This
guide covers the four operations the rest of the library leans on: Cholesky
factors, general (LU) solves and determinants, conditioning, and symmetric
eigen-decompositions. Everything runs on `probstat.linalg`, a dependency-free
dense layer whose matrices are tuples of row tuples.

Every snippet below was executed; the block after it is its exact output, and
`tests/test_prerequisite_guides.py` re-runs all of them. Snippets share one
Python namespace and run top to bottom.

Background reading: [REF-STRANG-2016, Chapters 2, 5, 6] for the concepts and
[REF-GOLUB-VANLOAN-2013, Chapters 3, 4, 8] for the algorithms.

## 1. Matrices as tuples and the basic products

```python
from probstat.linalg import matmul, matvec, transpose, sandwich, identity

A = ((4.0, 2.0, 0.6), (2.0, 5.0, 1.0), (0.6, 1.0, 3.0))
x = (1.0, -1.0, 2.0)
print(matvec(A, x))
print(matmul(A, identity(3)) == A)
print(transpose(((1.0, 2.0, 3.0),)))
H = ((1.0, 1.0, 0.0),)
print(sandwich(H, A))
```

```text
(3.2, -1.0, 5.6)
True
((1.0,), (2.0,), (3.0,))
((13.0,),)
```

`sandwich(H, S)` is `H S H'` [EQ-LINALG-007]: the covariance of `H X` when
`Cov(X) = S`. Above, `Var(X1 + X2) = 4 + 5 + 2*2 = 13`.

## 2. Cholesky: the square root of a covariance

A symmetric positive-definite (SPD) matrix factors as `A = L L'` with lower
triangular `L` [EQ-LINALG-001]. The factor gives solves by two triangular
substitutions [EQ-LINALG-002, EQ-LINALG-003], the log-determinant
`2 sum log L_ii` [EQ-LINALG-006], and Gaussian sampling `x = mu + L z`
[EQ-DIST-MVN-005].

```python
from probstat.linalg import cholesky, cholesky_solve, solve_spd, log_det_spd
import math

L = cholesky(A)
for row in L:
    print(tuple(round(v, 6) for v in row))
reconstructed = matmul(L, transpose(L))
print(max(abs(reconstructed[i][j] - A[i][j]) for i in range(3) for j in range(3)) < 1e-14)
b = (1.0, 2.0, 3.0)
print(tuple(round(v, 6) for v in cholesky_solve(L, b)))
print(tuple(round(v, 6) for v in solve_spd(A, b)))
print(f"{log_det_spd(A):.6f} {math.log(44.6):.6f}")  # det A = 44.6 by cofactors
```

```text
(2.0, 0.0, 0.0)
(1.0, 2.0, 0.0)
(0.3, 0.35, 1.669581)
True
(0.004484, 0.212556, 0.928251)
(0.004484, 0.212556, 0.928251)
3.797734 3.797734
```

A matrix that is not positive definite is rejected instead of being silently
regularized:

```python
try:
    cholesky(((1.0, 2.0), (2.0, 1.0)))
except ValueError as error:
    print(error)
```

```text
matrix must be positive definite
```

## 3. General systems and determinants by LU

Non-symmetric matrices (Jacobians, linear maps) need Gaussian elimination with
partial pivoting, `PA = LU` [EQ-LINALG-004]. The same factorization gives the
determinant `det A = (-1)^s prod U_ii` and the scale-free `log|det A|`
[EQ-LINALG-008], which is the Jacobian term of a change of variables.

```python
from probstat.linalg import lu_solve, determinant, log_abs_determinant

M = ((0.0, 2.0, 1.0), (1.0, 1.0, 0.0), (3.0, 0.0, 1.0))
print(tuple(round(v, 6) for v in lu_solve(M, (3.0, 2.0, 4.0))))
print(round(determinant(M), 12))
print(f"{log_abs_determinant(M):.6f} {math.log(5.0):.6f}")
print(determinant(((1.0, 2.0), (2.0, 4.0))))
```

```text
(1.0, 1.0, 1.0)
-5.0
1.609438 1.609438
0.0
```

The first pivot of `M` is zero, so elimination must swap rows; one swap flips
the sign of the determinant.

## 4. Conditioning: how much a solve can amplify error

For SPD matrices the 2-norm condition number is `kappa = lambda_max /
lambda_min`. Solving `A x = b` in floating point loses roughly `log10(kappa)`
of the ~16 significant digits available. The Hilbert matrix `H_ij = 1/(i+j+1)`
is the classic ill-conditioned example.

```python
from probstat.linalg import symmetric_eigen

def hilbert(n):
    return tuple(tuple(1.0 / (i + j + 1) for j in range(n)) for i in range(n))

for n in (3, 6, 9):
    Hn = hilbert(n)
    values, _ = symmetric_eigen(Hn)
    kappa = values[-1] / values[0]
    ones = (1.0,) * n
    solution = solve_spd(Hn, matvec(Hn, ones))
    error = max(abs(s - 1.0) for s in solution)
    print(f"n={n} kappa={kappa:.2e} max error={error:.1e}")
```

```text
n=3 kappa=5.24e+02 max error=2.1e-14
n=6 kappa=1.50e+07 max error=1.5e-10
n=9 kappa=4.93e+11 max error=4.6e-07
```

The exact solution is all ones in every case; the error grows with `kappa`.
In UQ this shows up as nearly collinear inputs, almost perfectly correlated
covariance entries, and Gaussian-process kernels evaluated at nearby points.

## 5. Symmetric eigen-decomposition

Cyclic Jacobi rotations diagonalize a real symmetric matrix,
`A = V diag(lambda) V'` with orthonormal columns `V` [EQ-LINALG-005]. The
eigenvalues of a covariance are the variances along its principal axes; a
negative eigenvalue proves that a candidate "covariance" is invalid.

```python
values, vectors = symmetric_eigen(A)
print(tuple(round(v, 6) for v in values))
V = vectors
D = tuple(tuple(values[i] if i == j else 0.0 for j in range(3)) for i in range(3))
rebuilt = matmul(matmul(V, D), transpose(V))
print(max(abs(rebuilt[i][j] - A[i][j]) for i in range(3) for j in range(3)) < 1e-12)
orthogonality = matmul(transpose(V), V)
print(max(abs(orthogonality[i][j] - (i == j)) for i in range(3) for j in range(3)) < 1e-12)
print(f"{sum(values):.6f} {A[0][0] + A[1][1] + A[2][2]:.6f}")
print(f"{math.prod(values):.6f} {determinant(A):.6f}")
bad, _ = symmetric_eigen(((1.0, 0.9, 0.0), (0.9, 1.0, 0.9), (0.0, 0.9, 1.0)))
print(tuple(round(v, 6) for v in bad))
```

```text
(2.372469, 2.722508, 6.905023)
True
True
12.000000 12.000000
44.600000 44.600000
(-0.272792, 1.0, 2.272792)
```

The last matrix has unit diagonal and pairwise correlations `0.9, 0.9, 0`,
which look plausible but are not jointly possible: one eigenvalue is negative.
Trace equals the sum of eigenvalues and the determinant equals their product
([REF-STRANG-2016, Chapter 6]).

## Common pitfalls

- **Symmetry is assumed, not checked.** `cholesky` (and therefore
  `solve_spd` and `log_det_spd`) reads only the lower triangle, so a
  non-symmetric input is factored as if its upper triangle mirrored the
  lower one. `symmetrize` raises when the two triangles disagree beyond a
  relative `1e-12` and returns the averaged matrix otherwise.
- **No jitter is added.** A singular covariance raises in `cholesky`; use
  `cholesky_semidefinite` for a positive-semidefinite matrix, or add a
  nugget to the diagonal deliberately.
- **Singular matrices behave differently by function.** `determinant`
  returns `0.0`, while `log_abs_determinant` and `lu_solve` raise
  `ValueError`.
- **Eigenvectors are columns.** `symmetric_eigen` returns the eigenvalues in
  ascending order and the eigenvectors as the columns of its second result,
  so the `k`-th eigenvector is `[row[k] for row in vectors]`, not
  `vectors[k]`.
- **The backend is process-wide.** `set_backend` changes it for every caller
  and returns the previous backend; `use_backend` restores it on exit.

```python
from probstat.linalg import cholesky_semidefinite, get_backend, symmetrize, use_backend

lopsided = ((4.0, 100.0), (1.0, 4.0))
print(cholesky(lopsided) == cholesky(((4.0, 1.0), (1.0, 4.0))))
singular = ((1.0, 1.0), (1.0, 1.0))
print(cholesky_semidefinite(singular), determinant(singular))
for call in (lambda: symmetrize(lopsided), lambda: cholesky(singular),
             lambda: log_abs_determinant(singular), lambda: lu_solve(singular, (1.0, 1.0))):
    try:
        call()
    except ValueError as error:
        print(error)
values, vectors = symmetric_eigen(A)
columns = [tuple(row[k] for row in vectors) for k in range(3)]
print(values[0] < values[1] < values[2])
print(all(max(abs(a - values[k] * c) for a, c in zip(matvec(A, columns[k]), columns[k])) < 1e-12
          for k in range(3)))
with use_backend("python"):
    print(get_backend().name)
```

```text
True
((1.0, 0.0), (1.0, 0.0)) 0.0
matrix must be symmetric
matrix must be positive definite
matrix is singular; log|det| is -infinity
matrix is singular to working precision
True
True
python
```

## Where this is used

- `multivariate.MultivariateNormal` stores a Cholesky factor for densities,
  Mahalanobis distances, sampling, and Schur-complement conditioning.
- `optimization.least_squares` solves damped normal equations with
  `solve_spd`; `gauss_newton_covariance` inverts `J'J`.
- `multivariate.transformed_density` uses `log_abs_determinant` for Jacobians.
- `reliability` (SORM curvatures) and `processes` (Karhunen-Loeve modes) use
  `symmetric_eigen`.

Next: [optimization](optimization.md) and
[multivariable calculus and conditioning](multivariable-calculus-and-conditioning.md).
