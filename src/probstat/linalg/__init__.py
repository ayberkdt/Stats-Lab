"""Dependency-free dense linear algebra shared by every higher layer.

The dense kernels of the ``n^3`` consumers are reached through a selectable
:class:`DenseBackend` (ADR-0081): the default ``"python"`` backend wraps the
routines below unchanged, and an optional ``"numpy"`` backend is loaded only
on explicit selection with :func:`set_backend` or :func:`use_backend`.

Main entry points:
    ``cholesky``: lower factor ``L`` of a positive-definite matrix,
        ``A = L L'``.
    ``solve_spd``: solve ``A x = b`` for a symmetric positive-definite ``A``.
    ``log_det_spd``: ``log det A`` from the Cholesky factor.
    ``lu_solve``: solve a general square system by pivoted LU.
    ``symmetric_eigen``: eigenvalues and eigenvectors of a symmetric matrix
        by Jacobi rotations.
    ``householder_qr``: thin QR of a tall matrix that flags numerically
        dependent columns.
    ``singular_value_decomposition``: thin SVD by one-sided Jacobi
        rotations, with ``pseudo_inverse`` and ``numerical_rank``.
    ``use_backend``: select a dense backend for the duration of a ``with``
        block.
    ``LinearOperator`` and the operator classes (experimental, 0.28):
        matrix-free maps with ``matvec`` and the Euclidean transpose
        ``rmatvec``, composition, stacking, counting, ``adjoint_test`` and
        ``weighted_adjoint``.
    ``DiagonalCovariance``, ``DenseCovariance``, ``BlockDiagonalCovariance``:
        covariance factors with whitening, solves and log-determinants.
    ``cgls``, ``lsqr``, ``lsmr``, ``conjugate_gradient``: matrix-free
        Krylov solvers; ``power_iteration``, ``lanczos_eigenpairs``,
        ``hutchinson_trace`` and ``randomized_svd`` for spectra and traces.

Assumptions:
    Matrices are finite, dense, and small enough for ``O(n^3)`` pure-Python
    work; they are passed as row sequences and returned as tuples of row
    tuples. Functions documented for symmetric input do not check symmetry.
    Results are floating-point and lose roughly ``log10(cond(A))`` digits.

Common pitfalls:
    - ``cholesky`` reads only the lower triangle and does not check
      symmetry: ``cholesky(((4.0, 100.0), (1.0, 4.0)))`` returns the factor
      of ``((4.0, 1.0), (1.0, 4.0))``. Pass a doubtful matrix through
      ``symmetrize``, which raises if it is not symmetric to tolerance.
    - No jitter is added: a singular covariance raises ``ValueError`` in
      ``cholesky`` and ``solve_spd``. ``cholesky_semidefinite`` accepts
      positive-semidefinite input; adding a nugget is the caller's choice.
    - ``determinant`` returns ``0.0`` for a singular matrix while
      ``log_abs_determinant`` and ``lu_solve`` raise ``ValueError``.
    - ``symmetric_eigen`` returns eigenvalues in ascending order and the
      eigenvectors as the COLUMNS of the second result, not its rows.
      ``singular_value_decomposition`` returns singular values in DESCENDING
      order, singular vectors as columns, and signs fixed only up to a
      simultaneous flip of ``u_i`` and ``v_i``.
    - ``numerical_rank`` and ``pseudo_inverse`` take a tolerance RELATIVE to
      the largest singular value (default ``max(m, n) * 2^-52``).
    - ``set_backend`` changes the backend for the whole process and
      returns the previous one; prefer ``use_backend``, which restores it.
      ``available_backends`` lists registered names, not whether NumPy
      imports.

Guide: docs/prerequisites/linear-algebra.md

References:
    [REF-GOLUB-VANLOAN-2013] (Cholesky, LU, QR, and Jacobi algorithms),
    [REF-HESTENES-1958] (one-sided Jacobi SVD), [REF-PENROSE-1955]
    (pseudo-inverse),
    [REF-STRANG-2016] (concepts), [REF-HARRIS-ETAL-2020] (the optional
    NumPy backend), [REF-PAIGE-SAUNDERS-1982], [REF-FONG-SAUNDERS-2011],
    [REF-HALKO-MARTINSSON-TROPP-2011], [REF-HUTCHINSON-1989] (matrix-free
    solvers and estimators).

"""

from .backend import (
    CholeskyFactorization,
    DenseBackend,
    NumPyBackend,
    PythonBackend,
    available_backends,
    get_backend,
    set_backend,
    use_backend,
)
from .covariance_operators import (
    BlockDiagonalCovariance,
    DenseCovariance,
    DiagonalCovariance,
    ScaledIdentityCovariance,
    is_covariance_factor,
)
from .dense import (
    Matrix,
    Vector,
    add,
    as_matrix,
    as_vector,
    cholesky,
    cholesky_semidefinite,
    cholesky_solve,
    determinant,
    dot,
    identity,
    inverse_spd,
    log_abs_determinant,
    log_det_spd,
    lu_solve,
    matmul,
    matvec,
    norm,
    outer,
    quadratic_form,
    sandwich,
    scale_matrix,
    solve_lower,
    solve_spd,
    solve_upper,
    symmetric_eigen,
    symmetrize,
    transpose,
)
from .krylov import (
    KrylovResult,
    LanczosResult,
    RandomizedSVD,
    SpectralNormEstimate,
    TraceEstimate,
    cgls,
    conjugate_gradient,
    hutchinson_trace,
    lanczos_eigenpairs,
    lsmr,
    lsqr,
    power_iteration,
    randomized_svd,
)
from .operators import (
    AdjointTestResult,
    BlockDiagonalOperator,
    CallbackOperator,
    ComposedOperator,
    Convolution1DOperator,
    CountingOperator,
    DenseOperator,
    DiagonalOperator,
    FirstDifferenceOperator,
    GridGradientOperator,
    HStackOperator,
    IdentityOperator,
    LinearOperator,
    OperatorBase,
    OperatorCounts,
    PermutationOperator,
    ScaledOperator,
    SelectionOperator,
    SumOperator,
    SymmetricPositiveDefinite,
    TransposedOperator,
    VStackOperator,
    ZeroOperator,
    adjoint_test,
    apply,
    apply_transpose,
    materialize,
    require_operator,
    weighted_adjoint,
)
from .qr import DEFAULT_RANK_TOLERANCE, HouseholderQR, householder_qr
from .svd import (
    DEFAULT_JACOBI_TOLERANCE,
    SingularValueDecomposition,
    numerical_rank,
    pseudo_inverse,
    singular_value_decomposition,
)

__all__ = [
    "AdjointTestResult",
    "BlockDiagonalCovariance",
    "BlockDiagonalOperator",
    "CallbackOperator",
    "ComposedOperator",
    "Convolution1DOperator",
    "CountingOperator",
    "DenseCovariance",
    "DenseOperator",
    "DiagonalCovariance",
    "DiagonalOperator",
    "FirstDifferenceOperator",
    "GridGradientOperator",
    "HStackOperator",
    "IdentityOperator",
    "KrylovResult",
    "LanczosResult",
    "LinearOperator",
    "OperatorBase",
    "OperatorCounts",
    "PermutationOperator",
    "RandomizedSVD",
    "ScaledIdentityCovariance",
    "ScaledOperator",
    "SelectionOperator",
    "SpectralNormEstimate",
    "SumOperator",
    "SymmetricPositiveDefinite",
    "TraceEstimate",
    "TransposedOperator",
    "VStackOperator",
    "ZeroOperator",
    "adjoint_test",
    "apply",
    "apply_transpose",
    "cgls",
    "conjugate_gradient",
    "hutchinson_trace",
    "is_covariance_factor",
    "lanczos_eigenpairs",
    "lsmr",
    "lsqr",
    "materialize",
    "power_iteration",
    "randomized_svd",
    "require_operator",
    "weighted_adjoint",
    "DEFAULT_JACOBI_TOLERANCE",
    "DEFAULT_RANK_TOLERANCE",
    "CholeskyFactorization",
    "DenseBackend",
    "HouseholderQR",
    "Matrix",
    "NumPyBackend",
    "PythonBackend",
    "SingularValueDecomposition",
    "Vector",
    "add",
    "as_matrix",
    "as_vector",
    "available_backends",
    "cholesky",
    "cholesky_semidefinite",
    "cholesky_solve",
    "determinant",
    "dot",
    "get_backend",
    "householder_qr",
    "identity",
    "inverse_spd",
    "log_abs_determinant",
    "log_det_spd",
    "lu_solve",
    "matmul",
    "matvec",
    "norm",
    "numerical_rank",
    "outer",
    "pseudo_inverse",
    "quadratic_form",
    "sandwich",
    "scale_matrix",
    "set_backend",
    "singular_value_decomposition",
    "solve_lower",
    "solve_spd",
    "solve_upper",
    "symmetric_eigen",
    "symmetrize",
    "transpose",
    "use_backend",
]
