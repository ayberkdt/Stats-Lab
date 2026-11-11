"""Tests for the dense backend boundary (ADR-0081).

Three groups:

* selection: the default backend, ``set_backend`` and ``use_backend``
  (including state restoration on an exception), and the ``ImportError``
  path when NumPy is absent (simulated by blocking the import, in-process
  and in a fresh interpreter);
* the reference backend: every :class:`PythonBackend` method is
  bit-identical to the :mod:`probstat.linalg` routine it wraps, and the
  explicit jitter policy;
* reference comparisons, run when NumPy is importable: the NumPy backend
  against the reference backend on random SPD matrices up to ``n = 200``,
  multivariate-normal densities and samples, and Gaussian-process fits on
  ``n = 300`` (fixed hyperparameters) and ``n = 60`` (optimized) points.

Agreement bounds follow the backward-error analyses of Higham (2002)
[REF-HIGHAM-2002, Chapters 3, 7, 8, and 10]: both backends are backward
stable, so each computed result is the exact result for a matrix within
``gamma_k |L||L'|`` of ``A`` (``gamma_k = k u / (1 - k u)``, ``u = 2^-53``),
and two such results differ by at most twice the first-order forward error
``kappa_2(A) * backward error`` [EQ-LINALG-012]. Eigenvalue bounds use
Weyl's perturbation inequality [REF-GOLUB-VANLOAN-2013, Section 8.1].
"""

from __future__ import annotations

import importlib.util
import math
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from probstat import linalg
from probstat.foundations.random_streams import resolve_stream
from probstat.linalg import (
    CholeskyFactorization,
    DenseBackend,
    Matrix,
    PythonBackend,
    Vector,
    available_backends,
    get_backend,
    set_backend,
    use_backend,
)
from probstat.multivariate import MultivariateNormal
from probstat.surrogates import (
    fit_gaussian_process,
    gaussian_process_log_marginal_likelihood,
    kernel_value,
)

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
UNIT_ROUNDOFF = 2.0**-53
SOURCE_ROOT = Path(__file__).parents[1] / "src"


def gamma(k: int) -> float:
    """Return Higham's ``gamma_k = k u / (1 - k u)``.

    References:
        [REF-HIGHAM-2002, Chapter 3], [EQ-LINALG-012].

    """
    return k * UNIT_ROUNDOFF / (1.0 - k * UNIT_ROUNDOFF)


def random_matrix(rows: int, columns: int, seed: int) -> Matrix:
    """Return a matrix of independent standard normal entries.

    References:
        [REF-GOLUB-VANLOAN-2013, Chapter 1]. No equation is evaluated.

    """
    generator = random.Random(seed)
    return tuple(
        tuple(generator.gauss(0.0, 1.0) for _ in range(columns)) for _ in range(rows)
    )


def random_spd(size: int, seed: int, shift: float = 0.1) -> Matrix:
    """Return ``B B' / n + shift I`` for a Gaussian ``B`` (exactly symmetric).

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

    """
    factor = random_matrix(size, size, seed)
    product = linalg.matmul(factor, linalg.transpose(factor))
    return linalg.symmetrize(
        linalg.add(
            linalg.scale_matrix(product, 1.0 / size), linalg.identity(size), scale=shift
        )
    )


def frobenius(matrix: Matrix) -> float:
    """Return the Frobenius norm.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 2.3]. No equation is evaluated.

    """
    return math.sqrt(math.fsum(value * value for row in matrix for value in row))


def difference(left: Matrix, right: Matrix) -> Matrix:
    """Return ``left - right``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No equation is evaluated.

    """
    return linalg.add(left, right, scale=-1.0)


def vector_norm(vector: Vector) -> float:
    """Return the Euclidean norm.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 2.2]. No equation is evaluated.

    """
    return math.hypot(*vector)


class _WrappedBackend:
    """A user-supplied backend that delegates to the reference backend.

    References:
        [REF-PROBSTAT-API]. No equation is evaluated.

    """

    def __init__(self) -> None:
        """Create the delegate.

        References:
            [REF-PROBSTAT-API]. No equation is evaluated.

        """
        self._inner = PythonBackend()

    @property
    def name(self) -> str:
        """Return the backend name.

        References:
            [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return "wrapped"

    def __getattr__(self, attribute: str) -> object:
        """Delegate every kernel to the reference backend.

        References:
            [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return getattr(self._inner, attribute)

    def cholesky(self, matrix: Matrix, *, jitter: float = 0.0) -> Matrix:
        """Delegate the factorization.

        References:
            [EQ-LINALG-001]. No new equation.

        """
        return self._inner.cholesky(matrix, jitter=jitter)

    def factorize(
        self, matrix: Matrix, *, jitter: float = 0.0
    ) -> CholeskyFactorization:
        """Delegate the retained factorization.

        References:
            [EQ-LINALG-001]. No new equation.

        """
        return self._inner.factorize(matrix, jitter=jitter)

    def solve_lower(self, lower: Matrix, right_hand_side: Vector) -> Vector:
        """Delegate forward substitution.

        References:
            [EQ-LINALG-002]. No new equation.

        """
        return self._inner.solve_lower(lower, right_hand_side)

    def solve_upper(self, upper: Matrix, right_hand_side: Vector) -> Vector:
        """Delegate back substitution.

        References:
            [EQ-LINALG-003]. No new equation.

        """
        return self._inner.solve_upper(upper, right_hand_side)

    def cholesky_solve(self, lower: Matrix, right_hand_side: Vector) -> Vector:
        """Delegate the factor solve.

        References:
            [EQ-LINALG-002]. No new equation.

        """
        return self._inner.cholesky_solve(lower, right_hand_side)

    def solve_spd(self, matrix: Matrix, right_hand_side: Vector) -> Vector:
        """Delegate the SPD solve.

        References:
            [EQ-LINALG-001]. No new equation.

        """
        return self._inner.solve_spd(matrix, right_hand_side)

    def log_det_spd(self, matrix: Matrix) -> float:
        """Delegate the log-determinant.

        References:
            [EQ-LINALG-006]. No new equation.

        """
        return self._inner.log_det_spd(matrix)

    def inverse_spd(self, matrix: Matrix) -> Matrix:
        """Delegate the inverse.

        References:
            [EQ-LINALG-011]. No new equation.

        """
        return self._inner.inverse_spd(matrix)

    def matmul(self, left: Matrix, right: Matrix) -> Matrix:
        """Delegate the matrix product.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 1.1]. No equation is evaluated.

        """
        return self._inner.matmul(left, right)

    def matvec(self, matrix: Matrix, vector: Vector) -> Vector:
        """Delegate the matrix-vector product.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 1.1]. No equation is evaluated.

        """
        return self._inner.matvec(matrix, vector)

    def symmetric_eigen(self, matrix: Matrix) -> tuple[Vector, Matrix]:
        """Delegate the eigendecomposition.

        References:
            [EQ-LINALG-005]. No new equation.

        """
        return self._inner.symmetric_eigen(matrix)


class BackendSelectionTests(unittest.TestCase):
    """Backend registry, selection, restoration, and the absent-NumPy path."""

    def setUp(self) -> None:
        """Restore the active backend after every test.

        References:
            [REF-PROBSTAT-API]. No equation is evaluated.

        """
        self.addCleanup(set_backend, get_backend())

    def test_default_backend_is_the_reference_backend(self) -> None:
        """The process starts on the dependency-free backend."""
        self.assertIsInstance(get_backend(), PythonBackend)
        self.assertEqual(get_backend().name, "python")
        self.assertEqual(available_backends(), ("python", "numpy"))
        self.assertIsInstance(get_backend(), DenseBackend)

    def test_set_backend_returns_previous_and_accepts_objects(self) -> None:
        """``set_backend`` swaps the backend and hands back the old one."""
        wrapped = _WrappedBackend()
        self.assertIsInstance(wrapped, DenseBackend)
        previous = set_backend(wrapped)
        self.assertIsInstance(previous, PythonBackend)
        self.assertIs(get_backend(), wrapped)
        self.assertIs(set_backend(" Python "), wrapped)
        self.assertIsInstance(get_backend(), PythonBackend)

    def test_set_backend_rejects_unknown_names_and_types(self) -> None:
        """Unknown names and non-backends leave the state unchanged."""
        before = get_backend()
        with self.assertRaisesRegex(ValueError, "unknown dense backend"):
            set_backend("cupy")
        with self.assertRaises(TypeError):
            set_backend(42)  # type: ignore[arg-type]
        self.assertIs(get_backend(), before)

    def test_use_backend_restores_state_on_exit_and_on_error(self) -> None:
        """The context manager restores the previous backend in every case."""
        before = get_backend()
        wrapped = _WrappedBackend()
        with use_backend(wrapped) as active:
            self.assertIs(active, wrapped)
            self.assertIs(get_backend(), wrapped)
            with use_backend("python") as inner:
                self.assertIsInstance(inner, PythonBackend)
            self.assertIs(get_backend(), wrapped)
        self.assertIs(get_backend(), before)
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with use_backend(wrapped):
                raise RuntimeError("body failed")
        self.assertIs(get_backend(), before)

    def test_a_custom_backend_drives_the_consumers(self) -> None:
        """A user backend is honored by the GP path and gives the same numbers."""
        points = [(0.0,), (0.3,), (0.7,), (1.0,)]
        values = [0.0, 1.0, 0.4, -0.2]
        reference = gaussian_process_log_marginal_likelihood(
            points, values, (0.0, math.log(0.4)), nugget=1e-8
        )
        with use_backend(_WrappedBackend()):
            routed = gaussian_process_log_marginal_likelihood(
                points, values, (0.0, math.log(0.4)), nugget=1e-8
            )
        self.assertAlmostEqual(routed[0], reference[0], delta=1e-12)
        for left, right in zip(routed[1], reference[1], strict=True):
            self.assertAlmostEqual(left, right, delta=1e-10)

    def test_missing_numpy_raises_a_clear_import_error(self) -> None:
        """Blocking the import gives installation advice and no state change."""
        before = get_backend()
        with mock.patch.dict(sys.modules, {"numpy": None}):
            with self.assertRaisesRegex(ImportError, r"probstat-library\[numerics\]"):
                linalg.NumPyBackend()
            with self.assertRaises(ImportError):
                set_backend("numpy")
            self.assertIs(get_backend(), before)
            with self.assertRaises(ImportError):
                with use_backend("numpy"):
                    self.fail("the block must not run")
            self.assertIs(get_backend(), before)

    def test_library_runs_without_numpy_in_a_fresh_interpreter(self) -> None:
        """With NumPy unimportable the default paths work and NumPy stays unloaded."""
        script = "\n".join(
            [
                "import sys, importlib.abc",
                "class Block(importlib.abc.MetaPathFinder):",
                "    def find_spec(self, name, path=None, target=None):",
                "        if name == 'numpy' or name.startswith('numpy.'):",
                "            raise ImportError('numpy is blocked')",
                "        return None",
                "sys.meta_path.insert(0, Block())",
                "from probstat import linalg",
                "from probstat.multivariate import MultivariateNormal",
                "from probstat.surrogates import fit_gaussian_process",
                "law = MultivariateNormal((0.0, 0.0), ((2.0, 0.5), (0.5, 1.0)))",
                "law.log_pdf((0.1, 0.2)); law.sample(3, seed=1)",
                "fit_gaussian_process([(0.0,), (0.5,), (1.0,)], [0.0, 1.0, 0.5],",
                "                     optimize=False)",
                "try:",
                "    linalg.set_backend('numpy')",
                "except ImportError as error:",
                "    assert 'numerics' in str(error)",
                "    print('blocked', linalg.get_backend().name,",
                "          'numpy' in sys.modules)",
            ]
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(SOURCE_ROOT), environment.get("PYTHONPATH", "")]
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
            timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "blocked python False")


class ReferenceBackendTests(unittest.TestCase):
    """The reference backend is the existing routines, bit for bit."""

    def setUp(self) -> None:
        """Build a small SPD system.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

        """
        self.backend = PythonBackend()
        self.matrix = random_spd(12, seed=7)
        self.rhs = random_matrix(1, 12, seed=8)[0]

    def test_every_kernel_is_bit_identical_to_the_dense_routine(self) -> None:
        """Wrapping changes no bit of any result."""
        backend, matrix, rhs = self.backend, self.matrix, self.rhs
        lower = linalg.cholesky(matrix)
        self.assertEqual(backend.cholesky(matrix), lower)
        self.assertEqual(
            backend.solve_lower(lower, rhs), linalg.solve_lower(lower, rhs)
        )
        upper = linalg.transpose(lower)
        self.assertEqual(
            backend.solve_upper(upper, rhs), linalg.solve_upper(upper, rhs)
        )
        self.assertEqual(
            backend.cholesky_solve(lower, rhs), linalg.cholesky_solve(lower, rhs)
        )
        self.assertEqual(backend.solve_spd(matrix, rhs), linalg.solve_spd(matrix, rhs))
        self.assertEqual(backend.log_det_spd(matrix), linalg.log_det_spd(matrix))
        self.assertEqual(backend.inverse_spd(matrix), linalg.inverse_spd(matrix))
        self.assertEqual(backend.matmul(matrix, lower), linalg.matmul(matrix, lower))
        self.assertEqual(backend.matvec(matrix, rhs), linalg.matvec(matrix, rhs))
        self.assertEqual(
            backend.symmetric_eigen(matrix), linalg.symmetric_eigen(matrix)
        )
        factor = backend.factorize(matrix)
        self.assertIsInstance(factor, CholeskyFactorization)
        self.assertEqual(factor.lower, lower)
        self.assertEqual(factor.jitter, 0.0)
        self.assertEqual(factor.solve_lower(rhs), linalg.solve_lower(lower, rhs))
        self.assertEqual(factor.solve(rhs), linalg.cholesky_solve(lower, rhs))
        self.assertEqual(factor.log_det(), linalg.log_det_spd(matrix))
        self.assertEqual(factor.inverse(), linalg.inverse_spd(matrix))
        expected = tuple(
            math.fsum(lower[i][j] * rhs[j] for j in range(i + 1))
            for i in range(len(rhs))
        )
        self.assertEqual(factor.multiply_lower(rhs), expected)

    def test_jitter_is_explicit_and_never_escalated(self) -> None:
        """``jitter`` factorizes ``A + delta I``; a failure is an error."""
        backend, matrix = self.backend, self.matrix
        shifted = linalg.add(matrix, linalg.identity(len(matrix)), scale=0.25)
        self.assertEqual(
            backend.cholesky(matrix, jitter=0.25), linalg.cholesky(shifted)
        )
        self.assertEqual(backend.factorize(matrix, jitter=0.25).jitter, 0.25)
        indefinite = ((1.0, 2.0), (2.0, 1.0))
        with self.assertRaisesRegex(ValueError, "positive definite"):
            backend.cholesky(indefinite)
        with self.assertRaisesRegex(ValueError, "positive definite"):
            backend.cholesky(indefinite, jitter=0.5)
        self.assertEqual(len(backend.cholesky(indefinite, jitter=1.5)), 2)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            backend.cholesky(matrix, jitter=-1e-9)
        with self.assertRaises(ValueError):
            backend.factorize(matrix, jitter=math.nan)

    def test_default_consumers_are_unchanged(self) -> None:
        """MVN log-density and samples equal the direct-routine formulas."""
        law = MultivariateNormal((0.5,) * 12, self.matrix)
        lower = linalg.cholesky(law.covariance)
        point = tuple(0.5 + value for value in self.rhs)
        whitened = linalg.solve_lower(lower, self.rhs)
        expected = -0.5 * (
            12 * math.log(2.0 * math.pi)
            + 2.0 * math.fsum(math.log(lower[i][i]) for i in range(12))
            + math.fsum(value * value for value in whitened)
        )
        self.assertEqual(law.log_pdf(point), expected)
        generator = resolve_stream(11).generator()
        independent = tuple(generator.gauss(0.0, 1.0) for _ in range(12))
        draw = tuple(
            0.5 + math.fsum(lower[i][j] * independent[j] for j in range(i + 1))
            for i in range(12)
        )
        self.assertEqual(law.sample(1, seed=11), (draw,))


@unittest.skipUnless(HAS_NUMPY, "NumPy is optional")
class NumPyReferenceComparisonTests(unittest.TestCase):
    """NumPy against the reference backend within backward-error bounds."""

    SIZES = (5, 50, 100, 200)

    def setUp(self) -> None:
        """Create both backends.

        References:
            [REF-HARRIS-ETAL-2020]. No equation is evaluated.

        """
        self.python = PythonBackend()
        self.numpy = linalg.NumPyBackend()

    def condition(self, matrix: Matrix) -> tuple[float, float, float]:
        """Return ``(kappa_2(A), trace(A) / ||A||_2, ||A||_2)`` by LAPACK ``eigh``.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 2.6], [EQ-LINALG-012].

        """
        values, _ = self.numpy.symmetric_eigen(matrix)
        trace = math.fsum(matrix[i][i] for i in range(len(matrix)))
        return values[-1] / values[0], trace / values[-1], values[-1]

    def test_cholesky_backward_error_is_componentwise_small(self) -> None:
        """``|A - L L'| <= (gamma_(n+1) + 2u) |L||L'|`` for both factors."""
        for size in self.SIZES:
            matrix = random_spd(size, seed=size)
            for backend in (self.python, self.numpy):
                lower = backend.cholesky(matrix)
                worst = 0.0
                for i in range(size):
                    for j in range(i + 1):
                        terms = [lower[i][k] * lower[j][k] for k in range(j + 1)]
                        residual = abs(matrix[i][j] - math.fsum(terms))
                        scale = math.fsum(abs(term) for term in terms)
                        worst = max(worst, residual / scale)
                with self.subTest(size=size, backend=backend.name):
                    self.assertLessEqual(worst, gamma(size + 1) + 2 * UNIT_ROUNDOFF)

    def test_factors_solves_and_log_determinants_agree(self) -> None:
        """Forward differences stay below ``2 kappa_2 gamma_k trace / ||A||``."""
        for size in self.SIZES:
            matrix = random_spd(size, seed=100 + size)
            rhs = random_matrix(1, size, seed=200 + size)[0]
            kappa, trace_ratio, largest = self.condition(matrix)
            factor_bound = 2.0 * kappa * gamma(size + 1) * trace_ratio
            solve_bound = 2.0 * kappa * gamma(3 * size + 1) * trace_ratio
            reference = self.python.cholesky(matrix)
            candidate = self.numpy.cholesky(matrix)
            # ||L||_2 = sqrt(||A||_2).
            factor_error = frobenius(difference(candidate, reference)) / math.sqrt(
                largest
            )
            x_ref = self.python.solve_spd(matrix, rhs)
            x_new = self.numpy.solve_spd(matrix, rhs)
            solve_error = vector_norm(
                tuple(a - b for a, b in zip(x_new, x_ref, strict=True))
            ) / vector_norm(x_ref)
            logdet_error = abs(
                self.numpy.log_det_spd(matrix) - self.python.log_det_spd(matrix)
            )
            factor = self.numpy.factorize(matrix)
            lower_solve = factor.solve_lower(rhs)
            lower_ref = self.python.solve_lower(reference, rhs)
            lower_error = vector_norm(
                tuple(a - b for a, b in zip(lower_solve, lower_ref, strict=True))
            ) / vector_norm(lower_ref)
            with self.subTest(size=size, kappa=kappa):
                self.assertLessEqual(factor_error, factor_bound)
                self.assertLessEqual(solve_error, solve_bound)
                self.assertLessEqual(lower_error, solve_bound)
                self.assertLessEqual(logdet_error, size * factor_bound)
                self.assertAlmostEqual(factor.log_det(), self.numpy.log_det_spd(matrix))

    def test_inverse_and_triangular_kernels_agree(self) -> None:
        """Inverse, back substitution, and factor solves agree to the bound."""
        for size in (5, 50, 100):
            matrix = random_spd(size, seed=300 + size)
            rhs = random_matrix(1, size, seed=400 + size)[0]
            kappa, trace_ratio, _ = self.condition(matrix)
            bound = 2.0 * kappa * gamma(3 * size + 1) * trace_ratio
            reference = self.python.inverse_spd(matrix)
            candidate = self.numpy.inverse_spd(matrix)
            relative = frobenius(difference(candidate, reference)) / frobenius(
                reference
            )
            lower = self.python.cholesky(matrix)
            upper = linalg.transpose(lower)
            back = self.numpy.solve_upper(upper, rhs)
            back_ref = self.python.solve_upper(upper, rhs)
            factor_solve = self.numpy.cholesky_solve(lower, rhs)
            factor_ref = self.python.cholesky_solve(lower, rhs)
            from_factor = self.numpy.factorize(matrix).inverse()
            with self.subTest(size=size, kappa=kappa):
                self.assertLessEqual(relative, size * bound)
                self.assertLessEqual(
                    frobenius(difference(from_factor, candidate))
                    / frobenius(candidate),
                    size * bound,
                )
                for new, old in ((back, back_ref), (factor_solve, factor_ref)):
                    error = vector_norm(
                        tuple(a - b for a, b in zip(new, old, strict=True))
                    ) / vector_norm(old)
                    self.assertLessEqual(error, bound)

    def test_matrix_products_agree_componentwise(self) -> None:
        """``|fl(AB) - AB| <= gamma_k |A||B|`` bounds the backend difference."""
        for rows, inner, columns in ((3, 4, 5), (60, 200, 40), (200, 200, 1)):
            left = random_matrix(rows, inner, seed=rows)
            right = random_matrix(inner, columns, seed=columns + 1)
            reference = self.python.matmul(left, right)
            candidate = self.numpy.matmul(left, right)
            bound = gamma(inner) + 2.0 * UNIT_ROUNDOFF
            worst = 0.0
            for i in range(rows):
                for j in range(columns):
                    scale = math.fsum(
                        abs(left[i][k] * right[k][j]) for k in range(inner)
                    )
                    worst = max(worst, abs(candidate[i][j] - reference[i][j]) / scale)
            vector = tuple(row[0] for row in right)
            product = self.numpy.matvec(left, vector)
            with self.subTest(shape=(rows, inner, columns)):
                self.assertLessEqual(worst, bound)
                self.assertIsInstance(candidate, tuple)
                self.assertIsInstance(candidate[0], tuple)
                for i in range(rows):
                    self.assertAlmostEqual(
                        product[i],
                        reference[i][0],
                        delta=bound
                        * math.fsum(abs(left[i][k] * vector[k]) for k in range(inner)),
                    )
        with self.assertRaisesRegex(ValueError, "inner matrix dimensions"):
            self.numpy.matmul(((1.0, 2.0),), ((1.0, 2.0),))

    def test_symmetric_eigenvalues_agree_by_weyl(self) -> None:
        """Eigenvalues agree within the two backward errors (Weyl)."""
        for size in (5, 20, 40):
            matrix = random_spd(size, seed=500 + size)
            values_ref, _ = self.python.symmetric_eigen(matrix)
            values, vectors = self.numpy.symmetric_eigen(matrix)
            norm = frobenius(matrix)
            # Jacobi stops at an off-diagonal Frobenius norm of 1e-14 ||A||_F.
            bound = (1e-14 + 2.0 * gamma(size)) * norm
            with self.subTest(size=size):
                for new, old in zip(values, values_ref, strict=True):
                    self.assertAlmostEqual(new, old, delta=bound)
                for column in range(size):
                    image = self.numpy.matvec(
                        matrix, tuple(row[column] for row in vectors)
                    )
                    residual = vector_norm(
                        tuple(
                            image[i] - values[column] * vectors[i][column]
                            for i in range(size)
                        )
                    )
                    self.assertLessEqual(residual, bound * size)

    def test_error_behavior_matches_the_reference_backend(self) -> None:
        """Non-PD, singular, and misshapen inputs raise ``ValueError``."""
        indefinite = ((1.0, 2.0), (2.0, 1.0))
        for backend in (self.python, self.numpy):
            with self.subTest(backend=backend.name):
                with self.assertRaisesRegex(ValueError, "positive definite"):
                    backend.cholesky(indefinite)
                with self.assertRaisesRegex(ValueError, "positive definite"):
                    backend.factorize(indefinite, jitter=0.5)
                self.assertEqual(len(backend.cholesky(indefinite, jitter=1.5)), 2)
                with self.assertRaisesRegex(ValueError, "singular"):
                    backend.solve_lower(((0.0, 0.0), (1.0, 1.0)), (1.0, 1.0))
                with self.assertRaisesRegex(ValueError, "non-negative"):
                    backend.cholesky(((1.0,),), jitter=-1.0)
        with self.assertRaisesRegex(ValueError, "square"):
            self.numpy.cholesky(((1.0, 2.0),))
        with self.assertRaisesRegex(ValueError, "positive definite"):
            self.numpy.cholesky(((math.nan, 0.0), (0.0, 1.0)))
        with self.assertRaisesRegex(ValueError, "length"):
            self.numpy.factorize(((2.0, 0.0), (0.0, 2.0))).solve((1.0,))

    def test_multivariate_normal_density_and_samples_agree(self) -> None:
        """A 200-dimensional MVN agrees in log-density and draws."""
        size = 200
        covariance = random_spd(size, seed=600)
        mean = random_matrix(1, size, seed=601)[0]
        point = random_matrix(1, size, seed=602)[0]
        kappa, trace_ratio, _ = self.condition(covariance)
        bound = 2.0 * kappa * gamma(3 * size + 1) * trace_ratio
        reference = MultivariateNormal(mean, covariance)
        with use_backend("numpy"):
            candidate = MultivariateNormal(mean, covariance)
            draws = candidate.sample(5, seed=9)
        quadratic = reference.mahalanobis_squared(point)
        self.assertLessEqual(
            abs(candidate.log_pdf(point) - reference.log_pdf(point)),
            bound * (size + quadratic),
        )
        self.assertLessEqual(
            abs(candidate.mahalanobis_squared(point) - quadratic), bound * quadratic
        )
        lower = reference._cholesky
        for new, old in zip(draws, reference.sample(5, seed=9), strict=True):
            self.assertIsInstance(new, tuple)
            spread = max(
                math.fsum(abs(lower[i][k]) for k in range(i + 1)) for i in range(size)
            )
            worst = max(abs(a - b) for a, b in zip(new, old, strict=True))
            # |z| <= 6 for these draws; the transform differs by gamma_n |L||z|.
            self.assertLessEqual(worst, 6.0 * spread * (gamma(size) + bound))
        conditional = candidate.conditional((0, 1, 2), point[:3])
        expected = reference.conditional((0, 1, 2), point[:3])
        error = max(
            abs(a - b) for a, b in zip(conditional.mean, expected.mean, strict=True)
        )
        self.assertLessEqual(error, bound * max(abs(value) for value in expected.mean))

    def test_gaussian_process_fit_on_300_points_agrees(self) -> None:
        """Likelihood, gradient, predictions, LOO, and conditioning agree."""
        generator = random.Random(2026)
        count = 300
        points = [(generator.random(), generator.random()) for _ in range(count)]
        values = [
            math.sin(6.0 * a) + b * b + 0.05 * generator.gauss(0.0, 1.0)
            for a, b in points
        ]
        theta = (0.0, math.log(0.3), math.log(0.3))
        options = {"optimize": False, "nugget": 1e-3, "initial_log_parameters": theta}
        reference = fit_gaussian_process(points, values, **options)  # type: ignore[arg-type]
        with use_backend("numpy"):
            candidate = fit_gaussian_process(points, values, **options)  # type: ignore[arg-type]
            gradient = gaussian_process_log_marginal_likelihood(
                points,
                [(v - reference.output_mean) / reference.output_scale for v in values],
                theta,
                nugget=1e-3,
            )[1]
        gradient_ref = gaussian_process_log_marginal_likelihood(
            points,
            [(v - reference.output_mean) / reference.output_scale for v in values],
            theta,
            nugget=1e-3,
        )[1]
        kappa = reference.condition_number
        relative = 2.0 * kappa * count * gamma(3 * count + 1)
        signal = math.exp(theta[0])
        scales = reference.hyperparameters.length_scales
        noise = reference.standardized_noise_variance
        kernel = tuple(
            tuple(
                kernel_value(reference.kernel, p, q, scales, signal)
                + (noise if i == j else 0.0)
                for j, q in enumerate(points)
            )
            for i, p in enumerate(points)
        )
        targets = tuple(
            (v - reference.output_mean) / reference.output_scale for v in values
        )
        alpha_norm = vector_norm(self.numpy.solve_spd(kernel, targets))
        target_norm = vector_norm(targets)
        self.assertLessEqual(
            abs(candidate.log_marginal_likelihood - reference.log_marginal_likelihood),
            relative * (count + target_norm * alpha_norm),
        )
        gradient_scale = max(abs(g) for g in gradient_ref) + count
        for new, old in zip(gradient, gradient_ref, strict=True):
            self.assertLessEqual(abs(new - old), relative * gradient_scale * count)
        self.assertLessEqual(
            abs(candidate.condition_number - kappa), relative * kappa
        )
        scale = reference.output_scale
        mean_bound = scale * math.sqrt(count) * signal * alpha_norm * relative
        variance_bound = 2.0 * scale * scale * signal * relative
        test_points = [(generator.random(), generator.random()) for _ in range(25)]
        for x in test_points:
            new_mean, new_var = candidate.predict_with_variance(x)
            old_mean, old_var = reference.predict_with_variance(x)
            with self.subTest(x=x):
                self.assertLessEqual(abs(new_mean - old_mean), mean_bound)
                self.assertLessEqual(abs(new_var - old_var), variance_bound)
        loo_new = candidate.leave_one_out()
        loo_old = reference.leave_one_out()
        self.assertAlmostEqual(
            loo_new.relative_error, loo_old.relative_error, delta=1e-6
        )

    def test_gaussian_process_optimized_fit_agrees(self) -> None:
        """BFGS from the same starts reaches the same optimum on both backends.

        Rounding differences change the BFGS path, so agreement is set by
        the optimizer's gradient tolerance (``1e-6``), not by Higham's
        bounds: ``1e-4`` in log hyperparameters and ``1e-7`` relative in the
        log marginal likelihood.
        """
        generator = random.Random(3)
        points = [(generator.random(), generator.random()) for _ in range(60)]
        values = [
            math.sin(6.0 * a) + b * b + 0.1 * generator.gauss(0.0, 1.0)
            for a, b in points
        ]
        reference = fit_gaussian_process(
            points, values, restarts=2, seed=4, estimate_noise=True
        )
        with use_backend("numpy"):
            candidate = fit_gaussian_process(
                points, values, restarts=2, seed=4, estimate_noise=True
            )
        self.assertTrue(reference.converged and candidate.converged)
        for new, old in zip(
            candidate.log_parameters, reference.log_parameters, strict=True
        ):
            self.assertAlmostEqual(new, old, delta=1e-4)
        self.assertAlmostEqual(
            candidate.log_marginal_likelihood,
            reference.log_marginal_likelihood,
            delta=1e-7 * abs(reference.log_marginal_likelihood),
        )
        self.assertAlmostEqual(
            candidate.predict((0.2, 0.3)), reference.predict((0.2, 0.3)), delta=1e-5
        )


if __name__ == "__main__":
    unittest.main()
