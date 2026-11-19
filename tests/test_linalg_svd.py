"""Tests for the one-sided Jacobi singular value decomposition (ADR-0110).

Reference values from R 4.6.2 (``svd`` calls LAPACK ``dgesdd``) were computed
offline and are hard-coded with the R expression that produced them; the
tests never call R. Singular vectors are compared up to a simultaneous sign
flip of ``u_i`` and ``v_i``, and floating-point values with relative
tolerances, never bit for bit.
"""

from __future__ import annotations

import math
import random
import unittest

from probstat.linalg import (
    SingularValueDecomposition,
    matmul,
    numerical_rank,
    pseudo_inverse,
    singular_value_decomposition,
    symmetric_eigen,
    transpose,
)

# R: A <- matrix(c(4, 1, -2, 2, 1, 3, 0, -1, 2, 0, 5, 1), nrow = 4)
A_ROWS = ((4.0, 1.0, 2.0), (1.0, 3.0, 0.0), (-2.0, 0.0, 5.0), (2.0, -1.0, 1.0))
# R: svd(A)$d
A_SINGULAR = (5.4836775158377096, 5.1551471053462583, 3.0583883377583163)
# R: svd(A)$u, by columns
A_U_COLUMNS = (
    (
        -0.42638303462784638,
        -0.05115363573111309,
        -0.8820246008917052,
        -0.19394178702606737,
    ),
    (
        0.76137573920978974,
        0.35853222068438895,
        -0.45340480952231155,
        0.29357402673925836,
    ),
    (
        -0.1167113892492152,
        0.83394245956253132,
        0.12350169531604514,
        -0.52503881476741932,
    ),
)
# R: svd(A)$v, by columns
A_V_COLUMNS = (
    (-0.069391415781147298, -0.070372875443636726, -0.9951042607775743),
    (0.950116989865744, 0.29938978325624965, -0.087426902325505595),
    (-0.30407653145934427, 0.95153214138229203, -0.046087383654143055),
)
# R: C <- matrix(c(2, -1, 0, 3, 1, 0, 4, 1, -2, 2), nrow = 2, byrow = TRUE)
C_ROWS = ((2.0, -1.0, 0.0, 3.0, 1.0), (0.0, 4.0, 1.0, -2.0, 2.0))
# R: svd(C)$d, svd(C)$u and svd(C)$v by columns
C_SINGULAR = (5.4253093121089977, 3.2505413192179851)
C_U_COLUMNS = (
    (-0.48476853239294526, 0.87464248124681765),
    (0.87464248124681765, 0.48476853239294521),
)
C_V_COLUMNS = (
    (
        -0.17870632050820331,
        0.73421407485276824,
        0.16121522864966664,
        -0.5904899380616383,
        0.23307729704523159,
    ),
    (
        0.53815189247139861,
        0.32746288811398472,
        0.14913470858742103,
        0.50895842153225612,
        0.56734536341054143,
    ),
)
# R: B <- matrix(1:9, 3, byrow = TRUE); svd(B)$d (the third is rounding noise)
B_ROWS = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
B_SINGULAR = (16.84810335261421, 1.0683695145547101)
# R: H <- 1 / (outer(1:6, 1:6, "+") - 1); svd(H)$d
HILBERT_SINGULAR = (
    1.6188998589243389,
    0.24236087057520955,
    0.016321521319875788,
    0.00061574835418264095,
    1.2570757122643422e-05,
    1.0827994845480084e-07,
)


def random_matrix(rows: int, columns: int, seed: int) -> tuple[tuple[float, ...], ...]:
    """Return a seeded standard-normal matrix.

    Args:
        rows: Row count.
        columns: Column count.
        seed: Generator seed.

    Returns:
        The matrix as row tuples.

    References:
        [REF-MATSUMOTO-NISHIMURA-1998]. No equation is evaluated.

    """
    generator = random.Random(seed)
    return tuple(
        tuple(generator.gauss(0.0, 1.0) for _ in range(columns)) for _ in range(rows)
    )


def column(matrix: tuple[tuple[float, ...], ...], index: int) -> tuple[float, ...]:
    """Return one column of a row-tuple matrix.

    Args:
        matrix: Matrix as row tuples.
        index: Column index.

    Returns:
        The column.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No equation is evaluated.

    """
    return tuple(row[index] for row in matrix)


class SingularValueDecompositionTests(unittest.TestCase):
    """Factor identities, R agreement, rank deficiency, and validation."""

    def assert_orthonormal_columns(
        self, matrix: tuple[tuple[float, ...], ...], tolerance: float = 1e-13
    ) -> None:
        """Assert ``M'M = I`` to an absolute tolerance.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 2.4]. No equation is evaluated.

        """
        gram = matmul(transpose(matrix), matrix)
        for i, row in enumerate(gram):
            for j, value in enumerate(row):
                self.assertAlmostEqual(value, 1.0 if i == j else 0.0, delta=tolerance)

    def assert_valid(
        self,
        matrix: tuple[tuple[float, ...], ...],
        result: SingularValueDecomposition,
        tolerance: float = 1e-13,
    ) -> None:
        """Assert reconstruction, orthonormality, order, and non-negativity.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 8.6]. No equation is evaluated.

        """
        rows, columns = len(matrix), len(matrix[0])
        k = min(rows, columns)
        self.assertEqual(result.shape, (rows, columns))
        self.assertEqual(len(result.singular_values), k)
        self.assertEqual(len(result.u[0]), k)
        self.assertEqual(len(result.v[0]), k)
        scale = max(1.0, result.singular_values[0])
        rebuilt = result.reconstruct()
        error = max(
            abs(rebuilt[i][j] - matrix[i][j])
            for i in range(rows)
            for j in range(columns)
        )
        self.assertLess(error, tolerance * scale)
        self.assert_orthonormal_columns(result.u)
        self.assert_orthonormal_columns(result.v)
        values = result.singular_values
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:], strict=False)))
        self.assertTrue(all(value >= 0.0 for value in values))

    def assert_vectors_match_up_to_sign(
        self,
        result: SingularValueDecomposition,
        u_columns: tuple[tuple[float, ...], ...],
        v_columns: tuple[tuple[float, ...], ...],
    ) -> None:
        """Assert singular vectors agree with a reference up to a joint sign.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 2.4]. No equation is evaluated.

        """
        for index, (u_ref, v_ref) in enumerate(zip(u_columns, v_columns, strict=True)):
            u = column(result.u, index)
            v = column(result.v, index)
            sign = math.copysign(
                1.0, math.fsum(a * b for a, b in zip(u, u_ref, strict=False))
            )
            for a, b in zip(u, u_ref, strict=True):
                self.assertAlmostEqual(sign * a, b, delta=1e-13)
            for a, b in zip(v, v_ref, strict=True):
                self.assertAlmostEqual(sign * a, b, delta=1e-13)

    def test_tall_matrix_matches_r(self) -> None:
        """A 4 x 3 matrix agrees with R's ``svd`` in values and vectors."""
        result = singular_value_decomposition(A_ROWS)
        self.assert_valid(A_ROWS, result)
        for value, expected in zip(result.singular_values, A_SINGULAR, strict=True):
            self.assertAlmostEqual(value, expected, delta=1e-13 * expected)
        self.assert_vectors_match_up_to_sign(result, A_U_COLUMNS, A_V_COLUMNS)

    def test_wide_matrix_matches_r(self) -> None:
        """A 2 x 5 matrix is decomposed through its transpose."""
        result = singular_value_decomposition(C_ROWS)
        self.assertEqual(result.shape, (2, 5))
        self.assert_valid(C_ROWS, result)
        for value, expected in zip(result.singular_values, C_SINGULAR, strict=True):
            self.assertAlmostEqual(value, expected, delta=1e-13 * expected)
        self.assert_vectors_match_up_to_sign(result, C_U_COLUMNS, C_V_COLUMNS)

    def test_hilbert_singular_values_match_r(self) -> None:
        """The 6 x 6 Hilbert matrix (condition 1.5e7) agrees with R.

        One-sided Jacobi has high relative accuracy for this graded matrix, so
        every singular value matches to ``1e-9`` relative.
        """
        hilbert = tuple(tuple(1.0 / (i + j + 1) for j in range(6)) for i in range(6))
        result = singular_value_decomposition(hilbert)
        self.assert_valid(hilbert, result)
        for value, expected in zip(
            result.singular_values, HILBERT_SINGULAR, strict=True
        ):
            self.assertAlmostEqual(value, expected, delta=1e-9 * expected)

    def test_squared_singular_values_equal_gram_eigenvalues(self) -> None:
        """``sigma_i^2`` are the eigenvalues of ``A'A`` (descending)."""
        for rows, columns, seed in ((7, 4, 1), (5, 5, 2), (3, 6, 3)):
            with self.subTest(rows=rows, columns=columns):
                matrix = random_matrix(rows, columns, seed)
                result = singular_value_decomposition(matrix)
                self.assert_valid(matrix, result)
                gram = matmul(transpose(matrix), matrix)
                eigenvalues, _ = symmetric_eigen(gram)
                descending = sorted(eigenvalues, reverse=True)
                scale = descending[0]
                for sigma, eigenvalue in zip(
                    result.singular_values, descending, strict=False
                ):
                    self.assertAlmostEqual(
                        sigma * sigma, eigenvalue, delta=1e-12 * scale
                    )

    def test_larger_random_matrices(self) -> None:
        """Reconstruction and orthonormality hold for 30 x 12 and 12 x 30."""
        for rows, columns in ((30, 12), (12, 30), (16, 16)):
            with self.subTest(rows=rows, columns=columns):
                matrix = random_matrix(rows, columns, rows * columns)
                self.assert_valid(matrix, singular_value_decomposition(matrix))

    def test_rank_deficient_matrix(self) -> None:
        """``1..9`` has rank two; its pseudo-inverse is exact in rationals."""
        result = singular_value_decomposition(B_ROWS)
        self.assert_valid(B_ROWS, result)
        for value, expected in zip(result.singular_values, B_SINGULAR, strict=False):
            self.assertAlmostEqual(value, expected, delta=1e-13 * expected)
        self.assertLess(result.singular_values[2], 1e-14)
        self.assertEqual(result.rank(), 2)
        self.assertEqual(numerical_rank(B_ROWS), 2)
        self.assertEqual(result.condition_number > 1e15, True)
        # R: MASS::ginv(B) = [[-23, -6, 11], [-2, 0, 2], [19, 6, -7]] / 36
        expected = ((-23.0, -6.0, 11.0), (-2.0, 0.0, 2.0), (19.0, 6.0, -7.0))
        inverse = pseudo_inverse(B_ROWS)
        for row, expected_row in zip(inverse, expected, strict=True):
            for value, numerator in zip(row, expected_row, strict=True):
                self.assertAlmostEqual(value, numerator / 36.0, delta=1e-13)

    def test_exactly_rank_deficient_tall_matrix(self) -> None:
        """Duplicate columns and a zero row give an exact zero and a full basis."""
        matrix = ((1.0, 1.0, 0.0), (2.0, 2.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        result = singular_value_decomposition(matrix)
        self.assert_valid(matrix, result)
        self.assertLess(result.singular_values[2], 1e-15)
        self.assertEqual(result.rank(), 2)
        self.assertEqual(result.rank(tolerance=0.9), 1)

    def test_zero_matrix(self) -> None:
        """The zero matrix has zero singular values and orthonormal factors."""
        zero = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        result = singular_value_decomposition(zero)
        self.assertEqual(result.singular_values, (0.0, 0.0))
        self.assert_orthonormal_columns(result.u)
        self.assert_orthonormal_columns(result.v)
        self.assertEqual(result.rank(), 0)
        self.assertEqual(result.condition_number, math.inf)
        self.assertEqual(pseudo_inverse(zero), ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))

    def test_pseudo_inverse_penrose_conditions(self) -> None:
        """``A A^+ A = A``, ``A^+ A A^+ = A^+``, and both products are symmetric."""
        for matrix in (A_ROWS, C_ROWS, random_matrix(6, 4, 9)):
            inverse = pseudo_inverse(matrix)
            left = matmul(matrix, inverse)
            right = matmul(inverse, matrix)
            first = matmul(left, matrix)
            second = matmul(right, inverse)
            for product, target in ((first, matrix), (second, inverse)):
                for row, target_row in zip(product, target, strict=True):
                    for value, expected in zip(row, target_row, strict=True):
                        self.assertAlmostEqual(value, expected, delta=1e-12)
            for square in (left, right):
                for i, row in enumerate(square):
                    for j, value in enumerate(row):
                        self.assertAlmostEqual(value, square[j][i], delta=1e-12)

    def test_pseudo_inverse_of_full_rank_square_is_inverse(self) -> None:
        """For a nonsingular matrix the pseudo-inverse is the inverse."""
        matrix = ((2.0, 1.0), (1.0, 3.0))
        inverse = pseudo_inverse(matrix)
        expected = ((0.6, -0.2), (-0.2, 0.4))
        for row, expected_row in zip(inverse, expected, strict=True):
            for value, target in zip(row, expected_row, strict=True):
                self.assertAlmostEqual(value, target, delta=1e-15)

    def test_rcond_truncates(self) -> None:
        """A relative cut-off above ``sigma_2 / sigma_1`` keeps one triplet."""
        matrix = ((3.0, 0.0), (0.0, 1e-3))
        self.assertEqual(numerical_rank(matrix, tolerance=1e-2), 1)
        truncated = pseudo_inverse(matrix, rcond=1e-2)
        self.assertAlmostEqual(truncated[0][0], 1.0 / 3.0, delta=1e-15)
        self.assertEqual(truncated[1][1], 0.0)
        self.assertAlmostEqual(pseudo_inverse(matrix)[1][1], 1e3, delta=1e-9)

    def test_condition_number(self) -> None:
        """``sigma_1 / sigma_k`` for a diagonal matrix."""
        result = singular_value_decomposition(((4.0, 0.0), (0.0, -0.5)))
        self.assertAlmostEqual(result.condition_number, 8.0, delta=1e-14)

    def test_invalid_arguments(self) -> None:
        """Bad shapes, values, and options raise explicit errors."""
        with self.assertRaises(ValueError):
            singular_value_decomposition(())
        with self.assertRaises(ValueError):
            singular_value_decomposition(((1.0, 2.0), (3.0,)))
        with self.assertRaises(ValueError):
            singular_value_decomposition(((math.nan, 1.0),))
        with self.assertRaises(ValueError):
            singular_value_decomposition(((1.0,),), tolerance=0.0)
        with self.assertRaises(ValueError):
            singular_value_decomposition(((1.0,),), max_sweeps=0)
        with self.assertRaises(TypeError):
            singular_value_decomposition(((1.0,),), max_sweeps=2.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            numerical_rank(((1.0,),), tolerance=1.0)
        with self.assertRaises(ValueError):
            pseudo_inverse(((1.0,),), rcond=-1e-3)
        with self.assertRaises(ValueError):
            singular_value_decomposition(random_matrix(8, 8, 4), max_sweeps=1)


if __name__ == "__main__":
    unittest.main()
