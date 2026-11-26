"""Tests for the matrix-free LIS and resolution probes (EQ-INV-LIS-014..019)."""

from __future__ import annotations

import math
import os
import unittest

from probstat.inverse.function_space import HatBasis
from probstat.inverse.function_space_extended import (
    averaging_kernel,
    likelihood_informed_subspace,
    local_average_functional,
    point_functional,
    resolution_operator,
)
from probstat.inverse.function_space_matrix_free import (
    jvp_vjp_operator,
    matrix_free_lis,
    probe_resolution,
    subspace_angles,
)
from probstat.linalg import (
    ComposedOperator,
    Convolution1DOperator,
    DenseCovariance,
    DenseOperator,
    DiagonalCovariance,
    DiagonalOperator,
    IdentityOperator,
    LinearOperator,
    SelectionOperator,
    symmetric_eigen,
)


def _kernel(sigma: float) -> list[float]:
    """Return a normalized Gaussian kernel truncated at three sigma."""
    half = int(3 * sigma)
    taps = [math.exp(-0.5 * (k / sigma) ** 2) for k in range(-half, half + 1)]
    total = sum(taps)
    return [t / total for t in taps]


def _conv_select(n: int, m: int, spacing: int, sigma: float) -> LinearOperator:
    """Return m point samples (spacing apart, centred) of a smoothed field."""
    start = n // 2 - (m * spacing) // 2
    indices = [start + spacing * i for i in range(m)]
    conv = Convolution1DOperator(_kernel(sigma), n, boundary="zero")
    return ComposedOperator([SelectionOperator(n, indices), conv])


def _unit(size: int, index: int) -> tuple[float, ...]:
    """Return a unit vector."""
    return tuple(1.0 if i == index else 0.0 for i in range(size))


def _dual_eigenvalues(
    forward: LinearOperator, factor: LinearOperator, noise_sd: float
) -> list[float]:
    """Nonzero spectrum of H through the data-space matrix K K' (independent).

    ``H = K'K`` and ``K K'`` share their nonzero eigenvalues; ``K K'`` is only
    ``m x m`` and is built by operator actions, so it works at any ``n``.
    """
    m = forward.shape[0]
    columns = [
        forward.matvec(factor.matvec(factor.rmatvec(forward.rmatvec(_unit(m, k)))))
        for k in range(m)
    ]
    gram = tuple(
        tuple(0.5 * (columns[k][j] + columns[j][k]) / noise_sd**2 for k in range(m))
        for j in range(m)
    )
    values, _ = symmetric_eigen(gram)
    return sorted(values, reverse=True)


class SubspaceAngleTests(unittest.TestCase):
    """Principal angles from sines."""

    def test_known_angles(self) -> None:
        """Identical, rotated by 1e-6 and orthogonal subspaces."""
        e1, e2, e3 = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        self.assertEqual(subspace_angles([e1, e2], [e1, e2]), (0.0, 0.0))
        theta = 1e-6
        rotated = (math.cos(theta), math.sin(theta), 0.0)
        angle = subspace_angles([e1], [rotated])[0]
        self.assertAlmostEqual(angle / theta, 1.0, places=8)
        angles = subspace_angles([e1], [e1, e2])
        self.assertAlmostEqual(angles[0], 0.0, places=12)
        self.assertAlmostEqual(angles[1], math.pi / 2, places=12)
        self.assertAlmostEqual(subspace_angles([e1], [e3])[0], math.pi / 2, places=12)

    def test_rejects_bad_bases(self) -> None:
        """Empty, ragged and non-orthonormal bases are errors."""
        with self.assertRaises(ValueError):
            subspace_angles([], [(1.0,)])
        with self.assertRaises(ValueError):
            subspace_angles([(1.0, 0.0)], [(1.0, 0.0, 0.0)])
        with self.assertRaises(ValueError):
            subspace_angles([(1.0, 1.0)], [(1.0, 0.0)])


class MatrixFreeLISTests(unittest.TestCase):
    """Matrix-free likelihood-informed subspace."""

    def test_closed_form_diagonal(self) -> None:
        """J = diag(a), P = I: H = diag(a^2 / s^2), directions are unit vectors."""
        a = (4.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05)
        sd = 0.5
        n = len(a)
        exact = sorted((x * x / sd**2 for x in a), reverse=True)
        for method in ("lanczos", "randomized"):
            with self.subTest(method=method):
                result = matrix_free_lis(
                    [DiagonalOperator(a)],
                    noise=DiagonalCovariance([sd**2] * n),
                    prior_factor=IdentityOperator(n),
                    threshold=10.0,
                    method=method,  # type: ignore[arg-type]
                )
                self.assertTrue(result.converged)
                self.assertEqual(result.rank, sum(1 for e in exact if e > 10.0))
                for found, want in zip(result.eigenvalues, exact, strict=False):
                    self.assertAlmostEqual(found / want, 1.0, places=9)
                order = sorted(range(n), key=lambda i: -a[i])
                for direction, index in zip(result.basis, order, strict=False):
                    self.assertAlmostEqual(abs(direction[index]), 1.0, places=8)
                self.assertLessEqual(result.max_retained_residual, 1e-8 * exact[0])
                self.assertEqual(result.point_count, 1)
                self.assertGreater(result.hessian_applications, 0)
                self.assertEqual(
                    result.jacobian_counts.forward, result.jacobian_counts.transpose
                )

    def test_matches_dense_two_points(self) -> None:
        """Eigenvalues, error bounds and subspace angle against the dense LIS."""
        n, m = 40, 10
        sd = 0.1
        jac = [_conv_select(n, m, 2, 2.0), _conv_select(n, m, 2, 2.5)]
        weights = (0.4, 0.6)
        factor = Convolution1DOperator([0.25, 0.5, 0.25], n, boundary="zero")
        noise = DiagonalCovariance([sd**2] * m)
        full = likelihood_informed_subspace(
            jac,
            noise=noise,
            prior_factor=factor,
            threshold=1e300,
            weights=weights,
        )
        values = full.eigenvalues
        k = 5
        threshold = math.sqrt(values[k - 1] * values[k])
        dense = likelihood_informed_subspace(
            jac, noise=noise, prior_factor=factor, threshold=threshold, weights=weights
        )
        self.assertEqual(dense.rank, k)
        for method in ("lanczos", "randomized"):
            with self.subTest(method=method):
                result = matrix_free_lis(
                    jac,
                    noise=noise,
                    prior_factor=factor,
                    threshold=threshold,
                    weights=weights,
                    method=method,  # type: ignore[arg-type]
                )
                self.assertTrue(result.converged, result.stop_reason)
                self.assertEqual(result.rank, k)
                for found, want in zip(result.eigenvalues[:k], values[:k], strict=True):
                    self.assertLessEqual(abs(found - want), 1e-8 * values[0])
                # a dense eigenvalue lies within each reported residual
                for found, res in zip(
                    result.eigenvalues, result.residual_norms, strict=True
                ):
                    nearest = min(abs(found - v) for v in values)
                    self.assertLessEqual(nearest, res + 1e-9 * values[0])
                angles = subspace_angles(dense.basis, result.basis)
                self.assertLess(max(angles), 1e-7)
                self.assertGreaterEqual(result.gap, 0.0)

    def test_jvp_vjp_pair(self) -> None:
        """A callback pair gives the same subspace as the operator."""
        a = (3.0, 2.0, 1.0, 0.1)
        n = len(a)
        pair = jvp_vjp_operator(
            lambda v: tuple(x * y for x, y in zip(a, v, strict=True)),
            lambda u: tuple(x * y for x, y in zip(a, u, strict=True)),
            (n, n),
        )
        noise = DiagonalCovariance([1.0] * n)
        left = matrix_free_lis(
            [pair],
            noise=noise,
            prior_factor=IdentityOperator(n),
            threshold=0.5,
            method="lanczos",
        )
        right = matrix_free_lis(
            [DiagonalOperator(a)],
            noise=noise,
            prior_factor=IdentityOperator(n),
            threshold=0.5,
            method="lanczos",
        )
        self.assertEqual(left.rank, right.rank)
        self.assertEqual(left.rank, 3)
        for x, y in zip(left.eigenvalues, right.eigenvalues, strict=True):
            self.assertAlmostEqual(x, y, places=10)

    def test_rank_cap(self) -> None:
        """max_rank removes directions above the threshold and says so."""
        a = (5.0, 4.0, 3.0, 2.0, 0.01, 0.01)
        n = len(a)
        result = matrix_free_lis(
            [DiagonalOperator(a)],
            noise=DiagonalCovariance([1.0] * n),
            prior_factor=IdentityOperator(n),
            threshold=1.0,
            max_rank=2,
            method="lanczos",
        )
        self.assertEqual(result.rank, 2)
        self.assertTrue(result.capped)
        self.assertTrue(result.converged)

    def test_beyond_dense_limit(self) -> None:
        """Size 450: the dense routine refuses; the matrix-free one is checked."""
        n, m, sd = 450, 24, 0.05
        forward = _conv_select(n, m, 3, 5.0)
        factor = DiagonalOperator([1.0] * n)
        noise = DiagonalCovariance([sd**2] * m)
        with self.assertRaises(ValueError):
            likelihood_informed_subspace(
                [forward], noise=noise, prior_factor=factor, threshold=1.0
            )
        oracle = _dual_eigenvalues(forward, factor, sd)
        above = sum(1 for v in oracle if v > 1.0)
        self.assertEqual(above, 11)
        for method in ("lanczos",):
            with self.subTest(method=method):
                result = matrix_free_lis(
                    [forward],
                    noise=noise,
                    prior_factor=factor,
                    threshold=1.0,
                    method=method,  # type: ignore[arg-type]
                )
                self.assertTrue(result.converged, result.stop_reason)
                self.assertEqual(result.rank, above)
                for found, want in zip(
                    result.eigenvalues[:above], oracle, strict=False
                ):
                    self.assertLessEqual(abs(found - want), 1e-7 * oracle[0])
                self.assertLessEqual(result.max_retained_residual, 1e-8 * oracle[0])
                # one forward and one transpose Jacobian action per H action
                self.assertEqual(
                    result.jacobian_counts.forward, result.hessian_applications
                )
                self.assertEqual(
                    result.jacobian_counts.transpose, result.hessian_applications
                )
                self.assertGreater(result.seconds, 0.0)

    def test_circulant_multiplicity(self) -> None:
        """Circulant eigenvalues come in pairs; the randomized method counts both."""
        n, sd = 60, 0.05
        taps = _kernel(3.0)
        half = len(taps) // 2
        conv = Convolution1DOperator(taps, n, boundary="periodic")
        noise = DiagonalCovariance([sd**2] * n)
        factor = IdentityOperator(n)
        # circulant: lambda_j = |sum_l k_l cos(w_j l)|^2 / sd^2, j and n-j equal
        spectrum = [
            math.fsum(
                t * math.cos(2 * math.pi * j * (i - half) / n)
                for i, t in enumerate(taps)
            )
            ** 2
            / sd**2
            for j in range(n)
        ]
        threshold = 100.0
        truth = sum(1 for v in spectrum if v > threshold)
        randomized = matrix_free_lis(
            [conv],
            noise=noise,
            prior_factor=factor,
            threshold=threshold,
            method="randomized",
        )
        self.assertTrue(randomized.converged)
        self.assertEqual(randomized.rank, truth)
        # distinct eigenvalues are found by both methods; Lanczos multiplicity
        # is not asserted (exact arithmetic gives one copy, rounding may add more)
        lanczos = matrix_free_lis(
            [conv],
            noise=noise,
            prior_factor=factor,
            threshold=threshold,
            method="lanczos",
        )
        self.assertTrue(lanczos.converged)
        self.assertGreaterEqual(lanczos.rank, (truth + 1) // 2)
        self.assertAlmostEqual(
            lanczos.eigenvalues[0], randomized.eigenvalues[0], delta=1e-7
        )

    def test_unresolved_rank_is_reported(self) -> None:
        """A too-small subspace budget yields converged = False, not a guess."""
        n = 60
        a = tuple(1.0 - 0.005 * i for i in range(n))
        result = matrix_free_lis(
            [DiagonalOperator(a)],
            noise=DiagonalCovariance([1.0] * n),
            prior_factor=IdentityOperator(n),
            threshold=0.5,
            method="lanczos",
            initial_size=4,
            max_size=8,
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.stop_reason, "max_steps")
        self.assertFalse(result.bracketed)
        self.assertGreater(max(result.residual_norms), 1e-6)

    def test_validation(self) -> None:
        """Invalid arguments are rejected."""
        n = 4
        ok = DiagonalOperator([1.0] * n)
        noise = DiagonalCovariance([1.0] * n)
        with self.assertRaises(ValueError):
            matrix_free_lis([], noise=noise, prior_factor=ok, threshold=1.0)
        with self.assertRaises(ValueError):
            matrix_free_lis([ok], noise=noise, prior_factor=ok, threshold=0.0)
        with self.assertRaises(ValueError):
            matrix_free_lis(
                [ok],
                noise=noise,
                prior_factor=ok,
                threshold=1.0,
                method="bogus",  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            matrix_free_lis(
                [ok],
                noise=DiagonalCovariance([1.0] * 3),
                prior_factor=ok,
                threshold=1.0,
            )
        with self.assertRaises(ValueError):
            matrix_free_lis(
                [ok], noise=noise, prior_factor=ok, threshold=1.0, weights=(-1.0,)
            )
        with self.assertRaises(TypeError):
            matrix_free_lis(
                [ok],
                noise=noise,
                prior_factor=ok,
                threshold=1.0,
                max_rank=1.5,  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            matrix_free_lis(
                [ok, DiagonalOperator([1.0] * 3)],
                noise=noise,
                prior_factor=ok,
                threshold=1.0,
            )


class ResolutionProbeTests(unittest.TestCase):
    """Resolution columns, rows, averaging kernels and trace."""

    def _small(
        self,
    ) -> tuple[HatBasis, DenseOperator, DiagonalCovariance, DenseCovariance]:
        """Return the small hat-basis problem used for dense comparison."""
        n = 12
        basis = HatBasis(n, 1.0)
        g_op = basis.evaluation_operator((0.15, 0.4, 0.5, 0.83))
        noise = DiagonalCovariance((0.01, 0.02, 0.01, 0.05))
        c = tuple(tuple(1.5 * 0.5 ** abs(i - j) for j in range(n)) for i in range(n))
        return basis, g_op, noise, DenseCovariance(c)

    def test_matches_dense(self) -> None:
        """Columns, rows, kernels, integrals and trace against the dense routine."""
        basis, g_op, noise, prior = self._small()
        n = basis.dimension
        dense = resolution_operator(basis, g_op, noise=noise, prior_factor=prior)
        point = point_functional(basis, 0.4)
        local = local_average_functional(basis, lambda s: 10.0 if s < 0.1 else 0.0)
        probe = probe_resolution(
            g_op,
            noise=noise,
            prior_factor=prior,
            column_indices=range(n),
            row_indices=(0, 5, 11),
            functionals=[point, local],
            mass=basis.mass_matrix(),
            integral_vector=dense.integral_vector,
            positions=basis.nodes,
            trace_probes=400,
            seed=3,
        )
        self.assertTrue(probe.all_converged)
        self.assertEqual(probe.nonconverged_count, 0)
        for j, column in zip(probe.column_indices, probe.columns, strict=True):
            for i in range(n):
                self.assertAlmostEqual(column[i], dense.matrix[i][j], delta=1e-8)
        for i, row in zip(probe.row_indices, probe.rows, strict=True):
            for j in range(n):
                self.assertAlmostEqual(row[j], dense.matrix[i][j], delta=1e-8)
        for ell, kernel in zip((point, local), probe.kernels, strict=True):
            reference = averaging_kernel(dense, basis, ell)
            for got, want in zip(
                kernel.coefficients, reference.coefficients, strict=True
            ):
                self.assertAlmostEqual(got, want, delta=1e-7 * (1.0 + abs(want)))
            self.assertAlmostEqual(kernel.integral, reference.integral, delta=1e-8)
            self.assertAlmostEqual(
                kernel.target_integral, reference.target_integral, delta=1e-8
            )
            self.assertEqual(len(kernel.records), 2)
            self.assertTrue(math.isfinite(kernel.centroid))
            self.assertGreater(kernel.rms_width, 0.0)
        # Hutchinson estimate within four standard errors of the dense trace
        assert probe.trace is not None
        self.assertEqual(probe.trace_method, "whitened")
        self.assertLessEqual(
            abs(probe.trace.estimate - dense.trace), 4.0 * probe.trace.standard_error
        )
        self.assertGreater(probe.trace.standard_error, 0.0)
        # the coefficient-space operator estimate agrees within its own error
        other = probe_resolution(
            g_op,
            noise=noise,
            prior_factor=prior,
            trace_probes=400,
            trace_method="coefficient",
            seed=4,
        )
        assert other.trace is not None
        self.assertLessEqual(
            abs(other.trace.estimate - dense.trace),
            4.0 * other.trace.standard_error,
        )

    def test_trace_identity(self) -> None:
        """The trace tr R equals sum lambda_i / (1 + lambda_i) over eigenvalues of H."""
        basis, g_op, noise, prior = self._small()
        dense = resolution_operator(basis, g_op, noise=noise, prior_factor=prior)
        lis = likelihood_informed_subspace(
            [g_op], noise=noise, prior_factor=prior, threshold=1e300
        )
        total = math.fsum(v / (1.0 + v) for v in lis.eigenvalues)
        self.assertAlmostEqual(total, dense.trace, places=9)

    def test_closed_form_scalar(self) -> None:
        """G = I, C = tau2 I, Gamma = s2 I: R = rho I with rho = tau2/(tau2+s2)."""
        n, s2, tau2 = 6, 0.25, 1.0
        rho = tau2 / (tau2 + s2)
        probe = probe_resolution(
            IdentityOperator(n),
            noise=DiagonalCovariance([s2] * n),
            prior_factor=DiagonalOperator([math.sqrt(tau2)] * n),
            column_indices=(2,),
            functionals=[_unit(n, 3)],
            trace_probes=5,
        )
        for i, value in enumerate(probe.columns[0]):
            self.assertAlmostEqual(value, rho if i == 2 else 0.0, places=12)
        self.assertAlmostEqual(probe.kernels[0].coefficients[3], rho, places=12)
        assert probe.trace is not None
        # H = I/s2 * tau2 is a multiple of the identity: every probe is exact
        self.assertAlmostEqual(probe.trace.estimate, rho * n, places=10)
        self.assertAlmostEqual(probe.trace.standard_error, 0.0, places=10)

    def test_nonconverged_cg_is_reported(self) -> None:
        """A strongly informative problem with a 2-iteration cap is flagged."""
        n = 30
        prior_sd = [10.0 ** (1.5 * i / (n - 1)) for i in range(n)]
        mixing = Convolution1DOperator([0.25, 0.5, 0.25], n, boundary="zero")
        kwargs = {
            "noise": DiagonalCovariance([1.0] * n),
            "prior_factor": DiagonalOperator(prior_sd),
            "column_indices": (0, 15, 29),
            "trace_probes": 3,
        }
        probe = probe_resolution(mixing, cg_max_iterations=2, **kwargs)  # type: ignore[arg-type]
        self.assertFalse(probe.all_converged)
        self.assertGreaterEqual(probe.nonconverged_count, 1)
        failed = [r for r in probe.column_records if not r.converged]
        self.assertTrue(failed)
        self.assertTrue(all(r.stop_reason == "max_iterations" for r in failed))
        self.assertGreater(max(r.relative_residual for r in failed), 1e-6)
        self.assertEqual(len(probe.columns), 3)
        # with the default cap the same problem converges
        good = probe_resolution(mixing, **kwargs)  # type: ignore[arg-type]
        self.assertTrue(good.all_converged)

    def test_beyond_dense_limit(self) -> None:
        """Size 450: resolution columns satisfy the posterior-precision identity."""
        n, m, sd, tau = 450, 24, 0.05, 2.0
        forward = _conv_select(n, m, 3, 5.0)
        factor = DiagonalOperator([tau] * n)
        noise = DiagonalCovariance([sd**2] * m)
        with self.assertRaises(ValueError):
            resolution_operator(
                HatBasis(n, 1.0), forward, noise=noise, prior_factor=factor
            )
        j = n // 2
        probe = probe_resolution(
            forward,
            noise=noise,
            prior_factor=factor,
            column_indices=(j,),
            trace_probes=4,
            seed=1,
        )
        self.assertTrue(probe.all_converged)
        # (G'Gamma^-1 G + C^-1) R e_j = G'Gamma^-1 G e_j, applied independently
        column = probe.columns[0]
        gx = forward.matvec(column)
        lhs = forward.rmatvec(tuple(v / sd**2 for v in gx))
        lhs = tuple(a + c / tau**2 for a, c in zip(lhs, column, strict=True))
        gu = forward.matvec(_unit(n, j))
        rhs = forward.rmatvec(tuple(v / sd**2 for v in gu))
        err = math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(lhs, rhs, strict=True)))
        scale = math.sqrt(math.fsum(b * b for b in rhs))
        self.assertLess(err / scale, 1e-7)
        # tr R = sum lambda/(1+lambda) with the dual-space eigenvalues
        oracle = _dual_eigenvalues(forward, factor, sd)
        exact = math.fsum(v / (1.0 + v) for v in oracle)
        assert probe.trace is not None
        self.assertLessEqual(
            abs(probe.trace.estimate - exact),
            5.0 * probe.trace.standard_error + 1e-9,
        )

    def test_validation(self) -> None:
        """Invalid indices, shapes and options are rejected."""
        n = 5
        kwargs = {
            "noise": DiagonalCovariance([1.0] * n),
            "prior_factor": DiagonalOperator([1.0] * n),
        }
        eye = IdentityOperator(n)
        with self.assertRaises(ValueError):
            probe_resolution(eye, column_indices=(5,), **kwargs)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            probe_resolution(eye, row_indices=(1.0,), **kwargs)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            probe_resolution(eye, functionals=[(1.0, 2.0)], **kwargs)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            probe_resolution(eye, trace_probes=-1, **kwargs)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            probe_resolution(eye, trace_method="x", **kwargs)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            probe_resolution(eye, mass=IdentityOperator(4), **kwargs)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            probe_resolution(
                eye,
                noise=DiagonalCovariance([1.0] * 3),
                prior_factor=DiagonalOperator([1.0] * n),
            )


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class Slow2000Tests(unittest.TestCase):
    """The 2000-unknown case the dense routines refuse."""

    def test_lis_and_resolution_2000(self) -> None:
        """Operator counts, time and oracle agreement at n = 2000."""
        n, m, sd = 2000, 24, 0.05
        forward = _conv_select(n, m, 3, 5.0)
        factor = DiagonalOperator([1.0] * n)
        noise = DiagonalCovariance([sd**2] * m)
        with self.assertRaises(ValueError):
            likelihood_informed_subspace(
                [forward], noise=noise, prior_factor=factor, threshold=1.0
            )
        oracle = _dual_eigenvalues(forward, factor, sd)
        above = sum(1 for v in oracle if v > 1.0)
        for method in ("lanczos", "randomized"):
            result = matrix_free_lis(
                [forward],
                noise=noise,
                prior_factor=factor,
                threshold=1.0,
                method=method,  # type: ignore[arg-type]
            )
            print(
                f"LIS n={n} {method}: rank={result.rank} (oracle {above}) "
                f"converged={result.converged} stop={result.stop_reason} "
                f"size={result.subspace_size} rounds={result.rounds} "
                f"H-applications={result.hessian_applications} "
                f"J counts={result.jacobian_counts} "
                f"P counts={result.factor_counts} "
                f"max residual={result.max_retained_residual:.2e} "
                f"seconds={result.seconds:.1f}"
            )
            self.assertTrue(result.converged)
            self.assertEqual(result.rank, above)
            for found, want in zip(result.eigenvalues[:above], oracle, strict=False):
                self.assertLessEqual(abs(found - want), 1e-7 * oracle[0])
        probe = probe_resolution(
            forward,
            noise=noise,
            prior_factor=factor,
            column_indices=(n // 2,),
            row_indices=(n // 2,),
            trace_probes=8,
            seed=2,
        )
        exact = math.fsum(v / (1.0 + v) for v in oracle)
        assert probe.trace is not None
        print(
            f"resolution n={n}: solves={probe.solve_count} "
            f"cg iterations={probe.total_cg_iterations} "
            f"max rel residual={probe.max_relative_residual:.2e} "
            f"all converged={probe.all_converged} G counts={probe.forward_counts} "
            f"trace={probe.trace.estimate:.4f} +- {probe.trace.standard_error:.4f} "
            f"(dual-space exact {exact:.4f}) seconds={probe.seconds:.1f}"
        )
        self.assertTrue(probe.all_converged)
        self.assertLessEqual(
            abs(probe.trace.estimate - exact), 5.0 * probe.trace.standard_error + 1e-9
        )


if __name__ == "__main__":
    unittest.main()
