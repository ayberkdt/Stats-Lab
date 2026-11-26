"""Tests for the synthetic 2D gravity inversion benchmark G01 (roadmap 20C).

Everything here is SYNTHETIC: no real gravity field, body or spacecraft. Order:
closed forms (line-mass limit, infinite Bouguer slab, additivity, symmetry),
the independent quadrature oracle for the prism primitive, the benchmark
structure (finer-cell truth, no inverse crime), the singular spectrum and the
exact null space, depth weighting against plain Tikhonov, the equivalent-layer
ambiguity, the exact Gaussian posterior as a solver oracle, and invalid input.

Measured on the development machine (default case, 2% noise): 12 x 5 cells and
16 stations give ``n = 60 > m = 16``, a numerical rank of 16 and a null space of
dimension 44; the singular values fall from 2.5 to 2.2e-2 (condition ~113).
"""

from __future__ import annotations

import math
import os
import unittest

from probstat.inverse import linear_gaussian_posterior
from probstat.inverse.benchmark_gravity import (
    GRAVITY_BENCHMARK_CATALOG,
    centroid_depth,
    depth_weighted_tikhonov,
    depth_weights,
    distinguishability,
    gravity_inversion_benchmark,
    gravity_kernel_matrix,
    gravity_spectrum,
    null_space_element,
    prism_attraction,
    sensitivity_depth_exponent,
    single_layer_fit,
)
from probstat.inverse.regularization import tikhonov
from probstat.linalg import ScaledIdentityCovariance, matvec

_NODES = (
    (-0.9061798459386640, 0.2369268850561891),
    (-0.5384693101056831, 0.4786286704993665),
    (0.0, 0.5688888888888889),
    (0.5384693101056831, 0.4786286704993665),
    (0.9061798459386640, 0.2369268850561891),
)


def _quadrature_oracle(
    x_obs: float,
    height: float,
    x_edges: tuple[float, float],
    z_edges: tuple[float, float],
    panels: int = 8,
) -> float:
    """Return ``2 int int b / (a^2 + b^2)`` by composite 5-point Gauss rules.

    Independent of the closed-form primitive: the integrand is evaluated
    pointwise on ``panels x panels`` sub-cells.
    """
    total = 0.0
    dx = (x_edges[1] - x_edges[0]) / panels
    dz = (z_edges[1] - z_edges[0]) / panels
    for i in range(panels):
        for k in range(panels):
            for sx, wx in _NODES:
                for sz, wz in _NODES:
                    x = x_edges[0] + dx * (i + 0.5 + 0.5 * sx)
                    z = z_edges[0] + dz * (k + 0.5 + 0.5 * sz)
                    a, b = x - x_obs, z + height
                    total += 0.25 * dx * dz * wx * wz * b / (a * a + b * b)
    return 2.0 * total


class ClosedFormTests(unittest.TestCase):
    """Exact limits and identities of the prism kernel."""

    def test_line_mass_limit(self) -> None:
        """A tiny prism is a line mass: ``g = 2 G area b / (a^2 + b^2)``."""
        eps = 1e-3
        for a, b in ((0.0, 1.0), (0.7, 1.3), (-2.0, 0.5)):
            g = prism_attraction(
                0.0, 0.0, (a - eps / 2, a + eps / 2), (b - eps / 2, b + eps / 2)
            )
            ref = 2.0 * eps * eps * b / (a * a + b * b)
            self.assertAlmostEqual(g / ref, 1.0, delta=1e-5)

    def test_bouguer_slab(self) -> None:
        """A very wide slab attracts with ``2 pi G rho t`` (relative error ~ 1e-4)."""
        t, big = 0.4, 1e4
        g = prism_attraction(0.0, 0.0, (-big, big), (0.3, 0.3 + t))
        self.assertAlmostEqual(g / (2.0 * math.pi * t), 1.0, delta=1e-3)
        self.assertLess(g, 2.0 * math.pi * t)

    def test_gravitational_constant_scales(self) -> None:
        """The attraction is proportional to ``G``."""
        base = prism_attraction(0.2, 0.1, (0.0, 1.0), (0.2, 0.9))
        scaled = prism_attraction(
            0.2, 0.1, (0.0, 1.0), (0.2, 0.9), gravitational_constant=6.5
        )
        self.assertAlmostEqual(scaled / base, 6.5, places=12)

    def test_additivity_over_subcells(self) -> None:
        """Splitting a cell into four leaves the attraction unchanged."""
        whole = prism_attraction(0.3, 0.05, (0.0, 1.0), (0.1, 0.9))
        parts = sum(
            prism_attraction(0.3, 0.05, (x0, x1), (z0, z1))
            for x0, x1 in ((0.0, 0.4), (0.4, 1.0))
            for z0, z1 in ((0.1, 0.5), (0.5, 0.9))
        )
        self.assertAlmostEqual(whole / parts, 1.0, places=12)

    def test_symmetry_and_positivity(self) -> None:
        """A symmetric layout gives a mirrored matrix with positive entries."""
        xs = (0.5, 1.0, 1.5)
        a = gravity_kernel_matrix(xs, (0.0, 0.5, 1.0, 1.5, 2.0), (0.0, 1.0), height=0.1)
        for row, mirrored in zip(a, reversed(a), strict=True):
            for j, v in enumerate(row):
                self.assertAlmostEqual(v, mirrored[len(row) - 1 - j], places=13)
                self.assertGreater(v, 0.0)
        # the middle station is mirror-symmetric to itself
        for j, v in enumerate(a[1]):
            self.assertAlmostEqual(v, a[1][len(a[1]) - 1 - j], places=13)

    def test_station_on_top_face_is_finite(self) -> None:
        """Height 0 with a cell touching the surface is a finite limit."""
        g = prism_attraction(0.5, 0.0, (0.0, 1.0), (0.0, 0.5))
        ref = _quadrature_oracle(0.5, 1e-9, (0.0, 1.0), (0.0, 0.5), panels=40)
        self.assertTrue(math.isfinite(g))
        self.assertAlmostEqual(g / ref, 1.0, delta=2e-3)


class QuadratureOracleTests(unittest.TestCase):
    """The exact primitive against pointwise quadrature of the kernel."""

    def test_primitive_matches_quadrature(self) -> None:
        """Several geometries agree with the Gauss oracle to 1e-7.

        The corner-singular case (station at a cell edge) limits the oracle.
        """
        cases = (
            (0.0, 0.2, (-0.5, 0.5), (0.1, 0.6)),
            (1.7, 0.0, (0.0, 1.0), (0.2, 0.9)),
            (-3.0, 0.5, (0.0, 2.0), (0.0, 1.0)),
            (0.4, 0.1, (0.0, 1.0), (0.3, 0.4)),
            (0.0, 0.05, (-1.0, 0.0), (0.05, 0.5)),
        )
        for xo, h, xe, ze in cases:
            exact = prism_attraction(xo, h, xe, ze)
            ref = _quadrature_oracle(xo, h, xe, ze)
            self.assertAlmostEqual(exact / ref, 1.0, delta=1e-7, msg=str((xo, h)))

    def test_matrix_entries_match_prisms(self) -> None:
        """Every matrix entry is the single-prism attraction, depth-major."""
        xs = (0.2, 0.9)
        ex, ez = (0.0, 0.5, 1.0), (0.0, 0.4, 1.0)
        a = gravity_kernel_matrix(xs, ex, ez, height=0.2)
        self.assertEqual((len(a), len(a[0])), (2, 4))
        for i, xo in enumerate(xs):
            for k in range(2):
                for j in range(2):
                    ref = prism_attraction(
                        xo, 0.2, (ex[j], ex[j + 1]), (ez[k], ez[k + 1])
                    )
                    self.assertEqual(a[i][k * 2 + j], ref)


class BenchmarkStructureTests(unittest.TestCase):
    """Finer-cell truth, noise reporting and seeds."""

    def test_no_inverse_crime(self) -> None:
        """Refinement 3 data differ from ``A x_true``; refinement 1 is exact."""
        crime = gravity_inversion_benchmark(refinement=1, truth="deep_block")
        fine = gravity_inversion_benchmark(refinement=3, truth="deep_block")
        self.assertTrue(crime.same_discretization)
        self.assertFalse(fine.same_discretization)
        lhs = matvec(crime.matrix, crime.truth)
        for a, b in zip(lhs, crime.exact_data, strict=True):
            self.assertAlmostEqual(a, b, places=12)
        diff = math.hypot(
            *(
                a - b
                for a, b in zip(
                    matvec(fine.matrix, fine.truth), fine.exact_data, strict=True
                )
            )
        )
        self.assertGreater(diff / math.hypot(*fine.exact_data), 1e-3)

    def test_truth_is_block_mean_of_fine_truth(self) -> None:
        """The projected truth lies in the range of the analytic densities."""
        case = gravity_inversion_benchmark(truth="two_bodies")
        self.assertEqual(len(case.truth), 60)
        self.assertGreaterEqual(min(case.truth), 0.0)
        self.assertLessEqual(max(case.truth), 1.0 + 1e-12)
        self.assertEqual(case.operator.shape, (16, 60))
        self.assertEqual(len(case.depth_centers), 60)

    def test_seed_and_noise_reporting(self) -> None:
        """Same seed same data; the realized noise norm is reported unrescaled."""
        a = gravity_inversion_benchmark(seed=5)
        b = gravity_inversion_benchmark(seed=5)
        c = gravity_inversion_benchmark(seed=6)
        self.assertEqual(a.data, b.data)
        self.assertNotEqual(a.data, c.data)
        self.assertEqual(a.exact_data, c.exact_data)
        err = math.hypot(*(y - e for y, e in zip(a.data, a.exact_data, strict=True)))
        self.assertAlmostEqual(err, a.realized_noise_norm, places=12)
        self.assertIn("depth_nonuniqueness", a.pathologies)
        self.assertEqual(GRAVITY_BENCHMARK_CATALOG[0].identifier, "G01")

    def test_prior_covariance_is_spd_exponential(self) -> None:
        """The optional prior is built; ``None`` disables it."""
        case = gravity_inversion_benchmark(x_cells=4, z_cells=2, stations=5)
        self.assertEqual(case.prior_covariance.dimension, 8)  # type: ignore[union-attr]
        none = gravity_inversion_benchmark(prior_length_scale=None)
        self.assertIsNone(none.prior_covariance)


class SpectrumAndNullSpaceTests(unittest.TestCase):
    """Singular spectrum, exact null space and distinguishability."""

    def setUp(self) -> None:
        """Build the default case once per test."""
        self.case = gravity_inversion_benchmark(truth="deep_block")

    def test_spectrum_counts(self) -> None:
        """More cells than data: the null space has dimension ``n - rank``."""
        sp = gravity_spectrum(self.case.matrix, self.case.noise_level)
        self.assertEqual(sp.numerical_rank, 16)
        self.assertEqual(sp.null_dimension, 60 - 16)
        self.assertEqual(sp.singular_values, self.case.singular_values)
        self.assertGreater(sp.condition_number, 50.0)
        self.assertLess(sp.resolved_modes, 17)
        self.assertEqual(
            sp.singular_values, tuple(sorted(sp.singular_values, reverse=True))
        )

    def test_gap_loses_information(self) -> None:
        """A raised observation height attenuates the small singular values."""
        low = gravity_inversion_benchmark(height=0.05)
        high = gravity_inversion_benchmark(height=0.6)
        ratio_low = low.singular_values[-1] / low.singular_values[0]
        ratio_high = high.singular_values[-1] / high.singular_values[0]
        self.assertLess(ratio_high, 0.2 * ratio_low)

    def test_null_vector_leaves_data_unchanged(self) -> None:
        """A large null-space density pattern produces no data (roundoff)."""
        n = 60
        e = [0.0] * n
        e[30] = 1.0
        null = null_space_element(self.case.matrix, e)
        self.assertGreater(math.hypot(*null), 0.5)
        change = math.hypot(*matvec(self.case.matrix, null))
        self.assertLess(change, 1e-12)
        shifted = tuple(t + 5.0 * v for t, v in zip(self.case.truth, null, strict=True))
        gap = distinguishability(
            self.case.matrix, self.case.noise_level, self.case.truth, shifted
        )
        self.assertLess(gap, 1e-9)
        self.assertGreater(
            math.hypot(*(s - t for s, t in zip(shifted, self.case.truth, strict=True))),
            1.0,
        )

    def test_range_component_is_distinguishable(self) -> None:
        """A strong shallow cell is visible far above the noise."""
        x = [0.0] * 60
        x[6] = 1.0
        gap = distinguishability(self.case.matrix, self.case.noise_level, x, [0.0] * 60)
        self.assertGreater(gap, 5.0)

    def test_sensitivity_decays_with_depth(self) -> None:
        """Column norms decay with depth: a negative log-log slope."""
        slope = sensitivity_depth_exponent(self.case.matrix, self.case.depth_centers)
        self.assertLess(slope, -0.2)
        self.assertGreater(slope, -2.0)


class DepthWeightingTests(unittest.TestCase):
    """Depth weighting against plain Tikhonov and the exact Gaussian posterior."""

    @staticmethod
    def _discrepancy(case, beta: float) -> tuple[float, tuple[float, ...]]:  # type: ignore[no-untyped-def]
        """Return ``(lambda, solution)`` with residual at the realized noise."""
        w = depth_weights(case.depth_centers, beta=beta)
        lo, hi = -3.0, 1.0
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            sol = depth_weighted_tikhonov(case.matrix, case.data, 10.0**mid, w)
            res = math.hypot(
                *(
                    a - b
                    for a, b in zip(matvec(case.matrix, sol), case.data, strict=True)
                )
            )
            if res > case.realized_noise_norm:
                hi = mid
            else:
                lo = mid
        return 10.0**lo, depth_weighted_tikhonov(case.matrix, case.data, 10.0**lo, w)

    def test_unit_weights_equal_plain_tikhonov(self) -> None:
        """``beta = 0`` reproduces ordinary Tikhonov to roundoff."""
        case = gravity_inversion_benchmark(truth="deep_block")
        w = depth_weights(case.depth_centers, beta=0.0)
        self.assertEqual(w, (1.0,) * 60)
        weighted = depth_weighted_tikhonov(case.matrix, case.data, 0.1, w)
        plain = tikhonov(case.matrix, case.data, 0.1).solution
        for a, b in zip(weighted, plain, strict=True):
            self.assertAlmostEqual(a, b, places=9)

    def test_depth_weighting_moves_mass_down(self) -> None:
        """Plain Tikhonov is too shallow for a deep block; beta = 2 corrects it."""
        case = gravity_inversion_benchmark(truth="deep_block", relative_noise=0.01)
        truth_depth = centroid_depth(case.truth, case.depth_centers)
        depths = {}
        for beta in (0.0, 2.0):
            _, sol = self._discrepancy(case, beta)
            depths[beta] = centroid_depth(sol, case.depth_centers)
        # measured: 0.464 (beta 0) and 0.655 (beta 2) against the truth 0.620
        self.assertLess(depths[0.0], truth_depth - 0.1)
        self.assertLess(depths[0.0], depths[2.0])
        self.assertLess(abs(depths[2.0] - truth_depth), abs(depths[0.0] - truth_depth))

    def test_tikhonov_equals_gaussian_posterior_mean(self) -> None:
        """An iid prior ``N(0, v I)`` gives Tikhonov with ``lambda = sigma/sqrt(v)``."""
        case = gravity_inversion_benchmark(truth="two_bodies", prior_length_scale=None)
        sd = 0.7
        post = linear_gaussian_posterior(
            case.matrix,
            case.data,
            noise=case.noise,
            prior_covariance=ScaledIdentityCovariance(60, sd * sd),
        )
        lam = case.noise_level / sd
        sol = tikhonov(case.matrix, case.data, lam).solution
        for a, b in zip(post.mean, sol, strict=True):
            self.assertAlmostEqual(a, b, places=8)

    def test_equivalent_layers_all_fit_within_noise_scale(self) -> None:
        """A single layer at ANY depth explains the deep-block data to the noise.

        Measured on the default seed, residual / noise-norm per layer:
        see the assertion bound (every layer below three noise norms).
        """
        case = gravity_inversion_benchmark(truth="deep_block", relative_noise=0.02)
        residuals = []
        for layer in range(5):
            _, res = single_layer_fit(case.matrix, case.data, 12, layer, 1e-6)
            residuals.append(res / case.realized_noise_norm)
        self.assertLess(max(residuals), 3.0)
        self.assertGreater(min(residuals), 0.2)


class InvalidInputTests(unittest.TestCase):
    """Rejected arguments and degenerate numerical cases."""

    def test_invalid_geometry(self) -> None:
        """Bad edges, depths and heights raise."""
        with self.assertRaises(ValueError):
            prism_attraction(0.0, -0.1, (0.0, 1.0), (0.1, 0.5))
        with self.assertRaises(ValueError):
            prism_attraction(0.0, 0.0, (1.0, 0.0), (0.1, 0.5))
        with self.assertRaises(ValueError):
            prism_attraction(0.0, 0.0, (0.0, 1.0), (-0.1, 0.5))
        with self.assertRaises(ValueError):
            prism_attraction(
                0.0, 0.0, (0.0, 1.0), (0.1, 0.5), gravitational_constant=0.0
            )
        with self.assertRaises(TypeError):
            prism_attraction("a", 0.0, (0.0, 1.0), (0.1, 0.5))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            gravity_kernel_matrix((0.5,), (0.0, 0.5, 0.4), (0.0, 1.0))
        with self.assertRaises(ValueError):
            gravity_kernel_matrix((0.5,), (0.0, 1.0), (-0.2, 1.0))
        with self.assertRaises(ValueError):
            gravity_kernel_matrix((), (0.0, 1.0), (0.0, 1.0))

    def test_invalid_benchmark_and_tools(self) -> None:
        """Unknown truths, bad sizes and mismatched vectors raise."""
        with self.assertRaises(ValueError):
            gravity_inversion_benchmark(truth="moon")
        with self.assertRaises(ValueError):
            gravity_inversion_benchmark(refinement=0)
        with self.assertRaises(ValueError):
            gravity_inversion_benchmark(x_cells=1)
        with self.assertRaises(ValueError):
            gravity_inversion_benchmark(height=-1.0)
        with self.assertRaises(ValueError):
            gravity_inversion_benchmark(relative_noise=0.0)
        with self.assertRaises(ValueError):
            depth_weights((0.1, 0.2), beta=-1.0)
        with self.assertRaises(ValueError):
            depth_weights((0.1, 0.2), z0=0.0)
        with self.assertRaises(ValueError):
            centroid_depth((0.0, 0.0), (0.1, 0.2))
        case = gravity_inversion_benchmark(x_cells=4, z_cells=2, stations=5)
        with self.assertRaises(ValueError):
            null_space_element(case.matrix, (1.0, 2.0))
        with self.assertRaises(ValueError):
            single_layer_fit(case.matrix, case.data, 4, 2, 1e-3)
        with self.assertRaises(ValueError):
            single_layer_fit(case.matrix, case.data, 3, 0, 1e-3)
        with self.assertRaises(ValueError):
            depth_weighted_tikhonov(case.matrix, case.data, 0.1, (1.0,) * 7)
        with self.assertRaises(ValueError):
            gravity_spectrum(case.matrix, 0.0)

    def test_underdetermined_least_squares_without_damping_is_rejected(self) -> None:
        """A single layer wider than the data is rank deficient without a ridge."""
        case = gravity_inversion_benchmark(x_cells=20, z_cells=2, stations=6)
        with self.assertRaises(ValueError):
            single_layer_fit(case.matrix, case.data, 20, 0, 0.0)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowDepthWeightingStudy(unittest.TestCase):
    """Seed sweep: depth weighting shifts the centroid down in (almost) every draw."""

    def test_seed_sweep(self) -> None:
        """Over 40 seeds beta = 2 is deeper than beta = 0 in at least 95%."""
        hits = 0
        for seed in range(100, 140):
            case = gravity_inversion_benchmark(
                truth="deep_block", relative_noise=0.01, seed=seed
            )
            d0 = centroid_depth(
                DepthWeightingTests._discrepancy(case, 0.0)[1], case.depth_centers
            )
            d2 = centroid_depth(
                DepthWeightingTests._discrepancy(case, 2.0)[1], case.depth_centers
            )
            hits += d2 > d0
        self.assertGreaterEqual(hits, 38)


if __name__ == "__main__":
    unittest.main()
