"""Tests for the extended inverse benchmarks (roadmap 19: L07a/b, N01, N02, P01, P02).

Order: closed-form and exact-geometry cases first, then independent oracles
(dense Kronecker blur, four-edge chord intersection, closed-form heat
spectrum), then the refinement and seed semantics, then adversarial inputs and
one solver smoke per family that only asserts improvement over a trivial
baseline, never a pinned accuracy.
"""

from __future__ import annotations

import math
import random
import unittest

from probstat.inverse import (
    LinearInverseProblem,
    check_derivatives,
    gauss_newton_krylov,
    krylov_regularization,
    spectral_diagnostics,
)
from probstat.inverse.benchmark_extended import (
    EXTENDED_BENCHMARK_CATALOG,
    GaussianBlur2DOperator,
    NonlinearBenchmark,
    RayTransformOperator,
    elliptic_benchmark,
    exponential_decay_benchmark,
    heat_initial_condition_benchmark,
    image_deblur_benchmark,
    masked_tomography_benchmark,
    parallel_beam_rays,
    ray_pixel_intersections,
    squared_map_benchmark,
)
from probstat.inverse.benchmark_problems import (
    INVERSE_BENCHMARK_CATALOG,
    InverseBenchmark,
    _gaussian_kernel,
)
from probstat.inverse.iterative import expected_noise_norm
from probstat.linalg import Convolution1DOperator, adjoint_test, materialize


def distance(left: object, right: object) -> float:
    """Return the Euclidean distance of two vectors.

    References:
        [EQ-INV-BENCH2-008].

    """
    return math.hypot(*(a - b for a, b in zip(left, right, strict=True)))  # type: ignore[call-overload]


def norm(vector: object) -> float:
    """Return the Euclidean norm.

    References:
        [EQ-INV-BENCH2-008].

    """
    return math.hypot(*vector)  # type: ignore[call-overload]


def random_vector(size: int, seed: int) -> list[float]:
    """Return a reproducible standard-normal vector.

    References:
        [EQ-INV-BENCH2-008].

    """
    generator = random.Random(seed)
    return [generator.gauss(0.0, 1.0) for _ in range(size)]


def jacobian(case: NonlinearBenchmark, point: tuple[float, ...]) -> list[list[float]]:
    """Materialize the Jacobian of a benchmark model column by column.

    References:
        [EQ-INV-BENCH2-004].

    """
    n = len(point)
    columns = [
        case.model.jvp(point, tuple(1.0 if i == j else 0.0 for i in range(n)))
        for j in range(n)
    ]
    return [[columns[j][i] for j in range(n)] for i in range(len(columns[0]))]


def chord_by_edges(size: int, angle: float, offset: float) -> float:
    """Chord of the square ``[0, n]^2`` by intersecting the four edges.

    Independent of the slab method of the implementation: the line is
    intersected with each edge segment, and the chord is the largest distance
    between the intersection points.

    References:
        [EQ-INV-BENCH2-002].

    """
    c, s = math.cos(angle), math.sin(angle)
    centre = size / 2.0
    px, py = centre - offset * s, centre + offset * c
    points: list[tuple[float, float]] = []
    corners = [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]
    for k in range(4):
        (x0, y0), (x1, y1) = corners[k], corners[(k + 1) % 4]
        ex, ey = x1 - x0, y1 - y0
        # Solve p + t d = e0 + u e for (t, u), u in [0, 1].
        det = c * (-ey) - s * (-ex)
        if abs(det) < 1e-12:
            continue
        rx, ry = x0 - px, y0 - py
        t = (rx * (-ey) - ry * (-ex)) / det
        u = (c * ry - s * rx) / det
        if -1e-12 <= u <= 1.0 + 1e-12:
            points.append((px + t * c, py + t * s))
    return max(
        (math.dist(a, b) for a in points for b in points),
        default=0.0,
    )


class BlurTests(unittest.TestCase):
    """The 2D blur operator and the L07a case."""

    def test_adjoint_dot_test(self) -> None:
        """``y'(A x) = (A'y)'x`` to rounding on a non-square image."""
        operator = GaussianBlur2DOperator(10, 14, width=0.06)
        for seed in (1, 2, 3):
            x = random_vector(140, seed)
            y = random_vector(140, seed + 10)
            result = adjoint_test(operator, x, y, tolerance=1e-12)
            self.assertLess(result.relative_error, 1e-12)

    def test_matches_dense_kronecker_oracle(self) -> None:
        """The two-pass blur equals ``B_r X B_c'`` from dense 1D matrices."""
        rows, cols, width = 9, 12, 0.07
        operator = GaussianBlur2DOperator(rows, cols, width=width)
        b_r = materialize(
            Convolution1DOperator(_gaussian_kernel(width, 1.0 / rows), rows)
        )
        b_c = materialize(
            Convolution1DOperator(_gaussian_kernel(width, 1.0 / cols), cols)
        )
        x = random_vector(rows * cols, 7)
        image = [x[i * cols : (i + 1) * cols] for i in range(rows)]
        left = [
            [sum(b_r[i][k] * image[k][j] for k in range(rows)) for j in range(cols)]
            for i in range(rows)
        ]
        expected = [
            sum(left[i][k] * b_c[j][k] for k in range(cols))
            for i in range(rows)
            for j in range(cols)
        ]
        self.assertLess(distance(operator.matvec(x), expected), 1e-13)

    def test_constant_image_is_preserved_away_from_boundary(self) -> None:
        """The kernels sum to one, so interior pixels of a constant stay constant."""
        rows = cols = 16
        operator = GaussianBlur2DOperator(rows, cols, width=0.05)
        hr, hc = operator.kernel_half_widths
        blurred = operator.matvec([3.0] * (rows * cols))
        for i in range(hr, rows - hr):
            for j in range(hc, cols - hc):
                self.assertAlmostEqual(blurred[i * cols + j], 3.0, places=13)
        self.assertLess(blurred[0], 3.0)  # the zero boundary darkens the corner
        self.assertLess(blurred[0], blurred[8 * cols + 8])

    def test_refinement_flag_semantics(self) -> None:
        """Refinement 1 is an inverse crime; refinement 2 leaves a small gap."""
        exact_case = image_deblur_benchmark(16, 16, refinement=1)
        self.assertTrue(exact_case.same_discretization)
        self.assertLess(
            distance(
                exact_case.exact_data, exact_case.operator.matvec(exact_case.truth)
            ),
            1e-13,
        )
        fine_case = image_deblur_benchmark(16, 16, refinement=2)
        self.assertFalse(fine_case.same_discretization)
        gap = distance(
            fine_case.exact_data, fine_case.operator.matvec(fine_case.truth)
        ) / norm(fine_case.exact_data)
        self.assertGreater(gap, 1e-3)
        self.assertLess(gap, 0.2)

    def test_seed_separation_and_reproducibility(self) -> None:
        """The phantom is analytic; seeds change only the noise."""
        a = image_deblur_benchmark(16, 16, seed=1)
        b = image_deblur_benchmark(16, 16, seed=2)
        c = image_deblur_benchmark(16, 16, seed=1)
        self.assertEqual(a.truth, b.truth)
        self.assertNotEqual(a.data, b.data)
        self.assertEqual(a.data, c.data)
        self.assertEqual(a.exact_data, b.exact_data)
        self.assertIn("noise", a.seeds)

    def test_invalid_inputs(self) -> None:
        """Bad sizes, widths, refinements and noise levels raise."""
        with self.assertRaises(ValueError):
            image_deblur_benchmark(1, 16)
        with self.assertRaises(ValueError):
            image_deblur_benchmark(16, 16, width=0.0)
        with self.assertRaises(ValueError):
            image_deblur_benchmark(16, 16, refinement=0)
        with self.assertRaises(ValueError):
            image_deblur_benchmark(16, 16, relative_noise=-1.0)
        with self.assertRaises(TypeError):
            GaussianBlur2DOperator(8, 8, width="wide")  # type: ignore[arg-type]

    def test_krylov_regularization_beats_zero_estimate(self) -> None:
        """Early-stopped LSQR on independent data improves on the zero image."""
        case = image_deblur_benchmark(16, 16, refinement=2, relative_noise=0.01)
        problem = LinearInverseProblem.create(
            case.operator, case.data, noise=case.noise
        )
        result = krylov_regularization(
            problem, noise_norm=expected_noise_norm(256), truth=case.truth
        )
        self.assertLess(distance(result.estimate, case.truth), norm(case.truth))


class TomographyTests(unittest.TestCase):
    """Ray-pixel intersections, the operator and the L07b case."""

    def test_chord_length_matches_four_edge_oracle(self) -> None:
        """Summed intersection lengths equal the chord of the square."""
        for angle in (0.0, 0.3, 0.7, math.pi / 4, 1.2, 2.0, 2.9, 4.0):
            for offset in (-4.1, -0.5, 0.0, 0.37, 3.3):
                segments = ray_pixel_intersections(12, angle, offset)
                total = math.fsum(length for _, length in segments)
                self.assertAlmostEqual(
                    total, chord_by_edges(12, angle, offset), places=9
                )
                for pixel, length in segments:
                    self.assertTrue(0 <= pixel < 144)
                    self.assertGreater(length, 0.0)
                    self.assertLessEqual(length, math.sqrt(2.0) + 1e-12)

    def test_ray_along_a_grid_line(self) -> None:
        """A ray on a grid line goes wholly to the larger-index row or column."""
        row = ray_pixel_intersections(12, 0.0, 0.0)  # y = 6
        self.assertEqual([p for p, _ in row], list(range(72, 84)))
        for _, length in row:
            self.assertAlmostEqual(length, 1.0, places=12)
        column = ray_pixel_intersections(12, math.pi / 2.0, 0.0)  # x = 6
        self.assertEqual(sorted(p for p, _ in column), [6 + 12 * i for i in range(12)])
        self.assertAlmostEqual(
            math.fsum(length for _, length in column), 12.0, places=9
        )
        top = ray_pixel_intersections(12, 0.0, 6.0)  # y = 12, clamped to the last row
        self.assertEqual([p for p, _ in top], list(range(132, 144)))
        bottom = ray_pixel_intersections(12, 0.0, -6.0)  # y = 0
        self.assertEqual([p for p, _ in bottom], list(range(12)))

    def test_diagonal_through_corners_and_missing_ray(self) -> None:
        """The main diagonal visits exactly the diagonal pixels; far rays miss."""
        diagonal = ray_pixel_intersections(12, math.pi / 4.0, 0.0)
        self.assertEqual(sorted(p for p, _ in diagonal), [13 * i for i in range(12)])
        for _, length in diagonal:
            self.assertAlmostEqual(length, math.sqrt(2.0), places=9)
        self.assertEqual(ray_pixel_intersections(12, 0.4, 100.0), ())

    def test_adjoint_dot_test(self) -> None:
        """The scatter-add transpose is exact to rounding."""
        rays = parallel_beam_rays(12, 12, 12)
        operator = RayTransformOperator(12, rays)
        for seed in (1, 2, 3):
            x = random_vector(144, seed)
            y = random_vector(len(rays), seed + 20)
            self.assertLess(adjoint_test(operator, x, y).relative_error, 1e-12)

    def test_operator_matches_dense_rows(self) -> None:
        """Materialized rows hold the stored lengths; a single pixel probes one."""
        operator = RayTransformOperator(6, [(0.4, 0.2), (1.1, -1.0), (2.5, 0.0)])
        dense = materialize(operator)
        for k, row in enumerate(operator.segments):
            expected = [0.0] * 36
            for pixel, length in row:
                expected[pixel] += length
            self.assertLess(distance(dense[k], expected), 1e-14)
        # a unit pixel image returns that pixel's intersection length
        pixel = operator.segments[0][0][0]
        unit = [0.0] * 36
        unit[pixel] = 1.0
        self.assertAlmostEqual(
            operator.matvec(unit)[0], operator.segments[0][0][1], places=14
        )
        self.assertAlmostEqual(
            math.fsum(operator.matvec([1.0] * 36)),
            math.fsum(length for row in operator.segments for _, length in row),
            places=9,
        )

    def test_flag_semantics(self) -> None:
        """Same-grid data are exact; analytic data leave a bounded gap."""
        crime = masked_tomography_benchmark(12, 12, rays_per_angle=12)
        self.assertTrue(crime.same_discretization)
        self.assertLess(
            distance(crime.exact_data, crime.operator.matvec(crime.truth)), 1e-12
        )
        independent = masked_tomography_benchmark(
            12, 12, rays_per_angle=12, analytic=True
        )
        self.assertFalse(independent.same_discretization)
        self.assertEqual(independent.truth, crime.truth)
        gap = distance(
            independent.exact_data, independent.operator.matvec(independent.truth)
        ) / norm(independent.exact_data)
        self.assertGreater(gap, 1e-3)
        self.assertLess(gap, 0.3)

    def test_limited_angle_flag_seed_and_invalid(self) -> None:
        """A limited span is declared; seeds change only the noise; bad input raises."""
        limited = masked_tomography_benchmark(
            12, 6, rays_per_angle=12, angle_span=math.pi / 2.0
        )
        self.assertIn("limited_angle", limited.pathologies)
        self.assertEqual(limited.operator.shape, (72, 144))
        a = masked_tomography_benchmark(seed=1)
        b = masked_tomography_benchmark(seed=2)
        self.assertEqual(a.truth, b.truth)
        self.assertNotEqual(a.data, b.data)
        for bad in (
            {"size": 3},
            {"angles": 0},
            {"rays_per_angle": 0},
            {"angle_span": 4.0},
            {"angle_span": -1.0},
            {"angles": []},
        ):
            with self.assertRaises(ValueError, msg=str(bad)):
                masked_tomography_benchmark(**bad)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            RayTransformOperator(6, [])
        with self.assertRaises(ValueError):
            ray_pixel_intersections(6, math.nan, 0.0)

    def _run(
        self, *, analytic: bool, relative_noise: float
    ) -> tuple[float, float, str]:
        """Return (error, ||truth||, stop reason) of early-stopped LSQR.

        References:
            [EQ-INV-BENCH2-002].

        """
        case = masked_tomography_benchmark(
            12, 12, rays_per_angle=12, analytic=analytic, relative_noise=relative_noise
        )
        problem = LinearInverseProblem.create(
            case.operator, case.data, noise=case.noise
        )
        result = krylov_regularization(
            problem,
            noise_norm=expected_noise_norm(len(case.data)),
            truth=case.truth,
        )
        return (
            distance(result.estimate, case.truth),
            norm(case.truth),
            result.stop_reason,
        )

    def test_krylov_regularization_beats_zero_estimate(self) -> None:
        """LSQR beats the zero image on same-grid and (noisier) analytic data."""
        error, scale, reason = self._run(analytic=False, relative_noise=0.01)
        self.assertEqual(reason, "discrepancy")
        self.assertLess(error, 0.5 * scale)
        error, scale, reason = self._run(analytic=True, relative_noise=0.1)
        self.assertEqual(reason, "discrepancy")
        self.assertLess(error, scale)

    def test_model_error_defeats_the_discrepancy_rule(self) -> None:
        """Analytic data with 1% noise carry ~10% model error: the rule is not met."""
        _, _, reason = self._run(analytic=True, relative_noise=0.01)
        self.assertEqual(reason, "max_iterations")


class ExponentialDecayTests(unittest.TestCase):
    """N01."""

    def test_derivative_checks(self) -> None:
        """Analytic JVP/VJP pass the Taylor sweep and the adjoint dot test."""
        case = exponential_decay_benchmark()
        result = check_derivatives(
            case.model, (1.7, 0.8), (0.3, -0.4), random_vector(12, 4)
        )
        self.assertTrue(result.taylor_passed)
        self.assertLess(result.dot_relative_error, 1e-12)

    def test_truth_reproduces_exact_data(self) -> None:
        """``G(truth)`` is the withheld exact data, and the design is declared."""
        case = exponential_decay_benchmark(amplitude=1.5, rate=0.7)
        self.assertEqual(case.model.value(case.truth), case.exact_data)
        self.assertEqual(case.truth, (1.5, 0.7))
        self.assertTrue(case.same_discretization)
        self.assertNotIn("ill_conditioned", case.pathologies)

    def test_weak_design_is_much_worse_conditioned(self) -> None:
        """Late times make amplitude and rate trade off (Jacobian condition)."""
        good = exponential_decay_benchmark(design="identifiable")
        weak = exponential_decay_benchmark(design="weak")
        cond_good = spectral_diagnostics(jacobian(good, good.truth)).condition_number
        cond_weak = spectral_diagnostics(jacobian(weak, weak.truth)).condition_number
        self.assertGreater(cond_weak, 30.0 * cond_good)
        self.assertIn("parameter_tradeoff", weak.pathologies)
        self.assertGreaterEqual(min(weak.model.value((1.0, 1.0))), 0.0)

    def test_noise_level_does_not_change_the_draws(self) -> None:
        """Standardized noise draws depend on the seed only."""
        a = exponential_decay_benchmark(relative_noise=1e-2, seed=5)
        b = exponential_decay_benchmark(relative_noise=1e-1, seed=5)
        za = [
            (d - e) / a.noise_level for d, e in zip(a.data, a.exact_data, strict=True)
        ]
        zb = [
            (d - e) / b.noise_level for d, e in zip(b.data, b.exact_data, strict=True)
        ]
        self.assertLess(distance(za, zb), 1e-9)
        c = exponential_decay_benchmark(seed=6)
        self.assertNotEqual(a.data, c.data)

    def test_invalid_inputs(self) -> None:
        """Contradictory designs, bad parameters and bad times raise."""
        with self.assertRaises(ValueError):
            exponential_decay_benchmark(design="bogus")
        with self.assertRaises(ValueError):
            exponential_decay_benchmark(times=[0.0, 1.0, 2.0], design="weak")
        with self.assertRaises(ValueError):
            exponential_decay_benchmark(design="custom")
        with self.assertRaises(ValueError):
            exponential_decay_benchmark(amplitude=0.0)
        with self.assertRaises(ValueError):
            exponential_decay_benchmark(rate=-1.0)
        with self.assertRaises(ValueError):
            exponential_decay_benchmark(times=[0.0, 1.0])
        with self.assertRaises(ValueError):
            exponential_decay_benchmark(times=[-1.0, 0.0, 1.0])
        custom = exponential_decay_benchmark(times=[0.0, 0.5, 1.0, 2.0])
        self.assertEqual(custom.model.output_dimension, 4)

    def test_gauss_newton_recovers_parameters(self) -> None:
        """On the identifiable design with low noise GN-Krylov finds ``(a, r)``."""
        case = exponential_decay_benchmark(relative_noise=1e-4)
        result = gauss_newton_krylov(
            case.model, case.data, noise=case.noise, initial=(1.0, 0.5)
        )
        self.assertTrue(result.converged, result.stop_reason)
        self.assertLess(distance(result.estimate, case.truth), 1e-2 * norm(case.truth))


class SquaredMapTests(unittest.TestCase):
    """N02."""

    def test_sign_symmetric_exact_optima(self) -> None:
        """Every sign pattern of the truth reproduces the exact data."""
        case = squared_map_benchmark(3)
        self.assertIn("multimodal", case.pathologies)
        negated = tuple(-v for v in case.truth)
        self.assertEqual(case.model.value(negated), case.model.value(case.truth))
        for pattern in range(8):
            flipped = tuple(
                v * (-1.0 if (pattern >> i) & 1 else 1.0)
                for i, v in enumerate(case.truth)
            )
            self.assertLess(distance(case.model.value(flipped), case.exact_data), 1e-14)

    def test_product_map_has_continuum_of_exact_optima(self) -> None:
        """``(a, b) -> (s a, b / s)`` and ``-x`` keep ``a b`` fixed."""
        case = squared_map_benchmark(4, kind="product")
        self.assertEqual(case.model.output_dimension, 2)
        self.assertIn("multimodal", case.pathologies)
        negated = tuple(-v for v in case.truth)
        self.assertEqual(case.model.value(negated), case.model.value(case.truth))
        a, b, c, d = case.truth
        moved = (2.5 * a, b / 2.5, -0.4 * c, d / -0.4)
        self.assertLess(distance(case.model.value(moved), case.exact_data), 1e-14)

    def test_derivative_checks(self) -> None:
        """Both maps pass the Taylor sweep and the adjoint dot test."""
        for kind, dimension in (("square", 3), ("product", 4)):
            case = squared_map_benchmark(dimension, kind=kind)
            m = case.model.output_dimension
            result = check_derivatives(
                case.model,
                (0.9, -1.3, 0.6, 1.1)[:dimension],
                (0.2, 0.5, -0.3, 0.7)[:dimension],
                random_vector(m, 3),
            )
            self.assertTrue(result.taylor_passed, kind)
            self.assertLess(result.dot_relative_error, 1e-12, kind)

    def test_seed_separation(self) -> None:
        """The truth depends on the seed only; the noise level changes only data."""
        a = squared_map_benchmark(3, seed=1, relative_noise=1e-2)
        b = squared_map_benchmark(3, seed=1, relative_noise=1e-1)
        c = squared_map_benchmark(3, seed=2)
        self.assertEqual(a.truth, b.truth)
        self.assertNotEqual(a.data, b.data)
        self.assertNotEqual(a.truth, c.truth)
        self.assertEqual(set(a.seeds), {"truth", "noise"})

    def test_invalid_inputs(self) -> None:
        """Unknown kinds and odd product dimensions raise."""
        with self.assertRaises(ValueError):
            squared_map_benchmark(3, kind="cube")
        with self.assertRaises(ValueError):
            squared_map_benchmark(3, kind="product")
        with self.assertRaises(ValueError):
            squared_map_benchmark(0)


class EllipticTests(unittest.TestCase):
    """P01."""

    def test_refinement_flag_semantics(self) -> None:
        """Refinement 1 matches the model; refinement 4 leaves a small gap."""
        same = elliptic_benchmark(16, 6, refinement=1)
        self.assertTrue(same.same_discretization)
        self.assertEqual(same.model.value(same.truth), same.exact_data)
        fine = elliptic_benchmark(16, 6, refinement=4)
        self.assertFalse(fine.same_discretization)
        gap = distance(fine.model.value(fine.truth), fine.exact_data) / norm(
            fine.exact_data
        )
        self.assertGreater(gap, 1e-5)
        self.assertLess(gap, 0.1)

    def test_prior_derivatives_and_shapes(self) -> None:
        """Dimensions agree, the prior is SPD, and the model derivatives pass."""
        case = elliptic_benchmark(12, [0.2, 0.5, 0.8])
        self.assertEqual(case.model.input_dimension, 12)
        self.assertEqual(case.model.output_dimension, 3)
        self.assertEqual(len(case.truth), 12)
        self.assertIsNotNone(case.prior_covariance)
        self.assertEqual(case.prior_mean, (0.0,) * 12)
        self.assertEqual(case.prior_covariance.dimension, 12)  # type: ignore[union-attr]
        result = check_derivatives(
            case.model, case.truth, random_vector(12, 1), random_vector(3, 2)
        )
        self.assertTrue(result.passed)

    def test_seeds_and_invalid_inputs(self) -> None:
        """Seeds change only the noise; bad arguments raise."""
        a = elliptic_benchmark(seed=1)
        b = elliptic_benchmark(seed=2)
        self.assertEqual(a.truth, b.truth)
        self.assertNotEqual(a.data, b.data)
        with self.assertRaises(ValueError):
            elliptic_benchmark(3)
        with self.assertRaises(ValueError):
            elliptic_benchmark(16, 6, refinement=0)
        with self.assertRaises(ValueError):
            elliptic_benchmark(16, 0)
        with self.assertRaises(ValueError):
            elliptic_benchmark(16, [0.5, 1.5])

    def test_gauss_newton_reduces_misfit(self) -> None:
        """With a Tikhonov weight GN-Krylov lowers the misfit below the start."""
        case = elliptic_benchmark(12, 6, relative_noise=1e-3)
        start = (0.0,) * 12
        before = distance(case.model.value(start), case.data)
        result = gauss_newton_krylov(
            case.model,
            case.data,
            noise=case.noise,
            regularization=1e-2,
            initial=start,
            max_iterations=10,
        )
        self.assertLess(result.residual_norm, 0.2 * before)


class HeatTests(unittest.TestCase):
    """P02."""

    def test_refinement_flag_semantics(self) -> None:
        """One step set is exact; four times more steps leave a small gap."""
        same = heat_initial_condition_benchmark(16, step_refinement=1)
        self.assertTrue(same.same_discretization)
        self.assertLess(
            distance(same.exact_data, same.operator.matvec(same.truth)), 1e-13
        )
        fine = heat_initial_condition_benchmark(16, step_refinement=4)
        self.assertFalse(fine.same_discretization)
        gap = distance(fine.exact_data, fine.operator.matvec(fine.truth)) / norm(
            fine.exact_data
        )
        self.assertGreater(gap, 1e-4)
        self.assertLess(gap, 0.1)

    def test_closed_form_singular_values(self) -> None:
        """The declared spectrum equals the dense SVD of the materialized operator."""
        case = heat_initial_condition_benchmark(12, steps=5)
        assert case.singular_values is not None
        dense = spectral_diagnostics(materialize(case.operator)).singular_values
        for exact, computed in zip(case.singular_values, dense, strict=True):
            self.assertAlmostEqual(
                exact, computed, delta=1e-10 * case.singular_values[0]
            )
        wide = heat_initial_condition_benchmark(24)
        assert wide.singular_values is not None
        self.assertLess(wide.singular_values[-1] / wide.singular_values[0], 1e-10)

    def test_seeds_and_invalid_inputs(self) -> None:
        """Seeds change only the noise; bad arguments raise."""
        a = heat_initial_condition_benchmark(seed=1)
        b = heat_initial_condition_benchmark(seed=2)
        self.assertEqual(a.truth, b.truth)
        self.assertNotEqual(a.data, b.data)
        with self.assertRaises(ValueError):
            heat_initial_condition_benchmark(3)
        with self.assertRaises(ValueError):
            heat_initial_condition_benchmark(16, step_refinement=0)
        with self.assertRaises(ValueError):
            heat_initial_condition_benchmark(16, final_time=0.0)
        with self.assertRaises(ValueError):
            heat_initial_condition_benchmark(16, scheme="explicit")

    def test_krylov_regularization_beats_zero_estimate(self) -> None:
        """Early-stopped LSQR beats the zero estimate on independent data."""
        case = heat_initial_condition_benchmark(
            24, step_refinement=4, relative_noise=1e-2
        )
        problem = LinearInverseProblem.create(
            case.operator, case.data, noise=case.noise
        )
        result = krylov_regularization(
            problem, noise_norm=expected_noise_norm(24), truth=case.truth
        )
        self.assertLess(distance(result.estimate, case.truth), norm(case.truth))


class CatalogTests(unittest.TestCase):
    """The extended catalog."""

    def test_catalog_entries(self) -> None:
        """Six unique new identifiers that do not clash with L01-L06."""
        identifiers = [case.identifier for case in EXTENDED_BENCHMARK_CATALOG]
        self.assertEqual(identifiers, ["L07a", "L07b", "N01", "N02", "P01", "P02"])
        existing = {case.identifier for case in INVERSE_BENCHMARK_CATALOG}
        self.assertTrue(existing.isdisjoint(identifiers))
        for case in EXTENDED_BENCHMARK_CATALOG:
            self.assertTrue(case.builder.startswith("inverse."))
            self.assertTrue(case.methods)
            self.assertTrue(case.pathologies)

    def test_builders_return_declared_identifiers(self) -> None:
        """Each builder returns the identifier its catalog entry declares."""
        built = {
            "L07a": image_deblur_benchmark(8, 8),
            "L07b": masked_tomography_benchmark(6, 6, rays_per_angle=6),
            "N01": exponential_decay_benchmark(),
            "N02": squared_map_benchmark(),
            "P01": elliptic_benchmark(),
            "P02": heat_initial_condition_benchmark(),
        }
        for case in EXTENDED_BENCHMARK_CATALOG:
            instance = built[case.identifier]
            self.assertEqual(instance.identifier, case.identifier)
            self.assertIsInstance(
                instance,
                NonlinearBenchmark
                if case.identifier in ("N01", "N02", "P01")
                else InverseBenchmark,
            )


if __name__ == "__main__":
    unittest.main()
