"""Tests for the shared modeling core: linalg, streams, designs, solvers, model."""

from __future__ import annotations

import math
import statistics
import unittest

from probstat.distributions import LogNormal, Normal, Uniform
from probstat.foundations import RandomStream, derive_seed, splitmix64
from probstat.linalg import (
    cholesky,
    inverse_spd,
    log_det_spd,
    lu_solve,
    matmul,
    matvec,
    sandwich,
    solve_spd,
    symmetric_eigen,
    transpose,
)
from probstat.modeling import (
    JointInput,
    ProbabilisticModel,
    RandomInput,
    gauss_hermite,
    nataf_correlation,
    standard_normal_cdf,
    standard_normal_quantile,
)
from probstat.multivariate import MultivariateNormal
from probstat.optimization import (
    brent_root,
    central_hessian,
    gauss_newton_covariance,
    least_squares,
    minimize_bfgs,
    minimize_nelder_mead,
)
from probstat.propagation import compare_designs, propagate
from probstat.sampling import halton, latin_hypercube, sobol, star_discrepancy_bound
from probstat.sampling.designs import _SOBOL_TABLE
from probstat.statistics import (
    excess_kurtosis,
    interquartile_range,
    median,
    median_absolute_deviation,
    quantile,
    ranks,
    skewness,
    spearman_correlation,
)
from probstat.uq import QuantityOfInterest


def _is_primitive(degree: int, code: int) -> bool:
    """Return whether the encoded GF(2) polynomial is primitive."""
    polynomial = (1 << degree) | 1
    for j in range(1, degree):
        if (code >> (degree - 1 - j)) & 1:
            polynomial |= 1 << (degree - j)
    order = (1 << degree) - 1
    value = 1
    for exponent in range(1, order + 1):
        value <<= 1
        if (value >> degree) & 1:
            value ^= polynomial
        if value == 1:
            return exponent == order
    return False


def _ishigami() -> ProbabilisticModel:
    """Return the Ishigami benchmark as a probabilistic model."""
    inputs = JointInput.independent(
        *(RandomInput(f"x{i}", Uniform(-math.pi, math.pi)) for i in (1, 2, 3))
    )

    def model(p):
        """Model."""
        return {
            "y": math.sin(p["x1"])
            + 7.0 * math.sin(p["x2"]) ** 2
            + 0.1 * p["x3"] ** 4 * math.sin(p["x1"])
        }

    return ProbabilisticModel(
        inputs, model, (QuantityOfInterest("y", "Ishigami output"),), name="ishigami"
    )


class LinearAlgebraTests(unittest.TestCase):
    """Dense linear algebra identities."""

    A = ((4.0, 1.0, 0.5), (1.0, 3.0, 0.2), (0.5, 0.2, 2.0))

    def test_cholesky_reconstructs_matrix(self) -> None:
        """Cholesky reconstructs matrix."""
        lower = cholesky(self.A)
        rebuilt = matmul(lower, transpose(lower))
        for row, expected in zip(rebuilt, self.A, strict=True):
            for value, target in zip(row, expected, strict=True):
                self.assertAlmostEqual(value, target, places=13)

    def test_solves_and_inverse_agree(self) -> None:
        """Solves and inverse agree."""
        b = (1.0, -2.0, 0.5)
        x = solve_spd(self.A, b)
        self.assertTrue(
            all(
                math.isclose(a, c, abs_tol=1e-13)
                for a, c in zip(matvec(self.A, x), b, strict=True)
            )
        )
        general = lu_solve(((0.0, 2.0), (3.0, 1.0)), (4.0, 5.0))
        self.assertAlmostEqual(general[0], 1.0)
        self.assertAlmostEqual(general[1], 2.0)
        identity = matmul(self.A, inverse_spd(self.A))
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(
                    identity[i][j], 1.0 if i == j else 0.0, places=12
                )

    def test_log_determinant_and_eigenvalues(self) -> None:
        """Log determinant and eigenvalues."""
        values, vectors = symmetric_eigen(self.A)
        self.assertAlmostEqual(
            log_det_spd(self.A), sum(math.log(v) for v in values), places=12
        )
        for k, value in enumerate(values):
            column = tuple(vectors[i][k] for i in range(3))
            image = matvec(self.A, column)
            for i in range(3):
                self.assertAlmostEqual(image[i], value * column[i], places=11)

    def test_failures_are_explicit(self) -> None:
        """Failures are explicit."""
        with self.assertRaises(ValueError):
            cholesky(((1.0, 2.0), (2.0, 1.0)))
        with self.assertRaises(ValueError):
            lu_solve(((1.0, 2.0), (2.0, 4.0)), (1.0, 1.0))

    def test_sandwich_matches_explicit_product(self) -> None:
        """Sandwich matches explicit product."""
        h = ((1.0, 2.0, 0.0), (0.0, 1.0, -1.0))
        result = sandwich(h, self.A)
        explicit = matmul(matmul(h, self.A), transpose(h))
        self.assertAlmostEqual(result[0][1], explicit[0][1])
        self.assertEqual(result[0][1], result[1][0])


class RandomStreamTests(unittest.TestCase):
    """Splittable seed derivation."""

    def test_reproducible_and_distinct(self) -> None:
        """Reproducible and distinct."""
        self.assertEqual(derive_seed(7, "a", 1), derive_seed(7, "a", 1))
        self.assertNotEqual(derive_seed(7, "a", 1), derive_seed(7, "a", 2))
        self.assertNotEqual(derive_seed(7, "a"), derive_seed(8, "a"))
        seeds = {child.seed for child in RandomStream(1).spawn(1000)}
        self.assertEqual(len(seeds), 1000)

    def test_splitmix_reference_value(self) -> None:
        # First output of SplitMix64 seeded with 0 (Steele, Lea & Flood 2014).
        """Splitmix reference value."""
        self.assertEqual(splitmix64(0), 0xE220A8397B1DCDAF)

    def test_rejects_invalid_keys(self) -> None:
        """Rejects invalid keys."""
        with self.assertRaises(ValueError):
            derive_seed(-1)
        with self.assertRaises(TypeError):
            derive_seed(1, 2.5)  # type: ignore[arg-type]


class SamplingDesignTests(unittest.TestCase):
    """Stratification and low-discrepancy properties."""

    def test_sobol_first_points(self) -> None:
        """Sobol first points."""
        points = sobol(8, 2).points
        self.assertEqual(
            [round(p[1], 6) for p in points[1:]],
            [0.5, 0.25, 0.75, 0.375, 0.875, 0.125, 0.625],
        )

    def test_sobol_polynomials_are_primitive(self) -> None:
        """Sobol polynomials are primitive."""
        for degree, code, initial in _SOBOL_TABLE:
            self.assertTrue(_is_primitive(degree, code))
            for k, m in enumerate(initial):
                self.assertEqual(m % 2, 1)
                self.assertLess(m, 2 ** (k + 1))

    def test_sobol_one_dimensional_balance(self) -> None:
        """Sobol one dimensional balance."""
        for scramble in (False, True):
            design = sobol(256, 16, scramble=scramble, seed=3)
            for axis in range(16):
                bins = sorted(int(point[axis] * 256) for point in design.points)
                self.assertEqual(bins, list(range(256)))

    def test_sobol_two_dimensional_net(self) -> None:
        """Sobol two dimensional net."""
        design = sobol(256, 2)
        for k in range(9):
            cells = {
                (int(p[0] * 2**k), int(p[1] * 2 ** (8 - k))) for p in design.points
            }
            self.assertEqual(len(cells), 256)

    def test_latin_hypercube_strata(self) -> None:
        """Latin hypercube strata."""
        design = latin_hypercube(50, 4, seed=11)
        for axis in range(4):
            self.assertEqual(
                sorted(int(p[axis] * 50) for p in design.points), list(range(50))
            )

    def test_low_discrepancy_beats_random(self) -> None:
        """Low discrepancy beats random."""
        random_design = latin_hypercube(128, 2, seed=1)
        self.assertLess(
            star_discrepancy_bound(sobol(128, 2)), star_discrepancy_bound(random_design)
        )
        self.assertLess(star_discrepancy_bound(halton(128, 2)), 0.05)
        scrambled = halton(64, 3, scramble=True, shift=True, seed=5)
        self.assertTrue(all(0.0 < v < 1.0 for p in scrambled.points for v in p))


class OptimizationTests(unittest.TestCase):
    """Minimizers, least squares, and root finding."""

    @staticmethod
    def rosenbrock(x):
        """Rosenbrock."""
        return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

    def test_bfgs_and_nelder_mead_find_rosenbrock_minimum(self) -> None:
        """Bfgs and nelder mead find rosenbrock minimum."""
        for result in (
            minimize_bfgs(self.rosenbrock, (-1.2, 1.0)),
            minimize_nelder_mead(self.rosenbrock, (-1.2, 1.0)),
        ):
            self.assertTrue(result.converged, result.message)
            self.assertAlmostEqual(result.x[0], 1.0, places=4)
            self.assertAlmostEqual(result.x[1], 1.0, places=4)

    def test_levenberg_marquardt_recovers_exponential(self) -> None:
        """Levenberg marquardt recovers exponential."""
        xs = [0.5 * i for i in range(20)]
        ys = [
            2.5 * math.exp(-0.3 * x) + (0.01 if i % 2 else -0.01)
            for i, x in enumerate(xs)
        ]
        result = least_squares(
            lambda p: [
                p[0] * math.exp(-p[1] * x) - y for x, y in zip(xs, ys, strict=True)
            ],
            (1.0, 1.0),
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.x[0], 2.5, places=1)
        self.assertAlmostEqual(result.x[1], 0.3, places=2)
        covariance = gauss_newton_covariance(result)
        self.assertGreater(covariance[0][0], 0.0)

    def test_brent_and_hessian(self) -> None:
        """Brent and hessian."""
        root = brent_root(lambda x: math.cos(x) - x, 0.0, 1.0)
        self.assertAlmostEqual(root.root, 0.7390851332151607, places=11)
        with self.assertRaises(ValueError):
            brent_root(lambda x: x * x + 1.0, -1.0, 1.0)
        hessian, _ = central_hessian(
            lambda x: x[0] ** 2 * x[1] + math.sin(x[1]), (1.0, 2.0)
        )
        self.assertAlmostEqual(hessian[0][0], 4.0, places=5)
        self.assertAlmostEqual(hessian[0][1], 2.0, places=5)
        self.assertAlmostEqual(hessian[1][1], -math.sin(2.0), places=5)


class OrderStatisticTests(unittest.TestCase):
    """Quantile conventions and shape summaries."""

    def test_type7_and_type1_quantiles(self) -> None:
        """Type7 and type1 quantiles."""
        data = (3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0)
        self.assertAlmostEqual(quantile(data, 0.25), 1.75)
        self.assertAlmostEqual(quantile(data, 0.25, method=1), 1.0)
        self.assertAlmostEqual(median(data), 3.5)
        self.assertAlmostEqual(interquartile_range(data), 5.25 - 1.75)
        self.assertEqual(quantile(data, 0.0), 1.0)
        self.assertEqual(quantile(data, 1.0), 9.0)

    def test_ranks_and_spearman(self) -> None:
        """Ranks and spearman."""
        self.assertEqual(ranks((10.0, 20.0, 20.0, 5.0)), (2.0, 3.5, 3.5, 1.0))
        x = [float(i) for i in range(10)]
        self.assertAlmostEqual(spearman_correlation(x, [v**3 for v in x]), 1.0)

    def test_shape_and_robust_scale(self) -> None:
        """Shape and robust scale."""
        sample = Normal(0.0, 2.0).sample(20000, seed=3)
        self.assertAlmostEqual(skewness(sample), 0.0, delta=0.06)
        self.assertAlmostEqual(excess_kurtosis(sample), 0.0, delta=0.12)
        self.assertAlmostEqual(median_absolute_deviation(sample), 2.0, delta=0.06)


class ModelingTests(unittest.TestCase):
    """Joint input laws and the probabilistic model contract."""

    def test_gauss_hermite_moments(self) -> None:
        """Gauss hermite moments."""
        nodes, weights = gauss_hermite(8)
        moments = [
            math.fsum(w * z**k for z, w in zip(nodes, weights, strict=True))
            for k in range(8)
        ]
        for k, expected in enumerate((1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 15.0, 0.0)):
            self.assertAlmostEqual(moments[k], expected, places=10)

    def test_standard_normal_helpers(self) -> None:
        """Standard normal helpers."""
        for p in (1e-300, 1e-10, 0.3, 0.999999):
            self.assertAlmostEqual(
                standard_normal_cdf(standard_normal_quantile(p)) / p, 1.0, places=11
            )

    def test_nataf_matches_lognormal_closed_form(self) -> None:
        """Nataf matches lognormal closed form."""
        first, second = LogNormal(0.0, 0.5), LogNormal(0.0, 0.8)
        target = 0.6
        exact = (
            math.log(
                1.0
                + target * math.sqrt((math.exp(0.25) - 1.0) * (math.exp(0.64) - 1.0))
            )
            / 0.4
        )
        self.assertAlmostEqual(
            nataf_correlation(first, second, target), exact, places=9
        )

    def test_unattainable_correlation_is_reported(self) -> None:
        """Unattainable correlation is reported."""
        with self.assertRaises(ValueError):
            nataf_correlation(LogNormal(0.0, 2.0), LogNormal(0.0, 2.0), -0.9)

    def test_copula_density_equals_multivariate_normal(self) -> None:
        """Copula density equals multivariate normal."""
        joint = JointInput(
            (RandomInput("x", Normal(1.0, 2.0)), RandomInput("y", Normal(-1.0, 0.5))),
            ((1.0, 0.3), (0.3, 1.0)),
        )
        reference = MultivariateNormal((1.0, -1.0), ((4.0, 0.3), (0.3, 0.25)))
        for point in ((0.2, -0.6), (3.0, -2.0)):
            self.assertAlmostEqual(
                joint.log_pdf(point), reference.log_pdf(point), places=12
            )

    def test_transform_round_trip_and_sample_correlation(self) -> None:
        """Transform round trip and sample correlation."""
        joint = JointInput.from_pearson(
            (
                RandomInput("a", LogNormal(0.0, 0.5)),
                RandomInput("b", LogNormal(0.0, 0.8)),
            ),
            ((1.0, 0.6), (0.6, 1.0)),
        )
        point = (1.3, 0.7)
        back = joint.from_standard_normal(joint.to_standard_normal(point))
        self.assertAlmostEqual(back[0], 1.3, places=12)
        self.assertAlmostEqual(back[1], 0.7, places=12)
        draws = joint.sample(30000, seed=1)
        observed = statistics.correlation([d[0] for d in draws], [d[1] for d in draws])
        self.assertAlmostEqual(observed, 0.6, delta=0.03)

    def test_model_validation_and_counting(self) -> None:
        """Model validation and counting."""
        model = _ishigami()
        self.assertEqual(model.dimension, 3)
        model.evaluate((0.1, 0.2, 0.3))
        model.scalar_function("y")((0.0, 0.0, 0.0))
        self.assertEqual(model.evaluations, 2)
        with self.assertRaises(KeyError):
            model.scalar_function("missing")
        with self.assertRaises(ValueError):
            ProbabilisticModel(model.inputs, model.model, ())
        with self.assertRaises(ValueError):
            ProbabilisticModel(
                model.inputs, model.model, model.quantities, fixed={"x1": 1.0}
            )


class PropagationStudyTests(unittest.TestCase):
    """Model-based forward propagation."""

    def test_ishigami_moments_and_quantiles(self) -> None:
        """Ishigami moments and quantiles."""
        study = propagate(_ishigami(), 4096, design="sobol", seed=7)
        output = study["y"]
        self.assertAlmostEqual(output.mean, 3.5, delta=0.01)
        self.assertAlmostEqual(
            output.standard_deviation, math.sqrt(13.844587940719254), delta=0.02
        )
        self.assertEqual(study.model_evaluations, 4096)
        self.assertLess(output.quantile(0.05), output.quantile(0.95))
        curve = output.exceedance_curve(10)
        self.assertEqual(curve[0][1], 1.0 - 1.0 / 4096)
        self.assertEqual(curve[-1][1], 0.0)
        correlations = study.rank_correlations("y")
        self.assertGreater(abs(correlations["x1"]), abs(correlations["x3"]))

    def test_quasi_random_designs_reduce_error(self) -> None:
        """Quasi random designs reduce error."""
        summary = compare_designs(_ishigami(), "y", 256, replicates=8, seed=1)
        self.assertLess(summary["sobol"][1], summary["mc"][1])
        self.assertLess(summary["lhs"][1], summary["mc"][1])

    def test_same_seed_reproduces_study(self) -> None:
        """Same seed reproduces study."""
        first = propagate(_ishigami(), 64, design="lhs", seed=5)
        second = propagate(_ishigami(), 64, design="lhs", seed=5)
        self.assertEqual(first["y"].values, second["y"].values)
        with self.assertRaises(ValueError):
            propagate(_ishigami(), 64, design="grid")


if __name__ == "__main__":
    unittest.main()
