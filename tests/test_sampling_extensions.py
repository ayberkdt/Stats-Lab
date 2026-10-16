"""Joe-Kuo Sobol, Owen scrambling, randomized Halton, maximin and Iman-Conover LHS.

Sobol references were produced in a separate verification environment with
``scipy.stats.qmc.Sobol(d, scramble=False).random(n)`` (SciPy 1.18): the
points of the first ``n = 2^m`` indices are exact multiples of ``2^-m``, so
the integer matrix ``round(2^m x)`` is compared through its SHA-256 digest
(and a few entries are listed explicitly).
"""

from __future__ import annotations

import hashlib
import math
import statistics
import unittest
import warnings

from probstat.sampling import (
    SOBOL_MAX_DIMENSION,
    SobolBalanceWarning,
    halton,
    iman_conover,
    latin_hypercube,
    maximin_criterion,
    maximin_latin_hypercube,
    sobol,
)

SCIPY_SOBOL_DIGESTS = {
    (50, 32): "8f49e334a3a6e91a20409e9c31e54fb4f14777b7c90338cf85488d46d513fe6a",
    (1024, 32): "77fd0942aa73ac5540e233e0163afe81d9d6ea92c55e2c2c9c1f2b8bc1708200",
    (50, 1024): "35d9be95d9386df18ecd6b6bca2293f567311fa00a074f6f040e5500dcc80b30",
}
# round(32 x) for points 1, 2, 3, 5, 7, 31 at dimensions 1, 2, 16, 17, 18, 50.
SCIPY_SELECTED = [
    [16, 16, 16, 16, 16, 16],
    [24, 8, 8, 24, 8, 24],
    [8, 24, 24, 8, 24, 8],
    [28, 28, 12, 28, 12, 28],
    [4, 20, 4, 4, 4, 4],
    [1, 17, 27, 7, 13, 23],
]
# ... and at dimensions 101, 512, 778, 1024 of the 1024-dimensional sequence.
SCIPY_HIGH = [
    [16, 16, 16, 16],
    [8, 24, 8, 24],
    [24, 8, 24, 8],
    [4, 28, 28, 12],
    [12, 4, 20, 20],
    [29, 25, 5, 11],
]


def digest(points: tuple[tuple[float, ...], ...], size: int) -> str:
    """Return the SHA-256 of ``round(size x)`` over all coordinates."""
    values = (str(round(x * size)) for point in points for x in point)
    return hashlib.sha256(",".join(values).encode()).hexdigest()


def spearman(first: list[float], second: list[float]) -> float:
    """Return the Spearman rank correlation of two tie-free samples."""

    def ranks(values: list[float]) -> list[float]:
        """Return 0-based ranks."""
        order = sorted(range(len(values)), key=values.__getitem__)
        result = [0.0] * len(values)
        for rank, index in enumerate(order):
            result[index] = float(rank)
        return result

    return statistics.correlation(ranks(first), ranks(second))


class JoeKuoSobolTests(unittest.TestCase):
    """Defect 0.6: Sobol points beyond 16 dimensions."""

    def test_matches_scipy_unscrambled_points(self) -> None:
        """Raw points agree exactly with SciPy for d = 50 and d = 1024."""
        self.assertEqual(SOBOL_MAX_DIMENSION, 1024)
        for (dimension, size), expected in SCIPY_SOBOL_DIGESTS.items():
            with self.subTest(dimension=dimension, size=size):
                design = sobol(size, dimension, centered=False)
                self.assertEqual(digest(design.points, size), expected)
        points = sobol(32, 1024, centered=False).points
        rows = (1, 2, 3, 5, 7, 31)
        self.assertEqual(
            [[round(32 * points[i][j]) for j in (0, 1, 15, 16, 17, 49)] for i in rows],
            SCIPY_SELECTED,
        )
        self.assertEqual(
            [[round(32 * points[i][j]) for j in (100, 511, 777, 1023)] for i in rows],
            SCIPY_HIGH,
        )
        with self.assertRaises(ValueError):
            sobol(4, 1025)

    def test_non_power_of_two_warns(self) -> None:
        """A size that is not a power of two emits a balance warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sobol(100, 3)
            sobol(128, 3)
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, SobolBalanceWarning)

    def test_unscrambled_points_are_cell_centres(self) -> None:
        """No origin; nets and one-dimensional balance survive the centring."""
        design = sobol(256, 20)
        self.assertTrue(all(0.0 < x < 1.0 for p in design.points for x in p))
        self.assertEqual(design.points[0], (1.0 / 512.0,) * 20)
        for axis in range(20):
            column = [p[axis] for p in design.points]
            self.assertEqual(sorted(int(x * 256) for x in column), list(range(256)))
            self.assertEqual(math.fsum(column) / 256, 0.5)
        pair = sobol(256, 2)
        for k in range(9):
            cells = {(int(p[0] * 2**k), int(p[1] * 2 ** (8 - k))) for p in pair.points}
            self.assertEqual(len(cells), 256)

    def test_owen_scrambling_keeps_nets_and_is_unbiased(self) -> None:
        """Owen-scrambled nets stay nets; points are uniform; variance drops."""
        design = sobol(256, 2, scramble="owen", seed=4)
        for k in range(9):
            cells = {
                (int(p[0] * 2**k), int(p[1] * 2 ** (8 - k))) for p in design.points
            }
            self.assertEqual(len(cells), 256)
        self.assertEqual(design.points, sobol(256, 2, scramble="owen", seed=4).points)
        firsts = [
            sobol(2, 1, scramble="owen", seed=s).points[0][0] for s in range(2000)
        ]
        self.assertLess(
            abs(statistics.fmean(firsts) - 0.5), 4 * math.sqrt(1 / 12 / 2000)
        )

        def integrand(point: tuple[float, ...]) -> float:
            """Return ``exp(x + y)``."""
            return math.exp(point[0] + point[1])

        exact = (math.e - 1.0) ** 2
        estimates = {}
        for mode in ("shift", "owen"):
            estimates[mode] = [
                statistics.fmean(
                    integrand(p) for p in sobol(256, 2, scramble=mode, seed=s).points
                )
                for s in range(120)
            ]
        owen = estimates["owen"]
        self.assertLess(
            abs(statistics.fmean(owen) - exact),
            4 * statistics.stdev(owen) / math.sqrt(120),
        )
        self.assertLess(
            statistics.stdev(owen), statistics.stdev(estimates["shift"]) / 3
        )
        with self.assertRaises(ValueError):
            sobol(4, 2, scramble="matousek", seed=1)


class HaltonRandomizationTests(unittest.TestCase):
    """Defect P2: randomized Halton points were biased towards zero."""

    def test_scrambled_points_are_uniform(self) -> None:
        """Over seeds, the first point's coordinates average to one half."""
        draws = [halton(1, 3, scramble=True, seed=s).points[0] for s in range(1500)]
        tolerance = 4 * math.sqrt(1 / 12 / 1500)
        for axis in range(3):
            mean = statistics.fmean(p[axis] for p in draws)
            self.assertLess(abs(mean - 0.5), tolerance)

    def test_scrambled_estimator_is_unbiased_and_stratified(self) -> None:
        """The randomized estimator is unbiased; base-b strata stay filled."""

        def integrand(point: tuple[float, ...]) -> float:
            """Return ``exp(x + y)``."""
            return math.exp(point[0] + point[1])

        exact = (math.e - 1.0) ** 2
        estimates = [
            statistics.fmean(
                integrand(p) for p in halton(100, 2, scramble=True, seed=s).points
            )
            for s in range(200)
        ]
        spread = statistics.stdev(estimates) / math.sqrt(len(estimates))
        self.assertLess(abs(statistics.fmean(estimates) - exact), 4 * spread)
        design = halton(30, 3, scramble=True, seed=9)
        for axis, base in enumerate((2, 3, 5)):
            cells = sorted(int(p[axis] * base) for p in design.points[:base])
            self.assertEqual(cells, list(range(base)))
        self.assertEqual(design.points, halton(30, 3, scramble=True, seed=9).points)


class LatinHypercubeExtensionTests(unittest.TestCase):
    """Maximin search and Iman-Conover rank correlation."""

    def test_maximin_improves_and_keeps_strata(self) -> None:
        """The search lowers ``phi_p`` and preserves the Latin property."""
        plain = latin_hypercube(40, 3, seed=7)
        start = latin_hypercube(40, 3, seed=7)
        improved = maximin_latin_hypercube(40, 3, seed=7, iterations=3000)
        self.assertEqual(
            improved.points,
            maximin_latin_hypercube(40, 3, seed=7, iterations=3000).points,
        )
        for axis in range(3):
            self.assertEqual(
                sorted(int(p[axis] * 40) for p in improved.points), list(range(40))
            )
        self.assertLess(maximin_criterion(improved), maximin_criterion(plain))
        self.assertLess(maximin_criterion(improved), maximin_criterion(start))

        def min_distance(points: tuple[tuple[float, ...], ...]) -> float:
            """Return the minimum pairwise distance."""
            return min(
                math.dist(a, b) for i, a in enumerate(points) for b in points[i + 1 :]
            )

        self.assertGreater(min_distance(improved.points), min_distance(plain.points))

    def test_iman_conover_rank_correlation(self) -> None:
        """Target Spearman correlations are met closely; strata are preserved."""
        target = ((1.0, 0.7, -0.3), (0.7, 1.0, 0.2), (-0.3, 0.2, 1.0))
        base = latin_hypercube(600, 3, seed=3)
        design = iman_conover(base, target, seed=3)
        for axis in range(3):
            self.assertEqual(
                sorted(p[axis] for p in design.points),
                sorted(p[axis] for p in base.points),
            )
        columns = [[p[a] for p in design.points] for a in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertAlmostEqual(
                    spearman(columns[i], columns[j]), target[i][j], delta=0.04
                )
        independent = [[p[a] for p in base.points] for a in range(3)]
        self.assertLess(abs(spearman(independent[0], independent[1])), 0.12)
        self.assertEqual(design.points, iman_conover(base, target, seed=3).points)
        with self.assertRaises(ValueError):
            iman_conover(base, ((1.0, 2.0, 0.0), (2.0, 1.0, 0.0), (0.0, 0.0, 1.0)))


if __name__ == "__main__":
    unittest.main()
