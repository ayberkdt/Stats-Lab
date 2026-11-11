"""Tests for the structural multivariate protocols in ``probstat.foundations``.

Every vector-valued family conforms by structure (no shared base class). A
protocol-generic Monte Carlo check then uses only the protocol members: the
sample mean of seeded draws must match the declared mean vector within four
standard errors computed from the declared covariance diagonal, and
``log_pdf`` must equal the closed-form bivariate normal log density.
"""

from __future__ import annotations

import math
import unittest

from probstat.distributions import Normal
from probstat.foundations import (
    HasDimension,
    MultivariateLogDensity,
    MultivariateLogMass,
    MultivariateMoments,
    MultivariateSampler,
)
from probstat.multivariate import (
    Dirichlet,
    DirichletMultinomial,
    Multinomial,
    MultivariateNormal,
    MultivariateStudentT,
)


def _check_sample_mean(test: unittest.TestCase, law: object, seed: int) -> None:
    """Compare the sample mean with the declared mean using protocol members."""
    assert isinstance(law, MultivariateSampler)
    assert isinstance(law, MultivariateMoments)
    draws = law.sample(4000, seed=seed)
    test.assertEqual(len(draws[0]), law.dimension)
    for j in range(law.dimension):
        mean = sum(float(row[j]) for row in draws) / len(draws)
        se = math.sqrt(law.covariance[j][j] / len(draws))
        test.assertAlmostEqual(mean, law.mean[j], delta=4 * se + 1e-12)


class ConformanceTests(unittest.TestCase):
    """The analytic families satisfy the protocols they should."""

    def test_continuous_laws(self) -> None:
        """Normal, Student-t, and Dirichlet laws have densities and moments."""
        laws = (
            MultivariateNormal((0.0, 1.0), ((2.0, 0.5), (0.5, 1.0))),
            MultivariateStudentT((0.0, 1.0), ((2.0, 0.5), (0.5, 1.0)), 6.0),
            Dirichlet((2.0, 3.0, 4.0)),
        )
        for law in laws:
            self.assertIsInstance(law, HasDimension)
            self.assertIsInstance(law, MultivariateLogDensity)
            self.assertIsInstance(law, MultivariateMoments)
            self.assertIsInstance(law, MultivariateSampler)
            self.assertNotIsInstance(law, MultivariateLogMass)

    def test_discrete_laws(self) -> None:
        """Multinomial laws have mass functions, moments, and samplers."""
        for law in (
            Multinomial(10, (0.2, 0.3, 0.5)),
            DirichletMultinomial(10, (1.0, 2.0, 3.0)),
        ):
            self.assertIsInstance(law, MultivariateLogMass)
            self.assertIsInstance(law, MultivariateMoments)
            self.assertIsInstance(law, MultivariateSampler)
            self.assertNotIsInstance(law, MultivariateLogDensity)

    def test_univariate_law_is_not_multivariate(self) -> None:
        """A scalar law has no dimension and conforms to none of them."""
        self.assertNotIsInstance(Normal(0.0, 1.0), HasDimension)
        self.assertNotIsInstance(object(), MultivariateSampler)


class ProtocolGenericTests(unittest.TestCase):
    """Generic code written against the protocols gives correct answers."""

    def test_sample_means_match_declared_means(self) -> None:
        """For every family the sample mean agrees with the declared mean."""
        laws = (
            MultivariateNormal((0.0, 1.0), ((2.0, 0.5), (0.5, 1.0))),
            MultivariateStudentT((0.0, 1.0), ((2.0, 0.5), (0.5, 1.0)), 6.0),
            Dirichlet((2.0, 3.0, 4.0)),
            Multinomial(10, (0.2, 0.3, 0.5)),
            DirichletMultinomial(10, (1.0, 2.0, 3.0)),
        )
        for seed, law in enumerate(laws):
            _check_sample_mean(self, law, seed)

    def test_log_density_matches_bivariate_normal_formula(self) -> None:
        """log_pdf through the protocol equals the closed-form density."""
        law: MultivariateLogDensity = MultivariateNormal(
            (0.0, 1.0), ((2.0, 0.5), (0.5, 1.0))
        )
        x, y = 0.3, 0.2
        det = 2.0 * 1.0 - 0.25
        dx, dy = x - 0.0, y - 1.0
        quadratic = (1.0 * dx * dx - 2 * 0.5 * dx * dy + 2.0 * dy * dy) / det
        expected = -math.log(2 * math.pi) - 0.5 * math.log(det) - 0.5 * quadratic
        self.assertAlmostEqual(law.log_pdf((x, y)), expected, places=12)


if __name__ == "__main__":
    unittest.main()
