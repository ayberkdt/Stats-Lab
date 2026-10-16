"""Copula fitting by Kendall's tau inversion and maximum pseudo-likelihood.

Reference values were computed outside the repository with scipy 1.18:
``scipy.stats.kendalltau`` (tau-b) and ``scipy.optimize.minimize_scalar``
(bounded, ``xatol=1e-12``) on the pseudo-log-likelihood written with the
textbook bivariate densities of Nelsen (2006) in numpy. No test imports numpy,
scipy, or mpmath.
"""

from __future__ import annotations

import math
import unittest
import warnings

from probstat.modeling import (
    ClaytonCopula,
    GaussianCopula,
    StudentTCopula,
    archimedean_theta_from_tau,
    fit_copula,
    frank_tau,
    kendall_tau,
    kendall_to_latent_correlation,
    pseudo_observations,
)

X = (0.62, 1.95, 0.33, 2.71, 1.12, 0.08, 1.48, 3.30, 0.91, 0.47)
X += (2.05, 1.70, 0.21, 0.76, 2.42, 1.31, 0.55, 2.98, 1.02, 0.15)
Y = (0.80, 1.70, 0.25, 2.20, 1.40, 0.30, 0.95, 3.10, 1.30, 0.20)
Y += (1.75, 2.30, 0.45, 0.60, 2.00, 0.85, 0.75, 2.60, 1.60, 0.05)


class KendallTauTests(unittest.TestCase):
    """Sample Kendall's tau-b."""

    def test_against_scipy(self) -> None:
        """Tau-b with ties and tau-a without ties match scipy."""
        xa = (1.2, 3.4, 2.2, 5.0, 4.1, 0.3, 2.2, 6.0, 3.3, 1.1)
        ya = (0.5, 2.0, 2.5, 4.4, 3.0, 0.1, 1.0, 5.5, 3.0, 2.0)
        self.assertAlmostEqual(kendall_tau(xa, ya), 0.7816608327818949, delta=1e-15)
        xb = (0.11, 0.52, 0.33, 0.95, 0.74, 0.28, 0.61, 0.07, 0.86, 0.43)
        yb = (0.20, 0.47, 0.41, 0.88, 0.52, 0.35, 0.72, 0.05, 0.69, 0.30)
        self.assertAlmostEqual(kendall_tau(xb, yb), 0.8222222222222221, delta=1e-15)
        self.assertAlmostEqual(kendall_tau(X, Y), 0.8, delta=1e-15)

    def test_errors(self) -> None:
        """Constant samples, short samples and length mismatch raise."""
        with self.assertRaises(ValueError):
            kendall_tau((1.0, 1.0, 1.0), (1.0, 2.0, 3.0))
        with self.assertRaises(ValueError):
            kendall_tau((1.0,), (2.0,))
        with self.assertRaises(ValueError):
            kendall_tau((1.0, 2.0), (1.0, 2.0, 3.0))

    def test_pseudo_observations(self) -> None:
        """Pseudo-observations are ranks over ``n + 1``."""
        names, rows = pseudo_observations({"a": (3.0, 1.0, 2.0), "b": (1.0, 2.0, 3.0)})
        self.assertEqual(names, ("a", "b"))
        self.assertEqual(rows, ((0.75, 0.25), (0.25, 0.5), (0.5, 0.75)))


class TauInversionTests(unittest.TestCase):
    """Closed-form and numerical inversions of Kendall's tau."""

    def test_closed_forms(self) -> None:
        """Clayton ``2t/(1-t)``, Gumbel ``1/(1-t)``, elliptical ``sin(pi t/2)``."""
        self.assertAlmostEqual(
            archimedean_theta_from_tau("clayton", 0.6), 3.0, delta=1e-14
        )
        self.assertAlmostEqual(
            archimedean_theta_from_tau("gumbel", 0.75), 4.0, delta=1e-14
        )
        self.assertAlmostEqual(
            kendall_to_latent_correlation(1.0 / 3.0), 0.5, delta=1e-15
        )
        self.assertEqual(kendall_to_latent_correlation(0.0), 0.0)

    def test_frank_inversion(self) -> None:
        """The Frank inverse reproduces tau through the Debye function."""
        for tau in (-0.6, -0.1, 0.05, 0.5, 0.9):
            theta = archimedean_theta_from_tau("frank", tau)
            self.assertAlmostEqual(frank_tau(theta), tau, delta=1e-12)
        with self.assertRaises(ValueError):
            archimedean_theta_from_tau("frank", -0.3, dimension=3)
        with self.assertRaises(ValueError):
            archimedean_theta_from_tau("clayton", -0.2)
        with self.assertRaises(ValueError):
            archimedean_theta_from_tau("joe", 0.2)

    def test_fit_by_kendall(self) -> None:
        """Sample tau 0.8 gives Clayton 8, Gumbel 5, and the Frank root."""
        data = {"x": X, "y": Y}
        self.assertAlmostEqual(fit_copula(data, "clayton").parameter, 8.0, delta=1e-12)
        self.assertAlmostEqual(fit_copula(data, "gumbel").parameter, 5.0, delta=1e-12)
        frank = fit_copula(data, "frank").parameter
        assert frank is not None
        self.assertAlmostEqual(frank_tau(frank), 0.8, delta=1e-12)
        gaussian = fit_copula(data, "gaussian")
        assert isinstance(gaussian.copula, GaussianCopula)
        self.assertAlmostEqual(
            gaussian.copula.correlation[0][1], math.sin(0.4 * math.pi), delta=1e-15
        )
        t_fit = fit_copula(data, "student-t", nu=5.0)
        self.assertEqual(t_fit.parameter, 5.0)
        independence = fit_copula(data, "independence")
        self.assertEqual(independence.log_pseudo_likelihood, 0.0)


class PseudoLikelihoodTests(unittest.TestCase):
    """Maximum pseudo-likelihood against scipy's bounded optimizer."""

    def test_one_parameter_families(self) -> None:
        """Clayton, Gumbel and Frank estimates and maxima match scipy."""
        reference = {
            "clayton": (3.2385731918877396, 12.081321231818793),
            "gumbel": (4.719686359649133, 20.806773605018705),
            "frank": (15.95787257496799, 19.092883895596625),
        }
        for family, (theta, value) in reference.items():
            fit = fit_copula({"x": X, "y": Y}, family, method="pseudo-likelihood")
            with self.subTest(family=family):
                assert fit.parameter is not None
                self.assertFalse(fit.at_boundary)
                self.assertAlmostEqual(fit.parameter / theta, 1.0, delta=2e-6)
                self.assertAlmostEqual(fit.log_pseudo_likelihood, value, delta=1e-9)

    def test_recovers_parameters_from_samples(self) -> None:
        """Clayton(3) samples: both estimators land near 3 (seeded)."""
        sample = ClaytonCopula(3.0).sample(800, seed=21)
        data = {"u": [row[0] for row in sample], "v": [row[1] for row in sample]}
        by_tau = fit_copula(data, "clayton").parameter
        by_likelihood = fit_copula(
            data, "clayton", method="pseudo-likelihood"
        ).parameter
        assert by_tau is not None and by_likelihood is not None
        self.assertLess(abs(by_tau - 3.0), 0.5)
        self.assertLess(abs(by_likelihood - 3.0), 0.5)

    def test_t_degrees_of_freedom(self) -> None:
        """Profile pseudo-likelihood over ``nu`` with ``R`` from tau (seeded)."""
        sample = StudentTCopula(((1.0, 0.5), (0.5, 1.0)), 3.0).sample(300, seed=22)
        data = {"u": [row[0] for row in sample], "v": [row[1] for row in sample]}
        fit = fit_copula(data, "student-t", method="pseudo-likelihood")
        assert fit.parameter is not None
        self.assertGreater(fit.parameter, 1.0)
        self.assertLess(fit.parameter, 15.0)

    def test_errors(self) -> None:
        """Unsupported combinations raise."""
        data = {"x": X, "y": Y}
        with self.assertRaises(ValueError):
            fit_copula(data, "gaussian", method="pseudo-likelihood")
        with self.assertRaises(ValueError):
            fit_copula(data, "student-t")
        with self.assertRaises(ValueError):
            fit_copula(data, "joe")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fit_copula({"x": X}, "clayton")
        negative = {"x": X, "y": tuple(-v for v in Y)}
        with self.assertRaises(ValueError):
            fit_copula(negative, "clayton")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fit_copula(data, "gaussian")


if __name__ == "__main__":
    unittest.main()
