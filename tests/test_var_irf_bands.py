"""Residual-bootstrap bands for VAR impulse responses.

Oracles: at horizon 0 the non-orthogonalized response is the identity, so
its band is degenerate; at horizon 1 it is ``Phi_1 = A_1``, so the bootstrap
standard errors must approach the least-squares coefficient standard errors
of [EQ-TS-VAR-001] (Monte Carlo error of a standard deviation from ``R``
draws is about ``1 / sqrt(2 R)``, 2.2% for ``R = 999``); Hall's interval is
the percentile interval reflected about the estimate; and on simulated data
from a known VAR(1) the 90% bands cover the true orthogonalized responses at
close to the nominal rate.

References:
    [REF-RUNKLE-1987], [REF-HALL-1992], [REF-LUTKEPOHL-2005, Chapter 3].

"""

from __future__ import annotations

import unittest
from dataclasses import replace

from probstat.linalg import Matrix, cholesky, matmul
from probstat.timeseries import ImpulseResponseBands, fit_var, simulate_var

A1 = ((0.5, 0.1), (0.2, 0.3))
SIGMA = ((1.0, 0.3), (0.3, 0.5))


def true_orthogonalized(horizon: int) -> list[Matrix]:
    """Return ``Theta_h = A_1^h P`` for the VAR(1) used here."""
    factor = cholesky(SIGMA)
    phi: Matrix = ((1.0, 0.0), (0.0, 1.0))
    responses = [matmul(phi, factor)]
    for _ in range(horizon):
        phi = matmul(phi, A1)
        responses.append(matmul(phi, factor))
    return responses


class ImpulseResponseBandTests(unittest.TestCase):
    """Bands of a two-variable VAR(1) fitted to 300 simulated rows."""

    @classmethod
    def setUpClass(cls) -> None:
        """Fit the VAR once."""
        cls.fit = fit_var(simulate_var((A1,), SIGMA, 300, seed=11), 1)

    def test_bootstrap_errors_match_least_squares_errors_at_horizon_one(self) -> None:
        """``Phi_1 = A_1``: bootstrap and OLS standard errors agree."""
        bands = self.fit.impulse_response_bands(
            1, replications=999, orthogonalized=False, seed=3
        )
        for i in range(2):
            for j in range(2):
                ols = self.fit.coefficient_standard_errors[0][i][j]
                self.assertAlmostEqual(
                    bands.standard_errors[1][i][j], ols, delta=0.12 * ols
                )

    def test_horizon_zero_unit_shock_band_is_degenerate(self) -> None:
        """``Phi_0 = I`` in every replication."""
        bands = self.fit.impulse_response_bands(
            2, replications=20, orthogonalized=False, seed=1
        )
        identity = ((1.0, 0.0), (0.0, 1.0))
        self.assertEqual(bands.lower[0], identity)
        self.assertEqual(bands.upper[0], identity)
        self.assertEqual(bands.standard_errors[0], ((0.0, 0.0), (0.0, 0.0)))

    def test_hall_interval_reflects_the_percentile_interval(self) -> None:
        """``[2 theta - q_hi, 2 theta - q_lo]`` from the same draws."""
        percentile = self.fit.impulse_response_bands(4, replications=99, seed=5)
        hall = self.fit.impulse_response_bands(
            4, replications=99, seed=5, interval="hall"
        )
        for h in range(5):
            for i in range(2):
                for j in range(2):
                    theta = percentile.point[h][i][j]
                    self.assertAlmostEqual(
                        hall.lower[h][i][j],
                        2.0 * theta - percentile.upper[h][i][j],
                        places=12,
                    )
                    self.assertAlmostEqual(
                        hall.upper[h][i][j],
                        2.0 * theta - percentile.lower[h][i][j],
                        places=12,
                    )

    def test_same_seed_same_bands(self) -> None:
        """Replication ``r`` draws from child ``("irf", r)`` of the seed."""
        first = self.fit.impulse_response_bands(3, replications=30, seed=7)
        second = self.fit.impulse_response_bands(3, replications=30, seed=7)
        self.assertEqual(first, second)
        self.assertIsInstance(first, ImpulseResponseBands)
        self.assertEqual(first.point, self.fit.impulse_response(3))

    def test_invalid_arguments(self) -> None:
        """Replications, level, interval, and horizon are validated."""
        with self.assertRaises(ValueError):
            self.fit.impulse_response_bands(2, replications=1)
        with self.assertRaises(ValueError):
            self.fit.impulse_response_bands(2, level=1.0)
        with self.assertRaises(ValueError):
            self.fit.impulse_response_bands(2, interval="bca")
        with self.assertRaises(ValueError):
            self.fit.impulse_response_bands(-1)


class ImpulseResponseCoverageTests(unittest.TestCase):
    """Pointwise coverage on 30 simulated series of length 200."""

    def test_ninety_percent_bands_cover_near_the_nominal_rate(self) -> None:
        """At least 22 of 30 (``P(X <= 21) < 0.02`` at the nominal 0.9).

        The bootstrap inherits the downward bias of least-squares VAR
        coefficients, so coverage sits somewhat below nominal in samples of
        this size [REF-KILIAN-1998]; a collapse well below it would fail.
        """
        truth = true_orthogonalized(3)
        hits = {(1, 0, 0): 0, (2, 1, 0): 0, (3, 0, 1): 0}
        for dataset in range(30):
            fit = fit_var(simulate_var((A1,), SIGMA, 200, seed=1000 + dataset), 1)
            bands = fit.impulse_response_bands(
                3, replications=99, level=0.9, seed=dataset
            )
            for h, i, j in hits:
                hits[(h, i, j)] += (
                    bands.lower[h][i][j] <= truth[h][i][j] <= bands.upper[h][i][j]
                )
        for key, count in hits.items():
            with self.subTest(response=key):
                self.assertGreaterEqual(count, 22)


if __name__ == "__main__":
    unittest.main()


class BiasCorrectedBandTests(unittest.TestCase):
    """Kilian's bootstrap-after-bootstrap on a persistent VAR(1)."""

    PERSISTENT = ((0.9, 0.0), (0.3, 0.7))

    def test_corrects_undercoverage_of_persistent_responses(self) -> None:
        """30 series of length 80: the corrected band covers far more often.

        At the nominal 90% the percentile band misses the long-horizon
        responses of this near-unit-root system in about half the series
        (least-squares bias toward zero); the bias-corrected band comes
        close to nominal [REF-KILIAN-1998].
        """
        factor = cholesky(SIGMA)
        phi: Matrix = ((1.0, 0.0), (0.0, 1.0))
        truth = [matmul(phi, factor)]
        for _ in range(8):
            phi = matmul(phi, self.PERSISTENT)
            truth.append(matmul(phi, factor))
        keys = ((4, 0, 0), (8, 0, 0))
        hits = {"percentile": 0, "bias-corrected": 0}
        for dataset in range(30):
            fit = fit_var(
                simulate_var((self.PERSISTENT,), SIGMA, 80, seed=500 + dataset), 1
            )
            for method in hits:
                bands = fit.impulse_response_bands(
                    8, replications=99, level=0.9, interval=method, seed=dataset
                )
                hits[method] += sum(
                    bands.lower[h][i][j] <= truth[h][i][j] <= bands.upper[h][i][j]
                    for h, i, j in keys
                )
        # Measured: 27 (percentile) and 48 (bias-corrected) of 60; nominal 54.
        self.assertGreaterEqual(hits["bias-corrected"], 44)
        self.assertGreaterEqual(hits["bias-corrected"] - hits["percentile"], 12)

    def test_correction_keeps_stability(self) -> None:
        """A bias that would push the system past a unit root is shrunk."""
        fit = fit_var(simulate_var((self.PERSISTENT,), SIGMA, 80, seed=3), 1)
        push = (((0.5, 0.0), (0.0, 0.5)),)
        corrected = fit._bias_corrected(fit.coefficients, push)
        self.assertTrue(replace(fit, coefficients=corrected).is_stable())
        self.assertNotEqual(corrected, fit.coefficients)
        self.assertEqual(
            fit._bias_corrected(fit.coefficients, ((((0.0,) * 2),) * 2,)),
            fit.coefficients,
        )

    def test_reproducible(self) -> None:
        """Same seed, same bands; the two stages use distinct streams."""
        fit = fit_var(simulate_var((self.PERSISTENT,), SIGMA, 80, seed=4), 1)
        first = fit.impulse_response_bands(
            3, replications=20, interval="bias-corrected", seed=9
        )
        second = fit.impulse_response_bands(
            3, replications=20, interval="bias-corrected", seed=9
        )
        self.assertEqual(first, second)
        self.assertEqual(first.interval, "bias-corrected")
