"""Regression tests for the input-modeling laboratory.

Oracles:

- R 4.6.2 (2026-10-29 ucrt), run offline on the 400 block maxima and block
  means of ``examples/input_modeling_lab.py`` (``SEED = 2026``) written with
  full precision (``repr``): ``e1071::skewness`` and ``e1071::kurtosis``
  (``packageVersion("e1071")`` 1.7.17) with ``type = 1`` and ``type = 3``;
  ``fitdistrplus::fitdist(x, d, method = "mme")`` and ``method = "mle"``
  with ``control = list(reltol = 1e-14, maxit = 10000)``
  (``packageVersion("fitdistrplus")`` 1.2.6); ``evd::fgev(x, std.err =
  TRUE, control = list(reltol = 1e-14, maxit = 10000))``
  (``packageVersion("evd")`` 2.3.7.1); ``stats::bw.nrd0(x)`` and
  ``stats::bw.ucv(x, upper = 3 * 1.144 * sd(x) * length(x)^(-1/5))``;
  ``stats::quantile(x, 0.99, type = 7)``; ``8 * sqrt(-log(1 - 0.99^(1/144)))``.
- Closed forms: the exact law ``F(x)^144`` of a maximum, the truncated
  normal moments of a maximum-entropy law with a mean and a variance,
  Silverman's two-dimensional factor ``(4/(N (d + 2)))^(1/(d + 4))``, and
  Python's ``statistics.correlation``.
- An independent composite Simpson rule for the integrated squared error.
- Monte Carlo statements (bootstrap p-values, parameter bands) are judged
  with margins far above their Monte Carlo error; no bootstrap digit is
  pinned.
"""

from __future__ import annotations

import importlib.util
import math
import statistics
import sys
import unittest
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

from probstat.distributions import TruncatedNormal
from probstat.inference import (
    NORMAL_FAMILY,
    bootstrap_goodness_of_fit,
    parameter_uncertainty_band,
)
from probstat.statistics import gaussian_kde, kde_loo_log_likelihood

ROOT = Path(__file__).resolve().parents[1]
LAB_PATH = ROOT / "examples" / "input_modeling_lab.py"


def _load_lab() -> ModuleType:
    """Import the laboratory script as a module.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    spec = importlib.util.spec_from_file_location("input_modeling_lab", LAB_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LAB = _load_lab()


def _weibull_maximum_pdf(x: float) -> float:
    """Return ``144 F^143 f`` for Weibull(2, 8), written out independently.

    References:
        [EQ-LAB-INPUT-001], [EQ-DIST-WEIB-001].

    """
    if x <= 0.0:
        return 0.0
    z = (x / 8.0) ** 2
    cdf = -math.expm1(-z)
    pdf = (2.0 / 8.0) * (x / 8.0) * math.exp(-z)
    return 144.0 * cdf**143 * pdf


class InputModelingLabTests(unittest.TestCase):
    """The laboratory's printed numbers agree with independent oracles."""

    @classmethod
    def setUpClass(cls) -> None:
        """Simulate the data and fit the candidates once (no bootstrap)."""
        cls.data = LAB.simulate_wind_data()
        cls.candidates = LAB.fit_candidates(cls.data.maxima, n_bootstrap=0)
        cls.by_family = {item.family: item for item in cls.candidates}

    def assertRelative(self, value: float, expected: float, tolerance: float) -> None:
        """Assert ``|value - expected| <= tolerance |expected|``."""
        self.assertLessEqual(
            abs(value - expected), tolerance * abs(expected), (value, expected)
        )

    def test_blocks_are_maxima_and_means_of_the_speeds(self) -> None:
        """Each block statistic is recomputed from its 144 speeds."""
        speeds = self.data.speeds
        self.assertEqual(len(speeds), 400 * 144)
        self.assertEqual(len(self.data.maxima), 400)
        for index in (0, 137, 399):
            block = speeds[index * 144 : (index + 1) * 144]
            self.assertEqual(self.data.maxima[index], max(block))
            self.assertAlmostEqual(self.data.means[index], math.fsum(block) / 144)

    def test_shape_summaries_match_e1071(self) -> None:
        """Type 1 and type 3 skewness and kurtosis equal ``e1071`` values."""
        expected = {
            "maxima": (
                0.558461779476562,
                0.556368857244205,
                0.41692667754812,
                0.399863399952115,
                2.08704232561529,
                18.5470166253726,
            ),
            "means": (
                -0.0205984711364707,
                -0.0205212751675102,
                0.0565469921478634,
                0.0412833606058252,
                0.333492998174223,
                7.07038203110714,
            ),
        }
        for name, values in (("maxima", self.data.maxima), ("means", self.data.means)):
            summary = LAB.describe(values)
            observed = (
                summary.skewness,
                summary.skewness_type3,
                summary.excess_kurtosis,
                summary.excess_kurtosis_type3,
                summary.standard_deviation,
                summary.median,
            )
            with self.subTest(sample=name):
                for value, target in zip(observed, expected[name], strict=True):
                    self.assertRelative(value, target, 1e-9)
        self.assertRelative(
            LAB.describe(self.data.maxima).first_quartile, 17.225648974098, 1e-12
        )

    def test_exact_law_of_the_maximum(self) -> None:
        """The exact 0.99 quantile, CDF, and density are consistent."""
        exact = LAB.exact_maximum_quantile(0.99)
        self.assertRelative(exact, 24.7483299453031, 1e-12)
        self.assertAlmostEqual(LAB.exact_maximum_cdf(exact), 0.99, places=13)
        for x in (15.0, 18.7, 24.0):
            self.assertRelative(
                LAB.exact_maximum_pdf(x), _weibull_maximum_pdf(x), 1e-12
            )
            step = 1e-5
            slope = (
                LAB.exact_maximum_cdf(x + step) - LAB.exact_maximum_cdf(x - step)
            ) / (2 * step)
            self.assertRelative(LAB.exact_maximum_pdf(x), slope, 1e-6)
        low, high, count = 8.0, 48.0, 4000
        width = (high - low) / count
        nodes = [low + k * width for k in range(count + 1)]
        weights = [
            1 if k in (0, count) else (4 if k % 2 else 2) for k in range(count + 1)
        ]

        def simpson(function: Callable[[float], float]) -> float:
            """Composite Simpson rule on the effective support."""
            terms = [w * function(x) for w, x in zip(weights, nodes, strict=True)]
            return math.fsum(terms) * width / 3.0

        centre = simpson(lambda x: x * _weibull_maximum_pdf(x))
        second = simpson(lambda x: (x - centre) ** 2 * _weibull_maximum_pdf(x))
        third = simpson(lambda x: (x - centre) ** 3 * _weibull_maximum_pdf(x))
        mean, sd, skew = LAB.exact_maximum_moments()
        self.assertAlmostEqual(simpson(_weibull_maximum_pdf), 1.0, places=10)
        self.assertAlmostEqual(mean, centre, places=8)
        self.assertAlmostEqual(sd, math.sqrt(second), places=8)
        self.assertAlmostEqual(skew, third / second**1.5, places=7)

    def test_likelihood_fits_match_fitdistrplus_and_evd(self) -> None:
        """Estimates and AIC equal the R fits; standard errors to 0.1%."""
        reference = {
            "Normal": ((18.7277210765582, 2.08443189016514), 1726.74790126921),
            "LogNormal": ((2.92394956204652, 0.109585631445014), 1709.47126578749),
            "Gamma": ((82.7389667928986, 4.4179944305641), 1713.6456836092),
            "Weibull": ((8.72927951050671, 19.6854522046351), 1800.39858982613),
            # evd::fgev: deviance 1701.45288028237 plus 2 k = 6.
            "GEV": (
                (17.8566099018469, 1.85865937754578, -0.121387611277718),
                1707.45288028237,
            ),
        }
        for family, (estimates, aic) in reference.items():
            fit = self.by_family[family].fit
            with self.subTest(family=family):
                for value, target in zip(fit.estimates, estimates, strict=True):
                    self.assertLessEqual(
                        abs(value - target), 5e-5 * max(1.0, abs(target))
                    )
                self.assertAlmostEqual(fit.aic, aic, delta=1e-6)
        errors = {
            "Normal": (0.104221594511401, 0.0736957198835559),
            "GEV": (0.103032664836043, 0.0722501153530489, 0.0318471043255057),
        }
        for family, targets in errors.items():
            fit = self.by_family[family].fit
            for value, target in zip(fit.standard_errors, targets, strict=True):
                self.assertRelative(value, target, 1e-3)
        self.assertEqual(
            [item.family for item in self.candidates],
            ["GEV", "LogNormal", "Gamma", "Normal", "Weibull"],
        )
        self.assertAlmostEqual(math.fsum(i.akaike_weight for i in self.candidates), 1.0)
        self.assertIs(LAB.choose_law(self.candidates), self.by_family["GEV"])

    def test_moment_fits_match_fitdistrplus_mme(self) -> None:
        """Closed-form moment estimates equal ``fitdist(method = "mme")``."""
        reference = {
            "Normal": (18.7277210765582, 2.08443189016514),
            "LogNormal": (2.92384882702033, 0.110959539162416),
            "Gamma": (80.7224709224441, 4.3103200113059),
        }
        for family, targets in reference.items():
            moments = self.by_family[family].moments
            for value, target in zip(moments.estimates, targets, strict=True):
                self.assertRelative(value, target, 1e-9)

    def test_naive_p_values_are_conservative_after_fitting(self) -> None:
        """The fitted normal passes the naive tests and fails the bootstrap.

        With ``B = 99`` refits the observed AD statistic of the normal fit
        exceeds every replicate in the laboratory run (``p = 1/100``); a
        threshold of 0.05 leaves four replicates of margin.
        """
        normal = self.by_family["Normal"]
        self.assertGreater(normal.naive_ks_p, 0.1)
        self.assertGreater(normal.naive_ad_p, 0.1)
        calibrated = bootstrap_goodness_of_fit(
            self.data.maxima, NORMAL_FAMILY, n_bootstrap=99, seed=LAB.SEED,
            fit=normal.fit,
        )  # fmt: skip
        self.assertEqual(calibrated.failed_refits, 0)
        self.assertLess(calibrated.ad_p_value, 0.05)
        self.assertLess(calibrated.ad_p_value, normal.naive_ad_p)
        self.assertLess(calibrated.ks_p_value, normal.naive_ks_p)
        self.assertLess(self.by_family["Weibull"].naive_ad_p, 0.01)

    def test_design_value_bands_against_the_exact_quantile(self) -> None:
        """Only the GEV band of the 0.99 quantile contains the exact value.

        The 95% band limits from 2000 draws carry a Monte Carlo error of a
        few hundredths of a m/s; the nearest excluded limit (lognormal, 24.52)
        lies 0.23 m/s below the exact value 24.75.
        """
        exact = LAB.exact_maximum_quantile(0.99)
        covered = {}
        for family, candidate in self.by_family.items():
            band = LAB.fitted_band(candidate.fit, (24.0,))
            direct = parameter_uncertainty_band(
                candidate.fit, (24.0,), draws=LAB.BAND_DRAWS, level=0.95,
                seed=LAB.SEED, quantile_probabilities=(0.99,),
            )  # fmt: skip
            self.assertEqual(band.quantile_upper, direct.quantile_upper)
            self.assertAlmostEqual(
                band.quantile_estimate[0],
                candidate.fit.distribution.quantile(0.99),
                places=10,
            )
            covered[family] = band.quantile_lower[0] <= exact <= band.quantile_upper[0]
        self.assertEqual(
            covered,
            {
                "GEV": True,
                "LogNormal": False,
                "Gamma": False,
                "Normal": False,
                "Weibull": False,
            },
        )
        # evd::qgev(0.99, ...) at the fgev estimates.
        self.assertRelative(
            self.by_family["GEV"].fit.distribution.quantile(0.99),
            24.4081634832954,
            1e-5,
        )
        self.assertRelative(
            LAB.quantile(self.data.maxima, 0.99), 24.1324779040903, 1e-12
        )

    def test_envelope_widens_while_the_quantile_band_settles(self) -> None:
        """Nested draws: the envelope never narrows; the band nears delta."""
        widths = LAB.envelope_widths(self.by_family["GEV"].fit)
        envelopes = [row[2] for row in widths]
        self.assertEqual(envelopes, sorted(envelopes))
        self.assertGreater(envelopes[-1], 1.4 * envelopes[0])
        draws, band, envelope, delta = widths[-1]
        self.assertEqual(draws, 10000)
        self.assertLess(abs(band - delta), 0.1 * delta)
        self.assertGreater(envelope, 1.5 * band)

    def test_kernel_bandwidths_and_integrated_squared_error(self) -> None:
        """Bandwidths match R; the ISE matches an independent Simpson rule."""
        kernels = {
            kernel.method: kernel for kernel in LAB.kernel_estimates(self.data.maxima)
        }
        self.assertRelative(kernels["Silverman"].bandwidth, 0.556912140973452, 1e-12)
        # R bins the pairwise distances, so bw.ucv agrees to about 1%.
        self.assertRelative(kernels["UCV"].bandwidth, 0.764524686525341, 0.01)
        self.assertIn("default search ended at its bound", kernels["UCV"].note)
        h = kernels["LOO likelihood"].bandwidth
        centre = kde_loo_log_likelihood(self.data.maxima, h)
        for factor in (0.9, 1.1):
            self.assertLess(
                kde_loo_log_likelihood(self.data.maxima, factor * h), centre
            )
        estimate = gaussian_kde(
            self.data.maxima, bandwidth=kernels["Silverman"].bandwidth
        )
        low, high, count = 8.0, 48.0, 4000
        width = (high - low) / count
        total = 0.0
        for k in range(count + 1):
            x = low + k * width
            weight = 1 if k in (0, count) else (4 if k % 2 else 2)
            total += weight * (estimate.pdf(x) - _weibull_maximum_pdf(x)) ** 2
        simpson = total * width / 3.0
        self.assertRelative(kernels["Silverman"].ise, simpson, 1e-6)
        best = min(kernels.values(), key=lambda kernel: kernel.ise)
        self.assertEqual(best.method, "Silverman")

    def test_expert_laws(self) -> None:
        """The two-moment law is a truncated normal; the skewed one turns up."""
        laws = LAB.expert_laws()
        plain, skewed = (laws[label] for label in LAB.EXPERT_LABELS)
        for law in (plain, skewed):
            self.assertLess(max(abs(value) for value in law.residuals), 1e-10)
        first, second = plain.multipliers
        oracle = TruncatedNormal(
            -first / (2.0 * second), math.sqrt(1.0 / (2.0 * second)), 12.0, 32.0
        )
        self.assertAlmostEqual(oracle.mean, 19.0, places=9)
        self.assertAlmostEqual(oracle.variance, 4.0, places=8)
        self.assertAlmostEqual(plain.quantile(0.99), oracle.quantile(0.99), places=8)
        self.assertGreater(skewed.pdf(32.0), skewed.pdf(30.0))
        self.assertAlmostEqual(skewed.skewness, LAB.GUMBEL_SKEWNESS, places=8)
        quantiles = [LAB.expert_laws(upper)[LAB.EXPERT_LABELS[1]].quantile(0.99)
                     for upper in (30.0, 35.0, 40.0)]  # fmt: skip
        self.assertEqual(quantiles, sorted(quantiles, reverse=True))

    def test_joint_sample(self) -> None:
        """Correlation and the Silverman factor follow their closed forms."""
        correlation, silverman, likelihood = LAB.joint_factors(
            self.data.means, self.data.maxima
        )
        self.assertAlmostEqual(
            correlation,
            statistics.correlation(self.data.means, self.data.maxima),
            places=12,
        )
        self.assertAlmostEqual(silverman, (4.0 / (400 * 4)) ** (1.0 / 6.0), places=12)
        self.assertGreater(likelihood, 0.0)

    def test_means_are_near_normal(self) -> None:
        """The normal law ranks first for the block means."""
        means = LAB.fit_candidates(self.data.means, n_bootstrap=0)
        self.assertEqual(means[0].family, "Normal")
        self.assertLess(abs(LAB.describe(self.data.means).skewness), 0.1)


if __name__ == "__main__":
    unittest.main()
