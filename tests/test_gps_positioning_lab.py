"""Regression tests for ``examples/gps_positioning_lab.py``.

Oracles: exact recovery of the true position (and clock offset) from
noise-free readings; the library Gauss-Newton covariance ``s^2 (J'J)^-1``
(``probstat.optimization.gauss_newton_covariance``) against the lab's
geometry-matrix covariances, known-clock and weighted unknown-clock; the
linearized covariance against the Monte Carlo covariance of the
least-squares fix; closed-form source variances; the closed-form pseudorange
likelihood against brute-force Monte Carlo over the clock shift and the
satellite offsets; the dilution-of-precision identities; and the fault-test
statistic against its chi-square law and its closed-form detection
probability. Every Monte Carlo claim is judged by its standard error, never
by a pinned floating-point value.

References:
    [REF-SEBER-WILD-2003, Section 2.1], [REF-OWEN-2013],
    [REF-NOCEDAL-WRIGHT-2006, Section 10.3], [REF-MISRA-ENGE-2006, Chapter 6],
    [REF-BROWN-1992].

"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
import unittest
from pathlib import Path

from probstat.optimization import gauss_newton_covariance
from probstat.statistics import quantile

LAB_PATH = Path(__file__).parents[1] / "examples" / "gps_positioning_lab.py"


def _load_lab():  # type: ignore[no-untyped-def]
    """Import the laboratory as a module without running it."""
    name = "_probstat_gps_positioning_lab"
    spec = importlib.util.spec_from_file_location(name, LAB_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lab = _load_lab()
ORIGIN = (0.0, 0.0, 0.0)


def _mean_se(values: list[float]) -> tuple[float, float]:
    """Return the sample mean and its standard error."""
    n = len(values)
    mean = math.fsum(values) / n
    spread = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(spread / n)


class NoiseFreeTests(unittest.TestCase):
    """Exact readings give back the true position."""

    def test_noise_free_fix_recovers_true_position(self) -> None:
        """From a start 5 km away the fix lands within a micrometre.

        Ranges of 2e7 m carry rounding of about 4e-9 m, so the solver may stop
        on "damping exceeded" at that floor; the cost and the position error
        are the evidence, not the flag.
        """
        for sky in (lab.GOOD_SKY, lab.POOR_SKY):
            centres = lab.constellation(sky)
            exact = lab.readings(
                centres, ORIGIN, lab.draw_sources(random.Random(1), 5, ())
            )
            fix = lab.solve_fix(centres, exact, start=(3000.0, -4000.0, 1500.0))
            self.assertLess(fix.cost, 1e-12)
            self.assertLess(lab.distance(fix.x, ORIGIN), 1e-6)

    def test_satellites_are_on_the_orbit_sphere(self) -> None:
        """Every centre lies at the orbit radius from the Earth centre."""
        earth = (0.0, 0.0, -lab.EARTH_RADIUS)
        for centre in lab.constellation(lab.GOOD_SKY):
            self.assertAlmostEqual(
                lab.distance(centre, earth) / lab.ORBIT_RADIUS, 1.0, places=12
            )
            self.assertTrue(2.0e7 < lab.distance(centre, ORIGIN) < 2.6e7)

    def test_unknown_clock_fix_recovers_position_and_clock(self) -> None:
        """A 75 km clock range is estimated; ignoring it costs kilometres."""
        centres = lab.constellation(lab.GOOD_SKY)
        exact = [lab.distance(c, ORIGIN) + lab.CLOCK_BIAS for c in centres]
        for sigmas in (None, lab.elevation_sigmas(lab.GOOD_SKY)):
            fix = lab.solve_clock_fix(centres, exact, sigmas).x
            self.assertLess(lab.distance(fix, ORIGIN), 1e-6)
            self.assertLess(abs(fix[3] / lab.CLOCK_BIAS - 1.0), 1e-10)
        self.assertGreater(lab.distance(lab.solve_fix(centres, exact).x, ORIGIN), 1e4)


class LinearizationTests(unittest.TestCase):
    """The geometry-matrix covariance against the library and Monte Carlo."""

    def test_gauss_newton_covariance_matches_geometry_matrix(self) -> None:
        """``s^2 (J'J)^-1`` equals ``A Sigma_e A'`` without the clock term."""
        centres = lab.constellation(lab.POOR_SKY)
        exact = lab.readings(centres, ORIGIN, lab.draw_sources(random.Random(2), 5, ()))
        result = lab.solve_fix(centres, exact)
        variances = lab.source_variances()
        library = gauss_newton_covariance(
            result, noise_variance=variances["satellite"] + variances["range"]
        )
        mine = lab.linearized_covariance(centres, active=("satellite", "range"))
        scale = max(abs(library[i][i]) for i in range(3))
        for i in range(3):
            for j in range(3):
                self.assertLess(abs(library[i][j] - mine[i][j]), 1e-9 * scale)
        self.assertLess(lab.gauss_newton_check(centres), 1e-9 * scale)

    def test_weighted_covariance_matches_library(self) -> None:
        """Whitened residuals make ``gauss_newton_covariance`` equal ``(G'WG)^-1``."""
        centres = lab.constellation(lab.GOOD_SKY)
        sigmas = lab.elevation_sigmas(lab.GOOD_SKY)
        lab_cov = lab.clock_fix_covariance(centres, sigmas, weighted=True)
        scale = max(lab_cov[i][i] for i in range(4))
        self.assertLess(lab.weighted_covariance_check(centres, sigmas), 1e-9 * scale)

    def test_weighted_fix_beats_ordinary_fix(self) -> None:
        """Gauss-Markov: ``(G'WG)^-1`` is below the ordinary sandwich covariance."""
        centres = lab.constellation(lab.GOOD_SKY)
        sigmas = lab.elevation_sigmas(lab.GOOD_SKY)
        ols = lab.clock_fix_covariance(centres, sigmas, weighted=False)
        wls = lab.clock_fix_covariance(centres, sigmas, weighted=True)
        for i in range(4):
            self.assertLessEqual(wls[i][i], ols[i][i] * (1.0 + 1e-12))

    def test_pdop_is_root_trace_of_covariance_per_unit_variance(self) -> None:
        """``PDOP^2`` is the trace of ``A A'`` (independent unit errors)."""
        centres = lab.constellation(lab.GOOD_SKY)
        a = lab.error_map(lab.geometry_matrix(centres, ORIGIN))
        trace = sum(sum(v * v for v in row) for row in a)
        self.assertLess(abs(lab.pdop_known_clock(centres) ** 2 / trace - 1.0), 1e-12)

    def test_dilution_of_precision_identities(self) -> None:
        """The DOP family is consistent and the clock costs precision.

        ``GDOP^2 = PDOP^2 + TDOP^2``, ``PDOP^2 = HDOP^2 + VDOP^2``, and
        estimating the clock never improves the position dilution.
        """
        for sky in (lab.GOOD_SKY, lab.POOR_SKY):
            centres = lab.constellation(sky)
            dop = lab.dilution_of_precision(centres)
            self.assertLess(abs((dop.pdop**2 + dop.tdop**2) / dop.gdop**2 - 1.0), 1e-12)
            self.assertLess(abs((dop.hdop**2 + dop.vdop**2) / dop.pdop**2 - 1.0), 1e-12)
            self.assertGreater(dop.pdop, lab.pdop_known_clock(centres))

    def test_small_noise_monte_carlo_covariance_matches_linearization(self) -> None:
        """Satellite-only errors (2 m) are linear; every entry within 4 SE."""
        centres = lab.constellation(lab.GOOD_SKY)
        study = lab.study_fix(centres, 1000, 11, ("satellite",))
        covariance = lab.sample_covariance(study.errors)
        errors = lab._covariance_se(study.errors)
        linear = lab.linearized_covariance(centres, active=("satellite",))
        for i in range(3):
            for j in range(3):
                self.assertLess(
                    abs(covariance[i][j] - linear[i][j]), 4.0 * errors[i][j], (i, j)
                )

    def test_clock_error_is_exactly_linear(self) -> None:
        """A common clock range ``b`` moves the known-clock fix by ``b A 1``."""
        centres = lab.constellation(lab.GOOD_SKY)
        a = lab.error_map(lab.geometry_matrix(centres, ORIGIN))
        bias = 0.25
        observed = [lab.distance(c, ORIGIN) + bias for c in centres]
        fix = lab.solve_fix(centres, observed).x
        for i in range(3):
            self.assertAlmostEqual(fix[i], bias * sum(a[i]), delta=1e-6)

    def test_all_source_mean_square_matches_linearization(self) -> None:
        """At 10 m range errors the fix is still linear within 4 SE."""
        centres = lab.constellation(lab.GOOD_SKY)
        study = lab.study_fix(centres, 1000, 12)
        linear = sum(lab.linearized_covariance(centres)[i][i] for i in range(3))
        self.assertLess(abs(study.mean_square - linear), 4.0 * study.mean_square_se)


class SourceTests(unittest.TestCase):
    """Closed-form source laws against simulation."""

    def test_line_of_sight_variance_of_ball_point(self) -> None:
        """A point uniform in a ball of radius ``r`` has axis variance ``r^2/5``."""
        rng = random.Random(3)
        n = 20000
        values = [lab.ball_point(rng, 2.0)[2] ** 2 for _ in range(n)]
        mean, se = _mean_se(values)
        self.assertLess(abs(mean - lab.source_variances()["satellite"]), 4.0 * se)
        self.assertTrue(all(v <= 4.0 for v in values))

    def test_offset_cdf_matches_ball_projection(self) -> None:
        """``G(x)`` is the CDF of one coordinate of a ball point (4 SE)."""
        rng = random.Random(4)
        n = 20000
        coords = [lab.ball_point(rng, lab.SATELLITE_RADIUS)[0] for _ in range(n)]
        for x in (-1.5, -0.4, 0.0, 0.7, 1.8):
            share, se = _mean_se([1.0 if c <= x else 0.0 for c in coords])
            self.assertLess(abs(share - lab.offset_cdf(x)), 4.0 * se + 1e-12, x)

    def test_residual_density_is_a_density(self) -> None:
        """``f_z`` is flat at ``1/(2h)`` inside and integrates to one."""
        h, a = lab.RANGE_HALF_WIDTH, lab.SATELLITE_RADIUS
        flat = lab.residual_density(0.5 * (h - a))
        self.assertLess(abs(flat * 2.0 * h - 1.0), 1e-14)
        self.assertEqual(lab.residual_density(h + a), 0.0)
        steps = 24000
        width = 2.0 * (h + a) / steps
        total = math.fsum(
            lab.residual_density(-h - a + (i + 0.5) * width) * width
            for i in range(steps)
        )
        self.assertLess(abs(total - 1.0), 1e-6)

    def test_variances_closed_form(self) -> None:
        """Uniform half-width ``h`` has variance ``h^2/3``."""
        variances = lab.source_variances()
        self.assertLess(abs(variances["range"] / (100.0 / 3.0) - 1.0), 1e-14)
        self.assertLess(
            abs(variances["clock"] / ((299_792_458.0 * 1e-9) ** 2 / 3.0) - 1.0),
            1e-14,
        )


class PosteriorTests(unittest.TestCase):
    """Importance-sampling estimate of ``f_R`` from one set of readings."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run one small importance-sampling study."""
        cls.centres = lab.constellation(lab.GOOD_SKY)
        cls.study = lab.study_posterior(cls.centres, 5, proposals=20000)

    def test_likelihood_matches_brute_force_monte_carlo(self) -> None:
        """The closed-form inner expectation equals a direct average (4 SE)."""
        observed = self.study.observed
        fix = self.study.fix
        density = (1.0 / (2.0 * lab.RANGE_HALF_WIDTH)) ** len(observed)
        for offset in ((0.0, 0.0, 0.0), (5.0, -3.0, 2.0)):
            r = tuple(f + o for f, o in zip(fix, offset, strict=True))
            exact = lab.marginal_likelihood(r, self.centres, observed)
            rng = random.Random(9)
            values = []
            for _ in range(6000):
                source = lab.draw_sources(rng, 5)
                inside = all(
                    abs(
                        d
                        - source.clock_range
                        - lab.distance(
                            r, tuple(c + o for c, o in zip(centre, shift, strict=True))
                        )
                    )
                    <= lab.RANGE_HALF_WIDTH
                    for centre, shift, d in zip(
                        self.centres, source.offsets, observed, strict=True
                    )
                )
                values.append(density if inside else 0.0)
            mean, se = _mean_se(values)
            self.assertGreater(exact, 0.0)
            self.assertLess(abs(mean - exact), 4.0 * se, offset)

    def test_clock_quadrature_is_exact(self) -> None:
        """Panels between the kinks make 8 and 16 nodes agree to rounding.

        Rounding grows where the likelihood is tiny (the difference of two
        nearly equal CDF values), hence a relative tolerance far above
        machine precision but far below any quadrature error.
        """
        self.assertLess(self.study.quadrature_gap, 1e-6)

    def test_proposal_boxes_cover_the_support(self) -> None:
        """No weighted proposal touches the edge of the proposal box."""
        self.assertEqual(self.study.edge_hits, 0)
        self.assertGreater(self.study.ess, 1000.0)

    def test_posterior_mean_is_near_the_fix(self) -> None:
        """``E[R]`` and the fix from the same readings agree to a few metres.

        Both are functions of the same readings; with a flat prior their gap
        is bounded by the posterior spread, far below the 90% radius.
        """
        gap = lab.distance(self.study.mean, self.study.fix)
        self.assertLess(gap, 0.5 * self.study.radius)

    def test_radius_matches_least_squares_scale(self) -> None:
        """The posterior 90% radius is within 30% of the linearized one."""
        linear = lab._linear_radius(lab.linearized_covariance(self.centres), 4)
        self.assertLess(abs(self.study.radius / linear - 1.0), 0.3)

    def test_weighted_quantile_reduces_to_type_one(self) -> None:
        """Equal weights give the type-1 sample quantile."""
        values = [random.Random(6).random()] + [0.1 * k for k in range(1, 50)]
        weights = [1.0] * len(values)
        self.assertEqual(
            lab._weighted_quantile(values, weights, 0.9),
            quantile(values, 0.9, method=1),
        )


class FaultDetectionTests(unittest.TestCase):
    """Residual-based fault detection with one redundant pseudorange."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the weighted geometry of the well-spread sky."""
        cls.centres = lab.constellation(lab.GOOD_SKY)
        cls.sigmas = lab.elevation_sigmas(lab.GOOD_SKY)
        cls.projector = lab.residual_projector(cls.centres, cls.sigmas)
        cls.threshold = lab.raim_threshold()

    def test_projector_has_one_redundancy(self) -> None:
        """``S`` is idempotent with trace ``n - 4 = 1``."""
        s = self.projector
        trace = sum(s[i][i] for i in range(5))
        self.assertLess(abs(trace - 1.0), 1e-10)
        for i in range(5):
            for j in range(5):
                square = sum(s[i][k] * s[k][j] for k in range(5))
                self.assertLess(abs(square - s[i][j]), 1e-10)

    def test_detection_probability_limits(self) -> None:
        """At zero bias the detection probability is the false-alarm rate."""
        self.assertLess(
            abs(lab.detection_probability(0.0, self.threshold) / lab.FALSE_ALARM - 1.0),
            1e-9,
        )
        for k in range(5):
            bias = lab.minimal_detectable_bias(
                self.projector, self.sigmas, k, self.threshold
            )
            lam = lab.fault_noncentrality(self.projector, self.sigmas, k, bias)
            self.assertLess(
                abs(
                    lab.detection_probability(lam, self.threshold) - lab.DETECTION_LEVEL
                ),
                1e-9,
            )

    def test_statistic_follows_chi_square_one(self) -> None:
        """Fault-free ``T`` has mean 1 and the nominal false-alarm rate (4 SE)."""
        study = lab.study_clock_fix(self.centres, self.sigmas, 600, 21, self.threshold)
        mean, se = _mean_se(list(study.statistics))
        self.assertLess(abs(mean - 1.0), 4.0 * se)
        alarm_se = math.sqrt(lab.FALSE_ALARM * (1.0 - lab.FALSE_ALARM) / 600)
        self.assertLess(abs(study.false_alarm - lab.FALSE_ALARM), 4.0 * alarm_se)

    def test_simulated_detection_matches_closed_form(self) -> None:
        """At the 90% minimal detectable bias the simulated rate agrees (4 SE)."""
        k = 1
        bias = lab.minimal_detectable_bias(
            self.projector, self.sigmas, k, self.threshold
        )
        rate, _ = lab.detection_rate(
            self.centres, self.sigmas, (k, bias), 400, 22, self.threshold
        )
        level = lab.DETECTION_LEVEL
        self.assertLess(abs(rate - level), 4.0 * math.sqrt(level * (1 - level) / 400))

    def test_fault_shift_is_linear(self) -> None:
        """A fault shifts the weighted fix by ``b (G'WG)^-1 (g_k, 1)' / sigma_k^2``."""
        k, bias = 3, 30.0
        exact = [lab.distance(c, ORIGIN) + lab.CLOCK_BIAS for c in self.centres]
        exact[k] += bias
        fix = lab.solve_clock_fix(self.centres, exact, self.sigmas).x
        shift = lab.fault_position_shift(self.centres, self.sigmas, k, bias)
        truth = (0.0, 0.0, 0.0, lab.CLOCK_BIAS)
        for i in range(4):
            self.assertLess(abs(fix[i] - truth[i] - shift[i]), 1e-3)


if __name__ == "__main__":
    unittest.main()
