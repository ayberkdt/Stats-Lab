"""Regression tests for ``examples/gps_positioning_lab.py``.

Oracles: exact recovery of the true position from noise-free readings; the
library Gauss-Newton covariance ``s^2 (J'J)^-1``
(``probstat.optimization.gauss_newton_covariance``) against the lab's
``A Sigma_e A'`` built from the geometry matrix; the linearized covariance
against the Monte Carlo covariance of the least-squares fix where the
problem is linear to far below the sampling error; closed-form source
variances; and the importance-sampling posterior against the fix from the
same readings. Every Monte Carlo claim is judged by its standard error.

References:
    [REF-SEBER-WILD-2003, Section 2.1], [REF-OWEN-2013],
    [REF-NOCEDAL-WRIGHT-2006, Section 10.3].

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
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(library[i][j], mine[i][j], delta=1e-9)
        self.assertLess(lab.gauss_newton_check(centres), 1e-9)

    def test_pdop_is_root_trace_of_covariance_per_unit_variance(self) -> None:
        """``PDOP^2`` is the trace of ``A A'`` (independent unit errors)."""
        centres = lab.constellation(lab.GOOD_SKY)
        a = lab.error_map(lab.geometry_matrix(centres, ORIGIN))
        trace = sum(sum(v * v for v in row) for row in a)
        self.assertAlmostEqual(lab.pdop(centres) ** 2, trace, places=12)

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
        """A common clock range ``b`` moves the fix by ``b A 1`` exactly."""
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
    """Closed-form source variances against simulation."""

    def test_line_of_sight_variance_of_ball_point(self) -> None:
        """A point uniform in a ball of radius ``r`` has axis variance ``r^2/5``."""
        rng = random.Random(3)
        n = 20000
        values = [lab.ball_point(rng, 2.0)[2] ** 2 for _ in range(n)]
        mean = math.fsum(values) / n
        se = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1) / n)
        self.assertLess(abs(mean - lab.source_variances()["satellite"]), 4.0 * se)
        self.assertTrue(all(v <= 4.0 for v in values))

    def test_variances_closed_form(self) -> None:
        """Uniform half-width ``h`` has variance ``h^2/3``."""
        variances = lab.source_variances()
        self.assertAlmostEqual(variances["range"], 100.0 / 3.0)
        self.assertAlmostEqual(
            variances["clock"], (299_792_458.0 * 1e-9) ** 2 / 3.0, places=15
        )


class PosteriorTests(unittest.TestCase):
    """Importance-sampling estimate of ``f_R`` from one set of readings."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run one small importance-sampling study."""
        cls.centres = lab.constellation(lab.GOOD_SKY)
        cls.study = lab.study_posterior(cls.centres, 5, proposals=6000, inner_draws=16)

    def test_proposal_box_covers_the_support(self) -> None:
        """No weighted proposal touches the edge of the main-stage box."""
        self.assertEqual(self.study.edge_hits, 0)
        self.assertGreater(self.study.ess, 200.0)

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
        values = [random.Random(6).random() for _ in range(1)] + [
            0.1 * k for k in range(1, 50)
        ]
        weights = [1.0] * len(values)
        self.assertEqual(
            lab._weighted_quantile(values, weights, 0.9),
            quantile(values, 0.9, method=1),
        )


if __name__ == "__main__":
    unittest.main()
