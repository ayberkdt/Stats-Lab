"""Tests for square-root, localized, relaxed, and smoothing ensemble methods."""

from __future__ import annotations

import math
import unittest

from probstat.data_assimilation import (
    Ensemble,
    ensemble_analysis,
    ensemble_kalman_filter,
)
from probstat.filtering import LinearGaussianStateSpace, simulate


def _constant_velocity(
    dt: float = 1.0, q: float = 0.05, r: float = 1.0
) -> LinearGaussianStateSpace:
    """Return the white-noise-acceleration constant-velocity tracking model."""
    transition = ((1.0, dt), (0.0, 1.0))
    process = (
        (q * dt**3 / 3.0, q * dt**2 / 2.0),
        (q * dt**2 / 2.0, q * dt),
    )
    return LinearGaussianStateSpace(
        transition,
        process,
        ((1.0, 0.0),),
        ((r,),),
        (0.0, 1.0),
        ((1.0, 0.0), (0.0, 0.5)),
    )


def _close(test: unittest.TestCase, actual, expected, rel: float = 1e-12) -> None:
    """Compare nested float sequences with a relative tolerance."""
    if isinstance(expected, (tuple, list)):
        test.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected, strict=True):
            _close(test, a, b, rel)
    else:
        test.assertTrue(
            math.isclose(actual, expected, rel_tol=rel, abs_tol=1e-15),
            f"{actual!r} != {expected!r}",
        )


class StochasticRegressionTests(unittest.TestCase):
    """The default stochastic EnKF keeps its seeded results (pinned 0.25.0)."""

    def test_filter_numbers_are_unchanged(self) -> None:
        """The guide's ``N = 50`` run and a gappy inflated run are pinned."""
        model = _constant_velocity()
        data = simulate(model, 5, seed=7).observations
        result = ensemble_kalman_filter(model, data, ensemble_size=50, seed=0)
        _close(self, result.analysis_means[-1], (4.608761863734528, 0.6753455863687211))
        _close(
            self,
            result.analysis_covariances[-1],
            (
                (0.6701501187237877, 0.20303441173497042),
                (0.20303441173497042, 0.12141981835645102),
            ),
        )
        gappy = [data[0], None, *data[2:]]
        result = ensemble_kalman_filter(
            model, gappy, ensemble_size=12, inflation=1.1, seed=3
        )
        _close(self, result.analysis_means[-1], (4.28362571914277, 0.6631789642506009))
        _close(
            self,
            result.analysis_covariances[-1],
            (
                (1.0670537019522461, 0.5015510743968069),
                (0.5015510743968069, 0.2841389980966667),
            ),
        )
        _close(self, result.forecast_means[1], (2.9555943794602704, 1.224434787148646))

    def test_nonlinear_analysis_is_unchanged(self) -> None:
        """One perturbed-observation analysis with a nonlinear ``h`` is pinned."""
        ensemble = Ensemble(((0.0, 1.0), (1.0, 3.0), (2.0, 2.0), (5.0, 0.0)))
        analysis = ensemble_analysis(
            ensemble,
            (1.5,),
            lambda x: (x[0] + 0.1 * x[1] ** 2,),
            ((0.5,),),
            seed=4,
        )
        _close(
            self,
            analysis.ensemble.members,
            (
                (1.3366696701547014, 0.6373376863921352),
                (0.9333105462556076, 3.018094037837626),
                (0.8532152714511132, 2.311143143404737),
                (1.9591397801259376, 0.8250395945394744),
            ),
        )
        _close(self, analysis.gain, ((0.9354604786076868,), (-0.2538071065989848,)))


if __name__ == "__main__":
    unittest.main()
