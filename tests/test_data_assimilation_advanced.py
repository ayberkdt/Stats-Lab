"""Tests for square-root, localized, relaxed, and smoothing ensemble methods."""

from __future__ import annotations

import math
import random
import unittest

from probstat.data_assimilation import (
    Ensemble,
    Localization,
    Relaxation,
    eakf_analysis,
    ensemble_analysis,
    ensemble_kalman_filter,
    ensemble_kalman_smoother,
    etkf_analysis,
    gaspari_cohn,
    letkf_analysis,
    localization_matrix,
    localized_ensemble_analysis,
    relax_to_prior_perturbations,
    relax_to_prior_spread,
)
from probstat.filtering import (
    GaussianState,
    LinearGaussianStateSpace,
    kalman_filter,
    linear_gaussian_update,
    lorenz63,
    lorenz63_rhs,
    lorenz96,
    lorenz96_rhs,
    rts_smoother,
    simulate,
)
from probstat.numerics import integrate_fixed_step, reverse_gradient


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


def _block(left, right):
    """Return the block-diagonal matrix ``diag(left, right)``."""
    n, m = len(left), len(right)
    upper = tuple(
        tuple(left[i][j] if j < n else 0.0 for j in range(n + m)) for i in range(n)
    )
    lower = tuple(
        tuple(right[i][j - n] if j >= n else 0.0 for j in range(n + m))
        for i in range(m)
    )
    return upper + lower


def _two_tracks() -> tuple[LinearGaussianStateSpace, Localization]:
    """Return two independent constant-velocity tracks far apart in space.

    Track A sits at coordinate 0 and track B at 100; with a Gaspari-Cohn
    half-width of 10 the localization weights are exactly one within a track
    and zero across, which is the true (zero) cross-track correlation, so the
    localized filters target the exact Kalman moments.
    """
    one = _constant_velocity()
    model = LinearGaussianStateSpace(
        _block(one.transition_matrix, one.transition_matrix),
        _block(one.process_covariance, one.process_covariance),
        ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
        ((1.0, 0.0), (0.0, 1.0)),
        (0.0, 1.0, 5.0, -1.0),
        _block(one.initial_covariance, one.initial_covariance),
    )
    localization = Localization((0.0, 0.0, 100.0, 100.0), (0.0, 100.0), 10.0)
    return model, localization


def _random_problem(seed: int, size: int, dimension: int, observed: int):
    """Return a random ensemble, linear operator, diagonal ``R``, and data."""
    stream = random.Random(seed)
    ensemble = Ensemble(
        tuple(
            tuple(stream.gauss(0.0, 1.0) for _ in range(dimension)) for _ in range(size)
        )
    )
    operator = tuple(
        tuple(stream.gauss(0.0, 1.0) for _ in range(dimension)) for _ in range(observed)
    )
    noise = tuple(
        tuple(0.3 + 0.1 * i if i == j else 0.0 for j in range(observed))
        for i in range(observed)
    )
    data = tuple(stream.gauss(0.0, 1.0) for _ in range(observed))
    return ensemble, operator, noise, data


def _apply(operator):
    """Return the linear map ``x -> A x``."""
    return lambda x: tuple(
        sum(a * b for a, b in zip(row, x, strict=True)) for row in operator
    )


def _kalman_from_ensemble(ensemble, operator, noise, data) -> GaussianState:
    """Return the exact Kalman update of the ensemble mean and covariance."""
    prior = GaussianState(ensemble.mean, ensemble.covariance)
    return linear_gaussian_update(
        prior, data, _apply(operator)(ensemble.mean), operator, noise
    ).state


def _transformed(ensemble: Ensemble, analysis) -> tuple[tuple[float, ...], ...]:
    """Return the members ``E^f T_j`` rebuilt from an analysis' transforms."""
    count = ensemble.size
    return tuple(
        tuple(
            sum(
                ensemble.members[a][j] * analysis.transform_for(j)[a][i]
                for a in range(count)
            )
            for j in range(len(ensemble.members[0]))
        )
        for i in range(count)
    )


class SquareRootAnalysisTests(unittest.TestCase):
    """ETKF and EAKF reproduce the Kalman update of the ensemble moments."""

    def test_etkf_equals_kalman_update_in_both_spaces(self) -> None:
        """Ensemble-space (p >= N) and observation-space (p < N) forms agree."""
        for size, observed in ((3, 4), (5, 5), (8, 2), (12, 1)):
            with self.subTest(size=size, observed=observed):
                ensemble, operator, noise, data = _random_problem(
                    size, size, 3, observed
                )
                analysis = etkf_analysis(
                    ensemble, data, _apply(operator), noise, transforms=True
                )
                exact = _kalman_from_ensemble(ensemble, operator, noise, data)
                _close(self, analysis.ensemble.mean, exact.mean, rel=1e-10)
                _close(self, analysis.ensemble.covariance, exact.covariance, rel=1e-10)
                _close(
                    self,
                    _transformed(ensemble, analysis),
                    analysis.ensemble.members,
                    1e-10,
                )
                self.assertEqual(analysis.method, "etkf")

    def test_etkf_with_correlated_noise(self) -> None:
        """A full ``R`` is whitened by its Cholesky factor."""
        ensemble, operator, _, data = _random_problem(3, 6, 3, 2)
        noise = ((0.5, 0.2), (0.2, 0.4))
        analysis = etkf_analysis(ensemble, data, _apply(operator), noise)
        exact = _kalman_from_ensemble(ensemble, operator, noise, data)
        _close(self, analysis.ensemble.mean, exact.mean, rel=1e-10)
        _close(self, analysis.ensemble.covariance, exact.covariance, rel=1e-10)

    def test_eakf_equals_kalman_update_for_diagonal_noise(self) -> None:
        """Serial scalar processing is exact for independent errors."""
        for size, observed in ((4, 3), (10, 5)):
            with self.subTest(size=size, observed=observed):
                ensemble, operator, noise, data = _random_problem(
                    10 + size, size, 3, observed
                )
                analysis = eakf_analysis(
                    ensemble, data, _apply(operator), noise, transforms=True
                )
                exact = _kalman_from_ensemble(ensemble, operator, noise, data)
                _close(self, analysis.ensemble.mean, exact.mean, rel=1e-10)
                _close(self, analysis.ensemble.covariance, exact.covariance, rel=1e-10)
                _close(
                    self,
                    _transformed(ensemble, analysis),
                    analysis.ensemble.members,
                    1e-10,
                )

    def test_eakf_requires_diagonal_noise_and_spread(self) -> None:
        """Correlated errors and a collapsed observed ensemble raise."""
        ensemble, operator, _, data = _random_problem(2, 5, 3, 2)
        with self.assertRaises(ValueError):
            eakf_analysis(ensemble, data, _apply(operator), ((0.5, 0.1), (0.1, 0.5)))
        flat = Ensemble(((1.0, 0.0), (1.0, 1.0), (1.0, 2.0)))
        with self.assertRaises(ValueError):
            eakf_analysis(flat, (0.5,), lambda x: (x[0],), ((1.0,),))

    def test_letkf_with_infinite_radius_equals_etkf_exactly(self) -> None:
        """Every local problem is the global one when nothing is tapered."""
        ensemble, operator, noise, data = _random_problem(5, 7, 4, 3)
        localization = Localization(range(4), (0.0, 1.5, 3.0), math.inf)
        local = letkf_analysis(
            ensemble, data, _apply(operator), noise, localization, transforms=True
        )
        global_ = etkf_analysis(
            ensemble, data, _apply(operator), noise, transforms=True
        )
        self.assertEqual(local.ensemble.members, global_.ensemble.members)
        self.assertEqual(len(local.transforms), 4)
        for transform in local.transforms:
            self.assertEqual(transform, global_.transforms[0])

    def test_letkf_keeps_components_without_nearby_observations(self) -> None:
        """A component beyond ``2c`` of every observation keeps its forecast."""
        ensemble, operator, noise, data = _random_problem(6, 6, 3, 1)
        localization = Localization((0.0, 1.0, 50.0), (0.0,), 2.0)
        analysis = letkf_analysis(ensemble, data, _apply(operator), noise, localization)
        for before, after in zip(
            ensemble.members, analysis.ensemble.members, strict=True
        ):
            self.assertEqual(before[2], after[2])
            self.assertNotEqual(before[0], after[0])


class LocalizationTests(unittest.TestCase):
    """Gaspari-Cohn weights and the localized analyses."""

    def test_gaspari_cohn_known_values(self) -> None:
        """One at zero, 5/24 at ``c``, zero from ``2c``, continuous at ``c``."""
        c = 3.0
        self.assertEqual(gaspari_cohn(0.0, c), 1.0)
        self.assertAlmostEqual(gaspari_cohn(c, c), 5.0 / 24.0, places=14)
        self.assertAlmostEqual(gaspari_cohn(2.0 * c, c), 0.0, places=14)
        self.assertEqual(gaspari_cohn(2.0 * c + 1e-9, c), 0.0)
        self.assertEqual(gaspari_cohn(-1.5, c), gaspari_cohn(1.5, c))
        below = gaspari_cohn(c * (1.0 - 1e-9), c)
        above = gaspari_cohn(c * (1.0 + 1e-9), c)
        self.assertAlmostEqual(below, above, places=7)
        # z = 1/2 and z = 3/2 from the two polynomial pieces.
        self.assertAlmostEqual(gaspari_cohn(0.5 * c, c), 263.0 / 384.0, places=13)
        self.assertAlmostEqual(gaspari_cohn(1.5 * c, c), 19.0 / 1152.0, places=13)
        values = [gaspari_cohn(0.01 * c * k, c) for k in range(201)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:], strict=False)))
        self.assertEqual(gaspari_cohn(1e6, math.inf), 1.0)
        with self.assertRaises(ValueError):
            gaspari_cohn(1.0, 0.0)

    def test_localization_matrix_is_periodic_and_symmetric(self) -> None:
        """Distances wrap on a circle, and the matrix is symmetric."""
        points = tuple(range(10))
        weights = localization_matrix(points, points, 2.0, period=10)
        self.assertEqual(weights[0][9], weights[0][1])
        self.assertEqual(weights[0][5], 0.0)
        for i in range(10):
            for j in range(10):
                self.assertEqual(weights[i][j], weights[j][i])
        planar = localization_matrix(((0.0, 0.0),), ((3.0, 4.0),), 5.0)
        self.assertAlmostEqual(planar[0][0], 5.0 / 24.0, places=14)
        with self.assertRaises(ValueError):
            localization_matrix((0.0,), ((0.0, 1.0),), 1.0)

    def test_schur_localization_without_taper_is_the_stochastic_analysis(self) -> None:
        """Unit weights reproduce :func:`ensemble_analysis` exactly."""
        ensemble, operator, noise, data = _random_problem(8, 6, 3, 2)
        localization = Localization((0.0, 1.0, 2.0), (0.0, 2.0), math.inf)
        localized = localized_ensemble_analysis(
            ensemble, data, _apply(operator), noise, localization, seed=5
        )
        plain = ensemble_analysis(ensemble, data, _apply(operator), noise, seed=5)
        self.assertEqual(localized.ensemble.members, plain.ensemble.members)
        self.assertEqual(localized.gain, plain.gain)

    def test_schur_localization_zeroes_distant_gain_entries(self) -> None:
        """The gain from an observation to a state beyond ``2c`` vanishes."""
        ensemble, operator, noise, data = _random_problem(9, 6, 3, 2)
        localization = Localization((0.0, 1.0, 20.0), (0.0, 20.0), 2.0)
        analysis = localized_ensemble_analysis(
            ensemble, data, _apply(operator), noise, localization, seed=1
        )
        self.assertEqual(analysis.gain[2][0], 0.0)
        self.assertEqual(analysis.gain[0][1], 0.0)
        self.assertNotEqual(analysis.gain[0][0], 0.0)

    def test_localized_filters_approach_kalman_moments(self) -> None:
        """Mean and variance errors stay within k standard errors as N grows.

        For each member count the bands are ``MEAN_BAND sqrt(P / N)`` for
        the mean and ``VARIANCE_BAND sqrt(2 / N) P`` for the variance, in
        units of the Monte Carlo standard errors of a mean and a variance of
        ``N`` independent Gaussian draws, around the exact Kalman moments.
        Ensemble errors accumulate over the cycles and exceed these units:
        offline runs over seeds 0-3 reached 2.8 (means) and 2.4 (variances).
        """
        model, localization = _two_tracks()
        data = simulate(model, 8, seed=7).observations
        exact = kalman_filter(model, data).filtered[-1]
        for size in (100, 1600):
            for options in (
                {"localization": localization},
                {"analysis": "letkf", "localization": localization},
            ):
                with self.subTest(size=size, **{"analysis": options.get("analysis")}):
                    result = ensemble_kalman_filter(
                        model, data, ensemble_size=size, seed=3, **options
                    )
                    for i in range(4):
                        variance = exact.covariance[i][i]
                        self.assertLess(
                            abs(result.analysis_means[-1][i] - exact.mean[i]),
                            MEAN_BAND * math.sqrt(variance / size),
                        )
                        self.assertLess(
                            abs(result.analysis_covariances[-1][i][i] - variance),
                            VARIANCE_BAND * math.sqrt(2.0 / size) * variance,
                        )


MEAN_BAND = 10.0
VARIANCE_BAND = 6.0
SMOOTHER_MEAN_BAND = 12.0


class RelaxationTests(unittest.TestCase):
    """RTPP and RTPS keep the mean and move the spread as stated."""

    def setUp(self) -> None:
        """Build a forecast and an ETKF analysis of it."""
        ensemble, operator, noise, data = _random_problem(21, 6, 3, 2)
        self.forecast = ensemble
        self.analysis = etkf_analysis(ensemble, data, _apply(operator), noise)

    def test_rtpp(self) -> None:
        """The mean is kept; alpha = 0 and 1 give analysis and forecast anomalies."""
        analysis = self.analysis.ensemble
        self.assertIs(
            relax_to_prior_perturbations(self.forecast, analysis, 0.0), analysis
        )
        relaxed = relax_to_prior_perturbations(self.forecast, analysis, 1.0)
        _close(self, relaxed.mean, analysis.mean, rel=1e-12)
        _close(self, relaxed.covariance, self.forecast.covariance, rel=1e-10)
        half = relax_to_prior_perturbations(self.forecast, analysis, 0.5)
        _close(self, half.mean, analysis.mean, rel=1e-12)

    def test_rtps_spread(self) -> None:
        """Every spread becomes ``(1 - alpha) sigma^a + alpha sigma^f``."""
        analysis = self.analysis.ensemble
        relaxed = relax_to_prior_spread(self.forecast, analysis, 0.3)
        _close(self, relaxed.mean, analysis.mean, rel=1e-12)
        for j in range(3):
            target = 0.7 * math.sqrt(analysis.covariance[j][j]) + 0.3 * math.sqrt(
                self.forecast.covariance[j][j]
            )
            self.assertAlmostEqual(math.sqrt(relaxed.covariance[j][j]), target, 12)

    def test_validation(self) -> None:
        """Unknown methods and factors outside [0, 1] raise."""
        with self.assertRaises(ValueError):
            Relaxation("rtpx", 0.5)
        with self.assertRaises(ValueError):
            Relaxation("rtps", 1.5)
        with self.assertRaises(ValueError):
            relax_to_prior_spread(self.forecast, self.analysis.ensemble, -0.1)


class FilterOptionTests(unittest.TestCase):
    """The filter records its settings and refuses inconsistent options."""

    def test_settings_are_recorded_and_validated(self) -> None:
        """Results carry the analysis, localization, and relaxation."""
        model, localization = _two_tracks()
        data = simulate(model, 3, seed=1).observations
        relaxation = Relaxation("rtps", 0.5)
        result = ensemble_kalman_filter(
            model,
            data,
            ensemble_size=10,
            seed=2,
            analysis="letkf",
            localization=localization,
            relaxation=relaxation,
        )
        self.assertEqual(result.analysis, "letkf")
        self.assertIs(result.localization, localization)
        self.assertIs(result.relaxation, relaxation)
        default = ensemble_kalman_filter(model, data, ensemble_size=10, seed=2)
        self.assertEqual(
            (default.analysis, default.localization, default.relaxation),
            ("stochastic", None, None),
        )
        for options in (
            {"analysis": "etkf", "localization": localization},
            {"analysis": "eakf", "localization": localization},
            {"analysis": "letkf"},
            {"analysis": "square-root"},
            {"localization": Localization((0.0,), (0.0, 1.0), 1.0)},
        ):
            with self.subTest(**{key: str(value) for key, value in options.items()}):
                with self.assertRaises(ValueError):
                    ensemble_kalman_filter(model, data, ensemble_size=10, **options)
        with self.assertRaises(TypeError):
            ensemble_kalman_filter(model, data, ensemble_size=10, relaxation=0.5)

    def test_letkf_filter_with_infinite_radius_equals_etkf_filter(self) -> None:
        """The cycles of both filters coincide, missing observations included."""
        model, _ = _two_tracks()
        data = list(simulate(model, 4, seed=4).observations)
        data[1] = None
        wide = Localization((0.0, 0.0, 100.0, 100.0), (0.0, 100.0), math.inf)
        local = ensemble_kalman_filter(
            model, data, ensemble_size=6, seed=8, analysis="letkf", localization=wide
        )
        global_ = ensemble_kalman_filter(
            model, data, ensemble_size=6, seed=8, analysis="etkf"
        )
        self.assertEqual(local.ensemble.members, global_.ensemble.members)
        self.assertEqual(local.analysis_means, global_.analysis_means)

    def test_eakf_and_etkf_filters_agree_on_a_linear_model(self) -> None:
        """Both square-root filters give the same first analysis moments.

        Later cycles differ because the two square roots give different
        members, which the forecast noise then spreads differently.
        """
        model, _ = _two_tracks()
        data = simulate(model, 5, seed=4).observations
        etkf = ensemble_kalman_filter(
            model, data, ensemble_size=8, seed=1, analysis="etkf"
        )
        eakf = ensemble_kalman_filter(
            model, data, ensemble_size=8, seed=1, analysis="eakf"
        )
        _close(self, eakf.analysis_means[1], etkf.analysis_means[1], rel=1e-10)
        _close(
            self,
            eakf.analysis_covariances[1],
            etkf.analysis_covariances[1],
            rel=1e-10,
        )
        self.assertEqual(eakf.forecast_means[0], etkf.forecast_means[0])


class SmootherTests(unittest.TestCase):
    """The fixed-lag EnKS reuses the filter and approaches the RTS smoother."""

    def test_lag_zero_is_the_filter_and_the_filter_part_is_unchanged(self) -> None:
        """Lag 0 returns the filter analyses; any lag leaves the filter as is."""
        model = _constant_velocity()
        data = list(simulate(model, 6, seed=2).observations)
        data[2] = None
        reference = ensemble_kalman_filter(model, data, ensemble_size=15, seed=4)
        zero = ensemble_kalman_smoother(model, data, ensemble_size=15, lag=0, seed=4)
        self.assertEqual(zero.smoothed_means, reference.analysis_means)
        self.assertEqual(zero.filter.analysis_means, reference.analysis_means)
        three = ensemble_kalman_smoother(model, data, ensemble_size=15, lag=3, seed=4)
        self.assertEqual(three.filter.ensemble.members, reference.ensemble.members)
        self.assertEqual(three.smoothed_means[-1], reference.analysis_means[-1])
        self.assertEqual(len(three.smoothed_means), 7)
        with self.assertRaises(ValueError):
            ensemble_kalman_smoother(model, data, ensemble_size=15, lag=-1)

    def test_smoother_approaches_rts(self) -> None:
        """Fixed-interval EnKS moments lie within k standard errors of RTS.

        Units as in the localized-filter test. Offline runs over seeds 0-3
        reached 7.9 (means) and 2.8 (variances): a smoothed early state
        collects the sampling error of every later analysis, hence the wider
        SMOOTHER_MEAN_BAND. The unsmoothed (filtered) position of ``x_0``
        is 19 units from the RTS mean at ``N = 100`` and 38 at ``N = 400``.
        """
        model = _constant_velocity()
        data = simulate(model, 6, seed=7).observations
        smoothed = rts_smoother(kalman_filter(model, data)).smoothed
        for analysis in ("stochastic", "etkf"):
            for size in (100, 400):
                with self.subTest(analysis=analysis, size=size):
                    result = ensemble_kalman_smoother(
                        model,
                        data,
                        ensemble_size=size,
                        lag=6,
                        seed=5,
                        analysis=analysis,
                    )
                    for k in (0, 3, 6):
                        for i in range(2):
                            variance = smoothed[k].covariance[i][i]
                            self.assertLess(
                                abs(result.smoothed_means[k][i] - smoothed[k].mean[i]),
                                SMOOTHER_MEAN_BAND * math.sqrt(variance / size),
                            )
                            self.assertLess(
                                abs(result.smoothed_covariances[k][i][i] - variance),
                                VARIANCE_BAND * math.sqrt(2.0 / size) * variance,
                            )

    def test_smoothing_lowers_the_variance_of_early_states(self) -> None:
        """Later observations shrink the spread of ``x_0`` for every variant."""
        model, localization = _two_tracks()
        data = simulate(model, 6, seed=3).observations
        for options in (
            {},
            {"analysis": "etkf"},
            {"analysis": "eakf", "relaxation": Relaxation("rtpp", 0.2)},
            {"analysis": "letkf", "localization": localization},
            {"localization": localization, "relaxation": Relaxation("rtps", 0.3)},
        ):
            with self.subTest(**{key: str(value) for key, value in options.items()}):
                result = ensemble_kalman_smoother(
                    model, data, ensemble_size=40, lag=6, seed=6, **options
                )
                filtered = result.filter.analysis_covariances[0]
                self.assertLess(result.smoothed_covariances[0][0][0], filtered[0][0])
                self.assertEqual(
                    result.smoothed_means[-1], result.filter.analysis_means[-1]
                )


class FixedStepTrajectoryTests(unittest.TestCase):
    """Fixed-step trajectories: grid, order, and differentiability."""

    def test_grid_and_exactness(self) -> None:
        """Heun and RK4 are exact for a linear-in-time solution."""
        trajectory = integrate_fixed_step(
            lambda t, y: (1.0, 2.0 * t), 0.5, (0.0, 0.25), step=0.1, steps=10
        )
        self.assertEqual(len(trajectory.states), 11)
        self.assertEqual(trajectory.times[-1], 0.5 + 10 * 0.1)
        self.assertEqual(trajectory.evaluations, 40)
        t = trajectory.times[-1]
        _close(self, trajectory.final_state, (t - 0.5, t * t), rel=1e-13)
        empty = integrate_fixed_step(
            lambda t, y: (y[0],), 0.0, (1.0,), step=0.1, steps=0
        )
        self.assertEqual(empty.states, ((1.0,),))
        for kwargs in (
            {"step": 0.0, "steps": 1},
            {"step": 0.1, "steps": -1},
            {"step": 0.1, "steps": 1, "method": "rk45"},
        ):
            with self.assertRaises(ValueError):
                integrate_fixed_step(lambda t, y: (y[0],), 0.0, (1.0,), **kwargs)

    def test_rk4_order_on_lorenz63(self) -> None:
        """Halving the step divides the RK4 error by about sixteen."""
        start = (1.0, 2.0, 20.0)
        reference = integrate_fixed_step(
            lorenz63_rhs, 0.0, start, step=0.2 / 1024, steps=1024
        ).final_state
        errors = []
        for steps in (8, 16, 32):
            final = integrate_fixed_step(
                lorenz63_rhs, 0.0, start, step=0.2 / steps, steps=steps
            ).final_state
            errors.append(
                max(abs(a - b) for a, b in zip(final, reference, strict=True))
            )
        for coarse, fine in zip(errors, errors[1:], strict=False):
            self.assertGreater(math.log2(coarse / fine), 3.6)
            self.assertLess(math.log2(coarse / fine), 4.4)

    def test_reverse_gradient_through_the_trajectory(self) -> None:
        """Reverse mode matches central differences of a final-state functional."""

        def functional(x0):
            final = integrate_fixed_step(
                lorenz63_rhs, 0.0, x0, step=0.01, steps=20
            ).final_state
            return final[0] * final[2] + final[1] * final[1]

        point = (1.0, 2.0, 20.0)
        gradient = reverse_gradient(functional, point)
        for i in range(3):
            h = 1e-5
            up = list(point)
            down = list(point)
            up[i] += h
            down[i] -= h
            central = (functional(up) - functional(down)) / (2.0 * h)
            self.assertTrue(math.isclose(gradient[i], central, rel_tol=1e-6))


class LorenzModelTests(unittest.TestCase):
    """Known facts about the Lorenz test systems."""

    def test_lorenz63_fixed_points(self) -> None:
        """The origin and ``C+-`` are equilibria of the tendency."""
        rho, beta = 28.0, 8.0 / 3.0
        root = math.sqrt(beta * (rho - 1.0))
        self.assertEqual(lorenz63_rhs(0.0, (0.0, 0.0, 0.0)), (0.0, 0.0, 0.0))
        for sign in (1.0, -1.0):
            tendency = lorenz63_rhs(0.0, (sign * root, sign * root, rho - 1.0))
            self.assertLess(max(map(abs, tendency)), 1e-12)
        model = lorenz63(step=0.01, substeps=5, observed_components=(0, 2))
        moved = model.transition((root, root, rho - 1.0))
        _close(self, moved, (root, root, rho - 1.0), rel=1e-12)
        self.assertEqual(model.observe((1.0, 2.0, 3.0)), (1.0, 3.0))
        with self.assertRaises(ValueError):
            lorenz63(observed_components=(0, 0))

    def test_lorenz96_equilibrium_and_shift_equivariance(self) -> None:
        """``x_i = F`` is a fixed point and cyclic shifts commute with the flow."""
        forcing = 8.0
        self.assertEqual(lorenz96_rhs(0.0, (forcing,) * 40), (0.0,) * 40)
        model = lorenz96()
        self.assertEqual(model.transition((forcing,) * 40), (forcing,) * 40)
        stream = random.Random(3)
        state = tuple(stream.gauss(forcing, 2.0) for _ in range(40))
        moved = model.transition(state)
        for shift in (1, 7):
            rolled = state[shift:] + state[:shift]
            expected = moved[shift:] + moved[:shift]
            self.assertEqual(model.transition(rolled), expected)
            self.assertEqual(
                lorenz96_rhs(0.0, rolled),
                tuple(
                    lorenz96_rhs(0.0, state)[shift:] + lorenz96_rhs(0.0, state)[:shift]
                ),
            )
        sparse = lorenz96(observe_every=4)
        self.assertEqual(sparse.observation_dimension, 10)
        self.assertEqual(sparse.observe(state), state[::4])
        with self.assertRaises(ValueError):
            lorenz96(dimension=3)

    def test_transition_jacobian_is_exact_for_the_discrete_map(self) -> None:
        """The forward-mode Jacobian matches central differences."""
        model = lorenz96(dimension=8, substeps=2, step=0.025)
        state = tuple(8.0 + 0.5 * math.sin(i) for i in range(8))
        jacobian = model.transition_jacobian(state)
        h = 1e-6
        for j in (0, 5):
            up = list(state)
            down = list(state)
            up[j] += h
            down[j] -= h
            plus, minus = model.transition(up), model.transition(down)
            for i in range(8):
                central = (plus[i] - minus[i]) / (2.0 * h)
                self.assertAlmostEqual(jacobian[i][j], central, places=7)


class LorenzTwinExperimentTests(unittest.TestCase):
    """Lorenz-96 twin experiment, ``J = 40``, ``N = 20``, every component observed.

    Over seeds 1-4 (40 cycles, the last 25 averaged) the LETKF with
    Gaspari-Cohn half-width 2 and RTPS 0.5 reached time-averaged analysis
    RMSE 0.31-0.36, the localized stochastic EnKF 0.31-0.36, and the
    non-localized stochastic EnKF 2.0-2.4, for an observation error of 1.
    Only inequalities with wide margins are asserted.
    """

    def test_localization_is_needed_with_twenty_members(self) -> None:
        """LETKF + RTPS beats the observation error; plain EnKF does not."""
        start = tuple(8.0 + (0.01 if i == 0 else 0.0) for i in range(40))
        spun = integrate_fixed_step(
            lorenz96_rhs, 0.0, start, step=0.05, steps=400
        ).final_state
        model = lorenz96(initial_mean=spun)
        steps, burn = 40, 15
        truth = simulate(model, steps, seed=1)
        localization = Localization(range(40), range(40), 2.0, period=40)

        def rmse(result) -> float:
            errors = [
                math.sqrt(
                    sum(
                        (a - b) ** 2
                        for a, b in zip(
                            result.analysis_means[k], truth.states[k], strict=True
                        )
                    )
                    / 40
                )
                for k in range(burn + 1, steps + 1)
            ]
            return sum(errors) / len(errors)

        letkf = ensemble_kalman_filter(
            model,
            truth.observations,
            ensemble_size=20,
            seed=101,
            analysis="letkf",
            localization=localization,
            relaxation=Relaxation("rtps", 0.5),
        )
        plain = ensemble_kalman_filter(
            model, truth.observations, ensemble_size=20, seed=101
        )
        self.assertLess(rmse(letkf), 0.6)
        self.assertGreater(rmse(plain), 1.0)
        self.assertGreater(rmse(plain), 2.0 * rmse(letkf))


if __name__ == "__main__":
    unittest.main()
