"""Filter consistency: NEES, NIS, innovation whiteness, covariance realism.

Oracles: chi-square acceptance bounds pinned from base R 4.6.2 (``qchisq``,
``qnorm``, ``qbinom``; each pin carries its R expression), the squared
Mahalanobis distance of ``probstat.validation.mahalanobis_metric``, and
seeded simulation of a correctly specified linear-Gaussian filter, which must
be accepted, against filters whose noise covariances are deliberately halved
or doubled, which must be flagged. The verdicts are judged on the chi-square
scale of the statistic against its bounds; no digit of a simulated statistic
is pinned. The tests never call R.

References:
    [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001], [REF-HORWOOD-ETAL-2014],
    [REF-MARDIA-KENT-BIBBY-1979], [REF-LJUNG-BOX-1978]. Checks
    [EQ-FILT-NEES-001], [EQ-FILT-NEES-002], [EQ-FILT-NIS-001],
    [EQ-FILT-WHITE-001], [EQ-FILT-REALISM-001].

"""

from __future__ import annotations

import math
import unittest

from probstat.filtering import (
    LinearGaussianStateSpace,
    covariance_realism,
    innovation_whiteness_test,
    kalman_filter,
    nees,
    nees_from_filter,
    nees_series,
    nees_test,
    nees_test_from_filters,
    nis_series,
    nis_test,
    nis_test_from_filter,
    simulate,
)
from probstat.foundations import RandomStream
from probstat.validation import mahalanobis_metric

TRANSITION = ((1.0, 1.0), (0.0, 1.0))
PROCESS = ((0.05, 0.02), (0.02, 0.04))
OBSERVATION = ((1.0, 0.0),)
NOISE = ((0.5,),)


def _model(scale: float = 1.0) -> LinearGaussianStateSpace:
    """Return a constant-velocity model with noise covariances times ``scale``.

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001]. No equation is evaluated.

    """
    return LinearGaussianStateSpace(
        TRANSITION,
        tuple(tuple(scale * v for v in row) for row in PROCESS),
        OBSERVATION,
        ((scale * NOISE[0][0],),),
        (0.0, 0.0),
        ((1.0, 0.0), (0.0, 1.0)),
    )


def _runs(scale: float, runs: int = 30, steps: int = 40) -> tuple[list, list]:
    """Simulate the true model and filter it assuming noise times ``scale``.

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001]. No equation is evaluated.

    """
    truth_model, filter_model = _model(1.0), _model(scale)
    results, truths = [], []
    for index in range(runs):
        data = simulate(truth_model, steps, seed=4000 + index)
        results.append(kalman_filter(filter_model, data.observations))
        truths.append(data.states)
    return results, truths


class BoundTests(unittest.TestCase):
    """Chi-square acceptance bounds against R.

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Section 5.4]. Checks
        [EQ-FILT-NEES-002], [EQ-FILT-NIS-001].

    """

    def test_nees_bounds_match_r_pins(self) -> None:
        """N = 20 runs of a 4-D state give the pinned two-sided bounds.

        References:
            [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Section 5.4]. Checks
            [EQ-FILT-NEES-002].

        """
        # qchisq(c(0.025, 0.975), 80) / 20
        result = nees_test([[4.0] * 5 for _ in range(20)], dimension=4)
        self.assertAlmostEqual(result.lower, 2.857658644178897, places=9)
        self.assertAlmostEqual(result.upper, 5.331428386583286, places=9)
        self.assertEqual(
            (result.degrees_of_freedom, result.runs, result.dimension), (80, 20, 4)
        )
        self.assertEqual(result.verdict, "consistent")
        # qchisq(c(0.05, 0.95), 150) / 50 for N = 50, n = 3, confidence 0.90
        narrow = nees_test([[3.0] * 2 for _ in range(50)], dimension=3, confidence=0.90)
        self.assertAlmostEqual(narrow.lower, 2.453835507742345, places=9)
        self.assertAlmostEqual(narrow.upper, 3.591612683083611, places=9)

    def test_nis_bounds_match_r_pins(self) -> None:
        """T = 100 steps of a 2-D innovation give the pinned bounds.

        References:
            [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Section 5.4]. Checks
            [EQ-FILT-NIS-001].

        """
        # qchisq(c(0.025, 0.975), 200) / 100
        identity = ((1.0, 0.0), (0.0, 1.0))
        result = nis_test([(1.0, 1.0)] * 100, [identity] * 100)
        self.assertAlmostEqual(result.lower, 1.627279825018463, places=9)
        self.assertAlmostEqual(result.upper, 2.41057895506311, places=9)
        self.assertEqual((result.degrees_of_freedom, result.steps), (200, 100))
        self.assertAlmostEqual(result.statistic, 2.0)
        self.assertEqual(result.verdict, "consistent")

    def test_missing_observations_reduce_degrees_of_freedom(self) -> None:
        """Only observed steps count; dof is their total dimension.

        References:
            [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Section 5.4]. Checks
            [EQ-FILT-NIS-001].

        """
        # qchisq(c(0.025, 0.975), 80) / 80
        innovations = [(1.0,) if k % 5 else None for k in range(100)]
        covariances = [((1.0,),) if k % 5 else None for k in range(100)]
        result = nis_test(innovations, covariances)
        self.assertEqual((result.steps, result.degrees_of_freedom), (80, 80))
        self.assertAlmostEqual(result.lower, 0.7144146610447242, places=9)
        self.assertAlmostEqual(result.upper, 1.332857096645822, places=9)
        self.assertEqual(result.per_step[0], None)
        self.assertEqual(result.per_step[1], 1.0)

    def test_verdict_words_follow_the_bounds(self) -> None:
        """A statistic above the upper bound is optimistic, below pessimistic.

        References:
            [REF-HORWOOD-ETAL-2014]. Checks [EQ-FILT-NEES-002].

        """
        high = nees_test([[8.0] * 5 for _ in range(20)], dimension=4)
        low = nees_test([[2.0] * 5 for _ in range(20)], dimension=4)
        self.assertEqual((high.verdict, high.fraction_above), ("optimistic", 1.0))
        self.assertEqual((low.verdict, low.fraction_below), ("pessimistic", 1.0))

    def test_validation(self) -> None:
        """Invalid arguments raise ``TypeError`` or ``ValueError``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. Checks
            [EQ-FILT-NEES-002].

        """
        with self.assertRaises(ValueError):
            nees_test([], dimension=2)
        with self.assertRaises(ValueError):
            nees_test([[1.0, 2.0], [1.0]], dimension=2)
        with self.assertRaises(ValueError):
            nees_test([[-1.0]], dimension=2)
        with self.assertRaises(ValueError):
            nees_test([[1.0]], dimension=0)
        with self.assertRaises(TypeError):
            nees_test([[1.0]], dimension=2.0)  # type: ignore[arg-type]
        for bad in (0.0, 1.0, -0.2, math.nan):
            with self.assertRaises(ValueError):
                nees_test([[1.0]], dimension=2, confidence=bad)
        with self.assertRaises(ValueError):
            nis_test([None, None], [None, None])
        with self.assertRaises(ValueError):
            nis_test([(1.0,)], [((1.0,),), ((1.0,),)])
        with self.assertRaises(ValueError):
            nis_series([(1.0,), None], [None, None])
        with self.assertRaises(ValueError):
            nis_test_from_filter(_runs(1.0, runs=1, steps=3)[0][0], burn_in=-1)
        with self.assertRaises(TypeError):
            nis_test_from_filter(_runs(1.0, runs=1, steps=3)[0][0], burn_in=1.5)  # type: ignore[arg-type]


class NeesStatisticTests(unittest.TestCase):
    """The per-step statistic against independent computations.

    References:
        [REF-MARDIA-KENT-BIBBY-1979]. Checks [EQ-FILT-NEES-001].

    """

    def test_nees_matches_the_closed_form_and_validation_metric(self) -> None:
        """The value equals the 2-D closed form and ``mahalanobis_metric``.

        References:
            [REF-MARDIA-KENT-BIBBY-1979]. Checks [EQ-FILT-NEES-001].

        """
        error, cov = (0.7, -1.2), ((2.0, 0.6), (0.6, 1.0))
        det = 2.0 * 1.0 - 0.6**2
        closed = (
            1.0 * error[0] ** 2 - 2 * 0.6 * error[0] * error[1] + 2.0 * error[1] ** 2
        ) / det
        self.assertAlmostEqual(nees(error, cov), closed, places=12)
        reference = mahalanobis_metric(error, (0.0, 0.0), cov)
        self.assertAlmostEqual(nees(error, cov), reference.squared_distance, places=12)

    def test_series_and_filter_helpers_align(self) -> None:
        """``nees_from_filter`` equals ``nees_series`` on the stored states.

        References:
            [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Section 5.4]. Checks
            [EQ-FILT-NEES-001].

        """
        results, truths = _runs(1.0, runs=1, steps=12)
        result, truth = results[0], truths[0]
        filtered = nees_from_filter(result, truth)
        self.assertEqual(len(filtered), 13)
        errors = [
            tuple(t - m for t, m in zip(x, s.mean, strict=True))
            for x, s in zip(truth, result.filtered, strict=True)
        ]
        manual = nees_series(errors, [s.covariance for s in result.filtered])
        self.assertEqual(filtered, manual)
        predicted = nees_from_filter(result, truth[1:], states="predicted")
        self.assertEqual(len(predicted), 12)
        with self.assertRaises(ValueError):
            nees_from_filter(result, truth[1:])
        with self.assertRaises(ValueError):
            nees_from_filter(result, truth, states="smoothed")

    def test_validation(self) -> None:
        """Bad shapes and non-positive-definite covariances are rejected.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. Checks
            [EQ-FILT-NEES-001].

        """
        with self.assertRaises(ValueError):
            nees((), ())
        with self.assertRaises(ValueError):
            nees((1.0, 2.0), ((1.0,),))
        with self.assertRaises(ValueError):
            nees((1.0, 2.0), ((1.0, 0.0), (0.0, 0.0)))
        with self.assertRaises(ValueError):
            nees((1.0, 2.0), ((1.0, 0.5), (0.0, 1.0)))
        with self.assertRaises(TypeError):
            nees((1.0, "x"), ((1.0, 0.0), (0.0, 1.0)))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            nees_series([(1.0,)], [])
        with self.assertRaises(ValueError):
            nees_series([], [])
        with self.assertRaises(ValueError):
            nees_test_from_filters([], [])
        with self.assertRaises(ValueError):
            nees_test_from_filters(_runs(1.0, runs=1, steps=3)[0], [])


class SimulatedFilterTests(unittest.TestCase):
    """A correct filter is accepted; halved or doubled noise is flagged.

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Section 5.4]. Checks
        [EQ-FILT-NEES-002], [EQ-FILT-NIS-001].

    """

    def test_correct_filter_is_consistent(self) -> None:
        """NEES, NIS and whiteness all accept the correctly specified filter.

        References:
            [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Section 5.4]. Checks
            [EQ-FILT-NEES-002], [EQ-FILT-NIS-001], [EQ-FILT-WHITE-001].

        """
        results, truths = _runs(1.0)
        test = nees_test_from_filters(results, truths)
        self.assertEqual(test.verdict, "consistent")
        self.assertLessEqual(test.fraction_below + test.fraction_above, 0.25)
        self.assertGreater(test.mean_statistic, test.lower)
        self.assertLess(test.mean_statistic, test.upper)
        accepted = [nis_test_from_filter(r).verdict == "consistent" for r in results]
        self.assertGreaterEqual(sum(accepted), 26)
        white = innovation_whiteness_test(
            results[0].innovations, results[0].innovation_covariances, max_lag=10
        )
        self.assertTrue(white.white)

    def test_halved_noise_covariances_are_optimistic(self) -> None:
        """A filter with noise covariances halved is flagged as optimistic.

        References:
            [REF-HORWOOD-ETAL-2014]. Checks [EQ-FILT-NEES-002],
            [EQ-FILT-NIS-001].

        """
        results, truths = _runs(0.5)
        test = nees_test_from_filters(results, truths)
        self.assertEqual(test.verdict, "optimistic")
        self.assertGreater(test.mean_statistic, test.upper)
        self.assertGreater(test.fraction_above, 0.9)
        verdicts = [nis_test_from_filter(r).verdict for r in results]
        self.assertGreater(verdicts.count("optimistic"), 20)
        self.assertEqual(verdicts.count("pessimistic"), 0)

    def test_doubled_noise_covariances_are_pessimistic(self) -> None:
        """A filter with noise covariances doubled is flagged as pessimistic.

        References:
            [REF-HORWOOD-ETAL-2014]. Checks [EQ-FILT-NEES-002],
            [EQ-FILT-NIS-001].

        """
        results, truths = _runs(2.0)
        test = nees_test_from_filters(results, truths)
        self.assertEqual(test.verdict, "pessimistic")
        self.assertLess(test.mean_statistic, test.lower)
        verdicts = [nis_test_from_filter(r).verdict for r in results]
        self.assertGreater(verdicts.count("pessimistic"), 20)
        self.assertEqual(verdicts.count("optimistic"), 0)

    def test_burn_in_drops_leading_steps(self) -> None:
        """``burn_in`` removes leading steps from the NIS average.

        References:
            [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Section 5.4]. Checks
            [EQ-FILT-NIS-001].

        """
        result = _runs(1.0, runs=1, steps=20)[0][0]
        full = nis_test_from_filter(result)
        trimmed = nis_test_from_filter(result, burn_in=5)
        self.assertEqual((full.steps, trimmed.steps), (20, 15))


class WhitenessTests(unittest.TestCase):
    """The innovation whiteness test on synthetic sequences.

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Section 5.4]. Checks
        [EQ-FILT-WHITE-001].

    """

    @staticmethod
    def _sequence(rho: float, size: int, seed: int) -> list[tuple[float]]:
        """Return a unit-variance AR(1) sequence with coefficient ``rho``.

        References:
            [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Section 5.4]. No equation is
            evaluated.

        """
        generator = RandomStream(seed).generator()
        value = generator.gauss(0.0, 1.0)
        values = []
        for _ in range(size):
            value = rho * value + math.sqrt(1.0 - rho * rho) * generator.gauss(0.0, 1.0)
            values.append((value,))
        return values

    def test_white_noise_passes_and_matches_the_r_bound(self) -> None:
        """White innovations pass; the bound is the pinned ``z / sqrt(T m)``.

        References:
            [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Section 5.4]. Checks
            [EQ-FILT-WHITE-001].

        """
        # qnorm(0.975) / sqrt(100 * 2)
        generator = RandomStream(31).generator()
        series = [(generator.gauss(0, 1), generator.gauss(0, 1)) for _ in range(100)]
        identity = ((1.0, 0.0), (0.0, 1.0))
        result = innovation_whiteness_test(series, [identity] * 100, max_lag=10)
        self.assertAlmostEqual(result.bound, 0.1385903824349677, places=12)
        self.assertTrue(result.white)
        self.assertEqual(result.degrees_of_freedom, 10)
        manual = (
            100
            * 2
            * 102
            * sum(r * r / (100 - k) for k, r in enumerate(result.autocorrelation, 1))
        )
        self.assertAlmostEqual(result.statistic, manual, places=10)
        self.assertGreater(result.p_value, 0.05)
        self.assertEqual(
            (result.steps, result.dimension, len(result.lags)), (100, 2, 10)
        )

    def test_correlated_innovations_are_rejected(self) -> None:
        """An AR(1) sequence with coefficient 0.6 fails with a lag-1 spike.

        References:
            [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Section 5.4]. Checks
            [EQ-FILT-WHITE-001].

        """
        series = self._sequence(0.6, 300, 5)
        result = innovation_whiteness_test(series, [((1.0,),)] * 300, max_lag=10)
        self.assertFalse(result.white)
        self.assertGreaterEqual(result.outside, 3)
        self.assertLess(result.p_value, 1e-10)
        self.assertAlmostEqual(result.autocorrelation[0], 0.6, delta=0.12)

    def test_standardisation_uses_the_stated_covariance(self) -> None:
        """Scaling innovations and covariances together changes nothing.

        References:
            [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Section 5.4]. Checks
            [EQ-FILT-WHITE-001].

        """
        series = self._sequence(0.0, 80, 9)
        base = innovation_whiteness_test(series, [((1.0,),)] * 80, max_lag=5)
        scaled = innovation_whiteness_test(
            [(3.0 * v[0],) for v in series], [((9.0,),)] * 80, max_lag=5
        )
        for a, b in zip(base.autocorrelation, scaled.autocorrelation, strict=True):
            self.assertAlmostEqual(a, b, places=12)

    def test_missing_steps_break_pairs_but_run(self) -> None:
        """Missing observations are skipped without error.

        References:
            [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Section 5.4]. Checks
            [EQ-FILT-WHITE-001].

        """
        series = self._sequence(0.0, 120, 13)
        innovations = [None if k % 7 == 0 else v for k, v in enumerate(series)]
        covariances = [None if k % 7 == 0 else ((1.0,),) for k in range(120)]
        result = innovation_whiteness_test(innovations, covariances, max_lag=6)
        self.assertEqual(result.steps, 120 - 18)
        self.assertTrue(result.white)

    def test_validation(self) -> None:
        """Invalid arguments raise ``TypeError`` or ``ValueError``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. Checks
            [EQ-FILT-WHITE-001].

        """
        one = [((1.0,),)] * 30
        data = [(0.1 * k,) for k in range(30)]
        with self.assertRaises(ValueError):
            innovation_whiteness_test(data, one, max_lag=0)
        with self.assertRaises(TypeError):
            innovation_whiteness_test(data, one, max_lag=2.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            innovation_whiteness_test(data, one, max_lag=40)
        with self.assertRaises(ValueError):
            innovation_whiteness_test(data, one[:-1], max_lag=2)
        with self.assertRaises(ValueError):
            innovation_whiteness_test([(0.0,)] * 30, one, max_lag=2)
        with self.assertRaises(ValueError):
            innovation_whiteness_test(data[:-1] + [(1.0, 2.0)], one, max_lag=2)
        with self.assertRaises(ValueError):
            innovation_whiteness_test([None] + data[1:], one, max_lag=2)


class CovarianceRealismTests(unittest.TestCase):
    """Distributional check of (error, covariance) pairs.

    References:
        [REF-HORWOOD-ETAL-2014]. Checks [EQ-FILT-REALISM-001].

    """

    @staticmethod
    def _pairs(
        count: int, scale: float, seed: int, *, heavy: bool = False
    ) -> tuple[list[tuple[float, ...]], list[tuple[tuple[float, ...], ...]]]:
        """Draw errors with covariance ``scale * P_i`` for varying ``P_i``.

        References:
            [REF-HORWOOD-ETAL-2014]. No equation is evaluated.

        """
        generator = RandomStream(seed).generator()
        errors, covariances = [], []
        for _ in range(count):
            a, b, c = (0.5 + 3.0 * generator.random() for _ in range(3))
            rho = 0.5 * (2.0 * generator.random() - 1.0)
            cov = (
                (a, rho * math.sqrt(a * b), 0.0),
                (rho * math.sqrt(a * b), b, 0.0),
                (0.0, 0.0, c),
            )
            sd = (math.sqrt(a), math.sqrt(b), math.sqrt(c))
            if heavy:
                # Multivariate t with 3 degrees of freedom rescaled to unit covariance.
                shock = 1.0 / math.sqrt(
                    sum(generator.gauss(0, 1) ** 2 for _ in range(3))
                )
                z = [generator.gauss(0, 1) * shock for _ in range(3)]
            else:
                z = [generator.gauss(0, 1) for _ in range(3)]
            # Correlate the first two coordinates through the Cholesky factor.
            e0 = sd[0] * z[0]
            e1 = rho * sd[1] * z[0] + sd[1] * math.sqrt(1 - rho * rho) * z[1]
            errors.append(
                (
                    math.sqrt(scale) * e0,
                    math.sqrt(scale) * e1,
                    math.sqrt(scale) * sd[2] * z[2],
                )
            )
            covariances.append(cov)
        return errors, covariances

    def test_correct_covariances_are_accepted(self) -> None:
        """Errors drawn from their covariances pass shape and scale checks.

        References:
            [REF-HORWOOD-ETAL-2014]. Checks [EQ-FILT-REALISM-001].

        """
        errors, covariances = self._pairs(400, 1.0, 17)
        result = covariance_realism(errors, covariances)
        self.assertEqual((result.count, result.dimension), (400, 3))
        self.assertTrue(result.shape_consistent)
        self.assertGreater(result.ks_p_value, 0.01)
        self.assertEqual(result.verdict, "consistent")
        self.assertLess(result.scale_interval[0], 1.0)
        self.assertGreater(result.scale_interval[1], 1.0)
        self.assertAlmostEqual(
            result.scale_factor, math.fsum(result.squared_distances) / 1200.0, places=12
        )

    def test_scale_error_is_detected_and_estimated(self) -> None:
        """Errors 2.25 times too large (variance) give an interval around 2.25.

        References:
            [REF-HORWOOD-ETAL-2014]. Checks [EQ-FILT-REALISM-001].

        """
        errors, covariances = self._pairs(400, 2.25, 19)
        result = covariance_realism(errors, covariances, test="kolmogorov-smirnov")
        self.assertEqual(result.verdict, "optimistic")
        self.assertLess(result.scale_interval[0], 2.25)
        self.assertGreater(result.scale_interval[1], 2.25)
        self.assertFalse(result.shape_consistent)
        self.assertEqual(result.p_value, result.ks_p_value)
        errors, covariances = self._pairs(400, 0.5, 23)
        low = covariance_realism(errors, covariances)
        self.assertEqual(low.verdict, "pessimistic")
        self.assertLess(low.scale_interval[1], 1.0)

    def test_scale_interval_matches_r_pins(self) -> None:
        """With all squared distances equal to two, the interval is pinned.

        References:
            [REF-HORWOOD-ETAL-2014]. Checks [EQ-FILT-REALISM-001].

        """
        # N = 10 pairs of dimension 2 with d^2 = 2 each: s = 1, N n = 20;
        # qchisq(c(0.025, 0.975), 20) = 9.590777392264867, 34.16960690283834
        errors = [(1.0, 1.0)] * 10
        covariances = [((1.0, 0.0), (0.0, 1.0))] * 10
        result = covariance_realism(errors, covariances)
        self.assertAlmostEqual(result.scale_factor, 1.0, places=14)
        self.assertAlmostEqual(
            result.scale_interval[0], 20.0 / 34.16960690283834, places=10
        )
        self.assertAlmostEqual(
            result.scale_interval[1], 20.0 / 9.590777392264867, places=10
        )

    def test_heavy_tails_fail_the_shape_test_even_with_a_matching_scale(self) -> None:
        """Student-like errors with unit variance are caught by the shape test.

        References:
            [REF-HORWOOD-ETAL-2014]. Checks [EQ-FILT-REALISM-001].

        """
        errors, covariances = self._pairs(600, 1.0, 29, heavy=True)
        result = covariance_realism(errors, covariances)
        self.assertFalse(result.shape_consistent)
        self.assertLess(result.p_value, 0.01)

    def test_validation(self) -> None:
        """Invalid arguments raise ``ValueError`` or ``TypeError``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. Checks
            [EQ-FILT-REALISM-001].

        """
        eye = ((1.0, 0.0), (0.0, 1.0))
        with self.assertRaises(ValueError):
            covariance_realism([], [])
        with self.assertRaises(ValueError):
            covariance_realism([(1.0, 0.0)], [eye, eye])
        with self.assertRaises(ValueError):
            covariance_realism([(1.0, 0.0)], [eye], test="cramer")
        with self.assertRaises(ValueError):
            covariance_realism([(1.0, 0.0)], [eye], confidence=1.0)
        with self.assertRaises(ValueError):
            covariance_realism([(1.0, 0.0), (1.0,)], [eye, eye])
        with self.assertRaises(TypeError):
            covariance_realism([(1.0, None)], [eye])  # type: ignore[list-item]


class SyntheticChiSquareTests(unittest.TestCase):
    """Seeded chi-square data through the Monte-Carlo test.

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Section 5.4]. Checks
        [EQ-FILT-NEES-002].

    """

    def test_chi_square_data_are_accepted_and_scaled_data_are_not(self) -> None:
        """NEES drawn as sums of squared normals pass; times 1.6 they do not.

        References:
            [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Section 5.4]. Checks
            [EQ-FILT-NEES-002].

        """
        generator = RandomStream(41).generator()

        def draw(scale: float) -> list[list[float]]:
            """Return 40 runs of 30 steps of ``scale * chi2(3)``.

            References:
                [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Section 5.4]. No equation
                is evaluated.

            """
            return [
                [
                    scale * sum(generator.gauss(0, 1) ** 2 for _ in range(3))
                    for _ in range(30)
                ]
                for _ in range(40)
            ]

        self.assertEqual(nees_test(draw(1.0), dimension=3).verdict, "consistent")
        self.assertEqual(nees_test(draw(1.6), dimension=3).verdict, "optimistic")
        self.assertEqual(nees_test(draw(0.6), dimension=3).verdict, "pessimistic")


if __name__ == "__main__":
    unittest.main()
