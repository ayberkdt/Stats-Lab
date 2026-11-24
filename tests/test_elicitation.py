"""Expert elicitation, three-point laws, opinion pools and Cooke's classical model.

Oracles are exact recoveries of known parameters from exact quantiles, the
closed-form two-quantile normal and lognormal solutions, the least-squares
orthogonality conditions, the precision-weighted formula and a brute-force
midpoint normalisation for the logarithmic pool, mixture and quantile-average
moment formulas, and hand-computed Cooke scores with chi-square p-values
pinned from base R 4.6.2 (each pin carries the R expression; the tests never
call R).

References:
    [REF-OHAGAN-ETAL-2006], [REF-COOKE-1991], [REF-HANEA-NANE-2021],
    [REF-GENEST-ZIDEK-1986], [REF-CLEMEN-WINKLER-1999], [REF-MALCOLM-1959],
    [REF-VOSE-2008].
    Checks [EQ-INF-ELICIT-001], [EQ-INF-ELICIT-002], [EQ-INF-ELICIT-003],
    [EQ-INF-ELICIT-004], [EQ-INF-ELICIT-005], [EQ-INF-ELICIT-006],
    [EQ-INF-ELICIT-007].

"""

from __future__ import annotations

import math
import unittest

from probstat.distributions import (
    PERT,
    Affine,
    Beta,
    Gamma,
    LogNormal,
    Mixture,
    Normal,
    StudentT,
    Triangular,
    Uniform,
    Weibull,
)
from probstat.inference import (
    LogarithmicPool,
    VincentizedLaw,
    compare_elicited_fits,
    cooke_bin_probabilities,
    cooke_calibration,
    cooke_classical_model,
    cooke_decision_maker_law,
    cooke_information,
    cooke_intrinsic_ranges,
    cooke_weights,
    fit_from_quantiles,
    linear_opinion_pool,
    logarithmic_opinion_pool,
    pert_from_quantiles,
    pert_from_three_points,
    quantile_average,
    triangular_from_quantiles,
    triangular_from_three_points,
)
from probstat.inference.elicitation import _decision_maker_item

P3 = (0.05, 0.5, 0.95)
Z95 = 1.6448536269514715  # R: qnorm(0.95)


def _quantiles(law: object, probabilities: tuple[float, ...] = P3) -> list[float]:
    """Return the quantiles of ``law`` at ``probabilities``.

    References:
        [EQ-INF-ELICIT-001]. No new equation.

    """
    return [law.quantile(p) for p in probabilities]  # type: ignore[attr-defined]


class FitFromQuantilesTests(unittest.TestCase):
    """Exact recovery, closed forms, optimality and validation of the fits.

    References:
        [REF-OHAGAN-ETAL-2006], [REF-SHELF-OAKLEY-OHAGAN]. Checks
        [EQ-INF-ELICIT-001].

    """

    def test_exact_recovery_of_every_family(self) -> None:
        """Exact quantiles of a law give back its parameters with zero residual.

        References:
            [EQ-INF-ELICIT-001], [REF-OHAGAN-ETAL-2006].

        """
        cases = {
            "normal": (Normal(3.0, 2.0), {"mu": 3.0, "sigma": 2.0}),
            "student_t": (StudentT(5.0, 3.0, 2.0), {"mu": 3.0, "sigma": 2.0}),
            "lognormal": (LogNormal(1.0, 0.4), {"mu": 1.0, "sigma": 0.4}),
            "gamma": (Gamma(3.0, 0.5), {"shape": 3.0, "rate": 0.5}),
            "weibull": (Weibull(2.0, 5.0), {"shape": 2.0, "scale": 5.0}),
            "triangular": (
                Triangular(1.0, 3.0, 9.0),
                {"lower": 1.0, "mode": 3.0, "upper": 9.0},
            ),
            "pert": (PERT(1.0, 3.0, 9.0), {"lower": 1.0, "mode": 3.0, "upper": 9.0}),
        }
        for family, (law, truth) in cases.items():
            with self.subTest(family=family):
                fit = fit_from_quantiles(
                    family, P3, _quantiles(law), degrees_of_freedom=5
                )
                self.assertLess(fit.max_abs_residual, 1e-7)
                self.assertLess(fit.misfit, 1e-7)
                for name, value in truth.items():
                    self.assertAlmostEqual(fit.parameter(name), value, delta=1e-5)
        uniform = fit_from_quantiles("uniform", (0.1, 0.5, 0.9), [3.0, 4.5, 6.0])
        self.assertAlmostEqual(uniform.parameter("lower"), 2.625, 12)
        self.assertAlmostEqual(uniform.parameter("upper"), 6.375, 12)
        interval = Affine(Beta(2.0, 5.0), 10.0, 20.0)
        beta = fit_from_quantiles("beta", P3, _quantiles(interval), bounds=(10.0, 30.0))
        self.assertAlmostEqual(beta.parameter("alpha"), 2.0, delta=1e-5)
        self.assertAlmostEqual(beta.parameter("beta"), 5.0, delta=1e-5)
        bounded = fit_from_quantiles(
            "pert", P3, _quantiles(PERT(1.0, 3.0, 9.0)), bounds=(1.0, 9.0)
        )
        self.assertAlmostEqual(bounded.parameter("mode"), 3.0, delta=1e-6)
        self.assertEqual(bounded.n_parameters, 1)

    def test_normal_two_quantile_closed_form(self) -> None:
        """Two normal quantiles give ``sigma = (v2 - v1)/(z2 - z1)`` exactly.

        References:
            [EQ-INF-ELICIT-001]. R: qnorm(0.95) -> 1.6448536269514715.

        """
        fit = fit_from_quantiles(
            "normal", (0.05, 0.95), [10.0 - 2.0 * Z95, 10.0 + 2.0 * Z95]
        )
        self.assertEqual(fit.method, "closed-form")
        self.assertAlmostEqual(fit.parameter("mu"), 10.0, 12)
        self.assertAlmostEqual(fit.parameter("sigma"), 2.0, 12)
        z1, z2 = -1.2815515655446008, 1.9599639845400536  # R: qnorm(c(0.1, 0.975))
        fit = fit_from_quantiles("normal", (0.1, 0.975), [4.0, 9.0])
        self.assertAlmostEqual(fit.parameter("sigma"), 5.0 / (z2 - z1), 12)
        self.assertAlmostEqual(
            fit.parameter("mu"), 4.0 - fit.parameter("sigma") * z1, 12
        )

    def test_lognormal_two_quantile_closed_form(self) -> None:
        """Two lognormal quantiles recover ``(mu, sigma)`` in closed form.

        References:
            [EQ-INF-ELICIT-001]. R: qlnorm(c(0.05, 0.95), 0.5, 0.3) ->
            1.0065653700873924, 2.7005517070620426.

        """
        fit = fit_from_quantiles(
            "lognormal", (0.05, 0.95), [1.0065653700873924, 2.7005517070620426]
        )
        self.assertEqual(fit.method, "closed-form")
        self.assertAlmostEqual(fit.parameter("mu"), 0.5, 12)
        self.assertAlmostEqual(fit.parameter("sigma"), 0.3, 12)

    def test_weighted_normal_fit_satisfies_the_normal_equations(self) -> None:
        """An inconsistent normal fit has weighted-orthogonal residuals.

        The least-squares residuals ``r`` satisfy ``sum w r = 0`` and
        ``sum w r z = 0`` with ``z`` the standard normal quantiles.

        References:
            [EQ-INF-ELICIT-001].

        """
        probs = (0.05, 0.25, 0.5, 0.9)
        values = [1.0, 3.5, 4.0, 9.0]
        weights = [1.0, 2.0, 3.0, 0.5]
        fit = fit_from_quantiles("normal", probs, values, weights=weights)
        z = [Normal(0.0, 1.0).quantile(p) for p in probs]
        self.assertGreater(fit.max_abs_residual, 0.1)
        self.assertAlmostEqual(
            sum(w * r for w, r in zip(weights, fit.residuals, strict=False)), 0.0, 10
        )
        self.assertAlmostEqual(
            sum(w * r * x for w, r, x in zip(weights, fit.residuals, z, strict=False)),
            0.0,
            10,
        )

    def test_numerical_fit_is_a_local_minimum(self) -> None:
        """Perturbing the fitted gamma parameters increases the misfit.

        References:
            [EQ-INF-ELICIT-001].

        """
        probs = (0.1, 0.25, 0.5, 0.75, 0.9)
        values = [1.0, 2.1, 3.0, 4.4, 5.6]
        fit = fit_from_quantiles("gamma", probs, values)
        self.assertEqual(fit.method, "least-squares")
        shape, rate = fit.parameter("shape"), fit.parameter("rate")

        def cost(a: float, b: float) -> float:
            """Return the weighted squared quantile misfit of ``Gamma(a, b)``.

            References:
                [EQ-INF-ELICIT-001].

            """
            law = Gamma(a, b)
            return sum(
                (law.quantile(p) - v) ** 2 for p, v in zip(probs, values, strict=False)
            )

        base = cost(shape, rate)
        for da, db in ((1.01, 1.0), (0.99, 1.0), (1.0, 1.01), (1.0, 0.99)):
            self.assertGreater(cost(shape * da, rate * db), base)

    def test_feedback_quantiles_and_residuals(self) -> None:
        """Feedback is the fitted law's quantiles; residuals are fitted minus elicited.

        References:
            [EQ-INF-ELICIT-001].

        """
        fit = fit_from_quantiles(
            "normal",
            (0.1, 0.5, 0.9),
            [0.0, 1.0, 3.0],
            feedback_probabilities=(0.01, 0.99),
        )
        self.assertEqual([p for p, _ in fit.feedback], [0.01, 0.99])
        for p, value in fit.feedback:
            self.assertAlmostEqual(value, fit.law.quantile(p), 12)
        for fitted, elicited, residual in zip(
            fit.fitted_values, fit.values, fit.residuals, strict=False
        ):
            self.assertAlmostEqual(residual, fitted - elicited, 12)
        default = fit_from_quantiles("normal", (0.1, 0.9), [0.0, 1.0])
        self.assertEqual(
            [p for p, _ in default.feedback], [0.01, 0.05, 0.5, 0.95, 0.99]
        )

    def test_validation(self) -> None:
        """Bad families, orderings, bounds and weights are rejected.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        with self.assertRaises(ValueError):
            fit_from_quantiles("cauchy", P3, [1.0, 2.0, 3.0])
        with self.assertRaises(TypeError):
            fit_from_quantiles(3, P3, [1.0, 2.0, 3.0])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fit_from_quantiles("normal", P3, [1.0, 3.0, 2.0])
        with self.assertRaises(ValueError):
            fit_from_quantiles("normal", (0.0, 0.5), [1.0, 2.0])
        with self.assertRaises(ValueError):
            fit_from_quantiles("normal", (0.5, 0.4), [1.0, 2.0])
        with self.assertRaises(ValueError):
            fit_from_quantiles("normal", P3, [1.0, 2.0])
        with self.assertRaises(ValueError):
            fit_from_quantiles("normal", (0.5,), [1.0])
        with self.assertRaises(TypeError):
            fit_from_quantiles("normal", "ab", [1.0, 2.0])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fit_from_quantiles("normal", P3, [1.0, 2.0, 3.0], weights=[1.0, 0.0, 1.0])
        with self.assertRaises(ValueError):
            fit_from_quantiles("normal", P3, [1.0, 2.0, 3.0], bounds=(0.0, 5.0))
        with self.assertRaises(ValueError):
            fit_from_quantiles("beta", P3, [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            fit_from_quantiles("beta", P3, [1.0, 2.0, 3.0], bounds=(2.0, 5.0))
        with self.assertRaises(ValueError):
            fit_from_quantiles("lognormal", P3, [-1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            fit_from_quantiles("triangular", (0.05, 0.95), [1.0, 2.0])
        with self.assertRaises(ValueError):
            fit_from_quantiles("normal", P3, [1.0, 2.0, 3.0], degrees_of_freedom=-1.0)

    def test_compare_ranks_by_misfit_and_keeps_every_feedback(self) -> None:
        """The comparison sorts by misfit, lists skipped families and keeps feedback.

        References:
            [EQ-INF-ELICIT-001].

        """
        probs = (0.05, 0.25, 0.5, 0.75, 0.95)
        comparison = compare_elicited_fits(
            probs,
            _quantiles(Gamma(4.0, 1.0), probs),
            families=("normal", "gamma", "beta"),
        )
        self.assertEqual(comparison.ranking[0], "gamma")
        self.assertEqual({name for name, _ in comparison.skipped}, {"beta"})
        misfits = [fit.misfit for fit in comparison.fits]
        self.assertEqual(misfits, sorted(misfits))
        self.assertLess(comparison.lowest_misfit.misfit, 1e-6)
        self.assertGreater(comparison.fits[-1].misfit, 1e-3)
        self.assertTrue(all(len(fit.feedback) == 5 for fit in comparison.fits))
        negative = compare_elicited_fits((0.1, 0.5, 0.9), [-2.0, 0.0, 2.5])
        self.assertIn("lognormal", {name for name, _ in negative.skipped})
        self.assertIn("normal", negative.ranking)
        everything = compare_elicited_fits(
            (0.05, 0.5, 0.95),
            _quantiles(Affine(Beta(2.0, 3.0), 0.0, 10.0)),
            bounds=(0.0, 10.0),
        )
        self.assertIn("beta", everything.ranking)
        with self.assertRaises(ValueError):
            compare_elicited_fits(P3, [1.0, 2.0, 3.0], families=("beta",))
        with self.assertRaises(ValueError):
            compare_elicited_fits(P3, [1.0, 2.0, 3.0], families=())
        with self.assertRaises(ValueError):
            compare_elicited_fits(P3, [3.0, 2.0, 1.0])


class ThreePointTests(unittest.TestCase):
    """Three-point judgements from absolute bounds and from quantiles.

    References:
        [REF-MALCOLM-1959], [REF-VOSE-2008]. Checks [EQ-INF-ELICIT-002].

    """

    def test_absolute_three_points(self) -> None:
        """PERT mean is ``(low + 4 mode + high)/6``, triangular ``(a + c + b)/3``.

        References:
            [REF-VOSE-2008], [EQ-DIST-PERT-001], [EQ-DIST-TRI-001].

        """
        pert = pert_from_three_points(10.0, 20.0, 40.0)
        self.assertAlmostEqual(pert.mean, (10.0 + 80.0 + 40.0) / 6.0, 12)
        wider = pert_from_three_points(10.0, 20.0, 40.0, shape=2.0)
        self.assertGreater(wider.variance, pert.variance)
        tri = triangular_from_three_points(10.0, 20.0, 40.0)
        self.assertAlmostEqual(tri.mean, 70.0 / 3.0, 12)
        for bad in ((5.0, 1.0, 9.0), (5.0, 5.0, 5.0), (5.0, 9.0, 8.0)):
            with self.assertRaises(ValueError):
                pert_from_three_points(*bad)
            with self.assertRaises(ValueError):
                triangular_from_three_points(*bad)
        with self.assertRaises(TypeError):
            pert_from_three_points("a", 1.0, 2.0)  # type: ignore[arg-type]

    def test_pert_bounds_from_quantiles_match_r(self) -> None:
        """5 % and 95 % quantiles of PERT(10, 20, 40) return the bounds 10 and 40.

        References:
            [EQ-INF-ELICIT-002]. R: 10 + 30 * qbeta(c(0.05, 0.95), 7/3, 11/3) ->
            13.284971759949833, 31.398621596396431; c(0.1, 0.9) ->
            14.590830810294378, 29.333700928617418.

        """
        fit = pert_from_quantiles(13.284971759949833, 20.0, 31.398621596396431)
        self.assertAlmostEqual(fit.low, 10.0, 8)
        self.assertAlmostEqual(fit.high, 40.0, 8)
        self.assertAlmostEqual(fit.law.quantile(0.05), 13.284971759949833, 9)
        self.assertAlmostEqual(fit.law.quantile(0.95), 31.398621596396431, 9)
        inner = pert_from_quantiles(
            14.590830810294378, 20.0, 29.333700928617418, p_low=0.1, p_high=0.9
        )
        self.assertAlmostEqual(inner.low, 10.0, 8)
        self.assertAlmostEqual(inner.high, 40.0, 8)

    def test_triangular_bounds_from_quantiles(self) -> None:
        """Triangular bounds are recovered from closed-form quantiles.

        The quantile function ``a + sqrt(p (b - a)(c - a))`` for ``p < F(c)``
        and ``b - sqrt((1 - p)(b - a)(b - c))`` otherwise is evaluated here
        independently of the library.

        References:
            [EQ-INF-ELICIT-002], [EQ-DIST-TRI-001].

        """
        a, c, b = 2.0, 3.0, 11.0

        def quantile(p: float) -> float:
            """Return the triangular quantile from its closed form.

            References:
                [EQ-DIST-TRI-001].

            """
            if p < (c - a) / (b - a):
                return a + math.sqrt(p * (b - a) * (c - a))
            return b - math.sqrt((1.0 - p) * (b - a) * (b - c))

        fit = triangular_from_quantiles(quantile(0.05), c, quantile(0.95))
        self.assertAlmostEqual(fit.low, a, 9)
        self.assertAlmostEqual(fit.high, b, 9)
        self.assertIsInstance(fit.law, Triangular)
        self.assertEqual(fit.family, "triangular")
        # A right-skewed law with its mode at the lower end: the mode sits
        # below the stated 5 % quantile (11 - 9 sqrt(1 - p) for c = a).
        low_q, high_q = 11.0 - 9.0 * math.sqrt(0.95), 11.0 - 9.0 * math.sqrt(0.05)
        skewed = triangular_from_quantiles(low_q, 2.0, high_q)
        self.assertAlmostEqual(skewed.low, 2.0, 8)
        self.assertAlmostEqual(skewed.high, 11.0, 8)
        self.assertLess(skewed.mode, skewed.quantile_low)

    def test_no_solution_is_reported(self) -> None:
        """A mode far outside the quantile interval, or bad ordering, is rejected.

        References:
            [EQ-INF-ELICIT-002].

        """
        with self.assertRaises(ValueError) as caught:
            pert_from_quantiles(10.0, 100.0, 12.0)
        self.assertIn("no pert law", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            triangular_from_quantiles(10.0, -50.0, 12.0)
        self.assertIn("mode must lie in", str(caught.exception))
        with self.assertRaises(ValueError):
            pert_from_quantiles(5.0, 4.0, 3.0)
        with self.assertRaises(ValueError):
            pert_from_quantiles(1.0, 2.0, 3.0, p_low=0.9, p_high=0.1)
        with self.assertRaises(ValueError):
            pert_from_quantiles(1.0, 2.0, 3.0, shape=-1.0)
        with self.assertRaises(TypeError):
            triangular_from_quantiles(1.0, "x", 3.0)  # type: ignore[arg-type]

    def test_ratio_is_monotone_so_the_solution_is_unique(self) -> None:
        """Round trips over a grid of mode fractions recover the bounds.

        References:
            [EQ-INF-ELICIT-002].

        """
        for mode in (0.0, 0.1, 0.5, 0.9, 1.0):
            law = PERT(0.0, mode, 1.0)
            fit = pert_from_quantiles(law.quantile(0.05), mode, law.quantile(0.95))
            self.assertAlmostEqual(fit.low, 0.0, 8)
            self.assertAlmostEqual(fit.high, 1.0, 8)


class OpinionPoolTests(unittest.TestCase):
    """Linear, logarithmic and quantile-averaged pools.

    References:
        [REF-GENEST-ZIDEK-1986], [REF-CLEMEN-WINKLER-1999]. Checks
        [EQ-INF-ELICIT-003] and [EQ-INF-ELICIT-004].

    """

    def test_linear_pool_moments(self) -> None:
        """The linear pool is the mixture with its mean and variance formulas.

        References:
            [EQ-INF-ELICIT-003].

        """
        laws = [Normal(0.0, 1.0), Normal(3.0, 2.0), Gamma(2.0, 1.0)]
        weights = [2.0, 1.0, 1.0]
        pool = linear_opinion_pool(laws, weights)
        self.assertIsInstance(pool, Mixture)
        w = [0.5, 0.25, 0.25]
        mean = sum(wi * law.mean for wi, law in zip(w, laws, strict=False))
        variance = sum(
            wi * (law.variance + (law.mean - mean) ** 2)
            for wi, law in zip(w, laws, strict=False)
        )
        self.assertAlmostEqual(pool.mean, mean, 10)
        self.assertAlmostEqual(pool.variance, variance, 10)
        self.assertAlmostEqual(
            pool.cdf(1.0),
            sum(wi * law.cdf(1.0) for wi, law in zip(w, laws, strict=False)),
            12,
        )
        equal = linear_opinion_pool(laws)
        self.assertEqual(equal.weights, (1 / 3, 1 / 3, 1 / 3))

    def test_log_pool_of_normals_is_precision_weighted(self) -> None:
        """Normal experts pool to the precision-weighted normal, in closed form.

        References:
            [EQ-INF-ELICIT-003].

        """
        laws = [Normal(1.0, 2.0), Normal(4.0, 1.0)]
        pool = logarithmic_opinion_pool(laws, [1.0, 3.0])
        w = (0.25, 0.75)
        precision = w[0] / 4.0 + w[1] / 1.0
        self.assertIsInstance(pool, Normal)
        self.assertAlmostEqual(pool.sigma, math.sqrt(1.0 / precision), 12)
        self.assertAlmostEqual(pool.mu, (w[0] * 1.0 / 4.0 + w[1] * 4.0) / precision, 12)

    def test_numerical_log_pool_matches_the_closed_form(self) -> None:
        """The numerical path reproduces the normal closed form.

        Wrapping each normal in :class:`Affine` hides it from the closed-form
        branch, so the quadrature path is checked against the formula.

        References:
            [EQ-INF-ELICIT-003].

        """
        wrapped = [
            Affine(Normal(0.0, 1.0), 1.0, 2.0),
            Affine(Normal(0.0, 1.0), 4.0, 1.0),
        ]
        pool = logarithmic_opinion_pool(wrapped, [1.0, 3.0])
        exact = logarithmic_opinion_pool(
            [Normal(1.0, 2.0), Normal(4.0, 1.0)], [1.0, 3.0]
        )
        self.assertIsInstance(pool, LogarithmicPool)
        for x in (0.0, 2.5, 3.4, 5.0):
            self.assertAlmostEqual(pool.pdf(x), exact.pdf(x), 8)
            self.assertAlmostEqual(pool.cdf(x), exact.cdf(x), 8)
        self.assertAlmostEqual(pool.mean, exact.mu, 8)
        self.assertAlmostEqual(pool.variance, exact.sigma**2, 8)
        for p in (0.05, 0.5, 0.99):
            self.assertAlmostEqual(pool.quantile(p), exact.quantile(p), 8)
        self.assertEqual(pool.quantile(0.0), pool.lower)
        self.assertEqual(pool.quantile(1.0), pool.upper)
        self.assertAlmostEqual(pool.sf(3.0), 1.0 - pool.cdf(3.0), 14)

    def test_log_pool_matches_brute_force_normalisation(self) -> None:
        """A gamma and a lognormal expert pool like a midpoint-rule normalisation.

        References:
            [EQ-INF-ELICIT-003].

        """
        first, second = Gamma(3.0, 1.0), LogNormal(1.0, 0.5)
        weights = (0.35, 0.65)
        pool = logarithmic_opinion_pool([first, second], weights)
        lower, upper, count = 1e-6, 60.0, 40_000
        step = (upper - lower) / count
        xs = [lower + (k + 0.5) * step for k in range(count)]
        g = [first.pdf(x) ** weights[0] * second.pdf(x) ** weights[1] for x in xs]
        z = sum(g) * step
        mean = sum(x * v for x, v in zip(xs, g, strict=False)) * step / z
        cut = int((2.5 - lower) / step)  # a cell edge near 2.5, no half cell
        edge = lower + cut * step
        below = sum(g[:cut]) * step / z
        self.assertAlmostEqual(pool.mean, mean, 5)
        self.assertAlmostEqual(pool.cdf(edge), below, 5)
        self.assertAlmostEqual(math.exp(pool.log_normaliser), z, 5)
        self.assertAlmostEqual(pool.pdf(2.5), g[int((2.5 - lower) / step)] / z, 4)
        self.assertEqual(pool.pdf(-1.0), 0.0)

    def test_log_pool_is_externally_bayesian_and_the_linear_pool_is_not(self) -> None:
        """Updating then pooling equals pooling then updating only for the log pool.

        With a normal likelihood the posterior of ``N(m, s^2)`` has precision
        ``1/s^2 + 1/t^2``; the log pool of the posteriors with weights summing
        to one equals the posterior of the log pool. The posterior of a linear
        pool reweights the components by their evidence, which differs from
        the original weights.

        References:
            [EQ-INF-ELICIT-003]. [REF-GENEST-ZIDEK-1986].

        """
        priors = [Normal(0.0, 1.0), Normal(3.0, 1.5)]
        weights = (0.4, 0.6)
        y, tau = 2.0, 0.8

        def update(law: Normal) -> Normal:
            """Return the normal posterior for ``y ~ N(x, tau^2)``.

            References:
                [EQ-INF-ELICIT-003].

            """
            precision = 1.0 / law.sigma**2 + 1.0 / tau**2
            mean = (law.mu / law.sigma**2 + y / tau**2) / precision
            return Normal(mean, math.sqrt(1.0 / precision))

        updated_then_pooled = logarithmic_opinion_pool(
            [update(p) for p in priors], weights
        )
        pooled_then_updated = update(logarithmic_opinion_pool(priors, weights))
        self.assertAlmostEqual(updated_then_pooled.mu, pooled_then_updated.mu, 12)
        self.assertAlmostEqual(updated_then_pooled.sigma, pooled_then_updated.sigma, 12)
        evidence = [
            Normal(p.mu, math.sqrt(p.sigma**2 + tau**2)).pdf(y) * w
            for p, w in zip(priors, weights, strict=False)
        ]
        reweighted = [e / sum(evidence) for e in evidence]
        self.assertGreater(abs(reweighted[0] - weights[0]), 0.01)

    def test_log_pool_rejects_disjoint_experts(self) -> None:
        """Experts with disjoint effective ranges have no meaningful log pool.

        References:
            [EQ-INF-ELICIT-003].

        """
        with self.assertRaises(ValueError):
            logarithmic_opinion_pool([Uniform(0.0, 1.0), Uniform(2.0, 3.0)])
        with self.assertRaises(ValueError):
            logarithmic_opinion_pool([Gamma(2.0, 1.0), LogNormal(0.0, 1.0)], tail=0.7)
        with self.assertRaises(ValueError):
            logarithmic_opinion_pool(
                [Gamma(2.0, 1.0), LogNormal(0.0, 1.0)], panels_per_law=1
            )

    def test_quantile_average_of_normals_is_normal(self) -> None:
        """Averaging normal quantile functions averages locations and scales.

        References:
            [EQ-INF-ELICIT-004].

        """
        laws = [Normal(0.0, 1.0), Normal(4.0, 3.0), Normal(-2.0, 0.5)]
        w = [0.5, 0.3, 0.2]
        pool = quantile_average(laws, w)
        self.assertIsInstance(pool, Normal)
        self.assertAlmostEqual(
            pool.mu, sum(a * law.mu for a, law in zip(w, laws, strict=False)), 12
        )
        self.assertAlmostEqual(
            pool.sigma, sum(a * law.sigma for a, law in zip(w, laws, strict=False)), 12
        )
        for p in (0.01, 0.3, 0.9):
            self.assertAlmostEqual(
                pool.quantile(p),
                sum(a * law.quantile(p) for a, law in zip(w, laws, strict=False)),
                12,
            )

    def test_vincentized_law_of_mixed_families(self) -> None:
        """The averaged quantile function defines a consistent law.

        References:
            [EQ-INF-ELICIT-004].

        """
        laws = [Gamma(3.0, 1.0), LogNormal(1.0, 0.4)]
        pool = quantile_average(laws, [1.0, 1.0])
        self.assertIsInstance(pool, VincentizedLaw)
        for p in (0.05, 0.5, 0.95):
            self.assertAlmostEqual(
                pool.quantile(p), 0.5 * (laws[0].quantile(p) + laws[1].quantile(p)), 12
            )
            self.assertAlmostEqual(pool.cdf(pool.quantile(p)), p, 9)
        self.assertAlmostEqual(pool.mean, 0.5 * (laws[0].mean + laws[1].mean), 12)
        x, h = 3.2, 1e-5
        slope = (pool.cdf(x + h) - pool.cdf(x - h)) / (2.0 * h)
        self.assertAlmostEqual(pool.pdf(x), slope, 7)
        # Two lognormals are comonotone under averaging: E[X1 X2] =
        # exp(1.5 + 1.2^2 / 2), so Var = (v1 + v2 + 2 cov) / 4 in closed form.
        first, second = LogNormal(1.0, 0.4), LogNormal(0.5, 0.8)
        both = quantile_average([first, second])
        cov = math.exp(1.5 + 0.72) - first.mean * second.mean
        expected = 0.25 * (first.variance + second.variance + 2.0 * cov)
        self.assertAlmostEqual(both.variance, expected, 6)
        self.assertAlmostEqual(pool.sf(3.0), 1.0 - pool.cdf(3.0), 14)
        self.assertEqual(pool.cdf(-5.0), 0.0)
        self.assertEqual(pool.pdf(-5.0), 0.0)
        with self.assertRaises(ValueError):
            pool.quantile(1.0)

    def test_pool_argument_validation(self) -> None:
        """Empty inputs, bad weights and non-laws are rejected by every pool.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        for pool in (linear_opinion_pool, logarithmic_opinion_pool, quantile_average):
            with self.assertRaises(ValueError):
                pool([])
            with self.assertRaises(ValueError):
                pool([Normal(0.0, 1.0), Normal(1.0, 1.0)], [1.0])
            with self.assertRaises(ValueError):
                pool([Normal(0.0, 1.0), Normal(1.0, 1.0)], [1.0, -1.0])
            with self.assertRaises(TypeError):
                pool([Normal(0.0, 1.0), object()])


def _rows(bins: list[int], scale: float) -> tuple[list[float], list[list[float]]]:
    """Return realizations ``0..N-1`` and quantile rows that place each in its bin.

    Bin 0 is ``r <= q05``, bin 1 ``q05 < r <= q50``, bin 2 ``q50 < r <= q95``,
    bin 3 ``r > q95``; ``scale`` sets the width of the stated intervals.

    References:
        [EQ-INF-ELICIT-005]. No new equation.

    """
    offsets = {
        0: (1.0, 5.0, 9.0),
        1: (-3.0, 1.0, 5.0),
        2: (-5.0, -1.0, 3.0),
        3: (-9.0, -5.0, -1.0),
    }
    truth = [float(i) for i in range(len(bins))]
    rows = [
        [r + scale * o for o in offsets[b]] for r, b in zip(truth, bins, strict=False)
    ]
    return truth, rows


class CookeScoreTests(unittest.TestCase):
    """Calibration, information, weights and the classical model.

    References:
        [REF-COOKE-1991]. Checks [EQ-INF-ELICIT-005], [EQ-INF-ELICIT-006]
        and [EQ-INF-ELICIT-007].

    """

    def test_bin_probabilities(self) -> None:
        """Bin probabilities are the differences of the stated probabilities.

        References:
            [EQ-INF-ELICIT-005].

        """
        for got, want in zip(
            cooke_bin_probabilities(), (0.05, 0.45, 0.45, 0.05), strict=False
        ):
            self.assertAlmostEqual(got, want, 15)
        self.assertAlmostEqual(sum(cooke_bin_probabilities((0.1, 0.4, 0.9))), 1.0, 15)

    def test_calibration_hand_computed_with_r_pins(self) -> None:
        """Calibration of tiny examples against hand statistics and ``pchisq``.

        References:
            [EQ-INF-ELICIT-005]. R: pchisq(8 * log(1/0.9), 3, lower.tail = FALSE)
            -> 0.83918457542661229; pchisq(8 * log(10), 3, lower.tail = FALSE) ->
            0.00036015914758921985; pchisq(40 * I, 3, lower.tail = FALSE) with
            s = (5, 5, 5, 5)/20 -> 0.00013804105275395107 (statistic
            20.433024950639627) and s = (0, 10, 10, 0)/20 ->
            0.23922206710966773.

        """
        rows = [[1.0, 2.0, 3.0]] * 4
        # Two realizations in each middle bin: s = (0, .5, .5, 0).
        middle = cooke_calibration(rows, [1.5, 1.5, 2.5, 2.5])
        self.assertEqual(middle.bin_counts, (0, 2, 2, 0))
        self.assertAlmostEqual(middle.statistic, 8.0 * math.log(1.0 / 0.9), 12)
        self.assertAlmostEqual(middle.score, 0.83918457542661229, 12)
        # Overconfident: two in each tail bin, s = (.5, 0, 0, .5).
        tails = cooke_calibration(rows, [0.0, 0.5, 3.5, 4.0])
        self.assertEqual(tails.bin_counts, (2, 0, 0, 2))
        self.assertAlmostEqual(tails.statistic, 8.0 * math.log(10.0), 12)
        self.assertAlmostEqual(tails.score, 0.00036015914758921985, 12)
        self.assertEqual(tails.degrees_of_freedom, 3)
        self.assertEqual(tails.n_realizations, 4)
        spread = _rows([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5, 1.0)
        calibration = cooke_calibration(spread[1], spread[0])
        self.assertEqual(calibration.bin_counts, (5, 5, 5, 5))
        self.assertAlmostEqual(calibration.statistic, 20.433024950639627, 11)
        self.assertAlmostEqual(calibration.score, 0.00013804105275395107, 14)
        centre = _rows([1] * 10 + [2] * 10, 1.0)
        self.assertAlmostEqual(
            cooke_calibration(centre[1], centre[0]).score, 0.23922206710966773, 12
        )

    def test_perfectly_calibrated_expert_scores_one(self) -> None:
        """Bin counts equal to ``N p`` give statistic zero and calibration one.

        References:
            [EQ-INF-ELICIT-005].

        """
        truth, rows = _rows([0] + [1] * 9 + [2] * 9 + [3], 1.0)
        result = cooke_calibration(rows, truth)
        self.assertEqual(result.bin_counts, (1, 9, 9, 1))
        self.assertAlmostEqual(result.statistic, 0.0, 12)
        self.assertAlmostEqual(result.score, 1.0, 12)
        self.assertEqual(result.empirical_probabilities, (0.05, 0.45, 0.45, 0.05))

    def test_published_calibration_examples(self) -> None:
        """Calibration of ten seed items against the worked values of a chapter.

        Hanea and Nane (2021), Section 3.2.1, state that with ten seed
        variables the best attainable calibration is 0.83 at empirical
        frequencies ``(0.1, 0.4, 0.4, 0.1)`` and that an expert with
        ``(0.1, 0.6, 0.2, 0.1)`` obtains 0.39. Both are rounded in the text,
        so the pins use a rounding tolerance. The chapter also states that
        twenty questions are the fewest that can reach a calibration of one.

        References:
            [REF-HANEA-NANE-2021, Section 3.2.1]. Checks [EQ-INF-ELICIT-005].

        """
        row = [10.0, 20.0, 30.0]
        values = {0: 5.0, 1: 15.0, 2: 25.0, 3: 40.0}

        def calibration(counts: tuple[int, ...]) -> float:
            """Return the calibration of realizations placed in the bins.

            References:
                [EQ-INF-ELICIT-005].

            """
            truth = [values[b] for b, c in enumerate(counts) for _ in range(c)]
            return cooke_calibration([row] * len(truth), truth).score

        self.assertAlmostEqual(calibration((1, 4, 4, 1)), 0.83, delta=0.005)
        self.assertAlmostEqual(calibration((1, 6, 2, 1)), 0.39, delta=0.005)
        self.assertAlmostEqual(calibration((1, 9, 9, 1)), 1.0, delta=1e-12)
        self.assertLess(calibration((1, 4, 4, 1)), 1.0)

    def test_published_information_examples(self) -> None:
        """Information scores of four percentage assessments on ``[0, 100]``.

        Hanea and Nane (2021), Table 2 and Section 3.2.2, list the assessments
        ``(5, 15, 25)``, ``(40, 50, 60)``, ``(15, 17, 75)`` and ``(30, 50, 70)``
        at the 5th, 50th and 95th percentiles with information scores 1.21,
        1.14, 1.15 and 0.55 for the intrinsic range ``[0, 100]``. The text
        rounds the second score down (the formula gives 1.1457), so the pins
        allow one unit in the second decimal.

        References:
            [REF-HANEA-NANE-2021, Table 2]. Checks [EQ-INF-ELICIT-006].

        """
        assessments = (
            (5.0, 15.0, 25.0),
            (40.0, 50.0, 60.0),
            (15.0, 17.0, 75.0),
            (30.0, 50.0, 70.0),
        )
        published = (1.21, 1.14, 1.15, 0.55)
        for row, want in zip(assessments, published, strict=True):
            got = cooke_information([row], [(0.0, 100.0)]).score
            self.assertAlmostEqual(got, want, delta=0.0105)
        # The chapter's closed form: sum p ln(p / width share) = sum p ln p'
        # + ln(U - L) with widths measured in the units of the range.
        row = assessments[0]
        edges = (0.0, *row, 100.0)
        closed = math.fsum(
            p * math.log(p * 100.0 / (b - a))
            for p, a, b in zip(
                (0.05, 0.45, 0.45, 0.05), edges[:-1], edges[1:], strict=True
            )
        )
        self.assertAlmostEqual(
            cooke_information([row], [(0.0, 100.0)]).score, closed, 13
        )

    def test_calibration_with_five_quantiles(self) -> None:
        """Five stated quantiles give six bins and five degrees of freedom.

        References:
            [EQ-INF-ELICIT-005]. R: pchisq(20 * I, 5, lower.tail = FALSE) with
            s = (2, 1, 3, 2, 1, 1)/10 and p = (.1, .15, .25, .25, .15, .1) ->
            0.92948219347307615 (statistic 1.3520834253140124).

        """
        probs = (0.1, 0.25, 0.5, 0.75, 0.9)
        row = [1.0, 2.0, 3.0, 4.0, 5.0]
        realizations = [0.5, 0.7, 1.5, 2.5, 2.6, 2.7, 3.5, 3.6, 4.5, 6.0]
        result = cooke_calibration([row] * 10, realizations, probabilities=probs)
        self.assertEqual(result.bin_counts, (2, 1, 3, 2, 1, 1))
        self.assertEqual(result.degrees_of_freedom, 5)
        self.assertAlmostEqual(result.statistic, 1.3520834253140124, 11)
        self.assertAlmostEqual(result.score, 0.92948219347307615, 12)

    def test_information_hand_computed_uniform_background(self) -> None:
        """One item with quantiles (2, 4, 6) and realization 3 on range (1.6, 6.4).

        The overshoot of 10 % of the width 4 gives the range ``[1.6, 6.4]``;
        the bin widths are 0.4, 2, 2, 0.4 of 4.8.

        References:
            [EQ-INF-ELICIT-006].

        """
        ranges = cooke_intrinsic_ranges([[[2.0, 4.0, 6.0]]], [3.0])
        self.assertAlmostEqual(ranges[0][0], 1.6, 12)
        self.assertAlmostEqual(ranges[0][1], 6.4, 12)
        expected = 2 * 0.05 * math.log(0.05 * 4.8 / 0.4) + 2 * 0.45 * math.log(
            0.45 * 4.8 / 2.0
        )
        result = cooke_information([[2.0, 4.0, 6.0]], ranges)
        self.assertAlmostEqual(result.score, expected, 12)
        self.assertAlmostEqual(result.by_item[0], expected, 12)
        # Narrower statements carry more information.
        narrow = cooke_information([[3.0, 4.0, 5.0]], ranges)
        self.assertGreater(narrow.score, result.score)

    def test_information_log_uniform_background(self) -> None:
        """The log-uniform background measures bins by ``ln(b/a)/ln(U/L)``.

        References:
            [EQ-INF-ELICIT-006].

        """
        bounds = [(1.0, 100.0)]
        row = [[2.0, 10.0, 50.0]]
        edges = (1.0, 2.0, 10.0, 50.0, 100.0)
        p = (0.05, 0.45, 0.45, 0.05)
        expected = sum(
            pj * math.log(pj / (math.log(b / a) / math.log(100.0)))
            for pj, a, b in zip(p, edges, edges[1:], strict=False)
        )
        result = cooke_information(row, bounds, background="log-uniform")
        self.assertAlmostEqual(result.score, expected, 12)
        ranges = cooke_intrinsic_ranges(
            [row], [5.0], overshoot=0.0, background="log-uniform"
        )
        self.assertAlmostEqual(ranges[0][0], 2.0, 12)
        self.assertAlmostEqual(ranges[0][1], 50.0, 12)
        with self.assertRaises(ValueError):
            cooke_intrinsic_ranges([[[-1.0, 1.0, 2.0]]], background="log-uniform")

    def test_weights_with_cutoff(self) -> None:
        """Weights are ``C I`` above the cut-off, zero below, normalised.

        References:
            [EQ-INF-ELICIT-007].

        """
        weights = cooke_weights([0.5, 0.01, 0.3], [2.0, 5.0, 1.0], 0.05)
        self.assertAlmostEqual(weights[0], 1.0 / 1.3, 12)
        self.assertEqual(weights[1], 0.0)
        self.assertAlmostEqual(weights[2], 0.3 / 1.3, 12)
        with self.assertRaises(ValueError):
            cooke_weights([0.1, 0.2], [1.0, 1.0], 0.5)
        with self.assertRaises(ValueError):
            cooke_weights([0.1], [1.0, 1.0], 0.0)
        with self.assertRaises(ValueError):
            cooke_weights([-0.1], [1.0], 0.0)

    def test_classical_model_prefers_the_calibrated_expert(self) -> None:
        """A calibrated expert outweighs an overconfident one with higher information.

        Expert A is calibrated (bin counts 1, 9, 9, 1) with wide statements,
        expert B overconfident (5, 5, 5, 5) with narrow ones.

        References:
            [EQ-INF-ELICIT-005], [EQ-INF-ELICIT-006], [EQ-INF-ELICIT-007].

        """
        truth, rows_a = _rows([0] + [1] * 9 + [2] * 9 + [3], 1.0)
        _, rows_b = _rows([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5, 0.1)
        result = cooke_classical_model([rows_a, rows_b], truth)
        a, b = result.experts
        self.assertAlmostEqual(a.calibration.score, 1.0, 12)
        self.assertAlmostEqual(b.calibration.score, 0.00013804105275395107, 14)
        self.assertGreater(b.information, a.information)
        self.assertLess(b.score, a.score)
        self.assertAlmostEqual(sum(result.weights), 1.0, 12)
        self.assertGreater(result.weights[0], result.weights[1])
        self.assertTrue(result.optimized)
        self.assertEqual(len(result.cutoff_scan), 2)
        best = max(item.score.score for item in result.cutoff_scan)
        self.assertAlmostEqual(result.decision_maker.score, best, 14)
        self.assertIn(result.cutoff, {a.calibration.score, b.calibration.score})
        self.assertEqual(len(result.equal_weight_decision_maker.quantiles), len(truth))

    def test_fixed_cutoff_and_single_expert_decision_maker(self) -> None:
        """A cut-off above one expert leaves a decision maker equal to the other.

        With one expert left the pooled density is that expert's, so the
        decision maker's calibration and information equal the expert's.

        References:
            [EQ-INF-ELICIT-007].

        """
        truth, rows_a = _rows([0] + [1] * 9 + [2] * 9 + [3], 1.0)
        _, rows_b = _rows([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5, 0.1)
        result = cooke_classical_model([rows_a, rows_b], truth, cutoff=0.5)
        self.assertFalse(result.optimized)
        self.assertEqual(result.weights, (1.0, 0.0))
        self.assertEqual(result.cutoff_scan, ())
        self.assertAlmostEqual(
            result.decision_maker.information, result.experts[0].information, 12
        )
        self.assertAlmostEqual(result.decision_maker.calibration.score, 1.0, 12)
        self.assertEqual(result.experts[1].weight, 0.0)
        for got, row in zip(result.decision_maker.quantiles, rows_a, strict=False):
            for g, q in zip(got, row, strict=False):
                self.assertAlmostEqual(g, q, 10)
        with self.assertRaises(ValueError):
            cooke_classical_model([rows_a, rows_b], truth, cutoff=2.0)

    def test_decision_maker_cells_match_the_mixture_law(self) -> None:
        """The exact cell quantiles equal the mixture law's quantiles.

        References:
            [EQ-INF-ELICIT-007].

        """
        rows = [[2.0, 4.0, 6.0], [3.0, 5.0, 9.0]]
        weights = [0.3, 0.7]
        bounds = (1.0, 10.0)
        theory = cooke_bin_probabilities()
        for background, bounds_ in (("uniform", bounds), ("log-uniform", bounds)):
            quantiles, information = _decision_maker_item(
                weights,
                [tuple(r) for r in rows],
                bounds_,
                theory,
                P3,
                background,  # type: ignore[arg-type]
            )
            law = cooke_decision_maker_law(
                weights, rows, bounds_, background=background
            )
            for p, q in zip(P3, quantiles, strict=False):
                self.assertAlmostEqual(law.quantile(p), q, 8)
            self.assertGreater(information, 0.0)
        single, info = _decision_maker_item(
            [1.0], [(2.0, 4.0, 6.0)], bounds, theory, P3, "uniform"
        )
        self.assertAlmostEqual(single[1], 4.0, 12)
        direct = cooke_information([[2.0, 4.0, 6.0]], [bounds]).score
        self.assertAlmostEqual(info, direct, 12)
        uniform_law = cooke_decision_maker_law([1.0], [rows[0]], bounds)
        bins = [(1.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 10.0)]
        expected = sum(
            p * 0.5 * (a + b) for p, (a, b) in zip(theory, bins, strict=False)
        )
        self.assertAlmostEqual(uniform_law.mean, expected, 12)

    def test_cooke_validation(self) -> None:
        """Shape, ordering and range errors are rejected.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        row = [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError):
            cooke_calibration([row], [1.0, 2.0])
        with self.assertRaises(ValueError):
            cooke_calibration([[1.0, 1.0, 3.0]], [1.0])
        with self.assertRaises(ValueError):
            cooke_calibration([[1.0, 2.0]], [1.0])
        with self.assertRaises(ValueError):
            cooke_calibration([], [])
        with self.assertRaises(ValueError):
            cooke_information([row], [(2.0, 4.0)])
        with self.assertRaises(ValueError):
            cooke_information([row], [(1.0, 3.0)], background="gaussian")
        with self.assertRaises(ValueError):
            cooke_information([row, row], [(0.0, 4.0)])
        with self.assertRaises(ValueError):
            cooke_intrinsic_ranges([[row]], overshoot=-0.1)
        with self.assertRaises(ValueError):
            cooke_intrinsic_ranges([[row], [row, row]])
        with self.assertRaises(ValueError):
            cooke_classical_model([], [1.0])
        with self.assertRaises(ValueError):
            cooke_classical_model([[row]], [1.0, 2.0])
        with self.assertRaises(ValueError):
            cooke_decision_maker_law([0.0, 0.0], [row, row], (0.0, 4.0))
        with self.assertRaises(ValueError):
            cooke_decision_maker_law([1.0], [row], (2.0, 4.0))
        with self.assertRaises(TypeError):
            cooke_calibration([["a", "b", "c"]], [1.0])  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main()
