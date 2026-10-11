"""Tests for generic MLE, bootstrap-calibrated GOF, model selection, and joint fits.

Reference values are hand formulas recomputed inside each test: closed-form
MLEs, analytic observed/Fisher information, and the Gaussian-copula
Spearman relation. Stochastic checks use fixed seeds and coarse tolerances.
"""

from __future__ import annotations

import dataclasses
import math
import unittest
import warnings

from probstat.distributions import (
    Beta,
    Exponential,
    Gamma,
    GeneralizedExtremeValue,
    Gumbel,
    Logistic,
    LogNormal,
    Normal,
    StudentT,
    Uniform,
    Weibull,
)
from probstat.foundations import RandomStream
from probstat.inference import (
    BETA_FAMILY,
    EXPONENTIAL_FAMILY,
    GAMMA_FAMILY,
    GEV_FAMILY,
    GUMBEL_FAMILY,
    LOGISTIC_FAMILY,
    LOGNORMAL_FAMILY,
    NORMAL_FAMILY,
    STANDARD_FAMILIES,
    STUDENT_T_FAMILY,
    UNIFORM_FAMILY,
    WEIBULL_FAMILY,
    DistributionFamily,
    ModelSelectionWarning,
    akaike_weights,
    bootstrap_goodness_of_fit,
    family_by_name,
    fit_distribution,
    fit_gamma_mle,
    fit_lognormal_mle,
    fit_multivariate_normal_mle,
    fit_normal_mle,
    fit_weibull_mle,
    information_criteria,
    kolmogorov_smirnov_test,
    pp_points,
    qq_points,
    select_distribution,
)
from probstat.inference.fitting import STUDENT_T_MAX_NU
from probstat.modeling import (
    JointInput,
    RandomInput,
    fit_gaussian_copula,
    fit_joint_input,
    normal_scores,
    spearman_to_latent_correlation,
)
from probstat.modeling.fitting import _repair_correlation
from probstat.numerics import trigamma


def offset_start(family: DistributionFamily) -> DistributionFamily:
    """Return a copy of ``family`` whose optimizer starts far from the MLE.

    Locations move by one data spread and positive parameters are scaled by
    1.7, so a numerical fit has to travel to agree with the closed form.

    References:
        Test helper. No equation is evaluated.

    """
    base_initial = family.initial

    def initial(sample: tuple[float, ...]) -> tuple[float, ...]:
        """Perturb the family's own start values.

        References:
            Test helper. No equation is evaluated.

        """
        start = base_initial(sample)
        return tuple(
            value * 1.7 if kind == "positive" else value + 1.0
            for value, kind in zip(start, family.parameter_kinds, strict=True)
        )

    return dataclasses.replace(
        family, name=f"{family.name}-offset", initial=initial, closed_form=None
    )


class GenericMaximumLikelihoodTests(unittest.TestCase):
    """The optimizer reproduces closed forms and analytic information."""

    def test_numerical_fit_matches_closed_form_estimates(self) -> None:
        """Offset-start BFGS agrees with the closed-form/profile MLEs."""
        weibull_data = Weibull(1.8, 4.0).sample(250, seed=1)
        cases = (
            (NORMAL_FAMILY, Normal(2.0, 3.0).sample(250, seed=2), fit_normal_mle),
            (
                LOGNORMAL_FAMILY,
                LogNormal(0.5, 0.6).sample(250, seed=3),
                fit_lognormal_mle,
            ),
            (EXPONENTIAL_FAMILY, Exponential(0.7).sample(250, seed=4), None),
            (GAMMA_FAMILY, Gamma(2.5, 1.3).sample(250, seed=5), fit_gamma_mle),
            (WEIBULL_FAMILY, weibull_data, fit_weibull_mle),
        )
        for family, data, closed in cases:
            with self.subTest(family=family.name):
                numeric = fit_distribution(data, offset_start(family))
                exact = fit_distribution(data, family)
                self.assertEqual(exact.method, "closed-form")
                self.assertTrue(numeric.converged)
                self.assertIn(numeric.method, ("bfgs", "nelder-mead+bfgs"))
                for a, b in zip(numeric.estimates, exact.estimates, strict=True):
                    self.assertAlmostEqual(a, b, delta=2e-6 * max(1.0, abs(b)))
                self.assertAlmostEqual(
                    numeric.log_likelihood, exact.log_likelihood, delta=1e-8 * 250
                )
                if closed is not None:
                    law = closed(data)
                    reference = tuple(
                        getattr(law, name) for name in family.parameter_names
                    )
                    for a, b in zip(exact.estimates, reference, strict=True):
                        self.assertEqual(a, b)
        exponential = Exponential(0.7).sample(250, seed=4)
        rate = fit_distribution(exponential, EXPONENTIAL_FAMILY).estimates[0]
        self.assertAlmostEqual(rate, len(exponential) / math.fsum(exponential))

    def test_forced_numerical_method_matches_auto(self) -> None:
        """``method='numerical'`` reaches the profile-likelihood Gamma MLE."""
        data = Gamma(0.8, 2.0).sample(200, seed=8)
        auto = fit_distribution(data, GAMMA_FAMILY)
        numeric = fit_distribution(data, GAMMA_FAMILY, method="numerical")
        self.assertEqual(numeric.method, "bfgs")
        for a, b in zip(numeric.estimates, auto.estimates, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-6 * b)

    def test_standard_errors_match_analytic_information(self) -> None:
        """Hessian SEs equal rate/sqrt(n), sigma/sqrt(n), and Gamma Fisher info."""
        data = Exponential(2.0).sample(400, seed=11)
        fit = fit_distribution(data, EXPONENTIAL_FAMILY)
        rate = fit.estimates[0]
        self.assertAlmostEqual(
            fit.standard_errors[0], rate / math.sqrt(400), delta=1e-6 * rate
        )
        normal = Normal(5.0, 2.0).sample(300, seed=12)
        fitted = fit_distribution(normal, NORMAL_FAMILY)
        mu, sigma = fitted.estimates
        self.assertAlmostEqual(fitted.standard_errors[0], sigma / math.sqrt(300), 7)
        self.assertAlmostEqual(
            fitted.standard_errors[1], sigma / math.sqrt(600), 7
        )
        self.assertIsNotNone(fitted.covariance)
        gamma_data = Gamma(3.0, 0.5).sample(300, seed=13)
        gamma = fit_distribution(gamma_data, GAMMA_FAMILY)
        shape, lam = gamma.estimates
        n = 300
        a, b, c = n * trigamma(shape), -n / lam, n * shape / lam**2
        determinant = a * c - b * b
        self.assertAlmostEqual(
            gamma.standard_errors[0], math.sqrt(c / determinant), delta=1e-5 * shape
        )
        self.assertAlmostEqual(
            gamma.standard_errors[1], math.sqrt(a / determinant), delta=1e-5 * lam
        )
        self.assertEqual(gamma.params["shape"], shape)
        self.assertEqual(gamma.standard_error_map["rate"], gamma.standard_errors[1])
        self.assertEqual(gamma.parameter_count, 2)

    def test_numerical_families_recover_generating_parameters(self) -> None:
        """Gumbel, GEV, Logistic, Student t, and Beta fits land near the truth."""
        cases = (
            (GUMBEL_FAMILY, Gumbel(10.0, 2.0), (10.0, 2.0), 0.1),
            (GEV_FAMILY, GeneralizedExtremeValue(5.0, 1.5, 0.2), (5.0, 1.5, 0.2), 0.2),
            (LOGISTIC_FAMILY, Logistic(-3.0, 0.5), (-3.0, 0.5), 0.1),
            (STUDENT_T_FAMILY, StudentT(4.0, 1.0, 2.0), (4.0, 1.0, 2.0), 0.35),
            (BETA_FAMILY, Beta(2.0, 5.0), (2.0, 5.0), 0.2),
        )
        for family, law, truth, tolerance in cases:
            with self.subTest(family=family.name):
                data = law.sample(500, seed=21)
                fit = fit_distribution(data, family)
                self.assertTrue(fit.converged, fit.message)
                for estimate, value, error in zip(
                    fit.estimates, truth, fit.standard_errors, strict=True
                ):
                    self.assertLess(
                        abs(estimate - value), tolerance * max(1.0, abs(value))
                    )
                    self.assertTrue(0.0 < error < math.inf)
                for index in range(fit.parameter_count):
                    for step in (-1e-3, 1e-3):
                        moved = list(fit.estimates)
                        moved[index] += step * max(1.0, abs(moved[index]))
                        perturbed = family.build(tuple(moved))
                        loglik = math.fsum(perturbed.log_pdf(x) for x in data)
                        self.assertLessEqual(loglik, fit.log_likelihood + 1e-9)

    def test_student_t_degrees_of_freedom_are_bounded(self) -> None:
        """Light-tailed data push nu to its bound, whose SE is then undefined."""
        data = Uniform(-1.0, 1.0).sample(300, seed=31)
        fit = fit_distribution(data, STUDENT_T_FAMILY)
        self.assertGreater(fit.estimates[0], 0.99 * STUDENT_T_MAX_NU)
        self.assertLessEqual(fit.estimates[0], STUDENT_T_MAX_NU)
        self.assertTrue(math.isnan(fit.standard_errors[0]))
        self.assertTrue(math.isfinite(fit.standard_errors[2]))
        self.assertIn("bound", fit.message)

    def test_uniform_is_closed_form_and_non_regular(self) -> None:
        """Uniform MLE is the sample range with undefined information SEs."""
        data = Uniform(-1.0, 3.0).sample(100, seed=41)
        fit = fit_distribution(data, UNIFORM_FAMILY)
        self.assertEqual(fit.estimates, (min(data), max(data)))
        width = max(data) - min(data)
        self.assertAlmostEqual(fit.log_likelihood, -100 * math.log(width))
        self.assertTrue(all(math.isnan(value) for value in fit.standard_errors))
        self.assertIsNone(fit.covariance)
        with self.assertRaises(ValueError):
            fit_distribution(data, UNIFORM_FAMILY, method="numerical")

    def test_information_criteria_and_weights(self) -> None:
        """AIC, AICc, BIC, and Akaike weights follow their definitions."""
        aic, aicc, bic = information_criteria(-100.0, 2, 50)
        self.assertEqual(aic, 204.0)
        self.assertAlmostEqual(aicc, 204.0 + 12.0 / 47.0)
        self.assertAlmostEqual(bic, 2.0 * math.log(50) + 200.0)
        self.assertEqual(information_criteria(-1.0, 3, 4)[1], math.inf)
        fit = fit_distribution(Normal().sample(60, seed=3), NORMAL_FAMILY)
        self.assertAlmostEqual(fit.aic, 4.0 - 2.0 * fit.log_likelihood)
        self.assertAlmostEqual(fit.bic, 2.0 * math.log(60) - 2.0 * fit.log_likelihood)
        weights = akaike_weights((10.0, 12.0, math.inf))
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertAlmostEqual(weights[0] / weights[1], math.e)
        self.assertEqual(weights[2], 0.0)
        for bad in ((), (math.nan,), (math.inf,)):
            with self.assertRaises(ValueError):
                akaike_weights(bad)
        with self.assertRaises(ValueError):
            information_criteria(-1.0, 0, 10)
        with self.assertRaises(ValueError):
            information_criteria(math.nan, 1, 10)

    def test_invalid_fitting_inputs_raise(self) -> None:
        """Domain, degeneracy, method, and type errors are explicit."""
        with self.assertRaises(ValueError):
            fit_distribution((1.0, -2.0, 3.0), GAMMA_FAMILY)
        with self.assertRaises(ValueError):
            fit_distribution((2.0, 2.0, 2.0), NORMAL_FAMILY)
        with self.assertRaises(ValueError):
            fit_distribution((0.2, 1.5), BETA_FAMILY)
        with self.assertRaises(ValueError):
            fit_distribution((1.0, 2.0, 3.0), NORMAL_FAMILY, method="magic")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fit_distribution((1.0, 2.0, 3.0), GUMBEL_FAMILY, method="closed-form")
        with self.assertRaises(TypeError):
            fit_distribution((1.0, 2.0), "Normal")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fit_distribution((), NORMAL_FAMILY)
        with self.assertRaises(ValueError):
            fit_distribution((1.0, math.nan), NORMAL_FAMILY)

    def test_family_lookup_and_validation(self) -> None:
        """Ready-made families resolve by name and descriptions are validated."""
        self.assertEqual(len(STANDARD_FAMILIES), 11)
        self.assertIs(family_by_name("weibull"), WEIBULL_FAMILY)
        self.assertIs(family_by_name(" GEV "), GEV_FAMILY)
        with self.assertRaises(KeyError):
            family_by_name("Cauchy")
        with self.assertRaises(TypeError):
            family_by_name(3)  # type: ignore[arg-type]
        kwargs = {
            "name": "X",
            "parameter_names": ("a",),
            "parameter_kinds": ("positive",),
            "build": Exponential,
            "initial": lambda sample: (1.0,),
        }
        DistributionFamily(**kwargs)  # type: ignore[arg-type]
        for change in (
            {"name": " "},
            {"parameter_names": ()},
            {"parameter_names": ("a", "b")},
            {"parameter_kinds": ("bounded",)},
            {"domain": "complex"},
            {"upper_bounds": (None, 1.0)},
            {"upper_bounds": (-1.0,)},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                DistributionFamily(**{**kwargs, **change})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            DistributionFamily(
                "Y",
                ("m",),
                ("location",),
                Normal,
                lambda s: (0.0,),
                upper_bounds=(1.0,),
            )
        with self.assertRaises(TypeError):
            DistributionFamily(**{**kwargs, "build": 3})  # type: ignore[arg-type]


class BootstrapGoodnessOfFitTests(unittest.TestCase):
    """Parametric-bootstrap calibration with estimated parameters."""

    def test_bootstrap_ks_p_values_are_roughly_uniform_under_the_null(self) -> None:
        """Fitted-null bootstrap p-values centre near 1/2; naive KS ones do not."""
        root = RandomStream(11)
        bootstrap_p: list[float] = []
        naive_p: list[float] = []
        for replicate in range(30):
            data = Normal(3.0, 2.0).sample(25, seed=root.child("data", replicate).seed)
            result = bootstrap_goodness_of_fit(
                data, NORMAL_FAMILY, n_bootstrap=39, seed=root.child("b", replicate)
            )
            self.assertEqual(result.refits + result.failed_refits, 39)
            bootstrap_p.append(result.ks_p_value)
            naive_p.append(kolmogorov_smirnov_test(data, fit_normal_mle(data)).p_value)
        mean_p = sum(bootstrap_p) / len(bootstrap_p)
        self.assertGreater(mean_p, 0.35)
        self.assertLess(mean_p, 0.65)
        self.assertLessEqual(sum(p < 0.1 for p in bootstrap_p), 9)
        self.assertGreater(sum(naive_p) / len(naive_p), mean_p + 0.1)

    def test_wrong_family_gets_small_p_values(self) -> None:
        """Exponential data are rejected as Normal by both KS and AD."""
        data = Exponential(1.0).sample(100, seed=5)
        result = bootstrap_goodness_of_fit(data, NORMAL_FAMILY, n_bootstrap=99, seed=3)
        self.assertLessEqual(result.ks_p_value, 0.05)
        self.assertLessEqual(result.ad_p_value, 0.05)
        self.assertTrue(result.rejected(0.05))
        self.assertEqual(result.refits, 99)
        self.assertEqual(result.sample_size, 100)
        again = bootstrap_goodness_of_fit(data, NORMAL_FAMILY, n_bootstrap=99, seed=3)
        self.assertEqual(result, again)
        with self.assertRaises(ValueError):
            result.rejected(1.0)

    def test_uniform_ad_p_value_is_undefined(self) -> None:
        """Observations on a fitted support boundary make AD infinite."""
        data = Uniform(0.0, 1.0).sample(60, seed=7)
        result = bootstrap_goodness_of_fit(data, UNIFORM_FAMILY, n_bootstrap=19, seed=1)
        self.assertEqual(result.ad_statistic, math.inf)
        self.assertTrue(math.isnan(result.ad_p_value))
        self.assertTrue(0.0 < result.ks_p_value <= 1.0)

    def test_numerical_family_bootstrap_and_errors(self) -> None:
        """Numerical refits work; invalid settings and mismatched fits raise."""
        data = Gumbel(1.0, 2.0).sample(80, seed=9)
        result = bootstrap_goodness_of_fit(data, GUMBEL_FAMILY, n_bootstrap=9, seed=2)
        self.assertEqual(result.refits + result.failed_refits, 9)
        self.assertGreater(result.ks_p_value, 0.05)
        with self.assertRaises(ValueError):
            bootstrap_goodness_of_fit(data, GUMBEL_FAMILY, n_bootstrap=0)
        normal_fit = fit_distribution(data, NORMAL_FAMILY)
        with self.assertRaises(ValueError):
            bootstrap_goodness_of_fit(data, GUMBEL_FAMILY, fit=normal_fit)
        with self.assertRaises(TypeError):
            bootstrap_goodness_of_fit(data, GUMBEL_FAMILY, fit="fit")  # type: ignore[arg-type]

    def test_qq_and_pp_points(self) -> None:
        """Q-Q and P-P pairs use Hazen positions and track the fitted law."""
        data = Normal(1.0, 2.0).sample(200, seed=17)
        law = fit_normal_mle(data)
        qq = qq_points(data, law)
        pp = pp_points(data, law)
        self.assertEqual(len(qq), 200)
        self.assertEqual([pair[1] for pair in qq], sorted(data))
        self.assertAlmostEqual(qq[0][0], law.quantile(0.5 / 200))
        self.assertAlmostEqual(pp[-1][1], 199.5 / 200)
        self.assertTrue(all(0.0 <= first <= 1.0 for first, _ in pp))
        self.assertLess(max(abs(a - b) for a, b in pp), 0.1)
        with self.assertRaises(TypeError):
            qq_points(data, object())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            pp_points(data, object())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            qq_points((), law)


class ModelSelectionTests(unittest.TestCase):
    """Information-criterion ranking with calibrated GOF and warnings."""

    candidates = (WEIBULL_FAMILY, GAMMA_FAMILY, LOGNORMAL_FAMILY)

    def test_selection_picks_the_generating_family(self) -> None:
        """Seeded Weibull, Gamma, and LogNormal data select their own family."""
        cases = (
            (Weibull(3.0, 2.0), "Weibull"),
            (Gamma(2.0, 1.0), "Gamma"),
            (LogNormal(0.0, 0.8), "LogNormal"),
        )
        for law, expected in cases:
            with self.subTest(family=expected):
                data = law.sample(300, seed=100)
                result = select_distribution(data, self.candidates, n_bootstrap=0)
                self.assertEqual(result.best.fit.family, expected)
                self.assertEqual(result.best.rank, 1)
                self.assertEqual(result.best.delta_aic, 0.0)
                self.assertAlmostEqual(
                    sum(c.akaike_weight for c in result.candidates), 1.0
                )
                self.assertAlmostEqual(
                    sum(c.bic_weight for c in result.candidates), 1.0
                )
                aics = [c.fit.aic for c in result.candidates]
                self.assertEqual(aics, sorted(aics))
                self.assertIsNone(result.best.goodness_of_fit)
                self.assertEqual(result.warnings, ())

    def test_selection_with_bootstrap_gof(self) -> None:
        """The true family passes calibrated GOF; wrong ones are rejected."""
        data = Weibull(3.0, 2.0).sample(300, seed=100)
        result = select_distribution(
            data, ("Weibull", "Gamma", "LogNormal"), n_bootstrap=39, seed=1
        )
        best = result.best
        self.assertEqual(best.fit.family, "Weibull")
        assert best.goodness_of_fit is not None
        self.assertFalse(best.goodness_of_fit.rejected(0.05))
        lognormal = result.assessment("LogNormal").goodness_of_fit
        assert lognormal is not None
        self.assertTrue(lognormal.rejected(0.05))
        with self.assertRaises(KeyError):
            result.assessment("Beta")
        bic_result = select_distribution(
            data, self.candidates, criterion="bic", n_bootstrap=0
        )
        self.assertEqual(bic_result.criterion, "bic")
        self.assertEqual(bic_result.best.fit.family, "Weibull")

    def test_warning_when_best_model_is_rejected(self) -> None:
        """Exponential data with Normal/Uniform candidates trigger a warning."""
        data = Exponential(1.0).sample(200, seed=3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = select_distribution(
                data, (NORMAL_FAMILY, UNIFORM_FAMILY), n_bootstrap=39, seed=4
            )
        self.assertTrue(
            any(issubclass(item.category, ModelSelectionWarning) for item in caught)
        )
        self.assertTrue(any("rejected" in text for text in result.warnings))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            small = select_distribution(
                data, (NORMAL_FAMILY,), n_bootstrap=19, seed=4
            )
        self.assertTrue(any("cannot produce" in text for text in small.warnings))

    def test_failures_are_recorded_and_errors_raised(self) -> None:
        """Inadmissible candidates are listed; invalid settings raise."""
        data = Gamma(2.0, 0.5).sample(120, seed=6)
        result = select_distribution(data, ("Gamma", "Beta"), n_bootstrap=0)
        self.assertEqual([c.fit.family for c in result.candidates], ["Gamma"])
        self.assertEqual(result.failures[0][0], "Beta")
        with self.assertRaises(ValueError):
            select_distribution(data, ("Beta",), n_bootstrap=0)
        with self.assertRaises(ValueError):
            select_distribution(data, ("Gamma", "gamma"), n_bootstrap=0)
        with self.assertRaises(KeyError):
            select_distribution(data, ("Cauchy",), n_bootstrap=0)
        with self.assertRaises(ValueError):
            select_distribution(data, (), n_bootstrap=0)
        with self.assertRaises(TypeError):
            select_distribution(data, "Gamma", n_bootstrap=0)
        with self.assertRaises(TypeError):
            select_distribution(data, (3,), n_bootstrap=0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            select_distribution(data, criterion="hqc", n_bootstrap=0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            select_distribution(data, alpha=0.0, n_bootstrap=0)
        with self.assertRaises(ValueError):
            select_distribution((1.0, 1.0), n_bootstrap=0)

    def test_default_candidates_cover_admissible_families(self) -> None:
        """With no candidate list every admissible ready-made family is tried."""
        data = Weibull(2.0, 1.0).sample(150, seed=12)
        result = select_distribution(data, n_bootstrap=0)
        fitted = {c.fit.family for c in result.candidates}
        self.assertIn("Weibull", fitted)
        self.assertEqual(len(fitted) + len(result.failures), 11)
        self.assertIn("Beta", {name for name, _ in result.failures})


class MultivariateAndCopulaFittingTests(unittest.TestCase):
    """MVN MLE and Gaussian-copula joint-input fitting."""

    def test_multivariate_normal_mle(self) -> None:
        """MVN MLE equals the sample mean and divisor-n covariance."""
        rows = ((1.0, 2.0), (2.0, 1.0), (4.0, 5.0), (3.0, 3.0))
        law = fit_multivariate_normal_mle(rows)
        self.assertEqual(law.mean, (2.5, 2.75))
        self.assertAlmostEqual(law.covariance[0][0], 1.25)
        self.assertAlmostEqual(law.covariance[0][1], 5.5 / 4.0)
        with self.assertRaises(ValueError):
            fit_multivariate_normal_mle(((1.0, 2.0), (2.0, 3.0)))
        with self.assertRaises(ValueError):
            fit_multivariate_normal_mle(((1.0, 2.0), (2.0, 4.0), (3.0, 6.0)))
        with self.assertRaises(ValueError):
            fit_multivariate_normal_mle(((1.0, 2.0), (2.0,), (3.0, 6.0)))

    def test_spearman_relation_and_normal_scores(self) -> None:
        """The map 2 sin(pi r_s / 6) inverts r_s = (6/pi) asin(rho/2)."""
        for rho in (-0.9, -0.3, 0.0, 0.5, 0.95):
            r_s = 6.0 / math.pi * math.asin(rho / 2.0)
            self.assertAlmostEqual(spearman_to_latent_correlation(r_s), rho)
        self.assertAlmostEqual(spearman_to_latent_correlation(1.0), 1.0)
        with self.assertRaises(ValueError):
            spearman_to_latent_correlation(1.5)
        with self.assertRaises(TypeError):
            spearman_to_latent_correlation(True)
        scores = normal_scores((3.0, 1.0, 2.0))
        self.assertAlmostEqual(scores[2], 0.0)
        self.assertAlmostEqual(scores[0], -scores[1])
        with self.assertRaises(ValueError):
            normal_scores((1.0,))

    def test_copula_fit_recovers_latent_correlation(self) -> None:
        """Both rank estimators recover a known latent correlation matrix."""
        truth = JointInput(
            (
                RandomInput("load", Weibull(2.5, 10.0)),
                RandomInput("strength", LogNormal(3.0, 0.2)),
                RandomInput("wind", Gumbel(20.0, 4.0)),
            ),
            ((1.0, 0.6, -0.3), (0.6, 1.0, 0.2), (-0.3, 0.2, 1.0)),
        )
        rows = truth.sample(800, seed=2026)
        data = {name: [row[i] for row in rows] for i, name in enumerate(truth.names)}
        assert truth.correlation is not None
        for method in ("spearman", "normal-scores"):
            with self.subTest(method=method):
                fit = fit_joint_input(
                    data,
                    candidates=("Weibull", "LogNormal", "Gamma", "Gumbel"),
                    dependence=method,  # type: ignore[arg-type]
                )
                self.assertEqual(fit.names, ("load", "strength", "wind"))
                self.assertEqual(
                    [f.family for f in fit.marginal_fits],
                    ["Weibull", "LogNormal", "Gumbel"],
                )
                latent = fit.copula.latent_correlation
                for i in range(3):
                    for j in range(3):
                        error = latent[i][j] - truth.correlation[i][j]
                        self.assertLess(abs(error), 0.06)
                self.assertFalse(fit.copula.repaired)
                self.assertIsInstance(fit.joint_input, JointInput)
                self.assertEqual(fit.joint_input.correlation, latent)

    def test_specified_marginals_and_independence(self) -> None:
        """User families are honoured and independence gives no correlation."""
        data = {
            "a": Gamma(2.0, 1.0).sample(100, seed=1),
            "b": Normal(0.0, 1.0).sample(100, seed=2),
        }
        fit = fit_joint_input(
            data, marginals={"a": "Gamma", "b": NORMAL_FAMILY}, dependence="independent"
        )
        self.assertEqual([f.family for f in fit.marginal_fits], ["Gamma", "Normal"])
        self.assertEqual(fit.selections, (None, None))
        self.assertTrue(fit.joint_input.is_independent)
        copula = fit_gaussian_copula(data, method="independent")
        self.assertEqual(copula.latent_correlation, ((1.0, 0.0), (0.0, 1.0)))

    def test_correlation_repair_and_joint_errors(self) -> None:
        """Non-PD pairwise matrices are repaired; malformed data raise."""
        broken = ((1.0, 0.9, -0.9), (0.9, 1.0, 0.9), (-0.9, 0.9, 1.0))
        repaired, flag = _repair_correlation(broken)
        self.assertTrue(flag)
        self.assertTrue(all(abs(repaired[i][i] - 1.0) < 1e-12 for i in range(3)))
        unchanged, flag = _repair_correlation(((1.0, 0.5), (0.5, 1.0)))
        self.assertFalse(flag)
        self.assertEqual(unchanged, ((1.0, 0.5), (0.5, 1.0)))
        good = {"a": (1.0, 2.0, 3.0, 4.0), "b": (2.0, 1.0, 4.0, 3.0)}
        with self.assertRaises(ValueError):
            fit_gaussian_copula({"a": (1.0, 2.0, 3.0), "b": (1.0, 2.0)})
        with self.assertRaises(ValueError):
            fit_gaussian_copula({"a": (1.0, 2.0, 3.0), "b": (1.0, 1.0, 1.0)})
        with self.assertRaises(ValueError):
            fit_gaussian_copula(good, method="kendall")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fit_gaussian_copula({})
        with self.assertRaises(TypeError):
            fit_gaussian_copula([(1.0, 2.0, 3.0)])  # type: ignore[arg-type]
        with self.assertRaises(KeyError):
            fit_joint_input(good, marginals={"c": "Normal"})
        with self.assertRaises(TypeError):
            fit_joint_input(good, marginals={"a": 3})  # type: ignore[dict-item]


if __name__ == "__main__":
    unittest.main()
