"""Tests for the probabilistic-programming interface ``probstat.inference.ppl``."""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

from probstat.distributions import (
    Beta,
    Binomial,
    Exponential,
    Gamma,
    LogNormal,
    Normal,
    RealSupport,
    TruncatedNormal,
    Uniform,
)
from probstat.inference import (
    MCMCResult,
    PosteriorPredictivePValues,
    hierarchical_normal_posterior,
    ppl,
    split_rhat,
)
from probstat.inference.ppl import ModelHandle, ModelSpecificationError

SCHOOLS_Y = (28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0)
SCHOOLS_S = (15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0)
TAU_PRIOR = TruncatedNormal(0.0, 5.0, lower=0.0)

NORMAL_DATA = (4.1, 5.3, 3.8, 6.0, 4.9, 5.5, 4.4, 5.1, 6.2, 3.9)
NORMAL_SIGMA = 1.2
PRIOR_MEAN, PRIOR_SD = 0.0, 3.0
POSTERIOR_PRECISION = 1.0 / PRIOR_SD**2 + len(NORMAL_DATA) / NORMAL_SIGMA**2
POSTERIOR_MEAN = (
    PRIOR_MEAN / PRIOR_SD**2 + math.fsum(NORMAL_DATA) / NORMAL_SIGMA**2
) / POSTERIOR_PRECISION
POSTERIOR_SD = POSTERIOR_PRECISION**-0.5

BETA_A, BETA_B, TRIALS, SUCCESSES = 2.0, 3.0, 20, 13


def eight_schools(m: ModelHandle) -> None:
    """Non-centred eight-schools model with a half-normal prior on ``tau``."""
    mu = m.sample("mu", Normal(0.0, 5.0))
    tau = m.sample("tau", TAU_PRIOR)
    theta = m.sample("theta", Normal(mu, tau), shape=8, reparam="noncentered")
    m.observe("y", ppl.elementwise(Normal, theta, SCHOOLS_S), SCHOOLS_Y)


def normal_model(m: ModelHandle) -> None:
    """Model a normal mean with known scale; ``condition`` supplies the data."""
    mu = m.sample("mu", Normal(PRIOR_MEAN, PRIOR_SD))
    m.observe("y", Normal(mu, NORMAL_SIGMA), None, shape=len(NORMAL_DATA))
    m.deterministic("twice_mu", 2.0 * mu)


NORMAL_CONDITIONED = ppl.condition(normal_model, {"y": NORMAL_DATA})


def beta_binomial(m: ModelHandle) -> None:
    """Beta prior and binomial likelihood (conjugate)."""
    p = m.sample("p", Beta(BETA_A, BETA_B))
    m.observe("k", Binomial(TRIALS, p), SUCCESSES)


@dataclass(frozen=True)
class ReflectedExponential:
    """``-X`` with ``X ~ Exponential(rate)`` shifted to the upper bound ``b``."""

    bound: float
    rate: float = 1.0

    @property
    def support(self) -> RealSupport:
        """Return ``(-inf, bound]``."""
        return RealSupport(None, self.bound, upper_closed=True)

    def log_pdf(self, x: float) -> float:
        """Return the log-density ``log rate - rate (bound - x)``."""
        if x > self.bound:
            return -math.inf
        return math.log(self.rate) - self.rate * (self.bound - x)

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[float, ...]:
        """Draw by reflecting exponential draws."""
        return tuple(
            self.bound - v for v in Exponential(self.rate).sample(size, seed=seed)
        )


def bounded_model(m: ModelHandle) -> None:
    """One latent site per bijection family."""
    m.sample("real", Normal(1.0, 2.0))
    m.sample("positive", Exponential(0.5))
    m.sample("shifted", Uniform(2.0, 5.0), shape=3)
    m.sample("gamma", Gamma(2.0, 1.0))
    m.sample("lognormal", LogNormal(0.0, 0.5))
    m.sample("unit", Beta(0.7, 0.9))
    m.sample("upper", ReflectedExponential(-1.0, 2.0))


class EightSchoolsTests(unittest.TestCase):
    """The PPL model reproduces the exact gridded hierarchical posterior."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run NUTS once on the PPL model and compute the grid posterior."""
        cls.result = ppl.sample(eight_schools, draws=150, warmup=150, chains=2, seed=1)
        cls.exact = hierarchical_normal_posterior(
            SCHOOLS_Y, SCHOOLS_S, mu_prior=(0.0, 5.0), log_tau_prior=TAU_PRIOR.log_pdf
        )

    def test_intermediate_representation(self) -> None:
        """Tracing records sites, bijections, and coordinate names."""
        program = ppl.trace(eight_schools)
        self.assertEqual(program.dimension, 10)
        self.assertEqual(
            program.unconstrained_names[:3], ("mu", "log(tau)", "theta_z[0]")
        )
        self.assertEqual(program.constrained_names[:3], ("mu", "tau", "theta[0]"))
        self.assertEqual([s.name for s in program.observed_sites], ["y"])
        self.assertEqual(program.site("tau").bijection, ppl.Bijection("lower", 0.0))
        self.assertEqual(program.site("theta").reparam, "noncentered")
        self.assertEqual(program.observed_data("y"), SCHOOLS_Y)
        point = (1.0, math.log(2.0), 0.5, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        values = program.constrain(point)
        self.assertAlmostEqual(values["tau"], 2.0, places=12)
        theta = values["theta"]
        assert isinstance(theta, tuple)
        self.assertAlmostEqual(theta[0], 2.0, places=12)
        self.assertAlmostEqual(theta[1], -1.0, places=12)

    def test_posterior_means_match_the_grid(self) -> None:
        """Posterior means of mu and tau lie within 4 Monte Carlo errors."""
        mu_mean, _ = self.exact.mu_moments()
        expected = {"mu": mu_mean, "tau": self.exact.tau_mean()}
        summaries = {s.name: s for s in self.result.summary()}
        for name, value in expected.items():
            summary = summaries[name]
            self.assertLess(abs(summary.mean - value), 4.0 * summary.mcse_mean, name)
            self.assertLess(summary.rank_normalized_rhat, 1.05, name)

    def test_engine_and_diagnostics_are_attached(self) -> None:
        """The raw NUTS result and per-chain statistics remain available."""
        hamiltonian = self.result.hamiltonian
        assert hamiltonian is not None
        self.assertEqual(hamiltonian.names, self.result.program.constrained_names)
        self.assertEqual(len(hamiltonian.statistics), 2)
        self.assertEqual(self.result.unconstrained.names[1], "log(tau)")
        self.assertTrue(all(t > 0.0 for t in self.result["tau"]))
        self.assertEqual(len(self.result["theta"]), 300)
        rhat = split_rhat(self.result.constrained.chain_columns("theta[0]"))
        self.assertLess(rhat, 1.05)


class ConjugateNormalTests(unittest.TestCase):
    """A normal-normal model matches the closed-form posterior."""

    def test_laplace_is_exact_for_a_gaussian_posterior(self) -> None:
        """The Laplace mode and standard deviation equal the closed form."""
        result = ppl.laplace(NORMAL_CONDITIONED, draws=200, seed=3)
        assert result.point is not None
        self.assertAlmostEqual(result.point["mu"], POSTERIOR_MEAN, delta=1e-6)
        self.assertAlmostEqual(result.point["twice_mu"], 2.0 * POSTERIOR_MEAN, 5)
        sd = result.engine.standard_deviations[0]  # type: ignore[union-attr]
        self.assertAlmostEqual(sd, POSTERIOR_SD, delta=1e-5)
        self.assertEqual(result.draw_count, 200)

    def test_nuts_matches_the_closed_form(self) -> None:
        """NUTS mean within 4 MCSE; sd within 10 %; deterministic site exact."""
        result = ppl.sample(NORMAL_CONDITIONED, draws=400, warmup=200, chains=2, seed=4)
        summary = result.summary()[0]
        self.assertLess(abs(summary.mean - POSTERIOR_MEAN), 4.0 * summary.mcse_mean)
        self.assertLess(abs(summary.standard_deviation / POSTERIOR_SD - 1.0), 0.1)
        for mu, twice in zip(result["mu"], result["twice_mu"], strict=True):
            self.assertEqual(twice, 2.0 * mu)  # type: ignore[operator]

    def test_hmc_metropolis_and_advi(self) -> None:
        """The other engines agree with the closed form within loose bounds."""
        hmc = ppl.sample(
            NORMAL_CONDITIONED, method="hmc", draws=300, warmup=200, chains=2, seed=5
        )
        metropolis = ppl.sample(
            NORMAL_CONDITIONED,
            method="metropolis",
            draws=2000,
            warmup=1000,
            chains=2,
            seed=6,
        )
        advi = ppl.fit_advi(NORMAL_CONDITIONED, draws=2000, seed=7)
        for result in (hmc, metropolis, advi):
            summary = result.summary()[0]
            self.assertLess(abs(summary.mean - POSTERIOR_MEAN), 0.1, result.method)
            ratio = summary.standard_deviation / POSTERIOR_SD
            self.assertLess(abs(ratio - 1.0), 0.25, result.method)
        self.assertIsInstance(metropolis.engine, MCMCResult)
        self.assertIsNone(advi.hamiltonian)


class BetaBinomialTests(unittest.TestCase):
    """The logit transform and its Jacobian are exact."""

    def test_unconstrained_density_is_beta_posterior_times_jacobian(self) -> None:
        """``log p(u) - log Beta(sigmoid(u)) - log J(u)`` is the log evidence."""
        program = ppl.trace(beta_binomial)
        exact = Beta(BETA_A + SUCCESSES, BETA_B + TRIALS - SUCCESSES)
        log_evidence = (
            math.lgamma(TRIALS + 1)
            - math.lgamma(SUCCESSES + 1)
            - math.lgamma(TRIALS - SUCCESSES + 1)
            + exact._log_beta_function()
            - Beta(BETA_A, BETA_B)._log_beta_function()
        )
        for u in (-4.0, -1.0, 0.0, 0.7, 3.0):
            x = 1.0 / (1.0 + math.exp(-u))
            reference = exact.log_pdf(x) + math.log(x) + math.log1p(-x)
            self.assertAlmostEqual(
                program.log_density((u,)) - reference, log_evidence, places=10
            )
        terms = program.terms((0.0,))
        self.assertAlmostEqual(terms.log_jacobian, math.log(0.25), places=14)
        self.assertAlmostEqual(terms.total, program.log_density((0.0,)), places=14)

    def test_unconstrained_density_integrates_to_the_evidence(self) -> None:
        """A trapezoid integral of ``exp(log p(u))`` over ``R`` is the evidence."""
        program = ppl.trace(beta_binomial)
        step = 0.01
        grid = [-30.0 + step * i for i in range(6001)]
        values = [math.exp(program.log_density((u,))) for u in grid]
        integral = step * (math.fsum(values) - 0.5 * (values[0] + values[-1]))
        evidence = math.comb(TRIALS, SUCCESSES) * math.exp(
            Beta(BETA_A + SUCCESSES, BETA_B + TRIALS - SUCCESSES)._log_beta_function()
            - Beta(BETA_A, BETA_B)._log_beta_function()
        )
        self.assertAlmostEqual(integral / evidence, 1.0, places=8)

    def test_nuts_matches_the_beta_posterior(self) -> None:
        """NUTS mean within 4 MCSE and sd within 10 % of the Beta posterior."""
        result = ppl.sample(beta_binomial, draws=500, warmup=200, chains=2, seed=8)
        exact = Beta(BETA_A + SUCCESSES, BETA_B + TRIALS - SUCCESSES)
        summary = result.summary()[0]
        self.assertLess(abs(summary.mean - exact.mean), 4.0 * summary.mcse_mean)
        ratio = summary.standard_deviation / math.sqrt(exact.variance)
        self.assertLess(abs(ratio - 1.0), 0.1)
        self.assertTrue(all(0.0 < p < 1.0 for p in result["p"]))  # type: ignore[operator]


class SupportTests(unittest.TestCase):
    """Bijections are correct and constrained draws stay in their support."""

    def test_bijection_choice_round_trip_and_jacobian(self) -> None:
        """Each family inverts exactly and has the derivative it reports."""
        cases = {
            RealSupport(): "identity",
            RealSupport(0.0, None, lower_closed=True): "lower",
            RealSupport(None, 3.0): "upper",
            RealSupport(-1.0, 4.0, True, True): "interval",
        }
        for support, kind in cases.items():
            bijection = ppl.bijection_for(support)
            self.assertEqual(bijection.kind, kind)
            for u in (-2.5, 0.0, 1.3):
                x = bijection.forward(u)
                self.assertTrue(support.contains(x))
                self.assertAlmostEqual(bijection.inverse(x), u, places=10)
                h = 1e-6
                slope = (bijection.forward(u + h) - bijection.forward(u - h)) / (2 * h)
                self.assertAlmostEqual(
                    bijection.log_abs_det_jacobian(u), math.log(abs(slope)), places=6
                )

    def test_draws_never_leave_their_support(self) -> None:
        """Every constrained draw of every site lies inside its support."""
        result = ppl.sample(
            bounded_model, method="metropolis", draws=600, warmup=600, chains=2, seed=9
        )
        program = result.program
        self.assertEqual(
            [s.bijection.kind for s in program.latent_sites if s.bijection],
            ["identity", "lower", "interval", "lower", "lower", "interval", "upper"],
        )
        for site in program.latent_sites:
            support = site.support
            assert isinstance(support, RealSupport)
            for draw in result[site.name]:
                elements = draw if isinstance(draw, tuple) else (draw,)
                for value in elements:
                    self.assertTrue(support.contains(value), (site.name, value))

    def test_parameter_dependent_bounds(self) -> None:
        """``Uniform(0, theta)`` re-derives its bijection at every evaluation."""

        def model(m: ModelHandle) -> None:
            """Upper bound of ``x`` is the latent ``theta``."""
            theta = m.sample("theta", Uniform(1.0, 10.0))
            m.sample("x", Uniform(0.0, theta))

        program = ppl.trace(model)
        for u in ((0.3, 2.0), (-1.0, -3.0), (2.0, 5.0)):
            values = program.constrain(u)
            self.assertTrue(0.0 < values["x"] < values["theta"])  # type: ignore[operator]
        theta = 1.0 + 9.0 / (1.0 + math.exp(-0.3))
        x_part = math.log(theta) - 2.0 - 2.0 * math.log1p(math.exp(-2.0))
        theta_part = math.log(9.0) - 0.3 - 2.0 * math.log1p(math.exp(-0.3))
        expected = (-math.log(9.0) + theta_part) + (-math.log(theta) + x_part)
        self.assertAlmostEqual(program.log_density((0.3, 2.0)), expected, places=10)


class PredictiveTests(unittest.TestCase):
    """Prior and posterior predictive simulation."""

    def test_prior_predictive_shapes_and_determinism(self) -> None:
        """Same seed, same draws; observed vector sites keep their length."""
        first = ppl.prior_predictive(normal_model, 7, seed=3)
        second = ppl.prior_predictive(normal_model, 7, seed=3)
        other = ppl.prior_predictive(normal_model, 7, seed=4)
        self.assertEqual(first, second)
        self.assertNotEqual(first.sites["y"], other.sites["y"])
        self.assertEqual(first.names, ("mu", "y", "twice_mu"))
        self.assertEqual(first.kinds["y"], "observed")
        self.assertEqual(len(first["y"]), 7)
        self.assertTrue(all(len(y) == 10 for y in first["y"]))  # type: ignore[arg-type]
        for mu, twice in zip(first["mu"], first["twice_mu"], strict=True):
            self.assertEqual(twice, 2.0 * mu)  # type: ignore[operator]

    def test_posterior_predictive_fixes_latents(self) -> None:
        """Latent values are copied from the posterior; data are re-simulated."""
        posterior = ppl.laplace(NORMAL_CONDITIONED, draws=50, seed=5)
        replicated = ppl.posterior_predictive(
            NORMAL_CONDITIONED, posterior, seed=6, max_draws=20
        )
        again = ppl.posterior_predictive(
            NORMAL_CONDITIONED, posterior, seed=6, max_draws=20
        )
        self.assertEqual(replicated, again)
        self.assertLessEqual(replicated.draws, 20)
        draws = posterior["mu"][::3]
        self.assertEqual(replicated["mu"], draws[: replicated.draws])
        self.assertTrue(all(len(y) == 10 for y in replicated["y"]))  # type: ignore[arg-type]
        self.assertNotEqual(replicated["y"][0], NORMAL_DATA)
        from_dicts = ppl.posterior_predictive(
            normal_model, [{"mu": 1.0}, {"mu": 2.0}], seed=1
        )
        self.assertEqual(from_dicts["mu"], (1.0, 2.0))

    def test_predictive_pvalues_integration(self) -> None:
        """The PPC wrapper returns p-values computed from replicated sites."""
        posterior = ppl.laplace(NORMAL_CONDITIONED, draws=200, seed=7)
        check = ppl.predictive_pvalues(
            NORMAL_CONDITIONED,
            posterior,
            "y",
            statistics={"max": max, "min": min},
            discrepancies={
                "sse": lambda y, theta: math.fsum((v - theta[0]) ** 2 for v in y)
            },
            keep_replicates=True,
            seed=8,
        )
        self.assertIsInstance(check, PosteriorPredictivePValues)
        self.assertEqual(check.draws, 200)
        self.assertEqual(len(check.replicates), 200)
        for value in check.p_values().values():
            self.assertTrue(0.0 < value < 1.0)


class ErrorTests(unittest.TestCase):
    """Invalid models fail with specific messages."""

    def test_duplicate_site_names(self) -> None:
        """A name used twice is rejected."""

        def model(m: ModelHandle) -> None:
            """Declare ``a`` twice."""
            m.sample("a", Normal(0.0, 1.0))
            m.observe("a", Normal(0.0, 1.0), 1.0)

        with self.assertRaisesRegex(ModelSpecificationError, "duplicate site name"):
            ppl.trace(model)

    def test_unknown_supports(self) -> None:
        """Discrete latent sites and objects without a support are rejected."""

        def discrete(m: ModelHandle) -> None:
            """Declare a binomial latent variable."""
            m.sample("k", Binomial(5, 0.5))

        class Opaque:
            """A distribution-like object without a support."""

            def log_pdf(self, x: float) -> float:
                """Return zero."""
                return 0.0

        def opaque(m: ModelHandle) -> None:
            """Latent site with an opaque distribution."""
            m.sample("x", Opaque())

        with self.assertRaisesRegex(ModelSpecificationError, "unsupported support"):
            ppl.trace(discrete)
        with self.assertRaisesRegex(ModelSpecificationError, "no 'support'"):
            ppl.trace(opaque)
        with self.assertRaisesRegex(ModelSpecificationError, "unsupported support"):
            ppl.bijection_for("not a support")

    def test_specification_errors(self) -> None:
        """Missing data, bad reparam, shapes, conditions, and structure changes."""

        def no_data(m: ModelHandle) -> None:
            """Observed site without data."""
            mu = m.sample("mu", Normal(0.0, 1.0))
            m.observe("y", Normal(mu, 1.0))

        def bad_reparam(m: ModelHandle) -> None:
            """Non-centred reparameterization of a gamma site."""
            m.sample("g", Gamma(2.0, 1.0), reparam="noncentered")

        def bad_shape(m: ModelHandle) -> None:
            """Shape that disagrees with a distribution list."""
            m.sample("v", [Normal(0.0, 1.0)] * 3, shape=2)

        def branching(m: ModelHandle) -> None:
            """Site structure depends on a parameter value."""
            mu = m.sample("mu", Normal(0.0, 1.0))
            if mu > 0.5:
                m.sample("extra", Normal(0.0, 1.0))

        with self.assertRaisesRegex(ModelSpecificationError, "has no data"):
            ppl.sample(no_data, draws=10, warmup=10, chains=1, seed=1)
        with self.assertRaisesRegex(ModelSpecificationError, "needs Normal"):
            ppl.trace(bad_reparam)
        with self.assertRaisesRegex(ModelSpecificationError, "does not match"):
            ppl.trace(bad_shape)
        with self.assertRaisesRegex(ModelSpecificationError, "never declares"):
            ppl.trace(ppl.condition(normal_model, {"z": 1.0}))
        with self.assertRaisesRegex(ModelSpecificationError, "was not present"):
            ppl.trace(branching).log_density((1.0,))
        with self.assertRaisesRegex(ModelSpecificationError, "no latent sites"):
            ppl.laplace(ppl.condition(normal_model, {"y": NORMAL_DATA, "mu": 1.0}))
        with self.assertRaises(ValueError):
            ppl.sample(NORMAL_CONDITIONED, method="gibbs")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ppl.elementwise(Normal, (1.0, 2.0), (1.0, 2.0, 3.0))

    def test_invalid_parameters_become_minus_infinity(self) -> None:
        """A distribution constructor error inside the model maps to ``-inf``."""

        def model(m: ModelHandle) -> None:
            """Scale ``|x - 0.5|`` is zero at ``x = 0.5``."""
            x = m.sample("x", Normal(0.0, 1.0))
            m.observe("y", Normal(0.0, abs(x - 0.5)), 1.0)

        program = ppl.trace(ppl.condition(model, {}))
        self.assertEqual(program.log_density((0.5,)), -math.inf)
        self.assertTrue(math.isfinite(program.log_density((0.1,))))
        with self.assertRaises(ValueError):
            program.terms((0.5,))


if __name__ == "__main__":
    unittest.main()
