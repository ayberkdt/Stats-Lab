"""Distribution log-densities with dual and reverse-mode parameters (ADR-0090).

The families of :mod:`probstat.distributions` and the multinomial and
multivariate normal laws accept :class:`~probstat.numerics.Dual` numbers and
:class:`~probstat.numerics.ReverseVariable` values in ``log_pdf`` and
``log_pmf``. The oracles are central finite differences, closed-form
derivatives, the float log-density itself (the value must not change) and,
for the probabilistic programs, conjugate posteriors judged by their Monte
Carlo standard error.
"""

from __future__ import annotations

import math
import unittest
import warnings

from probstat import distributions as d
from probstat.inference import ppl
from probstat.multivariate import Dirichlet, Multinomial, MultivariateNormal
from probstat.numerics import (
    AutodiffUnsupportedError,
    Dual,
    digamma,
    gradient,
    reverse_gradient,
    reverse_hessian,
    trigamma,
)


def central(function, point, relative_step=1e-6):
    """Return the central-difference gradient (truncation error O(h^2))."""
    result = []
    for index, value in enumerate(point):
        step = relative_step * max(1.0, abs(value))
        up = list(point)
        down = list(point)
        up[index] = value + step
        down[index] = value - step
        result.append((function(up) - function(down)) / (2.0 * step))
    return result


def quiet(function, *args, **kwargs):
    """Call a sampler with convergence warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


# (name, log-density of a parameter vector, point); the last coordinate of a
# continuous case is the evaluation point x, so its derivative is checked too.
FAMILIES = (
    ("Normal", lambda v: d.Normal(v[0], v[1]).log_pdf(v[2]), (0.3, 1.7, 1.1)),
    ("Beta", lambda v: d.Beta(v[0], v[1]).log_pdf(v[2]), (2.0, 3.5, 0.3)),
    ("Beta large", lambda v: d.Beta(v[0], v[1]).log_pdf(v[2]), (200.0, 350.0, 0.36)),
    ("Cauchy", lambda v: d.Cauchy(v[0], v[1]).log_pdf(v[2]), (0.3, 1.7, 2.1)),
    ("ChiSquare", lambda v: d.ChiSquare(v[0]).log_pdf(v[1]), (3.5, 2.2)),
    ("Exponential", lambda v: d.Exponential(v[0]).log_pdf(v[1]), (1.5, 0.7)),
    ("FisherF", lambda v: d.FisherF(v[0], v[1]).log_pdf(v[2]), (4.0, 7.0, 1.3)),
    (
        "FisherF large",
        lambda v: d.FisherF(v[0], v[1]).log_pdf(v[2]),
        (400.0, 700.0, 1.05),
    ),
    ("Gamma", lambda v: d.Gamma(v[0], v[1]).log_pdf(v[2]), (2.5, 1.5, 1.2)),
    ("Gamma large", lambda v: d.Gamma(v[0], v[1]).log_pdf(v[2]), (250.0, 3.0, 80.0)),
    (
        "GEV",
        lambda v: d.GeneralizedExtremeValue(v[0], v[1], v[2]).log_pdf(v[3]),
        (0.2, 1.3, 0.25, 1.0),
    ),
    (
        "GPD",
        lambda v: d.GeneralizedPareto(v[0], v[1], v[2]).log_pdf(v[3]),
        (0.2, 1.3, 0.25, 1.0),
    ),
    ("Gumbel", lambda v: d.Gumbel(v[0], v[1]).log_pdf(v[2]), (0.2, 1.3, 1.0)),
    (
        "Gumbel min",
        lambda v: d.Gumbel(v[0], v[1], "min").log_pdf(v[2]),
        (0.2, 1.3, 1.0),
    ),
    ("Laplace", lambda v: d.Laplace(v[0], v[1]).log_pdf(v[2]), (0.2, 1.3, 1.0)),
    ("Logistic", lambda v: d.Logistic(v[0], v[1]).log_pdf(v[2]), (0.2, 1.3, 1.0)),
    ("LogNormal", lambda v: d.LogNormal(v[0], v[1]).log_pdf(v[2]), (0.2, 0.8, 1.5)),
    ("Pareto", lambda v: d.Pareto(v[0], v[1]).log_pdf(v[2]), (1.2, 2.5, 3.0)),
    (
        "PERT",
        lambda v: d.PERT(v[0], v[1], v[2], v[3]).log_pdf(v[4]),
        (0.0, 1.0, 3.0, 4.0, 1.7),
    ),
    ("Rayleigh", lambda v: d.Rayleigh(v[0]).log_pdf(v[1]), (1.3, 0.9)),
    (
        "StudentT",
        lambda v: d.StudentT(v[0], v[1], v[2]).log_pdf(v[3]),
        (4.5, 0.2, 1.3, 1.0),
    ),
    (
        "StudentT large",
        lambda v: d.StudentT(v[0], v[1], v[2]).log_pdf(v[3]),
        (450.0, 0.2, 1.3, 1.0),
    ),
    (
        "Triangular",
        lambda v: d.Triangular(v[0], v[1], v[2]).log_pdf(v[3]),
        (0.0, 1.0, 3.0, 1.7),
    ),
    (
        "TruncatedNormal interval",
        lambda v: d.TruncatedNormal(v[0], v[1], lower=v[2], upper=v[3]).log_pdf(v[4]),
        (0.3, 1.2, -0.5, 2.0, 0.4),
    ),
    (
        "TruncatedNormal lower",
        lambda v: d.TruncatedNormal(v[0], v[1], lower=v[2]).log_pdf(v[3]),
        (0.3, 1.2, 1.5, 2.0),
    ),
    (
        "TruncatedNormal upper",
        lambda v: d.TruncatedNormal(v[0], v[1], upper=v[2]).log_pdf(v[3]),
        (0.3, 1.2, -1.5, -2.0),
    ),
    ("Uniform", lambda v: d.Uniform(v[0], v[1]).log_pdf(v[2]), (0.3, 2.2, 1.0)),
    ("Weibull", lambda v: d.Weibull(v[0], v[1]).log_pdf(v[2]), (1.7, 1.3, 0.9)),
    (
        "Bernoulli",
        lambda v: d.Bernoulli(v[0]).log_pmf(1) + 2.0 * d.Bernoulli(v[0]).log_pmf(0),
        (0.3,),
    ),
    ("Binomial", lambda v: d.Binomial(10, v[0]).log_pmf(3), (0.3,)),
    (
        "Categorical",
        lambda v: d.Categorical({"a": v[0], "b": 1.0 - v[0]}).log_pmf("a"),
        (0.3,),
    ),
    ("Geometric", lambda v: d.Geometric(v[0]).log_pmf(3), (0.3,)),
    ("NegativeBinomial", lambda v: d.NegativeBinomial(4, v[0]).log_pmf(3), (0.3,)),
    ("Poisson", lambda v: d.Poisson(v[0]).log_pmf(3), (2.3,)),
    (
        "Multinomial",
        lambda v: Multinomial(9, (v[0], v[1], 1.0 - v[0] - v[1])).log_pmf((2, 3, 4)),
        (0.2, 0.3),
    ),
    (
        "MultivariateNormal",
        lambda v: MultivariateNormal((v[0], v[1]), ((1.0, 0.3), (0.3, 2.0))).log_pdf(
            (v[2], v[3])
        ),
        (0.2, -0.4, 0.5, 0.7),
    ),
)


class FamilyGradientTests(unittest.TestCase):
    """Every family is differentiated exactly in both modes."""

    def test_gradients_match_finite_differences(self) -> None:
        """Reverse equals forward to rounding and central differences to 1e-7."""
        for name, function, point in FAMILIES:
            with self.subTest(family=name):
                reverse = reverse_gradient(function, point)
                forward = gradient(function, point)
                finite = central(function, point)
                for r, f, c in zip(reverse, forward, finite, strict=True):
                    self.assertAlmostEqual(r, f, delta=1e-12 * max(1.0, abs(f)))
                    self.assertAlmostEqual(r, c, delta=1e-7 * max(1.0, abs(c)))

    def test_values_are_the_float_values(self) -> None:
        """The value carried by a variable is the float log-density exactly."""
        for name, function, point in FAMILIES:
            with self.subTest(family=name):
                seeded = function([Dual(value, (1.0,)) for value in point])
                self.assertEqual(seeded.value, function(list(point)))

    def test_closed_form_gradients(self) -> None:
        """Normal, Gamma, Beta, Poisson, Binomial, Multinomial: textbook scores."""
        mu, sigma, x = 0.3, 1.7, 1.1
        z = (x - mu) / sigma
        normal = reverse_gradient(
            lambda v: d.Normal(v[0], v[1]).log_pdf(x), (mu, sigma)
        )
        self.assertAlmostEqual(normal[0], z / sigma, delta=1e-15)
        self.assertAlmostEqual(normal[1], (z * z - 1.0) / sigma, delta=1e-15)
        k, rate, y = 2.5, 1.5, 1.2
        gamma = reverse_gradient(lambda v: d.Gamma(v[0], v[1]).log_pdf(y), (k, rate))
        self.assertAlmostEqual(
            gamma[0], math.log(rate) + math.log(y) - digamma(k), delta=1e-14
        )
        self.assertAlmostEqual(gamma[1], k / rate - y, delta=1e-14)
        a, b, t = 2.0, 3.5, 0.3
        beta = reverse_gradient(lambda v: d.Beta(v[0], v[1]).log_pdf(t), (a, b))
        self.assertAlmostEqual(
            beta[0], math.log(t) - digamma(a) + digamma(a + b), delta=1e-14
        )
        self.assertAlmostEqual(
            beta[1], math.log1p(-t) - digamma(b) + digamma(a + b), delta=1e-14
        )
        poisson = reverse_gradient(lambda v: d.Poisson(v[0]).log_pmf(3), (2.3,))
        self.assertAlmostEqual(poisson[0], 3.0 / 2.3 - 1.0, delta=1e-15)
        binomial = reverse_gradient(
            lambda v: d.Binomial(10, v[0]).log_pmf(3), (0.3,)
        )
        self.assertAlmostEqual(binomial[0], 3.0 / 0.3 - 7.0 / 0.7, delta=1e-13)
        # Multinomial with the last probability 1 - p1 - p2: x_i/p_i - x_3/p_3.
        multinomial = reverse_gradient(
            lambda v: Multinomial(9, (v[0], v[1], 1.0 - v[0] - v[1])).log_pmf(
                (2, 3, 4)
            ),
            (0.2, 0.3),
        )
        self.assertAlmostEqual(multinomial[0], 2.0 / 0.2 - 4.0 / 0.5, delta=1e-12)
        self.assertAlmostEqual(multinomial[1], 3.0 / 0.3 - 4.0 / 0.5, delta=1e-12)

    def test_second_derivatives(self) -> None:
        """Forward-over-reverse Hessians: Gamma shape (-trigamma) and Normal."""
        for k, x in ((2.5, 1.2), (250.0, 80.0)):
            with self.subTest(shape=k):
                hessian = reverse_hessian(
                    lambda v, x=x: d.Gamma(v[0], 1.5).log_pdf(x), (k,)
                )
                self.assertAlmostEqual(
                    hessian[0][0], -trigamma(k), delta=1e-12 * trigamma(k)
                )
        mu, sigma, x = 0.3, 1.7, 1.1
        hessian = reverse_hessian(
            lambda v: d.Normal(v[0], v[1]).log_pdf(x), (mu, sigma)
        )
        z = (x - mu) / sigma
        self.assertAlmostEqual(hessian[0][0], -1.0 / sigma**2, delta=1e-14)
        self.assertAlmostEqual(
            hessian[1][1], (1.0 - 3.0 * z * z) / sigma**2, delta=1e-14
        )

    def test_validation_reads_the_primal_value(self) -> None:
        """An invalid primal value is rejected; a valid variable is kept."""
        with self.assertRaises(ValueError):
            d.Normal(0.0, Dual(-1.0, (1.0,)))
        with self.assertRaises(ValueError):
            d.Poisson(Dual(math.inf, (1.0,)))
        with self.assertRaises(ValueError):
            Multinomial(3, (Dual(0.5, (1.0,)), 0.6))
        variable = Dual(2.0, (1.0,))
        self.assertIs(d.Gamma(variable).shape, variable)

    def test_other_methods_still_require_floats(self) -> None:
        """The cdf and sf raise a TypeError instead of dropping a derivative.

        ``AutodiffUnsupportedError`` subclasses ``TypeError``; the gamma and
        beta functions reach a float validator first.
        """
        with self.assertRaises(AutodiffUnsupportedError):
            d.Normal(Dual(0.3, (1.0,)), 1.0).cdf(0.0)
        with self.assertRaises(TypeError):
            d.Gamma(Dual(2.0, (1.0,))).cdf(1.0)
        with self.assertRaises(TypeError):
            d.Beta(2.0, Dual(3.0, (1.0,))).sf(0.4)


DATA = (1.2, 0.4, 2.1, 1.7, 0.9)
COUNTS = (2, 4, 3, 5)
TRIALS = ((3, 10), (6, 10), (4, 10))


def normal_mean(m):
    """Declare a normal mean with unit variance and a N(0, 10^2) prior."""
    mu = m.sample("mu", d.Normal(0.0, 10.0))
    m.observe("y", d.Normal(mu, 1.0), DATA)


def normal_precision(m):
    """Declare a normal precision with mean 1 and a Gamma(2, 1) prior."""
    tau = m.sample("tau", d.Gamma(2.0, 1.0))
    m.observe("y", d.Normal(1.0, 1.0 / tau**0.5), DATA)


def poisson_gamma(m):
    """Declare Poisson counts with a Gamma(3, 1) prior on the rate."""
    lam = m.sample("lam", d.Gamma(3.0, 1.0))
    m.observe("k", d.Poisson(lam), COUNTS)


def beta_binomial(m):
    """Declare binomial successes with a Beta(2, 3) prior."""
    p = m.sample("p", d.Beta(2.0, 3.0))
    for index, (successes, n) in enumerate(TRIALS):
        m.observe(f"b{index}", d.Binomial(n, p), successes)


def dirichlet_multinomial(m):
    """Declare multinomial counts with a Dirichlet(2, 1.5, 1) prior."""
    p = m.sample("p", Dirichlet((2.0, 1.5, 1.0)))
    m.observe("x", Multinomial(20, p), (12, 5, 3))


def student_location(m):
    """Declare Student-t observations with a N(0, 10^2) location prior."""
    mu = m.sample("mu", d.Normal(0.0, 10.0))
    m.observe("y", d.StudentT(4.0, mu, 1.0), DATA)


def student_posterior_mean() -> float:
    """Posterior mean of ``student_location`` by trapezoidal quadrature."""
    grid = [-6.0 + 14.0 * i / 20000 for i in range(20001)]
    law = [
        -0.5 * (mu / 10.0) ** 2
        + sum(d.StudentT(4.0, mu, 1.0).log_pdf(y) for y in DATA)
        for mu in grid
    ]
    peak = max(law)
    weights = [math.exp(v - peak) for v in law]
    return sum(w * mu for w, mu in zip(weights, grid, strict=True)) / sum(weights)


class ProgramGradientTests(unittest.TestCase):
    """Programs written with library laws take exact reverse-mode gradients."""

    def test_program_gradients_match_finite_differences(self) -> None:
        """ModelIR.gradient (reverse) equals central differences to 1e-7."""
        for model in (
            normal_mean,
            normal_precision,
            poisson_gamma,
            beta_binomial,
            dirichlet_multinomial,
            student_location,
        ):
            with self.subTest(model=model.__name__):
                program = ppl.trace(model)
                evaluator = program.gradient_evaluator("auto")
                u = tuple(0.1 * (i + 1) for i in range(program.dimension))
                exact = evaluator.gradient(u)
                self.assertEqual(evaluator.method, "reverse")
                self.assertIsNone(evaluator.fallback_reason)
                finite = program.gradient(u, method="finite-difference")
                for a, b in zip(exact, finite, strict=True):
                    self.assertAlmostEqual(a, b, delta=1e-7 * max(1.0, abs(b)))

    def test_closed_form_program_gradient(self) -> None:
        """Poisson-gamma on log(lam): d/du = (3 + sum k) - (1 + n) lam."""
        program = ppl.trace(poisson_gamma)
        u = 0.4
        lam = math.exp(u)
        expected = 3.0 + sum(COUNTS) - (1.0 + len(COUNTS)) * lam
        self.assertAlmostEqual(
            program.gradient((u,), method="reverse")[0], expected, delta=1e-12
        )

    def test_samplers_recover_conjugate_posteriors(self) -> None:
        """NUTS and HMC means agree with the conjugate means within 4 MCSE."""
        n = len(DATA)
        squares = sum((y - 1.0) ** 2 for y in DATA)
        successes = sum(s for s, _ in TRIALS)
        failures = sum(t - s for s, t in TRIALS)
        cases = (
            (normal_mean, "nuts", {"mu": sum(DATA) / (n + 0.01)}),
            (normal_precision, "hmc", {"tau": (2.0 + n / 2) / (1.0 + squares / 2)}),
            (poisson_gamma, "nuts", {"lam": (3.0 + sum(COUNTS)) / (1.0 + 4)}),
            (
                beta_binomial,
                "hmc",
                {"p": (2.0 + successes) / (5.0 + successes + failures)},
            ),
            (
                dirichlet_multinomial,
                "nuts",
                {"p[0]": 14.0 / 24.5, "p[1]": 6.5 / 24.5, "p[2]": 4.0 / 24.5},
            ),
            (student_location, "nuts", {"mu": student_posterior_mean()}),
        )
        for model, method, expected in cases:
            with self.subTest(model=model.__name__, method=method):
                posterior = quiet(
                    ppl.sample,
                    model,
                    method=method,
                    draws=400,
                    warmup=300,
                    chains=2,
                    seed=11,
                )
                self.assertEqual(posterior.gradient_method, "reverse")
                self.assertIsNone(posterior.gradient_fallback_reason)
                rows = {row.name: row for row in posterior.summary()}
                for name, value in expected.items():
                    row = rows[name]
                    self.assertLess(abs(row.mean - value), 4.0 * row.mcse_mean, name)

    def test_laplace_uses_reverse_hessian(self) -> None:
        """Laplace on library laws: reverse Hessians and the exact Gaussian case."""
        exact = ppl.laplace(normal_mean, draws=10, seed=1)
        self.assertEqual(exact.gradient_method, "reverse")
        self.assertEqual(exact.engine.hessian_method, "reverse")
        self.assertAlmostEqual(
            exact.engine.covariance[0][0], 1.0 / (0.01 + len(DATA)), delta=1e-12
        )
        # Poisson-gamma on u = log(lam): the mode solves (3 + S) = 5 exp(u),
        # and the curvature there is 5 exp(u) = 3 + S.
        fit = ppl.laplace(poisson_gamma, draws=10, seed=1)
        self.assertEqual(fit.engine.hessian_method, "reverse")
        self.assertAlmostEqual(
            fit.engine.covariance[0][0], 1.0 / (3.0 + sum(COUNTS)), delta=1e-6
        )


if __name__ == "__main__":
    unittest.main()
