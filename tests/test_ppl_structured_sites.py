"""Structured (simplex, ordered, correlation, covariance) and discrete PPL sites."""

from __future__ import annotations

import math
import random
import unittest

from probstat.distributions import (
    Bernoulli,
    Beta,
    Binomial,
    IntegerSupport,
    Normal,
    Poisson,
    RealSupport,
    Uniform,
)
from probstat.inference import ppl
from probstat.inference.ppl import (
    ModelHandle,
    ModelSpecificationError,
    OrderedVector,
    StructuredSupport,
)
from probstat.linalg import log_abs_determinant
from probstat.multivariate import (
    Dirichlet,
    InverseWishart,
    LKJCholesky,
    LKJCorrelation,
    Multinomial,
    MultivariateNormal,
    Wishart,
    dirichlet_posterior,
    lkj_log_normalizer,
)
from probstat.numerics import admath, gauss_kronrod
from probstat.optimization import central_jacobian

LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)

# Every support with a point in its unconstrained space.
SUPPORTS = (
    (StructuredSupport("simplex", 4), (0.3, -0.7, 1.1)),
    (StructuredSupport("ordered", 3), (0.3, -0.7, 1.1)),
    (StructuredSupport("ordered", 3, 1.5), (0.3, -0.7, 1.1)),
    (StructuredSupport("cholesky_correlation", 3), (0.3, -0.7, 1.1)),
    (StructuredSupport("correlation", 3), (0.3, -0.7, 1.1)),
    (StructuredSupport("correlation", 4), (0.3, -0.7, 1.1, 0.2, -0.4, 0.9)),
    (StructuredSupport("covariance", 3), (0.3, -0.7, 1.1, 0.2, -0.4, 0.9)),
)

SCALE = ((2.0, 0.3), (0.3, 1.0))


def log_beta_function(a: float, b: float) -> float:
    """Return ``log B(a, b)``."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class TransformTests(unittest.TestCase):
    """Bijections, log-Jacobians, and round trips of the structured supports."""

    def test_log_jacobian_matches_finite_difference_determinant(self) -> None:
        """``log |det J|`` equals the log-determinant of a central-difference J."""
        for support, u in SUPPORTS:
            with self.subTest(kind=support.kind, d=support.dimension):

                def free(v: tuple[float, ...], s: StructuredSupport = support) -> tuple:
                    """Free coordinates of the image of ``v``."""
                    return s.free_coordinates(s.forward(v).value)

                jacobian, _ = central_jacobian(free, u)
                self.assertAlmostEqual(
                    support.forward(u).log_jacobian,
                    log_abs_determinant(jacobian),
                    delta=1e-7,
                )

    def test_round_trips(self) -> None:
        """``T^-1(T(u)) = u`` and ``T(T^-1(x)) = x``."""
        for support, u in SUPPORTS:
            with self.subTest(kind=support.kind, d=support.dimension):
                value = support.forward(u).value
                back = support.inverse(value)
                self.assertLess(
                    max(abs(a - b) for a, b in zip(back, u, strict=True)), 1e-12
                )
                again = support.flatten(support.forward(back).value)
                self.assertLess(
                    max(
                        abs(a - b)
                        for a, b in zip(again, support.flatten(value), strict=True)
                    ),
                    1e-12,
                )
                self.assertEqual(
                    support.unflatten(support.flatten(value)),
                    support.checked_value(value, "x"),
                )

    def test_images_lie_in_the_support(self) -> None:
        """Simplex sums to one, factors have unit rows, matrices are SPD."""
        generator = random.Random(4)
        for support, u in SUPPORTS:
            for _ in range(20):
                point = tuple(generator.uniform(-3.0, 3.0) for _ in u)
                value = support.forward(point).value
                if support.kind == "simplex":
                    self.assertAlmostEqual(math.fsum(value), 1.0, places=14)
                    self.assertTrue(all(v > 0.0 for v in value))
                elif support.kind == "ordered":
                    self.assertTrue(
                        all(b > a for a, b in zip(value, value[1:], strict=False))
                    )
                elif support.kind == "cholesky_correlation":
                    for row in value:
                        self.assertAlmostEqual(math.fsum(v * v for v in row), 1.0)
                else:
                    if support.kind == "correlation":
                        self.assertTrue(all(value[i][i] == 1.0 for i in range(3)))
                    self.assertGreater(log_abs_determinant(value), -math.inf)

    def test_invalid_supports(self) -> None:
        """Unknown kinds, small dimensions, misplaced bounds, and bad points."""
        with self.assertRaises(ValueError):
            StructuredSupport("sphere", 3)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            StructuredSupport("simplex", 1)
        with self.assertRaises(ValueError):
            StructuredSupport("simplex", 3, 0.0)
        with self.assertRaises(ValueError):
            StructuredSupport("simplex", 3).inverse((0.5, 0.6, -0.1))
        with self.assertRaises(ValueError):
            StructuredSupport("ordered", 3).inverse((1.0, 0.5, 2.0))
        with self.assertRaises(ValueError):
            StructuredSupport("simplex", 3).forward((0.0,))


# ---------------------------------------------------------------------------
# LKJ
# ---------------------------------------------------------------------------


def lkj_volume_three(eta: float) -> float:
    """Integrate ``det(R)^(eta - 1)`` over the 3 x 3 correlation matrices.

    With ``a = sin(phi)``, ``b = sin(psi)`` the outer integrands are smooth;
    for fixed ``(a, b)`` the admissible ``c = r_23`` fill
    ``ab +- sqrt((1 - a^2)(1 - b^2))``.
    """

    def inner(a: float, b: float) -> float:
        """Integral over ``c`` of the determinant power."""
        half_width = math.sqrt((1.0 - a * a) * (1.0 - b * b))

        def power(c: float) -> float:
            """``det(R)^(eta - 1)`` at ``(a, b, c)``."""
            det = 1.0 - a * a - b * b - c * c + 2.0 * a * b * c
            return max(det, 0.0) ** (eta - 1.0)

        return gauss_kronrod(
            power, a * b - half_width, a * b + half_width, rel_tol=1e-10
        ).value

    def middle(phi: float) -> float:
        """Integral over ``b`` for ``a = sin(phi)``."""
        a = math.sin(phi)
        return gauss_kronrod(
            lambda psi: math.cos(psi) * inner(a, math.sin(psi)),
            -math.pi / 2,
            math.pi / 2,
            rel_tol=1e-9,
        ).value * math.cos(phi)

    return gauss_kronrod(middle, -math.pi / 2, math.pi / 2, rel_tol=1e-9).value


class LKJTests(unittest.TestCase):
    """Density, normalizing constant, and sampling of the LKJ law."""

    def test_normalizing_constant_matches_quadrature(self) -> None:
        """``c_d(eta)`` equals the integral of ``det^(eta-1)`` for d = 2, 3."""
        for eta in (0.5, 1.0, 2.0, 3.7):
            exact = gauss_kronrod(
                lambda t, e=eta: math.cos(t) ** (2.0 * e - 1.0),
                -math.pi / 2,
                math.pi / 2,
                rel_tol=1e-12,
            ).value
            self.assertAlmostEqual(
                lkj_log_normalizer(2, eta), math.log(exact), places=10
            )
        self.assertAlmostEqual(math.exp(lkj_log_normalizer(3, 1.0)), math.pi**2 / 2)
        for eta in (1.0, 2.0, 3.0):
            with self.subTest(eta=eta):
                self.assertAlmostEqual(
                    lkj_log_normalizer(3, eta),
                    math.log(lkj_volume_three(eta)),
                    places=7,
                )

    def test_eta_one_two_by_two_is_uniform(self) -> None:
        """LKJ(1) on 2 x 2 matrices makes ``r`` uniform on (-1, 1)."""
        law = LKJCorrelation(2, 1.0)
        for r in (-0.9, -0.2, 0.0, 0.55, 0.99):
            self.assertAlmostEqual(law.log_pdf(((1.0, r), (r, 1.0))), math.log(0.5))

        def model(m: ModelHandle) -> None:
            """Declare a single 2 x 2 correlation site."""
            m.sample("R", LKJCorrelation(2, 1.0))

        program = ppl.trace(model)
        self.assertEqual(program.dimension, 1)
        for u in (-2.0, -0.3, 0.0, 1.4):
            expected = math.log(0.5) + math.log(1.0 - math.tanh(u) ** 2)
            self.assertAlmostEqual(program.log_density((u,)), expected, places=12)
        draws = [m[0][1] for m in law.sample(4000, seed=12)]
        mean = math.fsum(draws) / len(draws)
        variance = math.fsum(r * r for r in draws) / len(draws) - mean * mean
        self.assertLess(abs(mean), 4.0 * math.sqrt(1.0 / 3.0 / len(draws)))
        # Var(r^2) = 1/5 - 1/9 for r ~ U(-1, 1).
        self.assertLess(abs(variance - 1.0 / 3.0), 4.0 * math.sqrt(4.0 / 45 / 4000))
        posterior = ppl.sample(
            model, method="hmc", draws=400, warmup=200, chains=2, seed=3
        )
        self.assertEqual(posterior.gradient_method, "reverse")
        summary = {row.name: row for row in posterior.summary()}
        self.assertLess(abs(summary["R[0,1]"].mean), 4.0 * summary["R[0,1]"].mcse_mean)

    def test_cholesky_density_equals_independent_beta_cpcs(self) -> None:
        """The PPL density of an LKJ Cholesky site factorizes into Beta CPCs."""
        d, eta = 4, 1.7

        def model(m: ModelHandle) -> None:
            """Declare a single LKJ Cholesky site."""
            m.sample("L", LKJCholesky(d, eta))

        program = ppl.trace(model)
        u = (0.4, -1.2, 0.3, 0.8, -0.1, 1.5)
        expected = 0.0
        offset = 0
        for i in range(d):
            for j in range(i):
                b = eta + 0.5 * (d - 2 - j)
                z = math.tanh(u[offset])
                offset += 1
                expected += (
                    (b - 1.0) * math.log((1.0 - z * z) / 4.0)
                    - log_beta_function(b, b)
                    - math.log(2.0)
                    + math.log(1.0 - z * z)
                )
        self.assertAlmostEqual(program.log_density(u), expected, places=11)
        factor = program.constrain(u)["L"]
        self.assertAlmostEqual(
            LKJCholesky(d, eta).log_pdf(factor)
            + StructuredSupport("cholesky_correlation", d).forward(u).log_jacobian,
            expected,
            places=11,
        )

    def test_sampled_correlations_have_the_beta_marginal_variance(self) -> None:
        """Each ``r_ij`` of LKJ(eta) has variance ``1 / (2 eta + d - 1)``."""
        d, eta, count = 4, 2.0, 3000
        matrices = LKJCorrelation(d, eta).sample(count, seed=8)
        for i, j in ((1, 0), (3, 2), (3, 0)):
            values = [m[i][j] for m in matrices]
            variance = math.fsum(v * v for v in values) / count
            expected = 1.0 / (2.0 * eta + d - 1.0)
            tolerance = 4.0 * expected * math.sqrt(2.0 / count)
            self.assertLess(abs(variance - expected), tolerance)
        factor = LKJCholesky(3, 2.0).sample(1, seed=1)[0]
        self.assertTrue(math.isfinite(LKJCholesky(3, 2.0).log_pdf(factor)))

    def test_invalid_lkj_arguments(self) -> None:
        """Bad orders, shapes, and non-correlation matrices are rejected."""
        with self.assertRaises(ValueError):
            LKJCholesky(1, 1.0)
        with self.assertRaises(ValueError):
            LKJCorrelation(3, 0.0)
        with self.assertRaises(ValueError):
            LKJCorrelation(2, 1.0).log_pdf(((1.0, 0.2), (0.3, 1.0)))
        with self.assertRaises(ValueError):
            LKJCholesky(2, 1.0).log_pdf(((1.0, 0.0), (0.5, 0.5)))
        self.assertEqual(
            LKJCorrelation(2, 1.0).log_pdf(((1.0, 1.5), (1.5, 1.0))), -math.inf
        )


# ---------------------------------------------------------------------------
# Structured sites in models
# ---------------------------------------------------------------------------

COUNTS = (12, 5, 3)
ALPHA = (2.0, 1.5, 1.0)


def dirichlet_multinomial(m: ModelHandle) -> None:
    """Dirichlet prior with multinomial counts."""
    p = m.sample("p", Dirichlet(ALPHA))
    m.observe("y", Multinomial(sum(COUNTS), p), COUNTS)


class StructuredSiteTests(unittest.TestCase):
    """Structured latent sites with the inference engines."""

    def test_dirichlet_multinomial_matches_conjugate_posterior(self) -> None:
        """NUTS posterior means agree with ``dirichlet_posterior`` (4 MCSE)."""
        exact = dirichlet_posterior(Dirichlet(ALPHA), COUNTS).posterior.mean
        posterior = ppl.sample(
            dirichlet_multinomial, draws=400, warmup=300, chains=2, seed=3
        )
        summary = {row.name: row for row in posterior.summary()}
        for k in range(3):
            row = summary[f"p[{k}]"]
            self.assertLess(abs(row.mean - exact[k]), 4.0 * row.mcse_mean)
            self.assertLess(row.rank_normalized_rhat, 1.05)
        for draw in posterior["p"]:
            self.assertAlmostEqual(math.fsum(draw), 1.0, places=12)  # type: ignore[arg-type]
        laplace = ppl.laplace(dirichlet_multinomial, draws=200, seed=2)
        advi = ppl.fit_advi(dirichlet_multinomial, draws=200, seed=2)
        for result in (laplace, advi):
            mean = result.mean("p")
            assert isinstance(mean, tuple)
            for k in range(3):
                self.assertLess(abs(mean[k] - exact[k]), 0.05)

    def test_simplex_gradient_is_exact_at_the_origin(self) -> None:
        """The stick-breaking gradient at ``u = 0`` matches central differences.

        The last stick has offset ``log 1 = 0``, so ``t = 0`` there; a
        softplus written with ``|t|`` had derivative 0 instead of 1/2 at that
        point, the PPL start, and the Laplace mode search never moved the
        coordinate.
        """

        def prior_only(m: ModelHandle) -> None:
            """Declare a Dirichlet site with no data."""
            m.sample("p", Dirichlet(ALPHA))

        for model in (prior_only,):
            program = ppl.trace(model)
            for u in ((0.0, 0.0), (0.4, 0.0), (0.0, -0.7)):
                exact = program.gradient(u, method="reverse")
                forward = program.gradient(u, method="autodiff")
                finite = program.gradient(u, method="finite-difference")
                for a, b, c in zip(exact, forward, finite, strict=True):
                    self.assertAlmostEqual(a, c, delta=1e-7)
                    self.assertAlmostEqual(b, c, delta=1e-7)

    def test_dirichlet_density_is_exact(self) -> None:
        """Prior + likelihood + Jacobian against a direct evaluation."""
        program = ppl.trace(dirichlet_multinomial)
        self.assertEqual(program.dimension, 2)
        self.assertEqual(program.constrained_names, ("p[0]", "p[1]", "p[2]"))
        u = (0.2, -0.3)
        support = StructuredSupport("simplex", 3)
        point = support.forward(u)
        expected = (
            Dirichlet(ALPHA).log_pdf(point.value)
            + Multinomial(sum(COUNTS), point.value).log_pmf(COUNTS)
            + point.log_jacobian
        )
        self.assertAlmostEqual(program.log_density(u), expected, places=11)

    def test_structured_gradients_forward_reverse_and_finite_difference(self) -> None:
        """Reverse and forward mode agree with central differences."""

        def model(m: ModelHandle) -> None:
            """Every structured prior in one traceable model."""
            m.sample("w", Dirichlet((1.5, 2.0, 2.5)))
            m.sample("c", OrderedVector(Normal(0.0, 3.0), 3))
            m.sample("L", LKJCholesky(3, 2.0))
            m.sample("R", LKJCorrelation(3, 1.5))
            m.sample("S", Wishart(SCALE, 4.0))
            m.sample("T", InverseWishart(SCALE, 5.0))

        program = ppl.trace(model)
        self.assertEqual(program.dimension, 2 + 3 + 3 + 3 + 3 + 3)
        generator = random.Random(2)
        u = tuple(generator.uniform(-1.0, 1.0) for _ in range(program.dimension))
        reverse = program.gradient(u, method="reverse")
        forward = program.gradient(u, method="forward")
        finite = program.gradient(u, method="finite-difference")
        for a, b, c in zip(reverse, forward, finite, strict=True):
            self.assertAlmostEqual(a, b, places=10)
            self.assertAlmostEqual(a, c, delta=1e-5 * max(1.0, abs(a)))
        values = program.constrain(u)
        support = StructuredSupport("covariance", 2)
        self.assertAlmostEqual(
            InverseWishart(SCALE, 5.0).log_pdf(values["T"])
            + support.forward(u[-3:]).log_jacobian,
            program.log_density(u)
            - program.log_density(u[:-3] + (0.0, 0.0, 0.0))
            + InverseWishart(SCALE, 5.0).log_pdf(support.forward((0.0,) * 3).value)
            + support.forward((0.0,) * 3).log_jacobian,
            places=10,
        )

    def test_inverse_wishart_covariance_posterior(self) -> None:
        """NUTS mean of ``Sigma`` matches the conjugate ``(Psi + S) / (nu + n - 3)``."""
        generator = random.Random(9)
        data = []
        for _ in range(12):
            a = generator.gauss(0.0, 1.0)
            b = 0.6 * a + generator.gauss(0.0, 0.8)
            data.append((a, b))
        nu = 6.0

        def model(m: ModelHandle) -> None:
            """Zero-mean bivariate normal with an inverse-Wishart covariance."""
            sigma = m.sample("Sigma", InverseWishart(SCALE, nu))
            law = MultivariateNormal((0.0, 0.0), sigma)
            for i, row in enumerate(data):
                m.observe(f"y[{i}]", law, row)

        scatter = [
            [math.fsum(r[i] * r[j] for r in data) for j in range(2)] for i in range(2)
        ]
        exact = [
            [(SCALE[i][j] + scatter[i][j]) / (nu + len(data) - 3.0) for j in range(2)]
            for i in range(2)
        ]
        posterior = ppl.sample(model, draws=250, warmup=250, chains=2, seed=4)
        summary = {row.name: row for row in posterior.summary()}
        for i, j in ((0, 0), (1, 0), (1, 1)):
            row = summary[f"Sigma[{i},{j}]"]
            self.assertLess(abs(row.mean - exact[i][j]), 4.0 * row.mcse_mean)
        mean = posterior.mean("Sigma")
        assert isinstance(mean, tuple) and isinstance(mean[0], tuple)
        self.assertAlmostEqual(mean[0][1], mean[1][0], places=12)

    def test_predictive_simulation_with_structured_sites(self) -> None:
        """Prior predictive draws of structured sites have the right shapes."""

        def model(m: ModelHandle) -> None:
            """Structured priors and an observation."""
            w = m.sample("w", Dirichlet((1.0, 1.0, 1.0)))
            m.sample("S", Wishart(SCALE, 4.0))
            m.sample("c", OrderedVector(Normal(0.0, 1.0), 2))
            m.observe("y", Normal(w[0], 1.0), 0.3)

        draws = ppl.prior_predictive(model, 50, seed=1)
        for w in draws["w"]:
            self.assertAlmostEqual(math.fsum(w), 1.0)  # type: ignore[arg-type]
        for s in draws["S"]:
            assert isinstance(s, tuple) and isinstance(s[0], tuple)
            self.assertEqual(len(s), 2)
        for c in draws["c"]:
            self.assertLess(c[0], c[1])  # type: ignore[index]
        mean = draws.mean("S")
        assert isinstance(mean, tuple) and isinstance(mean[0], tuple)
        self.assertEqual(len(mean[0]), 2)


# ---------------------------------------------------------------------------
# Discrete sites
# ---------------------------------------------------------------------------

DATA_GENERATOR = random.Random(11)
MIXTURE_Y = tuple(
    DATA_GENERATOR.gauss(-2.0, 1.0)
    if DATA_GENERATOR.random() < 0.3
    else DATA_GENERATOR.gauss(2.0, 1.0)
    for _ in range(30)
)


class Indicator:
    """A traceable Bernoulli law (``log_pmf`` uses ``admath``)."""

    support = IntegerSupport(0, 1)

    def __init__(self, p: float) -> None:
        """Store the success probability."""
        self.p = p

    def log_pmf(self, k: int) -> float:
        """Return ``log p`` or ``log(1 - p)``."""
        return admath.log(self.p) if k == 1 else admath.log1p(-self.p)

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[int, ...]:
        """Draw indicators."""
        generator = random.Random(seed)
        return tuple(int(generator.random() < float(self.p)) for _ in range(size))


class UnitNormal:
    """A traceable normal law with unit scale."""

    support = RealSupport(None, None)

    def __init__(self, mu: float) -> None:
        """Store the mean."""
        self.mu = mu

    def log_pdf(self, x: float) -> float:
        """Return the log-density."""
        z = x - self.mu
        return -0.5 * z * z - LOG_SQRT_TWO_PI

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[float, ...]:
        """Draw values."""
        generator = random.Random(seed)
        return tuple(generator.gauss(float(self.mu), 1.0) for _ in range(size))


def mixture(m: ModelHandle) -> None:
    """Two-component normal mixture with marginalized indicators."""
    w = m.sample("w", Dirichlet((2.0, 2.0)))
    mu = m.sample("mu", OrderedVector(Normal(0.0, 5.0), 2))
    for i in m.plate("data", len(MIXTURE_Y)):
        z = m.sample(f"z[{i}]", Indicator(w[1]), marginalize=True)
        m.observe(f"y[{i}]", UnitNormal(mu[z]), MIXTURE_Y[i])


def mixture_log_likelihood(w: float, mu0: float, mu1: float) -> float:
    """Exact marginal log-likelihood of the mixture data."""
    total = 0.0
    for y in MIXTURE_Y:
        a = math.exp(-0.5 * (y - mu0) ** 2 - LOG_SQRT_TWO_PI)
        b = math.exp(-0.5 * (y - mu1) ** 2 - LOG_SQRT_TWO_PI)
        total += math.log((1.0 - w) * a + w * b)
    return total


def membership(w: float, mu0: float, mu1: float, y: float) -> float:
    """Exact ``P(z = 1 | w, mu, y)``."""
    a = (1.0 - w) * math.exp(-0.5 * (y - mu0) ** 2)
    b = w * math.exp(-0.5 * (y - mu1) ** 2)
    return b / (a + b)


def simpson_weights(count: int, step: float) -> list[float]:
    """Composite Simpson weights for ``count`` (odd) nodes."""
    weights = [2.0 if k % 2 == 0 else 4.0 for k in range(count)]
    weights[0] = weights[-1] = 1.0
    return [step / 3.0 * value for value in weights]


def exact_mixture_means() -> tuple[float, float, float]:
    """Posterior means of ``(w, mu0, mu1)`` by a Simpson grid."""
    n = 31
    grids = []
    for low, high in ((0.15, 0.97), (-4.5, 0.5), (0.3, 3.8)):
        step = (high - low) / (n - 1)
        grids.append(([low + k * step for k in range(n)], simpson_weights(n, step)))
    (ws, ww), (m0s, m0w), (m1s, m1w) = grids
    pdf0 = [[math.exp(-0.5 * (y - m) ** 2) for y in MIXTURE_Y] for m in m0s]
    pdf1 = [[math.exp(-0.5 * (y - m) ** 2) for y in MIXTURE_Y] for m in m1s]
    log_terms: list[tuple[float, float, float, float]] = []
    for w, weight_w in zip(ws, ww, strict=True):
        prior_w = math.log(6.0 * w * (1.0 - w) * weight_w)
        for a, (m0, weight0) in enumerate(zip(m0s, m0w, strict=True)):
            prior0 = -0.5 * (m0 / 5.0) ** 2 + math.log(weight0)
            for b, (m1, weight1) in enumerate(zip(m1s, m1w, strict=True)):
                if m1 <= m0:
                    continue
                total = prior_w + prior0 - 0.5 * (m1 / 5.0) ** 2 + math.log(weight1)
                for p0, p1 in zip(pdf0[a], pdf1[b], strict=True):
                    total += math.log((1.0 - w) * p0 + w * p1)
                log_terms.append((total, w, m0, m1))
    peak = max(t[0] for t in log_terms)
    mass = [math.exp(t[0] - peak) for t in log_terms]
    norm = math.fsum(mass)
    return tuple(  # type: ignore[return-value]
        math.fsum(p * t[k] for p, t in zip(mass, log_terms, strict=True)) / norm
        for k in (1, 2, 3)
    )


class DiscreteSiteTests(unittest.TestCase):
    """Enumeration of discrete sites and recovery of their posterior."""

    def test_plate_enumeration_density_is_exact(self) -> None:
        """The marginalized log-density equals the hand-written mixture density."""
        program = ppl.trace(mixture)
        self.assertEqual(program.dimension, 3)
        self.assertEqual(len(program.discrete_sites), len(MIXTURE_Y))
        u = (0.1, -1.0, 1.2)
        values = program.constrain(u)
        w = values["w"][1]  # type: ignore[index]
        mu0, mu1 = values["mu"]  # type: ignore[misc]
        simplex = StructuredSupport("simplex", 2).forward(u[:1])
        ordered = StructuredSupport("ordered", 2).forward(u[1:])
        expected = (
            Dirichlet((2.0, 2.0)).log_pdf(simplex.value)
            + simplex.log_jacobian
            + math.log(2.0)
            + Normal(0.0, 5.0).log_pdf(mu0)
            + Normal(0.0, 5.0).log_pdf(mu1)
            + ordered.log_jacobian
            + mixture_log_likelihood(w, mu0, mu1)
        )
        self.assertAlmostEqual(program.log_density(u), expected, places=10)
        gradient = program.gradient(u, method="reverse")
        finite = program.gradient(u, method="finite-difference")
        for a, b in zip(gradient, finite, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-5)

    def test_recovered_indicator_probabilities_are_exact_bayes(self) -> None:
        """Conditional enumeration reproduces ``P(z_i = 1 | theta, y_i)``."""
        program = ppl.trace(mixture)
        u = (0.4, -1.5, 1.0)
        values = program.constrain(u)
        w = values["w"][1]  # type: ignore[index]
        mu0, mu1 = values["mu"]  # type: ignore[misc]
        conditional = program.discrete_conditional(u)
        self.assertEqual(conditional.names, ())
        groups = conditional.groups[0]
        self.assertEqual(len(groups), len(MIXTURE_Y))
        for group, y in zip(groups, MIXTURE_Y, strict=True):
            weights = [math.exp(v) for v in group.log_weights]
            probability = weights[1] / (weights[0] + weights[1])
            self.assertAlmostEqual(probability, membership(w, mu0, mu1, y), places=12)

    def test_mixture_posterior_recovers_means_weights_and_indicators(self) -> None:
        """NUTS on the marginalized mixture agrees with the exact grid posterior."""
        posterior = ppl.sample(mixture, draws=200, warmup=200, chains=2, seed=5)
        self.assertEqual(posterior.gradient_method, "reverse")
        exact = exact_mixture_means()
        summary = {row.name: row for row in posterior.summary()}
        for name, value in zip(("w[1]", "mu[0]", "mu[1]"), exact, strict=True):
            row = summary[name]
            self.assertLess(abs(row.mean - value), 4.0 * row.mcse_mean + 2e-3, name)
            self.assertLess(row.rank_normalized_rhat, 1.05)
        assert posterior.discrete is not None
        draws = posterior.draws()
        for i in (0, 7, 19):
            name = f"z[{i}]"
            expected = math.fsum(
                membership(d["w"][1], d["mu"][0], d["mu"][1], MIXTURE_Y[i])  # type: ignore[index]
                for d in draws
            ) / len(draws)
            probabilities = posterior.discrete_probabilities(name)
            self.assertAlmostEqual(probabilities[1], expected, places=12)
            self.assertAlmostEqual(posterior.mean(name), expected, places=12)
            sampled = posterior[name]
            frequency = sum(sampled) / len(sampled)  # type: ignore[arg-type]
            spread = math.sqrt(max(expected * (1.0 - expected), 0.01) / len(sampled))
            self.assertLess(abs(frequency - expected), 5.0 * spread)
        replicated = ppl.posterior_predictive(mixture, posterior, seed=1, max_draws=20)
        self.assertEqual(len(replicated["y[0]"]), 20)
        self.assertEqual(replicated.kinds["z[0]"], "discrete")

    def test_global_discrete_site_and_truncated_poisson(self) -> None:
        """Joint enumeration of a truncated count matches the closed form."""
        k, upper, rate = 4, 20, 6.0
        law = ppl.TruncatedDiscrete(Poisson(rate), 0, upper)
        self.assertAlmostEqual(math.fsum(law.pmf(n) for n in range(upper + 1)), 1.0)

        def model(m: ModelHandle) -> None:
            """Binomial count with an unknown, truncated-Poisson number of trials."""
            p = m.sample("p", Beta(2.0, 2.0))
            n = m.sample("n", law, marginalize=True)
            m.observe("k", Binomial(n, p), k)

        program = ppl.trace(model)
        u = (-0.4,)
        p = 1.0 / (1.0 + math.exp(0.4))
        terms = [law.log_pmf(n) + Binomial(n, p).log_pmf(k) for n in range(upper + 1)]
        peak = max(terms)
        log_marginal = peak + math.log(math.fsum(math.exp(t - peak) for t in terms))
        jacobian = math.log(p) + math.log(1.0 - p)
        self.assertAlmostEqual(
            program.log_density(u),
            Beta(2.0, 2.0).log_pdf(p) + jacobian + log_marginal,
            places=11,
        )
        conditional = program.discrete_conditional(u)
        self.assertEqual(conditional.names, ("n",))
        total = math.fsum(math.exp(v) for v in conditional.log_weights)
        for n, value in enumerate(conditional.log_weights):
            self.assertAlmostEqual(
                math.exp(value) / total, math.exp(terms[n] - log_marginal)
            )
        # p(n | k) ∝ TP(n) C(n, k) B(2 + k, 2 + n - k) / B(2, 2).
        weights = []
        for n in range(upper + 1):
            if n < k:
                weights.append(0.0)
                continue
            log_weight = (
                law.log_pmf(n)
                + math.lgamma(n + 1)
                - math.lgamma(k + 1)
                - math.lgamma(n - k + 1)
                + log_beta_function(2.0 + k, 2.0 + n - k)
            )
            weights.append(math.exp(log_weight))
        exact_mean = math.fsum(n * w for n, w in enumerate(weights))
        exact_mean /= math.fsum(weights)
        posterior = ppl.sample(model, draws=250, warmup=150, chains=2, seed=6)
        rows = posterior.discrete.probabilities["n"]  # type: ignore[union-attr]
        conditional_means = [
            math.fsum(n * q for n, q in enumerate(row)) for row in rows
        ]
        center = math.fsum(conditional_means) / len(conditional_means)
        spread = math.sqrt(
            math.fsum((c - center) ** 2 for c in conditional_means) / len(rows)
        )
        ess = posterior.summary()[0].effective_sample_size
        self.assertAlmostEqual(posterior.mean("n"), center, places=12)
        self.assertLess(abs(center - exact_mean), 4.0 * spread / math.sqrt(ess) + 1e-3)
        self.assertTrue(all(n >= k for n in posterior["n"]))  # type: ignore[operator]

    def test_categorical_labels_and_terms(self) -> None:
        """A global Categorical site with string labels and ``terms``."""
        from probstat.distributions import Categorical

        data = (0.4, 1.9, 2.2)

        def model(m: ModelHandle) -> None:
            """Let a regime label choose the observation mean."""
            mu = m.sample("mu", Normal(0.0, 3.0))
            regime = m.sample(
                "regime", Categorical({"low": 0.6, "high": 0.4}), marginalize=True
            )
            shift = 0.0 if regime == "low" else 2.0
            m.observe("y", Normal(mu + shift, 1.0), data[0])
            for i in m.plate("rest", 2):
                m.observe(f"r[{i}]", Normal(mu + shift, 1.0), data[i + 1])

        program = ppl.trace(model)
        mu = 0.3
        branch = [
            math.log(prior)
            + math.fsum(Normal(mu + shift, 1.0).log_pdf(y) for y in data)
            for prior, shift in ((0.6, 0.0), (0.4, 2.0))
        ]
        peak = max(branch)
        marginal = peak + math.log(math.fsum(math.exp(b - peak) for b in branch))
        terms = program.terms((mu,))
        self.assertAlmostEqual(terms.log_likelihood, marginal, places=12)
        self.assertAlmostEqual(
            program.log_density((mu,)),
            Normal(0.0, 3.0).log_pdf(mu) + marginal,
            places=12,
        )
        posterior = ppl.laplace(model, draws=100, seed=1)
        probabilities = posterior.discrete_probabilities("regime")
        self.assertEqual(set(probabilities), {"low", "high"})
        self.assertAlmostEqual(math.fsum(probabilities.values()), 1.0)
        self.assertTrue(set(posterior["regime"]) <= {"low", "high"})

    def test_invalid_models_raise(self) -> None:
        """Unbounded, misplaced, and malformed sites are rejected.

        Nested plates, vector-valued discrete sites, and continuous sites after
        a global discrete site are supported since ADR-0084
        (tests/test_ppl_dynamic_structure.py).
        """

        def unbounded(m: ModelHandle) -> None:
            """Declare a Poisson latent count without truncation."""
            m.sample("mu", Normal(0.0, 1.0))
            m.sample("n", Poisson(3.0), marginalize=True)

        def continuous_marginalized(m: ModelHandle) -> None:
            """``marginalize`` on a continuous site."""
            m.sample("x", Normal(0.0, 1.0), marginalize=True)

        def local_latent_after_discrete(m: ModelHandle) -> None:
            """Declare a deterministic site after a discrete one in a plate."""
            for i in m.plate("data", 2):
                m.sample(f"z[{i}]", Bernoulli(0.5), marginalize=True)
                m.deterministic(f"d[{i}]", 1.0)

        def too_many(m: ModelHandle) -> None:
            """Thirteen global binary sites (8192 configurations)."""
            m.sample("x", Normal(0.0, 1.0))
            for i in range(13):
                m.sample(f"z{i}", Bernoulli(0.5), marginalize=True)

        def structured_shape(m: ModelHandle) -> None:
            """Give a structured site a shape."""
            m.sample("p", Dirichlet((1.0, 1.0)), shape=2)

        def interval_ordered(m: ModelHandle) -> None:
            """Order draws of a bounded base law."""
            m.sample("c", OrderedVector(Uniform(0.0, 1.0), 2))

        def unmarked(m: ModelHandle) -> None:
            """Declare a bounded discrete site without ``marginalize``."""
            m.sample("k", Binomial(5, 0.5))

        cases = (
            (unbounded, "unbounded"),
            (continuous_marginalized, "finite discrete support"),
            (local_latent_after_discrete, "follows a discrete latent site"),
            (too_many, "at most 4096"),
            (structured_shape, "do not apply"),
            (interval_ordered, "OrderedVector needs"),
            (unmarked, "marginalize=True"),
        )
        for model, message in cases:
            with self.subTest(model=model.__name__):
                with self.assertRaisesRegex(ModelSpecificationError, message):
                    ppl.trace(model)
        with self.assertRaises(ValueError):
            ppl.TruncatedDiscrete(Poisson(1.0), 5, 2)
        with self.assertRaises(ValueError):
            OrderedVector(Normal(0.0, 1.0), 1)
        with self.assertRaisesRegex(ModelSpecificationError, "no discrete sites"):
            ppl.trace(dirichlet_multinomial).discrete_conditional((0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
