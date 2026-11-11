"""Dynamic site structure, nested plates, and vector-valued discrete PPL sites."""

from __future__ import annotations

import itertools
import math
import random
import unittest

from probstat.distributions import (
    Bernoulli,
    Categorical,
    IntegerSupport,
    Normal,
    Poisson,
    RealSupport,
)
from probstat.inference import effective_sample_size, ppl
from probstat.inference.ppl import ModelHandle, ModelSpecificationError
from probstat.multivariate import Multinomial
from probstat.numerics import admath

LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)


def log_phi(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Return the normal log-density."""
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma) - LOG_SQRT_TWO_PI


def log_sum_exp(values: list[float]) -> float:
    """Return ``log sum exp`` of floats."""
    peak = max(values)
    return peak + math.log(math.fsum(math.exp(v - peak) for v in values))


def simpson(low: float, high: float, count: int) -> tuple[list[float], list[float]]:
    """Return composite Simpson nodes and weights (``count`` odd)."""
    step = (high - low) / (count - 1)
    nodes = [low + k * step for k in range(count)]
    weights = [
        step / 3.0 * (1.0 if k in (0, count - 1) else (4.0 if k % 2 else 2.0))
        for k in range(count)
    ]
    return nodes, weights


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


class Gaussian:
    """A traceable normal law."""

    support = RealSupport(None, None)

    def __init__(self, mu: float, sigma: float = 1.0) -> None:
        """Store the mean and the scale."""
        self.mu = mu
        self.sigma = sigma

    def log_pdf(self, x: float) -> float:
        """Return the log-density."""
        z = (x - self.mu) / self.sigma
        return -0.5 * z * z - math.log(self.sigma) - LOG_SQRT_TWO_PI

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[float, ...]:
        """Draw values."""
        generator = random.Random(seed)
        return tuple(generator.gauss(float(self.mu), self.sigma) for _ in range(size))


# ---------------------------------------------------------------------------
# Dynamic structure: one or two components
# ---------------------------------------------------------------------------

DATA = random.Random(3)
COMPONENT_Y = tuple(
    [DATA.gauss(0.0, 1.0) for _ in range(12)] + [DATA.gauss(3.0, 1.0) for _ in range(3)]
)
WEIGHT = 0.2


def components(m: ModelHandle) -> None:
    """Choose one or two unit-variance components with a marginalized ``k``.

    With ``k = 2`` a second mean and one indicator per observation exist;
    with ``k = 1`` neither does.
    """
    k = m.sample("k", Categorical({1: 0.5, 2: 0.5}), marginalize=True)
    mu0 = m.sample("mu[0]", Normal(0.0, 2.0))
    mu1 = m.sample("mu[1]", Normal(3.0, 1.0)) if k == 2 else 0.0
    for i in m.plate("data", len(COMPONENT_Y)):
        if k == 1:
            m.observe(f"y[{i}]", Gaussian(mu0), COMPONENT_Y[i])
        else:
            z = m.sample(f"z[{i}]", Indicator(WEIGHT), marginalize=True)
            m.observe(f"y[{i}]", Gaussian(mu1 if z == 1 else mu0), COMPONENT_Y[i])


def component_log_joint(mu0: float, mu1: float) -> tuple[float, float]:
    """Return ``log p(k, mu, y)`` for ``k = 1`` (mu1 padded) and ``k = 2``."""
    prior = math.log(0.5) + log_phi(mu0, 0.0, 2.0) + log_phi(mu1, 3.0, 1.0)
    one = prior + math.fsum(log_phi(y, mu0) for y in COMPONENT_Y)
    two = prior + math.fsum(
        math.log(
            (1.0 - WEIGHT) * math.exp(log_phi(y, mu0))
            + WEIGHT * math.exp(log_phi(y, mu1))
        )
        for y in COMPONENT_Y
    )
    return one, two


def component_marginal_likelihoods() -> tuple[float, float]:
    """Return ``p(y | k = 1)`` and ``p(y | k = 2)`` by Simpson quadrature."""
    nodes, weights = simpson(-4.0, 6.0, 801)
    one = math.fsum(
        w
        * math.exp(log_phi(x, 0.0, 2.0) + math.fsum(log_phi(y, x) for y in COMPONENT_Y))
        for x, w in zip(nodes, weights, strict=True)
    )
    a, wa = simpson(-3.0, 3.0, 301)
    b, wb = simpson(-2.0, 7.0, 301)
    density1 = [[math.exp(log_phi(y, x)) for y in COMPONENT_Y] for x in b]
    two = 0.0
    for x0, w0 in zip(a, wa, strict=True):
        density0 = [math.exp(log_phi(y, x0)) for y in COMPONENT_Y]
        prior0 = math.exp(log_phi(x0, 0.0, 2.0))
        for x1, w1, row in zip(b, wb, density1, strict=True):
            likelihood = math.prod(
                (1.0 - WEIGHT) * p0 + WEIGHT * p1
                for p0, p1 in zip(density0, row, strict=True)
            )
            two += w0 * w1 * prior0 * math.exp(log_phi(x1, 3.0, 1.0)) * likelihood
    return one, two


class DynamicStructureTests(unittest.TestCase):
    """A discrete site that decides which continuous sites exist."""

    @classmethod
    def setUpClass(cls) -> None:
        """Compute the exact marginal likelihoods once."""
        one, two = component_marginal_likelihoods()
        cls.evidence = (one, two)
        cls.exact_two = two / (one + two)

    def test_traced_structures_and_padded_density(self) -> None:
        """Both structures are traced; the padded density is the hand mixture."""
        program = ppl.trace(components)
        self.assertTrue(program.dynamic)
        self.assertTrue(program.per_configuration)
        self.assertEqual(program.unconstrained_names, ("mu[0]", "mu[1]"))
        one, two = program.structures
        self.assertEqual(one.choices, (("k", 1),))
        self.assertEqual(two.choices, (("k", 2),))
        self.assertEqual(one.padded, ("mu[1]",))
        self.assertEqual(two.padded, ())
        self.assertNotIn("z[0]", one.names)
        self.assertIn("z[0]", two.names)
        self.assertEqual(program.presence("mu[1]"), frozenset({1}))
        for u in ((0.3, 2.5), (-1.0, 0.0), (1.2, 4.1)):
            log_one, log_two = component_log_joint(*u)
            self.assertAlmostEqual(
                program.log_density(u), log_sum_exp([log_one, log_two]), places=10
            )
            conditional = program.discrete_conditional(u)
            self.assertEqual(conditional.names, ("k",))
            weights = [math.exp(w) for w in conditional.log_weights]
            self.assertAlmostEqual(
                weights[1] / sum(weights),
                1.0 / (1.0 + math.exp(log_one - log_two)),
                places=12,
            )
            self.assertEqual(conditional.groups[0], ())
            self.assertEqual(len(conditional.groups[1]), len(COMPONENT_Y))
        reverse = program.gradient((0.3, 2.5), method="reverse")
        finite = program.gradient((0.3, 2.5), method="finite-difference")
        for a, b in zip(reverse, finite, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-5)
        terms = program.terms((0.3, 2.5))
        self.assertAlmostEqual(terms.total, program.log_density((0.3, 2.5)), places=10)

    def test_padding_integrates_out(self) -> None:
        """Quadrature of the PPL density gives the evidence and ``p(k | y)``.

        The pseudo-prior of ``mu[1]`` under ``k = 1`` integrates to one, so
        ``int p(u) du = sum_k p(k) p(y | k)`` and the integral of the
        configuration weights gives the exact posterior model probability.
        """
        program = ppl.trace(components)
        a, wa = simpson(-1.5, 2.5, 81)
        b, wb = simpson(-1.0, 7.0, 81)
        total = 0.0
        two = 0.0
        for x0, w0 in zip(a, wa, strict=True):
            for x1, w1 in zip(b, wb, strict=True):
                weights = program.discrete_conditional((x0, x1)).log_weights
                mass = [math.exp(v) for v in weights]
                total += w0 * w1 * (mass[0] + mass[1])
                two += w0 * w1 * mass[1]
        one_exact, two_exact = self.evidence
        self.assertAlmostEqual(total / (0.5 * (one_exact + two_exact)), 1.0, places=4)
        self.assertAlmostEqual(two / total, self.exact_two, places=4)

    def test_posterior_model_probability_matches_marginal_likelihood_ratio(
        self,
    ) -> None:
        """NUTS on the padded density recovers ``p(k = 2 | y)``."""
        posterior = ppl.sample(components, draws=300, warmup=300, chains=2, seed=7)
        self.assertEqual(posterior.gradient_method, "reverse")
        assert posterior.discrete is not None
        probabilities = posterior.discrete_probabilities("k")
        self.assertAlmostEqual(math.fsum(probabilities.values()), 1.0, places=12)
        rows = posterior.discrete.probabilities["k"]
        per_draw = [row[1] for row in rows]
        half = len(per_draw) // 2
        ess = effective_sample_size((per_draw[:half], per_draw[half:]))
        mean = math.fsum(per_draw) / len(per_draw)
        spread = math.sqrt(
            math.fsum((p - mean) ** 2 for p in per_draw) / len(per_draw) / ess
        )
        self.assertAlmostEqual(probabilities[2], mean, places=12)
        self.assertLess(abs(probabilities[2] - self.exact_two), 4.0 * spread + 1e-3)
        self.assertAlmostEqual(
            posterior.presence_probability("mu[1]"), probabilities[2], places=12
        )
        present = posterior.present("mu[1]")
        self.assertEqual(sum(present), sum(1 for k in posterior["k"] if k == 2))
        draws = posterior["mu[1]"]
        self.assertTrue(
            all((d is None) == (not p) for d, p in zip(draws, present, strict=True))
        )
        self.assertTrue(
            all(
                posterior["z[0]"][s] is None
                for s in range(len(present))
                if not present[s]
            )
        )
        for draw in posterior.draws():
            self.assertEqual("mu[1]" in draw, draw["k"] == 2)
            self.assertEqual("z[0]" in draw, draw["k"] == 2)
        self.assertTrue(math.isfinite(posterior.mean("mu[1]")))  # type: ignore[arg-type]
        replicated = ppl.posterior_predictive(
            components, posterior, seed=2, max_draws=30
        )
        self.assertEqual(len(replicated["y[0]"]), 30)
        prior = ppl.prior_predictive(components, 60, seed=3)
        absent = sum(1 for k in prior["k"] if k == 1)
        self.assertEqual(sum(1 for v in prior["mu[1]"] if v is None), absent)
        self.assertGreater(absent, 0)
        self.assertLess(absent, 60)

    def test_continuous_site_after_global_discrete_site(self) -> None:
        """A continuous site whose law depends on a global discrete value."""
        y = 1.4

        def model(m: ModelHandle) -> None:
            """``z ~ Bernoulli(0.3)``, ``x ~ N(2 z, 1)``, ``y ~ N(x, 0.5)``."""
            z = m.sample("z", Indicator(0.3), marginalize=True)
            x = m.sample("x", Normal(2.0 * z, 1.0))
            m.deterministic("shift", x - 2.0 * z)
            m.observe("y", Gaussian(x, 0.5), y)

        program = ppl.trace(model)
        self.assertFalse(program.dynamic)
        self.assertTrue(program.per_configuration)
        for x in (-0.5, 1.0, 2.2):
            terms = [
                math.log(p) + log_phi(x, 2.0 * z, 1.0) + log_phi(y, x, 0.5)
                for z, p in ((0, 0.7), (1, 0.3))
            ]
            self.assertAlmostEqual(
                program.log_density((x,)), log_sum_exp(terms), places=12
            )
        # y | z ~ N(2 z, 1 + 0.25), so p(z = 1 | y) has a closed form.
        evidence = [
            math.log(p) + log_phi(y, 2.0 * z, math.sqrt(1.25))
            for z, p in ((0, 0.7), (1, 0.3))
        ]
        exact = 1.0 / (1.0 + math.exp(evidence[0] - evidence[1]))
        posterior = ppl.sample(model, draws=400, warmup=300, chains=2, seed=4)
        assert posterior.discrete is not None
        per_draw = [row[1] for row in posterior.discrete.probabilities["z"]]
        half = len(per_draw) // 2
        ess = effective_sample_size((per_draw[:half], per_draw[half:]))
        mean = math.fsum(per_draw) / len(per_draw)
        spread = math.sqrt(
            math.fsum((p - mean) ** 2 for p in per_draw) / len(per_draw) / ess
        )
        estimate = posterior.discrete_probabilities("z")[1]
        self.assertLess(abs(estimate - exact), 4.0 * spread + 1e-3)
        # Values are reported under the drawn configuration of each draw.
        for shift, x, z in zip(
            posterior["shift"], posterior["x"], posterior["z"], strict=True
        ):
            self.assertAlmostEqual(shift, x - 2.0 * z, places=12)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Nested plates
# ---------------------------------------------------------------------------

GROUPS = ((0.1, -0.4), (2.3, 1.7, 2.9), (-0.2, 5.5, 0.4, 0.3))
LABEL_P = 0.4
OUTLIER_P = 0.1


def nested(m: ModelHandle) -> None:
    """Give groups a discrete label and observations an outlier indicator."""
    mu = m.sample("mu", ppl.OrderedVector(Normal(0.0, 3.0), 2))
    for g in m.plate("groups", len(GROUPS)):
        b = m.sample(f"b[{g}]", Normal(0.0, 0.5))
        c = m.sample(f"c[{g}]", Indicator(LABEL_P), marginalize=True)
        for j in m.plate("obs", len(GROUPS[g])):
            o = m.sample(f"o[{g},{j}]", Indicator(OUTLIER_P), marginalize=True)
            m.observe(
                f"y[{g},{j}]", Gaussian(mu[c] + b, 1.0 if o == 0 else 4.0), GROUPS[g][j]
            )


def observation_term(y: float, mean: float, outlier: int) -> float:
    """Return ``log p(o) + log N(y; mean, s_o)``."""
    mass = OUTLIER_P if outlier else 1.0 - OUTLIER_P
    return math.log(mass) + log_phi(y, mean, 4.0 if outlier else 1.0)


def label_mass(label: int) -> float:
    """Return ``log p(c)``."""
    return math.log(LABEL_P if label else 1.0 - LABEL_P)


class NestedPlateTests(unittest.TestCase):
    """Per-group and per-observation discrete sites in nested plates."""

    U = (0.4, 0.8, 0.3, -0.2, 0.5)

    def values(self) -> tuple[tuple[float, float], tuple[float, ...]]:
        """Return ``mu`` and the group offsets at :attr:`U`."""
        program = ppl.trace(nested)
        values = program.constrain(self.U)
        mu = values["mu"]
        b = tuple(values[f"b[{g}]"] for g in range(len(GROUPS)))
        return mu, b  # type: ignore[return-value]

    def test_log_density_equals_hand_enumeration(self) -> None:
        """Nested enumeration equals the brute-force sum over 4096 configurations."""
        program = ppl.trace(nested)
        self.assertEqual(program.dimension, 2 + len(GROUPS))
        self.assertEqual(program.site("o[2,3]").plates, ("groups", "obs"))
        self.assertEqual(program.site("c[1]").plates, ("groups",))
        mu, b = self.values()
        prior_terms = program.terms(self.U)
        nested_sum = math.fsum(
            log_sum_exp(
                [
                    label_mass(c)
                    + math.fsum(
                        log_sum_exp(
                            [observation_term(y, mu[c] + b[g], o) for o in (0, 1)]
                        )
                        for y in ys
                    )
                    for c in (0, 1)
                ]
            )
            for g, ys in enumerate(GROUPS)
        )
        self.assertAlmostEqual(prior_terms.log_likelihood, nested_sum, places=11)
        cells = [(g, j) for g, ys in enumerate(GROUPS) for j in range(len(ys))]
        brute: list[float] = []
        for labels in itertools.product((0, 1), repeat=len(GROUPS)):
            for outliers in itertools.product((0, 1), repeat=len(cells)):
                brute.append(
                    math.fsum(label_mass(c) for c in labels)
                    + math.fsum(
                        observation_term(GROUPS[g][j], mu[labels[g]] + b[g], o)
                        for (g, j), o in zip(cells, outliers, strict=True)
                    )
                )
        self.assertEqual(len(brute), 4096)
        self.assertAlmostEqual(
            prior_terms.log_likelihood, log_sum_exp(brute), places=10
        )
        self.assertAlmostEqual(
            program.log_density(self.U),
            prior_terms.log_prior + prior_terms.log_jacobian + log_sum_exp(brute),
            places=10,
        )
        reverse = program.gradient(self.U, method="reverse")
        finite = program.gradient(self.U, method="finite-difference")
        for a, c in zip(reverse, finite, strict=True):
            self.assertAlmostEqual(a, c, delta=1e-5)

    def test_recovered_label_and_outlier_probabilities_are_exact_bayes(self) -> None:
        """Group labels and nested indicators match exact Bayes."""
        program = ppl.trace(nested)
        mu, b = self.values()
        conditional = program.discrete_conditional(self.U)
        groups = conditional.groups[0]
        self.assertEqual(len(groups), len(GROUPS))
        for g, (group, ys) in enumerate(zip(groups, GROUPS, strict=True)):
            branch = [
                label_mass(c)
                + math.fsum(
                    log_sum_exp([observation_term(y, mu[c] + b[g], o) for o in (0, 1)])
                    for y in ys
                )
                for c in (0, 1)
            ]
            exact_label = 1.0 / (1.0 + math.exp(branch[0] - branch[1]))
            weights = [math.exp(v) for v in group.log_weights]
            self.assertAlmostEqual(weights[1] / sum(weights), exact_label, places=12)
            self.assertEqual(len(group.children), 2)
            for c in (0, 1):
                self.assertEqual(len(group.children[c]), len(ys))
                for child, y in zip(group.children[c], ys, strict=True):
                    local = [observation_term(y, mu[c] + b[g], o) for o in (0, 1)]
                    exact = 1.0 / (1.0 + math.exp(local[0] - local[1]))
                    mass = [math.exp(v) for v in child.log_weights]
                    self.assertAlmostEqual(mass[1] / sum(mass), exact, places=12)
        from probstat.inference.ppl._discrete import recover_discrete

        recovered = recover_discrete([conditional] * 4000, 9)
        g, j = 2, 1
        branch = [
            label_mass(c)
            + math.fsum(
                log_sum_exp([observation_term(y, mu[c] + b[g], o) for o in (0, 1)])
                for y in GROUPS[g]
            )
            for c in (0, 1)
        ]
        label = 1.0 / (1.0 + math.exp(branch[0] - branch[1]))
        outlier = 0.0
        for c, weight in ((0, 1.0 - label), (1, label)):
            local = [observation_term(GROUPS[g][j], mu[c] + b[g], o) for o in (0, 1)]
            outlier += weight / (1.0 + math.exp(local[0] - local[1]))
        self.assertAlmostEqual(recovered.marginal(f"o[{g},{j}]")[1], outlier, places=12)
        self.assertAlmostEqual(recovered.marginal(f"c[{g}]")[1], label, places=12)
        frequency = sum(recovered.draws[f"o[{g},{j}]"]) / 4000
        self.assertLess(abs(frequency - outlier), 5.0 * math.sqrt(0.25 / 4000))

    def test_nested_plates_without_outer_discrete_site(self) -> None:
        """Independent inner iterations under a plain outer plate."""
        data = ((0.5, 1.5), (2.0,))

        def model(m: ModelHandle) -> None:
            """Per-observation indicators two plates deep."""
            mu = m.sample("mu", Normal(0.0, 1.0))
            for g in m.plate("groups", len(data)):
                for j in m.plate("obs", len(data[g])):
                    z = m.sample(f"z[{g},{j}]", Indicator(0.3), marginalize=True)
                    m.observe(f"y[{g},{j}]", Gaussian(mu + 2.0 * z), data[g][j])

        program = ppl.trace(model)
        mu = 0.7
        expected = log_phi(mu) + math.fsum(
            log_sum_exp(
                [math.log(0.7) + log_phi(y, mu), math.log(0.3) + log_phi(y, mu + 2.0)]
            )
            for ys in data
            for y in ys
        )
        self.assertAlmostEqual(program.log_density((mu,)), expected, places=12)
        conditional = program.discrete_conditional((mu,))
        self.assertEqual(len(conditional.groups[0]), 3)


# ---------------------------------------------------------------------------
# Vector-valued discrete sites
# ---------------------------------------------------------------------------

LOADINGS = (1.0, 2.0, -1.5)
PRIORS = (0.2, 0.5, 0.7)
COUNT_P = (0.2, 0.3, 0.5)
Y_VECTOR = 1.3
X_COUNTS = 0.4


def vector_model(m: ModelHandle) -> None:
    """Independent indicators as one site and a multinomial count vector."""
    mu = m.sample("mu", Normal(0.0, 2.0))
    z = m.sample("z", [Indicator(p) for p in PRIORS], marginalize=True)
    shift = math.fsum(a * v for a, v in zip(LOADINGS, z, strict=True))
    m.observe("y", Gaussian(mu + shift, 0.7), Y_VECTOR)
    n = m.sample("n", Multinomial(3, COUNT_P), marginalize=True)
    m.observe("x", Gaussian(mu + 0.5 * n[0] - 0.5 * n[2], 1.0), X_COUNTS)


def indicator_terms(mu: float) -> dict[tuple[int, ...], float]:
    """Return ``log p(z) + log p(y | mu, z)`` for every indicator vector."""
    result = {}
    for z in itertools.product((0, 1), repeat=3):
        mass = math.fsum(
            math.log(p if v else 1.0 - p) for p, v in zip(PRIORS, z, strict=True)
        )
        shift = math.fsum(a * v for a, v in zip(LOADINGS, z, strict=True))
        result[z] = mass + log_phi(Y_VECTOR, mu + shift, 0.7)
    return result


def count_terms(mu: float) -> dict[tuple[int, ...], float]:
    """Return ``log p(n) + log p(x | mu, n)`` for every count vector."""
    result = {}
    for n0 in range(4):
        for n1 in range(4 - n0):
            n = (n0, n1, 3 - n0 - n1)
            log_mass = (
                math.lgamma(4)
                - math.fsum(math.lgamma(v + 1) for v in n)
                + math.fsum(v * math.log(p) for v, p in zip(n, COUNT_P, strict=True))
            )
            result[n] = log_mass + log_phi(X_COUNTS, mu + 0.5 * n[0] - 0.5 * n[2])
    return result


class VectorDiscreteTests(unittest.TestCase):
    """Joint enumeration of vector-valued discrete sites."""

    def test_log_density_and_conditional_probabilities(self) -> None:
        """Product and multinomial supports equal exact Bayes."""
        program = ppl.trace(vector_model)
        self.assertEqual(program.site("z").size, 3)
        self.assertEqual(len(program.site("z").enumeration), 8)
        self.assertEqual(len(program.site("n").enumeration), 10)
        self.assertEqual(len(program.structures), 80)
        self.assertFalse(program.per_configuration)
        mu = 0.35
        z_terms = indicator_terms(mu)
        n_terms = count_terms(mu)
        expected = (
            log_phi(mu, 0.0, 2.0)
            + log_sum_exp(list(z_terms.values()))
            + log_sum_exp(list(n_terms.values()))
        )
        self.assertAlmostEqual(program.log_density((mu,)), expected, places=11)
        conditional = program.discrete_conditional((mu,))
        from probstat.inference.ppl._discrete import recover_discrete

        draws = recover_discrete([conditional], 1)
        z_norm = log_sum_exp(list(z_terms.values()))
        for value, probability in zip(
            draws.supports["z"], draws.probabilities["z"][0], strict=True
        ):
            self.assertAlmostEqual(
                probability, math.exp(z_terms[value] - z_norm), places=12
            )
        n_norm = log_sum_exp(list(n_terms.values()))
        for value, probability in zip(
            draws.supports["n"], draws.probabilities["n"][0], strict=True
        ):
            self.assertAlmostEqual(
                probability, math.exp(n_terms[value] - n_norm), places=12
            )

    def test_posterior_recovery_of_vector_sites(self) -> None:
        """Rao-Blackwellized joint and elementwise probabilities over draws."""
        posterior = ppl.laplace(vector_model, draws=300, seed=5)
        mus = posterior["mu"]
        exact: dict[tuple[int, ...], float] = {}
        for mu in mus:
            terms = indicator_terms(float(mu))  # type: ignore[arg-type]
            norm = log_sum_exp(list(terms.values()))
            for z, value in terms.items():
                exact[z] = exact.get(z, 0.0) + math.exp(value - norm) / len(mus)
        probabilities = posterior.discrete_probabilities("z")
        for z, value in exact.items():
            self.assertAlmostEqual(probabilities[z], value, places=12)
        mean = posterior.mean("z")
        for k in range(3):
            marginal = math.fsum(p for z, p in exact.items() if z[k] == 1)
            self.assertAlmostEqual(mean[k], marginal, places=12)  # type: ignore[index]
        sampled = posterior["z"]
        self.assertTrue(all(isinstance(v, tuple) and len(v) == 3 for v in sampled))
        counts = posterior["n"]
        self.assertTrue(all(sum(v) == 3 for v in counts))  # type: ignore[arg-type]
        replicated = ppl.posterior_predictive(
            vector_model, posterior, seed=1, max_draws=10
        )
        self.assertEqual(len(replicated["z"]), 10)

    def test_vector_site_in_a_plate(self) -> None:
        """``shape=2`` indicators enumerated per plate iteration."""
        data = (0.2, 2.1, 3.9)

        def model(m: ModelHandle) -> None:
            """Two indicators per observation shift its mean."""
            mu = m.sample("mu", Normal(0.0, 1.0))
            for i in m.plate("data", len(data)):
                z = m.sample(f"z[{i}]", Indicator(0.4), shape=2, marginalize=True)
                m.observe(f"y[{i}]", Gaussian(mu + 2.0 * (z[0] + z[1])), data[i])

        program = ppl.trace(model)
        mu = -0.1
        expected = log_phi(mu) + math.fsum(
            log_sum_exp(
                [
                    math.log(0.4 if a else 0.6)
                    + math.log(0.4 if b else 0.6)
                    + log_phi(y, mu + 2.0 * (a + b))
                    for a in (0, 1)
                    for b in (0, 1)
                ]
            )
            for y in data
        )
        self.assertAlmostEqual(program.log_density((mu,)), expected, places=12)


# ---------------------------------------------------------------------------
# Data-dependent structure and refused structures
# ---------------------------------------------------------------------------


class StructureBoundaryTests(unittest.TestCase):
    """Structure from data is accepted; unsupported structure is refused."""

    def test_structure_from_data(self) -> None:
        """A loop whose length is set by data supplied at run time."""

        def groups(m: ModelHandle) -> None:
            """One latent mean per group of the conditioned data."""
            sizes = m.sample("sizes", Categorical({(2, 3): 1.0}), marginalize=True)
            for g, size in enumerate(sizes):
                theta = m.sample(f"theta[{g}]", Normal(0.0, 1.0))
                for j in m.plate(f"obs{g}", size):
                    m.observe(f"y[{g},{j}]", Gaussian(theta), 0.1 * j)

        program = ppl.trace(groups)
        self.assertEqual(program.dimension, 2)
        self.assertEqual(len(program.structures), 1)

        def by_data(m: ModelHandle, data: tuple[tuple[float, ...], ...]) -> None:
            """Group means whose number and plate sizes follow the data."""
            for g, values in enumerate(data):
                theta = m.sample(f"theta[{g}]", Normal(0.0, 1.0))
                for j in m.plate(f"group{g}", len(values)):
                    m.observe(f"y[{g},{j}]", Gaussian(theta), values[j])

        small = ppl.trace(lambda m: by_data(m, ((0.5, 0.1), (1.0,))))
        large = ppl.trace(lambda m: by_data(m, ((0.5,), (1.0,), (2.0, 3.0, 1.0))))
        self.assertEqual((small.dimension, large.dimension), (2, 3))
        self.assertEqual(len(large.observed_sites), 5)

    def test_unsupported_structures_raise(self) -> None:
        """Value-dependent, local, unbounded, and conflicting structures."""

        def continuous_branch(m: ModelHandle) -> None:
            """Declare a site only for some continuous values."""
            mu = m.sample("mu", Normal(0.0, 1.0))
            if mu > 0.5:
                m.sample("extra", Normal(0.0, 1.0))

        def local_branch(m: ModelHandle) -> None:
            """Declare an observation only for one value of a local site."""
            mu = m.sample("mu", Normal(0.0, 1.0))
            for i in m.plate("data", 2):
                z = m.sample(f"z[{i}]", Bernoulli(0.5), marginalize=True)
                if z == 1:
                    m.observe(f"y[{i}]", Gaussian(mu), 0.3)

        def local_continuous(m: ModelHandle) -> None:
            """Declare a continuous site after a local discrete site."""
            for i in m.plate("data", 2):
                z = m.sample(f"z[{i}]", Bernoulli(0.5), marginalize=True)
                m.sample(f"x[{i}]", Normal(float(z), 1.0))

        def unbounded_count(m: ModelHandle) -> None:
            """Draw a Poisson number of components."""
            k = m.sample("k", Poisson(2.0), marginalize=True)
            for j in range(k):
                m.sample(f"mu[{j}]", Normal(0.0, 1.0))

        def geometric(m: ModelHandle) -> None:
            """Flip until a one: infinitely many structures."""
            count = 0
            while m.sample(f"flip[{count}]", Bernoulli(0.5), marginalize=True) == 1:
                count += 1
            m.sample("x", Normal(float(count), 1.0))

        def endless(m: ModelHandle) -> None:
            """Flip until a one along the first support value: one endless run."""
            count = 0
            while m.sample(f"flip[{count}]", Bernoulli(0.5), marginalize=True) == 0:
                count += 1

        def conflicting(m: ModelHandle) -> None:
            """Use one name as latent in one structure, deterministic in another."""
            k = m.sample("k", Bernoulli(0.5), marginalize=True)
            if k == 1:
                m.sample("x", Normal(0.0, 1.0))
            else:
                m.deterministic("x", 0.0)

        def same_name(m: ModelHandle) -> None:
            """Nest a plate in a plate of the same name."""
            for _ in m.plate("data", 2):
                for _ in m.plate("data", 2):
                    pass

        def too_large(m: ModelHandle) -> None:
            """Declare a vector site with 2^13 joint values."""
            m.sample("z", Bernoulli(0.5), shape=13, marginalize=True)

        def too_many_counts(m: ModelHandle) -> None:
            """Declare a multinomial site with too many count vectors."""
            m.sample("n", Multinomial(40, (0.2, 0.3, 0.1, 0.4)), marginalize=True)

        with self.assertRaisesRegex(ModelSpecificationError, "continuous parameter"):
            ppl.trace(continuous_branch).log_density((1.0,))
        cases = (
            (local_branch, "same for every value of its discrete sites"),
            (local_continuous, "follows a discrete latent site"),
            (unbounded_count, "unbounded"),
            (geometric, "unbounded"),
            (endless, "unbounded"),
            (conflicting, "different kind"),
            (same_name, "same name"),
            (too_large, "at most 4096"),
            (too_many_counts, "at most 4096"),
        )
        for model, message in cases:
            with self.subTest(model=model.__name__):
                with self.assertRaisesRegex(ModelSpecificationError, message):
                    ppl.trace(model)


if __name__ == "__main__":
    unittest.main()
