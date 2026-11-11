"""Variable elimination along Markov chains in the PPL layer (ADR-0088).

A two-state hidden Markov model with Gaussian emissions is checked against
brute-force enumeration of every state path and against an independent
scaled forward recursion of [REF-RABINER-1989]. The library has
no other discrete-state hidden Markov code to compare with (the filtering
package holds Kalman, extended Kalman, and particle filters).
"""

from __future__ import annotations

import itertools
import math
import random
import time
import unittest
from collections.abc import Callable, Sequence

from probstat.distributions import IntegerSupport, Normal, RealSupport
from probstat.inference import ppl
from probstat.inference.ppl import ModelHandle, ModelSpecificationError
from probstat.inference.ppl._discrete import MarkovGroup, recover_discrete
from probstat.numerics import admath

LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)


def log_phi(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Return the normal log-density."""
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma) - LOG_SQRT_TWO_PI


def log_sum_exp(values: Sequence[float]) -> float:
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


class Categorical3:
    """A traceable law on ``{0, 1, 2}``."""

    support = IntegerSupport(0, 2)

    def __init__(self, p: Sequence[float]) -> None:
        """Store the probabilities."""
        self.p = tuple(p)

    def log_pmf(self, k: int) -> float:
        """Return ``log p_k``."""
        return math.log(self.p[k])


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


INITIAL = (0.6, 0.4)
TRANSITION = ((0.85, 0.15), (0.25, 0.75))
TRUE_MEANS = (-1.0, 1.5)
PRIOR_MEANS = ((-1.0, 1.0), (1.5, 1.0))


def simulate(count: int, seed: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Simulate states and observations of the reference chain."""
    generator = random.Random(seed)
    states: list[int] = []
    for t in range(count):
        p = INITIAL[1] if t == 0 else TRANSITION[states[-1]][1]
        states.append(int(generator.random() < p))
    return tuple(states), tuple(generator.gauss(TRUE_MEANS[z], 1.0) for z in states)


_, LONG_Y = simulate(40, 11)
_, SHORT_Y = simulate(10, 5)


def hmm_model(data: tuple[float, ...]) -> Callable[[ModelHandle], None]:
    """Return the two-state HMM with unknown emission means for ``data``."""

    def model(m: ModelHandle) -> None:
        """Gaussian emissions with unit scale and a known transition matrix."""
        mu0 = m.sample("mu[0]", Normal(*PRIOR_MEANS[0]))
        mu1 = m.sample("mu[1]", Normal(*PRIOR_MEANS[1]))
        for t, previous in m.markov("chain", len(data)):
            p = INITIAL[1] if previous is None else TRANSITION[previous][1]
            z = m.sample(f"z[{t}]", Indicator(p), marginalize=True)
            m.observe(f"y[{t}]", Gaussian(mu1 if z == 1 else mu0), data[t])

    return model


def path_log_joint(
    path: tuple[int, ...], means: tuple[float, float], data: Sequence[float]
) -> float:
    """Return ``log p(z, y | mu)`` of one state path."""
    total = math.log(INITIAL[path[0]])
    for previous, state in zip(path, path[1:], strict=False):
        total += math.log(TRANSITION[previous][state])
    emissions = (log_phi(y, means[z]) for y, z in zip(data, path, strict=True))
    return total + math.fsum(emissions)


def scaled_forward(means: tuple[float, float], data: Sequence[float]) -> float:
    """Return ``log p(y | mu)`` by the scaled forward recursion (Rabiner 1989)."""
    emit = [[math.exp(log_phi(y, means[k])) for k in (0, 1)] for y in data]
    alpha = [INITIAL[k] * emit[0][k] for k in (0, 1)]
    total = 0.0
    for t in range(1, len(data)):
        scale = alpha[0] + alpha[1]
        total += math.log(scale)
        alpha = [a / scale for a in alpha]
        alpha = [
            emit[t][k] * math.fsum(alpha[i] * TRANSITION[i][k] for i in (0, 1))
            for k in (0, 1)
        ]
    return total + math.log(alpha[0] + alpha[1])


def prior_log(means: tuple[float, float]) -> float:
    """Return the log prior density of the emission means."""
    return math.fsum(
        log_phi(mean, *PRIOR_MEANS[k]) for k, mean in enumerate(means)
    )


class ForwardAlgorithmTests(unittest.TestCase):
    """The eliminated chain equals enumeration over every path."""

    POINTS = ((-0.9, 1.4), (0.3, 0.8), (-2.0, 2.5))

    def test_log_likelihood_equals_brute_force_enumeration(self) -> None:
        """Up to ``n = 10`` the forward algorithm equals ``2^n`` paths."""
        for count in (1, 2, 5, 10):
            data = SHORT_Y[:count]
            program = ppl.trace(hmm_model(data))
            self.assertFalse(program.per_configuration)
            self.assertEqual(program.dimension, 2)
            self.assertEqual(program.site("z[0]").markov, "chain")
            for means in self.POINTS:
                brute = log_sum_exp(
                    [
                        path_log_joint(path, means, data)
                        for path in itertools.product((0, 1), repeat=count)
                    ]
                )
                terms = program.terms(means)
                with self.subTest(count=count, means=means):
                    self.assertLess(abs(terms.log_likelihood - brute), 1e-12)
                    self.assertLess(
                        abs(terms.log_likelihood - scaled_forward(means, data)), 1e-12
                    )
                    self.assertLess(
                        abs(program.log_density(means) - prior_log(means) - brute),
                        1e-12,
                    )

    def test_gradient_and_cost(self) -> None:
        """Reverse-mode gradient agrees with differences; cost is ``O(n K^2)``."""
        executions = [0]
        data = LONG_Y

        def counted(m: ModelHandle) -> None:
            """Run the HMM and count the step-body executions."""
            mu0 = m.sample("mu[0]", Normal(*PRIOR_MEANS[0]))
            mu1 = m.sample("mu[1]", Normal(*PRIOR_MEANS[1]))
            for t, previous in m.markov("chain", len(data)):
                executions[0] += 1
                p = INITIAL[1] if previous is None else TRANSITION[previous][1]
                z = m.sample(f"z[{t}]", Indicator(p), marginalize=True)
                m.observe(f"y[{t}]", Gaussian(mu1 if z == 1 else mu0), data[t])

        program = ppl.trace(counted)
        executions[0] = 0
        value = program.log_density((-0.9, 1.4))
        self.assertEqual(executions[0], 2 + 4 * (len(data) - 1))
        self.assertLess(
            abs(value - prior_log((-0.9, 1.4)) - scaled_forward((-0.9, 1.4), data)),
            1e-11,
        )
        reverse = program.gradient((-0.9, 1.4), method="reverse")
        finite = program.gradient((-0.9, 1.4), method="finite-difference")
        for a, b in zip(reverse, finite, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-5)

    def test_three_states_and_two_sites_per_step(self) -> None:
        """A step with two discrete sites (four joint states) and ``K = 3``."""
        data = (0.4, 2.2, -0.7, 1.1, 3.0)
        rates = ((0.7, 0.2, 0.1), (0.2, 0.6, 0.2), (0.1, 0.3, 0.6))

        def model(m: ModelHandle) -> None:
            """Run a three-state chain plus a per-step flag that depends on it."""
            shift = m.sample("shift", Normal(0.0, 1.0))
            for t, previous in m.markov("chain", len(data)):
                state = previous[0] if previous is not None else None
                law = (1 / 3, 1 / 3, 1 / 3) if state is None else rates[state]
                s = m.sample(f"s[{t}]", Categorical3(law), marginalize=True)
                f = m.sample(f"f[{t}]", Indicator(0.2 + 0.2 * s), marginalize=True)
                m.observe(f"y[{t}]", Gaussian(shift + s + 0.5 * f), data[t])

        program = ppl.trace(model)
        shift = 0.3
        brute: list[float] = []
        best: tuple[float, tuple[int, ...], tuple[int, ...]] | None = None
        for states in itertools.product(range(3), repeat=len(data)):
            for flags in itertools.product((0, 1), repeat=len(data)):
                total = math.log(1 / 3)
                for t, (s, f) in enumerate(zip(states, flags, strict=True)):
                    if t:
                        total += math.log(rates[states[t - 1]][s])
                    p = 0.2 + 0.2 * s
                    total += math.log(p if f else 1.0 - p)
                    total += log_phi(data[t], shift + s + 0.5 * f)
                brute.append(total)
                if best is None or total > best[0]:
                    best = (total, states, flags)
        self.assertEqual(len(brute), 6**5)
        expected = log_phi(shift) + log_sum_exp(brute)
        self.assertLess(abs(program.log_density((shift,)) - expected), 1e-12)
        assert best is not None
        mode = program.most_probable_discrete((shift,))
        self.assertEqual(tuple(mode[f"s[{t}]"] for t in range(5)), best[1])
        self.assertEqual(tuple(mode[f"f[{t}]"] for t in range(5)), best[2])

    def test_chains_in_a_plate_and_plates_in_a_chain(self) -> None:
        """Independent sequences in a plate; per-step outlier indicators."""
        sequences = ((0.2, 1.9, 2.1), (-1.2, -0.8), (1.4, -0.9, 1.7))

        def sequences_model(m: ModelHandle) -> None:
            """One chain per sequence."""
            mu0 = m.sample("mu[0]", Normal(*PRIOR_MEANS[0]))
            mu1 = m.sample("mu[1]", Normal(*PRIOR_MEANS[1]))
            for g in m.plate("sequences", len(sequences)):
                ys = sequences[g]
                for t, previous in m.markov(f"chain{g}", len(ys)):
                    p = INITIAL[1] if previous is None else TRANSITION[previous][1]
                    z = m.sample(f"z[{g},{t}]", Indicator(p), marginalize=True)
                    m.observe(f"y[{g},{t}]", Gaussian(mu1 if z else mu0), ys[t])

        program = ppl.trace(sequences_model)
        means = (-0.7, 1.6)
        expected = prior_log(means) + math.fsum(
            log_sum_exp(
                [
                    path_log_joint(path, means, ys)
                    for path in itertools.product((0, 1), repeat=len(ys))
                ]
            )
            for ys in sequences
        )
        self.assertLess(abs(program.log_density(means) - expected), 1e-12)

        pairs = ((0.1, 0.4), (2.3, 1.8), (-1.1, 5.0), (1.2, 1.6))

        def outliers(m: ModelHandle) -> None:
            """Two observations per step, each with an outlier indicator."""
            mu0 = m.sample("mu[0]", Normal(*PRIOR_MEANS[0]))
            mu1 = m.sample("mu[1]", Normal(*PRIOR_MEANS[1]))
            for t, previous in m.markov("chain", len(pairs)):
                p = INITIAL[1] if previous is None else TRANSITION[previous][1]
                z = m.sample(f"z[{t}]", Indicator(p), marginalize=True)
                for j in m.plate(f"obs{t}", 2):
                    o = m.sample(f"o[{t},{j}]", Indicator(0.1), marginalize=True)
                    m.observe(
                        f"y[{t},{j}]",
                        Gaussian(mu1 if z else mu0, 4.0 if o else 1.0),
                        pairs[t][j],
                    )

        program = ppl.trace(outliers)
        brute: list[float] = []
        joint: dict[tuple[int, ...], float] = {}
        for path in itertools.product((0, 1), repeat=len(pairs)):
            for flags in itertools.product((0, 1), repeat=2 * len(pairs)):
                total = math.log(INITIAL[path[0]])
                for t in range(1, len(pairs)):
                    total += math.log(TRANSITION[path[t - 1]][path[t]])
                for t in range(len(pairs)):
                    for j in (0, 1):
                        o = flags[2 * t + j]
                        total += math.log(0.1 if o else 0.9)
                        total += log_phi(pairs[t][j], means[path[t]], 4.0 if o else 1.0)
                brute.append(total)
                joint[path + flags] = total
        self.assertEqual(len(brute), 4096)
        normalizer = log_sum_exp(brute)
        self.assertLess(
            abs(program.log_density(means) - prior_log(means) - normalizer), 1e-12
        )
        conditional = program.discrete_conditional(means)
        chain = conditional.groups[0][0]
        self.assertIsInstance(chain, MarkovGroup)
        recovered = recover_discrete([conditional], 1)
        for t in range(len(pairs)):
            exact = math.fsum(
                math.exp(v - normalizer) for key, v in joint.items() if key[t] == 1
            )
            self.assertAlmostEqual(
                recovered.probabilities[f"z[{t}]"][0][1], exact, places=12
            )
            for j in (0, 1):
                index = len(pairs) + 2 * t + j
                exact = math.fsum(
                    math.exp(v - normalizer)
                    for key, v in joint.items()
                    if key[index] == 1
                )
                self.assertAlmostEqual(
                    recovered.probabilities[f"o[{t},{j}]"][0][1], exact, places=12
                )
        best = max(joint, key=joint.__getitem__)
        mode = program.most_probable_discrete(means)
        self.assertEqual(tuple(mode[f"z[{t}]"] for t in range(4)), best[:4])
        self.assertEqual(
            tuple(mode[f"o[{t},{j}]"] for t in range(4) for j in (0, 1)), best[4:]
        )


class RecoveryTests(unittest.TestCase):
    """Smoothed marginals, FFBS draws, and the Viterbi path at fixed means."""

    MEANS = (-0.8, 1.3)

    def exact(self) -> tuple[list[float], dict[tuple[int, ...], float]]:
        """Return the exact smoothed ``p(z_t = 1 | mu, y)`` and path weights."""
        weights = {
            path: path_log_joint(path, self.MEANS, SHORT_Y)
            for path in itertools.product((0, 1), repeat=len(SHORT_Y))
        }
        normalizer = log_sum_exp(list(weights.values()))
        probabilities = {k: math.exp(v - normalizer) for k, v in weights.items()}
        marginals = [
            math.fsum(p for path, p in probabilities.items() if path[t] == 1)
            for t in range(len(SHORT_Y))
        ]
        return marginals, probabilities

    def test_smoothed_marginals_and_ffbs_draws(self) -> None:
        """Recovered marginals equal exact smoothing; FFBS draws match it."""
        program = ppl.trace(hmm_model(SHORT_Y))
        conditional = program.discrete_conditional(self.MEANS)
        marginals, probabilities = self.exact()
        count = 4000
        recovered = recover_discrete([conditional] * count, 3)
        for t, exact in enumerate(marginals):
            row = recovered.probabilities[f"z[{t}]"][0]
            self.assertAlmostEqual(row[1], exact, places=12)
            self.assertAlmostEqual(row[0] + row[1], 1.0, places=12)
            frequency = sum(recovered.draws[f"z[{t}]"]) / count
            bound = 5.0 * math.sqrt(exact * (1.0 - exact) / count) + 1e-9
            self.assertLess(abs(frequency - exact), bound)
        # Joint draws: the pair (z_3, z_4) follows the exact pairwise law.
        for a, b in itertools.product((0, 1), repeat=2):
            exact = math.fsum(
                p for path, p in probabilities.items() if path[3] == a and path[4] == b
            )
            frequency = (
                sum(
                    1
                    for s in range(count)
                    if recovered.draws["z[3]"][s] == a
                    and recovered.draws["z[4]"][s] == b
                )
                / count
            )
            bound = 5.0 * math.sqrt(exact * (1.0 - exact) / count) + 1e-9
            self.assertLess(abs(frequency - exact), bound)

    def test_viterbi_path_is_the_brute_force_mode(self) -> None:
        """``most_probable_discrete`` returns the most probable path."""
        program = ppl.trace(hmm_model(SHORT_Y))
        _, probabilities = self.exact()
        best = max(probabilities, key=probabilities.__getitem__)
        mode = program.most_probable_discrete(self.MEANS)
        self.assertEqual(tuple(mode[f"z[{t}]"] for t in range(len(SHORT_Y))), best)


def ordered_hmm(m: ModelHandle) -> None:
    """Run the HMM of :data:`LONG_Y` with ordered emission means."""
    mu = m.sample("mu", ppl.OrderedVector(Normal(0.0, 3.0), 2))
    for t, previous in m.markov("chain", len(LONG_Y)):
        p = INITIAL[1] if previous is None else TRANSITION[previous][1]
        z = m.sample(f"z[{t}]", Indicator(p), marginalize=True)
        m.observe(f"y[{t}]", Gaussian(mu[z]), LONG_Y[t])


class PosteriorTests(unittest.TestCase):
    """NUTS on the marginal density of the emission means."""

    def exact_posterior(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the posterior means and standard deviations by quadrature.

        The ordered prior is ``2 N(mu_0; 0, 9) N(mu_1; 0, 9)`` on
        ``mu_0 < mu_1``; the grid covers more than six posterior standard
        deviations around the mode.
        """
        a, wa = simpson(-2.3, 0.9, 101)
        b, wb = simpson(-0.4, 4.0, 101)
        logs = [
            [
                math.log(2.0)
                + log_phi(x, 0.0, 3.0)
                + log_phi(y, 0.0, 3.0)
                + scaled_forward((x, y), LONG_Y)
                if x < y
                else -math.inf
                for y in b
            ]
            for x in a
        ]
        peak = max(max(row) for row in logs)
        mass = 0.0
        first = [0.0, 0.0]
        second = [0.0, 0.0]
        for x, w0, row in zip(a, wa, logs, strict=True):
            for y, w1, value in zip(b, wb, row, strict=True):
                weight = w0 * w1 * math.exp(value - peak)
                mass += weight
                first[0] += weight * x
                first[1] += weight * y
                second[0] += weight * x * x
                second[1] += weight * y * y
        means = (first[0] / mass, first[1] / mass)
        sds = (
            math.sqrt(second[0] / mass - means[0] ** 2),
            math.sqrt(second[1] / mass - means[1] ** 2),
        )
        return means, sds

    def test_nuts_recovers_emission_means(self) -> None:
        """Posterior means within four MCSE of quadrature; truth covered."""
        started = time.perf_counter()
        posterior = ppl.sample(ordered_hmm, draws=250, warmup=250, chains=2, seed=3)
        elapsed = time.perf_counter() - started
        self.assertEqual(posterior.gradient_method, "reverse")
        means, sds = self.exact_posterior()
        summary = {row.name: row for row in posterior.summary()}
        for k in (0, 1):
            row = summary[f"mu[{k}]"]
            self.assertLess(abs(row.mean - means[k]), 4.0 * row.mcse_mean)
            self.assertLess(abs(row.mean - TRUE_MEANS[k]), 4.0 * sds[k])
        assert posterior.discrete is not None
        probabilities = posterior.discrete_probabilities("z[0]")
        self.assertAlmostEqual(math.fsum(probabilities.values()), 1.0, places=12)
        self.assertEqual(len(posterior["z[39]"]), posterior.draw_count)
        self.assertLess(elapsed, 600.0)


class MarkovBoundaryTests(unittest.TestCase):
    """Unsupported chain structures are refused with clear messages."""

    def test_refused_structures(self) -> None:
        """Continuous sites, nested chains, varying structure, bad sizes."""

        def continuous_in_step(m: ModelHandle) -> None:
            """Declare a continuous site in a step."""
            for t, _ in m.markov("chain", 3):
                m.sample(f"x[{t}]", Normal(0.0, 1.0))

        def nested_chain(m: ModelHandle) -> None:
            """Nest a chain in a chain."""
            for _ in m.markov("outer", 2):
                for _ in m.markov("inner", 2):
                    pass

        def varying(m: ModelHandle) -> None:
            """Declare an observation only in one state."""
            mu = m.sample("mu", Normal(0.0, 1.0))
            for t, _ in m.markov("chain", 3):
                z = m.sample(f"z[{t}]", Indicator(0.5), marginalize=True)
                if z == 1:
                    m.observe(f"y[{t}]", Gaussian(mu), 0.2)

        def bad_size(m: ModelHandle) -> None:
            """Declare a chain of length zero."""
            for _ in m.markov("chain", 0):
                pass

        with self.assertRaisesRegex(ModelSpecificationError, "Markov step"):
            ppl.trace(continuous_in_step)
        with self.assertRaisesRegex(ModelSpecificationError, "cannot nest"):
            ppl.trace(nested_chain)
        with self.assertRaisesRegex(ModelSpecificationError, "must be the same"):
            ppl.trace(varying)
        with self.assertRaisesRegex(ModelSpecificationError, "positive integer"):
            ppl.trace(bad_size)

    def test_prior_predictive_follows_the_chain(self) -> None:
        """Generate mode passes the drawn previous state to the next step."""
        sticky = ((0.99, 0.01), (0.01, 0.99))

        def model(m: ModelHandle) -> None:
            """Simulate a nearly constant chain."""
            for t, previous in m.markov("chain", 6):
                p = 0.5 if previous is None else sticky[previous][1]
                m.sample(f"z[{t}]", Indicator(p), marginalize=True)

        prior = ppl.prior_predictive(model, 200, seed=4)
        changes = sum(
            1
            for s in range(200)
            for t in range(1, 6)
            if prior[f"z[{t}]"][s] != prior[f"z[{t - 1}]"][s]
        )
        self.assertLess(changes, 30)


if __name__ == "__main__":
    unittest.main()
