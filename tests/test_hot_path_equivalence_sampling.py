"""Bit-for-bit equivalence of the optimized sampling and transform hot paths.

The marginal transforms of :class:`~probstat.modeling.JointInput`, the
quantile / inverse-survival functions of ``Normal``, ``LogNormal`` and
``Uniform``, the Monte Carlo / Latin hypercube / Sobol generators, the
subset-simulation chains and the random-walk Metropolis loop were rewritten
for speed with the promise that no result changes: same floats (compared by
``float.hex``), same exceptions (type and message), same random-stream
consumption order.

The ``_ref_*`` code below is the implementation of git revision ``315d501``
(the last commit before the rewrite), copied verbatim apart from removed
docstrings and the ``_ref_`` prefix. Every test feeds fixed-seed and
adversarial inputs to the current and to the reference routine. The golden
digests at the end were produced by that revision for whole workloads
(subset simulation, crude Monte Carlo, Sobol indices, Metropolis chains,
designs).

References:
    [REF-PROBSTAT-VALIDATION]. Equivalence harness only; no equation is
    evaluated.

"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import random
import unittest
import warnings
from collections.abc import Callable
from statistics import NormalDist
from typing import Any

from probstat.distributions import LogNormal, Normal, Poisson, Uniform
from probstat.foundations.validation import require_probability
from probstat.inference.mcmc import random_walk_metropolis
from probstat.linalg import as_vector, cholesky, matvec
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.modeling import inputs as inputs_module
from probstat.modeling.benchmarks import ishigami
from probstat.reliability import ModelLimitState, model_monte_carlo, subset_simulation
from probstat.reliability import simulation as simulation_module
from probstat.sampling import designs as designs_module
from probstat.sampling import latin_hypercube, monte_carlo_design, sobol
from probstat.sensitivity import estimate_sobol_indices
from probstat.uq import QuantityOfInterest

# ---------------------------------------------------------------- references


def _ref_standard_normal_cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _ref_marginal_from_latent(law, z):
    inverse_survival = (
        None if inputs_module.is_discrete_law(law) else getattr(law, "isf", None)
    )
    if z > 0.0 and inverse_survival is not None:
        return float(inverse_survival(max(_ref_standard_normal_cdf(-z), 1e-300)))
    probability = min(max(_ref_standard_normal_cdf(z), 1e-300), 1.0 - 2.0**-53)
    return float(law.quantile(probability))


def _ref_from_standard_normal(joint, u):
    vector = as_vector(u, name="u")
    if len(vector) != joint.dimension:
        raise ValueError("u length must equal the dimension")
    latent = vector if joint.is_independent else matvec(joint._lower, vector)
    return tuple(
        _ref_marginal_from_latent(item.distribution, z)
        for item, z in zip(joint.inputs, latent, strict=True)
    )


def _ref_from_unit(joint, point):
    coordinates = as_vector(point, name="point")
    if len(coordinates) != joint.dimension:
        raise ValueError("design point length must equal the dimension")
    if any(not 0.0 < value < 1.0 for value in coordinates):
        raise ValueError("design coordinates must lie strictly inside (0, 1)")
    if joint.is_independent:
        return tuple(
            float(item.distribution.quantile(value))
            for item, value in zip(joint.inputs, coordinates, strict=True)
        )
    return _ref_from_standard_normal(
        joint,
        tuple(inputs_module.standard_normal_quantile(v) for v in coordinates),
    )


def _ref_as_mapping(joint, x):
    vector = as_vector(x, name="x")
    if len(vector) != joint.dimension:
        raise ValueError("input vector length must equal the dimension")
    return dict(zip(joint.names, vector, strict=True))


def _ref_normal_quantile(self, probability):
    target = require_probability(probability, name="probability")
    if target == 0.0:
        return -math.inf
    if target == 1.0:
        return math.inf
    return NormalDist(mu=self.mu, sigma=self.sigma).inv_cdf(target)


def _ref_normal_isf(self, probability):
    q = require_probability(probability, name="probability")
    if q == 0.0:
        return math.inf
    if q == 1.0:
        return -math.inf
    return self.mu - self.sigma * NormalDist().inv_cdf(q)


def _ref_lognormal_quantile(self, probability):
    target = require_probability(probability, name="probability")
    if target == 0.0:
        return 0.0
    if target == 1.0:
        return math.inf
    normal_quantile = NormalDist(mu=self.mu, sigma=self.sigma).inv_cdf(target)
    try:
        return math.exp(normal_quantile)
    except OverflowError as error:
        raise OverflowError("lognormal quantile exceeds float range") from error


def _ref_lognormal_isf(self, probability):
    q = require_probability(probability, name="probability")
    if q == 0.0:
        return math.inf
    if q == 1.0:
        return 0.0
    try:
        return math.exp(self.mu - self.sigma * NormalDist().inv_cdf(q))
    except OverflowError as error:
        raise OverflowError("lognormal quantile exceeds float range") from error


def _ref_uniform_quantile(self, probability):
    target = require_probability(probability, name="probability")
    if target == 1.0:
        return self.upper
    return min(self.lower + target * self.width, self.upper)


def _ref_uniform_isf(self, probability):
    q = require_probability(probability, name="probability")
    if q == 1.0:
        return self.lower
    return max(self.upper - q * (self.upper - self.lower), self.lower)


def _ref_open_unit(value):
    tiny = 2.0**-53
    return min(max(value, tiny), 1.0 - tiny)


def _ref_monte_carlo_points(size, dimension, seed):
    generator = designs_module.resolve_stream(seed).generator()
    return tuple(
        tuple(_ref_open_unit(generator.random()) for _ in range(dimension))
        for _ in range(size)
    )


def _ref_latin_hypercube_points(size, dimension, seed, centered):
    generator = designs_module.resolve_stream(seed).generator()
    columns = []
    for _ in range(dimension):
        permutation = list(range(size))
        generator.shuffle(permutation)
        columns.append(
            [
                _ref_open_unit(
                    (stratum + (0.5 if centered else generator.random())) / size
                )
                for stratum in permutation
            ]
        )
    return tuple(tuple(column[row] for column in columns) for row in range(size))


def _ref_sobol_points(size, dimension, seed, mode, skip, centered):
    bits_const = designs_module._SOBOL_BITS
    total = skip + size
    bits = max(1, (total - 1).bit_length())
    directions = designs_module._sobol_directions(dimension)
    stream = designs_module.resolve_stream(seed) if mode != "none" else None
    generator = stream.generator() if stream else None
    shifts = [
        generator.getrandbits(bits_const)
        if (generator is not None and mode == "shift")
        else 0
        for _ in range(dimension)
    ]
    centre = 0
    if mode == "none" and (centered is None or centered):
        centre = 1 << (bits_const - bits - 1)
    state = [0] * dimension
    raw = []
    for index in range(total):
        if index >= skip:
            raw.append(list(state))
        changed = ((index + 1) & -(index + 1)).bit_length() - 1
        for axis in range(dimension):
            state[axis] ^= directions[axis][changed]
    if mode == "owen":
        columns = []
        for axis in range(dimension):
            tree = {}
            columns.append(
                designs_module._owen_scramble(
                    [row[axis] for row in raw], bits, generator, tree
                )
            )
        raw = [[columns[axis][i] for axis in range(dimension)] for i in range(size)]
    scale = 2.0**-bits_const
    return tuple(
        tuple(
            _ref_open_unit(((row[axis] ^ shifts[axis]) + centre) * scale)
            for axis in range(dimension)
        )
        for row in raw
    )


def _ref_modified_metropolis_chain(
    evaluate, generator, start, start_value, threshold, length, spread, points, values
):
    current, current_value = start, start_value
    points.append(current)
    values.append(current_value)
    accepted = 0
    for _ in range(length - 1):
        candidate = []
        for coordinate in current:
            trial = coordinate + spread * generator.gauss(0.0, 1.0)
            ratio = math.exp(0.5 * (coordinate * coordinate - trial * trial))
            candidate.append(trial if generator.random() < ratio else coordinate)
        proposal = tuple(candidate)
        if proposal != current:
            proposal_value = evaluate(proposal)
            if proposal_value <= threshold:
                current, current_value = proposal, proposal_value
                accepted += 1
        points.append(current)
        values.append(current_value)
    return accepted


def _ref_conditional_chain(
    evaluate,
    generator,
    start,
    start_value,
    threshold,
    length,
    correlations,
    spreads,
    points,
    values,
):
    current, current_value = start, start_value
    points.append(current)
    values.append(current_value)
    accepted = 0
    for _ in range(length - 1):
        proposal = tuple(
            rho * coordinate + sigma * generator.gauss(0.0, 1.0)
            for coordinate, rho, sigma in zip(
                current, correlations, spreads, strict=True
            )
        )
        proposal_value = evaluate(proposal)
        if proposal_value <= threshold:
            current, current_value = proposal, proposal_value
            accepted += 1
        points.append(current)
        values.append(current_value)
    return accepted


def _ref_rwm_chain(log_density, start, draws, warmup, thin, covariance, seed):
    """Non-adaptive branch of the original ``_run_chain`` (``adapt=False``)."""
    from probstat.foundations.random_streams import resolve_stream

    generator = resolve_stream(seed).generator()
    dimension = len(start)
    evaluations = 0

    def evaluate(point):
        nonlocal evaluations
        evaluations += 1
        return log_density(point)

    state = start
    current = evaluate(state)
    lower = cholesky(covariance)
    samples, trace = [], []
    warm_accepted = kept_accepted = 0
    total = warmup + draws * thin
    for iteration in range(total):
        z = tuple(generator.gauss(0.0, 1.0) for _ in range(dimension))
        step = matvec(lower, z)
        proposal = tuple(a + b for a, b in zip(state, step, strict=True))
        proposed = evaluate(proposal)
        accepted = math.log(1.0 - generator.random()) < proposed - current
        if accepted:
            state, current = proposal, proposed
        if iteration < warmup:
            warm_accepted += accepted
        else:
            kept_accepted += accepted
            if (iteration - warmup + 1) % thin == 0:
                samples.append(state)
                trace.append(current)
    return samples, trace, kept_accepted, warm_accepted, evaluations


# ------------------------------------------------------------------- helpers


def _hex(value: Any) -> Any:
    """Return a structure of exact bit patterns for floats (recursively)."""
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, dict):
        return {k: _hex(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_hex(v) for v in value)
    return value


def _outcome(call: Callable[[], Any]) -> tuple[Any, ...]:
    """Return ``("ok", bits)`` or ``("err", type name, message)``."""
    try:
        return ("ok", _hex(call()))
    except Exception as error:  # noqa: BLE001 - the error itself is compared
        return ("err", type(error).__name__, str(error))


def _flatten(obj: Any, out: list[str]) -> None:
    """Append a canonical text of ``obj`` (floats by ``hex``) to ``out``."""
    if isinstance(obj, float):
        out.append(obj.hex())
    elif isinstance(obj, (bool, int, str)) or obj is None:
        out.append(repr(obj))
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for item in dataclasses.fields(obj):
            _flatten(getattr(obj, item.name), out)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _flatten(key, out)
            _flatten(value, out)
    elif isinstance(obj, (list, tuple)):
        out.append("[")
        for value in obj:
            _flatten(value, out)
        out.append("]")
    else:
        out.append(f"<{type(obj).__name__}>")


def _digest(obj: Any) -> str:
    out: list[str] = []
    _flatten(obj, out)
    return hashlib.sha256("|".join(out).encode()).hexdigest()[:16]


class _Odd(float):
    """A float subclass: must take the validator path, not the fast path."""


def _mixed_joint(correlated: bool = False) -> JointInput:
    items = (
        RandomInput("a", Normal(1.0, 2.0)),
        RandomInput("b", LogNormal(0.1, 0.4)),
        RandomInput("c", Uniform(-2.0, 3.0)),
    )
    if correlated:
        matrix = ((1.0, 0.5, 0.2), (0.5, 1.0, -0.3), (0.2, -0.3, 1.0))
        return JointInput(items, matrix)
    return JointInput(items)


def _limit_state(dimension: int, lognormal: bool) -> ModelLimitState:
    inputs = JointInput.independent(
        *[
            RandomInput(
                f"u{i}",
                LogNormal(0.0, 0.3) if (lognormal and i % 2) else Normal(0.0, 1.0),
            )
            for i in range(dimension)
        ]
    )
    names = [f"u{i}" for i in range(dimension)]
    model = ProbabilisticModel(
        inputs,
        lambda p: {"y": 4.0 - sum(p[n] for n in names) / math.sqrt(dimension)},
        (QuantityOfInterest("y", "response"),),
    )
    return ModelLimitState(model, "y", 0.0, "below")


_PROBABILITIES = (
    0.0,
    1.0,
    0.5,
    1e-300,
    5e-324,
    1.0 - 2.0**-53,
    0.25,
    0.999,
    -0.0,
    -1e-12,
    1.0000000001,
    math.nan,
    math.inf,
    -math.inf,
    True,
    1,
    0,
    "0.5",
    None,
    _Odd(0.3),
)


class MarginalFunctionTests(unittest.TestCase):
    """``quantile`` and ``isf`` of the three marginals equal the originals."""

    def _compare(self, law: Any, new_name: str, reference: Callable[..., Any]) -> None:
        rng = random.Random(11)
        probes = list(_PROBABILITIES) + [rng.random() for _ in range(300)]
        probes += [10.0 ** rng.uniform(-300, -1) for _ in range(100)]
        for probability in probes:
            with self.subTest(law=law, method=new_name, probability=probability):
                self.assertEqual(
                    _outcome(lambda p=probability: getattr(law, new_name)(p)),
                    _outcome(lambda p=probability: reference(law, p)),
                )

    def test_normal(self) -> None:
        """Check normal against the original."""
        for law in (Normal(0.0, 1.0), Normal(-3.5, 0.02), Normal(1e5, 1e3)):
            self._compare(law, "quantile", _ref_normal_quantile)
            self._compare(law, "isf", _ref_normal_isf)

    def test_lognormal(self) -> None:
        """Check lognormal against the original."""
        for law in (LogNormal(0.0, 1.0), LogNormal(5.0, 3.0), LogNormal(-2.0, 0.1)):
            self._compare(law, "quantile", _ref_lognormal_quantile)
            self._compare(law, "isf", _ref_lognormal_isf)

    def test_lognormal_overflow_message(self) -> None:
        """Check lognormal overflow message against the original."""
        law = LogNormal(700.0, 5.0)
        self.assertEqual(
            _outcome(lambda: law.quantile(1.0 - 2.0**-53)),
            _outcome(lambda: _ref_lognormal_quantile(law, 1.0 - 2.0**-53)),
        )

    def test_uniform(self) -> None:
        """Check uniform against the original."""
        for law in (Uniform(0.0, 1.0), Uniform(-2.0, 3.0), Uniform(1e-3, 7e-3)):
            self._compare(law, "quantile", _ref_uniform_quantile)
            self._compare(law, "isf", _ref_uniform_isf)


class TransformTests(unittest.TestCase):
    """``JointInput`` transforms equal the originals bit for bit."""

    def test_standard_normal_cdf_constant(self) -> None:
        """Check standard normal cdf constant against the original."""
        self.assertEqual(inputs_module._SQRT2.hex(), math.sqrt(2.0).hex())
        rng = random.Random(3)
        zs = [rng.gauss(0.0, 3.0) for _ in range(2000)] + [-38.0, 0.0, 8.3, 40.0]
        for z in zs:
            self.assertEqual(
                inputs_module.standard_normal_cdf(z).hex(),
                _ref_standard_normal_cdf(z).hex(),
            )

    def test_from_standard_normal(self) -> None:
        """Check from standard normal against the original."""
        rng = random.Random(5)
        for correlated in (False, True):
            joint = _mixed_joint(correlated)
            vectors: list[Any] = [
                tuple(rng.gauss(0.0, 2.5) for _ in range(3)) for _ in range(500)
            ]
            vectors += [
                (0.0, 0.0, 0.0),
                (-0.0, 9.0, -9.0),
                (40.0, -40.0, 12.0),
                (1.0, 2.0),
                (1.0, math.nan, 0.0),
                (math.inf, 0.0, 0.0),
                (1, 2, 3),
                (True, 0.0, 0.0),
                ("x", 0.0, 0.0),
                (_Odd(0.5), 1.0, 2.0),
                [0.1, 0.2, 0.3],
                iter((0.1, 0.2, 0.3)),
                5,
            ]
            for vector in vectors:
                with self.subTest(correlated=correlated):
                    if hasattr(vector, "__next__"):
                        vector = tuple(vector)
                    self.assertEqual(
                        _outcome(lambda v=vector, j=joint: j.from_standard_normal(v)),
                        _outcome(
                            lambda v=vector, j=joint: _ref_from_standard_normal(j, v)
                        ),
                    )

    def test_from_standard_normal_discrete(self) -> None:
        """Check from standard normal discrete against the original."""
        joint = JointInput.independent(
            RandomInput("n", Normal(0.0, 1.0)), RandomInput("k", Poisson(4.0))
        )
        rng = random.Random(8)
        for _ in range(300):
            u = (rng.gauss(0, 2), rng.gauss(0, 2))
            self.assertEqual(
                _outcome(lambda u=u: joint.from_standard_normal(u)),
                _outcome(lambda u=u: _ref_from_standard_normal(joint, u)),
            )

    def test_from_unit(self) -> None:
        """Check from unit against the original."""
        rng = random.Random(6)
        for correlated in (False, True):
            joint = _mixed_joint(correlated)
            points: list[Any] = [
                tuple(rng.random() for _ in range(3)) for _ in range(500)
            ]
            points += [
                (0.0, 0.5, 0.5),
                (0.5, 1.0, 0.5),
                (-0.0, 0.5, 0.5),
                (5e-324, 1.0 - 2.0**-53, 0.5),
                (0.5, 0.5),
                (0.5, math.nan, 0.5),
                (0.5, math.inf, 0.5),
                (0.5, 1.5, 0.5),
                (0.5, -0.5, 0.5),
                (0.5, 0.5, 1),
                (_Odd(0.5), 0.5, 0.5),
                ("a", 0.5, 0.5),
                [0.2, 0.3, 0.4],
                7,
            ]
            for point in points:
                with self.subTest(correlated=correlated, point=point):
                    self.assertEqual(
                        _outcome(lambda p=point, j=joint: j.from_unit(p)),
                        _outcome(lambda p=point, j=joint: _ref_from_unit(j, p)),
                    )

    def test_as_mapping_and_names(self) -> None:
        """Check as mapping and names against the original."""
        joint = _mixed_joint()
        self.assertEqual(joint.names, ("a", "b", "c"))
        self.assertEqual(joint.names, tuple(item.name for item in joint.inputs))
        for vector in ((1.0, 2.0, 3.0), (1.0, 2.0), (1.0, math.nan, 3.0), (1, 2, 3)):
            self.assertEqual(
                _outcome(lambda v=vector: joint.as_mapping(v)),
                _outcome(lambda v=vector: _ref_as_mapping(joint, v)),
            )

    def test_names_survive_replace_and_pickle(self) -> None:
        """Check names survive replace and pickle against the original."""
        import pickle

        joint = _mixed_joint(True)
        clone = pickle.loads(pickle.dumps(joint))
        self.assertEqual(clone.names, joint.names)
        other = dataclasses.replace(joint, inputs=joint.inputs[:3])
        self.assertEqual(other.names, joint.names)

    def test_sample_and_design(self) -> None:
        """Check sample and design against the original."""
        for correlated in (False, True):
            joint = _mixed_joint(correlated)
            design = monte_carlo_design(200, 3, seed=4)
            expected = tuple(_ref_from_unit(joint, p) for p in design.points)
            self.assertEqual(
                _hex(joint.sample_design(design)),
                _hex(expected),
            )


class ModelEvaluateTests(unittest.TestCase):
    """The tuple branch of ``ProbabilisticModel.evaluate`` is unchanged."""

    def test_evaluate_outputs_and_errors(self) -> None:
        """Check evaluate outputs and errors against the original."""
        state = _limit_state(3, True)
        model = state.model
        rng = random.Random(2)
        for _ in range(100):
            x = tuple(rng.uniform(0.5, 2.0) for _ in range(3))
            self.assertEqual(
                _hex(dict(model.evaluate(x))), _hex(dict(model.evaluate(list(x))))
            )
        for bad in ((1.0, 2.0), (1.0, math.nan, 2.0), (1.0, "a", 2.0)):
            first = _outcome(lambda b=bad: dict(model.evaluate(b)))
            second = _outcome(lambda b=bad: dict(model.evaluate(list(b))))
            self.assertEqual(first, second)
            self.assertEqual(first[0], "err")
        self.assertEqual(
            _outcome(lambda: dict(model.evaluate((1.0, math.inf, 2.0)))),
            ("err", "ValueError", "x[1] must be finite"),
        )


class GeneratorTests(unittest.TestCase):
    """Monte Carlo, Latin hypercube and Sobol designs equal the originals."""

    def test_monte_carlo(self) -> None:
        """Check monte carlo against the original."""
        for size, dimension, seed in ((1, 1, 0), (50, 1, 1), (400, 7, 2), (3, 30, 3)):
            self.assertEqual(
                _hex(monte_carlo_design(size, dimension, seed=seed).points),
                _hex(_ref_monte_carlo_points(size, dimension, seed)),
            )

    def test_latin_hypercube(self) -> None:
        """Check latin hypercube against the original."""
        for size, dimension, seed in ((1, 1, 0), (2, 3, 1), (97, 6, 2), (500, 2, 3)):
            for centered in (False, True):
                self.assertEqual(
                    _hex(
                        latin_hypercube(
                            size, dimension, seed=seed, centered=centered
                        ).points
                    ),
                    _hex(_ref_latin_hypercube_points(size, dimension, seed, centered)),
                )

    def test_sobol(self) -> None:
        """Check sobol against the original."""
        cases = (
            (1, 1, 0, False, 0, None),
            (64, 5, 0, False, 0, None),
            (64, 5, 0, False, 0, False),
            (128, 10, 0, False, 5, True),
            (100, 4, 7, True, 0, None),
            (128, 6, 7, "owen", 0, None),
            (60, 3, 9, "owen", 11, None),
            (256, 40, 1, True, 3, None),
            (2**10, 3, 0, False, 0, None),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for size, dimension, seed, scramble, skip, centered in cases:
                mode = (
                    "none"
                    if scramble is False
                    else "shift"
                    if scramble is True
                    else scramble
                )
                with self.subTest(size=size, dimension=dimension, mode=mode):
                    design = sobol(
                        size,
                        dimension,
                        seed=seed,
                        scramble=scramble,
                        skip=skip,
                        centered=centered,
                    )
                    self.assertEqual(
                        _hex(design.points),
                        _hex(
                            _ref_sobol_points(
                                size, dimension, seed, mode, skip, centered
                            )
                        ),
                    )

    def test_validation_unchanged(self) -> None:
        """Check validation unchanged against the original."""
        for call in (
            lambda: monte_carlo_design(0, 2),
            lambda: latin_hypercube(3, 0),
            lambda: sobol(4, 2000),
            lambda: sobol(4, 2, skip=-1),
            lambda: sobol(4, 2, scramble="bad"),
        ):
            self.assertEqual(_outcome(call)[0], "err")


class ChainTests(unittest.TestCase):
    """The subset-simulation chains and the Metropolis loop equal the originals."""

    def test_modified_metropolis_chain(self) -> None:
        """Check modified metropolis chain against the original."""
        evaluate = lambda p: 3.0 - sum(p)  # noqa: E731
        for seed in range(4):
            starts = [(0.1, -0.2, 0.3), (2.0, 1.0, 0.0)]
            for spread in (0.3, 1.0, 2.5):
                a = random.Random(seed)
                b = random.Random(seed)
                pa: list[Any] = []
                va: list[float] = []
                pb: list[Any] = []
                vb: list[float] = []
                ra = simulation_module._modified_metropolis_chain(
                    evaluate, a, starts[0], evaluate(starts[0]), 1.0, 60, spread, pa, va
                )
                rb = _ref_modified_metropolis_chain(
                    evaluate, b, starts[0], evaluate(starts[0]), 1.0, 60, spread, pb, vb
                )
                self.assertEqual(ra, rb)
                self.assertEqual(_hex(pa), _hex(pb))
                self.assertEqual(_hex(va), _hex(vb))
                self.assertEqual(a.getstate(), b.getstate())

    def test_conditional_chain(self) -> None:
        """Check conditional chain against the original."""
        evaluate = lambda p: 3.0 - sum(p)  # noqa: E731
        rho = [0.8, 0.8, 0.8]
        sigma = [math.sqrt(1.0 - 0.64)] * 3
        for seed in range(4):
            a = random.Random(seed)
            b = random.Random(seed)
            pa: list[Any] = []
            va: list[float] = []
            pb: list[Any] = []
            vb: list[float] = []
            start = (0.5, 0.2, -0.1)
            ra = simulation_module._conditional_chain(
                evaluate, a, start, evaluate(start), 1.0, 50, rho, sigma, pa, va
            )
            rb = _ref_conditional_chain(
                evaluate, b, start, evaluate(start), 1.0, 50, rho, sigma, pb, vb
            )
            self.assertEqual(ra, rb)
            self.assertEqual(_hex(pa), _hex(pb))
            self.assertEqual(_hex(va), _hex(vb))
            self.assertEqual(a.getstate(), b.getstate())

    def test_random_walk_metropolis(self) -> None:
        """Check random walk metropolis against the original."""
        covariance = ((0.5, 0.1, 0.0), (0.1, 0.4, 0.05), (0.0, 0.05, 0.3))

        def log_density(x: tuple[float, ...]) -> float:
            return -0.5 * sum(v * v for v in x)

        result = random_walk_metropolis(
            log_density,
            [0.2, -0.1, 0.4],
            proposal_covariance=covariance,
            draws=300,
            warmup=50,
            thin=2,
            seed=17,
            check_convergence=False,
        )
        chain = result.chains[0]
        from probstat.foundations.random_streams import resolve_stream

        stream = resolve_stream(17).spawn(1, label="mcmc-chain")[0]
        samples, trace, kept, warm, evals = _ref_rwm_chain(
            log_density, (0.2, -0.1, 0.4), 300, 50, 2, covariance, stream
        )
        self.assertEqual(_hex(chain.samples), _hex(tuple(samples)))
        self.assertEqual(_hex(chain.log_density), _hex(tuple(trace)))
        self.assertEqual(chain.evaluations, evals)
        self.assertEqual(chain.acceptance_rate.hex(), (kept / 600).hex())
        self.assertEqual(chain.warmup_acceptance_rate.hex(), (warm / 50).hex())

    def test_metropolis_errors(self) -> None:
        """Check metropolis errors against the original."""
        covariance = ((1.0, 0.0), (0.0, 1.0))
        bad = lambda x: -math.inf  # noqa: E731
        outcome = _outcome(
            lambda: random_walk_metropolis(
                bad, [0.0, 0.0], proposal_covariance=covariance, draws=5, seed=1
            )
        )
        self.assertEqual(
            outcome,
            ("err", "ValueError", "initial point must have a finite log-density"),
        )


# Digests recorded from revision 315d501 (see the module docstring).
_GOLDEN = {
    "subset_2d": "950f0072ddc08c63",
    "subset_10d_lognormal": "2b5d309d49fcae56",
    "subset_cs": "4d01a07ffa00a7dd",
    "crude_mc": "279e220937b3f503",
    "sobol_ishigami": "a9a9a205b951c54a",
    "rwm_5d": "99e9276453c7fc85",
    "design_sobol": "652c62a4d91375f8",
    "design_lhs": "a39404a2e2339ba7",
}


def _end_to_end() -> dict[str, str]:
    """Run the whole workloads and digest every field of the results."""
    result: dict[str, str] = {}
    result["subset_2d"] = _digest(
        subset_simulation(_limit_state(2, False), samples_per_level=500, seed=1)
    )
    result["subset_10d_lognormal"] = _digest(
        subset_simulation(_limit_state(10, True), samples_per_level=500, seed=1)
    )
    result["subset_cs"] = _digest(
        subset_simulation(
            _limit_state(4, True), samples_per_level=400, sampler="cs", seed=5
        )
    )
    result["crude_mc"] = _digest(model_monte_carlo(_limit_state(5, True), 4000, seed=3))
    rng_scale = lambda row: tuple(-math.pi + 2.0 * math.pi * v for v in row)  # noqa: E731
    sample_a = [rng_scale(r) for r in sobol(1024, 3, seed=1).points]
    sample_b = [rng_scale(r) for r in sobol(1024, 3, seed=2).points]
    result["sobol_ishigami"] = _digest(
        [
            estimate_sobol_indices(
                lambda x: ishigami(*x), sample_a, sample_b, estimator=name
            )
            for name in ("saltelli2010", "jansen", "martinez")
        ]
    )
    covariance = tuple(
        tuple(0.5 if i == j else 0.0 for j in range(5)) for i in range(5)
    )
    result["rwm_5d"] = _digest(
        random_walk_metropolis(
            lambda x: -0.5 * sum(v * v for v in x),
            [0.0] * 5,
            proposal_covariance=covariance,
            draws=3000,
            warmup=200,
            chains=2,
            seed=2,
            check_convergence=False,
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result["design_sobol"] = _digest(
            [
                sobol(300, 7),
                sobol(257, 5, seed=3, scramble=True),
                sobol(128, 4, seed=3, scramble="owen"),
                sobol(64, 3, skip=10, centered=False),
            ]
        )
    result["design_lhs"] = _digest(
        [
            latin_hypercube(100, 6, seed=2),
            latin_hypercube(50, 3, seed=2, centered=True),
            monte_carlo_design(500, 4, seed=9),
        ]
    )
    return result


class EndToEndTests(unittest.TestCase):
    """Whole-workload digests equal those recorded from the original code."""

    def test_golden_digests(self) -> None:
        """Check golden digests against the original."""
        produced = _end_to_end()
        for name, expected in _GOLDEN.items():
            with self.subTest(workload=name):
                self.assertEqual(produced[name], expected)


if __name__ == "__main__":
    unittest.main()
