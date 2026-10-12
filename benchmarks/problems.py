"""Benchmark problem registry with known reference answers.

Each :class:`Problem` builds a *fresh* instance per run (so the model's
evaluation counter starts at zero) and supplies reference values from
:mod:`benchmarks.references`, which never imports probstat. The calibration
problem regenerates its data from the seed, so its references depend on the
seed; all other references are constants.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from benchmarks import references as ref
from probstat.distributions import LogNormal, Normal, Uniform
from probstat.inverse import CalibrationProblem, Experiment, NoiseModel
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.reliability import ModelLimitState
from probstat.uq import QuantityOfInterest

CATEGORIES = ("reliability", "sensitivity", "propagation", "calibration")

#: Large seeded Monte Carlo run of the four-branch system, executed once with
#: ``references.four_branch_monte_carlo(10_000_000, 20261012)`` (20.9 s). It is
#: an independent check of the quadrature reference, not the reference itself.
FOUR_BRANCH_MC = {
    "samples": 10_000_000,
    "seed": 20261012,
    "probability": 0.0044378,
    "standard_error": 2.101929097557765e-05,
}


@dataclass(frozen=True)
class Reference:
    """Known answer for one quantity.

    Args:
        value: Reference value.
        scale: Optional error scale; the report divides the absolute error by
            it (the exact posterior standard deviation for calibration means).

    """

    value: float
    scale: float | None = None


@dataclass(frozen=True)
class Instance:
    """One freshly built problem instance.

    Args:
        model: Probabilistic model whose counter measures evaluations.
        payload: Category-specific objects (limit state, calibration problem,
            quantity names, ...).

    """

    model: ProbabilisticModel
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Return one payload entry."""
        return self.payload[key]


@dataclass(frozen=True)
class Problem:
    """A benchmark problem.

    Args:
        key: Short identifier used on the command line and in results.
        category: One of :data:`CATEGORIES`.
        title: Human-readable name.
        description: One-sentence statement of the problem.
        reference_note: How the reference values were obtained.
        build: ``seed -> Instance`` with a fresh model.
        references: ``seed -> {quantity: Reference}``.
        error_kinds: Per quantity group, ``"relative"``, ``"absolute"``, or
            ``"scaled"`` (absolute error divided by :attr:`Reference.scale`).
        tags: Free labels used by method applicability rules.
        dimension: Number of random inputs.

    """

    key: str
    category: str
    title: str
    description: str
    reference_note: str
    build: Callable[[int], Instance]
    references: Callable[[int], Mapping[str, Reference]]
    error_kinds: Mapping[str, str]
    tags: frozenset[str] = frozenset()
    dimension: int = 0


def quantity_group(quantity: str) -> str:
    """Return the group of a quantity name, the part before ``[``."""
    return quantity.split("[", 1)[0]


def _constant(values: Mapping[str, float]) -> Callable[[int], Mapping[str, Reference]]:
    """Return a seed-independent reference function."""
    frozen = {name: Reference(value) for name, value in values.items()}

    def lookup(seed: int) -> Mapping[str, Reference]:
        """Return the constant references (the seed is ignored)."""
        del seed
        return frozen

    return lookup


def _standard_inputs(dimension: int) -> JointInput:
    """Return independent standard normal inputs ``u1..ud``."""
    return JointInput.independent(
        *(RandomInput(f"u{i + 1}", Normal(0.0, 1.0)) for i in range(dimension))
    )


def _standard_state(
    function: Callable[[Sequence[float]], float], dimension: int, label: str
) -> Instance:
    """Wrap ``G(u)`` on standard normal inputs as a ``G <= 0`` limit state."""
    names = tuple(f"u{i + 1}" for i in range(dimension))

    def response(p: Mapping[str, float]) -> dict[str, float]:
        """Evaluate the limit-state function from named inputs."""
        return {"g": function(tuple(p[name] for name in names))}

    model = ProbabilisticModel(
        _standard_inputs(dimension),
        response,
        (QuantityOfInterest("g", "limit-state margin"),),
        name=label,
    )
    return Instance(model, {"limit_state": ModelLimitState(model, "g", 0.0, "below")})


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------


def _linear(beta: float, dimension: int) -> Callable[[int], Instance]:
    """Return a builder of ``G = beta - sum(u) / sqrt(d)``."""
    root = math.sqrt(dimension)

    def build(seed: int) -> Instance:
        """Build the hyperplane limit state."""
        del seed
        return _standard_state(
            lambda u: beta - math.fsum(u) / root, dimension, f"linear-{dimension}d"
        )

    return build


PARABOLIC_BETA, PARABOLIC_CURVATURE = 3.0, 0.5


def _build_parabolic(seed: int) -> Instance:
    """Build ``G = beta + kappa u1^2 / 2 - u2``."""
    del seed
    return _standard_state(
        lambda u: PARABOLIC_BETA + 0.5 * PARABOLIC_CURVATURE * u[0] ** 2 - u[1],
        2,
        "parabolic",
    )


def _build_four_branch(seed: int) -> Instance:
    """Build the four-branch system with the minimum and each branch as QoIs."""
    del seed

    def response(p: Mapping[str, float]) -> dict[str, float]:
        """Evaluate the system margin and the four branch margins."""
        x1, x2 = p["u1"], p["u2"]
        s = 0.1 * (x1 - x2) ** 2
        branches = (
            3.0 + s - (x1 + x2) / ref.SQRT2,
            3.0 + s + (x1 + x2) / ref.SQRT2,
            (x1 - x2) + 6.0 / ref.SQRT2,
            (x2 - x1) + 6.0 / ref.SQRT2,
        )
        out = {"g": min(branches)}
        out.update({f"g{i + 1}": value for i, value in enumerate(branches)})
        return out

    quantities = (QuantityOfInterest("g", "system margin"),) + tuple(
        QuantityOfInterest(f"g{i}", f"branch {i} margin") for i in range(1, 5)
    )
    model = ProbabilisticModel(_standard_inputs(2), response, quantities, "four-branch")
    components = tuple(
        ModelLimitState(model, f"g{i}", 0.0, "below") for i in range(1, 5)
    )
    return Instance(
        model,
        {
            "limit_state": ModelLimitState(model, "g", 0.0, "below"),
            "components": components,
        },
    )


LN_CAPACITY = (math.log(15.0), 0.15)
LN_DEMAND = (math.log(6.0), 0.25)


def _build_lognormal(seed: int) -> Instance:
    """Build ``g = R - S`` with independent lognormal capacity and demand."""
    del seed
    inputs = JointInput.independent(
        RandomInput("R", LogNormal(*LN_CAPACITY), "capacity"),
        RandomInput("S", LogNormal(*LN_DEMAND), "demand"),
    )
    model = ProbabilisticModel(
        inputs,
        lambda p: {"margin": p["R"] - p["S"]},
        (QuantityOfInterest("margin", "capacity minus demand"),),
        name="lognormal-rs",
    )
    return Instance(
        model, {"limit_state": ModelLimitState(model, "margin", 0.0, "below")}
    )


# ---------------------------------------------------------------------------
# Sensitivity and propagation
# ---------------------------------------------------------------------------

ISHIGAMI_A, ISHIGAMI_B = 7.0, 0.1
G_COEFFICIENTS = (0.0, 1.0, 4.5, 9.0, 99.0, 99.0)
EXPONENTIAL_RATES = (0.2, 0.4, 0.6, 0.8, 1.0)
SHAPLEY_COEFFICIENTS = (1.0, 1.0, 1.0)
SHAPLEY_SDS = (1.0, 1.0, 2.0)
SHAPLEY_CORRELATION = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.9), (0.0, 0.9, 1.0))


def _build_ishigami(seed: int) -> Instance:
    """Build the Ishigami model on ``U(-pi, pi)^3``."""
    del seed
    inputs = JointInput.independent(
        *(RandomInput(f"x{i}", Uniform(-math.pi, math.pi)) for i in (1, 2, 3))
    )
    model = ProbabilisticModel(
        inputs,
        lambda p: {"y": ref.ishigami((p["x1"], p["x2"], p["x3"]))},
        (QuantityOfInterest("y", "Ishigami output"),),
        name="ishigami",
    )
    return Instance(model, {"quantity": "y", "names": ("x1", "x2", "x3")})


def _build_g_function(seed: int) -> Instance:
    """Build the Sobol G-function on ``U(0, 1)^6``."""
    del seed
    names = tuple(f"x{i + 1}" for i in range(len(G_COEFFICIENTS)))
    inputs = JointInput.independent(
        *(RandomInput(name, Uniform(0.0, 1.0)) for name in names)
    )
    model = ProbabilisticModel(
        inputs,
        lambda p: {"y": ref.g_function([p[n] for n in names], G_COEFFICIENTS)},
        (QuantityOfInterest("y", "G-function output"),),
        name="g-function",
    )
    return Instance(model, {"quantity": "y", "names": names})


def _build_correlated_linear(seed: int) -> Instance:
    """Build ``Y = X1 + X2 + X3`` with Gaussian inputs, corr(X2, X3) = 0.9."""
    del seed
    names = ("x1", "x2", "x3")
    inputs = JointInput(
        tuple(
            RandomInput(name, Normal(0.0, sd))
            for name, sd in zip(names, SHAPLEY_SDS, strict=True)
        ),
        SHAPLEY_CORRELATION,
    )
    model = ProbabilisticModel(
        inputs,
        lambda p: {
            "y": math.fsum(
                b * p[n] for b, n in zip(SHAPLEY_COEFFICIENTS, names, strict=True)
            )
        },
        (QuantityOfInterest("y", "linear output"),),
        name="correlated-linear",
    )
    return Instance(model, {"quantity": "y", "names": names})


def _build_exponential(seed: int) -> Instance:
    """Build ``f(x) = exp(sum c_i x_i)`` on ``U(0, 1)^5``."""
    del seed
    names = tuple(f"x{i + 1}" for i in range(len(EXPONENTIAL_RATES)))
    inputs = JointInput.independent(
        *(RandomInput(name, Uniform(0.0, 1.0)) for name in names)
    )
    model = ProbabilisticModel(
        inputs,
        lambda p: {
            "f": math.exp(
                math.fsum(
                    c * p[n] for c, n in zip(EXPONENTIAL_RATES, names, strict=True)
                )
            )
        },
        (QuantityOfInterest("f", "smooth integrand"),),
        name="exponential-5d",
    )
    return Instance(model, {"quantity": "f", "names": names})


def shapley_covariance() -> tuple[tuple[float, ...], ...]:
    """Return the covariance of the correlated linear model's inputs."""
    return tuple(
        tuple(
            SHAPLEY_CORRELATION[i][j] * SHAPLEY_SDS[i] * SHAPLEY_SDS[j]
            for j in range(3)
        )
        for i in range(3)
    )


def ishigami_shapley() -> tuple[float, float, float]:
    """Return exact Shapley effects of Ishigami (independent inputs).

    For independent inputs ``Sh_i = sum_{u containing i} V_u / (|u| V)``; the
    only interaction is ``V13``, shared equally by ``x1`` and ``x3``.
    """
    first, total = ref.ishigami_indices(ISHIGAMI_A, ISHIGAMI_B)
    interaction = total[2]
    return first[0] + 0.5 * interaction, first[1], 0.5 * interaction


def _index_references(
    names: Sequence[str],
    first: Sequence[float],
    total: Sequence[float],
    shapley: Sequence[float] | None = None,
) -> dict[str, float]:
    """Return ``S1[x]``, ``ST[x]`` (and ``Sh[x]``) reference entries."""
    values: dict[str, float] = {}
    for name, s1, st in zip(names, first, total, strict=True):
        values[f"S1[{name}]"] = s1
        values[f"ST[{name}]"] = st
    if shapley is not None:
        for name, sh in zip(names, shapley, strict=True):
            values[f"Sh[{name}]"] = sh
    return values


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

CAL_CONDITIONS = tuple(2.0 * i / 7.0 for i in range(8))
CAL_NOISE_SD = 0.5
CAL_PRIOR_SD = 1.0


def calibration_data(seed: int) -> tuple[tuple[float, float], tuple[float, ...]]:
    """Return the true ``(a, b)`` drawn from the prior and noisy observations."""
    generator = random.Random(1_000_003 * seed + 17)
    truth = (generator.gauss(0.0, CAL_PRIOR_SD), generator.gauss(0.0, CAL_PRIOR_SD))
    observations = tuple(
        truth[0] + truth[1] * t + generator.gauss(0.0, CAL_NOISE_SD)
        for t in CAL_CONDITIONS
    )
    return truth, observations


def _build_calibration(seed: int) -> Instance:
    """Build the linear-Gaussian calibration problem for one data set."""
    truth, observations = calibration_data(seed)
    inputs = JointInput.independent(
        RandomInput("a", Normal(0.0, CAL_PRIOR_SD), "intercept"),
        RandomInput("b", Normal(0.0, CAL_PRIOR_SD), "slope"),
    )
    model = ProbabilisticModel(
        inputs,
        lambda p: {"y": p["a"] + p["b"] * p["t"]},
        (QuantityOfInterest("y", "observed response"),),
        name="linear-calibration",
        fixed={"t": 0.0},
    )
    problem = CalibrationProblem(
        model,
        tuple(
            Experiment({"y": y}, {"t": t})
            for y, t in zip(observations, CAL_CONDITIONS, strict=True)
        ),
        (NoiseModel("y", standard_deviation=CAL_NOISE_SD),),
    )
    return Instance(model, {"problem": problem, "truth": truth, "names": ("a", "b")})


def _calibration_references(seed: int) -> Mapping[str, Reference]:
    """Return exact posterior means and sds and the true parameters."""
    truth, observations = calibration_data(seed)
    mean, cov = ref.linear_gaussian_posterior(
        CAL_CONDITIONS, observations, CAL_NOISE_SD, CAL_PRIOR_SD
    )
    sds = (math.sqrt(cov[0][0]), math.sqrt(cov[1][1]))
    values: dict[str, Reference] = {}
    for i, name in enumerate(("a", "b")):
        values[f"mean[{name}]"] = Reference(mean[i], sds[i])
        values[f"sd[{name}]"] = Reference(sds[i])
        values[f"theta[{name}]"] = Reference(truth[i], sds[i])
    return values


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def registry() -> tuple[Problem, ...]:
    """Return every benchmark problem in report order."""
    ishigami_mean, ishigami_variance = ref.ishigami_moments(ISHIGAMI_A, ISHIGAMI_B)
    ishigami_first, ishigami_total = ref.ishigami_indices(ISHIGAMI_A, ISHIGAMI_B)
    g_first, g_total = ref.g_function_indices(G_COEFFICIENTS)
    exp_mean, exp_variance = ref.exponential_integral(EXPONENTIAL_RATES)
    shapley = ref.linear_gaussian_shapley_reference(
        SHAPLEY_COEFFICIENTS, shapley_covariance()
    )
    pf = {"Pf": "relative"}
    return (
        Problem(
            "linear-2d",
            "reliability",
            "Linear limit state, d = 2",
            "G = 3 - (u1 + u2)/sqrt 2 in standard normal space.",
            "Exact: P_f = Phi(-3).",
            _linear(3.0, 2),
            _constant({"Pf": ref.linear_failure_probability(3.0)}),
            pf,
            frozenset({"linear"}),
            2,
        ),
        Problem(
            "linear-10d",
            "reliability",
            "Linear limit state, d = 10",
            "G = 3.5 - sum(u)/sqrt 10 in standard normal space.",
            "Exact: P_f = Phi(-3.5).",
            _linear(3.5, 10),
            _constant({"Pf": ref.linear_failure_probability(3.5)}),
            pf,
            frozenset({"linear", "high-dimensional"}),
            10,
        ),
        Problem(
            "parabolic",
            "reliability",
            "Parabolic limit state",
            "G = 3 + 0.25 u1^2 - u2 (beta = 3, curvature 0.5).",
            "1-D quadrature E[Phi(-(3 + 0.25 Z^2))], Simpson, 4000 panels on "
            "[-12, 12]; identical to 16 digits with 8000 panels.",
            _build_parabolic,
            _constant(
                {
                    "Pf": ref.parabolic_failure_probability(
                        PARABOLIC_BETA, PARABOLIC_CURVATURE
                    )
                }
            ),
            pf,
            frozenset({"curved"}),
            2,
        ),
        Problem(
            "four-branch",
            "reliability",
            "Four-branch series system",
            "min of four branches (Echard et al. 2011, k = 6); two design "
            "points on each axis pair.",
            "1-D quadrature in rotated coordinates (Simpson, 4000 panels); "
            "confirmed by a seeded crude Monte Carlo run of 10^7 samples: "
            f"{FOUR_BRANCH_MC['probability']:.5e} +/- "
            f"{FOUR_BRANCH_MC['standard_error']:.2e} (z = -0.93).",
            _build_four_branch,
            _constant({"Pf": ref.four_branch_failure_probability()}),
            pf,
            frozenset({"system", "multimodal"}),
            2,
        ),
        Problem(
            "lognormal-rs",
            "reliability",
            "Lognormal capacity-demand",
            "g = R - S, ln R ~ N(ln 15, 0.15^2), ln S ~ N(ln 6, 0.25^2).",
            "Exact: P_f = Phi(-ln(15/6) / sqrt(0.15^2 + 0.25^2)).",
            _build_lognormal,
            _constant(
                {"Pf": ref.lognormal_failure_probability(*LN_CAPACITY, *LN_DEMAND)}
            ),
            pf,
            frozenset({"non-normal"}),
            2,
        ),
        Problem(
            "ishigami-indices",
            "sensitivity",
            "Ishigami Sobol and Shapley indices",
            "Ishigami function (a = 7, b = 0.1) on U(-pi, pi)^3.",
            "Analytic variance decomposition (V1, V2, V13); Shapley effects "
            "split V13 equally between x1 and x3.",
            _build_ishigami,
            _constant(
                _index_references(
                    ("x1", "x2", "x3"),
                    ishigami_first,
                    ishigami_total,
                    ishigami_shapley(),
                )
            ),
            {"S1": "absolute", "ST": "absolute", "Sh": "absolute"},
            frozenset({"independent", "smooth", "uniform"}),
            3,
        ),
        Problem(
            "g-function",
            "sensitivity",
            "Sobol G-function, d = 6",
            "prod (|4 x_i - 2| + a_i)/(1 + a_i), a = (0, 1, 4.5, 9, 99, 99).",
            "Analytic: V_i = 1/(3 (1 + a_i)^2).",
            _build_g_function,
            _constant(
                _index_references(
                    tuple(f"x{i + 1}" for i in range(6)), g_first, g_total
                )
            ),
            {"S1": "absolute", "ST": "absolute"},
            frozenset({"independent", "non-smooth", "uniform"}),
            6,
        ),
        Problem(
            "correlated-linear",
            "sensitivity",
            "Linear model, correlated Gaussian inputs",
            "Y = X1 + X2 + X3, sd = (1, 1, 2), corr(X2, X3) = 0.9.",
            "Closed-form Shapley effects s_J' Sigma_JJ^-1 s_J (independent "
            "implementation; equals probstat's linear_gaussian_shapley).",
            _build_correlated_linear,
            _constant({f"Sh[x{i + 1}]": value for i, value in enumerate(shapley)}),
            {"Sh": "absolute"},
            frozenset({"correlated"}),
            3,
        ),
        Problem(
            "ishigami-moments",
            "propagation",
            "Ishigami mean and variance",
            "Mean and variance of the Ishigami output on U(-pi, pi)^3.",
            "Analytic: mean a/2, variance a^2/8 + b pi^4/5 + b^2 pi^8/18 + 1/2.",
            _build_ishigami,
            _constant({"mean": ishigami_mean, "variance": ishigami_variance}),
            {"mean": "relative", "variance": "relative"},
            frozenset({"uniform"}),
            3,
        ),
        Problem(
            "exponential-5d",
            "propagation",
            "Smooth 5-D integrand",
            "E[exp(sum c_i X_i)], X ~ U(0, 1)^5, c = (0.2, 0.4, 0.6, 0.8, 1.0).",
            "Analytic: prod (e^c - 1)/c; variance from prod (e^2c - 1)/(2c).",
            _build_exponential,
            _constant({"mean": exp_mean, "variance": exp_variance}),
            {"mean": "relative", "variance": "relative"},
            frozenset({"uniform"}),
            5,
        ),
        Problem(
            "linear-calibration",
            "calibration",
            "Linear-Gaussian calibration",
            "y = a + b t + eps, 8 points on t in [0, 2], sigma = 0.5 known, "
            "prior N(0, I); true (a, b) drawn from the prior per seed.",
            "Exact conjugate posterior (2 x 2 algebra).",
            _build_calibration,
            _calibration_references,
            {"mean": "scaled", "sd": "relative", "theta": "scaled"},
            frozenset({"calibration"}),
            2,
        ),
    )
