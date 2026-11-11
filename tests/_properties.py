"""Standard-library property-testing harness for the probstat test suite.

The harness draws random *valid* parameters for every distribution family
from generators that encode each family's constructor constraints, runs a
property on a fixed number of cases, and reports the first failing case
with its family, parameters, case index and seed. There is no shrinking:
a failure is reproduced exactly by :func:`reproduce` from the reported
seed, which is enough for the small parameter vectors involved.

Runs are deterministic. Every case seed is a CRC-32 of
``(PROPERTY_SEED, property, family, index)``, so adding a property or a
family never changes the cases of another one. Two environment variables
lengthen local runs without editing the tests:

* ``PROBSTAT_PROPERTY_CASES`` replaces the default case count of the cheap
  properties (``DEFAULT_CASES``); the expensive ones (quadrature, sampling)
  scale in proportion;
* ``PROBSTAT_PROPERTY_SEED`` replaces ``PROPERTY_SEED`` to explore a fresh
  set of cases.

The module is test-side only and imports nothing outside the standard
library and ``probstat`` itself.

References:
    [REF-CLAESSEN-HUGHES-2000] (random property testing),
    [REF-CHEN-ETAL-2018] (metamorphic testing), [REF-HIGHAM-2002]
    (tolerances set from conditioning).

"""

from __future__ import annotations

import math
import os
import random
import unittest
import zlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import probstat.distributions as dist

PROPERTY_SEED = 20261111
DEFAULT_CASES = 50
HEAVY_FRACTION = 5
CASES_VARIABLE = "PROBSTAT_PROPERTY_CASES"
SEED_VARIABLE = "PROBSTAT_PROPERTY_SEED"


def _environment_integer(name: str, default: int) -> int:
    """Return a positive integer from the environment, or ``default``.

    References:
        Test configuration; no equation is evaluated.

    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def property_seed() -> int:
    """Return the master seed (``PROBSTAT_PROPERTY_SEED`` or ``PROPERTY_SEED``).

    References:
        Test configuration; no equation is evaluated.

    """
    return _environment_integer(SEED_VARIABLE, PROPERTY_SEED)


def case_count(heavy: bool = False) -> int:
    """Return the number of cases per property.

    Args:
        heavy: Whether the property is expensive (quadrature or sampling);
            heavy properties run ``1 / HEAVY_FRACTION`` of the cheap count.

    References:
        Test configuration; no equation is evaluated.

    """
    cheap = _environment_integer(CASES_VARIABLE, DEFAULT_CASES)
    return max(3, cheap // HEAVY_FRACTION) if heavy else cheap


def case_seed(property_name: str, family: str, index: int) -> int:
    """Return the deterministic seed of one case.

    References:
        Test configuration; no equation is evaluated.

    """
    key = f"{property_seed()}:{property_name}:{family}:{index}"
    return zlib.crc32(key.encode("utf-8"))


class Draw:
    """Seeded generator of parameter values (a thin ``random.Random`` wrapper).

    References:
        [REF-CLAESSEN-HUGHES-2000]. No equation is evaluated.

    """

    def __init__(self, seed: int) -> None:
        """Seed the generator.

        References:
            Test configuration; no equation is evaluated.

        """
        self.seed = seed
        self.random = random.Random(seed)

    def uniform(self, low: float, high: float) -> float:
        """Return a uniform real in ``[low, high]``."""
        return self.random.uniform(low, high)

    def log_uniform(self, low: float, high: float) -> float:
        """Return a log-uniform positive real in ``[low, high]``."""
        return math.exp(self.random.uniform(math.log(low), math.log(high)))

    def integer(self, low: int, high: int) -> int:
        """Return a uniform integer in ``[low, high]``."""
        return self.random.randint(low, high)

    def choice(self, options: tuple[Any, ...] | list[Any]) -> Any:
        """Return one of ``options``."""
        return options[self.random.randrange(len(options))]

    def probability(self, low: float = 0.0, high: float = 1.0) -> float:
        """Return a probability, log-uniform near zero half of the time."""
        if self.random.random() < 0.5:
            return self.uniform(max(low, 1e-3), min(high, 1.0 - 1e-3))
        return self.log_uniform(max(low, 1e-6), min(high, 0.5))

    def unit_points(self, count: int) -> list[float]:
        """Return ``count`` sorted probabilities spread over ``(0, 1)``.

        Half of them are uniform on ``(1e-3, 1 - 1e-3)``; the rest are
        log-uniform in either tail down to ``1e-9``.

        """
        points: list[float] = []
        for index in range(count):
            if index % 2 == 0:
                points.append(self.uniform(1e-3, 1.0 - 1e-3))
            else:
                tail = self.log_uniform(1e-9, 1e-3)
                points.append(tail if self.random.random() < 0.5 else 1.0 - tail)
        return sorted(points)


# Parameter kinds used by the invalid-parameter (boundary) property.
REAL = "real"
POSITIVE = "positive"
NON_NEGATIVE = "non_negative"
PROBABILITY_OPEN_LEFT = "probability_open_left"
PROBABILITY = "probability"
NON_NEGATIVE_INTEGER = "non_negative_integer"
POSITIVE_INTEGER = "positive_integer"

INVALID_VALUES: dict[str, tuple[Any, ...]] = {
    REAL: (math.nan, math.inf, -math.inf, "1.0"),
    POSITIVE: (0.0, -1.0, math.nan, math.inf, "1.0"),
    NON_NEGATIVE: (-1e-300, -1.0, math.nan, math.inf, "1.0"),
    PROBABILITY_OPEN_LEFT: (0.0, -0.1, 1.1, math.nan),
    PROBABILITY: (-0.1, 1.1, math.nan, math.inf),
    NON_NEGATIVE_INTEGER: (-1, 2.5, True, math.nan),
    POSITIVE_INTEGER: (0, -1, 2.5, True),
}


@dataclass(frozen=True)
class FamilySpec:
    """One distribution family with its valid-parameter generator.

    Args:
        name: Case label (family plus variant).
        family: Exported class (or factory) the spec covers.
        parameters: Draws a valid keyword-argument mapping.
        kind: ``"continuous"``, ``"discrete"`` or ``"categorical"``.
        build: Builds the law from the parameters (default ``family(**p)``).
        constraints: Parameter kinds for the invalid-parameter property;
            empty when the family has coupled constraints tested elsewhere.
        kinks: Points of the support where the density is not smooth (used
            as quadrature breakpoints).
        extreme: Draws parameters at the edges of the operating range.
        sample_size: Draws per sampling case (smaller for laws sampled by
            numerical inversion).
        tail_condition: Relative condition number of the law's survival
            function at a point, in units of the rounding error of its
            ingredients (zero when ``sf`` is evaluated directly).

    References:
        [REF-CLAESSEN-HUGHES-2000]. No equation is evaluated.

    """

    name: str
    family: Any
    parameters: Callable[[Draw], dict[str, Any]]
    kind: str = "continuous"
    build: Callable[[dict[str, Any]], Any] | None = None
    constraints: Mapping[str, str] = field(default_factory=dict)
    kinks: Callable[[Any], tuple[float, ...]] = lambda law: ()
    extreme: Callable[[Draw], dict[str, Any]] | None = None
    sample_size: int = 2000
    tail_condition: Callable[[Any, float], float] = lambda law, x: 0.0

    def make(self, parameters: dict[str, Any]) -> Any:
        """Build the law for ``parameters``."""
        if self.build is not None:
            return self.build(parameters)
        return self.family(**parameters)


@dataclass(frozen=True)
class Case:
    """One generated case: family spec, index, seed, parameters, law, generator."""

    spec: FamilySpec
    index: int
    seed: int
    parameters: dict[str, Any]
    law: Any
    draw: Draw


def cases(
    property_name: str,
    spec: FamilySpec,
    *,
    count: int | None = None,
    extreme: bool = False,
) -> Iterator[Case]:
    """Yield the deterministic cases of one property for one family.

    The parameter generator and the property share one seeded stream, so a
    case is reproduced by its seed alone.

    References:
        [REF-CLAESSEN-HUGHES-2000]. No equation is evaluated.

    """
    total = case_count() if count is None else count
    generator = (
        spec.extreme if extreme and spec.extreme is not None else spec.parameters
    )
    for index in range(total):
        seed = case_seed(property_name, spec.name, index)
        draw = Draw(seed)
        parameters = generator(draw)
        yield Case(spec, index, seed, parameters, spec.make(parameters), draw)


def reproduce(property_name: str, spec: FamilySpec, index: int) -> Case:
    """Return the case ``index`` of ``property_name`` for ``spec``.

    References:
        Test configuration; no equation is evaluated.

    """
    seed = case_seed(property_name, spec.name, index)
    draw = Draw(seed)
    parameters = spec.parameters(draw)
    return Case(spec, index, seed, parameters, spec.make(parameters), draw)


def _describe(value: Any) -> str:
    """Return a compact representation of a parameter value."""
    if isinstance(value, float):
        return repr(value)
    text = repr(value)
    return text if len(text) <= 160 else text[:157] + "..."


def for_all(
    test: unittest.TestCase,
    property_name: str,
    specs: list[FamilySpec],
    check: Callable[[Case], None],
    *,
    heavy: bool = False,
    extreme: bool = False,
    applies: Callable[[FamilySpec], bool] = lambda spec: True,
) -> int:
    """Run ``check`` on every case of every applicable family.

    Each family runs in its own ``subTest`` and stops at its first failing
    case, which is reported with the property, family, case index, seed,
    parameters and the original message. Any exception counts as a
    failure, so an unexpected ``ValueError`` from a valid parameter vector
    is reported in the same way as a broken identity.

    Returns:
        The number of families checked.

    References:
        [REF-CLAESSEN-HUGHES-2000]. No equation is evaluated.

    """
    checked = 0
    count = case_count(heavy)
    for spec in specs:
        if not applies(spec):
            continue
        checked += 1
        with test.subTest(property=property_name, family=spec.name):
            failure: str | None = None
            for case in cases(property_name, spec, count=count, extreme=extreme):
                try:
                    check(case)
                except Exception as error:  # noqa: BLE001 - reported below
                    parameters = ", ".join(
                        f"{key}={_describe(value)}"
                        for key, value in case.parameters.items()
                    )
                    failure = (
                        f"property {property_name!r} failed for {spec.name} "
                        f"(case {case.index} of {count}, case seed {case.seed}, "
                        f"{SEED_VARIABLE}={property_seed()}): {parameters}\n"
                        f"{type(error).__name__}: {error}"
                    )
                    break
            if failure is not None:
                test.fail(failure)
    return checked


# ---------------------------------------------------------------------------
# Family catalogue.
# ---------------------------------------------------------------------------


def _location(draw: Draw) -> float:
    """Return a location parameter in ``[-100, 100]``."""
    return draw.uniform(-100.0, 100.0)


def _scale(draw: Draw) -> float:
    """Return a scale parameter in ``[1e-3, 1e3]``."""
    return draw.log_uniform(1e-3, 1e3)


def _shape_xi(draw: Draw) -> float:
    """Return an extreme-value shape, exactly zero one time in five."""
    return 0.0 if draw.random.random() < 0.2 else draw.uniform(-0.6, 0.6)


def _triangular(draw: Draw) -> dict[str, Any]:
    """Return ``lower <= mode <= upper`` with a strictly positive width."""
    lower = _location(draw)
    upper = lower + _scale(draw)
    roll = draw.random.random()
    if roll < 0.15:
        mode = lower
    elif roll < 0.3:
        mode = upper
    else:
        mode = draw.uniform(lower, upper)
    return {"lower": lower, "mode": mode, "upper": upper}


def _pert(draw: Draw) -> dict[str, Any]:
    """Return PERT parameters ``low <= mode <= high`` and ``lam > 0``."""
    triangle = _triangular(draw)
    return {
        "low": triangle["lower"],
        "mode": triangle["mode"],
        "high": triangle["upper"],
        "lam": draw.log_uniform(0.5, 20.0),
    }


def _truncated_normal(draw: Draw) -> dict[str, Any]:
    """Return a truncated normal with one, two or no finite bounds."""
    mu = _location(draw)
    sigma = _scale(draw)
    style = draw.choice(("both", "lower", "upper", "tail", "none"))
    a = draw.uniform(-4.0, 3.0)
    b = a + draw.log_uniform(0.05, 6.0)
    parameters: dict[str, Any] = {"mu": mu, "sigma": sigma}
    if style == "both":
        parameters.update(lower=mu + a * sigma, upper=mu + b * sigma)
    elif style == "lower":
        parameters.update(lower=mu + a * sigma)
    elif style == "upper":
        parameters.update(upper=mu + b * sigma)
    elif style == "tail":
        parameters.update(lower=mu + draw.uniform(4.0, 12.0) * sigma)
    return parameters


def _empirical(draw: Draw) -> dict[str, Any]:
    """Return a sample of 3 to 30 distinct normal observations."""
    size = draw.integer(3, 30)
    centre = _location(draw)
    spread = _scale(draw)
    values = sorted(
        {centre + spread * draw.random.gauss(0.0, 1.0) for _ in range(size)}
    )
    return {"sample": tuple(values)}


_BASES: tuple[tuple[str, Callable[[Draw], Any]], ...] = (
    ("normal", lambda d: dist.Normal(d.uniform(-5, 5), d.log_uniform(0.1, 10))),
    ("gamma", lambda d: dist.Gamma(d.log_uniform(0.5, 20), d.log_uniform(0.1, 10))),
    ("beta", lambda d: dist.Beta(d.log_uniform(0.5, 10), d.log_uniform(0.5, 10))),
    ("logistic", lambda d: dist.Logistic(d.uniform(-5, 5), d.log_uniform(0.1, 10))),
    ("weibull", lambda d: dist.Weibull(d.log_uniform(0.8, 5), d.log_uniform(0.1, 10))),
)


def _base(draw: Draw) -> Any:
    """Return a smooth continuous base law for the wrappers."""
    return draw.choice(_BASES)[1](draw)


def _affine(draw: Draw) -> dict[str, Any]:
    """Return an affine image ``loc + scale Y`` (``scale`` of either sign)."""
    scale = draw.log_uniform(1e-2, 1e2) * draw.choice((1.0, -1.0))
    return {"base": _base(draw), "loc": _location(draw), "scale": scale}


def _truncated(draw: Draw) -> dict[str, Any]:
    """Return a base law truncated to an interval of its quantiles."""
    base = _base(draw)
    p_low = draw.uniform(0.0, 0.6)
    p_high = draw.uniform(p_low + 0.05, 1.0)
    lower = base.quantile(p_low) if p_low > 0.02 else -math.inf
    upper = base.quantile(p_high) if p_high < 0.98 else math.inf
    return {"base": base, "lower": lower, "upper": upper}


def _mixture(draw: Draw) -> dict[str, Any]:
    """Return a mixture of two or three smooth laws with positive weights."""
    count = draw.integer(2, 3)
    raw = [draw.uniform(0.1, 1.0) for _ in range(count)]
    total = math.fsum(raw)
    weights = [value / total for value in raw]
    weights[-1] = 1.0 - math.fsum(weights[:-1])
    return {
        "components": tuple(_base(draw) for _ in range(count)),
        "weights": tuple(weights),
    }


def _hypergeometric(draw: Draw) -> dict[str, Any]:
    """Return ``0 <= successes, draws <= population``."""
    population = draw.integer(1, 300)
    return {
        "population": population,
        "successes": draw.integer(0, population),
        "draws": draw.integer(0, population),
    }


def _discrete_uniform(draw: Draw) -> dict[str, Any]:
    """Return integer bounds ``low <= high``."""
    low = draw.integer(-1000, 1000)
    return {"low": low, "high": low + draw.integer(0, 500)}


def _categorical(draw: Draw) -> dict[str, Any]:
    """Return a probability mapping over 1 to 12 string labels."""
    count = draw.integer(1, 12)
    raw = [draw.log_uniform(1e-3, 1.0) for _ in range(count)]
    total = math.fsum(raw)
    masses = [value / total for value in raw]
    masses[-1] = 1.0 - math.fsum(masses[:-1])
    return {
        "probabilities": {f"label{index}": mass for index, mass in enumerate(masses)}
    }


def _component_ends(law: Any) -> tuple[float, ...]:
    """Return the finite support ends of a mixture's components (density kinks)."""
    ends: list[float] = []
    for component in law.components:
        for end in (component.quantile(0.0), component.quantile(1.0)):
            if math.isfinite(end):
                ends.append(float(end))
    return tuple(ends)


def _truncation_condition(law: Any, x: float) -> float:
    """Return the cancellation factor of ``S_T(x) = (S(x) - S(b)) / Z``.

    The truncated survival function is a difference of two base-law tail
    values of magnitude ``m`` divided by the mass ``Z``; its relative error
    is about ``m / (Z S_T(x))`` rounding errors [REF-HIGHAM-2002, Section 1.7].

    """
    base = law.base
    lower_cdf = 0.0 if law.lower == -math.inf else float(base.cdf(law.lower))
    if lower_cdf > 0.5:
        magnitude = float(base.sf(x))
        mass = float(base.sf(law.lower)) - (
            0.0 if law.upper == math.inf else float(base.sf(law.upper))
        )
    else:
        magnitude = 1.0 if law.upper == math.inf else float(base.cdf(law.upper))
        mass = magnitude - lower_cdf
    tail = float(law.sf(x))
    return magnitude / (mass * tail) if tail > 0.0 else math.inf


def _with_point(name: str) -> Callable[[Any], tuple[float, ...]]:
    """Return a kink function reading one attribute of the law."""
    return lambda law: (float(getattr(law, name)),)


def continuous_specs() -> list[FamilySpec]:
    """Return the continuous univariate families with their generators.

    The ranges are the well-conditioned operating range of each family
    (``extreme`` widens them); every value satisfies the constructor's
    constraint.

    References:
        [REF-JKB-1994], [REF-JKB-1995]. No equation is evaluated.

    """
    loc_scale = {"mu": REAL, "sigma": POSITIVE}
    return [
        FamilySpec(
            "Normal",
            dist.Normal,
            lambda d: {"mu": _location(d), "sigma": _scale(d)},
            constraints=loc_scale,
            extreme=lambda d: {
                "mu": d.uniform(-1e6, 1e6),
                "sigma": d.log_uniform(1e-8, 1e8),
            },
        ),
        FamilySpec(
            "Cauchy",
            dist.Cauchy,
            lambda d: {"loc": _location(d), "scale": _scale(d)},
            constraints={"loc": REAL, "scale": POSITIVE},
            kinks=_with_point("loc"),
        ),
        FamilySpec(
            "Laplace",
            dist.Laplace,
            lambda d: {"loc": _location(d), "scale": _scale(d)},
            constraints={"loc": REAL, "scale": POSITIVE},
            kinks=_with_point("loc"),
        ),
        FamilySpec(
            "Logistic",
            dist.Logistic,
            lambda d: {"mu": _location(d), "s": _scale(d)},
            constraints={"mu": REAL, "s": POSITIVE},
        ),
        FamilySpec(
            "Gumbel",
            dist.Gumbel,
            lambda d: {
                "mu": _location(d),
                "beta": _scale(d),
                "kind": d.choice(("max", "min")),
            },
            constraints={"mu": REAL, "beta": POSITIVE},
        ),
        FamilySpec(
            "LogNormal",
            dist.LogNormal,
            lambda d: {"mu": d.uniform(-3.0, 3.0), "sigma": d.log_uniform(0.05, 2.0)},
            constraints=loc_scale,
            extreme=lambda d: {
                "mu": d.uniform(-30.0, 30.0),
                "sigma": d.log_uniform(1e-4, 5.0),
            },
        ),
        FamilySpec(
            "Exponential",
            dist.Exponential,
            lambda d: {"rate": _scale(d)},
            constraints={"rate": POSITIVE},
            extreme=lambda d: {"rate": d.log_uniform(1e-12, 1e12)},
        ),
        FamilySpec(
            "Gamma",
            dist.Gamma,
            lambda d: {
                "shape": d.log_uniform(0.1, 100.0),
                "rate": d.log_uniform(1e-2, 1e2),
            },
            constraints={"shape": POSITIVE, "rate": POSITIVE},
            extreme=lambda d: {
                "shape": d.choice((d.log_uniform(1e-3, 0.1), d.log_uniform(1e2, 1e6))),
                "rate": d.log_uniform(1e-6, 1e6),
            },
        ),
        FamilySpec(
            "ChiSquare",
            dist.ChiSquare,
            lambda d: {"nu": d.log_uniform(0.2, 200.0)},
            constraints={"nu": POSITIVE},
            extreme=lambda d: {
                "nu": d.choice((d.log_uniform(1e-3, 0.2), d.log_uniform(1e3, 1e6)))
            },
        ),
        FamilySpec(
            "Beta",
            dist.Beta,
            lambda d: {
                "alpha": d.log_uniform(0.1, 50.0),
                "beta": d.log_uniform(0.1, 50.0),
            },
            constraints={"alpha": POSITIVE, "beta": POSITIVE},
            extreme=lambda d: {
                "alpha": d.choice((d.log_uniform(1e-3, 0.1), d.log_uniform(50.0, 1e5))),
                "beta": d.choice((d.log_uniform(1e-3, 0.1), d.log_uniform(50.0, 1e5))),
            },
        ),
        FamilySpec(
            "FisherF",
            dist.FisherF,
            lambda d: {
                "nu1": d.log_uniform(0.5, 200.0),
                "nu2": d.log_uniform(0.5, 200.0),
            },
            constraints={"nu1": POSITIVE, "nu2": POSITIVE},
            extreme=lambda d: {
                "nu1": d.log_uniform(1e-2, 1e5),
                "nu2": d.log_uniform(1e-2, 1e5),
            },
        ),
        FamilySpec(
            "StudentT",
            dist.StudentT,
            lambda d: {
                "nu": d.log_uniform(0.5, 200.0),
                "mu": _location(d),
                "sigma": _scale(d),
            },
            constraints={"nu": POSITIVE, "mu": REAL, "sigma": POSITIVE},
            extreme=lambda d: {
                "nu": d.choice((d.log_uniform(1e-2, 0.5), d.log_uniform(1e3, 1e9))),
                "mu": 0.0,
                "sigma": 1.0,
            },
        ),
        FamilySpec(
            "Weibull",
            dist.Weibull,
            lambda d: {
                "shape": d.log_uniform(0.3, 20.0),
                "scale": d.log_uniform(1e-2, 1e2),
            },
            constraints={"shape": POSITIVE, "scale": POSITIVE},
            extreme=lambda d: {
                "shape": d.choice(
                    (d.log_uniform(0.02, 0.3), d.log_uniform(20.0, 500.0))
                ),
                "scale": d.log_uniform(1e-6, 1e6),
            },
        ),
        FamilySpec(
            "Rayleigh",
            dist.Rayleigh,
            lambda d: {"scale": _scale(d)},
            constraints={"scale": POSITIVE},
        ),
        FamilySpec(
            "Pareto",
            dist.Pareto,
            lambda d: {
                "scale": d.log_uniform(1e-2, 1e2),
                "shape": d.log_uniform(0.3, 20.0),
            },
            constraints={"scale": POSITIVE, "shape": POSITIVE},
            kinks=_with_point("scale"),
            extreme=lambda d: {
                "scale": d.log_uniform(1e-6, 1e6),
                "shape": d.choice((d.log_uniform(1e-2, 0.3), d.log_uniform(20.0, 1e3))),
            },
        ),
        FamilySpec(
            "GeneralizedExtremeValue",
            dist.GeneralizedExtremeValue,
            lambda d: {"mu": _location(d), "sigma": _scale(d), "xi": _shape_xi(d)},
            constraints={"mu": REAL, "sigma": POSITIVE, "xi": REAL},
            extreme=lambda d: {
                "mu": 0.0,
                "sigma": 1.0,
                "xi": d.choice(
                    (d.uniform(-3.0, -0.6), d.uniform(0.6, 3.0), d.uniform(-1e-9, 1e-9))
                ),
            },
        ),
        FamilySpec(
            "GeneralizedPareto",
            dist.GeneralizedPareto,
            lambda d: {"mu": _location(d), "sigma": _scale(d), "xi": _shape_xi(d)},
            constraints={"mu": REAL, "sigma": POSITIVE, "xi": REAL},
            kinks=_with_point("mu"),
            extreme=lambda d: {
                "mu": 0.0,
                "sigma": 1.0,
                "xi": d.choice(
                    (d.uniform(-3.0, -0.6), d.uniform(0.6, 3.0), d.uniform(-1e-9, 1e-9))
                ),
            },
        ),
        FamilySpec(
            "Uniform",
            dist.Uniform,
            lambda d: (lambda low: {"lower": low, "upper": low + _scale(d)})(
                _location(d)
            ),
        ),
        FamilySpec(
            "Triangular",
            dist.Triangular,
            _triangular,
            kinks=_with_point("mode"),
        ),
        FamilySpec("PERT", dist.PERT, _pert),
        FamilySpec("TruncatedNormal", dist.TruncatedNormal, _truncated_normal),
        FamilySpec(
            "Empirical",
            dist.Empirical,
            _empirical,
            kinks=lambda law: tuple(law.values),
            sample_size=1000,
        ),
        FamilySpec("Affine", dist.Affine, _affine),
        FamilySpec(
            "Truncated",
            dist.Truncated,
            _truncated,
            tail_condition=_truncation_condition,
        ),
        # Mixture variates are drawn by inverting component quantiles
        # (Brent's method for gamma and beta components), hence fewer draws.
        FamilySpec(
            "Mixture",
            dist.Mixture,
            _mixture,
            kinks=_component_ends,
            sample_size=200,
        ),
    ]


def discrete_specs() -> list[FamilySpec]:
    """Return the integer-valued families with their generators.

    References:
        [REF-JKB-2005]. No equation is evaluated.

    """
    return [
        FamilySpec(
            "Bernoulli",
            dist.Bernoulli,
            lambda d: {"p": d.probability()},
            kind="discrete",
            constraints={"p": PROBABILITY},
        ),
        FamilySpec(
            "Binomial",
            dist.Binomial,
            lambda d: {"n": d.integer(0, 300), "p": d.probability()},
            kind="discrete",
            constraints={"n": NON_NEGATIVE_INTEGER, "p": PROBABILITY},
            extreme=lambda d: {
                "n": d.integer(10**5, 10**7),
                "p": d.choice((d.log_uniform(1e-9, 1e-5), d.uniform(0.3, 0.7))),
            },
        ),
        FamilySpec(
            "Geometric",
            dist.Geometric,
            lambda d: {"p": d.uniform(0.02, 1.0) if d.random.random() < 0.7 else 1.0},
            kind="discrete",
            constraints={"p": PROBABILITY_OPEN_LEFT},
            extreme=lambda d: {"p": d.log_uniform(1e-9, 1e-3)},
        ),
        FamilySpec(
            "NegativeBinomial",
            dist.NegativeBinomial,
            lambda d: {"r": d.integer(1, 60), "p": d.uniform(0.05, 1.0)},
            kind="discrete",
            constraints={"r": POSITIVE_INTEGER, "p": PROBABILITY_OPEN_LEFT},
            extreme=lambda d: {
                "r": d.integer(1, 10**4),
                "p": d.log_uniform(1e-6, 1e-2),
            },
        ),
        FamilySpec(
            "Poisson",
            dist.Poisson,
            lambda d: {"lam": d.log_uniform(1e-2, 500.0)},
            kind="discrete",
            constraints={"lam": NON_NEGATIVE},
            extreme=lambda d: {
                "lam": d.choice((d.log_uniform(1e-12, 1e-2), d.log_uniform(1e4, 1e8)))
            },
        ),
        FamilySpec(
            "Hypergeometric",
            dist.Hypergeometric,
            _hypergeometric,
            kind="discrete",
            extreme=lambda d: (
                lambda n: {
                    "population": n,
                    "successes": d.integer(0, n),
                    "draws": d.integer(0, n),
                }
            )(d.integer(10**4, 10**5)),
        ),
        FamilySpec(
            "DiscreteUniform",
            dist.DiscreteUniform,
            _discrete_uniform,
            kind="discrete",
        ),
    ]


def categorical_specs() -> list[FamilySpec]:
    """Return the label-valued families (no order, so no distribution function).

    References:
        [REF-ROSS-2019, Section 4.6]. No equation is evaluated.

    """
    return [
        FamilySpec(
            "Categorical",
            dist.Categorical,
            _categorical,
            kind="categorical",
        )
    ]


def all_specs() -> list[FamilySpec]:
    """Return every univariate family spec."""
    return continuous_specs() + discrete_specs() + categorical_specs()


# Classes exported from ``probstat.distributions`` that are not laws.
NON_LAW_EXPORTS = frozenset(
    {
        "ContinuousDistribution",
        "DiscreteDistribution",
        "FiniteSupport",
        "IntegerSupport",
        "RealSupport",
    }
)
