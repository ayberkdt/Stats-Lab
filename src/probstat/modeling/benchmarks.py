"""Standard benchmark models for reliability and sensitivity analysis.

Each builder returns a :class:`~probstat.modeling.ProbabilisticModel` with
its published input law, so a method can be checked against values reported
in the literature and compared with other methods on the same object.

- :func:`flood_model`: the dike overflow and maintenance-cost model of a
  river section (eight independent inputs, two outputs).
- :func:`four_branch_model`: the four-branch series limit state on two
  standard-normal inputs, with four symmetric design points.
- :func:`ishigami_model`: the three-input Ishigami function with uniform
  inputs on ``[-pi, pi]`` and known Sobol indices.

References:
    [REF-IOOSS-LEMAITRE-2015], [REF-WAARTS-2000],
    [REF-ISHIGAMI-HOMMA-1990], [REF-SALTELLI-2008, Chapter 4].

"""

from __future__ import annotations

import math
from collections.abc import Mapping

from probstat.distributions import (
    Gumbel,
    Normal,
    Triangular,
    Truncated,
    TruncatedNormal,
    Uniform,
)
from probstat.uq import QuantityOfInterest

from .inputs import JointInput, RandomInput
from .model import ProbabilisticModel

__all__ = [
    "FLOOD_INPUT_NAMES",
    "FLOOD_NOMINAL",
    "FLOOD_NOMINAL_STANDARD_DEVIATIONS",
    "FOUR_BRANCH_REFERENCE_PROBABILITY",
    "ISHIGAMI_FIRST_ORDER",
    "ISHIGAMI_TOTAL",
    "flood_cost",
    "flood_model",
    "flood_overflow",
    "four_branch",
    "four_branch_model",
    "ishigami",
    "ishigami_model",
]

FLOOD_INPUT_NAMES: tuple[str, ...] = ("Q", "Ks", "Zv", "Zm", "Hd", "Cb", "L", "B")
"""Input order of the flood model: discharge, Strickler coefficient,
downstream and upstream river-bed levels, dike height, bank level, section
length, and river width."""

FLOOD_NOMINAL: Mapping[str, float] = {
    "Q": 1355.0,
    "Ks": 30.5,
    "Zv": 50.0,
    "Zm": 55.0,
    "Hd": 8.0,
    "Cb": 55.5,
    "L": 5000.0,
    "B": 300.0,
}
"""Nominal values used for deterministic (local) studies of the flood model."""

FLOOD_NOMINAL_STANDARD_DEVIATIONS: Mapping[str, float] = {
    "Q": 559.8,
    "Ks": 7.28,
    "Zv": 0.41,
    "Zm": 0.41,
    "Hd": math.sqrt(1.0 / 3.0),
    "Cb": 0.205,
    "L": 4.1,
    "B": 2.05,
}
"""Standard deviations paired with :data:`FLOOD_NOMINAL` in local studies."""

FOUR_BRANCH_REFERENCE_PROBABILITY = 2.2245e-3
"""Failure probability of :func:`four_branch` with ``k = 7``: crude Monte Carlo
with ``2 x 10^8`` draws gives ``2.2245e-3`` with a standard error of
``3.3e-6``, in line with the ``2.22e-3`` reported in the literature."""

ISHIGAMI_FIRST_ORDER: tuple[float, float, float] = (0.3139, 0.4424, 0.0)
"""First-order Sobol indices of :func:`ishigami` with ``a = 7, b = 0.1``."""

ISHIGAMI_TOTAL: tuple[float, float, float] = (0.5576, 0.4424, 0.2437)
"""Total-effect Sobol indices of :func:`ishigami` with ``a = 7, b = 0.1``."""


def flood_overflow(parameters: Mapping[str, float]) -> float:
    """Return the dike overflow ``S`` (m) of the flood model.

    ``H = (Q / (Ks B sqrt((Zm - Zv) / L)))^0.6`` is the water height and
    ``S = Zv + H - Hd - Cb``; the dike overflows when ``S > 0``.

    Args:
        parameters: Mapping with the eight names of :data:`FLOOD_INPUT_NAMES`.

    Returns:
        Overflow in metres (negative values are a freeboard).

    Raises:
        ValueError: If ``Zm <= Zv`` or a positive input is not positive.

    Equation:
        [EQ-MOD-FLOOD-001].

    References:
        [REF-IOOSS-LEMAITRE-2015].

    """
    q, ks, zv, zm = (
        parameters["Q"],
        parameters["Ks"],
        parameters["Zv"],
        parameters["Zm"],
    )
    length, width = parameters["L"], parameters["B"]
    if zm <= zv:
        raise ValueError("the upstream level Zm must exceed the downstream level Zv")
    if q <= 0.0 or ks <= 0.0 or length <= 0.0 or width <= 0.0:
        raise ValueError("Q, Ks, L, and B must be positive")
    height = (q / (ks * width * math.sqrt((zm - zv) / length))) ** 0.6
    return float(zv + height - parameters["Hd"] - parameters["Cb"])


def flood_cost(parameters: Mapping[str, float]) -> float:
    """Return the annual dike maintenance cost ``C`` (million euros).

    ``C = 1`` when the dike overflows, otherwise
    ``0.2 + 0.8 (1 - exp(-1000 / S^4))``, plus ``max(Hd, 8) / 20`` for the
    dike itself. At ``S = 0`` the no-overflow branch tends to ``1``, which is
    the value returned.

    Args:
        parameters: Mapping with the eight names of :data:`FLOOD_INPUT_NAMES`.

    Returns:
        Maintenance cost in million euros.

    Raises:
        ValueError: As :func:`flood_overflow`.

    Equation:
        [EQ-MOD-FLOOD-002].

    References:
        [REF-IOOSS-LEMAITRE-2015].

    """
    overflow = flood_overflow(parameters)
    if overflow >= 0.0:
        damage = 1.0
    else:
        damage = 0.2 + 0.8 * -math.expm1(-1000.0 / overflow**4)
    return damage + max(parameters["Hd"], 8.0) / 20.0


def _flood_response(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return both flood outputs for one input mapping.

    Args:
        parameters: Mapping with the eight flood inputs.

    Returns:
        Mapping with ``overflow`` (m) and ``cost`` (million euros).

    References:
        [REF-IOOSS-LEMAITRE-2015]. Combines [EQ-MOD-FLOOD-001] and
        [EQ-MOD-FLOOD-002].

    """
    return {"overflow": flood_overflow(parameters), "cost": flood_cost(parameters)}


def flood_model() -> ProbabilisticModel:
    """Return the flood model with its eight independent input laws.

    Inputs, all independent: discharge ``Q`` Gumbel (mode 1013, scale 555)
    truncated to ``[500, 3000]`` m3/s; Strickler coefficient ``Ks`` normal
    (30, 8) truncated below at 15; river-bed levels ``Zv`` triangular
    (49, 50, 51) m and ``Zm`` triangular (54, 55, 56) m; dike height ``Hd``
    uniform on ``[7, 9]`` m; bank level ``Cb`` triangular (55, 55.5, 56) m;
    length ``L`` triangular (4990, 5000, 5010) m; width ``B`` triangular
    (295, 300, 305) m. Outputs: ``overflow`` (m) and ``cost`` (million
    euros).

    Returns:
        Probabilistic model named ``"flood"``.

    Equation:
        [EQ-MOD-FLOOD-001], [EQ-MOD-FLOOD-002], [EQ-DIST-TRUNC-001].

    References:
        [REF-IOOSS-LEMAITRE-2015].

    """
    inputs = JointInput.independent(
        RandomInput(
            "Q",
            Truncated(Gumbel(1013.0, 555.0), 500.0, 3000.0),
            "maximum annual discharge",
            "m3/s",
        ),
        RandomInput(
            "Ks", TruncatedNormal(30.0, 8.0, lower=15.0), "Strickler coefficient"
        ),
        RandomInput(
            "Zv", Triangular(49.0, 50.0, 51.0), "downstream river-bed level", "m"
        ),
        RandomInput(
            "Zm", Triangular(54.0, 55.0, 56.0), "upstream river-bed level", "m"
        ),
        RandomInput("Hd", Uniform(7.0, 9.0), "dike height", "m"),
        RandomInput("Cb", Triangular(55.0, 55.5, 56.0), "bank level", "m"),
        RandomInput("L", Triangular(4990.0, 5000.0, 5010.0), "section length", "m"),
        RandomInput("B", Triangular(295.0, 300.0, 305.0), "river width", "m"),
    )
    return ProbabilisticModel(
        inputs,
        _flood_response,
        (
            QuantityOfInterest("overflow", "water level above the dike crest", "m"),
            QuantityOfInterest("cost", "annual dike maintenance cost", "MEUR"),
        ),
        name="flood",
    )


def four_branch(u1: float, u2: float, k: float = 7.0) -> float:
    """Return the four-branch series limit state (failure when ``<= 0``).

    ``g = min(3 + (u1 - u2)^2 / 10 - |u1 + u2| / sqrt(2),
    k / sqrt(2) - |u1 - u2|)``. With ``k = 7`` the failure probability under
    independent standard normals is about ``2.22e-3``; the four design points
    lie at distance 3 (two of them) and ``3.5`` (two of them), so a
    single-point FORM gives ``Phi(-3) = 1.35e-3``.

    Args:
        u1: First standard-normal coordinate.
        u2: Second standard-normal coordinate.
        k: Positive offset of the two outer branches.

    Returns:
        Limit-state value.

    Equation:
        [EQ-REL-WAARTS-001].

    References:
        [REF-WAARTS-2000], [REF-ECHARD-GAYTON-LEMAIRE-2011].

    """
    inner = 3.0 + (u1 - u2) ** 2 / 10.0 - abs(u1 + u2) / math.sqrt(2.0)
    outer = k / math.sqrt(2.0) - abs(u1 - u2)
    return min(inner, outer)


def _four_branch_response(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return the four-branch value for one input mapping.

    Args:
        parameters: Mapping with ``u1``, ``u2`` and the fixed offset ``k``.

    Returns:
        Mapping with the limit-state value ``g``.

    References:
        [REF-WAARTS-2000]. Evaluates [EQ-REL-WAARTS-001].

    """
    return {"g": four_branch(parameters["u1"], parameters["u2"], parameters["k"])}


def four_branch_model(k: float = 7.0) -> ProbabilisticModel:
    """Return the four-branch limit state on two independent standard normals.

    Args:
        k: Positive offset of the outer branches (7 in the course example,
            6 in the AK-MCS paper).

    Returns:
        Probabilistic model named ``"four-branch"`` with output ``g``.

    Raises:
        ValueError: If ``k`` is not finite and positive.

    Equation:
        [EQ-REL-WAARTS-001].

    References:
        [REF-WAARTS-2000], [REF-ECHARD-GAYTON-LEMAIRE-2011].

    """
    offset = float(k)
    if not math.isfinite(offset) or offset <= 0.0:
        raise ValueError("k must be finite and positive")
    inputs = JointInput.independent(
        RandomInput("u1", Normal(0.0, 1.0)),
        RandomInput("u2", Normal(0.0, 1.0)),
    )
    return ProbabilisticModel(
        inputs,
        _four_branch_response,
        (QuantityOfInterest("g", "four-branch limit-state value"),),
        name="four-branch",
        fixed={"k": offset},
    )


def ishigami(x1: float, x2: float, x3: float, a: float = 7.0, b: float = 0.1) -> float:
    """Return the Ishigami function ``sin x1 + a sin^2 x2 + b x3^4 sin x1``.

    Args:
        x1: First input.
        x2: Second input.
        x3: Third input.
        a: Weight of the additive ``x2`` term.
        b: Weight of the ``x1``-``x3`` interaction.

    Returns:
        Function value.

    Equation:
        [EQ-LAB-ISHIGAMI-001].

    References:
        [REF-ISHIGAMI-HOMMA-1990], [REF-SALTELLI-2008, Chapter 4].

    """
    return math.sin(x1) + a * math.sin(x2) ** 2 + b * x3**4 * math.sin(x1)


def _ishigami_response(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return the Ishigami value for one input mapping.

    Args:
        parameters: Mapping with ``x1``, ``x2``, ``x3``, ``a``, and ``b``.

    Returns:
        Mapping with the output ``y``.

    References:
        [REF-ISHIGAMI-HOMMA-1990]. Evaluates [EQ-LAB-ISHIGAMI-001].

    """
    return {
        "y": ishigami(
            parameters["x1"],
            parameters["x2"],
            parameters["x3"],
            parameters["a"],
            parameters["b"],
        )
    }


def ishigami_model(a: float = 7.0, b: float = 0.1) -> ProbabilisticModel:
    """Return the Ishigami function with inputs independent uniform on ``[-pi, pi]``.

    With ``a = 7`` and ``b = 0.1`` the first-order indices are
    :data:`ISHIGAMI_FIRST_ORDER` and the total indices :data:`ISHIGAMI_TOTAL`:
    ``x3`` acts only through its interaction with ``x1``.

    Args:
        a: Finite weight of the ``x2`` term.
        b: Finite weight of the ``x1``-``x3`` interaction.

    Returns:
        Probabilistic model named ``"ishigami"`` with output ``y``.

    Raises:
        ValueError: If ``a`` or ``b`` is not finite.

    Equation:
        [EQ-LAB-ISHIGAMI-001].

    References:
        [REF-ISHIGAMI-HOMMA-1990], [REF-SALTELLI-2008, Chapter 4].

    """
    weights = float(a), float(b)
    if not all(math.isfinite(value) for value in weights):
        raise ValueError("a and b must be finite")
    inputs = JointInput.independent(
        *(RandomInput(name, Uniform(-math.pi, math.pi)) for name in ("x1", "x2", "x3"))
    )
    return ProbabilisticModel(
        inputs,
        _ishigami_response,
        (QuantityOfInterest("y", "Ishigami function value"),),
        name="ishigami",
        fixed={"a": weights[0], "b": weights[1]},
    )
