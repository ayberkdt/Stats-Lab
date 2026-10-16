"""Stage chaining: turn posterior draws into an input law, and hash input laws.

After a calibration the posterior is known only through MCMC draws, while
every forward stage needs a :class:`~probstat.modeling.JointInput` (marginals
plus a Gaussian copula). :func:`posterior_joint_input` builds one from the
draws:

- ``method="gaussian_copula"``: each calibrated input gets the parametric
  family with the best AIC among ``candidates``
  (:func:`~probstat.modeling.fit_joint_input`), and the dependence is a
  Gaussian copula fitted from Spearman ranks.
- ``method="empirical"``: each calibrated input gets the
  :class:`EmpiricalMarginal` of its draws (the continuous law whose quantile
  function is the type-7 sample quantile), and the dependence is a Gaussian
  copula fitted from normal scores.

Both are approximations of the posterior: a Gaussian copula cannot represent
curved (banana-shaped) or tail dependence, the parametric route adds
family misfit, and the empirical route has bounded support (the range of the
draws) and Monte Carlo noise. Inputs that were not adopted keep their prior
marginals and their prior dependence among themselves; they are made
independent of the adopted inputs.

:func:`input_law_hash` gives a stable short hash of a joint input law (input
names, marginal families and parameters, copula correlation) so a report can
state exactly which law each stage used.
"""

from __future__ import annotations

import bisect
import dataclasses
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from probstat.foundations.validation import require_finite_real
from probstat.modeling import (
    ContinuousMarginal,
    JointInput,
    RandomInput,
    fit_gaussian_copula,
    fit_joint_input,
)

PosteriorMethod = Literal["gaussian_copula", "empirical"]
POSTERIOR_METHODS = ("gaussian_copula", "empirical")
DEFAULT_POSTERIOR_FAMILIES = ("Normal", "LogNormal", "Gamma", "Weibull", "Gumbel")


@dataclass(frozen=True, slots=True)
class EmpiricalMarginal:
    """Continuous law whose quantile function is the type-7 sample quantile.

    For distinct sorted support points ``x_1 < ... < x_m`` the CDF is the
    piecewise-linear interpolation of ``F(x_k) = (k - 1) / (m - 1)``: each
    gap ``[x_k, x_{k+1}]`` carries probability ``1 / (m - 1)`` spread
    uniformly, so the law is continuous with a piecewise-constant density.
    The support is ``[x_1, x_m]``: tails beyond the observed range are cut.
    Repeated values (Metropolis rejections) are collapsed to one support
    point, which slightly flattens the density where the chain stuck.

    Args:
        points: At least two finite values (sorted and de-duplicated on
            construction).

    References:
        [REF-HYNDMAN-FAN-1996].

    """

    points: tuple[float, ...]
    _mean: float = field(init=False, repr=False, compare=False)
    _variance: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Sort, de-duplicate, and cache the moments.

        Raises:
            ValueError: If fewer than two distinct finite values are given.

        Equation:
            [EQ-WF-EMPIRICAL-001].

        References:
            [REF-HYNDMAN-FAN-1996].

        """
        values = sorted({require_finite_real(v, name="point") for v in self.points})
        if len(values) < 2:
            raise ValueError("an empirical marginal needs two distinct values")
        weight = 1.0 / (len(values) - 1)
        mean = math.fsum(
            weight * 0.5 * (a + b) for a, b in zip(values, values[1:], strict=False)
        )
        second = math.fsum(
            weight * (a * a + a * b + b * b) / 3.0
            for a, b in zip(values, values[1:], strict=False)
        )
        object.__setattr__(self, "points", tuple(values))
        object.__setattr__(self, "_mean", mean)
        object.__setattr__(self, "_variance", max(0.0, second - mean * mean))

    @property
    def mean(self) -> float:
        """Return the exact mean of the interpolated law.

        Equation:
            [EQ-WF-EMPIRICAL-001].

        References:
            [REF-HYNDMAN-FAN-1996].

        """
        return self._mean

    @property
    def variance(self) -> float:
        """Return the exact variance of the interpolated law.

        Equation:
            [EQ-WF-EMPIRICAL-001].

        References:
            [REF-HYNDMAN-FAN-1996].

        """
        return self._variance

    def cdf(self, x: float) -> float:
        """Return the piecewise-linear CDF.

        Equation:
            [EQ-WF-EMPIRICAL-001].

        References:
            [REF-HYNDMAN-FAN-1996].

        """
        value = require_finite_real(x, name="x")
        points = self.points
        if value <= points[0]:
            return 0.0
        if value >= points[-1]:
            return 1.0
        k = bisect.bisect_right(points, value) - 1
        a, b = points[k], points[k + 1]
        return (k + (value - a) / (b - a)) / (len(points) - 1)

    def pdf(self, x: float) -> float:
        """Return the piecewise-constant density (zero outside the support).

        Equation:
            [EQ-WF-EMPIRICAL-001].

        References:
            [REF-HYNDMAN-FAN-1996].

        """
        value = require_finite_real(x, name="x")
        points = self.points
        if value < points[0] or value > points[-1]:
            return 0.0
        k = min(bisect.bisect_right(points, value) - 1, len(points) - 2)
        return 1.0 / ((len(points) - 1) * (points[k + 1] - points[k]))

    def quantile(self, probability: float) -> float:
        """Return the type-7 quantile of the support points.

        Equation:
            [EQ-WF-EMPIRICAL-001], [EQ-STAT-QUANT-001].

        References:
            [REF-HYNDMAN-FAN-1996].

        """
        p = require_finite_real(probability, name="probability")
        if not 0.0 <= p <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        points = self.points
        position = p * (len(points) - 1)
        k = min(int(position), len(points) - 2)
        return points[k] + (position - k) * (points[k + 1] - points[k])


def _stable(value: object) -> object:
    """Return a canonical, address-free description of a value.

    Dataclasses are described by their type and init fields, floats by
    ``repr`` (exact round trip), containers recursively; any other object
    by its type name and its non-callable public attributes.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, tuple | list):
        return [_stable(item) for item in value]
    if isinstance(value, Mapping):
        return sorted((str(k), _stable(v)) for k, v in value.items())
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [
            kind,
            [
                (item.name, _stable(getattr(value, item.name)))
                for item in dataclasses.fields(value)
                if item.init
            ],
        ]
    public = sorted(
        (name, _stable(getattr(value, name)))
        for name in dir(value)
        if not name.startswith("_")
        and isinstance(getattr(value, name, None), bool | int | float | str)
    )
    return [kind, public]


def input_law_hash(inputs: JointInput) -> str:
    """Return a stable 12-hex-digit hash of a joint input law.

    The hash covers input names, marginal types and parameters, and the
    copula correlation, so equal laws built in different sessions share it.

    Args:
        inputs: Joint input law.

    Returns:
        First 12 hexadecimal digits of a SHA-256 digest.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    description = [
        [item.name, _stable(item.distribution)] for item in inputs.inputs
    ]
    payload = repr([description, _stable(inputs.correlation)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _thin(count: int, limit: int) -> list[int]:
    """Return at most ``limit`` evenly spaced indices of ``range(count)``.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 11]. No equation is evaluated.

    """
    if count <= limit:
        return list(range(count))
    return [int(i * count / limit) for i in range(limit)]


def posterior_joint_input(
    prior: JointInput,
    draws: Sequence[Mapping[str, float]],
    names: Sequence[str] | None = None,
    *,
    method: PosteriorMethod = "gaussian_copula",
    candidates: Sequence[str] = DEFAULT_POSTERIOR_FAMILIES,
    max_draws: int = 4000,
) -> JointInput:
    """Approximate a posterior by a marginals-plus-Gaussian-copula input law.

    Args:
        prior: Current joint input law (provides input order, descriptions,
            units, and the law of inputs that are not adopted).
        draws: Posterior draws keyed by input name (extra keys such as noise
            scales are ignored).
        names: Inputs to replace (default: every prior input present in the
            draws).
        method: ``"gaussian_copula"`` (parametric marginals by AIC) or
            ``"empirical"`` (:class:`EmpiricalMarginal`).
        candidates: Candidate family names for ``"gaussian_copula"``.
        max_draws: Draws used after even thinning (at least 20).

    Returns:
        New joint input law in the prior's input order.

    Raises:
        ValueError: If the method, names, or draw count are invalid.

    Equation:
        [EQ-INF-IC-001], [EQ-MOD-COPULA-FIT-001], [EQ-MOD-COPULA-FIT-002],
        [EQ-WF-EMPIRICAL-001].

    References:
        [REF-GENEST-GHOUDI-RIVEST-1995], [REF-LIU-DERKIUREGHIAN-1986],
        [REF-HYNDMAN-FAN-1996].

    """
    if method not in POSTERIOR_METHODS:
        raise ValueError(f"method must be one of {POSTERIOR_METHODS}")
    if isinstance(max_draws, bool) or not isinstance(max_draws, int):
        raise ValueError("max_draws must be an integer")
    if max_draws < 20:
        raise ValueError("max_draws must be at least 20")
    samples = tuple(draws)
    if len(samples) < 20:
        raise ValueError("at least 20 posterior draws are needed")
    available = [name for name in prior.names if name in samples[0]]
    chosen = list(available if names is None else names)
    if not chosen:
        raise ValueError("no prior input appears in the posterior draws")
    for name in chosen:
        if name not in prior.names:
            raise ValueError(f"{name!r} is not an input of the model")
        if name not in samples[0]:
            raise ValueError(f"{name!r} has no posterior draws")
    if len(set(chosen)) != len(chosen):
        raise ValueError("adopted input names must be distinct")
    adopted = [name for name in prior.names if name in chosen]
    keep = _thin(len(samples), max_draws)
    data = {name: [float(samples[i][name]) for i in keep] for name in adopted}
    if method == "gaussian_copula":
        fit = fit_joint_input(
            data,
            candidates=tuple(candidates),
            dependence="spearman" if len(adopted) > 1 else "independent",
        )
        marginals: dict[str, ContinuousMarginal] = {
            item.name: item.distribution for item in fit.joint_input.inputs
        }
        block = fit.copula.latent_correlation
    else:
        marginals = {name: EmpiricalMarginal(tuple(data[name])) for name in adopted}
        block = (
            fit_gaussian_copula(data, method="normal-scores").latent_correlation
            if len(adopted) > 1
            else ((1.0,),)
        )
    position = {name: index for index, name in enumerate(adopted)}
    new_inputs: list[RandomInput] = []
    for item in prior.inputs:
        if item.name in marginals:
            new_inputs.append(
                RandomInput(
                    item.name, marginals[item.name], item.description, item.unit
                )
            )
        else:
            new_inputs.append(item)
    size = len(new_inputs)
    matrix = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    names_in_order = [item.name for item in new_inputs]
    for i, first in enumerate(names_in_order):
        for j, second in enumerate(names_in_order):
            if i == j:
                continue
            if first in position and second in position:
                matrix[i][j] = block[position[first]][position[second]]
            elif (
                first not in position
                and second not in position
                and prior.correlation is not None
            ):
                matrix[i][j] = prior.correlation[i][j]
    independent = all(
        matrix[i][j] == 0.0 for i in range(size) for j in range(size) if i != j
    )
    return JointInput(
        tuple(new_inputs),
        None if independent else tuple(tuple(row) for row in matrix),
    )
