"""Model handle, bijections, and the traced intermediate representation.

A model is a plain function ``model(m)`` that declares random quantities
through the handle ``m``. The same function is executed in four modes:

``trace``
    Once, to build the :class:`ModelIR`. Every latent site takes the value
    its bijection assigns to ``u = 0`` (so no randomness is involved), and the
    ordered site list, sizes, supports, and bijections are recorded.
``density``
    For every log-density evaluation. Latent values are read from an
    unconstrained vector ``u`` in site order and mapped through the site's
    bijection; prior, likelihood, and log-Jacobian terms are accumulated.
``generate``
    For prior and posterior predictive simulation. Latent sites are either
    fixed by name (posterior predictive) or drawn from their priors (prior
    predictive); observed sites are always drawn from their distributions.

The bijection of a site is chosen from the support of the distribution that
is active *at that evaluation*, so bounds that depend on earlier latent
values (``Uniform(0, theta)``) are handled; the map is lower triangular in
site order, so the log-Jacobian is the sum of the per-site terms.

Floating-point limit: on an interval ``(a, b)`` the map
``x = a + (b - a) sigmoid(u)`` rounds to exactly ``b`` once ``u`` exceeds
about 37, where the density is evaluated at the bound and treated as zero.
For a density with an integrable singularity at the upper bound (for
example ``Beta(1, 0.05)``) this silently drops mass (16% for that law in a
review); the lower bound at ``a = 0`` keeps full resolution. Reparameterize
such a site (for example ``1 - x``) so the singular end sits at zero.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from probstat.distributions import Normal, RealSupport
from probstat.foundations.random_streams import RandomStream
from probstat.linalg import Vector, as_vector
from probstat.optimization import central_gradient

SiteKind = Literal["latent", "observed", "deterministic"]
SiteValue = float | tuple[float, ...]
Model = Callable[["ModelHandle"], object]

_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_NUMERIC_ERRORS = (ValueError, ArithmeticError)


class ModelSpecificationError(ValueError):
    """A model violates the rules of the probabilistic-programming interface.

    Raised for duplicate or unknown site names, unsupported supports or
    shapes, missing observed data, and models whose site structure changes
    between executions. It is never converted into a ``-inf`` log-density.

    References:
        [REF-CARPENTER-ETAL-2017], [REF-BINGHAM-ETAL-2019].

    """


# ---------------------------------------------------------------------------
# Bijections to unconstrained space
# ---------------------------------------------------------------------------


def _softplus(t: float) -> float:
    """Return ``log(1 + exp(t))`` without overflow.

    References:
        [EQ-PPL-TRANSFORM-002]. No new equation is introduced.

    """
    return max(t, 0.0) + math.log1p(math.exp(-abs(t)))


def _sigmoid(t: float) -> float:
    """Return ``1 / (1 + exp(-t))`` without overflow.

    References:
        [EQ-PPL-TRANSFORM-002]. No new equation is introduced.

    """
    if t >= 0.0:
        return 1.0 / (1.0 + math.exp(-t))
    e = math.exp(t)
    return e / (1.0 + e)


@dataclass(frozen=True, slots=True)
class Bijection:
    """Smooth bijection from ``R`` onto the interior of an interval support.

    ``kind`` is ``"identity"`` (``x = u`` on ``R``), ``"lower"``
    (``x = a + exp(u)`` on ``(a, inf)``), ``"upper"`` (``x = b - exp(u)`` on
    ``(-inf, b)``), or ``"interval"`` (``x = a + (b - a) sigmoid(u)`` on
    ``(a, b)``). Closed endpoints are not reached; they have probability zero
    under a continuous law.

    Args:
        kind: Bijection family.
        lower: Finite lower bound ``a`` (``"lower"``, ``"interval"``).
        upper: Finite upper bound ``b`` (``"upper"``, ``"interval"``).

    Raises:
        ValueError: If the bounds do not match ``kind``.

    References:
        [REF-CARPENTER-ETAL-2017], [REF-KUCUKELBIR-ETAL-2017].

    """

    kind: Literal["identity", "lower", "upper", "interval"]
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        """Check that the bounds agree with the bijection family.

        References:
            Project validation policy. No equation is evaluated.

        """
        needs_lower = self.kind in ("lower", "interval")
        needs_upper = self.kind in ("upper", "interval")
        if self.kind not in ("identity", "lower", "upper", "interval"):
            raise ValueError(f"unknown bijection kind {self.kind!r}")
        if needs_lower != (self.lower is not None):
            raise ValueError(f"bijection {self.kind!r} has a wrong lower bound")
        if needs_upper != (self.upper is not None):
            raise ValueError(f"bijection {self.kind!r} has a wrong upper bound")
        if self.kind == "interval":
            assert self.lower is not None and self.upper is not None
            if not self.lower < self.upper:
                raise ValueError("interval bijection needs lower < upper")

    def forward(self, u: float) -> float:
        """Map an unconstrained value to the support.

        Args:
            u: Unconstrained real value.

        Returns:
            Constrained value ``x = T(u)``.

        Raises:
            OverflowError: If ``exp(u)`` overflows (``u > ~709``).

        Equation:
            [EQ-PPL-TRANSFORM-001], [EQ-PPL-TRANSFORM-002].

        References:
            [REF-CARPENTER-ETAL-2017].

        """
        if self.kind == "identity":
            return u
        if self.kind == "lower":
            assert self.lower is not None
            return self.lower + math.exp(u)
        if self.kind == "upper":
            assert self.upper is not None
            return self.upper - math.exp(u)
        assert self.lower is not None and self.upper is not None
        return self.lower + (self.upper - self.lower) * _sigmoid(u)

    def log_abs_det_jacobian(self, u: float) -> float:
        """Return ``log |dT/du|`` at ``u``.

        Args:
            u: Unconstrained real value.

        Returns:
            ``0`` (identity), ``u`` (lower or upper bound), or
            ``log(b - a) - softplus(u) - softplus(-u)`` (interval).

        Equation:
            [EQ-PPL-TRANSFORM-001], [EQ-PPL-TRANSFORM-002].

        References:
            [REF-CARPENTER-ETAL-2017].

        """
        if self.kind == "identity":
            return 0.0
        if self.kind in ("lower", "upper"):
            return u
        assert self.lower is not None and self.upper is not None
        return math.log(self.upper - self.lower) - _softplus(u) - _softplus(-u)

    def inverse(self, x: float) -> float:
        """Map a value in the interior of the support to ``R``.

        Args:
            x: Constrained value strictly inside the support.

        Returns:
            ``u = T^-1(x)``.

        Raises:
            ValueError: If ``x`` is not strictly inside the support.

        Equation:
            [EQ-PPL-TRANSFORM-001], [EQ-PPL-TRANSFORM-002].

        References:
            [REF-CARPENTER-ETAL-2017].

        """
        if self.kind == "identity":
            return float(x)
        if self.kind == "lower":
            assert self.lower is not None
            if not x > self.lower:
                raise ValueError("value must exceed the lower bound")
            return math.log(x - self.lower)
        if self.kind == "upper":
            assert self.upper is not None
            if not x < self.upper:
                raise ValueError("value must be below the upper bound")
            return math.log(self.upper - x)
        assert self.lower is not None and self.upper is not None
        if not self.lower < x < self.upper:
            raise ValueError("value must lie strictly inside the interval")
        return math.log(x - self.lower) - math.log(self.upper - x)

    def label(self, name: str) -> str:
        """Return the name of the unconstrained coordinate of site ``name``.

        References:
            Project API policy. No equation is evaluated.

        """
        if self.kind == "identity":
            return name
        if self.kind == "lower":
            assert self.lower is not None
            if self.lower == 0.0:
                return f"log({name})"
            return f"log({name} - {self.lower:g})"
        if self.kind == "upper":
            assert self.upper is not None
            return f"log({self.upper:g} - {name})"
        assert self.lower is not None and self.upper is not None
        if self.lower == 0.0 and self.upper == 1.0:
            return f"logit({name})"
        return f"log(({name} - {self.lower:g}) / ({self.upper:g} - {name}))"


_IDENTITY = Bijection("identity")
_BIJECTIONS: dict[RealSupport, Bijection] = {}
_CACHE_LIMIT = 4096


def bijection_for(support: object, *, site: str = "site") -> Bijection:
    """Choose the bijection to unconstrained space for an interval support.

    Args:
        support: A :class:`~probstat.distributions.RealSupport`.
        site: Site name used in error messages.

    Returns:
        Identity for ``R``, shifted log for ``(a, inf)``, negated shifted log
        for ``(-inf, b)``, and scaled logit for ``(a, b)``.

    Raises:
        ModelSpecificationError: If the support is not a continuous
            ``RealSupport`` (for example an integer support of a discrete
            law) or is a single point.

    Equation:
        [EQ-PPL-TRANSFORM-001], [EQ-PPL-TRANSFORM-002].

    References:
        [REF-CARPENTER-ETAL-2017], [REF-KUCUKELBIR-ETAL-2017].

    """
    if isinstance(support, RealSupport):
        cached = _BIJECTIONS.get(support)
        if cached is not None:
            return cached
    if not isinstance(support, RealSupport):
        kind = type(support).__name__
        raise ModelSpecificationError(
            f"latent site {site!r}: unsupported support {kind}; latent sites need "
            "a continuous RealSupport (discrete latent variables must be "
            "marginalized out by hand)"
        )
    lower, upper = support.lower, support.upper
    if lower is not None and upper is not None:
        if not lower < upper:
            raise ModelSpecificationError(
                f"latent site {site!r}: degenerate support [{lower}, {upper}]"
            )
        bijection = Bijection("interval", lower, upper)
    elif lower is not None:
        bijection = Bijection("lower", lower, None)
    elif upper is not None:
        bijection = Bijection("upper", None, upper)
    else:
        bijection = _IDENTITY
    if len(_BIJECTIONS) < _CACHE_LIMIT:
        _BIJECTIONS[support] = bijection
    return bijection


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------


def _support_of(distribution: object, site: str) -> object:
    """Return ``distribution.support`` or raise a clear error.

    References:
        Project API policy. No equation is evaluated.

    """
    try:
        return distribution.support  # type: ignore[attr-defined]
    except AttributeError:
        raise ModelSpecificationError(
            f"site {site!r}: distribution {type(distribution).__name__} has no "
            "'support' attribute, so no bijection can be chosen"
        ) from None


def _site_bijection(distribution: object, site: str) -> Bijection:
    """Return the bijection for one latent element's distribution.

    References:
        [REF-CARPENTER-ETAL-2017]. Equation [EQ-PPL-TRANSFORM-001].

    """
    if type(distribution) is Normal:
        return _IDENTITY
    return bijection_for(_support_of(distribution, site), site=site)


def _log_density(distribution: Any, value: Any) -> float:
    """Return ``log_pdf`` (continuous) or ``log_pmf`` (discrete) at ``value``.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 1]. Equation [EQ-DIST-NORM-001] for the
        inlined normal case; no new equation is introduced.

    """
    if type(distribution) is Normal:
        # Same formula as Normal.log_pdf ([EQ-DIST-NORM-001]) without the
        # argument validation, which dominates the cost of small models.
        sigma: float = distribution.sigma
        z: float = (value - distribution.mu) / sigma
        return -0.5 * z * z - math.log(sigma) - _LOG_SQRT_TWO_PI
    method = getattr(distribution, "log_pdf", None)
    if method is None:
        method = getattr(distribution, "log_pmf", None)
    if method is None:
        raise ModelSpecificationError(
            f"distribution {type(distribution).__name__} has neither log_pdf "
            "nor log_pmf"
        )
    return float(method(value))


def _draw(distribution: Any, generator: random.Random) -> Any:
    """Draw one value with a seed taken from ``generator``.

    References:
        [REF-DEVROYE-1986, Chapter I]. No new equation is introduced.

    """
    sampler = getattr(distribution, "sample", None)
    if sampler is None:
        raise ModelSpecificationError(
            f"distribution {type(distribution).__name__} has no sample method"
        )
    values = sampler(1, seed=generator.getrandbits(63))
    return values[0]


def _elements(
    distribution: object, size: int | None, site: str
) -> tuple[int | None, tuple[Any, ...]]:
    """Resolve a site's size and its per-element distributions.

    A single distribution is shared by every element; a list or tuple gives
    one distribution per element.

    References:
        Project API policy. No equation is evaluated.

    """
    if isinstance(distribution, list | tuple):
        parts = tuple(distribution)
        if not parts:
            raise ModelSpecificationError(f"site {site!r}: empty distribution list")
        if size is not None and size != len(parts):
            raise ModelSpecificationError(
                f"site {site!r}: shape {size} does not match {len(parts)} "
                "distributions"
            )
        return len(parts), parts
    if size is None:
        return None, (distribution,)
    return size, (distribution,) * size


def _checked_shape(shape: object, site: str) -> int | None:
    """Validate a ``shape`` argument (``None`` or a positive integer).

    References:
        Project validation policy. No equation is evaluated.

    """
    if shape is None:
        return None
    if isinstance(shape, tuple) and len(shape) == 1:
        shape = shape[0]
    if isinstance(shape, bool) or not isinstance(shape, int) or shape < 1:
        raise ModelSpecificationError(
            f"site {site!r}: shape must be None or a positive integer"
        )
    return shape


def _pack(values: Sequence[Any], size: int | None) -> Any:
    """Return a scalar for scalar sites and a tuple for vector sites.

    References:
        Project API policy. No equation is evaluated.

    """
    return values[0] if size is None else tuple(values)


def elementwise(factory: Callable[..., object], *arguments: object) -> tuple[Any, ...]:
    """Build one distribution per element by broadcasting arguments.

    Scalars are repeated; sequences must share one length. For example
    ``elementwise(Normal, theta, s)`` returns
    ``(Normal(theta[0], s[0]), Normal(theta[1], s[1]), ...)``, which can be
    passed to :meth:`ModelHandle.sample` or :meth:`ModelHandle.observe`.

    Args:
        factory: Distribution constructor.
        *arguments: Scalars or equal-length sequences.

    Returns:
        Tuple of distributions.

    Raises:
        ValueError: If sequence arguments differ in length or none is given.

    References:
        [REF-BINGHAM-ETAL-2019]. No equation is evaluated.

    """
    columns: list[Sequence[object] | None] = [
        a
        if isinstance(a, list | tuple)
        or (isinstance(a, Sequence) and not isinstance(a, str))
        else None
        for a in arguments
    ]
    lengths = {len(c) for c in columns if c is not None}
    if len(lengths) > 1:
        raise ValueError("sequence arguments must share one length")
    if not lengths:
        raise ValueError("elementwise needs at least one sequence argument")
    count = lengths.pop()
    expanded = [
        column if column is not None else (argument,) * count
        for column, argument in zip(columns, arguments, strict=True)
    ]
    return tuple(factory(*row) for row in zip(*expanded, strict=True))


# ---------------------------------------------------------------------------
# Site records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SiteInfo:
    """One named site of a traced model.

    Args:
        name: Unique site name.
        kind: ``"latent"``, ``"observed"``, or ``"deterministic"``.
        size: ``None`` for a scalar site, else the vector length.
        distribution: Class name of the site's distribution (``""`` for a
            deterministic site); ``"Normal x 8"`` style for per-element lists.
        support: Support at trace time (``None`` for deterministic sites).
        bijection: Bijection at trace time (latent sites only).
        reparam: ``"noncentered"`` or ``None``.
        value: Value at trace time; for observed sites this is the data.

    References:
        [REF-BINGHAM-ETAL-2019], [REF-CARPENTER-ETAL-2017].

    """

    name: str
    kind: SiteKind
    size: int | None
    distribution: str
    support: object
    bijection: Bijection | None
    reparam: str | None
    value: Any
    element_bijections: tuple[Bijection, ...] = ()

    @property
    def length(self) -> int:
        """Return the number of scalar elements (``1`` for a scalar site).

        References:
            Project API policy. No equation is evaluated.

        """
        return 1 if self.size is None else self.size

    def flat_names(self, base: str | None = None) -> tuple[str, ...]:
        """Return element names ``name`` or ``name[0], name[1], ...``.

        References:
            Project API policy. No equation is evaluated.

        """
        label = self.name if base is None else base
        if self.size is None:
            return (label,)
        return tuple(f"{label}[{i}]" for i in range(self.size))


def missing_data_message(site: str) -> str:
    """Return the error message for an observed site without data.

    References:
        Project API policy. No equation is evaluated.

    """
    return (
        f"observed site {site!r} has no data; pass a value or use "
        f"condition(model, {{{site!r}: data}})"
    )


def _check_data(site: str, values: Sequence[object]) -> None:
    """Reject observed data that are not finite real numbers.

    References:
        Project validation policy. No equation is evaluated.

    """
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ModelSpecificationError(
                f"observed site {site!r}: data must be real numbers"
            )
        if not math.isfinite(value):
            raise ModelSpecificationError(
                f"observed site {site!r}: data must be finite"
            )


def _describe(parts: tuple[Any, ...], size: int | None) -> str:
    """Return a short description of a site's distribution(s).

    References:
        Project API policy. No equation is evaluated.

    """
    names = {type(p).__name__ for p in parts}
    label = names.pop() if len(names) == 1 else "mixed"
    return label if size is None else f"{label} x {size}"


# ---------------------------------------------------------------------------
# The handle passed to model functions
# ---------------------------------------------------------------------------


class ModelHandle:
    """The object a model function receives as its argument ``m``.

    Users call :meth:`sample`, :meth:`observe`, and :meth:`deterministic`;
    the inference functions construct the handle in the right mode.

    Args:
        mode: ``"trace"``, ``"density"``, or ``"generate"``.
        unconstrained: Unconstrained vector (``density`` mode).
        layout: Expected site sequence; when given, each site is checked
            against it so that structural changes are reported.
        fixed: Constrained latent values by name (``generate`` mode).
        require_fixed: Whether every latent site must appear in ``fixed``.
        generator: Random generator (``generate`` mode).
        conditioned: Observed values by site name (see :func:`condition`).

    References:
        [REF-BINGHAM-ETAL-2019], [REF-CARPENTER-ETAL-2017].

    """

    __slots__ = (
        "_conditioned",
        "_cursor",
        "_fixed",
        "_generator",
        "_layout",
        "_mode",
        "_names",
        "_offset",
        "_require_fixed",
        "_unconstrained",
        "log_jacobian",
        "log_likelihood",
        "log_prior",
        "records",
        "values",
    )

    def __init__(
        self,
        mode: Literal["trace", "density", "generate"],
        *,
        unconstrained: Vector = (),
        layout: tuple[SiteInfo, ...] | None = None,
        fixed: Mapping[str, SiteValue] | None = None,
        require_fixed: bool = False,
        generator: random.Random | None = None,
        conditioned: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize an empty execution record.

        References:
            Project API policy. No equation is evaluated.

        """
        self._mode = mode
        self._unconstrained = unconstrained
        self._layout = layout
        self._fixed = dict(fixed or {})
        self._require_fixed = require_fixed
        self._generator = generator
        self._conditioned: dict[str, object] = dict(conditioned or {})
        self._cursor = 0
        self._offset = 0
        self._names: set[str] = set()
        self.records: list[SiteInfo] = []
        self.values: dict[str, Any] = {}
        self.log_prior = 0.0
        self.log_likelihood = 0.0
        self.log_jacobian = 0.0

    # -- bookkeeping -------------------------------------------------------

    def add_conditions(self, data: Mapping[str, object]) -> None:
        """Register observed values by name; outer registrations win.

        Args:
            data: ``{site name: observed value}``.

        References:
            [REF-BINGHAM-ETAL-2019]. No equation is evaluated.

        """
        for name, value in data.items():
            self._conditioned.setdefault(name, value)

    def _register(self, name: object) -> str:
        """Reject non-string and duplicate site names.

        References:
            Project API policy. No equation is evaluated.

        """
        if not isinstance(name, str) or not name:
            raise ModelSpecificationError("site names must be non-empty strings")
        if name in self._names:
            raise ModelSpecificationError(f"duplicate site name {name!r}")
        self._names.add(name)
        return name

    def _check_layout(self, name: str, kind: SiteKind, size: int | None) -> None:
        """Compare a site with the traced structure.

        References:
            Project API policy. No equation is evaluated.

        """
        if self._layout is None:
            return
        index = self._cursor
        if index >= len(self._layout):
            raise ModelSpecificationError(
                f"site {name!r} was not present when the model was traced; the "
                "site structure must not depend on parameter values"
            )
        expected = self._layout[index]
        if (expected.name, expected.kind, expected.size) != (name, kind, size):
            raise ModelSpecificationError(
                f"model structure changed: expected {expected.kind} site "
                f"{expected.name!r} (size {expected.size}) at position {index}, "
                f"got {kind} site {name!r} (size {size})"
            )
        self._cursor += 1

    def finish(self) -> None:
        """Check that every traced site was visited.

        References:
            Project API policy. No equation is evaluated.

        """
        if self._layout is not None and self._cursor != len(self._layout):
            missing = self._layout[self._cursor].name
            raise ModelSpecificationError(
                f"model structure changed: site {missing!r} was not visited"
            )

    @property
    def conditioned_names(self) -> frozenset[str]:
        """Return the site names registered through :func:`condition`.

        References:
            [REF-BINGHAM-ETAL-2019]. No equation is evaluated.

        """
        return frozenset(self._conditioned)

    # -- public site declarations -----------------------------------------

    def sample(
        self,
        name: str,
        distribution: object,
        *,
        shape: int | None = None,
        reparam: str | None = None,
    ) -> Any:
        """Declare a latent random variable (or an observed one if conditioned).

        Args:
            name: Unique site name.
            distribution: A distribution with ``support``, ``log_pdf`` and
                ``sample``; or a list/tuple of them (one per element).
            shape: ``None`` for a scalar, ``n`` for an i.i.d. vector of length
                ``n`` (inferred from a distribution list).
            reparam: ``"noncentered"`` samples ``z ~ N(0, 1)`` and returns
                ``loc + scale z``; only for ``Normal`` sites.

        Returns:
            A float (scalar site) or a tuple of floats (vector site).

        Raises:
            ModelSpecificationError: For duplicate names, bad shapes, an
                unsupported support, or an invalid ``reparam``.

        Equation:
            [EQ-PPL-JOINT-001], [EQ-PPL-NONCENTERED-001].

        References:
            [REF-BINGHAM-ETAL-2019], [REF-CARPENTER-ETAL-2017],
            [REF-PAPASPILIOPOULOS-ROBERTS-SKOLD-2007].

        """
        site = self._register(name)
        if site in self._conditioned:
            self._names.discard(site)
            data = self._conditioned[site]
            return self.observe(site, distribution, data, shape=shape)
        size, parts = _elements(distribution, _checked_shape(shape, site), site)
        if reparam is not None:
            if reparam != "noncentered":
                raise ModelSpecificationError(
                    f"site {site!r}: unknown reparam {reparam!r} (use 'noncentered')"
                )
            if not all(isinstance(p, Normal) for p in parts):
                raise ModelSpecificationError(
                    f"site {site!r}: reparam='noncentered' needs Normal distributions"
                )
        self._check_layout(site, "latent", size)
        if reparam is not None:
            bijections = (_IDENTITY,) * len(parts)
        elif isinstance(distribution, list | tuple):
            bijections = tuple(_site_bijection(p, site) for p in parts)
        else:
            bijections = (_site_bijection(parts[0], site),) * len(parts)
        values = self._latent_values(site, size, parts, bijections, reparam)
        packed = _pack(values, size)
        self.values[site] = packed
        if self._mode == "density":
            return packed
        self.records.append(
            SiteInfo(
                site,
                "latent",
                size,
                _describe(parts, size),
                _support_of(parts[0], site),
                bijections[0],
                reparam,
                packed,
                tuple(bijections),
            )
        )
        return packed

    def _latent_values(
        self,
        site: str,
        size: int | None,
        parts: tuple[Any, ...],
        bijections: tuple[Bijection, ...],
        reparam: str | None,
    ) -> list[float]:
        """Produce latent values and accumulate density terms for the mode.

        References:
            [REF-CARPENTER-ETAL-2017], [REF-PAPASPILIOPOULOS-ROBERTS-SKOLD-2007].
            Equations [EQ-PPL-JOINT-001], [EQ-PPL-NONCENTERED-001].

        """
        count = len(parts)
        if self._mode == "generate":
            if site in self._fixed:
                fixed = self._fixed[site]
                if size is None:
                    if isinstance(fixed, tuple):
                        raise ModelSpecificationError(
                            f"site {site!r}: fixed value must be a scalar"
                        )
                    return [float(fixed)]
                if isinstance(fixed, list):
                    fixed = tuple(fixed)
                if not isinstance(fixed, tuple) or len(fixed) != size:
                    raise ModelSpecificationError(
                        f"site {site!r}: fixed value must be a sequence of "
                        f"length {size}"
                    )
                return [float(v) for v in fixed]
            if self._require_fixed:
                raise ModelSpecificationError(
                    f"latent site {site!r} has no posterior value"
                )
            assert self._generator is not None
            return [float(_draw(p, self._generator)) for p in parts]
        if self._mode == "trace":
            u_values: Sequence[float] = (0.0,) * count
        else:
            start = self._offset
            stop = start + count
            if stop > len(self._unconstrained):
                raise ModelSpecificationError(
                    "unconstrained vector is shorter than the model dimension"
                )
            u_values = self._unconstrained[start:stop]
            self._offset = stop
        density = self._mode == "density"
        if reparam is not None:
            values = [p.mu + p.sigma * u for p, u in zip(parts, u_values, strict=True)]
            if density:
                self.log_prior += math.fsum(-0.5 * u * u for u in u_values) - (
                    count * _LOG_SQRT_TWO_PI
                )
            return values
        values = []
        prior = 0.0
        jacobian = 0.0
        for part, bijection, u in zip(parts, bijections, u_values, strict=True):
            value = bijection.forward(u)
            if density:
                jacobian += bijection.log_abs_det_jacobian(u)
                prior += _log_density(part, value)
            values.append(value)
        self.log_prior += prior
        self.log_jacobian += jacobian
        return values

    def observe(
        self,
        name: str,
        distribution: object,
        value: object = None,
        *,
        shape: int | None = None,
    ) -> Any:
        """Declare an observed random variable and its data.

        Args:
            name: Unique site name.
            distribution: A distribution (shared by all elements) or a list
                of distributions (one per element).
            value: Observed scalar or sequence; may be ``None`` when the data
                are supplied later by :func:`condition` or when the model is
                only simulated.
            shape: Vector length when ``value`` is ``None`` and a single
                distribution is given.

        Returns:
            The observed data, or a simulated value in ``generate`` mode.

        Raises:
            ModelSpecificationError: For duplicate names, shape mismatches,
                or missing data during inference.

        Equation:
            [EQ-PPL-JOINT-001], [EQ-PPL-PREDICTIVE-001].

        References:
            [REF-BINGHAM-ETAL-2019], [REF-GELMAN-ETAL-2013, Chapter 6].

        """
        site = self._register(name)
        if site in self._conditioned:
            value = self._conditioned[site]
        checked_shape = _checked_shape(shape, site)
        data: Any = value
        if value is not None:
            if isinstance(value, str) or not (
                isinstance(value, Sequence) or isinstance(value, int | float)
            ):
                raise ModelSpecificationError(
                    f"observed site {site!r}: data must be a number or a sequence"
                )
            if isinstance(value, Sequence):
                data = tuple(value)
                if not data:
                    raise ModelSpecificationError(
                        f"observed site {site!r}: data must be non-empty"
                    )
                if checked_shape is not None and checked_shape != len(data):
                    raise ModelSpecificationError(
                        f"observed site {site!r}: shape does not match the data"
                    )
                checked_shape = len(data)
            elif checked_shape is not None:
                raise ModelSpecificationError(
                    f"observed site {site!r}: scalar data with a vector shape"
                )
        size, parts = _elements(distribution, checked_shape, site)
        if value is not None and (size is None) != (not isinstance(data, tuple)):
            raise ModelSpecificationError(
                f"observed site {site!r}: {len(parts)} distributions do not match "
                "the data"
            )
        self._check_layout(site, "observed", size)
        if self._mode == "trace" and value is not None:
            _check_data(site, (data,) if size is None else data)
        if self._mode == "generate":
            assert self._generator is not None
            drawn = [_draw(p, self._generator) for p in parts]
            result = _pack(drawn, size)
        elif value is None:
            if self._mode == "density":
                raise ModelSpecificationError(missing_data_message(site))
            result = None
        else:
            elements = (data,) if size is None else data
            if self._mode == "density":
                total = 0.0
                for part, element in zip(parts, elements, strict=True):
                    total += _log_density(part, element)
                self.log_likelihood += total
                return data
            result = data
        self.records.append(
            SiteInfo(
                site,
                "observed",
                size,
                _describe(parts, size),
                getattr(parts[0], "support", None),
                None,
                None,
                result,
            )
        )
        return result

    def deterministic(self, name: str, value: object) -> Any:
        """Record a deterministic function of latent values.

        Args:
            name: Unique site name.
            value: A real number or a sequence of real numbers.

        Returns:
            ``value`` as a float or a tuple of floats.

        Raises:
            ModelSpecificationError: For duplicate names or non-numeric values.

        References:
            [REF-BINGHAM-ETAL-2019]. No equation is evaluated.

        """
        site = self._register(name)
        size: int | None
        if isinstance(value, Sequence) and not isinstance(value, str):
            try:
                elements = tuple(float(v) for v in value)
            except (TypeError, ValueError):
                raise ModelSpecificationError(
                    f"deterministic site {site!r} must hold real numbers"
                ) from None
            packed: SiteValue = elements
            size = len(elements)
            if size == 0:
                raise ModelSpecificationError(
                    f"deterministic site {site!r} must be non-empty"
                )
        else:
            try:
                packed = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                raise ModelSpecificationError(
                    f"deterministic site {site!r} must hold real numbers"
                ) from None
            size = None
        self._check_layout(site, "deterministic", size)
        self.values[site] = packed
        if self._mode == "density":
            return packed
        self.records.append(
            SiteInfo(site, "deterministic", size, "", None, None, None, packed)
        )
        return packed


def condition(model: Model, data: Mapping[str, object]) -> Model:
    """Return a model in which the named sites are observed with ``data``.

    A ``sample`` site whose name appears in ``data`` becomes an observed
    site; an ``observe`` site takes its data from ``data`` (replacing the
    value written in the model). When conditions are nested, the outermost
    one wins.

    Args:
        model: Model function.
        data: ``{site name: observed value}``.

    Returns:
        Conditioned model function.

    Raises:
        TypeError: If ``model`` is not callable.

    References:
        [REF-BINGHAM-ETAL-2019]. No equation is evaluated.

    """
    if not callable(model):
        raise TypeError("model must be callable")
    values = dict(data)

    def conditioned(m: ModelHandle) -> object:
        """Register the conditions and run the wrapped model.

        References:
            [REF-BINGHAM-ETAL-2019]. No equation is evaluated.

        """
        m.add_conditions(values)
        return model(m)

    conditioned.__name__ = getattr(model, "__name__", "model")
    return conditioned


# ---------------------------------------------------------------------------
# Intermediate representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LogDensityTerms:
    """Decomposition of the joint log-density at one unconstrained point.

    Args:
        log_prior: ``sum log p(x_k | parents)`` over latent elements (the
            ``N(0, 1)`` density of ``z`` for non-centred sites).
        log_likelihood: ``sum log p(y | ...)`` over observed elements.
        log_jacobian: ``sum log |dT_k/du_k|``.
        values: Constrained latent and deterministic values by name.

    References:
        [REF-CARPENTER-ETAL-2017]. Equation [EQ-PPL-JOINT-001].

    """

    log_prior: float
    log_likelihood: float
    log_jacobian: float
    values: dict[str, SiteValue]

    @property
    def total(self) -> float:
        """Return the joint log-density on the unconstrained scale.

        References:
            [EQ-PPL-JOINT-001], [REF-CARPENTER-ETAL-2017].

        """
        return self.log_prior + self.log_likelihood + self.log_jacobian


@dataclass(frozen=True, slots=True)
class ModelIR:
    """Traced intermediate representation of a model.

    Obtained by :func:`trace`. Holds the ordered site list and turns the
    model into the inputs every sampler in :mod:`probstat.inference` needs: a
    log-density on ``R^d`` (prior + likelihood + log-Jacobian), its gradient
    by central differences, and maps from unconstrained draws to named
    constrained values.

    Args:
        model: The model function.
        sites: Every site in execution order.

    References:
        [REF-CARPENTER-ETAL-2017], [REF-BINGHAM-ETAL-2019].

    """

    model: Model
    sites: tuple[SiteInfo, ...]

    # -- structure ---------------------------------------------------------

    def _of_kind(self, kind: SiteKind) -> tuple[SiteInfo, ...]:
        """Return the sites of one kind in execution order.

        References:
            Project API policy. No equation is evaluated.

        """
        return tuple(site for site in self.sites if site.kind == kind)

    @property
    def latent_sites(self) -> tuple[SiteInfo, ...]:
        """Return latent sites in execution order.

        References:
            Project API policy. No equation is evaluated.

        """
        return self._of_kind("latent")

    @property
    def observed_sites(self) -> tuple[SiteInfo, ...]:
        """Return observed sites in execution order.

        References:
            Project API policy. No equation is evaluated.

        """
        return self._of_kind("observed")

    @property
    def deterministic_sites(self) -> tuple[SiteInfo, ...]:
        """Return deterministic sites in execution order.

        References:
            Project API policy. No equation is evaluated.

        """
        return self._of_kind("deterministic")

    def site(self, name: str) -> SiteInfo:
        """Return the site called ``name``.

        Raises:
            KeyError: If no site has that name.

        References:
            Project API policy. No equation is evaluated.

        """
        for site in self.sites:
            if site.name == name:
                return site
        raise KeyError(f"unknown site {name!r}")

    @property
    def dimension(self) -> int:
        """Return the dimension ``d`` of the unconstrained space.

        References:
            [REF-CARPENTER-ETAL-2017]. No equation is evaluated.

        """
        return sum(site.length for site in self.latent_sites)

    @property
    def unconstrained_names(self) -> tuple[str, ...]:
        """Return coordinate names such as ``mu``, ``log(tau)``, ``theta_z[0]``.

        References:
            Project API policy. No equation is evaluated.

        """
        names: list[str] = []
        for site in self.latent_sites:
            assert site.bijection is not None
            if site.reparam == "noncentered":
                names.extend(site.flat_names(f"{site.name}_z"))
            elif site.size is not None and len(site.element_bijections) == site.size:
                # Mixed supports in one list site: label every element by its
                # own transform.
                names.extend(
                    bijection.label(f"{site.name}[{index}]")
                    for index, bijection in enumerate(site.element_bijections)
                )
            else:
                names.extend(site.flat_names(site.bijection.label(site.name)))
        return tuple(names)

    @property
    def value_sites(self) -> tuple[SiteInfo, ...]:
        """Return latent and deterministic sites (the reported quantities).

        References:
            Project API policy. No equation is evaluated.

        """
        return tuple(s for s in self.sites if s.kind in ("latent", "deterministic"))

    @property
    def constrained_names(self) -> tuple[str, ...]:
        """Return flat names of latent and deterministic elements.

        References:
            Project API policy. No equation is evaluated.

        """
        return tuple(name for s in self.value_sites for name in s.flat_names())

    def observed_data(self, name: str) -> SiteValue:
        """Return the data of an observed site as floats.

        Raises:
            KeyError: If ``name`` is not an observed site.

        References:
            Project API policy. No equation is evaluated.

        """
        site = self.site(name)
        if site.kind != "observed":
            raise KeyError(f"{name!r} is not an observed site")
        if site.value is None:
            raise ModelSpecificationError(missing_data_message(name))
        if site.size is None:
            return float(site.value)
        return tuple(float(v) for v in site.value)

    # -- evaluation --------------------------------------------------------

    def _run_density(self, u: Sequence[float]) -> ModelHandle:
        """Execute the model in density mode and check its structure.

        References:
            [EQ-PPL-JOINT-001]. No new equation is introduced.

        """
        point = tuple(u)
        if len(point) != self.dimension:
            raise ModelSpecificationError(
                f"unconstrained vector has length {len(point)}, expected "
                f"{self.dimension}"
            )
        handle = ModelHandle("density", unconstrained=point, layout=self.sites)
        self.model(handle)
        handle.finish()
        return handle

    def terms(self, u: Sequence[float]) -> LogDensityTerms:
        """Return the prior, likelihood, and Jacobian terms at ``u``.

        Numerical failures inside the model propagate here (unlike
        :meth:`log_density`), which makes this the method to debug with.

        Args:
            u: Unconstrained vector of length :attr:`dimension`.

        Returns:
            Density terms and the constrained values.

        Equation:
            [EQ-PPL-JOINT-001].

        References:
            [REF-CARPENTER-ETAL-2017].

        """
        handle = self._run_density(u)
        return LogDensityTerms(
            handle.log_prior,
            handle.log_likelihood,
            handle.log_jacobian,
            dict(handle.values),
        )

    def log_density(self, u: Sequence[float]) -> float:
        """Return the joint log-density on the unconstrained scale.

        ``log p(u) = sum_k log p(T_k(u_k) | parents) + log p(y | ...)
        + sum_k log |dT_k/du_k|``. Errors raised by distribution constructors
        or ``log_pdf`` (an invalid parameter value, an overflow) and
        non-finite results other than ``-inf`` are mapped to ``-inf``, which
        every sampler treats as "outside the support". Specification errors
        still propagate.

        Args:
            u: Unconstrained vector.

        Returns:
            Finite log-density or ``-inf``.

        Equation:
            [EQ-PPL-JOINT-001].

        References:
            [REF-CARPENTER-ETAL-2017], [REF-KUCUKELBIR-ETAL-2017].

        """
        try:
            handle = self._run_density(u)
        except ModelSpecificationError:
            raise
        except _NUMERIC_ERRORS:
            return -math.inf
        total = handle.log_prior + handle.log_likelihood + handle.log_jacobian
        if math.isnan(total) or total == math.inf:
            return -math.inf
        return total

    def gradient(self, u: Sequence[float], *, relative_step: float = 1e-6) -> Vector:
        """Return the central finite-difference gradient of :meth:`log_density`.

        Each component costs two model executions with step
        ``h_i = relative_step * max(1, |u_i|)``; the truncation error is
        ``O(h^2)``. The samplers in :mod:`probstat.inference` compute the same
        approximation internally when no gradient is supplied.

        Args:
            u: Unconstrained vector with finite log-density nearby.
            relative_step: Relative finite-difference step.

        Returns:
            Approximate gradient.

        Raises:
            ValueError: If a perturbed point has a non-finite log-density.

        Equation:
            [EQ-OPT-FD-001].

        References:
            [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """
        point = as_vector(u, name="u")
        value, _ = central_gradient(
            self.log_density, point, relative_step=relative_step
        )
        return value

    def constrain(self, u: Sequence[float]) -> dict[str, SiteValue]:
        """Map an unconstrained vector to named latent and deterministic values.

        Args:
            u: Unconstrained vector.

        Returns:
            ``{site name: float or tuple}`` in execution order.

        References:
            [REF-CARPENTER-ETAL-2017]. Equation [EQ-PPL-TRANSFORM-001].

        """
        return dict(self._run_density(u).values)

    def flat_constrained(self, u: Sequence[float]) -> Vector:
        """Return constrained values flattened in :attr:`constrained_names` order.

        References:
            Project API policy. No equation is evaluated.

        """
        return flatten(self.constrain(u), self.value_sites)

    def unflatten(self, flat: Sequence[float]) -> dict[str, SiteValue]:
        """Invert :meth:`flat_constrained` into named values.

        Args:
            flat: Vector ordered as :attr:`constrained_names`.

        Returns:
            ``{site name: float or tuple}``.

        Raises:
            ValueError: If the length does not match.

        References:
            Project API policy. No equation is evaluated.

        """
        values = tuple(flat)
        if len(values) != len(self.constrained_names):
            raise ValueError("flat vector does not match the constrained layout")
        result: dict[str, SiteValue] = {}
        offset = 0
        for site in self.value_sites:
            if site.size is None:
                result[site.name] = float(values[offset])
            else:
                result[site.name] = tuple(
                    float(v) for v in values[offset : offset + site.size]
                )
            offset += site.length
        return result

    def initial_point(self, stream: RandomStream, *, radius: float = 2.0) -> Vector:
        """Draw a starting point with finite log-density.

        Coordinates are drawn uniformly from ``[-radius, radius]`` on the
        unconstrained scale (the convention of Stan); ``radius = 0`` returns
        the origin. Up to 100 draws are tried.

        Args:
            stream: Random stream for the draws.
            radius: Non-negative half-width.

        Returns:
            Unconstrained starting vector.

        Raises:
            ValueError: If no draw has a finite log-density.

        References:
            [REF-CARPENTER-ETAL-2017]. No equation is evaluated.

        """
        if not (math.isfinite(radius) and radius >= 0.0):
            raise ValueError("init_radius must be finite and non-negative")
        generator = stream.generator()
        for _ in range(100):
            point = tuple(
                generator.uniform(-radius, radius) for _ in range(self.dimension)
            )
            if math.isfinite(self.log_density(point)):
                return point
            if radius == 0.0:
                break
        raise ValueError(
            "could not find a starting point with finite log-density; check the "
            "data and the priors (ModelIR.terms shows the failing term)"
        )


def flatten(values: Mapping[str, SiteValue], sites: Sequence[SiteInfo]) -> Vector:
    """Flatten named site values in the order of ``sites``.

    References:
        Project API policy. No equation is evaluated.

    """
    flat: list[float] = []
    for site in sites:
        value = values[site.name]
        if isinstance(value, tuple):
            flat.extend(float(v) for v in value)
        else:
            flat.append(float(value))
    return tuple(flat)


def trace(model: Model) -> ModelIR:
    """Execute a model once and return its intermediate representation.

    Every latent site takes ``T(0)`` (``0`` on ``R``, ``a + 1`` on
    ``(a, inf)``, the midpoint of ``(a, b)``, ``loc`` for non-centred sites)
    so that tracing is deterministic. The site structure (names, kinds,
    sizes, order) must not depend on parameter values; later executions are
    checked against the traced structure.

    Args:
        model: Function ``model(m)`` using :class:`ModelHandle`.

    Returns:
        The traced :class:`ModelIR`.

    Raises:
        TypeError: If ``model`` is not callable.
        ModelSpecificationError: For duplicate site names, unsupported
            supports, or condition names that match no site. Observed sites
            without data are allowed here (their recorded value is ``None``)
            but rejected by the inference functions.

    References:
        [REF-BINGHAM-ETAL-2019], [REF-CARPENTER-ETAL-2017].

    """
    if not callable(model):
        raise TypeError("model must be callable")
    handle = ModelHandle("trace")
    model(handle)
    seen = {record.name for record in handle.records}
    unknown = sorted(handle.conditioned_names - seen)
    if unknown:
        raise ModelSpecificationError(
            f"condition() names sites that the model never declares: {unknown}"
        )
    return ModelIR(model, tuple(handle.records))
