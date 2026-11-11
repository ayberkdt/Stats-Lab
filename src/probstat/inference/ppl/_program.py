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

Structured sites (simplex, ordered, correlation, covariance) use the
transforms of :class:`~._structured.StructuredSupport`. Discrete sites
declared with ``marginalize=True`` take their first support value in
``trace`` mode and the value of the current configuration in ``density``
mode; :class:`ModelIR` re-runs the model once per configuration of the
discrete sites outside plates, and :meth:`ModelHandle.plate` re-runs one
plate iteration per configuration of its own discrete sites, so that the
density on ``u`` is the marginal over every discrete site.

Automatic differentiation: density mode is transparent to
:class:`~probstat.numerics.Dual` numbers and to reverse-mode
:class:`~probstat.numerics.ReverseVariable` values (bijections,
log-Jacobians, the inlined normal density and non-centred sites use
:class:`~probstat.numerics.admath`). Whenever the *model code and its
distribution objects* accept them (the program is traceable), the
inference entry points take the exact reverse-mode gradient of
:meth:`ModelIR.log_density`: one model execution recorded on a tape and one
backward sweep ([EQ-AD-REVERSE-001], ADR-0069). The distribution classes of
:mod:`probstat.distributions` validate their parameters as floats, so a site
such as ``Normal(mu, tau)`` with a latent ``mu`` is not traceable; the
entry points then fall back to finite differences and record why
(``gradient_method``, ``gradient_fallback_reason``). Duck-typed
distributions whose ``log_pdf`` uses ``admath`` are differentiated exactly.

Floating-point limit: on an interval ``(a, b)`` the map
``x = a + (b - a) sigmoid(u)`` rounds to exactly ``b`` once ``u`` exceeds
about 37, where the density is evaluated at the bound and treated as zero.
For a density with an integrable singularity at the upper bound (for
example ``Beta(1, 0.05)``) this silently drops mass (16% for that law in a
review); the lower bound at ``a = 0`` keeps full resolution. Reparameterize
such a site (for example ``1 - x``) so the singular end sits at zero.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from probstat.distributions import Normal, RealSupport
from probstat.foundations.random_streams import RandomStream
from probstat.linalg import Vector, as_vector
from probstat.numerics import Dual, GradientEvaluator, ReverseVariable, admath
from probstat.optimization import central_gradient

from ._discrete import (
    ENUMERATION_LIMIT,
    DiscreteConditional,
    EnumerationGroup,
    LocalEnumeration,
    finite_support_values,
    log_sum_exp,
    primal,
)
from ._errors import ModelSpecificationError
from ._structured import StructuredSupport, structure_for, structured_log_density

SiteKind = Literal["latent", "observed", "deterministic", "discrete"]
SiteValue = float | tuple[float, ...] | tuple[tuple[float, ...], ...]
Model = Callable[["ModelHandle"], object]

_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_DIFFERENTIABLE = (Dual, ReverseVariable)
_NUMERIC_ERRORS = (ValueError, ArithmeticError)


# ---------------------------------------------------------------------------
# Bijections to unconstrained space
# ---------------------------------------------------------------------------


def _softplus(t: float) -> float:
    """Return ``log(1 + exp(t))`` without overflow (dual-transparent).

    References:
        [EQ-PPL-TRANSFORM-002]. No new equation is introduced.

    """
    positive = t if t > 0.0 else 0.0 * t
    return cast(float, positive + admath.log1p(admath.exp(-abs(t))))


def _sigmoid(t: float) -> float:
    """Return ``1 / (1 + exp(-t))`` without overflow (dual-transparent).

    References:
        [EQ-PPL-TRANSFORM-002]. No new equation is introduced.

    """
    if t >= 0.0:
        return cast(float, 1.0 / (1.0 + admath.exp(-t)))
    e = admath.exp(t)
    return cast(float, e / (1.0 + e))


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
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

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
            return cast(float, self.lower + admath.exp(u))
        if self.kind == "upper":
            assert self.upper is not None
            return cast(float, self.upper - admath.exp(u))
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
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

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
            "a continuous RealSupport (a scalar discrete site with a finite "
            "support can be summed out with marginalize=True)"
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
        Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

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
    result = method(value)
    if isinstance(result, _DIFFERENTIABLE):
        return result  # type: ignore[return-value]
    return float(result)


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
        Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

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
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

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
        Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

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
        kind: ``"latent"``, ``"observed"``, ``"deterministic"``, or
            ``"discrete"`` (a latent site summed out by enumeration).
        size: ``None`` for a scalar site, else the number of scalar entries
            (the vector length, or ``d * d`` for a matrix site).
        distribution: Class name of the site's distribution (``""`` for a
            deterministic site); ``"Normal x 8"`` style for per-element lists.
        support: Support at trace time (``None`` for deterministic sites).
        bijection: Bijection at trace time (scalar and vector latent sites).
        reparam: ``"noncentered"`` or ``None``.
        value: Value at trace time; for observed sites this is the data.
        element_bijections: Bijection of every element of a vector site.
        structure: Structured support of a simplex, ordered, correlation, or
            covariance site, else ``None``.
        enumeration: Support values of a discrete site, else ``()``.
        plate: Name of the enclosing plate, else ``None``.

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
    structure: StructuredSupport | None = None
    enumeration: tuple[Any, ...] = ()
    plate: str | None = None

    @property
    def length(self) -> int:
        """Return the number of scalar elements (``1`` for a scalar site).

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return 1 if self.size is None else self.size

    @property
    def free_length(self) -> int:
        """Return the number of unconstrained coordinates of a latent site.

        Equals :attr:`length` except for structured sites (for example
        ``K - 1`` for a ``K``-simplex) and discrete sites (``0``).

        References:
            [REF-STAN-REFERENCE-MANUAL-2-36]. No equation is evaluated.

        """
        if self.kind == "discrete":
            return 0
        if self.structure is not None:
            return self.structure.free_size
        return self.length

    def flat_names(self, base: str | None = None) -> tuple[str, ...]:
        """Return element names ``name``, ``name[0], ...``, or ``name[i,j]``.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        label = self.name if base is None else base
        if self.structure is not None:
            return self.structure.flat_names(label)
        if self.size is None:
            return (label,)
        return tuple(f"{label}[{i}]" for i in range(self.size))


def missing_data_message(site: str) -> str:
    """Return the error message for an observed site without data.

    References:
        Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

    """
    return (
        f"observed site {site!r} has no data; pass a value or use "
        f"condition(model, {{{site!r}: data}})"
    )


def _check_data(site: str, values: Sequence[object]) -> None:
    """Reject observed data that are not finite real numbers.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

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
        Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

    """
    names = {type(p).__name__ for p in parts}
    label = names.pop() if len(names) == 1 else "mixed"
    return label if size is None else f"{label} x {size}"


# ---------------------------------------------------------------------------
# The handle passed to model functions
# ---------------------------------------------------------------------------


class ModelHandle:
    """The object a model function receives as its argument ``m``.

    Users call :meth:`sample`, :meth:`observe`, :meth:`deterministic`, and
    :meth:`plate`; the inference functions construct the handle in the right
    mode.

    Args:
        mode: ``"trace"``, ``"density"``, or ``"generate"``.
        unconstrained: Unconstrained vector (``density`` mode).
        layout: Expected site sequence; when given, each site is checked
            against it so that structural changes are reported.
        fixed: Constrained latent values by name (``generate`` mode).
        require_fixed: Whether every continuous latent site must appear in
            ``fixed``.
        generator: Random generator (``generate`` mode).
        conditioned: Observed values by site name (see :func:`condition`).
        global_assignment: Support indices of the discrete sites outside
            plates, in execution order (``density`` mode).
        record_enumeration: Whether to keep the enumerated branch weights of
            every plate iteration in :attr:`enumeration_log`.

    References:
        [REF-BINGHAM-ETAL-2019], [REF-CARPENTER-ETAL-2017].

    """

    __slots__ = (
        "_conditioned",
        "_cursor",
        "_enum",
        "_fixed",
        "_generator",
        "_global_assignment",
        "_global_position",
        "_global_seen",
        "_layout",
        "_local_seen",
        "_local_sink",
        "_mode",
        "_name_log",
        "_names",
        "_offset",
        "_plate",
        "_require_fixed",
        "_unconstrained",
        "discrete_term",
        "enumeration_log",
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
        global_assignment: tuple[int, ...] = (),
        record_enumeration: bool = False,
    ) -> None:
        """Initialize an empty execution record.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

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
        self._name_log: list[str] = []
        self._global_assignment = global_assignment
        self._global_position = 0
        self._global_seen = False
        self._local_seen = False
        self._plate: str | None = None
        self._enum: LocalEnumeration | None = None
        self._local_sink: Any = None
        self.discrete_term: Any = None
        self.enumeration_log: list[EnumerationGroup] | None = (
            [] if record_enumeration else None
        )

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
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        if not isinstance(name, str) or not name:
            raise ModelSpecificationError("site names must be non-empty strings")
        if name in self._names:
            raise ModelSpecificationError(f"duplicate site name {name!r}")
        self._names.add(name)
        self._name_log.append(name)
        return name

    def _check_order(self, site: str, kind: str) -> None:
        """Reject a continuous latent or deterministic site after a discrete one.

        Within its scope (the whole model for discrete sites outside plates,
        one plate iteration for discrete sites inside), every discrete site
        must follow the continuous latent and deterministic sites; only
        observed and discrete sites may come after it. This keeps the
        unconstrained coordinates and the reported values the same in every
        enumerated configuration.

        References:
            [EQ-PPL-ENUM-001], [REF-STAN-USERS-GUIDE-2-36].

        """
        if self._global_seen or self._local_seen:
            raise ModelSpecificationError(
                f"{kind} site {site!r} follows a discrete latent site in its scope; "
                "declare continuous latent and deterministic sites before the "
                "discrete sites (only observed and discrete sites may follow them)"
            )

    def _add_likelihood(self, value: Any) -> None:
        """Add a likelihood term to the active enumeration sink or the total.

        References:
            [EQ-PPL-JOINT-001], [EQ-PPL-ENUM-001].

        """
        if self._local_sink is not None:
            self._local_sink = self._local_sink + value
        elif self.discrete_term is not None:
            self.discrete_term = self.discrete_term + value
        else:
            self.log_likelihood += value

    def _check_layout(self, name: str, kind: SiteKind, size: int | None) -> None:
        """Compare a site with the traced structure.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

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
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

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
        marginalize: bool = False,
    ) -> Any:
        """Declare a latent random variable (or an observed one if conditioned).

        Three kinds of latent site exist:

        * scalar and vector sites of continuous laws on intervals, mapped by
          the bijections of :class:`Bijection`;
        * structured sites whose distribution lives on a simplex
          (:class:`~probstat.multivariate.Dirichlet`), on ordered vectors
          (:class:`OrderedVector`), on Cholesky factors of correlation
          matrices (:class:`~probstat.multivariate.LKJCholesky`), on
          correlation matrices (:class:`~probstat.multivariate.LKJCorrelation`),
          or on covariance matrices (Wishart, inverse Wishart), mapped by the
          transforms of :class:`StructuredSupport`;
        * with ``marginalize=True``, scalar discrete sites with a finite
          support, summed out of the density by enumeration and recovered
          afterwards from their exact conditional law.

        Args:
            name: Unique site name.
            distribution: A distribution with ``support``, ``log_pdf`` (or
                ``log_pmf``) and ``sample``; or a list/tuple of them (one per
                element).
            shape: ``None`` for a scalar, ``n`` for an i.i.d. vector of length
                ``n`` (inferred from a distribution list); not allowed for
                structured or discrete sites.
            reparam: ``"noncentered"`` samples ``z ~ N(0, 1)`` and returns
                ``loc + scale z``; only for ``Normal`` sites.
            marginalize: Sum a finite discrete site out by enumeration. Inside
                an iteration of :meth:`plate` the site is enumerated with that
                iteration only; elsewhere it is enumerated jointly with the
                other such sites by re-running the model.

        Returns:
            A float (scalar site), a tuple of floats (vector site), a tuple of
            rows (matrix site), or a support value (discrete site).

        Raises:
            ModelSpecificationError: For duplicate names, bad shapes, an
                unsupported or unbounded support, an invalid ``reparam``, or a
                continuous latent site declared after a discrete one in its
                scope.

        Equation:
            [EQ-PPL-JOINT-001], [EQ-PPL-NONCENTERED-001],
            [EQ-PPL-TRANSFORM-003], [EQ-PPL-ENUM-001].

        References:
            [REF-BINGHAM-ETAL-2019], [REF-CARPENTER-ETAL-2017],
            [REF-PAPASPILIOPOULOS-ROBERTS-SKOLD-2007],
            [REF-STAN-REFERENCE-MANUAL-2-36].

        """
        site = self._register(name)
        if site in self._conditioned:
            self._names.discard(site)
            data = self._conditioned[site]
            return self.observe(site, distribution, data, shape=shape)
        if marginalize:
            return self._discrete_site(site, distribution, shape, reparam)
        structure = structure_for(distribution)
        if structure is not None:
            return self._structured_site(site, distribution, structure, shape, reparam)
        self._check_order(site, "latent")
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
                return [float(v) for v in cast(tuple[float, ...], fixed)]
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
                self.log_prior += cast(
                    float, admath.fsum(-0.5 * u * u for u in u_values)
                ) - (count * _LOG_SQRT_TWO_PI)
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

    def _structured_site(
        self,
        site: str,
        distribution: object,
        structure: StructuredSupport,
        shape: object,
        reparam: str | None,
    ) -> Any:
        """Produce a simplex, ordered, correlation, or covariance site.

        References:
            [EQ-PPL-TRANSFORM-003], [EQ-PPL-TRANSFORM-004],
            [EQ-PPL-TRANSFORM-005], [EQ-PPL-TRANSFORM-006],
            [REF-STAN-REFERENCE-MANUAL-2-36].

        """
        if shape is not None or reparam is not None:
            raise ModelSpecificationError(
                f"site {site!r}: shape and reparam do not apply to a structured "
                f"{structure.kind} site"
            )
        self._check_order(site, "latent")
        size = structure.constrained_size
        self._check_layout(site, "latent", size)
        if self._layout is not None:
            expected = self._layout[self._cursor - 1].structure
            if expected != structure:
                raise ModelSpecificationError(
                    f"model structure changed: site {site!r} has support "
                    f"{structure} instead of {expected}"
                )
        value: Any
        if self._mode == "generate":
            if site in self._fixed:
                value = structure.checked_value(self._fixed[site], site)
            elif self._require_fixed:
                raise ModelSpecificationError(
                    f"latent site {site!r} has no posterior value"
                )
            else:
                assert self._generator is not None
                value = structure.checked_value(
                    _draw(distribution, self._generator), site
                )
        elif self._mode == "trace":
            value = structure.forward((0.0,) * structure.free_size).value
        else:
            start = self._offset
            stop = start + structure.free_size
            if stop > len(self._unconstrained):
                raise ModelSpecificationError(
                    "unconstrained vector is shorter than the model dimension"
                )
            point = structure.forward(self._unconstrained[start:stop])
            self._offset = stop
            self.log_jacobian += point.log_jacobian
            self.log_prior += structured_log_density(
                distribution, structure, point, _log_density
            )
            value = point.value
        self.values[site] = value
        if self._mode == "density":
            return value
        self.records.append(
            SiteInfo(
                site,
                "latent",
                size,
                type(distribution).__name__,
                structure,
                None,
                None,
                value,
                structure=structure,
                plate=self._plate,
            )
        )
        return value

    def _discrete_site(
        self,
        site: str,
        distribution: object,
        shape: object,
        reparam: str | None,
    ) -> Any:
        """Produce a discrete site that is summed out by enumeration.

        In ``density`` mode the site takes the value of the current
        configuration and its log-mass starts (or extends) the enumeration
        sink of its scope; in ``trace`` mode it takes its first support value;
        in ``generate`` mode it is fixed by name or drawn from its law.

        References:
            [EQ-PPL-ENUM-001], [REF-STAN-USERS-GUIDE-2-36],
            [REF-OBERMEYER-ETAL-2019].

        """
        if shape is not None or reparam is not None or isinstance(
            distribution, list | tuple
        ):
            raise ModelSpecificationError(
                f"discrete site {site!r}: only scalar sites with one distribution "
                "can be marginalized (use a plate for repeated indicators)"
            )
        values = finite_support_values(_support_of(distribution, site), site)
        if values is None:
            raise ModelSpecificationError(
                f"site {site!r}: marginalize=True needs a finite discrete support "
                "(IntegerSupport with both bounds, or FiniteSupport)"
            )
        self._check_layout(site, "discrete", None)
        local = self._plate is not None
        value: Any
        if self._mode == "density":
            if local:
                assert self._enum is not None
                index = self._enum.index_for(site, values)
            else:
                assert self._layout is not None
                if self._layout[self._cursor - 1].enumeration != values:
                    raise ModelSpecificationError(
                        f"model structure changed: the support of discrete site "
                        f"{site!r} differs from the traced one"
                    )
                index = self._global_assignment[self._global_position]
                self._global_position += 1
            value = values[index]
            term = _log_density(distribution, value)
            if local:
                self._local_seen = True
                sink = self._local_sink
                self._local_sink = term if sink is None else sink + term
            else:
                self._global_seen = True
                sink = self.discrete_term
                self.discrete_term = term if sink is None else sink + term
            return value
        if self._mode == "generate":
            if site in self._fixed:
                value = self._fixed[site]
                if value not in values:
                    raise ModelSpecificationError(
                        f"discrete site {site!r}: fixed value {value!r} is not in "
                        "the support"
                    )
            else:
                assert self._generator is not None
                value = _draw(distribution, self._generator)
        else:
            value = values[0]
        if local:
            self._local_seen = True
        else:
            self._global_seen = True
        self.records.append(
            SiteInfo(
                site,
                "discrete",
                None,
                type(distribution).__name__,
                _support_of(distribution, site),
                None,
                None,
                value,
                enumeration=values,
                plate=self._plate,
            )
        )
        return value

    def plate(self, name: str, size: int) -> Iterator[int]:
        """Iterate over ``size`` conditionally independent replicates.

        ``for i in m.plate("data", n): ...`` runs the loop body once per
        index in ``trace`` and ``generate`` mode. In ``density`` mode, an
        iteration that declares discrete sites with ``marginalize=True`` is
        re-run once per joint configuration of those sites and contributes
        ``log sum_z exp(terms after the first discrete site)``, so ``n``
        binary indicators cost ``2 n`` executions of the body instead of the
        ``2^n`` of joint enumeration. The declaration asserts that iteration
        ``i`` does not influence iteration ``j``; the loop body must not have
        side effects other than site declarations, because it is repeated.

        Args:
            name: Plate name (used in error messages and site records).
            size: Positive number of iterations.

        Returns:
            Iterator over ``0, ..., size - 1``.

        Raises:
            ModelSpecificationError: For an invalid name or size, or a plate
                nested in another plate.

        Equation:
            [EQ-PPL-ENUM-001].

        References:
            [REF-OBERMEYER-ETAL-2019], [REF-BINGHAM-ETAL-2019].

        """
        if not isinstance(name, str) or not name:
            raise ModelSpecificationError("plate names must be non-empty strings")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ModelSpecificationError(
                f"plate {name!r}: size must be a positive integer"
            )
        if self._plate is not None:
            raise ModelSpecificationError(
                f"plate {name!r} is nested in plate {self._plate!r}; nested "
                "plates are not supported"
            )
        return self._plate_iterations(name, size)

    def _plate_iterations(self, name: str, size: int) -> Iterator[int]:
        """Yield plate indices, enumerating local discrete sites in density mode.

        References:
            [EQ-PPL-ENUM-001], [REF-OBERMEYER-ETAL-2019].

        """
        self._plate = name
        try:
            for index in range(size):
                self._local_seen = False
                if self._mode != "density":
                    yield index
                    continue
                yield from self._enumerated_iteration(index)
        finally:
            self._plate = None
            self._local_seen = False
            self._local_sink = None
            self._enum = None

    def _checkpoint(self) -> tuple[Any, ...]:
        """Return the state to restore before re-running a plate iteration.

        References:
            [EQ-PPL-ENUM-001]. No new equation is introduced.

        """
        return (
            self._cursor,
            self._offset,
            len(self._name_log),
            self.log_prior,
            self.log_likelihood,
            self.log_jacobian,
            self.discrete_term,
        )

    def _restore(self, checkpoint: tuple[Any, ...]) -> None:
        """Rewind the handle to a checkpoint taken at an iteration start.

        References:
            [EQ-PPL-ENUM-001]. No new equation is introduced.

        """
        (
            self._cursor,
            self._offset,
            names,
            self.log_prior,
            self.log_likelihood,
            self.log_jacobian,
            self.discrete_term,
        ) = checkpoint
        for registered in self._name_log[names:]:
            self._names.discard(registered)
        del self._name_log[names:]

    def _enumerated_iteration(self, index: int) -> Iterator[int]:
        """Run one plate iteration once per configuration of its discrete sites.

        The first pass discovers the discrete sites (all at support index
        ``0``); every further configuration rewinds the handle to the
        iteration start and re-runs the body. The terms after the first
        discrete site of each pass are combined by a log-sum-exp.

        References:
            [EQ-PPL-ENUM-001], [REF-OBERMEYER-ETAL-2019].

        """
        checkpoint = self._checkpoint()
        state = LocalEnumeration()
        self._enum = state
        self._local_sink = None
        yield index
        if not state.sites:
            self._enum = None
            return
        totals = [self._local_sink]
        configurations = tuple(
            itertools.product(*(range(len(values)) for _, values in state.sites))
        )
        for configuration in configurations[1:]:
            self._restore(checkpoint)
            state.assignment = configuration
            state.position = 0
            self._local_sink = None
            self._local_seen = False
            yield index
            if state.position != len(state.sites):
                raise ModelSpecificationError(
                    "model structure changed: a plate iteration declared a different "
                    "number of discrete sites in another enumerated branch"
                )
            totals.append(self._local_sink)
        self._enum = None
        self._local_sink = None
        self._local_seen = False
        if self.enumeration_log is not None:
            self.enumeration_log.append(
                EnumerationGroup(
                    tuple(name for name, _ in state.sites),
                    tuple(values for _, values in state.sites),
                    configurations,
                    tuple(primal(total) for total in totals),
                )
            )
        self._add_likelihood(log_sum_exp(totals))

    def _observe_joint(
        self, site: str, distribution: Any, value: object, shape: object
    ) -> Any:
        """Declare one vector- or matrix-valued observation of a joint law.

        References:
            [REF-BINGHAM-ETAL-2019]. Equation [EQ-PPL-JOINT-001].

        """
        if shape is not None:
            raise ModelSpecificationError(
                f"observed site {site!r}: shape does not apply to a joint law"
            )
        data: Any = None
        flat: tuple[object, ...] = ()
        if value is not None:
            if isinstance(value, str) or not isinstance(value, Sequence):
                raise ModelSpecificationError(
                    f"observed site {site!r}: data of a joint law must be a "
                    "vector or a matrix"
                )
            rows = tuple(value)
            if rows and all(
                isinstance(r, Sequence) and not isinstance(r, str) for r in rows
            ):
                data = tuple(tuple(r) for r in rows)
                flat = tuple(v for row in data for v in row)
            else:
                data = rows
                flat = rows
            if not flat:
                raise ModelSpecificationError(
                    f"observed site {site!r}: data must be non-empty"
                )
        size = len(flat) if flat else None
        self._check_layout(site, "observed", size)
        if self._mode == "trace" and value is not None:
            _check_data(site, flat)
        if self._mode == "generate":
            assert self._generator is not None
            result = _draw(distribution, self._generator)
        elif value is None:
            if self._mode == "density":
                raise ModelSpecificationError(missing_data_message(site))
            result = None
        else:
            if self._mode == "density":
                self._add_likelihood(_log_density(distribution, data))
                return data
            result = data
        self.records.append(
            SiteInfo(
                site,
                "observed",
                size,
                type(distribution).__name__,
                getattr(distribution, "support", None),
                None,
                None,
                result,
            )
        )
        return result

    def observe(
        self,
        name: str,
        distribution: object,
        value: object = None,
        *,
        shape: int | None = None,
    ) -> Any:
        """Declare an observed random variable and its data.

        A distribution with a ``dimension`` attribute (the laws of
        :mod:`probstat.multivariate`, such as ``Multinomial`` or
        ``MultivariateNormal``) is a joint law: ``value`` is then one
        vector or matrix observation and contributes one ``log_pdf`` (or
        ``log_pmf``) term.

        Args:
            name: Unique site name.
            distribution: A distribution (shared by all elements), a list
                of distributions (one per element), or one joint law.
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
        if not isinstance(distribution, list | tuple) and hasattr(
            distribution, "dimension"
        ):
            return self._observe_joint(site, distribution, value, shape)
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
                self._add_likelihood(total)
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
        self._check_order(site, "deterministic")
        size: int | None
        density = self._mode == "density"
        if isinstance(value, Sequence) and not isinstance(value, str):
            try:
                elements = tuple(
                    v if density and isinstance(v, _DIFFERENTIABLE) else float(v)
                    for v in value
                )
            except (TypeError, ValueError):
                raise ModelSpecificationError(
                    f"deterministic site {site!r} must hold real numbers"
                ) from None
            packed: SiteValue = cast(tuple[float, ...], elements)
            size = len(elements)
            if size == 0:
                raise ModelSpecificationError(
                    f"deterministic site {site!r} must be non-empty"
                )
        elif density and isinstance(value, _DIFFERENTIABLE):
            packed = value  # type: ignore[assignment]
            size = None
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
    _global_sizes: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Cache the support sizes of the discrete sites outside plates.

        References:
            [EQ-PPL-ENUM-001]. No new equation is introduced.

        """
        sizes = tuple(
            len(site.enumeration)
            for site in self.sites
            if site.kind == "discrete" and site.plate is None
        )
        object.__setattr__(self, "_global_sizes", sizes)

    # -- structure ---------------------------------------------------------

    def _of_kind(self, kind: SiteKind) -> tuple[SiteInfo, ...]:
        """Return the sites of one kind in execution order.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return tuple(site for site in self.sites if site.kind == kind)

    @property
    def latent_sites(self) -> tuple[SiteInfo, ...]:
        """Return latent sites in execution order.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return self._of_kind("latent")

    @property
    def observed_sites(self) -> tuple[SiteInfo, ...]:
        """Return observed sites in execution order.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return self._of_kind("observed")

    @property
    def deterministic_sites(self) -> tuple[SiteInfo, ...]:
        """Return deterministic sites in execution order.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return self._of_kind("deterministic")

    @property
    def discrete_sites(self) -> tuple[SiteInfo, ...]:
        """Return the marginalized discrete sites in execution order.

        References:
            [EQ-PPL-ENUM-001], [REF-STAN-USERS-GUIDE-2-36].

        """
        return self._of_kind("discrete")

    def _global_configurations(self) -> tuple[tuple[int, ...], ...]:
        """Return every joint configuration of the discrete sites outside plates.

        References:
            [EQ-PPL-ENUM-001]. No new equation is introduced.

        """
        return tuple(itertools.product(*(range(size) for size in self._global_sizes)))

    def site(self, name: str) -> SiteInfo:
        """Return the site called ``name``.

        Raises:
            KeyError: If no site has that name.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

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
        return sum(site.free_length for site in self.latent_sites)

    @property
    def unconstrained_names(self) -> tuple[str, ...]:
        """Return coordinate names such as ``mu``, ``log(tau)``, ``theta_z[0]``.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        names: list[str] = []
        for site in self.latent_sites:
            if site.structure is not None:
                names.extend(site.structure.free_names(site.name))
                continue
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
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return tuple(s for s in self.sites if s.kind in ("latent", "deterministic"))

    @property
    def constrained_names(self) -> tuple[str, ...]:
        """Return flat names of latent and deterministic elements.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return tuple(name for s in self.value_sites for name in s.flat_names())

    def observed_data(self, name: str) -> SiteValue:
        """Return the data of an observed site as floats.

        Raises:
            KeyError: If ``name`` is not an observed site.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

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

    def _run_density(
        self,
        u: Sequence[float],
        assignment: tuple[int, ...] | None = None,
        *,
        record: bool = False,
    ) -> ModelHandle:
        """Execute the model in density mode and check its structure.

        Args:
            u: Unconstrained vector.
            assignment: Support indices of the discrete sites outside plates
                (default: the first value of each).
            record: Whether to record the enumerated plate iterations.

        References:
            [EQ-PPL-JOINT-001]. No new equation is introduced.

        """
        point = tuple(u)
        if len(point) != self.dimension:
            raise ModelSpecificationError(
                f"unconstrained vector has length {len(point)}, expected "
                f"{self.dimension}"
            )
        handle = ModelHandle(
            "density",
            unconstrained=point,
            layout=self.sites,
            global_assignment=(
                assignment if assignment is not None else (0,) * len(self._global_sizes)
            ),
            record_enumeration=record,
        )
        self.model(handle)
        handle.finish()
        return handle

    def _enumerated_terms(
        self, u: Sequence[float]
    ) -> tuple[ModelHandle, Any]:
        """Run every global configuration and log-sum-exp the enumerated terms.

        Returns the handle of the first configuration (whose continuous
        terms and values are shared by all configurations) and
        ``log sum_c exp(terms after the first global discrete site)``.

        References:
            [EQ-PPL-ENUM-001], [REF-STAN-USERS-GUIDE-2-36].

        """
        first: ModelHandle | None = None
        terms: list[Any] = []
        for assignment in self._global_configurations():
            handle = self._run_density(u, assignment)
            if first is None:
                first = handle
            terms.append(handle.discrete_term)
        assert first is not None
        return first, log_sum_exp(terms)

    def discrete_conditional(self, u: Sequence[float]) -> DiscreteConditional:
        """Return the exact conditional law of the discrete sites at ``u``.

        For each joint configuration ``c`` of the discrete sites outside
        plates, the unnormalized ``log p(c, y | theta)`` and, for each plate
        iteration, the unnormalized ``log p(z_i, y_i | theta, c)`` of its
        branches.

        Args:
            u: Unconstrained vector of floats.

        Returns:
            The enumeration record used by posterior recovery.

        Raises:
            ModelSpecificationError: If the model has no discrete sites.

        Equation:
            [EQ-PPL-ENUM-002].

        References:
            [REF-STAN-USERS-GUIDE-2-36].

        """
        if not self.discrete_sites:
            raise ModelSpecificationError("the model has no discrete sites")
        globals_ = tuple(
            site for site in self.discrete_sites if site.plate is None
        )
        configurations = self._global_configurations()
        weights: list[float] = []
        groups: list[tuple[EnumerationGroup, ...]] = []
        for assignment in configurations:
            handle = self._run_density(u, assignment, record=True)
            term = handle.discrete_term
            weights.append(0.0 if term is None else primal(term))
            assert handle.enumeration_log is not None
            groups.append(tuple(handle.enumeration_log))
        return DiscreteConditional(
            tuple(site.name for site in globals_),
            tuple(site.enumeration for site in globals_),
            configurations,
            tuple(weights),
            tuple(groups),
        )

    def terms(self, u: Sequence[float]) -> LogDensityTerms:
        """Return the prior, likelihood, and Jacobian terms at ``u``.

        Numerical failures inside the model propagate here (unlike
        :meth:`log_density`), which makes this the method to debug with.
        With marginalized discrete sites, ``log_likelihood`` holds the
        enumerated terms (the discrete log-masses and the observations
        that follow the first discrete site of a scope) after the
        log-sum-exp over configurations.

        Args:
            u: Unconstrained vector of length :attr:`dimension`.

        Returns:
            Density terms and the constrained values.

        Equation:
            [EQ-PPL-JOINT-001].

        References:
            [REF-CARPENTER-ETAL-2017].

        """
        if self._global_sizes:
            handle, enumerated = self._enumerated_terms(u)
            return LogDensityTerms(
                handle.log_prior,
                handle.log_likelihood + enumerated,
                handle.log_jacobian,
                dict(handle.values),
            )
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
        still propagate, and so do ``TypeError``s (for example a distribution
        that rejects :class:`~probstat.numerics.Dual` parameters), so that
        automatic differentiation can fall back to finite differences.

        Args:
            u: Unconstrained vector of floats, or of duals (forward mode) or
                reverse-mode variables (the result is then of the same
                kind).

        Returns:
            Finite log-density or ``-inf``.

        Equation:
            [EQ-PPL-JOINT-001].

        References:
            [REF-CARPENTER-ETAL-2017], [REF-KUCUKELBIR-ETAL-2017].

        """
        try:
            if self._global_sizes:
                handle, enumerated = self._enumerated_terms(u)
            else:
                handle = self._run_density(u)
                enumerated = None
        except ModelSpecificationError:
            raise
        except _NUMERIC_ERRORS:
            return -math.inf
        total = handle.log_prior + handle.log_likelihood + handle.log_jacobian
        if enumerated is not None:
            total = total + enumerated
        if admath.isnan(total) or total == math.inf:
            return -math.inf
        return total

    def gradient(
        self,
        u: Sequence[float],
        *,
        relative_step: float = 1e-6,
        method: str = "auto",
    ) -> Vector:
        """Return the gradient of :meth:`log_density`.

        ``method="auto"`` (default) records the model once on a
        reverse-mode tape (exact gradient from one backward sweep) and
        falls back to central differences when the program is not
        traceable; ``"reverse"`` raises instead of falling back;
        ``"autodiff"``/``"forward"`` uses forward-mode dual numbers (no
        fallback); ``"finite-difference"`` uses central differences only,
        two model executions per component with step
        ``h_i = relative_step * max(1, |u_i|)`` and truncation error
        ``O(h^2)``.

        Args:
            u: Unconstrained vector with finite log-density nearby.
            relative_step: Relative finite-difference step.
            method: ``"auto"`` (reverse mode with fallback), any other
                :data:`~probstat.numerics.GRADIENT_OPTIONS` entry.

        Returns:
            Gradient.

        Raises:
            ValueError: If the log-density (or a perturbed point) is not
                finite, or the method is unknown.
            AutodiffUnsupportedError: For ``"autodiff"`` or ``"reverse"`` on a
                model that is not traceable.

        Equation:
            [EQ-OPT-FD-001], [EQ-AD-DUAL-001], [EQ-AD-REVERSE-001].

        References:
            [REF-NOCEDAL-WRIGHT-2006, Section 8.1],
            [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        point = as_vector(u, name="u")
        return self.gradient_evaluator(method, relative_step=relative_step).gradient(
            point
        )

    def gradient_evaluator(
        self, method: object = "auto", *, relative_step: float = 1e-6
    ) -> GradientEvaluator:
        """Return a gradient oracle for :meth:`log_density`.

        ``"auto"`` (and ``None``) mean reverse mode with a
        finite-difference fallback (``"auto-reverse"`` of
        :class:`~probstat.numerics.GradientEvaluator`).

        Args:
            method: ``"auto"``, a
                :data:`~probstat.numerics.GRADIENT_OPTIONS` entry, or a
                callable gradient.
            relative_step: Relative finite-difference step.

        Returns:
            Evaluator whose ``method`` and ``fallback_reason`` record what
            was used.

        Raises:
            ValueError: If the method is unknown.

        Equation:
            [EQ-OPT-FD-001], [EQ-AD-DUAL-001], [EQ-AD-REVERSE-001].

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """

        def finite_difference(
            function: Callable[[Vector], float], point: Vector
        ) -> Sequence[float]:
            """Return the central-difference gradient of ``function``.

            References:
                [EQ-OPT-FD-001], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

            """
            values, _ = central_gradient(function, point, relative_step=relative_step)
            return values

        return GradientEvaluator(
            self.log_density,
            ppl_gradient_option(method),  # type: ignore[arg-type]
            finite_difference=finite_difference,
        )

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
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

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
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        values = tuple(flat)
        if len(values) != len(self.constrained_names):
            raise ValueError("flat vector does not match the constrained layout")
        result: dict[str, SiteValue] = {}
        offset = 0
        for site in self.value_sites:
            if site.structure is not None:
                result[site.name] = site.structure.unflatten(
                    values[offset : offset + site.length]
                )
            elif site.size is None:
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


def ppl_gradient_option(option: object) -> object:
    """Map the PPL gradient option to the engine option.

    ``"auto"`` and ``None`` become ``"auto-reverse"`` (reverse mode with a
    finite-difference fallback); every other value is passed on unchanged.

    Args:
        option: Gradient option given to a PPL entry point.

    Returns:
        The option for :class:`~probstat.numerics.GradientEvaluator` and
        the samplers.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

    """
    if option is None or option == "auto":
        return "auto-reverse"
    return option


def flatten(values: Mapping[str, SiteValue], sites: Sequence[SiteInfo]) -> Vector:
    """Flatten named site values in the order of ``sites``.

    References:
        Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

    """
    flat: list[float] = []
    for site in sites:
        value = values[site.name]
        if site.structure is not None:
            flat.extend(site.structure.flatten(value))
        elif isinstance(value, tuple):
            flat.extend(float(v) for v in value)  # type: ignore[arg-type]
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
    configurations = math.prod(
        len(record.enumeration)
        for record in handle.records
        if record.kind == "discrete" and record.plate is None
    )
    if configurations > ENUMERATION_LIMIT:
        raise ModelSpecificationError(
            f"the discrete sites outside plates have {configurations} joint "
            f"configurations; at most {ENUMERATION_LIMIT} can be enumerated "
            "(declare conditionally independent indicators inside m.plate)"
        )
    return ModelIR(model, tuple(handle.records))
