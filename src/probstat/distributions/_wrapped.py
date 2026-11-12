"""Shared helpers of the composite laws ``Affine``, ``Truncated``, ``Mixture``.

Each composite law accepts any object with the structural marginal interface
``pdf``, ``cdf``, ``quantile``, ``mean`` and ``variance``
(``probstat.modeling.ContinuousMarginal``) and satisfies that interface
itself, so wrapped laws can be model inputs. Optional ``sf``, ``isf``,
``log_pdf`` and ``sample`` methods of the base law are used when present
(for tail accuracy) and replaced by ``1 - cdf``, ``quantile(1 - q)`` and
``log pdf`` otherwise; the helpers here implement that dispatch.
"""

from __future__ import annotations

import math
from typing import Any, cast

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)

from ._base import ContinuousDistribution
from ._continuous import open_unit, safe_log

_REQUIRED = ("pdf", "cdf", "quantile", "mean", "variance")


def _check_law(law: object, name: str) -> None:
    """Require the structural marginal interface.

    Raises:
        TypeError: If a required attribute is missing.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    for attribute in _REQUIRED:
        if not hasattr(law, attribute):
            raise TypeError(f"{name} lacks required {attribute!r}")


def _sf(law: Any, x: float) -> float:
    """Return ``S(x)`` from ``law.sf`` or ``1 - law.cdf``.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 1]. No new equation.

    """
    method = getattr(law, "sf", None)
    if method is not None:
        return float(method(x))
    return 1.0 - float(law.cdf(x))


def _isf(law: Any, q: float) -> float:
    """Return ``S^-1(q)`` from ``law.isf`` or ``law.quantile(1 - q)``.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 2]. No new equation.

    """
    method = getattr(law, "isf", None)
    if method is not None:
        return float(method(q))
    return float(law.quantile(1.0 - q))


def _log_pdf(law: Any, x: float) -> float:
    """Return ``log f(x)`` from ``law.log_pdf`` or ``log law.pdf``.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 3]. No new equation.

    """
    method = getattr(law, "log_pdf", None)
    if method is not None:
        try:
            return float(method(x))
        except ValueError:
            return -math.inf
    return safe_log(float(law.pdf(x)))


def _default_property(instance: object, name: str) -> float:
    """Return the numerical default of a :class:`ContinuousDistribution` property.

    References:
        [EQ-DIST-BASE-001]. No new equation.

    """
    descriptor = getattr(ContinuousDistribution, name)
    return float(cast(property, descriptor).__get__(instance))


def _finite_point(x: float) -> float:
    """Validate a real point, allowing infinities (handled by the guards).

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    return require_finite_real(x, name="x")


def _base_sample(
    law: Any, size: int, seed: int | RandomStream | None
) -> tuple[float, ...]:
    """Sample a base law by its own sampler when present, else by inversion.

    References:
        [REF-DEVROYE-1986, Chapter II]. No new equation.

    """
    count = require_non_negative_integer(size, name="size")
    stream = resolve_stream(seed)
    sampler = getattr(law, "sample", None)
    if sampler is not None:
        return tuple(float(v) for v in sampler(count, seed=stream))
    generator = stream.generator()
    return tuple(float(law.quantile(open_unit(generator))) for _ in range(count))
