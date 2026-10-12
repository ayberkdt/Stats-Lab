"""API stability labels and the deprecation helper.

Every top-level package carries one of three labels (ADR-0048):

- ``stable``: names and semantics change only after a deprecation period of
  at least one minor release, announced by :func:`deprecated`.
- ``provisional``: the design is settled but signatures may still change;
  changes are deprecated first whenever a shim is practical.
- ``experimental``: recently added; anything may change in any release.

While the library is below 1.0 the labels describe intent, not a semantic
versioning guarantee; the changelog records every breaking change.
"""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Literal, ParamSpec, TypeVar

Stability = Literal["stable", "provisional", "experimental"]

_P = ParamSpec("_P")
_R = TypeVar("_R")

PACKAGE_STABILITY: Mapping[str, Stability] = MappingProxyType(
    {
        "foundations": "stable",
        "numerics": "stable",
        "linalg": "stable",
        "probability": "stable",
        "statistics": "stable",
        "distributions": "stable",
        "sampling": "stable",
        "optimization": "stable",
        "monte_carlo": "provisional",
        "multivariate": "provisional",
        "inference": "provisional",
        "uq": "provisional",
        "processes": "provisional",
        "modeling": "provisional",
        "propagation": "provisional",
        "inverse": "provisional",
        "reliability": "provisional",
        "sensitivity": "provisional",
        "surrogates": "provisional",
        "validation": "provisional",
        "risk": "provisional",
        "filtering": "provisional",
        "data_assimilation": "provisional",
        "workflow": "provisional",
        "visualization": "provisional",
        "information": "experimental",
        "causal": "experimental",
        "design": "experimental",
        "conformal": "experimental",
        "timeseries": "experimental",
    }
)


def package_stability(name: str) -> Stability:
    """Return the stability label of a top-level package.

    Args:
        name: Package name with or without the ``probstat.`` prefix.

    Returns:
        ``"stable"``, ``"provisional"``, or ``"experimental"``.

    Raises:
        KeyError: If the package has no label.

    References:
        Project API policy (ADR-0048). No equation is evaluated.

    """
    key = name.removeprefix("probstat.").split(".")[0]
    return PACKAGE_STABILITY[key]


def deprecated(
    *, since: str, removal: str, alternative: str | None = None
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Return a decorator that marks a callable as deprecated.

    Calling the decorated function emits a :class:`DeprecationWarning` that
    names the version that deprecated it, the version that will remove it,
    and the replacement; the docstring gains a ``Deprecated`` note.

    Args:
        since: Version in which the deprecation was announced, e.g. ``"0.15"``.
        removal: Earliest version in which the callable may be removed.
        alternative: Replacement to use instead, if any.

    Returns:
        A decorator that preserves the wrapped callable's signature.

    Raises:
        ValueError: If ``since`` or ``removal`` is empty.

    References:
        Project API policy (ADR-0048). No equation is evaluated.

    """
    if not since.strip() or not removal.strip():
        raise ValueError("since and removal must be non-empty version strings")

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        """Wrap ``function`` so that each call warns.

        References:
            Project API policy (ADR-0048). No equation is evaluated.

        """
        message = (
            f"{function.__qualname__} is deprecated since {since} and may be "
            f"removed in {removal}"
            + (f"; use {alternative} instead" if alternative else "")
        )

        @functools.wraps(function)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            """Warn, then call the deprecated function.

            References:
                Project API policy (ADR-0048). No equation is evaluated.

            """
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return function(*args, **kwargs)

        note = f"\n\nDeprecated:\n    {message}.\n"
        # cleandoc first, so the unindented note does not stop inspect.getdoc
        # from removing the body's indentation.
        wrapper.__doc__ = inspect.cleandoc(function.__doc__ or "") + note
        return wrapper

    return decorate
