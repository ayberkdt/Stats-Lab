"""Callable aliases, machine epsilon, and value validation shared by the solvers.

Private to :mod:`probstat.optimization`; the public names are the ones the
package re-exports.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector

ScalarFunction = Callable[[Vector], float]
VectorFunction = Callable[[Vector], Sequence[float]]


def _finite_value(value: object, name: str) -> float:
    """Validate one finite objective or residual value.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    return require_finite_real(value, name=name)


_EPS = 2.0**-52
