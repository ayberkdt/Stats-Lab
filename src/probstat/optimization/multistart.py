"""Seeded multi-start minimization over a box of starting points."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.linalg import Vector, as_vector

from ._common import ScalarFunction
from .bounds import Bounds, BoxTransform
from .nelder_mead import minimize_nelder_mead
from .quasi_newton import minimize_bfgs
from .results import MultiStartResult, OptimizationResult


def multi_start(
    function: ScalarFunction,
    bounds: Bounds,
    *,
    starts: int = 10,
    seed: int | RandomStream | None = None,
    method: str = "bfgs",
    x0: Sequence[float] | None = None,
    gradient: Callable[[Vector], Sequence[float]] | None = None,
    bounded: bool = True,
    **options: Any,
) -> MultiStartResult:
    """Run a local minimizer from several seeded random starts in a box.

    Starts are drawn independently and uniformly in the finite box from the
    stream ``resolve_stream(seed).child("multi-start")``; ``x0``, if given,
    is the first start. Each run is a :func:`minimize_bfgs` or
    :func:`minimize_nelder_mead` call (with the same box as ``bounds`` when
    ``bounded`` is true). A run that raises ``ValueError`` (for example a
    non-finite objective at its start) is recorded as a failed run.

    Args:
        function: Scalar objective.
        bounds: Finite ``(low, high)`` per coordinate.
        starts: Number of starts (at least one).
        seed: Integer seed, stream, or ``None`` for fresh entropy.
        method: ``"bfgs"`` or ``"nelder-mead"``.
        x0: Optional deterministic first start.
        gradient: Optional analytic gradient (BFGS only).
        bounded: Constrain every local run to the box.
        **options: Further keyword arguments for the local minimizer.

    Returns:
        Multi-start result with every run and the best one.

    Raises:
        ValueError: If a bound is infinite, ``starts < 1``, the method is
            unknown, or every run failed.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 2], [REF-STEELE-LEA-FLOOD-2014].

    """
    if isinstance(starts, bool) or not isinstance(starts, int) or starts < 1:
        raise ValueError("starts must be a positive integer")
    if method not in ("bfgs", "nelder-mead"):
        raise ValueError("method must be 'bfgs' or 'nelder-mead'")
    pairs = tuple(bounds)
    size = len(pairs)
    transform = BoxTransform.from_bounds(pairs, size)
    if not all(
        math.isfinite(low) and math.isfinite(high)
        for low, high in zip(transform.lower, transform.upper, strict=True)
    ):
        raise ValueError("multi_start needs finite bounds to draw starts")
    stream = resolve_stream(seed)
    generator = stream.child("multi-start").generator()
    points: list[Vector] = []
    if x0 is not None:
        first = as_vector(x0, name="x0")
        if len(first) != size:
            raise ValueError("x0 length must match the bounds")
        points.append(first)
    while len(points) < starts:
        points.append(
            tuple(
                low + (high - low) * generator.random()
                for low, high in zip(transform.lower, transform.upper, strict=True)
            )
        )
    runs: list[OptimizationResult] = []
    for point in points:
        try:
            if method == "bfgs":
                run = minimize_bfgs(
                    function,
                    point,
                    gradient=gradient,
                    bounds=pairs if bounded else None,
                    **options,
                )
            else:
                run = minimize_nelder_mead(
                    function, point, bounds=pairs if bounded else None, **options
                )
        except ValueError as error:
            run = OptimizationResult(
                point, math.inf, math.nan, 0, 0, False, f"run failed: {error}"
            )
        runs.append(run)
    finite = [run for run in runs if math.isfinite(run.value)]
    if not finite:
        raise ValueError("every multi-start run failed")
    converged = [run for run in finite if run.converged]
    pool = converged if converged else finite
    best = min(pool, key=lambda run: run.value)
    return MultiStartResult(best, tuple(runs), tuple(points), stream.seed)
