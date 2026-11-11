"""Sobol indices for stochastic simulators from replicated evaluations.

A stochastic simulator returns ``Y = g(X, omega)``: repeated runs at the same
input ``x`` differ because of intrinsic randomness ``omega`` (a random seed,
an unresolved microstructure). The law of total variance splits the output
variance into the variance of the mean response ``m(x) = E[Y | X = x]`` and
the mean intrinsic variance ``E[Var(Y | X)]``:

``Var(Y) = Var(m(X)) + E[Var(Y | X)]``.

This module evaluates the simulator ``R >= 2`` times, with independent
seeds, at every point of a pick-freeze design (matrices ``A``, ``B``, and
``A_B^(i)``) and reports

* Sobol indices of the mean response, ``S_i = V_i / Var(m)`` and
  ``ST_i = T_i / Var(m)``;
* the shares of the total output variance, ``V_i / Var(Y)``, which equal the
  first-order indices of ``Y`` with ``omega`` treated as an extra input; and
* the intrinsic-noise index ``S_omega = E[Var(Y | X)] / Var(Y)``, the total
  effect of the uncontrollable randomness.

Replication removes the noise bias exactly: independent noise cancels in the
pick-freeze product for ``V_i``, and the within-point sample variances
estimate (without bias) the noise term that inflates the Jansen total-effect
estimator and the variance of the replicate means.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)

StochasticModel = Callable[[tuple[float, ...], int], float]


@dataclass(frozen=True, slots=True)
class StochasticSobolResult:
    """Variance decomposition of a stochastic simulator.

    Args:
        first_order: First-order indices of the mean response ``m``.
        total_effect: Total-effect indices of the mean response ``m``.
        first_order_output: ``V_i / Var(Y)``, first-order shares of the total
            output variance.
        noise_index: ``E[Var(Y | X)] / Var(Y)``, the intrinsic-noise share.
        mean_variance: Estimated ``Var(m(X))``.
        noise_variance: Estimated ``E[Var(Y | X)]``.
        output_variance: ``mean_variance + noise_variance``.
        sample_size: Rows ``N`` of each base matrix.
        replications: Runs ``R`` per design point.
        evaluations: Simulator calls, ``N (d + 2) R``.

    References:
        [EQ-SENS-STOCH-001], [EQ-SENS-STOCH-002], [EQ-SENS-STOCH-003],
        [REF-IOOSS-RIBATET-MARREL-2009].

    """

    first_order: tuple[float, ...]
    total_effect: tuple[float, ...]
    first_order_output: tuple[float, ...]
    noise_index: float
    mean_variance: float
    noise_variance: float
    output_variance: float
    sample_size: int
    replications: int
    evaluations: int

    @property
    def unexplained_mean_variance(self) -> float:
        """Return ``1 - sum(first_order)``, the interaction share of ``Var(m)``.

        References:
            [EQ-SENS-ANOVA-001], [REF-SOBOL-1993].

        """
        return 1.0 - math.fsum(self.first_order)


def _design(
    rows: Iterable[Sequence[object]], name: str
) -> tuple[tuple[float, ...], ...]:
    """Validate a non-empty rectangular finite input matrix.

    References:
        [REF-SALTELLI-2010]. No equation is evaluated.

    """
    matrix = tuple(
        tuple(
            require_finite_real(value, name=f"{name}[{i}][{j}]")
            for j, value in enumerate(row)
        )
        for i, row in enumerate(rows)
    )
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError(f"{name} must be a non-empty rectangular matrix")
    return matrix


def _replicate(
    model: StochasticModel,
    point: tuple[float, ...],
    replications: int,
    stream: RandomStream,
) -> tuple[float, float]:
    """Return the mean and unbiased sample variance of ``R`` independent runs.

    Args:
        model: Stochastic simulator ``model(x, seed)``.
        point: Input vector.
        replications: Number ``R >= 2`` of runs.
        stream: Stream for this design point; run ``r`` gets ``child(r)``.

    Returns:
        ``(mean, variance)`` of the runs.

    Raises:
        ValueError: If an output is not finite.

    Equation:
        [EQ-SENS-STOCH-002].

    References:
        [REF-IOOSS-RIBATET-MARREL-2009, Section 3.2].

    """
    outputs = [
        require_finite_real(model(point, stream.child(r).seed), name="model output")
        for r in range(replications)
    ]
    return statistics.fmean(outputs), statistics.variance(outputs)


def stochastic_sobol_indices(
    model: StochasticModel,
    sample_a: Iterable[Sequence[object]],
    sample_b: Iterable[Sequence[object]],
    *,
    replications: int = 4,
    seed: int | RandomStream | None = None,
) -> StochasticSobolResult:
    """Estimate Sobol indices of a stochastic simulator by replicated pick-freeze.

    With replicate means ``ybar`` and within-point variances ``s^2`` at each
    design point, ``Var(m) = var(ybar over A and B) - mean(s^2) / R``,
    ``V_i = mean(ybar_B (ybar_ABi - ybar_A))`` (``ybar_B`` centred), and
    ``T_i = mean((ybar_A - ybar_ABi)^2 - (s^2_A + s^2_ABi) / R) / 2``; the
    noise variance is the mean of ``s^2`` over all design points.

    Args:
        model: Simulator ``model(x, seed) -> float``; ``x`` is a tuple of
            inputs and ``seed`` a non-negative integer that must fully
            determine the intrinsic randomness of that run.
        sample_a: Independent base sample of mutually independent inputs.
        sample_b: Equally shaped independent base sample.
        replications: Runs ``R >= 2`` per design point.
        seed: Integer seed or stream for the run seeds; ``None`` is not
            reproducible.

    Returns:
        Indices of the mean response, output-variance shares, the noise
        index, the variance components, and the evaluation cost.

    Raises:
        TypeError: If ``model`` is not callable or inputs are not real.
        ValueError: If the matrices are malformed or unequal, fewer than two
            rows or replications are given, an output is not finite, or the
            estimated mean-response variance is not positive.

    Equation:
        [EQ-SENS-STOCH-001], [EQ-SENS-STOCH-002], [EQ-SENS-STOCH-003],
        [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001].

    References:
        [REF-IOOSS-RIBATET-MARREL-2009, Equations 15, 20, and 21],
        [REF-MARREL-ETAL-2012], [REF-SALTELLI-2010], [REF-JANSEN-1999].

    """
    if not callable(model):
        raise TypeError("model must be callable")
    a = _design(sample_a, "sample_a")
    b = _design(sample_b, "sample_b")
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("sample matrices must have equal shape")
    if len(a) < 2:
        raise ValueError("at least two rows are required")
    runs = require_non_negative_integer(replications, name="replications")
    if runs < 2:
        raise ValueError("replications must be at least 2")
    n, dimension = len(a), len(a[0])
    stream = resolve_stream(seed).child("stochastic-sobol")
    mean_a, var_a = zip(
        *(
            _replicate(model, row, runs, stream.child("A", k))
            for k, row in enumerate(a)
        ),
        strict=True,
    )
    mean_b, var_b = zip(
        *(
            _replicate(model, row, runs, stream.child("B", k))
            for k, row in enumerate(b)
        ),
        strict=True,
    )
    noise_terms = list(var_a) + list(var_b)
    centre = statistics.fmean(mean_a + mean_b)
    first_numerators: list[float] = []
    total_numerators: list[float] = []
    for i in range(dimension):
        mean_ab, var_ab = zip(
            *(
                _replicate(
                    model,
                    a_row[:i] + (b_row[i],) + a_row[i + 1 :],
                    runs,
                    stream.child("AB", i, k),
                )
                for k, (a_row, b_row) in enumerate(zip(a, b, strict=True))
            ),
            strict=True,
        )
        noise_terms.extend(var_ab)
        first_numerators.append(
            math.fsum(
                (mean_b[k] - centre) * (mean_ab[k] - mean_a[k]) for k in range(n)
            )
            / n
        )
        total_numerators.append(
            0.5
            * math.fsum(
                (mean_a[k] - mean_ab[k]) ** 2 - (var_a[k] + var_ab[k]) / runs
                for k in range(n)
            )
            / n
        )
    noise = statistics.fmean(noise_terms)
    mean_variance = statistics.variance(mean_a + mean_b) - noise / runs
    if not mean_variance > 0.0:
        raise ValueError(
            "the estimated variance of the mean response is not positive; "
            "increase the sample size or the replications"
        )
    output_variance = mean_variance + noise
    return StochasticSobolResult(
        first_order=tuple(v / mean_variance for v in first_numerators),
        total_effect=tuple(t / mean_variance for t in total_numerators),
        first_order_output=tuple(v / output_variance for v in first_numerators),
        noise_index=noise / output_variance,
        mean_variance=mean_variance,
        noise_variance=noise,
        output_variance=output_variance,
        sample_size=n,
        replications=runs,
        evaluations=n * (dimension + 2) * runs,
    )
