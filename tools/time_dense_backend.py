"""Time the reference and NumPy dense backends on SPD kernels and GP likelihoods.

Usage::

    PYTHONPATH=src python tools/time_dense_backend.py [--sizes 100 200 400]

For each size ``n`` the script reports the best of three wall times (seconds,
``time.perf_counter``) for

* ``spd``: ``factorize`` + ``solve`` + ``log_det`` of a random SPD matrix
  through the backend API, tuple conversions included;
* ``gp``: one Gaussian-process log marginal likelihood with its gradient
  (Matern 5/2, two inputs), the evaluation that BFGS repeats during a fit;
* ``mvn``: constructing a multivariate normal and evaluating one log-density.

The numbers recorded in ADR-0081 come from this script. Requires NumPy.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from collections.abc import Callable

from probstat.linalg import (
    Matrix,
    NumPyBackend,
    PythonBackend,
    use_backend,
)
from probstat.multivariate import MultivariateNormal
from probstat.surrogates import gaussian_process_log_marginal_likelihood


def best_time(task: Callable[[], object], repeats: int = 3) -> float:
    """Return the smallest wall time of ``repeats`` runs of ``task``.

    References:
        [REF-PYTHON-MATH]. No equation is evaluated.

    """
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        task()
        times.append(time.perf_counter() - start)
    return min(times)


def spd_matrix(size: int, seed: int) -> Matrix:
    """Return ``B B' / n + 0.1 I`` for a Gaussian ``B`` (built with NumPy).

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

    """
    generator = random.Random(seed)
    factor = tuple(
        tuple(generator.gauss(0.0, 1.0) for _ in range(size)) for _ in range(size)
    )
    product = NumPyBackend().matmul(factor, tuple(zip(*factor, strict=True)))
    return tuple(
        tuple(
            0.5 * (product[i][j] + product[j][i]) / size + (0.1 if i == j else 0.0)
            for j in range(size)
        )
        for i in range(size)
    )


def measure(
    backend: PythonBackend | NumPyBackend,
    matrix: Matrix,
    rhs: tuple[float, ...],
    points: list[tuple[float, float]],
    values: list[float],
) -> dict[str, float]:
    """Return the best wall times of the three tasks under one backend.

    References:
        [EQ-LINALG-001], [EQ-SUR-GP-004], [EQ-DIST-MVN-001].

    """
    theta = (0.0, math.log(0.3), math.log(0.3))
    mean = (0.0,) * len(matrix)

    def spd() -> object:
        """Factorize, solve, and take the log-determinant.

        References:
            [EQ-LINALG-001], [EQ-LINALG-006].

        """
        factor = backend.factorize(matrix)
        return factor.solve(rhs), factor.log_det()

    def gp() -> object:
        """Evaluate the GP likelihood and gradient once.

        References:
            [EQ-SUR-GP-004].

        """
        return gaussian_process_log_marginal_likelihood(
            points, values, theta, nugget=1e-6
        )

    def mvn() -> object:
        """Build a multivariate normal and evaluate one log-density.

        References:
            [EQ-DIST-MVN-001].

        """
        return MultivariateNormal(mean, matrix).log_pdf(rhs)

    with use_backend(backend):
        return {"spd": best_time(spd), "gp": best_time(gp), "mvn": best_time(mvn)}


def main() -> None:
    """Print a Markdown table of reference and NumPy wall times.

    References:
        [REF-HARRIS-ETAL-2020], [EQ-LINALG-001], [EQ-SUR-GP-004].

    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 400])
    arguments = parser.parse_args()
    print("| n | task | python (s) | numpy (s) | speed-up |")
    print("|---|---|---|---|---|")
    for size in arguments.sizes:
        matrix = spd_matrix(size, seed=size)
        rhs = tuple(random.Random(size + 1).gauss(0.0, 1.0) for _ in range(size))
        generator = random.Random(size + 2)
        points = [(generator.random(), generator.random()) for _ in range(size)]
        values = [math.sin(6.0 * a) + b * b for a, b in points]
        reference = measure(PythonBackend(), matrix, rhs, points, values)
        candidate = measure(NumPyBackend(), matrix, rhs, points, values)
        for task in ("spd", "gp", "mvn"):
            print(
                f"| {size} | {task} | {reference[task]:.4f} | {candidate[task]:.4f} | "
                f"{reference[task] / candidate[task]:.1f}x |"
            )


if __name__ == "__main__":
    main()
