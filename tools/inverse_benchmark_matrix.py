"""Run the inverse benchmark method-by-case matrix and write machine-readable rows.

Every (case, method, seed) cell is run once with predeclared settings; the
truth is used only to score the result. Failed runs are kept as rows with
their error message. The output is a JSON document with a schema version, the
environment, raw rows and per-cell medians::

    PYTHONPATH=src python tools/inverse_benchmark_matrix.py --quick --output out.json

``--quick`` uses seeds 11, 23, 37 and small sizes (a few minutes in pure
Python); the default uses ten seeds. Wall times are measured and reported,
never asserted.

References:
    [REF-GAZZOLA-HANSEN-NAGY-2019], [REF-KAIPIO-SOMERSALO-2007],
    [EQ-INV-BENCH-001].

"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from probstat.inverse import (
    InverseBenchmark,
    LinearInverseProblem,
    correlated_gaussian_benchmark,
    deconvolution_benchmark,
    diagonal_spectrum_benchmark,
    expected_noise_norm,
    krylov_regularization,
    l1_regularized_least_squares,
    linear_gaussian_posterior,
    logarithmic_grid,
    regularization_path,
    select_regularization,
    sparse_sensing_benchmark,
    total_variation_regularized_least_squares,
)
from probstat.linalg import FirstDifferenceOperator

SCHEMA_VERSION = "inverse-benchmark-matrix/1"
QUICK_SEEDS = (11, 23, 37)
FULL_SEEDS = (11, 23, 37, 41, 53, 67, 71, 83, 97, 101)


def _relative_error(estimate: tuple[float, ...], truth: tuple[float, ...]) -> float:
    """Return ``||x - x_true|| / ||x_true||``.

    References:
        [EQ-INV-BENCH-001].

    """
    denominator = math.hypot(*truth)
    error = math.hypot(*(a - b for a, b in zip(estimate, truth, strict=True)))
    return error / denominator if denominator > 0.0 else error


def _problem(case: InverseBenchmark) -> LinearInverseProblem:
    """Return the solver view of a benchmark (no truth).

    References:
        [EQ-INV-LIN-001].

    """
    return LinearInverseProblem.create(case.operator, case.data, noise=case.noise)


def _krylov(case: InverseBenchmark) -> dict[str, Any]:
    """Early-stopped LSQR with the discrepancy principle.

    References:
        [EQ-INV-ITER-002].

    """
    m = case.operator.shape[0]
    result = krylov_regularization(_problem(case), noise_norm=expected_noise_norm(m))
    return {
        "estimate": result.estimate,
        "converged": result.converged,
        "stop_reason": result.stop_reason,
        "forward": result.counts.forward,
        "transpose": result.counts.transpose,
    }


def _path_rule(rule: str) -> Callable[[InverseBenchmark], dict[str, Any]]:
    """Return a runner that selects ``lambda`` by one rule on a fixed grid.

    References:
        [EQ-INV-SEL-001].

    """

    def run(case: InverseBenchmark) -> dict[str, Any]:
        """Run the path and select by ``rule``.

        References:
            [EQ-INV-SEL-001].

        """
        m = case.operator.shape[0]
        path = regularization_path(
            _problem(case),
            logarithmic_grid(1e-3, 1e3, 19),
            noise_norm=expected_noise_norm(m),
        )
        selection = select_regularization(path, rule)
        if selection.estimate is None:
            raise RuntimeError(f"no {rule} choice ({selection.status})")
        return {
            "estimate": selection.estimate,
            "converged": selection.status == "interior",
            "stop_reason": selection.status,
            "regularization": selection.regularization,
        }

    return run


def _lasso(case: InverseBenchmark) -> dict[str, Any]:
    """FISTA lasso with a fixed weight relative to the zero threshold.

    References:
        [EQ-INV-SPARSE-001].

    """
    result = l1_regularized_least_squares(
        case.operator, case.data, 0.05, noise=case.noise, max_iterations=3000
    )
    return {
        "estimate": result.solution,
        "converged": result.converged,
        "stop_reason": result.stop_reason,
    }


def _total_variation(case: InverseBenchmark) -> dict[str, Any]:
    """PDHG total variation with a fixed weight.

    References:
        [EQ-INV-SPARSE-001].

    """
    n = case.operator.shape[1]
    result = total_variation_regularized_least_squares(
        case.operator,
        case.data,
        0.5,
        noise=case.noise,
        gradient=FirstDifferenceOperator(n, spacing=1.0 / n),
        max_iterations=5000,
    )
    return {
        "estimate": result.solution,
        "converged": result.converged,
        "stop_reason": result.stop_reason,
    }


def _posterior(case: InverseBenchmark) -> dict[str, Any]:
    """Exact Gaussian posterior mean (cases with a prior only).

    References:
        [EQ-INV-BENCH-001], [REF-STUART-2010].

    """
    if case.prior_covariance is None:
        raise RuntimeError("case has no prior")
    posterior = linear_gaussian_posterior(
        case.operator,
        case.data,
        noise=case.noise,
        prior_covariance=case.prior_covariance,
    )
    return {"estimate": posterior.mean, "converged": True, "stop_reason": "exact"}


METHODS: dict[str, Callable[[InverseBenchmark], dict[str, Any]]] = {
    "lsqr_discrepancy": _krylov,
    "tikhonov_gcv": _path_rule("gcv"),
    "tikhonov_upre": _path_rule("upre"),
    "tikhonov_discrepancy": _path_rule("discrepancy"),
    "lasso_fista": _lasso,
    "tv_pdhg": _total_variation,
    "gaussian_posterior": _posterior,
}

CASES: dict[str, Callable[[int, bool], InverseBenchmark]] = {
    "L01": lambda seed, quick: diagonal_spectrum_benchmark(
        24 if quick else 64, seed=seed
    ),
    "L04": lambda seed, quick: deconvolution_benchmark(
        32 if quick else 96, refinement=2, seed=seed
    ),
    "L05": lambda seed, quick: sparse_sensing_benchmark(20, 40, sparsity=3, seed=seed),
    "L06": lambda seed, quick: correlated_gaussian_benchmark(16, 8, seed=seed),
}

COMPATIBLE = {
    "L01": (
        "lsqr_discrepancy",
        "tikhonov_gcv",
        "tikhonov_upre",
        "tikhonov_discrepancy",
    ),
    "L04": ("lsqr_discrepancy", "tikhonov_gcv", "tikhonov_upre", "tv_pdhg"),
    "L05": ("lasso_fista", "lsqr_discrepancy"),
    "L06": ("gaussian_posterior", "tikhonov_gcv", "lsqr_discrepancy"),
}


def run_matrix(seeds: tuple[int, ...], quick: bool) -> dict[str, Any]:
    """Run every compatible (case, method, seed) cell.

    Args:
        seeds: Root seeds of the benchmark instances.
        quick: Small sizes when ``True``.

    Returns:
        JSON-ready document with raw rows and per-cell medians.

    References:
        [EQ-INV-BENCH-001].

    """
    rows: list[dict[str, Any]] = []
    for case_id, builder in CASES.items():
        for method in COMPATIBLE[case_id]:
            for seed in seeds:
                case = builder(seed, quick)
                start = time.perf_counter()
                row: dict[str, Any] = {
                    "case": case_id,
                    "method": method,
                    "seed": seed,
                    "same_discretization": case.same_discretization,
                }
                try:
                    outcome = METHODS[method](case)
                    row.update(
                        status="ok",
                        relative_error=_relative_error(
                            tuple(outcome["estimate"]), case.truth
                        ),
                        converged=outcome["converged"],
                        stop_reason=outcome["stop_reason"],
                    )
                    for key in ("forward", "transpose", "regularization"):
                        if key in outcome:
                            row[key] = outcome[key]
                except Exception as error:  # noqa: BLE001 - failures are data
                    row.update(
                        status="failed", error=f"{type(error).__name__}: {error}"
                    )
                row["wall_seconds"] = time.perf_counter() - start
                rows.append(row)
    summary = []
    for case_id in CASES:
        for method in COMPATIBLE[case_id]:
            errors = [
                r["relative_error"]
                for r in rows
                if r["case"] == case_id
                and r["method"] == method
                and r["status"] == "ok"
            ]
            failed = sum(
                1
                for r in rows
                if r["case"] == case_id
                and r["method"] == method
                and r["status"] != "ok"
            )
            entry: dict[str, Any] = {
                "case": case_id,
                "method": method,
                "runs": len(errors),
                "failed": failed,
            }
            if errors:
                quartiles = (
                    statistics.quantiles(errors, n=4)
                    if len(errors) > 1
                    else [errors[0]] * 3
                )
                entry.update(
                    median_relative_error=statistics.median(errors),
                    q1=quartiles[0],
                    q3=quartiles[2],
                )
            summary.append(entry)
    return {
        "schema": SCHEMA_VERSION,
        "mode": "quick" if quick else "full",
        "seeds": list(seeds),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "rows": rows,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the matrix and write JSON.

    References:
        [EQ-INV-BENCH-001].

    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="three seeds, small sizes")
    parser.add_argument("--output", type=Path, default=None, help="JSON output path")
    arguments = parser.parse_args(argv)
    document = run_matrix(
        QUICK_SEEDS if arguments.quick else FULL_SEEDS, arguments.quick
    )
    text = json.dumps(document, indent=2)
    if arguments.output is None:
        print(text)
    else:
        arguments.output.write_text(text + "\n", encoding="utf-8")
        for entry in document["summary"]:
            median = entry.get("median_relative_error")
            shown = f"{median:.3g}" if median is not None else "-"
            print(
                f"{entry['case']} {entry['method']:<22} runs={entry['runs']} "
                f"failed={entry['failed']} median_rel_error={shown}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
