"""Seeded execution, timing, failure capture, and aggregation.

For every (problem, applicable method, seed) the harness builds a fresh
instance, times the adapter with :func:`time.perf_counter`, reads the cost from
the model's evaluation counter (so a method cannot under-report it), records
warnings, and turns any exception into a ``"failed"`` record. Records are flat
dictionaries so the JSON file is easy to analyze elsewhere.
"""

from __future__ import annotations

import math
import os
import statistics
import time
import traceback
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from benchmarks.methods import EXPECTED_FAILURES, Estimate, Method, MethodOutput
from benchmarks.problems import Problem, quantity_group

#: A reported standard error below this fraction of |reference| (or below
#: ``SE_FLOOR_ABSOLUTE``) is floored there when forming z: differences of
#: that size are at the level of the library's solver tolerances (FORM and
#: line-sampling root finding use 1e-6), not sampling error.
SE_FLOOR_RELATIVE = 1e-6
SE_FLOOR_ABSOLUTE = 1e-15
#: Two-sided 95 % normal quantile for coverage of ``estimate +/- z SE``.
Z95 = 1.959963984540054

Record = dict[str, Any]
Progress = Callable[[str], None]


def seeds_for(problem: Problem, method: Method, count: int) -> tuple[int, ...]:
    """Return the seeds a method runs on (one for deterministic methods)."""
    return (1,) if method.deterministic else tuple(range(1, count + 1))


def _score(
    quantity: str,
    estimate: Estimate,
    reference: float,
    scale: float | None,
) -> Record:
    """Return error, z, and coverage fields of one estimate."""
    error = estimate.value - reference
    record: Record = {
        "quantity": quantity,
        "group": quantity_group(quantity),
        "estimate": estimate.value,
        "reference": reference,
        "abs_error": abs(error),
        "rel_error": abs(error) / abs(reference) if reference != 0.0 else None,
        "scaled_error": abs(error) / scale if scale else None,
        "standard_error": estimate.standard_error,
        "z": None,
        "se_floored": None,
        "interval": list(estimate.interval) if estimate.interval else None,
        "interval_level": estimate.level,
        "covered": None,
    }
    se = estimate.standard_error
    if se is not None and math.isfinite(se):
        floor = max(SE_FLOOR_RELATIVE * abs(reference), SE_FLOOR_ABSOLUTE)
        record["z"] = error / max(se, floor)
        record["se_floored"] = se < floor
        record["covered"] = abs(record["z"]) <= Z95
        record["interval_level"] = 0.95
    if estimate.interval is not None:
        lower, upper = estimate.interval
        if math.isfinite(lower) and math.isfinite(upper):
            record["covered"] = lower <= reference <= upper
    return record


def run_one(
    problem: Problem, method: Method, seed: int, sizes: Mapping[str, Any]
) -> list[Record]:
    """Run one method on one problem for one seed and return its records."""
    base: Record = {
        "problem": problem.key,
        "category": problem.category,
        "method": method.name,
        "kind": method.kind,
        "seed": seed,
        "status": "ok",
        "error": None,
        "expected_failure": None,
        "warnings": [],
        "notes": [],
        "evaluations": 0,
        "wall_time_s": 0.0,
    }
    reason = method.skip(problem, sizes)
    if reason is not None:
        return [dict(base, status="skipped", notes=[reason], quantity=None)]
    instance = problem.build(seed)
    before = instance.model.evaluations
    start = time.perf_counter()
    output: MethodOutput | None = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            output = method.run(instance, seed, sizes, problem)
        except Exception as error:  # noqa: BLE001 - every failure is recorded
            frame = traceback.extract_tb(error.__traceback__)[-1]
            base["status"] = "failed"
            base["error"] = (
                f"{type(error).__name__}: {error} "
                f"({os.path.basename(frame.filename)}:{frame.lineno})"
            )
    known = EXPECTED_FAILURES.get((problem.key, method.name))
    if base["error"] and known and known[0] in base["error"]:
        base["expected_failure"] = known[1]
    base["wall_time_s"] = time.perf_counter() - start
    base["evaluations"] = instance.model.evaluations - before
    base["warnings"] = sorted({str(item.message).splitlines()[0] for item in caught})
    if output is None:
        return [dict(base, quantity=None)]
    base["notes"] = list(output.notes)
    references = problem.references(seed)
    records: list[Record] = []
    for quantity, estimate in output.estimates.items():
        if quantity not in references:
            continue
        target = references[quantity]
        scored = _score(quantity, estimate, target.value, target.scale)
        records.append({**base, **scored})
    if not records:
        records.append(
            dict(
                base,
                status="failed",
                error="no estimate for any reference quantity",
                quantity=None,
            )
        )
    return records


def run_benchmarks(
    problems: Sequence[Problem],
    methods: Sequence[Method],
    sizes: Mapping[str, Any],
    *,
    seeds: Mapping[str, int] | None = None,
    progress: Progress | None = None,
) -> list[Record]:
    """Run every applicable method on every problem over the configured seeds.

    Args:
        problems: Problems to run.
        methods: Candidate methods; each runs on problems of its category for
            which its ``applies`` rule holds.
        sizes: Size table of the current mode.
        seeds: Optional seed count per category overriding ``sizes["seeds"]``.
        progress: Optional callback receiving one line per finished problem.

    Returns:
        Flat records, one per (problem, method, seed, quantity), plus one per
        failed or skipped run.

    """
    counts = dict(sizes["seeds"])
    if seeds:
        counts.update(seeds)
    records: list[Record] = []
    for position, problem in enumerate(problems, start=1):
        start = time.perf_counter()
        chosen = [
            m for m in methods if m.category == problem.category and m.applies(problem)
        ]
        failed = 0
        for method in chosen:
            for seed in seeds_for(problem, method, counts[problem.category]):
                produced = run_one(problem, method, seed, sizes)
                failed += any(item["status"] == "failed" for item in produced)
                records.extend(produced)
        if progress is not None:
            progress(
                f"[{position}/{len(problems)} {problem.category}] {problem.key}: "
                f"{len(chosen)} methods, "
                f"{counts[problem.category]} seeds, {failed} failed run(s), "
                f"{time.perf_counter() - start:.1f} s"
            )
    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _median(values: Iterable[float]) -> float | None:
    """Return the median of finite values, or ``None`` if there are none."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(finite) if finite else None


def aggregate(records: Sequence[Record], problems: Sequence[Problem]) -> list[Record]:
    """Summarize records per (problem, method, quantity group).

    Error metric per run is the worst quantity of the group (relative,
    absolute, or scaled, per the problem's ``error_kinds``); the summary
    reports its median over seeds. ``|z|`` statistics pool all quantities and
    seeds; coverage counts how often the reference lies in the method's own
    reported interval (or ``estimate +/- 1.96 SE``).
    """
    kinds = {problem.key: problem.error_kinds for problem in problems}
    order: list[tuple[str, str]] = []
    runs: dict[tuple[str, str], dict[int, list[Record]]] = {}
    for record in records:
        key = (record["problem"], record["method"])
        if key not in runs:
            runs[key] = {}
            order.append(key)
        runs[key].setdefault(record["seed"], []).append(record)
    summary: list[Record] = []
    for key in order:
        by_seed = runs[key]
        first = next(iter(by_seed.values()))[0]
        statuses = {seed: _run_status(items) for seed, items in by_seed.items()}
        ok_seeds = [seed for seed, status in statuses.items() if status == "ok"]
        common = {
            "problem": key[0],
            "category": first["category"],
            "method": key[1],
            "kind": first["kind"],
            "runs": len(by_seed),
            "ok": len(ok_seeds),
            "failed": sum(1 for s in statuses.values() if s == "failed"),
            "skipped": sum(1 for s in statuses.values() if s == "skipped"),
            "evaluations": _mean(by_seed[s][0]["evaluations"] for s in ok_seeds),
            "wall_time_s": _mean(by_seed[s][0]["wall_time_s"] for s in ok_seeds),
            "notes": _unique(
                n for items in by_seed.values() for n in items[0]["notes"]
            ),
            "warnings": _unique(
                w for items in by_seed.values() for w in items[0]["warnings"]
            ),
            "errors": _unique(
                items[0]["error"] for items in by_seed.values() if items[0]["error"]
            ),
            "expected_failure": next(
                (
                    items[0]["expected_failure"]
                    for items in by_seed.values()
                    if items[0].get("expected_failure")
                ),
                None,
            ),
        }
        groups: dict[str, list[Record]] = {}
        for seed in ok_seeds:
            for item in by_seed[seed]:
                if item.get("group") is not None:
                    groups.setdefault(item["group"], []).append(item)
        if not groups:
            summary.append({**common, "group": None})
            continue
        for group, items in groups.items():
            kind = kinds.get(key[0], {}).get(group, "relative")
            summary.append({**common, "group": group, **_group_stats(items, kind)})
    return summary


def _run_status(items: Sequence[Record]) -> str:
    """Return the status of one run from its records."""
    statuses = {item["status"] for item in items}
    for status in ("failed", "skipped"):
        if status in statuses:
            return status
    return "ok"


def _mean(values: Iterable[float]) -> float | None:
    """Return the arithmetic mean, or ``None`` for no values."""
    data = list(values)
    return math.fsum(data) / len(data) if data else None


def _unique(values: Iterable[str]) -> list[str]:
    """Return distinct values in first-seen order."""
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)


def _group_stats(items: Sequence[Record], kind: str) -> Record:
    """Return error, z, coverage, and estimate statistics of one group."""
    field = {
        "relative": "rel_error",
        "absolute": "abs_error",
        "scaled": "scaled_error",
    }[kind]
    per_seed: dict[int, list[float]] = {}
    estimates: dict[str, list[float]] = {}
    ratios: list[float] = []
    by_quantity: dict[str, list[float]] = {}
    for item in items:
        value = item.get(field)
        if value is None:
            value = item["abs_error"]
        per_seed.setdefault(item["seed"], []).append(value)
        estimates.setdefault(item["quantity"], []).append(item["estimate"])
        if item["reference"]:
            ratios.append(item["estimate"] / item["reference"])
            by_quantity.setdefault(item["quantity"], []).append(
                item["estimate"] / item["reference"]
            )
    zs = [abs(item["z"]) for item in items if item["z"] is not None]
    covered = [item["covered"] for item in items if item["covered"] is not None]
    levels = {item["interval_level"] for item in items if item["covered"] is not None}
    return {
        "error_kind": kind,
        "error": _median(max(values) for values in per_seed.values()),
        "estimates": {q: _mean(v) for q, v in estimates.items()},
        "references": {item["quantity"]: item["reference"] for item in items},
        "ratio_mean": _mean(ratios),
        "ratios": {q: _mean(v) for q, v in by_quantity.items()},
        "z_median": _median(zs),
        "z_max": max(zs) if zs else None,
        "z_count": len(zs),
        "z_floored": sum(1 for item in items if item.get("se_floored")),
        "covered": sum(covered),
        "coverage_n": len(covered),
        "interval_level": sorted(levels, key=lambda v: -1.0 if v is None else v)[-1]
        if levels
        else None,
    }
