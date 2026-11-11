"""Markdown report generated from aggregated benchmark results.

Every table has at most seven columns; per-method notes, warnings, and
failures are listed under the table. The closing section, "What the numbers
say", is computed from the summary with the thresholds it states, not
written by hand.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from benchmarks.harness import SE_FLOOR_RELATIVE
from benchmarks.problems import Problem
from probstat.numerics import (
    DataProfile,
    PerformanceProfile,
    data_profile,
    performance_profile,
)

Row = Mapping[str, Any]

#: Thresholds used by "What the numbers say".
ACCURATE_REL = 0.05
INACCURATE_REL = 0.10
UNDER_REPORT_Z = 2.0
OVER_CONSERVATIVE_Z = 0.25
SD_RATIO_BAND = (0.8, 1.25)
CATEGORY_TITLES = {
    "reliability": "Reliability: failure probability",
    "sensitivity": "Sensitivity indices",
    "propagation": "Propagation and integration",
    "calibration": "Bayesian calibration",
}
DASH = "-"
#: Error at or below which a method "solves" a problem instance in the
#: performance profiles: the *accurate* thresholds of "What the numbers say".
SOLVED_ERROR = {"relative": ACCURATE_REL, "absolute": 0.02, "scaled": 0.05}
#: Ratios at which the performance-profile table reports rho(tau).
PROFILE_TAUS = (4.0, 16.0)
#: Budgets (simplex gradients, n + 1 evaluations each) of the data profile.
DATA_BUDGETS = (10.0, 100.0, 1000.0, 10000.0)


def problem_meta(problem: Problem) -> dict[str, Any]:
    """Return the JSON description of a problem."""
    return {
        "key": problem.key,
        "category": problem.category,
        "title": problem.title,
        "description": problem.description,
        "reference_note": problem.reference_note,
        "dimension": problem.dimension,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def number(value: float | None) -> str:
    """Format a value with four significant digits (scientific when small)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return DASH
    if value == 0.0:
        return "0"
    if 1e-2 <= abs(value) < 1e4:
        return f"{value:.4g}"
    return f"{value:.3e}"


def percent(value: float | None) -> str:
    """Format a relative error as a percentage."""
    if value is None or math.isnan(value):
        return DASH
    if value * 100.0 >= 0.1:
        return f"{value * 100.0:.1f} %"
    if value == 0.0:
        return "0 %"
    return f"{value * 100.0:.0e} %"


def seconds(value: float | None) -> str:
    """Format a wall time."""
    if value is None:
        return DASH
    return f"{value:.2f} s" if value >= 0.01 else "< 0.01 s"


def count(value: float | None) -> str:
    """Format a mean evaluation count."""
    return DASH if value is None else f"{round(value):,}"


def _z(value: float) -> str:
    """Format one ``|z|`` value, capping absurd magnitudes."""
    return "> 1000" if value > 1000.0 else f"{value:.2f}"


def z_text(row: Row) -> str:
    """Format the median and maximum ``|z|``."""
    if not row.get("z_count"):
        return DASH
    return f"{_z(row['z_median'])} (max {_z(row['z_max'])})"


def coverage_text(row: Row) -> str:
    """Format how often the reference lay in the reported interval."""
    total = row.get("coverage_n") or 0
    if not total:
        return DASH
    suffix = " bounds" if row.get("kind") == "bounds" else ""
    return f"{row['covered']}/{total}{suffix}"


def status_text(row: Row) -> str | None:
    """Return a failure/skip label for rows without estimates."""
    if row["ok"] == row["runs"]:
        return None
    if row["skipped"] == row["runs"]:
        return "skipped"
    if row["ok"] == 0:
        return f"failed ({row['failed']}/{row['runs']})"
    return None


def table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    """Return Markdown table lines."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def notes_block(rows: Iterable[Row]) -> list[str]:
    """Return bullet notes: failures, partial failures, warnings, remarks."""
    lines: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row["method"] in seen:
            continue
        seen.add(row["method"])
        parts: list[str] = []
        if row["failed"]:
            errors = "; ".join(_clip(e) for e in row["errors"][:2])
            label = (
                "FAILED (known limitation)" if row.get("expected_failure") else "FAILED"
            )
            parts.append(f"{label} in {row['failed']}/{row['runs']} run(s): {errors}")
        if row["skipped"]:
            parts.append(_clip(row["notes"][0]) if row["notes"] else "skipped")
        elif row["notes"]:
            shown = [_clip(n) for n in row["notes"][:2]]
            extra = len(row["notes"]) - len(shown)
            parts.append("; ".join(shown) + (f" (+{extra} more)" if extra > 0 else ""))
        if row["warnings"]:
            parts.append("warned: " + "; ".join(_clip(w) for w in row["warnings"][:2]))
        if parts:
            lines.append(f"- **{row['method']}**: " + " | ".join(parts))
    return lines


def _clip(text: str, limit: int = 150) -> str:
    """Shorten long messages and escape table-breaking characters."""
    text = " ".join(str(text).split()).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Tables per category
# ---------------------------------------------------------------------------

SCALAR_HEADER = (
    "Method",
    "Estimate (mean)",
    "Rel. error (median)",
    "Median abs z (max)",
    "Coverage",
    "Evals (mean)",
    "Time (mean)",
)


def scalar_tables(problem: Problem, rows: Sequence[Row]) -> list[str]:
    """Return one table per scalar quantity group of a problem."""
    lines: list[str] = []
    groups = [g for g in problem.error_kinds if any(r.get("group") == g for r in rows)]
    for group in groups:
        reference = next(r for r in rows if r.get("group") == group)["references"]
        value = next(iter(reference.values()))
        if len(groups) > 1:
            lines.append(f"**{group}** (reference {number(value)})")
            lines.append("")
        body = [["**Reference**", f"**{number(value)}**", "", "", "", "", ""]]
        for row in rows:
            label = status_text(row)
            if label is not None:
                if row.get("group") in (None, group) and row is _first(rows, row):
                    body.append([row["method"], label, DASH, DASH, DASH, DASH, DASH])
                continue
            if row.get("group") != group:
                continue
            estimate = next(iter(row["estimates"].values()))
            body.append(
                [
                    row["method"],
                    number(estimate),
                    percent(row["error"]),
                    z_text(row),
                    coverage_text(row),
                    count(row["evaluations"]),
                    seconds(row["wall_time_s"]),
                ]
            )
        lines.extend(table(SCALAR_HEADER, body))
        lines.append("")
    return lines


def _first(rows: Sequence[Row], row: Row) -> Row:
    """Return the first summary row of the same method (to avoid duplicates)."""
    return next(r for r in rows if r["method"] == row["method"])


def index_table(problem: Problem, rows: Sequence[Row]) -> list[str]:
    """Return the sensitivity-index table of a problem."""
    header = (
        "Method",
        "Index",
        "Estimates (mean over seeds)",
        "Max abs. error (median)",
        "Coverage",
        "Evals (mean)",
        "Time (mean)",
    )
    body: list[list[str]] = []
    for group in problem.error_kinds:
        sample = next((r for r in rows if r.get("group") == group), None)
        if sample is None:
            continue
        body.append(
            [
                "**Reference**",
                group,
                "**" + _vector(sample["references"]) + "**",
                "",
                "",
                "",
                "",
            ]
        )
    done: set[str] = set()
    for row in rows:
        label = status_text(row)
        if label is not None:
            if row["method"] not in done:
                body.append([row["method"], DASH, label, DASH, DASH, DASH, DASH])
                done.add(row["method"])
            continue
        body.append(
            [
                row["method"],
                row["group"],
                _vector(row["estimates"]),
                number(row["error"]),
                coverage_text(row),
                count(row["evaluations"]),
                seconds(row["wall_time_s"]),
            ]
        )
    return [*table(header, body), ""]


def _vector(values: Mapping[str, float]) -> str:
    """Format an index vector compactly."""
    return " ".join(f"{v:.3f}" for v in values.values())


def calibration_table(problem: Problem, rows: Sequence[Row]) -> list[str]:
    """Return the calibration table (one row per method)."""
    header = (
        "Method",
        "Mean error (post. sd, median)",
        "SD ratio a / b",
        "Median abs z vs MCSE (max)",
        "90 % coverage of truth",
        "Evals (mean)",
        "Time (mean)",
    )
    body: list[list[str]] = []
    for method in _unique(r["method"] for r in rows):
        group = {r.get("group"): r for r in rows if r["method"] == method}
        any_row = next(iter(group.values()))
        label = status_text(any_row)
        if label is not None:
            body.append([method, label, DASH, DASH, DASH, DASH, DASH])
            continue
        mean, sd, theta = group.get("mean"), group.get("sd"), group.get("theta")
        ratios = " / ".join(f"{v:.2f}" for v in sd["ratios"].values()) if sd else DASH
        body.append(
            [
                method,
                f"{mean['error']:.3f}" if mean else DASH,
                ratios,
                z_text(mean) if mean else DASH,
                coverage_text(theta) if theta else DASH,
                count(any_row["evaluations"]),
                seconds(any_row["wall_time_s"]),
            ]
        )
    return [*table(header, body), ""]


def _unique(values: Iterable[str]) -> list[str]:
    """Return distinct values in first-seen order."""
    return list(dict.fromkeys(values))


# ---------------------------------------------------------------------------
# Performance and data profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CategoryProfiles:
    """Performance and data profiles of the methods of one category.

    ``instances`` are the scored (problem, quantity group) pairs; a method
    solves an instance when its median error meets :data:`SOLVED_ERROR`, at
    a cost of its mean model evaluations. ``excluded`` lists methods that
    record no model evaluations, whose cost cannot enter a ratio.
    """

    category: str
    instances: tuple[str, ...]
    performance: PerformanceProfile
    data: DataProfile
    excluded: tuple[str, ...]


def _instances(problem: Problem) -> list[tuple[str, str]]:
    """Return the (group, error kind) pairs of a problem that are scored."""
    return [
        (group, kind)
        for group, kind in problem.error_kinds.items()
        if problem.category != "calibration" or group == "mean"
    ]


def category_profiles(
    problems: Sequence[Problem], summary: Sequence[Row], category: str
) -> CategoryProfiles | None:
    """Return the profiles of one category, or ``None`` with < 2 methods.

    Cost ``t_(p,s)``: mean model evaluations of method ``s`` on instance
    ``p`` when its median error is at most :data:`SOLVED_ERROR`; otherwise
    (inaccurate, failed, skipped, or not run) the instance counts as a
    failure with ratio infinity (Dolan and Moré 2002). The data profile
    divides the same counts by ``n_p + 1`` (Moré and Wild 2009).
    """
    chosen = [p for p in problems if p.category == category]
    rows = [
        r for r in summary if r["category"] == category and r["kind"] != "reference"
    ]
    methods = _unique(r["method"] for r in rows)
    evaluated = {r["method"] for r in rows if (r.get("evaluations") or 0) > 0}
    excluded = tuple(m for m in methods if m not in evaluated)
    methods = [m for m in methods if m in evaluated]
    labels: list[str] = []
    dimensions: list[int] = []
    costs: dict[str, list[float | None]] = {m: [] for m in methods}
    for problem in chosen:
        groups = _instances(problem)
        for group, kind in groups:
            scored = {
                r["method"]: r
                for r in rows
                if r["problem"] == problem.key and r.get("group") == group
            }
            if not scored:
                continue
            labels.append(problem.key if len(groups) == 1 else f"{problem.key} {group}")
            dimensions.append(problem.dimension)
            for method in methods:
                costs[method].append(_solved_cost(scored.get(method), kind))
    if len(methods) < 2 or not labels:
        return None
    return CategoryProfiles(
        category=category,
        instances=tuple(labels),
        performance=performance_profile(costs),
        data=data_profile(costs, dimensions),
        excluded=excluded,
    )


def _solved_cost(row: Row | None, kind: str) -> float | None:
    """Return a row's mean evaluations if its error solves the instance."""
    if row is None or row.get("error") is None or not row.get("evaluations"):
        return None
    if row["error"] > SOLVED_ERROR[kind]:
        return None
    return float(row["evaluations"])


def _share(value: float, total: int) -> str:
    """Format a profile value ``k / total`` as the fraction ``k/total``."""
    return f"{round(value * total)}/{total}"


def _ratio(value: float) -> str:
    """Format a performance ratio or budget (``never`` for infinity)."""
    if not math.isfinite(value):
        return "never"
    if value >= 1000.0:
        return f"{round(value):,}"
    return "1" if value == 1.0 else f"{value:.3g}"


def profile_section(
    profiles: CategoryProfiles, figure: tuple[str, str] | None = None
) -> list[str]:
    """Return the "Performance profiles" subsection of one category.

    ``figure`` holds the light and dark figure paths; the dark one is shown
    to readers whose system prefers a dark colour scheme.
    """
    performance, data = profiles.performance, profiles.data
    total = performance.problems
    order = sorted(
        performance.solvers,
        key=lambda m: (-performance.area(m), -performance.wins(m)),
    )
    lines = [
        "### Performance profiles",
        "",
        f"{total} problem instance(s): {', '.join(profiles.instances)}. Largest "
        f"finite ratio tau_max = {_ratio(performance.max_ratio)}; rows are sorted "
        "by area.",
        "",
    ]
    if figure:
        light, dark = figure
        lines.extend(
            [
                "<picture>",
                f'  <source media="(prefers-color-scheme: dark)" srcset="{dark}">',
                f'  <img alt="Performance profile, {profiles.category}" src="{light}">',
                "</picture>",
                "",
            ]
        )
    header = (
        "Method",
        "Accurate on",
        "Wins rho(1)",
        *(f"rho({tau:g})" for tau in PROFILE_TAUS),
        "tau for 50 %",
        "Area",
    )
    body = [
        [
            method,
            _share(performance.solved_fraction(method), total),
            _share(performance.wins(method), total),
            *(_share(performance.value(method, tau), total) for tau in PROFILE_TAUS),
            _ratio(performance.tau_at(method, 0.5)),
            f"{performance.area(method):.2f}",
        ]
        for method in order
    ]
    lines.extend(table(header, body))
    lines.append("")
    data_header = (
        "Method",
        *(f"d({kappa:g})" for kappa in DATA_BUDGETS),
        "kappa for 50 %",
    )
    data_body = [
        [
            method,
            *(_share(data.value(method, kappa), total) for kappa in DATA_BUDGETS),
            _ratio(data.kappa_at(method, 0.5)),
        ]
        for method in order
    ]
    lines.extend(table(data_header, data_body))
    lines.append("")
    if profiles.excluded:
        lines.append(
            "- Not profiled (no model evaluations recorded, so no cost ratio): "
            + ", ".join(profiles.excluded)
        )
        lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


def findings(problems: Sequence[Problem], summary: Sequence[Row]) -> list[str]:
    """Return the computed "What the numbers say" bullets."""
    lines: list[str] = []
    accurate, inaccurate, cheapest = [], [], []
    under, conservative, failures = [], [], []
    for problem in problems:
        rows = [r for r in summary if r["problem"] == problem.key]
        for group, kind in problem.error_kinds.items():
            scored = [
                r
                for r in rows
                if r.get("group") == group
                and r["kind"] != "reference"
                and r.get("error") is not None
            ]
            if not scored:
                continue
            label = (
                problem.key
                if len(problem.error_kinds) == 1
                else f"{problem.key} {group}"
            )
            if problem.category == "calibration" and group != "mean":
                continue
            best = min(scored, key=lambda r: r["error"])
            accurate.append(f"{label}: {best['method']} ({_metric(best, kind)})")
            limit = {"relative": INACCURATE_REL, "absolute": 0.05, "scaled": 0.1}[kind]
            bad = [r for r in scored if r["error"] > limit]
            if bad:
                inaccurate.append(
                    f"{label}: "
                    + ", ".join(f"{r['method']} ({_metric(r, kind)})" for r in bad)
                )
            good = [
                r
                for r in scored
                if r["error"] <= SOLVED_ERROR[kind]
                and r["evaluations"]
                and not _under_reports(r)
            ]
            if good:
                cheap = min(good, key=lambda r: r["evaluations"])
                cheapest.append(
                    f"{label}: {cheap['method']} ({count(cheap['evaluations'])} "
                    f"evaluations, {_metric(cheap, kind)})"
                )
        for row in rows:
            if _under_reports(row):
                under.append(
                    f"{problem.key} {row['group']}: {row['method']} "
                    f"(median abs z {_z(row['z_median'])}, coverage "
                    f"{row['covered']}/{row['coverage_n']})"
                )
            elif (
                row.get("z_count", 0) >= 4
                and row.get("z_floored", 0) < row["z_count"]
                and row["z_median"] is not None
                and row["z_median"] < OVER_CONSERVATIVE_Z
                and row["kind"] == "estimator"
            ):
                conservative.append(
                    f"{problem.key} {row['group']}: {row['method']} "
                    f"(median abs z {row['z_median']:.2f})"
                )
            if row["failed"] and row.get("group") in (
                None,
                next(iter(problem.error_kinds)),
            ):
                failures.append(
                    f"{row['method']} on {problem.key}: {row['failed']}/{row['runs']} "
                    f"run(s), {_clip(row['errors'][0], 110) if row['errors'] else ''}"
                    + (" (known limitation)" if row.get("expected_failure") else "")
                )
    calibration = [r for r in summary if r["category"] == "calibration"]
    sd_flags = [
        f"{r['method']} (" + " / ".join(f"{v:.2f}" for v in r["ratios"].values()) + ")"
        for r in calibration
        if r.get("group") == "sd"
        and any(
            not SD_RATIO_BAND[0] <= v <= SD_RATIO_BAND[1] for v in r["ratios"].values()
        )
    ]
    lines.append(
        "Computed from the summary above with fixed thresholds: *accurate* means "
        f"median relative error <= {ACCURATE_REL:.0%} (absolute error <= 0.02 for "
        "indices, <= 0.05 posterior sd for calibration means); *inaccurate* means > "
        f"{INACCURATE_REL:.0%} (> 0.05 for indices, > 0.1 posterior sd for "
        f"calibration means); *under-reports* means median abs z > {UNDER_REPORT_Z:g} "
        "or the reported interval covered the reference in fewer than half of at "
        "least four cases; *conservative* means median abs z < "
        f"{OVER_CONSERVATIVE_Z:g} over at least four cases (excluding methods whose "
        "reported SE is below the SE floor, i.e. exact up to solver tolerance). "
        "With the few seeds of quick mode, coverage and z statistics are coarse."
    )
    lines.append("")
    sections = (
        ("Most accurate method per problem", accurate),
        ("Inaccurate at this budget", inaccurate),
        ("Cheapest accurate method (fewest model evaluations)", cheapest),
        ("Under-report their error", under),
        (
            "Conservative error reports (reported SE overstates the actual error)",
            conservative,
        ),
        (
            "Posterior sd ratio outside "
            f"[{SD_RATIO_BAND[0]}, {SD_RATIO_BAND[1]}] (calibration)",
            sd_flags,
        ),
        ("Failures", failures),
    )
    for title, items in sections:
        lines.append(f"**{title}.**" + ("" if items else " None."))
        lines.append("")
        lines.extend(f"- {item}" for item in items)
        if items:
            lines.append("")
    return lines


def _metric(row: Row, kind: str) -> str:
    """Format a row's error in its own metric."""
    if kind == "relative":
        return percent(row["error"])
    if kind == "scaled":
        return f"{row['error']:.3f} posterior sd"
    return f"max abs. error {row['error']:.3f}"


def _under_reports(row: Row) -> bool:
    """Return whether a row's reported error is too small for its actual error."""
    if row.get("kind") not in ("estimator",):
        return False
    if row.get("z_count") and row["z_median"] is not None:
        if row["z_median"] > UNDER_REPORT_Z:
            return True
    total = row.get("coverage_n") or 0
    return total >= 4 and row["covered"] < 0.5 * total


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def render(
    meta: Mapping[str, Any],
    problems: Sequence[Problem],
    summary: Sequence[Row],
    figures: Mapping[str, tuple[str, str]] | None = None,
) -> str:
    """Return the complete Markdown report.

    ``figures`` optionally maps a category to the light and dark paths
    (relative to the report) of its performance-profile figure.
    """
    seeds = meta["seeds"]
    lines = [
        "# Benchmarks",
        "",
        "Generated by `" + meta["command"] + "` "
        f"({meta['mode']} mode, {meta['generated_utc']}, "
        f"total {meta['total_time_s']:.1f} s).",
        "Do not edit by hand: rerun the command to regenerate this file and "
        "`benchmarks/results/" + meta["mode"] + ".json` (every individual run).",
        "",
        "**Caveats.** Times are wall-clock seconds of pure-Python code on one "
        f"machine ({meta['platform']}, Python {meta['python']}) and are indicative "
        "only; model evaluations (read from each model's evaluation counter) are "
        "machine-independent and are the cost measure to compare. Seeds per "
        f"category: {', '.join(f'{k} {v}' for k, v in seeds.items())}; deterministic "
        "methods (FORM, SORM, bounds, closed forms) run once. Quick mode uses small "
        "budgets and few seeds: its numbers show behaviour, not converged accuracy.",
        "",
        "**Columns.** *Rel. error*: median over seeds of |estimate - reference| / "
        "|reference|. *Median abs z*: |estimate - reference| / (the method's own "
        "reported standard error), pooled over seeds; about 0.67 is ideal, values "
        "well above 2 mean the method under-reports its error (a reported SE below "
        f"{SE_FLOOR_RELATIVE:g} x |reference| is floored there, the level of solver "
        "tolerances). *Coverage*: how often the reference "
        "lay inside the method's own interval (estimate +/- 1.96 SE, a bootstrap "
        "95 % interval, or a 90 % credible interval for calibration). Methods "
        "marked as approximations (FORM, SORM, PCE, Laplace, ADVI) report no "
        "error bar, so only their error is shown. Failures are caught and listed "
        "under each table, never dropped.",
        "",
        "**Profiles** (ADR-0077). Each category ends with performance profiles "
        "(Dolan and "
        "Moré 2002) and data profiles (Moré and Wild 2009) over its problem "
        "instances (a problem and quantity group). A method *solves* an instance "
        "when its median error meets the *accurate* threshold of 'What the "
        "numbers say'; its cost t is then its mean model evaluations, and its "
        "ratio is r = t / (smallest t of any method on that instance). An "
        "inaccurate, failed, skipped, or inapplicable method has r = infinity. "
        "*Accurate on*: instances solved at any cost. *rho(tau)*: instances "
        "solved with r <= tau; *Wins rho(1)* counts a tie for every tied method. "
        "*tau for 50 %*: smallest ratio at which half the instances are solved. "
        "*Area*: mean of rho over log2 tau in [0, log2 tau_max], 1 when a method "
        "is the cheapest accurate one everywhere. *d(kappa)*: instances solved "
        "within kappa (n + 1) evaluations, n the input dimension; *kappa for "
        "50 %*: that budget for half of them. With quick-mode budgets and a "
        "handful of instances per category the profiles are coarse.",
        "",
    ]
    for category in ("reliability", "sensitivity", "propagation", "calibration"):
        chosen = [p for p in problems if p.category == category]
        if not chosen:
            continue
        lines.append(f"## {CATEGORY_TITLES[category]}")
        lines.append("")
        for problem in chosen:
            rows = [r for r in summary if r["problem"] == problem.key]
            lines.append(f"### {problem.title}")
            lines.append("")
            lines.append(f"{problem.description} Reference: {problem.reference_note}")
            lines.append("")
            if category in ("reliability", "propagation"):
                lines.extend(scalar_tables(problem, rows))
            elif category == "sensitivity":
                lines.extend(index_table(problem, rows))
            else:
                lines.extend(calibration_table(problem, rows))
            notes = notes_block(rows)
            if notes:
                lines.extend(notes)
                lines.append("")
        profiles = category_profiles(chosen, summary, category)
        if profiles is not None:
            lines.extend(profile_section(profiles, (figures or {}).get(category)))
    lines.append("## What the numbers say")
    lines.append("")
    lines.extend(findings(problems, summary))
    return "\n".join(lines).rstrip() + "\n"
