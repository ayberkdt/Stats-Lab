"""Run the probstat benchmark suite and write JSON results and a Markdown report.

Usage (from the repository root)::

    python benchmarks/run.py --quick      # CI-sized, under a minute
    python benchmarks/run.py              # full sizes and seed counts

The exit status is 1 when a method run failed unexpectedly (every failure is
recorded in both outputs; known limitations are listed in
``benchmarks.methods.EXPECTED_FAILURES``), unless ``--allow-failures`` is given.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _prepare_path() -> None:
    """Make ``benchmarks`` and (if not installed) ``probstat`` importable."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        import probstat  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(ROOT / "src"))


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Benchmark probstat methods against known reference answers."
    )
    parser.add_argument(
        "--quick", action="store_true", help="CI-sized run (< 60 s); default is full"
    )
    parser.add_argument(
        "--seeds", type=int, default=None, help="seeds per category (overrides mode)"
    )
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated problem keys or categories, e.g. 'reliability,ishigami'",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="results file (default benchmarks/results/<mode>.json)",
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "docs" / "benchmarks.md"),
        help="Markdown report path, or '-' to skip the report",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="exit 0 even if a method run failed unexpectedly",
    )
    return parser.parse_args(argv)


def select(problems: tuple[Any, ...], only: str) -> tuple[Any, ...]:
    """Return the problems matching ``--only`` (all when empty)."""
    wanted = {item.strip().lower() for item in only.split(",") if item.strip()}
    if not wanted:
        return problems
    chosen = tuple(
        p
        for p in problems
        if p.category in wanted or p.key in wanted or any(w in p.key for w in wanted)
    )
    if not chosen:
        raise SystemExit(f"--only matched no problem: {sorted(wanted)}")
    return chosen


def main(argv: list[str] | None = None) -> int:
    """Run the suite; return the process exit status."""
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    _prepare_path()
    from benchmarks import harness, methods, problems, report

    mode = "quick" if arguments.quick else "full"
    sizes = methods.SIZES[mode]
    seeds = None
    if arguments.seeds is not None:
        if arguments.seeds < 1:
            raise SystemExit("--seeds must be at least 1")
        seeds = {category: arguments.seeds for category in problems.CATEGORIES}
    chosen = select(problems.registry(), arguments.only)
    command = "python benchmarks/run.py" + "".join(
        f" {item}" for item in (sys.argv[1:] if argv is None else argv)
    )
    print(f"probstat benchmarks, {mode} mode: {len(chosen)} problems", flush=True)
    start = time.perf_counter()
    records = harness.run_benchmarks(
        chosen,
        methods.registry(),
        sizes,
        seeds=seeds,
        progress=lambda line: print("  " + line, flush=True),
    )
    elapsed = time.perf_counter() - start
    summary = harness.aggregate(records, chosen)
    import probstat

    meta = {
        "command": command,
        "mode": mode,
        "seeds": seeds or sizes["seeds"],
        "sizes": {k: v for k, v in sizes.items() if k != "seeds"},
        "total_time_s": elapsed,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "probstat_version": getattr(probstat, "__version__", "unknown"),
    }
    json_path = Path(arguments.json or ROOT / "benchmarks" / "results" / f"{mode}.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "problems": [report.problem_meta(p) for p in chosen],
        "summary": summary,
        "records": records,
    }
    json_path.write_text(json.dumps(payload, indent=1, default=str) + "\n", "utf-8")
    written = [str(json_path)]
    if arguments.report != "-":
        report_path = Path(arguments.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report.render(meta, chosen, summary), "utf-8")
        written.append(str(report_path))
    runs = {(r["problem"], r["method"], r["seed"]): r for r in records}
    failed = [r for r in runs.values() if r["status"] == "failed"]
    unexpected = [r for r in failed if not r.get("expected_failure")]
    skipped = sum(1 for r in runs.values() if r["status"] == "skipped")
    print(
        f"Done in {elapsed:.1f} s: {len(runs)} runs, {len(failed)} failed "
        f"({len(unexpected)} unexpected), {skipped} skipped. "
        f"Wrote {' and '.join(written)}. Next: read {written[-1]}.",
        flush=True,
    )
    if unexpected and not arguments.allow_failures:
        for record in unexpected:
            print(
                f"  UNEXPECTED FAILURE {record['problem']} / {record['method']} "
                f"seed {record['seed']}: {record['error']}",
                flush=True,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
