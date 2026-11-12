"""Render, audit, and record the figures of the foundation guides (ADR-0105).

Each :class:`guide_figures.GuideFigure` is drawn twice, in the light
(porcelain) and the dark (graphite) palette, from the same seeded
computation. Before a file is written the figure goes through
``theme.fit_text_to_canvas`` and the datawiz web audit
(``datawiz.figur_denetle(figure, ortam="web")``); any error fails the build
and nothing of that figure is written. The light and dark runs must return
identical facts.

Outputs, per group module ``tools/guide_figures/<group>.py``:

- ``docs/images/guides/<name>.png`` and ``<name>-dark.png``;
- ``docs/images/guides/<group>.json``: for every figure its guide, text
  alternative, and facts, which ``tests/test_guide_figures.py`` checks
  against the caption in the guide without Matplotlib or datawiz.

Usage::

    PYTHONPATH=src python tools/build_guide_figures.py            # all
    PYTHONPATH=src python tools/build_guide_figures.py --group inference
    PYTHONPATH=src python tools/build_guide_figures.py --only qq-plot-tails

Needs Matplotlib and the datawiz toolkit; neither is a library dependency.

References:
    [REF-PROBSTAT-REPORTING], [REF-MATPLOTLIB], [REF-PNG-SPEC-2003].

"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "images" / "guides"
GUIDES = ROOT / "docs" / "foundations"
DPI = 160


def _audit(figure: Any) -> list[str]:
    """Return the datawiz web-audit errors of a figure (empty when clean).

    References:
        [REF-PROBSTAT-REPORTING].

    """
    import datawiz

    findings = datawiz.figur_denetle(figure, ortam="web")
    return [
        f"{finding.kural}: {finding.mesaj} @ {finding.nerede}"
        for finding in findings
        if str(finding.duzey).upper().startswith("HATA") and not finding.muaf
    ]


def _render(spec: Any, palette: Any, path: Path) -> dict[str, str]:
    """Draw one figure in one palette, audit it, and save it.

    References:
        [REF-MATPLOTLIB], [REF-PNG-SPEC-2003].

    """
    from probstat.visualization import theme

    plt = theme.pyplot()
    with theme.chart_theme(palette):
        figure, facts = spec.draw()
        theme.fit_text_to_canvas(figure)
        errors = _audit(figure)
        if errors:
            plt.close(figure)
            raise SystemExit(
                f"{path.name}: datawiz audit failed:\n  " + "\n  ".join(errors)
            )
        theme.save_figure(figure, path, dpi=DPI, description=spec.alt)
    plt.close(figure)
    return {str(key): str(value) for key, value in facts.items()}


def main(argv: Sequence[str] | None = None) -> int:
    """Build the requested guide figures and their manifests.

    References:
        [REF-PROBSTAT-REPORTING].

    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--group", action="append", default=[], help="group module")
    parser.add_argument("--only", action="append", default=[], help="figure name")
    arguments = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT / "tools"))
    import matplotlib

    matplotlib.use("Agg")
    import guide_figures

    from probstat.visualization.theme import GRAPHITE, PORCELAIN

    OUTPUT.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    for group, figures in guide_figures.groups():
        for spec in figures:
            if spec.name in seen:
                raise SystemExit(
                    f"figure {spec.name} declared in {seen[spec.name]} and {group}"
                )
            seen[spec.name] = group
            if not (GUIDES / f"{spec.guide}.md").is_file():
                raise SystemExit(
                    f"{spec.name}: no guide docs/foundations/{spec.guide}.md"
                )
        if arguments.group and group not in arguments.group:
            continue
        manifest_path = OUTPUT / f"{group}.json"
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared = {spec.name for spec in figures}
        manifest = {name: entry for name, entry in manifest.items() if name in declared}
        for spec in figures:
            if arguments.only and spec.name not in arguments.only:
                continue
            light = _render(spec, PORCELAIN, OUTPUT / f"{spec.name}.png")
            dark = _render(spec, GRAPHITE, OUTPUT / f"{spec.name}-dark.png")
            if light != dark:
                raise SystemExit(
                    f"{spec.name}: light and dark runs disagree: {light} vs {dark}"
                )
            manifest[spec.name] = {"guide": spec.guide, "alt": spec.alt, "facts": light}
            print(f"{group}: {spec.name} ok ({len(light)} facts)")
        ordered = {name: manifest[name] for name in sorted(manifest)}
        manifest_path.write_text(
            json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
