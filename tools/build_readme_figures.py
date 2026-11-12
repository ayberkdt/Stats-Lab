"""Build the two block diagrams of the README (ADR-0099).

``docs/images/readme-one-model-workflow{,-dark}.png`` shows how one
``ProbabilisticModel`` feeds every ``UQStudy`` stage and one report, with the
systems-engineering branch (``SystemModel`` to ``MissionAssessment``).
``docs/images/readme-package-layers{,-dark}.png`` shows the package layers,
read from the ``LAYER`` dictionary of ``tests/test_architecture.py`` (the map
the architecture test enforces), so the picture cannot list a package in a
layer the test does not.

Both diagrams are drawn with the datawiz diagram toolkit
(``datawiz.diyagram.Diyagram``): it lays the blocks on a grid, compiles them
with LaTeX, and audits the compiled page (overlapping words, text crossing a
line or a box edge, overflow, text under 6 pt, contrast). The build fails on
any audit error. ``docs/images/readme-diagrams.json`` records what was drawn
(stages, classes, and layers) so that ``tests/test_documentation_examples.py``
can check the drawing against the code without LaTeX.

Usage::

    PYTHONPATH=src python tools/build_readme_figures.py

Needs a LaTeX installation and the datawiz toolkit, which are not library
dependencies; the README test needs neither.

References:
    [REF-PROBSTAT-REPORTING], [REF-PYTHON-AST].

"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"
ARCHITECTURE_TEST = ROOT / "tests" / "test_architecture.py"
MANIFEST = IMAGES / "readme-diagrams.json"
WORKFLOW_NAME = "readme-one-model-workflow"
LAYERS_NAME = "readme-package-layers"
THEME = "derin"
PACKAGES_PER_ROW = 6

# Classes drawn in the workflow diagram, with the package that exports them.
WORKFLOW_CLASSES = {
    "JointInput": "modeling",
    "ProbabilisticModel": "modeling",
    "UQStudy": "workflow",
    "SystemModel": "systems",
    "MissionAssessment": "workflow",
}
# UQStudy stages drawn in the workflow diagram, with a two-word answer each.
WORKFLOW_STAGES = (
    ("propagate", "output distribution"),
    ("sensitivity", "which input matters"),
    ("reliability", "failure probability"),
    ("calibrate", "inputs from data"),
    ("validate", "scoped verdict"),
    ("extremes", "return levels"),
    ("design", "next experiment"),
)


def architecture_layers(path: Path = ARCHITECTURE_TEST) -> dict[int, list[str]]:
    """Read the package layers from the architecture test.

    Args:
        path: ``tests/test_architecture.py``.

    Returns:
        Layer number to subpackage names (without the ``probstat.`` prefix),
        in the order of the ``LAYER`` dictionary. The root package
        ``probstat`` (version metadata only) is left out.

    Raises:
        ValueError: If the file has no literal ``LAYER`` dictionary.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "LAYER"
        ):
            table = ast.literal_eval(node.value)
            layers: dict[int, list[str]] = {}
            for module, layer in table.items():
                if module == "probstat":
                    continue
                layers.setdefault(int(layer), []).append(module.split(".", 1)[1])
            return dict(sorted(layers.items()))
    raise ValueError(f"no LAYER dictionary in {path}")


def _diagram_class() -> Any:
    """Import the datawiz diagram class.

    Returns:
        ``datawiz.diyagram.Diyagram``.

    Raises:
        SystemExit: If datawiz is not importable.

    References:
        [REF-PROBSTAT-REPORTING]. No mathematical equation is evaluated.

    """
    try:
        module = importlib.import_module("datawiz.diyagram")
    except ImportError as error:  # pragma: no cover - depends on the machine
        raise SystemExit(f"the datawiz toolkit is required: {error}") from error
    return module.Diyagram


def workflow_diagram(dark: bool) -> Any:
    """Lay out the "one model, every question" diagram.

    Args:
        dark: Draw the dark-background variant.

    Returns:
        An unsaved ``Diyagram``.

    References:
        [REF-PROBSTAT-REPORTING]. No mathematical equation is evaluated.

    """
    diagram = _diagram_class()(
        "One model answers every question",
        alt="Inputs and simulator are defined once; every stage reuses them "
        "under one seed tree",
        tema=THEME,
        koyu=dark,
        blok_genislik_mm=30,
        punto="normalsize",
        satir_araligi_mm=3.0,
    )
    middle = len(WORKFLOW_STAGES) // 2
    diagram.blok("inputs", "JointInput", middle - 1, 0, alt="marginals + copula")
    diagram.blok("simulator", "Simulator", middle + 1, 0, alt="named outputs")
    diagram.blok(
        "model", "ProbabilisticModel", middle, 1, alt="one model contract", vurgu=True
    )
    diagram.blok("study", "UQStudy", middle, 2, alt="one seed tree", renk=2)
    for row, (stage, answer) in enumerate(WORKFLOW_STAGES):
        diagram.blok(stage, stage, row, 3, alt=answer, renk=3)
    diagram.blok("report", "Report", middle, 4, alt="Markdown, HTML, figures", renk=4)
    bottom = len(WORKFLOW_STAGES)
    diagram.blok("system", "SystemModel", bottom, 0, alt="BDD, Markov, CCF", renk=5)
    diagram.blok(
        "mission", "MissionAssessment", bottom, 2, alt="exact R(T), MTTF, cost", renk=5
    )
    diagram.bag("inputs", "model")
    diagram.bag("simulator", "model")
    diagram.bag("model", "study")
    for stage, _ in WORKFLOW_STAGES:
        diagram.bag("study", stage)
        diagram.bag(stage, "report")
    diagram.bag("system", "mission")
    diagram.bag("mission", "report")
    diagram.grup([stage for stage, _ in WORKFLOW_STAGES], "UQStudy stages", renk=3)
    return diagram


def layer_diagram(layers: dict[int, list[str]], dark: bool) -> Any:
    """Lay out the package-layer map, highest layer on top.

    Args:
        layers: Layer number to package names, from
            :func:`architecture_layers`.
        dark: Draw the dark-background variant.

    Returns:
        An unsaved ``Diyagram``.

    References:
        [REF-PROBSTAT-REPORTING]. No mathematical equation is evaluated.

    """
    diagram = _diagram_class()(
        "Packages form layers; imports point down only",
        alt="Read from the layer map that tests/test_architecture.py enforces; "
        "filled block: modeling, home of ProbabilisticModel",
        tema=THEME,
        koyu=dark,
        blok_genislik_mm=30,
        punto="normalsize",
        satir_araligi_mm=3.0,
        sutun_araligi_mm=4.0,
    )
    row = 0
    for layer in sorted(layers, reverse=True):
        names = layers[layer]
        diagram.not_(f"layer{layer}", f"Layer {layer}", row, 0)
        for index, name in enumerate(names):
            diagram.blok(
                name,
                name,
                row + index // PACKAGES_PER_ROW,
                index % PACKAGES_PER_ROW + 1,
                vurgu=name == "modeling",
            )
        row += (len(names) - 1) // PACKAGES_PER_ROW + 1
    return diagram


def manifest(layers: dict[int, list[str]]) -> dict[str, Any]:
    """Describe what the two diagrams draw.

    Args:
        layers: Layer number to package names.

    Returns:
        JSON-ready record of the workflow classes and stages and of the
        layers, checked by ``tests/test_documentation_examples.py``.

    References:
        [REF-PROBSTAT-REPORTING]. No mathematical equation is evaluated.

    """
    return {
        "source": "tools/build_readme_figures.py",
        "workflow": {
            "images": [f"{WORKFLOW_NAME}.png", f"{WORKFLOW_NAME}-dark.png"],
            "classes": WORKFLOW_CLASSES,
            "stages": [stage for stage, _ in WORKFLOW_STAGES],
        },
        "layers": {
            "images": [f"{LAYERS_NAME}.png", f"{LAYERS_NAME}-dark.png"],
            "from": "tests/test_architecture.py",
            "packages": {str(layer): names for layer, names in layers.items()},
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Draw, audit, and write both diagrams in light and dark.

    Args:
        argv: Command-line arguments (``--out`` overrides ``docs/images``).

    Returns:
        ``0`` when every audit reports no error, ``1`` otherwise.

    References:
        [REF-PROBSTAT-REPORTING]. No mathematical equation is evaluated.

    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, default=IMAGES)
    args = parser.parse_args(argv)
    layers = architecture_layers()
    builders = {
        WORKFLOW_NAME: lambda dark: workflow_diagram(dark),
        LAYERS_NAME: lambda dark: layer_diagram(layers, dark),
    }
    failed = False
    with tempfile.TemporaryDirectory() as scratch:
        for name, build in builders.items():
            for dark in (False, True):
                stem = f"{name}-dark" if dark else name
                _, report = build(dark).kaydet(str(Path(scratch) / stem), svg=False)
                print(f"{stem}: {report.strip()}")
                failed = failed or " 0 HATA" not in report
                shutil.copyfile(Path(scratch) / f"{stem}.png", args.out / f"{stem}.png")
    (args.out / MANIFEST.name).write_text(
        json.dumps(manifest(layers), indent=2) + "\n", encoding="utf-8"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
