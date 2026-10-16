"""Generate ``docs/api-reference.md`` from the public API of ``probstat``.

Every package's ``__all__`` (and that of public subpackages such as
``inference.survival``) is listed with the kind of each name and the first
line of its docstring. Run from the repository root::

    PYTHONPATH=src python tools/generate_api_reference.py

``tests/test_api_reference.py`` fails when the committed file is stale.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from pathlib import Path
from types import ModuleType

GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "Foundations",
        "Seeds and validation, special functions, dense linear algebra, "
        "optimizers, and sampling designs.",
        ("foundations", "numerics", "linalg", "optimization", "sampling"),
    ),
    (
        "Probability and statistics",
        "Exact finite probability, distribution families, descriptive and "
        "test statistics, multivariate laws, stochastic processes, and "
        "information measures.",
        (
            "probability",
            "distributions",
            "statistics",
            "multivariate",
            "processes",
            "information",
        ),
    ),
    (
        "Inference and modelling",
        "Estimation, Bayesian computation, regression, survival, time series, "
        "causal effects, conformal prediction, and experimental design.",
        ("inference", "timeseries", "causal", "conformal", "design"),
    ),
    (
        "Forward uncertainty quantification",
        "The central probabilistic model, uncertainty taxonomy, propagation, "
        "Monte Carlo, sensitivity, and surrogates.",
        (
            "modeling",
            "uq",
            "propagation",
            "monte_carlo",
            "sensitivity",
            "surrogates",
        ),
    ),
    (
        "Backward analysis and validation",
        "Calibration, filtering and data assimilation, and model validation.",
        ("inverse", "filtering", "data_assimilation", "validation"),
    ),
    (
        "Reliability and risk",
        "Structural reliability, exact system reliability (BDDs, importance, "
        "common-cause failures, phased missions), risk measures, fault trees, "
        "and extremes.",
        ("reliability", "systems", "risk"),
    ),
    (
        "Orchestration and figures",
        "The `UQStudy` workflow and the themed figure library.",
        ("workflow", "visualization"),
    ),
)


def _kind(value: object) -> str:
    """Return a short kind label for a public object.

    References:
        Project documentation tooling. No equation is evaluated.

    """
    if inspect.isclass(value):
        return "class"
    if inspect.isfunction(value) or inspect.isbuiltin(value):
        return "function"
    if isinstance(value, ModuleType):
        return "module"
    return "constant"


def _summary(value: object) -> str:
    """Return the first docstring line of a class or function, else ``""``.

    References:
        Project documentation tooling. No equation is evaluated.

    """
    if not (inspect.isclass(value) or inspect.isroutine(value)):
        return ""
    doc = inspect.getdoc(value) or ""
    first = doc.strip().splitlines()[0] if doc.strip() else ""
    return first.replace("|", "\\|")


def _public_modules(name: str) -> list[ModuleType]:
    """Return a package and its public subpackages that define ``__all__``.

    References:
        Project documentation tooling. No equation is evaluated.

    """
    package = importlib.import_module(f"probstat.{name}")
    found = [package]
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
        if info.ispkg and not info.name.startswith("_"):
            child = importlib.import_module(f"{package.__name__}.{info.name}")
            if getattr(child, "__all__", None):
                found.append(child)
    return found


def _anchor(text: str) -> str:
    """Return the GitHub heading anchor for ``text``.

    References:
        Project documentation tooling. No equation is evaluated.

    """
    kept = "".join(c for c in text.lower() if c.isalnum() or c in " -_")
    return kept.replace(" ", "-")


def render() -> str:
    """Return the Markdown API reference for the installed ``probstat``.

    Raises:
        ValueError: If a package is missing from, or listed twice in,
            ``GROUPS``, or has no stability label.

    References:
        Project documentation tooling. No equation is evaluated.

    """
    import probstat
    from probstat.foundations.stability import PACKAGE_STABILITY

    packages = sorted(
        info.name
        for info in pkgutil.iter_modules(probstat.__path__)
        if info.ispkg and not info.name.startswith("_")
    )
    grouped = [name for _, _, names in GROUPS for name in names]
    if sorted(grouped) != packages:
        missing = sorted(set(packages) - set(grouped))
        extra = sorted(set(grouped) - set(packages))
        raise ValueError(f"GROUPS out of date: missing {missing}, unknown {extra}")
    unlabelled = sorted(set(packages) - set(PACKAGE_STABILITY))
    if unlabelled:
        raise ValueError(f"packages without a stability label: {unlabelled}")
    total = 0
    sections: list[str] = []
    for title, blurb, names in GROUPS:
        sections.append(f"## {title}\n\n{blurb}\n")
        for name in names:
            for module in _public_modules(name):
                exported = sorted(module.__all__, key=str.lower)
                total += len(exported)
                dotted = module.__name__
                label = PACKAGE_STABILITY[dotted.split(".")[1]]
                sections.append(f"### `{dotted}` · {label}\n")
                doc = (module.__doc__ or "").strip().splitlines()
                if doc:
                    sections.append(f"{doc[0]}\n")
                sections.append("| Name | Kind | Summary |\n|---|---|---|")
                for item in exported:
                    value = getattr(module, item)
                    row = f"| `{item}` | {_kind(value)} | {_summary(value)} |"
                    sections.append(row)
                sections.append("")
    contents = "\n".join(f"- [{title}](#{_anchor(title)})" for title, _, _ in GROUPS)
    header = (
        "# API reference\n\n"
        "Generated by `tools/generate_api_reference.py` from each package's "
        f"`__all__`; {total} public names in {len(packages)} packages. Each "
        "entry shows the first line of the docstring; the full docstring "
        "states arguments, equations (`EQ-*`, see "
        "[equation-registry.md](equation-registry.md)) and references "
        "(`REF-*`, see [references.md](references.md)). Each package heading "
        "carries its stability label (ADR-0048): *stable* changes only after "
        "a deprecation period, *provisional* may still change but is "
        "deprecated first where practical, *experimental* may change in any "
        "release.\n\n"
        f"{contents}\n"
    )
    return header + "\n" + "\n".join(sections).rstrip() + "\n"


def main() -> None:
    """Write ``docs/api-reference.md`` next to this script's repository.

    References:
        Project documentation tooling. No equation is evaluated.

    """
    target = Path(__file__).resolve().parents[1] / "docs" / "api-reference.md"
    target.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    sys.exit(main())
