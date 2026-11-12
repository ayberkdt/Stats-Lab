"""Build the data, figures, and checks of the overview presentation (ADR-0097).

Every number on ``docs/presentation/index.html`` and ``index.tr.html`` comes
from one JSON file, ``docs/presentation/deck-data.json``, written here from a
fixed-seed run of the library, a read of ``benchmarks/results/quick.json``,
and counts read from the repository. The two HTML decks carry each number in
an element ``<span data-k="dotted.key" data-fmt="...">``; this tool writes the
formatted value into that element, so the static HTML is complete without
JavaScript and a stale value is a test failure, not a silent drift.

Usage::

    PYTHONPATH=src python tools/build_presentation_data.py   # data, HTML, checks
    PYTHONPATH=src python tools/build_presentation_data.py --figures  # + charts

``--figures`` needs Matplotlib and the datawiz figure toolkit: each chart is
drawn at slide size (datawiz ``sunum`` target), audited with
``datawiz.figur_denetle(fig, ortam="sunum")`` and written in a light and a
dark variant; the run fails when any audit reports an error.

References:
    [REF-PROBSTAT-REPORTING], [REF-WHATWG-HTML], [REF-MATPLOTLIB].

"""

from __future__ import annotations

import argparse
import ast
import contextlib
import html
import importlib.util
import io
import json
import math
import re
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "docs" / "presentation"
DATA_PATH = DECK / "deck-data.json"
FIGURES = DECK / "figures"
CHECKS = DECK / "checks"
DECK_FILES = {"en": DECK / "index.html", "tr": DECK / "index.tr.html"}
SEED = 2026
MISSION_HOURS = 720.0
SCHEMA = 1

# Repository counts change with every feature; the deck shows them as
# floors ("2,100+ tests") so that the slides stay true between rebuilds.
FLOORS = {"tests": 100, "equations": 50, "references": 50, "adrs": 10}


# --------------------------------------------------------------------------
# Repository counts
# --------------------------------------------------------------------------


def _floor(value: int, step: int) -> int:
    """Round a count down to a multiple of ``step``.

    Args:
        value: Exact count.
        step: Display granularity.

    Returns:
        ``step * floor(value / step)``.

    References:
        [REF-PROBSTAT-REPORTING]. No mathematical equation is evaluated.

    """
    return step * (value // step)


def count_tests(root: Path = ROOT) -> int:
    """Count the test methods that ``unittest`` discovery collects.

    Every ``test*`` method defined directly in a class of a
    ``tests/test_*.py`` module is one test case, which equals
    ``unittest.defaultTestLoader.discover("tests").countTestCases()`` for this
    repository without importing the test modules.

    Args:
        root: Repository root.

    Returns:
        Number of test methods.

    References:
        [REF-PYTHON-AST], [REF-PYTHON-UNITTEST].

    """
    total = 0
    for path in sorted((root / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                total += sum(
                    1
                    for item in node.body
                    if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                    and item.name.startswith("test")
                )
    return total


def _central_model_questions(root: Path) -> int:
    """Count the questions of the central-model table in ``docs/architecture.md``.

    Args:
        root: Repository root.

    Returns:
        Number of table rows between the ``| Question | Call |`` header and
        the next blank line.

    References:
        [REF-PROBSTAT-API]. No mathematical equation is evaluated.

    """
    lines = (root / "docs" / "architecture.md").read_text(encoding="utf-8").splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith("| Question | Call |")
    )
    rows = 0
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows += 1
    return rows


def repository_counts(root: Path = ROOT) -> dict[str, Any]:
    """Read the verification counts shown on the deck from the repository.

    Args:
        root: Repository root.

    Returns:
        Floors of the test, equation, reference, and ADR counts, the number
        of central-model questions, top-level packages, runtime
        dependencies, and the CI Python versions.

    References:
        [REF-PROBSTAT-REPORTING], [REF-PYTHON-AST].

    """
    registry = (root / "docs" / "equation-registry.md").read_text(encoding="utf-8")
    bibliography = (root / "docs" / "references.md").read_text(encoding="utf-8")
    equations = len(re.findall(r"^### EQ-", registry, flags=re.MULTILINE))
    references = len(re.findall(r"^## REF-", bibliography, flags=re.MULTILINE))
    adrs = len(list((root / "docs" / "adr").glob("ADR-*.md")))
    packages = sum(
        1
        for path in (root / "src" / "probstat").iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    )
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    dependencies = re.search(
        r"^dependencies = \[(.*?)\]", pyproject, flags=re.MULTILINE | re.DOTALL
    )
    runtime = (
        0
        if dependencies is None
        else len(re.findall(r'"[^"]+"', dependencies.group(1)))
    )
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    matrix = re.search(r"python-version: \[(.*?)\]", workflow)
    versions = [] if matrix is None else re.findall(r'"([^"]+)"', matrix.group(1))
    labs = len(list((root / "examples").glob("*_lab.py")))
    licence = re.search(r'^license = "([^"]+)"', pyproject, flags=re.MULTILINE)
    theme = (root / "src" / "probstat" / "visualization" / "theme.py").read_text(
        encoding="utf-8"
    )
    floor = re.search(r"^ANNOTATION_SIZE = ([0-9.]+)", theme, flags=re.MULTILINE)
    return {
        "licence": licence.group(1) if licence else "",
        "figure_text_floor_pt": float(floor.group(1)) if floor else 0.0,
        "tests_floor": _floor(count_tests(root), FLOORS["tests"]),
        "equations_floor": _floor(equations, FLOORS["equations"]),
        "references_floor": _floor(references, FLOORS["references"]),
        "adrs_floor": _floor(adrs, FLOORS["adrs"]),
        "questions": _central_model_questions(root),
        "packages": packages,
        "runtime_dependencies": runtime,
        "labs": labs,
        "ci_python_first": versions[0] if versions else "",
        "ci_python_last": versions[-1] if versions else "",
        "ci_python_count": len(versions),
    }


# --------------------------------------------------------------------------
# The quick-start study (README "One model, every analysis")
# --------------------------------------------------------------------------


def _slug(text: str) -> str:
    """Turn a method name into a JSON key.

    Args:
        text: Method or problem name.

    Returns:
        Lower-case key with ``_`` separators.

    References:
        [REF-PROBSTAT-REPORTING]. No mathematical equation is evaluated.

    """
    return re.sub(r"[^a-z0-9]+", "_", text.lower().replace("'", "")).strip("_")


def quick_start_study() -> tuple[dict[str, Any], dict[str, float]]:
    """Run the README quick-start study and collect the numbers of the deck.

    The model is the README example: ``margin = strength - load`` with
    ``load ~ LogNormal(0.9, 0.25)`` and ``strength ~ Normal(4.0, 0.4)`` (kN),
    one ``UQStudy`` with root seed 2026. The reference failure probability
    ``P(strength < load) = integral F_S(l) f_L(l) dl`` is evaluated by
    adaptive quadrature, independently of every estimator.

    Returns:
        The study section of the deck data and the standard errors of its
        Monte Carlo quantities (dotted key to standard error).

    References:
        [EQ-REL-FORM-001], [REF-HASOFER-LIND-1974], [REF-SALTELLI-2010],
        [REF-ROCKAFELLAR-URYASEV-2000].

    """
    from probstat.distributions import LogNormal, Normal
    from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
    from probstat.numerics import integrate
    from probstat.risk import custom_loss
    from probstat.uq import QuantityOfInterest
    from probstat.workflow import UQStudy

    load, strength = LogNormal(0.9, 0.25), Normal(4.0, 0.4)
    inputs = JointInput.independent(
        RandomInput("load", load, unit="kN"),
        RandomInput("strength", strength, unit="kN"),
    )
    model = ProbabilisticModel(
        inputs,
        lambda p: {"margin": p["strength"] - p["load"]},
        (QuantityOfInterest("margin", "capacity minus demand", "kN"),),
    )
    study = UQStudy(model, seed=SEED)
    forward = study.propagate(4096, design="sobol")
    sobol = study.sensitivity("margin", 1024)
    methods = study.reliability("margin", 0.0, failure="below")
    loss_rate = 50.0
    risk = study.risk("margin", custom_loss(lambda m: loss_rate * max(0.0, -m)), 20_000)

    reference = integrate(lambda x: strength.cdf(x) * load.pdf(x), 0.0, math.inf).value
    margin = forward["margin"]
    rows: dict[str, dict[str, Any]] = {}
    errors: dict[str, float] = {}
    for row in methods.rows:
        key = _slug(row.method)
        cov = row.coefficient_of_variation
        entry: dict[str, Any] = {
            "method": row.method,
            "probability": row.probability,
            "evaluations": row.evaluations,
            "cov": cov,
        }
        if cov is not None:
            se = cov * row.probability
            entry["ci95_low"] = row.probability - 1.959963984540054 * se
            entry["ci95_high"] = row.probability + 1.959963984540054 * se
            errors[f"study.reliability.methods.{key}.probability"] = se
            errors[f"study.reliability.methods.{key}.ci95_low"] = se
            errors[f"study.reliability.methods.{key}.ci95_high"] = se
            errors[f"study.reliability.methods.{key}.cov"] = cov
        rows[key] = entry
    form, sorm, monte_carlo = rows["form"], rows["sorm_hr"], rows["mc"]
    stages = {record.stage: record.evaluations for record in study.records}
    first = dict(zip(sobol.input_names, sobol.first_order, strict=True))
    total = dict(zip(sobol.input_names, sobol.total_effect, strict=True))
    ranking = sobol.ranking()
    section = {
        "seed": SEED,
        "forward": {
            "samples": 4096,
            "design": "sobol",
            "quantile_level": 0.05,
            "q05": margin.quantile(0.05),
            "mean": margin.mean,
            "evaluations": stages["propagation"],
        },
        "sensitivity": {
            "base_samples": 1024,
            "top": ranking[0],
            "second": ranking[1],
            "top_total": total[ranking[0]].estimate,
            "top_first": first[ranking[0]].estimate,
            "second_total": total[ranking[1]].estimate,
            "evaluations": stages["sensitivity"],
        },
        "reliability": {
            "threshold": 0.0,
            "ci_level": 0.95,
            "methods": rows,
            "reference": reference,
            "form_relative_error": abs(form["probability"] - reference) / reference,
            "sorm_relative_error": abs(sorm["probability"] - reference) / reference,
            "mc_relative_error": abs(monte_carlo["probability"] - reference)
            / reference,
            "mc_half_width": monte_carlo["ci95_high"] - monte_carlo["probability"],
            "form_in_mc_ci": monte_carlo["ci95_low"]
            <= form["probability"]
            <= monte_carlo["ci95_high"],
            "cost_ratio": monte_carlo["evaluations"] / sorm["evaluations"],
            "methods_count": len(rows),
            "evaluations": stages["reliability"],
        },
        "risk": {
            "samples": 20_000,
            "level": 0.99,
            "loss_rate": loss_rate,
            "var99": risk.value_at_risk[0.99],
            "cvar99": risk.conditional_value_at_risk[0.99],
            "cvar95": risk.conditional_value_at_risk[0.95],
            "probability_of_loss": risk.probability_of_loss,
            "evaluations": stages["risk"],
        },
        "total_evaluations": study.total_evaluations,
        "stages": len(study.records),
    }
    # Tail measures from 20,000 Monte Carlo losses: the standard error of a
    # tail mean is the tail standard deviation over the root of the tail
    # count; it also bounds the quantile's sampling error here.
    losses = sorted(risk.losses)
    for level, keys in ((0.99, ("var99", "cvar99")), (0.95, ("cvar95",))):
        tail = losses[int(level * len(losses)) :]
        mean = sum(tail) / len(tail)
        spread = math.sqrt(sum((x - mean) ** 2 for x in tail) / (len(tail) - 1))
        for key in keys:
            errors[f"study.risk.{key}"] = spread / math.sqrt(len(tail))
    # Derived Monte Carlo quantities inherit the estimator's standard error. The
    # subset-simulation run count depends on its Markov chains (one more
    # intermediate level costs at most one level of samples), and with it the
    # stage and study totals.
    mc_se = errors["study.reliability.methods.mc.probability"]
    errors["study.reliability.mc_relative_error"] = mc_se / reference
    errors["study.reliability.mc_half_width"] = mc_se
    level_cost = float(rows["subset"]["evaluations"])
    for key in (
        "study.reliability.methods.subset.evaluations",
        "study.reliability.evaluations",
        "study.total_evaluations",
    ):
        errors[key] = level_cost / 4.0
    errors["study.risk.probability_of_loss"] = math.sqrt(
        risk.probability_of_loss * (1.0 - risk.probability_of_loss) / len(losses)
    )
    return section, errors


# --------------------------------------------------------------------------
# The mission laboratory (examples/mission_systems_lab.py)
# --------------------------------------------------------------------------


def _load_lab(name: str) -> Any:
    """Import an example laboratory module by file name.

    Args:
        name: File stem under ``examples/``.

    Returns:
        The imported module.

    References:
        [REF-PYTHON-DATAMODEL]. No mathematical equation is evaluated.

    """
    path = ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_deck_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def mission_reliability() -> dict[str, Any]:
    """Assess the lunar lander of the mission laboratory exactly.

    The lander (pump pair as a Markov chain, 2-out-of-3 computers with a
    beta-factor group, battery pair, transponder, igniter) is built by the
    laboratory's own ``component_data`` and ``lander`` functions; the
    reliability over the 720 h mission, the MTTF, minimal cut sets, and
    importance come from the binary decision diagram, with no sampling.

    Returns:
        Mission section of the deck data.

    References:
        [EQ-SYS-BDD-001], [EQ-SYS-IMP-001], [REF-BRYANT-1986],
        [REF-RAUSAND-HOYLAND-2004].

    """
    from probstat.workflow import MissionAssessment

    lab = _load_lab("mission_systems_lab")
    with contextlib.redirect_stdout(io.StringIO()):
        data = lab.component_data()
    system = lab.lander(data)
    loss_cost = 450e6
    result = MissionAssessment(
        system, MISSION_HOURS, name="lunar lander", mission_loss_cost=loss_cost
    ).run()
    reliability = result.reliability
    ranked = sorted(result.importance, key=lambda item: item.criticality, reverse=True)
    importance = {
        _slug(item.component): {
            "component": item.component,
            "criticality": item.criticality,
        }
        for item in ranked
    }
    return {
        "hours": MISSION_HOURS,
        "reliability": reliability.reliability,
        "unreliability": reliability.unreliability,
        "mttf_hours": reliability.mttf.value,
        "cut_sets": reliability.cut_set_count,
        "single_points": len(reliability.single_points_of_failure),
        "components": len(system.components)
        if hasattr(system, "components")
        else len(importance),
        "top": ranked[0].component,
        "top_criticality": ranked[0].criticality,
        "second": ranked[1].component,
        "second_criticality": ranked[1].criticality,
        "importance": importance,
        "loss_cost_musd": loss_cost / 1e6,
        "expected_loss_musd": reliability.unreliability * loss_cost / 1e6,
    }


# --------------------------------------------------------------------------
# Benchmarks (benchmarks/results/quick.json)
# --------------------------------------------------------------------------


def benchmark_highlights(root: Path = ROOT) -> dict[str, Any]:
    """Read the accuracy and cost highlights of the recorded quick benchmark.

    Args:
        root: Repository root.

    Returns:
        Benchmark section of the deck data: run metadata, the sampling-design
        comparison on the Ishigami mean, the posterior-spread errors on the
        linear-Gaussian calibration, and the FORM failure on the four-branch
        system.

    References:
        [REF-OWEN-1995], [REF-KUCUKELBIR-ETAL-2017], [REF-PROBSTAT-REPORTING].

    """
    record = json.loads(
        (root / "benchmarks" / "results" / "quick.json").read_text(encoding="utf-8")
    )
    meta, summary = record["meta"], record["summary"]

    def row(problem: str, method: str, group: str | None = None) -> Mapping[str, Any]:
        """Return the summary row of one problem, method, and group.

        References:
            [REF-PROBSTAT-REPORTING]. No equation is evaluated.

        """
        return next(
            item
            for item in summary
            if item["problem"] == problem
            and item["method"] == method
            and (group is None or item.get("group") == group)
        )

    designs = ("Monte Carlo", "Latin hypercube", "Scrambled Halton", "Scrambled Sobol'")
    sampling = {
        _slug(name): {
            "method": name,
            "error": row("ishigami-moments", name, "mean")["error"],
            "evaluations": row("ishigami-moments", name, "mean")["evaluations"],
        }
        for name in designs
    }
    samplers = (
        "Laplace",
        "Static HMC",
        "ADVI full-rank",
        "NUTS",
        "Adaptive Metropolis",
        "ADVI mean-field",
    )
    calibration = {
        _slug(name): {
            "method": name,
            "sd_error": row("linear-calibration", name, "sd")["error"],
            "evaluations": row("linear-calibration", name, "sd")["evaluations"],
        }
        for name in samplers
    }
    mcmc = [
        calibration[_slug(name)]["sd_error"]
        for name in ("Static HMC", "NUTS", "Adaptive Metropolis")
    ]
    four_branch = {
        item["method"]: item
        for item in summary
        if item["problem"] == "four-branch" and item.get("group") in (None, "Pf")
    }
    return {
        "probstat_version": meta["probstat_version"],
        "generated": meta["generated_utc"][:10],
        "problems": len(record["problems"]),
        "rows": len(summary),
        "seeds_propagation": meta["seeds"]["propagation"],
        "seeds_calibration": meta["seeds"]["calibration"],
        "sampling": sampling,
        "sampling_runs": int(sampling["monte_carlo"]["evaluations"]),
        "sampling_gain": sampling["monte_carlo"]["error"]
        / sampling["scrambled_sobol"]["error"],
        "calibration": calibration,
        "mcmc_sd_error_max": max(mcmc),
        "advi_mf_sd_error": calibration["advi_mean_field"]["sd_error"],
        "four_branch_form_ok": four_branch["FORM"]["ok"],
        "four_branch_akmcs_evaluations": four_branch["AK-MCS"]["evaluations"],
        "four_branch_akmcs_error": four_branch["AK-MCS"]["error"],
    }


# --------------------------------------------------------------------------
# The data function
# --------------------------------------------------------------------------


def build_data() -> dict[str, Any]:
    """Compute every number the deck shows.

    Returns:
        The deck data: ``meta``, ``repository``, ``study``, ``mission``,
        ``benchmark`` and ``stochastic`` (dotted key to the Monte Carlo
        standard error used to compare a fresh run with the stored file).

    References:
        [REF-PROBSTAT-REPORTING].

    """
    from probstat import __version__

    study, errors = quick_start_study()
    return {
        "meta": {"schema": SCHEMA, "probstat_version": __version__, "seed": SEED},
        "repository": repository_counts(),
        "study": study,
        "mission": mission_reliability(),
        "benchmark": benchmark_highlights(),
        "stochastic": dict(sorted(errors.items())),
    }


def lookup(data: Mapping[str, Any], key: str) -> Any:
    """Resolve a dotted key in the deck data.

    Args:
        data: Deck data.
        key: Dotted path such as ``"study.reliability.methods.form.probability"``.

    Returns:
        The stored value.

    Raises:
        KeyError: If any part of the path is missing.

    References:
        [REF-PROBSTAT-REPORTING]. No mathematical equation is evaluated.

    """
    value: Any = data
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(key)
        value = value[part]
    return value


# --------------------------------------------------------------------------
# Number formats (mirrored by docs/presentation/runtime/deck-data.mjs)
# --------------------------------------------------------------------------

_FORMAT = re.compile(r"^(int|plus|str|f(\d)|pct(\d)|x(\d)|sig(\d)|musd(\d))$")


def _group(digits: str, separator: str) -> str:
    """Insert thousands separators into a string of digits.

    References:
        [REF-PYTHON-FORMAT-SPEC]. No equation is evaluated.

    """
    return f"{int(digits):,}".replace(",", separator)


def _fixed(value: float, places: int, lang: str) -> str:
    """Format a real number with fixed decimals in the deck language.

    References:
        [REF-PYTHON-FORMAT-SPEC]. No equation is evaluated.

    """
    text = f"{value:,.{places}f}"
    if lang == "tr":
        text = text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return text.replace("-", "−")


def format_value(value: Any, fmt: str, lang: str) -> str:
    """Format one deck value the way the slides print it.

    Formats: ``int`` (thousands separators), ``plus`` (a floor, ``2,100+``),
    ``str`` (verbatim), ``f<n>`` (fixed decimals), ``pct<n>`` (a fraction as a
    percentage; Turkish puts the sign first), ``x<n>`` (a ratio, ``×286``),
    ``sig<n>`` (``n`` significant digits), ``musd<n>`` (millions of dollars).

    Args:
        value: Value from the deck data.
        fmt: Format name.
        lang: ``"en"`` or ``"tr"``.

    Returns:
        Display text.

    Raises:
        ValueError: For an unknown format or language.

    References:
        [REF-PYTHON-FORMAT-SPEC]. No mathematical equation is evaluated.

    """
    match = _FORMAT.match(fmt)
    if match is None or lang not in ("en", "tr"):
        raise ValueError(f"unknown format {fmt!r} or language {lang!r}")
    separator = "," if lang == "en" else "."
    if fmt == "str":
        return str(value)
    if fmt == "int":
        # half away from zero for positive counts, as JavaScript's Math.round
        return _group(str(math.floor(float(value) + 0.5)), separator)
    if fmt == "plus":
        return _group(str(int(value)), separator) + "+"
    places = int(next(group for group in match.groups()[1:] if group is not None))
    if fmt.startswith("f"):
        return _fixed(float(value), places, lang)
    if fmt.startswith("pct"):
        number = _fixed(100.0 * float(value), places, lang)
        return f"%{number}" if lang == "tr" else f"{number}%"
    if fmt.startswith("x"):
        return "×" + _fixed(float(value), places, lang)
    if fmt.startswith("musd"):
        number = _fixed(float(value), places, lang)
        return f"{number} M$" if lang == "tr" else f"${number}M"
    exponent = math.floor(math.log10(abs(float(value)))) if value else 0
    return _fixed(float(value), max(0, places - 1 - exponent), lang)


# --------------------------------------------------------------------------
# HTML binding
# --------------------------------------------------------------------------

BOUND = re.compile(
    r'(<(?P<tag>[a-z0-9]+)\b(?P<attrs>[^<>]*\bdata-k="(?P<key>[^"]+)"[^<>]*)>)(?P<text>[^<]*)(</(?P=tag)>)'
)


def deck_language(text: str) -> str:
    """Return the ``lang`` of a deck's ``<html>`` element.

    References:
        [REF-WHATWG-HTML]. No equation is evaluated.

    """
    match = re.search(r'<html[^>]*\blang="([a-z]+)"', text)
    if match is None:
        raise ValueError("deck has no <html lang>")
    return match.group(1)


def bound_values(text: str) -> Iterator[tuple[str, str, str]]:
    """Yield ``(key, format, text)`` of every data-bound element of a deck.

    References:
        [REF-WHATWG-HTML]. No equation is evaluated.

    """
    for match in BOUND.finditer(text):
        fmt = re.search(r'\bdata-fmt="([^"]+)"', match.group("attrs"))
        yield (
            match.group("key"),
            fmt.group(1) if fmt else "str",
            html.unescape(match.group("text")),
        )


def fill_html(text: str, data: Mapping[str, Any]) -> str:
    """Write every bound value of the deck data into a deck's HTML.

    Args:
        text: Deck HTML.
        data: Deck data.

    Returns:
        HTML whose ``data-k`` elements hold the formatted values.

    Raises:
        KeyError: If a bound key is missing from the data.

    References:
        [REF-WHATWG-HTML]. No mathematical equation is evaluated.

    """
    lang = deck_language(text)

    def replace(match: re.Match[str]) -> str:
        """Format one bound element.

        References:
            [REF-WHATWG-HTML]. No equation is evaluated.

        """
        fmt = re.search(r'\bdata-fmt="([^"]+)"', match.group("attrs"))
        value = format_value(
            lookup(data, match.group("key")), fmt.group(1) if fmt else "str", lang
        )
        return match.group(1) + html.escape(value, quote=False) + match.group(6)

    return BOUND.sub(replace, text)


# --------------------------------------------------------------------------
# Slide extraction (manifests for the sunumatik validators)
# --------------------------------------------------------------------------


class _SlideParser(HTMLParser):
    """Collect the slides of a deck with their visible copy and notes.

    References:
        [REF-WHATWG-HTML]. No equation is evaluated.

    """

    def __init__(self) -> None:
        """Start with no slides.

        References:
            [REF-WHATWG-HTML]. No equation is evaluated.

        """
        super().__init__(convert_charrefs=True)
        self.slides: list[dict[str, Any]] = []
        self._stack: list[tuple[str, dict[str, str]]] = []
        self._capture: list[tuple[str, list[str], int]] = []

    def _current(self) -> dict[str, Any] | None:
        """Return the slide being parsed.

        References:
            [REF-WHATWG-HTML]. No equation is evaluated.

        """
        inside = any(
            tag == "section" and "slayt" in attrs.get("class", "")
            for tag, attrs in self._stack
        )
        return self.slides[-1] if inside and self.slides else None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Open an element and start capturing the roles it carries.

        References:
            [REF-WHATWG-HTML]. No equation is evaluated.

        """
        values = {name: value or "" for name, value in attrs}
        if tag in ("img", "source", "br", "meta", "link", "input"):
            slide = self._current()
            if tag == "img" and slide is not None:
                slide["images"].append(
                    {"src": values.get("src", ""), "alt": values.get("alt", "")}
                )
            return
        self._stack.append((tag, values))
        classes = values.get("class", "").split()
        if tag == "section" and "slayt" in classes:
            fields = {
                name[7:]: value
                for name, value in values.items()
                if name.startswith("data-f-")
            }
            self.slides.append(
                {
                    "id": values.get("id", ""),
                    "type": values.get("data-tur", ""),
                    "breathing": values.get("data-nefes") == "true",
                    "headline": "",
                    "kicker": "",
                    "fields": fields,
                    "bullets": [],
                    "body": [],
                    "labels": [],
                    "notes": "",
                    "images": [],
                    "keys": [],
                }
            )
            return
        slide = self._current()
        if slide is None:
            return
        if "data-k" in values:
            slide["keys"].append(values["data-k"])
        role = None
        if tag in ("h1", "h2") and not slide["headline"]:
            role = "headline"
        elif "ust" in classes:
            role = "kicker"
        elif tag == "aside" and "notlar" in classes:
            role = "notes"
        elif "data-alan" in values:
            role = "alan:" + values["data-alan"]
        elif tag == "li" and "madde" in classes:
            role = "bullet"
        elif "govde" in classes:
            role = "body"
        elif "etiket" in classes:
            role = "label"
        if role is not None:
            self._capture.append((role, [], len(self._stack)))

    def handle_endtag(self, tag: str) -> None:
        """Close an element and store the text of a finished role.

        References:
            [REF-WHATWG-HTML]. No equation is evaluated.

        """
        if not self._stack:
            return
        depth = len(self._stack)
        while self._capture and self._capture[-1][2] == depth:
            role, parts, _ = self._capture.pop()
            self._store(role, " ".join("".join(parts).split()))
        self._stack.pop()

    def handle_data(self, data: str) -> None:
        """Append text to every open capture.

        References:
            [REF-WHATWG-HTML]. No equation is evaluated.

        """
        for _, parts, _ in self._capture:
            parts.append(data)

    def _store(self, role: str, text: str) -> None:
        """Store captured text on the current slide.

        References:
            [REF-WHATWG-HTML]. No equation is evaluated.

        """
        slide = self.slides[-1]
        if role == "headline":
            slide["headline"] = text
        elif role == "kicker":
            slide["kicker"] = text
        elif role == "notes":
            slide["notes"] = text
        elif role == "bullet":
            slide["bullets"].append(text)
        elif role == "body":
            slide["body"].append(text)
        elif role == "label":
            slide["labels"].append(text)
        else:
            name = role.partition(":")[2]
            if name.endswith("[]"):
                slide["fields"].setdefault(name[:-2], []).append(text)
            else:
                slide["fields"][name] = text


def extract_slides(text: str) -> list[dict[str, Any]]:
    """Parse the slides of a deck.

    Args:
        text: Deck HTML.

    Returns:
        One dictionary per ``section.slayt`` with its page type, headline,
        kicker, fields, bullets, body lines, labels, notes, images, and bound
        keys.

    References:
        [REF-WHATWG-HTML]. No mathematical equation is evaluated.

    """
    parser = _SlideParser()
    parser.feed(text)
    parser.close()
    return parser.slides


def _page_type_fields(slide: Mapping[str, Any]) -> dict[str, Any]:
    """Shape a slide's fields the way validate-page-types.mjs reads them.

    References:
        [REF-WHATWG-HTML]. No equation is evaluated.

    """
    fields: dict[str, Any] = dict(slide["fields"])
    if "adim_ad" in fields:
        fields["adimlar"] = [
            {"ad": name, "cumle": sentence}
            for name, sentence in zip(
                fields.pop("adim_ad"), fields.pop("adim_cumle", []), strict=True
            )
        ]
    if "sutun" in fields:
        fields["sutunlar"] = fields.pop("sutun")
    if "satir" in fields:
        named = fields.pop("bedel", [])
        costs = {named} if isinstance(named, str) else set(named)
        fields["satirlar"] = [
            {"olcut": name, "bedel": name in costs} for name in fields.pop("satir")
        ]
    if "kaynak_ogesi" in fields:
        fields["kaynaklar"] = fields.pop("kaynak_ogesi")
    if slide["images"] and "gorsel" not in fields:
        fields["gorsel"] = slide["images"][0]["alt"]
    return fields


def manifests(
    slides: Sequence[Mapping[str, Any]], lang: str, title: str
) -> dict[str, Any]:
    """Build the inputs of the sunumatik validators from the parsed slides.

    Args:
        slides: Output of :func:`extract_slides`.
        lang: Deck language.
        title: Deck title.

    Returns:
        ``page_types`` (validate-page-types.mjs), ``copy``
        (validate-slide-copy.mjs and validate-assertions.mjs), and ``deck``
        (validate-deck-manifest.mjs) manifests.

    References:
        [REF-PROBSTAT-REPORTING]. No mathematical equation is evaluated.

    """
    page_types = {
        "title": title,
        "profile": "speaker-led",
        "slides": [
            {
                "id": slide["id"],
                "type": slide["type"],
                "kicker": slide["kicker"],
                "headline": slide["headline"],
                "fields": _page_type_fields(slide),
                "bullets": list(slide["bullets"]),
                "notes": slide["notes"],
                **({"breathing": True} if slide["breathing"] else {}),
            }
            for slide in slides
        ],
    }
    copy = {
        "profile": "speaker-led",
        "language": lang,
        "slides": [
            {
                "id": slide["id"],
                "role": "appendix" if slide["type"] == "ek" else slide["type"],
                "kicker": slide["kicker"],
                "headline": slide["headline"],
                "body": list(slide["body"]),
                "bullets": list(slide["bullets"]),
                "labels": list(slide["labels"]),
                "notes": slide["notes"],
                "font_px": dict(FONT_PX),
            }
            for slide in slides
        ],
    }
    deck = {
        "metadata": {"title": title, "language": lang},
        "slides": [
            {
                "id": slide["id"],
                "role": slide["type"],
                "headline": slide["headline"],
                "evidence_ids": sorted(set(slide["keys"])),
                "visual_ids": [image["src"] for image in slide["images"]],
            }
            for slide in slides
        ],
    }
    return {"page_types": page_types, "copy": copy, "deck": deck}


# Type sizes of docs/presentation/deck.css (stage pixels); the copy manifest
# reports them to validate-slide-copy.mjs.
FONT_PX = {"headline": 60, "body": 34, "label": 26, "caption": 24, "citation": 24}


def evidence_ledger(data: Mapping[str, Any]) -> dict[str, Any]:
    """Build the evidence ledger of the deck's claims.

    Args:
        data: Deck data.

    Returns:
        Ledger in the shape of sunumatik's ``validate-evidence-ledger.mjs``:
        claims with their source files and keys, sources, and figures.

    References:
        [REF-PROBSTAT-REPORTING]. No mathematical equation is evaluated.

    """
    sources = [
        ("src-readme", "README quick start (the study model)", "README.md"),
        (
            "src-tool",
            "Deck data builder (fixed seed 2026)",
            "tools/build_presentation_data.py",
        ),
        ("src-bench", "Recorded quick benchmark", "benchmarks/results/quick.json"),
        (
            "src-mission",
            "Mission systems laboratory",
            "examples/mission_systems_lab.py",
        ),
        ("src-arch", "Architecture: central-model questions", "docs/architecture.md"),
        ("src-registry", "Equation registry", "docs/equation-registry.md"),
        ("src-refs", "Bibliography", "docs/references.md"),
        ("src-adr", "Architecture decision records", "docs/adr"),
        ("src-tests", "Test suite", "tests"),
        ("src-ci", "Continuous integration workflow", ".github/workflows/ci.yml"),
    ]
    study, mission, bench = data["study"], data["mission"], data["benchmark"]
    claims = [
        (
            "questions",
            f"One simulator raises {data['repository']['questions']} "
            "uncertainty questions",
            "background",
            ["src-arch"],
        ),
        (
            "forward",
            f"5% quantile of the margin is {study['forward']['q05']:.3f} kN",
            "observation",
            ["src-readme", "src-tool"],
        ),
        (
            "sobol",
            f"{study['sensitivity']['top']} has total Sobol index "
            f"{study['sensitivity']['top_total']:.2f}",
            "observation",
            ["src-tool"],
        ),
        (
            "form",
            "FORM failure probability lies inside the Monte Carlo 95% interval",
            "observation",
            ["src-tool"],
        ),
        (
            "cost",
            f"Monte Carlo used {study['reliability']['cost_ratio']:.0f} times "
            "the runs of SORM",
            "observation",
            ["src-tool"],
        ),
        (
            "cvar",
            f"CVaR at 99% is {study['risk']['cvar99']:.1f} loss units",
            "observation",
            ["src-tool"],
        ),
        (
            "calibration",
            "Mean-field ADVI misses the exact posterior sd by "
            f"{bench['advi_mf_sd_error']:.0%}",
            "observation",
            ["src-bench"],
        ),
        (
            "mission",
            f"Mission reliability over {mission['hours']:.0f} h is "
            f"{mission['reliability']:.3f}",
            "observation",
            ["src-mission", "src-tool"],
        ),
        (
            "importance",
            f"{mission['top']} has criticality importance "
            f"{mission['top_criticality']:.2f}",
            "observation",
            ["src-mission", "src-tool"],
        ),
        (
            "sampling",
            "Scrambled Sobol' points cut the Ishigami mean error "
            f"{bench['sampling_gain']:.0f}-fold",
            "observation",
            ["src-bench"],
        ),
        (
            "counts",
            "Test, equation, reference, and ADR counts are floors of repository counts",
            "observation",
            ["src-tests", "src-registry", "src-refs", "src-adr", "src-ci"],
        ),
        (
            "four-branch",
            "FORM does not solve the four-branch system benchmark",
            "limitation",
            ["src-bench"],
        ),
    ]
    return {
        "claims": [
            {
                "id": f"claim-{key}",
                "statement": statement,
                "type": kind,
                "source_ids": ids,
                "status": "verified",
                "notes": "value read from deck-data.json",
            }
            for key, statement, kind, ids in claims
        ],
        "sources": [
            {
                "id": key,
                "title": title,
                "authors_or_institution": "Stats-Lab (probstat)",
                "url": f"https://github.com/ayberkdt/Stats-Lab/blob/main/{path}",
                "accessed_at": "2026-11-12",
                "dataset_version": data["meta"]["probstat_version"],
            }
            for key, title, path in sources
        ],
        "figures": [
            {
                "id": f"figure-{name}",
                "source_id": "src-tool",
                "creator": "Stats-Lab (probstat)",
                "license": "BSD-3-Clause",
                "credit_line": "Drawn from deck-data.json",
                "transformations": [],
                "reuse_status": "verified",
            }
            for name in CHARTS
        ],
    }


# --------------------------------------------------------------------------
# Figures (Matplotlib + datawiz, slide size)
# --------------------------------------------------------------------------

CHARTS = ("reliability", "calibration", "mission", "sampling")

LABELS = {
    "en": {
        "pf": "Failure probability (%)",
        "runs": "{n} runs",
        "reference": "exact {v}",
        "sd_error": "Error of the posterior standard deviation (%)",
        "criticality": "Criticality importance (%)",
        "mean_error": "Error of the mean, {n} runs (%)",
        "methods": {
            "form": "FORM",
            "sorm_breitung": "SORM (Breitung)",
            "sorm_hr": "SORM (HR)",
            "is": "Importance sampling",
            "subset": "Subset simulation",
            "mc": "Monte Carlo",
            "laplace": "Laplace",
            "static_hmc": "Static HMC",
            "advi_full_rank": "ADVI full-rank",
            "nuts": "NUTS",
            "adaptive_metropolis": "Adaptive Metropolis",
            "advi_mean_field": "ADVI mean-field",
            "monte_carlo": "Monte Carlo",
            "latin_hypercube": "Latin hypercube",
            "scrambled_halton": "Halton",
            "scrambled_sobol": "Sobol'",
        },
        "components": {
            "transponder": "Transponder",
            "pumps": "Pump pair",
            "igniter": "Igniter",
            "cpu1": "Computer 1",
            "cpu2": "Computer 2",
            "cpu3": "Computer 3",
            "battery1": "Battery 1",
            "battery2": "Battery 2",
            "ccf_cpu1_cpu2_cpu3": "Computers, common cause",
        },
    },
    "tr": {
        "pf": "Göçme olasılığı (%)",
        "runs": "{n} koşu",
        "reference": "kesin {v}",
        "sd_error": "Sonsal standart sapmanın hatası (%)",
        "criticality": "Kritiklik önemi (%)",
        "mean_error": "Ortalamanın hatası, {n} koşu (%)",
        "methods": {
            "form": "FORM",
            "sorm_breitung": "SORM (Breitung)",
            "sorm_hr": "SORM (HR)",
            "is": "Önem örneklemesi",
            "subset": "Alt küme benzetimi",
            "mc": "Monte Carlo",
            "laplace": "Laplace",
            "static_hmc": "Statik HMC",
            "advi_full_rank": "ADVI tam kerte",
            "nuts": "NUTS",
            "adaptive_metropolis": "Uyarlamalı Metropolis",
            "advi_mean_field": "ADVI ortalama alan",
            "monte_carlo": "Monte Carlo",
            "latin_hypercube": "Latin hiperküp",
            "scrambled_halton": "Halton",
            "scrambled_sobol": "Sobol'",
        },
        "components": {
            "transponder": "Transponder",
            "pumps": "Pompa çifti",
            "igniter": "Ateşleyici",
            "cpu1": "Bilgisayar 1",
            "cpu2": "Bilgisayar 2",
            "cpu3": "Bilgisayar 3",
            "battery1": "Batarya 1",
            "battery2": "Batarya 2",
            "ccf_cpu1_cpu2_cpu3": "Bilgisayarlar, ortak neden",
        },
    },
}


def _datawiz() -> Any:
    """Import the datawiz figure toolkit or explain how to provide it.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    try:
        import datawiz
    except ImportError as error:  # pragma: no cover - depends on the workstation
        raise SystemExit(
            "--figures needs the datawiz toolkit (import datawiz); the deck data and "
            "HTML can be rebuilt without it."
        ) from error
    return datawiz


def _palette(dark: bool) -> Any:
    """Return the library chart palette for a light or dark slide.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from probstat.visualization.theme import GRAPHITE, PORCELAIN

    return GRAPHITE if dark else PORCELAIN


def _chart_reliability(
    ax: Any, data: Mapping[str, Any], lang: str, palette: Any
) -> None:
    """Dot plot of the failure probability of every method against the exact value.

    References:
        [EQ-REL-FORM-001], [REF-HASOFER-LIND-1974]. No new equation is evaluated.

    """
    words = LABELS[lang]
    rel = data["study"]["reliability"]
    order = ["form", "sorm_hr", "is", "subset", "mc"]
    focus = {"sorm_hr": palette.model, "mc": palette.observed}
    for position, key in enumerate(reversed(order)):
        row = rel["methods"][key]
        color = focus.get(key, palette.muted)
        if "ci95_low" in row:
            ax.hlines(
                position,
                100.0 * row["ci95_low"],
                100.0 * row["ci95_high"],
                color=color,
                lw=3.0,
                zorder=2,
            )
        ax.plot(
            100.0 * row["probability"],
            position,
            "o",
            ms=14,
            color=color,
            mec=palette.canvas,
            mew=2.0,
            zorder=3,
        )
        ax.annotate(
            words["runs"].format(n=format_value(row["evaluations"], "int", lang)),
            (1.01, position),
            xycoords=("axes fraction", "data"),
            va="center",
            ha="left",
            color=color,
            fontweight="bold" if key in focus else "normal",
        )
    reference = 100.0 * rel["reference"]
    ax.axvline(reference, color=palette.accent, lw=2.0, ls=(0, (3.5, 3.0)), zorder=1)
    ax.annotate(
        words["reference"].format(v=format_value(rel["reference"], "pct2", lang)),
        (reference, len(order) - 0.35),
        xytext=(10, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        color=palette.accent,
    )
    ax.set_yticks(range(len(order)), [words["methods"][key] for key in reversed(order)])
    ax.set_xlim(2.9, 5.1)
    ax.set_ylim(-0.6, len(order) + 0.1)
    ax.set_xlabel(words["pf"])


def _bars(
    ax: Any,
    names: Sequence[str],
    values: Sequence[float],
    highlight: str,
    color: str,
    palette: Any,
    places: int,
    lang: str,
) -> None:
    """Horizontal bars with one highlighted bar and value labels.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    top = max(values)
    for position, (name, value) in enumerate(zip(names, values, strict=True)):
        shade = color if name == highlight else palette.muted
        ax.barh(position, value, 0.66, color=shade, edgecolor="none")
        ax.annotate(
            _fixed(value, places, lang),
            (value + 0.012 * top, position),
            va="center",
            ha="left",
            color=shade,
            fontweight="bold" if name == highlight else "normal",
        )
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(0.0, 1.16 * top)
    ax.set_ylim(-0.6, len(names) - 0.4)


def _chart_calibration(
    ax: Any, data: Mapping[str, Any], lang: str, palette: Any
) -> None:
    """Bars of the posterior-sd error of each sampler on the exact benchmark.

    References:
        [REF-KUCUKELBIR-ETAL-2017]. No equation is evaluated.

    """
    words = LABELS[lang]
    rows = data["benchmark"]["calibration"]
    keys = sorted(rows, key=lambda key: rows[key]["sd_error"])
    names = [words["methods"][key] for key in keys]
    _bars(
        ax,
        names,
        [100.0 * rows[key]["sd_error"] for key in keys],
        words["methods"]["advi_mean_field"],
        palette.model,
        palette,
        1,
        lang,
    )
    ax.set_xlabel(words["sd_error"])


def _chart_mission(ax: Any, data: Mapping[str, Any], lang: str, palette: Any) -> None:
    """Bars of the criticality importance of the lander's blocks.

    References:
        [EQ-SYS-IMP-001], [REF-RAUSAND-HOYLAND-2004]. No equation is evaluated.

    """
    words = LABELS[lang]
    rows = data["mission"]["importance"]
    keys = sorted(rows, key=lambda key: rows[key]["criticality"])[-4:]
    names = [words["components"].get(key, rows[key]["component"]) for key in keys]
    top = words["components"].get(_slug(data["mission"]["top"]), data["mission"]["top"])
    _bars(
        ax,
        names,
        [100.0 * rows[key]["criticality"] for key in keys],
        top,
        palette.observed,
        palette,
        1,
        lang,
    )
    ax.set_xlabel(words["criticality"])


def _chart_sampling(ax: Any, data: Mapping[str, Any], lang: str, palette: Any) -> None:
    """Bars of the error of the Ishigami mean by sampling design.

    References:
        [REF-OWEN-1995], [REF-SOBOL-1967]. No equation is evaluated.

    """
    words = LABELS[lang]
    rows = data["benchmark"]["sampling"]
    keys = sorted(rows, key=lambda key: rows[key]["error"])
    names = [words["methods"][key] for key in keys]
    _bars(
        ax,
        names,
        [100.0 * rows[key]["error"] for key in keys],
        words["methods"]["scrambled_sobol"],
        palette.observed,
        palette,
        2,
        lang,
    )
    ax.set_xlabel(
        words["mean_error"].format(
            n=format_value(data["benchmark"]["sampling_runs"], "int", lang)
        )
    )


DRAW: dict[str, Callable[[Any, Mapping[str, Any], str, Any], None]] = {
    "reliability": _chart_reliability,
    "calibration": _chart_calibration,
    "mission": _chart_mission,
    "sampling": _chart_sampling,
}


def render_figures(data: Mapping[str, Any], out: Path = FIGURES) -> dict[str, Any]:
    """Draw, audit, and save every chart in both languages and both themes.

    Each chart is opened with ``datawiz.figur("sunum", hedef="sunum-tam")``
    (305 mm wide, 24 pt text, 18 pt floor), coloured with the library's
    semantic palette (``PORCELAIN`` light, ``GRAPHITE`` dark), audited with
    ``datawiz.figur_denetle(fig, ortam="sunum")``, and saved as PNG through
    ``datawiz.kaydet``, which also writes the audit report beside it.

    Args:
        data: Deck data.
        out: Output directory.

    Returns:
        Audit summary: per file, the number of errors and warnings and the
        rule names reported.

    Raises:
        SystemExit: If any audit reports an error.

    References:
        [REF-MATPLOTLIB], [REF-W3C-WCAG21-2018].

    """
    dw = _datawiz()
    import matplotlib
    from datawiz.cila import cila as polish

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    failed: list[str] = []
    for name in CHARTS:
        for lang in ("en", "tr"):
            for dark in (False, True):
                palette = _palette(dark)
                stem = f"{name}-{lang}-{'dark' if dark else 'light'}"
                fig, ax = dw.figur(
                    "sunum",
                    hedef="sunum-tam",
                    oran=0.32,
                    koyu=dark,
                    ondalik="," if lang == "tr" else ".",
                )
                plt.rcParams["savefig.facecolor"] = palette.canvas
                fig.set_facecolor(palette.canvas)
                ax.set_facecolor(palette.canvas)
                DRAW[name](ax, data, lang, palette)
                ax.grid(axis="y", visible=False)
                ax.grid(axis="x", color=palette.rule, lw=0.8)
                ax.tick_params(axis="y", length=0)
                ax.spines["left"].set_visible(False)
                # sunumatik chart-aesthetics: no glow on data marks, so the datawiz
                # dark-mode glow is off; the rest of its polish pass runs unchanged.
                polish(fig, y_adi="ust", isilti_=False)
                dw.kaydet(
                    fig,
                    str(out / stem),
                    bicimler=("png",),
                    ortam="sunum",
                    hedef="sunum-tam",
                    iz=False,
                    sessiz=True,
                )
                findings = dw.figur_denetle(
                    fig, ortam="sunum", hedef="sunum-tam", zemin=palette.canvas
                )
                errors = [
                    item for item in findings if getattr(item, "duzey", "") == "HATA"
                ]
                warnings = [
                    item for item in findings if getattr(item, "duzey", "") == "UYARI"
                ]
                summary[stem] = {
                    "errors": len(errors),
                    "warnings": len(warnings),
                    "rules": sorted(
                        {
                            getattr(item, "kural", str(item))
                            for item in errors + warnings
                        }
                    ),
                }
                if errors:
                    failed.append(stem)
                plt.close(fig)
    (out / "audit.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if failed:
        raise SystemExit("figure audit errors: " + ", ".join(failed))
    return summary


# Presentation contract (sunumatik orchestrate-science-presentation,
# references/workflow-contract.md).
DECK_PLAN: dict[str, Any] = {
    "presentation": {
        "title": "One probabilistic model answers every uncertainty question",
        "purpose": "inform",
        "venue": "library overview: talks, onboarding, README link",
        "audience": "engineers and scientists who simulate under uncertainty",
        "expertise": "mixed",
        "duration_minutes": 15,
        "density": "speaker-led",
        "language": "en, tr",
        "aspect_ratio": "16:9",
    },
    "sources": {
        "authoritative_files": [
            "docs/presentation/deck-data.json",
            "README.md",
            "benchmarks/results/quick.json",
            "examples/mission_systems_lab.py",
            "docs/architecture.md",
        ],
        "external_research_required": False,
    },
    "design": {
        "source": "generated",
        "theme": "probstat PORCELAIN (light) / GRAPHITE (dark)",
    },
    "runtime": {
        "profile": "static fixed-stage HTML, sunumatik deste-kabuk",
        "offline_required": True,
    },
    "content": {"equations": False, "data_visuals": True, "speaker_notes": True},
    "deliverables": ["html (en)", "html (tr)", "print to PDF from the runtime"],
    "validation": {
        "projector": True,
        "accessibility": "WCAG-2.2-AA",
        "evidence_traceability": True,
    },
    "risks": [
        "benchmark file recorded with an earlier release than the study",
        "repository counts shown as floors; a crossed floor requires a rebuild",
    ],
}

VISUAL_SPEC: dict[str, Any] = {
    "visuals": [
        {
            "id": "reliability",
            "question": "Which reliability method reaches the exact failure "
            "probability, at what cost?",
            "type": "dot plot with 95% intervals and a reference line",
            "units": "percent",
            "alt": "Failure probability of five methods against the exact value; "
            "SORM sits on the exact line.",
            "data_sources": ["deck-data.json: study.reliability"],
        },
        {
            "id": "calibration",
            "question": "How well do the samplers recover an exact posterior spread?",
            "type": "horizontal bars, one highlighted",
            "units": "percent",
            "alt": "Posterior standard deviation error of six methods; only "
            "mean-field ADVI is far off.",
            "data_sources": ["deck-data.json: benchmark.calibration"],
        },
        {
            "id": "mission",
            "question": "Which block causes the lander's mission failures?",
            "type": "horizontal bars, one highlighted",
            "units": "percent",
            "alt": "Criticality importance of the four most important blocks; "
            "the transponder dominates.",
            "data_sources": ["deck-data.json: mission.importance"],
        },
        {
            "id": "sampling",
            "question": "How much does the sampling design change the error "
            "at equal cost?",
            "type": "horizontal bars, one highlighted",
            "units": "percent",
            "alt": "Relative error of the Ishigami mean for four designs at equal "
            "runs; Sobol' is smallest.",
            "data_sources": ["deck-data.json: benchmark.sampling"],
        },
    ],
}


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def write_json(path: Path, value: Any) -> None:
    """Write JSON with a trailing newline.

    References:
        [REF-PROBSTAT-REPORTING]. No equation is evaluated.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Rebuild the deck data, bound HTML values, checks, and optionally charts.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code.

    References:
        [REF-PROBSTAT-REPORTING].

    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--figures", action="store_true", help="also draw and audit the slide charts"
    )
    args = parser.parse_args(argv)
    data = build_data()
    write_json(DATA_PATH, data)
    for lang, path in DECK_FILES.items():
        if not path.exists():
            continue
        text = fill_html(path.read_text(encoding="utf-8"), data)
        path.write_text(text, encoding="utf-8", newline="\n")
        title = re.search(r"<title>([^<]+)</title>", text)
        built = manifests(
            extract_slides(text), lang, html.unescape(title.group(1)) if title else ""
        )
        for name, manifest in built.items():
            write_json(CHECKS / f"{name}.{lang}.json", manifest)
    write_json(CHECKS / "evidence-ledger.json", evidence_ledger(data))
    write_json(CHECKS / "deck-plan.json", DECK_PLAN)
    write_json(CHECKS / "visual-spec.json", VISUAL_SPEC)
    if args.figures:
        summary = render_figures(data)
        print(
            f"figures: {len(summary)} files, "
            f"{sum(item['errors'] for item in summary.values())} errors"
        )
    print(f"wrote {DATA_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
