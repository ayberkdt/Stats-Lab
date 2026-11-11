"""Generate ``docs/math-reference/`` from ``docs/equation-registry.md``.

The registry is the authoritative list of equations evaluated by code. This
tool regroups it by implementation: one page per top-level ``probstat``
package lists every record whose ``- Code:`` line names that package, grouped
by the module that defines the first named object, with the equation, its
domain, meaning, code line, source keys resolved to their bibliography
entries in ``docs/references.md``, and a link back to the registry record.
An index page counts the records per package and lists every ``- Code:``
target that does not resolve. Run from the repository root::

    PYTHONPATH=src python tools/generate_math_reference.py

``tests/test_math_reference.py`` fails when the committed pages are stale.

Code targets are read from the backticked spans of a ``- Code:`` line:

1. A dotted path whose first segment is a top-level ``probstat`` package,
   such as ``distributions.Normal.quantile``, optionally followed by a call
   argument list, such as ``sampling.sobol(scramble=True)``. It must import:
   the longest importable module prefix is imported and the remaining
   segments are attributes, dataclass fields, or annotated attributes.
2. A member shorthand ``Name.member`` whose ``Name`` is a segment of an
   earlier target of the same record, such as ``LogNormal.sf`` after
   ``distributions.LogNormal.cdf``, or ``.member`` after any target. It is
   expanded against that target and must import. A ``Name.member`` span that
   cannot be expanded is reported as unqualified.
3. A laboratory object ``examples.<lab>.<name>``, which must be defined at
   the top level of ``examples/<lab>.py``, and a repository file
   ``tests/<file>.py``, ``examples/<file>.py`` or ``benchmarks/<file>.py``,
   optionally with ``::<name>``, which must exist.

Every other backticked span (bare identifiers, argument values, formulas) is
descriptive. A target with a segment that starts with ``_`` is private; it
must still import, and the index lists it. Each record needs at least one
target. A record whose targets are all repository targets is listed on the
laboratories-and-tests page.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import pkgutil
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "equation-registry.md"
REFERENCES = ROOT / "docs" / "references.md"
OUTPUT = ROOT / "docs" / "math-reference"
REPOSITORY_PAGE = "laboratories-and-tests"
FIELDS = ("Domain", "Meaning", "Code", "Source")
RECORD_HEADING = re.compile(r"^### (EQ-[A-Z0-9-]+) — (.+)$")
FIELD_LINE = re.compile(r"^- (Domain|Meaning|Code|Source):(.*)$")
REFERENCE_KEY = re.compile(r"\bREF-[A-Z0-9]+(?:-[A-Z0-9]+)*\b")
SPAN = re.compile(r"`([^`]+)`")
QUALIFIED = re.compile(r"^([a-z_][a-z0-9_]*)((?:\.[A-Za-z_][A-Za-z0-9_]*)+)$")
MEMBER = re.compile(r"^([A-Z][A-Za-z0-9_]*)((?:\.[A-Za-z_][A-Za-z0-9_]*)+)$")
LEADING_DOT = re.compile(r"^\.([A-Za-z_][A-Za-z0-9_]*)$")
LAB_OBJECT = re.compile(r"^examples\.([A-Za-z0-9_]+)\.([A-Za-z_][A-Za-z0-9_]*)$")
REPOSITORY_FILE = re.compile(
    r"^((?:tests|examples|benchmarks)/[A-Za-z0-9_/]+\.py)(?:::([A-Za-z_]\w*))?$"
)
CALL_SUFFIX = re.compile(r"\(.*\)$")


@dataclass(frozen=True)
class Record:
    """One ``### EQ-*`` record of the equation registry."""

    identifier: str
    title: str
    equation: str
    fields: tuple[tuple[str, str], ...]

    def field(self, name: str) -> str:
        """Return the raw text of one field, continuation lines included.

        Args:
            name: One of ``Domain``, ``Meaning``, ``Code``, ``Source``.

        Returns:
            The field text after its label, with line breaks kept.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        return dict(self.fields)[name]


@dataclass(frozen=True)
class Target:
    """One resolved or unresolved code target of a record."""

    written: str
    path: str
    kind: str
    package: str
    module: str
    private: bool
    problem: str


def anchor(text: str) -> str:
    """Return the GitHub heading anchor for ``text``.

    Args:
        text: Heading text without the leading hashes.

    Returns:
        Lower-case text without punctuation, spaces replaced by hyphens.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated; the rule
        matches ``tools/generate_api_reference.py``.

    """
    kept = "".join(c for c in text.lower() if c.isalnum() or c in " -_")
    return kept.replace(" ", "-")


def parse_registry(text: str) -> list[Record]:
    """Return the records of the equation registry in file order.

    Args:
        text: Contents of ``docs/equation-registry.md``.

    Returns:
        Records with their equation text and the four fields.

    Raises:
        ValueError: If a record lacks a field, repeats one, orders them
            differently from Domain, Meaning, Code, Source, or reuses an
            identifier.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    records: list[Record] = []
    current: list[str] | None = None
    heading: re.Match[str] | None = None
    blocks: list[tuple[re.Match[str], list[str]]] = []
    for line in text.splitlines():
        match = RECORD_HEADING.match(line)
        if match is not None:
            current = []
            heading = match
            blocks.append((heading, current))
        elif line.startswith(("## ", "### ")):
            current = None
        elif current is not None:
            current.append(line)
    seen: set[str] = set()
    for match, lines in blocks:
        identifier, title = match.group(1), match.group(2).strip()
        if identifier in seen:
            raise ValueError(f"{identifier} is defined twice")
        seen.add(identifier)
        equation: list[str] = []
        fields: list[tuple[str, list[str]]] = []
        for line in lines:
            field = FIELD_LINE.match(line)
            if field is not None:
                fields.append((field.group(1), [field.group(2).strip()]))
            elif fields:
                fields[-1][1].append(line)
            else:
                equation.append(line)
        names = tuple(name for name, _ in fields)
        if names != FIELDS:
            raise ValueError(f"{identifier}: fields {names}, expected {FIELDS}")
        records.append(
            Record(
                identifier=identifier,
                title=title,
                equation="\n".join(equation).strip(),
                fields=tuple((name, "\n".join(body).strip()) for name, body in fields),
            )
        )
    return records


def parse_references(text: str) -> dict[str, str]:
    """Return each ``REF-*`` key of the bibliography with its entry text.

    Args:
        text: Contents of ``docs/references.md``.

    Returns:
        Mapping from key to the entry paragraph joined onto one line.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    entries: dict[str, str] = {}
    for block in re.split(r"(?m)^## ", text)[1:]:
        key, _, body = block.partition("\n")
        entries[key.strip()] = " ".join(body.split())
    return entries


def _packages() -> tuple[str, ...]:
    """Return the top-level public packages of ``probstat``.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    import probstat

    return tuple(
        sorted(
            info.name
            for info in pkgutil.iter_modules(probstat.__path__)
            if info.ispkg and not info.name.startswith("_")
        )
    )


def _resolve_object(path: str) -> tuple[str, str]:
    """Import ``probstat.<path>`` and return its defining module or a problem.

    Args:
        path: Dotted path below ``probstat``.

    Returns:
        ``(module, "")`` with the dotted name of the module that defines the
        object (for attributes and fields, the module of their owner), or
        ``("", problem)`` when the path does not resolve.

    References:
        [REF-PYTHON-DATAMODEL]. No mathematical equation is evaluated.

    """
    parts = path.split(".")
    value: object = None
    used = 0
    for count in range(len(parts), 0, -1):
        try:
            value = importlib.import_module("probstat." + ".".join(parts[:count]))
        except ImportError:
            continue
        used = count
        break
    if used == 0:
        return "", f"no module probstat.{parts[0]}"
    home = "probstat." + ".".join(parts[:used])
    for name in parts[used:]:
        if value is not None and hasattr(value, name):
            value = getattr(value, name)
        elif inspect.isclass(value) and _declares(value, name):
            value = None
        else:
            return "", f"{name!r} not found in {home}"
        defined = getattr(value, "__module__", None)
        if isinstance(value, ModuleType):
            home = value.__name__
        elif (inspect.isclass(value) or inspect.isroutine(value)) and str(
            defined
        ).startswith("probstat."):
            home = str(defined)
    return home, ""


def _declares(owner: type, name: str) -> bool:
    """Return whether a class declares ``name`` as a field or annotation.

    References:
        [REF-PYTHON-DATAMODEL]. No mathematical equation is evaluated.

    """
    if dataclasses.is_dataclass(owner):
        if name in {field.name for field in dataclasses.fields(owner)}:
            return True
    return any(name in getattr(base, "__annotations__", {}) for base in owner.__mro__)


def _top_level_names(path: Path) -> set[str]:
    """Return the names defined at the top level of a Python file.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _repository_target(written: str, file: str, name: str | None) -> Target:
    """Check a repository file target, optionally naming a top-level object.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    path = ROOT / file
    problem = ""
    if not path.is_file():
        problem = f"{file} does not exist"
    elif name is not None and name not in _top_level_names(path):
        problem = f"{name!r} is not defined in {file}"
    label = file if name is None else f"{file}::{name}"
    private = name is not None and name.startswith("_")
    return Target(written, label, "repository", "", "", private, problem)


def code_targets(record: Record, packages: tuple[str, ...]) -> list[Target]:
    """Return the code targets of one record, each resolved or with a problem.

    Args:
        record: Registry record.
        packages: Top-level ``probstat`` package names.

    Returns:
        Targets in the order they are written on the ``- Code:`` line.

    References:
        [REF-PYTHON-DATAMODEL]. No mathematical equation is evaluated.

    """
    targets: list[Target] = []
    earlier: list[str] = []
    for span in SPAN.findall(record.field("Code")):
        written = CALL_SUFFIX.sub("", span.strip())
        lab = LAB_OBJECT.match(written)
        file = REPOSITORY_FILE.match(written)
        if lab is not None:
            targets.append(
                _repository_target(span, f"examples/{lab.group(1)}.py", lab.group(2))
            )
            continue
        if file is not None:
            targets.append(_repository_target(span, file.group(1), file.group(2)))
            continue
        path = _expand(written, earlier, packages)
        if path is None:
            if MEMBER.match(written):
                targets.append(
                    Target(span, written, "object", "", "", False, "unqualified")
                )
            continue
        earlier.append(path)
        module, problem = _resolve_object(path)
        private = any(part.startswith("_") for part in path.split("."))
        package = path.split(".")[0]
        targets.append(Target(span, path, "object", package, module, private, problem))
    return targets


def _expand(written: str, earlier: list[str], packages: tuple[str, ...]) -> str | None:
    """Return the full dotted path of a span, or ``None`` if it is descriptive.

    References:
        [REF-PYTHON-DATAMODEL]. No mathematical equation is evaluated.

    """
    qualified = QUALIFIED.match(written)
    if qualified is not None and qualified.group(1) in packages:
        return written
    leading = LEADING_DOT.match(written)
    if leading is not None and earlier:
        return earlier[-1].rsplit(".", 1)[0] + "." + leading.group(1)
    member = MEMBER.match(written)
    if member is not None:
        head = member.group(1)
        for previous in reversed(earlier):
            parts = previous.split(".")
            if head in parts:
                return ".".join(parts[: parts.index(head) + 1]) + member.group(2)
    return None


@dataclass(frozen=True)
class Entry:
    """A record with its targets and the page and module that hold it."""

    record: Record
    targets: tuple[Target, ...]
    page: str
    module: str
    others: tuple[str, ...]


def build_entries() -> list[Entry]:
    """Parse the registry and place every record on exactly one page.

    Returns:
        One entry per record in registry order.

    References:
        [REF-PYTHON-DATAMODEL]. No mathematical equation is evaluated.

    """
    packages = _packages()
    entries: list[Entry] = []
    for record in parse_registry(REGISTRY.read_text(encoding="utf-8")):
        targets = tuple(code_targets(record, packages))
        objects = [t for t in targets if t.kind == "object" and t.package]
        if objects:
            page = objects[0].package
            module = objects[0].module or f"probstat.{page}"
        else:
            page = REPOSITORY_PAGE
            module = targets[0].path.split("::")[0] if targets else "no target"
        others = tuple(
            sorted({t.package for t in objects if t.package and t.package != page})
        )
        entries.append(Entry(record, targets, page, module, others))
    return entries


def problems(entries: list[Entry]) -> list[tuple[str, str, str]]:
    """Return ``(record, target, problem)`` for every unresolved target.

    A record without any target is reported with an empty target.

    References:
        [REF-PYTHON-DATAMODEL]. No mathematical equation is evaluated.

    """
    found: list[tuple[str, str, str]] = []
    for entry in entries:
        if not entry.targets:
            found.append((entry.record.identifier, "", "no code target"))
        for target in entry.targets:
            if target.problem:
                found.append((entry.record.identifier, target.written, target.problem))
    return found


def _source_lines(record: Record, bibliography: dict[str, str]) -> list[str]:
    """Return the Source field followed by one line per cited entry.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    source = record.field("Source")
    lines = [f"- Source: {source}"]
    keys = list(dict.fromkeys(REFERENCE_KEY.findall(source)))
    for key in keys:
        citation = bibliography.get(key)
        if citation is None:
            raise ValueError(f"{record.identifier}: {key} is not in references.md")
        lines.append(f"  - [{key}](../references.md#{anchor(key)}): {citation}")
    return lines


def _entry_markdown(entry: Entry, bibliography: dict[str, str]) -> str:
    """Return the Markdown block of one record on its package page.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    record = entry.record
    heading = f"{record.identifier} — {record.title}"
    lines = [f"### {heading}", ""]
    if record.equation:
        lines += [record.equation, ""]
    for name in ("Domain", "Meaning", "Code"):
        lines.append(f"- {name}: {record.field(name)}")
    lines += _source_lines(record, bibliography)
    if entry.others:
        named = ", ".join(f"[`{other}`]({other}.md)" for other in entry.others)
        lines.append(f"- Also names: {named}")
    lines.append(
        f"- Registry: [{record.identifier}](../equation-registry.md#{anchor(heading)})"
    )
    return "\n".join(lines) + "\n"


def _count(number: int, noun: str) -> str:
    """Return ``number`` followed by ``noun``, pluralized with ``s``.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _page(
    name: str,
    entries: list[Entry],
    shared: list[Entry],
    bibliography: dict[str, str],
) -> str:
    """Return one package page (or the laboratories-and-tests page).

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    modules: dict[str, list[Entry]] = {}
    for entry in entries:
        modules.setdefault(entry.module, []).append(entry)
    if name == REPOSITORY_PAGE:
        title = "Laboratories and tests"
        blurb = (
            "Records whose `- Code:` line names only laboratory scripts or "
            "test modules: worked models of the laboratories and closed forms "
            "used as test oracles, grouped by the first file named."
        )
    else:
        title = f"`probstat.{name}`"
        blurb = (
            f"Records whose first code target lies in `probstat.{name}`, "
            "grouped by the module that defines that target."
        )
    lines = [
        f"# {title} equations",
        "",
        f"Generated by `tools/generate_math_reference.py` from the "
        f"[equation registry](../equation-registry.md); "
        f"{_count(len(entries), 'record')} in "
        f"{_count(len(modules), 'file' if name == REPOSITORY_PAGE else 'module')}. "
        f"{blurb} Back to the [index](index.md).",
        "",
    ]
    for module in sorted(modules):
        count = len(modules[module])
        lines.append(f"- [`{module}`](#{anchor(module)}): {count}")
    if shared:
        shared_heading = "Records placed on other pages"
        lines.append(f"- [{shared_heading}](#{anchor(shared_heading)}): {len(shared)}")
    lines.append("")
    for module in sorted(modules):
        lines += [f"## `{module}`", ""]
        for entry in modules[module]:
            lines.append(_entry_markdown(entry, bibliography))
    if shared:
        lines += [
            "## Records placed on other pages",
            "",
            f"These records also name `probstat.{name}` in their `- Code:` line.",
            "",
        ]
        for entry in shared:
            record = entry.record
            target = anchor(f"{record.identifier} — {record.title}")
            lines.append(
                f"- [{record.identifier}]({entry.page}.md#{target}): {record.title}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _index(entries: list[Entry], pages: dict[str, list[Entry]]) -> str:
    """Return the index page with counts and the resolution report.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    shared: dict[str, int] = {}
    for entry in entries:
        for other in entry.others:
            shared[other] = shared.get(other, 0) + 1
    order = sorted(pages, key=lambda name: (-len(pages[name]), name))
    top = order[0]
    targets = [t for e in entries for t in e.targets]
    objects = [t for t in targets if t.kind == "object"]
    lines = [
        "# Mathematical reference",
        "",
        "Generated by `tools/generate_math_reference.py` from the "
        f"[equation registry](../equation-registry.md): {len(entries)} records, "
        f"{len(objects)} code objects and "
        f"{len(targets) - len(objects)} repository targets. Each record "
        "appears on exactly one page: that of the package of the first object "
        "its `- Code:` line names. Every other package it names lists it under "
        "*Records placed on other pages*. Source keys resolve in "
        "[references.md](../references.md). Regenerate with "
        "`PYTHONPATH=src python tools/generate_math_reference.py`; "
        "`tests/test_math_reference.py` fails when these pages are stale.",
        "",
        "## Records per package",
        "",
        f"`{top}` holds the most records ({len(pages[top])} of {len(entries)}).",
        "",
        "| Page | Records | Modules | Also named by |",
        "|---|---:|---:|---:|",
    ]
    for name in order:
        modules = len({entry.module for entry in pages[name]})
        label = "Laboratories and tests" if name == REPOSITORY_PAGE else f"`{name}`"
        lines.append(
            f"| [{label}]({name}.md) | {len(pages[name])} | {modules} | "
            f"{shared.get(name, 0)} |"
        )
    lines += [
        "",
        "*Records* are placed on the page; *Modules* group them; *Also named "
        "by* counts records placed on another page whose `- Code:` line also "
        "names the package.",
    ]
    unresolved = problems(entries)
    lines += ["", "## Unresolved code targets", ""]
    if unresolved:
        for identifier, target, problem in unresolved:
            shown = f"`{target}`: " if target else ""
            lines.append(f"- {identifier}: {shown}{problem}")
    else:
        lines.append(
            "None: every target of every `- Code:` line resolves to an "
            "importable object or an existing repository file."
        )
    private = [
        (entry.record.identifier, target.path)
        for entry in entries
        for target in entry.targets
        if target.private
    ]
    lines += [
        "",
        "## Private targets",
        "",
        "Targets with a segment that starts with `_`. They import, but they "
        "are implementation details rather than public API.",
        "",
    ]
    lines += [f"- {identifier}: `{path}`" for identifier, path in private] or ["None."]
    return "\n".join(lines).rstrip() + "\n"


def render() -> dict[str, str]:
    """Return every page of the mathematical reference by file name.

    Returns:
        Mapping from file name in ``docs/math-reference/`` to its contents.

    Raises:
        ValueError: If the registry is malformed or cites an unknown key.

    References:
        [REF-PYTHON-DATAMODEL]. No mathematical equation is evaluated.

    """
    entries = build_entries()
    bibliography = parse_references(REFERENCES.read_text(encoding="utf-8"))
    pages: dict[str, list[Entry]] = {}
    for entry in entries:
        pages.setdefault(entry.page, []).append(entry)
    files = {"index.md": _index(entries, pages)}
    for name, placed in sorted(pages.items()):
        shared = [entry for entry in entries if name in entry.others]
        files[f"{name}.md"] = _page(name, placed, shared, bibliography)
    return files


def main() -> None:
    """Write ``docs/math-reference/`` and remove pages that are no longer made.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    files = render()
    OUTPUT.mkdir(exist_ok=True)
    for stale in sorted(OUTPUT.glob("*.md")):
        if stale.name not in files:
            stale.unlink()
    for name, text in files.items():
        (OUTPUT / name).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {len(files)} pages to {OUTPUT}")


if __name__ == "__main__":
    main()
