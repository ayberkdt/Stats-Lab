"""Run the Python examples of the Markdown documentation and check what they claim.

The engine behind ``tests/test_documentation_examples.py`` (ADR-0095). It
reads the fenced code blocks of a Markdown file, executes every ````python``
block in document order in one namespace per file, and compares the values
the text quotes with the values the code produces. Two forms of quotation
are recognized:

* **Inline values.** A top-level expression statement (or a ``print`` of one
  argument, or an assignment to one name) whose trailing comment *starts*
  with a Python literal, as in ``methods.row("FORM").probability  # 0.0362``
  or ``sobol.ranking()  # ('load', 'strength')``. The literal is the longest
  prefix of the comment that :func:`ast.literal_eval` accepts and that ends
  before a space, comma, semicolon, or the end of the comment; prose after
  it is ignored. A ``print`` followed by comment-only lines that end the
  block (or a paragraph of it) is compared with those lines as its standard
  output; an expression followed by comment-only lines is compared with them
  as its value (a leading literal) or its ``repr``. Line breaks inside such
  comments count as spaces.
* **Output blocks.** A ````text`` block that follows a Python block with only
  white space between them holds the standard output of that block.

Numbers match *to the shown precision*: a quoted ``0.124`` accepts any value
in ``[0.1235, 0.1245]`` and a quoted ``49`` any value within ``0.5``.
Everything else (strings, names, punctuation) must match exactly, with runs
of horizontal white space in output blocks treated as one space.

A block preceded by ``<!-- not-executed: reason -->`` (only white space in
between) is skipped and reported as an exemption. Each file runs in its own
temporary working directory, so files written by an example never land in
the repository, and figures use Matplotlib's non-interactive ``Agg``
backend.

References:
    [REF-PYTHON-AST] (statement parsing and literal evaluation),
    [REF-PYTHON-UNITTEST] (the test runner that reports the failures).

"""

from __future__ import annotations

import ast
import contextlib
import functools
import io
import math
import numbers
import os
import re
import sys
import tempfile
import tokenize
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
DOCS = REPOSITORY_ROOT / "docs"
FENCE = re.compile(r"^```([\w+-]*)[ \t]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
EXEMPTION = re.compile(r"<!--\s*not-executed:\s*(.*?)\s*-->\s*\Z", re.DOTALL)
NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w.])")
NUMBER_TEXT = re.compile(r"[-+]?(\d*)(?:\.(\d*))?(?:[eE]([-+]?\d+))?")
HORIZONTAL_SPACE = re.compile(r"[ \t]+")
LITERAL_END = frozenset(" \t,;")
CHECKED_LANGUAGE = "python"
OUTPUT_LANGUAGE = "text"


def documented_files() -> tuple[Path, ...]:
    """Return every Markdown file whose Python examples are executed.

    Returns:
        ``README.md``, the foundation and prerequisite guides, and the
        learning path, in a stable order.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    return (
        REPOSITORY_ROOT / "README.md",
        *sorted((DOCS / "foundations").glob("*.md")),
        *sorted((DOCS / "prerequisites").glob("*.md")),
        DOCS / "learning-path.md",
    )


@dataclass(frozen=True, slots=True)
class CodeBlock:
    """One fenced code block of a Markdown file.

    Attributes:
        language: Info string of the fence (``python``, ``text``, ...).
        body: Text between the fences.
        line: One-based line number of the first line of the body.
        exemption: Reason given by a preceding ``not-executed`` comment, or
            ``None`` when the block is executed.
        output: Body of a ````text`` block that directly follows a Python
            block, or ``None``.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """

    language: str
    body: str
    line: int
    exemption: str | None
    output: str | None


def code_blocks(text: str) -> tuple[CodeBlock, ...]:
    """Split Markdown text into its fenced code blocks.

    Args:
        text: Markdown source.

    Returns:
        Every fenced block in document order, with its exemption reason and
        documented output attached.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    matches = list(FENCE.finditer(text))
    blocks: list[CodeBlock] = []
    previous_end = 0
    for index, match in enumerate(matches):
        before = text[previous_end : match.start()]
        exemption = EXEMPTION.search(before)
        output = None
        if index + 1 < len(matches):
            following = matches[index + 1]
            gap = text[match.end() : following.start()]
            if following.group(1) == OUTPUT_LANGUAGE and not gap.strip():
                output = following.group(2)
        blocks.append(
            CodeBlock(
                language=match.group(1),
                body=match.group(2),
                line=text.count("\n", 0, match.start()) + 2,
                exemption=exemption.group(1) if exemption else None,
                output=output,
            )
        )
        previous_end = match.end()
    return tuple(blocks)


def literal_prefix(comment: str) -> str | None:
    """Return the longest leading Python literal of a comment, if any.

    Args:
        comment: Comment text without the ``#``.

    Returns:
        Source of the literal, or ``None`` when the comment is prose.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    text = comment.strip()
    for end in range(len(text), 0, -1):
        if end < len(text) and text[end] not in LITERAL_END:
            continue
        candidate = text[:end].rstrip()
        if not candidate or candidate.endswith(","):
            continue
        try:
            ast.literal_eval(candidate)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        return candidate
    return None


def number_tolerance(text: str) -> float:
    """Return half a unit in the last shown digit of a number.

    Args:
        text: Number as written, for example ``0.0362`` or ``1.2e-05``.

    Returns:
        Half the spacing of the last displayed digit.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    match = NUMBER_TEXT.fullmatch(text.strip().replace("_", ""))
    if match is None:
        return 0.0
    decimals = len(match.group(2) or "")
    exponent = int(match.group(3) or 0)
    return 0.5 * 10.0 ** (exponent - decimals)


def _number_matches(actual: object, expected: float, shown: str) -> bool:
    """Compare a value with a quoted number at the quoted precision.

    Args:
        actual: Value produced by the code.
        expected: Quoted number.
        shown: The quoted number as written.

    Returns:
        Whether the value rounds to the quoted number.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    if isinstance(actual, bool) or not isinstance(actual, numbers.Real):
        try:
            value = float(actual)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
    else:
        value = float(actual)
    if math.isnan(expected):
        return math.isnan(value)
    if math.isinf(expected):
        return value == expected
    tolerance = number_tolerance(shown)
    return abs(value - expected) <= tolerance * (1.0 + 1e-9) + 1e-15 * abs(expected)


def value_mismatch(actual: object, node: ast.expr, source: str) -> str | None:
    """Compare a value with a quoted literal, element by element.

    Args:
        actual: Value produced by the code.
        node: Parsed quoted literal.
        source: Source text of the quoted literal (for the shown precision).

    Returns:
        ``None`` when they match, otherwise a short description of the first
        difference.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    shown = ast.get_source_segment(source, node) or ""
    expected = ast.literal_eval(node)
    if isinstance(node, ast.Tuple | ast.List):
        if isinstance(actual, str | bytes) or not isinstance(actual, Sequence):
            try:
                actual = tuple(actual)  # type: ignore[call-overload]
            except TypeError:
                return f"expected a sequence {shown}, got {actual!r}"
        if len(actual) != len(node.elts):
            return f"expected {len(node.elts)} items {shown}, got {actual!r}"
        for item, element in zip(actual, node.elts, strict=True):
            problem = value_mismatch(item, element, source)
            if problem is not None:
                return problem
        return None
    if isinstance(node, ast.Dict):
        if not isinstance(actual, Mapping):
            return f"expected a mapping {shown}, got {actual!r}"
        keys = [ast.literal_eval(key) for key in node.keys if key is not None]
        if set(keys) != set(actual):
            return f"expected keys {sorted(map(repr, keys))}, got {list(actual)!r}"
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            assert key_node is not None
            problem = value_mismatch(
                actual[ast.literal_eval(key_node)], value_node, source
            )
            if problem is not None:
                return problem
        return None
    if isinstance(expected, bool) or expected is None:
        return None if actual == expected else f"expected {shown}, got {actual!r}"
    if isinstance(expected, int | float):
        if _number_matches(actual, float(expected), shown):
            return None
        return f"expected {shown} (to the shown precision), got {actual!r}"
    if isinstance(expected, set | frozenset):
        return None if set(actual) == expected else f"expected {shown}, got {actual!r}"  # type: ignore[call-overload]
    if actual == expected or str(actual) == expected:
        return None
    return f"expected {shown}, got {actual!r}"


def output_mismatch(actual: str, expected: str) -> str | None:
    """Compare printed output with a documented output block.

    Numbers are compared to the precision shown in ``expected``; all other
    text must agree exactly, line by line, after runs of spaces and tabs are
    collapsed and trailing white space is removed.

    Args:
        actual: Captured standard output.
        expected: Documented output.

    Returns:
        ``None`` when they match, otherwise the first differing line.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    actual_lines = [line.rstrip() for line in actual.rstrip().splitlines()]
    expected_lines = [line.rstrip() for line in expected.rstrip().splitlines()]
    if len(actual_lines) != len(expected_lines):
        return (
            f"{len(expected_lines)} documented lines, {len(actual_lines)} printed:\n"
            + "\n".join(actual_lines)
        )
    for number, (got, want) in enumerate(
        zip(actual_lines, expected_lines, strict=True), start=1
    ):
        got_parts = NUMBER.split(HORIZONTAL_SPACE.sub(" ", got))
        want_parts = NUMBER.split(HORIZONTAL_SPACE.sub(" ", want))
        got_numbers = NUMBER.findall(HORIZONTAL_SPACE.sub(" ", got))
        want_numbers = NUMBER.findall(HORIZONTAL_SPACE.sub(" ", want))
        same = got_parts == want_parts and len(got_numbers) == len(want_numbers)
        if same:
            same = all(
                _number_matches(float(g), float(w), w)
                for g, w in zip(got_numbers, want_numbers, strict=True)
            )
        if not same:
            return f"line {number}: documented {want!r}, printed {got!r}"
    return None


@dataclass(frozen=True, slots=True)
class _Check:
    """One quoted value attached to a top-level statement.

    Attributes:
        kind: ``"value"`` for an inline literal, ``"stdout"`` for comment
            lines that show what a ``print`` writes.
        literal: Quoted literal source or the quoted output text.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """

    kind: str
    literal: str


def _comments(source: str) -> dict[int, tuple[int, str]]:
    """Map line numbers to their comment (column and text without ``#``).

    Args:
        source: Python source.

    Returns:
        One entry per line that carries a comment.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    found: dict[int, tuple[int, str]] = {}
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    with contextlib.suppress(tokenize.TokenError, IndentationError):
        for token in tokens:
            if token.type == tokenize.COMMENT:
                found[token.start[0]] = (token.start[1], token.string[1:])
    return found


def _statement_check(
    statement: ast.stmt, lines: list[str], comments: dict[int, tuple[int, str]]
) -> _Check | None:
    """Return the quoted value attached to one top-level statement.

    Args:
        statement: Parsed top-level statement.
        lines: Source lines of the block.
        comments: Comments of the block by line number.

    Returns:
        The check to perform, or ``None`` when nothing is quoted.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    end = statement.end_lineno or statement.lineno
    is_print = (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "print"
    )
    shown: list[str] = []
    following = end + 1
    while following <= len(lines) and lines[following - 1].lstrip().startswith("#"):
        text = lines[following - 1].lstrip()[1:]
        shown.append(text[1:] if text.startswith(" ") else text)
        following += 1
    if is_print and shown:
        ends_section = following > len(lines) or not lines[following - 1].strip()
        if ends_section:
            return _Check("stdout", "\n".join(shown))
    elif shown and isinstance(statement, ast.Expr) and end not in comments:
        return _Check("display", "\n".join(shown))
    checkable = isinstance(statement, ast.Expr) or (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    )
    if not checkable or end not in comments:
        return None
    column, text = comments[end]
    if column < (statement.end_col_offset or 0):
        return None
    literal = literal_prefix(text)
    return None if literal is None else _Check("value", literal)


@dataclass(slots=True)
class DocumentRun:
    """The outcome of running every Python block of one Markdown file.

    Attributes:
        path: The Markdown file.
        executed: Number of Python blocks executed.
        exemptions: ``(line, reason)`` of every skipped Python block.
        value_checks: Number of inline values compared.
        output_checks: Number of output blocks and printed comments compared.
        failures: One message per exception or mismatch.
        outputs: ``(printed, documented)`` per executed block; ``documented``
            is ``None`` when the block has no output block.
        namespace: The shared namespace after the last block.
        missing_module: Name of an optional module the examples need and the
            environment lacks, or ``None``.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """

    path: Path
    executed: int = 0
    exemptions: list[tuple[int, str]] = field(default_factory=list)
    value_checks: int = 0
    output_checks: int = 0
    failures: list[str] = field(default_factory=list)
    outputs: list[tuple[str, str | None]] = field(default_factory=list)
    namespace: dict[str, object] = field(default_factory=dict)
    missing_module: str | None = None


OPTIONAL_MODULES = frozenset({"matplotlib", "numpy"})


def _run_block(block: CodeBlock, run: DocumentRun, label: str) -> bool:
    """Execute one Python block statement by statement and check its quotes.

    Args:
        block: The block.
        run: Accumulated outcome of the file; updated in place.
        label: File name used in messages and tracebacks.

    Returns:
        Whether execution may continue with the next block.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    where = f"{label}:{block.line}"
    try:
        tree = ast.parse(block.body, filename=label)
    except SyntaxError as error:
        run.failures.append(f"{where}: syntax error: {error}")
        return False
    ast.increment_lineno(tree, block.line - 1)
    lines = [""] * (block.line - 1) + block.body.splitlines()
    comments = {
        number + block.line - 1: value
        for number, value in _comments(block.body).items()
    }
    block_output = io.StringIO()
    for statement in tree.body:
        check = _statement_check(statement, lines, comments)
        statement_output = io.StringIO()
        value: object = None
        try:
            with contextlib.redirect_stdout(statement_output):
                if check is not None and isinstance(statement, ast.Expr):
                    target = statement.value
                    if check.kind == "value" and isinstance(target, ast.Call) and (
                        isinstance(target.func, ast.Name) and target.func.id == "print"
                    ):
                        if len(target.args) == 1 and not target.keywords:
                            target = target.args[0]
                            value = eval(  # noqa: S307
                                compile(ast.Expression(target), label, "eval"),
                                run.namespace,
                            )
                            print(value)
                        else:
                            exec(  # noqa: S102
                                compile(ast.Module([statement], []), label, "exec"),
                                run.namespace,
                            )
                    else:
                        value = eval(  # noqa: S307
                            compile(ast.Expression(target), label, "eval"),
                            run.namespace,
                        )
                else:
                    exec(  # noqa: S102
                        compile(ast.Module([statement], []), label, "exec"),
                        run.namespace,
                    )
        except ModuleNotFoundError as error:
            block_output.write(statement_output.getvalue())
            if error.name and error.name.split(".")[0] in OPTIONAL_MODULES:
                run.missing_module = error.name.split(".")[0]
            else:
                run.failures.append(f"{where}: {error!r}")
            return False
        except Exception:  # noqa: BLE001 - every failure is reported
            block_output.write(statement_output.getvalue())
            trace = traceback.format_exc(limit=-3)
            run.failures.append(
                f"{where}: statement at line {statement.lineno}\n{trace}"
            )
            return False
        printed = statement_output.getvalue()
        block_output.write(printed)
        if check is None:
            continue
        line = statement.end_lineno or statement.lineno
        if check.kind == "stdout":
            run.output_checks += 1
            # Comment lines wrap freely, so line breaks count as spaces.
            problem = output_mismatch(
                " ".join(printed.split()), " ".join(check.literal.split())
            )
        elif check.kind == "display":
            run.value_checks += 1
            literal = literal_prefix(" ".join(check.literal.split()))
            if literal is not None:
                node = ast.parse(literal, mode="eval").body
                problem = value_mismatch(value, node, literal)
            else:
                problem = output_mismatch(
                    " ".join(repr(value).split()), " ".join(check.literal.split())
                )
        else:
            if isinstance(statement, ast.Assign):
                name = statement.targets[0]
                assert isinstance(name, ast.Name)
                value = run.namespace[name.id]
            run.value_checks += 1
            node = ast.parse(check.literal, mode="eval").body
            problem = value_mismatch(value, node, check.literal)
        if problem is not None:
            run.failures.append(f"{label}:{line}: {problem}")
    printed = block_output.getvalue()
    run.outputs.append((printed, block.output))
    if block.output is not None:
        run.output_checks += 1
        problem = output_mismatch(printed, block.output)
        if problem is not None:
            run.failures.append(f"{where}: output block differs: {problem}")
    return True


def _prepare_matplotlib() -> None:
    """Select the non-interactive backend before an example imports pyplot.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    if "matplotlib.pyplot" not in sys.modules:
        with contextlib.suppress(ImportError):
            import matplotlib

            matplotlib.use("Agg")


def _close_figures() -> None:
    """Close every open Matplotlib figure, if pyplot was imported.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")  # type: ignore[attr-defined]


@functools.cache
def run_document(path: Path) -> DocumentRun:
    """Execute every Python block of a Markdown file and check its quotes.

    Blocks run in document order in one namespace, inside a temporary working
    directory. Execution stops at the first exception, because later blocks
    usually depend on earlier ones; value mismatches are all collected. The
    result is cached, so tests that inspect the same file share one run.

    Args:
        path: Markdown file.

    Returns:
        Counts, failures, outputs, and the final namespace.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    run = DocumentRun(path=path, namespace={"__name__": f"doc_{path.stem}"})
    label = str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/")
    _prepare_matplotlib()
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        os.chdir(directory)
        try:
            for block in code_blocks(path.read_text(encoding="utf-8")):
                if block.language != CHECKED_LANGUAGE:
                    continue
                if block.exemption is not None:
                    run.exemptions.append((block.line, block.exemption))
                    continue
                run.executed += 1
                if not _run_block(block, run, label):
                    break
        finally:
            os.chdir(previous)
            _close_figures()
    return run
