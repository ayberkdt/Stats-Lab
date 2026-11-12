"""Figure specifications for the foundation guides (ADR-0105).

Every guide in ``docs/foundations/`` shows at least one figure that states
its finding. A figure is declared once, in a group module of this package,
as a :class:`GuideFigure`: the guide it belongs to, its text alternative, and
a ``draw`` function that runs a seeded library computation and returns the
Matplotlib figure together with the facts its caption quotes. The facts are
the formatted strings the caption in the guide must contain, so a changed
number fails ``tests/test_guide_figures.py`` instead of leaving a stale
caption.

``tools/build_guide_figures.py`` renders each figure in the light and the
dark palette, audits both with datawiz, writes the PNG files to
``docs/images/guides/``, and records the facts in one manifest per group
module (``docs/images/guides/<group>.json``).

References:
    [REF-PROBSTAT-REPORTING], [REF-MATPLOTLIB].

"""

from __future__ import annotations

import importlib
import pkgutil
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class GuideFigure:
    """One figure of one guide.

    Attributes:
        name: File stem, lowercase words joined by hyphens; unique across
            all groups. The files are ``<name>.png`` and ``<name>-dark.png``.
        guide: Stem of the guide in ``docs/foundations/`` that embeds it.
        alt: Text alternative: what the figure shows, in one sentence
            without the finding.
        draw: Seeded function returning ``(figure, facts)``; ``facts`` maps
            a short key to the formatted string the caption quotes.

    References:
        [REF-PROBSTAT-REPORTING].

    """

    name: str
    guide: str
    alt: str
    draw: Callable[[], tuple[Any, dict[str, str]]]

    def __post_init__(self) -> None:
        """Reject names that are not lowercase hyphenated words.

        References:
            [REF-PROBSTAT-REPORTING].

        """
        if not NAME_PATTERN.match(self.name):
            raise ValueError(f"figure name {self.name!r} is not kebab-case")
        if self.name.endswith("-dark"):
            raise ValueError(f"figure name {self.name!r} must not end in -dark")


def groups() -> Iterator[tuple[str, list[GuideFigure]]]:
    """Yield every group module's name and its ``FIGURES`` list, sorted.

    References:
        [REF-PROBSTAT-REPORTING].

    """
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda item: item.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        yield info.name, list(module.FIGURES)
