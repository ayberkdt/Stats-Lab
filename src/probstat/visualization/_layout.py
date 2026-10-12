"""Layout helpers shared by the reliability, scoring, and transport figures.

They keep long computed titles inside the figure width: the assertion title
and the muted subtitle are wrapped at word boundaries to a width derived
from the figure size, and the title is lifted above a multi-line subtitle.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Sequence
from typing import Any

from .theme import TITLE_SIZE, active_palette, title_block

_TITLE_CHARS_PER_INCH = 10.5
_SUBTITLE_CHARS_PER_INCH = 14.5
_SUBTITLE_LINE_POINTS = 12.5


def wrapped_title_block(
    axes: Any,
    title: str,
    parts: Sequence[str],
    *,
    width_inches: float | None = None,
) -> None:
    """Draw an assertion title and a subtitle wrapped to the figure width.

    The subtitle is ``parts`` joined by `` · `` and broken between parts
    where possible, so a mark description never splits across lines.

    Args:
        axes: Axes that carries the title (the top panel of a figure).
        title: Computed assertion title.
        parts: Short descriptions of the marks, in reading order.
        width_inches: Text width (default: the figure width).

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    width = width_inches if width_inches is not None else axes.figure.get_figwidth()
    title_chars = max(30, math.floor(width * _TITLE_CHARS_PER_INCH))
    subtitle_chars = max(40, math.floor(width * _SUBTITLE_CHARS_PER_INCH))
    lines: list[str] = []
    for part in parts:
        if not part:
            continue
        candidate = f"{lines[-1]} · {part}" if lines else part
        if lines and len(candidate) <= subtitle_chars:
            lines[-1] = candidate
        else:
            lines.extend(
                textwrap.wrap(part, subtitle_chars, break_on_hyphens=False) or [part]
            )
    if lines:
        lines[0] = lines[0][:1].upper() + lines[0][1:]
    wrapped = textwrap.fill(title, title_chars, break_on_hyphens=False)
    subtitle = "\n".join(lines) if lines else None
    title_block(axes, wrapped, subtitle)
    if len(lines) > 1:
        palette = active_palette()
        heading = wrapped[:1].upper() + wrapped[1:]
        axes.set_title(
            heading,
            loc="left",
            fontsize=TITLE_SIZE,
            fontweight="bold",
            color=palette.ink,
            pad=24.0 + _SUBTITLE_LINE_POINTS * (len(lines) - 1),
        )


def halo() -> list[Any]:
    """Return a canvas-coloured stroke that keeps labels legible over marks.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    from matplotlib import patheffects

    return [patheffects.withStroke(linewidth=3.0, foreground=active_palette().canvas)]
