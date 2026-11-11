"""Text alternatives and contrast checks for the themed figures (ADR-0078).

A figure is only accessible when its content is also available as text
(WCAG 2.1 success criterion 1.1.1) and its text is legible against the
canvas (criteria 1.4.3 and 1.4.11). This module derives a plain-language
description from what a figure already states (its assertion title, the
subtitle naming the marks, axis names, ranges, and scales) and computes the
WCAG relative luminance and contrast ratio used to test both palettes.

The description is written into the saved file itself: PNG ``Title`` and
``Description`` text chunks (predefined keywords of the PNG specification),
SVG ``Title`` and ``Description`` metadata, and PDF ``Title`` and
``Subject`` entries. The HTML study report reuses the same text as the
``alt`` attribute of every embedded image.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .theme import ChartPalette

_COLORBAR_LABEL = "<colorbar>"


def _channel(value: int) -> float:
    """Return the linear-light value of one 8-bit sRGB channel.

    Args:
        value: Channel value in ``0..255``.

    Returns:
        Linearized channel in ``[0, 1]``.

    References:
        [EQ-VIZ-WCAG-001], [REF-W3C-WCAG21-2018].

    """
    scaled = value / 255.0
    if scaled <= 0.03928:
        return scaled / 12.92
    return float(((scaled + 0.055) / 1.055) ** 2.4)


def _rgb(colour: str) -> tuple[int, int, int]:
    """Parse a ``#RRGGBB`` colour token.

    Args:
        colour: Hexadecimal colour with a leading ``#``.

    Returns:
        The red, green, and blue channels in ``0..255``.

    Raises:
        ValueError: If the token is not of the form ``#RRGGBB``.

    References:
        [REF-W3C-WCAG21-2018]. No equation is evaluated.

    """
    text = colour.strip()
    digits = text[1:]
    if not (
        len(text) == 7
        and text.startswith("#")
        and all(character in "0123456789abcdefABCDEF" for character in digits)
    ):
        raise ValueError(f"expected a #RRGGBB colour, got {colour!r}")
    return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)


def relative_luminance(colour: str) -> float:
    """Return the WCAG 2.1 relative luminance of an sRGB colour.

    ``L = 0.2126 R + 0.7152 G + 0.0722 B`` on linearized channels
    [EQ-VIZ-WCAG-001]; black is 0 and white is 1.

    Args:
        colour: ``#RRGGBB`` colour token.

    Returns:
        Relative luminance in ``[0, 1]``.

    Raises:
        ValueError: If ``colour`` is not a ``#RRGGBB`` token.

    References:
        [EQ-VIZ-WCAG-001], [REF-W3C-WCAG21-2018].

    """
    red, green, blue = _rgb(colour)
    return 0.2126 * _channel(red) + 0.7152 * _channel(green) + 0.0722 * _channel(blue)


def contrast_ratio(foreground: str, background: str) -> float:
    """Return the WCAG 2.1 contrast ratio of two sRGB colours.

    ``(L1 + 0.05) / (L2 + 0.05)`` with ``L1`` the lighter relative luminance
    [EQ-VIZ-WCAG-002]. The ratio is symmetric and lies in ``[1, 21]``. Level
    AA asks for at least 4.5 for body text, 3 for large text and for
    graphical objects such as plotted lines.

    Args:
        foreground: ``#RRGGBB`` text or mark colour.
        background: ``#RRGGBB`` canvas colour.

    Returns:
        Contrast ratio.

    Raises:
        ValueError: If a colour is not a ``#RRGGBB`` token.

    References:
        [EQ-VIZ-WCAG-002], [EQ-VIZ-WCAG-001], [REF-W3C-WCAG21-2018].

    """
    lighter, darker = sorted(
        (relative_luminance(foreground), relative_luminance(background)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def legible_colour(
    colour: str, background: str, toward: str, minimum: float = 4.5
) -> str:
    """Return ``colour`` mixed toward ``toward`` until it is legible as text.

    The colour is blended with ``toward`` (normally the palette ink) in
    sRGB steps of 5% until its contrast ratio with ``background`` reaches
    ``minimum`` [EQ-VIZ-WCAG-002]; a colour that already passes is returned
    unchanged. The hue family is kept, so a label still reads as belonging
    to its series while meeting the WCAG text level.

    Args:
        colour: ``#RRGGBB`` series or mark colour.
        background: ``#RRGGBB`` colour behind the text.
        toward: ``#RRGGBB`` colour to blend toward (higher contrast end).
        minimum: Required contrast ratio (4.5 for small text).

    Returns:
        A ``#RRGGBB`` colour with at least ``minimum`` contrast, or
        ``toward`` when no blend reaches it.

    Raises:
        ValueError: If a colour is not a ``#RRGGBB`` token.

    References:
        [EQ-VIZ-WCAG-002], [EQ-VIZ-WCAG-001], [REF-W3C-WCAG21-2018].

    """
    start, end = _rgb(colour), _rgb(toward)
    _rgb(background)
    for step in range(21):
        share = step / 20.0
        mixed = "#" + "".join(
            f"{round(a + (b - a) * share):02X}" for a, b in zip(start, end, strict=True)
        )
        if contrast_ratio(mixed, background) >= minimum:
            return mixed if step else colour
    return toward


def palette_contrast(palette: ChartPalette) -> dict[str, float]:
    """Return the contrast of every text and mark token against the canvas.

    Args:
        palette: Chart palette to measure.

    Returns:
        Mapping ``ink``, ``muted``, ``accent``, ``series[0]`` ... ``series[5]``
        to its contrast ratio with ``palette.canvas``.

    References:
        [EQ-VIZ-WCAG-002], [REF-W3C-WCAG21-2018].

    """
    ratios = {
        "ink": contrast_ratio(palette.ink, palette.canvas),
        "muted": contrast_ratio(palette.muted, palette.canvas),
        "accent": contrast_ratio(palette.accent, palette.canvas),
    }
    for index, colour in enumerate(palette.series):
        ratios[f"series[{index}]"] = contrast_ratio(colour, palette.canvas)
    return ratios


def _clean(text: str) -> str:
    """Collapse whitespace (wrapped titles carry line breaks) in a label.

    References:
        [REF-W3C-WCAG21-2018]. No equation is evaluated.

    """
    return " ".join(text.split())


def _number(value: float) -> str:
    """Format an axis limit compactly for a text description.

    References:
        [REF-W3C-WCAG21-2018]. No equation is evaluated.

    """
    return f"{value:.4g}"


def _sentence(text: str) -> str:
    """Return ``text`` ending with a full stop.

    References:
        [REF-W3C-WCAG21-2018]. No equation is evaluated.

    """
    return text if text.endswith((".", "!", "?")) else text + "."


def _axes_title(ax: Any) -> str:
    """Return the first non-empty title of an axes (left, centre, right).

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    for location in ("left", "center", "right"):
        title = _clean(ax.get_title(loc=location))
        if title:
            return title
    return ""


def _axes_subtitle(ax: Any) -> str:
    """Return the subtitle that ``title_block`` anchors at the top-left corner.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    for text in ax.texts:
        anchor = getattr(text, "xy", None)
        if (
            getattr(text, "xycoords", None) == "axes fraction"
            and anchor is not None
            and tuple(anchor) == (0.0, 1.0)
        ):
            return _clean(text.get_text())
    return ""


def _axis_phrase(name: str, label: str, limits: tuple[float, float], scale: str) -> str:
    """Describe one axis: its name, range, and non-linear scale.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    low, high = sorted(limits)
    phrase = f"{name} axis"
    if label:
        phrase += f" {label}"
    phrase += f", {_number(low)} to {_number(high)}"
    if scale != "linear" and scale not in label.lower():
        phrase += f" ({scale} scale)"
    return phrase


def figure_title(figure: Any) -> str:
    """Return the headline of a figure: its suptitle or first axes title.

    Args:
        figure: Matplotlib figure or axes.

    Returns:
        Title text with whitespace collapsed; empty when the figure has none.

    References:
        [REF-MATPLOTLIB], [REF-W3C-WCAG21-2018]. No equation is evaluated.

    """
    figure = getattr(figure, "figure", figure)
    suptitle = getattr(figure, "_suptitle", None)
    if suptitle is not None and _clean(suptitle.get_text()):
        return _clean(suptitle.get_text())
    for ax in figure.axes:
        title = _axes_title(ax)
        if title:
            return title
    return ""


def describe_figure(figure: Any) -> str:
    """Return a plain-language text alternative for a themed figure.

    The description repeats what the figure itself states: the suptitle,
    then for every visible panel its assertion title, the subtitle naming the
    marks, and each axis with its name, displayed range, and scale; a colour
    bar is described by its label and range. Nothing is inferred from the
    data that the figure does not print, so the text cannot contradict the
    image (WCAG 2.1 success criterion 1.1.1).

    Args:
        figure: Matplotlib figure or axes.

    Returns:
        One paragraph of sentences; ``"Figure without titles or axes."`` when
        nothing is labelled.

    References:
        [REF-W3C-WCAG21-2018], [REF-MATPLOTLIB]. No equation is evaluated.

    """
    figure = getattr(figure, "figure", figure)
    sentences: list[str] = []
    suptitle = getattr(figure, "_suptitle", None)
    if suptitle is not None and _clean(suptitle.get_text()):
        sentences.append(_sentence(_clean(suptitle.get_text())))
    panels = [ax for ax in figure.axes if ax.get_visible() and ax.axison]
    bars = [ax for ax in panels if ax.get_label() == _COLORBAR_LABEL]
    plots = [ax for ax in panels if ax.get_label() != _COLORBAR_LABEL]
    for number, ax in enumerate(plots, start=1):
        parts: list[str] = []
        if len(plots) > 1:
            parts.append(f"Panel {number} of {len(plots)}.")
        title = _axes_title(ax)
        if title:
            parts.append(_sentence(title))
        subtitle = _axes_subtitle(ax)
        if subtitle:
            parts.append(_sentence(subtitle))
        axes_text = [
            _axis_phrase(
                "Horizontal",
                _clean(ax.get_xlabel()),
                ax.get_xlim(),
                ax.get_xscale(),
            ),
            _axis_phrase(
                "vertical",
                _clean(ax.get_ylabel()),
                ax.get_ylim(),
                ax.get_yscale(),
            ),
        ]
        parts.append("; ".join(axes_text) + ".")
        sentences.append(" ".join(parts))
    for ax in bars:
        colorbar = getattr(ax, "_colorbar", None)
        vertical = getattr(colorbar, "orientation", "vertical") == "vertical"
        label = _clean(ax.get_ylabel() if vertical else ax.get_xlabel())
        limits = ax.get_ylim() if vertical else ax.get_xlim()
        low, high = sorted(limits)
        named = f" {label}" if label else ""
        sentences.append(
            f"Colour scale{named}, {_number(low)} to {_number(high)}."
        )
    if not sentences:
        return "Figure without titles or axes."
    return " ".join(sentences)


def image_metadata(figure: Any, path: str | Path) -> dict[str, str] | None:
    """Return the file metadata that carries a figure's text alternative.

    PNG files take the predefined ``Title`` and ``Description`` text keywords
    of the PNG specification, SVG files the ``Title`` and ``Description``
    metadata entries, and PDF files the ``Title`` and ``Subject`` entries of
    the document information dictionary. Other formats get no metadata.

    Args:
        figure: Matplotlib figure or axes.
        path: Destination path; only its suffix is used.

    Returns:
        Keyword-to-text mapping for ``Figure.savefig(metadata=...)``, or
        ``None`` for a format without a supported description field.

    References:
        [REF-PNG-SPEC-2003], [REF-W3C-WCAG21-2018], [REF-MATPLOTLIB].
        No equation is evaluated.

    """
    suffix = Path(path).suffix.lower()
    description = describe_figure(figure)
    title = figure_title(figure) or description
    if suffix in {".png", ".svg"}:
        return {"Title": title, "Description": description}
    if suffix == ".pdf":
        return {"Title": title, "Subject": description}
    return None
