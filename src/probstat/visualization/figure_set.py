"""The figure-set contract of the library in the datawiz set format (ADR-0098).

:mod:`probstat.visualization.theme` holds the contract as code: the entity
table (:data:`~probstat.visualization.theme.ENTITY_STYLES`), the recurring
quantities (:data:`~probstat.visualization.theme.QUANTITIES`), and the
category orders. This module writes the same tables as one JSON document per
palette, the format read by the datawiz set tooling (``datawiz.Seti`` and
``python -m datawiz.tutarlilik``), so the figures of every lab can be audited
against the contract as a set: an entity in one colour, a quantity with one
name, unit spelling, and axis role in every figure.

The documents live in ``docs/figures/stats-lab.seti.json`` (light) and
``docs/figures/stats-lab-dark.seti.json`` (dark); ``tools/
generate_figure_set_contract.py`` rewrites them and
``tests/test_figure_set_contract.py`` checks that they equal
:func:`figure_set_contract`.
"""

from __future__ import annotations

import json
from typing import Any

from .theme import (
    CATEGORY_ORDERS,
    ENTITY_STYLES,
    GRAPHITE,
    METHOD_NAMES,
    PORCELAIN,
    QUANTITIES,
    ChartPalette,
)

CONTRACT_FILES = {
    PORCELAIN.name: "stats-lab.seti.json",
    GRAPHITE.name: "stats-lab-dark.seti.json",
}
"""File name of the contract of each palette under ``docs/figures/``."""


def _dashes(pattern: tuple[float, tuple[float, ...]] | None) -> Any:
    """Return a dash pattern in JSON form (``"-"`` for a solid line).

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    if pattern is None:
        return "-"
    offset, sequence = pattern
    return [offset, list(sequence)]


def figure_set_contract(palette: ChartPalette) -> dict[str, Any]:
    """Return the figure-set contract for one palette as a datawiz set.

    Every entity of :data:`~probstat.visualization.theme.ENTITY_STYLES`
    becomes a ``varliklar`` record with its resolved colour, marker, dash
    pattern, line width, and colour role; context entities are marked
    ``baglam``. Every quantity of
    :data:`~probstat.visualization.theme.QUANTITIES` becomes a
    ``nicelikler`` record with its axis title, unit, and axis role
    (``"serbest"`` for a free role). Units are written in brackets
    (``etiket_bicimi = "koseli"``) and decimals with a point.

    Args:
        palette: ``PORCELAIN`` or ``GRAPHITE``.

    Returns:
        A JSON-serializable mapping in the datawiz set format, version 1.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    entities = {
        name: {
            "renk": style.colour(palette).lower(),
            "baglam": style.role == "context",
            "isaret": style.marker,
            "cizgi": _dashes(style.dashes),
            "kalinlik": style.width,
            "rol": style.role,
            "sira": index,
        }
        for index, (name, style) in enumerate(ENTITY_STYLES.items())
    }
    quantities = {
        key: {
            "etiket": quantity.name,
            "birim": quantity.unit,
            "eksen": "serbest" if quantity.axis == "free" else quantity.axis,
            "sinir": None,
            "olcek": None,
            "bicim": None,
        }
        for key, quantity in QUANTITIES.items()
    }
    return {
        "_aciklama": (
            "Figure-set contract of probstat (ADR-0098), generated from "
            "probstat.visualization.theme by tools/generate_figure_set_contract.py; "
            "do not edit by hand."
        ),
        "surum": 1,
        "dergi": "web",
        "hedef": "web-640",
        "ortam": "web",
        "palet": palette.name,
        "koyu": palette is GRAPHITE,
        "etiket_bicimi": "koseli",
        "cift_kodlama": True,
        "ondalik_ayirici": ".",
        "tema": {
            "canvas": palette.canvas,
            "ink": palette.ink,
            "muted": palette.muted,
            "rule": palette.rule,
            "accent": palette.accent,
            "series": list(palette.series),
            "sequential": list(palette.sequential),
        },
        "varliklar": entities,
        "nicelikler": quantities,
        "siralar": {group: list(items) for group, items in CATEGORY_ORDERS.items()},
        "yontem_adlari": dict(METHOD_NAMES),
        "haritalar": {},
        "figurler": {},
    }


def contract_text(palette: ChartPalette) -> str:
    """Return the contract of ``palette`` as the exact text of its JSON file.

    Args:
        palette: ``PORCELAIN`` or ``GRAPHITE``.

    Returns:
        Indented JSON with a trailing newline.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    return json.dumps(figure_set_contract(palette), ensure_ascii=False, indent=1) + "\n"
