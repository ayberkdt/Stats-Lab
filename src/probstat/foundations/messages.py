"""Message catalogue for user-facing report and explainer prose (ADR-0085).

Every sentence a report or an interactive page shows the reader is one
entry of a catalogue: a key mapped to one format string with named
placeholders, such as ``"Root seed: {seed}"``. Code never assembles a
sentence from translated fragments; it formats one entry and, where a list
of facts is shown, joins whole entries with a separator.

Numbers are formatted by the format specification written in the entry
(``{count:,}``, ``{value:.3g}``) and then localized: the decimal and group
separators come from the locale's number symbols (Unicode CLDR), so the
Turkish catalogue prints ``0,0421`` and ``4.096`` where English prints
``0.0421`` and ``4,096``.

An entry whose wording depends on a count is a mapping from CLDR plural
categories (``"one"``, ``"other"``) to format strings; :func:`message`
selects the form from the ``count`` field. English and Turkish both use
the categories ``one`` and ``other``; Turkish nouns do not inflect after a
numeral, so both Turkish forms are usually equal.

The locale is always an explicit argument (default ``"en"``). Nothing is
read from the environment, so a report is the same on every machine.
"""

from __future__ import annotations

import html
import string
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._messages_en import MESSAGES as _ENGLISH
from ._messages_tr import MESSAGES as _TURKISH

DEFAULT_LOCALE = "en"

Entry = str | Mapping[str, str]


@dataclass(frozen=True, slots=True)
class NumberSymbols:
    """Decimal and group separators of one locale.

    Args:
        decimal: Decimal separator.
        group: Thousands group separator.

    References:
        [REF-UNICODE-CLDR], [REF-UNICODE-TR35]. No equation is evaluated.

    """

    decimal: str
    group: str


NUMBER_SYMBOLS: Mapping[str, NumberSymbols] = {
    "en": NumberSymbols(decimal=".", group=","),
    "tr": NumberSymbols(decimal=",", group="."),
}

LANGUAGE_NAMES: Mapping[str, str] = {"en": "English", "tr": "Türkçe"}

_CATALOGUES: Mapping[str, Mapping[str, Entry]] = {"en": _ENGLISH, "tr": _TURKISH}

SUPPORTED_LOCALES: tuple[str, ...] = tuple(_CATALOGUES)


def check_locale(locale: object) -> str:
    """Return a supported locale code or raise.

    Args:
        locale: Candidate locale code such as ``"en"`` or ``"tr"``.

    Returns:
        The locale code.

    Raises:
        TypeError: If ``locale`` is not a string.
        ValueError: If no catalogue exists for ``locale``.

    References:
        [REF-UNICODE-TR35]. No equation is evaluated.

    """
    if not isinstance(locale, str):
        raise TypeError("locale must be a string such as 'en' or 'tr'")
    if locale not in _CATALOGUES:
        known = ", ".join(SUPPORTED_LOCALES)
        raise ValueError(f"unknown locale {locale!r}; supported locales: {known}")
    return locale


def catalogue(locale: str = DEFAULT_LOCALE) -> Mapping[str, Entry]:
    """Return the message catalogue of a locale.

    Args:
        locale: Supported locale code.

    Returns:
        Read-only mapping from message key to entry.

    Raises:
        ValueError: If the locale is unknown.

    References:
        [REF-UNICODE-TR35]. No equation is evaluated.

    """
    return _CATALOGUES[check_locale(locale)]


def plural_category(count: float, locale: str = DEFAULT_LOCALE) -> str:
    """Return the CLDR plural category of a count.

    English: ``one`` for the integer 1 without visible fraction digits,
    ``other`` otherwise. Turkish: ``one`` for the value 1, ``other``
    otherwise.

    Args:
        count: The number the wording depends on.
        locale: Supported locale code.

    Returns:
        ``"one"`` or ``"other"``.

    References:
        [REF-UNICODE-CLDR], [REF-UNICODE-TR35]. No equation is evaluated.

    """
    code = check_locale(locale)
    if code == "en":
        return "one" if isinstance(count, int) and count == 1 else "other"
    return "one" if count == 1 else "other"


def format_number(value: float, spec: str = "", locale: str = DEFAULT_LOCALE) -> str:
    """Format a number with a format specification and localized separators.

    The value is formatted by Python's format specification mini-language
    (which always writes ``.`` and ``,``), then the decimal and group
    separators are replaced by those of the locale.

    Args:
        value: Integer or float.
        spec: Format specification such as ``",.0f"`` or ``".3g"``.
        locale: Supported locale code.

    Returns:
        Localized text.

    References:
        [REF-UNICODE-CLDR], [REF-PYTHON-FORMAT-SPEC]. No equation is evaluated.

    """
    symbols = NUMBER_SYMBOLS[check_locale(locale)]
    text = format(value, spec)
    if symbols.decimal == "." and symbols.group == ",":
        return text
    return "".join(
        symbols.decimal
        if character == "."
        else symbols.group
        if character == ","
        else character
        for character in text
    )


class _LocaleFormatter(string.Formatter):
    """String formatter that localizes numeric fields.

    References:
        [REF-PYTHON-FORMAT-SPEC], [REF-UNICODE-CLDR]. No equation is evaluated.

    """

    def __init__(self, locale: str, escape: bool) -> None:
        """Bind the locale and whether field values are HTML-escaped.

        References:
            [REF-PYTHON-FORMAT-SPEC]. No equation is evaluated.

        """
        super().__init__()
        self._locale = locale
        self._escape = escape

    def format_field(self, value: Any, format_spec: str) -> str:
        """Format one field, localizing numbers and escaping text if asked.

        References:
            [REF-PYTHON-FORMAT-SPEC], [REF-UNICODE-CLDR]. No equation is evaluated.

        """
        if isinstance(value, int | float) and not isinstance(value, bool):
            text = format_number(value, format_spec, self._locale)
        else:
            text = format(value, format_spec)
        return html.escape(text) if self._escape else text


def _template(key: str, locale: str, fields: Mapping[str, object]) -> str:
    """Return the format string of a key, choosing a plural form if needed.

    Raises:
        KeyError: If the key is missing from the catalogue.
        ValueError: If a plural entry is formatted without ``count``.

    References:
        [REF-UNICODE-CLDR], [REF-UNICODE-TR35]. No equation is evaluated.

    """
    entry = catalogue(locale)[key]
    if isinstance(entry, str):
        return entry
    if "count" not in fields:
        raise ValueError(f"message {key!r} has plural forms and needs a count field")
    count = fields["count"]
    if not isinstance(count, int | float) or isinstance(count, bool):
        raise TypeError("count must be a number")
    return entry[plural_category(count, locale)]


def message(key: str, locale: str = DEFAULT_LOCALE, /, **fields: object) -> str:
    """Return one catalogue sentence with its fields filled in.

    Args:
        key: Message key, for example ``"report.root_seed"``.
        locale: Supported locale code.
        **fields: Values of the named placeholders; numbers are localized.

    Returns:
        Formatted text.

    Raises:
        KeyError: If the key or a placeholder value is missing.
        ValueError: If the locale is unknown.

    References:
        [REF-UNICODE-CLDR], [REF-PYTHON-FORMAT-SPEC]. No equation is evaluated.

    """
    return _LocaleFormatter(locale, escape=False).vformat(
        _template(key, locale, fields), (), fields
    )


def message_or(key: str, default: str, locale: str = DEFAULT_LOCALE) -> str:
    """Return a field-free catalogue entry, or ``default`` when it is absent.

    Used for recorded identifiers (such as a stage status) that have a
    catalogue word when known and are shown as recorded otherwise.

    Args:
        key: Message key.
        default: Text returned when the catalogue has no such key.
        locale: Supported locale code.

    Returns:
        Catalogue text or ``default``.

    References:
        [REF-UNICODE-TR35]. No equation is evaluated.

    """
    return message(key, locale) if key in catalogue(locale) else default


def html_message(key: str, locale: str = DEFAULT_LOCALE, /, **fields: object) -> str:
    """Return a catalogue sentence as HTML text: literal parts and fields escaped.

    Catalogue entries are plain text; markup is added by the caller around
    the returned string, never inside the entry.

    Args:
        key: Message key.
        locale: Supported locale code.
        **fields: Values of the named placeholders.

    Returns:
        Escaped, formatted text safe in element content and attribute values.

    References:
        [REF-WHATWG-HTML], [REF-UNICODE-CLDR]. No equation is evaluated.

    """
    formatter = _LocaleFormatter(locale, escape=True)
    template = _template(key, locale, fields)
    parts: list[str] = []
    for literal, name, spec, conversion in formatter.parse(template):
        parts.append(html.escape(literal))
        if name is None:
            continue
        value, _ = formatter.get_field(name, (), fields)
        value = formatter.convert_field(value, conversion)
        parts.append(formatter.format_field(value, spec or ""))
    return "".join(parts)


def placeholders(template: str) -> frozenset[str]:
    """Return the placeholder names of a format string with their specs.

    Each item is ``"name"`` or ``"name:spec"``, so two locales can be
    checked to format the same fields the same way.

    Args:
        template: Format string.

    Returns:
        Set of ``name[:spec]`` items.

    References:
        [REF-PYTHON-FORMAT-SPEC]. No equation is evaluated.

    """
    found: set[str] = set()
    for _, name, spec, _ in string.Formatter().parse(template):
        if name is not None:
            found.add(f"{name}:{spec}" if spec else name)
    return frozenset(found)
