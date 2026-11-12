# Overview presentation

A sixteen-slide overview of the library, in English (`index.html`) and Turkish
(`index.tr.html`), built on the sunumatik deck runtime and audited with the
datawiz figure checks ([ADR-0097](../adr/ADR-0097-overview-presentation.md)).

## Open

Serve this folder with any local static server and open the deck in a
browser; ES modules do not load from `file://`.

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs/presentation
# http://127.0.0.1:8000/index.html      (English)
# http://127.0.0.1:8000/index.tr.html   (Turkish)
```

Keys: arrows or Space to move, `O` overview, `N` speaker notes, `F` full
screen, `Z` zoom, `P` print to PDF, `?` shortcuts. `#6` opens slide six,
`?theme=light` or `?theme=dark` overrides the system colour scheme, and
`?export=1` hides the controls for screenshots. Nothing is fetched from the
network.

## Rebuild

Every number on the slides is an element `<span data-k="key">` whose text is
written from `deck-data.json`; do not type numbers into the HTML.

```bash
PYTHONPATH=src python tools/build_presentation_data.py            # data, HTML values, checks
PYTHONPATH=src python tools/build_presentation_data.py --figures  # also the charts (needs datawiz)
node docs/presentation/runtime/denetle-cli.mjs docs/presentation/index.html docs/presentation/index.tr.html
```

The first command reruns the README quick-start study (seed 2026) and the
mission laboratory, reads `benchmarks/results/quick.json` and the repository
counts, writes `deck-data.json`, fills both decks, and writes the validator
inputs in `checks/`. `tests/test_presentation.py` fails when the stored data
no longer equals a fresh run; rerun the command after a change that moves a
shown number (for example a test count crossing the next hundred).

## Layout

| Path | Content |
|---|---|
| `index.html`, `index.tr.html` | the two decks, same slides and same data keys |
| `deck-data.json` | every number the decks show |
| `figures/` | slide charts in both languages and themes, with their datawiz audit reports and `audit.json` |
| `runtime/deste-kabuk.*` | deck runtime copied from sunumatik (`presets/core`), chrome labels translatable |
| `runtime/deck-data.mjs`, `runtime/denetle-cli.mjs` | number formats and the data consistency check, adapted from sunumatik `deck_starter` |
| `runtime/deck.mjs`, `runtime/deck.css` | mounting, theme switch, slide styles |
| `checks/` | generated inputs of the sunumatik validators (page types, copy, deck manifest, evidence ledger, plan, visual spec) |
