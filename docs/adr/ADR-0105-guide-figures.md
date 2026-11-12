# ADR-0105: Checked figures in every foundation guide

- Status: Accepted
- Date: 2026-11-12

## Context

The 53 guides in `docs/foundations/` explain each method with cited formulas
and executable examples, but before this decision none of them showed a
figure. Four guides named a PNG that a lab writes to `examples/generated/`, a
directory that is not committed, so a reader on GitHub saw nothing. The labs
draw about a hundred audited figures, yet the explanation and the picture
lived in different places, and a caption written next to a figure could keep
a number after the computation changed.

## Decision

1. **One to three figures per guide.** Each guide shows the idea a reader
   most needs to see, computed by the library from a fixed seed. The title
   states the finding with its number; the subtitle says what the marks are.
2. **Declared once, drawn by a tool.** A figure is a `GuideFigure` in a group
   module of `tools/guide_figures/` (guide, text alternative, and a `draw`
   function returning the figure and the facts its caption quotes).
   `tools/build_guide_figures.py` renders every figure in the light and the
   dark palette, runs the datawiz web audit on both (any error fails the
   build and writes nothing), requires both runs to return the same facts,
   and writes the PNG files and one manifest per group to
   `docs/images/guides/`.
3. **Embedded with both palettes.** A guide embeds a figure as a `<picture>`
   element whose dark source GitHub selects for readers in dark mode, with
   the recorded text alternative, followed by a one-paragraph italic caption.
4. **Checked without Matplotlib.** `tests/test_guide_figures.py` requires a
   figure in every guide, each figure embedded exactly once in its own guide
   with its recorded text alternative, every recorded fact quoted verbatim in
   the caption, both PNG files present and under 400 KB, and no image file or
   picture element without a manifest record.

## Consequences

- A changed estimate fails the caption test until the figures are rebuilt
  and the caption is updated, so captions cannot drift from the computation.
- The repository grows by the committed PNG files (two per figure at
  160 dpi); the size bound keeps the total in the low tens of megabytes.
- Rebuilding needs Matplotlib and datawiz, which are not library
  dependencies; the tests need neither.

## References

- [REF-PROBSTAT-REPORTING], [REF-MATPLOTLIB], [REF-PNG-SPEC-2003],
  [REF-W3C-WCAG21-2018].
