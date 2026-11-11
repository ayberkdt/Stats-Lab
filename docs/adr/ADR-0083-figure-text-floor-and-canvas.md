# ADR-0083: Nine-point text floor and text inside the canvas

- Status: Accepted
- Date: 2026-11-11

## Context

An audit of every figure the example labs write (100 figures) found that
direct labels were set at 8.5 pt and source notes at 7.5 pt, below a 9 pt floor
for figures read on screen, and that long assertion titles and line-end labels
extended past the figure box. Saving with tight bounds kept that text in the
PNG, but the figure object itself still placed it outside its own canvas, so
any other save path (fixed bounds, a different backend, embedding in a page)
could cut it.

## Decision

1. `ANNOTATION_SIZE` and `SOURCE_SIZE` in `probstat.visualization.theme` are
   9 pt. Tick labels were already 9 pt, so no text on a themed figure is
   smaller.
2. `save_figure` calls the new `fit_text_to_canvas` before writing. It
   (a) wraps axes titles that are wider than the figure, keeping every word
   in order, and (b) if any artist still lies outside the figure box, grows
   the canvas by the overflow plus a 0.08 in margin. Axes, figure texts, and
   their data keep their physical size and relative placement; an active
   layout engine is frozen at its computed layout first so the move is not
   undone.
3. Figures whose text already fits are left unchanged.

## Consequences

- Every themed figure changes slightly: labels are 0.5 pt (source notes 1.5 pt)
  larger, and some figures are wider or carry a two-line title.
- Figure legends anchored to the figure are not moved; none of the library's
  figures use them.

## Verification

- `tests/test_figure_canvas.py`: an overflowing label grows the canvas and the
  axes keep their width to one pixel; a 150-character title is wrapped into
  several lines with its words unchanged; a figure that fits keeps its size;
  the size tokens are at least 9 pt.
- The figure audit of the three figures added by ADR-0078 went from six
  errors (two per figure: text below 9 pt and text outside the box) to none.
