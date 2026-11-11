"""Text stays inside the figure box and on-screen text is at least 9 pt (ADR-0083).

Oracles: the figure's tight bounding box measured by Matplotlib itself, and
the font-size tokens of the theme.
"""

from __future__ import annotations

import importlib.util
import unittest

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


@unittest.skipUnless(HAS_MATPLOTLIB, "Matplotlib is not installed")
class FitTextToCanvasTests(unittest.TestCase):
    """:func:`fit_text_to_canvas` wraps wide titles and grows the canvas."""

    def setUp(self) -> None:
        """Select the non-interactive backend.

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        import matplotlib

        matplotlib.use("Agg")

    def _inside(self, figure: object) -> bool:
        """Return whether every artist lies within the figure box.

        References:
            [REF-MATPLOTLIB]. No equation is evaluated.

        """
        renderer = figure.canvas.get_renderer()  # type: ignore[attr-defined]
        tight = figure.get_tightbbox(renderer)  # type: ignore[attr-defined]
        width, height = figure.get_size_inches()  # type: ignore[attr-defined]
        return bool(
            tight.x0 >= -1e-9
            and tight.y0 >= -1e-9
            and tight.x1 <= width + 1e-9
            and tight.y1 <= height + 1e-9
        )

    def test_overflowing_label_grows_canvas_and_keeps_axes_size(self) -> None:
        """A label past the right edge fits after the call; axes keep size."""
        import matplotlib.pyplot as plt

        from probstat.visualization.theme import fit_text_to_canvas

        figure, ax = plt.subplots(figsize=(4.0, 3.0))
        ax.plot([0, 1], [0, 1])
        ax.annotate(
            "a long direct label at the line end",
            xy=(1.0, 1.0),
            xytext=(8.0, 0.0),
            textcoords="offset points",
            annotation_clip=False,
        )
        figure.canvas.draw()
        self.assertFalse(self._inside(figure))
        before = ax.get_window_extent().width
        fit_text_to_canvas(figure)
        self.assertTrue(self._inside(figure))
        self.assertAlmostEqual(ax.get_window_extent().width, before, delta=1.0)
        self.assertGreater(figure.get_size_inches()[0], 4.0)
        plt.close(figure)

    def test_wide_title_is_wrapped(self) -> None:
        """A title wider than the figure is broken into several lines."""
        import matplotlib.pyplot as plt

        from probstat.visualization.theme import fit_text_to_canvas, title_block

        figure, ax = plt.subplots(figsize=(3.0, 2.5))
        words = "word " * 30
        title_block(ax, words.strip())
        fit_text_to_canvas(figure)
        self.assertIn("\n", ax.get_title(loc="left"))
        wrapped = ax.get_title(loc="left").split()
        self.assertEqual([w.lower() for w in wrapped], words.split())
        self.assertTrue(self._inside(figure))
        plt.close(figure)

    def test_figure_inside_canvas_is_unchanged(self) -> None:
        """A figure whose text already fits keeps its size."""
        import matplotlib.pyplot as plt

        from probstat.visualization.theme import fit_text_to_canvas

        figure, ax = plt.subplots(figsize=(5.0, 4.0))
        ax.plot([0, 1], [0, 1])
        figure.tight_layout()
        fit_text_to_canvas(figure)
        self.assertEqual(tuple(figure.get_size_inches()), (5.0, 4.0))
        plt.close(figure)

    def test_smallest_text_tokens_are_nine_points(self) -> None:
        """Annotation and source sizes meet the 9 pt on-screen floor."""
        from probstat.visualization import theme

        self.assertGreaterEqual(theme.ANNOTATION_SIZE, 9.0)
        self.assertGreaterEqual(theme.SOURCE_SIZE, 9.0)
        self.assertGreaterEqual(theme.TICK_SIZE, 9.0)


if __name__ == "__main__":
    unittest.main()
