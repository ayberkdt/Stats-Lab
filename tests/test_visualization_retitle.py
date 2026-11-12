"""Public access to the title block: ``title_artists`` and ``retitle``.

References:
    [REF-MATPLOTLIB].

"""

from __future__ import annotations

import unittest

from probstat.visualization.theme import matplotlib_available

if matplotlib_available():
    import matplotlib

    matplotlib.use("Agg")


@unittest.skipUnless(matplotlib_available(), "Matplotlib is not installed")
class RetitleTests(unittest.TestCase):
    """A title block can be found and restated without private attributes."""

    def setUp(self) -> None:
        """Draw a title block on fresh axes."""
        from probstat.visualization.theme import new_axes, pyplot, title_block

        self.plt = pyplot()
        self.axes = new_axes(None)
        title_block(self.axes, "first finding", "Marks: first")

    def tearDown(self) -> None:
        """Close the figure."""
        self.plt.close(self.axes.figure)

    def test_title_artists_are_the_title_and_subtitle(self) -> None:
        """Both artists carry their gid and their text."""
        from probstat.visualization.theme import (
            SUBTITLE_GID,
            TITLE_GID,
            title_artists,
        )

        texts = {a.get_gid(): a.get_text() for a in title_artists(self.axes)}
        self.assertEqual(
            texts, {TITLE_GID: "First finding", SUBTITLE_GID: "Marks: first"}
        )
        self.assertEqual(self.axes.get_title(loc="left"), "First finding")

    def test_retitle_replaces_text_and_keeps_style(self) -> None:
        """Weight and size stay; capitalization follows title_block."""
        from probstat.visualization.theme import retitle, title_artists

        before = {a.get_gid(): a.get_fontsize() for a in title_artists(self.axes)}
        retitle(self.axes, "second finding", "Marks: second")
        retitle(self.axes, "x3 keeps its spelling")
        self.assertEqual(self.axes.get_title(loc="left"), "x3 keeps its spelling")
        after = {a.get_gid(): a for a in title_artists(self.axes)}
        self.assertEqual(
            {gid: artist.get_fontsize() for gid, artist in after.items()}, before
        )
        self.assertIn("Marks: second", [a.get_text() for a in after.values()])

    def test_retitle_without_a_block_raises(self) -> None:
        """Axes without a title block, or a block without a subtitle."""
        from probstat.visualization.theme import new_axes, retitle, title_block

        bare = new_axes(None)
        try:
            with self.assertRaises(ValueError):
                retitle(bare, "nothing to replace")
            title_block(bare, "only a title")
            with self.assertRaises(ValueError):
                retitle(bare, "still only a title", "no subtitle to replace")
        finally:
            self.plt.close(bare.figure)


if __name__ == "__main__":
    unittest.main()
