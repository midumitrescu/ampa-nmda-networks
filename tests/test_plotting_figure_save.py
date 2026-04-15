"""Tests for Plotting.save_current_figure: ensure figure file is generated and valid."""
import os
import tempfile
import unittest
import matplotlib.pyplot as plt

from Plotting import save_current_figure


class TestSaveCurrentFigure(unittest.TestCase):
    """Check that save_current_figure creates the expected PNG file."""

    def test_save_current_figure_generates_file(self):
        """Creating a figure and calling save_current_figure produces an existing PNG with size > 0."""
        plt.figure()
        plt.plot([0, 1], [0, 1])
        save_name = "test_plotting_save_figure"
        with tempfile.TemporaryDirectory() as out_dir:
            path = save_current_figure(save_name=save_name, out_dir=out_dir)
            plt.close("all")
            self.assertIsNotNone(path)
            self.assertEqual(path, os.path.abspath(os.path.join(out_dir, f"{save_name}.png")))
            self.assertTrue(os.path.isfile(path), f"File not created: {path}")
            self.assertGreater(os.path.getsize(path), 0, "Saved figure file is empty")

    def test_save_current_figure_with_descriptor(self):
        """save_current_figure with caller_test_case and descriptor uses them in the filename."""
        plt.figure()
        plt.scatter([1, 2], [1, 2])
        with tempfile.TemporaryDirectory() as out_dir:
            path = save_current_figure(
                out_dir=out_dir,
                caller_test_case=self,
                descriptor="scatter",
            )
            plt.close("all")
            self.assertIsNotNone(path)
            self.assertTrue(path.endswith("_scatter.png"), f"Expected *_scatter.png, got {path}")
            self.assertTrue(os.path.isfile(path), f"File not created: {path}")
            self.assertGreater(os.path.getsize(path), 0)

    def test_save_current_figure_returns_none_when_no_name(self):
        """When save_name is None and no caller/script is given, no file is written and None is returned."""
        plt.figure()
        with tempfile.TemporaryDirectory() as out_dir:
            path = save_current_figure(save_name=None, out_dir=out_dir)
            plt.close("all")
            self.assertIsNone(path)
            self.assertEqual(len(os.listdir(out_dir)), 0)
