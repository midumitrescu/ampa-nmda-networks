"""
Custom unittest loader for Siegert.

Convention: test_* = sanity/unit tests (in tests/). test_scripts_* = runnable experiments/plots.

SiegertSanityTestLoader: only test_*, exclude test_scripts_* (so scripts don't run as sanity).
SiegertScriptTestLoader: only test_scripts_* (for run_scripts.py).
"""
import unittest


class SiegertSanityTestLoader(unittest.TestLoader):
    """Loads only test_* methods, excluding test_scripts_* (script runners)."""

    def getTestCaseNames(self, testCaseClass):
        def is_sanity_test(name):
            return name.startswith("test_") and not name.startswith("test_scripts_")
        return sorted(
            name for name in super().getTestCaseNames(testCaseClass)
            if is_sanity_test(name)
        )


class SiegertScriptTestLoader(unittest.TestLoader):
    """Loads only test_scripts_* methods (runnable experiments/plots)."""

    def getTestCaseNames(self, testCaseClass):
        return sorted(
            name for name in super().getTestCaseNames(testCaseClass)
            if name.startswith("test_scripts_")
        )
