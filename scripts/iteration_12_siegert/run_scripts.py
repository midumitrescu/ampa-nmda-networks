#!/usr/bin/env python3
"""
Run only Siegert scripts (test_scripts_*). Uses SiegertScriptTestLoader.

From project root:
  PYTHONPATH=src:scripts python scripts/iteration_12_siegert/run_scripts.py
  PYTHONPATH=src:scripts python scripts/iteration_12_siegert/run_scripts.py test_scripts_plot_gain  # one script
"""
import sys
import unittest
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
for p in (_root, _root / "src", _root / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from iteration_12_siegert.test_loader import SiegertScriptTestLoader


def run_siegert_scripts(filter_name=None):
    loader = SiegertScriptTestLoader()
    suite = unittest.TestSuite()
    from iteration_12_siegert.gain_computations import GainScripts, SolveForGainAndRateScripts
    from iteration_12_siegert.look_for_all_mu_sigma_for_fixed_rate_scripts import LookForAllSolutionsScripts
    for cls in (GainScripts, SolveForGainAndRateScripts, LookForAllSolutionsScripts):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    if filter_name:
        suite = unittest.TestSuite(t for t in suite if filter_name in str(t))
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    filter_name = sys.argv[1] if len(sys.argv) > 1 else None
    result = run_siegert_scripts(filter_name=filter_name)
    sys.exit(0 if result.wasSuccessful() else 1)
