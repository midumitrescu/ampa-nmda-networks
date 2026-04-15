#!/usr/bin/env python3
"""
Run only Siegert sanity tests (test_*), excluding test_scripts_*.

From project root:
  PYTHONPATH=src:scripts python scripts/iteration_12_siegert/run_sanity_tests.py
"""
import sys
import unittest
import importlib.util
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
for p in (_root, _root / "src", _root / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from iteration_12_siegert.test_loader import SiegertSanityTestLoader


def run_siegert_sanity_tests():
    loader = SiegertSanityTestLoader()
    suite = unittest.TestSuite()
    sanity_dir = _root / "tests" / "iteration_12_siegert"
    for name, class_name in [
        ("test_look_for_all_solutions_sanity", "LookForAllSolutionsSanityTests"),
        ("test_gain_sanity", "SiegertGainSanityTests"),
    ]:
        path = sanity_dir / f"{name}.py"
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        cls = getattr(mod, class_name)
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    result = run_siegert_sanity_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
