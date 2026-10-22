"""Route release-only legacy test selections to their canonical source files.

The working tree discovers nested tests directly. New authenticated releases
also contain old flat aliases for explicit HPC selections; collect the actual
module before importing it, preserving pytest's import-path mismatch checks.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/layout"))
from experiment_layout import resolve


def pytest_pycollect_makemodule(module_path, parent):
    canonical = resolve(module_path)
    if canonical != module_path:
        from pytest import Module
        from _pytest.python import importtestmodule

        class CanonicalModule(Module):
            def _getobj(self):
                return importtestmodule(canonical, self.config)

        # Keep the requested node path for pytest's CLI argument matching, while
        # importing the canonical module with normal rewriting/mismatch checks.
        return CanonicalModule.from_parent(parent, path=module_path)
    return None
