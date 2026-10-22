"""Compatibility adapter; canonical implementation: tests/models/test_opendde.py."""
from pathlib import Path as _CompatPath
import sys as _compat_sys
_compat_root = next(parent for parent in _CompatPath(__file__).resolve().parents
                    if (parent / "scripts/layout/experiment_layout.py").is_file())
_compat_sys.path.insert(0, str(_compat_root / "scripts/layout"))
from experiment_layout import execute_legacy as _execute_legacy
_execute_legacy(globals(), _compat_root / 'tests/models/test_opendde.py')
