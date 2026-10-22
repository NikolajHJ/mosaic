"""Compatibility entry point. Implementation: scripts/campaign25/brd4_25_atoms.py."""
from pathlib import Path as _CompatPath
import sys as _compat_sys
_compat_root = _CompatPath(__file__).resolve().parents[1]
_compat_sys.path.insert(0, str(_compat_root / "scripts/layout"))
from experiment_layout import execute_legacy as _execute_legacy
_execute_legacy(globals(), _compat_root / 'scripts/campaign25/brd4_25_atoms.py')
