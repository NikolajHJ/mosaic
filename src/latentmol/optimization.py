"""Compatibility import; implementation lives in :mod:`latentmol.design.step_control`."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("latentmol.design.step_control")
_sys.modules[__name__] = _implementation
