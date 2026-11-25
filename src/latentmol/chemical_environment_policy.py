"""Compatibility import; implementation lives in :mod:`latentmol.guidance.environment_policy`."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("latentmol.guidance.environment_policy")
_sys.modules[__name__] = _implementation
