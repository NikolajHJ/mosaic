"""Compatibility import; implementation lives in :mod:`latentmol.guidance.connectivity`."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("latentmol.guidance.connectivity")
_sys.modules[__name__] = _implementation
