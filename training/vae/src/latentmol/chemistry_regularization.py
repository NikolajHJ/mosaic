"""Compatibility import; implementation lives in :mod:`latentmol.guidance.regularization`."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("latentmol.guidance.regularization")
_sys.modules[__name__] = _implementation
