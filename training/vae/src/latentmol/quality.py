"""Compatibility import; implementation lives in :mod:`latentmol.evaluation.quality`."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("latentmol.evaluation.quality")
_sys.modules[__name__] = _implementation
