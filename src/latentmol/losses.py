"""Compatibility import; implementation lives in :mod:`latentmol.training.losses`."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("latentmol.training.losses")
_sys.modules[__name__] = _implementation
