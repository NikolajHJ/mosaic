"""Compatibility import; implementation lives in :mod:`latentmol.design.priors`."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("latentmol.design.priors")
_sys.modules[__name__] = _implementation
