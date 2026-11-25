"""Compatibility import; implementation lives in :mod:`latentmol.design.latent`."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("latentmol.design.latent")
_sys.modules[__name__] = _implementation
