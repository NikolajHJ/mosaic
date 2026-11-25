"""Compatibility import; implementation lives in :mod:`latentmol.training.workflow`."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("latentmol.training.workflow")
if __name__ == "__main__":
    _implementation.main()
else:
    _sys.modules[__name__] = _implementation
