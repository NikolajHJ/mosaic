"""Compatibility import; implementation lives in :mod:`latentmol.evaluation.panel`."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("latentmol.evaluation.panel")
if __name__ == "__main__":
    _implementation.main()
else:
    _sys.modules[__name__] = _implementation
