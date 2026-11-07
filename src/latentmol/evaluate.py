"""Compatibility import; implementation lives in :mod:`latentmol.evaluation.evaluate`."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module("latentmol.evaluation.evaluate")
if __name__ == "__main__":
    _implementation.main()
else:
    _sys.modules[__name__] = _implementation
