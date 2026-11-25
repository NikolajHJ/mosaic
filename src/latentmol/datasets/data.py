"""Grouped API for the dataset implementation retained at latentmol.data."""
import importlib as _importlib
import sys as _sys
_implementation = _importlib.import_module("latentmol.data")
if __name__ == "__main__":
    _implementation.main()
else:
    _sys.modules[__name__] = _implementation
