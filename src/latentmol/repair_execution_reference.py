"""Load a byte-pinned historical preparation oracle for allocated benchmarks.

This helper is never used by the training objective. Relative imports share
the current graph classes, while the historical module's own preparation code
remains unchanged. The caller must separately compare the preserved search
oracle and the optimized search before interpreting preparation parity.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys


def load_preparation_reference(path, expected_sha256):
    path = Path(path)
    if (not path.is_absolute() or path.name != "repair_failure_guidance.py"
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256):
        raise ValueError("Historical preparation oracle lost its exact source identity")
    name = "latentmol._repair_preparation_oracle_" + expected_sha256
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__) != path:
            raise ValueError("Historical oracle module name collided with another path")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclass resolves its owning module here.
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
