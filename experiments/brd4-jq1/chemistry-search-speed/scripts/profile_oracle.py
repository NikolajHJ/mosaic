"""Isolated timing telemetry for unchanged Python-to-RDKit oracle calls.

This module never drops a native call, changes an argument, replaces a returned
value, catches an error as chemistry success, or changes an oracle/search policy.
The measured timings include profiler overhead and belong outside speed rows.
"""

from contextlib import contextmanager
from functools import wraps
import time


_ACTIVE = False


def _catch_errors_mode(args, kwargs):
    # Native SanitizeMol(mol, sanitizeOps=..., catchErrors=False). Do not bind,
    # coerce, validate or reinterpret arguments before the native call itself.
    if "catchErrors" in kwargs:
        value = kwargs["catchErrors"]
    elif len(args) >= 3:
        value = args[2]
    else:
        return "unspecified"
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "other_type"


def _timed(original, stats, name):
    @wraps(original)
    def observed(*args, **kwargs):
        stats["calls"] += 1
        if name == "SanitizeMol":
            stats["catch_errors_counts"][_catch_errors_mode(args, kwargs)] += 1
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        except BaseException:
            stats["errors"] += 1
            raise
        finally:
            stats["seconds"] += time.perf_counter() - started
    return observed


@contextmanager
def profile_oracle():
    """Return JSON timing counters while passing exact calls to both natives.

    Intended for a single serial, isolated, allocated profiling process. The
    timer covers only calls made by Python through these two Chem attributes;
    native-internal calls and separate pool workers are outside its scope.
    """
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError("Native-oracle profiling contexts cannot be nested")
    from rdkit import Chem

    originals = {
        "DetectChemistryProblems": Chem.DetectChemistryProblems,
        "SanitizeMol": Chem.SanitizeMol,
    }
    result = {
        "version": "chemistry-native-oracle-call-timing-v1",
        "scope": "current_process_python_calls_only; profiling overhead included",
        "arguments_and_results_changed": False,
        "functions": {
            name: {
                "calls": 0,
                "seconds": 0.0,
                "errors": 0,
                **({"catch_errors_counts": {
                    "true": 0, "false": 0, "unspecified": 0, "other_type": 0,
                }} if name == "SanitizeMol" else {}),
            }
            for name in originals
        },
    }
    _ACTIVE = True
    try:
        for name, original in originals.items():
            setattr(Chem, name, _timed(original, result["functions"][name], name))
        yield result
    finally:
        for name, original in originals.items():
            setattr(Chem, name, original)
        _ACTIVE = False
