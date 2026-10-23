"""Lazy factory lookup. New model plugins need no changes to the experiment loop.

Custom factories are executable Python from trusted local code, not data-only
configuration. Merely listing or planning a study never imports these factories.
"""

from importlib import import_module


BUILTINS = {
    "boltz2": "latentmol.multimodel.boltz:load_boltz",
    "nesso1": "latentmol.multimodel.nesso:load_nesso",
    "protenix": "latentmol.multimodel.protenix:load_protenix",
}


def factories(extra=None):
    extra = extra or {}
    if set(extra) & set(BUILTINS):
        raise ValueError("Custom plugins cannot shadow built-in names; choose a new name")
    return {**BUILTINS, **extra}


def load_branch(name, *, spec, device, options, source_sha256, extra=None):
    lookup = factories(extra)
    if name not in lookup:
        raise ValueError(f"Unknown model {name!r}; available: {sorted(lookup)}")
    module, attribute = lookup[name].split(":")
    factory = getattr(import_module(module), attribute)
    return factory(spec=spec, device=device, options=options, source_sha256=source_sha256)
