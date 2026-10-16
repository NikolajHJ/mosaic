"""Fixed-size ligand graph tools with explicit, lazy native model imports."""

from importlib import import_module

_EXPORTS = {
    "RelaxedGraph": ("state", "RelaxedGraph"),
    "ELEMENTS": ("state", "ELEMENTS"),
    "BOND_STATES": ("state", "BOND_STATES"),
    "initialize_graph": ("state", "initialize_graph"),
    "mix_with_uniform": ("state", "mix_with_uniform"),
    "PreparedContext": ("contracts", "PreparedContext"),
    "MolecularTrunkOutput": ("contracts", "MolecularTrunkOutput"),
    "prepare_context": ("features", "prepare_context"),
    "inject_graph": ("features", "inject_graph"),
    "FrozenProtenixTiny": ("protenix", "FrozenProtenixTiny"),
    "NativeContactLoss": ("losses", "NativeContactLoss"),
    "DBMolContactLoss": ("losses", "DBMolContactLoss"),
    "DistogramDiagnosticLoss": ("losses", "DistogramDiagnosticLoss"),
    "evaluate_graph": ("optimizers", "evaluate_graph"),
    "projected_step": ("optimizers", "projected_step"),
    "TrajectoryConfig": ("io", "TrajectoryConfig"),
    "run_trajectory": ("io", "run_trajectory"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module}"), attribute)
    globals()[name] = value
    return value
