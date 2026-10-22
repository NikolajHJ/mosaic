# Shared boltz tools

Shared allocated CPU-proof verification used by Boltz and the three-model/JQ1 controllers. Scientific model code remains in src/mosaic/.

These are shared dependencies, not independent scientific experiments. [Experiment inventory](../../experiments/INDEX.md) lists their consumers.

| Canonical file | Role | Legacy compatibility path |
| --- | --- | --- |
| [check_boltz_cpu_proof.py](check_boltz_cpu_proof.py) | Read-only producer proof check, shared by the allocated native launcher. | `scripts/check_boltz_cpu_proof.py` |

Use canonical paths for new work. Old Python/shell paths forward to these files; retained declarative files are checked against their canonical owner. All original numerical/allocation/scientific guards apply. No model, numerical test or scheduler command was run for this migration.
