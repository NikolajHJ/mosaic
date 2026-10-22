# Shared chemistry tools

Shared chemistry driver, reference preparation, repair oracle generation/readiness and base report rendering. Used by BRD4 chemistry-5000, chemistry-ramp and Boltz-affinity reports.

These are shared dependencies, not independent scientific experiments. [Experiment inventory](../../experiments/INDEX.md) lists their consumers.

| Canonical file | Role | Legacy compatibility path |
| --- | --- | --- |
| [prepare_chemistry_reference.py](prepare_chemistry_reference.py) | Allocated CPU: empirical training-only composition, without a model load. | `scripts/prepare_chemistry_reference.py` |
| [create_chemistry_oracles.py](create_chemistry_oracles.py) | Allocated CPU: emit values/VJPs from the pinned, unmodified Torch source. | `scripts/create_chemistry_oracles.py` |
| [vendor_repair_reference.py](vendor_repair_reference.py) | Extract detached DBMol repair code without importing or executing it. | `scripts/vendor_repair_reference.py` |
| [validate_chemistry_readiness.py](validate_chemistry_readiness.py) | Authenticate actual allocated fixture/reference outputs before native launch. | `scripts/validate_chemistry_readiness.py` |
| [render_chemistry_trajectory.py](render_chemistry_trajectory.py) | Allocated CPU saved-only report: raw graphs, contact and repair components. | `scripts/render_chemistry_trajectory.py` |
| [chemistry_report_template.html](chemistry_report_template.html) | Saved-artifact report template; rendering changes no scientific outcome. | `scripts/chemistry_report_template.html` |
| [brd4_chemistry.py](brd4_chemistry.py) | Allocated native pilots and a fresh configured BRD4 chemistry trajectory. | `examples/brd4_chemistry.py` |

Use canonical paths for new work. Old Python/shell paths forward to these files; retained declarative files are checked against their canonical owner. All original numerical/allocation/scientific guards apply. No model, numerical test or scheduler command was run for this migration.
