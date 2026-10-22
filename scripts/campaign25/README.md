# Shared campaign25 tools

Shared 25-slot driver, versioned protocol contracts, pocket-controller utilities and storage accounting. Used by BRD4 three-model, JQ1 backtracking/LR0.3/fixed/higher-rate/lower-rate campaigns. Fixed storage helpers have one owner here.

These are shared dependencies, not independent scientific experiments. [Experiment inventory](../../experiments/INDEX.md) lists their consumers.

| Canonical file | Role | Legacy compatibility path |
| --- | --- | --- |
| [campaign25_contract.py](campaign25_contract.py) | Standard-library contract for the new three-arm 25-slot campaign. | `scripts/campaign25_contract.py` |
| [campaign25_pocket_ops.py](campaign25_pocket_ops.py) | Allocated, fail-closed stage transitions for the three-arm 25-slot campaign. | `scripts/campaign25_pocket_ops.py` |
| [fixed25_storage.py](fixed25_storage.py) | Finite work3 quota and inode gates for the fixed-step three-arm experiment. | `scripts/fixed25_storage.py` |
| [fixed25_split_storage.py](fixed25_split_storage.py) | All-work3 storage adoption after authenticated Mosaic archival cleanup. | `scripts/fixed25_split_storage.py` |
| [fixed25_recover_storage.py](fixed25_recover_storage.py) | Declared storage-only recovery for the two unstarted fixed-step Boltz arms. | `scripts/fixed25_recover_storage.py` |
| [brd4_25_atoms.py](brd4_25_atoms.py) | Allocated fresh 25-slot three-arm validation/optimization, preserving native gates. | `examples/brd4_25_atoms.py` |

Use canonical paths for new work. Old Python/shell paths forward to these files; retained declarative files are checked against their canonical owner. All original numerical/allocation/scientific guards apply. No model, numerical test or scheduler command was run for this migration.
