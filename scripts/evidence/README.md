# Shared evidence tools

Authenticated artifact collection, transfer orchestration and byte verification. Used by native, BRD4, JQ1 and reporting campaigns.

These are shared dependencies, not independent scientific experiments. [Experiment inventory](../../experiments/INDEX.md) lists their consumers.

| Canonical file | Role | Legacy compatibility path |
| --- | --- | --- |
| [collect_protenix_evidence.py](collect_protenix_evidence.py) | Package named Mosaic run directories on allocated HPC, preserving evidence. | `scripts/collect_protenix_evidence.py` |
| [download_hpc_when_complete.py](download_hpc_when_complete.py) | One-shot bounded HPC completion download; local work is file I/O only. | `scripts/download_hpc_when_complete.py` |
| [verify_hpc_evidence.py](verify_hpc_evidence.py) | Verify and safely extract an HPC evidence archive using only stdlib I/O. | `scripts/verify_hpc_evidence.py` |

Use canonical paths for new work. Old Python/shell paths forward to these files; retained declarative files are checked against their canonical owner. All original numerical/allocation/scientific guards apply. No model, numerical test or scheduler command was run for this migration.
