# Shared hpc tools

Shared quota admission, node-local temporary storage and verified archive recovery. These existing files stay at their canonical scripts/hpc paths unchanged.

These are shared dependencies, not independent scientific experiments. [Experiment inventory](../../experiments/INDEX.md) lists their consumers.

| Canonical file | Role | Legacy compatibility path |
| --- | --- | --- |
| [boltz_node_tmpdir.sh](boltz_node_tmpdir.sh) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/boltz_node_tmpdir.sh` |
| [recover_verified_500_archive.py](recover_verified_500_archive.py) | Remove one redundant archive; retain its verified local copy and raw HPC data. | `scripts/hpc/recover_verified_500_archive.py` |
| [stage0_quota_preflight.py](stage0_quota_preflight.py) | Read DTU quota once; decline numerical work when headroom is insufficient. | `scripts/hpc/stage0_quota_preflight.py` |
| [zhome_quota_preflight.py](zhome_quota_preflight.py) | Conservatively gate this campaign's small/medium artifacts on DTU home. | `scripts/hpc/zhome_quota_preflight.py` |

Use canonical paths for new work. Old Python/shell paths forward to these files; retained declarative files are checked against their canonical owner. All original numerical/allocation/scientific guards apply. No model, numerical test or scheduler command was run for this migration.

[Historical receipts](receipts/README.md) retain their original paths and bytes. Do not replay them.
