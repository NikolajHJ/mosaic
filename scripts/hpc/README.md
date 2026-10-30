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


The shared [pre-model startup checker](startup/README.md) serves the fresh
Boltz30k comparison and independent DBMol pilot. It tests actual generated
driver startup in allocated CPU callbacks; its receipt is separate from native
gradient readiness and never changes saved scientific results.


Default storage admission for new releases is now documented in
[HPC_STORAGE_POLICY.md](../../docs/HPC_STORAGE_POLICY.md) and implemented by
[storage_accounting.py](storage_accounting.py). It uses exact live quota once,
remaining writes once, measured remaining-update forecasts and explicit reserves.
The endpoint owner provides authenticated experiment integration.28portable
shared regression cases live in tests/core/test_storage_accounting.py. Old
sealed resource policies remain historical and must not be edited in place.

For new exploratory submissions, prefer the [32 GB V100 profile](../../docs/GPU_POLICY.md)
in [gpu_policy.py](gpu_policy.py); retain L40S when compatibility, memory or measured
runtime makes it preferable. This default does not alter existing allocations.
