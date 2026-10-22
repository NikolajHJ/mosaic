# Script navigation and ownership

Read the required [experiment folder policy and inventory](../experiments/README.md)
and [agent instructions](../AGENTS.md) before adding or running scripts.

This root contains navigation only. Commands live in named capability folders
or [experiment homes](../experiments/INDEX.md). Each folder has a README explaining
its purpose and consumers. Edit canonical implementations, not adapters.

- [Complete migration map](../experiments/MIGRATION.md) lists every old path,
  canonical owner and compatibility exception.
- [releases/](releases/README.md): source packaging and authentication.
- [evidence/](evidence/README.md): collection, transfer and byte verification.
- [graph/](graph/README.md): drivers, fixtures and launchers shared by graph controls.
- [chemistry/](chemistry/README.md): shared chemistry driver, references and oracles.
- [campaign25/](campaign25/README.md): shared 25-slot driver, contracts and storage.
- [boltz/](boltz/README.md): shared CPU-proof verification.
- [reporting/](reporting/README.md): generic saved-trajectory display tools.
- [layout/](layout/README.md): path resolution, adapters and local static checks.
- [compat/](compat/README.md): former flat script/test entry points and assets,
  clearly separated from working implementations and pytest discovery.
- [hpc/](hpc/README.md): shared allocation helpers and legacy launcher adapters.
  A `.lsf` file alone is not a protocol or authorization to submit it.
- [hpc/receipts/](hpc/receipts/README.md) preserves historical submissions.
  Keep recorded paths; do not replay these scripts to launch new work.
- [templates/](templates/README.md): retained declarative compatibility templates.
  The former loose HTML asset is now under `compat/assets/`.
- [Experiment ledger / agent logbook](../docs/experiment_ledger.md) records what
  actually ran, when, which jobs/artifacts belong to it and what failed.

The [nested layout map](../experiments/folder-layout.json) records every moved
adapter and shared test. Old flat Python names can be resolved/executed through
`scripts/layout/experiment_layout.py resolve OLD_PATH` or `run OLD_PATH [ARGS ...]`.
New source archives add authenticated old aliases for legacy launchers without
recreating those loose files in this checkout. Existing HPC releases, historical
receipt directories and report outputs remain in place.

Default [test discovery](../tests/README.md) selects nested shared tests and
canonical experiment tests once. Shared package algorithms stay in `src/mosaic/`.
Do not add loose implementations to this root; create a documented capability or
use the owning experiment.

[Nested-layout verification](../experiments/nested-layout-verification.json)
records the preservation, discovery and disposable archive checks; allocated
scientific regressions remain required before future execution.
