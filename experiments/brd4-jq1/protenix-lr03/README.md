# JQ1 Protenix learning-rate 0.3 study

> **Account portability, reviewed 22 October 2026:** historical remote paths
> are operator-specific. Existing controllers/LSF launchers may still pin
> private roots, runtimes and proofs; setting example environment variables
> does not override those constants. Follow the
> [account-independent HPC guide](../../../docs/HPC_GUIDE.md) and inspect
> the actual entry point before using another account. Current immutable
> releases and recorded scientific gates remain unchanged.

Experiment ID: `brd4-jq1/protenix-lr03`.

Test the explicitly authorized Protenix-only learning rate 0.3 from the original authenticated initialization.

## Protocol, lineage and current status

Working sources organized on 22 October 2026. **No numerical validation or scientific experiment was executed as part of this migration.** Existing numerical outcomes, failed attempts and currently allocated jobs keep their original source/manifest bindings. The migrated working tree requires fresh allocated validation before future scientific execution; old CPU/native certificates are not certificates for these new source paths.

The following existing protocol/history documents remain in place to preserve historical links and exact commands:

- [brd4_jq1_25_protenix_lr03](../../../docs/brd4_jq1_25_protenix_lr03.md).

They define the actual model/context, frozen/trainable boundaries, gates, optimizer/search, seeds, runtime/quota limits and interpretation limits. Library algorithms remain in `src/mosaic/`; authenticated molecular fixtures/oracle snapshots remain shared under `tests/data/`. No loss, coefficient, allocation limit, native guard or acceptance rule is intentionally changed by this path migration.

Declared configuration files (preserved byte for byte):

- [brd4_jq1_25_protenix_lr03_10000.json](configs/brd4_jq1_25_protenix_lr03_10000.json): `{"seed":17,"steps":10000,"initial_step_size":0.3,"initialization":{"kind":"chain_uniform","mixing":0.5},"version":"mosaic-brd4-jq1-25-protenix-lr0p3-v1","arm":"protenix","n_atoms":25,"repair_schedule":null}`

## Owned files

| Canonical file | Purpose | Preserved legacy entry/input |
| --- | --- | --- |
| [protenix_lr03_ops.py](scripts/protenix_lr03_ops.py) | Independent Protenix LR0.3 CPU -> native -> long dispatch; no Boltz mutations. | `scripts/protenix_lr03_ops.py` |
| [protenix_lr03_storage.py](scripts/protenix_lr03_storage.py) | Finite work3 quota and inode gates for the independent LR0.3 experiment. | `scripts/protenix_lr03_storage.py` |
| [protenix_lr03_cpu.lsf](hpc/protenix_lr03_cpu.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/protenix_lr03_cpu.lsf` |
| [protenix_lr03_stage.lsf](hpc/protenix_lr03_stage.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/protenix_lr03_stage.lsf` |
| [brd4_jq1_25_protenix_lr03_10000.json](configs/brd4_jq1_25_protenix_lr03_10000.json) | Declared configuration/input; bytes preserved exactly. | `examples/configs/brd4_jq1_25_protenix_lr03_10000.json` |
| [test_protenix_lr03_ops.py](tests/test_protenix_lr03_ops.py) | Allocated scope/rate/dispatch contracts for the independent Protenix comparison. | `tests/test_protenix_lr03_ops.py` |
| [test_protenix_lr03_storage.py](tests/test_protenix_lr03_storage.py) | Allocated finite work3 quota contracts; no home-budget substitution. | `tests/test_protenix_lr03_storage.py` |

The complete [old-to-new map](../../MIGRATION.md) records each compatibility exception. Legacy scripts contain forwarding code only. Declarative compatibility copies remain at old config/template paths for callers that open those paths directly; use the canonical files for new work and keep bytes synchronized.

## Shared dependencies and artifact storage

- [campaign25 tooling](../../../scripts/campaign25/README.md): Shared 25-slot driver, versioned protocol contracts, pocket-controller utilities and storage accounting. Used by BRD4 three-model, JQ1 backtracking/LR0.3/fixed/higher-rate/lower-rate campaigns. Fixed storage helpers have one owner here.
- [boltz tooling](../../../scripts/boltz/README.md): Shared allocated CPU-proof verification used by Boltz and the three-model/JQ1 controllers. Scientific model code remains in src/mosaic/.
- [releases tooling](../../../scripts/releases/README.md): Source archive packaging and authentication. Used by every experiment; include experiments/, canonical scripts and legacy adapters in every future release.
- [hpc tooling](../../../scripts/hpc/README.md): Shared quota admission, node-local temporary storage and verified archive recovery. These existing files stay at their canonical scripts/hpc paths unchanged.

Large arrays, structures, checkpoints, reports and sealed source bundles remain at their existing approved HPC/ignored `results/hpc/` and `results/releases/` locations. Resolve exact paths and identities from the run receipts; this migration creates no replacement scientific artifact storage.

## Commands and execution location

Working directory: repository/release root. Local work is limited to source inspection, standard-library path/AST/byte checks and packaging. Model imports, numerical/chemistry tests and scientific rendering retain the experiment's allocated DTU HPC requirement.

The exact non-numerical layout check is:

```powershell
py -3.13 scripts/layout/verify_layout.py --root .
```

Canonical CLI discovery on the authorized allocation (not executed by the migration):

```bash
python experiments/brd4-jq1/protenix-lr03/scripts/protenix_lr03_ops.py --help
python experiments/brd4-jq1/protenix-lr03/scripts/protenix_lr03_storage.py --help
```

Allocated experiment-only test command, using the environment specified in the protocol:

```bash
python -m pytest experiments/brd4-jq1/protenix-lr03/tests/test_protenix_lr03_ops.py experiments/brd4-jq1/protenix-lr03/tests/test_protenix_lr03_storage.py
```

For real submissions, use the complete resource/argv vectors in the linked protocol and saved run receipts, replacing working-source file references through the migration map and sealing a fresh source manifest. Old receipt scripts are evidence, not reusable launchers. This reorganization does not authorize replay, restart or new scientific execution.

## Runs and logbook

Global chronological history: [experiment ledger](../../../docs/experiment_ledger.md). The records below index preserved attempts; their quoted saved states are not fresh scheduler observations.

| Historical attempt | Record |
| --- | --- |
| `20261021-brd4-jq1-25-protenix-02` | [Run record](runs/20261021-brd4-jq1-25-protenix-02/README.md) |
| `20261021-brd4-jq1-25-protenix-lr03-01` | [Run record](runs/20261021-brd4-jq1-25-protenix-lr03-01/README.md) |
