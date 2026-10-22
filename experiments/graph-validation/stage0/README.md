# Stage 0 relaxed molecular graph validation

> **Account portability, reviewed 22 October 2026:** historical remote paths
> are operator-specific. Existing controllers/LSF launchers may still pin
> private roots, runtimes and proofs; setting example environment variables
> does not override those constants. Follow the
> [account-independent HPC guide](../../../docs/HPC_GUIDE.md) and inspect
> the actual entry point before using another account. Current immutable
> releases and recorded scientific gates remain unchanged.

Experiment ID: `graph-validation/stage0`.

Validate shared fixed-occupancy graph state, native feature injection, projected optimization and frozen Tiny parity before interpreting graph optimization.

## Protocol, lineage and current status

Working sources organized on 22 October 2026. **No numerical validation or scientific experiment was executed as part of this migration.** Existing numerical outcomes, failed attempts and currently allocated jobs keep their original source/manifest bindings. The migrated working tree requires fresh allocated validation before future scientific execution; old CPU/native certificates are not certificates for these new source paths.

The following existing protocol/history documents remain in place to preserve historical links and exact commands:

- [stage_0_implementation_plan](../../../docs/stage_0_implementation_plan.md).
- [stage_0_usage](../../../docs/stage_0_usage.md).
- [stage_0_native_contract](../../../docs/stage_0_native_contract.md).

They define the actual model/context, frozen/trainable boundaries, gates, optimizer/search, seeds, runtime/quota limits and interpretation limits. Library algorithms remain in `src/mosaic/`; authenticated molecular fixtures/oracle snapshots remain shared under `tests/data/`. No loss, coefficient, allocation limit, native guard or acceptance rule is intentionally changed by this path migration.

Declared configuration files (preserved byte for byte):

- [small_molecule_stage0.json](configs/small_molecule_stage0.json): `{"seed":17,"initialization":{"mode":"interior_hard","mixing":0.15,"carbon_probability":0.75,"expected_degree":2.0,"atom_noise":0.02,"bond_noise":0.01}}`

## Owned files

| Canonical file | Purpose | Preserved legacy entry/input |
| --- | --- | --- |
| [small_molecule_stage0.lsf](hpc/small_molecule_stage0.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/small_molecule_stage0.lsf` |
| [small_molecule_stage0.py](scripts/small_molecule_stage0.py) | Restartable Stage 0 engineering example. Numerical execution belongs on DTU HPC. | `examples/small_molecule_stage0.py` |
| [small_molecule_stage0.json](configs/small_molecule_stage0.json) | Declared configuration/input; bytes preserved exactly. | `examples/configs/small_molecule_stage0.json` |

The complete [old-to-new map](../../MIGRATION.md) records each compatibility exception. Legacy scripts contain forwarding code only. Declarative compatibility copies remain at old config/template paths for callers that open those paths directly; use the canonical files for new work and keep bytes synchronized.

## Shared dependencies and artifact storage

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
python experiments/graph-validation/stage0/scripts/small_molecule_stage0.py --help
```

For real submissions, use the complete resource/argv vectors in the linked protocol and saved run receipts, replacing working-source file references through the migration map and sealing a fresh source manifest. Old receipt scripts are evidence, not reusable launchers. This reorganization does not authorize replay, restart or new scientific execution.

## Runs and logbook

Global chronological history: [experiment ledger](../../../docs/experiment_ledger.md). The records below index preserved attempts; their quoted saved states are not fresh scheduler observations.

No dated campaign directory is assigned here in `scripts/hpc/receipts/`. This does **not** mean the experiment never ran: consult the linked legacy protocol and global ledger for earlier evidence, actual job IDs and original artifact locations.
