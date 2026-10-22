# BRD4 chemistry guidance

> **Account portability, reviewed 22 October 2026:** historical remote paths
> are operator-specific. Existing controllers/LSF launchers may still pin
> private roots, runtimes and proofs; setting example environment variables
> does not override those constants. Follow the
> [account-independent HPC guide](../../../docs/HPC_GUIDE.md) and inspect
> the actual entry point before using another account. Current immutable
> releases and recorded scientific gates remain unchanged.

Experiment ID: `brd4/chemistry-5000`.

Compare a rule-based chemistry-guided molecular trajectory with the frozen native contact objective, retaining invalid graphs and original repair/search semantics.

## Protocol, lineage and current status

Working sources organized on 22 October 2026. **No numerical validation or scientific experiment was executed as part of this migration.** Existing numerical outcomes, failed attempts and currently allocated jobs keep their original source/manifest bindings. The migrated working tree requires fresh allocated validation before future scientific execution; old CPU/native certificates are not certificates for these new source paths.

The following existing protocol/history documents remain in place to preserve historical links and exact commands:

- [brd4_chemistry_5000](../../../docs/brd4_chemistry_5000.md).

They define the actual model/context, frozen/trainable boundaries, gates, optimizer/search, seeds, runtime/quota limits and interpretation limits. Library algorithms remain in `src/mosaic/`; authenticated molecular fixtures/oracle snapshots remain shared under `tests/data/`. No loss, coefficient, allocation limit, native guard or acceptance rule is intentionally changed by this path migration.

Declared configuration files (preserved byte for byte):

- [brd4_chemistry_5000.json](configs/brd4_chemistry_5000.json): `{"version":"brd4-contact-chemistry-logits-v1","n_atoms":6,"seed":17,"steps":5000,"initial_step_size":0.03}`

## Owned files

| Canonical file | Purpose | Preserved legacy entry/input |
| --- | --- | --- |
| [brd4_chemistry.lsf](hpc/brd4_chemistry.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_chemistry.lsf` |
| [brd4_chemistry_preview.lsf](hpc/brd4_chemistry_preview.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_chemistry_preview.lsf` |
| [brd4_chemistry_readiness.lsf](hpc/brd4_chemistry_readiness.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_chemistry_readiness.lsf` |
| [brd4_chemistry_repackage.lsf](hpc/brd4_chemistry_repackage.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_chemistry_repackage.lsf` |
| [brd4_chemistry_report.lsf](hpc/brd4_chemistry_report.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_chemistry_report.lsf` |
| [submit_chemistry_native.sh](hpc/submit_chemistry_native.sh) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/submit_chemistry_native.sh` |
| [submit_chemistry_readiness.sh](hpc/submit_chemistry_readiness.sh) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/submit_chemistry_readiness.sh` |
| [brd4_chemistry_5000.json](configs/brd4_chemistry_5000.json) | Declared configuration/input; bytes preserved exactly. | `examples/configs/brd4_chemistry_5000.json` |

The complete [old-to-new map](../../MIGRATION.md) records each compatibility exception. Legacy scripts contain forwarding code only. Declarative compatibility copies remain at old config/template paths for callers that open those paths directly; use the canonical files for new work and keep bytes synchronized.

## Shared dependencies and artifact storage

- [chemistry tooling](../../../scripts/chemistry/README.md): Shared chemistry driver, reference preparation, repair oracle generation/readiness and base report rendering. Used by BRD4 chemistry-5000, chemistry-ramp and Boltz-affinity reports.
- [graph tooling](../../../scripts/graph/README.md): Shared graph/control driver, fixture preparation, launch admission and allocation scripts. Used by relaxed-gradients, carbon-chain, known-ligand and BRD4 initialization controls.
- [evidence tooling](../../../scripts/evidence/README.md): Authenticated artifact collection, transfer orchestration and byte verification. Used by native, BRD4, JQ1 and reporting campaigns.
- [releases tooling](../../../scripts/releases/README.md): Source archive packaging and authentication. Used by every experiment; include experiments/, canonical scripts and legacy adapters in every future release.
- [hpc tooling](../../../scripts/hpc/README.md): Shared quota admission, node-local temporary storage and verified archive recovery. These existing files stay at their canonical scripts/hpc paths unchanged.

Large arrays, structures, checkpoints, reports and sealed source bundles remain at their existing approved HPC/ignored `results/hpc/` and `results/releases/` locations. Resolve exact paths and identities from the run receipts; this migration creates no replacement scientific artifact storage.

## Commands and execution location

Working directory: repository/release root. Local work is limited to source inspection, standard-library path/AST/byte checks and packaging. Model imports, numerical/chemistry tests and scientific rendering retain the experiment's allocated DTU HPC requirement.

The exact non-numerical layout check is:

```powershell
py -3.13 scripts/layout/verify_layout.py --root .
```

For real submissions, use the complete resource/argv vectors in the linked protocol and saved run receipts, replacing working-source file references through the migration map and sealing a fresh source manifest. Old receipt scripts are evidence, not reusable launchers. This reorganization does not authorize replay, restart or new scientific execution.

## Runs and logbook

Global chronological history: [experiment ledger](../../../docs/experiment_ledger.md). The records below index preserved attempts; their quoted saved states are not fresh scheduler observations.

| Historical attempt | Record |
| --- | --- |
| `20261019-brd4-chemistry` | [Run record](runs/20261019-brd4-chemistry/README.md) |
