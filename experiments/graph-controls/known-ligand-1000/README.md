# Known-ligand 1000-update control

> **Account portability, reviewed 22 October 2026:** historical remote paths
> are operator-specific. Existing controllers/LSF launchers may still pin
> private roots, runtimes and proofs; setting example environment variables
> does not override those constants. Follow the
> [account-independent HPC guide](../../../docs/HPC_GUIDE.md) and inspect
> the actual entry point before using another account. Current immutable
> releases and recorded scientific gates remain unchanged.

Experiment ID: `graph-controls/known-ligand-1000`.

Prepare and validate a known-ligand complex and compare the native and relaxed controls over the declared 1000-update budget.

## Protocol, lineage and current status

Working sources organized on 22 October 2026. **No numerical validation or scientific experiment was executed as part of this migration.** Existing numerical outcomes, failed attempts and currently allocated jobs keep their original source/manifest bindings. The migrated working tree requires fresh allocated validation before future scientific execution; old CPU/native certificates are not certificates for these new source paths.

The following existing protocol/history documents remain in place to preserve historical links and exact commands:

- [known_ligand_1000](../../../docs/known_ligand_1000.md).

They define the actual model/context, frozen/trainable boundaries, gates, optimizer/search, seeds, runtime/quota limits and interpretation limits. Library algorithms remain in `src/mosaic/`; authenticated molecular fixtures/oracle snapshots remain shared under `tests/data/`. No loss, coefficient, allocation limit, native guard or acceptance rule is intentionally changed by this path migration.

Declared configuration files (preserved byte for byte):

- [known_ligand_1000.json](configs/known_ligand_1000.json): `{"seed":17}`
- [known_ligand_pilot.json](configs/known_ligand_pilot.json): `{"seed":17}`

## Owned files

| Canonical file | Purpose | Preserved legacy entry/input |
| --- | --- | --- |
| [assess_known_ligand_pilot.py](scripts/assess_known_ligand_pilot.py) | Authenticate native readiness and conservatively budget compact continuation. | `scripts/assess_known_ligand_pilot.py` |
| [known_ligand_1000.lsf](hpc/known_ligand_1000.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/known_ligand_1000.lsf` |
| [known_ligand_assess.lsf](hpc/known_ligand_assess.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/known_ligand_assess.lsf` |
| [known_ligand_collect.lsf](hpc/known_ligand_collect.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/known_ligand_collect.lsf` |
| [known_ligand_controls.lsf](hpc/known_ligand_controls.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/known_ligand_controls.lsf` |
| [known_ligand_readiness.lsf](hpc/known_ligand_readiness.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/known_ligand_readiness.lsf` |
| [known_ligand_1000.json](configs/known_ligand_1000.json) | Declared configuration/input; bytes preserved exactly. | `examples/configs/known_ligand_1000.json` |
| [known_ligand_pilot.json](configs/known_ligand_pilot.json) | Declared configuration/input; bytes preserved exactly. | `examples/configs/known_ligand_pilot.json` |

The complete [old-to-new map](../../MIGRATION.md) records each compatibility exception. Legacy scripts contain forwarding code only. Declarative compatibility copies remain at old config/template paths for callers that open those paths directly; use the canonical files for new work and keep bytes synchronized.

## Shared dependencies and artifact storage

- [graph tooling](../../../scripts/graph/README.md): Shared graph/control driver, fixture preparation, launch admission and allocation scripts. Used by relaxed-gradients, carbon-chain, known-ligand and BRD4 initialization controls.
- [reporting tooling](../../../scripts/reporting/README.md): Generic saved Protenix trajectory renderer, raw-graph depictions, SVG display and localhost serving. Used by graph controls and BRD4 reports. Rendering still requires its existing allocated-HPC conditions.
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

Canonical CLI discovery on the authorized allocation (not executed by the migration):

```bash
python experiments/graph-controls/known-ligand-1000/scripts/assess_known_ligand_pilot.py --help
```

For real submissions, use the complete resource/argv vectors in the linked protocol and saved run receipts, replacing working-source file references through the migration map and sealing a fresh source manifest. Old receipt scripts are evidence, not reusable launchers. This reorganization does not authorize replay, restart or new scientific execution.

## Runs and logbook

Global chronological history: [experiment ledger](../../../docs/experiment_ledger.md). The records below index preserved attempts; their quoted saved states are not fresh scheduler observations.

| Historical attempt | Record |
| --- | --- |
| `20261016-known-ligand-1000` | [Run record](runs/20261016-known-ligand-1000/README.md) |
