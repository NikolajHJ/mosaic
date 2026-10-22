# BRD4 best-primal continuation

> **Account portability, reviewed 22 October 2026:** historical remote paths
> are operator-specific. Existing controllers/LSF launchers may still pin
> private roots, runtimes and proofs; setting example environment variables
> does not override those constants. Follow the
> [account-independent HPC guide](../../../docs/HPC_GUIDE.md) and inspect
> the actual entry point before using another account. Current immutable
> releases and recorded scientific gates remain unchanged.

Experiment ID: `brd4/best-primal-continuation`.

Continue selected authenticated states under the declared best-primal acceptance protocol while preserving parent state and fixed objective identity.

## Protocol, lineage and current status

Working sources organized on 22 October 2026. **No numerical validation or scientific experiment was executed as part of this migration.** Existing numerical outcomes, failed attempts and currently allocated jobs keep their original source/manifest bindings. The migrated working tree requires fresh allocated validation before future scientific execution; old CPU/native certificates are not certificates for these new source paths.

The following existing protocol/history documents remain in place to preserve historical links and exact commands:

- [brd4_best_primal_continuation](../../../docs/brd4_best_primal_continuation.md).

They define the actual model/context, frozen/trainable boundaries, gates, optimizer/search, seeds, runtime/quota limits and interpretation limits. Library algorithms remain in `src/mosaic/`; authenticated molecular fixtures/oracle snapshots remain shared under `tests/data/`. No loss, coefficient, allocation limit, native guard or acceptance rule is intentionally changed by this path migration.

## Owned files

| Canonical file | Purpose | Preserved legacy entry/input |
| --- | --- | --- |
| [prepare_brd4_continuation.py](scripts/prepare_brd4_continuation.py) | Allocated CPU readiness: seal exact parent prefixes and continuation inputs. | `scripts/prepare_brd4_continuation.py` |
| [brd4_continuation_readiness.lsf](hpc/brd4_continuation_readiness.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_continuation_readiness.lsf` |
| [brd4_continuation_report.lsf](hpc/brd4_continuation_report.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_continuation_report.lsf` |
| [brd4_continue.lsf](hpc/brd4_continue.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_continue.lsf` |
| [continue_brd4.py](scripts/continue_brd4.py) | Continue an authenticated BRD4 campaign with explicit best-primal fallback. | `examples/continue_brd4.py` |

The complete [old-to-new map](../../MIGRATION.md) records each compatibility exception. Legacy scripts contain forwarding code only. Declarative compatibility copies remain at old config/template paths for callers that open those paths directly; use the canonical files for new work and keep bytes synchronized.

## Shared dependencies and artifact storage

- [graph tooling](../../../scripts/graph/README.md): Shared graph/control driver, fixture preparation, launch admission and allocation scripts. Used by relaxed-gradients, carbon-chain, known-ligand and BRD4 initialization controls.
- [reporting tooling](../../../scripts/reporting/README.md): Generic saved Protenix trajectory renderer, raw-graph depictions, SVG display and localhost serving. Used by graph controls and BRD4 reports. Rendering still requires its existing allocated-HPC conditions.
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
python experiments/brd4/best-primal-continuation/scripts/prepare_brd4_continuation.py --help
python experiments/brd4/best-primal-continuation/scripts/continue_brd4.py --help
```

For real submissions, use the complete resource/argv vectors in the linked protocol and saved run receipts, replacing working-source file references through the migration map and sealing a fresh source manifest. Old receipt scripts are evidence, not reusable launchers. This reorganization does not authorize replay, restart or new scientific execution.

## Runs and logbook

Global chronological history: [experiment ledger](../../../docs/experiment_ledger.md). The records below index preserved attempts; their quoted saved states are not fresh scheduler observations.

| Historical attempt | Record |
| --- | --- |
| `20261017-brd4-best-primal` | [Run record](runs/20261017-brd4-best-primal/README.md) |
