# BRD4 chemistry compatibility probe

> **Account portability, reviewed 22 October 2026:** historical remote paths
> are operator-specific. Existing controllers/LSF launchers may still pin
> private roots, runtimes and proofs; setting example environment variables
> does not override those constants. Follow the
> [account-independent HPC guide](../../../docs/HPC_GUIDE.md) and inspect
> the actual entry point before using another account. Current immutable
> releases and recorded scientific gates remain unchanged.

Experiment ID: `brd4/chemistry-compatibility`.

Probe compatibility of continuous and repair-based chemistry components independently of successful production optimization.

## Protocol, lineage and current status

Working sources organized on 22 October 2026. **No numerical validation or scientific experiment was executed as part of this migration.** Existing numerical outcomes, failed attempts and currently allocated jobs keep their original source/manifest bindings. The migrated working tree requires fresh allocated validation before future scientific execution; old CPU/native certificates are not certificates for these new source paths.

The following existing protocol/history documents remain in place to preserve historical links and exact commands:

- [brd4_continuous_chemistry](../../../docs/brd4_continuous_chemistry.md).

They define the actual model/context, frozen/trainable boundaries, gates, optimizer/search, seeds, runtime/quota limits and interpretation limits. Library algorithms remain in `src/mosaic/`; authenticated molecular fixtures/oracle snapshots remain shared under `tests/data/`. No loss, coefficient, allocation limit, native guard or acceptance rule is intentionally changed by this path migration.

## Owned files

| Canonical file | Purpose | Preserved legacy entry/input |
| --- | --- | --- |
| [chemistry_compatibility_probe.py](scripts/chemistry_compatibility_probe.py) | Experimental saved-state chemistry probes; not a production loss or default. | `scripts/chemistry_compatibility_probe.py` |
| [run_chemistry_compatibility_probe.py](scripts/run_chemistry_compatibility_probe.py) | Allocated investigation of saved gradients; no Boltz execution or optimization. | `scripts/run_chemistry_compatibility_probe.py` |
| [chemistry_compatibility_probe.lsf](hpc/chemistry_compatibility_probe.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/chemistry_compatibility_probe.lsf` |
| [test_chemistry_compatibility_probe.py](tests/test_chemistry_compatibility_probe.py) | Allocated research-probe checks against independent categorical enumeration. | `tests/test_chemistry_compatibility_probe.py` |

The complete [old-to-new map](../../MIGRATION.md) records each compatibility exception. Legacy scripts contain forwarding code only. Declarative compatibility copies remain at old config/template paths for callers that open those paths directly; use the canonical files for new work and keep bytes synchronized.

## Shared dependencies and artifact storage

- [chemistry tooling](../../../scripts/chemistry/README.md): Shared chemistry driver, reference preparation, repair oracle generation/readiness and base report rendering. Used by BRD4 chemistry-5000, chemistry-ramp and Boltz-affinity reports.
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
```

Allocated experiment-only test command, using the environment specified in the protocol:

```bash
python -m pytest experiments/brd4/chemistry-compatibility/tests/test_chemistry_compatibility_probe.py
```

For real submissions, use the complete resource/argv vectors in the linked protocol and saved run receipts, replacing working-source file references through the migration map and sealing a fresh source manifest. Old receipt scripts are evidence, not reusable launchers. This reorganization does not authorize replay, restart or new scientific execution.

## Runs and logbook

Global chronological history: [experiment ledger](../../../docs/experiment_ledger.md). The records below index preserved attempts; their quoted saved states are not fresh scheduler observations.

| Historical attempt | Record |
| --- | --- |
| `20261020-brd4-chemistry-compatibility-probe-01` | [Run record](runs/20261020-brd4-chemistry-compatibility-probe-01/README.md) |
