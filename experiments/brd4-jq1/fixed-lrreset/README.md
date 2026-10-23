# JQ1 lower-rate restart, chemistry endpoint 100

> **Account portability, reviewed 22 October 2026:** historical remote paths
> are operator-specific. Existing controllers/LSF launchers may still pin
> private roots, runtimes and proofs; setting example environment variables
> does not override those constants. Follow the
> [account-independent HPC guide](../../../docs/HPC_GUIDE.md) and inspect
> the actual entry point before using another account. Current immutable
> releases and recorded scientific gates remain unchanged.

Experiment ID: `brd4-jq1/fixed-lrreset`.

Compare fresh Protenix LR 0.3 and both Boltz branches LR 0.005, with chemistry 0.001 held to 5000 then geometric to 100. Existing attempt01 failure and sealed attempt02 remain independent of this migration.

## Protocol, lineage and current status

Working sources organized on 22 October 2026. **No numerical validation or scientific experiment was executed as part of this migration.** Existing numerical outcomes, failed attempts and currently allocated jobs keep their original source/manifest bindings. The migrated working tree requires fresh allocated validation before future scientific execution; old CPU/native certificates are not certificates for these new source paths.

The following existing protocol/history documents remain in place to preserve historical links and exact commands:

- [brd4_jq1_25_fixed_lrreset](../../../docs/brd4_jq1_25_fixed_lrreset.md).

They define the actual model/context, frozen/trainable boundaries, gates, optimizer/search, seeds, runtime/quota limits and interpretation limits. Library algorithms remain in `src/mosaic/`; authenticated molecular fixtures/oracle snapshots remain shared under `tests/data/`. No loss, coefficient, allocation limit, native guard or acceptance rule is intentionally changed by this path migration.

Declared configuration files (preserved byte for byte):

- [lrreset_predecessors.json](configs/lrreset_predecessors.json): `{"version":"jq1-lrreset-finished-predecessors-v1"}`
- [brd4_jq1_25_fixed_lrreset_boltz2_10000.json](configs/brd4_jq1_25_fixed_lrreset_boltz2_10000.json): `{"seed":17,"steps":10000,"initial_step_size":0.005,"initialization":{"kind":"chain_uniform","mixing":0.5},"version":"mosaic-brd4-jq1-25-fixed-step-lrreset-v1","arm":"boltz2","n_atoms":25,"repair_schedule":null,"acceptance_policy":"fixed_step"}`
- [brd4_jq1_25_fixed_lrreset_boltz2_chemistry_10000.json](configs/brd4_jq1_25_fixed_lrreset_boltz2_chemistry_10000.json): `{"seed":17,"steps":10000,"initial_step_size":0.005,"initialization":{"kind":"chain_uniform","mixing":0.5},"version":"mosaic-brd4-jq1-25-fixed-step-lrreset-v1","arm":"boltz2_chemistry","n_atoms":25,"repair_schedule":{"version":"delayed-geometric-v1","initial":0.001,"final":100.0,"hold_updates":5000,"total_updates":10000},"acceptance_policy":"fixed_step"}`
- [brd4_jq1_25_fixed_lrreset_protenix_10000.json](configs/brd4_jq1_25_fixed_lrreset_protenix_10000.json): `{"seed":17,"steps":10000,"initial_step_size":0.3,"initialization":{"kind":"chain_uniform","mixing":0.5},"version":"mosaic-brd4-jq1-25-fixed-step-lrreset-v1","arm":"protenix","n_atoms":25,"repair_schedule":null,"acceptance_policy":"fixed_step"}`

## Owned files

| Canonical file | Purpose | Preserved legacy entry/input |
| --- | --- | --- |
| [fixed25_lrreset_ops.py](scripts/fixed25_lrreset_ops.py) | Fresh lower-rate fixed-step JQ1 campaign after new allocated CPU/native gates. | `scripts/fixed25_lrreset_ops.py` |
| [fixed25_lrreset_storage.py](scripts/fixed25_lrreset_storage.py) | Fresh three-arm high-rate campaign storage, independent of old run placement. | `scripts/fixed25_lrreset_storage.py` |
| [fixed25_lrreset_cpu.lsf](hpc/fixed25_lrreset_cpu.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/fixed25_lrreset_cpu.lsf` |
| [fixed25_lrreset_stage.lsf](hpc/fixed25_lrreset_stage.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/fixed25_lrreset_stage.lsf` |
| [lrreset_predecessors.json](configs/lrreset_predecessors.json) | Declared configuration/input; bytes preserved exactly. | `scripts/hpc/lrreset_predecessors.json` |
| [brd4_jq1_25_fixed_lrreset_boltz2_10000.json](configs/brd4_jq1_25_fixed_lrreset_boltz2_10000.json) | Declared configuration/input; bytes preserved exactly. | `examples/configs/brd4_jq1_25_fixed_lrreset_boltz2_10000.json` |
| [brd4_jq1_25_fixed_lrreset_boltz2_chemistry_10000.json](configs/brd4_jq1_25_fixed_lrreset_boltz2_chemistry_10000.json) | Declared configuration/input; bytes preserved exactly. | `examples/configs/brd4_jq1_25_fixed_lrreset_boltz2_chemistry_10000.json` |
| [brd4_jq1_25_fixed_lrreset_protenix_10000.json](configs/brd4_jq1_25_fixed_lrreset_protenix_10000.json) | Declared configuration/input; bytes preserved exactly. | `examples/configs/brd4_jq1_25_fixed_lrreset_protenix_10000.json` |
| [test_fixed25_lrreset_contract.py](tests/test_fixed25_lrreset_contract.py) | Allocated scope/rate/readiness fixtures for the explicitly boosted campaign. | `tests/test_fixed25_lrreset_contract.py` |
| [test_fixed25_lrreset_ops.py](tests/test_fixed25_lrreset_ops.py) | Allocated high-rate launcher fixtures; synthetic metadata certifies no model. | `tests/test_fixed25_lrreset_ops.py` |
| [test_fixed25_lrreset_storage.py](tests/test_fixed25_lrreset_storage.py) | Allocated filesystem budget fixtures; no real quota/capacity is certified. | `tests/test_fixed25_lrreset_storage.py` |

The complete [old-to-new map](../../MIGRATION.md) records each compatibility exception. Legacy scripts contain forwarding code only. Declarative compatibility copies remain at old config/template paths for callers that open those paths directly; use the canonical files for new work and keep bytes synchronized.

## Shared dependencies and artifact storage

- [campaign25 tooling](../../../scripts/campaign25/README.md): Shared 25-slot driver, versioned protocol contracts, pocket-controller utilities and storage accounting. Used by BRD4 three-model, JQ1 backtracking/LR0.3/fixed/higher-rate/lower-rate campaigns. Fixed storage helpers have one owner here.
- [boltz tooling](../../../scripts/boltz/README.md): Shared allocated CPU-proof verification used by Boltz and the three-model/JQ1 controllers. Scientific model code remains in src/mosaic/.
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
python experiments/brd4-jq1/fixed-lrreset/scripts/fixed25_lrreset_ops.py --help
```

Allocated experiment-only test command, using the environment specified in the protocol:

```bash
python -m pytest experiments/brd4-jq1/fixed-lrreset/tests/test_fixed25_lrreset_contract.py experiments/brd4-jq1/fixed-lrreset/tests/test_fixed25_lrreset_ops.py experiments/brd4-jq1/fixed-lrreset/tests/test_fixed25_lrreset_storage.py
```

For real submissions, use the complete resource/argv vectors in the linked protocol and saved run receipts, replacing working-source file references through the migration map and sealing a fresh source manifest. Old receipt scripts are evidence, not reusable launchers. This reorganization does not authorize replay, restart or new scientific execution.

## Runs and logbook

Global chronological history: [experiment ledger](../../../docs/experiment_ledger.md). The records below index preserved attempts; their quoted saved states are not fresh scheduler observations.

| Historical attempt | Record |
| --- | --- |
| `20261022-brd4-jq1-25-fixed-lrreset-01` | [Run record](runs/20261022-brd4-jq1-25-fixed-lrreset-01/README.md) |
| `20261022-brd4-jq1-25-fixed-lrreset-02` | [Run record](runs/20261022-brd4-jq1-25-fixed-lrreset-02/README.md) |

Current-source preservation: saved attempt02 source manifest `832f833dd486c2ff305f7e3eb33fc775c87810a2e12a8bc597c5df744699c5d3` at `/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-release-02/source` is external and untouched. Saved initial jobs are CPU `29781363` and callback `29781364`; no fresh scheduler state or pass is claimed here.

A concurrent agent's later [ledger entry](../../../docs/experiment_ledger.md)
records the original sealed attempt02 CPU stage passing 723 cases and its
callback stopping at native-dispatch storage admission, before native/long
submission. It also describes a separate authorized cleanup in progress. Those
updates are preserved, are independent of this migration, and do not validate
the migrated working source. Consult that agent's latest receipts for follow-up.


Latest authorized initialization recovery: [attempt04](runs/20261022-brd4-jq1-25-fixed-lrreset-04/README.md), with exact baseline identity preserved and all fresh gates repeated. [Attempt03](runs/20261022-brd4-jq1-25-fixed-lrreset-03/README.md) remains the preserved Pro initialization failure and two passed Boltz validations; no old run is restarted.


23October current separate endpoint/report investigation: [analysis03](runs/20261023-brd4-jq1-lrreset-analysis-03/README.md). OriginalPro29792670 passed10000/finalderivatives; normalBol29792671 saved10000 but failedterminalE_upperFD; chemistry29792672 remainsoriginalqueuedjob. Full284analysisCPUfixtures passed; separateBoltzdiagnostic and saved-only collectors/renderers submitted, no originalrun restart or settingchange.
