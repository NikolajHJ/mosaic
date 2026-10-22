# JQ1 higher fixed learning rates

> **Account portability, reviewed 22 October 2026:** historical remote paths
> are operator-specific. Existing controllers/LSF launchers may still pin
> private roots, runtimes and proofs; setting example environment variables
> does not override those constants. Follow the
> [account-independent HPC guide](../../../docs/HPC_GUIDE.md) and inspect
> the actual entry point before using another account. Current immutable
> releases and recorded scientific gates remain unchanged.

Experiment ID: `brd4-jq1/fixed-lrboost`.

Compare Protenix LR 3.0 and both Boltz branches LR 0.05 for 10000 fresh updates, chemistry 0.001 held to 5000 then geometric to 50; preserve failed terminal derivatives.

## Protocol, lineage and current status

Working sources organized on 22 October 2026. **No numerical validation or scientific experiment was executed as part of this migration.** Existing numerical outcomes, failed attempts and currently allocated jobs keep their original source/manifest bindings. The migrated working tree requires fresh allocated validation before future scientific execution; old CPU/native certificates are not certificates for these new source paths.

The following existing protocol/history documents remain in place to preserve historical links and exact commands:

- [brd4_jq1_25_fixed_lrboost](../../../docs/brd4_jq1_25_fixed_lrboost.md).
- [brd4_jq1_25_endpoint_diagnostics](../../../docs/brd4_jq1_25_endpoint_diagnostics.md).
- [brd4_jq1_25_lrboost_report_plan](../../../docs/brd4_jq1_25_lrboost_report_plan.md).

They define the actual model/context, frozen/trainable boundaries, gates, optimizer/search, seeds, runtime/quota limits and interpretation limits. Library algorithms remain in `src/mosaic/`; authenticated molecular fixtures/oracle snapshots remain shared under `tests/data/`. No loss, coefficient, allocation limit, native guard or acceptance rule is intentionally changed by this path migration.

Declared configuration files (preserved byte for byte):

- [brd4_jq1_25_fixed_lrboost_boltz2_10000.json](configs/brd4_jq1_25_fixed_lrboost_boltz2_10000.json): `{"seed":17,"steps":10000,"initial_step_size":0.05,"initialization":{"kind":"chain_uniform","mixing":0.5},"version":"mosaic-brd4-jq1-25-fixed-step-lrboost-v1","arm":"boltz2","n_atoms":25,"repair_schedule":null,"acceptance_policy":"fixed_step"}`
- [brd4_jq1_25_fixed_lrboost_boltz2_chemistry_10000.json](configs/brd4_jq1_25_fixed_lrboost_boltz2_chemistry_10000.json): `{"seed":17,"steps":10000,"initial_step_size":0.05,"initialization":{"kind":"chain_uniform","mixing":0.5},"version":"mosaic-brd4-jq1-25-fixed-step-lrboost-v1","arm":"boltz2_chemistry","n_atoms":25,"repair_schedule":{"version":"delayed-geometric-v1","initial":0.001,"final":50.0,"hold_updates":5000,"total_updates":10000},"acceptance_policy":"fixed_step"}`
- [brd4_jq1_25_fixed_lrboost_protenix_10000.json](configs/brd4_jq1_25_fixed_lrboost_protenix_10000.json): `{"seed":17,"steps":10000,"initial_step_size":3.0,"initialization":{"kind":"chain_uniform","mixing":0.5},"version":"mosaic-brd4-jq1-25-fixed-step-lrboost-v1","arm":"protenix","n_atoms":25,"repair_schedule":null,"acceptance_policy":"fixed_step"}`

## Owned files

| Canonical file | Purpose | Preserved legacy entry/input |
| --- | --- | --- |
| [fixed25_lrboost_ops.py](scripts/fixed25_lrboost_ops.py) | Fresh high-rate fixed-step JQ1 campaign after new allocated CPU/native gates. | `scripts/fixed25_lrboost_ops.py` |
| [fixed25_lrboost_storage.py](scripts/fixed25_lrboost_storage.py) | Fresh three-arm high-rate campaign storage, independent of old run placement. | `scripts/fixed25_lrboost_storage.py` |
| [capture_jq1_lrboost_completion.py](scripts/capture_jq1_lrboost_completion.py) | Allocated saved-frame audit of passed or explicitly failed native studies. | `scripts/capture_jq1_lrboost_completion.py` |
| [jq1_lrboost_report_contract.py](scripts/jq1_lrboost_report_contract.py) | Saved-only high-rate JQ1 report contract, with explicit failed-run scope. | `scripts/jq1_lrboost_report_contract.py` |
| [jq1_lrboost_report_ops.py](scripts/jq1_lrboost_report_ops.py) | Separate allocated endpoint diagnostics and saved-only report orchestration. | `scripts/jq1_lrboost_report_ops.py` |
| [lrboost_endpoint_diagnostics.py](scripts/lrboost_endpoint_diagnostics.py) | Standalone allocated diagnostics of two preserved failed native endpoint checks. | `scripts/lrboost_endpoint_diagnostics.py` |
| [publish_jq1_lrboost_report.py](scripts/publish_jq1_lrboost_report.py) | Publish independently byte-verified higher-rate JQ1 report assets. | `scripts/publish_jq1_lrboost_report.py` |
| [render_jq1_lrboost_progress.py](scripts/render_jq1_lrboost_progress.py) | Allocated higher-rate progress editions from authenticated saved prefixes only. | `scripts/render_jq1_lrboost_progress.py` |
| [render_jq1_lrboost_report.py](scripts/render_jq1_lrboost_report.py) | Allocated higher-rate saved-only JQ1 report with exact bounded FP32 assets. | `scripts/render_jq1_lrboost_report.py` |
| [fixed25_lrboost_cpu.lsf](hpc/fixed25_lrboost_cpu.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/fixed25_lrboost_cpu.lsf` |
| [fixed25_lrboost_stage.lsf](hpc/fixed25_lrboost_stage.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/fixed25_lrboost_stage.lsf` |
| [jq1_lrboost_analysis_stage.lsf](hpc/jq1_lrboost_analysis_stage.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/jq1_lrboost_analysis_stage.lsf` |
| [brd4_jq1_25_fixed_lrboost_boltz2_10000.json](configs/brd4_jq1_25_fixed_lrboost_boltz2_10000.json) | Declared configuration/input; bytes preserved exactly. | `examples/configs/brd4_jq1_25_fixed_lrboost_boltz2_10000.json` |
| [brd4_jq1_25_fixed_lrboost_boltz2_chemistry_10000.json](configs/brd4_jq1_25_fixed_lrboost_boltz2_chemistry_10000.json) | Declared configuration/input; bytes preserved exactly. | `examples/configs/brd4_jq1_25_fixed_lrboost_boltz2_chemistry_10000.json` |
| [brd4_jq1_25_fixed_lrboost_protenix_10000.json](configs/brd4_jq1_25_fixed_lrboost_protenix_10000.json) | Declared configuration/input; bytes preserved exactly. | `examples/configs/brd4_jq1_25_fixed_lrboost_protenix_10000.json` |
| [jq1_lrboost_report.html](templates/jq1_lrboost_report.html) | Saved-artifact report template; rendering changes no scientific outcome. | `scripts/templates/jq1_lrboost_report.html` |
| [test_fixed25_lrboost_contract.py](tests/test_fixed25_lrboost_contract.py) | Allocated scope/rate/readiness fixtures for the explicitly boosted campaign. | `tests/test_fixed25_lrboost_contract.py` |
| [test_fixed25_lrboost_ops.py](tests/test_fixed25_lrboost_ops.py) | Allocated high-rate launcher fixtures; synthetic metadata certifies no model. | `tests/test_fixed25_lrboost_ops.py` |
| [test_fixed25_lrboost_storage.py](tests/test_fixed25_lrboost_storage.py) | Allocated filesystem budget fixtures; no real quota/capacity is certified. | `tests/test_fixed25_lrboost_storage.py` |
| [test_jq1_lrboost_ops.py](tests/test_jq1_lrboost_ops.py) | Allocated orchestration contracts: resource scope, evidence and failed studies. | `tests/test_jq1_lrboost_ops.py` |
| [test_jq1_lrboost_publish.py](tests/test_jq1_lrboost_publish.py) | Allocated stdlib fixtures for immutable, byte-only fixed-report delivery. | `tests/test_jq1_lrboost_publish.py` |
| [test_jq1_lrboost_render.py](tests/test_jq1_lrboost_render.py) | Allocated CPU presentation fixtures; synthetic saved arrays never certify a run. | `tests/test_jq1_lrboost_render.py` |
| [test_jq1_lrboost_report_contract.py](tests/test_jq1_lrboost_report_contract.py) | Allocated metadata contracts for honest high-rate saved-run presentation. | `tests/test_jq1_lrboost_report_contract.py` |
| [test_lrboost_endpoint_diagnostics.py](tests/test_lrboost_endpoint_diagnostics.py) | Allocated diagnostic contract/toy fixtures; never a native endpoint pass. | `tests/test_lrboost_endpoint_diagnostics.py` |

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
python experiments/brd4-jq1/fixed-lrboost/scripts/fixed25_lrboost_ops.py --help
python experiments/brd4-jq1/fixed-lrboost/scripts/capture_jq1_lrboost_completion.py --help
python experiments/brd4-jq1/fixed-lrboost/scripts/jq1_lrboost_report_ops.py --help
```

Allocated experiment-only test command, using the environment specified in the protocol:

```bash
python -m pytest experiments/brd4-jq1/fixed-lrboost/tests/test_fixed25_lrboost_contract.py experiments/brd4-jq1/fixed-lrboost/tests/test_fixed25_lrboost_ops.py experiments/brd4-jq1/fixed-lrboost/tests/test_fixed25_lrboost_storage.py experiments/brd4-jq1/fixed-lrboost/tests/test_jq1_lrboost_ops.py experiments/brd4-jq1/fixed-lrboost/tests/test_jq1_lrboost_publish.py experiments/brd4-jq1/fixed-lrboost/tests/test_jq1_lrboost_render.py experiments/brd4-jq1/fixed-lrboost/tests/test_jq1_lrboost_report_contract.py experiments/brd4-jq1/fixed-lrboost/tests/test_lrboost_endpoint_diagnostics.py
```

For real submissions, use the complete resource/argv vectors in the linked protocol and saved run receipts, replacing working-source file references through the migration map and sealing a fresh source manifest. Old receipt scripts are evidence, not reusable launchers. This reorganization does not authorize replay, restart or new scientific execution.

## Runs and logbook

Global chronological history: [experiment ledger](../../../docs/experiment_ledger.md). The records below index preserved attempts; their quoted saved states are not fresh scheduler observations.

| Historical attempt | Record |
| --- | --- |
| `20261021-brd4-jq1-25-fixed-lrboost-01` | [Run record](runs/20261021-brd4-jq1-25-fixed-lrboost-01/README.md) |
| `20261022-brd4-jq1-lrboost-analysis-01` | [Run record](runs/20261022-brd4-jq1-lrboost-analysis-01/README.md) |
| `20261022-brd4-jq1-lrboost-viewer-focus-01` | [Run record](runs/20261022-brd4-jq1-lrboost-viewer-focus-01/README.md) |
