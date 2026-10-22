# JQ1-pocket three-arm comparison

> **Account portability, reviewed 22 October 2026:** historical remote paths
> are operator-specific. Existing controllers/LSF launchers may still pin
> private roots, runtimes and proofs; setting example environment variables
> does not override those constants. Follow the
> [account-independent HPC guide](../../../docs/HPC_GUIDE.md) and inspect
> the actual entry point before using another account. Current immutable
> releases and recorded scientific gates remain unchanged.

Experiment ID: `brd4-jq1/backtracking-25`.

Compare the three frozen-model branches in the authenticated JQ1-defined BRD4 pocket using the original declared acceptance policy.

## Protocol, lineage and current status

Working sources organized on 22 October 2026. **No numerical validation or scientific experiment was executed as part of this migration.** Existing numerical outcomes, failed attempts and currently allocated jobs keep their original source/manifest bindings. The migrated working tree requires fresh allocated validation before future scientific execution; old CPU/native certificates are not certificates for these new source paths.

The following existing protocol/history documents remain in place to preserve historical links and exact commands:

- [brd4_jq1_25_atom_comparison](../../../docs/brd4_jq1_25_atom_comparison.md).
- [brd4_jq1_25_report](../../../docs/brd4_jq1_25_report.md).

They define the actual model/context, frozen/trainable boundaries, gates, optimizer/search, seeds, runtime/quota limits and interpretation limits. Library algorithms remain in `src/mosaic/`; authenticated molecular fixtures/oracle snapshots remain shared under `tests/data/`. No loss, coefficient, allocation limit, native guard or acceptance rule is intentionally changed by this path migration.

Declared configuration files (preserved byte for byte):

- [brd4_jq1_25_boltz2_10000.json](configs/brd4_jq1_25_boltz2_10000.json): `{"seed":17,"steps":10000,"initial_step_size":0.03,"initialization":{"kind":"chain_uniform","mixing":0.5},"version":"mosaic-brd4-jq1-25-three-arm-v1","arm":"boltz2","n_atoms":25,"repair_schedule":null}`
- [brd4_jq1_25_boltz2_chemistry_10000.json](configs/brd4_jq1_25_boltz2_chemistry_10000.json): `{"seed":17,"steps":10000,"initial_step_size":0.03,"initialization":{"kind":"chain_uniform","mixing":0.5},"version":"mosaic-brd4-jq1-25-three-arm-v1","arm":"boltz2_chemistry","n_atoms":25,"repair_schedule":{"version":"delayed-geometric-v1","initial":0.001,"final":50.0,"hold_updates":5000,"total_updates":10000}}`
- [brd4_jq1_25_protenix_10000.json](configs/brd4_jq1_25_protenix_10000.json): `{"seed":17,"steps":10000,"initial_step_size":0.03,"initialization":{"kind":"chain_uniform","mixing":0.5},"version":"mosaic-brd4-jq1-25-three-arm-v1","arm":"protenix","n_atoms":25,"repair_schedule":null}`

## Owned files

| Canonical file | Purpose | Preserved legacy entry/input |
| --- | --- | --- |
| [derive_brd4_jq1_pocket.py](scripts/derive_brd4_jq1_pocket.py) | Extract the immutable JQ1 pocket on an allocated CPU node. | `scripts/derive_brd4_jq1_pocket.py` |
| [capture_jq1_completion.py](scripts/capture_jq1_completion.py) | Capture a native final outcome promptly, before LSF live retention expires. | `scripts/capture_jq1_completion.py` |
| [jq1_report_contract.py](scripts/jq1_report_contract.py) | Saved-artifact contracts for the separate three-arm JQ1 25-slot report. | `scripts/jq1_report_contract.py` |
| [render_jq1_25_report.py](scripts/render_jq1_25_report.py) | Allocated saved-only JQ1 report with bounded, exact FP32 frame assets. | `scripts/render_jq1_25_report.py` |
| [render_jq1_25_interim.py](scripts/render_jq1_25_interim.py) | Allocated, saved-only progress editions; never certify an unfinished study. | `scripts/render_jq1_25_interim.py` |
| [publish_jq1_report.py](scripts/publish_jq1_report.py) | Publish byte-verified saved HTML assets; local stdlib file I/O only. | `scripts/publish_jq1_report.py` |
| [campaign25_pocket_cpu.lsf](hpc/campaign25_pocket_cpu.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/campaign25_pocket_cpu.lsf` |
| [campaign25_pocket_stage.lsf](hpc/campaign25_pocket_stage.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/campaign25_pocket_stage.lsf` |
| [jq1_25_interim_stage.lsf](hpc/jq1_25_interim_stage.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/jq1_25_interim_stage.lsf` |
| [jq1_25_report_stage.lsf](hpc/jq1_25_report_stage.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/jq1_25_report_stage.lsf` |
| [brd4_jq1_25_boltz2_10000.json](configs/brd4_jq1_25_boltz2_10000.json) | Declared configuration/input; bytes preserved exactly. | `examples/configs/brd4_jq1_25_boltz2_10000.json` |
| [brd4_jq1_25_boltz2_chemistry_10000.json](configs/brd4_jq1_25_boltz2_chemistry_10000.json) | Declared configuration/input; bytes preserved exactly. | `examples/configs/brd4_jq1_25_boltz2_chemistry_10000.json` |
| [brd4_jq1_25_protenix_10000.json](configs/brd4_jq1_25_protenix_10000.json) | Declared configuration/input; bytes preserved exactly. | `examples/configs/brd4_jq1_25_protenix_10000.json` |
| [jq1_25_report.html](templates/jq1_25_report.html) | Saved-artifact report template; rendering changes no scientific outcome. | `scripts/templates/jq1_25_report.html` |
| [test_jq1_25_interim.py](tests/test_jq1_25_interim.py) | Allocated fixtures for append-only prefixes and sparse interim presentation. | `tests/test_jq1_25_interim.py` |
| [test_jq1_25_report.py](tests/test_jq1_25_report.py) | Allocated report fixtures: exact saved frames, failures and bounded assets. | `tests/test_jq1_25_report.py` |

The complete [old-to-new map](../../MIGRATION.md) records each compatibility exception. Legacy scripts contain forwarding code only. Declarative compatibility copies remain at old config/template paths for callers that open those paths directly; use the canonical files for new work and keep bytes synchronized.

## Shared dependencies and artifact storage

- [campaign25 tooling](../../../scripts/campaign25/README.md): Shared 25-slot driver, versioned protocol contracts, pocket-controller utilities and storage accounting. Used by BRD4 three-model, JQ1 backtracking/LR0.3/fixed/higher-rate/lower-rate campaigns. Fixed storage helpers have one owner here.
- [boltz tooling](../../../scripts/boltz/README.md): Shared allocated CPU-proof verification used by Boltz and the three-model/JQ1 controllers. Scientific model code remains in src/mosaic/.
- [chemistry tooling](../../../scripts/chemistry/README.md): Shared chemistry driver, reference preparation, repair oracle generation/readiness and base report rendering. Used by BRD4 chemistry-5000, chemistry-ramp and Boltz-affinity reports.
- [graph tooling](../../../scripts/graph/README.md): Shared graph/control driver, fixture preparation, launch admission and allocation scripts. Used by relaxed-gradients, carbon-chain, known-ligand and BRD4 initialization controls.
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
python experiments/brd4-jq1/backtracking-25/scripts/derive_brd4_jq1_pocket.py --help
python experiments/brd4-jq1/backtracking-25/scripts/capture_jq1_completion.py --help
python experiments/brd4-jq1/backtracking-25/scripts/render_jq1_25_report.py --help
```

Allocated experiment-only test command, using the environment specified in the protocol:

```bash
python -m pytest experiments/brd4-jq1/backtracking-25/tests/test_jq1_25_interim.py experiments/brd4-jq1/backtracking-25/tests/test_jq1_25_report.py
```

For real submissions, use the complete resource/argv vectors in the linked protocol and saved run receipts, replacing working-source file references through the migration map and sealing a fresh source manifest. Old receipt scripts are evidence, not reusable launchers. This reorganization does not authorize replay, restart or new scientific execution.

## Runs and logbook

Global chronological history: [experiment ledger](../../../docs/experiment_ledger.md). The records below index preserved attempts; their quoted saved states are not fresh scheduler observations.

| Historical attempt | Record |
| --- | --- |
| `20261021-brd4-jq1-25-01` | [Run record](runs/20261021-brd4-jq1-25-01/README.md) |
| `20261021-brd4-jq1-25-report-01` | [Run record](runs/20261021-brd4-jq1-25-report-01/README.md) |
| `20261021-brd4-jq1-25-report-02` | [Run record](runs/20261021-brd4-jq1-25-report-02/README.md) |
| `20261021-brd4-jq1-25-report-03` | [Run record](runs/20261021-brd4-jq1-25-report-03/README.md) |
| `20261021-brd4-jq1-25-report-04` | [Run record](runs/20261021-brd4-jq1-25-report-04/README.md) |
| `20261021-brd4-jq1-gradient-audit-01` | [Run record](runs/20261021-brd4-jq1-gradient-audit-01/README.md) |
| `20261021-brd4-jq1-gradient-audit-02` | [Run record](runs/20261021-brd4-jq1-gradient-audit-02/README.md) |
| `20261021-brd4-jq1-gradient-audit-03` | [Run record](runs/20261021-brd4-jq1-gradient-audit-03/README.md) |
