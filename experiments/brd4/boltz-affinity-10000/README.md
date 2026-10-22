# BRD4 frozen Boltz-2 affinity optimization

> **Account portability, reviewed 22 October 2026:** historical remote paths
> are operator-specific. Existing controllers/LSF launchers may still pin
> private roots, runtimes and proofs; setting example environment variables
> does not override those constants. Follow the
> [account-independent HPC guide](../../../docs/HPC_GUIDE.md) and inspect
> the actual entry point before using another account. Current immutable
> releases and recorded scientific gates remain unchanged.

Experiment ID: `brd4/boltz-affinity-10000`.

Optimize the frozen detached-context Boltz-2 affinity branch with its separate raw native gradients, contact/affinity interpretation and independent arms.

## Protocol, lineage and current status

Working sources organized on 22 October 2026. **No numerical validation or scientific experiment was executed as part of this migration.** Existing numerical outcomes, failed attempts and currently allocated jobs keep their original source/manifest bindings. The migrated working tree requires fresh allocated validation before future scientific execution; old CPU/native certificates are not certificates for these new source paths.

The following existing protocol/history documents remain in place to preserve historical links and exact commands:

- [brd4_boltz_affinity_10000](../../../docs/brd4_boltz_affinity_10000.md).
- [brd4_boltz_affinity_report](../../../docs/brd4_boltz_affinity_report.md).

They define the actual model/context, frozen/trainable boundaries, gates, optimizer/search, seeds, runtime/quota limits and interpretation limits. Library algorithms remain in `src/mosaic/`; authenticated molecular fixtures/oracle snapshots remain shared under `tests/data/`. No loss, coefficient, allocation limit, native guard or acceptance rule is intentionally changed by this path migration.

Declared configuration files (preserved byte for byte):

- [brd4_boltz_affinity_mix50_10000.json](configs/brd4_boltz_affinity_mix50_10000.json): `{"n_atoms":6,"seed":17,"steps":10000,"initial_step_size":0.03,"initialization":{"kind":"chain_uniform","mixing":0.5},"repair_schedule":{"version":"delayed-geometric-v1","initial":0.001,"final":50.0,"hold_updates":5000,"total_updates":10000},"version":"brd4-boltz-affinity-chemistry-v1"}`
- [brd4_boltz_affinity_random_10000.json](configs/brd4_boltz_affinity_random_10000.json): `{"n_atoms":6,"seed":17,"steps":10000,"initial_step_size":0.03,"initialization":{"kind":"dirichlet","concentration":1.0,"seed":20261019},"repair_schedule":{"version":"delayed-geometric-v1","initial":0.001,"final":50.0,"hold_updates":5000,"total_updates":10000},"version":"brd4-boltz-affinity-chemistry-v1"}`

## Owned files

| Canonical file | Purpose | Preserved legacy entry/input |
| --- | --- | --- |
| [assess_boltz_arm.py](scripts/assess_boltz_arm.py) | Adopt a passed arm from a failed aggregate readiness, without rerunning science. | `scripts/assess_boltz_arm.py` |
| [assess_boltz_cpu.py](scripts/assess_boltz_cpu.py) | Authenticate allocated CPU test outcomes before native GPU readiness. | `scripts/assess_boltz_cpu.py` |
| [assess_boltz_native.py](scripts/assess_boltz_native.py) | Seal actual native gates/pilots for both requested initializations. | `scripts/assess_boltz_native.py` |
| [submit_boltz_arms.py](scripts/submit_boltz_arms.py) | Allocated controller: submit independent arms only after actual native success. | `scripts/submit_boltz_arms.py` |
| [render_boltz_affinity.py](scripts/render_boltz_affinity.py) | Allocated saved-only Boltz affinity report with authenticated raw trajectories. | `scripts/render_boltz_affinity.py` |
| [brd4_boltz_cpu.lsf](hpc/brd4_boltz_cpu.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_boltz_cpu.lsf` |
| [brd4_boltz_independent_cpu.lsf](hpc/brd4_boltz_independent_cpu.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_boltz_independent_cpu.lsf` |
| [brd4_boltz_launcher_cpu.lsf](hpc/brd4_boltz_launcher_cpu.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_boltz_launcher_cpu.lsf` |
| [brd4_boltz_ready.lsf](hpc/brd4_boltz_ready.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_boltz_ready.lsf` |
| [brd4_boltz_report.lsf](hpc/brd4_boltz_report.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_boltz_report.lsf` |
| [brd4_boltz_run.lsf](hpc/brd4_boltz_run.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_boltz_run.lsf` |
| [brd4_boltz_submit_queue.lsf](hpc/brd4_boltz_submit_queue.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_boltz_submit_queue.lsf` |
| [brd4_boltz_affinity.py](scripts/brd4_boltz_affinity.py) | Allocated Boltz affinity readiness or a fresh Mosaic delayed-chemistry arm. | `examples/brd4_boltz_affinity.py` |
| [brd4_boltz_affinity_mix50_10000.json](configs/brd4_boltz_affinity_mix50_10000.json) | Declared configuration/input; bytes preserved exactly. | `examples/configs/brd4_boltz_affinity_mix50_10000.json` |
| [brd4_boltz_affinity_random_10000.json](configs/brd4_boltz_affinity_random_10000.json) | Declared configuration/input; bytes preserved exactly. | `examples/configs/brd4_boltz_affinity_random_10000.json` |
| [test_boltz_affinity_report.py](tests/test_boltz_affinity_report.py) | Allocated saved-artifact report fixtures; no molecular scorer is executed. | `tests/test_boltz_affinity_report.py` |
| [test_boltz_arm_readiness.py](tests/test_boltz_arm_readiness.py) | Evidence-only adoption contracts; run on an allocated DTU HPC CPU. | `tests/test_boltz_arm_readiness.py` |
| [test_boltz_hpc_tmpdir.py](tests/test_boltz_hpc_tmpdir.py) | Allocated Linux CPU regression checks for native Boltz AF_UNIX startup. | `tests/test_boltz_hpc_tmpdir.py` |
| [test_boltz_native_mapping.py](tests/test_boltz_native_mapping.py) | Allocated CPU tests of native Torch mapping; no JAX/decoder dependencies. | `tests/test_boltz_native_mapping.py` |
| [test_boltz_submission_controller.py](tests/test_boltz_submission_controller.py) | Stdlib launch-contract checks; execute only on allocated DTU HPC CPU. | `tests/test_boltz_submission_controller.py` |

The complete [old-to-new map](../../MIGRATION.md) records each compatibility exception. Legacy scripts contain forwarding code only. Declarative compatibility copies remain at old config/template paths for callers that open those paths directly; use the canonical files for new work and keep bytes synchronized.

## Shared dependencies and artifact storage

- [boltz tooling](../../../scripts/boltz/README.md): Shared allocated CPU-proof verification used by Boltz and the three-model/JQ1 controllers. Scientific model code remains in src/mosaic/.
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

Canonical CLI discovery on the authorized allocation (not executed by the migration):

```bash
python experiments/brd4/boltz-affinity-10000/scripts/assess_boltz_arm.py --help
python experiments/brd4/boltz-affinity-10000/scripts/assess_boltz_cpu.py --help
python experiments/brd4/boltz-affinity-10000/scripts/assess_boltz_native.py --help
```

Allocated experiment-only test command, using the environment specified in the protocol:

```bash
python -m pytest experiments/brd4/boltz-affinity-10000/tests/test_boltz_affinity_report.py experiments/brd4/boltz-affinity-10000/tests/test_boltz_arm_readiness.py experiments/brd4/boltz-affinity-10000/tests/test_boltz_hpc_tmpdir.py experiments/brd4/boltz-affinity-10000/tests/test_boltz_native_mapping.py experiments/brd4/boltz-affinity-10000/tests/test_boltz_submission_controller.py
```

For real submissions, use the complete resource/argv vectors in the linked protocol and saved run receipts, replacing working-source file references through the migration map and sealing a fresh source manifest. Old receipt scripts are evidence, not reusable launchers. This reorganization does not authorize replay, restart or new scientific execution.

## Runs and logbook

Global chronological history: [experiment ledger](../../../docs/experiment_ledger.md). The records below index preserved attempts; their quoted saved states are not fresh scheduler observations.

| Historical attempt | Record |
| --- | --- |
| `20261019-brd4-boltz-affinity-01` | [Run record](runs/20261019-brd4-boltz-affinity-01/README.md) |
| `20261019-brd4-boltz-affinity-02` | [Run record](runs/20261019-brd4-boltz-affinity-02/README.md) |
| `20261019-brd4-boltz-independent-01` | [Run record](runs/20261019-brd4-boltz-independent-01/README.md) |
| `20261019-brd4-boltz-independent-02` | [Run record](runs/20261019-brd4-boltz-independent-02/README.md) |
| `20261020-brd4-boltz-mix50-completion-02` | [Run record](runs/20261020-brd4-boltz-mix50-completion-02/README.md) |
| `20261020-brd4-boltz-report-01` | [Run record](runs/20261020-brd4-boltz-report-01/README.md) |
