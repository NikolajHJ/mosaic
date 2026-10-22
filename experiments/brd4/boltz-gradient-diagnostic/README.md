# BRD4 Boltz gradient diagnostics

> **Account portability, reviewed 22 October 2026:** historical remote paths
> are operator-specific. Existing controllers/LSF launchers may still pin
> private roots, runtimes and proofs; setting example environment variables
> does not override those constants. Follow the
> [account-independent HPC guide](../../../docs/HPC_GUIDE.md) and inspect
> the actual entry point before using another account. Current immutable
> releases and recorded scientific gates remain unchanged.

Experiment ID: `brd4/boltz-gradient-diagnostic`.

Investigate failed raw-gradient/finite-difference behavior without promoting a diagnostic or changing the historical native acceptance result.

## Protocol, lineage and current status

Working sources organized on 22 October 2026. **No numerical validation or scientific experiment was executed as part of this migration.** Existing numerical outcomes, failed attempts and currently allocated jobs keep their original source/manifest bindings. The migrated working tree requires fresh allocated validation before future scientific execution; old CPU/native certificates are not certificates for these new source paths.

The following existing protocol/history documents remain in place to preserve historical links and exact commands:

- [brd4_boltz_random_gradient_diagnostic](../../../docs/brd4_boltz_random_gradient_diagnostic.md).
- [brd4_boltz_balance_review](../../../docs/brd4_boltz_balance_review.md).

They define the actual model/context, frozen/trainable boundaries, gates, optimizer/search, seeds, runtime/quota limits and interpretation limits. Library algorithms remain in `src/mosaic/`; authenticated molecular fixtures/oracle snapshots remain shared under `tests/data/`. No loss, coefficient, allocation limit, native guard or acceptance rule is intentionally changed by this path migration.

## Owned files

| Canonical file | Purpose | Preserved legacy entry/input |
| --- | --- | --- |
| [analyze_boltz_balance.py](scripts/analyze_boltz_balance.py) | Allocated saved-evidence analysis; never rerun a model or chemistry evaluator. | `scripts/analyze_boltz_balance.py` |
| [analyze_boltz_gradient_diagnostic.py](scripts/analyze_boltz_gradient_diagnostic.py) | Allocated NumPy analysis of authenticated saved native diagnostic records. | `scripts/analyze_boltz_gradient_diagnostic.py` |
| [check_boltz_diagnostic_cpu.py](scripts/check_boltz_diagnostic_cpu.py) | Authenticate persisted CPU certificates without depending on live LSF retention. | `scripts/check_boltz_diagnostic_cpu.py` |
| [diagnose_boltz_random_gradient.py](scripts/diagnose_boltz_random_gradient.py) | Allocated, evidence-only replay of the failed random Boltz bond derivative. | `scripts/diagnose_boltz_random_gradient.py` |
| [brd4_boltz_diagnostic_certificate_cpu.lsf](hpc/brd4_boltz_diagnostic_certificate_cpu.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_boltz_diagnostic_certificate_cpu.lsf` |
| [brd4_boltz_diagnostic_cpu.lsf](hpc/brd4_boltz_diagnostic_cpu.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_boltz_diagnostic_cpu.lsf` |
| [brd4_boltz_gradient_analysis_cpu.lsf](hpc/brd4_boltz_gradient_analysis_cpu.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_boltz_gradient_analysis_cpu.lsf` |
| [brd4_boltz_random_diagnostic.lsf](hpc/brd4_boltz_random_diagnostic.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_boltz_random_diagnostic.lsf` |
| [brd4_boltz_random_diagnostic_retry.lsf](hpc/brd4_boltz_random_diagnostic_retry.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_boltz_random_diagnostic_retry.lsf` |
| [test_boltz_diagnostic_cpu.py](tests/test_boltz_diagnostic_cpu.py) | Allocated stdlib fixtures for persisted diagnostic producer certificates. | `tests/test_boltz_diagnostic_cpu.py` |
| [test_boltz_gradient_diagnostic_contract.py](tests/test_boltz_gradient_diagnostic_contract.py) | Allocated stdlib contracts: diagnostic replay cannot weaken readiness gates. | `tests/test_boltz_gradient_diagnostic_contract.py` |

The complete [old-to-new map](../../MIGRATION.md) records each compatibility exception. Legacy scripts contain forwarding code only. Declarative compatibility copies remain at old config/template paths for callers that open those paths directly; use the canonical files for new work and keep bytes synchronized.

## Shared dependencies and artifact storage

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
python experiments/brd4/boltz-gradient-diagnostic/scripts/analyze_boltz_gradient_diagnostic.py --help
python experiments/brd4/boltz-gradient-diagnostic/scripts/check_boltz_diagnostic_cpu.py --help
python experiments/brd4/boltz-gradient-diagnostic/scripts/diagnose_boltz_random_gradient.py --help
```

Allocated experiment-only test command, using the environment specified in the protocol:

```bash
python -m pytest experiments/brd4/boltz-gradient-diagnostic/tests/test_boltz_diagnostic_cpu.py experiments/brd4/boltz-gradient-diagnostic/tests/test_boltz_gradient_diagnostic_contract.py
```

For real submissions, use the complete resource/argv vectors in the linked protocol and saved run receipts, replacing working-source file references through the migration map and sealing a fresh source manifest. Old receipt scripts are evidence, not reusable launchers. This reorganization does not authorize replay, restart or new scientific execution.

## Runs and logbook

Global chronological history: [experiment ledger](../../../docs/experiment_ledger.md). The records below index preserved attempts; their quoted saved states are not fresh scheduler observations.

| Historical attempt | Record |
| --- | --- |
| `20261020-brd4-boltz-balance-01` | [Run record](runs/20261020-brd4-boltz-balance-01/README.md) |
| `20261020-brd4-boltz-diagnostic-launch-05` | [Run record](runs/20261020-brd4-boltz-diagnostic-launch-05/README.md) |
| `20261020-brd4-boltz-diagnostic-launch-06` | [Run record](runs/20261020-brd4-boltz-diagnostic-launch-06/README.md) |
| `20261020-brd4-boltz-gradient-analysis-01` | [Run record](runs/20261020-brd4-boltz-gradient-analysis-01/README.md) |
