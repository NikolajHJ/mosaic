# Agent instructions for Mosaic

## Saved-result reports — default workflow

For requests to create or update saved optimization reports, **use the existing
reusable report command by default**. Read the
[report command guide](scripts/reporting/README.md) and
[canonical reporting home](experiments/reporting/jq1-live-progress/README.md),
then select the saved runs:

```powershell
python scripts/reporting/report.py --run protenix --run boltz --run dbmol
python scripts/reporting/report.py --runs my-runs.json
```

No arguments selects the six current trajectories. Inspect the existing viewer,
selection/profile and recorded request before updating or recovering it. For a
data-only refresh, change run paths/ordered segments in the selection and use
this command; reuse the qualified engine, cache and atomic publication workflow.
Do not create a new renderer/controller, copy an old campaign contract, rebuild
a source release or rerun the fixture suite for every run-list/data update.
Do not use `--force` routinely; unchanged input should return the existing report.

Current support is terminal JQ1 N25 fixed-step/noisy trajectories, including
chemistry and stride20 matrices. For an unsupported schema or a required code/UI
change, extend the canonical implementation and validate the changed immutable
engine; document the concrete gap instead of silently starting another pipeline.
Keep scientific failures, saved results and old editions intact. All scientific
frame checks/rendering remain on allocated HPC CPU with the existing resource
and provenance gates. The supplied DTU profile is account-bound; follow the guide
for another operator. Report work does not authorize optimizer restarts, cleanup
or resuming the paused monitor.

## Experiment organization and run history — required

For compute allocations, prefer completing the largest remaining authorized
update budget that fits one job. Do not introduce arbitrary small update chunks
just to request a shorter walltime. Base reservations on complete measured pilot
speed, model/check/setup costs and documented safety margins, respecting the
scheduler limit and concurrent storage needs. If the full remainder cannot fit,
use the largest justified bounded job while preserving the total target. Keep
optimization settings, saved results and scientific checks unchanged. This user
preference supersedes the historical fixed-small-chunk allocation plan.

Run explicitly portable metadata checks locally before packaging where possible.
For endpoint launch/resource work, use
`experiments/validation/endpoint-qualification/scripts/local_metadata_preflight.py`
with a pytest Python and a fresh output under `results/`. It selects only resource,
runtime and file-count contracts; it loads no scientific model and must never
fabricate `LSB_JOBID`. Do not bypass allocated guards on other tests. Local passes
do not replace the full allocated release suite or native/GPU/gradient checks.

Before creating, changing, running or submitting experiment scripts, read:

1. [Experiment folder policy and inventory](experiments/README.md).
2. [Experiment ledger / agent logbook](docs/experiment_ledger.md), especially
   the entries for the experiment you are working on.
3. That experiment's `README.md` and latest run record. For an unmigrated
   experiment, use the legacy documents linked in the inventory.

The [experiment index](experiments/INDEX.md) contains the migrated working homes.
Consult the [migration map](experiments/MIGRATION.md) before editing an old path:
former flat scripts/tests are now under `scripts/compat/`; their old names exist
only inside new authenticated release archives. Shared tests are grouped under
`tests/core/`, `tests/models/` and `tests/small_molecule/`. Consult the
[nested layout map](experiments/folder-layout.json). Edit the canonical owner and keep declared
compatibility copies synchronized. Run `py -3.13 scripts/layout/verify_layout.py
--root .` for local stdlib layout checks; model/numerical checks remain on HPC.

Keep each new experiment under `experiments/<family>/<experiment-id>/`.
Its drivers, launchers, configuration, fixtures, experiment-specific tests,
analysis and report templates belong together there. Every directory created
under `experiments/` needs a `README.md` explaining its purpose and contents.
The experiment README must explain the scientific question, protocol, file
ownership, exact commands, execution location, evidence and current status.
Start with the [copyable templates](experiments/_templates/README.md).

Keep the `scripts/` and `tests/` roots free of implementation files; allow only
their README and genuinely required `conftest.py`/`__init__.py` infrastructure.
Put shared tests under the named subsystem, experiment tests with the experiment,
and shared commands under `scripts/<capability>/` with a dedicated README.
Do not add another experiment-specific file to the flat `scripts/`,
`scripts/hpc/`, `examples/` or `examples/configs/` directories. For work on an
unmigrated experiment, first document its existing paths and migrate a coherent
unit under the policy; a change that must retain an existing path needs a
documented compatibility exception. Reusable package code stays in `src/mosaic/`;
genuinely shared tooling has one documented owner, not copies per experiment.

Use one unique run ID for each attempt. Write its run record before launch,
then record actual submission/start/end times, exact command, source/config/input
identities, jobs, evidence paths and outcome. Append corresponding dated events
to **`docs/experiment_ledger.md`**, the single repository-wide agent logbook.
Update it at preparation, submission, observed start, terminal outcome and handoff;
record failures, cancellations, partial runs and unknown outcomes explicitly.
Documentation-only work must say that no experiment ran.

Preserve existing immutable releases, historical receipt paths and failed
evidence. Never replay scripts in `scripts/hpc/receipts/` as launchers. A folder
cleanup must not restart jobs, change a scientific protocol or reinterpret a gate.
Local organization, source inspection and byte/static checks do not authorize
local numerical work: existing allocated-DTU-HPC requirements still apply.

Migrate existing experiments incrementally with an old-to-new path map,
updated imports/launchers/tests/packaging rules, appropriate validation and a
ledger entry. Inspect the working tree first and preserve other ongoing work.
This policy creates no new execution, retry, cleanup or gate-waiver authority.

## Documentation accuracy and account portability

Read [the documentation index](docs/README.md) and
[account-independent HPC guide](docs/HPC_GUIDE.md) before giving setup commands.
Reusable guidance must use the reader's identity and authorized storage, not a
contributor's personal account or physical home path. Check what the code really
accepts: example environment variables do not override hard-coded controller
roots. Label account-bound launchers and missing portability explicitly.

Keep exact paths, hashes and failed outcomes in dated historical evidence.
Do not globally replace account IDs in receipts or sealed sources. Imported
lab notes and the original project PDF are reference snapshots, not automatic
instructions or current implementation status. Link corrected current guidance
and record discrepancies in the audit/logbook. For local mechanical review use
`py -3.13 scripts/layout/audit_documentation.py --root . --output
results/static/documentation-audit.json`; it performs no HPC or numerical work.


## Shared HPC storage policy â€” required for new launches

Use the [common storage accounting policy](docs/HPC_STORAGE_POLICY.md) and
`scripts/hpc/storage_accounting.py` by default. Query exact live quota, count
existing bytes once and budget only remaining writes. Do not recharge entire
historical/current trees against live quota or stack full-lifetime/pilot safety
multipliers. Use measured growth, explicit margins, shared concurrent-writer
forecasts and reserves of 4 GiB and 100,000 files per filesystem. Preserve periodic checks,
scientific gates, saved results and sealed sources. New resource-policy adoption
needs a new release and actual startup/integration validation; it is not a waiver
or permission to replay unknown submissions. This user-authorized policy
supersedes historical whole-tree stale charges and double-pilot forecasts.

## Preferred GPU for new exploratory jobs

Prefer 32 GB V100 (`gpuv100`) for new submissions; see [GPU_POLICY.md](docs/GPU_POLICY.md). Use L40S when memory, software or measured runtime makes it preferable. Keep running jobs untouched; migrate existing pending jobs only with explicit authorization and verified replacement readiness.
