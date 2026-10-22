# Agent instructions for Mosaic

## Experiment organization and run history — required

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
