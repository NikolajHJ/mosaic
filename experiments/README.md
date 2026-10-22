# Experiments: organization policy and inventory

Effective 22 October 2026. This is the required organization policy for agents
and contributors. The root [AGENTS.md](../AGENTS.md) and
[README](../README.md) point here. The single repository-wide run logbook is
[docs/experiment_ledger.md](../docs/experiment_ledger.md).

**Existing scripts are now organized:** start with the
[24 experiment homes and their run indexes](INDEX.md). The
[complete migration map](MIGRATION.md) names 237 canonical files, their old
compatibility paths and shared owners. This working-tree migration does not
change current immutable HPC releases or relocate historical receipts.
The [nested working layout](folder-layout.json) subsequently groups all remaining
flat script adapters under `scripts/compat/` and shared tests by subsystem.
Only navigation/infrastructure belongs at the `scripts/` and `tests/` roots.
Old flat script/test names are authenticated aliases inside new release archives,
not loose working files. See the [script](../scripts/README.md) and
[test](../tests/README.md) indexes before choosing an entry point.

## One scientific experiment, one documented home

Use `experiments/<family>/<experiment-id>/`. A family groups a scientific topic
such as `brd4-jq1`; an experiment describes a specific question and protocol such
as `fixed-lrboost`. Do not organize experiments by author, file type or date alone.
Use lowercase names separated by hyphens, stable after the first recorded run.

Related comparison arms, seeds and identical-protocol repeats live in the same
experiment. Give a materially changed objective, input/pocket, occupancy,
acceptance rule or learning-rate study a separate experiment directory with a
link to its parent. An engineering retry uses a new run/release ID within the
same experiment and explains what changed. Never overwrite an earlier attempt.

The following is an illustrative layout, not an existing or executed experiment:

```text
experiments/
  README.md
  _templates/                          # Copyable documentation templates
    README.md
    experiment.md
    run.md
  brd4-jq1/
    README.md                          # Family question and experiment index
    fixed-lrboost/
      README.md                        # Protocol, file map, commands, status
      scripts/                         # Drivers, analysis, collectors, renderers
        README.md
        run.py
      hpc/                             # This experiment's allocation launchers
        README.md
        native.lsf
      configs/
        README.md
        boltz2.json
      tests/                           # Experiment-only tests and small fixtures
        README.md
      templates/                       # This experiment's report assets
        README.md
      runs/
        README.md                      # Attempt index
        20261022T120000Z-cpu-01/
          README.md                    # Run record, timestamps and evidence
          receipts/                    # Small source/command/job receipts
            README.md
```

Create only the directories an experiment needs. **Every directory created
under `experiments/` must have a `README.md`.** Family and support-directory
READMEs may be short: explain their role, list children/files and point to the
experiment README. The experiment README must be a full protocol and file map;
each run README must be a full attempt record. Use the
[experiment and run templates](_templates/README.md).

## File ownership and supporting artifacts

| Kind of file | Required home |
| --- | --- |
| Experiment-specific Python, shell, LSF and other scripts | The owning experiment's `scripts/` or `hpc/` |
| Protocol/configuration, small inputs, fixtures and experiment-only tests | The owning experiment's `configs/`, `inputs/` or `tests/` |
| Analysis, plots/report source, HTML templates and experiment notes | The owning experiment's `scripts/`, `templates/` or `docs/` |
| Small per-attempt command, source, scheduler, validation and collection receipts | `runs/<run-id>/receipts/`, with a run README alongside |
| Reusable Mosaic functionality | `src/mosaic/`, with shared package tests in `tests/` |
| Proven shared command-line infrastructure | `scripts/<capability>/`, with its own README and named users |
| General teaching notebooks and model-interface examples | `examples/`; experiment-specific drivers belong with their experiment |
| Large outputs, arrays, structures, weights, caches and release archives | Approved external/HPC storage or ignored artifact storage, referenced from the run record |

Keep experiment-only files physically together. Shared scientific inputs and
reusable components may have a single separate owner; link their exact path and
version/hash from the experiment file map instead of copying them. Do not create
a vague `misc`, `utils` or `old` folder. Extract a shared helper only when named
consumers and its contract justify it; document dependencies and avoid
experiment-to-experiment imports. A generic renderer can be shared; a renderer
that assumes one campaign's arms, constants or proofs belongs to that campaign.

Heavy generated artifacts are an explicit storage exception, not a second code
home. Preserve their approved location, source/run identity and integrity
receipts. Add narrowly scoped ignore rules before generating new large outputs;
do not assume `experiments/**/outputs/` or every path under `results/` is ignored.
Never commit credentials or private environment contents to a run record.

Use documented entry points and resolve data/config/template paths explicitly.
Record the working directory, interpreter and full command. Do not depend on an
accidental current directory or leave unexplained `sys.path` changes.

## Required experiment README

An agent must be able to understand and reproduce the experiment from its README:

- Stable ID, scientific question/hypothesis, comparison arms and parent studies.
- Protocol and meaningful settings: model/data/context identities, seeds,
  objective, optimizer/acceptance policy and frozen/trainable boundaries.
- Execution requirements, resource limits, readiness/acceptance gates and
  authorized scope; distinguish preparation from permission to execute.
- A file map covering every owned file or coherent subtree, its role and entry
  point, plus every shared/legacy/external dependency with exact location.
- Exact preparation, validation, launch, collection and report commands, their
  working directory and execution location. Mark historical commands as such.
- Current status with an observation date, run index, evidence links, limitations,
  failures and next steps. Link the global ledger; do not duplicate its history.

## Agent logbook and run records

Use **[docs/experiment_ledger.md](../docs/experiment_ledger.md)** for chronological,
repository-wide events. Do not create a competing root logbook. The experiment
README is the current overview; `runs/<run-id>/README.md` describes one attempt;
the ledger says what happened across experiments and when, with links to both.

Use a unique, filesystem-safe run ID, preferably
`YYYYMMDDTHHMMSSZ-<stage>-<attempt>`, and record the actor/agent and parent run.
Existing campaign/run IDs remain valid and must not be renamed retrospectively.
Record separate child jobs/arms and their dependencies under a campaign attempt
when appropriate; do not call the campaign passed because one child passed.

1. **Before execution:** create the run record and append a `prepared` event.
   Include experiment folder, purpose, authorization/scope, planned command,
   resources, source/config/input identities and intended artifact locations.
2. **On submission and observed start:** append dated events with actual command,
   job IDs/dependencies and saved scheduler evidence. Keep planned, submitted,
   start and observation times separate. Use ISO 8601 UTC or an explicit offset.
3. **On completion, failure, cancellation or handoff:** append the observed state,
   end time if known, exit code, validation/scientific outcome, artifact and
   receipt links, limitations and next action. Update the run README and the
   experiment's dated current status/run index.

Use explicit states such as `prepared`, `submitted`, `running`, `completed`,
`failed`, `cancelled`, `partial`, `blocked` and `unknown`. Record scheduler state
separately from scientific acceptance and report/publication status. Submission
is not a pass; scheduler completion is not scientific acceptance; a missing or
aged-out job record is not proof of success or failure. Keep skipped checks,
unreached stages and unrun work visible. Never invent times, job IDs or results.

The run record must retain exact commands and working directory; commit plus
immutable source-manifest/archive identities (including uncommitted source);
config, input, model and parent checkpoint identities as applicable; environment,
resource/allocation details; and evidence/collection paths with integrity hashes.
Large artifacts can remain external, but a bare machine-specific URL/path
without provenance is insufficient. Log scientific/report/engineering work with
its actual scope. A documentation or migration-only event says **no experiment
executed**; a static check cannot become a numerical acceptance result.

Preserve earlier ledger entries and raw receipts. Append corrections and newer
observations with links to the superseded statement instead of rewriting failed
history. Agents working concurrently append distinct entries and retain each
other's events. The [run template](_templates/run.md) supplies the required fields.

## Current repository map and proposed destinations

The pre-policy tree mixed experiment code across `scripts/`, `scripts/hpc/`,
`scripts/templates/`, `examples/`, `examples/configs/`, `tests/` and `docs/`.
Historical receipt directories already separate many dated campaigns under
`scripts/hpc/receipts/`. The table below retains the original discovery map;
[INDEX.md](INDEX.md) and [MIGRATION.md](MIGRATION.md) describe the implemented
ownership and destinations, including shared-driver exceptions. No execution or
scientific pass follows from a file migration.

| Proposed experiment home | Existing documentation | Representative existing files |
| --- | --- | --- |
| `native-inference/protenix-tiny` | [Native Tiny setup](../docs/hpc_protenix_tiny.md) | `examples/hpc_protenix_tiny.py`, `scripts/hpc/protenix_tiny.lsf`, `stage_tiny_assets.py` |
| `graph-validation/stage0` | [Plan](../docs/stage_0_implementation_plan.md), [usage](../docs/stage_0_usage.md), [contract](../docs/stage_0_native_contract.md) | `examples/small_molecule_stage0.py`, `scripts/hpc/small_molecule_stage0.lsf`; packaging/verifiers need a shared-tool audit |
| `graph-validation/relaxed-gradients` | [Gradient tests](../docs/protenix_relaxed_gradient_tests.md) | `examples/protenix_relaxed_gradients.py`, matching LSF launcher |
| `graph-controls/carbon-chain-100` and `carbon-chain-500` | [100 updates](../docs/carbon_chain_100.md), [500 updates](../docs/carbon_chain_500.md) | `scripts/hpc/carbon_chain_*.lsf`, corresponding configs/fixtures |
| `graph-controls/known-ligand-1000` | [Known ligand](../docs/known_ligand_1000.md) | `prepare_known_ligand_fixture.py`, `assess_known_ligand_pilot.py`, `scripts/hpc/known_ligand_*.lsf` |
| Separate `brd4/contact-5000`, `uniform-5000`, `noisy-uniform-5000` and `canonical-diffuse-5000` experiments | [Contact](../docs/brd4_5000.md), [uniform](../docs/brd4_uniform_5000.md), [noisy](../docs/brd4_noisy_uniform_5000.md), [canonical](../docs/brd4_canonical_diffuse_5000.md) | `prepare_brd4_fixture.py`, `assess_uniform_control.py`, matching `scripts/hpc/brd4_*.lsf` |
| `brd4/best-primal-continuation` | [Continuation](../docs/brd4_best_primal_continuation.md) | `examples/continue_brd4.py`, `prepare_brd4_continuation.py`, continuation launchers |
| Separate `brd4/chemistry-5000`, `chemistry-ramp-10000` and `chemistry-compatibility` experiments | [Chemistry](../docs/brd4_chemistry_5000.md), [ramp](../docs/brd4_chemistry_ramp_10000.md), [compatibility](../docs/brd4_continuous_chemistry.md) | `examples/brd4_chemistry.py`, `*chemistry*` scripts/launchers; shared reference/oracle helpers need an ownership audit |
| `brd4/boltz-affinity-10000` and `brd4/boltz-gradient-diagnostic` | [Affinity](../docs/brd4_boltz_affinity_10000.md), [diagnostic](../docs/brd4_boltz_random_gradient_diagnostic.md) | `examples/brd4_boltz_affinity.py`, `assess_boltz_*`, `analyze_boltz_*`, `diagnose_boltz_random_gradient.py`, matching launchers |
| `brd4/three-model-25` | [25-atom comparison](../docs/brd4_25_atom_comparison.md) | `examples/brd4_25_atoms.py`, `campaign25_*`, `validate_brd4_*`; shared with JQ1, audit before moving |
| `brd4-jq1/backtracking-25` and `brd4-jq1/protenix-lr03` | [Pocket comparison](../docs/brd4_jq1_25_atom_comparison.md), [LR 0.3](../docs/brd4_jq1_25_protenix_lr03.md) | `derive_brd4_jq1_pocket.py`, `campaign25_pocket_ops.py`, `protenix_lr03_*`, corresponding configs |
| `brd4-jq1/fixed-steps` | [Fixed steps](../docs/brd4_jq1_25_fixed_steps.md), [report plan](../docs/brd4_jq1_25_fixed_report_plan.md) | `fixed25_ops.py`, `fixed25_split_*`, `fixed25_recover_*`, `jq1_fixed_report_*`, fixed-report scripts/templates |
| `brd4-jq1/fixed-lrboost` | [Higher rates](../docs/brd4_jq1_25_fixed_lrboost.md), [diagnostics](../docs/brd4_jq1_25_endpoint_diagnostics.md), [report plan](../docs/brd4_jq1_25_lrboost_report_plan.md) | `fixed25_lrboost_*`, `lrboost_endpoint_diagnostics.py`, `jq1_lrboost_*`, higher-rate report scripts/templates |
| `reporting/milestones` and `reporting/brd4-comparison` | [Milestones](../docs/mosaic_six_milestones_report.md), [combined report](../docs/brd4_combined_report.md) | `assemble_milestone_report.py`, `scripts/templates/mosaic_milestones.html`, combined-report launcher |

Bare script names in the table are under `scripts/` unless qualified. Source
already declares an LR-reset protocol and a new dated receipt directory; assign
its future work a separate `brd4-jq1/fixed-lrreset` home after reading its actual
scope/receipts. This inventory establishes no execution outcome for it.
Report-only corrections can be child run records of their owning experiment;
cross-experiment reports have their own `reporting/` experiment and source links.

Potential shared infrastructure includes source packaging/verification, quota
preflight, authenticated evidence collection and generic trajectory display.
Audit imports and constants before placing it in documented capability folders
such as `scripts/releases/`, `scripts/hpc/` or `scripts/reporting/`. Likewise,
shared scientific algorithms remain in `src/mosaic/`, not copied script trees.

## Migrate safely, one coherent experiment at a time

The initial migration follows this procedure and keeps documented legacy
adapters and declarative copies. New work must use the canonical owners in the
index. Apply the same procedure to future additions or adapter retirement.

1. Read current docs/ledger, inspect working-tree changes and check for live jobs,
   callbacks, report servers and other agents using the files. Inventory all
   drivers, imports, configs, fixtures, tests, templates and packaging allowlists.
2. Create the family/experiment READMEs and an old-to-new path table. Assign each
   file an owner; name shared dependencies and preserved historical locations.
3. Move only mutable working sources in a coherent change. Update imports,
   `__file__`-relative paths, CLI references, launchers, test discovery, report
   assets, ignore rules and release include/authentication rules. New experiment
   paths must be covered by source manifests; current packagers/verifiers may
   enumerate only `src`, `tests`, `examples` and `scripts`.
4. Preserve sealed archives, manifests, historical receipt directories, old
   commands and active-job source bindings. Link them from the new README.
   A documented forwarding wrapper may retain an old mutable entry point;
   do not duplicate implementations or replay a receipt script.
5. Check links, paths, syntax and package contents locally as permitted. Run
   necessary imports/CLI/test/native checks only in their authorized environment.
   A move is not a scientific change or permission to launch a new experiment.
6. Update the inventory, run index and global ledger with moved/preserved paths,
   checks actually performed and remaining limitations. A migration with required
   execution checks pending must stay marked pending, not fully validated.

A compatibility exception must name the exact legacy path, reason, owner/home
and next migration step in the experiment README and ledger. It cannot silently
become permission for new flat scripts. Do not move live dependencies until
their consumers are accounted for; keep prior failures and all existing gates.
