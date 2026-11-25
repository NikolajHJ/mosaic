
### 2026-11-05 — Protenix notebook reading companion

On NewPipeline_draft, added examples/protenij_script.py as a commented,
sequential adaptation of examples/protenij.py for code study, at the user's
explicit request to keep this teaching example under examples/. Matched the
branch's sampler=protenix.design_sampler(20) call. Replaced Marimo presentation
with ordinary Python comments, prints, plotting and PDB file output, and put
prediction before displays that depend on it. No experiment, model execution,
submission or run-test was performed. Existing configs/ work was preserved.

### 2026-11-05 — Draft configurable small-molecule runner

Updated the user's generate_molecule.py on NewPipeline_draft after inspecting
Stage0/Stage1/Stage2/Stage3/VAE and the empty configs/*.yaml files. The draft
accepts general and stage YAML paths, selects a StageX/VAE pipeline, and outlines
model/target preparation, initialization, loss construction, optimization,
evaluation and result saving. Missing implementations are marked TODO. Existing
YAML files and stage folders were preserved. This is a design/study draft:
no experiment, model import, optimization or scheduler submission ran. Python
syntax was checked by AST parsing only; runtime behavior remains untested.

The shared checkout switched branches during editing. The draft was moved
into the separate mosaic-NewPipeline_draft worktree, on NewPipeline_draft;
only this task's additions were removed from the other working tree.

### 2026-11-06 — General YAML for the draft molecule runner

Filled `configs/general_config.yaml` on `NewPipeline_draft` in the original
Mosaic checkout. It points to the existing `IL7RA.cif` protein file (chain A).
Pocket residues and starting molecule remain explicit placeholders. This is
configuration only; no experiment was run or submitted.

### 2026-11-06 — Ported the small-molecule Stage 0 workflow

Copied the graph, feature, model, loss, and update code from
codex/mosaic-experiments (2eeeba20261633cd297ff79b15b527465ead890e) into NewPipeline_draft.
The YAML entry point now selects Stage0/pipeline.py for fixed-size Protenix-Tiny
optimization. Protein/pocket selection, prepared or SMILES initialization,
contact losses, projected updates, and trajectory output follow the existing
single-run example. The single-recycle trunk calculation is kept with the
molecular adapter to avoid a dependency on the newer protein sampler. Original
experiment scripts, releases, and results were not moved. Stage 1–3 and VAE
are future extensions. This was a code port; no experiment was run or submitted.

### 2026-11-06 - Plain-language Stage 0 file descriptions

Rewrote the opening Python descriptions and the README file guide on
NewPipeline_draft. Checked that the Python code was unchanged apart from
module docstrings. Documentation only; no experiment ran.

### 2026-11-06 - Simplified Stage 0 input checks

Removed routine input checks from the NewPipeline_draft ligand code.
Kept the checks for model/input compatibility, atom and token mapping,
missing pocket residues, and numerical failures. ValueError statements
went from 93 to 40. No experiment was run or submitted.

### 2026-11-06 - Expanded the Stage 0 file descriptions

Expanded the opening descriptions and README file guide on NewPipeline_draft.
They now explain each file's inputs, outputs, and place in the workflow.
Compared Python syntax trees to confirm that only module docstrings changed.
Documentation only; no experiment was run or submitted.

### 2026-11-06 - Implemented the Stage 1 baseline

Added Stage1/pipeline.py, diagnostics.py and configs/stage1.yaml on
NewPipeline_draft. The canonical short-run configuration and commands are
in experiments/brd4/stage1-baseline. The shared 3MXF fixture is copied
byte-for-byte from codex/mosaic-experiments (2eeeba20261633cd297ff79b15b527465ead890e).
Stage 1 uses whole-target contact at 8 angstroms, a fixed 25-slot noisy
carbon-heavy initialization, and unconstrained projected updates. The
geometry-off path has no valid ligand frames; the native PAE trial has
not been run. RDKit and valence checks report raw results without repair.
No model evaluation, numerical experiment or scheduler submission ran.

### 2026-11-06 - Added the YAML-selectable Boltz2 backend

Stage 0 and Stage 1 on NewPipeline_draft now select Protenix-Tiny or Boltz2
from the general YAML. Ported the fixed-pose affinity worker, gradient bridge
and preparation code from codex/mosaic-experiments (2eeeba20261633cd297ff79b15b527465ead890e).
The affinity path uses CPU JAX and a pinned native CUDA worker. It is separate
from protocol v12's Protenix contact baseline. Added a BRD4 config template
and preparation guide; no model assets were downloaded and no experiment
was run or submitted.

### 2026-11-09 - Integrated the existing Stage 2 operators

Ported operators and optional hardening from stage2/ at
codex/mosaic-experiments (2eeeba20261633cd297ff79b15b527465ead890e) to src/mosaic/small_molecule/Stage2 on
NewPipeline_draft. The YAML chooses the chemistry rules; the general YAML
still selects Protenix-Tiny contacts or Boltz2 affinity. Stage 2 reuses the
Stage 1 loop and logs the operator results after each update. Valence only
is the default. A matching saved checkpoint can start paired hardening
and relaxed runs. Historical launchers, sweeps and evidence remain on the
source branch. No numerical experiment ran or was submitted.

### 2026-11-11 - Organized the shared small-molecule code

Moved the reusable graph, pocket, loss, optimization and model code into
`src/mosaic/small_molecule/core/` on NewPipeline_draft. Updated the stage
imports, Boltz2 worker launch and package guides. Checked Python syntax,
the YAML entry points for Stages 0–2, and Boltz2's worker module resolution
with a stand-in for the model. No experiment ran or was submitted.

### 2026-11-11 - Moved ligand initialization into the general YAML

Stages 0–2 now read the atom count and starting molecule settings from
the general YAML on NewPipeline_draft. Moved the existing Stage 1 and 2
random-start values and optional checkpoint path without changing their
values or the update rules. The launcher still accepts legacy YAML pairs.
Checked the config pairs, stage dispatch, and starting settings using
model stubs. No experiment ran or was submitted.

### 2026-11-12 - Implemented the LR60 DeFoG Stage 3 ablation campaign

Question: which frozen one-query teacher settings improve the best valid hard
Protenix molecule, and how do time, strength, projection and cooling affect it?
Implemented on `codex/defog-stage3-campaign` in the
[existing learned-prior home](../experiments/learned-prior/protenix-defog-composition-30000/README.md).
The resolved design contains 155 settings, 3 matched start types and 5 seeds:
2,325 planned runs of 500 updates at atom/bond LR60, native coefficient 1.
A separate reference wave has 30 runs in 15 seed/start bundles. This preparation
does not modify earlier experiments or submit GPU jobs.

Primary selection is the highest frozen native hard contact score among raw
sanitized connected argmax molecules, with initial-only successes explicit.
Saved outputs include the top 20 distinct valid molecules, all ordered candidate
graphs, complete probability trajectories, selected discovery gradients/targets,
chemical descriptors, bounded endpoint repair witnesses, and streaming tidy CSV
tables. Optional descriptor failures retain candidates with missing metadata.

Validation: `python -m pytest experiments/learned-prior/protenix-defog-composition-30000/tests -q`
passed 134 CPU tests. An independently retained three-update mock pair completed
and produced its plotting tables and molecule archives; runtime versions and
code identity are in
[the local checks](../results/implementation-checks/stage3-20261112a/checks.json).
No native Protenix/DeFoG optimization, GPU performance or numerical qualification
is claimed from these software fixtures. Native outcomes remain unmeasured.

### 2026-11-12 - Reduced the initial DeFoG panel to three seeds

At the user's request, the default LR60/500-update campaign now uses seeds
17, 23 and 42. All 155 settings and three start types remain: 1,395 planned
runs, 396 GPU bundles and 697,500 updates. The reference wave is 18 runs in
nine bundles. The original five-seed config and generated plans are preserved;
seeds 71 and 101 can be added later with identical settings and existing run IDs.
The relevant design, reference execution/collection and dispatch checks passed
38 CPU tests. [New plan checks](../results/implementation-checks/stage3-three-seeds-20261112a/plan-checks.json)
record model-free expansion. No native optimization or GPU submission occurred.

### 2026-11-13 - Submitted the three-seed DeFoG Stage 3 campaign

All 396 GPU bundles were accepted, covering 1,395 unique 500-update runs at LR60
with native coefficient 1. Placement: 132 V100 and 132 A100 bundles, plus 44 each
on L40S, A10 and A40. Shared-control seed/start blocks remain on one GPU family.
The source is read-only commit 779bc35; scientific settings are unchanged.
Lossless record compression was added after the initial quota check, and 46
focused CPU checks passed. Following the user's storage cleanup, the fresh 20 GiB
own-campaign capacity check passed. Active outputs are on work3; the unused home
preparation has no submitted jobs.

At 00:48 CET, receipts reconciled every job and LSF reported 396 pending, with no
native optimization result yet. See the [submission note](../experiments/learned-prior/protenix-defog-composition-30000/run-20261113.md)
and [retained receipts](../results/hpc/stage3-protenix-defog-lr60-3seeds-20261112a/submissions.json).

### 2026-11-13 - Diagnosed archive line endings and submitted LF retry

At 14:49 CET, no Stage 3 optimization run had begun: 222 bundles had failed
bootstrap with `Experimental 3MXF bytes differ`, while 174 remained pending.
Git for Windows converted the archived PDB from LF to CRLF under
`core.autocrlf=true`; the Git blob itself matched the pinned LF hash. The
corrected `git -c core.autocrlf=false archive` reproduces the pinned PDB exactly,
without coordinate, objective, source-logic or authentication changes. Its
archived and extracted PDB hashes and actual pocket parsing passed before queueing.

The 174 authenticated pending/no-progress jobs were retired with the scheduler's
pending-only filter. No old jobs remained live and all old optimization counts
were zero; source and failure evidence remain intact. The fresh `lf-retry1`
attempt retains all 1,395 settings/seed/start runs and the same five-family GPU
placement. Nine reference bundles were accepted; an active controller will
release the remaining 387 after a teacher-on reference records at least 25 normal
updates. At 15:10 CET, all nine were pending and the controller was alive. Three
isolated controller checks passed without model or scheduler calls. No
native completion or chemistry result is claimed. The fresh 10 GiB own-output
allowance passed the capacity check. See the [recovery note](../experiments/learned-prior/protenix-defog-composition-30000/run-20261113.md).

### 2026-11-15 - Most LR60 DeFoG arms complete; stronger guidance is promising

The LF retry released all 396 bundles automatically on 13 November. At 21:21 CET
on 15 November, 1,294/1,395 arms had completed 500 updates, 96 were unstarted and
five were partial after valence-projection KKT tolerance failures. Old packaging
failures remain separate, with zero optimization updates. Frozen checks passed
for the retained completed/partial outputs.

Constant weight 0.5, time 0.8, connected projection and no cooling found new valid
connected molecules in all nine seed/start arms, with four valid endpoints.
On eight full-budget matched pairs (the fragmented seed 17 control stopped at 226),
mean best-valid hard contact was 0.4061 versus 0.2848 for projection alone, winning
8/8, with 3/8 valid endpoints versus 0/8. Weight 0.0625 gave 0.3226 and 6/8 wins with
zero valid endpoints on the same pairs. This is exploratory selection among 155
settings and three seeds, not an independent confirmation. Most configurations
still fragment late; only 11 of 1,294 completed endpoints were valid/connected.
See the [dated note and evidence](../experiments/learned-prior/protenix-defog-composition-30000/run-20261113.md#15-november-progress-and-preliminary-results).

### 2026-11-16 - Remaining Stage 3 bundles moved off A100

The user's requested queue change retired nine authenticated pending/no-output
A100 submissions and replaced the same 33 runs on V100 (three bundles), L40S
(three), A10 (two) and A40 (one). All were accepted; at 09:26 CET the three V100
jobs had started. The existing A100 bundle finished normally, leaving no active
campaign A100 jobs. Scientific source/settings and budgets are unchanged.
Original and replacement receipts are retained. Migrated fragmented seed-23
arms now differ in GPU family from some matched controls; record this in final
comparisons. The campaign had 1,357 complete and five previous partial runs;
this scheduling change adds no molecular-quality conclusion. See the
[migration note](../experiments/learned-prior/protenix-defog-composition-30000/run-20261113.md#16-november-remaining-bundles-moved-off-a100).

At 16:54 CET the final four still-pending bundles (13 runs) were moved to V100
at the user's request. History confirmed zero RUN seconds after pending-only
retirement; replacements 30082261–30082264 were accepted and pending.
Scientific settings and budgets are unchanged, with prior receipts retained.
See the [V100 migration evidence](../experiments/learned-prior/protenix-defog-composition-30000/run-20261113.md#16-november-final-13-runs-moved-to-v100).

### 2026-11-17 - Stage 3 campaign finished

The last 13 runs completed 500 updates on V100. Of 1,395 planned runs, 1,390
completed and five previously reported runs stopped early on projection
tolerances. All 13 final runs found a new valid connected molecule, though none
ended valid. Among all 1,390 heterogeneous completed runs, 1,143 found a valid
connected molecule, 1,027 found a new valid identity, and 11 ended valid and
connected. Pooled counts include controls and are not a DeFoG effect estimate.
See the [completion note](../experiments/learned-prior/protenix-defog-composition-30000/run-20261113.md#17-november-campaign-completion).

The five partial runs and matched weight-0.5 treatment were submitted as six
separate V100 jobs with a projection solver recovery and retained KKT warnings.
All were pending at submission; numerical and chemistry outcomes remain open.
See the [retry note](../experiments/learned-prior/protenix-defog-composition-30000/run-20261113.md#17-november-projection-recovery-reruns).

### 2026-11-18 - Projection recovery reruns completed

All six reruns completed 500 updates with frozen checks passing. Five found new
valid connected molecules; none ended valid/connected. For fragmented seed 17,
weight 0.5 scored 0.33080 against the projection-only control's 0.28018. The
earlier weight-0.5 attempt scored 0.46203 and ended valid; trajectories diverged
despite the same initialization, software versions and GPU model, for reasons
not yet established. No solver retries or KKT warnings were observed in the six
new operator logs, so the real original failure inputs have not been reproduced.
See the [rerun results](../experiments/learned-prior/protenix-defog-composition-30000/run-20261113.md#18-november-projection-recovery-results).

### 2026-11-21 - DeFoG and structural repair on five fragmented starts

All five matched 500-update BRD4 pairs completed. Both methods found valid connected molecules on every seed. DeFoG at weight 0.5 with connected projection had higher best valid hard contact on 3/5 seeds (mean 0.400 versus 0.366 for repair at weight 1 without projection). Repair ended valid on two seeds and DeFoG on one. The first 20 distinct valid candidates per trajectory gave means of 0.345 and 0.359 respectively. [Per-seed results and extraction method](../experiments/learned-prior/protenix-defog-composition-30000/repair-pair-20261118.md#21-november--completed-five-seed-comparison).

### 2026-11-21 - Strong repair and held-out Protenix scoring

Three 500-update runs using repair weight 10, no Stage 2 projection and LR 60 finished valid and connected (seeds 17, 23, 42). Mean best valid contact was 0.288, versus 0.352 for repair 1 and 0.376 for projected DeFoG on the same starts. Rescoring all 80 preselected molecules on five fresh common keys gave exact agreement with the same-device original-key replays. [Run results, raw scores and updated report](../experiments/learned-prior/protenix-defog-composition-30000/repair-pair-20261118.md#21-november--stronger-unprojected-repair-and-fresh-protenix-keys).

### 2026-11-21 - Stage 3 standalone DeFoG closeout

With DeFoG weight 0.5, teacher time 0.8, LR 60 and 500 updates on fragmented seeds 17, 23 and 42, disabling Stage 2 projection yielded 0/1,503 valid connected observations and endpoints of six, two and seven components. The matched connected-projection runs found valid molecules on 3/3 starts and recorded 525/1,503 qualifying observations. The new A100 runs passed frozen checks and all objective budgets. [Closeout results and PDF](../experiments/learned-prior/protenix-defog-composition-30000/repair-pair-20261118.md#21-november--defog-05-without-projection).

### 2026-11-25 - Sampled DeFoG endpoints versus the one-query target

In the saved 500-update Boltz/DBMol trajectory, all 99 successive refreshes changed the slot-aligned target, including 81 with unchanged input argmax. CPU job 30363657 held five input graphs and model time fixed and repeated both methods eight times per graph. All 140 within-input pairs of full sampled endpoints differed (median 11 labels). One-query probabilities repeated exactly. Of 40 unfiltered samples, 39 were valid and connected, compared with three of five one-query argmax readouts. This measures repeatability and output chemistry on a small control panel. [Protocol, measurements and scripts](../experiments/learned-prior/protenix-defog-composition-30000/target-repeatability-20261125.md) · [Updated report](stage3_defog_report_draft.pdf)

A saved-output calculation also compared categorical draws on both sides. On each of the five fixed inputs, the measured mean label disagreement between full continuations exceeded that expected between two draws from the one-query probabilities. For hexane the means were 10.89 and 1.43 labels per pair. [Calculation and results by control](../experiments/learned-prior/protenix-defog-composition-30000/target-repeatability-20261125.md#does-drawing-a-categorical-answer-explain-all-of-the-variation)

### 2026-11-25 - One-query repair direction and teacher controls

CPU job 30364221 completed 36 queries on 12 archived valid BRD4 graphs and their two single-bond corruptions. The best valid one-edit repair ranked in the top ten for 12/12 valence defects and 1/12 deleted bridges. A quarter-step toward the target repaired zero hard graphs. A three-quarter-step repaired 2/12 valence defects and 0/12 deleted bridges, and kept 8/12 clean controls valid. The checkpoint remained frozen. Three GPU jobs carry the six 500-update frozen-initial and norm-matched shuffled controls. [Experiment note and results](../experiments/learned-prior/protenix-defog-composition-30000/teacher-mechanism-20261125.md)
