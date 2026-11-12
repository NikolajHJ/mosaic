
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
