
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
