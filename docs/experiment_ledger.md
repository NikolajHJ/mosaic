
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
