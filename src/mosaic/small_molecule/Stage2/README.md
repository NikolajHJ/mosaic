# Stage 2: add chemistry after the model update

`pipeline.py` uses the Stage 1 design loop, including the model selected in
the general YAML. The objective stays the same: contacts for Protenix-Tiny
or fixed-pose affinity for Boltz2. `operators.py` comes from the original
`stage2/` folder on `codex/mosaic-experiments`. It acts after the usual
gradient update and probability projection. It does not add a loss term.

The default, `mode: valence`, adjusts bond probabilities only when an atom's
expected bond order exceeds what its element probabilities allow.
`mode: none` runs the matched relaxed control. `connected` tries to bridge
fragments, and `sparse` adds weak-bond shrinkage. Use those options as
separate comparisons. `hardening.py` optionally makes the model see more
discrete atom and bond choices as the run progresses; the optimizer still
updates probabilities. Compare hardening against a relaxed run starting
from the same checkpoint.

Use a general YAML from the [BRD4 baseline](../../../../experiments/brd4/stage1-baseline/README.md)
and choose a new output folder. On an allocated DTU HPC GPU, run from the
repository root:

```bash
PYTHONPATH=src python -m mosaic.small_molecule.generate_molecule \
  --general-config experiments/brd4/stage1-baseline/general.yaml \
  --stage-config configs/stage2.yaml
```

For Boltz2, use the Boltz2 general YAML and prepared context described in
the [Boltz2 guide](../../../../experiments/brd4/stage1-baseline/boltz2.md).
The Stage 2 YAML contains only stage settings, including the chemistry
mode. Leave `start_checkpoint` empty for a fresh start with the same
initialization as Stage 1. To continue, point it at a `checkpoint.npz` from
this runner, relative to the stage YAML, with its manifest and target
record alongside it. Keep the target and model context matched.

Stage 2 saves the same trace, raw molecule checks and probability
trajectory as Stage 1. Each updated state's trace also records what the
chemistry operators changed. The original studies and their saved
evidence remain on the donor branch. This integration has not been run
with a GPU model.
