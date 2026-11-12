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

Optional [Stage 3 DeFoG guidance](../../../../stage3/README.md) adds a frozen,
one-query teacher loss before the update. Install it with
`python -m pip install -e './stage3[jax]'` in the runner's environment and add
the following block to a Stage 2 YAML:

```yaml
defog:
  source: /path/to/DeFoG
  checkpoint: /path/to/guacamol.ckpt
  python: /path/to/defog-env/bin/python
  weight: 0.0625
  time: 0.8
```

The source and checkpoint are external assets; the separate interpreter hosts
the CPU teacher. Paths are relative to the Stage 2 YAML. Weight and teacher time
can also use linear/cosine schedule dictionaries; the
[complete YAML](../../../../stage3/examples/stage2-defog.yaml) shows examples.
DeFoG composes with the hardening hook and leaves the selected chemistry
operators in place. `none`, `valence`, `connected`, `sparse_only`, and `sparse`
remain separate projection choices. None guarantees a connected, sanitized
argmax molecule; see the [benchmark evidence](../../../../stage3/BENCHMARKS.md).

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
Choose the chemistry mode in the Stage 2 YAML. For a fresh start, set
`n_atoms` and `initialization.random` in the general YAML. To continue, set
`initialization.checkpoint` to a `checkpoint.npz` from this runner, relative
to the general YAML. Keep its manifest and target record alongside it and
use the same atom count, target, and model context.

Stage 2 saves the same trace, raw molecule checks and probability
trajectory as Stage 1. Each updated state's trace also records what the
chemistry operators changed. The original studies and their saved
evidence remain on the donor branch. This integration has not been run
with a GPU model.
