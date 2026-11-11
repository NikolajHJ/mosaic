# Small-molecule design

Stage 0 starts with a ligand and tries to improve its predicted contacts with
a protein pocket. The number of ligand atoms stays fixed, but the optimizer
can change the type of each atom and the bonds between them.

Start with `generate_molecule.py`, then read `Stage0/pipeline.py` to follow the
whole run: prepare the inputs, score the molecule, update it, and save results.

Each stage has its own folder. The `core/` folder holds code that the stages
share for preparing inputs, calling the models, and updating the molecule.

| File | What it does |
| --- | --- |
| `generate_molecule.py` | Start here. See how the two YAML files are read and how the requested stage is selected. |
| `Stage0/pipeline.py` | Read this next for the full run: prepare the pocket and ligand, score each candidate, update it, and save the trajectory. |
| `core/pocket.py` | Reads the protein structure and identifies pocket residues from a reference ligand or a residue list. |
| `core/boltz2/` | Prepares the fixed protein–ligand pose and runs the Boltz2 affinity worker. |
| `core/experiment.py` | Extracts the target sequence and pocket from the structure file. Also works out the loss weights at each update. |
| `core/state.py` | Defines the atom and bond probability arrays. Each possible bond is stored once, including pairs that currently have no bond. |
| `core/features.py` | Prepares the model input once, then fills in the current ligand probabilities at each update. |
| `core/contracts.py` | Defines what each step passes to the next, including which model positions are ligand atoms and which are protein residues. |
| `core/protenix.py` | Runs the fixed Protenix-Tiny model and returns distance scores that the loss can use. |
| `core/native_hook.py` | Keeps information from the starting ligand shape out of the design as atom and bond choices change. |
| `core/losses.py` | Turns distance predictions into contact or pocket coverage scores. |
| `core/optimizers.py` | Uses gradients and the learning rate to change the candidate while keeping its probabilities valid. |

In `configs/general_config.yaml`, set the protein file and chain, and fill in
the pocket residues or give a `reference_ligand` (chain, residue name and number).
Set `n_atoms` to the number of ligand atom slots. Put the starting molecule
settings under `initialization` in the same general YAML.

- **Stage 0, Protenix-Tiny** uses `initialization.smiles` for a clean molecule
  or `initialization.input` for a folder with a prepared graph and its
  `initialization.json`. Set `n_atoms` to the number of heavy atoms.
- **Stage 0, Boltz2** starts from the prepared chain or uses a graph from
  `initialization.input`. Match `n_atoms` to the prepared context.
- **Stages 1 and 2** use `initialization.random` for a fresh start. Set its
  carbon probability, expected degree, and atom and bond noise. Set
  `initialization.checkpoint` to continue from a compatible saved graph.

Paths in the general YAML are relative to that file.

On an allocated DTU HPC GPU, use an environment with the audited Protenij
revision recorded in `core/native_hook.py` and the staged Protenix-Tiny assets.
Set `model_cache` to their directory if it differs from the Mosaic cache default.

```bash
PYTHONPATH=src python -m mosaic.small_molecule.generate_molecule \
  --general-config configs/general_config.yaml \
  --stage-config configs/stage0.yaml
```

The stage YAML lists losses and their weights. `contact` and `pocket_coverage`
are available. An optional `weight_schedule` takes one-based `update` keyframes
with `weights_pct` entries for every configured loss, summing to 100.

The output folder contains copies of both YAMLs, the selected target, the loss
trace, and atom and bond probabilities for the initial state and every update.
The stage uses the target sequence without a searched MSA or a coordinate
template. Contact scores do not measure binding affinity.

For the unconstrained **Stage 1** baseline, see the [BRD4 run guide](../../../experiments/brd4/stage1-baseline/README.md). Stage 1 scores contacts over the whole protein domain and measures the raw discrete molecule without changing it. See the [Stage 2 guide](Stage2/README.md) to add chemistry after each update. Stage 3 and VAE remain extension points.

To use **Boltz2**, set `model: boltz2` in the general YAML and supply a prepared
context, affinity checkpoint, and GPU worker interpreter. Both Stage 0 and
Stage 1 then use fixed-pose affinity instead of the Protenix-Tiny contact
score. See [the Boltz2 guide](../../../experiments/brd4/stage1-baseline/boltz2.md)
for the example config and preparation steps.
