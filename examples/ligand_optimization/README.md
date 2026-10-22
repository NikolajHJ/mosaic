# Ligand optimization

This workflow changes a ligand's atom and bond probabilities to improve a chosen score for a protein pocket. You choose the starting molecule, protein and pocket, model, loss or losses, learning rate, and number of updates. The optimizer saves a record of each step.

There are three steps:

1. Prepare the starting molecule with `prepare.py`.
2. Describe the experiment in a copy of `config.example.json`.
3. Submit `run.py` as a GPU job on DTU HPC.

You normally leave `prepare.py` and `run.py` as they are. Make a separate config file and results folder for each experiment.

The other relevant files are `examples\ligand_optimiztion\batch.py` and `src\mosiac\small_molecule\experiment.py`

## 1. Choose a starting molecule

`prepare.py` reads a molecule from either a SMILES string or an SDF file and makes the atom and bond probabilities used to start optimization. It does not create or optimize 3D coordinates.

Run the command from the Mosaic repository folder, using the project's Python environment. For example, to start from ethanol with some atom and bond noise:

```bash
PYTHONPATH=src python examples/ligand_optimization/prepare.py \
  --smiles 'CCO' --atom-noise 0.2 --bond-noise 0.4 --seed 17 \
  --output examples/ligand_optimization/initialization/ethanol-noisy
```

Noise controls how much the initial atom or bond assignments are mixed with random alternatives. Set `--atom-noise` or `--bond-noise` to `0` to keep that part of the molecule as given. A value of `1` makes it fully random; values in between mix the original and random assignments. Both default to `0`, so this makes a clean starting molecule:

```bash
PYTHONPATH=src python examples/ligand_optimization/prepare.py \
  --smiles 'CCO' --output examples/ligand_optimization/initialization/ethanol-clean
```

To use an SDF instead, replace `--smiles 'CCO'` with `--sdf path/to/ligand.sdf`. The `--seed` value makes the random noise reproducible. Each run needs its own new output folder; existing folders are not overwritten. The folder contains the starting atom/bond probabilities (`initial_graph.npz`) and a small record describing the molecule and preparation (`initialization.json`).

Molecules must be neutral, contain at least two non-hydrogen atoms, and use elements and bond types supported by Mosaic.

## 2. Describe the experiment

Copy the example settings file and edit the copy:

```bash
cp examples/ligand_optimization/config.example.json examples/ligand_optimization/brd4.json
```

In `brd4.json`, choose:

| Setting | What it controls |
| --- | --- |
| Model | `protenix-tiny` or `boltz2`. Model files must already be available on HPC. |
| Protein structure and chain | The PDB or mmCIF file and protein chain that define the target. |
| Pocket | Either a reference ligand in the structure, or a list of pocket residues. |
| Initialization | The folder made in step 1. For the noisy ethanol example above, use `"initialization": "./initialization/ethanol-noisy"`. |
| Updates | How many gradient updates to perform. The run does exactly this many. |
| Learning rate | The size of each update to the ligand probabilities. |
| Losses and their weights | What the optimizer tries to improve and how much each objective contributes. |

When you select a pocket using a reference ligand, the ligand's position is used to find nearby protein residues. The reference ligand itself is left out of the model input. Alternatively, list residue numbers explicitly. Check that the chain and residue numbers match the structure file.

The loss weight schedule lets you change priorities during a run. For example, you can start with 100% contact loss and gradually increase the contribution from a second loss. Updates in the schedule start at 1, and the percentages at each listed point must add to 100. Values between listed points are interpolated smoothly.

For Protenix-Tiny, the structure file provides the protein sequence and helps identify the pocket residues. Its 3D coordinates are not passed to this ligand-graph optimization. The supported losses are `contact` and `pocket_coverage`.

## 3. Submit the run on DTU HPC

The optimization needs an allocated GPU. Running `python run.py` directly on your computer or on an HPC login node will not start a run. From the Mosaic folder on HPC, create a directory for the job logs, then submit the job. Update the project path, config name and run name as needed:

```bash
mkdir -p results/ligand_runs
PROJECT=/zhome/10/4/206201/Bachelor/mosaic
bsub -q gpua10 -n 4 -R 'span[hosts=1]' -R 'rusage[mem=8GB]' \
  -gpu 'num=1:mode=exclusive_process' -W 02:00 \
  -J ligand-opt -oo results/ligand_runs/my-run.stdout.%J.log \
  -eo results/ligand_runs/my-run.stderr.%J.log \
  bash -lc "cd '$PROJECT' && source .venv/bin/activate && \
  PYTHONPATH=src python examples/ligand_optimization/run.py \
  --config examples/ligand_optimization/brd4.json \
  --output results/ligand_runs/my-run"
```

The example requests one A10 GPU. Use a queue and resources available to your HPC account; queue availability can change. Make sure the Python environment and model files are available from the job. Each run needs a new output folder, because existing results are not overwritten.

### What `batch.py` does

Use `batch.py` when you want to run several experiments without submitting each one by hand. Each config file describes one experiment: its molecule initialization, protein pocket, model, losses, and optimization settings. The script submits one separate HPC job for each config. Use `--repeat Y` to submit the same config Y times; repeats keep the same molecule and settings but use different random seeds. For example, two configs with `--repeat 3` create six jobs.

Run the script from the Mosaic repository folder on HPC, after the `.venv` environment has been set up. Give it an output folder that does not already contain files, then list the config files:

```bash
python examples/ligand_optimization/batch.py --repeat 3 \
  --output-root results/ligand_batch/october \
  examples/ligand_optimization/brd4.json \
  examples/ligand_optimization/another-target.json
```

The script only submits jobs; the model runs later on allocated HPC GPUs, not on the login node. By default, each job requests one GPU from `gpua10`, 4 CPU cores, 8 GB memory, and up to 2 hours. Set `--queue`, `--cores`, `--memory`, or `--walltime` to change these requests. Check which queues and resources are currently available to your account.

The output folder contains one results folder per run, plus `batch_status.md` and a `configs/` folder. The copied configs show the exact settings and random seed used for each job. The tracker lists each run as preparing, submitted, running, finished, or failed; its heading counts finished runs and changes to `DONE` once every job has reached a final state. `DONE` means all jobs stopped, so check the success and failure counts to see whether they all completed successfully. The individual job logs are saved beside the tracker.

## What the results contain

The output folder contains:

- `trace.json`: the total loss, each individual loss, and their weights at the starting point and after every update. This is the main file for following how the optimization behaved.
- `trajectory.npz`: atom and bond probabilities at each step. These are not 3D ligand structures.
- `config.json`, `target.json`, and `identity.json`: a copy of the settings, the selected pocket, and details identifying the model and inputs.
- `manifest.json`: whether the run completed and how many updates it performed.
- `context/` for Protenix-Tiny: the prepared model input. If a run fails, it writes `failure.json`; Boltz2 also writes `boltz_worker.log`.

## Current model support

Protenix-Tiny can combine the `contact` and `pocket_coverage` losses in one run. The scheduled percentages set their relative coefficients; because the losses may have different numerical scales, a 50/50 schedule does not necessarily mean each accounts for half of the final numeric loss.

Boltz2 currently supports only its differentiable `affinity` loss. It requires a separately prepared, pose-checked context containing either 6 or 25 ligand slots, plus paths to that context, its checkpoint, and the worker's Python environment in the config. The optimizer runs on CPU while the model worker uses the allocated GPU. Use the Boltz2 preparation instructions before choosing this model.
