"""Start here to run a small-molecule design from two YAML files.

The general YAML describes the run: protein, pocket, starting molecule,
model, number of updates, learning rate, and results folder. The stage
YAML names the workflow and provides its own settings, such as the losses
and their weights. The command-line arguments are paths to those files.

This file reads the settings and calls the chosen stage's run() function.
The stage does the design work; this file only decides what to start and
where the results should go.
"""

import argparse
import os
from importlib import import_module
from pathlib import Path


STAGES = {
    "stage0": "Stage0",
    "stage1": "Stage1",
    "stage2": "Stage2",
    "stage3": "Stage3",
    "vae": "VAE",
}


def read_config(path):
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--general-config", type=Path, required=True)
    parser.add_argument("--stage-config", type=Path, required=True)
    args = parser.parse_args()

    general_path = args.general_config.resolve()
    stage_path = args.stage_config.resolve()
    general = read_config(general_path)
    stage_config = read_config(stage_path)
    stage = stage_config["stage"].lower()
    model = general["model"]
    if model not in ("protenix-tiny", "boltz2"):
        raise ValueError("Choose protenix-tiny or boltz2 in the general YAML.")
    # Boltz2 keeps its GPU memory in a separate PyTorch worker.
    if model == "boltz2":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    # Future stages supply their own pipeline.py with the same run() entry point.
    workflow = import_module(f"mosaic.small_molecule.{STAGES[stage]}.pipeline")
    output = general_path.parent / Path(general["output_dir"]).expanduser()
    workflow.run(general, stage_config, general_path, stage_path, output)
    print(f"Results saved to {output.resolve()}")


if __name__ == "__main__":
    main()
