"""DRAFT: one entry point for the small-molecule workflows.

Example, from the repository root with Mosaic installed:
    python -m mosaic.small_molecule.generate_molecule \
        --general-config configs/general_config.yaml \
        --stage-config configs/stage0.yaml

Suggested general YAML (field names are a proposal, not an existing standard):
    model: protenix-tiny
    protein: {structure: path/to/complex.pdb, chain: A, use_msa: true}
    pocket:
      reference_ligand: {chain: A, residue_name: JQ1, number: 1}
      cutoff_angstrom: 6.0
    initialization: {input: path/to/prepared_molecule}
    epochs: 1000
    learning_rate: 0.01
    seed: 17
    output_dir: ../results/my-unique-run  # Relative to the general YAML file.

Suggested stage YAML:
    stage: stage0
    losses:
      - {name: contact, weight: 1.0}
    parameters: {}  # Settings specific to this workflow.

The YAML files currently contain no settings. The StageX/VAE implementations
also do not exist yet: the TODOs below describe the missing connections.
No model, optimization or run-test has been executed for this draft.
"""

import argparse
from pathlib import Path


def read_config(path):
    # TODO: declare PyYAML as a direct project dependency if it is not already.
    import yaml

    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not config:
        raise ValueError(f"Fill in the YAML settings first: {path}")
    return config


def load_workflow(stage):
    # Each empty folder should eventually contain pipeline.py. Import only
    # the selected workflow so other stages do not need to be installed/ready.
    # TODO: add pipeline.py in these folders, using the functions called below.
    if stage == "stage0":
        from mosaic.small_molecule.Stage0 import pipeline
    elif stage == "stage1":
        from mosaic.small_molecule.Stage1 import pipeline
    elif stage == "stage2":
        from mosaic.small_molecule.Stage2 import pipeline
    elif stage == "stage3":
        from mosaic.small_molecule.Stage3 import pipeline
    elif stage == "vae":
        from mosaic.small_molecule.VAE import pipeline
    else:
        raise ValueError("stage must be stage0, stage1, stage2, stage3, or VAE")
    return pipeline


def prepare_model_and_target(general, config_directory):
    # TODO: connect this to shared helpers under mosaic.small_molecule.
    # Check that an appropriate HPC allocation exists before numerical work.
    # 1. Read the protein structure and select the requested chain.
    # 2. Resolve the pocket from a reference ligand OR explicit residues.
    # 3. Exclude the reference ligand from the target model input.
    # 4. Load the requested model and prepare its inputs, including a protein
    #    MSA/template if requested and supported by that model.
    # 5. Return the frozen model and prepared target/pocket information.
    # Resolve input paths relative to config_directory, not an account's home.
    # A pocket selection alone does not constrain the predicted 3D coordinates;
    # the chosen model must explicitly support any requested structural input.
    raise NotImplementedError("TODO: shared model, protein and pocket preparation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--general-config", type=Path, required=True)
    parser.add_argument("--stage-config", type=Path, required=True)
    args = parser.parse_args()
    general_path = args.general_config.resolve()
    stage_path = args.stage_config.resolve()
    general = read_config(general_path)
    stage_config = read_config(stage_path)


    epochs = general["epochs"]
    learning_rate = general["learning_rate"]

    stage = str(stage_config["stage"]).lower()
    workflow = load_workflow(stage)

    # This follows the protein-hallucination example: model + target inputs,
    # then a starting representation, a combined loss and an optimizer.
    model, target = prepare_model_and_target(general, general_path.parent)

    # TODO in each pipeline.py: initialize_molecule reads the separately
    # prepared input. For VAE, this may instead initialize latent variables.
    # Keep the initial state separate so results can include it as epoch 0.
    molecule = workflow.initialize_molecule(
        settings=general["initialization"],
        parameters=stage_config.get("parameters", {}),
        config_directory=general_path.parent,
        seed=general.get("seed", 17),
    )

    # TODO: build_loss combines the requested differentiable losses. A later
    # implementation can read changing weights from the stage YAML too.
    loss = workflow.build_loss(
        model=model, target=target, settings=stage_config.get("losses", []),
    )

    # TODO: optimize performs exactly 'epochs' updates and returns a result
    # containing the final molecule plus the initial state and full trajectory.
    # Record each loss, its weight, and relevant contact/affinity metrics at
    # every update. The stage owns its algorithm; VAE may optimize a latent
    # vector while another stage directly optimizes atom/bond probabilities.
    result = workflow.optimize(
        molecule=molecule, loss=loss, epochs=epochs,
        learning_rate=learning_rate, seed=general.get("seed", 17),
        parameters=stage_config.get("parameters", {}),
    )

    # TODO: evaluate should assess the final molecule with the selected model.
    # Label contact probability and affinity separately in the saved results.
    evaluation = workflow.evaluate(result=result, model=model, target=target)

    output = Path(general["output_dir"]).expanduser()
    if not output.is_absolute():
        output = general_path.parent / output
    output.mkdir(parents=True, exist_ok=False)  # Preserve earlier runs.
    (output / "general_config.yaml").write_bytes(general_path.read_bytes())
    (output / "stage_config.yaml").write_bytes(stage_path.read_bytes())

    # TODO: implement save_results (prefer shared output helpers) to write the
    # trajectory, evaluation, model/input identities and completion status.
    workflow.save_results(result=result, evaluation=evaluation, output=output)
    print(f"Results saved to {output.resolve()}")


if __name__ == "__main__":
    main()
