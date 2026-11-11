"""Add chemistry rules after each ligand update.

Stage 2 uses the same starting graph, model, and gradient updates as Stage 1.
After an update, it can reduce excess bond order and optionally bridge
fragments or shrink weak bonds. The selected operators come from the YAML.
"""

from contextlib import ExitStack
import os


def run(general, stage, general_path, stage_path, output):
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Run Stage 2 inside an allocated DTU HPC GPU job.")

    from mosaic.small_molecule.Stage1.pipeline import optimize

    from .operators import Stage2Config, post_step

    operators = Stage2Config(**(stage["operators"] | stage.get("improvements", {})))
    hardening = stage.get("hardening", {})
    prepare_loss = None
    if hardening.get("enabled", False):
        from .hardening import hardening_factory

        if not general["initialization"].get("checkpoint"):
            raise ValueError("Hardening needs a starting checkpoint shared with its relaxed control.")

        def prepare_loss(loss, epochs):
            return hardening_factory(loss, epochs, hardening.get("gradient", "inherited"))

    with ExitStack() as resources:
        optimize(
            general, stage, general_path, stage_path, output, resources,
            after_update=post_step(operators) if operators.mode != "none" else None,
            prepare_loss=prepare_loss,
            stage_details={"operators": operators.record(), "hardening": hardening},
        )
