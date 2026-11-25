"""Train the three-block encoder alongside the existing next epoch-48 allocation.

The production worker provides the control. This command only reads the completed
predecessor and only writes below its own new output directory. It uses the real
training cursor, optimizer, shuffled rows, RNGs, prior reference and quality panels.
"""

import argparse
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
import json
import os
from pathlib import Path
import re
import time

import torch

from latentmol.datasets.bulk_view import BalancedSparseCorpus
from latentmol.datasets.ordering import verify_canonical_training_cache
from latentmol.experiments import write_json
from latentmol.training.combined_continuation import _check_cuda_requirement, _load, _train
from latentmol.training.motif_activation import _digest, policies
from latentmol.training.projected_continuation import _arm_identity, _read_resume, _student, projected_update


PILOT_RECIPE = Path(__file__).with_name("epoch48-encoder-upgrade.json")


def effective_config(source, state, source_identity, pilot, *, state_identity=None):
    """Inherit the selected motifs after verifying the handoff and the resumed branch."""
    metadata = state["experiment_state"]
    motif = metadata["motif_activation"]
    handoff = metadata["roundtrip_handoff"]
    _, options = policies(source)
    selected = options[motif["winner"]]
    if (handoff["target_protocol"] != source_identity or state["config_sha256"] != (state_identity or source_identity)
            or motif["protocol"] != handoff["source_protocol"]
            or motif["policy_sha256"] != _digest(selected)
            or not 0 < motif["adopted_at_update"] <= state["continuation"]["updates"]):
        raise ValueError("The source checkpoint and its selected motif policy disagree")
    # Production configures ring monitoring BEFORE applying the winner, so retain
    # that diagnostic even when the inherited winner excludes the ring-size loss.
    panel = dict(source["quality_panel"], monitor_motifs=True,
                 monitor_ring_sizes=bool(source["functional_groups"].get("ring_size")),
                 generated_roundtrip_points=128)
    roundtrip = dict(source["generated_roundtrip"])
    if pilot is not None:
        roundtrip["excess_confidence"] = pilot["excess_confidence"]
    return dict(source, functional_groups=selected, quality_panel=panel, generated_roundtrip=roundtrip)


def candidate_student(*args, encoder_seed, **kwargs):
    """Restore the seven-round production model and AdamW state before expansion."""
    from latentmol.training.encoder_expansion import expand_encoder

    model, optimizer, training, generators = _student(*args, **kwargs)
    expand_encoder(model, optimizer, seed=encoder_seed)
    return model, optimizer, replace(training, model=model.config), generators


def run(args, pilot, *, early_arm=None, source_sha256=None):
    """Restore production state. Early forks require an independently pinned checkpoint."""
    entered = time.monotonic()
    if early_arm is not None and (early_arm not in {"candidate", "control"} or not
            isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha256)):
        raise ValueError("An early fork requires a named arm and pinned checkpoint SHA-256")
    control_arm = early_arm == "control"
    base_path = PILOT_RECIPE.with_name(pilot["source_recipe"])
    if args.config.resolve(strict=True) != base_path.resolve(strict=True):
        raise ValueError("Use the production source recipe selected by the encoder pilot")
    source = json.loads(base_path.read_bytes())
    updates = pilot["updates"] if args.updates is None else args.updates
    wall = pilot["wall_seconds"] if args.wall_seconds is None else args.wall_seconds
    if updates < 1 or wall <= 120:
        raise ValueError("Require positive update budget and time for the checkpoint reserve")
    source_dir = args.resume_receipt.resolve().parent
    control_dir = args.production_control.resolve()
    dest = args.output.resolve()
    if dest.is_relative_to(source_dir) or source_dir.is_relative_to(dest):
        raise ValueError("Training output overlaps its source")
    if control_arm:
        if dest != control_dir:
            raise ValueError("Control arm must write to its designated control directory")
    elif dest.is_relative_to(control_dir) or control_dir.is_relative_to(dest):
        raise ValueError("Candidate and control output directories must be separate")
    if not os.environ.get("LSB_JOBID", "").isdigit() or not torch.cuda.is_available():
        raise RuntimeError("Run the pilot on an allocated GPU")
    _check_cuda_requirement(dict(source, cuda_requirement=source["cuda_by_extra_rounds"]["3"]))
    torch.set_num_threads(source["execution"]["torch_threads"])
    parent = json.loads(args.parent_receipt.read_bytes())
    source = dict(source, checkpoint={"path": parent["checkpoint"], "sha256": parent["sha256"]})
    corpus = BalancedSparseCorpus(source["data"], expected_sha256=source["data_sha256"])
    data = corpus.split("train")
    verify_canonical_training_cache(data, range(16))
    assets = _load(source, corpus, "cuda")
    model, payload, *_ = assets
    if (len(data) != 60_000_000 or payload["config_sha256"] != source["parent_config_sha256"]
            or payload["dataset_sha256"] != source["data_sha256"]
            or payload["continuation"]["updates"] != source["parent_updates"]):
        raise ValueError("Unexpected epoch-47 base checkpoint or data")
    source_identity = _arm_identity(base_path, parent["sha256"], 3)
    receipt, state = _read_resume(args.resume_receipt, source_identity, source, model, payload, 3, len(data))
    if early_arm is None:
        terminal = json.loads((source_dir / "summary.json").read_bytes())
        if (terminal["status"] != "paused" or terminal["stop_reason"] != "allocation_budget_reached"
                or terminal["latest"] != receipt or terminal["segment_updates"] <= 0):
            raise ValueError("The predecessor has not completed an unfinished allocation")
    elif receipt["sha256"] != source_sha256:
        raise ValueError("The authenticated saved checkpoint differs from the pinned early fork")
    config = effective_config(source, state, source_identity, None if control_arm else pilot)
    identity = source_identity if control_arm else _digest(dict(experiment="epoch48-three-block-encoder",
        pilot=pilot, predecessor_sha256=receipt["sha256"], source_identity=source_identity))
    dest.mkdir(parents=True, exist_ok=False)
    write_json(dest / "fork.json", dict(parent=parent, resume=receipt, production_control=str(control_dir),
               target_updates=updates, wall_seconds=wall, source_config_sha256=source_identity,
               config_sha256=identity, encoder_seed=None if control_arm else pilot["encoder_seed"],
               generated_roundtrip=config["generated_roundtrip"], motif_winner=state["experiment_state"]["motif_activation"]["winner"],
               batch_size=config["batch_size"], microbatch_size=config["microbatch"]))
    if marker := os.environ.get("DBMOL_LAUNCH_READY"):
        write_json(Path(marker), dict(status="models_and_inputs_ready", job_id=os.environ["LSB_JOBID"],
                                     nonce=os.environ["DBMOL_LAUNCH_NONCE"]))
    metadata = dict(state["experiment_state"])
    factory = _student
    if not control_arm:
        factory = partial(candidate_student, encoder_seed=pilot["encoder_seed"])
        metadata["encoder_upgrade"] = {"parent_sha256": receipt["sha256"], "policy": pilot}
    student = partial(factory, count=3, seed=source["extension_seed"], atom_order=source["atom_order"])
    _train(config, identity, data, assets, dest, "cuda", source["batch_size"], state, updates,
           entered + wall, source["microbatch"], validation=corpus.split("val"),
           student_factory=student, teacher_factory=lambda *_: nullcontext(),
           update_fn=projected_update, experiment_state=metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="The production source recipe for CUDA and training")
    parser.add_argument("--parent-receipt", type=Path, required=True, help="Original epoch-47 parent.json")
    parser.add_argument("--resume-receipt", type=Path, required=True, help="Finished segment's latest.json")
    parser.add_argument("--production-control", type=Path, required=True, help="Next ordinary segment directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--wall-seconds", type=int)
    run(parser.parse_args(), json.loads(PILOT_RECIPE.read_bytes()))
