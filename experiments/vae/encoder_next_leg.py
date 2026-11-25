"""Continue the trained three-block encoder branch with one new generated-parent objective."""

import argparse
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
import json
import math
import os
from pathlib import Path
import time

import torch

from experiments.vae.encoder_upgrade_pilot import PILOT_RECIPE, effective_config
from latentmol.datasets.bulk_view import BalancedSparseCorpus
from latentmol.datasets.ordering import verify_canonical_training_cache
from latentmol.experiments import file_sha256, write_json
from latentmol.models import GraphAutoencoder
from latentmol.training.combined_continuation import _check_cuda_requirement, _load, _train
from latentmol.training.motif_activation import _digest
from latentmol.training.projected_continuation import _arm_identity, _read_resume, _student, projected_update


def successor_config(source, fork, state, base_protocol, recipe, policy):
    """Authenticate the trained branch. Preserve its selected motifs and confidence supervision."""
    candidate = _digest(dict(experiment="epoch48-three-block-encoder", pilot=recipe,
                             predecessor_sha256=fork["resume"]["sha256"], source_identity=base_protocol))
    expected_upgrade = dict(parent_sha256=fork["resume"]["sha256"], policy=recipe)
    if (fork["source_config_sha256"] != base_protocol or fork["config_sha256"] != candidate
            or fork["encoder_seed"] != recipe["encoder_seed"]
            or state["experiment_state"].get("encoder_upgrade") != expected_upgrade):
        raise ValueError("Source is not the trained three-block encoder branch")
    config = effective_config(source, state, base_protocol, recipe, state_identity=candidate)
    if (fork["generated_roundtrip"] != config["generated_roundtrip"]
            or fork["motif_winner"] != state["experiment_state"]["motif_activation"]["winner"]):
        raise ValueError("Candidate generated-parent or motif policy differs from its fork")
    if set(policy) != {"parent_source", "false_positive_bonds"} or policy["parent_source"] != "eval":
        raise ValueError("Expected eval-mode parent generation and a false-positive bond objective")
    bonds = policy["false_positive_bonds"]
    if (not isinstance(bonds, dict) or set(bonds) != {"weight", "max_pairs"}
            or type(bonds["max_pairs"]) is not int or bonds["max_pairs"] < 1
            or type(bonds["weight"]) not in (float, int) or not math.isfinite(bonds["weight"]) or bonds["weight"] <= 0):
        raise ValueError("Invalid false-positive bond policy")
    config["generated_roundtrip"] = dict(config["generated_roundtrip"], **policy)
    config["quality_panel"] = dict(config["quality_panel"], generated_cycle_bins=True)
    return config


def check_terminal(summary, receipt, fork):
    if (summary["status"] != "paused"
            or summary["stop_reason"] not in {"allocation_budget_reached", "update_budget_reached"}
            or summary["latest"] != receipt or summary["updates"] != receipt["updates"]
            or summary["segment_updates"] <= 0
            or summary["segment_updates"] != receipt["updates"] - fork["resume"]["updates"]):
        raise ValueError("Expected the candidate's terminal checkpoint for its unfinished epoch")


def check_operator_stop(path, directory, receipt, fork):
    """Accept the explicit stop handoff after the operator has verified the LSF writer exited."""
    if path.resolve(strict=True) != directory.parent / "operator-stop.json":
        raise ValueError("Expected the pinned operator-stop receipt alongside the checkpoint directory")
    proof = json.loads(path.read_bytes())
    original_dir = Path(proof["source_directory"]).resolve(strict=True)
    original = proof["source_latest"]
    filename = f"checkpoint-{receipt['updates']:09d}.pt"
    if (proof["status"] != "operator_stopped" or proof["terminal_status"] != "EXIT"
            or proof["job_id"] != 30330582 or original_dir == directory
            or original_dir != Path(fork["production_control"]).resolve().parent / "candidate"
            or proof["latest"] != receipt or original != dict(receipt, checkpoint=str(original_dir / filename))
            or Path(receipt["checkpoint"]) != directory / filename
            or not fork["resume"]["updates"] < receipt["updates"] <= proof["last_logged_update"]
            or file_sha256(directory / "fork.json") != proof["fork_sha256"]
            or file_sha256(original_dir / "fork.json") != proof["fork_sha256"]):
        raise ValueError("Operator-stop handoff differs from the candidate source or pinned checkpoint")
    # _read_resume below checks the copied bytes, authenticated model, optimizer geometry and cursor.
    return dict(job_id=proof["job_id"], saved_update=receipt["updates"],
                last_logged_update=proof["last_logged_update"], receipt_sha256=file_sha256(path))


def run(args):
    entered = time.monotonic()
    recipe = json.loads(PILOT_RECIPE.read_bytes())
    base = PILOT_RECIPE.with_name(recipe["source_recipe"])
    if args.config.resolve(strict=True) != base.resolve(strict=True):
        raise ValueError("Unexpected base training recipe")
    directory = args.source_dir.resolve(strict=True)
    output = args.output.resolve()
    if output.is_relative_to(directory) or directory.is_relative_to(output) or output.exists():
        raise ValueError("A new output directory disjoint from the source arm is required")
    if not os.environ.get("LSB_JOBID", "").isdigit() or not torch.cuda.is_available():
        raise RuntimeError("Requires an allocated CUDA worker")
    source = json.loads(base.read_bytes())
    fork = json.loads((directory / "fork.json").read_bytes())
    latest = directory / "latest.json"
    receipt = json.loads(latest.read_bytes())
    stop = None
    if stop_receipt := getattr(args, "operator_stop_receipt", None):
        stop = check_operator_stop(stop_receipt, directory, receipt, fork)
    else:
        check_terminal(json.loads((directory / "summary.json").read_bytes()), receipt, fork)
    policy = json.loads(args.policy.read_bytes())
    parent = fork["parent"]
    source = dict(source, checkpoint={"path": parent["checkpoint"], "sha256": parent["sha256"]})
    base_protocol = _arm_identity(base, parent["sha256"], 3)
    _check_cuda_requirement(dict(source, cuda_requirement=source["cuda_by_extra_rounds"]["3"]))
    torch.set_num_threads(source["execution"]["torch_threads"])
    corpus = BalancedSparseCorpus(source["data"], expected_sha256=source["data_sha256"])
    data = corpus.split("train")
    verify_canonical_training_cache(data, range(16))
    old_model, payload, *references = _load(source, corpus, "cuda")
    if (len(data) != 60_000_000 or payload["config_sha256"] != source["parent_config_sha256"]
            or payload["dataset_sha256"] != source["data_sha256"]
            or payload["continuation"]["updates"] != source["parent_updates"]):
        raise ValueError("Unexpected epoch-47 base checkpoint or dataset")
    # _student strictly restores trained tensors, positional AdamW moments and RNG.
    # Its template must already include the three encoder blocks when the optimizer is created.
    template = GraphAutoencoder(old_model.spec, replace(old_model.config, encoder_degree_features=True,
        encoder_graph_attention_layers=3, atom_refinement_dropout=source["atom_refinement_dropout"]))
    verified, state = _read_resume(latest, fork["config_sha256"], source, template, payload, 3, len(data))
    if verified != receipt:
        raise ValueError("Terminal receipt changed while restoring")
    config = successor_config(source, fork, state, base_protocol, recipe, policy)
    handoff = dict(experiment="epoch48-generated-bond-successor", source_protocol=fork["config_sha256"],
                   predecessor_sha256=verified["sha256"], policy=policy)
    if stop:
        handoff["operator_stop"] = stop
    identity = _digest(handoff)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "fork.json", dict(parent=parent, resume=verified, config_sha256=identity,
        source_config_sha256=fork["config_sha256"], generated_roundtrip=config["generated_roundtrip"],
        motif_winner=fork["motif_winner"], quality_panel=config["quality_panel"],
        wall_seconds=source["wall_seconds"], target_updates=source["leg_updates"]))
    if marker := os.environ.get("DBMOL_LAUNCH_READY"):
        write_json(Path(marker), dict(status="models_and_inputs_ready", job_id=os.environ["LSB_JOBID"],
                                      nonce=os.environ["DBMOL_LAUNCH_NONCE"]))
    metadata = dict(state["experiment_state"], generated_bond_handoff=handoff)
    student = partial(_student, count=3, seed=source["extension_seed"], atom_order=source["atom_order"])
    return _train(config, identity, data, (template, payload, *references), output, "cuda",
                  source["batch_size"], state, source["leg_updates"], entered + source["wall_seconds"],
                  source["microbatch"], validation=corpus.split("val"), student_factory=student,
                  teacher_factory=lambda *_: nullcontext(), update_fn=projected_update,
                  experiment_state=metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--operator-stop-receipt", type=Path,
                        help="Operator-verified EXIT and checkpoint handoff for an early cutover")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
