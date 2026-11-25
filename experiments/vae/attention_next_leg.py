"""Resume the epoch-48 VAE with wider atom attention and incidence bond attention."""

import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import torch

from experiments.vae.encoder_upgrade_pilot import PILOT_RECIPE, effective_config
from latentmol.datasets.bulk_view import BalancedSparseCorpus
from latentmol.datasets.ordering import verify_canonical_training_cache
from latentmol.experiments import write_json
from latentmol.models import GraphAutoencoder
from latentmol.training.attention_expansion import expand_attention
from latentmol.training.combined_continuation import _check_cuda_requirement, _load, _train
from latentmol.training.motif_activation import _digest
from latentmol.training.projected_continuation import _arm_identity, _read_resume, _student, projected_update


def inherited_config(base, recipe, source, state, fork):
    """Recover the exact objectives selected in the predecessor and verify the handoff."""
    prior = state["experiment_state"]["generated_bond_handoff"]
    if (_digest(prior) != fork["config_sha256"] or prior["experiment"] != "epoch48-generated-bond-successor"
            or prior["predecessor_sha256"] != fork["resume"]["sha256"]
            or prior["source_protocol"] != fork["source_config_sha256"]):
        raise ValueError("Mismatched generated-bond predecessor")
    identity = _arm_identity(base, fork["parent"]["sha256"], 3)
    config = effective_config(source, state, identity, recipe, state_identity=fork["config_sha256"])
    config["generated_roundtrip"] = dict(config["generated_roundtrip"], **prior["policy"])
    config["quality_panel"] = dict(config["quality_panel"], generated_cycle_bins=True)
    if (config["generated_roundtrip"] != fork["generated_roundtrip"] or config["quality_panel"] != fork["quality_panel"]
            or fork["motif_winner"] != state["experiment_state"]["motif_activation"]["winner"]):
        raise ValueError("Training losses or quality monitoring differ from the source fork")
    return config


def run(args):
    entered = time.monotonic()
    recipe = json.loads(PILOT_RECIPE.read_bytes())
    base = PILOT_RECIPE.with_name(recipe["source_recipe"])
    directory = args.source_dir.resolve(strict=True)
    output = args.output.resolve()
    if (args.config.resolve() != base.resolve() or output.exists() or output.is_relative_to(directory)
            or directory.is_relative_to(output)):
        raise ValueError("Unexpected training recipe or overlapping output directory")
    if not os.environ.get("LSB_JOBID", "").isdigit() or not torch.cuda.is_available():
        raise RuntimeError("Requires an allocated GPU")
    receipt = json.loads((directory / "latest.json").read_bytes())
    cutoff = json.loads((directory.parent / "cutover.json").read_bytes())
    if (cutoff["latest"] != receipt or cutoff["terminal_status"] not in {"DONE", "EXIT"}
            or cutoff["last_logged_update"] < receipt["updates"]):
        raise ValueError("Pinned checkpoint differs from the stopped worker's receipt")
    fork = json.loads((directory / "fork.json").read_bytes())
    source = json.loads(base.read_bytes())
    parent = fork["parent"]
    source["checkpoint"] = dict(path=parent["checkpoint"], sha256=parent["sha256"])
    _check_cuda_requirement(dict(source, cuda_requirement=source["cuda_by_extra_rounds"]["3"]))
    torch.set_num_threads(source["execution"]["torch_threads"])
    corpus = BalancedSparseCorpus(source["data"], expected_sha256=source["data_sha256"])
    data = corpus.split("train")
    verify_canonical_training_cache(data, range(16))
    old, payload, *references = _load(source, corpus, "cuda")
    if (len(data) != 60_000_000 or payload["config_sha256"] != source["parent_config_sha256"]
            or payload["continuation"]["updates"] != source["parent_updates"]):
        raise ValueError("Unexpected original training checkpoint or corpus")
    template = GraphAutoencoder(old.spec, replace(old.config, encoder_degree_features=True,
        encoder_graph_attention_layers=3, atom_refinement_dropout=source["atom_refinement_dropout"]))
    verified, state = _read_resume(directory / "latest.json", fork["config_sha256"], source,
                                   template, payload, 3, len(data))
    if verified != receipt:
        raise ValueError("Pinned source changed during restoration")
    config = inherited_config(base, recipe, source, state, fork)
    handoff = dict(experiment="epoch48-atom-and-bond-attention-v1", source_protocol=fork["config_sha256"],
                   predecessor_sha256=receipt["sha256"], initialization_seed=2026112507)
    identity = _digest(handoff)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "fork.json", dict(parent=parent, resume=receipt, config_sha256=identity,
               source_config_sha256=fork["config_sha256"], generated_roundtrip=config["generated_roundtrip"],
               motif_winner=fork["motif_winner"], quality_panel=config["quality_panel"],
               wall_seconds=config["wall_seconds"], target_updates=config["leg_updates"], attention=handoff))

    def student(*inputs, **kwargs):
        model, optimizer, training, generators = _student(*inputs, **kwargs, count=3,
            seed=source["extension_seed"], atom_order=source["atom_order"])
        old_count = sum(p.numel() for p in model.parameters())
        expand_attention(model, optimizer, seed=handoff["initialization_seed"])
        count = sum(p.numel() for p in model.parameters())
        if (old_count, count) != (4_526_158, 5_058_652):
            raise RuntimeError(f"Unexpected parameter totals {old_count} -> {count}")
        write_json(output / "attention-graft.json", dict(start_update=receipt["updates"],
                   original_parameters=old_count, expanded_parameters=count, model=vars(model.config)))
        if marker := os.environ.get("DBMOL_LAUNCH_READY"):
            write_json(Path(marker), dict(status="models_and_inputs_ready", job_id=os.environ["LSB_JOBID"],
                                          nonce=os.environ["DBMOL_LAUNCH_NONCE"]))
        return model, optimizer, replace(training, model=model.config), generators

    return _train(config, identity, data, (template, payload, *references), output, "cuda",
                  config["batch_size"], state, config["leg_updates"], entered + config["wall_seconds"],
                  config["microbatch"], validation=corpus.split("val"), student_factory=student,
                  teacher_factory=lambda *_: nullcontext(), update_fn=projected_update,
                  experiment_state=dict(state["experiment_state"], attention_graft=handoff))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())
