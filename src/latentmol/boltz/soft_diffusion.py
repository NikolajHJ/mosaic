"""Small inference-only soft-graph probe of the pinned Boltz-2 diffusion model.

The existing probability adapter and expected typed-bond trunk are reused.
Diffusion consumes continuous element/charge features without graph hardening.
Reference conformer, atom names, atom count, masks and MSA remain conditioning.
"""

import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from latentmol.experiments import file_sha256
from latentmol.representation import RelaxedGraph
from .adapter import adapt_graph
from .contracts import PreparedComplex
from .dbmol_model import load_confidence_model
from .model import attached_trunk


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_soft_graph(path, prepared):
    saved = json.loads(Path(path).read_text())
    spec = prepared.spec
    if saved.get("snapshot_version") != "shared-relaxed-graph-v1" or saved["graph_spec"] != spec.to_dict():
        raise ValueError("Saved soft-graph format/schema differs from prepared complex")
    if saved["row_slots"] != list(range(spec.max_nodes)):
        raise ValueError("This probe requires canonical complete slot order")
    if saved["active_slots"] != prepared.active.nonzero().flatten().tolist():
        raise ValueError("Soft-graph occupancy differs from prepared slot correspondence")
    shapes = ((spec.max_nodes, len(spec.atomic_numbers)),
              (spec.max_nodes, len(spec.formal_charges)),
              (spec.max_nodes, spec.max_nodes, len(spec.bond_names)))
    values = []
    for key, shape in zip(("atoms", "charges", "bonds"), shapes):
        value = torch.tensor(saved[key], dtype=torch.float32)
        if tuple(value.shape) != shape or not bool(value.isfinite().all()):
            raise ValueError("Malformed/nonfinite soft " + key)
        if bool((value < 0).any()) or bool((value > 1).any()) or not torch.allclose(value.sum(-1), torch.ones(shape[:-1]), atol=2e-6, rtol=0):
            raise ValueError("Invalid probability simplex: " + key)
        values.append(value.unsqueeze(0))
    graph = RelaxedGraph(*values)
    if not torch.allclose(graph.bonds, graph.bonds.transpose(1, 2), atol=2e-6, rtol=0):
        raise ValueError("Asymmetric soft bonds")
    if not torch.equal(graph.bonds[0].diagonal(dim1=0, dim2=1)[0], torch.ones(spec.max_nodes)):
        raise ValueError("Self bonds must be NONE")
    if not torch.allclose(graph.occupancy[0], prepared.active.float(), atol=1e-6, rtol=0):
        raise ValueError("Fixed-occupancy diffusion probe cannot change atom count")
    active_probs = graph.atoms[0, prepared.active]
    if not bool(((active_probs > .01) & (active_probs < .99)).any()):
        raise ValueError("Probe input lacks a materially soft atom mixture")
    return graph, saved


def frozen_versions(model):
    return {name: (id(value), value._version, value.requires_grad)
            for name, value in list(model.named_parameters()) + list(model.named_buffers())}


def predict(model, prepared, graph, *, steps=40, seed=17, progress=None):
    """Run native conditioning/sampling; record denoised estimates without edits."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Diffusion probe requires a frozen eval model")
    if steps < 2 or steps > 100:
        raise ValueError("Small probe budget requires 2–100 diffusion steps")
    features, native_bonds = adapt_graph(graph, prepared)
    baseline = {key: value.detach().clone() for key, value in features.items()}
    versions = frozen_versions(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    frames, noise_levels, consumption = [], [], {}

    def check_conditioning(module, arguments, kwargs):
        actual = kwargs["feats"]["ref_element"][0, prepared.slot_to_atom[prepared.active]]
        expected = features["ref_element"][0, prepared.slot_to_atom[prepared.active]]
        if not torch.equal(actual, expected):
            raise ValueError("Diffusion received a different element representation")
        consumption.update(calls=consumption.get("calls", 0) + 1,
                           native_element_rows=actual.detach().cpu().tolist(),
                           graph_hardened=False)

    handle = model.diffusion_conditioning.register_forward_pre_hook(check_conditioning, with_kwargs=True)
    original = model.structure_module.preconditioned_network_forward

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        if not bool(result.isfinite().all()):
            raise ValueError("Nonfinite native denoised coordinates")
        frames.append(result.detach().cpu().numpy().copy())
        sigma = args[1] if len(args) > 1 else kwargs["sigma"]
        noise_levels.append(float(sigma))
        if progress:
            progress({"event": "denoised", "step": len(frames), "sigma": float(sigma)})
        return result

    model.structure_module.preconditioned_network_forward = capture
    try:
        with torch.inference_mode():
            inputs, single, pair = attached_trunk(model, features, native_bonds, recycling_steps=0)
            if not all(bool(x.isfinite().all()) for x in (inputs, single, pair)):
                raise ValueError("Nonfinite structural trunk")
            conditioning_values = model.diffusion_conditioning(
                s_trunk=single, z_trunk=pair,
                relative_position_encoding=model.rel_pos(features), feats=features)
            conditioning = dict(zip(("q", "c", "to_keys", "atom_enc_bias", "atom_dec_bias", "token_trans_bias"), conditioning_values))
            steering = dict(model.steering_args)
            for key in ("fk_steering", "physical_guidance_update", "contact_guidance_update"):
                steering[key] = False
            result = model.structure_module.sample(
                s_trunk=single.float(), s_inputs=inputs.float(), feats=features,
                num_sampling_steps=steps, atom_mask=features["atom_pad_mask"].float(),
                multiplicity=1, max_parallel_samples=1, steering_args=steering,
                diffusion_conditioning=conditioning)
            coordinates = result["sample_atom_coords"]
            if coordinates.shape != prepared.coordinates.reshape(1, -1, 3).shape or not bool(coordinates.isfinite().all()):
                raise ValueError("Malformed/nonfinite final diffusion coordinates")
    finally:
        model.structure_module.preconditioned_network_forward = original
        handle.remove()
    if len(frames) != steps or consumption.get("calls") != 1:
        raise ValueError("Unexpected native sampling/conditioning call count")
    if frozen_versions(model) != versions:
        raise ValueError("Frozen model parameter/buffer version changed")
    if set(features) != set(baseline) or any(not torch.equal(features[k], baseline[k]) for k in baseline):
        raise ValueError("Native diffusion modified its feature inputs")
    return coordinates.detach().cpu().numpy(), np.concatenate(frames, axis=0), noise_levels, consumption


def run(args, output):
    if not os.environ.get("LSB_JOBID") or args.device != "cuda":
        raise ValueError("Native diffusion runs only inside an allocated GPU job")
    if file_sha256(args.context) != args.context_sha256 or file_sha256(args.probabilities) != args.probabilities_sha256:
        raise ValueError("Input asset checksum mismatch")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prepared = PreparedComplex.load(args.context, device="cpu")
    graph, saved = load_soft_graph(args.probabilities, prepared)
    model = load_confidence_model({"path": str(args.checkpoint), "sha256": args.checkpoint_sha256}).to(args.device)
    device = torch.cuda.get_device_properties(0)
    cpu_prepared = prepared
    prepared = prepared.to(args.device)
    graph = RelaxedGraph(*(x.to(args.device) for x in (graph.atoms, graph.charges, graph.bonds)))
    print(json.dumps({"event": "model_ready", "device": device.name, "steps": args.steps}), flush=True)
    started = time.monotonic()
    coordinates, frames, sigmas, consumed = predict(
        model, prepared, graph, steps=args.steps, seed=args.seed,
        progress=lambda row: print(json.dumps(row), flush=True))
    atom_to_token = cpu_prepared.features["atom_to_token"][0].float()
    protein_atoms = (atom_to_token @ (cpu_prepared.features["mol_type"][0] == 0).float()).bool()
    protein_atoms &= cpu_prepared.features["atom_pad_mask"][0].bool()
    with (output / "coordinates.npz").open("xb") as stream:
        np.savez_compressed(stream, predicted=coordinates, denoised_estimates=frames,
                            reference=cpu_prepared.coordinates.reshape(1, -1, 3).numpy(),
                            sigmas=np.asarray(sigmas),
                            slot_to_atom=cpu_prepared.slot_to_atom.numpy(),
                            protein_atom_mask=protein_atoms.numpy(),
                            atom_pad_mask=cpu_prepared.features["atom_pad_mask"][0].numpy())
    write_json(output / "soft_graph.json", saved)
    if file_sha256(args.context) != args.context_sha256 or file_sha256(args.probabilities) != args.probabilities_sha256:
        raise ValueError("Input file changed during inference")
    return {"status": "completed", "job_id": os.environ["LSB_JOBID"],
            "model": "Boltz-2 structural checkpoint", "boltz_commit": cpu_prepared.metadata["boltz_commit"],
            "source_archive_sha256": os.environ.get("LATENTMOL_SOURCE_SHA256"),
            "checkpoint_sha256": args.checkpoint_sha256, "prepared_sha256": args.context_sha256,
            "probabilities_sha256": args.probabilities_sha256, "soft_graph_step": saved["step"],
            "seed": args.seed, "sampling_steps": args.steps, "samples": 1, "recycling_steps": 0,
            "guidance": "all physical/contact/FK guidance disabled",
            "mapping": "existing continuous expected-element/charge and expected typed-bond adapter; no hardening",
            "retained_conditioning": "reference conformer, atom names, count, tokenization, MSA, templates and other native features",
            "soft_conditioning_consumption": consumed, "model_versions_unchanged": True,
            "adapted_features_unchanged": True, "source_files_unchanged": True,
            "coordinate_shape": list(coordinates.shape), "denoised_estimates": len(frames),
            "elapsed_prediction_seconds": time.monotonic()-started,
            "device": device.name, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "claim_limit": "Exploratory soft-input prediction; no physical relaxation, validity, affinity or derivative claim. Native model was trained on discrete chemistry."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("context", "probabilities", "checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        result = run(args, args.output)
    except BaseException as error:
        write_json(args.output / "result.json", {"status": "failed", "error": str(error),
                   "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic()-started})
        raise
    result["elapsed_seconds"] = time.monotonic()-started
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
