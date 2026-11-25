"""User-authorized, bounded local VAE comparison; never an HPC resume.

Both arms inherit checkpoint42 weights and AdamW moments, frozen35 occupancy
and frozen36 valid-target anchors. Local minibatches and RNGs are deliberately
new, identical across arms, and recorded. This entrypoint performs no SSH,
scheduler, remote write, automatic continuation or production promotion.
"""
import argparse
import base64
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

# Set before importing/initializing CUDA; keep deterministic kernels enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher
from latentmol.chemical_environment_policy import assess_environment
from latentmol.chemistry import encode_smiles
from latentmol.chemistry_classification import classify_repair_graph
from latentmol.chemistry_continuation import chemistry_update, load_parent_config
from latentmol.chemistry_valid_reference_anchor import FrozenValidReferenceAnchor
from latentmol.data import collate_graphs
from latentmol.evaluate import load_model, reconstruction_metrics
from latentmol.neighbor_training import NeighborEnvironmentGuidance
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.sampling import sample_hard_graph
from latentmol.train import seed_everything, state_digest
from latentmol.research_profile import RESEARCH_PROFILES
from latentmol.experiments import source_snapshot


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def quality_warnings(initial, observation, panel_count):
    """A deteriorating arm is useful pilot evidence, even with finite gradients."""
    warnings = []
    if observation['panels'][0]['argmax']['raw_valid'] < initial['panels'][0]['argmax']['raw_valid'] - .10 * panel_count:
        warnings.append('Raw prior validity dropped by more than 10 percentage points')
    if observation['reconstruction']['exact_reconstruction'] < initial['reconstruction']['exact_reconstruction'] - .05:
        warnings.append('Exact reconstruction dropped by more than 5 percentage points')
    return warnings


def record_quality_observation(initial, observation, panel_count, *, research_profile, folder, step):
    if research_profile not in RESEARCH_PROFILES:
        raise ValueError('Unknown research profile: ' + str(research_profile))
    warnings = quality_warnings(initial, observation, panel_count)
    if warnings:
        record = dict(local_update=step, research_profile=research_profile, warnings=warnings)
        with (folder / 'quality-warnings.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if research_profile == 'qualified':
            raise RuntimeError('Local pilot safety stop: large validity/reconstruction deterioration')
    return warnings


def cohort(value, name, spec):
    source = value["cohorts"][name]
    arrays = {}
    for filename, record in source["arrays"].items():
        data = base64.b64decode(record["base64"], validate=True)
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
        arrays[filename] = np.frombuffer(data, dtype=record["dtype"]).reshape(record["shape"])
    rows = []
    for index, smiles in enumerate(source["smiles"]):
        encoded = encode_smiles(smiles, spec)
        for key, filename in (("atom_types", "atom_types.npy"), ("charges", "charges.npy"), ("bond_types", "bond_types.npy")):
            if not np.array_equal(encoded[key], arrays[filename][index]):
                raise ValueError("Transferred source graph disagrees with original canonical SMILES")
        rows.append(dict(smiles=smiles, atoms=torch.tensor(arrays["atom_types.npy"][index], dtype=torch.long),
                         charges=torch.tensor(arrays["charges.npy"][index], dtype=torch.long),
                         bonds=torch.tensor(arrays["bond_types.npy"][index], dtype=torch.long)))
    return rows


def summarize_graphs(hard, spec):
    records, identities = [], []
    reasons = Counter()
    for a, q, b in zip(hard.atoms, hard.charges, hard.bonds):
        record = classify_repair_graph(a, q, b, spec)
        records.append(record)
        reasons[record["reason"] or "valid"] += 1
        if record["raw_chemical_valid"]:
            identities.append(record["smiles"])
    active = hard.atoms != 0
    sulfur = hard.atoms == spec.atomic_numbers.index(16)
    assessments = assess_environment(hard.atoms.tolist(), hard.charges.tolist(), hard.bonds.tolist(), spec)
    return dict(attempts=len(records), raw_valid=sum(r["raw_chemical_valid"] is True for r in records),
                strict_valid=sum(r["strict_teacher_valid"] is True for r in records),
                representation_failures=sum(r["failure_kind"] == "representation" for r in records),
                failure_reasons=dict(reasons), unique_valid=len(set(identities)),
                hard_sulfur_fraction=float(sulfur.sum() / active.sum().clamp_min(1)),
                graphs_with_three_or_more_sulfur=int((sulfur.sum(-1) >= 3).sum()),
                mean_hard_size=float(active.sum(-1).float().mean()),
                nonempty=int(active.any(-1).sum()),
                local_valence_violations=sum(not r["local_valence_ok"] for r in assessments),
                scope_exclusions=sum(not r["within_common_organic_scope"] for r in assessments),
                unassessed_graphs=sum(not r["fully_assessed"] for r in assessments),
                records=records)


@torch.no_grad()
def prior_panel(model, folder, name, *, seed, count, batch_size, draws):
    """Raw argmax/categorical outputs, with every invalid graph retained."""
    spec = model.spec
    device = next(model.parameters()).device
    z = torch.randn(count, model.config.latent, generator=torch.Generator().manual_seed(seed))
    generator = torch.Generator().manual_seed(seed + 1000)
    groups = {key: [] for key in ("argmax", "categorical")}
    expected_elements = torch.zeros(len(spec.atomic_numbers), dtype=torch.double)
    atom_entropy = 0.
    for start in range(0, count, batch_size):
        decoded = model.decode(z[start:start + batch_size].to(device))
        probs = decoded.atoms.softmax(-1).cpu().double()
        expected_elements += probs.sum((0, 1))
        atom_entropy += float(-(probs * probs.clamp_min(1e-30).log()).sum())
        detached = DecoderOutput(*(x.detach().cpu() for x in (decoded.atoms, decoded.charges, decoded.bonds)))
        groups["argmax"].append(detached.harden(spec))
        for _ in range(draws):
            groups["categorical"].append(sample_hard_graph(detached, spec, generator=generator))
    out = dict(seed=seed, latent_sha256=hashlib.sha256(z.numpy().tobytes()).hexdigest(),
               expected_sulfur_fraction=float(expected_elements[spec.atomic_numbers.index(16)] / expected_elements[1:].sum()),
               expected_size=float(expected_elements[1:].sum() / count),
               atom_slot_entropy_nats=atom_entropy / count / spec.max_nodes)
    for mode, batches in groups.items():
        if not batches:
            continue
        hard = HardGraphBatch(*(torch.cat([getattr(b, field) for b in batches]) for field in ("atoms", "charges", "bonds")))
        result = summarize_graphs(hard, spec)
        np.savez_compressed(folder / (name + "-" + mode + ".npz"),
                            **{k: getattr(hard, k).numpy().astype(np.int8) for k in ("atoms", "charges", "bonds")})
        write(folder / (name + "-" + mode + "-classifications.json"), result.pop("records"))
        out[mode] = result
    return out


def evaluate(model, rows, config, folder, update, args, *, final=False):
    # Dedicated evaluation generators must not move either training RNG stream.
    cuda_rng = torch.cuda.get_rng_state()
    cpu_rng = torch.get_rng_state()
    mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            prefix = "eval-%04d" % update
            panels = [prior_panel(model, folder, prefix + "-fixed", seed=148201,
                                  count=args.panel_count, batch_size=args.microbatch, draws=4 if final else 0)]
            if final:
                panels.append(prior_panel(model, folder, prefix + "-heldout", seed=148202,
                                          count=args.panel_count, batch_size=args.microbatch, draws=4))
            loader = DataLoader(rows, batch_size=args.microbatch, shuffle=False, collate_fn=collate_graphs)
            reconstruction = reconstruction_metrics(model, loader, config.loss, "cuda", len(rows), training_mode="vae")[0]
            result = dict(update=update, panels=panels, reconstruction=reconstruction,
                          scope="Local paired panels; no full-test, synthesis or binding claim")
            write(folder / (prefix + ".json"), result)
            return result
    finally:
        model.train(mode)
        torch.cuda.set_rng_state(cuda_rng)
        torch.set_rng_state(cpu_rng)


def run_arm(name, args, inputs, config, rows, protocol, composition):
    folder = args.output / name
    folder.mkdir()
    # Start with CPU loading so saved HPC RNG tensors are not treated as GPU state.
    model, checkpoint = load_model(args.inputs / "epoch42.pt", "cpu")
    model = model.to("cuda").train()
    reference, _ = load_model(args.inputs / "epoch35.pt", "cpu")
    reference = reference.to("cuda").eval().requires_grad_(False)
    source36, _ = load_model(args.inputs / "epoch36.pt", "cpu")
    anchor = FrozenValidReferenceAnchor(source36.to("cuda"), reference_epoch=36,
                                     checkpoint_sha256=inputs["checkpoints"]["epoch36"]["sha256"])
    del source36
    object.__setattr__(model, "_valid_reference_anchor", anchor)
    reference_sha = state_digest(reference.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_updates = checkpoint["continuation"]["updates"]
    before_sha = state_digest(model.state_dict())
    del checkpoint
    seed_everything(148102)
    generators = {"fresh_prior": torch.Generator().manual_seed(148103)}
    order = np.random.default_rng(148104)
    guidance = (NeighborEnvironmentGuidance(model.spec, dict(version="common-organic-neighbor-v1", weight=args.weight))
                if name == "neighbor" else None)
    torch.cuda.reset_peak_memory_stats()
    initial = evaluate(model, rows["val"], config, folder, 0, args, final=True)
    history, training_seconds, warnings = [], 0., []
    teacher = AtomPreservingRepairTeacher(model.spec, ProbabilityRepairConfig(**protocol["search"]),
                backend="optimized", workers=0, classification_version="repair-chemistry-classification-v1")
    with teacher:
        for step in range(1, args.updates + 1):
            ids = order.choice(len(rows["train"]), args.real_batch, replace=False).tolist()
            hard = collate_graphs([rows["train"][i] for i in ids])
            prior_before = hashlib.sha256(generators["fresh_prior"].get_state().numpy().tobytes()).hexdigest()
            posterior_before = hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
            torch.cuda.synchronize()
            started = time.perf_counter()
            metric = chemistry_update(model, reference, optimizer, hard, config, protocol, composition, teacher,
                       generators, start_updates + step, 43, 35, microbatch_size=args.microbatch,
                       overlap=False, step_control=dict(version="adamw-displacement-scale-v1", factor=.25),
                       additive_guidance=guidance)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            training_seconds += elapsed
            record = dict(local_update=step, source_indices=[inputs["cohorts"]["train"]["source_indices"][i] for i in ids],
                          fresh_prior_rng_before=prior_before, posterior_rng_before=posterior_before,
                          update_seconds=elapsed, metrics=metric)
            with (folder / "updates.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            if step in (1, 2) or step % 10 == 0:
                print(json.dumps(dict(arm=name, update=step, seconds=round(elapsed, 3), total=metric["total"],
                    added_loss=metric.get("additive_guidance", {}).get("weighted_loss", 0.))), flush=True)
            if step in (10, 25, 50, 100, args.updates):
                observation = evaluate(model, rows["val"], config, folder, step, args, final=step == args.updates)
                history.append(observation)
                warnings.extend(record_quality_observation(initial, observation, args.panel_count,
                    research_profile=args.research_profile, folder=folder, step=step))
    anchor.assert_unchanged()
    assert state_digest(reference.state_dict()) == reference_sha
    result = dict(status="completed_local_pilot_not_production_adoption", arm=name, completed_updates=args.updates,
                  source_epoch=42, source_checkpoint_sha256=inputs["checkpoints"]["epoch42"]["sha256"],
                  initial_model_sha256=before_sha, final_model_sha256=state_digest(model.state_dict()),
                  training_seconds=training_seconds, mean_update_seconds=training_seconds / args.updates,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  initial=initial, final=history[-1], frozen_anchors_unchanged=True,
                  research_profile=args.research_profile, quality_warnings=warnings)
    # Diagnostic checkpoint is deliberately unsuitable for a production resume.
    torch.save(dict(diagnostic_only=True, promotable=False, model_state=model.state_dict(),
                    model_config=model.config.__dict__, graph_spec=model.spec.to_dict(), optimizer_state=optimizer.state_dict()),
               folder / "diagnostic-only.pt")
    write(folder / "result.json", result)
    del model, reference, anchor, optimizer
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--real-batch", type=int, default=16)
    p.add_argument("--microbatch", type=int, default=8)
    p.add_argument("--panel-count", type=int, default=1024)
    p.add_argument("--weight", type=float, default=.1)
    p.add_argument("--arm", choices=("control", "neighbor", "both"), default="both")
    p.add_argument("--research-profile", choices=RESEARCH_PROFILES, default="exploratory",
                   help="Exploratory records quality regressions and completes the bounded pilot; qualified preserves the old quality stop.")
    args = p.parse_args()
    if any(v < 1 for v in (args.updates, args.real_batch, args.microbatch, args.panel_count)):
        p.error("Positive explicit pilot budgets required")
    if not torch.cuda.is_available():
        raise RuntimeError("This declared local pilot requires the laptop GPU")
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = json.loads((args.inputs / "inputs.json").read_text())
    for name, asset in inputs["checkpoints"].items():
        assert sha(args.inputs / (name + ".pt")) == asset["sha256"]
    spec = GraphSpec.from_dict(inputs["manifest"]["spec"])
    rows = {name: cohort(inputs, name, spec) for name in ("train", "val")}
    assert not set(inputs["cohorts"]["train"]["source_indices"]) & set(inputs["cohorts"]["val"]["source_indices"])
    config_path = args.output / "parent-config.json"
    config_path.write_text(json.dumps(inputs["files"]["config"]["value"]))
    config = replace(load_parent_config(config_path), device="cuda")
    protocol = deepcopy(inputs["files"]["plan"]["value"]["protocol"])
    protocol.update(real_batch_size=args.real_batch, microbatch_size=args.microbatch)
    composition = inputs["files"]["composition"]["value"]
    torch.set_num_threads(4)
    environment = dict(python=__import__("sys").version, packages={n: importlib.metadata.version(n) for n in ("torch", "rdkit", "numpy", "pytest", "PyYAML")},
                       gpu=torch.cuda.get_device_name(), gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
                       cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"])
    source_root = Path(__file__).resolve().parents[1]
    if args.research_profile == 'qualified':
        source = {str(f.relative_to(source_root)): sha(f)
                  for d in ('src/latentmol', 'scripts') for f in (source_root / d).rglob('*.py')}
    else:
        source = source_snapshot(source_root, selected_files=(Path(__file__),
            source_root / 'src/latentmol/guidance/neighbor.py',
            source_root / 'src/latentmol/chemistry_continuation.py'))
    write(args.output / "manifest.json", dict(environment=environment, source=source, inputs_sha256=sha(args.inputs / "inputs.json"),
          config=config.__dict__ | {"model": config.model.__dict__, "loss": config.loss.__dict__}, protocol=protocol,
          pilot=dict(updates=args.updates, real_batch=args.real_batch, microbatch=args.microbatch, panel_count=args.panel_count, weight=args.weight,
                     research_profile=args.research_profile),
          local_authorization="User requested local environment, checks and small matched VAE training pilot,30Oct2026",
          comparison="Same checkpoint42/AdamW/anchors/train rows/noise seeds; 1:2 prior ratio; new local streams and smaller batch, not exact HPC resume",
          hpc_mutations=False, production_adoption=False))
    try:
        names = ("control", "neighbor") if args.arm == "both" else (args.arm,)
        results = {name: run_arm(name, args, inputs, config, rows, protocol, composition) for name in names}
        if len(results) == 2:
            logs = [[json.loads(line) for line in (args.output / name / "updates.jsonl").read_text().splitlines()] for name in names]
            for a, b in zip(*logs):
                assert all(a[k] == b[k] for k in ("source_indices", "fresh_prior_rng_before", "posterior_rng_before"))
            assert results["control"]["initial_model_sha256"] == results["neighbor"]["initial_model_sha256"]
        write(args.output / "result.json", dict(status="completed_local_comparison_not_production_adoption", arms=results,
              matched_real_rows_and_noise_streams=True if len(results) == 2 else None, hpc_mutations=False))
    except BaseException as error:
        write(args.output / "failed.json", dict(status="failed_local_pilot", error=type(error).__name__ + ": " + str(error),
                                               hpc_mutations=False, production_adoption=False))
        raise


if __name__ == "__main__":
    main()
