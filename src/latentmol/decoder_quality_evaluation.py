"""Fixed latent-panel decoder evaluation with complete hard-draw denominators.

This module is an evaluation boundary: RDKit inspection, bounded repair search
and hard component counting are outside autograd. It never replaces the shared
raw graph by a repaired witness. All invalid draws and duplicates remain in the
records and rate denominators. Training novelty uses only the supplied training
SMILES set, not validation/test molecules.

Sampling proceeds by latent ID after decoding, with one dedicated CPU generator.
Changing decode batch size therefore does not change RNG consumption; numerical
differences in model batch kernels can still change categorical probabilities.
Paired uncertainty resamples latent IDs, keeping each latent's complete draw
cluster together. A validity gain alone does not establish an overall decoder
advantage: reconstruction retention is a separate controller gate.
"""

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import rdBase

from latentmol.chemistry import inspect_graph
from latentmol.chemistry_regularization import ChemistryRegularizationConfig, chemistry_regularization_losses
from latentmol.connectivity_regularization import component_costs
from latentmol.experiments import file_sha256, write_json
from latentmol.probability_repair import ProbabilityRepairConfig, search_probability_repair
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.sampling import sample_hard_graph


PANEL_VERSION = "decoder-prior-quality-v1"
PROTOCOL_KEYS = ("version", "latent_sha256", "latent_shape", "schema", "schema_sha256",
                 "samples_per_latent", "sample_seed", "draw_policy", "temperature",
                 "training_smiles_sha256", "training_smiles_count", "repair_latent_ids", "repair_config")


def _json_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256(json.dumps({"dtype": str(tensor.dtype), "shape": list(tensor.shape)}, sort_keys=True).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _model_sha(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(json.dumps({"name": name, "tensor_sha256": _tensor_sha(value)}, sort_keys=True).encode())
    return digest.hexdigest()


def _fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def _row(graph: HardGraphBatch, spec: GraphSpec, latent_id: int, kind: str, draw: int | None,
         training_smiles: set[str], relaxed_risk: dict) -> dict:
    atoms, charges, bonds = (value[0].cpu().numpy() for value in (graph.atoms, graph.charges, graph.bonds))
    inspected = inspect_graph(atoms, charges, bonds, spec)
    components = component_costs(graph)
    active = atoms != 0
    element_counts = Counter(str(spec.atomic_numbers[int(label)]) for label in atoms[active])
    charge_counts = Counter(str(spec.formal_charges[int(label)]) for label in charges[active])
    pair = np.triu(active[:, None] & active[None, :], k=1)
    bond_counts = Counter(spec.bond_names[int(label)] for label in bonds[pair])
    aromatic_label = spec.bond_names.index("AROMATIC") if "AROMATIC" in spec.bond_names else None
    aromatic_edges = int(np.count_nonzero((bonds == aromatic_label) & pair)) if aromatic_label is not None else 0
    valid = inspected.smiles is not None
    return {"latent_id": latent_id, "kind": kind, "draw": draw,
            "valid": valid, "smiles": inspected.smiles,
            "inspection": asdict(inspected), "atom_count": int(active.sum()),
            "component_count": int(components["components"][0]),
            "connected_nonempty": bool(active.any() and components["components"][0] == 1),
            "unrestricted_added_edges": int(components["minimum_unrestricted_added_edges"][0]),
            "novel_vs_training": bool(valid and inspected.smiles not in training_smiles) if valid else None,
            "element_counts": dict(element_counts), "charge_counts": dict(charge_counts),
            "bond_counts": dict(bond_counts), "aromatic_edges": aromatic_edges,
            "relaxed_rule_risk": relaxed_risk,
            "labels": {"atoms": atoms.tolist(), "charges": charges.tolist(), "bonds": bonds.tolist()}}


def _summarize_rows(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["valid"]]
    unique = {row["smiles"] for row in valid}
    novel = [row for row in valid if row["novel_vs_training"]]
    histograms = {name: Counter() for name in ("atom_count", "component_count", "element", "charge", "bond", "aromatic_edges", "failure_reason")}
    for row in rows:
        for name in ("atom_count", "component_count", "aromatic_edges"):
            histograms[name][str(row[name])] += 1
        for name, field in (("element", "element_counts"), ("charge", "charge_counts"), ("bond", "bond_counts")):
            histograms[name].update(row[field])
        if not row["valid"]:
            histograms["failure_reason"][row["inspection"]["failure_reason"] or "unclassified"] += 1
    return {"attempted": len(rows), "valid_count": len(valid), "validity": _fraction(len(valid), len(rows)),
            "invalid_count": len(rows) - len(valid), "unique_valid_count": len(unique),
            "unique_valid_fraction": _fraction(len(unique), len(valid)),
            "unique_valid_per_attempt": _fraction(len(unique), len(rows)),
            "duplicate_valid_count": len(valid) - len(unique),
            "novel_valid_vs_training_count": len(novel),
            "novel_valid_vs_training_fraction": _fraction(len(novel), len(valid)),
            "unique_novel_valid_count": len({row["smiles"] for row in novel}),
            "connected_nonempty_count": sum(row["connected_nonempty"] for row in rows),
            "connected_nonempty_fraction": _fraction(sum(row["connected_nonempty"] for row in rows), len(rows)),
            "mean_atom_count": float(np.mean([row["atom_count"] for row in rows])) if rows else None,
            "histograms": {name: dict(sorted(values.items())) for name, values in histograms.items()}}


@torch.no_grad()
def evaluate_random_panel(model, z: torch.Tensor, training_smiles: set[str], output_dir: Path, *,
                          batch_size: int = 64, samples_per_latent: int = 4, sample_seed: int = 1707,
                          repair_limit: int = 64, repair_config: ProbabilityRepairConfig | None = None) -> dict:
    """Evaluate one immutable CPU latent panel with fixed hard-draw/search quotas.

    output_dir must be new. The caller supplies prior latents and experiment
    source/checkpoint identities; this receipt also fingerprints actual model
    tensors, latent bytes, schema, training set and every saved artifact. Model
    training flags and tensors are verified/restored without changing weights.
    """
    for name, value, minimum in (("batch_size", batch_size, 1), ("samples_per_latent", samples_per_latent, 1),
                                 ("repair_limit", repair_limit, 0), ("sample_seed", sample_seed, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(z, torch.Tensor) or z.device.type != "cpu" or z.ndim != 2 or min(z.shape) < 1 or not z.is_floating_point() or not bool(torch.isfinite(z).all()):
        raise ValueError("Latent panel must be a finite nonempty floating CPU [latent_id,dimension] tensor")
    if not isinstance(training_smiles, (set, frozenset)) or any(not isinstance(value, str) or not value for value in training_smiles):
        raise ValueError("Training SMILES must be a set of nonempty canonical strings")
    if not isinstance(model.spec, GraphSpec):
        raise ValueError("Model must expose its shared GraphSpec")
    if hasattr(model, "config") and hasattr(model.config, "latent") and model.config.latent != z.shape[1]:
        raise ValueError("Latent panel dimension differs from the decoder configuration")
    repair_config = ProbabilityRepairConfig() if repair_config is None else repair_config
    if not isinstance(repair_config, ProbabilityRepairConfig):
        raise ValueError("Expected a ProbabilityRepairConfig")
    spec, count = model.spec, len(z)
    protocol = {"version": PANEL_VERSION, "latent_sha256": _tensor_sha(z), "latent_shape": list(z.shape),
                "schema": spec.to_dict(), "schema_sha256": spec.fingerprint(),
                "samples_per_latent": samples_per_latent, "sample_seed": sample_seed,
                "draw_policy": "argmax plus independent factorized CPU draws in latent-ID order; invalids and duplicates retained",
                "temperature": 1.0, "training_smiles_sha256": _json_sha(sorted(training_smiles)),
                "training_smiles_count": len(training_smiles),
                "repair_latent_ids": list(range(min(repair_limit, count))), "repair_config": repair_config.policy()}
    panel_sha = _json_sha(protocol)
    model_sha = _model_sha(model)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {**protocol, "panel_sha256": panel_sha, "model_state_sha256": model_sha,
                "batch_size": batch_size, "rdkit_version": rdBase.rdkitVersion,
                "model_batch_numerics": "RNG consumption independent of batch size; model-kernel roundoff can change probabilities",
                "status": "running"}
    write_json(directory / "panel.json", manifest)
    generator = torch.Generator(device="cpu").manual_seed(sample_seed)
    first_parameter = next(iter(model.parameters()), None)
    first_buffer = next(iter(model.buffers()), None)
    location = first_parameter if first_parameter is not None else first_buffer
    device = location.device if location is not None else torch.device("cpu")
    dtype = location.dtype if location is not None and location.is_floating_point() else z.dtype
    training_flags = [(module, module.training) for module in model.modules()]
    rows, repairs, per_latent = [], [], []
    argmax_labels = {name: [] for name in ("atoms", "charges", "bonds")}
    sampled_labels = {name: [] for name in ("atoms", "charges", "bonds")}
    config = ChemistryRegularizationConfig(isolated_weight=0)
    scope = None
    try:
        model.eval()
        with (directory / "rows.jsonl").open("x", encoding="utf-8") as rows_file, (directory / "repairs.jsonl").open("x", encoding="utf-8") as repair_file:
            for start in range(0, count, batch_size):
                decoded_native = model.decode(z[start:start + batch_size].to(device=device, dtype=dtype))
                if not isinstance(decoded_native, DecoderOutput):
                    raise ValueError("Model decode must return DecoderOutput")
                decoded = DecoderOutput(*(value.detach().cpu() for value in
                                          (decoded_native.atoms, decoded_native.charges, decoded_native.bonds)))
                # This computes the distributional local proxy at fixed z. It is
                # not a binary validity verdict for any particular hard draw.
                regularization = chemistry_regularization_losses(decoded, spec, config)
                scope = regularization["scope"]
                argmax = decoded.harden(spec)
                probabilities = (decoded.atoms.softmax(-1), decoded.charges.softmax(-1),
                                 decoded.symmetric_bond_logits.softmax(-1))
                for offset in range(len(decoded.atoms)):
                    latent_id = start + offset
                    relaxed_risk = {name: float(value[offset]) for name, value in regularization["per_graph"].items()}
                    relaxed_diagnostics = {name: value[offset].tolist()
                        for name, value in regularization["diagnostics"].items() if not name.startswith("dp_")}
                    one_output = DecoderOutput(*(value[offset:offset+1] for value in
                                                 (decoded.atoms, decoded.charges, decoded.bonds)))
                    raw = HardGraphBatch(*(value[offset:offset+1] for value in
                                           (argmax.atoms, argmax.charges, argmax.bonds)))
                    raw_row = _row(raw, spec, latent_id, "argmax", None, training_smiles, relaxed_risk)
                    cluster = [raw_row]
                    for name in argmax_labels:
                        argmax_labels[name].append(raw_row["labels"][name])
                    cluster_labels = {name: [] for name in sampled_labels}
                    for draw in range(samples_per_latent):
                        graph = sample_hard_graph(one_output, spec, generator=generator, temperature=1.0)
                        sampled = _row(graph, spec, latent_id, "sample", draw, training_smiles, relaxed_risk)
                        cluster.append(sampled)
                        for name in cluster_labels:
                            cluster_labels[name].append(sampled["labels"][name])
                    for name in sampled_labels:
                        sampled_labels[name].append(cluster_labels[name])
                    for record in cluster:
                        rows_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    rows_file.flush()
                    rows.extend(cluster)
                    per_latent.append({"latent_id": latent_id, "argmax_valid": int(raw_row["valid"]),
                                       "sample_valid_count": sum(record["valid"] for record in cluster[1:]),
                                       "sample_attempted": samples_per_latent,
                                       "argmax_atom_count": raw_row["atom_count"],
                                       "sample_mean_atom_count": float(np.mean([record["atom_count"] for record in cluster[1:]])),
                                       "relaxed_rule_risk": relaxed_risk,
                                       "relaxed_diagnostics": relaxed_diagnostics})
                    if latent_id < min(repair_limit, count):
                        receipt = search_probability_repair(tuple(value[0] for value in (raw.atoms, raw.charges, raw.bonds)),
                            tuple(value[offset] for value in probabilities), spec, repair_config)
                        receipt = {"latent_id": latent_id, "candidate_kind": "argmax", **receipt,
                                   "selection_policy": "Predeclared initial latent IDs, independent of raw validity or outcome"}
                        repairs.append(receipt)
                        repair_file.write(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
                        repair_file.flush()
        if _model_sha(model) != model_sha:
            raise RuntimeError("Evaluation modified the model tensors")
        np.savez_compressed(directory / "labels.npz",
            **{f"argmax_{name}": np.asarray(values, dtype=np.int16) for name, values in argmax_labels.items()},
            **{f"sample_{name}": np.asarray(values, dtype=np.int16) for name, values in sampled_labels.items()})
        raw_rows = [record for record in rows if record["kind"] == "argmax"]
        sample_rows = [record for record in rows if record["kind"] == "sample"]
        costs = [record["best_observed_weighted_cost"] for record in repairs if record["best_observed_weighted_cost"] is not None]
        repair_summary = {"attempted": len(repairs), "latent_ids": protocol["repair_latent_ids"],
                          "status_counts": dict(Counter(record["status"] for record in repairs)),
                          "checks": sum(record["checks"] for record in repairs),
                          "capped_depths": sum(depth["capped"] for record in repairs for depth in record["checks_by_depth"]),
                          "verified_witness_or_already_valid_count": len(costs),
                          "mean_best_observed_weighted_cost_conditional_on_witness": float(np.mean(costs)) if costs else None,
                          "distance_is_globally_proven": False, "no_repair_substitution": True}
        metrics = {"version": PANEL_VERSION, "status": "completed", "panel_sha256": panel_sha,
                   "model_state_sha256": model_sha, "latent_count": count,
                   "samples_per_latent": samples_per_latent, "argmax": _summarize_rows(raw_rows),
                   "sample": _summarize_rows(sample_rows), "all_attempts": _summarize_rows(rows),
                   "repair": repair_summary, "per_latent": per_latent,
                   "relaxed_rule_risk_means": {name: float(np.mean([record["relaxed_rule_risk"][name] for record in per_latent]))
                                               for name in per_latent[0]["relaxed_rule_risk"]},
                   "relaxed_diagnostic_means": {name: np.asarray([record["relaxed_diagnostics"][name] for record in per_latent]).mean(0).tolist()
                                                for name in per_latent[0]["relaxed_diagnostics"]},
                   "rule_scope": scope, "novelty_scope": "Supplied training split only",
                   "validity_scope": "Nonempty connected radical-free RDKit-sanitizable raw graph; no synthesis or binding claim",
                   "reconstruction_retention": "Assessed separately by controller",
                   "advantage_established": False,
                   "artifacts": {name: {"path": name, "sha256": file_sha256(directory / name)}
                                 for name in ("rows.jsonl", "repairs.jsonl", "labels.npz")}}
        write_json(directory / "metrics.json", metrics)
        write_json(directory / "panel.json", {**manifest, "status": "completed", "metrics_sha256": file_sha256(directory / "metrics.json")})
        return metrics
    except BaseException as error:
        write_json(directory / "panel.json", {**manifest, "status": "failed", "error_type": type(error).__name__,
                                               "error": str(error), "retained_hard_attempts": len(rows), "retained_repair_attempts": len(repairs)})
        raise
    finally:
        for module, training in training_flags:
            module.training = training


def _cluster_interval(delta: np.ndarray, generator: np.random.Generator, draws: int) -> dict:
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError("Bootstrap requires finite per-latent deltas")
    values = []
    for start in range(0, draws, 128):
        indices = generator.integers(0, len(delta), size=(min(128, draws - start), len(delta)))
        values.append(delta[indices].mean(-1))
    bootstrap = np.concatenate(values)
    return {"mean_difference_candidate_minus_reference": float(delta.mean()),
            "ci95": np.quantile(bootstrap, (.025, .975)).tolist(), "latent_clusters": len(delta),
            "bootstrap_draws": draws, "method": "Paired latent-ID cluster percentile bootstrap"}


def paired_compare(reference_dir: Path, candidate_dir: Path, *, bootstrap_seed: int = 2907,
                   bootstrap_draws: int = 2000) -> dict:
    """Pair complete latent clusters; keep validity and retention gates separate."""
    if type(bootstrap_seed) is not int or bootstrap_seed < 0 or type(bootstrap_draws) is not int or bootstrap_draws < 1:
        raise ValueError("Bootstrap seed/draw budget must be nonnegative/positive integers")
    reference_dir, candidate_dir = Path(reference_dir), Path(candidate_dir)
    manifests = [json.loads((folder / "panel.json").read_text(encoding="utf-8")) for folder in (reference_dir, candidate_dir)]
    results = [json.loads((folder / "metrics.json").read_text(encoding="utf-8")) for folder in (reference_dir, candidate_dir)]
    if any(record.get("status") != "completed" or record.get("version") != PANEL_VERSION for record in manifests + results):
        raise ValueError("Paired comparison requires completed compatible panel receipts")
    if manifests[0]["panel_sha256"] != manifests[1]["panel_sha256"]:
        raise ValueError("Paired latent/schema/train/draw/repair protocol identities differ")
    for folder, manifest, result in zip((reference_dir, candidate_dir), manifests, results):
        if _json_sha({name: manifest[name] for name in PROTOCOL_KEYS}) != manifest["panel_sha256"]:
            raise ValueError("Panel protocol content does not match its declared identity")
        if result["panel_sha256"] != manifest["panel_sha256"] or file_sha256(folder / "metrics.json") != manifest["metrics_sha256"]:
            raise ValueError("Panel metrics identity differs from its completion receipt")
        if result["model_state_sha256"] != manifest["model_state_sha256"]:
            raise ValueError("Panel model state identity differs from its metrics")
        if result["latent_count"] != manifest["latent_shape"][0] or result["samples_per_latent"] != manifest["samples_per_latent"]:
            raise ValueError("Panel metrics have an inconsistent latent/sample quota")
        if result["argmax"]["attempted"] != result["latent_count"] or result["sample"]["attempted"] != result["latent_count"] * result["samples_per_latent"]:
            raise ValueError("Panel metrics lost a hard-draw denominator")
        for artifact in result["artifacts"].values():
            if file_sha256(folder / artifact["path"]) != artifact["sha256"]:
                raise ValueError("Saved hard-draw or repair artifact hash mismatch")
    ref, cand = results
    ref_rows, cand_rows = ref["per_latent"], cand["per_latent"]
    if [record["latent_id"] for record in ref_rows] != list(range(ref["latent_count"])) or [record["latent_id"] for record in cand_rows] != list(range(ref["latent_count"])):
        raise ValueError("Paired per-latent records must preserve the complete ordered panel")
    if any(record["sample_attempted"] != ref["samples_per_latent"] for record in ref_rows + cand_rows):
        raise ValueError("A latent cluster lost its declared sample quota")
    generator = np.random.default_rng(bootstrap_seed)
    comparisons = {}
    for kind in ("argmax", "sample"):
        left = np.asarray([record["argmax_valid"] if kind == "argmax" else record["sample_valid_count"] / record["sample_attempted"] for record in ref_rows], dtype=np.float64)
        right = np.asarray([record["argmax_valid"] if kind == "argmax" else record["sample_valid_count"] / record["sample_attempted"] for record in cand_rows], dtype=np.float64)
        comparison = _cluster_interval(right - left, generator, bootstrap_draws)
        comparison.update(reference_valid_count=ref[kind]["valid_count"], candidate_valid_count=cand[kind]["valid_count"],
                          reference_attempted=ref[kind]["attempted"], candidate_attempted=cand[kind]["attempted"],
                          ci_positive=bool(comparison["ci95"][0] > 0),
                          unique_valid_count_change=cand[kind]["unique_valid_count"] - ref[kind]["unique_valid_count"],
                          uniqueness_per_attempt_no_decrease=bool(cand[kind]["unique_valid_per_attempt"] >= ref[kind]["unique_valid_per_attempt"]))
        comparisons[kind] = comparison
    size_delta = np.asarray([right["argmax_atom_count"] - left["argmax_atom_count"] for left, right in zip(ref_rows, cand_rows)], dtype=np.float64)
    rule_delta = {name: float(np.mean([right["relaxed_rule_risk"][name] - left["relaxed_rule_risk"][name]
                                     for left, right in zip(ref_rows, cand_rows)]))
                  for name in ref_rows[0]["relaxed_rule_risk"]}
    diagnostic_delta = {name: (np.asarray([record["relaxed_diagnostics"][name] for record in cand_rows]).mean(0)
                               - np.asarray([record["relaxed_diagnostics"][name] for record in ref_rows]).mean(0)).tolist()
                        for name in ref_rows[0]["relaxed_diagnostics"]}
    validity_positive = all(comparisons[kind]["ci_positive"] for kind in ("argmax", "sample"))
    uniqueness_retained = all(comparisons[kind]["uniqueness_per_attempt_no_decrease"] for kind in ("argmax", "sample"))
    comparison = {"version": "decoder-paired-quality-v1", "panel_sha256": ref["panel_sha256"],
                  "reference_model_state_sha256": ref["model_state_sha256"], "candidate_model_state_sha256": cand["model_state_sha256"],
                  "bootstrap_seed": bootstrap_seed, "comparisons": comparisons,
                  "argmax_atom_count_delta": _cluster_interval(size_delta, generator, bootstrap_draws),
                  "sample_mean_atom_count_delta": float(np.mean([right["sample_mean_atom_count"] - left["sample_mean_atom_count"] for left, right in zip(ref_rows, cand_rows)])),
                  "relaxed_rule_risk_deltas": rule_delta,
                  "relaxed_diagnostic_deltas": diagnostic_delta,
                  "decision": {"advantage_established": False,
                               "validity_positive_ci_both_modes": validity_positive,
                               "unique_valid_per_attempt_retained_both_modes": uniqueness_retained,
                               "reconstruction_retention": "not assessed here; controller gate required",
                               "status": "pending_reconstruction_retention" if validity_positive and uniqueness_retained else "no_established_advantage"},
                  "scope": "Paired decoder-generation engineering evidence only; validity/proxy gains do not establish binding, synthesis or exact repair distance"}
    receipt = candidate_dir / f"paired_comparison-{ref['model_state_sha256'][:12]}.json"
    comparison["receipt_path"] = str(receipt)
    write_json(receipt, comparison)
    return comparison
