"""Raw, unrepaired VAE diagnostics shared by historical and new checkpoints.

No scorer or chemical prior is trained here. Invalid/duplicate/empty draws stay
in the denominator. Categorical sampling uses conditional bonds followed by
hard occupancy masking, exactly as in the existing shared graph sampler.
UNTESTED on HPC: intended for the source-release engineering gate first.
"""

from collections import Counter
import json
import gzip
from pathlib import Path

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
import torch
from torch.utils.data import DataLoader, Subset

from latentmol.chemistry import inspect_graph
from latentmol.data import collate_graphs
from latentmol.evaluate import reconstruction_metrics
from latentmol.losses import LossConfig
from latentmol.representation import DecoderOutput
from latentmol.sampling import sample_hard_graph, subset_indices


def checked_molecule(smiles):
    """Parse a reported identity without converting failures into molecules.

    Graph sanitization and serialization roundtrip are separate observations.
    In particular, RDKit can serialize an aromatic graph whose SMILES then
    fails kekulization on parsing. Keep that attempted graph in the denominator
    and report the serialization failure; never repair it for these metrics.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None, "missing_smiles"
    try:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
    except (ValueError, RuntimeError) as error:
        return None, type(error).__name__ + ": " + str(error)
    return (mol, None) if mol is not None else (None, "rdkit_parse_failed")


def checked_fingerprint(generator, record):
    """Use the same parsing eligibility for generated and anchor fingerprints."""
    if not record.get("valid"):
        return None
    mol, _ = checked_molecule(record.get("smiles"))
    return generator.GetFingerprint(mol) if mol is not None else None


def summarize(rows):
    """Keep all attempts; identity metrics require a reparsable raw SMILES.

    The defensive parsing also makes saved historical rows safe to summarize.
    Their original files/flags are never overwritten by this eligibility check.
    """
    n = len(rows)
    eligible = []
    for original in rows:
        row = dict(original)
        mol, parse_error = checked_molecule(row.get("smiles")) if row["valid"] else (None, None)
        if row["valid"] and mol is None:
            row.update(valid=False, failure_reason="smiles_roundtrip",
                       smiles_roundtrip_valid=False, smiles_roundtrip_failure_detail=parse_error)
        elif mol is not None:
            row["rings"] = int(mol.GetRingInfo().NumRings())
        eligible.append(row)
    valid = [row for row in eligible if row["valid"]]
    unique = sorted({row["smiles"] for row in valid})
    atoms, charges, bonds, failures = Counter(), Counter(), Counter(), Counter()
    for row in eligible:
        atoms.update(row["elements"])
        charges.update(row["formal_charges"])
        bonds.update(row["bond_types"])
        if not row["valid"]:
            failures[row["failure_reason"] or "unknown"] += 1
    # Bounded deterministic diversity cohort; does not condition validity yield.
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [checked_fingerprint(generator, {"valid": True, "smiles": value})
                    for value in unique[:256]]
    # This is defensive against a parser failure after initial eligibility.
    # The exact eligible fingerprint denominator is always reported separately.
    fingerprints = [value for value in fingerprints if value is not None]
    similarities = []
    for i, fingerprint in enumerate(fingerprints):
        similarities.extend(DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints[:i]))
    result = {
        "attempts": n, "valid_count": len(valid), "validity": len(valid) / n if n else None,
        "unique_valid": len(unique), "unique_valid_yield": len(unique) / n if n else None,
        "failures": dict(failures), "all_attempt_element_counts": dict(atoms),
        "all_attempt_charge_counts": dict(charges), "all_attempt_bond_counts": dict(bonds),
        "all_attempt_atom_count_mean": sum(r["atoms"] for r in rows) / n if n else None,
        "atom_count_histogram": dict(Counter(str(r["atoms"]) for r in rows)),
        "valid_ring_count_mean": sum(r["rings"] for r in valid) / len(valid) if valid else None,
        "diversity_mean_pairwise_tanimoto": sum(similarities) / len(similarities) if similarities else None,
        "diversity_cohort_policy": "first 256 lexicographically sorted unique valid identities",
        "diversity_cohort_size": len(fingerprints),
        "graph_sanitization_valid_count": sum(row.get("graph_sanitization_valid", row["valid"])
                                               for row in rows),
        "smiles_roundtrip_failures": sum(row.get("failure_reason") == "smiles_roundtrip"
                                           for row in eligible),
        "validity_definition": "raw graph produces a SMILES that RDKit can parse without repair",
        "uncertainty_note": "draws from the same latent are correlated; no pooled binomial interval claimed",
    }
    targeted = [r for r in eligible if "target_smiles" in r]
    if targeted:
        result["exact_reconstruction"] = sum(r["valid"] and r["smiles"] == r["target_smiles"]
                                             for r in targeted) / len(targeted)
    comparison_keys = sorted({key for row in rows for key in row
                              if key.startswith("similarity_to_") or key.startswith("edits_to_")})
    result["anchor_comparisons"] = {
        key: {"eligible": sum(row.get(key) is not None for row in rows),
              "mean": sum(row[key] for row in rows if row.get(key) is not None) /
              sum(row.get(key) is not None for row in rows)
              if any(row.get(key) is not None for row in rows) else None}
        for key in comparison_keys}
    novelty_rows = [r for r in valid if "novel_against_comparison_training" in r]
    if novelty_rows:
        result["novel_against_comparison_training_fraction_of_valid"] = sum(
            r["novel_against_comparison_training"] for r in novelty_rows) / len(novelty_rows)
    return result


def graph_record(hard, row, spec, **identity):
    inspected = inspect_graph(hard.atoms[row], hard.charges[row], hard.bonds[row], spec)
    active = hard.atoms[row] != 0
    atom_labels = hard.atoms[row][active].tolist()
    charges = hard.charges[row][active].tolist()
    pair = active[:, None] & active[None, :] & torch.ones_like(hard.bonds[row], dtype=torch.bool).triu(1)
    bonds = hard.bonds[row][pair].tolist()
    graph_valid = bool(inspected.smiles)
    mol, parse_error = checked_molecule(inspected.smiles) if graph_valid else (None, None)
    valid = graph_valid and mol is not None
    return {
        **identity, "valid": valid, "smiles": inspected.smiles,
        "graph_sanitization_valid": graph_valid, "smiles_roundtrip_valid": valid,
        "smiles_roundtrip_failure_detail": parse_error,
        "failure_reason": "smiles_roundtrip" if graph_valid and not valid else inspected.failure_reason,
        "failure_detail": parse_error if graph_valid and not valid else inspected.detail,
        "atoms": len(atom_labels), "elements": dict(Counter(str(spec.atomic_numbers[i]) for i in atom_labels)),
        "formal_charges": dict(Counter(str(spec.formal_charges[i]) for i in charges)),
        "bond_types": dict(Counter(spec.bond_names[i] for i in bonds if i)),
        "rings": int(mol.GetRingInfo().NumRings()) if mol is not None else None,
        "graph": {"atoms": hard.atoms[row].tolist(), "charges": hard.charges[row].tolist(),
                  "edges": [[int(i), int(j), int(hard.bonds[row, i, j])]
                            for i, j in ((hard.bonds[row] != 0).triu(1)).nonzero().tolist()]},
    }


@torch.no_grad()
def argmax_records(model, codes, batch_size):
    """Use the same bounded decoder batch for anchors and final candidates."""
    rows = []
    for chunk in codes.split(batch_size):
        hard = model.decode(chunk.to(next(model.parameters()).device)).harden(model.spec).to("cpu")
        rows.extend(graph_record(hard, i, model.spec) for i in range(len(chunk)))
    return rows


@torch.no_grad()
def evaluate_codes(model, codes, output, protocol, *, targets=None, training_smiles=None, anchors=None):
    """Decode fixed CPU latents in bounded GPU chunks; preserve every attempted graph."""
    size = protocol["evaluation_batch_size"]
    draws = protocol["categorical_samples_per_latent"]
    generator = torch.Generator(device="cpu").manual_seed(protocol["evaluation_seed"] + 1)
    rows = {"argmax": [], "categorical": []}
    expected_size, element_mass, entropy = 0.0, None, 0.0
    output.parent.mkdir(parents=True, exist_ok=True)
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    anchor_fingerprints = {}
    for label, values in (anchors or {}).items():
        anchor_fingerprints[label] = [checked_fingerprint(fingerprint_generator, r) for r in values]
    with gzip.open(str(output) + ".gz", "xt", encoding="utf-8") as handle:
        for begin in range(0, len(codes), size):
            decoded = model.decode(codes[begin:begin + size].to(next(model.parameters()).device))
            decoded = DecoderOutput(decoded.atoms.cpu(), decoded.charges.cpu(), decoded.bonds.cpu())
            probabilities = decoded.atoms.softmax(-1)
            expected_size += float((1 - probabilities[..., 0]).sum())
            mass = probabilities.sum((0, 1)).double()
            element_mass = mass if element_mass is None else element_mass + mass
            entropy += float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
            for method, hard, draw in [("argmax", decoded.harden(model.spec), -1)] + [
                    ("categorical", sample_hard_graph(decoded, model.spec, generator=generator), d)
                    for d in range(draws)]:
                for i in range(len(hard.atoms)):
                    record = graph_record(hard, i, model.spec, latent_index=begin + i,
                                          method=method, categorical_draw=draw)
                    if targets is not None:
                        record["target_smiles"] = targets[begin + i]
                    if training_smiles is not None and record["valid"]:
                        record["novel_against_comparison_training"] = record["smiles"] not in training_smiles
                    for label, values in (anchors or {}).items():
                        anchor = values[begin + i]
                        fingerprint = anchor_fingerprints[label][begin + i]
                        observed_fingerprint = checked_fingerprint(fingerprint_generator, record) if fingerprint is not None else None
                        record["similarity_to_" + label] = (float(DataStructs.TanimotoSimilarity(
                            fingerprint, observed_fingerprint))
                            if fingerprint is not None and observed_fingerprint is not None else None)
                        # Slot-aligned categorical changes are available even
                        # for invalid graphs; this is not minimum graph edit distance.
                        observed_graph, target_graph = record["graph"], anchor["graph"]
                        atom_changes = sum(a != b for a, b in zip(observed_graph["atoms"], target_graph["atoms"]))
                        charge_changes = sum(a != b for a, b in zip(observed_graph["charges"], target_graph["charges"]))
                        edges = {(i, j): bond for i, j, bond in observed_graph["edges"]}
                        target_edges = {(i, j): bond for i, j, bond in target_graph["edges"]}
                        bond_changes = sum(edges.get(key, 0) != target_edges.get(key, 0) for key in edges.keys() | target_edges.keys())
                        record["edits_to_" + label] = atom_changes + charge_changes + bond_changes
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                    # Full graphs are retained on disk, not in summary memory.
                    record.pop("graph")
                    rows[method].append(record)
    return {**{method: summarize(values) for method, values in rows.items()},
            "relaxed": {"latents": len(codes), "expected_atom_count_mean": expected_size / len(codes),
                        "element_expected_counts_per_latent_including_no_atom":
                        (element_mass / len(codes)).tolist(),
                        "atom_entropy_per_slot": entropy / (len(codes) * model.spec.max_nodes)},
            "sampling": "independent categorical labels conditioned on one shared decoded latent; no repair"}


@torch.no_grad()
def evaluate_panel(model, dataset, protocol, output, *, limit, deep, loss=None, training_smiles=None):
    """Validation during training, test only in an explicit post-horizon stage."""
    output.mkdir(parents=True, exist_ok=False)
    model.eval()
    indices = subset_indices(len(dataset), limit, protocol["evaluation_seed"], shuffle_full=True)
    loader = DataLoader(Subset(dataset, indices), batch_size=protocol["evaluation_batch_size"],
                        shuffle=False, collate_fn=collate_graphs,
                        generator=torch.Generator(device="cpu").manual_seed(protocol["evaluation_seed"] + 2))
    device = next(model.parameters()).device
    metrics, examples = reconstruction_metrics(model, loader, loss or LossConfig(size_weight=0), device,
                                               len(indices), training_mode="vae", retain_graphs=True)
    with gzip.open(output / "reconstruction.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in examples:
            handle.write(json.dumps(row) + "\n")
    result = {"split": dataset.split, "indices": indices, "reconstruction": metrics,
              "reconstruction_validity_definition": "historical inspect_graph sanitization and nonempty SMILES; not an additional SMILES-reparse check",
              "generation_validity_definition": "raw graph produces a SMILES that RDKit can parse without repair"}
    if not deep:
        return result
    codes = torch.cat([model.encode(hard.to(device).to_relaxed(model.spec))[0].cpu() for hard in loader])
    result["sampled_reconstruction"] = evaluate_codes(
        model, codes, output / "sampled-reconstruction.jsonl", protocol,
        targets=[dataset.smiles[i] for i in indices], training_smiles=training_smiles)
    generator = torch.Generator(device="cpu").manual_seed(protocol["evaluation_seed"])
    prior = torch.randn(protocol["prior_count"], model.config.latent, generator=generator)
    result["prior"] = evaluate_codes(model, prior, output / "prior.jsonl", protocol, training_smiles=training_smiles)
    geometry_count = min(len(codes), protocol["local_geometry_count"])
    anchors = codes[:geometry_count]
    anchor_records = argmax_records(model, anchors, protocol["evaluation_batch_size"])
    direction = torch.randn(anchors.shape, generator=generator)
    spread = max(float(anchors.double().std(0, unbiased=False).square().mean().sqrt()), 1e-3)
    result["neighborhood"] = []
    for scale in protocol["local_geometry_scales"]:
        report = evaluate_codes(model, anchors + scale * spread * direction,
                                output / f"neighborhood-{scale}.jsonl", protocol,
                                targets=[dataset.smiles[i] for i in indices[:geometry_count]],
                                anchors={"baseline_decode": anchor_records})
        result["neighborhood"].append({"relative_scale": scale, "latent_rms_scale": spread, **report})
    pairs = min(protocol["interpolation_pairs"], len(codes) // 2)
    endpoint_codes = torch.cat((codes[:pairs], codes[pairs:2 * pairs]))
    endpoint_rows = argmax_records(model, endpoint_codes, protocol["evaluation_batch_size"])
    result["interpolation"] = []
    for alpha in protocol["interpolation_alphas"]:
        interpolated = (1 - alpha) * codes[:pairs] + alpha * codes[pairs:2 * pairs]
        report = evaluate_codes(model, interpolated, output / f"interpolation-{alpha}.jsonl", protocol,
                                anchors={"endpoint_a": endpoint_rows[:pairs], "endpoint_b": endpoint_rows[pairs:]})
        result["interpolation"].append({"alpha": alpha, "endpoint_indices":
                                       [indices[:pairs], indices[pairs:2 * pairs]], **report})
    return result
