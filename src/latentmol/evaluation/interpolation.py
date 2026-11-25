"""Frozen molecular interpolation with explicit, encoding-consistent atom maps.

Compare canonical encodings with re-encoded, atom-matched inputs. Never permute
only a target while retaining an inconsistent canonical endpoint embedding.
Endpoint failures are recorded, not excluded from the primary denominator.
No training, repairs, external binding model or automatic continuation occurs.
"""
import argparse
from collections import defaultdict
import gzip
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import torch

from latentmol.chemistry import encode_smiles
from latentmol.datasets.data import GraphDataset, collate_graphs
from latentmol.evaluation.checkpoint_comparison import (
    _checked_decode, graph_record, paired_difference, summarize_rows,
)
from latentmol.evaluation.evaluate import load_model
from latentmol.evaluation.molecular_pairs import permute_graph, relabel_categories, select_pairs
from latentmol.experiments import file_sha256, write_json
from latentmol.representation import DecoderOutput, HardGraphBatch
from latentmol.sampling import sample_hard_graph
from latentmol.train import state_digest


def interpolation_codes(a, b, fractions):
    """Pair-major ordering: all declared fractions for pair zero, then pair one."""
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Equal matrices of endpoint embeddings are required")
    weights = a.new_tensor(fractions)[None, :, None]
    return ((1-weights)*a[:, None] + weights*b[:, None]).reshape(-1, a.shape[-1])


def _probabilities(decoded):
    return tuple(value.softmax(-1) for value in
                 (decoded.atoms, decoded.charges, decoded.symmetric_bond_logits))


def _js(p, q):
    """Jensen–Shannon divergence per categorical vector, normalized to [0,1]."""
    p, q = p.double(), q.double()
    middle = (p+q)*.5
    kl_p = torch.where(p > 0, p*(p.clamp_min(1e-30).log()-middle.clamp_min(1e-30).log()), 0.)
    kl_q = torch.where(q > 0, q*(q.clamp_min(1e-30).log()-middle.clamp_min(1e-30).log()), 0.)
    return ((kl_p+kl_q).sum(-1)*.5/math.log(2)).clamp(0, 1)


def probability_distance(observed, target, hard_a, hard_b):
    """Compare raw conditional categories; separate true occupied/union masks.

    These are probability distances, not edit distance to a valid molecule.
    Occupancy-gated bond marginals are deliberately not mixed with conditional
    bond probabilities. Padding and abundant NONE pairs cannot hide union-edge
    disagreement in the equal-term summary.
    """
    occupied = hard_a.atoms[0] != 0
    if not torch.equal(occupied, hard_b.atoms[0] != 0):
        raise ValueError("This interpolation study requires equal occupied slots")
    pairs = (occupied[:, None] & occupied[None, :]).triu(1)
    union = pairs & ((hard_a.bonds[0] != 0) | (hard_b.bonds[0] != 0))
    atoms, charges, bonds = (_js(p, q) for p, q in zip(observed, target))
    average = lambda value, mask: float(value[mask].mean()) if bool(mask.any()) else None
    result = dict(atoms=average(atoms, occupied), charges=average(charges, occupied),
        bonds_all_pairs=average(bonds, pairs), bonds_union_edges=average(bonds, union),
        padding_atoms=average(atoms, ~occupied))
    values = [result[k] for k in ("atoms", "charges", "bonds_union_edges") if result[k] is not None]
    result["equal_term_mean"] = sum(values)/len(values)
    return result


def _one_hot(graph, spec):
    value = graph.to_relaxed(spec)
    return value.atoms[0], value.charges[0], value.bonds[0]


def load_pair_graphs(dataset, pairs, spec):
    """Load both canonical endpoints and the lossless B-to-A target labels."""
    graphs_a, graphs_b, matched_b = [], [], []
    for pair in pairs:
        endpoints = []
        for endpoint in ("a", "b"):
            hard = collate_graphs([dataset[pair[endpoint+"_index"]]])
            expected = encode_smiles(pair[endpoint+"_smiles"], spec)
            if any(not np.array_equal(getattr(hard, field)[0].numpy(), expected[key])
                   for field, key in (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))):
                raise ValueError("RDKit canonical atom indices differ from cached endpoint labels")
            endpoints.append(hard)
        a, b = endpoints
        matched = permute_graph(b, pair["mapping"]["b_original_atom_for_aligned_slot"])
        if graph_record(matched, 0, spec)["smiles"] != pair["b_smiles"]:
            raise ValueError("Atom mapping changed the endpoint's chemical identity")
        graphs_a.append(a)
        graphs_b.append(b)
        matched_b.append(matched)
    return graphs_a, graphs_b, matched_b


def _stack(graphs):
    return HardGraphBatch(*(torch.cat([getattr(g, name) for g in graphs])
                            for name in ("atoms", "charges", "bonds")))


@torch.inference_mode()
def _encode(model, graphs, batch_size):
    result = []
    device = next(model.parameters()).device
    for start in range(0, len(graphs), batch_size):
        mu, logvar = model.encode(_stack(graphs[start:start+batch_size]).to(device).to_relaxed(model.spec))
        if not bool(torch.isfinite(mu).all() and torch.isfinite(logvar).all()):
            raise FloatingPointError("Nonfinite endpoint encoding")
        result.append(mu.cpu())
    return torch.cat(result)


def _label_exact(hard, i, target):
    return all(torch.equal(getattr(hard, key)[i], getattr(target, key)[0])
               for key in ("atoms", "charges", "bonds"))


def _summarize(rows, *, correlated=False):
    if not rows:
        return dict(attempts=0, validity=None, strict_validity=None)
    result = summarize_rows(rows)
    if correlated:
        result.pop("validity_wilson95")
        result.pop("strict_validity_wilson95")
        groups = defaultdict(list)
        for row in rows:
            groups[row["pair_id"]].append(row["valid"])
        rates = np.array([np.mean(values) for values in groups.values()])
        se = float(rates.std(ddof=1)/math.sqrt(len(rates))) if len(rates) > 1 else None
        mean = float(rates.mean())
        result.update(pair_clusters=len(rates),
            pair_cluster_normal95=[max(0., mean-1.959964*se), min(1., mean+1.959964*se)] if se is not None else None)
    for name in ("same_as_a", "same_as_b", "different_from_both_endpoints"):
        result[name+"_count"] = sum(r[name] for r in rows)
    for name in ("tanimoto_to_a", "tanimoto_to_b"):
        values = [r[name] for r in rows if r[name] is not None]
        result[name] = dict(eligible=len(values), mean=sum(values)/len(values) if values else None)
    return result


@torch.inference_mode()
def evaluate_arm(model, pairs, graphs_a, graphs_b, z_a, z_b, config, folder, *, mapped_b=None):
    """Evaluate raw outputs. Optional pair alignment applies to probability diagnostics."""
    folder.mkdir()
    fractions = [0., *config["fractions"], 1.]
    count = len(fractions)
    codes = interpolation_codes(z_a, z_b, fractions)
    np.save(folder/"latents.npy", codes.numpy(), allow_pickle=False)
    chunks = [_checked_decode(model, chunk) for chunk in codes.split(config["batch_size"])]
    decoded = DecoderOutput(*(torch.cat([getattr(chunk, name) for chunk in chunks])
                              for name in ("atoms", "charges", "bonds")))
    del chunks
    probabilities = _probabilities(decoded)
    snapshots = min(config["probability_snapshot_pairs"], len(pairs))*count
    np.savez_compressed(folder/"probability-snapshots.npz", fractions=np.array(fractions),
        atoms=probabilities[0][:snapshots].numpy(), charges=probabilities[1][:snapshots].numpy(),
        conditional_bonds=probabilities[2][:snapshots].numpy())
    point_metrics = []
    for pair_id, pair in enumerate(pairs):
        a, b = graphs_a[pair_id], graphs_b[pair_id]
        if mapped_b is not None:
            b = mapped_b[pair_id]
            slots = pair["mapping"]["b_original_atom_for_aligned_slot"]
            order = torch.tensor([slots + list(range(len(slots), model.spec.max_nodes))])

            def in_a_frame(values):
                return tuple(v[0] for v in relabel_categories(*(x[None] for x in values), order))

        ya, yb = _one_hot(a, model.spec), _one_hot(b, model.spec)
        begin = pair_id*count
        pa = tuple(p[begin] for p in probabilities)
        pb = tuple(p[begin+count-1] for p in probabilities)
        if mapped_b is not None:
            pb = in_a_frame(pb)
        previous = None
        for j, fraction in enumerate(fractions):
            position = begin+j
            current = tuple(p[position] for p in probabilities)
            truth = tuple((1-fraction)*x+fraction*y for x, y in zip(ya, yb))
            decoded_mix = tuple((1-fraction)*x+fraction*y for x, y in zip(pa, pb))
            distance = probability_distance(current, truth, a, b)
            frame = "a"
            if mapped_b is not None:
                alternate = in_a_frame(current)
                alternate_distance = probability_distance(alternate, truth, a, b)
                if alternate_distance["equal_term_mean"] < distance["equal_term_mean"]:
                    current, distance, frame = alternate, alternate_distance, "b"
            occupied = a.atoms[0] != 0
            atom_prob = current[0]
            entropy = -(atom_prob[occupied]*atom_prob[occupied].clamp_min(1e-30).log()).sum(-1).mean()
            point_metrics.append(dict(pair_id=pair_id, fraction=fraction,
                latent_l2=float(codes[position].norm()), expected_atoms=float((1-atom_prob[:, 0]).sum()),
                atom_entropy=float(entropy),
                target_frame=frame, js_to_known_graph_mix=distance,
                js_to_decoded_endpoint_mix=probability_distance(current, decoded_mix, a, b),
                js_from_previous=probability_distance(current, previous, a, b) if previous is not None else None))
            previous = current
    write_json(folder/"probability-metrics.json", point_metrics)
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = {}
    def fingerprint(smiles):
        if smiles not in fingerprints:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError("A reported valid identity cannot be fingerprinted")
            fingerprints[smiles] = fp_generator.GetFingerprint(molecule)
        return fingerprints[smiles]
    generator = torch.Generator().manual_seed(config["categorical_seed"])
    rows = defaultdict(list)
    endpoints = [dict(pair_id=i) for i in range(len(pairs))]
    with gzip.open(folder/"raw-graphs.jsonl.gz", "xt", encoding="utf-8") as stream:
        for draw in range(-1, config["categorical_draws"]):
            method = "argmax" if draw < 0 else "categorical"
            hard = decoded.harden(model.spec) if draw < 0 else sample_hard_graph(
                decoded, model.spec, generator=generator, temperature=1.)
            for i in range(len(codes)):
                pair_id, offset = divmod(i, count)
                pair, fraction = pairs[pair_id], fractions[offset]
                row = graph_record(hard, i, model.spec, pair_id=pair_id, fraction=fraction,
                                   method=method, categorical_draw=draw)
                row.update(same_as_a=row["valid"] and row["smiles"] == pair["a_smiles"],
                    same_as_b=row["valid"] and row["smiles"] == pair["b_smiles"])
                row["different_from_both_endpoints"] = row["valid"] and not row["same_as_a"] and not row["same_as_b"]
                for endpoint in ("a", "b"):
                    row["tanimoto_to_"+endpoint] = float(DataStructs.TanimotoSimilarity(
                        fingerprint(row["smiles"]), fingerprint(pair[endpoint+"_smiles"]))) if row["valid"] else None
                if draw < 0 and offset in (0, count-1):
                    endpoint = "a" if offset == 0 else "b"
                    target = graphs_a[pair_id] if offset == 0 else graphs_b[pair_id]
                    endpoints[pair_id][endpoint+"_graph_exact"] = row["same_as_"+endpoint]
                    endpoints[pair_id][endpoint+"_labels_exact"] = _label_exact(hard, i, target)
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                # Keep argmax labels for the saved-only report; categorical raw
                # graphs remain on disk instead of multiplying summary memory.
                if draw >= 0:
                    row.pop("graph")
                rows[(fraction, method)].append(row)
    cohorts = dict(all_pairs=set(range(len(pairs))),
        both_endpoints_graph_exact={e["pair_id"] for e in endpoints if e["a_graph_exact"] and e["b_graph_exact"]},
        both_endpoints_labels_exact={e["pair_id"] for e in endpoints if e["a_labels_exact"] and e["b_labels_exact"]})
    report = []
    for fraction in fractions:
        views = {}
        for cohort, eligible in cohorts.items():
            views[cohort] = {method: _summarize([r for r in rows[(fraction, method)] if r["pair_id"] in eligible],
                correlated=method == "categorical") for method in ("argmax", "categorical")}
        metrics = [r for r in point_metrics if r["fraction"] == fraction]
        report.append(dict(fraction=fraction, views=views,
            mean_known_mix_js=float(np.mean([m["js_to_known_graph_mix"]["equal_term_mean"] for m in metrics])),
            mean_decoded_mix_js=float(np.mean([m["js_to_decoded_endpoint_mix"]["equal_term_mean"] for m in metrics]))))
    all_interior_valid = sum(all(rows[(fraction, "argmax")][i]["valid"] for fraction in config["fractions"])
                             for i in range(len(pairs)))
    interior_key = ("all_three_interior_argmax_valid_pairs" if len(config["fractions"]) == 3
                    else "all_interior_argmax_valid_pairs")
    result = dict(points=report, endpoints=endpoints, cohort_pair_counts={k: len(v) for k, v in cohorts.items()},
        latent_distance_l2=[float(value) for value in (z_b-z_a).norm(dim=-1)],
        validity_scope="All selected pairs primary; endpoint-conditioned views are diagnostic subsets, not filtered success rates",
        novelty_scope="Different from both endpoints only; no training-set novelty or synthesis claim",
        probability_scope="Normalized categorical JS; conditional bonds, known occupied slots and union edges; no chemistry-distance claim",
        target_frames="Minimum target JS across the two mapped frames" if mapped_b is not None else "fixed",
        **{interior_key: all_interior_valid})
    write_json(folder/"summary.json", result)
    return result, rows


def _validate(config):
    if config.get("version") != "matched-molecular-interpolation-v1" or config.get("split") != "val":
        raise ValueError("Expected an evaluation-only matched validation-pair protocol")
    if config.get("fractions") != [.25, .5, .75]:
        raise ValueError("The declared interior fractions must be 25%, 50% and 75%")
    for key in ("pair_count", "pool_count", "batch_size", "categorical_draws", "max_candidate_pairs",
                "neighbors_per_molecule", "max_pairs_per_scaffold", "mcs_timeout_seconds",
                "match_limit", "alignment_limit", "probability_snapshot_pairs"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError("Positive bounded budget required: "+key)
    if not 0 < config["min_similarity"] <= 1 or not 0 < config["min_core_fraction"] <= 1:
        raise ValueError("Similarity and common-core fractions must be in (0,1]")
    if not 1 <= config["min_atoms"] <= config["max_atoms"]:
        raise ValueError("Invalid occupied-atom range")


@torch.inference_mode()
def run_interpolation(config, output, *, device="cuda"):
    _validate(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(output/"config.json", config)
    try:
        checkpoint_path, data_path = Path(config["checkpoint"]), Path(config["data"])
        if file_sha256(checkpoint_path) != config["checkpoint_sha256"]:
            raise ValueError("Epoch checkpoint bytes differ")
        if file_sha256(data_path/"manifest.json") != config["data_manifest_sha256"]:
            raise ValueError("Dataset manifest differs")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("The allocated CUDA device is unavailable")
        data = GraphDataset(data_path, config["split"])
        if data.manifest["dataset_sha256"] != config["dataset_sha256"]:
            raise ValueError("Dataset identity differs")
        def progress(value):
            write_json(output/"progress.json", value)
            print(json.dumps(value), flush=True)
        pairs, selection = select_pairs(data.smiles, config, progress=progress)
        write_json(output/"pairs.json", dict(pairs=pairs, selection=selection))
        if not pairs:
            raise ValueError("No compatible chemical neighbors found within the declared selection budget")
        model, checkpoint = load_model(checkpoint_path, device, data.spec)
        if checkpoint.get("epoch") != config["epoch"] or checkpoint.get("dataset_sha256") != config["dataset_sha256"]:
            raise ValueError("Checkpoint epoch/dataset identity differs")
        model.requires_grad_(False)
        before = state_digest(model.state_dict())
        del checkpoint
        graphs_a, graphs_b, matched_b = load_pair_graphs(data, pairs, model.spec)
        z_a = _encode(model, graphs_a, config["batch_size"])
        z_b = _encode(model, graphs_b, config["batch_size"])
        z_matched = _encode(model, matched_b, config["batch_size"])
        np.savez_compressed(output/"endpoint-latents.npz", a=z_a.numpy(), b_canonical=z_b.numpy(), b_matched=z_matched.numpy())
        if os.environ.get("DBMOL_LAUNCH_READY"):
            write_json(Path(os.environ["DBMOL_LAUNCH_READY"]), dict(status="models_and_inputs_ready",
                nonce=os.environ["DBMOL_LAUNCH_NONCE"], job_id=os.environ.get("LSB_JOBID")))
        progress(dict(stage="evaluating", pairs=len(pairs), elapsed_seconds=time.monotonic()-started))
        results, raw = {}, {}
        for name, b, z in (("canonical", graphs_b, z_b), ("matched", matched_b, z_matched)):
            results[name], raw[name] = evaluate_arm(model, pairs, graphs_a, b, z_a, z, config, output/name)
            progress(dict(stage="evaluating", completed_arms=list(results), pairs=len(pairs),
                          elapsed_seconds=time.monotonic()-started))
        if state_digest(model.state_dict()) != before or file_sha256(checkpoint_path) != config["checkpoint_sha256"]:
            raise RuntimeError("Unexpected frozen-model/checkpoint writes")
        paired = {str(fraction): paired_difference([r["valid"] for r in raw["canonical"][(fraction, "argmax")]],
                    [r["valid"] for r in raw["matched"][(fraction, "argmax")]]) for fraction in config["fractions"]}
        result = dict(version=config["version"], status="completed", epoch=config["epoch"],
            checkpoint_sha256=config["checkpoint_sha256"], selection=selection, arms=results,
            matched_minus_canonical_argmax=paired, model_unchanged=True, training_started=False,
            final_test_split_untouched=True, elapsed_seconds=time.monotonic()-started,
            permutation_latent_shift_l2=(z_matched-z_b).norm(dim=-1).tolist(), job_id=os.environ.get("LSB_JOBID"),
            warning="Matched encodings can be out of distribution for the slot-sensitive encoder; inspect endpoint graph AND label reconstruction",
            uncertainty="Selected disjoint chemical pairs; categorical draws share a latent. Pointwise descriptive intervals, no multiple-testing correction")
        write_json(output/"summary.json", result)
        write_json(output/"progress.json", dict(stage="completed", elapsed_seconds=result["elapsed_seconds"]))
        from latentmol.reporting.interpolation import render_interpolation_report
        try:
            render_interpolation_report(output)
        except Exception as error:
            write_json(output/"report-failure.json", dict(error=type(error).__name__+": "+str(error),
                scientific_summary_saved=True))
        return result
    except BaseException as error:
        write_json(output/"failure.json", dict(status="failed", error=type(error).__name__+": "+str(error),
            elapsed_seconds=time.monotonic()-started, training_started=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run_interpolation(json.loads(args.config.read_text(encoding="utf-8")), args.output, device=args.device)
    print(json.dumps(dict(status=result["status"], pairs=result["selection"]["selected_pairs"], output=str(args.output))))


if __name__ == "__main__":
    main()
