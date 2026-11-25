"""Evaluate encodings under relabeling of every atom slot, including padding."""
import gzip
import json

import torch

from latentmol.datasets.data import collate_graphs
from latentmol.evaluation.checkpoint_comparison import _checked_decode, graph_record
from latentmol.representation import HardGraphBatch


def permute_batch(graph, orders):
    """Each row gets one full bijection, shared by atoms, charges and bond axes."""
    batch, nodes = graph.atoms.shape
    if orders.shape != (batch, nodes) or not torch.equal(
            orders.sort(-1).values, torch.arange(nodes, device=orders.device).expand(batch, -1)):
        raise ValueError("Each graph needs a full slot permutation")
    orders = orders.to(graph.atoms.device)
    bonds = graph.bonds.gather(1, orders[:, :, None].expand(-1, -1, nodes))
    return HardGraphBatch(graph.atoms.gather(1, orders), graph.charges.gather(1, orders),
                         bonds.gather(2, orders[:, None, :].expand(-1, nodes, -1)))


@torch.inference_mode()
def permutation_metrics(model, dataset, indices, output, *, seed, permutations=4, batch_size=64):
    if not indices or type(permutations) is not int or permutations < 1:
        raise ValueError("Atom-order evaluation requires molecules and permutations")
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    counts = dict(canonical_valid=0, canonical_exact=0, permuted_valid=0, permuted_exact=0,
        identical_output_labels=0, identical_valid_identity=0, valid_baseline_attempts=0,
        posterior_within_tolerance=0)
    mu_l2, mu_max, logvar_max = [], [], []
    with gzip.open(output, "xt", encoding="utf-8") as stream:
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start+batch_size]
            target = collate_graphs([dataset[i] for i in chosen]).to(device)
            mu, logvar = model.encode(target.to_relaxed(model.spec))
            if not bool(torch.isfinite(mu).all() and torch.isfinite(logvar).all()):
                raise FloatingPointError("Nonfinite canonical posterior")
            baseline = _checked_decode(model, mu).harden(model.spec)
            baseline_records = []
            for i, index in enumerate(chosen):
                row = graph_record(baseline, i, model.spec, dataset_index=index, trial=0)
                row["target_smiles"] = dataset.smiles[index]
                row["exact"] = row["valid"] and row["smiles"] == row["target_smiles"]
                counts["canonical_valid"] += row["valid"]; counts["canonical_exact"] += row["exact"]
                baseline_records.append(row)
                stream.write(json.dumps(row, allow_nan=False)+"\n")
            for trial in range(1, permutations+1):
                orders = torch.stack([torch.randperm(model.spec.max_nodes, generator=generator) for _ in chosen])
                permuted = permute_batch(target, orders)
                moved_mu, moved_logvar = model.encode(permuted.to_relaxed(model.spec))
                if not bool(torch.isfinite(moved_mu).all() and torch.isfinite(moved_logvar).all()):
                    raise FloatingPointError("Nonfinite reordered posterior")
                hard = _checked_decode(model, moved_mu).harden(model.spec)
                delta = (moved_mu-mu).detach().cpu()
                variance_delta = (moved_logvar-logvar).detach().cpu()
                close = (torch.isclose(moved_mu, mu, rtol=1e-5, atol=1e-6).all(-1) &
                         torch.isclose(moved_logvar, logvar, rtol=1e-5, atol=1e-6).all(-1)).cpu()
                for i, index in enumerate(chosen):
                    row = graph_record(hard, i, model.spec, dataset_index=index, trial=trial,
                        input_permutation=orders[i].tolist(), mu_l2_shift=float(delta[i].norm()),
                        mu_max_abs_shift=float(delta[i].abs().max()),
                        logvar_max_abs_shift=float(variance_delta[i].abs().max()))
                    row["target_smiles"] = dataset.smiles[index]
                    row["exact"] = row["valid"] and row["smiles"] == row["target_smiles"]
                    row["identical_output_labels"] = all(torch.equal(getattr(hard, key)[i], getattr(baseline, key)[i])
                                                          for key in ("atoms", "charges", "bonds"))
                    row["identical_valid_identity"] = (row["valid"] and baseline_records[i]["valid"] and
                                                         row["smiles"] == baseline_records[i]["smiles"])
                    counts["permuted_valid"] += row["valid"]; counts["permuted_exact"] += row["exact"]
                    counts["identical_output_labels"] += row["identical_output_labels"]
                    counts["identical_valid_identity"] += row["identical_valid_identity"]
                    counts["valid_baseline_attempts"] += baseline_records[i]["valid"]
                    counts["posterior_within_tolerance"] += bool(close[i])
                    mu_l2.append(row["mu_l2_shift"]); mu_max.append(row["mu_max_abs_shift"])
                    logvar_max.append(row["logvar_max_abs_shift"])
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
    attempts = len(indices)*permutations
    return dict(molecules=len(indices), permutations=permutations, permuted_attempts=attempts, counts=counts,
        canonical_exact_reconstruction=counts["canonical_exact"]/len(indices),
        permuted_exact_reconstruction=counts["permuted_exact"]/attempts,
        permuted_validity=counts["permuted_valid"]/attempts,
        identical_output_label_fraction=counts["identical_output_labels"]/attempts,
        identity_stability_on_valid_baselines=counts["identical_valid_identity"]/counts["valid_baseline_attempts"]
            if counts["valid_baseline_attempts"] else None,
        posterior_within_tolerance_fraction=counts["posterior_within_tolerance"]/attempts,
        mean_mu_l2_shift=sum(mu_l2)/attempts, max_mu_abs_shift=max(mu_max), max_logvar_abs_shift=max(logvar_max),
        numerical_tolerance=dict(rtol=1e-5, atol=1e-6),
        scope="All slots relabeled; same graph identity; robustness is separate from reconstruction accuracy; repeated permutations share source molecules")
