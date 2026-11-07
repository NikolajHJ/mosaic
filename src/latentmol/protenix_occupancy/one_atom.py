"""One optional atom with an exact expectation over two native topologies."""

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch

from latentmol.chemistry import decode_graph, encode_smiles, inspect_graph
from latentmol.evaluation.evaluate import load_model
from latentmol.multimodel.external import file_sha256, tensor_tree_sha256
from latentmol.protenix_occupancy.native import load_frozen_tiny
from latentmol.protenix_occupancy.study import derivative_agreement, selected_outputs, write_json
from latentmol.representation import HardGraphBatch


ENDPOINTS = (
    ("full", "features", "pair_indices"),
    ("deleted", "compact_features", "compact_pair_indices"),
    ("fresh_full", "fresh_full_features", "fresh_full_pair_indices"),
    ("fresh_deleted", "fresh_deleted_features", "fresh_pair_indices"),
)
START_Q = 0.1
LATENT_RADIUS = 1.0
LATENT_STEP = 0.125
LATENT_UPDATES = 16


def optional_log_odds(decoded, slot):
    """Compare the total mass of all elements to the absent category."""
    logits = decoded.atoms[0, slot].double()
    return logits[1:].logsumexp(dim=0) - logits[0]


def expected_loss(q, deleted_loss, full_loss):
    """Mix scalar losses after scoring each native topology."""
    return torch.lerp(deleted_loss, full_loss, q)


def labels(smiles, spec, device):
    encoded = encode_smiles(smiles, spec)
    return HardGraphBatch(*(torch.as_tensor(encoded[key], device=device).long()[None]
                            for key in ("atom_types", "charges", "bond_types")))


def optional_slot(full, spec, atomic_number, deleted_smiles):
    indices = (full.atoms[0] == spec.atomic_numbers.index(atomic_number)).nonzero().flatten()
    if indices.numel() != 1:
        raise ValueError("Optional terminal element is not unique in the canonical input")
    slot = int(indices.item())
    atoms, bonds = full.atoms[0].clone(), full.bonds[0].clone()
    if int(bonds[slot].count_nonzero()) != 1:
        raise ValueError("Optional element must be terminal")
    atoms[slot] = 0
    bonds[slot, :] = bonds[:, slot] = 0
    charges = full.charges[0].clone()
    charges[slot] = spec.formal_charges.index(0)
    if decode_graph(atoms, charges, bonds, spec) != deleted_smiles:
        raise ValueError("Removing the mapped slot does not reconstruct the native deleted ligand")
    return slot


def hard_observation(decoded, spec, slot, present=None):
    graph = decoded.harden(spec)
    atoms, charges, bonds = graph.atoms[0], graph.charges[0], graph.bonds[0]
    if present is not None:
        # Apply the controlled occupancy decision only. The rest of the graph
        # still comes from the current decoder, including the optional element.
        atoms[slot] = decoded.atoms[0, slot, 1:].argmax() + 1 if present else 0
        charges[slot] = (decoded.charges[0, slot].argmax() if present
                         else spec.formal_charges.index(0))
        incident = decoded.symmetric_bond_logits[0, slot].argmax(-1)
        incident = incident.masked_fill(atoms == 0, 0) if present else torch.zeros_like(incident)
        incident[slot] = 0
        bonds[slot, :] = bonds[:, slot] = incident
    result = inspect_graph(atoms, charges, bonds, spec)
    return dict(atom_count=int(atoms.count_nonzero()), smiles=result.smiles,
                failure=result.failure_reason, optional_atomic_number=spec.atomic_numbers[int(atoms[slot])])


def latent_probe(model, bundle, losses, atomic_number):
    meta, spec = bundle["metadata"], model.spec
    full = labels(meta["full_smiles"], spec, losses["full"].device)
    slot = optional_slot(full, spec, atomic_number, meta["deleted_smiles"])
    with torch.no_grad():
        anchor, _ = model.encode(full.to_relaxed(spec))
        initial_u = optional_log_odds(model.decode(anchor), slot)
    if not torch.isfinite(anchor).all() or not initial_u.isfinite():
        raise RuntimeError("Nonfinite latent anchor or occupancy logits")
    offset = math.log(START_Q / (1.0 - START_Q)) - initial_u
    point = anchor
    trace = []
    native_delta = losses["full"] - losses["deleted"]

    def loss_at(z):
        q = (optional_log_odds(model.decode(z), slot) + offset).sigmoid()
        return expected_loss(q, losses["deleted"], losses["full"])

    for update in range(LATENT_UPDATES + 1):
        point = point.detach().requires_grad_()
        decoded = model.decode(point)
        u = optional_log_odds(decoded, slot)
        q = (u + offset).sigmoid()
        loss = expected_loss(q, losses["deleted"], losses["full"])
        if not loss.requires_grad:
            raise RuntimeError("Latent occupancy score is detached")
        gradient, occupancy_gradient = torch.autograd.grad(loss, (point, u))
        norm = float(gradient.norm())
        if not math.isfinite(norm) or not occupancy_gradient.isfinite() or not loss.isfinite():
            raise RuntimeError("Nonfinite occupancy score or gradient")
        with torch.no_grad():
            displacement = float((point - anchor).norm())
            row = dict(update=update, q=float(q), raw_q=float(u.sigmoid()), loss=float(loss),
                       latent_radius=displacement, latent_gradient_norm=norm,
                       occupancy_logit_gradient=float(occupancy_gradient),
                       raw_expected_count=float(decoded.probabilities().expected_size.item()),
                       raw=hard_observation(decoded, spec, slot),
                       controlled=hard_observation(decoded, spec, slot, bool(q >= 0.5)))
            trace.append(row)
            if not norm:
                break
            direction = gradient / norm
            if update == 0:
                row["finite_difference"] = {
                    str(step): derivative_agreement(norm, float(
                        (loss_at(point + step * direction) - loss_at(point - step * direction))
                        / (2 * step)))
                    for step in (0.01, 0.03)
                }
                row["analytic_logit_gradient"] = float(q * (1 - q) * native_delta)
            if update < LATENT_UPDATES:
                delta = point - LATENT_STEP * direction - anchor
                point = anchor + delta * (LATENT_RADIUS / delta.norm().clamp_min(LATENT_RADIUS))
    return dict(slot=slot, applied_logit_offset=float(offset), trace=trace)


def native_endpoints(native, bundle):
    targets = bundle["target_bins"].cuda()
    references, repeats = {}, {}
    for name, feature_key, pair_key in ENDPOINTS:
        features = native.prepare_features(bundle[feature_key])
        pairs = bundle[pair_key].cuda()
        if pairs.shape != bundle["pair_indices"].shape:
            raise ValueError("Endpoint changed the retained-pair readout")
        for collection in (references, repeats):
            result = selected_outputs(native.native_logits(features), pairs, targets)
            if any(not value.isfinite().all() for value in result.values()):
                raise RuntimeError("Nonfinite Protenix endpoint output")
            collection[name] = result

    losses = {name: values["loss"] for name, values in references.items()}
    equality = {}
    for name, q in (("deleted", 0.0), ("full", 1.0)):
        occupancy = losses[name].new_tensor(q)
        mixed_loss = expected_loss(occupancy, losses["deleted"], losses["full"])
        mixed_probabilities = torch.lerp(references["deleted"]["probabilities"].double(),
                                       references["full"]["probabilities"].double(), occupancy)
        loss_error = float((mixed_loss - repeats[name]["loss"]).abs())
        max_tv = float((mixed_probabilities - repeats[name]["probabilities"]).abs().sum(-1).max() / 2)
        equality[name] = dict(loss_error=loss_error, max_pair_tv=max_tv,
                              passed=loss_error <= 1e-6 and max_tv <= 1e-4)
    return losses, repeats, equality


def assess_case(losses, repeats, equality, probe, meta):
    compact_delta = float(losses["full"] - losses["deleted"])
    fresh_delta = float(losses["fresh_full"] - losses["fresh_deleted"])
    noise = max(float((losses[name] - values["loss"]).abs()) for name, values in repeats.items())
    threshold = max(1e-6, 4 * noise)
    initial, final = probe["trace"][0], probe["trace"][-1]
    agreement = abs(compact_delta) > threshold and abs(fresh_delta) > threshold and compact_delta * fresh_delta > 0
    q_change = final["q"] - initial["q"]
    moved_toward_preferred = q_change * compact_delta < 0
    useful = (agreement and moved_toward_preferred and
              any(test["passed"] and test["finite_difference"] > 0
                  for test in initial.get("finite_difference", {}).values()))
    preferred = ((meta["full_smiles"] if fresh_delta < 0 else meta["deleted_smiles"])
                 if abs(fresh_delta) > threshold else None)
    return dict(compact_full_minus_deleted=compact_delta, fresh_full_minus_deleted=fresh_delta,
                maximum_repeat_loss_range=noise, endpoint_direction_agrees=agreement,
                endpoint_equivalence_passed=all(row["passed"] for row in equality.values()),
                occupancy_change=q_change, moved_toward_preferred=moved_toward_preferred,
                useful_gradient=useful, fresh_preferred_smiles=preferred,
                raw_matches_native_endpoints=[initial["raw"]["smiles"] in
                    (meta["full_smiles"], meta["deleted_smiles"]), final["raw"]["smiles"] in
                    (meta["full_smiles"], meta["deleted_smiles"])],
                raw_preferred_at_start=preferred is not None and initial["raw"]["smiles"] == preferred,
                raw_preferred_at_end=preferred is not None and final["raw"]["smiles"] == preferred,
                controlled_preferred_at_start=preferred is not None and initial["controlled"]["smiles"] == preferred,
                controlled_preferred_at_end=preferred is not None and final["controlled"]["smiles"] == preferred)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    job_id = os.environ.get("LSB_JOBID", "")
    if not job_id.isdigit() or not torch.cuda.is_available():
        raise RuntimeError("Run native Protenix inside an allocated GPU job")
    started = time.monotonic()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    checkpoint = Path(config["checkpoint"])
    if file_sha256(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Epoch-45 checkpoint SHA256 mismatch")
    model, _ = load_model(checkpoint, "cuda")
    model.requires_grad_(False)
    model_state = tensor_tree_sha256(model.state_dict())
    native = load_frozen_tiny(**config["native"])
    native_state = native.snapshot_state()
    cases = []
    for item in config["cases"]:
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError("Prepared native endpoint bundle SHA256 mismatch")
        bundle = torch.load(item["path"], map_location="cpu", weights_only=True)
        if bundle["case_id"] != item["id"]:
            raise ValueError("Prepared native case identity mismatch")
        cases.append((item, bundle))
    if proof := os.environ.get("DBMOL_LAUNCH_READY"):
        write_json(Path(proof), dict(status="models_and_inputs_ready", job_id=job_id,
                                    nonce=os.environ["DBMOL_LAUNCH_NONCE"]))
    rows = []
    for item, bundle in cases:
        losses, repeats, equality = native_endpoints(native, bundle)
        probe = latent_probe(model, bundle, losses, item["optional_atomic_number"])
        assessment = assess_case(losses, repeats, equality, probe, bundle["metadata"])
        row = dict(case_id=item["id"], endpoints={key: float(value) for key, value in losses.items()},
                   endpoint_checks=equality, latent=probe, assessment=assessment)
        rows.append(row)
        write_json(args.output / (item["id"] + ".json"), row)
    native.assert_unchanged(native_state)
    if (model_state != tensor_tree_sha256(model.state_dict())
            or any(p.requires_grad or p.grad is not None for p in model.parameters())
            or any(module.training for module in model.modules())):
        raise RuntimeError("Frozen VAE state or mode changed")
    summaries = [row["assessment"] for row in rows]
    write_json(args.output / "result.json", dict(
        status="completed", job_id=job_id, method="native_two_topology_expectation",
        checkpoint_sha256=config["checkpoint_sha256"], native_model_calls=8 * len(rows),
        seconds=time.monotonic() - started, cases=rows,
        supports_budget_followup=all(row["endpoint_equivalence_passed"] for row in summaries)
        and any(row["useful_gradient"] for row in summaries)))


if __name__ == "__main__":
    main()
