"""Matched raw-generation and held-out reconstruction panels for the shared ring pilot."""

from collections import Counter, defaultdict
import gzip
import json

import torch
from rdkit import Chem, rdBase

from experiments.vae.training_data_eda.teacher_motif_training_pilot import hard_records
from latentmol.representation import HardGraphBatch
from latentmol.sampling import sample_hard_graph


def _has_saturated_ring(smiles):
    """An isolated all-single six-membered neutral C/N ring; independent of slot order."""
    if not smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    info = mol.GetRingInfo()
    for ring in info.AtomRings():
        if len(ring) != 6 or any(info.NumAtomRings(i) != 1 for i in ring):
            continue
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if any(a.GetAtomicNum() not in (6, 7) or a.GetFormalCharge() for a in atoms):
            continue
        if all(mol.GetBondBetweenAtoms(i, j).GetBondType() == Chem.BondType.SINGLE
               for i, j in zip(ring, ring[1:] + ring[:1])):
            return True
    return False


def _score_graphs(hard, decoded, spec, gate, generator):
    """Choose output using only the model's decoded graph, before reading any target."""
    base = hard_records(hard.to("cpu"), spec)
    if gate is None:
        return base
    from experiments.vae.training_data_eda.shared_ring_gate import apply_ring_gate

    chosen, info = apply_ring_gate(gate, hard, spec, decoded=decoded, generator=generator, include_ceiling=True)
    # Both policies have only the same two allowed outcomes. Score each possible edit once.
    candidates = [j for j, flag in enumerate(info["saturated_by_row"]) if flag]
    ceiling = info["all_eligible"]
    changed = torch.as_tensor(info["changed_by_row"], device=hard.bonds.device, dtype=torch.bool)
    if (not torch.equal(chosen.bonds, torch.where(changed[:, None, None], ceiling.bonds, hard.bonds))
            or not torch.equal(chosen.atoms, hard.atoms) or not torch.equal(chosen.charges, hard.charges)):
        raise RuntimeError("Gate readout does not match the returned graph")
    edited = {}
    if candidates:
        subset = HardGraphBatch(*(field[candidates] for field in (ceiling.atoms, ceiling.charges, ceiling.bonds)))
        edited = dict(zip(candidates, hard_records(subset.to("cpu"), spec), strict=True))
    current = []
    for j, original in enumerate(base):
        row = dict(edited[j] if info["changed_by_row"][j] else original)
        for key in ("eligible", "saturated", "fired", "changed", "probability"):
            row["gate_" + key] = info[key + "_by_row"][j]
        if row["gate_changed"]:
            # The blanket output equals the chosen output in this case.
            row["base"] = original
        elif j in edited:
            row["blanket_alternative"] = edited[j]
        current.append(row)
    return current


def _accumulate(row, counts, identities):
    counts["attempts"] += 1
    for key in ("valid", "aromatic", "amide", "ring3", "ring8plus",
                "invalid_with_explicit_aromatic"):
        counts[key] += int(row[key])
    if not row["valid"]:
        counts["failure_" + (row["failure"] or "unknown")] += 1
    if row["valid"]:
        identities["valid_smiles"].add(row["smiles"])
        if row["scaffold"]:
            identities["nonempty_scaffolds"].add(row["scaffold"])
    if row["aromatic"]:
        identities["valid_aromatic_smiles"].add(row["smiles"])
        if row["scaffold"]:
            identities["aromatic_nonempty_scaffolds"].add(row["scaffold"])
    if "gate_changed" in row:
        for key in ("gate_eligible", "gate_saturated", "gate_fired", "gate_changed"):
            counts[key] += int(row[key])
        if row["gate_eligible"]:
            kind = "saturated" if row["gate_saturated"] else "aromatic"
            counts["gate_" + kind + "_probability_sum"] += row["gate_probability"]
    if "target" in row:
        counts["exact"] += int(row["exact"])
        counts["saturated_targets"] += int(row["target_has_saturated_ring"])
        if row["target_has_saturated_ring"]:
            counts["saturated_target_exact"] += int(row["exact"])
            counts["saturated_target_still_has_saturated_ring"] += int(row["has_saturated_ring"])
            if "gate_changed" in row:
                counts["saturated_target_gate_changes"] += int(row["gate_changed"])


@torch.inference_mode()
def evaluate_endpoint(model, gate, validation, output, *, draws, seed, batch_size=16):
    """Score the same latents, bond draws and canonical held-out IDs at every endpoint.

    A gate gets its own RNG for categorical ring decisions. The argmax and
    held-out panels use the gate's deterministic threshold. Distinct aromatic
    yield uses *all* attempted draws as its denominator, including invalids.
    """
    if min(draws, batch_size) <= 0:
        raise ValueError("draws and batch_size must be positive")
    device = next(model.parameters()).device
    codes = torch.randn(draws, model.config.latent, generator=torch.Generator().manual_seed(seed))
    categorical = torch.Generator(device=device).manual_seed(seed + 1)
    gate_rng = torch.Generator(device=device).manual_seed(seed + 2)
    counts = defaultdict(Counter)
    unique = defaultdict(lambda: defaultdict(set))
    was_training = model.training
    gate_was_training = gate.training if gate is not None else False
    model.eval()
    if gate is not None:
        gate.eval()
    try:
        with gzip.open(output, "xt", encoding="utf-8", compresslevel=2) as stream, rdBase.BlockLogs():
            def record(panel, rows, targets=None):
                for idx, row in enumerate(rows):
                    index = counts[panel]["attempts"]
                    # Always aromatize every eligible ring for the context-blind comparator.
                    # A changed ring has the same outcome in both learned and blanket outputs.
                    scored = [(panel, row)]
                    if gate is not None:
                        blanket = row.get("blanket_alternative", row)
                        scored.append(("blanket_" + panel,
                                       {key: val for key, val in blanket.items() if not key.startswith("gate_")}))
                    for panel_name, outcome in scored:
                        outcome["invalid_with_explicit_aromatic"] = not outcome["valid"] and any(
                            label == model.spec.bond_names.index("AROMATIC") for _, _, label in outcome["edges"])
                        if targets is not None:
                            target = targets[idx]
                            outcome.update(target=target, exact=outcome["valid"] and outcome["smiles"] == target,
                                           target_has_saturated_ring=_has_saturated_ring(target),
                                           has_saturated_ring=outcome["valid"] and _has_saturated_ring(outcome["smiles"]))
                        _accumulate(outcome, counts[panel_name], unique[panel_name])
                    stream.write(json.dumps({"panel": panel, "index": index,
                                             "seed": seed, "batch_size": batch_size, **row},
                                            separators=(",", ":"), allow_nan=False) + "\n")

            for block in codes.split(batch_size):
                decoded = model.decode(block.to(device))
                record("argmax", _score_graphs(decoded.harden(model.spec), decoded, model.spec, gate, None))
                sampled = sample_hard_graph(decoded, model.spec, generator=categorical)
                record("categorical", _score_graphs(sampled, decoded, model.spec, gate, gate_rng))
            for graphs, targets in validation:
                if len(targets) != len(graphs.atoms):
                    raise ValueError("Held-out targets and graph batch differ in length")
                mean, _ = model.encoder(graphs.to(device), from_canonical_cache=True)
                decoded = model.decode(mean)
                generated = _score_graphs(decoded.harden(model.spec), decoded, model.spec, gate, None)
                record("reconstruction", generated, targets)
    finally:
        model.train(was_training)
        if gate is not None:
            gate.train(gate_was_training)
    result = {}
    for panel, metrics in counts.items():
        sizes = {key: len(unique[panel][key]) for key in
                 ("valid_smiles", "nonempty_scaffolds", "valid_aromatic_smiles", "aromatic_nonempty_scaffolds")}
        result[panel] = dict(metrics, unique=sizes,
                             distinct_valid_aromatic_per_1000=1000 * sizes["valid_aromatic_smiles"] / metrics["attempts"])
    return result


def compare_evaluations(control_path, variant_path):
    """Paired counts preserve draws where a new result was gained and where one was lost."""
    changes = defaultdict(lambda: defaultdict(Counter))
    with gzip.open(control_path, "rt", encoding="utf8") as a, gzip.open(variant_path, "rt", encoding="utf8") as b:
        for baseline, tested in zip(a, b, strict=True):
            left, right = json.loads(baseline), json.loads(tested)
            if any(left[k] != right[k] for k in ("panel", "index", "seed", "batch_size")) or left.get("target") != right.get("target"):
                raise ValueError("Paired panels differ in draw, seed, batching, or held-out identity")
            labels = ["valid", "aromatic", "amide", "ring3", "ring8plus", "invalid_with_explicit_aromatic"]
            if left["panel"] == "reconstruction":
                labels.append("exact")
                if left["target_has_saturated_ring"]:
                    left["saturated_target_exact"], right["saturated_target_exact"] = left["exact"], right["exact"]
                    labels.extend(("has_saturated_ring", "saturated_target_exact"))
            for label in labels:
                item = changes[left["panel"]][label]
                item["attempts"] += 1
                item["gained"] += int(right[label] and not left[label])
                item["lost"] += int(left[label] and not right[label])
    return {panel: {name: dict(c, net=c["gained"] - c["lost"]) for name, c in values.items()}
            for panel, values in changes.items()}
