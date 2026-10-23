"""Supervised, checkpoint-bound early-feature fitting; numerical work is HPC-only.

The VAE determines each teacher molecule. Only small input-bridge parameters are
fitted to exact frozen native projections; affinity and other desired scores are
never labels. Dataset samples retain z and hard labels rather than pair states.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from rdkit import Chem, rdBase
import torch

from latentmol.chemistry import encode_smiles
from latentmol.chemistry_classification import classify_repair_graph
from latentmol.data import GraphDataset, collate_graphs, sha256_file
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.train import state_digest
from .contracts import NATIVE_BOND_IDS
from .embedding_bridge import BRIDGE_VERSION, HiddenEmbeddingBridge, NativeProjectionBank


DATA_VERSION = "epoch36-hidden-native-paired-data-v1"
TRAINING_VERSION = "epoch36-supervised-embedding-fit-v1"
EPOCH36_SHA256 = "3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f"
SPLITS = ("train", "dev", "audit")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _check_deadline(deadline) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Embedding experiment internal runtime deadline reached")


def _vae_identity(vae) -> dict:
    """Validate a caller-authenticated checkpoint binding, not infer file hashes."""
    identity = getattr(vae, "embedding_bridge_checkpoint_identity", None)
    if identity != {"epoch": 36, "sha256": EPOCH36_SHA256}:
        raise ValueError("Bridge requires caller-authenticated original epoch-36 checkpoint identity")
    if vae.training or any(parameter.requires_grad for parameter in vae.parameters()):
        raise ValueError("Bridge preparation/fitting requires the VAE frozen in eval mode")
    return dict(identity)


def _source_holdout_identities(data: GraphDataset) -> set[str]:
    """Read val/test identity metadata only; do not encode or evaluate their graphs."""
    cache = data.cache_dir
    if data.manifest["dataset_format_version"] == 1:
        with np.load(cache / "dataset.npz", allow_pickle=False) as archive:
            labels, identities = archive["split"], archive["smiles"]
            return set(identities[labels != 0].tolist())
    labels = np.load(cache / "split.npy", mmap_mode="r", allow_pickle=False)
    offsets = np.load(cache / "smiles_offsets.npy", mmap_mode="r", allow_pickle=False)
    result = set()
    with (cache / "smiles.utf8").open("rb") as stream:
        for index in np.flatnonzero(labels != 0):
            begin, end = int(offsets[index]), int(offsets[index + 1])
            stream.seek(begin)
            result.add(stream.read(end - begin).decode("utf-8"))
    return result


def partition_source_roots(identities: list[str], seed: int, counts: dict[str, int]) -> dict[str, list[int]]:
    """Allocate source training identities before any posterior/local generation."""
    if set(counts) != set(SPLITS) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError("Each bridge partition requires a positive source-root count")
    if len(set(identities)) != len(identities) or sum(counts.values()) > len(identities):
        raise ValueError("Bridge source roots require enough unique VAE-training identities")
    order = sorted(range(len(identities)), key=lambda i: hashlib.sha256(
        f"{seed}|source-root|{identities[i]}".encode()).hexdigest())
    result, offset = {}, 0
    for split in SPLITS:
        result[split] = order[offset:offset + counts[split]]
        offset += counts[split]
    return result


def _prior_split(group: str, seed: int, counts: dict[str, int]) -> str:
    value = int(hashlib.sha256(f"{seed}|{group}".encode()).hexdigest(), 16) / 2**256
    total, cumulative = sum(counts.values()), 0
    for split in SPLITS:
        cumulative += counts[split] / total
        if value < cumulative:
            return split
    return SPLITS[-1]


def conditioned_hard(decoded: DecoderOutput, active: torch.Tensor, spec: GraphSpec) -> HardGraphBatch:
    """Harden the same fixed-mask policy used by native scoring, without repair."""
    if (active.dtype != torch.bool or active.shape != decoded.atoms.shape[:2]
            or not bool(active.any(-1).all())):
        raise ValueError("Conditioned teacher needs one nonempty boolean mask per latent")
    atoms = (decoded.atoms[..., 1:].argmax(-1) + 1).masked_fill(~active, 0)
    charges = decoded.charges.argmax(-1).masked_fill(~active, spec.formal_charges.index(0))
    pairs = active[:, :, None] & active[:, None, :]
    diagonal = torch.eye(spec.max_nodes, dtype=torch.bool, device=active.device)[None]
    bonds = decoded.symmetric_bond_logits.argmax(-1).masked_fill(~pairs | diagonal, 0)
    return HardGraphBatch(atoms, charges, bonds)


def slot_to_canonical_alignment(hard: HardGraphBatch, spec: GraphSpec, smiles: str) -> list[int]:
    """Find and verify an exact element/charge/typed-bond map, including holes."""
    if hard.atoms.shape != (1, spec.max_nodes):
        raise ValueError("Alignment accepts exactly one hard graph")
    hard = hard.to("cpu")
    active = torch.nonzero(hard.atoms[0] != 0, as_tuple=False).flatten().tolist()
    encoded = encode_smiles(smiles, spec)
    canonical = Chem.MolFromSmiles(encoded["smiles"])
    rw = Chem.RWMol()
    for slot in active:
        atom = Chem.Atom(spec.atomic_numbers[int(hard.atoms[0, slot])])
        atom.SetFormalCharge(spec.formal_charges[int(hard.charges[0, slot])])
        rw.AddAtom(atom)
    for left, slot_left in enumerate(active):
        for right in range(left + 1, len(active)):
            name = spec.bond_names[int(hard.bonds[0, slot_left, active[right]])]
            if name == "NONE":
                continue
            rw.AddBond(left, right, getattr(Chem.BondType, name))
            if name == "AROMATIC":
                rw.GetAtomWithIdx(left).SetIsAromatic(True)
                rw.GetAtomWithIdx(right).SetIsAromatic(True)
                rw.GetBondBetweenAtoms(left, right).SetIsAromatic(True)
    molecule = rw.GetMol()
    with rdBase.BlockLogs():
        Chem.SanitizeMol(molecule)
    mapping = canonical.GetSubstructMatch(molecule, useChirality=False)
    if len(mapping) != len(active) or len(set(mapping)) != len(active):
        raise ValueError("No complete native/canonical atom alignment")
    result = [-1] * spec.max_nodes
    for index, slot in enumerate(active):
        native = mapping[index]
        result[slot] = native
        if (int(encoded["atom_types"][native]) != int(hard.atoms[0, slot])
                or int(encoded["charges"][native]) != int(hard.charges[0, slot])):
            raise ValueError("Native atom alignment changes element or charge")
        for other_index, other_slot in enumerate(active):
            if int(encoded["bond_types"][native, mapping[other_index]]) != int(hard.bonds[0, slot, other_slot]):
                raise ValueError("Native atom alignment changes typed-bond identity")
    return result


def prepare_dataset(vae, data_path, output, seed, train_count=4096, dev_count=512,
                    audit_count=512, prior_attempts=1024, *, local_scale=0.05,
                    deadline=None) -> dict:
    """Build immutable compact paired data from VAE-training roots and fresh priors.

    Counts are selected source-root counts, not accepted-example targets. Each
    source root supplies mean/posterior/local attempts; each prior root supplies
    natural/local attempts. Invalid/excluded/colliding attempts stay recorded.
    A root's natural mean/prior decode fixes occupancy for all its descendants.
    ``deadline`` is an absolute ``time.monotonic`` time.
    """
    identity = _vae_identity(vae)
    if (type(prior_attempts) is not int or prior_attempts < 0
            or not math.isfinite(local_scale) or local_scale <= 0):
        raise ValueError("Require nonnegative prior attempts and positive finite local scale")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    before = state_digest(vae.state_dict())
    data = GraphDataset(data_path, "train")
    if data.spec != vae.spec:
        raise ValueError("Source dataset and frozen VAE graph schemas differ")
    heldout = _source_holdout_identities(data)
    counts = dict(train=train_count, dev=dev_count, audit=audit_count)
    roots = partition_source_roots(data.smiles, seed, counts)
    # Reserve all selected source identities first. A reconstruction from another
    # root cannot leak a selected dev/audit molecule into bridge training.
    owners = {data.smiles[index]: split for split, indices in roots.items() for index in indices}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = next(vae.parameters()).device
    samples, attempts, reasons = [], [], Counter()

    def retain(z, hard, split, group, origin):
        hard = hard.to("cpu")
        assessment = classify_repair_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], vae.spec)
        record = dict(attempt=len(attempts), split=split, group=group, origin=origin,
                      classification=assessment, accepted=False)
        reason = None
        alignment = None
        canonical = assessment["smiles"]
        if not assessment["strict_teacher_valid"]:
            reason = "invalid:" + str(assessment["reason"])
        elif canonical in heldout:
            reason = "source_heldout_identity"
        elif canonical in owners and owners[canonical] != split:
            reason = "cross_bridge_split_identity"
        else:
            try:
                alignment = slot_to_canonical_alignment(hard, vae.spec, canonical)
            except ValueError as error:
                reason = "identity_alignment"
                record["alignment_detail"] = str(error)
        if reason is not None:
            reasons[reason] += 1
            record["rejection_reason"] = reason
        else:
            owners[canonical] = split
            samples.append(dict(z=z.detach().cpu().reshape(-1).float(),
                atoms=hard.atoms[0].to(torch.uint8), charges=hard.charges[0].to(torch.uint8),
                bonds=hard.bonds[0].to(torch.uint8), active=hard.atoms[0] != 0,
                slot_to_canonical=torch.tensor(alignment, dtype=torch.int16),
                split=split, group=group, origin=origin, identity=canonical))
            record.update(accepted=True, sample_index=len(samples) - 1)
        attempts.append(record)

    try:
        with torch.no_grad():
            # Order is fixed prospectively; assigning an unseen decoded identity
            # to audit/dev first prevents train acceptance from consuming it.
            for split in ("audit", "dev", "train"):
                for index in roots[split]:
                    _check_deadline(deadline)
                    graph = collate_graphs([data[index]]).to(device)
                    mu, logvar = vae.encode(graph.to_relaxed(vae.spec))
                    base = vae.decode(mu)
                    active = base.harden(vae.spec).atoms != 0
                    group = "source:" + hashlib.sha256(data.smiles[index].encode()).hexdigest()
                    if not bool(active.any()):
                        retain(mu, base.harden(vae.spec), split, group, "posterior_mean")
                        continue
                    posterior = mu + torch.randn(mu.shape, generator=generator).to(mu) * (0.5 * logvar).exp()
                    local = mu + local_scale * torch.randn(mu.shape, generator=generator).to(mu)
                    latents = torch.cat((mu, posterior, local), 0)
                    labels = conditioned_hard(vae.decode(latents), active.expand(3, -1), vae.spec)
                    for row, origin in enumerate(("posterior_mean", "posterior_sample", "local")):
                        retain(latents[row], HardGraphBatch(labels.atoms[row:row + 1],
                            labels.charges[row:row + 1], labels.bonds[row:row + 1]), split, group, origin)
            for index in range(prior_attempts):
                _check_deadline(deadline)
                group = f"prior:{seed}:{index}"
                split = _prior_split(group, seed, counts)
                z = torch.randn((1, vae.config.latent), generator=generator).to(device)
                base = vae.decode(z)
                active = base.harden(vae.spec).atoms != 0
                if not bool(active.any()):
                    retain(z, base.harden(vae.spec), split, group, "prior")
                    continue
                local = z + local_scale * torch.randn(z.shape, generator=generator).to(z)
                latents = torch.cat((z, local), 0)
                labels = conditioned_hard(vae.decode(latents), active.expand(2, -1), vae.spec)
                for row, origin in enumerate(("prior", "prior_local")):
                    retain(latents[row], HardGraphBatch(labels.atoms[row:row + 1],
                        labels.charges[row:row + 1], labels.bonds[row:row + 1]), split, group, origin)
        after = state_digest(vae.state_dict())
        if before != after or _vae_identity(vae) != identity:
            raise RuntimeError("Frozen VAE changed during paired-data construction")
        if not samples:
            raise ValueError("No supported strict-valid paired rows were produced")
        payload = {name: torch.stack([row[name] for row in samples]) for name in
                   ("z", "atoms", "charges", "bonds", "active", "slot_to_canonical")}
        payload.update({name: [row[name] for row in samples] for name in
                        ("split", "group", "origin", "identity")})
        torch.save(payload, output / "samples.pt")
        with (output / "attempts.jsonl").open("w", encoding="utf-8") as stream:
            for record in attempts:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        split_counts = {split: sum(row["split"] == split for row in samples) for split in SPLITS}
        manifest = dict(version=DATA_VERSION, status="prepared_not_trained", seed=seed,
            spec=vae.spec.to_dict(), spec_fingerprint=vae.spec.fingerprint(),
            vae_checkpoint=identity, vae_state_sha256_before=before, vae_state_sha256_after=after,
            source_dataset_sha256=data.manifest["dataset_sha256"],
            source_split_sha256=data.manifest["split_sha256"],
            source_manifest_sha256=sha256_file(data.cache_dir / "manifest.json"),
            selected_source_roots=roots, selected_source_root_counts=counts,
            source_heldout_identity_count=len(heldout), source_heldout_usage="identity exclusion only; no model evaluation",
            prior_root_attempts=prior_attempts, local_scale=local_scale,
            graph_policy="root-natural-occupancy-conditioned-descendants-v1",
            root_order=["audit", "dev", "train", "hash-partitioned-prior"],
            attempts=len(attempts), accepted=len(samples), accepted_by_split=split_counts,
            unique_identities_by_split={split: len({r["identity"] for r in samples if r["split"] == split}) for split in SPLITS},
            rejected=dict(reasons), all_attempts_retained=True,
            samples_file="samples.pt", samples_sha256=sha256_file(output / "samples.pt"),
            attempts_file="attempts.jsonl", attempts_sha256=sha256_file(output / "attempts.jsonl"))
        _write_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        with (output / "attempts.jsonl").open("w", encoding="utf-8") as stream:
            for record in attempts:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        _write_json(output / "preparation-failure.json", dict(status="failed_or_partial_not_prepared",
            error_type=type(error).__name__, error=str(error), completed_attempts=len(attempts),
            accepted=len(samples), vae_state_unchanged=state_digest(vae.state_dict()) == before))
        raise


def _load_dataset(path, vae):
    path = Path(path)
    manifest_path = path if path.is_file() else path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    if (manifest.get("version") != DATA_VERSION or manifest.get("vae_checkpoint") != _vae_identity(vae)
            or manifest.get("spec_fingerprint") != vae.spec.fingerprint()
            or manifest.get("vae_state_sha256_after") != state_digest(vae.state_dict())
            or sha256_file(root / "samples.pt") != manifest.get("samples_sha256")
            or sha256_file(root / "attempts.jsonl") != manifest.get("attempts_sha256")):
        raise ValueError("Paired dataset/checkpoint/schema/file identity failed authentication")
    data = torch.load(root / "samples.pt", map_location="cpu", weights_only=True)
    count = manifest["accepted"]
    expected = {"z", "atoms", "charges", "bonds", "active", "slot_to_canonical",
                "split", "group", "origin", "identity"}
    if set(data) != expected or any(len(value) != count for value in data.values()):
        raise ValueError("Paired sample payload dimensions disagree with manifest")
    groups, identities = {}, {}
    for split, group, identity in zip(data["split"], data["group"], data["identity"], strict=True):
        if (split not in SPLITS or groups.get(group, split) != split
                or identities.get(identity, split) != split):
            raise ValueError("Paired dataset leaks groups or decoded identities between splits")
        groups[group], identities[identity] = split, split
    if not torch.equal(data["active"], data["atoms"] != 0):
        raise ValueError("Paired sample occupancy disagrees with teacher hard graph")
    return data, manifest, sha256_file(manifest_path)


def _hard_rows(data, rows, device):
    return HardGraphBatch(*(data[name][rows].to(device=device, dtype=torch.long)
                            for name in ("atoms", "charges", "bonds")))


def _forward_rows(vae, bank, bridge, data, rows, device):
    hard = _hard_rows(data, rows, device)
    with torch.no_grad():
        decoded, features = vae.decode_with_features(data["z"][rows].to(device))
        reproduced = conditioned_hard(decoded, data["active"][rows].to(device), vae.spec)
        if any(not torch.equal(left, right) for left, right in
               ((reproduced.atoms, hard.atoms), (reproduced.charges, hard.charges),
                (reproduced.bonds, hard.bonds))):
            raise ValueError("Frozen VAE no longer reproduces the stored teacher molecule at z")
        targets = bank.targets(hard, vae.spec)
    predictions = bridge(features.atom_hidden, features.bond_input,
                         data["active"][rows].to(device), features.charge_input)
    return hard, targets, predictions


@torch.no_grad()
def _identity_class_counts(bank, predictions, hard, spec):
    """Expose rare-class denominators rather than hide them in an atom mean."""
    active = hard.atoms != 0
    pairs = active[:, :, None] & active[:, None, :]
    result = {}
    for descriptor in bank.descriptors:
        projection = bank.projection(descriptor.key)
        if descriptor.kind == "atom_element":
            names = [str(value) for value in spec.atomic_numbers[1:]]
            prototypes = projection[hard.atoms.new_tensor(spec.atomic_numbers)[1:]]
            labels, mask = hard.atoms - 1, active
        elif descriptor.kind == "atom_charge":
            names = [str(value) for value in spec.formal_charges]
            prototypes = projection[0] * projection.new_tensor(spec.formal_charges)[:, None]
            labels, mask = hard.charges, active
        elif descriptor.kind == "connectivity":
            names, prototypes = ["NONE", "PRESENT"], projection
            labels, mask = (hard.bonds != 0).long(), pairs
        else:
            names = list(spec.bond_names)
            prototypes = projection[hard.atoms.new_tensor([NATIVE_BOND_IDS[name] for name in names])]
            labels, mask = hard.bonds, pairs
        value = predictions[descriptor.key]
        predicted = torch.cdist(value.reshape(-1, descriptor.width), prototypes).argmin(-1).reshape_as(labels)
        result[descriptor.key] = {name: {"count": int(((labels == index) & mask).sum()),
            "correct": int(((labels == index) & (predicted == index) & mask).sum())}
            for index, name in enumerate(names)}
    return result


@torch.no_grad()
def _evaluate(vae, bank, bridge, data, rows, device, batch_size, deadline):
    if not rows:
        raise ValueError("Bridge development/audit cohort has no accepted rows")
    bridge.eval()
    totals, coverage, classes, seen = Counter(), Counter(), {}, 0
    for start in range(0, len(rows), batch_size):
        _check_deadline(deadline)
        chosen = rows[start:start + batch_size]
        hard, targets, predictions = _forward_rows(vae, bank, bridge, data, chosen, device)
        loss, metrics = bank.loss(predictions, targets)
        metrics.update(bank.identity_metrics(predictions, hard, vae.spec))
        if not bool(loss.isfinite()) or any(not bool(value.isfinite().all()) for value in metrics.values()):
            raise FloatingPointError("Nonfinite supervised bridge evaluation")
        for name, value in metrics.items():
            if name.endswith("_count"):
                coverage[name] += int(value)
            else:
                totals[name] += float(value) * len(chosen)
        for projection, labels in _identity_class_counts(bank, predictions, hard, vae.spec).items():
            destination = classes.setdefault(projection, {})
            for label, counts in labels.items():
                bucket = destination.setdefault(label, {"count": 0, "correct": 0})
                bucket["count"] += counts["count"]
                bucket["correct"] += counts["correct"]
        seen += len(chosen)
    # Batch-weighted evaluation matches the declared group-balanced training
    # loss. Row/identity counts additionally retain actual coverage separately.
    weighted = {key: value / seen for key, value in totals.items()}
    for projection, labels in classes.items():
        supported = []
        for counts in labels.values():
            counts["accuracy"] = counts["correct"] / counts["count"] if counts["count"] else None
            if counts["count"]:
                supported.append(counts["accuracy"])
        total = sum(counts["count"] for counts in labels.values())
        weighted[f"{projection}/count_weighted_identity_accuracy"] = (
            sum(counts["correct"] for counts in labels.values()) / total if total else 0.0)
        weighted[f"{projection}/macro_supported_class_accuracy"] = sum(supported) / len(supported) if supported else 0.0
    return {"examples": seen, "coverage_counts": dict(coverage),
            "per_class_identity": classes,
            "pair_count_policy": "directed occupied pairs including native NONE diagonal",
            "batch_weighted_metrics": weighted}


def train_bridge(vae, bank: NativeProjectionBank, dataset_path, output, device,
                 steps=2000, batch_size=32, learning_rate=1e-3, seed=137, deadline=None,
                 *, hidden_dim=128, audit_thresholds=None, evaluate_every=100) -> dict:
    """Fit only bridge parameters, select on dev and consume audit exactly once.

    ``audit_thresholds`` maps metric names to explicit ``min`` and/or ``max``
    bounds. Without thresholds, a fitted bridge remains unvalidated. The result
    never authorizes native scoring or optimization, even after feature gates.
    ``deadline`` is an absolute ``time.monotonic`` time; a partial fit is not a pass.
    """
    identity = _vae_identity(vae)
    if (type(steps) is not int or steps < 1 or type(batch_size) is not int or batch_size < 1
            or type(evaluate_every) is not int or evaluate_every < 1
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("Invalid supervised bridge training budget")
    thresholds = dict(audit_thresholds or {})
    for metric, bounds in thresholds.items():
        if (not isinstance(metric, str) or not isinstance(bounds, dict) or not bounds
                or set(bounds) - {"min", "max"}
                or any(not math.isfinite(value) for value in bounds.values())):
            raise ValueError("Audit thresholds need named finite min/max bounds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    before, bank_before = state_digest(vae.state_dict()), state_digest(bank.state_dict())
    data, manifest, manifest_sha = _load_dataset(dataset_path, vae)
    split_rows = {split: [i for i, value in enumerate(data["split"]) if value == split] for split in SPLITS}
    if any(not rows for rows in split_rows.values()):
        raise ValueError("All bridge train/dev/audit splits require accepted examples")
    device = torch.device(device)
    vae.to(device)
    bank.to(device).eval().requires_grad_(False)
    # fork_rng prevents bridge initialization from advancing the VAE campaign's
    # process RNG. Sampling below uses its own CPU generator.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        with torch.no_grad():
            _, feature = vae.decode_with_features(data["z"][:1].to(device))
        bridge = HiddenEmbeddingBridge(bank, atom_input_dim=feature.atom_hidden.shape[-1],
            pair_input_dim=feature.bond_input.shape[-1], charge_input_dim=feature.charge_input.shape[-1],
            hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=learning_rate, weight_decay=1e-4)
    bridge_ids = {id(parameter) for parameter in bridge.parameters()}
    if ({id(parameter) for group in optimizer.param_groups for parameter in group["params"]} != bridge_ids
            or bridge_ids & {id(parameter) for parameter in vae.parameters()}):
        raise RuntimeError("Supervised optimizer contains non-bridge parameters")
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    training = dict(version=TRAINING_VERSION, bridge_version=BRIDGE_VERSION,
        checkpoint=identity, dataset_manifest_sha256=manifest_sha, steps=steps,
        batch_size=batch_size, learning_rate=learning_rate, weight_decay=1e-4,
        seed=seed, evaluate_every=evaluate_every, hidden_dim=hidden_dim,
        selection="minimum dev batch-weighted group-balanced feature loss; earliest tie",
        audit_thresholds=thresholds, desired_score_labels=False,
        trainable_scope="HiddenEmbeddingBridge parameters only")
    _write_json(output / "training-plan.json", training)
    best_loss, best_step, history, completed = math.inf, None, [], 0
    result = dict(status="running_not_passed", native_validation="not_run",
                  optimization="not_run", audit_consumed=False)
    started = time.monotonic()

    def checkpoint(step, metrics):
        torch.save(dict(version=TRAINING_VERSION, bridge_configuration=bridge.configuration(),
            bridge_state={key: value.detach().cpu() for key, value in bridge.state_dict().items()},
            bank_metadata=bank.metadata(), bank_state={key: value.detach().cpu() for key, value in bank.state_dict().items()},
            vae_checkpoint=identity, vae_state_sha256=before, dataset_manifest_sha256=manifest_sha,
            training=training, step=step, dev=metrics), output / "best-bridge.pt")

    try:
        for step in range(1, steps + 1):
            _check_deadline(deadline)
            bridge.train()
            positions = torch.randint(len(split_rows["train"]), (batch_size,), generator=generator).tolist()
            rows = [split_rows["train"][position] for position in positions]
            _, targets, predictions = _forward_rows(vae, bank, bridge, data, rows, device)
            loss, metrics = bank.loss(predictions, targets)
            if not bool(loss.isfinite()):
                raise FloatingPointError("Nonfinite supervised bridge loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None and not bool(parameter.grad.isfinite().all()) for parameter in bridge.parameters()):
                raise FloatingPointError("Nonfinite supervised bridge gradient")
            optimizer.step()
            completed = step
            if step == 1 or step % evaluate_every == 0 or step == steps:
                dev = _evaluate(vae, bank, bridge, data, split_rows["dev"], device, batch_size, deadline)
                record = dict(step=step, train_loss=float(loss.detach()), dev=dev)
                history.append(record)
                with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                dev_loss = dev["batch_weighted_metrics"]["loss"]
                if dev_loss < best_loss:
                    best_loss, best_step = dev_loss, step
                    checkpoint(step, dev)
        saved = torch.load(output / "best-bridge.pt", map_location=device, weights_only=True)
        bridge.load_state_dict(saved["bridge_state"], strict=True)
        bridge.eval().requires_grad_(False)
        _check_deadline(deadline)
        # The sealed threshold plan predates this single held-out audit. No
        # audit metric influences checkpoint selection or optimizer updates.
        result["audit_consumed"] = True
        audit = _evaluate(vae, bank, bridge, data, split_rows["audit"], device, batch_size, deadline)
        checks = {}
        for name, bounds in thresholds.items():
            if name not in audit["batch_weighted_metrics"]:
                raise ValueError("Unknown predeclared audit metric: " + name)
            value = audit["batch_weighted_metrics"][name]
            checks[name] = all(value >= bound if kind == "min" else value <= bound for kind, bound in bounds.items())
        result.update(status=("passed_feature_audit_not_native_validated" if checks and all(checks.values())
                             else "failed_feature_audit" if checks else "trained_unvalidated"),
                      audit=audit, audit_checks=checks, best_step=best_step,
                      best_dev_loss=best_loss, checkpoint_path=str(output / "best-bridge.pt"),
                      checkpoint_sha256=sha256_file(output / "best-bridge.pt"))
    except TimeoutError as error:
        result.update(status="partial_runtime_not_passed", error=str(error))
    except Exception as error:
        result.update(status="failed_not_passed", error_type=type(error).__name__, error=str(error))
        _write_json(output / "training-result.json", {**result, "completed_steps": completed})
        raise
    finally:
        unchanged = (state_digest(vae.state_dict()) == before and state_digest(bank.state_dict()) == bank_before
                     and _vae_identity(vae) == identity)
        result.update(completed_steps=completed, requested_steps=steps, elapsed_seconds=time.monotonic() - started,
                      vae_checkpoint=identity, vae_state_sha256=before, frozen_vae_and_bank_unchanged=unchanged,
                      dataset_manifest_sha256=manifest_sha, training_plan_sha256=sha256_file(output / "training-plan.json"),
                      bridge_parameter_count=sum(parameter.numel() for parameter in bridge.parameters()))
        if not unchanged:
            result["status"] = "failed_frozen_identity"
        _write_json(output / "training-result.json", result)
        if not unchanged:
            raise RuntimeError("Frozen VAE or native projection bank changed while fitting bridge")
    return result
