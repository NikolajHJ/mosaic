"""Compact, grouped VAE-training-only DiffSBDD chemical supervision.

All views target the natural hard graph actually decoded at that latent. Invalid
and ineligible views remain in the immutable attempt denominator. Split groups
are connected components of source identities, decoded identities and scaffolds,
constructed before any split is assigned or head fitting begins.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch

from latentmol.chemistry import encode_smiles
from latentmol.chemistry_classification import classify_repair_graph
from latentmol.data import GraphDataset, collate_graphs, sha256_file
from latentmol.representation import HardGraphBatch
from latentmol.train import state_digest
from .bridge import ATOMIC_NUMBER, PILOT_ELEMENTS


DATA_VERSION = "diffsbdd-decoded-graph-grouped-pairs-v1"
EPOCH39_SHA256 = "e5d7f99b9eb0fe04cd8b6b800e4bd90b6f1c24a65368216ce384a95061a7f3b0"
SPLITS = ("train", "dev", "audit")
VIEW_NAMES = ("posterior_mean", "posterior_draw_1", "posterior_draw_2", "local_rms_0.02", "local_rms_0.05")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("DiffSBDD mapping internal runtime deadline reached")


def vae_identity(vae):
    identity = getattr(vae, "diffsbdd_checkpoint_identity", None)
    if identity != {"epoch": 39, "sha256": EPOCH39_SHA256}:
        raise ValueError("DiffSBDD mapping requires caller-authenticated completed epoch-39 VAE")
    if vae.training or any(p.requires_grad for p in vae.parameters()):
        raise ValueError("The VAE must be frozen in eval mode")
    return dict(identity)


def deterministic_seed(seed, *parts):
    encoded = json.dumps([seed, *parts], separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63 - 1)


def scaffold_identity(identity):
    molecule = Chem.MolFromSmiles(identity)
    if molecule is None:
        raise ValueError("Scaffold requires a sanitized canonical molecular identity")
    # The empty scaffold is shared by acyclic molecules. Do not replace it with
    # the molecule identity: that would introduce a hidden scaffold exception.
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)


def partition_groups(records, seed=17):
    """Union every source/decoded identity and scaffold, then hash components.

    Each record represents a source root and has ``source_identity``,
    ``source_scaffold`` and ``decoded`` entries containing identity/scaffold.
    Identity keys deliberately share a namespace for source and decoded forms.
    """
    count = len(records)
    parent, owners = list(range(count)), {}

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    keys_by_root = []
    for index, record in enumerate(records):
        keys = {("identity", record["source_identity"]), ("scaffold", record["source_scaffold"])}
        for decoded in record.get("decoded", []):
            if decoded.get("identity") is not None:
                keys.update((("identity", decoded["identity"]), ("scaffold", decoded["scaffold"])))
        keys_by_root.append(keys)
        for key in keys:
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index
    components = {}
    for index, keys in enumerate(keys_by_root):
        components.setdefault(find(index), set()).update(keys)
    assignment = {}
    for root, keys in components.items():
        blob = json.dumps(sorted(keys), separators=(",", ":"), ensure_ascii=True).encode()
        group = hashlib.sha256(blob).hexdigest()
        quantile = int(hashlib.sha256(f"{seed}|{group}".encode()).hexdigest(), 16) / 2**256
        assignment[root] = (group, "train" if quantile < 0.8 else "dev" if quantile < 0.9 else "audit")
    return [assignment[find(index)] for index in range(count)]


def reserved_identities(data):
    """Read validation/test role metadata without decoding/evaluating graphs."""
    cache = data.cache_dir
    if data.manifest["dataset_format_version"] == 1:
        with np.load(cache / "dataset.npz", allow_pickle=False) as archive:
            return set(archive["smiles"][archive["split"] != 0].tolist())
    labels = np.load(cache / "split.npy", mmap_mode="r", allow_pickle=False)
    offsets = np.load(cache / "smiles_offsets.npy", mmap_mode="r", allow_pickle=False)
    result = set()
    with (cache / "smiles.utf8").open("rb") as stream:
        for index in np.flatnonzero(labels != 0):
            begin, end = int(offsets[index]), int(offsets[index + 1])
            stream.seek(begin)
            result.add(stream.read(end - begin).decode("utf-8"))
    return result


def assess_graph(hard, spec, *, heavy_atoms=None):
    """No charge cancellation, fragment retention, repair or atom replacement."""
    if hard.atoms.shape != (1, spec.max_nodes):
        raise ValueError("Assess exactly one graph at a time")
    graph = hard.to("cpu")
    assessment = classify_repair_graph(graph.atoms[0], graph.charges[0], graph.bonds[0], spec)
    identity = assessment.get("smiles") if assessment["strict_teacher_valid"] else None
    record = dict(classification=assessment, identity=identity, scaffold=None, eligible=False, reason=None)
    if identity is not None:
        record["scaffold"] = scaffold_identity(identity)
    active = graph.atoms[0] != 0
    if not assessment["strict_teacher_valid"]:
        record["reason"] = "invalid:" + str(assessment["reason"])
    elif not bool(active.any()) or (heavy_atoms is not None and int(active.sum()) != heavy_atoms):
        record["reason"] = "changed_or_ineligible_atom_count"
    elif bool((graph.charges[0, active] != spec.formal_charges.index(0)).any()):
        record["reason"] = "charged_atom"
    elif any(spec.atomic_numbers[int(index)] not in {ATOMIC_NUMBER[x] for x in PILOT_ELEMENTS}
             for index in graph.atoms[0, active]):
        record["reason"] = "unsupported_element"
    elif len(Chem.GetMolFrags(Chem.MolFromSmiles(identity))) != 1:
        record["reason"] = "disconnected_graph"
    else:
        record["eligible"] = True
    return record


def slot_to_canonical(hard, spec, identity):
    """Return a fully verified exact map, preserving arbitrary/symmetric slots."""
    graph = hard.to("cpu")
    active = torch.nonzero(graph.atoms[0] != 0, as_tuple=False).flatten().tolist()
    encoded = encode_smiles(identity, spec)
    canonical = Chem.MolFromSmiles(encoded["smiles"])
    rw = Chem.RWMol()
    for slot in active:
        atom = Chem.Atom(spec.atomic_numbers[int(graph.atoms[0, slot])])
        atom.SetFormalCharge(spec.formal_charges[int(graph.charges[0, slot])])
        rw.AddAtom(atom)
    for left, slot in enumerate(active):
        for right in range(left + 1, len(active)):
            name = spec.bond_names[int(graph.bonds[0, slot, active[right]])]
            if name != "NONE":
                rw.AddBond(left, right, getattr(Chem.BondType, name))
                if name == "AROMATIC":
                    rw.GetAtomWithIdx(left).SetIsAromatic(True)
                    rw.GetAtomWithIdx(right).SetIsAromatic(True)
                    rw.GetBondBetweenAtoms(left, right).SetIsAromatic(True)
    molecule = rw.GetMol()
    Chem.SanitizeMol(molecule)
    mapping = canonical.GetSubstructMatch(molecule, useChirality=False)
    if len(mapping) != len(active) or len(set(mapping)) != len(active):
        raise ValueError("No complete exact canonical atom mapping")
    result = [-1] * spec.max_nodes
    for index, slot in enumerate(active):
        native = mapping[index]
        result[slot] = native
        if (int(encoded["atom_types"][native]) != int(graph.atoms[0, slot])
                or int(encoded["charges"][native]) != int(graph.charges[0, slot])):
            raise ValueError("Canonical mapping changes element or charge")
        for other_index, other_slot in enumerate(active):
            if int(encoded["bond_types"][native, mapping[other_index]]) != int(graph.bonds[0, slot, other_slot]):
                raise ValueError("Canonical mapping changes typed bond")
    return result


def latent_views(mu, logvar, source_identity, seed=17):
    """Five saved-seed views; local displacements have the specified exact RMS."""
    result = []
    for name in VIEW_NAMES:
        view_seed = deterministic_seed(seed, source_identity, name)
        generator = torch.Generator(device="cpu").manual_seed(view_seed)
        noise = torch.randn(mu.shape, generator=generator).to(mu)
        if name == "posterior_mean":
            value = mu
        elif name.startswith("posterior_draw"):
            value = mu + noise * (0.5 * logvar).exp()
        else:
            scale = float(name.rsplit("_", 1)[-1])
            value = mu + scale * noise / noise.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
        result.append((name, view_seed, value))
    return result


def prepare_dataset(vae, data_path, output, *, requested_sources=20000,
                    max_source_attempts=100000, seed=17, heavy_atoms=None,
                    deadline=None, artifact_cap_bytes=2 * 1024**3):
    """Select first eligible manifest-order sources, record five natural decodes.

    A source shortage, timeout, empty split or artifact-cap failure is preserved
    and stops before fitting. Failed decoded views are never replaced. This
    function expects the authenticated frozen VAE supplied by the asset runner.
    """
    identity = vae_identity(vae)
    if (type(requested_sources) is not int or requested_sources < 1
            or type(max_source_attempts) is not int or max_source_attempts < requested_sources
            or (heavy_atoms is not None and (type(heavy_atoms) is not int or heavy_atoms < 1))):
        raise ValueError("Invalid fixed corpus selection bounds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    before = state_digest(vae.state_dict())
    data = GraphDataset(data_path, "train")
    if data.spec != vae.spec:
        raise ValueError("Training corpus and VAE schemas differ")
    lineage = dict(dataset_sha256=data.manifest["dataset_sha256"], split_sha256=data.manifest["split_sha256"])
    if getattr(vae, "diffsbdd_training_data_identity", None) != lineage:
        raise ValueError("Paired corpus must match the authenticated VAE training dataset and split identity")
    holdouts = reserved_identities(data)
    device = next(vae.parameters()).device
    source_records, roots, samples, attempts, seen = [], [], [], [], set()
    manifest = dict(version=DATA_VERSION, status="preparing_not_passed", vae_checkpoint=identity,
        source_dataset_sha256=data.manifest["dataset_sha256"], spec_fingerprint=vae.spec.fingerprint(),
        requested_sources=requested_sources, max_source_attempts=max_source_attempts,
        heavy_atoms=heavy_atoms, seed=seed, view_names=list(VIEW_NAMES), source_data_identity=lineage,
        reserved_identities_sha256=hashlib.sha256(json.dumps(sorted(holdouts)).encode()).hexdigest(),
        artifact_cap_bytes=artifact_cap_bytes, labels="natural hard graph actually decoded at each z",
        grouping="connected components of source identity, decoded identity and Bemis-Murcko scaffold")
    try:
        with torch.no_grad():
            for index in range(min(len(data), max_source_attempts)):
                check_deadline(deadline)
                source = data.smiles[index]
                graph = collate_graphs([data[index]])
                admission = assess_graph(graph, vae.spec, heavy_atoms=heavy_atoms)
                reason = admission["reason"]
                if source in seen:
                    reason = "duplicate_source_identity"
                elif source in holdouts:
                    reason = "reserved_vae_identity"
                global_index = int(data._indices[index]) if data._indices is not None else None
                source_records.append(dict(index=index, global_manifest_index=global_index, identity=source,
                    accepted=reason is None, reason=reason))
                if reason is not None:
                    continue
                seen.add(source)
                source_count = int((graph.atoms != 0).sum())
                root = dict(source_identity=source, source_scaffold=scaffold_identity(source), decoded=[])
                mu, logvar = vae.encode(graph.to(device).to_relaxed(vae.spec))
                for name, view_seed, z in latent_views(mu, logvar, source, seed):
                    check_deadline(deadline)
                    hard = vae.decode(z).harden(vae.spec).to("cpu")
                    assessment = assess_graph(hard, vae.spec, heavy_atoms=source_count)
                    root["decoded"].append(dict(identity=assessment["identity"], scaffold=assessment["scaffold"]))
                    reason = assessment["reason"]
                    if assessment["identity"] in holdouts:
                        reason = "reserved_vae_decoded_identity"
                    record = dict(attempt=len(attempts), root_index=len(roots), source_index=index,
                        source_identity=source, view=name, seed=view_seed, assessment=assessment,
                        accepted=False, rejection_reason=reason)
                    if reason is None:
                        alignment = slot_to_canonical(hard, vae.spec, assessment["identity"])
                        record.update(accepted=True, sample_index=len(samples))
                        samples.append(dict(z=z.detach().cpu().reshape(-1).float(),
                            atoms=hard.atoms[0].to(torch.uint8), charges=hard.charges[0].to(torch.uint8),
                            bonds=hard.bonds[0].to(torch.uint8), active=hard.atoms[0] != 0,
                            slot_to_canonical=torch.tensor(alignment, dtype=torch.int16),
                            source_identity=source, source_scaffold=root["source_scaffold"],
                            identity=assessment["identity"], scaffold=assessment["scaffold"],
                            root_index=len(roots), view=name, seed=view_seed))
                    attempts.append(record)
                roots.append(root)
                if len(roots) == requested_sources:
                    break
        if len(roots) != requested_sources:
            raise ValueError(f"Insufficient eligible source identities: {len(roots)}/{requested_sources}; no fit permitted")
        assignments = partition_groups(roots, seed)
        for row in samples:
            row["group"], row["split"] = assignments[row["root_index"]]
        for record in attempts:
            record["group"], record["split"] = assignments[record["root_index"]]
        tensor_keys = ("z", "atoms", "charges", "bonds", "active", "slot_to_canonical")
        if not samples:
            raise ValueError("No eligible naturally decoded views; no fit permitted")
        payload = {key: torch.stack([row[key] for row in samples]) for key in tensor_keys}
        payload.update({key: [row[key] for row in samples] for key in samples[0] if key not in tensor_keys})
        split_counts = Counter(payload["split"])
        if any(split_counts[split] == 0 for split in SPLITS):
            raise ValueError("Grouped corpus has an empty split; preserve grouping and stop before fit")
        estimated_tensor_bytes = sum(t.numel() * t.element_size() for t in payload.values() if torch.is_tensor(t))
        if estimated_tensor_bytes > artifact_cap_bytes * 0.8:
            raise ValueError("Compact tensor payload exceeds artifact admission; no unbounded cache write")
        torch.save(payload, output / "samples.pt")
        manifest.update(status="prepared_not_fitted", split_counts=dict(split_counts), accepted=len(samples),
                        samples_sha256=sha256_file(output / "samples.pt"),
                        group_counts=dict(Counter(split for _, split in set(assignments))),
                        atom_count_coverage={split: dict(Counter(str(int(row["active"].sum()))
                            for row in samples if row["split"] == split)) for split in SPLITS})
    except Exception as error:
        manifest.update(status="failed_not_fitted", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        with (output / "source-attempts.jsonl").open("x", encoding="utf-8") as stream:
            for record in source_records:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        with (output / "attempts.jsonl").open("x", encoding="utf-8") as stream:
            for record in attempts:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        after = state_digest(vae.state_dict())
        manifest.update(source_attempts=len(source_records), selected_sources=len(roots),
            view_attempts=len(attempts), accepted_views=len(samples),
            rejection_counts=dict(Counter(r["rejection_reason"] for r in attempts if not r["accepted"])),
            vae_state_sha256_before=before, vae_state_sha256_after=after,
            attempts_sha256=sha256_file(output / "attempts.jsonl"),
            source_attempts_sha256=sha256_file(output / "source-attempts.jsonl"))
        if before != after or vae_identity(vae) != identity:
            manifest["status"] = "failed_frozen_vae_changed"
        actual_bytes = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
        manifest["artifact_bytes_before_manifest"] = actual_bytes
        if actual_bytes > artifact_cap_bytes - 1024**2:
            manifest["status"] = "failed_artifact_cap"
        write_json(output / "manifest.json", manifest)
    if manifest["status"] != "prepared_not_fitted":
        raise RuntimeError("Paired-data preparation failed frozen state or artifact admission")
    return manifest


def load_dataset(path, vae):
    """Authenticate compact payload and verify source/identity/scaffold separation."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("version") != DATA_VERSION or manifest.get("status") != "prepared_not_fitted"
            or manifest.get("vae_checkpoint") != vae_identity(vae)
            or manifest.get("spec_fingerprint") != vae.spec.fingerprint()
            or manifest.get("vae_state_sha256_after") != state_digest(vae.state_dict())
            or any(sha256_file(path / name) != manifest.get(key) for name, key in
                   (("samples.pt", "samples_sha256"), ("attempts.jsonl", "attempts_sha256"),
                    ("source-attempts.jsonl", "source_attempts_sha256")))):
        raise ValueError("Paired corpus/checkpoint/schema identity authentication failed")
    payload = torch.load(path / "samples.pt", map_location="cpu", weights_only=True)
    count = manifest["accepted"]
    expected = {"z", "atoms", "charges", "bonds", "active", "slot_to_canonical", "source_identity",
                "source_scaffold", "identity", "scaffold", "root_index", "view", "seed", "group", "split"}
    if set(payload) != expected or any(len(value) != count for value in payload.values()):
        raise ValueError("Compact paired payload dimensions/keys disagree with manifest")
    owners = {}
    for index, split in enumerate(payload["split"]):
        if split not in SPLITS:
            raise ValueError("Unknown bridge split")
        for kind, value in (("group", payload["group"][index]),
                            ("identity", payload["source_identity"][index]),
                            ("identity", payload["identity"][index]),
                            ("scaffold", payload["source_scaffold"][index]),
                            ("scaffold", payload["scaffold"][index])):
            key = (kind, value)
            if key in owners and owners[key] != split:
                raise ValueError("Paired data leaks source/decoded identity or scaffold across splits")
            owners[key] = split
    if not torch.equal(payload["active"], payload["atoms"] != 0):
        raise ValueError("Stored natural graph occupancy disagrees with labels")
    if not bool(torch.isfinite(payload["z"]).all()):
        raise ValueError("Stored latents contain nonfinite values")
    return payload, manifest, sha256_file(path / "manifest.json")
