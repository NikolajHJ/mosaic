"""Prepare small, frozen-encoder structural-prior banks from cached real molecules.

This CPU preparation performs no training and makes no synthesis-route claim.
The bounded SMILES view reads the ChEMBL training component of the existing
MOSES/ChEMBL combined cache. It does not import another branch's corpus loader,
materialize its full split, or decompress all graphs. Selected molecules use
the project's existing chemistry codec and encoder input-ordering policy.
Full corpus byte authentication is not claimed by this exploratory reader;
manifest identity, mapped-file sizes and selected offsets are recorded/checked.
Run substantive preparation in an allocated CPU job, not on the login node.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from latentmol.chemistry import MoleculeRejected, encode_smiles, inspect_graph
from latentmol.design.latent_protenix import _model_identity
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.graph_policy import graph_from_decoder
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


VERSION = "real-molecule-latent-anchor-bank-v1"
ALIGNED_VERSION = "masked-frozen-decoder-anchor-bank-v2"


def cached_training_rows(dataset, spec, *, seed=13113, scan_limit=2048, occupied_atoms=25):
    """Read at most scan_limit seeded source rows; return eligible codec records.

    This cache's arrays contain its ChEMBL component, while MOSES is a separate
    referenced cache. Only split==0 rows participate, so no validation/test
    molecule is used as an anchor. No source graph or cache is rewritten.
    """
    from rdkit import Chem
    from rdkit.Contrib.SA_Score import sascorer

    dataset = Path(dataset)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("version") != "moses-chembl-combined-v1"
            or GraphSpec.from_dict(manifest["spec"]).fingerprint() != spec.fingerprint()
            or manifest.get("spec_fingerprint") != spec.fingerprint()):
        raise ValueError("Library input must be the shared-schema MOSES/ChEMBL combined cache")
    if not 1 <= scan_limit <= 2048 or not 1 <= occupied_atoms <= spec.max_nodes:
        raise ValueError("Library scan must be bounded to 2048 examples with compatible atom count")
    for name in ("atoms.npy", "split.npy", "smiles_offsets.npy", "smiles.utf8"):
        if (dataset / name).stat().st_size != manifest["files"][name]["bytes"]:
            raise ValueError(f"Library source file size differs from manifest: {name}")
    atoms = np.load(dataset / "atoms.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(dataset / "split.npy", mmap_mode="r", allow_pickle=False)
    offsets = np.load(dataset / "smiles_offsets.npy", mmap_mode="r", allow_pickle=False)
    count = sum(manifest["sources"]["chembl"]["counts"].values())
    if (atoms.shape != (count, spec.max_nodes) or labels.shape != (count,)
            or offsets.shape != (count + 1,) or atoms.dtype != np.int8
            or labels.dtype != np.int8 or offsets.dtype != np.int64
            or int(offsets[0]) != 0 or int(offsets[-1]) != (dataset / "smiles.utf8").stat().st_size):
        raise ValueError("Library source arrays or SMILES offsets do not match the cache schema")
    selected = np.random.default_rng(seed).choice(count, size=min(count, scan_limit), replace=False)
    records, seen, counts = [], set(), Counter()
    with (dataset / "smiles.utf8").open("rb") as stream:
        for index in selected:
            counts["source_rows_examined"] += 1
            if int(labels[index]) not in (0, 1, 2):
                raise ValueError("Selected library cache row has an incompatible split label")
            if int(labels[index]) != 0:
                counts["not_training"] += 1
                continue
            if int(np.count_nonzero(atoms[index])) != occupied_atoms:
                counts["different_atom_count"] += 1
                continue
            begin, end = int(offsets[index]), int(offsets[index + 1])
            if not 0 <= begin < end <= int(offsets[-1]) or end - begin > 100000:
                raise ValueError("Invalid selected SMILES offset")
            stream.seek(begin)
            raw = stream.read(end - begin).decode("utf-8")
            try:
                encoded = encode_smiles(raw, spec)
            except MoleculeRejected as error:
                counts["codec_rejection:" + error.reason] += 1
                continue
            if np.count_nonzero(encoded["atom_types"]) != occupied_atoms:
                raise ValueError("Cache atom count and current canonical codec disagree")
            if encoded["smiles"] in seen:
                counts["duplicate_identity"] += 1
                continue
            seen.add(encoded["smiles"])
            sa = float(sascorer.calculateScore(Chem.MolFromSmiles(encoded["smiles"])))
            if not math.isfinite(sa):
                raise ValueError("Nonfinite source-molecule SA")
            records.append(dict(index=int(index), source="chembl_training_component",
                                source_smiles=raw, sa_score=sa, **encoded))
    counts["eligible_distinct_training_molecules"] = len(records)
    return records, dict(dataset=str(dataset), dataset_sha256=manifest["dataset_sha256"],
        manifest_sha256=file_sha256(manifest_path), source_component="chembl",
        scan_seed=seed, scan_limit=scan_limit, counts=dict(counts),
        identity_scope="manifest identity/file sizes/selected offsets; corpus file digests not rehashed",
        canonicalization="existing encode_smiles and frozen encoder's native ordering policy")


def select_banks(records, *, bank_size=32, sa_ceiling=3.):
    """Disjoint equal-size banks in deterministic scan order, without decode selection.

    The control has no SA threshold; removal of positive-bank identities is the
    only additional control restriction. Its measured SA distribution is saved.
    """
    if not 1 <= bank_size <= 32 or not math.isfinite(sa_ceiling):
        raise ValueError("Library bank size must be in 1..32 and SA ceiling finite")
    easy = [row for row in records if row["sa_score"] <= sa_ceiling]
    count = min(bank_size, len(easy), len(records) // 2)
    if count < 1:
        raise ValueError("Bounded scan supplied no usable paired low-SA/control bank")
    low = easy[:count]
    excluded = {row["smiles"] for row in low}
    controls = [row for row in records if row["smiles"] not in excluded][:count]
    return low, controls


def _hard_batch(rows):
    return HardGraphBatch(*(torch.tensor(np.stack([row[key] for row in rows]), dtype=torch.long)
                            for key in ("atom_types", "charges", "bond_types")))


@torch.no_grad()
def _decode_diagnostic(model, latent, source, occupied_atoms):
    """Retain raw and fixed-occupancy argmax outcomes; no repair or filtering."""
    from rdkit import Chem
    from rdkit.Contrib.SA_Score import sascorer

    decoded = model.decode(latent[None])
    if any(not bool(torch.isfinite(value).all()) for value in (decoded.atoms, decoded.charges, decoded.bonds)):
        raise ValueError("Nonfinite frozen decoder output in library preparation")
    raw = decoded.harden(model.spec)
    active = torch.arange(model.spec.max_nodes) < occupied_atoms
    graph = graph_from_decoder(decoded, model.spec, fixed_active=active)
    fixed = HardGraphBatch(graph.atoms.argmax(-1), graph.charges.argmax(-1), graph.bonds.argmax(-1))
    result = {}
    for name, hard in (("raw", raw), ("fixed_occupancy", fixed)):
        check = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], model.spec)
        mol = Chem.MolFromSmiles(check.smiles) if check.smiles else None
        result[name] = dict(valid=check.smiles is not None, smiles=check.smiles,
            failure=check.failure_reason, source_identity_reconstructed=check.smiles == source["smiles"],
            heavy_atoms=int((hard.atoms[0] != 0).sum()),
            sa_score=float(sascorer.calculateScore(mol)) if mol is not None else None)
    target = _hard_batch([source])
    pair = active[:, None] & active[None, :] & torch.triu(torch.ones_like(fixed.bonds[0], dtype=torch.bool), 1)
    result["fixed_label_accuracy"] = dict(
        atoms=float((fixed.atoms[0, active] == target.atoms[0, active]).float().mean()),
        charges=float((fixed.charges[0, active] == target.charges[0, active]).float().mean()),
        bonds=float((fixed.bonds[0][pair] == target.bonds[0][pair]).float().mean()) if bool(pair.any()) else None)
    return result


@torch.no_grad()
def encode_banks(model, low, controls, *, checkpoint_sha256, selection, occupied_atoms=25,
                 match_control_norm=True):
    """Encode posterior means, optionally norm-match controls, and diagnose all decodes."""
    model.eval().requires_grad_(False)
    before = _model_identity(model)
    batches = []
    for rows in (low, controls):
        mean, logvar = model.encode(_hard_batch(rows).to_relaxed(model.spec))
        if mean.ndim != 2 or not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(logvar).all()):
            raise ValueError("Nonfinite or incompatible frozen encoder output")
        batches.append(mean.detach().cpu())
    factor = 1.
    if match_control_norm:
        positive_norm = batches[0].double().norm(dim=-1).mean()
        control_norm = batches[1].double().norm(dim=-1).mean()
        if float(control_norm) <= 1e-12 or float(positive_norm) <= 1e-12:
            raise ValueError("Cannot match flat/zero latent-bank norms")
        factor = float(positive_norm / control_norm)
    used = (batches[0], batches[1] * factor)
    if any(not bool(points.isfinite().all()) for points in used):
        raise ValueError("Nonfinite norm-matched library latents")
    # Both arms use the same distance scale. It is a typical RMS-per-coordinate
    # separation of low-SA anchor means, not a synthesis-probability calibration.
    pair_distances = torch.pdist(used[0].double()) / math.sqrt(used[0].shape[-1])
    scale = max(float(pair_distances.median()), 1e-4) if len(pair_distances) else 1.
    outputs = {}
    for name, rows, original, latent in zip(("low-sa", "unfiltered"), (low, controls), batches, used):
        molecules = []
        for row, mu, point in zip(rows, original, latent):
            details = {key: row[key] for key in ("index", "source", "smiles", "source_smiles", "sa_score")}
            details["mu_norm"] = float(mu.norm())
            details["used_latent_norm"] = float(point.norm())
            details["mu_decode"] = _decode_diagnostic(model, mu, row, occupied_atoms)
            details["used_decode"] = (_decode_diagnostic(model, point, row, occupied_atoms)
                                      if name == "unfiltered" and factor != 1. else details["mu_decode"])
            molecules.append(details)
        outputs[name] = dict(version=VERSION, checkpoint_sha256=checkpoint_sha256,
            graph_spec=model.spec.to_dict(), bank=name, latents=latent.tolist(), encoder_means=original.tolist(),
            scale=scale, scale_definition="median pairwise RMS-coordinate distance in low-SA bank; shared across arms",
            occupied_atoms=occupied_atoms, selection=selection, molecules=molecules,
            norm_matching=dict(enabled=bool(match_control_norm), control_multiplier=factor,
                policy="single scalar matches control mean L2 norm to low-SA mean L2 norm; transformed control decodes recorded"),
            interpretation="initialization/structural prior from real training molecules; not route-confirmed synthesis",
            frozen_state_sha256=before)
    if _model_identity(model) != before:
        raise ValueError("Frozen encoder/decoder state changed during library preparation")
    return outputs


def prepare_library(dataset, checkpoint, checkpoint_sha256, output, *, seed=13113, scan_limit=2048,
                    bank_size=32, sa_ceiling=3., occupied_atoms=25, match_control_norm=True):
    started = perf_counter()
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError("Preserve existing library output")
    if file_sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("Library decoder checkpoint SHA differs")
    model, _ = load_model(checkpoint, "cpu")
    rows, provenance = cached_training_rows(dataset, model.spec, seed=seed, scan_limit=scan_limit,
                                             occupied_atoms=occupied_atoms)
    low, controls = select_banks(rows, bank_size=bank_size, sa_ceiling=sa_ceiling)
    selection = dict(**provenance, requested_bank_size=bank_size, actual_bank_size=len(low),
        low_sa_ceiling=sa_ceiling, distinct_source_identities=True, disjoint_banks=True,
        control_selection="same heavy atom count; no SA threshold; exclude selected low-SA identities",
        decoded_validity_used_for_selection=False,
        shortfall=bank_size - len(low))
    banks = encode_banks(model, low, controls, checkpoint_sha256=checkpoint_sha256,
        selection=selection, occupied_atoms=occupied_atoms, match_control_norm=match_control_norm)
    output.mkdir(parents=True, exist_ok=False)
    for name, bank in banks.items():
        with (output / (name + ".json")).open("x", encoding="utf-8") as stream:
            json.dump(bank, stream, indent=2, allow_nan=False)
            stream.write("\n")
    summary = dict(status="completed", checkpoint_sha256=checkpoint_sha256, selection=selection,
        seconds=perf_counter() - started, model_state_unchanged=True,
        banks={name: dict(count=len(bank["molecules"]), path=str(output / (name + ".json")),
            source_mean_sa=sum(row["sa_score"] for row in bank["molecules"]) / len(bank["molecules"]),
            fixed_decode_valid=sum(row["used_decode"]["fixed_occupancy"]["valid"] for row in bank["molecules"]))
            for name, bank in banks.items()})
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def remap_source_slots(source, active_slots, spec):
    """Place the same source graph in explicit decoder slots, without reordering z.

    This changes the reconstruction target only. The encoder mean is merely an
    initializer; a slot permutation is not assumed to have a latent equivalent.
    """
    slots = list(active_slots)
    if (not slots or len(slots) != len(set(slots))
            or any(type(i) is not int or not 0 <= i < spec.max_nodes for i in slots)):
        raise ValueError("Aligned bank requires distinct compatible active slots")
    original = np.flatnonzero(source["atom_types"])
    if len(original) != len(slots):
        raise ValueError("Source atom count differs from requested aligned occupancy")
    atoms = np.zeros(spec.max_nodes, dtype=np.int64)
    charges = np.full(spec.max_nodes, spec.formal_charges.index(0), dtype=np.int64)
    bonds = np.zeros((spec.max_nodes, spec.max_nodes), dtype=np.int64)
    atoms[slots] = np.asarray(source["atom_types"])[original]
    charges[slots] = np.asarray(source["charges"])[original]
    bonds[np.ix_(slots, slots)] = np.asarray(source["bond_types"])[np.ix_(original, original)]
    checked = inspect_graph(atoms, charges, bonds, spec)
    if checked.smiles != source["smiles"]:
        raise ValueError("Source slot remapping changed the intended molecular identity")
    return {**source, "atom_types": atoms, "charges": charges, "bond_types": bonds}


def _aligned_losses(decoded, targets, slots):
    """Per-example CE of the intended fixed-occupancy atom/charge/bond graph."""
    atom = F.cross_entropy(decoded.atoms[:, slots, 1:].transpose(1, 2),
                           targets.atoms[:, slots] - 1, reduction="none").mean(-1)
    charge = F.cross_entropy(decoded.charges[:, slots].transpose(1, 2),
                             targets.charges[:, slots], reduction="none").mean(-1)
    pairs = torch.triu_indices(len(slots), len(slots), 1, device=decoded.atoms.device)
    a, b = slots[pairs[0]], slots[pairs[1]]
    bond = (F.cross_entropy(decoded.symmetric_bond_logits[:, a, b].transpose(1, 2),
                            targets.bonds[:, a, b], reduction="none").mean(-1)
            if len(a) else atom.new_zeros(atom.shape))
    return atom + charge + bond


def _aligned_readout(decoded, index, spec, active, source_smiles):
    """Hard chemistry is a detached preparation diagnostic, outside the fit loss."""
    from rdkit import Chem
    from rdkit.Contrib.SA_Score import sascorer

    singleton = DecoderOutput(*(getattr(decoded, key)[index:index + 1]
                                for key in ("atoms", "charges", "bonds")))
    graph = graph_from_decoder(singleton, spec, fixed_active=active)
    hard = HardGraphBatch(*(getattr(graph, key).argmax(-1) for key in ("atoms", "charges", "bonds")))
    inspected = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], spec)
    mol = Chem.MolFromSmiles(inspected.smiles) if inspected.smiles else None
    return dict(valid=mol is not None, smiles=inspected.smiles, failure=inspected.failure_reason,
                source_identity_reconstructed=inspected.smiles == source_smiles,
                sa_score=float(sascorer.calculateScore(mol)) if mol is not None else None,
                heavy_atoms=int((hard.atoms[0] != 0).sum()))


def fit_aligned_latents(model, encoder_means, sources, active_slots, *, updates=160,
                        learning_rate=.1, decoded_sa_ceiling=None, diagnostic_every=10,
                        progress=None):
    """Bounded latent inversion through a frozen decoder; never train its weights.

    Every example receives the same update count. Retain the lowest-CE valid
    snapshot (prefer exact source identity) at regular diagnostics, rather than
    trusting the terminal alone. Positive/control runs use identical fit rules;
    only the explicit decoded-SA eligibility ceiling differs.
    """
    if (type(updates) is not int or not 0 <= updates <= 400
            or not math.isfinite(learning_rate) or not 0 < learning_rate <= 1
            or type(diagnostic_every) is not int or diagnostic_every < 1
            or (decoded_sa_ceiling is not None and not math.isfinite(decoded_sa_ceiling))):
        raise ValueError("Aligned fitting needs finite, bounded updates and learning rate")
    if progress is not None and not callable(progress):
        raise ValueError("Aligned fitting progress must be a callable")
    started = perf_counter()
    model.eval().requires_grad_(False)
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Frozen aligned-bank model already carries parameter gradients")
    identity = _model_identity(model)
    means = torch.as_tensor(encoder_means, dtype=torch.float32)
    if (means.ndim != 2 or not len(means) or len(means) != len(sources)
            or not bool(means.isfinite().all())):
        raise ValueError("Aligned fitting requires finite paired encoder means")
    targets = _hard_batch([remap_source_slots(row, active_slots, model.spec) for row in sources])
    active = torch.zeros(model.spec.max_nodes, dtype=torch.bool)
    active[list(active_slots)] = True
    slots = torch.tensor(list(active_slots), dtype=torch.long)
    points = means.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([points], lr=learning_rate)
    best, diagnostics = [None] * len(sources), [[] for _ in sources]
    initial, terminal = None, None
    for step in range(updates + 1):
        decoded = model.decode(points)
        if any(not bool(getattr(decoded, key).isfinite().all()) for key in ("atoms", "charges", "bonds")):
            raise ValueError("Nonfinite frozen decoder logits in aligned-bank inversion")
        losses = _aligned_losses(decoded, targets, slots)
        if not losses.requires_grad or not bool(losses.isfinite().all()):
            raise ValueError("Detached or nonfinite aligned-bank reconstruction loss")
        if initial is None:
            initial = losses.detach().tolist()
        if step % diagnostic_every == 0 or step == updates:
            with torch.no_grad():
                for i, source in enumerate(sources):
                    record = dict(step=step, reconstruction_loss=float(losses[i]),
                                  **_aligned_readout(decoded, i, model.spec, active, source["smiles"]))
                    diagnostics[i].append(record)
                    eligible = (record["valid"] and (decoded_sa_ceiling is None
                                                    or record["sa_score"] <= decoded_sa_ceiling))
                    rank = (not record["source_identity_reconstructed"], record["reconstruction_loss"], step)
                    if eligible and (best[i] is None or rank < best[i][0]):
                        best[i] = rank, points[i].detach().clone(), record
            if progress is not None:
                latest = [records[-1] for records in diagnostics]
                progress(dict(step=step, planned_updates=updates, sources=len(sources),
                    mean_reconstruction_loss=float(losses.detach().mean()),
                    current_valid=sum(record["valid"] for record in latest),
                    current_exact_source=sum(record["source_identity_reconstructed"] for record in latest),
                    retained_eligible=sum(record is not None for record in best),
                    elapsed_seconds=perf_counter() - started))
        terminal = losses.detach().tolist()
        if step == updates:
            break
        optimizer.zero_grad(set_to_none=True)
        gradient, = torch.autograd.grad(losses.sum(), points)
        if gradient.shape != points.shape or not bool(gradient.isfinite().all()):
            raise ValueError("Nonfinite or incompatible aligned-bank latent gradient")
        # Bound extreme rows without changing the number of updates per source.
        norms = gradient.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        points.grad = gradient * (5. / norms).clamp_max(1.)
        optimizer.step()
        if not bool(points.isfinite().all()):
            raise ValueError("Nonfinite aligned-bank latent update")
    if _model_identity(model) != identity or any(p.grad is not None for p in model.parameters()):
        raise ValueError("Frozen encoder/decoder changed during aligned-bank inversion")
    return dict(best=best, diagnostics=diagnostics, initial_loss=initial, terminal_loss=terminal,
                terminal_latents=points.detach(), frozen_state_sha256=identity,
                fit=dict(updates_per_source=updates, learning_rate=learning_rate,
                         diagnostic_every=diagnostic_every, gradient_row_norm_cap=5.,
                         objective="fixed-active real-atom CE + charge CE + upper-triangle typed-bond CE",
                         decoded_sa_ceiling=decoded_sa_ceiling, model_training=False))


def prepare_aligned_library(low_bank, control_bank, parent_panel, checkpoint,
                            checkpoint_sha256, output, *, bank_size=16, updates=160,
                            learning_rate=.1, decoded_sa_ceiling=3.5):
    """Prepare new equal-size aligned banks from immutable original source assets."""
    started = perf_counter()
    paths = [Path(low_bank), Path(control_bank), Path(parent_panel), Path(checkpoint)]
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve existing aligned library output")
    if not 1 <= bank_size <= 32:
        raise ValueError("Aligned bank size must be in 1..32")
    if file_sha256(paths[-1]) != checkpoint_sha256:
        raise ValueError("Aligned-bank checkpoint SHA differs")
    input_hashes = {str(p): file_sha256(p) for p in paths}
    model, _ = load_model(paths[-1], "cpu")
    model.eval().requires_grad_(False)
    identity = _model_identity(model)
    panel = json.loads(paths[2].read_text(encoding="utf-8"))
    if (panel.get("checkpoint_sha256") != checkpoint_sha256
            or panel.get("graph_spec") != model.spec.to_dict()
            or panel.get("decoder_sha256") != identity or not panel.get("points")):
        raise ValueError("Alignment parent panel differs from the frozen model/schema")
    slots = panel["points"][0]["active_slots"]
    if any(p["active_slots"] != slots for p in panel["points"]):
        raise ValueError("Alignment founders must share one explicit occupancy mask")
    original, fits, sources = {}, {}, {}
    for name, path in zip(("low-sa", "unfiltered"), paths[:2]):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("checkpoint_sha256") != checkpoint_sha256 or data.get("graph_spec") != model.spec.to_dict():
            raise ValueError("Source bank differs from the alignment checkpoint/schema")
        rows = []
        for row in data["molecules"]:
            encoded = encode_smiles(row["smiles"], model.spec)
            rows.append(dict(**encoded, source=row.get("source", "source_bank"),
                             index=row.get("index"), source_smiles=row.get("source_smiles", row["smiles"]),
                             sa_score=row["sa_score"]))
        means = torch.tensor(data["encoder_means"], dtype=torch.float32)
        if means.shape != (len(rows), model.config.latent):
            raise ValueError("Original encoder means do not match the frozen model")
        original[name], sources[name] = data, rows
        fits[name] = fit_aligned_latents(model, means, rows, slots, updates=updates,
            learning_rate=learning_rate,
            decoded_sa_ceiling=decoded_sa_ceiling if name == "low-sa" else None,
            progress=lambda record, bank=name: print(json.dumps(
                dict(event="aligned_bank_progress", bank=bank, **record), allow_nan=False), flush=True))
    # Eligibility is recorded after identical fitting; do not rescale fitted z.
    eligible, used_smiles = {}, set()
    for name in ("low-sa", "unfiltered"):
        eligible[name] = []
        for i, best in enumerate(fits[name]["best"]):
            if best is None:
                continue
            if best[2]["smiles"] in used_smiles:
                continue
            used_smiles.add(best[2]["smiles"])
            eligible[name].append((i, best))
    count = min(bank_size, *(len(eligible[name]) for name in eligible))
    selected = {name: eligible[name][:count] for name in eligible}
    low_points = torch.stack([best[1] for _, best in selected["low-sa"]]) if count else None
    pair = torch.pdist(low_points.double()) / math.sqrt(low_points.shape[-1]) if count > 1 else None
    scale = max(float(pair.median()), 1e-4) if pair is not None and len(pair) else 1.
    selection = dict(policy="source-graph latent inversion, actual-mask decoded eligibility",
        requested_bank_size=bank_size, actual_bank_size=count, shortfall=bank_size - count,
        low_sa_decoded_ceiling=decoded_sa_ceiling, control_sa_ceiling=None,
        same_fit_budget=True, source_split="unchanged source bank training molecules",
        active_slots=slots, disjoint_decoded_identities=True, post_fit_norm_scaling=False)
    outputs = {}
    for name in ("low-sa", "unfiltered"):
        fit = fits[name]
        molecules, latents, means = [], [], []
        for i, best in selected[name]:
            source, record = sources[name][i], best[2]
            molecules.append(dict(index=source["index"], source=source["source"],
                smiles=source["smiles"], source_smiles=source["source_smiles"], sa_score=source["sa_score"],
                decoded_smiles=record["smiles"], decoded_sa=record["sa_score"], fit_step=record["step"],
                source_identity_reconstructed=record["source_identity_reconstructed"],
                used_latent_norm=float(best[1].norm()),
                used_decode=dict(fixed_occupancy=record),
                diagnostics=fit["diagnostics"][i]))
            latents.append(best[1].tolist())
            means.append(original[name]["encoder_means"][i])
        outputs[name] = dict(version=ALIGNED_VERSION, bank=name,
            graph_spec=model.spec.to_dict(), checkpoint_sha256=checkpoint_sha256,
            active_slots=slots, occupied_atoms=len(slots), selection=selection, fit=fit["fit"],
            latents=latents, encoder_means=means, molecules=molecules, scale=scale,
            scale_definition="median pairwise RMS-coordinate distance in selected aligned low-SA bank; shared",
            norm_matching=dict(enabled=False, policy="no post-fit rescaling; actual-mask validity retained"),
            candidate_summary=dict(attempted=len(sources[name]), eligible_distinct=len(eligible[name]),
                valid_snapshots=sum(any(x["valid"] for x in d) for d in fit["diagnostics"]),
                initial_loss=fit["initial_loss"], terminal_loss=fit["terminal_loss"]),
            source_assets=input_hashes, frozen_state_sha256=identity,
            interpretation="Frozen-decoder latent inversion; decoded prior, not route-confirmed synthesis")
    if (_model_identity(model) != identity or any(p.grad is not None for p in model.parameters())
            or any(file_sha256(Path(p)) != digest for p, digest in input_hashes.items())):
        raise ValueError("Frozen alignment model or source asset changed")
    output.mkdir(parents=True, exist_ok=False)
    for name, bank in outputs.items():
        with (output / (name + ".json")).open("x", encoding="utf-8") as stream:
            json.dump(bank, stream, indent=2, allow_nan=False)
    summary = dict(status="completed", usable=count > 0, selection=selection,
        checkpoint_sha256=checkpoint_sha256, model_state_unchanged=True,
        seconds=perf_counter() - started,
        banks={name: dict(count=count, eligible_distinct=len(eligible[name]),
            exact_source_identities=sum(m["source_identity_reconstructed"] for m in bank["molecules"]),
            decoded_mean_sa=sum(m["decoded_sa"] for m in bank["molecules"]) / count if count else None,
            source_mean_sa=sum(m["sa_score"] for m in bank["molecules"]) / count if count else None,
            path=str(output / (name + ".json"))) for name, bank in outputs.items()})
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13113)
    parser.add_argument("--scan-limit", type=int, default=2048)
    parser.add_argument("--bank-size", type=int, default=32)
    parser.add_argument("--sa-ceiling", type=float, default=3.)
    parser.add_argument("--occupied-atoms", type=int, default=25)
    parser.add_argument("--no-match-control-norm", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--align-low-bank", type=Path)
    parser.add_argument("--align-control-bank", type=Path)
    parser.add_argument("--parent-panel", type=Path)
    parser.add_argument("--fit-updates", type=int, default=160)
    parser.add_argument("--fit-learning-rate", type=float, default=.1)
    parser.add_argument("--decoded-sa-ceiling", type=float, default=3.5)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("CPU thread count must be positive")
    torch.set_num_threads(args.threads)
    if args.align_low_bank:
        if args.dataset or not args.align_control_bank or not args.parent_panel:
            parser.error("Alignment mode requires both source banks and a parent panel, without --dataset")
        summary = prepare_aligned_library(args.align_low_bank, args.align_control_bank,
            args.parent_panel, args.checkpoint, args.checkpoint_sha256, args.output,
            bank_size=args.bank_size, updates=args.fit_updates, learning_rate=args.fit_learning_rate,
            decoded_sa_ceiling=args.decoded_sa_ceiling)
    else:
        if not args.dataset or args.align_control_bank or args.parent_panel:
            parser.error("Original library preparation requires --dataset; alignment requires --align-low-bank")
        summary = prepare_library(args.dataset, args.checkpoint, args.checkpoint_sha256, args.output,
            seed=args.seed, scan_limit=args.scan_limit, bank_size=args.bank_size, sa_ceiling=args.sa_ceiling,
            occupied_atoms=args.occupied_atoms, match_control_norm=not args.no_match_control_norm)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
