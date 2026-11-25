"""Replayable direct graph coordinates and score-free proposals for fixed-slot Boltz searches."""
from dataclasses import replace
import hashlib

import torch

from latentmol.design.boltz_speed import _point_digest
from latentmol.design.latent_protenix import hard_from_graph
from latentmol.design.protenix_restart_proposals import LATENT_RADII, propose_start
from latentmol.design.protenix_restarts import Candidate
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.design.protenix_search_state import _pack
from latentmol.graph_policy import graph_from_decoder
from latentmol.representation import DecoderOutput


def decode_active_logits(spec, slots, left, right, atom_rows, charge_rows, bond_rows):
    """Scatter one row per occupied decision; the two directions of each bond share one variable."""
    n = spec.max_nodes
    atoms = atom_rows.new_zeros(1, n, len(spec.atomic_numbers))
    charges = charge_rows.new_zeros(1, n, len(spec.formal_charges))
    bonds = bond_rows.new_zeros(1, n, n, len(spec.bond_names))
    atoms[0, slots, 1:] = atom_rows
    charges[0, slots] = charge_rows
    bonds[0, left, right] = bond_rows
    bonds[0, right, left] = bond_rows
    return DecoderOutput(atoms, charges, bonds)


class DirectGraphCoordinates:
    """Reshape free logits into the shared scorer input without a learned decoder.

    Serialized rows are ordered by ascending active slot, then ascending pairs
    of active slots. Atom rows omit NO_ATOM because occupancy stays fixed.
    """
    raw_coordinate_key = "graph_logits"

    def __init__(self, spec, active):
        if active.dtype != torch.bool or active.shape != (spec.max_nodes,) or active.sum() < 2:
            raise ValueError("Direct coordinates need a boolean mask with at least two occupied slots")
        self.spec = spec
        self.active = active.detach().clone()
        self.slots = active.nonzero().flatten()
        n = len(self.slots)
        self.left, self.right = self.slots[torch.triu_indices(n, n, 1, device=active.device)]
        self.shapes = ((n, len(spec.atomic_numbers)-1), (n, len(spec.formal_charges)),
                       (len(self.left), len(spec.bond_names)))
        self.sizes = tuple(rows * width for rows, width in self.shapes)

    def rows(self, packed):
        if (packed.shape != (1, sum(self.sizes)) or packed.device != self.active.device
                or packed.dtype != torch.float32 or not packed.isfinite().all()):
            raise ValueError("Expected one finite FP32 packed graph on the active mask's device")
        return tuple(chunk.reshape(shape) for chunk, shape in zip(packed.split(self.sizes, dim=1), self.shapes))

    def decode(self, packed):
        return decode_active_logits(self.spec, self.slots, self.left, self.right, *self.rows(packed))

    def pack(self, decoded):
        spec = self.spec
        n = spec.max_nodes
        values = (decoded.atoms, decoded.charges, decoded.bonds)
        shapes = ((1, n, len(spec.atomic_numbers)), (1, n, len(spec.formal_charges)), (1, n, n, len(spec.bond_names)))
        if any(x.dtype != torch.float32 or x.device != self.active.device or x.shape != shape
               or not x.isfinite().all() for x, shape in zip(values, shapes)):
            raise ValueError("Invalid decoder tensors for the direct graph schema or active mask")
        rows = (decoded.atoms[0, self.slots, 1:], decoded.charges[0, self.slots],
                decoded.symmetric_bond_logits[0, self.left, self.right])
        return torch.cat([row.reshape(1, -1) for row in rows], dim=1)


@torch.no_grad()
def convert_point(decoder, design, point):
    """Convert founder or first-wave control coordinates once, preserving the full relaxed graph."""
    coords = DirectGraphCoordinates(decoder.spec, point.active)
    decoded = decoder.decode(point.candidate.latent)
    packed = coords.pack(decoded)
    kwargs = dict(fixed_active=point.active, temperature=design.temperature_initial)
    original = graph_from_decoder(decoded, decoder.spec, **kwargs)
    replay = graph_from_decoder(coords.decode(packed), decoder.spec, **kwargs)
    hard = hard_from_graph(replay)
    for field in ("atoms", "charges", "bonds"):
        if (not torch.equal(getattr(original, field), getattr(replay, field))
                or not torch.equal(getattr(hard, field), getattr(point.candidate.hard, field))):
            raise ValueError("Conversion changed the relaxed graph or the saved hard pose")
    return replace(point, candidate=replace(point.candidate, latent=packed, hard=hard))


def graph_founder_state(decoder, design, state):
    archive = {smi: convert_point(decoder, design, p) for smi, p in state["archive"].items()}
    return dict(state, archive=archive)


def pack_graph_point(point):
    record = _pack(point)
    record["graph_logits"] = record.pop("latent")
    return record


def _label_preserving_start(coords, packed, seed):
    """Break exact collisions with a small, seeded increase in existing argmax margins."""
    generator = torch.Generator(device=packed.device).manual_seed(seed)
    directions = []
    for rows in coords.rows(packed):
        winners = rows.argmax(-1, keepdim=True)
        gains = torch.rand(winners.shape, generator=generator, device=packed.device) + .5
        delta = torch.zeros_like(rows).scatter_(1, winners, gains)
        directions.append((delta - delta.mean(-1, keepdim=True)).flatten())
    direction = torch.cat(directions)
    return packed + direction * (min(LATENT_RADII) / direction.norm())


@torch.no_grad()
def direct_archive_starts(spec, design, state, count, seed, *, seen, parent_count):
    """Use the control's radius/direction screen in direct logit coordinates after wave one."""
    parents = sorted((p for smi, p in state["archive"].items() if smi not in state["ineligible"]),
                     key=lambda p: (p.fresh_loss, p.candidate.smiles))[:parent_count]
    if len(parents) != parent_count or design.noise_initial != 0 or any(
            int(p.active.sum()) != design.occupied_atoms for p in parents):
        raise ValueError("Direct restarts require eligible matching-size parents and zero update noise")
    starts, records = [], []
    for slot in range(count):
        parent = parents[slot % parent_count]
        row = parent.candidate
        coords = DirectGraphCoordinates(spec, parent.active)
        kwargs = dict(fixed_active=parent.active, temperature=design.temperature_initial)
        old = graph_from_decoder(coords.decode(row.latent), spec, **kwargs)
        name = f"start-{slot+1:02d}"
        seed0 = trial_seed = seed + slot
        for attempt in range(9):
            packed, _, info = propose_start(coords, row, parent.active, policy="latent", seed=trial_seed,
                                            temperature=design.temperature_initial)
            key = _point_digest(packed, parent.active)
            if key not in seen:
                break
            salt = f"{seed0}:{name}:{attempt+1}".encode()
            trial_seed = int.from_bytes(hashlib.sha256(salt).digest()[:8], "big") % (2**63)
        else:
            packed = _label_preserving_start(coords, row.latent, trial_seed)
            key = _point_digest(packed, parent.active)
            if key in seen:
                raise ValueError("Unable to refill an exact duplicate direct start within the retry budget")
            info = dict(parent=info["parent"], proposal=info["parent"], fallback="increase_parent_argmax_margins")
        seen.add(key)
        new = graph_from_decoder(coords.decode(packed), spec, **kwargs)
        hard = hard_from_graph(new)
        proposal = info["proposal"]
        if info.get("fallback") and any(not torch.equal(getattr(hard, k), getattr(row.hard, k))
                                        for k in ("atoms", "charges", "bonds")):
            raise ValueError("Duplicate-refill proposal changed its reported hard parent")
        candidate = Candidate(proposal["smiles"], packed, hard, 0., name, 0)
        starts.append(SearchPoint(candidate, parent.active, None, parent.ancestry + (name,)))
        info.pop("latent_displacement", None)
        info.update(policy="graph_logits", coordinate_displacement=float((packed-row.latent).norm()),
                    radius_cap=max(LATENT_RADII),
                    relaxed_l2=dict(atoms=float((new.atoms-old.atoms).norm()),
                                    charges=float((new.charges-old.charges).norm()),
                                    bonds_upper=float((new.bonds-old.bonds)[0, coords.left, coords.right].norm())))
        records.append(dict(name=name, seed=trial_seed, source_branch=row.branch, source_step=row.step,
                            source_smiles=row.smiles, source_loss=parent.fresh_loss, proposal=info,
                            rejected_exact_starts=9 if info.get("fallback") else attempt))
    return starts, records
