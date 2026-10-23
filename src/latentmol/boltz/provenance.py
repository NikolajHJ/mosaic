"""Lightweight tensor-context identity, shared by scoring and HPC diagnostics."""

import hashlib

from .contracts import PreparedComplex


def context_digest(context: PreparedComplex) -> str:
    """Detect mutation of shared fixed feature storage, coordinates and slot maps."""
    digest = hashlib.sha256()
    tensors = {**context.features, "__coordinates": context.coordinates,
               "__slot_to_atom": context.slot_to_atom, "__slot_to_token": context.slot_to_token,
               "__seed_atoms": context.seed.atoms, "__seed_charges": context.seed.charges,
               "__seed_bonds": context.seed.bonds}
    for name, value in sorted(tensors.items()):
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}".encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
