"""Portable checkpoints at completed Protenix search wave boundaries."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from latentmol.design.protenix_restarts import Candidate
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.representation import HardGraphBatch


def search_identity(settings, checkpoint_sha256, decoder_sha256, spec):
    resolved = {key: value for key, value in settings.items() if key != "config_source"}
    payload = dict(settings=resolved, checkpoint_sha256=checkpoint_sha256,
                   decoder_sha256=decoder_sha256, graph_spec=spec.to_dict())
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _pack(point):
    row = point.candidate
    return dict(smiles=row.smiles, latent=row.latent, atoms=row.hard.atoms, charges=row.hard.charges,
                bonds=row.hard.bonds, relaxed_loss=row.relaxed_loss, branch=row.branch, step=row.step,
                active=point.active, fresh_loss=point.fresh_loss, ancestry=list(point.ancestry),
                repair_edits=point.repair_edits, repair_smiles=point.repair_smiles,
                repair_target_sha256=point.repair_target_sha256)


def _unpack(row, spec):
    n = spec.max_nodes
    latent, active = row["latent"], row["active"]
    if (latent.ndim != 2 or latent.shape[0] != 1 or not latent.is_floating_point()
            or not bool(latent.isfinite().all()) or active.shape != (n,) or active.dtype != torch.bool
            or row["atoms"].shape != (1, n) or row["charges"].shape != (1, n)
            or row["bonds"].shape != (1, n, n)):
        raise ValueError("Incompatible tensor shapes or values in search checkpoint")
    hard = HardGraphBatch(row["atoms"], row["charges"], row["bonds"])
    candidate = Candidate(row["smiles"], latent, hard, row["relaxed_loss"], row["branch"], row["step"])
    return SearchPoint(candidate, active, row["fresh_loss"], tuple(row["ancestry"]),
                       row["repair_edits"], row["repair_smiles"], row["repair_target_sha256"])


def save_search_state(path, identity, *, founders, archive, invalid_pool, attempted, used_priors,
                      ineligible, history, seen_starts, prior_seed, summaries, founder_records, segments):
    state = dict(version=1, identity=identity, founders=[_pack(p) for p in founders],
                 archive={smiles: _pack(point) for smiles, point in archive.items()},
                 invalid_pool=[_pack(p) for p in invalid_pool], attempted=sorted(attempted),
                 used_priors=sorted(used_priors), ineligible=sorted(ineligible),
                 history={key: sorted(steps) for key, steps in history.items()},
                 seen_starts=sorted(seen_starts), prior_seed=prior_seed, summaries=summaries,
                 founder_records=founder_records, segments=segments)
    with NamedTemporaryFile(dir=path.parent, prefix=".search-state-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_search_state(path, identity, spec):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("version") != 1 or state.get("identity") != identity:
        raise ValueError("Search checkpoint belongs to different model, scorer, or settings")
    rows = state["summaries"]
    if not rows or [row["cycle"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Search checkpoint must cover consecutive completed waves")
    updates = sum(row["updates"] for row in rows)
    attempts = sum(row["fresh_attempts"] for row in rows)
    if updates != rows[-1]["cumulative_updates"] or attempts != rows[-1]["cumulative_fresh_attempts"]:
        raise ValueError("Search checkpoint budgets disagree with the completed waves")
    for key in ("founders", "invalid_pool"):
        state[key] = [_unpack(point, spec) for point in state[key]]
    state["archive"] = {smiles: _unpack(point, spec) for smiles, point in state["archive"].items()}
    for key in ("attempted", "used_priors", "ineligible", "seen_starts"):
        state[key] = set(state[key])
    state["history"] = {key: set(steps) for key, steps in state["history"].items()}
    return state
