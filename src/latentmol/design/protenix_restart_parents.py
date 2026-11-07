"""Load independent archived Protenix restart parents from completed flow cohorts."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import zip_longest
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from latentmol.chemistry import inspect_graph
from latentmol.design.latent_protenix import _model_identity, choose_active, hard_from_graph
from latentmol.design.protenix_restarts import Candidate
from latentmol.design.temperature_study import hard_at
from latentmol.graph_policy import graph_from_decoder


@dataclass(frozen=True)
class RestartParent:
    candidate: Candidate
    active: torch.Tensor
    fresh_loss: float | None
    provenance: dict


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _saved_latent(raw, step, width):
    points = []
    for path in sorted(raw.glob("states-*.npz")):
        with np.load(path, allow_pickle=False) as block:
            indices = np.flatnonzero(block["steps"] == step)
            points.extend(torch.from_numpy(block["latent"][index].copy())[None] for index in indices)
    if (len(points) != 1 or points[0].shape != (1, width)
            or points[0].dtype != torch.float32 or not bool(points[0].isfinite().all())):
        raise ValueError(f"Require one finite compatible saved latent at step {step} in {raw}")
    return points[0]


def _archive_candidates(root, seed, checkpoint_sha256):
    panel = _read_json(root / "panel.json")
    fresh_path = root / "fresh" / "summary.json"
    fresh = _read_json(fresh_path)
    starts, train = panel["starts"], set(panel["train"])
    if (len(starts) != len(set(starts)) or not train <= set(starts)
            or panel["checkpoint_sha256"] != checkpoint_sha256
            or fresh["status"] != "completed" or fresh["geometry_seed"] != 1201):
        raise ValueError(f"Wrong checkpoint, scoring protocol, or cohort partition in {root}")
    held_out = [start for start in starts if start not in train]
    scored = defaultdict(list)
    for row in fresh["candidates"]:
        if row["start"] in train:
            continue
        if row["start"] not in starts or row["arm"] not in ("teacher", "sgd_short"):
            raise ValueError(f"Unrecognized saved cohort trajectory: {root}")
        if row["status"] == "scored":
            if not isinstance(row["loss"], (int, float)) or not math.isfinite(row["loss"]):
                raise ValueError(f"Nonfinite fresh score in {fresh_path}")
            scored[row["start"]].append(row)
    random.Random(seed).shuffle(held_out)
    return [(root, start, sorted(scored[start], key=lambda row: (row["loss"], row["step"], row["smiles"])))
            for start in held_out if scored[start]]


@torch.no_grad()
def _restore_parent(decoder, root, row, *, decoder_sha256, checkpoint_sha256, occupied_atoms, temperature):
    # Both scoring arms reference the same teacher history. sgd_short is a prefix.
    trajectory = root / row["start"] / "teacher"
    run = _read_json(trajectory / "run.json")
    slots = run["active_slots"]
    if (run["status"] != "completed" or run["checkpoint_sha256"] != checkpoint_sha256
            or run["decoder_sha256"] != decoder_sha256 or run["graph_spec"] != decoder.spec.to_dict()
            or len(slots) != occupied_atoms or len(set(slots)) != occupied_atoms
            or any(type(slot) is not int or not 0 <= slot < decoder.spec.max_nodes for slot in slots)):
        raise ValueError(f"Archived trajectory identity or slots differ: {trajectory}")
    step = 0 if row["role"] == "initial" and row["step"] == -1 else row["step"]
    if type(step) is not int or not run["start_step"] <= step <= run["completed_step"]:
        raise ValueError(f"Archived score points to an invalid step in {trajectory}")
    observations = trajectory / "arms" / "latent"
    raw = observations / "raw-blocks"
    point = _saved_latent(raw, step, decoder.config.latent)
    archived_hard = hard_at(raw, step)
    active = torch.zeros(decoder.spec.max_nodes, dtype=torch.bool)
    active[slots] = True
    decoded = decoder.decode(point)
    graph = graph_from_decoder(decoded, decoder.spec, fixed_active=active, temperature=temperature)
    if any(not bool(torch.isfinite(getattr(graph, key)).all()) for key in ("atoms", "charges", "bonds")):
        raise ValueError(f"Nonfinite restored decoder graph at {trajectory}:{step}")
    hard = hard_from_graph(graph)
    checked = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], decoder.spec)
    if (checked.smiles != row["smiles"]
            or any(not torch.equal(getattr(hard, key), getattr(archived_hard, key))
                   for key in ("atoms", "charges", "bonds"))):
        raise ValueError(f"Restored decoder graph differs from scored graph at {trajectory}:{step}")
    with (observations / "metrics.jsonl").open(encoding="utf-8") as stream:
        metrics = [metric for line in stream if (metric := json.loads(line))["step"] == step]
    if len(metrics) != 1 or not math.isfinite(metrics[0]["score"]["binding_loss"]):
        raise ValueError(f"Require one finite binding loss at {trajectory}:{step}")
    relaxed_loss = metrics[0]["score"]["binding_loss"]
    candidate = Candidate(checked.smiles, point, hard, relaxed_loss, str(trajectory), step)
    return RestartParent(candidate, active, row["loss"], dict(
        kind="archived_flow_elite", cohort=str(root), start=row["start"], trajectory=str(trajectory),
        step=step, scoring_arm=row["arm"], scoring_role=row["role"],
        fresh_summary=str(root / "fresh" / "summary.json"), geometry_seed=1201,
        checkpoint_sha256=checkpoint_sha256, decoder_sha256=decoder_sha256))


def load_parent_panel(decoder, cohort_roots, *, count, seed, temperature=1.0, occupied_atoms=25,
                      checkpoint_sha256=None):
    """Round-robin held-out starts across cohorts, taking the best fresh score per start."""
    roots = [Path(root) for root in cohort_roots]
    if type(count) is not int or count < 1 or not roots or len(set(roots)) != len(roots):
        raise ValueError("Require a positive parent count and distinct archived cohort roots")
    if not checkpoint_sha256:
        checkpoint_sha256 = _read_json(roots[0] / "panel.json")["checkpoint_sha256"]
    choices = [_archive_candidates(root, seed + index, checkpoint_sha256) for index, root in enumerate(roots)]
    decoder_sha256 = _model_identity(decoder)
    parents = []
    used_starts, used_smiles = set(), set()
    for group in zip_longest(*choices):
        for selection in group:
            if selection is None:
                continue
            root, start, rows = selection
            if start in used_starts:
                raise ValueError("Archived cohorts share a held-out starting trajectory")
            used_starts.add(start)
            row = next((item for item in rows if item["smiles"] not in used_smiles), None)
            if row is None:
                continue
            parent = _restore_parent(decoder, root, row, decoder_sha256=decoder_sha256,
                checkpoint_sha256=checkpoint_sha256, occupied_atoms=occupied_atoms, temperature=temperature)
            parents.append(parent)
            used_smiles.add(row["smiles"])
            if len(parents) == count:
                return parents
    raise ValueError(f"Only {len(parents)} distinct independently sourced parents for a panel of {count}")


@torch.no_grad()
def make_prior_parent(decoder, seed, *, temperature=1.0, occupied_atoms=25):
    """Draw one unscored prior graph for independent restarts; invalid draws return None."""
    generator = torch.Generator().manual_seed(seed)
    point = torch.randn((1, decoder.config.latent), generator=generator)
    decoded = decoder.decode(point)
    active = choose_active(decoded, decoder.spec, occupied_atoms)
    graph = graph_from_decoder(decoded, decoder.spec, fixed_active=active, temperature=temperature)
    if any(not bool(torch.isfinite(getattr(graph, key)).all()) for key in ("atoms", "charges", "bonds")):
        raise ValueError("Nonfinite prior decoder graph")
    hard = hard_from_graph(graph)
    checked = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], decoder.spec)
    if not checked.smiles or "." in checked.smiles:
        return None
    candidate = Candidate(checked.smiles, point, hard, None, f"prior-{seed}", 0)
    return RestartParent(candidate, active, None, dict(kind="fresh_prior", seed=seed))
