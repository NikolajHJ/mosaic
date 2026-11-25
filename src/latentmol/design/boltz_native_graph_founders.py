"""Direct graph founders built from a prepared ligand pose and freshly scored in it."""
from dataclasses import replace
import math
from pathlib import Path

import torch

from latentmol.chemistry import inspect_graph
from latentmol.design.boltz_direct_graph import DirectGraphLogits
from latentmol.design.boltz_direct_pilot import make_founder
from latentmol.design.boltz_graph_restarts import DirectGraphCoordinates, pack_graph_point
from latentmol.design.latent_protenix import hard_from_graph
from latentmol.design.protenix_restarts import Candidate, write_json
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance.boltz_search import hard_graph_record, hard_pose_key


@torch.no_grad()
def native_graph_founders(scorer, design, output, *, seed, count):
    """Return directly replayable logits for one crystal ligand and distinct pose-guided trees."""
    if type(count) is not int or count < 1 or type(seed) is not int or not 0 <= seed < 2**63 - 32*count:
        raise ValueError("Native founder generation requires a positive count and valid seed")
    if scorer.execution["autocast"] != "none" or scorer.execution["tf32"]:
        raise ValueError("Native graph founders require full FP32 hard scoring")
    context = scorer.prepared
    spec = context.spec
    active = context.active.detach().cpu()
    if int(active.sum()) != design.occupied_atoms:
        raise ValueError("Prepared native ligand size differs from founder search")
    xyz = context.coordinates[0, 0, context.slot_to_atom[context.active]].detach().cpu()
    coords = DirectGraphCoordinates(spec, active)
    points, records, unique = [], [], set()
    for attempt in range(32 * count):
        draw_seed = seed + attempt
        reference = attempt == 0
        hard = context.seed.to("cpu") if reference else make_founder(spec, active, xyz, draw_seed)
        smi = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], spec).smiles
        if not smi or "." in smi or (reference and smi != context.metadata["seed_smiles"]):
            raise ValueError("Founder did not reproduce a valid connected molecule in the prepared pose")
        if smi in unique:
            continue
        unique.add(smi)
        origin = "crystal_reference" if reference else "spatial_tree"
        name = "founder-crystal" if reference else f"founder-tree-{draw_seed}"
        initial_logits = DirectGraphLogits(hard, spec, draw_seed)
        decoded, _ = initial_logits()
        packed = coords.pack(decoded)
        replay = hard_from_graph(graph_from_decoder(coords.decode(packed), spec, fixed_active=active,
                                                    temperature=design.temperature_initial))
        if any(not torch.equal(getattr(replay, key), getattr(hard, key)) for key in ("atoms", "charges", "bonds")):
            raise ValueError("Direct founder logits changed the source hard graph or slot ordering")
        # Founders use only a hard score for ranking. Relaxed loss starts at the first child observation.
        candidate = Candidate(smi, packed, hard, None, name, 0)
        points.append(SearchPoint(candidate, active, None, (name,)))
        record = hard_graph_record(hard)
        records.append(dict(smiles=smi, hard_graph=record, scoring_key=hard_pose_key(record),
                            branch=name, origin=origin, seed=draw_seed))
        if len(points) == count:
            break
    if len(points) != count:
        raise ValueError("Could not generate the requested number of distinct graph founders")

    return score_founder_panel(scorer, output, points, records, pack_graph_point, "relaxed_graph_logits",
                              dict(reference_ligand=context.metadata["target"]["pocket"]["reference_ligand_locator"],
                                   active_slots=active.nonzero().flatten().tolist(), seed=seed))


def score_founder_panel(scorer, output, points, records, pack_point, space, metadata):
    count = len(points)
    folder = Path(output) / "founders"
    folder.mkdir()
    scored = scorer.score_valid_records(folder, records)
    by_key = {r["scoring_key"]: r for r in scored}
    if len(scored) != count or set(by_key) != {r["scoring_key"] for r in records}:
        raise ValueError("Founder scoring returned inconsistent pose keys, molecule identities, or losses")
    archive = {}
    for point, row in zip(points, records, strict=True):
        scored_row = by_key[row["scoring_key"]]
        if (scored_row["smiles"] != row["smiles"] or scored_row["loss"] is None
                or not math.isfinite(scored_row["loss"])):
            raise ValueError("Founder score must be finite and belong to the requested molecule")
        row["fresh_loss"] = scored_row["loss"]
        archive[point.candidate.smiles] = replace(point, fresh_loss=scored_row["loss"])
    with (folder / "founders.pt").open("xb") as stream:
        torch.save([pack_point(p) for p in archive.values()], stream)
    write_json(folder / "founders.json", dict(optimization_space=space, founder_fresh_attempts=count,
               candidates=records, **metadata))
    return dict(archive=archive, ineligible=set(), excluded_smiles=set(archive),
                coordinate_space=space, founder_fresh_attempts=count)
