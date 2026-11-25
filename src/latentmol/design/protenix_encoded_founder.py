"""Start a local latent search from one exactly reconstructed hard molecule."""
from pathlib import Path
import json
import math

from rdkit import Chem
import torch

from latentmol.chemistry import inspect_graph
from latentmol.design.latent_protenix import hard_from_graph
from latentmol.design.protenix_conformer_rescore import score_geometry_seed
from latentmol.design.protenix_restart_parents import RestartParent
from latentmol.design.protenix_restarts import Candidate, fresh_score_records, write_json
from latentmol.evaluation.synthesis_accessibility import audited_sa_score
from latentmol.graph_policy import graph_from_decoder


@torch.no_grad()
def replay_encoded_founder(model, record, design, checkpoint_sha256):
    if record["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Encoded founder uses a different VAE checkpoint")
    latent = torch.tensor(record["latent"], dtype=torch.float32)
    slots = record["active_slots"]
    if (latent.shape != (1, model.config.latent) or not torch.isfinite(latent).all()
            or len(slots) != design.occupied_atoms or len(set(slots)) != len(slots)
            or any(type(i) is not int or not 0 <= i < model.spec.max_nodes for i in slots)):
        raise ValueError("Incompatible encoded founder latent or occupied slots")
    active = torch.zeros(model.spec.max_nodes, dtype=torch.bool)
    active[slots] = True
    graph = graph_from_decoder(model.decode(latent), model.spec, fixed_active=active,
                               temperature=design.temperature_initial)
    if any(not torch.isfinite(getattr(graph, key)).all() for key in ("atoms", "charges", "bonds")):
        raise ValueError("Nonfinite encoded founder graph")
    hard = hard_from_graph(graph)
    checked = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], model.spec)
    if not checked.smiles or "." in checked.smiles or checked.smiles != record["smiles"]:
        raise ValueError("Encoded founder failed exact canonical molecule replay")
    return Candidate(checked.smiles, latent, hard, None, "founder-01", 0), active


def score_encoded_founder(model, record, design, checkpoint_sha256, output, fresh_score):
    candidate, active = replay_encoded_founder(model, record, design, checkpoint_sha256)
    output.mkdir(parents=True)
    root = output / "founder-score"
    root.mkdir()
    rows = fresh_score(root, [dict(smiles=candidate.smiles, arm="founder", start="founder-01",
                                   role="initial", step=0)])
    if (len(rows) != 1 or rows[0]["smiles"] != candidate.smiles or rows[0]["status"] != "scored"
            or not isinstance(rows[0]["loss"], (int, float)) or not math.isfinite(rows[0]["loss"])):
        raise ValueError("Encoded founder requires a successful fresh native pocket score")
    return RestartParent(candidate, active, rows[0]["loss"], record["source"])


def summarize_encoded_search(runtime, founder, held_out, fresh_score=fresh_score_records):
    """Measure the score-ranked novel molecules against a freshly scored founder."""
    root = Path(runtime.output)
    panel = json.loads((root / "diversity.json").read_text())["score_only"]["molecules"]
    baseline = json.loads((root / "run.json").read_text())["parents"][0]
    if baseline["smiles"] != founder["smiles"]:
        raise ValueError("Founder baseline and encoded source identity disagree")
    recorded = []
    for row in panel:
        smi = row["smiles"]
        if smi == founder["smiles"]:
            raise ValueError("Novel panel includes the warm-start founder")
        recorded.append(dict(smiles=smi, branch=row["branch"], step=row["step"], fresh_loss=row["fresh_loss"],
                             change_from_founder=row["fresh_loss"] - baseline["fresh_loss"],
                             audited_sa=audited_sa_score(Chem.MolFromSmiles(smi))))
    chosen = [baseline, *recorded[:3]]
    fresh = json.loads(runtime.fresh_config.read_text())
    assets = str((runtime.fresh_config.parent / fresh["assets_config"]).resolve(strict=True))
    records = [dict(smiles=row["smiles"], start=f"rank-{i}", arm="confirmation", role="edit", step=row["step"])
               for i, row in enumerate(chosen)]
    confirmations = {}
    for seed in held_out:
        rows = score_geometry_seed(runtime, root / f"confirmation-{seed}", fresh, assets,
                                   seed, records, fresh_score)
        confirmations[str(seed)] = {r["smiles"]: rows[r["start"]] for r in records}
    write_json(root / "novel-panel.json", dict(
        founder=dict(smiles=baseline["smiles"], fresh_loss=baseline["fresh_loss"],
                     audited_sa=audited_sa_score(Chem.MolFromSmiles(baseline["smiles"]))),
        novel=recorded, held_out=confirmations, confirmation_fresh_attempts=len(records)*len(held_out)))
