"""Fixed one-pass evaluations for the frozen-decoder encoder distillation pilot."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path

from rdkit import rdBase
import torch

from latentmol.chemistry import encode_canonical_smiles
from latentmol.datasets.data import collate_graphs
from latentmol.experiments import file_sha256
from latentmol.representation import GraphSpec, HardGraphBatch
from latentmol.sampling import _validate_graph_fields
from latentmol.training.generated_roundtrip import canonical_generated_parents
from latentmol.training.occupancy_monitor import _inspect


@dataclass
class EvaluationPanel:
    """Targets in verified canonical slots on CPU, stored in source order."""

    names: list[str]
    source_rows: list[int]
    graphs: HardGraphBatch


@contextmanager
def _eval_mode(model):
    # Encoder training may leave the frozen decoder in eval mode. Restore both independently.
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, mode in modes:
            module.training = mode


@torch.no_grad()
def build_panels(model, val_path, *, seed=2026112471, draws=512, heldout_count=512, batch=8):
    """Build once before training, then exclude all returned identities from every training source."""
    if min(draws, heldout_count, batch) < 1:
        raise ValueError("Panel sizes must be positive")
    val_path = Path(val_path)
    val = json.loads(val_path.read_bytes())
    if GraphSpec.from_dict(val["spec"]) != model.spec or heldout_count > len(val["rows"]):
        raise ValueError("Held-out source has the wrong schema or insufficient rows")
    val_rows = val["rows"][:heldout_count]
    heldout_names = [row[2] for row in val_rows]
    if len(set(heldout_names)) != heldout_count:
        raise ValueError("Duplicate identities in the saved PubChem validation panel")
    real = []
    for name in heldout_names:
        encoded = encode_canonical_smiles(name, model.spec)
        real.append({key: torch.as_tensor(encoded[field], dtype=torch.long) for key, field in
                     (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))})
    heldout = EvaluationPanel(heldout_names, [int(row[0]) for row in val_rows], collate_graphs(real))
    excluded = set(heldout_names)
    prior = torch.randn(draws, model.config.latent, generator=torch.Generator().manual_seed(seed))
    device = next(model.parameters()).device
    names, indices, batches = [], [], []
    rejected = Counter()
    with _eval_mode(model), rdBase.BlockLogs():
        for start in range(0, draws, batch):
            decoded = model.decode(prior[start:start + batch].to(device))
            _validate_graph_fields(decoded, model.spec)
            parents, info = canonical_generated_parents(decoded.harden(model.spec), model.spec,
                                                        include_smiles=True)
            rejected.update(info["rejections"])
            if parents is None:
                continue
            keep = []
            for local, (index, name) in enumerate(zip(info["rows"], info["smiles"], strict=True)):
                if name in excluded:
                    rejected["duplicate_or_heldout_identity"] += 1
                    continue
                excluded.add(name)
                names.append(name)
                indices.append(start + index)
                keep.append(local)
            if keep:
                batches.append(HardGraphBatch(parents.atoms[keep], parents.charges[keep],
                                              parents.bonds[keep]).to("cpu"))
    if not names or len(names) + sum(rejected.values()) != draws:
        raise ValueError("Empty or incompletely accounted generated evaluation panel")
    generated = EvaluationPanel(names, indices, HardGraphBatch(*(
        torch.cat([getattr(b, field) for b in batches]) for field in ("atoms", "charges", "bonds"))))
    provenance = dict(prior_seed=seed, prior_draws=draws, generated_unique=len(names),
                      prior_rejections=dict(rejected), heldout_count=heldout_count,
                      heldout_seed=val["seed"], heldout_dataset_sha256=val["dataset_sha256"],
                      heldout_panel_sha256=file_sha256(val_path))
    return {"generated": generated, "heldout": heldout}, excluded, provenance


@torch.no_grad()
def evaluate_panels(model, panels, *, batch=8):
    """One posterior-mean encoding and raw argmax decode per parent. Keep invalid outputs in counts."""
    if batch < 1:
        raise ValueError("Evaluation batch must be positive")
    scores = {}
    device = next(model.parameters()).device
    with _eval_mode(model), rdBase.BlockLogs():
        for key, panel in panels.items():
            rows = []
            for start in range(0, len(panel.names), batch):
                sl = slice(start, start + batch)
                targets = HardGraphBatch(panel.graphs.atoms[sl], panel.graphs.charges[sl],
                                         panel.graphs.bonds[sl]).to(device)
                mu, _ = model.encoder(targets, from_canonical_cache=True)
                if not torch.isfinite(mu).all():
                    raise FloatingPointError("Nonfinite encoder mean in one-pass evaluation")
                decoded = model.decode(mu)
                _validate_graph_fields(decoded, model.spec)
                hard = decoded.harden(model.spec)
                # Use the same canonical-slot edge convention as the production fixed panels.
                extra = ((hard.bonds != 0) & (targets.bonds == 0)).triu(1).sum((-1, -2)).tolist()
                missing = ((hard.bonds == 0) & (targets.bonds != 0)).triu(1).sum((-1, -2)).tolist()
                for index, target, pred, ex, miss in zip(panel.source_rows[sl], panel.names[sl],
                                                        _inspect(hard, model.spec), extra, missing, strict=True):
                    rows.append(dict(source_row=index, target=target, smiles=pred.smiles, exact=pred.smiles == target,
                                     valid=pred.smiles is not None, extra=ex, missing=miss))
            scores[key] = {field: sum(r[field] for r in rows) for field in ("exact", "valid", "extra", "missing")}
            scores[key].update(n=len(rows), rows=rows)
    return scores
