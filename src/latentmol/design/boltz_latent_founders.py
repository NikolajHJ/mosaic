"""Screen native-size VAE prior decodes for a fixed-pose Boltz search."""
import torch
from rdkit import Chem

from latentmol.chemistry import inspect_graph
from latentmol.design.latent_protenix import hard_from_graph
from latentmol.design.protenix_restarts import Candidate
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance.boltz_search import hard_graph_record, hard_pose_key


@torch.no_grad()
def prior_founder_panel(model, design, *, count, max_draws, seed, history):
    """Retain distinct connected, round-trip-stable graphs with the pocket's native atom count."""
    if not 0 < count <= max_draws or not 0 <= seed < 2**63:
        raise ValueError("Invalid VAE prior panel seed or draw budget")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    points, records, seen = [], [], set(history)
    for index in range(max_draws):
        z = torch.randn(1, model.config.latent, generator=rng)
        decoded = model.decode(z)
        if not all(t.isfinite().all() for t in (decoded.atoms, decoded.charges, decoded.symmetric_bond_logits)):
            raise ValueError("Nonfinite VAE prior prediction")
        active = decoded.atoms[0].argmax(-1) != 0
        if int(active.sum()) != design.occupied_atoms:
            continue
        hard = hard_from_graph(graph_from_decoder(decoded, model.spec, fixed_active=active,
                                                   temperature=design.temperature_initial))
        smi = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], model.spec).smiles
        if not smi or smi in seen:
            continue
        parsed = Chem.MolFromSmiles(smi)
        if parsed is None or Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=False) != smi:
            continue
        name = f"prior-{seed}-draw-{index}"
        candidate = Candidate(smi, z, hard, None, name, 0)
        points.append(SearchPoint(candidate, active, None, (name,)))
        graph = hard_graph_record(hard)
        records.append(dict(smiles=smi, hard_graph=graph, scoring_key=hard_pose_key(graph), branch=name,
                            seed=seed, prior_draw=index, active_slots=active.nonzero().flatten().tolist()))
        seen.add(smi)
        if len(points) == count:
            return points, records
    raise ValueError(f"Only {len(points)} distinct valid native-size prior founders among {max_draws} draws")
