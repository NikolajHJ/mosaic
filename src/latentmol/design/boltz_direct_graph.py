"""Optimize free molecular logits against the fixed-pose Boltz2 DBMol objective."""

import hashlib
import json
import math
import os
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from latentmol.chemistry import inspect_graph
from latentmol.design.boltz_graph_restarts import decode_active_logits
from latentmol.design.latent_protenix import LatentDesignConfig, _json, hard_from_graph, structural_target
from latentmol.design.protenix_restarts import write_json
from latentmol.evaluation.synthesis_accessibility import audited_sa_score
from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance.boltz_search import hard_graph_record, hard_pose_key
from latentmol.guidance.regularization import fixed_occupancy_valence_excess
from latentmol.repair_guidance.loss import fixed_target_nll


_CARBONYL = Chem.MolFromSmarts("[CX3]=[OX1]")


def _soft_start(labels, categories, keep, generator):
    """Seeded bounded logit jitter changes margins while preserving each hard label."""
    keep = torch.as_tensor(keep, dtype=torch.float32).expand_as(labels)
    alternatives = (1 - keep) / max(1, categories - 1)
    probs = alternatives[:, None].expand(-1, categories).clone()
    probs.scatter_(1, labels[:, None], keep[:, None])
    jitter = (torch.rand(probs.shape, generator=generator) - .5) * .24
    return torch.nn.Parameter(probs.log() + jitter)


class DirectGraphLogits(torch.nn.Module):
    """One trainable categorical row per atom, charge, and undirected active pair."""

    def __init__(self, initial_hard, spec, seed):
        super().__init__()
        hard = initial_hard.to("cpu")
        n = spec.max_nodes
        for tensor, shape, classes in ((hard.atoms, (1, n), spec.atomic_numbers),
                                       (hard.charges, (1, n), spec.formal_charges),
                                       (hard.bonds, (1, n, n), spec.bond_names)):
            if (tensor.shape != shape or tensor.dtype != torch.long
                    or bool((tensor < 0).any()) or bool((tensor >= len(classes)).any())):
                raise ValueError("Initial hard graph has incompatible shape, dtype or category")
        active = hard.atoms[0] != 0
        slots = active.nonzero().flatten()
        if (len(slots) < 2 or not torch.equal(hard.bonds, hard.bonds.transpose(1, 2))
                or hard.bonds[0].diagonal().any() or hard.bonds[0, ~active].any()):
            raise ValueError("Expected at least two occupied slots and a symmetric loop-free bond graph")
        left, right = slots[torch.triu_indices(len(slots), len(slots), offset=1)]
        self.register_buffer("active", active)
        self.register_buffer("slots", slots)
        self.register_buffer("left", left)
        self.register_buffer("right", right)
        self.spec = spec
        rng = torch.Generator().manual_seed(seed)
        self.atom_logits = _soft_start(hard.atoms[0, slots] - 1, len(spec.atomic_numbers) - 1, .8, rng)
        self.charge_logits = _soft_start(hard.charges[0, slots], len(spec.formal_charges), .8, rng)
        bonds = hard.bonds[0, left, right]
        self.bond_logits = _soft_start(bonds, len(spec.bond_names), torch.where(bonds == 0, .985, .8), rng)

    def forward(self):
        decoded = decode_active_logits(self.spec, self.slots, self.left, self.right,
                                       self.atom_logits, self.charge_logits, self.bond_logits)
        return decoded, graph_from_decoder(decoded, self.spec, fixed_active=self.active)

    def start_digest(self):
        digest = hashlib.sha256(self.spec.fingerprint().encode())
        for tensor in (self.active, self.atom_logits, self.charge_logits, self.bond_logits):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def motif_and_triangles(graph, spec, active):
    atoms = graph.atoms[0, active]
    bonds = graph.bonds[0, active][:, active]
    neutral = graph.charges[0, active, spec.formal_charges.index(0)]
    carbon = atoms[:, spec.atomic_numbers.index(6)] * neutral
    oxygen = atoms[:, spec.atomic_numbers.index(8)] * neutral
    double = bonds[:, :, spec.bond_names.index("DOUBLE")]
    # A carbonyl oxygen is terminal. Omit the candidate C--O pair itself from
    # the product of no-bond probabilities over the oxygen's other neighbors.
    log_none = bonds[..., 0].clamp_min(1e-12).log()
    terminal_oxygen = (log_none.sum(dim=0, keepdim=True) - log_none).exp()
    strongest_carbonyl = (carbon[:, None] * double * oxygen[None, :] * terminal_oxygen).amax()
    # Symmetric adjacency, zero diagonal. Each unordered triangle has six traversals.
    adjacency = bonds[:, :, 1:].sum(-1)
    triangles = ((adjacency @ adjacency) * adjacency).sum() / 6
    return strongest_carbonyl, triangles


def carbonyl_loss(strongest_carbonyl):
    """Log penalty for a missing carbonyl, saturated at probability one half."""
    return (math.log(.5) - strongest_carbonyl.clamp_min(1e-12).log()).relu()


def _properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return dict(carbonyl=mol.HasSubstructMatch(_CARBONYL), aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
                triangles=sum(len(ring) == 3 for ring in mol.GetRingInfo().AtomRings()), audited_sa=audited_sa_score(mol))


def _update(optimizer, params, binding, chemistry, step_cap):
    # Differentiate the native scalar separately to detect a severed input channel.
    binding_grad = torch.autograd.grad(binding, params, retain_graph=True, allow_unused=True)
    chemistry_grad = torch.autograd.grad(chemistry, params)
    before = [p.detach().clone() for p in params]
    for p, core, extra in zip(params, binding_grad, chemistry_grad, strict=True):
        if core is None:
            raise ValueError("Detached native gradient for direct graph logits")
        p.grad = (core + extra).detach()
        if not torch.isfinite(p.grad).all():
            raise ValueError("Nonfinite direct graph gradient")
    optimizer.step()
    with torch.no_grad():
        for p, old in zip(params, before, strict=True):
            delta = p - old
            scale = (step_cap / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
            p.copy_(old + delta * scale)
            if not torch.isfinite(p).all():
                raise ValueError("Nonfinite direct graph logit update")
    return {name: float(grad.norm()) for name, grad in zip(("atoms", "charges", "bonds"), binding_grad, strict=True)}


def run_child(scorer, spec, initial_hard, output, *, seed, updates, lr, motif_weight, triangle_weight,
              fresh_budget, excluded_smiles=(), step_cap=.25, ready=None,
              pharmaconet=None, pharmaconet_weight=0.):
    """Run a single direct-graph trajectory; count gradient updates and hard scores separately.

    The supplied graph determines ascending slot-to-native-pose correspondence. Optimize
    only the active categorical rows, on CPU in FP32; the native scorer routes its input
    to its allocated GPU. Use Adam with an L2 displacement cap on each categorical row.
    """
    if (type(seed) is not int or not 0 <= seed < 2**63 or type(updates) is not int or updates < 1
            or type(fresh_budget) is not int or fresh_budget < 0):
        raise ValueError("Invalid direct graph seed or budgets")
    if (not all(math.isfinite(x) for x in (lr, step_cap, motif_weight, triangle_weight))
            or min(lr, step_cap) <= 0 or min(motif_weight, triangle_weight) < 0):
        raise ValueError("Invalid direct graph optimizer or chemistry weights")
    if (not math.isfinite(pharmaconet_weight) or pharmaconet_weight < 0
            or pharmaconet_weight and pharmaconet is None):
        raise ValueError("A positive PharmacoNet guidance weight needs a fixed pocket map")
    logits = DirectGraphLogits(initial_hard, spec, seed)
    params = tuple(logits.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    # These settings supply only the existing detached repair oracle's search limits.
    repair_config = LatentDesignConfig()
    initial = initial_hard.to("cpu")
    founder = inspect_graph(initial.atoms[0], initial.charges[0], initial.bonds[0], spec).smiles
    excluded = set(excluded_smiles)
    if founder:
        excluded.add(founder)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(parameterization="direct_graph_logits", seed=seed, updates=updates, lr=lr,
                    optimized_variables=["element logits", "formal-charge logits", "undirected bond logits"],
                    pose_and_occupancy="fixed",
                    optimizer="Adam", per_row_step_cap=step_cap, motif_weight=motif_weight,
                    triangle_weight=triangle_weight, repair_weight=1., valence_weight=1.,
                    carbonyl_saturation=.5, unpenalized_triangle_count=1,
                    founder_smiles=founder, founder_scoring_key=hard_pose_key(hard_graph_record(initial)),
                    start_sha256=logits.start_digest(), fresh_budget=fresh_budget,
                    excluded_smiles=sorted(excluded), training_noise=0.,
                    initial_keep_probabilities=dict(element=.8, charge=.8, bond=.8, no_bond=.985),
                    initial_logit_jitter_uniform=[-.12, .12])
    if pharmaconet is not None:
        settings.update(pharmaconet_weight=pharmaconet_weight, pharmaconet_hotspot_ids=pharmaconet.ids,
                        pharmaconet_site_weights=pharmaconet.weights.tolist(),
                        spatial_candidate_ring_counts={str(n): len(rows) for n, rows in pharmaconet.spatial_rings.items()})
    write_json(output / "start.json", settings | dict(hard_graph=hard_graph_record(initial)))
    candidates, properties = {}, {}
    valid_states = 0
    with (output / "metrics.jsonl").open("x", encoding="utf-8", buffering=1) as metrics:
        for step in range(updates + 1):
            decoded, graph = logits()
            binding, aux = scorer.score_at(graph, logits.active)
            if binding.numel() != 1 or not binding.requires_grad or not torch.isfinite(binding):
                raise ValueError("Require a finite attached native scalar for direct graph optimization")
            hard = hard_from_graph(graph)
            check = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], spec)
            target, repair = structural_target(graph, spec, logits.active, repair_config, step)
            repair_nll = binding.new_zeros(())
            if target is not None:
                repair_nll, _ = fixed_target_nll(decoded, graph, target, spec, normalization="mean_per_decision")
            valence = fixed_occupancy_valence_excess(graph, spec, logits.active)
            motif, triangles = motif_and_triangles(graph, spec, logits.active)
            motif_penalty = carbonyl_loss(motif)
            chemistry = repair_nll + valence + motif_weight * motif_penalty + triangle_weight * (triangles - 1).relu()
            guidance_log = {}
            if pharmaconet is not None:
                with torch.set_grad_enabled(bool(pharmaconet_weight)):
                    map_loss, matches = pharmaconet(graph, logits.active, spec)
                if not torch.isfinite(map_loss) or (pharmaconet_weight and not map_loss.requires_grad):
                    raise ValueError("Nonfinite or detached PharmacoNet guidance")
                guidance_log = dict(pharmaconet_loss=float(map_loss.detach()), pharmaconet_matches=matches)
                if pharmaconet_weight:
                    if step == 0:
                        gradients = torch.autograd.grad(map_loss, params, retain_graph=True)
                        if any(not grad.isfinite().all() for grad in gradients):
                            raise ValueError("Nonfinite PharmacoNet gradient")
                        guidance_log["pharmaconet_gradient_norms"] = dict(zip(
                            ("atoms", "charges", "bonds"), [float(grad.norm()) for grad in gradients], strict=True))
                    chemistry = chemistry + pharmaconet_weight * map_loss
            if not torch.isfinite(chemistry):
                raise ValueError("Nonfinite direct graph chemistry objective")
            value = float(binding.detach())
            row = dict(step=step, smiles=check.smiles, failure_reason=check.failure_reason,
                       relaxed_dbmol=value, total_loss=value + float(chemistry.detach()), components=aux,
                       repair_status=repair["status"], repair_loss=float(repair_nll.detach()),
                       valence_excess=float(valence.detach()), carbonyl_probability=float(motif.detach()),
                       motif_penalty=float(motif_penalty.detach()), soft_triangles=float(triangles.detach()),
                       **guidance_log)
            if check.smiles:
                valid_states += 1
                if check.smiles not in properties:
                    properties[check.smiles] = _properties(check.smiles)
                row["hard_properties"] = properties[check.smiles]
                prior = candidates.get(check.smiles)
                if check.smiles not in excluded and (prior is None or value < prior["relaxed_loss"]):
                    record = hard_graph_record(hard)
                    candidates[check.smiles] = dict(smiles=check.smiles, hard_graph=record, scoring_key=hard_pose_key(record),
                                                    step=step, relaxed_loss=value, properties=properties[check.smiles])
            if step < updates:
                row["binding_gradient_norms"] = _update(optimizer, params, binding, chemistry, step_cap)
            metrics.write(json.dumps(row, allow_nan=False) + "\n")
            if step == 0 and ready is not None:
                _json(ready, dict(status="models_and_inputs_ready", nonce=os.getenv("DBMOL_LAUNCH_NONCE"),
                                  job_id=os.getenv("LSB_JOBID")))
    ranked = sorted(candidates.values(), key=lambda r: (r["relaxed_loss"], r["smiles"]))
    write_json(output / "valid-candidates.json", ranked)
    selected = ranked[:fresh_budget]
    scored = scorer.score_valid_records(output, selected) if selected else []
    if ([r["scoring_key"] for r in scored] != [r["scoring_key"] for r in selected]
            or any(not math.isfinite(r["loss"]) for r in scored)):
        raise ValueError("Invalid hard-score receipt")
    top = sorted(scored, key=lambda r: (r["loss"], r["smiles"]))
    result = settings | dict(status="completed", relaxed_observations=updates+1, valid_states=valid_states,
                             distinct_valid=len(properties), distinct_novel_valid=len(candidates),
                             fresh_attempts=len(selected), fresh_successes=len(scored),
                             best_loss=top[0]["loss"] if top else None, top20=top[:20])
    write_json(output / "result.json", result)
    return result
