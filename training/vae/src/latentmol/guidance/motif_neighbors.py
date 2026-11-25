"""Learn local neighborhoods of naturally occurring motifs in frozen-label prior decodes."""

import math

import numpy as np
from rdkit import rdBase
import torch
from torch.nn import functional as F

from latentmol.chemistry import _label_molecule
from latentmol.guidance.motif_completion import AMIDE, _roundtrips, _valid


def _natural_sites(hard, spec):
    """Select one actually present site per group per molecule, up to two sites per group."""
    arrays = [x.cpu().numpy() for x in (hard.atoms, hard.charges, hard.bonds)]
    counts = {"aromatic": 0, "amide": 0}
    sites = []
    for row, (a, c, b) in enumerate(zip(*arrays)):
        slots = np.flatnonzero(a)
        if not len(slots):
            continue
        mol = _label_molecule(a, c, b[np.ix_(slots, slots)], spec, slots)
        if not _valid(mol) or not _roundtrips(mol):
            continue
        motifs = []
        if counts["aromatic"] < 2:
            for ring in mol.GetRingInfo().AtomRings():
                edges = list(zip(ring, ring[1:] + ring[:1]))
                if all(mol.GetBondBetweenAtoms(i, j).GetIsAromatic() for i, j in edges):
                    motifs.append(("aromatic", edges))
                    break
        if counts["amide"] < 2 and (match := mol.GetSubstructMatch(AMIDE)):
            carbon, oxygen, nitrogen = match
            motifs.append(("amide", [(carbon, oxygen), (carbon, nitrogen)]))
        for name, edges in motifs:
            core = sorted({tuple(sorted((int(slots[i]), int(slots[j])))) for i, j in edges})
            nodes = sorted({k for edge in core for k in edge})
            incident = {tuple(sorted((i, int(j)))) for i in nodes for j in np.flatnonzero(b[i])}
            sites.append((row, name, nodes, core, sorted(incident.difference(core))))
            counts[name] += 1
        if all(n == 2 for n in counts.values()):
            break
    return sites, counts


def _presence(decoded):
    """Probability that both slots occur and their bond is present under the decoder law."""
    atom = decoded.atoms.float()
    bond = decoded.symmetric_bond_logits.float()
    occupied = (torch.logsumexp(atom[..., 1:], -1) - atom[..., 0]).sigmoid()
    connected = (torch.logsumexp(bond[..., 1:], -1) - bond[..., 0]).sigmoid()
    return occupied[:, None, :] * occupied[:, :, None] * connected


def _incident_kl(source, neighbor, hard_bonds, nodes):
    """Average presence KL within each available class, avoiding dilution by numerous absent pairs."""
    incident = torch.zeros_like(source, dtype=torch.bool)
    incident[nodes, :] = True
    incident[:, nodes] = True
    i, j = torch.where(incident.triu(1))
    p = source[i, j].clamp(1e-6, 1 - 1e-6)
    q = neighbor[i, j].clamp(1e-6, 1 - 1e-6)
    kl = p * (p.log() - q.log()) + (1 - p) * (torch.log1p(-p) - torch.log1p(-q))
    present = hard_bonds[i, j] != 0
    # Handle an all-present local neighborhood without a device synchronization.
    positive, negative = present.sum(), (~present).sum()
    sums = ((kl * present).sum() / positive.clamp_min(1) + (kl * ~present).sum() / negative.clamp_min(1))
    return sums / ((positive > 0).to(kl.dtype) + (negative > 0))


def prior_motif_neighbor_loss(model, z, decoded, spec, generator, *, rho=.995):
    """Supervise only mapped motifs and their incident edges on correlated Gaussian neighbors.

    Targets are observations on the existing detached center forward. RDKit supplies slot
    membership. The original raw labels supply bond targets, retaining Kekulé encodings.
    Only the auxiliary neighboring decode contributes parameter gradients. Each of at most
    four sites is normalized by the entire original batch.
    """
    batch = len(decoded.atoms)
    if (not batch or z.ndim != 2 or len(z) != batch or spec.bond_names[0] != "NONE"
            or not math.isfinite(rho) or not 0 <= rho <= 1 or not isinstance(generator, torch.Generator)):
        raise ValueError("Incompatible motif-neighbor prior, generator, graph schema or correlation")
    with torch.no_grad(), rdBase.BlockLogs():
        hard = decoded.harden(spec)
        sites, counts = _natural_sites(hard, spec)
    centers = sorted({s[0] for s in sites})
    stats = dict(neighbor_aromatic_sites=counts["aromatic"], neighbor_amide_sites=counts["amide"],
                 neighbor_center_count=len(centers))
    if not centers:
        return decoded.bonds[0, 0, 0, 0] * 0, stats
    with torch.no_grad():
        old = z[centers].detach()
        new_noise = torch.randn(old.shape, dtype=old.dtype, generator=generator, device=generator.device).to(old.device)
        neighbor_z = rho * old + math.sqrt(1 - rho * rho) * new_noise
        # Restrict soft-teacher work to the selected centers.
        teacher = type(decoded)(*(t[centers] for t in (decoded.atoms, decoded.charges, decoded.bonds)))
        source_presence = _presence(teacher)
    # Disable dropout during the neighbor pass so proximity is not obscured by a new dropout draw.
    # Restore any mixed per-module train/eval flags. The fast path uses a private noise
    # generator; fork_rng also protects callers if an eval-mode decoder uses randomness.
    flags = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.random.fork_rng(devices=[z.device] if z.is_cuda else []):
            neighbor = model.decode(neighbor_z)
    finally:
        for module, flag in flags:
            module.training = flag
    output = neighbor.symmetric_bond_logits.float()
    neighbor_presence = _presence(neighbor)
    sums = output.new_zeros(5)
    for row, _, nodes, core, shell in sites:
        k = centers.index(row)
        core_i, core_j = torch.as_tensor(core, device=output.device).T
        core_ce = F.cross_entropy(output[k, core_i, core_j, 1:], hard.bonds[row, core_i, core_j] - 1)
        shell_ce = core_ce * 0
        if shell:
            i, j = torch.as_tensor(shell, device=output.device).T
            shell_ce = F.cross_entropy(output[k, i, j, 1:], hard.bonds[row, i, j] - 1)
        atom_ce = F.cross_entropy(neighbor.atoms[k, nodes].float(), hard.atoms[row, nodes])
        charge_ce = F.cross_entropy(neighbor.charges[k, nodes].float(), hard.charges[row, nodes])
        local_kl = _incident_kl(source_presence[k], neighbor_presence[k], hard.bonds[row], nodes)
        sums = sums + torch.stack([core_ce, shell_ce, atom_ce, charge_ce, local_kl])
    terms = sums / batch
    stats.update(zip(("neighbor_core_ce", "neighbor_shell_ce", "neighbor_atom_ce", "neighbor_charge_ce",
                      "neighbor_topology_kl"), terms.detach().cpu().tolist()))
    return torch.dot(terms, terms.new_tensor([1., .25, .25, .10, .5])), stats
