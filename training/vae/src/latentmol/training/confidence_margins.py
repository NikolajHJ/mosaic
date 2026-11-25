"""Encoder-only excess-confidence loss on exact generated graph round trips."""

import math

from rdkit import Chem
import torch

from latentmol.chemistry import MoleculeRejected, canonical_slot_order


def validate_excess_confidence(policy):
    if (not isinstance(policy, dict) or set(policy) != {"weight", "tolerance"}
            or any(type(value) not in (float, int) or not math.isfinite(value) or value < 0
                   for value in policy.values())):
        raise ValueError("excess_confidence requires finite nonnegative weight and logit tolerance")


def _exact_order(hard, row, targets, spec):
    """Canonical target slot -> output slot, only for an exact label-preserving match."""
    a, q, b = hard.atoms[row], hard.charges[row], hard.bonds[row]
    try:
        order = torch.tensor(canonical_slot_order(a, q, b, spec))
    except MoleculeRejected:
        return None
    if (torch.equal(a[order], targets.atoms[row]) and torch.equal(q[order], targets.charges[row])
            and torch.equal(b[order[:, None], order[None, :]], targets.bonds[row])):
        return order
    return None


def _site_classes(atoms, charges, bonds, spec):
    """Symmetry classes for occupied atoms and present undirected bonds in canonical slots.

    A colored vertex for each bond lets RDKit rank bonds themselves, including bonds
    with symmetry-equivalent endpoints. Neither sanitization nor coordinates enter.
    """
    active = atoms.nonzero().flatten()
    edges = torch.nonzero(torch.triu(bonds, diagonal=1))
    graph = Chem.RWMol()
    def add_site(color):
        vertex = Chem.Atom(0)
        vertex.SetNoImplicit(True)
        vertex.SetIsotope(int(color))
        return graph.AddAtom(vertex)
    charge_types = len(spec.formal_charges)
    nodes = {i: add_site(1 + int(atoms[i]) * charge_types + int(charges[i])) for i in active.tolist()}
    for i, j in edges.tolist():
        vertex = add_site(1 + len(spec.atomic_numbers) * charge_types + int(bonds[i, j]))
        graph.AddBond(nodes[i], vertex, Chem.BondType.SINGLE)
        graph.AddBond(vertex, nodes[j], Chem.BondType.SINGLE)
    graph.UpdatePropertyCache(strict=False)
    classes = torch.tensor(list(Chem.CanonicalRankAtoms(graph, breakTies=False)))
    return active, edges, classes[:len(active)], classes[len(active):]


def _grouped_margins(logits, classes):
    top = logits.topk(2, dim=-1).values
    margins = top[:, 0] - top[:, 1]
    # Compare the ordered gaps within each symmetry class. Equivalent sites have
    # no unique correspondence; arbitrary permutations must not incur a penalty.
    by_gap = margins.detach().argsort(stable=True)
    return margins[by_gap[classes[by_gap].argsort(stable=True)]]


def matched_confidence_loss(predicted, reference, targets, spec, tolerance):
    """One-sided mean logit-gap excess over exact parents, with equal weight per parent.

    Reference logits are deterministic and detached. Every contributing parent has
    the same atom, charge and bond labels at both points, up to slot relabeling.
    Head means are averaged for occupied atoms and existing bonds (atoms alone
    for a one-atom graph). Missing edges, diagonals and padded slots are excluded.
    """
    parents = targets.to("cpu")
    student = predicted.harden(spec).to("cpu")
    teacher = reference.harden(spec).to("cpu")
    teacher_matches = student_matches = 0
    losses = []
    p_bonds, r_bonds = predicted.symmetric_bond_logits, reference.symmetric_bond_logits.detach()
    device = predicted.atoms.device
    for row in range(len(parents.atoms)):
        t_order = _exact_order(teacher, row, parents, spec)
        s_order = _exact_order(student, row, parents, spec)
        teacher_matches += int(t_order is not None)
        student_matches += int(s_order is not None)
        if t_order is None or s_order is None:
            continue
        active, edges, atoms_cls, bond_cls = _site_classes(
            parents.atoms[row], parents.charges[row], parents.bonds[row], spec)
        def hinge(student_logits, teacher_logits, classes):
            classes = classes.to(device)
            excess = _grouped_margins(student_logits, classes) - _grouped_margins(teacher_logits, classes)
            return (excess - tolerance).relu().mean()
        a = s_order[active].to(device)
        b = t_order[active].to(device)
        atom_loss = hinge(predicted.atoms[row, a], reference.atoms[row, b].detach(), atoms_cls)
        if len(edges):
            i, j = s_order[edges].to(device).T
            k, l = t_order[edges].to(device).T
            bond_loss = hinge(p_bonds[row, i, j], r_bonds[row, k, l], bond_cls)
            losses.append((atom_loss + bond_loss) * .5)
        else:
            losses.append(atom_loss)
    loss = torch.stack(losses).mean() if losses else predicted.atoms.new_zeros(())
    return loss, dict(reference_matches=teacher_matches, posterior_matches=student_matches,
                      exact_pairs=len(losses), loss=float(loss.detach()))
