"""Match live relaxed ligand chemistry to a frozen PharmacoNet pocket map."""
import math

import torch
from rdkit import Chem


# These deliberately describe a subset of donor/acceptor chemistry. The first
# query atom is the feature center. Use the same classes in both pose readouts.
_FEATURE_QUERIES = {
    "acceptor": ("[O+0;D1]=[#6]", "[O+0;D2](-[#6])-[#6]", "[N+0;D1]#[#6]", "[nH0+0;D2](:[a]):[a]"),
    "donor": ("[O+0;H1;D1]-[#6]", "[N+0;H2;D1]-[#6]", "[N+0;H1;D2](-[#6])-[#6]"),
}
_FEATURE_QUERIES = {k: tuple(Chem.MolFromSmarts(s) for s in queries) for k, queries in _FEATURE_QUERIES.items()}


def hard_feature_atoms(molecule):
    """Indices in a sanitized RDKit molecule, with its original atom ordering."""
    features = {name: {match[0] for query in queries for match in molecule.GetSubstructMatches(query)}
                for name, queries in _FEATURE_QUERIES.items()}
    hydrophobic = set()
    for atom in molecule.GetAtoms():
        neighbors = [a for a in atom.GetNeighbors() if a.GetAtomicNum() != 1]
        if atom.GetFormalCharge() or not neighbors or any(a.GetAtomicNum() != 6 for a in neighbors):
            continue
        if atom.GetAtomicNum() == 6 or (atom.GetAtomicNum() in (17, 35, 53) and len(neighbors) == 1
                and molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbors[0].GetIdx()).GetBondType() == Chem.BondType.SINGLE):
            hydrophobic.add(atom.GetIdx())
    features["hydrophobic"] = hydrophobic
    return features


def hard_feature_groups(molecule):
    """One group per feature; a ring contributes once at its heavy-atom centroid."""
    groups = {name: [(i,) for i in sorted(indices)] for name, indices in hard_feature_atoms(molecule).items()}
    groups["aromatic"] = []
    for ring in molecule.GetRingInfo().AtomRings():
        if len(ring) not in (5, 6):
            continue
        atoms = [molecule.GetAtomWithIdx(i) for i in ring]
        elements = {a.GetAtomicNum() for a in atoms}
        eligible = elements <= {6, 7} if len(ring) == 6 else elements <= {6, 7, 8, 16} and elements != {6}
        if (eligible and all(a.GetFormalCharge() == 0 for a in atoms)
                and all(molecule.GetBondBetweenAtoms(ring[i-1], ring[i]).GetIsAromatic() for i in range(len(ring)))):
            groups["aromatic"].append(tuple(ring))
    return groups


def _relaxed_features(graph, active, spec):
    """Probabilities of complete local features under independent graph decisions.

    Count exactly one or two compatible bonds while requiring every other
    edge at the central atom to be absent. This is well-defined at one-hot
    inputs and prevents an overbonded heteroatom from collecting a feature.
    """
    atoms = graph.atoms[0, active]
    bonds = graph.bonds[0, active][:, active]
    neutral = graph.charges[0, active, spec.formal_charges.index(0)]
    carbon = atoms[:, spec.atomic_numbers.index(6)]
    nitrogen = atoms[:, spec.atomic_numbers.index(7)] * neutral
    oxygen = atoms[:, spec.atomic_numbers.index(8)] * neutral
    neighbor = torch.stack([bonds[..., spec.bond_names.index(b)] * carbon[None, :]
                            for b in ("SINGLE", "DOUBLE", "TRIPLE")] +
                           [bonds[..., spec.bond_names.index("AROMATIC")]], dim=-1)
    shape = (len(atoms), neighbor.shape[-1])
    none = torch.ones_like(neighbor[:, 0, :1])
    one = atoms.new_zeros(shape)
    two = atoms.new_zeros(shape)
    for slot in range(len(atoms)):
        absent = bonds[:, slot, :1]
        bond = neighbor[:, slot]
        two, one, none = (two * absent + one * bond,
                          one * absent + none * bond,
                          none * absent)
    acceptor = oxygen * (one[:, 1] + two[:, 0]) + nitrogen * (one[:, 2] + two[:, 3])
    donor = oxygen * one[:, 0] + nitrogen * (one[:, 0] + two[:, 0])
    absent = bonds[..., 0]
    all_carbon_neighbors = (absent + (1 - absent) * carbon[None, :]).prod(1) - none[:, 0]
    halogen = atoms[:, [spec.atomic_numbers.index(z) for z in (17, 35, 53)]].sum(1)
    hydrophobic = neutral * (carbon * all_carbon_neighbors + halogen * one[:, 0])
    return torch.stack((acceptor, donor, hydrophobic))


def _spatial_rings(xyz):
    # Only the fixed slot geometry is inspected, never reference ligand bonds.
    # These permissive covalent distances cover aromatic C/N/O/S rings. Check
    # both traversal directions once; the min slot starts each cycle.
    distance = torch.cdist(xyz, xyz)
    neighbors = [row.nonzero().flatten().tolist() for row in ((distance >= 1.15) & (distance <= 1.85))]
    rings = {5: [], 6: []}
    for start, adjacent in enumerate(neighbors):
        todo = [(start, i) for i in adjacent if i > start]
        while todo:
            path = todo.pop()
            if len(path) in rings and path[-1] in adjacent and path[1] < path[-1]:
                rings[len(path)].append(path)
                if len(rings[len(path)]) > 4096:
                    raise ValueError("Fixed pose has too many spatial candidate rings")
            if len(path) < 6:
                todo.extend(path + (i,) for i in neighbors[path[-1]] if i > start and i not in path)
    return {n: torch.tensor(paths, dtype=torch.long).reshape(-1, n) for n, paths in rings.items()}


def _ring_pattern(atoms, edges):
    # Harmonic pooling makes a missing edge a bottleneck while giving useful
    # gradients before an entire ring is present. It is a score, not a joint probability.
    parts = torch.cat((atoms, edges), dim=-1)
    eps = 1.e-7
    score = (parts + eps).reciprocal().mean(dim=-1).reciprocal() - eps
    return torch.where(parts.amin(dim=-1) > 0, score, 0.)


def _relaxed_rings(graph, active, spec, spatial_rings):
    atoms = graph.atoms[0, active]
    bonds = graph.bonds[0, active][:, active]
    neutral = graph.charges[0, active, spec.formal_charges.index(0)]
    aromatic_cn = atoms[:, [spec.atomic_numbers.index(z) for z in (6, 7)]].sum(1) * neutral
    lone_pair = atoms[:, [spec.atomic_numbers.index(z) for z in (7, 8, 16)]].sum(1) * neutral
    single, double, aromatic = [spec.bond_names.index(k) for k in ("SINGLE", "DOUBLE", "AROMATIC")]
    scores = []
    for length, ring in spatial_rings.items():
        if not len(ring):
            continue
        ring_bonds = bonds[ring, ring.roll(-1, 1)]
        # Two alternating Kekulé patterns in a six-ring. For a five-ring
        # each cyclic choice of an eligible lone-pair donor gives one pattern.
        n_patterns = 2 if length == 6 else length
        patterns = []
        for start in range(n_patterns):
            index = (torch.arange(length) - start) % length
            types = aromatic_cn[ring]
            if length == 5:
                types = torch.where(index == 0, lone_pair[ring], types)
            is_double = (index % 2 == 1) if length == 5 else (index % 2 == 0)
            edges = torch.where(is_double, ring_bonds[..., double], ring_bonds[..., single])
            patterns.extend((_ring_pattern(types, edges), _ring_pattern(types, ring_bonds[..., aromatic])))
        scores.append(torch.stack(patterns).amax(0))
    return torch.cat(scores) if scores else atoms.new_empty(0)


def registered_reference(map_data, metadata, slots, native_coords):
    """Use CCD atom names to recover deposited coordinates in the map frame."""
    pocket = metadata["target"]["pocket"]
    if (map_data["pdb_id"] != pocket["pdb_id"] or map_data["chain"] != pocket["protein_chain"]):
        raise ValueError("PharmacoNet pocket and Boltz pocket differ")
    rows = sorted(metadata["slot_map"], key=lambda r: r["slot"])
    if [r["slot"] for r in rows] != slots:
        raise ValueError("PharmacoNet reference slot map differs from native occupancy")
    names = [r["atom_name"] for r in rows]
    if len(set(names)) != len(names) or set(map_data["reference_atoms"]) != set(names):
        raise ValueError("PharmacoNet and Boltz reference atom names disagree")
    xyz = torch.tensor([map_data["reference_atoms"][name] for name in names], dtype=torch.float64)
    target = native_coords.detach().cpu().double()
    if xyz.shape != (len(slots), 3) or xyz.shape != target.shape or not xyz.isfinite().all() or not target.isfinite().all():
        raise ValueError("Invalid PharmacoNet or native reference coordinates")
    # Boltz normally translates the reference. A proper rigid rotation is also
    # fine. Check the correspondence in 3D and reject reflections or reordered atoms.
    a, b = xyz - xyz.mean(0), target - target.mean(0)
    u, _, vh = torch.linalg.svd(a.T @ b)
    fix = torch.eye(3, dtype=a.dtype)
    fix[2, 2] = torch.det(u @ vh).sign()
    max_error = float(torch.linalg.vector_norm(a @ (u @ fix @ vh) - b, dim=-1).max())
    if max_error > .03:
        raise ValueError(f"PharmacoNet crystal coordinate registration failed ({max_error:.4g} angstrom)")
    return names, xyz.float(), max_error


class PocketFeatureLoss:
    """Precompute fixed per-hotspot geometry; evaluate soft chemistry on the live graph."""

    def __init__(self, hotspots, reference_xyz):
        self.ids = [row["id"] for row in hotspots]
        if (not self.ids or any(not isinstance(i, str) or not i for i in self.ids)
                or len(set(self.ids)) != len(self.ids)):
            raise ValueError("A nonempty PharmacoNet map with unique hotspot IDs is required")
        roles = ("acceptor", "donor", "hydrophobic", "aromatic")
        if any(row["feature"] not in roles for row in hotspots):
            raise ValueError("Unsupported ligand-side PharmacoNet feature")
        centers = torch.tensor([r["center"] for r in hotspots], dtype=torch.float32)
        radii = torch.tensor([r["radius"] for r in hotspots], dtype=torch.float32)
        weights = torch.tensor([r["score"] for r in hotspots], dtype=torch.float32)
        if (centers.shape != (len(hotspots), 3) or radii.shape != (len(hotspots),) or weights.shape != radii.shape
                or not centers.isfinite().all() or not radii.isfinite().all() or not weights.isfinite().all()
                or radii.min() <= 0 or weights.min() <= 0 or reference_xyz.ndim != 2 or reference_xyz.shape[1] != 3
                or not reference_xyz.isfinite().all() or reference_xyz.requires_grad):
            raise ValueError("Invalid PharmacoNet map geometry or weights")
        self.roles = torch.tensor([roles.index(row["feature"]) for row in hotspots])
        self.weights = torch.zeros_like(weights)
        # A family with six hydrophobic sites gets the same total weight as
        # the single H-bond site. Percentiles split weight within a family.
        families = self.roles.unique()
        for family in families:
            selected = self.roles == family
            self.weights[selected] = weights[selected] / (len(families) * weights[selected].sum())
        xyz = reference_xyz.detach().cpu().float()
        distance = torch.cdist(centers, xyz)
        self.geometry = torch.exp(-.5 * (distance / radii[:, None]).square())
        self.spatial_rings = _spatial_rings(xyz) if "aromatic" in [r["feature"] for r in hotspots] else {}
        centroid_parts = [xyz[ring].mean(dim=1) for ring in self.spatial_rings.values() if len(ring)]
        centroids = torch.cat(centroid_parts) if centroid_parts else xyz.new_empty(0, 3)
        self.ring_geometry = torch.exp(-.5 * (torch.cdist(centers, centroids) / radii[:, None]).square())
        if centroid_parts:
            normals = []
            for ring in self.spatial_rings.values():
                if len(ring):
                    points = xyz[ring]
                    normals.append(torch.linalg.svd(points - points.mean(1, keepdim=True), full_matrices=False).Vh[:, -1])
            normals = torch.cat(normals)
            for i, row in enumerate(hotspots):
                kind = row.get("interaction_type")
                if kind in ("PiStacking_T", "PiStacking_P"):
                    normal = torch.tensor(row["protein_normal"], dtype=xyz.dtype)
                    if normal.shape != (3,) or not normal.isfinite().all() or abs(float(normal.norm()) - 1.) > .02:
                        raise ValueError("Invalid aromatic hotspot protein normal")
                    cos2 = (normals @ (normal / normal.norm())).square().clamp(0., 1.)
                    self.ring_geometry[i] *= 1. - cos2 if kind == "PiStacking_T" else cos2

    def __call__(self, graph, active, spec):
        if int(active.sum()) != self.geometry.shape[1]:
            raise ValueError("PharmacoNet active slot count differs from fixed geometry")
        features = _relaxed_features(graph, active, spec)
        ring_scores = _relaxed_rings(graph, active, spec, self.spatial_rings)
        matches = []
        for i, role in enumerate(self.roles.tolist()):
            if role < len(features):
                matches.append((features[role] * self.geometry[i].to(features)).max())
            elif ring_scores.numel():
                matches.append((ring_scores * self.ring_geometry[i].to(features)).max())
            else:
                # A pose without candidate rings yields zero reachable ring coverage.
                matches.append(features.sum() * 0.)
        match = torch.stack(matches)
        # Each hotspot can be filled once. The floor bounds the penalty and
        # prevents very distant, unreachable hotspots from dominating a trajectory.
        term = math.log(1.05) - (match + .05).log()
        loss = (term * self.weights.to(term)).sum()
        return loss, dict(zip(self.ids, match.detach().cpu().tolist(), strict=True))
