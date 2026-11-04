"""Paired independent and structural proposals for saved neutral N25 graphs."""

import numpy as np


NODES = 25
UPPER = np.triu_indices(NODES, 1)
ATOM_SHAPE = (NODES, 6)
BOND_SHAPE = (len(UPPER[0]), 5)


def proposal_seed(seed, checkpoint_index, attempt):
    """Derive one reproducible proposal stream from its stable study indices."""
    sequence = np.random.SeedSequence([230017, seed, checkpoint_index, attempt])
    return int(sequence.generate_state(1)[0])


def _probabilities(values, shape):
    probabilities = np.asarray(values, dtype=np.float64)
    if (
        probabilities.shape != shape
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
        or not np.allclose(probabilities.sum(-1), 1, rtol=1e-5, atol=1e-6)
    ):
        raise ValueError("Require normalized finite probabilities in the neutral N25 schema")
    return probabilities


def independent_proposal(atom_probabilities, bond_probabilities, seed):
    """Sample every atom and unordered bond exactly once, without rejection."""
    rng = np.random.Generator(np.random.PCG64(seed))
    candidate = []
    for values, shape in ((atom_probabilities, ATOM_SHAPE), (bond_probabilities, BOND_SHAPE)):
        cumulative = np.cumsum(_probabilities(values, shape), axis=-1)
        uniforms = rng.random((shape[0], 1)) * cumulative[:, -1:]
        candidate.append(np.sum(uniforms >= cumulative, axis=-1).astype(np.int64))
    return tuple(candidate)


def structural_proposal(candidate, atom_probabilities, bond_probabilities, *, search=None, config=None):
    """Apply the existing bounded structural builder to a paired raw draw."""
    from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
    from mosaic.small_molecule.chemistry.state import mosaic_spec
    from mosaic.small_molecule.chemistry.structural import search_structural_repair

    atoms, bonds = (np.asarray(values) for values in candidate)
    for labels, (rows, categories) in ((atoms, ATOM_SHAPE), (bonds, BOND_SHAPE)):
        if (
            labels.shape != (rows,)
            or labels.dtype.kind not in "iu"
            or np.any((labels < 0) | (labels >= categories))
        ):
            raise ValueError("Candidate labels do not match the neutral N25 schema")

    spec = mosaic_spec(NODES)
    atom_rows = _probabilities(atom_probabilities, ATOM_SHAPE)
    bond_rows = _probabilities(bond_probabilities, BOND_SHAPE)
    charge_rows = np.ones((NODES, 1), dtype=np.float64)
    padded_atoms = np.pad(atom_rows, ((0, 0), (1, 0)))
    symmetric_bonds = np.zeros((NODES, NODES), dtype=np.int64)
    symmetric_bonds[UPPER] = symmetric_bonds.T[UPPER] = bonds
    symmetric_rows = np.zeros((NODES, NODES, BOND_SHAPE[1]), dtype=np.float64)
    symmetric_rows[UPPER] = symmetric_rows.transpose(1, 0, 2)[UPPER] = bond_rows
    diagonal = np.arange(NODES)
    symmetric_rows[diagonal, diagonal, 0] = 1

    result = search_structural_repair(
        (atoms.astype(np.int64) + 1, np.zeros(NODES, dtype=np.int64), symmetric_bonds),
        (padded_atoms, charge_rows, symmetric_rows),
        spec,
        ProbabilityRepairConfig() if search is None else search,
        config,
    )
    status = result["status"]
    if status == "already_valid":
        return (atoms.copy(), bonds.copy()), status
    if status in ("unsupported", "censored"):
        return None, status
    if status != "witness_found":
        raise RuntimeError(f"Unexpected structural proposal status {status!r}")
    labels = result["weighted_witness"]["labels"]
    if not np.array_equal(labels["atoms"], atoms + 1) or np.any(labels["charges"]):
        raise RuntimeError("Structural repair changed the fixed atoms or charges")
    return (atoms.copy(), np.asarray(labels["bonds"], dtype=np.int64)[UPPER]), status
