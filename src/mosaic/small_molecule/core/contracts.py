"""Define the information passed between model preparation, scoring, and updates.

PreparedContext holds the arrays built before the run, including the
lists that map ligand atoms and protein residues to model positions.
Its n_atoms property counts the ligand atom entries. MolecularTrunkOutput
holds the model's distance scores and the same position lists, so a loss
can look up the right ligand-to-pocket pairs.

RawEvaluation records the candidate, its loss, and the gradients used to
change it. ProjectedStep records the new candidate after the update.
Keeping both states separate makes it clear which ligand a score came from.
"""

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import numpy as np

from .state import BOND_STATES, ELEMENTS, RelaxedGraph

CONTRACT_VERSION = "small_molecule_v0.1"
CHECKPOINT_VERSION = 1
GEOMETRY_MODE = "off"
RNG_POLICIES = ("fixed", "split")


class PreparedContext(eqx.Module):
    """Prepared once outside differentiation, with explicit occupied slot maps.

    Array features are immutable JAX values. Injection creates a new dictionary;
    it never modifies this context. Native reference features exist only for the
    separately labelled hard/conformer baseline, not as an optimization cache.
    """

    base_features: dict[str, Any]
    ligand_atom_indices: tuple[int, ...] = eqx.field(static=True)
    ligand_token_indices: tuple[int, ...] = eqx.field(static=True)
    protein_token_indices: tuple[int, ...] = eqx.field(static=True)
    reference_geometry_mask: jax.Array
    atom_names: tuple[str, ...] = eqx.field(static=True)
    geometry_mode: str = eqx.field(static=True, default=GEOMETRY_MODE)
    identities: dict[str, str] = eqx.field(static=True, default_factory=dict)
    native_features: dict[str, Any] | None = None

    @property
    def n_atoms(self) -> int:
        return len(self.ligand_atom_indices)

    @property
    def n_tokens(self) -> int:
        return int(self.base_features["restype"].shape[0])


class MolecularTrunkOutput(eqx.Module):
    """Real native distogram output, with no fabricated protein-only fields.

    ``distogram_bins`` are the audited 64 upper class edges, including +inf.
    A strict cutoff selects complete bins below it; a straddling class is
    excluded. This is a declared discrete-bin score, not an exact distance CDF.
    """

    distogram_logits: jax.Array
    distogram_bins: jax.Array
    ligand_token_indices: tuple[int, ...] = eqx.field(static=True)
    protein_token_indices: tuple[int, ...] = eqx.field(static=True)
    trunk_state: Any = None


@dataclass(frozen=True, slots=True)
class RawEvaluation:
    """The loss and raw derivatives belong to this exact pre-update graph."""

    graph: RelaxedGraph
    value: Any
    aux: Any
    gradients: RelaxedGraph
    key: jax.Array


@dataclass(frozen=True, slots=True)
class ProjectedStep:
    evaluation: RawEvaluation
    next_graph: RelaxedGraph
    eta_X: float
    eta_E: float
    projection_residuals: dict[str, float]


def validate_context(context: PreparedContext) -> None:
    """Check that Protenix prepared the intended ligand and protein inputs."""
    ligand_atoms = list(context.ligand_atom_indices)
    ligand_tokens = set(context.ligand_token_indices)
    protein_tokens = set(context.protein_token_indices)
    atom_tokens = np.asarray(context.base_features["atom_to_token_idx"])
    if (
        not ligand_atoms
        or len(ligand_atoms) != len(ligand_tokens)
        or not np.array_equal(atom_tokens[ligand_atoms], context.ligand_token_indices)
        or any(np.count_nonzero(atom_tokens == token) != 1 for token in ligand_tokens)
    ):
        raise ValueError("The model must have one token per ligand atom, in the same order.")
    if ligand_tokens & protein_tokens or ligand_tokens | protein_tokens != set(range(context.n_tokens)):
        raise ValueError("Every model token must belong to either the ligand or the protein.")

    geometry = np.asarray(context.reference_geometry_mask)
    protein_atoms = np.isin(atom_tokens, context.protein_token_indices)
    if np.any(geometry[ligand_atoms] != 0) or np.any(geometry[protein_atoms] != 1):
        raise ValueError("Turn off ligand reference geometry and keep protein geometry.")

    # Turning off ligand coordinates must not hide the atoms themselves.
    for field, indices in (
        ("atom_pad_mask", ligand_atoms),
        ("token_pad_mask", list(context.ligand_token_indices)),
        ("ref_mask", ligand_atoms),
    ):
        if field in context.base_features and np.any(np.asarray(context.base_features[field])[indices] != 1):
            raise ValueError(f"{field} hides a ligand atom needed by the model.")


def contract_manifest() -> dict[str, Any]:
    return {
        "version": CONTRACT_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "elements": list(ELEMENTS),
        "bonds": list(BOND_STATES),
        "pair_order": "lexicographic i<j",
        "geometry_mode": GEOMETRY_MODE,
        "occupancy": "all ligand heavy-atom slots occupied",
        "formal_charge": 0,
        "feature_loss": "bond existence only; no bond-order, hydrogen, stereo or ligand geometry channels",
    }
