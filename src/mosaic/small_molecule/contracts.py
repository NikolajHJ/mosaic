"""The shared Stage 0 boundaries; no upstream imports or checkpoint loading."""

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
    """Host-side preparation/resume checks; never call inside a model JIT."""
    if context.geometry_mode != GEOMETRY_MODE:
        raise ValueError(f"Unsupported relaxed geometry mode: {context.geometry_mode}")
    if not context.ligand_atom_indices or len(context.ligand_atom_indices) != len(
        context.ligand_token_indices
    ):
        raise ValueError("One occupied ligand token per heavy-atom slot is required")
    atom_count = int(context.base_features["ref_element"].shape[0])
    token_count = context.n_tokens
    for name, indices, limit in (
        ("ligand atoms", context.ligand_atom_indices, atom_count),
        ("ligand tokens", context.ligand_token_indices, token_count),
        ("protein tokens", context.protein_token_indices, token_count),
    ):
        if len(indices) != len(set(indices)) or any(
            not isinstance(i, int) or i < 0 or i >= limit for i in indices
        ):
            raise ValueError(f"Invalid or duplicated {name} indices")
    if set(context.ligand_token_indices) & set(context.protein_token_indices):
        raise ValueError("Ligand and protein token sets overlap")
    if (
        len(context.atom_names) != context.n_atoms
        or len(set(context.atom_names)) != context.n_atoms
    ):
        raise ValueError("Ligand slot names must be fixed and unique")
    if set(context.ligand_token_indices) | set(context.protein_token_indices) != set(
        range(token_count)
    ):
        raise ValueError(
            "Every native token must belong to the declared ligand or protein context"
        )
    mapping = np.asarray(context.base_features["atom_to_token_idx"])
    if mapping.shape != (atom_count,) or not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError("Native atom/token mapping must be an integer vector")
    if np.any(mapping < 0) or np.any(mapping >= token_count):
        raise ValueError("Native atom/token mapping contains an out-of-range token")
    if not np.array_equal(
        mapping[list(context.ligand_atom_indices)], context.ligand_token_indices
    ):
        raise ValueError("Ligand atom/token map disagrees with native features")
    if any(
        np.count_nonzero(mapping == token) != 1
        for token in context.ligand_token_indices
    ):
        raise ValueError(
            "Each ligand token must contain exactly one occupied native atom"
        )
    geometry = np.asarray(context.reference_geometry_mask)
    if geometry.shape != (atom_count,) or not np.isfinite(geometry).all():
        raise ValueError(
            "Reference geometry mask must be finite, one value per native atom"
        )
    if not np.isin(geometry, (0, 1)).all() or np.any(
        geometry[list(context.ligand_atom_indices)] != 0
    ):
        raise ValueError(
            "Ligand reference geometry must be disabled independently of occupancy"
        )
    protein_atoms = np.isin(mapping, context.protein_token_indices)
    if np.any(geometry[protein_atoms] != 1):
        raise ValueError("Native protein reference geometry must remain enabled")
    for field, indices in (
        ("atom_pad_mask", context.ligand_atom_indices),
        ("token_pad_mask", context.ligand_token_indices),
    ):
        if field in context.base_features:
            mask = np.asarray(context.base_features[field])
            expected = atom_count if field == "atom_pad_mask" else token_count
            if mask.shape != (expected,) or np.any(mask[list(indices)] != 1):
                raise ValueError(f"Ligand occupancy cannot be disabled by {field}")
    ref_mask = np.asarray(context.base_features["ref_mask"])
    if ref_mask.shape != (atom_count,) or np.any(
        ref_mask[list(context.ligand_atom_indices)] != 1
    ):
        raise ValueError(
            "The native ligand identity/element embedding gate must remain active"
        )
    charge = np.asarray(context.base_features["ref_charge"])
    if (
        charge.shape != (atom_count,)
        or not np.isfinite(charge).all()
        or np.any(charge[list(context.ligand_atom_indices)] != 0)
    ):
        raise ValueError("V0 ligand formal charge is fixed at zero")


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
