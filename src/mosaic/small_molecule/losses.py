"""Native distogram reductions with explicit ligand/target token mappings."""

import math
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp

from mosaic.common import LossTerm

from .contracts import MolecularTrunkOutput
from .state import RelaxedGraph


class NativeContactLoss(LossTerm):
    """Negative mean ligand best-target contact probability (debugging score).

    ``S = mean_i_in_ligand max_j_in_protein sum_bins<cutoff softmax(logits_ij)``.
    The native-bin upper-edge convention excludes the cutoff-straddling class.
    Optional protein_positions select zero-based positions within the explicit
    protein token list. None preserves the original whole-protein reduction.
    This is not an affinity, binding or chemical-validity prediction.
    """

    cutoff_angstrom: float = eqx.field(static=True, default=8.0)
    protein_positions: tuple[int, ...] | None = eqx.field(static=True, default=None)

    def __post_init__(self):
        if not math.isfinite(self.cutoff_angstrom) or self.cutoff_angstrom <= 0:
            raise ValueError("Contact cutoff must be positive and finite")
        if self.protein_positions is not None:
            if (
                not isinstance(self.protein_positions, tuple)
                or not self.protein_positions
            ):
                raise ValueError("Pocket protein_positions must be a nonempty tuple")
            if any(
                isinstance(i, bool) or not isinstance(i, Integral) or i < 0
                for i in self.protein_positions
            ):
                raise ValueError("Pocket positions must be nonnegative integers")
            if len(set(self.protein_positions)) != len(self.protein_positions):
                raise ValueError("Pocket positions must be unique")

    def __call__(self, graph: RelaxedGraph, output: MolecularTrunkOutput, *, key):
        if not output.ligand_token_indices or not output.protein_token_indices:
            raise ValueError(
                "NativeContactLoss requires nonempty explicit ligand and protein token sets"
            )
        ligand = jnp.asarray(output.ligand_token_indices, dtype=jnp.int32)
        positions = self.protein_positions
        if positions is not None and max(positions) >= len(
            output.protein_token_indices
        ):
            raise ValueError("Pocket position is outside the protein token set")
        selected = (
            output.protein_token_indices
            if positions is None
            else tuple(output.protein_token_indices[i] for i in positions)
        )
        protein = jnp.asarray(selected, dtype=jnp.int32)
        logits = output.distogram_logits[ligand[:, None], protein[None, :], :]
        probabilities = jax.nn.softmax(logits, axis=-1)
        contacts = jnp.sum(
            probabilities * (output.distogram_bins < self.cutoff_angstrom), axis=-1
        )
        score = jnp.mean(jnp.max(contacts, axis=-1))
        return -score, {
            "contact_score": score,
            "contact_cutoff_angstrom": jnp.asarray(self.cutoff_angstrom),
            "conservative_upper_edge_bins": jnp.asarray(True),
            "diagnostic_only": jnp.asarray(False),
        }


class DistogramDiagnosticLoss(LossTerm):
    """One off-diagonal native logit for engineering derivative checks only.

    This is deliberately labelled ``diagnostic_only``. It is neither a ligand
    design objective nor a substitute ligand confidence/PAE metric.
    Pair indices address ligand slots, never the beginning of the token array.
    """

    ligand_pair: tuple[int, int] = eqx.field(static=True, default=(0, 1))
    bin_index: int = eqx.field(static=True, default=0)

    def __post_init__(self):
        if len(self.ligand_pair) != 2 or self.ligand_pair[0] == self.ligand_pair[1]:
            raise ValueError("The diagnostic requires two distinct ligand slots")
        if any(
            isinstance(i, bool) or not isinstance(i, int) or i < 0
            for i in self.ligand_pair
        ):
            raise ValueError(
                "Diagnostic ligand slots must be nonnegative integer indices"
            )
        if (
            isinstance(self.bin_index, bool)
            or not isinstance(self.bin_index, int)
            or self.bin_index < 0
        ):
            raise ValueError("Diagnostic bin must be a nonnegative integer index")

    def __call__(self, graph: RelaxedGraph, output: MolecularTrunkOutput, *, key):
        i, j = self.ligand_pair
        if max(i, j) >= len(output.ligand_token_indices):
            raise ValueError("Diagnostic ligand pair is outside the prepared slots")
        if self.bin_index >= output.distogram_logits.shape[-1]:
            raise ValueError("Diagnostic bin is outside the native distogram classes")
        a, b = output.ligand_token_indices[i], output.ligand_token_indices[j]
        value = output.distogram_logits[a, b, self.bin_index]
        return value, {"diagnostic_only": jnp.asarray(True), "diagnostic_logit": value}
