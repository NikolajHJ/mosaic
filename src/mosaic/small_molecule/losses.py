"""Native distogram reductions with explicit ligand/target token mappings."""

import math

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
    This is not an affinity, binding or chemical-validity prediction.
    """

    cutoff_angstrom: float = eqx.field(static=True, default=8.0)

    def __post_init__(self):
        if not math.isfinite(self.cutoff_angstrom) or self.cutoff_angstrom <= 0:
            raise ValueError("Contact cutoff must be positive and finite")

    def __call__(self, graph: RelaxedGraph, output: MolecularTrunkOutput, *, key):
        if not output.ligand_token_indices or not output.protein_token_indices:
            raise ValueError("NativeContactLoss requires nonempty explicit ligand and protein token sets")
        ligand = jnp.asarray(output.ligand_token_indices, dtype=jnp.int32)
        protein = jnp.asarray(output.protein_token_indices, dtype=jnp.int32)
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
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in self.ligand_pair):
            raise ValueError("Diagnostic ligand slots must be nonnegative integer indices")
        if isinstance(self.bin_index, bool) or not isinstance(self.bin_index, int) or self.bin_index < 0:
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
