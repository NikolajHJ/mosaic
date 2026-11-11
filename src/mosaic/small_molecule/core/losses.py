"""Convert the model's distance predictions into a loss to minimize.

For each ligand atom and pocket residue, the model gives scores for
several distance ranges. The contact losses turn those scores into
probabilities and add up the ranges below the chosen contact distance.

The contact loss rewards each ligand atom for having a strong predicted
contact with at least one pocket residue. The pocket coverage loss looks
from the residue side and rewards having a nearby ligand atom across
the pocket. These are ways to guide the optimizer, not affinity estimates.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from mosaic.common import LossTerm

from .contracts import MolecularTrunkOutput
from .state import RelaxedGraph


class NativeContactLoss(LossTerm):
    """Negative mean ligand best-target contact probability (debugging score).

    ``S = mean_i_in_ligand max_j_in_protein sum_bins<cutoff softmax(logits_ij)``.
    Optional fixed native ``pocket_token_indices`` restrict only the partner
    set; no requirement to touch every pocket residue and no new coverage term.
    The native-bin upper-edge convention excludes the cutoff-straddling class.
    This is not an affinity, binding or chemical-validity prediction.
    """

    cutoff_angstrom: float = eqx.field(static=True, default=8.0)
    pocket_token_indices: tuple[int, ...] | None = eqx.field(static=True, default=None)


    def __call__(self, graph: RelaxedGraph, output: MolecularTrunkOutput, *, key):
        ligand = jnp.asarray(output.ligand_token_indices, dtype=jnp.int32)
        partners = output.protein_token_indices if self.pocket_token_indices is None else self.pocket_token_indices
        if not set(partners).issubset(output.protein_token_indices) or set(partners) & set(output.ligand_token_indices):
            raise ValueError("Pocket IDs must belong only to the native protein map")
        protein = jnp.asarray(partners, dtype=jnp.int32)
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
            "pocket_restricted": jnp.asarray(self.pocket_token_indices is not None),
            "contact_partner_count": jnp.asarray(len(partners)),
        }


class DBMolContactLoss(LossTerm):
    """DBMol-inspired pocket coverage and anchor contact, minimized together.

    Pocket entries are fixed native protein token IDs, not positions in the
    protein map or inferred nearest neighbours. With pocket residue ``i`` and
    occupied ligand slot ``j``, ``pair_ij = -log(P_ij(contact) + log_epsilon)``.
    The pocket term is ``mean_i min_j pair_ij``; the anchor term is
    ``min_ij pair_ij``. Their equally weighted mean retains both reductions.

    Contact probability uses native classes whose *upper* edge is strictly
    below the cutoff. This conservative complete-bin score differs from an
    exact distance CDF and excludes the cutoff-straddling class. The additive
    logarithm epsilon follows the declared objective; it is not clipping.
    This score supplies no affinity, confidence or chemical-validity claim.
    The frozen protein pocket stays outside the differentiated graph state.
    """

    pocket_token_indices: tuple[int, ...] = eqx.field(static=True)
    cutoff_angstrom: float = eqx.field(static=True, default=8.0)
    log_epsilon: float = eqx.field(static=True, default=1e-8)


    def __call__(self, graph: RelaxedGraph, output: MolecularTrunkOutput, *, key):
        if not set(self.pocket_token_indices).issubset(output.protein_token_indices):
            raise ValueError("Every pocket token ID must belong to the protein map")
        logits = output.distogram_logits
        ligand = jnp.asarray(output.ligand_token_indices, dtype=jnp.int32)
        pocket = jnp.asarray(self.pocket_token_indices, dtype=jnp.int32)
        probabilities = jax.nn.softmax(
            logits[pocket[:, None], ligand[None, :], :], axis=-1
        )
        contacts = jnp.sum(
            probabilities * (output.distogram_bins < self.cutoff_angstrom), axis=-1
        )
        pair_loss = -jnp.log(contacts + self.log_epsilon)
        pocket_loss = jnp.mean(jnp.min(pair_loss, axis=1))
        anchor_loss = jnp.min(pair_loss)
        value = 0.5 * (pocket_loss + anchor_loss)
        return value, {
            "dbmol_contact_loss": value,
            "pocket_contact_loss": pocket_loss,
            "anchor_contact_loss": anchor_loss,
            "contact_cutoff_angstrom": jnp.asarray(self.cutoff_angstrom),
            "contact_log_epsilon": jnp.asarray(self.log_epsilon),
            "pocket_token_count": jnp.asarray(len(self.pocket_token_indices)),
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


    def __call__(self, graph: RelaxedGraph, output: MolecularTrunkOutput, *, key):
        i, j = self.ligand_pair
        if max(i, j) >= len(output.ligand_token_indices):
            raise ValueError("Diagnostic ligand pair is outside the prepared slots")
        if self.bin_index >= output.distogram_logits.shape[-1]:
            raise ValueError("Diagnostic bin is outside the native distogram classes")
        a, b = output.ligand_token_indices[i], output.ligand_token_indices[j]
        value = output.distogram_logits[a, b, self.bin_index]
        return value, {"diagnostic_only": jnp.asarray(True), "diagnostic_logit": value}
