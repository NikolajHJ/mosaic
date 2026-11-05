"""Commented, sequential version of protenij.py for studying protein design.

This preserves the notebook's model, inputs, losses and optimization settings.
Marimo cells become ordinary statements in dependency order. Notebook messages
become comments/prints, and the download button becomes a file write.
Not run or validated: this is a reading companion, not a tuned experiment.
"""

import importlib.resources

import equinox as eqx  # Included in the original notebook; unused below.
import gemmi
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from protenix.protenij import TrunkEmbedding  # Original, unused import.

import mosaic
import mosaic.losses.structure_prediction as sp
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.models.protenix import ProtenixMini, Protenix2025, ProtenixBase
from mosaic.notebook_utils import pdb_viewer
from mosaic.optimizers import simplex_APGM
from mosaic.proteinmpnn.mpnn import ProteinMPNN
from mosaic.structure_prediction import TargetChain


def main():
    # 1. Load the frozen prediction model. The sequence is optimized, not the
    # model weights. Despite the notebook filename, this selects Protenix v1.0.
    print("Demo design using Protenix v1.0")
    protenix = Protenix2025()
    binder_length = 120

    # 2. Read the target protein and extract its first chain's sequence.
    # IL7RA.cif is relative to the directory from which the script is launched.
    target_structure = gemmi.read_structure("IL7RA.cif")
    target_structure.remove_ligands_and_waters()
    target_sequence = gemmi.one_letter_code(
        [r.name for r in target_structure[0][0]]
    )

    # 3. Prepare inputs for the binder-target complex. The target sequence is
    # fixed; its structure is provided as a template and an MSA is requested.
    # The writer (design_structure) later turns predictions into a structure.
    design_features, design_structure = protenix.binder_features(
        binder_length=binder_length,
        chains=[
            TargetChain(
                target_sequence,
                use_msa=True,
                template_chain=target_structure[0][0],
            )
        ],
    )

    # 4. Load soluble ProteinMPNN. Its inverse-folding loss encourages a
    # sequence that is compatible with the structure predicted during design.
    mpnn = ProteinMPNN.from_pretrained(
        importlib.resources.files(mosaic)
        / "proteinmpnn/weights/soluble_v_48_020.pt"
    )

    # 5. Combine objectives. Coefficients are weights, not percentages.
    # No epitope_idx is supplied, so contacts consider the whole target.
    # PAE measures uncertainty in relative placement; pLDDT measures local
    # confidence; ipTM concerns the interface. Losses encode the direction
    # needed for minimization. Terms multiplied by zero are inactive.
    structure_loss = (
        sp.BinderTargetContact()
        + 1 * sp.WithinBinderContact()
        + 0.05 * sp.TargetBinderPAE()
        + 0.05 * sp.BinderTargetPAE()
        + 0.025 * sp.IPTMLoss()
        + 0.4 * sp.WithinBinderPAE()
        + 0.025 * sp.pTMEnergy()
        + 0.1 * sp.PLDDTLoss()
        + 10.0 * InverseFoldingSequenceRecovery(mpnn, temp=jax.numpy.array(0.001))
        - 0.0 * sp.HelixLoss()
        + 0.0 * sp.BinderTargetIPSAE()
        + 0.0 * sp.TargetBinderIPSAE()
    )

    # 6. Connect those objectives to Protenix and the prepared inputs.
    # The notebook notes that optimization can be finicky and that increasing
    # recycling_steps above one can help. Its original setting is retained.
    loss = protenix.build_multisample_loss(
        loss=structure_loss,
        features=design_features,
        recycling_steps=1,
        sampler=protenix.design_sampler(20),
    )

    # The notebook warns that the first JAX compilation can take four minutes.
    # Later calls with compatible shapes can reuse compiled calculations.
    print("The first Protenix JIT compilation may take several minutes.")

    # 7. Initialize a soft sequence: one row per binder position, with
    # probabilities over 20 amino acids. Gumbel noise gives a random start;
    # softmax makes each row sum to one. Numpy chooses a new seed each run.
    PSSM = jax.nn.softmax(
        0.5
        * jax.random.gumbel(
            key=jax.random.key(np.random.randint(1000000)),
            shape=(binder_length, 20),
        )
    )

    # 8. Optimize the soft sequence for 100 steps using projected gradients.
    # Projection keeps the rows as valid amino-acid probability distributions.
    _, PSSM = simplex_APGM(
        loss_function=loss,
        x=PSSM,
        n_steps=100,
        stepsize=0.15 * np.sqrt(binder_length),
        momentum=0.3,
        scale=1.0,
        max_gradient_norm=1.0,
    )

    # 9. A further 20 steps, labelled "sharpening" in the notebook.
    # All arguments and return-value selections below match the source,
    # including log(PSSM), logspace=False and selecting the first return value.
    # These choices are reproduced for study, rather than silently revised.
    PSSM_sharper, _ = simplex_APGM(
        loss_function=loss,
        x=jnp.log(PSSM + 1e-5),
        n_steps=20,
        stepsize=0.5 * np.sqrt(binder_length),
        momentum=0.0,
        scale=1.3,
        logspace=False,
        max_gradient_norm=1.0,
    )
    plt.figure()
    plt.imshow(PSSM_sharper)
    plt.title("Designed amino-acid probabilities")

    # 10. Re-predict with four recycling steps to evaluate the design.
    # Marimo resolves dependencies automatically; in a regular script this
    # prediction must precede every display that refers to protenix_pred.
    print("Let's evaluate our design")
    protenix_pred = protenix.predict(
        PSSM=PSSM_sharper,
        features=design_features,
        recycling_steps=4,
        key=jax.random.key(0),
        writer=design_structure,
    )

    # The original interactive structure viewer is retained. A standalone
    # script does not automatically display the returned notebook widget.
    viewer = pdb_viewer(protenix_pred.st)

    # 11. Inspect uncertainty and confidence across binder and target.
    plt.figure()
    plt.imshow(protenix_pred.pae)
    plt.title("Predicted aligned error (PAE)")
    plt.figure()
    plt.plot(protenix_pred.plddt)
    plt.vlines(
        [binder_length],
        protenix_pred.plddt.min(),
        protenix_pred.plddt.max(),
        linestyle="dashed",
        color="red",
    )
    plt.title("pLDDT; red line separates binder and target")
    print("Interface predicted TM score (ipTM):", protenix_pred.iptm)

    # 12. Replace the notebook's download button with the same PDB output.
    with open("protenix_complex.pdb", "w", encoding="utf-8") as output:
        output.write(protenix_pred.st.make_minimal_pdb())
    plt.show()
    return protenix_pred, viewer


if __name__ == "__main__":
    main()
