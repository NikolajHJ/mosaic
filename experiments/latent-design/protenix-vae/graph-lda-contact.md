# Can Protenix improve a molecule decoded by the pretrained ZINC graph VAE?

This pilot optimizes one 90-dimensional latent vector per starting molecule.
The frozen model is the **plain PIG-VAE** from the Graph LDA authors. It uses
their published ZINC weights and encoder posterior mean. This is an inference
experiment. The fitted encoder, decoder and Protenix weights stay frozen.
[Upstream model and code](https://github.com/daiki-ko/graph_lda/tree/591abff0efddb5dc75dfdb63db37a9c2ac39be69) ·
[Plain ZINC checkpoint](https://drive.google.com/file/d/16wbGSoUOb5uqmjt_qfGi_SNgyWDQhyrq/view).

## Design

The four starts come from the PubChem validation panel already used in the
[compact decoder comparison](../../vae/bond-message-comparison.md). Each
contains 25 heavy atoms, one connected component and no individually charged
atoms. We chose the panel before evaluating the new checkpoint. The same
25-site BRD4 pocket context serves all relaxed evaluations. The source graph
and the hard graph decoded at update zero can differ. Report both.

* Run 32 updates of contact-only gradient descent with learning rate 60 and
  decoder temperature 1.0. A second trajectory starts from the same latent
  and takes seeded random directions with the corresponding gradient trajectory's
  step lengths. Each arm has the same relaxed scoring budget.
* Backpropagate the Mosaic Protenix contact derivative through element
  probabilities and bond presence into the latent. The bridge sees
  single, double, triple and aromatic bonds as present. An exchange between
  bond types at equal presence has no direct contact derivative. The graph
  decoder retains the full bond distribution. Occupancy and formal charge are
  fixed. No chemistry term or denoiser is added.
* Save the hard graph at each update, including invalid results. Fresh scoring
  uses native Protenix-tiny FILE featurization, regenerated ligand conformers
  and the same BRD4 pocket. Each arm can nominate up to three distinct valid
  edits. A smaller fresh contact loss indicates a better model prediction.
  Compare each edit with the original input as well as the decoded start
  wherever a baseline is scoreable. Count unsuccessful starts in the overall
  result. Predictions here do not measure experimental binding.

The four-start, one-pocket result is exploratory. A positive relaxed trajectory
on its own cannot establish improvement after discretization.

[Run config](graph-lda-contact.json) · [Relaxed scorer config](native-dtu.json) ·
[Fresh scoring config](flow-fresh-dtu.json)

## Submitted pilot and initial measurements, 20 November 2026

LSF accepted **30129063** at 10:20 CET on `gpuv100`. It was pending at
submission. All eight trajectories and their fresh scoring share one allocation
with one 32 GB V100, four CPU slots and a two-hour wall-time limit. The detached
CPU decoder uses the posterior mean and fixed hard slot ranks. The actual
update is plain SGD, `z_next = z - 60 * gradient`, without normalization or
clipping. There are 264 relaxed evaluations including initial and final states.
The native binder sees no formal-charge changes. Results go to
`/work3/s234818/latentmol/runs/graph-lda-contact-20261120a`, with
`comparison.json` and `interactive-report/index.html` generated at completion.
The fresh evaluation uses its own pocket-contact reduction and conformer, so
numerical differences are interpreted only within each scoring protocol.

The real decoder loaded locally. All four inputs passed the 25-atom eligibility
checks. Their source graphs have 27–28 bonds. Their unmodified hard decodes
have 50–53 bonds, and all four failed RDKit sanitization. Test observables for
element probability and total bond presence had finite derivatives at all four
latents. These CPU measurements exercise the decoder only. The GPU experiment
keeps all four failures and can still discover a valid molecule after moving.

[CPU panel results](../../../outputs/graph-lda-assets-591abff0/encoded-panel-cpu.json) ·
[Submission receipt](../../../outputs/graph-lda-assets-591abff0/submission.json) ·
[Allocated launcher](run_pretrained_contact.sh) · [Pilot runner](pretrained_contact.py)
