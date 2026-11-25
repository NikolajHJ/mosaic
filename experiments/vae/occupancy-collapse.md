# Why Gaussian draws fill all 50 atom slots in epoch 47

19 November 2026. The 3.94M-parameter VAE reconstructs more held-out PubChem
and ChEMBL molecules after 47,682,960 examples in epoch 47. On the same 256
Gaussian vectors, its valid argmax decodes fell from 190 at epoch 46 to six.
Full 50-atom graphs rose from zero to 249. Only 0.40% of the 60M training
molecules have 50 atoms. All measurements below use local CPU copies of the
checkpoints. The [subsequent epoch-47 continuation](epoch47-occupancy-continuation.md)
adopts the candidate loss at the next saved boundary.
[Checkpoint and data comparison](large-corpus.md#epoch-47-after-47682960-real-molecules-19-november)

## Where the count changes

On those Gaussian vectors, epoch 47's mean hard atom count after the initial
atom readout and the four successive refinement rounds was **31.25, 25.76,
27.68, 49.83, 49.86**. After round three, 237/256 already filled every slot.
The mean soft count at the end was also 49.86. Giving epoch 47 the older final
atom readout still left 249/256 full. On actual PubChem posterior samples the
same decoder produced a mean of 22.19 atoms and zero full graphs on 64 inputs.
Independently permuting the 64 values within each latent dimension preserved
every empirical marginal while destroying the relationships between dimensions.
Then 59/64 draws hit the cap. These interventions implicate the joint latent
structure and the refinement network. A scalar norm correction or a different
atom numbering cannot explain these measurements.
[Geometry probe](../../outputs/runs/vae-occupancy-geometry-20261119a/summary.json)

Swapping only round three from epoch 46 into epoch 47 removed all 93 cap hits
on a matched subset of 96 prior draws. Valid outputs rose only from 1 to 4,
and exact PubChem reconstructions fell from 58 to 51 on 64 real molecules.
In the reciprocal swap, epoch 47's third round on the epoch-46 backbone gave
zero cap hits and 9/96 valid priors. The third round acts in conjunction with
the other learned layers. Correcting cardinality alone leaves atom and bond
compatibility unresolved. [Reciprocal swaps](../../outputs/runs/vae-occupancy-geometry-20261119a/round3-swap.json)

## Why the training objective permits it

The real branch scores Gaussian posterior reconstructions against real graphs.
It supervises padding with weight 0.2; the size-loss weight is zero. The separate
random branch draws 16 standard-normal latents per 240 real molecules. The
teacher searches for bond edits while holding the student's atoms, charges and
number of atoms fixed. Each successful repair becomes a cross-entropy target
for **all** of its atoms, charges and bonds. Every already-valid or unrepaired
candidate contributes zero to this term. For a repaired occupied slot the
derivative with respect to its empty-slot logit is positive, so gradient
descent makes absence less probable. An empty slot also reinforces its current
state. The denominator is detached, and the 15:1 sampling ratio does not
divide the 0.375 loss weight. A falling repair loss can accompany worsening
size and validity. Epoch 46 also used a frozen occupancy reference, verified
valid prior targets, composition guidance and more prior draws.
[Epoch-47 configuration](epoch46-pubchem60m-prior15-v100-benchmark.json) ·
[Training update](../../src/latentmol/training/prior_repair.py) ·
[All-head repair objective](../../src/latentmol/atom_preserving_chemistry.py)

On 16 Gaussian training draws and 16 independent probe draws, removing the
direct atom/charge terms while retaining the native bond normalization reduced
the predicted increase in mean soft atom count from a repair-gradient step by
about **86% at epoch 46** and **4% at late epoch 47**. These are local
directional derivatives of vanilla gradient descent, not the observed steps
of historical AdamW. By the late checkpoint, bond gradients alone still
increased occupancy through shared decoder weights. An epoch-46 reference
with binary cross-entropy on the occupancy logit
`logsumexp(real_atom_logits) - no_atom_logit` gave the opposite direction.
Its derivative with respect to that logit is predicted occupancy minus
reference occupancy, retaining corrective signal on confidently wrong slots.
An external count histogram helps specify the desired population, but
gradients through nearly saturated probabilities can be weak.
[Gradient controls](../../outputs/runs/vae-occupancy-gradients-20261119a/diagnostic-47.json)

The healthy parent also has strong real-reconstruction gradients through the
shared decoder. On four independent minibatches of 16 true training examples,
the real gradient moved probe occupancy up once and down three times. A single
cause of the first historical upward move cannot be identified from these
endpoint probes. [Real-data control](../../outputs/runs/vae-occupancy-gradients-20261119a/real-control-46.json)

## Separating size correction from chemical recovery

We evaluated 128 new shared Gaussian vectors. All hard counts, chemical
validity and distinctness below are raw, before any repair. We used 512
training examples to estimate posterior moments. A separate 97-parameter
final occupancy readout was fitted with binary cross-entropy on 192 of
those real encodings (true occupancy labels) and 192 independent prior
vectors (soft epoch-46 occupancy labels). The readout keeps the student's
conditional element probabilities, charge and bond logits. All other
model parameters were frozen.

| Frozen model or intervention | At 50 / 128 | Valid / 128 | Distinct valid molecules |
| --- | ---: | ---: | ---: |
| Epoch 46 | 0 | 93 | 93 |
| Late epoch 47 | 125 | 4 | 4 |
| Use epoch-46 occupancy only at the final readout | 0 | 35 | 17 |
| Fit a new final occupancy readout | 0 | 28 | 10 |
| Feed epoch-46 occupancy into all refinement rounds | 0 | 8 | 6 |
| Fit a full Gaussian to 512 epoch-47 training posteriors, 10% diagonal shrinkage | 114 | 2 | 2 |

Of the 35 valid outputs with the final epoch-46 occupancy substituted, 30
contained only carbon as a heavy element. With the fitted readout the tally
was 25/28. On 96 separate PubChem validation molecules the fitted readout
kept 87 exact versus 89 before intervention. Simple threshold or size controls
can thus appear successful on a count dashboard while losing much of the
generative chemistry. Gaussian fits using just the mean or diagonal covariance
also left at least 118/128 at capacity. This does not test a nonlinear learned
prior. Frozen encoder+decoder parameter blends at 25%, 50% and 75% toward
epoch 47 removed the cap on the separate 96-point panel but none retained
both the epoch-46 generation and epoch-47 reconstruction results.
[All occupancy and prior arms](../../outputs/runs/vae-occupancy-recovery-20261119a/evaluated/summary.json) ·
[Generated identities](../../outputs/runs/vae-occupancy-recovery-20261119a/evaluated/rows.json) ·
[Model blends](../../outputs/runs/vae-occupancy-geometry-20261119a/weight-blends.json)

To reproduce the frozen-model interventions, use the saved
[training membership](../../outputs/runs/vae-occupancy-recovery-20261119a/train-panel.json)
and [validation membership](../../outputs/runs/vae-review-20261119-47682960/pubchem-val-panel.json).

```sh
PYTHONPATH=src python experiments/vae/occupancy_recovery.py --parent EPOCH46.pt --student PARTIAL47.pt --train train-panel.json --validation pubchem-val-panel.json --output NEW_DIRECTORY
```

On these 512 training encodings the mean per-dimension aggregate posterior
standard deviation was 1.0001. Matching that statistic, or even an estimated
full covariance, gave little benefit. A learned nonlinear prior remains a
separate option if the project can change its sampling distribution. The
two-stage VAE literature motivates that experiment; its reported results do
not establish that it will work on these molecular graphs.
[Dai and Wipf, Diagnosing and Enhancing VAE Models](https://arxiv.org/abs/1903.05789)

## A usable way to initialize epoch 47 today

From 256 Gaussian draws, epoch 46 generated 190 distinct valid molecules in
the shared vocabulary. We encoded each through epoch 47 and decoded its
posterior mean. This yielded 144 distinct valid molecules (144/256 original
attempts), 99 exact founder identities, 45 valid edits and zero cap hits.
One stochastic posterior draw per founder yielded 126/256 valid, with 57
exact founders. Both use epoch 46 to supply the starting chemistry. Keep the
original founder whenever its identity matters, and independently score any
changed molecule. These are generation and reconstruction checks with no
binding-score or optimization claim.
[Results on the original 256 attempts](../../outputs/runs/vae-occupancy-geometry-20261119a/founder-handoff.json)

## Direct training test from the collapsed checkpoint

Two decoder copies started from the same late-47 weights. We froze the encoder,
kept true reconstruction on four real training molecules per update and used
four fresh Gaussian vectors per update. Each arm ran 24 matched updates with
a fresh AdamW optimizer (learning rate 0.0002, norm clip 1). Both used binary
cross-entropy against epoch 46's soft occupancy at the same latent. The second
arm also used conditional element, charge and bond cross-entropy on the 78/96
training latents where the parent gave a valid graph. Weights were 1 each for
real reconstruction, occupancy and the optional chemistry objective. The
historical 15:1 production recipe was not replayed by this experiment.

| Model on the same held-out panel | At 50 / 128 | Median atoms | Valid and distinct / 128 | Exact PubChem / 64 |
| --- | ---: | ---: | ---: | ---: |
| Epoch 46 | 0 | 22 | 91 | 49 |
| Late epoch 47 | 125 | 50 | 7 | 61 |
| After occupancy + real replay | 0 | 22 | 5 | 60 |
| After occupancy + real replay + valid-parent chemistry | 0 | 21 | 9 | 58 |

The occupancy loss reversed saturation, even from the collapsed model. The
additional chemistry term recovered only two net valid outputs over the
baseline on this sample and lost three exact reconstructions. Neither arm
recovered an unconditional generator comparable with epoch 46. These pilots
used one seed, a bounded budget and unmodified held-out labels; no new model
checkpoint was deployed. [Results](../../outputs/runs/vae-occupancy-distill-20261119a/summary.json) ·
[Matched training script](../../outputs/runs/vae-occupancy-distill-20261119a/compare.py)

For the next controlled training experiment, start an independent fork from
epoch 46 before the collapse. Replay identical 60M data, seeds, update budget
and 15:1 exposure in the control and candidate. On repaired prior graphs,
supervise the corrected bond distribution; supply separate occupancy targets
and valid chemistry from the frozen parent for the same random vectors. This
is a training reference with varying counts, never a fixed count at generation.
Protect diversity and the useful PubChem/ChEMBL learning by comparing raw
prior validity, distinct valid structures, occupancy histograms, and exact
held-out reconstruction at shared checkpoints. Extend through the region
where the control deteriorates. The prior ratio and reference weights can
be studied separately once that comparison establishes whether the proposed
recipe prevents drift.
