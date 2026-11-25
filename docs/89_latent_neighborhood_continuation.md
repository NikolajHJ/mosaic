# Smaller-step VAE continuation with local latent sampling

**Recovery31 October2026:** see the [current experiment note](../experiments/vae/README.md)
for the new home-control/node-local launcher, epoch43 metrics and resubmitted
attempt002. Probe29860453 is pending behind launcher check29860450; all queues
are administratively inactive. No GPU pass or amended44 training yet. The dated
source001 preparation/submission below remains historical evidence.

Prepared 30 October 2026. **At 13:13 CET, the disposable GPU probe is submitted
but pending; the new training objective has not started.** The user
requested introducing the smaller optimizer steps and local latent perturbation
training at the next epoch boundary. Epoch 43 was running when preparation
started, so the intended first new epoch is 44, conditional on its predecessor
finishing and the new launch checks passing.

The [actual scheduler and handoff snapshot](../outputs/hpc-results/source155-vae-local-neighborhood-attempt001/snapshot-1793362424215721000/status.json)
records:

| Item | Saved status at 13:13 CET |
| --- | --- |
| Disposable full-batch probe | GPU job **29853841 PEND**, waiting for job-slot limits on five hosts. |
| Probe-result callback | **29853842 PEND** on its normal dependency. |
| Existing epoch 43 training | **29849760 RUN**, unchanged. |
| Old next-epoch callback | **29849761 PSUSP**, intentionally held for the exclusive next-epoch handoff. |

Neither a GPU probe pass nor new epoch-44 training is established. The first
callback-hold reply arrived before the scheduler showed the held state. The
saved intent allowed an idempotent `arm` retry to confirm `PSUSP` and finish
the armed receipt without issuing a second `bstop`. Running source, weights,
losses and checkpoint files were not changed by that callback handoff.

## Question and bounded comparison

Can a smaller parameter step and explicit chemistry training near real-molecule
embeddings improve generation and latent-space robustness while retaining
reconstruction, connectivity and chemical diversity? Reconstruction is already
excellent on the small monitored panel, whereas standard-Gaussian prior samples
are less consistently valid. See the [existing continuation](80_epoch40_to45_continuation.md)
for that campaign's actual results and [the local neighbor-loss pilot](88_local_neighbor_loss_pilot.md)
for the failed alternative that must not be adopted here.

Both proposed changes enter together. Comparing successive epochs therefore
does **not** isolate whether any improvement comes from smaller steps, local
sampling or their interaction. A later matched ablation would be needed for
that attribution. These changes do not establish that a VAE guarantees smooth
chemical semantics or validity between arbitrary embeddings.

## Optimizer change

The established continuation retains inherited AdamW first and second moments,
performs one native AdamW step and then applies 0.25 of its parameter displacement.
The proposed continuation applies **0.125 of that native displacement**, half
the currently applied step. AdamW moments still advance once per logical update.
The stored learning rate is not divided a second time, and the moments are not
reset. This is a declared optimizer-trajectory change, not an exact replay of
the old continuation. Gradient clipping and nonfinite checks remain separate
from displacement scaling.

## Local latent mixture

The real batch remains 256 molecules. Its reconstruction and annealed Gaussian
KL terms remain unchanged. The synthetic branch remains 512 points, preserving
the **1:2 real-to-synthetic count and the number of decoder/repair examples**.
At its full setting, the synthetic branch contains:

| Synthetic source | Points per full logical batch | Training objective |
| --- | ---: | --- |
| Independent standard Gaussian prior |384 | Established repair/reference/composition/occupancy objective. |
| Small perturbations around real-molecule embeddings |128 | The same established chemistry objective. |

The local fraction uses ramp multiplier
\(r_k=\min(k/100,1)\), with the first new update numbered \(k=1\).
Its first multiplier is therefore 1/100, not zero. The local count is
\(\lfloor N_{\mathrm{synthetic}}\,0.25\,r_k\rfloor\); final short batches
use the same floor rule. The actual number of each kind is recorded. This
replaces part of the previous pure
prior branch rather than adding 512 more points. The new branch is therefore a
mixture distribution, and its training statistics must not be described as
pure-standard-Gaussian statistics.

For a selected real molecule with encoder outputs \(\mu(x),\log\sigma^2(x)\),
the local coordinate is

\[
 u=\frac{\epsilon}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}\epsilon_j^2}},
 \quad \epsilon\sim\mathcal N(0,I),
 \qquad
 z_{\mathrm{local}}
 =\operatorname{stopgrad}\!\left[
     \mu(x)+0.05\,\sigma(x)\odot u\right].
\]

The direction is drawn from Gaussian noise and then normalized to unit RMS.
Thus the posterior-standardized displacement has RMS exactly 0.05, within
floating-point precision; its direction is not an unnormalized Gaussian draw.
This is not a 0.05 Euclidean radius in 256 dimensions. Record the resulting displacement
norms. The encoder is detached only for choosing these synthetic coordinates.
It continues to learn through the ordinary real-molecule reconstruction/KL path.
This prevents it from moving the synthetic coordinates merely to evade their
chemistry loss. The proposal deliberately stays local; it does not require a
large neighborhood to reconstruct one fixed molecule or assign one target to
an arbitrary interpolation between unrelated molecules.

An independent local-sampling RNG controls source-row selection and perturbation
noise. Its state must be included in new continuation checkpoints. Coordinates
are drawn once per logical update and reused unchanged by the no-gradient
teacher preparation and the live differentiable decoder pass. Microbatching
must not redraw them or change the complete-branch normalization.

## Loss and preserved meaning

Local points use the existing probability-aware full-graph repair guidance,
with frozen detached targets and ordinary decoder gradients. This is a change
in **where** chemistry training is applied, not a new neighbor-valence formula.
The repair search and chemical validity checks remain outside autograd. The
frozen same-z valid-target reference and occupancy reference keep their existing
roles; their checkpoint identities are not silently advanced to the latest VAE.
Composition regularization remains a population term.

The failed `common-organic-neighbor-v1` weight 0.1 addition is **not enabled**.
Its local pilot reduced valence errors while increasing disconnected molecules;
this continuation must continue reporting connectivity separately. Validity of
a hardened graph is not a guarantee that all categorical samples from its
probabilities are valid, and a finite attached gradient is not evidence of a
chemical improvement.

## Epoch boundary and outputs

Use a completed predecessor checkpoint, retain model/optimizer/RNG state and
record the change as a new continuation configuration. Do not change a live
allocation's objective in place or overwrite its source, receipts or state.
The controller has queued a separate, disposable **100-update full-batch GPU
probe from authenticated epoch 42** before adoption. The probe is not the live
continuation and its experimental updates will not replace epoch 43. Its
explicit `arm` operation holds only the exact
identified pending old callback; epoch 43's GPU allocation continues unchanged.
A failed probe restores the old callback. A passed probe queues epoch 44 with
a `done` dependency on the original epoch-43 GPU job and retires the held old
callback. The new controller then advances serially to epoch 45. Submission
and arming are recorded above; those conditional successor actions have not
yet occurred in the saved snapshot.

The handoff must prevent two successor writers. It records the old and new
scheduler state and exact predecessor checkpoint, and fails on an ambiguous or
competing writer. Starting epoch 44 additionally requires the predecessor's
completed-epoch checks. An objective-change handoff is explicitly different
from an exact resume under the old scientific configuration.

The intended endpoint remains 45 unless the user changes it. Save each completed
epoch and retain raw invalid outputs in evaluation artifacts. Follow the
[current exploratory policy](EXPLORATORY_RESEARCH_POLICY.md): finite computations,
attached required paths, compatible representations, frozen-reference integrity
and exclusive output ownership remain required. Report quality changes and any
explicit retained campaign stopping rules honestly; do not silently weaken an
old receipt reader or reinterpret a historical failure.

The inherited health baselines remain original epoch 35 and original epoch 36.
They are not silently ratcheted to the new predecessor checkpoint at epoch 43.
The new predecessor is still evaluated for comparison.

## Code, checks and pending evidence

| Responsibility | Code |
| --- | --- |
| Versioned mixture, normalized perturbations, detached sampling and separate teacher coverage | [`LocalLatentPrior`](../src/latentmol/latent_neighborhood.py) |
| Existing chemistry update, with an absent-by-default `prior_sampler` hook | [`chemistry_continuation.py`](../src/latentmol/chemistry_continuation.py) |
| Legacy 0.25 and explicit versioned 0.125 displacement policies | [`chemistry_optimizer_step.py`](../src/latentmol/chemistry_optimizer_step.py) |
| New objective-change setup, inherited baselines, checkpoint/RNG handling and allocated training | [`vae_neighborhood_continuation.py`](../src/latentmol/vae_neighborhood_continuation.py) |
| Probe, explicit callback arming, exclusive epoch handoff and serial endpoint | [`vae_neighborhood_campaign.py`](../scripts/hpc/vae_neighborhood_campaign.py) |
| Mixture, tail, RNG, live-path and optimizer regressions | [`test_latent_neighborhood.py`](../tests/test_latent_neighborhood.py) |
| Metadata-only scheduler and actual launch-command regressions | [`test_vae_neighborhood_campaign.py`](../tests/test_vae_neighborhood_campaign.py) |

The [working-code local regression receipt](../outputs/implementation-checks/latent-neighborhood-005.xml)
records **177 passing cases** for the changed behavior and relevant existing
contracts. The same **177 cases passed against the packaged source snapshot**
in [the sealed-source receipt](../outputs/implementation-checks/latent-neighborhood-sealed-001.xml),
with zero failures, errors or skips. One additional
[driver-health regression passed](../outputs/implementation-checks/latent-neighborhood-health-002.xml)
after an incomplete test fixture was corrected; production code was unchanged
by that fixture correction. This is local regression evidence, not allocated
GPU training, native parity or a chemical-improvement result. No separate broad
qualification campaign is required for this exploratory change.

The submitted source snapshot and study are explicitly identified:

```text
source155-vae-local-neighborhood-attempt001
archive SHA256 e012ec856c4f29b132f7ad57346ee7b9b5156c20563e97b595cbbc63f104a4ef
study   SHA256 8f7e23b5e03cfdd084300372cb63d48512864eb2488f44d8ba68520fea288e77
```

The snapshot above retains the exact submitted command, source paths,
checkpoint identity, submission receipts and callback hold/armed evidence.
The GPU probe requests a four-hour allocation, with a 12,600-second internal
budget; each new full-epoch allocation is limited to 24 hours, with an
82,800-second internal budget. Submission is not execution or completion.

Compare the same fixed standard-Gaussian panel and reconstruction cohort at the
predecessor and new checkpoints. Keep separate argmax and categorical-sampling
validity, chemical versus representation failures, disconnected/empty graph
counts, molecule-size distribution, hard and expected sulfur frequency, diversity
and runtime. Add local-neighborhood observations where practical, with their
coordinate distribution and perturbation scale stated explicitly. Pure-prior
evaluation remains necessary even though training uses a mixture.

The allocated GPU probe will test the actual full-batch update, finite attached
gradients, runtime and inherited fixed-panel checks before adopting the new
epoch. It is queued but has not started in the saved snapshot.
The absent-sampler path, one-time coordinate use, independent sampler RNG and
0.125 displacement behavior are covered by the local regression tests above.

The actual completed-checkpoint handoff and GPU/training findings are **pending**.
Append the actual experiment outcome to the [ledger](05_experiment_ledger.md)
when evidence exists. The source and probe are submitted; this note makes no
claim that the new objective has passed its GPU probe or entered training.
